"""DeepMIMO v4 ray-traced scenarios as SampleTables (optional package ``deepmimo``, extra ``[sim]``).

DeepMIMO stores, per transmitter (base station / AP) and receiver (user grid point), the
ray-traced paths: ``power`` (dBW; ``10 log10 |a|^2`` of the complex path amplitude ``a`` for a
transmit power of 0 dBW, so it is the path gain in dB), ``phase`` (degrees, the angle of ``a``),
``delay`` (s), angles of arrival and departure ``aoa_az``, ``aoa_el``, ``aod_az``, ``aod_el``
(degrees; azimuth from +x, "elevation" is the zenith angle, 0 = straight up: DeepMIMO's own
array response ``deepmimo.generator.geometry._array_response_phase`` uses ``cos(theta)`` for
the z component), ``rx_pos`` / ``tx_pos`` (m) and interaction codes ``inter``; invalid (padded)
paths are NaN. This module converts those units once, at the boundary, into the IndoorLoc
layout (CONTRACTS section 2): dBm, metres, radians. Because of the 0 dBW (= 30 dBm)
normalisation, ``received_power_dbm`` is the RSSI of a 30 dBm transmitter; for a transmit
power ``P`` dBm pass ``power_offset_db = P - 30``.

The conversions are plain numpy functions of the path arrays (``received_power_dbm``,
``first_arrival_range``, ``strongest_path_azimuth``, ``paths_to_csi``, ``scenario_to_table``),
so they are tested on synthetic arrays; only ``DeepMIMO.load`` needs the package. Each
transmitter becomes one anchor; rows are receivers.

Status: written against the DeepMIMO 4.0.5 sources (``deepmimo.load``, ``deepmimo.download``,
``deepmimo.config`` "scenarios_folder", the ``Dataset`` matrices, ``MacroDataset.datasets``).
Checked once outside the test suite with deepmimo 4.0.5 installed (Python 3.11, numpy 2.2):
``DeepMIMO.load`` read a small scenario written in DeepMIMO's on-disk format, and
``paths_to_csi`` matched ``Dataset.compute_channels`` (1x1 antennas, isotropic, no filter,
after DeepMIMO's ``1/sqrt(N_subcarriers)`` scaling) to a relative error of 3e-7. A 1x1 array
does not exercise the angle convention; the zenith reading of ``aod_el`` and the 0 dBW power
normalisation were checked separately in the 4.0.5 sources (``generator/geometry.py``,
``utils/info.py``). It has not been run on a published ray-traced scenario. The repository tests use a stand-in module,
because deepmimo 4.0.5 pins numpy < 2.3.

References
----------
A. Alkhateeb, "DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO
    applications", Information Theory and Applications Workshop (ITA), 2019.
    arXiv:1902.06435. https://deepmimo.net
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np

from ...core import SampleTable, requires
from .._base import Dataset
from .channel import HT20_SUBCARRIERS, OFDM_SPACING_HZ, ofdm_csi, subcarrier_offsets
from .geometry import wrap_angle
from .propagation import SPEED_OF_LIGHT, apply_sensitivity

MODALITIES = ("rssi", "ranges", "tdoa", "aoa", "csi")
_MATRICES = ("power", "phase", "delay", "aod_az", "aod_el", "rx_pos", "tx_pos")


def received_power_dbm(power_dbw, phase_deg=None, *, coherent: bool = False) -> np.ndarray:
    """(N,) total received power in dBm from per-path powers (N, P) in dBW; NaN without paths.

    Incoherent (default): ``10 log10(sum_p 10 ** (P_p / 10)) + 30``, the wideband average
    power an RSSI reports. Coherent: ``|sum_p sqrt(P_p) exp(j phi_p)|^2`` (a narrowband tone).
    DeepMIMO path powers are normalised to a 0 dBW transmitter, so this is the RSSI of a
    30 dBm transmitter; add ``P_tx_dBm - 30`` for another transmit power.
    """
    power = np.atleast_2d(np.asarray(power_dbw, dtype=np.float64))
    valid = ~np.isnan(power)
    lin = np.where(valid, 10.0 ** (power / 10.0), 0.0)
    if coherent:
        if phase_deg is None:
            raise ValueError("coherent summation needs phase_deg")
        phase = np.deg2rad(np.where(valid, np.asarray(phase_deg, dtype=np.float64), 0.0))
        total = np.abs(np.sum(np.sqrt(lin) * np.exp(1j * phase), axis=1)) ** 2
    else:
        total = lin.sum(axis=1)
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(total) + 30.0
    out[~valid.any(axis=1) | (total <= 0)] = np.nan
    return out


def first_arrival_range(delay_s) -> np.ndarray:
    """(N,) ``c * min_p delay_p`` in metres: the ToA range of the earliest path; NaN without paths."""
    delay = np.atleast_2d(np.asarray(delay_s, dtype=np.float64))
    out = np.full(len(delay), np.nan)
    has = ~np.isnan(delay).all(axis=1)
    out[has] = SPEED_OF_LIGHT * np.nanmin(delay[has], axis=1)
    return out


def strongest_path_azimuth(power_dbw, azimuth_deg) -> np.ndarray:
    """(N,) azimuth (radians, ``[-pi, pi)``) of the strongest path; NaN without paths."""
    power = np.atleast_2d(np.asarray(power_dbw, dtype=np.float64))
    az = np.atleast_2d(np.asarray(azimuth_deg, dtype=np.float64))
    out = np.full(len(power), np.nan)
    has = ~np.isnan(power).all(axis=1)
    best = np.nanargmax(np.where(np.isnan(power[has]), -np.inf, power[has]), axis=1)
    out[has] = wrap_angle(np.deg2rad(az[has][np.arange(has.sum()), best]))
    return out


def path_gains(power_dbw, phase_deg) -> np.ndarray:
    """(N, P) complex amplitudes ``sqrt(10 ** (P / 10)) exp(j phi)`` (NaN for absent paths)."""
    power = np.asarray(power_dbw, dtype=np.float64)
    phase = np.deg2rad(np.asarray(phase_deg, dtype=np.float64))
    return np.sqrt(10.0 ** (power / 10.0)) * np.exp(1j * phase)


def paths_to_csi(power_dbw, phase_deg, delay_s, azimuth_deg, zenith_deg, offsets_hz, *, n_antennas: int,
                 orientation: float = 0.0, spacing_wavelengths: float = 0.5) -> np.ndarray:
    """(N, M, K) CSI at an M-element ULA whose boresight points along ``orientation`` (rad).

    Uses ``channel.ofdm_csi`` with gains ``sqrt(P) exp(j phi)``, the stored delays and the
    departure angles at the transmitter (by reciprocity, the uplink angles of arrival at the
    AP); the zenith angle ``theta`` becomes elevation ``pi/2 - theta``. Like DeepMIMO's own
    OFDM generator, the stored phase is the path phase at the carrier and subcarrier ``k`` adds
    ``exp(-j 2 pi df_k tau)``; unlike it, no ``1/sqrt(N_subcarriers)`` normalisation is applied.
    """
    theta = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64)) - orientation
    elevation = np.pi / 2.0 - np.deg2rad(np.asarray(zenith_deg, dtype=np.float64))
    return ofdm_csi(path_gains(power_dbw, phase_deg), delay_s, theta, offsets_hz, n_antennas=n_antennas,
                    spacing_wavelengths=spacing_wavelengths, elevation=elevation)


def scenario_to_table(links, *, modality: str = "rssi", dim: int = 3, power_offset_db: float = 0.0,
                      sensitivity_dbm: float | None = None, coherent: bool = False, n_antennas: int = 8,
                      spacing_wavelengths: float = 0.5, orientation: float = 0.0, subcarriers=HT20_SUBCARRIERS,
                      subcarrier_spacing_hz: float = OFDM_SPACING_HZ, drop_empty: bool = True) -> SampleTable:
    """Build a SampleTable from per-transmitter path arrays.

    ``links`` is a sequence with one mapping per transmitter holding the DeepMIMO matrices of
    one receiver set: ``power``, ``phase``, ``delay``, ``aod_az``, ``aod_el`` (N, P), ``rx_pos``
    (N, 3), ``tx_pos`` (3,) or (1, 3), and optionally ``tx_id`` (str) and ``rx_set`` (int).
    Mappings that share an ``rx_set`` must describe the same receivers; several receiver sets
    are stacked row-wise. Modalities: ``rssi`` (N, A) float32 dBm for a 30 dBm transmitter
    (+ ``power_offset_db``, e.g. -10 for a 20 dBm AP; NaN below ``sensitivity_dbm``),
    ``ranges`` (N, A) first-arrival ranges, ``tdoa`` (N, A - 1) against transmitter 0, ``aoa``
    (N, A) strongest-path departure azimuth relative to ``orientation``, ``csi``
    (N, A * M, 1, K) complex64 (one ULA of ``n_antennas`` per transmitter). ``drop_empty``
    removes receivers with no path to any transmitter.
    """
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, not {modality!r}")
    keyed = [(str(link.get("tx_id", i)), int(link.get("rx_set", 0)), link) for i, link in enumerate(links)]
    if not keyed:
        raise ValueError("no transmitter/receiver links given")
    tx_ids = list(dict.fromkeys(k[0] for k in keyed))
    order = {t: a for a, t in enumerate(tx_ids)}
    rx_sets = sorted({k[1] for k in keyed})
    tx_pos = np.zeros((len(tx_ids), 3))
    offsets = subcarrier_offsets(subcarriers, subcarrier_spacing_hz)
    blocks, positions, set_col, rx_idx = [], [], [], []
    for rs in rx_sets:
        members = sorted((k for k in keyed if k[1] == rs), key=lambda k: order[k[0]])
        if [k[0] for k in members] != tx_ids:
            raise ValueError(f"receiver set {rs} must be linked to every transmitter exactly once "
                             "(give each link a tx_id when there are several receiver sets)")
        members = [k[2] for k in members]
        rx = np.asarray(members[0]["rx_pos"], dtype=np.float64).reshape(-1, 3)
        cols = []
        for a, link in enumerate(members):
            if not np.allclose(np.asarray(link["rx_pos"], dtype=np.float64).reshape(-1, 3), rx):
                raise ValueError(f"receiver set {rs}: transmitters see different receiver positions")
            tx_pos[a] = np.asarray(link["tx_pos"], dtype=np.float64).reshape(-1)[:3]
            if modality == "rssi":
                cols.append(received_power_dbm(link["power"], link.get("phase"), coherent=coherent)[:, None])
            elif modality in ("ranges", "tdoa"):
                cols.append(first_arrival_range(link["delay"])[:, None])
            elif modality == "aoa":
                cols.append((wrap_angle(strongest_path_azimuth(link["power"], link["aod_az"]) - orientation))[:, None])
            else:
                cols.append(paths_to_csi(link["power"], link["phase"], link["delay"], link["aod_az"], link["aod_el"],
                                         offsets, n_antennas=n_antennas, orientation=orientation,
                                         spacing_wavelengths=spacing_wavelengths).astype(np.complex64))
        blocks.append(np.concatenate(cols, axis=1))
        positions.append(rx)
        set_col.append(np.full(len(rx), rs, dtype=np.int64))
        rx_idx.append(np.arange(len(rx)))
    X, pos = np.concatenate(blocks), np.concatenate(positions)
    rx_set, rx_index = np.concatenate(set_col), np.concatenate(rx_idx)
    if modality == "csi":
        empty = ~np.any(np.abs(X) > 0, axis=(1, 2))
        X = X[:, :, None, :].astype(np.complex64)
    else:
        empty = np.isnan(X).all(axis=1)
    if modality == "rssi":
        X = X + power_offset_db
        if sensitivity_dbm is not None:
            X = apply_sensitivity(X, sensitivity_dbm)
        X = X.astype(np.float32)
    elif modality == "tdoa":
        X = X[:, 1:] - X[:, :1]
    if drop_empty and empty.any():
        keep = ~empty
        X, pos, rx_set, rx_index = X[keep], pos[keep], rx_set[keep], rx_index[keep]
    ids = np.array([f"rx{s}-{i:06d}" for s, i in zip(rx_set, rx_index)])
    meta = {"modality": modality, "units": {"rssi": "dBm", "ranges": "m", "tdoa": "m", "aoa": "rad",
                                            "csi": "complex channel gain (linear, unitless)"}[modality],
            "crs": "local", "pos_names": ("x", "y", "z")[:dim], "pos_units": "m",
            "anchors": tx_pos[:, :dim], "anchor_orientations": np.full(len(tx_ids), float(orientation)),
            "anchor_ids": tuple(tx_ids)}
    if modality in ("rssi", "ranges", "aoa"):
        meta["feature_names"] = tuple(tx_ids)
    elif modality == "tdoa":
        meta.update(feature_names=tuple(f"{t}-{tx_ids[0]}" for t in tx_ids[1:]), reference_anchor=0)
    else:
        meta.update(subcarrier_offsets_hz=offsets, n_antennas=n_antennas,
                    antenna_anchor=np.repeat(np.arange(len(tx_ids)), n_antennas), csi_axes=("rx", "tx", "subcarrier"))
    groups = {"source": np.full(len(X), "simulated"), "rx_set": rx_set}
    return SampleTable(X, pos[:, :dim], None, None, groups, ids, meta)


@contextlib.contextmanager
def _scenarios_folder(dm, folder: Path):
    """Point DeepMIMO's scenario folder at ``folder`` for the duration of a call."""
    old = dm.config.get("scenarios_folder")
    dm.config.set("scenarios_folder", str(folder))
    try:
        yield
    finally:
        dm.config.set("scenarios_folder", old)


class DeepMIMO(Dataset):
    """DeepMIMO v4 ray-traced scenarios (Alkhateeb, ITA 2019) as RSSI, ranging, AoA or CSI tables.

    The scenario folder (``params.json`` plus the ``.npz`` matrices DeepMIMO writes) is
    expected at ``root / scenario`` (``root`` defaults to ``default_root() / "deepmimo"``);
    with ``download=True`` a missing scenario is fetched with ``deepmimo.download`` into that
    folder (it never prompts). Loading uses ``deepmimo.load(scenario, tx_sets=..., rx_sets=...,
    max_paths=...)``; every transmitter becomes an anchor and every receiver a row. There is
    no official split: the only split is ``"all"`` (build splits in L4 from positions or
    ``groups["rx_set"]``). ``meta["carrier_hz"]`` and ``meta["rt_params"]`` record the ray
    tracer's frequency, name and version.

    Parameters
    ----------
    scenario : DeepMIMO scenario name (e.g. ``"asu_campus_3p5"``).
    modality : ``"rssi"`` (the ``wifi_rssi`` layout: (N, A) dBm), ``"ranges"``, ``"tdoa"``,
        ``"aoa"`` or ``"csi"``; see ``scenario_to_table``.
    tx_sets, rx_sets, max_paths : passed to ``deepmimo.load``.
    dim : 3 keeps (x, y, z); 2 drops the height.
    power_offset_db, sensitivity_dbm, coherent : RSSI options. DeepMIMO normalises path powers
        to a 0 dBW (30 dBm) transmitter, so the default RSSI is that of a 30 dBm transmitter;
        ``power_offset_db = P_tx_dBm - 30`` models another transmit power (e.g. -10 for 20 dBm).
    n_antennas, spacing_wavelengths, orientation, subcarriers, subcarrier_spacing_hz : CSI array
        and OFDM grid (default 802.11n HT20).

    References
    ----------
    A. Alkhateeb, "DeepMIMO: A generic deep learning dataset for millimeter wave and massive
        MIMO applications", ITA 2019. arXiv:1902.06435. https://deepmimo.net
    """

    name = "deepmimo"
    urls = ()
    files = {"all": ()}
    split_aliases = {}
    meta = {"modality": "rssi", "modalities": MODALITIES, "crs": "local", "pos_units": "m",
            "source": "simulated", "license": "per scenario (see deepmimo.net)",
            "citation": "Alkhateeb, DeepMIMO: A generic deep learning dataset for millimeter wave and massive "
                        "MIMO applications, ITA 2019", "url": "https://deepmimo.net", "doi": "arXiv:1902.06435"}

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, scenario: str = "asu_campus_3p5",
                 modality: str = "rssi", tx_sets="all", rx_sets="rx_only", max_paths: int = 25, dim: int = 3,
                 power_offset_db: float = 0.0, sensitivity_dbm: float | None = None, coherent: bool = False,
                 n_antennas: int = 8, spacing_wavelengths: float = 0.5, orientation: float = 0.0,
                 subcarriers=HT20_SUBCARRIERS, subcarrier_spacing_hz: float = OFDM_SPACING_HZ):
        super().__init__(root, download=download, verify=verify)
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, not {modality!r}")
        self.scenario, self.modality = str(scenario).lower(), modality
        self.tx_sets, self.rx_sets, self.max_paths, self.dim = tx_sets, rx_sets, int(max_paths), int(dim)
        self.options = dict(power_offset_db=power_offset_db, sensitivity_dbm=sensitivity_dbm, coherent=coherent,
                            n_antennas=n_antennas, spacing_wavelengths=spacing_wavelengths, orientation=orientation,
                            subcarriers=np.asarray(subcarriers), subcarrier_spacing_hz=subcarrier_spacing_hz)

    @property
    def scenario_folder(self) -> Path:
        return self.root / self.scenario

    def check(self, split: str) -> list[Path]:
        self._split(split)
        params = self.scenario_folder / "params.json"
        if not params.is_file():
            if not self.download:
                raise FileNotFoundError(f"{params} not found; pass root=... or download=True")
            self._download(split)
            if not params.is_file():
                raise FileNotFoundError(f"{params} not found after downloading {self.scenario}")
        return [params]

    def _download(self, split=None) -> None:
        dm = requires("deepmimo", "sim")
        self.root.mkdir(parents=True, exist_ok=True)
        with _scenarios_folder(dm, self.root):
            dm.download(self.scenario)

    def load(self, split: str = "all") -> SampleTable:
        split = self._split(split)
        self.check(split)
        dm = requires("deepmimo", "sim")
        with _scenarios_folder(dm, self.root):
            data = dm.load(self.scenario, tx_sets=self.tx_sets, rx_sets=self.rx_sets, max_paths=self.max_paths,
                           matrices=list(_MATRICES))
        children = list(getattr(data, "datasets", None) or [data])
        links = []
        for child in children:
            txrx = child["txrx"]
            link = {k: np.asarray(child[k]) for k in _MATRICES}
            link.update(tx_id=f"tx{txrx['tx_set_id']}-{txrx['tx_idx']}", rx_set=int(txrx["rx_set_id"]))
            links.append(link)
        table = scenario_to_table(links, modality=self.modality, dim=self.dim, **self.options)
        rt = {k: v for k, v in dict(children[0]["rt_params"]).items() if isinstance(v, (str, int, float, bool))}
        return table.replace(meta={**self.meta, **table.meta, "name": self.name, "split": split,
                                   "scenario": self.scenario, "rt_params": rt, "carrier_hz": rt.get("frequency"),
                                   "deepmimo_version": getattr(dm, "__version__", None)})


__all__ = ["MODALITIES", "DeepMIMO", "first_arrival_range", "path_gains", "paths_to_csi", "received_power_dbm",
           "scenario_to_table", "strongest_path_azimuth"]
