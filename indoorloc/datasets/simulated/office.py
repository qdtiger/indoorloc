"""SyntheticOffice: a deterministic, physically grounded office simulator (no download)."""
from __future__ import annotations

import hashlib
import json
import zlib

import numpy as np

from ..._version import __version__
from ...core import SampleTable
from .._base import Dataset
from . import building as bld
from . import magnetic as mag
from . import optical
from .channel import HT20_SUBCARRIERS, OFDM_SPACING_HZ, multipath, paths_to_csi, subcarrier_offsets
from .measurements import simulate_aoa, simulate_ranges, simulate_tdoa
from .propagation import (COST231_FLOOR_EXPONENT_B, COST231_FLOOR_LOSS_DB, COST231_WALL_LOSS_DB,
                          INH_OFFICE_SHADOW_STD_DB, INH_SHADOW_DECORRELATION_M, SPEED_OF_LIGHT, ShadowingMap,
                          apply_sensitivity, inh_office_path_loss, log_distance_path_loss, multi_wall_path_loss,
                          thermal_noise_dbm)

MODALITIES = ("wifi_rssi", "ble_rssi", "ranges", "tdoa", "aoa", "csi", "imu", "vlc", "magnetic")
_NO_AP = ("imu", "vlc", "magnetic")  # modalities without generic anchors (n_aps does not apply)
_ALIASES = {"trajectory": "imu"}
PATH_LOSS_MODELS = ("multiwall", "log_distance", "3gpp_inh")

# Per-modality defaults; every key can be overridden through ``physics=``.
_DEFAULTS = {
    "wifi_rssi": dict(n_aps=8, layout="grid", anchor_height=2.6, frequency_hz=2.437e9, tx_power_dbm=20.0,
                      sensitivity_dbm=-95.0, noise_std=3.0, scan_rate_hz=1.0, quantize_db=1.0),
    "ble_rssi": dict(n_aps=12, layout="grid", anchor_height=2.6, frequency_hz=2.44e9, tx_power_dbm=0.0,
                     sensitivity_dbm=-100.0, noise_std=4.0, scan_rate_hz=1.0, quantize_db=1.0),
    "ranges": dict(n_aps=4, layout="perimeter", anchor_height=(2.6, 0.6), frequency_hz=6.4896e9, noise_std=0.1,
                   nlos_bias_mean=0.5, scan_rate_hz=5.0),
    "tdoa": dict(n_aps=5, layout="perimeter", anchor_height=(2.6, 0.6), frequency_hz=6.4896e9, noise_std=0.1,
                 nlos_bias_mean=0.5, scan_rate_hz=5.0),
    "aoa": dict(n_aps=4, layout="perimeter", anchor_height=2.6, frequency_hz=5.18e9, noise_std=float(np.deg2rad(3.0)),
                nlos_std=float(np.deg2rad(10.0)), scan_rate_hz=5.0),
    "csi": dict(n_aps=3, layout="perimeter", anchor_height=2.6, frequency_hz=5.18e9, tx_power_dbm=20.0,
                noise_figure_db=7.0, subcarrier_spacing_hz=OFDM_SPACING_HZ,
                subcarriers=tuple(HT20_SUBCARRIERS.tolist()),
                antenna_spacing_wavelengths=0.5, reflection_coef=(0.4, 0.6), scatter_coef=0.3, impairments=True,
                max_timing_offset_s=50e-9, scan_rate_hz=5.0),
    "imu": dict(n_aps=0, layout="grid", anchor_height=2.6, step_amplitude=2.0, acc_noise_std=0.05,
                gyro_noise_std=0.005, gyro_bias_std=0.002),
    # Visible light: receiver and noise constants as usually quoted from Komine & Nakagawa (2004),
    # Table I; LED power, half-power angle, spacing and height, the receiver FOV and the detection
    # threshold are assumptions of this simulator.
    "vlc": dict(n_aps=0, led_spacing=2.5, anchor_height=3.0, tx_power_w=5.0, half_power_angle_deg=60.0,
                area_m2=1e-4, fov_deg=70.0, filter_gain=1.0, refractive_index=1.5, responsivity=0.54,
                bandwidth_hz=100e6, background_current_a=5100e-6, temperature_k=295.0, open_loop_gain=10.0,
                capacitance_f_per_m2=1.12e-6, fet_noise_factor=1.5, transconductance_s=30e-3,
                noise_bandwidth_factors=(0.562, 0.0868), detection_snr=5.0, noise_std=None, scan_rate_hz=5.0),
    # Magnetic: a synthetic anomaly field (``simulated.magnetic``); not fitted to a real building.
    "magnetic": dict(n_aps=0, earth_field_ut=50.0, inclination_deg=60.0, declination_deg=0.0, dipole_density=0.1,
                     moment_range=(20.0, 200.0), wall_fraction=0.5, slab_depth=0.1, noise_std=0.5, bias_std=0.0,
                     scan_rate_hz=10.0),
}
_COMMON = dict(device_height=1.2, grid_margin=0.3,
               wall_loss_db=(COST231_WALL_LOSS_DB["light"], COST231_WALL_LOSS_DB["heavy"]),
               floor_loss_db=COST231_FLOOR_LOSS_DB, floor_exponent_b=COST231_FLOOR_EXPONENT_B, exponent=3.0,
               decorrelation_distance=INH_SHADOW_DECORRELATION_M["nlos"], room_width=4.0, corridor_width=2.4,
               door_width=1.0, floor_height=3.5)
_SHADOW_STD = {"multiwall": 4.0, "log_distance": 6.0}
_UNITS = {"wifi_rssi": "dBm", "ble_rssi": "dBm", "ranges": "m", "tdoa": "m", "aoa": "rad",
          "csi": "complex channel gain (linear, unitless)", "imu": "m/s^2 (acc), rad/s (gyr)", "vlc": "W",
          "magnetic": "uT"}
MAGNETIC_FEATURES = ("B", "B_h", "B_v")


def _plain(value):
    """JSON-ready copy of a config value (tuples and numpy arrays/scalars become lists/numbers)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return _plain(value.tolist())
    return value


def _read_only(value):
    if isinstance(value, np.ndarray):
        value = value.view()
        value.flags.writeable = False
    elif isinstance(value, dict):
        value = {k: _read_only(v) for k, v in value.items()}
    return value


class SyntheticOffice(Dataset):
    """A simulated multi-storey office with WiFi/BLE RSSI, ranging, TDoA, AoA, CSI, IMU, VLC or magnetic data.

    Everything is generated from ``seed`` (no download): a corridor office floor plan with
    heavy outer walls and light partitions (``building.office_floor_plan``), anchors on every
    storey, then measurements from physical models:

    * ``wifi_rssi`` / ``ble_rssi``: transmit power minus path loss (``path_loss``:
      ``"multiwall"`` COST 231 multi-wall model with exact wall and floor counts,
      ``"log_distance"`` with exponent 3 plus the COST 231 floor term, or ``"3gpp_inh"``
      TR 38.901 InH-Office with LOS decided by the geometry (no wall or floor on the direct
      path) plus the COST 231 floor term for links between storeys, which TR 38.901 InH does
      not model), spatially correlated log-normal shadowing per AP and storey (Gudmundson
      exponential correlation, 6 m decorrelation distance for every model; ``"3gpp_inh"``
      scales it by 3 dB on LOS and 8.03 dB on NLOS links unless ``shadowing_std_db`` is
      given), i.i.d. temporal noise ``noise_std`` dB per scan, integer-dBm quantisation, and
      NaN below the receiver sensitivity.
    * ``ranges`` (UWB ToA): Gaussian noise ``noise_std`` m plus an exponential positive bias on
      links whose direct path crosses a wall or floor. ``tdoa``: the same ToA model,
      differenced against anchor 0. ``aoa``: azimuth relative to each anchor's boresight
      (facing the building centre) with Gaussian noise and extra error on NLOS links.
    * ``csi``: an M-element ULA (``n_antennas``) at each AP over 802.11n HT20 subcarriers at
      5.18 GHz: line of sight, first-order specular reflections off every wall of the AP's
      storey (image method) and ``n_scatterers`` point scatterers per storey, with wall
      penetration losses (counted in plan view against the AP storey's walls only; links
      between storeys add the COST 231 floor term to every path), thermal noise
      (-174 dBm/Hz, 7 dB noise figure, 20 dBm transmit power) and, by default, a random
      common phase and timing offset per packet and AP.
      ``X`` is (N, n_aps * n_antennas, 1, n_subcarriers) complex64; ``meta["antenna_anchor"]``
      maps each receive antenna row to its AP. Antenna ``m`` of an AP sits at
      ``(m - (M - 1) / 2) * meta["antenna_spacing_m"]`` from the AP along the horizontal
      direction ``anchor_orientations + pi / 2``, so a device at bearing ``theta`` from the
      boresight (the ``aoa`` convention) gives an inter-antenna phase step of
      ``2 pi (d / lambda) cos(elevation) sin(theta)`` (``channel.ula_steering``).
    * ``imu`` (alias ``trajectory``): 6-axis body-frame IMU at ``imu_rate_hz`` along the walks,
      ``meta["channels"]`` names the columns; only the ``trajectory`` split exists, and it is
      what ``split=None`` loads.
    * ``vlc``: received optical power (W) from LEDs on a ceiling grid in every room
      (``led_spacing`` m, ``anchor_height`` 3.0 m, facing down, Lambertian order from
      ``half_power_angle_deg``, ``tx_power_w`` each) at a photodiode facing up (``area_m2``,
      ``fov_deg``, ideal concentrator of index ``refractive_index``): the line-of-sight
      Lambertian gain (``optical.lambertian_gain``), zero through walls and floors (doorways
      pass light), no reflections. Noise: the shot and thermal noise of Komine & Nakagawa
      (2004) at ``bandwidth_hz`` (``noise_std`` given: a constant Gaussian std in W instead);
      readings below ``detection_snr`` (5) times the zero-signal noise std are NaN (not seen):
      with a few hundred LEDs a 3-sigma threshold lets noise through at distant LEDs, whose
      noise-level readings then invert to plausible but wrong distances.
      ``meta``: ``led_positions`` (A, 3), ``anchor_normals`` (A, 3), ``lambertian_order``,
      ``tx_power_w``, ``receiver_area_m2``, ``receiver_fov_rad``, ``filter_gain``,
      ``concentrator_gain``, ``device_height`` (the receiver height above each storey floor).
      Not simulated: wall reflections (Gu et al. 2016, cited in ``signals.vlc``),
      LED-to-LED power tolerance, receiver tilt and changing ambient light. VLC accuracy on this
      data is therefore an upper bound.
    * ``magnetic``: per-sample features ``[B, B_h, B_v]`` (uT; the ``magnetic`` layout of
      ``signals.magnetic``) of a **synthetic** field: the Earth's field (``earth_field_ut``,
      ``inclination_deg``, ``declination_deg``; the local frame is taken as East-North-Up) plus
      magnetic dipoles in the floor slabs, half under walls (``simulated.magnetic``;
      ``dipole_density`` per m^2 per slab, moments log-uniform in ``moment_range`` A m^2).
      The device is held flat facing its walking direction (random headings at static
      points); readings get white noise ``noise_std`` uT per axis and, if ``bias_std > 0``, a
      residual hard-iron offset per device (one device per train/test split and per walk).
      Walks carry magnetic sequences (``trajectory`` split, 10 Hz by default).
      ``meta``: ``earth_field`` (3,), ``dipole_positions`` (K, 3), ``dipole_moments`` (K, 3).
      The field is perfectly repeatable, unlike measured ones (``simulated.magnetic``), so
      magnetic accuracy on this data is an upper bound.

    Splits: ``train`` is the reference grid (``grid_spacing`` m, 0.3 m from walls) with
    ``samples_per_point`` scans per point (groups ``point``, ``room``); ``test`` has
    ``n_test`` uniformly random free points, one scan each; ``trajectory`` has
    ``n_trajectories`` random walks of ``trajectory_duration`` s sampled at ``scan_rate_hz``
    (IMU: ``imu_rate_hz``), with groups ``trajectory``, ``time`` (s from the walk's start)
    and ``room`` (IMU adds ``step``, the number of completed steps). The walks, test points,
    floor plan and anchors do not depend on the modality, so the IMU and RSSI tables of one
    seed describe the same walks and can be joined on (trajectory, time). Every table has
    ``groups["source"] == "simulated"``.

    ``meta`` holds ``anchors`` (A, dim) in the frame of ``pos``, ``anchor_floor``,
    ``anchor_orientations`` (boresight azimuths, radians), ``floor_plan`` (the arrays of
    ``building.FloorPlan.to_meta``: ``walls`` (W, 4), ``wall_floor``, ``wall_type``, ``rooms``,
    ``bounds``, ``floor_height``, ``connectors``, ...), ``simulation`` (every parameter used)
    and ``config_sha256`` (a digest of those parameters and the library version).

    Parameters
    ----------
    seed : int, the only source of randomness (independent streams per component).
    modality : one of ``wifi_rssi``, ``ble_rssi``, ``ranges``, ``tdoa``, ``aoa``, ``csi``, ``imu``,
        ``vlc``, ``magnetic``.
    n_floors : storeys (floor labels 0..n_floors-1).
    n_aps : anchors per storey (default per modality: 8 WiFi, 12 BLE, 4 ranges, 5 TDoA, 4 AoA, 3 CSI);
        not used by ``imu``, ``vlc`` (LEDs follow ``led_spacing``) and ``magnetic``.
    grid_spacing, samples_per_point, n_test, n_trajectories, trajectory_duration : split sizes.
    dim : 2 (pos is x, y) or 3 (pos is x, y, z). Physics is always 3-D; with ``dim=2`` the
        ranging and angle anchors are mounted at device height, so on one storey the
        noiseless ranges, TDoA and bearings equal the 2-D geometry of ``pos`` and
        ``meta["anchors"]`` exactly (RSSI and CSI APs stay on the ceiling, 2.6 m).
    size : (width, depth) of the footprint in metres.
    path_loss : ``"multiwall"``, ``"log_distance"`` or ``"3gpp_inh"`` (RSSI modalities).
    noise_std : per-reading noise (dB for RSSI, m for ranges/TDoA, rad for AoA, W for VLC, uT per
        axis for magnetic); default per modality (VLC: the physical shot + thermal noise model).
    shadowing_std_db : shadowing standard deviation (default 4 dB multi-wall, 6 dB log-distance;
        3GPP uses 3 / 8.03 dB for LOS / NLOS links).
    n_antennas, n_scatterers : CSI array size and scatterers per storey.
    scan_rate_hz, imu_rate_hz : sampling rates along trajectories.
    physics : dict overriding any default (``tx_power_dbm``, ``sensitivity_dbm``, ``frequency_hz``,
        ``wall_loss_db``, ``floor_loss_db``, ``exponent``, ``decorrelation_distance``,
        ``nlos_bias_mean``, ``nlos_std``, ``reflection_coef``, ``impairments``, ``quantize_db``,
        ``device_height``, ``anchor_height``, ``layout``, ...).
    root, download, verify : accepted for the ``load_dataset`` signature and ignored.

    References
    ----------
    E. Damosso (ed.), "COST Action 231: Digital mobile radio towards future generation systems,
        Final report", European Commission EUR 18957, 1999, section 4.7.
    3GPP TR 38.901 V17.0.0, "Study on channel model for frequencies from 0.5 to 100 GHz", 2022,
        Tables 7.4.1-1, 7.4.2-1, 7.5-6.
    M. Gudmundson, "Correlation model for shadow fading in mobile radio systems", Electronics
        Letters 27(23):2145-2146, 1991. DOI 10.1049/el:19911328
    S. Gezici et al., "Localization via ultra-wideband radios", IEEE Signal Processing Magazine
        22(4):70-84, 2005. DOI 10.1109/MSP.2005.1458289
    M. Kotaru, K. Joshi, D. Bharadia and S. Katti, "SpotFi: Decimeter level localization using
        WiFi", ACM SIGCOMM 2015. DOI 10.1145/2785956.2787487
    R. Harle, "A survey of indoor inertial positioning systems for pedestrians", IEEE
        Communications Surveys & Tutorials 15(3):1281-1293, 2013. DOI 10.1109/SURV.2012.121912.00075
    T. Komine, M. Nakagawa, "Fundamental analysis for visible-light communication system using LED
        lights", IEEE Transactions on Consumer Electronics 50(1):100-107, 2004. DOI 10.1109/TCE.2004.1277847
    B. Li, T. Gallagher, A. G. Dempster, C. Rizos, "How feasible is the use of magnetic field alone
        for indoor positioning?", IPIN 2012. DOI 10.1109/IPIN.2012.6418880
    """

    name = "synthetic_office"
    urls = ()
    files = {"train": (), "test": (), "trajectory": ()}
    split_aliases = {"trajectories": "trajectory", "walk": "trajectory"}  # no validation split: carve one from train
    meta = {"modality": "wifi_rssi", "modalities": MODALITIES, "crs": "local", "pos_units": "m",
            "license": "CC0-1.0 (generated data)", "source": "simulated",
            "citation": "IndoorLoc SyntheticOffice simulator (indoorloc.datasets.simulated)",
            "url": "https://github.com/qdtiger/indoorloc"}

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, seed: int = 0,
                 modality: str = "wifi_rssi", n_floors: int = 1, n_aps: int | None = None, grid_spacing: float = 2.0,
                 samples_per_point: int = 5, n_test: int = 200, n_trajectories: int = 4,
                 trajectory_duration: float = 60.0, dim: int = 2, size=(40.0, 20.0), path_loss: str = "multiwall",
                 noise_std: float | None = None, shadowing_std_db: float | None = None, n_antennas: int = 4,
                 n_scatterers: int = 0, scan_rate_hz: float | None = None, imu_rate_hz: float = 50.0,
                 physics: dict | None = None):
        super().__init__(root, download=download, verify=verify)
        modality = _ALIASES.get(modality, modality)
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES} (or 'trajectory' for imu), not {modality!r}")
        if modality in ("vlc", "magnetic") and n_aps is not None:
            raise ValueError(f"n_aps does not apply to {modality} (VLC LEDs follow physics['led_spacing'])")
        if path_loss not in PATH_LOSS_MODELS:
            raise ValueError(f"path_loss must be one of {PATH_LOSS_MODELS}, not {path_loss!r}")
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if int(seed) != seed or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name, value, low in (("n_floors", n_floors, 1), ("samples_per_point", samples_per_point, 1),
                                 ("n_test", n_test, 1), ("n_trajectories", n_trajectories, 1),
                                 ("n_antennas", n_antennas, 1), ("n_scatterers", n_scatterers, 0)):
            if int(value) != value or value < low:
                raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
        for name, value in (("grid_spacing", grid_spacing), ("trajectory_duration", trajectory_duration),
                            ("imu_rate_hz", imu_rate_hz),
                            ("scan_rate_hz", 1.0 if scan_rate_hz is None else scan_rate_hz)):
            if not float(value) > 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        unknown = set(physics or {}) - set(_COMMON) - set().union(*(set(d) for d in _DEFAULTS.values()))
        if unknown:
            raise ValueError(f"unknown physics keys: {sorted(unknown)}")
        self.seed, self.modality, self.n_floors, self.dim = int(seed), modality, int(n_floors), int(dim)
        self.grid_spacing, self.samples_per_point = float(grid_spacing), int(samples_per_point)
        self.n_test = int(n_test)
        self.n_trajectories, self.trajectory_duration = int(n_trajectories), float(trajectory_duration)
        self.size, self.path_loss = (float(size[0]), float(size[1])), path_loss
        self.n_antennas, self.n_scatterers, self.imu_rate_hz = int(n_antennas), int(n_scatterers), float(imu_rate_hz)
        p = {**_COMMON, **_DEFAULTS[modality], **(physics or {})}
        p["n_aps"] = int(p["n_aps"] if n_aps is None else n_aps)
        if noise_std is not None:
            p["noise_std"] = float(noise_std)
        if scan_rate_hz is not None:
            p["scan_rate_hz"] = float(scan_rate_hz)
        p["shadowing_std_db"] = (float(shadowing_std_db) if shadowing_std_db is not None
                                 else _SHADOW_STD.get(path_loss))  # None: 3GPP per-link LOS/NLOS values
        if self.dim == 2 and modality in ("ranges", "tdoa", "aoa"):
            p["anchor_height"] = p["device_height"]  # planar geometry: 2-D ranges/angles are exact on a storey
        need = 2 if modality == "tdoa" else 1
        if modality not in _NO_AP and p["n_aps"] < need:
            raise ValueError(f"n_aps={p['n_aps']} is too small for {modality} (need >= {need})")
        rate = self.imu_rate_hz if modality == "imu" else p["scan_rate_hz"]
        if not float(rate) > 0 or self.trajectory_duration * rate < 1.0 - 1e-9:
            raise ValueError(f"trajectory_duration * rate must give at least one sample per walk, got "
                             f"{self.trajectory_duration} s at {rate} Hz")
        self.physics = p
        self._world = None

    # ------------------------------------------------------------------ plumbing
    def _rng(self, *keys: str) -> np.random.Generator:
        """An independent stream per component, stable across modalities and Python runs."""
        return np.random.default_rng([self.seed, *(zlib.crc32(k.encode()) for k in keys)])

    def check(self, split: str) -> list:
        """Nothing to download or verify: the data is generated."""
        self._split(split)
        return []

    def _download(self, split=None) -> None:  # pragma: no cover - never called
        return None

    def config(self) -> dict:
        """Every parameter that determines the generated data."""
        phys = _plain(self.physics)
        return {"generator": f"{type(self).__module__}.{type(self).__name__}", "version": __version__,
                "seed": self.seed, "modality": self.modality, "n_floors": self.n_floors, "dim": self.dim,
                "grid_spacing": self.grid_spacing, "samples_per_point": self.samples_per_point, "n_test": self.n_test,
                "n_trajectories": self.n_trajectories, "trajectory_duration": self.trajectory_duration,
                "size": list(self.size), "path_loss": self.path_loss, "n_antennas": self.n_antennas,
                "n_scatterers": self.n_scatterers, "imu_rate_hz": self.imu_rate_hz, "physics": phys}

    @property
    def world(self) -> dict:
        """Floor plan, anchors, shadowing maps and scatterers (built once per instance)."""
        if self._world is None:
            p = self.physics
            plan = bld.office_floor_plan(size=self.size, n_floors=self.n_floors, room_width=p["room_width"],
                                         corridor_width=p["corridor_width"], door_width=p["door_width"],
                                         floor_height=p["floor_height"], random_state=self._rng("floor-plan"))
            w = {"plan": plan}
            if p["n_aps"] > 0:
                xyz, fl = bld.place_anchors(plan, p["n_aps"], layout=p["layout"], height=p["anchor_height"],
                                            random_state=self._rng("anchors", p["layout"]))
                w.update(anchors=xyz, anchor_floor=fl, orientations=bld.facing_centre(plan, xyz[:, :2]))
            if self.modality in ("wifi_rssi", "ble_rssi"):
                w["shadowing"] = ShadowingMap.generate(plan.bounds, std_db=1.0,
                                                       decorrelation_distance=p["decorrelation_distance"],
                                                       n_maps=len(xyz) * plan.n_floors,
                                                       random_state=self._rng("shadowing", self.modality))
            if self.modality == "vlc":
                xyz, fl = optical.room_grid_leds(plan, p["led_spacing"], p["anchor_height"])
                w.update(anchors=xyz, anchor_floor=fl)
            if self.modality == "magnetic":
                src, mom = mag.place_dipoles(plan, density=p["dipole_density"], moment_range=p["moment_range"],
                                             wall_fraction=p["wall_fraction"], depth=p["slab_depth"],
                                             random_state=self._rng("magnetic-sources"))
                w.update(dipoles=src, moments=mom, earth=mag.earth_field(p["earth_field_ut"], p["inclination_deg"],
                                                                          p["declination_deg"]))
            if self.modality == "csi" and self.n_scatterers > 0:
                rng = self._rng("scatterers")
                sc = []
                for f in range(plan.n_floors):
                    xy, _ = bld.random_points(plan, self.n_scatterers, margin=0.2, floors=[f], random_state=rng)
                    sc.append(np.column_stack([xy, plan.height_of(f, rng.uniform(0.3, 2.7, len(xy)))]))
                w["scatterers"] = sc
            self._world = w
        return self._world

    @property
    def floor_plan(self) -> bld.FloorPlan:
        """The generated building (the same arrays as ``meta["floor_plan"]``)."""
        return self.world["plan"]

    # ------------------------------------------------------------------ splits
    @property
    def default_splits(self) -> tuple[str, ...]:
        """``("train", "test")``; ``("trajectory",)`` for ``modality="imu"``, which has no other split,
        so ``load_dataset("synthetic_office", modality="imu")`` and ``.load()`` return the walks."""
        return ("trajectory",) if self.modality == "imu" else super().default_splits

    def load(self, split: str | None = None) -> SampleTable:
        """One split as a SampleTable; ``None`` means the first of :attr:`default_splits`
        (``"train"``, or ``"trajectory"`` for ``modality="imu"``)."""
        split = self._split(split)
        if self.modality == "imu" and split != "trajectory":
            raise ValueError(f"the imu modality only has the 'trajectory' split, not {split!r} "
                             "(split=None loads it)")
        table = {"train": self._train, "test": self._test, "trajectory": self._trajectory}[split]()
        return table.replace(meta={**self.meta, **table.meta, "name": self.name, "split": split})

    def _xyz(self, xy, floor) -> np.ndarray:
        plan = self.world["plan"]
        return np.column_stack([xy[:, :2], plan.height_of(floor, self.physics["device_height"])])

    def _train(self) -> SampleTable:
        plan = self.world["plan"]
        xy, fl = bld.reference_grid(plan, self.grid_spacing, margin=self.physics["grid_margin"])
        if len(xy) == 0:
            raise ValueError(f"no reference point lies {self.physics['grid_margin']} m from the walls on a "
                             f"{self.grid_spacing} m grid; use a larger size or a smaller grid_spacing")
        k = self.samples_per_point
        X = self._measure(self._xyz(xy, fl), fl, k, self._rng("noise", self.modality, "train"))
        point = np.repeat(np.arange(len(xy)), k)
        groups = {"point": point, "room": plan.room_of(xy, fl)[point]}
        return self._table("train", X, self._xyz(xy, fl)[point], fl[point], groups)

    def _test(self) -> SampleTable:
        plan = self.world["plan"]
        xy, fl = bld.random_points(plan, self.n_test, margin=self.physics["grid_margin"],
                                   random_state=self._rng("test-points"))
        X = self._measure(self._xyz(xy, fl), fl, 1, self._rng("noise", self.modality, "test"))
        return self._table("test", X, self._xyz(xy, fl), fl, {"room": plan.room_of(xy, fl)})

    def _trajectory(self) -> SampleTable:
        plan = self.world["plan"]
        graph = bld.navigation_graph(plan)
        imu = self.modality == "imu"
        rate = self.imu_rate_hz if imu else self.physics["scan_rate_hz"]
        times = np.arange(int(np.floor(self.trajectory_duration * rate + 1e-9))) / rate
        parts = []
        for j in range(self.n_trajectories):
            route = bld.random_walk(plan, self.trajectory_duration, graph=graph,
                                    device_height=self.physics["device_height"], random_state=self._rng("walk", str(j)))
            traj = route.sample(times)
            if imu:
                p = self.physics
                X = bld.synthesize_imu(traj, step_amplitude=p["step_amplitude"], acc_noise_std=p["acc_noise_std"],
                                       gyro_noise_std=p["gyro_noise_std"], gyro_bias_std=p["gyro_bias_std"],
                                       random_state=self._rng("imu", str(j))).astype(np.float32)
            else:
                X = self._measure(traj.pos, traj.floor, 1, self._rng("noise", self.modality, "trajectory", str(j)),
                                  heading=traj.heading)
            groups = {"trajectory": np.full(len(times), j, dtype=np.int64), "time": times,
                      "room": plan.room_of(traj.pos, traj.floor)}
            if imu:
                groups["step"] = traj.step
            parts.append((X, traj.pos, traj.floor, groups))
        X = np.concatenate([q[0] for q in parts])
        groups = {k: np.concatenate([q[3][k] for q in parts]) for k in parts[0][3]}
        meta = {"channels": bld.IMU_CHANNELS, "rate_hz": self.imu_rate_hz} if imu else {"rate_hz": rate}
        return self._table("trajectory", X, np.concatenate([q[1] for q in parts]),
                           np.concatenate([q[2] for q in parts]), groups, meta)

    def _table(self, split, X, xyz, floor, groups, extra=None) -> SampleTable:
        w, n = self.world, len(X)
        cfg = self.config()
        meta = {"modality": self.modality, "units": _UNITS[self.modality],
                "pos_names": ("x", "y", "z")[:self.dim], "floors": tuple(range(self.n_floors)),
                "floor_plan": w["plan"].to_meta(), "simulation": cfg,
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
        if self.modality == "vlc":
            p, a = self.physics, len(w["anchors"])
            m = optical.lambertian_order(np.deg2rad(p["half_power_angle_deg"]))
            meta.update(anchors=w["anchors"][:, :self.dim].copy(), anchor_floor=w["anchor_floor"],
                        feature_names=tuple(f"LED{i:03d}" for i in range(a)), led_positions=w["anchors"].copy(),
                        anchor_normals=np.tile([0.0, 0.0, -1.0], (a, 1)), lambertian_order=m,
                        tx_power_w=float(p["tx_power_w"]), receiver_area_m2=float(p["area_m2"]),
                        receiver_fov_rad=float(np.deg2rad(p["fov_deg"])), filter_gain=float(p["filter_gain"]),
                        concentrator_gain=self._concentrator_gain(), device_height=float(p["device_height"]))
        elif self.modality == "magnetic":
            meta.update(feature_names=MAGNETIC_FEATURES, earth_field=w["earth"], dipole_positions=w["dipoles"],
                        dipole_moments=w["moments"])
        elif "anchors" in w and self.modality != "imu":
            a = len(w["anchors"])
            names = tuple(f"AP{i:02d}" for i in range(a))
            meta.update(anchors=w["anchors"][:, :self.dim].copy(), anchor_floor=w["anchor_floor"],
                        anchor_orientations=w["orientations"], feature_names=names)
            if self.modality == "tdoa":
                meta.update(feature_names=tuple(f"{b}-{names[0]}" for b in names[1:]), reference_anchor=0)
            if self.modality == "csi":
                p = self.physics
                offsets = subcarrier_offsets(p["subcarriers"], p["subcarrier_spacing_hz"])
                lam = SPEED_OF_LIGHT / p["frequency_hz"]
                meta.pop("feature_names")
                meta.update(carrier_hz=p["frequency_hz"], subcarrier_offsets_hz=offsets,
                            subcarriers=np.asarray(p["subcarriers"], dtype=np.int64),  # CONTRACTS.md: meta["subcarriers"]
                            bandwidth_hz=p["subcarrier_spacing_hz"] * 64, n_antennas=self.n_antennas,
                            antenna_spacing_m=p["antenna_spacing_wavelengths"] * lam,
                            antenna_anchor=np.repeat(np.arange(a), self.n_antennas),
                            csi_axes=("rx", "tx", "subcarrier"))
        meta.update(extra or {})
        groups = {"source": np.full(n, "simulated"), **groups}
        ids = np.char.add(f"{split}-", np.char.zfill(np.arange(n).astype(str), 5))
        return SampleTable(X, xyz[:, :self.dim], floor, None, groups, ids, _read_only(meta))

    # -------------------------------------------------------------- measurements
    def _links(self, xyz):
        """(M, A) 3-D distances, (M, A, n_materials) wall counts and (M, A) floor counts."""
        w = self.world
        plan, anchors = w["plan"], w["anchors"]
        m, a = len(xyz), len(anchors)
        dev = np.repeat(xyz, a, axis=0)
        ap = np.tile(anchors, (m, 1))
        dist = np.linalg.norm(dev - ap, axis=1).reshape(m, a)
        walls = plan.wall_crossings(dev, ap).reshape(m, a, -1)
        floors = plan.floor_crossings(dev, ap).reshape(m, a)
        return dist, walls, floors

    def _floor_term(self, floors) -> np.ndarray:
        p = self.physics
        kf = np.asarray(floors, dtype=np.float64)
        safe = np.maximum(kf, 1.0)
        return np.where(kf > 0, safe ** ((safe + 2) / (safe + 1) - p["floor_exponent_b"]) * p["floor_loss_db"], 0.0)

    def _concentrator_gain(self) -> float:
        p = self.physics
        return float(p["refractive_index"]) ** 2 / np.sin(np.deg2rad(p["fov_deg"])) ** 2

    def _vlc_noise_std(self, power) -> np.ndarray:
        """Optical-power noise std (W) at received power ``power`` (W)."""
        p = self.physics
        if p["noise_std"] is not None:
            return np.full(np.shape(power), float(p["noise_std"]))
        i2, i3 = p["noise_bandwidth_factors"]
        var = optical.noise_variance(power, responsivity=p["responsivity"], bandwidth=p["bandwidth_hz"],
                                     background_current=p["background_current_a"], area=p["area_m2"],
                                     temperature=p["temperature_k"], open_loop_gain=p["open_loop_gain"],
                                     capacitance_per_area=p["capacitance_f_per_m2"],
                                     channel_noise_factor=p["fet_noise_factor"],
                                     transconductance=p["transconductance_s"], i2=i2, i3=i3)
        return np.sqrt(var) / p["responsivity"]

    def _vlc(self, xyz, floor, repeats: int, rng) -> np.ndarray:
        w, p = self.world, self.physics
        plan, leds, lf = w["plan"], w["anchors"], w["anchor_floor"]
        P = np.zeros((len(xyz), len(leds)))
        order = optical.lambertian_order(np.deg2rad(p["half_power_angle_deg"]))
        for f in np.unique(floor):
            rows, cols = np.flatnonzero(floor == f), np.flatnonzero(lf == f)   # slabs are opaque
            H = optical.lambertian_gain(xyz[rows], leds[cols], order=order, area=p["area_m2"],
                                        fov=np.deg2rad(p["fov_deg"]), filter_gain=p["filter_gain"],
                                        concentrator_gain=self._concentrator_gain())
            r, c = np.nonzero(H > 0)
            if len(r):
                blocked = plan.wall_crossings(xyz[rows[r]], leds[cols[c]]).sum(axis=-1) > 0
                H[r[blocked], c[blocked]] = 0.0
            P[np.ix_(rows, cols)] = p["tx_power_w"] * H
        X = np.repeat(P, repeats, axis=0)
        std = self._vlc_noise_std(X)
        if np.any(std > 0):
            X = X + std * rng.standard_normal(X.shape)
        threshold = float(p["detection_snr"]) * float(self._vlc_noise_std(np.zeros(1))[0])
        return np.where(X > threshold, X, np.nan)

    def _magnetic(self, xyz, repeats: int, rng, heading) -> np.ndarray:
        w, p = self.world, self.physics
        field = w["earth"] + mag.dipole_field(xyz, w["dipoles"], w["moments"])
        field = np.repeat(field, repeats, axis=0)
        bias = rng.normal(0.0, p["bias_std"], 3) if p["bias_std"] > 0 else None
        if heading is None:  # static scans: the device faces a random direction
            heading = rng.uniform(-np.pi, np.pi, len(field))
        else:
            heading = np.repeat(np.asarray(heading, dtype=np.float64), repeats)
        readings = mag.device_readings(field, heading, noise_std=p["noise_std"], bias=bias, random_state=rng)
        return mag.features(readings)

    def _measure(self, xyz, floor, repeats: int, rng, heading=None) -> np.ndarray:
        """Measurements at M unique positions, ``repeats`` independent readings each (point-major).
        ``heading`` (M,) is the device heading along a walk (magnetic only; random if None)."""
        mod, p = self.modality, self.physics
        if mod == "vlc":
            return self._vlc(xyz, floor, repeats, rng)
        if mod == "magnetic":
            return self._magnetic(xyz, repeats, rng, heading)
        if mod in ("wifi_rssi", "ble_rssi"):
            dist, walls, floors = self._links(xyz)
            f = p["frequency_hz"]
            los = (walls.sum(axis=-1) == 0) & (floors == 0)
            if self.path_loss == "multiwall":
                pl = multi_wall_path_loss(dist, f, walls, floors, wall_loss_db=p["wall_loss_db"],
                                          floor_loss_db=p["floor_loss_db"], b=p["floor_exponent_b"])
                std = np.full(dist.shape, p["shadowing_std_db"])
            elif self.path_loss == "log_distance":
                pl = log_distance_path_loss(dist, f, exponent=p["exponent"]) + self._floor_term(floors)
                std = np.full(dist.shape, p["shadowing_std_db"])
            else:
                pl = inh_office_path_loss(dist, f, los) + self._floor_term(floors)
                std = (np.where(los, INH_OFFICE_SHADOW_STD_DB["los"], INH_OFFICE_SHADOW_STD_DB["nlos"])
                       if p["shadowing_std_db"] is None else np.full(dist.shape, p["shadowing_std_db"]))
            maps = self.world["shadowing"]
            nf = self.world["plan"].n_floors
            rows = np.arange(len(xyz))
            shadow = np.column_stack([np.stack([maps[a * nf + g](xyz) for g in range(nf)])[floor, rows]
                                      for a in range(dist.shape[1])])
            mean = p["tx_power_dbm"] - pl + std * shadow
            X = np.repeat(mean, repeats, axis=0)
            if p["noise_std"] > 0:
                X = X + rng.normal(0.0, p["noise_std"], X.shape)
            if p.get("quantize_db"):
                X = np.round(X / p["quantize_db"]) * p["quantize_db"]
            return apply_sensitivity(X, p["sensitivity_dbm"]).astype(np.float32)
        if mod in ("ranges", "tdoa", "aoa"):
            _, walls, floors = self._links(xyz)
            nlos = np.repeat((walls.sum(axis=-1) > 0) | (floors > 0), repeats, axis=0)
            pos = np.repeat(xyz, repeats, axis=0)
            anchors = self.world["anchors"]
            if mod == "ranges":
                return simulate_ranges(pos, anchors, noise_std=p["noise_std"], nlos=nlos,
                                       nlos_bias_mean=p["nlos_bias_mean"], random_state=rng)
            if mod == "tdoa":
                return simulate_tdoa(pos, anchors, noise_std=p["noise_std"], nlos=nlos,
                                     nlos_bias_mean=p["nlos_bias_mean"], random_state=rng)
            return simulate_aoa(pos, anchors, self.world["orientations"], noise_std=p["noise_std"], nlos=nlos,
                                nlos_std=p["nlos_std"], random_state=rng)
        if mod == "csi":
            return self._csi(xyz, repeats, rng)
        raise ValueError(f"no per-point measurements for modality {mod!r}")

    def _csi(self, xyz, repeats: int, rng) -> np.ndarray:
        w, p = self.world, self.physics
        plan, anchors = w["plan"], w["anchors"]
        offsets = subcarrier_offsets(p["subcarriers"], p["subcarrier_spacing_hz"])
        refl = np.asarray(p["reflection_coef"], dtype=np.float64)
        loss = np.asarray(p["wall_loss_db"], dtype=np.float64)
        blocks = []
        for a, (ap, fa) in enumerate(zip(anchors, w["anchor_floor"])):
            sel = plan.wall_floor == fa
            ap_rep = np.broadcast_to(ap, xyz.shape)
            floors = plan.floor_crossings(xyz, ap_rep)
            scat = w.get("scatterers", [None] * plan.n_floors)[fa]
            paths = multipath(xyz, ap_rep, plan.walls[sel], frequency_hz=p["frequency_hz"],
                              wall_loss_db=loss[plan.wall_type[sel]], reflection_coef=refl[plan.wall_type[sel]],
                              scatterers=scat, scatter_coef=p["scatter_coef"], extra_loss_db=self._floor_term(floors))
            blocks.append(paths_to_csi(paths, offsets, n_antennas=self.n_antennas, orientation=w["orientations"][a],
                                       spacing_wavelengths=p["antenna_spacing_wavelengths"]))
        H = np.repeat(np.concatenate(blocks, axis=1), repeats, axis=0)          # (N, A*M, K)
        n, _, k = H.shape
        noise_dbm = thermal_noise_dbm(p["subcarrier_spacing_hz"], noise_figure_db=p["noise_figure_db"])
        per_sc_dbm = p["tx_power_dbm"] - 10.0 * np.log10(k)
        sigma = np.sqrt(10.0 ** ((noise_dbm - per_sc_dbm) / 10.0) / 2.0)
        H = H + sigma * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
        if p["impairments"]:
            a = len(anchors)
            phase = rng.uniform(0.0, 2.0 * np.pi, (n, a))
            sto = rng.uniform(0.0, p["max_timing_offset_s"], (n, a))
            rot = np.exp(1j * phase[..., None] - 2j * np.pi * sto[..., None] * offsets)   # (N, A, K)
            H = H * np.repeat(rot, self.n_antennas, axis=1)
        return H[:, :, None, :].astype(np.complex64)


__all__ = ["MODALITIES", "PATH_LOSS_MODELS", "SyntheticOffice"]
