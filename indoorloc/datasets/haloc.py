"""HALOC: complex WiFi CSI of a person walking a hallway, with 3-D position labels (ESP32-S3)."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import PurePosixPath

import numpy as np

from ..core import SampleTable
from ._base import Dataset

# Positions in the 128 (imag, real) pairs of an ESP32 HT packet with the secondary channel below:
# L-LTF then HT-LTF, each 64 subcarriers in ascending order -32..31 (null guards and DC verified
# on the data: every packet of every sequence is zero exactly outside these positions).
_LLTF = np.r_[6:32, 33:59]          # subcarriers -26..-1, 1..26 (the authors' 52)
_HTLTF = np.r_[68:96, 97:125]       # subcarriers -28..-1, 1..28
SUBCARRIERS = {
    "lltf": (_LLTF, np.r_[-26:0, 1:27]),
    "htltf": (_HTLTF, np.r_[-28:0, 1:29]),
}
# The packet format these positions assume (every one of the 138,879 packets has it):
# HT (sig_mode 1), 20 MHz (bandwidth 0), no STBC, secondary channel below (2), 128 pairs (len 256).
_FORMAT = {"sig_mode": "1", "bandwidth": "0", "stbc": "0", "secondary_channel": "2", "len": "256"}
_SUBCARRIER_SPACING_HZ = 312.5e3  # 802.11n, 20 MHz

# The Zenodo archive, read in place (28.6 MB; the six CSVs inside are 127 MB). md5 f3bf1e60cc2f0d1bd6e6a75cce247437
# as listed by Zenodo; sha256 of the members: 0.csv 4c882722..., 1.csv aeb98f30..., 2.csv 9666f0a4...,
# 3.csv 3c253aeb..., 4.csv 221cf633..., 5.csv 82caaf40...
_ARCHIVE = ("HALOC.zip", "51183ac2d1ca5126095a583906da843145b779ae47baec9121d987590439b1dc")
_SPLITS = {"train": (0, 1, 2, 3), "valid": (4,), "test": (5,), "all": (0, 1, 2, 3, 4, 5)}  # the authors' split


def parse_esp32_csi(buffer: str, pairs: np.ndarray) -> np.ndarray:
    """``"[i0,r0,i1,r1,...]"`` (ESP-IDF ``wifi_csi_info_t.buf``) -> complex64 CSI of the given pair positions.

    ESP-IDF stores each subcarrier as two signed bytes, imaginary part first, then real part.
    """
    raw = np.array(buffer.strip()[1:-1].split(","), dtype=np.int16)
    return (raw[2 * pairs + 1] + 1j * raw[2 * pairs]).astype(np.complex64)


class HALOC(Dataset):
    """HALOC: WiFi CSI and 3-D positions of a person walking a hallway (Strohmayer & Kampel, ICLR 2024 Tiny Papers).

    An ESP32-S3 behind a directional antenna captured the CSI of about 100 packets/s from a
    transmitter while one person walked up and down a hallway; every packet is labelled with
    the person's 3-D position. Six sequences (``0.csv`` .. ``5.csv``, 138,879 packets in all)
    come with the authors' split: train = sequences 0-3, valid = 4, test = 5. The sequences are
    read straight from the Zenodo archive ``HALOC.zip`` (one sha256), which stays packed.

    ``X``      (N, 1, 1, n_sub) complex64 (one receive and one transmit antenna), the raw int8 I/Q
               of ESP-IDF, **not calibrated** (the receiver's gain control changes their scale).
               ``subcarriers="lltf"`` (default): the 52 L-LTF data subcarriers the authors use;
               ``"htltf"``: the 56 HT-LTF data subcarriers. ``meta["subcarriers"]`` holds their 802.11
               subcarrier numbers (the convention of ``indoorloc.signals.csi``),
               ``meta["subcarrier_offsets_hz"]`` their offsets from ``meta["carrier_hz"]`` (channel 11,
               2.462 GHz; 312.5 kHz spacing). Every packet is checked to be HT, 20 MHz, non-STBC with
               the secondary channel below, the format these subcarrier positions belong to.
    ``pos``    (x, y, z) in metres as given by the authors (x runs 0-20 m along the hallway; z is
               about 1.2-1.3 m).
    ``groups`` ``trajectory`` (sequence 0-5), ``time`` (seconds since the sequence's first packet,
               from the ESP32 microsecond clock).

    The authors' loader (github.com/StrohmayerJ/HALOC) builds subcarrier ``i`` as
    ``complex(buf[2i], buf[2i-1])``, which pairs the imaginary part of subcarrier ``i`` with the real
    part of subcarrier ``i-1``. This loader follows the ESP-IDF layout (``buf[2i]`` imaginary,
    ``buf[2i+1]`` real). On the first 3,000 packets of sequence 0 it gives smoother spectra: mean
    absolute amplitude step between adjacent subcarriers 0.41 vs 0.60, mean phase step 0.034 vs
    0.064 rad. Amplitude features therefore differ slightly from the authors' pipeline.
    The per-packet RSSI, noise floor and rate fields of the CSV are not loaded.

    Parameters
    ----------
    sequences : None (every sequence of the split) or an iterable of sequence numbers (0-5) to
        load a subset, e.g. ``HALOC(sequences=[0]).load("train")``; only the splits holding one of
        them remain. Split ``"all"`` holds all six.
    subcarriers : "lltf" (default) or "htltf".

    References
    ----------
    Strohmayer, J., Kampel, M., "WiFi CSI-based Long-Range Person Localization Using Directional
    Antennas", The Second Tiny Papers Track at ICLR 2024. https://openreview.net/forum?id=AOJFcEh5Eb

    Strohmayer, J., Kampel, M., "HALOC Dataset", Zenodo, 2024. https://doi.org/10.5281/zenodo.10715595

    Espressif Systems, "ESP-IDF Programming Guide: Wi-Fi Channel State Information".
    https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/wifi.html
    """

    name = "haloc"
    urls = ("https://zenodo.org/records/10715595/files/HALOC.zip?download=1",)
    files = dict.fromkeys(_SPLITS, _ARCHIVE)
    split_aliases = {"validation": "valid", "val": "valid"}
    meta = {
        "modality": "csi",
        "units": "raw int8 I/Q of ESP-IDF (not calibrated)",
        "crs": "local",
        "pos_names": ("x", "y", "z"),
        "pos_units": "m",
        "csi_axes": ("rx", "tx", "subcarrier"),
        "device": "ESP32-S3 with a directional antenna, channel 11, HT20",
        "license": "CC BY 4.0 (Zenodo record; the description asks for non-commercial research use)",
        "doi": "10.5281/zenodo.10715595",
        "citation": "Strohmayer, Kampel, WiFi CSI-based Long-Range Person Localization Using Directional "
                    "Antennas, ICLR 2024 Tiny Papers",
        "url": "https://github.com/StrohmayerJ/HALOC",
    }

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, sequences=None,
                 subcarriers: str = "lltf"):
        super().__init__(root, download=download, verify=verify)
        if subcarriers not in SUBCARRIERS:
            raise ValueError(f"subcarriers must be one of {sorted(SUBCARRIERS)}, got {subcarriers!r}")
        self.subcarriers = subcarriers
        every = _SPLITS["all"]
        self.sequences = every if sequences is None else tuple(sorted({int(s) for s in sequences}))
        if not self.sequences or set(self.sequences) - set(every):
            raise ValueError(f"sequences must be a non-empty subset of {list(every)}, got {sequences!r}")
        # a split exists only if it holds a selected sequence
        self.files = {split: _ARCHIVE for split, seqs in _SPLITS.items() if set(seqs) & set(self.sequences)}

    def _parse(self, path, split):
        pairs, indices = SUBCARRIERS[self.subcarriers]
        seqs = [q for q in _SPLITS[split] if q in self.sequences]
        X, pos, trajectory, time, ids = [], [], [], [], []
        with zipfile.ZipFile(path) as archive:
            members = {PurePosixPath(m).name: m for m in archive.namelist()}
            for seq in seqs:
                if f"{seq}.csv" not in members:
                    raise ValueError(f"{path}: no {seq}.csv in the archive")
                with archive.open(members[f"{seq}.csv"]) as raw:
                    reader = csv.reader(io.TextIOWrapper(raw, encoding="ascii", newline=""))
                    col = {name: i for i, name in enumerate(next(reader))}
                    rows = list(reader)
                for key, expected in _FORMAT.items():
                    if any(r[col[key]] != expected for r in rows):
                        raise ValueError(f"{seq}.csv: a packet has {key} != {expected}; the subcarrier positions "
                                         "of this loader hold only for HT20 packets with the secondary channel below")
                X.append(np.stack([parse_esp32_csi(r[col["data"]], pairs) for r in rows]))
                pos.append(np.array([[r[col["x"]], r[col["y"]], r[col["z"]]] for r in rows], dtype=np.float64))
                clock = np.array([int(r[col["local_timestamp"]]) for r in rows], dtype=np.int64) & 0xFFFFFFFF
                step = np.diff(clock)
                step[step < 0] += 1 << 32  # the microsecond counter is 32 bits wide
                time.append(np.concatenate([[0.0], np.cumsum(step) / 1e6]))
                trajectory.append(np.full(len(rows), seq, dtype=np.int64))
                ids += [f"seq{seq}-{i:05d}" for i in range(len(rows))]
        X = np.concatenate(X)[:, None, None, :]
        return SampleTable(X, np.concatenate(pos), ids=np.array(ids),
                           groups={"trajectory": np.concatenate(trajectory), "time": np.concatenate(time)},
                           meta={"subcarriers": indices.copy(), "training_field": self.subcarriers,
                                 "carrier_hz": 2.462e9, "subcarrier_offsets_hz": indices * _SUBCARRIER_SPACING_HZ,
                                 "sequences": tuple(seqs)})
