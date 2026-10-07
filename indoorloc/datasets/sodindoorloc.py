"""SODIndoorLoc: WiFi RSSI fingerprints from three buildings in three Chinese cities (Bi et al., 2022)."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import numpy as np

from ..core import SampleTable
from ._base import Dataset

BUILDINGS = ("CETC331", "HCXY", "SYL")  # BuildingID 1, 2, 3 in the files

# Files are fetched from a pinned commit, so the checksums below can never drift.
_COMMIT = "bfe8f63f55957cc25b23ba91c408ae345a3fb0b7"
_MIRRORS = ("https://raw.githubusercontent.com/bijingxue/SODIndoorLoc/{commit}/{path}",
            "https://raw.githubusercontent.com/renwudao24/SODIndoorLoc/{commit}/{path}")  # the repo's former name
_SHA256 = {  # computed from the files downloaded at _COMMIT (their git blob ids match the repository tree)
    "CETC331/Training_CETC331.csv": "c5975a9e2c8fe03e0ce8ec11175d005032867f6267747dfeae7f2ca1720a143e",
    "CETC331/Testing_CETC331.csv": "3832048128c8f772438020ee38c38e2b4b0bf4d8591c3fa32d5322d390c86388",
    "HCXY/Training_HCXY_All_30.csv": "82c8ad73a3827670b7a1f6abc448e2a7a5a789590cc1f2a310a23a2feecb3b8b",
    "HCXY/Training_HCXY_All_Avg.csv": "fecf7fa93d466940123ec3d443fb35a2790d9a4d15896ac534a81af8a38c48a5",
    "HCXY/Training_HCXY_AP_30.csv": "2d8cc427525598ba342d58d7d9f88a1b68fddc6d0df77a37e39138e3b6f99c59",
    "HCXY/Training_HCXY_AP_Avg.csv": "405b3b657ac5d619310a2dd1dc0af4ff2a4a881052f6ff8cdeefcc1b288dfac5",
    "HCXY/Testing_HCXY_All.csv": "76cbd00ce1103f05cac277729e3382c63803613d40846ad8c2202d6934f7454b",
    "HCXY/Testing_HCXY_AP.csv": "c86f02ae7b436da338bdb9b10ce7702b71e0449fbcc6696c709716ed14142ee4",
    "SYL/Training_SYL_All_30.csv": "5a6c238eb75cf000f5e5aaa17a5c39b1f5ffa9a23bd80e5c38cf9efc8ba9e4dc",
    "SYL/Training_SYL_All_Avg.csv": "df807a9030182bfdaa0e76024a97eb2db97e764222387689d3fb84c062e06912",
    "SYL/Training_SYL_AP_30.csv": "af77fc212b9d6432b11c799cd3bcaada53e1b0cae27435b7ceadb941cd586b6c",
    "SYL/Training_SYL_AP_Avg.csv": "2b7f781b4fc9b5fda404c31ce5d4987f745a6b2bb36d37763e22e1fc8be5a8a9",
    "SYL/Testing_SYL_All.csv": "46c71543530e657d2d88175702b796c359ab10fb0ea6bab12f9a1762fbdcae42",
    "SYL/Testing_SYL_AP.csv": "f39fa92d6d85c6803c5748bd1677f68ee632d4d9770d4f23211c1e117785082b",
}
# How the "Avg" sheets were made, rebuilt cell for cell from the 30-scan sheets (tests/datasets):
#   Training_HCXY_All_Avg = round(mean of the detected scans), +100 if no scan detected the MAC;
#   the other three       = round(mean of the 30 scans with -105 dBm for each undetected scan),
# rounding half away from zero. The latter store -105 where the MAC was (almost) never heard.
_EXTRA_MISSING = {"HCXY/Training_HCXY_AP_Avg.csv": -105.0, "SYL/Training_SYL_All_Avg.csv": -105.0,
                  "SYL/Training_SYL_AP_Avg.csv": -105.0}
# Erratum: in this sheet BuildingID counts 3, 4, ..., 298 down the rows (a spreadsheet fill-series slip);
# every other column equals Training_SYL_AP_Avg.csv row for row, so the building is SYL (3) throughout.
_BAD_BUILDING_ID = {"SYL/Training_SYL_All_Avg.csv"}
_LABELS = ("ECoord", "NCoord", "FloorID", "BuildingID", "SceneID", "UserID", "PhoneID", "SampleTimes")
_MACS = ("all", "preinstalled")


def _sheet(building: str, split: str, macs: str = "all", averaged: bool = False) -> str:
    """Relative path of the sheet holding ``split`` of ``building`` for the chosen variant."""
    if building == "CETC331":  # one sheet per split: its 52 MACs are exactly the 26 dual-band pre-installed APs,
        return f"CETC331/{'Training' if split == 'train' else 'Testing'}_CETC331.csv"  # one scan per RP
    kind = "All" if macs == "all" else "AP"
    if split == "test":
        return f"{building}/Testing_{building}_{kind}.csv"
    return f"{building}/Training_{building}_{kind}_{'Avg' if averaged else '30'}.csv"


def _entries(buildings, split: str, macs: str = "all", averaged: bool = False) -> tuple:
    return tuple((path, _SHA256[path]) for path in (_sheet(b, split, macs, averaged) for b in buildings))


class SODIndoorLoc(Dataset):
    """SODIndoorLoc (Bi et al., Satellite Navigation 2022): WiFi RSSI in three buildings, official train/test.

    Three buildings in three different cities: CETC331 (floors 1-3, 52 MACs, one scan per
    reference point), HCXY (floor 4, 347 MACs, 6 phones) and SYL (floor 4, 363 MACs, 2 phones);
    1630 reference points with 21 205 training scans and 272 test points with 2720 test scans
    in total. MACs of different buildings are distinct, so with several buildings ``X`` is
    block-structured: each building contributes its own columns (``meta["feature_names"]``
    = ``"<building>/MAC<n>"``) and is NaN in the columns of the others.

    ``X`` is (N, n_mac) float32 RSSI in dBm; the files' "not detected" value +100 becomes NaN.
    ``pos`` holds (ECoord, NCoord) in metres, unchanged. **Each building has its own local
    east/north frame** (``meta["crs"] = "local-per-building"``): positions of different
    buildings are not comparable, so a position error means something only when the building
    is right. ``floor`` and ``building`` keep the files' FloorID (1-4) and BuildingID (1-3).
    Groups: ``device`` (PhoneID 1-9, unique across buildings), ``user`` (UserID 1-10),
    ``scene`` (1 corridor, 2 office room, 3 meeting room) and ``scan_index`` (SampleTimes: the
    repeat number of the scan at its point, 1-30 in training, 1-10 in testing).

    Parameters
    ----------
    building : None (all three), a building name ("CETC331", "HCXY", "SYL", any case) or
        BuildingID (1-3), or a sequence of them. Columns and rows always follow ``BUILDINGS``.
    macs : "all" (every detected MAC, the default) or "preinstalled" (only the MACs of the
        105 pre-installed APs: 52 in CETC331, 56 in HCXY, 46 in SYL).
    averaged : False (30 scans per training point, the default) or True (the "Avg" training
        sheets: one row per point, same order as the 30-scan sheet). The two averaging rules
        were rebuilt exactly (every cell) from the 30-scan sheets. The HCXY "all" sheet is the
        mean of the scans that detected the MAC and keeps +100 for a MAC never detected. The
        SYL sheets and the HCXY "preinstalled" sheet substitute -105 dBm for every undetected
        scan before averaging, so a rarely heard MAC is biased low (heard once at -80 dBm it
        averages to -104), and a result of -105 (117 837 cells never heard, 4 heard once in
        30 scans) becomes NaN. Both round to whole dBm, half away from zero. CETC331 has a
        single training sheet with one scan per point, used for every variant. The
        BuildingID column of ``Training_SYL_All_Avg.csv`` is corrupt (it counts 3..298 down
        the rows); the loader sets it to 3 (SYL), which the other columns confirm.
        ``meta["raw_missing_value"]`` lists the markers of the loaded sheets.

    References
    ----------
    Bi, J., Wang, Y., Yu, B., Cao, H., Shi, T., Huang, L., "Supplementary open dataset for WiFi
    indoor localization based on received signal strength", Satellite Navigation 3, 25, 2022.
    https://doi.org/10.1186/s43020-022-00086-y. Data: https://github.com/bijingxue/SODIndoorLoc
    """

    name = "sodindoorloc"
    urls = {path: tuple(m.format(commit=_COMMIT, path=path) for m in _MIRRORS) for path in _SHA256}
    files = {split: _entries(BUILDINGS, split) for split in ("train", "test")}  # default variant
    # No 'validation' alias: the source has no validation file (carve one from train, e.g. kfold on groups).
    meta = {
        "modality": "wifi_rssi",
        "units": "dBm",
        "raw_missing_value": 100,
        "crs": "local-per-building",
        "pos_names": ("x", "y"),  # ECoord (east), NCoord (north)
        "pos_units": "m",
        "floors": (1, 2, 3, 4),
        "buildings": (1, 2, 3),
        "building_names": {1: "CETC331", 2: "HCXY", 3: "SYL"},
        "scene_names": {1: "corridor", 2: "office room", 3: "meeting room"},
        "license": "not stated in the repository; cite the paper",
        "doi": "10.1186/s43020-022-00086-y",
        "citation": "Bi et al., Supplementary open dataset for WiFi indoor localization based on received "
                    "signal strength, Satellite Navigation 3:25, 2022",
        "url": "https://github.com/bijingxue/SODIndoorLoc",
    }

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, building=None,
                 macs: str = "all", averaged: bool = False):
        super().__init__(root, download=download, verify=verify)
        if building is None:
            building = BUILDINGS
        error = ValueError(f"building must be None, a name from {BUILDINGS} (any case), a BuildingID 1-3, "
                           f"or a sequence of them; got {building!r}")
        try:
            wanted = [building] if isinstance(building, (str, int, np.integer)) else list(building)
        except TypeError:
            raise error from None
        by_id = lambda b: isinstance(b, (int, np.integer)) and not isinstance(b, bool) and 1 <= b <= 3  # noqa: E731
        wanted = {BUILDINGS[b - 1] if by_id(b) else str(b).upper() for b in wanted}  # name (any case) or BuildingID
        if not wanted or wanted - set(BUILDINGS):
            raise error
        if macs not in _MACS:
            raise ValueError(f"macs must be one of {_MACS}, got {macs!r}")
        self.selected = tuple(b for b in BUILDINGS if b in wanted)
        self.macs = macs
        self.averaged = bool(averaged)
        self.files = {split: _entries(self.selected, split, macs, self.averaged) for split in ("train", "test")}

    def _parse(self, paths, split):
        paths = [paths] if isinstance(paths, Path) else list(paths)
        sheets = [(b, rel) for b, (rel, _) in zip(self.selected, self.files[split])]
        blocks = []  # one dict per building, in BUILDINGS order
        for (building, rel), path in zip(sheets, paths):
            with open(path, encoding="utf-8-sig") as fh:
                header = fh.readline().strip().split(",")
            col = {name: i for i, name in enumerate(header)}
            missing = [c for c in _LABELS if c not in col]
            if missing:
                raise ValueError(f"{path.name}: columns {missing} not found")
            macs = sorted((c for c in header if c.startswith("MAC")), key=lambda c: int(c[3:]))
            raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            rssi = raw[:, [col[m] for m in macs]].astype(np.float32)  # columns by name, in MAC order
            rssi[rssi == 100] = np.nan
            if rel in _EXTRA_MISSING:
                rssi[rssi == _EXTRA_MISSING[rel]] = np.nan
            if np.any(rssi > 0):
                raise ValueError(f"{path.name}: positive RSSI other than the +100 'not detected' marker")
            building_id = BUILDINGS.index(building) + 1
            if rel not in _BAD_BUILDING_ID and np.any(raw[:, col["BuildingID"]] != building_id):
                raise ValueError(f"{path.name}: BuildingID does not match building {building}")
            raw[:, col["BuildingID"]] = building_id
            stem = PurePosixPath(rel).stem
            blocks.append({"names": [f"{building}/{m}" for m in macs], "rssi": rssi,
                           "ids": np.array([f"{stem}-{i:05d}" for i in range(len(raw))]),
                           **{c: raw[:, col[c]] for c in _LABELS}})

        names = [name for b in blocks for name in b["names"]]
        X = np.full((sum(len(b["rssi"]) for b in blocks), len(names)), np.nan, dtype=np.float32)
        row = column = 0
        for b in blocks:  # block-diagonal: a building's scans only fill its own MAC columns
            n, f = b["rssi"].shape
            X[row:row + n, column:column + f] = b["rssi"]
            row, column = row + n, column + f
        cat = lambda key: np.concatenate([b[key] for b in blocks])  # noqa: E731
        groups = {"device": cat("PhoneID").astype(np.int64), "user": cat("UserID").astype(np.int64),
                  "scene": cat("SceneID").astype(np.int64), "scan_index": cat("SampleTimes").astype(np.int64)}
        pos = np.column_stack([cat("ECoord"), cat("NCoord")])
        markers = sorted({100} | {int(_EXTRA_MISSING[rel]) for _, rel in sheets if rel in _EXTRA_MISSING})
        return SampleTable(X, pos, cat("FloorID"), cat("BuildingID"), groups, cat("ids"),
                           meta={"feature_names": tuple(names), "selected_buildings": self.selected,
                                 "macs": self.macs, "averaged": self.averaged,
                                 "raw_missing_value": markers[0] if len(markers) == 1 else tuple(markers)})

