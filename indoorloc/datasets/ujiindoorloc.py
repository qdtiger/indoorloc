"""UJIIndoorLoc: WiFi RSSI fingerprints from 3 buildings of Universitat Jaume I."""
from __future__ import annotations

import numpy as np

from ..core import SampleTable
from ._base import Dataset


class UJIIndoorLoc(Dataset):
    """UJIIndoorLoc (Torres-Sospedra et al., IPIN 2014), official train/validation files.

    ``X`` is (N, 520) float32 RSSI in dBm; the file's "not detected" value 100 becomes NaN.
    ``pos`` holds the LONGITUDE/LATITUDE columns unchanged. Despite their names these are
    Web Mercator (EPSG:3857) easting/northing in metres, not ground metres: at the campus
    (39.99 N) one Mercator metre is ``meta["ground_scale"]`` = cos(lat) = 0.7661 ground
    metres (spherical approximation; exact ellipsoidal scales differ by about 0.2 %).
    Errors are reported in EPSG:3857 metres, as in the literature.
    Groups: user, device (phone), time (unix s), space, relative_position. The validation
    file stores 0 for the unknown user/space/relative_position: those columns are left out
    of the test split and listed in ``meta["unknown_groups"]`` (no sentinels, rule 5.4).
    """

    name = "ujiindoorloc"
    urls = ("https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip",)
    files = {
        "train": ("trainingData.csv", "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168"),
        "test": ("validationData.csv", "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0"),
    }
    split_aliases = {"validation": "test", "val": "test"}
    meta = {
        "modality": "wifi_rssi",
        "units": "dBm",
        "raw_missing_value": 100,
        "crs": "EPSG:3857",
        "pos_names": ("x", "y"),
        "pos_units": "m (Web Mercator, not ground metres)",
        "ground_scale": 0.7661271545776605,  # cos(39.99263 deg), from the training-set mean northing
        "floors": (0, 1, 2, 3, 4),
        "buildings": (0, 1, 2),
        "doi": "10.24432/C5MS59",
        "license": "CC BY 4.0",
        "citation": "Torres-Sospedra et al., UJIIndoorLoc, IPIN 2014",
    }
    feature_names = tuple(f"WAP{i:03d}" for i in range(1, 521))
    _groups = {"user": "USERID", "device": "PHONEID", "time": "TIMESTAMP",
               "space": "SPACEID", "relative_position": "RELATIVEPOSITION"}
    _unknown_groups = {"test": ("user", "space", "relative_position")}  # all 0 in validationData.csv

    def _parse(self, path, split):
        with open(path, encoding="ascii") as fh:
            col = {name: i for i, name in enumerate(fh.readline().strip().split(","))}
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        X = raw[:, [col[a] for a in self.feature_names]].astype(np.float32)  # columns by name
        X[X == self.meta["raw_missing_value"]] = np.nan
        pos = raw[:, [col["LONGITUDE"], col["LATITUDE"]]]
        unknown = self._unknown_groups.get(split, ())
        for key in unknown:
            if np.any(raw[:, col[self._groups[key]]] != 0):
                raise ValueError(f"{path.name}: {self._groups[key]} holds values; it was declared unknown")
        groups = {key: raw[:, col[c]].astype(np.int64) for key, c in self._groups.items() if key not in unknown}
        ids = np.array([f"{split}-{i:05d}" for i in range(len(raw))])
        return SampleTable(X, pos, raw[:, col["FLOOR"]], raw[:, col["BUILDINGID"]], groups, ids,
                           meta={"feature_names": self.feature_names, "unknown_groups": unknown})
