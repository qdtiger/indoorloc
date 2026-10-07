"""BLE RSSI Dataset for Indoor Localization and Navigation (UCI 435): 13 iBeacons in Waldo Library."""
from __future__ import annotations

import calendar
import csv

import numpy as np

from ..core import SampleTable
from ._base import Dataset


def cell_to_grid(label: str) -> tuple[float, float]:
    """``"K04"`` -> ``(11.0, 4.0)``: the column letter (A = 1) and the row number of the source's map grid."""
    letter, row = label[:1].upper(), label[1:]
    if not ("A" <= letter <= "Z") or not row.isdigit():
        raise ValueError(f"not a grid cell label: {label!r} (expected a letter and a row number, e.g. 'K04')")
    return float(ord(letter) - ord("A") + 1), float(row)


def _wall_clock_seconds(stamp: str) -> float:
    """``"10-18-2016 11:15:21"`` (month-day-year, 24 h) -> seconds since 1970-01-01 of that wall-clock time."""
    date, clock = stamp.split()
    month, day, year = (int(v) for v in date.split("-"))
    hour, minute, second = (int(v) for v in clock.split(":"))
    return float(calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0)))


class BLERSSIUCI(Dataset):
    """BLE RSSI of 13 iBeacons on the first floor of Waldo Library, Western Michigan University (UCI, 2018).

    An iPhone 6S recorded the RSSI of 13 iBeacons (b3001..b3013) during the library's opening
    hours in 2016: 1,420 scans labelled with the map cell where they were taken (105 cells), and
    5,191 unlabelled scans collected for semi-supervised learning (Mohammadi et al., 2018).

    ``X``      (N, 13) float32 RSSI in dBm; the file's "out of range" value -200 becomes NaN, and so
               do 15 readings of -198/-199 in the labelled file (the same sentinel off by one or
               two: every other reading in that file is between -88 and -55 dBm).
    ``pos``    **grid cells, not metres.** The label ``"K04"`` is column K, row 4 of the map shipped
               with the data (``iBeacon_Layout.jpg``) and becomes ``(11, 4)`` (A = 1). On that map the
               columns A-U run left to right and the rows 1-18 run **top to bottom**, so the frame is
               the map's image frame (y down). The source does not state the cell size, so errors are
               in cells (``meta["pos_units"]``). Use ``cell_to_grid`` to convert labels.
    ``groups`` ``time``: the file's wall-clock timestamp (local time of the library, US Eastern)
               as seconds since 1970-01-01 read as if it were UTC (sortable; not true unix time).
               ``point``: the cell label (``"K04"``), the reference point of the scan (for cell
               classification or splits by position).
    ``floor``/``building``: None (one floor of one building).

    Splits: ``"all"`` (the labelled file) and ``"unlabeled"`` (the unlabelled file: ``pos`` is
    all NaN and ``point`` is listed in ``meta["unknown_groups"]``). There is no official
    train/test split. The literature mostly reports cell-classification accuracy on random
    splits; consecutive scans (every 1-2 s at the same cell) are near duplicates, so a random
    split flatters any method. A split by recording day (``groups["time"] // 86400``) or by
    cell (``groups["point"]``, positions never seen in training) measures generalisation.

    The timestamps are month-day-year (``10-18-2016``), not day-month-year as the UCI page says.

    References
    ----------
    Mohammadi, M., Al-Fuqaha, A., "BLE RSSI Dataset for Indoor localization and Navigation",
    UCI Machine Learning Repository, 2018. https://doi.org/10.24432/C54G80

    Mohammadi, M., Al-Fuqaha, A., Guizani, M., Oh, J.-S., "Semisupervised Deep Reinforcement
    Learning in Support of IoT and Smart City Services", IEEE Internet of Things Journal 5(2),
    624-635, 2018. https://doi.org/10.1109/JIOT.2017.2712560
    """

    name = "ble_rssi_uci"
    urls = ("https://archive.ics.uci.edu/static/public/435/"
            "ble+rssi+dataset+for+indoor+localization+and+navigation.zip",)
    files = {
        "all": ("iBeacon_RSSI_Labeled.csv", "2be36c37b2dd34e1eb59420d358a7cc4078718f6b5d49cbf4598753a2b4fecc9"),
        "unlabeled": ("iBeacon_RSSI_Unlabeled.csv", "265439b23f3118c8b68dcf7068404c59e025ee43bb10bd0c7f4248c07f5f1c7c"),
    }
    split_aliases = {"labeled": "all", "labelled": "all", "unlabelled": "unlabeled"}
    meta = {
        "modality": "ble_rssi",
        "units": "dBm",
        "raw_missing_value": -200,
        "crs": "local",
        "pos_names": ("column", "row"),
        "pos_units": "grid cells of the source map (column letter A=1, row number counted downward); "
                     "cell size not stated",
        "time_units": "s, local wall clock (US Eastern) read as UTC",
        "device": "iPhone 6S",
        "license": "CC BY 4.0",
        "doi": "10.24432/C54G80",
        "citation": "Mohammadi, Al-Fuqaha, Guizani, Oh, Semisupervised Deep Reinforcement Learning in Support "
                    "of IoT and Smart City Services, IEEE Internet of Things Journal 5(2), 2018",
        "url": "https://archive.ics.uci.edu/dataset/435/ble+rssi+dataset+for+indoor+localization+and+navigation",
    }
    feature_names = tuple(f"b{3001 + i}" for i in range(13))
    _missing_below = -150.0  # no BLE receiver reports below about -110 dBm

    def _parse(self, path, split):
        with open(path, newline="", encoding="ascii") as fh:
            header, *rows = list(csv.reader(fh))
        col = {name.strip(): i for i, name in enumerate(header)}
        X = np.array([[row[col[b]] for b in self.feature_names] for row in rows], dtype=np.float32)
        X[X <= self._missing_below] = np.nan  # -200, and a few -198/-199 in the labelled file
        time = np.array([_wall_clock_seconds(row[col["date"]]) for row in rows])
        cells = np.array([row[col["location"]].strip() for row in rows])
        if split == "unlabeled":
            if np.any(cells != "?"):
                raise ValueError(f"{path.name}: expected '?' in every location field of the unlabelled file")
            pos, groups, unknown = np.full((len(rows), 2), np.nan), {"time": time}, ("point",)
        else:
            pos = np.array([cell_to_grid(c) for c in cells])
            groups, unknown = {"time": time, "point": cells}, ()
        ids = np.array([f"{split}-{i:05d}" for i in range(len(rows))])
        return SampleTable(X, pos, groups=groups, ids=ids,
                           meta={"feature_names": self.feature_names, "unknown_groups": unknown})
