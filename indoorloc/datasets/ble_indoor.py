"""BBIL, the BLE Beacon Indoor Localization dataset: BLE beacons carried through two rooms, fixed receivers."""
from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime

import numpy as np

from ..core import SampleTable
from ._base import Dataset


def _utc_seconds(stamp: str) -> float:
    """``"2018-09-18T20:03:15.500000Z"`` -> unix seconds (``fromisoformat`` takes "Z" only from 3.11)."""
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


class BLEIndoor(Dataset):
    """BBIL: RSSI of walking BLE beacons at fixed Raspberry Pi receivers (Kennedy, Spachos, Taylor, 2019).

    Participants carried a Gimbal Series 10 iBeacon (0 dBm, 10 Hz) and walked for about three
    minutes at a time, several times a day from September 2018 to May 2019, while Raspberry Pi 3
    receivers ("edges", 1.6 m high) recorded the beacon's RSSI. Ground truth comes from
    landmark presses in a phone app, linearly interpolated between presses. Every row is one
    0.5 s period of one recording (the authors' ``*_data_wide.csv`` files, read straight from
    the release archive).

    The two experiments are **two different rooms**, not floors of one building, each with
    its own receivers and its own local frame:

    ======== ============ ================= ==================== =====================
    building room         experiment folder size (authors)       receivers (features)
    ======== ============ ================= ==================== =====================
    0        office       experiment1       about 11 m x 25 m    edge_1-3, edge_8-13
    1        lab          experiment2       about 11 m x 7 m     edge_50 ... edge_58
    ======== ============ ================= ==================== =====================

    ``building`` therefore holds the room (names in ``meta["building_names"]``); positions of
    different rooms must never be compared. ``floor`` is None.

    ``X``      (N, n_edges) float32 RSSI in dBm, one column per receiver of the selected rooms
               (``meta["feature_names"]``). The authors' wide files already fill every receiver
               in every period; a receiver missing from a whole recording, or belonging to the
               other room, is NaN. Values are averages over the period (not whole dBm).
    ``pos``    (x, y) in metres, the room's local frame (``realx``, ``realy``).
    ``groups`` ``trajectory`` (recording, e.g. ``"office/2018-09-18T20-03-15-500000_1"``),
               ``device`` (beacon id), ``time`` (unix seconds, UTC).
    ``meta``   ``anchors`` (n_edges, 2): each receiver's position in its own room's frame (from
               ``edges.csv``, which also holds the authors' per-beacon path-loss fits, not loaded);
               ``anchor_height`` (n_edges,) in metres; ``building_names``.

    Splits: the official ``train`` / ``valid`` / ``test`` recordings. The authors ask for results on
    ``test`` without using it for model selection, reported as the mean and the 90th percentile
    of the L2 error over all test rows. Their best results: office (experiment 1) LSTM mean
    1.5399 m, P90 2.3059 m; lab (experiment 2) Elman network mean 0.9620 m, P90 1.3792 m.

    Parameters
    ----------
    room : "all" (default), "office" / "experiment1", "lab" / "experiment2", or a sequence of these.

    References
    ----------
    Kennedy, M., Spachos, P., Taylor, G. W., "BLE beacon indoor localization dataset",
    Scholars Portal Dataverse, V1, 2019. https://doi.org/10.5683/SP2/UTZTFT
    (data and documentation: https://github.com/co60ca/BBIL)
    """

    name = "ble_indoor"
    urls = ("https://github.com/co60ca/BBIL/releases/download/v0.9.0/experiment.zip",)
    _archive = ("experiment.zip", "32fd58a9403013162cb0d44d30a854802773f188d37be491a2f21f100befd173")
    files = {"train": _archive, "valid": _archive, "test": _archive}
    split_aliases = {"validation": "valid", "val": "valid"}
    meta = {
        "modality": "ble_rssi",
        "units": "dBm",
        "crs": "local (one frame per room; see building)",
        "pos_names": ("x", "y"),
        "pos_units": "m",
        "buildings": (0, 1),
        "license": "MIT",
        "doi": "10.5683/SP2/UTZTFT",
        "citation": "Kennedy, Spachos, Taylor, BLE beacon indoor localization dataset, Scholars Portal Dataverse, 2019",
        "url": "https://github.com/co60ca/BBIL",
    }
    rooms = {"office": (0, "experiment1"), "lab": (1, "experiment2")}
    _room_aliases = {"experiment1": "office", "experiment2": "lab"}

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, room="all"):
        super().__init__(root, download=download, verify=verify)
        wanted = [room] if isinstance(room, str) else list(room)
        if "all" in wanted:
            wanted = list(self.rooms)
        wanted = [self._room_aliases.get(str(r).lower(), str(r).lower()) for r in wanted]
        unknown = sorted(set(wanted) - set(self.rooms))
        if unknown or not wanted:
            raise ValueError(f"unknown room {unknown or room!r}; choose from {sorted(self.rooms)}, their "
                             f"experiment folders {sorted(self._room_aliases)}, or 'all'")
        self.room = tuple(r for r in self.rooms if r in wanted)  # fixed order: office, lab

    def _parse(self, path, split):
        with zipfile.ZipFile(path) as archive:
            members = sorted(archive.namelist())
            parts = [self._read_room(archive, members, room, split) for room in self.room]
        # one column per receiver of the selected rooms; the other room's columns stay NaN
        offsets = np.cumsum([0, *(len(p["edges"]) for p in parts)])
        X = np.full((sum(len(p["pos"]) for p in parts), offsets[-1]), np.nan, dtype=np.float32)
        start = 0
        for r, part in enumerate(parts):
            X[start:start + len(part["pos"]), offsets[r]:offsets[r + 1]] = part["X"]
            start += len(part["pos"])
        cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
        edges = [xyz for p in parts for xyz in p["edges"].values()]
        return SampleTable(
            X, cat("pos"), building=cat("building"), ids=cat("ids"),
            groups={"trajectory": cat("trajectory"), "device": cat("device"), "time": cat("time")},
            meta={"feature_names": tuple(name for p in parts for name in p["edges"]),
                  "anchors": np.array([xyz[:2] for xyz in edges]).reshape(-1, 2),
                  "anchor_height": np.array([xyz[2] for xyz in edges]),
                  "buildings": tuple(self.rooms[r][0] for r in self.room),
                  "building_names": {self.rooms[r][0]: r for r in self.room}})

    def _read_room(self, archive: zipfile.ZipFile, members, room: str, split: str) -> dict:
        code, folder = self.rooms[room]
        edges = self._read_edges(archive, f"{folder}/{split}/edges.csv")
        column = {name: j for j, name in enumerate(edges)}
        recordings = [m for m in members if m.startswith(f"{folder}/{split}/") and m.endswith("_data_wide.csv")]
        if not recordings:
            raise ValueError(f"{archive.filename}: no {folder}/{split}/*_data_wide.csv")
        X, pos, trajectory, device, time = [], [], [], [], []
        for member in recordings:
            with archive.open(member) as raw:
                header, *rows = list(csv.reader(io.TextIOWrapper(raw, encoding="ascii")))
            col = {name: i for i, name in enumerate(header)}
            extra = [h for h in header if h.startswith("edge_") and h not in column]
            if extra:
                raise ValueError(f"{member}: receivers {extra} are not in {folder}/{split}/edges.csv")
            present = [h for h in header if h in column]  # parse by name; absent receivers stay NaN
            block = np.full((len(rows), len(edges)), np.nan, dtype=np.float32)
            block[:, [column[h] for h in present]] = np.array([[r[col[h]] for h in present] for r in rows],
                                                               dtype=np.float64).reshape(len(rows), -1)
            X.append(block)
            pos += [(float(r[col["realx"]]), float(r[col["realy"]])) for r in rows]
            stem = member.rsplit("/", 1)[1].removesuffix("_data_wide.csv")
            trajectory += [f"{room}/{stem}"] * len(rows)
            device += [int(r[col["beaconid"]]) for r in rows]
            time += [_utc_seconds(r[col["Datetime"]]) for r in rows]
        n = len(pos)
        return {"X": np.concatenate(X), "pos": np.array(pos).reshape(n, 2), "edges": edges,
                "building": np.full(n, code, dtype=np.int64),
                "ids": np.array([f"{room}-{split}-{i:05d}" for i in range(n)]),
                "trajectory": np.array(trajectory), "device": np.array(device, dtype=np.int64), "time": np.array(time)}

    @staticmethod
    def _read_edges(archive: zipfile.ZipFile, member: str) -> dict[str, tuple[float, float, float]]:
        """Receiver name -> (x, y, z) from ``edges.csv`` (one row per beacon and receiver), by edge id."""
        with archive.open(member) as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="ascii")))
        edges = {}
        for r in rows:
            xyz = (float(r["edge_x"]), float(r["edge_y"]), float(r["edge_z"]))
            if edges.setdefault(int(r["edgenodeid"]), xyz) != xyz:
                raise ValueError(f"{member}: receiver {r['edgenodeid']} has two positions")
        return {f"edge_{e}": edges[e] for e in sorted(edges)}
