"""UJI BLE RSS database: iBeacon fingerprints in a library and an office lab of Universitat Jaume I."""
from __future__ import annotations

import csv

import numpy as np

from ..core import SampleTable
from ._base import Dataset

# Reference-point lists of the authors' train/test configurations (ips/splitTrnTst.m, UJI_BLE_DB.zip).
_LIB_REDUCED_TRAIN_DROP = (2, 8, 14, 20, 29, 35, 41, 47)
_LIB_OUTER_TEST_DROP = (1, 17, 18, 34, 35, 51, 52, 68)
_GEO_FULL_LIMITS = (1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    44, 45, 49, 50, 54, 55, 59, 60, 64, 65, 66, 67, 68)
_GEO_ALL_HORIZ_INNER_MIDDLE = (7, 12, 17, 22, 27, 42, 47, 52, 57, 62)
_GEO_REDUCED_HORIZ_INNER_MIDDLE = (12, 22, 47, 57)
_GEO_LIMITS_REMOVE = (2, 3, 10, 14, 20, 24, 25, 29, 31, 33, 36, 38, 40, 44, 45, 49, 55, 59, 66, 67)
_GEO_TRIANGLES = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 36, 38, 40, 42, 44, 46, 48, 50, 52,
                  54, 56, 58, 60, 62, 64, 66, 68)


def _without(points, drop) -> tuple[int, ...]:
    return tuple(p for p in points if p not in drop)


def _rest(train, n: int = 68) -> tuple[int, ...]:
    return _without(range(1, n + 1), train)


# name -> (train campaign, train points, test campaign, test points); campaign codes as in the ids
PROTOCOLS = {
    "lib": {
        "full_train_full_test": (2, tuple(range(1, 49)), 3, tuple(range(1, 69))),
        "reduced_train_full_test": (2, _without(range(1, 49), _LIB_REDUCED_TRAIN_DROP), 3, tuple(range(1, 69))),
        "full_train_no_outer_test": (2, tuple(range(1, 49)), 3, _without(range(1, 69), _LIB_OUTER_TEST_DROP)),
        "reduced_train_no_outer_test": (2, _without(range(1, 49), _LIB_REDUCED_TRAIN_DROP),
                                        3, _without(range(1, 69), _LIB_OUTER_TEST_DROP)),
    },
    "geo": {
        "full_limits": (1, tuple(sorted(_GEO_FULL_LIMITS + _GEO_ALL_HORIZ_INNER_MIDDLE)),
                        1, _rest(_GEO_FULL_LIMITS + _GEO_ALL_HORIZ_INNER_MIDDLE)),
        "reduced_limits": (1, _without(sorted(_GEO_FULL_LIMITS + _GEO_REDUCED_HORIZ_INNER_MIDDLE), _GEO_LIMITS_REMOVE),
                           1, _rest(_without(_GEO_FULL_LIMITS + _GEO_REDUCED_HORIZ_INNER_MIDDLE, _GEO_LIMITS_REMOVE))),
        "triangles": (1, _GEO_TRIANGLES, 1, _rest(_GEO_TRIANGLES)),
    },
}
DEFAULT_PROTOCOL = {"lib": "full_train_no_outer_test", "geo": "full_limits"}  # the authors' f7_basicPos.m


class IBeaconRSSI(Dataset):
    """UJI BLE RSS database (Mendoza-Silva et al., Data 2019): iBeacon fingerprints in two zones.

    Accent Systems iBKS 105 beacons were measured by smartphones in two zones of Universitat
    Jaume I, each with its own beacons and its own local frame:

    ========= ======= ================================================ ===================
    building  zone    phones, transmit power, campaigns                fingerprints
    ========= ======= ================================================ ===================
    1         geo     Galaxy A5 2017; -4, -12, -20 dBm; campaign 1     2,652 (22 beacons)
    2         lib     Galaxy A5 2017, BQ Aquaris X5 Plus, Galaxy S6;   2,100 (24 beacons)
                      -12 dBm; campaigns 2 (48 points) and 3 (68)
    ========= ======= ================================================ ===================

    ``building`` holds the authors' zone code (geo = 1, the Geotec lab; lib = 2, the library;
    ``meta["building_names"]``). Positions of the two zones are in different frames.

    ``X``      (N, n_beacons) float32 RSS in dBm, one column per beacon of the selected zones
               (``meta["feature_names"]``, the ids of ``data/dep/<zone>.csv``); the file's "not
               detected" value 100 becomes NaN, as does the other zone's beacons. 236 lib and 353 geo
               fingerprints heard no beacon at all (all-NaN rows); they are kept.
    ``pos``    (x, y) metres in the zone's local frame.
    ``groups`` ``point``: the reference position, zone and coordinates as in the file
               (``"lib/29.22,12.91"``), for splits by position. ``device`` (phone: "A5", "BQ", "S6"),
               ``power`` (beacon transmit power, dBm), ``campaign`` and ``point_number``: the authors'
               codes, decoded from the 8-digit fingerprint id (phone, power, campaign, point (3 digits),
               sample (2 digits)). **A point number is not a position:** in every zone and campaign
               the numbers k and n + 1 - k (n = 48 or 68 points) name the same position, measured
               twice, so the 4,752 fingerprints hold 92 positions (geo 34, lib 24 + 34). Splitting by
               ``point_number`` would test at training positions; split by ``point``. Lib campaigns
               2 and 3 share no position. The file repeats 12 fingerprint ids (the BQ phone measured
               lib points 17 and 31 twice), so sample ids are the file rows (``"lib-0792"``).
    ``meta``   ``anchors`` (n_beacons, 2) beacon positions, each in its zone's frame.

    Splits
    ------
    ``"all"``: every fingerprint. ``"train"`` / ``"test"``: the authors' configurations of
    ``ips/splitTrnTst.m`` (``PROTOCOLS``), by default the ones of their example script
    ``f7_basicPos.m``: lib trains on campaign 2 and tests on campaign 3 without its 8 outer points
    (``"full_train_no_outer_test"``); geo trains on the boundary and middle-row points and tests on
    the rest (``"full_limits"``). Both keep every phone and power level: select a device or power
    through ``groups`` to reproduce a single set as in the paper. Every configuration keeps both
    numbers of a position on the same side, so train and test positions are disjoint.

    Parameters
    ----------
    zone : "all" (default), "lib", "geo", or a sequence of these.
    protocol : dict zone -> configuration name for ``"train"`` / ``"test"``; defaults to
        ``DEFAULT_PROTOCOL``.

    References
    ----------
    Mendoza-Silva, G. M., Matey-Sanz, M., Torres-Sospedra, J., Huerta, J., "BLE RSS Measurements
    Dataset for Research on Accurate Indoor Positioning", Data 4(1), 12, 2019.
    https://doi.org/10.3390/data4010012

    Mendoza-Silva, G. M., Matey-Sanz, M., Torres-Sospedra, J., Huerta, J., "BLE RSS measurements
    database and supporting materials", Zenodo, 2018. https://doi.org/10.5281/zenodo.1618692
    """

    name = "ibeacon_rssi"
    urls = ("https://zenodo.org/records/1618692/files/UJI_BLE_DB.zip?download=1",)
    _zone_files = {
        "geo": (("data/rss/geo_rss.csv", "3b7e6cb0e47866e46a134187808bec77219aaf22a0bdd1dd8205ffa2b765febe"),
                ("data/rss/geo_crd.csv", "ca7e326dc10f6cddc38924516c856baff2f4729814c1c6ec7aac6223c253888e"),
                ("data/rss/geo_ids.csv", "88f9f6c14df21a84b9f2283558737bb161cc9d62777f408cdfe276b751b861bd"),
                ("data/dep/geo.csv", "ec02d2e1e9e109fbaa414365fb8b0efecfccafdb086c798e1c42f36f4bf73c38")),
        "lib": (("data/rss/lib_rss.csv", "92f3ed84556facf38a0017cfce9a8c42673075168385dfc48dca10f389640707"),
                ("data/rss/lib_crd.csv", "70c9eee5b5d6bf5f87171c33715df11198e60f0335009e47c4c0a91d77b56e2e"),
                ("data/rss/lib_ids.csv", "6642b5f47ef59539e17969f4a1c37afcd135275141e1b9d60204ac30ac417035"),
                ("data/dep/lib.csv", "fb502b04f5ff20c9973b5fe9da07edaef1e97516082aa8c37e18695371db23ff")),
    }
    files = dict.fromkeys(("all", "train", "test"), _zone_files["geo"] + _zone_files["lib"])
    meta = {
        "modality": "ble_rssi",
        "units": "dBm",
        "raw_missing_value": 100,
        "crs": "local (one frame per zone; see building)",
        "pos_names": ("x", "y"),
        "pos_units": "m",
        "buildings": (1, 2),
        "license": "CC BY 4.0 (data), MIT (scripts)",
        "doi": "10.5281/zenodo.1618692",
        "citation": "Mendoza-Silva, Matey-Sanz, Torres-Sospedra, Huerta, BLE RSS Measurements Dataset for "
                    "Research on Accurate Indoor Positioning, Data 4(1), 12, 2019",
        "url": "https://zenodo.org/records/1618692",
    }
    zones = {"geo": (1, "geotec"), "lib": (2, "library")}  # zone -> (authors' code, name)
    phones = {1: "A5", 2: "BQ", 3: "S6"}  # Galaxy A5 2017, BQ Aquaris X5 Plus, Galaxy S6 (getFilterDefs.m)
    powers = {1: -4, 2: -12, 3: -20}  # power code -> dBm

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, zone="all", protocol=None):
        super().__init__(root, download=download, verify=verify)
        wanted = [zone] if isinstance(zone, str) else list(zone)
        wanted = list(self.zones) if "all" in wanted else [str(z).lower() for z in wanted]
        if not wanted or set(wanted) - set(self.zones):
            raise ValueError(f"unknown zone {zone!r}; choose from {sorted(self.zones)} or 'all'")
        self.zone = tuple(z for z in self.zones if z in wanted)
        self.protocol = {**DEFAULT_PROTOCOL, **(protocol or {})}
        for z, name in self.protocol.items():
            if z not in PROTOCOLS or name not in PROTOCOLS[z]:
                raise ValueError(f"unknown protocol {z!r}: {name!r}; available: "
                                 f"{ {k: sorted(v) for k, v in PROTOCOLS.items()} }")
        entries = tuple(e for z in self.zone for e in self._zone_files[z])
        self.files = dict.fromkeys(type(self).files, entries)  # check and fetch only the selected zones

    def _parse(self, paths, split):
        path_of = {rel: p for (rel, _), p in zip(self._entries(split), paths)}
        parts = [self._read_zone(path_of, z, split) for z in self.zone]
        offsets = np.cumsum([0, *(len(p["beacons"]) for p in parts)])
        X = np.full((sum(len(p["pos"]) for p in parts), offsets[-1]), np.nan, dtype=np.float32)
        start = 0
        for k, part in enumerate(parts):
            X[start:start + len(part["pos"]), offsets[k]:offsets[k + 1]] = part["X"]
            start += len(part["pos"])
        cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
        meta = {"feature_names": tuple(b for p in parts for b in p["beacons"]),
                "anchors": np.concatenate([p["anchors"] for p in parts]),
                "buildings": tuple(self.zones[z][0] for z in self.zone),
                "building_names": {self.zones[z][0]: self.zones[z][1] for z in self.zone}}
        if split != "all":
            meta["protocol"] = {z: self.protocol[z] for z in self.zone}
        return SampleTable(X, cat("pos"), building=cat("building"), ids=cat("ids"), meta=meta,
                           groups={key: cat(key) for key in ("point", "device", "power", "campaign", "point_number")})

    def _read_zone(self, path_of: dict, zone: str, split: str) -> dict:
        rss = np.loadtxt(path_of[f"data/rss/{zone}_rss.csv"], delimiter=",", ndmin=2)
        crd_text = np.loadtxt(path_of[f"data/rss/{zone}_crd.csv"], delimiter=",", ndmin=2, dtype=str)
        crd = crd_text.astype(np.float64)
        fid = np.loadtxt(path_of[f"data/rss/{zone}_ids.csv"], dtype=np.int64, ndmin=1)
        with open(path_of[f"data/dep/{zone}.csv"], newline="", encoding="ascii") as fh:
            deployment = list(csv.DictReader(fh))
        if not len(rss) == len(crd) == len(fid) or rss.shape[1] != len(deployment):
            raise ValueError(f"{zone}: {rss.shape} RSS, {len(crd)} coordinates, {len(fid)} ids, "
                             f"{len(deployment)} beacons do not match")
        phone, power, campaign = fid // 10**7, fid // 10**6 % 10, fid // 10**5 % 10
        point = fid // 10**2 % 1000
        rows = np.arange(len(fid))
        if split != "all":
            train_c, train_p, test_c, test_p = PROTOCOLS[zone][self.protocol[zone]]
            c, p = (train_c, train_p) if split == "train" else (test_c, test_p)
            rows = np.flatnonzero((campaign == c) & np.isin(point, p))
        X = rss[rows].astype(np.float32)
        X[X == self.meta["raw_missing_value"]] = np.nan
        return {"X": X, "pos": crd[rows, :2], "beacons": [d["id"] for d in deployment],
                "anchors": np.array([[float(d["x"]), float(d["y"])] for d in deployment]),
                "building": np.full(len(rows), self.zones[zone][0], dtype=np.int64),
                "ids": np.array([f"{zone}-{r:04d}" for r in rows]),  # row of the file: the file's ids repeat
                "device": np.array([self.phones[int(v)] for v in phone[rows]]),
                "power": np.array([self.powers[int(v)] for v in power[rows]], dtype=np.int64),
                "point": np.array([f"{zone}/{x},{y}" for x, y in crd_text[rows, :2]]),
                "campaign": campaign[rows], "point_number": point[rows]}
