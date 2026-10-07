"""Indoor Location Competition 2.0 sample data: smartphone traces with IMU, WiFi, iBeacon and floor plans.

Reads the sample traces that Microsoft Research and XYZ10 published with the competition's code
(two shopping malls, 14 floors, 1,095 traces) into SampleTables of one modality: WiFi scans,
iBeacon readings, IMU time series or the surveyor's waypoints. ``read_trace`` and
``read_floor_plan`` parse single files for users who want the raw records.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path

import numpy as np

from ..core import SampleTable
from ._base import Dataset

_COMMIT = "8ab177cd7be2700442714b7d13fa452bedc685f5"  # 2021-08-12, the repository's latest commit
_REPO = f"https://raw.githubusercontent.com/location-competition/indoor-location-competition-20/{_COMMIT}/"

#: Column names of an IMU table, in order (``meta["channels"]``).
IMU_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z",
                "rv_x", "rv_y", "rv_z", "rv_w")
IMU_UNITS = ("m/s^2",) * 3 + ("rad/s",) * 3 + ("uT",) * 3 + ("1",) * 4
_SENSORS = {"acc": "TYPE_ACCELEROMETER", "gyr": "TYPE_GYROSCOPE", "mag": "TYPE_MAGNETIC_FIELD",
            "rv": "TYPE_ROTATION_VECTOR"}
# record kind -> (line type, names of the value fields read, in file order)
_RECORDS = {
    **{k: (v, ("x", "y", "z")) for k, v in _SENSORS.items()},
    "waypoint": ("TYPE_WAYPOINT", ("x", "y")),
    "wifi": ("TYPE_WIFI", ("ssid", "bssid", "rssi", "freq", "seen")),
    "beacon": ("TYPE_BEACON", ("uuid", "major", "minor", "tx_power", "rssi", "distance", "mac")),
}
_INT_FIELDS = {"freq", "seen"}
_STR_FIELDS = {"ssid", "bssid", "uuid", "major", "minor", "mac"}
_MODALITIES = {"wifi": "wifi_rssi", "wifi_rssi": "wifi_rssi", "ble": "ble_rssi", "ble_rssi": "ble_rssi",
               "beacon": "ble_rssi", "imu": "imu", "waypoints": "waypoints", "waypoint": "waypoints"}
_OUTLINE, _UNIT = 0, 1  # wall_type codes: the floor's outline, the outline of a unit (shop, room, ...)


def floor_level(name: str) -> int:
    """Floor folder name -> integer level: ``"F1"`` -> 1, ``"F8"`` -> 8, ``"B1"`` -> -1.

    ``F<n>`` (or ``<n>F``) is the n-th storey above ground and ``B<n>`` (or ``<n>B``) the
    n-th basement, so there is no level 0; labels that number the ground floor 0 need an
    explicit conversion.
    """
    match = re.fullmatch(r"([FB])(\d+)|(\d+)([FB])", str(name).strip().upper())
    if match is None:
        raise ValueError(f"unknown floor name {name!r}; expected F<n> or B<n> (e.g. 'F1', 'B1')")
    letter, digits = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(3))
    level = int(digits)
    if level == 0:
        raise ValueError(f"floor name {name!r} has level 0; the dataset counts storeys from 1")
    return level if letter == "F" else -level


def read_trace(path, kinds=tuple(_RECORDS)) -> dict:
    """Parse one trace file (``data/<site>/<floor>/path_data_files/<id>.txt``) into numpy arrays.

    Every record kind maps to a dict of equal-length arrays; ``t`` is the line's first column,
    Unix time in milliseconds (int64; sensor event time for sensors, system time for WiFi and
    iBeacon scans, per the repository's README):

    ``acc``, ``gyr``, ``mag``, ``rv``  ``t`` and ``xyz`` (n, 3) float64: Android's
        ``SensorEvent.values[0:3]`` of TYPE_ACCELEROMETER (m/s^2), TYPE_GYROSCOPE (rad/s),
        TYPE_MAGNETIC_FIELD (uT) and TYPE_ROTATION_VECTOR (the x, y, z part of the
        orientation quaternion).
    ``waypoint``  ``t`` and ``xy`` (n, 2) float64, metres in the floor map's frame.
    ``wifi``      ``t``, ``ssid``, ``bssid`` (str), ``rssi`` (float64 dBm), ``freq`` (int64 MHz),
                  ``seen`` (int64 ms: the access point's last-seen timestamp).
    ``beacon``    ``t``, ``uuid``, ``major``, ``minor``, ``mac`` (str), ``tx_power``, ``rssi``
                  (float64 dBm) and ``distance`` (float64 m, the recording app's estimate from
                  ``tx_power`` and ``rssi`` with the README's empirical formula).

    plus ``header``: the ``key:value`` pairs of the ``#`` lines (``SiteID``, ``FloorName``,
    ``Brand``, ``Model``, ``startTime``, ...; the first occurrence of each key). Only the
    ``kinds`` asked for are parsed (one kind name or a sequence of them). Line types the README
    does not document (``TYPE_BLUE``, ``TYPE_BLU4``, ``TYPE_DIST1``, ...) and the uncalibrated
    sensors are not read. A line of a parsed kind with too few fields or a non-numeric
    timestamp or value raises ``ValueError``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    header: dict = {}
    for line in re.findall(r"^#[^\n]*", text, re.M):
        for pair in line.rstrip("\r").split("\t")[1:]:
            key, sep, value = pair.partition(":")
            if sep and key and key not in header:
                header[key] = value
    out = {"header": header}
    text = "\n" + text + "\n"  # every line between two newlines: patterns start with a literal (fast)
    for kind in (kinds,) if isinstance(kinds, str) else kinds:
        if kind not in _RECORDS:
            raise ValueError(f"unknown record kind {kind!r}; choose from {tuple(_RECORDS)}")
        type_name, fields = _RECORDS[kind]
        pattern = rf"\n(-?\d+)\t{type_name}" + r"\t([^\t\r\n]*)" * len(fields)
        rows = re.findall(pattern, text)
        # every line whose second column is the type, whatever its first column and field count
        n_lines = len(re.findall(rf"\n[^\t\n]*\t{type_name}(?=[\t\r\n])", text))
        if len(rows) != n_lines:
            raise ValueError(f"{path.name}: {n_lines - len(rows)} of {n_lines} {type_name} lines are malformed")
        cols = list(zip(*rows)) if rows else [()] * (len(fields) + 1)
        try:
            record = {"t": np.array(cols[0], dtype=np.int64)}
            values = dict(zip(fields, cols[1:]))
            if kind in _SENSORS:
                record["xyz"] = np.array([values[a] for a in "xyz"], dtype=np.float64).T.reshape(-1, 3)
            elif kind == "waypoint":
                record["xy"] = np.array([values["x"], values["y"]], dtype=np.float64).T.reshape(-1, 2)
            else:
                for name, col in values.items():
                    dtype = str if name in _STR_FIELDS else np.int64 if name in _INT_FIELDS else np.float64
                    record[name] = np.array(col, dtype=dtype)
        except ValueError as err:
            raise ValueError(f"{path.name}: a {type_name} line has a non-numeric value ({err})") from None
        out[kind] = record
    return out


def _polygons(geometry) -> list:
    """The rings of a GeoJSON Polygon or MultiPolygon, as (K, 2) arrays."""
    if geometry["type"] == "Polygon":
        return [np.asarray(ring, dtype=np.float64) for ring in geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return [np.asarray(ring, dtype=np.float64) for poly in geometry["coordinates"] for ring in poly]
    return []


def _segments(ring: np.ndarray) -> np.ndarray:
    """(K, 4) wall segments of a ring (closed back to its first vertex), zero-length edges dropped."""
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[:1]])
    seg = np.hstack([ring[:-1], ring[1:]])
    return seg[np.any(seg[:, :2] != seg[:, 2:], axis=1)]


def read_floor_plan(geojson, floor_info, level: int = 0) -> dict:
    """``meta["floor_plan"]`` arrays of one floor from ``geojson_map.json`` and ``floor_info.json``.

    The GeoJSON holds longitude/latitude polygons: the floor's outline (its first feature, a
    MultiPolygon on every floor) and one polygon per unit (shops, rooms, ...); ``floor_info.json``
    gives the map's ``width`` and ``height`` in metres. The repository does not document how the
    waypoint frame relates to the GeoJSON; this function assumes the frame of the raster floor
    plan: origin at the south-west corner of the outline's bounding box, x east, y north, the box
    spanning ``[0, width] x [0, height]``, and maps every vertex linearly from the box. Evidence:
    converted with an equirectangular projection (R = 6,378,137 m), the outline boxes of all 14
    floors measure the stated width and height to within 1.2 mm; the repository's own PDR
    (``compute_f.compute_rel_positions``) steps ``L sin(azimuth)`` along x and ``L cos(azimuth)``
    along y, i.e. x east and y north; and few waypoint-to-waypoint legs cross a unit outline in
    this frame (15 of 855 on site1/F1, 7 of 188 on site2/F4) against 179-330 and 83-108 when the
    frame is mirrored in x, in y or in both (``tests/apps/test_apps_realdata.py``).

    Returns ``walls`` (W, 4) float64 ``[x0, y0, x1, y1]`` metres (every polygon edge; a door is
    not a gap here, since the GeoJSON draws closed units), ``wall_floor`` (W,) int64 = ``level``,
    ``wall_type`` (W,) int64 (0 = floor outline, 1 = unit outline; ``materials`` names them),
    ``materials``, ``bounds`` ``(0, 0, width, height)`` and ``origin_lonlat`` (2,) float64, the
    frame's origin (the outline box's south-west corner) as GeoJSON longitude and latitude in
    degrees (the repository does not state the datum). ``apps.maps.FloorMap.from_dict`` reads
    this dict directly (it ignores ``origin_lonlat``).
    """
    def load(source):
        if isinstance(source, dict):
            return source
        return json.loads(Path(source).read_text(encoding="utf-8"))

    info = load(floor_info)["map_info"]
    width, height = float(info["width"]), float(info["height"])
    features = load(geojson)["features"]
    if not features or not _polygons(features[0]["geometry"]):
        raise ValueError("the GeoJSON's first feature must be the floor outline (a Polygon or MultiPolygon)")
    outline = _polygons(features[0]["geometry"])
    lo = np.min([r[:, :2].min(axis=0) for r in outline], axis=0)
    hi = np.max([r[:, :2].max(axis=0) for r in outline], axis=0)
    scale = np.array([width, height]) / (hi - lo)
    walls, kinds = [], []
    for i, feature in enumerate(features):
        for ring in _polygons(feature.get("geometry") or {"type": None}):
            if len(ring) >= 2:
                seg = _segments((ring[:, :2] - lo) * scale)
                walls.append(seg)
                kinds.append(np.full(len(seg), _OUTLINE if i == 0 else _UNIT, dtype=np.int64))
    walls = np.concatenate(walls) if walls else np.zeros((0, 4))
    return {"walls": walls, "wall_floor": np.full(len(walls), int(level), dtype=np.int64),
            "wall_type": np.concatenate(kinds) if kinds else np.zeros(0, np.int64),
            "materials": ("floor outline", "unit outline"),
            "bounds": np.array([0.0, 0.0, width, height]), "origin_lonlat": lo.astype(np.float64)}


def _merge_plans(plans: list) -> dict:
    """One ``floor_plan`` dict for several floors (walls stacked, bounds = their union); the
    per-floor ``origin_lonlat`` goes to the table's ``meta["frame_origin_lonlat"]`` instead."""
    plans = [{k: v for k, v in p.items() if k != "origin_lonlat"} for p in plans]
    if len(plans) == 1:
        return plans[0]
    bounds = np.array([p["bounds"] for p in plans])
    return {"walls": np.concatenate([p["walls"] for p in plans]),
            "wall_floor": np.concatenate([p["wall_floor"] for p in plans]),
            "wall_type": np.concatenate([p["wall_type"] for p in plans]),
            "materials": plans[0]["materials"],
            "bounds": np.array([*bounds[:, :2].min(axis=0), *bounds[:, 2:].max(axis=0)])}


def _positions(t_ms: np.ndarray, wp_t: np.ndarray, wp: np.ndarray):
    """Waypoints linearly interpolated at ``t_ms`` and a mask of the times inside their span."""
    if len(wp_t) == 0:
        return np.full((len(t_ms), 2), np.nan), np.zeros(len(t_ms), dtype=bool)
    order = np.argsort(wp_t, kind="stable")
    wp_t, wp = wp_t[order].astype(np.float64), wp[order]
    t = t_ms.astype(np.float64)
    pos = np.column_stack([np.interp(t, wp_t, wp[:, 0]), np.interp(t, wp_t, wp[:, 1])])
    return pos, (t >= wp_t[0]) & (t <= wp_t[-1])


def _majority(col: np.ndarray, label: np.ndarray, n_columns: int) -> tuple[str, ...]:
    """The most frequent ``label`` of each column index (ties: the alphabetically first)."""
    if n_columns == 0:
        return ()
    names, code = np.unique(label, return_inverse=True)
    pairs, counts = np.unique(col.astype(np.int64) * len(names) + code, return_counts=True)
    pc, pl = pairs // len(names), pairs % len(names)
    order = np.lexsort((pl, -counts, pc))  # by column, most frequent first, then by name
    head = np.ones(len(order), dtype=bool)
    head[1:] = pc[order][1:] != pc[order][:-1]
    return tuple(names[pl[order][head]].tolist())


def _align(t_ms: np.ndarray, record: dict) -> np.ndarray:
    """A sensor's (n, 3) values on the accelerometer clock ``t_ms``.

    The same timestamps (every trace of the repository): the values as they are. Otherwise
    linear interpolation in time inside the sensor's span and NaN outside it (``apps.pdr.imu_arrays``
    fills such gaps); NaN throughout for a sensor without samples.
    """
    t, xyz = record["t"], record["xyz"]
    if len(t) == len(t_ms) and np.array_equal(t, t_ms):
        return xyz
    out = np.full((len(t_ms), 3), np.nan)
    if len(t) == 0:
        return out
    order = np.argsort(t, kind="stable")
    t, xyz = t[order].astype(np.float64), xyz[order]
    q = t_ms.astype(np.float64)
    inside = (q >= t[0]) & (q <= t[-1])
    for j in range(3):
        out[inside, j] = np.interp(q[inside], t, xyz[:, j])
    return out


class ILC2020(Dataset):
    """Indoor Location Competition 2.0 sample traces (Microsoft Research and XYZ10; Hu et al., MobiCom 2023).

    Site surveyors walked paths through two shopping malls in Hangzhou (the trace headers name
    site1 杭州西溪银泰城 and site2 杭州大悦城) holding an Android phone flat in front of the body, while an
    app logged the accelerometer, gyroscope, magnetometer and rotation vector (50 Hz in every
    trace), WiFi scans and iBeacon advertisements, and the surveyor labelled their position on
    the floor map (``TYPE_WAYPOINT``, metres). Phones: OPPO PBCM10 (1,080 traces) and HUAWEI
    LYA-AL00 (15 traces on site1 F2-F4, which also log a barometer that is not read). The
    competition's GitHub repository ships these sample traces with a GeoJSON floor plan per
    floor; the competition's full data (hundreds of buildings, 60 GB according to the authors'
    Zenodo record) is on Kaggle and is not read by this class.

    ======= ======================================================= =========
    site    floors (traces)                                          traces
    ======= ======================================================= =========
    site1   B1 (160), F1 (120), F2 (123), F3 (117), F4 (122)         642
    site2   B1 (70), F1 (99), F2 (45), F3 (40), F4 (27), F5 (44),    453
            F6 (67), F7 (33), F8 (28)
    ======= ======================================================= =========

    One table holds one modality (``modality=``), every row of the selected traces:

    ``"wifi"``       one row per WiFi scan (the lines that share a system timestamp). ``X`` (N, n_ap)
                     float32 dBm, NaN where an access point was not in the scan; ``meta["feature_names"]``
                     = BSSIDs, sorted, only those heard in the returned rows; ``meta["ssids"]`` = the
                     most frequent SSID of each column. Android reports cached results of earlier scans
                     with an older last-seen time; entries older than ``wifi_max_age`` seconds at the
                     scan are dropped (see below). A BSSID listed twice in one scan (the same access
                     point on two channels) keeps its most recently seen entry (ties: the stronger).
    ``"ble"``        one row per iBeacon timestamp (each advertisement has its own, so a row holds one
                     beacon), or with ``ble_window=w`` one row per consecutive ``w``-second window of a
                     trace (the mean dBm of each beacon in it; the row's time is the mean time of its
                     readings). Columns are beacon MAC addresses: on site1/F1, 242 transmitters share
                     the iBeacon id ``9195B3AD-A9D0-4500-85FF-9FB0F65A5201_0_0`` (UUID, major, minor)
                     and 34 share ``FDA50693-A4E2-4FB1-AFCF-C6EB07647825_10073_61418``, so the
                     repository's own ``uuid_major_minor`` key would merge them; ``meta["ibeacon_ids"]``
                     gives each column's. Several readings of one beacon in a row are averaged.
                     Only the documented ``TYPE_BEACON`` (iBeacon) lines are read, not the undocumented
                     ``TYPE_BLUE`` lines (a MAC address and an RSSI, 17-21 per second on site1/F1 and
                     site2/F4 against 5.4 and 0.2 iBeacon readings per second).
    ``"imu"``        one row per accelerometer sample. ``X`` (N, 13) float32 with ``meta["channels"]``
                     = ``IMU_CHANNELS``: ``acc_x/y/z`` (m/s^2), ``gyr_x/y/z`` (rad/s), ``mag_x/y/z`` (uT),
                     calibrated, in Android's device frame (x right, y up the screen, z out of it: up
                     while the phone is held flat), and the rotation vector ``rv_x/y/z/w`` (the
                     device's orientation quaternion in Android's East-North-Up world frame). The
                     file stores x, y, z; ``rv_w`` = sqrt(max(0, 1 - x^2 - y^2 - z^2)) as in Android's
                     ``SensorManager.getQuaternionFromVector``.
                     Every trace logs the four sensors with identical timestamps; other sensors would be
                     interpolated onto the accelerometer clock (NaN outside their span).
                     ``meta["rate_hz"]`` is the median accelerometer rate of the loaded traces.
    ``"waypoints"``  the surveyor's waypoints, ``X`` of shape (N, 0).

    Every table has ``pos`` (N, 2) metres in the floor map's frame (``crs`` ``"local"``: origin
    at the south-west corner of the floor outline, x east, y north; see ``read_floor_plan``),
    ``floor`` = the integer level (``floor_level``: F1 -> 1, F2 -> 2, B1 -> -1; no level 0),
    ``groups["trajectory"]`` (int codes of the sorted trace ids, ``meta["trajectory_names"]``),
    ``groups["time"]`` (Unix seconds, float64), ``groups["device"]`` (phone brand and model from
    the trace header) and ``ids`` ``"<trace id>/<unix ms>"``.
    ``meta["floor_plan"]`` holds ``walls`` (W, 4), ``wall_floor``, ``wall_type``, ``materials``
    and ``bounds`` of the loaded floors (``apps.maps.FloorMap.from_dict(meta["floor_plan"])``).

    **Positions between waypoints are interpolated linearly in time**, which assumes a straight
    walk at a steady pace between two marks; the positions are measured at the waypoints only
    (the ``"waypoints"`` table). The repository's sample code places a scan differently: at the
    nearest step of a PDR track corrected to the waypoints (``compute_f.compute_step_positions``).
    A WiFi row's time and position are those of the scan's system timestamp, while its entries were
    last seen up to ``wifi_max_age`` earlier: with the default 2 s, the position at the median
    last-seen time of a row's entries lies 1.1 m (mean) from the row's position on site1/F1 and
    site2/F4. Rows before a trace's first or after its last waypoint have no
    label: ``outside_waypoints="drop"`` (default) leaves them out, ``"nan"`` keeps them with
    ``pos`` NaN (e.g. to run a tracker over every scan; methods refuse NaN positions, so select
    ``np.isfinite(t.pos).all(1)`` to train). ``meta["n_outside_waypoints"]`` counts them. 4 traces
    have no WiFi scan and 133 no iBeacon reading; they add no rows to those tables.

    **Each floor has its own frame** (its outline's box). site2's F1-F8 share one box, whose
    origin lies 90.5 m east and 71.8 m north of site2/B1's; the origins of site1's F1-F4 lie
    67.7 m east and 51.4-54.3 m north of site1/B1's. With several floors, compare positions
    within a floor only, and give L5 one floor's walls (``FloorMap.walls_on(level)``,
    ``ParticleFilter(floor=level)``). ``meta["frame_origin_lonlat"]`` (F, 2) gives each loaded
    floor's origin (longitude, latitude in degrees, rows in ``meta["floors"]`` order). To move
    floor i into floor 0's frame, add ``R * radians(lon_i - lon_0) * cos(radians(lat_0))`` to x
    and ``R * radians(lat_i - lat_0)`` to y, with R = 6,378,137 m, the same equirectangular
    scale that reproduces ``floor_info``'s sizes.

    **WiFi cached entries.** ``wifi_max_age`` (seconds, default 2.0; None keeps every entry) drops
    entries whose last-seen time is older than that at the scan, so a row holds what the scan
    itself observed. In the site1/F1 and site2/F4 traces scans come every 2.3 s (median), 45 % of
    entries are at most 2 s old, 48 % of those in a trace's second and later scans were seen
    after the previous scan, and 26 % are older than 10 s, i.e. reported again from earlier
    scans at positions metres away (over the traces of all 14 floors: 42 % at most 2 s old,
    27 % older than 10 s). The choice matters and has no single optimum:
    measured on site1/F1 with WKNN (k=5, missing = -100 dBm, 5 folds grouped by trace, errors to
    the interpolated positions), the mean error is 7.67 m keeping every entry, 5.99 m at 5 s (the
    previous scan's access points add columns), 6.63 m at 3 s, 6.42 m at 2 s and 9.83 m at 1 s
    (2,223 of the 2,224 labelled scans keep an entry at 2 s, 2,176 at 1 s).

    There is no official split. Rows of one trace are strongly correlated (consecutive scans a
    few metres apart, cached WiFi entries repeated), so **split by trace**:
    ``evaluation.protocols.leave_one_group_out(table.groups["trajectory"])`` or, cheaper,
    ``kfold(len(table), 5, groups=table.groups["trajectory"])``. For a split in time use
    ``groups["time"]``: the trace ids (MongoDB object ids) carry an upload time 11 s to 25 h after
    the recording started, and their order is not the recording order.

    Parameters
    ----------
    site : "site1" (default) or "site2".
    floor : a floor name ("F1" default, "B1", "1F", ...) or level (1, -1, ...), a sequence of them, or
        "all" (the site's floors).
    modality : "wifi" (default), "ble", "imu" or "waypoints".
    outside_waypoints : "drop" (default) or "nan", see above (waypoint tables have no such rows).
    wifi_max_age : seconds or None, see above (WiFi only).
    ble_window : None (one row per advertisement timestamp) or a window length in seconds (BLE only).

    Files are fetched one by one from the repository at a pinned commit (only those absent under
    ``root``, only for the selected floors; gzip-compressed in transit) and sha256-checked before
    they are stored. On disk a floor is 43 MB (site2/F4) to 370 MB (site1/F3) of text, 2.2 GB
    for all 14. The manifest lists every file of the repository's ``data`` folder at that commit
    except the 14 floor images, which are not read. All 1,095 traces at that commit were
    downloaded, checked against the manifest and GitHub's blob hashes, and parsed with
    ``read_trace``: 8,908 waypoints, 20,069 WiFi scans (5.67 M entries), 196,643 iBeacon readings
    and 2.72 M IMU samples.

    References
    ----------
    Hu, Y., Fan, X., Yin, Z., Qian, F., Ji, Z., Shu, Y., Han, Y., Xu, Q., Liu, J., Bahl, P., "The
    Wisdom of 1,170 Teams: Lessons and Experiences from a Large Indoor Localization Competition",
    Proc. 29th Annual International Conference on Mobile Computing and Networking (ACM MobiCom),
    2023, pp. 1-15. https://doi.org/10.1145/3570361.3592507

    Hu, Y., Fan, X., Yin, Z., Qian, F., Ji, Z., Shu, Y., Han, Y., Xu, Q., Liu, J., Bahl, P., "Indoor
    Location Competition 2.0 Dataset", Zenodo, 2023 (the paper's sample data, CC BY 4.0).
    https://doi.org/10.5281/zenodo.8265879

    Data read here: https://github.com/location-competition/indoor-location-competition-20 (MIT
    license, copyright XYZ10, Inc.); it was not compared with the Zenodo copy. Full dataset:
    https://aka.ms/location20dataset and https://www.kaggle.com/c/indoor-location-navigation.
    """

    name = "ilc2020"
    urls: dict = {}   # relative path -> raw GitHub url at _COMMIT, filled from _MANIFEST below
    files: dict = {}  # {"all": ((relative path, sha256), ...)} of every floor, likewise
    meta = {
        "modality": "wifi_rssi",  # of the default modality="wifi"; "ble_rssi", "imu", "waypoints" on request
        "modalities": ("wifi_rssi", "ble_rssi", "imu", "waypoints"),
        "units": "dBm",
        "crs": "local",
        "pos_names": ("x", "y"),
        "pos_units": "m",
        "sites": ("site1", "site2"),
        "license": "MIT",
        "doi": "10.1145/3570361.3592507",
        "citation": "Hu, Fan, Yin, Qian, Ji, Shu, Han, Xu, Liu, Bahl, The Wisdom of 1,170 Teams: Lessons and "
                    "Experiences from a Large Indoor Localization Competition, ACM MobiCom 2023",
        "url": "https://github.com/location-competition/indoor-location-competition-20",
    }

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, site: str = "site1",
                 floor="F1", modality: str = "wifi", outside_waypoints: str = "drop",
                 wifi_max_age: float | None = 2.0, ble_window: float | None = None):
        super().__init__(root, download=download, verify=verify)
        floors_of = self.floors()
        if site not in floors_of:
            raise ValueError(f"unknown site {site!r}; choose from {tuple(floors_of)}")
        by_level = {floor_level(name): name for name in floors_of[site]}

        def canonical(f) -> str:  # "F1", "1F", "f1" or the level 1 -> "F1"
            is_int = isinstance(f, (int, np.integer)) and not isinstance(f, bool)
            try:
                level = int(f) if is_int else floor_level(f)
            except (TypeError, ValueError):
                level = None
            if level not in by_level:
                raise ValueError(f"unknown floor {f!r} for {site}; choose from {floors_of[site]}, their levels "
                                 f"{tuple(by_level)} or 'all'")
            return by_level[level]

        single = isinstance(floor, (str, int, np.integer))
        wanted = [floor] if single else list(floor)
        if any(isinstance(f, str) and f.strip().lower() == "all" for f in wanted):
            wanted = list(floors_of[site])
        wanted = {canonical(f) for f in wanted}
        if not wanted:
            raise ValueError(f"no floor selected; choose from {floors_of[site]} or 'all'")
        if str(modality).lower() not in _MODALITIES:
            raise ValueError(f"unknown modality {modality!r}; choose from 'wifi', 'ble', 'imu', 'waypoints'")
        if outside_waypoints not in ("drop", "nan"):
            raise ValueError(f"outside_waypoints must be 'drop' or 'nan', got {outside_waypoints!r}")
        for key, value in (("wifi_max_age", wifi_max_age), ("ble_window", ble_window)):
            if value is not None and not (isinstance(value, (int, float)) and not isinstance(value, bool)
                                          and np.isfinite(value) and value > 0):
                raise ValueError(f"{key} must be None or a positive number of seconds, got {value!r}")
        if ble_window is not None and round(ble_window * 1000.0) < 1:
            raise ValueError(f"ble_window must be at least 0.001 s (the timestamps are in ms), got {ble_window!r}")
        self.site = site
        self.floor = tuple(sorted(wanted, key=floor_level))
        self.modality = _MODALITIES[str(modality).lower()]
        self.outside_waypoints = outside_waypoints
        self.wifi_max_age = wifi_max_age
        self.ble_window = ble_window
        folders = tuple(f"data/{site}/{f}/" for f in self.floor)
        self.files = {"all": tuple(e for e in type(self).files["all"] if e[0].startswith(folders))}

    @classmethod
    def floors(cls) -> dict[str, tuple[str, ...]]:
        """``{site: floor names}`` of the declared files, floors in level order."""
        found: dict = {}
        for rel, _ in cls.files["all"]:
            _, site, name = rel.split("/")[:3]
            found.setdefault(site, set()).add(name)
        return {s: tuple(sorted(names, key=floor_level)) for s, names in sorted(found.items())}

    # ------------------------------------------------------------------ parsing
    def _parse(self, paths, split):
        paths = paths if isinstance(paths, list) else [paths]
        by_rel = dict(zip((rel for rel, _ in self._entries(split)), paths))
        kinds = {"wifi_rssi": ("waypoint", "wifi"), "ble_rssi": ("waypoint", "beacon"),
                 "imu": ("waypoint", *_SENSORS), "waypoints": ("waypoint",)}[self.modality]
        plans, traces = [], []
        for name in self.floor:
            prefix = f"data/{self.site}/{name}/"
            level = floor_level(name)
            plans.append(read_floor_plan(by_rel[prefix + "geojson_map.json"], by_rel[prefix + "floor_info.json"],
                                         level))
            traces += [(Path(rel).stem, level, path) for rel, path in by_rel.items()
                       if rel.startswith(prefix + "path_data_files/")]
        traces.sort()  # by trace id: stable trajectory codes
        names = tuple(t[0] for t in traces)
        if len(set(names)) != len(names):
            raise ValueError("a trace id occurs on two floors")
        parts = []
        for code, (trace, level, path) in enumerate(traces):
            record = read_trace(path, kinds)
            part = getattr(self, f"_rows_{self.modality}")(record)
            if len(np.unique(part["id_t"])) != len(part["id_t"]):  # ids "<trace>/<ms>" must be unique
                raise ValueError(f"{Path(path).name}: two {self.modality} rows share a timestamp")
            wp_t, wp = record["waypoint"]["t"], record["waypoint"]["xy"]
            if self.modality == "waypoints":
                inside = np.ones(len(part["t"]), dtype=bool)
                pos = wp[np.argsort(wp_t, kind="stable")] if len(wp) else wp
            else:
                pos, inside = _positions(part["t"], wp_t, wp)
            pos[~inside] = np.nan
            header = record["header"]
            part.update(pos=pos, inside=inside, code=code, level=level, trace=trace,
                        device=" ".join(filter(None, (header.get("Brand", ""), header.get("Model", "")))) or "unknown")
            parts.append(part)
        return self._table(parts, names, plans)

    def _rows_waypoints(self, record) -> dict:
        t = np.sort(record["waypoint"]["t"], kind="stable")
        return {"t": t, "id_t": t}

    def _rows_imu(self, record) -> dict:
        order = np.argsort(record["acc"]["t"], kind="stable")  # rows in time order (the files already are)
        t = record["acc"]["t"][order]
        X = np.hstack([record["acc"]["xyz"][order], _align(t, record["gyr"]), _align(t, record["mag"]),
                       _align(t, record["rv"])])
        w = np.sqrt(np.clip(1.0 - np.sum(X[:, 9:12] ** 2, axis=1), 0.0, None))  # Android getQuaternionFromVector
        return {"t": t, "id_t": t, "X": np.hstack([X, w[:, None]]).astype(np.float32)}

    def _rows_wifi_rssi(self, record) -> dict:
        w = record["wifi"]
        keep = np.ones(len(w["t"]), dtype=bool)
        if self.wifi_max_age is not None:
            keep = (w["t"] - w["seen"]) <= self.wifi_max_age * 1000.0
        t, bssid, rssi, seen, ssid = (w[k][keep] for k in ("t", "bssid", "rssi", "seen", "ssid"))
        scans, row = np.unique(t, return_inverse=True)
        _, key = np.unique(bssid, return_inverse=True)
        # one entry per (scan, BSSID): the most recently seen, then the strongest
        order = np.lexsort((rssi, seen, key, row))
        last = np.ones(len(order), dtype=bool)
        last[:-1] = (row[order][1:] != row[order][:-1]) | (key[order][1:] != key[order][:-1])
        pick = order[last]
        return {"t": scans, "id_t": scans, "row": row[pick], "key": bssid[pick], "value": rssi[pick],
                "label": ssid[pick]}

    def _rows_ble_rssi(self, record) -> dict:
        b = record["beacon"]
        t, mac, rssi = b["t"], b["mac"], b["rssi"]
        ibeacon = np.char.add(np.char.add(np.char.add(b["uuid"], "_"), np.char.add(b["major"], "_")), b["minor"]) \
            if len(t) else np.array([], dtype=str)
        if self.ble_window is None:
            slot = t
        else:
            slot = (t - (t.min() if len(t) else 0)) // int(round(self.ble_window * 1000.0))
        slots, row = np.unique(slot, return_inverse=True)
        n_rows = len(slots)
        row_t = np.bincount(row, weights=t.astype(np.float64), minlength=n_rows) / np.maximum(
            np.bincount(row, minlength=n_rows), 1)
        id_t = slots if self.ble_window is None else (
            (t.min() if len(t) else 0) + slots * int(round(self.ble_window * 1000.0)))
        # several readings of one beacon in a row: mean dBm
        macs, key = np.unique(mac, return_inverse=True)
        cells, first, inv = np.unique(row * max(len(macs), 1) + key, return_index=True, return_inverse=True)
        mean = np.bincount(inv, weights=rssi) / np.bincount(inv)
        return {"t": row_t, "id_t": id_t, "row": cells // max(len(macs), 1), "key": mac[first], "value": mean,
                "label": ibeacon[first]}

    def _table(self, parts, names, plans) -> SampleTable:
        drop = self.outside_waypoints == "drop"
        n_outside = int(sum(np.sum(~p["inside"]) for p in parts))
        rows = [np.flatnonzero(p["inside"]) if drop else np.arange(len(p["inside"])) for p in parts]
        t = np.concatenate([np.asarray(p["t"], dtype=np.float64)[r] for p, r in zip(parts, rows)] or [np.zeros(0)])
        n = len(t)
        meta = {"modality": self.modality, "site": self.site, "floors": tuple(floor_level(f) for f in self.floor),
                "floor_names": self.floor, "trajectory_names": names, "floor_plan": _merge_plans(plans),
                "frame_origin_lonlat": np.array([p["origin_lonlat"] for p in plans], dtype=np.float64),
                "outside_waypoints": self.outside_waypoints, "n_outside_waypoints": n_outside,
                "time_units": "s (Unix)"}
        if self.modality in ("wifi_rssi", "ble_rssi"):
            offsets = np.cumsum([0] + [len(r) for r in rows])
            new_row = [np.full(len(p["inside"]), -1, dtype=np.int64) for p in parts]
            for nr, r, off in zip(new_row, rows, offsets):
                nr[r] = off + np.arange(len(r))
            entry_row = np.concatenate([nr[p["row"]] for nr, p in zip(new_row, parts)] or [np.zeros(0, np.int64)])
            keep = entry_row >= 0
            key = np.concatenate([p["key"] for p in parts] or [np.zeros(0, str)])[keep]
            value = np.concatenate([p["value"] for p in parts] or [np.zeros(0)])[keep]
            label = np.concatenate([p["label"] for p in parts] or [np.zeros(0, str)])[keep]
            columns, col = np.unique(key, return_inverse=True)
            X = np.full((n, len(columns)), np.nan, dtype=np.float32)
            X[entry_row[keep], col] = value
            meta.update(units="dBm", feature_names=tuple(columns.tolist()))
            labels = _majority(col, label, len(columns))
            if self.modality == "wifi_rssi":
                meta.update(ssids=labels, wifi_max_age_s=self.wifi_max_age)
            else:
                meta.update(ibeacon_ids=labels, ble_window_s=self.ble_window)
        elif self.modality == "imu":
            X = np.concatenate([p["X"][r] for p, r in zip(parts, rows)] or [np.zeros((0, 13), np.float32)])
            dt = np.concatenate([np.diff(np.asarray(p["t"], dtype=np.float64)) for p in parts] or [np.zeros(0)])
            rate = float(np.round(1000.0 / np.median(dt[dt > 0]), 2)) if np.any(dt > 0) else None
            meta.update(units="SI, per channel (channel_units)", channels=IMU_CHANNELS, feature_names=IMU_CHANNELS,
                        channel_units=IMU_UNITS, rate_hz=rate, frame="Android device frame; rv: device to ENU")
        else:
            X = np.zeros((n, 0), dtype=np.float32)
            meta.update(units=None)
        cat = lambda get, dtype=None: np.concatenate(  # noqa: E731
            [np.asarray(get(p))[r] for p, r in zip(parts, rows)] or [np.zeros(0, dtype)])
        pos = cat(lambda p: p["pos"], np.float64).reshape(-1, 2)
        floor = cat(lambda p: np.full(len(p["inside"]), p["level"], dtype=np.int64), np.int64)
        trajectory = cat(lambda p: np.full(len(p["inside"]), p["code"], dtype=np.int64), np.int64)
        device = cat(lambda p: np.full(len(p["inside"]), p["device"]), str)
        ids = cat(lambda p: np.char.add(f"{p['trace']}/", np.asarray(p["id_t"], dtype=np.int64).astype(str)), str)
        return SampleTable(X, pos, floor=floor,
                           groups={"trajectory": trajectory, "time": t / 1000.0, "device": device},
                           ids=ids, meta=meta)


def _layout(manifest: str) -> tuple[dict, dict]:
    """``files`` and ``urls`` from the manifest below: a folder line, then ``<file> <sha256>`` lines
    (a trace file is named by its id alone)."""
    entries, folder = [], ""
    for line in manifest.strip().splitlines():
        if line.endswith("/"):
            folder = line
        else:
            filename, sha = line.split()
            filename = filename if filename.endswith(".json") else f"path_data_files/{filename}.txt"
            entries.append((folder + filename, sha))
    return {"all": tuple(entries)}, {rel: _REPO + urllib.parse.quote(rel) for rel, _ in entries}


# sha256 of every file the loader reads at _COMMIT (the floor images are not read), computed while
# streaming each file from raw.githubusercontent.com and checked against GitHub's git blob hash.
_MANIFEST = """
data/site1/B1/
floor_info.json 05e5fc2d6c5ff39eb0519de44537fcf194dfa93af0d887c2fcf4b01f3fc5f80d
geojson_map.json 7511c06f80223c70651641f312172dfcd489c16da73fcb329dfab7f71f206e0f
5dda14979191710006b5720e 3bc74a131b911659ee06ff8d1089541534a64aeb1304453ddb1a89f4a43eb41a
5dda1499c5b77e0006b1752f 614a2d89c69c91a323090ede33edac6b94f1bdd7441df35118ad41aeefd594d4
5dda149dc5b77e0006b17531 b46261810c5d71abe54dc288d7594c28b3ab7530120ba939a370023283ccd7ec
5dda149f9191710006b57212 eb3480d690fde6421906d042367f66edf4b07cf2834385bb318f0289201e98ff
5dda14a2c5b77e0006b17533 e3b28708629829950e69da144513ef54ec353a63d40f982b3c7bd9252f84d959
5dda14a39191710006b57214 a273ef6987a9d38e6b6b55b26a1f96a0e89ff28b3fda9e1253a3b16a6f236762
5dda14a5c5b77e0006b17535 e1586a3d444a53c6ba33e84ebc1010909acddbfb6d734f21c670a436db884840
5dda14a79191710006b57216 e1670d92a75340a9108a3857a16f90fa86d0b13bc0fe35de898e8c6b42b6a584
5dda14aac5b77e0006b17537 967b362fc7d590742fc0526c03981904df677e4817397791c9c158c889ff0f91
5dda14ab9191710006b57218 6a9d627931e30949d232bf20d7ce5e056a202d0ce44907aa5055f2cb5b0a2900
5dda14af9191710006b5721a 01607d5cd2a07b192a8d9dee9164036f82f16f9da652f519aa0f4b6ef1341121
5dda14b1c5b77e0006b1753b 5e7ba396db8f4f95d459150659a32f2ce76271da6c9c5a36bc83d68de432060d
5dda14b49191710006b5721c e0381b8db91765b452d7ab39d40a9b661938bf01e5491139514d77a44a15b0f3
5dda14b6c5b77e0006b1753d 0c9efd92c9991a770ee7a6a1eba0ba0cdf45249db771d8d926451946704d96b4
5dda14b79191710006b5721e afa6f5d08984510cc0703672f16dc65f3a307e0a8576e88806d7967a7b11ecad
5dda14b9c5b77e0006b1753f 6decaa4a32c70248cd0215e5e922d02c755bbff01f77357afbe2cad11b9742fb
5dda14bf9191710006b57220 867b860d9735b8680fa0a5a3e5f90fe655c6bc0c037a82d132ca6be29963949f
5dda14ca9191710006b57222 b7cd712d09ff89025faf8a9dbe185936c8d5b70bbb99b3283f536cfc6e5d69f8
5dda14d4c5b77e0006b17545 49b5b6b5238ed1a20d7b6f3d8ab7f2b3acef70fb1e40d2e0fcd7854ca3b6541f
5dda14d79191710006b57226 2cc235f55faf5a559255177c261ddaacf66916006d6c2968741420b051b504b4
5dda14d9c5b77e0006b17547 3d86b48c009bf0c4777f7b52415390ddfb5e1794774c9643b46851079c09de2b
5dda2565c5b77e0006b175b9 c7b2d13af93dc60dbcacf39ded16e3c598abfcc8c5d2938346251caca636da09
5dda2566c5b77e0006b175bb 210b7351ee93c12c3a6cb573659b931cf694a6ba87ef8ecc49b804d2bffd8c99
5dda256a9191710006b572a9 4ebd2d4b7d9741f0386622f36730d1e8a308a8b7843f68d1a25694c38f112e71
5dda256d9191710006b572ab 62c01216797acd8846908ea42b1d7ba2ca018d953d369967ad2d3178f47478a7
5dda2570c5b77e0006b175bd 63fc9d41d8394c128d44b66f63fb2fe4e95dab937841206a32b32b8cba723b3f
5dda25719191710006b572af cae79c76b3ec3b9d303e2de62080f93ebabb17bc5c0376c81f75bfd19fc9950e
5dda257b9191710006b572b3 5e50ed939e1945faa3a3ce97cc1d0aff51ffd8a39864e39a08d7ecf0f3e0bdbf
5dda257f9191710006b572b5 92dcbdf96138c1e930df5a65a0b9ae9916a40adfc0f5ffa5d96cb79d61c2ad25
5dda2589c5b77e0006b175c5 3d6b2849f734fdc919836e800e98c318f485890819acc21261f02e8780412f81
5dda258a9191710006b572b9 fdcdc0605a6eef698b19f28c2f802166e79abc766a08b147b35f834455785ff4
5dda258cc5b77e0006b175c7 5eea5b0713d9c9f7d9351871546338b6d4f2a77f740bc522a74d196e625c767d
5dda258dc5b77e0006b175c9 64c374fc77c4c24146f95750306591b9744959b3adb7575c62bee5170ed1662b
5dda258e9191710006b572bb 4ef8f701c192513de95fdc6fe1894ec03eb23ffc22b31392c9396221f5ac96b8
5dda258fc5b77e0006b175cb d893a25a8cd784566e0240c1318c60644e2482e82e5678a34e73b155d74799eb
5dda25909191710006b572bd 6533c58aa5d794bbcd4d0cbcb77bb1c14014b6c0c8e699ce097377e7674a685e
5dda2592c5b77e0006b175cd f40ebe2963572dfe6ed66712d29cf3bf7d0800e4e8be9eb5b86fae6605dca1f3
5dda2593c5b77e0006b175cf 119bc09392f3e508dc6568c253d3f7a44f414dff719de0c9dade9a757a39d0c1
5dda25949191710006b572bf a9a85a46fc6e30224b2751893497652471cc6a93333ad3f4d036b9f7c95cb6d6
5dda2598c5b77e0006b175d1 473316fcda80eeb299391309ad0d901089ec55142d39f24f40c2dea84a3ae984
5dda25999191710006b572c3 304a2aa068c7ea37dfdb93641c8ebdbad322c3502b4e27cc0b4a61243618f35b
5dda2599c5b77e0006b175d3 d95f928623b711150d3f91c60988d76d016b1ba7fa94318a900e2710e380ae7e
5dda259b9191710006b572c5 62a44c08c628406ec3684be65408bc575fe5271d7ef0d800c5663f2df86c0090
5dda331ac5b77e0006b17623 63add147f1c46a8e96e75b7f8ccf8725bcc99f8e2771987c782eca9b83da2969
5dda331d9191710006b57314 97d1624f046469330fb07553554f2e04ab3561ff52339df4d0f9a3d392409308
5dda331f9191710006b57316 0d4303f5d6d89bbb39d32690983b2904c198bb5717a3ea2cdd34b538a945784a
5dda331fc5b77e0006b1762b c48d659edbec03c33495c39b8aec8c9be60ee5ceecfeef7d8d9ed82124572cea
5dda33219191710006b57318 49e0b325147419d483624199381b5e94e6c1f5f2e1bf1f4d940ef15e6910854b
5dda3323c5b77e0006b1762f c27cf82384c02dc39ea2e044f06ce7a90372921893551ce6fad049a3bab49f4d
5dda33259191710006b5731a d5db0d8ac4c2f652a16135d0ad011450420c4efa27a9d57c71884056d8dd4a9b
5dda3331c5b77e0006b17635 cb1c5700e208d68742f2bdf88089fde8af3b97973b518fd2bdc63a42124c5d38
5dda33329191710006b57320 671bf4f4c03aeb176b884d042899c68d2e25c44f6516cf02981d9f67de210219
5dda3332c5b77e0006b17637 5c56539adf9468dddb65f0a806303963f2248009dfe2a5c066bdab354e262c06
5dda33339191710006b57322 8450b88474d480729d3b0e096f78db324a7e4728fcc7e208509b18052c3489a7
5dda33349191710006b57324 ce2a10d93cccf510e99346bb35ee0bdd4993ae12ffaf9a589a856719f6eb0a84
5dda3335c5b77e0006b17639 51746d2bea9859d1134050ef7422a9443258ad8d725ff6bc31808ba08cebe00d
5dda3337c5b77e0006b1763b 2346ea7f3a5ac3969faf644d478f369d1bdc7e4d20f2f7a7c7038a94c7983021
5dda33389191710006b57326 608f0f2267cd4d4c68be3cb0f9d53dd975cc7aa703d2510934579ba65be5e666
5dda333ac5b77e0006b1763d 2db4bdea23dfdf60fb5c5c8e9aa8b81c72ef5b154cf2a2b7439f954bc89bb62c
5dda333b9191710006b57328 0e6a1e9490f58988da3cc7e7d2be24131f7c01920001eb05b60bf6b0556b79d0
5dda333bc5b77e0006b1763f 55fe7113cc3e7b9b0c999a7a6269119c9aed385796daa72dac64453bbf9a247b
5dda333c9191710006b5732a 64025572bea9455a9b107f183cef031584597b809bfb8ad7fd28d79922392226
5dda333cc5b77e0006b17641 e5e302a39895e281d588f19fed4b2c5b9935a9e2d6fefc27adc26bbabb59b587
5dda333e9191710006b5732c 77287aaff7e1ba70d7d3a9905962c96edcbb6b4651ff80f0ce16f12fa14287aa
5dda333f9191710006b5732e a5f6018c89a37f07e18aa01d629e8a44436df150534e263160789387027290b9
5dda333fc5b77e0006b17644 45c2d10e6052376260bffd01b17d2292de900d37f6e304f28ad4bc8813cdce82
5dda33409191710006b57330 4a18762efb9d530789618bf13de1f2f7be50a5711894265ab9175f74a6e9df5d
5dda33409191710006b57332 6f69e279edfbd620e7b2ece632ebe93aad17b449d45c0e389d848f855e632d98
5dda33429191710006b57334 c6d54690c1443cb8d2571ed1986e23fc70c76a654a7634df0d9b801e560cd309
5dda33429191710006b57336 2afa6f2e4e8b15a62eb4b4361a0b51931d16dbba3edc46d97136942a2a8666f9
5dda3342c5b77e0006b17646 a8b70f8c88c9b05c18e0da47b87e0240ab4e665fe9cad6e3d3c8adb4e621a26b
5dda3343c5b77e0006b17648 3cdf49d87d0e9ea5ad012c2e723862389b3e47f56064dd353daef738d694c768
5dda33449191710006b57338 228ec02c2af0e97f421cfcf048c5c1ca69f89267a03202622a95a01f8b8e78d4
5dda3344c5b77e0006b1764a 6caa9640fbfcdbb1e16e92eef86ecc1759ac356e05af8d6b8def9483f450f6fa
5dda3346c5b77e0006b1764c 57685c947eefe96e9c94809b1330fc169d88aa672d34b8abe8fe102eed87a8dd
5dda33479191710006b5733c 0ae17ebbc4f0f4833e0ebfaa4832495a90efd6cd621963c4e1105c7491865de0
5dda3347c5b77e0006b1764e b14bc930c03a3302d4ded0ff33c2e8023f373f7978ee536e7eed47f3ff81fb48
5dda3348c5b77e0006b17650 91e1b9764b743e2481e2c55f6c741cfce5618411f5b8ee9ea72703ce3e4a7162
5dda334a9191710006b57340 294573b08946a1f0cfad61d29ecd13346b83e23f2ee0855dfe748c6019f76cea
5dda334b9191710006b57342 11345a09db49a99b84bf5ba64cd97b87777f638e6530de14dbe7ca17dc49eb9f
5dda334cc5b77e0006b17652 3fa982f581f7828cbd34c39ad2c355c560f8f919fae19f0b9b22a1b885827b46
5dda334d9191710006b57344 7e3af569c065d3c537599b28f096e5799b0e38c3b4f37d89f8f9c3caecc67864
5dda334d9191710006b57346 f0932d6388ca404c892891832a71dfcd321e00b7798fcfba583a27c52faea602
5dda334e9191710006b57348 f03d827fd0fc6242bc30a3dae5a524f5aa3eeb1db54ad284b1655544a17363b5
5dda334ec5b77e0006b17654 032e148a788d1a6693f8062bd62333a540af293196045c0179947fc282e2bbf4
5dda3350c5b77e0006b17658 026b8f8c32ce59efbe6adb5a0859924868ef6a88e3dd5321f9413640baabd515
5dda38729191710006b57350 d83d889d5045c6a3e028fcc896b09fcec0e360e18329ab3479951d0ffc47eeff
5dda38739191710006b57352 7bfb4c65e0dc41b22ba45d8c9629ef292ea2a33a96d8738c0c267d1a6dcd476c
5dda38749191710006b57354 134dc71367ea6b01b703f07068e5a6a4986dcd7e1935b7cf72c5f071a141bdb1
5dda387ac5b77e0006b1768a 023b2fb72f89aafaf2a2f9ebc4ad725801c389b11cc5d236f3726b7031d6a641
5dda387c9191710006b57358 4704b04b9828e603747945db9a2e88969ce6bbe55dc68541a333b278ad76ae58
5dda387e9191710006b5735a dd17afd8629cd406147056ae5218574fd588f5cb21a635df0905abddaa4496da
5dda387e9191710006b5735c 955cfdc6143e88de281a5d8ee99791c1180852172b9baf64ed7b74edbd3f99b8
5dda387ec5b77e0006b1768e 76c038d753f78cab54b5442b29e28cd1414589088328c1a4b0eca373b5e04f77
5dda38809191710006b5735e de38a448dbacb1d42ddfd9f6ee658993f8b6ab42022a173402c1e28b473ecbc1
5dda38819191710006b57360 b426323633eeb3c5d8a59733093d8e1309db2b82b9ea7d6150201e60ab20c8ae
5dda38819191710006b57362 b21ad1b7420698319a96b33a825dc23c52e1dd9bab64bfd701c059c235b87ed8
5dda3883c5b77e0006b17692 07727b33f158ea13a7b69d0b2a96d8d2525fd9cbd50003b34112c05e0b6a5daf
5dda38859191710006b57364 ce1dc52c588c186e846eaf3eed31783a658b9e9390cbed1b7bec077b983879c6
5dda3887c5b77e0006b17694 b930dec570425a7122731024cb53cee67cd0b328dac062f1db21f4d6254bdf1b
5ddb8844c5b77e0006b17977 03a991961edd425eca27592cdd41b279dd8ab6b18377cfc559b9e152866bdb3c
5ddb88459191710006b57612 376e7c643f48da793c351a8ae60884d7911edd8bf69b7945a5306fe510222740
5ddb8a039191710006b5761d ee5590b438d85a7928f6fd8e26299040412488702fd1fd2a2fe1c1b7648678db
5ddb8a06c5b77e0006b1797c 1789a0efaba1640a1bb774ef4f894b5a7a1c731efc87abb05cd20f8524a51e1b
5ddb8a07c5b77e0006b1797e 62c6fa113021230624c53bc65ec19e9fef48718eb83361e09730bf12c731a005
5ddb8a08c5b77e0006b17980 dc44ce26b31ec4245047d9865eb80b92e5144f03cc5257314fe57c6eab22a477
5ddb8eafc5b77e0006b1798d 9e12bb0e4afa54c18b1c03bda4200007f59a51f6f22b56cc832539a83fd5631d
5ddb8eafc5b77e0006b1798f 9b43ba823a439efe59c93e8e3e21a06e07db2a09ee98716f33b514b2ff99713c
5ddb8eb0c5b77e0006b17991 4246a6fa20e581053d6469c1e24d458b0d20d90051c20c18090f4f567f6ae30e
5ddb8eb19191710006b57620 a723d3cb3ce417d3917bc81627a76c828abf82dc5ae19a97977b77090c006901
5ddb8eb2c5b77e0006b17993 31a2bbc46f6fb1fa160010ca581b556ff47338fbccdf48e659d1c384390b8a35
5ddb8eb2c5b77e0006b17995 ca91a33cc6c141398dff7f454a559894b1185b26171d39e994e44e91af87b402
5ddb8eb49191710006b57622 b949a1eee5360d4f62b389ad0691d07bfd96cc02a17952196fcb198bfdc5d326
5ddb8eb5c5b77e0006b17997 06547edad77bf48b1676725720ba2f0ab560245111f0d71a0b010b9a6b5678aa
5ddb8eb6c5b77e0006b17999 bb54bb694d20e15c56c373fa64cabe1e132af9650f94a9be7825c7294d40e740
5ddb8eb79191710006b57624 78daa1ee6184ae80d5943028c7ec9425592283b0e5d37fa47fc1c56bc1cedf5c
5ddb8eb7c5b77e0006b1799b 4e6645f5dfc8c7eb6bb022e65193297a93694093d0a0da3d6dce8f9f50ac2938
5ddb8eb89191710006b57626 bcda85a3478c3a9ac1070078035e3e94c38cc19a2c1c7b9ae1de9d0ad34e6c30
5ddb8eb9c5b77e0006b1799d 1e98e477ce9413a6ca7da6430adbdd5fa4a5a9cf429e15d7b939f83400527eca
5ddb8eba9191710006b57628 92f243cb5d2f0783e22066faa8d25f589208a920db142dc9c79c1572430302e2
5ddb8ebac5b77e0006b1799f b3f27805e60c3bed71b17c20715f6d57b7d0dac56c5cf3c17e77731276d30439
5ddb8ebc9191710006b5762a 7b7443830b013060038cfbbaba456dea1849f796ad93549eb604600856dbbcfb
5ddb92fac5b77e0006b179a2 eb49496c482b175ab6aa26633d02269cd3fb47a43cbbb2781fae8d4e2b041c48
5ddb93029191710006b57635 9eca0bf35f14f5b84a1685fcc2acd729e1b44d4e1d6d71eed27ef70a921ed811
5ddb9302c5b77e0006b179a4 b3325c035c0a57a814ec2b5985f8f53806cd90ea92b18e9c71eab1fbb058e1b6
5ddb93049191710006b57637 a5c5a51a81aec47ddb52c224f832d4170ca5901ba267fe01e34a520a4f0118ad
5ddb93079191710006b5763b 1e8607310f195c673747b32cdf46cbfe404ae23814d1ae696bfe47c32c6d83a4
5ddb93099191710006b5763d ae5e7c5843af07680fca64aa9b9a50d4a685adf65fe44c1f8b0fa1b391321be3
5ddb9309c5b77e0006b179a6 06f58d0e7cb3863e97a02dea55624dde4fe761e2a5c6e10981a4a02b4ea591e2
5ddb930a9191710006b5763f 64ebb24fee59c9c653417a6b0fdd301390231c0a9fa347802ae268152d9de47c
5ddb930b9191710006b57641 7743f9f146b92623bda5f8ace4f58542e3dbc64755377a2c35ce725c2972e869
5ddb930c9191710006b57643 4a5fa60c4e44fa2c1db367601dd49a96b9366f149b05f7ae3cb30fcb28323ec4
5ddb930cc5b77e0006b179aa c77eb2fcb5c7533586bb5986a051bc2f4677a6d90e77a1b5c7caff206869c234
5ddb930dc5b77e0006b179ac 37f063103984e2e93f6080f476acc000234dfe9e8b687110b768c8ea5706c35e
5ddb930e9191710006b57645 f65c98522d7583cc653887733e5be7c46b222ad28b5fdeeccddbde01890b8879
5ddb949cc5b77e0006b179ae 0e2ccd42e9a038314e910c77d169eac0c3acd74ce93bf42ea7dabaf2f255d254
5de8c24d376b9d0006fdaa23 3e20a2311f0811cbcb908b50163379827f68b45642280a43f38cd47d5b09a13e
5de8c24d7491b00006eaafdb 5d5292a64e11b269585f64e736aff545a13dfe81921616a1c76b3d84f88662bc
5de8c709376b9d0006fdaa35 a56dd88f2bc230291963d525a8487577f5643c50c158e529e747b05b4d618858
5de8c70b376b9d0006fdaa37 71074b81ad3938e28e2d6267e30650b1f6352622ffcb83b5f9c326e7377dda35
5de8c70b7491b00006eaafe3 06f9089285636208caf526c87902cf21a62e9ff67b19ba169acedd7662c6afa1
5de8c70c376b9d0006fdaa39 ca9221d39152c76a666200882d1d6a7adffc746b74a0cd6bc8c93664238b75ba
5de8c70e376b9d0006fdaa3b 9067d4cb1521efb918fa5a6b254384832dab92fc2c12728e10cdb55e4c841096
5de9c2e19799280006a13e9c f212da21209b4f2511228eff3e5f8b2c29d1dc36c84f9d00879191e46fb02e9e
5de9ce75e8a6030006a80e0c 17ac833e0459aec75af9c382d8208655269737aeb26b330953f69ab05af82e6f
5de9ce763cb9290006540b5c f4cc01136cd01282a9fd7572fe764ef7efdcd4b61603671cb0a1ac996920ab5e
5de9ce77e8a6030006a80e0e de1f812b16cccb5a7af9e73d70f433e1878f706e13941325d07c556e08a8e5b7
5de9ce783cb9290006540b5e 829f5ea5238bef460e3b340350e5da14008b21daf07ee3e48f93f736bf3ae8b6
5de9ce79e8a6030006a80e10 57a78f5cd1213701ec81981841d404137d739e5e53fd5185059c4a639b6a032c
5de9ce7a3cb9290006540b60 71b12ec8bbdf0c8a0627446ffbcb123dae81aa44c7dbf5e34ea7d4e9ef4f320a
5de9ce7be8a6030006a80e12 7e402f2f74e7de8183aa5f0b4ee884fe417751b47f6776b983329ce79307716b
5de9ce7c3cb9290006540b62 cbc2cff0d02561955d3098b2280e01e0cdc219b51d7facfa49b4089da7315633
5de9ce7c3cb9290006540b64 6a00e4c393bedcb02909604cf581d112e5a841ff98e42204c119271a827f56a1
5de9ce7de8a6030006a80e14 905687c7b4372c3e98129d21dde0712d2babc2566efa47145a528a1910dae1ab
5de9ce84e8a6030006a80e16 b09b9c198b2024e4a270626bfcf0bd113a735a7c9d12c7cba25654b0110a604c
5de9ce863cb9290006540b68 689d8c1426aaa98682ae761d1717d285e9405b4588465480d386c6107dfba468
5de9ce87e8a6030006a80e18 2e502a8d49c69a35f1de7fc274651332af067a54eb221856af8287c89cb12f35
5de9ce883cb9290006540b6a f040f1c025f971d300be7ea497fd3654cc0000957cadac914e3855f9aa34ebac
5de9ce89e8a6030006a80e1a fa540077c28e4825219866412926f1bc29d99f70a92687632d97ec15b0e53fef
5de9ce8b3cb9290006540b6c 7e23b56563dd2c71c56e9f871e8ca53a0fd1dbc6671f54586e42111f215be627
data/site1/F1/
floor_info.json b58f35112842b0c9c865d6e986cfbf2be21624698f477d6c00fecbaad1824dcf
geojson_map.json 13c976c6dae79ed5896487fe299a89cfccf567c397244b81a0ae22391d875ebc
5dd9e7aac5b77e0006b1732b d4b03798d8e19e19adf60b71bd0ab8462c24e37a2e97d8ea7de43ba4cf4c3f81
5dd9e7abc5b77e0006b1732d 68718994cd6386726652977b8fdc133cad1b77c4f167979a47033bb271cf2ea9
5dd9e7b29191710006b5705b cea4bb58decdb63a499b6789c3a1b97696a779462d61188d8e574ef129bad066
5dd9e7b7c5b77e0006b1732f 8c96ba5633cf75e5590b7251abf1d9da79d7ef18b5e81dbdbde8d20efcf17389
5dd9e7ba9191710006b5705d a2cc72edfaa82d67cb899c71dd0aa656cc46165605fe73a527014a29288625dc
5dd9e7bdc5b77e0006b17331 ae2016b844895c9f1f89e7fd201c46c023c7fdccdff1b9608ef3a42cb2d1e936
5dd9e7bf9191710006b5705f 3008c24a1dda1289b2495bcd0faa195f170df471f6596528c0bb324d5cb6d662
5dd9e7c1c5b77e0006b17333 3ab9c636e7ff30f73b13a8b8f24ea50df7d0196e1961da1b71c2f336051b5748
5dd9e7c29191710006b57061 52992cd8190ac408f114cd11cb23cd7ca9620647f5d2868360918dfda687ffd5
5dd9e7c4c5b77e0006b17335 1d22eb1d45ae5064643e7ec0277525ef193ce7e9cb633c5d34b13f70746f16c7
5dd9e7c59191710006b57063 2bea02f95e4818e751623659193bbe0d0d8484f3112c67f994ab850ae9503abc
5dd9e7c59191710006b57065 cb0a332b90b7e52b5f914a6289ef015f312f53959cfb967bcd1a726ad3d9c4bf
5dd9e7c5c5b77e0006b17337 fb25bfb90e0c3c7ef8b0514cfd062466e7833b29f493b1903a56978dc86e199b
5dd9e7c6c5b77e0006b17339 b9bb56db1916ade6732b1e50fbb4b7578eec2036d3c3c9c7f7c9198658705a87
5dd9e7c79191710006b57067 c302240ab8ba54c504e2db46c7e8313fb3ef4bb95b00fd5f0ad23582755b8ebd
5dd9e7c8c5b77e0006b1733b 07563724759cfc6299097d6a2e2d207724534600420548317e1fe4284b8ad1fc
5dd9e7c99191710006b57069 49057148f2a388045aa45fadd51cc1378c212c9115a433355c467171c66b03e8
5dd9e7cac5b77e0006b1733d fe853e825fc47f7bfaf4c0dcde2928778870c429b2cae4e10d3e06766c0240cc
5dd9e7cb9191710006b5706b 9c24f492140d2e7b4dcb2c53aead6985ffa6a2c220a97f18b0e98b009ddc79e0
5dd9e7ccc5b77e0006b1733f af269cdd55b7fc97567b6c2b0b76b07f1edff983b2a397d76eb9334a7f3cdd8d
5dd9e7ce9191710006b5706d f4cc929d66dfd54b389509a255983d204bfef7259fe917c0b49b3896002f91cd
5dd9e7cf9191710006b5706f 73eb7c182cff69a41f2bfe093c289aaf246656519f80daa58eb998ab3f8a30ef
5dd9e7cfc5b77e0006b17341 8886b5e0e2f5c03efcfd894aeffa87ef60f04de39e34429f4ec208bd286692d2
5dd9e7d1c5b77e0006b17343 74818cb217218ac364cedaec01a8384d42c986f8ccedaa1b1818cc0198f8f679
5dd9e7d29191710006b57071 815a0d8e95dfbb7387afb64edc66910a41f5117884ea53c5a038e04c0d290e23
5dd9e7d4c5b77e0006b17345 cf17f92aca78eda882edd299844c71b849ee4bc8fda0444d39105bb290dc69ce
5dd9e7d69191710006b57073 67e288a566c0b7d72285050211e67a0e6f338c6dbf39f04930120668fdfd1220
5dd9e7d7c5b77e0006b17347 f5453fdef89edddcd6aba3e36a14c03a388ed3af8d790be7ed4225099b56d1f7
5dd9e7d89191710006b57075 5cdf8b06c4867a5f487180470bc9fcfb142e6f5332159c52c79c48ff6c567326
5dd9e7dac5b77e0006b17349 29a1ee250d1ecd505bf3a0315e14fcdcf2e93f9718a4ea6298dda6e0561bd544
5dd9ef829191710006b5707a b5e11734bae1fc89ee501b44c88a6a14edc9f88784deac96c9df097cc17b2bbf
5dd9ef84c5b77e0006b17355 c93701c52301b2dd1bb65d43a978f67a3ef609fd8afd540bf3d7aa9c9473e09b
5dd9ef859191710006b5707c 8ed09bfe8760ca38deae9c698b8271240ac5f177b63b35f0877ae86d4e48da9e
5dd9ef87c5b77e0006b17357 1c5da44e44c596bb1f2606fae2d9c4a70395a3e13fcc802432ff9265af66a8a9
5dd9ef8a9191710006b5707e a9d3b99d15417c49987b0eb3ba9bbdf9763cfb5047a16e55873d6697b77a01e3
5dd9ef8dc5b77e0006b17359 241ab7154d3ca457407cf0c7aca6180ba1cec32f88076740615b1aa5cd72863f
5dd9ef8f9191710006b57080 478c206017cb9f9f70d469cd5a547465e8d55cd7f0f3432c4a7d26478f825675
5dd9ef919191710006b57082 1de95445747c70bc112415b99eba55fd46fecf492941b9654e4c3773453bf6e9
5dd9ef91c5b77e0006b1735b 2e7bc08bbf14930e6546f9b93bd24122ca3f4f47eaaaabcb1dd7f2ada03ca790
5dd9ef94c5b77e0006b1735d 673d5da53b40df8a139afd4d4d3b228ed97b1efe80f0f3956a75302ff9d4f873
5dd9ef959191710006b57084 1b40b8fafe5e3119db4548efd3b16461740660624706bf0246fc47aa2c0b959d
5dd9ef95c5b77e0006b1735f 3509e14222ad58b47acd0f8183beabdbb5816468cf17e036f17682995509e80b
5dd9ef979191710006b57086 fc99cacc2b6abe06d4346c7a1121e4ecc042f5c202723926b39734a4343b741d
5dd9ef999191710006b57088 9f316d6414120b283bbc0b5e6da4f4deba434dd72fd99ad656ff500cdd407e98
5dd9ef99c5b77e0006b17361 0d63d96180ade991ee81f087a8115160d97deeee0f0ec3665b10c198eb9e4319
5dd9efa29191710006b5708a 6c9794a7e3f2e2b1c5f2ad6f3ba4f1d01f1979fc8bbe71da9eae9911ae1d3490
5dd9efa2c5b77e0006b17363 085576a0c8d9d28bb0f04589bef9fdfb4ecacca261c78d2cc60f31f9beeb1b23
5dd9efa5c5b77e0006b17365 1f19d4a39094060e931be3c9293808f122bea0578911c20ed5d166bd00fda46b
5dd9efa69191710006b5708c de28c1ef3a18f1020d023da5ee1b95e3f7ca8b74c672a3d33c87aa1b4e74c35b
5dd9efa79191710006b5708e 299ef7fb05e48de6eea91fdfa4d09f8f9f469e2e46292dd0503ea97dbf396864
5dd9efa7c5b77e0006b17367 0bb4e946c35cb6907d47e3dbddb2bb1fbc465b7d9473b5992e34b34ee6e79905
5dd9efa8c5b77e0006b17369 d4b36d08c696e40dccfe85600179b215911cafb95791954090fdf385fb766e81
5dd9efa99191710006b57090 86b75278ddc60e8088d6a64f1706e3156d969be0e7d7e9425da75c1f093183dc
5dd9efa99191710006b57092 5708d390c75d84b8e0e98ca55898f82505ce9e344a4a52eb1e2fbda48163e9cf
5dd9efabc5b77e0006b1736b 4cc2b3093de5b419ba0d0d7dc5f4b3afaaa74d0b66bd6b14c6a61b341a1d573a
5dd9efac9191710006b57094 f1216b0478dffb82e3ed23b6789b0a3a2d45777b95d9520448835899461d0139
5dd9efacc5b77e0006b1736d d299110257c1e12b2e117363aaf4842d173ca2eab6973d331cb365dd1bc0461f
5dd9efad9191710006b57096 aaf3ce5d3ccf0b14c23aa6a014a8ea1e76e214140f5959cf33ff0a9118beab52
5dd9efafc5b77e0006b1736f cfaf3a46c3fa324b765fd1d9497a888985a9797a96796ebd46b3a184782c1070
5dd9efb0c5b77e0006b17371 20afa3bad1683b30537f99a117aa0646c96729a42b4651f748bd072183cc3a45
5dd9fd2cc5b77e0006b173ba 8b9ead2f9c09ec5a73c595fa965dbcfc8b49efd2b66ae100f6def601218c3889
5dd9fd30c5b77e0006b173bc 58e5ca2774643a461143ed8bbd620cfcd0ee45d5c45a98041976b3cfd899fb7f
5dd9fd37c5b77e0006b173be f0e0ddb4f1186eb444eea02a854cfcaf4cab7c4614d7bf8f5e62b8e5efdba3a1
5dd9fd3a9191710006b570d2 c0a4ffaf27c071d5302398551ff98d4d4f9be0de03d30659ad660fb0337dd571
5dd9fd3b9191710006b570d4 e74cb0576b88944b1ad6699f781b7356a74ad2787295d7846b5f213710a1378d
5dd9fd3bc5b77e0006b173c0 c590067f77f115b3cb18b495fe5fcb35543621e24966b9a74c112886a9c3fe02
5dd9fd3dc5b77e0006b173c2 f08d706e32a1457989296efa160d138b878351cd5970ea90500fe3228455e3e6
5dd9fd3e9191710006b570d6 5c0ffbf31a91e3cb46d0bc57c41c2610d407c6ea656b165709eaa131973de230
5dd9fd40c5b77e0006b173c4 f9f1189c593f0232dc9a817fe8757157bf92de75afa77356476891e13fa3e51e
5dd9fd419191710006b570d8 199a79bc410a4693817f30eecad052e7c7bd6f3cf338cb060ab122a854ffd32c
5dd9fd43c5b77e0006b173c6 ce32d92677476a40b9163489868bee5c5246bf414e9b514d5d0237aa1b6205ec
5dd9fd469191710006b570da 6559a7ed4448c9ffc88bcbd662b1411040e3f582e792d8054c530567a30881df
5dd9fd47c5b77e0006b173c8 7c1a6e5fac0b078f076865251ade050426cb5a615275a33ca49281ffa67772ae
5dd9fd489191710006b570dc 6d924f07c1dbb34337bb16450de9aa0b23daad2be8e5266ec468d87f0e09d63b
5dd9fd48c5b77e0006b173ca deb290ec752bff4ed80d832016ba487ce1e818aa6af62ecfdbf3568c90999c6b
5dd9fd499191710006b570de 66cc3e95f645e5f06f300acedfa262327095853fee56076f4f0ced2a7961b3ea
5dd9fd49c5b77e0006b173cc 45be1978f9a003f01c608417af54fabac91da086ec358aa913e0df76af01b263
5dd9fd4c9191710006b570e0 9dc17a4f7febf3b9056bef3e7333d27539abae56837bb366fdbf3b8a65e166ac
5dd9fd4ec5b77e0006b173ce 38835be278f99c92c236861d9be0f1f85b434da298277bdcb679b0848843be4e
5dd9fd4f9191710006b570e2 e231cf0d95ea5df6a1d4d418b1f90e387004af04c7506f9b67185894ace0c8ea
5dd9fd4fc5b77e0006b173d0 5df56f80df9a2fd552d46017bb2b110d446ab45c42a299099ec194ee2727b3ac
5dd9fd519191710006b570e4 44b65790f86da2e57e770effa55f07751d7195bc3cffad6da60a3f806acf7c13
5dd9fd53c5b77e0006b173d2 37ed3d5baf7ae2c3fc57a2972a7d2a5fe5ef34b19182e575034d1e95069da9a1
5dd9fd549191710006b570e6 5c2d4779b370b4a7af5874ede1b8c0932f7ba1b6792a648f940fe6e34b295419
5dd9fd57c5b77e0006b173d4 0936b8bbaec828a805243247f0da5365a4016694424ca8675d03a35b3ca6aaba
5dd9fd599191710006b570e8 62dd7098edf216b54d89926bf448bce6bd2ab858857b9a616623805ae557e52f
5dd9fd5ac5b77e0006b173d6 86c6cf1afd3a109dd3784a9192cb5a4e5a7dc7e2731a0783cb7628f303ab4fd8
5dd9fd5b9191710006b570ea d97ce0551180158669321c73fe375e645f08aaec05dbe6ac2a8c5419d77e6806
5dd9fd5dc5b77e0006b173d8 e3605dc360ae9a87354802ab4959da416c51072af030f84f2c147ab3df8a5a8f
5dd9fd5e9191710006b570ec 3d7e65b6c5ab298ab8b173e00d13b9b7845de5e77bb387e72c8a12f5b12d71c8
5dd9fd5ec5b77e0006b173da 1172e5c8bc91ab2b8a945fdbdc41e8a17d86b4ea795ea3a8b3e09ed57426ae32
5dd9fd5f9191710006b570ee 343fc3731ffaabab396d1ddba96ee467852536f18aa5d63efc65fffef97fd55e
5dd9fd60c5b77e0006b173dc 38c7f8cfb670ea1bb547f146d491d513bb56e4565417286f7ff262abce20248d
5dd9fd619191710006b570f0 7ceca2a644e50d94ff3a785fa063700659d45d75b2214531d3555dc02ee1ae85
5dd9fd61c5b77e0006b173de 0da6ed9fd830d766e55b512fd29560ddb562d2c1d567095df467afa1c4d68e1f
5dd9fd629191710006b570f2 5b5c27e042f30f3a49a6c8006efc8a43e516d2566c84d9fe52eee57263f9c59e
5dd9fd63c5b77e0006b173e0 b1a38a6cc1fe40fa9d30a9bf545f2a621d21e8ba81697637444745bf9959ba40
5dd9fd649191710006b570f4 25929afbbaff364fc6d8d67b668ebd6f869ed33db1f6c539c1d1623708eaedef
5dd9fd65c5b77e0006b173e2 ace7441a905747b714874935c3b1f910afb1a88e166e130dfdf3ba679d907aca
5dd9fd669191710006b570f6 1d50ac70d3aa3d7b9ec57d206129e3260394cafdac8d75b2c7a718ae037c987b
5dd9fd67c5b77e0006b173e4 ddaea5de2d238aa4c496f298176c55e510c752c327e703941c382c209e492be7
5dda0214c5b77e0006b17406 633e3fc1eb03e6cc818d540b1968023063c91b5f932fc9856b9722788096f9a6
5dda0216c5b77e0006b17408 9fae4a023fed68a8fbdc86954d4cf32ee6b2cc9012073a761ba6c76e852805a8
5dda02179191710006b5710e 5a7e518c3668902ad92ef853bde6f72b9cc5936867d10f068e8b90a0709a709d
5dda02189191710006b57110 e853272290198853780f9ccba0ab43d331fdfc069f1cb76e95a5fc5a70f7b996
5dda021ac5b77e0006b1740a e50bcd51f2feed7d57eeaa94aceea576370bab1a8073ca95cd65eb35631a5317
5dda021c9191710006b57112 5845f3bbf1042337e75278804191fcfa1f9786da34ce0d8d46f0be7422aa5acf
5dda021dc5b77e0006b1740c 63fcc219c29d93a40bb95e804ecd192933979759078c4958734ecc67aa5215cb
5dda021e9191710006b57114 1be2b7f75f2ccbe353b814131681dec939020b6276caeab417e9b6b0308324a9
5dda021fc5b77e0006b1740e 3daca08bf9de01039e281c7ba1c595f353b3fd1e592d94de465cd288fc231b24
5dda02209191710006b57116 9932359307ab0c4bf4b037fe6ed0f19b210e4370cde8499082c86b557b9cc11b
5dda0221c5b77e0006b17410 694cc4ba6b0edcce6fdffb6d7326c6a46c7b6e57e827e55ce6f732fa028b3dcc
5dda02239191710006b57118 56411fa68dfce81113823ea6b4e6a1c126db77cffbc60438fe4aed076176561f
5dda0225c5b77e0006b17412 57c820ffe939c2f9be5e2fe8281c59e063f62f2b1bf1f1694087c41f227a7f91
5ddb9632c5b77e0006b179b1 c42069fc6d85ec103768ada14e50977ccfb9b148c0af240c5378431ff370b096
5ddb963a9191710006b5765c 380fd5b76a2960abe7950c7786bf81006ec3a1c246af71096d3ec90092e0e37c
5ddb96f29191710006b57667 da245c7f873f84cc2c63084c24d57a3d7d9e10a78c66fe203d5d0093c8ec58d0
5ddb96f39191710006b57669 6163e0bac70a71adde5f6adc7322c5e20d66374520232dfd43725113f66651a7
5ddb979ec5b77e0006b179b7 b5e2cdb00bda3b664093cc71314e211f0d3695e9b05e54b98a47667662672925
5ddb97a19191710006b57674 2c6cf79e2eeadc342669f697af8dc3e395097e8919409255301bf3dd6d6b6913
data/site1/F2/
floor_info.json 1efaabd2362cb0be5d83213898f8c6e2972d74574a4557c40f8e67800f7d109b
geojson_map.json cb8400d8ea4c7f178f9733ad9cfa4e921df45d623f509f5c887280cf66ac4286
5dda04049191710006b5712e 58ac043598adc20461fad1453e1ec654d1a4ff48a72bcb0c9d0f88a7a879c18e
5dda0405c5b77e0006b17428 8e4888e6873f612ccdd8f6902bbfa65be06c5a648f729d3cb3f067823f883fdd
5dda04079191710006b57130 d3b4b751cfc85fd4d4368bf5257e8b7a6676f229379dbdba64714d4bb2047085
5dda040ac5b77e0006b1742a a352dc194470e0a290a0eeeccf4ba631a634b1439cdf791c5ab45ccf03081229
5dda040c9191710006b57132 396d3f3b7e4cf45f0cc9e54130b6e8d8433f7235796a6bec16360b033f5e5222
5dda040dc5b77e0006b1742c 54b538e4994c880076e1e1e909e0aecb54725821d129a4eb269dafee21f3c3a3
5dda401ec5b77e0006b176b5 15b2128a17897784f08d8ec192a22ff5ce4a0e24d15c0af9c16ab3639e556fa1
5dda40219191710006b57384 87f138f0e50e8619f1a74f0dc18389d39ffb7abe6e72e751e837a1a04c0b1751
5dda4023c5b77e0006b176b7 15ab63ac914a79983ef9815fb6c5f645a562fc298a2f15f9083373554b7042ac
5dda40259191710006b57386 5232fccd07f7686db4761858adc5d55294b34c814ad70cba753fbaf0ec657888
5dda4026c5b77e0006b176b9 313d8cf1e00616054f442bbeb76039af0d6fa3f9e0b6feca0fc5bfec15b4accc
5dda4027c5b77e0006b176bb 2de8b482b6165d11a4cb4adedc0db2567b619f13e40c0f6492e2bc28777bd17a
5dda40299191710006b57388 75374bf76914c8e31257e7607bd4fea1ffa899396892dadeedbc5d775e6defd7
5dda402bc5b77e0006b176bd 55b4197fe0283531f0fa9f422281cc1ac361f9575e39f41e27051f139e30aec8
5dda402cc5b77e0006b176bf 0f808275f1765b03793db4d66b1306da2658383fb58ffb9daad460cd7538d087
5dda402d9191710006b5738a c959d75763eb67620e0c6e2d4265b90ebede9768e65afffe25bcf884a1586a76
5dda402fc5b77e0006b176c1 af393ec86d40ad2652fea80932d12991aad2c72407a5f66cb8287a53a8a9cfe0
5dda40329191710006b5738c 8f1241352031dae0b233d760d12fd1f31383bcf1b7fbf2a8fe14746c647a09e6
5dda4033c5b77e0006b176c3 e8b8e2824a837efffddf7b8ab99063202aea83b6f1dcf22b3981cf1c2104d0e7
5dda40359191710006b5738e 4f228f8df1ff8f71946bf2b853d129f34f5b3e5523e8670efa65a0b132a3dace
5dda4035c5b77e0006b176c5 212a0fb31ed9c07baea83dc430018cfc25ca9fe3fe423e519b448265172c7077
5dda4036c5b77e0006b176c7 f5752b9fe5ad3aba05b559d8fe32f22e45072b1b46babc297830e08f4879e38d
5dda40389191710006b57390 d4b5845f998ca1d455411093464e1f99016e52dd29ac3dfb3121bf9a2a250afe
5dda4038c5b77e0006b176c9 f29d02d95d41531bf849e9b5c8bd3fc70707875dfe26aa84a85866188dbc78f6
5dda40399191710006b57392 5c6a21fe5b32d026715ca565023b70d5c931139c193c563073680013fe4b43be
5dda403a9191710006b57394 d83210322ddee01bd418b3ed36593bce5f1b0051bc55f413b33f1adf1310f8f5
5dda403ac5b77e0006b176cb 4f5dd1c0bd2cf483ac669896cfe1715493848717a682e4376339a7920715bb7a
5dda51f7c5b77e0006b176e5 0da9191d648b8f6e54ec8b69cea6756e7c89705969cc1a4c000f8ad8580c81b9
5dda51fb9191710006b573ad bb60cfdb4d5a16598b5877d262ddcadc3c685cd13bddc2a280b4ddbe61f7b221
5dda5200c5b77e0006b176e7 02339e9a8d3ea6722526452594e35329dfb0440484a1a61e1b99bf3d6e970c58
5dda52029191710006b573af f6b3389e48961277dff0e219a736c5d40708e619b1cbd80167748f31fccd1454
5dda5205c5b77e0006b176e9 4b5a31ce50b698cc0534655f10c3deb752767bb2954a684780c94e3b4116b6ac
5dda52069191710006b573b1 de5d451ff47de5d3888bd35b86e37be2258b163b7c27c6abe07b91583b9b117c
5dda5209c5b77e0006b176eb af4d29428dc77cdaef8d3279f75ef070a795085886f361bda2f59bf39050082c
5dda520d9191710006b573b3 44199dd767a173a787154f67f06807e9a17791c3204441588239d48872231e59
5dda520ec5b77e0006b176ed 448ba6b31813d84f350ec5655bacdfc484ddd4dbb893e8d3202dcc8d21b6a6d4
5dda52109191710006b573b5 010c526bc356fb423a43da4d82d289f209ab9ea3282ddf22fdf8c58b22368263
5dda5214c5b77e0006b176ef 50c868aa17fa7b827e1cc0e584b014a9056883a726061d09af1a1dfb9e2ef392
5dda52179191710006b573b7 3c1f86c72e17bbc5679cf1fe2834f233e0fd2a59fadbf761930f28794107d044
5dda5219c5b77e0006b176f1 4e9102b50b0153e513bf4c9a7d0642237a17f8d5a4a427732e567282d08ba267
5dda521e9191710006b573b9 866df4642a26d3673e4c28d3646b7d4a897a4d1a99e5cf1d694a7bcd041e9d7c
5dda5225c5b77e0006b176f3 70988e4ce6034c33da6a20febcc1d0c9f5283d72fbadc186e0a5b8470495eae3
5dda522b9191710006b573bb b39ba5f9d70f2fb392728aa817d737c47482c7b006fbb8debbd07f710ad6cc59
5dda522ec5b77e0006b176f5 2fe9605fe97f644e62bd5b69ed38cf56317b4499e48ccb79c67e0fa60115cd0a
5dda522f9191710006b573bd 8b5643f69ed46fdb82d8d8f157e1906651b4c78a01c43dbd7c22e414301a4bd1
5dda5233c5b77e0006b176f7 fa4788720db19ccc2ee816f5f4a137c9f8242bfc689f9f6e3c04d59bd4e97d61
5dda523b9191710006b573bf 6a3d85bb9c225114d9b393659543f9cdfaebeba4948e0d061e447135e1a65039
5dda523fc5b77e0006b176f9 7b93ed4c0610d10064abaa0a00b1527eead36638640d66f209aa213f310187c1
5dda52459191710006b573c1 877b2c5fd09270e82462a5be52d4ddc785dd978eef3dab05fe204010f71b695e
5dda5247c5b77e0006b176fb eb61eb4e6e435d5250f199487c7c8b1583f9aa4122adfaf675ee323f64365b8a
5dda52499191710006b573c3 ee3483696becd7015f8323980bea060e9c650417eb2041b59b1e9c1d61540f19
5dda524bc5b77e0006b176fd f70e543231a3fb0ac39e7c4ec31d02f8a16080539f0d53acc424a912e5e4e3ec
5dda524d9191710006b573c5 1866b3b91eb86bb79835a5a2cf6c1cc918ac971d05f7639e57ee555ab1e7916a
5dda524fc5b77e0006b176ff 14f510eeb7e246092faf9a502cbb141e1ee766f0e64f9294a950324ab864346d
5dda52519191710006b573c7 9ce09e3db7143b26264d2353535d7bbc5716491036d3c4b54e4d315bd40ca8d5
5dda5257c5b77e0006b17701 4b39622b799d15b376e303ee783dc373390c18073dfbe07ff34f61aa25af6f62
5dda525d9191710006b573c9 f5c1fbc061cc39f196dd6e1265c1092a51580d8e55db9acbeb83221adbe44a9a
5dda525fc5b77e0006b17703 ae9d34fce9cf5b1c5da0e730443030740b42025dde3bea58795e208714ed1518
5dda52629191710006b573cb 87883f8b64ad810e6806a394d641f1ed51202cc8c01e171e7ad149a1e71eea10
5dda5263c5b77e0006b17705 d8b5cae6f79c0b2b94b63d01137a8db4b97309be27085767d5e02ac53503faa7
5dda52659191710006b573cd fe5e2b22e7791d8de5cf53bb6ef2be9076b6bc2678350f59c55e851e59082ac3
5dda5266c5b77e0006b17707 c5f2bf7d278381b50ea27109f42e3edbe98d9ee40a7b5366b7f0c5056bff5e47
5dda52689191710006b573cf 94bfa56c482147cae84b76bff942475c2bbe9e1d2d69ef3019195891399d6f49
5dda5a83c5b77e0006b17709 1e00a303beeb3f840eac4031d55dcde0b58a017261de0e6b5299b351370fe9b2
5dda5a8b9191710006b573da 301751093d3e9fd962d98893c479aaf20aeabd9218d5d0d9c041a51987c0eccc
5dda5a8d9191710006b573dc 1c318a690719c5819f0b116aa20f62b88767acc3213c8599057cfa25ff20be9b
5dda5a99c5b77e0006b1770b 9e24e1424cb6dc265a6344273113b2ad084fea6658d80de738775172d417069a
5dda5a9b9191710006b573de 486703673844311e95efe5cb3e7e1f06f128857bf45ea9ccf9a522a4838e77ef
5dda5a9dc5b77e0006b1770d e4f8516813f3015410880384c14d15f86c216f700d05e3a5bc2d94ca5d300685
5dda5aa09191710006b573e0 dba4d0ba5ce898b806544376018025942b5704305c7ff4ee0a4aa02f93ecd6de
5dda5aa3c5b77e0006b1770f e94e68c730a1b2f60d4adfc62e5751cd6a5992d73738abf8154cc3a3374c5c57
5dda5aa59191710006b573e2 b5e9bd29d1fcd5f00823983d7e36ee02451fca8dd4bf59c335e4678a0fb10b1d
5dda5ae49191710006b573e5 feacdd92292f0fb124cb39f05a2764a138d401de269e3af0b7c6075bfd1d1f6e
5dda5ae5c5b77e0006b17711 be10a0d80c616c405ab4d7121f46d0cf470f4b47d2611340abfa0953e2ae3b27
5dda5ae9c5b77e0006b17713 e07605c79e7675604114c534e5d2d7e9f040d6d0178335e64f8eb6bab2a7bd8e
5dda5aed9191710006b573e7 f101267590bf7e5bd6138bbf3c8ea64afcc21388003e58bd2f94375c006843ea
5dda5af09191710006b573e9 eb9e017415d92904c7f0fec95f6d86337f361a01cfb814a98b58f27e5d31a31d
5dda5af2c5b77e0006b17715 f9a43b49b65149a558b0d2edea2ab5bdfa3e18dc7a46e7e39974ec37713fcbcf
5dda5af39191710006b573eb b23739aff0e6665c04522f8f9f370a329598807c457edf2efc7276783ab9c57e
5dda5af5c5b77e0006b17717 595b306b11b9b7b1cace886e02a0e78ce267ccecbc825a7b8b5b3d060c607048
5dda5af79191710006b573ed e20241a057d708168ed38d540e6e3b9bf3ce8c27175a7ff3d2d24cc69e2ac2b1
5dda5afcc5b77e0006b17719 600fd287f09bf0845638622242a0e1474a33ac7a1ccb43d1eb287f65283deea1
5dda5afd9191710006b573ef 9229beb289b3120f4b292ef7a643dd96e488178b3226445988857eea605e32e2
5dda5afec5b77e0006b1771b 3b2ec4e03dd7a2e1cebc126dc0d1f237137cf053fe3508c32e88aa0a32b0b0f9
5dda5aff9191710006b573f1 28e3833c8f2f677f28596fb3fa534ba210536f71cf46fa17d3b2dafd50412271
5dda5affc5b77e0006b1771d 5bff56eb6dfecb5c113464e4108868d382ae913a05ee306e6951f1ca95880c30
5dda5b009191710006b573f3 330f7a08e674d97fb3f184fcb3ba5f20292751ab2381af04d71d27d969d543f4
5dda5b019191710006b573f5 118c6081e5a4de16e784944427e4aeacadcf7b674b80c9d14bc99529e56d19dd
5dda5b01c5b77e0006b1771f 553101e8cfa1c6e880e4bce9edb0bf84dc74b00ada0a70fd538607ce4ab4d7d1
5dda5b02c5b77e0006b17721 626492bd3e4c12e921112e3ac907c399e37d7e95c27e0a032e0b092aab540f5c
5dda5b039191710006b573f7 3d22e18aeefd268bd6681560607f100ad35b606ed83f3ca7690169b06a13ff16
5dda5b04c5b77e0006b17723 5887662c0bf039770483e2f39fe99a5040fdc37b21b74462ceb18e894592ea08
5dda5b059191710006b573f9 42f215e86cb1965291c3178ca5a44b27d91bfa54e021f222f7dab594f3d5fae4
5dda5b05c5b77e0006b17725 bfb30643711eef2c803b61db3f3d8894742108c7affe103e53ebf2070cb86497
5ddb98f59191710006b57684 64c899dcba7ed40ff3de4896e884c431d723c9d360984f63a7930d5b77419d8d
5ddb98fac5b77e0006b179c3 7c00ea63daeca308ab19de3eb846b5921a5cb917a08f0fb96ef2a6b6d6ede8eb
5ddb98fb9191710006b57686 00b07b0d00effb78720ee7ef5f37d183036ec82ddec37e75f0978b6e12e39e9d
5ddb9900c5b77e0006b179c5 61ad31a5791ec60d7b546feb326af198bcd387e90619c20148d74e346596933e
5ddb99da9191710006b57688 20824cd348ba3e899bfbb07d4d2a8e41e03e40f31620b16540e623a39d875ce4
5ddb99de9191710006b5768a ea0a548f65de72936eefba0bb72344a2551b80e7111594e9e311bdbf0f6d6efd
5ddb99dec5b77e0006b179d1 ce0e3a7bd428c67eb7896ff07078854804578a36c34382a5c58c170e770fe3b5
5ddb9ae59191710006b57696 0ebdf94502108c6363c3b15fdd2d12ea79f1762845899b6b165dc6581f3897f1
5ddb9ae6c5b77e0006b179d3 766c22dc9946905c941930a985dc36e727d68db143ed6b08581e714cf5f34f44
5ddb9ae7c5b77e0006b179d5 45239026de9a6853ba1953903d44d7e7bf7cf7166bb3d452188252b9407ec899
5ddb9af09191710006b57698 6acb3c021de82e7080caccd40daae16f412cd7d836e51793c487ff6fc3e43f59
5ddb9c639191710006b576a4 faa6ad012f4c2eb5c5f538fde7b0142600ed61b2e830416cf20def4b2d30a83c
5ddb9c64c5b77e0006b179d8 f74361087b85965e4260f05798af1a99fa46b48dd05d42f5d68e07b0b94c6ffc
5ddb9c64c5b77e0006b179da 0e76204e9b3968a75c2e1ce13da202ffaf0268dc8696c07ffcb4443eca5f503b
5ddb9c6e9191710006b576a6 555e39c30360dd977f83df18fb3eb7c04e84d2b5de0b544455c96c227aa00bf3
5ddb9c739191710006b576a8 65922c8f913d0fda342e658e9be06d63250da83a63ff54e5c9e3cabda448cfad
5ddb9c73c5b77e0006b179dc fdb4504f40d438b9364305b97d292f6c10aa7f3aea58b0c3062f1df9aec4d8e0
5de75447376b9d0006fda9ce 227409f0a84c2d78e3b42279b70b9320e5abeb7d9a1abd20a48911ef009028be
5de75b78376b9d0006fda9db c452a46d60c5fd2c80216d1848be289d31c39907350a55120aec905d605a85f2
5de8a6d7376b9d0006fdaa0b 9b17a7a1e8948cdbdad54e22ec10f766e479ba9836027c8f4f4e5544d090741d
5de8de33376b9d0006fdaa59 add76fa83a17814b305a802a0a5b18b3663277287ea4c27628cefbe48c84a360
5de8de397491b00006eaaff6 aa35d52d25aa9724e8aa08c2c6c50771032e980bb08f89088ea6f138ebe7fe36
5de8de3a376b9d0006fdaa5b facb6150302c016f710281354cd571fddf82c638fdec67eb76e826796dae5be4
5de8de3b376b9d0006fdaa5d a12561a0111ac3a1dfce5a6a4370a7990af3b47f198fb4b1ab8863ff7fb7cadd
5de8de3e7491b00006eaaff8 d3a21551cf85a23ff0342aec4a2432e2d0e31a1bb4bc4a71f5055638abc262d8
5de8de407491b00006eaaffa 426f655be11ca2fce36a0e356fa9a5ae92e3e635cc9c0cf78e38bc60b9ae672a
5de8de447491b00006eaaffc 9b6f56ac79340d4dda0d6d0059647552298e18aaf3af0a546ae48cfea6c44f7c
5de8de47376b9d0006fdaa5f 04d1d631942853d4253022a4be9866d8a726d4acc36426e86a0624c5120ce7c7
5de8de4c376b9d0006fdaa61 cdf77c1e9f7e6ab77a986e22c3bb972803ff2ef96a5d771f155f6de3ab947a2c
data/site1/F3/
floor_info.json cd984f024ed6476716ef0d3ecdabebaf55c72bac581311f8a30eccc94076c7a0
geojson_map.json 5e5aa4b3c774c6109746d0a1dd184bfdd370fecf2855db436e9e91ae4e033acc
5dda057d9191710006b5713d e4e0367ee24d2e2d03ebe75603e5dbdfee472e86c675add9e5842bd26050f44c
5dda057ec5b77e0006b17442 0b9180439caaeb52e8ee25068a25e93a67d85184db8ae7e0c66270ef0c34fef1
5dda057f9191710006b5713f 4af48cd6668e323ff4bc6d422d566fc9fdbd6703b3bee43590f89a5cfbb3ad69
5dda0581c5b77e0006b17444 863f05b3ffdf4f3e78c2e7e536f64b79be6a1cf4fa4441fd44175e0e3cd77607
5dda687c9191710006b5748d 0df8d347086e493fffd5e0ad3bc0fa7b1dfe2a7e09d12ee978ad62dde3dd5d82
5dda687e9191710006b5748f 48d9d64ab4a885b0c8ecd68bd6c7f3743d10cc468e9a1e3349f0fe66223f5d81
5dda68819191710006b57491 232ca063fe9a6c31573b09196427be17f73aa227652ce5797e1f296f973025fb
5dda6885c5b77e0006b177c7 77168f83b5f09a61534abf60f925cdaf952c8728db3c81d02bc70623904c6852
5dda688b9191710006b57493 386f2114f286dd742096a641d5f16af7efdb4ec91fe210168c737dd2d699cd82
5dda688fc5b77e0006b177c9 413fdd2acc5d76105630eb69f2a3d77527e12f6d14d6fd4011d7b4fb00c458a9
5dda68929191710006b57495 6ea64eabbbb9d5532d1f4de7800239f3ae55179e0182664f2b55289f2f73d2d6
5dda6894c5b77e0006b177cb 0bb43a71064d995c7704dae409b27311d2052be80c36f9cc53c75d140bc3b9d3
5dda68959191710006b57497 705654113e08874f807839e312425fd9cd0f4e7d51772489e2558b1b40720703
5dda6897c5b77e0006b177cd e8807e237c4969ef2ed6b4d788910706327fff0e0c687bee475c90ff7bf70727
5dda689a9191710006b57499 3f7c811636bad40d66c268ffd9f298af70cf9de50cbc0b32411561e8943cb0f9
5dda689cc5b77e0006b177cf d9d51072d5e09085a598c18dbca82e8e5707966b43bfbc916f873a12c2ee2fa4
5dda689d9191710006b5749b 0d645efa7218d35e99bbb4a643af7b534906c3a6e6ca9ce0c30317b19fcde570
5dda689fc5b77e0006b177d1 aad0a8a60d5e52656ab620bc0d2c7152f27e510d034ef0ee1b50b989a0c6bb83
5dda68a19191710006b5749d c1789ce74ffa80a06c83f0aaa4dbc257b7fa6df76466ef0836d15e8b933e815f
5dda68a3c5b77e0006b177d3 e60d278af0d5006cd16a378434b8ad8914a242390d7319ae6baeb929aedf593b
5dda68a59191710006b5749f 0b6e11e27d4acdf84a2157df45fea7bdff45e268050172ab97aa60f314e74810
5dda68a8c5b77e0006b177d5 6a68f3aaffa8b4885efc004e5a36e7d00872b659068a60bf3102f4d72a6081aa
5dda68ac9191710006b574a1 4e2e004413517a4a589c7f4ddd1d005466c4936629c8c94f90329e1adb8e9341
5dda68b3c5b77e0006b177d7 fb028a796275fa82d948aa2e252d145508009edf7d047860b853c5b7d13c20aa
5dda68b89191710006b574a3 c16c2682ee363ab0ce353464687a7cbb17c8e48f9102860d61b210cb7b38ee0b
5dda68bdc5b77e0006b177d9 c754a2ca824e78b34c5826d71ff97e7875e8789ea2178b7ff2459d5493e79a47
5dda68c29191710006b574a5 02754300ae26f08a2494c4719e1d335a41376c8f1b17e5d6ae54e07ee11261a3
5dda68c5c5b77e0006b177db 9543d2ec7b46175ec6dee052860f779ffe0c34e16efdb11f81bfc562ce2d91c4
5dda68ca9191710006b574a7 202822bfc6d07b064342eae266cd543a3ee5f960706ca8a935682bddf74d7e5d
5dda68ccc5b77e0006b177dd 3c78148c7139e0b5edd4e15d0760d088a2b3189b106bfa95f74f5a12a616afd5
5dda68cf9191710006b574a9 c85089cc32609b1bdd68bc4ea4843680f6a59295dc4fc81987e66ed00bf77b81
5dda68d2c5b77e0006b177df d27475ed6aeaded8aa6cf14ae6eadafdb0dbf5ad8daf6f4358916fc30fcf4caf
5dda68d99191710006b574ab 155ff9cd1534d5bbbb024ff34d7bdedf80977b30e162adedae6608317991fcda
5dda68dcc5b77e0006b177e1 569467b108a5744c4895df0f442a7516b18538de085c15ca7a931ab2841f487d
5dda68df9191710006b574ad fc3318cf5d943cf78cfdfa8cdeb9529421db7bd29f11daab27b49de935b39282
5dda68e1c5b77e0006b177e3 fa5e1312b470f67cc6c2ecdeb220d2ee6464a2eedc9f1b6f2bf50652d7188ff3
5dda68e29191710006b574af 735d18e1f0f084d9b0c93db08a2c30e70ec1a1954c13ed5b10339be9198a5bc1
5dda68e3c5b77e0006b177e5 6263430774a64fd9cfb188e9b1f52ecb89bb517b7939f3a3c624e9f06131c431
5dda68e49191710006b574b1 564ae16d3dc5e07fc7ee4fd14885d2ad926b1adc6472310f6519c527bf54a050
5dda68e6c5b77e0006b177e7 adfddd362848de029bbebfdeb7af4704b6953d5bd6c19124ff637c9aa80aecb9
5dda68e89191710006b574b3 7bff138d89fcc80962521904f42feec728477b6168b7e6d1c156e03056063cb2
5dda68e8c5b77e0006b177e9 a8f516cca4e4f11c080d1454d20997fc1ca48bac0272b65ee84f64c08c9a6546
5dda74199191710006b574b6 47e5b617629efd99622d9c79e8554e90b3cfe09f51b686c02b2b8da24e0558c1
5dda741d9191710006b574b8 6ea73f435b02b461444f828a637acff0e2bc39f92d4c6d0ce50f3e6f8f78fb5f
5dda7422c5b77e0006b177f5 80d84be40e798a76fd8c3da677821bc79dabac196b9bf2eca81e55841eb12a20
5dda74279191710006b574ba 93e55182919e450ba50b43aae55d58c16dab3b883b04518c077e5507b9e194b4
5dda742c9191710006b574bc 332d229f5d18627787489fd2321a3b6f829be64c0f0761a4868027d57405ebbb
5dda7430c5b77e0006b177f7 d094de98e8f8f5425abab3bf607065c68e7f5c993763fffd2eda98126ffdda9f
5dda7437c5b77e0006b177f9 8e8d1f40d6a0c31345ca7cac90d77dd0d3274fdc67eadcc2ccbdd197f2fad92a
5dda743e9191710006b574be 569f4ab2d04fb089ca518bb7228e4cf02acf380b02449cdb7c29b91602a1f0c6
5dda74429191710006b574c0 6edf1e46e2d2878163ea52a6aa1d290c5af82e1c9410320f11f283f352ba15be
5dda7446c5b77e0006b177fb e2cc3a5ade52acbbb4d6580343690bdf9a0fbcda799b7d36e68c490187712ca1
5dda74499191710006b574c2 67c19a9259dfe66452e0b34ef34efd5b435cfe97822962b30e0a1d17351a9296
5dda744e9191710006b574c4 8cd859c74af8a6fbcf00d5ce6385a52f3ab31ae47c2425145056bceb892ff60d
5dda7450c5b77e0006b177fd ee34b740b54023cb57523e3d17047b0cbdf72cac68e6fb9842f9b89e8fbb52c3
5dda74569191710006b574c6 b6f7bdde69d739138cb006a9fb6437b47363eafaf0ebe3d197e98830ec1dfd64
5dda745d9191710006b574c8 7251ef1fd34764449af6b1d750601f029886acb7a9f505936152784a90a490ae
5dda7460c5b77e0006b177ff 816cfa931a043133645ead067f18e092f06a5382bfa6fea34647d046932b434e
5dda74679191710006b574ca 15471c071ccf364a9acd7a825e00666bef0edb6cd7fdba10a716d46f82cc9564
5dda74719191710006b574cc 65210549a74c7a68be42b6290500728ef9bb6f6cabf2e1cf3e76d9baf1a65b11
5dda7475c5b77e0006b17801 8e4b5e982da259b4d8ebcbf66ec451e555d7196ee4d938dbc62bdbec8eeaad18
5dda747ac5b77e0006b17803 2775c599aada789d268483cead8269b92d12c39864c4834db60d5714f7bd6045
5dda747c9191710006b574ce b5fb8f8c85321e8a3110777acfa0ef63876badc0733e866d3c672a3de4b7e18c
5dda7480c5b77e0006b17805 8ead06ce9ce464d466c8772b9c3e9655c4ebf260a3fcce9e74a8ca5a5a9539b8
5dda7482c5b77e0006b17807 25734fa54c5f69dd2db7feb4f7ff76e302ba52f250d5e2c8b5e260f1cf553b15
5dda74849191710006b574d0 9af50a7c2c16e7b5785bff632609ea72a822e041b382f8aef43a3ff2e5e938c9
5dda7487c5b77e0006b17809 f8330f718362d465e51ec5f5dbb3d8a2e89bfa36d48a36f322835f3ea7219952
5dda74899191710006b574d2 478dcca3384d2e8106f537d54a5d7b61cf45046f3d3b87d0ddfb181e6df2d2ef
5dda748bc5b77e0006b1780b df57176d2d5fabf0eeaeb6decc9664067eb798c94bce2e3adabe71454465e60d
5dda748d9191710006b574d4 f9411b6037d859ec0aec86b12f00fb7eba7bd3517eae9cf4b874cb3c2ca32072
5dda748f9191710006b574d6 65add7cf6045e632543323975b8706303cc14ade52ac9d2a1d915fb57a9adaad
5dda7491c5b77e0006b1780d 0efb1f7381064b7ffeb2abc836700b110f54e56bd8d0b6cc26b5ee4aff26f378
5dda74939191710006b574d8 70e72da80e2a822546222aa00112e7e2a76fe73911c2479a01c0eba2a20a9bbf
5dda7497c5b77e0006b1780f b782d628cee61134ab4c161b95b0dd1b76475928e3a89a037ebd9b2459ba3dee
5dda749a9191710006b574da 66a11db7f64e55e13645676501ae2e6ff9c1fb7cca9abe719cb74e163db86df2
5dda749d9191710006b574dc 2533fd3b45f2ecca5ac5fc74b7274be6fd663617d697a22f25d208db2d3c2470
5dda74a0c5b77e0006b17811 45db3c0e406b5738f09c3d347e829d7d2f38a9e644fa18eb3b2075998e1e679e
5dda74a19191710006b574de 242f6c1800f4e24c18fb7fa233ca01ea808c33d9c15a70688de182be6a24d382
5dda74a6c5b77e0006b17813 2415e786e1d3939320aaaf41456db44853d4f108068370015a9335bdc7932229
5dda74abc5b77e0006b17815 1e80b14854a3cf924bcd177c1d05b4f037ad150f0b5b6988295f3a6d8dae2455
5dda74af9191710006b574e0 253447689d84d461981c3927f6e0cee783bad32a2beed5bc9856065a0b7d3faa
5dda74b4c5b77e0006b17817 a5653dd35bd9e6f6a7802b2e28076a6bea131bff63f15a5d5fd41e3e49e19e41
5ddb9e19c5b77e0006b179de a7ca7be081345d48fd610a3a3fa9f226194e037a5529cc4d4b96b2215fac2c27
5ddb9e1ac5b77e0006b179e0 486414580d9b63f1f8a8bd7adacaf57af5af4015eea39cfff6f7776ba7ccd44a
5ddb9eb4c5b77e0006b179eb aec9848320939f4bb9f69a3d466c9c123ecdbfc30bddcc5e830d30fad4d997c3
5ddb9eb69191710006b576b5 0fd3ca650ac0c7b9497906e5fcd2768a22856f7163f8f6bff431b7abc0209f9c
5ddb9f739191710006b576c2 1ca0c87d75586b74c5b33028c592baeb6d7ee7ebda7a51d0308d44f4fc502d6f
5ddb9f749191710006b576c4 4c134ade2e6b69c8c6af912fe20878ecef7d7cbd9e4ba1dc8fd468deaa26db31
5ddb9f74c5b77e0006b179ef 937cbd904a8d030b55feed804858f84bb505919453b7a81c2cd7f628dd6ca5e8
5ddba028c5b77e0006b179f6 946aec6a7281b678358181791c9182eac3434dbd39796946caae7184fc2e7687
5ddba02ac5b77e0006b179f8 4e2a825fc280c004db46309f250aa30099033b2953b0c986ed701f4ef7d44ff8
5ddba0cf9191710006b576d3 7c356b0f7dffc801c76d7c8aa1acaf72176ad660864faf6bda13694f9fc9c18c
5ddba0d0c5b77e0006b17a03 0eddf56aeca9fa113b1eded0d351606d3d31e68cfefdb18240e0ca3bd789130c
5ddba0d2c5b77e0006b17a05 ac131780e34918e0824b036de51032695c6f7162dd6165f6cac1d7cdb5e6a3e5
5ddbab37c5b77e0006b17a39 180f62025c71df28029e7472b233e82af756fab880d9ef0da33287cf6d65bffe
5ddbab389191710006b576f1 411822dd2959e9bfa4c5d03853cffd3ba615d6a9dd64ddee8f296f5a243eb7d7
5ddbab3dc5b77e0006b17a3b b56197440609876f190cf83e0d2344d77ad4166da3af239f22be71b005de28f8
5ddbab3f9191710006b576f3 e61bbea5f6f5a3d292d61af37cd2fbc102d53d5d03ba99624b3490ed71834bde
5ddbab409191710006b576f5 8d41176fb2c67efea681109b4237de9fca21df02e77dd957862a180af444ac05
5ddbab419191710006b576f7 7e449160706ba12ebfd26f3e32e9834dd8d34ef14df9f1d559a564d5ca33a48e
5de730de2a632c00064f7aa4 a74515257db89e86eefba233fea8e4cd57a225cfc2950c74801282a3288ac766
5de730f42a632c00064f7aa6 f7b0d7cf561ed2ac78d8baf7194f8729f2032feb0163b46cdd6555df5d95928c
5de735722a632c00064f7ab5 20a1a37fedd5ff4fa92914f56e0a1cdb3a0979b30e44bb81cfbf1b1866b7d52b
5de766127491b00006eaaf84 1db100500c0c6c1bea988b542ddfaeddec2ef01cfd0920cf195cd791d77097c9
5de7663d7491b00006eaaf87 1db100500c0c6c1bea988b542ddfaeddec2ef01cfd0920cf195cd791d77097c9
5de773827491b00006eaaf94 4de4a0ccab5989898110c81b3c618ff19012aa464873c7391160f24d745c32e1
5de773a37491b00006eaaf96 698b07190c6609d1b5d7140cc8f7848c2903476e79ef4dc3896914a1ea86d655
5de779e47491b00006eaafa6 397836918bf2b79d6fab55ccd6c14fc5dc9e96ceff6df98c1bd45ee935d8f3e6
5de779e5376b9d0006fda9f6 bb6526947cb25d45676d9f2b28c939493e31a2b001bc7889519b0dc3a1f24969
5de779e6376b9d0006fda9f8 e855d5cca807ab4a20702a7bf5ed7e3bec81c2bacbfda556b4ef01d4027a8180
5de8a46c376b9d0006fdaa06 9017a852e20be484dbcde90c0018c4de0fde223458bc615e530540a9a98f586f
5de8a4757491b00006eaafad 99296faa17bd20f18d77f77f7c7bd51a3b839b2d24adec534a3a8c95d8c6ba60
5de8e4e0376b9d0006fdaa72 91648cdbaeb49a6ed107b40e4875abcadf6b605a2cef0523b36c9171ea19c47a
5de8e4e47491b00006eab002 00894eaab5e2ba493c5b25bf14a66eba7fcabb373ab8ab5d46cc7130c90bc957
5de8e4e5376b9d0006fdaa74 8451da7bdcf186023729107c6e4f0805986d97fe3311556a146d8701c1507264
5de8e4eb7491b00006eab004 a404823bb1a49cfa784f7fe95e3d01e2497933462fd20c47d3b4ec75b692fa6a
5de8e4ee376b9d0006fdaa76 4047000f1d3b548c75eedad851954af28a70a9994eda6fc218478bd8dc8494d7
data/site1/F4/
floor_info.json cd984f024ed6476716ef0d3ecdabebaf55c72bac581311f8a30eccc94076c7a0
geojson_map.json fa125266374d2867935ad1d0ed29b3ce1e95c8b42a78d337e4afd3b246406298
5ddb6533c5b77e0006b17902 99824fc866e9cad7f950356b1116ebfab1c04c572d9855f0ec1639ae00e4613f
5ddb65369191710006b5759f e94db49fc434823d91d0c601f420720739487840cd8599437e2590ec39cd9e4e
5ddb6538c5b77e0006b17904 d70099add5de5cf020250cc0b9f14c86a0df7314b507300b2e01c7ae198cf8bc
5ddb653a9191710006b575a1 34741314e470188823d5968deb31a167d2133dce66769f60114ab3eab3a465d6
5ddb653c9191710006b575a3 9d8937806ed67e4c9cf36e1d7dfc2fb8e62c8f7673a6d598c152ea70a321a27a
5ddb653d9191710006b575a5 136eb38cc6897a7edef80e6045113e747de463e1041bc867ccb2ac580153a31d
5ddb653f9191710006b575a7 36c34ec6f38ca58ab5ee08ebae7a2cf5cf21d86fcd7b79baa63863493ebcf5fb
5ddb653fc5b77e0006b17906 fb1da5dd40210e953dd2d7e1536a6b8f1f77b744dc9d795acac83df87be501c5
5ddb65409191710006b575a9 f4b391393812895395be018b841f4cbb9d8aaae985817379a3db6c22c93b185c
5ddb6542c5b77e0006b17908 406ea7b57cfcf2fd6f4c0aaef0ddef80299ec32ab0c24dfd45c11868a3e5dcaa
5ddb65439191710006b575ab cda3d746080c08d4f2bd6ab252c6ccfdbb482fa183e7c47e1af3972fe1194d27
5ddb65459191710006b575ad a04d0bcf7b50aa07e1a179aee636deda676cf19b4e5aca0c812595fb86906030
5ddb6548c5b77e0006b1790a 58153d3119337c070dae3c40087bd39810c08fce59b3d37ab3a99f27ff01b9a1
5ddb654ac5b77e0006b1790c 1d20336fe264ed8ebc674d20622afc4b14191e64d330a75992b2901483141bda
5ddb654bc5b77e0006b1790e 82d9f922563a0f810efb8b8f310726bfde0f28e38c697d7978c98f42285bcb93
5ddb654dc5b77e0006b17910 0ee69e91c3953e1ba58bb66b8068de26a55c300d8924fc9e9f282928d0caea3f
5ddb654fc5b77e0006b17912 f204e7b6cffde80d3434eb76e38318dbedaa6bc0c36fa32afa74a4d138122848
5ddb65519191710006b575af 3cd43d4798eda0d5b2856a02cf38b655c74cef7a50aed987c2e1da25feef6304
5ddb6552c5b77e0006b17914 857810ab4b695e44778ba68ac8d7b95fe2efbaf097c3967ce51d84874ed05db2
5ddb65539191710006b575b1 8f5f15ec10a0d8adca1f38ed68697a814de53f6791e216921493dfd9e53c4526
5ddb6555c5b77e0006b17916 f56beefe82d5e9c95382d1bfae48e7270319a42db99dcff07d3400d2e51753be
5ddb65579191710006b575b3 be01cb7273e808b13f2085d30c98ad8c15732d9f0354ad8fd8618e91670e093b
5ddb6558c5b77e0006b17918 933b87bbf0e3c51f0e53240ff65517c910c2f4abf0662fc42efccf5f281450a2
5ddb655a9191710006b575b5 0ab77ddbe26f1e47242b84962a8d318c35a6b300b8f8a8909b2c427ed0b5fda9
5ddb655b9191710006b575b7 e37fababc2529000dec23cb058fa0efbd9658a3e7be6d454f995870c41781aa3
5ddb655cc5b77e0006b1791a 273888c561e17e544055ac2cd45cad22ceb923c52bf461a3baab31b92ac23b92
5ddb655d9191710006b575b9 6056224caede9c120929e85e50d29c85e137fdc6e15946e0cdd8be0f05c3786f
5ddb655ec5b77e0006b1791c 76eebc2c4179b1d8b183f0f9a101ceb783a2a55f849ad7a85673c597dc5fb1d2
5ddb655f9191710006b575bb 43d420d477f520a5c27fe8f5a9ba31a99fccffa3f01def338cf7f3057bef1652
5ddb6560c5b77e0006b1791e 4c53ddaa85746d2684fab475b89e790661e7cfbd34f89cc515eb88b7bec9cfa1
5ddb65619191710006b575bd 674b026ba241c5282e5cbc8b68df038a7a904e34a5d94433fc3a9becb74981c4
5ddb6561c5b77e0006b17920 7fcb5ea5633c3b659c9ac1ac356c0258b8723711cab2796df5061e27fa69557f
5ddb65629191710006b575bf a7193621a7397612e30b91563a19db63f694c34ce5b01e79e20ddb16e7b2132c
5ddb6563c5b77e0006b17922 0764496fc8117f788f765a7b2fa4ee22aa96d2f97c7390152b13aab1923304c2
5ddb65649191710006b575c1 42bd154c0bf73881fd8047871d2e9b222f74f075b3a7f1c3295f5a05589afd99
5ddb6564c5b77e0006b17924 7e93b11b33ccb70422fd595fdda9731347b069b75472e6378477e7381472b7ec
5ddb65659191710006b575c3 0f5409bca4f8e722b63010064eadfdbaf2ce3cb3e8451f8d54674a13f48aebc8
5ddb65679191710006b575c5 e5c35d8296a23cbd26acc1cfeb7a3555fa25b7d09128f435bd3f7f2578f3a2dc
5ddb6568c5b77e0006b17926 7798fba8ed5897810fcdd952b4f5256bda58e2888805f87c1ae83086e7530f5a
5ddb656ac5b77e0006b17928 500df5d5226854c47fb3a5835a77e291f79ea538f7a2c66effdd3e674fa9b8e9
5ddb656b9191710006b575c7 56a3feb7e52ccb61aa3d8dbabe52f5dbde8b712e9d4b8a2793ece20aaae23ab3
5ddb656c9191710006b575c9 c1023a2d6af8ba8e818685bbffebcf82668ea2910d2ca5096b1dc945ca060fdf
5ddb656cc5b77e0006b1792a 9f8cf8946a346efe43105d2422ba9eb1a0e4db8a29ea279b1657c3885277a39d
5ddb656dc5b77e0006b1792c bbfe00b1442c8c37d9df9bccbfe0b62cb8c11e590e3e78f8aea5f0b5869b50c9
5ddb656f9191710006b575cb dd7ca9ed93eb4254e59d94a9c107b3150e83db19b06fe26d1e6b5d79a63c53e0
5ddb656fc5b77e0006b1792e 9bf06634dc7a14ecd5e81bbb55cd3cdf118b4e1e6bce1b565b32c0ecff199087
5ddb65719191710006b575cd 708e7bbc9cafb8c7f509f63001117df7327c42d0fc0b5fff27ae610b303d486d
5ddb6571c5b77e0006b17930 fd816c866f528cd8001ae3575fe71e38d9680b844f638192fa48e5408a1aad2c
5ddb6573c5b77e0006b17932 b4d73a1c855ff8f7670842b579accbade6f173d4c136d8fbcbef2edc58128505
5ddb65749191710006b575cf ddf640b4acb034ac0cafad45b00fa4d1276191ed5f65adc4ecc4f5e2f15ef29d
5ddb6574c5b77e0006b17934 54bce58dd61b42787cc66e38c2db5d49d08551bb80622899cc2bf1be13a4dc2f
5ddb65759191710006b575d1 ebe0b6757c2151be1325261b7d6b271cc511866b8afec910d24593a6dd3f0a43
5ddb6575c5b77e0006b17936 072369a5386e72ae48f26e317cb6df2a535c2f6e1fbdc8d6acfdf6f037ccc1f2
5ddb65769191710006b575d3 094bf86c575b231671fe8f2239628f78c13d3feb2a4d59458434486701ce196c
5ddb6576c5b77e0006b17938 54051b4ccdef2bcd747a2d07666c542156a9eabce8c95b74571a2bb79f9efaa8
5ddb65789191710006b575d5 36c5cb2b1683281ac8fd39fce008f94a0bc83c627cfdf57e52cfc01f511736b5
5ddb65799191710006b575d7 90a67ca177dd4f7c51ff1b9baed5029e99dacae07be4da171d54378388b60220
5ddb657ac5b77e0006b1793a 0393e51692662b6ecbd1dc12f7db8fc03cf63fad180247b84003b6104b4670df
5ddb657c9191710006b575d9 335d3e6555387cb0dc7a0a9e357753aba3d6e6154a0391e8e122e35dcba15226
5ddb657d9191710006b575db 2a65951547a770d39e5eb3925f4e7ce3b50d970dc42b2ab2f39418c5ecd6e943
5ddb6ebac5b77e0006b1793e a1ac28208658d1f0eb65451d7a2506a4b8c760b8d5c2324a3315565fc400e205
5ddb6ebfc5b77e0006b17940 b73eed27033768ccd261957ec8d938b6bea7fabb37e78b7fd17a2e1b940b8c69
5ddb6ec9c5b77e0006b17942 30306c2a1736a57cc237af6d5c0f4b3e50888121d0f29a05df9b47fe3a18aa49
5ddb6ef99191710006b575e9 caa857b58181657eb53f1334dced254ce4df25f1dadcfe1cf68246a52ff6c6a3
5ddb6efec5b77e0006b17945 54b975dc62526f93b0106ed688bf72328424182c33026eb938397c0731a0f175
5ddb6effc5b77e0006b17947 aea691201b933987b381171bf276be658d26b12575e8b98e245e78fefff0c70b
5ddb6f00c5b77e0006b17949 90237e0e5bb13f433caab86d450aaf6f9831deee759f1046708ff010ce3e92e6
5ddb6f019191710006b575eb 4c585e82346f6300f5ba4ecd93c36e201aaec31942338ba55d7188e42d3fbcbe
5ddb6f01c5b77e0006b1794b 7b1ccb1330b36e3b470cc66e86686f428ab00f5f499a7002d177151a1d05989f
5ddb6f029191710006b575ed 36e0cadf5679fafa7e9b2d336595d75cc246335f6ed24f5bc6740b146d91fcad
5ddb6f04c5b77e0006b1794d adef0930212b93e4386137b31a0fe7975bf59cfac5488da63fc6845f1ae27b9c
5ddb6f069191710006b575ef f60112a1a260f050360265fdca458464fcbe21bcdd2dfec4a1c67e6446cdd496
5ddb6f079191710006b575f1 26add9db76a69fd786e7441e08e19662c86a9e9999613844f68925e01dfdebc9
5ddb6f07c5b77e0006b1794f 544ca6d14f525a6a26cca00701c45d72ac34631a8c99d5d3b1a7a823bcc45cc0
5ddb6f08c5b77e0006b17951 2831c3ef599d16cc80bb129b49c925b87fce69d92656584127f90e6cccfaab05
5ddb6f09c5b77e0006b17953 7e9ebc0619a1d275263abf871a6fd6ae39ff8cbc07d16f2a1725413404217ab2
5ddb6f09c5b77e0006b17955 98e758e2450bb2f87ad71085d404ab74317e18d26d6c62bf2ea9ed95e64b92fb
5ddb6f0a9191710006b575f3 9ba1af4ca27002fe3c0d596fbad01657f606ac36d8eaded4c21ad2ec791b3ae3
5ddb6f0c9191710006b575f5 e5e9aa5d1f6f2c9a3bbfd921d0068cdd144d2debdb5cf41a48e4d8a46285db1f
5ddb6f0cc5b77e0006b17957 aed01949b885fb08a8f5bc82da4e16747e5bffb1525271a8af1c9e0d591bf8e4
5ddb6f0cc5b77e0006b17959 4c17332f323e06010b355e082a55f9db8842cfd9a96a278835607b9ca6ff1111
5ddb6f0d9191710006b575f7 ede2f32c211e6ee827b22d70be894704f591aab9b7e733a1afbe4b401ae3b48e
5ddb6f0d9191710006b575f9 855419dd91dc3de1661c232ffbdf80e8d5d8eb09b45e8b7de61bb0afe538acf0
5ddb6f0f9191710006b575fb 29edecdb89586a69f50ce50467f9e915363ffc929a35c09bb83420a1d60e74f4
5ddb6f0f9191710006b575fd 63a1c4cf4b2ed3b5460a6e67ed4df66a4a813ddbf94846074410fa73ab4331c5
5ddb6f129191710006b575ff a0f385608cfdebeb0f48ee611dce94c824928b52531e901c155b7d1881b9d0d0
5ddb6f12c5b77e0006b1795b f34bfa5d8f6ddd1e40101a68e11e3489e26436d763c9d1e24378e52d87b90316
5ddb6f13c5b77e0006b1795d f2df41f7dd0ff663ff4a0c60420b24c30ce4860efe5584c6724ccf7692a5ab97
5ddb6f149191710006b57601 4a9a295763b08a58b9be1a270915f2984eb76cb99e5c693c2c79472424cfb397
5ddb6f159191710006b57603 a3bb5b10e162c54052c7aae8c411630f9711603e49c999654a2c6cf97fd7bb9f
5ddb6f15c5b77e0006b1795f d471221ed3bb6400e3401d15896a8ba1e56610ada4139037b8281b2c689f0d91
5ddb6f16c5b77e0006b17961 479999ddb35848a7498c2a381115fae8d7d7738888546c8cfe9d2049a938254e
5ddb6f179191710006b57605 e5593c328172cd75a2e6e6638012a8e4c2165bd56b4ee5d2a57a600bc6c2542b
5ddb6f17c5b77e0006b17963 bef26bb64c50b3a48069828a8c19464cbcecff306e612e381115f8b11820affe
5ddb6f18c5b77e0006b17965 7663754635dd04586bd6231111bc7a0fc0ce8d32e67df9d952cec6cfa0e8d91e
5ddb6f199191710006b57607 d8416455d154bd5882065261726918df2b6ebf6cfb3007f9d738d9b69f185e36
5ddb6f199191710006b57609 b537ba295d3ca643016e0a44c31a57c149cf8ba5e07de4dbbb9b3d73128b1781
5ddb6f1a9191710006b5760b f6bb6767107d8826cef04d863b52d8bc914964bf35fed2ddf38a0ce813639566
5ddb6f1bc5b77e0006b17967 e6b287d2ebe39e353ee10cf776a9074251ff19f0c10f00baa1b40354a9f2173d
5ddba2b69191710006b576d6 c5914be73daaa03339886645288d230d30a1284da3a826fd84d7860f7d1b2dc8
5ddba2b79191710006b576d8 e36fec9a94a10144e482471765ed0d19b5086c9283ca95c130c6aa5d46b5ff18
5ddba2b99191710006b576da a4b95041cda73c1587acb7a15e1fec606370e8b01c0bb17e892b448face3b3f3
5ddba2ba9191710006b576dc 53c4817328261b4cd15b88788ff46d48d52373b0334b3f92767648051ebe0497
5ddba2bbc5b77e0006b17a10 2e1a2cd66e946844db74c72add639c0e0f5e0eacfba8679ada5284ddc51a2e81
5ddba2bc9191710006b576de aedf911b03c31dc441815077321dbf60061a6956a20f1a2eb1ddaf85ac485a6d
5ddba2bcc5b77e0006b17a12 f2b65e7f8851a97161964d5003d1265258e7da3755a6700ebdabb72ab0ebd551
5ddba3edc5b77e0006b17a1d db7af976788e90369a0b62ee6cb659111e290dc62f6858bf317a94f357de2798
5ddba3eec5b77e0006b17a1f 83e80da3ddb5d17a6bb77a2e17b9f65d9330c7f5586bbb2a439907bd1589cddd
5ddba4ea9191710006b576e2 84d876bf4a52b7d49a540bad35b902e820aa4e97df7eeda4d741e776e44caf26
5ddba4ebc5b77e0006b17a2c 91a94b78c9cb8b2f41e6af1cca4843244cef85ebb6f6c6b79b9dee155cceec6e
5ddba57cc5b77e0006b17a37 a83628daad4fe7c6c50aab871584edb4a7e99cdc0ce73eb01dbcb7b0b1310cbd
5ddba57d9191710006b576e4 bad0d37038d9c0527bf1997ff215e32b14df94f2ff9991ca235b580a355e0fd0
5de8a9c77491b00006eaafc0 866a54187e7f6b822328ba0440e97bdc1f82d428612467fe63bc91809471fbf2
5de8a9e57491b00006eaafc2 0e1efecd5830e5e7b932f76029698d7f54345d7af445bebc45ecf85e58322941
5de8a9fd7491b00006eaafc4 8f5f26e7bbb1e4793b382741b427a610ca4b810707b5cb400fdec81fe75e9966
5de8ebff1ba5a200068722a1 789832bd38ef0555bc426a08466d3d0eaa5a8a78f3040762123a97d506ab3cdc
5de8ec021ba5a200068722a3 7e19c528cf11eedc6e59b1ff8dc58d0a1909d03425ba35e3f79bdb3007b89c05
5de8ec02376b9d0006fdaa8d a067c103af585638043a9750c0bfe1d5f572c497b53b434c153372a017c66330
5de8ec0b1ba5a200068722a5 38674e05b922d7f4ec5ab30c5875efc107561f33789c9f9cd1d18570ec85f6bb
5de8efc61ba5a200068722b7 e865577e108bba4797ccb5ee566fbf308b9d9af7fb55654bb81f659ec60d48a1
5de8efca1ba5a200068722b9 673f1dc0c6d82648471e54cf182362ec020366e5a60ad25f638b33aa5b25f9a7
5de8efda552b1a0006d287b7 124880bcb425a6fa4f9e92b916ba742b6d54a935830a575e5540d7726018a171
data/site2/B1/
floor_info.json 2654e7d102e258dc91a8704b2b9fdd5ad6c7d16839971bae2f253d1c31e22d1b
geojson_map.json d7dfe22542b592c3c27fecac56244d50e00dbb38f0092940787cc3633eb5bdd3
5dd5069f50e04e0006f56287 f2b95d6fa52191e47590c101a57e7c5a74e41e6880517e4084e93f5e71c53a41
5dd5069f50e04e0006f56289 27fcb33262210c1b50831f8d69227afbb739ab1eff98d978a6603c9ec5e969e7
5dd506a6d48f840006f14810 54a116be49b34730bc7a7783654ac8184fa1938e964eff150ad5daa59b61c990
5dd506abd48f840006f14812 6d2875aaeb6d9745b58c9dae8d4e30cfff68b729f7f32ade2adfadc009e32d4a
5dd506ac50e04e0006f5628f 8d7a61838991fbf3a794b7a3d0d8385853104d4ddf2cf2da808f79eddde56c41
5dd506b050e04e0006f56291 400f68f247431dc3cc59dd50f924ef810f4aff7a9b8e227999fef77bd5ea8e14
5dd506b6d48f840006f1481a 115ac559b01d5d0d4d6bdac3fb1865a5d5519a5a6f5e4580bac2719671d6a9cb
5dd506b750e04e0006f56297 f279da1c1804fc096bd8358314ad1c1edfaa4bebbc9e89f043ae2b88c419798d
5dd506b850e04e0006f56299 d4378f0dd920689684838ada81cb2acaaaffb211b17bfe25ea96485dc061b593
5dd506b8d48f840006f1481c 52beda235696677cb5f2d74062e316062b2f45c88fc5f802bc49f374d9891901
5dd506b9d48f840006f1481e 5f2b8789ee82e66e142d2163a631e8248873bbdbdb821ae55b56fabc0cf9d92c
5dd506ba50e04e0006f5629b b435938555fc64a2c18b8ed8f3738c9a471c65cfca8a6250c88c82345bfd094c
5dd506bc50e04e0006f5629d 54b605cc5f0461d85e5315142a2d34586603271f933a8fc0f4e66d5f67881d9d
5dd506bd50e04e0006f5629f 48ddf712364e076ed4918a0bffef4022ea3ad49d728a22b1036018b88a0be41a
5dd506bed48f840006f14820 18349d6069936cb6872f6b6b2b82b0308d48c5ec11b6c890b5afd906e80ba75f
5dd506c050e04e0006f562a3 05f610ca1def02c9b7235359959b017ec1e99193e0bf97cd8de003e5cca76425
5dd506c150e04e0006f562a5 d27a70bb0ca061cf16cbf0f38bc38f1c6119000cc8b0555f37adb6e9a8ff0781
5dd506c1d48f840006f14824 cd18d55c7266189d2843e042767d8789f274b0994903c17a4c21245a3449210d
5dd506c2d48f840006f14826 1762438687496e82b82ec96683ed0d974a687975e81dba86c3d5d0f0d305bb08
5dd506c350e04e0006f562a7 74c4e91c370b70715cc40500d3ec3b44208c72896896c6b5228b173848f8d6a1
5dd5118050e04e0006f5636a 88a6758347423230f9e1916b670fb34e84c553df272006711e9bd563a0ad0d05
5dd511abd48f840006f148dc 9936adbdbba83a48d8328ac908c7d135a290199d3fb70ab2d9c996f74dfab0da
5dd511bcd48f840006f148de c1b53d472e8ee45323e4667a68b66fe6949c4828b7715d534862da9d071550d4
5dd511c850e04e0006f5636e f2ceacb565b4c6cb92f1c465d600518006445c4fb509eab2bb7613e356d74a91
5dd511ca50e04e0006f56370 483734f0c462f067c2a9cf6977a3e0ae0078c33fcbe5da1527feb8c96f568ef9
5dd511d350e04e0006f56372 4d01089a640c217a45d291f77a96b15fec7ecd7c115f29bc2b238512d6ed9e34
5dd511d550e04e0006f56374 78b620d13f805585e08cb29f22a4f495d111f0a000f11103bd51c581ca059247
5dd511d5d48f840006f148e0 2d82b14a3343e8e02001b0b820bb9a304bbef1cf98e97a63fabcf27dfcc75b22
5dd511d5d48f840006f148e2 2087d6c2cf43431dfc9671362b16d13ced6591bbc3dae1a5c2cb7ea6a49ebff0
5dd511d650e04e0006f56376 84d22d7f633716acda138c752d1c8ae647f9349c2db2cbad3a711ba9d2db3f95
5dd511d950e04e0006f5637a d2f1fc1da3f6b1646c6b702b81e250a9f556b0d7d920d157155f48cd37232b9a
5dd511dad48f840006f148e4 7757b948b8a15e3b94162ebc9564fd338ff498d01250501e06c15ce8b7d6daf6
5dd511dcd48f840006f148e6 02c737fce57d0f733d830e0722e144c6cde986c6bb12838e63f706ff127a258b
5dd511de50e04e0006f5637e 7896ea455cba3c92b11c7610afda64539ba3a2a76da9e853e9f351605488baf9
5dd511df50e04e0006f56380 b6dd71497ba27061bba278367cc492f7aa89bdc0b2ed51f8712b925fbaf9a67d
5dd511e1d48f840006f148ea 386b9c1bdb9b4a9ffe38fc04785d8ff158273735cf79f5c382fab978b5904b3f
5dd511e250e04e0006f56382 d92453700844e230abfe0b8b978453f7c0d714bafb48365ff8a78671f52c3827
5dd511e3d48f840006f148ec 164b83ebe6d1dd8631d5d6b199a0f7a552fd21188283320be0104d0badac1c81
5dd511e550e04e0006f56386 6693aae4dc4e3323a913bf373726dad7cfc195d5730fb4fe7cf09c4d9e26ca91
5dd511e5d48f840006f148ee 6f17765f43ed043dc448ccd27998ac06d1e8e91b6d7d7ace4cf4b8c14c168dbb
5dd511e7d48f840006f148f0 7bc82057f5e28c0f1a379d896b373fdd630f415465cbf7f3fde57882fd72673a
5dd511e850e04e0006f56388 73cd3292beb46854e65a6d5600c37740669ee1de98fbbafd980ed80186e40877
5dd511e9d48f840006f148f2 dfcffdab855337030cfeb99435ad412f47cf97b2542c00a1155215240c8c1e9e
5dd511ebd48f840006f148f4 1d01a7ef84f731773b31b476d1021b968179ebeede041a6161d1e1ea669eec66
5dd511ecd48f840006f148f6 c3115b003abdfea8e34c5a431eec7bf1bc318be681f2483c5fbd349b55c489b0
5dd511ed50e04e0006f5638e 7ffdee8a949a920230aaec07bbd124ec0d22b2dcd702e023449652df8e7af57e
5dd511ee50e04e0006f56390 831c4a5e7efda67a162e662bf6a9b565fd4da972ccca7a7c27543d48c558124f
5dd511eed48f840006f148f8 1c5f28f8cb6f8a30d6b266918f1a26c3eaa1c8271998c3f8262810f8f1303546
5dd511efd48f840006f148fa ba582f78f1641c0ed2be3c0cb70eb52950aa8b70cf91b2c47977c6ff498cbf4d
5dd6190fd48f840006f14d2c b510a224563a6d400f2e55294d5e426fedbfd002deee9f033f3d3901ed850c9f
5dd619177da0810006e2400e cbf046434267a9753c174925ab7de024ba19982f9f454a6ffcd3abe47661687e
5dd61a487da0810006e24019 65440965f466e0125757c1356bd2c256fbb69c83572e05bee724369504c92d59
5dd61a4ad48f840006f14d32 74e12aa119727a079234c8a2ab2c3faf08267fcf1cb925ed3b7ace4ed9aa8ba2
5dd61bdbd48f840006f14d4e 0818e6b138f553e3aa38efbd8b4bf9b55dc5c5e25500306e51acd6438cfa8027
5dd61bdc7da0810006e2402f 11091228c8583e996ab7f0f8bc85359502b2e33c69d45e87b7acea9eda2ba84d
5dd61bddd48f840006f14d50 7d52990ff2b9a72d7e7a66b68f1b27cdac152a5a57b8a5565ebc82f8890eff10
5dd61bde7da0810006e24031 6b4c41468885112535aa086e364752fea353c35496ed7284a83a5db9437daf6a
5dd61bdf7da0810006e24033 31c3f4d0282b78af109ed32a4c7cb11f6435ea3abe61958c0c44eda94b90d6b4
5dd61be7d48f840006f14d52 4b5832fd1fc4216e05d57d0e6cbaba1912ee1333d58136741b451efcb7a40c7c
5dd61d68d48f840006f14d6b dd4f33b8b12105910761e6c5cd2ca5ffca83b1609d34ecfd6b2ce29b73813c4b
5dd61d697da0810006e24043 fec64ec2da333ded9ed51dc942b4c1d6576bf443faf5277e92e5240b4aa13a0d
5dd61d6b7da0810006e24046 4771e3320b4a28bf409add70fc3047d1f3532a0dba8e706a6910fe3a66e2bfd3
5dd61d6bd48f840006f14d6d 7f45fbda3f0563f41c09a1ded7da0686a85f4c0abd8ffd6a66bc2738521f67b7
5dd61e6c7da0810006e24059 02611990de3e7c528c92a11d7a62ec7ea8edc112014bd4c63017ee3e0c087adb
5dd61e6ed48f840006f14d7a 4963124fc68815a71163b0c8ea4c5746168d3631637dd21e2a6be561ea9cf6ca
5dd61e6f7da0810006e2405b b6f772ed7f3d3299165287cb249245787961f25076ac99abdb8e6edb782abaae
5dd61f5e7da0810006e24077 66f0c55acf77eac1112999656eb6cc1b8ac3ae4603ed189d634186b93cda4c74
5dd61f5f7da0810006e24079 3c461a1fffa55ea003f5d5bfa263dfd2d9058605a28f2bec3881541e9bcd6dd8
5dd61f60d48f840006f14d8a 90acc92823ff356d1ad7481369717161c7f6fb7e48db213e28314a39f9c55cb9
5dd61f617da0810006e2407b bb5d4e1eeed92114d63280d650ad10d13d01851c46fbbe37dc95daea8b8a6d5b
data/site2/F1/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json e103a6d9cdde62134eb04f7d6f67db13cbe6c4f5a1569b3ac9e8fbe288477143
5dd35c6844333f00067aa0ba 064604c880f06371728828b838cd71c90e256f5b87ec041f1a392fe0ec270eef
5dd35c6927889b0006b76848 c09e931c248312ee2d3a8b6805f0ad3404815d27de16b08d718c6266cddfaa5b
5dd35c6944333f00067aa0bc 1ad9419669e15eb7c70356388490dce91d953785379da244f6a136f1c06be718
5dd35c6a27889b0006b7684a da51d45e0e32b692bedf1e849583d87043dff2a6357c5c8b1b750239d0e4c24d
5dd35c6b44333f00067aa0be 9342956471080c766f9165144f0ba427447839bc31f3a35ff7da2a3a7c0bdb01
5dd35c6c27889b0006b7684c d8cd28ef8f7d6ab3a9540b9e35463a1aa3c70c6e15c49f3f5e5e550ed6c6a6ff
5dd35c6e44333f00067aa0c0 58d892a0bcfc395235aa5661c849005bc246e71d856712b2dc6604c8b68e3b07
5dd35c6e44333f00067aa0c2 206b7ad71bf02ef5b934a41cbf999de9fac2fbc806f4fdde63e41017f8a7c56e
5dd35c7027889b0006b7684e d37d52c7793b1946edb669f7d1e7dae689ad6de038fe3f591a7b280b91b277c8
5dd35c7144333f00067aa0c4 72d429fcbdf8457e6065434715e87b33d8681d080973881ed88d8629e0fb8c75
5dd35c7327889b0006b76850 87215e0e55b47133ce704eb2d85a195733cb8230be2ce3a8c465dc263d7f9d2a
5dd35c7444333f00067aa0c6 87b537eec020b3001e276fc12f4ebedd33ab2fc1dbaf98a9233221abe7304a34
5dd35c7527889b0006b76852 335ecc74d57e8b0970c7aecc14f7c7898d3e2d1fe1b05d43bea0f2304391f86e
5dd35c7627889b0006b76854 126f126ed7e885e4245c2c574b5851e42d1b379336eef13ff92f38cc7439cdb8
5dd35c7644333f00067aa0c8 ff013797ee887ac1476bc30d255a5846244ce7ecb37412fac048163d150ee0f8
5dd35c7844333f00067aa0ca 0a93b635801a10c0b9a5e22876062d5070940fe4c88cc86c9164f9c039e27e70
5dd35c7844333f00067aa0cc 57290f27caa57a670c47f24cf7aa8d1fba4a564437cb2296ac8367863d14d44a
5dd35c7c44333f00067aa0ce ef7675e43cb775f883c58dff6262cedc9f893e5ecfa37d7a7b0e213c1c21db87
5dd35c7e44333f00067aa0d0 0c66c2b98b89b4990f61d9bef212a4b9052ad205c145196fcbe0230b9e81bcfa
5dd35c8127889b0006b76856 ba33f4777d906be56619507b0609e8528f0cc786ec2e7537169b60ee7036bd69
5dd35c8244333f00067aa0d2 a9708766b17a306e098159c2a43805aea08a8ff7cf7cf7f9980aa8c037d24332
5dd35c8327889b0006b76858 793cc345907d4e01b82220a40c0ba288d068b0195ea1a4ba0094332819bd9764
5dd35c8444333f00067aa0d4 65e8edff3355df1ab8ada4b3b91cad50d66b8383d1350d71b722a24bf3a2c393
5dd35c8627889b0006b7685a 6f92a79a1d5853a085be4a2707bca540fdbcb9ed9e11e4e8e39d180095a4fc69
5dd35c8744333f00067aa0d6 df7b83f6578c930ce66ba3e8688dd8fefc0ebd98581f0456fe7abcc37fcdf269
5dd35c8827889b0006b7685c eb906106a15c5115d2db7fa9d926361acfc59beae1b14c64ac89dffc0d358e0b
5dd35c8a27889b0006b7685e 700ba632b2c5e9db83814aa0be474a60e05a00a4bdba86204573a127caefce6a
5dd35c8d44333f00067aa0d8 32302a0ad6c3dc71bf46663de6494b3d946e9230a2c489ad0b4add0bc50edf61
5dd35c8e44333f00067aa0da 9bf2c24da93ffbb5211159ded798a6dfb3b860f27c55550540ea8c06d77d2308
5dd35c8f27889b0006b76860 1363fb73ae50a03c45405714e6b02ea505a7ca0911c02b914d7b6565f6011e98
5dd35c9044333f00067aa0dc 6d283afc1e5ff5880608830204a90d0cb072f41063e12bd2cb6b1f4cff1197de
5dd35c9244333f00067aa0de d48d1fc28c8c2187459883fd4590d45a7bec8fd199a33bb395f032d06eeb9faa
5dd35c9527889b0006b76862 f158428e8ed09c712995d863815452821d31a6168c41e045d85f093562f03eb2
5dd35c9844333f00067aa0e0 672b6d61faa6013c08b133230af90480bdc34d907b2ce5903ba62175357232d4
5dd35c9a27889b0006b76864 df45bab92768f268eeeff92b27cdde7d798718e95b90ccf427e2bcbd40921728
5dd35c9a27889b0006b76866 a65574b683beb2b9035f144e989bf2656b3c0efb4c97f42a9b0cd2cb0f11293a
5dd35c9b44333f00067aa0e2 d036d4424affd4dc71905f2ec329689c81fa44575ece68e153786ca045480cfd
5dd35c9c27889b0006b76868 4843b70056321bf53b02c17096d7bedf413a3f4bc74f6afbce0669e522d566f0
5dd35c9d44333f00067aa0e4 fdc8ca7db66d885a001c6cf44419a49ce945fc681e06ca1b833d0db59f11e581
5dd35c9e27889b0006b7686a d6aaa283b057ba910cd2b2316d74af701b0099d019feb8075ab02347cc9b502b
5dd35c9e44333f00067aa0e6 ed9e9384f174589b955166a0398a0ea44a2af36cb6b1186b1d28d919ec1b87d1
5dd365e244333f00067aa10e 1294ff2be5bbb5a2b8344a458603a7ba3156fe98600530aa867f0c100629bd40
5dd365e927889b0006b768a3 f2128b79d322673c6eade3e8571e85bb84a91b5fff9c6b81718c065f100872e3
5dd365ec44333f00067aa110 aa9cabd69f2bf3fb0dbf24b0c473692b56f96f8c0a9e41f15dde6aeadf907234
5dd365ee44333f00067aa112 ba79976d3786b1ad1d44eb6f54c3bdccbe832ef8336f09117b6b6bcc1055e347
5dd365f027889b0006b768a5 084fdb9bb7fadcd6fd3bf1fc8d49e2add4e52a076975d6538711c0b68c99302c
5dd365f144333f00067aa114 185bd1fda236bfb860d1fe3238ff5f847b9f2554fc7bcf77b80a63115571082a
5dd365f244333f00067aa116 49125434590d1f0a403225824565c32c15296e101ad430e02c1e120b35a3548e
5dd365f344333f00067aa118 13a1bd766a08d66ccdd9961f8ccbde46fc9c05eaa67e8789254a5a489f54df90
5dd365f527889b0006b768a7 6656b21f08bafd7bb63c399f764fd0655b7d40ea8d7ea9a9c688c2619b71bfbc
5dd365f727889b0006b768a9 0253a979eefa2f801aaf7ee3f402ca31a1201d29f2868618d35fab6e959c4888
5dd365f944333f00067aa11a a520a8dcb2b70075da86fbf6188eae95c14be8791a951ae87b99e93105ae2d4e
5dd365fa27889b0006b768ab 0eaaf40f2dacb9737f38b4bee27c8aaa07ff5a607e4429d56259b31ae1143671
5dd365fb27889b0006b768ad 54a2bc9675cd5578d80bbec0569f00b8142755b8455d6f56d60fd0bdc517856a
5dd365fc27889b0006b768af c151aee74c7f0565d0eaf791a0714bda3402e468eb93409e6fe834d2575f030e
5dd365fd27889b0006b768b1 b18804b34631116cf3cd8017ba3eebbd63e6d6428e5e3a15bac4bed7e4cc8581
5dd365fe44333f00067aa11c b7e9bb444df483dfe7add61997a1aeafa050e98e67d700f602935c858305f7d5
5dd365fe44333f00067aa11e cbb4a8b2db3f63c2adc84687a208e19c7732facb43c03decdc3a5e0fb1bef46a
5dd365ff44333f00067aa120 9dac292c167c1030ff4a6e5a4e30dc090c6c36699807a1932f5eaf1f5ad46907
5dd3660044333f00067aa122 9c21e1568c598bbb460fbb3d7927ef9a65ef58aa472cce2fabc4ada18fa674ed
5dd3660127889b0006b768b3 f8ffcdd2257aa04544e3c9944cdcf4a12a51d9ab1a81962f4bd294a7cd49b885
5dd3660144333f00067aa124 0c24952675ce6cfc3b0f0fe7d76708a08d9be672f73b9ad002b827c259044167
5dd3660244333f00067aa126 33042a194385dd8f6cca665d0a4d780583d308f865b56a33d7a4d1ce6b9a8263
5dd3660327889b0006b768b5 a771713907e79453215342ab90991df3fed88c47d90994bac8927aac8918ca44
5dd3660444333f00067aa128 bb80daa272d762f642f9c530ab05a24a9b81bc6a99fc304789027d022823adf6
5dd4a18427889b0006b7758f 3313b97aa0273e4c1d235642a05bdb2d4d4b37a09a7e96ca504b1808992d4e49
5dd4a18527889b0006b77591 8c36c5e6142a74aa4e80054a26c3d702ec3056956c636c24b2c018f294d7983a
5dd4a18927889b0006b77593 1bb8d3a0fc22c2ad61157653df6443aac53691bfd59dde7c740c4eb86fda7bdf
5dd4a18b44333f00067aaddb ba427d390b93bb4f018c71cc8828e345a265765bfc93a19ece552250fc592b41
5dd4a18d27889b0006b77595 524aad2fb63960eb4f93e7bddae9e933930858c3aa3562c4da530891a1857fd1
5dd4a18e44333f00067aaddd 4dc912c66f58f4a5f997a08a22b8e09a9bb64c5e918703e748aaa0484e5bde8e
5dd4a19144333f00067aaddf b7284bfbc1a3f33808d2300771c67328f44829dfb0e20376eb675536d4d14d3e
5dd4a19327889b0006b77597 e69dd8291986f9140367aeb4172c8af05234d822f61eb86d11bbc7eb9cab7691
5dd4a19444333f00067aade1 e74e40f7942258cd519d53cdcf8b408f0fb8e11fd231474d615596ddc2e640ef
5dd4a19644333f00067aade3 9d7e7fe9d39005a76dbbfa202329ed02e2cd482bb6f5b2e07087d1b65f29dd8a
5dd4a19944333f00067aade5 6ec1c9474acda29ccf18b211677633593a80596d98f937f20c42ddfa26cece74
5dd6139bd48f840006f14cbf 323ddff9084affd8c9c2c55fde31c1a7afce149c2e065821da64fdcc1efc6166
5dd6139cd48f840006f14cc1 88bf1ebf428480f149919dca21864b3a223e47a31724241e4e44e6270197e7eb
5dd61435d48f840006f14cd6 edb1c2e4a1b0ee089482c25b91f0f52a1aa8cac464fc4eda04e482e21a16eb21
5dd614367da0810006e23fe0 7975a26fac6baef30e2e25afd20962f3ab1abadd516985419a5a7a011dc24d42
5dd614e77da0810006e23ff0 2579ed0cd42d7b61e2382bea3a9dfc6670954a9f1c1a3cfa4888feaa3f03e9cc
5dd614e8d48f840006f14cdd 5e30c80eb50324d413403e4b6e72a5f440c272f345f7c2ece393bd087cb2ac6f
5dd61594d48f840006f14cf0 3b37e092a6d198d3655e4ec93e475c9466e49ad4ec67339ef84f06adad404649
5dd61595d48f840006f14cf2 3baedc73488754e28ca8e61717f293973e9bcbdcad11cb06654ea0e7d76cab72
5dd61660d48f840006f14cff be33d538a1b6dcc6bb5b1869534b697fc048242b78566a4a0ac31f9a20afdce1
5dd616657da0810006e23ff5 210db953ca954e5bab1088e59c1d61feeb264596020116e239d19b5953b9b0d3
5dd61666d48f840006f14d01 d79dd0e38e20cb8cc6416c32637527e1ec121b692b19f1e74b4fcd674329e249
5dd6177ed48f840006f14d11 45d39eca1c8157aede80e283a7cddea2477a4f3bdbc8602078b1ab8640d2b79d
5dd6177fd48f840006f14d13 6c81eb21603132cd3a1df6137875784b65e4294e5750e3264f2a62fdfb3adf3b
5dd617817da0810006e23ffc 1412e4947dd96deb5c8aa9eb06c9e9d167aa8b5eec24e4bc12104ea74d16a575
5dd617877da0810006e23ffe ae88c6eb725f709e6f2f242c4e80071b2f971c4ce27294ae06be1b50dcbd47c3
5dd6178ed48f840006f14d15 5032382fce76291be0f3471cd7f3e7f0d2b57260ffbc62d095ad4618403f77d1
5dd62761d48f840006f14dda 3712b1dd678f946fb6c5e0aa641990892aaace2879003727dff4bcdce8befd25
5dd62763d48f840006f14ddc 8335da7170351a101595bc3ec9450974abeb2814facaf92d9e9988864af1d2f4
5dd6276ed48f840006f14dde 84a9d8c7ec554eef787417b375be5ae7edf63d6b8ea148c0705d2de74aa4cb50
5dd62779d48f840006f14de0 ec415ce412f9769b09ab2e3b4ddf6b0ac2e1c4fc93bf13ee45c80c0c7c1cce1a
5dd6277c7da0810006e240c6 32c65415874644ee3cfe078b62e93674922549fd780529b1818074cb88e49a28
5dd62780d48f840006f14de2 da5d60918d0a578c6a49d91af0aa0d975214156bf28b6e8f773d869412a70cf5
5dd627827da0810006e240c8 e75a287b6218aeb90045a4d6e75060ffef0daa0694551aada98dc35d23244a76
data/site2/F2/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json dffcc8b90ce219d59579c6866c8d743c1708564b5f5cf92e242a166412cfc2d8
5dd36ca744333f00067aa167 c1eef541523551cdb68a98cd0cf8730f5b221a5d2ed7d569bab3a13512fd28ec
5dd36cb827889b0006b768d8 75e9957790b2b77a3b724ddab0e063452e6fe9ec94e279f64c5f7e84408db5f9
5dd36cc627889b0006b768de f826f3b1827634b43f281c779a0f1e52f4d2ff5804aa56dba21127774c2aa736
5dd378b527889b0006b76900 904ab48c0ced41ff89013604825d52f8aca8ebf2e8be747a9e5f7a37051b9476
5dd3791327889b0006b76907 40e44460a1af2bb0ca040b1a2a7f7571927beb3924b263504425fa81d4bd8735
5dd3791a27889b0006b7690b 3de2e65381bfbb20dc1bd7bf9ca9d565f039ab1c31a51e7f1e2e0f76de9428df
5dd3792527889b0006b7690f fe3f7a37287a82a84300bad474a7671a9c190a85278b565048922c14a972c09b
5dd3792c44333f00067aa1c3 d9df0f198b89cf4641dd3f64992d891b7b763807cb2ca9e138087cdefeec5a9c
5dd3792c44333f00067aa1c5 bb77dc30a9ae3f2dbf46932cbf3b21855b01ebeb2a5a9567b986a3e3369bc60c
5dd3792e27889b0006b76915 f56a43cf254b27f658107e58f915d2bda1af886c3d8c8cc5fcfd4310303af1b6
5dd3792f27889b0006b76917 798154a8aff319feb8e424c943823c52248b7234448d4045bd35ce51ca72d5e9
5dd3793144333f00067aa1c7 e009c19e0e387443aa9cd8178009a026838c8871ed35535d01e742500f908f71
5dd3793644333f00067aa1cc c8d21daa20da3f828d2345f538b775000b2fac4fdd985fa42a12271a52568976
5dd3794027889b0006b7691f cfba8083d5f2cc431eda285a03738a979f42464a50062ed89ea0b04459fd6e95
5dd37eec27889b0006b7698c 30eba1c0c18b9aa3faaf0211363ab2731df87ba9a452a58db5a40f7e07a2cbd8
5dd37ef227889b0006b7698e 5b0c898075a5fb4584e60c3f8a708be5abfc10be35ea274cb3131bcee41cc3dd
5dd37efb27889b0006b76994 b693a64b7b2e913dc8e54c4328e7e9e1c98c4e0866ba6c4ef243b80fe7690bad
5dd37efc27889b0006b76996 12d2fce23bafc8659fbe7d1ae6ddc2e7f9273d3d4538fe4505443663da192f17
5dd37efd44333f00067aa243 3cfc602905e8ca16e697bcf29c13b842333ddb30a7d862d2df249b1194680057
5dd37efe27889b0006b76998 4c3e1699a20ac18a9599ca6789dfc42570c4b51905374e468133e97bd2e5eeb8
5dd37efe44333f00067aa245 ded847627692f665efad858142c6ae1be79dd9d51cae1ca13b65fd2663d677a5
5dd37eff27889b0006b7699a 2a6edb1dbe1317b3bf6ecd4cab021e32cb5af1aac7dd54b95c13e998abc76628
5dd37f0044333f00067aa247 b1139d9aa7183f8f4a80fe9ec9821c12daaa879c72a21fbb2a0bf3e90b6b89cf
5dd37f0127889b0006b7699c 36e65f5bec7b4fd4b92dfc4c114527342397ebc69b55470f24f2ae2daa5c02f3
5dd37f0244333f00067aa249 cf7d2cf3cd9319c271ff5285c1b825113a0f19b4143861d209a5f53013529c60
5dd37f0327889b0006b7699e d97857e80021dd2415425c3a38c4dac44272c3d4066c51b3f48f5c88e9a3afe5
5dd37f0344333f00067aa24b 7f9d5aa057fd75bdb1f447d6571cc19639628d6bfc0fcb9d3a0b4f64eb403aac
5dd60b8750e04e0006f56699 01b6c74c9e416e503b5bc64ee739d75a12353781ac04d5aa673d0a9c1bdecd6a
5dd60b8750e04e0006f5669b 6cc5eff112c29efa0de0df4b5d0f13887bdf09890a272e828b0ca99c736f7dd1
5dd60b88d48f840006f14c44 a814f4efbbfc325f1d1876d57ea26d486356537f4c6772c32224f2f5ea6c1b11
5dd60b8950e04e0006f5669d e03be4bcb96ca6a821219e318dd77770648b4660d40a4eaa8e1a0b61c52c61a8
5dd60c7cd48f840006f14c49 586f8cac2d17950a276a7295ef4db0e8a5eb080715e29d902fd2893c34428c8b
5dd60ec17da0810006e23f99 47d202f941d554b5675eaaf65c6d1bc7f83953452255c808e2fcb9452b8377d7
5dd60ec67da0810006e23f9b 69b27a745af11a5ca34ae5a46e456fc12dd583f6f78cdc5c66740c36d3fb0d95
5dd60eced48f840006f14c55 9a5cfbaa5b7880421f5dac8e92471f1422702f5c693805ce24349ee73b25e1e8
5dd60ed1d48f840006f14c57 fdc6784c7a0c54dde659bfd43d91ecfa0913631ed44f639ca7a604a0e42491c3
5dd60ed37da0810006e23f9d 499bdab1dfc59b36dfe1febff87065c7123935a5063e9aa479ad357939c205bf
5dd60ed97da0810006e23f9f f481fdd67c794847435442e227e8d03574ba581a0c3c1957441db0d4980a318b
5dd610afd48f840006f14c78 4bde66cd83f06f25eb1f4ec7c1ce1d66c9351a5b9b08cd61549e84899aa6f685
5dd610b0d48f840006f14c7c 0fe6c7ee3b774455e7f8dbf239e1b80a196db830a227593a07a16fb74190d610
5dd610b47da0810006e23fa9 cefc7f64e1246a10edf90081d8404620649ad81b9f193313450b193feb828792
5dd611a27da0810006e23fb4 5adb383fa5ca72d0651a81a50df6d37abe23a91cd93702f9b23a09c4aa07f5e0
5dd611a37da0810006e23fb8 78bb9aa8f52a7151fa3ed0b7ce490ccf424f20a8d51d1776f1fc4d92049e2dd2
5dd611a5d48f840006f14c7f 6566ea4221bc638fbeef738b3eadc3bb3205164012365bf768a63954a7246a52
5dd61290d48f840006f14c98 81ba776eaf9c1e41349d477c5525c95808bf1c3362b9614617eaf07257d4cbfe
data/site2/F3/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json ae5015aed4895ec135e01d21fa15537fbafc5fcdd00aae9c906307be285f00c6
5dd38ffd27889b0006b76aca 13dca3b13c7a66a22d6aa38897e4d5b1e822d3e2426592a424a2b42d8f272eff
5dd38fff44333f00067aa387 850b5f63748da578338137f67ed593c9df0b079c4fff7ebcd0969691f8913abe
5dd3901227889b0006b76ad4 69093d1285f7e45a69f0907557c8f30e2ec2e9e3001a7544b0fbc201d85da7c6
5dd3901744333f00067aa391 badd4726497d28c14e677cb048319b8cfc40bed08fbdd21e9a0d24e17235a2b2
5dd3901a44333f00067aa393 e1027812d9a0cdd389f90dd391829412c84c2c787e670e021339cfcfb1104384
5dd3901e27889b0006b76adc 3a7516ba1c764378dc893849c4594efac9d426b88563b4a0a93b4e21818e2784
5dd3902627889b0006b76ae0 84499300a453aee52c14b67f95464a0a9ca331f0b7e7e80af126bd956ef37f29
5dd3902e27889b0006b76ae6 7dd70c98b07f772bad50c746aaebc03b43489044e9174ca2ce5e9aed787915e4
5dd3903644333f00067aa3a9 199fc5fabc0d2bcc0049a7b6455969d333cbbd852254bebb4b9a8af108f57b55
5dd3903927889b0006b76af4 66117ec7afa9a22d146e5e8d7a2faa03a89bfd7586f5a2f6ede708a5f304022b
5dd3904027889b0006b76afc 3a8a4dd160c6772c16037f3c4816eeb5876cec24e550c26a8db393507bdcbacf
5dd3904327889b0006b76b00 acfacd7f6b13d9484b07777565cc3f660a31299ea7b5fde2315ad2b394e45fc1
5dd3904544333f00067aa3bb cfedd8793e05ba8d8518e9683c06c51839c6c3bb0bb26a6d9240e0abcd123a12
5dd3904744333f00067aa3bd 161c18938dad6418fcfc6946c41f50113a1b86ff633b92367b82eec8f8b1bbf3
5dd398c544333f00067aa431 bda91ee77c65bf78a2ac9abb3941262859be29893e989d34be3a8ce93ef8a5d3
5dd398d127889b0006b76b83 0c93316a7a3b34436257cbe50315e8a191cdbf91f78d3fe1cd22d8a3039217d0
5dd398d327889b0006b76b87 d327f0cf5d9d2caf03b2b10fe7b64312624698cd8950ed566e0265caede20704
5dd398d527889b0006b76b8b e67a8f982640c70301fc1d962031c3c2ac18068dd788317b94e5746b077be0f5
5dd398d727889b0006b76b8d e7b6edf5a3fb68354c6d46e8af67e473aab7b295549ead46f73629605ea6f0b8
5dd398d744333f00067aa43c e0b9f6a2df31c288c6788d6df41a5048c90795e82810bf10ee4ca8159c298b1e
5dd398d844333f00067aa43e f9d55829099cb15fa37ec9847992edda9f6d58b1e5d99e7dc2e098e9ff89f549
5dd398d927889b0006b76b91 4417036c699716a13f96daac8aa9791f739fc43c6fdcd1be2e1496ac61056af4
5dd398d944333f00067aa442 44ab29ff58150dfc00e216b667ca2e8668a04fa047aa12108a5774f4f118a9dc
5dd398db44333f00067aa445 875e0269d17aaeb4610619a57e8a0b1248829800e539246c51659168f7ae30c2
5dd398dc44333f00067aa447 746f475936eea08574a8d7b6c1ea3ccf6012ac61eec8128944618d08dfe5808d
5dd398de44333f00067aa449 345ed600936fa84700eb721ec074a578da31a3d613733acba45ff4e0f2476623
5dd398e127889b0006b76b9c 7bfb5d1e14175612b6451512a6f2dfbbbd89e36de8635933a9833f3e1fceb092
5dd51864d48f840006f14961 83444443c5ad36a9a3c6adaba1a011ccbe694532cadee4a070ccaf6420935ad6
5dd51866d48f840006f14965 77a995ac96e456524e88a369a171e4c0d4ca166cbe5345906967419b18cfd3bd
5dd51875d48f840006f1496b 2c895303845ee3ddc8bff5270393029fa98ed1a9d93b2c92a29b8aa9876c15a9
5dd51a70d48f840006f149bd 1d847ffa1df1f54aeab1eeb1a6b32b1582000f840aafe444dbe2747789eb0f1d
5dd51a7650e04e0006f5642c d362c110d8ff9d23225562ff1151e2d8b96fe8d47486dc0c955b4b6d006fc31b
5dd51a7850e04e0006f5642e 766b1444269d891a145c8a2326adb034df133c357f97167131377117f8fb9c80
5dd51bf850e04e0006f5643c 9f21b859b80118d5a81a00df2b6a88ac58d200f5b71656704b21a38d5e1e9923
5dd51bfd50e04e0006f5643e 2a00bbc42787e2406bfab2b120c2750746510c01687afd4af7393a5b2e11f028
5dd51c0350e04e0006f56442 6f72326527f474242542cd85c92d97bd1266051a5ac670042dbdb926e5569307
5dd51c0550e04e0006f56444 419e5117d2301c7d44a9cc46c7a98f8b1458301b9c6c28bcb29f98c15184ceea
5dd51c0650e04e0006f56446 00cd7f3ea6a7e18d6dc7f0422db8ceef165a966886a6c4d964ea1435b972d46f
5dd51c07d48f840006f149c1 b769f98a013002fb89650617619f530d5a9397dada24163953df5963f2db0ef3
5dd51cd1d48f840006f149cd 7f3c41bae0b1fac45261e685e8aec8eb7415317301c097110318197ba7d388b6
data/site2/F4/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json d78d4715194f9318f6639cd1febee7ea4c37a04ebb5e7a940912b3436ddc977b
5dd3a32544333f00067aa5ab b4fb559840d8329a3a5d76d13b9eceac0885eac5cd45d0c804da02929cf42d89
5dd3a32a44333f00067aa5ad 905621450ae6cd795cdac7d7feda1eeedbe331b55a143488798432bf4a1fa52f
5dd3a33227889b0006b76d2d 0c1473f7ecd2e07087cd02f1f9c2589eb6ad4e479b69313cd12747b2b272e99a
5dd3a34c44333f00067aa5c0 87f00c7331e245da22274000dba83031266b6725b39924c53c5b51508ca386a2
5dd3abef27889b0006b76daf a1343591855bdaa772ae2f83fe792007ee7e09a1584dc2864c368ded18a498b5
5dd3ac0427889b0006b76db6 e67af34ff9024ab82704c153c19e72691ece20dc6b9a435d84158e58a0e1662e
5dd3ac5444333f00067aa662 e79e3c7ffcf67cbecb5213ee27fb8a999a548b4eca4f9c9781d5578cf62541db
5dd3ac5f44333f00067aa66a 1472a5374edffeab0b0b6c0ca9c6657b673a6dd1f4922914245b76a9ceda3f44
5dd3ac7144333f00067aa670 45de1c45358ab3b3e847093988fc7dcb651074c8974e01a85661edf0eaeec4c9
5dd3b10f44333f00067aa751 9e3ecd1c50921554958efb4b6afd5bf6a8bf4eb1b757a2b9e978930573680ecd
5dd3bb5727889b0006b76f98 34945a943e29be1359100a581a09d4ed3407a28ceb4a9a5042445fc820165efa
5dd3bbc027889b0006b76fb5 6eae08f6da37d09bbd0b9974deee7dfb2de8565ceb674a0b4be93ba9c02c6558
5dd3bbcb44333f00067aa7fa 71d4ce626a92bb546668c100ffd686d43cefe143f7debbcbe93ddc461d19c988
5dd3bbcc44333f00067aa7fc 6775c66783ad5108e4a1ed1a6f8bc9255501bfab0c69e4b39028e1aabc1aa069
5dd3bbd144333f00067aa802 f5215708be37cf8471d84db7fc9e98e42c46d00636a1873abfbdc7a827cab69a
5dd3bbd344333f00067aa804 18d8ff5e05cb69abc96cb456815ec258d9142afc168d656fac61509305070505
5dd3bbd627889b0006b76fcb f01e620631b331e2361818136163f42e023070aa3b9e1ab13598def1caa6ee7e
5dd5205350e04e0006f5645c 1ea820fea098f695d750dbd38790b93203e9b485882d2fc3b104151476d80640
5dd52058d48f840006f149d1 f17e07a1deeca0ef8bc4911b6ede4ba331917e5e724da770366fa68999d870fe
5dd5205dd48f840006f149d3 dabe334a6e471eae9984483e04ea5fdf64e4b912bb580047eaa85a46ae70d568
5dd5206850e04e0006f56460 2b1247ee3156759ae0ac48c4f7768cfc552c370904848fd113dc2aa5abae5e45
5dd5206950e04e0006f56464 d95e573cd9a22aeafae7530214c51099acdc1dce0c16ce8247ecf68bb604620e
5dd522ffd48f840006f149fb 2412681c72fecd7afdb50efdc2810d25652acb75fa6d12c434c74dd820638b12
5dd525f4d48f840006f14a47 e98c11b9d0ee710e762fee647828abf901170fcc6c4c960266848c33339c7739
5dd52602d48f840006f14a4f 8538683c8e4df59d5f01519d91130ac00e00fbba6b5f8f5589422044411f7d6c
5dd5260750e04e0006f564f1 58665e32634879df7337022374ade1bfa2b1b78a6a9d9517917f947b272ed55d
5dd5262250e04e0006f564f5 00707f00df5c43a81d7dbfca520a4361c2bc83fe97dabeb5a35ba7b476e4d96c
data/site2/F5/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json d6e0955996434a8bef6f61adfacb09bc94fb991029efdf7953ee1967f234cec8
5dd3c97844333f00067aa90f e5107a2b69754658b09acb5fc8133e13cd6bb1476e08c307dfc2454d84706b61
5dd3c98744333f00067aa915 4f8b587ca6b7c86d433b2996b1e1565df3cfd5a19f18930cad693f3ada2eb7b9
5dd3c99344333f00067aa919 7b8271033f7ef86a32e4aba20b8508d444e0f9eca134c50d1d350e5822bdfc65
5dd3c99427889b0006b770d7 0189d0bc9c328f660ccd1b8dfe2a3b2d7f84618ecad37218597dc2e1ef2de96d
5dd3c99844333f00067aa91b 94300b3cc65000d2ef0fe007d3d5e8c361f77c002316dc29b45d4f44dca19ccf
5dd3c99f27889b0006b770df 7b7e576ef331e9f1572dedffd93c0319baa489b532f315b7a839dd2488a49948
5dd3c9a027889b0006b770e1 48a5f2db57142ac3ef46630254760200fe2922ef77a04668f2af490f4318ff35
5dd3c9a044333f00067aa923 ddf67f6bfdd4d073aa1eec78198da99db37187a4bb07a6a41744ba0867dfb6c9
5dd3c9a444333f00067aa927 ddaf7db9e59fff848bb708f92247f521b32a80163bc6da1d9154f642740a9b5e
5dd3c9a627889b0006b770e9 3a9125a25ad63a55717891835991f3908be04879cd96df8483d1f95d123d8d62
5dd3ce5c27889b0006b7711b b830ac763fd63a510c96e225a54821d8a70ff09b3b1d1345e40782e436e48e94
5dd3ce6444333f00067aa975 8b6a82a318b5af26004e67f12df3314dc0874920818a3e6ae57795e9c91f574c
5dd3ce6827889b0006b7711d 45bdcd3a7caf1d716d515805f7588d9297ca4afd0dc833656f19cd71b7943ddc
5dd3ce6927889b0006b7711f 751a7e4ec79e8a90454c17737c11c63205ba85bbf5aefca9a38c1dc9dd70653f
5dd3ce9f27889b0006b77127 f0e3effbeeae89ddf515ec69b66d6fbc26405f9fb3ab1aeccaad5ea0968ee32a
5dd3cea244333f00067aa97e 3e8166a355415e2c5719c5cf3e4055135e05a8c0c5d72d13d0874e79160506af
5dd3cea344333f00067aa980 7f3e80598ab05c107654dd31b8925a359b0d6814d903e5992aee69576ff2951f
5dd3d84227889b0006b77211 cc9394397c639743095b9d3be8214cd0cddc09e0775592524576fc41caf946ac
5dd3d84f27889b0006b77213 401a27ebb707d179a9b30c7234528699b0961f57dfeb0e4c16f6793a436d29ff
5dd3d85744333f00067aaa74 59237871b44999bdf74362287bc8b864eb3e1a2afe848b731a84a2d0b32e8df5
5dd3d85a44333f00067aaa76 c497465bf469c3d3cd8e4fb89c2a0241e3266cae9225816cfb0cb9b11cb01383
5dd3d85b44333f00067aaa78 d534ca8ade5ea4599e97c97342fda85baff64926eee964498edd7be220c297d2
5dd3d86544333f00067aaa80 7a19c84779c853c263ba38e96668c1b80753f4add336703e9019b6f103273d96
5dd3d86744333f00067aaa82 5ceb61a8d0ed28a282e06809b45476e5112c35d2e84ffd9937e95a15c7a06818
5dd3d86744333f00067aaa84 3a9c9e17f946937b638e25197dd86c86695811f9ccad8720174b43de065626ed
5dd3d86a44333f00067aaa88 24b3330e3031aec0d47aa12513bfd4540cc819bcea2b2ed6f93aa5b2565d4910
5dd3d86e27889b0006b77227 1011b17e544684a4622cb80ecc138db8587e84407448eb64b8824e50c59bc276
5dd3d86e44333f00067aaa8c 6c220a18ddb37f46b59dadda34b3b92b7b95dbddcc36ac19656c9a0ebc546816
5dd529ea50e04e0006f56529 eb7d1d8547091e9223b85f43e00bb982c7ca3b5f05eee41b945b34e93e572044
5dd529efd48f840006f14a85 bca26070d4e6da994ecf742490720883211d5bc1171d09adb918fbd9c0bbb1cd
5dd529f0d48f840006f14a87 45769685f9de1b50d58eff921bde7dab4bf52ef030fd4a456cd243af94a4b2a3
5dd529f550e04e0006f5652b 9420dd72399944de6bc025c4dcef8781d4a7d0a9144fcc6645f0d80b075d40d6
5dd529f550e04e0006f5652d 6954fbc391de30f079041f359c6baf60e273004a0d126cce2f9b3754c7b636ae
5dd529f650e04e0006f5652f 6f303cce92b5813e06a41b896d9b2007d93e872d4ec24b3705643fd285ccbb4a
5dd529f7d48f840006f14a89 a4ef6f3f95d8c692ea940296d2a74f1b59ce69d7942dca2e561db02323b54ebd
5dd529f850e04e0006f56531 b3fa91b3503b07ae503f8b6db1d6e5bd3bc257c23240b41049e28d577982de36
5dd52a03d48f840006f14a8d 19f8b6389d254c50afb08edc7805c55d837e3fdcd93afb79ce5b77a8e1ca86b6
5dd52a0450e04e0006f56533 b78f02a5471769c488465129ea5b92f8e917b41d46da42f20a3997f03fc194fe
5dd52a04d48f840006f14a8f b780b5f1faae152cfa6c805190217308e6dc7d5b45b5fb58a5ccd98ce922cc46
5dd52a08d48f840006f14a91 2a355dbdf089eb7ae3bad76bcdd7cace1bc064786b4346b87057178dbdb2a256
5dd52a0ad48f840006f14a93 6d19b443107ece319269361f5c7ba6de9f44c055291dcf559131cd365d4c7c7b
5dd52b7fd48f840006f14aa1 5b731fc388550e7bbd746a24a2a55cc7922aeea7b7c9d452a3eb01efb82d4412
5dd52c46d48f840006f14ab2 6e8f37cc122569903e53ad69237829bd6eadec328a142d22459fce610b6647ce
5dd52e09d48f840006f14ac8 e7593a2fa8452040e6293aa501893e2703528914d28842d38deee49a2178c593
data/site2/F6/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json 5b848a9c4a8dc5316d6fa48277483a45bfdc2669e4e2ee7dd4580b0f79903ce9
5dd4ad6a44333f00067aaed4 0eb86669539fb940cf6d4fec2101e4e89002ee74044f2580b1145aca3a4c11bc
5dd4ad6e27889b0006b7768e 61ab54d3d3bd146cbbf6dc2b20192142be6a7b95daf4be1f273e9cb0e838e485
5dd4ad7744333f00067aaed8 43fb2c55241849ab021d26df06c8b9d6a8630ea974042f0eff937a684fddade4
5dd4ad7a44333f00067aaeda 3dd46f49da240bf86043887e9b4eaec7b11fe111538abd3ae4c87a4c30ecfcf7
5dd4ad7e44333f00067aaedc d7dbe8c9d16b19c54d1e62b213eccf7e55e34db12812fbfbd2e4f5cf322dedfe
5dd4ad8144333f00067aaede 0596748d49f202058b4c52316ff74a318df7aef27a7e1f814d89ee75280a50e4
5dd4adc044333f00067aaee1 cd11f06427b25df4ce37df914eecc950c240652f5f7ee830b00f802e9e64353f
5dd4adde27889b0006b77695 ecf5dd6f7b46af1d7bcc85630b35bb409892f1cda639a1df2bc01c7b11ba7e9a
5dd4ade744333f00067aaee7 0ed5ab3dec92bc2df435a759aebad75377423a34c271aaa736b26bb5dd8ecf35
5dd4ae3327889b0006b77697 5ac65bbfaa2160b3f9cbdf3acc551f5c7d20a7d943a8377e6ccf2f0d3fbdbbe2
5dd4ae3444333f00067aaeea 1ba855e8749d8bd233db673b5bda7f2fae7da24ab642dccb3cbd920a12cb901a
5dd4ae3c44333f00067aaeec 3d896ab81224dbc8052507a1ded85310c90cea71579dc12c4322e2fe55169698
5dd4ae3e27889b0006b77699 79518bb0cf0d268cfd8f0267146c3a7ba6de34d5015e54b2fa34958fc601ff03
5dd4ae3f27889b0006b7769b 24f7b6145e134e28eb7006ac604f83d74dedb104db5ae42afe6bf4f009d94986
5dd4ae4227889b0006b7769d a2c28df3914dc32b1bea3418768489429183c30114790139afd9e3bba7e1bc9f
5dd4ae4c27889b0006b776a3 f796ff22b6ebf1864e96e647eea2c1197f67fb1822fff61c2965e2e6a62ac564
5dd4ae5d27889b0006b776ab b57a1c9eef92d46703669236555c67ede1fcf004d8c7916a9c815495da9c74f3
5dd4ae5f27889b0006b776ad 2e65f95ff2b201fb6a3476001b378187c4b043ca3cac2fbb15bc56f2bbb801d0
5dd4ae6027889b0006b776af 974b4972b123a1917ba086899456d9b5aa47fd8b2c381a8f03b66527738fc27b
5dd4ae6044333f00067aaef8 d8e37432af78c14312eea88a178133c6da6e8f04726a3e3a4f7cb332ca135151
5dd4b78344333f00067aaf48 bf2cefb536a52793de0753beeb48656b9705abe3630060689bef1264beaa480d
5dd4b78927889b0006b77716 59a7b195681776a1928f2b4d78d561680f664b776d6d15665ab000b185d58f9f
5dd4b78b44333f00067aaf4e 5eedcd04df752177bb8477880b35d24ace7ca22b7fd68477d84bc2d80f900897
5dd4b78d44333f00067aaf50 5c95c2cd41a6f2d6c2b4a346382d7b18f81a6dbc0bc807c521c71cef02ae041b
5dd4b79b44333f00067aaf54 3ebf2c8c09d0dece0d39ad06d2a047ac04bd560ec03d2bd2bdc048ab8623509f
5dd4b7a144333f00067aaf57 4e15412885496fe031eb89c59990e1b23f9606cc8fc075cdc249957157900330
5dd4b7d227889b0006b7771a 4b0af12a0451abce9c6174e8b0ebfd50c05e94a1112bc6d136f7ead1e0415fb9
5dd4b7d644333f00067aaf5f 7960fadf9531b926e9c0bf80cfab7ac6f1caa7c06c34e88c42ef8025026952ff
5dd4b7e444333f00067aaf67 3acef478fbc1d175dad94eb097aa1b34ab0a3f0da5c80cae604afdc4c062ef07
5dd4b7e627889b0006b7771f 3f2e61b1d0f2d7645ee71146e9c5f5d82f0b570df6c3c48cda8b392b117e1394
5dd4b7e944333f00067aaf69 8d83d4fa68423807fcbdc44eaeacea338d7b99b0c6d628bcf0d8cef41dedfd40
5dd4b7eb27889b0006b77721 4655d3ff75436bb16044288bef5f3e1cc1d3d79333829705d6dd47a0463c396d
5dd4b7fc44333f00067aaf6b adc1c76de4be9ce0fd6c57288650749ac0722d2b98fb0a3ad34b87e028a5acc2
5dd4b7fd44333f00067aaf6d 3dbbf38d13d118ff664b91f83242488ee6c273a3859fe3c2a4eafdb9faaf3c02
5dd4b7fe27889b0006b77729 2488b75ef3ad67d2cbbbdb19e22f4c847ff4f9e3bad9bfa22dad7394a2345d0c
5dd4b80627889b0006b7772d d5220e8d59c7fa0f8a4c9841d182f093ee6e9c7fc63a4ec0461263ba5df14071
5dd4b80744333f00067aaf76 1371721f9bdd00c865492a47ddbafd1f62cbcef2057840ed4cd028dec374c16a
5dd4b80827889b0006b7772f 823f2f725504800ab4dcbc238f09e9aaccfccf6f564cbd332fd590d95006072e
5dd4b80b27889b0006b77733 1ee7799c59d2e38ea1696521eb76f7c658c18d5aa4538909b8eaba4a8a192b0e
5dd4b80e44333f00067aaf7c 07395da8bcd567c5582542caabb6f16a5af3f1b031f7981303923c2b95a8cd20
5dd4b80f27889b0006b77737 87c25de1d3b51e7877515aba3cd1c2d8b1d46f5a4d8932dcf4591e70074ccd2d
5dd4be3744333f00067ab08e 254cdb730b05c0ac6067f72c507ef5dc68ec144fc135ea53a145ba8d3df27f53
5dd4be3844333f00067ab090 ddff16e87798f4145baabb623e911db77d94c28fa9a6ab47149a6f16a0091c92
5dd4be4644333f00067ab094 03f4e514e24b8f27f913d978c84b0b4b31e1e019de49f3e6f3abdb01ae31af89
5dd4bea327889b0006b7787c f5b1ba977c60756acad2bffcd90e8f42b54cbb587a700a6c5570c3ef1d7dce10
5dd4beb144333f00067ab09e 47d6cca43d5e37138e9eff123cf5fdffe61699dd867e389bc6e89d7596d9cd3a
5dd4bec427889b0006b77886 1e1a9977cdfb49d8b4db9558eb36b156c031b9329bde66f46bfae1b94a559ef8
5dd4bf0127889b0006b7788c d29a57efad7b64c0501dc60107325e1d4cd910df75e71c971bd547acf30ad020
5dd4bf0d44333f00067ab0a3 ea693784cfcbe49a654998ac1d4413cd43fee4e88db278f04f709345087b4cda
5dd4bf1027889b0006b7788e 77bb34518e3772d0b61f4d1f627424af15e75f64b2e0d90a6fd958da45eea03b
5dd4bf1544333f00067ab0a7 49fd16dc0ea547f9360c0f39643aa00ad06b9f63c3896b396bc02c770b5cdd44
5dd4bf1b44333f00067ab0a9 96d8b0e1c38a1ce380b82b66021c041d55b2db135e60e02e80f7a14d24a432aa
5dd4bf2144333f00067ab0af 717e89e29f90c53687b53b5421f96b78cd54598f4b29cb169068fe3a0316af50
5dd4bf2227889b0006b77895 3b18d2e6948baf09780546f652f1c46f91db6909f5d9beec291279ad5f9a462b
5dd530db50e04e0006f5656b daacd33ac6afebf5405b46661f73b0281ac3157ef4c5133a00e4bae6296af926
5dd5319f50e04e0006f56584 9a9cfff4dd25636c7e7b6b8bb99059d1ae7e885027bdf3da1ac1443e9cd76e6b
5dd531a0d48f840006f14b24 246b70eb9aeb7475c39ec9d882e8ba2bc2d10d329f39394938e5d5773421ee80
5dd5336b50e04e0006f5658e e4f638480b64d58e6a720c5f39bb0534cf962dd149d8db6cf791b3b28109db06
5dd53377d48f840006f14b33 07e716c0bb485953630eb5730db4c17c28f7a95bf1486dfaa6e4bdcdff12ab39
5dd5337ad48f840006f14b35 8a4676fc2ac0d8190cf367dc80c45dba55e33534105b654047448c2be547a12c
5dd5337e50e04e0006f56592 7a3dd293018675b746a5e53816305af083681e9022fd1bbbbfedff0ea8ce339e
5dd534c5d48f840006f14b38 ed868d60c2ab09a166defacee59de6fc4bb43cd9ec80c6f65e680bbdcb1d67d0
5dd536aed48f840006f14b46 72a60939e67405a17cec365ded8a2562d2f04d66b4cd6729307d81015476dc19
5dd536afd48f840006f14b48 8ac29f7d72a943f8682415db90e97c06d49149daf44622d9364c00af8fb36938
5dd536bad48f840006f14b4a 49e2aef078b745ea66469573e12fe67f82388ab02807757332270046865a978c
5dd536bc50e04e0006f565a1 98e4a0e71981a3492a94c87caaf4d84ae4074fa34622eedab6c8188b579addab
5dd5380bd48f840006f14b5a 96e44b3710a99029cc08558733942839d14dc094f9476625f636799106cce265
data/site2/F7/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json 82ec3d36b188ae9cfe7751a33ca3672e8342b8f5e8ce80bd9a58d50113b8dae5
5dd4c93744333f00067ab1ac 3a6f5f4dcd97df8e3d5fc7677614d6675a40797fcb73e5563840e899389d60c9
5dd4c93944333f00067ab1ae 468b91d2968d80bcb0ca76a393ba4ec142ed561eb3f73232e637d037cae1d8a5
5dd4c95e27889b0006b7799d 9e62fb72da3582a6177ff4afacc7e3f76e7b3f1ff3bd5a0abff5a2bd67a3c037
5dd4c96927889b0006b7799f fd16c2177b2c472325c1661394ffcb7e844d9d5170f0a9105060af3be1caca94
5dd4c96f44333f00067ab1b8 c95b143a52fd74286e2f487c7484f052fb2e2f2bf6134f341dc4c25064d5d2eb
5dd4c97127889b0006b779a7 c2afdf658686d6c3fbf4fe3b84c529d60bb837127cbe255f6707671339f3780c
5dd4c97244333f00067ab1ba 86d790ba0f2dfa394e0346ca4785608c4e48cc4bc227feee48abee764376fb7d
5dd4c97344333f00067ab1bc 78b269f7796b56be4a5ede77b095cc1d956e8b271e026ba3c19ada4d444727ae
5dd4c97427889b0006b779aa 5306e808b77ff6a297d4ba8229d50df9069570dca489b0d3d32ed7a2179979e7
5dd4c97c44333f00067ab1c4 6a28052c2d5fb6e27eccce5bcd62bcdeedcbe481d8951019cbeec3d3b0503c4e
5dd4c97f27889b0006b779ae 681704e7829adf88ffaf76732797640c0d47f3bd70bee39d651037eff03fbab0
5dd4c97f44333f00067ab1c6 1d0cd47f9ee198efd52e1a5edec3c9f01403deb325eb25ac98db1ec977bbfe37
5dd4c98227889b0006b779b2 9a3d6b5cc8105e3a3c7d0de6ec84e1ca1d0fa2daae5d4cb1c3bf12012fa3cb70
5dd4c98544333f00067ab1ca ba84d1a048e0810fd438693c157949d2efbccb1a62e3b5f415d9858e86c6591b
5dd4c98e44333f00067ab1ce d181db15cd3bc4059dbca41b8cd09dbc1d689b4d75e8f4848b653beef26f9936
5dd4c99227889b0006b779bc a0a892c57aa3529b6b90dc15dee76ca8b6534b30fb51dded3f3236aac177c53c
5dd4d35bd48f840006f14474 febae675079592f222ce1942ee2fdaee30fbb2d5b46ee71223163330b6e3382e
5dd4d36fd48f840006f1447a 4b5a802affec79b06ce94cd913027c2f20b843b39713db98c2f068bc7f30d623
5dd4d37350e04e0006f55ebb def3552d5d66b40fd3eb483342d449de5fa592ee1d2f31981a199f06cb837e53
5dd4d3b8d48f840006f14481 f77f61c18a451acc6bdd389040c1f0ff823dafa3c55c4d09941a61d19928649e
5dd4d3bad48f840006f14483 58582fbb499e7424ea296054af3deafeb7672c62627dd2d7a98b3eca3ce27650
5dd4d3c350e04e0006f55ec7 c0348d07a7d12f6cad1d61418e1bb7abf9f5e1ee032583d7d514e7353ba074b0
5dd4d3f5d48f840006f14487 139e4215b7f5a14648b1a1f0969b3c83c08d527f8e91de56f3c0b06ee59702a8
5dd4d3f750e04e0006f55ece 1cf1e8220ef444b79981f9bcdd49a8f7e387a3bb4b71deb74aa1e3438037d48b
5dd4d401d48f840006f1448d f667f68b853faf92594d865d446212ef8bacfb9c421cbfcc6b8ac9cfd8cf82bb
5dd4d402d48f840006f1448f 2566d405a822f8d7e336ea897b37cce05d59270d225d801e9c7c8e95864bfcf6
5dd4d40f50e04e0006f55ed8 2c64f1e1224c253d13801efd976085419546a877ff09f489b77130e11559816b
5dd4d41ad48f840006f1449d 744060643d5028391a39be4cf94f90cb4c7f5d216ae186cc5201d6028478fbe2
5dd5ffb0d48f840006f14bff 30b8c6c4a2f0a3afe76f28c6bfb1a6f9416e35dfeb6b907fb1ce079ae926849b
5dd5ffbb50e04e0006f56653 ea40ed226e3a60ce060fd4e54d9b4d19d1aa9347c5977f82b66ef2c8ea5bf759
5dd6016550e04e0006f56674 cc6799c2e6f48d2e57bcce2847722f347e69fd9529c7602e86ccb2821c1e8457
5dd602ab50e04e0006f5667d d92f35ac488e01428d0f75152c66b0f7c3c7d90038abd26d2e5318eb3fa98686
5dd605af50e04e0006f5668d a6adedc7c910b7e5a788e36fce278fb9831b4879a462abe3d45c60e3b1840bbd
data/site2/F8/
floor_info.json 37beacfbd6ffdc522c82a4b3187e176d04b7ee34da8550301ec9f50d30d97519
geojson_map.json cb66b1eb7d852d7a480d117534da243194dd8193743e0d8ad4c1921243c8bf25
5dd4da7d50e04e0006f55f19 b6f66ada7fd638ae8f4581b58950ce2d0e38c19c86ae3956b0c3049ca5cfd091
5dd4da9cd48f840006f144e0 af65e015f712406668cb616e4310e689f02e4ee7a18f922ce77fa086fb29d785
5dd4da9d50e04e0006f55f1f 0f53349b4ca5e24e2c58645d44d1eca3abc35c5edb50d371a04d7b93f35721fb
5dd4da9e50e04e0006f55f21 8318d6c458005edb9fbb1e185f0fc2c60a0e4d698ec1b2c3f441dfb895016954
5dd4daa2d48f840006f144e5 e48a400858a5528310a67f95ee9e4f248987212cc11c234190e3bbeca50126f9
5dd4daa850e04e0006f55f29 62762a8172d7267d4711e84151226166c71dcdc573c9e1499fd326d1e61037d7
5dd4dab150e04e0006f55f2f 1446c95147899188d2cb82a40d21daa865f6b23b3d4ed0bd51b39ced4399cf2f
5dd4e29150e04e0006f55fbc 251916c26fbc7099066f97af4dc46c166273e8f312faadff1a4d0a7c95783d87
5dd4e2a8d48f840006f1456d 6c369eff414083e06cc11ff452ad30b2c4c049ad71d8e15e8bffb6a73bef948f
5dd4e2c650e04e0006f55fc7 87e1f588c795dcce242d87256c6dee3dccda88f5d6d6e29053c1607cdd203aca
5dd4e32450e04e0006f55fe7 e23eacd18a7d5104433f8b8091ed5aa06607a0a827d86cd6a27a86e59e3d1d96
5dd4e33850e04e0006f55fef 2e62b292ec6b1988588afe84ed59429a2f86878eb810a8f691209a61ecf710e7
5dd4e33cd48f840006f14597 caa69f6dd3a25d8bc8fa4dacaf45cea49286354c983d1355b029bee5a56492c3
5dd4e33d50e04e0006f55ff5 a81a89e2cad071da3bcbdd0a2a8c6b0bfa022a4793a0cca1bd95054adcbe4aa3
5dd4e33fd48f840006f14599 b1733f69c070fe54fd9cf2757254d7a33a4dfca282c829d1088ce0d1325a2359
5dd5f8d650e04e0006f565b9 c9e9e9d9bdf7eeb2444d2ad6c46a5c0cf413b2b1a84711556acf000a96792698
5dd5f910d48f840006f14b73 c2e29598ddef4e600efd815635fcd0f1ac2912717980330e61da64388b418ca8
5dd5f931d48f840006f14b7f 3651f6c80d9d454701b6d7d9b944d51a5187c4e4a1629eb0fb2079bdfff060a0
5dd5f967d48f840006f14b8d 4f500d8181f5b857b9cd80cc914dbcc5b5140f544dda44cae86edd14bd1885a3
5dd5f976d48f840006f14b95 3f9dc2b758b4bb3621601bbef13f24158308e9c34168c658f40fab54f0f98d9c
5dd5fc58d48f840006f14bc1 283b8f0ceaf30971b45bc1ff67cfbe2269873aa652afb558d0968938e37e534e
5ddbb8dac5b77e0006b17a3f 7113e4c7ef146704f1480d954f2818e5f1b4685e318138e15a65294848964baa
5ddbb8dcc5b77e0006b17a43 b04d77b6e6ad150286889345752e51429f3a076a2c8c643391b5dc026f57b909
5ddbb8e0c5b77e0006b17a45 0abb9c6fe44d79a8eec6a704b284500c7b54d5be1941ebd539b50bc795048d8c
5ddbb90a9191710006b57709 2d0c3f7569dd05d8a79456f91591c7b7b05d94eae2565e19bbfce93d95318e91
5ddbb9109191710006b5770d aa3c6e5da22fddddd40b47ce3385295fe0b58073dd53190c62f4cc0c7b387ba5
5ddbb912c5b77e0006b17a4d dd924f4670fff38c47d823891b18814b2b76bd809691a887ed2b68552905c580
5ddbb91ac5b77e0006b17a51 3f600058725918484e668a5b7dd355e416864f3a7973d73784b3c95ee94e2e71
"""
ILC2020.files, ILC2020.urls = _layout(_MANIFEST)
