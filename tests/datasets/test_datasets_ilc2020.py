"""L1 ILC2020 loader: Indoor Location Competition 2.0 sample traces.

Tiny synthetic traces and floor plans written in the official format check the parser with
known answers (scan grouping, cached WiFi entries, interpolated positions, IMU alignment, the
GeoJSON-to-metres scaling), without network. The real-data checks at the end run only when the
repository's files are under ``$INDOORLOC_DATA/ilc2020`` (default
``~/.cache/indoorloc/datasets/ilc2020``) and pin counts measured on them.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from conftest import DATA_ROOT
from indoorloc.apps import FloorMap
from indoorloc.apps.pdr import imu_arrays
from indoorloc.datasets import DATASETS, dataset_info, load_dataset
from indoorloc.datasets.ilc2020 import _COMMIT, IMU_CHANNELS, ILC2020, floor_level, read_floor_plan, read_trace

T0 = 1_600_000_000_000  # Unix ms of the synthetic traces
HEADER = ("#\tstartTime:{t0}\n"
          "#\tSiteID:abc\tSiteName:Mall\tFloorId:f1\tFloorName:{floor}\n"
          "#\tBrand:ACME\tModel:P1\tAndroidName:10\tAPILevel:29\n")


def trace_text(floor="F1", *, wifi=(), beacons=(), waypoints=((0, 0.0, 0.0), (10_000, 10.0, 0.0)),
               imu_ms=range(0, 10_001, 20), gyro_ms=None, extra=()) -> str:
    """A trace in the repository's format; times in ms after T0."""
    lines = [HEADER.format(t0=T0, floor=floor)]
    for ms in imu_ms:
        k = ms / 1000.0
        lines.append(f"{T0 + ms}\tTYPE_ACCELEROMETER\t{k:.3f}\t0.5\t9.81\t3\n")
        lines.append(f"{T0 + ms}\tTYPE_ACCELEROMETER_UNCALIBRATED\t{k:.3f}\t0.5\t9.81\t0.0\t0.0\t0.0\t3\n")
        lines.append(f"{T0 + ms}\tTYPE_MAGNETIC_FIELD\t20.0\t-5.0\t-40.0\t3\n")
        lines.append(f"{T0 + ms}\tTYPE_ROTATION_VECTOR\t0.0\t0.0\t0.6\t3\n")
    for ms in (imu_ms if gyro_ms is None else gyro_ms):
        lines.append(f"{T0 + ms}\tTYPE_GYROSCOPE\t{ms / 1000.0:.3f}\t0.0\t-0.1\t3\n")
    for ms, x, y in waypoints:
        lines.append(f"{T0 + ms}\tTYPE_WAYPOINT\t{x}\t{y}\n")
    for ms, ssid, bssid, rssi, freq, seen_ms in wifi:
        lines.append(f"{T0 + ms}\tTYPE_WIFI\t{ssid}\t{bssid}\t{rssi}\t{freq}\t{T0 + seen_ms}\n")
    for ms, uuid, major, minor, rssi, mac in beacons:
        lines.append(f"{T0 + ms}\tTYPE_BEACON\t{uuid}\t{major}\t{minor}\t-56\t{rssi}\t3.5\t{mac}\t{T0 + ms}\n")
    lines += list(extra)
    lines.append(f"#\tendTime:{T0 + 10_000}\n")
    return "".join(lines)


def geojson(lon0=120.0, lat0=30.0, dlon=0.001, dlat=0.0005) -> dict:
    """An outline (lon/lat box, as a MultiPolygon) and one unit: the box's south-west quarter."""
    box = [[lon0, lat0], [lon0 + dlon, lat0], [lon0 + dlon, lat0 + dlat], [lon0, lat0 + dlat], [lon0, lat0]]
    quarter = [[lon0, lat0], [lon0 + dlon / 2, lat0], [lon0 + dlon / 2, lat0 + dlat / 2], [lon0, lat0 + dlat / 2],
               [lon0, lat0]]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"type": "floor"}, "geometry": {"type": "MultiPolygon",
                                                                           "coordinates": [[box]]}},
        {"type": "Feature", "properties": {"name": "shop"}, "geometry": {"type": "Polygon",
                                                                          "coordinates": [quarter]}}]}


def make_root(tmp_path, traces: dict, *, floors=("F1",), site="site1", width=100.0, height=50.0):
    """Write a fake repository layout; returns (ILC2020 subclass declaring exactly these files, root)."""
    entries = []

    def put(rel, text):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        entries.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))

    for floor in floors:
        put(f"data/{site}/{floor}/floor_info.json", json.dumps({"map_info": {"height": height, "width": width}}))
        put(f"data/{site}/{floor}/geojson_map.json", json.dumps(geojson()))
    for (floor, trace_id), text in traces.items():
        put(f"data/{site}/{floor}/path_data_files/{trace_id}.txt", text)

    class Fake(ILC2020):
        files = {"all": tuple(sorted(entries))}
        urls = {}
    return Fake, tmp_path


def rehash(Fake, root):
    """``Fake`` with the sha256 of its files recomputed after a test rewrote some of them."""
    class Rehashed(Fake):
        files = {"all": tuple((rel, hashlib.sha256((root / rel).read_bytes()).hexdigest())
                              for rel, _ in Fake.files["all"])}
    return Rehashed


# ---------------------------------------------------------------- helpers
def test_floor_names_map_to_signed_levels_without_zero():
    assert [floor_level(n) for n in ("F1", "F8", "B1", "B2", "1F", "2b", " f3 ")] == [1, 8, -1, -2, 1, -2, 3]
    for bad in ("G", "F0", "L1", "F", ""):
        with pytest.raises(ValueError):
            floor_level(bad)


def test_read_trace_parses_every_record_kind(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text(trace_text(wifi=[(1500, "Cafe Wifi", "aa:bb:cc:00:00:01", -61, 2412, 1400)],
                               beacons=[(1200, "UUID-1", "7", "9", -80, "E0:00:00:00:00:01")],
                               extra=[f"{T0}\tTYPE_BLUE\t1\t2\t3\n"]))
    tr = read_trace(path)
    assert tr["header"]["FloorName"] == "F1" and tr["header"]["Model"] == "P1"
    assert tr["header"]["startTime"] == str(T0)
    assert tr["acc"]["t"].dtype == np.int64 and tr["acc"]["xyz"].shape == (501, 3)  # uncalibrated lines not read
    assert tr["acc"]["xyz"][50].tolist() == [1.0, 0.5, 9.81]
    assert tr["waypoint"]["xy"].tolist() == [[0.0, 0.0], [10.0, 0.0]]
    wifi = tr["wifi"]
    assert (wifi["t"][0] - T0, wifi["ssid"][0], wifi["bssid"][0], wifi["rssi"][0], wifi["freq"][0],
            wifi["seen"][0] - T0) == (1500, "Cafe Wifi", "aa:bb:cc:00:00:01", -61.0, 2412, 1400)
    b = tr["beacon"]
    assert (b["uuid"][0], b["major"][0], b["minor"][0], b["tx_power"][0], b["rssi"][0], b["distance"][0],
            b["mac"][0]) == ("UUID-1", "7", "9", -56.0, -80.0, 3.5, "E0:00:00:00:00:01")
    only = read_trace(path, ("waypoint",))
    assert set(only) == {"header", "waypoint"}


def test_read_trace_refuses_malformed_lines(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text(trace_text(extra=[f"{T0}\tTYPE_WIFI\tssid-only\n"]))
    with pytest.raises(ValueError, match="1 of 1 TYPE_WIFI lines are malformed"):
        read_trace(path, ("wifi",))
    path.write_text(trace_text(extra=[f"{T0}\tTYPE_WAYPOINT\tnorth\t3.0\n"]))
    with pytest.raises(ValueError, match="non-numeric"):
        read_trace(path, ("waypoint",))
    path.write_text(trace_text(extra=[f"t{T0}\tTYPE_WAYPOINT\t1.0\t3.0\n"]))  # a broken timestamp is not skipped
    with pytest.raises(ValueError, match="1 of 3 TYPE_WAYPOINT lines are malformed"):
        read_trace(path, "waypoint")
    with pytest.raises(ValueError, match="unknown record kind"):
        read_trace(path, ("gps",))


def test_read_trace_takes_one_kind_by_name_and_field_values_that_look_like_types(tmp_path):
    path = tmp_path / "t.txt"  # an SSID that reads like a line type is still just an SSID
    path.write_text(trace_text(wifi=[(1500, "TYPE_WIFI", "aa:bb:cc:00:00:01", -61, 2412, 1400)]))
    wifi = read_trace(path, "wifi")
    assert set(wifi) == {"header", "wifi"} and wifi["wifi"]["ssid"].tolist() == ["TYPE_WIFI"]


def test_read_trace_reads_windows_line_endings_like_unix_ones(tmp_path):
    text = trace_text(wifi=[(1500, "", "aa:bb:cc:00:00:01", -61, 2412, 1400)],
                      beacons=[(1200, "UUID-1", "7", "9", -80, "E0:00:00:00:00:01")])
    (tmp_path / "lf.txt").write_bytes(text.encode())
    (tmp_path / "crlf.txt").write_bytes(text.replace("\n", "\r\n").encode())
    lf, crlf = read_trace(tmp_path / "lf.txt"), read_trace(tmp_path / "crlf.txt")
    assert crlf["header"] == lf["header"] and crlf["wifi"]["ssid"].tolist() == [""]  # an empty SSID stays empty
    for kind in ("acc", "rv", "waypoint", "wifi", "beacon"):
        for key, value in lf[kind].items():
            np.testing.assert_array_equal(crlf[kind][key], value, err_msg=f"{kind}.{key}")


# ---------------------------------------------------------------- floor plan
def test_floor_plan_scales_the_outline_box_to_floor_info_metres(tmp_path):
    (tmp_path / "g.json").write_text(json.dumps(geojson()))
    (tmp_path / "i.json").write_text(json.dumps({"map_info": {"height": 50.0, "width": 100.0}}))
    plan = read_floor_plan(tmp_path / "g.json", tmp_path / "i.json", level=-1)
    walls = plan["walls"]
    assert walls.shape == (8, 4) and walls.dtype == np.float64  # 4 outline + 4 unit edges
    np.testing.assert_allclose(walls[:4], [[0, 0, 100, 0], [100, 0, 100, 50], [100, 50, 0, 50], [0, 50, 0, 0]],
                               atol=1e-9)
    np.testing.assert_allclose(walls[4:], [[0, 0, 50, 0], [50, 0, 50, 25], [50, 25, 0, 25], [0, 25, 0, 0]], atol=1e-9)
    assert plan["wall_type"].tolist() == [0] * 4 + [1] * 4 and plan["materials"][1] == "unit outline"
    assert plan["wall_floor"].tolist() == [-1] * 8 and plan["bounds"].tolist() == [0, 0, 100, 50]
    fmap = FloorMap.from_dict(plan)  # L5 reads the dict as it is
    assert fmap.crosses([[10.0, 10.0]], [[60.0, 10.0]]).tolist() == [True]  # leaves the unit
    assert fmap.crosses([[60.0, 30.0]], [[90.0, 40.0]]).tolist() == [False]


def test_floor_plan_drops_zero_length_edges_and_closes_rings():
    feature = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 0], [1, 1]]]}  # repeated vertex, open ring
    plan = read_floor_plan({"features": [{"geometry": feature}]}, {"map_info": {"width": 2.0, "height": 2.0}})
    np.testing.assert_allclose(plan["walls"], [[0, 0, 2, 0], [2, 0, 2, 2], [2, 2, 0, 0]])


# ---------------------------------------------------------------- WiFi
WIFI = [  # (ms, ssid, bssid, dBm, MHz, last seen ms)
    (2500, "A", "aa:00:00:00:00:02", -70, 2412, 2300),
    (2500, "B", "aa:00:00:00:00:01", -55, 5805, 2400),
    (2500, "B", "aa:00:00:00:00:01", -58, 5825, 2450),   # same AP on a second channel, seen later: kept
    (2500, "old", "aa:00:00:00:00:09", -40, 2437, -9000),  # cached from an earlier scan (11.5 s old)
    (5000, "A", "aa:00:00:00:00:02", -72, 2412, 4800),
    (5000, "A2", "aa:00:00:00:00:03", -80, 2462, 4900),
    (5000, "A2", "aa:00:00:00:00:03", -81, 2462, 4900),   # a tie in last-seen time: the stronger is kept
    (12000, "A", "aa:00:00:00:00:02", -75, 2412, 11900),  # after the last waypoint
]


def test_wifi_rows_are_scans_with_interpolated_positions(tmp_path):
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(wifi=WIFI)})
    t = Fake(root).load("all")
    assert t.meta["modality"] == "wifi_rssi" and t.meta["units"] == "dBm" and t.X.dtype == np.float32
    assert t.meta["feature_names"] == ("aa:00:00:00:00:01", "aa:00:00:00:00:02", "aa:00:00:00:00:03")
    assert t.meta["ssids"] == ("B", "A", "A2") and t.meta["wifi_max_age_s"] == 2.0
    np.testing.assert_array_equal(t.X, [[-58, -70, np.nan], [np.nan, -72, -80]])
    np.testing.assert_allclose(t.pos, [[2.5, 0.0], [5.0, 0.0]])  # 10 m in 10 s, linear in time
    np.testing.assert_allclose(t.groups["time"], [(T0 + 2500) / 1000, (T0 + 5000) / 1000])
    assert t.floor.tolist() == [1, 1] and t.groups["trajectory"].tolist() == [0, 0]
    assert t.meta["trajectory_names"] == ("5f0000000000000000000001",) and t.groups["device"][0] == "ACME P1"
    assert t.ids.tolist() == [f"5f0000000000000000000001/{T0 + 2500}", f"5f0000000000000000000001/{T0 + 5000}"]
    assert t.meta["n_outside_waypoints"] == 1 and t.meta["crs"] == "local" and t.meta["pos_units"] == "m"
    assert t.meta["floor_plan"]["walls"].shape == (8, 4) and len(t.meta["sha256"]) == 3


def test_wifi_options_keep_cached_entries_and_unlabelled_scans(tmp_path):
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(wifi=WIFI)})
    t = Fake(root, wifi_max_age=None, outside_waypoints="nan").load("all")
    assert t.meta["feature_names"][-1] == "aa:00:00:00:00:09" and t.X[0, -1] == -40
    assert len(t) == 3 and np.isnan(t.pos[2]).all() and np.isfinite(t.pos[:2]).all()
    assert t.meta["n_outside_waypoints"] == 1 and t.X[2, 1] == -75


def test_scans_emptied_by_the_age_filter_disappear(tmp_path):
    wifi = [(3000, "x", "aa:00:00:00:00:01", -60, 2412, 0), (6000, "x", "aa:00:00:00:00:01", -65, 2412, 5900)]
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(wifi=wifi)})
    assert Fake(root).load("all").groups["time"].tolist() == [(T0 + 6000) / 1000]
    assert len(Fake(root, wifi_max_age=3.5).load("all")) == 2


# ---------------------------------------------------------------- BLE
BEACONS = [  # (ms, uuid, major, minor, dBm, MAC)
    (1000, "U", "0", "0", -80, "E0:00:00:00:00:02"),
    (1300, "U", "0", "0", -70, "E0:00:00:00:00:01"),
    (1600, "U", "0", "0", -90, "E0:00:00:00:00:02"),
    (2600, "V", "1", "2", -60, "E0:00:00:00:00:03"),
]


def test_ble_rows_per_timestamp_or_per_window(tmp_path):
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(beacons=BEACONS)})
    t = Fake(root, modality="ble").load("all")
    assert t.meta["modality"] == "ble_rssi" and t.X.shape == (4, 3) and t.meta["ble_window_s"] is None
    assert t.meta["feature_names"] == ("E0:00:00:00:00:01", "E0:00:00:00:00:02", "E0:00:00:00:00:03")
    assert t.meta["ibeacon_ids"] == ("U_0_0", "U_0_0", "V_1_2")  # one iBeacon id, two transmitters
    assert (np.isfinite(t.X).sum(axis=1) == 1).all()
    np.testing.assert_allclose(t.pos[:, 0], [1.0, 1.3, 1.6, 2.6])
    w = Fake(root, modality="ble", ble_window=1.0).load("all")  # windows [1.0, 2.0) and [2.0, 3.0) s
    np.testing.assert_array_equal(w.X, [[-70, -85, np.nan], [np.nan, np.nan, -60]])  # mean of -80 and -90 dBm
    np.testing.assert_allclose(w.groups["time"] - T0 / 1000, [(1.0 + 1.3 + 1.6) / 3, 2.6])
    assert w.ids.tolist() == [f"5f0000000000000000000001/{T0 + 1000}", f"5f0000000000000000000001/{T0 + 2000}"]


# ---------------------------------------------------------------- IMU and waypoints
def test_imu_rows_follow_the_accelerometer_clock(tmp_path):
    gyro_ms = range(10, 5000, 20)  # a slower-starting, shorter gyroscope on another clock
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(gyro_ms=gyro_ms)})
    t = Fake(root, modality="imu", outside_waypoints="nan").load("all")
    assert t.meta["channels"] == IMU_CHANNELS and t.X.shape == (501, 13) and t.X.dtype == np.float32
    assert t.meta["rate_hz"] == 50.0 and t.meta["channel_units"][:4] == ("m/s^2",) * 3 + ("rad/s",)
    np.testing.assert_allclose(t.X[:, 0], np.arange(501) * 0.02, atol=1e-6)
    # the gyroscope is interpolated inside its span (x = time in s) and NaN outside it
    gyr = t.X[:, 3]
    inside = (t.groups["time"] * 1000 - T0 >= 10) & (t.groups["time"] * 1000 - T0 <= 4990)
    np.testing.assert_allclose(gyr[inside], (t.groups["time"][inside] * 1000 - T0) / 1000.0, atol=1e-5)
    assert np.isnan(gyr[~inside]).all() and (~inside).sum() == 501 - 249
    np.testing.assert_allclose(t.X[:, 12], 0.8, atol=1e-6)  # rv_w = sqrt(1 - 0.6^2)
    arrays = imu_arrays(t)  # L5 fills the NaN gaps
    assert np.isfinite(arrays["gyro"]).all() and arrays["acc"].shape == (501, 3)
    assert t.meta["n_outside_waypoints"] == 0 and np.isfinite(t.pos).all()


def test_waypoint_tables_and_outside_rows_of_imu(tmp_path):
    wps = ((2000, 1.0, 2.0), (6000, 5.0, 2.0), (8000, 5.0, 6.0))
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(waypoints=wps)})
    w = Fake(root, modality="waypoints").load("all")
    assert w.X.shape == (3, 0) and w.pos.tolist() == [[1, 2], [5, 2], [5, 6]] and w.meta["modality"] == "waypoints"
    imu = Fake(root, modality="imu").load("all")  # rows before 2 s and after 8 s dropped
    assert len(imu) == 301 and imu.meta["n_outside_waypoints"] == 200
    np.testing.assert_allclose(imu.pos[imu.groups["time"] == (T0 + 7000) / 1000], [[5.0, 4.0]])


def test_several_floors_trajectory_codes_and_floor_plans(tmp_path):
    traces = {("F1", "5f00000000000000000000b1"): trace_text(wifi=WIFI[:2]),
              ("B1", "5f00000000000000000000a1"): trace_text("B1", wifi=WIFI[4:5]),
              ("F1", "5f00000000000000000000a2"): trace_text(wifi=WIFI[4:5])}
    Fake, root = make_root(tmp_path, traces, floors=("F1", "B1"))
    assert Fake.floors() == {"site1": ("B1", "F1")}
    t = Fake(root, floor="all").load("all")
    names = t.meta["trajectory_names"]
    assert names == ("5f00000000000000000000a1", "5f00000000000000000000a2", "5f00000000000000000000b1")
    assert t.floor.tolist() == [-1, 1, 1] and t.groups["trajectory"].tolist() == [0, 1, 2]
    assert t.meta["floors"] == (-1, 1) and t.meta["floor_names"] == ("B1", "F1")
    plan = t.meta["floor_plan"]
    assert sorted(np.unique(plan["wall_floor"]).tolist()) == [-1, 1] and len(plan["walls"]) == 16
    assert FloorMap.from_dict(plan).floors == (-1, 1)
    only = Fake(root, floor=["b1"]).load("all")
    assert only.floor.tolist() == [-1] and len(only.meta["sha256"]) == 3


def test_frame_origins_align_floors_with_the_documented_formula(tmp_path):
    R, lon0, lat0, dlon, dlat = 6378137.0, 120.0, 30.0, 0.001, 0.0005
    width = R * np.deg2rad(dlon) * np.cos(np.deg2rad(lat0 + dlat / 2))  # sizes as floor_info states them
    height = R * np.deg2rad(dlat)
    traces = {("F1", "5f00000000000000000000a1"): trace_text(), ("B1", "5f00000000000000000000b1"): trace_text("B1")}
    Fake, root = make_root(tmp_path, traces, floors=("F1", "B1"), width=width, height=height)
    # B1: an outline of the same size, 0.0002 deg further east and 0.0001 deg further south
    (root / "data/site1/B1/geojson_map.json").write_text(json.dumps(geojson(lon0 + 0.0002, lat0 - 0.0001)))
    t = rehash(Fake, root)(root, floor="all", modality="waypoints").load("all")
    origins = t.meta["frame_origin_lonlat"]
    assert t.meta["floors"] == (-1, 1) and "origin_lonlat" not in t.meta["floor_plan"]
    np.testing.assert_allclose(origins, [[lon0 + 0.0002, lat0 - 0.0001], [lon0, lat0]], rtol=0, atol=1e-12)
    (lon_b, lat_b), (lon_f, lat_f) = origins
    shift = [R * np.deg2rad(lon_b - lon_f) * np.cos(np.deg2rad(lat_f)), R * np.deg2rad(lat_b - lat_f)]
    # F1's own map scale puts B1's origin (lon_b, lat_b) at 0.2 width east and 0.2 height south
    np.testing.assert_allclose(shift, [0.2 * width, -0.2 * height], atol=1e-3)
    assert abs(shift[0] - 19.28) < 0.01 and abs(shift[1] + 11.13) < 0.01  # metres
    one = rehash(Fake, root)(root, floor="B1", modality="waypoints").load("all")
    np.testing.assert_allclose(one.meta["frame_origin_lonlat"], [[lon0 + 0.0002, lat0 - 0.0001]])


def test_invalid_options_fail_fast(tmp_path):
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text()})
    for kwargs in ({"site": "site9"}, {"floor": "F7"}, {"floor": []}, {"modality": "gps"},
                   {"outside_waypoints": "clip"}, {"wifi_max_age": 0}, {"ble_window": -1.0}, {"wifi_max_age": True},
                   {"ble_window": 0.0004}, {"wifi_max_age": float("nan")},
                   {"floor": 0}, {"floor": "G"}, {"floor": True}):
        with pytest.raises(ValueError):
            Fake(root, **kwargs)


def test_floors_can_be_named_or_given_as_levels(tmp_path):
    traces = {("F1", "5f00000000000000000000a1"): trace_text(), ("B1", "5f00000000000000000000b1"): trace_text("B1")}
    Fake, root = make_root(tmp_path, traces, floors=("F1", "B1"))
    for floor, expected in (("F1", ("F1",)), (1, ("F1",)), ("1f", ("F1",)), (np.int64(-1), ("B1",)),
                            (["F1", 1, "b1"], ("B1", "F1")), ("ALL", ("B1", "F1"))):
        assert Fake(root, floor=floor).floor == expected, floor
    t = Fake(root, floor=[1, "F1"], modality="waypoints").load("all")  # a floor named twice is read once
    assert len(t) == 2 and t.floor.tolist() == [1, 1]


def test_trajectory_codes_do_not_depend_on_the_modality(tmp_path):
    # the first trace (by id) has no iBeacon reading: it adds no BLE row but keeps its code
    traces = {("F1", "5f00000000000000000000a1"): trace_text(wifi=WIFI[:2]),
              ("F1", "5f00000000000000000000a2"): trace_text(wifi=WIFI[4:5], beacons=BEACONS[:1])}
    Fake, root = make_root(tmp_path, traces)
    wifi, ble = Fake(root).load("all"), Fake(root, modality="ble").load("all")
    assert wifi.meta["trajectory_names"] == ble.meta["trajectory_names"]
    assert wifi.groups["trajectory"].tolist() == [0, 1] and ble.groups["trajectory"].tolist() == [1]


def test_rows_sharing_a_timestamp_are_refused_not_given_duplicate_ids(tmp_path):
    wps = ((0, 0.0, 0.0), (5000, 5.0, 0.0), (5000, 5.0, 1.0), (10_000, 10.0, 0.0))
    Fake, root = make_root(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(waypoints=wps)})
    with pytest.raises(ValueError, match="share a timestamp"):
        Fake(root, modality="waypoints").load("all")


class _Response(io.BytesIO):
    """A stand-in for ``urllib``'s HTTP response: a body plus its ``Content-Encoding`` header."""

    def __init__(self, body: bytes, gzipped: bool):
        super().__init__(gzip.compress(body) if gzipped else body)
        self.headers = {"Content-Encoding": "gzip"} if gzipped else {}


def _serve(monkeypatch, served: dict, *, gzipped=False, corrupt=(), fail=False) -> list:
    """Patch ``urllib.request.urlopen`` to serve ``served`` (relpath -> bytes) without network;
    returns the list of requested urls."""
    asked = []

    def urlopen(request, timeout=None):
        asked.append(request.full_url)
        assert request.get_header("Accept-encoding") == "gzip"
        if fail:
            raise urllib.error.URLError("network unreachable")
        rel = request.full_url.split("example.invalid/", 1)[1]
        return _Response(served[rel] + (b"x" if rel in corrupt else b""), gzipped)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", lambda s: None)  # no waiting between retries
    return asked


def _remote(tmp_path, traces, **kwargs):
    """A fake repository written under ``tmp_path`` and served by url; returns (class, root, served)."""
    Fake, root = make_root(tmp_path, traces, **kwargs)
    served = {rel: (root / rel).read_bytes() for rel, _ in Fake.files["all"]}

    class Remote(Fake):
        urls = {rel: f"https://example.invalid/{rel}" for rel in served}
    return Remote, root, served


@pytest.mark.parametrize("gzipped", [False, True])
def test_download_fetches_only_the_absent_files(tmp_path, monkeypatch, gzipped):
    traces = {("F1", "5f0000000000000000000001"): trace_text(wifi=WIFI),
              ("F1", "5f0000000000000000000002"): trace_text(wifi=WIFI[4:6])}
    Remote, root, served = _remote(tmp_path, traces)
    asked = _serve(monkeypatch, served, gzipped=gzipped)
    missing = "data/site1/F1/path_data_files/5f0000000000000000000002.txt"
    (root / missing).unlink()
    t = Remote(root, download=True).load("all")
    assert asked == [f"https://example.invalid/{missing}"] and (root / missing).read_bytes() == served[missing]
    assert t.meta["trajectory_names"] == ("5f0000000000000000000001", "5f0000000000000000000002")
    assert len(Remote.urls) == 4 and "urls" not in vars(Remote(root))  # the class keeps its full manifest
    asked.clear()
    Remote(root, download=True).load("all")
    assert asked == []  # nothing absent, nothing fetched
    assert not list(root.rglob("*.part"))


def test_download_of_several_floors_puts_each_floor_map_in_its_own_folder(tmp_path, monkeypatch):
    # every floor has a floor_info.json and a geojson_map.json: a file must only land at its own path
    traces = {("F1", "5f00000000000000000000b1"): trace_text(wifi=WIFI[:2]),
              ("B1", "5f00000000000000000000a1"): trace_text("B1", wifi=WIFI[4:5])}
    Remote, root, _ = _remote(tmp_path, traces, floors=("F1", "B1"))
    (root / "data/site1/B1/floor_info.json").write_text(json.dumps({"map_info": {"height": 60.0, "width": 80.0}}))
    Remote = rehash(Remote, root)
    served = {rel: (root / rel).read_bytes() for rel in Remote.urls}
    for rel in served:
        (root / rel).unlink()
    asked = _serve(monkeypatch, served, gzipped=True)
    t = Remote(root, download=True, floor="all").load("all")
    assert sorted(asked) == sorted(f"https://example.invalid/{rel}" for rel in served)  # each file once
    assert all((root / rel).read_bytes() == body for rel, body in served.items())
    plan = t.meta["floor_plan"]
    assert plan["bounds"].tolist() == [0, 0, 100, 60]  # F1 100 x 50 m and B1 80 x 60 m
    b1 = plan["walls"][plan["wall_floor"] == -1]
    assert b1[:, [0, 2]].max() == 80.0 and b1[:, [1, 3]].max() == 60.0


def test_a_corrupted_or_failed_download_leaves_nothing_behind(tmp_path, monkeypatch):
    Remote, root, served = _remote(tmp_path, {("F1", "5f0000000000000000000001"): trace_text(wifi=WIFI)})
    missing = "data/site1/F1/path_data_files/5f0000000000000000000001.txt"
    (root / missing).unlink()
    _serve(monkeypatch, served, corrupt={missing})
    with pytest.raises(ValueError, match="checksum mismatch"):
        Remote(root, download=True).load("all")
    assert not (root / missing).exists() and not list(root.rglob("*.part"))
    asked = _serve(monkeypatch, served, fail=True)
    with pytest.raises(RuntimeError, match="could not download"):
        Remote(root, download=True).load("all")
    assert len(asked) == 3 and not (root / missing).exists()  # three tries
    with pytest.raises(FileNotFoundError, match="download=True"):
        Remote(root).load("all")  # download=False (the class default) never fetches


# ---------------------------------------------------------------- declarations
def test_manifest_declares_every_floor_with_pinned_urls():
    assert DATASETS.get("ilc2020") is ILC2020
    info = dataset_info("ilc2020")
    for key in ("modality", "units", "crs", "pos_units", "license", "doi", "citation", "url"):
        assert key in info, key
    assert info["splits"] == ("all",) and info["license"] == "MIT" and info["pos_units"] == "m"
    floors = ILC2020.floors()
    assert floors == {"site1": ("B1", "F1", "F2", "F3", "F4"),
                      "site2": ("B1", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")}
    entries = ILC2020.files["all"]
    assert len(entries) == 1123 and len({rel for rel, _ in entries}) == 1123
    for rel, digest in entries:
        assert re.fullmatch(r"[0-9a-f]{64}", digest), rel
        assert ILC2020.urls[rel].startswith(
            f"https://raw.githubusercontent.com/location-competition/indoor-location-competition-20/{_COMMIT}/data/")
    traces = [rel for rel, _ in entries if rel.endswith(".txt")]
    per_floor = {}
    for rel in traces:
        per_floor[tuple(rel.split("/")[1:3])] = per_floor.get(tuple(rel.split("/")[1:3]), 0) + 1
    table = {"site1": {"B1": 160, "F1": 120, "F2": 123, "F3": 117, "F4": 122},  # the class docstring's table
             "site2": {"B1": 70, "F1": 99, "F2": 45, "F3": 40, "F4": 27, "F5": 44, "F6": 67, "F7": 33, "F8": 28}}
    assert per_floor == {(site, f): n for site, row in table.items() for f, n in row.items()}
    assert len(traces) == 1095
    for site, row in table.items():
        for f in row:
            for name in ("floor_info.json", "geojson_map.json"):
                assert f"data/{site}/{f}/{name}" in ILC2020.urls
    one = ILC2020(root="/nonexistent", site="site2", floor="F4")
    assert len(one.files["all"]) == 29  # floor_info, geojson and 27 traces


# ---------------------------------------------------------------- real data
REAL = DATA_ROOT / "ilc2020"
SITE2_F4 = REAL / "data" / "site2" / "F4" / "geojson_map.json"
SITE1_F1 = REAL / "data" / "site1" / "F1" / "geojson_map.json"


@pytest.mark.skipif(not SITE2_F4.is_file(), reason=f"no ILC 2020 site2/F4 at {SITE2_F4.parent}")
def test_real_site2_f4_counts():
    kw = {"site": "site2", "floor": "F4", "root": REAL, "download": False}
    wp = load_dataset("ilc2020", modality="waypoints", **kw)
    assert len(wp) == 215 and len(np.unique(wp.groups["trajectory"])) == 27 and set(wp.floor.tolist()) == {4}
    assert wp.meta["floor_plan"]["bounds"][2:].round(2).tolist() == [236.71, 219.75]
    # the outline's lon/lat box in metres (equirectangular, WGS84 radius) is floor_info's width x height
    outline = np.concatenate([np.asarray(r) for poly in json.loads(SITE2_F4.read_text())["features"][0]["geometry"]
                              ["coordinates"] for r in poly])
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    metres = np.deg2rad(hi - lo) * 6378137.0 * np.array([np.cos(np.deg2rad((lo[1] + hi[1]) / 2)), 1.0])
    np.testing.assert_allclose(metres, wp.meta["floor_plan"]["bounds"][2:], atol=1e-3)
    np.testing.assert_array_equal(wp.meta["frame_origin_lonlat"], [lo])  # the frame's origin: the box's SW corner
    fmap = FloorMap.from_dict(wp.meta["floor_plan"])
    assert fmap.contains(wp.pos).all()
    wifi = load_dataset("ilc2020", **kw)
    assert wifi.X.shape == (490, 1715) and wifi.meta["n_outside_waypoints"] == 10
    assert np.isfinite(wifi.pos).all() and len(np.unique(wifi.ids)) == len(wifi)
    imu = load_dataset("ilc2020", modality="imu", outside_waypoints="nan", **kw)
    assert imu.X.shape == (54967, 13) and imu.meta["rate_hz"] == 50.0 and np.isfinite(imu.X).all()
    assert imu.meta["n_outside_waypoints"] == 1372 and np.isnan(imu.pos).any(axis=1).sum() == 1372
    quat = imu.X[:, 9:13].astype(np.float64)  # unit quaternions (rv_w completes x, y, z)
    assert np.abs(np.linalg.norm(quat, axis=1) - 1).max() < 1e-3


@pytest.mark.skipif(not SITE1_F1.is_file(), reason=f"no ILC 2020 site1/F1 at {SITE1_F1.parent}")
def test_real_site1_f1_counts():
    kw = {"site": "site1", "floor": "F1", "root": REAL, "download": False}
    wifi = load_dataset("ilc2020", **kw)
    assert wifi.X.shape == (2223, 2330) and len(np.unique(wifi.groups["trajectory"])) == 120
    assert np.isfinite(wifi.X).any(axis=0).all() and np.nanmax(wifi.X) < 0
    ble = load_dataset("ilc2020", modality="ble", **kw)
    assert ble.X.shape == (30857, 297) and (np.isfinite(ble.X).sum(axis=1) >= 1).all()
    assert ble.meta["ibeacon_ids"].count("9195B3AD-A9D0-4500-85FF-9FB0F65A5201_0_0") == 242  # one id, 242 MACs
