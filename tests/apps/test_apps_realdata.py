"""L5 on real traces: Indoor Location Competition 2.0 sample data (Microsoft Research, MIT licence).

Smartphone traces (accelerometer, gyroscope, magnetometer, rotation vector at 50 Hz, WiFi
scans) with surveyor-labelled waypoints and a GeoJSON floor plan, read with
``load_dataset("ilc2020", ...)`` (``site2/F4``: 27 traces; ``site1/F1``: 120 traces, used for
the WiFi + fusion check). Skipped unless the repository's files are under
``$INDOORLOC_DATA/ilc2020`` (default ``~/.cache/indoorloc/datasets/ilc2020``); fetch them with
``load_dataset("ilc2020", site="site2", floor="F4")`` (and ``site="site1", floor="F1"``).

The tables keep every row (``outside_waypoints="nan"``) so the trackers see each trace from its
first sample, as a phone would; the WiFi entries are those seen at most 2 s before the scan
(``wifi_max_age=2.0``, the loader's default).
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import DATA_ROOT
from indoorloc.apps import PDR, FloorMap, KalmanTracker, ParticleFilter, PDRFusion, StepDetector
from indoorloc.apps.pdr import imu_arrays, magnetic_heading, step_lengths, wrap_angle
from indoorloc.datasets import load_dataset

ROOT = DATA_ROOT / "ilc2020"
FLOOR = ROOT / "data" / "site2" / "F4"
DENSE = ROOT / "data" / "site1" / "F1"
pytestmark = pytest.mark.skipif(not (FLOOR / "geojson_map.json").is_file(), reason=f"no ILC 2020 sample at {FLOOR}")


def _load(site, floor, modality, **options):
    return load_dataset("ilc2020", root=ROOT, download=False, site=site, floor=floor, modality=modality,
                        outside_waypoints="nan", **options)


def _read(site, floor, wifi: bool = False) -> tuple[list[dict], tuple]:
    """Per trace: IMU arrays (``imu_arrays``), rotation vector, waypoints and (optionally) WiFi scans;
    plus the waypoint and WiFi tables."""
    imu, wp = _load(site, floor, "imu"), _load(site, floor, "waypoints")
    scans = _load(site, floor, "wifi") if wifi else None
    rv = [imu.meta["channels"].index(c) for c in ("rv_x", "rv_y", "rv_z", "rv_w")]
    traces = []
    for code in range(len(wp.meta["trajectory_names"])):
        rows = imu.groups["trajectory"] == code
        d = imu_arrays(imu[rows])
        d["rot"] = np.asarray(imu.X[rows][:, rv], dtype=np.float64)
        mine = wp.groups["trajectory"] == code
        d["wp_t"], d["wp"] = wp.groups["time"][mine], wp.pos[mine]
        if scans is not None:
            mine = scans.groups["trajectory"] == code
            d["scan_t"], d["X"], d["scan_pos"] = scans.groups["time"][mine], scans.X[mine], scans.pos[mine]
        traces.append(d)
    return traces, (wp, scans)


def _interp(t_query, t, pos):
    ok = np.isfinite(pos).all(axis=1)
    return np.stack([np.interp(t_query, t[ok], pos[ok, i]) for i in (0, 1)], axis=1)


def _rotation_heading(rot):
    """Heading (counter-clockwise from east) of the device +y axis from Android's rotation vector."""
    x, y, z, w = rot.T
    return np.arctan2(1 - 2 * (x * x + z * z), 2 * (x * y - z * w))


@pytest.fixture(scope="module")
def traces():
    paths, _ = _read("site2", "F4")
    det = StepDetector()
    for d in paths:
        d["steps"] = det.detect(d["acc"], d["t"])
        d["length"] = np.linalg.norm(np.diff(d["wp"], axis=0), axis=1).sum()
        seg_heading = np.arctan2(*np.diff(d["wp"], axis=0)[:, ::-1].T)
        seg = np.clip(np.searchsorted(d["wp_t"], d["steps"].t) - 1, 0, len(seg_heading) - 1)
        d["true_heading"] = seg_heading[seg]
    return paths


def test_step_detector_gives_human_step_lengths_on_real_walks(traces):
    steps = sum(len(d["steps"]) for d in traces)
    walked = sum(d["length"] for d in traces)
    assert len(traces) == 27 and 0.55 < walked / steps < 0.8  # 1275 m in 2010 steps: 0.634 m
    # Weinberg constant fitted leave-one-path-out predicts each path's length
    feature = [np.sum(np.maximum(d["steps"].peak - d["steps"].valley, 0) ** 0.25) for d in traces]
    rel = []
    for i, d in enumerate(traces):
        k = (walked - d["length"]) / (sum(feature) - feature[i])
        rel.append(abs(step_lengths(d["steps"], "weinberg", k).sum() / d["length"] - 1))
    assert np.median(rel) < 0.12  # measured: 8.1 % median, 11.0 % mean


def test_pdr_with_phone_heading_tracks_real_walks_from_a_known_start(traces):
    errors = []
    for i, d in enumerate(traces):
        train = [p for j, p in enumerate(traces) if j != i]
        k = sum(p["length"] for p in train) / sum(np.sum(np.maximum(p["steps"].peak - p["steps"].valley, 0) ** 0.25)
                                                   for p in train)
        offset = np.angle(np.mean(np.exp(1j * np.concatenate(
            [p["true_heading"] - _rotation_heading(p["rot"])[p["steps"].index] for p in train]))))
        yaw = _rotation_heading(d["rot"]) + offset
        track = PDR(k=k).run(d["acc"], t=d["t"], yaw=yaw, start=d["wp"][0])  # held at the start before t0
        errors.append(np.linalg.norm(track.position_at(d["wp_t"]) - d["wp"], axis=1))
    e = np.concatenate(errors)
    # measured on all 215 waypoints: mean 4.61 m, median 3.65 m, P90 10.00 m
    assert len(e) == 215 and np.mean(e) < 5.5 and np.median(e) < 4.5


def test_tilt_compensated_compass_agrees_with_the_phone_orientation(traces):
    diffs = []
    for d in traces:
        mag = magnetic_heading(d["mag"], d["acc"], d["t"])
        diffs.append(wrap_angle(mag - _rotation_heading(d["rot"]))[d["steps"].index])
    diff = np.concatenate(diffs)
    spread = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * diff)))))  # circular std, radians
    # measured: 7.7 degrees, mean offset -0.2 degrees; the rotation vector's w is completed as
    # sqrt(1 - |v|^2) (w >= 0, Android's convention), so this also checks that reconstruction
    assert np.rad2deg(spread) < 10.0


def test_geojson_floor_plan_is_consistent_with_the_walked_paths(traces):
    fmap = FloorMap.from_dict(_load("site2", "F4", "waypoints").meta["floor_plan"])
    segs = np.concatenate([np.stack([d["wp"][:-1], d["wp"][1:]], 1) for d in traces])
    assert fmap.crosses(segs[:, 0], segs[:, 1]).mean() < 0.05  # measured: 7 of 188 waypoint legs touch a wall


@pytest.mark.skipif(not (DENSE / "geojson_map.json").is_file(), reason=f"no ILC 2020 site1/F1 at {DENSE}")
def test_wifi_tracking_and_pdr_fusion_on_a_densely_surveyed_floor():
    """WKNN on WiFi (trained on the other traces), then KalmanTracker and PDR + WiFi fusion with the
    floor plan, on every 12th trace of site1/F1 (10 traces spread over the survey sessions;
    consecutive trace ids come from one session in one area). Errors at the waypoints."""
    from indoorloc.methods import create_model
    from indoorloc.signals import FillMissing

    paths, (wp, scans) = _read("site1", "F1", wifi=True)
    paths = [d for d in paths if len(d["wp"]) >= 2 and len(d["scan_t"])]
    test = paths[::12]
    train = [d for i, d in enumerate(paths) if i % 12]
    heard = np.isfinite(np.concatenate([d["X"] for d in train])).any(axis=0)  # APs of the training traces
    inside = [np.isfinite(d["scan_pos"]).all(axis=1) for d in train]  # scans within the waypoint span
    X = np.concatenate([d["X"][m][:, heard] for d, m in zip(train, inside)])
    Y = np.concatenate([d["scan_pos"][m] for d, m in zip(train, inside)])  # waypoints interpolated in time
    model = create_model("wknn", k=5, preprocess=FillMissing(-100.0)).fit(X, Y)
    walked = sum(np.linalg.norm(np.diff(d["wp"], axis=0), axis=1).sum() for d in train)
    k = walked / sum(step_lengths(StepDetector().detect(d["acc"], d["t"]), "weinberg", 1.0).sum() for d in train)
    fmap = FloorMap.from_dict(wp.meta["floor_plan"])
    errs = {"wifi": [], "kf": [], "fusion": []}
    for d in test:
        Xt = d["X"][:, heard]
        usable = np.isfinite(Xt).sum(axis=1) >= 5
        fixes, scan_t = model.localize(Xt[usable]), d["scan_t"][usable]
        sel = d["wp_t"] >= scan_t[0]
        wt, truth = d["wp_t"][sel], d["wp"][sel]
        err = lambda t, pos: np.linalg.norm(_interp(wt, t, pos) - truth, axis=1)  # noqa: E731
        errs["wifi"].append(err(scan_t, fixes.pos))
        errs["kf"].append(err(scan_t, KalmanTracker(process_noise=0.5, min_meas_std=2.0).filter(fixes, scan_t).pos))
        steps = StepDetector().detect(d["acc"], d["t"])
        # heading: phone orientation, offset unknown (uniform prior on the heading bias)
        heading = _rotation_heading(d["rot"])[steps.index]
        pf = ParticleFilter(1000, step_length_std=0.1, heading_std=0.08, heading_drift_std=0.01, min_meas_std=3.0,
                            floor_map=fmap, recovery=(0.05, 0.5), random_state=0)
        t, est = PDRFusion(pf, heading_bias_std=None).run((steps.t, step_lengths(steps, "weinberg", k), heading),
                                                           fixes, scan_t)
        errs["fusion"].append(err(t, est.pos))
    mean = {name: float(np.concatenate(e).mean()) for name, e in errs.items()}
    # measured (67 waypoints): WKNN 7.30 m, KalmanTracker 5.89 m, PDR + WiFi + map fusion 5.90 m (means;
    # fusion 5.88 m if a BSSID listed twice in one scan keeps its last line, as the earlier test-only
    # parser did, instead of its most recently seen entry)
    assert sum(len(e) for e in errs["wifi"]) == 67
    assert mean["wifi"] < 9.0  # a sanity bound on the fingerprinting itself
    assert mean["kf"] < 0.9 * mean["wifi"] and mean["fusion"] < 0.9 * mean["wifi"]
