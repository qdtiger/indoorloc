"""L3 visible-light positioning and geomagnetic sequence matching: exact cases, brute force, planted truths."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import SampleTable, load_model
from indoorloc.core.estimator import clone
from indoorloc.methods import create_model
from indoorloc.methods.magnetic import MagneticDTWLocalizer, dtw_distance, subsequence_dtw
from indoorloc.methods.vlc import LambertianLocalizer
from indoorloc.signals import vlc

DEG = np.pi / 180.0
LEDS = np.array([[x, y, 3.0] for x in (1.0, 3.0, 5.0) for y in (1.0, 3.0)])      # 3 x 2 ceiling grid
OPTICS = dict(order=1.5, fov=80 * DEG, area=1e-4)


def _points(n, seed=0, z=0.8):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0.5, 5.5, n), rng.uniform(0.5, 3.5, n), np.full(n, z)])


# ----------------------------------------------------------------------------------------- VLC
@pytest.mark.parametrize("solver", ["ranges", "nls"])
def test_vlc_is_exact_without_noise_at_a_known_height(solver):
    pts = _points(60)
    P = vlc.received_power(pts, LEDS, tx_power=2.0, **OPTICS)
    model = LambertianLocalizer(LEDS, tx_power=2.0, receiver_height=0.8, solver=solver, **OPTICS)
    pred = model.localize(P)                                        # geometry only: no fit needed
    np.testing.assert_allclose(pred.pos, pts[:, :2], atol=1e-9)
    assert np.all(pred.spread < 1e-9)                               # exact data: zero residual scale
    P[:, :2] = np.nan                                               # two LEDs not seen: 4 left
    np.testing.assert_allclose(model.predict(P), pts[:, :2], atol=1e-9)
    P[:, 2] = 0.0                                                   # 3 left, still exact in 2-D
    np.testing.assert_allclose(model.predict(P), pts[:, :2], atol=1e-8)
    P[:, 3] = np.nan                                                # 2 LEDs: under-determined
    assert np.isnan(model.predict(P)).all()


def test_vlc_nls_is_exact_in_3d_and_with_tilted_leds_and_receiver():
    pts = _points(40, seed=1, z=0.8) + np.c_[np.zeros((40, 2)), np.linspace(-0.3, 0.4, 40)]
    P = vlc.received_power(pts, LEDS, tx_power=2.0, **OPTICS)
    np.testing.assert_allclose(LambertianLocalizer(LEDS, tx_power=2.0, **OPTICS).predict(P), pts, atol=1e-9)
    normals = np.array([[0.2, 0.0, -1.0], [0.0, -0.2, -1.0], [0.1, 0.1, -1.0],
                        [-0.2, 0.0, -1.0], [0.0, 0.25, -1.0], [-0.1, -0.1, -1.0]])
    rx = np.array([0.1, -0.05, 1.0])
    P = vlc.received_power(pts, LEDS, tx_power=2.0, led_normals=normals, receiver_normals=rx, **OPTICS)
    tilted = LambertianLocalizer(LEDS, normals=normals, receiver_normal=rx, tx_power=2.0, **OPTICS)
    np.testing.assert_allclose(tilted.predict(P), pts, atol=1e-9)
    flat = LambertianLocalizer(LEDS, tx_power=2.0, **OPTICS)        # wrong model: biased
    assert np.nanmax(np.linalg.norm(flat.predict(P) - pts, axis=1)) > 0.05


def test_vlc_calibration_learns_the_emitted_power_of_each_led():
    pts = _points(80, seed=2)
    tx = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 0.8])
    P = vlc.received_power(pts, LEDS, tx_power=tx, **OPTICS)
    model = LambertianLocalizer(LEDS, receiver_height=0.8, calibrate=True, **OPTICS).fit(P, pts[:, :2])
    np.testing.assert_allclose(model.tx_power_, tx, rtol=1e-12)
    assert model.calibrated_.all() and model.sigma_ < 1e-15
    np.testing.assert_allclose(model.predict(P), pts[:, :2], atol=1e-9)
    np.testing.assert_allclose(model.predict_power(pts[:5, :2]), P[:5], rtol=1e-12)
    P_seen = P.copy()
    P_seen[2:, 5] = np.nan                                          # LED 5 seen twice: keeps its nominal power
    partial = LambertianLocalizer(LEDS, receiver_height=0.8, calibrate=True, tx_power=0.8, **OPTICS).fit(P_seen, pts)
    assert not partial.calibrated_[5] and partial.tx_power_[5] == 0.8
    with pytest.raises(Exception, match="not fitted|fit"):
        LambertianLocalizer(LEDS, receiver_height=0.8, calibrate=True, **OPTICS).predict(P)


def test_vlc_spread_matches_the_error_under_gaussian_power_noise():
    pts = _points(400, seed=3)
    P = vlc.received_power(pts, LEDS, tx_power=2.0, **OPTICS)
    sigma = 2e-8                                                    # W
    noisy = P + np.random.default_rng(4).normal(0.0, sigma, P.shape)
    pred = LambertianLocalizer(LEDS, tx_power=2.0, receiver_height=0.8, sigma=sigma, **OPTICS).localize(noisy)
    rmse = np.sqrt(np.mean(np.sum((pred.pos - pts[:, :2]) ** 2, axis=1)))
    assert 0.8 < np.sqrt(np.mean(pred.spread ** 2)) / rmse < 1.25  # CRLB-shaped spread = realised RMSE
    ranges = LambertianLocalizer(LEDS, tx_power=2.0, receiver_height=0.8, solver="ranges", **OPTICS).predict(noisy)
    assert np.sqrt(np.mean(np.sum((ranges - pts[:, :2]) ** 2, axis=1))) > rmse   # ML on power beats RSS ranging


def test_vlc_collinear_leds_leave_a_mirror_ambiguity_that_spread_reports():
    line = np.array([[x, 2.0, 3.0] for x in (0.0, 2.0, 4.0, 6.0)])  # one row of corridor lights
    pts = np.array([[2.7, 2.6, 0.8], [3.9, 1.5, 0.8]])
    P = vlc.received_power(pts, line, **OPTICS)
    for solver in ("ranges", "nls"):                                 # a point and its mirror fit equally well
        pred = LambertianLocalizer(line, receiver_height=0.8, solver=solver, **OPTICS).localize(P)
        assert np.isnan(pred.pos).all() and np.isnan(pred.spread).all()
        res = LambertianLocalizer(line, receiver_height=0.8, solver=solver, **OPTICS).evaluate(P, pts[:, :2])
        assert res.n_failed == 2
    off = np.vstack([line, [[3.0, 5.0, 3.0]]])                         # one LED off the line resolves it
    P = vlc.received_power(pts, off, **OPTICS)
    np.testing.assert_allclose(LambertianLocalizer(off, receiver_height=0.8, **OPTICS).predict(P), pts[:, :2],
                               atol=1e-9)


def test_vlc_from_meta_of_a_simulated_table_and_the_estimator_contract(tmp_path):
    from indoorloc.datasets import load_dataset

    train, test = load_dataset("synthetic_office", modality="vlc", noise_std=0.0, size=(16.0, 10.0), n_test=100,
                               grid_spacing=3.0, samples_per_point=1)
    model = LambertianLocalizer.from_meta(train.meta)
    assert model.receiver_height == 1.2 and model.order == pytest.approx(1.0) and model.anchors.shape[1] == 3
    pred = model.localize(test)
    err = np.linalg.norm(pred.pos - test.pos, axis=1)
    exact = np.isfinite(pred.spread)
    assert exact.mean() > 0.9 and np.all(err[exact] < 1e-8)       # noise-free: exact wherever identifiable
    assert np.array_equal(pred.ids, test.ids)
    with pytest.raises(ValueError, match="vlc"):
        LambertianLocalizer.from_meta({"modality": "wifi_rssi"})
    with pytest.raises(ValueError, match="receiver_height"):
        LambertianLocalizer.from_meta({**train.meta, "floors": (0, 1)})
    # 3-D positions on two storeys: the height is estimated too; slabs hide the other storey's LEDs
    train3, test3 = load_dataset("synthetic_office", modality="vlc", noise_std=0.0, size=(16.0, 10.0), n_test=60,
                                 grid_spacing=3.0, samples_per_point=1, dim=3, n_floors=2)
    pred3 = LambertianLocalizer.from_meta(train3.meta).localize(test3)
    placed = np.isfinite(pred3.pos).all(axis=1)
    assert pred3.pos.shape == (60, 3) and placed.mean() > 0.9
    np.testing.assert_allclose(pred3.pos[placed], test3.pos[placed], atol=1e-8)
    assert set(np.unique(test3.floor[placed])) == {0, 1}
    # registry, parameters, clone, persistence
    m = create_model("vlc", anchors=LEDS, receiver_height=0.8, calibrate=True, **OPTICS)
    assert isinstance(m, LambertianLocalizer) and m.get_params()["solver"] == "nls"
    pts = _points(30, seed=5)
    P = vlc.received_power(pts, LEDS, tx_power=1.7, **OPTICS)
    m.fit(P, pts[:, :2])
    assert not hasattr(clone(m), "tx_power_")
    m.save(tmp_path / "vlc")
    back = load_model(tmp_path / "vlc")
    np.testing.assert_array_equal(back.predict(P), m.predict(P))
    with pytest.raises(ValueError, match="receiver_height"):
        LambertianLocalizer(LEDS, solver="ranges").predict(P)
    with pytest.raises(ValueError, match="not both"):
        LambertianLocalizer(LEDS, order=1.0, half_power_angle=60 * DEG).predict(P)
    with pytest.raises(ValueError, match="one received power per LED"):
        LambertianLocalizer(LEDS).predict(P[:, :4])
    np.testing.assert_allclose(LambertianLocalizer(LEDS, half_power_angle=60 * DEG, receiver_height=0.8).predict(P),
                               LambertianLocalizer(LEDS, order=1.0, receiver_height=0.8).predict(P))


# ----------------------------------------------------------------------------------------- DTW
def _brute_symmetric1(a, b):
    """Textbook DTW with unit-weight steps (1, 0), (0, 1), (1, 1)."""
    D = np.full((len(a) + 1, len(b) + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[-1, -1]


_P1_STEPS = (((1, 1, 1.0),), ((1, 1, 0.5), (1, 2, 0.5)), ((1, 1, 1.0), (2, 1, 1.0)))


def _brute_p1_paths(L, M, entries=None):
    """Every asymmetricP1 path as (end, start, cells). ``entries``: enter from the virtual row -1
    at these columns (open begin); None: begin at cell (0, 0) with weight 1 (Sakoe & Chiba)."""
    out = []

    def walk(i, j, cells, first):
        if i == L - 1:
            out.append((j, first, cells))
        for step in _P1_STEPS:
            new = [(i + di, j + dj, w) for di, dj, w in step]
            if all(0 <= r < L and 0 <= c < M for r, c, _ in new):
                walk(new[-1][0], new[-1][1], cells + new, new[0][1] if first is None else first)

    if entries is None:
        walk(0, 0, [(0, 0, 1.0)], 0)
    for k in entries or ():
        walk(-1, k, [], None)
    return out


def _cost_of(cells, q, r):
    return sum(w * np.linalg.norm(q[i] - r[j]) for i, j, w in cells)


def test_dtw_equals_brute_force_for_both_step_patterns():
    rng = np.random.default_rng(0)
    for trial in range(40):
        L, M, F = int(rng.integers(1, 5)), int(rng.integers(1, 8)), int(rng.integers(1, 3))
        q, r = rng.normal(size=(L, F)), rng.normal(size=(M, F))
        if trial % 3 == 0:
            q, r = np.round(q), np.round(r)                           # exact ties
        # symmetric1: the best segment ending at j, and the returned start achieves it
        cost, start = subsequence_dtw(q, r, return_start=True)
        brute = [min(_brute_symmetric1(q, r[s:j + 1]) for s in range(j + 1)) for j in range(M)]
        np.testing.assert_allclose(cost, brute, atol=1e-12)
        np.testing.assert_allclose([_brute_symmetric1(q, r[s:j + 1]) for j, s in enumerate(start)], cost, atol=1e-12)
        assert dtw_distance(q, r) == pytest.approx(_brute_symmetric1(q, r), abs=1e-12)
        # asymmetricP1 against an enumeration of all admissible paths (every one carries weight L)
        paths = _brute_p1_paths(L, M, range(-1, M))
        assert all(sum(w for *_, w in cells) == pytest.approx(L) for *_, cells in paths)
        cost, start = subsequence_dtw(q, r, step_pattern="asymmetricP1", return_start=True)
        brute = np.full(M, np.inf)
        for j, _, cells in paths:
            brute[j] = min(brute[j], _cost_of(cells, q, r))
        np.testing.assert_allclose(cost, brute, atol=1e-12)
        for j in np.flatnonzero(np.isfinite(brute)):
            best_from_start = min(_cost_of(c, q, r) for e, s, c in paths if e == j and s == start[j])
            assert best_from_start == pytest.approx(cost[j], abs=1e-12)
        anchored = [_cost_of(c, q, r) for e, _, c in _brute_p1_paths(L, M) if e == M - 1]
        assert dtw_distance(q, r, step_pattern="asymmetricP1") == pytest.approx(min(anchored, default=np.inf),
                                                                                 abs=1e-12)


def test_dtw_matches_the_reference_implementation_of_giorgino():
    """Values computed with dtw-python 1.7.5 (the Python port of Giorgino's R ``dtw`` package):
    ``dtw(q, r, step_pattern=..., dist_method="euclidean").distance`` and, for the open-begin/open-end
    subsequence, the last row of ``costMatrix``."""
    q = np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 1.0], [2.5, 0.0], [1.0, -1.0]])
    r = np.array([[0.5, 1.0], [0.0, 0.5], [1.5, 1.5], [2.0, 2.0], [2.5, 0.5], [2.0, -0.5], [1.0, -1.0], [0.0, 0.0]])
    assert dtw_distance(q, r) == pytest.approx(5.121320343559643, abs=1e-12)
    assert dtw_distance(r, q) == pytest.approx(5.121320343559643, abs=1e-12)
    assert dtw_distance(q, r, step_pattern="asymmetricP1") == pytest.approx(3.7248737341529163, abs=1e-12)
    assert np.isinf(dtw_distance(q[:2], r, step_pattern="asymmetricP1"))       # dtw-python: no warping path
    assert np.isinf(dtw_distance([5.0], [5.0, 7.0], step_pattern="asymmetricP1"))  # the path starts at (0, 0)
    # open begin: dtw-python cannot start a path at the first two reference samples (its virtual row
    # has no columns left of the reference); make them prohibitive and the rows agree
    far = np.vstack([[[1e4, 1e4], [1e4, 1e4]], r])
    package = [np.nan, np.nan, 14144.103929175453, 4.756616537982939, 3.5705425906983637, 3.2686595939953778, 2.0,
               2.0606601717798214, 2.608494600052545, 4.063958203834218]
    cost, start = subsequence_dtw(q[1:4], far, step_pattern="asymmetricP1", return_start=True)
    np.testing.assert_allclose(cost[2:], package[2:], rtol=1e-12)
    assert np.argmin(cost) == 6 and start[6] == 4                               # dtw-python: index2 from 4 to 6
    assert cost[1] > 1e4                                                        # here a path may start at sample 0


def test_subsequence_dtw_finds_planted_and_time_warped_segments():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(400, 3))
    for pattern in ("symmetric1", "asymmetricP1"):
        cost, start = subsequence_dtw(ref[150:190], ref, step_pattern=pattern, return_start=True)
        assert np.argmin(cost) == 189 and cost.min() == 0.0 and start[189] == 150
    slow = np.repeat(ref[150:175], 2, axis=0)                            # the same walk at half the speed
    cost, start = subsequence_dtw(slow, ref, step_pattern="asymmetricP1", return_start=True)
    assert np.argmin(cost) == 174 and cost.min() == 0.0 and start[174] == 150
    fast = ref[150:210:2]                                                # twice as fast
    cost = subsequence_dtw(fast, ref, step_pattern="asymmetricP1")
    assert np.argmin(cost) == 208
    assert np.isinf(dtw_distance(ref[:3], ref[:10], step_pattern="asymmetricP1"))  # slope > 2: no path
    batch = subsequence_dtw(np.stack([ref[10:20], ref[50:60]]), ref, step_pattern="asymmetricP1")
    np.testing.assert_array_equal(batch[1], subsequence_dtw(ref[50:60], ref, step_pattern="asymmetricP1"))
    assert subsequence_dtw(np.arange(3.0), np.arange(10.0)).shape == (10,)          # 1-D sequences
    with pytest.raises(ValueError, match="step_pattern"):
        subsequence_dtw(ref[:5], ref, step_pattern="itakura")
    with pytest.raises(ValueError, match="features"):
        subsequence_dtw(ref[:5, :2], ref)
    with pytest.raises(ValueError, match="NaN"):
        dtw_distance([1.0, np.nan], [1.0, 2.0])


# ------------------------------------------------------------------------------ magnetic DTW
def _field(p):
    """A smooth synthetic [B, B_h, B_v] map (uT) over the plane."""
    x, y = p[:, 0], p[:, 1]
    return np.column_stack([48 + 3 * np.sin(1.3 * x + 0.2 * y) + 2 * np.cos(0.7 * x - 1.1 * y),
                            22 + 2 * np.cos(0.9 * x - 0.4 * y) + np.sin(1.7 * y),
                            -41 + 2.5 * np.sin(0.5 * x + 1.4 * y)])


def _walk(waypoints, speed=1.3, rate=10.0):
    w = np.asarray(waypoints, dtype=np.float64)
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
    s = np.arange(0.0, seg.sum(), speed / rate)
    cum = np.r_[0.0, np.cumsum(seg)]
    return np.column_stack([np.interp(s, cum, w[:, 0]), np.interp(s, cum, w[:, 1])])


def _reference():
    walks = [_walk([(0, 0), (20, 0), (20, 12)]), _walk([(0, 6), (14, 6), (14, 12)]), _walk([(3, 12), (3, 0)])]
    pos = np.concatenate(walks)
    traj = np.repeat(np.arange(len(walks)), [len(w) for w in walks])
    floor = np.where(traj == 2, 1, 0)
    return SampleTable(_field(pos), pos, floor=floor, groups={"trajectory": traj},
                       meta={"modality": "magnetic", "feature_names": ("B", "B_h", "B_v")})


@pytest.mark.parametrize("pattern", ["asymmetricP1", "symmetric1"])
def test_magnetic_dtw_recovers_a_replayed_walk_exactly(pattern):
    ref = _reference()
    model = MagneticDTWLocalizer(window=20, step_pattern=pattern).fit(ref, trajectory=ref.groups["trajectory"])
    rows = np.flatnonzero(ref.groups["trajectory"] == 0)[40:120]
    pred = model.localize(ref.X[rows])
    assert np.isnan(pred.pos[:19]).all()                              # fewer than `window` samples yet
    np.testing.assert_array_equal(pred.pos[19:], ref.pos[rows[19:]])  # the matched end sample itself
    np.testing.assert_array_equal(pred.spread[19:], 0.0)              # unique exact match
    assert (pred.floor == 0).all()
    res = model.evaluate(ref.X[rows], ref.pos[rows], floor=ref.floor[rows])
    assert res.n_failed == 19 and res.max_error == 0.0 and res.floor_accuracy == 100.0   # percent
    np.testing.assert_array_equal(model.localize_windows(np.stack([ref.X[rows[k - 19:k + 1]] for k in (30, 60)])).pos,
                                  ref.pos[rows[[30, 60]]])


def test_magnetic_dtw_handles_other_speeds_directions_and_short_windows():
    ref = _reference()
    t = ref.groups["trajectory"]
    fast = _walk([(2, 0), (20, 0), (20, 10)], speed=2.0)                # 1.5x the reference speed, same path
    model = MagneticDTWLocalizer(window=25).fit(ref, trajectory=t)
    err = np.linalg.norm(model.predict(_field(fast)) - fast, axis=1)[24:]
    assert np.median(err) < 0.1 and np.percentile(err, 95) < 0.3        # reference samples are 0.13 m apart
    back = _walk([(20, 12), (20, 0), (8, 0)])                            # walked the other way
    err = np.linalg.norm(model.predict(_field(back)) - back, axis=1)[24:]
    assert np.max(err) < 0.1
    one_way = MagneticDTWLocalizer(window=25, bidirectional=False).fit(ref, trajectory=t)
    assert np.median(np.linalg.norm(one_way.predict(_field(back)) - back, axis=1)[24:]) > 1.0
    short = MagneticDTWLocalizer(window=25, min_window=10).fit(ref, trajectory=t)
    pred = short.localize(_field(back))
    assert np.isnan(pred.pos[:9]).all() and np.isfinite(pred.pos[9:]).all()


def test_magnetic_dtw_spread_reports_repeated_patterns():
    x = np.column_stack([np.arange(320) * 0.125, np.zeros(320)])       # a 40 m corridor, 0.125 m steps
    period = 20.0                                                        # the field repeats every 20 m (160 steps)
    feats = np.column_stack([45 + 3 * np.sin(2 * np.pi * x[:, 0] / period + np.sin(2 * np.pi * x[:, 0] / 5)),
                             20 + np.cos(2 * np.pi * x[:, 0] / period), np.full(len(x), -40.0)])
    model = MagneticDTWLocalizer(window=30, bidirectional=False).fit(feats, x)
    pred = model.localize_windows(feats[None, 171:201])                  # ends at x = 25 m; its copy at 5 m
    assert pred.pos[0, 0] in (5.0, 25.0)
    assert pred.spread[0] == pytest.approx(period / np.sqrt(2), rel=1e-9)   # RMS distance of {0, 20 m}
    offset = model.localize_windows(feats[None, 100:130] + 0.01)         # a biased device: both copies still tie
    assert offset.pos[0, 0] in (16.125, 36.125) and offset.spread[0] == pytest.approx(period / np.sqrt(2), rel=1e-6)


def test_magnetic_dtw_walks_are_kept_apart():
    ref = _reference()
    model = MagneticDTWLocalizer(window=15).fit(ref, trajectory=ref.groups["trajectory"])
    q1, q2 = _walk([(1, 0), (12, 0)]), _walk([(14, 7), (14, 11)])
    X = np.concatenate([_field(q1), _field(q2)])
    walks = np.repeat([5, 9], [len(q1), len(q2)])
    table = SampleTable(X, np.concatenate([q1, q2]), groups={"trajectory": walks})
    pred = model.localize_walks(table)
    np.testing.assert_array_equal(pred.pos[:len(q1)], model.localize(X[:len(q1)]).pos)
    np.testing.assert_array_equal(pred.pos[len(q1):], model.localize(X[len(q1):]).pos)   # restarts per walk
    assert np.isnan(pred.pos[len(q1):len(q1) + 14]).all() and np.array_equal(pred.ids, table.ids)
    assert pred.floor is not None
    with pytest.raises(ValueError, match="trajectory"):
        model.localize_walks(X)
    with pytest.raises(ValueError, match="contiguous"):
        MagneticDTWLocalizer().fit(ref.X, ref.pos, trajectory=np.tile([0, 1], len(ref) // 2 + 1)[:len(ref)])


def test_magnetic_dtw_estimator_contract(tmp_path):
    ref = _reference()
    model = create_model("magnetic_dtw", window=12)
    assert isinstance(model, MagneticDTWLocalizer) and model.get_params()["step_pattern"] == "asymmetricP1"
    model.fit(ref, trajectory=ref.groups["trajectory"])
    q = _field(_walk([(3, 1), (3, 9)]))
    model.save(tmp_path / "dtw")
    back = load_model(tmp_path / "dtw")
    np.testing.assert_array_equal(back.localize(q).pos, model.localize(q).pos)
    np.testing.assert_array_equal(MagneticDTWLocalizer(window=12).fit(ref, trajectory=ref.groups["trajectory"])
                                  .localize(q).pos, model.localize(q).pos)                # deterministic
    for bad in ({"window": 0}, {"window": 2.5}, {"min_window": 20}, {"ambiguity": -1.0},
                {"step_pattern": "symmetric2"}):
        with pytest.raises(ValueError):
            MagneticDTWLocalizer(**{"window": 12, **bad}).fit(ref, trajectory=ref.groups["trajectory"])
    with pytest.raises(ValueError, match="NaN"):
        model.localize(np.full((20, 3), np.nan))


def test_magnetic_dtw_on_simulated_walks_beats_unconstrained_warping():
    """Simulated (SyntheticOffice, synthetic dipole field): reference walks with fresh 0.5 uT noise."""
    from indoorloc.datasets.simulated import SyntheticOffice
    from indoorloc.datasets.simulated import magnetic as smag

    ds = SyntheticOffice(seed=3, modality="magnetic", size=(24.0, 12.0), n_trajectories=4, trajectory_duration=30.0)
    ref = ds.load("trajectory")
    rows = np.flatnonzero(ref.groups["trajectory"] == 1)
    xyz = np.column_stack([ref.pos[rows], np.full(len(rows), 1.2)])
    field = ds.world["earth"] + smag.dipole_field(xyz, ds.world["dipoles"], ds.world["moments"])
    query = smag.features(smag.device_readings(field, 0.0, noise_std=0.5, random_state=7))
    median = {}
    for pattern in ("asymmetricP1", "symmetric1"):
        model = MagneticDTWLocalizer(window=30, step_pattern=pattern).fit(ref, trajectory=ref.groups["trajectory"])
        median[pattern] = np.nanmedian(np.linalg.norm(model.predict(query) - ref.pos[rows], axis=1))
    assert median["asymmetricP1"] < 0.6 and median["symmetric1"] > 1.5 * median["asymmetricP1"]
