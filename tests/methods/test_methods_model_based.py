"""Known-result tests of the L3 model-based methods: geometric, pathloss, aoa, interpolation.

Closed-form cases (exact recovery without noise, hand-computed examples, textbook GDOP) and
statistical results from the original papers (Gauss-Newton and Chan's estimator reach the
Cramer-Rao scale sigma * GDOP; MUSIC finds planted angles; spatial smoothing resolves
coherent paths; the path-loss exponent is recovered within its standard error).
"""
from __future__ import annotations

import csv
import sys

import numpy as np
import pytest

from conftest import DATA_ROOT, HEAVY
from indoorloc.core import NotFittedError, clone, load_model
from indoorloc.methods import create_model
from indoorloc.methods.aoa import AoALocalizer, music, music_spectrum, spatial_covariance, triangulate, ula_steering
from indoorloc.methods.geometric import (TDOALocalizer, TrilaterationLocalizer, WeightedCentroidLocalizer, chan_tdoa,
                                         gdop, linear_trilateration)
from indoorloc.methods.interpolation import RadioMapInterpolator, densify, gaussian_rbf, grid_points, idw
from indoorloc.methods.pathloss import PathLossLocalizer, log_distance_rssi

ROOM = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 15.0], [20.0, 15.0], [10.0, -2.0], [10.0, 17.0], [-2.0, 7.0],
                 [22.0, 7.0]])


def _ranges(pos, anchors):
    return np.sqrt(np.sum(np.square(np.asarray(pos)[:, None, :] - anchors[None]), axis=2))


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum(np.square(a - b), axis=1))))


# --------------------------------------------------------------------------------------------
# Trilateration
# --------------------------------------------------------------------------------------------

def test_linear_trilateration_hand_example():
    """Anchors (0,0), (10,0), (0,10) and the point (3,4): ranges 5, sqrt(65), sqrt(45)."""
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    r = np.array([[5.0, np.sqrt(65.0), np.sqrt(45.0)]])
    assert np.allclose(linear_trilateration(r, anchors), [[3.0, 4.0]], atol=1e-12)
    assert np.allclose(TrilaterationLocalizer(anchors).predict(r), [[3.0, 4.0]], atol=1e-12)


@pytest.mark.parametrize("solver", ["linear", "gauss_newton", "irls"])
@pytest.mark.parametrize("dim", [2, 3])
def test_trilateration_is_exact_without_noise(solver, dim):
    rng = np.random.default_rng(dim)
    anchors = rng.uniform(0, 10, (7, dim))
    pos = rng.uniform(2, 8, (100, dim))
    r = _ranges(pos, anchors)
    r[::3, 1] = np.nan  # a missing range is dropped, the rest still determine the point
    model = TrilaterationLocalizer(anchors, solver=solver, loss="tukey")
    pred = model.localize(r)
    assert np.abs(pred.pos - pos).max() < 1e-9
    assert np.all(pred.spread < 1e-6)  # residual-based spread of an exact fit
    few = np.full((1, 7), np.nan)
    few[0, :dim] = r[0, :dim]  # D ranges cannot fix a point in D dimensions
    assert np.isnan(model.localize(few).pos).all() and np.isnan(model.localize(few).spread).all()


def test_gdop_of_anchors_evenly_spaced_on_a_circle():
    """N anchors evenly on a circle around the target: H^T H = N/2 I, so GDOP = 2 / sqrt(N)
    for ranges, and the same for TDoA because the unit vectors sum to zero."""
    for n in (3, 4, 6, 8):
        ang = 2 * np.pi * np.arange(n) / n + 0.3
        anchors = 7.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1) + [2.0, -1.0]
        assert gdop(anchors, [[2.0, -1.0]])[0] == pytest.approx(2 / np.sqrt(n), rel=1e-12)
        assert gdop(anchors, [[2.0, -1.0]], kind="tdoa")[0] == pytest.approx(2 / np.sqrt(n), rel=1e-12)


def test_gauss_newton_reaches_sigma_times_gdop():
    """At small noise the ML (Gauss-Newton) error is efficient: RMSE ~ sigma * GDOP (Torrieri 1984),
    and the reported spread with a known sigma is that bound."""
    rng = np.random.default_rng(1)
    target, sigma, n = np.array([6.0, 5.0]), 0.1, 4000
    r = _ranges(np.tile(target, (n, 1)), ROOM) + sigma * rng.standard_normal((n, len(ROOM)))
    bound = sigma * gdop(ROOM, target)[0]
    pred = TrilaterationLocalizer(ROOM, sigma=sigma).localize(r)
    assert _rmse(pred.pos, target) == pytest.approx(bound, rel=0.05)
    assert np.median(pred.spread) == pytest.approx(bound, rel=0.01)
    estimated = TrilaterationLocalizer(ROOM).localize(r).spread  # sigma from each sample's residuals
    assert np.sqrt(np.mean(estimated ** 2)) == pytest.approx(bound, rel=0.1)


def test_irls_rejects_an_nlos_range():
    """One anchor with a +5 m non-line-of-sight bias: least squares is pulled by metres, the
    Tukey M-estimate is as good as the oracle that drops the blocked anchor."""
    rng = np.random.default_rng(2)
    pos = rng.uniform([2, 2], [18, 13], (300, 2))
    r = _ranges(pos, ROOM) + 0.05 * rng.standard_normal((300, len(ROOM)))
    keep = np.arange(len(ROOM)) != 2
    oracle = _rmse(TrilaterationLocalizer(ROOM[keep]).predict(r[:, keep]), pos)
    r[:, 2] += 5.0
    plain = _rmse(TrilaterationLocalizer(ROOM).predict(r), pos)
    huber = _rmse(TrilaterationLocalizer(ROOM, solver="irls").predict(r), pos)
    tukey = _rmse(TrilaterationLocalizer(ROOM, solver="irls", loss="tukey").predict(r), pos)
    assert plain > 1.0 and huber < 0.1
    assert tukey < 1.1 * oracle


def test_calibration_learns_per_anchor_bias_and_noise():
    rng = np.random.default_rng(3)
    bias = np.array([0.3, -0.2, 0.5, 0.0, 0.1, 0.4, -0.1, 0.25])
    pos = rng.uniform([1, 1], [19, 14], (2000, 2))
    r = _ranges(pos, ROOM) + bias + 0.02 * rng.standard_normal((2000, len(ROOM)))
    model = TrilaterationLocalizer(ROOM, calibrate=True)
    with pytest.raises(NotFittedError):
        model.localize(r[:3])  # calibration needs labelled data first
    model.fit(r[:1000], pos[:1000])
    assert np.abs(model.bias_ - bias).max() < 0.005
    assert model.sigma_ == pytest.approx(0.02, rel=0.1)
    assert _rmse(model.predict(r[1000:]), pos[1000:]) < 0.03
    assert _rmse(TrilaterationLocalizer(ROOM).predict(r[1000:]), pos[1000:]) > 0.1
    plain = TrilaterationLocalizer(ROOM).fit(r, pos)
    assert not hasattr(plain, "bias_")  # without calibrate, fit learns nothing but the input size


# --------------------------------------------------------------------------------------------
# TDoA
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
def test_chan_is_exact_without_noise(dim):
    rng = np.random.default_rng(10 + dim)
    anchors = rng.uniform(0, 10, (6, dim))
    pos = rng.uniform(1, 9, (200, dim))
    r = _ranges(pos, anchors)
    d = r[:, 1:] - r[:, :1]
    assert np.abs(chan_tdoa(d, anchors) - pos).max() < 1e-8
    d[::4, 2] = np.nan
    assert np.abs(TDOALocalizer(anchors).predict(d) - pos).max() < 1e-9
    assert np.abs(TDOALocalizer(anchors, refine=False).predict(d) - pos).max() < 1e-8


def test_chan_minimal_case_keeps_the_root_with_nonnegative_ranges():
    """3 anchors in 2-D: two differences, a quadratic in |x - a0|; roots giving negative ranges
    are impossible and are discarded."""
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    pos = np.random.default_rng(4).uniform(0.5, 9.5, (300, 2))
    r = _ranges(pos, anchors)
    est = chan_tdoa(r[:, 1:] - r[:, :1], anchors)
    assert np.abs(est - pos).max() < 1e-8
    assert np.isnan(chan_tdoa(np.array([[np.nan, 1.0]]), anchors)).all()  # one difference: no fix


def test_chan_reaches_the_crlb_at_small_noise():
    """Chan & Ho (1994): the two-step WLS attains the CRLB when the noise is small; the
    Taylor-series refinement stays there. Range noise sigma per anchor -> TDoA cov sigma^2 (I + 11^T)."""
    rng = np.random.default_rng(5)
    target, sigma, n = np.array([6.0, 5.0]), 0.05, 4000
    r = _ranges(np.tile(target, (n, 1)), ROOM) + sigma * rng.standard_normal((n, len(ROOM)))
    d = r[:, 1:] - r[:, :1]
    bound = sigma * gdop(ROOM, target, kind="tdoa")[0]
    chan = chan_tdoa(d, ROOM)
    refined = TDOALocalizer(ROOM, sigma=sigma).localize(d)
    assert _rmse(chan, target) == pytest.approx(bound, rel=0.06)
    assert _rmse(refined.pos, target) == pytest.approx(bound, rel=0.06)
    assert np.median(refined.spread) == pytest.approx(bound, rel=0.01)


def test_tdoa_refinement_reaches_the_crlb_where_chan_step2_degenerates():
    """A target level with the reference anchor (x = a_0x): Chan's step 2 works on squared
    coordinates, so the signed square root of a noisy x^2 near 0 loses efficiency (about twice
    the CRLB); the maximum-likelihood refinement of TDOALocalizer (the default) reaches the bound."""
    rng = np.random.default_rng(8)
    target, sigma, n = np.array([0.0, 7.0]), 0.05, 3000
    r = _ranges(np.tile(target, (n, 1)), ROOM) + sigma * rng.standard_normal((n, len(ROOM)))
    d = r[:, 1:] - r[:, :1]
    bound = sigma * gdop(ROOM, target, kind="tdoa")[0]
    assert _rmse(TDOALocalizer(ROOM).predict(d), target) == pytest.approx(bound, rel=0.06)
    assert _rmse(chan_tdoa(d, ROOM), target) > 1.5 * bound  # the documented limitation of step 2


def test_tdoa_calibration_learns_difference_bias():
    rng = np.random.default_rng(6)
    pos = rng.uniform([1, 1], [19, 14], (1500, 2))
    r = _ranges(pos, ROOM)
    offsets = np.linspace(-0.3, 0.4, len(ROOM) - 1)
    d = r[:, 1:] - r[:, :1] + offsets
    model = TDOALocalizer(ROOM, calibrate=True).fit(d[:500], pos[:500])
    assert np.abs(model.bias_ - offsets).max() < 1e-9
    assert np.abs(model.predict(d[500:]) - pos[500:]).max() < 1e-8


# --------------------------------------------------------------------------------------------
# Weighted centroid
# --------------------------------------------------------------------------------------------

def test_weighted_centroid_closed_forms():
    anchors = np.array([[0.0, 0.0], [10.0, 0.0]])
    w = 10 ** (-0.3)  # -43 dBm is 10^-0.3 of the power of -40 dBm
    rssi = np.array([[-40.0, -43.0]])
    assert WeightedCentroidLocalizer(anchors).predict(rssi)[0, 0] == pytest.approx(10 * w / (1 + w), rel=1e-12)
    assert WeightedCentroidLocalizer(anchors, degree=2).predict(rssi)[0, 0] == pytest.approx(
        10 * w ** 2 / (1 + w ** 2), rel=1e-12)
    dist = np.array([[1.0, 3.0]])  # 1/d weights: 1 and 1/3
    assert WeightedCentroidLocalizer(anchors, weights="distance").predict(dist)[0, 0] == pytest.approx(2.5)
    assert WeightedCentroidLocalizer(anchors, weights="uniform").predict(rssi).tolist() == [[5.0, 0.0]]
    assert WeightedCentroidLocalizer(anchors, k=1).predict(rssi).tolist() == [[0.0, 0.0]]
    assert WeightedCentroidLocalizer(anchors, weights="distance").predict([[0.0, 3.0]]).tolist() == [[0.0, 0.0]]
    pred = WeightedCentroidLocalizer(anchors, weights="uniform").localize([[np.nan, -80.0], [np.nan, np.nan]])
    assert pred.pos[0].tolist() == [10.0, 0.0] and pred.spread[0] == 0.0  # missing anchors are ignored
    assert np.isnan(pred.pos[1]).all() and np.isnan(pred.spread[1])  # nothing heard
    square = WeightedCentroidLocalizer(ROOM[:4]).localize(np.full((1, 4), -60.0))
    assert np.allclose(square.pos, [[10.0, 7.5]]) and square.spread[0] == pytest.approx(12.5)


def test_power_weights_equal_distance_weights_under_log_distance():
    """p ~ d^-n, so power weights of degree g/n equal 1/d^g weights on the true distances."""
    rng = np.random.default_rng(7)
    pos = rng.uniform([0, 0], [20, 15], (50, 2))
    d = _ranges(pos, ROOM)
    rssi = log_distance_rssi(pos, ROOM, -40.0, 2.5)
    by_power = WeightedCentroidLocalizer(ROOM, degree=3 / 2.5).predict(rssi)
    by_distance = WeightedCentroidLocalizer(ROOM, weights="distance", degree=3).predict(d)
    assert np.allclose(by_power, by_distance, rtol=0, atol=1e-10)


# --------------------------------------------------------------------------------------------
# Path loss
# --------------------------------------------------------------------------------------------

def _pathloss_world(seed, n=3000, noise=0.0):
    rng = np.random.default_rng(seed)
    p0, n_exp = rng.uniform(-45, -35, len(ROOM)), rng.uniform(2.0, 3.5, len(ROOM))
    pos = rng.uniform([0, 0], [20, 15], (n, 2))
    X = log_distance_rssi(pos, ROOM, p0, n_exp) + noise * rng.standard_normal((n, len(ROOM)))
    return X, pos, p0, n_exp, rng


def test_pathloss_fit_is_exact_without_noise():
    X, pos, p0, n_exp, rng = _pathloss_world(20)
    X[::5, 3] = np.nan  # unheard readings are ignored
    model = PathLossLocalizer(ROOM).fit(X, pos)
    assert np.abs(model.p0_ - p0).max() < 1e-9 and np.abs(model.exponent_ - n_exp).max() < 1e-10
    query = rng.uniform([1, 1], [19, 14], (100, 2))
    assert np.abs(model.predict(log_distance_rssi(query, ROOM, p0, n_exp)) - query).max() < 1e-8
    assert np.allclose(model.predict_rssi(query), log_distance_rssi(query, ROOM, p0, n_exp))
    joint = PathLossLocalizer().fit(X, pos)  # anchor positions unknown: estimated jointly
    assert np.abs(joint.anchors_ - ROOM).max() < 1e-6 and np.abs(joint.exponent_ - n_exp).max() < 1e-6


def test_pathloss_exponent_recovered_under_shadowing():
    """Least squares of RSSI on -10 log10(d): n is recovered within its standard error
    sigma / sqrt(sum (L - mean L)^2), and the residual std estimates the shadowing sigma."""
    sigma = 3.0
    X, pos, p0, n_exp, _ = _pathloss_world(21, n=4000, noise=sigma)
    model = PathLossLocalizer(ROOM).fit(X, pos)
    L = -10 * np.log10(np.maximum(_ranges(pos, ROOM), 0.1))
    se = sigma / np.sqrt(np.sum(np.square(L - L.mean(axis=0)), axis=0))
    assert np.all(np.abs(model.exponent_ - n_exp) < 4 * se)
    assert np.allclose(model.sigma_, sigma, rtol=0.05)
    shared = PathLossLocalizer(ROOM, per_anchor=False, p0=-40.0).fit(log_distance_rssi(pos, ROOM, -40.0, 2.7), pos)
    assert shared.exponent_ == pytest.approx(np.full(len(ROOM), 2.7))
    joint = PathLossLocalizer().fit(X, pos)
    assert np.sqrt(np.sum(np.square(joint.anchors_ - ROOM), axis=1)).max() < 0.6


def test_pathloss_ml_error_matches_the_rss_crlb():
    """Known model, 2 dB shadowing: the ML error at a fixed point matches the RSS Cramer-Rao
    bound (Patwari et al. 2003), which is what ``spread`` reports."""
    rng = np.random.default_rng(22)
    p0, n_exp = np.full(len(ROOM), -40.0), np.full(len(ROOM), 2.5)
    target = np.array([7.0, 4.0])
    X = log_distance_rssi(np.tile(target, (2000, 1)), ROOM, p0, n_exp) + 2.0 * rng.standard_normal((2000, len(ROOM)))
    model = PathLossLocalizer(ROOM, p0=p0, exponent=n_exp, sigma=2.0)  # fully specified: no fit needed
    pred = model.localize(X)
    bound = model.localize(log_distance_rssi(target, ROOM, p0, n_exp)).spread[0]
    b = 10 * 2.5 / np.log(10)
    diff = target - ROOM
    fisher = (b / 2.0) ** 2 * np.einsum("ai,aj->ij", diff, diff / (np.sum(diff ** 2, 1) ** 2)[:, None])
    assert bound == pytest.approx(np.sqrt(np.trace(np.linalg.inv(fisher))), rel=1e-9)
    assert _rmse(pred.pos, target) == pytest.approx(bound, rel=0.1)


def test_pathloss_bounds_and_fit_requirements():
    X, pos, p0, n_exp, _ = _pathloss_world(23, n=500, noise=6.0)
    with pytest.raises(NotFittedError):
        PathLossLocalizer(ROOM).localize(X[:2])  # p0/exponent unknown
    boxed = PathLossLocalizer(ROOM, bounds="fit").fit(X, pos)
    est = boxed.predict(X)
    assert np.all(est >= boxed.bounds_[0]) and np.all(est <= boxed.bounds_[1])
    assert _rmse(est, pos) < _rmse(PathLossLocalizer(ROOM).fit(X, pos).predict(X), pos)
    few = np.full((1, len(ROOM)), np.nan)
    few[0, :2] = X[0, :2]
    assert np.isnan(boxed.predict(few)).all()  # fewer than D + 1 anchors heard


def test_fully_specified_pathloss_model_stays_usable_after_a_small_fit():
    """With anchors, P0 and n given nothing per anchor is learned, so fitting on fewer than
    ``min_readings`` scans must not switch anchors off: it only estimates sigma (pooled)."""
    X, pos, p0, n_exp, _ = _pathloss_world(24, n=200, noise=3.0)
    model = PathLossLocalizer(ROOM, p0=p0, exponent=n_exp).fit(X[:3], pos[:3])
    assert model.used_.all() and np.allclose(model.sigma_, model.sigma_[0])  # one pooled sigma
    unfitted = PathLossLocalizer(ROOM, p0=p0, exponent=n_exp).predict(X[3:])
    assert np.isfinite(model.predict(X[3:])).all() and np.isfinite(unfitted).all()
    learned = PathLossLocalizer(ROOM).fit(X[:3], pos[:3])  # P0 and n from 3 scans: refused
    assert not learned.used_.any()


# --------------------------------------------------------------------------------------------
# AoA
# --------------------------------------------------------------------------------------------

def _noise(rng, shape, std):
    return std * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def test_music_finds_planted_angles():
    rng = np.random.default_rng(30)
    M, K, planted = 8, 200, np.deg2rad([-20.0, 35.0])
    Y = ula_steering(planted, M).T @ _noise(rng, (2, K), 1.0) + _noise(rng, (M, K), 0.1)  # 20 dB SNR
    est = np.sort(music(Y, 2))
    assert np.abs(np.rad2deg(est - planted)).max() < 0.05
    spectrum = music_spectrum(spatial_covariance(Y), np.deg2rad([-20.0, 0.0, 35.0]), 2)
    assert spectrum[1] < spectrum[0] / 100 and spectrum[1] < spectrum[2] / 100
    assert music(Y, 2).shape == (2,) and music(np.stack([Y, Y]), 2).shape == (2, 2)


def test_spatial_smoothing_resolves_coherent_paths():
    """Two fully coherent paths (a reflection of the same signal) make the signal covariance
    rank one and bias MUSIC; smoothing over 4 subarrays of 5 elements restores the rank
    (Shan, Wax, Kailath 1985), forward-backward averaging does it with fewer elements."""
    rng = np.random.default_rng(31)
    M, K, planted = 8, 400, np.deg2rad([-10.0, 20.0])
    s = _noise(rng, K, 1.0)
    Y = ula_steering(planted, M).T @ np.stack([s, 0.8 * np.exp(0.7j) * s]) + _noise(rng, (M, K), 0.05)
    err = lambda est: np.abs(np.rad2deg(np.sort(est) - planted)).max()  # noqa: E731
    assert err(music(Y, 2)) > 1.5
    assert err(music(Y, 2, subarray=5)) < 0.1
    assert err(music(Y, 2, subarray=6, forward_backward=True)) < 0.1


def test_triangulation_hand_example_and_orientation_convention():
    anchors = np.array([[0.0, 0.0], [10.0, 0.0]])
    bearings = np.deg2rad([[45.0, 135.0]])
    assert np.allclose(triangulate(bearings, anchors), [[5.0, 5.0]], atol=1e-12)
    orient = np.deg2rad([90.0, 90.0])  # both arrays face +y: local angles are -45 and +45 degrees
    local = np.deg2rad([[-45.0, 45.0]])
    for solver in ("linear", "gauss_newton"):
        assert np.allclose(AoALocalizer(anchors, orient, solver=solver).predict(local), [[5.0, 5.0]], atol=1e-12)
    assert np.isnan(AoALocalizer(anchors).predict([[0.7, np.nan]])).all()  # one bearing: no fix


def test_aoa_localizes_from_ula_snapshots():
    rng = np.random.default_rng(32)
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0], [10.0, 8.0]])
    orient = np.deg2rad([45.0, 135.0, -45.0, -135.0])  # each array faces the room centre
    pos = rng.uniform([1, 1], [9, 7], (40, 2))
    glob = np.arctan2(pos[:, None, 1] - anchors[None, :, 1], pos[:, None, 0] - anchors[None, :, 0])
    local = np.angle(np.exp(1j * (glob - orient)))
    exact = AoALocalizer(anchors, orient).localize(local)
    assert np.abs(exact.pos - pos).max() < 1e-9
    M, K = 8, 64
    Y = ula_steering(local, M)[..., None] * _noise(rng, (40, 4, 1, K), 1.0) + _noise(rng, (40, 4, M, K), 0.05)
    model = AoALocalizer(anchors, orient)
    assert np.abs(model.local_angles(Y.astype(np.complex64)) - local).max() < np.deg2rad(0.2)
    assert np.sqrt(np.sum(np.square(model.predict(Y.astype(np.complex64)) - pos), axis=1)).max() < 0.05


def test_aoa_calibration_learns_orientation_offsets():
    rng = np.random.default_rng(33)
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0], [10.0, 8.0]])
    pos = rng.uniform([1, 1], [9, 7], (300, 2))
    glob = np.arctan2(pos[:, None, 1] - anchors[None, :, 1], pos[:, None, 0] - anchors[None, :, 0])
    offset = np.deg2rad([2.0, -3.0, 1.0, 0.5])  # mounting errors of the arrays
    X = glob + offset + np.deg2rad(0.5) * rng.standard_normal(glob.shape)
    model = AoALocalizer(anchors, calibrate=True).fit(X[:200], pos[:200])
    assert np.abs(np.rad2deg(model.bias_ - offset)).max() < 0.15
    assert model.sigma_ == pytest.approx(np.deg2rad(0.5), rel=0.15)
    assert _rmse(model.predict(X[200:]), pos[200:]) < 0.5 * _rmse(AoALocalizer(anchors).predict(X[200:]), pos[200:])


def test_aoa_calibration_is_circular_for_an_array_facing_backwards():
    """Orientation pi: the offsets straddle the +-pi cut, where a linear median of wrapped
    residuals is biased by degrees; the circular calibration matches the true orientations."""
    rng = np.random.default_rng(34)
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0], [10.0, 8.0]])
    pos = rng.uniform([1, 1], [9, 7], (300, 2))
    glob = np.arctan2(pos[:, None, 1] - anchors[None, :, 1], pos[:, None, 0] - anchors[None, :, 0])
    orient = np.array([0.8, np.pi, -0.7, 0.01 - np.pi])
    local = np.angle(np.exp(1j * (glob - orient))) + np.deg2rad(1.0) * rng.standard_normal(glob.shape)
    model = AoALocalizer(anchors, calibrate=True).fit(local[:200], pos[:200])  # orientations unknown
    offset = np.angle(np.exp(1j * (model.bias_ + orient)))  # bias_ should be -orient (mod 2 pi)
    assert np.abs(np.rad2deg(offset)).max() < 0.2
    assert model.sigma_ == pytest.approx(np.deg2rad(1.0), rel=0.15)
    oracle = _rmse(AoALocalizer(anchors, orient).predict(local[200:]), pos[200:])
    assert _rmse(model.predict(local[200:]), pos[200:]) < 1.05 * oracle


# --------------------------------------------------------------------------------------------
# Radio-map interpolation
# --------------------------------------------------------------------------------------------

def test_idw_shepard_closed_form():
    """Values 0 and 1 at x = 0 and 1, query 0.25, power 2: weights 16 and 16/9 -> 0.1."""
    assert idw([[0.0], [1.0]], [0.0, 1.0], [[0.25]]).item() == pytest.approx(0.1, rel=1e-14)
    assert idw([[0.0], [1.0], [3.0]], [0.0, 1.0, 9.0], [[0.25]], k=2).item() == pytest.approx(0.1, rel=1e-14)
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # a duplicated point is averaged
    vals = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0], [3.0, np.nan]])
    out = idw(pts, vals, [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    assert out[0, 0] == 2.0 and out[1].tolist() == [2.0, 5.0]
    assert out[0, 1] == pytest.approx(6.0)  # not observed at (0, 0): the equidistant observers average
    assert out[2, 1] == pytest.approx(6.0)
    assert np.isnan(idw([[0.0]], [[np.nan]], [[1.0]])).all()


def test_gaussian_rbf_interpolates_a_smooth_map():
    g = grid_points([[0, 0], [14, 14]], 1.0)
    assert g.shape == (225, 2)
    f = lambda x: np.sin(x[:, 0] / 3) + np.cos(x[:, 1] / 4)  # noqa: E731
    q = np.random.default_rng(40).uniform(2, 12, (200, 2))
    exact = RadioMapInterpolator("rbf", smoothing=0.0).fit(g, f(g))
    assert np.abs(exact.predict(g) - f(g)).max() < 1e-8
    assert np.abs(exact.predict(q) - f(q)).max() < 0.03
    auto = RadioMapInterpolator("rbf").fit(g, f(g))  # leave-one-out picks (almost) no smoothing
    assert auto.smoothing_ <= 1e-4 and np.abs(auto.predict(q) - f(q)).max() < 0.03
    assert np.abs(gaussian_rbf(g, f(g), q, length_scale=1.5, smoothing=0.0) - f(q)).max() < 0.005
    assert np.abs(idw(g, f(g), q) - f(q)).max() > 0.03  # the RBF beats IDW on smooth maps
    far = gaussian_rbf(g, f(g), [[1e3, 1e3]])
    assert far[0] == pytest.approx(f(g).mean())  # returns to the column mean away from the data


def test_rbf_smoothing_by_leave_one_out_tames_noisy_neighbours():
    """Noisy scans a few centimetres apart: exact interpolation must pass through both values
    and oscillates far outside the data range; the leave-one-out ridge term stays in range."""
    rng = np.random.default_rng(42)
    pos = np.repeat(rng.uniform(0, 10, (40, 2)), 2, axis=0) + rng.normal(0, 0.02, (80, 2))
    rssi = -60 - 2 * pos[:, :1] + 6 * rng.standard_normal((80, 1))  # 6 dB shadowing
    q = grid_points([[0, 0], [10, 10]], 0.5)
    exact = gaussian_rbf(pos, rssi, q, smoothing=0.0)
    auto = RadioMapInterpolator("rbf").fit(pos, rssi)
    assert np.ptp(exact) > 10 * np.ptp(rssi)
    assert auto.smoothing_ >= 0.1 and np.ptp(auto.predict(q)) < np.ptp(rssi)
    truth = -60 - 2 * q[:, :1]
    inside = np.all((q > 1) & (q < 9), axis=1)
    assert np.sqrt(np.mean((auto.predict(q)[inside] - truth[inside]) ** 2)) < 6.0  # below the shadowing std


def _two_ap_map(p):
    return np.stack([-40 - 20 * np.log10(1 + np.hypot(*(p - c).T)) for c in ([2, 2], [8, 5])], axis=1)


def test_radio_map_interpolator_contract_and_densify(tmp_path):
    rng = np.random.default_rng(41)
    radio_map = _two_ap_map
    pos = rng.uniform(0, 10, (80, 2))
    X = radio_map(pos)
    X[::4, 1] = np.nan
    query = rng.uniform(1, 9, (50, 2))
    for method in ("idw", "rbf"):
        model = RadioMapInterpolator(method).fit(pos, X)
        assert np.allclose(model.predict(pos)[~np.isnan(X)], X[~np.isnan(X)])
        assert 0.8 < model.score(query, radio_map(query)) <= 1.0  # R^2 on unseen positions
        model.save(tmp_path / method)
        assert np.array_equal(load_model(tmp_path / method).predict(pos[:5]), model.predict(pos[:5]))
    Xg, pg = densify(X, pos, step=1.0, max_distance=1.0)
    assert Xg.shape == (len(pg), 2) and np.all(np.min(np.hypot(*(pg[:, None] - pos[None]).T), axis=0) <= 1.0 + 1e-9)
    sklearn = pytest.importorskip("sklearn.model_selection")
    search = sklearn.GridSearchCV(RadioMapInterpolator("rbf"), {"length_scale": [0.5, 1.0, 2.0]}, cv=3)
    search.fit(pos, np.nan_to_num(X, nan=-100.0))
    assert search.best_params_["length_scale"] in (0.5, 1.0, 2.0)


# --------------------------------------------------------------------------------------------
# Contract: registry, sklearn API, persistence, validation, light imports
# --------------------------------------------------------------------------------------------

def _cases():
    anchors2 = ROOM[:5]
    r = _ranges(np.array([[5.0, 5.0], [12.0, 3.0]]), anchors2)
    return [
        ("trilateration", {"anchors": anchors2}, r),
        ("multilateration", {"anchors": anchors2, "solver": "irls"}, r),
        ("tdoa", {"anchors": anchors2}, r[:, 1:] - r[:, :1]),
        ("centroid", {"anchors": anchors2}, -40 - 25 * np.log10(r)),
        ("pathloss", {"anchors": anchors2, "p0": -40.0, "exponent": 2.5}, -40 - 25 * np.log10(r)),
        ("aoa", {"anchors": anchors2}, np.arctan2(5.0 - anchors2[:, 1], 5.0 - anchors2[:, 0])[None]),
    ]


@pytest.mark.parametrize("name,params,X", _cases(), ids=[c[0] for c in _cases()])
def test_model_based_contract(name, params, X, tmp_path):
    model = create_model(name, **params)
    unfitted = model.localize(X)  # geometry-only models need no fit
    assert unfitted.pos.dtype == np.float64 and unfitted.pos.shape == (len(X), 2)
    assert unfitted.floor is None and unfitted.spread.shape == (len(X),)
    twin = clone(model)
    assert np.array_equal(twin.get_params()["anchors"], params["anchors"]) and not hasattr(twin, "n_features_in_")
    assert repr(twin).startswith(type(model).__name__)
    y = unfitted.pos
    assert np.isfinite(y).all()
    model.fit(X, y)
    assert model.n_features_in_ == X.shape[1]
    assert np.allclose(model.predict(X), y)  # fitting on its own exact output changes nothing
    model.save(tmp_path / name)
    again = load_model(tmp_path / name)
    assert np.array_equal(again.predict(X), model.predict(X))
    with pytest.raises(ValueError):
        create_model(name, **params).localize(np.zeros((1, X.shape[1] + 1)))
    with pytest.raises(ValueError, match="anchor|not fitted"):  # pathloss can learn anchors, but only in fit
        create_model(name).localize(X)


def test_spreads_equal_the_independent_crlbs():
    """With a known noise std, every model-based spread is the Cramer-Rao bound at the estimate.
    Cross-checked against the separate L4 implementation (evaluation.bounds), including a target
    outside the anchor hull; ``gdop(kind="tdoa")`` is the PDOP of the pseudorange model."""
    from indoorloc.evaluation import bounds

    t = np.array([[6.0, 5.0], [15.0, 3.0], [25.0, 20.0]])
    r = _ranges(t, ROOM)
    ang = np.arctan2(t[:, None, 1] - ROOM[None, :, 1], t[:, None, 0] - ROOM[None, :, 0])
    rssi = log_distance_rssi(t, ROOM, -40.0, 2.5)
    pairs = [
        (TrilaterationLocalizer(ROOM, sigma=0.1).localize(r).spread, bounds.toa_crlb(ROOM, t, 0.1)),
        (TDOALocalizer(ROOM, sigma=0.1).localize(r[:, 1:] - r[:, :1]).spread, bounds.tdoa_crlb(ROOM, t, 0.1)),
        (AoALocalizer(ROOM, sigma=0.02).localize(ang).spread, bounds.aoa_crlb(ROOM, t, 0.02)),
        (PathLossLocalizer(ROOM, p0=-40.0, exponent=2.5, sigma=4.0).localize(rssi).spread,
         bounds.rss_crlb(ROOM, t, 4.0, 2.5)),
        (gdop(ROOM, t), bounds.gdop(ROOM, t)),
        (gdop(ROOM, t, kind="tdoa"), bounds.dop(ROOM, t, clock_bias=True)["pdop"]),
    ]
    for ours, theirs in pairs:
        assert np.allclose(ours, theirs, rtol=1e-9, atol=0)


def test_model_based_methods_take_sample_tables():
    """Data enters as a SampleTable (``ranges`` modality with ``meta["anchors"]``): fit, localize
    (ids kept) and evaluate go through the table; the helpers accept it too."""
    from indoorloc.core import SampleTable

    rng = np.random.default_rng(51)
    pos = rng.uniform([1, 1], [19, 14], (30, 2))
    table = SampleTable(_ranges(pos, ROOM) + 0.2, pos, ids=np.array([f"s{i}" for i in range(30)]),
                        meta={"modality": "ranges", "anchors": ROOM})
    model = create_model("trilateration", anchors=table.meta["anchors"], calibrate=True).fit(table)
    pred = model.localize(table)
    assert np.allclose(model.bias_, 0.2) and pred.ids.tolist() == table.ids.tolist()
    assert model.evaluate(table).mean_error < 1e-9
    rssi = SampleTable(log_distance_rssi(pos, ROOM, -40.0, 2.0), pos)
    assert np.allclose(WeightedCentroidLocalizer(ROOM).centroid_weights(rssi).sum(axis=1), 1.0)
    ang = SampleTable(np.arctan2(pos[:, None, 1] - ROOM[None, :, 1], pos[:, None, 0] - ROOM[None, :, 0]), pos)
    assert np.allclose(AoALocalizer(ROOM).bearings(ang), ang.X)


@pytest.mark.filterwarnings("ignore:Estimator .* does not inherit from:UserWarning")
def test_radio_map_interpolator_passes_sklearn_checks():
    """sklearn's estimator checks, except the one every indoorloc predictor fails by design:
    ``core.NotFittedError`` subclasses ValueError/AttributeError like sklearn's, but is not
    sklearn's class (indoorloc does not import sklearn)."""
    pytest.importorskip("sklearn")
    from sklearn.utils.estimator_checks import check_estimator

    for est in (RadioMapInterpolator(), RadioMapInterpolator("rbf")):
        results = check_estimator(est, on_fail=None)
        failed = {r["check_name"] for r in results if r["status"] == "failed"}
        assert failed <= {"check_estimators_unfitted"}, failed


def test_sklearn_clone_and_cross_validation():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(50)
    pos = rng.uniform([1, 1], [19, 14], (60, 2))
    r = _ranges(pos, ROOM) + 0.3 + 0.01 * rng.standard_normal((60, len(ROOM)))
    model = TrilaterationLocalizer(ROOM, calibrate=True)
    assert clone(model).get_params()["calibrate"] is True
    scores = cross_val_score(model, r, pos, cv=3)
    assert np.all(-scores < 0.05) and sklearn.__version__


def test_model_based_modules_import_numpy_only():
    import subprocess

    code = ("import sys; import indoorloc.methods.geometric, indoorloc.methods.pathloss, "
            "indoorloc.methods.aoa, indoorloc.methods.interpolation; "
            f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "[]"


# --------------------------------------------------------------------------------------------
# Real data (skipped unless the raw BBIL files are present)
# --------------------------------------------------------------------------------------------

def _bbil_root():
    base = DATA_ROOT / "ble_indoor"
    hits = sorted(base.rglob("experiment1/edges.csv")) if base.is_dir() else []
    return hits[0].parent if hits else None


@pytest.mark.skipif(_bbil_root() is None, reason="BBIL raw files not found under $INDOORLOC_DATA/ble_indoor")
def test_bbil_pathloss_with_known_receivers():
    """BBIL experiment 1 (Kennedy, Spachos, Taylor; github.com/co60ca/BBIL): 9 BLE receivers at
    known positions. Measured with this library (see the PR report): mean error 3.88 m for the
    path-loss MAP estimate in the training box, 3.52 m for WKNN."""
    root = _bbil_root()

    def read(path):
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def load(split):
        rows = [r for f in sorted((root / split).glob("*_data_wide.csv")) for r in read(f)]
        names = sorted({c for r in rows for c in r if c.startswith("edge_")}, key=lambda c: int(c[5:]))
        X = np.array([[float(r[c]) if r.get(c) not in (None, "") else np.nan for c in names] for r in rows])
        return X, np.array([[float(r["realx"]), float(r["realy"])] for r in rows]), names

    Xtr, ytr, names = load("train")
    Xte, yte, _ = load("test")
    edges = {int(r["edgenodeid"]): (float(r["edge_x"]), float(r["edge_y"])) for r in read(root / "edges.csv")}
    anchors = np.array([edges[int(c[5:])] for c in names])
    model = PathLossLocalizer(anchors, bounds="fit").fit(Xtr, ytr)
    err = np.sqrt(np.sum(np.square(model.predict(Xte) - yte), axis=1))
    assert np.all((model.exponent_ > 1.0) & (model.exponent_ < 4.0))
    assert np.mean(err) < 4.5


def test_refinement_never_returns_far_field_positions_under_large_noise():
    """Gauss-Newton from a poor closed-form start used to end kilometres away (TDoA at 1-3 m noise,
    AoA at 30 deg); the second start and the far-field test turn such rows into NaN (n_failed)."""
    from indoorloc.evaluation import evaluate
    from indoorloc.methods.aoa import AoALocalizer
    from indoorloc.methods.geometric import FAR_FIELD, TDOALocalizer

    rng = np.random.default_rng(0)
    anchors = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0], [20.0, 10.0]])
    truth = rng.uniform([0, 0], [40, 20], (400, 2))
    ranges = np.linalg.norm(truth[:, None] - anchors[None], axis=2) + rng.normal(0, 3.0, (400, 5))
    tdoa = ranges[:, 1:] - ranges[:, :1]
    angles = np.arctan2(truth[:, None, 1] - anchors[None, :4, 1], truth[:, None, 0] - anchors[None, :4, 0])
    bearings = angles + rng.normal(0, np.deg2rad(30), angles.shape)
    radius = FAR_FIELD * np.sqrt(np.mean(np.sum(np.square(anchors - anchors.mean(0)), axis=1)))
    for model, X, used in ((TDOALocalizer(anchors), tdoa, anchors), (AoALocalizer(anchors[:4]), bearings, anchors[:4])):
        pred = model.fit(X, truth).localize(X)
        placed = np.all(np.isfinite(pred.pos), axis=1)
        assert placed.mean() > 0.9
        assert np.all(np.linalg.norm(pred.pos[placed] - used.mean(axis=0), axis=1) <= FAR_FIELD * np.sqrt(
            np.mean(np.sum(np.square(used - used.mean(0)), axis=1))) + 1e-9)
        result = evaluate(truth, pred.pos)
        assert result.n_failed == int((~placed).sum()) and result.max_error < radius


# --------------------------------------------------------------------------------------------
# Usability: fit without labels, evaluate without truth, known tag height, from_meta, AoA frames
# --------------------------------------------------------------------------------------------

def test_evaluate_and_score_need_the_true_positions():
    """``evaluate(X)`` / ``score(X)`` with an array and no ``y`` name the fix; ``score`` is NaN
    when a sample cannot be placed while ``evaluate`` counts it in ``n_failed``."""
    pos = np.array([[5.0, 5.0], [12.0, 3.0], [8.0, 9.0]])
    r = _ranges(pos, ROOM[:4])
    model = TrilaterationLocalizer(ROOM[:4])
    for call in (model.evaluate, model.score):
        with pytest.raises(ValueError, match=r"pass y \(positions\) or a SampleTable"):
            call(r)
    assert model.score(r, pos) == pytest.approx(0.0, abs=1e-9)
    r[1, :2] = np.nan  # two ranges left: that sample cannot be placed in 2-D
    assert np.isnan(model.score(r, pos))
    result = model.evaluate(r, pos)
    assert result.n_failed == 1 and result.mean_error < 1e-9


def _vlc_case():
    from indoorloc.methods.vlc import LambertianLocalizer

    leds = np.array([[1.0, 1.0, 3.0], [5.0, 1.0, 3.0], [1.0, 5.0, 3.0], [5.0, 5.0, 3.0]])
    model = LambertianLocalizer(leds, receiver_height=1.2)
    pos = np.array([[2.0, 2.5], [3.5, 4.0]])
    return model, model.predict_power(pos), pos


def test_geometry_only_models_fit_without_labels():
    """Nothing is learned from positions unless ``calibrate=True``: ``fit(X)`` then only records
    the input size and changes no prediction; with ``calibrate=True`` positions stay required
    and the error says why. PathLossLocalizer learns its noise scale from labels: y required."""
    anchors = ROOM[:5]
    t = np.array([[5.0, 5.0], [12.0, 3.0]])
    r = _ranges(t, anchors)
    angles = np.arctan2(t[:, None, 1] - anchors[None, :, 1], t[:, None, 0] - anchors[None, :, 0])
    vlc, power, vlc_pos = _vlc_case()
    cases = [(TrilaterationLocalizer(anchors), r, t), (TDOALocalizer(anchors), r[:, 1:] - r[:, :1], t),
             (WeightedCentroidLocalizer(anchors, weights="distance"), r, None), (AoALocalizer(anchors), angles, t),
             (vlc, power, vlc_pos)]
    for model, X, truth in cases:
        before = model.localize(X).pos
        assert model.fit(X) is model and model.n_features_in_ == X.shape[1] and model.target_ndim_ == 2
        assert np.array_equal(model.predict(X), before)
        if truth is not None:
            assert np.allclose(before, truth, atol=1e-6)
        if "calibrate" in model.get_params():
            with pytest.raises(ValueError, match="requires y.*calibrate=True"):
                clone(model).set_params(calibrate=True).fit(X)
    # a refit without positions drops an earlier calibration
    model = TrilaterationLocalizer(anchors, calibrate=True).fit(r + 0.3, t)
    assert np.allclose(model.bias_, 0.3)
    model.set_params(calibrate=False).fit(r)
    assert not hasattr(model, "bias_") and np.allclose(model.predict(r), t)
    with pytest.raises(ValueError, match="requires y to be passed"):
        PathLossLocalizer(anchors, p0=-40.0, exponent=2.5).fit(-40 - 25 * np.log10(r))


def test_known_height_removes_the_ceiling_mirror_ambiguity():
    """UWB anchors on the ceiling (2.2-2.8 m) and a tag carried at 1.2 m: the 3-D solve cannot
    tell the tag from its mirror image above the anchors' plane and returns such points; with
    ``height=1.2`` there are none (output ``(x, y, 1.2)`` in the anchors' frame), the error is
    the horizontal CRLB with the height known, and ``spread`` equals that bound (computed here
    independently). Anchors exactly in one plane: the 3-D solve places nothing, ``height`` all."""
    rng = np.random.default_rng(3)
    anchors = np.array([[0.0, 0.0, 2.5], [12.0, 0.0, 2.2], [12.0, 8.0, 2.8], [0.0, 8.0, 2.4], [6.0, 4.0, 2.6]])
    truth = np.column_stack([rng.uniform([0.5, 0.5], [11.5, 7.5], (500, 2)), np.full(500, 1.2)])
    r = _ranges(truth, anchors)
    exact = TrilaterationLocalizer(anchors, height=1.2).localize(r)
    assert exact.pos.shape == (500, 3) and np.allclose(exact.pos, truth, atol=1e-9)
    X = r + 0.1 * rng.standard_normal(r.shape)
    free = TrilaterationLocalizer(anchors).localize(X).pos
    assert np.mean(free[:, 2] > anchors[:, 2].min()) > 0.1  # mirror solutions above the anchors
    fixed = TrilaterationLocalizer(anchors, height=1.2, sigma=0.1).localize(X)
    assert np.all(fixed.pos[:, 2] == 1.2)
    u = (truth[:, None] - anchors[None]) / r[..., None]
    crlb = 0.1 * np.sqrt(np.trace(np.linalg.inv(np.einsum("nai,naj->nij", u[..., :2], u[..., :2])), axis1=1, axis2=2))
    err = np.linalg.norm(fixed.pos - truth, axis=1)
    assert err.max() < 6 * crlb.max()  # no mirror (or any gross) solution
    assert 0.9 < np.sqrt(np.mean(err ** 2) / np.mean(crlb ** 2)) < 1.1
    ideal = TrilaterationLocalizer(anchors, height=1.2, sigma=0.1).localize(r).spread
    assert np.allclose(ideal, crlb, rtol=1e-9)
    for solver in ("linear", "irls"):
        est = TrilaterationLocalizer(anchors, height=1.2, solver=solver).predict(X)
        assert np.linalg.norm(est - truth, axis=1).max() < 6 * crlb.max()
    model = TrilaterationLocalizer(anchors, height=1.2, calibrate=True).fit(r + 0.25, truth)  # learns the bias
    assert np.allclose(model.bias_, 0.25) and np.allclose(model.predict(r + 0.25), truth, atol=1e-9)
    flat = anchors * [1.0, 1.0, 0.0] + [0.0, 0.0, 2.6]  # exactly coplanar
    rf = _ranges(truth, flat)
    assert np.isnan(TrilaterationLocalizer(flat).predict(rf)).all()
    assert np.allclose(TrilaterationLocalizer(flat, height=1.2).predict(rf), truth, atol=1e-9)
    missing = rf.copy()
    missing[0, :3] = np.nan  # two ranges left: not placed, z included
    assert np.isnan(TrilaterationLocalizer(flat, height=1.2).predict(missing)[0]).all()
    with pytest.raises(ValueError, match="3-D anchors"):
        TrilaterationLocalizer(ROOM, height=1.2).localize(_ranges(truth[:3, :2], ROOM))


def test_from_meta_takes_the_geometry_of_a_table():
    """``from_meta`` reads ``meta["anchors"]`` (and the AoA orientations: the ``aoa`` modality
    holds angles in each anchor's frame, so forgetting them gives errors of metres)."""
    anchors = ROOM[:4]
    orient = np.array([0.6, 2.4, -2.4, -0.6])
    pos = np.array([[5.0, 5.0], [12.0, 3.0], [8.0, 9.0]])
    r = _ranges(pos, anchors)
    local = np.arctan2(pos[:, None, 1] - anchors[None, :, 1], pos[:, None, 0] - anchors[None, :, 0]) - orient
    local = np.angle(np.exp(1j * local))
    tri = TrilaterationLocalizer.from_meta({"modality": "ranges", "anchors": anchors}, solver="irls")
    assert tri.solver == "irls" and np.allclose(tri.predict(r), pos)
    tdoa = TDOALocalizer.from_meta({"modality": "tdoa", "anchors": anchors, "reference_anchor": 0})
    assert np.allclose(tdoa.predict(r[:, 1:] - r[:, :1]), pos)
    aoa = AoALocalizer.from_meta({"modality": "aoa", "anchors": anchors, "anchor_orientations": orient})
    assert np.allclose(aoa.orientations, orient) and np.allclose(aoa.predict(local), pos)
    assert np.linalg.norm(AoALocalizer(anchors).predict(local) - pos, axis=1).max() > 1.0  # orientations forgotten
    with pytest.raises(ValueError, match="orientations="):
        AoALocalizer.from_meta({"modality": "aoa", "anchors": anchors})
    assert AoALocalizer.from_meta({"anchors": anchors}, orientations=None).orientations is None
    with pytest.raises(ValueError, match="'tdoa' table"):
        TDOALocalizer.from_meta({"modality": "ranges", "anchors": anchors})
    with pytest.raises(ValueError, match="reference_anchor"):
        TDOALocalizer.from_meta({"modality": "tdoa", "anchors": anchors, "reference_anchor": 2})
    with pytest.raises(ValueError, match="meta\\['anchors'\\]"):
        TrilaterationLocalizer.from_meta({"modality": "ranges"})
    ceiling = np.column_stack([anchors, [2.5, 2.2, 2.8, 2.4]])
    tri = TrilaterationLocalizer.from_meta({"modality": "ranges", "anchors": ceiling}, height=1.2)
    tag = np.column_stack([pos, np.full(3, 1.2)])
    assert tri.height == 1.2 and np.allclose(tri.predict(_ranges(tag, ceiling)), tag)


def test_aoa_fit_warns_when_the_angles_are_in_another_frame():
    """Local angles fitted without their orientations disagree with the labelled positions by
    far more than noise: ``fit`` warns and names the fix. Global bearings with 20 degrees of
    noise, local angles with their orientations, and ``calibrate=True`` do not warn."""
    import warnings

    rng = np.random.default_rng(7)
    anchors = ROOM[:4]
    orient = np.array([0.6, 2.4, -2.4, -0.6])
    pos = rng.uniform([2, 2], [18, 13], (200, 2))
    glob = np.arctan2(pos[:, None, 1] - anchors[None, :, 1], pos[:, None, 0] - anchors[None, :, 0])
    local = np.angle(np.exp(1j * (glob - orient)))
    with pytest.warns(UserWarning, match=r"anchor\(s\) \[1, 2\].*orientations="):
        AoALocalizer(anchors).fit(local, pos)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AoALocalizer(anchors).fit(glob + np.deg2rad(20) * rng.standard_normal(glob.shape), pos)
        AoALocalizer(anchors, orient).fit(local, pos)
        AoALocalizer(anchors, calibrate=True).fit(local, pos)
        AoALocalizer(anchors).fit(local)  # no positions: nothing to check
