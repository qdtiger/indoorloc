"""L5 tracking: Kalman filters, RTS smoother, EKF on ranges (known-result tests)."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.apps.tracking import (ConstantVelocityKF, ExtendedKalmanTracker, KalmanTracker, motion_model,
                                     multilaterate)
from indoorloc.core import Prediction, load_model


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


def test_motion_model_closed_forms_and_dt_consistency():
    F, Q = motion_model(2.0, "cv", dim=1, q=0.5)
    assert np.allclose(F, [[1, 2], [0, 1]])
    assert np.allclose(Q, 0.5 * np.array([[8 / 3, 2], [2, 2]]))  # q [[dt^3/3, dt^2/2], [dt^2/2, dt]]
    F, Q = motion_model(1.0, "cv", dim=1, q=4.0, noise="discrete")
    assert np.allclose(Q, 4.0 * np.array([[0.25, 0.5], [0.5, 1.0]]))  # sigma^2 G G', G = [dt^2/2, dt]
    # continuous-time models compose exactly: two steps of dt/2 equal one step of dt
    for motion in ("cv", "ca"):
        F1, Q1 = motion_model(0.7, motion, dim=2, q=0.3)
        F2, Q2 = motion_model(1.4, motion, dim=2, q=0.3)
        assert np.allclose(F1 @ F1, F2)
        assert np.allclose(F1 @ Q1 @ F1.T + Q1, Q2)
    F, _ = motion_model(1.0, "ca", dim=2)
    assert F.shape == (6, 6) and np.allclose(F[0], [1, 0, 1, 0, 0.5, 0])  # state [x, y, vx, vy, ax, ay]
    with pytest.raises(ValueError, match="motion"):
        motion_model(1.0, "jerk")


def test_steady_state_gain_equals_the_kalata_alpha_beta_filter():
    # Kalata (IEEE TAES 1984): with piecewise-constant white acceleration (std sw), sample time T and
    # position noise sv, the steady-state Kalman gain is (alpha, beta / T) with tracking index
    # lam = sw T^2 / sv.  P. R. Kalata, IEEE TAES 20(2):174-182, 1984. DOI 10.1109/TAES.1984.310438
    T, sw, sv = 0.5, 0.8, 2.0
    lam = sw * T ** 2 / sv
    root = np.sqrt(lam ** 2 + 8 * lam)
    alpha = -(lam ** 2 + 8 * lam - (lam + 4) * root) / 8
    beta = (lam ** 2 + 4 * lam - lam * root) / 4
    kt = KalmanTracker(process_noise=sw ** 2, noise="discrete", meas_std=sv, use_spread=False)
    z = np.zeros((400, 1))
    kt.filter(z, t=np.arange(400) * T)
    kt.predict(kt.t_ + T)  # P_ is now the steady-state prediction covariance
    P = kt.P_
    K = P[:, 0] / (P[0, 0] + sv ** 2)
    assert K[0] == pytest.approx(alpha, rel=1e-9)
    assert K[1] * T == pytest.approx(beta, rel=1e-9)


def test_rts_smoother_equals_the_batch_least_squares_solution():
    # The RTS smoother is the exact MAP estimate of the linear-Gaussian model: build the batch
    # information matrix (prior + dynamics + measurements) and solve it directly.
    t = np.array([0.0, 0.5, 1.7, 2.0, 3.1, 4.0])
    z = np.array([0.3, 1.1, 2.2, 3.9, 4.1, 6.0])[:, None]
    r, q, v0 = 0.7, 0.4, 1.5
    kt = KalmanTracker(process_noise=q, meas_std=r, use_spread=False, init_vel_std=v0)
    kt.smooth(z, t)
    n, S = len(t), 2
    Lam, eta = np.zeros((n * S, n * S)), np.zeros(n * S)
    P0inv = np.diag([1 / r ** 2, 1 / v0 ** 2])
    Lam[:2, :2] += P0inv
    eta[:2] += P0inv @ np.array([z[0, 0], 0.0])
    H = np.array([[1.0, 0.0]])
    for k in range(1, n):
        F, Q = motion_model(t[k] - t[k - 1], "cv", dim=1, q=q)
        Qi = np.linalg.inv(Q)
        A = np.zeros((S, n * S))
        A[:, (k - 1) * S:k * S], A[:, k * S:(k + 1) * S] = -F, np.eye(S)
        Lam += A.T @ Qi @ A
        Lam[k * S:(k + 1) * S, k * S:(k + 1) * S] += H.T @ H / r ** 2
        eta[k * S:(k + 1) * S] += (H.T * z[k, 0] / r ** 2).ravel()
    x = np.linalg.solve(Lam, eta).reshape(n, S)
    cov = np.linalg.inv(Lam)
    assert np.allclose(kt.states_, x, atol=1e-10)
    for k in range(n):
        assert np.allclose(kt.covariances_[k], cov[k * S:(k + 1) * S, k * S:(k + 1) * S], atol=1e-10)


def test_noise_free_linear_motion_is_tracked_exactly_and_ca_removes_the_cv_lag():
    t = np.arange(60, dtype=float)
    line = np.stack([2.0 + 0.8 * t, -1.0 + 0.3 * t], axis=1)
    kt = KalmanTracker(process_noise=1e-3, meas_std=0.5, use_spread=False)
    out = kt.filter(line, t)
    assert np.abs(out.pos[-1] - line[-1]).max() < 1e-3
    assert np.allclose(kt.x_[2:], [0.8, 0.3], atol=1e-3)  # the velocity is learned
    parabola = np.stack([0.05 * t ** 2, t], axis=1)  # constant acceleration 0.1 m/s^2 along x
    cv = KalmanTracker("cv", process_noise=1e-3, meas_std=0.5, use_spread=False).filter(parabola, t)
    ca = KalmanTracker("ca", process_noise=1e-3, meas_std=0.5, use_spread=False).filter(parabola, t)
    assert abs(ca.pos[-1, 0] - parabola[-1, 0]) < 1e-3
    assert abs(cv.pos[-1, 0] - parabola[-1, 0]) > 0.1  # the CV model lags an accelerating target


def test_smoothing_reduces_the_error_of_filtering_which_reduces_the_error_of_raw_fixes():
    rng = np.random.default_rng(0)
    t = np.arange(200, dtype=float)
    vel = np.cumsum(rng.normal(0, 0.1, (200, 2)), axis=0) + [1.0, 0.5]
    truth = np.cumsum(vel, axis=0)
    fixes = truth + rng.normal(0, 3.0, truth.shape)
    kt = KalmanTracker(process_noise=0.01, meas_std=3.0, use_spread=False)
    e_raw, e_f, e_s = _rmse(fixes, truth), _rmse(kt.filter(fixes, t).pos, truth), _rmse(kt.smooth(fixes, t).pos, truth)
    assert e_s < 0.8 * e_f < 0.8 * e_raw


def test_measurement_noise_from_spread_beats_a_fixed_noise():
    rng = np.random.default_rng(1)
    t = np.arange(300, dtype=float)
    truth = np.stack([np.cos(t / 40) * 20, np.sin(t / 40) * 20], axis=1)
    spread = np.where(np.arange(300) % 2, 8.0, 1.0)  # every other fix is poor and says so
    fixes = truth + rng.normal(0, 1, truth.shape) * spread[:, None]
    pred = Prediction(fixes, spread=spread)
    adaptive = KalmanTracker(process_noise=0.05, min_meas_std=0.1).filter(pred, t)
    fixed = KalmanTracker(process_noise=0.05, use_spread=False, meas_std=float(np.sqrt(np.mean(spread ** 2))))
    assert _rmse(adaptive.pos, truth) < 0.8 * _rmse(fixed.filter(pred, t).pos, truth)


def test_missing_fixes_are_predicted_and_outliers_are_gated():
    t = np.arange(40, dtype=float)
    line = np.stack([0.5 * t, 0.25 * t], axis=1)
    z = line.copy()
    z[20:25] = np.nan  # five lost scans
    z[30] += [60.0, 0.0]  # one wild fix
    kt = KalmanTracker(process_noise=1e-4, meas_std=0.3, use_spread=False, gate=3.0)
    out = kt.filter(z, t)
    assert np.abs(out.pos[20:25] - line[20:25]).max() < 0.01  # constant velocity carries the track
    assert np.isnan(kt.nis_[20:25]).all() and not kt.accepted_[30] and kt.nis_[30] > 9.0
    assert np.abs(out.pos[30:] - line[30:]).max() < 0.05
    ungated = KalmanTracker(process_noise=1e-4, meas_std=0.3, use_spread=False).filter(z, t)
    assert np.abs(ungated.pos[30] - line[30]).max() > 10.0


def test_rows_before_the_first_fix_are_nan_and_labels_pass_through():
    z = np.array([[np.nan, np.nan], [1.0, 1.0], [2.0, 2.0]])
    pred = Prediction(z, floor=[3, 3, 4], ids=["a", "b", "c"], spread=[np.nan, 1.0, 1.0])
    out = KalmanTracker().filter(pred)
    assert np.isnan(out.pos[0]).all() and np.allclose(out.pos[1], [1.0, 1.0])
    assert out.floor.tolist() == [3, 3, 4] and out.ids.tolist() == ["a", "b", "c"]
    assert out.spread[1] == pytest.approx(np.sqrt(2.0))  # sqrt(trace(P_pos)) with r = 1 m


def test_online_updates_equal_the_offline_filter_and_survive_save_load(tmp_path):
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.uniform(0.5, 1.5, 30))
    z = np.cumsum(rng.normal(0, 1, (30, 2)), axis=0)
    z[7] = np.nan
    offline = KalmanTracker(motion="ca", process_noise=0.2).filter(z, t)
    kt = KalmanTracker(motion="ca", process_noise=0.2).reset()
    online = np.array([kt.update(tk, None if np.isnan(zk).any() else zk) for tk, zk in zip(t, z)])
    assert np.allclose(online, offline.pos)
    head = KalmanTracker(motion="ca", process_noise=0.2).reset()
    for tk, zk in zip(t[:10], z[:10]):
        head.update(tk, zk)
    again = load_model(head.save(tmp_path / "kf"))
    assert np.allclose([again.update(tk, zk) for tk, zk in zip(t[10:], z[10:])], offline.pos[10:])


def test_constant_velocity_kf_is_the_discrete_noise_kalman_tracker():
    rng = np.random.default_rng(3)
    z = np.cumsum(rng.normal(0, 1, (25, 2)), axis=0)
    kf = ConstantVelocityKF(accel_std=0.7)
    sketch = [kf.reset(z[0], 2.0).x_[:2].copy()] + [kf.step(zk, 2.0, 1.0) for zk in z[1:]]
    general = KalmanTracker(process_noise=0.49, noise="discrete", meas_std=2.0, use_spread=False).filter(z)
    assert np.allclose(sketch, general.pos, atol=1e-12)


def test_multilateration_is_exact_on_noise_free_ranges():
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0], [10.0, 8.0]])
    p = np.array([3.3, 6.1])
    r = np.linalg.norm(anchors - p, axis=1)
    assert np.allclose(multilaterate(r, anchors), p, atol=1e-9)
    assert multilaterate(np.r_[r[:2], np.nan, np.nan], anchors) is None  # 2 ranges cannot fix 2-D
    a3 = np.c_[anchors, [2.5, 2.8, 2.5, 3.0]]
    r3 = np.linalg.norm(a3 - np.r_[p, 1.2], axis=1)
    assert np.allclose(multilaterate(r3, a3, height=1.2), p, atol=1e-9)


def test_ekf_on_ranges_converges_and_keeps_tracking_with_two_anchors():
    anchors = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 20.0], [30.0, 20.0]])
    t = np.arange(80, dtype=float)
    truth = np.stack([2 + 0.3 * t, 3 + 0.15 * t], axis=1)
    R = np.linalg.norm(truth[:, None] - anchors[None], axis=2)
    exact = ExtendedKalmanTracker(anchors, range_std=0.1, process_noise=1e-4).filter(R, t)
    assert np.abs(exact.pos[-1] - truth[-1]).max() < 1e-3
    rng = np.random.default_rng(4)
    noisy = R + rng.normal(0, 0.3, R.shape)
    noisy[30:, 2:] = np.nan  # two anchors lost: snapshot multilateration is impossible from t = 30
    assert all(multilaterate(r, anchors) is None for r in noisy[30:])
    ekf = ExtendedKalmanTracker(anchors, range_std=0.3, process_noise=1e-3)
    out = ekf.smooth(noisy, t)
    snap = np.array([multilaterate(r, anchors) for r in noisy[:30]])
    assert _rmse(out.pos[:30], truth[:30]) < 0.7 * _rmse(snap, truth[:30])
    assert np.abs(out.pos[30:] - truth[30:]).max() < 1.0  # tight coupling carries on with two anchors


def test_ekf_gate_drops_a_nlos_range_and_planar_tracking_with_3d_anchors():
    anchors = np.array([[0.0, 0.0, 2.5], [12.0, 0.0, 2.7], [0.0, 9.0, 2.6], [12.0, 9.0, 2.4]])
    t = np.arange(30, dtype=float)
    truth = np.stack([1 + 0.3 * t, 1 + 0.2 * t], axis=1)
    R = np.linalg.norm(np.c_[truth, np.full(30, 1.1)][:, None] - anchors[None], axis=2)
    R[20, 1] += 5.0  # a non-line-of-sight range, 5 m too long
    gated = ExtendedKalmanTracker(anchors, range_std=0.05, process_noise=1e-3, height=1.1, gate=4.0)
    out = gated.filter(R, t)
    assert np.abs(out.pos - truth).max() < 0.05
    plain = ExtendedKalmanTracker(anchors, range_std=0.05, process_noise=1e-3, height=1.1).filter(R, t)
    assert np.abs(plain.pos[20] - truth[20]).max() > 0.5


def test_track_falls_back_to_the_fixed_noise_without_a_spread():
    from indoorloc.apps.tracking import track

    class NoSpread:  # a localizer whose Prediction carries no spread
        def localize(self, X):
            return Prediction(np.asarray(X, dtype=float)[:, :2])

    rng = np.random.default_rng(5)
    z = np.cumsum(rng.normal(0, 1, (12, 2)), axis=0)
    out = [xy for _, _, xy in track(NoSpread(), ConstantVelocityKF(accel_std=0.4), [(float(k), zk) for k, zk in
                                                                                    enumerate(z)], fixed_std=2.5)]
    ref = KalmanTracker(process_noise=0.16, noise="discrete", meas_std=2.5, use_spread=False).filter(z)
    assert np.allclose(out, ref.pos, atol=1e-12)
