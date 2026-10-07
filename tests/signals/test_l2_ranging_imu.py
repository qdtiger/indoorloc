"""L2 ranging and IMU helpers: closed-form conversions, bias models, NLOS flags, filters."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.signals import imu, ranging

NAN = np.nan


def test_time_and_rssi_conversions():
    assert ranging.SPEED_OF_LIGHT == 299_792_458.0
    assert ranging.toa_to_distance(1e-8) == pytest.approx(2.99792458)
    assert ranging.rtt_to_distance(2e-8) == pytest.approx(2.99792458)  # half the round trip
    assert ranging.rtt_to_distance(2e-8 + 5e-6, turnaround_s=5e-6) == pytest.approx(2.99792458, rel=1e-9)
    # log-distance model: 20 dB per decade at n = 2
    d = np.array([1.0, 10.0, 100.0, NAN])
    rssi = ranging.distance_to_rssi(d, p0_dbm=-59.0, n=2.0)
    assert np.array_equal(rssi, [-59.0, -79.0, -99.0, NAN], equal_nan=True)
    np.testing.assert_allclose(ranging.rssi_to_distance(rssi[:3], -59.0, 2.0), d[:3])
    assert ranging.rssi_to_distance(-89.0, -59.0, n=3.0) == pytest.approx(10.0)


def test_range_bias_fit_recovers_scale_and_offset_per_anchor():
    rng = np.random.default_rng(0)
    true = rng.uniform(1.0, 30.0, size=(50, 3))
    measured = true * np.array([1.02, 0.99, 1.0]) + np.array([0.35, -0.2, 0.6])
    measured[4, 1] = NAN  # an outage is ignored
    scale, offset = ranging.fit_range_bias(measured, true)
    np.testing.assert_allclose(scale, [1.02, 0.99, 1.0], atol=1e-12)
    np.testing.assert_allclose(offset, [0.35, -0.2, 0.6], atol=1e-12)
    corrected = ranging.correct_range_bias(measured, scale, offset)
    np.testing.assert_allclose(corrected, np.where(np.isnan(measured), NAN, true))
    s, o = ranging.fit_range_bias(true[:, 0] + 0.5, true[:, 0])
    assert isinstance(s, float) and s == pytest.approx(1.0) and o == pytest.approx(0.5)
    with pytest.raises(ValueError, match="distinct"):
        ranging.fit_range_bias(np.ones(5), np.ones(5))


def test_nlos_power_flag():
    rx = np.array([-80.0, -80.0, -80.0, NAN])
    fp = np.array([-82.0, -88.0, -95.0, -90.0])
    assert ranging.nlos_flags_power(rx, fp, threshold_db=6.0).tolist() == [False, True, True, False]


def test_nlos_std_flag_detects_jitter_not_motion():
    t = np.arange(40.0)
    walking = 5.0 + 0.5 * t  # a target moving away: a trend, not jitter
    jitter = walking + np.where(t >= 20, np.tile([3.0, -3.0], 20), 0.0)  # NLOS from t = 20
    flags = ranging.nlos_flags_std(np.column_stack([walking, jitter]), window=6, threshold_m=1.0)
    assert not flags[:, 0].any()  # degree-1 detrending removes the motion exactly
    assert not flags[:20, 1].any() and flags[25:, 1].all()
    assert not ranging.nlos_flags_std(walking, 6, 1.0, degree=1).any()
    # without detrending, motion looks like jitter: a 0.5 m/sample ramp has std 0.935 m over 6 samples
    assert ranging.nlos_flags_std(walking, 6, 0.9, degree=0)[5:].all()
    assert not ranging.nlos_flags_std(walking[:3], 6, 1.0).any()


def test_nlos_geometry_flag_is_the_triangle_inequality():
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    point = np.array([5.0, 0.0])
    d = np.linalg.norm(anchors - point, axis=1)
    assert not ranging.nlos_flags_geometry(d, anchors).any()  # LOS ranges are always consistent
    biased = d + np.array([12.0, 0.0, 0.0])  # d0 - d1 = 12 > |a0 - a1| = 10
    assert ranging.nlos_flags_geometry(biased, anchors).tolist() == [True, False, False]
    small = d + np.array([3.0, 0.0, 0.0])  # a small bias keeps the ranges consistent: not detectable
    assert not ranging.nlos_flags_geometry(small, anchors).any()
    assert ranging.nlos_flags_geometry(np.stack([d, biased]), anchors).shape == (2, 3)
    assert not ranging.nlos_flags_geometry(biased, anchors, tol_m=2.5).any()


def test_imu_magnitude_and_moving_average():
    assert imu.magnitude(np.array([[3.0, 4.0, 12.0], [0.0, 0.0, 0.0]])).tolist() == [13.0, 0.0]
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert imu.moving_average(x, 3).tolist() == [1.5, 2.0, 3.0, 4.0, 4.5]  # truncated at the edges
    assert imu.moving_average(x, 3, centered=False).tolist() == [1.0, 1.5, 2.0, 3.0, 4.0]
    assert imu.moving_average(np.array([1.0, NAN, 3.0]), 3).tolist() == [1.0, 2.0, 3.0]
    two = imu.moving_average(np.column_stack([x, 2 * x]), 3)
    assert two.shape == (5, 2) and two[:, 1].tolist() == [3.0, 4.0, 6.0, 8.0, 9.0]
    with pytest.raises(ValueError, match="odd"):
        imu.moving_average(x, 4)


def test_low_pass_step_response_is_closed_form():
    step = np.r_[0.0, np.ones(20)]
    y = imu.low_pass(step, alpha=0.3)
    np.testing.assert_allclose(y[1:], 1 - 0.7 ** np.arange(1, 21))  # y[t] = 1 - (1 - alpha)^t
    alpha = imu.smoothing_factor(cutoff_hz=1.0, rate_hz=50.0)
    assert alpha == pytest.approx((1 / 50) / (1 / (2 * np.pi) + 1 / 50))
    np.testing.assert_allclose(imu.low_pass(step, cutoff_hz=1.0, rate_hz=50.0), imu.low_pass(step, alpha))
    # starts at the first reading; a NaN sample holds the output
    assert np.array_equal(imu.low_pass(np.array([NAN, 2.0, NAN, 4.0]), 0.5), [NAN, 2.0, 2.0, 3.0], equal_nan=True)
    with pytest.raises(ValueError, match="alpha"):
        imu.low_pass(step)


def test_remove_gravity_separates_constant_gravity_from_motion():
    T = 2000
    gravity = np.array([0.0, 3.0, np.sqrt(imu.GRAVITY ** 2 - 9.0)])
    motion = np.zeros((T, 3))
    motion[:, 0] = 2.0 * np.sin(2 * np.pi * 2.0 * np.arange(T) / 100.0)  # 2 Hz sway at 100 Hz
    linear, g = imu.remove_gravity(gravity + motion, cutoff_hz=0.05, rate_hz=100.0)
    assert np.linalg.norm(gravity) == pytest.approx(imu.GRAVITY)
    # a first-order low-pass passes |H| = 1 / sqrt(1 + (f / fc)^2) = 0.025 at 2 Hz: 2 m/s^2 -> 0.05
    assert np.abs(g[-500:] - gravity).max() < 0.06
    np.testing.assert_allclose(linear + g, gravity + motion, rtol=0, atol=1e-12)
    still, g_still = imu.remove_gravity(np.tile(gravity, (10, 1)))
    assert np.abs(still).max() == 0.0 and np.allclose(g_still, gravity)


def test_path_loss_parameters_are_validated():
    for bad in ({"n": 0.0}, {"n": -2.0}, {"d0_m": 0.0}):
        with pytest.raises(ValueError, match="must be positive"):
            ranging.rssi_to_distance(-70.0, -40.0, **bad)
        with pytest.raises(ValueError, match="must be positive"):
            ranging.distance_to_rssi(3.0, -40.0, **bad)
    # the reference distance scales the result: p0 at d0 = 2 m, 20 dB per decade -> 20 m
    assert ranging.rssi_to_distance(-60.0, -40.0, n=2.0, d0_m=2.0) == pytest.approx(20.0)
