"""L5 PDR: step detection, step length models, heading, dead-reckoned trajectories."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.apps.pdr import (GRAVITY, PDR, StepDetector, calibrate_step_length, complementary_heading,
                                integrate_heading, kim_step_length, magnetic_heading, moving_average, step_lengths,
                                weinberg_step_length, wrap_angle, yaw_rate)
from indoorloc.core import SampleTable

FS = 100.0


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def _walk_signal(freq, seconds, amp=2.5, harmonic=0.8, noise=0.3, seed=0):
    """|a| of walking: gravity + a bounce per step (phase-integrated cadence) + harmonic + noise."""
    t = np.arange(int(seconds * FS)) / FS
    phase = np.pi + 2 * np.pi * np.cumsum(np.broadcast_to(freq, t.shape)) / FS  # start in a valley
    rng = np.random.default_rng(seed)
    mag = GRAVITY + amp * np.cos(phase) + harmonic * np.cos(2 * phase + 0.4) + rng.normal(0, noise, len(t))
    return t, np.stack([np.zeros_like(t), np.zeros_like(t), mag], axis=1), phase


def test_step_count_on_synthetic_walking_waveforms():
    t, acc, _ = _walk_signal(2.0, 10.0)
    steps = StepDetector().detect(acc, t)
    assert len(steps) == 20 and np.allclose(np.diff(steps.t), 0.5, atol=0.03)
    t, acc, phase = _walk_signal(np.linspace(1.4, 2.4, 3000), 30.0, seed=1)  # cadence sweeps 1.4 -> 2.4 Hz
    expected = int(np.floor(phase[-1] / (2 * np.pi)))  # cos peaks at phase = 2 pi, 4 pi, ...
    assert len(StepDetector().detect(acc, t)) == expected
    quiet = np.c_[np.zeros((1000, 2)), GRAVITY + np.random.default_rng(2).normal(0, 0.2, 1000)]
    assert len(StepDetector().detect(quiet, rate_hz=FS)) == 0  # standing still: no false steps


def test_min_interval_and_valley_reject_double_peaks():
    t = np.arange(300) / FS
    mag = np.full(300, GRAVITY - 1.0)
    for c in (0.5, 0.62, 1.5, 2.5):  # 0.62 s: a second bump 0.12 s after the first, with a valley between
        mag += 3.0 * np.exp(-0.5 * ((t - c) / 0.02) ** 2)
    det = StepDetector(smoothing=0.0, gravity=GRAVITY)
    assert det.detect(mag, t).t.tolist() == pytest.approx([0.5, 1.5, 2.5])
    assert len(StepDetector(smoothing=0.0, gravity=GRAVITY, min_interval=0.1).detect(mag, t)) == 4


def test_step_length_models_closed_forms_and_calibration():
    assert weinberg_step_length(3.0, -1.0, k=0.5) == pytest.approx(0.5 * np.sqrt(2))  # k (4)^(1/4)
    assert kim_step_length(8.0, k=0.5) == pytest.approx(1.0)  # k (8)^(1/3)
    t, acc, _ = _walk_signal(2.0, 10.0, amp=2.0, harmonic=0.0, noise=0.0)
    steps = StepDetector(smoothing=0.0).detect(acc, t)
    assert np.allclose(steps.peak[1:-1], 2.0) and np.allclose(steps.valley[1:-1], -2.0)  # a pure cosine
    assert np.allclose(steps.mean_abs[1:-1], 2 * 2.0 / np.pi, rtol=0.01)  # mean |A cos| = 2A / pi
    L = step_lengths(steps, "weinberg", k=0.5)
    assert np.allclose(L[1:-1], 0.5 * 4.0 ** 0.25)
    k = calibrate_step_length(steps, 14.0, "kim")
    assert step_lengths(steps, "kim", k).sum() == pytest.approx(14.0)
    assert np.allclose(step_lengths(steps, "constant", 0.65), 0.65)


def test_heading_integration_is_exact_for_linear_yaw_rates():
    t = np.arange(1001) / FS
    assert integrate_heading(np.full_like(t, 0.1), t)[-1] == pytest.approx(1.0)
    assert np.allclose(integrate_heading(0.3 * t, t, heading0=0.5), 0.5 + 0.15 * t ** 2)  # trapezoid: exact
    assert np.allclose(wrap_angle([np.pi, -np.pi, 3 * np.pi / 2]), [-np.pi, -np.pi, -np.pi / 2])


def test_yaw_rate_about_gravity_and_tilt_compensated_compass():
    b_world = np.array([0.0, 22.0, -41.0])  # field points north and down (ENU), microtesla
    for psi, roll, pitch in [(0.3, 0.0, 0.0), (2.5, 0.4, -0.3), (-1.2, -0.6, 0.5)]:
        R = _rot("z", psi - np.pi / 2) @ _rot("x", roll) @ _rot("y", pitch)  # device -> world; device +y at psi
        n = 200
        acc = np.tile(R.T @ [0, 0, GRAVITY], (n, 1))
        mag = np.tile(R.T @ b_world, (n, 1))
        gyro = np.tile(R.T @ [0, 0, 0.25], (n, 1))  # turning counter-clockwise at 0.25 rad/s
        assert np.allclose(magnetic_heading(mag, acc, rate_hz=FS), psi)
        assert np.allclose(yaw_rate(gyro, acc, rate_hz=FS), 0.25)
    assert np.allclose(magnetic_heading(mag, acc, rate_hz=FS, offset=0.1), -1.2 + 0.1)


def test_complementary_filter_steady_state_error_is_bias_times_tau():
    t = np.arange(20000) / FS  # 200 s
    bias, tau = 0.01, 5.0
    fused = complementary_heading(np.full_like(t, bias), np.zeros_like(t), t, time_constant=tau)
    assert fused[-1] == pytest.approx(bias * tau, rel=1e-6)
    assert integrate_heading(np.full_like(t, bias), t)[-1] == pytest.approx(bias * t[-1])  # gyro alone drifts
    mag = np.zeros_like(t)
    mag[5000:6000] = np.nan  # a disturbed stretch is skipped
    assert np.isfinite(complementary_heading(np.zeros_like(t), mag, t)).all()


def _square_walk(gyro_bias=0.0):
    """Four 14-step legs with left turns, phone flat, cadence 2 Hz, 0.7 m steps (A = 2)."""
    n_steps = 58
    t = np.arange(int(n_steps * 0.5 * FS)) / FS
    phase = np.pi + 2 * np.pi * 2.0 * t  # steps (peaks) at 0.25 s + 0.5 s k
    acc = np.c_[np.zeros((len(t), 2)), GRAVITY + 2.0 * np.cos(phase)]
    rate = np.zeros(len(t))
    for c in (7.0, 14.0, 21.0):  # quarter turns between steps 14/15, 28/29, 42/43
        rate[(t >= c - 0.1) & (t < c + 0.1)] = (np.pi / 2) / 0.2
    heading = integrate_heading(rate, t)
    gyro = np.c_[np.zeros((len(t), 2)), rate + gyro_bias]
    b = np.array([0.0, 22.0, -41.0])
    mag = np.stack([_rot("z", h - np.pi / 2).T @ b for h in heading])
    return t, acc, gyro, mag, heading


def test_pdr_closes_a_square_loop():
    t, acc, gyro, mag, heading = _square_walk()
    k = 0.7 / 4.0 ** 0.25  # Weinberg constant giving 0.7 m for a +-2 m/s^2 bounce
    track = PDR(StepDetector(smoothing=0.0), k=k, initial_heading=0.0).run(acc, gyro, t=t)
    assert len(track) == 58 and track.distance == pytest.approx(58 * 0.7, rel=1e-6)
    truth = np.cumsum(0.7 * np.stack([np.cos(heading[track.index]), np.sin(heading[track.index])], 1), 0)
    assert np.abs(track.pos - truth).max() < 1e-6
    assert np.linalg.norm(track.pos[55]) < 1.0  # back near the start after four legs
    assert np.allclose(track.position_at([track.t0, track.t[3]]), [[0, 0], track.pos[3]])


def test_fused_heading_removes_the_gyro_drift_that_gyro_only_pdr_accumulates():
    t, acc, gyro, mag, heading = _square_walk(gyro_bias=0.02)
    gyro_only = PDR(heading="gyro", initial_heading=0.0).headings(acc, gyro, mag, t)
    fused = PDR(heading="fused", time_constant=2.0).headings(acc, gyro, mag, t)
    compass = PDR(heading="mag").headings(acc, gyro, mag, t)
    err = lambda h: np.abs(wrap_angle(h - heading))[-500:].max()  # noqa: E731
    assert err(gyro_only) > 0.5 and err(fused) < 0.06 and err(compass) < 1e-9


def test_pdr_reads_an_imu_sample_table():
    t, acc, gyro, mag, _ = _square_walk()
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
    table = SampleTable(np.c_[acc, gyro, mag].astype(np.float32), np.zeros((len(t), 2)),
                        groups={"time": t, "trajectory": np.zeros(len(t), int)},
                        meta={"modality": "imu", "channels": names, "rate_hz": FS})
    pdr = PDR(initial_heading=0.0)
    from_table, from_arrays = pdr.run(table), pdr.run(acc.astype(np.float32), gyro.astype(np.float32), t=t)
    assert np.allclose(from_table.pos, from_arrays.pos)


def test_moving_average_is_centred_and_keeps_length():
    x = np.arange(10.0)
    assert np.allclose(moving_average(x, 3)[1:-1], x[1:-1]) and len(moving_average(x, 4)) == 10


def test_an_imu_table_with_several_trajectories_is_refused():
    from indoorloc.apps.pdr import imu_arrays

    t, acc, gyro, mag, _ = _square_walk()
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    two = SampleTable(np.c_[acc, gyro].astype(np.float32), np.zeros((len(t), 2)),
                      groups={"time": t, "trajectory": np.arange(len(t)) % 2},
                      meta={"modality": "imu", "channels": names, "rate_hz": FS})
    with pytest.raises(ValueError, match="several trajectories"):
        imu_arrays(two)
    one = imu_arrays(two[two.groups["trajectory"] == 0])
    assert one["mag"] is None and one["acc"].shape == (len(t) // 2, 3)


def test_missing_imu_samples_are_interpolated_not_propagated():
    # NaN is a missing sample (the L1 convention): a gyro and a magnetometer at half the table's rate
    # and 3 % of lost accelerometer rows must not stop the detector or poison the heading integral.
    t, acc, gyro, mag, heading = _square_walk()
    full = PDR(StepDetector(smoothing=0.0), k=0.7 / 4.0 ** 0.25, initial_heading=0.0).run(acc, gyro, t=t)
    rng = np.random.default_rng(7)
    acc_g, gyro_g, mag_g = acc.copy(), gyro.copy(), mag.copy()
    lost = rng.random(len(t)) < 0.03
    lost[full.index] = False  # keep the peaks themselves; their neighbours may be lost
    acc_g[lost] = np.nan
    gyro_g[1::2] = np.nan
    mag_g[1::2] = np.nan
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
    table = SampleTable(np.c_[acc_g, gyro_g, mag_g], np.zeros((len(t), 2)),
                        groups={"time": t, "trajectory": np.zeros(len(t), int)},
                        meta={"modality": "imu", "channels": names, "rate_hz": FS})
    track = PDR(StepDetector(smoothing=0.0), k=0.7 / 4.0 ** 0.25, initial_heading=0.0).run(table)
    assert len(track) == 58 and np.array_equal(track.index, full.index)
    # measured: identical to the full-rate track (the yaw-rate pulses are flat, so their linear
    # interpolation keeps the trapezoid integral, and no step's peak or valley row was lost)
    assert np.abs(track.pos - full.pos).max() < 1e-9
    assert len(StepDetector().detect(np.where(np.arange(len(t))[:, None] == 300, np.nan, acc), t)) == 58
