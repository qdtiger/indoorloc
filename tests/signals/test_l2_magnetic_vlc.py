"""L2 magnetometer, visible-light and ultrasound/two-way ranging science: closed forms and planted truths."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import SampleTable
from indoorloc.signals import MagneticFeatures, MagnetometerCalibration
from indoorloc.signals import magnetic as mg
from indoorloc.signals import ranging, vlc

DEG = np.pi / 180.0


def _earth(total=48.0, inclination=60.0, declination=0.0):
    i, d = inclination * DEG, declination * DEG
    return total * np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), -np.sin(i)])  # East-North-Up


def _rotation(axis, angle):
    """Rodrigues: rotation by ``angle`` about ``axis``."""
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _device_poses(n, seed=0, max_tilt=0.6):
    """Random device-to-world rotations: any yaw, tilt up to ``max_tilt`` rad; and their yaw of +y."""
    rng = np.random.default_rng(seed)
    R = np.stack([_rotation([0, 0, 1], rng.uniform(-np.pi, np.pi))
                  @ _rotation(np.r_[rng.normal(size=2), 0.0], rng.uniform(0, max_tilt)) for _ in range(n)])
    forward = R[:, :, 1]                                   # the device +y axis in the world frame
    return R, np.arctan2(forward[:, 1], forward[:, 0])


def _sphere(n, seed=0):
    u = np.random.default_rng(seed).normal(size=(n, 3))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


# ------------------------------------------------------------------------------- magnetometer
def test_components_and_heading_are_exact_under_any_tilt():
    b = _earth(declination=5.0)
    R, true_heading = _device_poses(300)
    mag = np.einsum("nji,j->ni", R, b)                     # world field in the device frame: R^T b
    grav = np.einsum("nji,j->ni", R, [0.0, 0.0, 9.81])     # a phone at rest reports +g along its up axis
    h, v = mg.field_components(mag, grav)
    np.testing.assert_allclose(h, 48.0 * np.cos(60 * DEG), atol=1e-12)
    np.testing.assert_allclose(v, -48.0 * np.sin(60 * DEG), atol=1e-12)    # positive up: negative in the north
    np.testing.assert_allclose(mg.inclination(mag, grav), 60 * DEG, atol=1e-12)
    np.testing.assert_allclose(mg.field_magnitude(mag), 48.0, atol=1e-12)
    heading = mg.heading(mag, grav, declination=5 * DEG)
    np.testing.assert_allclose(mg.wrap_angle(heading - true_heading), 0.0, atol=1e-12)
    # without the declination the heading is counted from magnetic east, 5 degrees clockwise of true east
    np.testing.assert_allclose(mg.wrap_angle(mg.heading(mag, grav) - true_heading - 5 * DEG), 0.0, atol=1e-12)
    feats = mg.magnetic_features(mag, grav)
    assert feats.shape == (300, 3) and np.allclose(feats[:, 1:], np.column_stack([h, v]))


def test_flat_device_defaults_and_single_readings():
    b = _earth()
    yaw = 30 * DEG                                         # device x axis 30 degrees left of east
    c, s = np.cos(yaw), np.sin(yaw)
    reading = np.array([c * b[0] + s * b[1], -s * b[0] + c * b[1], b[2]])
    assert mg.heading(reading, forward=(1.0, 0.0, 0.0)) == pytest.approx(yaw)
    assert mg.heading(reading) == pytest.approx(yaw + np.pi / 2)            # default forward: device +y
    h, v = mg.field_components(reading)
    assert np.ndim(h) == 0 and h == pytest.approx(np.hypot(b[0], b[1])) and v == pytest.approx(b[2])
    assert mg.magnetic_features(reading).shape == (3,)
    np.testing.assert_allclose(mg.wrap_angle([np.pi, -np.pi, 3 * np.pi, 0.5]), [-np.pi, -np.pi, -np.pi, 0.5])
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        mg.field_magnitude(np.ones((4, 2)))
    with pytest.raises(ValueError, match="forward"):
        mg.heading(reading, forward=(0.0, 0.0, 0.0))


def test_ellipsoid_fit_recovers_planted_hard_and_soft_iron():
    A = np.array([[1.10, 0.05, -0.02], [0.05, 0.93, 0.04], [-0.02, 0.04, 1.02]])   # symmetric soft iron
    hard = np.array([12.0, -30.0, 7.5])
    field = 50.0 * _sphere(400)
    raw = field @ A.T + hard
    offset, W = mg.fit_ellipsoid(raw, field_strength=50.0)
    np.testing.assert_allclose(offset, hard, atol=1e-9)
    np.testing.assert_allclose(W, np.linalg.inv(A), atol=1e-12)            # symmetric A: exactly A^-1
    np.testing.assert_allclose(mg.apply_calibration(raw, offset, W), field, atol=1e-9)
    # a rotation in A is not observable: W A is then orthogonal and W symmetric
    Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    offset, W = mg.fit_ellipsoid(field @ (Q @ A).T + hard, field_strength=50.0)
    np.testing.assert_allclose((W @ Q @ A) @ (W @ Q @ A).T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    # unknown field strength: unit determinant, W proportional to A^-1
    offset, W = mg.fit_ellipsoid(raw)
    assert np.linalg.det(W) == pytest.approx(1.0) and np.allclose(offset, hard)
    np.testing.assert_allclose(W / W[0, 0], np.linalg.inv(A) / np.linalg.inv(A)[0, 0], atol=1e-9)


def test_ellipsoid_fit_under_noise_and_its_guards():
    rng = np.random.default_rng(2)
    A = np.diag([1.2, 0.9, 1.05])
    hard = np.array([-20.0, 5.0, 40.0])
    raw = 50.0 * _sphere(2000, seed=3) @ A.T + hard + rng.normal(0.0, 0.5, (2000, 3))
    raw[7] = np.nan                                                          # dropped
    offset, W = mg.fit_ellipsoid(raw, field_strength=50.0)
    assert np.abs(offset - hard).max() < 0.1 and np.abs(W - np.linalg.inv(A)).max() < 5e-3
    calibrated = np.linalg.norm(mg.apply_calibration(raw[np.isfinite(raw).all(1)], offset, W), axis=1)
    assert calibrated.std() < 0.6 < 10.0 < np.linalg.norm(raw[np.isfinite(raw).all(1)], axis=1).std()
    flat = 50.0 * np.column_stack([np.cos(np.linspace(0, 6, 50)), np.sin(np.linspace(0, 6, 50)), np.zeros(50)])
    with pytest.raises(ValueError, match="span all three axes"):             # walking with the phone flat
        mg.fit_ellipsoid(flat + hard)
    with pytest.raises(ValueError, match="at least 9"):
        mg.fit_ellipsoid(raw[:5])


def test_magnetometer_calibration_transform_on_arrays_and_imu_tables():
    A, hard = np.diag([1.1, 0.95, 1.0]), np.array([3.0, -4.0, 10.0])
    field = 45.0 * _sphere(300, seed=4)
    raw = field @ A.T + hard
    cal = MagnetometerCalibration(field_strength=45.0).fit(raw)
    np.testing.assert_allclose(cal.transform(raw), field, atol=1e-9)
    assert cal.field_strength_ == 45.0 and cal.n_features_in_ == 3
    np.testing.assert_allclose(cal.transform(raw[0]), field[0], atol=1e-9)
    channels = ("acc_x", "acc_y", "acc_z", "mag_x", "mag_y", "mag_z")
    X = np.column_stack([np.tile([0.0, 0.0, 9.81], (300, 1)), raw]).astype(np.float32)
    table = SampleTable(X, np.zeros((300, 2)), meta={"modality": "imu", "channels": channels})
    out = MagnetometerCalibration(field_strength=45.0).fit(table).transform(table)
    assert out.X.dtype == np.float32 and np.array_equal(out.X[:, :3], X[:, :3])      # other channels kept
    np.testing.assert_allclose(out.X[:, 3:], field, atol=2e-4)
    unknown = MagnetometerCalibration().fit(raw)                              # median calibrated magnitude
    assert unknown.field_strength_ == pytest.approx(np.median(np.linalg.norm(unknown.transform(raw), axis=1)))
    gaps = raw.copy()
    gaps[[3, 50]] = np.nan                                                    # dropped rows do not poison F
    assert MagnetometerCalibration().fit(gaps).field_strength_ == pytest.approx(unknown.field_strength_, rel=1e-9)
    features = SampleTable(mg.magnetic_features(raw), np.zeros((300, 2)),
                           meta={"modality": "magnetic", "feature_names": ("B", "B_h", "B_v")})
    for step in (MagnetometerCalibration().fit, cal.transform):              # [B, B_h, B_v] are not readings
        with pytest.raises(ValueError, match="raw magnetometer"):
            step(features)


def test_magnetometer_calibration_persists_without_pickle(tmp_path):
    from indoorloc.core import load_model

    raw = 45.0 * _sphere(200, seed=8) @ np.diag([1.1, 0.95, 1.0]).T + [3.0, -4.0, 10.0]
    cal = MagnetometerCalibration().fit(raw)
    cal.save(tmp_path / "cal")
    back = load_model(tmp_path / "cal")
    np.testing.assert_array_equal(back.transform(raw), cal.transform(raw))
    assert back.field_strength_ == cal.field_strength_ and back.get_params() == cal.get_params()


def test_magnetic_features_turn_imu_tables_into_magnetic_tables():
    b = _earth()
    R, _ = _device_poses(40, seed=5, max_tilt=0.4)
    mag = np.einsum("nji,j->ni", R, b)
    grav = np.einsum("nji,j->ni", R, [0.0, 0.0, 9.81])
    expected = np.column_stack([np.full(40, 48.0), np.full(40, 48.0 * np.cos(60 * DEG)),
                                np.full(40, -48.0 * np.sin(60 * DEG))])
    walk = np.repeat([0, 1], 20)
    groups = {"trajectory": walk, "time": np.tile(np.arange(20) / 10.0, 2)}
    names = ("acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z", "mag_x", "mag_y", "mag_z")
    acc = grav + np.random.default_rng(6).normal(0, 1.0, grav.shape)          # walking: noisy accelerometer
    t = SampleTable(np.column_stack([acc, grav, mag]), np.zeros((40, 2)), groups=groups,
                    meta={"modality": "imu", "channels": names, "rate_hz": 10.0})
    out = MagneticFeatures().fit_transform(t)                                  # gravity channels win
    assert out.meta["modality"] == "magnetic" and out.meta["units"] == "uT" and "channels" not in out.meta
    assert out.meta["feature_names"] == ("B", "B_h", "B_v") and out.meta["rate_hz"] == 10.0
    np.testing.assert_allclose(out.X, expected, atol=1e-9)
    assert np.array_equal(out.groups["trajectory"], walk)
    assert MagneticFeatures().transform(out) is out                            # already magnetic
    # accelerometer only: its moving average per walk gives the up direction (static device: exact)
    still = SampleTable(np.column_stack([np.tile(grav[0], (40, 1)), np.tile(mag[0], (40, 1))]), np.zeros((40, 2)),
                        groups=groups, meta={"modality": "imu", "channels": names[:3] + names[6:], "rate_hz": 10.0})
    np.testing.assert_allclose(MagneticFeatures(smoothing=1.0).transform(still).X, expected, atol=1e-9)
    with pytest.raises(ValueError, match="rate_hz"):
        MagneticFeatures().transform(still.replace(meta={"modality": "imu", "channels": still.meta["channels"]}))
    # arrays: (N, 3) readings of a flat device, or (N, 6) readings with a gravity reading per row
    np.testing.assert_allclose(MagneticFeatures().transform(np.column_stack([mag, grav])), expected, atol=1e-9)
    assert MagneticFeatures().transform(mag[:1]).shape == (1, 3)
    with pytest.raises(ValueError, match="lacks"):
        MagneticFeatures().transform(t.replace(meta={"modality": "imu", "channels": names[:6]}))


# ---------------------------------------------------------------------------------- visible light
def test_lambertian_gain_closed_form():
    led = np.array([[0.0, 0.0, 3.0]])
    below = vlc.channel_gain([[0.0, 0.0, 0.5]], led, order=1.0, area=1e-4)
    assert below[0, 0] == pytest.approx(2 * 1e-4 / (2 * np.pi * 2.5 ** 2), rel=1e-14)   # (m+1) A / (2 pi h^2)
    p = np.array([[1.0, 2.0, 0.5]])
    d = np.sqrt(1 + 4 + 2.5 ** 2)
    cos = 2.5 / d                                                                  # phi = psi for vertical axes
    for m in (0.5, 1.0, 2.0, 6.6):
        H = vlc.channel_gain(p, led, order=m, area=2e-4, filter_gain=0.9, concentrator_gain=1.7)
        assert H[0, 0] == pytest.approx((m + 1) * 2e-4 / (2 * np.pi * d * d) * cos ** m * 0.9 * 1.7 * cos, rel=1e-13)
    psi = np.arccos(cos)                                                          # 41.8 degrees
    assert vlc.channel_gain(p, led, fov=psi - 1e-6)[0, 0] == 0.0 < vlc.channel_gain(p, led, fov=psi + 1e-6)[0, 0]
    assert vlc.channel_gain([[0.0, 0.0, 3.5]], led)[0, 0] == 0.0                  # above the LED: behind it
    assert vlc.lambertian_order(60 * DEG) == pytest.approx(1.0)
    assert vlc.half_power_angle(vlc.lambertian_order(15 * DEG)) == pytest.approx(15 * DEG)
    assert vlc.lambertian_order(15 * DEG) == pytest.approx(-np.log(2) / np.log(np.cos(15 * DEG)))  # 20.0
    assert vlc.concentrator_gain(1.5, 70 * DEG) == pytest.approx(1.5 ** 2 / np.sin(70 * DEG) ** 2)
    with pytest.raises(ValueError, match="half-power"):
        vlc.lambertian_order(95 * DEG)


def test_tilted_receiver_and_led_use_the_true_angles():
    led = np.array([[2.0, 1.0, 3.0]])
    n_led = np.array([0.3, 0.0, -1.0]) / np.linalg.norm([0.3, 0.0, -1.0])
    n_rx = np.array([0.0, 0.2, 1.0]) / np.linalg.norm([0.0, 0.2, 1.0])
    p = np.array([[0.5, 0.0, 1.0]])
    v = p[0] - led[0]
    d = np.linalg.norm(v)
    cos_phi, cos_psi = v @ n_led / d, -v @ n_rx / d
    H = vlc.channel_gain(p, led, order=2.0, led_normals=n_led, receiver_normals=n_rx)
    assert H[0, 0] == pytest.approx(3 * 1e-4 / (2 * np.pi * d * d) * cos_phi ** 2 * cos_psi, rel=1e-13)


def test_power_distance_round_trip():
    heights = np.array([2.0, 2.5, 1.8])
    d = np.array([[2.0, 3.0, 4.0], [2.5, 2.6, 5.0]])                                # (N, A), d >= h
    for m in (1.0, 3.0):
        P = vlc.distance_to_power(d, heights, tx_power=1.5, order=m, area=1e-4, concentrator_gain=2.0)
        np.testing.assert_allclose(vlc.power_to_distance(P, heights, tx_power=1.5, order=m, concentrator_gain=2.0),
                                   d, rtol=1e-13)
        r = vlc.power_to_distance(P, heights, tx_power=1.5, order=m, concentrator_gain=2.0, horizontal=True)
        np.testing.assert_allclose(r, np.sqrt(np.maximum(d * d - heights * heights, 0)), atol=1e-7)
    # the vertical-link formula is the channel gain of an LED facing down and a receiver facing up
    leds = np.array([[0.0, 0.0, 3.0], [4.0, 0.0, 3.2]])
    pts = np.array([[1.0, 1.5, 0.8], [3.0, -1.0, 0.8]])
    P = vlc.received_power(pts, leds, tx_power=[2.0, 3.0], order=1.5)
    dist = np.linalg.norm(pts[:, None] - leds[None], axis=2)
    np.testing.assert_allclose(vlc.power_to_distance(P, leds[:, 2] - 0.8, tx_power=[2.0, 3.0], order=1.5), dist,
                               rtol=1e-13)
    out = vlc.power_to_distance(np.array([0.0, np.nan]), 2.0)
    assert np.isinf(out[0]) and np.isnan(out[1])
    with pytest.raises(ValueError, match="above"):
        vlc.power_to_distance(1e-6, -1.0)


def test_receiver_noise_model_of_komine_and_nakagawa():
    q, k = 1.602176634e-19, 1.380649e-23
    B, R, Ibg, A, T, eta = 100e6, 0.54, 5100e-6, 1e-4, 295.0, 1.12e-6

    def dark(B):                               # background shot noise + feedback-resistor and FET thermal noise
        return (2 * q * Ibg * 0.562 * B + 8 * np.pi * k * T * eta * A * 0.562 * B ** 2 / 10.0
                + 16 * np.pi ** 2 * k * T * 1.5 * eta ** 2 * A ** 2 * 0.0868 * B ** 3 / 30e-3)

    at_zero = dark(B)
    assert vlc.noise_variance(0.0) == pytest.approx(at_zero, rel=1e-12)
    P = np.array([1e-6, 1e-4, 1e-3])
    np.testing.assert_allclose(vlc.noise_variance(P) - at_zero, 2 * q * R * P * B, rtol=1e-9)  # signal shot noise
    assert vlc.noise_variance(-1.0) == vlc.noise_variance(0.0)
    assert vlc.noise_variance(0.0, bandwidth=1e6) == pytest.approx(dark(1e6), rel=1e-12)       # averaging helps
    assert dark(1e6) < 0.01 * at_zero


# --------------------------------------------------------------------------- ultrasound, two-way
def test_speed_of_sound_and_ultrasound_ranging():
    assert ranging.speed_of_sound(0.0) == pytest.approx(331.3)
    assert ranging.speed_of_sound(20.0) == pytest.approx(331.3 * np.sqrt(293.15 / 273.15))       # 343.2 m/s
    np.testing.assert_allclose(ranging.speed_of_sound([-10.0, 30.0]), 331.3 * np.sqrt(1 + np.array([-10, 30]) / 273.15))
    assert ranging.ultrasound_tof_to_distance(0.01, 20.0) == pytest.approx(3.43214622682568)
    # 1 degC error: 1 / (2 T) = 0.17 % of the range at 20 degC
    assert ranging.speed_of_sound(21.0) / ranging.speed_of_sound(20.0) - 1 == pytest.approx(1 / (2 * 293.65), rel=1e-3)
    exact, cricket = ranging.rf_ultrasound_distance(0.01), ranging.rf_ultrasound_distance(0.01, exact=False)
    c = ranging.speed_of_sound(20.0)
    assert exact == pytest.approx(0.01 / (1 / c - 1 / ranging.SPEED_OF_LIGHT), rel=1e-14)
    assert (exact - cricket) / cricket == pytest.approx(c / ranging.SPEED_OF_LIGHT, rel=1e-5)   # about 1e-6
    with pytest.raises(ValueError, match="absolute zero"):
        ranging.speed_of_sound(-300.0)


def test_two_way_ranging_with_and_without_clock_drift():
    tau, reply_b, reply_a = 30e-9, 800e-6, 300e-6                 # 9 m; asymmetric reply times
    for e_a, e_b in ((0.0, 0.0), (20e-6, -15e-6)):              # relative clock errors of initiator/responder
        round1, reply1 = (2 * tau + reply_b) * (1 + e_a), reply_b * (1 + e_b)
        round2, reply2 = (2 * tau + reply_a) * (1 + e_b), reply_a * (1 + e_a)
        ss = ranging.twr_tof(round1, reply1)
        ds = ranging.ds_twr_tof(round1, reply1, round2, reply2)
        assert ss - tau == pytest.approx(tau * e_a + reply_b * (e_a - e_b) / 2, abs=1e-18)
        assert ds - tau == pytest.approx(tau * (e_a + e_b) / 2, rel=1e-3, abs=1e-18)   # first order: drift-tolerant
    assert ranging.SPEED_OF_LIGHT * (ss - tau) > 4.0                         # 35 ppm on 800 us: 4.2 m
    assert abs(ranging.SPEED_OF_LIGHT * (ds - tau)) < 1e-4
    assert ranging.twr_distance(2 * tau + reply_b, reply_b) == pytest.approx(ranging.SPEED_OF_LIGHT * tau)
    speed = ranging.speed_of_sound(20.0)                                     # ultrasound two-way ranging
    t = 3.0 / speed
    assert ranging.ds_twr_distance(2 * t + 0.01, 0.01, 2 * t + 0.02, 0.02, speed=speed) == pytest.approx(3.0)
