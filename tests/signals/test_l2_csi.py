"""L2 CSI: amplitude, phase sanitization (Sen et al. 2012, SpotFi), subcarrier selection, antenna products."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import SampleTable
from indoorloc.signals import CSIAmplitude, CSIPhaseSanitize, SubcarrierSelect, WiFiSignal, csi

K = csi.INTEL5300_SUBCARRIERS_20MHZ.astype(np.float64)
NAN = np.nan


def _residual_phase(n_chains, seed=0):
    """Smooth phases with zero mean and no linear trend in K (so a line fit leaves them intact)."""
    rng = np.random.default_rng(seed)
    ph = 0.3 * np.sin(np.outer(rng.uniform(0.1, 0.3, n_chains), K) + rng.uniform(0, 6, (n_chains, 1)))
    kc = K - K.mean()
    ph -= ph.mean(axis=-1, keepdims=True)
    return ph - np.outer(ph @ kc / (kc @ kc), kc)


def _csi(n=5, seed=0):
    """(n, 3 rx, 1 tx, 30) CSI: residual phase + per-antenna AoA phase + per-packet slope and offset."""
    rng = np.random.default_rng(seed)
    true = _residual_phase(n * 3, seed).reshape(n, 3, 1, 30)
    aoa = np.array([0.0, 0.9, -1.3])[None, :, None, None]
    slope = rng.uniform(-0.1, 0.1, (n, 1, 1, 1))  # sampling time offset: common to all antennas
    offset = rng.uniform(-np.pi, np.pi, (n, 1, 1, 1))
    amp = rng.uniform(0.5, 2.0, true.shape)
    H = (amp * np.exp(1j * (true + aoa + slope * K + offset))).astype(np.complex64)
    return H, true, aoa, amp


def test_per_chain_lstsq_recovers_the_residual_phase():
    H, true, _, amp = _csi()
    t = SampleTable(H, np.zeros((5, 2)), meta={"modality": "csi", "subcarriers": K})
    out = CSIPhaseSanitize(output="phase")(t)
    assert out.meta["modality"] == "csi_phase" and out.meta["units"] == "rad" and out.X.dtype == np.float32
    np.testing.assert_allclose(out.X, true, atol=2e-5)  # float32 CSI: ~1e-6 rad phase resolution
    cplx = CSIPhaseSanitize()(t)  # complex output keeps |H| and the modality
    assert cplx.meta["modality"] == "csi" and cplx.X.dtype == np.complex64
    np.testing.assert_allclose(np.abs(cplx.X), amp, rtol=1e-6)
    np.testing.assert_allclose(np.angle(cplx.X * np.exp(-1j * true)), 0, atol=2e-5)


def test_joint_fit_keeps_the_inter_antenna_phase_differences_spotfi():
    H, true, aoa, _ = _csi()
    joint = CSIPhaseSanitize(K, joint=True, output="phase")(H)
    diff = np.angle(np.exp(1j * (joint[:, 1] - joint[:, 0])))
    want = np.angle(np.exp(1j * (true[:, 1] - true[:, 0] + aoa[:, 1] - aoa[:, 0])))
    np.testing.assert_allclose(diff, want, atol=2e-5)  # AoA information survives
    shifted = np.angle(H).astype(np.float64)
    shifted[:, 2] += 2 * np.pi  # a chain's offset is only known modulo 2 pi
    np.testing.assert_allclose(CSIPhaseSanitize(K, joint=True, output="phase")(shifted), joint, atol=2e-5)
    per_chain = CSIPhaseSanitize(K, output="phase")(H)
    assert np.abs(np.mean(per_chain[:, 1] - per_chain[:, 0], axis=-1)).max() < 1e-5  # ... not per chain


def test_endpoints_method_is_sen_et_al_formula():
    ph = np.unwrap(np.random.default_rng(3).uniform(-0.2, 0.2, 30).cumsum())
    a = (ph[-1] - ph[0]) / (K[-1] - K[0])
    b = ph.mean()
    np.testing.assert_allclose(csi.sanitize_phase(ph, K, method="endpoints"), ph - a * K - b, atol=1e-12)
    # a pure line is removed exactly by either method
    line = 0.37 * K - 1.1
    for method in ("lstsq", "endpoints"):
        residual = csi.sanitize_phase(line, K, method=method)
        assert np.abs(residual - residual.mean()).max() < 1e-12
    assert np.abs(csi.sanitize_phase(line, K)).max() < 1e-12


def test_sanitize_input_rules():
    H, *_ = _csi(2)
    with pytest.raises(ValueError, match="subcarrier indices"):
        CSIPhaseSanitize(subcarriers=np.arange(10))(H)
    with pytest.raises(ValueError, match="output='phase'"):
        CSIPhaseSanitize()(np.angle(H))
    assert CSIPhaseSanitize(K, output="phase")(np.angle(H).astype(np.float64)).dtype == np.float64
    with pytest.raises(TypeError, match="RSSI"):
        CSIPhaseSanitize()(WiFiSignal(np.zeros(30)))
    with pytest.raises(ValueError, match="method"):
        csi.sanitize_phase(np.zeros(30), method="median")


def test_amplitude_db_and_modality():
    H = np.array([[[[3 + 4j, 0j, 10 + 0j]]]], dtype=np.complex64)
    t = SampleTable(H, np.zeros((1, 2)), meta={"modality": "csi", "units": "raw"})
    lin = CSIAmplitude()(t)
    assert lin.meta["modality"] == "csi_amp" and lin.meta["units"] == "raw" and lin.X.dtype == np.float32
    assert lin.X.ravel().tolist() == [5.0, 0.0, 10.0]
    db = CSIAmplitude(db=True)(t)
    assert db.meta["units"] == "dB" and np.array_equal(db.X.ravel(), [20 * np.log10(np.float32(5)), NAN, 20],
                                                       equal_nan=True)
    assert CSIAmplitude()(H).shape == H.shape and CSIAmplitude(db=True)(np.array([10.0]))[0] == 20.0


def test_subcarrier_select_keeps_meta_aligned():
    H, *_ = _csi(2)
    t = SampleTable(H, np.zeros((2, 2)), meta={"modality": "csi", "subcarriers": K,
                                               "feature_names": tuple(f"sc{int(k)}" for k in K)})
    out = SubcarrierSelect([0, 15, -1])(t)
    assert out.X.shape == (2, 3, 1, 3) and np.array_equal(out.X, H[..., [0, 15, 29]])
    assert out.meta["subcarriers"].tolist() == [-28, 1, 28] and out.meta["feature_names"] == ("sc-28", "sc1", "sc28")
    assert SubcarrierSelect(np.arange(0, 30, 2))(np.abs(H)).shape == (2, 3, 1, 15)
    with pytest.raises(ValueError, match="positions"):
        SubcarrierSelect([30])(H)


def test_conjugate_multiplication_and_ratio_cancel_common_offsets():
    H, *_ = _csi(4)
    rng = np.random.default_rng(5)
    common = np.exp(1j * rng.uniform(-np.pi, np.pi, (4, 1, 1, 30)))  # CFO/SFO: same on every antenna
    gain = rng.uniform(0.5, 2.0, (4, 1, 1, 1))  # AGC: same on every antenna
    Hc = H.astype(np.complex128)
    np.testing.assert_allclose(csi.conjugate_multiply(Hc * common), csi.conjugate_multiply(Hc), atol=1e-12)
    np.testing.assert_allclose(csi.csi_ratio(Hc * common * gain), csi.csi_ratio(Hc), atol=1e-12)
    assert np.allclose(csi.csi_ratio(Hc)[:, 0], 1.0)
    zero = Hc.copy()
    zero[0, 0, 0, 0] = 0
    assert np.isnan(csi.csi_ratio(zero)[0, 1, 0, 0])


def test_joint_sanitization_and_conjugate_product_recover_the_aoa_of_a_ula():
    """Textbook narrowband ULA model (as in SpotFi): with half-wavelength spacing, antenna m of a
    single path from angle theta sees an extra phase -pi m sin(theta). The direct path's ToF, a random
    per-packet sampling time offset and a random common phase all add the same linear-in-k phase on
    every antenna, so joint (SpotFi) sanitization and the conjugate product keep -pi m sin(theta).
    theta = 20 deg keeps -2 pi sin(theta) (antenna 2) inside (-pi, pi), i.e. unambiguous."""
    theta = np.deg2rad(20.0)
    fc, df, c = 5.32e9, 312.5e3, 299_792_458.0  # channel 64 and the 802.11n subcarrier spacing
    d = c / fc / 2
    rng = np.random.default_rng(7)
    n, m = 20, np.arange(3)
    tau = 25.0 / c + rng.uniform(0, 50e-9, (n, 1, 1))  # ToF of the path + sampling time offset per packet
    beta = rng.uniform(-np.pi, np.pi, (n, 1, 1))
    f = fc + K * df
    ph = -2 * np.pi * f * (tau + m[None, :, None] * d * np.sin(theta) / c) + beta  # exact wideband phase
    H = (rng.uniform(0.5, 1.5, (n, 3, 30)) * np.exp(1j * ph)).astype(np.complex64)[:, :, None, :]
    out = CSIPhaseSanitize(K, joint=True, output="phase")(H)[:, :, 0]
    for mm in (1, 2):
        step = np.angle(np.mean(np.exp(1j * (out[:, mm] - out[:, 0]))))  # mean over packets and subcarriers
        assert np.rad2deg(np.arcsin(-step / (np.pi * mm))) == pytest.approx(20.0, abs=0.01)
        cm = np.angle(np.mean(csi.conjugate_multiply(H)[:, mm]))
        assert np.rad2deg(np.arcsin(-cm / (np.pi * mm))) == pytest.approx(20.0, abs=0.01)
    per_chain = CSIPhaseSanitize(K, output="phase")(H)[:, :, 0]  # per-chain fits erase the AoA
    assert abs(np.angle(np.mean(np.exp(1j * (per_chain[:, 1] - per_chain[:, 0]))))) < 1e-3


def test_amplitude_of_real_input_is_kept_not_rectified():
    db = SampleTable(np.array([[-1.8, 32.0]], np.float32), np.zeros((1, 2)),
                     meta={"modality": "csi_amp", "units": "dB"})
    out = CSIAmplitude()(db)
    assert out.X.tolist() == db.X.tolist() and out.meta["units"] == "dB"  # a negative dB value keeps its sign
    with pytest.raises(ValueError, match="already in dB"):
        CSIAmplitude(db=True)(db)
    with pytest.raises(ValueError, match="negative values"):
        CSIAmplitude(db=True)(np.array([[-1.8, 32.0]]))
    assert CSIAmplitude(db=True)(np.array([[1.0, 100.0]])).tolist() == [[0.0, 40.0]]  # linear -> dB


def test_subcarrier_select_accepts_a_boolean_mask():
    H, *_ = _csi(2)
    mask = np.zeros(30, bool)
    mask[[0, 15, 29]] = True
    assert np.array_equal(SubcarrierSelect(mask)(H), H[..., [0, 15, 29]])
    with pytest.raises(ValueError, match="one entry per subcarrier"):
        SubcarrierSelect(mask[:10])(H)
    with pytest.raises(TypeError, match="integer positions"):
        SubcarrierSelect([0.5, 2.0])(H)
    with pytest.raises(ValueError, match="non-empty"):
        SubcarrierSelect(np.zeros(30, bool))(H)
