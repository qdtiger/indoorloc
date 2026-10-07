"""L2 DeviceCalibration: offset, linear and quantile maps, paired and unpaired, known results."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import NotFittedError, SampleTable, load_model
from indoorloc.signals import DeviceCalibration, WiFiSignal

NAN = np.nan


def _reference(n=400, f=20, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.integers(-100, -30, size=(n, f)).astype(np.float64)
    R[rng.random(R.shape) < 0.2] = NAN
    return R


def test_paired_linear_recovers_the_device_line_exactly():
    R = _reference()
    X = (R + 12.0) / 0.8  # the new device reads x with 0.8 x - 12 = reference
    cal = DeviceCalibration("linear", paired=True).fit(X, reference=R)
    assert cal.coef_ == pytest.approx(0.8, abs=1e-12) and cal.intercept_ == pytest.approx(-12.0, abs=1e-9)
    np.testing.assert_allclose(cal(X), R, atol=1e-9)
    assert np.array_equal(np.isnan(cal(X)), np.isnan(X))  # not heard stays not heard


def test_unpaired_fits_match_the_distributions_not_the_rows():
    R = _reference()
    rng = np.random.default_rng(1)
    X = rng.permutation((R - 7.0).reshape(-1)).reshape(R.shape)  # same values, unrelated rows
    offset = DeviceCalibration("offset").fit(X, reference=R)
    assert offset.coef_ == 1.0 and offset.intercept_ == pytest.approx(7.0, abs=1e-9)
    # a linear relation between the distributions is recovered from matched quantiles
    Xl = rng.permutation(((R + 12.0) / 0.8).reshape(-1)).reshape(R.shape)
    lin = DeviceCalibration("linear", n_quantiles=51).fit(Xl, reference=R)
    assert lin.coef_ == pytest.approx(0.8, abs=1e-9) and lin.intercept_ == pytest.approx(-12.0, abs=1e-7)
    with pytest.raises(ValueError, match="same shape"):
        DeviceCalibration(paired=True).fit(X[:10], reference=R)


def test_quantile_map_inverts_a_monotone_nonlinear_distortion():
    rng = np.random.default_rng(2)
    ref = np.sort(rng.uniform(-100.0, -30.0, size=301))
    distort = lambda v: -100.0 + 70.0 * ((v + 100.0) / 70.0) ** 1.7  # noqa: E731, monotone on [-100, -30]
    dev = rng.permutation(distort(ref))
    cal = DeviceCalibration("quantile", n_quantiles=301).fit(dev[:, None], reference=ref[:, None])
    # with one knot per sample the knots are the order statistics, so training values map exactly
    np.testing.assert_allclose(cal(distort(ref)[:, None])[:, 0], ref, atol=1e-9)
    # between knots: piecewise-linear interpolation of a smooth map
    grid = np.linspace(-95.0, -35.0, 50)
    assert np.max(np.abs(cal(distort(grid)[:, None])[:, 0] - grid)) < 0.5
    # beyond the knots: slope 1 continuation, no clipping
    lo, hi = cal.quantiles_[0], cal.quantiles_[-1]
    assert cal(np.array([[lo - 5.0]]))[0, 0] == pytest.approx(cal.reference_quantiles_[0] - 5.0)
    assert cal(np.array([[hi + 3.0]]))[0, 0] == pytest.approx(cal.reference_quantiles_[-1] + 3.0)


def test_quantile_knots_collapse_integer_ties():
    X = np.array([[-80.0], [-80.0], [-60.0], [-60.0]])
    R = np.array([[-85.0], [-75.0], [-55.0], [-45.0]])
    cal = DeviceCalibration("quantile", n_quantiles=4).fit(X, reference=R)
    assert cal.quantiles_.tolist() == [-80.0, -60.0] and cal.reference_quantiles_.tolist() == [-80.0, -50.0]
    assert cal(np.array([[-70.0]]))[0, 0] == -65.0


def test_containers_dtype_and_errors(tmp_path):
    R = _reference(50, 4)
    X = R.astype(np.float32) - 5
    cal = DeviceCalibration("offset", paired=True).fit(SampleTable(X, np.zeros((50, 2))),
                                                      reference=SampleTable(R, np.zeros((50, 2))))
    assert cal(X).dtype == np.float32 and cal.intercept_ == pytest.approx(5.0)
    view = cal(WiFiSignal(X[0], ("a", "b", "c", "d")))
    assert isinstance(view, WiFiSignal) and np.allclose(view.rssi, R[0], equal_nan=True)
    loaded = load_model(DeviceCalibration("quantile").fit(X, reference=R).save(tmp_path / "c"))
    assert loaded.quantiles_.dtype == np.float64 and np.allclose(loaded(X), R, atol=1e-5, equal_nan=True)
    with pytest.raises(TypeError, match="reference"):
        DeviceCalibration().fit(X)
    with pytest.raises(NotFittedError):
        DeviceCalibration().transform(X)
    with pytest.raises(ValueError, match="method"):
        DeviceCalibration("affine").fit(X, reference=R)
    with pytest.raises(ValueError, match="not positive"):
        DeviceCalibration("linear", paired=True).fit(-R, reference=R)
    with pytest.raises(ValueError, match="at least 2"):
        DeviceCalibration("linear").fit(np.array([[NAN, -50.0]]), reference=R)
