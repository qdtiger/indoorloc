from __future__ import annotations

import numpy as np
import pytest

from indoorloc.evaluation.bounds import (aoa_crlb, crlb_covariance, crlb_rmse, dop, gdop, rss_crlb, tdoa_crlb,
                                         tdoa_fim, toa_crlb, toa_fim)


def _circle(n, radius=10.0, center=(0.0, 0.0)):
    angle = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(angle), np.sin(angle)]) * radius + np.asarray(center)


@pytest.mark.parametrize("n", [3, 4, 7, 12])
def test_toa_and_gdop_for_anchors_evenly_spaced_on_a_circle(n):
    # sum u u^T = (n / 2) I  =>  RMSE bound 2 sigma / sqrt(n), GDOP 2 / sqrt(n)
    anchors = _circle(n)
    assert toa_crlb(anchors, [0.0, 0.0], 0.3) == pytest.approx(2 * 0.3 / np.sqrt(n), rel=1e-12)
    assert gdop(anchors, [0.0, 0.0]) == pytest.approx(2 / np.sqrt(n), rel=1e-12)
    assert gdop(_circle(4), [0.0, 0.0]) == pytest.approx(1.0)  # the textbook square


def test_equal_noise_toa_bound_is_sigma_times_gdop_everywhere():
    rng = np.random.default_rng(0)
    anchors = rng.uniform(-20, 20, (6, 3))
    points = rng.uniform(-5, 5, (50, 3))
    assert np.allclose(toa_crlb(anchors, points, 0.7), 0.7 * gdop(anchors, points), rtol=1e-10)
    per_anchor = toa_crlb(anchors, points, np.full(6, 1.4))
    assert np.allclose(per_anchor, 2 * toa_crlb(anchors, points, 0.7), rtol=1e-12)  # bound scales with sigma


def test_rss_bounds_match_patwari_closed_forms():
    sigma, n_p = 4.0, 2.5
    anchors = _circle(7, radius=10.0)
    expected = 2 * 10.0 * sigma * np.log(10) / (10 * n_p * np.sqrt(7))
    assert rss_crlb(anchors, [0.0, 0.0], sigma, n_p) == pytest.approx(expected, rel=1e-12)
    # one anchor, one dimension: std(d_hat) >= d sigma ln(10) / (10 n_p)  (Patwari et al. 2005)
    assert rss_crlb([[0.0]], [7.0], sigma, 2.0) == pytest.approx(7.0 * sigma * np.log(10) / 20.0, rel=1e-12)
    # RSS gets worse with distance, ToA does not: doubling the geometry doubles the RSS bound
    assert rss_crlb(2 * anchors, [0.0, 0.0], sigma, n_p) == pytest.approx(2 * expected, rel=1e-12)


def test_aoa_bound_for_two_perpendicular_bearings():
    # both anchors at distance d, bearings 90 degrees apart: J = I / (sigma d)^2 -> RMSE sqrt(2) sigma d
    assert aoa_crlb([[5.0, 0.0], [0.0, 5.0]], [0.0, 0.0], 0.01) == pytest.approx(np.sqrt(2) * 0.01 * 5.0)
    with pytest.raises(ValueError, match="2-D"):
        aoa_crlb(np.zeros((2, 3)), np.ones(3), 0.1)


def test_tdoa_bound_equals_toa_with_an_unknown_clock_offset():
    # Differencing against a reference removes a common offset without losing information:
    # the TDoA FIM equals the Schur complement of the pseudorange FIM with a clock-bias column.
    rng = np.random.default_rng(1)
    anchors = rng.uniform(-10, 10, (5, 2))
    points = rng.uniform(-3, 3, (8, 2))
    sigma = np.array([0.2, 0.3, 0.1, 0.5, 0.4])
    u = points[:, None, :] - anchors[None]
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    H = np.concatenate([u, np.ones(u.shape[:2] + (1,))], axis=2) / sigma[None, :, None]
    position_block = np.linalg.inv(np.einsum("pai,paj->pij", H, H))[:, :2, :2]
    for reference in range(5):
        assert np.allclose(np.linalg.inv(tdoa_fim(anchors, points, sigma, reference=reference)), position_block,
                           rtol=1e-9, atol=1e-12)
    assert np.allclose(tdoa_crlb(anchors, points, sigma), np.sqrt(np.trace(position_block, axis1=1, axis2=2)))


def test_dop_components_for_symmetric_layouts():
    octahedron = np.vstack([np.eye(3), -np.eye(3)])  # H^T H = 2 I
    d = dop(octahedron, [0.0, 0.0, 0.0])
    assert d["pdop"] == pytest.approx(np.sqrt(1.5)) and d["hdop"] == pytest.approx(1.0)
    assert d["vdop"] == pytest.approx(np.sqrt(0.5)) and d["gdop"] == d["pdop"]
    square = dop(_circle(4, radius=1.0), [0.0, 0.0], clock_bias=True)  # sum u = 0: blocks decouple
    assert square["tdop"] == pytest.approx(0.5) and square["hdop"] == pytest.approx(1.0)
    assert square["gdop"] == pytest.approx(np.sqrt(1.25))


def test_degenerate_geometry_is_inf_and_a_point_on_an_anchor_is_nan():
    anchors = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    bound = toa_crlb(anchors, [[5.0, 0.0], [0.0, 0.0], [1.0, 1.0]], 1.0)
    assert np.isinf(bound[0]) and np.isnan(bound[1]) and np.isfinite(bound[2])
    assert np.isinf(crlb_covariance(toa_fim(anchors, [5.0, 0.0], 1.0))).all()
    assert isinstance(toa_crlb(anchors, [1.0, 1.0], 1.0), float)
    assert crlb_rmse(np.diag([4.0, 1.0])) == pytest.approx(np.sqrt(0.25 + 1.0))
    with pytest.raises(ValueError, match="positive"):
        toa_crlb(anchors, [1.0, 1.0], 0.0)
    with pytest.raises(ValueError, match="same D"):
        toa_crlb(anchors, [1.0, 1.0, 1.0], 1.0)


def test_maximum_likelihood_ranging_attains_the_toa_bound():
    """Monte Carlo: at small noise the ML (Gauss-Newton least-squares) estimate is efficient,
    so its empirical RMSE must match the CRLB; a wrong Fisher information would not."""
    rng = np.random.default_rng(2)
    anchors = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 15.0], [20.0, 15.0], [10.0, -5.0]])
    truth = np.array([6.0, 4.0])
    sigma = np.array([0.05, 0.1, 0.05, 0.2, 0.1])
    trials = 4000
    ranges = np.linalg.norm(truth - anchors, axis=1) + rng.normal(size=(trials, 5)) * sigma
    est = np.tile(truth + 0.5, (trials, 1))
    w = 1.0 / sigma ** 2
    for _ in range(10):  # weighted Gauss-Newton, vectorised over trials
        diff = est[:, None, :] - anchors[None]
        dist = np.linalg.norm(diff, axis=2)
        J = diff / dist[..., None]
        r = ranges - dist
        A = np.einsum("a,tai,taj->tij", w, J, J)
        b = np.einsum("a,tai,ta->ti", w, J, r)
        est += np.linalg.solve(A, b[..., None])[..., 0]
    rmse = np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1)))
    assert rmse == pytest.approx(toa_crlb(anchors, truth, sigma), rel=0.05)


def _numerical_fim(h, point, cov, step=1e-6):
    """J = G^T C^-1 G with G the central-difference Jacobian of the noise-free measurements h(p):
    the Gaussian Fisher information (Kay 1993, eq. 3.31), computed without any closed form."""
    point = np.asarray(point, dtype=np.float64)
    G = np.column_stack([(h(point + step * e) - h(point - step * e)) / (2 * step) for e in np.eye(len(point))])
    return G.T @ np.linalg.solve(cov, G)


def test_every_fisher_information_matches_a_finite_difference_jacobian():
    """Checks the derivative of each measurement model (signs, 1/d factors, ln 10, the shared
    reference noise of TDoA) independently of the closed forms in the module."""
    from indoorloc.evaluation.bounds import aoa_fim, rss_fim

    rng = np.random.default_rng(3)
    anchors = rng.uniform(-10, 10, (5, 2))
    sigma = rng.uniform(0.1, 0.5, 5)
    sigma_db = rng.uniform(2.0, 6.0, 5)
    n_p = rng.uniform(1.8, 3.5, 5)
    dist = lambda p: np.linalg.norm(p - anchors, axis=1)  # noqa: E731
    for point in rng.uniform(-4, 4, (6, 2)):
        np.testing.assert_allclose(toa_fim(anchors, point, sigma),
                                   _numerical_fim(dist, point, np.diag(sigma ** 2)), rtol=1e-6)
        rss = lambda p: -30.0 - 10.0 * n_p * np.log10(dist(p))  # noqa: E731  (P0 = -30 dBm drops out)
        np.testing.assert_allclose(rss_fim(anchors, point, sigma_db, n_p),
                                   _numerical_fim(rss, point, np.diag(sigma_db ** 2)), rtol=1e-6)
        bearing = lambda p: np.arctan2(p[1] - anchors[:, 1], p[0] - anchors[:, 0])  # noqa: E731
        np.testing.assert_allclose(aoa_fim(anchors, point, sigma),
                                   _numerical_fim(bearing, point, np.diag(sigma ** 2)), rtol=1e-6)
        for ref in (0, 3):
            others = np.delete(np.arange(5), ref)
            diff = lambda p: dist(p)[others] - dist(p)[ref]  # noqa: E731
            cov = np.diag(sigma[others] ** 2) + sigma[ref] ** 2  # the reference's noise is common to all
            np.testing.assert_allclose(tdoa_fim(anchors, point, sigma, reference=ref),
                                       _numerical_fim(diff, point, cov), rtol=1e-6)
