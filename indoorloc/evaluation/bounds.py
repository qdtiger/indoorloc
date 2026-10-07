"""L4 performance bounds: Cramér-Rao lower bounds (CRLB) and dilution of precision (DOP).

A bound answers "how well could any unbiased estimator do with this anchor geometry and
this measurement noise?", so a method's error can be read against the physics of the
setup instead of only against other methods. Every function is vectorised over points:
``points`` is ``(P, D)`` (or one ``(D,)`` point) and ``anchors`` is ``(A, D)`` in the same
frame (``SampleTable.meta["anchors"]``); results are ``(P,)`` (or a scalar for one point).

Measurement models (``u_i = (p - a_i) / d_i`` is the unit vector from anchor ``i`` to the
point, ``d_i`` the distance), each with independent Gaussian noise of standard deviation
``sigma`` (a scalar or one value per anchor):

* ToA: ranges ``r_i = d_i + n_i`` [m]; ``J = sum_i u_i u_i^T / sigma_i^2``.
* RSS (log-distance path loss): ``P_i = P0 - 10 n_p log10(d_i) + X_i`` [dB];
  ``J = b sum_i u_i u_i^T / d_i^2`` with ``b = (10 n_p / (sigma ln 10))^2``.
* AoA (2-D): bearings ``theta_i + n_i`` [rad]; ``J = sum_i v_i v_i^T / (sigma_i d_i)^2`` with
  ``v_i = (-u_iy, u_ix)``.
* TDoA: range differences ``d_i - d_ref`` built from ToA noise [m]; ``J = H^T C^-1 H`` with
  ``H_i = u_i - u_ref`` and ``C = diag(sigma_i^2) + sigma_ref^2 1 1^T`` over ``i != ref``.

The CRLB on the covariance of an unbiased position estimate is ``J^-1``; the returned
scalar bound is the position RMSE ``sqrt(trace(J^-1))`` in coordinate units. It is ``inf``
where the geometry makes ``J`` singular (e.g. collinear anchors in 2-D) and ``nan`` for a
point that coincides with an anchor (the measurement is not differentiable there).

GDOP (Sharp et al. 2012) is the same geometry without the noise: with ``H`` the matrix of
unit vectors, ``G = (H^T H)^-1`` and ``GDOP = sqrt(trace G)``; for equal ToA noise
``toa_crlb == sigma * gdop``. With ``clock_bias=True`` a column of ones models an unknown
receiver clock offset (pseudoranges), as in GNSS.

Known results used in the tests: ``N >= 3`` anchors evenly spaced on a circle around the point
give ``sum u_i u_i^T = (N / 2) I``, hence ToA ``2 sigma / sqrt(N)``, GDOP ``2 / sqrt(N)`` and
RSS ``2 R sigma ln(10) / (10 n_p sqrt(N))``; one anchor at distance ``d`` bounds RSS ranging
by ``d sigma ln(10) / (10 n_p)`` (Patwari et al. 2005); the TDoA bound equals the ToA bound
with an unknown clock offset.

References
----------
S. M. Kay, "Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory",
Prentice Hall, 1993. ISBN 0-13-345711-7 (Fisher information of Gaussian models, Ch. 3).
N. Patwari, A. O. Hero, M. Perkins, N. S. Correal, R. J. O'Dea, "Relative location estimation
in wireless sensor networks", IEEE Transactions on Signal Processing 51(8):2137-2148, 2003.
DOI: 10.1109/TSP.2003.814469 (RSS and ToA CRLB).
N. Patwari, J. N. Ash, S. Kyperountas, A. O. Hero, R. L. Moses, N. S. Correal, "Locating the
nodes: cooperative localization in wireless sensor networks", IEEE Signal Processing Magazine
22(4):54-69, 2005. DOI: 10.1109/MSP.2005.1458287 (RSS ranging bound).
Y. T. Chan, K. C. Ho, "A simple and efficient estimator for hyperbolic location", IEEE
Transactions on Signal Processing 42(8):1905-1915, 1994. DOI: 10.1109/78.301830 (TDoA CRLB
with correlated differences).
M. Gavish, A. J. Weiss, "Performance analysis of bearing-only target location algorithms",
IEEE Transactions on Aerospace and Electronic Systems 28(3):817-828, 1992. DOI: 10.1109/7.256302
(AoA CRLB).
I. Sharp, K. Yu, M. Hedley, "On the GDOP and Accuracy for Indoor Positioning", IEEE Transactions
on Aerospace and Electronic Systems 48(3):2032-2051, 2012. DOI: 10.1109/TAES.2012.6237577
(GDOP for indoor anchor layouts).
"""
from __future__ import annotations

import numpy as np

_LN10 = np.log(10.0)


def _geometry(anchors, points):
    """Unit vectors (P, A, D), distances (P, A) and whether one point was given."""
    anchors = np.asarray(anchors, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    single = points.ndim == 1
    points = np.atleast_2d(points)
    if anchors.ndim != 2 or points.ndim != 2 or anchors.shape[1] != points.shape[1]:
        raise ValueError(f"anchors must be (A, D) and points (P, D) or (D,) with the same D; got "
                         f"{anchors.shape} and {np.shape(points)}")
    if len(anchors) == 0:
        raise ValueError("no anchors")
    diff = points[:, None, :] - anchors[None, :, :]
    dist = np.sqrt(np.einsum("pad,pad->pa", diff, diff))
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = diff / dist[..., None]  # NaN at an anchor: the bound is undefined there
    return unit, dist, single


def _per_anchor(value, n_anchors: int, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    v = np.full(n_anchors, float(v)) if v.ndim == 0 else v
    if v.shape != (n_anchors,):
        raise ValueError(f"{name} must be a scalar or one value per anchor ({n_anchors},), got {v.shape}")
    if not np.all(v > 0) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be positive and finite")
    return v


def _out(value, single: bool):
    return value[0] if single else value


def _fim(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """sum_i w_i v_i v_i^T for every point: (P, A, D), (P, A) or (A,) -> (P, D, D)."""
    return np.einsum("pa,pai,paj->pij", np.broadcast_to(weights, vectors.shape[:2]), vectors, vectors)


# --------------------------------------------------------------------------- Fisher information
def toa_fim(anchors, points, sigma) -> np.ndarray:
    """Fisher information of range (ToA / RTT / UWB) measurements, ``sigma`` in metres."""
    unit, _, single = _geometry(anchors, points)
    s = _per_anchor(sigma, unit.shape[1], "sigma")
    return _out(_fim(unit, 1.0 / s ** 2), single)


def rss_fim(anchors, points, sigma_db, path_loss_exponent) -> np.ndarray:
    """Fisher information of log-distance RSS measurements (Patwari et al. 2003).

    ``sigma_db`` is the shadowing standard deviation in dB and ``path_loss_exponent`` the
    ``n_p`` of ``P = P0 - 10 n_p log10(d)``; ``P0`` is known and drops out of the bound.
    """
    unit, dist, single = _geometry(anchors, points)
    s = _per_anchor(sigma_db, unit.shape[1], "sigma_db")
    n_p = _per_anchor(path_loss_exponent, unit.shape[1], "path_loss_exponent")
    b = (10.0 * n_p / (s * _LN10)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return _out(_fim(unit, b / dist ** 2), single)


def aoa_fim(anchors, points, sigma) -> np.ndarray:
    """Fisher information of 2-D bearing (angle-of-arrival) measurements, ``sigma`` in radians."""
    unit, dist, single = _geometry(anchors, points)
    if unit.shape[2] != 2:
        raise ValueError(f"aoa_fim models 2-D bearings; got {unit.shape[2]}-D coordinates")
    s = _per_anchor(sigma, unit.shape[1], "sigma")
    normal = np.stack([-unit[..., 1], unit[..., 0]], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _out(_fim(normal, 1.0 / (s ** 2 * dist ** 2)), single)


def tdoa_fim(anchors, points, sigma, *, reference: int = 0) -> np.ndarray:
    """Fisher information of range differences ``d_i - d_ref`` built from ToA noise ``sigma``.

    The differences share the reference anchor's noise, so their covariance is
    ``diag(sigma_i^2, i != ref) + sigma_ref^2 * 1 1^T`` (Chan & Ho 1994); it is inverted in
    closed form (Sherman-Morrison). Needs at least two anchors.
    """
    unit, _, single = _geometry(anchors, points)
    n_anchors = unit.shape[1]
    if n_anchors < 2:
        raise ValueError("TDoA needs at least two anchors")
    if not -n_anchors <= reference < n_anchors:
        raise ValueError(f"reference={reference} is not an anchor index")
    ref = reference % n_anchors
    s = _per_anchor(sigma, n_anchors, "sigma")
    others = np.delete(np.arange(n_anchors), ref)
    H = unit[:, others, :] - unit[:, ref:ref + 1, :]  # (P, A-1, D)
    lam_inv = 1.0 / s[others] ** 2  # C = diag(1 / lam_inv) + s_ref^2 1 1^T
    c = s[ref] ** 2 / (1.0 + s[ref] ** 2 * lam_inv.sum())
    W = np.diag(lam_inv) - c * np.outer(lam_inv, lam_inv)  # C^-1
    return _out(np.einsum("pai,ab,pbj->pij", H, W, H), single)


# --------------------------------------------------------------------------- bounds
def _trace_inverse(fim: np.ndarray) -> np.ndarray:
    """trace(J^-1) = sum of 1/eigenvalues of the symmetric FIM; inf if singular, nan if undefined."""
    J = np.asarray(fim, dtype=np.float64)
    single = J.ndim == 2
    J = J[None] if single else J
    out = np.full(len(J), np.nan)
    ok = np.all(np.isfinite(J), axis=(1, 2))
    if ok.any():
        eig = np.linalg.eigvalsh(J[ok])  # ascending
        tol = eig[:, -1:] * (J.shape[-1] * 1e3 * np.finfo(np.float64).eps)
        with np.errstate(divide="ignore"):
            tr = np.where(np.all(eig > tol, axis=1), np.sum(1.0 / np.where(eig > tol, eig, 1.0), axis=1), np.inf)
        out[ok] = tr
    return out[0] if single else out


def crlb_rmse(fim) -> np.ndarray | float:
    """Position RMSE bound ``sqrt(trace(J^-1))`` from a (D, D) or (P, D, D) Fisher information."""
    tr = _trace_inverse(fim)
    return float(np.sqrt(tr)) if np.ndim(tr) == 0 else np.sqrt(tr)


def crlb_covariance(fim) -> np.ndarray:
    """The covariance bound ``J^-1`` itself, (D, D) or (P, D, D); singular ``J`` gives ``inf`` entries."""
    J = np.asarray(fim, dtype=np.float64)
    single = J.ndim == 2
    J = J[None] if single else J
    out = np.full(J.shape, np.inf)
    finite = np.all(np.isfinite(J), axis=(1, 2))
    out[~finite] = np.nan
    ok = finite & np.isfinite(_trace_inverse(J))
    out[ok] = np.linalg.inv(J[ok])
    return out[0] if single else out


def toa_crlb(anchors, points, sigma) -> np.ndarray | float:
    """ToA/ranging position RMSE bound (metres); see :func:`toa_fim`."""
    return crlb_rmse(toa_fim(anchors, points, sigma))


def rss_crlb(anchors, points, sigma_db, path_loss_exponent) -> np.ndarray | float:
    """RSS (log-distance path loss) position RMSE bound; see :func:`rss_fim`."""
    return crlb_rmse(rss_fim(anchors, points, sigma_db, path_loss_exponent))


def aoa_crlb(anchors, points, sigma) -> np.ndarray | float:
    """2-D AoA position RMSE bound; see :func:`aoa_fim`."""
    return crlb_rmse(aoa_fim(anchors, points, sigma))


def tdoa_crlb(anchors, points, sigma, *, reference: int = 0) -> np.ndarray | float:
    """TDoA position RMSE bound; see :func:`tdoa_fim`."""
    return crlb_rmse(tdoa_fim(anchors, points, sigma, reference=reference))


# --------------------------------------------------------------------------- dilution of precision
def dop(anchors, points, *, clock_bias: bool = False) -> dict:
    """Dilution of precision of range measurements: ``{"gdop", "pdop", "hdop"[, "vdop"][, "tdop"]}``.

    ``H`` has one row per anchor, the unit vector ``u_i`` (and a trailing 1 with
    ``clock_bias``); ``G = (H^T H)^-1``. GDOP = sqrt(trace G), PDOP = sqrt of the position
    block's trace, HDOP = sqrt(G_xx + G_yy), VDOP = sqrt(G_zz) (3-D only), TDOP =
    sqrt(G_tt) in range units (``clock_bias`` only). Values are ``(P,)`` arrays (floats for
    one point), ``inf`` for degenerate geometry.
    """
    unit, _, single = _geometry(anchors, points)
    D = unit.shape[2]
    H = np.concatenate([unit, np.ones(unit.shape[:2] + (1,))], axis=2) if clock_bias else unit
    G = crlb_covariance(np.einsum("pai,paj->pij", H, H))
    diag = np.diagonal(G, axis1=1, axis2=2)
    out = {"gdop": np.sqrt(diag.sum(axis=1)), "pdop": np.sqrt(diag[:, :D].sum(axis=1)),
           "hdop": np.sqrt(diag[:, :min(D, 2)].sum(axis=1))}
    if D == 3:
        out["vdop"] = np.sqrt(diag[:, 2])
    if clock_bias:
        out["tdop"] = np.sqrt(diag[:, D])
    return {k: float(v[0]) if single else v for k, v in out.items()}


def gdop(anchors, points, *, clock_bias: bool = False) -> np.ndarray | float:
    """Geometric dilution of precision ``sqrt(trace((H^T H)^-1))``; see :func:`dop`."""
    return dop(anchors, points, clock_bias=clock_bias)["gdop"]


__all__ = ["aoa_crlb", "aoa_fim", "crlb_covariance", "crlb_rmse", "dop", "gdop", "rss_crlb", "rss_fim",
           "tdoa_crlb", "tdoa_fim", "toa_crlb", "toa_fim"]
