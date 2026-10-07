"""Model-based localization from geometric measurements (numpy only).

Measurements arrive as ``X`` and the anchor geometry as the constructor parameter
``anchors=(A, D)`` (CONTRACTS.md, section 5); NaN marks a missing measurement.

    TrilaterationLocalizer      ranges (ToA, RTT, UWB)         X (N, A) metres
    TDOALocalizer               range differences to anchor 0  X (N, A - 1) metres
    WeightedCentroidLocalizer   RSSI or ranges                 X (N, A) dBm or metres

The array functions behind them are public: ``linear_trilateration``, ``chan_tdoa``
and ``gdop``. ``fit`` is optional for these models: without labelled data they
localize straight away, and ``fit(X)`` without positions only records the input size;
with ``calibrate=True`` ``fit(X, positions)`` learns a per-anchor bias and the noise scale
from labelled measurements (positions are then required). ``from_meta(table.meta)`` builds
a localizer from the geometry a table carries (``meta["anchors"]``).

The batched Gauss-Newton/IRLS solver in this module (``_gauss_newton``) is shared by
``methods.pathloss`` and ``methods.aoa``: one independent small problem per row,
vectorised over rows, with step halving so the cost never increases.
"""
from __future__ import annotations

import numpy as np

from ..core import Prediction
from .base import BaseLocalizer, _unpack

_TINY = 1e-300
_TUNING = {"huber": 1.345, "tukey": 4.685}  # 95 % efficiency under Gaussian noise (Holland & Welsch 1977)
_MAD = 1.482602218505602  # 1 / Phi^-1(3/4): MAD -> standard deviation under Gaussian noise


# ---------------------------------------------------------------------------------------------
# Shared numerics
# ---------------------------------------------------------------------------------------------

def _as_anchors(anchors, owner: str, dims=None) -> np.ndarray:
    if anchors is None:
        raise ValueError(f"{owner} needs the anchor geometry: anchors=(A, D) array of anchor positions")
    a = np.asarray(anchors, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"anchors must be an (A, D) array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("anchors contain NaN or inf")
    if dims is not None and a.shape[1] not in dims:
        raise ValueError(f"{owner} works in {' or '.join(map(str, dims))}-D, got anchors of shape {a.shape}")
    return a


def _finite_rows(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(A with identity in place of non-finite matrices, mask of the finite ones)``: eigh
    raises on NaN, which would fail the whole batch for one degenerate row."""
    finite = np.all(np.isfinite(A), axis=(1, 2))
    return np.where(finite[:, None, None], A, np.eye(A.shape[-1])), finite


def _solve_sym(A: np.ndarray, b: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Solve a stack of symmetric PSD systems ``A x = b``; NaN rows where ``A`` is (near)
    singular or ``A``/``b`` hold NaN."""
    A, finite = _finite_rows(A)
    finite &= np.all(np.isfinite(b), axis=1)
    ev, V = np.linalg.eigh(A)
    ok = finite & (ev[:, -1] > 0) & (ev[:, 0] > rcond * ev[:, -1])
    safe = np.where(ok[:, None], ev, 1.0)
    x = np.einsum("nij,nj->ni", V, np.einsum("nji,nj->ni", V, np.where(ok[:, None], b, 0.0)) / safe)
    x[~ok] = np.nan
    return x


def _inv_trace(A: np.ndarray, first: int | None = None, rcond: float = 1e-12) -> np.ndarray:
    """``trace(inv(A)[:first, :first])`` for a stack of symmetric PSD matrices; NaN if singular
    or not finite."""
    A, finite = _finite_rows(A)
    ev, V = np.linalg.eigh(A)
    ok = finite & (ev[:, -1] > 0) & (ev[:, 0] > rcond * ev[:, -1])
    safe = np.where(ok[:, None], ev, 1.0)
    k = A.shape[1] if first is None else first
    tr = np.einsum("nij,nj->n", np.square(V[:, :k, :]), 1.0 / safe)
    return np.where(ok, tr, np.nan)


def _rho(u, loss: str, c: float):
    """Loss of a standardised residual ``u``: 'l2', 'huber' or 'tukey' (biweight)."""
    if loss == "l2":
        return 0.5 * u * u
    a = np.abs(u)
    if loss == "huber":
        return np.where(a <= c, 0.5 * u * u, c * a - 0.5 * c * c)
    return (c * c / 6.0) * (1.0 - np.clip(1.0 - np.square(u / c), 0.0, None) ** 3)


def _irls_weight(u, loss: str, c: float):
    """IRLS weight psi(u) / u of the losses in ``_rho``."""
    if loss == "l2":
        return np.ones_like(u)
    if loss == "huber":
        return c / np.maximum(np.abs(u), c)
    return np.square(np.clip(1.0 - np.square(u / c), 0.0, None))


def _masked_median(values, valid):
    """Row-wise median over the valid entries (NaN for rows without any); no warnings."""
    n = valid.sum(axis=1)
    a = np.sort(np.where(valid, values, np.inf), axis=1)
    lo = np.take_along_axis(a, np.maximum((n - 1) // 2, 0)[:, None], axis=1)[:, 0]
    hi = np.take_along_axis(a, np.maximum(n // 2, 0)[:, None], axis=1)[:, 0]
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, 0.5 * (lo + hi), np.nan)


def _robust_scale(e, valid, floor):
    """1.4826 * median |e| over the valid entries of each row, at least ``floor``."""
    return np.maximum(_MAD * np.nan_to_num(_masked_median(np.abs(e), valid)), floor)


def _gauss_newton(x0, model, *, max_iter: int = 50, tol: float = 1e-10, loss: str = "l2",
                  c: float | None = None, scale=None, project=None):
    """Minimise ``sum rho((z - h(x)) / s)`` independently for every row, vectorised over rows.

    ``model(x, rows, jac)`` returns ``(e, J, valid)`` (``(e, valid)`` if ``jac`` is False):
    residuals ``e = z - h(x)`` of shape (n, m) with 0 where a measurement is missing,
    the Jacobian ``dh/dx`` (n, m, p) and the validity mask. ``loss`` is 'l2' (Gauss-Newton)
    or 'huber'/'tukey' (IRLS, Holland & Welsch 1977); the scale ``s`` is ``scale`` ((N,)
    array) or, if None, 1.4826 * median |e| re-estimated at every iteration (robust
    losses only), floored at 1e-9 times the largest initial residual so that an exact
    fit never divides by zero. Each step is halved until the cost does not increase, so
    the cost is monotone for a fixed scale. ``project(x, rows)`` maps candidates into the
    feasible set (box constraints). Rows whose start is NaN stay NaN.
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    c = _TUNING.get(loss, 1.0) if c is None else float(c)
    active = np.all(np.isfinite(x), axis=1)
    fixed = None if scale is None else np.broadcast_to(np.asarray(scale, np.float64), (len(x),))
    floor = np.full(len(x), 1e-9)
    for it in range(max_iter):
        rows = np.flatnonzero(active)
        if rows.size == 0:
            break
        xr = x[rows]
        e, J, valid = model(xr, rows, True)
        if loss == "l2":
            s = np.ones(len(rows))
        elif fixed is not None:
            s = fixed[rows]
        else:
            if it == 0:
                floor[rows] = 1e-9 * (1.0 + np.max(np.abs(e), axis=1, initial=0.0))
            s = _robust_scale(e, valid, floor[rows])
        u = e / s[:, None]
        w = valid * _irls_weight(u, loss, c)
        cost0 = np.sum(valid * _rho(u, loss, c), axis=1)
        step = _solve_sym(np.einsum("nm,nmi,nmj->nij", w, J, J), np.einsum("nm,nmi,nm->ni", w, J, e))
        todo = np.all(np.isfinite(step), axis=1)
        t = np.ones(len(rows))
        new = xr.copy()
        moved = np.zeros(len(rows), dtype=bool)
        for _ in range(40):  # step halving
            idx = np.flatnonzero(todo)
            if idx.size == 0:
                break
            cand = xr[idx] + t[idx, None] * step[idx]
            if project is not None:
                cand = project(cand, rows[idx])
            ec, vc = model(cand, rows[idx], False)
            ok = np.sum(vc * _rho(ec / s[idx, None], loss, c), axis=1) <= cost0[idx] * (1 + 1e-12) + 1e-300
            new[idx[ok]] = cand[ok]
            moved[idx[ok]] = True
            todo[idx[ok]] = False
            t[idx[~ok]] *= 0.5
        delta = np.sqrt(np.sum(np.square(new - xr), axis=1))
        x[rows] = new
        done = ~moved | (delta <= tol * (1.0 + np.sqrt(np.sum(np.square(xr), axis=1))))
        active[rows[done]] = False
    return x


#: An estimate farther than this many anchor spreads from the anchors' centroid is reported as
#: NaN ("not placed") unless the data resolve it (see :func:`_far_field`): least squares has then
#: run into the far field of the geometry, where the cost is flat along the range.
FAR_FIELD = 10.0


def _refine(starts, model, *, max_iter: int, tol: float, **options) -> np.ndarray:
    """Gauss-Newton from every start in ``starts`` (a list of (N, p) arrays); per row keep the
    solution with the lowest cost (the first start wins ties within 1e-12, so an ambiguous
    minimal case keeps the closed form's choice instead of flipping on rounding noise).

    Gauss-Newton only descends from its start, so a poor start (a closed form under large noise)
    can slide into a far-field valley of the cost; the caller then applies :func:`_far_field`.
    """
    best, best_cost = None, None
    for start in starts:
        x = _gauss_newton(start, model, max_iter=max_iter, tol=tol, **options)
        cost = np.full(len(x), np.inf)
        rows = np.flatnonzero(np.all(np.isfinite(x), axis=1))
        if rows.size:
            e, valid = model(x[rows], rows, False)
            cost[rows] = np.sum(valid * e * e, axis=1)
        if best is None:
            best, best_cost = x, cost
        else:
            with np.errstate(invalid="ignore"):
                better = np.isfinite(cost) & ~(cost >= best_cost - 1e-12 * (1.0 + best_cost))
            best[better], best_cost[better] = x[better], cost[better]
    return best


def _far_field(pos: np.ndarray, anchors: np.ndarray, gdop: np.ndarray, sigma, exact: np.ndarray) -> np.ndarray:
    """Rows to report as NaN: farther than ``FAR_FIELD`` anchor spreads (RMS distance of the
    anchors from their centroid) from the centroid, unless the data vouch for the estimate there:
    its fit is exact with redundancy to spare (``exact``), or the KNOWN noise scale ``sigma``
    (the ``sigma`` parameter or the calibrated ``sigma_``; NaN if neither) times the unit-noise
    uncertainty ``gdop`` at the estimate is smaller than its distance.

    A run into the flat far field has an enormous (or singular) GDOP; a residual-based noise
    estimate cannot vouch for it because a few redundant measurements can fit a runaway almost
    as well as the truth. A precisely measured target outside a small anchor cluster is kept."""
    centre = anchors.mean(axis=0)
    size = max(float(np.sqrt(np.mean(np.sum(np.square(anchors - centre), axis=1)))), _TINY)
    dist = np.sqrt(np.sum(np.square(pos - centre), axis=1))
    with np.errstate(invalid="ignore"):
        return (dist > FAR_FIELD * size) & ~(exact | (np.asarray(sigma, np.float64) * gdop < dist))


def _exact_fit(e: np.ndarray, w: np.ndarray, n_unknowns: int, scale: float) -> np.ndarray:
    """Rows whose weighted RMS residual is at rounding level (``1e-9 * scale``) although there
    are more measurements than unknowns: noise-free data, which pin the estimate down."""
    n = w.sum(axis=1)
    rms = np.sqrt(np.sum(w * e * e, axis=1) / np.maximum(n, 1.0))
    return (n - n_unknowns > 1e-9) & (rms <= 1e-9 * scale)


def _range_model(anchors: np.ndarray, z: np.ndarray, offset: bool = False, height=None):
    """Residuals of ranges ``z`` (N, A) at candidate positions; ``offset=True`` adds a free
    common term (pseudoranges: the unknown vector is ``[x, b]`` and ``h = |x - a| + b``).
    ``height`` (3-D anchors): the unknown is the horizontal position ``(x, y)`` of a point at
    that known z, and the slant ranges ``|(x, y, height) - a|`` are the model."""
    D = anchors.shape[1] if height is None else anchors.shape[1] - 1

    def model(x, rows, jac):
        pos = x[:, :D]
        if height is not None:
            pos = np.concatenate([pos, np.full((len(x), 1), float(height))], axis=1)
        diff = pos[:, None, :] - anchors[None]
        d = np.sqrt(np.einsum("nad,nad->na", diff, diff))
        pred = d + x[:, D:D + 1] if offset else d
        zr = z[rows]
        valid = np.isfinite(zr)
        e = np.where(valid, zr - pred, 0.0)
        if not jac:
            return e, valid
        J = diff[..., :D] / np.maximum(d, _TINY)[..., None]
        if offset:
            J = np.concatenate([J, np.ones(J.shape[:2] + (1,))], axis=2)
        return e, np.where(valid[..., None], J, 0.0), valid

    return model


def _spread(J, w, sigma, first: int | None = None) -> np.ndarray:
    """``sigma * sqrt(trace((J^T W J)^-1))``: the CRLB-shaped RMS position error at the estimate."""
    H = np.einsum("nm,nmi,nmj->nij", w, J, J)
    return np.asarray(sigma, dtype=np.float64) * np.sqrt(_inv_trace(H, first))


def _residual_sigma(e, w, n_unknowns: int) -> np.ndarray:
    """Per-row noise scale sqrt(sum w e^2 / (sum w - p)); NaN without redundancy."""
    dof = w.sum(axis=1) - n_unknowns
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dof > 1e-9, np.sqrt(np.sum(w * e * e, axis=1) / np.where(dof > 1e-9, dof, 1.0)), np.nan)


def _calibration(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-column median bias and the pooled robust (MAD) noise scale of labelled residuals."""
    valid = np.isfinite(residuals)
    bias = np.nan_to_num(_masked_median(residuals.T, valid.T))  # anchors never measured: no bias
    centred = np.abs(residuals - bias).reshape(1, -1)
    sigma = float(_MAD * _masked_median(centred, np.isfinite(centred))[0])
    return bias, sigma


# ---------------------------------------------------------------------------------------------
# Array functions
# ---------------------------------------------------------------------------------------------

def linear_trilateration(ranges, anchors) -> np.ndarray:
    """Closed-form least-squares position from ranges; NaN ranges are dropped per row.

    Squaring ``|x - a_i| = r_i`` gives ``2 a_i^T x - |x|^2 = |a_i|^2 - r_i^2``, linear in
    ``x`` once the equation mean over the row's valid anchors is subtracted (differencing
    against the mean equation is identical to solving with ``|x|^2`` as a free unknown,
    and is independent of which anchor would be the reference). The solution is exact
    for exact ranges and a good start for Gauss-Newton otherwise.

    ranges  (N, A) metres, NaN = missing.  anchors (A, D).
    Returns (N, D) float64; NaN rows have fewer than D + 1 valid ranges or anchors that
    do not span D dimensions.

    References: J. J. Caffery, "A new approach to the geometry of TOA location", IEEE VTC
    Fall, 2000. DOI 10.1109/VETECF.2000.886153. A. H. Sayed, A. Tarighat, N. Khajehnouri,
    "Network-based wireless location", IEEE Signal Processing Magazine, 2005.
    DOI 10.1109/MSP.2005.1458275.
    """
    anchors = _as_anchors(anchors, "linear_trilateration")
    r = np.asarray(ranges, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != len(anchors):
        raise ValueError(f"ranges must be (N, {len(anchors)}), got shape {r.shape}")
    origin = anchors.mean(axis=0)  # centred coordinates keep |a|^2 small (projected CRS: 1e6 m)
    a = anchors - origin
    valid = np.isfinite(r)
    m = valid.sum(axis=1)
    v = valid.astype(np.float64)
    b = np.where(valid, np.sum(a * a, axis=1) - np.square(np.where(valid, r, 0.0)), 0.0)  # (N, A)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_bar = (v @ a) / m[:, None]  # (N, D)
        b_bar = np.sum(b, axis=1) / m
    ac = 2.0 * (a[None] - a_bar[:, None, :]) * v[..., None]  # (N, A, D)
    bc = (b - b_bar[:, None]) * v
    x = np.full((len(r), a.shape[1]), np.nan)
    ok = m >= a.shape[1] + 1
    if ok.any():
        H = np.einsum("nad,nae->nde", ac[ok], ac[ok])
        x[ok] = _solve_sym(H, np.einsum("nad,na->nd", ac[ok], bc[ok]), rcond=1e-10)
    return x + origin


def chan_tdoa(tdoa, anchors) -> np.ndarray:
    """Chan and Ho's closed-form TDoA position estimate (two-step weighted least squares).

    ``tdoa[:, i - 1] = |x - a_i| - |x - a_0|`` in metres (range differences to anchor 0);
    NaN = missing. Step 1 solves the linear system in ``[x, |x - a_0|]`` by WLS with the
    weight ``(B Q B)^-1`` (``Q`` = covariance of range differences sharing a reference,
    ``B`` = diag of the anchor distances, re-estimated once); step 2 imposes
    ``|x - a_0|^2 = sum (x - a_0)^2`` by a second WLS. With exact data the result is exact.
    Rows with exactly D valid differences use the closed-form quadratic of the minimal
    case; of its roots, those giving negative ranges are discarded, and if two remain the
    one nearer the anchor centroid is returned (the minimal case is ambiguous).

    tdoa (N, A - 1), anchors (A, D) -> (N, D) float64, NaN where fewer than D differences
    are valid or the geometry is degenerate.

    Deviation from the paper: where a coordinate of the step-1 estimate relative to the
    reference anchor is zero to 1e-9 relative precision (step 2 is then singular), the
    step-1 estimate is returned.

    Known limitation (inherent in step 2, which estimates the squared coordinates
    ``(x_k - a_0k)^2`` and takes signed square roots): when the target is within a few noise
    standard deviations of the reference anchor in some coordinate, that coordinate loses
    efficiency (about twice the CRLB for a target level with anchor 0). ``TDOALocalizer``
    (``refine=True``, the default) refines by maximum likelihood and reaches the bound there.

    References: Y. T. Chan, K. C. Ho, "A simple and efficient estimator for hyperbolic
    location", IEEE Transactions on Signal Processing 42(8), 1994. DOI 10.1109/78.301830.
    """
    anchors = _as_anchors(anchors, "chan_tdoa")
    d = np.asarray(tdoa, dtype=np.float64)
    A, D = anchors.shape
    if d.ndim != 2 or d.shape[1] != A - 1:
        raise ValueError(f"tdoa must be (N, {A - 1}) for {A} anchors, got shape {d.shape}")
    rel = anchors[1:] - anchors[0]  # the reference anchor is the origin
    K = np.sum(rel * rel, axis=1)
    valid = np.isfinite(d)
    v = valid.astype(np.float64)
    dz = np.where(valid, d, 0.0)
    m = valid.sum(axis=1)
    out = np.full((len(d), D), np.nan)

    over = np.flatnonzero(m >= D + 1)
    if over.size:
        vo, do = v[over], dz[over]
        G = np.concatenate([np.broadcast_to(2.0 * rel, (len(over), A - 1, D)), 2.0 * do[..., None]], axis=2)
        G = G * vo[..., None]
        h = (K - do * do) * vo
        Qinv = vo[:, :, None] * np.eye(A - 1) - vo[:, :, None] * vo[:, None, :] / (m[over, None, None] + 1.0)
        z = _solve_sym(np.einsum("nki,nkl,nlj->nij", G, Qinv, G), np.einsum("nki,nkl,nl->ni", G, Qinv, h))
        r = np.sqrt(np.sum(np.square(z[:, None, :D] - rel[None]), axis=2))  # distances to anchors 1..A-1
        r = np.where(vo > 0, np.maximum(r, 1e-12), 1.0)
        Pinv = Qinv / (r[:, :, None] * r[:, None, :])
        N1 = np.einsum("nki,nkl,nlj->nij", G, Pinv, G)
        z = _solve_sym(N1, np.einsum("nki,nkl,nl->ni", G, Pinv, h))
        # step 2: (x_k^2, r0^2) = G' (x_k^2) with G' = [I; 1^T], weight (B' cov(z) B')^-1, cov(z) = N1^-1
        Gp = np.vstack([np.eye(D), np.ones((1, D))])
        scale = np.max(np.abs(z), axis=1, keepdims=True)
        good = np.all(np.isfinite(z), axis=1) & np.all(np.abs(z) > 1e-9 * np.maximum(scale, 1.0), axis=1)
        x = z[:, :D].copy()
        if good.any():
            zg = z[good]
            W2 = N1[good] / (4.0 * zg[:, :, None] * zg[:, None, :])
            zp = _solve_sym(np.einsum("ki,nkl,lj->nij", Gp, W2, Gp), np.einsum("ki,nkl,nl->ni", Gp, W2, zg * zg))
            x2 = np.sign(zg[:, :D]) * np.sqrt(np.abs(zp))
            fine = np.all(np.isfinite(x2), axis=1)
            x[np.flatnonzero(good)[fine]] = x2[fine]
        out[over] = x + anchors[0]

    centroid = anchors.mean(axis=0) - anchors[0]
    for i in np.flatnonzero(m == D):  # minimal case: x = alpha + beta r0, then |x|^2 = r0^2
        vi = valid[i]
        Ar = 2.0 * rel[vi]
        di = d[i, vi]
        if abs(np.linalg.det(Ar)) < 1e-12 * max(1.0, np.abs(Ar).max()) ** D:
            continue
        alpha = np.linalg.solve(Ar, K[vi] - di * di)
        beta = np.linalg.solve(Ar, -2.0 * di)
        qa, qb, qc = beta @ beta - 1.0, 2.0 * alpha @ beta, alpha @ alpha
        if abs(qa) < 1e-12:
            roots = np.array([-qc / qb]) if qb != 0 else np.array([])
        else:
            disc = qb * qb - 4.0 * qa * qc
            sq = np.sqrt(max(disc, 0.0))
            roots = np.array([(-qb - sq) / (2 * qa), (-qb + sq) / (2 * qa)])
        tol = 1e-9 * (1.0 + np.abs(di).max())
        roots = [r0 for r0 in roots if r0 >= -tol and np.all(r0 + di >= -tol)]  # ranges must be >= 0
        if roots:
            cands = [alpha + beta * max(r0, 0.0) for r0 in roots]
            best = min(range(len(cands)), key=lambda j: (np.sum(np.square(cands[j] - centroid)), j))
            out[i] = cands[best] + anchors[0]
    return out


def gdop(anchors, positions, kind: str = "ranges") -> np.ndarray:
    """Geometric dilution of precision at each position: ``sqrt(trace((H^T H)^-1))``.

    ``kind="ranges"``: ``H`` stacks the unit vectors from the anchors, so the RMS position
    error of an efficient range-based estimator is ``sigma * GDOP`` for i.i.d. range
    noise ``sigma``. ``kind="tdoa"``: range differences to anchor 0 with covariance
    ``sigma^2 (I + 1 1^T)``, computed as pseudoranges with a free common offset; only the
    position block of ``(H^T H)^-1`` is summed (the offset is a nuisance), so again the
    TDoA Cramer-Rao bound is ``sigma * GDOP`` (in GNSS terms this is the PDOP of the
    pseudorange model, ``evaluation.dop(..., clock_bias=True)["pdop"]``).

    References: D. J. Torrieri, "Statistical theory of passive location systems", IEEE
    Transactions on Aerospace and Electronic Systems AES-20(2), 1984.
    DOI 10.1109/TAES.1984.310439.
    """
    anchors = _as_anchors(anchors, "gdop")
    p = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    diff = p[:, None, :] - anchors[None]
    H = diff / np.maximum(np.sqrt(np.sum(diff * diff, axis=2)), _TINY)[..., None]
    first = None
    if kind == "tdoa":
        H = np.concatenate([H, np.ones(H.shape[:2] + (1,))], axis=2)
        first = anchors.shape[1]
    elif kind != "ranges":
        raise ValueError(f"kind must be 'ranges' or 'tdoa', got {kind!r}")
    return np.sqrt(_inv_trace(np.einsum("nai,naj->nij", H, H), first))


# ---------------------------------------------------------------------------------------------
# Localizers
# ---------------------------------------------------------------------------------------------

def _meta_anchors(meta, modality: str, owner: str) -> np.ndarray:
    """``meta["anchors"]`` of a table of the given modality (a ``meta`` without ``modality``
    is taken as it is)."""
    if meta.get("modality", modality) != modality:
        raise ValueError(f"{owner}.from_meta needs the meta of a {modality!r} table, got modality "
                         f"{meta.get('modality')!r}")
    if meta.get("anchors") is None:
        raise ValueError(f"{owner}.from_meta needs meta['anchors'], the (A, D) anchor positions")
    return np.asarray(meta["anchors"], dtype=np.float64)


class GeometricLocalizer(BaseLocalizer):
    """Base of the model-based localizers: the geometry is a parameter, so ``localize``
    works before ``fit`` unless a parameter has to be learned (``_requires_fit``).

    ``fit(X)`` without positions is accepted when nothing is learned from them
    (``_labels_optional``: a ``calibrate`` parameter set to False); it records the input size.
    Subclasses implement ``_check_input(X)`` (measurement layout against the geometry); their
    ``_fit`` receives ``pos=None`` when no positions were given.
    Floor and building are not predicted: these models are single-floor geometry.
    """

    _allow_nan = True

    @property
    def _requires_fit(self) -> bool:
        return bool(getattr(self, "calibrate", False))

    @property
    def _labels_optional(self) -> bool:
        # only a model whose fit learns from positions just for calibration; one that always learns
        # from them (PathLossLocalizer: sigma_, P0, exponent) has no ``calibrate`` and keeps y required
        return "calibrate" in self._get_param_names() and not self.calibrate

    def _missing_labels_message(self) -> str:
        msg = super()._missing_labels_message()
        if getattr(self, "calibrate", False):
            msg += ("; calibrate=True learns the calibration from labelled measurements (with calibrate=False, "
                    "fit(X) needs no positions and localize works without fit)")
        return msg

    def _validate(self, X, reset: bool = False) -> np.ndarray:
        if reset or hasattr(self, "n_features_in_") or self._requires_fit:
            X = super()._validate(X, reset)
        else:  # geometry-only model used without fit: validate against the geometry instead
            X = super()._validate(X, reset=True)
        self._check_input(X)
        return X

    def _check_input(self, X) -> None:
        raise NotImplementedError

    def _check_labels(self, pos, anchors) -> None:
        if pos is not None and pos.shape[1] != anchors.shape[1]:
            raise ValueError(f"positions are {pos.shape[1]}-D but anchors are {anchors.shape[1]}-D")

    def _clear_calibration(self) -> None:
        for name in ("bias_", "sigma_"):
            self.__dict__.pop(name, None)

    def _noise(self, fallback: np.ndarray) -> np.ndarray:
        """Noise scale for ``spread``: ``sigma``, else the calibrated ``sigma_``, else per row."""
        if getattr(self, "sigma", None) is not None:
            return np.full(len(fallback), float(self.sigma))
        if np.isfinite(getattr(self, "sigma_", np.nan)):
            return np.full(len(fallback), self.sigma_)
        return fallback


class TrilaterationLocalizer(GeometricLocalizer):
    """Position from ranges to anchors at known positions (ToA, two-way RTT, UWB).

    ``solver``:
      ``"linear"``        closed-form least squares on the differenced squared-range equations
                          (``linear_trilateration``);
      ``"gauss_newton"``  maximum likelihood under i.i.d. Gaussian range noise: Gauss-Newton on
                          ``sum (r_i - |x - a_i|)^2`` started from the linear solution (Foy 1976);
      ``"irls"``          robust M-estimate by iteratively reweighted least squares with the
                          ``loss`` ``"huber"`` or ``"tukey"`` (biweight; started from the Huber
                          solution), scale 1.4826 * median |residual| unless ``sigma`` is given.
                          Down-weights NLOS ranges and outliers.

    NaN ranges are dropped per sample; rows with fewer than D + 1 valid ranges return NaN.
    ``Prediction.spread`` = ``sigma * sqrt(trace((J^T W J)^-1))`` at the estimate, i.e. sigma
    times the GDOP of the anchors used: ``sigma`` if given, else the calibrated ``sigma_``,
    else the per-sample residual scale. With ``calibrate=True``, ``fit(X, positions)`` learns
    a per-anchor range bias ``bias_`` (median residual, e.g. an NLOS or antenna delay) and the
    noise scale ``sigma_``; otherwise ``fit`` only records the input size (``fit(X)`` needs no
    positions) and ``localize`` works without ``fit``.

    Mirror ambiguity of 3-D anchors in (nearly) one plane, e.g. UWB anchors on the ceiling: a
    point and its mirror image across the anchors' plane have the same ranges, so a 3-D solve
    cannot tell a tag below the ceiling from one above it. With noise it returns either (or a
    point pulled towards the plane, since the height is barely observable). When the tag height
    is known (a device carried at a fixed height, a tag on a robot or a trolley), pass
    ``height=``: only the horizontal position ``(x, y)`` is estimated, and the output stays in
    the anchors' 3-D frame, ``(x, y, height)``, so it compares with 3-D positions directly
    (take ``pos[:, :2]`` for a floor plan). Rows with fewer than 3 valid ranges return NaN.
    The linear start trilaterates the horizontal ranges ``sqrt(max(r^2 - (z_a - height)^2, 0))``
    against the anchors' ``(x, y)``; Gauss-Newton and IRLS then fit the measured slant ranges
    ``|(x, y, height) - a|`` (the maximum-likelihood problem), and ``spread`` is the horizontal
    CRLB with the height known. Anchors exactly in one plane make the 3-D solve singular (NaN
    for every row); ``height=`` places them.

    Parameters
    ----------
    anchors : (A, D) array, anchor positions in the frame of the target positions.
    solver : "linear", "gauss_newton" or "irls".
    loss : "huber" or "tukey" (only for ``solver="irls"``).
    tuning : loss constant in units of the noise scale (default 1.345 Huber, 4.685 Tukey).
    sigma : range noise std (metres) for ``spread`` and the robust scale; None = estimated.
    calibrate : learn ``bias_`` and ``sigma_`` in ``fit``.
    max_iter, tol : Gauss-Newton/IRLS iterations and relative step tolerance.
    height : known z of the target in the frame of (A, 3) anchors: solve for (x, y) only and
        return (x, y, height). None (default) = solve for every coordinate.

    References
    ----------
    W. H. Foy, "Position-location solutions by Taylor-series estimation", IEEE Transactions on
    Aerospace and Electronic Systems AES-12(2), 1976. DOI 10.1109/TAES.1976.308294.
    J. J. Caffery, "A new approach to the geometry of TOA location", IEEE VTC Fall, 2000.
    DOI 10.1109/VETECF.2000.886153.
    P. W. Holland, R. E. Welsch, "Robust regression using iteratively reweighted least-squares",
    Communications in Statistics - Theory and Methods 6(9), 1977. DOI 10.1080/03610927708827533.
    P. J. Huber, "Robust estimation of a location parameter", Annals of Mathematical Statistics
    35(1), 1964. DOI 10.1214/aoms/1177703732.
    A. E. Beaton, J. W. Tukey, "The fitting of power series, meaning polynomials, illustrated on
    band-spectroscopic data", Technometrics 16(2), 1974. DOI 10.1080/00401706.1974.10489171.
    """

    def __init__(self, anchors=None, solver: str = "gauss_newton", loss: str = "huber", tuning=None,
                 sigma=None, calibrate: bool = False, max_iter: int = 50, tol: float = 1e-10, height=None):
        self.anchors = anchors
        self.solver = solver
        self.loss = loss
        self.tuning = tuning
        self.sigma = sigma
        self.calibrate = calibrate
        self.max_iter = max_iter
        self.tol = tol
        self.height = height

    @classmethod
    def from_meta(cls, meta, **params) -> TrilaterationLocalizer:
        """A localizer configured from the ``meta`` of a ``ranges`` table (``meta["anchors"]``).
        ``params`` override or add constructor arguments, e.g.
        ``TrilaterationLocalizer.from_meta(train.meta, height=1.2)`` for ceiling anchors."""
        return cls(**{"anchors": _meta_anchors(meta, "ranges", cls.__name__), **params})

    def _geometry(self):
        """(anchors, height or None, number of estimated coordinates)."""
        anchors = _as_anchors(self.anchors, type(self).__name__)
        if self.height is None:
            return anchors, None, anchors.shape[1]
        h = float(self.height)
        if anchors.shape[1] != 3 or not np.isfinite(h):
            raise ValueError(f"height= is the known z of the target and needs 3-D anchors (A, 3); got anchors of "
                             f"shape {anchors.shape} and height={self.height!r}")
        return anchors, h, 2

    def _check_input(self, X):
        anchors = _as_anchors(self.anchors, type(self).__name__)
        if X.ndim != 2 or X.shape[1] != len(anchors):
            raise ValueError(f"X must hold one range per anchor, shape (N, {len(anchors)}); got {X.shape}")
        if X.dtype.kind == "c":
            raise ValueError("ranges must be real-valued")

    def _check_options(self):
        if self.solver not in ("linear", "gauss_newton", "irls"):
            raise ValueError(f"solver must be 'linear', 'gauss_newton' or 'irls', got {self.solver!r}")
        if self.solver == "irls" and self.loss not in ("huber", "tukey"):
            raise ValueError(f"loss must be 'huber' or 'tukey', got {self.loss!r}")

    def _fit(self, X, pos, floor, building):
        self._check_options()
        anchors, _, _ = self._geometry()
        self._check_labels(pos, anchors)
        self._clear_calibration()
        if self.calibrate:
            true = np.sqrt(np.sum(np.square(pos[:, None, :] - anchors[None]), axis=2))
            self.bias_, self.sigma_ = _calibration(np.asarray(X, np.float64) - true)

    def _localize(self, X):
        self._check_options()
        anchors, h, D = self._geometry()
        z = np.asarray(X, dtype=np.float64) - getattr(self, "bias_", 0.0)
        if h is None:
            x = linear_trilateration(z, anchors)
        else:  # horizontal ranges for the linear start; the refinement fits the slant ranges
            with np.errstate(invalid="ignore"):
                horizontal = np.sqrt(np.maximum(z * z - np.square(anchors[:, 2] - h), 0.0))
            x = linear_trilateration(np.where(np.isfinite(z), horizontal, np.nan), anchors[:, :2])
        model = _range_model(anchors, z, height=h)
        scale = None if self.sigma is None else np.full(len(z), float(self.sigma))
        if self.solver == "gauss_newton":
            x = _gauss_newton(x, model, max_iter=self.max_iter, tol=self.tol)
        elif self.solver == "irls":
            fixed = scale if scale is not None else (np.full(len(z), self.sigma_) if np.isfinite(
                getattr(self, "sigma_", np.nan)) else None)
            x = _gauss_newton(x, model, max_iter=self.max_iter, tol=self.tol, loss="huber",
                              c=self.tuning if self.loss == "huber" else None, scale=fixed)
            if self.loss == "tukey":
                x = _gauss_newton(x, model, max_iter=self.max_iter, tol=self.tol, loss="tukey", c=self.tuning,
                                  scale=fixed)
        spread = np.full(len(z), np.nan)
        rows = np.flatnonzero(np.all(np.isfinite(x), axis=1))
        if rows.size:
            e, J, valid = model(x[rows], rows, True)
            w = valid.astype(np.float64)
            if self.solver == "irls":
                c = _TUNING[self.loss] if self.tuning is None else float(self.tuning)
                s = self._noise(_robust_scale(e, valid, 1e-12 * (1.0 + np.max(np.abs(e), axis=1))))
                w = valid * _irls_weight(e / s[:, None], self.loss, c)
                sigma = s
            else:
                sigma = self._noise(_residual_sigma(e, w, D))
            spread[rows] = _spread(J, w, sigma)
        if h is not None:  # back to the anchors' 3-D frame; rows not placed stay NaN
            x = np.column_stack([x, np.where(np.isfinite(x[:, 0]), h, np.nan)])
        return Prediction(x, spread=spread)


class TDOALocalizer(GeometricLocalizer):
    """Position from range differences (TDoA) with Chan and Ho's closed form.

    ``X[:, i - 1] = |x - a_i| - |x - a_0|`` in metres (time differences times the propagation
    speed), relative to anchor 0 as in the ``tdoa`` modality of CONTRACTS.md; NaN = missing.
    The closed-form estimate (``chan_tdoa``) is refined, if ``refine=True``, by Gauss-Newton
    (Taylor-series, Foy 1976) on the maximum-likelihood cost under i.i.d. arrival-time noise;
    the TDoA covariance ``sigma^2 (I + 1 1^T)`` is handled exactly by writing the measurements
    as pseudoranges ``[0, X]`` with a free common offset. Rows with fewer than D valid
    differences return NaN; rows with exactly D use the ambiguous minimal solution. The
    refinement also starts from the anchors' centroid (which also places rows where the closed
    form is singular) and keeps the lower cost. An estimate farther than ``FAR_FIELD`` (10)
    anchor spreads from the anchors is returned as NaN, so ``evaluate`` counts a run into the
    far field in ``n_failed`` instead of averaging a position kilometres away, unless the data
    vouch for it: an exact fit with redundancy, or a known noise scale (``sigma`` or the
    calibrated ``sigma_``) whose CRLB at the estimate is smaller than its distance. So a
    precisely measured target outside a small anchor cluster is kept.

    ``Prediction.spread`` = ``sigma * sqrt(trace(CRLB/sigma^2))`` at the estimate, where
    ``sigma`` is the per-anchor range noise std (``sigma``, else the calibrated ``sigma_``, else
    estimated from the residuals when there is redundancy, else NaN). With ``calibrate=True``,
    ``fit(X, positions)`` learns a per-difference bias ``bias_`` and ``sigma_``.

    Parameters
    ----------
    anchors : (A, D) array; anchor 0 is the reference.
    refine : Gauss-Newton refinement of the closed-form estimate.
    sigma : range (not range-difference) noise std in metres, for ``spread``.
    calibrate : learn ``bias_`` (A - 1,) and ``sigma_`` in ``fit``.
    max_iter, tol : refinement iterations and relative step tolerance.

    References
    ----------
    Y. T. Chan, K. C. Ho, "A simple and efficient estimator for hyperbolic location", IEEE
    Transactions on Signal Processing 42(8), 1994. DOI 10.1109/78.301830.
    W. H. Foy, "Position-location solutions by Taylor-series estimation", IEEE Transactions on
    Aerospace and Electronic Systems AES-12(2), 1976. DOI 10.1109/TAES.1976.308294.
    """

    def __init__(self, anchors=None, refine: bool = True, sigma=None, calibrate: bool = False,
                 max_iter: int = 50, tol: float = 1e-10):
        self.anchors = anchors
        self.refine = refine
        self.sigma = sigma
        self.calibrate = calibrate
        self.max_iter = max_iter
        self.tol = tol

    @classmethod
    def from_meta(cls, meta, **params) -> TDOALocalizer:
        """A localizer configured from the ``meta`` of a ``tdoa`` table: ``meta["anchors"]``, whose
        anchor 0 must be the reference (``meta["reference_anchor"]``, 0 if absent). ``params``
        override or add constructor arguments, e.g. ``TDOALocalizer.from_meta(meta, sigma=0.1)``."""
        anchors = _meta_anchors(meta, "tdoa", cls.__name__)
        if int(meta.get("reference_anchor", 0)) != 0:
            raise ValueError(f"TDOALocalizer takes differences to anchor 0, but meta['reference_anchor'] = "
                             f"{meta['reference_anchor']}: reorder the anchors (and X) so the reference comes first")
        return cls(**{"anchors": anchors, **params})

    def _check_input(self, X):
        anchors = _as_anchors(self.anchors, type(self).__name__)
        if len(anchors) < 2:
            raise ValueError("TDoA needs at least 2 anchors")
        if X.ndim != 2 or X.shape[1] != len(anchors) - 1:
            raise ValueError(f"X must hold range differences to anchor 0, shape (N, {len(anchors) - 1}); "
                             f"got {X.shape}")
        if X.dtype.kind == "c":
            raise ValueError("range differences must be real-valued")

    def _fit(self, X, pos, floor, building):
        anchors = _as_anchors(self.anchors, type(self).__name__)
        self._check_labels(pos, anchors)
        self._clear_calibration()
        if self.calibrate:
            dist = np.sqrt(np.sum(np.square(pos[:, None, :] - anchors[None]), axis=2))
            bias, sigma = _calibration(np.asarray(X, np.float64) - (dist[:, 1:] - dist[:, :1]))
            self.bias_, self.sigma_ = bias, sigma / np.sqrt(2.0)  # a difference carries two range errors

    def _localize(self, X):
        anchors = _as_anchors(self.anchors, type(self).__name__)
        D = anchors.shape[1]
        z = np.asarray(X, dtype=np.float64) - getattr(self, "bias_", 0.0)
        x = chan_tdoa(z, anchors)
        rho = np.concatenate([np.where(np.isfinite(z).any(axis=1, keepdims=True), 0.0, np.nan), z], axis=1)
        model = _range_model(anchors, rho, offset=True)
        have = np.isfinite(rho)

        def start(pos):  # [position, offset]: the offset that best explains rho from pos
            gap = rho - np.sqrt(np.sum(np.square(pos[:, None, :] - anchors[None]), axis=2))
            with np.errstate(invalid="ignore", divide="ignore"):
                b = np.sum(np.where(have, gap, 0.0), axis=1, keepdims=True) / have.sum(axis=1, keepdims=True)
            return np.concatenate([pos, b], axis=1)

        xb = start(x)
        if self.refine:
            # second start: the anchors' centroid, for every row with the D differences a position
            # needs, also where the closed form is singular (a target equidistant from all anchors,
            # where every difference is 0, or a noisy row that makes Chan's step 1 singular)
            start2 = start(np.broadcast_to(anchors.mean(axis=0), x.shape))
            start2[np.isfinite(z).sum(axis=1) < D] = np.nan
            xb = _refine([xb, start2], model, max_iter=self.max_iter, tol=self.tol)
        spread = np.full(len(z), np.nan)
        rows = np.flatnonzero(np.all(np.isfinite(xb), axis=1))
        if rows.size:
            e, J, valid = model(xb[rows], rows, True)
            w = valid.astype(np.float64)
            gdop = np.sqrt(_inv_trace(np.einsum("nm,nmi,nmj->nij", w, J, J), D))  # position CRLB / sigma
            spread[rows] = self._noise(_residual_sigma(e, w, D + 1)) * gdop
            if self.refine:
                # the data must determine the position there (a start Gauss-Newton could not leave,
                # e.g. the centroid of a degenerate geometry, is no fix), and a far estimate must be resolved
                size = float(np.sqrt(np.mean(np.sum(np.square(anchors - anchors.mean(axis=0)), axis=1))))
                known = self._noise(np.full(len(rows), np.nan))  # sigma or the calibrated sigma_, else NaN
                drop = ~np.isfinite(gdop) | _far_field(xb[rows, :D], anchors, gdop, known,
                                                      _exact_fit(e, w, D + 1, 1.0 + size))
                xb[rows[drop]], spread[rows[drop]] = np.nan, np.nan
        return Prediction(xb[:, :D], spread=spread)


class WeightedCentroidLocalizer(GeometricLocalizer):
    """Weighted centroid of the anchors that hear the target.

    ``x = sum_i w_i a_i / sum_i w_i`` over the anchors with a (non-NaN) measurement:

      ``weights="power"``     X is RSSI in dBm and ``w_i = p_i^degree`` with ``p_i`` the linear
                              received power (mW). Under a log-distance model ``p ~ d^-n``, so
                              ``1 / d^g`` weights with ranges taken from RSSI are the power
                              weights with ``degree = g / n``.
      ``weights="distance"``  X holds distances (metres) and ``w_i = 1 / d_i^degree`` (Blumenthal
                              et al. 2007); a zero (or negative) distance puts the estimate on
                              that anchor.
      ``weights="uniform"``   plain centroid of the anchors heard (Bulusu et al. 2000).

    ``k`` keeps only the k largest weights per sample (None = every anchor heard). Rows with no
    measurement return NaN. ``Prediction.spread`` = weighted RMS distance of the anchors from
    the estimate (as for k-NN). Nothing is learned: ``fit`` only records the input size (``fit(X)``
    needs no positions), and ``localize`` works without ``fit``.

    Parameters
    ----------
    anchors : (A, D) array of anchor (AP, beacon) positions.
    weights : "power", "distance" or "uniform".
    degree : exponent ``g`` of the weights.
    k : number of strongest anchors used per sample, or None for all.

    References
    ----------
    J. Blumenthal, R. Grossmann, F. Golatowski, D. Timmermann, "Weighted centroid localization
    in Zigbee-based sensor networks", IEEE International Symposium on Intelligent Signal
    Processing (WISP), 2007. DOI 10.1109/WISP.2007.4447528.
    N. Bulusu, J. Heidemann, D. Estrin, "GPS-less low-cost outdoor localization for very small
    devices", IEEE Personal Communications 7(5), 2000. DOI 10.1109/98.878533.
    """

    _labels_optional = True  # nothing is ever learned from positions

    def __init__(self, anchors=None, weights: str = "power", degree: float = 1.0, k=None):
        self.anchors = anchors
        self.weights = weights
        self.degree = degree
        self.k = k

    def _check_input(self, X):
        anchors = _as_anchors(self.anchors, type(self).__name__)
        if X.ndim != 2 or X.shape[1] != len(anchors):
            raise ValueError(f"X must hold one measurement per anchor, shape (N, {len(anchors)}); got {X.shape}")
        if X.dtype.kind == "c":
            raise ValueError("measurements must be real-valued")

    def _check_options(self):
        if self.weights not in ("power", "distance", "uniform"):
            raise ValueError(f"weights must be 'power', 'distance' or 'uniform', got {self.weights!r}")
        if self.k is not None and int(self.k) < 1:
            raise ValueError(f"k must be a positive integer or None, got {self.k!r}")

    def _fit(self, X, pos, floor, building):
        self._check_options()
        self._check_labels(pos, _as_anchors(self.anchors, type(self).__name__))

    def centroid_weights(self, X) -> np.ndarray:
        """Normalised weights (N, A) of each anchor; rows sum to 1 (NaN rows: nothing heard).
        ``X`` is an array or a SampleTable."""
        return self._weights(self._validate(_unpack(X)[0]))

    def _weights(self, X) -> np.ndarray:
        self._check_options()
        X = np.asarray(X, dtype=np.float64)
        heard = np.isfinite(X)
        g = float(self.degree)
        if self.weights == "uniform":
            w = heard.astype(np.float64)
        elif self.weights == "power":  # 10^(g x / 10), scaled by the row maximum to avoid underflow
            top = np.max(np.where(heard, X, -np.inf), axis=1, keepdims=True)
            with np.errstate(invalid="ignore"):
                w = np.where(heard, 10.0 ** (g * (X - np.where(np.isfinite(top), top, 0.0)) / 10.0), 0.0)
        else:
            d = np.where(heard, np.maximum(X, 0.0), np.inf)
            exact = heard & (d <= 0.0)
            with np.errstate(divide="ignore"):
                w = np.where(heard, np.power(np.where(heard, d, 1.0), -g), 0.0)
            hit = exact.any(axis=1)
            first = np.argmax(exact, axis=1)
            w[hit] = 0.0
            w[np.flatnonzero(hit), first[hit]] = 1.0
        if self.k is not None and int(self.k) < X.shape[1]:
            order = np.argsort(-w, axis=1, kind="stable")  # ties keep the anchor order
            drop = order[:, int(self.k):]
            np.put_along_axis(w, drop, 0.0, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return w / w.sum(axis=1, keepdims=True)

    def _localize(self, X):
        anchors = _as_anchors(self.anchors, type(self).__name__)
        w = self._weights(X)
        empty = ~np.isfinite(w).all(axis=1)
        w = np.where(empty[:, None], 0.0, w)
        pos = w @ anchors
        spread = np.sqrt(np.einsum("na,na->n", w, np.sum(np.square(anchors[None] - pos[:, None, :]), axis=2)))
        pos[empty] = np.nan
        spread[empty] = np.nan
        return Prediction(pos, spread=spread)


__all__ = ["GeometricLocalizer", "TDOALocalizer", "TrilaterationLocalizer", "WeightedCentroidLocalizer",
           "chan_tdoa", "gdop", "linear_trilateration"]
