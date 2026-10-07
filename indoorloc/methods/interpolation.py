"""Radio-map construction: interpolate sparse fingerprints to new positions (numpy only).

A radio map is a function from position to fingerprint (one RSSI column per AP). These
helpers densify a sparse survey so that a fingerprinting method (k-NN, ...) can be trained
on a regular grid:

    idw(points, values, query)            inverse-distance weighting (Shepard 1968)
    gaussian_rbf(points, values, query)   Gaussian radial-basis-function interpolation
    RadioMapInterpolator                  the same as an sklearn-style regressor: fit(pos, X) -> predict(pos)
    grid_points(bounds, step)             a regular grid;  densify(X, pos, step=...) grid + interpolation

NaN in ``values`` means "not observed at this point": every column (AP) is interpolated from
the points that observed it only, so a column is never pulled towards a fill value. To treat
"not heard" as a weak signal instead, fill it first (``FillMissing(-104)``). Interpolate each
floor separately; duplicated positions are averaged first.
"""
from __future__ import annotations

import numpy as np

from ..core import Estimator


def _points(points, name: str) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    p = p[:, None] if p.ndim == 1 else p
    if p.ndim != 2 or not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be a finite (N, D) array, got shape {p.shape}")
    return p


def _values(values, n: int) -> np.ndarray:
    v = np.asarray(values)
    if v.dtype.kind == "c":
        raise ValueError("Complex data not supported: values must be real (e.g. RSSI in dBm)")
    v = v.astype(np.float64)
    v = v[:, None] if v.ndim == 1 else v
    if v.ndim != 2 or len(v) != n:
        raise ValueError(f"values must be (N, F) with N={n}, got shape {v.shape}")
    if np.isinf(v).any():
        raise ValueError("values contain inf")
    return v


def _check_positions(X, owner: str) -> np.ndarray:
    """sklearn-style checks of an estimator input: dense, real, 2-D, non-empty and finite."""
    if hasattr(X, "toarray"):
        raise TypeError("sparse input is not supported; pass a dense array (X.toarray())")
    X = np.asarray(X)
    if X.dtype.kind == "c":
        raise ValueError(f"Complex data not supported by {owner}: X holds real coordinates")
    if X.ndim != 2:
        raise ValueError(f"Expected a 2-D array of positions (P, D), got shape {X.shape}. Reshape your data: "
                         "x[:, None] for 1-D coordinates, x[None, :] for a single position")
    for axis, what in ((0, "sample"), (1, "feature")):
        if X.shape[axis] == 0:
            raise ValueError(f"Found array with 0 {what}(s) (shape={X.shape}) while a minimum of 1 is required.")
    X = X.astype(np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("X (positions) contains NaN or inf")
    return X


def _unique(points: np.ndarray, values: np.ndarray):
    """Merge duplicated positions: NaN-aware mean of their values."""
    uniq, inverse = np.unique(points, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if len(uniq) == len(points):
        order = np.argsort(inverse)
        return uniq, values[order]
    obs = np.isfinite(values)
    total = np.zeros((len(uniq), values.shape[1]))
    count = np.zeros((len(uniq), values.shape[1]))
    np.add.at(total, inverse, np.where(obs, values, 0.0))
    np.add.at(count, inverse, obs)
    with np.errstate(invalid="ignore", divide="ignore"):
        return uniq, np.where(count > 0, total / count, np.nan)


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.square(a[:, None, :] - b[None]), axis=2)


def _rows_per_chunk(n_points: int, dims: int) -> int:
    return max(1, 4_000_000 // max(n_points * dims, 1))


def idw(points, values, query, *, power: float = 2.0, k=None) -> np.ndarray:
    """Shepard's inverse-distance-weighted interpolation.

    ``v(q) = sum_i w_i v_i / sum_i w_i`` with ``w_i = 1 / |q - p_i|^power`` over the points that
    observed the column (``k``: only the k nearest points, the "modified" local variant;
    ties broken by point order). Exact at the data points. Returns (Q, F), or (Q,) for 1-D
    ``values``; NaN where no (nearest) point observed the column.

    References: D. Shepard, "A two-dimensional interpolation function for irregularly-spaced
    data", Proceedings of the 23rd ACM National Conference, 1968. DOI 10.1145/800186.810616.
    """
    p = _points(points, "points")
    v = _values(values, len(p))
    q = _points(query, "query")
    if q.shape[1] != p.shape[1]:
        raise ValueError(f"query is {q.shape[1]}-D but points are {p.shape[1]}-D")
    out = _idw(*_unique(p, v), q, float(power), k)
    return out[:, 0] if np.ndim(values) == 1 else out


def _idw(p: np.ndarray, v: np.ndarray, q: np.ndarray, power: float, k) -> np.ndarray:
    """``idw`` on validated arrays with distinct points ``p`` (P, D), ``v`` (P, F), ``q`` (Q, D)."""
    obs = np.isfinite(v)
    v0 = np.where(obs, v, 0.0)
    m = obs.astype(np.float64)
    out = np.empty((len(q), v.shape[1]))
    step = _rows_per_chunk(len(p), p.shape[1])
    for s in range(0, len(q), step):
        d2 = _sq_dist(q[s:s + step], p)
        exact = (d2 == 0.0).astype(np.float64)
        with np.errstate(divide="ignore"):
            w = np.where(d2 > 0.0, d2 ** (-0.5 * float(power)), 0.0)
        if k is not None and int(k) < len(p):
            far = np.argsort(d2, axis=1, kind="stable")[:, int(k):]
            np.put_along_axis(w, far, 0.0, axis=1)
            np.put_along_axis(exact, far, 0.0, axis=1)
        num_e, den_e = exact @ v0, exact @ m
        num, den = w @ v0, w @ m
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + step] = np.where(den_e > 0, num_e / np.where(den_e > 0, den_e, 1.0),
                                       np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan))
    return out


def _default_length_scale(points: np.ndarray) -> float:
    """Mean point spacing ``(bounding-box volume / P)^(1/D)``, at least the median
    nearest-neighbour distance (which covers flat, e.g. corridor-shaped, surveys)."""
    if len(points) < 2:
        return 1.0
    nearest = np.empty(len(points))
    step = _rows_per_chunk(len(points), points.shape[1])
    for s in range(0, len(points), step):
        d2 = _sq_dist(points[s:s + step], points)
        d2[np.arange(len(d2)), np.arange(s, s + len(d2))] = np.inf
        nearest[s:s + step] = np.sqrt(d2.min(axis=1))
    span = points.max(axis=0) - points.min(axis=0)
    spacing = float(np.prod(span) / len(points)) ** (1.0 / points.shape[1])
    return max(spacing, float(np.median(nearest)))


def _solve_spd(K: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve ``K X = B`` for symmetric positive (semi)definite ``K`` by Cholesky, adding the
    smallest jitter from 0, 1e-12, ..., 1e-6 (times the diagonal) that makes it succeed."""
    scale = float(np.mean(np.diag(K))) if len(K) else 1.0
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            L = np.linalg.cholesky(K + jitter * scale * np.eye(len(K)))
        except np.linalg.LinAlgError:
            continue
        return np.linalg.solve(L.T, np.linalg.solve(L, B))
    return np.linalg.lstsq(K, B, rcond=None)[0]


_SMOOTHING_GRID = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def _rbf_groups(obs: np.ndarray):
    """Columns sharing the same set of observing points: (row indices, column indices) pairs."""
    patterns, group = np.unique(obs.T, axis=0, return_inverse=True)
    group = group.reshape(-1)
    return [(np.flatnonzero(pattern), np.flatnonzero(group == g)) for g, pattern in enumerate(patterns)
            if pattern.any()]


def _loo_smoothing(K: np.ndarray, centred: np.ndarray, groups) -> float:
    """Ridge term with the smallest leave-one-out squared error, pooled over all columns.

    For ``(K + s I) c = v`` the leave-one-out residual at point i is ``c_i / [(K + s I)^-1]_ii``
    (closed form, one eigendecomposition per group of columns).
    """
    sse = np.zeros(len(_SMOOTHING_GRID))
    for rows, cols in groups:
        ev, U = np.linalg.eigh(K[np.ix_(rows, rows)])
        ev = np.maximum(ev, 0.0)
        Utv = U.T @ centred[np.ix_(rows, cols)]
        U2 = np.square(U)
        for i, sm in enumerate(_SMOOTHING_GRID):
            inv = 1.0 / (ev + sm)
            coef = U @ (inv[:, None] * Utv)
            sse[i] += np.sum(np.square(coef / (U2 @ inv)[:, None]))
    return float(_SMOOTHING_GRID[int(np.argmin(sse))])


def _rbf_fit(p: np.ndarray, v: np.ndarray, length_scale: float, smoothing):
    """Coefficients (P, F), column means (F,) and the ridge term of a Gaussian RBF fit per column."""
    obs = np.isfinite(v)
    count = obs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, np.sum(np.where(obs, v, 0.0), axis=0) / np.maximum(count, 1), np.nan)
    centred = np.where(obs, v - mean, 0.0)
    coef = np.zeros(v.shape)
    K = np.exp(-_sq_dist(p, p) / (2.0 * length_scale ** 2))
    groups = _rbf_groups(obs)
    if isinstance(smoothing, str):
        if smoothing != "loo":
            raise ValueError(f"smoothing must be a number >= 0 or 'loo', got {smoothing!r}")
        smoothing = _loo_smoothing(K, centred, groups)
    elif not float(smoothing) >= 0:
        raise ValueError(f"smoothing must be a number >= 0 or 'loo', got {smoothing!r}")
    for rows, cols in groups:
        Kg = K[np.ix_(rows, rows)] + float(smoothing) * np.eye(rows.size)
        coef[np.ix_(rows, cols)] = _solve_spd(Kg, centred[np.ix_(rows, cols)])
    return coef, mean, float(smoothing)


def gaussian_rbf(points, values, query, *, length_scale=None, smoothing="loo") -> np.ndarray:
    """Gaussian radial-basis-function interpolation of each column.

    ``v(q) = mean + sum_i c_i exp(-|q - p_i|^2 / (2 l^2))`` with ``(K + smoothing I) c = v - mean``
    over the points that observed the column; far from the data the map returns to the
    column mean. ``smoothing=0`` interpolates exactly (for noise-free data only: noisy scans
    taken close together make exact interpolation oscillate wildly); a positive value gives
    a ridge (smoothing) fit; ``"loo"`` (default) picks it from 1e-8 ... 10 by the closed-form
    leave-one-out error pooled over all columns (Rippa 1999). ``length_scale`` None = the
    mean point spacing ``(bounding-box volume / P)^(1/D)`` (at least the median
    nearest-neighbour distance). Costs O(P^2) memory and O(P^3) time: meant for sparse
    surveys (up to a few thousand distinct points); use ``idw`` with ``k`` for larger maps.

    References: R. L. Hardy, "Multiquadric equations of topography and other irregular
    surfaces", Journal of Geophysical Research 76(8), 1971. DOI 10.1029/JB076i008p01905.
    M. D. Buhmann, "Radial Basis Functions: Theory and Implementations", Cambridge University
    Press, 2003. DOI 10.1017/CBO9780511543241. S. Rippa, "An algorithm for selecting a good
    value for the parameter c in radial basis function interpolation", Advances in
    Computational Mathematics 11, 1999. DOI 10.1023/A:1018975909870.
    """
    model = RadioMapInterpolator("rbf", length_scale=length_scale, smoothing=smoothing)
    return model.fit(_points(points, "points"), values).predict(_points(query, "query"))


def grid_points(bounds, step) -> np.ndarray:
    """Regular grid over ``bounds = [[min...], [max...]]`` (2, D) with spacing ``step`` (scalar or
    (D,)), both ends included (up to rounding). Returns (G, D), the last axis varying fastest."""
    b = np.asarray(bounds, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != 2 or np.any(b[1] < b[0]):
        raise ValueError(f"bounds must be [[min...], [max...]] with max >= min, got {b.tolist()}")
    st = np.broadcast_to(np.asarray(step, dtype=np.float64), b.shape[1:])
    if np.any(st <= 0):
        raise ValueError(f"step must be positive, got {step}")
    axes = [lo + st_d * np.arange(int(np.floor((hi - lo) / st_d + 1e-9)) + 1) for lo, hi, st_d in zip(b[0], b[1], st)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, b.shape[1])


class RadioMapInterpolator(Estimator):
    """Radio map as a regressor: ``fit(positions, fingerprints)``, ``predict(positions)``.

    ``method="idw"`` (Shepard 1968; ``power``, ``k``) or ``"rbf"`` (Gaussian RBF;
    ``length_scale``, ``smoothing``). Note the roles: the positions are the input and the
    fingerprints the target (the inverse of a localizer). NaN in the fingerprints = not
    observed (see the module docstring); positions must be finite. ``score`` is the
    coefficient of determination R^2 averaged over the columns, on observed entries, so
    ``GridSearchCV`` can tune ``length_scale`` or ``power``.

    Parameters
    ----------
    method : "idw" or "rbf".
    power : IDW distance exponent.
    k : IDW neighbourhood size (None = all points).
    length_scale : RBF width in coordinate units (None = mean point spacing, see ``gaussian_rbf``).
    smoothing : RBF ridge term: a number (0 = exact interpolation) or ``"loo"`` (chosen by
        leave-one-out error; the value used is ``smoothing_``).

    References
    ----------
    D. Shepard, "A two-dimensional interpolation function for irregularly-spaced data",
    Proceedings of the 23rd ACM National Conference, 1968. DOI 10.1145/800186.810616.
    M. D. Buhmann, "Radial Basis Functions: Theory and Implementations", Cambridge University
    Press, 2003. DOI 10.1017/CBO9780511543241.
    J. Talvitie, M. Renfors, E. S. Lohan, "Distance-based interpolation and extrapolation
    methods for RSS-based localization with indoor wireless signals", IEEE Transactions on
    Vehicular Technology 64(4), 2015. DOI 10.1109/TVT.2015.2397598.
    S. Rippa, "An algorithm for selecting a good value for the parameter c in radial basis
    function interpolation", Advances in Computational Mathematics 11, 1999.
    DOI 10.1023/A:1018975909870.
    """

    _estimator_type = "regressor"
    _allow_nan = False  # sklearn's tag is about X, the positions; NaN fingerprints (y) are fine

    def __init__(self, method: str = "idw", power: float = 2.0, k=None, length_scale=None, smoothing="loo"):
        self.method = method
        self.power = power
        self.k = k
        self.length_scale = length_scale
        self.smoothing = smoothing

    def fit(self, X, y=None):
        """``X`` = positions (P, D), ``y`` = fingerprints (P, F) or (P,) with NaN = not observed."""
        if self.method not in ("idw", "rbf"):
            raise ValueError(f"method must be 'idw' or 'rbf', got {self.method!r}")
        p = _check_positions(X, type(self).__name__)
        if y is None:
            raise ValueError(f"{type(self).__name__} requires y to be passed, but the target y is None "
                             "(pass the fingerprints observed at the positions X)")
        y = np.asarray(y)
        v = _values(y, len(p))
        if not np.isfinite(v).any():
            raise ValueError("y contains no observed value (all NaN)")
        p, v = _unique(p, v)
        self.points_, self.values_ = p, v
        self.n_features_in_ = p.shape[1]
        self.n_outputs_ = v.shape[1]
        self.target_ndim_ = y.ndim
        if self.method == "rbf":
            ls = _default_length_scale(p) if self.length_scale is None else float(self.length_scale)
            if not ls > 0:
                raise ValueError(f"length_scale must be positive, got {ls}")
            self.length_scale_ = ls
            self.coef_, self.mean_, self.smoothing_ = _rbf_fit(p, v, ls, self.smoothing)
        return self

    def predict(self, X) -> np.ndarray:
        """Interpolated fingerprints (Q, F) at positions ``X`` (Q, D)."""
        self._check_fitted()
        q = _check_positions(X, type(self).__name__)
        if q.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {q.shape[1]} features, but RadioMapInterpolator is expecting "
                             f"{self.n_features_in_} features as input")
        if self.method == "idw":
            out = _idw(self.points_, self.values_, q, float(self.power), self.k)
        else:
            out = np.empty((len(q), self.n_outputs_))
            step = _rows_per_chunk(len(self.points_), q.shape[1])
            for s in range(0, len(q), step):
                Kq = np.exp(-_sq_dist(q[s:s + step], self.points_) / (2.0 * self.length_scale_ ** 2))
                out[s:s + step] = Kq @ self.coef_ + self.mean_
        return out[:, 0] if self.target_ndim_ == 1 else out

    def score(self, X, y) -> float:
        """Mean over columns of R^2 on the observed entries (columns with < 2 of them skipped)."""
        pred = np.asarray(self.predict(X), np.float64)
        pred = pred.reshape(len(pred), -1)
        true = _values(y, len(pred))
        ok = np.isfinite(true) & np.isfinite(pred)
        cnt = ok.sum(axis=0)
        mu = np.sum(np.where(ok, true, 0.0), axis=0) / np.maximum(cnt, 1)
        ss_tot = np.sum(np.where(ok, np.square(true - mu), 0.0), axis=0)
        ss_res = np.sum(np.where(ok, np.square(true - pred), 0.0), axis=0)
        use = (cnt >= 2) & (ss_tot > 0)
        if not use.any():
            return float("nan")
        return float(np.mean(1.0 - ss_res[use] / ss_tot[use]))


def densify(X, pos, *, step, method: str = "idw", max_distance=None, bounds=None, **params):
    """Interpolate a sparse survey onto a regular grid: returns ``(X_grid, pos_grid)``.

    ``X`` (N, F) fingerprints at ``pos`` (N, D); the grid spans ``bounds`` (default: the
    bounding box of ``pos``) with spacing ``step``. ``max_distance`` keeps only grid points
    within that distance of a surveyed position (no extrapolation into unsurveyed space);
    None keeps the whole grid. ``params`` go to ``RadioMapInterpolator`` (``power``, ``k``,
    ``length_scale``, ``smoothing``). Concatenate with the survey to train a fingerprinting
    method on the densified map.
    """
    p = _points(pos, "pos")
    grid = grid_points(np.stack([p.min(axis=0), p.max(axis=0)]) if bounds is None else bounds, step)
    if max_distance is not None:
        keep = np.zeros(len(grid), dtype=bool)
        chunk = _rows_per_chunk(len(p), p.shape[1])
        for s in range(0, len(grid), chunk):
            keep[s:s + chunk] = _sq_dist(grid[s:s + chunk], p).min(axis=1) <= float(max_distance) ** 2
        grid = grid[keep]
    model = RadioMapInterpolator(method, **params).fit(p, np.asarray(X, dtype=np.float64).reshape(len(p), -1))
    return model.predict(grid).reshape(len(grid), -1), grid


__all__ = ["RadioMapInterpolator", "densify", "gaussian_rbf", "grid_points", "idw"]
