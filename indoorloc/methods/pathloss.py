"""Log-distance path-loss model: calibration from labelled RSSI and maximum-likelihood localization.

    RSSI_j(x) = P0_j - 10 n_j log10(max(|x - a_j|, d_min) / d0) + e_j,   e_j ~ N(0, sigma_j^2)

``PathLossLocalizer`` fits ``P0`` (RSSI at the reference distance ``d0``), the exponent ``n``
and the shadowing std ``sigma`` of every anchor (AP, beacon) from labelled training scans
and, when the anchor positions are unknown, the positions too. A query scan is then
localized by maximum likelihood (numpy only).
"""
from __future__ import annotations

import numpy as np

from ..core import Prediction
from .geometric import GeometricLocalizer, _as_anchors, _gauss_newton, _residual_sigma, _spread

_LN10 = np.log(10.0)


def _box_grid(low, high, per_axis: int) -> np.ndarray:
    """``per_axis`` evenly spaced values per axis between ``low`` and ``high`` (inclusive), as (G, D)."""
    axes = [np.linspace(a, b, per_axis) for a, b in zip(low, high)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def log_distance_rssi(positions, anchors, p0, exponent, *, d0: float = 1.0, min_distance: float = 0.1) -> np.ndarray:
    """Mean RSSI (N, A) in dBm of the log-distance model at ``positions`` (N, D).

    ``p0`` and ``exponent`` are scalars or (A,) arrays; distances are clipped at
    ``min_distance`` (the model is not valid in the near field).

    References: T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed.,
    Prentice Hall, 2002 (section 4.9, log-distance path loss). ISBN 0-13-042232-0.
    """
    x = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    a = _as_anchors(anchors, "log_distance_rssi")
    d = np.sqrt(np.sum(np.square(x[:, None, :] - a[None]), axis=2))
    return np.asarray(p0, np.float64) - 10.0 * np.asarray(exponent, np.float64) * np.log10(
        np.maximum(d, min_distance) / d0)


class PathLossLocalizer(GeometricLocalizer):
    """Maximum-likelihood localization with a log-distance path-loss model per anchor.

    Fitting (``fit(X, positions)``, X = RSSI (N, A) dBm with NaN = not heard):

    * anchors known: per anchor, ``P0`` and ``n`` by linear least squares of RSSI on
      ``-10 log10(d / d0)`` (closed form), ``n`` clipped to ``exponent_bounds``;
    * ``anchors=None``: the anchor position is estimated jointly with ``P0`` and ``n`` by
      Gauss-Newton on the same residuals (as in EZ, Chintalapudi et al. 2010, but with
      labelled positions), from two starts: the power-weighted centroid of the positions
      where the anchor was heard, and the best point of a coarse grid (about 1000 points over
      the training bounding box enlarged by half its size on every side) on the cost with
      ``P0`` and ``n`` solved in closed form (variable projection); the lower cost wins;
    * ``per_anchor=False``: one ``(P0, n)`` shared by all anchors (fitted after the positions
      when those are estimated).

    ``p0``/``exponent`` given as numbers (or (A,) arrays) are held fixed; ``sigma`` (dB) given
    fixes the shadowing std, otherwise it is the residual std of each anchor's fit. Anchors
    heard in fewer than ``min_readings`` training scans are not used (``used_`` is False)
    when something has to be learned for them. With anchors, ``p0`` and ``exponent`` all
    given, the model needs no fit; ``fit`` then only estimates ``sigma_`` (the pooled
    residual std for anchors with fewer than ``min_readings`` readings) and uses every anchor.

    Localization maximises the Gaussian likelihood of the heard anchors,
    ``sum_j ((RSSI_j - P0_j + 10 n_j log10(d_j / d0)) / sigma_j)^2``, by Gauss-Newton from
    ``n_starts`` deterministic starts (the power-weighted centroid of the heard anchors and the
    strongest anchors, pulled 10 % towards that centroid), keeping the lowest cost. Missing
    readings are ignored (no censoring model). Rows hearing fewer than D + 1 used anchors
    return NaN. ``Prediction.spread`` = ``sqrt(trace(F^-1))`` with ``F`` the Fisher information
    of the model at the estimate (the RSS Cramer-Rao bound, Patwari et al. 2003); without a
    known ``sigma`` it is scaled by the per-sample residual std. ``predict_rssi(positions)``
    returns the fitted radio map, e.g. to build a RADAR-style model-based fingerprint database.

    Parameters
    ----------
    anchors : (A, D) anchor positions, or None to estimate them in ``fit``.
    p0 : RSSI at ``d0`` (dBm), scalar or (A,); None = fitted.
    exponent : path-loss exponent, scalar or (A,); None = fitted.
    per_anchor : fit ``P0``, ``n`` and ``sigma`` per anchor (True) or shared (False).
    sigma : shadowing std in dB, scalar or (A,); None = fitted (or estimated per sample).
    d0 : reference distance in coordinate units (1 m).
    min_distance : distances are clipped here (coordinate units).
    exponent_bounds : (low, high) bounds of fitted exponents.
    min_readings : minimum training readings for an anchor to be used.
    bounds : None (unconstrained maximum likelihood), ``"fit"`` (the bounding box of the training
        positions, learned in ``fit`` as ``bounds_``) or ``[[min...], [max...]]``: estimates are
        kept inside this box (maximum a posteriori under a uniform prior on the box).
    n_starts : number of Gauss-Newton starts in localization.
    max_iter, tol : Gauss-Newton iterations and relative step tolerance.

    Deviation from RADAR: no wall-attenuation term (walls are not part of the input).

    References
    ----------
    T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., Prentice
    Hall, 2002. ISBN 0-13-042232-0.
    S. Y. Seidel, T. S. Rappaport, "914 MHz path loss prediction models for indoor wireless
    communications in multifloored buildings", IEEE Transactions on Antennas and Propagation
    40(2), 1992. DOI 10.1109/8.127405.
    P. Bahl, V. N. Padmanabhan, "RADAR: an in-building RF-based user location and tracking
    system", IEEE INFOCOM, 2000. DOI 10.1109/INFCOM.2000.832252.
    K. Chintalapudi, A. Padmanabha Iyer, V. N. Padmanabhan, "Indoor localization without the
    pain", ACM MobiCom, 2010. DOI 10.1145/1859995.1860016.
    N. Patwari, A. O. Hero, M. Perkins, N. S. Correal, R. J. O'Dea, "Relative location
    estimation in wireless sensor networks", IEEE Transactions on Signal Processing 51(8),
    2003. DOI 10.1109/TSP.2003.814469.
    """

    def __init__(self, anchors=None, p0=None, exponent=None, per_anchor: bool = True, sigma=None,
                 d0: float = 1.0, min_distance: float = 0.1, exponent_bounds=(1.0, 6.0), min_readings: int = 5,
                 bounds=None, n_starts: int = 4, max_iter: int = 50, tol: float = 1e-9):
        self.anchors = anchors
        self.p0 = p0
        self.exponent = exponent
        self.per_anchor = per_anchor
        self.sigma = sigma
        self.d0 = d0
        self.min_distance = min_distance
        self.exponent_bounds = exponent_bounds
        self.min_readings = min_readings
        self.bounds = bounds
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.tol = tol

    @property
    def _requires_fit(self) -> bool:
        return self.anchors is None or self.p0 is None or self.exponent is None or isinstance(self.bounds, str)

    def _check_input(self, X):
        if X.dtype.kind == "c":
            raise ValueError("RSSI must be real-valued")
        if X.ndim != 2:
            raise ValueError(f"X must be RSSI of shape (N, A), got {X.shape}")
        if self.anchors is not None and X.shape[1] != len(_as_anchors(self.anchors, type(self).__name__)):
            raise ValueError(f"X has {X.shape[1]} columns but there are {len(self.anchors)} anchors")

    # -- fitting -----------------------------------------------------------------------------

    def _per_anchor(self, value, A: int, name: str):
        if value is None:
            return None
        v = np.asarray(value, dtype=np.float64)
        if v.ndim == 0:
            return np.full(A, float(v))
        if v.shape != (A,):
            raise ValueError(f"{name} must be a scalar or an ({A},) array, got shape {v.shape}")
        return v

    def _regress(self, y, L, valid, p0, n, pooled: bool):
        """Least-squares (P0, n) of ``y = P0 + n L`` per column (or pooled), with fixed values
        kept and ``n`` clipped to the bounds; returns p0, n, residual sum of squares, count."""
        lo, hi = self.exponent_bounds
        v = valid.astype(np.float64)
        yz, Lz = np.where(valid, y, 0.0), np.where(valid, L, 0.0)
        s = lambda a: a.sum(axis=None if pooled else 0)  # noqa: E731
        c, sL, sLL, sy, sLy = s(v), s(Lz), s(Lz * Lz), s(yz), s(Lz * yz)
        with np.errstate(invalid="ignore", divide="ignore"):
            if n is None and p0 is None:
                det = c * sLL - sL * sL
                n_hat = np.where(det > 1e-12 * np.maximum(c * sLL, 1e-300), (c * sLy - sL * sy) / det, np.nan)
            elif n is None:
                n_hat = (sLy - p0 * sL) / sLL
            else:
                n_hat = n
            n_hat = np.clip(n_hat, lo, hi) if n is None else n_hat
            p0_hat = (sy - n_hat * sL) / c if p0 is None else p0
        res = np.where(valid, y - (p0_hat + n_hat * L), 0.0)
        return p0_hat, n_hat, s(res * res), c

    def _sigma(self, ss, c, n_free):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(ss / np.maximum(c - n_free, 1.0))

    def _fit(self, X, pos, floor, building):
        y = np.asarray(X, dtype=np.float64)
        A = y.shape[1]
        D = pos.shape[1]
        valid = np.isfinite(y)
        p0 = self._per_anchor(self.p0, A, "p0")
        n = self._per_anchor(self.exponent, A, "exponent")
        lo, hi = self.exponent_bounds
        if not 0 < lo <= hi:
            raise ValueError(f"exponent_bounds must satisfy 0 < low <= high, got {self.exponent_bounds}")
        if not self.per_anchor and (np.ndim(self.p0) > 0 or np.ndim(self.exponent) > 0):
            raise ValueError("per_anchor=False shares one model: give p0 and exponent as scalars (or None)")
        counts = valid.sum(axis=0)
        enough = counts >= max(int(self.min_readings), 1)
        # an anchor whose (position, P0, n) are all given is usable whatever its training count
        learn = self.anchors is None or p0 is None or n is None
        used = enough.copy() if learn else np.ones(A, dtype=bool)
        if self.anchors is None:
            anchors, p0_j, n_j = self._fit_positions(y, pos, valid & used, p0, n)
        else:
            anchors = _as_anchors(self.anchors, type(self).__name__)
            self._check_labels(pos, anchors)
            p0_j = n_j = None
        L = -10.0 * np.log10(np.maximum(np.sqrt(np.sum(np.square(pos[:, None, :] - anchors[None]), axis=2)),
                                        self.min_distance) / self.d0)
        if self.per_anchor:
            free = int(p0 is None) + int(n is None)
            if p0_j is None:
                p0_j, n_j, ss, c = self._regress(y, L, valid, p0, n, pooled=False)
                sig = self._sigma(ss, c, free)
            else:  # residual std with the degrees of freedom of the joint (position, P0, n) fit
                _, _, ss, c = self._regress(y, L, valid, p0_j, n_j, pooled=False)
                sig = self._sigma(ss, c, D + free)
        else:
            use = valid & used
            p_sh = None if p0 is None else float(p0[0])
            n_sh = None if n is None else float(n[0])
            p_, n_, ss, c = self._regress(y[use], L[use], np.ones(int(use.sum()), bool), p_sh, n_sh, pooled=True)
            sig = self._sigma(ss, c, int(p_sh is None) + int(n_sh is None))
            p0_j, n_j, sig = np.full(A, p_), np.full(A, n_), np.full(A, sig)
        if self.per_anchor and not learn:  # sigma of rarely heard anchors: the pooled residual std
            with np.errstate(invalid="ignore", divide="ignore"):
                sig = np.where(enough, sig, np.sqrt(np.sum(ss) / np.sum(c)))  # dof = count: nothing fitted
        used &= np.isfinite(p0_j) & np.isfinite(n_j)
        fixed_sigma = self._per_anchor(self.sigma, A, "sigma")
        self.anchors_ = anchors
        self.p0_ = np.where(used, p0_j, np.nan)
        self.exponent_ = np.where(used, n_j, np.nan)
        # a zero residual std (exact training data) would give infinite weights: floor at 1e-6 dB;
        # NaN (no training reading at all) makes localization estimate the scale per sample
        self.sigma_ = np.where(used, np.maximum(sig, 1e-6) if fixed_sigma is None else fixed_sigma, np.nan)
        self.used_ = used
        self.__dict__.pop("bounds_", None)
        if isinstance(self.bounds, str) and self.bounds == "fit":
            self.bounds_ = np.stack([pos.min(axis=0), pos.max(axis=0)])

    def _fit_positions(self, y, pos, valid, p0, n):
        """Joint (position, P0, n) of every anchor by Gauss-Newton on its own readings, from two
        starts (the power-weighted centroid of the positions where it is heard, and the best
        point of a coarse grid search on the variable-projection cost); the lower cost wins."""
        A = y.shape[1]
        D = pos.shape[1]
        lo, hi = self.exponent_bounds
        anchors = np.full((A, D), np.nan)
        p0_out, n_out = np.full(A, np.nan), np.full(A, np.nan)
        counts = valid.sum(axis=0)
        order = np.argsort(counts, kind="stable")
        order = order[counts[order] > 0]
        extent = np.maximum(pos.max(axis=0) - pos.min(axis=0), 1.0)
        per_axis = max(2, int(round(1000 ** (1.0 / D))))
        grid = _box_grid(pos.min(axis=0) - 0.5 * extent, pos.max(axis=0) + 0.5 * extent, per_axis)
        n_col = D + (p0 is None)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and (stop + 1 - start) * counts[order[stop]] <= 1_000_000:
                stop += 1
            cols = order[start:stop]
            start = stop
            width = int(counts[cols].max())
            idx = np.zeros((len(cols), width), dtype=np.intp)
            obs = np.zeros((len(cols), width), dtype=bool)
            for r, j in enumerate(cols):
                hit = np.flatnonzero(valid[:, j])
                idx[r, :len(hit)] = hit
                obs[r, :len(hit)] = True
            P = pos[idx]  # (C, W, D)
            Y = np.where(obs, y[idx, cols[:, None]], np.nan)
            fp0 = None if p0 is None else p0[cols]
            fn = None if n is None else n[cols]
            top = np.max(np.where(obs, Y, -np.inf), axis=1, keepdims=True)
            w = np.where(obs, 10.0 ** ((np.where(obs, Y, top) - top) / 10.0), 0.0)
            centroid = np.einsum("cw,cwd->cd", w, P) / w.sum(axis=1, keepdims=True)
            searched = np.stack([self._grid_start(grid, P[r, obs[r]], Y[r, obs[r]],
                                                  None if fp0 is None else fp0[r], None if fn is None else fn[r])
                                 for r in range(len(cols))])
            model = self._anchor_model(P, Y, obs, fp0, fn)

            def project(t, rows):
                if n is None:
                    t = t.copy()
                    t[:, n_col] = np.clip(t[:, n_col], lo, hi)
                return t

            best, best_cost = None, None
            for a0 in (centroid, searched):
                L0 = -10.0 * np.log10(np.maximum(np.sqrt(np.sum(np.square(P - a0[:, None]), axis=2)),
                                                 self.min_distance) / self.d0)
                p_init, n_init, _, _ = self._regress(Y.T, L0.T, obs.T, fp0, fn, pooled=False)
                theta = [a0]
                if p0 is None:
                    theta.append(np.where(np.isfinite(p_init), p_init, top[:, 0])[:, None])
                if n is None:
                    theta.append(np.nan_to_num(n_init, nan=2.0)[:, None])
                theta = _gauss_newton(np.concatenate(theta, axis=1), model, max_iter=max(self.max_iter, 100),
                                      tol=self.tol, project=project)
                e, _ = model(theta, np.arange(len(cols)), False)
                cost = np.sum(e * e, axis=1)
                if best is None:
                    best, best_cost = theta, cost
                else:
                    better = cost < best_cost  # ties keep the centroid start
                    best[better], best_cost[better] = theta[better], cost[better]
            anchors[cols] = best[:, :D]
            p0_out[cols] = best[:, D] if p0 is None else p0[cols]
            n_out[cols] = best[:, n_col] if n is None else n[cols]
        missing = ~np.all(np.isfinite(anchors), axis=1)
        anchors[missing] = pos.mean(axis=0)  # never heard: a placeholder, the anchor is unused
        return anchors, p0_out, n_out

    def _grid_start(self, grid, P, y, p0, n):
        """Grid point with the lowest variable-projection cost: (P0, n) solved in closed form at
        every candidate position (at most 1000 readings, evenly strided, for speed)."""
        stride = max(1, len(y) // 1000)
        P, y = P[::stride], y[::stride]
        d = np.sqrt(np.sum(np.square(grid[:, None, :] - P[None]), axis=2))  # (G, m)
        L = -10.0 * np.log10(np.maximum(d, self.min_distance) / self.d0).T  # (m, G)
        _, _, ss, _ = self._regress(np.broadcast_to(y[:, None], L.shape), L, np.ones(L.shape, bool), p0, n,
                                    pooled=False)
        return grid[int(np.argmin(np.where(np.isfinite(ss), ss, np.inf)))]

    def _anchor_model(self, P, Y, obs, p0, n):
        D = P.shape[2]
        d0, dmin = self.d0, self.min_distance

        def model(theta, rows, jac):
            a = theta[:, :D]
            k = D
            if p0 is None:
                p, k = theta[:, k], k + 1
            else:
                p = p0[rows]
            ex = theta[:, k] if n is None else n[rows]
            diff = a[:, None, :] - P[rows]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            dc = np.maximum(dist, dmin)
            logd = np.log10(dc / d0)
            pred = p[:, None] - 10.0 * ex[:, None] * logd
            valid = obs[rows]
            e = np.where(valid, Y[rows] - pred, 0.0)
            if not jac:
                return e, valid
            near = dist < dmin
            ga = np.where(near[..., None], 0.0, -(10.0 * ex[:, None, None] / _LN10) * diff / (dc * dc)[..., None])
            parts = [ga]
            if p0 is None:
                parts.append(np.ones(e.shape + (1,)))
            if n is None:
                parts.append(-10.0 * logd[..., None])
            J = np.concatenate(parts, axis=2)
            return e, np.where(valid[..., None], J, 0.0), valid

        return model

    # -- model -------------------------------------------------------------------------------

    def _model(self):
        """(anchors, p0, exponent, sigma or None, used) from the fit, or from the parameters."""
        if hasattr(self, "anchors_"):
            return self.anchors_, self.p0_, self.exponent_, self.sigma_, self.used_
        anchors = _as_anchors(self.anchors, type(self).__name__)
        A = len(anchors)
        sigma = self._per_anchor(self.sigma, A, "sigma")
        return (anchors, self._per_anchor(self.p0, A, "p0"), self._per_anchor(self.exponent, A, "exponent"),
                sigma, np.ones(A, dtype=bool))

    def predict_rssi(self, positions) -> np.ndarray:
        """Mean RSSI (N, A) of the model at ``positions`` (NaN for unused anchors): a radio map."""
        if self._requires_fit:
            self._check_fitted("anchors_")
        anchors, p0, n, _, used = self._model()
        out = log_distance_rssi(positions, anchors, p0, n, d0=self.d0, min_distance=self.min_distance)
        return np.where(used, out, np.nan)

    def _rssi_model(self, z, anchors, p0, n, inv_sigma):
        d0, dmin = self.d0, self.min_distance
        b = 10.0 * n / _LN10

        def model(x, rows, jac):
            diff = x[:, None, :] - anchors[None]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            dc = np.maximum(dist, dmin)
            pred = p0 - 10.0 * n * np.log10(dc / d0)
            zr = z[rows]
            valid = np.isfinite(zr)
            e = np.where(valid, (zr - pred) * inv_sigma, 0.0)
            if not jac:
                return e, valid
            J = np.where((dist < dmin)[..., None], 0.0, -(b * inv_sigma)[None, :, None] * diff / (dc * dc)[..., None])
            return e, np.where(valid[..., None], J, 0.0), valid

        return model

    def _box(self, D: int):
        """(2, D) box of admissible positions, or None."""
        if self.bounds is None:
            return None
        if isinstance(self.bounds, str):
            if self.bounds != "fit":
                raise ValueError(f"bounds must be None, 'fit' or [[min...], [max...]], got {self.bounds!r}")
            self._check_fitted("bounds_")
            return self.bounds_
        box = np.asarray(self.bounds, dtype=np.float64)
        if box.shape != (2, D) or np.any(box[1] < box[0]):
            raise ValueError(f"bounds must be [[min...], [max...]] of shape (2, {D}), got {box.tolist()}")
        return box

    def _localize(self, X):
        anchors, p0, n, sigma, used = self._model()
        D = anchors.shape[1]
        box = self._box(D)
        project = None if box is None else (lambda x, rows: np.clip(x, box[0], box[1]))
        z = np.where(used, np.asarray(X, dtype=np.float64), np.nan)
        known = sigma is not None and bool(np.all(np.isfinite(sigma[used])))
        inv_sigma = np.where(used, 1.0 / np.where(used, sigma, 1.0), 0.0) if known else np.ones(len(anchors))
        model = self._rssi_model(z, anchors, np.where(used, p0, 0.0), np.where(used, n, 0.0), inv_sigma)
        heard = np.isfinite(z)
        enough = heard.sum(axis=1) >= D + 1
        # deterministic starts: power-weighted centroid, then the strongest anchors pulled towards it
        top = np.max(np.where(heard, z, -np.inf), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", over="ignore"):
            w = np.where(heard, 10.0 ** ((z - np.where(np.isfinite(top), top, 0.0)) / 10.0), 0.0)
            centroid = (w @ anchors) / w.sum(axis=1, keepdims=True)
        strongest = np.argsort(-np.where(heard, z, -np.inf), axis=1, kind="stable")
        starts = [centroid]
        for s in range(max(int(self.n_starts), 1) - 1):
            j = strongest[:, min(s, len(anchors) - 1)]
            ok = heard[np.arange(len(z)), j]
            starts.append(np.where(ok[:, None], 0.9 * anchors[j] + 0.1 * centroid, np.nan))
        best = np.full((len(z), D), np.nan)
        best_cost = np.full(len(z), np.inf)
        for x0 in starts:
            x0 = np.where(enough[:, None], x0, np.nan)
            if box is not None:
                x0 = np.clip(x0, box[0], box[1])
            x = _gauss_newton(x0, model, max_iter=self.max_iter, tol=self.tol, project=project)
            rows = np.flatnonzero(np.all(np.isfinite(x), axis=1))
            if rows.size == 0:
                continue
            e, valid = model(x[rows], rows, False)
            cost = np.sum(e * e, axis=1)
            better = cost < best_cost[rows]  # strict: ties keep the earlier start
            best[rows[better]] = x[rows[better]]
            best_cost[rows[better]] = cost[better]
        spread = np.full(len(z), np.nan)
        rows = np.flatnonzero(np.all(np.isfinite(best), axis=1))
        if rows.size:
            e, J, valid = model(best[rows], rows, True)
            w = valid.astype(np.float64)
            scale = np.ones(len(rows)) if known else _residual_sigma(e, w, D)
            spread[rows] = _spread(J, w, scale)
        return Prediction(best, spread=spread)


__all__ = ["PathLossLocalizer", "log_distance_rssi"]
