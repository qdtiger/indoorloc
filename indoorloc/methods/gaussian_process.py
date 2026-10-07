"""Gaussian-process radio maps: signal strength as a smooth function of position (numpy only)."""
from __future__ import annotations

import warnings

import numpy as np

from .base import BaseLocalizer
from .probabilistic import centre_of_mass, fixed_sum, group_rows

_LOG_2PI = float(np.log(2.0 * np.pi))
JITTER = 1e-8  # added to the unit-variance correlation matrix: a tiny nugget that keeps it positive definite


def _sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(len(A), len(B)) squared Euclidean distances, from exact coordinate differences."""
    out = np.zeros((len(A), len(B)))
    for d in range(A.shape[1]):
        diff = A[:, d, None] - B[None, :, d]
        out += diff * diff
    return out


def gp_log_marginal(d2, counts, ybar, within_ss, length_scale, noise_ratio):
    """Profile log marginal likelihood of repeated readings, every column of ``ybar`` at once.

    Model (per column / AP): raw readings y_ij = f(x_i) + e_ij at n distinct inputs with
    m_i repeats, f ~ GP(0, s R) with R = exp(-d^2 / (2 l^2)) (+ JITTER I) and
    e ~ N(0, g s). The m_i readings at x_i are summarised exactly by their mean ``ybar``
    (already centred on the prior mean) and within sum of squares ``within_ss``, and the
    signal variance s is profiled out: s_hat = (ybar' (R + g D)^-1 ybar + SS / g) / M with
    D = diag(1 / m_i) and M = sum m_i. Returns ``(log p(y | l, g, s_hat), s_hat)``; the value
    equals the Gaussian log density of all M raw readings under the same model.
    """
    n, m_total = len(counts), float(counts.sum())
    K = np.exp(-d2 / (2.0 * length_scale ** 2)) + np.diag(noise_ratio / counts + JITTER)
    chol = np.linalg.cholesky(K)
    v = np.linalg.solve(chol, ybar)
    quad = np.einsum("ij,ij->j", v, v)
    s = (quad + within_ss / noise_ratio) / m_total
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    ll = (-0.5 * m_total * (np.log(s) + 1.0 + _LOG_2PI) - 0.5 * log_det
          - 0.5 * (m_total - n) * np.log(noise_ratio) - 0.5 * np.sum(np.log(counts)))
    return ll, s


class GPRadioMapLocalizer(BaseLocalizer):
    """Gaussian-process radio map with maximum-likelihood localization (Ferris et al., 2006).

    Each access point's signal strength is a Gaussian process over position with a constant
    mean (the AP's average reading), a squared-exponential kernel s * exp(-|x - x'|^2 / (2 l^2))
    and Gaussian reading noise of variance g * s. Repeated scans at a reference position enter
    exactly through their mean and within-position scatter. For every AP the length scale l
    and noise ratio g maximise the marginal likelihood over a log-spaced grid, with the signal
    variance s profiled out in closed form. The GP's predictive mean and variance (of a new
    reading, noise included) at the candidate positions form the radio map; a query scan is
    scored by the Gaussian log-likelihood summed over APs and the estimate is the most likely
    candidate (``n_candidates=1``) or the posterior-weighted mean of the best few.

    Input must be complete: fill missing readings first (``preprocess=FillMissing(-104)``),
    so "not heard" is modelled as a weak reading. APs whose readings never vary carry no
    positional information and are left out (if none varies, all candidates tie and the
    first, in sorted order, is returned). The kernel is over position only, so on
    multi-floor data wrap the model per floor: ``HierarchicalLocalizer(position_model=
    GPRadioMapLocalizer())``. Fitting costs O(n^3) per grid point for n distinct positions.

    Parameters
    ----------
    n_candidates : number of best candidates averaged (1 = maximum likelihood).
    length_scales : grid of kernel length scales, in position units. None: 16 values from half
        the median nearest-neighbour spacing of the reference positions to their diameter.
    noise_ratios : grid of noise-to-signal variance ratios g. None: 9 values from 1e-3 to 10.
    candidates : (M, D) positions to score (e.g. a dense grid); None = the distinct training
        positions. Arbitrary candidates take floor/building from the nearest reference position.

    Deviations from the paper: hyper-parameters are chosen per AP on a grid (with the signal
    variance profiled), where the paper fits one global setting for all APs by conjugate
    gradients; the mean is constant (the AP's average reading) instead of the paper's
    linear-in-distance offset m |x - x_AP| + b; the per-AP likelihoods are multiplied as they
    are, while the paper tempers the product with gamma = 1/n (a geometric mean), which
    leaves the maximum-likelihood candidate unchanged but gives ``n_candidates > 1`` sharper
    weights; localization is static maximum likelihood over candidate positions (the paper's
    particle filter belongs to L5 tracking). As in the paper, an undetected AP is the bottom of
    the signal scale, hence the filled input. Kernel solves use LAPACK, so the last bits of
    the radio map can depend on the BLAS thread count; the per-candidate scores are summed
    in a fixed order.

    Attributes: ``active_`` (n_features,) bool, ``length_scale_``, ``noise_ratio_``,
    ``signal_var_`` (per active AP), ``candidates_`` (M, D), ``map_mean_``/``map_var_`` (M, n_active).

    References
    ----------
    Ferris, B., Hähnel, D., Fox, D., "Gaussian processes for signal strength-based location
    estimation", Robotics: Science and Systems II, 2006. DOI 10.15607/RSS.2006.II.039
    Rasmussen, C. E., Williams, C. K. I., "Gaussian Processes for Machine Learning", MIT Press,
    2006 (Algorithm 2.1; eq. 5.8). URL http://gaussianprocess.org/gpml/
    """

    def __init__(self, n_candidates: int = 1, length_scales=None, noise_ratios=None, candidates=None):
        self.n_candidates = n_candidates
        self.length_scales = length_scales
        self.noise_ratios = noise_ratios
        self.candidates = candidates

    def _fit(self, X, pos, floor, building):
        if int(self.n_candidates) < 1:
            raise ValueError(f"n_candidates must be >= 1, got {self.n_candidates}")
        X = X.reshape(len(X), -1).astype(np.float64)
        refs, inverse, counts, order, starts = group_rows(pos)
        counts = counts.astype(np.float64)
        means = np.add.reduceat(X[order], starts, axis=0) / counts[:, None]
        dev = X - means[inverse]
        within = np.add.reduceat((dev * dev)[order], starts, axis=0).sum(axis=0)
        self.active_ = X.max(axis=0) > X.min(axis=0)  # none active: every candidate ties (first one wins)
        for name, labels in (("floor", floor), ("building", building)):
            if labels is not None and len(np.unique(np.column_stack([inverse, labels]), axis=0)) > len(refs):
                warnings.warn(f"some positions carry several {name} labels; the radio map is over position "
                              f"only (use HierarchicalLocalizer to fit one map per {name})", stacklevel=3)

        self.feature_mean_ = X.mean(axis=0)
        self.offset_ = self.feature_mean_[self.active_]
        self.center_ = refs.mean(axis=0)
        self.refs_ = refs - self.center_
        self.counts_ = counts
        self.ybar_ = means[:, self.active_] - self.offset_
        within = within[self.active_]
        d2 = _sq_dist(self.refs_, self.refs_)
        scales = self._length_grid(d2) if self.length_scales is None else np.asarray(self.length_scales, float)
        ratios = np.geomspace(1e-3, 10.0, 9) if self.noise_ratios is None else np.asarray(self.noise_ratios, float)
        if scales.ndim != 1 or ratios.ndim != 1 or not (np.all(scales > 0) and np.all(ratios > 0)):
            raise ValueError("length_scales and noise_ratios must be 1-D grids of positive values")

        n_ap = self.ybar_.shape[1]
        best, choice, signal = np.full(n_ap, -np.inf), np.zeros((n_ap, 2), dtype=np.int64), np.zeros(n_ap)
        for i, length in enumerate(scales):
            for j, ratio in enumerate(ratios):
                ll, s = gp_log_marginal(d2, counts, self.ybar_, within, length, ratio)
                better = ll > best  # strict: the first grid point wins ties
                best[better], signal[better] = ll[better], s[better]
                choice[better] = (i, j)
        self.log_marginal_ = best
        self.length_scale_, self.noise_ratio_ = scales[choice[:, 0]], ratios[choice[:, 1]]
        self.signal_var_ = signal

        if self.candidates is None:
            cand = refs
            cand_floor = self._majority(floor, inverse, len(refs))
            cand_building = self._majority(building, inverse, len(refs))
        else:
            cand = np.asarray(self.candidates, dtype=np.float64).reshape(-1, pos.shape[1])
            nearest = np.argmin(_sq_dist(cand - self.center_, self.refs_), axis=1)  # ties: lowest index
            cand_floor = None if floor is None else self._majority(floor, inverse, len(refs))[nearest]
            cand_building = None if building is None else self._majority(building, inverse, len(refs))[nearest]
        self.candidates_, self.candidate_floor_, self.candidate_building_ = cand, cand_floor, cand_building
        self.map_mean_, self.map_var_ = self._predict_active(cand)
        self.map_log_det_ = fixed_sum(np.log(self.map_var_) + _LOG_2PI)

    @staticmethod
    def _length_grid(d2):
        n = len(d2)
        if n < 2:
            return np.geomspace(0.5, 50.0, 16)
        off = d2 + np.diag(np.full(n, np.inf))
        spacing = float(np.median(np.sqrt(off.min(axis=1))))
        diameter = float(np.sqrt(d2.max()))
        low = 0.5 * spacing if spacing > 0 else 0.5
        return np.geomspace(low, max(diameter, 2.0 * low), 16)

    @staticmethod
    def _majority(labels, inverse, n):
        """Most frequent label per reference position (ties: smallest label)."""
        if labels is None:
            return None
        classes, codes = np.unique(labels, return_inverse=True)
        votes = np.zeros((n, len(classes)))
        np.add.at(votes, (inverse, codes.reshape(-1)), 1.0)
        return classes[votes.argmax(axis=1)]

    def _predict_active(self, positions):
        """Predictive mean and variance of a new reading, (M, n_active), at absolute positions."""
        P = np.asarray(positions, dtype=np.float64) - self.center_
        mean = np.empty((len(P), self.ybar_.shape[1]))
        var = np.empty_like(mean)
        d2 = _sq_dist(self.refs_, self.refs_)
        d2_star = _sq_dist(self.refs_, P)
        pairs = np.column_stack([self.length_scale_, self.noise_ratio_])
        for length, ratio in np.unique(pairs, axis=0):
            aps = np.flatnonzero((self.length_scale_ == length) & (self.noise_ratio_ == ratio))
            K = np.exp(-d2 / (2.0 * length ** 2)) + np.diag(ratio / self.counts_ + JITTER)
            chol = np.linalg.cholesky(K)
            v_star = np.linalg.solve(chol, np.exp(-d2_star / (2.0 * length ** 2)))  # (n, M)
            v = np.linalg.solve(chol, self.ybar_[:, aps])
            mean[:, aps] = self.offset_[aps] + v_star.T @ v
            f_var = np.maximum(1.0 - np.einsum("ij,ij->j", v_star, v_star), 0.0)
            var[:, aps] = self.signal_var_[aps] * (f_var[:, None] + ratio)
        return mean, var

    def radio_map(self, positions) -> tuple[np.ndarray, np.ndarray]:
        """Predicted reading mean and variance (M, n_features) at positions (M, D).

        Features that never vary in training return their constant value with variance 0.
        """
        self._check_fitted()
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, self.refs_.shape[1])
        mean = np.zeros((len(positions), len(self.active_)))
        var = np.zeros_like(mean)
        mean[:, ~self.active_] = self.feature_mean_[~self.active_]
        mean[:, self.active_], var[:, self.active_] = self._predict_active(positions)
        return mean, var

    def log_likelihood(self, X) -> np.ndarray:
        """(N, M) Gaussian log-likelihood of each scan at every candidate position ``candidates_``."""
        X = self._validate(X)
        return self._log_likelihood(X.reshape(len(X), -1))

    def _log_likelihood(self, X) -> np.ndarray:
        S = np.asarray(X, dtype=np.float64)[:, self.active_]
        n_cand, n_ap = self.map_mean_.shape
        out = np.empty((len(S), n_cand))
        chunk = max(1, 2_000_000 // max(1, n_cand * n_ap))
        for start in range(0, len(S), chunk):
            diff = S[start:start + chunk, None, :] - self.map_mean_[None]
            out[start:start + chunk] = -0.5 * (self.map_log_det_ + fixed_sum(diff * diff / self.map_var_))
        return out

    def _localize(self, X):
        ll = self._log_likelihood(X.reshape(len(X), -1))
        return centre_of_mass(ll, self.candidates_, self.candidate_floor_, self.candidate_building_,
                              self.n_candidates)


__all__ = ["GPRadioMapLocalizer", "gp_log_marginal"]
