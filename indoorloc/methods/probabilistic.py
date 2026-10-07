"""Probabilistic fingerprinting: per-location signal-strength distributions (Horus).

Also holds the helpers that the likelihood-based localizers share: an order-fixed sum
(so a log-likelihood never depends on batching or threads) and the posterior-weighted
centre of mass of the best candidate locations with label votes.
"""
from __future__ import annotations

import numpy as np

from ..core import Prediction
from .base import BaseLocalizer

_LOG_2PI = float(np.log(2.0 * np.pi))


def fixed_sum(a: np.ndarray) -> np.ndarray:
    """Sum over the last axis by a fixed pairwise tree of elementwise adds.

    Each output depends only on its own row, never on the other rows, on chunking or on
    threads (the same argument as ``neighbors.sq_distances``).
    """
    s = np.asarray(a, dtype=np.float64)
    if s.shape[-1] == 0:
        return np.zeros(s.shape[:-1])
    while s.shape[-1] > 1:
        if s.shape[-1] % 2:
            s = np.concatenate([s, np.zeros(s.shape[:-1] + (1,))], axis=-1)
        s = s[..., 0::2] + s[..., 1::2]
    return s[..., 0]


def log_gauss(x, mean, std) -> np.ndarray:
    """Elementwise log N(x; mean, std**2)."""
    z = (x - mean) / std
    return -0.5 * _LOG_2PI - np.log(std) - 0.5 * z * z


def _label_vote(labels, idx, w):
    """Weighted vote of candidate labels; ties go to the smallest label."""
    if labels is None:
        return None
    classes, codes = np.unique(labels, return_inverse=True)
    scores = np.zeros((len(idx), len(classes)))
    np.add.at(scores, (np.arange(len(idx))[:, None], codes[idx]), w)
    return classes[scores.argmax(axis=1)]


def centre_of_mass(loglik: np.ndarray, positions: np.ndarray, floors=None, buildings=None, n: int = 1) -> Prediction:
    """Posterior-weighted mean of the ``n`` most likely candidates (uniform prior).

    ``loglik`` is (N, L) over L candidate ``positions`` (L, D). The top ``n`` are ranked by
    log-likelihood, ties by candidate index; their weights are the posterior probabilities
    renormalised over the top ``n`` (Horus's continuous-space estimator). ``n=1`` is the
    maximum-likelihood candidate. Floor/building: the same weighted vote over the top ``n``.
    ``spread``: weighted RMS distance of the top ``n`` from the estimate.
    """
    n = min(int(n), loglik.shape[1])
    top = np.argsort(-loglik, axis=1, kind="stable")[:, :n]
    ll = np.take_along_axis(loglik, top, axis=1)
    w = np.exp(ll - ll[:, :1])
    w /= w.sum(axis=1, keepdims=True)
    cand = positions[top]  # (N, n, D)
    pos = np.einsum("nk,nkd->nd", w, cand)
    spread = np.sqrt(np.einsum("nk,nk->n", w, np.square(cand - pos[:, None]).sum(-1)))
    return Prediction(pos, _label_vote(floors, top, w), _label_vote(buildings, top, w), spread=spread)


def group_rows(keys: np.ndarray):
    """Unique rows of ``keys`` (sorted) with the row order and slice starts to reduce each group."""
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return unique, inverse, counts, order, starts


class HorusLocalizer(BaseLocalizer):
    """Horus: maximum-likelihood fingerprinting with per-location, per-AP Gaussian RSSI models.

    Training groups the scans by reference location (identical position, floor and
    building) and fits, for every location L and access point a, a Gaussian to the
    readings of a heard at L (mean, and standard deviation floored at ``min_std``). A query
    scan s is scored by the log-likelihood log P(s | L) = sum_a log P(s_a | L) with APs
    independent, and the estimate is the posterior-weighted centre of mass of the
    ``n_candidates`` most likely locations (uniform prior); floor and building come from
    the same weighted vote. ``n_candidates=1`` returns the maximum-likelihood location.

    Missing readings (NaN) are modelled, not filled. Detection is a Bernoulli event with
    probability pi_La = (h + k) / (n + 2k), from h detections in the n scans at L and the
    pseudo-count k = ``detection_smoothing``: a heard reading contributes
    log pi_La + log N(s_a; mu_La, sigma_La), a missing one log(1 - pi_La). An AP never heard
    at L has no fitted Gaussian; a reading of it is scored against N(floor, ``unseen_std``),
    where floor is the weakest reading in the training data (the edge of coverage), so an
    unexpected strong AP counts against L without vetoing it outright. An AP heard in every
    training scan carries no evidence about where it drops out, so its detection is not
    modelled (pi_La = 1, and a missing query reading of it is ignored); in particular, filled
    input (no NaN) reduces exactly to the classic model sum_a log N(s_a; mu_La, sigma_La),
    with no preference for locations that have more scans.

    Parameters
    ----------
    n_candidates : number of best locations averaged (Horus's centre of mass).
    min_std : lower bound on every per-location standard deviation, in dBm. Twenty scans
        underestimate the spread a new device or day produces; the floor keeps a single
        AP from dominating.
    detection_smoothing : pseudo-count k > 0 of the detection probability (0.5 = Jeffreys
        prior, 1 = Laplace).
    unseen_std : standard deviation (dBm) of a reading from an AP never heard at L.

    The defaults were chosen on a split of the UJIIndoorLoc training file (25 % of reference
    positions held out), never on its validation file.

    Deviations from the paper: the Gaussian density replaces Horus's probability of the
    1-dB bin around an integer reading (the two differ by a factor 1 + (z^2 - 1) / (24 sigma^2),
    under 4 % for sigma >= 4 dB and |z| <= 4); the detection model is an addition (the
    paper does not say how absent APs are scored); the clustering, correlation-handling
    (autoregressive) and small-scale compensation modules are not implemented (time
    averaging belongs to L5 tracking).

    Attributes: ``locations_`` (L, D), ``location_floor_``/``location_building_`` (L,) or
    None, ``location_counts_`` (L,), ``mean_``/``std_``/``detect_prob_`` (L, n_features),
    ``detection_modelled_`` (n_features,) bool (False: the AP was heard in every training scan).

    References
    ----------
    Youssef, M., Agrawala, A., "The Horus WLAN location determination system", MobiSys 2005.
    DOI 10.1145/1067170.1067193
    Youssef, M., Agrawala, A., Shankar, A. U., "WLAN location determination via clustering and
    probability distributions", PerCom 2003. DOI 10.1109/PERCOM.2003.1192736
    """

    _allow_nan = True

    def __init__(self, n_candidates: int = 3, min_std: float = 4.0, detection_smoothing: float = 0.5,
                 unseen_std: float = 8.0):
        self.n_candidates = n_candidates
        self.min_std = min_std
        self.detection_smoothing = detection_smoothing
        self.unseen_std = unseen_std

    def _fit(self, X, pos, floor, building):
        if int(self.n_candidates) < 1:
            raise ValueError(f"n_candidates must be >= 1, got {self.n_candidates}")
        for name in ("min_std", "detection_smoothing", "unseen_std"):
            if not float(getattr(self, name)) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        X = X.reshape(len(X), -1).astype(np.float64)
        heard = ~np.isnan(X)
        if not heard.any():
            raise ValueError("no reading is heard in the training data (X is all NaN)")
        keys = np.column_stack([pos] + [np.asarray(v, np.float64) for v in (floor, building) if v is not None])
        unique, inverse, counts, order, starts = group_rows(keys)
        d = pos.shape[1]

        values = np.where(heard, X, 0.0)
        n_heard = np.add.reduceat(heard[order].astype(np.float64), starts, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.add.reduceat(values[order], starts, axis=0) / n_heard
            dev = np.where(heard, X - mean[inverse], 0.0)
            std = np.sqrt(np.add.reduceat((dev * dev)[order], starts, axis=0) / n_heard)
        seen = n_heard > 0
        self.rssi_floor_ = float(np.min(X[heard]))
        mean = np.where(seen, mean, self.rssi_floor_)
        std = np.where(seen, np.maximum(std, float(self.min_std)), float(self.unseen_std))
        k = float(self.detection_smoothing)
        smoothed = (n_heard + k) / (counts[:, None] + 2.0 * k)
        # An AP heard in every training scan (all of them, if the input was filled) says nothing about
        # where it drops out: its detection is not modelled. Smoothing it would add sum_a log pi_La, a
        # term that favours locations with more scans (it spans 92 nats across filled UJIIndoorLoc locations).
        modelled = ~heard.all(axis=0)

        self.locations_ = unique[:, :d].copy()
        self.location_floor_ = unique[:, d].astype(np.int64) if floor is not None else None
        self.location_building_ = unique[:, -1].astype(np.int64) if building is not None else None
        self.location_counts_ = counts
        self.detection_modelled_ = modelled
        self.mean_, self.std_ = mean, std
        self.detect_prob_ = np.where(modelled, smoothed, 1.0)
        self.log_detect_ = np.log(self.detect_prob_)                  # 0 where not modelled
        self.log_miss_ = np.log1p(-np.where(modelled, smoothed, 0.0))  # 0 there too: a missing reading is ignored
        self.miss_total_ = fixed_sum(self.log_miss_)  # log-likelihood of a scan that hears nothing

    def log_likelihood(self, X) -> np.ndarray:
        """(N, L) log P(scan | location) for every reference location ``locations_``."""
        X = self._validate(X)
        return self._log_likelihood(X.reshape(len(X), -1))

    def _log_likelihood(self, X) -> np.ndarray:
        out = np.empty((len(X), len(self.locations_)))
        for i, x in enumerate(np.asarray(X, dtype=np.float64)):
            a = np.flatnonzero(~np.isnan(x))  # only heard APs differ from the "hears nothing" baseline
            terms = (self.log_detect_[:, a] - self.log_miss_[:, a]
                     + log_gauss(x[a], self.mean_[:, a], self.std_[:, a]))
            out[i] = self.miss_total_ + fixed_sum(terms)
        return out

    def _localize(self, X):
        ll = self._log_likelihood(X.reshape(len(X), -1))
        return centre_of_mass(ll, self.locations_, self.location_floor_, self.location_building_,
                              self.n_candidates)


__all__ = ["HorusLocalizer", "centre_of_mass", "fixed_sum", "log_gauss"]
