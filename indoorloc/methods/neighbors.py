"""k-nearest-neighbour fingerprinting with exact, thread-independent tie-breaking."""
from __future__ import annotations

import numpy as np

from ..core import Prediction
from .base import BaseLocalizer


def sq_distances(A: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance of every row of ``A`` to ``q``.

    Summed by a fixed pairwise tree of elementwise adds, so the value depends only on
    the two rows, never on how many rows are passed, on chunking or on threads (numpy
    does not specify the order of ``sum(axis=1)``; elementwise IEEE adds are exact
    operations with one fixed rounding each).
    """
    s = np.square(A - q)
    while s.shape[1] > 1:
        if s.shape[1] % 2:
            s = np.concatenate([s, np.zeros((len(s), 1))], axis=1)
        s = s[:, 0::2] + s[:, 1::2]
    return s[:, 0]


def kneighbors(X_fit, X, k: int, *, chunk_size: int = 256, fit_sq=None) -> tuple[np.ndarray, np.ndarray]:
    """Distances and indices of the ``k`` nearest rows of ``X_fit`` for each row of ``X``.

    Contract: neighbours are ordered by (``sq_distances`` value, training index). The
    result depends on neither BLAS/OpenMP threads nor query batching. A GEMM pass
    shortlists every row within a rounding margin of the k-th distance; the shortlist
    is re-scored with ``sq_distances`` and sorted stably. With integer-valued features
    (e.g. RSSI in whole dBm) every distance is an exact integer, so ties are exact ties.
    """
    X_fit = np.asarray(X_fit, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if not 1 <= k <= len(X_fit):
        raise ValueError(f"k={k} must be between 1 and the number of training samples ({len(X_fit)})")
    fit_sq = np.einsum("ij,ij->i", X_fit, X_fit) if fit_sq is None else fit_sq
    # Shortlist margin, F features, S = |q|^2 + |t|^2 (first order in eps):
    #   GEMM value d2 = (|q|^2 + |t|^2) - 2 q.t, two F-term dot products, one add, one
    #   subtract:                      |d2 - d| <= (F + 1.5) * eps * S
    #   sq_distances (subtract, square, pairwise tree): |s - d| <= (ceil(log2 F) + 3) * eps * S
    #   rounding of kth + margin:                         <= eps * S
    # A row that ranks in the top k by s (ties included) lies within the sum of its own and the
    # k-th row's errors: (2F + 2 ceil(log2 F) + 10) * eps * S_max. (4F + 16) * eps * S_max exceeds
    # that by at least 8 eps S_max for every F >= 1, which covers the second-order terms.
    rel_margin = (4 * X.shape[1] + 16) * np.finfo(np.float64).eps
    dist = np.empty((len(X), k))
    idx = np.empty((len(X), k), dtype=np.intp)
    for start in range(0, len(X), chunk_size):
        Q = X[start:start + chunk_size]
        q_sq = np.einsum("ij,ij->i", Q, Q)
        d2 = q_sq[:, None] + fit_sq[None, :] - 2.0 * (Q @ X_fit.T)
        limit = np.partition(d2, k - 1, axis=1)[:, k - 1] + rel_margin * (q_sq + fit_sq.max())
        for r, q in enumerate(Q):
            cand = np.flatnonzero(d2[r] <= limit[r])  # ascending training index
            exact = sq_distances(X_fit[cand], q)
            order = np.argsort(exact, kind="stable")[:k]  # equal distances keep index order
            idx[start + r] = cand[order]
            dist[start + r] = np.sqrt(exact[order])
    return dist, idx


def _weights(dist: np.ndarray, kind: str) -> np.ndarray:
    if kind == "uniform":
        return np.ones_like(dist)
    with np.errstate(divide="ignore"):
        w = 1.0 / dist
    exact = np.isinf(w)
    rows = exact.any(axis=1)
    w[rows] = exact[rows]  # an exact match takes all the weight (sklearn's convention)
    return w


def _vote(labels, idx, w):
    """Weighted majority label of the neighbours; ties go to the smallest label (as sklearn)."""
    if labels is None:
        return None
    classes, codes = np.unique(labels, return_inverse=True)
    scores = np.zeros((len(idx), len(classes)))
    np.add.at(scores, (np.arange(len(idx))[:, None], codes[idx]), w)
    return classes[scores.argmax(axis=1)]


class KNNLocalizer(BaseLocalizer):
    """k-NN fingerprinting: average the positions of the ``k`` nearest reference scans.

    Floor and building come from a vote of the same neighbours. ``weights``: "uniform",
    or "distance" for inverse-distance weights (WKNN). N-D inputs are flattened.
    ``Prediction.spread`` = weighted RMS distance of the neighbours from the estimate.
    Neighbours at equal distance are ordered by training index, so results do not depend on
    BLAS/OpenMP threads or query batching (see :func:`kneighbors`).

    References
    ----------
    P. Bahl, V. N. Padmanabhan, "RADAR: an in-building RF-based user location and tracking
    system", IEEE INFOCOM 2000. DOI 10.1109/INFCOM.2000.832252.
    J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for
    WLAN fingerprint-based indoor localization problems", IPIN 2014.
    DOI 10.1109/IPIN.2014.7275492 (the k-NN baseline of the dataset).
    """

    def __init__(self, k: int = 5, weights: str = "uniform", chunk_size: int = 256):
        self.k = k
        self.weights = weights
        self.chunk_size = chunk_size

    def _fit(self, X, pos, floor, building):
        if self.weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got {self.weights!r}")
        if not 1 <= self.k <= len(X):
            raise ValueError(f"k={self.k} exceeds the number of training samples ({len(X)} sample(s))")
        self.X_fit_ = X.reshape(len(X), -1).astype(np.float64)
        self.fit_sq_ = np.einsum("ij,ij->i", self.X_fit_, self.X_fit_)  # cached: fast single-scan queries
        self.pos_, self.floor_, self.building_ = pos, floor, building

    def kneighbors(self, X) -> tuple[np.ndarray, np.ndarray]:
        X = self._validate(X)
        return kneighbors(self.X_fit_, X.reshape(len(X), -1), self.k, chunk_size=self.chunk_size,
                          fit_sq=self.fit_sq_)

    def _localize(self, X):
        dist, idx = kneighbors(self.X_fit_, X.reshape(len(X), -1), self.k, chunk_size=self.chunk_size,
                               fit_sq=self.fit_sq_)
        w = _weights(dist, self.weights)
        neighbours = self.pos_[idx]  # (N, k, D)
        pos = np.einsum("nk,nkd->nd", w, neighbours) / w.sum(axis=1, keepdims=True)
        spread = np.sqrt(np.einsum("nk,nk->n", w, np.square(neighbours - pos[:, None]).sum(-1))
                         / w.sum(axis=1))
        return Prediction(pos, _vote(self.floor_, idx, w), _vote(self.building_, idx, w), spread=spread)


class WKNNLocalizer(KNNLocalizer):
    """Weighted k-NN: KNNLocalizer with inverse-distance weights by default."""

    def __init__(self, k: int = 5, weights: str = "distance", chunk_size: int = 256):
        super().__init__(k=k, weights=weights, chunk_size=chunk_size)
