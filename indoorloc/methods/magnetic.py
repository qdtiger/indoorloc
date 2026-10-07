"""Geomagnetic sequence matching: subsequence dynamic time warping against recorded walks.

A single magnetic reading is ambiguous (the same field magnitude occurs in many places),
but a sequence of readings along a walk is distinctive. ``MagneticDTWLocalizer`` stores
reference walks (``magnetic`` tables: per-sample ``[B, B_h, B_v]`` in uT, or any per-sample
features) and localizes the most recent window of a query walk by finding the reference
subsequence it matches best under dynamic time warping, which absorbs differences in
walking speed and sampling rate.

Functions (numpy only, vectorised over queries and reference samples):

    ``subsequence_dtw``   cost of the best warping path of a whole query ending at every
                          reference sample (free start), optionally with its start sample;
    ``dtw_distance``      DTW between two sequences (both ends aligned).

The local distance is the Euclidean distance between feature vectors; no global band.
Two step patterns (the names of Giorgino's ``dtw`` package; Sakoe & Chiba 1978, Table I):

``"symmetric1"``
    steps ``(1, 0), (0, 1), (1, 1)`` with unit weights, ``D[i, j] = c[i, j] +
    min(D[i-1, j-1], D[i-1, j], D[i, j-1])``: the textbook (subsequence) DTW (Muller 2007).
    The warping is unconstrained, so a long query can collapse onto a few reference
    samples. A row is one vectorised pass: with ``A[j] = c[i, j] + min(D[i-1, j-1],
    D[i-1, j])`` and ``S[j] = c[i, 0] + ... + c[i, j]``, ``D[i, j] = S[j] + min_{k <= j}
    (A[k] - S[k])``, a cumulative minimum.
``"asymmetricP1"``
    Sakoe & Chiba's slope constraint ``P = 1`` in the asymmetric form (weights on the
    query axis)::

        D[i, j] = min( D[i-1, j-2] + (c[i, j-1] + c[i, j]) / 2,
                       D[i-1, j-1] +  c[i, j],
                       D[i-2, j-1] +  c[i-1, j] + c[i, j] )

    The local slope stays between 1/2 and 2 (a reference segment of about ``L/2`` to
    ``2 L`` samples for a query of ``L``) and every path carries a total weight of exactly
    ``L``, so ``cost / L`` is the mean distance per query sample and segments of different
    lengths compete fairly. Each row depends only on the two rows before it: one
    vectorised pass per row, no prefix sums.

Boundaries. ``dtw_distance`` (both ends aligned) starts every path at cell ``(0, 0)`` with
weight 1 (Sakoe & Chiba's ``g(1, 1) = d(1, 1)``), as Giorgino's package does. The open
begin of ``subsequence_dtw`` follows Giorgino's construction: a zero-cost virtual row above
the first query sample from which a path may enter. Here that row also extends two virtual
columns to the left of the reference, so an ``asymmetricP1`` segment may start at the first
reference sample. Giorgino's package has no such columns, so it cannot start a segment at
reference samples 0 or 1. Its costs therefore differ only for paths that start there.

Both cost ``O(L M F)`` per query of ``L`` samples against ``M`` reference samples of ``F``
features. The dynamic program is exact for either pattern: tests compare it with a
brute-force enumeration of warping paths and with values computed by dtw-python 1.7.5, the
Python port of Giorgino's package.

Example (simulated data)::

    walks = load_dataset("synthetic_office", modality="magnetic", split="trajectory", n_trajectories=40)
    ref, query = walks[walks.groups["trajectory"] < 30], walks[walks.groups["trajectory"] >= 30]
    model = MagneticDTWLocalizer(window=30).fit(ref, trajectory=ref.groups["trajectory"])
    evaluate(query, model.localize_walks(query))            # windows never cross walks
"""
from __future__ import annotations

import numpy as np

from ..core import Prediction, SampleTable
from .base import BaseLocalizer, _unpack

STEP_PATTERNS = ("symmetric1", "asymmetricP1")


# ------------------------------------------------------------------------------------ DTW
def _as_sequence(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or len(a) == 0:
        raise ValueError(f"{name} must be a non-empty (T,) or (T, F) sequence, got shape {np.shape(x)}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or inf")
    return a


def _check_pattern(step_pattern) -> str:
    if step_pattern not in STEP_PATTERNS:
        raise ValueError(f"step_pattern must be one of {STEP_PATTERNS}, got {step_pattern!r}")
    return step_pattern


def _cost(q, R) -> np.ndarray:
    """(Q, M) Euclidean distances between query samples (Q, F) and reference samples (M, F)."""
    diff = q[:, None, :] - R[None]
    return np.sqrt(np.einsum("qmf,qmf->qm", diff, diff))


def _scan(A, c):
    """Row of the symmetric1 DTW matrix from ``A`` (vertical/diagonal candidates) and the row
    costs ``c``: ``D[j] = min(A[j], D[j-1] + c[j])`` for every row of ``A`` at once, plus the
    column ``k <= j`` whose ``A[k]`` the optimum comes from (the latest one on ties)."""
    S = np.cumsum(c, axis=1)
    B = A - S
    Bmin = np.minimum.accumulate(B, axis=1)
    M = A.shape[1]
    src = np.maximum.accumulate(np.where(B <= Bmin, np.arange(M), 0), axis=1)
    return Bmin + S, src


def _rows_symmetric1(Q, R, free_start: bool, want_start: bool):
    n, L, _ = Q.shape
    M = len(R)
    c = _cost(Q[:, 0], R)
    # free start: D[0, j] = c[0, j]; anchored start: the first query sample spans reference[0..j]
    D = c if free_start else np.cumsum(c, axis=1)
    start = None
    if want_start:
        start = np.broadcast_to(np.arange(M), (n, M)) if free_start else np.zeros((n, M), np.int64)
    inf = np.full((n, 1), np.inf)
    for i in range(1, L):
        c = _cost(Q[:, i], R)
        diag = np.concatenate([inf, D[:, :-1]], axis=1)
        use_diag = diag <= D                                        # ties prefer the diagonal step
        A = c + np.where(use_diag, diag, D)
        D_new, src = _scan(A, c)
        if want_start:
            prev = np.concatenate([np.zeros((n, 1), np.int64), start[:, :-1]], axis=1)
            start = np.take_along_axis(np.where(use_diag, prev, start), src, axis=1)
        D = D_new
    return D, start


def _rows_p1(Q, R, free_start: bool, want_start: bool):
    n, L, _ = Q.shape
    M = len(R)
    # column p of D1/D2 holds reference index p - 2 (two virtual columns on the left).
    # Free start: row -1 is a virtual zero-cost row (Giorgino's open-begin) and a path may enter
    # from any of its cells, the two virtual columns included, so a segment may begin at any
    # reference sample. Anchored start: the path begins at cell (0, 0) with weight 1 (Sakoe &
    # Chiba's g(1, 1) = d(1, 1)); row -1 is then unreachable.
    D2 = np.full((n, M + 2), np.inf)
    D1 = np.zeros((n, M + 2)) if free_start else np.full((n, M + 2), np.inf)
    S2 = np.zeros((n, M + 2), np.int64)
    S1 = np.broadcast_to(np.arange(-1, M + 1), (n, M + 2))          # a path from (-1, k) starts at k + 1
    pad, inf2, zero2 = np.full((n, 1), np.inf), np.full((n, 2), np.inf), np.zeros((n, 2), np.int64)
    cp_prev = None
    for i in range(L):
        c = _cost(Q[:, i], R)
        cp = np.concatenate([pad, c], axis=1)                       # cp[:, j + 1] = c[:, j]; j = -1 is inf
        if i == 0 and not free_start:
            D = np.full((n, M), np.inf)
            D[:, 0] = c[:, 0]
            if want_start:
                S2, S1 = S1, np.zeros((n, M + 2), np.int64)
            D2, D1 = D1, np.concatenate([inf2, D], axis=1)
            cp_prev = cp
            continue
        cand = [D1[:, 1:M + 1] + c,                                 # (i-1, j-1) -> (i, j)
                D1[:, :M] + 0.5 * (cp[:, :M] + c)]                  # (i-1, j-2) -> (i, j-1) -> (i, j)
        src = [S1[:, 1:M + 1], S1[:, :M]]
        if cp_prev is not None:
            cand.append(D2[:, 1:M + 1] + cp_prev[:, 1:] + c)        # (i-2, j-1) -> (i-1, j) -> (i, j)
            src.append(S2[:, 1:M + 1])
        if want_start:
            cand = np.stack(cand)
            k = np.argmin(cand, axis=0)[None]                       # ties: diagonal, then the others
            D = np.take_along_axis(cand, k, axis=0)[0]
            S2, S1 = S1, np.concatenate([zero2, np.take_along_axis(np.stack(src), k, axis=0)[0]], axis=1)
        else:
            D = cand[0]
            for other in cand[1:]:
                D = np.minimum(D, other)
        D2, D1 = D1, np.concatenate([inf2, D], axis=1)
        cp_prev = cp
    return D1[:, 2:], (S1[:, 2:] if want_start else None)


def _dtw_rows(Q, R, step_pattern: str, free_start: bool, want_start: bool):
    """Last DTW row (n, M) for a batch of queries ``Q`` (n, L, F) against ``R`` (M, F)."""
    rows = _rows_p1 if step_pattern == "asymmetricP1" else _rows_symmetric1
    return rows(Q, R, free_start, want_start)


def subsequence_dtw(query, reference, *, step_pattern: str = "symmetric1", return_start: bool = False):
    """Cost of the best warping path of the whole ``query`` that ends at each ``reference`` sample.

    ``query`` ``(L,)``/``(L, F)`` (or a batch ``(Q, L, F)``), ``reference`` ``(M,)``/``(M, F)``.
    Returns ``cost`` ``(M,)`` (``(Q, M)`` for a batch): ``cost[j]`` is the accumulated distance
    of the optimal alignment of all ``L`` query samples with a contiguous reference segment
    ``reference[s:j + 1]`` for the best start ``s`` (open begin and end, Muller 2007,
    chapter 4); ``inf`` where the step pattern admits no path. With ``return_start=True``
    also returns that ``s`` (int64, same shape). ``step_pattern`` is ``"symmetric1"``
    (unconstrained) or ``"asymmetricP1"`` (slope in [1/2, 2], total weight ``L``; module
    docstring). Among equal-cost predecessors the diagonal step wins, so ``s`` is
    deterministic. ``argmin(cost)`` is the end of the best-matching reference segment.

    References
        H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word
        recognition", IEEE Transactions on Acoustics, Speech, and Signal Processing 26(1):43-49,
        1978. https://doi.org/10.1109/TASSP.1978.1163055
        M. Muller, "Information Retrieval for Music and Motion", Springer, 2007, chapter 4 (dynamic
        time warping, subsequence DTW). https://doi.org/10.1007/978-3-540-74048-3
        T. Giorgino, "Computing and visualizing dynamic time warping alignments in R: the dtw
        package", Journal of Statistical Software 31(7), 2009. https://doi.org/10.18637/jss.v031.i07
    """
    _check_pattern(step_pattern)
    R = _as_sequence(reference, "reference")
    q = np.asarray(query, dtype=np.float64)
    batch = q.ndim == 3
    if not batch:
        q = _as_sequence(q, "query")[None]
    elif q.shape[1] == 0 or not np.all(np.isfinite(q)):
        raise ValueError("query windows must be non-empty and finite")
    if q.shape[2] != R.shape[1]:
        raise ValueError(f"query has {q.shape[2]} features but reference has {R.shape[1]}")
    cost, start = _dtw_rows(q, R, step_pattern, True, return_start)
    if not batch:
        cost = cost[0]
        start = None if start is None else start[0]
    return (cost, np.array(start, dtype=np.int64)) if return_start else cost


def dtw_distance(a, b, *, step_pattern: str = "symmetric1") -> float:
    """DTW distance between two sequences ``(T1,)``/``(T1, F)`` and ``(T2,)``/``(T2, F)``: the
    accumulated Euclidean cost of the best alignment of both sequences end to end (Sakoe &
    Chiba 1978). ``"symmetric1"``: unit-weight steps, no constraint; ``"asymmetricP1"``: slope
    in [1/2, 2] with total weight ``T1`` (``inf`` when the lengths admit no such path). The
    path starts at cell ``(0, 0)``, so the value equals ``dtw(a, b, step_pattern=...).distance``
    of Giorgino's ``dtw`` package with Euclidean local distance."""
    _check_pattern(step_pattern)
    A_, B_ = _as_sequence(a, "a"), _as_sequence(b, "b")
    if A_.shape[1] != B_.shape[1]:
        raise ValueError(f"a has {A_.shape[1]} features but b has {B_.shape[1]}")
    D, _ = _dtw_rows(A_[None], B_, step_pattern, False, False)
    return float(D[0, -1])


# ------------------------------------------------------------------------------ localizer
class MagneticDTWLocalizer(BaseLocalizer):
    """Localization by subsequence DTW of the recent magnetic sequence against reference walks.

    **Reference data** (``fit``): one or more walks, rows in time order, ``X`` ``(N, F)``
    per-sample features (the ``magnetic`` layout ``[B, B_h, B_v]`` in uT, or any subset, e.g.
    the magnitude alone as in LocateMe), positions and optionally floor/building. Walk
    boundaries come from the fit parameter ``trajectory`` ``(N,)`` (a walk's rows must be
    contiguous): ``model.fit(table, trajectory=table.groups["trajectory"])``; without it the
    rows form a single walk. No warping path crosses a walk boundary. With
    ``bidirectional=True`` every walk is also matched in reverse, so a corridor recorded in
    one direction serves queries walking the other way (the features are direction
    independent for a device held flat).

    **Queries** (how windows are formed):

    * ``localize(X)`` / ``predict(X)``: ``X`` ``(T, F)`` is **one** query walk in time order.
      Row ``t`` is localized from the window of the ``window`` samples ending at ``t``
      (rows ``t - window + 1 .. t``); earlier rows use the samples available if there are
      at least ``min_window`` of them, else return a NaN position (``evaluate`` counts them
      in ``n_failed``; floor and building still come from the short match). Causal, so it
      can run on a stream.
    * ``localize_walks(X, trajectory=None)``: several query walks at once (a table uses
      ``groups["trajectory"]``); windows never cross walks.
    * ``localize_windows(windows)``: ``(Q, L, F)`` windows, one estimate per window.

    The estimate is the position (and floor/building) of the reference sample where the
    best-matching segment ends: the query's last sample. ``Prediction.spread`` is the RMS
    distance from the estimate of every reference end sample whose match cost is within a
    factor ``1 + ambiguity`` of the best one (0 for a unique match; large when distant
    places fit about equally well). A window that no reference segment admits (every walk
    shorter than about half the window under ``"asymmetricP1"``) gets a NaN position.

    **Step pattern.** The default ``"asymmetricP1"`` (slope constraint ``P = 1`` of Sakoe &
    Chiba) lets the walking speed of the query differ from the reference by up to a factor
    of 2 and gives every alignment the same total weight. The unconstrained
    ``"symmetric1"`` lets a noisy window shrink onto a few reference samples whose noise
    happens to fit. On ``SyntheticOffice`` magnetic walks (simulated, seed 0, 0.5 uT noise),
    30-sample windows of a reference walk, re-observed with fresh noise, were matched to
    segments of median 22 samples, and as few as 7 (``"asymmetricP1"``: 25 to 35). Over
    1,010 windows from 10 query walks against 30 reference walks, the median error was
    0.90 m against 0.35 m with 3 s windows, and 1.00 m against 0.49 m with 10 s windows.

    **Measured data.** Magnetic sequences alone are far weaker on real recordings than on the
    simulator. On the ILC 2020 sample (site1/F1, a mall about 190 m x 160 m, 120 traces of one
    phone model, ``MagneticFeatures`` resampled to 10 Hz, 5 folds grouped by trace, truth =
    waypoints interpolated in time) the median error was 48 m with 10 s windows against 70 m
    for a random reference sample; |B| recorded by two different traces less than 0.5 m apart
    differed by a median 3.5 uT, about half the 6.7 uT between random places. Use it inside a
    tracker (an L5 particle filter with PDR) rather than as a stand-alone fix there.

    Parameters
    ----------
    window : query length in samples (at 10 Hz and 1.3 m/s, 30 samples cover about 4 m).
    min_window : shortest window used at the start of a walk (None = ``window``).
    bidirectional : also match the reversed reference walks.
    step_pattern : ``"asymmetricP1"`` (default) or ``"symmetric1"`` (see ``subsequence_dtw``).
    ambiguity : relative cost margin of the candidates that make up ``spread``.
    max_block : cap on the (queries x reference samples x features) block evaluated at once,
        which bounds the memory (about 16 bytes per element).

    Deviation from LocateMe (Subbu et al. 2013): LocateMe matches the field magnitude
    recorded along known paths and refines with a nearest-neighbour step; here the features
    are free (default: all columns of ``X``), matching is open-begin/open-end subsequence DTW
    with a slope constraint over every reference walk, and no map or motion model is used
    (combine with an L5 tracker or particle filter for that).

    References
    ----------
    K. P. Subbu, B. Gozick, R. Dantu, "LocateMe: magnetic-fields-based indoor localization using
    smartphones", ACM Transactions on Intelligent Systems and Technology 4(4), 2013.
    DOI 10.1145/2508037.2508054.
    H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition",
    IEEE Transactions on Acoustics, Speech, and Signal Processing 26(1):43-49, 1978.
    DOI 10.1109/TASSP.1978.1163055.
    M. Muller, "Information Retrieval for Music and Motion", Springer, 2007, chapter 4.
    DOI 10.1007/978-3-540-74048-3.
    T. Giorgino, "Computing and visualizing dynamic time warping alignments in R: the dtw package",
    Journal of Statistical Software 31(7), 2009. DOI 10.18637/jss.v031.i07 (step pattern names).
    """

    _allow_nan = False

    def __init__(self, window: int = 30, min_window=None, bidirectional: bool = True,
                 step_pattern: str = "asymmetricP1", ambiguity: float = 0.1, max_block: int = 4_000_000):
        self.window = window
        self.min_window = min_window
        self.bidirectional = bidirectional
        self.step_pattern = step_pattern
        self.ambiguity = ambiguity
        self.max_block = max_block

    # -- fitting -----------------------------------------------------------------------------

    def _check_options(self):
        if int(self.window) != self.window or int(self.window) < 1:
            raise ValueError(f"window must be a positive integer, got {self.window!r}")
        mw = self.window if self.min_window is None else self.min_window
        if int(mw) != mw or not 1 <= int(mw) <= int(self.window):
            raise ValueError(f"min_window must be an integer in [1, window], got {self.min_window!r}")
        if not float(self.ambiguity) >= 0:
            raise ValueError(f"ambiguity must be >= 0, got {self.ambiguity!r}")
        _check_pattern(self.step_pattern)

    def _fit(self, X, pos, floor, building, trajectory=None):
        self._check_options()
        if X.ndim != 2:
            raise ValueError(f"X must be (N, F) per-sample features, got shape {X.shape}")
        if trajectory is None:
            bounds = np.array([0, len(X)], dtype=np.int64)
        else:
            t = np.asarray(trajectory)
            if t.shape != (len(X),):
                raise ValueError(f"trajectory must be ({len(X)},), got shape {t.shape}")
            change = np.flatnonzero(t[1:] != t[:-1]) + 1
            bounds = np.concatenate([[0], change, [len(X)]]).astype(np.int64)
            if len(np.unique(t)) != len(bounds) - 1:
                raise ValueError("every walk's rows must be contiguous and in time order (sort the table by "
                                 "(trajectory, time) first)")
        self.reference_ = np.array(X, dtype=np.float64)
        self.positions_ = np.array(pos, dtype=np.float64)
        self.floor_ = None if floor is None else np.array(floor)
        self.building_ = None if building is None else np.array(building)
        self.walk_bounds_ = bounds

    # -- matching ----------------------------------------------------------------------------

    def _segments(self):
        """(features (M, F), reference row of each sample (M,)) per walk, reversed copies included."""
        out = []
        for a, b in zip(self.walk_bounds_[:-1], self.walk_bounds_[1:]):
            rows = np.arange(a, b)
            out.append((self.reference_[rows], rows))
            if self.bidirectional and b - a > 1:
                out.append((self.reference_[rows[::-1]], rows[::-1]))
        return out

    def _match(self, windows: np.ndarray):
        """Best end sample (reference row), its mean cost per query sample, the ambiguity
        spread and whether any path exists, for (Q, L, F) windows."""
        Q, L, F = windows.shape
        segments = self._segments()
        M_all = sum(len(s) for s, _ in segments)
        rows_all = np.concatenate([r for _, r in segments])
        chunk = max(1, int(self.max_block) // max(M_all * max(F, 1), 1))
        best_row = np.zeros(Q, dtype=np.int64)
        best_cost = np.full(Q, np.inf)
        spread = np.full(Q, np.nan)
        amb = float(self.ambiguity)
        for lo in range(0, Q, chunk):
            W = windows[lo:lo + chunk]
            costs = np.concatenate([_dtw_rows(W, R, self.step_pattern, True, False)[0] for R, _ in segments],
                                   axis=1)                                 # (q, M_all)
            j = np.argmin(costs, axis=1)                                   # first minimum: deterministic
            cmin = costs[np.arange(len(W)), j]
            row = rows_all[j]
            near = costs <= cmin[:, None] * (1.0 + amb) + 1e-12 * (1.0 + np.abs(cmin[:, None]))
            d2 = np.sum(np.square(self.positions_[rows_all][None] - self.positions_[row][:, None, :]), axis=2)
            with np.errstate(invalid="ignore"):
                s = np.sqrt(np.sum(np.where(near, d2, 0.0), axis=1) / near.sum(axis=1))
            found = np.isfinite(cmin)
            spread[lo:lo + len(W)] = np.where(found, s, np.nan)
            best_row[lo:lo + len(W)] = row
            best_cost[lo:lo + len(W)] = cmin / L
        return best_row, best_cost, spread

    def _prediction(self, row, spread, ok) -> Prediction:
        """Positions of the matched rows (NaN where ``ok`` is False or no path matched, i.e. a
        NaN ``spread``) with their floor/building."""
        ok = ok & np.isfinite(spread)
        pos = np.where(ok[:, None], self.positions_[row], np.nan)
        spread = np.where(ok, spread, np.nan)
        floor = None if self.floor_ is None else self.floor_[row]
        building = None if self.building_ is None else self.building_[row]
        return Prediction(pos, floor=floor, building=building, spread=spread)

    def localize_windows(self, windows) -> Prediction:
        """One estimate per window: ``windows`` ``(Q, L, F)`` (or ``(Q, L)`` for one feature),
        consecutive samples in time order; the estimate is for each window's last sample."""
        self._check_fitted("reference_")
        W = np.asarray(windows, dtype=np.float64)
        if W.ndim == 2:
            W = W[..., None]
        if W.ndim != 3 or W.shape[2] != self.reference_.shape[1] or W.shape[1] == 0:
            raise ValueError(f"windows must be (Q, L, {self.reference_.shape[1]}), got shape {np.shape(windows)}")
        if not np.all(np.isfinite(W)):
            raise ValueError("windows contain NaN or inf")
        if len(W) == 0:
            return Prediction(np.zeros((0, self.positions_.shape[1])))
        row, _, spread = self._match(W)
        return self._prediction(row, spread, np.ones(len(W), dtype=bool))

    def _localize(self, X):
        self._check_options()
        X = np.asarray(X, dtype=np.float64)
        T = len(X)
        L = int(self.window)
        mw = L if self.min_window is None else int(self.min_window)
        size = np.minimum(np.arange(T) + 1, L)
        row = np.zeros(T, dtype=np.int64)
        spread = np.full(T, np.nan)
        full = np.flatnonzero(size == L)
        if full.size:
            win = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)  # (T - L + 1, F, L)
            W = np.ascontiguousarray(np.moveaxis(win[full - (L - 1)], 2, 1))
            row[full], _, spread[full] = self._match(W)
        # the first rows of a walk: shorter windows; below min_window only floor/building are kept
        for t in np.flatnonzero(size < L):
            row[t:t + 1], _, spread[t:t + 1] = self._match(X[None, :t + 1])
        return self._prediction(row, spread, size >= mw)

    def localize_walks(self, X, trajectory=None) -> Prediction:
        """``localize`` applied to each query walk separately (rows of a walk in time order).
        ``trajectory`` defaults to ``X.groups["trajectory"]`` for a SampleTable; ids are kept."""
        ids = X.ids if isinstance(X, SampleTable) else None
        if trajectory is None:
            if not isinstance(X, SampleTable) or "trajectory" not in X.groups:
                raise ValueError("give trajectory=, or a SampleTable with groups['trajectory']")
            trajectory = X.groups["trajectory"]
        data = self._validate(_unpack(X)[0])
        t = np.asarray(trajectory)
        if t.shape != (len(data),):
            raise ValueError(f"trajectory must be ({len(data)},), got shape {t.shape}")
        D = self.positions_.shape[1]
        pos = np.full((len(data), D), np.nan)
        spread = np.full(len(data), np.nan)
        labels = [None if lab is None else np.zeros(len(data), dtype=lab.dtype)
                  for lab in (self.floor_, self.building_)]
        for walk in np.unique(t):
            rows = np.flatnonzero(t == walk)
            p = self._localize(data[rows])
            pos[rows], spread[rows] = p.pos, p.spread
            for k, lab in enumerate((p.floor, p.building)):
                if lab is not None:
                    labels[k][rows] = lab
        return Prediction(pos, floor=labels[0], building=labels[1], ids=ids, spread=spread)


__all__ = ["MagneticDTWLocalizer", "dtw_distance", "subsequence_dtw"]
