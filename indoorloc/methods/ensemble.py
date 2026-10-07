"""Model ensembles: several localizers combined into one (sensor fusion is L5, not here).

Members are localizers given as instances or registry names (``["wknn", "rf", HorusLocalizer()]``);
each is cloned before fitting, so the parameters stay untouched. An ensemble accepts
missing readings (NaN) exactly when all its members do: mix NaN-aware members (Horus,
gradient boosting) with others through ``LocalizerPipeline(FillMissing(-104), member)``.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..core import Prediction, clone
from .base import BaseLocalizer
from .probabilistic import group_rows


def resolve_localizer(member) -> BaseLocalizer:
    """A fresh, unfitted localizer from a registry name or a localizer instance."""
    if isinstance(member, str):
        from . import create_model  # the registry imports the member's module only now

        return create_model(member)
    if not isinstance(member, BaseLocalizer):
        raise TypeError(f"members must be localizers or registry names, got {type(member).__name__}")
    return clone(member)


def accepts(members, flag: str) -> bool:
    """Whether every member takes NaN (``flag="_allow_nan"``) or complex input: a meta-estimator
    declares, and checks, only what all its members accept."""
    out = True
    for m in members:
        if isinstance(m, (str, BaseLocalizer)):
            out &= bool(getattr(resolve_localizer(m) if isinstance(m, str) else m, flag))
        elif flag == "_allow_nan" and hasattr(m, "__sklearn_tags__"):  # a scikit-learn classifier stage
            try:
                out &= bool(m.__sklearn_tags__().input_tags.allow_nan)
            except AttributeError:  # tags without input_tags: assume it rejects NaN
                out = False
        else:
            out = False
    return out


def _vote(predictions, attr: str, weights) -> np.ndarray | None:
    """Weighted majority of the members that predict ``attr``; ties go to the smallest label."""
    cols = [(getattr(p, attr), w) for p, w in zip(predictions, weights) if w > 0 and getattr(p, attr) is not None]
    if not cols:
        return None
    labels = np.stack([c for c, _ in cols])  # (M', N)
    classes, codes = np.unique(labels, return_inverse=True)
    codes = codes.reshape(labels.shape)
    scores = np.zeros((labels.shape[1], len(classes)))
    rows = np.arange(labels.shape[1])
    for m, (_, w) in enumerate(cols):
        scores[rows, codes[m]] += w
    return classes[scores.argmax(axis=1)]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted median over axis 0 of ``values`` (M, ...) with positive ``weights`` (M,).

    The smallest value whose cumulative weight reaches half the total; when it reaches
    exactly half, the mean of that value and the next (so equal weights give ``np.median``).
    """
    order = np.argsort(values, axis=0, kind="stable")
    v = np.take_along_axis(values, order, axis=0)
    cum = np.cumsum(np.asarray(weights, dtype=np.float64)[order], axis=0)
    half = cum[-1] / 2.0
    k = np.argmax(cum >= half, axis=0)[None]
    lower = np.take_along_axis(v, k, axis=0)[0]
    upper = np.take_along_axis(v, np.minimum(k + 1, len(v) - 1), axis=0)[0]
    exact = np.take_along_axis(cum, k, axis=0)[0] == half
    return np.where(exact, (lower + upper) / 2.0, lower)


def _check_weights(weights, n: int) -> np.ndarray:
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"weights must be {n} finite non-negative numbers with a positive sum, got {weights!r}")
    return w


class _MetaLocalizer(BaseLocalizer):
    """Takes NaN / complex input exactly when every member does (members check their own input)."""

    def _members(self, *, required: bool = False) -> list:
        """``localizers`` as a list; a bare name or model (not a sequence) is a clear error."""
        members = self.localizers
        if isinstance(members, (str, BaseLocalizer)) or not hasattr(members, "__iter__"):
            raise TypeError(f"localizers must be a sequence of localizers or registry names, e.g. "
                            f"['wknn', 'horus'], got {members!r}")
        members = list(members)
        if required and not members:
            raise ValueError(f"{type(self).__name__} needs at least one localizer")
        return members

    @property
    def _allow_nan(self) -> bool:
        return accepts(self._members(), "_allow_nan")

    @property
    def _allow_complex(self) -> bool:
        return accepts(self._members(), "_allow_complex")


class EnsembleLocalizer(_MetaLocalizer):
    """Averaging ensemble: combine the positions of several localizers, vote their labels.

    Every member is fitted on the full training data. Positions are combined per query by
    ``combine``: "mean" (weighted by ``weights``), "median" (coordinate-wise weighted
    median, robust to one member's outliers) or "inverse_spread" (weights / spread**2, from
    each member's own ``Prediction.spread``; a member with spread 0 takes all the weight).
    Floor and building are a weighted majority vote of the members that predict them
    (ties: smallest label). ``Prediction.spread`` is the weighted RMS distance of the
    members' positions from the combined one (their disagreement).

    Parameters
    ----------
    localizers : sequence of localizers or registry names.
    weights : (M,) non-negative member weights, or None for equal weights.
    combine : "mean", "median" or "inverse_spread".

    References
    ----------
    Dietterich, T. G., "Ensemble methods in machine learning", Multiple Classifier Systems
    (MCS) 2000, LNCS 1857. DOI 10.1007/3-540-45014-9_1
    """

    def __init__(self, localizers=(), weights=None, combine: str = "mean"):
        self.localizers = localizers
        self.weights = weights
        self.combine = combine

    def _fit(self, X, pos, floor, building):
        members = self._members(required=True)
        if self.combine not in ("mean", "median", "inverse_spread"):
            raise ValueError(f"combine must be 'mean', 'median' or 'inverse_spread', got {self.combine!r}")
        weights = _check_weights(self.weights, len(members))
        self.localizers_ = [resolve_localizer(m).fit(X, pos, floor=floor, building=building) for m in members]
        self.weights_ = weights

    def _localize(self, X):
        keep = np.flatnonzero(self.weights_ > 0)
        preds = [self.localizers_[i].localize(X) for i in keep]
        w = self.weights_[keep]
        P = np.stack([p.pos for p in preds])  # (M, N, D)
        W = np.repeat(w[:, None], P.shape[1], axis=1)  # (M, N)
        if self.combine == "inverse_spread":
            if any(p.spread is None for p in preds):
                missing = [type(self.localizers_[i]).__name__ for i, p in zip(keep, preds) if p.spread is None]
                raise ValueError(f"combine='inverse_spread' needs Prediction.spread from every member; "
                                 f"{missing} return none")
            S2 = np.square(np.stack([p.spread for p in preds]))
            exact = S2 == 0
            with np.errstate(divide="ignore"):
                W = np.where(exact.any(axis=0), exact * W, W / S2)
        if self.combine == "median":
            pos = weighted_median(P, w)
        else:
            pos = np.einsum("mn,mnd->nd", W, P) / W.sum(axis=0)[:, None]
        spread = np.sqrt(np.einsum("mn,mn->n", W, np.square(P - pos).sum(-1)) / W.sum(axis=0))
        return Prediction(pos, _vote(preds, "floor", w), _vote(preds, "building", w), spread=spread)


def simplex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares convex combination: argmin ||y - sum_m w_m P_m||^2, w >= 0, sum w = 1.

    ``P`` is (M, N, D) member predictions and ``y`` (N, D) targets. Solved exactly by
    enumerating supports (smallest first; ties keep the earlier support) with the sum
    constraint eliminated, so it is translation invariant (coordinates may carry large
    offsets). Sums are einsum loops, not BLAS: the weights do not depend on threads.
    """
    M = len(P)
    if M > 12:
        raise ValueError(f"the default meta-learner enumerates supports and takes at most 12 members, got {M}; "
                         "pass final_estimator=...")
    A = P.reshape(M, -1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    best_rss, best = np.inf, None
    for size in range(1, M + 1):
        for S in itertools.combinations(range(M), size):
            last = S[-1]
            r = b - A[last]
            if size == 1:
                head = np.zeros(0)
                resid = r
            else:
                Dm = A[list(S[:-1])] - A[last]
                try:
                    head = np.linalg.solve(np.einsum("in,jn->ij", Dm, Dm), np.einsum("in,n->i", Dm, r))
                except np.linalg.LinAlgError:
                    continue  # collinear members: a smaller support gives the same fit
                resid = r - np.einsum("i,in->n", head, Dm)
            w_S = np.append(head, 1.0 - head.sum())
            if np.any(w_S < 0):
                continue
            rss = float(np.einsum("n,n->", resid, resid))
            if rss < best_rss:
                best_rss, best = rss, (S, w_S)
    w = np.zeros(M)
    w[list(best[0])] = best[1]
    return w


class StackingLocalizer(_MetaLocalizer):
    """Stacked generalization: a meta-learner fitted on out-of-fold member predictions.

    The training set is split into ``cv`` folds; each member is refitted on all folds but
    one and predicts the held-out fold, which gives every training scan an out-of-fold
    prediction from every member. The meta-learner is fitted on those predictions; the
    members are then refitted on all data for inference.

    Folds are deterministic. By default, scans that share a position (the repeated scans
    of a reference point) always share a fold, and distinct positions are dealt to the folds
    in sorted order, so an out-of-fold prediction never sees its own reference point; pass
    ``fit(..., groups=...)`` to group by something else (user, device, ...), or give ``cv``
    as a splitter object with ``split(X, y, groups)`` (e.g. sklearn's GroupKFold).

    The default meta-learner (``final_estimator=None``) is a constrained linear stack: member
    weights ``weights_`` fitted by exact least squares over all coordinate axes, non-negative
    (Breiman's constraint) and, as an addition, summing to one: the combination is convex, so
    it does not depend on where the coordinate frame puts its origin (an intercept-free fit
    without that constraint does). Any regressor with ``fit(Z, pos)``/``predict(Z)`` can
    replace it; it then sees Z = the members' predicted coordinates (plus the raw features
    if ``passthrough``). Floor and building: vote of the members, weighted by ``weights_``
    (equal weights with a custom meta-learner). ``member_errors_``: each member's mean
    out-of-fold position error.

    Save/load works when members and meta-learner are indoorloc localizers (or registry
    names) and ``cv`` is an int; a scikit-learn meta-learner or splitter object cannot be
    written without pickle.

    References
    ----------
    Wolpert, D. H., "Stacked generalization", Neural Networks 5(2), 1992.
    DOI 10.1016/S0893-6080(05)80023-1
    Breiman, L., "Stacked regressions", Machine Learning 24(1), 1996. DOI 10.1007/BF00117832
    """

    def __init__(self, localizers=(), final_estimator=None, cv=5, passthrough: bool = False):
        self.localizers = localizers
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    def _folds(self, pos, groups):
        n = len(pos)
        if hasattr(self.cv, "split"):
            return [(np.asarray(tr), np.asarray(te)) for tr, te in self.cv.split(np.zeros((n, 1)), pos, groups)]
        k = int(self.cv)
        if k < 2:
            raise ValueError(f"cv must be >= 2 folds or a splitter, got {self.cv!r}")
        if groups is None:
            _, group, *_ = group_rows(pos)
        else:
            groups = np.asarray(groups)
            if len(groups) != n:
                raise ValueError(f"groups has {len(groups)} rows, expected {n}")
            group = np.unique(groups, return_inverse=True)[1].reshape(-1)
        if group.max() + 1 < k:
            raise ValueError(f"cv={k} folds need at least {k} groups (distinct positions), got {group.max() + 1} "
                             f"group(s) in {n} sample(s)")
        fold = group % k
        return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]

    def _meta_features(self, P, X):
        Z = np.concatenate(list(P), axis=1)  # (N, M * D)
        return np.concatenate([Z, X.reshape(len(X), -1)], axis=1) if self.passthrough else Z

    def _fit(self, X, pos, floor, building, groups=None):
        templates = [resolve_localizer(m) for m in self._members(required=True)]
        take = lambda labels, rows: None if labels is None else labels[rows]  # noqa: E731
        oof = np.full((len(templates), *pos.shape), np.nan)
        for train, test in self._folds(pos, groups):
            for m, template in enumerate(templates):
                model = clone(template).fit(X[train], pos[train], floor=take(floor, train),
                                            building=take(building, train))
                oof[m, test] = model.localize(X[test]).pos
        if np.isnan(oof).any():
            raise ValueError("the cv folds must hold out every training sample exactly once")
        self.member_errors_ = np.sqrt(np.square(oof - pos).sum(-1)).mean(axis=1)
        if self.final_estimator is None:
            self.weights_ = simplex_weights(oof, pos)
            self.final_estimator_ = None
        else:
            self.weights_ = np.full(len(templates), 1.0 / len(templates))
            meta = self.final_estimator
            meta = resolve_localizer(meta) if isinstance(meta, str) else clone(meta)
            self.final_estimator_ = meta.fit(self._meta_features(oof, X), pos)
        self.localizers_ = [t.fit(X, pos, floor=floor, building=building) for t in templates]

    def _localize(self, X):
        preds = [m.localize(X) for m in self.localizers_]
        P = np.stack([p.pos for p in preds])
        spread = None
        if self.final_estimator_ is None:
            pos = np.einsum("m,mnd->nd", self.weights_, P)
            spread = np.sqrt(np.einsum("m,mn->n", self.weights_, np.square(P - pos).sum(-1)))
        else:
            pos = np.asarray(self.final_estimator_.predict(self._meta_features(P, X)),
                             dtype=np.float64).reshape(len(X), -1)
        return Prediction(pos, _vote(preds, "floor", self.weights_), _vote(preds, "building", self.weights_),
                          spread=spread)


__all__ = ["EnsembleLocalizer", "StackingLocalizer", "resolve_localizer", "simplex_weights", "weighted_median"]
