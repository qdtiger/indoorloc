"""Named evaluation protocols the benchmark needs beyond the ones built into indoorloc.

Each object is an :class:`indoorloc.evaluation.Protocol` assembled from the public L4 split
functions, so the command line uses it by path, from the repository root::

    indoorloc benchmark --dataset hwild --dataset-option environment=office \\
        --protocol benchmarks.protocols:LEAVE_ONE_USER_OUT --preprocess CSIAmplitude --method wknn

``stratified_kfold`` serves the label-only runner (``benchmarks/labels.py``): the library's
``kfold`` groups rows but does not stratify them.

Every protocol returns sorted, disjoint ``int64`` index arrays, and the result file records the
sha256 of every fold's indices (``split_summary``), so a changed split never passes silently.

References
----------
F. Potortì et al., "Comparing the Performance of Indoor Localization Systems through the EvAAL
Framework", Sensors 17(10):2327, 2017. DOI: 10.3390/s17102327 (test data must be independent of
the training data: other users, other positions, later campaigns).
G. M. Mendoza-Silva, P. Richter, J. Torres-Sospedra, E. S. Lohan, J. Huerta, "Long-Term WiFi
Fingerprinting Dataset for Research on Robust Indoor Positioning", Data 3(1):3, 2018.
DOI: 10.3390/data3010003 (the within-month protocol).
R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model
selection", IJCAI 1995. URL: https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf
(stratified k-fold cross-validation).
"""
from __future__ import annotations

import numpy as np

from indoorloc.evaluation import Fold, Protocol, kfold, leave_one_group_out

__all__ = ["LEAVE_ONE_USER_OUT", "POINT_KFOLD_5", "WITHIN_MONTH", "stratified_kfold"]


def _group(table, key: str, protocol: str) -> np.ndarray:
    if key not in table.groups:
        raise ValueError(f"protocol {protocol!r} needs groups[{key!r}]; this table has {sorted(table.groups)}")
    return np.asarray(table.groups[key])


def _leave_one_user_out(table, _seed) -> list[Fold]:
    users = _group(table, "user", "leave-one-user-out")
    return [Fold(f"user={u}", tr, te) for u, (tr, te) in zip(np.unique(users), leave_one_group_out(users))]


def _point_kfold(table, seed) -> list[Fold]:
    points = _group(table, "point", "point-kfold-5")
    return [Fold(f"points fold={i}", tr, te) for i, (tr, te) in enumerate(kfold(len(table), 5, groups=points,
                                                                                 random_state=seed))]


def _within_month(table, _seed) -> list[Fold]:
    split = _group(table, "split", "within-month")
    month = _group(table, "month", "within-month")
    folds = []
    for m in np.unique(month):
        train = np.flatnonzero((month == m) & (split == "train")).astype(np.int64)
        test = np.flatnonzero((month == m) & (split == "test")).astype(np.int64)
        if len(train) and len(test):
            folds.append(Fold(f"month={int(m)}", train, test))
    return folds


LEAVE_ONE_USER_OUT = Protocol(
    "leave-one-user-out",
    "each user (groups['user']) tests once, trained on the other users of the same selection",
    _leave_one_user_out, ("groups['user']",))

POINT_KFOLD_5 = Protocol(
    "point-kfold-5",
    "5 folds over reference points (groups['point']): every point is tested once and never seen in training "
    "(GroupKFold rule of indoorloc.evaluation.kfold, seeded)",
    _point_kfold, ("groups['point']",))

WITHIN_MONTH = Protocol(
    "within-month",
    "the dataset authors' protocol: for every month m, train on m's training sets and test on m's test sets",
    _within_month, ("groups['split']", "groups['month']"))


def stratified_kfold(labels, n_splits: int = 5, *, random_state=0) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds in which every class is spread as evenly as possible (sklearn ``StratifiedKFold`` idea).

    The rows of each class, in sorted class order, are shuffled with one
    ``np.random.default_rng(random_state)`` and dealt to the folds in turn; the dealing continues
    where the previous class stopped, so fold sizes differ by at most one. Returns ``n_splits``
    sorted ``(train_idx, test_idx)`` pairs; every row is tested exactly once.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) < n_splits:
        raise ValueError(f"labels must be 1-D with at least n_splits={n_splits} rows, got shape {y.shape}")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits!r}")
    classes, codes = np.unique(y, return_inverse=True)
    counts = np.bincount(codes)
    if counts.min() < n_splits:
        raise ValueError(f"class {classes[counts.argmin()]!r} has {counts.min()} rows, fewer than n_splits={n_splits}")
    rng = np.random.default_rng(random_state)
    fold_of = np.empty(len(y), dtype=np.int64)
    start = 0
    for c in range(len(classes)):
        rows = np.flatnonzero(codes == c)[rng.permutation(counts[c])]
        fold_of[rows] = (start + np.arange(len(rows))) % n_splits
        start = (start + len(rows)) % n_splits
    return [(np.flatnonzero(fold_of != f).astype(np.int64), np.flatnonzero(fold_of == f).astype(np.int64))
            for f in range(n_splits)]
