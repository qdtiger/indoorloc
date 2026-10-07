"""L4 evaluation protocols: which rows train a model and which rows test it.

Every protocol is a function of plain arrays (a sample count, a label column or a group
column of a :class:`~indoorloc.core.SampleTable`) that returns sorted ``int64`` index arrays
``(train_idx, test_idx)`` into that table, never copies of the data. A split is therefore
cheap, auditable (``np.save`` the indices, or hash them with :func:`split_summary`) and
applies to every column of the table at once::

    train_idx, test_idx = cross_time_split(table.groups["time"], test_size=0.2)
    model.fit(table[train_idx]); model.evaluate(table[test_idx])

All randomness goes through ``np.random.default_rng(random_state)``: the same seed gives the
same indices on every platform. numpy does not promise that ``Generator`` streams never change
between versions, so result files record the sha256 of the indices (:func:`split_summary`);
a changed split is detected, never silent.

Single-split functions return one ``(train_idx, test_idx)`` pair; multi-fold functions
(:func:`kfold`, :func:`leave_one_group_out`) return a list of such pairs. The named
protocols in :data:`PROTOCOLS` (``"official"``, ``"random-80-20"``, ``"cross-device"``, ...)
wrap these functions for whole tables and label every fold; the command line uses them.

References
----------
F. Potortì et al., "Comparing the Performance of Indoor Localization Systems through the
EvAAL Framework", Sensors 17(10):2327, 2017. DOI: 10.3390/s17102327 (why evaluation must
use data and conditions independent of the training data).
G. M. Mendoza-Silva, P. Richter, J. Torres-Sospedra, E. S. Lohan, J. Huerta, "Long-Term
WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning", Data 3(1):3, 2018.
DOI: 10.3390/data3010003 (train on early months, test on later ones: cross-time).
L. Klus et al., "TUJI1 Dataset: Multi-device dataset for indoor localization with high
measurement density", Data in Brief 54:110356, 2024. DOI: 10.1016/j.dib.2024.110356
(hold out one device: cross-device).
F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12:2825-2830, 2011.
URL: https://jmlr.org/papers/v12/pedregosa11a.html (KFold / GroupKFold / LeaveOneGroupOut
conventions followed here, re-implemented on numpy).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import numpy as np

from ..core import Registry, SampleTable

Split = tuple[np.ndarray, np.ndarray]  # (train_idx, test_idx): two sorted int64 arrays


def _num_samples(n) -> int:
    """A count from an int, a SampleTable or any array-like with a first axis."""
    if isinstance(n, (int, np.integer)) and not isinstance(n, bool):
        count = int(n)
    elif isinstance(n, SampleTable):
        count = len(n)
    else:
        count = len(np.asarray(n))
    if count < 2:
        raise ValueError(f"a split needs at least 2 samples, got {count}")
    return count


def _column(values, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D column with one value per sample, got shape {arr.shape}")
    if len(arr) < 2:
        raise ValueError(f"a split needs at least 2 samples, got {len(arr)}")
    if arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError(f"{name} holds NaN; unknown values must be left out of the table (rule: no sentinels)")
    return arr


def _n_test(n: int, test_size) -> int:
    """sklearn's rule: a float is a fraction (rounded up), an int is a count.

    The product is rounded to 9 decimals before ``ceil`` so that e.g. 0.7 * 10 gives 7, not
    the 8 that ``ceil(7.000000000000001)`` would.
    """
    if isinstance(test_size, (float, np.floating)):
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size as a fraction must be in (0, 1), got {test_size}")
        k = math.ceil(round(float(test_size) * n, 9))
    elif isinstance(test_size, (int, np.integer)) and not isinstance(test_size, bool):
        k = int(test_size)
    else:
        raise TypeError(f"test_size must be a float fraction or an int count, got {test_size!r}")
    if not 1 <= k <= n - 1:
        raise ValueError(f"test_size={test_size} leaves an empty train or test set for n={n}")
    return k


def _pair(train_mask: np.ndarray, test_mask: np.ndarray, what: str) -> Split:
    train, test = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"{what}: the {'train' if len(train) == 0 else 'test'} set is empty")
    return train.astype(np.int64), test.astype(np.int64)


# --------------------------------------------------------------------------- random splits
def random_split(n, test_size=0.2, *, stratify=None, random_state=0) -> Split:
    """Uniformly random hold-out split (``sklearn.model_selection.train_test_split`` rules).

    n             sample count, a SampleTable or any array with samples on the first axis.
    test_size     fraction in (0, 1) of the rows (rounded up) or an absolute count.
    stratify      optional (N,) labels (e.g. ``building * 100 + floor``); every class then
                  contributes to the test set in proportion to its size. The per-class counts
                  are allocated by largest remainder so they sum exactly to the test size
                  (ties go to the class that sorts first).
    random_state  seed for ``np.random.default_rng``.

    Returns sorted ``(train_idx, test_idx)``. Randomly split fingerprints from the same
    survey point land on both sides, so this protocol is optimistic for fingerprinting
    compared with the dataset's official split; prefer grouped protocols when groups exist.
    """
    count = _num_samples(n)
    k = _n_test(count, test_size)
    rng = np.random.default_rng(random_state)
    test = np.zeros(count, dtype=bool)
    if stratify is None:
        test[rng.permutation(count)[:k]] = True
    else:
        labels = _column(stratify, "stratify")
        if len(labels) != count:
            raise ValueError(f"stratify has {len(labels)} rows, expected {count}")
        classes, codes, sizes = np.unique(labels, return_inverse=True, return_counts=True)
        quota = sizes * (k / count)
        take = np.floor(quota).astype(np.int64)
        order = np.argsort(-(quota - take), kind="stable")  # largest remainder, ties by class order
        take[order[:k - take.sum()]] += 1
        for c in range(len(classes)):  # classes in sorted order, one generator: deterministic
            rows = np.flatnonzero(codes == c)
            test[rows[rng.permutation(len(rows))[:take[c]]]] = True
    return _pair(~test, test, "random_split")


def kfold(n, n_splits: int = 5, *, groups=None, shuffle: bool = True, random_state=0) -> list[Split]:
    """K folds; each row (or each group, if ``groups`` is given) is tested exactly once.

    Without ``groups``: rows (shuffled if ``shuffle``) are cut into ``n_splits`` contiguous
    folds whose sizes differ by at most one, larger folds first (sklearn ``KFold``).
    With ``groups`` (N,): no group is split across train and test. Groups are assigned,
    largest first, to the fold with the fewest rows so far (sklearn ``GroupKFold``); ties
    between equal-sized groups follow a seeded shuffle if ``shuffle`` else the sorted group
    order, ties between folds go to the lowest fold index.

    Returns a list of ``n_splits`` sorted ``(train_idx, test_idx)`` pairs.
    """
    count = _num_samples(n)
    if groups is not None and len(np.asarray(groups)) != count:
        raise ValueError(f"groups has {len(np.asarray(groups))} rows, expected {count}")
    if not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits!r}")
    rng = np.random.default_rng(random_state)
    fold_of = np.empty(count, dtype=np.int64)
    if groups is None:
        if n_splits > count:
            raise ValueError(f"n_splits={n_splits} exceeds the number of samples ({count})")
        order = rng.permutation(count) if shuffle else np.arange(count)
        for f, part in enumerate(np.array_split(order, n_splits)):
            fold_of[part] = f
    else:
        g = _column(groups, "groups")
        _, codes, sizes = np.unique(g, return_inverse=True, return_counts=True)
        if n_splits > len(sizes):
            raise ValueError(f"n_splits={n_splits} exceeds the number of groups ({len(sizes)})")
        tiebreak = rng.permutation(len(sizes)) if shuffle else np.arange(len(sizes))
        order = np.lexsort((tiebreak, -sizes))  # largest group first
        load = np.zeros(n_splits, dtype=np.int64)
        group_fold = np.empty(len(sizes), dtype=np.int64)
        for gi in order:
            f = int(np.argmin(load))  # lowest index among the lightest folds
            group_fold[gi] = f
            load[f] += sizes[gi]
        fold_of = group_fold[codes]
    return [_pair(fold_of != f, fold_of == f, f"kfold fold {f}") for f in range(n_splits)]


# --------------------------------------------------------------------------- grouped splits
def group_split(groups, test_groups, *, train_groups=None) -> Split:
    """Rows whose group is in ``test_groups`` test; the rest (or ``train_groups``) train.

    groups        (N,) group column, e.g. ``table.groups["device"]`` or ``table.building``.
    test_groups   a value or a list of values held out for testing.
    train_groups  optional values to train on; rows in neither list are left out
                  (use it to train on a single device, month, building, ...).

    Raises if a listed value never occurs (a typo would otherwise pass silently) or if the
    two lists overlap.
    """
    g = _column(groups, "groups")
    test_values = np.atleast_1d(np.asarray(test_groups))
    present = np.unique(g)
    missing = np.setdiff1d(test_values, present)
    if len(missing):
        raise ValueError(f"test_groups {missing.tolist()} do not occur; available: {present.tolist()}")
    test = np.isin(g, test_values)
    if train_groups is None:
        train = ~test
    else:
        train_values = np.atleast_1d(np.asarray(train_groups))
        missing = np.setdiff1d(train_values, present)
        if len(missing):
            raise ValueError(f"train_groups {missing.tolist()} do not occur; available: {present.tolist()}")
        if len(np.intersect1d(train_values, test_values)):
            raise ValueError("train_groups and test_groups overlap")
        train = np.isin(g, train_values)
    return _pair(train, test, "group_split")


def leave_one_group_out(groups) -> list[Split]:
    """One fold per distinct group value: fold ``i`` tests on ``np.unique(groups)[i]``.

    E.g. ``leave_one_group_out(table.building)`` (leave one building out) or
    ``leave_one_group_out(table.groups["device"])`` (leave one device out).
    """
    g = _column(groups, "groups")
    values = np.unique(g)
    if len(values) < 2:
        raise ValueError(f"leave_one_group_out needs at least 2 groups, got {values.tolist()}")
    return [_pair(g != v, g == v, f"leave out {v!r}") for v in values]


def cross_device_split(device, test_devices, *, train_devices=None) -> Split:
    """Train on some devices, test on others (device heterogeneity).

    ``device`` is the (N,) device column (``table.groups["device"]``). Rows of
    ``test_devices`` test; the remaining rows (or only ``train_devices``) train.
    A thin, named wrapper of :func:`group_split`; for every device in turn use
    ``leave_one_group_out(device)`` or the ``"cross-device"`` protocol.
    """
    return group_split(device, test_devices, train_groups=train_devices)


def cross_time_split(time, *, test_size=0.2, cutoff=None, gap=0.0) -> Split:
    """Train on the past, test on the future (signal drift, AP changes, furniture).

    time       (N,) sortable column: unix seconds, a month number, a session index, ...
    cutoff     rows with ``time >= cutoff`` test. If None, the cutoff is the time of the
               first row of the latest ``test_size`` fraction (count rounded up):
               ``np.sort(time)[N - n_test]``. Rows sharing the cutoff time all test, so
               the test set can be slightly larger than ``n_test`` but a timestamp is never
               split between the two sides.
    gap        rows with ``cutoff - gap <= time < cutoff`` are dropped from training (a
               buffer against near-duplicate scans at the boundary); same units as ``time``.

    Returns sorted ``(train_idx, test_idx)``; every training time precedes every test time.
    """
    t = _column(time, "time")
    if cutoff is None:
        cutoff = np.sort(t)[len(t) - _n_test(len(t), test_size)]
    test = t >= cutoff
    train = t < cutoff - gap if gap else ~test
    return _pair(train, test, f"cross_time_split(cutoff={cutoff!r})")


# --------------------------------------------------------------------------- tables and audit
def _same_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise equality of two equally shaped arrays, NaN equal to NaN (missing == missing)."""
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    both_nan = np.isnan(a) & np.isnan(b) if a.dtype.kind in "fc" and b.dtype.kind in "fc" else False
    return np.all((a == b) | both_nan, axis=1)


def pool_splits(tables: Mapping[str, SampleTable]) -> SampleTable:
    """Stack a dataset's split tables into one table and record where each row came from.

    ``groups["split"]`` names the source split of every row (the ``"official"`` protocol
    tests on ``"test"``). Group columns that some split does not know are dropped and listed
    in ``meta["unknown_groups"]``; floor/building labels become None unless every split has
    them. ``meta["sha256"]`` maps each split to its file digest(s).

    Ids decide what a row is. A row whose id already occurred in an earlier split, with the
    same features and position, is the same sample listed twice (e.g. an ``"all"`` split that
    is the union of ``"train"`` and ``"test"``): it is kept once, under the earlier split, and
    ``meta["pooled_duplicates"]`` counts the dropped rows per split. Without this a random
    split of the pool would test on scans that are also in the training set. Ids that repeat
    with *different* contents are per-split row numbers, not sample ids: every id is then
    prefixed with ``"<split>/"``. A mixture of both is refused.
    """
    items = [(str(name), t) for name, t in tables.items()]
    if not items:
        raise ValueError("pool_splits needs at least one table")
    shapes = {t.X.shape[1:] for _, t in items}
    if len(shapes) != 1:
        raise ValueError(f"the splits have different feature shapes {sorted(shapes)}; they cannot be pooled")
    names = [t.meta.get("feature_names") for _, t in items]
    if all(n is not None for n in names) and len({tuple(n) for n in names}) > 1:
        raise ValueError("the splits list their features (meta['feature_names']) in different orders or sets; "
                         "align the columns before pooling")
    common = set.intersection(*(set(t.groups) for _, t in items)) - {"split"}
    X = np.concatenate([t.X for _, t in items])
    pos = np.concatenate([t.pos for _, t in items])
    ids = np.concatenate([t.ids for _, t in items])
    source = np.repeat(np.arange(len(items)), [len(t) for _, t in items])
    keep = np.ones(len(ids), dtype=bool)
    dropped = {}
    _, first_of, inverse = np.unique(ids, return_index=True, return_inverse=True)
    first = first_of[inverse.ravel()]  # row of the first occurrence of each row's id
    repeat = np.flatnonzero(first != np.arange(len(ids)))
    if len(repeat):
        if np.any(source[repeat] == source[first[repeat]]):
            raise ValueError("a split repeats an id within itself; ids must identify samples uniquely")
        same = _same_rows(X[repeat], X[first[repeat]]) & _same_rows(pos[repeat], pos[first[repeat]])
        if same.all():  # the same samples listed in two splits: keep the first listing
            keep[repeat] = False
            counts = np.bincount(source[repeat], minlength=len(items))
            dropped = {items[i][0]: int(c) for i, c in enumerate(counts) if c}
        elif not same.any():  # row numbers reused by every split: make them unique
            ids = np.concatenate([np.char.add(f"{name}/", t.ids.astype(str)) for name, t in items])
        else:
            a, b = items[source[first[repeat[0]]]][0], items[source[repeat[0]]][0]
            raise ValueError(f"splits {a!r} and {b!r} share {len(repeat)} ids, {int(same.sum())} of them with "
                             "identical rows and the others with different rows; ids must identify samples")
    rows = slice(None) if keep.all() else keep  # a slice keeps the (large) feature array a view
    labels = {}
    for field in ("floor", "building"):
        cols = [getattr(t, field) for _, t in items]
        labels[field] = None if any(c is None for c in cols) else np.concatenate(cols)[rows]
    groups = {k: np.concatenate([np.asarray(t.groups[k]) for _, t in items])[rows] for k in sorted(common)}
    groups["split"] = np.concatenate([np.full(len(t), name) for name, t in items])[rows]
    unknown = set().union(*(set(t.groups) for _, t in items)) - common - {"split"}
    unknown |= set().union(*(set(t.meta.get("unknown_groups", ())) for _, t in items))
    meta = {**items[0][1].meta, "split": tuple(n for n, _ in items),
            "sha256": {n: t.meta.get("sha256") for n, t in items},
            "unknown_groups": tuple(sorted(unknown - set(groups)))}
    if dropped:
        meta["pooled_duplicates"] = dropped
    return SampleTable(X[rows], pos[rows], labels["floor"], labels["building"], groups, ids[rows], meta)


def split_summary(train_idx, test_idx, n: int | None = None) -> dict:
    """Sizes, a disjointness check and sha256 digests of a split (for result files).

    The digests (of the int64 little-endian index bytes) let anyone confirm that two runs
    used exactly the same rows without storing the indices.
    """
    train = np.asarray(train_idx, dtype="<i8")
    test = np.asarray(test_idx, dtype="<i8")
    overlap = int(len(np.intersect1d(train, test)))
    if overlap:
        raise ValueError(f"train and test share {overlap} row(s)")
    out = {"n_train": int(len(train)), "n_test": int(len(test)),
           "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
           "test_sha256": hashlib.sha256(test.tobytes()).hexdigest()}
    if n is not None:
        out["n_unused"] = int(n - len(train) - len(test))
    return out


# --------------------------------------------------------------------------- named protocols
class Fold(NamedTuple):
    """One labelled split of a table: ``name`` (e.g. ``"device=13"``), ``train``, ``test``."""

    name: str
    train: np.ndarray
    test: np.ndarray


def _one_table(table, protocol: str) -> SampleTable:
    """A protocol splits ONE table: the ``(train, test)`` of ``load_dataset`` must be pooled first."""
    if isinstance(table, (tuple, list, Mapping)):
        raise TypeError(f"protocol {protocol!r} splits one SampleTable, got a {type(table).__name__} of "
                        f"{len(table)} items (e.g. the (train, test) of load_dataset). Pool the splits first: table = "
                        "indoorloc.evaluation.pool_splits({'train': train, 'test': test}) (groups['split'] records "
                        f"where each row came from), then get_protocol({protocol!r}).folds(table)")
    return table


def _rows(table, protocol: str) -> int:
    return len(_one_table(table, protocol))


def _need_group(table: SampleTable, key: str, protocol: str) -> np.ndarray:
    _one_table(table, protocol)
    if key == "building":
        values = table.building
    elif key == "floor":
        values = table.floor
    else:
        values = table.groups.get(key)
    if values is None:
        known = sorted(table.groups)
        unknown = list(table.meta.get("unknown_groups", ()))
        raise ValueError(f"protocol {protocol!r} needs {key!r} labels, which this table does not have "
                         f"(groups: {known}; unknown: {unknown})")
    return np.asarray(values)


@dataclass(frozen=True)
class Protocol:
    """A named, documented rule that turns a table into labelled folds.

    ``split(table, random_state)`` returns ``[Fold, ...]``; :meth:`folds` checks every fold
    (non-empty, sorted unique row indices inside the table, train and test disjoint).
    ``needs`` names the table columns it reads. A protocol splits one table: pool a dataset's
    ``(train, test)`` with :func:`pool_splits` first (a tuple is a TypeError that says so).
    """

    name: str
    summary: str
    split: Callable[[SampleTable, int], list[Fold]]
    needs: tuple[str, ...] = ()

    def folds(self, table: SampleTable, *, random_state=0) -> list[Fold]:
        """The checked folds of ``table`` (one SampleTable; see :func:`pool_splits`)."""
        folds = list(self.split(_one_table(table, self.name), random_state))
        if not folds:
            raise ValueError(f"protocol {self.name!r} produced no folds")
        for fold in folds:
            for side in (fold.train, fold.test):
                side = np.asarray(side)
                if side.ndim != 1 or len(side) == 0 or side.dtype.kind not in "iu":
                    raise ValueError(f"{self.name} / {fold.name}: folds must be non-empty 1-D integer index arrays")
                if np.any(np.diff(side) <= 0) or side[0] < 0 or side[-1] >= len(table):
                    raise ValueError(f"{self.name} / {fold.name}: indices must be sorted, unique and < {len(table)}")
            split_summary(fold.train, fold.test)  # raises on overlap
        return folds


def _official(table, _seed):
    split = _need_group(table, "split", "official")
    if not {"train", "test"} <= set(np.unique(split).tolist()):
        raise ValueError(f"the official protocol needs the dataset's own 'train' and 'test' splits; this "
                         f"table has {sorted(set(split.tolist()))}. Choose another protocol "
                         f"({', '.join(n for n in list_protocols() if n != 'official')})")
    train, test = group_split(split, ["test"], train_groups=["train"])
    return [Fold("official", train, test)]


def _logo(key: str, label: str, protocol: str):
    def split(table, _seed):
        column = _need_group(table, key, protocol)
        return [Fold(f"{label}={v}", tr, te) for v, (tr, te) in zip(np.unique(column), leave_one_group_out(column))]
    return split


def _group_kfold(key: str, protocol: str, n_splits: int = 5):
    def split(table, seed):
        column = _need_group(table, key, protocol)
        return [Fold(f"{key} fold={i}", tr, te)
                for i, (tr, te) in enumerate(kfold(len(table), n_splits, groups=column, random_state=seed))]
    return split


def _within_month(table, _seed):
    """Mendoza-Silva et al. (2018): for each month, its own training sets vs its own test sets."""
    split = _need_group(table, "split", "within-month")
    month = _need_group(table, "month", "within-month")
    folds = []
    for m in np.unique(month):
        train = np.flatnonzero((month == m) & (split == "train")).astype(np.int64)
        test = np.flatnonzero((month == m) & (split == "test")).astype(np.int64)
        if len(train) and len(test):
            folds.append(Fold(f"month={m}", train, test))
    return folds


PROTOCOLS = Registry("protocol", {
    "official": Protocol(
        "official", "the dataset's own train/test files (the setting of most published numbers)", _official,
        ("groups['split']",)),
    "random-80-20": Protocol(
        "random-80-20", "uniformly random 80 % train / 20 % test rows (seeded; optimistic for fingerprinting)",
        lambda t, seed: [Fold("random-80-20", *random_split(_rows(t, "random-80-20"), 0.2, random_state=seed))]),
    "kfold-5": Protocol(
        "kfold-5", "5-fold cross-validation over shuffled rows (seeded)",
        lambda t, seed: [Fold(f"fold={i}", tr, te)
                         for i, (tr, te) in enumerate(kfold(_rows(t, "kfold-5"), 5, random_state=seed))]),
    "cross-device": Protocol(
        "cross-device", "leave one device out: each device tests once, trained on all other devices",
        _logo("device", "device", "cross-device"), ("groups['device']",)),
    "cross-time": Protocol(
        "cross-time", "train on the earliest 80 % of rows by time, test on the latest 20 % (no timestamp split)",
        lambda t, seed: [Fold("cross-time", *cross_time_split(_need_group(t, "time", "cross-time"), test_size=0.2))],
        ("groups['time']",)),
    "leave-one-building-out": Protocol(
        "leave-one-building-out", "each building tests once, trained on the other buildings (transfer)",
        _logo("building", "building", "leave-one-building-out"), ("building",)),
    "leave-one-user-out": Protocol(
        "leave-one-user-out", "each user tests once, trained on the other users (EvAAL: test users unseen in training)",
        _logo("user", "user", "leave-one-user-out"), ("groups['user']",)),
    "leave-one-trajectory-out": Protocol(
        "leave-one-trajectory-out", "each recorded trajectory tests once, trained on all others (no within-walk leakage)",
        _logo("trajectory", "trajectory", "leave-one-trajectory-out"), ("groups['trajectory']",)),
    "trajectory-kfold-5": Protocol(
        "trajectory-kfold-5", "5 folds of whole trajectories (seeded): no walk is split between train and test",
        _group_kfold("trajectory", "trajectory-kfold-5"), ("groups['trajectory']",)),
    "point-kfold-5": Protocol(
        "point-kfold-5", "5 folds of whole reference points (seeded): a tested point is never seen in training",
        _group_kfold("point", "point-kfold-5"), ("groups['point']",)),
    "within-month": Protocol(
        "within-month", "for every month, train on its training sets and test on its test sets (LongTermWiFi)",
        _within_month, ("groups['split']", "groups['month']")),
})


def get_protocol(name: str) -> Protocol:
    """The registered :class:`Protocol` called ``name`` (case-insensitive)."""
    return PROTOCOLS.get(name)


def list_protocols() -> list[str]:
    """Names of the registered evaluation protocols (built-in and ``register_protocol``), sorted."""
    return PROTOCOLS.names()


def register_protocol(name: str, protocol: Protocol, *, force: bool = False) -> Protocol:
    """Add a named protocol (e.g. a fixed device pair) for the CLI and ``get_protocol``."""
    if not isinstance(protocol, Protocol):
        raise TypeError("register_protocol expects a Protocol(name, summary, split)")
    return PROTOCOLS.register(name, protocol, force=force)


__all__ = ["PROTOCOLS", "Fold", "Protocol", "cross_device_split", "cross_time_split", "get_protocol", "group_split",
           "kfold", "leave_one_group_out", "list_protocols", "pool_splits", "random_split", "register_protocol",
           "split_summary"]
