from __future__ import annotations

import hashlib

import numpy as np
import pytest

from indoorloc.core import SampleTable
from indoorloc.evaluation import (PROTOCOLS, Protocol, cross_device_split, cross_time_split, get_protocol, group_split,
                                  kfold, leave_one_group_out, list_protocols, pool_splits, random_split,
                                  register_protocol, split_summary)


def _is_partition(train, test, n):
    assert train.dtype == np.int64 and test.dtype == np.int64
    assert np.all(np.diff(train) > 0) and np.all(np.diff(test) > 0)  # sorted, unique
    assert len(np.intersect1d(train, test)) == 0
    return np.array_equal(np.union1d(train, test), np.arange(n))


# --------------------------------------------------------------------------- random_split
def test_random_split_sizes_follow_sklearn_rounding_and_seed():
    train, test = random_split(10, 0.25, random_state=3)
    assert len(test) == 3 and _is_partition(train, test, 10)  # ceil(2.5) = 3, as sklearn
    assert len(random_split(10, 0.7)[1]) == 7  # 0.7 * 10 = 7.000000000000001 must not round up to 8
    assert len(random_split(10, 4)[1]) == 4  # an int is a count
    again = random_split(10, 0.25, random_state=3)
    assert np.array_equal(test, again[1])
    assert any(not np.array_equal(test, random_split(10, 0.25, random_state=s)[1]) for s in range(4, 10))


def test_random_split_accepts_a_table_and_rejects_degenerate_sizes():
    table = SampleTable(np.zeros((6, 1)), np.zeros((6, 2)))
    assert _is_partition(*random_split(table, 0.5), 6)
    for bad in (0.0, 1.0, 0, 6):
        with pytest.raises(ValueError):
            random_split(6, bad)
    with pytest.raises(ValueError, match="at least 2"):
        random_split(1)


def test_stratified_split_allocates_by_largest_remainder():
    labels = np.repeat([7, 8, 9], [50, 30, 20])
    train, test = random_split(len(labels), 0.2, stratify=labels, random_state=0)
    assert np.bincount(labels[test] - 7).tolist() == [10, 6, 4] and _is_partition(train, test, 100)
    # quotas 1.5, 1.5, 2.0 for 5 test rows: floors 1, 1, 2 and the one extra row goes to the first
    # of the tied remainders (class 0)
    labels = np.repeat([0, 1, 2], [3, 3, 4])
    _, test = random_split(10, 0.5, stratify=labels, random_state=1)
    assert np.bincount(labels[test]).tolist() == [2, 1, 2]


# --------------------------------------------------------------------------- kfold
def test_kfold_tests_every_row_once_with_sklearn_fold_sizes():
    folds = kfold(10, 3, shuffle=False)
    assert [f[1].tolist() for f in folds] == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]  # sklearn KFold sizes
    shuffled = kfold(10, 3, random_state=0)
    assert sorted(np.concatenate([t for _, t in shuffled]).tolist()) == list(range(10))
    assert all(_is_partition(tr, te, 10) for tr, te in shuffled)
    assert [t.tolist() for _, t in kfold(10, 3, random_state=0)] == [t.tolist() for _, t in shuffled]


def test_group_kfold_assigns_largest_groups_to_the_lightest_fold():
    groups = np.repeat(["a", "b", "c", "d", "e"], [5, 4, 3, 2, 1])
    folds = kfold(len(groups), 2, groups=groups, shuffle=False)
    # a -> 0 (5, 0); b -> 1 (5, 4); c -> 1 (5, 7); d -> 0 (7, 7); e -> 0 on the tie (8, 7)
    assert [sorted(set(groups[te])) for _, te in folds] == [["a", "d", "e"], ["b", "c"]]
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
    with pytest.raises(ValueError, match="exceeds the number of groups"):
        kfold(len(groups), 6, groups=groups)
    with pytest.raises(ValueError, match="rows"):
        kfold(3, 2, groups=groups)


# --------------------------------------------------------------------------- grouped splits
def test_group_split_and_leave_one_group_out():
    device = np.array([3, 1, 2, 3, 1, 2, 2])
    train, test = group_split(device, [2])
    assert test.tolist() == [2, 5, 6] and train.tolist() == [0, 1, 3, 4]
    train, test = cross_device_split(device, 2, train_devices=[3])
    assert test.tolist() == [2, 5, 6] and train.tolist() == [0, 3]  # device 1 left out
    with pytest.raises(ValueError, match=r"do not occur; available: \[1, 2, 3\]"):
        group_split(device, [9])
    with pytest.raises(ValueError, match="overlap"):
        group_split(device, [2], train_groups=[2, 3])
    folds = leave_one_group_out(device)
    assert [te.tolist() for _, te in folds] == [[1, 4], [2, 5, 6], [0, 3]]  # np.unique order: 1, 2, 3
    assert all(_is_partition(tr, te, 7) for tr, te in folds)
    with pytest.raises(ValueError, match="at least 2 groups"):
        leave_one_group_out([1, 1, 1])
    with pytest.raises(ValueError, match="NaN"):
        leave_one_group_out([1.0, np.nan, 2.0])


def test_group_split_works_on_string_groups():
    split = np.array(["train", "train", "test", "extra"])
    train, test = group_split(split, ["test"], train_groups=["train"])
    assert train.tolist() == [0, 1] and test.tolist() == [2]


def test_cross_time_split_never_splits_a_timestamp():
    t = np.array([5, 1, 9, 2, 8, 3, 7, 4, 6, 5])
    train, test = cross_time_split(t, test_size=0.2)  # 2 test rows: times 8 and 9
    assert sorted(t[test].tolist()) == [8, 9] and t[train].max() < t[test].min()
    ties = np.array([1, 2, 2, 2])
    train, test = cross_time_split(ties, test_size=0.5)  # cutoff = sorted[2] = 2: all three 2s test
    assert train.tolist() == [0] and test.tolist() == [1, 2, 3]
    train, test = cross_time_split(t, cutoff=5, gap=2)
    assert t[train].max() < 3 and t[test].min() == 5  # times 3 and 4 are dropped
    with pytest.raises(ValueError, match="empty"):
        cross_time_split(t, cutoff=100)


# --------------------------------------------------------------------------- tables and audit
def _tables():
    train = SampleTable(np.arange(8.0).reshape(4, 2), np.zeros((4, 2)), floor=[0, 0, 1, 1], building=[0, 0, 0, 0],
                        groups={"device": [1, 1, 2, 2], "user": [5, 5, 6, 6]}, meta={"sha256": "aa", "name": "t"})
    test = SampleTable(np.arange(4.0).reshape(2, 2), np.ones((2, 2)), floor=[0, 1], building=[0, 0],
                       groups={"device": [2, 3]}, meta={"sha256": "bb", "unknown_groups": ("user",)})
    return train, test


def test_pool_splits_keeps_common_groups_and_records_sources():
    train, test = _tables()
    pooled = pool_splits({"train": train, "test": test})
    assert len(pooled) == 6 and pooled.groups["split"].tolist() == ["train"] * 4 + ["test"] * 2
    assert sorted(pooled.groups) == ["device", "split"] and pooled.meta["unknown_groups"] == ("user",)
    assert pooled.ids.tolist() == ["train/0", "train/1", "train/2", "train/3", "test/0", "test/1"]  # made unique
    assert pooled.meta["sha256"] == {"train": "aa", "test": "bb"} and pooled.floor.tolist() == [0, 0, 1, 1, 0, 1]
    unlabelled = SampleTable(np.zeros((1, 2)), np.zeros((1, 2)), groups={"device": [4]})
    assert pool_splits({"a": train, "b": unlabelled}).floor is None  # labels only if every split has them
    with pytest.raises(ValueError, match="feature shapes"):
        pool_splits({"a": train, "b": SampleTable(np.zeros((1, 3)), np.zeros((1, 2)))})


def test_pool_splits_keeps_a_sample_listed_in_two_splits_once():
    # an "all" split that repeats "train" and "test" (same ids, same rows) plus one row of its own
    X = np.array([[-50.0, np.nan], [-60.0, -70.0], [-55.0, -65.0], [np.nan, -80.0], [-40.0, -45.0]])
    pos = np.arange(10.0).reshape(5, 2)
    ids = np.array(["p0", "p1", "p2", "p3", "p4"])
    whole = SampleTable(X, pos, floor=[0, 0, 1, 1, 2], groups={"device": [1, 1, 2, 2, 3]}, ids=ids)
    pooled = pool_splits({"train": whole[[0, 1]], "test": whole[[2, 3]], "all": whole})
    assert pooled.ids.tolist() == ids.tolist() and pooled.meta["pooled_duplicates"] == {"all": 4}
    assert pooled.groups["split"].tolist() == ["train", "train", "test", "test", "all"]  # the first listing wins
    assert np.array_equal(pooled.X, X, equal_nan=True) and pooled.floor.tolist() == [0, 0, 1, 1, 2]
    # a random split of the pool can no longer test on a scan that is also in the training set
    train, test = random_split(pooled, 0.4, random_state=0)
    assert not set(pooled.ids[train]) & set(pooled.ids[test])
    (fold,) = get_protocol("official").folds(pooled)
    assert fold.train.tolist() == [0, 1] and fold.test.tolist() == [2, 3]
    # the same id with a different row is a contradiction, unless every repeated id differs (row numbers)
    clash = SampleTable(X[[2, 3]], pos[[0, 3]], ids=["p0", "p3"])
    with pytest.raises(ValueError, match="share 2 ids, 1 of them with identical rows"):
        pool_splits({"train": whole[[0, 1, 2, 3]], "test": clash})
    with pytest.raises(ValueError, match="within itself"):
        pool_splits({"train": SampleTable(X[:2], pos[:2], ids=["a", "a"]), "test": whole[[4]]})
    renamed = SampleTable(X[:2], pos[:2], meta={"feature_names": ("ap2", "ap1")})
    with pytest.raises(ValueError, match="feature_names"):
        pool_splits({"train": SampleTable(X[:2], pos[:2], meta={"feature_names": ("ap1", "ap2")}), "test": renamed})


def test_split_summary_hashes_little_endian_int64_indices():
    summary = split_summary([0, 2], np.array([1, 3], dtype=np.int32), n=5)
    assert summary["test_sha256"] == hashlib.sha256(np.array([1, 3], dtype="<i8").tobytes()).hexdigest()
    assert (summary["n_train"], summary["n_test"], summary["n_unused"]) == (2, 2, 1)
    with pytest.raises(ValueError, match="share 1 row"):
        split_summary([0, 1], [1, 2])


# --------------------------------------------------------------------------- named protocols
def test_named_protocols_turn_a_pooled_table_into_labelled_folds():
    train, test = _tables()
    pooled = pool_splits({"train": train, "test": test})
    assert {"official", "random-80-20", "kfold-5", "cross-device", "cross-time",
            "leave-one-building-out"} <= set(list_protocols())
    (fold,) = get_protocol("official").folds(pooled)
    assert fold.name == "official" and fold.train.tolist() == [0, 1, 2, 3] and fold.test.tolist() == [4, 5]
    folds = get_protocol("cross-device").folds(pooled)
    assert [f.name for f in folds] == ["device=1", "device=2", "device=3"]
    assert sum(len(f.test) for f in folds) == len(pooled)
    with pytest.raises(ValueError, match="needs 'time' labels"):
        get_protocol("cross-time").folds(pooled)
    with pytest.raises(ValueError, match="at least 2 groups"):  # one building only
        get_protocol("leave-one-building-out").folds(pooled)
    with pytest.raises(ValueError, match="own 'train' and 'test'"):
        get_protocol("official").folds(pool_splits({"all": train}))
    with pytest.raises(KeyError, match="unknown protocol"):
        get_protocol("no-such-protocol")


def test_random_protocols_are_seeded():
    table = SampleTable(np.zeros((50, 1)), np.zeros((50, 2)))
    first = PROTOCOLS.get("random-80-20").folds(table, random_state=7)[0]
    assert np.array_equal(first.test, get_protocol("random-80-20").folds(table, random_state=7)[0].test)
    assert len(first.test) == 10


def test_register_protocol():
    def two_devices(table, seed):
        from indoorloc.evaluation.protocols import Fold

        return [Fold("1->2", *cross_device_split(table.groups["device"], [2], train_devices=[1]))]

    register_protocol("test-device-1-to-2", Protocol("test-device-1-to-2", "train on 1, test on 2", two_devices),
                      force=True)
    train, _ = _tables()
    (fold,) = get_protocol("test-device-1-to-2").folds(train)
    assert fold.train.tolist() == [0, 1] and fold.test.tolist() == [2, 3]
    with pytest.raises(TypeError):
        register_protocol("bad", lambda t, s: [])
