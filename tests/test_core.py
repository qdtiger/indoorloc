from __future__ import annotations

import collections

import numpy as np
import pytest

from indoorloc.core import Prediction, Registry, SampleTable, requires


def test_table_coerces_and_validates_columns():
    t = SampleTable(np.zeros((3, 2), np.float32), [[1, 2], [3, 4], [5, 6]], floor=[0, 1, 2])
    assert t.pos.dtype == np.float64 and t.floor.dtype == np.int64 and t.ids.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="pos must be"):
        SampleTable(np.zeros((3, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="integer labels"):
        SampleTable(np.zeros((2, 2)), np.zeros((2, 2)), floor=[0.0, np.nan])
    with pytest.raises(ValueError, match="samples first"):
        SampleTable(np.zeros(3), np.zeros((3, 2)))


def test_row_subset_and_concat_keep_every_column_aligned():
    t = SampleTable(np.arange(6.0).reshape(3, 2), np.zeros((3, 2)), floor=[0, 1, 2],
                    groups={"source": ["sim", "sim", "real"]}, meta={"crs": "EPSG:3857"})
    sub = t[t.floor > 0]
    assert sub.ids.tolist() == [1, 2] and sub.groups["source"].tolist() == ["sim", "real"]
    assert sub.X[0].tolist() == [2.0, 3.0] and sub.meta == {"crs": "EPSG:3857"}
    with pytest.raises(ValueError, match="duplicate ids"):
        SampleTable.concat([t, sub])  # ids 1 and 2 twice: rows could no longer be told apart
    both = SampleTable.concat([t, sub.replace(ids=np.array([11, 12]))])
    assert len(both) == 5 and both.floor.tolist() == [0, 1, 2, 1, 2] and both.groups["source"][-1] == "real"
    assert both.ids.tolist() == [0, 1, 2, 11, 12]


def test_containers_are_read_only_views_of_the_callers_arrays():
    X = np.zeros((2, 3), np.float32)
    t = SampleTable(X, np.zeros((2, 2)), floor=[0, 1], groups={"user": [5, 6]})
    p = Prediction(np.zeros((2, 2)), floor=[0, 1], spread=[1.0, 2.0])
    for arr in (t.X, t.pos, t.floor, t.groups["user"], t.ids, t.to_numpy()[0], p.pos, p.floor, p.spread):
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1
    X[0, 0] = -50.0  # the caller's own array stays writable; the table is a view, not a copy
    assert t.X[0, 0] == -50.0 and np.shares_memory(t.X, X)


def test_prediction_rows_and_spread():
    p = Prediction([[0, 0], [1, 1]], floor=[3, -1], spread=[0.5, 2.0])
    assert p[1].floor.tolist() == [-1] and p[1].spread.tolist() == [2.0] and len(p[:1]) == 1


def test_registry_is_lazy_resolves_aliases_and_refuses_silent_overwrite():
    reg = Registry("thing", {"a": "collections:OrderedDict", "b": "a"})
    assert reg.names() == ["a"] and reg.names(aliases=True) == ["a", "b"]
    assert reg.get("B") is collections.OrderedDict
    assert reg.get("json:dumps").__name__ == "dumps"  # a full path needs no registration
    with pytest.raises(KeyError, match="available: a"):
        reg.get("missing")
    with pytest.raises(KeyError, match="already registered"):
        reg.register("a", dict)
    reg.register("a", dict, force=True)
    assert reg.get("a") is dict


def test_registries_record_the_modules_they_name():
    Registry("thing", {"x": "some_plugin.models:Thing", "alias": "x"})
    assert "some_plugin.models" in Registry.trusted_modules() and "x" not in Registry.trusted_modules()


def test_requires_names_the_extra():
    with pytest.raises(ImportError, match=r"indoorloc\[deep\]"):
        requires("surely_not_installed_pkg", "deep")
