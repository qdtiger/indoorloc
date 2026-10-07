from __future__ import annotations

import pytest

from indoorloc.core import Prediction, SampleTable
from indoorloc.evaluation import evaluate


def test_evaluate_plain_arrays():
    r = evaluate([[0, 0], [3, 4], [1, 1]], [[0, 1], [0, 0], [1, 1]],
                 floor_true=[1, 2, 3], floor_pred=[1, 2, 0])
    assert r.errors.tolist() == [1.0, 5.0, 0.0] and (r.mean_error, r.median_error) == (2.0, 1.0)
    assert r.floor_accuracy == pytest.approx(200 / 3) and r.building_accuracy is None
    xs, ps = r.cdf()
    assert xs.tolist() == [0.0, 1.0, 5.0] and ps.tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert r.cdf([0.5, 1.0, 10.0]).tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_every_axis_counts_and_negative_floors_are_real_floors():
    assert evaluate([0.0, 1.0], [1.0, 3.0]).mean_error == 1.5  # 1-D corridor
    assert evaluate([[0, 0, 0], [0, 0, 0]], [[0, 0, 3], [0, 0, 4]]).errors.tolist() == [3.0, 4.0]
    assert evaluate([[0, 0]] * 4, [[0, 0]] * 4, floor_true=[-1, -1, 0, 1], floor_pred=[0, 0, 0, 1]).floor_accuracy == 50.0
    assert evaluate([[0, 0]], [[3, 4]], scale=0.5).mean_error == 2.5


def test_labels_come_from_core_types_and_explicit_arguments_win():
    table = SampleTable([[0.0], [0.0]], [[0, 0], [0, 2]], floor=[0, 1])
    r = evaluate(table, Prediction([[0, 0], [0, 0]], floor=[0, 0]))
    assert r.mean_error == 1.0 and r.floor_accuracy == 50.0
    assert evaluate(table, Prediction([[0, 0], [0, 0]], floor=[0, 0]), floor_true=[0, 0]).floor_accuracy == 100.0


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="differ in shape"):
        evaluate([[0, 0]], [[0, 0, 0]])


def test_rows_are_matched_by_id_when_both_sides_carry_ids():
    table = SampleTable([[0.0], [0.0]], [[0, 0], [0, 2]], ids=["a", "b"])
    assert evaluate(table, Prediction([[0, 0], [0, 2]], ids=["a", "b"])).mean_error == 0.0
    with pytest.raises(ValueError, match="different ids"):
        evaluate(table, Prediction([[0, 2], [0, 0]], ids=["b", "a"]))  # reordered: would score 2.0
