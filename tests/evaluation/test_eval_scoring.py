from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import Prediction, SampleTable
from indoorloc.evaluation import (bootstrap_ci, cep, evaal_etri_score, evaluate, ipin_score, penalized_errors,
                                  percentile_error, success_rate)


def test_potorti_2017_worked_example_scores_34_m():
    # Potortì et al., Sensors 2017, Section 3.2: "if the x y error is 4 m and the estimated floor is 2
    # while it should be 0, the computed error for that estimate will be 4 + 2P = 34 m" (P = 15 m).
    err = penalized_errors([[0.0, 0.0]], [[0.0, 4.0]], floor_true=[0], floor_pred=[2])
    assert err.tolist() == [34.0]
    assert ipin_score([[0.0, 0.0]], [[0.0, 4.0]], floor_true=[0], floor_pred=[2]) == 34.0


def test_building_penalty_negative_floors_and_the_75th_percentile():
    y = np.zeros((4, 2))
    pred = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    err = penalized_errors(y, pred, floor_true=[-1, -1, 0, 0], floor_pred=[-1, 0, 0, 0],
                           building_true=[0, 0, 0, 1], building_pred=[0, 0, 0, 0])
    assert err.tolist() == [1.0, 17.0, 3.0, 54.0]  # basement to ground floor is one floor
    assert ipin_score(y, pred) == pytest.approx(3.25)  # no labels: 75th percentile of 1, 2, 3, 4 (linear)
    assert ipin_score(y, pred, method="hazen") == pytest.approx(3.5)  # MATLAB prctile convention


def test_evaal_etri_rules_are_4_m_per_floor_and_50_m_per_building():
    # Torres-Sospedra et al. 2017, Section 3.1: mean of (2-D error + 4 m per floor + 50 m wrong building)
    score = evaal_etri_score([[0, 0], [0, 0]], [[3, 4], [0, 1]], floor_true=[1, 2], floor_pred=[3, 2],
                             building_true=[0, 1], building_pred=[0, 0])
    assert score == pytest.approx(((5 + 8) + (1 + 50)) / 2)


def test_horizontal_axes_and_scale_affect_only_the_distance():
    err = penalized_errors([[0, 0, 0]], [[3, 4, 12]], floor_true=[0], floor_pred=[1], scale=0.5)
    assert err.tolist() == [2.5 + 15.0]  # height ignored, 5 m * 0.5, penalty unscaled
    assert penalized_errors([[0, 0, 0]], [[3, 4, 12]], horizontal_axes=3).tolist() == [13.0]


def test_tables_and_predictions_supply_labels_and_one_sided_labels_are_refused():
    table = SampleTable(np.zeros((2, 1)), [[0, 0], [0, 0]], floor=[1, 1], building=[0, 0], ids=["a", "b"])
    pred = Prediction([[0, 1], [0, 0]], floor=[1, 0], building=[0, 0], ids=["a", "b"])
    assert penalized_errors(table, pred).tolist() == [1.0, 15.0]
    with pytest.raises(ValueError, match="missing from the predictions"):
        penalized_errors(table, Prediction([[0, 0], [0, 0]]))  # the truth has floors, the method none
    no_labels = Prediction([[0, 0], [0, 0]])
    assert penalized_errors(table, no_labels, floor_penalty=0, building_penalty=0).tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="not aligned"):
        penalized_errors(table, Prediction([[0, 0], [0, 0]], floor=[1, 1], building=[0, 0], ids=["b", "a"]))


def test_statistics_match_the_evaluate_summary():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=(200, 2)), rng.normal(size=(200, 2))
    res = evaluate(y, p)
    assert percentile_error(res.errors, 75) == res.p75_error and cep(res.errors) == res.median_error
    assert percentile_error(res.errors, 90) == res.p90_error
    assert success_rate([1.0, 2.0, 3.0, 4.0], 2.0) == 50.0 and success_rate([1.0], 0.5) == 0.0
    with pytest.raises(ValueError):
        percentile_error([1.0, -1.0], 50)
    with pytest.raises(ValueError):
        cep([], 50)


def test_bootstrap_ci_is_deterministic_batch_independent_and_matches_the_clt():
    e = np.random.default_rng(1).exponential(5.0, size=2000)
    lo, hi = bootstrap_ci(e, "mean", random_state=3)
    assert (lo, hi) == bootstrap_ci(e, "mean", random_state=3, batch_size=37)
    assert lo < e.mean() < hi
    half = 1.959964 * e.std(ddof=1) / np.sqrt(len(e))  # normal approximation of the mean's 95 % interval
    assert (hi - lo) / 2 == pytest.approx(half, rel=0.1)
    assert bootstrap_ci(np.full(10, 2.5), "median") == (2.5, 2.5)
    assert bootstrap_ci(e, lambda x, axis: x.max(axis=axis), n_resamples=50)[1] <= e.max()
    with pytest.raises(ValueError, match="unknown statistic"):
        bootstrap_ci(e, "mode")
