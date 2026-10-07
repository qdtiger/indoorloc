"""Usability of the evaluation API: clear errors for common mistakes, compact reprs, create_model inputs."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import Prediction, Registry, SampleTable
from indoorloc.evaluation import (evaal_etri_score, evaluate, get_protocol, ipin_score, list_protocols,
                                  penalized_errors, pool_splits)
from indoorloc.evaluation.report import benchmark_report, results_table
from indoorloc.methods import LocalizerPipeline, create_model, list_models
from indoorloc.methods.neighbors import KNNLocalizer
from indoorloc.signals import Compose, FillMissing, RSSINormalize

ORDER = "truth first"


def _table(n=6, floor=True, seed=0) -> SampleTable:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 10, (n, 2))
    X = -40.0 - 3.0 * np.hypot(pos[:, :1] - np.array([[0.0, 10.0, 5.0]]), pos[:, 1:] - np.array([[0.0, 0.0, 10.0]]))
    X[0, 1] = np.nan  # a missing reading
    groups = {"device": np.array(["a", "b"] * (n // 2))}
    return SampleTable(X.astype(np.float32), pos, np.zeros(n, int) if floor else None, None, groups,
                       meta={"name": "tiny", "modality": "wifi_rssi"})


# --------------------------------------------------------------------------- 1. evaluate argument order
def test_evaluate_refuses_swapped_arguments():
    table = _table()
    pred = Prediction(table.pos + 1.0, floor=table.floor)
    assert evaluate(table, pred).mean_error == pytest.approx(np.sqrt(2))  # the right order still works
    with pytest.raises(TypeError, match="y_true is a Prediction.*truth first .*PDRFusion.run returns"):
        evaluate(pred, table)
    with pytest.raises(TypeError, match="y_true is a Prediction"):
        evaluate(pred, table.pos)
    with pytest.raises(TypeError, match="y_pred is a SampleTable"):
        evaluate(table.pos, table)


def test_evaluate_refuses_the_tuple_of_pdrfusion_run_and_a_step_track():
    from indoorloc.apps.pdr import StepTrack

    table = _table()
    pred = Prediction(table.pos)
    t = np.arange(len(table), dtype=float)
    with pytest.raises(TypeError, match=r"tuple holding a Prediction.*pass the Prediction"):
        evaluate(table, (t, pred))
    with pytest.raises(TypeError, match="tuple of arrays of different shapes"):
        evaluate(table, (t, pred.pos))  # e.g. (t, pos) unpacked by hand
    with pytest.raises(TypeError, match="tuple of arrays of different shapes"):
        evaluate(table.to_numpy(), pred)  # (X, pos)
    track = StepTrack(t=t, index=np.arange(len(t)), length=np.ones(len(t)), heading=np.zeros(len(t)),
                      pos=np.asarray(table.pos), start=np.zeros(2), t0=-1.0)
    with pytest.raises(TypeError, match=r"y_pred is a StepTrack.*track\.pos.*truth first"):
        evaluate(table, track)

    class StepTrack:  # noqa: F811 (a look-alike: L4 detects it by name, never by importing apps)
        pass

    with pytest.raises(TypeError, match="StepTrack"):
        evaluate(table, StepTrack())


def test_evaluate_still_takes_plain_tuples_and_lists_of_positions():
    res = evaluate(((0.0, 0.0), (3.0, 4.0)), [(0.0, 1.0), (0.0, 0.0)])
    assert res.mean_error == pytest.approx(3.0) and res.n == 2
    assert evaluate((0.0, 2.0), (1.0, 2.0)).mean_error == pytest.approx(0.5)  # 1-D positions as a tuple


def test_scores_refuse_swapped_arguments_with_their_own_name():
    table = _table()
    pred = Prediction(table.pos, floor=table.floor)
    for fn in (ipin_score, evaal_etri_score, penalized_errors):
        with pytest.raises(TypeError, match=rf"{fn.__name__}\(y_true, y_pred\): truth first"):
            fn(pred, table)


# --------------------------------------------------------------------------- 2. scores and unplaced samples
@pytest.mark.parametrize("score", [ipin_score, evaal_etri_score])
def test_ipin_and_evaal_etri_scores_refuse_unplaced_samples_alike(score):
    table = _table()
    pos = np.array(table.pos)
    pos[2] = np.nan
    pred = Prediction(pos, floor=table.floor)
    with pytest.raises(ValueError, match=rf"{score.__name__}: 1 of 6 samples were not placed .*n_failed=1.*"
                                         r"np\.isfinite\(pred\.pos\)\.all\(axis=1\)"):
        score(table, pred)
    placed = np.isfinite(pred.pos).all(axis=1)  # the remedy the message gives
    assert score(table[placed], pred[placed]) == pytest.approx(0.0)
    assert evaluate(table, pred).n_failed == 1


def test_penalized_errors_keeps_nan_for_unplaced_but_refuses_nan_truth():
    err = penalized_errors([[0, 0], [1, 1]], [[0, 0], [np.nan, np.nan]], floor_penalty=0, building_penalty=0)
    assert err[0] == 0.0 and np.isnan(err[1])  # unchanged: the caller decides what to do with them
    with pytest.raises(ValueError, match="y_true has NaN or infinite positions in 1 of 2 rows"):
        ipin_score([[0, 0], [np.nan, 1]], [[0, 0], [1, 1]], floor_penalty=0, building_penalty=0)
    # only the horizontal axes are scored: an unknown height does not matter
    assert ipin_score([[0, 0, np.nan]], [[0, 0, 1.0]], floor_penalty=0, building_penalty=0) == 0.0


# --------------------------------------------------------------------------- 3. reports show n_failed
def test_results_table_adds_n_failed_and_a_note_only_when_needed():
    ok = evaluate([[0, 0], [3, 4]], [[0, 1], [0, 0]])
    partial = evaluate([[0, 0], [3, 4], [1, 1]], [[0, 1], [0, 0], [np.nan, np.nan]])
    clean = results_table({"a": ok}, metrics=("mean_error",))
    assert "n_failed" not in clean and "not placed" not in clean
    md = results_table({"a": ok, "b": partial}, metrics=("mean_error",))
    lines = md.splitlines()
    assert lines[:4] == ["| Method | mean_error [m] | n_failed |", "|---|---|---|", "| a | 3.000 | 0 |",
                         "| b | 3.000 | 1 |"]
    assert lines[4] == "" and lines[5].startswith("Not placed (b): 1 of 3 samples were not placed")
    text = results_table({"b": {"n": 3, "n_failed": 1, "mean_error": 3.0, "failed_note": "custom note"}},
                         metrics=("mean_error",), style="text")
    assert text.splitlines()[0].split() == ["Method", "mean_error", "[m]", "n_failed"]
    assert text.endswith("Not placed (b): custom note")


def test_benchmark_report_shows_n_failed_per_fold_and_the_failed_note():
    ok = {"n": 2, "n_failed": 0, "mean_error": 1.0}
    bad = {"n": 2, "n_failed": 1, "mean_error": 1.0}
    pooled = {"n": 4, "n_failed": 1, "mean_error": 1.0,
              "failed_note": "1 of 4 samples were not placed (NaN estimate): error statistics cover the placed ones"}
    payload = {"dataset": {"name": "tiny"}, "units": "m",
               "protocol": {"name": "cross-device", "folds": [
                   {"name": "device=1", "n_train": 2, "n_test": 2, "test_sha256": "c" * 64},
                   {"name": "device=2", "n_train": 2, "n_test": 2, "test_sha256": "d" * 64}]},
               "methods": [{"label": "knn", "pooled": pooled, "folds": [
                   {"fold": "device=1", "fit_s": 0.1, "predict_s": 0.1, "metrics": ok},
                   {"fold": "device=2", "fit_s": 0.1, "predict_s": 0.1, "metrics": bad}]}]}
    md = benchmark_report(payload)
    assert "| Method | mean_error [m] | n_failed |" in md and "| knn | 1.000 | 1 |" in md
    assert "| Method | Fold | mean_error [m] | n_failed |" in md and "| knn | device=2 | 1.000 | 1 |" in md
    assert "Not placed (knn): 1 of 4 samples were not placed (NaN estimate)" in md
    payload["methods"][0]["pooled"] = {**ok, "n": 4}
    payload["methods"][0]["folds"][1]["metrics"] = ok
    assert "n_failed" not in benchmark_report(payload)


# --------------------------------------------------------------------------- 4. protocols split ONE table
def test_protocols_refuse_the_train_test_tuple_and_name_pool_splits():
    train, test = _table(seed=0), _table(seed=1)
    cross_device = get_protocol("cross-device")
    with pytest.raises(TypeError, match=r"splits one SampleTable, got a tuple of 2 items .*pool_splits"):
        cross_device.folds((train, test))
    for name in ("random-80-20", "kfold-5", "cross-device", "official"):
        with pytest.raises(TypeError, match="pool_splits"):
            get_protocol(name).split((train, test), 0)  # len((train, test)) == 2 used to split two "rows"
    with pytest.raises(TypeError, match="got a dict of 2 items"):
        cross_device.folds({"train": train, "test": test})
    pooled = pool_splits({"train": train, "test": test.replace(ids=np.arange(6, 12))})
    folds = cross_device.folds(pooled)
    assert [f.name for f in folds] == ["device=a", "device=b"] and sum(len(f.test) for f in folds) == 12


def test_list_functions_have_one_line_docstrings():
    for fn in (list_protocols, list_models):
        assert fn.__doc__ and "\n" not in fn.__doc__.strip()
    assert "cross-device" in list_protocols() and "knn" in list_models()


# --------------------------------------------------------------------------- 5. compact reprs
def test_prediction_repr_is_compact():
    n = 1000
    pos = np.zeros((n, 2))
    assert repr(Prediction(pos)) == "Prediction(n=1000, pos=(1000, 2))"
    pos[:3] = np.nan
    full = Prediction(pos, floor=np.zeros(n, int), building=np.ones(n, int), spread=np.ones(n), ids=np.arange(n))
    assert repr(full) == "Prediction(n=1000, pos=(1000, 2), floor/building/spread/ids, 3 not placed)"
    assert repr(Prediction(pos[:2, :1], spread=np.ones(2))) == "Prediction(n=2, pos=(2, 1), spread, 2 not placed)"
    assert repr(Prediction(np.zeros((0, 2)))) == "Prediction(n=0, pos=(0, 2))"  # empty: no reshape error


def test_evaluation_results_repr_is_its_summary():
    res = evaluate(np.zeros((500, 2)), np.ones((500, 2)))
    assert repr(res) == res.summary() == str(res) and len(repr(res)) < 120
    assert repr(_table()).startswith("SampleTable(n=6, X=(6, 3) float32")  # unchanged


# --------------------------------------------------------------------------- 6. create_model inputs
def test_create_model_takes_a_class():
    model = create_model(KNNLocalizer, k=1)
    assert type(model) is KNNLocalizer and model.k == 1
    table = _table()
    piped = create_model(KNNLocalizer, preprocess=FillMissing(-104), k=1).fit(table)
    assert isinstance(piped, LocalizerPipeline) and piped.evaluate(table).mean_error == pytest.approx(0.0)
    with pytest.raises(TypeError, match="method is named by a string .* got int"):
        create_model(3)


def test_create_model_wraps_a_list_of_transforms_in_compose():
    table = _table()
    model = create_model("knn", preprocess=[FillMissing(-104), RSSINormalize()], k=1)
    assert isinstance(model, LocalizerPipeline) and isinstance(model.preprocess, Compose)
    assert [type(t) for t in model.preprocess.transforms] == [FillMissing, RSSINormalize]
    assert model.fit(table).evaluate(table).mean_error == pytest.approx(0.0)
    assert type(create_model("knn", preprocess=())) is KNNLocalizer  # nothing to apply: the bare model
    direct = LocalizerPipeline((FillMissing(-104), RSSINormalize()), KNNLocalizer(k=1)).fit(table)
    assert isinstance(direct.preprocess_, Compose) and isinstance(direct.preprocess, tuple)  # params untouched
    np.testing.assert_allclose(direct.predict(table.X), model.predict(table.X))


def test_registry_names_are_strings():
    reg = Registry("thing", {"a": "json:dumps"})
    assert 3 not in reg and "A" in reg and reg.get(dict) is dict
    with pytest.raises(TypeError, match="thing name must be a string"):
        reg.register(dict, "json:loads")
    with pytest.raises(TypeError, match="thing is named by a string"):
        reg.get(3.5)
