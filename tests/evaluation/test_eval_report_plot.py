from __future__ import annotations

import sys

import numpy as np
import pytest

from indoorloc.evaluation import evaluate
from indoorloc.evaluation.bounds import gdop, toa_crlb
from indoorloc.evaluation.report import benchmark_report, literature_table, results_table


def test_results_table_markdown_and_text():
    r = evaluate([[0, 0], [3, 4]], [[0, 1], [0, 0]], floor_true=[0, 1], floor_pred=[0, 0])
    md = results_table({"knn": r, "other": {"mean_error": 1.25}}, metrics=("mean_error", "floor_accuracy",
                                                                           "building_accuracy"))
    assert md.splitlines() == ["| Method | mean_error [m] | floor_accuracy [%] |", "|---|---|---|",
                               "| knn | 3.000 | 50.000 |", "| other | 1.250 | - |"]  # building: no entry has it
    text = results_table({"knn": r}, metrics=("mean_error",), style="text", digits=1).splitlines()
    assert text[0].split() == ["Method", "mean_error", "[m]"] and text[2].split() == ["knn", "3.0"]
    with pytest.raises(ValueError, match="style"):
        results_table({"knn": r}, style="html")


def _payload():
    metrics = {"n": 2, "mean_error": 1.0, "median_error": 1.0, "floor_accuracy": 50.0, "mean_error_ci95": [0.5, 1.5]}
    return {"format": "indoorloc-benchmark", "dataset": {"name": "tiny", "n_samples": {"train": 4, "test": 2},
                                                         "sha256": {"train": "a" * 64, "test": "b" * 64}},
            "protocol": {"name": "cross-device", "folds": [
                {"name": "device=1", "n_train": 4, "n_test": 1, "test_sha256": "c" * 64},
                {"name": "device=2", "n_train": 4, "n_test": 1, "test_sha256": "d" * 64}]},
            "methods": [{"label": "knn", "pooled": metrics,
                         "folds": [{"fold": "device=1", "fit_s": 0.1, "predict_s": 0.2, "metrics": metrics},
                                   {"fold": "device=2", "fit_s": 0.1, "predict_s": 0.2, "metrics": metrics}]}],
            "environment": {"indoorloc": "0.2", "python": "3.x", "numpy": "2.x",
                            "git": {"commit": "0123456789abcdef", "dirty": True}},
            "seed": 0, "units": "m", "wall_time_s": 1.5, "command": ["indoorloc", "benchmark"],
            "literature_note": "published numbers exist for other protocols"}


def test_benchmark_report_shows_folds_splits_and_provenance():
    md = benchmark_report(_payload())
    for part in ("# Benchmark: tiny / protocol cross-device", "## Per fold", "| knn | device=1 |", "## Splits",
                 "cccccccccccc", "0123456789ab (uncommitted changes)", "train: aaaaaaaaaaaa...",
                 "95 % bootstrap CI [0.500, 1.500]", "Literature: published numbers exist for other protocols"):
        assert part in md, part
    assert "Provenance\n==========" in benchmark_report(_payload(), style="text")


def test_literature_table_from_a_stored_comparison():
    stored = {"protocol": "official", "metrics": ["mean_error"], "literature": [
        {"method": "A", "values": {"mean_error": 2.0}, "same_protocol": None, "check": "unchecked",
         "source": {"authors": "X. Cao, Y. Zhuang, X. Yang", "year": 2021, "venue": "Satellite Navigation 2:27",
                    "doi": "10.1186/s43020-021-00058-8"}}]}
    md = literature_table(stored)
    assert "| A | 2.00 | unknown | unchecked | Cao et al. (2021), Satellite Navigation 2:27. " \
           "doi:10.1186/s43020-021-00058-8 |" in md


# --------------------------------------------------------------------------- plots
@pytest.fixture
def plt():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def test_plot_cdf_draws_one_step_curve_per_result(plt):
    from indoorloc.evaluation.plot import plot_cdf

    r = evaluate([[0, 0]] * 4, [[1, 0], [2, 0], [3, 0], [4, 0]])
    ax = plot_cdf({"a": r, "b": np.array([0.5, 1.5])})
    lines = ax.get_lines()
    assert [ln.get_label() for ln in lines] == ["a", "b"]
    x, y = lines[0].get_data()
    assert x.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0] and y.tolist() == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert ax.get_ylim() == (0.0, 100.0) and ax.get_xlim() == (0.0, 4.0)


def test_plot_error_map_and_bound_map(plt):
    from indoorloc.evaluation.plot import plot_bound_map, plot_error_map

    true = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ax = plot_error_map(true, true + [[0.0, 3.0], [0.0, 0.0], [4.0, 0.0]], arrows=True)
    assert sorted(ax.collections[-1].get_array().tolist()) == [0.0, 3.0, 4.0]  # coloured by error
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    ax = plot_bound_map(anchors, lambda p: toa_crlb(anchors, p, 0.3), resolution=20)
    assert ax.get_xlabel() == "x [m]" and len(ax.get_lines()) == 1  # the anchor markers
    plot_bound_map(anchors, lambda p: gdop(anchors, p), resolution=10, extent=(1, 9, 1, 9), colorbar=False)
    ax = plot_bound_map(anchors, lambda p: gdop(anchors, p), resolution=10, levels=np.array([0.5, 1.0, 2.0, 4.0]))
    assert ax.collections[0].levels.tolist() == [0.5, 1.0, 2.0, 4.0]  # explicit contour levels (an array)
    with pytest.raises(ValueError, match="pass pos_pred or errors"):
        plot_error_map(true)


def test_missing_matplotlib_names_the_extra(monkeypatch):
    from indoorloc.evaluation import plot

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"indoorloc\[plot\]"):
        plot.plot_cdf([1.0, 2.0])
