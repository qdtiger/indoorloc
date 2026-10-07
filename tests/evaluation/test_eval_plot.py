"""evaluation.plot: CDFs of several methods (with unplaced samples), error maps, trajectories.

Figures are checked through the artists they return (Agg backend), never through pixels.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from indoorloc.core import Prediction, SampleTable
from indoorloc.evaluation import evaluate


@pytest.fixture
def plt():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


PLAN = {"walls": np.array([[0.0, 0.0, 10.0, 0.0], [10.0, 0.0, 10.0, 5.0], [0.0, 5.0, 10.0, 5.0],
                           [0.0, 0.0, 0.0, 5.0], [5.0, 0.0, 5.0, 3.0]]),
        "wall_floor": np.array([0, 0, 0, 0, 1]), "floor_height": 3.0}


# --------------------------------------------------------------------------- CDF
def test_cdf_of_several_methods_counts_unplaced_samples_as_never_reached(plt):
    from indoorloc.evaluation.plot import plot_cdf

    truth = np.zeros((4, 2))
    good = evaluate(truth, [[1, 0], [2, 0], [3, 0], [4, 0]])
    partial = evaluate(truth, [[0, 1], [0, 2], [0, 3], [np.nan, np.nan]])  # one sample not placed
    assert partial.n_failed == 1
    ax = plot_cdf({"good": good, "partial": partial})
    (xg, yg), (xp, yp) = (line.get_data() for line in ax.get_lines())
    assert xg.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0] and yg.tolist() == [0.0, 25.0, 50.0, 75.0, 100.0]
    # 3 of 4 placed: the curve stops at 75 % and runs flat to the right edge (4 m, the other curve's max)
    assert xp.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0] and yp.tolist() == [0.0, 25.0, 50.0, 75.0, 75.0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["good", "partial (1 of 4 not placed)"]
    assert ax.get_xlim() == (0.0, 4.0) and ax.get_ylim() == (0.0, 100.0)


def test_cdf_fraction_axis_and_a_method_that_placed_nothing(plt):
    from indoorloc.evaluation.plot import plot_cdf

    ax = plot_cdf({"a": np.array([0.5, 1.0]), "none": np.array([np.nan, np.nan])}, percent=False, xmax=2.0)
    (xa, ya), (xn, yn) = (line.get_data() for line in ax.get_lines())
    assert ya.tolist() == [0.0, 0.5, 1.0] and xn.tolist() == [0.0, 2.0] and yn.tolist() == [0.0, 0.0]
    assert ax.get_ylim() == (0.0, 1.0) and ax.get_ylabel() == "CDF"
    ax = plot_cdf(np.array([1.0, np.nan]))  # unnamed curve: the legend still reports the failure
    assert ax.get_legend().get_texts()[0].get_text() == "1 of 2 not placed"


def test_cdf_x_limit_ignores_a_long_tail(plt):
    from indoorloc.evaluation.plot import plot_cdf

    errors = np.r_[np.linspace(0.0, 2.0, 999), 500.0]  # one gross outlier
    ax = plot_cdf(errors)
    assert ax.get_xlim()[1] == pytest.approx(np.percentile(errors, 99))


# --------------------------------------------------------------------------- error map
def test_error_map_colours_placed_samples_and_marks_unplaced_ones(plt):
    from indoorloc.evaluation.plot import plot_error_map

    true = SampleTable(np.zeros((4, 1)), [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    pred = Prediction([[1.0, 4.0], [2.0, 1.0], [np.nan, np.nan], [4.0, 2.0]])
    ax = plot_error_map(true, pred, arrows=True, floor_plan=PLAN, floor=0, vmax=5.0, unit="m (ground)", pos_unit="m")
    walls, arrows, points, failed = ax.collections
    assert len(walls.get_segments()) == 4  # the floor-1 wall is not drawn
    assert points.get_array().tolist() == [0.0, 1.0, 3.0]  # placed samples, largest error on top
    assert points.get_clim() == (0.0, 5.0) and failed.get_offsets().tolist() == [[3.0, 1.0]]
    assert "not placed (1)" in [t.get_text() for t in ax.get_legend().get_texts()]
    assert ax.get_xlabel() == "x [m]" and ax.figure.axes[-1].get_ylabel() == "error [m (ground)]"


# --------------------------------------------------------------------------- trajectories
def _walks():
    t = np.arange(10.0)
    pos = np.column_stack([t, np.zeros(10)])
    pos[5:, 1] = 2.0  # the second walk runs along y = 2
    table = SampleTable(np.zeros((10, 1)), pos, floor=np.zeros(10, np.int64),
                        groups={"trajectory": np.repeat([0, 1], 5), "time": np.r_[t[:5], t[:5]]})
    est = pos + 0.1
    est[0] = np.nan  # no fix yet
    return table, est


def test_trajectories_draw_truth_estimates_and_walls_and_break_between_walks(plt):
    from indoorloc.evaluation.plot import plot_trajectories

    table, est = _walks()
    ax = plot_trajectories(table, {"fixes": est, "smoothed": Prediction(table.pos + 0.05)}, floor_plan=PLAN, floor=0)
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert list(lines) == ["ground truth", "start", "end", "fixes", "smoothed"]
    x, y = lines["ground truth"].get_data()
    assert len(x) == 11 and np.isnan(x[5]) and np.isnan(y[5])  # a NaN row between the two walks
    assert lines["start"].get_data()[0].tolist() == [0.0, 5.0] and lines["end"].get_data()[0].tolist() == [4.0, 9.0]
    fx, _ = lines["fixes"].get_data()
    assert np.isnan(fx[0]) and np.isnan(fx[5]) and np.nanmax(fx) == pytest.approx(9.1)
    assert len(ax.collections) == 1 and len(ax.collections[0].get_segments()) == 4  # floor-0 walls
    assert lines["ground truth"].get_color() == "k" and lines["ground truth"].get_zorder() > lines["fixes"].get_zorder()


def test_trajectories_accept_a_floor_map_object_and_one_unnamed_track(plt):
    from indoorloc.apps import FloorMap
    from indoorloc.evaluation.plot import plot_trajectories

    fmap = FloorMap(PLAN["walls"], floor=PLAN["wall_floor"])
    ax = plot_trajectories(estimates=np.array([[1.0, 1.0], [2.0, 2.0]]), floor_plan=fmap)
    assert [line.get_label() for line in ax.get_lines()] == ["estimate"]
    assert len(ax.collections[0].get_segments()) == 5  # floor=None: every wall
    with pytest.raises(ValueError, match="one row per time step"):
        plot_trajectories(np.zeros((3, 2)), {"short": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="truth and/or estimates"):
        plot_trajectories()
    with pytest.raises(ValueError, match="trajectory has 2 ids"):
        plot_trajectories(np.zeros((3, 2)), trajectory=[0, 1])


def test_missing_matplotlib_names_the_extra_for_every_figure(monkeypatch):
    from indoorloc.evaluation import plot

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    for draw in (lambda: plot.plot_trajectories(np.zeros((2, 2))), lambda: plot.plot_error_map(np.zeros((2, 2)),
                                                                                                errors=[0.0, 1.0])):
        with pytest.raises(ImportError, match=r"indoorloc\[plot\]"):
            draw()


# --------------------------------------------------------------------------- review additions
def test_cdf_curve_is_the_l4_error_cdf_and_sits_right_of_the_placed_median(plt):
    """The drawn step curve equals evaluation.error_cdf at every threshold (failures in the denominator)."""
    from indoorloc.evaluation import error_cdf
    from indoorloc.evaluation.plot import plot_cdf

    errors = np.random.default_rng(0).gamma(2.0, 2.0, 50)
    errors[[3, 7, 9, 20, 31]] = np.nan  # 5 of 50 not placed
    result = evaluate(np.zeros((50, 1)), errors[:, None])  # |0 - e| = e
    x, y = plot_cdf(result).get_lines()[0].get_data()
    t = np.linspace(0.0, 30.0, 3001)
    drawn = y[np.searchsorted(x, t, side="right") - 1]  # value of a where="post" step curve at t
    assert np.array_equal(drawn, 100.0 * error_cdf(errors, t)) and y.max() == 90.0
    # evaluate's median uses the 45 placed samples; the curve crosses 50 % at the 25th of 50 errors
    assert np.sort(errors[np.isfinite(errors)])[24] > result.median_error


def test_misaligned_rows_are_refused_as_in_evaluate(plt):
    from indoorloc.evaluation.plot import plot_error_map, plot_trajectories

    truth = SampleTable(np.zeros((4, 1)), np.arange(8.0).reshape(4, 2), ids=["a", "b", "c", "d"])
    shuffled = Prediction(truth.pos[::-1], ids=np.array(["d", "c", "b", "a"]))
    with pytest.raises(ValueError, match="rows are not aligned"):
        plot_error_map(truth, shuffled)
    with pytest.raises(ValueError, match=r"estimates\['p'\].*not aligned"):
        plot_trajectories(truth, {"p": shuffled})
    aligned = Prediction(truth.pos + 1.0, ids=truth.ids)
    ax = plot_error_map(truth[[1, 3]], aligned[[1, 3]])  # the same subset of both sides is fine
    assert ax.collections[0].get_array().tolist() == [np.sqrt(2.0)] * 2


def test_long_coordinate_labels_get_fewer_ticks(plt):
    """EPSG:3857 eastings (-7650 ...) would overlap at matplotlib's default tick density."""
    from matplotlib.ticker import AutoLocator, MaxNLocator

    from indoorloc.evaluation.plot import plot_error_map, plot_trajectories

    mercator = np.column_stack([np.linspace(-7700.0, -7300.0, 9), np.linspace(4864750.0, 4865000.0, 9)])
    ax = plot_error_map(mercator, errors=np.arange(9.0))
    assert type(ax.xaxis.get_major_locator()) is MaxNLocator
    fig = ax.figure
    fig.canvas.draw()
    assert len([t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]) <= 6
    office = plot_trajectories(np.column_stack([np.linspace(0, 40, 9), np.zeros(9)]))
    assert type(office.xaxis.get_major_locator()) is AutoLocator  # short labels: matplotlib's default
