"""datasets.plot: spatial distribution figures of SampleTables (2-D per floor, 3-D stacked, density).

Checked through the returned Figure's axes and artists (Agg backend), not pixels. The plotly
page (``distribution_html``) is exercised only where plotly is installed.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from conftest import DATA_ROOT
from indoorloc.core import SampleTable

UJI = DATA_ROOT / "ujiindoorloc"
ILC_FLOOR = DATA_ROOT / "ilc2020" / "data" / "site2" / "F4"


@pytest.fixture
def plt():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


PLAN = {"walls": np.array([[0.0, 0.0, 10.0, 0.0], [0.0, 6.0, 10.0, 6.0], [4.0, 0.0, 4.0, 6.0]]),
        "wall_floor": np.array([0, 0, 1]), "floor_height": 3.5, "bounds": np.array([0.0, 0.0, 10.0, 6.0])}


def _tables():
    """train: 6 samples on floor 0, 4 on floor 1 (two devices); test: 2 + 1 samples."""
    pos = np.array([[1, 1], [2, 1], [3, 1], [4, 2], [5, 2], [6, 2], [1, 5], [2, 5], [3, 5], [9, 5]], float)
    floor = np.array([0] * 6 + [1] * 4)
    meta = {"name": "toy", "split": "train", "crs": "local", "pos_units": "m", "floor_plan": PLAN}
    train = SampleTable(np.zeros((10, 1)), pos, floor, groups={"device": np.array([7, 8] * 5)}, meta=meta)
    test = SampleTable(np.zeros((3, 1)), [[1.5, 1.5], [2.5, 1.5], [8.0, 4.0]], [0, 0, 1],
                       groups={"device": np.array([7, 7, 8])}, meta={**meta, "split": "test"})
    return train, test


def _points(ax):
    """Number of scatter points on an axes (walls and meshes are other collection types)."""
    return sum(len(c.get_offsets()) for c in ax.collections if type(c).__name__ == "PathCollection")


def test_2d_panels_per_floor_legend_counts_and_walls(plt):
    from indoorloc.datasets.plot import plot_distribution

    fig = plot_distribution(_tables())
    floor0, floor1 = fig.axes
    assert [floor0.get_title(), floor1.get_title()] == ["floor 0 (n=8)", "floor 1 (n=5)"]
    scat = [c for c in floor0.collections if c.get_label() in ("train", "test")]
    assert [c.get_label() for c in scat] == ["train", "test"] and [len(c.get_offsets()) for c in scat] == [6, 2]
    walls = [c for c in floor0.collections if c not in scat]
    assert len(walls) == 1 and len(walls[0].get_segments()) == 2  # only floor-0 walls on the floor-0 panel
    legend = fig.legends[0] if fig.legends else floor0.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["train (n=10)", "test (n=3)"]
    assert floor0.get_xlabel() == "x [m]" and floor0.get_shared_x_axes().joined(floor0, floor1)
    assert fig.get_suptitle() == "toy: 13 samples (crs local)"


def test_3d_view_stacks_floors_at_the_storey_height(plt):
    from indoorloc.datasets.plot import plot_distribution

    train, test = _tables()
    fig = plot_distribution({"fit": train, "eval": test}, view="3d", color_by="floor")
    (ax,) = fig.axes
    assert ax.name == "3d" and ax.get_zlabel() == "height [m]"
    z = {c.get_label(): np.asarray(c._offsets3d[2]) for c in ax.collections if c.get_label().startswith("floor")}
    assert set(z) == {"floor 0", "floor 1"} and np.all(z["floor 1"] == 3.5) and np.all(z["floor 0"] == 0.0)
    fig = plot_distribution(train, view="3d", floor_height=1.0, floor_plan=None)
    assert fig.axes[0].get_zlabel() == "floor" and list(fig.axes[0].get_zticks()) == [0, 1]


def test_density_counts_every_sample_once(plt):
    from indoorloc.datasets.plot import plot_distribution

    train, test = _tables()
    fig = plot_distribution([train, test], kind="density", bin_size=1.0)
    meshes = [c for ax in fig.axes[:2] for c in ax.collections if c.__class__.__name__ == "QuadMesh"]
    assert len(meshes) == 2 and sum(float(m.get_array().sum()) for m in meshes) == 13
    assert fig.axes[-1].get_ylabel() == "samples per 1 x 1 m cell"  # the shared colour bar, with the unit


def test_colour_by_a_group_floors_filter_subsample_and_relative_frame(plt):
    from indoorloc.datasets.plot import plot_distribution

    train, _ = _tables()
    fig = plot_distribution(train, color_by="device", floors=[0], relative=True)
    (ax,) = fig.axes
    labels = [c.get_label() for c in ax.collections if c.get_label().startswith("device")]
    assert labels == ["device=7", "device=8"]
    xy = np.concatenate([c.get_offsets() for c in ax.collections if c.get_label().startswith("device")])
    assert xy.min(axis=0).tolist() == [0.0, 0.0] and ax.get_xlabel() == "x - 1.0 [m]"
    many = SampleTable(np.zeros((500, 1)), np.random.default_rng(0).uniform(0, 10, (500, 2)),
                       groups={"time": np.linspace(0, 1, 500)}, meta={"name": "t"})
    a = plot_distribution(many, max_points=50, color_by="time", random_state=1)
    b = plot_distribution(many, max_points=50, color_by="time", random_state=1)
    pa, pb = (f.axes[0].collections[0].get_offsets() for f in (a, b))
    assert len(pa) == 50 and np.array_equal(pa, pb)  # a seeded subset
    assert a.get_suptitle() == "t: 500 samples, 50 drawn" and a.axes[-1].get_ylabel() == "time"  # colour bar
    sub = plot_distribution(train, max_points=3, random_state=0)
    assert [ax.get_title() for ax in sub.axes] == ["floor 0 (n=6)", "floor 1 (n=4)"]  # full counts
    assert sum(_points(ax) for ax in sub.axes) == 3


def test_per_building_frames_get_their_own_panels(plt):
    from indoorloc.datasets.plot import plot_distribution

    pos = np.array([[0, 0], [1, 1], [900, 900], [901, 901]], float)
    t = SampleTable(np.zeros((4, 1)), pos, floor=[1, 1, 4, 4], building=[1, 1, 2, 2],
                    meta={"name": "sod-like", "crs": "local-per-building"})
    fig = plot_distribution(t)
    assert [ax.get_title() for ax in fig.axes] == ["building 1, floor 1 (n=2)", "building 2, floor 4 (n=2)"]
    assert not fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])  # separate frames


def test_invalid_requests_are_explained(plt):
    from indoorloc.datasets.plot import plot_distribution

    train, _ = _tables()
    flat = SampleTable(np.zeros((2, 1)), [[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="view"):
        plot_distribution(train, view="4d")
    with pytest.raises(ValueError, match="2-D view"):
        plot_distribution(train, view="3d", kind="density")
    with pytest.raises(ValueError, match="groups"):
        plot_distribution(train, color_by="user")
    with pytest.raises(ValueError, match="floor labels or 3-D positions"):
        plot_distribution(flat, view="3d")
    with pytest.raises(ValueError, match="needs floor labels"):
        plot_distribution(flat, panels="floor")
    with pytest.raises(TypeError, match="not a SampleTable"):
        plot_distribution([np.zeros((2, 2))])


def test_floor_plan_helper_draws_2d_and_3d_walls(plt):
    from indoorloc.datasets.plot import plot_floor_plan

    fig = plt.figure()
    lines = plot_floor_plan(PLAN, fig.add_subplot(1, 2, 1), floor=1)
    assert len(lines.get_segments()) == 1
    lines3d = plot_floor_plan(PLAN, fig.add_subplot(1, 2, 2, projection="3d"), floor=0, z=3.5)
    fig.canvas.draw()  # 3-D segments are projected at draw time
    assert len(lines3d.get_segments()) == 2
    assert plot_floor_plan({"walls": np.zeros((0, 4))}, fig.axes[0]) is None


def test_missing_plotting_packages_name_the_extra(monkeypatch):
    from indoorloc.datasets import plot

    train, _ = _tables()
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match=r"indoorloc\[plot\]"):
        plot.plot_distribution(train)
    with pytest.raises(ImportError, match=r"indoorloc\[plot\]"):
        plot.distribution_html(train)


def test_interactive_page_has_one_trace_per_category_and_a_floor_menu(tmp_path):
    pytest.importorskip("plotly")  # not installed in the default test environment
    from indoorloc.datasets.plot import distribution_html

    train, test = _tables()
    fig = distribution_html((train, test), tmp_path / "d3.html", view="3d")
    assert [t.name for t in fig.data] == ["train", "test", "walls", "walls"]
    assert (tmp_path / "d3.html").read_text().count("plotly") > 0
    fig = distribution_html((train, test), None, view="2d")
    assert [bool(t.visible) for t in fig.data] == [True, False, True, False, True, False]  # floor 0 first
    menu = fig.layout.updatemenus[0].buttons
    assert [b.label for b in menu] == ["floor 0", "floor 1"] and menu[1].args[0]["visible"][1]
    fig = distribution_html((train, test), None, view="3d", floors=[1])  # walls of the shown floors only
    walls = [t for t in fig.data if t.name == "walls"]
    assert len(walls) == 1 and set(walls[0].z) == {3.5}


@pytest.mark.skipif(not (UJI / "trainingData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_real_ujiindoorloc_has_five_floor_panels_and_three_buildings(plt):
    from indoorloc.datasets import load_dataset
    from indoorloc.datasets.plot import plot_distribution

    train, test = load_dataset("ujiindoorloc", root=UJI, download=False)
    fig = plot_distribution((train, test))
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert [t.split(" (")[0] for t in titles] == [f"floor {f}" for f in range(5)]
    assert sum(int(t.split("n=")[1].rstrip(")").replace(",", "")) for t in titles) == 19937 + 1111
    fig = plot_distribution(train, view="3d")
    labels = {c.get_label() for c in fig.axes[0].collections}
    assert {"building 0", "building 1", "building 2"} <= labels


def test_the_distribution_example_runs_offline(plt, tmp_path, monkeypatch, capsys):
    """examples/dataset_distribution.py on the simulated office (the 0.1 demos it replaces had rotted)."""
    import runpy

    from conftest import PROJECT

    monkeypatch.setattr(sys, "argv", ["dataset_distribution.py", "--out", str(tmp_path), "--floors", "2"])
    runpy.run_path(str(PROJECT / "examples" / "dataset_distribution.py"), run_name="__main__")
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["density_2d.png", "distribution_2d.png", "distribution_3d.png", "error_cdf.png",
                       "error_map.png", "trajectories.png"]
    out = capsys.readouterr().out
    assert "(simulated)" in out and "wknn" in out and "Kalman RTS" in out


# --------------------------------------------------------------------------- review additions
def test_tables_in_different_frames_are_refused(plt):
    from indoorloc.datasets.plot import plot_distribution

    train, test = _tables()
    mercator = SampleTable(np.zeros((2, 1)), [[-7600.0, 4864900.0], [-7500.0, 4864800.0]],
                           meta={"name": "toy", "crs": "EPSG:3857"})
    other_site = SampleTable(np.zeros((2, 1)), [[0.0, 0.0], [1.0, 1.0]], meta={"name": "elsewhere", "crs": "local"})
    for mixed in ((train, mercator), {"a": train, "b": other_site}):
        with pytest.raises(ValueError, match="different coordinate frames"):
            plot_distribution(mixed)
    plot_distribution((train, test))  # the splits of one dataset share its frame


def test_floors_with_their_own_frames_never_share_axes(plt):
    from indoorloc.datasets.plot import plot_distribution

    pos = np.array([[0.0, 0.0], [2.0, 1.0], [60.0, 40.0], [70.0, 45.0]])  # floor 2's origin is elsewhere
    t = SampleTable(np.zeros((4, 1)), pos, floor=[1, 1, 2, 2],
                    meta={"name": "ilc-like", "crs": "local (one frame per floor)"})
    fig = plot_distribution(t)
    assert [ax.get_title() for ax in fig.axes] == ["floor 1 (n=2)", "floor 2 (n=2)"]
    assert not fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])
    for request in ({"view": "3d"}, {"panels": "none"}):
        with pytest.raises(ValueError, match="own frame"):
            plot_distribution(t, **request)
    (ax,) = plot_distribution(t, view="3d", floors=[2]).axes  # one floor alone is fine
    fig = plot_distribution(t, kind="density", bin_size=1.0)  # one grid per floor frame, every sample once
    meshes = [c for ax in fig.axes[:2] for c in ax.collections if type(c).__name__ == "QuadMesh"]
    assert sum(float(m.get_array().sum()) for m in meshes) == 4 and meshes[1].get_coordinates()[0, 0, 0] == 60.0


def test_colours_do_not_depend_on_the_drawn_subset(plt):
    """max_points draws a subset; a category left out of it must not shift the colours of the others."""
    from indoorloc.datasets.plot import plot_distribution

    device = np.r_[1, np.full(99, 2)]
    t = SampleTable(np.zeros((100, 1)), np.random.default_rng(0).uniform(0, 5, (100, 2)), groups={"device": device},
                    meta={"name": "t"})
    colour = {}
    for max_points in (None, 5):
        fig = plot_distribution(t, color_by="device", max_points=max_points, random_state=3)
        colour[max_points] = {c.get_label(): tuple(c.get_facecolor()[0]) for c in fig.axes[0].collections}
    assert "device=1" not in colour[5]  # the seeded subset of 5 misses the single device-1 sample
    assert colour[5]["device=2"] == colour[None]["device=2"]


@pytest.mark.skipif(not (ILC_FLOOR / "geojson_map.json").is_file(), reason="ILC 2020 site2/F4 not found")
def test_real_ilc2020_floor_plan_is_drawn_and_its_map_saves_exactly(plt, tmp_path):
    """A measured floor plan (geojson walls) under the waypoints, as a FloorMap for tracks and in a saved model."""
    from indoorloc.apps import FloorMap, ParticleFilter
    from indoorloc.core import load_model
    from indoorloc.datasets import load_dataset
    from indoorloc.datasets.plot import plot_distribution
    from indoorloc.evaluation.plot import plot_trajectories

    wp = load_dataset("ilc2020", root=DATA_ROOT / "ilc2020", download=False, site="site2", floor="F4",
                      modality="waypoints")
    plan = wp.meta["floor_plan"]
    (ax,) = plot_distribution(wp).axes
    assert ax.get_title() == f"floor 4 (n={len(wp):,})"
    walls = [c for c in ax.collections if type(c).__name__ == "LineCollection"]
    assert len(walls) == 1 and len(walls[0].get_segments()) == int(np.sum(plan["wall_floor"] == 4)) > 100
    fmap = FloorMap.from_dict(plan)
    first = wp[wp.groups["trajectory"] == wp.groups["trajectory"][0]]
    ax = plot_trajectories(first, floor_plan=fmap, floor=4)
    assert len(ax.collections[0].get_segments()) == len(fmap.walls_on(4))
    again = load_model(ParticleFilter(50, floor_map=fmap, floor=4).save(tmp_path / "pf"))
    assert all(np.array_equal(a, b) for a, b in [(fmap.walls, again.floor_map.walls),
                                                 (fmap.wall_floor, again.floor_map.wall_floor),
                                                 (fmap.bounds, again.floor_map.bounds)])
