"""L4 figures: error CDFs, error maps, trajectories and bound maps (matplotlib, imported inside each function).

plot_cdf           empirical error CDFs of one or several methods (samples a method could not
                   place count as never reached, so such a curve stays below 100 %)
plot_error_map     true positions coloured by their error, optional arrows to the estimates
plot_trajectories  ground-truth and estimated tracks (L3 fixes, L5 trackers) over a floor plan
plot_bound_map     a positioning bound (CRLB, GDOP) over a grid, with the anchors

Every function draws on ``ax`` (a new figure if None) and returns the axes, so figures
compose and stay testable with the Agg backend. A floor plan is the ``meta["floor_plan"]``
dict of a dataset (``walls`` (W, 4) ``[x0, y0, x1, y1]``, optional ``wall_floor``) or any
object with ``walls`` and ``wall_floor`` attributes (``apps.maps.FloorMap``). The matplotlib
colour cycle styles the methods; only ground truth (black) and walls, arrows and unplaced samples
(greys) have fixed colours. New figures use matplotlib's constrained layout, and plan views print
real coordinates (no "1e6" offset: EPSG:3857 northings are about 4.9e6 m).
``pip install 'indoorloc[plot]'``.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..core import requires
from .functional import position_errors


def _axes(ax, figsize=(6.0, 4.5)):
    if ax is not None:
        return ax
    plt = requires("matplotlib.pyplot", "plot")
    return plt.subplots(figsize=figsize, layout="constrained")[1]  # long axis labels are not cut off


def _plain_ticks(ax) -> None:
    """Tick labels in real coordinates, without matplotlib's offset (``4.8649 ... 1e6``); at
    most five x intervals when the labels have 5+ characters (EPSG:3857 eastings such as
    ``-7650`` overlap at matplotlib's default density)."""
    try:
        ax.ticklabel_format(style="plain", useOffset=False)
    except AttributeError:  # the caller's axes use a non-scalar formatter: leave it alone
        return
    if max(len(f"{v:.0f}") for v in ax.get_xlim()) >= 5:
        ticker = requires("matplotlib.ticker", "plot")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


def _check_aligned(truth, other, name: str) -> None:
    """Refuse rows that do not line up, as ``evaluate`` does: both sides carry ids and differ."""
    a, b = getattr(truth, "ids", None), getattr(other, "ids", None)
    if a is not None and b is not None and not np.array_equal(np.asarray(a), np.asarray(b)):
        raise ValueError(f"{name} and the ground truth carry different ids: their rows are not aligned "
                         "(was one side reordered or subset?)")


def _errors_of(value, name) -> np.ndarray:
    e = np.asarray(value.errors if hasattr(value, "errors") else value, dtype=np.float64).reshape(-1)
    n_failed = getattr(value, "n_failed", None)
    if n_failed is not None and int(np.sum(~np.isfinite(e))) > n_failed:  # evaluate() on room-level data
        raise ValueError(f"{name!r} has NaN errors that are not unplaced samples (n_failed={n_failed}): its "
                         "positions have no axes (room-level data), so there is no error CDF to draw")
    return e


def _xy(value, name: str) -> np.ndarray:
    """(N, 2) plan coordinates of an array, a SampleTable or a Prediction (first two axes)."""
    a = np.asarray(getattr(value, "pos", value), dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(f"{name} must be (N, >=2) positions, got shape {a.shape}")
    return a[:, :2]


def _walls(floor_plan, floor=None) -> np.ndarray:
    """(W, 2, 2) wall segments of ``floor`` (all floors if None) from a plan dict or object."""
    get = floor_plan.get if isinstance(floor_plan, Mapping) else lambda k, d=None: getattr(floor_plan, k, d)
    walls = get("walls")
    if walls is None:
        raise ValueError("floor_plan has no 'walls'")
    walls = np.asarray(walls, dtype=np.float64).reshape(-1, 2, 2)
    wall_floor = get("wall_floor")
    if floor is not None and wall_floor is not None:
        walls = walls[np.asarray(wall_floor).reshape(-1) == int(floor)]
    return walls


def _draw_walls(ax, floor_plan, floor=None, **style):
    """Draw the walls as one LineCollection (under the data) and return it."""
    collections = requires("matplotlib.collections", "plot")
    style = {"colors": "0.35", "linewidths": 1.2, "zorder": 0.5, "label": "walls", **style}
    lines = collections.LineCollection(_walls(floor_plan, floor), **style)
    ax.add_collection(lines)
    ax.autoscale_view()
    return lines


def plot_cdf(errors, ax=None, *, label: str | None = None, unit: str = "m", percent: bool = True,
             xmax: float | None = None, **kwargs):
    """Empirical CDF of positioning errors as a step curve, one per method.

    errors   an (N,) array, an EvaluationResults, or ``{label: either}`` for several curves.
    percent  y axis in percent (as in the literature) rather than a fraction.
    xmax     x-axis limit (default: the largest error, or the 99th percentile if the tail is
             more than 3x longer than it, so a few outliers do not squash the curves).

    NaN errors are samples the method could not place (``EvaluationResults.n_failed``). They
    stay in the denominator, as errors never reached: the curve ends at ``(N - n_failed) / N``
    and is drawn flat to the right edge, and its label says how many were not placed. The curve
    is ``evaluation.error_cdf(errors, t)`` (the same convention); note that
    ``EvaluationResults.median_error`` and the other statistics use the placed samples only, so
    with failures the 50 % crossing lies right of ``median_error``.
    """
    ax = _axes(ax)
    curves = errors if isinstance(errors, Mapping) else {label: errors}
    scale = 100.0 if percent else 1.0
    steps, top = [], 0.0
    for name, value in curves.items():
        e = _errors_of(value, name)
        placed = np.sort(e[np.isfinite(e)])
        n_failed = len(e) - len(placed)
        if n_failed and name is not None:
            name = f"{name} ({n_failed} of {len(e)} not placed)"
        elif n_failed:
            name = f"{n_failed} of {len(e)} not placed"
        x = np.concatenate([[0.0], placed])
        p = np.arange(len(placed) + 1) / max(len(e), 1) * scale
        steps.append((name, x, p, n_failed))
        if len(placed):
            p99 = np.percentile(placed, 99)
            top = max(top, p99 if placed[-1] > 3 * p99 else placed[-1])
    right = xmax if xmax is not None else top or 1.0
    for name, x, p, n_failed in steps:
        if n_failed:  # the plateau of a curve that never reaches 100 %
            x, p = np.append(x, max(right, x[-1])), np.append(p, p[-1])
        ax.step(x, p, where="post", label=name, **kwargs)
    ax.set_xlim(0.0, right)
    ax.set_ylim(0.0, scale)
    ax.set_xlabel(f"Positioning error [{unit}]")
    ax.set_ylabel("CDF [%]" if percent else "CDF")
    ax.grid(True, alpha=0.3)
    if any(name is not None for name, *_ in steps):
        ax.legend(loc="lower right")
    return ax


def plot_error_map(pos_true, pos_pred=None, ax=None, *, errors=None, arrows: bool = False, cmap: str = "viridis",
                   unit: str = "m", s: float = 12.0, colorbar: bool = True, vmax: float | None = None,
                   floor_plan=None, floor: int | None = None, pos_unit: str | None = None):
    """True positions coloured by their error; optional arrows to the estimates.

    pos_true    (N, >=2) true coordinates (first two axes plotted), or a SampleTable.
    pos_pred    (N, >=2) estimates, or a Prediction; needed for ``arrows`` or if ``errors`` is None.
    errors      (N,) errors to colour by (default: Euclidean error over all axes).
    unit        unit of the errors; ``pos_unit`` that of the axes (default: ``unit``). They differ
                when errors are scaled, e.g. EPSG:3857 axes with errors in ground metres.
    vmax        top of the colour scale (share it between the panels of several methods); larger
                errors take the top colour and the colour bar ends in an arrow.
    floor_plan  walls drawn underneath (``meta["floor_plan"]`` or a FloorMap); ``floor``
                selects the walls of one floor (select the rows of that floor yourself).

    Samples with a NaN error (not placed by the method) are drawn as grey crosses, with a
    legend entry that counts them, instead of silently disappearing. A SampleTable and a
    Prediction that both carry ``ids`` must list the same ids in the same order.
    """
    if pos_pred is not None:
        _check_aligned(pos_true, pos_pred, "pos_pred")
    ax = _axes(ax, (6.0, 5.0))
    true = np.asarray(getattr(pos_true, "pos", pos_true), dtype=np.float64)
    pred = None if pos_pred is None else np.asarray(getattr(pos_pred, "pos", pos_pred), dtype=np.float64)
    if errors is None:
        if pred is None:
            raise ValueError("pass pos_pred or errors")
        errors = position_errors(true, pred)
    errors = np.asarray(errors, dtype=np.float64)
    if true.ndim != 2 or true.shape[1] < 2 or len(errors) != len(true):
        raise ValueError(f"pos_true must be (N, >=2) with one error per row; got {true.shape} and {errors.shape}")
    if floor_plan is not None:
        _draw_walls(ax, floor_plan, floor)
    placed = np.isfinite(errors)
    order = np.flatnonzero(placed)[np.argsort(errors[placed], kind="stable")]  # largest errors drawn last, on top
    if arrows:
        if pred is None:
            raise ValueError("arrows=True needs pos_pred")
        d = pred[:, :2] - true[:, :2]
        ax.quiver(true[order, 0], true[order, 1], d[order, 0], d[order, 1], angles="xy", scale_units="xy",
                  scale=1.0, width=0.002, color="0.5", alpha=0.6)
    points = ax.scatter(true[order, 0], true[order, 1], c=errors[order], cmap=cmap, s=s, vmin=0.0, vmax=vmax)
    if colorbar:
        clipped = vmax is not None and bool(placed.any()) and float(errors[placed].max()) > vmax
        ax.figure.colorbar(points, ax=ax, label=f"error [{unit}]", extend="max" if clipped else "neither")
    if not placed.all():
        ax.scatter(true[~placed, 0], true[~placed, 1], marker="x", color="0.5", s=s,
                   label=f"not placed ({int((~placed).sum())})")
        ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"x [{pos_unit or unit}]")
    ax.set_ylabel(f"y [{pos_unit or unit}]")
    _plain_ticks(ax)
    return ax


def plot_trajectories(truth=None, estimates=None, floor_plan=None, ax=None, *, floor: int | None = None,
                      trajectory=None, unit: str = "m", endpoints: bool = True, linewidth: float = 1.4):
    """Ground-truth and estimated tracks in plan view, over the walls of a floor plan.

    truth        (T, >=2) true positions, a SampleTable (its ``groups["trajectory"]`` is used)
                 or None; drawn in black with a circle at the start and a square at the end.
    estimates    ``{name: (T, >=2) array or Prediction}`` (e.g. raw L3 fixes, a Kalman
                 track, ``PDRFusion.run(...)[1]``), or one array (named "estimate"). NaN rows
                 (no estimate yet) leave gaps.
    floor_plan   ``meta["floor_plan"]`` of the dataset or a FloorMap; ``floor`` selects the
                 walls of one floor (None: every wall).
    trajectory   (T,) ids; the lines break where the id changes, so several walks stored in one
                 table are not joined (default: ``truth.groups["trajectory"]`` if present).

    Returns the axes. Each track is one ``Line2D`` labelled with its name (``"ground truth"``
    for ``truth``); the start and end markers are two more lines labelled ``"start"``/``"end"``.
    Estimates that carry ``ids`` (a Prediction) must match the ids of ``truth`` row for row.
    """
    if trajectory is None and truth is not None:
        trajectory = getattr(truth, "groups", {}).get("trajectory")
    true_xy = None if truth is None else _xy(truth, "truth")
    if estimates is not None and not isinstance(estimates, Mapping):
        estimates = {"estimate": estimates}
    for name, value in (estimates or {}).items():
        _check_aligned(truth, value, f"estimates[{name!r}]")
    ax = _axes(ax, (7.0, 5.5))
    tracks = [(str(name), _xy(value, f"estimates[{name!r}]")) for name, value in (estimates or {}).items()]
    if true_xy is None and not tracks:
        raise ValueError("pass truth and/or estimates")
    lengths = ([] if true_xy is None else [len(true_xy)]) + [len(xy) for _, xy in tracks]
    n = lengths[0]
    if any(m != n for m in lengths):
        raise ValueError(f"every track needs one row per time step, got lengths {lengths}")
    breaks = np.zeros(0, dtype=np.int64)
    if trajectory is not None:
        ids = np.asarray(trajectory).reshape(-1)
        if len(ids) != n:
            raise ValueError(f"trajectory has {len(ids)} ids for {n} rows")
        breaks = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    if floor_plan is not None:
        _draw_walls(ax, floor_plan, floor)
    if true_xy is not None:
        line = np.insert(true_xy, breaks, np.nan, axis=0)  # a NaN row between two walks breaks the line
        ax.plot(line[:, 0], line[:, 1], color="k", lw=linewidth + 0.4, label="ground truth", zorder=3)
        if endpoints and n:
            starts, ends = np.r_[0, breaks], np.r_[breaks - 1, n - 1]
            ax.plot(true_xy[starts, 0], true_xy[starts, 1], "o", color="k", mfc="white", ms=6, zorder=4, label="start")
            ax.plot(true_xy[ends, 0], true_xy[ends, 1], "s", color="k", ms=5, zorder=4, label="end")
    for name, xy in tracks:
        line = np.insert(xy, breaks, np.nan, axis=0)
        ax.plot(line[:, 0], line[:, 1], lw=linewidth, alpha=0.85, label=name, zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    _plain_ticks(ax)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    return ax


def plot_bound_map(anchors, bound, *, extent=None, resolution: int = 100, ax=None, cmap: str = "magma",
                   unit: str = "m", levels=None, colorbar: bool = True):
    """A 2-D map of a positioning bound over a grid, with the anchors marked.

    bound     a function ``points (P, 2) -> (P,)``, e.g.
              ``lambda p: toa_crlb(anchors, p, sigma=0.3)`` or ``lambda p: gdop(anchors, p)``.
    extent    ``(xmin, xmax, ymin, ymax)``; default: the anchors' box plus 10 % margin.
    Values that are infinite (degenerate geometry) or NaN are left blank.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(f"plot_bound_map draws 2-D anchors (A, 2), got {anchors.shape}")
    if extent is None:
        lo, hi = anchors.min(axis=0), anchors.max(axis=0)
        pad = 0.1 * np.maximum(hi - lo, 1.0)
        extent = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
    xs = np.linspace(extent[0], extent[1], resolution)
    ys = np.linspace(extent[2], extent[3], resolution)
    gx, gy = np.meshgrid(xs, ys)
    values = np.asarray(bound(np.column_stack([gx.ravel(), gy.ravel()])), dtype=np.float64).reshape(gx.shape)
    values = np.where(np.isfinite(values), values, np.nan)
    ax = _axes(ax, (6.0, 5.0))
    mesh = ax.contourf(gx, gy, values, levels=20 if levels is None else levels, cmap=cmap)
    ax.plot(anchors[:, 0], anchors[:, 1], "^", color="cyan", markeredgecolor="k", markersize=9, label="anchors")
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=f"bound [{unit}]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x [{unit}]")
    ax.set_ylabel(f"y [{unit}]")
    return ax


__all__ = ["plot_bound_map", "plot_cdf", "plot_error_map", "plot_trajectories"]
