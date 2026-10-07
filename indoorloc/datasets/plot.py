"""L1 figures: where the samples of a dataset lie (matplotlib, imported inside each function).

plot_distribution   2-D panels (one per floor, or per building and floor) or a 3-D view with
                    the floors stacked; points coloured by split, floor, building or any
                    ``groups`` column, or a density map of samples per cell; the walls of
                    ``meta["floor_plan"]`` drawn underneath. Returns the matplotlib Figure.
plot_floor_plan     the walls of a ``meta["floor_plan"]`` dict on 2-D (or 3-D) axes.
distribution_html   the same view as an interactive, self-contained plotly page (optional
                    dependency; see its docstring).

This ports the 0.1 ``indoorloc.visualization.distribution`` module (plotly pages built from
0.1 dataset objects) to SampleTables. Differences from 0.1: coordinates stay in the frame of
the dataset (0.1 shifted them to start at 0; pass ``relative=True`` for that), the static
figures use matplotlib (``pip install 'indoorloc[plot]'``) and are testable with the Agg
backend, and nothing opens a browser. Only ``SampleTable`` fields are read: ``pos``,
``floor``, ``building``, ``groups`` and ``meta`` (``name``, ``split``, ``pos_names``,
``pos_units``, ``crs``, ``floor_plan``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..core import SampleTable, requires

_MAX_CATEGORIES = 20  # more distinct values than this are coloured on a continuous scale
# matplotlib's tab10 / tab20 as hex, for the plotly page (which must not import matplotlib)
_TAB10 = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf")
_TAB20 = ("#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd",
          "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
          "#17becf", "#9edae5")


# ----------------------------------------------------------------------------------- inputs
def _labelled(tables) -> list[tuple[str, SampleTable]]:
    """``table``, ``(train, test)``, ``[t1, t2]`` or ``{label: table}`` -> ``[(label, table)]``."""
    if isinstance(tables, SampleTable):
        tables = [tables]
    if isinstance(tables, Mapping):
        items = [(str(k), t) for k, t in tables.items()]
    else:
        tables = list(tables)
        items = []
        for i, t in enumerate(tables):
            meta = getattr(t, "meta", {})
            label = meta.get("split") or (meta.get("name") if len(tables) == 1 else None) or f"table {i}"
            items.append((str(label), t))
        labels = [label for label, _ in items]
        items = [(f"{label} [{i}]" if labels.count(label) > 1 else label, t) for i, (label, t) in enumerate(items)]
    if not items:
        raise ValueError("no tables to plot")
    for label, t in items:
        if not isinstance(t, SampleTable):
            raise TypeError(f"{label!r} is a {type(t).__name__}, not a SampleTable")
        if t.pos.shape[1] < 2:
            raise ValueError(f"{label!r} has {t.pos.shape[1]}-D positions; a distribution plot needs x and y")
    return items


def _subsample(n: int, max_points: int | None, rng) -> np.ndarray:
    if max_points is None or n <= max_points:
        return np.arange(n)
    return np.sort(rng.choice(n, size=int(max_points), replace=False))


def _column(table: SampleTable, color_by: str, label: str) -> np.ndarray:
    if color_by == "split":
        return np.full(len(table), label, dtype=object)
    if color_by in ("floor", "building"):
        values = getattr(table, color_by)
        if values is None:
            raise ValueError(f"cannot colour by {color_by}: {label!r} has no {color_by} labels")
        return values
    if color_by not in table.groups:
        raise ValueError(f"cannot colour by {color_by!r}: {label!r} has groups {sorted(table.groups)}")
    return table.groups[color_by]


def _frames_per_building(meta: Mapping) -> bool:
    """True when ``meta["crs"]`` says each building label has its own coordinate frame.

    The L1 loaders state it in the crs text: ``"local-per-building"`` (SODIndoorLoc),
    ``"local (one frame per room; see building)"`` (BLE-Indoor, H-WILD), ``"... per zone; see
    building"`` (iBeacon RSSI), ``"local grid (one per area; see building)"`` (CSI fingerprint).
    Points of two such buildings must never share one pair of axes.
    """
    return "building" in str(meta.get("crs") or "").lower()


def _frames_per_floor(meta: Mapping) -> bool:
    """True when ``meta["crs"]`` says each floor has its own frame (e.g. ``"local (one frame
    per floor)"``): floor panels then get their own axes and floors are never stacked in 3-D."""
    return "floor" in str(meta.get("crs") or "").lower()


def _one_frame(items) -> Mapping:
    """``meta`` of the first table, after checking that every table is in one coordinate frame:
    the same ``crs`` and, for local frames, no two different dataset names."""
    crs = sorted({str(t.meta.get("crs") or "local") for _, t in items})
    names = sorted({str(t.meta["name"]) for _, t in items if t.meta.get("name") is not None})
    if len(crs) > 1 or (crs[0].startswith("local") and len(names) > 1):
        raise ValueError(f"the tables are in different coordinate frames (crs {crs}, datasets {names}); "
                         "their positions cannot share axes: plot them separately")
    return items[0][1].meta


def _plain_ticks(ax) -> None:
    """Real coordinates without matplotlib's offset (``4.8649 ... 1e6``); at most five x
    intervals when the labels have 5+ characters (EPSG:3857 eastings such as ``-7650``
    overlap at matplotlib's default density)."""
    ax.ticklabel_format(style="plain", useOffset=False)
    if max(len(f"{v:.0f}") for v in ax.get_xlim()) >= 5:
        ticker = requires("matplotlib.ticker", "plot")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


def _check_floor_frames(meta: Mapping, floors_drawn) -> None:
    """Refuse to draw several floors on one axes (or stack them in 3-D) when each has its own frame."""
    if _frames_per_floor(meta) and len(set(floors_drawn)) > 1:
        raise ValueError(f"each floor of this dataset has its own frame (crs {meta.get('crs')!r}), so floors "
                         f"{sorted(set(floors_drawn))} cannot share axes or be stacked in 3-D; use view='2d' with "
                         "panels='floor' (the default) or select one floor with floors=")


def _default_color_by(items, view: str, panels: str) -> str:
    if len(items) > 1:
        return "split"
    t = items[0][1]
    if t.building is not None and "building" not in panels and len(np.unique(t.building)) > 1:
        return "building"
    if view == "3d" and t.floor is not None:
        return "floor"
    return "split"


def _short_unit(unit: str) -> str:
    """An axis-label unit: long explanations (``"grid cells of the source map (column letter
    A=1, ...); cell size not stated"``) are cut at their first parenthesis or semicolon."""
    unit = str(unit or "")
    if len(unit) <= 40:
        return unit
    for sep in (" (", ";", ","):
        unit = unit.split(sep)[0]
    return unit.strip()


def _nice_step(x: float) -> float:
    """The largest 1, 2, 2.5 or 5 x 10^k that does not exceed ``x`` (a readable cell size)."""
    if not (x > 0 and math.isfinite(x)):
        return 1.0
    base = 10.0 ** math.floor(math.log10(x))
    return max(m * base for m in (1.0, 2.0, 2.5, 5.0) if m * base <= x * (1 + 1e-12))


class _Colours:
    """Categorical colours (tab10/tab20) for up to 20 values, otherwise a continuous scale."""

    def __init__(self, plt, values: list[np.ndarray], color_by: str, order: list | None, cmap: str):
        self.color_by = color_by
        joined = np.concatenate(values)
        cats = order if order is not None else np.unique(joined).tolist()
        dtype_kind = joined.dtype.kind  # ints: categories if few; floats: continuous; str/bool: categories
        self.categorical = order is not None or dtype_kind not in "iuf" or (
            dtype_kind in "iu" and len(cats) <= _MAX_CATEGORIES)
        if self.categorical and len(cats) > _MAX_CATEGORIES:
            raise ValueError(f"{color_by!r} has {len(cats)} distinct non-numeric values; at most "
                             f"{_MAX_CATEGORIES} can be told apart by colour")
        if self.categorical:
            palette = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            self.categories = cats
            self.colours = {c: palette(i) for i, c in enumerate(cats)}
        else:
            finite = joined[np.isfinite(joined.astype(np.float64))]
            self.norm = plt.Normalize(*(finite.min(), finite.max()) if len(finite) else (0.0, 1.0))
            self.cmap = plt.get_cmap(cmap)

    def name(self, value) -> str:
        if self.color_by == "split":
            return str(value)
        return f"{self.color_by} {value}" if self.color_by in ("floor", "building") else f"{self.color_by}={value}"


def _hex_palette(n: int) -> tuple[str, ...]:
    """Distinct colours for ``n <= 20`` categories (tab10, or tab20 beyond ten)."""
    if n > _MAX_CATEGORIES:
        raise ValueError(f"at most {_MAX_CATEGORIES} categories can be told apart by colour, got {n}")
    return _TAB10 if n <= 10 else _TAB20


def _z_label(meta: Mapping, has_floor: bool, floor_height: float, unit: str) -> str:
    """Label of the 3-D axis: floors (stacked labels) or the dataset's third coordinate."""
    if has_floor:
        return "floor" if floor_height == 1.0 else f"height [{unit or 'm'}]"
    names = tuple(meta.get("pos_names") or ())
    return (names[2] if len(names) >= 3 else "z") + (f" [{unit}]" if unit else "")


def _check_floor_filter(items, floors) -> None:
    if floors is not None and any(t.floor is None for _, t in items):
        raise ValueError("floors= selects rows by floor label, but a table has no floor labels")


def _axis_labels(meta: Mapping, relative_origin) -> tuple[str, str, str]:
    names = tuple(meta.get("pos_names") or ("x", "y", "z"))
    unit = _short_unit(meta.get("pos_units") or ("m" if str(meta.get("crs") or "local").startswith("local") else ""))
    out = []
    for d, name in enumerate(names[:2]):
        offset = "" if relative_origin is None else f" - {relative_origin[d]:,.1f}"
        out.append(f"{name}{offset}" + (f" [{unit}]" if unit else ""))
    return out[0], out[1], unit


def _plan_walls(floor_plan, floor=None) -> np.ndarray:
    """(W, 2, 2) walls of one floor (every floor if None) of a plan dict or object."""
    get = floor_plan.get if isinstance(floor_plan, Mapping) else lambda k, d=None: getattr(floor_plan, k, d)
    walls = get("walls")
    if walls is None:
        return np.zeros((0, 2, 2))
    walls = np.asarray(walls, dtype=np.float64).reshape(-1, 2, 2)
    wall_floor = get("wall_floor")
    if floor is not None and wall_floor is not None:
        walls = walls[np.asarray(wall_floor).reshape(-1) == int(floor)]
    return walls


def plot_floor_plan(floor_plan, ax=None, *, floor: int | None = None, z: float | None = None, offset=(0.0, 0.0),
                    **style):
    """Draw the walls of ``floor_plan`` (``meta["floor_plan"]``: ``walls`` (W, 4)
    ``[x0, y0, x1, y1]`` and optionally ``wall_floor``; or an object with those attributes)
    as one line collection under the data. ``floor`` selects one floor's walls; ``z`` draws
    them at that height on 3-D axes; ``offset`` is subtracted from x and y. Returns the
    collection (None if the plan has no walls)."""
    walls = _plan_walls(floor_plan, floor) - np.asarray(offset, dtype=np.float64)[:2]
    if ax is None:
        ax = requires("matplotlib.pyplot", "plot").figure().add_subplot(projection=None if z is None else "3d")
    if len(walls) == 0:
        return None
    style = {"colors": "0.4", "linewidths": 0.8, "zorder": 0.5, **style}
    if z is None:
        lines = requires("matplotlib.collections", "plot").LineCollection(walls, **style)
        ax.add_collection(lines)
        ax.autoscale_view()
        return lines
    art3d = requires("mpl_toolkits.mplot3d.art3d", "plot")
    segs = np.concatenate([walls, np.full(walls.shape[:2] + (1,), float(z))], axis=2)
    lines = art3d.Line3DCollection(segs, **style)
    ax.add_collection3d(lines)
    return lines


# ----------------------------------------------------------------------------------- figure
def plot_distribution(tables, *, view: str = "2d", color_by: str | None = None, panels: str = "auto",
                      kind: str = "scatter", bin_size: float | None = None, floors=None,
                      floor_height: float | None = None, floor_plan="auto", relative: bool = False,
                      max_points: int | None = None, random_state=0, marker_size: float | None = None,
                      alpha: float = 0.7, cmap: str = "viridis", ncols: int = 3, figsize=None,
                      title: str | None = None):
    """Spatial distribution of the samples of one or several tables; returns the Figure.

    Parameters
    ----------
    tables : SampleTable, sequence of tables (e.g. ``load_dataset(name)`` -> ``(train, test)``)
        or ``{label: table}``. Tables are labelled by the mapping keys, else ``meta["split"]``.
    view : ``"2d"`` (one panel per floor, equal axes) or ``"3d"`` (floors stacked: z is
        ``floor * floor_height``; without floor labels, the third coordinate of ``pos``).
    color_by : ``"split"`` (one colour per table), ``"floor"``, ``"building"`` or the name of
        a ``groups`` column (``"device"``, ``"user"``, ``"time"``, ...). Up to 20 integer or
        string values get distinct colours and a legend; floats and larger integer sets get a
        continuous ``cmap`` and a colour bar (a requested colouring always shows its legend,
        even when every sample has the same value). Default: split for several tables, else
        building if there are several, else floor in 3-D, else one colour.
    panels : ``"floor"``, ``"building"``, ``"building_floor"``, ``"none"`` or ``"auto"``.
        ``"auto"``: when ``meta["crs"]`` says each building has its own frame (the crs text
        mentions "building", e.g. SODIndoorLoc's ``"local-per-building"`` or BLE-Indoor's
        ``"local (one frame per room; see building)"``), one panel per building (and floor, if
        labelled), so points of different frames never share axes; else ``"floor"`` when there
        are floor labels, else one panel. In 3-D, panels split by building only. The floor
        panels of one building share their x and y axes unless the crs text says each floor
        has its own frame (it mentions "floor"); such floors are never stacked in 3-D.
        Tables in different frames (another ``crs``, or two datasets with local frames) are
        refused.
    kind : ``"scatter"`` or ``"density"`` (2-D only: samples per ``bin_size`` cell over every
        table, zero cells blank, one shared colour scale; ``color_by`` is ignored).
    bin_size : density cell size in coordinate units (default: a round 1, 2, 2.5 or 5 x 10^k
        near 1/60 of the larger extent).
    floors : floors to show (default: all).
    floor_height : storey height for the 3-D view (default: ``meta["floor_plan"]["floor_height"]``
        if the dataset has one, else 1 so that z counts floors). The 3-D box draws x and y at
        one scale; z is stretched for readability (it is not to scale).
    floor_plan : ``"auto"`` (``meta["floor_plan"]`` of the first table, if any), a plan dict,
        an object with ``walls``/``wall_floor``, or None.
    relative : subtract the joint minimum of x and y (the 0.1 behaviour); the axis labels
        state the offset. The default keeps the dataset frame.
    max_points : scatter at most this many points per table (a seeded random subset; the legend
        and titles always give the full counts; a density map always counts every sample).
    random_state : seed of that subset.

    Returns
    -------
    matplotlib.figure.Figure; ``fig.axes`` holds one axes per panel (plus colour bars).
    """
    plt = requires("matplotlib.pyplot", "plot")
    if view not in ("2d", "3d"):
        raise ValueError(f"view must be '2d' or '3d', not {view!r}")
    if kind not in ("scatter", "density"):
        raise ValueError(f"kind must be 'scatter' or 'density', not {kind!r}")
    if kind == "density" and view == "3d":
        raise ValueError("kind='density' is a 2-D view")
    items = _labelled(tables)
    meta = _one_frame(items)
    has_floor = all(t.floor is not None for _, t in items)
    has_building = all(t.building is not None for _, t in items)
    if panels == "auto":
        per_building = _frames_per_building(meta) and has_building
        panels = ("building_floor" if has_floor else "building") if per_building else ("floor" if has_floor else "none")
    if panels not in ("floor", "building", "building_floor", "none"):
        raise ValueError(f"panels must be 'floor', 'building', 'building_floor', 'none' or 'auto', not {panels!r}")
    if "floor" in panels and not has_floor:
        raise ValueError(f"panels={panels!r} needs floor labels in every table")
    if "building" in panels and not has_building:
        raise ValueError(f"panels={panels!r} needs building labels in every table")
    if view == "3d":
        panels = "building" if "building" in panels else "none"
    if view == "3d" and not has_floor and any(t.pos.shape[1] < 3 for _, t in items):
        raise ValueError("the 3-D view needs floor labels or 3-D positions")
    _check_floor_filter(items, floors)
    asked = color_by is not None  # a requested colouring always gets its key, even with one value
    color_by = color_by or _default_color_by(items, view, panels)

    # rows: the floor filter (``full``, what titles and legends count), then the seeded subset drawn
    rng = np.random.default_rng(random_state)
    full = [np.arange(len(t)) if floors is None or t.floor is None else np.flatnonzero(np.isin(t.floor, floors))
            for _, t in items]
    counts = {label: len(idx) for (label, _), idx in zip(items, full)}
    rows = full if kind == "density" else [idx[_subsample(len(idx), max_points, rng)] for idx in full]
    if not sum(counts.values()):
        raise ValueError("no samples to plot (check floors=)")
    if has_floor and "floor" not in panels:  # several floors on one axes (3-D, or 2-D without floor panels)
        _check_floor_frames(meta, [int(f) for (_, t), r in zip(items, full) for f in np.unique(t.floor[r])])
    if bin_size is not None and not (float(bin_size) > 0 and math.isfinite(float(bin_size))):
        raise ValueError(f"bin_size must be a positive cell size, got {bin_size!r}")
    xy_all = np.concatenate([t.pos[r, :2] for (_, t), r in zip(items, full)])
    xy_all = xy_all[np.all(np.isfinite(xy_all), axis=1)]
    if not len(xy_all):
        raise ValueError("no finite positions to plot")
    origin = xy_all.min(axis=0) if relative else None
    shift = np.zeros(2) if origin is None else origin
    xlabel, ylabel, unit = _axis_labels(meta, origin)
    plan = meta.get("floor_plan") if isinstance(floor_plan, str) and floor_plan == "auto" else floor_plan
    if floor_height is None:
        floor_height = float((plan.get("floor_height") if isinstance(plan, Mapping) else None) or 1.0)

    def panel_keys(t, r) -> np.ndarray:  # (n, 2) int64: (building, floor), 0 where not a panel axis
        b = t.building[r] if "building" in panels else np.zeros(len(r), np.int64)
        f = t.floor[r] if "floor" in panels else np.zeros(len(r), np.int64)
        return np.stack([b, f], axis=1).astype(np.int64)

    keys = [panel_keys(t, r) for (_, t), r in zip(items, rows)]
    full_keys = [panel_keys(t, r) for (_, t), r in zip(items, full)]
    panel_ids = [tuple(k) for k in np.unique(np.concatenate(full_keys), axis=0).tolist()]
    n_panels = len(panel_ids)
    ncols = max(1, min(int(ncols), n_panels))
    nrows = math.ceil(n_panels / ncols)
    if figsize is None and view == "2d":
        extent = xy_all.max(axis=0) - xy_all.min(axis=0)
        aspect = float(np.clip(extent[1] / extent[0], 0.3, 2.0)) if extent[0] > 0 else 1.0
        figsize = (4.6 * ncols + 2.0, (4.6 * aspect + 0.9) * nrows + 0.5)
    elif figsize is None:
        figsize = (6.5 * ncols + 1.5, 5.5 * nrows)
    fig = plt.figure(figsize=figsize, layout="constrained")
    per_floor = _frames_per_floor(meta)
    axes, frame_axes = [], {}  # 2-D panels of one frame (building, and floor if floors have frames) share x and y
    for i, (b, f) in enumerate(panel_ids):
        if view == "2d":
            frame = (b, f if per_floor else 0)
            first = frame_axes.get(frame)
            axes.append(fig.add_subplot(nrows, ncols, i + 1, sharex=first, sharey=first))
            frame_axes.setdefault(frame, axes[-1])
        else:
            axes.append(fig.add_subplot(nrows, ncols, i + 1, projection="3d"))

    colours = None
    if kind == "scatter":  # colours from every selected row, so a subset (max_points) keeps them
        columns = [np.asarray(_column(t, color_by, label)) for label, t in items]
        colours = _Colours(plt, [c[idx] for c, idx in zip(columns, full)], color_by,
                           [label for label, _ in items] if color_by == "split" else None, cmap)
        values = [c[r] for c, r in zip(columns, rows)]
    size = marker_size if marker_size is not None else (6.0 if view == "2d" else 4.0)
    draw_order = sorted(range(len(items)), key=lambda i: -len(rows[i]))
    if kind == "density":  # one cell size for every panel (comparable counts), one grid per frame
        def frame_of(key):  # the frame a panel lies in: its building, and its floor if floors have frames
            return int(key[0]), int(key[1]) if per_floor else 0

        boxes = {}  # frame -> (lo, hi) of its finite positions
        for (_, t), r, k in zip(items, full, full_keys):
            xy = t.pos[r, :2] - shift
            ok = np.all(np.isfinite(xy), axis=1)
            for key in np.unique(k[ok], axis=0).tolist():
                pts = xy[ok & np.all(k == key, axis=1)]
                lo, hi = boxes.get(frame_of(key), (pts.min(axis=0), pts.max(axis=0)))
                boxes[frame_of(key)] = (np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0)))
        step = float(bin_size) if bin_size is not None else _nice_step(
            max(float(np.max(hi - lo)) for lo, hi in boxes.values()) / 60.0)
        edges = {fr: [lo[d] + step * np.arange(max(1, math.ceil((hi[d] - lo[d]) / step + 1e-9)) + 1) for d in range(2)]
                 for fr, (lo, hi) in boxes.items()}
        grids = []

    for ax, (b, f) in zip(axes, panel_ids):
        on_panel = [np.all(k == (b, f), axis=1) for k in keys]
        if kind == "density":
            pts = np.concatenate([t.pos[r[m], :2] for (_, t), r, m in zip(items, rows, on_panel)]) - shift
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            frame = frame_of((b, f))
            if frame in edges:
                grids.append((ax, edges[frame], np.histogram2d(pts[:, 0], pts[:, 1], bins=edges[frame])[0]))
        else:
            for i in draw_order:  # the largest table first, so a small one stays visible on top
                t, r, m, v = items[i][1], rows[i], on_panel[i], values[i]
                xy, v = t.pos[r[m], :2] - shift, v[m]
                args = [xy[:, 0], xy[:, 1]]
                if view == "3d":  # every table stacks by floor label, or every table uses its z
                    args.append(t.floor[r[m]] * floor_height if has_floor else t.pos[r[m], 2])
                if colours.categorical:
                    for c in colours.categories:
                        hit = v == c
                        if hit.any():
                            ax.scatter(*[a[hit] for a in args], s=size, color=colours.colours[c], alpha=alpha,
                                       linewidths=0, label=colours.name(c))
                else:
                    ax.scatter(*args, s=size, c=v.astype(np.float64), cmap=colours.cmap, norm=colours.norm,
                               alpha=alpha, linewidths=0)
        if plan is not None and view == "2d":
            plot_floor_plan(plan, ax, floor=f if "floor" in panels else None, offset=shift)
        elif plan is not None and has_floor:
            for fl in sorted({int(x) for (_, t), r in zip(items, rows) for x in t.floor[r]}):
                plot_floor_plan(plan, ax, floor=fl, z=fl * floor_height, offset=shift)
        name = ", ".join(([f"building {b}"] if "building" in panels else [])
                         + ([f"floor {f}"] if "floor" in panels else []))
        n_here = sum(int(np.all(k == (b, f), axis=1).sum()) for k in full_keys)
        if name or n_panels > 1:
            ax.set_title(f"{name} (n={n_here:,})", fontsize="medium")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if view == "2d":
            _plain_ticks(ax)
            ax.set_aspect("equal", adjustable="box")  # shared axes: the floors of a building at one scale
            ax.grid(True, alpha=0.25)
        else:
            ax.ticklabel_format(style="plain", useOffset=False)  # real coordinates, no "1e6" offset
            ax.set_zlabel(_z_label(meta, has_floor, floor_height, unit))
            if floor_height == 1.0 and has_floor:
                ax.set_zticks(sorted({int(x) for (_, t), r in zip(items, rows) for x in t.floor[r]}))

    if kind == "density":
        vmax = max(1.0, max(float(h.max()) for *_, h in grids))
        for ax, (ex, ey), hist in grids:
            mesh = ax.pcolormesh(ex, ey, np.ma.masked_equal(hist.T, 0), cmap=cmap, vmin=1.0, vmax=vmax, zorder=0.4)
            _plain_ticks(ax)  # the limits are known only now
        cell = f"{step:g} x {step:g}" + (f" {unit}" if unit else "")
        fig.colorbar(mesh, ax=axes, label=f"samples per {cell} cell", shrink=0.8)
    elif colours.categorical:
        found = {}
        for ax in axes:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                found.setdefault(lab, h)
        names = [colours.name(c) for c in colours.categories if colours.name(c) in found]
        if len(names) > 1 or color_by == "split" or asked:
            labels = [f"{n} (n={counts[n]:,})" if color_by == "split" else n for n in names]
            handles = [found[n] for n in names]
            try:  # beside the panels (matplotlib >= 3.7), so no data is hidden
                fig.legend(handles, labels, loc="outside right upper", markerscale=2.0, fontsize="small")
            except ValueError:
                axes[0].legend(handles, labels, markerscale=2.0, fontsize="small")
    else:
        fig.colorbar(plt.cm.ScalarMappable(norm=colours.norm, cmap=colours.cmap), ax=axes, label=color_by,
                     shrink=0.8)

    n_total, shown = sum(counts.values()), sum(len(r) for r in rows)  # density: every row is counted
    subtitle = f"{n_total:,} samples" + (f", {shown:,} drawn" if shown < n_total else "")
    crs = meta.get("crs")
    fig.suptitle(title if title is not None else
                 f"{meta.get('name', 'dataset')}: {subtitle}" + (f" (crs {crs})" if crs else ""))
    return fig


# ----------------------------------------------------------------------------------- interactive page
def distribution_html(tables, path=None, *, view: str = "3d", color_by: str | None = None, floors=None,
                      floor_height: float | None = None, max_points: int | None = None, random_state=0,
                      title: str | None = None, include_plotlyjs="cdn"):
    """An interactive plotly version of :func:`plot_distribution` (``plotly`` must be installed).

    ``view="3d"``: every floor stacked (z = ``floor * floor_height``), one legend entry per
    colour category (click to hide). ``view="2d"``: one floor at a time, chosen from a dropdown
    menu, as in the 0.1 page. When ``meta["crs"]`` gives each building its own frame (see
    :func:`plot_distribution`), the dropdown picks the building (3-D) or the building and floor
    (2-D), so points of different frames are never drawn together (tables in different frames,
    and floors with frames of their own in 3-D, are refused). Hovering a point shows its
    table, floor, building and coordinates. Writes ``path`` (a single HTML file) when given and
    returns the plotly Figure. ``include_plotlyjs="cdn"`` keeps the file small but needs
    internet access to view; ``True`` embeds plotly.js (about 4.5 MB) for offline viewing.

    Categorical colouring only (split, floor, building or a group with at most 20 values).
    The default test environment has no plotly; its test runs where plotly is installed.
    """
    go = requires("plotly.graph_objects", "plot")
    if view not in ("2d", "3d"):
        raise ValueError(f"view must be '2d' or '3d', not {view!r}")
    items = _labelled(tables)
    meta = _one_frame(items)
    has_floor = all(t.floor is not None for _, t in items)
    per_building = _frames_per_building(meta) and all(t.building is not None for _, t in items)
    if view == "3d" and not has_floor and any(t.pos.shape[1] < 3 for _, t in items):
        raise ValueError("the 3-D view needs floor labels or 3-D positions")
    _check_floor_filter(items, floors)
    rng = np.random.default_rng(random_state)
    color_by = color_by or _default_color_by(items, view, "building" if per_building else "none")
    plan = meta.get("floor_plan")
    if floor_height is None:
        floor_height = float((plan.get("floor_height") if isinstance(plan, Mapping) else None) or 1.0)
    xlabel, ylabel, unit = _axis_labels(meta, None)
    by_floor = view == "2d" and has_floor  # the dropdown picks a floor (2-D) and/or a building frame

    def group_of(b, f):  # the dropdown entry of a point: (building or None, floor or None), or None
        return None if not (per_building or by_floor) else (b if per_building else None, f if by_floor else None)

    parts = []  # (category, group, xy, z, hover text)
    n_total = n_drawn = 0
    drawn_floors = set()  # the floors whose walls the 3-D view draws
    for label, t in items:
        idx = np.arange(len(t)) if floors is None else np.flatnonzero(np.isin(t.floor, floors))
        r = idx[_subsample(len(idx), max_points, rng)]
        n_total, n_drawn = n_total + len(idx), n_drawn + len(r)
        if has_floor:
            drawn_floors.update(int(f) for f in np.unique(t.floor[idx]))
        v = np.asarray(_column(t, color_by, label))[r]
        fl = None if t.floor is None else t.floor[r]
        bl = None if t.building is None else t.building[r]
        if view == "3d":  # as in plot_distribution: all tables by floor label, or all by their z
            z = fl.astype(np.float64) * floor_height if has_floor else t.pos[r, 2]
        else:
            z = None
        text = np.array([f"{label}" + ("" if fl is None else f" | floor {fl[k]}") +
                         ("" if bl is None else f" | building {bl[k]}") for k in range(len(r))], dtype=object)
        values = np.unique(v).tolist() if color_by != "split" else [label]
        if len(values) > _MAX_CATEGORIES:
            raise ValueError(f"distribution_html colours by at most {_MAX_CATEGORIES} categories; "
                             f"{color_by!r} has {len(values)} in {label!r}")
        zeros = np.zeros(len(r), np.int64)
        keys = np.stack([bl if per_building else zeros, fl if by_floor else zeros], axis=1)
        for c in values:
            m = v == c
            name = str(c) if color_by == "split" else (f"{color_by} {c}" if color_by in ("floor", "building")
                                                      else f"{color_by}={c}")
            for b, f in np.unique(keys[m], axis=0).tolist():
                mk = m & (keys[:, 0] == b) & (keys[:, 1] == f)
                parts.append((name, group_of(b, f), t.pos[r[mk], :2], None if z is None else z[mk], text[mk]))

    if view == "3d" and has_floor:
        _check_floor_frames(meta, drawn_floors)
    names = list(dict.fromkeys(p[0] for p in parts))
    colour = dict(zip(names, _hex_palette(len(names))))  # one distinct colour per category
    hover = f"%{{text}}<br>{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<extra></extra>"
    traces = []  # (group the trace belongs to, or None = always shown; trace)
    for name, group, xy, z, text in parts:
        marker = {"size": 2.5 if view == "3d" else 5, "color": colour[name], "opacity": 0.75}
        if view == "3d":
            traces.append((group, go.Scatter3d(x=xy[:, 0], y=xy[:, 1], z=z, mode="markers", name=name, marker=marker,
                                               text=text, hovertemplate=hover, legendgroup=name)))
        else:
            traces.append((group, go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers", name=name, marker=marker,
                                               text=text, hovertemplate=hover, legendgroup=name)))
    groups = sorted({g for g, _ in traces if g is not None},
                    key=lambda g: tuple(-math.inf if x is None else x for x in g))
    if plan is not None:  # walls as NaN-separated polylines, one trace per floor (and per dropdown entry)
        line = {"color": "#666666", "width": 1.5}
        if view == "3d":
            wall_floors = sorted(drawn_floors)  # only the floors shown (floors= filters the walls too)
            wall_groups = groups or [None]
        else:
            wall_floors, wall_groups = [None], groups or [None]
        first = True
        for g in wall_groups:
            for f in wall_floors:
                walls = _plan_walls(plan, f if view == "3d" else (None if g is None else g[1]))
                if not len(walls):
                    continue
                xs, ys = (np.column_stack([walls[:, 0, d], walls[:, 1, d], np.full(len(walls), np.nan)]).ravel()
                          for d in (0, 1))
                if view == "3d":
                    trace = go.Scatter3d(x=xs, y=ys, z=np.full(len(xs), f * floor_height), mode="lines", line=line,
                                         name="walls", legendgroup="walls", hoverinfo="skip", showlegend=first)
                else:
                    trace = go.Scatter(x=xs, y=ys, mode="lines", line=line, name="walls", legendgroup="walls",
                                       hoverinfo="skip")
                traces.append((g, trace))
                first = False
    fig = go.Figure([trace for _, trace in traces])
    heading = title or (f"{meta.get('name', 'dataset')}: {n_total:,} samples"
                        + (f", {n_drawn:,} drawn" if n_drawn < n_total else ""))
    if view == "3d":
        fig.update_layout(scene={"xaxis_title": xlabel, "yaxis_title": ylabel,
                                 "zaxis_title": _z_label(meta, has_floor, floor_height, unit)})
    else:
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, yaxis={"scaleanchor": "x", "scaleratio": 1})
    if groups:
        def caption(g):
            where = ", ".join(([f"building {g[0]}"] if g[0] is not None else [])
                              + ([f"floor {g[1]}"] if g[1] is not None else []))
            return where, f"{heading}; {where}: {sum(len(p[2]) for p in parts if p[1] == g):,} shown"

        for (g, _), trace in zip(traces, fig.data):  # the figure holds copies of the traces
            trace.visible = g in (None, groups[0])
        buttons = [{"label": caption(g)[0], "method": "update",
                    "args": [{"visible": [h in (None, g) for h, _ in traces]}, {"title.text": caption(g)[1]}]}
                   for g in groups]
        fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "xanchor": "right",
                                        "y": 1.02, "yanchor": "bottom"}])
        heading = caption(groups[0])[1]
    fig.update_layout(title={"text": heading}, template="plotly_white", legend={"itemsizing": "constant"})
    if path is not None:
        fig.write_html(str(path), include_plotlyjs=include_plotlyjs, full_html=True)
    return fig


__all__ = ["distribution_html", "plot_distribution", "plot_floor_plan"]
