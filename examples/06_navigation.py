"""Indoor navigation on a three-storey floor plan: A* routes, stairs vs. elevator, turn-by-turn instructions.

    python examples/06_navigation.py        # about 2 s, no download

The floor plan comes from L1: ``load_dataset("synthetic_office", n_floors=3)`` generates a
40 m x 20 m corridor office per storey (heavy outer walls, light partitions, 1 m doors) with a
staircase at the west end and a lift at the east end, and stores it as arrays in
``meta["floor_plan"]``. L5 takes it from there:

    FloorMap.from_dict(plan)      walls per floor + connectors (stairs, elevator) linking the floors
    Navigator(0.25, 0.3).fit(map) one occupancy grid per floor (0.25 m cells, walls inflated by a
                                  0.3 m body radius); A* over the stacked grids, where a connector
                                  ride costs ``wait + cost * storeys`` metres of walking-equivalent
                                  (library defaults: stairs 0 + 10 m per storey, elevator
                                  20 m + 3 m per storey), then line-of-sight smoothing
    route.instructions            turn-by-turn instructions (turn angle classes, connector rides)

Two routes between the same rooms (the south-west office of the ground floor and the
north-east office of the top floor): the cheapest one (any connector) and a step-free one
(``connector_kinds=("elevator",)``). The printed costs and lengths come from the router:
``route.cost`` is the A* objective, i.e. the grid path in metres (8-connected, before
smoothing, plus the planar offset of the connector landings) plus the connector rides, while
``route.length`` is the smoothed walk; so ``cost - rides`` exceeds ``length`` by the smoothing
gain (0.8 m on both routes here). The figure shows the walls, the cells A* may not enter
(grey), both routes and the numbered instruction points on each storey. The building is
SIMULATED (a generated floor plan), so the routes illustrate the method, not a real site.

Result (seed 0, 2026-09-29): cheapest route 47.6 m of walking, cost 68.4 m (20 m of it the
stairs, two storeys); step-free route 49.1 m of walking, cost 75.8 m (26 m of it the elevator).

Output: ``assets/figures/navigation.png`` (``--out`` to change the folder).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import indoorloc as iloc

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
# One style for every example figure: white background, readable sizes, hairline grid, and one
# categorical palette in a fixed slot order. Neighbouring slots pass colour-vision-deficiency checks,
# but only the first three stay apart as ALL pairs; a figure that shows more series at once picks
# a subset and adds a second cue (dashes or labels), as 01 and 04 do.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.0,
    "legend.fontsize": 8.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddcd8",
    "grid.linewidth": 0.6, "grid.linestyle": "-", "lines.linewidth": 1.8, "lines.markersize": 5.0,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}

RESOLUTION, CLEARANCE = 0.25, 0.3  # grid cell and body radius, metres
ROUTES = {"cheapest (any connector)": None, "step-free (elevator only)": ("elevator",)}


def corner_rooms(plan: dict) -> tuple[tuple[np.ndarray, int], tuple[np.ndarray, int]]:
    """Centres of the south-west office on the lowest floor and the north-east office on the top floor."""
    rooms = np.asarray(plan["rooms"], dtype=np.float64)
    floor = np.asarray(plan["room_floor"])
    offices = np.asarray(plan["room_kind"]) == "office"
    centre = (rooms[:, :2] + rooms[:, 2:]) / 2
    low, top = floor.min(), floor.max()
    first = np.flatnonzero(offices & (floor == low))
    last = np.flatnonzero(offices & (floor == top))
    a = first[np.argmin(centre[first].sum(axis=1))]
    b = last[np.argmax(centre[last].sum(axis=1))]
    return (centre[a], int(low)), (centre[b], int(top))


def ride_cost(fmap, route) -> float:
    """Summed ``Connector.link_cost`` of the route's floor changes (the connector part of ``route.cost``)."""
    kinds = [ins.action for ins in route.instructions if ins.action in {c.kind for c in fmap.connectors}]
    floors = np.asarray(route.floor)
    change = np.flatnonzero(floors[1:] != floors[:-1])
    if len(change) != len(kinds):
        raise RuntimeError(f"{len(change)} floor changes but {len(kinds)} connector instructions")
    total = 0.0
    for i, kind in zip(change, kinds):
        a, b = int(floors[i]), int(floors[i + 1])
        con = next(c for c in fmap.connectors if c.kind == kind and a in c.floors and b in c.floors)
        total += con.link_cost(a, b)
    return total


def figure(fmap, routes: dict, start, goal, path: Path, n_floors: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from indoorloc.datasets.plot import plot_floor_plan

    colour = dict(zip(routes, PALETTE))
    floors = list(fmap.floors)[::-1]  # top storey first
    xmin, ymin, xmax, ymax = fmap.bounds
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11.0, 2.6 * n_floors + 0.4))
        grid = fig.add_gridspec(n_floors, 2, width_ratios=[1.2, 1.0], wspace=0.08, hspace=0.3, top=0.94, bottom=0.06)
        axes = [fig.add_subplot(grid[i, 0]) for i in range(n_floors)]
        for ax, fl in zip(axes, floors):
            occ = fmap.occupancy_grid(RESOLUTION, fl, CLEARANCE)
            h, w = occ.shape
            extent = (occ.origin[0], occ.origin[0] + w * RESOLUTION, occ.origin[1], occ.origin[1] + h * RESOLUTION)
            ax.imshow(np.where(occ.blocked, 1.0, np.nan), origin="lower", extent=extent, cmap="Greys", vmin=0,
                      vmax=3, interpolation="nearest", zorder=0)
            plot_floor_plan({"walls": fmap.walls.reshape(-1, 4), "wall_floor": fmap.wall_floor}, ax, floor=fl,
                            colors=MUTED, linewidths=1.0)
            for con in fmap.connectors:
                if fl in con.floors:
                    x, y = con.landing(fl)
                    ax.plot(x, y, "s" if con.kind == "stairs" else "D", color=INK, mfc="white", ms=9, zorder=5)
                    ax.annotate(con.kind, (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                                fontsize=7.5, color=INK)
            for j, (name, route) in enumerate(routes.items()):
                on = route.floor == fl
                if not on.any():
                    continue
                pts = route.points[on]
                # the second route is drawn thinner on top, so legs both routes share stay visible
                ax.plot(pts[:, 0], pts[:, 1], color=colour[name], lw=3.2 - 1.4 * j, zorder=3 + j)
                for k, ins in enumerate(route.instructions, start=1):
                    if ins.floor == fl and ins.action not in ("start", "arrive"):
                        ax.annotate(str(k), ins.pos, xytext=(5 - 12 * j, 5), textcoords="offset points",
                                    fontsize=7.5, color=colour[name], fontweight="bold")
            if start[1] == fl:
                ax.plot(*start[0], "o", color=INK, mfc="white", ms=9, zorder=6)
                ax.annotate("start", start[0], xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)
            if goal[1] == fl:
                ax.plot(*goal[0], "*", color=INK, ms=13, zorder=6)
                ax.annotate("goal", goal[0], xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xlim(xmin - 0.5, xmax + 0.5)
            ax.set_ylim(ymin - 0.5, ymax + 0.5)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_title(f"floor {fl}", loc="left")
            ax.set_ylabel("y [m]")
        axes[-1].set_xlabel("x [m]")

        text = fig.add_subplot(grid[:, 1])
        text.axis("off")
        y = 1.0
        for name, route in routes.items():
            text.text(0.0, y, f"{name}: walk {route.length:.1f} m, cost {route.cost:.1f} m", color=colour[name],
                      fontsize=9, fontweight="bold", va="top", transform=text.transAxes)
            y -= 0.045
            for k, ins in enumerate(route.instructions, start=1):
                text.text(0.02, y, f"{k}. {ins.text}", fontsize=8.2, color=INK, va="top", transform=text.transAxes)
                y -= 0.037
            y -= 0.04
        text.text(0.0, max(y, 0.02), "Numbers on the map mark where an instruction applies.\n"
                  f"Grey cells: within {CLEARANCE} m of a wall (A* keeps out).\n"
                  "Cost = A* objective: grid path before smoothing + connector rides\n"
                  "(stairs 10 m per storey; elevator 20 m wait + 3 m per storey);\n"
                  "walk = the smoothed route.", fontsize=7.5, color=MUTED, va="top",
                  transform=text.transAxes)
        fig.suptitle(f"SyntheticOffice floor plan (simulated, seed 0, {n_floors} storeys), Navigator: A* on "
                     f"{RESOLUTION} m grids", fontsize=9, color=MUTED, y=1.0)
        fig.savefig(path)
        plt.close(fig)


def main(*, n_floors: int = 3, seed: int = 0, out: Path | str = FIGURES, verbose: bool = True) -> dict:
    """Plan both routes; return their lengths, costs and instructions."""
    t0 = time.perf_counter()
    table = iloc.load_dataset("synthetic_office", split="test", seed=seed, n_floors=n_floors, n_test=1)
    plan = table.meta["floor_plan"]
    fmap = iloc.FloorMap.from_dict(plan)
    start, goal = corner_rooms(plan)
    routes, timing = {}, {}
    for name, kinds in ROUTES.items():
        t1 = time.perf_counter()
        nav = iloc.Navigator(resolution=RESOLUTION, clearance=CLEARANCE, connector_kinds=kinds).fit(fmap)
        route = nav.route(start[0], goal[0], start_floor=start[1], goal_floor=goal[1])
        timing[name] = time.perf_counter() - t1
        if route is None:
            raise RuntimeError(f"no {name} route found")
        routes[name] = route
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "navigation.png"
    figure(fmap, routes, start, goal, path, len(fmap.floors))
    if verbose:
        print(f"SyntheticOffice seed {seed}: {len(fmap.floors)} floors, {len(fmap.walls)} wall segments, connectors "
              + ", ".join(f"{c.kind} at ({c.xy[0, 0]:.0f}, {c.xy[0, 1]:.0f})" for c in fmap.connectors))
        print(f"from ({start[0][0]:.1f}, {start[0][1]:.1f}) on floor {start[1]} to ({goal[0][0]:.1f}, "
              f"{goal[0][1]:.1f}) on floor {goal[1]}")
        for name, route in routes.items():
            rides = ride_cost(fmap, route)
            print(f"\n{name}: walk {route.length:.2f} m, cost {route.cost:.2f} m (grid path {route.cost - rides:.2f} m "
                  f"+ rides {rides:.0f} m), {len(route.points)} vertices, fit + route {timing[name] * 1000:.0f} ms")
            for k, ins in enumerate(route.instructions, start=1):
                print(f"  {k}. {ins.text}")
        print(f"\nfigure: {path}  ({time.perf_counter() - t0:.1f} s)")
    return {name: {"length": r.length, "cost": r.cost, "rides": ride_cost(fmap, r),
                   "instructions": [i.text for i in r.instructions],
                   "floors": sorted({int(f) for f in r.floor})} for name, r in routes.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--floors", type=int, default=3, help="storeys of the simulated office (default 3)")
    parser.add_argument("--out", type=Path, default=FIGURES, help="folder for the figure")
    args = parser.parse_args()
    main(n_floors=args.floors, out=args.out)
