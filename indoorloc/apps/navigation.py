"""L5 navigation: shortest routes on a floor plan, across floors, with turn-by-turn instructions.

:func:`astar` finds a shortest 8-connected (or 4-connected) path on an occupancy grid; a
diagonal move may not cut the corner of a blocked cell, so a route never squeezes between two
diagonally touching blocked cells. :class:`Navigator` rasterises a
:class:`~indoorloc.apps.maps.FloorMap` (one grid per floor, walls inflated by ``clearance``)
and runs A* on the stacked grids, where stairs and elevators are extra edges between floors
costing ``wait + cost * storeys`` metres plus the planar offset of their landings. The
heuristic (octile distance plus the cheapest per-storey cost times the storeys left) is
admissible and consistent, so routes are optimal on the grid graph. The grid route is then
shortened by line-of-sight post-smoothing and described as turn-by-turn instructions.

References
----------
P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of
    Minimum Cost Paths", IEEE Transactions on Systems Science and Cybernetics 4(2):100-107,
    1968. DOI 10.1109/TSSC.1968.300136.
A. Botea, M. Muller, J. Schaeffer, "Near Optimal Hierarchical Path-Finding", Journal of Game
    Development 1(1):7-28, 2004 (A* with line-of-sight post-smoothing).
J. Amanatides, A. Woo, "A Fast Voxel Traversal Algorithm for Ray Tracing", Eurographics 1987,
    pp. 3-10 (grid traversal used by the line-of-sight test).
"""
from __future__ import annotations

import heapq
import math
from typing import NamedTuple

import numpy as np

from ..core import Estimator
from .maps import FloorMap

_SQRT2 = math.sqrt(2.0)
_MOVES8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_MOVES4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _grid_distance(dr: int, dc: int, connectivity: int) -> float:
    dr, dc = abs(dr), abs(dc)
    if connectivity == 4:
        return float(dr + dc)
    return max(dr, dc) + (_SQRT2 - 1.0) * min(dr, dc)  # octile distance


def _search(layers, start, goal, connectivity, links, storey_cost, floor_of):
    """A* over stacked grids. Nodes are ``(layer, row, col)``; ``links[node]`` lists
    ``(node2, cost)`` inter-floor edges; costs are in cells. Returns ``(path, cost)``."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    moves = _MOVES8 if connectivity == 8 else _MOVES4
    h_, w_ = layers[0].shape
    free = [(~b).ravel().tolist() for b in layers]
    gl, gr, gc = goal
    g_floor = floor_of[gl]

    def heuristic(node):
        l, r, c = node
        return _grid_distance(r - gr, c - gc, connectivity) + storey_cost * abs(floor_of[l] - g_floor)

    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    counter = 0
    h0 = heuristic(start)
    heap = [(h0, h0, counter, start)]
    while heap:
        _, _, _, node = heapq.heappop(heap)
        if node in closed:
            continue
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], g[goal]
        closed.add(node)
        l, r, c = node
        fl = free[l]
        base = g[node]
        edges = []
        for dr, dc in moves:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h_ and 0 <= cc < w_) or not fl[rr * w_ + cc]:
                continue
            if dr and dc:  # no corner cutting: both orthogonal neighbours must be free
                if not (fl[r * w_ + cc] and fl[rr * w_ + c]):
                    continue
                edges.append(((l, rr, cc), _SQRT2))
            else:
                edges.append(((l, rr, cc), 1.0))
        edges.extend(links.get(node, ()))
        for nxt, cost in edges:
            if nxt in closed:
                continue
            new = base + cost
            if new < g.get(nxt, math.inf):
                g[nxt] = new
                parent[nxt] = node
                counter += 1
                hn = heuristic(nxt)
                heapq.heappush(heap, (new + hn, hn, counter, nxt))
    return None, math.inf


def astar(blocked, start, goal, *, connectivity: int = 8) -> np.ndarray | None:
    """Shortest path on a grid: ``(K, 2)`` int ``(row, col)`` cells from ``start`` to ``goal``
    inclusive, or None if unreachable. ``blocked`` is ``(H, W)`` bool. Straight moves cost 1
    and diagonal moves sqrt(2) (never cutting a blocked corner); the octile heuristic
    (Manhattan for 4-connectivity) keeps the result optimal. Ties are broken deterministically
    (lowest f, then lowest h, then first pushed)."""
    b = np.asarray(blocked, dtype=bool)
    if b.ndim != 2:
        raise ValueError(f"blocked must be (H, W), got shape {b.shape}")
    s, t = tuple(int(v) for v in start), tuple(int(v) for v in goal)
    for name, (r, c) in (("start", s), ("goal", t)):
        if not (0 <= r < b.shape[0] and 0 <= c < b.shape[1]) or b[r, c]:
            raise ValueError(f"{name} cell {(r, c)} is outside the grid or blocked")
    path, _ = _search([b], (0, *s), (0, *t), connectivity, {}, 0.0, [0])
    return None if path is None else np.array([(r, c) for _, r, c in path], dtype=np.int64)


def path_length(points) -> float:
    """Length of a polyline ``(K, D)`` (sum of segment lengths; 0 for fewer than 2 points)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def line_of_sight(grid, p0, p1) -> bool:
    """True if the segment ``p0 -> p1`` (metric ``(x, y)``) crosses only free cells of the
    ``OccupancyGrid``. Cells are visited exactly (Amanatides-Woo traversal); passing through a
    cell corner requires both side cells to be free (no corner cutting)."""
    res, x0, y0 = float(grid.resolution), float(grid.origin[0]), float(grid.origin[1])
    blocked = grid.blocked
    h_, w_ = blocked.shape
    ax, ay = (float(p0[0]) - x0) / res, (float(p0[1]) - y0) / res
    bx, by = (float(p1[0]) - x0) / res, (float(p1[1]) - y0) / res
    c, r = math.floor(ax), math.floor(ay)
    c_end, r_end = math.floor(bx), math.floor(by)

    def bad(rr, cc):
        return not (0 <= rr < h_ and 0 <= cc < w_) or blocked[rr, cc]

    dx, dy = bx - ax, by - ay
    sc, sr = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    t_dx = abs(1.0 / dx) if dx else math.inf
    t_dy = abs(1.0 / dy) if dy else math.inf
    t_x = ((c + (sc > 0)) - ax) / dx if dx else math.inf
    t_y = ((r + (sr > 0)) - ay) / dy if dy else math.inf
    for _ in range(abs(c_end - c) + abs(r_end - r) + 2):
        if bad(r, c):
            return False
        if (c == c_end and r == r_end) or min(t_x, t_y) > 1.0:
            return True
        if abs(t_x - t_y) <= 1e-12:  # exactly through a corner
            if bad(r, c + sc) or bad(r + sr, c):
                return False
            c, r, t_x, t_y = c + sc, r + sr, t_x + t_dx, t_y + t_dy
        elif t_x < t_y:
            c, t_x = c + sc, t_x + t_dx
        else:
            r, t_y = r + sr, t_y + t_dy
    return not bad(r, c)


def smooth_path(points, grid) -> np.ndarray:
    """Greedy line-of-sight post-smoothing: from each kept vertex jump to the farthest later
    vertex it can see on ``grid``. Returns a sub-sequence of ``points`` ``(K', 2)`` (never
    longer than the input path, never through a blocked cell)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) <= 2:
        return p.copy()
    keep, i = [0], 0
    while i < len(p) - 1:
        j = i + 1
        while j + 1 < len(p) and line_of_sight(grid, p[i], p[j + 1]):
            j += 1
        keep.append(j)
        i = j
    return p[keep]


class Instruction(NamedTuple):
    """One turn-by-turn instruction.

    action    "start", "straight", "slight_left", "left", "sharp_left", ... "u_turn",
              the connector kind ("stairs", "elevator", ...) or "arrive"
    distance  metres to walk after this instruction, up to the next one
    angle     signed turn in degrees (positive = left / counter-clockwise), 0 if none
    pos       (2,) where the instruction applies
    floor     floor id (None for single-floor routes built from bare points)
    text      human-readable sentence
    """

    action: str
    distance: float
    angle: float
    pos: np.ndarray
    floor: int | None
    text: str


def _turn(angle: float, straight: float) -> str:
    a = abs(angle)
    if a < straight:
        return "straight"
    side = "left" if angle > 0 else "right"
    if a < 60:
        return f"slight_{side}"
    if a < 135:
        return side
    if a < 170:
        return f"sharp_{side}"
    return "u_turn"


def turn_instructions(points, floors=None, *, straight: float = 20.0, connectors=None) -> list[Instruction]:
    """Turn-by-turn instructions for a polyline ``(K, 2)`` (optionally with ``(K,)`` floors).

    Turns below ``straight`` degrees are merged into the current leg; others are classified
    as slight (< 60), normal (< 135), sharp (< 170) or a U-turn, left for counter-clockwise.
    A floor change between two consecutive points becomes a connector instruction
    (``connectors[i]`` names its kind for the i-th change; default "stairs").
    """
    p = np.asarray(points, dtype=np.float64)
    f = [None] * len(p) if floors is None else [int(v) for v in floors]
    rides = list(connectors or [])
    out: list[Instruction] = []
    if len(p) == 0:
        return out

    def add(action, pos, floor, angle=0.0, text=""):
        out.append(Instruction(action, 0.0, float(angle), np.asarray(pos, dtype=np.float64).copy(), floor, text))

    add("start", p[0], f[0])
    heading = None
    for k in range(len(p) - 1):
        if f[k] != f[k + 1]:
            kind = rides.pop(0) if rides else "stairs"
            add(kind, p[k], f[k], text=f"Take the {kind} from floor {f[k]} to floor {f[k + 1]}")
            heading = None
            continue
        seg = p[k + 1] - p[k]
        dist = float(np.hypot(*seg))
        if dist == 0:
            continue
        h = math.atan2(seg[1], seg[0])
        if heading is not None:
            angle = math.degrees((h - heading + math.pi) % (2 * math.pi) - math.pi)
            action = _turn(angle, straight)
            if action != "straight":
                add(action, p[k], f[k], angle)
        heading = h
        last = out[-1]
        out[-1] = last._replace(distance=last.distance + dist)
    add("arrive", p[-1], f[-1])
    return [ins._replace(text=_describe(ins)) for ins in out]


def _describe(ins: Instruction) -> str:
    walk = f", then walk {ins.distance:.1f} m" if ins.distance > 0 else ""
    if ins.text:  # connector rides carry their sentence
        return ins.text + walk
    if ins.action == "start":
        return f"Start{' on floor ' + str(ins.floor) if ins.floor is not None else ''}" + (
            f" and walk {ins.distance:.1f} m" if ins.distance > 0 else "")
    if ins.action == "arrive":
        return "Arrive at the destination"
    if ins.action == "u_turn":
        return f"Make a U-turn{walk}"
    turn = ins.action.split("_")
    adverb = {"slight": "Bear", "sharp": "Turn sharp"}.get(turn[0], "Turn")
    return f"{adverb} {turn[-1]}{walk}"


class Route(NamedTuple):
    """A route: ``points`` ``(K, 2)`` metric vertices, ``floor`` ``(K,)``, ``length`` walked in the
    plane (metres), ``cost`` the A* objective (metres, connector costs included), and the
    turn-by-turn ``instructions``."""

    points: np.ndarray
    floor: np.ndarray
    length: float
    cost: float
    instructions: list


class Navigator(Estimator):
    """Shortest walking routes on a (multi-floor) floor plan.

    ``fit(floor_map)`` rasterises each floor at ``resolution`` metres with walls inflated by
    ``clearance`` and records the connector landings; ``route(start, goal, start_floor=,
    goal_floor=)`` runs A* over the floors, post-smooths each floor's leg by line of sight
    (``smooth=True``) and returns a :class:`Route` with instructions.

    Parameters
    ----------
    resolution : float
        Grid cell size, metres.
    clearance : float
        Minimum distance kept from walls, metres (a body radius).
    connectivity : {8, 4}
    smooth : bool
        Line-of-sight post-smoothing of the grid path.
    connector_kinds : tuple of str or None
        Connector kinds allowed (e.g. ``("elevator",)`` for a step-free route); None = all.

    ``save`` stores the fitted floor plan as arrays (``FloorMap.to_dict``); ``load_model``
    re-runs the deterministic :meth:`fit` on it, so the loaded navigator gives the same routes.

    References
    ----------
    P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of
        Minimum Cost Paths", IEEE Trans. Systems Science and Cybernetics 4(2):100-107, 1968.
        DOI 10.1109/TSSC.1968.300136.
    A. Botea, M. Muller, J. Schaeffer, "Near Optimal Hierarchical Path-Finding", Journal of
        Game Development 1(1):7-28, 2004.
    """

    _requires_fit = True

    def __init__(self, resolution: float = 0.25, clearance: float = 0.2, connectivity: int = 8,
                 smooth: bool = True, connector_kinds=None):
        self.resolution = resolution
        self.clearance = clearance
        self.connectivity = connectivity
        self.smooth = smooth
        self.connector_kinds = connector_kinds

    def fit(self, floor_map, y=None):
        """Rasterise ``floor_map`` (a ``FloorMap``); ``y`` is ignored."""
        floors = tuple(floor_map.floors)
        grids = [floor_map.occupancy_grid(self.resolution, f, self.clearance) for f in floors]
        res = float(self.resolution)
        storey_costs = []
        links: dict = {}
        rides: dict = {}
        kinds = None if self.connector_kinds is None else set(self.connector_kinds)
        for ci, con in enumerate(floor_map.connectors):
            if kinds is not None and con.kind not in kinds:
                continue
            storey_costs.append(con.cost)
            cells = {}
            for fl in con.floors:
                if fl in floors:
                    g = grids[floors.index(fl)]
                    cell = g.cell(con.landing(fl))[0]
                    if not g.is_free(cell)[0]:
                        cell = g.nearest_free(con.landing(fl))[0]
                    cells[fl] = (floors.index(fl), int(cell[0]), int(cell[1]))
            for a, b in con.links():
                if a in cells and b in cells:
                    na, nb = cells[a], cells[b]
                    planar = _grid_distance(na[1] - nb[1], na[2] - nb[2], self.connectivity)
                    cost = con.link_cost(a, b) / res + planar
                    links.setdefault(na, []).append((nb, cost))
                    links.setdefault(nb, []).append((na, cost))
                    rides[(na, nb)] = rides[(nb, na)] = ci
        self.map_ = floor_map
        self.floors_ = floors
        self.grids_ = grids
        self.links_ = links
        self.rides_ = rides
        self.storey_cost_ = (min(storey_costs) / res) if storey_costs else 0.0
        return self

    _DERIVED = ("floors_", "grids_", "links_", "rides_", "storey_cost_")

    def _get_state(self) -> dict:
        state = {k: v for k, v in super()._get_state().items() if k not in self._DERIVED}
        if "map_" in state:
            state["map_"] = state["map_"].to_dict()
        return state

    def _set_state(self, state: dict) -> None:
        state = dict(state)
        saved = state.pop("map_", None)
        super()._set_state(state)
        if saved is not None:
            self.fit(FloorMap.from_dict(saved))

    def _layer(self, floor) -> int:
        if floor is None:
            if len(self.floors_) != 1:
                raise ValueError(f"the map has floors {self.floors_}: pass start_floor= and goal_floor=")
            return 0
        if int(floor) not in self.floors_:
            raise ValueError(f"floor {floor} is not in the map (floors {self.floors_})")
        return self.floors_.index(int(floor))

    def _cell(self, layer: int, xy) -> tuple[int, int, int]:
        g = self.grids_[layer]
        cell = g.cell(xy)[0]
        if not g.is_free(cell)[0]:
            cell = g.nearest_free(xy)[0]
        return layer, int(cell[0]), int(cell[1])

    def route(self, start, goal, *, start_floor=None, goal_floor=None) -> Route | None:
        """Shortest route from ``start`` to ``goal`` (metric ``(x, y)``); None if unreachable.
        A point inside a wall's clearance is snapped to the nearest free cell first."""
        self._check_fitted("grids_")
        start, goal = np.asarray(start, dtype=np.float64).reshape(2), np.asarray(goal, dtype=np.float64).reshape(2)
        s = self._cell(self._layer(start_floor), start)
        t = self._cell(self._layer(goal_floor), goal)
        path, cost = _search([g.blocked for g in self.grids_], s, t, self.connectivity, self.links_,
                             self.storey_cost_, list(self.floors_))
        if path is None:
            return None
        # legs: runs of nodes on one floor; the route starts / ends at the exact points
        legs, rides = [[path[0]]], []
        for a, b in zip(path[:-1], path[1:]):
            if a[0] != b[0]:
                rides.append(self.map_.connectors[self.rides_[(a, b)]])
                legs.append([b])
            else:
                legs[-1].append(b)
        points, floors = [], []
        for i, leg in enumerate(legs):
            g = self.grids_[leg[0][0]]
            pts = g.center([(r, c) for _, r, c in leg])
            # the exact start / goal replace the centre of their cell when that cell is free (it is
            # the path's end cell) and are joined to it otherwise (the point was snapped); when
            # both lie in one cell, that single centre is dropped and the route is [start, goal]
            lo = 1 if i == 0 and g.is_free(g.cell(start))[0] else 0
            hi = len(pts) - 1 if i == len(legs) - 1 and g.is_free(g.cell(goal))[0] else len(pts)
            pts = pts[lo:max(lo, hi)]
            if i == 0:
                pts = np.vstack([start, pts])
            if i == len(legs) - 1:
                pts = np.vstack([pts, goal])
            if self.smooth:
                pts = smooth_path(pts, g) if len(pts) > 2 else pts
            points.append(pts)
            floors.append(np.full(len(pts), self.floors_[leg[0][0]]))
        pts, fl = np.vstack(points), np.concatenate(floors)
        length = sum(path_length(p) for p in points)
        instr = turn_instructions(pts, fl if len(self.floors_) > 1 else None, connectors=[c.kind for c in rides])
        return Route(pts, fl, length, float(cost) * float(self.resolution), instr)


__all__ = ["Instruction", "Navigator", "Route", "astar", "line_of_sight", "path_length", "smooth_path",
           "turn_instructions"]
