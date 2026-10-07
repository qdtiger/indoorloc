"""Buildings for simulation: floor plans, anchor placement, reference grids, walks and IMU.

``FloorPlan``          walls (plan-view segments, per floor, per material), rooms, bounds,
                       storey height and vertical connectors (stairs / lifts); counts the walls
                       and floors a 3-D link penetrates, exactly.
``office_floor_plan``  a corridor office: one corridor along the long axis, a row of rooms on
                       either side with one door each, heavy outer walls, light partitions.
``place_anchors``      AP / anchor layouts: ``"grid"`` (even coverage, for RSSI), ``"perimeter"``
                       (corners first, good geometry for ranging) or ``"random"``.
``reference_grid``     fingerprint reference points on a regular grid, away from walls.
``random_points``      uniform test points in free space.
``random_walk``        a pedestrian walk through free space: shortest paths on a navigation
                       grid between random goals, shortened by line-of-sight pruning, with
                       circular arcs at corners, turns on the spot and lift/stair rides. The
                       result is a ``Route``: exact, continuous-time piecewise motion that can be
                       sampled at any rate, so heading, yaw rate and position always agree.
``synthesize_imu``     body-frame accelerometer and gyroscope signals consistent with a walk.

Coordinates: metres, x/y in plan, z up; storey ``f`` spans ``[f * floor_height, (f + 1) *
floor_height)``. Headings are radians counter-clockwise from +x (so a positive yaw rate is a
left turn, as a z-up gyroscope reports it).

References
----------
E. Damosso (ed.), COST Action 231 Final report, 1999, section 4.7 (walls/floors counted
    along the direct path, the geometry ``wall_crossings`` computes).
P. E. Hart, N. J. Nilsson and B. Raphael, "A formal basis for the heuristic determination of
    minimum cost paths", IEEE Trans. Systems Science and Cybernetics 4(2):100-107, 1968.
    DOI 10.1109/TSSC.1968.300136 (A* search on the navigation grid)
R. Harle, "A survey of indoor inertial positioning systems for pedestrians", IEEE
    Communications Surveys & Tutorials 15(3):1281-1293, 2013.
    DOI 10.1109/SURV.2012.121912.00075 (step-cycle accelerometer signal, step length and heading)
R. W. Bohannon, "Comfortable and maximum walking speed of adults aged 20-79 years: reference
    values and determinants", Age and Ageing 26(1):15-19, 1997. DOI 10.1093/ageing/26.1.15
    (comfortable walking speed of about 1.3-1.5 m/s)
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from .geometry import (crossing_matrix, inside_rects, intersection_params, point_segment_distance,
                       segment_segment_distance, wrap_angle)

MATERIALS = ("light", "heavy")
GRAVITY = 9.80665
IMU_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")


def _ro(a, dtype, shape=None) -> np.ndarray:
    arr = np.array(a, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    arr.flags.writeable = False
    return arr


# ----------------------------------------------------------------------------------- floor plan
@dataclass(frozen=True, eq=False)
class FloorPlan:
    """A multi-storey building in plan view.

    Parameters
    ----------
    walls : (W, 4) float64 ``[x1, y1, x2, y2]`` segments; each spans its whole storey.
    wall_floor : (W,) int64 storey of each wall.
    wall_type : (W,) int64 index into ``materials`` (e.g. 0 = light partition, 1 = heavy wall).
    bounds : (xmin, ymin, xmax, ymax) of the building footprint.
    n_floors, floor_height : storeys and storey height (m).
    rooms : (R, 4) axis-aligned rectangles ``[xmin, ymin, xmax, ymax]``; room_floor (R,);
        room_kind (R,) str (``"office"``, ``"corridor"``, ...).
    connectors : (K, 2) plan positions of vertical connectors that link every pair of
        adjacent storeys; connector_kind (K,) str (``"stairs"``, ``"lift"``).
    materials : names of the wall types.

    ``to_meta()`` returns these arrays as a plain dict (what ``SampleTable.meta["floor_plan"]``
    holds), ``FloorPlan.from_meta`` rebuilds the object.
    """

    walls: np.ndarray
    wall_floor: np.ndarray
    wall_type: np.ndarray
    bounds: np.ndarray
    n_floors: int = 1
    floor_height: float = 3.5
    rooms: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    room_floor: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    room_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U8"))
    connectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    connector_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype="<U8"))
    materials: tuple = MATERIALS

    def __post_init__(self):
        set_ = object.__setattr__
        walls = _ro(self.walls, np.float64, (-1, 4))
        set_(self, "walls", walls)
        set_(self, "wall_floor", _ro(np.broadcast_to(self.wall_floor, (len(walls),)), np.int64))
        set_(self, "wall_type", _ro(np.broadcast_to(self.wall_type, (len(walls),)), np.int64))
        set_(self, "bounds", _ro(self.bounds, np.float64, (4,)))
        rooms = _ro(self.rooms, np.float64, (-1, 4))
        set_(self, "rooms", rooms)
        set_(self, "room_floor", _ro(np.broadcast_to(self.room_floor, (len(rooms),)), np.int64))
        set_(self, "room_kind", _ro(np.broadcast_to(np.asarray(self.room_kind, dtype=str), (len(rooms),)), str))
        conn = _ro(self.connectors, np.float64, (-1, 2))
        set_(self, "connectors", conn)
        set_(self, "connector_kind", _ro(np.broadcast_to(np.asarray(self.connector_kind, dtype=str),
                                                         (len(conn),)), str))
        set_(self, "n_floors", int(self.n_floors))
        set_(self, "floor_height", float(self.floor_height))
        set_(self, "materials", tuple(self.materials))
        if self.n_floors < 1 or self.floor_height <= 0:
            raise ValueError("n_floors must be >= 1 and floor_height > 0")
        if len(walls) and (self.wall_floor.min() < 0 or self.wall_floor.max() >= self.n_floors):
            raise ValueError("wall_floor must lie in [0, n_floors)")
        if len(walls) and (self.wall_type.min() < 0 or self.wall_type.max() >= len(self.materials)):
            raise ValueError("wall_type must index materials")

    # ------------------------------------------------------------------ queries
    def walls_on(self, floor: int) -> np.ndarray:
        """(W_f, 4) walls of one storey."""
        return self.walls[self.wall_floor == int(floor)]

    def floor_of(self, z) -> np.ndarray:
        """Storey index of heights ``z`` (clipped to the building)."""
        f = np.floor(np.asarray(z, dtype=np.float64) / self.floor_height).astype(np.int64)
        return np.clip(f, 0, self.n_floors - 1)

    def height_of(self, floor, above_floor: float) -> np.ndarray:
        """Absolute height of a point ``above_floor`` metres above storey ``floor``."""
        return np.asarray(floor, dtype=np.float64) * self.floor_height + above_floor

    def crossed_walls(self, a, b) -> np.ndarray:
        """(M, W) bool: does the 3-D segment ``a -> b`` penetrate wall ``w``?

        ``a``, ``b`` are (M, 3) absolute positions (or broadcastable; (M, 2) means height 0). A
        wall of storey ``f`` counts when the plan-view segment crosses it at a point whose
        height along the link lies within storey ``f`` (walls span their storey from floor to
        ceiling), which is exact for vertical walls. Per-wall losses are then
        ``plan.crossed_walls(a, b) @ loss_db``; floors are counted by ``floor_crossings``.
        """
        a = np.atleast_2d(np.asarray(a, dtype=np.float64))
        b = np.atleast_2d(np.asarray(b, dtype=np.float64))
        a, b = np.broadcast_arrays(a, b)
        out = np.zeros((len(a), len(self.walls)), dtype=bool)
        if len(self.walls) == 0 or len(a) == 0:
            return out
        step = max(1, (1 << 20) // len(self.walls))
        for start in range(0, len(a), step):
            rows = slice(start, start + step)
            t, _ = intersection_params(a[rows], b[rows], self.walls)             # (m, W)
            hit = ~np.isnan(t)
            if a.shape[1] > 2:
                za, zb = a[rows, 2], b[rows, 2]
                zt = za[:, None] + np.nan_to_num(t) * (zb - za)[:, None]
                hit &= self.floor_of(zt) == self.wall_floor[None, :]
            else:
                hit &= self.wall_floor[None, :] == 0
            out[rows] = hit
        return out

    def wall_crossings(self, a, b) -> np.ndarray:
        """(M, n_materials) number of walls of each material the 3-D segments ``a -> b`` cross
        (the ``k_wi`` of the COST 231 multi-wall model; see ``crossed_walls``)."""
        onehot = np.eye(len(self.materials), dtype=np.int64)[self.wall_type]      # (W, n_mat)
        return self.crossed_walls(a, b).astype(np.int64) @ onehot

    def floor_crossings(self, a, b) -> np.ndarray:
        """(M,) number of floor slabs between the storeys of ``a`` and ``b`` ((M, 3) positions)."""
        a = np.atleast_2d(np.asarray(a, dtype=np.float64))
        b = np.atleast_2d(np.asarray(b, dtype=np.float64))
        if a.shape[1] < 3:
            return np.zeros(np.broadcast_shapes(a.shape, b.shape)[0], dtype=np.int64)
        return np.abs(self.floor_of(a[:, 2]) - self.floor_of(b[:, 2]))

    def nlos_mask(self, a, b) -> np.ndarray:
        """(M,) True where the direct 3-D path ``a -> b`` penetrates a wall or a floor."""
        return (self.wall_crossings(a, b).sum(axis=1) > 0) | (self.floor_crossings(a, b) > 0)

    def room_of(self, xy, floor) -> np.ndarray:
        """(M,) index into ``rooms`` of the room containing each point (first match), -1 if none."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        floor = np.broadcast_to(np.asarray(floor, dtype=np.int64), (len(xy),))
        inside = inside_rects(xy, self.rooms) & (floor[:, None] == self.room_floor[None, :])
        return np.where(inside.any(axis=1), np.argmax(inside, axis=1), -1)

    def distance_to_walls(self, xy, floor) -> np.ndarray:
        """(M,) distance from each point to the nearest wall of its storey (inf if none)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        floor = np.broadcast_to(np.asarray(floor, dtype=np.int64), (len(xy),))
        out = np.full(len(xy), np.inf)
        for f in np.unique(floor):
            sel = floor == f
            walls = self.walls_on(f)
            if len(walls):
                out[sel] = point_segment_distance(xy[sel], walls).min(axis=1)
        return out

    def is_free(self, xy, floor, margin: float = 0.0) -> np.ndarray:
        """(M,) inside the footprint and at least ``margin`` from every wall of the storey."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        x0, y0, x1, y1 = self.bounds
        inside = (xy[:, 0] >= x0 + margin) & (xy[:, 0] <= x1 - margin) & \
                 (xy[:, 1] >= y0 + margin) & (xy[:, 1] <= y1 - margin)
        return inside & (self.distance_to_walls(xy, floor) >= margin - 1e-9)

    def visible(self, a_xy, b_xy, floor: int, clearance: float = 0.0) -> np.ndarray:
        """(M,) plan-view segments that cross no wall of ``floor`` and keep ``clearance`` from them."""
        walls = self.walls_on(floor)
        a_xy, b_xy = np.broadcast_arrays(np.atleast_2d(a_xy)[:, :2], np.atleast_2d(b_xy)[:, :2])
        if len(walls) == 0:
            return np.ones(len(a_xy), dtype=bool)
        if clearance > 0:
            return segment_segment_distance(a_xy, b_xy, walls).min(axis=1) >= clearance - 1e-9
        return ~crossing_matrix(a_xy, b_xy, walls).any(axis=1)

    # -------------------------------------------------------------- conversion
    def to_meta(self) -> dict:
        """Plain arrays for ``SampleTable.meta`` (L5 map-constrained filters read these)."""
        return {"walls": self.walls, "wall_floor": self.wall_floor, "wall_type": self.wall_type,
                "materials": self.materials, "bounds": self.bounds, "n_floors": self.n_floors,
                "floor_height": self.floor_height, "rooms": self.rooms, "room_floor": self.room_floor,
                "room_kind": self.room_kind, "connectors": self.connectors, "connector_kind": self.connector_kind}

    @classmethod
    def from_meta(cls, meta) -> FloorPlan:
        keys = ("walls", "wall_floor", "wall_type", "bounds", "n_floors", "floor_height", "rooms", "room_floor",
                "room_kind", "connectors", "connector_kind", "materials")
        return cls(**{k: meta[k] for k in keys if k in meta})


def office_floor_plan(*, size=(40.0, 20.0), n_floors: int = 1, room_width: float = 4.0,
                      corridor_width: float = 2.4, door_width: float = 1.0, floor_height: float = 3.5,
                      random_state=None) -> FloorPlan:
    """A corridor office: rooms on both sides of a central corridor, one door per room.

    The footprint ``size = (width, depth)`` is split into a corridor of ``corridor_width``
    along x at mid-depth and ``round(width / room_width)`` rooms per side. Outer walls are
    heavy, partitions and corridor walls light. Each door is placed at a random offset along
    its room's corridor wall (``random_state``), independently on every storey. A staircase
    and a lift sit at the two ends of the corridor. Rooms and the corridor tile the floor, so
    every point of the footprint belongs to a room (``room_kind`` tells them apart).
    """
    width, depth = (float(v) for v in size)
    if corridor_width >= depth or door_width >= room_width or min(width, depth) <= 0:
        raise ValueError("need corridor_width < depth and door_width < room_width")
    rng = np.random.default_rng(random_state)
    n_rooms = max(1, int(round(width / room_width)))
    rw = width / n_rooms
    yc0 = (depth - corridor_width) / 2.0
    yc1 = yc0 + corridor_width
    walls, wall_floor, wall_type = [], [], []
    rooms, room_floor, room_kind = [], [], []
    slack = max(0.0, rw / 2.0 - door_width / 2.0 - 0.3)
    for f in range(n_floors):
        outer = [(0, 0, width, 0), (width, 0, width, depth), (width, depth, 0, depth), (0, depth, 0, 0)]
        inner = []
        for i in range(n_rooms):
            x0, x1 = i * rw, (i + 1) * rw
            for y in (yc0, yc1):
                xd = (x0 + x1) / 2.0 + rng.uniform(-slack, slack)
                inner += [(x0, y, xd - door_width / 2.0, y), (xd + door_width / 2.0, y, x1, y)]
            if i:
                inner += [(x0, 0.0, x0, yc0), (x0, yc1, x0, depth)]
            rooms += [(x0, 0.0, x1, yc0), (x0, yc1, x1, depth)]
            room_kind += ["office", "office"]
        rooms.append((0.0, yc0, width, yc1))
        room_kind.append("corridor")
        room_floor += [f] * (2 * n_rooms + 1)
        walls += outer + inner
        wall_floor += [f] * (len(outer) + len(inner))
        wall_type += [1] * len(outer) + [0] * len(inner)
    ym = (yc0 + yc1) / 2.0
    inset = min(1.0, width / 4.0)
    return FloorPlan(walls=np.array(walls, dtype=np.float64), wall_floor=wall_floor, wall_type=wall_type,
                     bounds=(0.0, 0.0, width, depth), n_floors=n_floors, floor_height=floor_height,
                     rooms=np.array(rooms), room_floor=room_floor, room_kind=room_kind,
                     connectors=np.array([(inset, ym), (width - inset, ym)]), connector_kind=["stairs", "lift"])


# ------------------------------------------------------------------------------ point layouts
def _nudge_free(plan: FloorPlan, xy, floor, margin: float) -> np.ndarray:
    """Move points closer than ``margin`` to a wall onto the nearest free spot of a small search ring."""
    xy = np.array(xy, dtype=np.float64)
    floor = np.broadcast_to(np.asarray(floor, dtype=np.int64), (len(xy),))
    bad = ~plan.is_free(xy, floor, margin)
    angles = np.arange(16) * (np.pi / 8)
    for i in np.flatnonzero(bad):
        for r in np.arange(0.1, 3.01, 0.1):
            cand = xy[i] + r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
            ok = plan.is_free(cand, floor[i], margin)
            if ok.any():
                xy[i] = cand[np.argmax(ok)]
                break
    return xy


def place_anchors(plan: FloorPlan, n_per_floor: int, *, layout: str = "grid", height=2.6, margin: float = 0.5,
                  random_state=None) -> tuple[np.ndarray, np.ndarray]:
    """Anchor / AP positions on every storey.

    ``layout``: ``"grid"`` spreads them over an r x c grid of cells matched to the footprint's
    aspect ratio (cell centres, jittered by up to a quarter cell); ``"perimeter"`` puts the
    first four in the corners (diagonal pairs first) and the rest along the walls at van der
    Corput fractions of the perimeter, inset by ``margin``; ``"random"`` samples free space.
    ``height`` is metres above each storey's floor (a scalar, or a sequence cycled over the
    anchors of a storey). Anchors closer than 0.3 m to a wall are nudged into free space.

    Returns ``(xyz (A, 3), floor (A,))`` with ``A = n_floors * n_per_floor``.
    """
    rng = np.random.default_rng(random_state)
    x0, y0, x1, y1 = plan.bounds
    w, d = x1 - x0, y1 - y0
    n = int(n_per_floor)
    if n < 1:
        raise ValueError("n_per_floor must be >= 1")
    if layout == "grid":
        r = max(1, int(round(np.sqrt(n * d / w))))
        c = int(np.ceil(n / r))
        cells = [(i, j) for i in range(r) for j in range(c)]
        pick = np.unique(np.round(np.linspace(0, len(cells) - 1, n)).astype(int))
        ij = np.array(cells)[pick]
        xy = np.stack([x0 + (ij[:, 1] + 0.5) * w / c, y0 + (ij[:, 0] + 0.5) * d / r], axis=1)
        xy += rng.uniform(-0.25, 0.25, xy.shape) * [w / c, d / r]
        base = [xy]
    elif layout == "perimeter":
        ix0, iy0, ix1, iy1 = x0 + margin, y0 + margin, x1 - margin, y1 - margin
        corners = np.array([(ix0, iy0), (ix1, iy1), (ix1, iy0), (ix0, iy1)])
        pts = list(corners[:n])
        lw, lh = ix1 - ix0, iy1 - iy0
        per = 2 * (lw + lh)
        k = 1
        while len(pts) < n:
            frac, denom, i = 0.0, 1.0, k            # van der Corput (base 2) fraction of the perimeter
            while i:
                denom *= 2
                frac += (i % 2) / denom
                i //= 2
            k += 1
            s = frac * per
            if s < lw:
                p = (ix0 + s, iy0)
            elif s < lw + lh:
                p = (ix1, iy0 + s - lw)
            elif s < 2 * lw + lh:
                p = (ix1 - (s - lw - lh), iy1)
            else:
                p = (ix0, iy1 - (s - 2 * lw - lh))
            if min(np.hypot(*(np.array(p) - q)) for q in pts) > 1e-6:
                pts.append(np.array(p))
        base = [np.array(pts)]
    elif layout == "random":
        base = None
    else:
        raise ValueError(f"layout must be 'grid', 'perimeter' or 'random', not {layout!r}")
    xyz, floors = [], []
    heights = np.resize(np.asarray(height, dtype=np.float64).ravel(), n)
    for f in range(plan.n_floors):
        if base is None:
            xy, _ = random_points(plan, n, margin=margin, floors=[f], random_state=rng)
        else:
            xy = _nudge_free(plan, base[0], f, min(0.3, margin))
        xyz.append(np.column_stack([xy, plan.height_of(f, heights)]))
        floors.append(np.full(n, f, dtype=np.int64))
    return np.concatenate(xyz), np.concatenate(floors)


def facing_centre(plan: FloorPlan, xy) -> np.ndarray:
    """(A,) boresight azimuths pointing from each anchor to the footprint centre (0 at the centre)."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    cx, cy = (plan.bounds[0] + plan.bounds[2]) / 2, (plan.bounds[1] + plan.bounds[3]) / 2
    dx, dy = cx - xy[:, 0], cy - xy[:, 1]
    return np.where(np.hypot(dx, dy) > 1e-9, np.arctan2(dy, dx), 0.0)


def reference_grid(plan: FloorPlan, spacing: float, *, margin: float = 0.3,
                   floors=None) -> tuple[np.ndarray, np.ndarray]:
    """Reference points on a regular grid centred in the footprint, at least ``margin`` from walls.

    Returns ``(xy (M, 2), floor (M,))``, storey-major then row-major (y, then x).
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    x0, y0, x1, y1 = plan.bounds

    def axis(lo, hi):
        n = int(np.floor((hi - lo - 2 * margin) / spacing + 1e-9)) + 1
        start = lo + (hi - lo - (n - 1) * spacing) / 2
        return start + spacing * np.arange(max(n, 1))

    gx, gy = np.meshgrid(axis(x0, x1), axis(y0, y1))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    xy, fl = [], []
    for f in (range(plan.n_floors) if floors is None else floors):
        ok = plan.is_free(grid, f, margin)
        xy.append(grid[ok])
        fl.append(np.full(int(ok.sum()), f, dtype=np.int64))
    return np.concatenate(xy), np.concatenate(fl)


def random_points(plan: FloorPlan, n: int, *, margin: float = 0.3, floors=None,
                  random_state=None) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points uniform over the free space of the chosen storeys (rejection sampling)."""
    rng = np.random.default_rng(random_state)
    floors = np.arange(plan.n_floors) if floors is None else np.asarray(floors, dtype=np.int64)
    x0, y0, x1, y1 = plan.bounds
    xy = np.zeros((0, 2))
    fl = np.zeros(0, dtype=np.int64)
    for _ in range(1000):
        if len(xy) >= n:
            break
        m = max(16, 2 * (n - len(xy)))
        cand = np.column_stack([rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)])
        cf = rng.choice(floors, m)
        ok = plan.is_free(cand, cf, margin)
        xy = np.concatenate([xy, cand[ok]])
        fl = np.concatenate([fl, cf[ok]])
    if len(xy) < n:
        raise RuntimeError("could not find enough free points; is the margin too large?")
    return xy[:n], fl[:n]


# ------------------------------------------------------------------------------- navigation
@dataclass(frozen=True, eq=False)
class NavigationGraph:
    """Free-space grid graph used to plan walks.

    nodes (V, 2) plan positions, node_floor (V,) storeys, a symmetric CSR adjacency
    (indptr (V + 1,), indices (E,), weights (E,) metres), component (V,) connected-component
    labels, and the ``margin`` every node keeps from the walls.
    """

    nodes: np.ndarray
    node_floor: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    component: np.ndarray
    margin: float


def navigation_graph(plan: FloorPlan, *, spacing: float = 0.5, margin: float = 0.25,
                     connector_cost: float = 10.0) -> NavigationGraph:
    """8-connected grid of free points whose edges cross no wall, plus connector edges.

    Nodes sit on a ``spacing`` grid offset by half a cell from the footprint corner and keep
    ``margin`` from every wall (with the defaults a 1 m door always contains a node pair that
    links the two sides). Each connector links the nodes nearest to it on adjacent storeys
    with an edge of weight ``connector_cost`` metres.
    """
    x0, y0, x1, y1 = plan.bounds
    xs = np.arange(x0 + spacing / 2, x1, spacing)
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    nx, ny = len(xs), len(ys)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    nodes, floors, src, dst, wts = [], [], [], [], []
    offset = 0
    for f in range(plan.n_floors):
        free = plan.is_free(grid, f, margin)
        ids = np.full(nx * ny, -1)
        ids[free] = offset + np.arange(int(free.sum()))
        idgrid = ids.reshape(ny, nx)
        local = grid[free]
        walls = plan.walls_on(f)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = idgrid[0:ny - dy, max(0, -dx):nx - max(0, dx)].ravel()
            b = idgrid[dy:ny, max(0, dx):nx + min(0, dx)].ravel()
            keep = (a >= 0) & (b >= 0)
            a, b = a[keep], b[keep]
            if len(walls) and len(a):
                ok = ~crossing_matrix(local[a - offset], local[b - offset], walls).any(axis=1)
                a, b = a[ok], b[ok]
            src.append(a)
            dst.append(b)
            wts.append(np.full(len(a), spacing * np.hypot(dx, dy)))
        nodes.append(local)
        floors.append(np.full(len(local), f, dtype=np.int64))
        offset += len(local)
    nodes = np.concatenate(nodes)
    node_floor = np.concatenate(floors)
    for cx, cy in plan.connectors:
        chain = []
        for f in range(plan.n_floors):
            on = np.flatnonzero(node_floor == f)
            if len(on):
                chain.append(on[np.argmin(np.hypot(nodes[on, 0] - cx, nodes[on, 1] - cy))])
        for a, b in itertools.pairwise(chain):
            src.append(np.array([a]))
            dst.append(np.array([b]))
            wts.append(np.array([float(connector_cost)]))
    src, dst, wts = (np.concatenate(v) if v else np.zeros(0) for v in (src, dst, wts))
    rows = np.concatenate([src, dst]).astype(np.int64)
    cols = np.concatenate([dst, src]).astype(np.int64)
    ww = np.concatenate([wts, wts]).astype(np.float64)
    order = np.lexsort((cols, rows))
    rows, cols, ww = rows[order], cols[order], ww[order]
    indptr = np.searchsorted(rows, np.arange(len(nodes) + 1)).astype(np.int64)
    return NavigationGraph(nodes, node_floor, indptr, cols, ww, _components(len(nodes), indptr, cols), float(margin))


def _components(n: int, indptr, indices) -> np.ndarray:
    label = np.full(n, -1, dtype=np.int64)
    ptr, idx = indptr.tolist(), indices.tolist()
    current = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        stack = [s]
        label[s] = current
        while stack:
            v = stack.pop()
            for u in idx[ptr[v]:ptr[v + 1]]:
                if label[u] < 0:
                    label[u] = current
                    stack.append(u)
        current += 1
    return label


def shortest_path(graph: NavigationGraph, start: int, goal: int) -> list[int] | None:
    """Node indices of a shortest path by A* (Hart et al. 1968) with the plan-view distance as
    heuristic (admissible: connector edges cost at least their plan length, which is ~0);
    ``None`` if the goal is unreachable. Deterministic: heap ties go to the lower node index."""
    if graph.component[start] != graph.component[goal]:
        return None
    xs, ys = graph.nodes[:, 0].tolist(), graph.nodes[:, 1].tolist()
    ptr, idx, wts = graph.indptr.tolist(), graph.indices.tolist(), graph.weights.tolist()
    gx, gy = xs[goal], ys[goal]
    best = {start: 0.0}
    parent = {start: -1}
    heap = [(0.0, start)]
    done = set()
    while heap:
        _, v = heapq.heappop(heap)
        if v == goal:
            path = [v]
            while parent[path[-1]] >= 0:
                path.append(parent[path[-1]])
            return path[::-1]
        if v in done:
            continue
        done.add(v)
        dv = best[v]
        for k in range(ptr[v], ptr[v + 1]):
            u = idx[k]
            du = dv + wts[k]
            if u not in done and du < best.get(u, np.inf) - 1e-12:
                best[u] = du
                parent[u] = v
                heapq.heappush(heap, (du + ((xs[u] - gx) ** 2 + (ys[u] - gy) ** 2) ** 0.5, u))
    return None


# ------------------------------------------------------------------------------------ routes
@dataclass(frozen=True, eq=False)
class Trajectory:
    """A route sampled at times ``t`` (T,).

    pos (T, 3) m, floor (T,), heading (T,) rad in ``[-pi, pi)``, yaw_rate (T,) rad/s,
    speed (T,) m/s, walking (T,) bool, step_phase (T,) step cycles walked so far and step
    (T,) completed steps (a step completes at each vertical-acceleration peak, at phase
    k - 0.5); step_frequency (Hz) and step_length (m) of the walker.
    """

    t: np.ndarray
    pos: np.ndarray
    floor: np.ndarray
    heading: np.ndarray
    yaw_rate: np.ndarray
    speed: np.ndarray
    walking: np.ndarray
    step_phase: np.ndarray
    step: np.ndarray
    step_frequency: float
    step_length: float


@dataclass(frozen=True, eq=False)
class Route:
    """Continuous-time pedestrian motion made of P primitives, evaluated exactly by ``sample``.

    Primitives are straight lines and circular arcs walked at constant ``speed``, turns on the
    spot and connector rides. Per primitive: kind (P,) str, t0 (P,) start time (s),
    duration (P,) s, p0 (P, 3) start position, h0 (P,) start heading (continuous, not
    wrapped), v (P,) speed along the heading, w (P,) yaw rate, drift (P, 3) extra velocity
    (connector rides) and walk_t0 (P,) walking time before the primitive. Heading and
    position are integrated in closed form, so the yaw rate integrates exactly to the heading
    and the heading is exactly the direction of motion.
    """

    kind: np.ndarray
    t0: np.ndarray
    duration: np.ndarray
    p0: np.ndarray
    h0: np.ndarray
    v: np.ndarray
    w: np.ndarray
    drift: np.ndarray
    walk_t0: np.ndarray
    speed: float
    step_frequency: float
    floor_height: float
    n_floors: int

    @property
    def total_duration(self) -> float:
        return float(self.t0[-1] + self.duration[-1]) if len(self.t0) else 0.0

    @property
    def length(self) -> float:
        """Plan-view distance walked (m)."""
        return float(np.sum(self.v * self.duration))

    @property
    def step_length(self) -> float:
        return self.speed / self.step_frequency

    def sample(self, times) -> Trajectory:
        """Exact state at ``times`` (s); after the end the walker stands still at the final pose."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        k = np.clip(np.searchsorted(self.t0, t, side="right") - 1, 0, len(self.t0) - 1)
        active = t < self.t0[k] + self.duration[k]
        tau = np.clip(t - self.t0[k], 0.0, self.duration[k])
        v, w, h0 = self.v[k], self.w[k], self.h0[k]
        h = h0 + w * tau
        turning = np.abs(w) > 1e-12
        safe_w = np.where(turning, w, 1.0)
        dx = np.where(turning, v / safe_w * (np.sin(h) - np.sin(h0)), v * tau * np.cos(h0))
        dy = np.where(turning, v / safe_w * (np.cos(h0) - np.cos(h)), v * tau * np.sin(h0))
        pos = self.p0[k] + np.column_stack([dx, dy, np.zeros_like(dx)]) + self.drift[k] * tau[:, None]
        walking = np.isin(self.kind[k], ("line", "arc")) & active & (v > 0)
        phase = self.step_frequency * (self.walk_t0[k] + np.where(np.isin(self.kind[k], ("line", "arc")), tau, 0.0))
        floor = np.clip(np.floor(pos[:, 2] / self.floor_height + 1e-9), 0, self.n_floors - 1).astype(np.int64)
        return Trajectory(t=t, pos=pos, floor=floor, heading=wrap_angle(h), yaw_rate=np.where(active, w, 0.0),
                          speed=np.where(walking, v, 0.0), walking=walking, step_phase=phase,
                          step=np.floor(phase + 0.5).astype(np.int64), step_frequency=self.step_frequency,
                          step_length=self.step_length)


def _string_pull(plan: FloorPlan, pts: np.ndarray, floor: int, clearance: float) -> np.ndarray:
    """Greedy line-of-sight pruning: from each kept point jump to the farthest point reachable
    by a straight segment that keeps ``clearance`` from the walls (the next node always is)."""
    if len(pts) <= 2:
        return pts
    out = [0]
    i = 0
    while i < len(pts) - 1:
        cand = np.arange(i + 1, len(pts))
        ok = plan.visible(np.repeat(pts[i:i + 1], len(cand), axis=0), pts[cand], floor, clearance)
        ok[0] = True
        i = int(cand[np.flatnonzero(ok)[-1]])
        out.append(i)
    return pts[out]


def _heading(vec) -> float:
    return float(np.arctan2(vec[1], vec[0]))


def _arc_points(t1, h_in, kappa, length, n=9) -> np.ndarray:
    s = np.linspace(0.0, length, n)
    h = h_in + kappa * s
    return t1 + np.column_stack([(np.sin(h) - np.sin(h_in)) / kappa, (np.cos(h_in) - np.cos(h)) / kappa])


def _floor_primitives(plan, pts, floor, heading, *, speed, turn_radius, spin_rate, spin_threshold,
                      clearance, safe_radius):
    """Primitives ``(kind, p0_xy, h0, v, w, duration)`` along one storey's polyline.

    ``heading`` is the walker's current (continuous) heading; the walk first turns on the spot
    to the first segment if needed. Returns (primitives, final xy, final heading).
    """
    keep = [0]
    for i in range(1, len(pts)):
        if np.hypot(*(pts[i] - pts[keep[-1]])) > 1e-9:
            keep.append(i)
    pts = np.asarray(pts, dtype=np.float64)[keep]
    prims = []
    pos, h = pts[0].copy(), float(heading)
    if len(pts) < 2:
        return prims, pos, h

    def turn_to(target):
        nonlocal h
        dh = float(wrap_angle(target - h))
        if abs(dh) > 1e-9:
            prims.append(("spin", pos.copy(), h, 0.0, np.sign(dh) * spin_rate, abs(dh) / spin_rate))
            h += dh

    turn_to(_heading(pts[1] - pts[0]))
    for i in range(1, len(pts)):
        u_in = (pts[i] - pts[i - 1]) / np.hypot(*(pts[i] - pts[i - 1]))
        end, corner = pts[i], None
        if i < len(pts) - 1:
            out = pts[i + 1] - pts[i]
            h_out = _heading(out)
            dh = float(wrap_angle(h_out - _heading(u_in)))
            if 1e-9 < abs(dh) <= spin_threshold:
                half = np.tan(abs(dh) / 2.0)
                t_len = min(turn_radius * half, float(np.hypot(*(pts[i] - pos))), 0.45 * float(np.hypot(*out)))
                while True:
                    radius = t_len / half
                    t1 = pts[i] - t_len * u_in
                    if t_len <= safe_radius:
                        break  # the arc lies inside the wall-free disc of radius safe_radius around the node
                    arc = _arc_points(t1, h, np.sign(dh) / radius, radius * abs(dh))
                    if plan.is_free(arc, floor, 0.5 * clearance).all() and plan.visible(arc[:-1], arc[1:], floor).all():
                        break
                    t_len = max(0.5 * t_len, 0.95 * safe_radius)
                end, corner = t1, (t_len, radius, dh, h_out)
        length = float(np.hypot(*(end - pos)))
        if length > 1e-9:
            prims.append(("line", pos.copy(), h, speed, 0.0, length / speed))
            pos = end.copy()
        if corner is not None:
            t_len, radius, dh, h_out = corner
            prims.append(("arc", pos.copy(), h, speed, speed * np.sign(dh) / radius, radius * abs(dh) / speed))
            h += dh
            pos = pts[i] + t_len * np.array([np.cos(h_out), np.sin(h_out)])
        elif i < len(pts) - 1:
            pos = pts[i].copy()
            turn_to(_heading(pts[i + 1] - pts[i]))
    return prims, pos, h


def random_walk(plan: FloorPlan, duration: float, *, speed=None, step_frequency=None, start=None,
                graph: NavigationGraph | None = None, device_height: float = 1.2, change_floor_prob: float = 0.25,
                connector_time: float = 8.0, turn_radius: float = 1.0, spin_rate: float = 2.0,
                spin_threshold: float = np.deg2rad(120.0), clearance: float = 0.35, min_goal_distance: float = 5.0,
                random_state=None) -> Route:
    """A pedestrian walking between random goals for at least ``duration`` seconds.

    The walker repeatedly picks a goal node of the navigation graph (on another storey with
    probability ``change_floor_prob``, otherwise at least ``min_goal_distance`` m away),
    follows the A* path, prunes it to line-of-sight waypoints that keep ``clearance`` from
    walls, and rounds each corner with a circular arc (tangent length up to
    ``turn_radius * tan(turn / 2)``, shrunk until the arc stays clear of walls); corners
    sharper than ``spin_threshold`` are taken by turning on the spot at ``spin_rate`` rad/s.
    Storey changes ride a connector for ``connector_time`` s per storey (no steps, height
    changing linearly). The device is carried ``device_height`` m above the floor.

    ``speed`` (m/s) and ``step_frequency`` (Hz) default to draws from N(1.3, 0.1) and
    N(1.8, 0.1) (clipped to [0.8, 1.8] and [1.4, 2.3]): a step length of about 0.72 m.
    ``start`` is ``(x, y, floor)``; by default a random node of the largest component.
    """
    rng = np.random.default_rng(random_state)
    graph = navigation_graph(plan) if graph is None else graph
    speed = float(np.clip(rng.normal(1.3, 0.1), 0.8, 1.8)) if speed is None else float(speed)
    step_frequency = (float(np.clip(rng.normal(1.8, 0.1), 1.4, 2.3)) if step_frequency is None
                      else float(step_frequency))
    main = np.flatnonzero(graph.component == np.argmax(np.bincount(graph.component)))
    if start is None:
        current = int(rng.choice(main))
    else:
        sx, sy, sf = start
        on = main[graph.node_floor[main] == int(sf)]
        current = int(on[np.argmin(np.hypot(graph.nodes[on, 0] - sx, graph.nodes[on, 1] - sy))])
    floors = np.unique(graph.node_floor[main])
    runs, covered = [], 0.0          # (storey, waypoints) runs; each leg is pruned on its own
    need = 1.25 * duration * speed + 10.0  # pruning shortens grid paths; spins and rides take time
    for _ in range(1000):
        if covered >= need:
            break
        f = graph.node_floor[current]
        if len(floors) > 1 and rng.random() < change_floor_prob:
            pool = main[graph.node_floor[main] == rng.choice(floors[floors != f])]
        else:
            pool = main[graph.node_floor[main] == f]
            far = np.hypot(*(graph.nodes[pool] - graph.nodes[current]).T) >= min_goal_distance
            pool = pool[far] if far.any() else pool
        goal = int(rng.choice(pool))
        path = shortest_path(graph, current, goal)
        if path is None or len(path) < 2:
            continue
        path = np.asarray(path)
        cuts = np.flatnonzero(np.diff(graph.node_floor[path]) != 0) + 1
        for piece in np.split(path, cuts):
            fl = int(graph.node_floor[piece[0]])
            pts = _string_pull(plan, graph.nodes[piece], fl, clearance)
            covered += float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
            if runs and runs[-1][0] == fl:
                runs[-1][1].append(pts[1:])
            else:
                runs.append((fl, [pts]))
        current = goal
    heading = None
    prims = []
    for r, (f, parts) in enumerate(runs):
        pts = np.concatenate(parts)
        if heading is None:
            moving = np.flatnonzero(np.hypot(*np.diff(pts, axis=0).T) > 1e-9)
            heading = _heading(pts[moving[0] + 1] - pts[moving[0]]) if len(moving) else 0.0
        z = float(plan.height_of(f, device_height))
        fp, pos, heading = _floor_primitives(plan, pts, f, heading, speed=speed, turn_radius=turn_radius,
                                             spin_rate=spin_rate, spin_threshold=spin_threshold,
                                             clearance=clearance, safe_radius=0.95 * graph.margin)
        pos = pts[-1] if not fp else pos
        prims += [(kind, np.r_[p, z], h0, v, w, T, np.zeros(3)) for kind, p, h0, v, w, T in fp]
        if r + 1 < len(runs):  # ride the connector to the next storey
            g, nxt = runs[r + 1][0], runs[r + 1][1][0][0]
            T = connector_time * abs(g - f)
            here, there = np.r_[pos, z], np.r_[nxt, plan.height_of(g, device_height)]
            prims.append(("lift", here, heading, 0.0, 0.0, T, (there - here) / T))
    heading = 0.0 if heading is None else heading
    if not prims:  # nowhere to go: stand still
        p = np.r_[graph.nodes[current], plan.height_of(graph.node_floor[current], device_height)]
        prims = [("spin", p, heading, 0.0, 0.0, float(duration), np.zeros(3))]
    kind = np.array([p[0] for p in prims])
    dur = np.array([p[5] for p in prims], dtype=np.float64)
    walking = np.isin(kind, ("line", "arc"))
    return Route(kind=kind, t0=np.concatenate([[0.0], np.cumsum(dur)[:-1]]), duration=dur,
                 p0=np.array([p[1] for p in prims], dtype=np.float64), h0=np.array([p[2] for p in prims]),
                 v=np.array([p[3] for p in prims]), w=np.array([p[4] for p in prims]),
                 drift=np.array([p[6] for p in prims], dtype=np.float64),
                 walk_t0=np.concatenate([[0.0], np.cumsum(np.where(walking, dur, 0.0))[:-1]]),
                 speed=speed, step_frequency=step_frequency, floor_height=plan.floor_height, n_floors=plan.n_floors)


# ----------------------------------------------------------------------------------------- IMU
def synthesize_imu(traj: Trajectory, *, step_amplitude: float = 2.0, gravity: float = GRAVITY,
                   acc_noise_std: float = 0.05, gyro_noise_std: float = 0.005, gyro_bias_std: float = 0.002,
                   random_state=None) -> np.ndarray:
    """(T, 6) body-frame IMU samples ``[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]`` (m/s^2, rad/s).

    The phone is held flat and points along the walking direction (x forward, y left, z up).
    While walking, the vertical specific force follows the step cycle,
    ``a_z = g - A cos(2 pi phase)`` (one peak per step, at phase k - 0.5), with a forward
    oscillation ``0.3 A sin(2 pi phase)`` and a lateral sway ``0.15 A sin(pi phase)`` at half
    the step frequency; turning adds the centripetal term ``speed * yaw_rate`` on y. The z
    gyroscope reads the yaw rate. Sensors add white noise and a constant gyroscope bias per
    call (``N(0, gyro_bias_std)`` per axis). With the noise terms at 0, integrating ``gyr_z``
    reproduces the heading and counting ``a_z`` peaks reproduces ``traj.step``: the signals
    are consistent with the path by construction, a sinusoidal idealisation of the gait
    cycle described in Harle (2013).
    """
    rng = np.random.default_rng(random_state)
    n = len(traj.t)
    walk = traj.walking.astype(np.float64)
    ph = 2.0 * np.pi * traj.step_phase
    a = step_amplitude
    imu = np.empty((n, 6))
    imu[:, 0] = walk * 0.3 * a * np.sin(ph)
    imu[:, 1] = traj.speed * traj.yaw_rate + walk * 0.15 * a * np.sin(ph / 2.0)
    imu[:, 2] = gravity - walk * a * np.cos(ph)
    imu[:, 3:5] = 0.0
    imu[:, 5] = traj.yaw_rate
    if acc_noise_std > 0:
        imu[:, :3] += rng.normal(0.0, acc_noise_std, (n, 3))
    if gyro_noise_std > 0:
        imu[:, 3:] += rng.normal(0.0, gyro_noise_std, (n, 3))
    if gyro_bias_std > 0:
        imu[:, 3:] += rng.normal(0.0, gyro_bias_std, 3)
    return imu


__all__ = ["GRAVITY", "IMU_CHANNELS", "MATERIALS", "FloorPlan", "NavigationGraph", "Route", "Trajectory",
           "facing_centre", "navigation_graph", "office_floor_plan", "place_anchors", "random_points",
           "random_walk", "reference_grid", "shortest_path", "synthesize_imu"]
