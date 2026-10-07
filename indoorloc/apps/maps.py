"""Floor plans for L5: wall segments, bounds, floors and the stairs/elevators between them.

A :class:`FloorMap` is data, not a model. It stores walls as line segments ``(M, 2, 2)`` in
the dataset frame (metres, ``x`` right and ``y`` up, the frame of ``SampleTable.pos``), the
floor of every wall, the walkable bounds and the vertical :class:`Connector` s. Three queries
are vectorised in numpy because the other L5 modules call them in their inner loops:

* :meth:`FloorMap.crosses` -- does a move ``p0 -> p1`` cross a wall? (particle filters: a
  particle whose move crosses a wall dies),
* :meth:`FloorMap.distance_to_walls` -- how far is a point from the nearest wall?
* :meth:`FloorMap.occupancy_grid` -- which cells of a regular grid are blocked (A* routing).

Segment intersection uses the orientation (cross-product) test with closed segments, so a
move that touches a wall counts as crossing it. Grid rasterisation marks a cell blocked when
its closed square intersects a wall (a separating-axis test), which makes the blocked cells
of any wall a 4-connected chain: an 8-connected path that may not cut corners cannot slip
through a diagonal wall.

References
----------
T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, "Introduction to Algorithms", 3rd ed.,
    MIT Press, 2009, Section 33.1 (segment intersection by orientation tests).
C. Ericson, "Real-Time Collision Detection", Morgan Kaufmann, 2005, Section 5.2.9 and 5.1.2
    (segment/box separating-axis test; closest point on a segment).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

_CHUNK = 1 << 20  # elements per broadcast temporary: bounds memory to a few MB per array


def _ro(a: np.ndarray) -> np.ndarray:
    a = a.view()
    a.flags.writeable = False
    return a


def _as_points(points, name: str = "points") -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 1:
        p = p[None]
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"{name} must be (N, 2) planar coordinates, got shape {p.shape}")
    return p


def _as_segments(segments, name: str = "segments") -> np.ndarray:
    s = np.asarray(segments, dtype=np.float64)
    if s.size == 0:
        return np.zeros((0, 2, 2))
    if s.ndim == 2 and s.shape[1] == 4:
        s = s.reshape(-1, 2, 2)
    if s.ndim != 3 or s.shape[1:] != (2, 2):
        raise ValueError(f"{name} must be (M, 2, 2) as [[x0, y0], [x1, y1]] per row (or (M, 4)), got {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{name} contain NaN or inf")
    return s


def _cross(ox, oy, ax, ay, bx, by):
    """z-component of (a - o) x (b - o): > 0 if o -> a -> b turns left."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def segments_intersect(p, q) -> np.ndarray:
    """``(N, M)`` bool: closed segment ``p[i]`` intersects closed segment ``q[j]``.

    ``p`` is ``(N, 2, 2)`` and ``q`` is ``(M, 2, 2)``, rows ``[[x0, y0], [x1, y1]]``. Two
    segments intersect iff each one's end points are not strictly on the same side of the
    other's supporting line and their bounding boxes overlap; the box test settles the
    collinear and zero-length cases. Touching (an end point on the other segment) counts.
    """
    p, q = _as_segments(p, "p"), _as_segments(q, "q")
    ax, ay, bx, by = (p[:, i, j][:, None] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    cx, cy, dx, dy = (q[:, i, j][None, :] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    o1 = np.sign(_cross(cx, cy, dx, dy, ax, ay))  # side of p0 w.r.t. line q
    o2 = np.sign(_cross(cx, cy, dx, dy, bx, by))
    o3 = np.sign(_cross(ax, ay, bx, by, cx, cy))  # side of q0 w.r.t. line p
    o4 = np.sign(_cross(ax, ay, bx, by, dx, dy))
    boxes = ((np.minimum(ax, bx) <= np.maximum(cx, dx)) & (np.minimum(cx, dx) <= np.maximum(ax, bx))
             & (np.minimum(ay, by) <= np.maximum(cy, dy)) & (np.minimum(cy, dy) <= np.maximum(ay, by)))
    return (o1 * o2 <= 0) & (o3 * o4 <= 0) & boxes


def point_segment_distance(points, segments) -> np.ndarray:
    """``(N, M)`` Euclidean distance from each point ``(N, 2)`` to each closed segment ``(M, 2, 2)``."""
    p, s = _as_points(points), _as_segments(segments)
    a, d = s[:, 0], s[:, 1] - s[:, 0]  # (M, 2)
    dd = np.einsum("ij,ij->i", d, d)
    rel = p[:, None, :] - a[None]  # (N, M, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(dd > 0, np.einsum("nmk,mk->nm", rel, d) / dd, 0.0)
    u = np.clip(u, 0.0, 1.0)
    diff = rel - u[..., None] * d[None]
    return np.sqrt(np.einsum("nmk,nmk->nm", diff, diff))


def _segment_hits_boxes(a, b, lo, hi) -> np.ndarray:
    """Closed segment ``a -> b`` against closed axis-aligned boxes ``lo``/``hi`` (K, 2)."""
    overlap = np.all((np.minimum(a, b) <= hi) & (np.maximum(a, b) >= lo), axis=1)
    d = b - a
    s = [d[0] * (y - a[1]) - d[1] * (x - a[0])
         for x, y in ((lo[:, 0], lo[:, 1]), (hi[:, 0], lo[:, 1]), (lo[:, 0], hi[:, 1]), (hi[:, 0], hi[:, 1]))]
    s = np.stack(s)
    return overlap & (s.min(axis=0) <= 0) & (s.max(axis=0) >= 0)


@dataclass(frozen=True, eq=False)
class Connector:
    """A vertical link between floors: stairs, an elevator, an escalator or a ramp.

    Parameters
    ----------
    kind : str
        ``"stairs"``, ``"elevator"``, ``"escalator"`` or ``"ramp"`` (navigation can exclude
        kinds, e.g. elevators only for step-free routes). Elevators link every pair of the
        floors they serve; the other kinds link consecutive floors only.
    floors : tuple of int
        Floors served, e.g. ``(0, 1, 2)`` (stored sorted).
    xy : array-like
        ``(2,)`` landing shared by every floor (a shaft), or ``(len(floors), 2)`` one landing
        per floor in the order of the sorted ``floors``.
    cost : float or None
        Cost per storey (``|floor_a - floor_b|``), in metres of walking-equivalent. Default:
        10 for stairs, escalators and ramps, 3 for elevators.
    wait : float
        Fixed cost of one use (e.g. waiting for an elevator), same units. Default: 0 for
        stairs, 20 for elevators.
    name : str
        Free-text label used in turn-by-turn instructions.
    """

    kind: str
    floors: tuple
    xy: np.ndarray
    cost: float | None = None
    wait: float | None = None
    name: str = ""
    _defaults = {"stairs": (10.0, 0.0), "elevator": (3.0, 20.0), "escalator": (10.0, 0.0), "ramp": (10.0, 0.0)}

    def __post_init__(self):
        if self.kind not in self._defaults:  # a typo ("lift") would silently behave like stairs
            raise ValueError(f"connector kind must be one of {sorted(self._defaults)}, got {self.kind!r}")
        floors = tuple(sorted(int(f) for f in self.floors))
        if len(floors) < 2 or len(set(floors)) != len(floors):
            raise ValueError(f"a connector serves at least two distinct floors, got {self.floors}")
        xy = np.array(self.xy, dtype=np.float64)  # a copy: the caller's array cannot change the connector
        if xy.shape == (2,):
            xy = np.tile(xy, (len(floors), 1))
        if xy.shape != (len(floors), 2) or not np.all(np.isfinite(xy)):
            raise ValueError(f"xy must be (2,) or ({len(floors)}, 2) finite landings, got shape {xy.shape}")
        cost, wait = self._defaults[self.kind]
        set_ = object.__setattr__
        set_(self, "floors", floors)
        set_(self, "xy", _ro(xy))
        set_(self, "cost", float(cost if self.cost is None else self.cost))
        set_(self, "wait", float(wait if self.wait is None else self.wait))
        if self.cost < 0 or self.wait < 0:
            raise ValueError("connector cost and wait must be >= 0")

    def __copy__(self):
        return self  # immutable: copies (and clones of the estimators holding it) share it

    def __deepcopy__(self, memo):
        return self

    def landing(self, floor: int) -> np.ndarray:
        """``(2,)`` landing position on ``floor``."""
        return self.xy[self.floors.index(int(floor))]

    def links(self) -> list[tuple[int, int]]:
        """Floor pairs ``(a, b)`` with ``a < b`` directly linked by this connector."""
        f = self.floors
        if self.kind == "elevator":
            return [(f[i], f[j]) for i in range(len(f)) for j in range(i + 1, len(f))]
        return list(zip(f[:-1], f[1:]))

    def link_cost(self, a: int, b: int) -> float:
        """Cost of riding from floor ``a`` to ``b``: ``wait + cost * |a - b|`` (floor ids count
        storeys; walking between the two landings is not included)."""
        if int(a) not in self.floors or int(b) not in self.floors:
            raise ValueError(f"{self.kind} {self.name!r} does not serve floors {a} and {b}")
        return self.wait + self.cost * abs(int(a) - int(b))


@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    """A rasterised floor: ``blocked[r, c]`` is True where a wall (plus clearance) lies.

    Row ``r`` indexes ``y`` and column ``c`` indexes ``x``; cell ``(r, c)`` covers
    ``[x0 + c*res, x0 + (c+1)*res] x [y0 + r*res, y0 + (r+1)*res]`` with ``origin = (x0, y0)``.
    """

    blocked: np.ndarray
    origin: np.ndarray
    resolution: float
    floor: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "blocked", _ro(np.asarray(self.blocked, dtype=bool)))
        object.__setattr__(self, "origin", _ro(np.asarray(self.origin, dtype=np.float64).reshape(2)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.blocked.shape

    def cell(self, xy) -> np.ndarray:
        """``(N, 2)`` int ``(row, col)`` of the cells containing the points (may be outside the grid)."""
        p = _as_points(xy)
        rc = np.floor((p - self.origin) / self.resolution).astype(np.int64)
        return rc[:, ::-1].copy()

    def center(self, cells) -> np.ndarray:
        """``(N, 2)`` metric ``(x, y)`` centres of ``(row, col)`` cells."""
        rc = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
        return self.origin + (rc[:, ::-1] + 0.5) * self.resolution

    def inside(self, cells) -> np.ndarray:
        rc = np.asarray(cells).reshape(-1, 2)
        h, w = self.shape
        return (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)

    def is_free(self, cells) -> np.ndarray:
        """True for cells inside the grid and not blocked."""
        rc = np.asarray(cells).reshape(-1, 2)
        ok = self.inside(rc)
        out = np.zeros(len(rc), dtype=bool)
        out[ok] = ~self.blocked[rc[ok, 0], rc[ok, 1]]
        return out

    def nearest_free(self, xy) -> np.ndarray:
        """``(N, 2)`` ``(row, col)`` of the free cell whose centre is nearest to each point."""
        free = np.argwhere(~self.blocked)
        if len(free) == 0:
            raise ValueError("the grid has no free cell")
        centers = self.center(free)
        p = _as_points(xy)
        out = np.empty((len(p), 2), dtype=np.int64)
        for i, q in enumerate(p):  # few queries (start / goal / landings); argmin is ties-by-index
            d = np.einsum("ij,ij->i", centers - q, centers - q)
            out[i] = free[np.argmin(d)]
        return out


class FloorMap:
    """Walls, bounds and floor connectors of a building, in the dataset frame (metres).

    Parameters
    ----------
    walls : array-like or mapping, optional
        Wall segments ``(M, 2, 2)`` (rows ``[[x0, y0], [x1, y1]]``; ``(M, 4)`` is accepted), or
        a mapping ``{floor: (M_f, 2, 2)}``. A door is a gap between two segments.
    floor : array-like of int, optional
        ``(M,)`` floor of each wall when ``walls`` is an array (default: every wall on floor 0).
    bounds : (xmin, ymin, xmax, ymax), optional
        Walkable region (moves ending outside it are invalid). Default: the bounding box of
        the walls and connector landings; no bounds at all for a map without walls.
    connectors : iterable of Connector
        Stairs, elevators, ... linking the floors.

    A FloorMap is immutable and validated on construction (it copies its inputs, and its
    arrays are read-only), so copies and clones of the estimators holding it share one map; it
    is data, so it is not an Estimator. ``FloorMap.from_polygons`` builds one from room
    outlines. :meth:`to_dict` / :meth:`from_dict` convert it to plain arrays and back exactly;
    ``core.persistence`` uses them (the class opts in with ``_save_via_dict = True``), so a
    ``ParticleFilter`` or ``PDRFusion`` holding a map saves and loads without pickle.

    References
    ----------
    T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, "Introduction to Algorithms", 3rd ed.,
        MIT Press, 2009, Sec. 33.1 (segment intersection, used by :meth:`crosses`).
    C. Ericson, "Real-Time Collision Detection", Morgan Kaufmann, 2005, Sec. 5.1.2 and 5.2.9
        (closest point on a segment; segment/box test, used by :meth:`occupancy_grid`).
    """

    _save_via_dict = True  # core.persistence stores to_dict() and rebuilds with from_dict() (no pickle)

    def __init__(self, walls=None, *, floor=None, bounds=None, connectors: Iterable[Connector] = ()):
        if isinstance(walls, Mapping):
            if floor is not None:
                raise ValueError("floor= gives the floor of each wall of an array; a {floor: walls} mapping "
                                 "already says it")
            parts = [(int(f), _as_segments(w, f"walls[{f!r}]")) for f, w in walls.items()]
            segs = np.concatenate([s for _, s in parts]) if parts else np.zeros((0, 2, 2))
            floors = (np.concatenate([np.full(len(s), f, np.int64) for f, s in parts]) if parts
                      else np.zeros(0, np.int64))
        else:
            segs = _as_segments(np.zeros((0, 2, 2)) if walls is None else walls, "walls")
            floors = np.zeros(len(segs), np.int64) if floor is None else np.asarray(floor)
            if floors.shape != (len(segs),) or (floors.dtype.kind not in "iu" and floors.size):
                raise ValueError(f"floor must be ({len(segs)},) integers, got {floors.dtype} {floors.shape}")
            floors = floors.astype(np.int64)
        self._connectors = tuple(connectors)
        if not all(isinstance(c, Connector) for c in self._connectors):
            raise TypeError("connectors must be Connector instances")
        if bounds is None and (len(segs) or self._connectors):
            pts = np.concatenate([segs.reshape(-1, 2)] + [c.xy for c in self._connectors])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            if not np.all(lo < hi):  # e.g. one straight wall: its box has no area to walk in
                raise ValueError(f"the walls and landings span no area (x {lo[0]:g}..{hi[0]:g}, y {lo[1]:g}.."
                                 f"{hi[1]:g}), so they give no default bounds; pass bounds=(xmin, ymin, xmax, ymax)")
            bounds = (*lo, *hi)
        if bounds is not None:
            bounds = np.array(bounds, dtype=np.float64).reshape(-1)
            if bounds.shape != (4,) or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
                raise ValueError(f"bounds must be (xmin, ymin, xmax, ymax) with min < max, got {bounds}")
            bounds = _ro(bounds)
        self._walls, self._wall_floor, self._bounds = _ro(np.array(segs)), _ro(np.array(floors)), bounds
        self._floors = tuple(sorted({*map(int, np.unique(floors)), *(f for c in self._connectors for f in c.floors)}
                                    or {0}))

    # ---------------------------------------------------------------- construction helpers
    @classmethod
    def from_polygons(cls, polygons, *, floor: int = 0, closed: bool = True, bounds=None, connectors=()):
        """Walls from outlines: ``polygons`` is a list of ``(K, 2)`` vertex arrays (a ring is
        closed back to its first vertex when ``closed``), or ``{floor: [polygons]}``."""
        def ring(poly):
            v = np.asarray(poly, dtype=np.float64)
            if v.ndim != 2 or v.shape[1] != 2 or len(v) < 2:
                raise ValueError(f"a polygon is (K>=2, 2) vertices, got {v.shape}")
            if closed and not np.array_equal(v[0], v[-1]):
                v = np.vstack([v, v[:1]])
            return np.stack([v[:-1], v[1:]], axis=1)

        if isinstance(polygons, Mapping):
            walls = {f: np.concatenate([ring(p) for p in polys]) for f, polys in polygons.items()}
            return cls(walls, bounds=bounds, connectors=connectors)
        segs = np.concatenate([ring(p) for p in polygons]) if len(polygons) else np.zeros((0, 2, 2))
        return cls(segs, floor=np.full(len(segs), floor), bounds=bounds, connectors=connectors)

    def __copy__(self):
        return self  # immutable: sharing is safe and keeps clones of a big map cheap

    def __deepcopy__(self, memo):
        return self

    def to_dict(self) -> dict:
        """Plain arrays, numbers and strings (what ``core.persistence`` stores without pickle):
        ``walls`` (M, 2, 2) float64, ``wall_floor`` (M,) int64, ``bounds`` (4,) float64 or None
        and ``connectors``, a list of dicts (``xy`` (F, 2) float64). :meth:`from_dict` rebuilds a
        map with identical arrays."""
        return {"walls": np.array(self._walls), "wall_floor": np.array(self._wall_floor),
                "bounds": None if self._bounds is None else np.array(self._bounds),
                "connectors": [{"kind": c.kind, "floors": list(c.floors), "xy": np.array(c.xy), "cost": c.cost,
                                "wait": c.wait, "name": c.name} for c in self._connectors]}

    @classmethod
    def from_dict(cls, state: Mapping) -> FloorMap:
        """Inverse of :meth:`to_dict`; also reads a dataset's ``meta["floor_plan"]``.

        A dataset plan (CONTRACTS section 2) has ``walls`` (W, 4), ``wall_floor``, ``bounds``
        and, if it knows them, ``connectors`` as a ``(K, 2)`` array of landing positions with
        ``connector_kind`` (``"stairs"``, ``"lift"``, ...) and ``n_floors``: each one becomes a
        :class:`Connector` serving every storey (``range(n_floors)``, else the floors that
        have walls; ``"lift"`` is an ``"elevator"``, default costs). Other keys (``rooms``,
        ``materials``, ``floor_height``, ...) are ignored. A plan with one storey gets no
        connectors (they would link nothing).
        """
        raw = state.get("connectors")
        raw = () if raw is None else raw
        walls = state.get("walls")
        walls = np.zeros((0, 2, 2)) if walls is None else np.asarray(walls, dtype=np.float64)
        floor = state.get("wall_floor")
        floor = None if floor is None else np.asarray(floor, dtype=np.int64).reshape(-1)
        if all(isinstance(c, Mapping) for c in raw):  # to_dict: one dict per Connector (or none)
            connectors = [Connector(c["kind"], tuple(c["floors"]), c["xy"], c.get("cost"), c.get("wait"),
                                    c.get("name", "")) for c in raw]
        else:  # a dataset plan: (K, 2) landing positions (an array or a list of pairs)
            connectors = cls._plan_connectors(raw, state, floor)
        return cls(walls, floor=floor, bounds=state.get("bounds"), connectors=connectors)

    _PLAN_KINDS = {"stairs": "stairs", "stair": "stairs", "staircase": "stairs", "lift": "elevator",
                   "elevator": "elevator", "escalator": "escalator", "ramp": "ramp"}

    @classmethod
    def _plan_connectors(cls, xy, state: Mapping, wall_floor) -> list[Connector]:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        kinds = state.get("connector_kind")
        kinds = ["stairs"] * len(xy) if kinds is None else [str(k) for k in np.asarray(kinds).reshape(-1)]
        if len(kinds) != len(xy):
            raise ValueError(f"floor plan has {len(xy)} connectors but {len(kinds)} connector_kind entries")
        if state.get("n_floors") is not None:
            floors = tuple(range(int(state["n_floors"])))
        else:
            floors = tuple(sorted({int(f) for f in (wall_floor if wall_floor is not None else ())}))
        if len(floors) < 2:
            return []
        out = []
        for i, (p, kind) in enumerate(zip(xy, kinds)):
            if kind.lower() not in cls._PLAN_KINDS:
                raise ValueError(f"connector_kind {kind!r} is not one of {sorted(cls._PLAN_KINDS)}")
            out.append(Connector(cls._PLAN_KINDS[kind.lower()], floors, p, name=f"{kind} {i}"))
        return out

    # ---------------------------------------------------------------- properties
    @property
    def walls(self) -> np.ndarray:
        """``(M, 2, 2)`` float64 wall segments (read-only)."""
        return self._walls

    @property
    def wall_floor(self) -> np.ndarray:
        """``(M,)`` int64 floor of each wall."""
        return self._wall_floor

    @property
    def bounds(self) -> np.ndarray | None:
        """``(xmin, ymin, xmax, ymax)`` or None (unbounded)."""
        return self._bounds

    @property
    def floors(self) -> tuple[int, ...]:
        """Sorted floor ids that have walls or connector landings (``(0,)`` for an empty map)."""
        return self._floors

    @property
    def connectors(self) -> tuple[Connector, ...]:
        return self._connectors

    def walls_on(self, floor=None) -> np.ndarray:
        """Walls of one floor; ``None`` means every wall of every floor."""
        return self._walls if floor is None else self._walls[self._wall_floor == int(floor)]

    def __repr__(self) -> str:
        b = "None" if self._bounds is None else "(" + ", ".join(f"{v:g}" for v in self._bounds) + ")"
        return (f"FloorMap(n_walls={len(self._walls)}, floors={self._floors}, bounds={b}, "
                f"n_connectors={len(self._connectors)})")

    # ---------------------------------------------------------------- queries
    def contains(self, points) -> np.ndarray:
        """``(N,)`` True where a point lies inside the (closed) bounds; always True if unbounded."""
        p = _as_points(points)
        if self._bounds is None:
            return np.ones(len(p), dtype=bool)
        x0, y0, x1, y1 = self._bounds
        return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)

    def crosses(self, p0, p1, floor=None) -> np.ndarray:
        """``(N,)`` True where the move ``p0[i] -> p1[i]`` touches or crosses a wall of ``floor``.

        Vectorised over moves and walls (chunked so a temporary never exceeds ~1M elements);
        walls whose bounding box misses every move's bounding box are skipped first.
        """
        p0, p1 = _as_points(p0, "p0"), _as_points(p1, "p1")
        if p0.shape != p1.shape:
            raise ValueError(f"p0 {p0.shape} and p1 {p1.shape} differ in shape")
        walls = self.walls_on(floor)
        out = np.zeros(len(p0), dtype=bool)
        if len(walls) == 0 or len(p0) == 0:
            return out
        lo, hi = np.minimum(p0, p1).min(axis=0), np.maximum(p0, p1).max(axis=0)
        near = np.all((walls.min(axis=1) <= hi) & (walls.max(axis=1) >= lo), axis=1)
        walls = walls[near]
        if len(walls) == 0:
            return out
        moves = np.stack([p0, p1], axis=1)
        step = max(1, _CHUNK // len(walls))
        for s in range(0, len(moves), step):
            out[s:s + step] = segments_intersect(moves[s:s + step], walls).any(axis=1)
        return out

    def valid_moves(self, p0, p1, floor=None) -> np.ndarray:
        """``(N,)`` True where ``p1`` is inside the bounds and ``p0 -> p1`` crosses no wall."""
        return self.contains(p1) & ~self.crosses(p0, p1, floor)

    def distance_to_walls(self, points, floor=None) -> np.ndarray:
        """``(N,)`` distance from each point to the nearest wall of ``floor`` (inf without walls)."""
        p = _as_points(points)
        walls = self.walls_on(floor)
        if len(walls) == 0:
            return np.full(len(p), np.inf)
        step = max(1, _CHUNK // len(walls))
        return np.concatenate([point_segment_distance(p[s:s + step], walls).min(axis=1)
                               for s in range(0, len(p), step)]) if len(p) else np.zeros(0)

    def occupancy_grid(self, resolution: float, floor=None, clearance: float = 0.0) -> OccupancyGrid:
        """Rasterise the walls of ``floor`` onto a grid covering the bounds.

        A cell is blocked when its closed square intersects a wall, or when its centre lies
        within ``clearance`` of a wall (keeps routes away from walls by a body radius).
        """
        if self._bounds is None:
            raise ValueError("an occupancy grid needs map bounds")
        res = float(resolution)
        if not res > 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        x0, y0, x1, y1 = self._bounds
        w, h = max(1, math.ceil((x1 - x0) / res - 1e-9)), max(1, math.ceil((y1 - y0) / res - 1e-9))
        blocked = np.zeros((h, w), dtype=bool)
        origin = np.array([x0, y0])
        pad = float(clearance)
        for a, b in self.walls_on(floor):
            lo = np.floor((np.minimum(a, b) - pad - origin) / res).astype(int) - 1
            hi = np.floor((np.maximum(a, b) + pad - origin) / res).astype(int) + 1
            c0, r0 = np.maximum(lo, 0)
            c1, r1 = np.minimum(hi, [w - 1, h - 1])
            if c0 > c1 or r0 > r1:
                continue
            rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            rr, cc = rr.ravel(), cc.ravel()
            box_lo = origin + np.stack([cc, rr], axis=1) * res
            hit = _segment_hits_boxes(a, b, box_lo, box_lo + res)
            if pad > 0:
                hit |= point_segment_distance(box_lo + res / 2, np.stack([a, b])[None])[:, 0] <= pad
            blocked[rr[hit], cc[hit]] = True
        return OccupancyGrid(blocked, origin, res, None if floor is None else int(floor))


__all__ = ["Connector", "FloorMap", "OccupancyGrid", "point_segment_distance", "segments_intersect"]
