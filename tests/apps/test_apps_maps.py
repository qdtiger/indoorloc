"""L5 maps: segment crossing, distances, rasterisation, floors and connectors."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from indoorloc.apps.maps import Connector, FloorMap, point_segment_distance, segments_intersect


def _exact_intersect(p, q) -> bool:
    """Reference: SEGMENTS-INTERSECT of Cormen et al. (Introduction to Algorithms, Sec. 33.1)
    in exact rational arithmetic."""
    p1, p2, p3, p4 = [(Fraction(v[0]), Fraction(v[1])) for v in (*p, *q)]

    def direction(a, b, c):  # cross product (c - a) x (b - a)
        return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1])

    def on_segment(a, b, c):  # c within the bounding box of a-b
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])

    d1, d2, d3, d4 = direction(p3, p4, p1), direction(p3, p4, p2), direction(p1, p2, p3), direction(p1, p2, p4)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return ((d1 == 0 and on_segment(p3, p4, p1)) or (d2 == 0 and on_segment(p3, p4, p2))
            or (d3 == 0 and on_segment(p1, p2, p3)) or (d4 == 0 and on_segment(p1, p2, p4)))


def test_segment_intersection_textbook_cases():
    wall = [[[0, 0], [4, 0]]]
    cases = {  # move -> crosses the wall?
        ((1, -1), (1, 1)): True,     # proper crossing
        ((1, 1), (3, 2)): False,     # clear of it
        ((4, 0), (5, 1)): True,      # starts on the wall's end point (touching counts)
        ((2, 1), (2, 0)): True,      # ends on the wall
        ((5, 0), (6, 0)): False,     # collinear, disjoint
        ((3, 0), (6, 0)): True,      # collinear, overlapping
        ((2, 0), (2, 0)): True,      # zero-length move lying on the wall
        ((2, 1), (2, 1)): False,     # zero-length move off the wall
        ((5, -1), (5, 1)): False,    # crosses the wall's line beyond its end
    }
    moves = np.array([list(k) for k in cases], dtype=float)
    assert segments_intersect(moves, wall)[:, 0].tolist() == list(cases.values())


def test_vectorised_crossing_matches_exact_arithmetic_on_integer_grids():
    rng = np.random.default_rng(0)
    moves = rng.integers(0, 6, (400, 2, 2)).astype(float)  # small integers: many touching / collinear cases
    walls = rng.integers(0, 6, (25, 2, 2)).astype(float)
    got = segments_intersect(moves, walls)
    want = np.array([[_exact_intersect(m, w) for w in walls] for m in moves])
    assert np.array_equal(got, want)
    fmap = FloorMap(walls, bounds=(0, 0, 5, 5))
    assert np.array_equal(fmap.crosses(moves[:, 0], moves[:, 1]), want.any(axis=1))


def test_point_segment_distance_closed_forms():
    seg = [[[0.0, 0.0], [4.0, 0.0]]]
    pts = [[2.0, 3.0], [-3.0, 4.0], [4.0, 0.0], [6.0, 0.0]]
    assert np.allclose(point_segment_distance(pts, seg)[:, 0], [3.0, 5.0, 0.0, 2.0])
    assert np.allclose(point_segment_distance([[1.0, 1.0]], [[[2.0, 2.0], [2.0, 2.0]]]), np.sqrt(2))


def test_moves_leaving_the_bounds_or_crossing_walls_are_invalid():
    fmap = FloorMap.from_polygons([[(0, 0), (10, 0), (10, 5), (0, 5)]], bounds=(0, 0, 10, 5))
    assert len(fmap.walls) == 4  # the ring is closed
    p0 = np.array([[5.0, 2.5], [5.0, 2.5], [9.0, 2.5]])
    p1 = np.array([[6.0, 3.0], [5.0, 7.0], [11.0, 2.5]])
    assert fmap.valid_moves(p0, p1).tolist() == [True, False, False]
    assert np.allclose(fmap.distance_to_walls([[5.0, 2.0], [1.0, 1.0]]), [2.0, 1.0])


def test_occupancy_grid_marks_exactly_the_cells_a_wall_touches():
    fmap = FloorMap([[[2.5, 0.0], [2.5, 3.0]]], bounds=(0, 0, 5, 3))
    grid = fmap.occupancy_grid(1.0)
    assert grid.shape == (3, 5)
    assert np.array_equal(np.flatnonzero(grid.blocked.any(axis=0)), [2]) and grid.blocked[:, 2].all()
    on_line = FloorMap([[[2.0, 0.0], [2.0, 3.0]]], bounds=(0, 0, 5, 3)).occupancy_grid(1.0)
    assert np.array_equal(np.flatnonzero(on_line.blocked.all(axis=0)), [1, 2])  # closed squares share the line
    wide = fmap.occupancy_grid(1.0, clearance=1.0)  # centres within 1 m of the wall: columns 1, 2, 3
    assert np.array_equal(np.flatnonzero(wide.blocked.all(axis=0)), [1, 2, 3])
    diag = FloorMap([[[0.0, 0.0], [4.0, 4.0]]], bounds=(0, 0, 4, 4)).occupancy_grid(1.0)
    rows, cols = np.nonzero(diag.blocked)  # a diagonal through cell corners blocks a 4-connected staircase
    assert {(r, c) for r, c in zip(rows, cols)} >= {(i, i) for i in range(4)} | {(i, i + 1) for i in range(3)}
    assert np.allclose(grid.center([[0, 0], [2, 4]]), [[0.5, 0.5], [4.5, 2.5]])
    assert grid.cell([[4.9, 2.1]]).tolist() == [[2, 4]]


def test_multi_floor_walls_and_connectors():
    stairs = Connector("stairs", (1, 0, 2), xy=(9.0, 1.0), name="east")
    lift = Connector("elevator", (0, 2), xy=[(1.0, 1.0), (1.5, 1.0)])
    fmap = FloorMap({0: [[[5, 0], [5, 10]]], 1: [[[0, 5], [10, 5]]]}, connectors=[stairs, lift])
    assert fmap.floors == (0, 1, 2) and stairs.floors == (0, 1, 2)
    assert fmap.crosses([[4, 1]], [[6, 1]], floor=0).tolist() == [True]
    assert fmap.crosses([[4, 1]], [[6, 1]], floor=1).tolist() == [False]
    assert stairs.links() == [(0, 1), (1, 2)] and lift.links() == [(0, 2)]
    assert stairs.link_cost(0, 2) == 20.0 and lift.link_cost(0, 2) == 20.0 + 2 * 3.0
    assert np.allclose(lift.landing(2), [1.5, 1.0])
    assert np.allclose(fmap.bounds, [0, 0, 10, 10])
    with pytest.raises(ValueError, match="two distinct floors"):
        Connector("stairs", (1,), xy=(0, 0))
    with pytest.raises(ValueError, match="does not serve"):
        stairs.link_cost(0, 5)


def test_floor_map_round_trips_through_plain_arrays():
    stairs = Connector("stairs", (0, 1), xy=[(9.0, 1.0), (9.5, 1.0)], cost=12.0, name="east")
    fmap = FloorMap({0: [[[5, 0], [5, 10]]], 1: [[[0, 5], [10, 5]]]}, bounds=(0, 0, 10, 10), connectors=[stairs])
    state = fmap.to_dict()
    assert all(isinstance(v, np.ndarray) for v in (state["walls"], state["wall_floor"], state["bounds"]))
    again = FloorMap.from_dict(state)
    assert np.array_equal(again.walls, fmap.walls) and np.array_equal(again.wall_floor, fmap.wall_floor)
    assert np.array_equal(again.bounds, fmap.bounds) and again.floors == fmap.floors
    c = again.connectors[0]
    assert (c.kind, c.floors, c.cost, c.wait, c.name) == ("stairs", (0, 1), 12.0, 0.0, "east")
    assert np.array_equal(c.xy, stairs.xy)
