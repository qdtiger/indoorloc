"""L5 navigation: A* optimality, corner rules, smoothing, multi-floor routes, instructions."""
from __future__ import annotations

import heapq
import math

import numpy as np
import pytest

from indoorloc.apps.maps import Connector, FloorMap, OccupancyGrid
from indoorloc.apps.navigation import Navigator, astar, line_of_sight, path_length, smooth_path, turn_instructions

S2 = math.sqrt(2.0)


def _dijkstra(blocked, start, goal):
    """Reference: plain Dijkstra with the same move rules (8-connected, no corner cutting)."""
    h, w = blocked.shape
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == goal:
            return d
        if d > dist[(r, c)]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr, dc) == (0, 0) or not (0 <= rr < h and 0 <= cc < w) or blocked[rr, cc]:
                    continue
                if dr and dc and (blocked[r, cc] or blocked[rr, c]):
                    continue
                nd = d + (S2 if dr and dc else 1.0)
                if nd < dist.get((rr, cc), math.inf):
                    dist[(rr, cc)] = nd
                    heapq.heappush(heap, (nd, (rr, cc)))
    return math.inf


def _check_moves(path, blocked):
    steps = np.abs(np.diff(path, axis=0))
    assert steps.max() == 1 and np.all(steps.sum(axis=1) >= 1) and not blocked[path[:, 0], path[:, 1]].any()


def test_astar_known_shortest_paths():
    open_grid = np.zeros((10, 10), dtype=bool)
    p = astar(open_grid, (0, 0), (3, 7))
    assert path_length(p) == pytest.approx(3 * S2 + 4)  # octile distance
    assert path_length(astar(open_grid, (0, 0), (3, 7), connectivity=4)) == pytest.approx(10.0)  # Manhattan
    wall = np.zeros((7, 7), dtype=bool)
    wall[0:6, 3] = True  # a wall with its only gap at row 6
    p = astar(wall, (0, 0), (0, 6))
    _check_moves(p, wall)
    # into the gap horizontally (a diagonal would cut the wall's corner): 2 x (2 sqrt 2 + 4) + 2
    assert path_length(p) == pytest.approx(4 * S2 + 10)
    assert [3] == [c for r, c in p if r == 6 and c == 3]  # it passes through the gap cell (6, 3)


def test_astar_matches_dijkstra_on_random_grids():
    rng = np.random.default_rng(0)
    for trial in range(25):
        blocked = rng.random((25, 30)) < 0.3
        free = np.argwhere(~blocked)
        s, g = (tuple(v) for v in free[rng.choice(len(free), 2, replace=False)])
        ref = _dijkstra(blocked, s, g)
        p = astar(blocked, s, g)
        if math.isinf(ref):
            assert p is None
        else:
            _check_moves(p, blocked)
            assert tuple(p[0]) == s and tuple(p[-1]) == g
            assert path_length(p) == pytest.approx(ref, abs=1e-9)


def test_astar_rejects_bad_endpoints_and_diagonal_squeezes():
    squeeze = np.array([[0, 1], [1, 0]], dtype=bool)  # free cells touch only at a corner
    assert astar(squeeze, (0, 0), (1, 1)) is None
    with pytest.raises(ValueError, match="blocked"):
        astar(squeeze, (0, 1), (1, 1))
    grid = OccupancyGrid(squeeze, origin=(0.0, 0.0), resolution=1.0)
    assert not line_of_sight(grid, (0.5, 0.5), (1.5, 1.5))  # the corner between two blocked cells
    open_grid = OccupancyGrid(np.zeros((5, 5), dtype=bool), origin=(0.0, 0.0), resolution=1.0)
    assert line_of_sight(open_grid, (0.5, 0.5), (4.5, 3.2))
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    assert not line_of_sight(OccupancyGrid(blocked, (0.0, 0.0), 1.0), (0.5, 2.5), (4.5, 2.5))


def test_smoothing_turns_a_staircase_into_a_straight_line():
    grid = OccupancyGrid(np.zeros((10, 10), dtype=bool), origin=(0.0, 0.0), resolution=1.0)
    cells = astar(grid.blocked, (0, 0), (9, 4))
    pts = grid.center(cells)
    smooth = smooth_path(pts, grid)
    assert len(smooth) == 2 and path_length(smooth) == pytest.approx(np.hypot(9, 4))
    assert path_length(smooth) <= path_length(pts)


def test_route_through_a_door_stays_clear_of_the_walls():
    room = FloorMap.from_polygons([[(0, 0), (20, 0), (20, 10), (0, 10)]])
    walls = np.concatenate([room.walls, [[[10.0, 0.0], [10.0, 7.0]]]])
    fmap = FloorMap(walls)
    route = Navigator(resolution=0.25, clearance=0.3).fit(fmap).route((5.0, 2.0), (15.0, 2.0))
    assert np.allclose(route.points[0], [5, 2]) and np.allclose(route.points[-1], [15, 2])
    assert not fmap.crosses(route.points[:-1], route.points[1:]).any()
    lower = 2 * np.hypot(5.0, 5.0)  # the any-angle optimum around the wall end (10, 7)
    assert lower < route.length < lower + 1.0 and route.cost >= route.length - 1e-9
    actions = [i.action for i in route.instructions]
    assert actions[0] == "start" and actions[-1] == "arrive" and "right" in "".join(actions)


def test_multi_floor_route_takes_the_cheaper_connector_unless_excluded():
    stairs = Connector("stairs", (0, 1), xy=(19.0, 9.0), name="east stairs")
    lift = Connector("elevator", (0, 1), xy=(1.0, 2.0))
    fmap = FloorMap({0: np.zeros((0, 2, 2)), 1: np.zeros((0, 2, 2))}, bounds=(0, 0, 20, 10),
                    connectors=[stairs, lift])
    nav = Navigator(resolution=0.5, clearance=0.0).fit(fmap)
    via_stairs = nav.route((1.0, 1.0), (19.0, 1.0), start_floor=0, goal_floor=1)
    assert via_stairs.floor[0] == 0 and via_stairs.floor[-1] == 1
    assert any(i.action == "stairs" for i in via_stairs.instructions)
    assert "Take the stairs from floor 0 to floor 1" in " ".join(i.text for i in via_stairs.instructions)
    # walking 1 -> stairs -> goal: |(1,1)-(19,9)| + 10 + 8 m (straight legs after smoothing)
    assert via_stairs.length == pytest.approx(np.hypot(18, 8) + 8.0, abs=0.75)
    step_free = Navigator(resolution=0.5, clearance=0.0, connector_kinds=("elevator",)).fit(fmap)
    via_lift = step_free.route((1.0, 1.0), (19.0, 1.0), start_floor=0, goal_floor=1)
    assert any(i.action == "elevator" for i in via_lift.instructions)
    assert via_lift.cost > via_stairs.cost
    with pytest.raises(ValueError, match="start_floor"):
        nav.route((1.0, 1.0), (19.0, 1.0))
    only_lift = Navigator(connector_kinds=("escalator",)).fit(fmap)
    assert only_lift.route((1.0, 1.0), (19.0, 1.0), start_floor=0, goal_floor=1) is None


def test_turn_by_turn_instructions_for_a_known_polyline():
    pts = [(0, 0), (10, 0), (10, 5), (15, 5), (20, 5.5), (12, 5.5)]
    ins = turn_instructions(pts)
    assert [i.action for i in ins] == ["start", "left", "right", "u_turn", "arrive"]
    assert np.allclose([i.distance for i in ins], [10.0, 5.0, 5.0 + np.hypot(5.0, 0.5), 8.0, 0.0])
    assert ins[1].angle == pytest.approx(90.0) and ins[2].angle == pytest.approx(-90.0)
    assert ins[1].text == "Turn left, then walk 5.0 m" and ins[-1].text == "Arrive at the destination"
    assert [i.action for i in turn_instructions([(0, 0), (5, 0), (8, 2)])] == ["start", "slight_left", "arrive"]
    assert path_length([(0, 0), (3, 4), (3, 0)]) == pytest.approx(9.0)


def test_start_and_goal_in_one_cell_keep_both_points():
    # regression: when start and goal fall in the same free cell the route is the straight segment
    fmap = FloorMap.from_polygons([[(0, 0), (10, 0), (10, 10), (0, 10)]])
    nav = Navigator(resolution=0.5, clearance=0.0).fit(fmap)
    route = nav.route((2.1, 2.1), (2.3, 2.4))
    assert np.allclose(route.points, [[2.1, 2.1], [2.3, 2.4]])
    assert route.length == pytest.approx(np.hypot(0.2, 0.3)) and route.cost == 0.0
    assert [i.action for i in route.instructions] == ["start", "arrive"]


def test_saved_navigator_gives_the_same_routes(tmp_path):
    from indoorloc.core import load_model

    stairs = Connector("stairs", (0, 1), xy=(19.0, 9.0), name="east stairs")
    fmap = FloorMap({0: [[[10.0, 0.0], [10.0, 7.0]]], 1: np.zeros((0, 2, 2))}, bounds=(0, 0, 20, 10),
                    connectors=[stairs])
    nav = Navigator(resolution=0.5, clearance=0.25).fit(fmap)
    again = load_model(nav.save(tmp_path / "nav"))  # arrays + JSON only: the plan, re-rasterised on load
    for start, goal, fs, fg in [((2.0, 2.0), (18.0, 2.0), 0, 0), ((2.0, 2.0), (5.0, 8.0), 0, 1)]:
        a = nav.route(start, goal, start_floor=fs, goal_floor=fg)
        b = again.route(start, goal, start_floor=fs, goal_floor=fg)
        assert np.array_equal(a.points, b.points) and np.array_equal(a.floor, b.floor) and a.cost == b.cost
    assert np.array_equal(again.map_.walls, fmap.walls) and again.map_.connectors[0].name == "east stairs"
