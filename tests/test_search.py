"""Tests: A* search and the BFS baseline.

Each check uses a method different from the one in the program.
"""

import delivery as fd
from delivery import world
from tests.helpers import bfs_length, valid_path


def test_astar_route_is_shortest():
    path, _ = fd.astar(world.START, world.GOAL, world.WALLS)
    assert valid_path(path, world.WALLS)
    assert len(path) - 1 == bfs_length(world.START, world.GOAL, world.WALLS) == 7
    assert world.ALERT_CELL in path          # the alert matters: it is on the route

def test_detour_avoids_bridge_and_is_shortest():
    walls = world.WALLS | {world.ALERT_CELL}
    detour, _ = fd.astar(world.START, world.GOAL, walls)
    assert valid_path(detour, walls)
    assert world.ALERT_CELL not in detour
    assert len(detour) - 1 == bfs_length(world.START, world.GOAL, walls) == 9

def test_heuristic_is_admissible_everywhere():
    for r in range(world.ROWS):
        for c in range(world.COLS):
            if (r, c) in world.WALLS:
                continue
            true_cost = bfs_length((r, c), world.GOAL, world.WALLS)
            assert fd.manhattan((r, c), world.GOAL) <= true_cost

def test_no_route_when_customer_is_walled_in():
    walls = world.WALLS | {(0, 5), (1, 4), (2, 5)}
    path, _ = fd.astar(world.START, world.GOAL, walls)
    assert path is None

def test_bfs_finds_same_length_but_expands_more():
    path, log = fd.astar(world.START, world.GOAL, world.WALLS)
    bpath, bexpanded = fd.bfs(world.START, world.GOAL, world.WALLS)
    assert len(bpath) == len(path)
    assert len(bexpanded) == 30 and len(log) == 10
