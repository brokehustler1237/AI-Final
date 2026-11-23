"""Tests: The three algorithms working together.

Each check uses a method different from the one in the program.
"""

import delivery as fd
from delivery import world
from tests.helpers import bfs_length


def test_decisions_for_each_observation():
    assert fd.plan_delivery("ALERT", verbose=False)["action"] == "Reroute"
    assert fd.plan_delivery("NO ALERT", verbose=False)["action"] == "Continue"
    assert fd.plan_delivery(None, verbose=False)["action"] == "Continue"

def test_assistant_beats_both_simple_policies():
    _, Q = fd.bellman_update(fd.rewards(extra_steps=2))
    res = fd.evaluate(Q, extra_steps=2, n=20000)
    a = res["Assistant"]
    assert a["expected"] > res["Always continue"]["expected"]
    assert a["expected"] > res["Always reroute"]["expected"]
    assert abs(a["expected"] - 8.575) < 1e-9            # worked out by hand
    assert abs(a["stuck"] - 0.15 * 0.15) < 1e-12        # blocked and no alert
    assert abs(a["simulated"] - a["expected"]) < 0.15   # simulation agrees

def test_alert_on_the_way_replans_from_courier_position():
    path, _ = fd.astar(world.START, world.GOAL, world.WALLS)
    i = path.index(world.ALERT_CELL) - 1
    d = fd.decide_at(path[i], len(path) - 1 - i, "ALERT")
    assert path[i] == (2, 1)
    assert d["detour"][0] == (2, 1) and world.ALERT_CELL not in d["detour"]
    assert d["extra"] == 4 and d["action"] == "Reroute"
    assert abs(d["threshold"] - 8 / 22.32) < 1e-9       # higher bar than at departure

def test_trip_outcomes():
    expected = {
        ("ALERT", False, "clear"): (18, False), ("ALERT", True, "clear"): (22, False),
        ("NO ALERT", False, "clear"): (14, False), ("NO ALERT", False, "blocked"): (28, True),
        (None, True, "blocked"): (28, True),
    }
    for (obs, way, truth), (minutes, stuck) in expected.items():
        trip = fd.simulate_trip(obs, way, truth)
        assert trip["minutes"] == minutes and trip["stuck"] == stuck, (obs, way, truth)
        assert trip["trip"][0] == world.START and trip["trip"][-1] == world.GOAL
        assert all(fd.manhattan(a, b) == 1 for a, b in zip(trip["trip"], trip["trip"][1:]))
        if stuck:
            assert world.ALERT_CELL not in trip["trip"]

def test_decision_map_is_monotone():
    beliefs, grid = fd.policy_grid()
    for b in beliefs:
        for extra in range(1, 11):
            if grid[(extra, b)] == "Reroute":
                assert grid[(extra - 1, b)] == "Reroute"
    for extra in range(0, 11):
        for lo, hi in zip(beliefs, beliefs[1:]):
            if grid[(extra, lo)] == "Reroute":
                assert grid[(extra, hi)] == "Reroute"

def test_bigger_maps():
    expected = {  # map: (route, detour, A* expanded, BFS expanded, on-the-way decision)
        "10x10": (16, 18, 28, 78, "Continue"),
        "14x14": (18, 22, 37, 140, "Reroute"),
    }
    try:
        for name, (route_len, detour_len, a_exp, b_exp, way_action) in expected.items():
            fd.use_map(name)
            path, log = fd.astar(world.START, world.GOAL, world.WALLS)
            detour, _ = fd.astar(world.START, world.GOAL, world.WALLS | {world.ALERT_CELL})
            _, bexp = fd.bfs(world.START, world.GOAL, world.WALLS)
            assert world.ALERT_CELL in path, name
            assert len(path) - 1 == bfs_length(world.START, world.GOAL, world.WALLS) == route_len, name
            assert len(detour) - 1 == detour_len, name
            assert (len(log), len(bexp)) == (a_exp, b_exp), name
            assert fd.plan_delivery("ALERT", verbose=False)["action"] == "Reroute", name
            assert fd.plan_delivery("NO ALERT", verbose=False)["action"] == "Continue", name
            i = path.index(world.ALERT_CELL) - 1
            assert fd.decide_at(path[i], len(path) - 1 - i, "ALERT")["action"] == way_action, name
            for r in range(world.ROWS):
                for c in range(world.COLS):
                    if (r, c) not in world.WALLS:
                        true_cost = bfs_length((r, c), world.GOAL, world.WALLS)
                        if true_cost is not None:
                            assert fd.manhattan((r, c), world.GOAL) <= true_cost
    finally:
        fd.use_map("6x6")
