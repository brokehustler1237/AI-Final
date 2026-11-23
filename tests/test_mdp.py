"""Tests: The MDP: Bellman update, decision, threshold, value iteration.

Each check uses a method different from the one in the program.
"""

import delivery as fd
from delivery.config import ACTIONS, STATES


def test_bellman_update_values():
    _, Q = fd.bellman_update(fd.rewards(extra_steps=2))
    expected = {("Clear", "Continue"): 18.64, ("Clear", "Reroute"): 14.64,
                ("Blocked", "Continue"): -1.52, ("Blocked", "Reroute"): 14.64}
    for key, value in expected.items():
        assert abs(Q[key] - value) < 1e-9, key

def test_threshold_is_the_break_even_point():
    _, Q = fd.bellman_update(fd.rewards(extra_steps=2))
    t = fd.threshold(Q)
    _, ev = fd.decide(t, Q)
    assert abs(ev["Continue"] - ev["Reroute"]) < 1e-9
    assert abs(t - 4 / 20.16) < 1e-9
    assert fd.decide(t - 0.01, Q)[0] == "Continue"
    assert fd.decide(t + 0.01, Q)[0] == "Reroute"

def test_long_detour_flips_the_decision():
    # At P(blocked) = 0.5 rerouting wins while the detour is short and loses
    # once it is long enough. Even a 6-block detour (reward -2) still wins,
    # because the Bellman update also counts that rerouting leads to a road
    # that is likely to stay clear. The decision only flips at 7 extra blocks.
    decisions = {}
    for extra in range(0, 11):
        _, Q = fd.bellman_update(fd.rewards(extra_steps=extra))
        decisions[extra] = fd.decide(0.5, Q)[0]
    assert decisions[2] == "Reroute"
    assert decisions[6] == "Reroute"
    assert decisions[8] == "Continue"
    flip = min(e for e, a in decisions.items() if a == "Continue")
    assert all(decisions[e] == "Reroute" for e in range(flip))
    assert all(decisions[e] == "Continue" for e in range(flip, 11))

def test_one_update_policy_matches_full_value_iteration():
    for extra in range(0, 11):
        R = fd.rewards(extra)
        _, Q = fd.bellman_update(R)
        one = {s: max(ACTIONS, key=lambda a: Q[(s, a)]) for s in STATES}
        _, _, full, _ = fd.value_iteration(R)
        assert one == full, extra
