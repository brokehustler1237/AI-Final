"""Puts the three algorithms together: decisions from any position, trip
simulation, the decision map and the evaluation of three policies.
"""

import random

from . import world
from .bayes import bayes_blocked
from .config import (MINUTES_PER_BLOCK, P_ALERT_IF_BLOCKED, P_ALERT_IF_CLEAR,
                     PRIOR_BLOCKED, STUCK_WAIT_MINUTES)
from .mdp import bellman_update, decide, rewards, threshold
from .search import astar


def decide_at(position, remaining_blocks, observation):
    """The whole decision from wherever the courier is: A* finds the detour
    from this cell, its length sets the reroute reward, Bayes gives the
    belief, one Bellman update gives Q, and the belief picks the action."""
    detour, _ = astar(position, world.GOAL, world.WALLS | {world.ALERT_CELL})
    belief, _ = bayes_blocked(PRIOR_BLOCKED, observation)
    if detour is None:
        return {"detour": None, "belief": belief, "action": "Continue"}
    extra = (len(detour) - 1) - remaining_blocks
    R = rewards(extra)
    V1, Q = bellman_update(R)
    action, ev = decide(belief, Q)
    return {"detour": detour, "extra": extra, "R": R, "V1": V1, "Q": Q,
            "belief": belief, "action": action, "ev": ev, "threshold": threshold(Q)}

def evaluate(Q, extra_steps, n=10000, seed=7):
    """Compare three policies over many evening deliveries.

    For each delivery the bridge is blocked with the prior probability, the
    alert service reports with its error rates, and the courier earns the
    immediate reward R(true state, action). Expected values are computed
    exactly; the simulation is there to show they hold up in practice.
    """
    rng = random.Random(seed)
    R = rewards(extra_steps)

    def assistant(observation):
        belief, _ = bayes_blocked(PRIOR_BLOCKED, observation)
        return decide(belief, Q)[0]

    policies = {
        "Always continue": lambda obs: "Continue",
        "Always reroute": lambda obs: "Reroute",
        "Assistant": assistant,
    }

    results = {}
    for name, policy in policies.items():
        # exact expectation over (true state, observation)
        exp_reward = exp_stuck = exp_reroute = 0.0
        for state, p_state in (("Blocked", PRIOR_BLOCKED), ("Clear", 1 - PRIOR_BLOCKED)):
            p_alert = P_ALERT_IF_BLOCKED if state == "Blocked" else P_ALERT_IF_CLEAR
            for obs, p_obs in (("ALERT", p_alert), ("NO ALERT", 1 - p_alert)):
                p = p_state * p_obs
                action = policy(obs)
                exp_reward += p * R[(state, action)]
                exp_stuck += p * (state == "Blocked" and action == "Continue")
                exp_reroute += p * (action == "Reroute")

        # simulation
        total = stuck = rerouted = 0
        for _ in range(n):
            state = "Blocked" if rng.random() < PRIOR_BLOCKED else "Clear"
            p_alert = P_ALERT_IF_BLOCKED if state == "Blocked" else P_ALERT_IF_CLEAR
            obs = "ALERT" if rng.random() < p_alert else "NO ALERT"
            action = policy(obs)
            total += R[(state, action)]
            stuck += state == "Blocked" and action == "Continue"
            rerouted += action == "Reroute"

        results[name] = {"expected": exp_reward, "stuck": exp_stuck,
                         "rerouted": exp_reroute, "simulated": total / n,
                         "sim_stuck": stuck / n}
    return results

def policy_grid(max_extra=10, steps=20):
    """Decision for every combination of risk and detour length."""
    beliefs = [i / steps for i in range(steps + 1)]
    grid = {}
    for extra in range(max_extra + 1):
        _, Q = bellman_update(rewards(extra))
        for b in beliefs:
            grid[(extra, b)] = decide(b, Q)[0]
    return beliefs, grid

def simulate_trip(observation="ALERT", on_the_way=False, truth="clear"):
    """What actually happens on the road.

    truth is whether the bridge really was blocked ('blocked' or 'clear').
    The assistant never sees it; it only matters if the courier drives up
    to the bridge. Returns the cells driven, the minutes, and a log."""
    path, _ = astar(world.START, world.GOAL, world.WALLS)
    if world.ALERT_CELL not in path:
        return {"trip": path, "minutes": (len(path) - 1) * MINUTES_PER_BLOCK,
                "action": "Continue", "events": ["bridge not on the route"], "stuck": False}
    bridge_i = path.index(world.ALERT_CELL)
    i = bridge_i - 1 if on_the_way else 0
    d = decide_at(path[i], len(path) - 1 - i, observation)
    events = ["alert decision at %s: %s" % (path[i], d["action"])]
    stuck = False
    if d["action"] == "Reroute":
        trip = path[:i + 1] + d["detour"][1:]
        wait = 0
    elif truth == "blocked":
        stop = bridge_i - 1
        replan, _ = astar(path[stop], world.GOAL, world.WALLS | {world.ALERT_CELL})
        wait = STUCK_WAIT_MINUTES
        stuck = True
        if replan is None:
            trip = path
            events.append("bridge blocked and no way around, courier waits %d min"
                          " at %s until it clears" % (wait, path[stop]))
        else:
            trip = path[:stop + 1] + replan[1:]
            events.append("bridge blocked, courier stuck at %s for %d min, A* replans"
                          % (path[stop], wait))
    else:
        trip = path
        wait = 0
        events.append("bridge clear, courier crosses")
    minutes = (len(trip) - 1) * MINUTES_PER_BLOCK + wait
    return {"trip": trip, "minutes": minutes, "action": d["action"],
            "events": events, "stuck": stuck, "decision": d}

def plan_delivery(observation="ALERT", verbose=True):
    """Run all three stages. Returns a dict so tests and the notebook can use it.
    With verbose=True every stage is also printed to the console."""
    if verbose:
        from .report import run_astar, run_bayes, run_evaluation, run_mdp
    if verbose:
        path, log = run_astar()
    else:
        path, log = astar(world.START, world.GOAL, world.WALLS)
    detour, detour_log = astar(world.START, world.GOAL, world.WALLS | {world.ALERT_CELL})
    if detour is None:
        # No way around the bridge: rerouting is not an available action.
        belief, _ = bayes_blocked(PRIOR_BLOCKED, observation)
        if verbose:
            print("No detour exists around %s, so the courier must continue."
                  % (world.ALERT_CELL,))
        return {"route": path, "explored": [c for c, *_ in log], "detour": None,
                "belief": belief, "Q": None, "ev": None, "threshold": None,
                "action": "Continue", "final_route": path}

    if verbose:
        belief = run_bayes(observation)
        action, Q, ev, t = run_mdp(belief, len(path) - 1, len(detour) - 1)
        run_evaluation(Q, len(detour) - len(path))
    else:
        belief, _ = bayes_blocked(PRIOR_BLOCKED, observation)
        R = rewards((len(detour) - 1) - (len(path) - 1))
        _, Q = bellman_update(R)
        action, ev = decide(belief, Q)
        t = threshold(Q)

    final = detour if action == "Reroute" else path
    result = {
        "route": path, "explored": [c for c, *_ in log],
        "detour": detour, "belief": belief, "Q": Q, "ev": ev,
        "threshold": t, "action": action, "final_route": final,
    }

    return result
