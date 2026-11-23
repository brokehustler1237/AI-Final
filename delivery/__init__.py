"""Mini Food Delivery Assistant.

AI Fundamentals (AIF1303), Final Project, topic 20.

A courier carries an order from a restaurant to a customer. The shortest
route crosses a bridge, and a traffic alert may say the bridge is blocked.
Three algorithms decide what the courier should do:

    search.py  A* search           shortest route, and the detour around the bridge
    bayes.py   Bayesian inference  how likely the bridge is really blocked
    mdp.py     MDP                 continue or reroute, from one Bellman update

planner.py connects them: the detour length from A* sets the reward for
rerouting, and the Bayes posterior is the belief the MDP decision is
weighted by. report.py prints every step; cli.py is the command line.

All numbers are assumptions chosen for the demonstration, not real data.
Standard library only.
"""

from . import world
from .bayes import bayes_blocked
from .mdp import bellman_update, decide, rewards, threshold, value_iteration
from .planner import decide_at, evaluate, plan_delivery, policy_grid, simulate_trip
from .search import astar, bfs, manhattan, neighbours
from .world import MAPS, use_map

__all__ = [
    "world", "MAPS", "use_map",
    "astar", "bfs", "manhattan", "neighbours",
    "bayes_blocked",
    "rewards", "bellman_update", "decide", "threshold", "value_iteration",
    "decide_at", "evaluate", "policy_grid", "simulate_trip", "plan_delivery",
]
