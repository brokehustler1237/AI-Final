"""Console output: prints input, steps, output and an explanation for every
stage, as the project guidelines ask.
"""

from . import world
from .bayes import bayes_blocked
from .config import (ACTIONS, GAMMA, MINUTES_PER_BLOCK, ON_TIME_REWARD,
                     PENALTY_PER_EXTRA_STEP, P_ALERT_IF_BLOCKED, P_ALERT_IF_CLEAR,
                     PRIOR_BLOCKED, STATES, STUCK_WAIT_MINUTES, TRANSITIONS)
from .mdp import bellman_update, decide, rewards, threshold, value_iteration
from .planner import decide_at, evaluate, plan_delivery, policy_grid, simulate_trip
from .search import astar, bfs

LINE = "=" * 64
THIN = "-" * 64


def draw(layers):
    """Print the map. layers = [(cells, symbol), ...]; later layers win.
    Blocked cells, S, C and the alert cell are always shown on top."""
    marks = {}
    for cells, symbol in layers:
        for cell in cells:
            marks[cell] = symbol
    w = 2 if world.COLS > 10 else 1
    wr = 2 if world.ROWS > 10 else 1
    print(" " * (4 + wr) + " ".join(str(c).rjust(w) for c in range(world.COLS)))
    for r in range(world.ROWS):
        row = []
        for c in range(world.COLS):
            cell = (r, c)
            if cell in world.WATER:
                ch = "="
            elif cell in world.WALLS:
                ch = "#"
            elif cell == world.START:
                ch = "S"
            elif cell == world.GOAL:
                ch = "C"
            elif cell == world.ALERT_CELL:
                ch = "!"
            else:
                ch = marks.get(cell, ".")
            row.append(ch.rjust(w))
        print("  %s  %s" % (str(r).rjust(wr), " ".join(row)))

def run_astar():
    print(LINE)
    print("1. A* SEARCH - shortest route from restaurant to customer")
    print(LINE)
    blocked = (str(sorted(world.WALLS)) if len(world.WALLS) <= 6
               else "%d cells (%d of them river)" % (len(world.WALLS), len(world.WATER)))
    print("Input : %dx%d map, start S%s, goal C%s, blocked %s"
          % (world.ROWS, world.COLS, world.START, world.GOAL, blocked))
    print("        heuristic h(n) = |row - goal_row| + |col - goal_col|")
    print()

    path, log = astar(world.START, world.GOAL, world.WALLS)

    print("Steps : order in which A* expanded cells")
    print("  %-4s %-8s %3s %3s %3s" % ("#", "cell", "g", "h", "f"))
    for i, (cell, g, h, f) in enumerate(log, 1):
        print("  %-4d %-8s %3d %3d %3d" % (i, str(cell), g, h, f))
    print()
    draw([([cell for cell, *_ in log], "~"), (path, "*")])
    print("  S restaurant  C customer  # blocked  * route  ~ explored")
    print("  ! the bridge (2, 2), on the route")
    print()
    print("Output: route = " + " -> ".join(str(c) for c in path))
    print("        length = %d blocks, about %d minutes, %d cells expanded"
          % (len(path) - 1, (len(path) - 1) * MINUTES_PER_BLOCK, len(log)))
    print()
    bfs_path, bfs_expanded = bfs(world.START, world.GOAL, world.WALLS)
    print("        BFS on the same map: %d blocks too, but %d cells expanded"
          % (len(bfs_path) - 1, len(bfs_expanded)))
    print()
    print("Why   : A* only expanded %d of %d open cells because f = g + h"
          % (len(log), world.ROWS * world.COLS - len(world.WALLS)))
    print("        keeps it moving toward the customer, and an admissible h")
    print("        guarantees the route it returns is the shortest.")
    print()
    return path, log

def run_bayes(observation="ALERT"):
    print(LINE)
    print("2. BAYESIAN INFERENCE - is the bridge %s really blocked?" % (world.ALERT_CELL,))
    print(LINE)
    print("Input : prior P(blocked) = %.2f" % PRIOR_BLOCKED)
    print("        P(alert | blocked) = %.2f, P(alert | clear) = %.2f"
          % (P_ALERT_IF_BLOCKED, P_ALERT_IF_CLEAR))
    print("        observation = %s" % observation)
    print()

    posterior, evidence = bayes_blocked(PRIOR_BLOCKED, observation)
    if observation == "ALERT":
        print("Steps : P(alert) = %.2f x %.2f + %.2f x %.2f = %.4f"
              % (P_ALERT_IF_BLOCKED, PRIOR_BLOCKED, P_ALERT_IF_CLEAR,
                 1 - PRIOR_BLOCKED, evidence))
        print("        P(blocked | alert) = %.2f x %.2f / %.4f = %.4f"
              % (P_ALERT_IF_BLOCKED, PRIOR_BLOCKED, evidence, posterior))
    elif observation == "NO ALERT":
        print("Steps : P(no alert) = %.2f x %.2f + %.2f x %.2f = %.4f"
              % (1 - P_ALERT_IF_BLOCKED, PRIOR_BLOCKED, 1 - P_ALERT_IF_CLEAR,
                 1 - PRIOR_BLOCKED, evidence))
        print("        P(blocked | no alert) = %.2f x %.2f / %.4f = %.4f"
              % (1 - P_ALERT_IF_BLOCKED, PRIOR_BLOCKED, evidence, posterior))
    else:
        print("Steps : no report, so the posterior equals the prior")
    print()
    print("Output: P(blocked) %.2f -> %.4f" % (PRIOR_BLOCKED, posterior))
    print()
    print("Why   : the alert service is fairly reliable, but blockages are")
    print("        rare, so one alert only raises the probability to about")
    print("        a coin flip rather than to certainty.")
    print()
    return posterior

def run_mdp(belief, route_len, detour_len):
    print(LINE)
    print("3. MDP - continue over the bridge, or reroute?")
    print(LINE)
    extra = detour_len - route_len
    R = rewards(extra)
    print("Input : states %s, actions %s, gamma = %.1f" % (STATES, ACTIONS, GAMMA))
    print("        detour from A* = %d blocks vs %d -> %d extra blocks"
          % (detour_len, route_len, extra))
    print("        reroute reward = %d - %d x %d = %d"
          % (ON_TIME_REWARD, PENALTY_PER_EXTRA_STEP, extra, R[("Clear", "Reroute")]))
    print()
    print("  %-8s %-9s %-24s %5s" % ("state", "action", "next state", "R"))
    for s in STATES:
        for a in ACTIONS:
            t = ", ".join("%s %.1f" % kv for kv in TRANSITIONS[(s, a)].items())
            print("  %-8s %-9s %-24s %+5d" % (s, a, t, R[(s, a)]))
    print()

    V1, Q = bellman_update(R)
    print("Steps : V1(s) = max_a R(s,a):  V1(Clear) = %d, V1(Blocked) = %d"
          % (V1["Clear"], V1["Blocked"]))
    print("        Q(s,a) = R(s,a) + %.1f x sum T(s'|s,a) V1(s')" % GAMMA)
    for s in STATES:
        for a in ACTIONS:
            parts = " + ".join("%.1fx%d" % (p, V1[s2])
                               for s2, p in TRANSITIONS[(s, a)].items())
            print("        Q(%s, %s) = %d + %.1f(%s) = %.2f"
                  % (s, a, R[(s, a)], GAMMA, parts, Q[(s, a)]))
    print()

    action, ev = decide(belief, Q)
    print("        weighted by the Bayes belief P(Blocked) = %.4f:" % belief)
    for a in ACTIONS:
        print("        EV(%s) = %.4f x %.2f + %.4f x %.2f = %.2f"
              % (a, 1 - belief, Q[("Clear", a)], belief, Q[("Blocked", a)], ev[a]))
    t = threshold(Q)
    print()
    print("Output: action = %s" % action.upper())
    print("        switch to Reroute whenever P(blocked) > %.3f" % t)
    print()
    _, _, policy, sweeps = value_iteration(R)
    one_step = {st: max(ACTIONS, key=lambda a: Q[(st, a)]) for st in STATES}
    print("Check : full value iteration converges in %d sweeps to the policy" % sweeps)
    print("        %s, the same as after one update: %s"
          % (", ".join("%s -> %s" % kv for kv in policy.items()),
             "yes" if policy == one_step else "NO"))
    print()
    print("Why   : rerouting costs a sure %d points, continuing gambles on"
          % (ON_TIME_REWARD - R[("Clear", "Reroute")]))
    print("        the bridge; once the risk passes %.0f%% the gamble is worse." % (t * 100))
    print()
    return action, Q, ev, t

def run_evaluation(Q, extra_steps):
    print(LINE)
    print("4. EVALUATION - 10,000 simulated evening deliveries")
    print(LINE)
    res = evaluate(Q, extra_steps)
    print("  %-17s %10s %10s %11s %10s" %
          ("policy", "avg reward", "simulated", "stuck at", "rerouted"))
    print("  %-17s %10s %10s %11s %10s" % ("", "(exact)", "", "bridge", ""))
    print("  " + "-" * 62)
    for name, r in res.items():
        print("  %-17s %10.2f %10.2f %10.1f%% %9.1f%%"
              % (name, r["expected"], r["simulated"], 100 * r["stuck"],
                 100 * r["rerouted"]))
    print()
    a, c = res["Assistant"], res["Always continue"]
    print("The assistant cuts orders stuck at the bridge from %.1f%% to %.1f%%"
          % (100 * c["stuck"], 100 * a["stuck"]))
    print("while rerouting only %.1f%% of orders, and earns the highest"
          % (100 * a["rerouted"]))
    print("average reward of the three policies.")
    print()
    return res

def run_on_the_way(observation="ALERT"):
    """Same three algorithms, but the alert reaches the courier one block
    before the bridge. A* replans from the courier's position."""
    path, _ = astar(world.START, world.GOAL, world.WALLS)
    i = path.index(world.ALERT_CELL) - 1
    pos = path[i]
    remaining = len(path) - 1 - i
    d = decide_at(pos, remaining, observation)
    d0 = decide_at(world.START, len(path) - 1, observation)

    print(LINE)
    print("5. ON THE WAY - the alert arrives when the courier is at %s" % (pos,))
    print(LINE)
    if d["detour"] is None:
        print("No detour exists from %s, so the courier continues." % (pos,))
        print()
        return d
    print("Input : courier at %s, %d blocks left on the planned route" % (pos, remaining))
    print("Steps : A* from %s with the bridge closed -> %d blocks, %d extra"
          % (pos, len(d["detour"]) - 1, d["extra"]))
    print("        reroute reward = %d - %d x %d = %d"
          % (ON_TIME_REWARD, PENALTY_PER_EXTRA_STEP, d["extra"], d["R"][("Clear", "Reroute")]))
    print("        Q(Clear, Continue) = %.2f   Q(Clear, Reroute) = %.2f"
          % (d["Q"][("Clear", "Continue")], d["Q"][("Clear", "Reroute")]))
    print("        Q(Blocked, Continue) = %.2f  Q(Blocked, Reroute) = %.2f"
          % (d["Q"][("Blocked", "Continue")], d["Q"][("Blocked", "Reroute")]))
    print("        EV(Continue) = %.2f, EV(Reroute) = %.2f with P(blocked) = %.3f"
          % (d["ev"]["Continue"], d["ev"]["Reroute"], d["belief"]))
    print()
    print("Output: action = %s, reroute threshold %.1f%% (was %.1f%% at departure)"
          % (d["action"].upper(), 100 * d["threshold"], 100 * d0["threshold"]))
    print()
    print("Why   : the later the alert, the longer the way around, so the risk")
    print("        has to be higher before turning back is worth it.")
    print()
    return d

def run_policy_map(current_extra, current_belief):
    print(LINE)
    print("6. DECISION MAP - R = reroute, . = continue, @ = this delivery")
    print(LINE)
    beliefs, grid = policy_grid()
    cb = min(beliefs, key=lambda b: abs(b - current_belief))
    width = 3 * len(beliefs) - 2
    print("  " + " " * 12 + "P(blocked) ->")
    print("  %-12s0%%%s100%%" % ("detour", " " * (width - 6)))
    for extra in range(10, -1, -1):
        row = "".join("@" if (extra == current_extra and b == cb)
                      else ("R" if grid[(extra, b)] == "Reroute" else ".")
                      for b in beliefs)
        print("  +%-2d %-7s %s" % (extra, "block" if extra == 1 else "blocks", "  ".join(row)))
    print()
    print("The longer the detour, the higher the risk must be before rerouting")
    print("pays. With no detour cost at all, any risk above 0 is enough.")
    print()

def main(observation="ALERT", on_the_way=False, truth="clear"):
    result = plan_delivery(observation, verbose=True)
    if on_the_way:
        run_on_the_way(observation)
    d = decide_at(world.START, len(result["route"]) - 1, observation)
    if d["detour"] is not None:
        run_policy_map(d["extra"], d["belief"])

    trip = simulate_trip(observation, on_the_way, truth)
    print(LINE)
    print("THE TRIP - %s, bridge actually %s"
          % ("alert on the way" if on_the_way else "alert before departure", truth.upper()))
    print(LINE)
    driven = set(trip["trip"])
    draw([(result["route"], "x"), (trip["trip"], "*")])
    print("  * cells the courier drove   x planned but not driven")
    print()
    for e in trip["events"]:
        print("  - " + e)
    print()
    print("Delivered in %d minutes (%d blocks%s)"
          % (trip["minutes"], len(trip["trip"]) - 1,
             ", plus %d min stuck" % STUCK_WAIT_MINUTES if trip["stuck"] else ""))
    if trip["action"] == "Reroute":
        print("The courier never reached the bridge. It was in fact %s, which the"
              % truth)
        print("assistant could not know; the decision is judged on the risk, not")
        print("on how this one trip turned out.")
    print(LINE)
