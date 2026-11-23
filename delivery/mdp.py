"""Algorithm 3: the Continue / Reroute decision as a Markov decision process.
"""

from .config import (ACTIONS, GAMMA, ON_TIME_REWARD, PENALTY_PER_EXTRA_STEP,
                     STATES, STUCK_REWARD, TRANSITIONS)


def rewards(extra_steps):
    reroute = ON_TIME_REWARD - PENALTY_PER_EXTRA_STEP * extra_steps
    return {
        ("Clear", "Continue"): ON_TIME_REWARD,
        ("Clear", "Reroute"): reroute,
        ("Blocked", "Continue"): STUCK_REWARD,
        ("Blocked", "Reroute"): reroute,
    }

def bellman_update(R):
    """V0 = 0, so the first sweep gives V1(s) = max_a R(s, a).
    One more Bellman update gives
        Q(s, a) = R(s, a) + gamma * sum_s' T(s' | s, a) * V1(s')."""
    V1 = {s: max(R[(s, a)] for a in ACTIONS) for s in STATES}
    Q = {}
    for s in STATES:
        for a in ACTIONS:
            future = sum(p * V1[s2] for s2, p in TRANSITIONS[(s, a)].items())
            Q[(s, a)] = R[(s, a)] + GAMMA * future
    return V1, Q

def decide(belief, Q):
    """Expected value of each action under P(Blocked) = belief."""
    ev = {a: (1 - belief) * Q[("Clear", a)] + belief * Q[("Blocked", a)]
          for a in ACTIONS}
    return max(ev, key=ev.get), ev

def threshold(Q):
    """Belief at which Continue and Reroute are worth the same."""
    gap_clear = Q[("Clear", "Continue")] - Q[("Clear", "Reroute")]
    gap_blocked = Q[("Blocked", "Continue")] - Q[("Blocked", "Reroute")]
    return gap_clear / (gap_clear - gap_blocked)

def value_iteration(R, tol=1e-9):
    """Full value iteration, run only to confirm that one Bellman update
    already gives the same policy."""
    V = {s: 0.0 for s in STATES}
    for sweep in range(1, 10000):
        Q = {(s, a): R[(s, a)] + GAMMA * sum(p * V[s2] for s2, p in TRANSITIONS[(s, a)].items())
             for s in STATES for a in ACTIONS}
        newV = {s: max(Q[(s, a)] for a in ACTIONS) for s in STATES}
        if max(abs(newV[s] - V[s]) for s in STATES) < tol:
            break
        V = newV
    policy = {s: max(ACTIONS, key=lambda a: Q[(s, a)]) for s in STATES}
    return V, Q, policy, sweep
