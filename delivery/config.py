"""Model parameters: probabilities, rewards and transitions.

All values are assumptions chosen for the demonstration, not real data.
"""

# Bayesian model of the bridge
PRIOR_BLOCKED = 0.15         # blocked on about 15% of evening rush hours
P_ALERT_IF_BLOCKED = 0.85    # alert service catches 85% of real blockages
P_ALERT_IF_CLEAR = 0.15      # and raises a false alarm 15% of the time

# MDP model
STATES = ["Clear", "Blocked"]
ACTIONS = ["Continue", "Reroute"]
ON_TIME_REWARD = 10          # order delivered on the planned route
STUCK_REWARD = -8            # courier stuck at a blocked bridge, food goes cold
PENALTY_PER_EXTRA_STEP = 2   # every extra block of detour costs 2 points
GAMMA = 0.9

# T[(state, action)] = distribution of the road state at the next decision
TRANSITIONS = {
    ("Clear", "Continue"):   {"Clear": 0.9, "Blocked": 0.1},
    ("Clear", "Reroute"):    {"Clear": 0.9, "Blocked": 0.1},
    ("Blocked", "Continue"): {"Clear": 0.3, "Blocked": 0.7},
    ("Blocked", "Reroute"):  {"Clear": 0.9, "Blocked": 0.1},
}

MINUTES_PER_BLOCK = 2
STUCK_WAIT_MINUTES = 6      # time lost finding the bridge blocked before turning back
