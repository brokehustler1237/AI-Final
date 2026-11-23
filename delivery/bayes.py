"""Algorithm 2: Bayesian inference about the bridge.
"""

from .config import P_ALERT_IF_BLOCKED, P_ALERT_IF_CLEAR


def bayes_blocked(prior, observation):
    """P(blocked | observation) for observation 'ALERT' or 'NO ALERT'.
    None means no report at all, so the prior is returned unchanged."""
    if observation is None:
        return prior, None
    if observation == "ALERT":
        l_blocked, l_clear = P_ALERT_IF_BLOCKED, P_ALERT_IF_CLEAR
    else:
        l_blocked, l_clear = 1 - P_ALERT_IF_BLOCKED, 1 - P_ALERT_IF_CLEAR
    evidence = l_blocked * prior + l_clear * (1 - prior)
    return l_blocked * prior / evidence, evidence
