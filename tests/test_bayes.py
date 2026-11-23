"""Tests: Bayesian inference about the bridge.

Each check uses a method different from the one in the program.
"""

from fractions import Fraction

import delivery as fd


def test_bayes_matches_exact_fractions():
    prior = Fraction(15, 100)
    hit, false_alarm = Fraction(85, 100), Fraction(15, 100)
    exact_alert = hit * prior / (hit * prior + false_alarm * (1 - prior))
    exact_quiet = (1 - hit) * prior / ((1 - hit) * prior + (1 - false_alarm) * (1 - prior))
    assert exact_alert == Fraction(1, 2)
    assert abs(fd.bayes_blocked(0.15, "ALERT")[0] - float(exact_alert)) < 1e-12
    assert abs(fd.bayes_blocked(0.15, "NO ALERT")[0] - float(exact_quiet)) < 1e-12
    assert fd.bayes_blocked(0.15, None)[0] == 0.15
