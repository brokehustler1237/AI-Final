"""Tests: the command line runs end to end on every map and scenario."""

import contextlib
import io

import pytest

from delivery import world
from delivery.cli import run

SCENARIOS = [
    (m, obs, way, truth)
    for m in ("6x6", "10x10", "14x14")
    for obs in ("alert", "no_alert", "none")
    for way in (False, True)
    for truth in ("clear", "blocked")
]


@pytest.mark.parametrize("map_name,obs,on_the_way,truth", SCENARIOS)
def test_every_scenario_runs_and_delivers(map_name, obs, on_the_way, truth):
    argv = ["--map", map_name, obs, "--truth", truth] + (["--on-the-way"] if on_the_way else [])
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            run(argv)
    finally:
        world.use_map("6x6")
    text = out.getvalue()
    for heading in ("1. A* SEARCH", "2. BAYESIAN INFERENCE", "3. MDP", "4. EVALUATION", "THE TRIP"):
        assert heading in text
    assert "Delivered in" in text
