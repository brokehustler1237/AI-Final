"""Run every test without pytest:  python3 run_tests.py
(With pytest installed, plain `pytest` does the same.)"""

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
for module in ("tests.test_search", "tests.test_bayes", "tests.test_mdp", "tests.test_planner"):
    mod = importlib.import_module(module)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        getattr(mod, name)()
        print("PASS  %s.%s" % (module.split(".")[-1], name))
        passed += 1

# the command line on every map and scenario (36 runs, output hidden)
import contextlib
import io

cli = importlib.import_module("tests.test_cli")
for args in cli.SCENARIOS:
    with contextlib.redirect_stdout(io.StringIO()):
        cli.test_every_scenario_runs_and_delivers(*args)
    passed += 1
print("PASS  test_cli: %d scenarios" % len(cli.SCENARIOS))
print("\nAll %d tests passed." % passed)
