"""Check that the web app computes exactly what the Python package computes.

    python3 app/check_app.py        (needs Node.js)

For every map and all 12 combinations of report, timing and true bridge
state, it compares: the route, the order in which A* expands cells, the BFS
count, the cells actually driven, the delivery time and the decision.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import delivery as fd  # noqa: E402
from delivery import world  # noqa: E402

NODE_SCRIPT = r"""
const L = require(process.argv[2]);
const spec = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const base = { prior: 0.15, hit: 0.85, falseAlarm: 0.15, onTime: 10, stuck: -8, penalty: 2, gamma: 0.9 };
const out = {};
for (const [name, s] of Object.entries(spec)) {
  const r = L.plan(s, { ...base, obs: "ALERT" });
  const b = L.bfs(s.start, s.goal, new Set(s.walls), s.rows, s.cols);
  out[name] = { route: r.route.length - 1, explored: r.log.map(e => e.cell.join(",")).join(" "), bfs: b.expanded.length, trips: {} };
  for (const obs of ["ALERT", "NO ALERT", "NONE"]) for (const way of [false, true]) for (const truth of ["clear", "blocked"]) {
    const t = L.simulateTrip(s, { ...base, obs }, way, truth);
    out[name].trips[[obs, way, truth].join("/")] = [t.trip.map(c => c.join(",")).join(" "), t.minutes, t.action];
  }
}
console.log(JSON.stringify(out));
"""


def main():
    if not shutil.which("node"):
        print("Node.js not found; skipping the app check.")
        return 0
    spec = {}
    for name in fd.MAPS:
        fd.use_map(name)
        spec[name] = {"rows": world.ROWS, "cols": world.COLS, "start": list(world.START),
                      "goal": list(world.GOAL), "bridge": list(world.ALERT_CELL),
                      "walls": ["%d,%d" % w for w in sorted(world.WALLS)]}
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = os.path.join(tmp, "spec.json")
        script_path = os.path.join(tmp, "check.js")
        json.dump(spec, open(spec_path, "w"))
        open(script_path, "w").write(NODE_SCRIPT)
        js = json.loads(subprocess.check_output(
            ["node", script_path, os.path.join(ROOT, "app", "logic.js"), spec_path]))

    ok = True
    for name in fd.MAPS:
        fd.use_map(name)
        path, log = fd.astar(world.START, world.GOAL, world.WALLS)
        _, bexp = fd.bfs(world.START, world.GOAL, world.WALLS)
        j = js[name]
        same_search = (j["route"] == len(path) - 1 and j["bfs"] == len(bexp)
                       and j["explored"] == " ".join("%d,%d" % e[0] for e in log))
        trips = 0
        for obs_js, obs_py in (("ALERT", "ALERT"), ("NO ALERT", "NO ALERT"), ("NONE", None)):
            for way in (False, True):
                for truth in ("clear", "blocked"):
                    t = fd.simulate_trip(obs_py, way, truth)
                    jt = j["trips"]["%s/%s/%s" % (obs_js, str(way).lower(), truth)]
                    trips += jt == [" ".join("%d,%d" % c for c in t["trip"]), t["minutes"], t["action"]]
        ok &= same_search and trips == 12
        print("%-6s search %s, trips matching %d/12" % (name, "match" if same_search else "DIFFER", trips))
    fd.use_map("6x6")
    print("App and Python agree." if ok else "MISMATCH between app and Python.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
