// Same algorithms as food_delivery.py, so the page and the Python agree.

const key = (r, c) => r + "," + c;
const manhattan = (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);

function neighbours(cell, walls, rows, cols) {
  const out = [];
  const moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
  for (const [dr, dc] of moves) {
    const r = cell[0] + dr, c = cell[1] + dc;
    if (r >= 0 && r < rows && c >= 0 && c < cols && !walls.has(key(r, c))) out.push([r, c]);
  }
  return out;
}

// A* with Manhattan heuristic. Frontier ordered by (f, h, insertion order),
// exactly like the Python heap, so both expand cells in the same order.
function astar(start, goal, walls, rows, cols) {
  let counter = 0;
  const h0 = manhattan(start, goal);
  const frontier = [[h0, h0, counter, start]];
  const g = new Map([[key(...start), 0]]);
  const cameFrom = new Map();
  const closed = new Set();
  const log = [];
  const less = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

  while (frontier.length) {
    let best = 0;
    for (let i = 1; i < frontier.length; i++) if (less(frontier[i], frontier[best]) < 0) best = i;
    const [f, h, , cell] = frontier.splice(best, 1)[0];
    const k = key(...cell);
    if (closed.has(k)) continue;
    closed.add(k);
    const frontierNow = frontier.filter(e => !closed.has(key(...e[3]))).map(e => e[3]);
    log.push({ cell, g: g.get(k), h, f, frontier: frontierNow });

    if (cell[0] === goal[0] && cell[1] === goal[1]) {
      const path = [cell];
      let cur = k;
      while (cameFrom.has(cur)) { const prev = cameFrom.get(cur); path.push(prev); cur = key(...prev); }
      return { path: path.reverse(), log };
    }
    for (const nxt of neighbours(cell, walls, rows, cols)) {
      const nk = key(...nxt);
      const ng = g.get(k) + 1;
      if (ng < (g.has(nk) ? g.get(nk) : Infinity)) {
        g.set(nk, ng);
        cameFrom.set(nk, cell);
        const hn = manhattan(nxt, goal);
        counter += 1;
        frontier.push([ng + hn, hn, counter, nxt]);
      }
    }
  }
  return { path: null, log };
}

function bayes(prior, hit, falseAlarm, obs) {
  if (obs === "NONE") return { posterior: prior, evidence: null, lb: null, lc: null };
  const lb = obs === "ALERT" ? hit : 1 - hit;
  const lc = obs === "ALERT" ? falseAlarm : 1 - falseAlarm;
  const evidence = lb * prior + lc * (1 - prior);
  return { posterior: (lb * prior) / evidence, evidence, lb, lc };
}

const STATES = ["Clear", "Blocked"];
const ACTIONS = ["Continue", "Reroute"];
const TRANSITIONS = {
  "Clear|Continue":   { Clear: 0.9, Blocked: 0.1 },
  "Clear|Reroute":    { Clear: 0.9, Blocked: 0.1 },
  "Blocked|Continue": { Clear: 0.3, Blocked: 0.7 },
  "Blocked|Reroute":  { Clear: 0.9, Blocked: 0.1 },
};

function rewards(extra, p) {
  const reroute = p.onTime - p.penalty * extra;
  return {
    "Clear|Continue": p.onTime, "Clear|Reroute": reroute,
    "Blocked|Continue": p.stuck, "Blocked|Reroute": reroute,
  };
}

function bellman(R, gamma) {
  const V1 = {};
  for (const s of STATES) V1[s] = Math.max(...ACTIONS.map(a => R[s + "|" + a]));
  const Q = {};
  for (const s of STATES) for (const a of ACTIONS) {
    const T = TRANSITIONS[s + "|" + a];
    let future = 0;
    for (const s2 of STATES) future += T[s2] * V1[s2];
    Q[s + "|" + a] = R[s + "|" + a] + gamma * future;
  }
  return { V1, Q };
}

function decide(belief, Q) {
  const ev = {};
  for (const a of ACTIONS) ev[a] = (1 - belief) * Q["Clear|" + a] + belief * Q["Blocked|" + a];
  return { action: ev.Reroute > ev.Continue ? "Reroute" : "Continue", ev };
}

function threshold(Q) {
  const gc = Q["Clear|Continue"] - Q["Clear|Reroute"];
  const gb = Q["Blocked|Continue"] - Q["Blocked|Reroute"];
  return gc / (gc - gb);
}

// Small seeded generator so the simulation gives the same numbers every load.
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function evaluate(p, Q, extra, n = 10000, seed = 7) {
  const R = rewards(extra, p);
  const assistant = obs => decide(bayes(p.prior, p.hit, p.falseAlarm, obs).posterior, Q).action;
  const policies = [
    ["Always continue", () => "Continue"],
    ["Always reroute", () => "Reroute"],
    ["Assistant", assistant],
  ];
  const rng = mulberry32(seed);
  const out = [];
  for (const [name, policy] of policies) {
    let expected = 0, stuck = 0, rerouted = 0;
    for (const [state, ps] of [["Blocked", p.prior], ["Clear", 1 - p.prior]]) {
      const pa = state === "Blocked" ? p.hit : p.falseAlarm;
      for (const [obs, po] of [["ALERT", pa], ["NO ALERT", 1 - pa]]) {
        const pr = ps * po, a = policy(obs);
        expected += pr * R[state + "|" + a];
        if (state === "Blocked" && a === "Continue") stuck += pr;
        if (a === "Reroute") rerouted += pr;
      }
    }
    let total = 0;
    for (let i = 0; i < n; i++) {
      const state = rng() < p.prior ? "Blocked" : "Clear";
      const pa = state === "Blocked" ? p.hit : p.falseAlarm;
      const obs = rng() < pa ? "ALERT" : "NO ALERT";
      total += R[state + "|" + policy(obs)];
    }
    out.push({ name, expected, stuck, rerouted, simulated: total / n });
  }
  return out;
}


// Breadth-first search, only to show what A*'s heuristic saves.
function bfs(start, goal, walls, rows, cols) {
  const queue = [start];
  const cameFrom = new Map([[key(...start), null]]);
  const expanded = [];
  while (queue.length) {
    const cell = queue.shift();
    expanded.push(cell);
    if (cell[0] === goal[0] && cell[1] === goal[1]) {
      const path = [cell];
      let cur = cameFrom.get(key(...cell));
      while (cur) { path.push(cur); cur = cameFrom.get(key(...cur)); }
      return { path: path.reverse(), expanded };
    }
    for (const nxt of neighbours(cell, walls, rows, cols)) {
      const nk = key(...nxt);
      if (!cameFrom.has(nk)) { cameFrom.set(nk, cell); queue.push(nxt); }
    }
  }
  return { path: null, expanded };
}

// Full value iteration, only to confirm one Bellman update gives the same policy.
function valueIteration(R, gamma, tol = 1e-9) {
  let V = { Clear: 0, Blocked: 0 }, Q = {}, sweep = 0;
  for (sweep = 1; sweep < 10000; sweep++) {
    Q = {};
    for (const s of STATES) for (const a of ACTIONS) {
      const T = TRANSITIONS[s + "|" + a];
      Q[s + "|" + a] = R[s + "|" + a] + gamma * (T.Clear * V.Clear + T.Blocked * V.Blocked);
    }
    const nV = {};
    for (const s of STATES) nV[s] = Math.max(Q[s + "|Continue"], Q[s + "|Reroute"]);
    const delta = Math.max(...STATES.map(s => Math.abs(nV[s] - V[s])));
    if (delta < tol) break;
    V = nV;
  }
  const policy = {};
  for (const s of STATES) policy[s] = Q[s + "|Reroute"] > Q[s + "|Continue"] ? "Reroute" : "Continue";
  return { V, Q, policy, sweeps: sweep };
}

// The whole decision from wherever the courier is (same as decide_at in Python).
function decideAt(s, p, pos, remaining) {
  const walls2 = new Set(s.walls); walls2.add(key(...s.bridge));
  const det = astar(pos, s.goal, walls2, s.rows, s.cols);
  const belief = bayes(p.prior, p.hit, p.falseAlarm, p.obs).posterior;
  if (!det.path) return { detour: null, belief, action: "Continue", pos, remaining };
  const extra = det.path.length - 1 - remaining;
  const R = rewards(extra, p);
  const b = bellman(R, p.gamma);
  const d = decide(belief, b.Q);
  const vi = valueIteration(R, p.gamma);
  return { detour: det.path, detourLog: det.log, extra, R, V1: b.V1, Q: b.Q, belief, action: d.action, ev: d.ev,
           threshold: threshold(b.Q), vi, pos, remaining };
}

const STUCK_WAIT_MIN = 6;

// What actually happens on the road (same as simulate_trip in Python).
function simulateTrip(s, p, onTheWay, truth) {
  const walls = new Set(s.walls);
  const path = astar(s.start, s.goal, walls, s.rows, s.cols).path;
  if (!path) return null;
  const bi = path.findIndex(c => c[0] === s.bridge[0] && c[1] === s.bridge[1]);
  if (bi < 0) return { trip: path, minutes: blocksOf(path) * 2, action: "Continue", stuck: false, offRoute: true, decisionIdx: 0 };
  const i = onTheWay ? bi - 1 : 0;
  const d = decideAt(s, p, path[i], path.length - 1 - i);
  let trip, wait = 0, stuck = false, stopIdx = null, replan = null;
  if (d.action === "Reroute") {
    trip = path.slice(0, i + 1).concat(d.detour.slice(1));
  } else if (truth === "blocked") {
    stopIdx = bi - 1;
    const walls2 = new Set(s.walls); walls2.add(key(...s.bridge));
    replan = astar(path[stopIdx], s.goal, walls2, s.rows, s.cols).path;
    wait = STUCK_WAIT_MIN; stuck = true;
    trip = replan ? path.slice(0, stopIdx + 1).concat(replan.slice(1)) : path;
  } else {
    trip = path;
  }
  return { trip, minutes: blocksOf(trip) * 2 + wait, action: d.action, stuck, decision: d, decisionIdx: i,
           bridgeIdx: bi, stopIdx, replan, route: path };
}
const blocksOf = path => path.length - 1;

// Decision for every combination of risk and detour length.
function policyGrid(p, maxExtra = 10, steps = 20) {
  const beliefs = Array.from({ length: steps + 1 }, (_, i) => i / steps);
  const grid = [];
  for (let extra = 0; extra <= maxExtra; extra++) {
    const Q = bellman(rewards(extra, p), p.gamma).Q;
    grid.push(beliefs.map(b => decide(b, Q).action));
  }
  return { beliefs, grid };
}

// Whole pipeline for one scenario.
function plan(s, p) {
  const walls = new Set(s.walls);
  const res = { status: "ok" };
  const main = astar(s.start, s.goal, walls, s.rows, s.cols);
  res.route = main.path; res.log = main.log;
  res.bayes = bayes(p.prior, p.hit, p.falseAlarm, p.obs);
  if (!main.path) { res.status = "no-route"; return res; }
  const onRoute = main.path.some(c => c[0] === s.bridge[0] && c[1] === s.bridge[1]);
  if (!onRoute) { res.status = "off-route"; res.action = "Continue"; res.final = main.path; return res; }
  const walls2 = new Set(walls); walls2.add(key(...s.bridge));
  const det = astar(s.start, s.goal, walls2, s.rows, s.cols);
  res.detour = det.path; res.detourLog = det.log;
  if (!det.path) { res.status = "no-detour"; res.action = "Continue"; res.final = main.path; return res; }
  res.extra = det.path.length - main.path.length;
  res.R = rewards(res.extra, p);
  const b = bellman(res.R, p.gamma);
  res.V1 = b.V1; res.Q = b.Q;
  const d = decide(res.bayes.posterior, b.Q);
  res.action = d.action; res.ev = d.ev;
  res.threshold = threshold(b.Q);
  res.final = res.action === "Reroute" ? det.path : main.path;
  res.evaluation = evaluate(p, b.Q, res.extra);
  return res;
}

if (typeof module !== "undefined") {
  module.exports = { astar, bayes, rewards, bellman, decide, threshold, evaluate, plan, key, manhattan, bfs, valueIteration, decideAt, simulateTrip, policyGrid };
}
