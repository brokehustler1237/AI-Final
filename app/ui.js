// Same maps as MAPS in food_delivery.py.  S restaurant, C customer, # closed, = river.
const MAP_TEXT = {
 "6x6": {
  "rows": [
   "...#..",
   "..#..C",
   "......",
   "S.#...",
   "......",
   "......"
  ],
  "bridge": [
   2,
   2
  ]
 },
 "10x10": {
  "rows": [
   "....=.....",
   ".##....##C",
   "....=.....",
   "..#.=.##..",
   "..#.=.....",
   "....=..#..",
   "S...=..#..",
   ".##.=.....",
   ".........#",
   "....=....."
  ],
  "bridge": [
   1,
   4
  ]
 },
 "14x14": {
  "rows": [
   "......=.......",
   ".###..=..##...",
   ".#............",
   ".#....=....##.",
   "......=......C",
   "..##..=..#....",
   "......=..#....",
   "..##.....###..",
   "......=.......",
   "S##...=...##..",
   "......=.......",
   "..#...=..#....",
   "..#......#..#.",
   "......=......."
  ],
  "bridge": [
   7,
   6
  ]
 }
};

function parseMap(name) {
  const m = MAP_TEXT[name], rows = m.rows.length, cols = m.rows[0].length;
  const s = { name, rows, cols, start: null, goal: null, bridge: m.bridge.slice(), walls: [], water: [] };
  m.rows.forEach((line, r) => [...line].forEach((ch, c) => {
    if (ch === "S") s.start = [r, c];
    if (ch === "C") s.goal = [r, c];
    if (ch === "#" || ch === "=") s.walls.push(key(r, c));
    if (ch === "=") s.water.push(key(r, c));
  }));
  return s;
}
let mapName = "6x6";
const DEFAULT_MAP = parseMap("6x6");
const DEFAULT_PARAMS = { prior: 0.15, hit: 0.85, falseAlarm: 0.15, onTime: 10, stuck: -8, penalty: 2, gamma: 0.9,
                         obs: "ALERT", timing: "depart", truth: "random" };
const MIN_PER_BLOCK = 2;
const CELL = 72, PAD = 28;
const SCALED = new Set(["search", "pre", "drive"]);
const DELAY = { search: 520, route: 1500, pre: 650, alert: 2600, decide: 3000, drive: 600, cross: 1800,
                stuck: 2600, replan: 2200, done: 0 };

let scen = JSON.parse(JSON.stringify(DEFAULT_MAP));
let params = { ...DEFAULT_PARAMS };
let roll = Math.random();
let res, frames = [], fi = -1, frame = { phase: "ready" }, timer = null, speed = 1, showBfs = false;

const $ = s => document.querySelector(s);
const ccx = c => PAD + c * CELL + CELL / 2;
const ccy = r => PAD + r * CELL + CELL / 2;
const same = (a, b) => !!a && !!b && a[0] === b[0] && a[1] === b[1];
const cs = c => "(" + c[0] + ", " + c[1] + ")";
const num = (x, d = 2) => { const s = Math.abs(x).toFixed(d); return (x < 0 && Number(s) !== 0 ? "−" : "") + s; };
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const blocks = p => p.length - 1;
const plural = (n, w) => n + " " + w + (n === 1 ? "" : "s");

function delayOf(phase) {
  const base = (DELAY[phase] || 600) / speed;
  return SCALED.has(phase) ? base * Math.max(0.42, 6 / scen.rows) : base;
}

/* ---------- model ---------- */

function compute() {
  const r = plan(scen, params);
  const walls = new Set(scen.walls);
  r.bfs = r.route ? bfs(scen.start, scen.goal, walls, scen.rows, scen.cols) : null;
  r.grid = policyGrid(params);
  if (r.route) {
    r.truthBlocked = params.truth === "blocked" || (params.truth === "random" && roll < r.bayes.posterior);
    r.trip = simulateTrip(scen, params, params.timing === "way", r.truthBlocked ? "blocked" : "clear");
    r.dec = r.trip.decision || null;
  }
  return r;
}

function buildFrames() {
  const F = [];
  res.log.forEach((e, i) => F.push({ phase: "search", stage: 0, i, j: 0 }));
  if (!res.route) { F.push({ phase: "done", stage: 5, j: 0 }); return F; }
  F.push({ phase: "route", stage: 1, j: 0 });
  const t = res.trip;
  if (t.offRoute) {
    for (let j = 1; j < t.trip.length; j++) F.push({ phase: "drive", stage: 4, j });
    F.push({ phase: "done", stage: 5, j: t.trip.length - 1 });
    return F;
  }
  const di = t.decisionIdx;
  for (let j = 1; j <= di; j++) F.push({ phase: "pre", stage: 1, j });
  F.push({ phase: "alert", stage: 2, j: di });
  F.push({ phase: "decide", stage: 3, j: di });
  for (let j = di + 1; j < t.trip.length; j++) {
    if (t.stuck && j === t.stopIdx + 1) {
      F.push({ phase: "stuck", stage: 4, j: t.stopIdx });
      if (t.replan) F.push({ phase: "replan", stage: 4, j: t.stopIdx });
    }
    const crossing = !t.stuck && t.action === "Continue" && j === t.bridgeIdx;
    F.push({ phase: crossing ? "cross" : "drive", stage: 4, j });
  }
  if (t.stuck && t.stopIdx === t.trip.length - 1) F.push({ phase: "stuck", stage: 4, j: t.stopIdx });
  if (t.stuck && t.stopIdx === di && !F.some(f => f.phase === "stuck")) {
    F.push({ phase: "stuck", stage: 4, j: di });
  }
  F.push({ phase: "done", stage: 5, j: t.trip.length - 1 });
  return F;
}

function recompute(reroll = false) {
  stop();
  if (reroll) roll = Math.random();
  res = compute();
  frames = buildFrames();
  fi = -1;
  frame = { phase: "ready", stage: 4.5, j: 0 };
  renderAll();
  if (typeof stopShift === "function") { stopShift(); shift = null; renderShift(); }
}

/* ---------- what is visible in a frame ---------- */

function stageOf() { return frame.stage; }
function revealed() {
  const t = res.trip;
  if (!t || t.offRoute) return false;
  if (frame.phase === "done") return true;
  if (t.stuck) return ["stuck", "replan"].includes(frame.phase) || (frame.phase === "drive" && frame.j > t.stopIdx);
  return t.action === "Continue" && (frame.phase === "cross" || (frame.phase === "drive" && frame.j > t.bridgeIdx));
}
function courierCell() {
  if (!res.route) return scen.start;
  if (frame.phase === "ready" || frame.phase === "search" || frame.phase === "route") return scen.start;
  return res.trip.trip[frame.j];
}

/* ---------- map ---------- */

function polyline(path, cls) {
  if (!path || path.length < 2) return "";
  return `<polyline class="${cls}" points="${path.map(c => ccx(c[1]) + "," + ccy(c[0])).join(" ")}"></polyline>`;
}

function restaurantIcon(c) {
  const x = ccx(c[1]), y = ccy(c[0]);
  return `<g transform="translate(${x - 16},${y - 20})"><rect class="icon-accent" x="1" y="4" width="30" height="8" rx="2"></rect>` +
    `<rect class="icon-body" x="4" y="12" width="24" height="17" rx="1.5"></rect><rect class="icon-fill" x="13" y="18" width="6" height="11"></rect></g>` +
    `<text class="icon-label" x="${x}" y="${y + 24}" text-anchor="middle">Restaurant</text>`;
}
function customerIcon(c) {
  const x = ccx(c[1]), y = ccy(c[0]);
  return `<g transform="translate(${x - 15},${y - 21})"><path class="icon-body" d="M3 14 L15 3 L27 14 V29 H3 Z"></path>` +
    `<rect class="icon-accent" x="12" y="19" width="6" height="10"></rect></g>` +
    `<text class="icon-label" x="${x}" y="${y + 24}" text-anchor="middle">Customer</text>`;
}

let lastAngle = 0;

function courierHeading() {
  if (!res.route) return lastAngle;
  const trip = res.trip.trip;
  const j = ["ready", "search", "route"].includes(frame.phase) ? 0 : frame.j;
  let a, b;
  if (j > 0) { a = trip[j - 1]; b = trip[j]; } else if (trip.length > 1) { a = trip[0]; b = trip[1]; } else return lastAngle;
  const target = Math.atan2(b[0] - a[0], b[1] - a[1]) * 180 / Math.PI;
  const delta = ((target - lastAngle) % 360 + 540) % 360 - 180;
  lastAngle += delta;
  return lastAngle;
}

function elapsedMinutes() {
  const t = res.trip;
  if (!t || ["ready", "search", "route"].includes(frame.phase)) return 0;
  if (frame.phase === "done") return t.minutes;
  let m = frame.j * MIN_PER_BLOCK;
  if (t.stuck && (["stuck", "replan"].includes(frame.phase) || frame.j > t.stopIdx)) m += STUCK_WAIT_MIN;
  return m;
}

function clockText(min) {
  const total = 18 * 60 + min;
  return String(Math.floor(total / 60) % 24).padStart(2, "0") + ":" + String(total % 60).padStart(2, "0");
}

function renderClock() {
  const r = res, t = r.trip, st = stageOf();
  $("#clock-now").textContent = clockText(elapsedMinutes());
  let eta = null, late = false;
  if (r.route && st >= 1) {
    eta = blocks(r.route) * MIN_PER_BLOCK;
    if (t && !t.offRoute && st >= 3 && t.action === "Reroute") eta = t.minutes;
    if (t && t.stuck && (["stuck", "replan", "done"].includes(frame.phase) || (frame.phase === "drive" && frame.j > t.stopIdx))) { eta = t.minutes; late = true; }
    if (frame.phase === "ready" && t) eta = t.action === "Reroute" ? t.minutes : blocks(r.route) * MIN_PER_BLOCK;
  }
  $("#clock-eta").textContent = eta == null ? "--:--" : clockText(eta);
  $("#eta-box").classList.toggle("late", late);
}

function renderMap() {
  const st = stageOf(), r = res, t = r.trip;
  const walls = new Set(scen.walls);
  const shownLog = frame.phase === "search" ? r.log.slice(0, frame.i + 1) : r.log;
  const explored = new Map(shownLog.map(e => [key(...e.cell), e]));
  const cur = frame.phase === "search" ? r.log[frame.i] : null;
  const frontier = new Set(cur ? cur.frontier.map(c => key(...c)) : []);
  const bfsSet = new Set(showBfs && r.bfs ? r.bfs.expanded.map(c => key(...c)) : []);
  const reportShown = st >= 2 && t && !t.offRoute;
  const rev = revealed();
  let base = "", roads = "", over = "";
  const water = new Set(scen.water || []);
  const svg = $("#map");
  svg.setAttribute("viewBox", `0 0 ${PAD + scen.cols * CELL + 12} ${PAD + scen.rows * CELL + 12}`);
  svg.classList.toggle("big", scen.rows > 6);
  const blockedAt = (r, c) => r < 0 || c < 0 || r >= scen.rows || c >= scen.cols || walls.has(key(r, c));
  const wet = (r, c) => water.has(key(r, c));

  for (let c = 0; c < scen.cols; c++) base += `<text class="axis" x="${ccx(c)}" y="${PAD - 10}" text-anchor="middle">${c}</text>`;
  for (let rr = 0; rr < scen.rows; rr++) base += `<text class="axis" x="${PAD - 13}" y="${ccy(rr) + 4}" text-anchor="middle">${rr}</text>`;

  for (let rr = 0; rr < scen.rows; rr++) for (let c = 0; c < scen.cols; c++) {
    const k = key(rr, c), x0 = PAD + c * CELL, y0 = PAD + rr * CELL, cell = [rr, c];
    const isBridge = same(cell, scen.bridge);
    let cls = "cell";
    if (water.has(k)) cls += " water";
    else if (walls.has(k)) cls += " wall";
    else {
      if (explored.has(k)) cls += " explored";
      if (isBridge && rev) cls += r.truthBlocked ? " revealed-blocked" : " revealed-clear";
      else if (isBridge && reportShown && params.obs === "ALERT") cls += " bridge-alert";
      else if (isBridge && reportShown && params.obs === "NO ALERT") cls += " bridge-clear";
      if (frontier.has(k)) cls += " frontier";
      if (cur && same(cur.cell, cell)) cls += " current";
      // roads: link this cell to its open right and lower neighbours
      if (c + 1 < scen.cols && !walls.has(key(rr, c + 1))) roads += `<line class="road" x1="${ccx(c)}" y1="${ccy(rr)}" x2="${ccx(c + 1)}" y2="${ccy(rr)}"></line>`;
      if (rr + 1 < scen.rows && !walls.has(key(rr + 1, c))) roads += `<line class="road" x1="${ccx(c)}" y1="${ccy(rr)}" x2="${ccx(c)}" y2="${ccy(rr + 1)}"></line>`;
    }
    base += `<rect class="${cls}" data-r="${rr}" data-c="${c}" x="${x0 + 3}" y="${y0 + 3}" width="${CELL - 6}" height="${CELL - 6}" rx="10"></rect>`;
    if (water.has(k)) {
      for (const dy of [22, 40, 58]) over += `<path class="wave" d="M${x0 + 12} ${y0 + dy} q7 -6 14 0 t14 0 t14 0 t14 0"></path>`;
      continue;
    }
    if (walls.has(k)) { over += `<text class="closed" x="${x0 + CELL / 2}" y="${y0 + CELL / 2 + 4}" text-anchor="middle">CLOSED</text>`; continue; }
    // a bridge has the river on both sides (or the river on one side and the map edge on the other)
    const edgeOrWet = (r2, c2) => r2 < 0 || c2 < 0 || r2 >= scen.rows || c2 >= scen.cols || wet(r2, c2);
    const riverNS = edgeOrWet(rr - 1, c) && edgeOrWet(rr + 1, c) && (wet(rr - 1, c) || wet(rr + 1, c));
    const riverEW = edgeOrWet(rr, c - 1) && edgeOrWet(rr, c + 1) && (wet(rr, c - 1) || wet(rr, c + 1));
    if (isBridge || riverNS || riverEW) {
      const vertical = !riverNS && (riverEW || (blockedAt(rr, c - 1) && blockedAt(rr, c + 1)));
      if (vertical) {
        over += `<line class="rail" x1="${x0 + 23}" y1="${y0 + 10}" x2="${x0 + 23}" y2="${y0 + CELL - 10}"></line>`;
        over += `<line class="rail" x1="${x0 + CELL - 23}" y1="${y0 + 10}" x2="${x0 + CELL - 23}" y2="${y0 + CELL - 10}"></line>`;
      } else {
        over += `<line class="rail" x1="${x0 + 10}" y1="${y0 + 23}" x2="${x0 + CELL - 10}" y2="${y0 + 23}"></line>`;
        over += `<line class="rail" x1="${x0 + 10}" y1="${y0 + CELL - 23}" x2="${x0 + CELL - 10}" y2="${y0 + CELL - 23}"></line>`;
      }
    }
    if (isBridge) {
      if (!rev && reportShown && params.obs === "ALERT") {
        over += `<path class="warn" d="M${x0 + CELL - 20} ${y0 + 29} l9 -16 l9 16 z"></path><text class="warn-mark" x="${x0 + CELL - 11}" y="${y0 + 27}" text-anchor="middle">!</text>`;
      }
    }
    if (bfsSet.has(k)) over += `<circle class="bfsdot" cx="${x0 + 13}" cy="${y0 + CELL - 13}" r="4"></circle>`;
    const e = explored.get(k);
    if (e && !same(cell, scen.start) && !same(cell, scen.goal)) {
      over += `<text class="gh" x="${x0 + 9}" y="${y0 + 17}">g${e.g} h${e.h}</text>`;
      over += `<text class="fval" x="${x0 + CELL - 9}" y="${y0 + CELL - 9}" text-anchor="end">f${e.f}</text>`;
    }
  }

  let lines = "";
  if (r.route && st >= 1) {
    const rerouted = t && t.action === "Reroute" && st >= 3.5;
    const abandoned = t && t.stuck && (frame.phase === "replan" || frame.phase === "done" || (frame.phase === "drive" && frame.j > t.stopIdx));
    lines += polyline(r.route, "route" + (rerouted || abandoned ? " faded" : ""));
  }
  const d = r.dec;
  if (d && d.detour && st >= 3) {
    const taken = t.action === "Reroute" && st >= 3.5;
    const cls = "detour" + (taken ? " taken" : (st >= 4 ? " ghost" : ""));
    if (!(t.stuck && !taken && frame.phase !== "decide" && st >= 4)) lines += polyline(d.detour, cls);
  }
  if (t && t.stuck && t.replan && (frame.phase === "replan" || frame.phase === "done" || (frame.phase === "drive" && frame.j > t.stopIdx))) {
    lines += polyline(t.replan, "detour taken");
  }
  if (t && !["ready", "search", "route"].includes(frame.phase)) lines += polyline(t.trip.slice(0, frame.j + 1), "trail");

  let fx = "";
  const bx = ccx(scen.bridge[1]), by = ccy(scen.bridge[0]);
  if (frame.phase === "alert" && params.obs === "ALERT" && t && !t.offRoute) fx += `<circle class="ping" cx="${bx}" cy="${by}" r="30"></circle>`;
  if (frame.phase === "stuck") fx += `<circle class="ping bad" cx="${bx}" cy="${by}" r="30"></circle>`;
  if (rev) {
    const x0 = PAD + scen.bridge[1] * CELL, y0 = PAD + scen.bridge[0] * CELL;
    if (r.truthBlocked) fx += `<line class="xmark" x1="${x0 + 22}" y1="${y0 + 22}" x2="${x0 + CELL - 22}" y2="${y0 + CELL - 22}"></line><line class="xmark" x1="${x0 + CELL - 22}" y1="${y0 + 22}" x2="${x0 + 22}" y2="${y0 + CELL - 22}"></line>`;
    else fx += `<path class="check" d="M${x0 + 22} ${y0 + 37} l9 9 l17 -19"></path>`;
  }
  let badge = "";
  if (frame.phase === "done" && r.route) {
    const gx = ccx(scen.goal[1]), gy = ccy(scen.goal[0]);
    fx += `<circle class="burst" cx="${gx}" cy="${gy}" r="30"></circle>`;
    const byy = scen.goal[0] === 0 ? gy + 44 : gy - 62;
    const bxx = Math.min(Math.max(gx, PAD + 48), PAD + scen.cols * CELL - 48);
    badge = `<rect class="badge-bg" x="${bxx - 44}" y="${byy - 13}" width="88" height="22" rx="11"></rect><text class="badge-tx" x="${bxx}" y="${byy + 3}" text-anchor="middle">DELIVERED</text>`;
  }

  $("#layer").innerHTML = base + roads + over + lines + fx + restaurantIcon(scen.start) + customerIcon(scen.goal) + badge;
  $("#bfs-legend").hidden = !showBfs;

  const pos = courierCell();
  const offset = same(pos, scen.start) || same(pos, scen.goal);
  const x = ccx(pos[1]) + (offset ? 22 : 0), y = ccy(pos[0]) + (offset ? -22 : 0);
  const delay = delayOf(frame.phase);
  const courier = $("#courier");
  courier.style.transition = `transform ${Math.min(0.45, Math.max(0.15, 0.7 * delay / 1000)).toFixed(2)}s ease-in-out`;
  courier.style.transform = `translate(${x}px, ${y}px)`;
  $("#courier-body").style.transform = `rotate(${courierHeading()}deg)`;
  renderClock();
}

/* ---------- narration ---------- */

function caption() {
  const r = res, t = r.trip, d = r.dec, b = r.bayes;
  const f = frame;
  if (!r.route) return ["No route: the customer is cut off by closed roads. Open one of them.", "bad"];
  const left = j => blocks(t.trip) - j;
  switch (f.phase) {
    case "ready": {
      const way = params.timing === "way" ? "The report reaches the courier one block before the bridge. " : "";
      return [`Ready. ${way}Press Play to watch A* search, the traffic report arrive and the decision being made, or step through with →.`, ""];
    }
    case "search": {
      const e = r.log[f.i];
      if (same(e.cell, scen.goal)) return [`A* takes the customer ${cs(e.cell)} off the frontier: f = ${e.f}, so no shorter route can exist. Search finished after ${r.log.length} expansions.`, "good"];
      return [`A* expands ${cs(e.cell)}: g = ${e.g} blocks driven so far, h = ${e.h} blocks still needed at best, f = g + h = ${e.f}. ${plural(e.frontier.length, "cell")} wait on the frontier.`, ""];
    }
    case "route": {
      const onRoute = !t.offRoute;
      return [`Shortest route: ${plural(blocks(r.route), "block")}, about ${blocks(r.route) * MIN_PER_BLOCK} min${onRoute ? `, over the bridge ${cs(scen.bridge)}` : ""}. BFS would have expanded ${r.bfs.expanded.length} cells to find a route of the same length; A* needed ${r.log.length}.`, "good"];
    }
    case "pre": return [`The courier sets off on the planned route. Now at ${cs(t.trip[f.j])}, ${plural(left(f.j), "block")} to go.`, ""];
    case "alert": {
      if (params.obs === "ALERT") return [`Traffic service: ALERT on the bridge ${cs(scen.bridge)}. Bayes' rule: P(blocked) = ${num(params.hit)} × ${num(params.prior)} / ${num(b.evidence, 4)} = ${pct(b.posterior)}, up from ${pct(params.prior)}.`, "bad"];
      if (params.obs === "NO ALERT") return [`Traffic service: no alert for the bridge. Bayes' rule lowers P(blocked) from ${pct(params.prior)} to ${pct(b.posterior)}, because most real blockages would have triggered an alert.`, "good"];
      return [`No traffic report. Without evidence the assistant keeps the prior: P(blocked) = ${pct(params.prior)}.`, ""];
    }
    case "decide": {
      if (!d || !d.detour) return [`There is no way around the bridge from ${cs(t.trip[f.j])}, so the courier has to continue.`, ""];
      return [`A* from ${cs(d.pos)} with the bridge closed: ${plural(blocks(d.detour), "block")}, ${d.extra} more than going on. MDP: EV(Continue) = ${num(d.ev.Continue)}, EV(Reroute) = ${num(d.ev.Reroute)}. Decision: ${d.action.toUpperCase()}.`, d.action === "Reroute" ? "" : "good"];
    }
    case "drive": {
      if (t.offRoute) return [`The bridge is not on the route, so the report changes nothing. Driving: ${cs(t.trip[f.j])}, ${plural(left(f.j), "block")} to go.`, ""];
      const what = t.stuck && f.j > t.stopIdx ? "Taking the new route" : t.action === "Reroute" ? "Taking the detour" : "Continuing on the planned route";
      return [`${what}: ${cs(t.trip[f.j])}, ${plural(left(f.j), "block")} to go.`, ""];
    }
    case "cross": return [`The courier crosses the bridge. It was clear, so continuing paid off this time.`, "good"];
    case "stuck": {
      if (!t.replan) return [`The bridge is blocked and there is no way around. The courier waits ${STUCK_WAIT_MIN} min at ${cs(t.trip[t.stopIdx])} until it clears.`, "bad"];
      return [`The bridge is blocked. The courier loses ${STUCK_WAIT_MIN} min at ${cs(t.trip[t.stopIdx])} before turning back. This is the risk the MDP weighed when it chose to continue.`, "bad"];
    }
    case "replan": return [`A* replans from ${cs(t.trip[t.stopIdx])} with the bridge closed: ${plural(blocks(t.replan), "block")} to the customer.`, ""];
    case "done": {
      if (t.offRoute) return [`Delivered in ${t.minutes} min.`, "good"];
      const truth = r.truthBlocked ? "blocked" : "clear";
      const how = params.truth === "random" ? ` (drawn at random with P = ${pct(b.posterior)})` : "";
      let tail;
      if (t.action === "Reroute") tail = r.truthBlocked
        ? `The bridge was in fact blocked${how}, so the detour saved the courier from getting stuck.`
        : `The bridge was in fact clear${how}. The detour cost ${t.minutes - blocks(r.route) * MIN_PER_BLOCK} extra min, but with the risk at ${pct(b.posterior)} it was still the right bet. A decision is judged on the risk, not on one outcome.`;
      else tail = t.stuck ? `The bridge was blocked${how}. Bad luck: at a ${pct(b.posterior)} risk, continuing was still the better bet on average.`
        : `The bridge was clear${how}.`;
      return [`Delivered in ${t.minutes} min. ${tail}`, t.stuck ? "bad" : "good"];
    }
  }
  return ["", ""];
}

function renderCaption() {
  const [text, tone] = caption();
  $("#caption-text").textContent = text;
  $("#caption").className = "caption" + (tone ? " " + tone : "");
  $("#caption-count").textContent = frame.phase === "ready" ? "Ready" : `Step ${fi + 1} / ${frames.length}`;
}

function renderPhases() {
  const way = params.timing === "way";
  const names = way ? ["Search route", "Set off", "Traffic report", "Decide", "Deliver"] : ["Search route", "Traffic report", "Decide", "Deliver"];
  const idx = { search: 0, route: 0, pre: 1, alert: way ? 2 : 1, decide: way ? 3 : 2, drive: way ? 4 : 3, cross: way ? 4 : 3,
                stuck: way ? 4 : 3, replan: way ? 4 : 3 }[frame.phase];
  const allDone = frame.phase === "done" || frame.phase === "ready";
  $("#phases").innerHTML = names.map((n, i) =>
    `<span class="phase${i === idx ? " on" : ""}${allDone || (idx != null && i < idx) ? " done" : ""}">${i + 1} ${n}</span>`).join("");
}

/* ---------- cards ---------- */

function renderAstar() {
  const r = res, st = stageOf();
  const open = scen.rows * scen.cols - scen.walls.length;
  $("#astar-input").textContent = `${scen.rows}×${scen.cols} map, restaurant ${cs(scen.start)}, customer ${cs(scen.goal)}, ${scen.walls.length} blocked cells${(scen.water || []).length ? ` (${scen.water.length} of them river)` : ""}. Every move costs 1. Heuristic h(n) is the Manhattan distance to the customer.`;
  const rows = frame.phase === "search" ? r.log.slice(0, frame.i + 1) : r.log;
  $("#trace").innerHTML = rows.map((e, i) => {
    const isCur = frame.phase === "search" && i === frame.i;
    return `<tr class="${isCur ? "cur" : same(e.cell, scen.goal) ? "goal" : ""}"><td>${i + 1}</td><td>${cs(e.cell)}</td><td>${e.g}</td><td>${e.h}</td><td>${e.f}</td></tr>`;
  }).join("");
  const w = $("#trace-wrap");
  if (frame.phase === "search") w.scrollTop = w.scrollHeight; else if (frame.phase === "ready") w.scrollTop = 0;

  let out;
  if (!r.route) out = "No route exists. The customer is cut off by closed roads.";
  else if (st < 1) out = `Searching… ${plural(rows.length, "cell")} expanded so far.`;
  else {
    const saved = 1 - r.log.length / r.bfs.expanded.length;
    out = `Route: <b>${plural(blocks(r.route), "block")}</b>, about ${blocks(r.route) * MIN_PER_BLOCK} min. A* expanded <b>${r.log.length}</b> of ${open} open cells; BFS needs <b>${r.bfs.expanded.length}</b> for the same length, so the heuristic saves ${pct(saved, 0)} of the work.`;
    const d = r.dec;
    if (d && d.detour && st >= 3) out += `<br>Detour from ${cs(d.pos)} with the bridge closed (A* run again): <b>${plural(blocks(d.detour), "block")}</b>, ${d.extra} more than going on.`;
    if (r.trip.offRoute) out += `<br>The bridge ${cs(scen.bridge)} is not on this route.`;
  }
  $("#astar-out").innerHTML = out;
  $("#astar-why").textContent = "f = g + h keeps the search moving toward the customer, and because h never overestimates the remaining distance, the first route A* completes is a shortest one.";
}

function renderBayes() {
  const st = stageOf(), r = res, b = r.bayes;
  $("#card-bayes").classList.toggle("pending", st < 2);
  $("#prior").value = params.prior; $("#prior-o").textContent = params.prior.toFixed(2);
  $("#hit").value = params.hit; $("#hit-o").textContent = params.hit.toFixed(2);
  $("#fa").value = params.falseAlarm; $("#fa-o").textContent = params.falseAlarm.toFixed(2);
  const pr = params.prior, hit = params.hit, fa = params.falseAlarm;
  let calc;
  if (params.obs === "ALERT") calc = `Observation: ALERT on the bridge ${cs(scen.bridge)}\nP(alert) = ${hit.toFixed(2)} × ${pr.toFixed(2)} + ${fa.toFixed(2)} × ${(1 - pr).toFixed(2)} = ${b.evidence.toFixed(4)}\nP(blocked | alert) = ${hit.toFixed(2)} × ${pr.toFixed(2)} / ${b.evidence.toFixed(4)} = ${b.posterior.toFixed(4)}`;
  else if (params.obs === "NO ALERT") calc = `Observation: NO ALERT for the bridge ${cs(scen.bridge)}\nP(no alert) = ${(1 - hit).toFixed(2)} × ${pr.toFixed(2)} + ${(1 - fa).toFixed(2)} × ${(1 - pr).toFixed(2)} = ${b.evidence.toFixed(4)}\nP(blocked | no alert) = ${(1 - hit).toFixed(2)} × ${pr.toFixed(2)} / ${b.evidence.toFixed(4)} = ${b.posterior.toFixed(4)}`;
  else calc = `Observation: none\nNo evidence arrived, so P(blocked) stays at the prior ${pr.toFixed(4)}`;
  $("#bayes-calc").textContent = calc;
  const tm = r.dec && r.dec.detour ? r.dec.threshold : null;
  const mark = tm != null && tm > 0 && tm < 1 ? `<span class="tmark" style="left:calc(${tm * 100}% - 1px)"></span>` : "";
  $("#bayes-bars").innerHTML =
    `<div class="barrow"><span>Before the report</span><span class="track"><span class="fill" style="left:0;width:${pr * 100}%"></span>${mark}</span><span class="val">${pct(pr)}</span></div>` +
    `<div class="barrow"><span>After the report</span><span class="track"><span class="fill post" style="left:0;width:${b.posterior * 100}%"></span>${mark}</span><span class="val">${pct(b.posterior)}</span></div>` +
    (mark ? `<div class="tnote">Blue line: the MDP reroutes above ${pct(tm)}</div>` : "");
  let why;
  if (params.obs === "ALERT") why = `One alert moves the risk from ${pct(pr)} to ${pct(b.posterior)}. It is not proof, because ${pct(fa, 0)} of clear bridges also trigger an alert.`;
  else if (params.obs === "NO ALERT") why = `A quiet report lowers the risk from ${pct(pr)} to ${pct(b.posterior)}, since ${pct(hit, 0)} of real blockages would have raised an alert.`;
  else why = `With no report the assistant can only use what it knew before: a ${pct(pr)} chance of a blockage at this hour.`;
  $("#bayes-why").textContent = why;
}

function renderMdp() {
  const st = stageOf(), r = res, d = r.dec;
  $("#card-mdp").classList.toggle("pending", st < 3);
  $("#penalty").value = params.penalty; $("#penalty-o").textContent = params.penalty.toFixed(1);
  $("#gamma").value = params.gamma; $("#gamma-o").textContent = params.gamma.toFixed(2);
  const body = $("#mdp-body");
  if (!r.route) { body.innerHTML = `<div class="lbl">Output</div><div class="out">There is no route, so there is nothing to decide.</div>`; return; }
  if (r.trip.offRoute) { body.innerHTML = `<div class="lbl">Output</div><div class="out">The bridge is not on the shortest route, so the report does not affect this delivery. The courier continues.</div>`; return; }
  if (!d || !d.detour) { body.innerHTML = `<div class="lbl">Output</div><div class="out">There is no way around the bridge from ${cs(r.trip.trip[r.trip.decisionIdx])}, so rerouting is not possible. The courier continues.</div>`; return; }

  const R = d.R, Q = d.Q, V1 = d.V1, belief = d.belief;
  const where = params.timing === "way" ? `The report arrives at ${cs(d.pos)}, ${plural(d.remaining, "block")} from the customer.` : `The report arrives before departure.`;
  let h = `<div class="lbl">Input</div><div>${where} States Clear and Blocked, actions Continue and Reroute, γ = ${params.gamma.toFixed(2)}. The A* detour adds ${plural(d.extra, "block")}, so rerouting is worth ${params.onTime} − ${params.penalty.toFixed(1)} × ${d.extra} = <b>${num(R["Clear|Reroute"], 1)}</b>.</div>`;
  const T = { "Clear|Continue": "Clear 0.9, Blocked 0.1", "Clear|Reroute": "Clear 0.9, Blocked 0.1", "Blocked|Continue": "Clear 0.3, Blocked 0.7", "Blocked|Reroute": "Clear 0.9, Blocked 0.1" };
  h += `<div class="lbl">Steps · one Bellman update</div><table class="q"><thead><tr><th>state</th><th>action</th><th>next state</th><th>R</th><th>Q</th></tr></thead><tbody>`;
  for (const s of ["Clear", "Blocked"]) {
    const best = Q[s + "|Reroute"] > Q[s + "|Continue"] ? "Reroute" : "Continue";
    for (const a of ["Continue", "Reroute"]) {
      const k = s + "|" + a;
      h += `<tr class="${a === best ? "win" : ""}"><td>${s}</td><td>${a}</td><td>${T[k]}</td><td>${num(R[k], 1)}</td><td class="qv">${num(Q[k])}</td></tr>`;
    }
  }
  h += `</tbody></table><pre class="calc" style="margin-top:8px">V1(s) = max R(s,a):  V1(Clear) = ${num(V1.Clear, 1)}, V1(Blocked) = ${num(V1.Blocked, 1)}\nQ(s,a) = R(s,a) + γ × Σ T(s'|s,a) × V1(s')\nEV(Continue) = ${(1 - belief).toFixed(3)} × ${num(Q["Clear|Continue"])} + ${belief.toFixed(3)} × ${num(Q["Blocked|Continue"])} = ${num(d.ev.Continue)}\nEV(Reroute)  = ${(1 - belief).toFixed(3)} × ${num(Q["Clear|Reroute"])} + ${belief.toFixed(3)} × ${num(Q["Blocked|Reroute"])} = ${num(d.ev.Reroute)}</pre>`;
  const one = { Clear: Q["Clear|Reroute"] > Q["Clear|Continue"] ? "Reroute" : "Continue", Blocked: Q["Blocked|Reroute"] > Q["Blocked|Continue"] ? "Reroute" : "Continue" };
  const agree = one.Clear === d.vi.policy.Clear && one.Blocked === d.vi.policy.Blocked;
  h += `<div style="font-size:13px;color:var(--muted);margin-top:6px">Check: full value iteration converges after ${d.vi.sweeps} sweeps to Clear → ${d.vi.policy.Clear}, Blocked → ${d.vi.policy.Blocked}${agree ? ", the same policy as one update." : ", which differs from one update."}</div>`;

  const evs = [d.ev.Continue, d.ev.Reroute];
  const lo = Math.min(0, ...evs), hi = Math.max(0, ...evs), span = hi - lo || 1;
  const bar = (v, cls) => {
    const zero = lo < 0 ? `<span class="zero" style="left:${(0 - lo) / span * 100}%"></span>` : "";
    return `<span class="track"><span class="fill ${cls}" style="left:${(Math.min(0, v) - lo) / span * 100}%;width:${Math.abs(v) / span * 100}%"></span>${zero}</span>`;
  };
  const win = d.action;
  h += `<div class="lbl">Output</div><div class="bars"><div class="barrow"><span>EV(Continue)</span>${bar(d.ev.Continue, "cont" + (win === "Continue" ? "" : " dim"))}<span class="val">${num(d.ev.Continue)}</span></div>` +
       `<div class="barrow"><span>EV(Reroute)</span>${bar(d.ev.Reroute, "rer" + (win === "Reroute" ? "" : " dim"))}<span class="val">${num(d.ev.Reroute)}</span></div></div>`;
  h += `<div class="out" style="margin-top:10px">Decision: <span class="pill ${win.toLowerCase()}">${win.toUpperCase()}</span> · reroute whenever P(blocked) &gt; ${pct(d.threshold)}</div>`;
  const why = win === "Reroute"
    ? `The risk of ${pct(belief)} is above the ${pct(d.threshold)} break-even point, so a sure detour beats gambling on the bridge.`
    : `The risk of ${pct(belief)} is below the ${pct(d.threshold)} break-even point, so the shorter route is worth the small chance of getting stuck.`;
  h += `<div class="why">${why}</div>`;
  body.innerHTML = h;
}

function renderDecisionMap() {
  const { beliefs, grid } = res.grid;
  const W = 520, H = 268, L = 52, T = 14, Rm = 12, B = 46;
  const cw = (W - L - Rm) / beliefs.length, ch = (H - T - B) / grid.length;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Decision for every risk and detour length">`;
  for (let e = 0; e < grid.length; e++) for (let i = 0; i < beliefs.length; i++) {
    const x = L + i * cw, y = T + (grid.length - 1 - e) * ch;
    s += `<rect class="${grid[e][i] === "Reroute" ? "c-rer" : "c-con"}" x="${x}" y="${y}" width="${cw + 0.5}" height="${ch + 0.5}"></rect>`;
  }
  // boundary between continue and reroute
  let path = "";
  for (let e = 0; e < grid.length; e++) {
    const i = grid[e].indexOf("Reroute");
    const x = L + (i < 0 ? beliefs.length : i) * cw, y1 = T + (grid.length - 1 - e) * ch, y2 = y1 + ch;
    path += (e === 0 ? `M${x} ${y2}` : ` L${x} ${y2}`) + ` L${x} ${y1}`;
  }
  s += `<path class="boundary" d="${path}"></path>`;
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    const x = L + (p * (beliefs.length - 1) + 0.5) * cw;
    s += `<text class="tick" x="${x}" y="${H - B + 16}" text-anchor="middle">${p * 100}%</text>`;
  }
  for (let e = 0; e <= 10; e += 2) s += `<text class="tick" x="${L - 8}" y="${T + (grid.length - 1 - e) * ch + ch / 2 + 4}" text-anchor="end">+${e}</text>`;
  s += `<text class="axl" x="${L + (W - L - Rm) / 2}" y="${H - 8}" text-anchor="middle">P(bridge blocked) after the report</text>`;
  s += `<text class="axl" transform="translate(14,${T + (H - T - B) / 2}) rotate(-90)" text-anchor="middle">extra blocks of detour</text>`;
  s += `<text class="lab con" x="${L + 12}" y="${T + 22}">CONTINUE</text>`;
  s += `<text class="lab rer" x="${W - Rm - 12}" y="${H - B - 12}" text-anchor="end">REROUTE</text>`;
  const d = res.dec;
  const showMe = d && d.detour && (stageOf() >= 3);
  if (showMe) {
    const e = Math.min(d.extra, 10);
    const x = L + (d.belief * (beliefs.length - 1) + 0.5) * cw, y = T + (grid.length - 1 - e) * ch + ch / 2;
    s += `<circle class="me-ring" cx="${x}" cy="${y}" r="11"></circle><circle class="me" cx="${x}" cy="${y}" r="6"></circle>`;
  }
  s += `</svg>`;
  $("#dmap").innerHTML = s;
  let note;
  if (!d || !d.detour) note = "The map shows the MDP decision for any risk and detour length with the current rewards and γ.";
  else if (!showMe) note = "The orange dot for this delivery appears at the Decide step.";
  else note = `Orange dot: this delivery, risk ${pct(d.belief)} and a detour of +${d.extra}${d.extra > 10 ? " (off the chart)" : ""}, so ${d.action.toUpperCase()}. Move the sliders or switch “Report arrives” to see the dot and the boundary move.`;
  $("#dmap-note").textContent = note;
}

function renderStrip() {
  const st = stageOf(), r = res;
  const tile = (k, v, s) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  if (!r.route) { $("#strip").innerHTML = tile("Route", "none", "customer unreachable") + tile("Bridge blocked", "–", "") + tile("Decision", "–", "") + tile("Delivered", "–", ""); return; }
  const t = r.trip, d = r.dec;
  const a = st >= 1 ? tile("Planned route", plural(blocks(r.route), "block"), `about ${blocks(r.route) * MIN_PER_BLOCK} min`) : tile("Planned route", "…", "searching");
  const b = t.offRoute ? tile("Bridge blocked", "–", "not on the route")
    : st >= 2 ? tile("Bridge blocked", pct(r.bayes.posterior, 0), params.obs === "NONE" ? "no report, prior only" : params.obs === "ALERT" ? "after the alert" : "after the quiet report")
    : tile("Bridge blocked", "…", "waiting for the report");
  const c = st >= 3 || t.offRoute ? tile("Decision", `<span class="pill ${t.action.toLowerCase()}">${t.action.toUpperCase()}</span>`, d && d.detour ? `threshold ${pct(d.threshold)}` : "no alternative")
    : tile("Decision", "…", "");
  let e;
  if (frame.phase === "done") e = tile("Delivered", `${t.minutes} min`, `${plural(blocks(t.trip), "block")}${t.stuck ? `, ${STUCK_WAIT_MIN} min stuck` : ""}`);
  else if (frame.phase === "ready") e = tile("Delivery", t.action === "Reroute" ? `${blocks(t.trip) * MIN_PER_BLOCK} min` : `${blocks(r.route) * MIN_PER_BLOCK} min`, t.action === "Reroute" ? "planned, via the detour" : "if the bridge is clear");
  else e = tile("Delivery", st >= 1 && frame.j > 0 ? `${plural(blocks(t.trip) - frame.j, "block")}` : "…", st >= 1 && frame.j > 0 ? "still to go" : "");
  $("#strip").innerHTML = a + b + c + e;
}

function renderNotice() {
  const r = res;
  let msg = "";
  if (!r.route) msg = "No route: the customer is cut off. Open one of the closed roads.";
  else if (r.trip.offRoute) msg = "The bridge is not on the shortest route, so the traffic report does not change this delivery. Use the Bridge tool to put it on the route.";
  $("#notice").textContent = msg;
  $("#notice").classList.toggle("show", !!msg);
}

function renderEval() {
  const r = res, box = $("#eval");
  if (r.status !== "ok") { box.innerHTML = `<p class="conclusion">The evaluation runs when the bridge is on the route and a detour exists.</p>`; return; }
  const ev = r.evaluation;
  const vals = ev.map(e => e.expected);
  const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals), span = hi - lo || 1;
  const best = ev.reduce((a, b) => (b.expected > a.expected + 1e-9 ? b : a));
  let h = `<div class="eval-scroll"><table class="evaltab"><thead><tr><th>Policy</th><th>Average reward per delivery</th><th>exact</th><th>simulated</th><th>stuck at bridge</th><th>rerouted</th></tr></thead><tbody>`;
  for (const e of ev) {
    const zero = lo < 0 ? `<span class="zero" style="left:${(0 - lo) / span * 100}%"></span>` : "";
    h += `<tr class="${e === best ? "best" : ""}"><td>${e.name === "Assistant" ? "Assistant (Bayes + MDP)" : e.name}</td>` +
      `<td><span class="track" style="display:block"><span class="${e.name === "Assistant" ? "fill assist" : "fill"}" style="left:${(Math.min(0, e.expected) - lo) / span * 100}%;width:${Math.abs(e.expected) / span * 100}%"></span>${zero}</span></td>` +
      `<td>${num(e.expected)}</td><td>${num(e.simulated)}</td><td>${pct(e.stuck)}</td><td>${pct(e.rerouted)}</td></tr>`;
  }
  h += `</tbody></table></div>`;
  const a = ev[2], c = ev[0];
  let concl = `The assistant cuts orders stuck at the bridge from ${pct(c.stuck)} to ${pct(a.stuck)} while rerouting ${pct(a.rerouted)} of orders.`;
  concl += best.name === "Assistant" ? " It earns the highest average reward of the three policies." : ` With these settings it does no better than “${best.name.toLowerCase()}”, because it always makes the same choice.`;
  box.innerHTML = h + `<p class="conclusion">${concl}</p>`;
}

function renderAll() {
  renderMap(); renderCaption(); renderPhases(); renderAstar(); renderBayes(); renderMdp();
  renderDecisionMap(); renderStrip(); renderNotice(); renderEval();
  $("#play").textContent = timer ? "Pause" : frame.phase === "done" ? "Play again" : fi >= 0 ? "Continue" : "Play demo";
}

/* ---------- playback ---------- */

function stop() { if (timer) { clearTimeout(timer); timer = null; } }
function advance() {
  if (fi >= frames.length - 1) return false;
  fi += 1; frame = frames[fi]; renderAll(); return true;
}
function tick() {
  timer = null;
  if (!advance() || frame.phase === "done") { renderAll(); return; }
  timer = setTimeout(tick, delayOf(frame.phase));
  $("#play").textContent = "Pause";
}
function startOver() {
  if (params.truth === "random") { roll = Math.random(); res = compute(); frames = buildFrames(); }
  fi = -1;
}
function togglePlay() {
  if (timer) { stop(); renderAll(); return; }
  if (frame.phase === "ready" || frame.phase === "done") startOver();
  timer = setTimeout(tick, 0);
  $("#play").textContent = "Pause";
}
function stepOnce() {
  stop();
  if (frame.phase === "ready" || frame.phase === "done") startOver();
  advance();
  renderAll();
}

$("#play").addEventListener("click", togglePlay);
$("#step").addEventListener("click", stepOnce);
$("#skip").addEventListener("click", () => { stop(); fi = frames.length - 1; frame = frames[fi]; renderAll(); });
document.addEventListener("keydown", e => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" && e.target.type === "range") return;
  if (e.key === " " && tag !== "button") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); stepOnce(); }
});

/* ---------- inputs ---------- */

const onRadio = (name, fn) => document.querySelectorAll(`input[name="${name}"]`).forEach(el => el.addEventListener("change", () => fn(el.value)));
onRadio("obs", v => { params.obs = v; recompute(); });
onRadio("timing", v => { params.timing = v; recompute(); });
onRadio("truth", v => { params.truth = v; recompute(v === "random"); });
onRadio("speed", v => { speed = Number(v); });
const bindSlider = (id, prop) => $("#" + id).addEventListener("input", e => { params[prop] = Number(e.target.value); recompute(); });
bindSlider("prior", "prior"); bindSlider("hit", "hit"); bindSlider("fa", "falseAlarm");
bindSlider("penalty", "penalty"); bindSlider("gamma", "gamma");
$("#bfs-toggle").addEventListener("change", e => { showBfs = e.target.checked; renderMap(); });

$("#map").addEventListener("click", e => {
  const t = e.target.closest("rect.cell");
  if (!t) return;
  const cell = [Number(t.dataset.r), Number(t.dataset.c)];
  const k = key(...cell);
  const tool = document.querySelector('input[name="tool"]:checked').value;
  const special = same(cell, scen.start) || same(cell, scen.goal) || same(cell, scen.bridge);
  const walls = new Set(scen.walls);
  if (tool === "wall") {
    if (special) return;
    if (walls.has(k)) { walls.delete(k); scen.water = (scen.water || []).filter(w => w !== k); }
    else walls.add(k);
    scen.walls = [...walls];
  } else {
    if (walls.has(k) || special) return;
    scen[tool] = cell;
  }
  recompute();
});
$("#reset-map").addEventListener("click", () => { scen = parseMap(mapName); recompute(); });
document.querySelectorAll('input[name="size"]').forEach(el => el.addEventListener("change", () => {
  mapName = el.value; scen = parseMap(mapName); recompute(true);
}));

function randomMap() {
  const n = scen.rows, all = [];
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) all.push([r, c]);
  const rnd = k => Math.floor(Math.random() * k);
  const pick = () => all[rnd(all.length)];
  for (let attempt = 0; attempt < 4000; attempt++) {
    const walls = new Set(), water = new Set();
    let start, goal, bridge, rc = -1;
    if (n <= 6) {
      start = pick(); goal = pick();
      if (manhattan(start, goal) < 6) continue;
      while (walls.size < 3) { const w = pick(); if (!same(w, start) && !same(w, goal)) walls.add(key(...w)); }
    } else {
      rc = Math.floor(n / 2) - 1 + rnd(3);
      const nb = n >= 14 ? 3 : 2, crossings = new Set();
      while (crossings.size < nb) crossings.add(rnd(n));
      for (let r = 0; r < n; r++) if (!crossings.has(r)) { walls.add(key(r, rc)); water.add(key(r, rc)); }
      start = [rnd(n), rnd(2)]; goal = [rnd(n), n - 1 - rnd(2)];
      const target = Math.round(n * n / 9);
      while (walls.size - water.size < target) {
        const w = pick();
        if (w[1] !== rc && !same(w, start) && !same(w, goal)) walls.add(key(...w));
      }
    }
    const route = astar(start, goal, walls, n, n).path;
    if (!route || route.length < n + 1) continue;
    if (n <= 6) bridge = route[Math.floor(route.length / 2)];
    else bridge = route.find(c => c[1] === rc);
    if (!bridge || same(bridge, start) || same(bridge, goal)) continue;
    const w2 = new Set(walls); w2.add(key(...bridge));
    const det = astar(start, goal, w2, n, n).path;
    if (!det) continue;
    const extra = det.length - route.length;
    if (extra < 2 || extra > (n <= 6 ? 6 : 8)) continue;
    return { name: "random", rows: n, cols: n, start, goal, bridge, walls: [...walls], water: [...water] };
  }
  return parseMap(mapName);
}
$("#random-map").addEventListener("click", () => { scen = randomMap(); recompute(true); });


/* ---------- evening shift ---------- */

let shift = null, shiftTimer = null, shiftN = 100;

function shiftSetup() {
  const r = res;
  if (!r.route || r.status !== "ok") return null;
  const w2 = new Set(scen.walls); w2.add(key(...scen.bridge));
  const route = r.route, detour = r.detour;
  const stop = route.findIndex(c => same(c, scen.bridge)) - 1;
  const replan = astar(route[stop], scen.goal, w2, scen.rows, scen.cols).path;
  const minutes = {
    "Continue|clear": blocks(route) * MIN_PER_BLOCK,
    "Continue|blocked": (replan ? stop + blocks(replan) : blocks(route)) * MIN_PER_BLOCK + STUCK_WAIT_MIN,
    "Reroute|clear": blocks(detour) * MIN_PER_BLOCK,
    "Reroute|blocked": blocks(detour) * MIN_PER_BLOCK,
  };
  const post = o => bayes(params.prior, params.hit, params.falseAlarm, o).posterior;
  const aAlert = decide(post("ALERT"), r.Q).action, aQuiet = decide(post("NO ALERT"), r.Q).action;
  const pol = [
    { id: "cont", name: "Always continue", act: () => "Continue", exact: r.evaluation[0].expected },
    { id: "rer", name: "Always reroute", act: () => "Reroute", exact: r.evaluation[1].expected },
    { id: "ast", name: "Assistant (Bayes + MDP)", act: o => (o === "ALERT" ? aAlert : aQuiet), exact: r.evaluation[2].expected },
  ];
  pol.forEach(p => Object.assign(p, { sum: 0, mins: 0, stuck: 0, series: [] }));
  const vals = Object.values(r.R);
  return { n: shiftN, i: 0, pol, minutes, R: r.R, ribbon: [], yMin: Math.min(...vals), yMax: Math.max(...vals), alerts: 0, blocked: 0, aAlert, aQuiet };
}

function shiftStep() {
  const s = shift;
  const blocked = Math.random() < params.prior;
  const alert = Math.random() < (blocked ? params.hit : params.falseAlarm);
  const obs = alert ? "ALERT" : "NO ALERT", truth = blocked ? "blocked" : "clear";
  s.i += 1; s.alerts += alert; s.blocked += blocked;
  for (const p of s.pol) {
    const a = p.act(obs);
    p.sum += s.R[(blocked ? "Blocked" : "Clear") + "|" + a];
    p.mins += s.minutes[a + "|" + truth];
    if (a === "Continue" && blocked) p.stuck += 1;
    p.series.push(p.sum / s.i);
    if (p.id === "ast") s.ribbon.push({ cls: a === "Reroute" ? "det" : blocked ? "stk" : "over", alert, blocked, a });
  }
}

function renderShift() {
  const area = $("#shift-area");
  if (!shift) shift = shiftSetup();
  const btn = $("#shift-run");
  if (!shift) {
    btn.disabled = true;
    area.innerHTML = `<p class="shift-note">The shift runs when the bridge is on the route and a detour exists. Reset the map to try it.</p>`;
    return;
  }
  btn.disabled = false;
  btn.textContent = shiftTimer ? "Stop" : shift.i >= shift.n ? "Run again" : shift.i > 0 ? "Continue" : "Run the shift";
  const s = shift;

  // chart
  const W = 760, H = 280, L = 46, Rm = 168, T = 18, B = 42;
  const pw = W - L - Rm, ph = H - T - B;
  const X = i => L + (s.n <= 1 ? 0 : (i / (s.n - 1)) * pw);
  const Y = v => T + (s.yMax - v) / (s.yMax - s.yMin) * ph;
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Running average reward of three strategies">`;
  const step = (s.yMax - s.yMin) > 12 ? 4 : 2;
  for (let v = Math.ceil(s.yMin / step) * step; v <= s.yMax; v += step) {
    g += `<line class="grid" x1="${L}" x2="${L + pw}" y1="${Y(v)}" y2="${Y(v)}"></line><text class="tick" x="${L - 8}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  for (const f of [0, 0.25, 0.5, 0.75, 1]) {
    const i = Math.round(f * (s.n - 1));
    g += `<text class="tick" x="${X(i)}" y="${H - B + 18}" text-anchor="middle">${i + 1}</text>`;
  }
  g += `<text class="axl" x="${L + pw / 2}" y="${H - 6}" text-anchor="middle">delivery number</text>`;
  g += `<text class="axl" transform="translate(13,${T + ph / 2}) rotate(-90)" text-anchor="middle">average reward so far</text>`;
  // exact expected values (dashed) + labels with simple collision avoidance
  const labs = s.pol.map(p => ({ p, y: Y(p.exact) })).sort((a, b) => a.y - b.y);
  for (let k = 1; k < labs.length; k++) if (labs[k].y - labs[k - 1].y < 16) labs[k].y = labs[k - 1].y + 16;
  for (const p of s.pol) g += `<line class="exp k-${p.id}" x1="${L}" x2="${L + pw}" y1="${Y(p.exact)}" y2="${Y(p.exact)}"></line>`;
  for (const p of s.pol) {
    if (p.series.length > 1) g += `<polyline class="run k-${p.id}" style="fill:none" points="${p.series.map((v, i) => X(i).toFixed(1) + "," + Y(v).toFixed(1)).join(" ")}"></polyline>`;
    if (p.series.length) g += `<circle class="k-${p.id}" cx="${X(p.series.length - 1)}" cy="${Y(p.series[p.series.length - 1])}" r="4"></circle>`;
  }
  for (const l of labs) {
    const short = { cont: "Always continue", rer: "Always reroute", ast: "Assistant" }[l.p.id];
    g += `<text class="lab k-${l.p.id}" style="stroke:none" x="${L + pw + 10}" y="${l.y + 4}">${short} ${num(l.p.exact)}</text>`;
  }
  if (s.i === 0) g += `<text class="axl" x="${L + pw / 2}" y="${T + 24}" text-anchor="middle">Dashed lines: exact expected values. Press “Run the shift” to watch the averages settle onto them.</text>`;
  g += `</svg>`;

  // scoreboard
  const best = s.i > 0 ? s.pol.reduce((a, b) => (b.sum > a.sum ? b : a)) : null;
  const board = s.pol.map(p => {
    const avg = s.i ? num(p.sum / s.i) : "–";
    const am = s.i ? (p.mins / s.i).toFixed(1) + " min" : "–";
    return `<div class="brow ${p.id}${s.i >= s.n && p === best ? " win" : ""}"><span class="name">${p.name}</span><span class="big">${avg}</span>` +
      `<span class="meta">stuck ${p.stuck} · avg trip ${am}</span></div>`;
  }).join("");

  const ribbon = s.ribbon.map((x, k) => `<i class="${x.cls}${x.alert ? " alert" : ""}" title="Delivery ${k + 1}: ${x.alert ? "alert" : "no alert"}, bridge ${x.blocked ? "blocked" : "clear"}, assistant ${x.a.toLowerCase()}"></i>`).join("");

  let note = "";
  if (s.i >= s.n) {
    const [c, rr, a] = s.pol;
    note = `After ${s.n} deliveries the bridge was blocked ${plural(s.blocked, "time")}. Always continuing got stuck ${plural(c.stuck, "time")}; the assistant got stuck ${plural(a.stuck, "time")}, because it rerouted whenever an alert came in. ` +
      `Average reward: assistant ${num(a.sum / s.n)}, always continue ${num(c.sum / s.n)}, always reroute ${num(rr.sum / s.n)}. ` +
      `With more deliveries the averages move closer to the dashed lines, which are the exact values from the Python evaluation.`;
  } else if (s.i > 0) note = `${s.i} of ${s.n} deliveries done…`;
  else note = `The assistant reroutes after an alert (${s.aAlert.toLowerCase()}) and otherwise takes the bridge (${s.aQuiet.toLowerCase()}).`;

  area.innerHTML = `<div class="shift-body"><div class="shift-chart">${g}</div><div><div class="counter">Delivery ${s.i} / ${s.n} · bridge blocked ${plural(s.blocked, "time")} · alerts ${s.alerts}</div><div class="board" style="margin-top:8px">${board}</div></div></div>` +
    `<div class="ribbon-wrap"><div class="control-label">Assistant, delivery by delivery</div><div class="ribbon">${ribbon}</div>` +
    `<div class="rlegend"><span><i style="background:var(--good)"></i>over the bridge</span><span><i style="background:var(--detour)"></i>rerouted</span><span><i style="background:var(--bad)"></i>stuck at the blocked bridge</span><span><i style="background:var(--surface-2);box-shadow:inset 0 3px 0 var(--alert)"></i>alert received</span></div></div>` +
    `<p class="shift-note">${note}</p>`;
}

function stopShift() { if (shiftTimer) { clearTimeout(shiftTimer); shiftTimer = null; } }
function shiftTick() {
  shiftTimer = null;
  if (!shift || shift.i >= shift.n) { renderShift(); return; }
  shiftStep();
  shiftTimer = shift.i < shift.n ? setTimeout(shiftTick, Math.min(120, Math.max(16, 6000 / shift.n))) : null;
  renderShift();
}
$("#shift-run").addEventListener("click", () => {
  if (shiftTimer) { stopShift(); renderShift(); return; }
  if (!shift || shift.i >= shift.n) shift = shiftSetup();
  if (!shift) return;
  shiftTimer = setTimeout(shiftTick, 0);
  renderShift();
});
document.querySelectorAll('input[name="shiftn"]').forEach(el => el.addEventListener("change", () => {
  shiftN = Number(el.value); stopShift(); shift = null; renderShift();
}));

recompute();
