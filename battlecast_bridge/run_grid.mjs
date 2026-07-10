// Battlecast research grid — headless Monte Carlo collection.
//
// Drives the vendored Battlecast combat engine (see PROVENANCE.md) through
// three designed grids and writes one JSON line per cell to results.jsonl:
//
//   1. guard  — boss CR × count × party level: fight-to-the-death truth for
//               calibrating the survival-physics guard (the region where
//               FIREBALL data is DM-mercy-contaminated);
//   2. mercy  — single SRD monsters × party levels: paired with the
//               production model's predictions to measure the "DM mercy"
//               gap (table reality vs deathmatch);
//   3. ood    — HP/AC-scaled clones of real monsters: does the engine agree
//               with our out-of-distribution verdicts?
//
// Usage:  node run_grid.mjs [--trials 2000] [--smoke]
//         (--smoke runs 3 cells x 200 trials, for CI-style checks)

import { appendFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// ── headless worker shim ───────────────────────────────────────────────────
let messageHandler = null;
let pending = null;
globalThis.self = {
  addEventListener: (type, fn) => { if (type === "message") messageHandler = fn; },
  postMessage: (msg) => { if (msg.type === "done" && pending) pending(msg); },
};
await import("./vendor/mc-worker-CQHYBfRR.js");
const { bt: MONSTERS } = await import("./vendor/spells-B65qMaNs.js");
const HEROES = await import("./vendor/heroes-BRXrKxCp.js");
const buildHero = HEROES.r; // (className, level) -> statblock

const args = process.argv.slice(2);
const SMOKE = args.includes("--smoke");
const TRIALS = SMOKE ? 200 : parseInt(args[args.indexOf("--trials") + 1] || "2000", 10);
const OUT = fileURLToPath(
  new URL(SMOKE ? "./results_smoke.jsonl" : "./results.jsonl", import.meta.url)
);

const BALANCED = ["Fighter", "Cleric", "Wizard", "Rogue"];

function parseCr(cr) {
  if (typeof cr === "number") return cr;
  const s = String(cr);
  if (s.includes("/")) { const [a, b] = s.split("/"); return Number(a) / Number(b); }
  return Number(s);
}

function statSum(m) {
  const a = m.abilities || {};
  return (a.str ?? 10) + (a.dex ?? 10) + (a.con ?? 10) + (a.int ?? 10) + (a.wis ?? 10) + (a.cha ?? 10);
}

const SIZE_NUM = { Tiny: 1, Small: 2, Medium: 3, Large: 4, Huge: 5, Gargantuan: 6 };

function monsterByCr(target) {
  // closest CR match; prefer plain melee-ish monsters with actions
  let best = null, bestGap = 1e9;
  for (const m of MONSTERS) {
    const cr = parseCr(m.cr);
    if (!m.actions || m.actions.length === 0) continue;
    const gap = Math.abs(cr - target);
    if (gap < bestGap) { best = m; bestGap = gap; }
  }
  return best;
}

async function runCell(redMonsters, blueLevel, trials) {
  const blue = BALANCED.map((c) => ({ data: buildHero(c, blueLevel), count: 1 }));
  const result = await new Promise((resolve) => {
    pending = (msg) => resolve(msg.result);
    messageHandler({
      data: {
        type: "start",
        redMonsters,
        blueMonsters: blue,
        numBattles: trials,
        gridSize: 20,
        teamTactics: undefined,
        terrainBlocked: [],
        terrainSightBlocked: [],
        fixedPlacement: undefined,
      },
    });
  });
  return result;
}

function record(grid, monster, count, level, trials, res, extra = {}) {
  const row = {
    grid,
    monster_name: monster.name,
    monster_cr: parseCr(monster.cr),
    monster_hp: monster.hp,
    monster_ac: monster.ac,
    monster_size_num: SIZE_NUM[monster.size] ?? 3,
    monster_stat_sum: statSum(monster),
    num_monsters: count,
    party_level: level,
    party_size: 4,
    composition: "Balanced",
    n_trials: trials,
    p_party_win: res.blueWins / res.totalBattles,
    p_draw: res.draws / res.totalBattles,
    avg_rounds: res.avgRounds,
    engine: "battlecast-2024srd",
    date: new Date().toISOString().slice(0, 10),
    ...extra,
  };
  appendFileSync(OUT, JSON.stringify(row) + "\n");
  return row;
}

// ── grid definitions ───────────────────────────────────────────────────────
const cells = [];

// 1. guard grid: 5 boss tiers x 6 counts x 6 levels = 180 cells
const BOSS_CRS = [2, 5, 10, 15, 21];
const COUNTS = [1, 2, 4, 8, 12, 19];
const LEVELS = [1, 5, 9, 13, 17, 20];
for (const crTarget of BOSS_CRS)
  for (const count of COUNTS)
    for (const level of LEVELS)
      cells.push({ grid: "guard", monster: monsterByCr(crTarget), count, level });

// 2. mercy grid: 25 monsters spread across the CR range x 4 levels = 100 cells
const sorted = MONSTERS.filter((m) => m.actions?.length).sort(
  (a, b) => parseCr(a.cr) - parseCr(b.cr)
);
const step = Math.max(1, Math.floor(sorted.length / 25));
const mercyMonsters = sorted.filter((_, i) => i % step === 0).slice(0, 25);
for (const m of mercyMonsters)
  for (const level of [3, 7, 11, 15])
    cells.push({ grid: "mercy", monster: m, count: 1, level });

// 3. ood grid: HP/AC-scaled clones at two party levels = 24 cells
const base = monsterByCr(5);
for (const hpMult of [1, 5, 20])
  for (const acBoost of [0, 8])
    for (const count of [1, 4])
      for (const level of [9, 20])
        cells.push({
          grid: "ood",
          monster: {
            ...base,
            name: `${base.name} xHP${hpMult}+AC${acBoost}`,
            hp: base.hp * hpMult,
            ac: base.ac + acBoost,
          },
          count,
          level,
        });

const todo = SMOKE ? cells.filter((c, i) => i % Math.floor(cells.length / 3) === 0).slice(0, 3) : cells;
writeFileSync(OUT, ""); // fresh file per run
console.log(`Battlecast grid: ${todo.length} cells x ${TRIALS} trials -> ${OUT}`);

const t0 = Date.now();
for (let i = 0; i < todo.length; i++) {
  const { grid, monster, count, level } = todo[i];
  // Adaptive precision: many-monster cells cost ~count x per battle, and
  // their win probabilities sit near the extremes where the binomial SE is
  // small — fewer trials lose almost nothing.
  const cellTrials = count <= 4 ? TRIALS : Math.max(400, Math.round((TRIALS * 4) / count));
  const res = await runCell([{ data: monster, count }], level, cellTrials);
  const row = record(grid, monster, count, level, cellTrials, res);
  if (i % 10 === 0 || SMOKE)
    console.log(
      `[${i + 1}/${todo.length}] ${grid}: ${count}x ${monster.name} (CR ${row.monster_cr}) vs L${level} -> P(win) ${row.p_party_win.toFixed(3)}`
    );
}
console.log(`Done in ${((Date.now() - t0) / 60000).toFixed(1)} min.`);
