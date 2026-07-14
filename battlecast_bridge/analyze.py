"""Analysis of the Battlecast grid produced by run_grid.mjs.

Outputs (all in figures/):
- guard grid: logistic fit of P(win|deathmatch) on the two damage-race
  features (rounds the party survives, rounds it needs to kill the roster),
  plotted against the old one-feature guard
  -> battlecast_guard_fit.png, constants in battlecast_summary.json
- mercy grid: simulated deathmatch P(win) vs the model's table-reality
  P(win), cell by cell; the signed gap is the DM-mercy effect
  -> battlecast_mercy_gap.png
- ood grid: rank agreement between model verdicts and the simulator
  -> battlecast_summary.json

Grid cells are rebuilt with the SAME monster profiles the app serves
(bestiary lookup incl. parsed offense). The first version of this script
used bare hp/ac profiles with DMG-band offense defaults, so the guard was
calibrated on features the serving path never produces - that mismatch is
how one Lich ended up rated "fair at level 3.25".

Run from the repo root AFTER run_grid.mjs, so the pickle and imports resolve.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import __main__
import joblib

from initial_learn import DnDFeatureEngineer, FEATURE_COLUMNS, RAW_INPUT_COLUMNS
from lethality_engine import _GUARD_A as PRODUCTION_A
from lethality_engine import _GUARD_B as PRODUCTION_B
from lethality_engine import _GUARD_C as PRODUCTION_C

# Frozen history, kept only as comparison lines in the plot. v1 was my
# hand-tuned guess, v2 the first Battlecast fit - still one-featured, so
# it capped fast wipes but was blind to slow attrition losses (a party
# that survives 4+ rounds but can't chew through a Lich). The production
# constants live in lethality_engine; main() warns if they drift from
# what this script fits (e.g. after regenerating the grid).
V1_HAND_TUNED = (2.197, -4.394)
V2_SURVIVAL_ONLY = (1.6302, -3.9771)
from lethality_engine import (
    MonsterProfile,
    encounter_row,
    load_monster_database,
    profile_from_db_row,
)

__main__.DnDFeatureEngineer = DnDFeatureEngineer

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(os.path.dirname(HERE), "figures")

INK, BLUE, RED, MUTED = "#0b0b0b", "#2a78d6", "#d03b3b", "#898781"


def load_results(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    # Draws (mutual annihilation / stalls) count as non-wins, matching the
    # training labels where only "Party Win" is positive.
    return df


_DB_BY_NAME: dict | None = None


def _bestiary() -> dict:
    global _DB_BY_NAME
    if _DB_BY_NAME is None:
        db = load_monster_database()
        _DB_BY_NAME = {
            str(n).lower(): pd.Series(rec)
            for n, rec in zip(db["Name"], db.to_dict("records"))
        }
    return _DB_BY_NAME


def to_profile(r: pd.Series) -> MonsterProfile:
    """The profile the APP would serve for this grid cell.

    Full bestiary stats (2014 sheet, parsed offense included), because the
    cap has to be calibrated in the feature space the serving path actually
    produces. Battlecast fights the 2024 statblocks, so for rebalanced
    monsters the pairing absorbs the edition gap - documented limitation,
    same one the mercy analysis carries. Monsters missing from the
    spreadsheet (incl. the OOD xHP/xAC clones) keep the grid row's raw
    stats with DMG-band offense defaults.
    """
    rec = _bestiary().get(str(r.monster_name).lower())
    if rec is not None:
        return profile_from_db_row(rec)
    return MonsterProfile(
        cr=float(r.monster_cr),
        hp=float(r.monster_hp),
        ac=float(r.monster_ac),
        size_num=float(r.monster_size_num),
        stat_sum=float(np.clip(r.monster_stat_sum, 60, 250)),
        name=str(r.monster_name),
    )


def model_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        encounter_row(
            to_profile(r),
            avg_party_level=float(r.party_level),
            party_size=4,
            num_monsters=int(r.num_monsters),
            has_healer=1,
            has_tank=1,
            has_arcane=1,
            has_martial_dps=1,
        )
        for _, r in df.iterrows()
    ]
    return pd.DataFrame(rows)


def main() -> int:
    results_path = os.path.join(HERE, "results.jsonl")
    if not os.path.exists(results_path):
        print(
            "results.jsonl not found — run `node battlecast_bridge/run_grid.mjs` first."
        )
        return 1
    df = load_results(results_path)
    print(
        f"Loaded {len(df)} grid cells "
        f"({(df.n_trials * 1).sum():,} simulated battles)."
    )

    pipe = joblib.load(os.path.join(os.path.dirname(HERE), "true_lethality_model.pkl"))
    raw = model_rows(df)
    engineer = DnDFeatureEngineer(
        1.5, list(FEATURE_COLUMNS) + ["rounds_to_kill_monster_raw"]
    )
    feats = engineer.fit(raw).transform(raw)
    df["ttk_party"] = feats["rounds_to_kill_party"].values
    # Unclipped, same as the serving-time guard (the model feature clips
    # at 50, which hides just how unkillable the OOD hp-clones are).
    df["ttk_kill"] = feats["rounds_to_kill_monster_raw"].values
    df["p_model"] = pipe.predict_proba(raw[list(RAW_INPUT_COLUMNS)])[:, 1]

    summary: dict = {
        "n_cells": int(len(df)),
        "max_trials_per_cell": int(df.n_trials.max()),
        "total_battles": int(df.n_trials.sum()),
    }

    # ── 1. Guard calibration ──────────────────────────────────────────────
    # The cap is a logistic in BOTH sides of the damage race:
    #   sigma(A * rounds_party_survives - C * ln(rounds_to_kill_roster) + B)
    # The survival term catches fast wipes; the log-kill term catches slow
    # attrition losses (survive four rounds, never chew through the boss).
    # A one-feature fit misses the second failure mode entirely: on grid
    # cells with survive/kill ratio <= 0.3 the party wins 15% of simulated
    # deathmatches while the old cap allowed up to ~0.95.
    from scipy.optimize import minimize

    # Fit on the guard grid PLUS the OOD clones: the clones populate the
    # "survives forever, kills never" corner (huge unclipped kill rounds)
    # that the guard grid's real bosses can't reach.
    go = df[df.grid.isin(["guard", "ood"])].copy()
    wins = (go.p_party_win * go.n_trials).to_numpy()
    n_tr = go.n_trials.to_numpy()
    tp = go.ttk_party.to_numpy()
    ltk = np.log(go.ttk_kill.to_numpy())

    def _binom_nll(params, X):
        z = np.clip(X @ params[:-1] + params[-1], -30, 30)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-9, 1 - 1e-9)
        return -(wins * np.log(p) + (n_tr - wins) * np.log(1 - p)).sum()

    def _fit(X, bounds):
        best = None
        for x0 in ([1.0] * X.shape[1] + [-2.0], [0.3] * X.shape[1] + [0.5]):
            r = minimize(
                _binom_nll, np.array(x0), args=(X,), bounds=bounds, method="L-BFGS-B"
            )
            if best is None or r.fun < best.fun:
                best = r
        return best

    # sign constraints keep the two guard axioms provable: cap monotone up
    # in party level, monotone down in monster count.
    race = _fit(np.column_stack([tp, -ltk]), [(0, None), (0, None), (None, None)])
    a_fit, c_fit, b_fit = float(race.x[0]), float(race.x[1]), float(race.x[2])
    surv = _fit(np.column_stack([tp]), [(0, None), (None, None)])

    def _cap(a, c, b, tp_, ltk_):
        return 1.0 / (1.0 + np.exp(-(a * tp_ - c * ltk_ + b)))

    cap_new = _cap(a_fit, c_fit, b_fit, tp, ltk)
    cap_v2 = _cap(V2_SURVIVAL_ONLY[0], 0.0, V2_SURVIVAL_ONLY[1], tp, ltk)
    brier_new = float(np.average((cap_new - go.p_party_win) ** 2, weights=n_tr))
    brier_v2 = float(np.average((cap_v2 - go.p_party_win) ** 2, weights=n_tr))
    summary["guard_fit"] = {
        "A_ttk_party": a_fit,
        "C_log_ttk_kill": c_fit,
        "B_intercept": b_fit,
        "grid_brier_race_fit": brier_new,
        "grid_brier_v2_survival_only": brier_v2,
        "production": [PRODUCTION_A, PRODUCTION_C, PRODUCTION_B],
        "v2_survival_only": list(V2_SURVIVAL_ONLY),
        "v1_hand_tuned": list(V1_HAND_TUNED),
        "survival_only_refit": [float(surv.x[0]), float(surv.x[1])],
    }
    print(
        f"Guard fit on deathmatch truth: "
        f"sigma({a_fit:.4f}*survive - {c_fit:.4f}*ln(kill) {b_fit:+.4f}) | "
        f"grid Brier {brier_new:.4f} (survival-only v2: {brier_v2:.4f})"
    )
    # Consistency check: the deployed constants should BE this fit.
    if (
        abs(a_fit - PRODUCTION_A) > 0.05
        or abs(c_fit - PRODUCTION_C) > 0.1
        or abs(b_fit - PRODUCTION_B) > 0.15
    ):
        print(
            f"⚠️  Production guard ({PRODUCTION_A}, {PRODUCTION_C}, "
            f"{PRODUCTION_B}) no longer matches this fit — update "
            f"_GUARD_A/_GUARD_C/_GUARD_B in lethality_engine.py."
        )

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    ratio = tp / go.ttk_kill.to_numpy()
    axl.scatter(ratio, go.p_party_win, s=14, alpha=0.55, color=BLUE)
    axl.set_xscale("log")
    axl.axvline(1.0, color=MUTED, lw=1, ls=":")
    axl.set_xlabel("survive rounds / kill rounds (log scale)")
    axl.set_ylabel("P(party wins) — simulated fight-to-the-death")
    axl.set_title("The damage race decides deathmatches")
    axl.grid(alpha=0.25)

    axr.scatter(
        cap_v2, go.p_party_win, s=14, alpha=0.45, color=RED, label="old survival-only cap"
    )
    axr.scatter(
        cap_new, go.p_party_win, s=14, alpha=0.55, color=BLUE, label="race cap (production)"
    )
    axr.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    axr.set_xlabel("cap value on the same grid cell")
    axr.set_ylabel("simulated P(win)")
    axr.set_title(
        f"Cap vs simulator (Brier {brier_new:.3f} vs {brier_v2:.3f} survival-only)"
    )
    axr.legend(fontsize=8)
    axr.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "battlecast_guard_fit.png"), bbox_inches="tight")
    plt.close(fig)

    # ── 1b. Guard lattice ─────────────────────────────────────────────────
    # The race cap generalizes (it only needs hp/ac/dpr) but the two TTK
    # features can't tell a Lich from a bag of hit points - cells with the
    # same (survive, kill) rounds have opposite simulated outcomes, because
    # AoE spell rotations aren't priced by the damage math. So wherever we
    # HAVE simulated truth, serve it directly: the guard grid becomes a
    # (CR x count x level) lattice the engine trilinearly interpolates,
    # and the final cap is min(race cap, lattice cap). Monotone massage
    # below only ever lowers a cell (cummin over count and CR, cummax over
    # level), so the axioms stay provable and the cap never gets looser.
    g = df[df.grid == "guard"]
    crs = sorted(g.monster_cr.unique())
    counts = sorted(g.num_monsters.unique())
    levels = sorted(g.party_level.unique())
    lat = np.full((len(crs), len(counts), len(levels)), np.nan)
    for _, r in g.iterrows():
        lat[
            crs.index(r.monster_cr),
            counts.index(r.num_monsters),
            levels.index(r.party_level),
        ] = r.p_party_win
    assert not np.isnan(lat).any(), "guard grid has holes - rerun run_grid.mjs"
    lat = np.maximum.accumulate(lat, axis=2)  # level up   -> never harder
    lat = np.minimum.accumulate(lat, axis=1)  # more bodies -> never easier
    lat = np.minimum.accumulate(lat, axis=0)  # higher CR   -> never easier
    lattice_path = os.path.join(HERE, "guard_lattice.json")
    with open(lattice_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "bosses": {
                    float(cr): str(g[g.monster_cr == cr].monster_name.iloc[0])
                    for cr in crs
                },
                "crs": [float(c) for c in crs],
                "counts": [int(c) for c in counts],
                "levels": [int(l) for l in levels],
                "p_win": np.round(lat, 4).tolist(),
            },
            fh,
            indent=1,
        )
    summary["guard_lattice"] = {
        "shape": list(lat.shape),
        "path": os.path.relpath(lattice_path, os.path.dirname(HERE)),
    }
    print(f"Guard lattice {lat.shape} -> {lattice_path}")

    # ── 2. Mercy gap ──────────────────────────────────────────────────────
    m = df[df.grid == "mercy"].copy()
    if m.empty:
        print("Mercy grid empty (partial results file?) — skipping.")
        with open(
            os.path.join(FIGURES, "battlecast_summary.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(summary, fh, indent=2)
        return 0
    m["gap"] = m.p_model - m.p_party_win
    summary["mercy_gap"] = {
        "mean_gap_model_minus_sim": float(m.gap.mean()),
        "corr_model_sim": float(np.corrcoef(m.p_model, m.p_party_win)[0, 1]),
    }
    print(
        f"Mercy grid: mean(p_model - p_sim) = {m.gap.mean():+.3f} | "
        f"corr = {summary['mercy_gap']['corr_model_sim']:.3f}"
    )

    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    sc = ax.scatter(
        m.p_party_win, m.p_model, c=m.monster_cr, cmap="viridis", s=26, alpha=0.8
    )
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.set_xlabel("Battlecast: P(win) in a fight to the death (optimal play)")
    ax.set_ylabel("Model: P(win) at a real table (FIREBALL)")
    ax.set_title("The DM-mercy / selection gap, cell by cell")
    fig.colorbar(sc, label="monster CR")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "battlecast_mercy_gap.png"), bbox_inches="tight")
    plt.close(fig)

    # ── 3. OOD agreement ──────────────────────────────────────────────────
    o = df[df.grid == "ood"].copy()
    if len(o):
        agree = float(np.corrcoef(o.p_model.rank(), o.p_party_win.rank())[0, 1])
        summary["ood_rank_correlation"] = agree
        print(f"OOD grid: Spearman rank agreement model vs sim = {agree:.3f}")

    with open(
        os.path.join(FIGURES, "battlecast_summary.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote figures/battlecast_summary.json + 2 figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
