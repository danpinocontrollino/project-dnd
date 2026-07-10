"""Analysis of the Battlecast research grid (see run_grid.mjs).

Three questions, three outputs (all under ../figures/):

1. **Guard calibration** — on the `guard` grid (boss x count x level), fit
   P(win | deathmatch) as a logistic function of the engine's
   rounds-to-kill-party estimate, and compare with the hand-tuned
   survival-physics guard sigma(2.197*ttk - 4.394).
   -> battlecast_guard_fit.png + fitted constants in battlecast_summary.json

2. **DM-mercy gap** — on the `mercy` grid (single SRD monsters), pair each
   simulated deathmatch P(win) with the production model's calibrated
   table-reality P(win) for the same configuration.  The signed gap *is*
   the mercy/selection effect, now measured cell by cell.
   -> battlecast_mercy_gap.png

3. **OOD agreement** — on the `ood` grid (HP/AC-scaled clones), check that
   the model's verdict ordering matches the simulator's.
   -> rows in battlecast_summary.json

Run AFTER run_grid.mjs:  python3 battlecast_bridge/analyze.py
(from the repo root, so the model pickle and modules resolve).
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
from lethality_engine import _GUARD_A as GUARD_SLOPE
from lethality_engine import _GUARD_B as GUARD_INTERCEPT
from lethality_engine import MonsterProfile, encounter_row

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


def to_profile(r: pd.Series) -> MonsterProfile:
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
    engineer = DnDFeatureEngineer(1.5, FEATURE_COLUMNS)
    feats = engineer.fit(raw).transform(raw)
    df["ttk_party"] = feats["rounds_to_kill_party"].values
    df["p_model"] = pipe.predict_proba(raw[list(RAW_INPUT_COLUMNS)])[:, 1]

    summary: dict = {
        "n_cells": int(len(df)),
        "trials_per_cell": int(df.n_trials.iloc[0]),
    }

    # ── 1. Guard calibration ──────────────────────────────────────────────
    g = df[df.grid == "guard"].copy()
    from sklearn.linear_model import LogisticRegression

    # Weighted logistic fit of simulated outcome on TTK (each cell expands
    # to wins/losses via sample weights -> proper binomial likelihood).
    X = g[["ttk_party"]].to_numpy()
    wins = (g.p_party_win * g.n_trials).to_numpy()
    losses = g.n_trials.to_numpy() - wins
    Xrep = np.vstack([X, X])
    yrep = np.concatenate([np.ones(len(g)), np.zeros(len(g))])
    wrep = np.concatenate([wins, losses])
    lr = LogisticRegression(C=1e6).fit(Xrep, yrep, sample_weight=wrep)
    a_fit, b_fit = float(lr.coef_[0][0]), float(lr.intercept_[0])
    summary["guard_fit"] = {
        "slope": a_fit,
        "intercept": b_fit,
        "hand_tuned_slope": GUARD_SLOPE,
        "hand_tuned_intercept": GUARD_INTERCEPT,
    }
    print(
        f"Guard fit on deathmatch truth: sigma({a_fit:.3f}*ttk {b_fit:+.3f}) "
        f"(hand-tuned: sigma({GUARD_SLOPE}*ttk {GUARD_INTERCEPT:+.3f}))"
    )

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.scatter(
        g.ttk_party,
        g.p_party_win,
        s=14,
        alpha=0.55,
        color=BLUE,
        label="Battlecast deathmatch (guard grid cells)",
    )
    xs = np.linspace(0, min(12, g.ttk_party.max() * 1.05), 200)
    ax.plot(
        xs,
        1 / (1 + np.exp(-(a_fit * xs + b_fit))),
        color=BLUE,
        lw=2,
        label=f"fit: σ({a_fit:.2f}·ttk{b_fit:+.2f})",
    )
    ax.plot(
        xs,
        1 / (1 + np.exp(-(GUARD_SLOPE * xs + GUARD_INTERCEPT))),
        color=RED,
        lw=2,
        ls="--",
        label=f"production guard: σ({GUARD_SLOPE}·ttk{GUARD_INTERCEPT:+.2f})",
    )
    ax.set_xlabel("Estimated rounds to kill the party (TTK)")
    ax.set_ylabel("P(party wins) — simulated fight-to-the-death")
    ax.set_title("Survival guard vs Battlecast deathmatch truth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "battlecast_guard_fit.png"), bbox_inches="tight")
    plt.close(fig)

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
