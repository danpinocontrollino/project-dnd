# 🎓 Exam Guide — True Lethality Engine

*How to present and defend this project at the Statistical Machine Learning exam.
Every number here is reproducible from the repo (`make retrain`, `make benchmark`,
`python3 gan_ablation.py`) and lives in `figures/`.*

---

## 1. The 60-second pitch

> D&D 5e rates monster difficulty with the Challenge Rating, a hand-made design
> heuristic. We replace it with an **empirical, calibrated estimate of
> P(party wins | encounter)**, learned from **34,907 real combats** (FIREBALL
> dataset, real Discord play). The model is a **Platt-calibrated,
> monotone-constrained XGBoost** on ~45 engineered combat-math features,
> validated with **campaign-grouped cross-validation** (AUC 0.657 [0.638, 0.674]
> holdout, 0.608 ± 0.043 grouped CV). From the calibrated probability we derive
> a product: the *True Lethality Level* — the party level at which a balanced
> party of 4 reaches a target win rate. The interesting statistics is in what
> the data **cannot** say: selection effects (DMs curate fights, players
> retreat) contaminate exactly the region the product queries, so the served
> model is wrapped in three explicit domain guards, each one documented,
> justified and covered by tests.

---

## 2. Pipeline walkthrough (what runs, in order, and why)

```
FIREBALL logs (raw, immutable)                 srd_5e_monsters.json (raw)
        │                                              │
        │  parse_fireball.py                           │  monster_offense.py
        │  · per-turn JSON → encounter-level rows      │  · regex-parse real statblocks:
        │  · outcome labels from HP strings            │    DPR, attack bonus, save DC,
        │  · roster aggregation: weighted means,       │    burst/nova (breath, spells)
        │    maxima for flags, sums for totals         │  · DMG p.274 fallback by CR
        ▼                                              ▼
clean_aggregated_combat_data.csv  ←──────  monster_offense_stats.csv
        │
        │  initial_learn.py
        │  · DnDFeatureEngineer (in-pipeline, serialized): hit-chance geometry,
        │    time-to-kill ratio, XP budget × DMG multiplier, trait interactions
        │  · Optuna tuning under StratifiedGroupKFold (study persisted, SQLite)
        │  · CalibratedClassifierCV (sigmoid/Platt, group-aware folds)
        │  · bootstrap 95% CIs; run appended to figures/experiments.jsonl
        ▼
true_lethality_model.pkl  (+ version-portable native JSON booster)
        │
        │  lethality_engine.py  (single inference core for app + CLI)
        │  · binary search for the target-win-rate level
        │  · 3 guards: monotone/OOD-clipping · roster dominance · survival physics
        │  · by-the-book DMG p.82 estimate for an honest comparison column
        ▼
app.py (Streamlit) / fair_fight_finder.py (CLI)

Quality gates:  tests/ (61 pytest)  +  behavior_suite.py (13 domain axioms)
Experiments:    model_comparison.py (course benchmark) · gan_ablation.py
```

One command reproduces everything: **`make retrain`** (data → train → tests → behavior suite).

---

## 3. The decisions an examiner will probe — and the defense

### 3.1 "Why grouped cross-validation?"
Encounters from the same campaign share party, DM and house rules. Random splits
put siblings in train *and* test → optimistic metrics. We group by campaign
(`StratifiedGroupKFold` / `GroupShuffleSplit` on the source-log ID) at **every**
stage — tuning, model selection, holdout, *and calibration folds*. Random-split
numbers are visibly higher; **that gap is measured leakage, not performance.**

### 3.2 "Your AUC is only 0.65. Is the model any good?"
Three answers. (a) It's an *honest* 0.65 — on unseen campaigns, with a bootstrap
CI [0.638, 0.674] that excludes 0.5 decisively. (b) The label is intrinsically
noisy: outcomes are inferred from chat-log HP strings, and the DM's hidden
adjustments are irreducible noise — the Bayes error here is high. (c) The product
consumes *calibrated probabilities*, not rankings: Brier 0.139 [0.134, 0.145]
and a near-diagonal reliability curve are the metrics that matter, and both are
reported per run in `figures/experiments.jsonl`.

### 3.3 "Why XGBoost? Did you try simpler models?" ⭐ the best story
Yes — and the simple model *won the metric and lost the job*. Under identical
grouped CV (`model_comparison.py`):

| Model | grouped-CV AUC | Brier |
|---|---|---|
| Ridge logistic (ERM + ℓ2) | **0.657 ± 0.031** | **0.136** |
| RFF kernel logistic (Rahimi–Recht) | 0.624 ± 0.013 | 0.142 |
| Monotone XGBoost (production) | 0.614 ± 0.033 | 0.141 |

We promoted the logistic — and it rated **8 Liches as trivial** and a 10,000-HP
monster as beatable. Cause: *DM-curation confounding*. In real logs, fights with
many monsters are mostly weak mobs that parties beat, so the unconstrained
monster-count coefficient comes out **positive**. The app asks **interventional**
questions ("same monster, more of them"), not observational ones; only the
monotone-constrained model answers them sanely. We accepted ~0.04 AUC for
decision-grade counterfactuals — prediction ≠ decision. (Constraints are 45
signed priors: party level ↑ ⇒ P(win) never ↓; enemy DPR ↑ ⇒ never ↑.)

### 3.4 "Why Platt (sigmoid) and not isotonic calibration?"
Isotonic produces a piecewise-constant map: the win-probability curve became a
staircase, and the binary search collapsed distinct encounters onto the same
plateau edge (1 Lich and 2 Liches got identical levels). Sigmoid is smooth,
strictly monotone, **and** measured slightly better (Brier 0.1394 vs 0.1397).
Calibration folds are group-aware for the same leakage reason as everywhere else.

### 3.5 "What are the three guards, and why are they legitimate?"
The model interpolates the data; the guards handle where data cannot go:
1. **OOD clipping + monotone constraints** — inputs snapped to legal 5e bands;
   extrapolation direction forced by domain priors (a 10,000-HP homebrew
   degrades gracefully to "beyond deadly").
2. **Roster dominance** — count-weighted averages dilute (Lich + 6 goblins has
   a lower avg CR than the Lich alone). Axiom: adding monsters can never make
   the fight easier ⇒ score every homogeneous sub-roster, take the min.
3. **Survival physics** — the key discovery: in fights where deterministic
   combat math says the party dies in ≤1 round, the logs still show **84.5%
   wins** — *DM mercy* (fudged dice, retreats, rescues). No model trained on
   that can answer "fight 19 Liches to the death". The guard caps
   P(win) ≤ σ(2.197·TTK_party − 4.394): 1 round ⇒ ≤10%, 2 ⇒ ≤50%, 3 ⇒ ≤90%,
   inert (>99.8%) wherever the party survives 5+ rounds — i.e. wherever the
   data is trustworthy. It's a declared, tested modeling assumption — a prior
   over a region with no usable data — not a learned parameter.

### 3.6 "The data is 83% wins. Did you rebalance?" ⭐ second-best story
We tested it properly (`gan_ablation.py`, leakage-clean: CTGAN fits only on
training-campaign losses; both variants evaluated on the same real held-out
campaigns):

| Training data | AUC | Brier | mean p̂ vs true base rate |
|---|---|---|---|
| Real only (production) | **0.657** | **0.139** | 0.83 vs 0.82 ✓ |
| Real + CTGAN synthetic losses | 0.638 | 0.186 | 0.61 vs 0.82 ✗ |

Balancing shifts the training prior to ~45%, deflating every probability by
~20 points — calibration destroyed, and discrimination *also* dropped. Class
imbalance is a property of reality here, not a defect: the base rate IS the
signal that DMs curate encounters. (Course link: ERM with the log-loss is a
proper scoring rule — it estimates P(y|x) under the *training* distribution;
change that distribution and you change what you estimate.)

### 3.7 "Is there distribution shift between train and test campaigns?"
Tested with a kernel two-sample test (unbiased MMD², RBF kernel, median
heuristic, permutation null): MMD² = 0.0003, p ≈ 0.14 → **no detectable
covariate shift**. So the grouped-CV/holdout gap is campaign-level *outcome*
variance (party skill, DM style), not feature shift — which is precisely the
random effect that grouping controls for.

### 3.8 "Where do kernels and GPs appear?" (course toolbox)
- **RFF kernel logistic** (§3.3): a Mercer-kernel machine made scalable to
  n≈28k with Random Fourier Features — exact kernel methods are O(n²–n³).
- **MMD test** (§3.7).
- **GP regression** for the secondary CR-predictor task (n=797 official
  monsters — exactly GP-sized): RBF+White kernel, marginal-likelihood fitting;
  beats XGBoost (MAE 0.88 vs 0.90) and its 95% predictive intervals achieve
  **0.95 empirical coverage** — calibrated uncertainty, verified.

### 3.9 "How do you know the *code* is right, not just the metrics?"
Two test layers: 61 pytest unit tests (statblock parsing regexes, feature math,
DMG book math, the physics guard) and `behavior_suite.py` — 13 behavioral
axioms the served model must satisfy (level monotonicity, count monotonicity,
roster dominance, tier ordering, boundary verdicts, OOD via the real app flow).
The behavior suite exists because a model once *passed all metrics and failed
all of these* (§3.3).

---

## 4. Key numbers to memorize

| Quantity | Value |
|---|---|
| Encounters / campaigns | 34,907 / 1,462 |
| Base rate P(win) | 0.83 |
| Holdout AUC (grouped) | **0.657** [0.638, 0.674] |
| Grouped 4-fold CV AUC | 0.608 ± 0.043 |
| Brier / log-loss | 0.139 [0.134, 0.145] / 0.447 |
| Ridge logistic CV AUC (rejected anyway) | 0.657 |
| GAN-balanced: Brier / mean p̂ | 0.186 / 0.61 (vs 0.82 true) |
| MMD² campaign shift | 0.0003, p ≈ 0.14 (no shift) |
| GP CR predictor | MAE 0.88, R² 0.95, coverage 0.95 |
| DM mercy (hopeless fights won) | 84.5% |
| Top SHAP features | party level, monster DPR, burst/PC-HP, action economy |
| XGBoost (Optuna) | 300 trees, depth 4, lr 0.149, subsample 0.68 |

---

## 5. Live demo script (2 minutes)

1. **Official tab → Lich** — point at the banner: *"DPR 45, burst 100: parsed
   from the real statblock text; burst 100 is Power Word Kill found by scanning
   the spell list."*
2. **Calculate** — the three metrics: book CR 21, True Lethality Level ~3–4
   with its exact win %, the P(win) 1→20 range. *"The discrepancy is the
   thesis: real parties beat liches far below CR 21."*
3. **Set count to 6** — the book column now shows the **DMG-adjusted CR** with
   the ×-multiplier arithmetic; the model's level rises monotonically.
4. **Set count to 19** — *"beyond deadly": the survival-physics guard, because
   the data here is DM-mercy-contaminated.*
5. **Sidebar** — model card (honest grouped metrics) and the target-win-rate
   slider (*"the product is a threshold on a calibrated probability"*).
6. **Encounter Builder** — Lich + 6 goblins: dominance guard keeps it at least
   as hard as the Lich alone despite diluted averages.

---

## 6. Honest limitations (say them before they're asked)

- **Selection bias is the elephant**: we observe *played* fights, curated by
  DMs. P(win) is "probability at a real table", not "probability in a fight to
  the death" — the physics guard patches the worst region, it doesn't fix the
  estimand.
- **Label noise**: outcomes inferred from HP strings in chat logs; "Ongoing"
  fights dropped; retreats and mercy are invisible.
- **Party features are coarse** (level, size, 4 role flags — no items, feats,
  spell lists); monster offense for non-SRD monsters is a CR-table prior.
- **AUC 0.65 ceiling**: with these features and this label noise, we're likely
  near the achievable Bayes error; the model ranks and calibrates, it does not
  divine.
- If we continued: hierarchical/mixed-effects modeling of campaigns (random
  intercept per table), per-spell burst modeling, conformal prediction for the
  win-probability intervals.

---

## 7. Repo orientation (post-cleanup, every file is load-bearing)

| File | Role |
|---|---|
| `monster_offense.py` | Statblock parsing + DMG tables + canonical trait extraction |
| `parse_fireball.py` | Raw logs → encounter dataset |
| `initial_learn.py` | Features + training + calibration + CIs + experiment log |
| `lethality_engine.py` | Inference core, 3 guards, DMG book math |
| `app.py` / `fair_fight_finder.py` | Streamlit site / CLI twin |
| `model_comparison.py` | Course benchmark (logistic, RFF, MMD, GP) |
| `gan_trial.py` / `gan_ablation.py` | CTGAN generation / leakage-clean ablation |
| `behavior_suite.py` + `tests/` | 13 axioms + 61 unit tests |
| `figures/` | metrics.json, experiments.jsonl, all plots & study artifacts |
| `Makefile` | `make retrain`, `make benchmark`, `make app`, `make help` |

*Guides: `README.md` (English, technical) · `README_IT.md` (Italian, everything
including game glossary) · this file (exam defense).*
