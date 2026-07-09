# 🐉 D&D True Lethality Engine
> **Predicting the True Danger of Dungeons & Dragons Combats through Machine Learning, Deep Mechanical Traits, & XGBoost**

> 🇮🇹 **Guida completa in italiano** (sito, gioco, definizioni, modello): [README_IT.md](README_IT.md)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.4+-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## 📖 Overview (The "Why")
Dungeons & Dragons 5th Edition (5e) relies on a mathematical system known as Challenge Rating (CR) to balance encounters. However, standard CR math is fundamentally flawed and scales linearly, often resulting in "balanced" encounters turning into Total Party Kills (TPKs) at higher levels. Furthermore, CR ignores deep tactical mechanics like Magic Resistance, Pack Tactics, and CC Immunity.

The **D&D True Lethality Engine** solves this problem by leveraging machine learning. By analyzing ~35,000 real-world D&D combat logs, parsing official monster stat blocks, applying exponential feature engineering, and utilizing an **XGBoost** model with strict monotonicity constraints, this project provides a highly calibrated, predictive tool for Dungeon Masters to balance combats with confidence.

---

## ✨ Key Features
- **⚔️ Real Offensive Stats (Attack Potency Engine)**: `monster_offense.py` parses **Damage Per Round, attack bonus and save DC** straight out of SRD statblock text (`+9 to hit`, `12 (2d6+5)`, Multiattack routines, `DC 14`), with the official DMG p.274 "Monster Statistics by CR" design table as the fallback for non-SRD monsters. The model no longer uses WotC CR as a proxy for how hard a monster hits — DPR is the **#2 most important feature by SHAP value**.
- **🧮 Combat-Math Features**: to-hit probability geometry (party attack bonus vs monster AC, monster attack bonus vs estimated party AC), effective-HP adjustments for resistances/regeneration, and a **time-to-kill lethality ratio** that captures the damage race directly.
- **🤖 XGBoost with Monotonic Constraints**: the model strictly obeys D&D physics (higher party level *must* increase win probability; more monster DPR *must* decrease it) — this doubles as the out-of-distribution guard for absurd homebrew inputs.
- **🎯 Calibrated Probabilities**: sigmoid (Platt) calibration via `CalibratedClassifierCV`, because the product decision — "which party level gives a 65% win rate?" — consumes raw probabilities. Isotonic was evaluated and rejected: its piecewise-constant map collapsed the binary search onto plateau edges (1 Lich and 2 Liches appraised at the same level), while sigmoid measured slightly *better* (Brier 0.1394 vs 0.1397) and keeps the win-rate curve smooth and strictly monotone.
- **🧪 Honest Validation**: Optuna hyperparameter tuning and evaluation under **StratifiedGroupKFold grouped by source campaign**, so scores measure generalization to *unseen campaigns* instead of leaking same-party encounters across splits.
- **🛡️ Deep Mechanical Traits**: 6 monster traits (Physical Resistance, CC Immunity, Magic Resistance, Pack Tactics, Spellcasting, Regeneration) and their relational interactions against party composition (*Legendary Attrition*, *Magic Res vs Arcane*, …).
- **📖 Official DMG XP Difficulty Scaling (fixed, twice)**: adjusted XP = total monster XP × the DMG p.82 encounter multiplier for monster count, **including the p.83 party-size adjustment** (parties of 1–2 shift the multiplier one step up the ladder, 6+ one step down) — the earliest version ignored the number of monsters entirely, the second ignored party size.
- **🎲 Fair Fight Finder**: sweeps 400 hypothetical parties (sizes 3–6 × levels 1–20 × 5 compositions) to recommend the party that lands closest to the target win rate, with explicit verdicts (`trivial` / `ok` / `beyond deadly`).

---

## 📈 Data Engineering & The "Linear Fallacy"
A core discovery of this project is the **Linear Fallacy** of traditional D&D encounter math. Standard calculations treat party power as a linear combination of `Size * Level`. 

Our `DnDFeatureEngineer` transforms this baseline into **True Power**, utilizing an exponential curve (`Size * Level^1.5`). We combined this with **Relational Mechanics** (tracking fragile damage dealers against spellcasters, or healers against legendary actions) and the **Official DMG XP Economy** (encoding the full CR → XP and Level → Threshold lookup tables from the Dungeon Master's Guide) to drastically improve model accuracy and capture the true non-linear scaling of D&D combat.

---

## 🚀 Installation & Setup

To run the True Lethality Engine locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danpinocontrollino/project-dnd.git
   cd project-dnd
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Usage

### 1. Parsing and Aggregating Logs
To join the FIREBALL combat logs with the official D&D 5e monster stat blocks and extract the deep mechanical traits:
```bash
python3 parse_fireball.py
```
*Outputs `clean_aggregated_combat_data.csv`.*

### 2. Training the Model
To train the XGBoost model on the aggregated data (with Optuna tuning, grouped CV, calibration, SHAP analysis):
```bash
python3 initial_learn.py --trials 40 --train-cr-predictor   # full run
python3 initial_learn.py --no-tune                          # fast run, default params
```
*Outputs the self-contained `true_lethality_model.pkl` pipeline, `figures/metrics.json`, and diagnostic plots (SHAP summary, calibration curve, feature importance, win-rate curve).*

*(Optional: Use `gan_trial.py` if you wish to experiment with CTGAN synthetic data generation for class balancing, though this is disabled for the production inference tool to preserve true probability calibration).*

### 3. Using the DM Tools
Web app (interactive win-probability curves, party-size heatmap, homebrew appraiser):
```bash
streamlit run app.py
```
CLI twin:
```bash
python3 fair_fight_finder.py
```
*Both tools share all simulation logic through `lethality_engine.py`, so they can never drift apart.*

---

## 📂 File Structure

```text
📁 D&D True Lethality Engine
├── 📄 monster_offense.py                 # Attack Potency engine: SRD statblock parsing + DMG p.274 CR fallback.
├── 📄 parse_fireball.py                  # ETL: joins combat logs with stats/traits/offense, encounter aggregation.
├── 📄 initial_learn.py                   # ML pipeline: features, Optuna tuning, grouped CV, calibration, SHAP.
├── 📄 lethality_engine.py                # Shared inference core: binary search, party sweeps, win curves.
├── 📄 app.py                             # Streamlit web app (Plotly win curves, size×level heatmap).
├── 📄 fair_fight_finder.py               # CLI twin of the web app.
├── 📄 gan_trial.py                       # Experimental CTGAN synthetic data generation.
├── 📄 true_lethality_model.pkl           # Exported, self-contained calibrated pipeline.
├── 📄 monster_offense_stats.csv          # Per-monster DPR / attack bonus / save DC table (generated).
├── 📄 clean_aggregated_combat_data.csv   # ~35,000 real-world encounters with traits + offense (generated).
├── 📄 Monster Spreadsheet - Official.csv # Master list of official D&D 5e monsters.
├── 📄 srd_5e_monsters.json               # SRD statblocks (source of parsed offensive stats).
└── 📁 figures/                           # SHAP summary, calibration curve, importances, metrics.json.
```

---

## 📊 Results
Under **honest, campaign-grouped validation** (no same-campaign leakage between train and test), the calibrated model reaches ROC-AUC ≈ 0.65 on held-out campaigns (grouped CV 0.61 ± 0.04), Brier score ≈ 0.14, with well-calibrated probabilities (see `figures/calibration_curve.png` — the calibration folds themselves are now campaign-grouped too). Naive random splits report much higher numbers — that gap *is* the leakage, and reporting it honestly is a feature of this project, not a bug.

SHAP analysis (`figures/shap_summary.png`, computed with XGBoost's native TreeSHAP) shows the strongest signals are **party level, monster damage-per-round, action economy (total monsters on the field), and to-hit geometry** — confirming that offensive potency, which official CR compresses into a single number, is a first-class driver of real combat outcomes.

A key empirical finding: real parties win **83% of resolved fights** — DMs curate encounters and players retreat from losing ones — so weak monsters saturate near the empirical win-rate ceiling at every level. The engine reports these as verdicts (`trivial` / `beyond deadly`) rather than pretending a fractional level exists.

### The survivability physics guard (DM-mercy contamination)

The mercy problem cuts deepest where the model needs data most: in fights the damage math says are *hopeless* (party deleted in ≤1 round), real tables still “won” **84.5%** of the time — fudged rolls, retreats relabeled, reinforcements. Trained on that, the raw model rated **19 Liches as beatable by a level-8 party**. No amount of learning fixes this, because the observational data cannot answer the app's counterfactual question (“fight to the death”). The engine therefore applies an explicit physics layer in `predict_win_for_parties`:

```
P(win) ≤ sigmoid(2.197 · rounds_to_kill_party − 4.394)
```

anchored so that a party deleted in 1 round caps at 10%, 2 rounds at 50%, 3 rounds at 90% — and the cap is inert (>99.8%) for any encounter the party survives 5+ rounds, i.e. everywhere the model's training data is trustworthy. The cap uses the same feature math the model trains on, is smooth, monotone in level, and decreasing in roster damage, so every engine guarantee (binary-search validity, count monotonicity, roster dominance) survives. Result: 1 Lich → level 3.25, 4 → 10.75, 8 → 19, **11+ → beyond deadly**. Pinned by `tests/test_survival_guard.py` and `behavior_suite.py` check 3b.

### The model-selection lesson (prediction ≠ decision)

The course benchmark (`model_comparison.py`, campaign-grouped 4-fold CV) produced a result worth a section of its own:

| Model | Grouped CV AUC | Brier |
|---|---|---|
| **Ridge logistic** (Lec 04+05) | **0.656 ± 0.031** | **0.1363** |
| RFF kernel logistic (Lec 06) | 0.624 ± 0.013 | 0.1418 |
| XGBoost, monotone-constrained (production) | 0.614 ± 0.033 | 0.1406 |

The penalized *linear* model wins on observational predictive risk — the engineered combat-math features carry the signal, and the flexible tree model overfits campaign idiosyncrasies ("small is the new big"). **And yet it is deliberately not the production model.** When promoted, it rated 8 Liches as *trivial* and a 10,000-HP monster as beatable: with no shape constraints, the linear fit absorbs DM-curation confounding (in real logs, many-monster fights are weak mobs that parties beat, so the monster-count coefficient comes out *positive*). The app asks **interventional** questions — "same monster, more of them" — and the monotone-constrained XGBoost, though ~0.04 AUC worse observationally, is the only candidate whose counterfactual sweeps respect domain physics. We accept the predictive-risk penalty to buy decision-grade behavior.

Two further course-toolbox results (`figures/course_benchmark.json`):
- **Kernel two-sample test (MMD, Lec 06 pt 3)**: MMD² = 0.00031, permutation p = 0.12 → no detectable covariate shift between training and held-out campaigns; the CV-holdout gap is campaign-level outcome variance, not distribution shift.
- **Gaussian Process CR predictor (Lec 06 pt 4)**: GP (RBF + White kernel) beats XGBoost on the 797-monster CR task — MAE 0.86 vs 0.94, R² 0.954 — with 95% predictive intervals achieving exactly **0.95 empirical coverage**.

---

## 🎓 Course-Concept Map (SL2026)

`model_comparison.py` benchmarks the course's method families against the production model **under the same campaign-grouped cross-validation**, and applies the kernel toolbox where it is the right tool (outputs land in `figures/model_comparison.png` + `figures/course_benchmark.json`):

| Lecture | Concept | Where it lives in this project |
|---|---|---|
| Lec 01–03 | Supervised learning, ERM, predictive risk | Party-win prediction as ERM under the log-loss; grouped CV as an honest estimate of predictive risk on *unseen campaigns* |
| Lec 04 | Decision theory, Bayes classifier, logistic loss | The calibrated model estimates the regression function η(x) = P(win \| x); the "fair fight" target is a decision *threshold* on η, not a hard classification; ridge-logistic benchmark uses the logistic surrogate loss |
| Lec 05 | Penalties, priors, smoothness | ℓ2-penalized logistic baseline; XGBoost's `reg_lambda`/`reg_alpha` are exactly the ℓ2/ℓ1 penalties of the lecture, tuned by Optuna; monotone constraints act as shape priors |
| Lec 06 pt 1 (+ Mairhuber–Curtis) | RKHS, Mercer kernels, data-dependent bases | RBF-kernel machine benchmark; Mairhuber–Curtis motivates the data-centered kernel basis it uses |
| Lec 06 pt 2 | Random feature approximations (Rahimi–Recht) | Exact kernel methods are O(n²)–O(n³) at n≈28k, so the kernel benchmark uses Random Fourier Features + linear logistic model |
| Lec 06 pt 3 | Kernel two-sample test (MMD) | Unbiased MMD² + permutation test between train-campaign and held-out-campaign feature distributions — quantifying the campaign shift that justifies group-aware validation |
| Lec 06 pt 4 | Gaussian Processes | GP regression (RBF + White kernel, marginal-likelihood fitting) as the CR-predictor baseline on n=797 monsters, with 95% predictive-interval coverage as the honesty check |
| — | Calibration | Platt scaling *is* a logistic model fit to classifier scores — the same logistic loss from Lec 04, reused to make probabilities decision-grade |

## 🧪 Testing

The parsing regexes and the feature engineer are covered by a pytest suite:

```bash
python -m pytest tests/ -q          # 40 unit tests: statblock parsing, DMG tables, traits, features
python behavior_suite.py            # behavioral checks on the trained model (dominance, monotonicity, OOD)
```

Notable regression pinned by the suite: versatile weapons ("7 (1d8+3) slashing damage, **or** 8 (1d10+3) ... two hands") used to have both alternatives *summed* into DPR, inflating ~30 SRD monsters (Kraken 127→93, Pit Fiend 120→99, swarm half-HP alternatives, ...). Damage alternatives are now merged with `max`, riders are still summed.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/danpinocontrollino/project-dnd/issues).

## 📝 License
This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.