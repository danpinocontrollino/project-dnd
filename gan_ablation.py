"""GAN ablation — does CTGAN class balancing help or hurt?

Answers, with numbers, the question the repository previously left implicit:
*when and where should gan_balanced_combat_data.csv be used?*

Protocol (leakage-clean):
  1. Split REAL campaigns into train/holdout with the exact production
     split (GroupShuffleSplit by campaign, test_size=0.2, random_state=42).
  2. Model A ("real"): the production pipeline trained on the real
     training campaigns only.
  3. Model B ("gan"): the same pipeline trained on real training campaigns
     PLUS the CTGAN synthetic losses (which gan_trial.py, by default, fits
     only on training-campaign losses — so no held-out information leaks
     into the synthetic rows).
  4. Both models are evaluated on the SAME real holdout campaigns:
     discrimination (ROC-AUC, PR-AUC) and calibration (Brier, log-loss,
     mean predicted probability vs. true base rate).

Expected outcome (and the reason the production model does NOT train on
the GAN CSV): balancing shifts the training base rate from ~84.5% wins to
~50%, so Model B's probabilities are systematically deflated — similar
ranking power, badly damaged calibration.  The product decision ("which
party level gives a 65% win rate?") consumes calibrated probabilities,
so calibration is the metric that matters.

Usage:  python3 gan_ablation.py  [--gan-csv gan_balanced_combat_data.csv]
Writes: figures/gan_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from initial_learn import (
    RAW_INPUT_COLUMNS,
    build_model,
    load_params_from_metrics,
    prepare_xy,
)

LOGGER = logging.getLogger(__name__)


def _evaluate(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "mean_predicted_p": float(np.mean(proba)),
    }


def _fit_variant(name, X_tr, y_tr, g_tr, params) -> "Pipeline":
    """Production-identical fit: calibrated pipeline with grouped Platt folds."""
    cal = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
            np.zeros(len(y_tr)), y_tr, g_tr
        )
    )
    pipe = build_model(params, calibrate=True, calibration_cv=cal)
    LOGGER.info(
        "Fitting %-4s variant on %d rows (base rate %.3f)",
        name,
        len(y_tr),
        float(y_tr.mean()),
    )
    pipe.fit(X_tr, y_tr)
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="clean_aggregated_combat_data.csv")
    parser.add_argument("--gan-csv", default="gan_balanced_combat_data.csv")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # ── real data, production split ────────────────────────────────────────
    X, y, groups = prepare_xy(pd.read_csv(args.data))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(splitter.split(X, y, groups))
    X_tr, y_tr, g_tr = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
    train_campaigns = set(g_tr)

    # ── GAN rows: synthetic only (real rows would duplicate X_tr) ─────────
    gan_df = pd.read_csv(args.gan_csv)
    synth = gan_df[gan_df["encounter_id"] == "synthetic"].copy()
    if synth.empty:
        raise SystemExit(
            "No synthetic rows found — regenerate the CSV with the current "
            "gan_trial.py (older versions did not tag encounter_id)."
        )
    Xs = synth.reindex(columns=list(RAW_INPUT_COLUMNS))
    ys = synth["target"].astype(int)
    gs = pd.Series(["synthetic"] * len(synth))

    X_gan = pd.concat([X_tr, Xs], ignore_index=True)
    y_gan = pd.concat([y_tr, ys], ignore_index=True)
    g_gan = pd.concat([g_tr.reset_index(drop=True), gs], ignore_index=True)

    params = load_params_from_metrics(os.path.join(args.figures_dir, "metrics.json"))

    results = {
        "holdout_base_rate": float(y_te.mean()),
        "n_holdout": int(len(y_te)),
        "n_train_real": int(len(y_tr)),
        "n_synthetic": int(len(synth)),
        "variants": {},
    }
    real_pipe = _fit_variant("real", X_tr, y_tr, g_tr, params)
    results["variants"]["real"] = _evaluate(real_pipe, X_te, y_te)
    gan_pipe = _fit_variant("gan", X_gan, y_gan, g_gan, params)
    results["variants"]["gan_balanced"] = _evaluate(gan_pipe, X_te, y_te)

    os.makedirs(args.figures_dir, exist_ok=True)
    out_path = os.path.join(args.figures_dir, "gan_ablation.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    r, g = results["variants"]["real"], results["variants"]["gan_balanced"]
    LOGGER.info("Holdout base rate: %.3f", results["holdout_base_rate"])
    LOGGER.info(
        "real         AUC %.4f | Brier %.4f | mean p %.3f",
        r["roc_auc"],
        r["brier"],
        r["mean_predicted_p"],
    )
    LOGGER.info(
        "gan_balanced AUC %.4f | Brier %.4f | mean p %.3f",
        g["roc_auc"],
        g["brier"],
        g["mean_predicted_p"],
    )
    LOGGER.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
