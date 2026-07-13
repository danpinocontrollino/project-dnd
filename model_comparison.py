"""Benchmarks tying the project back to the course material (SL2026).

Everything below runs under the SAME campaign-grouped CV as the production
model, otherwise the comparison means nothing:

1. **Penalized logistic regression** — ERM with the logistic loss (Lec 04)
   plus an l2/ridge penalty (Lec 05, "penalties, priors and smoothness").
   The linear baseline: how far do the engineered features alone go?

2. **Random-Fourier-Feature kernel logistic regression** — a Mercer-kernel
   machine (Lec 06 pt 1) made scalable with the Rahimi-Recht random
   feature approximation (Lec 06 pt 2): z(x) s.t. <z(x), z(x')> ~ K(x, x')
   for the Gaussian/RBF kernel, then a *linear* logistic model in feature
   space.  Exact kernel methods are O(n^2)-O(n^3) and infeasible at n~28k.

3. **XGBoost** (the production model, uncalibrated here for a fair
   apples-to-apples risk comparison).

Also included:

4. **Kernel two-sample test (MMD)** — Lec 06 pt 3.  Tests H0: the feature
   distribution of *training* campaigns equals that of *held-out*
   campaigns, via the unbiased MMD^2 estimator with an RBF kernel (median
   heuristic bandwidth) and a permutation null.  This quantifies the
   campaign shift that motivates group-aware validation in the first place.

5. **Gaussian Process regression for the CR predictor** — Lec 06 pt 4.
   n=797 official monsters is exactly GP-sized; RBF + White kernel with
   marginal-likelihood hyperparameter fitting, compared to XGBoost on MAE.

Usage:  python3 model_comparison.py  [--figures-dir figures]
"""

from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from initial_learn import (
    CR_PREDICTOR_FEATURES,
    DEFAULT_XGB_PARAMS,
    DnDFeatureEngineer,
    FEATURE_COLUMNS,
    _make_xgb,
    _prepare_cr_dataset,
    prepare_xy,
)

LOGGER = logging.getLogger(__name__)


# ── Candidate models (all consume the engineered feature matrix) ──────────


def _candidates(random_state: int = 42) -> dict:
    """Model zoo.  Each pipeline starts from the engineered features."""
    prep = lambda: [
        ("feature_engineer", DnDFeatureEngineer(1.5, FEATURE_COLUMNS)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        # Lec 04 (logistic loss) + Lec 05 (ridge penalty): penalized ERM.
        "Ridge Logistic (Lec 04+05)": Pipeline(
            prep()
            + [
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                )
            ]
        ),
        # Lec 06 pt 1-2: RBF Mercer kernel via Rahimi-Recht random Fourier
        # features -> linear logistic model in the random feature space.
        "RFF Kernel Logistic (Lec 06)": Pipeline(
            prep()
            + [
                (
                    "rff",
                    RBFSampler(
                        gamma="scale", n_components=500, random_state=random_state
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        # Production learner (uncalibrated for a fair risk comparison).
        "XGBoost (production)": Pipeline(
            [
                ("feature_engineer", DnDFeatureEngineer(1.5, FEATURE_COLUMNS)),
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", _make_xgb(_load_tuned_params())),
            ]
        ),
    }


def _load_tuned_params(figures_dir: str = "figures") -> dict:
    try:
        with open(os.path.join(figures_dir, "metrics.json"), encoding="utf-8") as fh:
            return json.load(fh)["params"]
    except (OSError, KeyError, json.JSONDecodeError):
        return DEFAULT_XGB_PARAMS


def benchmark_classifiers(
    input_csv: str = "clean_aggregated_combat_data.csv",
    n_splits: int = 4,
) -> pd.DataFrame:
    """Grouped-CV comparison of the course model zoo on the win task."""
    raw = pd.read_csv(input_csv)
    X, y, groups = prepare_xy(raw)
    LOGGER.info("Benchmark matrix %s | %d campaign groups", X.shape, groups.nunique())

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []
    for name, pipe in _candidates().items():
        aucs, briers, lls = [], [], []
        for tr, te in cv.split(X, y, groups):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y.iloc[te], p))
            briers.append(brier_score_loss(y.iloc[te], p))
            lls.append(log_loss(y.iloc[te], p))
        records.append(
            {
                "model": name,
                "cv_auc_mean": np.mean(aucs),
                "cv_auc_std": np.std(aucs),
                "cv_brier_mean": np.mean(briers),
                "cv_brier_std": np.std(briers),
                "cv_logloss_mean": np.mean(lls),
                "cv_logloss_std": np.std(lls),
            }
        )
        LOGGER.info(
            "%-30s AUC %.4f±%.4f | Brier %.4f | logloss %.4f",
            name,
            np.mean(aucs),
            np.std(aucs),
            np.mean(briers),
            np.mean(lls),
        )
    return pd.DataFrame(records)


def plot_benchmark(df: pd.DataFrame, out_path: str) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    order = df.sort_values("cv_auc_mean")
    y_pos = np.arange(len(order))
    ax.barh(
        y_pos,
        order["cv_auc_mean"],
        xerr=order["cv_auc_std"],
        color="#2a78d6",
        edgecolor="black",
        linewidth=0.4,
        capsize=4,
        height=0.55,
    )
    for i, (_, r) in enumerate(order.iterrows()):
        ax.text(
            r["cv_auc_mean"] + r["cv_auc_std"] + 0.004,
            i,
            f"{r['cv_auc_mean']:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order["model"])
    ax.axvline(0.5, color="#898781", linestyle="--", linewidth=1)
    ax.text(0.501, len(order) - 0.4, "coin flip", color="#898781", fontsize=8)
    ax.set_xlabel("Grouped CV ROC-AUC (unseen campaigns), ±1 SD")
    ax.set_title("Win-prediction risk: course model zoo vs production XGBoost")
    ax.set_xlim(0.45, max(0.72, order["cv_auc_mean"].max() + 0.06))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved benchmark figure -> %s", out_path)


# ── Kernel two-sample test: campaign shift (Lec 06 pt 3) ──────────────────


def _rbf_kernel_matrix(A: np.ndarray, B: np.ndarray, bandwidth: float) -> np.ndarray:
    sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-sq / (2.0 * bandwidth**2))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """Unbiased MMD^2 estimator (Gretton et al.) with an RBF kernel."""
    Kxx = _rbf_kernel_matrix(X, X, bandwidth)
    Kyy = _rbf_kernel_matrix(Y, Y, bandwidth)
    Kxy = _rbf_kernel_matrix(X, Y, bandwidth)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()


def campaign_shift_mmd(
    input_csv: str = "clean_aggregated_combat_data.csv",
    n_per_side: int = 1500,
    n_permutations: int = 200,
    random_state: int = 42,
) -> dict:
    """Test H0: train-campaign features ~ held-out-campaign features.

    Uses the same 80/20 GroupShuffleSplit as training, the median-heuristic
    bandwidth, and a permutation null for the p-value.
    """
    from sklearn.model_selection import GroupShuffleSplit

    raw = pd.read_csv(input_csv)
    X, y, groups = prepare_xy(raw)
    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            X, y, groups
        )
    )

    eng = DnDFeatureEngineer(1.5, FEATURE_COLUMNS)
    Z = eng.fit(X).transform(X)
    Z = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(Z), columns=Z.columns
    )
    Z = StandardScaler().fit_transform(Z)

    rng = np.random.default_rng(random_state)
    A = Z[rng.choice(tr, size=min(n_per_side, len(tr)), replace=False)]
    B = Z[rng.choice(te, size=min(n_per_side, len(te)), replace=False)]

    # Median heuristic on the pooled sample.
    pooled = np.vstack([A, B])
    sub = pooled[rng.choice(len(pooled), size=min(1000, len(pooled)), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    bandwidth = float(np.sqrt(np.median(d2[d2 > 0]) / 2.0))

    observed = mmd2_unbiased(A, B, bandwidth)

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(len(pooled))
        null[i] = mmd2_unbiased(
            pooled[perm[: len(A)]], pooled[perm[len(A) :]], bandwidth
        )
    p_value = float((null >= observed).mean())

    result = {
        "mmd2_observed": float(observed),
        "mmd2_null_95pct": float(np.quantile(null, 0.95)),
        "p_value": p_value,
        "bandwidth_median_heuristic": bandwidth,
        "n_per_side": int(len(A)),
        "n_permutations": n_permutations,
    }
    LOGGER.info(
        "Campaign-shift MMD^2 = %.5f (null 95%% = %.5f) -> p = %.3f  [%s]",
        observed,
        result["mmd2_null_95pct"],
        p_value,
        "H0 REJECTED: campaigns shift" if p_value < 0.05 else "no detectable shift",
    )
    return result


# ── Gaussian Process CR predictor (Lec 06 pt 4) ───────────────────────────


def gp_cr_predictor(
    official_csv: str = "Monster Spreadsheet (D&D5e) - Official Stats.csv",
    test_size: float = 0.2,
) -> dict:
    """GP regression vs XGBoost on the 797-monster CR prediction task."""
    df = _prepare_cr_dataset(official_csv)
    X = df[list(CR_PREDICTOR_FEATURES)].to_numpy(dtype=float)
    y = df["cr_num"].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    # RBF kernel + noise; hyperparameters by marginal-likelihood maximization.
    kernel = ConstantKernel(1.0) * RBF(
        length_scale=np.ones(Xtr.shape[1])
    ) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=42, n_restarts_optimizer=2
    )
    gp.fit(Xtr, y_train)
    y_gp, y_gp_std = gp.predict(Xte, return_std=True)

    import xgboost as xgb

    xgbr = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    xgbr.fit(X_train, y_train)
    y_xgb = xgbr.predict(X_test)

    # Check the GP's 95% interval actually covers ~95% of the test points.
    lo, hi = y_gp - 1.96 * y_gp_std, y_gp + 1.96 * y_gp_std
    coverage = float(((y_test >= lo) & (y_test <= hi)).mean())

    result = {
        "gp_mae": float(mean_absolute_error(y_test, y_gp)),
        "gp_r2": float(r2_score(y_test, y_gp)),
        "gp_mean_pred_std": float(y_gp_std.mean()),
        "gp_95_interval_coverage": coverage,
        "xgb_mae": float(mean_absolute_error(y_test, y_xgb)),
        "xgb_r2": float(r2_score(y_test, y_xgb)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "learned_kernel": str(gp.kernel_),
    }
    LOGGER.info(
        "CR predictor — GP: MAE %.3f R2 %.3f (95%% coverage %.2f) | XGB: MAE %.3f R2 %.3f",
        result["gp_mae"],
        result["gp_r2"],
        coverage,
        result["xgb_mae"],
        result["xgb_r2"],
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="clean_aggregated_combat_data.csv")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--skip-gp", action="store_true")
    parser.add_argument("--skip-mmd", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    os.makedirs(args.figures_dir, exist_ok=True)
    out: dict = {}

    df_bench = benchmark_classifiers(args.input)
    df_bench.to_csv(os.path.join(args.figures_dir, "model_comparison.csv"), index=False)
    plot_benchmark(df_bench, os.path.join(args.figures_dir, "model_comparison.png"))
    out["classifier_benchmark"] = df_bench.to_dict(orient="records")

    if not args.skip_mmd:
        out["campaign_shift_mmd"] = campaign_shift_mmd(args.input)
    if not args.skip_gp:
        out["gp_cr_predictor"] = gp_cr_predictor()

    with open(
        os.path.join(args.figures_dir, "course_benchmark.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(out, fh, indent=2)
    LOGGER.info("Wrote %s/course_benchmark.json", args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
