"""True Lethality Engine — training pipeline.

Reads ``clean_aggregated_combat_data.csv`` (produced by ``parse_fireball.py``),
engineers D&D combat-math features inside a serializable sklearn transformer,
tunes an XGBoost classifier with Optuna under group-aware cross-validation,
calibrates its probabilities, and exports the whole pipeline as
``true_lethality_model.pkl``.

Design notes
------------
* **Group-aware validation.** Encounters from the same campaign log share a
  party, a DM, and house rules.  Random splits leak this; we split by source
  file via ``StratifiedGroupKFold`` / ``GroupShuffleSplit``.
* **Calibration.** The product decision ("which party level gives a 65% win
  rate?") consumes raw probabilities, so the serving model is wrapped in
  isotonic ``CalibratedClassifierCV``.  A twin uncalibrated XGBoost is fitted
  on the same features for interpretation (gain importance + native SHAP).
* **OOD hardening.** Inputs are clipped to sane 5e bands inside the
  transformer, and monotone constraints force sensible extrapolation for
  absurd homebrew (10,000 HP can only ever lower the win probability).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

# Headless / script-safe plotting (avoids macOS GUI backend crashes).
os.environ.setdefault("MPLBACKEND", "Agg")

from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from monster_offense import (
    CR_TO_XP,
    cr_to_xp,
    encounter_xp_multiplier,
    extract_legendary_mobility,
    extract_official_traits,
    offense_from_cr,
)

LOGGER = logging.getLogger(__name__)

CLASS_COLUMNS: Tuple[str, ...] = (
    "num_barbarian",
    "num_bard",
    "num_cleric",
    "num_druid",
    "num_fighter",
    "num_monk",
    "num_paladin",
    "num_ranger",
    "num_rogue",
    "num_sorcerer",
    "num_warlock",
    "num_wizard",
)

# Per-player XP thresholds by level: (Easy, Medium, Hard, Deadly) — DMG p.82.
LEVEL_THRESHOLDS: Dict[int, Tuple[int, int, int, int]] = {
    1: (25, 50, 75, 100),
    2: (50, 100, 150, 200),
    3: (75, 150, 225, 400),
    4: (125, 250, 375, 500),
    5: (250, 500, 750, 1100),
    6: (300, 600, 900, 1400),
    7: (350, 750, 1100, 1700),
    8: (450, 900, 1400, 2100),
    9: (550, 1100, 1600, 2400),
    10: (600, 1200, 1900, 2800),
    11: (800, 1600, 2400, 3600),
    12: (1000, 2000, 3000, 4500),
    13: (1100, 2200, 3400, 5100),
    14: (1250, 2500, 3800, 5700),
    15: (1400, 2800, 4300, 6400),
    16: (1600, 3200, 4800, 7200),
    17: (2000, 3900, 5900, 8800),
    18: (2100, 4200, 6300, 9500),
    19: (2400, 4900, 7300, 10900),
    20: (2800, 5700, 8500, 12700),
}

# Backward-compatible aliases (app.py / fair_fight_finder.py import these).
_cr_to_xp = cr_to_xp


def _classify_xp_tier(monster_xp: float, level: float, party_size: float) -> int:
    """0=Easy, 1=Medium, 2=Hard, 3=Deadly, 4=Super-Deadly (>2x Deadly)."""
    lvl = max(1, min(20, int(round(level))))
    easy, medium, hard, deadly = LEVEL_THRESHOLDS[lvl]
    ps = max(1.0, float(party_size))
    easy, medium, hard, deadly = (t * ps for t in (easy, medium, hard, deadly))
    if monster_xp >= 2 * deadly:
        return 4
    if monster_xp >= deadly:
        return 3
    if monster_xp >= hard:
        return 2
    if monster_xp >= medium:
        return 1
    return 0


# ── Raw input columns the pipeline accepts ─────────────────────────────────
# Only the CORE columns are strictly required; the transformer derives every
# OPTIONAL column from core values when absent (backward compatibility with
# older CSVs and simple inference callers).
RAW_INPUT_COLUMNS: Tuple[str, ...] = (
    "avg_party_level",
    "party_size",
    "num_monsters",
    "avg_monster_cr",
    "avg_monster_hp",
    "avg_monster_ac",
    "has_healer",
    "has_tank",
    "has_arcane",
    "has_martial_dps",
    "monster_is_legendary",
    "monster_has_mobility",
    "avg_monster_size_num",
    "avg_monster_stat_sum",
    "monster_has_physical_res",
    "monster_immune_to_cc",
    "monster_has_magic_res",
    "monster_has_pack_tactics",
    "monster_has_spellcasting",
    "monster_has_regeneration",
    # Offensive potency + roster shape (derived from the above when missing)
    "num_monsters_total",
    "max_monster_cr",
    "total_monster_hp",
    "total_monster_xp",
    "avg_monster_dpr",
    "total_monster_dpr",
    "max_monster_atk_bonus",
    "max_monster_save_dc",
    "max_monster_burst",
)

OPTIONAL_RAW_COLUMNS: Tuple[str, ...] = (
    "num_monsters_total",
    "max_monster_cr",
    "total_monster_hp",
    "total_monster_xp",
    "avg_monster_dpr",
    "total_monster_dpr",
    "max_monster_atk_bonus",
    "max_monster_save_dc",
    "max_monster_burst",
)

# ── Final engineered feature matrix ────────────────────────────────────────
FEATURE_COLUMNS: Tuple[str, ...] = (
    "avg_party_level",
    "party_size",
    "num_monsters",
    "num_monsters_total",
    "cr_to_party_level",
    "monster_hp_per_player",
    "avg_monster_ac",
    "has_healer",
    "has_tank",
    "has_arcane",
    "has_martial_dps",
    "monster_is_legendary",
    "monster_has_mobility",
    "avg_monster_size_num",
    "avg_monster_stat_sum",
    "monster_has_physical_res",
    "monster_immune_to_cc",
    "monster_has_magic_res",
    "monster_has_pack_tactics",
    "monster_has_spellcasting",
    "monster_has_regeneration",
    # Offensive potency
    "max_monster_cr",
    "avg_monster_dpr",
    "total_monster_dpr",
    "max_monster_atk_bonus",
    "max_monster_save_dc",
    "max_monster_burst",
    # Exponential power curve + legacy interactions
    "true_party_power",
    "cr_to_party_power",
    "mobility_threat",
    "legendary_attrition",
    "stat_power_mismatch",
    "physical_res_vs_martial",
    "magic_res_vs_arcane",
    "pack_tactics_vs_tank",
    # Combat-math core: the damage race
    "party_hit_chance",
    "monster_hit_chance",
    "rounds_to_kill_monster",
    "rounds_to_kill_party",
    "lethality_log_ratio",
    "save_dc_pressure",
    "action_economy_ratio",
    "burst_vs_pc_hp",
    # DMG XP budget (fixed: total XP x encounter multiplier)
    "xp_difficulty_tier",
    "xp_budget_ratio",
)


# ── Party combat-math estimators ───────────────────────────────────────────
# Coarse but *ordered* heuristics: XGBoost only needs the relative geometry
# of the fight to be right, not the exact numbers.

def _party_proficiency(level: pd.Series) -> pd.Series:
    return 2 + ((level.clip(1, 20) - 1) // 4)


def _party_attack_bonus(level: pd.Series) -> pd.Series:
    # Proficiency + primary stat mod growing 3 -> 5.5 over 20 levels.
    return _party_proficiency(level) + 3.0 + level.clip(1, 20) / 8.0


def _party_ac_estimate(level: pd.Series, has_tank: pd.Series) -> pd.Series:
    return 14.0 + level.clip(1, 20) / 4.0 + has_tank.astype(float)


def _party_dpr_per_member(level: pd.Series) -> pd.Series:
    # ~5 at level 1, ~12 at 5 (Extra Attack), ~24 at 11, ~41 at 20.
    return 3.0 + 1.9 * level.clip(1, 20)


def _party_hp_per_member(level: pd.Series) -> pd.Series:
    return 4.5 + 5.5 * level.clip(1, 20)


class DnDFeatureEngineer(BaseEstimator, TransformerMixin):
    """Converts raw encounter rows into the engineered feature matrix.

    Serialized inside the pipeline .pkl so inference callers pass raw
    columns and never re-implement feature math.

    Parameters
    ----------
    power_exponent : float
        Exponent of the True Party Power curve (spell/feature scaling is
        super-linear in level).
    output_columns : tuple of str
        Ordered output feature names — must match the downstream model.
    """

    def __init__(
        self,
        power_exponent: float = 1.5,
        output_columns: Tuple[str, ...] = FEATURE_COLUMNS,
    ):
        self.power_exponent = power_exponent
        self.output_columns = output_columns

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_out_ = list(self.output_columns)
        return self

    # ── derivation of optional raw columns ────────────────────────────────
    @staticmethod
    def _ensure_raw_columns(out: pd.DataFrame) -> pd.DataFrame:
        if "num_monsters" not in out.columns:
            out["num_monsters"] = 1
        n = out["num_monsters"].clip(lower=1)

        # NaN in an *existing* column previously fell through to the global
        # median imputer — row-inconsistent.  Fill from derived values instead.
        if "num_monsters_total" not in out.columns:
            out["num_monsters_total"] = out["num_monsters"]
        else:
            out["num_monsters_total"] = out["num_monsters_total"].fillna(
                out["num_monsters"]
            )
        if "max_monster_cr" not in out.columns:
            out["max_monster_cr"] = out["avg_monster_cr"]
        else:
            out["max_monster_cr"] = out["max_monster_cr"].fillna(
                out["avg_monster_cr"]
            )
        if "total_monster_hp" not in out.columns:
            out["total_monster_hp"] = out["avg_monster_hp"] * n
        else:
            out["total_monster_hp"] = out["total_monster_hp"].fillna(
                out["avg_monster_hp"] * n
            )

        cr_offense = out["avg_monster_cr"].map(offense_from_cr)
        if "avg_monster_dpr" not in out.columns:
            out["avg_monster_dpr"] = cr_offense.map(lambda t: t[1])
        else:
            out["avg_monster_dpr"] = out["avg_monster_dpr"].fillna(
                cr_offense.map(lambda t: t[1])
            )
        if "total_monster_dpr" not in out.columns:
            out["total_monster_dpr"] = out["avg_monster_dpr"] * n
        else:
            out["total_monster_dpr"] = out["total_monster_dpr"].fillna(
                out["avg_monster_dpr"] * n
            )
        if "max_monster_atk_bonus" not in out.columns:
            out["max_monster_atk_bonus"] = cr_offense.map(lambda t: t[0])
        else:
            out["max_monster_atk_bonus"] = out["max_monster_atk_bonus"].fillna(
                cr_offense.map(lambda t: t[0])
            )
        if "max_monster_save_dc" not in out.columns:
            out["max_monster_save_dc"] = cr_offense.map(lambda t: t[2])
        else:
            out["max_monster_save_dc"] = out["max_monster_save_dc"].fillna(
                cr_offense.map(lambda t: t[2])
            )
        if "total_monster_xp" not in out.columns:
            out["total_monster_xp"] = out["avg_monster_cr"].map(cr_to_xp) * n
        else:
            out["total_monster_xp"] = out["total_monster_xp"].fillna(
                out["avg_monster_cr"].map(cr_to_xp) * n
            )
        if "max_monster_burst" not in out.columns:
            # No statblock -> assume the nova equals sustained DPR.
            out["max_monster_burst"] = out["avg_monster_dpr"]
        else:
            out["max_monster_burst"] = out["max_monster_burst"].fillna(
                out["avg_monster_dpr"]
            )
        return out

    @staticmethod
    def _clip_ood(out: pd.DataFrame) -> pd.DataFrame:
        """Adversarial homebrew hardening: clip every raw input to legal 5e
        bands so a 10,000 HP / AC 3 monster lands on the model's trained
        manifold edge instead of an arbitrary tree leaf."""
        bounds = {
            "avg_party_level": (1, 20),
            "party_size": (1, 8),
            "num_monsters": (1, 30),
            "num_monsters_total": (1, 30),
            "avg_monster_cr": (0, 30),
            "max_monster_cr": (0, 30),
            "avg_monster_hp": (1, 1000),
            "total_monster_hp": (1, 8000),
            "avg_monster_ac": (10, 30),
            "avg_monster_stat_sum": (60, 250),
            "avg_monster_size_num": (1, 6),
            "avg_monster_dpr": (0.5, 350),
            "total_monster_dpr": (0.5, 1500),
            "max_monster_atk_bonus": (0, 20),
            "max_monster_save_dc": (8, 30),
            "max_monster_burst": (0.5, 400),
            "total_monster_xp": (0, 1_000_000),
        }
        for col, (lo, hi) in bounds.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lo, upper=hi)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        # ── Derive party_size / role flags from class counts if absent ────
        if "party_size" not in out.columns or out["party_size"].isna().all():
            class_cols = [c for c in CLASS_COLUMNS if c in out.columns]
            if class_cols:
                out["party_size"] = out[class_cols].sum(axis=1)
        out["party_size"] = out["party_size"].replace(0, np.nan)
        out["avg_party_level"] = out["avg_party_level"].replace(0, np.nan)

        role_specs = {
            "has_healer": ("num_cleric", "num_druid", "num_bard"),
            "has_tank": ("num_barbarian", "num_fighter", "num_paladin"),
            "has_arcane": ("num_wizard", "num_sorcerer", "num_warlock"),
            "has_martial_dps": ("num_rogue", "num_monk", "num_ranger"),
        }
        for flag, sources in role_specs.items():
            if flag not in out.columns:
                present = pd.Series(False, index=out.index)
                for src in sources:
                    present = present | (out.get(src, 0) > 0)
                out[flag] = present.astype(int)

        for col in (
            "monster_is_legendary", "monster_has_mobility",
            "avg_monster_size_num", "avg_monster_stat_sum",
            "monster_has_physical_res", "monster_immune_to_cc",
            "monster_has_magic_res", "monster_has_pack_tactics",
            "monster_has_spellcasting", "monster_has_regeneration",
        ):
            if col not in out.columns:
                out[col] = 0

        out = self._ensure_raw_columns(out)
        out = self._clip_ood(out)

        level = out["avg_party_level"].fillna(1.0)
        party_size = out["party_size"].fillna(4.0)
        n_total = out["num_monsters_total"].fillna(out["num_monsters"]).clip(lower=1)

        # ── Ratio + power-curve features ──────────────────────────────────
        total_party_level = out["avg_party_level"] * out["party_size"]
        out["cr_to_party_level"] = (
            out["avg_monster_cr"] * out["num_monsters"]
        ) / total_party_level
        out["monster_hp_per_player"] = out["total_monster_hp"] / out["party_size"]

        out["true_party_power"] = out["party_size"] * (
            out["avg_party_level"] ** self.power_exponent
        )
        out["cr_to_party_power"] = (
            out["avg_monster_cr"] * out["num_monsters"]
        ) / out["true_party_power"].replace(0, np.nan)

        # ── Legacy relational interactions ────────────────────────────────
        # Mobility punishes parties with no reliable ranged answer: neither
        # arcane casters nor martial ranged chassis (ranger/rogue).
        out["mobility_threat"] = (
            out["monster_has_mobility"]
            * (1 - out[["has_arcane", "has_martial_dps"]].max(axis=1))
        ).astype(float)
        out["legendary_attrition"] = (
            out["monster_is_legendary"] * (1 - out["has_healer"])
        ).astype(float)
        _monster_budget = out["avg_monster_size_num"] * out["avg_monster_stat_sum"]
        out["stat_power_mismatch"] = _monster_budget / out["true_party_power"].replace(
            0, np.nan
        )
        out["physical_res_vs_martial"] = (
            out["monster_has_physical_res"] * out["has_martial_dps"]
        ).astype(float)
        out["magic_res_vs_arcane"] = (
            out["monster_has_magic_res"] * out["has_arcane"]
        ).astype(float)
        out["pack_tactics_vs_tank"] = (
            out["monster_has_pack_tactics"] * out["has_tank"]
        ).astype(float)

        # ── Combat-math core: to-hit geometry and the damage race ─────────
        party_atk = _party_attack_bonus(level)
        party_ac = _party_ac_estimate(level, out["has_tank"])

        # P(hit) on a d20: (21 + bonus - AC) / 20, clamped to nat-1/nat-20.
        out["party_hit_chance"] = (
            (21.0 + party_atk - out["avg_monster_ac"]) / 20.0
        ).clip(0.05, 0.95)
        out["monster_hit_chance"] = (
            (21.0 + out["max_monster_atk_bonus"] - party_ac) / 20.0
        ).clip(0.05, 0.95)

        # Effective monster HP inflated by mitigation mechanics.
        effective_monster_hp = out["total_monster_hp"] * (
            1.0
            + 0.40 * out["monster_has_physical_res"]
            + 0.30 * out["monster_is_legendary"]
            + 0.25 * out["monster_has_regeneration"]
        )
        party_dpr = party_size * _party_dpr_per_member(level)
        party_hp = party_size * _party_hp_per_member(level) * (
            1.0 + 0.25 * out["has_healer"]
        )
        # Pack tactics grants advantage: ~+40% effective hit rate.
        monster_effective_dpr = (
            out["total_monster_dpr"]
            * out["monster_hit_chance"]
            * (1.0 + 0.4 * out["monster_has_pack_tactics"])
        )

        out["rounds_to_kill_monster"] = (
            effective_monster_hp / (party_dpr * out["party_hit_chance"]).replace(0, np.nan)
        ).clip(upper=50)
        out["rounds_to_kill_party"] = (
            party_hp / monster_effective_dpr.replace(0, np.nan)
        ).clip(upper=50)
        # >0 means the party wins the damage race; the single most important
        # quantity in 5e combat.  Log keeps it symmetric around 0.
        out["lethality_log_ratio"] = np.log(
            out["rounds_to_kill_party"] / out["rounds_to_kill_monster"].replace(0, np.nan)
        ).clip(-4, 4)

        # Save DC vs a typical mid save bonus (~2 + level/4).
        out["save_dc_pressure"] = (
            out["max_monster_save_dc"] - (12.0 + level / 4.0)
        ).clip(-10, 15)

        out["action_economy_ratio"] = (n_total / party_size).clip(0, 10)

        # Nova pressure: burst damage relative to a single PC's HP pool.
        # >1 means the monster's scariest action can drop a character from
        # full health in one turn (Power Word Kill, ancient dragon breath).
        out["burst_vs_pc_hp"] = (
            out["max_monster_burst"] / _party_hp_per_member(level)
        ).clip(0, 10)

        # ── DMG XP budget (fixed) ─────────────────────────────────────────
        # Adjusted XP = total monster XP x DMG multiplier for monster count,
        # with the official party-size adjustment (DMG p.83: <3 PCs shift the
        # multiplier one step up the ladder, >=6 PCs one step down).
        multipliers = pd.Series(
            [
                encounter_xp_multiplier(n_val, ps)
                for n_val, ps in zip(n_total, party_size)
            ],
            index=out.index,
        )
        adjusted_xp = out["total_monster_xp"] * multipliers
        out["xp_difficulty_tier"] = pd.Series(
            [
                float(_classify_xp_tier(xp_val, lvl, ps))
                for xp_val, lvl, ps in zip(adjusted_xp, level, party_size)
            ],
            index=out.index,
        )
        deadly = pd.Series(
            [
                LEVEL_THRESHOLDS[max(1, min(20, int(round(lvl))))][3] * ps
                for lvl, ps in zip(level, party_size)
            ],
            index=out.index,
        )
        out["xp_budget_ratio"] = (adjusted_xp / deadly.replace(0, np.nan)).clip(0, 20)

        return out[list(self.output_columns)]

    def get_feature_names_out(self, input_features=None):
        return list(self.output_columns)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy convenience wrapper: derive party_size/role flags on a full frame.

    Kept for notebooks and gan_trial.py.  Model training no longer calls the
    full feature math here — the pipeline's transformer owns it.
    """
    out = df.copy()
    out["party_size"] = out[list(CLASS_COLUMNS)].sum(axis=1)
    out["party_size"] = out["party_size"].replace(0, np.nan)
    out["has_healer"] = (
        (out["num_cleric"] > 0) | (out["num_druid"] > 0) | (out["num_bard"] > 0)
    ).astype(int)
    out["has_tank"] = (
        (out["num_barbarian"] > 0) | (out["num_fighter"] > 0) | (out["num_paladin"] > 0)
    ).astype(int)
    out["has_arcane"] = (
        (out["num_wizard"] > 0) | (out["num_sorcerer"] > 0) | (out["num_warlock"] > 0)
    ).astype(int)
    out["has_martial_dps"] = (
        (out["num_rogue"] > 0) | (out["num_monk"] > 0) | (out["num_ranger"] > 0)
    ).astype(int)
    return out


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _encounter_group(encounter_id: str) -> str:
    """Campaign-level group key: the source .jsonl basename when present."""
    s = str(encounter_id)
    if ".jsonl" in s:
        return s.split(".jsonl")[0]
    return s


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y, groups) from the parser CSV (or GAN-balanced CSV)."""
    if "final_outcome" in df.columns:
        df = df.dropna(subset=["avg_monster_cr", "avg_monster_hp", "avg_monster_ac"])
        allowed = {"Party Win", "Party Loss/Casualty", "Mixed/Pyrrhic"}
        df = df[df["final_outcome"].isin(allowed)].copy()
        df["target"] = (df["final_outcome"] == "Party Win").astype(int)
        df = engineer_features(df)
    else:
        df = df.dropna(subset=["avg_monster_cr", "avg_monster_hp", "avg_monster_ac"]).copy()
        LOGGER.info("GAN-balanced format detected (no final_outcome column).")

    core_required = [
        c for c in RAW_INPUT_COLUMNS
        if c in df.columns and c not in OPTIONAL_RAW_COLUMNS
    ]
    df = df.dropna(subset=core_required + ["target"])

    present = [c for c in RAW_INPUT_COLUMNS if c in df.columns]
    X = df[present]
    y = df["target"]
    if "encounter_id" in df.columns:
        groups = df["encounter_id"].map(_encounter_group)
    else:
        groups = pd.Series(np.arange(len(df)), index=df.index).astype(str)
    return X, y, groups


DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
}

# Domain-knowledge monotone constraints (also the OOD extrapolation guard).
MONOTONE_CONSTRAINTS: Dict[str, int] = {
    "avg_party_level": 1,
    "party_size": 1,
    "num_monsters": -1,
    "num_monsters_total": -1,
    "cr_to_party_level": -1,
    "monster_hp_per_player": -1,
    "avg_monster_ac": -1,
    "has_healer": 1,
    "has_tank": 1,
    "has_arcane": 1,
    "has_martial_dps": 1,
    "monster_is_legendary": -1,
    "monster_has_mobility": -1,
    "avg_monster_size_num": -1,
    "avg_monster_stat_sum": -1,
    "monster_has_physical_res": -1,
    "monster_immune_to_cc": -1,
    "monster_has_magic_res": -1,
    "monster_has_pack_tactics": -1,
    "monster_has_spellcasting": -1,
    "monster_has_regeneration": -1,
    "max_monster_cr": -1,
    "avg_monster_dpr": -1,
    "total_monster_dpr": -1,
    "max_monster_atk_bonus": -1,
    "max_monster_save_dc": -1,
    "max_monster_burst": -1,
    "true_party_power": 1,
    "cr_to_party_power": -1,
    "mobility_threat": -1,
    "legendary_attrition": -1,
    "stat_power_mismatch": -1,
    "physical_res_vs_martial": -1,
    "magic_res_vs_arcane": -1,
    "pack_tactics_vs_tank": -1,
    "party_hit_chance": 1,
    "monster_hit_chance": -1,
    "rounds_to_kill_monster": -1,
    "rounds_to_kill_party": 1,
    "lethality_log_ratio": 1,
    "save_dc_pressure": -1,
    "action_economy_ratio": -1,
    "burst_vs_pc_hp": -1,
    "xp_difficulty_tier": -1,
    "xp_budget_ratio": -1,
}


def _make_xgb(params: Dict[str, Any], scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
    constraints = tuple(MONOTONE_CONSTRAINTS.get(c, 0) for c in FEATURE_COLUMNS)
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        monotone_constraints=constraints,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )


# Ridge strength for the logistic production model: grouped-CV sweep over
# C in [3e-3, 3] was flat (AUC 0.655-0.657); C=0.03 sat at the optimum.
LOGISTIC_C = 0.03


def build_model(
    params: Optional[Dict[str, Any]] = None,
    *,
    model_type: str = "xgb",
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
    calibration_cv: Any = 3,
    scale_pos_weight: float = 1.0,
) -> Pipeline:
    """Self-contained pipeline: FeatureEngineer -> Imputer [-> Scaler] -> clf.

    ``model_type`` selects the learner:

    * ``"xgb"`` (production) — monotone-constrained XGBoost.
    * ``"logistic"`` — l2-penalized logistic regression.  It *wins on
      observational predictive risk* under campaign-grouped CV (AUC 0.657
      vs 0.614) — but it is NOT the production model, deliberately: the app
      asks *interventional* questions ("same monster, more of them"), and
      the unconstrained linear fit absorbs DM-curation confounding (in real
      logs, many-monster fights are weak mobs that parties beat, so the
      monster-count coefficient comes out POSITIVE).  Behaviorally it rated
      8 Liches as trivial and a 10,000-HP monster as beatable.  XGBoost's
      monotone constraints encode the domain priors that make those
      counterfactual sweeps sane — we accept a small AUC penalty for
      decision-grade behavior.  (Full story: figures/course_benchmark.json
      and the README's course-concept section.)

    Calibration is sigmoid (Platt): isotonic's piecewise-constant map
    collapsed the win-rate binary search onto plateau edges (1 Lich and
    2 Liches appraised identically) and measured slightly worse.

    ``calibration_cv`` defaults to 3 random stratified folds, but callers
    with group structure should pass precomputed grouped splits (a list of
    (train_idx, test_idx) pairs) so the calibration folds don't leak
    campaigns — the pipeline preserves row order, so raw-X indices remain
    valid on the transformed matrix the calibrator sees.
    """
    steps = [
        (
            "feature_engineer",
            DnDFeatureEngineer(power_exponent=1.5, output_columns=FEATURE_COLUMNS),
        ),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if model_type == "logistic":
        steps.append(("scaler", StandardScaler()))
        clf: BaseEstimator = LogisticRegression(
            penalty="l2", C=LOGISTIC_C, max_iter=2000, random_state=42
        )
    elif model_type == "xgb":
        clf = _make_xgb(params or DEFAULT_XGB_PARAMS, scale_pos_weight)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    if calibrate:
        clf = CalibratedClassifierCV(clf, method=calibration_method, cv=calibration_cv)
    steps.append(("model", clf))
    return Pipeline(steps=steps)


def load_params_from_metrics(path: str = "figures/metrics.json") -> Dict[str, Any]:
    """Tuned hyperparameters from a previous run, or the defaults.

    Shared by run_training (--no-tune) and the ablation scripts so every
    variant trains with the same configuration.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            params = json.load(fh)["params"]
        LOGGER.info("Reusing tuned params from %s: %s", path, params)
        return dict(params)
    except (OSError, KeyError, json.JSONDecodeError):
        LOGGER.info("No previous tuned params found; using defaults.")
        return dict(DEFAULT_XGB_PARAMS)


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_trials: int = 30,
    n_splits: int = 4,
) -> Dict[str, Any]:
    """Optuna search over XGBoost hyperparameters with group-aware CV.

    The objective is mean ROC-AUC under StratifiedGroupKFold, so a
    configuration only scores well if it generalizes to *unseen campaigns*
    — the strongest available guard against fitting log noise.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        }
        pipe = build_model(params, calibrate=False)
        scores = cross_val_score(
            pipe, X, y, groups=groups, cv=cv, scoring="roc_auc", n_jobs=1
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LOGGER.info(
        "Optuna best CV ROC-AUC=%.4f with params: %s",
        study.best_value,
        study.best_params,
    )
    return study.best_params


# ── Plotting ───────────────────────────────────────────────────────────────

def _use_clean_grid_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        sns.set_theme(style="whitegrid")


def plot_feature_importances(
    model: xgb.XGBClassifier, features: Sequence[str], out_path: str
) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)
    y_pos = np.arange(len(features))

    _use_clean_grid_style()
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    palette = sns.color_palette("mako", n_colors=len(features))
    ax.barh(
        y_pos, importances[order],
        color=[palette[i] for i in range(len(features))],
        edgecolor="black", linewidth=0.3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in order], fontsize=7)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("True Lethality Engine — which signals drive splits?")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved feature importance figure -> %s", out_path)


def compute_native_shap(
    model: xgb.XGBClassifier, X_transformed: pd.DataFrame
) -> pd.DataFrame:
    """Exact TreeSHAP values via XGBoost's built-in pred_contribs — no shap
    package required.  Returns a DataFrame of per-row contributions."""
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_transformed, feature_names=list(X_transformed.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    return pd.DataFrame(
        contribs[:, :-1], columns=list(X_transformed.columns)
    )  # last column is the bias term


def plot_shap_summary(
    shap_df: pd.DataFrame, X_transformed: pd.DataFrame, out_path: str, top_k: int = 15
) -> None:
    """Beeswarm-style SHAP summary using matplotlib only."""
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top = list(mean_abs.head(top_k).index)

    _use_clean_grid_style()
    fig, ax = plt.subplots(figsize=(9, 0.5 * top_k + 2), dpi=300)
    rng = np.random.default_rng(42)
    for i, feat in enumerate(reversed(top)):
        vals = shap_df[feat].to_numpy()
        raw = X_transformed[feat].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(raw, [5, 95])
        norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
        jitter = rng.normal(0, 0.08, size=len(vals))
        sc = ax.scatter(
            vals, np.full(len(vals), i) + jitter,
            c=norm, cmap="coolwarm", s=4, alpha=0.35, linewidths=0,
        )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(list(reversed(top)), fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to log-odds of Party Win")
    ax.set_title("Native TreeSHAP summary (color = feature value, blue→red)")
    fig.colorbar(sc, ax=ax, label="feature value (5–95 pct scaled)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved SHAP summary -> %s", out_path)


def plot_calibration_curve(
    y_true: pd.Series, y_prob: np.ndarray, out_path: str, n_bins: int = 12
) -> None:
    df = pd.DataFrame({"y": y_true.to_numpy(), "p": y_prob})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    agg = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")
    )
    _use_clean_grid_style()
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(agg["mean_pred"], agg["frac_pos"], marker="o", color="#1b9e77",
            label="Model")
    ax.set_xlabel("Predicted P(Party Win)")
    ax.set_ylabel("Observed win frequency")
    ax.set_title("Calibration on held-out campaigns")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved calibration curve -> %s", out_path)


def plot_correlation_heatmap(
    df: pd.DataFrame, features: Sequence[str], out_path: str
) -> None:
    corr = df[list(features)].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(13, 11), dpi=300)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
        square=True, linewidths=0.4, annot_kws={"size": 4.5},
        cbar_kws={"shrink": 0.75, "label": "Spearman ρ"}, ax=ax,
    )
    ax.tick_params(labelsize=6)
    ax.set_title("Spearman correlation among engineered features")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved correlation heatmap -> %s", out_path)


def plot_win_probability_curve(
    df: pd.DataFrame,
    pipe: Pipeline,
    out_path: str,
    *,
    n_bins: int = 18,
) -> None:
    """Binned empirical win rate vs cr_to_party_level with model overlay."""
    proba_win = pipe.predict_proba(df[[c for c in RAW_INPUT_COLUMNS if c in df.columns]])[:, 1]
    tmp = df.assign(p_win_hat=proba_win, cr_ratio=df["cr_to_party_level"]).dropna(
        subset=["cr_ratio"]
    )
    try:
        tmp["bin"] = pd.qcut(tmp["cr_ratio"], q=n_bins, duplicates="drop")
    except ValueError:
        tmp["bin"] = pd.cut(tmp["cr_ratio"], bins=n_bins)

    agg = tmp.groupby("bin", observed=False).agg(
        empirical_win_rate=("target", "mean"),
        mean_predicted=("p_win_hat", "mean"),
        n=("target", "size"),
        cr_mid=("cr_ratio", "median"),
    )

    _use_clean_grid_style()
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
    ax.plot(agg["cr_mid"], agg["empirical_win_rate"], marker="o", linewidth=2,
            label="Empirical win rate (binned)", color="#1b9e77")
    ax.plot(agg["cr_mid"], agg["mean_predicted"], marker="s", linewidth=2,
            linestyle="--", label="Calibrated XGBoost mean P(Party Win)",
            color="#d95f02")
    ax2 = ax.twinx()
    mids = agg["cr_mid"].to_numpy(dtype=float)
    width = float(np.ptp(mids) / max(len(agg) * 2.5, 1)) if mids.size else 0.01
    ax2.bar(agg["cr_mid"], agg["n"], width=width, alpha=0.15, color="gray",
            label="Encounters per bin")
    ax2.set_ylabel("Count per bin")
    ax.set_xlabel(r"Encounter ratio: monster CR burden $\div$ total party level")
    ax.set_ylabel("Probability of Party Win")
    ax.set_title("Win probability vs normalized challenge (empirical vs model)")
    ax.set_ylim(0, 1.05)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved probability curve -> %s", out_path)


# ── Training ───────────────────────────────────────────────────────────────

def plot_logistic_coefficients(
    pipe: Pipeline, features: Sequence[str], out_path: str
) -> None:
    """Standardized log-odds coefficients of the calibrated logistic model."""
    model = pipe.named_steps["model"]
    # CalibratedClassifierCV ensembles cv=3 fitted clones; average their coefs.
    coefs = np.mean(
        [c.estimator.coef_.ravel() for c in model.calibrated_classifiers_], axis=0
    )
    order = np.argsort(coefs)
    _use_clean_grid_style()
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    colors = ["#d03b3b" if c < 0 else "#2a78d6" for c in coefs[order]]
    ax.barh(np.arange(len(features)), coefs[order], color=colors,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([features[i] for i in order], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-odds of Party Win per +1 SD of feature")
    ax.set_title("Ridge logistic — standardized coefficients")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved coefficient plot -> %s", out_path)


def run_training(
    input_csv: str,
    figures_dir: str,
    model_output: str = "true_lethality_model.pkl",
    *,
    model_type: str = "xgb",
    test_size: float = 0.2,
    tune: bool = True,
    n_trials: int = 30,
) -> Dict[str, Any]:
    os.makedirs(figures_dir, exist_ok=True)

    raw = pd.read_csv(input_csv)
    X, y, groups = prepare_xy(raw)
    LOGGER.info(
        "Training matrix: %s | base rate P(win)=%.3f | %d campaign groups",
        X.shape, y.mean(), groups.nunique(),
    )

    # Group-aware holdout: test campaigns are never seen in training.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups.iloc[train_idx]

    params = DEFAULT_XGB_PARAMS
    if model_type != "xgb":
        # Logistic has one hyperparameter (C), fixed from a grouped-CV sweep.
        if tune:
            LOGGER.info("model_type=%s: Optuna applies to XGBoost only; skipping.", model_type)
    elif tune:
        LOGGER.info("Tuning hyperparameters with Optuna (%d trials)...", n_trials)
        params = tune_hyperparameters(X_train, y_train, g_train, n_trials=n_trials)
    else:
        # Reuse the last tuned configuration when available.
        params = load_params_from_metrics(os.path.join(figures_dir, "metrics.json"))

    # Cross-validated generalization estimate with the chosen params.
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    cv_auc = cross_val_score(
        build_model(params, model_type=model_type, calibrate=False),
        X_train, y_train, groups=g_train, cv=cv, scoring="roc_auc", n_jobs=1,
    )
    LOGGER.info("CV ROC-AUC (unseen campaigns): %.4f ± %.4f", cv_auc.mean(), cv_auc.std())

    # Group-aware calibration folds: the calibrator would otherwise use
    # random stratified folds, leaking same-campaign encounters between the
    # base-model fit and the Platt fit (positional indices stay valid because
    # the pipeline transforms preserve row order).
    cal_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    cal_cv = list(
        cal_splitter.split(np.zeros(len(y_train)), y_train, g_train)
    )
    pipe = build_model(
        params, model_type=model_type, calibrate=True, calibration_cv=cal_cv
    )
    pipe.fit(X_train, y_train)

    # ── Holdout evaluation ────────────────────────────────────────────────
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "holdout_pr_auc": float(average_precision_score(y_test, y_prob)),
        "holdout_brier": float(brier_score_loss(y_test, y_prob)),
        "holdout_log_loss": float(log_loss(y_test, y_prob)),
        "cv_roc_auc_mean": float(cv_auc.mean()),
        "cv_roc_auc_std": float(cv_auc.std()),
        "base_rate": float(y.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_type": model_type,
        "params": params,
    }
    LOGGER.info(
        "Holdout — acc %.4f | ROC-AUC %.4f | PR-AUC %.4f | Brier %.4f | logloss %.4f",
        metrics["holdout_accuracy"], metrics["holdout_roc_auc"],
        metrics["holdout_pr_auc"], metrics["holdout_brier"],
        metrics["holdout_log_loss"],
    )
    LOGGER.info("\n%s", classification_report(y_test, y_pred, digits=3))

    joblib.dump(pipe, model_output)
    LOGGER.info("Saved calibrated pipeline -> %s", model_output)
    with open(os.path.join(figures_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # ── Interpretation ────────────────────────────────────────────────────
    # SHAP twin is always XGBoost (tree SHAP is exact); for the logistic
    # production model we additionally plot its standardized coefficients.
    if model_type == "logistic":
        plot_logistic_coefficients(
            pipe, FEATURE_COLUMNS,
            os.path.join(figures_dir, "logistic_coefficients.png"),
        )
    engineer = pipe.named_steps["feature_engineer"]
    Xt_train = engineer.transform(X_train)
    interp = _make_xgb(params)
    interp.fit(Xt_train.fillna(Xt_train.median()), y_train)

    # Version-portable fallback artifact: joblib pickles break across XGBoost
    # versions, the native JSON format does not.  (Uncalibrated twin — the
    # calibrated pipeline remains the serving model.)
    interp.get_booster().save_model(
        os.path.splitext(model_output)[0] + "_uncalibrated_xgb.json"
    )

    plot_feature_importances(interp, FEATURE_COLUMNS, "ultimate_feature_importance.png")
    plot_feature_importances(
        interp, FEATURE_COLUMNS, os.path.join(figures_dir, "feature_importance.png")
    )

    sample = Xt_train.fillna(Xt_train.median()).sample(
        n=min(4000, len(Xt_train)), random_state=42
    )
    shap_df = compute_native_shap(interp, sample)
    plot_shap_summary(shap_df, sample, os.path.join(figures_dir, "shap_summary.png"))
    shap_rank = shap_df.abs().mean().sort_values(ascending=False)
    shap_rank.to_csv(os.path.join(figures_dir, "shap_ranking.csv"))
    LOGGER.info("Top SHAP features:\n%s", shap_rank.head(10).to_string())

    plot_calibration_curve(
        y_test, y_prob, os.path.join(figures_dir, "calibration_curve.png")
    )

    df_model = engineer.transform(X)
    df_model["target"] = y.values
    for col in RAW_INPUT_COLUMNS:
        if col not in df_model.columns and col in X.columns:
            df_model[col] = X[col].values
    plot_correlation_heatmap(
        df_model, FEATURE_COLUMNS, os.path.join(figures_dir, "feature_correlation_heatmap.png")
    )
    plot_win_probability_curve(
        df_model, pipe, os.path.join(figures_dir, "win_rate_vs_cr_ratio.png")
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# CR Predictor — appraise "what WotC would rate this" from raw stats
# ══════════════════════════════════════════════════════════════════════════

CR_PREDICTOR_FEATURES: Tuple[str, ...] = (
    "hp",
    "ac",
    "stat_sum",
    "size_num",
    "is_legendary",
    "has_mobility",
    "physical_res",
    "cc_immune",
    "magic_res",
    "pack_tactics",
    "spellcasting",
    "regeneration",
)


def parse_cr_value(raw: Any) -> float:
    """Safe CR parser handling fractions ('1/4') — replaces eval()."""
    s = str(raw).strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", s)
    if m:
        denom = float(m.group(2))
        return float(m.group(1)) / denom if denom else np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _prepare_cr_dataset(official_csv: str) -> pd.DataFrame:
    df = pd.read_csv(official_csv)
    df["cr_num"] = df["CR"].apply(parse_cr_value)

    for col in ["STR", "DEX", "CON", "INT", "WIS", "CHA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["stat_sum"] = df[["STR", "DEX", "CON", "INT", "WIS", "CHA"]].sum(axis=1)
    df["hp"] = pd.to_numeric(df["HP"], errors="coerce")
    df["ac"] = pd.to_numeric(df["AC"], errors="coerce")

    size_map = {"tiny": 1, "small": 2, "medium": 3, "large": 4, "huge": 5, "gargantuan": 6}
    df["size_num"] = df["Size"].str.strip().str.lower().map(size_map)

    # Canonical trait extraction (previously used slightly *different* regexes
    # than the app's inference path — e.g. missing "nonmagicalimmu" — so the
    # CR predictor trained on different trait definitions than it now serves).
    df[["is_legendary", "has_mobility"]] = extract_legendary_mobility(
        df["Additional"], df["Speeds"]
    )
    df[
        ["physical_res", "cc_immune", "magic_res",
         "pack_tactics", "spellcasting", "regeneration"]
    ] = extract_official_traits(df["WRI"], df["Additional"])[
        ["physical_res", "cc_immune", "magic_res",
         "pack_tactics", "spellcasting", "regeneration"]
    ]

    return df.dropna(subset=["cr_num", "hp", "ac", "stat_sum", "size_num"])


def train_cr_predictor(
    official_csv: str = "Monster Spreadsheet (D&D5e) - Official Stats.csv",
    output_path: str = "cr_predictor_model.pkl",
    test_size: float = 0.2,
) -> None:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    LOGGER.info("Training CR Predictor from official monster statblocks...")
    df = _prepare_cr_dataset(official_csv)
    LOGGER.info("Loaded %d monsters for CR prediction training.", len(df))

    X = df[list(CR_PREDICTOR_FEATURES)]
    y = df["cr_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    cr_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1,
    )
    cr_model.fit(X_train, y_train)

    y_pred = cr_model.predict(X_test)
    LOGGER.info(
        "CR Predictor — MAE: %.2f | R²: %.4f",
        mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred),
    )
    joblib.dump(cr_model, output_path)
    # Version-proof twin: XGBoost pickles are NOT portable across versions
    # (loading one in a different xgboost breaks feature-name validation at
    # predict time).  The native JSON format is — consumers prefer it.
    json_path = os.path.splitext(output_path)[0] + ".json"
    cr_model.save_model(json_path)
    LOGGER.info("Saved CR Predictor -> %s (+ %s)", output_path, json_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="clean_aggregated_combat_data.csv")
    p.add_argument("--figures-dir", default="figures")
    p.add_argument("--model-output", default="true_lethality_model.pkl")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--model", choices=["logistic", "xgb"], default="xgb",
        help="Production learner. Logistic wins raw grouped-CV AUC but "
             "fails interventional sanity (see build_model docstring); "
             "xgb's monotone constraints make it the deployment choice.",
    )
    p.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning.")
    p.add_argument("--trials", type=int, default=30, help="Optuna trial count.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--train-cr-predictor", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.verbose)
    try:
        run_training(
            args.input,
            args.figures_dir,
            args.model_output,
            model_type=args.model,
            test_size=args.test_size,
            tune=not args.no_tune,
            n_trials=args.trials,
        )
    except Exception:
        LOGGER.exception("Training pipeline failed.")
        return 1

    if args.train_cr_predictor:
        try:
            train_cr_predictor()
        except Exception:
            LOGGER.exception("CR Predictor training failed.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
