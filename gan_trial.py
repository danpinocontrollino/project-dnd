import argparse

import numpy as np
import pandas as pd
from ctgan import CTGAN
from sklearn.model_selection import GroupShuffleSplit

from initial_learn import _encounter_group

parser = argparse.ArgumentParser(description="CTGAN synthetic loss generation")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--output", default="gan_balanced_combat_data.csv")
parser.add_argument(
    "--include-holdout",
    action="store_true",
    help="Fit CTGAN on ALL losses (legacy behavior). Default EXCLUDES the "
    "production holdout campaigns so the synthetic rows cannot leak "
    "held-out information into any model trained on this CSV.",
)
args = parser.parse_args()

print("Loading the clean D&D dataset...")
df = pd.read_csv("clean_aggregated_combat_data.csv")

# Campaign group per row: the SAME anti-leakage key used by run_training.
df["campaign"] = df["encounter_id"].map(_encounter_group)

# ─── Step 1: Light pre-processing for the BASE columns ───────────────────
# With the Area 3 self-contained pipeline, CTGAN should generate RAW
# independent variables.  The pipeline's DnDFeatureEngineer will compute
# all derived features (ratios, power curves, interactions) automatically.
class_columns = [
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
]

df["party_size"] = df[class_columns].sum(axis=1).replace(0, 1)
df["avg_party_level"] = df["avg_party_level"].replace(0, 1)

df["has_healer"] = (
    (df["num_cleric"] > 0) | (df["num_druid"] > 0) | (df["num_bard"] > 0)
).astype(int)
df["has_tank"] = (
    (df["num_barbarian"] > 0) | (df["num_fighter"] > 0) | (df["num_paladin"] > 0)
).astype(int)
df["has_arcane"] = (
    (df["num_wizard"] > 0) | (df["num_sorcerer"] > 0) | (df["num_warlock"] > 0)
).astype(int)
df["has_martial_dps"] = (
    (df["num_rogue"] > 0) | (df["num_monk"] > 0) | (df["num_ranger"] > 0)
).astype(int)

# Binary target
df["target"] = df["final_outcome"].apply(lambda x: 1 if x == "Party Win" else 0)

# ─── BASE features for CTGAN ─────────────────────────────────────────────
# These are the RAW independent variables CTGAN will learn.
# NOTE: We use avg_monster_cr and avg_monster_hp (raw stats) instead of
#       cr_to_party_level and monster_hp_per_player (derived ratios).
#       The pipeline's DnDFeatureEngineer will derive those automatically.
#       We do NOT include interaction features (mobility_threat, etc.)
#       because they are deterministic functions of these base columns.
base_features = [
    "avg_party_level",
    "party_size",
    "avg_monster_cr",
    "avg_monster_hp",
    "avg_monster_ac",
    "has_healer",
    "has_tank",
    "has_arcane",
    "has_martial_dps",
    # Area 1: Monster trait base columns (from parse_fireball.py)
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
    # Target (always last)
    "target",
]

df_model = df[base_features + ["encounter_id", "campaign"]].dropna(subset=base_features)

# ─── Step 2: Isolate the Minority Class (The Losses) ─────────────────────
# LEAKAGE GUARD (default ON): replicate the production holdout split
# (GroupShuffleSplit by campaign, test_size=0.2, random_state=42) and fit
# CTGAN only on losses from TRAINING campaigns.  A GAN fitted on all data
# would encode held-out campaigns into the synthetic rows, contaminating
# any later evaluation against that holdout.
df_losses = df_model[df_model["target"] == 0]
if not args.include_holdout:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(
        splitter.split(df_model, df_model["target"], df_model["campaign"])
    )
    train_campaigns = set(df_model.iloc[train_idx]["campaign"])
    df_losses = df_losses[df_losses["campaign"].isin(train_campaigns)]
print(f"Training GAN on {len(df_losses)} real Party Losses...")
df_losses = df_losses[base_features]

# ─── Step 3: Discrete columns for the GAN ────────────────────────────────
# Prevents CTGAN from generating "0.5 Healers" or "3.14 Party Size".
discrete_columns = [
    "party_size",
    "has_healer",
    "has_tank",
    "has_arcane",
    "has_martial_dps",
    "monster_is_legendary",
    "monster_has_mobility",
    "monster_has_physical_res",
    "monster_immune_to_cc",
    "monster_has_magic_res",
    "monster_has_pack_tactics",
    "monster_has_spellcasting",
    "monster_has_regeneration",
    "target",
]

# ─── Step 4: Train the Generative Adversarial Network ────────────────────
ctgan = CTGAN(epochs=args.epochs, verbose=True)
ctgan.fit(df_losses, discrete_columns)

# ─── Step 5: Generate Synthetic Losses ────────────────────────────────────
# Balance against ALL real losses in df_model (df_losses may be the
# train-campaign subset when the leakage guard is active).
num_synthetic_rows = len(df_model[df_model["target"] == 1]) - len(
    df_model[df_model["target"] == 0]
)
print(f"Generating {num_synthetic_rows} synthetic D&D encounters to balance classes...")
synthetic_losses = ctgan.sample(num_synthetic_rows)

# ─── Step 6: Post-generation sanity clamping ──────────────────────────────
# CTGAN can occasionally generate out-of-range values for continuous columns.
# Clamp to physically valid D&D ranges.
synthetic_losses["avg_party_level"] = synthetic_losses["avg_party_level"].clip(1, 20)
synthetic_losses["party_size"] = synthetic_losses["party_size"].clip(1, 8)
synthetic_losses["avg_monster_cr"] = synthetic_losses["avg_monster_cr"].clip(0, 30)
synthetic_losses["avg_monster_hp"] = synthetic_losses["avg_monster_hp"].clip(1, 700)
synthetic_losses["avg_monster_ac"] = synthetic_losses["avg_monster_ac"].clip(5, 25)
synthetic_losses["avg_monster_size_num"] = synthetic_losses[
    "avg_monster_size_num"
].clip(1, 6)
synthetic_losses["avg_monster_stat_sum"] = synthetic_losses[
    "avg_monster_stat_sum"
].clip(10, 180)
# Force binary columns to {0, 1}
for col in [
    "has_healer",
    "has_tank",
    "has_arcane",
    "has_martial_dps",
    "monster_is_legendary",
    "monster_has_mobility",
    "monster_has_physical_res",
    "monster_immune_to_cc",
    "monster_has_magic_res",
    "monster_has_pack_tactics",
    "monster_has_spellcasting",
    "monster_has_regeneration",
    "target",
]:
    synthetic_losses[col] = synthetic_losses[col].round().clip(0, 1).astype(int)

# ─── Step 7: Combine real + synthetic ─────────────────────────────────────
# No manual feature engineering needed here.  The training pipeline's
# DnDFeatureEngineer will compute all derived features (cr_to_party_level,
# true_party_power, mobility_threat, etc.) automatically at train time.
# Synthetic rows carry a sentinel encounter_id: real rows keep their
# campaign id so grouped workflows remain possible downstream.
synthetic_losses["encounter_id"] = "synthetic"
df_balanced = pd.concat(
    [df_model.drop(columns=["campaign"]), synthetic_losses], ignore_index=True
)

print(f"\nOriginal Dataset Size: {len(df_model)}")
print(f"New Balanced Dataset Size: {len(df_balanced)}")
print(f"Wins:   {len(df_balanced[df_balanced['target'] == 1])}")
print(f"Losses: {len(df_balanced[df_balanced['target'] == 0])}")
print(f"\nFinal columns ({len(df_balanced.columns)}):")
print(list(df_balanced.columns))

# Save the final balanced dataset to feed to the pipeline!
df_balanced.to_csv(args.output, index=False)
print(
    f"\nSaved to {args.output}!  NOTE: this CSV is for the gan_ablation.py experiment,\n"
    "NOT for the production model (base-rate shift destroys calibration)."
)
