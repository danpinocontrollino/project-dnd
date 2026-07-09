import pandas as pd
import numpy as np
from ctgan import CTGAN

print("Loading the clean D&D dataset...")
df = pd.read_csv('clean_aggregated_combat_data.csv')

# ─── Step 1: Light pre-processing for the BASE columns ───────────────────
# With the Area 3 self-contained pipeline, CTGAN should generate RAW
# independent variables.  The pipeline's DnDFeatureEngineer will compute
# all derived features (ratios, power curves, interactions) automatically.
class_columns = [
    'num_barbarian', 'num_bard', 'num_cleric', 'num_druid', 'num_fighter',
    'num_monk', 'num_paladin', 'num_ranger', 'num_rogue', 'num_sorcerer',
    'num_warlock', 'num_wizard',
]

df['party_size'] = df[class_columns].sum(axis=1).replace(0, 1)
df['avg_party_level'] = df['avg_party_level'].replace(0, 1)

df['has_healer'] = ((df['num_cleric'] > 0) | (df['num_druid'] > 0) | (df['num_bard'] > 0)).astype(int)
df['has_tank'] = ((df['num_barbarian'] > 0) | (df['num_fighter'] > 0) | (df['num_paladin'] > 0)).astype(int)
df['has_arcane'] = ((df['num_wizard'] > 0) | (df['num_sorcerer'] > 0) | (df['num_warlock'] > 0)).astype(int)
df['has_martial_dps'] = ((df['num_rogue'] > 0) | (df['num_monk'] > 0) | (df['num_ranger'] > 0)).astype(int)

# Binary target
df['target'] = df['final_outcome'].apply(lambda x: 1 if x == 'Party Win' else 0)

# ─── BASE features for CTGAN ─────────────────────────────────────────────
# These are the RAW independent variables CTGAN will learn.
# NOTE: We use avg_monster_cr and avg_monster_hp (raw stats) instead of
#       cr_to_party_level and monster_hp_per_player (derived ratios).
#       The pipeline's DnDFeatureEngineer will derive those automatically.
#       We do NOT include interaction features (mobility_threat, etc.)
#       because they are deterministic functions of these base columns.
base_features = [
    'avg_party_level', 'party_size',
    'avg_monster_cr', 'avg_monster_hp', 'avg_monster_ac',
    'has_healer', 'has_tank', 'has_arcane', 'has_martial_dps',
    # Area 1: Monster trait base columns (from parse_fireball.py)
    'monster_is_legendary', 'monster_has_mobility',
    'avg_monster_size_num', 'avg_monster_stat_sum',
    'monster_has_physical_res', 'monster_immune_to_cc', 'monster_has_magic_res',
    'monster_has_pack_tactics', 'monster_has_spellcasting', 'monster_has_regeneration',
    # Target (always last)
    'target',
]

df_model = df[base_features].dropna()

# ─── Step 2: Isolate the Minority Class (The Losses) ─────────────────────
df_losses = df_model[df_model['target'] == 0]
print(f"Training GAN on {len(df_losses)} real Party Losses...")

# ─── Step 3: Discrete columns for the GAN ────────────────────────────────
# Prevents CTGAN from generating "0.5 Healers" or "3.14 Party Size".
discrete_columns = [
    'party_size',
    'has_healer', 'has_tank', 'has_arcane', 'has_martial_dps',
    'monster_is_legendary', 'monster_has_mobility',
    'monster_has_physical_res', 'monster_immune_to_cc', 'monster_has_magic_res',
    'monster_has_pack_tactics', 'monster_has_spellcasting', 'monster_has_regeneration',
    'target',
]

# ─── Step 4: Train the Generative Adversarial Network ────────────────────
ctgan = CTGAN(epochs=300, verbose=True)
ctgan.fit(df_losses, discrete_columns)

# ─── Step 5: Generate Synthetic Losses ────────────────────────────────────
num_synthetic_rows = len(df_model[df_model['target'] == 1]) - len(df_losses)
print(f"Generating {num_synthetic_rows} synthetic D&D encounters to balance classes...")
synthetic_losses = ctgan.sample(num_synthetic_rows)

# ─── Step 6: Post-generation sanity clamping ──────────────────────────────
# CTGAN can occasionally generate out-of-range values for continuous columns.
# Clamp to physically valid D&D ranges.
synthetic_losses['avg_party_level'] = synthetic_losses['avg_party_level'].clip(1, 20)
synthetic_losses['party_size'] = synthetic_losses['party_size'].clip(1, 8)
synthetic_losses['avg_monster_cr'] = synthetic_losses['avg_monster_cr'].clip(0, 30)
synthetic_losses['avg_monster_hp'] = synthetic_losses['avg_monster_hp'].clip(1, 700)
synthetic_losses['avg_monster_ac'] = synthetic_losses['avg_monster_ac'].clip(5, 25)
synthetic_losses['avg_monster_size_num'] = synthetic_losses['avg_monster_size_num'].clip(1, 6)
synthetic_losses['avg_monster_stat_sum'] = synthetic_losses['avg_monster_stat_sum'].clip(10, 180)
# Force binary columns to {0, 1}
for col in ['has_healer', 'has_tank', 'has_arcane', 'has_martial_dps',
            'monster_is_legendary', 'monster_has_mobility', 
            'monster_has_physical_res', 'monster_immune_to_cc', 'monster_has_magic_res',
            'monster_has_pack_tactics', 'monster_has_spellcasting', 'monster_has_regeneration',
            'target']:
    synthetic_losses[col] = synthetic_losses[col].round().clip(0, 1).astype(int)

# ─── Step 7: Combine real + synthetic ─────────────────────────────────────
# No manual feature engineering needed here.  The training pipeline's
# DnDFeatureEngineer will compute all derived features (cr_to_party_level,
# true_party_power, mobility_threat, etc.) automatically at train time.
df_balanced = pd.concat([df_model, synthetic_losses], ignore_index=True)

print(f"\nOriginal Dataset Size: {len(df_model)}")
print(f"New Balanced Dataset Size: {len(df_balanced)}")
print(f"Wins:   {len(df_balanced[df_balanced['target'] == 1])}")
print(f"Losses: {len(df_balanced[df_balanced['target'] == 0])}")
print(f"\nFinal columns ({len(df_balanced.columns)}):")
print(list(df_balanced.columns))

# Save the final balanced dataset to feed to the pipeline!
df_balanced.to_csv('gan_balanced_combat_data.csv', index=False)
print("\nSaved to gan_balanced_combat_data.csv! Ready to train the Random Forest.")