import joblib, pandas as pd
pipeline = joblib.load('true_lethality_model.pkl')
import __main__
from initial_learn import DnDFeatureEngineer, RAW_INPUT_COLUMNS
__main__.DnDFeatureEngineer = DnDFeatureEngineer

def find_true_cr(hp, ac, size_num, is_leg, has_mob, phys_res, cc_imm, mag_res, pack, spell, regen, stat_sum, baseline_cr):
    for guess_level in [x * 0.25 for x in range(1, 121)]:
        df_sim = pd.DataFrame([{
            'avg_party_level': float(guess_level), 'party_size': 4, 'avg_monster_cr': baseline_cr,
            'avg_monster_hp': hp, 'avg_monster_ac': ac,
            'has_healer': 1, 'has_tank': 1, 'has_arcane': 1, 'has_martial_dps': 1,
            'monster_is_legendary': is_leg, 'monster_has_mobility': has_mob,
            'avg_monster_size_num': size_num, 'avg_monster_stat_sum': stat_sum,
            'monster_has_physical_res': phys_res, 'monster_immune_to_cc': cc_imm,
            'monster_has_magic_res': mag_res, 'monster_has_pack_tactics': pack,
            'monster_has_spellcasting': spell, 'monster_has_regeneration': regen,
        }])
        prob = pipeline.predict_proba(df_sim[list(RAW_INPUT_COLUMNS)])[0, 1]
        if prob >= 0.65:
            return guess_level
    return 30.0

print(find_true_cr(900, 6, 6, 0, 1, 0, 0, 0, 1, 0, 0, 120, 19.75))
