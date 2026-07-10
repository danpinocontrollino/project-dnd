"""Unit tests for the DnDFeatureEngineer and the shared inference engine."""

import numpy as np
import pandas as pd
import pytest

from initial_learn import (
    DnDFeatureEngineer,
    FEATURE_COLUMNS,
    MONOTONE_CONSTRAINTS,
    RAW_INPUT_COLUMNS,
    _classify_xp_tier,
    parse_cr_value,
)
from lethality_engine import MonsterProfile, encounter_row, roster_monster_fields


def _row(**overrides):
    base = dict(
        avg_party_level=5.0,
        party_size=4,
        num_monsters=1,
        avg_monster_cr=5.0,
        avg_monster_hp=100.0,
        avg_monster_ac=15.0,
        has_healer=1,
        has_tank=1,
        has_arcane=1,
        has_martial_dps=1,
        monster_is_legendary=0,
        monster_has_mobility=0,
        avg_monster_size_num=3.0,
        avg_monster_stat_sum=150.0,
        monster_has_physical_res=0,
        monster_immune_to_cc=0,
        monster_has_magic_res=0,
        monster_has_pack_tactics=0,
        monster_has_spellcasting=0,
        monster_has_regeneration=0,
    )
    base.update(overrides)
    return base


def _transform(rows):
    df = pd.DataFrame(rows)
    eng = DnDFeatureEngineer().fit(df)
    return eng.transform(df)


class TestFeatureEngineer:
    def test_output_matches_feature_columns(self):
        out = _transform([_row()])
        assert list(out.columns) == list(FEATURE_COLUMNS)

    def test_optional_columns_derived(self):
        out = _transform([_row()])
        # DMG p.274 CR-5 midpoint DPR = 35.5, one monster.
        assert out.loc[0, "avg_monster_dpr"] == pytest.approx(35.5)
        assert out.loc[0, "total_monster_dpr"] == pytest.approx(35.5)

    def test_nan_in_existing_total_columns_filled_from_row(self):
        # Regression: NaN in an existing total_* column used to fall through
        # to the global median imputer instead of the row-derived value.
        row = _row(
            num_monsters=3,
            total_monster_hp=np.nan,
            total_monster_xp=np.nan,
            max_monster_cr=np.nan,
            num_monsters_total=np.nan,
            total_monster_dpr=np.nan,
            avg_monster_dpr=20.0,
            max_monster_atk_bonus=6.0,
            max_monster_save_dc=15.0,
            max_monster_burst=20.0,
        )
        out = _transform([row])
        assert out.loc[0, "monster_hp_per_player"] == pytest.approx(300.0 / 4)
        assert out.loc[0, "total_monster_dpr"] == pytest.approx(60.0)
        assert out.loc[0, "max_monster_cr"] == pytest.approx(5.0)

    def test_ood_clipping(self):
        out = _transform([_row(avg_monster_hp=50_000, avg_party_level=99)])
        assert out.loc[0, "avg_party_level"] == 20
        assert out.loc[0, "monster_hp_per_player"] <= 8000 / 1  # clipped bands

    def test_lethality_ratio_sign(self):
        easy = _transform([_row(avg_monster_cr=0.25, avg_monster_hp=5)])
        hard = _transform(
            [_row(avg_monster_cr=24, avg_monster_hp=600, avg_party_level=3)]
        )
        assert easy.loc[0, "lethality_log_ratio"] > hard.loc[0, "lethality_log_ratio"]

    def test_party_size_multiplier_affects_xp_ratio(self):
        # Same encounter, 2 PCs vs 6 PCs: the DMG p.83 adjustment must make
        # the small party's adjusted budget ratio strictly larger even after
        # threshold scaling.
        small = _transform([_row(party_size=2)])
        large = _transform([_row(party_size=6)])
        assert small.loc[0, "xp_budget_ratio"] > large.loc[0, "xp_budget_ratio"]


class TestMonotoneConstraintCoverage:
    def test_every_feature_has_a_constraint_decision(self):
        # Every serving feature must appear in the constraints map (0 is a
        # valid decision, but absence means "forgot to think about it").
        missing = [c for c in FEATURE_COLUMNS if c not in MONOTONE_CONSTRAINTS]
        assert missing == []

    def test_no_stale_constraints(self):
        stale = [c for c in MONOTONE_CONSTRAINTS if c not in FEATURE_COLUMNS]
        assert stale == []


class TestHelpers:
    def test_parse_cr_value_fraction(self):
        assert parse_cr_value("1/4") == 0.25
        assert parse_cr_value("17") == 17.0
        assert np.isnan(parse_cr_value("garbage"))

    def test_classify_xp_tier(self):
        # Level 5, party of 4: deadly threshold = 1100*4 = 4400.
        assert _classify_xp_tier(100, 5, 4) == 0  # easy
        assert _classify_xp_tier(4400, 5, 4) == 3  # deadly
        assert _classify_xp_tier(9000, 5, 4) == 4  # super-deadly


class TestRosterAggregation:
    def test_single_monster_counts(self):
        m = MonsterProfile(cr=5, hp=100, ac=15)
        f = roster_monster_fields([(m, 3)])
        assert f["num_monsters"] == 3
        assert f["total_monster_hp"] == 300
        assert f["avg_monster_hp"] == 100

    def test_mixed_roster_weighted_mean_and_maxima(self):
        lich = MonsterProfile(cr=21, hp=135, ac=17, is_legendary=1)
        goblin = MonsterProfile(cr=0.25, hp=7, ac=15)
        f = roster_monster_fields([(lich, 1), (goblin, 6)])
        assert f["num_monsters_total"] == 7
        assert f["max_monster_cr"] == 21
        assert f["monster_is_legendary"] == 1  # any-monster max
        assert f["total_monster_hp"] == 135 + 6 * 7  # count-weighted sum
        assert f["avg_monster_cr"] == pytest.approx((21 + 6 * 0.25) / 7)

    def test_encounter_row_covers_all_raw_inputs(self):
        m = MonsterProfile(cr=5, hp=100, ac=15)
        row = encounter_row(
            m,
            avg_party_level=5,
            party_size=4,
            num_monsters=1,
            has_healer=1,
            has_tank=1,
            has_arcane=1,
            has_martial_dps=1,
        )
        missing = [c for c in RAW_INPUT_COLUMNS if c not in row]
        assert missing == []


class TestMonsterProfileImputation:
    def test_missing_offense_imputed_from_cr(self):
        m = MonsterProfile(cr=5, hp=100, ac=15)
        assert m.dpr == pytest.approx(35.5)
        assert m.atk_bonus == 6.0
        assert m.burst == m.dpr  # no statblock nova -> burst == DPR

    def test_zero_attack_bonus_is_preserved(self):
        # Regression: the old UI 0-sentinel made atk_bonus=0 impossible.
        m = MonsterProfile(cr=0, hp=1, ac=5, atk_bonus=0)
        assert m.atk_bonus == 0


class TestCrPredictorArtifacts:
    def test_json_loader_uses_raw_booster(self):
        # Regression for TWO cross-version failures: (a) xgboost pickles
        # break across xgboost versions, (b) the sklearn wrapper's
        # load_model breaks on old-xgboost + new-sklearn ("_estimator_type
        # undefined").  The loader must return a raw Booster from the JSON.
        import xgboost as xgb
        from lethality_engine import load_cr_predictor, predict_wotc_cr

        model = load_cr_predictor()
        assert isinstance(model, xgb.Booster)
        cr = predict_wotc_cr(
            model,
            {
                "hp": 135,
                "ac": 17,
                "stat_sum": 160,
                "size_num": 3,
                "is_legendary": 1,
                "has_mobility": 0,
                "physical_res": 0,
                "cc_immune": 1,
                "magic_res": 0,
                "pack_tactics": 0,
                "spellcasting": 1,
                "regeneration": 0,
            },
        )
        assert 0.25 <= cr <= 30
        assert (cr * 4) == int(cr * 4)  # quarter-step snapping

    def test_predict_wotc_cr_is_order_insensitive(self):
        from lethality_engine import load_cr_predictor, predict_wotc_cr

        model = load_cr_predictor()
        feats = {
            "hp": 200,
            "ac": 18,
            "stat_sum": 170,
            "size_num": 5,
            "is_legendary": 1,
            "has_mobility": 1,
            "physical_res": 1,
            "cc_immune": 1,
            "magic_res": 1,
            "pack_tactics": 0,
            "spellcasting": 0,
            "regeneration": 0,
        }
        scrambled = dict(reversed(list(feats.items())))
        assert predict_wotc_cr(model, feats) == predict_wotc_cr(model, scrambled)

    def test_booster_matches_legacy_pickle(self):
        import joblib
        from lethality_engine import load_cr_predictor, predict_wotc_cr

        booster = load_cr_predictor()
        wrapper = joblib.load("cr_predictor_model.pkl")
        feats = {
            "hp": 135,
            "ac": 17,
            "stat_sum": 160,
            "size_num": 3,
            "is_legendary": 1,
            "has_mobility": 0,
            "physical_res": 0,
            "cc_immune": 1,
            "magic_res": 0,
            "pack_tactics": 0,
            "spellcasting": 1,
            "regeneration": 0,
        }
        assert predict_wotc_cr(booster, feats) == predict_wotc_cr(wrapper, feats)
