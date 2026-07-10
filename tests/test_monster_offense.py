"""Unit tests for monster_offense.py — statblock parsing and DMG tables.

The versatile-weapon test pins the bug fixed in July 2026: "7 (1d8+3)
slashing damage, or 8 (1d10+3) ... two hands" was summed to DPR 15.
"""

import numpy as np
import pandas as pd
import pytest

from monster_offense import (
    _damage_values,
    _parse_multiattack_count,
    cr_to_xp,
    encounter_xp_multiplier,
    extract_legendary_mobility,
    extract_official_traits,
    offense_from_cr,
    parse_statblock_offense,
)

LONGSWORD = (
    "<p><em><strong>Longsword.</strong></em> <em>Melee Weapon Attack:</em> "
    "+5 to hit, reach 5 ft., one target. <em>Hit:</em> 7 (1d8 + 3) slashing "
    "damage, or 8 (1d10 + 3) slashing damage if used with two hands.</p>"
)

FLAME_RIDER = (
    "<p><em><strong>Burning Blade.</strong></em> <em>Melee Weapon Attack:</em> "
    "+6 to hit, reach 5 ft., one target. <em>Hit:</em> 7 (1d8 + 3) slashing "
    "damage plus 3 (1d6) fire damage.</p>"
)


class TestVersatileWeaponFix:
    def test_alternative_damage_is_maxed_not_summed(self):
        assert _damage_values(
            "7 (1d8 + 3) slashing damage, or 8 (1d10 + 3) "
            "slashing damage if used with two hands"
        ) == [8.0]

    def test_rider_damage_is_still_separate(self):
        assert _damage_values(
            "7 (1d8 + 3) slashing damage plus 3 (1d6) " "fire damage"
        ) == [7.0, 3.0]

    def test_full_statblock_versatile(self):
        res = parse_statblock_offense(LONGSWORD, "", cr=1)
        assert res["dpr"] == pytest.approx(8.0)  # not 15

    def test_full_statblock_rider_summed(self):
        res = parse_statblock_offense(FLAME_RIDER, "", cr=1)
        assert res["dpr"] == pytest.approx(10.0)  # 7 + 3

    def test_no_damage_returns_empty(self):
        assert _damage_values("The creature glares menacingly.") == []


class TestMultiattack:
    def test_no_multiattack_is_one(self):
        assert _parse_multiattack_count("Bite. +4 to hit.") == 1

    def test_word_count(self):
        assert (
            _parse_multiattack_count(
                "Multiattack. The dragon makes three attacks: one bite, two claws."
            )
            == 3
        )

    def test_unparsed_defaults_to_two(self):
        assert _parse_multiattack_count("Multiattack. A flurry of blows.") == 2


class TestDmgTables:
    def test_cr_to_xp_exact(self):
        assert cr_to_xp(5) == 1800.0
        assert cr_to_xp(0.25) == 50.0

    def test_cr_to_xp_interpolates(self):
        assert 1800.0 < cr_to_xp(5.5) < 2300.0

    def test_cr_to_xp_nan_is_zero(self):
        assert cr_to_xp(float("nan")) == 0.0
        assert cr_to_xp(None) == 0.0

    def test_offense_from_cr_midpoint(self):
        atk, dpr, dc = offense_from_cr(5)
        assert (atk, dpr, dc) == (6.0, 35.5, 15.0)

    def test_offense_from_cr_nan_maps_to_cr1(self):
        assert offense_from_cr(float("nan")) == offense_from_cr(1)


class TestXpMultiplier:
    def test_base_ladder(self):
        assert encounter_xp_multiplier(1) == 1.0
        assert encounter_xp_multiplier(2) == 1.5
        assert encounter_xp_multiplier(6) == 2.0
        assert encounter_xp_multiplier(10) == 2.5
        assert encounter_xp_multiplier(14) == 3.0
        assert encounter_xp_multiplier(20) == 4.0

    def test_small_party_shifts_up(self):
        # DMG p.83: fewer than 3 PCs -> next multiplier up the ladder.
        assert encounter_xp_multiplier(1, party_size=2) == 1.5
        assert encounter_xp_multiplier(20, party_size=2) == 5.0

    def test_large_party_shifts_down(self):
        # DMG p.83: 6+ PCs -> next multiplier down (x0.5 for one monster).
        assert encounter_xp_multiplier(1, party_size=6) == 0.5
        assert encounter_xp_multiplier(3, party_size=7) == 1.5

    def test_standard_party_unchanged(self):
        for n in (1, 2, 5, 12):
            assert encounter_xp_multiplier(n, party_size=4) == encounter_xp_multiplier(
                n
            )

    def test_nan_party_size_ignored(self):
        assert encounter_xp_multiplier(2, party_size=float("nan")) == 1.5


class TestTraitExtraction:
    def test_all_six_flags(self):
        wri = pd.Series(["nonmagicalres stunnedimmu", "", None])
        add = pd.Series(
            [
                "magic resistance. pack tactics.",
                "innate spellcasting, regenerates",
                None,
            ]
        )
        t = extract_official_traits(wri, add)
        assert t.loc[0].tolist() == [1, 1, 1, 1, 0, 0]
        assert t.loc[1, "spellcasting"] == 1
        assert t.loc[1, "regeneration"] == 1
        assert t.loc[2].sum() == 0  # NaN-safe

    def test_nonmagicalimmu_counts_as_physical_res(self):
        # This variant was missing from the CR predictor's old local regex.
        t = extract_official_traits(pd.Series(["nonmagicalimmu"]), pd.Series([""]))
        assert t.loc[0, "physical_res"] == 1

    def test_frightened_immunity_counts_as_cc(self):
        # This variant was missing from parse_fireball's old local regex.
        t = extract_official_traits(pd.Series(["frightenedimmu"]), pd.Series([""]))
        assert t.loc[0, "cc_immune"] == 1

    def test_legendary_mobility(self):
        lm = extract_legendary_mobility(
            pd.Series(["Legendary Resistance", "", None]),
            pd.Series(["walk 30ft, fly 60ft", "walk 30ft", None]),
        )
        assert lm["is_legendary"].tolist() == [1, 0, 0]
        assert lm["has_mobility"].tolist() == [1, 0, 0]


class TestStatblockEdgeCases:
    def test_empty_returns_none(self):
        assert parse_statblock_offense("", "", cr=1) is None

    def test_dpr_clamped_to_dmg_band(self):
        # A CR-1 monster can't have DPR 400 no matter what the regex reads.
        crazy = (
            "<p><em><strong>Doom.</strong></em> +4 to hit. "
            "<em>Hit:</em> 400 (40d20) necrotic damage.</p>"
        )
        res = parse_statblock_offense(crazy, "", cr=1)
        _, table_dpr, _ = offense_from_cr(1)
        assert res["dpr"] <= 3.0 * table_dpr

    def test_spell_list_feeds_burst(self):
        traits = "<p>Innate Spellcasting: power word kill, fireball</p>"
        res = parse_statblock_offense("", traits, cr=21)
        assert res["burst_dmg"] >= 100.0  # Power Word Kill
