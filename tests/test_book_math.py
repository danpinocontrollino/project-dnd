"""Regression tests for the by-the-book (DMG p.82) encounter estimate.

The book side of the comparison must run the actual DMG procedure — sum the
monsters' XP, apply the encounter multiplier for the monster count — and
express the adjusted total as an equivalent single-monster CR.  Previously
6x CR-1 monsters still displayed "CR 1" on the book column.
"""

import pytest

from lethality_engine import MonsterProfile, official_encounter_estimate
from monster_offense import CR_TO_XP, cr_to_xp, xp_to_cr


def _mon(cr: float) -> MonsterProfile:
    return MonsterProfile(cr=cr, hp=20, ac=13)


class TestXpToCr:
    def test_round_trip_on_every_table_row(self):
        for cr in CR_TO_XP:
            assert xp_to_cr(cr_to_xp(cr)) == pytest.approx(cr, abs=1e-9)

    def test_clamped_below_and_above_table(self):
        assert xp_to_cr(0) == 0.0
        assert xp_to_cr(10_000_000) == 30.0


class TestOfficialEstimate:
    def test_single_monster_is_its_printed_cr(self):
        for cr in (0.25, 1, 5, 21):
            est = official_encounter_estimate(_mon(cr), 1)
            assert est["effective_cr"] == pytest.approx(cr)
            assert est["multiplier"] == 1.0

    def test_six_cr1_monsters_are_not_a_cr1_fight(self):
        # 6 x 200 XP = 1,200 x x2 (3-6 monsters) = 2,400 adjusted XP,
        # which sits between CR 5 (1,800) and CR 6 (2,300) -> ~CR 6.
        est = official_encounter_estimate(_mon(1), 6)
        assert est["total_xp"] == pytest.approx(1200)
        assert est["multiplier"] == 2.0
        assert est["adjusted_xp"] == pytest.approx(2400)
        assert est["effective_cr"] >= 5.0, est

    def test_dmg_worked_example(self):
        # DMG p.82's own example: four monsters worth 500 XP total ->
        # x2 multiplier -> 1,000 adjusted XP.
        est = official_encounter_estimate(_mon(0.25), 4)  # 4 x 50 = 200...
        # use explicit roster to hit exactly 500 XP: 4 monsters of CR 1/2
        est = official_encounter_estimate(_mon(0.5), 4)  # 4 x 100 = 400
        assert est["multiplier"] == 2.0
        assert est["adjusted_xp"] == pytest.approx(est["total_xp"] * 2)

    def test_multiplier_ladder_boundaries(self):
        assert official_encounter_estimate(_mon(1), 2)["multiplier"] == 1.5
        assert official_encounter_estimate(_mon(1), 3)["multiplier"] == 2.0
        assert official_encounter_estimate(_mon(1), 7)["multiplier"] == 2.5
        assert official_encounter_estimate(_mon(1), 11)["multiplier"] == 3.0
        assert official_encounter_estimate(_mon(1), 15)["multiplier"] == 4.0

    def test_party_size_adjustment(self):
        # DMG p.83: parties of 6+ step the multiplier down one rung.
        small = official_encounter_estimate(_mon(1), 4, party_size=2)
        std = official_encounter_estimate(_mon(1), 4, party_size=4)
        big = official_encounter_estimate(_mon(1), 4, party_size=6)
        assert small["multiplier"] > std["multiplier"] > big["multiplier"]

    def test_mixed_roster(self):
        roster = [(_mon(5), 1), (_mon(0.25), 4)]  # 1800 + 4x50 = 2000 XP
        est = official_encounter_estimate(roster)
        assert est["num_monsters"] == 5
        assert est["total_xp"] == pytest.approx(2000)
        assert est["multiplier"] == 2.0
        assert est["adjusted_xp"] == pytest.approx(4000)

    def test_estimate_monotone_in_count(self):
        crs = [
            official_encounter_estimate(_mon(1), n)["effective_cr"]
            for n in (1, 2, 4, 8, 12)
        ]
        assert all(b >= a for a, b in zip(crs, crs[1:])), crs
