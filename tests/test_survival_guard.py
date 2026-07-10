"""Regression tests for the survivability physics guard.

The FIREBALL logs are contaminated by DM mercy: hopeless fights were still
"won" 84.5% of the time at real tables, so the raw model rated 19 Liches as
beatable by a level-8 party.  ``lethality_engine._survival_cap`` fixes this
by capping P(win) with a sigmoid of ``rounds_to_kill_party``.  These tests
pin the cap's anchor points and its monotonicity properties — they need no
trained model, only the deterministic feature math.
"""

import numpy as np
import pytest

from lethality_engine import (
    _GUARD_A,
    _GUARD_B,
    MonsterProfile,
    _party_config,
    _survival_cap,
    encounter_row,
)


def _cap_sigmoid(rtk: float) -> float:
    return 1.0 / (1.0 + np.exp(-(_GUARD_A * rtk + _GUARD_B)))


class TestGuardAnchors:
    """The documented anchor points of the cap function.

    Constants are calibrated against the Battlecast deathmatch grid (180
    cells x 2,000 trials; logistic fit of simulated P(win) on
    rounds_to_kill_party — see battlecast_bridge/PROVENANCE.md), replacing
    the original hand-tuned 0.10/0.50/0.90 anchors which proved slightly
    too generous in the 2-3 round zone.
    """

    def test_party_deleted_in_one_round_is_hopeless(self):
        assert _cap_sigmoid(1.0) == pytest.approx(0.087, abs=0.01)

    def test_two_round_survival_is_a_long_shot(self):
        # Simulated deathmatches: surviving only ~2 rounds wins ~1/3 of the
        # time, not the coin flip the hand-tuned guard assumed.
        assert _cap_sigmoid(2.0) == pytest.approx(0.33, abs=0.01)

    def test_three_round_survival_mostly_unbinds(self):
        assert _cap_sigmoid(3.0) == pytest.approx(0.71, abs=0.01)

    def test_normal_encounters_unaffected(self):
        # The serving model's ceiling is ~0.89 (empirical base-rate cap),
        # so the guard is inert wherever cap > 0.93 — i.e. rtk >= 4.
        assert _cap_sigmoid(4.0) > 0.92
        assert _cap_sigmoid(5.0) > 0.98


def _rows_for(monster, num_monsters, levels, party_size=4):
    return [
        encounter_row(
            monster,
            num_monsters=num_monsters,
            **_party_config(
                level, party_size, {"healer": 1, "tank": 1, "arcane": 1, "dps": 1}
            ),
        )
        for level in levels
    ]


LICH = MonsterProfile(
    cr=21,
    hp=135,
    ac=17,
    size_num=3,
    stat_sum=193,
    is_legendary=1,
    magic_res=0,
    spellcasting=1,
    atk_bonus=12,
    dpr=45,
    save_dc=20,
    burst=100,
    name="Lich",
)
GOBLIN = MonsterProfile(
    cr=0.25,
    hp=7,
    ac=15,
    size_num=2,
    stat_sum=61,
    atk_bonus=4,
    dpr=5,
    save_dc=13,
    burst=5,
    name="Goblin",
)


class TestGuardOnEncounters:
    def test_mass_boss_capped_even_at_level_20(self):
        cap = _survival_cap(_rows_for(LICH, 19, [20]))[0]
        assert cap < 0.20, f"19 Liches at L20 must be near-hopeless, cap={cap:.3f}"

    def test_single_boss_not_capped_at_crossing_levels(self):
        # One lich around its appraised level ~3-5: the model, not the
        # guard, must own that region (cap comfortably above the target).
        cap = _survival_cap(_rows_for(LICH, 1, [5]))[0]
        assert cap > 0.80, f"guard must not dominate a single-boss fight, cap={cap:.3f}"

    def test_trivial_encounters_untouched(self):
        cap = _survival_cap(_rows_for(GOBLIN, 1, [1, 10, 20]))
        assert (cap > 0.99).all()

    def test_cap_monotone_in_party_level(self):
        levels = list(np.arange(1, 20.5, 0.5))
        cap = _survival_cap(_rows_for(LICH, 11, levels))
        assert (np.diff(cap) >= -1e-12).all()

    def test_cap_monotone_decreasing_in_monster_count(self):
        caps = [_survival_cap(_rows_for(LICH, n, [12]))[0] for n in (1, 2, 4, 8, 16)]
        assert all(b <= a + 1e-12 for a, b in zip(caps, caps[1:])), caps
