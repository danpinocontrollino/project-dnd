"""Tests for the deathmatch cap in lethality_engine.

Context: the logs record hopeless fights as "won" 84.5% of the time (DM
mercy), which is how the model ended up rating 19 liches as beatable at
level 8 - and, later, one lich as "fair at level 3.25". The cap is now
min(race cap, Battlecast lattice); these tests pin its anchor points and
monotonicity. No trained model needed, just the feature math and the
lattice JSON.
"""

import numpy as np
import pytest

from lethality_engine import (
    _GUARD_A,
    _GUARD_B,
    _GUARD_C,
    MonsterProfile,
    _lattice_cap,
    _load_lattice,
    _party_config,
    _survival_cap,
    encounter_row,
)


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
HP_WALL = MonsterProfile(
    cr=21,
    hp=10_000,
    ac=25,
    size_num=6,
    stat_sum=250,
    dpr=45,
    atk_bonus=12,
    save_dc=20,
    burst=100,
    name="HP Wall",
)


class TestLatticeAnchors:
    """The lattice must reproduce the simulator exactly at grid nodes.

    Reference values are straight out of battlecast_bridge/results.jsonl
    (Lich cells, 2,000 trials each); the monotone massage never touches
    the Lich row because CR 21 is the top of the grid.
    """

    def test_lattice_file_ships_with_the_repo(self):
        assert _load_lattice() is not None, "battlecast_bridge/guard_lattice.json"

    def test_one_lich_is_hopeless_at_level_5(self):
        # Battlecast: 0 wins in 2,000 deathmatches. The old survival-only
        # cap sat at 0.95 here - the attrition blind spot this fixes.
        cap = _survival_cap(_rows_for(LICH, 1, [5]))[0]
        assert cap <= 0.001, f"cap={cap:.4f}"

    def test_one_lich_matches_sim_at_level_9(self):
        cap = _survival_cap(_rows_for(LICH, 1, [9]))[0]
        assert cap == pytest.approx(0.282, abs=0.01)

    def test_one_lich_not_capped_at_high_levels(self):
        # Sim says 0.987 at L13 and 1.0 from L17: the model (ceiling
        # ~0.86), not the cap, must own that region.
        caps = _survival_cap(_rows_for(LICH, 1, [13, 17, 20]))
        assert (caps > 0.9).all(), caps

    def test_two_liches_match_sim(self):
        for level, p_sim in ((9, 0.0), (13, 0.074), (17, 0.725)):
            cap = _survival_cap(_rows_for(LICH, 2, [level]))[0]
            assert cap == pytest.approx(p_sim, abs=0.01), (level, cap)

    def test_mass_boss_hopeless_even_at_level_20(self):
        for n in (8, 19):
            cap = _survival_cap(_rows_for(LICH, n, [20]))[0]
            assert cap <= 0.001, f"{n} liches at L20: cap={cap:.4f}"

    def test_lattice_abstains_below_cr_2(self):
        # Weak monsters were never the bug; goblin swarms are priced by
        # the model and the race cap, not the boss lattice.
        assert _lattice_cap(_rows_for(GOBLIN, 8, [1]))[0] == 1.0


class TestRaceCap:
    def test_constants_are_the_fitted_ones(self):
        # battlecast_bridge/analyze.py refuses to drift silently; keep the
        # test in sync with figures/battlecast_summary.json.
        assert (_GUARD_A, _GUARD_C, _GUARD_B) == (0.1924, 2.0856, 4.1217)

    def test_huge_hp_homebrew_stays_capped(self):
        # Outside the lattice's sight (it clamps to the CR-21 row, which
        # is high at L20 for n=1) the race cap must still catch a
        # 10,000-HP wall: the party survives forever but never wins.
        # What matters is that the ceiling sits far below the 0.65
        # appraisal target, so the verdict is beyond_deadly at any level.
        cap = _survival_cap(_rows_for(HP_WALL, 1, [20]))[0]
        assert cap < 0.10, f"cap={cap:.4f}"

    def test_trivial_encounters_untouched(self):
        cap = _survival_cap(_rows_for(GOBLIN, 1, [1, 10, 20]))
        assert (cap > 0.95).all(), cap


class TestMonotonicity:
    def test_cap_monotone_in_party_level(self):
        levels = list(np.arange(1, 20.5, 0.5))
        for n in (1, 2, 11):
            cap = _survival_cap(_rows_for(LICH, n, levels))
            assert (np.diff(cap) >= -1e-12).all(), f"n={n}"

    def test_cap_monotone_decreasing_in_monster_count(self):
        caps = [_survival_cap(_rows_for(LICH, n, [12]))[0] for n in (1, 2, 4, 8, 16)]
        assert all(b <= a + 1e-12 for a, b in zip(caps, caps[1:])), caps
