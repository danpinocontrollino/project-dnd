"""Sanity checks the deployed model has to pass. Run after ANY model change:

    python3 behavior_suite.py

Good AUC is not enough - we had a logistic model that beat xgb on AUC and
then rated 8 liches as a trivial fight. The app asks what-if questions
(more monsters, higher level), so each check below is a rule of the game
the model can't be allowed to break. Non-zero exit if anything fails.
"""

from __future__ import annotations

import sys
import warnings

import __main__
import joblib
import numpy as np

from initial_learn import DnDFeatureEngineer
from lethality_engine import (
    MonsterProfile,
    encounter_row,
    lethality_appraisal,
    load_monster_database,
    predict_win_probability,
    profile_from_db_row,
)

__main__.DnDFeatureEngineer = DnDFeatureEngineer
warnings.filterwarnings("ignore")

FAILURES: list = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    pipe = joblib.load("true_lethality_model.pkl")
    db = load_monster_database()

    def prof(name: str) -> MonsterProfile:
        return profile_from_db_row(db[db["Name"].str.lower() == name.lower()].iloc[0])

    lich, ogre, goblin = prof("Lich"), prof("Ogre"), prof("Goblin")

    print("1. Monster-count monotonicity (more liches can never be easier):")
    levels = [lethality_appraisal(pipe, lich, n)["level"] for n in (1, 2, 3, 4, 6, 8)]
    check(
        "lich count scaling",
        all(b >= a for a, b in zip(levels, levels[1:])),
        f"levels={levels}",
    )

    print("2. Roster dominance (boss + minions >= boss alone):")
    solo = lethality_appraisal(pipe, lich, 1)["level"]
    mixed = lethality_appraisal(pipe, [(lich, 1), (ogre, 2), (goblin, 6)])["level"]
    check("dominance", mixed >= solo, f"solo={solo}, mixed={mixed}")

    print("3. Win probability monotone non-decreasing in party level:")
    grid = np.arange(1, 20.001, 0.05)
    rows = [
        encounter_row(
            lich,
            avg_party_level=l,
            party_size=4,
            num_monsters=1,
            has_healer=1,
            has_tank=1,
            has_arcane=1,
            has_martial_dps=1,
        )
        for l in grid
    ]
    p = predict_win_probability(pipe, rows)
    check("level monotonicity", bool(np.all(np.diff(p) >= -1e-9)))
    check(
        "curve not a staircase (calibration granularity)",
        len(np.unique(p.round(5))) >= 30,
        f"{len(np.unique(p.round(5)))} distinct values",
    )

    print("3b. Mass-boss physics (DM-mercy data must not make TPKs look fair):")
    for n in (11, 19):
        a = lethality_appraisal(pipe, lich, n)
        check(
            f"{n}x Lich is beyond deadly",
            a["verdict"] == "beyond_deadly",
            f"verdict={a['verdict']}, level={a['level']}, p20={a['p_level_20']:.2f}",
        )
    from lethality_engine import predict_win_for_parties, _party_config, _BALANCED

    guarded = predict_win_for_parties(
        pipe,
        lich,
        [_party_config(l, 4, _BALANCED) for l in np.arange(1, 20.001, 0.25)],
        11,
    )
    check(
        "guarded curve still monotone in level", bool(np.all(np.diff(guarded) >= -1e-9))
    )

    print("4. Tier ordering (P(win) at level 1 decreases with monster strength):")
    p1s = [
        lethality_appraisal(pipe, prof(n), 1)["p_level_1"]
        for n in ("Goblin", "Ogre", "Aboleth", "Tarrasque")
    ]
    check(
        "goblin > ogre > aboleth > tarrasque at level 1",
        all(b <= a + 0.02 for a, b in zip(p1s, p1s[1:])),
        f"p1={[round(x, 2) for x in p1s]}",
    )

    print("5. Boundary verdicts:")
    check(
        "goblin trivial", lethality_appraisal(pipe, goblin, 1)["verdict"] == "trivial"
    )
    check(
        "tarrasque beyond_deadly",
        lethality_appraisal(pipe, prof("Tarrasque"), 1)["verdict"] == "beyond_deadly",
    )

    print("6. OOD homebrew via the real app flow (CR predictor -> appraisal):")
    import pandas as pd

    try:
        crp = joblib.load("cr_predictor_model.pkl")
        df_cr = pd.DataFrame(
            [
                {
                    "hp": 2000,
                    "ac": 30,
                    "stat_sum": 250,
                    "size_num": 6,
                    "is_legendary": 1,
                    "has_mobility": 1,
                    "physical_res": 1,
                    "cc_immune": 1,
                    "magic_res": 1,
                    "pack_tactics": 0,
                    "spellcasting": 1,
                    "regeneration": 1,
                }
            ]
        )
        god_cr = max(0.25, round(float(crp.predict(df_cr)[0]) * 4) / 4)
        god = MonsterProfile(
            cr=god_cr,
            hp=10000,
            ac=50,
            dpr=999,
            atk_bonus=99,
            save_dc=50,
            burst=400,
            is_legendary=1,
            regeneration=1,
            magic_res=1,
            physical_res=1,
            cc_immune=1,
            stat_sum=250,
            size_num=6,
        )
        verdict = lethality_appraisal(pipe, god, 1)["verdict"]
        check(
            "10k-HP god-monster beyond_deadly",
            verdict == "beyond_deadly",
            f"CR predictor said {god_cr}, verdict={verdict}",
        )
    except FileNotFoundError:
        check(
            "10k-HP god-monster (cr_predictor missing)",
            False,
            "cr_predictor_model.pkl not found",
        )

    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} check(s) FAILED: {FAILURES}")
        return 1
    print("✅ All behavioral checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
