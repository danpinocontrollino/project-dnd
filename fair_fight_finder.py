"""Fair Fight Finder — CLI for the True Lethality Engine.

Interactive terminal twin of the Streamlit app.  All simulation logic lives
in ``lethality_engine`` so the CLI and web app can never drift apart.
"""

from __future__ import annotations

import difflib
import sys
import warnings

import __main__
import joblib
import pandas as pd

from initial_learn import DnDFeatureEngineer, _classify_xp_tier
from lethality_engine import (
    PARTY_COMPOSITIONS,
    ROLE_DEFINITIONS,
    TARGET_WIN_RATE,
    MonsterProfile,
    fair_fight_matches,
    lethality_appraisal,
    load_cr_predictor,
    load_monster_database,
    official_encounter_estimate,
    predict_wotc_cr,
    profile_from_db_row,
    roster_monster_fields,
    simulate_party_grid,
)
from monster_offense import encounter_xp_multiplier, offense_from_cr

# Backward-compat shim for OLD pickles trained via `python3 initial_learn.py`.
__main__.DnDFeatureEngineer = DnDFeatureEngineer
warnings.filterwarnings("ignore")

TIER_NAMES = {0: "Easy", 1: "Medium", 2: "Hard", 3: "Deadly", 4: "SUPER-DEADLY"}


def _ask_int(prompt: str, default: int = 0) -> int:
    raw = input(prompt).strip()
    return int(raw) if raw.isdigit() else default


def _ask_flag(prompt: str) -> int:
    return 1 if input(prompt).strip() in {"1", "y", "yes"} else 0


def load_pipeline():
    try:
        pipeline = joblib.load("true_lethality_model.pkl")
    except FileNotFoundError:
        sys.exit("Error: true_lethality_model.pkl not found — run initial_learn.py first.")
    if "feature_engineer" not in getattr(pipeline, "named_steps", {}):
        sys.exit("Legacy model detected (no bundled feature engineering) — retrain.")
    cr_predictor = load_cr_predictor()
    if cr_predictor is None:
        print("Note: no CR predictor artifact found — WotC-rating prediction disabled.")
    return pipeline, cr_predictor


def prompt_homebrew(cr_predictor) -> MonsterProfile:
    print("\n--- CUSTOM / HOMEBREW APPRAISAL ---")
    hp = float(input("Monster HP (e.g. 150): "))
    ac = float(input("Monster AC (e.g. 17): "))
    size_map = {"tiny": 1, "small": 2, "medium": 3, "large": 4, "huge": 5, "gargantuan": 6}
    size_num = size_map.get(input("Size (Tiny..Gargantuan): ").strip().lower(), 3)

    is_leg = _ask_flag("Legendary? (1/0): ")
    has_mob = _ask_flag("Fly or swim speed? (1/0): ")
    phys_res = _ask_flag("Nonmagical physical resistance? (1/0): ")
    cc_imm = _ask_flag("Immune to stun/paralyze/charm? (1/0): ")
    mag_res = _ask_flag("Magic resistance? (1/0): ")
    pack = _ask_flag("Pack tactics? (1/0): ")
    spell = _ask_flag("Spellcaster? (1/0): ")
    regen = _ask_flag("Regeneration? (1/0): ")

    stat_raw = input("Ability score sum (Enter = auto-estimate): ").strip()
    stat_sum = float(stat_raw) if stat_raw else min(50 + hp * 0.3, 180)

    dpr_raw = input("Damage per round (Enter = auto from CR): ").strip()
    atk_raw = input("Attack bonus (Enter = auto): ").strip()
    dc_raw = input("Save DC (Enter = auto): ").strip()

    baseline_cr = max(0.25, round((hp / 15) * 4) / 4)
    if cr_predictor is not None:
        baseline_cr = predict_wotc_cr(cr_predictor, {
            "hp": min(hp, 2000), "ac": max(10, min(ac, 30)),
            "stat_sum": max(60, min(stat_sum, 250)), "size_num": size_num,
            "is_legendary": is_leg, "has_mobility": has_mob,
            "physical_res": phys_res, "cc_immune": cc_imm,
            "magic_res": mag_res, "pack_tactics": pack,
            "spellcasting": spell, "regeneration": regen,
        })
        print(f"\n⚖️  XGBoost predicts WotC would rate this: CR {baseline_cr:g}")

    return MonsterProfile(
        cr=baseline_cr, hp=hp, ac=ac, size_num=size_num, stat_sum=stat_sum,
        is_legendary=is_leg, has_mobility=has_mob, physical_res=phys_res,
        cc_immune=cc_imm, magic_res=mag_res, pack_tactics=pack,
        spellcasting=spell, regeneration=regen,
        dpr=float(dpr_raw) if dpr_raw else None,
        atk_bonus=float(atk_raw) if atk_raw else None,
        save_dc=float(dc_raw) if dc_raw else None,
    )


def print_composition_legend() -> None:
    print("\n─── PARTY ROLES & COMPOSITIONS ───")
    for role, d in ROLE_DEFINITIONS.items():
        print(f"  {role:<12} = {d['classes']}")
    for comp in PARTY_COMPOSITIONS:
        roles = [
            label for label, key in
            [("healer", "healer"), ("tank", "tank"),
             ("arcane", "arcane"), ("martial DPS", "dps")]
            if comp[key]
        ]
        print(f"  {comp['name']:<14} [{', '.join(roles) or 'none'}]")
        print(f"    {comp['desc']}")


def acquire_monster(query: str, db: pd.DataFrame, cr_predictor):
    """Resolve a name against the bestiary, or fall through to homebrew."""
    if query in {"c", "custom"} or (not query):
        monster = prompt_homebrew(cr_predictor)
        est = offense_from_cr(monster.cr)
        if monster.dpr == est[1]:
            print(f"   Offense auto-estimated: DPR {est[1]:.0f} · +{est[0]:.0f} · DC {est[2]:.0f}")
        return monster
    if not db.empty:
        matches = difflib.get_close_matches(query, db["clean_name"].dropna(), n=1, cutoff=0.6)
        if matches:
            monster = profile_from_db_row(db[db["clean_name"] == matches[0]].iloc[0])
            print(
                f"✅ {monster.name} — CR {monster.cr:g} · {monster.hp:.0f} HP · "
                f"AC {monster.ac:.0f} · DPR {monster.dpr:.0f} · "
                f"burst {monster.burst:.0f} · +{monster.atk_bonus:.0f} to hit · "
                f"DC {monster.save_dc:.0f}"
            )
            return monster
    print(f"❌ No match for '{query}'.")
    return None


def main() -> None:
    print("Loading True Lethality Engine...")
    pipeline, cr_predictor = load_pipeline()

    print("Loading official monster database...")
    try:
        db = load_monster_database()
    except Exception as exc:
        print(f"Warning: bestiary unavailable ({exc}); homebrew mode only.")
        db = pd.DataFrame()

    print("\n=== DUNGEON MASTER: FAIR FIGHT FINDER ===")
    print("Build your encounter — mix official and homebrew monsters freely.")

    roster = []
    while True:
        if roster:
            query = input(
                "\nAdd another monster? (name / 'c' for custom / Enter to finish): "
            ).strip().lower()
            if not query:
                break
        else:
            query = input("Monster name ('c' or Enter for custom/homebrew): ").strip().lower()
        monster = acquire_monster(query, db, cr_predictor)
        if monster is None:
            continue
        count = _ask_int(f"How many {monster.name}? [1]: ", default=1) or 1
        roster.append((monster, count))

    fields = roster_monster_fields(roster)
    n_total = fields["num_monsters_total"]
    weighted_cr = fields["avg_monster_cr"]
    print(
        f"\nEncounter: "
        + ", ".join(f"{c}x {m.name}" for m, c in roster)
        + f"\nTotals: {n_total:.0f} monsters · {fields['total_monster_hp']:.0f} pooled HP · "
          f"{fields['total_monster_dpr']:.0f} pooled DPR · apex CR {fields['max_monster_cr']:g} · "
          f"nova {fields['max_monster_burst']:.0f} dmg"
    )

    print("\nBinary-searching the lethality frontier...")
    appraisal = lethality_appraisal(pipeline, roster)

    book = official_encounter_estimate(roster)
    if book["num_monsters"] > 1:
        print(
            f"\n📖 By-the-book (DMG p.82): {book['total_xp']:,.0f} XP total"
            f" x {book['multiplier']:g} multiplier for"
            f" {book['num_monsters']:.0f} monsters ="
            f" {book['adjusted_xp']:,.0f} adjusted XP"
            f" ~ one CR {book['effective_cr']:g} monster"
        )
    else:
        print(f"\n📖 Official rating: CR {book['effective_cr']:g}")
    if appraisal["verdict"] == "trivial":
        print(f"⚡ TRUE LETHALITY LEVEL: ≤ 1 — even a level-1 party wins "
              f"{appraisal['p_level_1']:.0%} of the time.")
    elif appraisal["verdict"] == "beyond_deadly":
        print(f"⚡ TRUE LETHALITY LEVEL: > 20 — a level-20 party only wins "
              f"{appraisal['p_level_20']:.0%} of the time. TPK machine.")
    else:
        lethality_level = appraisal["level"]
        print(f"⚡ TRUE LETHALITY LEVEL: {lethality_level:g} "
              f"({appraisal['p_at_level']:.1%} win at this level)")
        diff = lethality_level - book['effective_cr']
        if abs(diff) >= 2:
            print(f"   (Massive discrepancy: {abs(diff):.1f} levels "
                  f"{'harder' if diff > 0 else 'easier'} than the rating suggests)")
        elif abs(diff) >= 1:
            print(f"   (Off by {abs(diff):.1f} levels)")

    print("\nSimulating 400 hypothetical parties...")
    df_sim = simulate_party_grid(pipeline, roster)
    best = fair_fight_matches(df_sim, top_n=5)

    print_composition_legend()

    comp_desc = {c["name"]: c["desc"] for c in PARTY_COMPOSITIONS}

    print("\n" + "=" * 56)
    print(f"⚔️  IDEAL PARTY MATCHES (target {TARGET_WIN_RATE:.0%} win rate)")
    print("=" * 56)
    for _, r in best.iterrows():
        # Party-size adjusted DMG multiplier (p.83), computed per row.
        adjusted_xp = fields["total_monster_xp"] * encounter_xp_multiplier(
            n_total, r["party_size"]
        )
        tier = _classify_xp_tier(adjusted_xp, r["avg_party_level"], r["party_size"])
        print(f"Level {int(r['avg_party_level'])} party "
              f"({int(r['party_size'])} players) — {r['comp_name']}")
        print(f"  Predicted win chance: {r['win_prob']:.1%}"
              f" | DMG tier: {TIER_NAMES.get(tier, '?')}")
        if fields["monster_is_legendary"] and not r["has_healer"]:
            print("  ⚠️ LEGENDARY ATTRITION: no healer vs a legendary!")
        if fields["monster_has_mobility"] and not (r["has_arcane"] or r["has_martial_dps"]):
            print("  ⚠️ MOBILITY THREAT: no ranged answer to a flyer/swimmer!")
        if fields["monster_has_magic_res"] and r["has_arcane"]:
            print("  ⚠️ MAGIC RESISTANCE: spells will struggle to land!")
        if fields["monster_has_physical_res"] and r["has_martial_dps"]:
            print("  ⚠️ PHYSICAL RESISTANCE: martial weapon damage is halved!")
        if fields["max_monster_burst"] >= 50:
            print(f"  💥 NOVA THREAT: scariest single action deals "
                  f"~{fields['max_monster_burst']:.0f} damage!")
        print("-" * 56)


if __name__ == "__main__":
    main()
