"""Inference core shared by the web app and the CLI.

Everything that turns a monster (or roster) + party config into predictions
lives here: building the raw input rows, the binary search for the "fair"
party level, the party sweep, the win curves, and the domain guards.

app.py and fair_fight_finder.py must import from here and never duplicate
this logic. We learned that the hard way: the two used to carry their own
copies and the CLI silently lost the num_monsters field at some point.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from initial_learn import CR_PREDICTOR_FEATURES, RAW_INPUT_COLUMNS, parse_cr_value
from monster_offense import (
    cr_to_xp,
    encounter_xp_multiplier,
    extract_legendary_mobility,
    extract_official_traits,
    offense_from_cr,
    xp_to_cr,
)

TARGET_WIN_RATE = 0.65
FAIR_WIN_BAND = (0.55, 0.75)

# The four party roles the model understands, and which PHB classes count.
ROLE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "Healer": {
        "classes": "Cleric, Druid, Bard",
        "meaning": "Restores HP and removes conditions mid-fight. Turns "
        "attrition wars (legendary monsters, regeneration) from "
        "losses into wins.",
    },
    "Tank": {
        "classes": "Barbarian, Fighter, Paladin",
        "meaning": "High AC/HP frontline that soaks attacks so fragile "
        "members don't. Counters pack tactics and melee swarms.",
    },
    "Arcane": {
        "classes": "Wizard, Sorcerer, Warlock",
        "meaning": "Ranged magical damage and battlefield control. The "
        "reliable answer to flying/swimming monsters — but loses "
        "value against magic resistance.",
    },
    "Martial DPS": {
        "classes": "Rogue, Monk, Ranger",
        "meaning": "Sustained weapon damage (often ranged). Wins damage "
        "races — but halved by nonmagical physical resistance.",
    },
}

PARTY_COMPOSITIONS: List[Dict[str, object]] = [
    {
        "name": "Balanced",
        "healer": 1,
        "tank": 1,
        "arcane": 1,
        "dps": 1,
        "desc": "One of each role — healer, tank, arcane caster, martial DPS. "
        "The textbook party with an answer to everything; the "
        "reference point for the True Lethality Level.",
    },
    {
        "name": "Glass Cannons",
        "healer": 0,
        "tank": 0,
        "arcane": 1,
        "dps": 1,
        "desc": "All offense, no healer, no tank. Kills fast but folds if "
        "the monster survives long enough to swing back — high "
        "variance against bursty or legendary monsters.",
    },
    {
        "name": "The Wall",
        "healer": 1,
        "tank": 1,
        "arcane": 0,
        "dps": 0,
        "desc": "Tanks and healers only. Nearly unkillable but slow to end "
        "fights — struggles against flyers (no ranged answer) and "
        "loses long attrition races to regenerating monsters.",
    },
    {
        "name": "Melee Rush",
        "healer": 0,
        "tank": 1,
        "arcane": 0,
        "dps": 1,
        "desc": "Frontline bruisers, no casters, no healer. Great against "
        "grounded brutes, helpless against flying/swimming monsters "
        "and punished hard by physical resistance.",
    },
    {
        "name": "Full Caster",
        "healer": 1,
        "tank": 0,
        "arcane": 1,
        "dps": 0,
        "desc": "Casters and support, no frontline. Excellent control and "
        "range, but pack tactics and melee swarms reach the squishy "
        "backline unopposed; magic resistance blunts the whole plan.",
    },
]


@dataclass
class MonsterProfile:
    """Everything the model needs to know about one monster statblock."""

    cr: float
    hp: float
    ac: float
    size_num: float = 3.0
    stat_sum: float = 150.0
    is_legendary: int = 0
    has_mobility: int = 0
    physical_res: int = 0
    cc_immune: int = 0
    magic_res: int = 0
    pack_tactics: int = 0
    spellcasting: int = 0
    regeneration: int = 0
    # Offensive potency; None -> imputed from the DMG p.274 table for the CR.
    atk_bonus: Optional[float] = None
    dpr: Optional[float] = None
    save_dc: Optional[float] = None
    burst: Optional[float] = None
    name: str = "Custom Monster"

    def __post_init__(self) -> None:
        table_atk, table_dpr, table_dc = offense_from_cr(self.cr)
        if self.atk_bonus is None or pd.isna(self.atk_bonus):
            self.atk_bonus = table_atk
        if self.dpr is None or pd.isna(self.dpr):
            self.dpr = table_dpr
        if self.save_dc is None or pd.isna(self.save_dc):
            self.save_dc = table_dc
        if self.burst is None or pd.isna(self.burst):
            # Without a statblock nova, assume burst == sustained DPR.
            self.burst = self.dpr


# A mixed encounter: [(monster, count), ...].  Every simulation function
# accepts either a single MonsterProfile (with num_monsters) or a Roster.
Roster = Sequence[Tuple[MonsterProfile, int]]
MonsterInput = Union[MonsterProfile, Roster]


def normalize_roster(
    monster: MonsterInput, num_monsters: int = 1
) -> List[Tuple[MonsterProfile, int]]:
    """Coerce a MonsterProfile or roster into [(profile, count), ...]."""
    if isinstance(monster, MonsterProfile):
        return [(monster, max(1, int(num_monsters)))]
    return [(m, max(1, int(c))) for m, c in monster]


# Backward-compatible private alias.
_normalize_roster = normalize_roster


def roster_monster_fields(roster: Roster) -> Dict[str, float]:
    """Encounter-level monster aggregates for a mixed roster.

    Mirrors the training-time aggregation in ``parse_fireball.py`` exactly:
    count-weighted means for continuous stats, max for binary flags and
    apex-threat numbers, sums for the pooled damage race.
    """
    pairs = [(m, max(1, int(c))) for m, c in roster]
    total = sum(c for _, c in pairs)

    def wmean(get) -> float:
        return sum(get(m) * c for m, c in pairs) / total

    return {
        "num_monsters": total,
        "num_monsters_total": total,
        "avg_monster_cr": wmean(lambda m: float(m.cr)),
        "max_monster_cr": max(float(m.cr) for m, _ in pairs),
        "avg_monster_hp": wmean(lambda m: float(m.hp)),
        "total_monster_hp": sum(float(m.hp) * c for m, c in pairs),
        "avg_monster_ac": wmean(lambda m: float(m.ac)),
        "avg_monster_size_num": wmean(lambda m: float(m.size_num)),
        "avg_monster_stat_sum": wmean(lambda m: float(m.stat_sum)),
        "monster_is_legendary": max(int(m.is_legendary) for m, _ in pairs),
        "monster_has_mobility": max(int(m.has_mobility) for m, _ in pairs),
        "monster_has_physical_res": max(int(m.physical_res) for m, _ in pairs),
        "monster_immune_to_cc": max(int(m.cc_immune) for m, _ in pairs),
        "monster_has_magic_res": max(int(m.magic_res) for m, _ in pairs),
        "monster_has_pack_tactics": max(int(m.pack_tactics) for m, _ in pairs),
        "monster_has_spellcasting": max(int(m.spellcasting) for m, _ in pairs),
        "monster_has_regeneration": max(int(m.regeneration) for m, _ in pairs),
        "avg_monster_dpr": wmean(lambda m: float(m.dpr)),
        "total_monster_dpr": sum(float(m.dpr) * c for m, c in pairs),
        "max_monster_atk_bonus": max(float(m.atk_bonus) for m, _ in pairs),
        "max_monster_save_dc": max(float(m.save_dc) for m, _ in pairs),
        "max_monster_burst": max(float(m.burst) for m, _ in pairs),
        "total_monster_xp": sum(cr_to_xp(m.cr) * c for m, c in pairs),
    }


def encounter_row(
    monster: MonsterInput,
    *,
    avg_party_level: float,
    party_size: int,
    num_monsters: int = 1,
    has_healer: int,
    has_tank: int,
    has_arcane: int,
    has_martial_dps: int,
) -> Dict[str, float]:
    """A complete raw-input row for the pipeline (all RAW_INPUT_COLUMNS).

    ``monster`` is a single MonsterProfile (repeated ``num_monsters`` times)
    or a Roster of (profile, count) pairs; ``num_monsters`` is ignored for
    rosters, whose counts are per-pair.
    """
    row = roster_monster_fields(_normalize_roster(monster, num_monsters))
    row.update(
        {
            "avg_party_level": float(avg_party_level),
            "party_size": int(party_size),
            "has_healer": int(has_healer),
            "has_tank": int(has_tank),
            "has_arcane": int(has_arcane),
            "has_martial_dps": int(has_martial_dps),
        }
    )
    return row


def predict_win_probability(pipe, rows: List[Dict[str, float]]) -> np.ndarray:
    df = pd.DataFrame(rows)
    return pipe.predict_proba(df[list(RAW_INPUT_COLUMNS)])[:, 1]


# --- survival cap ---
# The logs lie exactly where it matters: fights that the damage math says
# are hopeless still show up as "won" 84.5% of the time, because DMs
# fudge, players run, reinforcements arrive. A model trained on that
# can't answer "what if we fight 19 liches to the death", so P(win) gets
# capped by deathmatch physics. Two caps, combined with min():
#
# 1. Race cap: sigmoid(A * rounds_party_survives - C * ln(rounds_to_kill)
#    + B). The survival term catches fast wipes, the kill term catches
#    slow attrition losses, which my first two guards missed completely:
#    a level-5 party "survives" a Lich for 4+ estimated rounds, so a
#    survival-only cap sat at 0.95 while Battlecast had the party at
#    0.000 over 2,000 deathmatches - they can't chew through 135
#    legendary HP behind AC 17 before the spell rotation lands. That
#    blind spot is how one Lich got appraised "fair at level 3.25".
#    A, C, B are a binomial-weighted logistic fit on the 180-cell
#    Battlecast guard grid rebuilt through the same bestiary profiles the
#    app serves (battlecast_bridge/analyze.py). History: v1 hand-tuned
#    (2.197, -4.394), v2 survival-only fit (1.6302, -3.9771), v3 this.
#
# 2. Lattice cap: the TTK features can't tell a Lich from a bag of hit
#    points (AoE rotations aren't priced by the damage math), so where we
#    HAVE simulated truth we serve it directly - trilinear interpolation
#    over the guard grid (CR {2,5,10,15,21} x count {1,2,4,8,12,19} x
#    level {1,5,9,13,17,20}), made monotone at build time. Below CR 2 it
#    abstains (weak monsters were never the bug); beyond the grid it
#    clamps to the nearest edge, and the race cap covers what the lattice
#    can't see (a 10,000-HP homebrew wall stays beyond deadly).
#
# Coefficient signs and the lattice massage are both constrained, so the
# combined cap stays monotone up in party level and monotone down in
# monster count - the axioms the behavior suite pins.
_GUARD_A = 0.1924
_GUARD_C = 2.0856
_GUARD_B = 4.1217

_LATTICE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "battlecast_bridge", "guard_lattice.json"
)
_LATTICE: dict | None = None


def _load_lattice() -> dict | None:
    global _LATTICE
    if _LATTICE is None and os.path.exists(_LATTICE_PATH):
        with open(_LATTICE_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
        _LATTICE = {
            "crs": np.array(raw["crs"], dtype=float),
            "counts": np.array(raw["counts"], dtype=float),
            "levels": np.array(raw["levels"], dtype=float),
            "p": np.array(raw["p_win"], dtype=float),
        }
    return _LATTICE


def _interp_1d(grid: np.ndarray, x: float) -> Tuple[int, int, float]:
    """Clamped linear-interpolation stencil: (lo, hi, weight of hi)."""
    x = float(np.clip(x, grid[0], grid[-1]))
    hi = int(np.searchsorted(grid, x))
    if hi == 0:
        return 0, 0, 0.0
    lo = hi - 1
    if hi == len(grid):
        return lo, lo, 0.0
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    return lo, hi, w


def _lattice_cap(rows: List[Dict[str, float]]) -> np.ndarray:
    lat = _load_lattice()
    out = np.ones(len(rows))
    if lat is None:
        return out
    for i, r in enumerate(rows):
        cr = float(r["avg_monster_cr"])
        if cr < lat["crs"][0]:
            continue  # below the grid the lattice has nothing to say
        c0, c1, wc = _interp_1d(lat["crs"], cr)
        n0, n1, wn = _interp_1d(lat["counts"], float(r["num_monsters_total"]))
        l0, l1, wl = _interp_1d(lat["levels"], float(r["avg_party_level"]))
        p = lat["p"]
        out[i] = (
            (1 - wc) * ((1 - wn) * ((1 - wl) * p[c0, n0, l0] + wl * p[c0, n0, l1])
                        + wn * ((1 - wl) * p[c0, n1, l0] + wl * p[c0, n1, l1]))
            + wc * ((1 - wn) * ((1 - wl) * p[c1, n0, l0] + wl * p[c1, n0, l1])
                    + wn * ((1 - wl) * p[c1, n1, l0] + wl * p[c1, n1, l1]))
        )
    return out


def _survival_cap(rows: List[Dict[str, float]]) -> np.ndarray:
    """Cap on P(win) from deathmatch physics (see the block comment above)."""
    from initial_learn import DnDFeatureEngineer, FEATURE_COLUMNS

    cols = list(FEATURE_COLUMNS) + ["rounds_to_kill_monster_raw"]
    feats = DnDFeatureEngineer(1.5, cols).transform(pd.DataFrame(rows))
    survive = feats["rounds_to_kill_party"].to_numpy(dtype=float)
    # Unclipped on purpose: at the model feature's clip (50 rounds) a
    # 10,000-HP wall and a merely tough boss look identical, and the wall
    # must stay hopeless.
    kill = feats["rounds_to_kill_monster_raw"].to_numpy(dtype=float)
    z = _GUARD_A * survive - _GUARD_C * np.log(np.maximum(kill, 1e-6)) + _GUARD_B
    race = 1.0 / (1.0 + np.exp(-z))
    return np.minimum(race, _lattice_cap(rows))


def predict_win_for_parties(
    pipe,
    monster: MonsterInput,
    party_configs: List[Dict[str, float]],
    num_monsters: int = 1,
) -> np.ndarray:
    """P(win) per party config, with the two domain guards applied.

    1. Dominance: on mixed rosters the weighted averages dilute (lich + six
       goblins has lower avg CR than the lich alone, and the model would
       call the bigger fight easier). So score every homogeneous sub-roster
       too and take the elementwise min - more monsters can't help you.
    2. Survival cap: without it the model stays near the 83% ceiling even
       for hopeless fights (the 19-liches-at-level-8 bug).
    """
    roster = _normalize_roster(monster, num_monsters)
    variants: List[List[Tuple[MonsterProfile, int]]] = [list(roster)]
    if len(roster) > 1:
        variants += [[(m, c)] for m, c in roster]

    rows = [
        encounter_row(variant, **cfg) for variant in variants for cfg in party_configs
    ]
    probs = np.minimum(predict_win_probability(pipe, rows), _survival_cap(rows))
    return probs.reshape(len(variants), len(party_configs)).min(axis=0)


def _party_config(
    level: float, party_size: int, comp: Dict[str, object]
) -> Dict[str, float]:
    return {
        "avg_party_level": float(level),
        "party_size": int(party_size),
        "has_healer": int(comp["healer"]),
        "has_tank": int(comp["tank"]),
        "has_arcane": int(comp["arcane"]),
        "has_martial_dps": int(comp["dps"]),
    }


_BALANCED = {"healer": 1, "tank": 1, "arcane": 1, "dps": 1}


def lethality_appraisal(
    pipe,
    monster: MonsterInput,
    num_monsters: int = 1,
    *,
    target: float = TARGET_WIN_RATE,
    party_size: int = 4,
    iterations: int = 18,
) -> Dict[str, float]:
    """Find the party level where a balanced party hits the target win rate.

    The model is only defined on levels 1-20 (inputs are clipped inside the
    transformer), so the search stays in that band and boundary cases get an
    explicit verdict instead of a fabricated fractional level:

    * ``trivial``       — P(win) >= target already at level 1,
    * ``beyond_deadly`` — P(win) < target even at level 20,
    * ``ok``            — the target is crossed inside [1, 20].
    """
    p_low, p_high = predict_win_for_parties(
        pipe,
        monster,
        [
            _party_config(1.0, party_size, _BALANCED),
            _party_config(20.0, party_size, _BALANCED),
        ],
        num_monsters,
    )
    base = {"p_level_1": float(p_low), "p_level_20": float(p_high)}
    if p_low >= target:
        return {**base, "level": 1.0, "verdict": "trivial", "p_at_level": float(p_low)}
    if p_high < target:
        return {
            **base,
            "level": 20.0,
            "verdict": "beyond_deadly",
            "p_at_level": float(p_high),
        }

    low, high = 1.0, 20.0
    for _ in range(iterations):
        guess = (low + high) / 2.0
        win_prob = predict_win_for_parties(
            pipe, monster, [_party_config(guess, party_size, _BALANCED)], num_monsters
        )[0]
        if win_prob < target:
            low = guess  # party too weak -> need higher level
        else:
            high = guess
    level = round(((low + high) / 2.0) * 4) / 4
    # Win probability AT the appraised level: distinguishes encounters that
    # share a level but differ in risk (1 Lich vs 2 Liches both near L5).
    p_at_level = float(
        predict_win_for_parties(
            pipe, monster, [_party_config(level, party_size, _BALANCED)], num_monsters
        )[0]
    )
    return {**base, "level": float(level), "verdict": "ok", "p_at_level": p_at_level}


def find_true_lethality_level(
    pipe,
    monster: MonsterInput,
    num_monsters: int = 1,
    *,
    target: float = TARGET_WIN_RATE,
    party_size: int = 4,
    iterations: int = 18,
) -> float:
    """Backward-compatible wrapper returning just the level."""
    return lethality_appraisal(
        pipe,
        monster,
        num_monsters,
        target=target,
        party_size=party_size,
        iterations=iterations,
    )["level"]


def win_curve(
    pipe,
    monster: MonsterInput,
    num_monsters: int = 1,
    *,
    party_size: int = 4,
    compositions: Optional[List[Dict[str, object]]] = None,
) -> pd.DataFrame:
    """P(win) for levels 1..20, one curve per composition — for plotting."""
    comps = compositions or PARTY_COMPOSITIONS
    configs, meta = [], []
    for comp in comps:
        for level in range(1, 21):
            configs.append(_party_config(level, party_size, comp))
            meta.append({"comp_name": comp["name"], "avg_party_level": level})
    out = pd.DataFrame(meta)
    out["win_prob"] = predict_win_for_parties(pipe, monster, configs, num_monsters)
    return out


def simulate_party_grid(
    pipe,
    monster: MonsterInput,
    num_monsters: int = 1,
) -> pd.DataFrame:
    """Sweep party sizes 3-6 x levels 1-20 x 5 compositions (400 parties)."""
    configs, meta = [], []
    for party_size in (3, 4, 5, 6):
        for level in range(1, 21):
            for comp in PARTY_COMPOSITIONS:
                configs.append(_party_config(level, party_size, comp))
                meta.append(
                    {
                        "avg_party_level": level,
                        "party_size": party_size,
                        "comp_name": comp["name"],
                        "has_healer": comp["healer"],
                        "has_tank": comp["tank"],
                        "has_arcane": comp["arcane"],
                        "has_martial_dps": comp["dps"],
                    }
                )
    out = pd.DataFrame(meta)
    out["win_prob"] = predict_win_for_parties(pipe, monster, configs, num_monsters)
    return out


def official_encounter_estimate(
    monster: MonsterInput,
    num_monsters: int = 1,
    *,
    party_size: int = 4,
) -> Dict[str, float]:
    """The encounter's difficulty by the book — DMG p.82, computed honestly.

    Previously the app compared the model's True Lethality Level against the
    single monster's printed CR, so 6x CR-1 monsters still displayed "CR 1"
    on the book side.  The DMG procedure is: sum every monster's XP, apply
    the encounter multiplier for the monster count (with the party-size
    step adjustment), and judge difficulty by the *adjusted* total.  We
    express that adjusted XP back in CR units — the CR of the single
    monster the DMG considers equally difficult — so the book column and
    the model's level are an apples-to-apples pair.

    Returns total_xp, multiplier, adjusted_xp, and effective_cr (snapped to
    quarter steps; equals the printed CR when num_monsters == 1).
    """
    roster = _normalize_roster(monster, num_monsters)
    fields = roster_monster_fields(roster)
    total_xp = float(fields["total_monster_xp"])
    n_total = float(fields["num_monsters_total"])
    multiplier = encounter_xp_multiplier(n_total, party_size)
    adjusted_xp = total_xp * multiplier
    effective_cr = max(0.25, round(xp_to_cr(adjusted_xp) * 4) / 4)
    return {
        "total_xp": total_xp,
        "num_monsters": n_total,
        "multiplier": multiplier,
        "adjusted_xp": adjusted_xp,
        "effective_cr": effective_cr,
    }


def fair_fight_matches(
    df_sim: pd.DataFrame, top_n: int = 10, target: float = TARGET_WIN_RATE
) -> pd.DataFrame:
    """Parties inside the fair-fight band (target ±0.10), closest first."""
    fair = df_sim[(df_sim["win_prob"] - target).abs() <= 0.10].copy()
    if fair.empty:
        fair = df_sim.copy()
        top_n = min(top_n, 5)
    fair["distance_from_ideal"] = (fair["win_prob"] - target).abs()
    return fair.sort_values("distance_from_ideal").head(top_n)


def load_cr_predictor(
    json_path: str = "cr_predictor_model.json",
    pkl_path: str = "cr_predictor_model.pkl",
):
    """Load the WotC-CR predictor, preferring the version-proof native JSON.

    XGBoost pickles break across xgboost versions, and the sklearn wrapper's
    ``load_model`` breaks on old-xgboost + new-sklearn pairings (sklearn 1.6
    removed ``_estimator_type``, raising ``TypeError: _estimator_type
    undefined``).  The low-level ``xgb.Booster`` depends on neither, so the
    JSON is loaded through it directly.  The pickle remains as a fallback
    for repos that only carry the .pkl.  Returns ``None`` when neither
    artifact exists — callers already handle that.
    """
    import os

    if os.path.exists(json_path):
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(json_path)
        return booster
    if os.path.exists(pkl_path):
        import joblib

        return joblib.load(pkl_path)
    return None


def predict_wotc_cr(cr_predictor, features: Dict[str, float]) -> float:
    """Predict the CR WotC would assign, snapped to quarter steps.

    Builds the frame by reindexing on ``CR_PREDICTOR_FEATURES`` so column
    names and order always match training exactly, regardless of the
    caller's dict ordering.  Accepts either a raw ``xgb.Booster`` (the
    version-proof JSON path) or a fitted sklearn-style regressor (the
    legacy pickle path).
    """
    import xgboost as xgb

    frame = (
        pd.DataFrame([features])
        .reindex(columns=list(CR_PREDICTOR_FEATURES))
        .astype(float)
    )
    if isinstance(cr_predictor, xgb.Booster):
        # Build from a raw numpy array with EXPLICIT feature names: some
        # pandas/xgboost version pairings fail to detect the DataFrame and
        # silently drop column names, making predict() raise "data did not
        # contain feature names".  Explicit names sidestep detection.
        dmat = xgb.DMatrix(
            frame.to_numpy(dtype=float),
            feature_names=list(CR_PREDICTOR_FEATURES),
        )
        raw = cr_predictor.predict(dmat)[0]
    else:
        raw = cr_predictor.predict(frame)[0]
    return max(0.25, round(float(raw) * 4) / 4)


def load_monster_database(
    csv_path: str = "Monster Spreadsheet (D&D5e) - Official Stats.csv",
    offense_csv: str = "monster_offense_stats.csv",
) -> pd.DataFrame:
    """Official bestiary + parsed offensive stats, ready for profile lookup."""
    db = pd.read_csv(csv_path)
    db["clean_name"] = (
        db["Name"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace("-", " ", regex=False)
    )
    db["cr_num"] = db["CR"].apply(parse_cr_value)

    # Canonical trait extraction — single source of truth in monster_offense.
    db[["is_legendary", "has_mobility"]] = extract_legendary_mobility(
        db["Additional"], db["Speeds"]
    )
    db[
        [
            "physical_res",
            "cc_immune",
            "magic_res",
            "pack_tactics",
            "spellcasting",
            "regeneration",
        ]
    ] = extract_official_traits(db["WRI"], db["Additional"])[
        [
            "physical_res",
            "cc_immune",
            "magic_res",
            "pack_tactics",
            "spellcasting",
            "regeneration",
        ]
    ]

    for col in ["STR", "DEX", "CON", "INT", "WIS", "CHA"]:
        db[col] = pd.to_numeric(db[col], errors="coerce")
    db["stat_sum"] = db[["STR", "DEX", "CON", "INT", "WIS", "CHA"]].sum(axis=1)

    size_map = {
        "tiny": 1,
        "small": 2,
        "medium": 3,
        "large": 4,
        "huge": 5,
        "gargantuan": 6,
    }
    db["size_num"] = db["Size"].str.strip().str.lower().map(size_map)

    try:
        offense = pd.read_csv(offense_csv)
        db = db.merge(offense, on="clean_name", how="left")
    except FileNotFoundError:
        db["atk_bonus"] = np.nan
        db["dpr"] = np.nan
        db["save_dc"] = np.nan
        db["offense_source"] = "dmg_cr_table"
    return db


def profile_from_db_row(row: pd.Series) -> MonsterProfile:
    return MonsterProfile(
        cr=float(row["cr_num"]),
        hp=float(row["HP"]),
        ac=float(row["AC"]),
        size_num=float(row["size_num"]) if pd.notna(row["size_num"]) else 3.0,
        stat_sum=float(row["stat_sum"]) if pd.notna(row["stat_sum"]) else 150.0,
        is_legendary=int(row["is_legendary"]),
        has_mobility=int(row["has_mobility"]),
        physical_res=int(row["physical_res"]),
        cc_immune=int(row["cc_immune"]),
        magic_res=int(row["magic_res"]),
        pack_tactics=int(row["pack_tactics"]),
        spellcasting=int(row["spellcasting"]),
        regeneration=int(row["regeneration"]),
        atk_bonus=row.get("atk_bonus"),
        dpr=row.get("dpr"),
        save_dc=row.get("save_dc"),
        burst=row.get("burst_dmg"),
        name=str(row["Name"]),
    )
