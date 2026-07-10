"""Monster offensive statistics engine.

Solves the "Attack Potency" limitation: the official stats CSV carries no
Damage Per Round (DPR), attack bonus, or spell save DC, forcing the model to
lean on WotC CR as a proxy for offense.  This module derives real offensive
stats from two local, license-safe sources:

1. **SRD statblock parsing** — ``srd_5e_monsters.json`` contains the full
   Actions HTML for 327 SRD monsters.  We regex-extract attack bonuses
   (``+9 to hit``), printed average damage (``12 (2d6 + 5)``), Multiattack
   routine counts (``makes three tentacle attacks``), and save DCs
   (``DC 14``), then estimate DPR as ``n_attacks x mean(per-attack damage)``.

2. **DMG p.274 fallback** — for monsters outside the SRD, we impute the
   expected offensive profile for their CR from the official "Monster
   Statistics by Challenge Rating" table (midpoint of the DPR band).

Parsed DPR is clamped to [0.25x, 3x] of the DMG expectation for the
monster's CR so one regex mishap can't inject a CR-1 monster with DPR 400.

Public API
----------
``build_offense_table(...)`` -> DataFrame indexed by clean_name with columns
    ``atk_bonus, dpr, save_dc, offense_source``
``offense_from_cr(cr)`` -> (atk_bonus, dpr, save_dc) tuple for any CR
``attach_offense(df, cr_col, name_col)`` -> merge offense columns onto any
    monster frame, falling back to the CR table row-by-row.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ── DMG p.274: Monster Statistics by Challenge Rating ─────────────────────
# CR -> (attack_bonus, dpr_low, dpr_high, save_dc).  DPR feature uses the
# band midpoint.  This is the official WotC design target for offense at
# each CR, which makes it the principled imputation when a statblock is
# unavailable.
DMG_OFFENSE_BY_CR: Dict[float, Tuple[int, int, int, int]] = {
    0.0: (3, 0, 1, 13),
    0.125: (3, 2, 3, 13),
    0.25: (3, 4, 5, 13),
    0.5: (3, 6, 8, 13),
    1.0: (3, 9, 14, 13),
    2.0: (3, 15, 20, 13),
    3.0: (4, 21, 26, 13),
    4.0: (5, 27, 32, 14),
    5.0: (6, 33, 38, 15),
    6.0: (6, 39, 44, 15),
    7.0: (6, 45, 50, 15),
    8.0: (7, 51, 56, 16),
    9.0: (7, 57, 62, 16),
    10.0: (7, 63, 68, 16),
    11.0: (8, 69, 74, 17),
    12.0: (8, 75, 80, 17),
    13.0: (8, 81, 86, 18),
    14.0: (8, 87, 92, 18),
    15.0: (8, 93, 98, 18),
    16.0: (9, 99, 104, 18),
    17.0: (10, 105, 110, 19),
    18.0: (10, 111, 116, 19),
    19.0: (10, 117, 122, 19),
    20.0: (10, 123, 140, 19),
    21.0: (11, 141, 158, 20),
    22.0: (11, 159, 176, 20),
    23.0: (11, 177, 194, 20),
    24.0: (12, 195, 212, 21),
    25.0: (12, 213, 230, 21),
    26.0: (12, 231, 248, 21),
    27.0: (13, 249, 266, 22),
    28.0: (13, 267, 284, 22),
    29.0: (13, 285, 302, 22),
    30.0: (14, 303, 320, 23),
}

_CR_KEYS = sorted(DMG_OFFENSE_BY_CR)

# ── DMG p.274: CR -> XP reward (canonical copy for the whole project) ──────
CR_TO_XP: Dict[float, int] = {
    0: 10,
    0.125: 25,
    0.25: 50,
    0.5: 100,
    1: 200,
    2: 450,
    3: 700,
    4: 1100,
    5: 1800,
    6: 2300,
    7: 2900,
    8: 3900,
    9: 5000,
    10: 5900,
    11: 7200,
    12: 8400,
    13: 10000,
    14: 11500,
    15: 13000,
    16: 15000,
    17: 18000,
    18: 20000,
    19: 22000,
    20: 25000,
    21: 33000,
    22: 41000,
    23: 50000,
    24: 62000,
    25: 75000,
    26: 90000,
    27: 105000,
    28: 120000,
    29: 135000,
    30: 155000,
}

_XP_KEYS = sorted(CR_TO_XP)


def cr_to_xp(cr: Optional[float]) -> float:
    """XP reward for a CR, linearly interpolated between table rows."""
    if cr is None or (isinstance(cr, float) and np.isnan(cr)):
        return 0.0
    cr = float(cr)
    if cr <= _XP_KEYS[0]:
        return float(CR_TO_XP[_XP_KEYS[0]])
    if cr >= _XP_KEYS[-1]:
        return float(CR_TO_XP[_XP_KEYS[-1]])
    return float(np.interp(cr, _XP_KEYS, [CR_TO_XP[k] for k in _XP_KEYS]))


def xp_to_cr(xp: float) -> float:
    """Inverse of ``cr_to_xp``: the CR whose XP reward matches ``xp``.

    Used by the by-the-book encounter estimate: an encounter with adjusted
    XP ``A`` is as difficult (per DMG p.82 math) as a single monster of CR
    ``xp_to_cr(A)``.  Linear interpolation between table rows; clamped to
    the 0-30 CR band.  Exact round-trip on table values, so a single
    monster's book estimate is always its printed CR.
    """
    if xp is None or (isinstance(xp, float) and np.isnan(xp)) or xp <= 0:
        return 0.0
    xp_values = [CR_TO_XP[k] for k in _XP_KEYS]
    if xp <= xp_values[0]:
        return float(_XP_KEYS[0])
    if xp >= xp_values[-1]:
        return float(_XP_KEYS[-1])
    return float(np.interp(xp, xp_values, _XP_KEYS))


# The DMG multiplier ladder: party-size adjustment shifts one step up/down.
_XP_MULT_LADDER: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def encounter_xp_multiplier(
    num_monsters: float, party_size: Optional[float] = None
) -> float:
    """DMG p.82 encounter multiplier for monster count (action economy).

    When ``party_size`` is given, the official party-size adjustment is
    applied: parties with fewer than 3 characters use the next multiplier
    up the ladder, parties with 6 or more use the next one down (DMG p.83).
    """
    n = max(1.0, float(num_monsters))
    if n <= 1:
        idx = 1
    elif n <= 2:
        idx = 2
    elif n <= 6:
        idx = 3
    elif n <= 10:
        idx = 4
    elif n <= 14:
        idx = 5
    else:
        idx = 6
    if party_size is not None and not pd.isna(party_size):
        if float(party_size) < 3:
            idx += 1
        elif float(party_size) >= 6:
            idx -= 1
    return _XP_MULT_LADDER[max(0, min(idx, len(_XP_MULT_LADDER) - 1))]


# ── Canonical deep-trait extraction (single source of truth) ────────────────
# These six flags were previously extracted in THREE places with slightly
# different regexes (parse_fireball.py, lethality_engine.py, initial_learn.py)
# — meaning the CR predictor trained on different trait definitions than the
# app used at inference.  Every consumer now goes through this table.
OFFICIAL_TRAIT_REGEX: Dict[str, Tuple[str, str]] = {
    # flag_name: (source column, regex)
    "physical_res": ("wri", r"nonmagicalres|nonmagicalimmu"),
    "cc_immune": ("wri", r"charmedimmu|stunnedimmu|paralyzedimmu|frightenedimmu"),
    "magic_res": ("additional", r"magic resist"),
    "pack_tactics": ("additional", r"pack tactics"),
    "spellcasting": ("additional", r"spellcast|innate spell"),
    "regeneration": ("additional", r"regenerat"),
}


def extract_official_traits(wri: pd.Series, additional: pd.Series) -> pd.DataFrame:
    """Six binary deep-trait flags from the official-stats WRI/Additional text.

    Accepts raw (possibly NaN) Series; lowercasing and fill happen here so
    every caller gets byte-identical behavior.
    """
    sources = {
        "wri": wri.fillna("").astype(str).str.lower(),
        "additional": additional.fillna("").astype(str).str.lower(),
    }
    return pd.DataFrame(
        {
            flag: sources[src].str.contains(pattern, regex=True, na=False).astype(int)
            for flag, (src, pattern) in OFFICIAL_TRAIT_REGEX.items()
        }
    )


def extract_legendary_mobility(
    additional: pd.Series, speeds: pd.Series
) -> pd.DataFrame:
    """Canonical is_legendary / has_mobility flags from official-stats text."""
    add = additional.fillna("").astype(str).str.lower()
    spd = speeds.fillna("").astype(str).str.lower()
    return pd.DataFrame(
        {
            "is_legendary": add.str.contains("legendary", na=False).astype(int),
            "has_mobility": spd.str.contains("fly|swim", regex=True, na=False).astype(
                int
            ),
        }
    )


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_TAG_RE = re.compile(r"<[^>]+>")
_ATK_BONUS_RE = re.compile(r"([+-]\s?\d+)\s+to\s+hit", re.IGNORECASE)
# Printed average damage: "12 (2d6 + 5) bludgeoning damage"
_AVG_DMG_RE = re.compile(
    r"(\d+)\s*\(\d+d\d+(?:\s*[+-]\s*\d+)?\)\s*[a-z ]*damage", re.IGNORECASE
)
# Flat damage with no dice expression: "1 piercing damage"
_FLAT_DMG_RE = re.compile(
    r"\b(\d+)\s+(?:acid|bludgeoning|cold|fire|force|lightning|necrotic|piercing|poison|psychic|radiant|slashing|thunder)\s+damage",
    re.IGNORECASE,
)
_SAVE_DC_RE = re.compile(r"\bDC\s+(\d+)", re.IGNORECASE)
_MULTIATTACK_MAKES_RE = re.compile(
    r"makes\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+)", re.IGNORECASE
)
# Statblock action headers: "<em><strong>Fire Breath (Recharge 5-6).</strong></em>"
_ACTION_HEADER_RE = re.compile(
    r"<(?:em|strong)>\s*<(?:em|strong)>([^<]+?)</(?:em|strong)>\s*</(?:em|strong)>",
    re.IGNORECASE,
)
_SAVE_CONTEXT_RE = re.compile(r"saving\s+throw|succeed\s+on\s+a\s+DC", re.IGNORECASE)

# ── PHB damage-spell table: spell name -> average damage on a hit/failed
# save at the spell's base casting level.  Used to scan Spellcasting /
# Innate Spellcasting trait lists, which name spells without printing their
# damage — this is how a Lich's Power Word Kill enters the burst feature.
SPELL_DAMAGE: Dict[str, float] = {
    "meteor swarm": 140.0,
    "power word kill": 100.0,  # flat 100: kills any creature <= 100 HP
    "disintegrate": 75.0,
    "finger of death": 62.0,
    "delayed blast fireball": 49.0,
    "prismatic spray": 49.0,
    "harm": 49.0,
    "chain lightning": 45.0,
    "incendiary cloud": 45.0,
    "fire storm": 38.0,
    "cone of cold": 36.0,
    "synaptic static": 36.0,
    "destructive wave": 35.0,
    "blight": 32.0,
    "circle of death": 29.0,
    "fireball": 28.0,
    "lightning bolt": 28.0,
    "flame strike": 28.0,
    "sunburst": 27.0,
    "ice storm": 23.0,
    "cloudkill": 22.0,
    "wall of fire": 22.0,
    "scorching ray": 21.0,
    "call lightning": 16.5,
    "inflict wounds": 16.0,
    "guiding bolt": 14.0,
    "shatter": 13.0,
    "magic missile": 10.5,
    "vampiric touch": 10.5,
    "moonbeam": 10.5,
}

# Cap used when converting a one-shot spell into a *sustained* DPR estimate:
# top-tier novas (Power Word Kill, Meteor Swarm) are once-per-day, so
# sustained casting looks like repeated fireball-class spells.
_SUSTAINED_SPELL_CAP = 45.0


def clean_monster_name(name: str) -> str:
    """Normalize a monster name into the join key used across the project."""
    return str(name).lower().strip().replace("-", " ")


def offense_from_cr(cr: Optional[float]) -> Tuple[float, float, float]:
    """Return (attack_bonus, dpr, save_dc) for a CR via the DMG p.274 table.

    Uses the nearest table row (CRs are snapped, not interpolated — the DMG
    bands are step functions by design).  ``None``/NaN CR maps to CR 1.
    """
    if cr is None or (isinstance(cr, float) and np.isnan(cr)):
        cr = 1.0
    cr = float(cr)
    nearest = min(_CR_KEYS, key=lambda k: abs(k - cr))
    atk, lo, hi, dc = DMG_OFFENSE_BY_CR[nearest]
    return float(atk), (lo + hi) / 2.0, float(dc)


def _strip_html(html: str) -> str:
    return _TAG_RE.sub(" ", html or "")


def _parse_multiattack_count(actions_text: str) -> int:
    """Number of attacks in the Multiattack routine (1 if no Multiattack)."""
    lower = actions_text.lower()
    idx = lower.find("multiattack")
    if idx == -1:
        return 1
    # Only look at the Multiattack sentence(s), not the whole action list.
    segment = actions_text[idx : idx + 300]
    m = _MULTIATTACK_MAKES_RE.search(segment)
    if not m:
        return 2  # Multiattack exists but count unparsed: minimum meaningful value
    token = m.group(1).lower()
    count = _NUMBER_WORDS.get(token)
    if count is None:
        try:
            count = int(token)
        except ValueError:
            count = 2
    return int(np.clip(count, 1, 10))


def _split_actions(html: str) -> List[str]:
    """Split statblock HTML into per-action text segments.

    SRD action headers render as ``<em><strong>Name.</strong></em>`` (order
    varies); each segment is the header plus its rule text, stripped of tags.
    """
    if not html:
        return []
    marked = _ACTION_HEADER_RE.sub(r"\n@@ACTION@@ \1 ", html)
    text = _strip_html(marked)
    segments = [s.strip() for s in text.split("@@ACTION@@") if s.strip()]
    return segments


def _damage_values(segment: str) -> List[float]:
    """Damage expressions in a text block, with alternatives merged.

    Damage riders remain separate entries ("7 slashing plus 3 (1d6) fire"
    -> [7, 3], summed or averaged by the caller), but damage **alternatives**
    are collapsed: versatile weapons print "7 (1d8+3) slashing damage,
    **or** 8 (1d10+3) ... if used with two hands", and the old pooled
    ``findall`` counted both (DPR 15 instead of 8).  Whenever the text
    between two damage expressions contains a standalone "or", the later
    value replaces the earlier via max() instead of appending.

    Returns ``[]`` when the block contains no damage expression at all.
    """
    matches = list(_AVG_DMG_RE.finditer(segment))
    if not matches:
        matches = list(_FLAT_DMG_RE.finditer(segment))

    values: List[float] = []
    for i, m in enumerate(matches):
        val = float(m.group(1))
        if i == 0:
            values.append(val)
            continue
        gap = segment[matches[i - 1].end() : m.start()].lower()
        if re.search(r"\bor\b", gap):
            values[-1] = max(values[-1], val)  # alternative, not a rider
        else:
            values.append(val)  # rider: base + extra damage
    return values


def _scan_spell_damage(text: str) -> float:
    """Max average damage among known spells named in a spell list."""
    lower = text.lower()
    best = 0.0
    for spell, dmg in SPELL_DAMAGE.items():
        if spell in lower:
            best = max(best, dmg)
    return best


def parse_statblock_offense(
    actions_html: str,
    traits_html: str = "",
    cr: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Extract (atk_bonus, dpr, save_dc, burst_dmg) from raw statblock HTML.

    Per-action segmentation keeps the estimates honest:

    * **Weapon attacks** (segments containing "to hit") feed the sustained
      DPR: ``multiattack_count x mean(per-action damage sums)``.  Damage
      riders inside one action ("plus 3 (1d6) fire") are summed together.
    * **Save-based actions** (breath weapons, gaze attacks) are *excluded*
      from the multiattack math — the old pooled regex inflated dragons by
      ~60% — and instead define ``burst_dmg`` (the nova).
    * **Spell lists** in Traits name spells without printing damage; a PHB
      damage table maps them (a Lich's Power Word Kill -> burst 100).  For
      sustained DPR, spell damage is capped at fireball-class output since
      top novas are once per day.

    DPR is clamped to [0.25x, 3x] of the DMG p.274 expectation for the CR.
    Returns ``None`` when no offensive information is present at all.
    """
    text = _strip_html(actions_html or "")
    if not text.strip() and not (traits_html or "").strip():
        return None

    traits_text = _strip_html(traits_html or "")
    full_text = text + " " + traits_text

    bonuses = [int(b.replace(" ", "")) for b in _ATK_BONUS_RE.findall(text)]
    dcs = [int(d) for d in _SAVE_DC_RE.findall(full_text)]

    weapon_damages: List[float] = []
    save_bursts: List[float] = []
    for segment in _split_actions(actions_html or ""):
        dmgs = _damage_values(segment)
        if not dmgs:
            continue
        if "to hit" in segment.lower():
            weapon_damages.append(float(sum(dmgs)))  # base + riders
        elif _SAVE_CONTEXT_RE.search(segment):
            save_bursts.append(float(sum(dmgs)))

    # Fallback for statblocks whose headers defeat the splitter.
    if not weapon_damages and not save_bursts:
        weapon_damages = _damage_values(text)

    spell_burst = _scan_spell_damage(full_text)

    if (
        not bonuses
        and not weapon_damages
        and not save_bursts
        and not dcs
        and not spell_burst
    ):
        return None

    table_atk, table_dpr, table_dc = offense_from_cr(cr)
    atk_bonus = float(max(bonuses)) if bonuses else table_atk
    save_dc = float(max(dcs)) if dcs else table_dc

    if weapon_damages:
        n_attacks = _parse_multiattack_count(text)
        dpr = n_attacks * float(np.mean(weapon_damages))
    else:
        dpr = 0.0
    # Sustained spellcasting competes with weapon routines.
    if spell_burst:
        dpr = max(dpr, min(spell_burst, _SUSTAINED_SPELL_CAP))
    if save_bursts and not weapon_damages and not spell_burst:
        dpr = max(dpr, float(np.mean(save_bursts)))
    if dpr <= 0:
        dpr = table_dpr
    if cr is not None and not (isinstance(cr, float) and np.isnan(cr)):
        dpr = float(np.clip(dpr, 0.25 * max(table_dpr, 1.0), 3.0 * max(table_dpr, 1.0)))

    # Nova: the single scariest thing this monster can do in one action.
    burst = max([dpr] + save_bursts + ([spell_burst] if spell_burst else []))

    return {
        "atk_bonus": atk_bonus,
        "dpr": dpr,
        "save_dc": save_dc,
        "burst_dmg": float(burst),
    }


def _parse_srd_cr(challenge: Any) -> Optional[float]:
    """Parse SRD 'Challenge' strings like ``"10 (5,900 XP)"`` or ``"1/4 (50 XP)"``."""
    s = str(challenge or "").strip()
    m = re.match(r"(\d+)\s*/\s*(\d+)", s)
    if m:
        denom = float(m.group(2))
        return float(m.group(1)) / denom if denom else None
    m = re.match(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def load_srd_offense(srd_json_path: str = "srd_5e_monsters.json") -> pd.DataFrame:
    """Parse every SRD statblock into an offense table keyed by clean_name."""
    with open(srd_json_path, "r", encoding="utf-8") as handle:
        monsters: List[Mapping[str, Any]] = json.load(handle)

    rows: List[Dict[str, Any]] = []
    for mon in monsters:
        cr = _parse_srd_cr(mon.get("Challenge"))
        offense = parse_statblock_offense(
            mon.get("Actions", ""), mon.get("Traits", ""), cr=cr
        )
        if offense is None:
            continue
        rows.append(
            {
                "clean_name": clean_monster_name(mon.get("name", "")),
                "atk_bonus": offense["atk_bonus"],
                "dpr": offense["dpr"],
                "save_dc": offense["save_dc"],
                "burst_dmg": offense["burst_dmg"],
                "offense_source": "srd_statblock",
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["clean_name"])
    LOGGER.info("Parsed offensive stats for %d SRD monsters.", len(df))
    return df


def build_offense_table(
    monster_names_and_crs: pd.DataFrame,
    srd_json_path: str = "srd_5e_monsters.json",
    output_csv: Optional[str] = "monster_offense_stats.csv",
) -> pd.DataFrame:
    """Build the full offense table for a bestiary frame.

    Parameters
    ----------
    monster_names_and_crs : DataFrame with columns ``clean_name`` and ``cr_num``.
    srd_json_path : path to the SRD statblock JSON.
    output_csv : where to persist the table (``None`` to skip writing).

    Returns
    -------
    DataFrame with one row per clean_name:
    ``clean_name, atk_bonus, dpr, save_dc, offense_source``.
    """
    base = monster_names_and_crs[["clean_name", "cr_num"]].drop_duplicates("clean_name")

    try:
        srd = load_srd_offense(srd_json_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        LOGGER.warning("SRD statblocks unavailable (%s); using DMG CR table only.", exc)
        srd = pd.DataFrame(
            columns=[
                "clean_name",
                "atk_bonus",
                "dpr",
                "save_dc",
                "burst_dmg",
                "offense_source",
            ]
        )

    merged = base.merge(srd, on="clean_name", how="left")

    missing = merged["atk_bonus"].isna()
    if missing.any():
        fallback = merged.loc[missing, "cr_num"].map(offense_from_cr)
        merged.loc[missing, "atk_bonus"] = fallback.map(lambda t: t[0])
        merged.loc[missing, "dpr"] = fallback.map(lambda t: t[1])
        merged.loc[missing, "save_dc"] = fallback.map(lambda t: t[2])
        # No statblock -> no visible nova; assume burst == sustained DPR.
        merged.loc[missing, "burst_dmg"] = merged.loc[missing, "dpr"]
        merged.loc[missing, "offense_source"] = "dmg_cr_table"

    n_parsed = (merged["offense_source"] == "srd_statblock").sum()
    LOGGER.info(
        "Offense table: %d/%d monsters from real statblocks, %d imputed from DMG CR table.",
        n_parsed,
        len(merged),
        len(merged) - n_parsed,
    )

    out = merged[
        ["clean_name", "atk_bonus", "dpr", "save_dc", "burst_dmg", "offense_source"]
    ]
    if output_csv:
        out.to_csv(output_csv, index=False)
        LOGGER.info("Wrote offense table -> %s", output_csv)
    return out


def attach_offense(
    df: pd.DataFrame,
    offense_table: pd.DataFrame,
    name_col: str = "clean_name",
) -> pd.DataFrame:
    """Left-join offense columns onto ``df`` by monster name."""
    return df.merge(
        offense_table.rename(columns={"clean_name": name_col}),
        on=name_col,
        how="left",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    bestiary = pd.read_csv("dnd_monsters.csv")
    bestiary["clean_name"] = bestiary["name"].map(clean_monster_name)

    def _cr_to_float(raw: Any) -> Optional[float]:
        s = str(raw).strip()
        if "/" in s:
            num, _, den = s.partition("/")
            try:
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return None
        try:
            return float(s)
        except ValueError:
            return None

    bestiary["cr_num"] = bestiary["cr"].map(_cr_to_float)
    table = build_offense_table(bestiary)
    print(table.groupby("offense_source").describe().T.round(2))
