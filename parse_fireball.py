from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import glob

import pandas as pd

from monster_offense import build_offense_table, cr_to_xp, extract_official_traits

LOGGER = logging.getLogger(__name__)

# FIREBALL is full of junk: homebrew above level 20, mis-parsed sheets,
# mass-summon fights with 200+ tokens. Cap everything to the legal range.
MAX_PARTY_LEVEL = 20.0
MAX_MONSTERS = 30

# The 12 base classes. Substring match on the lowercased class field.
CORE_CLASSES: Tuple[str, ...] = (
    "barbarian",
    "bard",
    "cleric",
    "druid",
    "fighter",
    "monk",
    "paladin",
    "ranger",
    "rogue",
    "sorcerer",
    "warlock",
    "wizard",
)

CLASS_COUNT_COLUMNS: Tuple[str, ...] = tuple(f"num_{c}" for c in CORE_CLASSES)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_challenge_rating(raw: Any) -> Optional[float]:
    """
    Convert D&D 5e challenge ratings like ``1/4`` or ``17`` to a float.

    Returns ``None`` when the value is missing or non-numeric so downstream
    aggregation can skip corrupt rows without failing the whole pipeline.
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, float) and pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", ""}:
        return None
    if "/" in s:
        parts = s.split("/", 1)
        try:
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            LOGGER.debug("Unparseable fractional CR: %r", raw)
            return None
    try:
        return float(s)
    except ValueError:
        LOGGER.debug("Unparseable CR: %r", raw)
        return None


def safe_float(raw: Any) -> Optional[float]:
    """Parse HP/AC with tolerant handling of blanks and European decimals."""
    if raw is None:
        return None
    try:
        if isinstance(raw, float) and pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        LOGGER.debug("Unparseable numeric: %r", raw)
        return None


def extract_total_pc_level(class_str: Optional[str]) -> int:
    """Sum every integer in a multiclass string, e.g. 'Fighter 3 / Rogue 2' -> 5.

    Any digit group counts toward total character level.
    """
    if not class_str:
        return 0
    levels = re.findall(r"\d+", str(class_str))
    return sum(int(x) for x in levels) if levels else 0


def is_dead(hp_str: Optional[str]) -> bool:
    """Guess dead/down from an Avrae HP string.

    It's a heuristic - the logs are messy - and I let it miss rather than
    over-fire, so unsure cases fall through to 'Ongoing'.
    """
    if not hp_str:
        return False
    hp_lower = str(hp_str).lower()
    if "dead" in hp_lower:
        return True
    if "<0/" in hp_lower:
        return True
    if hp_lower == "0" or hp_lower.startswith("0/"):
        return True
    return False


def infer_turn_outcome(
    pcs: Sequence[Mapping[str, Any]],
    monsters: Sequence[Mapping[str, Any]],
) -> str:
    """Four-way label for a snapshot from who's dead.

    'Ongoing' snapshots get dropped later - training only on resolved fights
    keeps the labels clean.
    """
    party_casualty = any(is_dead(pc.get("hp")) for pc in pcs)
    dead_monsters = [m for m in monsters if is_dead(m.get("hp"))]

    if dead_monsters and not party_casualty:
        return "Party Win"
    if party_casualty and not dead_monsters:
        return "Party Loss/Casualty"
    if dead_monsters and party_casualty:
        return "Mixed/Pyrrhic"
    return "Ongoing"


def count_core_classes(pcs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """
    Count how many PCs bring each core class keyword (substring search).

    Multiclass PCs can contribute +1 to multiple tallies simultaneously.
    """
    counts = {f"num_{c}": 0 for c in CORE_CLASSES}
    for pc in pcs:
        class_string = str(pc.get("class", "")).lower()
        for c in CORE_CLASSES:
            if c in class_string:
                counts[f"num_{c}"] += 1
    return counts


def make_encounter_id(data: Mapping[str, Any], file_path: str, line_idx: int) -> str:
    """Stable key to group turns into one encounter.

    Use combat_id if the log has it, otherwise fall back to
    '{jsonl_basename}_{before_state_idx}'.
    """
    explicit = data.get("combat_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    before_idx = data.get("before_state_idx", line_idx)
    return f"{os.path.basename(file_path)}_{before_idx}"


def match_monster_name(
    actor: Mapping[str, Any], monster_lookup: Mapping[str, Any]
) -> Optional[str]:
    """Return lookup key (clean_name) if ``name`` or ``race`` hits the index."""
    m_name = str(actor.get("name", "")).lower().strip()
    m_race = str(actor.get("race", "")).lower().strip()
    if m_name and m_name in monster_lookup:
        return m_name
    if m_race and m_race in monster_lookup:
        return m_race
    return None


def final_outcome_from_series(outcomes: Iterable[str]) -> str:
    """Last non-ongoing label in temporal order; if none, ``Ongoing``."""
    resolved = [o for o in outcomes if o != "Ongoing"]
    return resolved[-1] if resolved else "Ongoing"


def summarize_monster_names(nested: Iterable[Iterable[str]]) -> str:
    """Readable '1x ogre, 2x bandit' string for the summary tables."""
    flat: List[str] = [m for sub in nested for m in sub]
    if not flat:
        return ""
    counts = Counter(flat)
    return ", ".join(f"{n}x {name}" for name, n in counts.items())


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_monster_frame(
    monsters_csv: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Load dnd_monsters.csv, return (annotated DataFrame, dict index).

    DataFrame has numeric CR/HP/AC for the joins; the dict is just for
    poking around when debugging.
    """
    try:
        df_raw = pd.read_csv(monsters_csv)
    except FileNotFoundError:
        LOGGER.exception("Monster CSV not found: %s", monsters_csv)
        raise
    except pd.errors.ParserError:
        LOGGER.exception("Failed to parse monster CSV (malformed rows).")
        raise

    df_raw = df_raw.copy()
    df_raw["clean_name"] = (
        df_raw["name"].astype(str).str.replace("-", " ", regex=False).str.lower()
    )
    df_raw["cr_num"] = df_raw["cr"].map(parse_challenge_rating)
    df_raw["hp_num"] = df_raw["hp"].map(safe_float)
    df_raw["ac_num"] = df_raw["ac"].map(safe_float)

    # Monster traits, part 1: the ordinal ones straight from dnd_monsters.csv
    df_raw["monster_is_legendary"] = (
        df_raw["legendary"].fillna("").str.strip().str.lower().eq("legendary")
    ).astype(int)
    _speed = df_raw["speed"].fillna("").str.lower()
    df_raw["monster_has_mobility"] = _speed.str.contains("fly|swim", regex=True).astype(
        int
    )

    SIZE_ORDINAL = {
        "tiny": 1,
        "small": 2,
        "medium": 3,
        "large": 4,
        "huge": 5,
        "gargantuan": 6,
    }
    df_raw["monster_size_num"] = (
        df_raw["size"].str.strip().str.lower().map(SIZE_ORDINAL)
    )

    _STAT_COLS = ["str", "dex", "con", "int", "wis", "cha"]
    df_raw["monster_stat_sum"] = df_raw[_STAT_COLS].sum(axis=1, min_count=1)
    _cr_medians = df_raw.groupby("cr_num")["monster_stat_sum"].transform("median")
    df_raw["monster_stat_sum"] = df_raw["monster_stat_sum"].fillna(_cr_medians)
    df_raw["monster_stat_sum"] = df_raw["monster_stat_sum"].fillna(
        df_raw["monster_stat_sum"].median()
    )

    # Part 2: the mechanical traits from the Official Stats CSV
    try:
        df_official = pd.read_csv("Monster Spreadsheet (D&D5e) - Official Stats.csv")

        # Names don't line up cleanly ("angel, deva", "assorted beast, aurochs"),
        # so try a few reorderings of each name as candidate keys.
        def _get_fuzzy_keys(name: str) -> List[str]:
            n = str(name).lower().strip()
            import re

            n_no_parens = re.sub(r"\(.*?\)", "", n).strip().replace("-", " ")
            keys = [n, n_no_parens]
            if "," in n_no_parens:
                parts = [p.strip() for p in n_no_parens.split(",")]
                keys.append(parts[-1])
                keys.append(parts[0])
                if len(parts) == 2:
                    keys.append(parts[1] + " " + parts[0])
            return keys

        # Build a lookup dictionary from the official stats
        official_lookup = {}
        for _, row in df_official.iterrows():
            wri = str(row.get("WRI", "")).lower()
            additional = str(row.get("Additional", "")).lower()
            # If both are nan/empty string, treat as empty
            if wri == "nan":
                wri = ""
            if additional == "nan":
                additional = ""

            for key in _get_fuzzy_keys(row.get("Name", "")):
                if key not in official_lookup:
                    official_lookup[key] = {"WRI": wri, "Additional": additional}

        # Apply lookup to our dnd_monsters
        def _lookup_trait(name: str, trait_col: str) -> str:
            return official_lookup.get(name, {}).get(trait_col, "")

        df_raw["_WRI"] = df_raw["clean_name"].apply(lambda n: _lookup_trait(n, "WRI"))
        df_raw["_Additional"] = df_raw["clean_name"].apply(
            lambda n: _lookup_trait(n, "Additional")
        )

        # Go through the shared extractor so these match the app exactly.
        # This file used to have its own slightly-different regexes and the
        # CR predictor ended up trained on different definitions - don't.
        traits = extract_official_traits(df_raw["_WRI"], df_raw["_Additional"])
        df_raw["monster_has_physical_res"] = traits["physical_res"]
        df_raw["monster_immune_to_cc"] = traits["cc_immune"]
        df_raw["monster_has_magic_res"] = traits["magic_res"]
        df_raw["monster_has_pack_tactics"] = traits["pack_tactics"]
        df_raw["monster_has_spellcasting"] = traits["spellcasting"]
        df_raw["monster_has_regeneration"] = traits["regeneration"]

        matches = (df_raw["_WRI"] != "") | (df_raw["_Additional"] != "")
        LOGGER.info(
            "Successfully merged Official Stats for %d / %d monsters.",
            matches.sum(),
            len(df_raw),
        )

    except FileNotFoundError:
        LOGGER.warning(
            "Monster Spreadsheet (D&D5e) - Official Stats.csv not found! Defaulting deep traits to 0."
        )
        for col in [
            "monster_has_physical_res",
            "monster_immune_to_cc",
            "monster_has_magic_res",
            "monster_has_pack_tactics",
            "monster_has_spellcasting",
            "monster_has_regeneration",
        ]:
            df_raw[col] = 0

    # Offense: DPR / attack bonus / save DC from the SRD statblocks, with the
    # DMG by-CR table filling in whatever isn't in the SRD.
    offense = build_offense_table(
        df_raw[["clean_name", "cr_num"]], output_csv="monster_offense_stats.csv"
    )
    df_raw = df_raw.merge(
        offense.rename(
            columns={
                "atk_bonus": "monster_atk_bonus",
                "dpr": "monster_dpr",
                "save_dc": "monster_save_dc",
                "burst_dmg": "monster_burst",
            }
        ).drop(columns=["offense_source"]),
        on="clean_name",
        how="left",
    )
    df_raw["monster_xp"] = df_raw["cr_num"].map(cr_to_xp)

    # Drop unusable stat rows for join quality metrics (keep row for existence).
    lookup = df_raw.set_index("clean_name").to_dict(orient="index")
    LOGGER.info("Loaded %d monsters from %s.", len(lookup), monsters_csv)
    return df_raw, lookup


def parse_turn_line(
    line: str,
    file_path: str,
    line_idx: int,
    monster_lookup: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Parse a single JSONL line into a flat turn record, or ``None`` if skipped.

    Raises ``json.JSONDecodeError`` to the caller for optional bad-line metrics.
    """
    data = json.loads(line)
    if "combat_state_after" not in data:
        return None

    state_after = data["combat_state_after"]
    if not isinstance(state_after, list):
        return None

    pcs = [a for a in state_after if a.get("class")]
    monsters_in_combat = [a for a in state_after if not a.get("class")]
    if not pcs or not monsters_in_combat:
        return None

    # Clip each PC's parsed level into the legal 1-20 band: multiclass strings
    # in real logs carry junk digits ("Fighter 3 (AC 17)") that would otherwise
    # inflate party level to 40+.
    pc_levels = [
        min(max(extract_total_pc_level(pc.get("class")), 1), int(MAX_PARTY_LEVEL))
        for pc in pcs
    ]
    avg_party_level = sum(pc_levels) / len(pc_levels) if pc_levels else 0.0
    class_counts = count_core_classes(pcs)
    outcome = infer_turn_outcome(pcs, monsters_in_combat)

    matched: List[str] = []
    for m in monsters_in_combat:
        key = match_monster_name(m, monster_lookup)
        if key is not None:
            matched.append(key)

    encounter_id = make_encounter_id(data, file_path, line_idx)

    row: Dict[str, Any] = {
        "encounter_id": encounter_id,
        "avg_party_level": float(avg_party_level),
        "combat_outcome": outcome,
        "monsters_present": matched,
        # True action economy: ALL hostile actors on the field this turn,
        # including homebrew monsters that miss the bestiary join.
        "monsters_on_field": len(monsters_in_combat),
    }
    row.update(class_counts)
    return row


def aggregate_encounters(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Collapse turn rows to encounter grain with composition maxima and outcome logic."""

    def agg_monsters(series: pd.Series) -> List[str]:
        nested = series.tolist()
        from collections import Counter

        max_counts = {}
        for sub in nested:
            counts = Counter(sub)
            for k, v in counts.items():
                max_counts[k] = max(max_counts.get(k, 0), v)
        out = []
        for k, v in max_counts.items():
            out.extend([k] * v)
        return out

    agg_spec: Dict[str, Any] = {
        "avg_party_level": ("avg_party_level", "mean"),
        "final_outcome": ("combat_outcome", final_outcome_from_series),
        "monster_summary": (
            "monsters_present",
            lambda s: summarize_monster_names([agg_monsters(s)]),
        ),
        "unique_monsters_list": ("monsters_present", agg_monsters),
        "num_monsters": ("monsters_present", lambda s: len(agg_monsters(s))),
        # Most enemies on the field at once, matched or not. This is the
        # action-economy number I actually trust - it survives join misses.
        "num_monsters_total": ("monsters_on_field", "max"),
    }
    for col in CLASS_COUNT_COLUMNS:
        agg_spec[col] = (col, "max")

    grouped = df_turns.groupby("encounter_id", sort=False).agg(**agg_spec)
    grouped = grouped.reset_index()
    return grouped


def attach_monster_stats(
    df_encounters: pd.DataFrame, df_monsters: pd.DataFrame
) -> pd.DataFrame:
    """Explode the matched monsters, join their stats, aggregate per encounter.

    Encounters with no matches come out as NaN - I keep them so I can see
    how many there are.
    """
    core = df_monsters[
        [
            "clean_name",
            "cr_num",
            "hp_num",
            "ac_num",
            "monster_is_legendary",
            "monster_has_mobility",
            "monster_size_num",
            "monster_stat_sum",
            "monster_has_physical_res",
            "monster_immune_to_cc",
            "monster_has_magic_res",
            "monster_has_pack_tactics",
            "monster_has_spellcasting",
            "monster_has_regeneration",
            "monster_atk_bonus",
            "monster_dpr",
            "monster_save_dc",
            "monster_burst",
            "monster_xp",
        ]
    ].drop_duplicates(subset=["clean_name"])

    exploded = df_encounters.explode("unique_monsters_list", ignore_index=True)
    exploded = exploded.rename(columns={"unique_monsters_list": "clean_name"})
    merged = exploded.merge(core, on="clean_name", how="left")

    stats = merged.groupby("encounter_id", sort=False).agg(
        avg_monster_cr=("cr_num", "mean"),
        avg_monster_hp=("hp_num", "mean"),
        avg_monster_ac=("ac_num", "mean"),
        # avg CR lies on mixed rosters: (1 dragon + 10 kobolds) averages out
        # to something trivial. So also keep max CR (the scary one) and the
        # totals (HP and XP summed over every monster instance).
        max_monster_cr=("cr_num", "max"),
        total_monster_hp=("hp_num", "sum"),
        total_monster_xp=("monster_xp", "sum"),
        # offense
        avg_monster_dpr=("monster_dpr", "mean"),
        total_monster_dpr=("monster_dpr", "sum"),
        max_monster_atk_bonus=("monster_atk_bonus", "max"),
        max_monster_save_dc=("monster_save_dc", "max"),
        # the single biggest hit anything in the roster can land (breath
        # weapon, best damage spell)
        max_monster_burst=("monster_burst", "max"),
        # flags use max: if ANY monster is legendary/flying/etc., the whole
        # encounter has to deal with it
        monster_is_legendary=("monster_is_legendary", "max"),
        monster_has_mobility=("monster_has_mobility", "max"),
        monster_has_physical_res=("monster_has_physical_res", "max"),
        monster_immune_to_cc=("monster_immune_to_cc", "max"),
        monster_has_magic_res=("monster_has_magic_res", "max"),
        monster_has_pack_tactics=("monster_has_pack_tactics", "max"),
        monster_has_spellcasting=("monster_has_spellcasting", "max"),
        monster_has_regeneration=("monster_has_regeneration", "max"),
        # Continuous features use mean — average size and stat budget across
        # all monsters in the encounter.
        avg_monster_size_num=("monster_size_num", "mean"),
        avg_monster_stat_sum=("monster_stat_sum", "mean"),
    )
    stats = stats.reset_index()

    out = df_encounters.drop(columns=["unique_monsters_list"]).merge(
        stats, on="encounter_id", how="left"
    )
    return out


def parse_and_aggregate_fireball(
    jsonl_dir: str,
    monsters_csv: str,
    output_csv: str,
    *,
    skip_ongoing: bool = True,
) -> pd.DataFrame:
    """
    End-to-end parse, aggregate, and export.

    Returns the encounter-level DataFrame for interactive notebooks.
    """
    df_monsters, monster_lookup = build_monster_frame(monsters_csv)

    jsonl_pattern = os.path.join(jsonl_dir, "*.jsonl")
    jsonl_files = sorted(glob.glob(jsonl_pattern))
    if not jsonl_files:
        LOGGER.warning("No JSONL files matched pattern: %s", jsonl_pattern)

    all_turns: List[Dict[str, Any]] = []
    json_errors = 0
    skipped_empty = 0
    total_lines = 0

    for file_path in jsonl_files:
        LOGGER.info("Processing %s", file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line_idx, line in enumerate(handle):
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = parse_turn_line(
                            line, file_path, line_idx, monster_lookup
                        )
                    except json.JSONDecodeError:
                        json_errors += 1
                        LOGGER.debug("JSON decode error at %s:%s", file_path, line_idx)
                        continue
                    except (TypeError, KeyError, AttributeError) as exc:
                        LOGGER.debug("Malformed combat state (%s): %s", file_path, exc)
                        skipped_empty += 1
                        continue

                    if record is None:
                        skipped_empty += 1
                        continue
                    all_turns.append(record)
        except OSError as exc:
            LOGGER.error("Failed to read %s: %s", file_path, exc)
            raise

    LOGGER.info(
        "Finished %d lines across %d files (json_errors=%d, skipped=%d, kept_turns=%d).",
        total_lines,
        len(jsonl_files),
        json_errors,
        skipped_empty,
        len(all_turns),
    )

    if not all_turns:
        LOGGER.error("No valid turns parsed; not writing CSV.")
        return pd.DataFrame()

    df_turns = pd.DataFrame(all_turns)
    df_encounters = aggregate_encounters(df_turns)

    if skip_ongoing:
        before = len(df_encounters)
        df_encounters = df_encounters[
            df_encounters["final_outcome"] != "Ongoing"
        ].copy()
        LOGGER.info(
            "Dropped %d ongoing encounters (unresolved).", before - len(df_encounters)
        )

    # Encounters with zero bestiary matches carry no monster signal at all —
    # they were previously exported (21% of rows) only to be dropped at train
    # time.  Remove them here so the CSV is the actual training population.
    before = len(df_encounters)
    df_encounters = df_encounters[df_encounters["num_monsters"] > 0].copy()
    LOGGER.info(
        "Dropped %d encounters with no matched monsters.", before - len(df_encounters)
    )

    # Winsorize action-economy outliers (mass-summon logs reach 241 actors).
    # The roster list is truncated too, so total_* sums match the capped count.
    df_encounters["num_monsters"] = df_encounters["num_monsters"].clip(
        upper=MAX_MONSTERS
    )
    df_encounters["num_monsters_total"] = df_encounters["num_monsters_total"].clip(
        upper=MAX_MONSTERS
    )
    df_encounters["unique_monsters_list"] = df_encounters["unique_monsters_list"].map(
        lambda lst: lst[:MAX_MONSTERS]
    )

    df_final = attach_monster_stats(df_encounters, df_monsters)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    except OSError:
        # dirname may be empty for bare filename
        pass

    try:
        df_final.to_csv(output_csv, index=False)
    except OSError:
        LOGGER.exception("Failed to write output CSV: %s", output_csv)
        raise

    LOGGER.info("Wrote %d encounter rows -> %s", len(df_final), output_csv)
    return df_final


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jsonl-dir",
        default="FIREBALL/filtered",
        help="Directory containing filtered *.jsonl FIREBALL extracts.",
    )
    p.add_argument(
        "--monsters-csv",
        default="dnd_monsters.csv",
        help="Official-ish monster stat compendium (CSV).",
    )
    p.add_argument(
        "--output",
        default="clean_aggregated_combat_data.csv",
        help="Encounter-level CSV for modeling (Part 2).",
    )
    p.add_argument(
        "--keep-ongoing",
        action="store_true",
        help="Retain unresolved 'Ongoing' encounters (normally dropped).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.verbose)

    try:
        parse_and_aggregate_fireball(
            args.jsonl_dir,
            args.monsters_csv,
            args.output,
            skip_ongoing=not args.keep_ongoing,
        )
    except Exception:
        LOGGER.exception("Pipeline failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
