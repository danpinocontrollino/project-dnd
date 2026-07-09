"""True Lethality Engine — Streamlit interface.

Loads the calibrated pipeline + CR predictor, and for any official or
homebrew monster finds the party level with a 65% predicted win rate,
sweeps 400 hypothetical parties, and plots the win-probability curves.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit config MUST be the very first Streamlit command.
st.set_page_config(
    page_title="True Lethality Engine",
    page_icon="🐉",
    layout="wide",
)

import __main__

from initial_learn import DnDFeatureEngineer, _classify_xp_tier, _cr_to_xp
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
    normalize_roster,
    predict_wotc_cr,
    profile_from_db_row,
    roster_monster_fields,
    simulate_party_grid,
    win_curve,
)
from monster_offense import encounter_xp_multiplier, offense_from_cr

# Backward-compat shim for OLD pickles trained via `python3 initial_learn.py`
# (those reference __main__.DnDFeatureEngineer).  Current pickles are trained
# through the module path and don't need this, but it's harmless to keep.
__main__.DnDFeatureEngineer = DnDFeatureEngineer

# ── Palette (validated categorical slots, theme-aware) ─────────────────────
try:
    _IS_DARK = st.context.theme.type == "dark"
except Exception:
    _IS_DARK = False

if _IS_DARK:
    SERIES = {
        "Balanced": "#3987e5",       # blue   (slot 1, dark step)
        "Glass Cannons": "#199e70",  # aqua   (slot 2)
        "The Wall": "#c98500",       # yellow (slot 3)
        "Melee Rush": "#008300",     # green  (slot 4)
        "Full Caster": "#9085e9",    # violet (slot 5)
    }
    INK = "#ffffff"
    INK_2 = "#c3c2b7"
    MUTED = "#898781"
    GRID = "#2c2c2a"
    SURFACE = "#1a1a19"
    BAND = "#383835"
else:
    SERIES = {
        "Balanced": "#2a78d6",
        "Glass Cannons": "#1baf7a",
        "The Wall": "#eda100",
        "Melee Rush": "#008300",
        "Full Caster": "#4a3aa7",
    }
    INK = "#0b0b0b"
    INK_2 = "#52514e"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    SURFACE = "#fcfcfb"
    BAND = "#f0efec"

TIER_NAMES = {0: "Easy", 1: "Medium", 2: "Hard", 3: "Deadly", 4: "☠️ Super-Deadly"}


# ── Cached loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load("true_lethality_model.pkl")
    except Exception as exc:
        st.error(f"Failed to load True Lethality model: {exc}")
        pipeline = None
    try:
        cr_predictor = load_cr_predictor()  # native JSON first, pkl fallback
    except Exception:
        cr_predictor = None
    return pipeline, cr_predictor


@st.cache_data
def load_db() -> pd.DataFrame:
    try:
        return load_monster_database()
    except Exception as exc:
        st.error(f"Failed to load monster database: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    try:
        with open("figures/metrics.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ── Charts ─────────────────────────────────────────────────────────────────
def _base_layout(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK)),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                  color=INK_2, size=12),
        xaxis=dict(title=xaxis_title, gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(title=yaxis_title, gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED)),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        margin=dict(l=55, r=15, t=50, b=10),
        hovermode="x unified",
    )


def plot_win_curves(df_curve: pd.DataFrame, appraisal: dict,
                    target: float) -> go.Figure:
    """Win probability vs party level, one 2px line per composition."""
    fig = go.Figure()

    fig.add_hrect(y0=max(0, target - 0.10), y1=min(1, target + 0.10),
                  fillcolor=BAND, opacity=0.6, line_width=0)
    fig.add_hline(
        y=target, line_dash="dash", line_color=MUTED, line_width=1,
        annotation_text=f"target {target:.0%}",
        annotation_font_color=MUTED,
    )

    for comp, color in SERIES.items():
        sub = df_curve[df_curve["comp_name"] == comp]
        fig.add_trace(
            go.Scatter(
                x=sub["avg_party_level"], y=sub["win_prob"],
                mode="lines", name=comp,
                line=dict(color=color, width=2),
                hovertemplate="Level %{x} · %{y:.1%}<extra>" + comp + "</extra>",
            )
        )

    if appraisal["verdict"] == "ok":
        fig.add_vline(
            x=appraisal["level"], line_dash="dot", line_color=INK,
            line_width=1.5,
            annotation_text=f"True Lethality Level {appraisal['level']:g}",
            annotation_font_color=INK,
        )
    _base_layout(fig, "Win probability vs. party level (party of 4)",
                 "Average party level", "P(Party Win)")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_xaxes(dtick=2)
    return fig


def plot_size_level_heatmap(df_sim: pd.DataFrame) -> go.Figure:
    """Sequential-blue heatmap: win prob by party size x level (Balanced)."""
    sub = df_sim[df_sim["comp_name"] == "Balanced"]
    grid = sub.pivot_table(index="party_size", columns="avg_party_level",
                           values="win_prob")
    ramp = [
        [0.0, "#cde2fb"], [0.25, "#9ec5f4"], [0.5, "#5598e7"],
        [0.75, "#256abf"], [1.0, "#0d366b"],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=[str(c) for c in grid.columns],
            y=[f"{r} PCs" for r in grid.index],
            colorscale=ramp, zmin=0, zmax=1,
            colorbar=dict(title="P(win)", tickformat=".0%"),
            hovertemplate="Level %{x} · %{y}<br>P(win) %{z:.1%}<extra></extra>",
            xgap=2, ygap=2,
        )
    )
    _base_layout(fig, "Balanced-party win probability by size and level",
                 "Average party level", "")
    fig.update_layout(hovermode="closest")
    return fig


def render_results(pipeline, monster, num_monsters: int,
                   baseline_cr: float, baseline_label: str,
                   target: float = TARGET_WIN_RATE):
    """``monster`` is a MonsterProfile or a roster [(profile, count), ...]."""
    roster = normalize_roster(monster, num_monsters)
    fields = roster_monster_fields(roster)
    with st.spinner("Binary-searching the lethality frontier…"):
        appraisal = lethality_appraisal(
            pipeline, monster, num_monsters, target=target
        )
        df_curve = win_curve(pipeline, monster, num_monsters)
        df_sim = simulate_party_grid(pipeline, monster, num_monsters)

    verdict = appraisal["verdict"]
    lethality_level = appraisal["level"]

    # By-the-book estimate (DMG p.82): sum XP, apply the encounter
    # multiplier, express the adjusted total back in CR units.  6x CR-1
    # monsters must NOT display as a CR-1 fight on the book side.
    book = official_encounter_estimate(monster, num_monsters)
    multiple = book["num_monsters"] > 1
    book_cr = book["effective_cr"] if multiple else baseline_cr

    st.divider()
    c1, c2, c3 = st.columns(3)
    if multiple:
        c1.metric(
            baseline_label, f"CR {book_cr:g}",
            delta=f"{book['num_monsters']:.0f} monsters, DMG-adjusted",
            delta_color="off",
            help="Computed with the official DMG p.82 procedure — see the "
                 "book-math line below.",
        )
    else:
        c1.metric(baseline_label, f"CR {baseline_cr:g}")
    level_text = {"trivial": "≤ 1", "beyond_deadly": "> 20"}.get(
        verdict, f"{lethality_level:g}"
    )
    c2.metric("⚡ True Lethality Level", level_text,
              delta=f"{appraisal['p_at_level']:.1%} win at this level",
              delta_color="off",
              help=f"Party level at which a balanced party of 4 reaches a "
                   f"{target:.0%} predicted win rate against this encounter. "
                   f"The sub-metric shows the exact win probability at the "
                   f"appraised level — encounters can share a level but "
                   f"differ in risk.")
    c3.metric("P(win) @ level 1 → 20",
              f"{appraisal['p_level_1']:.0%} → {appraisal['p_level_20']:.0%}")

    if multiple:
        st.caption(
            f"📖 Book math (DMG p.82): {book['total_xp']:,.0f} XP total "
            f"× **×{book['multiplier']:g}** encounter multiplier for "
            f"{book['num_monsters']:.0f} monsters = "
            f"**{book['adjusted_xp']:,.0f} adjusted XP** — as difficult as "
            f"a single **CR {book_cr:g}** monster."
        )

    if verdict == "trivial":
        st.info(
            f"🕊️ Even a **level-1** balanced party is predicted to win "
            f"{appraisal['p_level_1']:.0%} of the time — real tables reliably "
            f"beat this encounter at any level. (FIREBALL parties win 83% of "
            f"all resolved fights, so weak monsters saturate early.)"
        )
    elif verdict == "beyond_deadly":
        st.error(
            f"☠️ **Beyond deadly.** Even a level-20 balanced party only wins "
            f"{appraisal['p_level_20']:.0%} of the time — below your "
            f"{target:.0%} target. This is a TPK machine."
        )
    else:
        diff = lethality_level - book_cr
        if abs(diff) >= 2:
            st.error(
                f"🚨 **Major discrepancy.** The engine rates this encounter "
                f"**{abs(diff):.1f} levels {'harder' if diff > 0 else 'easier'}** "
                f"than the official rating suggests."
            )
        elif abs(diff) >= 1:
            st.warning(
                f"⚠️ The official rating is off by **{abs(diff):.1f} levels** "
                f"for this encounter."
            )

    st.plotly_chart(plot_win_curves(df_curve, appraisal, target),
                    use_container_width=True, theme=None)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(plot_size_level_heatmap(df_sim),
                        use_container_width=True, theme=None)
    with col_b:
        st.subheader("⚔️ Fairest matchups")
        best = fair_fight_matches(df_sim, target=target)
        # DMG tier from adjusted XP: total roster XP x encounter multiplier,
        # party-size adjusted per row (DMG p.83).
        table = []
        for _, r in best.iterrows():
            adjusted_xp = fields["total_monster_xp"] * encounter_xp_multiplier(
                fields["num_monsters_total"], r["party_size"]
            )
            tier = _classify_xp_tier(
                adjusted_xp, r["avg_party_level"], r["party_size"]
            )
            table.append({
                "Party level": int(r["avg_party_level"]),
                "Size": int(r["party_size"]),
                "Composition": r["comp_name"],
                "Win %": f"{r['win_prob']:.1%}",
                "DMG tier": TIER_NAMES.get(tier, "?"),
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True,
                     hide_index=True)

        threat_flags = []
        if fields["monster_is_legendary"]:
            threat_flags.append("👑 Legendary — bring a healer for the attrition war")
        if fields["monster_has_mobility"]:
            threat_flags.append("🪽 Fly/swim speed — melee-only parties will struggle")
        if fields["monster_has_magic_res"]:
            threat_flags.append("🛡️ Magic resistance — spell-heavy parties lose value")
        if fields["monster_has_physical_res"]:
            threat_flags.append("⚔️ Physical resistance — martial damage is halved")
        if fields["monster_has_pack_tactics"] and fields["num_monsters_total"] > 1:
            threat_flags.append("🐺 Pack tactics × multiple monsters — advantage everywhere")
        if fields["max_monster_burst"] >= 50:
            threat_flags.append(
                f"💥 Nova threat — its scariest single action deals "
                f"~{fields['max_monster_burst']:.0f} damage"
            )
        for flag in threat_flags:
            st.caption(flag)


# ── Page ───────────────────────────────────────────────────────────────────
pipeline, cr_predictor = load_models()
db = load_db()
metrics = load_metrics()

with st.sidebar:
    st.title("🐉 True Lethality Engine")
    st.markdown(
        "The official CR system is a design *intention*. This engine is "
        "trained on **34,907 real combat encounters** from the FIREBALL "
        "dataset and predicts the *actual* win probability of a party — "
        "including monster **damage per round, attack bonus and save DC** "
        "parsed from real statblocks."
    )
    if metrics:
        st.divider()
        st.caption("Model card (held-out campaigns)")
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", f"{metrics.get('holdout_roc_auc', 0):.3f}")
        m2.metric("Brier", f"{metrics.get('holdout_brier', 0):.3f}")
        st.caption(
            f"Grouped CV AUC {metrics.get('cv_roc_auc_mean', 0):.3f} "
            f"± {metrics.get('cv_roc_auc_std', 0):.3f} · Platt-calibrated "
            f"probabilities · monotone-constrained XGBoost"
        )
    st.divider()
    target_win = st.slider(
        "🎯 Target win rate", 0.50, 0.90, TARGET_WIN_RATE, 0.05,
        help="The predicted win probability that defines a 'fair fight'. "
             "Note: real tables win 83% of resolved fights, so values near "
             "0.85 mean 'typical curated encounter' and 0.55 means "
             "'genuine coin flip'.",
    )
    st.caption("Statistical Machine Learning — Final Project")

if pipeline is None:
    st.stop()

st.title("⚔️ Fair Fight Finder")

with st.expander("📚 What do the party compositions mean?"):
    st.markdown("**The four roles** (derived from the classes in the combat logs):")
    st.dataframe(
        pd.DataFrame(
            [
                {"Role": role, "Classes": d["classes"], "Why it matters": d["meaning"]}
                for role, d in ROLE_DEFINITIONS.items()
            ]
        ),
        use_container_width=True, hide_index=True,
    )
    st.markdown("**The five simulated compositions** (which roles are present):")
    check = lambda v: "✅" if v else "—"
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Composition": c["name"],
                    "Healer": check(c["healer"]),
                    "Tank": check(c["tank"]),
                    "Arcane": check(c["arcane"]),
                    "Martial DPS": check(c["dps"]),
                    "Playstyle": c["desc"],
                }
                for c in PARTY_COMPOSITIONS
            ]
        ),
        use_container_width=True, hide_index=True,
    )

tab1, tab2, tab3 = st.tabs(
    ["📖 Official Monster", "🛠️ Homebrew Appraiser", "🧟 Encounter Builder"]
)

# ── Tab 1: official monster lookup ─────────────────────────────────────────
with tab1:
    monster_names = db["Name"].dropna().tolist() if not db.empty else []
    selected = st.selectbox(
        "Search the official bestiary",
        options=[""] + monster_names,
        help=f"{len(monster_names)} official monsters, offensive stats parsed "
             "from SRD statblocks where available.",
    )

    if selected:
        row = db[db["Name"] == selected].iloc[0]
        monster = profile_from_db_row(row)

        source = str(row.get("offense_source", "dmg_cr_table"))
        badge = ("🎯 real statblock" if source == "srd_statblock"
                 else "📐 DMG design table")
        st.success(
            f"**{monster.name}** — CR {monster.cr:g} · {int(monster.hp)} HP · "
            f"AC {int(monster.ac)} · DPR {monster.dpr:.0f} · "
            f"burst {monster.burst:.0f} · +{monster.atk_bonus:.0f} to hit · "
            f"DC {monster.save_dc:.0f} ({badge})"
        )

        n_official = st.number_input(
            "Number of monsters (action economy)", 1, 30, 1,
            key="n_official",
        )
        if st.button("Calculate True Lethality", type="primary",
                     key="btn_official"):
            render_results(pipeline, monster, int(n_official),
                           monster.cr, "📖 Official Monster Manual",
                           target=target_win)

# ── Tab 2: homebrew appraiser ──────────────────────────────────────────────
with tab2:
    st.markdown("Enter raw homebrew stats — the engine appraises the rest.")

    col1, col2, col3 = st.columns(3)
    with col1:
        hp = st.number_input("Hit points", 1, 2000, 100)
        ac = st.number_input("Armor class", 1, 30, 15)
        stat_sum = st.number_input(
            "Ability score sum", 10, 300, 150, step=5,
            help="Goblin ≈ 60 · Adult dragon ≈ 130 · God ≈ 180",
        )
        size_name = st.selectbox(
            "Size", ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"],
            index=2,
        )
    with col2:
        st.caption("Offense (leave empty = auto-estimate from CR)")
        # value=None instead of a 0-sentinel: an attack bonus of 0 is a legal
        # statblock value (CR 0 critters) and must be enterable.
        dpr_in = st.number_input(
            "Damage per round", 1, 400, value=None, placeholder="auto",
        )
        atk_in = st.number_input(
            "Attack bonus", 0, 20, value=None, placeholder="auto",
        )
        dc_in = st.number_input(
            "Spell/ability save DC", 1, 30, value=None, placeholder="auto",
        )
        burst_in = st.number_input(
            "Burst damage", 1, 400, value=None, placeholder="auto",
            help="The scariest single action: breath weapon, top damage "
                 "spell. Empty = assume equal to DPR.",
        )
        num_homebrew = st.number_input(
            "Number of monsters (action economy)", 1, 30, 1, key="n_homebrew",
        )
    with col3:
        is_leg = st.checkbox("Legendary (actions/resistances)")
        has_mob = st.checkbox("High mobility (fly/swim)")
        regen = st.checkbox("Regeneration")
        phys_res = st.checkbox("Nonmagical physical resistance")
        mag_res = st.checkbox("Magic resistance")
        cc_imm = st.checkbox("Immune to hard CC")
        spell = st.checkbox("Spellcaster")
        pack = st.checkbox("Pack tactics")

    if st.button("Appraise & Find Matches", type="primary", key="btn_custom"):
        size_map = {"Tiny": 1, "Small": 2, "Medium": 3, "Large": 4,
                    "Huge": 5, "Gargantuan": 6}

        # First appraisal: what would WotC math rate this?
        if cr_predictor is not None:
            baseline_cr = predict_wotc_cr(cr_predictor, {
                "hp": min(hp, 2000), "ac": max(10, min(ac, 30)),
                "stat_sum": max(60, min(stat_sum, 250)),
                "size_num": size_map[size_name],
                "is_legendary": int(is_leg), "has_mobility": int(has_mob),
                "physical_res": int(phys_res), "cc_immune": int(cc_imm),
                "magic_res": int(mag_res), "pack_tactics": int(pack),
                "spellcasting": int(spell), "regeneration": int(regen),
            })
            baseline_label = "🤖 Predicted WotC rating"
        else:
            baseline_cr = max(0.25, round((hp / 15) * 4) / 4)
            baseline_label = "HP heuristic baseline"

        monster = MonsterProfile(
            cr=baseline_cr, hp=hp, ac=ac,
            size_num=size_map[size_name], stat_sum=stat_sum,
            is_legendary=int(is_leg), has_mobility=int(has_mob),
            physical_res=int(phys_res), cc_immune=int(cc_imm),
            magic_res=int(mag_res), pack_tactics=int(pack),
            spellcasting=int(spell), regeneration=int(regen),
            atk_bonus=atk_in, dpr=dpr_in,
            save_dc=dc_in, burst=burst_in,
        )
        if dpr_in is None:
            est = offense_from_cr(baseline_cr)
            st.info(
                f"Offense auto-estimated from CR {baseline_cr:g}: "
                f"DPR {est[1]:.0f} · +{est[0]:.0f} to hit · DC {est[2]:.0f}. "
                "Enter real values above for a sharper appraisal."
            )
        render_results(pipeline, monster, int(num_homebrew),
                       baseline_cr, baseline_label, target=target_win)

# ── Tab 3: mixed-monster encounter builder ─────────────────────────────────
with tab3:
    st.markdown(
        "Mix **different monsters** — official, homebrew, or both — into a "
        "single encounter. Stats aggregate exactly like the training data "
        "(count-weighted averages, pooled damage, apex-threat maxima), and a "
        "dominance guard ensures adding monsters can never make the fight "
        "*easier*."
    )
    if "roster" not in st.session_state:
        st.session_state.roster = []

    col_off, col_home = st.columns(2)
    with col_off:
        st.subheader("📖 Add official monster")
        sel_roster = st.selectbox(
            "Monster", options=[""] + (db["Name"].dropna().tolist() if not db.empty else []),
            key="roster_official_sel",
        )
        cnt_official = st.number_input("Count", 1, 30, 1, key="roster_official_cnt")
        if st.button("➕ Add to encounter", key="roster_add_official"):
            if sel_roster:
                prof = profile_from_db_row(db[db["Name"] == sel_roster].iloc[0])
                st.session_state.roster.append(
                    {"profile": prof, "count": int(cnt_official)}
                )
            else:
                st.warning("Pick a monster first.")

    with col_home:
        st.subheader("🛠️ Add homebrew monster")
        with st.form("roster_homebrew_form"):
            hb_name = st.text_input("Name", "Custom Horror")
            f1, f2 = st.columns(2)
            with f1:
                hb_hp = st.number_input("HP", 1, 2000, 60)
                hb_ac = st.number_input("AC", 1, 30, 14)
                hb_stat = st.number_input("Ability sum", 10, 300, 120, step=5)
                hb_size = st.selectbox(
                    "Size", ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan"],
                    index=2,
                )
            with f2:
                hb_dpr = st.number_input(
                    "DPR", 1, 400, value=None, placeholder="auto")
                hb_atk = st.number_input(
                    "Atk bonus", 0, 20, value=None, placeholder="auto")
                hb_dc = st.number_input(
                    "Save DC", 1, 30, value=None, placeholder="auto")
                hb_burst = st.number_input(
                    "Burst dmg", 1, 400, value=None, placeholder="auto")
            hb_traits = st.multiselect(
                "Traits",
                ["Legendary", "High mobility", "Regeneration",
                 "Physical resistance", "Magic resistance", "CC immune",
                 "Spellcaster", "Pack tactics"],
            )
            hb_count = st.number_input("Count", 1, 30, 1)
            if st.form_submit_button("➕ Add to encounter"):
                size_map = {"Tiny": 1, "Small": 2, "Medium": 3, "Large": 4,
                            "Huge": 5, "Gargantuan": 6}
                t = set(hb_traits)
                if cr_predictor is not None:
                    hb_cr = predict_wotc_cr(cr_predictor, {
                        "hp": min(hb_hp, 2000), "ac": max(10, min(hb_ac, 30)),
                        "stat_sum": max(60, min(hb_stat, 250)),
                        "size_num": size_map[hb_size],
                        "is_legendary": int("Legendary" in t),
                        "has_mobility": int("High mobility" in t),
                        "physical_res": int("Physical resistance" in t),
                        "cc_immune": int("CC immune" in t),
                        "magic_res": int("Magic resistance" in t),
                        "pack_tactics": int("Pack tactics" in t),
                        "spellcasting": int("Spellcaster" in t),
                        "regeneration": int("Regeneration" in t),
                    })
                else:
                    hb_cr = max(0.25, round((hb_hp / 15) * 4) / 4)
                prof = MonsterProfile(
                    cr=hb_cr, hp=hb_hp, ac=hb_ac,
                    size_num=size_map[hb_size], stat_sum=hb_stat,
                    is_legendary=int("Legendary" in t),
                    has_mobility=int("High mobility" in t),
                    physical_res=int("Physical resistance" in t),
                    cc_immune=int("CC immune" in t),
                    magic_res=int("Magic resistance" in t),
                    pack_tactics=int("Pack tactics" in t),
                    spellcasting=int("Spellcaster" in t),
                    regeneration=int("Regeneration" in t),
                    dpr=hb_dpr, atk_bonus=hb_atk,
                    save_dc=hb_dc, burst=hb_burst,
                    name=hb_name or "Custom Horror",
                )
                st.session_state.roster.append(
                    {"profile": prof, "count": int(hb_count)}
                )

    # ── Current roster ────────────────────────────────────────────────────
    if st.session_state.roster:
        st.divider()
        st.subheader("Current encounter")
        for i, entry in enumerate(st.session_state.roster):
            p = entry["profile"]
            rc1, rc2 = st.columns([8, 1])
            rc1.markdown(
                f"**{entry['count']}× {p.name}** — CR {p.cr:g} · {p.hp:.0f} HP · "
                f"AC {p.ac:.0f} · DPR {p.dpr:.0f} · burst {p.burst:.0f} · "
                f"DC {p.save_dc:.0f}"
            )
            if rc2.button("🗑️", key=f"roster_del_{i}", help="Remove"):
                st.session_state.roster.pop(i)
                st.rerun()

        roster_pairs = [
            (e["profile"], e["count"]) for e in st.session_state.roster
        ]
        fields_preview = roster_monster_fields(roster_pairs)
        st.caption(
            f"Totals: {fields_preview['num_monsters_total']:.0f} monsters · "
            f"{fields_preview['total_monster_hp']:.0f} pooled HP · "
            f"{fields_preview['total_monster_dpr']:.0f} pooled DPR · "
            f"apex CR {fields_preview['max_monster_cr']:g} · "
            f"{fields_preview['total_monster_xp']:.0f} XP"
        )

        bc1, bc2 = st.columns([1, 1])
        if bc1.button("Calculate True Lethality", type="primary", key="btn_roster"):
            weighted_cr = fields_preview["avg_monster_cr"]
            render_results(
                pipeline, roster_pairs, 1,
                round(weighted_cr * 4) / 4, "📊 Roster avg CR (weighted)",
                target=target_win,
            )
        if bc2.button("Clear encounter", key="btn_roster_clear"):
            st.session_state.roster = []
            st.rerun()
    else:
        st.info("The encounter is empty — add monsters above.")

st.divider()
st.caption(
    "Probabilities are Platt-calibrated and OOD-hardened: inputs are "
    "clipped to legal 5e bands and the model is monotone-constrained, so a "
    "10,000 HP homebrew degrades gracefully instead of extrapolating."
)
