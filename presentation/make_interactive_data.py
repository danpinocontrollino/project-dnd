"""Precompute the model outputs embedded in the interactive slide deck.

Writes presentation/interactive_data.json and inlines the same JSON into
slides_interactive.html (Safari blocks fetch on file://, so the deck has to
carry its data inside). Contents:

  lich      P(win) for 1..19 liches x party level 1..20, computed through
            the real serving path (guards included) - the demo is not a mock
  guard     (TTK, p_sim) points from the Battlecast guard grid
  mercy     (p_sim, p_model, CR) triples from the mercy grid
  shap      top-10 features from figures/shap_ranking.csv
  guard_ab / guard_old   calibrated vs original guard constants

Run: make present-data
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "battlecast_bridge"))

import __main__  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from initial_learn import (  # noqa: E402
    DnDFeatureEngineer,
    FEATURE_COLUMNS,
    RAW_INPUT_COLUMNS,
)

__main__.DnDFeatureEngineer = DnDFeatureEngineer

from lethality_engine import (  # noqa: E402
    _BALANCED,
    _GUARD_A,
    _GUARD_B,
    _party_config,
    load_monster_database,
    predict_win_for_parties,
    profile_from_db_row,
)


def main() -> None:
    pipe = joblib.load("true_lethality_model.pkl")
    db = load_monster_database()
    lich = profile_from_db_row(db[db["Name"].str.lower() == "lich"].iloc[0])

    # 1. Lich Lab grid — production path, guards included.
    grid = {}
    for n in range(1, 20):
        cfgs = [_party_config(l, 4, _BALANCED) for l in range(1, 21)]
        p = predict_win_for_parties(pipe, lich, cfgs, n)
        grid[str(n)] = [round(float(x), 4) for x in p]

    # 2 + 3. Battlecast guard scatter and mercy pairs.
    from analyze import load_results, model_rows

    df = load_results("battlecast_bridge/results.jsonl")
    raw = model_rows(df)
    feats = DnDFeatureEngineer(1.5, FEATURE_COLUMNS).transform(raw)
    df["ttk"] = feats["rounds_to_kill_party"].values
    df["p_model"] = pipe.predict_proba(raw[list(RAW_INPUT_COLUMNS)])[:, 1]

    g = df[df.grid == "guard"]
    guard_pts = [
        [round(float(t), 2), round(float(p), 3)]
        for t, p in zip(g.ttk, g.p_party_win)
    ]
    m = df[df.grid == "mercy"]
    mercy_pts = [
        [round(float(a), 3), round(float(b), 3), float(c)]
        for a, b, c in zip(m.p_party_win, m.p_model, m.monster_cr)
    ]

    # 4. SHAP top 10.
    shap = pd.read_csv("figures/shap_ranking.csv", index_col=0).iloc[:, 0].head(10)
    shap_data = [[k, round(float(v), 3)] for k, v in shap.items()]

    out = {
        "lich": grid,
        "guard": guard_pts,
        "mercy": mercy_pts,
        "shap": shap_data,
        "guard_ab": [round(_GUARD_A, 4), round(_GUARD_B, 4)],
        "guard_old": [2.197, -4.394],
    }
    path = os.path.join(REPO, "presentation", "interactive_data.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path):,} bytes)")

    # Inline the same JSON into the deck so a plain double-click works in
    # Safari (fetch() is blocked on file:// URLs).
    import re
    html_path = os.path.join(REPO, "presentation", "slides_interactive.html")
    html = open(html_path, encoding="utf-8").read()
    payload = json.dumps(out, separators=(",", ":"))
    html, n = re.subn(
        r'(<script type="application/json" id="interactiveData">).*?(</script>)',
        lambda m: m.group(1) + "\n" + payload + "\n" + m.group(2),
        html, flags=re.S,
    )
    open(html_path, "w", encoding="utf-8").write(html)
    print(f"inlined into slides_interactive.html ({n} block)" )
    print("lich n=1 :", grid["1"][:3], "->", grid["1"][-1])
    print("lich n=19:", grid["19"][:3], "->", grid["19"][-1])


if __name__ == "__main__":
    main()
