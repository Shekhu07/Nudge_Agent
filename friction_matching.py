"""Friction-matching layer: maps a user profile's behavior pattern to the closest
Part 1/3 friction theme. Deterministic, rule-based scoring — no LLM here. The LLM
only writes the nudge copy downstream (agent.py), constrained to the theme this
layer selects.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Categories the agent may suggest, kept away from each profile's existing staples.
# (Lives here, in the deterministic layer, so the adjacency ranker below and the agent
# both compose from one list and cannot drift.)
SUGGESTABLE_CATEGORIES = [
    "personal care", "home & cleaning", "baby care", "pet supplies",
    "beauty & cosmetics", "packaged gourmet foods", "health & wellness",
    "kitchen & dining", "stationery & office",
]

# Adjacency weights: how strongly an EXISTING category pulls toward each suggestable one.
# This is the (B) fix — the agent was collapsing to "personal care" for almost every
# profile (first-item + safe-default bias), so category choice was effectively constant.
# Scoring a per-user best-fit shortlist here makes the choice reasoned and varied, and the
# agent picks from the ranked shortlist rather than a flat list it always reads top-down.
CATEGORY_ADJACENCY = {
    "groceries":     {"home & cleaning": 1, "kitchen & dining": 1, "health & wellness": 1, "personal care": 1},
    "fresh_produce": {"health & wellness": 2, "packaged gourmet foods": 1, "kitchen & dining": 1},
    "dairy":         {"health & wellness": 1, "baby care": 2, "packaged gourmet foods": 1},
    "snacks":        {"packaged gourmet foods": 2, "beauty & cosmetics": 1},
    "beverages":     {"packaged gourmet foods": 2, "health & wellness": 1},
    "household":     {"home & cleaning": 2, "kitchen & dining": 1, "stationery & office": 1, "pet supplies": 1},
    "personal_care": {"beauty & cosmetics": 2, "health & wellness": 1, "baby care": 1},
}


def _norm(cat):
    return cat.lower().strip().replace(" & ", "_").replace(" ", "_")


def rank_suggestable_categories(profile, top_n=4):
    """Return the top_n best-fit suggestable categories for a profile, ranked by
    adjacency to its existing basket and excluding any overlap. Deterministic:
    ties break alphabetically so the same basket always yields the same shortlist.
    """
    existing = {_norm(c) for c in profile.get("top_categories", [])}
    scores = {c: 0 for c in SUGGESTABLE_CATEGORIES}
    for ecat in existing:
        for scat, w in CATEGORY_ADJACENCY.get(ecat, {}).items():
            if scat in scores:
                scores[scat] += w
    ranked = sorted(SUGGESTABLE_CATEGORIES, key=lambda c: (-scores[c], c))
    ranked = [c for c in ranked if _norm(c) not in existing]
    return ranked[:top_n] or [c for c in SUGGESTABLE_CATEGORIES if _norm(c) not in existing]


def load_themes():
    with open(DATA_DIR / "friction_themes.json") as f:
        return json.load(f)["themes"]


def load_profiles():
    with open(DATA_DIR / "synthetic_profiles.json") as f:
        return json.load(f)["profiles"]


def _theme_by_id(themes, theme_id):
    return next(t for t in themes if t["id"] == theme_id)


def match_profile_to_theme(profile, themes):
    """Return (theme, match_reason) for the closest friction theme.

    Rules, in priority order:
    1. A packaging/fulfillment incident -> packaging_fulfillment theme.
    2. Any other unresolved-or-resolved quality incident -> poor_quality_unreliable.
    3. No incident at all -> low_intent_no_incident (out of the primary nudge's scope;
       flagged honestly rather than pretending the trust nudge fixes it).
    """
    incident = profile.get("recent_incident") or {}
    had_incident = incident.get("happened", False)
    incident_type = incident.get("type")

    if had_incident and incident_type == "packaging_crush":
        return (
            _theme_by_id(themes, "packaging_fulfillment"),
            "Fragile-item packaging failure, not source-quality — matched to the "
            "fulfillment sub-theme.",
        )

    if had_incident:
        resolved = incident.get("resolved")
        detail = "unresolved" if resolved is False else "resolved but trust-eroding"
        return (
            _theme_by_id(themes, "poor_quality_unreliable"),
            f"Prior {incident_type} incident in {incident.get('category')} "
            f"({detail}) — matched to the top quality/reliability theme.",
        )

    return (
        _theme_by_id(themes, "low_intent_no_incident"),
        "No incident on record — stagnation looks intent-driven, not trust-driven. "
        "Out of the primary nudge's scope; nudge stays soft and honest.",
    )


if __name__ == "__main__":
    themes = load_themes()
    for p in load_profiles():
        theme, reason = match_profile_to_theme(p, themes)
        print(f"{p['user_id']:8} {p['persona'][:45]:45} -> {theme['name']}")
        print(f"         {reason}\n")
