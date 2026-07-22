"""Friction-matching layer: maps a user profile's behavior pattern to the closest
Part 1/3 friction theme. Deterministic, rule-based scoring — no LLM here. The LLM
only writes the nudge copy downstream (agent.py), constrained to the theme this
layer selects.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


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
