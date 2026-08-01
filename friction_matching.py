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
    "kitchen & dining", "books, toys & stationery", "electronics accessories",
    "pharmacy",
]

# Adjacency weights: how strongly an EXISTING category pulls toward each suggestable one.
# This is the (B) fix. Two failure modes were corrected here:
#   1. The agent collapsed to "personal care" (first-item + safe-default bias).
#   2. After that, it collapsed to "health & wellness"/"beauty & cosmetics" — a different
#      label but the same grooming-ish bucket, which still reads as "personal care items".
# So grocery/produce baskets deliberately pull toward DISTINCT, non-grooming categories
# (kitchen & dining, home & cleaning, packaged gourmet, pet supplies, baby care); the
# grooming categories only score when the basket actually signals them.
#
# "electronics accessories" (small/cheap items only — chargers, cables, power banks, not
# full electronics) only scores from "household", deliberately kept narrow rather than
# given a broad default weight: unlike the other suggestable categories, a refund/freshness
# guarantee is a weak trust lever for electronics — real e-commerce platforms build
# electronics trust via warranty/authenticity or pre-acceptance inspection instead, not a
# refund promise. So this category is paired with a separate inspect-before-accepting
# driver in agent.py rather than the standard two, and should surface only when
# basket-adjacency genuinely supports it, not as a frequent default.
#
# "pharmacy" is a KNOWN COMPROMISE, kept on record rather than hidden: real pharmacy trust
# is built through licensed-pharmacy sourcing and prescription verification, not a
# refund/freshness promise, and "no-questions-asked refund" doesn't meaningfully apply to
# medicine. Added anyway on explicit instruction, using the standard two-driver copy rather
# than a bespoke pharmacy-specific driver (no rule 7-style carve-out in agent.py). Scored
# narrowly from "personal_care" only.
#
# "books, toys & stationery" replaces the old "stationery & office" label with a broader,
# lower-stakes grouping (matches how it's commonly grouped in retail) — the standard two
# drivers fit fine here per research; no special handling needed.
import hashlib

CATEGORY_ADJACENCY = {
    "groceries":     {"kitchen & dining": 2, "home & cleaning": 1, "packaged gourmet foods": 1},
    "fresh_produce": {"kitchen & dining": 2, "packaged gourmet foods": 1, "health & wellness": 1},
    "dairy":         {"baby care": 2, "packaged gourmet foods": 1, "health & wellness": 1},
    "snacks":        {"packaged gourmet foods": 2, "kitchen & dining": 1},
    "beverages":     {"packaged gourmet foods": 2, "kitchen & dining": 1},
    "household":     {"home & cleaning": 2, "books, toys & stationery": 1, "pet supplies": 1, "kitchen & dining": 1,
                       "electronics accessories": 1},
    "personal_care": {"beauty & cosmetics": 2, "health & wellness": 1, "baby care": 1, "pharmacy": 1},
}


def _norm(cat):
    return cat.lower().strip().replace(" & ", "_").replace(" ", "_")


def adjacency_scores(profile):
    """Raw adjacency weight per suggestable category for this profile.

    These are the real, deterministic weights the ranker sums out of CATEGORY_ADJACENCY —
    small integers, NOT probabilities or confidence scores. The operator console renders
    them verbatim so the ranking it shows is the ranking the agent was actually given.
    """
    existing = {_norm(c) for c in profile.get("top_categories", [])}
    scores = {c: 0 for c in SUGGESTABLE_CATEGORIES}
    for ecat in existing:
        for scat, w in CATEGORY_ADJACENCY.get(ecat, {}).items():
            if scat in scores:
                scores[scat] += w
    return scores


def rank_suggestable_categories(profile, top_n=4, exclude=None):
    """Return the top_n best-fit suggestable categories for a profile, ranked by
    adjacency to its existing basket and excluding any overlap.

    `exclude` optionally removes one or more categories from the pool on top of the
    existing-basket exclusion — used to keep the push-nudge agent from repeating a
    category the shopper just trialed via the checkout cart-filler (the "Cart -> Nudge
    flow" feature, deployed standalone on Render — see render_cart_nudge/ in the main
    Blinkit_PRD repo, not in this clone): each mechanic should point at a DIFFERENT
    never-bought category, not the same one twice.

    Deterministic and stable per user_id, so results are reproducible.

    TIE-BREAK, fixed 2026-08-01: candidates are grouped into score tiers (highest
    adjacency first). Whole tiers that fit within top_n are taken entirely, in
    alphabetical order (order doesn't matter for a fully-included tier). The BOUNDARY
    tier — the one that only partially fits — is rotated by a per-user, per-tier seed
    before taking the remaining slots, rather than always taking its alphabetically-
    first members.

    Why this matters: the previous version sorted ALL candidates by (-score, name) and
    sliced to top_n in one step, so a boundary tie was always broken alphabetically.
    For every profile whose basket touched "household", categories tied at score 1 for
    the last slot (electronics accessories, books/toys & stationery, pet supplies, ...)
    always lost to whichever came first alphabetically — every single time, for every
    profile, forever, regardless of the per-click rotation feature in app.py (which
    only reorders whichever four categories already survived the cut). Verified:
    "electronics accessories" could never enter the candidate list for ANY of the 8
    synthetic profiles under the old tie-break, even though 3 of them had a genuine
    adjacency score > 0 for it.

    The fix makes this fair across ANY category caught in a boundary tie, not just
    electronics — a category with real (nonzero) adjacency now gets a real, reproducible
    per-user chance to be considered instead of losing purely to alphabetical order.
    Categories with genuinely zero adjacency for a profile are never promoted; this only
    changes who wins among candidates that already passed the relevance bar.
    """
    existing = {_norm(c) for c in profile.get("top_categories", [])}
    if exclude:
        excl = {exclude} if isinstance(exclude, str) else exclude
        existing |= {_norm(c) for c in excl}
    scores = adjacency_scores(profile)
    candidates = [c for c in SUGGESTABLE_CATEGORIES if _norm(c) not in existing]
    if not candidates:
        return []

    seed = int(hashlib.md5(str(profile.get("user_id", "")).encode()).hexdigest(), 16)

    tiers = {}
    for c in candidates:
        tiers.setdefault(scores[c], []).append(c)
    tier_scores = sorted(tiers, reverse=True)

    pool = []
    for i, sc in enumerate(tier_scores):
        tier = sorted(tiers[sc])
        if len(pool) + len(tier) <= top_n:
            pool.extend(tier)
            continue
        need = top_n - len(pool)
        if need <= 0:
            break
        # Rotate this boundary tier by a seed that varies per tier index, so it
        # doesn't collapse to the same offset as the final full-pool rotation below.
        tier_offset = (seed + i) % len(tier)
        rotated_tier = tier[tier_offset:] + tier[:tier_offset]
        pool.extend(rotated_tier[:need])
        break

    if not pool:
        pool = candidates[:]
    if len(pool) > 1:
        offset = seed % len(pool)
        pool = pool[offset:] + pool[:offset]
    return pool


def primary_suggested_category(profile):
    """The single category the push-nudge agent is instructed to prefer for this user
    (agent.py rule 3: "Prefer the highest-ranked candidate").

    This is the coordination anchor between the two nudge mechanics: the push queue
    (auto_targeting) and the checkout cart-filler (cart_filler) should point the same
    user at the same next category by default, rather than each running an independent
    ranker and risking a different pick. Deliberately reuses rank_suggestable_categories'
    own top-4 candidate list (the exact list the LLM sees) rather than adding a Groq call
    here — the cart-filler layer stays LLM-free and free of Groq quota cost, and the
    ranker's #1 is a reliable proxy since the agent is prompted to prefer it.
    """
    pool = rank_suggestable_categories(profile, top_n=4)
    return pool[0] if pool else None


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
