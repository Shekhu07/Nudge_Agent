"""Feature 2 — checkout cart-threshold "filler" micro-trials (playbook Pillar 4).

When a user's cart is short of the free-delivery threshold, surface a low-cost item
from a category they have NEVER bought, turning the delivery-fee top-up into a
zero-friction trial of a new category. This targets the low-intent segment the push
nudge deliberately skips: a ₹60 add needs no pre-existing intent.

Deterministic — no LLM (the selection is mechanical and the copy is templated, so it
costs no Groq quota). The FILLER_CATALOG below is SYNTHETIC demo data, not a real Blinkit
catalog; prices are illustrative.
"""
from friction_matching import (SUGGESTABLE_CATEGORIES, _norm, primary_suggested_category,
                               rank_suggestable_categories)

# Free-delivery threshold (illustrative). Real Blinkit thresholds vary by city/time.
FREE_DELIVERY_THRESHOLD = 199

# Delivery fee waived when the threshold is met (illustrative, like the threshold and the
# catalog prices). Surfaced in the UI as the "delivery saved" figure — it is a stated demo
# constant, not a measured or claimed Blinkit fee.
DELIVERY_FEE = 35

# SYNTHETIC low-cost filler items per suggestable category. Prices are illustrative.
FILLER_CATALOG = {
    "home & cleaning":        [{"name": "Stainless Steel Cleaner", "price": 65},
                               {"name": "Microfiber Cloth (3 pack)", "price": 55}],
    "stationery & office":    [{"name": "Sticky Notes (400 sheets)", "price": 60},
                               {"name": "Gel Pen Set (5)", "price": 45}],
    "kitchen & dining":       [{"name": "Silicone Spatula", "price": 70},
                               {"name": "Microfibre Kitchen Towel", "price": 50}],
    "packaged gourmet foods": [{"name": "Artisan Cracker Pack", "price": 75},
                               {"name": "Single-Origin Coffee Sachets", "price": 60}],
    "health & wellness":      [{"name": "Vitamin C Effervescent Tube", "price": 80},
                               {"name": "Electrolyte Sachets (5)", "price": 55}],
    "beauty & cosmetics":     [{"name": "Lip Balm Duo", "price": 65},
                               {"name": "Sheet Mask (2 pack)", "price": 50}],
    "personal care":          [{"name": "Travel Face Wash", "price": 60},
                               {"name": "Bamboo Cotton Buds", "price": 45}],
    "pet supplies":           [{"name": "Dog Biscuit Snack Pack", "price": 70},
                               {"name": "Catnip Toy", "price": 60}],
    "baby care":              [{"name": "Baby Wipes (72s)", "price": 75},
                               {"name": "Baby Lotion Mini", "price": 65}],
}


def never_bought_categories(profile):
    """Suggestable categories the user has never purchased, ranked by basket adjacency
    so the offered fillers still feel relevant rather than random."""
    existing = {_norm(c) for c in profile.get("top_categories", [])}
    ranked = rank_suggestable_categories(profile, top_n=len(SUGGESTABLE_CATEGORIES))
    return [c for c in ranked if _norm(c) not in existing and c in FILLER_CATALOG]


def suggest_fillers(profile, cart_total, threshold=FREE_DELIVERY_THRESHOLD, max_items=3):
    """Return the cart-filler carousel for a given cart total.

    Prefers items priced at or above the gap (a single add unlocks free delivery),
    falling back to the cheapest never-bought-category items if none individually
    covers it. Every item comes from a category the user has never bought.

    Within each of those two tiers, the category `primary_suggested_category()` picked
    for this user — the same category the push-nudge agent is instructed to prefer —
    sorts first. This is the cross-mechanic coordination: absent a reason to do otherwise
    (i.e. it doesn't cover the gap as well as another option), the checkout filler
    reinforces the same next-category the push queue already nudges toward, rather than
    two independent rankers potentially pointing the same user at two different
    categories in the same week.
    """
    gap = threshold - cart_total
    if gap <= 0:
        return {"qualifies": True, "gap": 0, "threshold": threshold,
                "cart_total": cart_total, "items": []}

    anchor = primary_suggested_category(profile)
    candidates = []
    for cat in never_bought_categories(profile):
        for item in FILLER_CATALOG[cat]:
            candidates.append({**item, "category": cat, "covers_gap": item["price"] >= gap,
                                "is_anchor": cat == anchor})

    # Items that unlock free delivery in one add first (cheapest such first), then the
    # rest by price — keeps the offered add-on as close to the gap as possible. The
    # anchor category sorts to the front of whichever tier it lands in.
    covering = sorted([c for c in candidates if c["covers_gap"]],
                       key=lambda c: (not c["is_anchor"], c["price"]))
    non_covering = sorted([c for c in candidates if not c["covers_gap"]],
                          key=lambda c: (not c["is_anchor"], -c["price"]))
    ordered = covering + non_covering

    # One filler per category so the carousel spans distinct new categories.
    seen, items = set(), []
    for c in ordered:
        if c["category"] in seen:
            continue
        seen.add(c["category"])
        items.append(c)
        if len(items) >= max_items:
            break

    return {"qualifies": False, "gap": gap, "threshold": threshold,
            "cart_total": cart_total, "items": items}


if __name__ == "__main__":
    from friction_matching import load_profiles
    for p in load_profiles()[:4]:
        res = suggest_fillers(p, cart_total=150)
        offers = ", ".join(f"{i['name']} ₹{i['price']} [{i['category']}]" for i in res["items"])
        print(f"{p['user_id']}  gap ₹{res['gap']}  ->  {offers}")
