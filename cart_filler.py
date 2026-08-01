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

# SYNTHETIC staple items for the categories a user ALREADY buys — used to render the
# checkout cart as real line items instead of the placeholder "Groceries / usual item"
# rows the screen used to show.
#
# Why adding this is consistent rather than a new liberty: the cart deliberately avoided
# naming products on the grounds that it would invent data — but FILLER_CATALOG below
# already invents names and prices for 11 categories and labels them illustrative. The
# honesty posture was already set; the cart lines just weren't using it. Same rule here:
# demo items, labelled as such in the UI, never presented as real Blinkit catalogue.
#
# Keyed on the profile's `buys_display` values (title-cased), not the normalised
# `top_categories` slugs.
STAPLES_CATALOG = {
    "Groceries":     [{"name": "Whole Wheat Atta (5kg)", "price": 62}],
    "Fresh Produce": [{"name": "Tomatoes (500g)", "price": 24}],
    "Dairy":         [{"name": "Toned Milk (500ml)", "price": 33}],
    "Snacks":        [{"name": "Salted Chips (Large)", "price": 30}],
    "Beverages":     [{"name": "Cola (750ml)", "price": 40}],
    "Household":     [{"name": "Dishwash Liquid (500ml)", "price": 58}],
    "Personal Care": [{"name": "Shower Gel (250ml)", "price": 68}],
}


def cart_lines(profile, cart_total, max_lines=3):
    """Build checkout line items that reconcile exactly to a given cart total.

    The cart total is operator-controlled (a slider), so the line items cannot be a fixed
    list — they have to add up to whatever total is set, or the bill shows an "Item total"
    that none of the visible rows explain. That mismatch was visible on the old screen:
    three unpriced rows sitting above a ₹150 subtotal that came from nowhere.

    Fills from the user's own habitual categories in order, in whole units, and returns
    any unallocated balance as `remainder` so the caller can render a final "other items"
    row. Guarantees: sum(line subtotals) + remainder == cart_total, and remainder >= 0.

    Deterministic — no LLM, no randomness.
    """
    cart_total = max(0, int(cart_total))
    lines, spent = [], 0
    cats = [c for c in (profile.get("buys_display") or [])[:max_lines]
            if c in STAPLES_CATALOG]
    for i, cat in enumerate(cats):
        item = STAPLES_CATALOG[cat][0]
        left = cart_total - spent
        if left < item["price"]:
            continue
        # Reserve rough room for the categories still to come so one staple can't eat
        # the whole cart and leave the rest of the basket invisible.
        share = left if i == len(cats) - 1 else left // (len(cats) - i)
        qty = max(1, min(3, share // item["price"]))
        if item["price"] * qty > left:
            qty = left // item["price"]
        if qty < 1:
            continue
        lines.append({**item, "category": cat, "qty": qty,
                      "subtotal": item["price"] * qty})
        spent += item["price"] * qty
    return {"lines": lines, "remainder": cart_total - spent, "cart_total": cart_total}


# SYNTHETIC low-cost filler items per suggestable category. Prices are illustrative.
FILLER_CATALOG = {
    "home & cleaning":        [{"name": "Stainless Steel Cleaner", "price": 65},
                               {"name": "Microfiber Cloth (3 pack)", "price": 55}],
    "books, toys & stationery": [{"name": "Sticky Notes (400 sheets)", "price": 60},
                               {"name": "Mini Puzzle Toy", "price": 70}],
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
    # Deliberately non-prescription, generic OTC/hygiene items only — see the "pharmacy is a
    # KNOWN COMPROMISE" note in friction_matching.py. No medicine-like items in a demo catalog.
    "pharmacy":               [{"name": "Adhesive Bandages Pack", "price": 45},
                               {"name": "Hand Sanitizer (50ml)", "price": 55}],
    # Small/cheap accessories only, matching the "no full electronics" note in
    # friction_matching.py — the filler mechanic needs sub-₹80 items, which full electronics
    # (power banks, chargers) rarely are.
    "electronics accessories": [{"name": "USB-C Cable (1m)", "price": 79},
                               {"name": "Phone Grip Stand", "price": 65}],
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
