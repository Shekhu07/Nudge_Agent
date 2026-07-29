"""Unit tests for the checkout cart-filler layer (deterministic, no LLM)."""
from cart_filler import (FILLER_CATALOG, FREE_DELIVERY_THRESHOLD, never_bought_categories,
                          suggest_fillers)
from friction_matching import primary_suggested_category


def _profile(user_id="U1", top_categories=None):
    return {"user_id": user_id, "top_categories": top_categories or []}


# --------------------------- never_bought_categories ---------------------------

def test_never_bought_excludes_existing_categories():
    p = _profile(top_categories=["personal care", "pet supplies"])
    result = never_bought_categories(p)
    assert "personal care" not in result
    assert "pet supplies" not in result


def test_never_bought_only_includes_categories_in_catalog():
    p = _profile(top_categories=[])
    result = never_bought_categories(p)
    assert set(result).issubset(set(FILLER_CATALOG))


# --------------------------- suggest_fillers: threshold logic ---------------------------

def test_qualifies_when_cart_meets_threshold():
    p = _profile()
    res = suggest_fillers(p, cart_total=FREE_DELIVERY_THRESHOLD)
    assert res["qualifies"] is True
    assert res["gap"] == 0
    assert res["items"] == []


def test_qualifies_when_cart_exceeds_threshold():
    p = _profile()
    res = suggest_fillers(p, cart_total=FREE_DELIVERY_THRESHOLD + 50)
    assert res["qualifies"] is True
    assert res["items"] == []


def test_gap_computed_correctly_when_under_threshold():
    p = _profile()
    res = suggest_fillers(p, cart_total=150, threshold=FREE_DELIVERY_THRESHOLD)
    assert res["qualifies"] is False
    assert res["gap"] == FREE_DELIVERY_THRESHOLD - 150


# --------------------------- suggest_fillers: item selection ---------------------------

def test_items_come_from_never_bought_categories_only():
    p = _profile(top_categories=["personal care"])
    res = suggest_fillers(p, cart_total=150)
    categories_offered = {it["category"] for it in res["items"]}
    assert "personal care" not in categories_offered


def test_one_item_per_category():
    p = _profile(top_categories=[])
    res = suggest_fillers(p, cart_total=100, max_items=5)
    categories = [it["category"] for it in res["items"]]
    assert len(categories) == len(set(categories))


def test_max_items_is_respected():
    p = _profile(top_categories=[])
    res = suggest_fillers(p, cart_total=100, max_items=2)
    assert len(res["items"]) <= 2


def test_covering_items_are_prioritized_over_non_covering():
    p = _profile(top_categories=[])
    res = suggest_fillers(p, cart_total=150, max_items=len(FILLER_CATALOG))
    gap = res["gap"]
    covering_flags = [it["covers_gap"] for it in res["items"]]
    # covers_gap correctness: matches price >= gap
    for it in res["items"]:
        assert it["covers_gap"] == (it["price"] >= gap)
    # all covering items must come before any non-covering item
    if True in covering_flags and False in covering_flags:
        assert covering_flags.index(True) < covering_flags.index(False)


def test_no_never_bought_categories_left_returns_empty_items():
    # A user who has "bought" every suggestable category has nothing left to offer.
    p = _profile(top_categories=list(FILLER_CATALOG.keys()))
    res = suggest_fillers(p, cart_total=100)
    assert res["items"] == []
    assert res["qualifies"] is False


# --------------------------- push-nudge / filler coordination ---------------------------

def test_is_anchor_only_true_for_the_anchor_category():
    p = _profile(user_id="SYN-555", top_categories=["groceries", "dairy"])
    anchor = primary_suggested_category(p)
    res = suggest_fillers(p, cart_total=150, max_items=len(FILLER_CATALOG))
    for it in res["items"]:
        assert it["is_anchor"] == (it["category"] == anchor)


def test_anchor_category_sorts_first_within_its_tier():
    p = _profile(user_id="SYN-901", top_categories=["household"])
    anchor = primary_suggested_category(p)
    res = suggest_fillers(p, cart_total=150, max_items=len(FILLER_CATALOG))
    items = res["items"]
    anchor_positions = [i for i, it in enumerate(items) if it["category"] == anchor]
    if not anchor_positions:
        return  # anchor category didn't survive to the offered set; nothing to assert
    pos = anchor_positions[0]
    tier_positions = [i for i, it in enumerate(items) if it["covers_gap"] == items[pos]["covers_gap"]]
    assert pos == min(tier_positions)


def test_is_anchor_false_for_everyone_when_no_candidates_left():
    p = _profile(top_categories=list(FILLER_CATALOG.keys()))
    res = suggest_fillers(p, cart_total=100)
    assert all(not it["is_anchor"] for it in res["items"])
