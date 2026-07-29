"""Unit tests for the checkout cart-filler layer (deterministic, no LLM)."""
from cart_filler import (FILLER_CATALOG, FREE_DELIVERY_THRESHOLD, never_bought_categories,
                          suggest_fillers)


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


def test_covering_items_are_prioritized_and_cheapest_first():
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
