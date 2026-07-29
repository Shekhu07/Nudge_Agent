"""Unit tests for the deterministic friction-matching layer (no LLM, no network)."""
import pytest

from friction_matching import (SUGGESTABLE_CATEGORIES, adjacency_scores, load_profiles,
                                load_themes, match_profile_to_theme,
                                primary_suggested_category, rank_suggestable_categories)


def _profile(user_id="U1", top_categories=None, recent_incident=None):
    return {
        "user_id": user_id,
        "top_categories": top_categories or [],
        "recent_incident": recent_incident,
    }


@pytest.fixture(scope="module")
def themes():
    return load_themes()


# --------------------------- match_profile_to_theme ---------------------------

def test_packaging_crush_routes_to_fulfillment_theme(themes):
    p = _profile(recent_incident={"happened": True, "type": "packaging_crush",
                                   "category": "stationery", "resolved": True})
    theme, reason = match_profile_to_theme(p, themes)
    assert theme["id"] == "packaging_fulfillment"
    assert "fulfillment" in reason.lower()


def test_unresolved_quality_incident_routes_to_top_theme(themes):
    p = _profile(recent_incident={"happened": True, "type": "expired",
                                   "category": "dairy", "resolved": False})
    theme, reason = match_profile_to_theme(p, themes)
    assert theme["id"] == "poor_quality_unreliable"
    assert "unresolved" in reason.lower()


def test_resolved_quality_incident_still_routes_to_top_theme(themes):
    p = _profile(recent_incident={"happened": True, "type": "damaged",
                                   "category": "electronics", "resolved": True})
    theme, reason = match_profile_to_theme(p, themes)
    assert theme["id"] == "poor_quality_unreliable"
    assert "resolved" in reason.lower()


def test_no_incident_routes_to_out_of_scope_theme(themes):
    p = _profile(recent_incident={"happened": False, "type": None,
                                   "category": None, "resolved": None})
    theme, reason = match_profile_to_theme(p, themes)
    assert theme["id"] == "low_intent_no_incident"
    assert theme.get("out_of_primary_scope") is True


def test_missing_incident_key_treated_as_no_incident(themes):
    p = _profile(recent_incident=None)
    theme, _ = match_profile_to_theme(p, themes)
    assert theme["id"] == "low_intent_no_incident"


# --------------------------- rank_suggestable_categories ---------------------------

def test_ranking_excludes_existing_categories():
    # "personal care" and "pet supplies" are themselves suggestable-category names, so
    # putting them in top_categories directly exercises the overlap-exclusion path.
    p = _profile(top_categories=["personal care", "pet supplies"])
    pool = rank_suggestable_categories(p)
    assert "personal care" not in pool
    assert "pet supplies" not in pool


def test_ranking_is_deterministic_across_calls():
    p = _profile(user_id="SYN-042", top_categories=["groceries", "dairy"])
    first = rank_suggestable_categories(p)
    for _ in range(5):
        assert rank_suggestable_categories(p) == first


def test_ranking_returns_subset_of_suggestable_categories():
    p = _profile(top_categories=["groceries"])
    pool = rank_suggestable_categories(p)
    assert set(pool).issubset(set(SUGGESTABLE_CATEGORIES))
    assert len(pool) > 0


# --------------------------- primary_suggested_category ---------------------------

def test_primary_suggested_category_matches_top4_ranker_pick():
    p = _profile(user_id="SYN-777", top_categories=["groceries", "dairy"])
    assert primary_suggested_category(p) == rank_suggestable_categories(p, top_n=4)[0]


def test_primary_suggested_category_none_when_everything_already_owned():
    p = _profile(top_categories=list(SUGGESTABLE_CATEGORIES))
    assert primary_suggested_category(p) is None


def test_primary_suggested_category_stable_across_calls():
    p = _profile(user_id="SYN-321", top_categories=["household"])
    first = primary_suggested_category(p)
    for _ in range(5):
        assert primary_suggested_category(p) == first


def test_ranking_falls_back_when_all_pass_categories_already_owned():
    # A user who somehow already "owns" every suggestable category should still get a
    # non-empty pool back (falls back to all suggestable categories minus overlap).
    p = _profile(top_categories=list(SUGGESTABLE_CATEGORIES))
    pool = rank_suggestable_categories(p)
    assert pool == []  # every suggestable category is excluded as already-owned


# --------------------------- adjacency_scores ---------------------------

def test_groceries_pulls_toward_distinct_non_grooming_categories():
    p = _profile(top_categories=["groceries"])
    scores = adjacency_scores(p)
    assert scores["kitchen & dining"] > 0
    assert scores["home & cleaning"] > 0
    # groceries should not directly pull toward grooming categories
    assert scores["beauty & cosmetics"] == 0


def test_unknown_existing_category_contributes_no_score():
    p = _profile(top_categories=["some_never_before_seen_category"])
    scores = adjacency_scores(p)
    assert all(v == 0 for v in scores.values())


def test_empty_basket_yields_all_zero_scores():
    p = _profile(top_categories=[])
    scores = adjacency_scores(p)
    assert set(scores) == set(SUGGESTABLE_CATEGORIES)
    assert all(v == 0 for v in scores.values())


# --------------------------- data-file smoke tests ---------------------------

def test_load_themes_has_expected_ids():
    ids = {t["id"] for t in load_themes()}
    assert {"poor_quality_unreliable", "packaging_fulfillment", "low_intent_no_incident"} <= ids


def test_load_profiles_returns_nonempty_list_with_required_fields():
    profiles = load_profiles()
    assert len(profiles) > 0
    for p in profiles:
        assert "user_id" in p
        assert "top_categories" in p
        assert "order_frequency" in p


def test_every_real_profile_matches_some_theme():
    themes = load_themes()
    for p in load_profiles():
        theme, reason = match_profile_to_theme(p, themes)
        assert theme is not None
        assert reason
