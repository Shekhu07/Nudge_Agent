"""Unit tests for the auto-nudge eligibility gate (deterministic, no LLM)."""
import pytest

from auto_targeting import (eligibility_detail, eligible_profiles, is_auto_eligible,
                             parse_tenure_months, to_notification)


def _profile(order_frequency="Daily", tenure="1y 0m"):
    return {"order_frequency": order_frequency, "tenure": tenure}


# --------------------------- parse_tenure_months ---------------------------

@pytest.mark.parametrize("tenure_str,expected", [
    ("1y 4m", 16),
    ("2y 1m", 25),
    ("11m", 11),
    ("8m", 8),
    ("3y", 36),
    ("", 0),
    (None, 0),
])
def test_parse_tenure_months(tenure_str, expected):
    assert parse_tenure_months(tenure_str) == expected


# --------------------------- is_auto_eligible ---------------------------

@pytest.mark.parametrize("freq,tenure,expected", [
    ("Daily", "1y 0m", True),
    ("Weekly", "7m", True),
    ("daily", "7m", True),        # case-insensitive
    ("2-3 times a week", "2y 0m", False),  # cadence not Daily/Weekly
    ("Daily", "6m", False),        # tenure must be STRICTLY greater than 6mo
    ("Daily", "0m", False),
])
def test_is_auto_eligible(freq, tenure, expected):
    assert is_auto_eligible(_profile(freq, tenure)) is expected


def test_is_auto_eligible_respects_custom_min_tenure():
    p = _profile("Weekly", "8m")
    assert is_auto_eligible(p, min_tenure_months=6) is True
    assert is_auto_eligible(p, min_tenure_months=12) is False


# --------------------------- eligibility_detail ---------------------------

def test_eligibility_detail_eligible_reason_mentions_cadence_and_tenure():
    ok, why = eligibility_detail(_profile("Daily", "1y 0m"))
    assert ok is True
    assert "Daily" in why
    assert "12mo" in why


def test_eligibility_detail_excluded_reason_lists_all_failing_conditions():
    ok, why = eligibility_detail(_profile("Monthly", "2m"))
    assert ok is False
    assert "Excluded" in why
    assert "cadence" in why.lower()
    assert "tenure" in why.lower()


# --------------------------- eligible_profiles ---------------------------

def test_eligible_profiles_filters_correctly():
    profiles = [
        _profile("Daily", "1y 0m"),      # eligible
        _profile("Monthly", "2y 0m"),    # wrong cadence
        _profile("Weekly", "2m"),        # too new
    ]
    result = eligible_profiles(profiles)
    assert result == [profiles[0]]


def test_eligible_profiles_uses_real_data_by_default():
    # Smoke test against the real synthetic cohort — should not error and should
    # return a strict subset of the full profile list.
    from friction_matching import load_profiles
    all_profiles = load_profiles()
    result = eligible_profiles()
    assert len(result) <= len(all_profiles)
    assert all(is_auto_eligible(p) for p in result)


# --------------------------- to_notification ---------------------------

def test_to_notification_shapes_expected_fields():
    nudge = {"emoji": "🧴", "headline": "Try this", "body": "Body text",
              "suggested_category": "personal care", "extra_field": "ignored"}
    notif = to_notification(nudge)
    assert notif == {"emoji": "🧴", "title": "Try this", "body": "Body text",
                      "category": "personal care"}


def test_to_notification_defaults_missing_fields():
    notif = to_notification({})
    assert notif["emoji"] == "🛒"
    assert notif["title"] == ""
    assert notif["body"] == ""
    assert notif["category"] == ""
