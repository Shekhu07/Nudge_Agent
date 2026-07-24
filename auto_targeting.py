"""Feature 1 — automated auto-nudge targeting layer.

Selects the cohort that should AUTOMATICALLY receive a "try a new category"
notification (vs. the operator picking a user by hand), then shapes the generated
nudge into a push-notification payload.

Eligibility rule (product decision): high-frequency, tenured customers —
order_frequency in {"Daily", "Weekly"} AND tenure > 6 months. These are habitual,
established users whose basket has settled, exactly the "stuck segment" the growth
goal targets.

This is a SIMULATION. No real push notification is sent — there is no live user
base or device. The layer is deterministic (no LLM); the nudge copy itself is
still generated live by the existing agent (agent.generate_nudge).
"""
import re

from friction_matching import load_profiles

MIN_TENURE_MONTHS = 6
ELIGIBLE_FREQS = {"daily", "weekly"}


def parse_tenure_months(tenure_str):
    """Parse the display tenure ("1y 4m", "11m", "2y 1m", "8m") into months."""
    if not tenure_str:
        return 0
    years = re.search(r"(\d+)\s*y", tenure_str)
    months = re.search(r"(\d+)\s*m", tenure_str)
    return (int(years.group(1)) * 12 if years else 0) + (int(months.group(1)) if months else 0)


def is_auto_eligible(profile, min_tenure_months=MIN_TENURE_MONTHS):
    """Daily or Weekly cadence AND tenure strictly greater than the minimum."""
    freq = (profile.get("order_frequency") or "").strip().lower()
    return freq in ELIGIBLE_FREQS and parse_tenure_months(profile.get("tenure")) > min_tenure_months


def eligibility_detail(profile, min_tenure_months=MIN_TENURE_MONTHS):
    """Return (eligible: bool, reason: str) explaining the decision for the UI."""
    freq = (profile.get("order_frequency") or "").strip()
    months = parse_tenure_months(profile.get("tenure"))
    freq_ok = freq.lower() in ELIGIBLE_FREQS
    tenure_ok = months > min_tenure_months
    if freq_ok and tenure_ok:
        return True, f"{freq} cadence · {months}mo tenure — auto-nudge cohort"
    misses = []
    if not freq_ok:
        misses.append(f"cadence is “{freq}”, not Daily/Weekly")
    if not tenure_ok:
        misses.append(f"tenure {months}mo ≤ {min_tenure_months}mo")
    return False, "Excluded: " + "; ".join(misses)


def eligible_profiles(profiles=None, min_tenure_months=MIN_TENURE_MONTHS):
    profiles = profiles if profiles is not None else load_profiles()
    return [p for p in profiles if is_auto_eligible(p, min_tenure_months)]


def to_notification(nudge):
    """Shape a generated nudge into a push-notification payload."""
    return {
        "emoji": nudge.get("emoji", "🛒"),
        "title": nudge.get("headline", ""),
        "body": nudge.get("body", ""),
        "category": nudge.get("suggested_category", ""),
    }


if __name__ == "__main__":
    for p in load_profiles():
        ok, why = eligibility_detail(p)
        mark = "✓ AUTO" if ok else "·  skip"
        print(f"{mark}  {p['user_id']:8} {p.get('order_frequency'):16} {p.get('tenure'):6} — {why}")
