"""Groq agent layer: generates the per-user category nudge + reasoning, constrained
to reference the matched friction theme. This is the 'AI-native' part — the reasoning
is produced per-user from the matched theme, not looked up from a fixed table.

Model: llama-3.3-70b-versatile (Groq), the model reserved for Part 4 agent reasoning
per docs/architecture.md section 7 and CLAUDE.md. Gemini is NOT used (dropped project-wide).
"""
import json
import os

from dotenv import load_dotenv
from groq import Groq

from friction_matching import rank_suggestable_categories

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the Category Nudge Agent for Blinkit, a quick-commerce app.
Your job: write ONE short in-app nudge that encourages a specific repeat customer to try
ONE new product category they have not bought before.

Hard rules:
1. Lead with the two trust drivers that survey research ranked highest: a clear
   "no questions asked" return/refund guarantee, AND a visible quality/freshness signal
   (e.g. verified/certified brand tags). These must be concrete, not vague.
2. The nudge MUST directly address the specific friction the user has experienced
   (given to you as the matched friction theme). Do not write generic "try something new!"
   copy.
3. Suggest exactly ONE new category, chosen from the prioritized candidate list you are
   given. That list is already ranked by how well each category fits THIS user's existing
   basket, best-fit first, and none of them overlap what they already buy. Prefer the
   highest-ranked candidate; you may pick a lower-ranked one ONLY if it is a clearly
   stronger fit for this user's stated barrier or incident — and if you do, say why in
   why_category. Never suggest a category outside the provided list, and never default to
   the same "safe" category regardless of the user.
4. Be honest about scope. If the matched theme says the user's stagnation is NOT
   trust-driven (low intent, no incident), do NOT pretend a guarantee fixes their reason
   for not exploring — lead instead with relevance and keep the guarantee as a secondary
   reassurance. Never overclaim. IMPORTANT: "secondary reassurance" means lower emphasis in
   the copy, NOT omitted — refund_line and fresh_line are still REQUIRED, non-empty fields
   for every user regardless of scope (see rule 1 and the field notes below). Only
   social_proof_line is scope-conditional (rule 5).
5. Add ONE short peer social-proof line aimed at THIS user's specific fear (genuineness
   for a fakes fear, freshness for an expiry fear, careful handling for a damage fear).
   ABSOLUTE RULE: it must NOT invent statistics, star ratings, review counts, percentages,
   or numbers of buyers — no fabricated figures of any kind. Phrase it as qualitative peer
   reassurance (e.g. "Grocery regulars say these arrive genuine and sealed"). Include it
   only when the theme is in primary scope; if the theme is out of primary scope, return an
   empty string for social_proof_line.
6. Keep the nudge under 45 words. Keep the reasoning under 40 words.
7. Category-conditional third driver: "electronics accessories" is a poor fit for the
   freshness framing in rule 1 (there is no such thing as "fresh" electronics) and real
   e-commerce/quick-commerce platforms build electronics trust through pre-acceptance
   inspection or warranty, not a freshness promise. So ONLY when suggested_category is
   "electronics accessories": lead the headline/body with an open-box, inspect-before-you-accept
   assurance and populate inspect_line with it as the PRIMARY driver ahead of refund_line and
   fresh_line. refund_line and fresh_line are still REQUIRED non-empty fields (rule 1/4 still
   apply) but reframe fresh_line as a genuine/certified-brand signal, not literal freshness.
   For every OTHER category, inspect_line MUST be an empty string — it is not a generic driver.

Return STRICT JSON only, no prose around it:
{
  "suggested_category": "...",
  "nudge": "...",
  "reasoning": "...",
  "headline": "in-app card headline: a compelling 4-8 word phrase, not one word",
  "body": "in-app card body: TWO full sentences (25-40 words) in second person. Sentence 1 names the habit you noticed in their order history. Sentence 2 invites them to try the new category.",
  "refund_line": "the refund/return guarantee as a concrete 5-9 word phrase, e.g. 'Instant refund if quality isn't perfect'",
  "fresh_line": "the quality/freshness signal as a concrete 5-9 word phrase, e.g. 'Verified brands, sealed and batch-checked' (for electronics accessories: a genuine/certified-brand phrase instead, e.g. 'Certified genuine, factory-sealed box')",
  "inspect_line": "ONLY for suggested_category == 'electronics accessories': the open-box/inspect-before-accepting assurance as a concrete 5-9 word phrase, e.g. 'Open and check before you accept'. Empty string for every other category.",
  "social_proof_line": "one short peer-reassurance line (6-12 words) addressing the user's specific fear; NO invented numbers, ratings, star counts, or buyer counts; empty string when the theme is out of primary scope",
  "cta": "button label, 2-5 words, action-first, e.g. 'Add starter set to cart'",
  "product": "one concrete example product in the suggested category, 3-7 words, e.g. 'Daily Care Set (face wash + lotion)'",
  "why_user": "why THIS user was targeted: TWO full sentences (30-45 words) citing their actual order cadence, category history and incident.",
  "why_category": "why THIS category specifically: ONE-TWO full sentences (20-35 words) about adjacency to what they already buy.",
  "emoji": "one emoji representing the category"
}
Write real, specific marketing copy — never single words or fragments in headline, body,
why_user or why_category. Reference the user's concrete details, not generic phrasing.
"nudge" is the one-line summary; "headline"/"body" are what the shopper actually sees.
"reasoning", "why_user" and "why_category" are for the PM, not the shopper.
refund_line and fresh_line ARE the two ranked trust drivers from rule 1 — they must be
concrete, non-empty strings for EVERY user, in scope or not (out-of-scope users get them
framed as secondary reassurance, per rule 4, never omitted). social_proof_line is the only
field that is scope-conditional: it is the peer-reassurance driver from rule 5, must NEVER
contain a fabricated number, rating, or count, and is the sole field that should be an
empty string when the theme is out of primary scope."""


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _CLIENT


def _build_user_prompt(profile, theme, match_reason, exclude_category=None):
    candidates = rank_suggestable_categories(profile, exclude=exclude_category)
    ranked_list = "\n".join(f"  {i}. {c}" for i, c in enumerate(candidates, 1))
    exclude_note = (
        f'\nNote: this user just added a checkout cart-filler item from "{exclude_category}" '
        "to their cart. That category is intentionally excluded from the candidate list below "
        "— pick a DIFFERENT never-bought category. The project's core thesis is nudging toward "
        "a new category every month, not repeating the one just trialed.\n"
        if exclude_category else ""
    )
    return f"""Matched friction theme: {theme['name']}
Theme description: {theme['description']}
Theme lead-with signals: {', '.join(theme.get('lead_with', []))}
Theme in primary nudge scope: {not theme.get('out_of_primary_scope', False)}
Why this user matched: {match_reason}
{exclude_note}
User (synthetic profile):
- Orders: {profile['order_frequency']}
- Existing categories: {', '.join(profile['top_categories'])}
- Recent incident: {json.dumps(profile.get('recent_incident'))}
- In their words: "{profile.get('stated_barrier', '')}"

Prioritized candidate categories (ranked best-fit first for THIS user, none overlap
their existing basket) — choose exactly ONE per rule 3:
{ranked_list}

Write the nudge now."""


def generate_nudge(profile, theme, match_reason, client=None, exclude_category=None):
    client = client or _client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.4,
        response_format={"type": "json_object"},
        timeout=20,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(profile, theme, match_reason, exclude_category)},
        ],
    )
    out = json.loads(resp.choices[0].message.content)
    out["_model"] = GROQ_MODEL
    out["_matched_theme"] = theme["name"]
    out["_match_reason"] = match_reason
    out["_in_primary_scope"] = not theme.get("out_of_primary_scope", False)
    return out


if __name__ == "__main__":
    from friction_matching import load_profiles, load_themes, match_profile_to_theme

    themes = load_themes()
    client = _client()
    for p in load_profiles()[:3]:
        theme, reason = match_profile_to_theme(p, themes)
        result = generate_nudge(p, theme, reason, client=client)
        print(f"=== {p['user_id']} — {theme['name']} ===")
        print(json.dumps(result, indent=2))
        print()
