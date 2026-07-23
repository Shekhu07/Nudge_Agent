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

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"

# Categories to consider suggesting, kept away from each profile's existing staples.
SUGGESTABLE_CATEGORIES = [
    "personal care", "home & cleaning", "baby care", "pet supplies",
    "beauty & cosmetics", "packaged gourmet foods", "health & wellness",
    "kitchen & dining", "stationery & office",
]

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
3. Suggest exactly ONE new category, drawn from the allowed list, that does not overlap
   the user's existing categories.
4. Be honest about scope. If the matched theme says the user's stagnation is NOT
   trust-driven (low intent, no incident), do NOT pretend a guarantee fixes their reason
   for not exploring — lead instead with relevance and keep the guarantee as a secondary
   reassurance. Never overclaim.
5. Keep the nudge under 45 words. Keep the reasoning under 40 words.

Return STRICT JSON only, no prose around it:
{
  "suggested_category": "...",
  "nudge": "...",
  "reasoning": "...",
  "headline": "in-app card headline: a compelling 4-8 word phrase, not one word",
  "body": "in-app card body: TWO full sentences (25-40 words) in second person. Sentence 1 names the habit you noticed in their order history. Sentence 2 invites them to try the new category.",
  "refund_line": "the refund/return guarantee as a concrete 5-9 word phrase, e.g. 'Instant refund if quality isn't perfect'",
  "fresh_line": "the quality/freshness signal as a concrete 5-9 word phrase, e.g. 'Verified brands, sealed and batch-checked'",
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
concrete and must appear whenever the theme is in primary scope."""


def _client():
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _build_user_prompt(profile, theme, match_reason):
    return f"""Matched friction theme: {theme['name']}
Theme description: {theme['description']}
Theme lead-with signals: {', '.join(theme.get('lead_with', []))}
Theme in primary nudge scope: {not theme.get('out_of_primary_scope', False)}
Why this user matched: {match_reason}

User (synthetic profile):
- Orders: {profile['order_frequency']}
- Existing categories: {', '.join(profile['top_categories'])}
- Recent incident: {json.dumps(profile.get('recent_incident'))}
- In their words: "{profile.get('stated_barrier', '')}"

Allowed new categories to choose ONE from (must not overlap existing):
{', '.join(SUGGESTABLE_CATEGORIES)}

Write the nudge now."""


def generate_nudge(profile, theme, match_reason, client=None):
    client = client or _client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(profile, theme, match_reason)},
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
