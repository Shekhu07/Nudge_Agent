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
{"suggested_category": "...", "nudge": "...", "reasoning": "..."}
The "reasoning" field is your internal explanation of why this nudge fits THIS user and
theme (for the PM, not shown to the user)."""


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
