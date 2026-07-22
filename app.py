"""Delivery layer (Gradio) — the HuggingFace Spaces entrypoint.

HF's Gradio SDK is free (no Docker required) and runs this file automatically.
The friction-matching and Groq agent layers are unchanged; only the UI differs from
the FastAPI variant in app_fastapi.py (which is kept for local/API use and Render).

Local run:  python app.py   (serves on http://127.0.0.1:7860)
"""
import os

import gradio as gr
from dotenv import load_dotenv

from agent import generate_nudge
from friction_matching import load_profiles, load_themes, match_profile_to_theme

load_dotenv()

_THEMES = load_themes()
_PROFILES = {p["user_id"]: p for p in load_profiles()}
_CHOICES = [f"{p['user_id']} — {p['persona']}" for p in _PROFILES.values()]


def _profile_from_choice(choice):
    user_id = choice.split(" — ")[0]
    return _PROFILES[user_id]


def run_nudge(choice):
    if "GROQ_API_KEY" not in os.environ:
        return "**Error:** `GROQ_API_KEY` is not configured on the server. Add it in the Space's Settings → Secrets."
    profile = _profile_from_choice(choice)
    theme, reason = match_profile_to_theme(profile, _THEMES)
    result = generate_nudge(profile, theme, reason)
    scope = ("✅ in primary scope" if result["_in_primary_scope"]
             else "⚠️ out of primary scope — soft, honest nudge (low-intent user)")
    return f"""### Suggested new category: **{result.get('suggested_category')}**

> ## &ldquo;{result.get('nudge')}&rdquo;

| | |
|---|---|
| **Matched friction theme** | {theme['name']} ({scope}) |
| **Why matched** | {reason} |
| **Agent reasoning** | {result.get('reasoning')} |
| **Model** | `{result['_model']}` |
"""


with gr.Blocks(title="Blinkit Category Nudge Agent") as demo:
    gr.Markdown(
        "# 🛒 Blinkit Category Nudge Agent\n"
        "Part 4 MVP — nudges a repeat buyer toward one new category, leading with the two "
        "research-ranked trust drivers (**refund guarantee + quality/freshness signal**).\n\n"
        "> ⚠️ **Synthetic demo data.** Profiles are hand-authored to reflect the real Part 2 "
        "&ldquo;stuck segment&rdquo;; no real Blinkit customer data is used."
    )
    profile_dd = gr.Dropdown(choices=_CHOICES, value=_CHOICES[0], label="Pick a synthetic user")
    go = gr.Button("Generate nudge", variant="primary")
    out = gr.Markdown()
    go.click(run_nudge, inputs=profile_dd, outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
