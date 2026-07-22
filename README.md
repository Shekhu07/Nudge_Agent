---
title: Blinkit Category Nudge Agent
emoji: 🛒
colorFrom: yellow
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Blinkit Category Nudge Agent (Part 4 MVP)

AI-native MVP for the NextLeap PM Fellowship graduation project. Given a repeat Blinkit
buyer's order history, it matches them to the strongest validated friction theme from the
Part 1 discovery engine + Part 3 problem statement, then uses a Groq LLM to generate a
**per-user, reasoned** category nudge — leading with the two research-ranked trust drivers
(a no-questions-asked refund guarantee + a visible quality/freshness signal).

This closes the project's traceability thread: Part 1 (why users don't explore) → Part 2
(survey confirmation) → Part 3 (problem statement) → **Part 4 (the nudge that addresses it).**

## Architecture

```
Synthetic profile  ->  Friction-Matching Layer  ->  Groq Agent            ->  FastAPI + UI
(data/*.json)          (friction_matching.py,        (agent.py,                (app.py)
                        deterministic rules)          llama-3.3-70b-versatile)
```

- **Data layer** — `data/synthetic_profiles.json` (clearly labeled SYNTHETIC; hand-authored
  to reflect the real Part 2 "stuck segment") and `data/friction_themes.json` (the real,
  locked Part 1/3 theme numbers).
- **Friction-matching layer** — deterministic rules map a profile's behavior to the closest
  theme. No LLM here.
- **Agent layer** — Groq `llama-3.3-70b-versatile` generates nudge copy + reasoning,
  constrained to the matched theme. Honestly scopes down for low-intent (no-incident) users
  instead of overclaiming a trust fix. Gemini is **not** used (dropped project-wide).
- **Delivery layer** — Gradio UI (`app.py`), the free HuggingFace Spaces entrypoint.
  A FastAPI + HTML variant with a JSON `/api/nudge` endpoint is also kept in
  `app_fastapi.py` for local/API use and non-HF hosts (Render etc.).

## Scope (honest boundaries)

Addresses only the ~50% of category stagnation that is quality/trust-driven, per
`docs/problem_statement.md` §6. Low-intent users (no incident) are matched to an
out-of-primary-scope theme and get a softer, relevance-led nudge. Single nudge mechanic
(refund + quality signal combined); the pre-acceptance-inspection driver is a deferred
backlog item.

## Run locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...          # or a .env file with GROQ_API_KEY

python app.py                    # Gradio UI on http://127.0.0.1:7860
# or, for the FastAPI + JSON API variant:
uvicorn app_fastapi:app --reload --port 7860
```

## Deploy to HuggingFace Spaces (production, free — Gradio SDK)

The Docker SDK is gated behind a paid plan on some accounts; the **Gradio SDK is free**
and is what this README's front-matter (`sdk: gradio`, `app_file: app.py`) targets.

1. Create a new Space → SDK: **Gradio** → Blank. Leave hardware on **CPU basic · Free**.
2. Push this `mvp/` directory's contents to the Space repo (or upload via the web UI).
   `Dockerfile` is harmless to include but unused by the Gradio SDK.
3. In the Space **Settings → Variables and secrets**, add secret `GROQ_API_KEY`.
4. HF installs `requirements.txt` and runs `app.py` automatically → stable public URL.

## Endpoints (FastAPI variant only, `app_fastapi.py`)

- `GET /` — demo UI
- `GET /api/profiles` — list synthetic profiles
- `POST /api/nudge` — `{"user_id": "SYN-001"}` → matched theme + generated nudge
