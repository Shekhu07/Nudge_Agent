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
## Design vs. data (important)

The UI is imported from the **"Blinkit category nudge agent redesign"** Claude Design
project — an operator console (user picker → profile → agent reasoning) beside a phone
mockup of what the shopper sees. The design *chrome* is reproduced faithfully; the
**content is real**, not the mockup's hard-coded strings:

| Mockup | Shipped |
|---|---|
| Hard-coded nudge copy, headlines and reasoning per persona | **Generated live by Groq** (`llama-3.3-70b-versatile`) on every click |
| Invented confidence scores (88% / 81% / 84% / 79%) | **Real Part 1 theme evidence share** (e.g. 770/1,094 · 70%); out-of-scope users show "out of primary scope" rather than a fake number |
| Product prices / MRP / % off | Dropped — replaced with an "Illustrative demo item" label rather than inventing pricing |
| Personas invented in the mockup | The project's **existing 8 synthetic profiles**, extended with display-only fields (name, tenure, locality…), all still labelled SYNTHETIC in `data/synthetic_profiles.json` |
| Social-proof with invented counts ("Over 1,200 buyers rated 5/5", from the external expansion playbook) | **Numbers-free `social_proof_line`** — qualitative peer reassurance generated per user, forbidden by the prompt from stating any figure/rating/count; empty for out-of-scope users |

Matching remains **deterministic** (no LLM), and the honest out-of-scope path is surfaced
in the UI: a low-intent user with no incident gets an explicit banner saying the trust fix
does not apply to them, per `docs/problem_statement.md` §6.

- **Delivery layer** — Gradio UI (`app.py`), the free HuggingFace Spaces entrypoint.
  A FastAPI + HTML variant with a JSON `/api/nudge` endpoint is also kept in
  `app_fastapi.py` for local/API use and non-HF hosts (Render etc.).

The Gradio app has **three tabs**, all over the same synthetic profiles:
1. **Operator console** — pick a user → deterministic friction match → live Groq nudge.
2. **Auto-nudge queue** (`auto_targeting.py`) — the cohort that would be *auto-nudged*
   on a schedule: `order_frequency = Weekly AND tenure > 6mo`. "Run scheduled batch"
   generates each eligible user's notification live. **Simulated — no real push is sent.**
3. **Checkout cart-filler** (`cart_filler.py`, playbook Pillar 4) — when a cart is under the
   ₹199 free-delivery threshold, offers a low-cost item from a **never-bought** category as a
   zero-friction trial. Deterministic (no LLM); `FILLER_CATALOG` is synthetic, prices
   illustrative. Aimed at the low-intent segment the push nudge skips.

## Scope (honest boundaries)

Addresses only the ~50% of category stagnation that is quality/trust-driven, per
`docs/problem_statement.md` §6. Low-intent users (no incident) are matched to an
out-of-primary-scope theme and get a softer, relevance-led nudge. Single nudge mechanic
(refund + quality signal combined, plus a numbers-free peer `social_proof_line` — Q15's
tied-third driver; all three are lines in one nudge, not separate mechanics). The
pre-acceptance-inspection driver and the checkout cart-filler micro-trial (Category-
Expansion playbook, Pillar 4 — aimed at the low-intent segment this MVP does not target)
are deferred backlog items; see `docs/implementation-plan.md` Phase 5.

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

1. Create a new Space → SDK: **Gradio** → Blank. **CPU basic · Free** is ideal; if your
   account only offers **ZeroGPU**, that works too (see note below).
2. Push this `mvp/` directory's contents to the Space repo (or upload via the web UI).
   `Dockerfile` is harmless to include but unused by the Gradio SDK.
3. In the Space **Settings → Variables and secrets**, add secret `GROQ_API_KEY`.
4. HF installs `requirements.txt` and runs `app.py` automatically → stable public URL.

**ZeroGPU note:** this app is CPU-only (all LLM work is remote via Groq), but ZeroGPU
Spaces refuse to boot without a `@spaces.GPU` function. `app.py` registers a tiny no-op
(`_zerogpu_warmup`) purely to satisfy that startup check, and `spaces` is in
`requirements.txt`. So it runs on either CPU basic or ZeroGPU with no code change.

## Endpoints (FastAPI variant only, `app_fastapi.py`)

- `GET /` — demo UI
- `GET /api/profiles` — list synthetic profiles
- `POST /api/nudge` — `{"user_id": "SYN-001"}` → matched theme + generated nudge
