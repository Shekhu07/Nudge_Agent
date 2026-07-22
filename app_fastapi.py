"""Delivery layer: FastAPI endpoint + minimal front end for the Category Nudge Agent.

Endpoints:
  GET  /              -> minimal HTML UI (pick a synthetic profile, see the nudge)
  GET  /api/profiles  -> list the synthetic profiles
  POST /api/nudge     -> {user_id} -> matched theme + generated nudge

Run locally:  uvicorn app:app --reload --port 7860
Deploy:       HuggingFace Spaces (Docker SDK) — see README.md. Port 7860 is the HF default.
"""
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent import generate_nudge
from friction_matching import load_profiles, load_themes, match_profile_to_theme

load_dotenv()

app = FastAPI(title="Blinkit Category Nudge Agent", version="1.0")

_THEMES = load_themes()
_PROFILES = {p["user_id"]: p for p in load_profiles()}


class NudgeRequest(BaseModel):
    user_id: str


@app.get("/api/profiles")
def list_profiles():
    return [
        {"user_id": p["user_id"], "persona": p["persona"],
         "top_categories": p["top_categories"],
         "recent_incident": p.get("recent_incident")}
        for p in _PROFILES.values()
    ]


@app.post("/api/nudge")
def make_nudge(req: NudgeRequest):
    profile = _PROFILES.get(req.user_id)
    if not profile:
        raise HTTPException(404, f"Unknown user_id {req.user_id}")
    if "GROQ_API_KEY" not in os.environ:
        raise HTTPException(500, "GROQ_API_KEY not configured on the server")
    theme, reason = match_profile_to_theme(profile, _THEMES)
    result = generate_nudge(profile, theme, reason)
    return {
        "user_id": req.user_id,
        "persona": profile["persona"],
        "matched_theme": theme["name"],
        "match_reason": reason,
        "in_primary_scope": result["_in_primary_scope"],
        "suggested_category": result.get("suggested_category"),
        "nudge": result.get("nudge"),
        "reasoning": result.get("reasoning"),
        "model": result["_model"],
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blinkit Category Nudge Agent</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; line-height: 1.5; }
  h1 { font-size: 1.4rem; margin-bottom: .2rem; }
  .sub { opacity: .7; font-size: .85rem; margin-bottom: 1.2rem; }
  .synthetic { background: #fff3cd; color: #664d03; border: 1px solid #ffe69c;
               padding: .5rem .75rem; border-radius: 6px; font-size: .8rem;
               margin-bottom: 1.2rem; }
  select, button { font-size: 1rem; padding: .5rem; border-radius: 6px; }
  button { cursor: pointer; margin-left: .5rem; }
  .card { border: 1px solid #8884; border-radius: 10px; padding: 1rem 1.2rem;
          margin-top: 1.2rem; }
  .nudge { font-size: 1.15rem; font-weight: 600; margin: .3rem 0 1rem; }
  .row { font-size: .85rem; margin: .25rem 0; }
  .label { opacity: .6; }
  .scope-out { color: #b45309; font-weight: 600; }
  .scope-in { color: #15803d; font-weight: 600; }
  code { font-size: .8rem; }
</style>
</head>
<body>
  <h1>Blinkit Category Nudge Agent</h1>
  <div class="sub">Part 4 MVP — nudges a repeat buyer toward one new category,
    leading with the two research-ranked trust drivers (refund guarantee + quality signal).</div>
  <div class="synthetic"><strong>Synthetic demo data.</strong> Profiles are hand-authored to
    reflect the real Part 2 &ldquo;stuck segment&rdquo;; no real Blinkit customer data is used.</div>

  <label for="profile">Pick a synthetic user:</label>
  <select id="profile"></select>
  <button onclick="run()">Generate nudge</button>

  <div id="out"></div>

<script>
async function load() {
  const r = await fetch('/api/profiles');
  const profiles = await r.json();
  const sel = document.getElementById('profile');
  for (const p of profiles) {
    const o = document.createElement('option');
    o.value = p.user_id;
    o.textContent = p.user_id + ' — ' + p.persona;
    sel.appendChild(o);
  }
}
async function run() {
  const user_id = document.getElementById('profile').value;
  const out = document.getElementById('out');
  out.innerHTML = '<p>Generating…</p>';
  const r = await fetch('/api/nudge', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id})
  });
  if (!r.ok) { out.innerHTML = '<p>Error: ' + (await r.text()) + '</p>'; return; }
  const d = await r.json();
  const scope = d.in_primary_scope
    ? '<span class="scope-in">in primary scope</span>'
    : '<span class="scope-out">out of primary scope — soft, honest nudge</span>';
  out.innerHTML = `
    <div class="card">
      <div class="row"><span class="label">Suggested new category:</span> ${d.suggested_category}</div>
      <div class="nudge">&ldquo;${d.nudge}&rdquo;</div>
      <div class="row"><span class="label">Matched friction theme:</span> ${d.matched_theme} (${scope})</div>
      <div class="row"><span class="label">Why matched:</span> ${d.match_reason}</div>
      <div class="row"><span class="label">Agent reasoning:</span> ${d.reasoning}</div>
      <div class="row"><span class="label">Model:</span> <code>${d.model}</code></div>
    </div>`;
}
load();
</script>
</body>
</html>"""
