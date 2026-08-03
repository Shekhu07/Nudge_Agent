"""Delivery layer (Gradio) — the HuggingFace Spaces entrypoint for the Part 4 MVP.

Visual design imported from the "Blinkit category nudge agent redesign" Claude Design
project: an operator console (user picker -> profile -> agent reasoning) beside a phone
mockup of what the shopper actually sees.

Real vs. mock, kept unambiguous (CLAUDE.md hard requirement):
  * User profiles are SYNTHETIC and labelled as such in the UI ("Synthetic demo data").
  * The nudge copy and the agent's reasoning are generated LIVE by Groq per request —
    not the mockup's hard-coded strings.
  * The friction themes and the confidence figure are the REAL locked Part 1 numbers
    (evidence share), not an LLM-invented percentage.

Local run:  python app.py   (http://127.0.0.1:7860)
"""
import html
import os
import random
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from dotenv import load_dotenv

from agent import PUSH_ANGLES, generate_nudge
from friction_matching import (SUGGESTABLE_CATEGORIES, adjacency_scores, load_profiles,
                               load_themes, match_profile_to_theme,
                               primary_suggested_category, rank_suggestable_categories)
from auto_targeting import (ELIGIBLE_FREQS, MIN_TENURE_MONTHS, eligibility_detail,
                            eligible_profiles, parse_tenure_months, to_notification)
from cart_filler import (DELIVERY_FEE, FREE_DELIVERY_THRESHOLD, cart_lines,
                         never_bought_categories, suggest_fillers)

# Per-category glyph for the checkout line items, so a cart isn't three identical 🛒 tiles.
STAPLE_EMOJI = {
    "Groceries": "🌾", "Fresh Produce": "🍅", "Dairy": "🥛", "Snacks": "🍪",
    "Beverages": "🥤", "Household": "🧴", "Personal Care": "🧼",
}

load_dotenv()

# --- ZeroGPU compatibility (see the discovery app for the full rationale) ---
try:
    import spaces

    GPU = spaces.GPU
except ImportError:
    def GPU(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn


@GPU(duration=1)
def _zerogpu_warmup():
    return "ok"
# ---------------------------------------------------------------------------

THEMES = load_themes()
PROFILES = load_profiles()
BY_ID = {p["user_id"]: p for p in PROFILES}

INK, YELLOW, BG = "#16130A", "#F8CD1B", "#F4F5F3"

FONT_LINK = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""

FORCE_LIGHT = """
function(){const u=new URL(window.location);
if(u.searchParams.get('__theme')!=='light'){u.searchParams.set('__theme','light');
window.location.replace(u.href);}}
"""

# Per-profile chip styling: the avatar tile (initials + colour) and the blurb line are
# generated as CSS so a plain Gradio Button can render the design's rich chip.
CHIP_CSS = "".join(f"""
#chip-{p['user_id']}::before{{content:"{p['initials']}";background:{p['avatar_bg']};
  width:38px;height:38px;border-radius:11px;display:flex;align-items:center;
  justify-content:center;font-weight:800;font-size:14px;color:#16130A;flex:none}}
#chip-{p['user_id']}::after{{content:"{p['blurb']}";display:block;font-size:11.5px;
  font-weight:500;color:#6B6B60;margin-top:2px;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}}
""" for p in PROFILES)

CSS = """
:root{color-scheme:light}
html,body,.gradio-container,.dark .gradio-container{background:#F4F5F3 !important}
/* THEME-PROOFING (above the class rules; not !important so ours + inline still win) */
.gradio-container.gradio-container.gradio-container *,.dark .gradio-container.gradio-container.gradio-container *{color:#16130A}
body,.gradio-container,.gradio-container *,button,input,textarea{
  font-family:'Plus Jakarta Sans',system-ui,-apple-system,sans-serif !important}
.gradio-container{max-width:1360px !important;margin:0 auto !important;padding:26px 28px 40px !important}
footer,.footer,.show-api,.built-with,.settings{display:none !important}
.block,.form,.gr-box,.gr-group,.panel,.gr-panel,.styler{
  background:transparent !important;border:none !important;box-shadow:none !important;padding:0 !important}
.gradio-container .gap{gap:0 !important}

/* top bar */
.nb-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:22px;
  flex-wrap:wrap;gap:14px}
.nb-brand{display:flex;align-items:center;gap:14px}
.nb-logo{width:46px;height:46px;border-radius:13px;background:#F8CD1B;display:flex;
  align-items:center;justify-content:center;font-size:24px;box-shadow:0 4px 14px rgba(248,205,27,.45)}
.nb-t1.nb-t1.nb-t1{font-size:19px;font-weight:800;letter-spacing:-.02em;color:#16130A}
.nb-t2.nb-t2.nb-t2{font-size:13px;font-weight:500;color:#63635A}
.nb-pill{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:700;
  padding:8px 14px;border-radius:999px}
.nb-dot{width:7px;height:7px;border-radius:50%;display:inline-block}

/* cards */
.nb-card{background:#fff !important;border:1px solid #E7E8E2 !important;border-radius:16px !important;
  padding:15px 17px !important;box-shadow:0 1px 2px rgba(0,0,0,.03) !important;margin-bottom:12px !important}
.nb-eyebrow.nb-eyebrow.nb-eyebrow{font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#6B6B60}
.nb-tile{flex:1;background:#F7F8F5;border-radius:12px;padding:9px 11px}

/* collapsible secondary panels (methodology-heavy content, hidden until asked for) */
details.nb-card{cursor:default}
details.nb-card>summary{cursor:pointer;list-style:none;display:flex;align-items:center;
  gap:8px;font-size:12px;font-weight:800;letter-spacing:.04em;color:#16130A}
details.nb-card>summary::-webkit-details-marker{display:none}
details.nb-card>summary .nb-chev{font-size:10px;color:#9A9A8C;transition:transform .15s;flex:none}
details.nb-card[open]>summary .nb-chev{transform:rotate(90deg)}
details.nb-card[open]>summary{margin-bottom:13px;padding-bottom:12px;border-bottom:1px solid #F0F1EC}
details.nb-card>summary .nb-sub{font-weight:500;color:#8E8F86;text-transform:none;letter-spacing:0}
.nb-tile-k.nb-tile-k.nb-tile-k{font-size:11px;font-weight:700;color:#6B6B60;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.nb-tile-v.nb-tile-v.nb-tile-v{font-size:14px;font-weight:700;color:#16130A}

/* user chips (Gradio buttons dressed as the design's chips) */
.nb-chip{display:flex !important;align-items:center !important;gap:9px !important;
  padding:9px 11px !important;border-radius:13px !important;text-align:left !important;
  font-size:14px !important;font-weight:700 !important;color:#16130A !important;
  background:#F7F8F5 !important;border:1.5px solid #EDEEE8 !important;box-shadow:none !important;
  width:100% !important;justify-content:flex-start !important;height:auto !important;
  min-height:0 !important;white-space:nowrap !important;overflow:hidden !important;
  flex-wrap:nowrap !important;line-height:1.25 !important}
.nb-chip:hover{border-color:#F8CD1B !important}
.nb-chip.primary{background:#FFFBEC !important;border:1.5px solid #F8CD1B !important;
  box-shadow:0 2px 10px rgba(248,205,27,.25) !important}

/* Dropdown + number inputs (cart-filler tab). Gradio's native dropdown uses Tailwind
   dark: variants (e.g. option rows get dark:bg-gray-600), but our theme-proofing forces
   dark ink even under .dark — so if HF's iframe forces dark mode the option text goes
   dark-on-dark and disappears. Pin the field and the options list to a light surface in
   BOTH themes so the ink stays readable. Selectors verified against the rendered DOM. */
.gradio-container input, .dark .gradio-container input,
.gradio-container .secondary-wrap, .dark .gradio-container .secondary-wrap,
.gradio-container ul.options, .dark .gradio-container ul.options,
.gradio-container ul.options li, .dark .gradio-container ul.options li,
.gradio-container ul.options li *, .dark .gradio-container ul.options li *{
  background:#fff !important;color:#16130A !important}
.gradio-container ul.options{border:1px solid #E7E8E2 !important;
  box-shadow:0 8px 26px rgba(0,0,0,.14) !important}
.gradio-container ul.options li.selected, .gradio-container ul.options li.active,
.gradio-container ul.options li:hover,
.dark .gradio-container ul.options li.selected, .dark .gradio-container ul.options li.active,
.dark .gradio-container ul.options li:hover{background:#FFFBEC !important;color:#16130A !important}

/* range sliders — the chrome reset above strips Gradio's own track styling, so the
   control renders as a near-invisible hairline. Restore an explicit track + thumb in
   the brand palette (matches the design's styled range input). */
.gradio-container input[type=range]{-webkit-appearance:none;appearance:none;height:6px;
  border-radius:99px;background:#EDEEE8 !important;outline:none;width:100%;margin:6px 0}
.gradio-container input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:20px;
  height:20px;border-radius:50%;background:#F8CD1B;border:3px solid #fff;
  box-shadow:0 2px 8px rgba(0,0,0,.2);cursor:pointer}
.gradio-container input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;
  background:#F8CD1B;border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.2);cursor:pointer}
#tenwrap{margin:10px 0 4px !important}
#tenwrap .wrap,#tenwrap .head{min-height:0 !important;margin-bottom:0 !important}
.gradio-container label[data-testid="block-label"],.gradio-container .head span{
  font-size:12px !important;font-weight:700 !important;color:#6B6B60 !important}

/* generate button */
#gen, #genbatch{width:100% !important;border:none !important;font-size:15px !important;font-weight:800 !important;
  padding:15px !important;border-radius:14px !important;color:#16130A !important;
  background:#F8CD1B !important;box-shadow:0 6px 18px rgba(248,205,27,.4) !important;
  margin-top:16px !important}

/* dark reasoning card */
.nb-dark.nb-dark.nb-dark{background:#16130A;border-radius:16px;padding:17px;color:#F4F5F3}
.nb-dark.nb-dark.nb-dark *{color:#CFD0C6}
.nb-dark.nb-dark.nb-dark .k{color:#F8CD1B;font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase}
.nb-dark.nb-dark.nb-dark .h{color:#fff;font-size:13px;font-weight:800;margin-bottom:3px}
.nb-ic{width:26px;height:26px;border-radius:8px;background:rgba(248,205,27,.16);display:flex;
  align-items:center;justify-content:center;font-size:13px;flex:none}

/* phone */
.nb-phonewrap{position:sticky;top:12px;display:flex;flex-direction:column;align-items:center;gap:10px}
.nb-phone{width:340px;background:#0E0E0C;border-radius:44px;padding:12px;
  box-shadow:0 24px 60px rgba(0,0,0,.22)}
.nb-screen{background:#fff;border-radius:33px;overflow:hidden}
@keyframes nudgePop{0%{opacity:0;transform:translateY(14px) scale(.98)}100%{opacity:1;transform:none}}
@keyframes barGrow{from{width:0}}
.nb-pop{animation:nudgePop .5s cubic-bezier(.2,.7,.3,1) both}

@media (max-width:1000px){
  .nb-cols{flex-direction:column !important}
  .nb-phonewrap{position:static;margin-top:8px}
}
""" + CHIP_CSS


def esc(t):
    return html.escape(str(t) if t is not None else "")


# ----------------------------- panels -----------------------------
def profile_html(p, theme, reason):
    tiles = [("Tenure", p["tenure"]), ("Cadence", p["cadence"]), ("Basket", p["avg_basket"])]
    tile_html = "".join(
        f'<div class="nb-tile"><div class="nb-tile-k">{esc(k)}</div>'
        f'<div class="nb-tile-v">{esc(v)}</div></div>' for k, v in tiles)
    buys = "".join(
        f'<span style="background:#EAF7EE;color:#146634;font-size:12.5px;font-weight:600;'
        f'padding:6px 12px;border-radius:999px">{esc(c)}</span>' for c in p["buys_display"])
    return f"""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:18px">
  <div style="width:56px;height:56px;border-radius:16px;display:flex;align-items:center;
    justify-content:center;font-weight:800;font-size:20px;background:{p['avatar_bg']};color:#16130A">{esc(p['initials'])}</div>
  <div style="flex:1;min-width:0">
    <div style="font-size:18px;font-weight:800;letter-spacing:-.01em;color:#16130A">{esc(p['display_name'])}</div>
    <div style="font-size:13px;font-weight:500;color:#63635A">{esc(p['persona'])}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:22px;font-weight:800;color:#16130A;line-height:1">{p['orders']}</div>
    <div style="font-size:11px;font-weight:600;color:#6B6B60;text-transform:uppercase;letter-spacing:.05em">orders</div>
  </div>
</div>
<div style="display:flex;gap:10px;margin-bottom:18px">{tile_html}</div>
<div class="nb-eyebrow" style="margin-bottom:9px">Buys regularly</div>
<div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:16px">{buys}</div>
<div style="display:flex;align-items:center;gap:10px;background:linear-gradient(90deg,#FFF9E3,#FFFDF6);
  border:1px dashed #EBD68A;border-radius:13px;padding:13px 15px">
  <div style="font-size:20px">🎯</div>
  <div style="flex:1">
    <div style="font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;color:#6F5700">
      Matched friction theme</div>
    <div style="font-size:15px;font-weight:800;color:#16130A">{esc(theme['name'])}</div>
    <div style="font-size:12px;font-weight:500;color:#63635A;margin-top:3px">{esc(reason)}</div>
  </div>
</div>"""


def alternatives_html(p, r=None):
    """The design's 'Also considered — ranked' panel, wired to the REAL deterministic
    ranker (friction_matching.rank_suggestable_categories + adjacency_scores).

    The imported mockup carried invented 0-1 scores (0.88 / 0.61 / 0.44) that read as
    model confidence. These are the actual integer adjacency weights instead, shown in
    the order the agent received the candidate list, with the agent's real pick marked.
    Nothing here is estimated.
    """
    pool = rank_suggestable_categories(p)
    scores = adjacency_scores(p)
    if not pool:
        return ""
    picked = (r or {}).get("suggested_category", "")
    picked_n = str(picked).lower().strip()
    top = max(scores[c] for c in pool) or 1
    best = max(pool, key=lambda c: scores[c])

    rows = ""
    for c in pool:
        is_pick = picked_n and c.lower() == picked_n
        if is_pick:
            tag, tag_css = "SELECTED", ("color:#16130A;background:#F8CD1B")
        elif c == best and picked_n:
            tag, tag_css = "RANKER #1", ("color:#F0E4B4;background:rgba(248,205,27,.16)")
        else:
            tag, tag_css = "considered", ("color:#8E8F86;background:rgba(255,255,255,.06)")
        rows += f"""
        <div style="display:flex;align-items:center;gap:11px">
          <div style="font-size:12.5px;font-weight:{'800' if is_pick else '600'};
            color:{'#F8CD1B' if is_pick else '#B7B8AE'};width:148px;flex:none;white-space:nowrap;
            overflow:hidden;text-overflow:ellipsis">{esc(c)}</div>
          <div style="flex:1;height:6px;background:rgba(255,255,255,.09);border-radius:99px;overflow:hidden">
            <div style="height:100%;border-radius:99px;width:{round(100*scores[c]/top)}%;
              background:{'#F8CD1B' if is_pick else 'rgba(255,255,255,.28)'};
              animation:barGrow .7s cubic-bezier(.2,.7,.3,1)"></div></div>
          <div style="font-size:12px;font-weight:700;color:#B7B8AE;width:16px;text-align:right;flex:none">{scores[c]}</div>
          <div style="font-size:9.5px;font-weight:800;letter-spacing:.05em;{tag_css};padding:3px 7px;
            border-radius:5px;width:78px;text-align:center;flex:none">{tag}</div>
        </div>"""

    foot = ("Adjacency weight from the user's existing basket — a deterministic integer, "
            "not a confidence score. Order is the candidate list handed to the agent "
            "(equally-adjacent candidates are rotated per user for variety).")
    return f"""
    <div style="display:flex;gap:11px">
      <div class="nb-ic">⚖️</div>
      <div style="flex:1;min-width:0">
        <div class="h" style="margin-bottom:7px">Also considered — ranked</div>
        <div style="display:flex;flex-direction:column;gap:6px">{rows}</div>
        <div style="font-size:10.5px;line-height:1.45;color:#8E8F86;margin-top:7px">{foot}</div>
      </div>
    </div>"""


def reasoning_html(theme, r=None, p=None):
    share = theme.get("evidence_share")
    pct = round(100 * share) if share else None
    # Real Part 1 evidence share — NOT an LLM-invented confidence score.
    if pct is not None:
        meter = (f'<div style="font-size:12px;font-weight:700;color:#C9CABF">'
                 f'theme evidence {theme["evidence_count"]:,}/{theme["evidence_total"]:,} · {pct}%</div>')
        bar = (f'<div style="height:5px;background:rgba(255,255,255,.1);border-radius:99px;'
               f'margin:10px 0 20px;overflow:hidden"><div style="height:100%;border-radius:99px;'
               f'background:linear-gradient(90deg,#F8CD1B,#1F9D55);width:{pct}%;'
               f'animation:barGrow .8s cubic-bezier(.2,.7,.3,1)"></div></div>')
    elif theme.get("out_of_primary_scope"):
        # Genuinely outside the trust-driven scope (low_intent_no_incident) — the nudge
        # mechanic is not the right lever and the UI must say so.
        meter, bar = ('<div style="font-size:12px;font-weight:700;color:#C9CABF">out of primary scope</div>',
                      '<div style="height:20px"></div>')
    else:
        # In scope, but a sub-theme Part 1 never counted separately (packaging_fulfillment).
        # A null share is NOT the same as out-of-scope — saying so contradicted the
        # (correctly absent) out-of-scope banner.
        meter, bar = ('<div style="font-size:12px;font-weight:700;color:#C9CABF">'
                      'sub-theme · not separately counted in Part 1</div>',
                      '<div style="height:20px"></div>')

    alts = alternatives_html(p, r) if p else ""

    if not r:
        pre = (f'<div style="margin-top:18px;padding-top:18px;'
               f'border-top:1px solid rgba(255,255,255,.1)">{alts}</div>') if alts else ""
        return f"""
<div class="nb-dark">
  <div style="display:flex;align-items:center;gap:9px">
    <div class="k">Why this nudge</div>
    <div style="flex:1;height:1px;background:rgba(255,255,255,.12)"></div>{meter}</div>
  {bar}
  <div style="font-size:13px;line-height:1.55;color:#B7B8AE">Press <b style="color:#F8CD1B">Generate nudge</b>
    to run the Groq agent for this user. The matched theme above and the candidate ranking
    below are already resolved deterministically — no LLM involved in either.</div>
  {pre}
</div>"""

    drivers = []
    if r.get("inspect_line"):
        drivers.append(("Inspect before accepting", r.get("inspect_line", "")))
    drivers += [("Refund guarantee", r.get("refund_line", "")),
                ("Freshness / quality signal", r.get("fresh_line", ""))]
    if r.get("social_proof_line"):
        drivers.append(("Peer social proof", r.get("social_proof_line", "")))
    d_html = "".join(f"""
      <div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
        border-radius:11px;padding:9px 11px;display:flex;gap:9px;align-items:flex-start">
        <div style="width:20px;height:20px;border-radius:6px;background:#F8CD1B;color:#16130A;
          font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center;flex:none">{i}</div>
        <div><div style="font-size:12.5px;font-weight:700;color:#fff">{esc(n)}</div>
        <div style="font-size:11.5px;line-height:1.45;color:#B7B8AE;margin-top:1px">{esc(v)}</div></div>
      </div>""" for i, (n, v) in enumerate(drivers, 1))

    scope = "" if r.get("_in_primary_scope") else (
        '<div style="background:rgba(248,205,27,.12);border:1px solid rgba(248,205,27,.3);'
        'border-radius:12px;padding:11px 13px;font-size:12.5px;line-height:1.5;color:#F0E4B4">'
        '⚠️ This user shows <b style="color:#F8CD1B">no quality incident</b> — their stagnation looks '
        'intent-driven. The agent leads with relevance instead of a guarantee, and does not '
        'claim the trust fix applies.</div>')

    return f"""
<div class="nb-dark">
  <div style="display:flex;align-items:center;gap:9px">
    <div class="k">Why this nudge</div>
    <div style="flex:1;height:1px;background:rgba(255,255,255,.12)"></div>{meter}</div>
  {bar}
  <div style="display:flex;flex-direction:column;gap:12px">
    {scope}
    <div style="display:flex;gap:11px"><div class="nb-ic">👤</div>
      <div><div class="h">Why this user</div>
      <div style="font-size:12.5px;line-height:1.5;color:#CFD0C6">{esc(r.get('why_user'))}</div></div></div>
    <div style="display:flex;gap:11px"><div class="nb-ic">🧭</div>
      <div><div class="h">Why this category</div>
      <div style="font-size:12.5px;line-height:1.5;color:#CFD0C6">{esc(r.get('why_category'))}</div></div></div>
    {alts}
    <div>
      <div style="display:flex;gap:11px;margin-bottom:8px"><div class="nb-ic">🛡️</div>
        <div class="h" style="padding-top:6px">Trust drivers, ranked (research-led)</div></div>
      <div style="display:flex;flex-direction:column;gap:6px">{d_html}</div>
    </div>
    <div style="font-size:11px;color:#8E8F86;font-family:monospace">model · {esc(r.get('_model'))}</div>
  </div>
</div>"""


def measurement_plan_html(r=None):
    """The design's 'Holdout test' panel — rebuilt as an experiment DESIGN, not results.

    The imported mockup shipped fabricated outcomes (+34% lift, 8.4% vs 11.3% converted,
    n=2,400, p<0.05, "simulated on Part 2 segment data"). No such experiment was ever run,
    so shipping those numbers would invent evidence — the exact failure CLAUDE.md forbids.
    What is real and worth showing is how the nudge WOULD be validated: the arms, the
    primary metric (the PRD's own growth goal) and the guardrail. No figures, and the
    "not yet run" state is stated in the panel rather than implied.
    """
    variant = esc(r.get("headline")) if r else "— generate a nudge to populate variant B —"
    arm = lambda tone, label, headline, note: f"""
      <div style="border:{'1.5px solid #F8CD1B' if tone else '1px solid #EDEEE8'};border-radius:14px;
        padding:14px;background:{'#FFFCF0' if tone else '#FAFBF8'}">
        <div style="display:flex;align-items:center;gap:7px;margin-bottom:9px">
          <span style="width:8px;height:8px;border-radius:50%;background:{'#E7B400' if tone else '#C4C5B9'}"></span>
          <span style="font-size:11.5px;font-weight:800;text-transform:uppercase;letter-spacing:.05em;
            color:{'#7A6100' if tone else '#6B6B60'}">{label}</span></div>
        <div style="font-size:13.5px;font-weight:700;line-height:1.35;margin-bottom:6px;
          color:{'#16130A' if tone else '#44443B'}">{headline}</div>
        <div style="font-size:12px;color:{'#7A6836' if tone else '#6B6B60'};line-height:1.45">{note}</div>
      </div>"""

    body = f"""
<div style="font-size:12.5px;line-height:1.55;color:#63635A;margin-bottom:14px">
  Design only. The MVP has not been shipped to real users, so there are
  <b style="color:#16130A">no conversion, lift or significance figures</b> — any number here
  would be invented.</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
  {arm(False, "Control · Variant A", "“Flat discount on your next order”",
       "Price-led and category-agnostic — the status quo nudge. Expected to deepen the existing basket rather than open a new category.")}
  {arm(True, "Agent · Variant B", f"“{variant}”",
       "Trust-led: names the user's own friction, leads with the two research-ranked drivers, offer last.")}
</div>
<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:14px">
  <div style="flex:1;min-width:200px;background:#F7F8F5;border-radius:12px;padding:11px 13px">
    <div class="nb-tile-k">Primary metric</div>
    <div style="font-size:13px;font-weight:700;color:#16130A;line-height:1.4">% of MAU purchasing from
      ≥1 new category that month</div></div>
  <div style="flex:1;min-width:200px;background:#F7F8F5;border-radius:12px;padding:11px 13px">
    <div class="nb-tile-k">Guardrail</div>
    <div style="font-size:13px;font-weight:700;color:#16130A;line-height:1.4">No fall in existing-category
      order frequency</div></div>
</div>
<div style="margin-top:13px;padding-top:12px;border-top:1px solid #F0F1EC;font-size:11.5px;
  font-weight:600;color:#6B6B60;line-height:1.55">
  Scope caveat carried from the problem statement: this can only move the trust-driven share of
  category stagnation, not the low-intent share — so the read-out would be split by whether the
  user had a quality incident.</div>"""
    return f"""<details class="nb-card">
  <summary><span class="nb-chev">▸</span>📊 Measurement plan — how we would prove the nudge works
    <span class="nb-sub">design only, not yet run</span></summary>
  {body}
</details>"""


def instrumentation_html():
    """The design's 'Outcome tracker' — rebuilt as the event spec, not live telemetry.

    The mockup showed a 'live · 7d' funnel (100 / 64 / 19 / 11.3%) and a 21-day repurchase
    stat. No user has ever received this nudge, so those percentages are fabricated. The
    honest artifact is the event ladder the MVP would emit; the stage names are a real
    design decision, the numbers do not exist yet.
    """
    stages = [
        ("nudge_delivered", "Nudge rendered in the user's feed"),
        ("card_viewed", "Card enters viewport for ≥1s"),
        ("added_to_cart", "Suggested-category item added"),
        ("first_category_order", "Order placed in a category never bought before"),
    ]
    rows = "".join(f"""
      <div style="display:flex;gap:10px;align-items:flex-start">
        <div style="width:19px;height:19px;border-radius:6px;background:#F0F1EC;color:#6B6B60;
          font-size:10.5px;font-weight:800;display:flex;align-items:center;justify-content:center;
          flex:none;margin-top:1px">{i}</div>
        <div style="min-width:0">
          <div style="font-size:12.5px;font-weight:700;color:#16130A;font-family:monospace">{esc(k)}</div>
          <div style="font-size:11.5px;color:#6B6B60;line-height:1.45">{esc(v)}</div></div>
      </div>""" for i, (k, v) in enumerate(stages, 1))
    body = f"""
<div style="display:flex;flex-direction:column;gap:10px">{rows}</div>
<div style="margin-top:13px;padding-top:12px;border-top:1px solid #F0F1EC;font-size:11.5px;
  font-weight:600;color:#6B6B60;line-height:1.55">
  The four events the MVP would emit, in order. Deliberately shown without percentages — the
  nudge has not run against real users, so a funnel here would be fabricated.</div>"""
    return f"""<details class="nb-card">
  <summary><span class="nb-chev">▸</span>📶 Instrumentation — event spec
    <span class="nb-sub">no live data</span></summary>
  {body}
</details>"""


def _phone_shell(p, inner, hint=None):
    # Search placeholder shows what this shopper actually habitually buys (real profile
    # field), so the mock phone chrome carries no invented query.
    hint = hint or (p.get("buys_display") or ["groceries"])[0].lower()
    return f"""
<div class="nb-phonewrap">
  <div class="nb-eyebrow">What the shopper sees</div>
  <div class="nb-phone"><div class="nb-screen">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:11px 24px 6px;
      font-size:12px;font-weight:700;color:#16130A"><span>9:41</span><span>📶 &nbsp;🔋</span></div>
    <div style="background:#F8CD1B;padding:12px 18px 16px">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div><div style="font-size:11px;font-weight:700;color:#6F5700">Delivery in</div>
          <div style="font-size:19px;font-weight:800;color:#16130A;line-height:1.1">8 minutes</div>
          <div style="font-size:11.5px;font-weight:600;color:#5C4E12">Home · {esc(p['locality'])}</div></div>
        <div style="width:38px;height:38px;border-radius:50%;background:#16130A;color:#F8CD1B;
          display:flex;align-items:center;justify-content:center;font-weight:800;font-size:14px">{esc(p['initials'])}</div>
      </div>
      <div style="margin-top:12px;background:#fff;border-radius:11px;padding:10px 13px;font-size:13px;
        color:#6B6B60;font-weight:500;display:flex;align-items:center;gap:8px">🔍 Search “{esc(hint)}”</div>
    </div>
    <div style="padding:16px 15px 22px;background:#F7F8F5;min-height:520px;display:flex;
      flex-direction:column">
      <div style="flex:1">{inner}</div>
      <div style="width:120px;height:5px;border-radius:99px;background:rgba(0,0,0,.18);
        margin:18px auto 2px"></div>
    </div>
  </div></div>
  <div style="font-size:12px;color:#63635A;font-weight:500;text-align:center;max-width:300px;line-height:1.5">
    Nudge leads with the two research-ranked trust drivers before the offer — not a generic discount.</div>
</div>"""


def _secondary_category(p, r):
    """Deterministic "you can also try" pick — costs no extra Groq call.

    Reuses the SAME ranked, per-user-rotated pool already computed for the operator's
    "Also considered — ranked" panel (rank_suggestable_categories), so the shopper-facing
    hint and the PM-facing panel always point at consistent alternatives instead of two
    independently-computed pickers that could disagree with each other.

    Returns the next-best candidate that ISN'T the one actually nudged, or None if the
    pool is exhausted (shouldn't happen with 11 suggestable categories, but a profile
    with a very full basket could in principle leave nothing else to suggest).
    """
    pool = rank_suggestable_categories(p)
    picked = str((r or {}).get("suggested_category", "")).lower().strip()
    for c in pool:
        if c.lower() != picked:
            return c
    return None


def phone_html(p, r=None):
    if not r:
        inner = """
<div style="border:1.5px dashed #DADBD3;border-radius:20px;padding:44px 20px;text-align:center">
  <div style="font-size:30px">⚡</div>
  <div style="font-size:13px;color:#63635A;margin-top:10px;line-height:1.5;font-weight:500">
    Press <b style="color:#16130A">Generate nudge</b> to see<br>what this shopper would receive.</div>
</div>"""
        return _phone_shell(p, inner)

    # Title-case BEFORE escaping. The other order title-cases the "&amp;" that esc()
    # produces into "&Amp;", which is not a valid HTML entity, so it renders literally
    # as "Kitchen &Amp; Dining". Hits the 5 categories containing "&".
    cat = esc(r.get("suggested_category", "").title())
    emoji = esc(r.get("emoji") or "🛍️")
    secondary = _secondary_category(p, r)
    inner = f"""
<div class="nb-pop" style="background:#fff;border-radius:20px;overflow:hidden;
  box-shadow:0 6px 22px rgba(0,0,0,.08);border:1px solid #EFEFE9">
  <div style="background:#FFF3D6;padding:14px 16px;display:flex;align-items:center;gap:11px">
    <div style="font-size:26px">{emoji}</div>
    <div><div style="font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:#6F5700">
      Just for you · {cat}</div>
      <div style="font-size:16px;font-weight:800;color:#16130A;line-height:1.2">{esc(r.get('headline'))}</div></div>
  </div>
  <div style="padding:15px 16px 17px">
    <div style="font-size:13.5px;line-height:1.55;color:#44443B;margin-bottom:13px">{esc(r.get('body'))}</div>
    <div style="display:flex;flex-direction:column;gap:8px;margin-bottom:15px">
      {(f'''<div style="display:flex;align-items:center;gap:9px;background:#E7F6F6;border:1px solid #BFE6E6;
        border-radius:11px;padding:9px 12px"><div style="font-size:16px">📦</div>
        <div style="font-size:12.5px;font-weight:700;color:#0B6E6E">{esc(r.get('inspect_line'))}</div></div>'''
        if r.get('inspect_line') else '')}
      {(f'''<div style="display:flex;align-items:center;gap:9px;background:#EFF4FF;border:1px solid #D6E1FB;
        border-radius:11px;padding:9px 12px"><div style="font-size:16px">🛡️</div>
        <div style="font-size:12.5px;font-weight:700;color:#2551C6">{esc(r.get('refund_line'))}</div></div>'''
        if r.get('refund_line') else '')}
      {(f'''<div style="display:flex;align-items:center;gap:9px;background:#EAF7EE;border:1px solid #C4E7CF;
        border-radius:11px;padding:9px 12px"><div style="font-size:16px">🌿</div>
        <div style="font-size:12.5px;font-weight:700;color:#146634">{esc(r.get('fresh_line'))}</div></div>'''
        if r.get('fresh_line') else '')}
      {(f'''<div style="display:flex;align-items:center;gap:9px;background:#F6EFFB;border:1px solid #E2D2F0;
        border-radius:11px;padding:9px 12px"><div style="font-size:16px">💬</div>
        <div style="font-size:12.5px;font-weight:700;color:#6B3FA0">{esc(r.get('social_proof_line'))}</div></div>'''
        if r.get('social_proof_line') else '')}
    </div>
    <div style="display:flex;gap:12px;align-items:center;border:1px solid #EFEFE9;border-radius:14px;
      padding:11px;margin-bottom:15px">
      <div style="width:58px;height:58px;border-radius:11px;background:#FFF3D6;display:flex;
        align-items:center;justify-content:center;font-size:28px;flex:none">{emoji}</div>
      <div style="flex:1;min-width:0">
        <div style="font-size:13.5px;font-weight:700;line-height:1.25;color:#16130A">{esc(r.get('product'))}</div>
        <div style="font-size:11.5px;color:#6B6B60;margin-top:5px;font-weight:600">Illustrative demo item</div>
      </div>
    </div>
    <div style="width:100%;background:#16130A;color:#F8CD1B;font-size:14.5px;font-weight:800;
      padding:14px;border-radius:13px;text-align:center">{esc(r.get('cta'))}</div>
    <div style="text-align:center;font-size:11px;font-weight:600;color:#6B6B60;margin-top:9px">
      First order in this category · {cat}</div>
    {(f'''<div style="margin-top:11px;display:flex;align-items:center;justify-content:center;
      gap:7px;background:#F7F8F5;border:1px dashed #DADBD3;border-radius:11px;padding:9px 12px">
      <span style="font-size:11px;color:#6B6B60;font-weight:600">You can also try</span>
      <span style="font-size:11.5px;color:#16130A;font-weight:800">{esc(secondary.title())}</span>
    </div>''' if secondary else '')}
  </div>
</div>
<div style="text-align:center;font-size:11px;color:#6B6B60;margin-top:16px;font-weight:600">
  Reorder your usuals below ↓</div>"""
    return _phone_shell(p, inner, hint=r.get("suggested_category"))


TOP = """
<div class="nb-top">
  <div class="nb-brand">
    <div class="nb-logo">🛒</div>
    <div><div class="nb-t1">Category Nudge Agent</div>
    <div class="nb-t2">Part 4 MVP · nudges a repeat buyer toward one new category</div></div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <div class="nb-pill" style="background:#FFF6D6;border:1px solid #F1DE8E;color:#7A6100">
      <span class="nb-dot" style="background:#E7B400"></span>Synthetic demo data</div>
    <div class="nb-pill" style="background:#EAF7EE;border:1px solid #BCE6C8;color:#146634">
      <span class="nb-dot" style="background:#1F9D55"></span>Agent online</div>
  </div>
</div>"""


# ----------------------- Feature 1: auto-nudge queue -----------------------
def auto_gate_head(min_tenure=MIN_TENURE_MONTHS):
    """The rule statement + live queued count. Rendered above the tenure slider so the
    rule and the control that changes it sit together, as in the design."""
    min_tenure = int(min_tenure)
    n = len(eligible_profiles(min_tenure_months=min_tenure))
    return f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap">
  <div style="flex:1;min-width:240px">
    <div class="nb-eyebrow">Eligibility gate · deterministic</div>
    <div style="font-size:14px;font-weight:700;color:#16130A;margin-top:5px;line-height:1.45">
      Cadence is <span style="color:#146634">Daily or Weekly</span> &nbsp;AND&nbsp; tenure &gt;
      <span style="color:#146634">{min_tenure} months</span></div>
    <div style="font-size:12.5px;color:#6B6B60;margin-top:6px;line-height:1.5">Habitual, established
      users whose basket has settled — the “stuck segment” from Part 2. No LLM in the gate; the copy
      is still generated per user.</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:30px;font-weight:800;color:#16130A;line-height:1">{n}</div>
    <div style="font-size:11px;font-weight:700;color:#6B6B60;text-transform:uppercase;
      letter-spacing:.05em">queued</div>
  </div>
</div>"""


def auto_funnel_html(min_tenure=MIN_TENURE_MONTHS):
    """The compact 3-stage funnel only (all profiles -> cadence passes -> in queue).

    Split out of the old auto_eligibility_html() so this stays visible at a glance while
    the much longer per-profile breakdown (below) can be tucked into a collapsed
    accordion — the full always-open list of all 8 profiles was the main source of
    clutter in this tab. Every count is still computed live through the real
    deterministic gate (auto_targeting.eligibility_detail).
    """
    min_tenure = int(min_tenure)
    incl = [p for p in PROFILES if eligibility_detail(p, min_tenure)[0]]
    freq_ok = [p for p in PROFILES if (p.get("order_frequency") or "").strip().lower()
               in ELIGIBLE_FREQS]
    stages = [(len(PROFILES), "All profiles", "synthetic user base"),
              (len(freq_ok), "Cadence passes", "Daily or Weekly"),
              (len(incl), "In queue", f"tenure &gt; {min_tenure}mo")]
    total = len(PROFILES) or 1
    funnel = "".join(f"""
      <div style="flex:1;min-width:120px;background:{'#16130A' if i == 2 else '#F7F8F5'};
        border-radius:14px;padding:13px 15px">
        <div style="font-size:24px;font-weight:800;line-height:1;
          color:{YELLOW if i == 2 else INK}">{n}</div>
        <div style="font-size:12px;font-weight:700;margin-top:3px;
          color:{'#CFD0C6' if i == 2 else INK}">{lab}</div>
        <div style="font-size:11px;margin-top:2px;color:{'#8E8F86' if i == 2 else '#6B6B60'}">{sub}</div>
        <div style="height:5px;border-radius:99px;margin-top:10px;
          background:{'rgba(255,255,255,.14)' if i == 2 else '#EDEEE8'};overflow:hidden">
          <div style="height:100%;width:{round(100 * n / total)}%;border-radius:99px;
            background:{YELLOW};animation:barGrow .7s cubic-bezier(.2,.7,.3,1)"></div></div>
      </div>""" for i, (n, lab, sub) in enumerate(stages))
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap">{funnel}</div>'


def auto_profile_list_html(min_tenure=MIN_TENURE_MONTHS):
    """The full in-queue / held-back per-profile breakdown, meant to sit inside a
    collapsed accordion (see the tab layout) rather than always on screen."""
    min_tenure = int(min_tenure)
    incl, excl = [], []
    for p in PROFILES:
        ok, why = eligibility_detail(p, min_tenure)
        (incl if ok else excl).append((p, why))

    def row(p, why, ok):
        """One profile row in the eligibility split.

        The two sides carry very different payloads, so they get different layouts.
        An in-queue row's chips are two short tokens ("Daily ✓", "16mo ✓") and sit
        happily beside the name. A held-back row's payload is a full sentence
        explaining the exclusion — pinning that beside the name (as a single shared
        layout did) left ~120px for the name column inside the half-width card, so
        names wrapped and the meta line broke to roughly one word per line.
        The reason therefore gets its own full-width row underneath instead.
        """
        months = parse_tenure_months(p.get("tenure"))
        # Name + cadence/tenure. Clipped rather than wrapped — these are short strings
        # and an ellipsis reads better than a two-line name in a dense list.
        ident = (f'<div style="flex:1;min-width:0">'
                 f'<div style="font-size:13px;font-weight:700;color:#16130A;white-space:nowrap;'
                 f'overflow:hidden;text-overflow:ellipsis">{esc(p["display_name"])}</div>'
                 f'<div style="font-size:11.5px;color:#6B6B60;white-space:nowrap;overflow:hidden;'
                 f'text-overflow:ellipsis">{esc(p.get("order_frequency"))} · '
                 f'{esc(p.get("tenure"))}</div></div>')
        avatar = (f'<div style="width:30px;height:30px;border-radius:9px;flex:none;display:flex;'
                  f'align-items:center;justify-content:center;font-weight:800;font-size:12px;'
                  f'background:{p["avatar_bg"]};color:#16130A">{esc(p["initials"])}</div>')

        if ok:
            chips = (f'<div style="display:flex;gap:5px;flex:none">'
                     f'<span style="font-size:10.5px;font-weight:700;color:#146634;background:#EAF7EE;'
                     f'border-radius:20px;padding:3px 9px;white-space:nowrap">'
                     f'{esc(p.get("order_frequency"))} ✓</span>'
                     f'<span style="font-size:10.5px;font-weight:700;color:#146634;background:#EAF7EE;'
                     f'border-radius:20px;padding:3px 9px;white-space:nowrap">{months}mo ✓</span></div>')
            return (f'<div style="display:flex;align-items:center;gap:10px;padding:9px 11px;'
                    f'background:#fff;border:1px solid #D8EEDF;border-radius:12px;margin-bottom:8px">'
                    f'{avatar}{ident}{chips}</div>')

        # Held back: reason on its own line, left-aligned, free to use the full card width.
        reason = esc(why.replace("Excluded: ", ""))
        return (f'<div style="padding:9px 11px;background:#fff;border:1px solid #EFEFE9;'
                f'border-radius:12px;margin-bottom:8px">'
                f'<div style="display:flex;align-items:center;gap:10px">{avatar}{ident}</div>'
                f'<div style="margin-top:8px;font-size:11px;font-weight:600;color:#7A6100;'
                f'background:#FBF4CE;border-radius:9px;padding:6px 10px;line-height:1.45">'
                f'{reason}</div></div>')

    inc_html = "".join(row(p, w, True) for p, w in incl) or \
        '<div style="font-size:12.5px;color:#6B6B60">No profile passes this threshold.</div>'
    exc_html = "".join(row(p, w, False) for p, w in excl) or \
        '<div style="font-size:12.5px;color:#6B6B60">Nobody held back at this threshold.</div>'

    return f"""
<div style="display:flex;gap:14px;flex-wrap:wrap">
  <div style="flex:1;min-width:250px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <span style="width:8px;height:8px;border-radius:50%;background:#1F9D55"></span>
      <div style="font-size:12px;font-weight:800;color:#16130A">In queue · {len(incl)}</div></div>
    {inc_html}
  </div>
  <div style="flex:1;min-width:250px">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
      <span style="width:8px;height:8px;border-radius:50%;background:#C4C5B9"></span>
      <div style="font-size:12px;font-weight:800;color:#16130A">Held back · {len(excl)}</div></div>
    {exc_html}
  </div>
</div>
<div style="display:flex;gap:10px;align-items:flex-start;margin-top:16px;background:#FFF9E3;
  border:1px dashed #EBD68A;border-radius:13px;padding:12px 14px">
  <div style="font-size:16px">⚠️</div>
  <div style="font-size:12px;color:#6F5700;line-height:1.5">Simulation only — no push notification
    leaves this demo. There is no live user base or device registry behind the queue.</div>
</div>"""


def _notification_card(p, notif, nudge=None):
    """One push payload, styled as an iOS-style lock-screen notification.

    The bubble contains ONLY what a real notification contains — app icon, app name,
    timestamp, title, body. The operator's routing info (who it went to, which category,
    the trust driver behind the pick) sits OUTSIDE the bubble as captions on the
    lock-screen background — no real notification carries its own targeting metadata, so
    keeping it out of the bubble avoids the preview reading as a debug view instead of a
    phone. `nudge` (the full generate_nudge() result, optional) supplies refund_line for
    the trust-driver caption; falls back to just the category badge if not given.
    """
    cat_label = str(notif.get("category") or "").title()
    badge = (
        f'<div style="margin-top:8px"><span style="font-size:10px;font-weight:800;'
        f'color:#16130A;background:{YELLOW};border-radius:20px;padding:4px 10px;'
        f'white-space:nowrap">🆕 New pick · {esc(cat_label)}</span></div>'
        if cat_label else ''
    )
    refund_line = (nudge or {}).get("refund_line") or ""
    trust = (
        f'<div style="margin-top:5px;font-size:10.5px;color:rgba(255,255,255,.62);'
        f'line-height:1.4">💯 {esc(refund_line)}</div>'
        if refund_line else ''
    )
    return (
        '<div style="margin-bottom:14px">'
        '<div class="nb-pop" style="background:rgba(255,255,255,.94);border-radius:17px;'
        'padding:11px 13px;box-shadow:0 4px 16px rgba(0,0,0,.16)">'
        '<div style="display:flex;align-items:center;gap:7px;margin-bottom:5px">'
        '<div style="width:19px;height:19px;border-radius:5px;background:#F8CD1B;display:flex;'
        f'align-items:center;justify-content:center;font-size:11px">🛒</div>'
        '<div style="font-size:10.5px;font-weight:700;color:#6B6B60;letter-spacing:.02em">Blinkit</div>'
        f'<div style="flex:1"></div><div style="font-size:10.5px;color:#8A8A7C">now</div></div>'
        f'<div style="font-size:12.5px;font-weight:700;color:#16130A;line-height:1.32">'
        f'{esc(notif.get("emoji", "🛒"))} {esc(notif.get("title"))}</div>'
        f'<div style="font-size:11.5px;color:#3C3C34;line-height:1.42;margin-top:2px">'
        f'{esc(notif.get("body"))}</div>'
        '</div>'
        f'{badge}{trust}'
        f'<div style="font-size:9.5px;font-weight:600;color:rgba(255,255,255,.45);'
        f'padding:4px 6px 0;letter-spacing:.02em">For {esc(p["display_name"])}</div>'
        '</div>')


def _coverage_strip(used):
    """All suggestable categories, with the ones this run actually nudged marked.

    Five eligible profiles cannot cover eleven categories in a single batch, and
    inventing extra recipients to force full coverage would fabricate demo data. So the
    full pool is shown instead, with this run's picks lit — honest about the gap while
    still surfacing every category the agent can choose from.
    """
    if not used:
        return ""
    used_n = {str(c).lower().strip() for c in used}
    chips = "".join(
        f'<span style="font-size:10px;font-weight:{"800" if c.lower() in used_n else "600"};'
        f'padding:4px 9px;border-radius:20px;white-space:nowrap;'
        f'background:{"#F8CD1B" if c.lower() in used_n else "#F2F3EE"};'
        f'color:{"#16130A" if c.lower() in used_n else "#8A8A7C"};'
        f'border:1px solid {"#E7B400" if c.lower() in used_n else "#E7E8E2"}">'
        f'{"● " if c.lower() in used_n else ""}{esc(c)}</span>'
        for c in SUGGESTABLE_CATEGORIES)
    return f"""
<div style="max-width:340px;margin:14px auto 0;padding-top:13px;border-top:1px solid #E7E8E2">
  <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:8px">
    <div class="nb-eyebrow" style="font-size:10.5px">Category coverage</div>
    <div style="font-size:10.5px;font-weight:700;color:#6B6B60">
      {len(used_n)} of {len(SUGGESTABLE_CATEGORIES)} this run</div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:5px">{chips}</div>
  <div style="font-size:10.5px;color:#6B6B60;line-height:1.5;margin-top:9px">
    Each queued user gets a <b>different</b> never-bought category, so one batch spans
    {len(used_n)} distinct shelves rather than repeating the same one. The unlit chips are
    reachable for other users — a batch of {len(used_n)} cannot cover all
    {len(SUGGESTABLE_CATEGORIES)} without inventing recipients.</div>
</div>"""


def _lock_screen(inner, queued, total, used_categories=None, footer=None):
    coverage = _coverage_strip(used_categories or [])
    footer = footer or f"{queued} of {total} profiles nudged · each payload is shaped from that user's generated nudge"
    return f"""
<div class="nb-eyebrow" style="text-align:center;margin-bottom:12px">Push payload preview</div>
<div style="width:320px;margin:0 auto;background:#0E0E0C;border-radius:42px;padding:11px;
  box-shadow:0 24px 60px rgba(0,0,0,.22)">
  <div style="border-radius:32px;overflow:hidden;background:linear-gradient(160deg,#3A4256,#1D2130 55%,#12141C);
    padding:14px 12px 18px;min-height:660px;display:flex;flex-direction:column">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:0 10px 6px;
      font-size:11.5px;font-weight:700;color:#fff"><span style="color:#fff">9:41</span><span style="color:#fff">📶 &nbsp;🔋</span></div>
    <div style="text-align:center;margin:14px 0 18px">
      <div style="font-size:12.5px;font-weight:600;color:rgba(255,255,255,.82)">Saturday, 26 July</div>
      <div style="font-size:52px;font-weight:300;color:#fff;line-height:1.05;letter-spacing:-.02em">9:41</div>
    </div>
    <div style="flex:1">{inner}</div>
    <div style="width:120px;height:5px;border-radius:99px;background:rgba(255,255,255,.35);
      margin:18px auto 2px"></div>
  </div>
</div>
<div style="text-align:center;font-size:11.5px;font-weight:600;color:#6B6B60;margin-top:12px">
  {footer}</div>
{coverage}"""


def _assign_distinct_categories(queue):
    """Give every queued user a DIFFERENT genuinely-adjacent category.

    Without this, several users in one batch land on the same category — the synthetic
    profiles have overlapping baskets, so "packaged gourmet foods" would win twice in a
    five-user run and the preview looked repetitive.

    Relevance still outranks variety: a user is only moved off a category if they have
    another candidate with a real adjacency score (> 0). The scarcest user goes first
    (fewest qualifying options), so a user with one viable category isn't left stranded
    by someone who had three to choose from. If a user genuinely has no untaken candidate,
    they keep their best one and the repeat is allowed rather than forcing an irrelevant
    suggestion — variety is never worth a nudge that doesn't fit.

    Deterministic: same queue in, same assignment out.
    """
    options = {}
    for p in queue:
        scores = adjacency_scores(p)
        pool = rank_suggestable_categories(p, top_n=len(SUGGESTABLE_CATEGORIES))
        options[p["user_id"]] = [c for c in pool if scores.get(c, 0) > 0] or pool

    assigned, taken = {}, set()
    for uid in sorted(options, key=lambda u: (len(options[u]), u)):
        pick = next((c for c in options[uid] if c not in taken), None)
        if pick is None:                      # every candidate already used
            pick = options[uid][0] if options[uid] else None
        if pick:
            assigned[uid] = pick
            taken.add(pick)
    return assigned


def on_run_auto_batch(min_tenure=MIN_TENURE_MONTHS, uid=None):
    """Preview the push payload for ONE selected synthetic user.

    Previously this ran the whole eligible queue at once and stacked every resulting
    notification onto a single lock screen (matching the deck's slide-8 screenshot) —
    realistic for "here's the batch," but it meant you could never see what one specific
    user's own notification looked like without scanning a pile of unrelated cards.
    `uid` scopes the run to just that user: the eligibility GATE still applies (this
    demonstrates Feature 1's targeting rule, so a user who doesn't qualify at the current
    tenure threshold correctly shows as not-nudged, same as being absent from the old
    batch view), but the lock screen renders at most one card.
    """
    total = len(PROFILES)
    queue = eligible_profiles(min_tenure_months=int(min_tenure))
    if "GROQ_API_KEY" not in os.environ:
        return ('<div class="nb-dark"><div class="k">Error</div><div style="margin-top:8px;font-size:13px">'
                'GROQ_API_KEY is not configured on the server.</div></div>')
    if uid:
        p = BY_ID.get(uid)
        queue = [x for x in queue if x["user_id"] == uid]
        if not queue:
            _, why = eligibility_detail(p, int(min_tenure)) if p else (False, "Unknown user.")
            name = esc(p["display_name"]) if p else esc(uid)
            return _lock_screen(
                f'<div style="background:rgba(255,255,255,.12);border-radius:15px;padding:16px;'
                f'font-size:12px;color:rgba(255,255,255,.85);text-align:center;line-height:1.5">'
                f'<b style="color:#fff">{name}</b> would not receive an auto-nudge at this tenure '
                f'threshold — {esc(why)}. Lower the threshold or pick a different user.</div>',
                0, total)
    if not queue:
        return _lock_screen(
            '<div style="background:rgba(255,255,255,.12);border-radius:15px;padding:16px;'
            'font-size:12px;color:rgba(255,255,255,.8);text-align:center;line-height:1.5">'
            'No profile passes the gate at this tenure threshold — nothing would be sent.</div>',
            0, total)

    assigned = _assign_distinct_categories(queue)
    # Spread push_title hook angles round-robin across the batch. Each nudge is an
    # independent Groq call that cannot see the others, so without this they converge on
    # one phrasing; hashing per user collides at this batch size, so assign by position.
    angles = {p["user_id"]: PUSH_ANGLES[i % len(PUSH_ANGLES)] for i, p in enumerate(queue)}

    def _gen(p):
        theme, reason = match_profile_to_theme(p, THEMES)
        # REAL Groq call, pinned to this user's pre-assigned category so the batch spans
        # distinct shelves instead of several users landing on the same one.
        return generate_nudge(p, theme, reason, forced_category=assigned.get(p["user_id"]),
                              push_angle=angles.get(p["user_id"]))

    cards, used = "", []
    with ThreadPoolExecutor(max_workers=min(8, len(queue))) as pool:
        futures = {pool.submit(_gen, p): p for p in queue}
        results = {}
        for fut in futures:
            p = futures[fut]
            try:
                results[p["user_id"]] = ("ok", fut.result())
            except Exception as e:  # noqa: BLE001
                results[p["user_id"]] = ("error", e)
    for p in queue:
        kind, payload = results[p["user_id"]]
        if kind == "error":
            cards += (f'<div style="background:rgba(255,255,255,.9);border-radius:15px;padding:11px;'
                      f'font-size:11.5px;color:#B00;margin-bottom:9px">Failed for '
                      f'{esc(p["display_name"])}: {esc(payload)}</div>')
        else:
            notif = to_notification(payload)
            cards += _notification_card(p, notif, nudge=payload)
            if notif.get("category"):
                used.append(notif["category"])
    footer = (f"Previewing the push {esc(queue[0]['display_name'])} would receive · "
              f"not part of a batch send") if uid else None
    return _lock_screen(cards, len(queue), total, used_categories=used, footer=footer)


# ----------------------- Feature 2: checkout cart-filler -----------------------
def cart_phone_html(uid, cart_total):
    """The design's checkout screen. Cart contents are shown as the user's OWN habitual
    categories (real profile field) with a single subtotal, rather than inventing
    line-item products and prices that don't exist."""
    p = BY_ID[uid]
    cart_total = int(cart_total)
    res = suggest_fillers(p, cart_total)
    thr, gap = res["threshold"], res["gap"]
    pct = min(100, round(cart_total / thr * 100)) if thr else 100
    qualifies = res["qualifies"]

    # Real line items that add up to the operator's chosen cart total. Previously these
    # were three unpriced "<Category> / usual item" placeholders sitting above an Item
    # total that nothing on screen explained.
    basket = cart_lines(p, cart_total)
    lines = "".join(f"""
      <div style="display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid #F2F2EC">
        <div style="width:34px;height:34px;border-radius:9px;background:#F7F8F5;display:flex;
          align-items:center;justify-content:center;font-size:16px;flex:none">{STAPLE_EMOJI.get(it['category'], '🛒')}</div>
        <div style="flex:1;min-width:0">
          <div style="font-size:12.5px;font-weight:700;color:#16130A;white-space:nowrap;
            overflow:hidden;text-overflow:ellipsis">{esc(it['name'])}</div>
          <div style="font-size:11px;color:#6B6B60">{esc(it['category'])} · ₹{it['price']}
            {f"× {it['qty']}" if it['qty'] > 1 else ""}</div>
        </div>
        <div style="font-size:12.5px;font-weight:700;color:#16130A;flex:none">₹{it['subtotal']}</div>
      </div>""" for it in basket["lines"])
    if basket["remainder"] > 0:
        lines += (
            '<div style="display:flex;align-items:center;gap:10px;padding:9px 0;'
            'border-bottom:1px solid #F2F2EC">'
            '<div style="width:34px;height:34px;border-radius:9px;background:#F7F8F5;display:flex;'
            'align-items:center;justify-content:center;font-size:14px;flex:none;color:#8A8A7C">＋</div>'
            '<div style="flex:1;min-width:0">'
            '<div style="font-size:12.5px;font-weight:600;color:#6B6B60">Other items in this cart</div>'
            '<div style="font-size:11px;color:#8A8A7C">not itemised in this demo</div></div>'
            f'<div style="font-size:12.5px;font-weight:700;color:#6B6B60;flex:none">'
            f'₹{basket["remainder"]}</div></div>')
    if not basket["lines"] and basket["remainder"] <= 0:
        lines = ('<div style="padding:18px 0;text-align:center;font-size:12px;color:#6B6B60">'
                 'Cart is empty — drag the slider to add items.</div>')

    if qualifies:
        band = ('#EAF7EE', '#C4E7CF', '#146634', '✓',
                'Free delivery unlocked — no filler needed.')
    else:
        band = ('#FFF9E3', '#EBD68A', '#6F5700', '🚚',
                f'Add ₹{gap} more to unlock free delivery')
    threshold_band = f"""
      <div style="background:{band[0]};border:1px solid {band[1]};border-radius:13px;padding:11px 13px;
        margin:12px 0">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
          <div style="font-size:15px">{band[3]}</div>
          <div style="font-size:12.5px;font-weight:700;color:{band[2]}">{band[4]}</div></div>
        <div style="height:7px;background:rgba(0,0,0,.07);border-radius:99px;overflow:hidden">
          <div style="height:100%;width:{pct}%;border-radius:99px;background:#F8CD1B;
            animation:barGrow .7s cubic-bezier(.2,.7,.3,1)"></div></div>
        <div style="display:flex;justify-content:space-between;font-size:10.5px;font-weight:600;
          color:#6B6B60;margin-top:5px"><span>₹{cart_total}</span>
          <span>free delivery at ₹{thr}</span></div>
      </div>"""

    def _filler_card(it):
        badges = []
        if it["covers_gap"]:
            badges.append('<span style="font-size:9.5px;font-weight:800;color:#146634;'
                           'background:#EAF7EE;border-radius:20px;padding:2px 8px;white-space:nowrap">'
                           'Unlocks free delivery</span>')
        badge_row = (f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:5px">'
                     f'{"".join(badges)}</div>' if badges else '')
        return f"""
          <div style="display:flex;align-items:center;gap:10px;background:#fff;border:1px solid #EFEFE9;
            border-radius:13px;padding:10px 11px;margin-bottom:8px">
            <div style="width:38px;height:38px;border-radius:10px;background:#FFF3D6;display:flex;
              align-items:center;justify-content:center;font-size:18px;flex:none">🛍️</div>
            <div style="flex:1;min-width:0">
              <div style="font-size:10px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;
                color:#2551C6">Your first order in {esc(it["category"])}</div>
              <div style="font-size:12.5px;font-weight:700;color:#16130A;line-height:1.25">{esc(it["name"])}</div>
              <div style="font-size:10.5px;color:#6B6B60;margin-top:2px">Never bought before ·
                zero-risk ₹{it["price"]} trial</div>
              {badge_row}
            </div>
            <div style="text-align:right;flex:none">
              <div style="font-size:13px;font-weight:800;color:#16130A">₹{it["price"]}</div>
              <div style="font-size:10.5px;font-weight:800;color:#F8CD1B;background:#16130A;
                border-radius:6px;padding:3px 9px;margin-top:4px">ADD</div></div>
          </div>"""

    fillers = ""
    if res["items"]:
        cards = "".join(_filler_card(it) for it in res["items"])
        fillers = (f'<div style="font-size:11.5px;font-weight:800;color:#16130A;margin:14px 0 8px">'
                   f'Add one of these to unlock it</div>{cards}')

    fee = "FREE" if qualifies else f"₹{DELIVERY_FEE}"
    payable = cart_total if qualifies else cart_total + DELIVERY_FEE
    inner = f"""
<div style="display:flex;align-items:center;gap:10px;padding:4px 2px 10px">
  <span style="font-size:16px;color:#16130A">←</span>
  <div style="font-size:14px;font-weight:800;color:#16130A">Your cart</div>
  <div style="flex:1"></div>
  <div style="font-size:11px;font-weight:700;color:#146634;background:#EAF7EE;border-radius:20px;
    padding:3px 9px">8 min</div>
</div>
{lines}
{threshold_band}
{fillers}
<div style="border-top:1px solid #F2F2EC;margin-top:12px;padding-top:11px">
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#44443B;margin-bottom:5px">
    <span>Item total</span><span style="font-weight:700">₹{cart_total}</span></div>
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#44443B;margin-bottom:8px">
    <span>Delivery fee</span>
    <span style="font-weight:800;color:{'#146634' if qualifies else '#44443B'}">{fee}</span></div>
  <div style="display:flex;justify-content:space-between;font-size:13.5px;font-weight:800;color:#16130A">
    <span>To pay</span><span>₹{payable}</span></div>
  <div style="width:100%;background:#16130A;color:#F8CD1B;font-size:13.5px;font-weight:800;
    padding:12px;border-radius:12px;text-align:center;margin-top:11px">Place order · ₹{payable}</div>
</div>"""

    return f"""
<div class="nb-eyebrow" style="text-align:center;margin-bottom:12px">Checkout screen</div>
<div class="nb-phone" style="margin:0 auto"><div class="nb-screen">
  <div style="display:flex;align-items:center;justify-content:space-between;padding:11px 24px 6px;
    font-size:12px;font-weight:700;color:#16130A"><span>9:41</span><span>📶 &nbsp;🔋</span></div>
  <div style="padding:8px 15px 20px;background:#fff">{inner}</div>
</div></div>
<div style="text-align:center;font-size:11.5px;color:#6B6B60;margin-top:12px;line-height:1.5;
  max-width:300px;margin-left:auto;margin-right:auto">Cart shows this shopper's own habitual
  categories. Synthetic demo catalog; the threshold, fee and prices are illustrative.</div>"""


def cart_logic_html(uid, cart_total):
    """'How the filler is chosen' + the full ranked candidate pool — both real:
    the rules describe the deterministic selection in cart_filler.suggest_fillers, and
    the pool is never_bought_categories() ranked by the same basket-adjacency ranker."""
    p = BY_ID[uid]
    cart_total = int(cart_total)
    res = suggest_fillers(p, cart_total)
    gap = res["gap"]
    scores = adjacency_scores(p)
    anchor = primary_suggested_category(p)

    rules = [
        ("Never-bought only", "Candidate categories exclude everything already in this user's basket."),
        ("Ranked by basket adjacency", "The same deterministic ranker the nudge agent uses, so the add-on still feels relevant."),
        ("Coordinated with the push queue", f"Defaults to the same category the push-nudge agent is instructed to prefer"
         f"{f' — <b>{esc(anchor)}</b> for this user' if anchor else ''} — so the two mechanics point at one next category, not two."),
        ("Closes the gap in one add", f"Items priced at or above the ₹{gap} gap come first, cheapest of those first, "
         "unless the anchor category can also close it."),
        ("One item per category", "The carousel spans distinct new categories rather than three of the same."),
    ]
    rule_html = "".join(f"""
      <div style="display:flex;gap:11px;align-items:flex-start;margin-bottom:11px">
        <div style="width:21px;height:21px;border-radius:7px;background:#F8CD1B;color:#16130A;
          font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center;
          flex:none;margin-top:1px">{i}</div>
        <div><div style="font-size:12.5px;font-weight:700;color:#16130A">{t}</div>
        <div style="font-size:11.5px;color:#6B6B60;line-height:1.45">{b}</div></div>
      </div>""" for i, (t, b) in enumerate(rules, 1))

    # never_bought_categories() applies the ranker's per-user rotation, so its raw order is
    # not monotonic in adjacency. Sort the *display* by weight so the "ranked by basket
    # adjacency" label is literally true; the offered three are still whatever
    # suggest_fillers() actually chose (gap-coverage first — see rule 3).
    pool = sorted(never_bought_categories(p), key=lambda c: (-scores.get(c, 0), c))
    offered = {it["category"] for it in res["items"]}
    pool_html = "".join(f"""
      <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #F2F2EC">
        <div style="font-size:12.5px;font-weight:{'800' if c in offered else '600'};
          color:{'#16130A' if c in offered else '#6B6B60'};flex:1;min-width:0">
          {'🔗 ' if c == anchor else ''}{esc(c)}</div>
        <div style="font-size:11.5px;font-weight:700;color:#6B6B60;flex:none">adjacency {scores.get(c, 0)}</div>
        <div style="font-size:9.5px;font-weight:800;letter-spacing:.05em;border-radius:5px;padding:3px 7px;
          width:74px;text-align:center;flex:none;
          background:{'#F8CD1B' if c in offered else '#F2F3EE'};
          color:{'#16130A' if c in offered else '#8A8A7C'}">{'OFFERED' if c in offered else 'in pool'}</div>
      </div>""" for c in pool) or '<div style="font-size:12.5px;color:#6B6B60">No never-bought category left.</div>'

    stats = [("Gap", "—" if res["qualifies"] else f"₹{gap}"),
             ("Delivery saved", f"₹{DELIVERY_FEE}"),
             ("Fillers offered", str(len(res["items"])))]
    stat_html = "".join(f"""
      <div class="nb-tile"><div class="nb-tile-k">{k}</div>
      <div class="nb-tile-v" style="font-size:16px">{v}</div></div>""" for k, v in stats)

    return f"""
<div style="display:flex;gap:10px;margin-bottom:18px">{stat_html}</div>
<div class="nb-eyebrow" style="margin-bottom:11px">How the filler is chosen</div>
{rule_html}
<div style="display:flex;gap:10px;align-items:flex-start;background:#FFF9E3;border:1px dashed #EBD68A;
  border-radius:13px;padding:12px 14px;margin:4px 0 18px">
  <div style="font-size:16px">🎯</div>
  <div style="font-size:11.5px;color:#6F5700;line-height:1.5">Complements the push nudge rather than
    repeating it: the queue targets high-intent habituals, while a ₹60 top-up needs
    <b>no pre-existing intent at all</b> — it reaches the low-intent segment the notification
    deliberately skips.</div>
</div>
<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:4px">
  <div class="nb-eyebrow">Candidate pool · {esc(p['display_name'])}</div>
  <div style="font-size:11px;font-weight:600;color:#6B6B60">ranked by basket adjacency</div>
</div>
<div style="font-size:11.5px;color:#6B6B60;margin-bottom:10px;line-height:1.45">Every candidate is from
  a category this user has never purchased. Top {len(res['items'])} shown in the app, one per category.
  {'🔗 marks the category the push-nudge agent is instructed to prefer for this user.' if anchor else ''}</div>
{pool_html}"""


def cart_filler_html(uid, cart_total):
    p = BY_ID[uid]
    cart_total = int(cart_total)
    res = suggest_fillers(p, cart_total)
    thr = res["threshold"]
    pct = min(100, int(cart_total / thr * 100)) if thr else 100
    bar = (f'<div style="height:9px;background:#EFEFE9;border-radius:6px;overflow:hidden;margin:9px 0 4px">'
           f'<div style="height:100%;width:{pct}%;background:#F8CD1B"></div></div>')
    header = (
        '<div class="nb-eyebrow">Checkout · free-delivery unlock</div>'
        f'<div style="font-size:13px;color:#6B6B60;margin:6px 0 2px">Cart ₹{cart_total} of ₹{thr} threshold '
        f'· <b style="color:#16130A">{esc(p["display_name"])}</b></div>{bar}')
    if res["qualifies"]:
        return (header + '<div style="margin-top:12px;font-size:13.5px;font-weight:700;color:#146634">'
                '✓ Already over the threshold — free delivery unlocked, no filler needed.</div>')
    items = ""
    for it in res["items"]:
        badge = ('<div style="font-size:10.5px;font-weight:800;color:#146634;background:#EAF7EE;'
                 'border-radius:20px;padding:3px 9px;white-space:nowrap">Unlocks free delivery</div>'
                 if it["covers_gap"] else '')
        items += (
            '<div style="display:flex;align-items:center;gap:12px;background:#fff;border:1px solid #EFEFE9;'
            'border-radius:13px;padding:11px 13px;margin-bottom:9px">'
            '<div style="width:44px;height:44px;border-radius:10px;background:#FFF3D6;display:flex;'
            'align-items:center;justify-content:center;font-size:20px;flex:none">🛒</div>'
            '<div style="flex:1;min-width:0"><div style="font-size:13.5px;font-weight:700;color:#16130A">'
            f'{esc(it["name"])} · <span style="color:#6B6B60">₹{it["price"]}</span></div>'
            f'<div style="font-size:11.5px;color:#2551C6;font-weight:600;margin-top:3px">'
            f'First time in {esc(it["category"])}</div></div>{badge}</div>')
    tip = (f'<div style="font-size:13.5px;font-weight:700;color:#16130A;margin:12px 0 10px">'
           f'Add ₹{res["gap"]} more for FREE delivery — try a new category:</div>')
    return (header + tip + items +
            '<div style="margin-top:6px;font-size:11.5px;color:#6B6B60">Every item is from a category this '
            'user has never bought. Synthetic demo catalog; prices illustrative.</div>')


def about_html():
    """The traceability story + the real-vs-synthetic ledger + honest limitations —
    surfaced inside the app itself (not just README) since an evaluator opening the
    live Space link will never see the GitHub README."""
    chain = [
        ("Part 1", "Discovery", "Two-Step Gated extraction over 1,094 gate-passing extractions — real, locked evidence counts."),
        ("Part 2", "Survey", "N=31 Phase 3 survey (frozen 2026-07-28); 18/31 (58%) self-reported “mostly stick to the same categories.”"),
        ("Part 3", "Problem statement", "~70% of category stagnation is quality/trust-driven (770/1,094); the rest is low-intent."),
        ("Part 4", "This agent", "Matches a user to the closest friction theme, then an LLM writes the nudge — this MVP."),
    ]
    chain_html = "".join(f"""
      <div style="flex:1;min-width:150px;background:#F7F8F5;border-radius:14px;padding:13px 15px">
        <div style="font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;color:#146634">{esc(a)}</div>
        <div style="font-size:14px;font-weight:800;color:#16130A;margin:3px 0 6px">{esc(b)}</div>
        <div style="font-size:11.5px;color:#6B6B60;line-height:1.45">{esc(c)}</div>
      </div>{'<div style="display:flex;align-items:center;color:#C4C5B9;font-size:18px;padding:0 2px">→</div>' if i < len(chain) else ''}"""
        for i, (a, b, c) in enumerate(chain, 1))

    ledger = [
        ("Friction theme evidence share (70%, 770/1,094)", "REAL", "Locked Part 1 numbers — shown verbatim, never an LLM-invented confidence score."),
        ("Trust-driver ranking (refund #1, freshness #2)", "REAL", "Q15 survey result (13/31, 10/31 picks) — fixed as the agent's rule 1, not a per-user guess."),
        ("Friction matching (which theme a user gets)", "REAL LOGIC", "Deterministic rules in friction_matching.py; no LLM involved in the match itself."),
        ("Candidate category ranking (“also considered”)", "REAL LOGIC", "Integer basket-adjacency weights, summed and shown as-is — not the 0.88/0.61/0.44-style confidence scores the original mockup invented."),
        ("Nudge copy, headline, reasoning", "LIVE LLM", "Generated per user per click by Groq llama-3.3-70b-versatile — not looked up from a fixed table."),
        ("User profiles (8 personas)", "SYNTHETIC", "Hand-authored to instantiate real survey patterns; no real Blinkit customer data was available."),
        ("Holdout-test lift / conversion figures", "REMOVED", "The imported mockup had fabricated +34% lift / p&lt;0.05 numbers. No experiment has run — replaced with a measurement plan, no invented figures."),
        ("Outcome-tracker funnel (7d, 41% repurchase)", "REMOVED", "Also fabricated in the mockup — replaced with the real event spec, explicitly labeled “no live data.”"),
        ("Catalog prices, delivery fee, threshold", "ILLUSTRATIVE", "Stated demo constants for the cart-filler tab; not claimed or measured Blinkit figures."),
    ]
    ledger_html = "".join(f"""
      <div style="display:flex;align-items:flex-start;gap:11px;padding:9px 0;border-bottom:1px solid #F2F2EC">
        <div style="font-size:9.5px;font-weight:800;letter-spacing:.05em;border-radius:5px;padding:3px 8px;
          width:88px;text-align:center;flex:none;margin-top:1px;
          background:{'#EAF7EE' if tag in ('REAL','REAL LOGIC') else '#FFF3D6' if tag=='LIVE LLM' else '#F3F4F0' if tag=='ILLUSTRATIVE' else '#FBE9E9'};
          color:{'#146634' if tag in ('REAL','REAL LOGIC') else '#7A6100' if tag=='LIVE LLM' else '#6B6B60' if tag=='ILLUSTRATIVE' else '#B23B3B'}">{tag}</div>
        <div style="flex:1;min-width:0">
          <div style="font-size:12.5px;font-weight:700;color:#16130A">{esc(item)}</div>
          <div style="font-size:11.5px;color:#6B6B60;line-height:1.45;margin-top:2px">{esc(note)}</div>
        </div>
      </div>""" for item, tag, note in ledger)

    limits = [
        "Cohort is 8 hand-authored synthetic profiles, not a real user base — the matching rules and ranker are built to generalize, but have only ever been exercised against these 8.",
        "No experiment has been run against real users. The measurement plan and instrumentation spec describe how it WOULD be validated; every number on those two panels would be invented if shown today.",
        "Single nudge mechanic in scope: refund guarantee + quality signal (+ numbers-free peer proof), matching only the ~70% quality/trust-driven share of stagnation. The low-intent ~30% gets a deliberately softer, honestly-scoped nudge — this MVP does not claim to fix that half.",
        "Deferred backlog: the pre-acceptance-inspection trust driver, and any richer targeting beyond the Daily/Weekly + tenure gate, are out of scope for this MVP and would be the next slice.",
        "LLM output is not schema-validated beyond response_format=json_object — a malformed or missing field degrades gracefully in the UI (rows hide rather than render empty) but isn't retried or re-prompted.",
    ]
    limits_html = "".join(f"""
      <div style="display:flex;gap:10px;align-items:flex-start;margin-bottom:10px">
        <div style="width:6px;height:6px;border-radius:50%;background:#E7B400;flex:none;margin-top:6px"></div>
        <div style="font-size:12.5px;color:#44443B;line-height:1.55">{esc(t)}</div>
      </div>""" for t in limits)

    return f"""
<div class="nb-card">
  <div class="nb-eyebrow" style="margin-bottom:6px">Traceability</div>
  <div style="font-size:16px;font-weight:800;color:#16130A;margin-bottom:14px">Part 1 → Part 4, in one thread</div>
  <div style="display:flex;align-items:stretch;gap:8px;flex-wrap:wrap">{chain_html}</div>
</div>
<div class="nb-card">
  <div class="nb-eyebrow" style="margin-bottom:6px">Real vs. synthetic vs. removed</div>
  <div style="font-size:16px;font-weight:800;color:#16130A;margin-bottom:4px">What's actually backing each number on screen</div>
  <div style="font-size:12.5px;color:#6B6B60;margin-bottom:12px;line-height:1.5">Two panels in an earlier imported design
    (a holdout-test result and a live outcome funnel) carried fabricated figures implying this MVP had already been
    validated. It hasn't — they were rebuilt as honest specs rather than shipped.</div>
  {ledger_html}
</div>
<div class="nb-card">
  <div class="nb-eyebrow" style="margin-bottom:6px">Known limitations · what's next</div>
  <div style="font-size:16px;font-weight:800;color:#16130A;margin-bottom:14px">Where this MVP deliberately stops</div>
  {limits_html}
</div>"""


# ----------------------------- app -----------------------------
def on_select(uid):
    p = BY_ID[uid]
    theme, reason = match_profile_to_theme(p, THEMES)
    chips = [gr.update(variant="primary" if x["user_id"] == uid else "secondary") for x in PROFILES]
    return [uid, profile_html(p, theme, reason), reasoning_html(theme, p=p),
            phone_html(p), measurement_plan_html()] + chips


# Share of single-user "Generate" clicks that explore OUTSIDE the adjacency-ranked pool.
# REPORTED BUG this fixes: "electronics accessories" and "pharmacy" score adjacency from
# only ONE existing category each ("household" and "personal_care"). For any profile
# whose basket lacks that one category — 5 of 8 synthetic profiles, for electronics —
# their score is a hard 0, and there are always >=4 OTHER categories scoring >0, so they
# can never enter the top-4 pool _rotated_category() used to sample from. No amount of
# re-clicking could ever surface them for those users; that was a structural exclusion,
# not something the old "rotate within the top 4" logic could randomize past.
EXPLORATION_RATE = 0.3


def _rotated_category(p):
    """Picks this user's next suggested category, mixing two behaviours:

    - Most of the time (1 - EXPLORATION_RATE): randomly among genuinely-adjacent
      candidates (adjacency score > 0) from the ranked top-4 pool — same as before,
      the ranker's ordinary "relevant variety" behaviour.
    - The rest of the time (EXPLORATION_RATE): uniformly among ALL never-bought
      categories for this user, INCLUDING zero-adjacency ones like electronics
      accessories or pharmacy for a profile that doesn't trigger them. This is the
      only path that gives those categories a real, nonzero chance of being nudged
      for such a profile — without it, repeated clicks only ever reshuffled the same
      already-qualifying 4, so certain categories were nudged for some profiles but
      structurally never for others.

    Falls back to the full pool if nothing scores above 0 (shouldn't happen in
    practice, every basket we have scores something, but kept as a safety net).
    """
    full_pool = rank_suggestable_categories(p, top_n=len(SUGGESTABLE_CATEGORIES))
    if not full_pool:
        return None
    if random.random() < EXPLORATION_RATE:
        return random.choice(full_pool)
    top4_pool = rank_suggestable_categories(p)
    scores = adjacency_scores(p)
    candidates = [c for c in top4_pool if scores.get(c, 0) > 0] or top4_pool
    return random.choice(candidates)


def on_generate(uid):
    p = BY_ID[uid]
    theme, reason = match_profile_to_theme(p, THEMES)
    if "GROQ_API_KEY" not in os.environ:
        err = ('<div class="nb-dark"><div class="k">Error</div>'
               '<div style="margin-top:8px;font-size:13px">GROQ_API_KEY is not configured on the '
               'server. Add it in the Space Settings → Variables and secrets.</div></div>')
        return err, phone_html(p), measurement_plan_html()
    try:
        forced = _rotated_category(p)
        r = generate_nudge(p, theme, reason, forced_category=forced)   # REAL Groq call
    except Exception as e:  # noqa: BLE001
        return (f'<div class="nb-dark"><div class="k">Error</div>'
                f'<div style="margin-top:8px;font-size:13px">{esc(e)}</div></div>',
                phone_html(p), measurement_plan_html())
    return reasoning_html(theme, r, p=p), phone_html(p, r), measurement_plan_html(r)


with gr.Blocks(title="Blinkit Category Nudge Agent", css=CSS, head=FONT_LINK,
               js=FORCE_LIGHT, theme=gr.themes.Base()) as demo:
    gr.HTML(FONT_LINK)
    gr.HTML(TOP)
    sel = gr.State(PROFILES[0]["user_id"])
    _p0 = PROFILES[0]
    _t0, _r0 = match_profile_to_theme(_p0, THEMES)

    with gr.Tabs():
        with gr.Tab("Operator console"):
            with gr.Row(elem_classes="nb-cols"):
                with gr.Column(scale=115):
                    with gr.Column(elem_classes="nb-card"):
                        gr.HTML('<div style="display:flex;align-items:center;justify-content:space-between;'
                                'margin-bottom:14px"><div class="nb-eyebrow">Step 1 · Pick a synthetic user</div>'
                                '<div style="font-size:12px;font-weight:600;color:#6B6B60">Part 2 “stuck segment”</div></div>')
                        chip_btns = []
                        for i in range(0, len(PROFILES), 2):
                            with gr.Row():
                                for p in PROFILES[i:i + 2]:
                                    chip_btns.append(gr.Button(
                                        p["display_name"], elem_id=f"chip-{p['user_id']}",
                                        elem_classes="nb-chip", size="sm",
                                        variant="primary" if p is PROFILES[0] else "secondary"))
                    with gr.Column(elem_classes="nb-card"):
                        prof_out = gr.HTML(profile_html(_p0, _t0, _r0))
                        gen = gr.Button("⚡ Generate nudge", elem_id="gen")
                    reason_out = gr.HTML(reasoning_html(_t0, p=_p0))
                    measure_out = gr.HTML(measurement_plan_html())
                with gr.Column(scale=85):
                    phone_out = gr.HTML(phone_html(_p0))
                    gr.HTML(instrumentation_html())

        with gr.Tab("Auto-nudge queue"):
            with gr.Column(elem_classes="nb-card"):
                gr.HTML('<div class="nb-eyebrow">Blink & Try It · auto-nudge</div>'
                        '<div style="font-size:15px;font-weight:800;color:#16130A;'
                        'margin-top:4px">Pick a user, then preview the push they would receive</div>')
                with gr.Row():
                    auto_user = gr.Dropdown(
                        choices=[(p["display_name"], p["user_id"]) for p in PROFILES],
                        value=PROFILES[0]["user_id"], label="Synthetic user", scale=70)
                    run_batch = gr.Button("▶ Run scheduled batch", elem_id="genbatch", scale=30)
            with gr.Row(elem_classes="nb-cols"):
                with gr.Column(scale=110):
                    with gr.Column(elem_classes="nb-card"):
                        head_out = gr.HTML(auto_gate_head())
                        with gr.Column(elem_id="tenwrap"):
                            tenure = gr.Slider(3, 12, value=MIN_TENURE_MONTHS, step=1,
                                               label="Minimum tenure threshold (months)")
                        funnel_out = gr.HTML(auto_funnel_html())
                        with gr.Accordion("See all 8 profiles — who's in queue and why", open=False):
                            gate_out = gr.HTML(auto_profile_list_html())
                with gr.Column(scale=90):
                    with gr.Column(elem_classes="nb-card"):
                        auto_out = gr.HTML(_lock_screen(
                            '<div style="background:rgba(255,255,255,.12);border-radius:15px;padding:16px;'
                            'font-size:12px;color:rgba(255,255,255,.85);text-align:center;line-height:1.5">'
                            'Press <b style="color:#fff">Run scheduled batch</b> to generate the payload '
                            'the selected user would receive.</div>',
                            len(eligible_profiles()), len(PROFILES)))

        with gr.Tab("Checkout cart-filler"):
            with gr.Row(elem_classes="nb-cols"):
                with gr.Column(scale=90):
                    with gr.Column(elem_classes="nb-card"):
                        cart_phone = gr.HTML(cart_phone_html(PROFILES[0]["user_id"], 150))
                with gr.Column(scale=110):
                    with gr.Column(elem_classes="nb-card"):
                        gr.HTML('<div class="nb-eyebrow">Simulate the cart</div>'
                                '<div style="font-size:13px;color:#44443B;margin:6px 0 12px;line-height:1.5">'
                                'Pick a user and drag the total to watch the carousel re-rank. Short of the ₹'
                                f'{FREE_DELIVERY_THRESHOLD} free-delivery threshold, we offer a low-cost item '
                                'from a category they have <b>never bought</b> — a zero-friction trial.</div>')
                        cart_user = gr.Dropdown(
                            choices=[(p["display_name"], p["user_id"]) for p in PROFILES],
                            value=PROFILES[0]["user_id"], label="User")
                        cart_amt = gr.Slider(0, FREE_DELIVERY_THRESHOLD + 60, value=150, step=5,
                                             label="Cart total (₹)")
                    with gr.Column(elem_classes="nb-card"):
                        cart_out = gr.HTML(cart_logic_html(PROFILES[0]["user_id"], 150))

        with gr.Tab("About & methodology"):
            gr.HTML(about_html())

    for btn, p in zip(chip_btns, PROFILES):
        btn.click(lambda uid=p["user_id"]: on_select(uid),
                  outputs=[sel, prof_out, reason_out, phone_out, measure_out] + chip_btns)
    gen.click(on_generate, inputs=sel, outputs=[reason_out, phone_out, measure_out])
    tenure.change(auto_gate_head, inputs=tenure, outputs=head_out)
    tenure.change(auto_funnel_html, inputs=tenure, outputs=funnel_out)
    tenure.change(auto_profile_list_html, inputs=tenure, outputs=gate_out)
    run_batch.click(on_run_auto_batch, inputs=[tenure, auto_user], outputs=auto_out)
    for _c in (cart_user, cart_amt):
        _c.change(cart_phone_html, inputs=[cart_user, cart_amt], outputs=cart_phone)
        _c.change(cart_logic_html, inputs=[cart_user, cart_amt], outputs=cart_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
