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

import gradio as gr
from dotenv import load_dotenv

from agent import generate_nudge
from friction_matching import load_profiles, load_themes, match_profile_to_theme
from auto_targeting import eligibility_detail, eligible_profiles, to_notification
from cart_filler import FREE_DELIVERY_THRESHOLD, suggest_fillers

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
.gradio-container *,.dark .gradio-container *{color:#16130A}
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
.nb-t1{font-size:19px;font-weight:800;letter-spacing:-.02em;color:#16130A}
.nb-t2{font-size:13px;font-weight:500;color:#63635A}
.nb-pill{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:700;
  padding:8px 14px;border-radius:999px}
.nb-dot{width:7px;height:7px;border-radius:50%;display:inline-block}

/* cards */
.nb-card{background:#fff !important;border:1px solid #E7E8E2 !important;border-radius:20px !important;
  padding:20px 22px !important;box-shadow:0 1px 2px rgba(0,0,0,.03) !important;margin-bottom:18px !important}
.nb-eyebrow{font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#6B6B60}
.nb-tile{flex:1;background:#F7F8F5;border-radius:12px;padding:11px 13px}
.nb-tile-k{font-size:11px;font-weight:700;color:#6B6B60;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.nb-tile-v{font-size:14px;font-weight:700;color:#16130A}

/* user chips (Gradio buttons dressed as the design's chips) */
.nb-chip{display:flex !important;align-items:center !important;gap:11px !important;
  padding:11px 13px !important;border-radius:14px !important;text-align:left !important;
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

/* generate button */
#gen, #genbatch{width:100% !important;border:none !important;font-size:15px !important;font-weight:800 !important;
  padding:15px !important;border-radius:14px !important;color:#16130A !important;
  background:#F8CD1B !important;box-shadow:0 6px 18px rgba(248,205,27,.4) !important;
  margin-top:16px !important}

/* dark reasoning card */
.nb-dark{background:#16130A;border-radius:20px;padding:22px;color:#F4F5F3}
.nb-dark *{color:#CFD0C6}
.nb-dark .k{color:#F8CD1B;font-size:12px;font-weight:800;letter-spacing:.09em;text-transform:uppercase}
.nb-dark .h{color:#fff;font-size:13px;font-weight:800;margin-bottom:3px}
.nb-ic{width:30px;height:30px;border-radius:9px;background:rgba(248,205,27,.16);display:flex;
  align-items:center;justify-content:center;font-size:15px;flex:none}

/* phone */
.nb-phonewrap{position:sticky;top:16px;display:flex;flex-direction:column;align-items:center;gap:14px}
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


def reasoning_html(theme, r=None):
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
    else:
        meter, bar = ('<div style="font-size:12px;font-weight:700;color:#C9CABF">out of primary scope</div>',
                      '<div style="height:20px"></div>')

    if not r:
        return f"""
<div class="nb-dark">
  <div style="display:flex;align-items:center;gap:9px">
    <div class="k">Why this nudge</div>
    <div style="flex:1;height:1px;background:rgba(255,255,255,.12)"></div>{meter}</div>
  {bar}
  <div style="font-size:13px;line-height:1.55;color:#B7B8AE">Press <b style="color:#F8CD1B">Generate nudge</b>
    to run the Groq agent for this user. The matched theme above is already resolved
    deterministically — no LLM involved in matching.</div>
</div>"""

    drivers = [("Refund guarantee", r.get("refund_line", "")),
               ("Freshness / quality signal", r.get("fresh_line", ""))]
    if r.get("social_proof_line"):
        drivers.append(("Peer social proof", r.get("social_proof_line", "")))
    d_html = "".join(f"""
      <div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
        border-radius:12px;padding:11px 13px;display:flex;gap:11px;align-items:flex-start">
        <div style="width:22px;height:22px;border-radius:7px;background:#F8CD1B;color:#16130A;
          font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;flex:none">{i}</div>
        <div><div style="font-size:13px;font-weight:700;color:#fff">{esc(n)}</div>
        <div style="font-size:12px;line-height:1.5;color:#B7B8AE;margin-top:2px">{esc(v)}</div></div>
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
  <div style="display:flex;flex-direction:column;gap:16px">
    {scope}
    <div style="display:flex;gap:13px"><div class="nb-ic">👤</div>
      <div><div class="h">Why this user</div>
      <div style="font-size:13px;line-height:1.55;color:#CFD0C6">{esc(r.get('why_user'))}</div></div></div>
    <div style="display:flex;gap:13px"><div class="nb-ic">🧭</div>
      <div><div class="h">Why this category</div>
      <div style="font-size:13px;line-height:1.55;color:#CFD0C6">{esc(r.get('why_category'))}</div></div></div>
    <div>
      <div style="display:flex;gap:13px;margin-bottom:10px"><div class="nb-ic">🛡️</div>
        <div class="h" style="padding-top:6px">Trust drivers, ranked (research-led)</div></div>
      <div style="display:flex;flex-direction:column;gap:8px">{d_html}</div>
    </div>
    <div style="font-size:11px;color:#8E8F86;font-family:monospace">model · {esc(r.get('_model'))}</div>
  </div>
</div>"""


def _phone_shell(p, inner):
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
    </div>
    <div style="padding:16px 15px 22px;background:#F7F8F5;min-height:340px">{inner}</div>
  </div></div>
  <div style="font-size:12px;color:#63635A;font-weight:500;text-align:center;max-width:300px;line-height:1.5">
    Nudge leads with the two research-ranked trust drivers before the offer — not a generic discount.</div>
</div>"""


def phone_html(p, r=None):
    if not r:
        inner = """
<div style="border:1.5px dashed #DADBD3;border-radius:20px;padding:44px 20px;text-align:center">
  <div style="font-size:30px">⚡</div>
  <div style="font-size:13px;color:#63635A;margin-top:10px;line-height:1.5;font-weight:500">
    Press <b style="color:#16130A">Generate nudge</b> to see<br>what this shopper would receive.</div>
</div>"""
        return _phone_shell(p, inner)

    cat = esc(r.get("suggested_category", "")).title()
    emoji = r.get("emoji") or "🛍️"
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
      <div style="display:flex;align-items:center;gap:9px;background:#EFF4FF;border:1px solid #D6E1FB;
        border-radius:11px;padding:9px 12px"><div style="font-size:16px">🛡️</div>
        <div style="font-size:12.5px;font-weight:700;color:#2551C6">{esc(r.get('refund_line'))}</div></div>
      <div style="display:flex;align-items:center;gap:9px;background:#EAF7EE;border:1px solid #C4E7CF;
        border-radius:11px;padding:9px 12px"><div style="font-size:16px">🌿</div>
        <div style="font-size:12.5px;font-weight:700;color:#146634">{esc(r.get('fresh_line'))}</div></div>
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
  </div>
</div>
<div style="text-align:center;font-size:11px;color:#6B6B60;margin-top:16px;font-weight:600">
  Reorder your usuals below ↓</div>"""
    return _phone_shell(p, inner)


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
def auto_eligibility_html():
    """Static cohort view: who the scheduled trigger would and wouldn't auto-nudge."""
    rows = ""
    for p in PROFILES:
        ok, why = eligibility_detail(p)
        badge = ("#1F9D55", "#EAF7EE", "✓ Auto-nudge") if ok else ("#9A8B00", "#FBF4CE", "· Skipped")
        rows += (
            f'<div style="display:flex;align-items:center;gap:12px;padding:10px 0;'
            f'border-bottom:1px solid #EFEFE9">'
            f'<div style="font-size:11px;font-weight:800;color:{badge[0]};background:{badge[1]};'
            f'border-radius:20px;padding:4px 10px;white-space:nowrap">{badge[2]}</div>'
            f'<div style="flex:1;min-width:0"><div style="font-size:13.5px;font-weight:700;color:#16130A">'
            f'{esc(p["display_name"])} · <span style="color:#6B6B60;font-weight:600">{esc(p.get("order_frequency"))}, '
            f'{esc(p.get("tenure"))}</span></div>'
            f'<div style="font-size:12px;color:#6B6B60;margin-top:2px">{esc(why)}</div></div></div>')
    n = len(eligible_profiles())
    return (
        '<div class="nb-eyebrow">Scheduled trigger · eligibility</div>'
        '<div style="font-size:13px;color:#44443B;margin:6px 0 14px;line-height:1.5">Rule: '
        '<b>order_frequency = Daily or Weekly</b> AND <b>tenure &gt; 6 months</b>. Habitual, '
        'established buyers whose basket has settled — the auto-nudge cohort.</div>'
        f'{rows}'
        f'<div style="margin-top:14px;font-size:12.5px;font-weight:700;color:#16130A">'
        f'{n} of {len(PROFILES)} synthetic users qualify.</div>')


def _notification_card(p, notif):
    return (
        '<div style="background:#fff;border:1px solid #EFEFE9;border-radius:16px;padding:14px 15px;'
        'box-shadow:0 6px 22px rgba(0,0,0,.06);margin-bottom:12px">'
        '<div style="display:flex;align-items:center;gap:9px;margin-bottom:8px">'
        '<div style="width:26px;height:26px;border-radius:7px;background:#F8CD1B;display:flex;'
        f'align-items:center;justify-content:center;font-size:15px">{notif.get("emoji","🛒")}</div>'
        '<div style="font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;'
        f'color:#6F5700">Blinkit · now · to {esc(p["display_name"])}</div></div>'
        f'<div style="font-size:14px;font-weight:800;color:#16130A;line-height:1.25">{esc(notif.get("title"))}</div>'
        f'<div style="font-size:12.5px;color:#44443B;line-height:1.5;margin-top:5px">{esc(notif.get("body"))}</div>'
        f'<div style="margin-top:9px;display:inline-block;font-size:11px;font-weight:700;color:#2551C6;'
        f'background:#EFF4FF;border-radius:20px;padding:3px 10px">New category · {esc(notif.get("category"))}</div>'
        '</div>')


def on_run_auto_batch():
    if "GROQ_API_KEY" not in os.environ:
        return ('<div class="nb-dark"><div class="k">Error</div><div style="margin-top:8px;font-size:13px">'
                'GROQ_API_KEY is not configured on the server.</div></div>')
    cards = ""
    for p in eligible_profiles():
        theme, reason = match_profile_to_theme(p, THEMES)
        try:
            notif = to_notification(generate_nudge(p, theme, reason))
        except Exception as e:  # noqa: BLE001
            cards += f'<div style="color:#B00">Failed for {esc(p["display_name"])}: {esc(e)}</div>'
            continue
        cards += _notification_card(p, notif)
    return ('<div class="nb-eyebrow" style="margin-bottom:10px">Generated notifications · '
            'live Groq</div>' + cards +
            '<div style="margin-top:8px;font-size:11.5px;color:#6B6B60">Simulated scheduled send — '
            'no real push is dispatched. Copy generated live per user.</div>')


# ----------------------- Feature 2: checkout cart-filler -----------------------
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


# ----------------------------- app -----------------------------
def on_select(uid):
    p = BY_ID[uid]
    theme, reason = match_profile_to_theme(p, THEMES)
    chips = [gr.update(variant="primary" if x["user_id"] == uid else "secondary") for x in PROFILES]
    return [uid, profile_html(p, theme, reason), reasoning_html(theme), phone_html(p)] + chips


def on_generate(uid):
    p = BY_ID[uid]
    theme, reason = match_profile_to_theme(p, THEMES)
    if "GROQ_API_KEY" not in os.environ:
        err = ('<div class="nb-dark"><div class="k">Error</div>'
               '<div style="margin-top:8px;font-size:13px">GROQ_API_KEY is not configured on the '
               'server. Add it in the Space Settings → Variables and secrets.</div></div>')
        return err, phone_html(p)
    try:
        r = generate_nudge(p, theme, reason)   # REAL Groq call
    except Exception as e:  # noqa: BLE001
        return (f'<div class="nb-dark"><div class="k">Error</div>'
                f'<div style="margin-top:8px;font-size:13px">{esc(e)}</div></div>', phone_html(p))
    return reasoning_html(theme, r), phone_html(p, r)


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
                    reason_out = gr.HTML(reasoning_html(_t0))
                with gr.Column(scale=85):
                    phone_out = gr.HTML(phone_html(_p0))

        with gr.Tab("Auto-nudge queue"):
            with gr.Row(elem_classes="nb-cols"):
                with gr.Column(scale=110):
                    with gr.Column(elem_classes="nb-card"):
                        gr.HTML(auto_eligibility_html())
                        run_batch = gr.Button("▶ Run scheduled batch", elem_id="genbatch")
                with gr.Column(scale=90):
                    with gr.Column(elem_classes="nb-card"):
                        auto_out = gr.HTML('<div style="font-size:13px;color:#6B6B60;line-height:1.55">'
                                           'Press <b style="color:#16130A">Run scheduled batch</b> to generate '
                                           'the notification each eligible user would receive.</div>')

        with gr.Tab("Checkout cart-filler"):
            with gr.Row(elem_classes="nb-cols"):
                with gr.Column(scale=110):
                    with gr.Column(elem_classes="nb-card"):
                        gr.HTML('<div class="nb-eyebrow">Simulate a checkout</div>'
                                '<div style="font-size:13px;color:#44443B;margin:6px 0 12px;line-height:1.5">'
                                'Pick a user and a cart total. If they are short of the ₹'
                                f'{FREE_DELIVERY_THRESHOLD} free-delivery threshold, we offer a low-cost item '
                                'from a category they have <b>never bought</b> — a zero-friction trial.</div>')
                        cart_user = gr.Dropdown(
                            choices=[(p["display_name"], p["user_id"]) for p in PROFILES],
                            value=PROFILES[0]["user_id"], label="User")
                        cart_amt = gr.Slider(0, FREE_DELIVERY_THRESHOLD + 60, value=150, step=5,
                                             label="Cart total (₹)")
                with gr.Column(scale=90):
                    with gr.Column(elem_classes="nb-card"):
                        cart_out = gr.HTML(cart_filler_html(PROFILES[0]["user_id"], 150))

    for btn, p in zip(chip_btns, PROFILES):
        btn.click(lambda uid=p["user_id"]: on_select(uid),
                  outputs=[sel, prof_out, reason_out, phone_out] + chip_btns)
    gen.click(on_generate, inputs=sel, outputs=[reason_out, phone_out])
    run_batch.click(on_run_auto_batch, outputs=auto_out)
    cart_user.change(cart_filler_html, inputs=[cart_user, cart_amt], outputs=cart_out)
    cart_amt.change(cart_filler_html, inputs=[cart_user, cart_amt], outputs=cart_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
