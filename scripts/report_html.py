#!/usr/bin/env python3
"""Render a run as a single self-contained HTML page.

    python scripts/report_html.py output/results_2026-08-06.parquet
    python scripts/report_html.py output/results_2026-08-06.parquet --open

No external requests: every byte — data, styles, script, charts — is inline,
so the page works from a file:// URL, from GitHub Pages, and from an email
attachment, and it cannot leak a request to a third party when someone opens
it.

Data is embedded as an array-of-arrays with a separate column list rather than
an array of objects. Repeating 40 field names across 9,000 rows costs several
megabytes of pure redundancy, and a page that large is slow before a single
row is drawn.

The page is deliberately honest about what it is showing. Every number here
comes from free data with a monthly, ~60-day-lagged price; the credit signal
is unvalidated; and roughly four fifths of rows carry a rating cap. Those
facts sit at the top of the page, not in a footnote, because a clean-looking
table of BUY ratings invites more confidence than this model has earned.
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.logging_setup import get_logger
from scripts.safe_json import dumps_for_script

log = get_logger('report_html')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')

# (field, label, format, width, tooltip). The tooltip is the point of the
# column list as much as the label is: half of these are jargon, and a number
# whose caveat is not attached to it will be read without the caveat.
COLUMNS = [
    ('cusip', 'CUSIP', 'text', 90,
     "The security's nine-character identifier. Bonds are identified per ISSUE, "
     "not per issuer — one company often has dozens outstanding, each with its "
     "own coupon, maturity and price."),
    ('issuer_name', 'Issuer', 'text', 210,
     "The issuer as reported by funds in their N-PORT filings. Spellings vary "
     "widely between funds; the crosswalk resolves them to one company where it "
     "can, and roughly half the universe it cannot."),
    ('asset_class', 'Class', 'cls', 78,
     "TSY = US Treasury note or bond, T-BILL = Treasury bill, IG / HY = "
     "corporate investment grade or high yield. IG vs HY is assigned by this "
     "model's own credit score, NOT by a rating agency."),
    ('maturity_date', 'Maturity', 'date', 92,
     "Final maturity. Call schedules are not available in free data, so a "
     "callable bond may be repaid years earlier than this date suggests."),
    ('years_to_maturity', 'Yrs', 'n1', 52,
     "Years from today to final maturity."),
    ('coupon_rate', 'Coupon', 'pct2', 68,
     "Annual coupon rate, paid semiannually for almost everything here. Zero "
     "for Treasury bills, which are sold at a discount instead."),
    ('clean_price_est', 'Price', 'n2', 68,
     "Estimated clean price per 100 face — the last fund mark aged onto today's "
     "curve via the spread it implied. NOT a live quote: the underlying mark is "
     "typically about 98 days old."),
    ('ytw', 'YTW', 'pct2', 68,
     "Yield to worst: the lower of yield-to-maturity and yield-to-call. With no "
     "call schedules in free data this equals yield-to-maturity, so for a "
     "callable bond it OVERSTATES what you are likely to receive."),
    ('modified_duration', 'Dur', 'n2', 58,
     "Modified duration: roughly the percentage the price falls if yields rise "
     "by one percentage point. A duration of 7 means a 1% yield rise costs "
     "about 7% of the price."),
    ('convexity', 'Cvx', 'n0', 56,
     "Convexity: how duration itself changes as yields move. Higher is better — "
     "the bond gains more in a rally than it loses in an equal selloff."),
    ('z_spread', 'Z-spd', 'bp', 66,
     "Z-spread in basis points: the constant spread over the entire Treasury "
     "zero curve that reprices this bond. This is what you are paid for taking "
     "credit risk instead of lending to the government."),
    ('fair_spread', 'Fair', 'bp', 62,
     "What the model thinks the spread should be, given the issuer's credit "
     "bucket and this bond's maturity. Anchored on what OTHER bonds in the same "
     "bucket actually trade at, so roughly half of each bucket is cheap by "
     "construction."),
    ('spread_mispricing', 'Mispr', 'bp', 66,
     "Observed spread minus fair spread. POSITIVE MEANS CHEAP — you are paid "
     "more than the model thinks the risk deserves. This is the model's main "
     "valuation signal, and the one the backtest supports: the cheapest fifth "
     "beat the richest by about 27bp per month."),
    ('implied_bucket', 'Model', 'bucket', 62,
     "The credit bucket implied by the issuer's financials — a six-factor "
     "scorecard, NOT an agency rating. It does not predict forward spread "
     "changes in testing, so read it as a description of the issuer, not a "
     "forecast."),
    ('market_bucket', 'Market', 'bucket', 64,
     "The credit bucket implied by the bond's OWN spread — where the market is "
     "actually pricing this credit today."),
    ('bucket_divergence_notches', 'Div', 'n0', 48,
     "Divergence in notches: Market minus Model. Positive means the market "
     "prices the credit WORSE than the financials suggest (a possible rising "
     "star); negative means better (possible fallen-angel risk)."),
    ('issuer_credit_score', 'Score', 'n0', 58,
     "Issuer credit score, 0-100. Weighted from market capitalisation, market "
     "leverage, interest coverage, Altman-Z, cash generation and book leverage "
     "— weights MEASURED against observed spreads rather than chosen."),
    ('n_funds', 'Funds', 'int', 56,
     "How many registered funds hold this bond. The price is the median across "
     "them with outliers rejected, so more funds means a more reliable mark. "
     "Below three, the row is capped."),
    ('total_held_usd', 'Held', 'usd', 74,
     "Total value held across all reporting funds. This is a LOWER BOUND on the "
     "issue size, never the issue size itself — it only counts funds that file "
     "N-PORT."),
    ('mark_age_days', 'Mark', 'days', 58,
     "Age of the underlying fund mark in days. N-PORT publishes monthly with "
     "roughly a 60-day lag, so about 98 days is normal rather than stale. Past "
     "100 days the row is capped."),
    ('_composite_score', 'Comp', 'n1', 60,
     "Composite score, 0-100, weighted across Valuation, Credit, Rates, "
     "Structure and Liquidity. A category that cannot describe an instrument — "
     "credit metrics for a Treasury — drops out entirely and the rest "
     "renormalise, rather than scoring zero."),
    ('rating', 'Rating', 'rating', 88,
     "BUY / LEAN BUY / HOLD / PASS, from the composite against thresholds "
     "calibrated separately per asset class. Caps can only LOWER a rating, "
     "never raise it, so a capped BUY shows as HOLD or PASS."),
    ('_caps', 'Caps', 'caps', 200,
     "Why this row's rating was capped. A cap does not say the bond is bad — it "
     "says the data does not support acting on it. Most often the issuer could "
     "not be identified, so no credit view is possible."),
]

CATEGORY_SCORES = [('_score_valuation', 'Valuation'), ('_score_credit', 'Credit'),
                   ('_score_rates', 'Rates'), ('_score_structure', 'Structure'),
                   ('_score_liquidity', 'Liquidity')]


def _clean(value):
    """Coerce one cell to something JSON can carry."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (int, float, str, bool)):
        return round(value, 6) if isinstance(value, float) else value
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    if hasattr(value, 'tolist'):
        try:
            return list(value.tolist())
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return str(value)


def load_rows(path):
    import pandas as pd
    frame = pd.read_parquet(path)
    return frame.to_dict('records')


def load_json(name):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def build_payload(rows):
    """Compact table data plus everything the summary panels need."""
    table = []
    for row in rows:
        caps = row.get('_rating_cap_reasons')
        caps = list(caps) if caps is not None and len(caps) else []
        record = dict(row)
        record['_caps'] = '; '.join(str(c) for c in caps)
        table.append([_clean(record.get(field)) for field, *_ in COLUMNS])

    ratings = Counter(r.get('rating') for r in rows if r.get('rating'))
    classes = Counter(r.get('asset_class') for r in rows if r.get('asset_class'))
    buckets = Counter(r.get('implied_bucket') for r in rows if r.get('implied_bucket'))
    capped = sum(1 for r in rows
                 if r.get('_rating_cap_reasons') is not None
                 and len(r.get('_rating_cap_reasons')) > 0)

    cap_reasons = Counter()
    for row in rows:
        reasons = row.get('_rating_cap_reasons')
        if reasons is None:
            continue
        for reason in list(reasons):
            cap_reasons[str(reason).split('(')[0].strip()] += 1

    # Per-category mean score, over rows where the category applied at all.
    categories = []
    for field, label in CATEGORY_SCORES:
        values = [r[field] for r in rows
                  if r.get(field) is not None and r[field] == r[field]]
        if values:
            categories.append({'label': label, 'n': len(values),
                               'mean': round(sum(values) / len(values), 1)})

    buy = [r for r in rows if r.get('rating') == 'BUY']
    concentration = Counter(r.get('peer_group', '?') for r in buy)

    return {
        'columns': [{'k': f, 'l': l, 'f': fmt, 'w': w, 't': tip}
                    for f, l, fmt, w, tip in COLUMNS],
        'rows': table,
        'ratings': dict(ratings),
        'classes': dict(classes),
        'buckets': dict(buckets),
        'capped': capped,
        'capReasons': cap_reasons.most_common(10),
        'categories': categories,
        'concentration': concentration.most_common(),
        'buyCount': len(buy),
        'total': len(rows),
    }


# ---------------------------------------------------------------------------
# Charts, drawn as inline SVG
# ---------------------------------------------------------------------------

def _svg_line_chart(series, width=460, height=190, pad=38, y_fmt='pct',
                    x_label='', y_label=''):
    """A small multi-series line chart. series: [{name, colour, points}]."""
    points = [p for s in series for p in s['points']]
    if not points:
        return '<p class="muted">no data</p>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    span = (y1 - y0) or (abs(y1) or 1) * 0.1
    y0, y1 = y0 - span * 0.12, y1 + span * 0.12

    def px(x):
        return pad + (x - x0) / (x1 - x0) * (width - pad - 12)

    def py(y):
        return height - pad - (y - y0) / (y1 - y0) * (height - pad - 16)

    def fmt(v):
        return f'{v * 100:.2f}%' if y_fmt == 'pct' else f'{v:.2f}x'

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" '
             f'role="img" aria-label="{y_label} against {x_label}">']
    # gridlines
    for i in range(4):
        y = y0 + (y1 - y0) * i / 3
        parts.append(f'<line class="grid" x1="{pad}" x2="{width - 12}" '
                     f'y1="{py(y):.1f}" y2="{py(y):.1f}"/>')
        parts.append(f'<text class="tick" x="{pad - 6}" y="{py(y) + 3:.1f}" '
                     f'text-anchor="end">{fmt(y)}</text>')
    for s in series:
        if not s['points']:
            continue
        d = ' '.join(f'{"M" if i == 0 else "L"}{px(x):.1f},{py(y):.1f}'
                     for i, (x, y) in enumerate(sorted(s['points'])))
        parts.append(f'<path d="{d}" fill="none" stroke="{s["colour"]}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
        for x, y in s['points']:
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" '
                         f'fill="{s["colour"]}"/>')
    for i, s in enumerate(series):
        parts.append(f'<rect x="{pad + i * 118}" y="6" width="9" height="9" '
                     f'rx="2" fill="{s["colour"]}"/>')
        parts.append(f'<text class="tick" x="{pad + 13 + i * 118}" y="14">'
                     f'{s["name"]}</text>')
    parts.append(f'<text class="tick" x="{width / 2}" y="{height - 6}" '
                 f'text-anchor="middle">{x_label}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _svg_bars(items, width=460, height=190, pad=38, colour='#4f7fd1'):
    """Horizontal bars for a small labelled distribution."""
    if not items:
        return '<p class="muted">no data</p>'
    top = max(v for _, v in items) or 1
    row_h = (height - 18) / len(items)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart">']
    for i, (label, value) in enumerate(items):
        y = 10 + i * row_h
        w = (value / top) * (width - pad - 74)
        parts.append(f'<text class="tick" x="{pad - 6}" y="{y + row_h / 2 + 3:.1f}" '
                     f'text-anchor="end">{label}</text>')
        parts.append(f'<rect x="{pad}" y="{y:.1f}" width="{max(w, 1):.1f}" '
                     f'height="{row_h * 0.66:.1f}" rx="2" fill="{colour}"/>')
        parts.append(f'<text class="tick" x="{pad + w + 6:.1f}" '
                     f'y="{y + row_h / 2 + 3:.1f}">{value:,}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def curve_chart(meta):
    from models.curve import TENOR_YEARS
    par = (meta or {}).get('par_curve') or {}
    points = sorted((TENOR_YEARS[k], v) for k, v in par.items()
                    if k in TENOR_YEARS and v is not None)
    if not points:
        return '<p class="muted">no curve</p>'
    return _svg_line_chart(
        [{'name': 'Treasury par curve', 'colour': '#4f7fd1', 'points': points}],
        y_fmt='pct', x_label='years to maturity')


def term_chart(term):
    """Fitted term factors per credit tier, against FRED's published slices."""
    if not term:
        return '<p class="muted">no fitted term structure</p>'
    colours = {'tight': '#3f9d6b', 'mid': '#4f7fd1', 'wide': '#c96a3f'}
    series = []
    for tier, payload in (term.get('by_tier') or {}).items():
        pts = [(t, f) for t, f in payload.get('points', [])]
        if pts:
            series.append({'name': tier, 'colour': colours.get(tier, '#888'),
                           'points': pts})
    fred = [(2.0, 0.59), (4.0, 0.86), (6.0, 1.03), (8.5, 1.22),
            (12.5, 1.18), (20.0, 1.27)]
    series.append({'name': 'FRED slices', 'colour': '#9aa0a6', 'points': fred})
    return _svg_line_chart(series, y_fmt='x', x_label='years to maturity')


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bond Analysis — {run_date}</title>
<style>
:root {{
  --bg:#fbfbfc; --panel:#fff; --ink:#1c1f24; --muted:#6b7280; --line:#e3e6ea;
  --buy:#1f7a4d; --lean:#4f7fd1; --hold:#8a8f98; --pass:#b4483c; --warn:#a8621b;
  --accent:#4f7fd1;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --bg:#14161a; --panel:#1c1f24; --ink:#e6e8eb; --muted:#9aa0a6;
    --line:#2b2f36; --buy:#4ec98a; --lean:#7aa7e8; --hold:#9aa0a6;
    --pass:#e0705f; --warn:#d99a4e; }}
}}
:root[data-theme="dark"] {{ --bg:#14161a; --panel:#1c1f24; --ink:#e6e8eb;
  --muted:#9aa0a6; --line:#2b2f36; --buy:#4ec98a; --lean:#7aa7e8;
  --hold:#9aa0a6; --pass:#e0705f; --warn:#d99a4e; }}
:root[data-theme="light"] {{ --bg:#fbfbfc; --panel:#fff; --ink:#1c1f24;
  --muted:#6b7280; --line:#e3e6ea; --buy:#1f7a4d; --lean:#4f7fd1;
  --hold:#8a8f98; --pass:#b4483c; --warn:#a8621b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1500px; margin:0 auto; padding:20px 18px 60px; }}
h1 {{ font-size:22px; margin:0 0 2px; font-weight:650; letter-spacing:-.01em; }}
h2 {{ font-size:15px; margin:0 0 12px; font-weight:600; }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:16px; margin-bottom:16px; }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }}
.cards {{ display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
  margin-bottom:16px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px 14px; }}
.card .v {{ font-size:22px; font-weight:650; letter-spacing:-.02em; }}
.card .k {{ color:var(--muted); font-size:11px; text-transform:uppercase;
  letter-spacing:.05em; margin-top:2px; }}
.caveat {{ border-left:3px solid var(--warn); background:color-mix(in srgb,var(--warn) 7%,transparent); }}
.caveat h2 {{ color:var(--warn); }}
.caveat ul {{ margin:0; padding-left:18px; }}
.caveat li {{ margin-bottom:6px; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th,td {{ padding:5px 8px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--line); }}
th {{ position:sticky; top:0; background:var(--panel); cursor:pointer;
  font-weight:600; user-select:none; z-index:2; font-size:11.5px;
  text-transform:uppercase; letter-spacing:.03em; color:var(--muted); }}
th:hover {{ color:var(--ink); }}
th.sorted::after {{ content:" \\2193"; }} th.sorted.asc::after {{ content:" \\2191"; }}
td.l,th.l {{ text-align:left; }}
tbody tr:hover {{ background:color-mix(in srgb,var(--accent) 7%,transparent); }}
.scroll {{ overflow:auto; max-height:74vh; border:1px solid var(--line);
  border-radius:10px; background:var(--panel); }}
.tag {{ display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px;
  font-weight:600; }}
.BUY {{ background:color-mix(in srgb,var(--buy) 18%,transparent); color:var(--buy); }}
.LEAN {{ background:color-mix(in srgb,var(--lean) 18%,transparent); color:var(--lean); }}
.HOLD {{ background:color-mix(in srgb,var(--hold) 18%,transparent); color:var(--hold); }}
.PASS {{ background:color-mix(in srgb,var(--pass) 18%,transparent); color:var(--pass); }}
.pos {{ color:var(--buy); }} .neg {{ color:var(--pass); }}
.caps {{ color:var(--warn); font-size:11px; }}
.muted {{ color:var(--muted); }}
.controls {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center;
  margin-bottom:12px; }}
input,select,button {{ font:inherit; padding:6px 10px; border-radius:7px;
  border:1px solid var(--line); background:var(--panel); color:var(--ink); }}
input:focus,select:focus {{ outline:2px solid var(--accent); outline-offset:-1px; }}
button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent); }}
.chart {{ width:100%; height:auto; }}
.grid line.grid {{ stroke:var(--line); }}
svg .grid {{ stroke:var(--line); stroke-width:1; }}
svg .tick {{ fill:var(--muted); font-size:9px; }}
.bar {{ height:7px; border-radius:4px; background:var(--accent); }}
.dist {{ display:grid; grid-template-columns:auto 1fr auto; gap:6px 10px;
  align-items:center; font-size:12.5px; }}
footer {{ color:var(--muted); font-size:12px; margin-top:26px;
  border-top:1px solid var(--line); padding-top:14px; }}
code {{ background:color-mix(in srgb,var(--muted) 14%,transparent);
  padding:1px 5px; border-radius:4px; font-size:12px; }}
/* Tooltips float on <body>, not as a ::after on the trigger: the table lives
   in an overflow:auto container that would clip a pseudo-element, and the
   header row is sticky, which creates its own stacking context. */
/* width:max-content sizes the box to its TEXT. Without it the box sizes to its
   containing block, which for an absolutely-positioned child of <body> is the
   initial containing block — and any renderer reporting that as narrow (or 0)
   wraps a two-line tooltip into a 90px-wide, 400px-tall ribbon. */
#tip {{ position:absolute; z-index:99; width:max-content; max-width:330px;
  padding:9px 11px;
  background:var(--panel); color:var(--ink); border:1px solid var(--line);
  border-radius:8px; box-shadow:0 6px 24px rgba(0,0,0,.16);
  font-size:12.5px; line-height:1.45; pointer-events:none; opacity:0;
  transition:opacity .12s; font-weight:400; text-transform:none;
  letter-spacing:normal; text-align:left; }}
#tip.on {{ opacity:1; }}
[data-tip] {{ cursor:help; }}
th[data-tip] {{ text-decoration:underline dotted
  color-mix(in srgb,var(--muted) 60%,transparent); text-underline-offset:3px; }}
.card[data-tip] .k::after {{ content:" \24D8"; opacity:.55; }}
.dist span[data-tip] {{ text-decoration:underline dotted
  color-mix(in srgb,var(--muted) 55%,transparent); text-underline-offset:2px; }}
h2[data-tip]::after {{ content:" \24D8"; opacity:.4; font-size:12px; }}
</style></head><body><div class="wrap">

<h1>Bond Analysis</h1>
<div class="sub">Run {run_date} &middot; curve {curve_date} &middot; {total:,} instruments
&middot; US Treasuries and corporate bonds &middot; free data only</div>

<div class="cards">{cards}</div>

<div class="panel caveat">
<h2>What this is, and what it is not</h2>
<ul>
<li><strong>Prices are monthly and about {mark_age} days old.</strong> They come from
SEC Form N-PORT fund holdings, the only free per-CUSIP source. Every price shown is
that mark aged onto today's curve — an estimate, never a quote.</li>
<li><strong>The credit signal is unvalidated.</strong> The backtest can measure relative
value (cheap bonds beat rich ones by ~27bp per month, monotonically across quintiles),
but the model's implied rating does <em>not</em> predict forward spread change. Treat the
credit column as a description, not a forecast.</li>
<li><strong>{capped_pct}% of rows carry a rating cap</strong> — mostly because the issuer
could not be identified. A capped row still shows its uncapped rating; the cap says the
data does not support acting on it.</li>
<li><strong>No agency ratings, no true OAS, no covenants.</strong> Z-spread stands in for
OAS, so callable bonds priced above par are systematically overstated and are capped.</li>
</ul>
</div>

<div class="grid">
  <div class="panel"><h2 data-tip="Thresholds are calibrated per asset class against each class own score distribution, so BUY means roughly the top 3% of that class rather than an absolute standard.">Rating distribution</h2><div class="dist">{rating_dist}</div></div>
  <div class="panel"><h2 data-tip="The model own credit view, from a six-factor scorecard. Cutpoints are matched to the mix the bond market itself prices, so the shape here should resemble the real corporate universe.">Implied credit bucket</h2><div class="dist">{bucket_dist}</div></div>
  <div class="panel"><h2 data-tip="Today par yield curve from home.treasury.gov, bootstrapped to zero rates and used to discount every cashflow in the model.">Treasury par curve</h2>{curve_svg}</div>
  <div class="panel"><h2 data-tip="How much wider a spread should be at each maturity, as a multiple of the five-year level. Fitted from about 130,000 observed spreads; FRED published slices stop at 15 years and overstate the long end by up to 17%.">Spread term structure, fitted vs FRED</h2>{term_svg}
    <p class="muted" style="font-size:12px;margin:8px 0 0">Fitted from ~130,000 observed
    spreads. Wide credits <em>invert</em> — distressed risk sits in near-dated paper.</p></div>
  <div class="panel"><h2 data-tip="Caps lower a rating when the evidence does not support acting on it. The dominant reason is an unidentified issuer, which is a coverage limit rather than a judgement on the bond.">Why rows are capped</h2>{cap_svg}</div>
  <div class="panel"><h2 data-tip="Average score in each of the five categories, over the rows where that category applied at all. A category that cannot describe an instrument drops out of its composite entirely rather than scoring zero.">Mean score by category</h2>{cat_svg}
    <p class="muted" style="font-size:12px;margin:8px 0 0">A category absent from a row
    was structurally inapplicable — a Treasury has no issuer balance sheet — and drops
    out of that row's composite rather than scoring zero.</p></div>
</div>

<div class="panel">
<h2>All {total:,} instruments</h2>
<div class="controls">
  <input id="q" type="search" placeholder="Search CUSIP or issuer…" style="min-width:230px">
  <select id="fRating"><option value="">All ratings</option>
    <option>BUY</option><option>LEAN BUY</option><option>HOLD</option><option>PASS</option></select>
  <select id="fClass"><option value="">All classes</option>{class_opts}</select>
  <select id="fBucket"><option value="">All buckets</option>{bucket_opts}</select>
  <label style="font-size:12.5px"><input type="checkbox" id="fUncapped"
    style="vertical-align:-2px"> Uncapped only</label>
  <button id="reset">Reset</button>
  <span id="count" class="muted"></span>
</div>
<div class="scroll"><table id="t"><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
<p class="muted" style="font-size:12px;margin:10px 0 0">Click a header to sort.
Showing the first 1,000 matching rows; narrow the filters to see the rest.</p>
</div>

<footer>
Generated {generated} by
<a href="https://github.com/danmcooper-ops/Bond-Analysis-">bond-analysis-model</a>.
Sources: SEC Form N-PORT, US Treasury, FRED, TreasuryDirect, SEC XBRL. Free data only.
<strong>Not investment advice.</strong> The credit signal is unvalidated; see the caveats above.
</footer>
</div>
<div id="tip" role="tooltip"></div>
<script id="payload" type="application/json">{payload}</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const COLS = D.columns, ROWS = D.rows, LIMIT = 1000;
const idx = {{}}; COLS.forEach((c,i)=>idx[c.k]=i);
let sortCol = idx['_composite_score'], sortAsc = false;

const bp = v => v==null ? '' : (v*10000).toFixed(0);
const esc = t => String(t==null?'':t).replace(/&/g,'&amp;').replace(/"/g,'&quot;')
  .replace(/</g,'&lt;').replace(/>/g,'&gt;');
function fmt(v, f) {{
  if (v==null || v==='') return '<span class="muted">—</span>';
  switch(f) {{
    case 'pct2': return (v*100).toFixed(2)+'%';
    case 'n0': return (+v).toFixed(0);
    case 'n1': return (+v).toFixed(1);
    case 'n2': return (+v).toFixed(2);
    case 'int': return (+v).toLocaleString();
    case 'bp': {{ const b=+bp(v); const c=b>0?'pos':(b<0?'neg':'');
      return '<span class="'+c+'">'+b+'</span>'; }}
    case 'usd': {{ const n=+v; return n>=1e9 ? (n/1e9).toFixed(1)+'bn'
      : n>=1e6 ? (n/1e6).toFixed(0)+'m' : (n/1e3).toFixed(0)+'k'; }}
    case 'days': return (+v).toFixed(0)+'d';
    case 'date': return String(v).slice(0,10);
    case 'cls': return String(v).replace('TREASURY_BILL','T-BILL')
      .replace('TREASURY','TSY').replace('CORP_','');
    case 'bucket': return v;
    case 'rating': {{ const k=String(v).replace(' BUY','').replace('LEAN','LEAN');
      const cn = v==='LEAN BUY'?'LEAN':v; return '<span class="tag '+cn+'">'+v+'</span>'; }}
    case 'caps': return v ? '<span class="caps">'+v+'</span>' : '';
    default: return String(v);
  }}
}}

function head() {{
  document.getElementById('head').innerHTML = COLS.map((c,i)=>{{
    const left = ['text','cls','rating','caps','bucket','date'].includes(c.f);
    const cls = (left?'l ':'') + (i===sortCol ? 'sorted'+(sortAsc?' asc':'') : '');
    return '<th class="'+cls+'" data-i="'+i+'" data-tip="'+esc(c.t)+'" '+
      'style="min-width:'+c.w+'px">'+c.l+'</th>';
  }}).join('');
  document.querySelectorAll('#head th').forEach(th=>th.onclick=()=>{{
    const i=+th.dataset.i; sortAsc = (i===sortCol) ? !sortAsc : false; sortCol=i;
    head(); draw();
  }});
}}

function filtered() {{
  const q=document.getElementById('q').value.trim().toLowerCase();
  const r=document.getElementById('fRating').value;
  const c=document.getElementById('fClass').value;
  const b=document.getElementById('fBucket').value;
  const un=document.getElementById('fUncapped').checked;
  return ROWS.filter(row=>{{
    if (r && row[idx['rating']]!==r) return false;
    if (c && row[idx['asset_class']]!==c) return false;
    if (b && row[idx['implied_bucket']]!==b) return false;
    if (un && row[idx['_caps']]) return false;
    if (q) {{ const hay=((row[idx['cusip']]||'')+' '+(row[idx['issuer_name']]||'')).toLowerCase();
      if (!hay.includes(q)) return false; }}
    return true;
  }});
}}

function draw() {{
  const rows = filtered();
  rows.sort((a,b)=>{{
    let x=a[sortCol], y=b[sortCol];
    if (x==null) return 1; if (y==null) return -1;
    if (typeof x==='string') return sortAsc ? x.localeCompare(y) : y.localeCompare(x);
    return sortAsc ? x-y : y-x;
  }});
  const shown = rows.slice(0, LIMIT);
  document.getElementById('count').textContent =
    rows.length.toLocaleString()+' match'+(rows.length===1?'':'es')+
    (rows.length>LIMIT ? ' (showing '+LIMIT.toLocaleString()+')' : '');
  document.getElementById('body').innerHTML = shown.map(row=>'<tr>'+
    COLS.map((c,i)=>{{
      const left = ['text','cls','rating','caps','bucket','date'].includes(c.f);
      // Long values are visually truncated by the column width, so carry the
      // full text natively — a hover tooltip the user cannot trigger by
      // keyboard is not a substitute for the value being readable.
      const long = (c.f==='caps'||c.f==='text') && row[i];
      const t = long ? ' title="'+esc(String(row[i]))+'"' : '';
      return '<td'+(left?' class="l"':'')+t+'>'+fmt(row[i],c.f)+'</td>';
    }}).join('')+'</tr>').join('');
}}

['q','fRating','fClass','fBucket','fUncapped'].forEach(id=>{{
  const el=document.getElementById(id);
  el.addEventListener(el.tagName==='INPUT'&&el.type==='search'?'input':'change', draw);
}});
document.getElementById('reset').onclick=()=>{{
  ['q','fRating','fClass','fBucket'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('fUncapped').checked=false; draw();
}};
head(); draw();

// --- tooltips --------------------------------------------------------------
// One floating element, positioned on hover or keyboard focus. Delegated from
// document so it covers rows drawn after load without rebinding.
const TIP = document.getElementById('tip');
function showTip(el) {{
  const text = el.getAttribute('data-tip');
  if (!text) return;
  TIP.textContent = text;
  // Park it at a known origin BEFORE measuring. Measuring while `left` is
  // still `auto` lays the box out at its static position, where the wrap — and
  // therefore the height — can differ from where it will actually sit.
  TIP.style.left = '0px';
  TIP.style.top = '0px';
  TIP.classList.add('on');

  const r = el.getBoundingClientRect();
  const t = TIP.getBoundingClientRect();
  // clientWidth reports 0 in some embedded/snapshot renderers; falling through
  // to it unguarded drives maxLeft negative and pins every tooltip to the left
  // edge of the page.
  const vw = document.documentElement.clientWidth || window.innerWidth || 1024;
  const vh = window.innerHeight || document.documentElement.clientHeight || 768;

  // Prefer below; flip above when that would run off the bottom of the window.
  let top = r.bottom + window.scrollY + 8;
  if (r.bottom + t.height + 16 > vh && r.top > t.height + 16)
    top = r.top + window.scrollY - t.height - 8;

  let left = r.left + window.scrollX;
  const minLeft = window.scrollX + 8;
  const maxLeft = Math.max(minLeft, window.scrollX + vw - t.width - 10);
  TIP.style.top = top + 'px';
  TIP.style.left = Math.max(minLeft, Math.min(left, maxLeft)) + 'px';
}}
function hideTip() {{ TIP.classList.remove('on'); }}
document.addEventListener('mouseover', e => {{
  const el = e.target.closest('[data-tip]');
  if (el) showTip(el); else if (!TIP.contains(e.target)) hideTip();
}});
// mouseover alone is not enough to hide: it only fires when some OTHER element
// receives the cursor, so leaving the window entirely leaves the tooltip
// pinned open over the page.
document.addEventListener('mouseout', e => {{
  const el = e.target.closest('[data-tip]');
  // mouseout also fires moving between an element and its own children.
  if (el && !el.contains(e.relatedTarget)) hideTip();
}});
document.addEventListener('focusin', e => {{
  const el = e.target.closest('[data-tip]');
  if (el) showTip(el);
}});
document.addEventListener('focusout', hideTip);
document.addEventListener('scroll', hideTip, true);
document.addEventListener('keydown', e => {{ if (e.key === 'Escape') hideTip(); }});
</script></body></html>"""


BUCKET_TIPS = {
    'AAA': 'Model-implied, not an agency rating. This bucket holds the '
           'strongest issuers by the six-factor scorecard — and the model is '
           'more generous than an agency would be, so read it as "top decile '
           'of this universe" rather than as a AAA.',
    'AA': 'Model-implied. Very strong issuers, one notch below the top band.',
    'A': 'Model-implied. Solid investment grade.',
    'BBB': 'Model-implied. The lowest investment-grade band, and the largest '
           'part of the real corporate bond market.',
    'BB': 'Model-implied. The top of high yield — often fallen angels or '
          'leveraged but stable issuers.',
    'B': 'Model-implied. Genuine high yield, where default risk is a real part '
         'of the return.',
    'CCC': 'Model-implied. Distressed. Very few bonds land here because the '
           'cutpoints are matched to the market\'s own mix, and the market '
           'prices almost nothing this wide.',
}


def _dist_rows(counts, order=None, total=None):
    """A label / bar / value grid for a small distribution."""
    items = ([(k, counts[k]) for k in order if k in counts] if order
             else sorted(counts.items(), key=lambda kv: -kv[1]))
    if not items:
        return '<span class="muted">no data</span>'
    top = max(v for _, v in items) or 1
    total = total or sum(v for _, v in items)
    out = []
    for label, value in items:
        width = 100.0 * value / top
        pct = 100.0 * value / total if total else 0
        tip = BUCKET_TIPS.get(label)
        attr = f' data-tip="{tip}"' if tip else ''
        out.append(
            f'<span{attr}>{label}</span>'
            f'<span><span class="bar" style="width:{width:.1f}%;display:block"></span></span>'
            f'<span class="muted">{value:,} &middot; {pct:.1f}%</span>')
    return ''.join(out)


def render(rows, meta, term, path):
    payload = build_payload(rows)

    ratings = payload['ratings']
    total = payload['total']
    capped_pct = round(100.0 * payload['capped'] / total) if total else 0
    mark_ages = [r.get('mark_age_days') for r in rows
                 if r.get('mark_age_days') is not None]
    mark_age = int(sorted(mark_ages)[len(mark_ages) // 2]) if mark_ages else 0

    card_specs = [
        ('Instruments', f'{total:,}',
         'Every US Treasury and corporate bond the model could price: held by '
         'at least two reporting funds, at least $10m in aggregate, and more '
         'than six months from maturity.'),
        ('BUY', f'{ratings.get("BUY", 0):,}',
         'Top of the composite ranking within its asset class, and not capped. '
         'Thresholds are quantile-matched per class, so this is deliberately a '
         'short list rather than everything that looks cheap.'),
        ('LEAN BUY', f'{ratings.get("LEAN BUY", 0):,}',
         'Ranks well but with less margin than a BUY, or carries a mild '
         'reservation.'),
        ('HOLD', f'{ratings.get("HOLD", 0):,}',
         'The middle of the distribution, plus everything demoted here by a '
         'cap. Most capped rows land in HOLD.'),
        ('PASS', f'{ratings.get("PASS", 0):,}',
         'Bottom of the ranking, or trading far through what the model thinks '
         'fair, or flagged for default, arrears or PIK.'),
        ('Capped', f'{capped_pct}%',
         'Share of rows whose rating was lowered because the data does not '
         'support acting on them — overwhelmingly because the issuer could not '
         'be identified. A capped row still shows its uncapped rating '
         'underneath in the data; the cap is a statement about our evidence, '
         'not about the bond.'),
        ('Median mark age', f'{mark_age}d',
         'How old the underlying fund price is. N-PORT publishes monthly with '
         'about a 60-day lag, so roughly three months is the normal state of '
         'this model, not a failure. Every price here is that mark aged onto '
         "today's curve."),
    ]
    cards = ''.join(
        f'<div class="card" data-tip="{tip.replace(chr(34), chr(39))}">'
        f'<div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v, tip in card_specs)

    classes = payload['classes']
    class_opts = ''.join(f'<option value="{c}">{c}</option>'
                         for c in sorted(classes))
    bucket_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
    bucket_opts = ''.join(f'<option value="{b}">{b}</option>'
                          for b in bucket_order if b in payload['buckets'])

    cap_svg = _svg_bars([(r[:26], n) for r, n in payload['capReasons']],
                        colour='#a8621b')
    cat_svg = _svg_bars([(c['label'], int(c['mean'])) for c in payload['categories']])

    html = PAGE.format(
        run_date=(meta or {}).get('run_date', ''),
        curve_date=(meta or {}).get('curve_date', ''),
        generated=datetime.now().strftime('%Y-%m-%d %H:%M'),
        total=total,
        cards=cards,
        capped_pct=capped_pct,
        mark_age=mark_age,
        rating_dist=_dist_rows(ratings, ['BUY', 'LEAN BUY', 'HOLD', 'PASS'], total),
        bucket_dist=_dist_rows(payload['buckets'], bucket_order),
        curve_svg=curve_chart(meta),
        term_svg=term_chart(term),
        cap_svg=cap_svg,
        cat_svg=cat_svg,
        class_opts=class_opts,
        bucket_opts=bucket_opts,
        payload=dumps_for_script(payload),
    )
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    size = os.path.getsize(path) / 1e6
    log.info('Wrote %s (%.1f MB, %d instruments)', os.path.basename(path),
             size, total)
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('snapshot', nargs='?', default=None,
                    help='results_*.parquet (default: the newest)')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--open', action='store_true', help='open it afterwards')
    args = ap.parse_args()

    path = args.snapshot
    if path is None:
        import glob
        candidates = sorted(glob.glob(os.path.join(OUTPUT_DIR,
                                                   'results_*.parquet')))
        if not candidates:
            raise SystemExit('[fatal] no results_*.parquet in output/')
        path = candidates[-1]

    stamp = os.path.basename(path)[8:18]
    rows = load_rows(path)
    meta = load_json(f'run_meta_{stamp}.json') or {}
    term = load_json('term_structure.json')

    out = args.out or os.path.join(OUTPUT_DIR, f'bond_analysis_{stamp}.html')
    render(rows, meta, term, out)
    print(f'\n  {out}\n  {os.path.getsize(out) / 1e6:.1f} MB, '
          f'{len(rows):,} instruments\n')

    if args.open:
        import subprocess
        subprocess.run(['open', out], check=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
