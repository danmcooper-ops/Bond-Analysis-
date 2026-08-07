# Bond Analysis Model

## Overview

Screens US corporate bonds (IG + HY) and US Treasuries into BUY / LEAN BUY /
HOLD / PASS, and accumulates a daily snapshot corpus. Free data sources only.

**Standalone project — no connection to any other repo.** It reads the sibling
equity model's output directory when one is present (issuer fundamentals it has
already computed), but degrades gracefully to SEC XBRL when it is absent. There
is no code dependency in either direction.

## What this is, and is not

A monthly-refreshed, fundamentally-anchored relative-value and credit-quality
screen over the **fund-held** US bond universe, with a daily mark-to-curve
overlay. It is not a bond desk: no live marks, no true OAS, no agency ratings,
no covenants. See "Honest limits" below — those constraints are load-bearing
and every one of them shows up in the report.

## Tech Stack

- Python 3.14 (`.venv/`)
- pandas, numpy, pyarrow, requests, lxml, jinja2, openpyxl, yfinance, curl_cffi
- **No scipy.** `models/solver.py` hand-rolls Brent + Newton so root-finding is
  testable and can honour the "return None, never raise" contract.

## Project Structure

```
data/     - ingestion clients (Treasury curve, FRED, N-PORT, TreasuryDirect,
            CUSIP crosswalk, issuer fundamentals) + vendored kernel modules
models/   - fixed-income analytics, all greenfield (daycount, schedule,
            pricing, curve, risk, spreads, total_return, credit)
scripts/  - scoring kernel, gate definitions, pipeline entry points, reporting
templates/- report.html (Jinja2 shell + client-side JS renderer)
tools/    - kernel_diff.py, the vendored-code drift report
tests/    - pytest; golden-value tests for the bond math
output/   - run artifacts (git-ignored); the accumulating snapshot corpus
docs/     - published report; tracked only on the `pages-live` branch
```

No `__init__.py` anywhere — scripts do `sys.path.insert(0, repo_root)`, the
same convention as the equity model.

## Data sources (all free)

| Source | Gives us | Cadence / lag |
|---|---|---|
| **SEC Form N-PORT Data Sets** (DERA bulk quarterly TSV) | per-CUSIP marks from every registered fund's holdings, plus coupon, maturity, default/arrears/PIK flags | monthly period-end, ~60-day lag |
| **home.treasury.gov** par yield curve XML | full-tenor daily par curve | daily, T+1, no key |
| **FRED** | `DGS*` constant-maturity yields; ICE BofA OAS by rating bucket; IG maturity slices for the spread term structure | daily, free key (keyless `fredgraph.csv` fallback) |
| **TreasuryDirect** | exact coupon/maturity per Treasury CUSIP | weekly refresh, no key |
| **SEC XBRL / equity snapshot** | issuer fundamentals for the credit scorecard | quarterly filings |
| **yfinance** | bond ETF total-return benchmarks | daily |

**Not used: FINRA TRACE.** The API Platform is $1,650/mo for firm credentials;
free credentials get aggregate datasets only, not per-CUSIP prices. Do not
build on it.

## Key design decisions

**The scoring kernel is vendored, not shared.** `scripts/scoring_kernel.py` is
copied from the equity model, not symlinked — a symlink would mean an edit over
there silently changes bond ratings, and a fresh clone or CI checkout would
break. `python tools/kernel_diff.py` reports drift function-by-function and
never auto-applies.

**Applicable vs missing is the central mechanism.** A gate whose
`applicable(row)` is False is excluded from numerator *and* denominator; a gate
with missing data scores 0 and stays in the denominator. This is what lets a
Treasury and a corporate bond share one rating scale: a Treasury masks every
credit gate, the whole Credit category drops out, and the composite
renormalises over the surviving categories. Upstream marks that path
"unreachable — defensive only"; here it is the normal path for every Treasury.
`tests/test_kernel.py` pins it.

**Per-asset-class rating thresholds, from v1.** Because a Treasury scores 10
gates across 3 categories and a corporate 25 across 5, their composites are not
on one scale. One threshold set would systematically mis-rate a class.

**Snapshots are parquet, not JSON.** ~30k rows x ~120 keys of pretty JSON would
be 45-75 MB/day — the exact problem that forced the equity repo onto a
single-commit `pages-live` branch. Parquet is ~5-10 MB/day and makes
backtest/calibrate (which load dozens of snapshots) roughly 10x faster.

**Publishing uses `pages-live` from day one.** Copy artifacts into the
worktree, `git commit --amend`, force-push. History never accumulates
artifacts.

## Honest limits

These are not caveats to bury; they belong on the report itself.

- **No real daily prices.** Marks are monthly at ~60-day lag. Every daily
  number is an extrapolation (spread aged forward on the FRED bucket-OAS move)
  and is labelled `clean_price_est`, never `price`.
- **No true OAS.** That needs a call schedule and a swaption vol surface,
  neither of which is free. We report Z-spread and compare it to the FRED
  bucket OAS index with a wedge *fitted from data, not assumed*. For callables
  priced above par with an unknown call schedule, the signal is weak — that
  case is a HOLD cap rather than a pretence.
- **No agency ratings.** The implied bucket is a fundamentals scorecard
  calibrated against market spreads. It will disagree with Moody's/S&P —
  sometimes correctly (that is the divergence signal), sometimes not.
- **No covenants.** Seniority is regex-inferred from the N-PORT title and
  defaults to senior unsecured, marked `seniority_source='default'`.
- **Coverage is what funds hold.** A bond no registered fund owns is invisible.
  The universe is survivorship-shaped toward index-eligible, liquid names.
- **A stale mark can masquerade as mispricing.** "The market hasn't caught up"
  is often "our data hasn't caught up". Divergence only counts as a fallen
  angel when the fundamental deterioration predates the mark date.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Pipeline (see the plan for the full milestone sequence):

```bash
python scripts/ingest_nport.py --quarter 2026q1        # monthly, 30-90 min
python scripts/build_universe.py --month 2026-06
python scripts/analyze_bonds.py --as-of 2026-08-06     # daily, 5-15 min
python scripts/report_html.py                          # render the HTML page
python scripts/publish.py                              # amend + force-push pages-live
```

## Conventions

- Ingestion clients are classes; analytics are standalone pure functions.
- Analytics never raise on bad input — they return `None`, and the caller sets
  a diagnostic field that a rating cap keys off.
- Every row records where its data came from and how old it is
  (`_fundamentals_source`, `_fundamentals_asof`, `mark_date`, `mark_age_days`,
  `cusip_match_method`, `cusip_match_confidence`).
- Enrichment scripts are idempotent: strip, recompute, reattach.

## Scheduling

Use the Claude scheduled-task mechanism, **not launchd**. A launchd-spawned
git/python cannot reach a repo under the TCC-protected `~/Desktop` — the
Real-Estate-Model repo documents this lesson; inherit it rather than
rediscovering it.
