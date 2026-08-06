# Bond Analysis Model

Screens US corporate bonds (investment grade and high yield) and US Treasuries
into **BUY / LEAN BUY / HOLD / PASS**, using only free data sources.

It answers one question per bond: *is the spread enough to compensate for this
issuer's credit risk, and is the duration right for the current curve?*

## How it works

1. **Marks** come from SEC Form N-PORT — every registered fund and ETF discloses
   its monthly holdings, so `VALUE_USD / BALANCE x 100` is a price per 100 face.
   Many funds hold the same CUSIP, so the median across them (with outliers
   rejected on a MAD test) is a defensible consensus mark.
2. **The risk-free curve** comes from the daily Treasury par yield curve,
   bootstrapped to zeros with monotone-cubic interpolation.
3. **A Z-spread** falls out of the mark and the curve.
4. **A fair spread** comes from the issuer's fundamentals: a transparent
   six-factor credit scorecard maps to a rating bucket, the FRED ICE BofA OAS
   index gives that bucket's market spread, and a maturity term factor adjusts
   it for this bond's tenor.
5. **The gap between observed and fair spread** is the valuation signal — the
   fixed-income analogue of margin of safety. Where the fundamental bucket and
   the market-implied bucket disagree, that is the rising-star / fallen-angel
   divergence signal.
6. **25 gates across 5 categories** (Valuation, Credit, Rates, Structure,
   Liquidity) roll up to a composite and a rating, with investability caps that
   demote anything the data cannot actually support.

Treasuries flow through the same machinery: every credit gate is structurally
inapplicable, the Credit category drops out, and the composite renormalises
over the rest. A Treasury is rated on yield versus cash, duration fit to the
curve regime, roll-down, and liquidity — which is how a Treasury should be
rated.

## What it is not

Not a bond desk. Marks are monthly at roughly a 60-day lag, so every daily
price is a labelled extrapolation. There is no true OAS (that needs call
schedules and a swaption vol surface, neither of which is free), no agency
ratings, and no covenant data. `CLAUDE.md` has the full list under "Honest
limits", and the report itself states them.

For a buyer holding individual bonds to maturity — where "is this issuer's
coverage deteriorating, and am I paid for the risk" matters far more than a
60-day-stale mark — that is a useful trade.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Price a single bond, no data setup required:

```bash
python scripts/price_bond.py --coupon 4.5 --maturity 2034-11-15 --price 97.25
```

## Layout

See `CLAUDE.md` for the directory map, data sources, design decisions, and the
scheduling notes.

## Relationship to the equity model

Standalone. The generic scoring machinery is **vendored** (copied, with the
upstream SHA recorded in each file header) rather than shared, so an edit to the
equity model can never silently change a bond rating. `python
tools/kernel_diff.py` reports drift function-by-function and never auto-applies.

When the sibling equity model's `output/` is present, issuer fundamentals are
read from its latest snapshot for free; when it is absent, they are fetched from
SEC XBRL instead. Neither repo imports the other.
