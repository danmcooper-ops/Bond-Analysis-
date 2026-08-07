#!/usr/bin/env python3
"""Measure the credit spread term structure from our own observed spreads.

    python scripts/fit_term_structure.py
    python scripts/fit_term_structure.py --apply

WHY THIS EXISTS
---------------
`fair_spread` scales a rating bucket's index OAS by a term factor, because a
2-year BBB and a 30-year BBB do not deserve the same spread. The factors came
from FRED's IG maturity slices, which stop at "15y+". Beyond roughly twenty
years the factor was therefore flat-extrapolated, and a 40-year bond was
assigned the same 1.27x as a 20-year one.

Extrapolating the slope would be inventing data. Measuring it is not: the
N-PORT panel contains roughly 98,000 investment-grade spread observations
across nine months and maturities out past forty years. This fits the shape
from those.

WHAT THE MEASUREMENT FOUND, WHICH WAS NOT WHAT WAS EXPECTED
-----------------------------------------------------------
    tenor      measured   FRED
    3-5y         0.91x    0.86
    5-7y         1.01x    1.03
    7-10y        1.12x    1.22
    10-15y       1.12x    1.18
    15-20y       1.09x    1.27     <-- FRED is 17% high here
    20-25y       1.08x     —
    25-32y       1.20x     —
    32y+         1.20x     —

The two agree closely from three to ten years, which is what makes the rest
believable — a fit that disagreed everywhere would just be measuring our own
pricing errors. Past ten years they part company: the real curve is nearly
FLAT, while FRED keeps climbing to 1.27x.

So the original diagnosis was wrong in direction. Flat extrapolation was not
understating the long end and making long bonds look cheap; FRED's 15y+ factor
OVERSTATES the fair spread out there, and the flat extrapolation carried that
overstatement further still. Long-dated fair spreads were too wide, which
distorts both the mispricing signal and the market-implied bucket.

Why would FRED's slices climb when like-for-like spreads do not? Because they
are sub-indices with different constituents: only the strongest issuers sell
forty-year paper, so the long slices are not the same credits as the short
ones. Ratio-ing one sub-index to another silently compares different
populations. Measuring across a single population does not.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.logging_setup import get_logger
from data.nport_client import NPORTClient
from data.treasury_curve_client import TreasuryCurveClient
from models.bond_types import from_row
from models.curve import YieldCurve
from models.pricing import bond_flows_and_stub
from models.schedule import accrued_interest, years_to_maturity
from models.spreads import z_spread

log = get_logger('fit_term_structure')

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
TERM_STRUCTURE_PATH = os.path.join(OUTPUT_DIR, 'term_structure.json')

# (lower, upper, label, midpoint used for interpolation)
BUCKETS = (
    (0.0, 3.0, '0-3y', 1.5), (3.0, 5.0, '3-5y', 4.0),
    (5.0, 7.0, '5-7y', 6.0), (7.0, 10.0, '7-10y', 8.5),
    (10.0, 15.0, '10-15y', 12.5), (15.0, 20.0, '15-20y', 17.5),
    (20.0, 25.0, '20-25y', 22.5), (25.0, 32.0, '25-32y', 28.5),
    (32.0, 100.0, '32y+', 38.0),
)

# Investment-grade proxy. The bucket labels are unavailable for most bonds
# (the crosswalk resolves under half the universe), so quality is proxied by
# spread level. Mixing high yield in would corrupt the shape badly: HY is
# heavily short-dated, so it inflates the short buckets and manufactures a
# rising term structure that is really a credit-mix artifact.
IG_SPREAD_MIN, IG_SPREAD_MAX = 0.0, 0.025

# THE TERM SHAPE IS NOT ONE CURVE. Measured across 72,000 observations, tight
# and mid credits both rise with maturity (~0.9x short to ~1.15x long) while
# WIDE credits INVERT — 194bp at 5-7y falling to 138bp at 20-25y. That is the
# classic distressed pattern: risk concentrates in near-dated paper, because a
# struggling issuer's problem is refinancing the next maturity, not the one in
# twenty years.
#
# A single universe-average factor therefore gets wide credits backwards by
# roughly 47%, assigning them a RISING fair spread where the market charges a
# falling one. Fitting one curve per tier costs nothing and fixes it.
#
# Tiers are keyed off observed spread rather than rating bucket because the
# crosswalk resolves under half the universe, and a tier assignment that only
# existed for matched issuers would leave the rest on the wrong curve.
TIERS = (
    ('tight', 0.0, 0.006),
    ('mid', 0.006, 0.012),
    ('wide', 0.012, 0.030),
)

# Which tier a rating bucket belongs to, for scoring.
BUCKET_TIER = {'AAA': 'tight', 'AA': 'tight', 'A': 'tight',
               'BBB': 'mid', 'BB': 'wide', 'B': 'wide', 'CCC': 'wide'}

MIN_PER_BUCKET = 50
MIN_MONTH_SIZE = 2000

# FRED's published factors, for the overlap comparison that validates the fit.
FRED_REFERENCE = {'0-3y': 0.59, '3-5y': 0.86, '5-7y': 1.03,
                  '7-10y': 1.22, '10-15y': 1.18, '15-20y': 1.27}


def _bucket_for(years):
    for lo, hi, label, midpoint in BUCKETS:
        if lo <= years < hi:
            return label, midpoint
    return None, None


def collect_spreads(min_funds=5):
    """Observed Z-spreads by maturity bucket, across every ingested month."""
    import pandas as pd

    client = NPORTClient()
    names = sorted(p for p in os.listdir(client.cache_dir)
                   if p.endswith('_marks.parquet')) \
        if os.path.isdir(client.cache_dir) else []
    if not names:
        raise SystemExit('[fatal] no marks — run scripts/ingest_nport.py first')

    frames = [pd.read_parquet(os.path.join(client.cache_dir, n)) for n in names]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[(panel['issuer_type'] == 'CORP')
                  & (panel['n_funds'] >= min_funds)
                  & (~panel['is_default']) & (~panel['is_convertible'])]
    panel = panel.drop_duplicates(subset=['cusip', 'report_date'])
    panel['month'] = panel['report_date'].astype(str).str[:10]

    months = [m for m, n in panel['month'].value_counts().items()
              if n >= MIN_MONTH_SIZE]
    log.info('Fitting across %d months with %d+ observations each',
             len(months), MIN_MONTH_SIZE)

    tc = TreasuryCurveClient()
    buckets = defaultdict(list)
    total = 0
    for month in sorted(months):
        as_of = datetime.strptime(month, '%Y-%m-%d').date()
        curve_date, par = tc.fetch_par_curve(as_of)
        if not par:
            continue
        curve = YieldCurve.from_par_dict(curve_date, par)

        for record in panel[panel['month'] == month].to_dict('records'):
            row = {**record, 'coupon_rate': record.get('annualized_rate')}
            maturity = row.get('maturity_date')
            if maturity is not None and hasattr(maturity, 'date'):
                row['maturity_date'] = maturity.date()
            bond, _ = from_row(row, settle=curve_date)
            if bond is None:
                continue
            flows, _ = bond_flows_and_stub(
                bond.coupon_rate, bond.maturity, curve_date,
                frequency=bond.frequency, convention=bond.convention,
                face=bond.face)
            if not flows:
                continue
            accrued = accrued_interest(curve_date, bond.coupon_rate,
                                       bond.maturity, frequency=bond.frequency,
                                       face=bond.face,
                                       convention=bond.convention)
            spread = z_spread(record['clean_price_marked'] + accrued, flows,
                              curve_date, curve)
            if spread is None or not (IG_SPREAD_MIN < spread < IG_SPREAD_MAX):
                continue
            label, _ = _bucket_for(years_to_maturity(curve_date, bond.maturity))
            if not label:
                continue
            buckets[label].append(spread)
            for tier, lo, hi in TIERS:
                if lo <= spread < hi:
                    buckets[f'{tier}|{label}'].append(spread)
                    break
            total += 1

    log.info('Collected %d investment-grade spread observations', total)
    return buckets


def fit(buckets):
    """Normalise each bucket's median spread to the whole-universe median."""
    everything = [s for values in buckets.values() for s in values]
    if len(everything) < 500:
        raise SystemExit(f'[fatal] only {len(everything)} observations')
    overall = statistics.median(everything)

    points, table = [], []
    for _lo, _hi, label, midpoint in BUCKETS:
        values = buckets.get(label, [])
        if len(values) < MIN_PER_BUCKET:
            continue
        median = statistics.median(values)
        ratio = median / overall
        points.append((midpoint, round(ratio, 4)))
        table.append({'bucket': label, 'midpoint_years': midpoint,
                      'n': len(values), 'median_spread': round(median, 6),
                      'factor': round(ratio, 4),
                      'fred_factor': FRED_REFERENCE.get(label)})
    # Per-tier shapes, each normalised to its OWN 3-5y anchor so the curve is
    # a pure shape and the level still comes from the bucket OAS.
    by_tier = {}
    for tier, _lo, _hi in TIERS:
        anchor_values = buckets.get(f'{tier}|3-5y', [])
        if len(anchor_values) < MIN_PER_BUCKET:
            continue
        anchor = statistics.median(anchor_values)
        tier_points, tier_table = [], []
        for _l, _h, label, midpoint in BUCKETS:
            values = buckets.get(f'{tier}|{label}', [])
            if len(values) < MIN_PER_BUCKET:
                continue
            median = statistics.median(values)
            tier_points.append((midpoint, round(median / anchor, 4)))
            tier_table.append({'bucket': label, 'n': len(values),
                               'median_spread': round(median, 6),
                               'factor': round(median / anchor, 4)})
        if len(tier_points) >= 4:
            by_tier[tier] = {'points': tier_points, 'table': tier_table,
                             'anchor_spread': round(anchor, 6)}

    return {'points': points, 'table': table, 'by_tier': by_tier,
            'overall_median_spread': round(overall, 6),
            'n_observations': len(everything),
            'ig_spread_band': [IG_SPREAD_MIN, IG_SPREAD_MAX],
            'bucket_tier': BUCKET_TIER,
            'source': 'nport_panel'}


def report(fitted):
    print(f"\n{'=' * 74}")
    print(f"  MEASURED SPREAD TERM STRUCTURE  —  "
          f"{fitted['n_observations']:,} IG observations")
    print(f"{'=' * 74}")
    print(f"\n  {'bucket':<9}{'n':>8}{'median':>10}{'measured':>11}"
          f"{'FRED':>8}{'gap':>9}")
    overlap = []
    for row in fitted['table']:
        fred = row['fred_factor']
        gap = f"{(row['factor'] / fred - 1) * 100:>+7.0f}%" if fred else '      —'
        if fred:
            overlap.append(abs(row['factor'] / fred - 1))
        print(f"  {row['bucket']:<9}{row['n']:>8}"
              f"{row['median_spread'] * 10000:>8.0f}bp"
              f"{row['factor']:>10.2f}x"
              f"{(f'{fred:.2f}x' if fred else '—'):>8}{gap}")

    if overlap:
        near = [g for g in overlap[:4]]
        print(f"\n  Agreement over the overlap where FRED has data is what makes")
        print(f"  the rest believable: mean |gap| of {statistics.mean(near) * 100:.0f}% "
              f"across the first four buckets.")

    tiers = fitted.get('by_tier') or {}
    if tiers:
        print(f"\n  BY CREDIT TIER (each normalised to its own 3-5y anchor)")
        labels = [r['bucket'] for r in tiers[next(iter(tiers))]['table']]
        print(f"    {'tenor':<9}" + ''.join(f"{t:>10}" for t in tiers))
        rows_by_label = {}
        for tier, payload in tiers.items():
            for row in payload['table']:
                rows_by_label.setdefault(row['bucket'], {})[tier] = row['factor']
        for _l, _h, label, _m in BUCKETS:
            if label not in rows_by_label:
                continue
            line = f"    {label:<9}"
            for tier in tiers:
                v = rows_by_label[label].get(tier)
                line += f"{(f'{v:.2f}x' if v else '—'):>10}"
            print(line)
        wide = tiers.get('wide')
        if wide:
            first, last = wide['table'][0]['factor'], wide['table'][-1]['factor']
            if last < first:
                print(f"\n    The WIDE tier INVERTS: {first:.2f}x at the short end down")
                print(f"    to {last:.2f}x at the long end. Distressed risk sits in")
                print(f"    near-dated paper — the issue is refinancing the next")
                print(f"    maturity, not the one in twenty years. A single rising")
                print(f"    universe curve gets these backwards.")

    long_end = [r for r in fitted['table'] if r['midpoint_years'] >= 15]
    if long_end:
        print(f"\n  BEYOND 15 YEARS — where FRED has nothing and the old model")
        print(f"  flat-extrapolated 1.27x:")
        for row in long_end:
            print(f"    {row['bucket']:<9}{row['factor']:>6.2f}x   "
                  f"(old model assumed 1.27x)")
        worst = max(long_end, key=lambda r: abs(r['factor'] - 1.27))
        print(f"\n  The old assumption overstated the {worst['bucket']} fair spread")
        print(f"  by {(1.27 / worst['factor'] - 1) * 100:.0f}%. An overstated fair "
              f"spread makes a bond look")
        print(f"  RICHER than it is, which biases the mispricing signal against")
        print(f"  long paper and shifts its market-implied bucket.")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-funds', type=int, default=5)
    ap.add_argument('--apply', action='store_true',
                    help='write output/term_structure.json for the model to use')
    args = ap.parse_args()

    fitted = fit(collect_spreads(min_funds=args.min_funds))
    fitted['fitted_at'] = date.today().isoformat()
    report(fitted)

    if args.apply:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(TERM_STRUCTURE_PATH, 'w', encoding='utf-8') as fh:
            json.dump(fitted, fh, indent=2)
        print(f"  Wrote {TERM_STRUCTURE_PATH}")
        print(f"  fair_spread will prefer this over the FRED slices.\n")
    else:
        print("  Re-run with --apply to write it for the model to use.\n")
    return 0


def load_fitted(path=None):
    """Read the fitted structure, or None. Used by the pipeline."""
    path = path or TERM_STRUCTURE_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    points = [(float(t), float(f)) for t, f in payload.get('points', [])]
    return points or None


def load_tiered(path=None):
    """{bucket: [(years, factor)]} — a term shape per rating bucket.

    Returns None when no tiered fit exists, so callers fall back to the single
    curve. Buckets share a tier: AAA/AA/A ride the tight curve, BBB the mid
    one, and BB/B/CCC the wide one that INVERTS with maturity.
    """
    path = path or TERM_STRUCTURE_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    by_tier = payload.get('by_tier') or {}
    mapping = payload.get('bucket_tier') or BUCKET_TIER
    if not by_tier:
        return None
    out = {}
    for bucket, tier in mapping.items():
        entry = by_tier.get(tier)
        if entry and entry.get('points'):
            out[bucket] = [(float(t), float(f)) for t, f in entry['points']]
    return out or None


if __name__ == '__main__':
    sys.exit(main())
