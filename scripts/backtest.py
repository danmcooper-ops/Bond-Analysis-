#!/usr/bin/env python3
"""Does any of this predict returns?

    python scripts/backtest.py
    python scripts/backtest.py --min-funds 5 --min-held 25e6

THE MILESTONE THAT VALIDATES OR KILLS THE THESIS. Everything up to here has
been construction: prices parsed, spreads computed, issuers matched, a ranking
produced. None of it is evidence that the ranking is worth acting on. This is.

MARKED-TO-MARKED, NOT MARKED-TO-MODEL
--------------------------------------
Returns are measured between two REAL N-PORT observations of the same CUSIP,
one month apart. The daily `clean_price_est` the pipeline produces is a model
output — ageing an old mark onto today's curve — and testing a signal against
a price the same model extrapolated grades the model partly on its own
extrapolation. That is the single easiest way to manufacture a backtest that
looks good and means nothing. Monthly, genuinely out-of-sample observations
are fewer and noisier, and they are the ones worth believing.

TOTAL RETURN, NOT PRICE RETURN. Coupon income dominates over a month for a
bond, and ranking on price change alone would sort the book roughly by coupon,
backwards.

EXCESS OVER A DURATION-MATCHED TREASURY. A month when yields fell rewards
every long bond regardless of credit. Subtracting the return of a Treasury
with the same duration isolates what the model actually claims skill at —
picking credits and relative value — from the duration exposure it merely
carries.

THE THREE TESTS
---------------
    mispricing decile   is the valuation signal monotone in forward excess
                        return? if it is not, the fair-spread model is not
                        measuring anything.
    divergence          do rising stars beat fallen angels? this is the
                        headline claim; if it fails, the claim is wrong.
    bucket ordering     does the implied bucket predict forward SPREAD
                        change? with no default data, this is the closest
                        available proxy for credit-model skill.

Every test reports a per-period breakdown as well as a pooled figure. Bond
panels are strongly cross-correlated — in a month when spreads widen, almost
everything loses — so a pooled t-statistic over thousands of bond-months
massively overstates the evidence. What matters is whether the effect shows up
in MOST PERIODS, not whether it is large once.
"""

import argparse
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cusip_crosswalk import CusipCrosswalk
from data.fred_client import FREDClient
from data.issuer_fundamentals import IssuerFundamentals
from data.logging_setup import get_logger
from data.nport_client import NPORTClient
from data.treasury_curve_client import TreasuryCurveClient
from models import credit
from models.bond_types import from_row
from models.curve import YieldCurve
from models.pricing import bond_flows_and_stub
from models.risk import convexity, macaulay_duration, modified_duration
from models.schedule import accrued_interest, years_to_maturity
from models.spreads import z_spread
from models.total_return import (coupons_between,
                                 duration_matched_treasury_return,
                                 realized_total_return)
from scripts.param_set import default_params

log = get_logger('backtest')

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')

# A month-on-month pair must be roughly a month apart. Gaps happen when a fund
# skips a filing; stitching across them mixes horizons.
MIN_GAP_DAYS, MAX_GAP_DAYS = 20, 45

MIN_PER_PERIOD = 50          # below this a period's statistics are noise


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def load_panel(min_funds, min_held):
    """Every (cusip, month) mark across all ingested quarters."""
    import pandas as pd

    client = NPORTClient()
    paths = sorted(p for p in os.listdir(client.cache_dir)
                   if p.endswith('_marks.parquet')) \
        if os.path.isdir(client.cache_dir) else []
    if not paths:
        raise SystemExit('[fatal] no marks parquet files — run '
                         'scripts/ingest_nport.py first')

    frames = []
    for name in paths:
        frame = pd.read_parquet(os.path.join(client.cache_dir, name))
        frames.append(frame)
        log.info('%s: %d CUSIP-months', name, len(frame))
    panel = pd.concat(frames, ignore_index=True)

    panel = panel[(panel['n_funds'] >= min_funds)
                  & (panel['total_held_usd'] >= min_held)
                  & (panel['issuer_type'] == 'CORP')
                  & (~panel['is_default'])
                  & (~panel['in_arrears'])
                  & (~panel['is_convertible'])]

    # Two quarters can report the same CUSIP-month; keep the widest coverage.
    panel = (panel.sort_values('n_funds', ascending=False)
             .drop_duplicates(subset=['cusip', 'report_date'], keep='first'))

    rows = panel.to_dict('records')
    for row in rows:
        for field in ('report_date', 'maturity_date'):
            value = row.get(field)
            if value is not None and hasattr(value, 'date'):
                row[field] = value.date()
    log.info('Panel: %d observations, %d CUSIPs, %d months',
             len(rows), len({r['cusip'] for r in rows}),
             len({r['report_date'] for r in rows}))
    return rows


def build_pairs(rows):
    """Consecutive same-CUSIP observations roughly a month apart."""
    by_cusip = defaultdict(list)
    for row in rows:
        by_cusip[row['cusip']].append(row)

    pairs, skipped = [], Counter()
    for cusip, observations in by_cusip.items():
        observations.sort(key=lambda r: r['report_date'])
        for earlier, later in zip(observations, observations[1:]):
            gap = (later['report_date'] - earlier['report_date']).days
            if gap < MIN_GAP_DAYS:
                skipped['gap too short'] += 1
                continue
            if gap > MAX_GAP_DAYS:
                skipped['gap too long (missed filing)'] += 1
                continue
            pairs.append((earlier, later, gap))
    log.info('Pairs: %d usable, %s', len(pairs),
             ', '.join(f'{k}={v}' for k, v in skipped.items()) or 'none skipped')
    return pairs


# ---------------------------------------------------------------------------
# Point-in-time signals
# ---------------------------------------------------------------------------

class PointInTime:
    """Curves, spreads and fundamentals as they stood on each mark date.

    Every input is pinned to the observation date. Using today's curve or
    today's balance sheet to score a 2025 observation is look-ahead bias, and
    it is the failure mode that makes a backtest look best right before it
    disappoints in production.
    """

    def __init__(self, allow_lookahead=False):
        self._allow_lookahead = allow_lookahead
        self._curves = {}
        self._oas = {}
        self._term = {}
        self._fundamentals = {}
        self._crosswalks = {}
        self._tc = TreasuryCurveClient()
        self._fred = FREDClient()

    def curve(self, when):
        if when not in self._curves:
            curve_date, par = self._tc.fetch_par_curve(when)
            self._curves[when] = (YieldCurve.from_par_dict(curve_date, par)
                                  if par else None)
        return self._curves[when]

    def bucket_oas(self, when):
        if when not in self._oas:
            self._oas[when] = self._fred.fetch_bucket_oas(when)
        return self._oas[when]

    def term_points(self, when):
        """The fitted structure when available, else FRED's slices.

        The fit is not point-in-time — it pools every month — so it does carry
        a little hindsight about the average SHAPE of the curve. That is a far
        smaller contamination than using FRED's slices, which are wrong by up
        to 17% at the long end in every period, and the shape moves slowly
        enough that pooling it is defensible where using a future balance
        sheet is not.
        """
        if when not in self._term:
            from scripts.fit_term_structure import load_fitted
            fitted = load_fitted()
            self._term[when] = (fitted if fitted
                                else self._fred.fetch_term_factors(when)['points'])
        return self._term[when]

    def fundamentals(self, when):
        """Fundamentals as they stood on `when`.

        With allow_lookahead the newest snapshot is used regardless of date.
        That is CONTAMINATED and clearly labelled everywhere it surfaces: it
        scores a 2025 observation with a balance sheet published in 2026, so
        any result flatters the model by exactly the amount of hindsight
        involved. It exists for one purpose — telling a WEAK signal apart from
        a BROKEN one when the clean test has no data at all. A contaminated
        result that shows nothing means the machinery is wrong; a contaminated
        result that shows something means the machinery works and is still
        unvalidated.
        """
        key = when if not self._allow_lookahead else 'latest'
        if key not in self._fundamentals:
            source = IssuerFundamentals(
                as_of=None if self._allow_lookahead else when)
            self._fundamentals[key] = source
            self._crosswalks[key] = CusipCrosswalk(index=source.names())
        return self._fundamentals[key], self._crosswalks[key]


def signals_at(row, pit, params):
    """Score one observation using only information available on its date."""
    when = row['report_date']
    curve = pit.curve(when)
    if curve is None:
        return None

    bond, reason = from_row({**row, 'coupon_rate': row.get('annualized_rate')},
                            settle=when)
    if bond is None:
        return None

    flows, _ = bond_flows_and_stub(bond.coupon_rate, bond.maturity, when,
                                   frequency=bond.frequency,
                                   convention=bond.convention, face=bond.face)
    if not flows:
        return None

    accrued = accrued_interest(when, bond.coupon_rate, bond.maturity,
                               frequency=bond.frequency, face=bond.face,
                               convention=bond.convention)
    clean = row['clean_price_marked']
    z = z_spread(clean + accrued, flows, when, curve)
    if z is None:
        return None

    ttm = years_to_maturity(when, bond.maturity)
    ytm_guess = bond.coupon_rate + z
    mac = macaulay_duration(flows, ytm_guess, frequency=bond.frequency)
    mod = modified_duration(mac, ytm_guess, frequency=bond.frequency)
    cvx = convexity(flows, ytm_guess, frequency=bond.frequency)

    source, crosswalk = pit.fundamentals(when)
    resolution = crosswalk.resolve(row['cusip'], [row.get('issuer_name')])
    entry = source.get(resolution.get('key')) if resolution.get('key') else None

    bucket = None
    if entry and (resolution.get('confidence') or 0) >= 0.80:
        result = credit.implied_bucket(
            {'int_cov': entry.get('int_cov'),
             'nd_ebitda': entry.get('nd_ebitda'),
             'fcf': entry.get('fcf'), 'total_debt': entry.get('total_debt'),
             'altman_z': entry.get('altman_z'),
             'revenue': entry.get('revenue'),
             'piotroski': entry.get('piotroski'),
             'cet1_ratio': entry.get('cet1_ratio'),
             'npl_ratio': entry.get('npl_ratio'),
             'sector': entry.get('sector')},
            sector=entry.get('sector'), params=params)
        bucket = result.get('bucket')

    oas = pit.bucket_oas(when)
    term = pit.term_points(when)
    beta = params.get('fair_spread_term_beta', 1.0)
    from scripts.fit_term_structure import load_tiered
    tiered = load_tiered()
    fair = credit.fair_spread(bucket, ttm, oas, term_points=term, beta=beta,
                              term_by_bucket=tiered)
    market = credit.market_implied_bucket(z, ttm, oas, term_points=term,
                                          beta=beta, term_by_bucket=tiered)
    gap = credit.divergence(bucket, market)

    return {
        'bond': bond, 'z_spread': z, 'modified_duration': mod,
        'convexity': cvx, 'years_to_maturity': ttm, 'accrued': accrued,
        'implied_bucket': bucket, 'market_bucket': market,
        'divergence': gap['notches'],
        'fair_spread': fair,
        'spread_mispricing': credit.spread_mispricing(z, fair),
    }


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def measure(pairs, pit, params):
    """Signal at t0 joined to realised excess return t0 -> t1."""
    out, dropped = [], Counter()
    for earlier, later, gap in pairs:
        signal = signals_at(earlier, pit, params)
        if signal is None:
            dropped['unscorable at t0'] += 1
            continue
        bond = signal['bond']

        accrued_end = accrued_interest(later['report_date'], bond.coupon_rate,
                                       bond.maturity, frequency=bond.frequency,
                                       face=bond.face,
                                       convention=bond.convention)
        coupons = coupons_between(bond.coupon_rate, bond.maturity,
                                  earlier['report_date'], later['report_date'],
                                  frequency=bond.frequency, face=bond.face)
        total = realized_total_return(earlier['clean_price_marked'],
                                      later['clean_price_marked'],
                                      signal['accrued'], accrued_end, coupons)
        if total is None:
            dropped['no return'] += 1
            continue

        curve0 = pit.curve(earlier['report_date'])
        curve1 = pit.curve(later['report_date'])
        benchmark = duration_matched_treasury_return(
            signal['modified_duration'], signal['convexity'], curve0, curve1,
            signal['years_to_maturity'], gap) if curve1 else None
        if benchmark is None:
            dropped['no benchmark'] += 1
            continue

        # A month-on-month move beyond this is a data error, not a market
        # move; leaving it in would let one bad mark dominate a decile.
        if abs(total) > 0.5:
            dropped['implausible return'] += 1
            continue

        end_signal = signals_at(later, pit, params)
        out.append({
            'cusip': earlier['cusip'],
            'period': earlier['report_date'],
            'horizon_days': gap,
            'total_return': total,
            'excess_return': total - benchmark,
            'z_spread': signal['z_spread'],
            'spread_mispricing': signal['spread_mispricing'],
            'divergence': signal['divergence'],
            'implied_bucket': signal['implied_bucket'],
            'market_bucket': signal['market_bucket'],
            'spread_change': (end_signal['z_spread'] - signal['z_spread']
                              if end_signal else None),
        })

    log.info('Measured %d bond-months (%s)', len(out),
             ', '.join(f'{k}={v}' for k, v in dropped.items()) or 'none dropped')
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _mean(values):
    return sum(values) / len(values) if values else None


def _by_period(records, key):
    groups = defaultdict(list)
    for record in records:
        if record.get(key) is not None and record.get('excess_return') is not None:
            groups[record['period']].append(record)
    return {p: rs for p, rs in groups.items() if len(rs) >= MIN_PER_PERIOD}


def decile_test(records, key, label, n_buckets=5):
    """Is `key` monotone in forward excess return?

    Reported per period as well as pooled, because bond returns are heavily
    cross-correlated: a pooled statistic over thousands of bond-months
    massively overstates the independent evidence. The number that matters is
    how OFTEN the top bucket beats the bottom, not by how much it did once.
    """
    periods = _by_period(records, key)
    if not periods:
        print(f"\n  {label}: no period has {MIN_PER_PERIOD}+ observations")
        return None

    spreads, bucket_means = [], defaultdict(list)
    for period, rows in sorted(periods.items()):
        rows = sorted(rows, key=lambda r: r[key])
        size = len(rows) // n_buckets
        if size < 5:
            continue
        buckets = [rows[i * size:(i + 1) * size] for i in range(n_buckets)]
        means = [_mean([r['excess_return'] for r in b]) for b in buckets]
        for i, m in enumerate(means):
            bucket_means[i].append(m)
        spreads.append((period, means[-1] - means[0], len(rows)))

    if not spreads:
        return None

    print(f"\n  {label}")
    print(f"    {'bucket':<10}{'mean excess':>14}   (low to high signal)")
    for i in range(n_buckets):
        avg = _mean(bucket_means[i])
        bar = '#' * max(0, int(abs(avg) * 4000))
        sign = '' if avg >= 0 else '-'
        print(f"    Q{i + 1:<9}{avg * 10000:>11.1f} bp   {sign}{bar[:36]}")

    wins = sum(1 for _, s, _ in spreads if s > 0)
    pooled = _mean([s for _, s, _ in spreads])
    print(f"\n    top-minus-bottom  {pooled * 10000:>7.1f} bp per period")
    print(f"    positive in       {wins}/{len(spreads)} periods")
    for period, spread, n in spreads:
        flag = '' if spread > 0 else '   <-- wrong sign'
        print(f"      {period}  n={n:<6} {spread * 10000:>8.1f} bp{flag}")

    monotone = all(_mean(bucket_means[i]) <= _mean(bucket_means[i + 1])
                   for i in range(n_buckets - 1))
    return {'spread': pooled, 'wins': wins, 'periods': len(spreads),
            'monotone': monotone}


def divergence_test(records):
    """Do rising stars beat fallen angels? The headline claim."""
    periods = _by_period(records, 'divergence')
    if not periods:
        print("\n  DIVERGENCE: too few scored observations")
        return None

    print(f"\n  DIVERGENCE — rising stars vs fallen angels")
    diffs = []
    for period, rows in sorted(periods.items()):
        rising = [r['excess_return'] for r in rows if r['divergence'] >= 1]
        falling = [r['excess_return'] for r in rows if r['divergence'] <= -1]
        if len(rising) < 10 or len(falling) < 10:
            continue
        diff = _mean(rising) - _mean(falling)
        diffs.append(diff)
        print(f"    {period}  rising n={len(rising):<5} {_mean(rising) * 10000:>7.1f} bp"
              f"   falling n={len(falling):<5} {_mean(falling) * 10000:>7.1f} bp"
              f"   diff {diff * 10000:>7.1f} bp")

    if not diffs:
        print("    no period had enough of both to compare")
        return None
    wins = sum(1 for d in diffs if d > 0)
    print(f"\n    rising minus falling: {_mean(diffs) * 10000:.1f} bp per period, "
          f"positive in {wins}/{len(diffs)}")
    return {'diff': _mean(diffs), 'wins': wins, 'periods': len(diffs)}


def bucket_test(records):
    """Does the implied bucket predict forward SPREAD change?

    With no realised-default data this is the closest available proxy for
    credit-model skill: worse-rated issuers should see spreads widen more.
    """
    rows = [r for r in records
            if r.get('implied_bucket') and r.get('spread_change') is not None]
    if len(rows) < 100:
        print("\n  BUCKET ORDERING: too few scored observations")
        return None

    print(f"\n  BUCKET ORDERING — forward spread change by implied bucket")
    print(f"    {'bucket':<8}{'n':>7}{'mean spread change':>22}")
    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row['implied_bucket']].append(row['spread_change'])
    order = [b for b in credit.CREDIT_BUCKETS if b in by_bucket]
    means = []
    for bucket in order:
        avg = _mean(by_bucket[bucket])
        means.append(avg)
        print(f"    {bucket:<8}{len(by_bucket[bucket]):>7}"
              f"{avg * 10000:>18.1f} bp")
    if len(means) >= 3:
        monotone = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
        print(f"\n    monotone (better credit widens less): {monotone}")
        return {'monotone': monotone, 'buckets': len(means)}
    return None


def summarise(records):
    print(f"\n{'=' * 78}")
    print(f"  MARKED-TO-MARKED BACKTEST  —  {len(records):,} bond-months")
    print(f"{'=' * 78}")

    periods = sorted({r['period'] for r in records})
    print(f"\n  Periods: {len(periods)}  ({periods[0]} to {periods[-1]})")
    excess = [r['excess_return'] for r in records if r['excess_return'] is not None]
    total = [r['total_return'] for r in records]
    print(f"  Mean total return   {_mean(total) * 10000:>8.1f} bp per month")
    print(f"  Mean excess return  {_mean(excess) * 10000:>8.1f} bp "
          f"(over duration-matched Treasury)")
    print(f"  Excess stdev        {statistics.pstdev(excess) * 10000:>8.1f} bp")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-funds', type=int, default=3)
    ap.add_argument('--min-held', type=float, default=10e6)
    ap.add_argument('--allow-lookahead', action='store_true',
                    help='CONTAMINATED diagnostic: score every observation '
                         'with the newest fundamentals regardless of date, to '
                         'tell a weak signal from a broken one')
    ap.add_argument('--limit', type=int, default=0,
                    help='cap the number of pairs (for a fast smoke run)')
    args = ap.parse_args()

    params = default_params()
    rows = load_panel(args.min_funds, args.min_held)
    pairs = build_pairs(rows)
    if args.limit:
        pairs = pairs[:args.limit]
        log.info('Limited to %d pairs', len(pairs))
    if not pairs:
        raise SystemExit('[fatal] no usable observation pairs — ingest more '
                         'quarters so the same CUSIP appears in consecutive '
                         'months')

    records = measure(pairs, PointInTime(args.allow_lookahead), params)
    if len(records) < MIN_PER_PERIOD:
        raise SystemExit(f'[fatal] only {len(records)} measured bond-months')

    if args.allow_lookahead:
        print("\n" + "!" * 78)
        print("  CONTAMINATED RUN — fundamentals are NOT point-in-time.")
        print("  Every observation is scored with a balance sheet published")
        print("  later than the observation date. Results are flattered by")
        print("  exactly that hindsight and are a machinery check ONLY.")
        print("!" * 78)

    summarise(records)
    # Spread level needs no fundamentals, so it is testable over the whole
    # panel. It is also the honest fallback: if carry alone explains the
    # forward return, the credit machinery has to beat it to justify itself.
    carry = decile_test(records, 'z_spread',
                        'SPREAD LEVEL — does carry alone predict excess return?')
    mispricing = decile_test(records, 'spread_mispricing',
                             'MISPRICING DECILES — cheap should beat rich')
    divergence = divergence_test(records)
    buckets = bucket_test(records)

    print(f"\n{'=' * 78}")
    print("  VERDICT")
    print(f"{'=' * 78}")
    verdicts = []
    if carry:
        ok = carry['wins'] > carry['periods'] / 2
        verdicts.append(('spread level predicts excess return', ok,
                         f"{carry['wins']}/{carry['periods']} periods"))
    if mispricing:
        ok = mispricing['wins'] > mispricing['periods'] / 2
        verdicts.append(('mispricing predicts excess return', ok,
                         f"{mispricing['wins']}/{mispricing['periods']} periods"))
    if divergence:
        ok = divergence['wins'] > divergence['periods'] / 2
        verdicts.append(('divergence separates rising from falling', ok,
                         f"{divergence['wins']}/{divergence['periods']} periods"))
    if buckets:
        verdicts.append(('implied bucket orders spread change',
                         buckets['monotone'], f"{buckets['buckets']} buckets"))
    for label, ok, detail in verdicts:
        print(f"    [{'PASS' if ok else 'FAIL'}]  {label:<44} {detail}")

    # The comparison that decides whether the credit machinery earns its keep.
    # A fair-spread model, a crosswalk and a scorecard are a great deal of
    # apparatus; if a bond's raw spread predicts forward excess return better
    # than the mispricing signal does, the apparatus is not paying for itself.
    if carry and mispricing:
        print(f"\n  DOES THE CREDIT MACHINERY BEAT RAW CARRY?")
        print(f"    spread level      {carry['spread'] * 10000:>7.1f} bp per period"
              f"   monotone={carry['monotone']}")
        print(f"    mispricing        {mispricing['spread'] * 10000:>7.1f} bp per period"
              f"   monotone={mispricing['monotone']}")
        if mispricing['spread'] < carry['spread']:
            ratio = (mispricing['spread'] / carry['spread']
                     if carry['spread'] else 0)
            print(f"\n    The fair-spread model captures {ratio:.0%} of what the raw")
            print(f"    spread already gives you. Everything the crosswalk, the")
            print(f"    scorecard and the term structure add sits inside that gap —")
            print(f"    a lot of apparatus for a fraction of a much simpler signal.")
            print(f"    Worth keeping only if it holds up point-in-time and adds")
            print(f"    something carry cannot: avoiding the credits that default.")
    if not verdicts:
        print("    Nothing could be tested — too little data.")
    scored = sum(1 for r in records if r.get('spread_mispricing') is not None)
    if scored < 0.2 * len(records):
        print(f"\n  NOT TESTED: the fundamentals-dependent signals (mispricing,")
        print(f"  divergence, bucket ordering) scored only {scored:,} of "
              f"{len(records):,} bond-months.")
        print(f"  The equity model's snapshots begin 2026-04-20, so there are no")
        print(f"  point-in-time fundamentals for earlier observation dates — and")
        print(f"  using CURRENT fundamentals to score a 2025 observation would be")
        print(f"  look-ahead bias, which is exactly what makes a backtest flatter.")
        print(f"  The N-PORT price history reaches back to 2019Q4; the credit")
        print(f"  history does not. Those signals become testable as snapshots")
        print(f"  accumulate forward, roughly one usable period per month.")

    print(f"\n  These are DISTRIBUTIONAL results over a short panel, not proof.")
    print(f"  A signal that fails here is not worth trading; one that passes")
    print(f"  is worth continuing to measure.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
