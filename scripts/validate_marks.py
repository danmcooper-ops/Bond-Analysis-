#!/usr/bin/env python3
"""Are N-PORT fund marks CLEAN prices or DIRTY ones?

    python scripts/validate_marks.py
    python scripts/validate_marks.py --quarter 2026q2 --min-funds 3

THE MOST IMPORTANT CHECK IN THE INGESTION, AND WHY
---------------------------------------------------
CURRENCY_VALUE / BALANCE * 100 is a fund's carrying fair value per 100 face.
Whether that includes accrued interest is not documented anywhere, and it
changes every downstream number: an incorrectly clean-assumed dirty price
overstates every yield and spread by the accrued amount, which averages about
a quarter of a coupon — on a 5% bond, roughly 60bp of price. That is larger
than most of the mispricing signal the model is looking for, and it would be
invisible, because a uniformly-biased spread still ranks bonds in roughly the
right order while making every one of them look cheap.

Treasuries make it testable. Their price is derivable EXACTLY from the par
curve plus a known coupon and maturity, with no credit spread in the way. So
for every Treasury in the N-PORT data we can compute what its clean and dirty
prices must have been on the mark date and see which the fund reported.

THE DECISIVE TEST is not "which is closer on average" — a small average gap
could come from anywhere. It is the REGRESSION OF THE RESIDUAL ON ACCRUED
INTEREST:

    (nport_implied - theoretical_clean)  ~  a + b * accrued_interest

    b close to 0  =>  marks are CLEAN   (residual unrelated to accrual)
    b close to 1  =>  marks are DIRTY   (residual IS the accrual)

Accrued interest cycles from zero to a full coupon and back within every
period, so if the marks carried it the correlation would be unmistakable.
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.logging_setup import get_logger
from data.nport_client import NPORTClient
from data.treasury_curve_client import TreasuryCurveClient
from data.treasury_direct_client import TreasuryDirectClient
from models.curve import YieldCurve
from models.daycount import ACT_ACT
from models.pricing import bond_flows_and_stub
from models.schedule import accrued_interest, previous_next_coupon
from models.spreads import price_from_zero_curve

log = get_logger('validate_marks')


def _ols(xs, ys):
    """Least-squares slope, intercept, R^2. None if degenerate."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return slope, intercept, r2


def _median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def build_reference(as_of):
    """Authoritative coupon/maturity per Treasury CUSIP, nominal issues only."""
    td = TreasuryDirectClient()
    records = td.fetch_outstanding(as_of=as_of, max_years=31)
    rows = td.to_bond_rows(records, as_of=as_of)
    return {r['cusip']: r for r in rows
            if r['coupon_type'] == 'Fixed' and r['frequency'] == 2
            and not r['is_inflation_linked']}


def collect(quarter, min_funds, as_of):
    """Pair every Treasury mark with its curve-implied clean and dirty price."""
    import pandas as pd

    client = NPORTClient()
    marks_path = os.path.join(client.cache_dir, f'{quarter}_marks.parquet')
    if not os.path.exists(marks_path):
        raise SystemExit(f'[fatal] {marks_path} not found — run '
                         f'scripts/ingest_nport.py --quarter {quarter} first')

    marks = pd.read_parquet(marks_path)
    treasuries = marks[(marks['issuer_type'] == 'UST')
                       & (marks['n_funds'] >= min_funds)]
    log.info('%s: %d Treasury marks with >=%d funds', quarter,
             len(treasuries), min_funds)

    reference = build_reference(as_of)
    log.info('TreasuryDirect reference: %d nominal coupon issues', len(reference))

    curves, tc = {}, TreasuryCurveClient()

    def curve_for(when):
        if when not in curves:
            curve_date, par = tc.fetch_par_curve(when)
            curves[when] = (YieldCurve.from_par_dict(curve_date, par)
                            if par else None)
        return curves[when]

    paired, skipped = [], defaultdict(int)
    for row in treasuries.to_dict('records'):
        ref = reference.get(row['cusip'])
        if ref is None:
            skipped['not a nominal Treasury in TreasuryDirect'] += 1
            continue
        settle = row['report_date']
        if hasattr(settle, 'date'):
            settle = settle.date()
        if ref['maturity_date'] <= settle:
            skipped['matured by the mark date'] += 1
            continue

        curve = curve_for(settle)
        if curve is None:
            skipped['no curve for the mark date'] += 1
            continue

        flows, _ = bond_flows_and_stub(
            ref['coupon_rate'], ref['maturity_date'], settle, frequency=2,
            convention=ACT_ACT, dated_date=ref.get('dated_date'))
        if not flows:
            skipped['no remaining cashflows'] += 1
            continue

        theoretical_dirty = price_from_zero_curve(flows, settle, curve)
        if theoretical_dirty is None:
            skipped['curve pricing failed'] += 1
            continue
        accrued = accrued_interest(settle, ref['coupon_rate'],
                                   ref['maturity_date'], frequency=2,
                                   convention=ACT_ACT,
                                   dated_date=ref.get('dated_date'))
        theoretical_clean = theoretical_dirty - accrued

        prev, nxt = previous_next_coupon(settle, ref['maturity_date'], 2,
                                         dated_date=ref.get('dated_date'))
        period_days = max((nxt - prev).days, 1)

        paired.append({
            'cusip': row['cusip'],
            'report_date': settle,
            'coupon': ref['coupon_rate'],
            'maturity': ref['maturity_date'],
            'observed': row['clean_price_marked'],
            'theoretical_clean': theoretical_clean,
            'theoretical_dirty': theoretical_dirty,
            'accrued': accrued,
            'accrual_fraction': (settle - prev).days / period_days,
            'n_funds': int(row['n_funds']),
        })

    if skipped:
        log.info('skipped: %s', ', '.join(f'{k}={v}' for k, v in skipped.items()))
    return paired


def report(paired):
    if len(paired) < 10:
        raise SystemExit(f'[fatal] only {len(paired)} usable pairs — '
                         f'not enough to conclude anything')

    err_clean = [p['observed'] - p['theoretical_clean'] for p in paired]
    err_dirty = [p['observed'] - p['theoretical_dirty'] for p in paired]
    accrued = [p['accrued'] for p in paired]

    print(f"\n{'=' * 76}")
    print(f"  ARE N-PORT MARKS CLEAN OR DIRTY?   {len(paired):,} Treasury "
          f"observations")
    print(f"{'=' * 76}")

    print(f"\n  Accrued interest in the sample: median "
          f"{_median(accrued):.4f}, max {max(accrued):.4f} per 100 face")
    print(f"  Accrual fraction spans "
          f"{min(p['accrual_fraction'] for p in paired):.2f}"
          f"-{max(p['accrual_fraction'] for p in paired):.2f} of a period, so "
          f"a dirty-price\n  convention would be plainly visible.")

    print(f"\n  RESIDUAL vs EACH HYPOTHESIS (observed minus theoretical)")
    print(f"    {'hypothesis':<12}{'median':>10}{'mean abs':>11}"
          f"{'within 0.10':>13}{'within 0.50':>13}")
    for label, errors in (('CLEAN', err_clean), ('DIRTY', err_dirty)):
        within10 = 100.0 * sum(1 for e in errors if abs(e) <= 0.10) / len(errors)
        within50 = 100.0 * sum(1 for e in errors if abs(e) <= 0.50) / len(errors)
        mean_abs = sum(abs(e) for e in errors) / len(errors)
        print(f"    {label:<12}{_median(errors):>10.4f}{mean_abs:>11.4f}"
              f"{within10:>12.1f}%{within50:>12.1f}%")

    print(f"\n  THE DECISIVE TEST — regress (observed - theoretical_clean) "
          f"on accrued")
    fit = _ols(accrued, err_clean)
    if fit is None:
        print("    degenerate; cannot conclude")
        return None
    slope, intercept, r2 = fit
    print(f"    slope     {slope:>8.4f}    (0 => clean, 1 => dirty)")
    print(f"    intercept {intercept:>8.4f}")
    print(f"    R^2       {r2:>8.4f}" if r2 is not None else "")

    if abs(slope) < 0.25:
        verdict, basis = 'CLEAN', 'nport_implied_clean'
    elif abs(slope - 1.0) < 0.25:
        verdict, basis = 'DIRTY', 'nport_implied_dirty'
    else:
        verdict, basis = 'INCONCLUSIVE', None

    print(f"\n  VERDICT: marks are {verdict}")
    if verdict == 'CLEAN':
        print(f"    The residual carries no accrual signal, so the reported "
              f"value excludes\n    accrued interest. price_basis="
              f"'{basis}' is correct as implemented.")
    elif verdict == 'DIRTY':
        print(f"    The residual tracks accrued interest almost one-for-one. "
              f"Every price\n    must have accrued SUBTRACTED before use — "
              f"otherwise every yield and\n    spread in the model is "
              f"overstated. Set price_basis='{basis}'\n    and subtract in "
              f"nport_consensus.")
    else:
        print(f"    Slope {slope:.3f} sits between the two hypotheses. Do not "
              f"proceed to M6\n    on an assumption; investigate before the "
              f"credit model is built on it.")

    # Same test per month: a convention that varies by period would average
    # into an ambiguous slope and hide itself.
    by_month = defaultdict(list)
    for p, e in zip(paired, err_clean):
        by_month[p['report_date']].append((p['accrued'], e))
    print(f"\n  BY MARK MONTH (a convention varying by period would hide in "
          f"the pooled fit)")
    print(f"    {'month':<13}{'n':>6}{'slope':>9}{'med resid':>11}")
    for month in sorted(by_month):
        pairs = by_month[month]
        if len(pairs) < 5:
            continue
        sub = _ols([a for a, _ in pairs], [e for _, e in pairs])
        resid = _median([e for _, e in pairs])
        slope_txt = f'{sub[0]:>9.4f}' if sub else '        —'
        print(f"    {str(month):<13}{len(pairs):>6}{slope_txt}{resid:>11.4f}")

    return verdict


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quarter', default='2026q2')
    ap.add_argument('--min-funds', type=int, default=3,
                    help='require this many funds behind each mark')
    ap.add_argument('--as-of', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                    default=date(2026, 8, 6))
    args = ap.parse_args()

    paired = collect(args.quarter, args.min_funds, args.as_of)
    verdict = report(paired)
    print()
    return 0 if verdict in ('CLEAN', 'DIRTY') else 2


if __name__ == '__main__':
    sys.exit(main())
