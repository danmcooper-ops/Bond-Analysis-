#!/usr/bin/env python3
"""Print the Treasury par curve, its bootstrapped zeros, and credit spreads.

The M2 deliverable and the daily smoke test: if this runs clean, every data
source the pipeline depends on is reachable and self-consistent.

    python scripts/show_curve.py
    python scripts/show_curve.py --date 2026-08-05
    python scripts/show_curve.py --no-fred        # curve only, offline-ish

It cross-checks the Treasury XML par curve against FRED's independent DGS*
constant-maturity series. They are two different publication paths for the
same underlying quotes, so agreement is expected and a disagreement above a
couple of basis points means one of the feeds changed shape or units.
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fred_client import FREDClient, term_factor_at
from data.treasury_curve_client import TreasuryCurveClient
from models.curve import TENOR_YEARS, YieldCurve
from scripts.config import CURVE_CROSSCHECK_TOL_BP


def _parse_date(text):
    return datetime.strptime(text, '%Y-%m-%d').date()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date', type=_parse_date, default=None,
                    help='as-of date (default: today)')
    ap.add_argument('--no-fred', action='store_true',
                    help='skip the credit-spread and cross-check sections')
    ap.add_argument('--force', action='store_true', help='bypass the cache')
    args = ap.parse_args()

    as_of = args.date or date.today()

    # -- par curve ----------------------------------------------------------
    tc = TreasuryCurveClient()
    if args.force:
        tc.fetch_year(as_of.year, force=True)
    curve_date, par = tc.fetch_par_curve(as_of)
    if not par:
        raise SystemExit(f"[error] no Treasury curve available near {as_of}")

    stale = (as_of - curve_date).days
    banner = f"  TREASURY PAR CURVE  {curve_date:%Y-%m-%d}"
    if stale:
        banner += f"   ({stale}d stale vs requested {as_of:%Y-%m-%d})"
    print(f"\n{'=' * 66}\n{banner}\n{'=' * 66}")

    yc = YieldCurve.from_par_dict(curve_date, par)
    print(f"\n{'Tenor':<8}{'Par':>10}{'Zero':>10}{'Fwd(1y)':>11}{'DF':>12}")
    print('-' * 51)
    for label in sorted(par, key=lambda k: TENOR_YEARS[k]):
        t = TENOR_YEARS[label]
        fwd = yc.forward(t, t + 1.0)
        print(f"{label:<8}{par[label]:>9.3%}{yc.zero(t):>10.3%}"
              f"{(f'{fwd:.3%}' if fwd else 'n/a'):>11}"
              f"{yc.discount(t):>12.6f}")

    err = yc.repricing_error()
    verdict = 'OK' if err < 1e-6 else 'FAIL'
    print(f"\n  Bootstrap check: max par repricing error "
          f"{err:.2e} per 100 face  [{verdict}]")
    if err >= 1e-6:
        print("  Every quoted par bond should reprice to exactly 100 off the "
              "bootstrapped zeros.\n  A non-trivial error means the bootstrap "
              "or the interpolator is broken.")

    r = tc.regime(as_of)
    if r:
        print(f"\n  Shape      {r['shape']}  ({r['direction']})")
        if r['slope_10y_3m'] is not None:
            print(f"  10y-3m     {r['slope_10y_3m'] * 10000:>7.1f} bp")
        if r['slope_10y_2y'] is not None:
            print(f"  10y-2y     {r['slope_10y_2y'] * 10000:>7.1f} bp")
        if r['level_pctile_1y'] is not None:
            print(f"  10y level  {r['level_10y']:.3%}  "
                  f"({r['level_pctile_1y']:.0f}th pctile of the last "
                  f"{r['history_days']} sessions)")
        if r['momentum_3m'] is not None:
            print(f"  3m change  {r['momentum_3m'] * 10000:>7.1f} bp")

    if args.no_fred:
        print()
        return 0

    # -- cross-check --------------------------------------------------------
    fred = FREDClient()

    # Compare the two feeds on a date they BOTH published. FRED lags Treasury
    # by about a day, so comparing "latest available" against "latest
    # available" reads an overnight rate move as a feed discrepancy.
    check_date = fred.latest_common_date(as_of=curve_date)
    tsy_history = tc.fetch_curve_history(
        curve_date - timedelta(days=14), curve_date)
    if check_date and check_date in tsy_history:
        cmt = fred.fetch_cmt_curve(check_date, exact_date=True)
        tsy_same_day = tsy_history[check_date]
        print(f"\n{'-' * 66}\n  CROSS-CHECK vs FRED constant-maturity "
              f"(independent source)\n{'-' * 66}")
        lag = (curve_date - check_date).days
        note = (f"   [FRED lags the par feed by {lag}d; comparing same-date]"
                if lag else '')
        print(f"  Both feeds on {check_date:%Y-%m-%d}{note}\n")
        worst, worst_tenor = 0.0, None
        for label in sorted(set(tsy_same_day) & set(cmt),
                            key=lambda k: TENOR_YEARS[k]):
            diff_bp = (tsy_same_day[label] - cmt[label]) * 10000
            if abs(diff_bp) > abs(worst):
                worst, worst_tenor = diff_bp, label
            flag = '  <-- ' if abs(diff_bp) > CURVE_CROSSCHECK_TOL_BP else ''
            print(f"  {label:<6}  treasury {tsy_same_day[label]:>7.3%}   "
                  f"fred {cmt[label]:>7.3%}   diff {diff_bp:>6.1f} bp{flag}")
        if abs(worst) > CURVE_CROSSCHECK_TOL_BP:
            print(f"\n  WARNING: {worst_tenor} differs by {worst:.1f} bp, above "
                  f"the {CURVE_CROSSCHECK_TOL_BP:.0f} bp tolerance.\n"
                  f"  These are two publication paths for the same quotes, "
                  f"compared on the same\n  date — a real gap means one feed "
                  f"changed shape or units.")
        else:
            where = f' at {worst_tenor}' if worst_tenor else ''
            print(f"\n  Max divergence {worst:.1f} bp{where} "
                  f"(tolerance {CURVE_CROSSCHECK_TOL_BP:.0f} bp)  [OK]")
    else:
        print("\n  [warn] No date published by both feeds within the lookback; "
              "cross-check skipped.")

    # -- credit -------------------------------------------------------------
    oas = fred.fetch_bucket_oas(curve_date)
    if oas:
        print(f"\n{'-' * 66}\n  ICE BofA OPTION-ADJUSTED SPREAD BY RATING "
              f"BUCKET\n{'-' * 66}")
        order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        prev = None
        for bucket in order:
            if bucket not in oas:
                continue
            v = oas[bucket]
            ten_year = yc.par(10.0)
            arrow = ''
            if prev is not None and v < prev:
                arrow = '  <-- NOT monotone in credit quality'
            prev = v
            print(f"  {bucket:<5} {v * 10000:>7.1f} bp    "
                  f"all-in ~{(ten_year + v):.3%}{arrow}")

    tf = fred.fetch_term_factors(curve_date)
    if tf['points']:
        print(f"\n{'-' * 66}\n  SPREAD TERM STRUCTURE (IG index = 1.00x, "
              f"{tf['ig_all'] * 10000:.0f} bp)\n{'-' * 66}")
        for label, ratio in sorted(tf['factors'].items(),
                                   key=lambda kv: kv[1]):
            print(f"  {label:<8} {ratio:>6.2f}x")
        print("\n  Interpolated factor by maturity:")
        for t in (2, 5, 10, 20, 30):
            print(f"    {t:>2}y  {term_factor_at(tf['points'], t):>5.2f}x", end='')
        print()

        # Worked example, so the fair-spread arithmetic is visible rather than
        # buried in the scoring layer.
        if 'BBB' in oas:
            for t in (2, 10, 30):
                factor = term_factor_at(tf['points'], t)
                fair = oas['BBB'] * factor
                print(f"\n  Example: {t}y BBB fair spread = "
                      f"{oas['BBB'] * 10000:.0f} bp x {factor:.2f} = "
                      f"{fair * 10000:.0f} bp"
                      f"   -> all-in {(yc.par(t) + fair):.3%}")

    cov = fred.coverage()
    if cov:
        firsts = sorted(v['first'] for v in cov.values())
        print(f"\n{'-' * 66}")
        print(f"  FRED history: {len(cov)} series, source={fred.history_source}, "
              f"earliest {firsts[0]}, shortest starts {firsts[-1]}")
        if fred.history_source == 'keyless':
            print("  Without FRED_API_KEY the ICE BofA series are capped to a "
                  "rolling ~3-year\n  window. Daily scoring is unaffected; the "
                  "walk-forward calibration at M8\n  will train on a much "
                  "shorter spread history than intended.")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
