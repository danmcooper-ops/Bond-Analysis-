#!/usr/bin/env python3
"""Price a single bond and print its full analytics.

Useful on its own, and it exercises the whole models/ stack with no data
dependencies — no N-PORT, no network, no API key. Give it a coupon, a maturity
and either a price or a yield.

    python scripts/price_bond.py --coupon 4.5 --maturity 2034-11-15 --price 97.25
    python scripts/price_bond.py --coupon 5 --maturity 2035-06-15 --yield 5.4
    python scripts/price_bond.py --coupon 4.25 --maturity 2034-11-15 --price 98 \\
        --treasury --curve "3M=4.30,2Y=4.50,5Y=4.80,10Y=5.10,30Y=5.50"

Supply a curve to get Z-spread, G-spread and key-rate durations as well.
"""

import argparse
import json
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.conventions import classify_by_cusip, conventions_for
from models.curve import TENOR_YEARS, YieldCurve
from models.pricing import (bond_flows_and_stub, current_yield, price_bond,
                            yield_to_maturity, yield_to_worst)
from models.risk import (convexity, dv01, effective_duration,
                         key_rate_durations, macaulay_duration,
                         modified_duration, spread_duration)
from models.schedule import accrued_interest, years_to_maturity
from models.spreads import g_spread, z_spread
from models.total_return import carry, roll_down


def _parse_date(text):
    for fmt in ('%Y-%m-%d', '%m/%d/%Y'):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Unparseable date: {text!r}")


def _parse_rate(value):
    """Accept 4.5 or 0.045 — anything above 1 is read as a percentage."""
    v = float(value)
    return v / 100.0 if abs(v) > 1.0 else v


def parse_curve(spec):
    """Parse '2Y=4.5,10Y=5.1' or a path to a JSON file of {tenor: rate}."""
    if not spec:
        return None
    if os.path.exists(spec):
        with open(spec, encoding='utf-8') as fh:
            raw = json.load(fh)
    else:
        raw = {}
        for part in spec.split(','):
            if '=' not in part:
                raise SystemExit(f"[error] bad curve segment: {part!r}")
            k, v = part.split('=', 1)
            raw[k.strip().upper()] = v
    par = {}
    for k, v in raw.items():
        if k not in TENOR_YEARS:
            raise SystemExit(f"[error] unknown tenor {k!r}. "
                             f"Known: {', '.join(TENOR_YEARS)}")
        par[k] = _parse_rate(v)
    return par


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coupon', required=True, type=_parse_rate,
                    help='annual coupon rate (4.5 or 0.045)')
    ap.add_argument('--maturity', required=True, type=_parse_date,
                    help='maturity date (YYYY-MM-DD)')
    price_or_yield = ap.add_mutually_exclusive_group(required=True)
    price_or_yield.add_argument('--price', type=float,
                                help='clean price per 100 face')
    price_or_yield.add_argument('--yield', dest='ytm', type=_parse_rate,
                                help='yield to maturity, to solve for a price')
    ap.add_argument('--settle', type=_parse_date, default=date.today(),
                    help='settlement date (default: today)')
    ap.add_argument('--frequency', type=int, default=None,
                    help='coupons per year (default: from the asset class)')
    ap.add_argument('--treasury', action='store_true',
                    help='use Treasury conventions (ACT/ACT)')
    ap.add_argument('--cusip', default=None,
                    help='infer the asset class from the CUSIP')
    ap.add_argument('--face', type=float, default=100.0)
    ap.add_argument('--curve', default=None,
                    help='par curve as "2Y=4.5,10Y=5.1" or a JSON file path; '
                         'enables the spread and key-rate section')
    ap.add_argument('--call', action='append', default=[], metavar='DATE:PRICE',
                    help='a call option, repeatable (e.g. 2030-06-15:102)')
    args = ap.parse_args()

    settle, maturity = args.settle, args.maturity
    if maturity <= settle:
        raise SystemExit(f"[error] maturity {maturity} is not after settle {settle}")

    asset_class = ('TREASURY' if args.treasury
                   else (classify_by_cusip(args.cusip) if args.cusip else None)
                   or 'CORP_IG')
    conv = conventions_for(asset_class, coupon_rate=args.coupon,
                           frequency=args.frequency)
    freq, convention = conv['frequency'], conv['convention']

    call_schedule = []
    for item in args.call:
        try:
            d, p = item.split(':', 1)
            call_schedule.append((_parse_date(d), float(p)))
        except ValueError:
            raise SystemExit(f"[error] bad --call value: {item!r} "
                             f"(expected DATE:PRICE)")

    common = dict(frequency=freq, convention=convention, face=args.face)

    # --- price / yield -----------------------------------------------------
    if args.price is not None:
        clean = args.price
        ytm = yield_to_maturity(clean, args.coupon, maturity, settle, **common)
        if ytm is None:
            raise SystemExit("[error] yield solver did not converge. In the "
                             "pipeline this sets ytm_solver_failed and the "
                             "row is capped at HOLD.")
        accrued = accrued_interest(settle, args.coupon, maturity,
                                   frequency=freq, face=args.face,
                                   convention=convention)
        dirty = clean + accrued
    else:
        ytm = args.ytm
        clean, dirty, accrued = price_bond(args.coupon, maturity, settle, ytm,
                                           **common)
        if clean is None:
            raise SystemExit(f"[error] yield {ytm:.4%} is outside the pricing "
                             f"domain for frequency {freq}")

    flows, w = bond_flows_and_stub(args.coupon, maturity, settle, **common)
    ttm = years_to_maturity(settle, maturity)

    mac = macaulay_duration(flows, ytm, frequency=freq, w=w)
    mod = modified_duration(mac, ytm, frequency=freq)
    cvx = convexity(flows, ytm, frequency=freq, w=w)
    eff = effective_duration(flows, ytm, frequency=freq, w=w)
    d01 = dv01(dirty, mod, face=args.face)

    print(f"\n{'=' * 62}")
    print(f"  {args.coupon:.3%} of {maturity:%Y-%m-%d}"
          f"   [{asset_class}, {convention}, {freq}x/yr]")
    print(f"  settled {settle:%Y-%m-%d}   {ttm:.2f} years to maturity")
    print(f"{'=' * 62}")

    print("\nPRICE")
    print(f"  Clean                 {clean:>12.4f}")
    print(f"  Accrued               {accrued:>12.4f}")
    print(f"  Dirty (invoice)       {dirty:>12.4f}")

    print("\nYIELD")
    print(f"  Yield to maturity     {ytm:>12.4%}")
    cy = current_yield(clean, args.coupon, face=args.face)
    if cy is not None:
        print(f"  Current yield         {cy:>12.4%}")
    ytw = yield_to_worst(clean, args.coupon, maturity, settle,
                         call_schedule=call_schedule or None, **common)
    if ytw['ytw'] is not None:
        note = (f"to {ytw['to_type']} {ytw['to_date']:%Y-%m-%d}"
                if ytw['call_data_available']
                else "to maturity (NO CALL DATA — may overstate)")
        print(f"  Yield to worst        {ytw['ytw']:>12.4%}   {note}")

    print("\nRISK")
    print(f"  Macaulay duration     {mac:>12.4f} yrs")
    print(f"  Modified duration     {mod:>12.4f}")
    print(f"  Effective duration    {eff:>12.4f}   (numerical cross-check)")
    print(f"  Convexity             {cvx:>12.4f}")
    print(f"  DV01                  {d01:>12.4f}   per {args.face:g} face")

    # --- curve-dependent ---------------------------------------------------
    par = parse_curve(args.curve)
    if par:
        curve = YieldCurve.from_par_dict(settle, par)
        z = z_spread(dirty, flows, settle, curve)
        g = g_spread(ytm, curve, ttm)
        print("\nSPREAD")
        print(f"  Benchmark ({len(par)} tenors) {curve.par(ttm):>10.4%}"
              f"   interpolated at {ttm:.2f}y")
        if g is not None:
            print(f"  G-spread              {g * 10000:>12.1f} bp")
        if z is not None:
            print(f"  Z-spread              {z * 10000:>12.1f} bp")
            sd = spread_duration(flows, settle, z, curve)
            if sd is not None:
                print(f"  Spread duration       {sd:>12.4f}")
        else:
            print("  Z-spread                       n/a  (solver did not converge)")

        krd = key_rate_durations(flows, settle, curve, spread=z or 0.0)
        if krd:
            print("\nKEY RATE DURATIONS")
            for k, v in krd.items():
                print(f"  {k:<21} {v:>12.4f}")
            print(f"  {'sum':<21} {sum(krd.values()):>12.4f}"
                  f"   (vs modified {mod:.4f})")

        print("\nCARRY & ROLL (12m)")
        c = carry(args.coupon, dirty, 365, face=args.face)
        r = roll_down(curve, ttm, 1.0, mod)
        if c is not None:
            print(f"  Carry                 {c * 10000:>12.1f} bp")
        if r is not None:
            print(f"  Roll-down             {r * 10000:>12.1f} bp")
            print(f"  Carry + roll          {(c + r) * 10000:>12.1f} bp")
        else:
            print("  Roll-down                      n/a  (matures inside horizon)")

    if not ytw['call_data_available'] and clean > 100.5 \
            and asset_class.startswith('CORP'):
        print("\n  NOTE: priced above par with no call schedule. If this bond "
              "is\n  callable, the Z-spread overstates the compensation on "
              "offer.\n  The pipeline caps this case at HOLD rather than "
              "guessing.")
    print()


if __name__ == '__main__':
    main()
