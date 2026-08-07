#!/usr/bin/env python3
"""The analysis pipeline: reference data in, rated bonds out.

    python scripts/analyze_bonds.py --universe treasury
    python scripts/analyze_bonds.py --universe treasury --as-of 2026-08-06
    python scripts/analyze_bonds.py --universe treasury --top 30 --json

Phases:
    0  curve, bootstrap, regime, bucket OAS, term factors
    1  universe assembly
    2  issuer fundamentals and implied credit bucket   (corporates, M6)
    3  per-bond analytics
    4  scoring and rating
    5  artifacts

HOW TREASURIES ARE PRICED, AND WHY IT IS NOT A SHORTCUT
--------------------------------------------------------
There is no free per-CUSIP Treasury price feed, so each Treasury is priced off
the bootstrapped curve rather than from a market mark. That is honest for this
asset class rather than a workaround. Nobody screens for mispriced Treasuries
— they are the most efficiently priced instruments in existence, and a model
claiming to find 20bp of alpha in the on-the-run 10-year would be finding a
bug in itself. The real question for a Treasury buyer is WHERE ON THE CURVE to
sit, and that is precisely what the Rates gates answer: is duration cheap
here, is convexity good per unit of duration, is there roll-down, does the
yield beat cash.

Every such row carries price_source='curve_implied' and the report says so.
When N-PORT marks arrive at M4, Treasuries held by funds gain a real observed
price and the Valuation gates start to bite.
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fred_client import FREDClient, term_factor_at
from data.logging_setup import get_logger
from data.mspd_client import MSPDClient
from data.treasury_curve_client import TreasuryCurveClient
from data.treasury_direct_client import TreasuryDirectClient
from models.bond_types import from_row
from models import credit, discount
from models.curve import YieldCurve
from models.pricing import (bond_flows_and_stub, current_yield,
                            price_from_yield, yield_from_price,
                            yield_to_worst)
from models.risk import (convexity, dv01, macaulay_duration,
                         modified_duration)
from models.schedule import accrued_interest, years_to_maturity
from models.spreads import (is_likely_callable, price_from_zero_curve,
                            yield_over_treasury, z_spread)
from models.total_return import carry, roll_down
from scripts.gates import SPEC
from scripts.param_set import default_params, validate_params
from scripts.scoring_kernel import score_and_rate

log = get_logger('analyze')

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')


# ---------------------------------------------------------------------------
# Phase 0: market context
# ---------------------------------------------------------------------------

def build_context(as_of, use_fred=True):
    """Curve, regime, bucket OAS and term factors — computed once per run."""
    tc = TreasuryCurveClient()
    curve_date, par = tc.fetch_par_curve(as_of)
    if not par:
        raise SystemExit(f'[fatal] no Treasury curve available near {as_of}')

    curve = YieldCurve.from_par_dict(curve_date, par)
    err = curve.repricing_error()
    if err > 1e-6:
        log.error('Bootstrap repricing error %.2e — curve is unreliable', err)
    else:
        log.info('Curve %s bootstrapped, repricing error %.2e', curve_date, err)

    # Curves at past mark dates, for the mark-to-curve overlay. Cached per run
    # because a few thousand bonds share a handful of month-end mark dates.
    historical = {}

    def curve_at(when):
        if when not in historical:
            past_date, past_par = tc.fetch_par_curve(when)
            historical[when] = (YieldCurve.from_par_dict(past_date, past_par)
                                if past_par else None)
        return historical[when]

    ctx = {
        'curve': curve,
        'curve_date': curve_date,
        'curve_at': curve_at,
        'par': par,
        'regime': tc.regime(as_of) or {},
        'front_end_yield': par.get('3M') or par.get('6M') or par.get('1M'),
        'bucket_oas': {},
        'term_points': [],
        'fred_source': None,
    }

    if use_fred:
        fred = FREDClient()
        ctx['bucket_oas'] = fred.fetch_bucket_oas(curve_date)
        ctx['term_points'] = fred.fetch_term_factors(curve_date)['points']
        ctx['fred_source'] = fred.history_source

    # A term structure fitted from our own observed spreads beats FRED's
    # maturity slices: those stop at 15y+ and are sub-indices with different
    # constituents, so they overstate long-dated fair spreads by up to 17%.
    from scripts.calibrate_credit import load_anchors
    from scripts.fit_term_structure import load_fitted, load_tiered
    ctx['term_by_bucket'] = load_tiered()
    ctx['bucket_anchors'] = load_anchors()
    if ctx['bucket_anchors']:
        log.info('Fair-spread anchors: fitted from the model\'s own buckets '
                 '(%d of 7)', len(ctx['bucket_anchors']))
    else:
        log.warning('No fitted bucket anchors — falling back to the published '
                    'index OAS, whose constituents are not this model\'s '
                    'bucket members. Run scripts/calibrate_credit.py --apply')
    fitted = load_fitted()
    if fitted:
        ctx['term_points'] = fitted
        ctx['term_source'] = 'fitted_nport_panel'
        log.info('Term structure: fitted from the N-PORT panel (%d points)',
                 len(fitted))
    else:
        ctx['term_source'] = 'fred_slices'
        log.warning('No fitted term structure — falling back to FRED slices, '
                    'which overstate the long end. Run '
                    'scripts/fit_term_structure.py --apply')

    return ctx


# ---------------------------------------------------------------------------
# Phase 1: universe
# ---------------------------------------------------------------------------

def attach_nport_marks(rows, as_of, quarter=None):
    """Stamp consensus marks onto rows by CUSIP. Returns the number matched.

    This is where the gate masking proves itself data-driven rather than
    hardcoded: a Treasury with no marks has its fund-liquidity gates
    structurally inapplicable, and the same Treasury with marks attached picks
    them up. Nothing about the asset class changed — only what is measurable
    about it.
    """
    from data.nport_client import NPORTClient
    from data.nport_consensus import latest_marks

    client = NPORTClient()
    if quarter is None:
        available = sorted(f.split('_')[0] for f in
                           os.listdir(client.cache_dir)
                           if f.endswith('_marks.parquet')) \
            if os.path.isdir(client.cache_dir) else []
        if not available:
            log.info('No N-PORT marks cached; prices will be curve-implied')
            return 0
        quarter = available[-1]

    path = os.path.join(client.cache_dir, f'{quarter}_marks.parquet')
    if not os.path.exists(path):
        log.warning('N-PORT marks not found: %s', path)
        return 0

    import pandas as pd
    marks = latest_marks(pd.read_parquet(path).to_dict('records'))
    by_cusip = {m['cusip']: m for m in marks}
    log.info('N-PORT %s: %d CUSIPs with marks', quarter, len(by_cusip))

    matched = 0
    for row in rows:
        mark = by_cusip.get((row.get('cusip') or '').strip().upper())
        if mark is None:
            continue
        report_date = mark['report_date']
        if hasattr(report_date, 'date'):
            report_date = report_date.date()
        # A mark from the future relative to the as-of date would be
        # look-ahead bias walked straight into the backtest.
        if report_date > as_of:
            continue
        matched += 1
        row.update({
            'clean_price_marked': mark['clean_price_marked'],
            'price_basis': mark['price_basis'],
            'mark_date': report_date,
            'mark_age_days': (as_of - report_date).days,
            'n_funds': int(mark['n_funds']),
            'total_held_usd': mark['total_held_usd'],
            'price_dispersion': mark['price_dispersion'],
            'fair_value_level': mark['fair_value_level'],
            '_identity_conflict': bool(mark['_identity_conflict']),
        })
        # Trouble flags come from the funds, which see them before we do.
        for flag in ('is_default', 'in_arrears', 'is_paid_kind'):
            if mark.get(flag):
                row[flag] = True

    log.info('N-PORT marks attached to %d of %d rows', matched, len(rows))
    return matched


def load_corporate_universe(quarter):
    """Read the universe built by scripts/build_universe.py."""
    path = os.path.join(OUTPUT_DIR, f'universe_{quarter}.parquet')
    if not os.path.exists(path):
        raise SystemExit(f'[fatal] {path} not found — run '
                         f'scripts/build_universe.py --quarter {quarter}')
    import pandas as pd
    frame = pd.read_parquet(path)
    # Parquet returns missing numerics as NaN, and NaN is not None: it passes
    # every `is None` guard downstream. Coerced at the boundary so the rest of
    # the pipeline sees one representation of "absent".
    rows = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
    for row in rows:
        for field in ('maturity_date', 'report_date', 'mark_date'):
            value = row.get(field)
            if value is not None and hasattr(value, 'date'):
                row[field] = value.date()
    log.info('Corporate universe %s: %d bonds', quarter, len(rows))
    return rows


def apply_credit_model(rows, params):
    """Phase 2: issuer credit score -> implied bucket -> asset class.

    Runs BEFORE the analytics because the bucket decides whether a bond is
    investment grade or high yield, and that in turn drives its peer pool.
    The MARKET-implied bucket has to wait until after the analytics, since it
    is read off the Z-spread the analytics produce.
    """
    scored = 0
    skipped_government = 0
    for row in rows:
        # Government and agency paper has no issuer balance sheet, and its
        # asset class is already known from the CUSIP. Running it through the
        # scorecard would relabel every Treasury CORP_IG (the default for an
        # unknown bucket) and silently destroy the masking that gives them a
        # rating scale at all — 402 Treasuries vanished from a combined run
        # this way before the guard existed.
        if (row.get('asset_class') or '').startswith(('TREASURY', 'AGENCY')):
            skipped_government += 1
            continue

        # Real dollar figures, not pre-divided ratios: mcap_to_debt and
        # fcf_to_debt both need the raw denominators, and an earlier version
        # passed total_debt as a unit placeholder which silently turned
        # mcap_to_debt into log10(mcap).
        fundamentals = {
            'int_cov': row.get('issuer_int_cov'),
            'nd_ebitda': row.get('issuer_nd_ebitda'),
            'fcf': row.get('issuer_fcf'),
            'total_debt': row.get('issuer_total_debt'),
            'altman_z': row.get('issuer_altman_z'),
            'revenue': row.get('issuer_revenue'),
            'piotroski': row.get('issuer_piotroski'),
            'cet1_ratio': row.get('issuer_cet1_ratio'),
            'npl_ratio': row.get('issuer_npl_ratio'),
            'mcap': row.get('issuer_mcap'),
            'sector': row.get('issuer_sector'),
        }
        result = credit.implied_bucket(fundamentals,
                                       sector=row.get('issuer_sector'),
                                       params=params)
        row['issuer_credit_score'] = result.get('score')
        row['issuer_credit_coverage'] = result.get('coverage')
        row['issuer_scorecard'] = result.get('scorecard')
        row['implied_bucket'] = result.get('bucket')
        row['_credit_confident'] = result.get('confident', False)
        row['asset_class'] = credit.asset_class_for(result.get('bucket'))
        if result.get('bucket'):
            scored += 1

    log.info('Credit model: %d of %d bonds have an implied bucket '
             '(%d government rows left untouched)',
             scored, len(rows) - skipped_government, skipped_government)
    return rows


def apply_fair_value(row, ctx, params, flows, settle):
    """Fair spread, fair price, mispricing and divergence for one bond.

    Runs after the Z-spread exists. Everything here is relative value: the
    bond's own spread against what an issuer of this quality and this tenor
    should be paying, per the market's own published pricing of that quality.
    """
    bucket = row.get('implied_bucket')
    ttm = row.get('years_to_maturity')
    observed_z = row.get('z_spread')
    beta = (params or {}).get('fair_spread_term_beta', 1.0)

    fair_z = credit.fair_spread(bucket, ttm, ctx.get('bucket_oas'),
                                term_points=ctx.get('term_points'),
                                wedge=ctx.get('wedge'), beta=beta,
                                term_by_bucket=ctx.get('term_by_bucket'),
                                bucket_anchors=ctx.get('bucket_anchors'))
    row['fair_spread'] = fair_z
    row['spread_mispricing'] = credit.spread_mispricing(observed_z, fair_z)

    if fair_z is not None:
        fair_dirty = credit.fair_price(flows, settle, ctx['curve'], fair_z)
        if fair_dirty is not None:
            fair_clean = fair_dirty - (row.get('accrued') or 0.0)
            row['fair_price'] = fair_clean
            row['price_mispricing'] = credit.price_mispricing(
                row.get('clean_price_est'), fair_clean)

    market = credit.market_implied_bucket(
        observed_z, ttm, ctx.get('bucket_oas'),
        term_points=ctx.get('term_points'), wedge=ctx.get('wedge'), beta=beta,
        term_by_bucket=ctx.get('term_by_bucket'),
        bucket_anchors=ctx.get('bucket_anchors'))
    row['market_bucket'] = market

    gap = credit.divergence(bucket, market,
                            fundamentals_asof=row.get('_fundamentals_asof'),
                            mark_date=row.get('mark_date'))
    row['bucket_divergence_notches'] = gap['notches']
    row['divergence_label'] = gap['label']
    row['_divergence_stale_risk'] = gap['stale_risk']
    # A divergence built from fundamentals NEWER than the price is comparing
    # two different moments; the gate must not read our own data lag as a
    # credit signal.
    if gap['stale_risk']:
        row['bucket_divergence_notches'] = None
    return row


def load_treasury_universe(as_of, max_years=31):
    td = TreasuryDirectClient()
    records = td.fetch_outstanding(as_of=as_of, max_years=max_years)
    rows = td.to_bond_rows(records, as_of=as_of)

    # Issue size comes from MSPD, not from TreasuryDirect's
    # currentlyOutstanding, which is populated for only ~40% of securities.
    # Partial coverage on a high-scoring gate made data availability the
    # largest single driver of the rating.
    matched = MSPDClient().attach(rows)
    if matched < 0.9 * len(rows):
        log.warning('MSPD covered only %d of %d rows — the Liquidity gate '
                    'will be uneven across the universe', matched, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Phase 3: per-bond analytics
# ---------------------------------------------------------------------------

def _price_with_overlay(row, bond, ctx, settle, flows, accrued):
    """Return (clean, dirty), ageing a stale mark forward onto today's curve.

    A MARK IS NOT A PRICE. N-PORT marks are month-end and reach us with a
    ~60-day lag, so using one directly as today's price is wrong by whatever
    the market did in between — and catastrophically wrong for short paper,
    which converges to par. A bill marked at 98.5 in April is worth ~99.9 in
    August when it has a week left; applied raw, that April price implied a
    54% yield and put four bills at the top of the BUY list.

    So the mark is used for what it actually observes — this bond's SPREAD to
    the curve on the day it was struck — and that spread is then applied to
    today's curve:

        z_at_mark  = z_spread(marked price, curve on the mark date)
        clean_est  = spread_to_price(z_at_mark, today's curve)

    For a Treasury the spread is ~0 and this collapses to repricing off
    today's curve, which is correct. For a corporate it preserves the credit
    information in the mark while removing the stale rate move — the honest
    version of "a daily price from monthly data". The overlay does not yet age
    the spread itself on the bucket-OAS move; that arrives with the corporate
    universe at M6.
    """
    curve = ctx['curve']
    marked = row.get('clean_price_marked')
    mark_date = row.get('mark_date')

    if marked is None or mark_date is None:
        dirty = price_from_zero_curve(flows, settle, curve, spread=0.0)
        if dirty is None:
            return None, None
        row['price_source'] = 'curve_implied'
        return dirty - accrued, dirty

    mark_curve = ctx['curve_at'](mark_date)
    if mark_curve is not None:
        # The bond as it stood on the mark date: its own remaining cashflows
        # then, not the ones it has now.
        past_flows, _ = bond_flows_and_stub(
            bond.coupon_rate, bond.maturity, mark_date,
            frequency=bond.frequency, convention=bond.convention,
            face=bond.face, dated_date=bond.dated_date, eom=bond.eom)
        past_accrued = accrued_interest(
            mark_date, bond.coupon_rate, bond.maturity,
            frequency=bond.frequency, face=bond.face,
            convention=bond.convention, dated_date=bond.dated_date,
            eom=bond.eom)
        if past_flows:
            z_at_mark = z_spread(marked + past_accrued, past_flows, mark_date,
                                 mark_curve)
            if z_at_mark is not None:
                dirty = price_from_zero_curve(flows, settle, curve,
                                              spread=z_at_mark)
                if dirty is not None:
                    row['z_spread_at_mark'] = z_at_mark
                    row['price_source'] = 'mark_aged_to_curve'
                    clean = dirty - accrued
                    drift = abs(clean - marked)
                    row['_mark_drift'] = drift
                    # A large gap between the raw mark and the aged estimate
                    # means the rate move since has dominated; worth surfacing
                    # rather than quietly presenting the estimate as a price.
                    row['_mark_drift_flag'] = drift > 3.0
                    return clean, dirty

    # The mark cannot be aged (no curve for that date, or the spread would not
    # solve). Falling back to the raw mark would reintroduce the stale-price
    # bug, so use the curve and keep the mark as reference only.
    dirty = price_from_zero_curve(flows, settle, curve, spread=0.0)
    if dirty is None:
        return None, None
    row['price_source'] = 'curve_implied_mark_unusable'
    return dirty - accrued, dirty


def _analyze_discount(row, bond, ctx, settle):
    """Bills and other single-cashflow instruments, on money-market terms."""
    curve = ctx['curve']
    flows = [(bond.maturity, bond.face)]

    # Discount instruments take the same overlay treatment, and need it most:
    # they converge to par fastest, so a stale mark is most wrong here.
    clean, dirty = _price_with_overlay(row, bond, ctx, settle, flows, 0.0)
    if clean is None:
        row['_drop_reason'] = 'curve pricing failed'
        return None

    # A discount instrument accrues nothing; clean and dirty coincide.
    row['accrued'] = 0.0
    row['clean_price_est'] = clean
    row['dirty_price'] = dirty

    stats = discount.analyze(bond.face, clean, settle, bond.maturity,
                             convention=bond.convention)
    if stats is None:
        row['ytm_solver_failed'] = True
        row['_drop_reason'] = 'discount analytics failed'
        return row
    row.update(stats)
    row['ytw_to_type'] = 'maturity'
    row['call_data_available'] = False
    row['current_yield'] = None          # no coupon to divide by price
    row['dv01'] = dv01(dirty, row['modified_duration'], face=bond.face)
    row['z_spread'] = z_spread(dirty, flows, settle, curve)
    row['yield_over_treasury'] = yield_over_treasury(
        row['ytm'], curve, row['years_to_maturity'])
    row['is_likely_callable'] = False
    row['carry_12m'] = 0.0               # income comes from pull-to-par
    row['roll_down_12m'] = roll_down(curve, row['years_to_maturity'], 1.0,
                                     row['modified_duration'])
    return row


def analyze_bond(row, ctx, settle, params):
    """Attach analytics to one row in place. Returns the row, or None.

    Never raises: a single pathological bond must not take down the run. Every
    failure sets a diagnostic field that a rating cap keys off.
    """
    bond, reason = from_row(row, settle=settle)
    if bond is None:
        row['_drop_reason'] = reason
        return None

    curve = ctx['curve']
    # Write the PARSED terms back onto the row. N-PORT reports the coupon in
    # percent and from_row normalises it to a decimal; leaving the raw value
    # in place makes every downstream consumer — report, snapshot, backtest —
    # read an 8.875% bond as 887.5%.
    row['coupon_rate'] = bond.coupon_rate
    row['frequency'] = bond.frequency
    row['convention'] = bond.convention
    row['years_to_maturity'] = ttm = years_to_maturity(settle, bond.maturity)
    row['_front_end_yield'] = ctx['front_end_yield']
    row['_curve_regime'] = ctx['regime']
    row['_curve_date'] = ctx['curve_date'].isoformat()

    # Discount instruments take a separate path with closed-form analytics.
    # Routing them through the coupon-bond formula discounts their single
    # cashflow over exactly one period regardless of maturity, which turned a
    # 5-day bill's yield into 0.05% instead of 3.9%.
    if bond.frequency == 0:
        return _analyze_discount(row, bond, ctx, settle)

    flows, w = bond_flows_and_stub(
        bond.coupon_rate, bond.maturity, settle, frequency=bond.frequency,
        convention=bond.convention, face=bond.face,
        dated_date=bond.dated_date, eom=bond.eom)
    if not flows:
        row['_drop_reason'] = 'no remaining cashflows'
        return None

    accrued = accrued_interest(settle, bond.coupon_rate, bond.maturity,
                               frequency=bond.frequency, face=bond.face,
                               convention=bond.convention,
                               dated_date=bond.dated_date, eom=bond.eom)
    row['accrued'] = accrued

    # -- price ---------------------------------------------------------------
    clean, dirty = _price_with_overlay(row, bond, ctx, settle, flows, accrued)
    if clean is None:
        row['_drop_reason'] = 'curve pricing failed'
        return None
    row['clean_price_est'] = clean
    row['dirty_price'] = dirty

    # -- yield ---------------------------------------------------------------
    ytm = yield_from_price(dirty, flows, frequency=bond.frequency, w=w)
    if ytm is None:
        row['ytm_solver_failed'] = True
        row['_drop_reason'] = 'yield solver did not converge'
        return row          # kept: the cap layer demotes it and reports why
    row['ytm'] = ytm
    row['current_yield'] = current_yield(clean, bond.coupon_rate, face=bond.face)

    ytw = yield_to_worst(clean, bond.coupon_rate, bond.maturity, settle,
                         call_schedule=bond.call_schedule,
                         frequency=bond.frequency, convention=bond.convention,
                         face=bond.face, dated_date=bond.dated_date,
                         eom=bond.eom)
    row['ytw'] = ytw['ytw']
    row['ytw_to_type'] = ytw['to_type']
    row['call_data_available'] = ytw['call_data_available']

    # -- risk ----------------------------------------------------------------
    mac = macaulay_duration(flows, ytm, frequency=bond.frequency, w=w)
    mod = modified_duration(mac, ytm, frequency=bond.frequency)
    cvx = convexity(flows, ytm, frequency=bond.frequency, w=w)
    row['macaulay_duration'] = mac
    row['modified_duration'] = mod
    row['convexity'] = cvx
    row['dv01'] = dv01(dirty, mod, face=bond.face)

    # -- spreads -------------------------------------------------------------
    row['z_spread'] = z_spread(dirty, flows, settle, curve)
    row['yield_over_treasury'] = yield_over_treasury(ytm, curve, ttm)
    row['is_likely_callable'] = is_likely_callable(row)

    # -- carry and roll ------------------------------------------------------
    row['carry_12m'] = carry(bond.coupon_rate, dirty, 365, face=bond.face)
    row['roll_down_12m'] = roll_down(curve, ttm, 1.0, mod)

    # -- relative value ------------------------------------------------------
    if row.get('implied_bucket') or row.get('z_spread') is not None:
        apply_fair_value(row, ctx, params, flows, settle)

    return row


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_bp(v):
    return f'{v * 10000:>7.0f}bp' if v is not None else '      —'


def print_summary(rows, ctx, dropped, settle):
    ratings = Counter(r.get('rating') for r in rows)
    total = len(rows)

    print(f"\n{'=' * 78}")
    print(f"  BOND ANALYSIS  settled {settle}  "
          f"curve {ctx['curve_date']}  n={total}")
    print(f"{'=' * 78}")

    regime = ctx['regime']
    if regime:
        print(f"  Curve: {regime.get('shape')} / {regime.get('direction')}   "
              f"10y {regime.get('level_10y', 0):.3%} "
              f"({regime.get('level_pctile_1y') or 0:.0f}th pctile)   "
              f"3m front {ctx['front_end_yield']:.3%}")

    print(f"\n  RATING DISTRIBUTION")
    for label in ('BUY', 'LEAN BUY', 'HOLD', 'PASS'):
        n = ratings.get(label, 0)
        pct = 100.0 * n / total if total else 0
        bar = '#' * int(pct / 2)
        print(f"    {label:<9} {n:>5}  {pct:>5.1f}%  {bar}")

    capped = sum(1 for r in rows if r.get('_rating_cap'))
    print(f"    {'capped':<9} {capped:>5}  "
          f"{100.0 * capped / total if total else 0:>5.1f}%   "
          f"(rating_raw preserved on every row)")

    # Which gates actually applied — the load-bearing check for this milestone.
    if rows:
        sample = rows[0]
        applicable = sample.get('_gates_applicable')
        inapplicable = sample.get('_gates_inapplicable')
        cats = sample.get('_composite_categories') or []
        print(f"\n  GATE APPLICABILITY (representative row)")
        print(f"    {applicable} applicable, {inapplicable} structurally "
              f"inapplicable, of {applicable + inapplicable} defined")
        print(f"    categories scoring: {', '.join(cats)}")
        dropped_cats = [c for c in SPEC.category_order if c not in cats]
        if dropped_cats:
            print(f"    categories dropped: {', '.join(dropped_cats)}  "
                  f"(composite renormalised over the rest)")

    print_gate_diagnostics(rows)
    print_concentration(rows)

    if dropped:
        print(f"\n  EXCLUDED FROM ANALYSIS ({sum(dropped.values())})")
        for reason, n in dropped.most_common():
            print(f"    {n:>5}  {reason}")

    if total and ratings.get('BUY', 0) / total > 0.05:
        print(f"\n  NOTE: {100.0 * ratings.get('BUY', 0) / total:.0f}% BUY is far "
              f"above the 1-3% a calibrated scale should produce.\n"
              f"  The thresholds in use (57/39/25) were quantile-matched on an "
              f"EQUITY universe\n  scoring 26 gates across 5 categories. A "
              f"Treasury scores 7 gates across 4,\n  so its composite is not "
              f"on that scale. Per-asset-class thresholds exist in\n"
              f"  param_set for exactly this reason but cannot be calibrated "
              f"until there is a\n  backtest to calibrate against (M8). Read "
              f"the RANKING, not the label, for now.")

    print(f"\n{'-' * 78}")


def print_concentration(rows):
    """Is the BUY list one idea, or several?

    A ranked screen can hand back twenty names that are all the same bet. For
    Treasuries that is the normal case rather than a pathology — with one
    issuer and one curve, "buy the long end" is a single view expressed
    however many times, and a buyer sizing thirteen positions off it is taking
    one concentrated duration position, not a diversified book. The model
    should say so rather than let the length of the list imply breadth.
    """
    buys = [r for r in rows if r.get('rating') == 'BUY']
    if len(buys) < 3:
        return

    buckets = Counter(r.get('peer_group', '?') for r in buys)
    top_bucket, top_n = buckets.most_common(1)[0]
    share = top_n / len(buys)

    print(f"\n  BUY-LIST CONCENTRATION ({len(buys)} names)")
    for bucket, n in buckets.most_common():
        print(f"    {bucket:<14} {n:>3}  {100.0 * n / len(buys):>5.1f}%")

    durations = [r['modified_duration'] for r in buys
                 if r.get('modified_duration') is not None]
    if durations:
        print(f"    duration range {min(durations):.1f}-{max(durations):.1f}, "
              f"mean {sum(durations) / len(durations):.1f}")

    if share >= 0.8:
        print(f"\n    {100 * share:.0f}% of the BUY list sits in {top_bucket}. "
              f"These are not\n    {len(buys)} independent ideas — they are one "
              f"duration position expressed\n    {len(buys)} ways. Size "
              f"accordingly.")


def print_gate_diagnostics(rows):
    """Per-gate coverage and score dispersion.

    Dispersion is the useful column. A gate whose score is near-identical for
    every row adds level to the composite but no discrimination — it cannot
    change anyone's rank, so it is pure inflation. That shows up here as a
    near-zero standard deviation, and it is worth catching early because the
    effect is invisible in a rating distribution: everything just drifts up
    together.
    """
    from statistics import pstdev

    from scripts.scoring_kernel import _score_key

    stats = []
    for gate in SPEC.gates:
        key = _score_key(gate.name)
        scores = [r[key] for r in rows if r.get(key) is not None]
        if not scores:
            continue
        stats.append((gate.name, len(scores), sum(scores) / len(scores),
                      pstdev(scores) if len(scores) > 1 else 0.0))
    if not stats:
        return

    print(f"\n  APPLICABLE GATES  ({len(stats)} of {len(SPEC.gates)} scored)")
    print(f"    {'gate':<32}{'n':>6}{'mean':>8}{'stdev':>8}")
    for name, n, mean, sd in stats:
        flag = '   <-- no discrimination' if sd < 1.0 else ''
        print(f"    {name:<32}{n:>6}{mean:>8.1f}{sd:>8.1f}{flag}")

    flat = [s for s in stats if s[3] < 1.0]
    if flat:
        print(f"\n    {len(flat)} gate(s) score every row alike: they lift the "
              f"composite without\n    changing any ranking. Expected here — "
              f"Analyzability is 100 for every\n    Treasury that survived "
              f"construction, by definition. Worth re-checking\n    once "
              f"corporates arrive, where it should start to discriminate.")


def print_table(rows, limit=25):
    ranked = sorted(rows, key=lambda r: (r.get('_composite_score') or -1),
                    reverse=True)[:limit]
    print(f"  TOP {len(ranked)} BY COMPOSITE\n")
    print(f"  {'cusip':<11}{'maturity':<12}{'cpn':>7}{'ytw':>8}{'dur':>7}"
          f"{'cvx':>7}{'roll':>9}{'score':>7}  {'rating':<9}{'caps'}")
    print(f"  {'-' * 76}")
    for r in ranked:
        caps = r.get('_rating_cap_reasons') or []
        cap_note = f"! {caps[0][:26]}" if caps else ''
        print(f"  {r.get('cusip', ''):<11}"
              f"{str(r.get('maturity_date', '')):<12}"
              f"{(r.get('coupon_rate') or 0):>6.2%}"
              f"{(r.get('ytw') or 0):>8.3%}"
              f"{(r.get('modified_duration') or 0):>7.2f}"
              f"{(r.get('convexity') or 0):>7.1f}"
              f"{_fmt_bp(r.get('roll_down_12m'))}"
              f"{(r.get('_composite_score') or 0):>7.1f}  "
              f"{r.get('rating', ''):<9}{cap_note}")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def write_snapshot(rows, ctx, settle, as_json=False):
    """Persist the run. Parquet by default.

    30k rows x ~120 keys of pretty JSON runs 45-75 MB/day, which is exactly
    the problem that forced the equity model onto a force-pushed single-commit
    branch. Parquet is ~5-10 MB and makes backtest/calibrate — which load
    dozens of snapshots — roughly 10x faster.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = settle.isoformat()

    meta = {
        'run_date': stamp,
        'curve_date': ctx['curve_date'].isoformat(),
        'front_end_yield': ctx['front_end_yield'],
        'regime': {k: (v.isoformat() if isinstance(v, date) else v)
                   for k, v in (ctx['regime'] or {}).items()},
        'bucket_oas': ctx['bucket_oas'],
        'term_points': ctx['term_points'],
        'fred_history_source': ctx['fred_source'],
        'count': len(rows),
        'par_curve': ctx['par'],
    }
    meta_path = os.path.join(OUTPUT_DIR, f'run_meta_{stamp}.json')
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2, default=str)

    # The regime dict is per-run context, not per-row data; it would bloat
    # every row and does not belong in a columnar store.
    flat = [{k: v for k, v in r.items() if k != '_curve_regime'} for r in rows]

    if as_json:
        path = os.path.join(OUTPUT_DIR, f'results_{stamp}.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'meta': meta, 'results': flat}, fh, default=str)
    else:
        import pandas as pd
        path = os.path.join(OUTPUT_DIR, f'results_{stamp}.parquet')
        pd.DataFrame(flat).astype(
            {c: 'object' for c in ('cusip', 'rating') if c in flat[0]}
        ).to_parquet(path, index=False)

    size_mb = os.path.getsize(path) / 1e6
    log.info('Wrote %s (%.2f MB) and %s', os.path.basename(path), size_mb,
             os.path.basename(meta_path))
    return path


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--universe', default='treasury',
                    choices=['treasury', 'corporate', 'all'])
    ap.add_argument('--quarter', default='2026q2',
                    help='N-PORT quarter backing the corporate universe')
    ap.add_argument('--as-of', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                    default=None)
    ap.add_argument('--max-years', type=int, default=31)
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--json', action='store_true',
                    help='write JSON instead of parquet (debugging)')
    ap.add_argument('--no-fred', action='store_true')
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--marks', default=None, metavar='QUARTER',
                    help='attach N-PORT consensus marks from this quarter '
                         '(default: the newest ingested)')
    ap.add_argument('--no-marks', action='store_true',
                    help='ignore N-PORT marks; price everything off the curve')
    args = ap.parse_args()

    settle = args.as_of or date.today()
    params = default_params()
    errors = validate_params(params)
    if errors:
        raise SystemExit('[fatal] invalid params:\n  ' + '\n  '.join(errors))

    log.info('Phase 0: market context')
    ctx = build_context(settle, use_fred=not args.no_fred)

    log.info('Phase 1: universe (%s)', args.universe)
    raw = []
    if args.universe in ('treasury', 'all'):
        raw += load_treasury_universe(settle, max_years=args.max_years)
    if args.universe in ('corporate', 'all'):
        raw += load_corporate_universe(args.quarter)
    log.info('  %d reference rows', len(raw))

    if not args.no_marks:
        attach_nport_marks(raw, settle, quarter=args.marks)

    log.info('Phase 2: credit model')
    apply_credit_model(raw, params)

    log.info('Phase 3: analytics')
    rows, dropped = [], Counter()
    for row in raw:
        out = analyze_bond(row, ctx, settle, params)
        if out is None:
            dropped[row.get('_drop_reason', 'unknown')] += 1
        else:
            rows.append(out)
    log.info('  %d analysed, %d excluded', len(rows), sum(dropped.values()))

    if not rows:
        raise SystemExit('[fatal] no analysable bonds')

    log.info('Phase 4: scoring')
    score_and_rate(rows, SPEC, params=params)

    print_summary(rows, ctx, dropped, settle)
    print_table(rows, limit=args.top)

    if not args.no_write:
        log.info('Phase 5: artifacts')
        write_snapshot(rows, ctx, settle, as_json=args.json)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
