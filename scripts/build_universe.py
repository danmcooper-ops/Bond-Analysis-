#!/usr/bin/env python3
"""Assemble the corporate bond universe and report crosswalk coverage.

    python scripts/build_universe.py --quarter 2026q2
    python scripts/build_universe.py --quarter 2026q2 --audit 30

Joins N-PORT consensus marks to issuer fundamentals via the CUSIP crosswalk,
applies the universe filters, and writes output/universe_{month}.parquet.

The coverage report is the point of this script as much as the parquet is.
It separates three outcomes that look identical downstream but need completely
different fixes:

    matched with fundamentals  -> full credit gates
    matched, no fundamentals   -> a real SEC filer we have not fetched
    unresolved name            -> private, foreign, or a structure with no
                                  US filing behind it

Only the middle band is a fetching problem. Conflating it with the third would
make the crosswalk look far worse than it is, and would send effort at name
matching when the real gap is coverage.
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cusip_crosswalk import CusipCrosswalk
from data.issuer_fundamentals import IssuerFundamentals
from data.logging_setup import get_logger
from data.nport_client import NPORTClient
from data.nport_consensus import latest_marks
from scripts.config import (MIN_FUNDS_HOLDING, MIN_TOTAL_HELD_USD,
                            MIN_YEARS_TO_MATURITY, PRICE_SANITY_MAX,
                            PRICE_SANITY_MIN)

log = get_logger('build_universe')

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')

CORPORATE_ISSUER_TYPES = ('CORP',)


def load_marks(quarter):
    client = NPORTClient()
    path = os.path.join(client.cache_dir, f'{quarter}_marks.parquet')
    if not os.path.exists(path):
        raise SystemExit(f'[fatal] {path} not found — run '
                         f'scripts/ingest_nport.py --quarter {quarter}')
    import pandas as pd
    return latest_marks(pd.read_parquet(path).to_dict('records'))


def group_by_issuer(marks):
    """{cusip6: {names, held_by_name, held}} — the crosswalk's input.

    Names are weighted by the dollars behind them so the consistency vote
    reflects conviction: the same issuer appears under five or ten spellings
    across funds, and a spelling backed by $8bn is better evidence than one
    backed by $2m.
    """
    groups = defaultdict(lambda: {'names': set(), 'held_by_name':
                                  defaultdict(float), 'held': 0.0})
    for mark in marks:
        prefix = mark['cusip'][:6].upper()
        name = (mark.get('issuer_name') or '').strip()
        held = mark.get('total_held_usd') or 0.0
        entry = groups[prefix]
        entry['held'] += held
        if name:
            entry['names'].add(name)
            entry['held_by_name'][name] += held
    return {k: {'names': sorted(v['names']),
                'held_by_name': dict(v['held_by_name']),
                'held': v['held']}
            for k, v in groups.items()}


def load_sec_filers():
    """[(ticker, title)] for every SEC filer, from company_tickers.json.

    A SECOND index, behind the fundamentals one. Its only job is to tell
    "we cannot identify this issuer" apart from "we identified it and hold no
    financials" — two outcomes that look identical on a bond row but need
    completely different work to fix. Without it every issuer outside the
    equity model's screened universe reads as a crosswalk failure, which
    points effort at name matching when the real gap is coverage.
    """
    import json

    from data.http import get_json

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'data', 'cache', 'sec')
    path = os.path.join(cache_dir, 'company_tickers.json')
    payload = None
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            payload = None
    if payload is None:
        payload = get_json('https://www.sec.gov/files/company_tickers.json')
        if payload:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)
    if not payload:
        log.warning('company_tickers.json unavailable; cannot distinguish '
                    '"no fundamentals" from "unidentified issuer"')
        return []
    return [(v.get('ticker'), v.get('title')) for v in payload.values()
            if v.get('ticker') and v.get('title')]


def build(quarter, as_of, min_funds, min_held, audit=0):
    marks = load_marks(quarter)
    corporates = [m for m in marks if m.get('issuer_type') in CORPORATE_ISSUER_TYPES]
    log.info('%s: %d corporate CUSIPs after de-duplicating to the latest month',
             quarter, len(corporates))

    fundamentals = IssuerFundamentals()
    crosswalk = CusipCrosswalk(index=fundamentals.names())
    filer_crosswalk = CusipCrosswalk(index=load_sec_filers())

    groups = group_by_issuer(corporates)
    resolutions = crosswalk.resolve_all(groups)

    # Anything the fundamentals index could not place gets a second pass
    # against the full SEC filer list, purely for diagnosis.
    for prefix, resolution in resolutions.items():
        if resolution.get('key') or resolution.get('method') == 'override_no_fundamentals':
            continue
        fallback = filer_crosswalk.resolve(prefix, groups[prefix].get('names'),
                                           groups[prefix].get('held_by_name'))
        if fallback.get('key'):
            resolution.update({
                'filer_key': fallback['key'],
                'filer_method': fallback.get('method'),
                'matched_name': fallback.get('matched_name'),
            })

    rows = []
    filtered = Counter()
    for mark in corporates:
        maturity = mark.get('maturity_date')
        if maturity is not None and hasattr(maturity, 'date'):
            maturity = maturity.date()
        report_date = mark.get('report_date')
        if hasattr(report_date, 'date'):
            report_date = report_date.date()

        if maturity is None:
            filtered['no maturity'] += 1
            continue
        years = (maturity - as_of).days / 365.25
        if years < MIN_YEARS_TO_MATURITY:
            filtered[f'under {MIN_YEARS_TO_MATURITY}y to maturity'] += 1
            continue
        if (mark.get('n_funds') or 0) < min_funds:
            filtered[f'fewer than {min_funds} funds'] += 1
            continue
        if (mark.get('total_held_usd') or 0) < min_held:
            filtered[f'under ${min_held / 1e6:.0f}m held'] += 1
            continue
        price = mark.get('clean_price_marked')
        if price is None or not (PRICE_SANITY_MIN <= price <= PRICE_SANITY_MAX):
            filtered['price outside sanity band'] += 1
            continue

        row = dict(mark)
        row['maturity_date'] = maturity
        row['report_date'] = report_date
        row['mark_date'] = report_date
        row['mark_age_days'] = (as_of - report_date).days
        row['years_to_maturity'] = years
        row['asset_class'] = 'CORP_IG'      # refined by the credit model at M6
        row['coupon_rate'] = mark.get('annualized_rate')

        resolution = resolutions.get(mark['cusip'][:6].upper(), {})
        fundamentals.attach(row, resolution, as_of)
        rows.append(row)

    log.info('Universe: %d bonds (%d filtered out)', len(rows),
             sum(filtered.values()))
    report(rows, groups, resolutions, fundamentals, filtered, audit)
    return rows


def report(rows, groups, resolutions, fundamentals, filtered, audit):
    total_held = sum(r.get('total_held_usd') or 0 for r in rows)

    print(f"\n{'=' * 78}")
    print(f"  CORPORATE UNIVERSE  —  {len(rows):,} bonds, "
          f"${total_held / 1e9:,.0f}bn held")
    print(f"{'=' * 78}")

    if filtered:
        print(f"\n  FILTERED OUT ({sum(filtered.values()):,})")
        for reason, n in filtered.most_common():
            print(f"    {n:>7,}  {reason}")

    # -- the three outcomes -------------------------------------------------
    with_fund = [r for r in rows if r.get('issuer_cik')]
    matched_no_fund = [r for r in rows if r.get('_fundamentals_missing')]
    unresolved = [r for r in rows
                  if not r.get('issuer_cik') and not r.get('_fundamentals_missing')]

    def share(subset):
        held = sum(r.get('total_held_usd') or 0 for r in subset)
        return len(subset), 100.0 * held / total_held if total_held else 0.0

    print(f"\n  ISSUER RESOLUTION (share of held value)")
    for label, subset in (('matched, has fundamentals', with_fund),
                          ('matched, no fundamentals yet', matched_no_fund),
                          ('unresolved issuer name', unresolved)):
        n, pct = share(subset)
        print(f"    {label:<32}{n:>8,} bonds{pct:>8.1f}%")
    print(f"\n    Only the middle band is a fetching problem — those issuers "
          f"file with the\n    SEC and a XBRL backend would recover them. The "
          f"third is the hard residual:\n    private issuers, foreign banks, "
          f"and structures with no US filing behind them.")

    # -- confidence ---------------------------------------------------------
    print(f"\n  MATCH CONFIDENCE")
    bands = Counter()
    for r in rows:
        conf = r.get('cusip_match_confidence') or 0.0
        band = ('>=0.95' if conf >= 0.95 else '0.90-0.95' if conf >= 0.90
                else '0.80-0.90' if conf >= 0.80 else '<0.80 (capped)')
        bands[band] += 1
    for band in ('>=0.95', '0.90-0.95', '0.80-0.90', '<0.80 (capped)'):
        if bands[band]:
            print(f"    {band:<18}{bands[band]:>8,}")

    print(f"\n  MATCH METHOD")
    for method, n in Counter(r.get('cusip_match_method') or 'none'
                             for r in rows).most_common(8):
        print(f"    {method:<28}{n:>8,}")

    subs = sum(1 for r in rows if r.get('issuer_is_finance_sub'))
    if subs:
        print(f"\n    {subs:,} bonds resolved through a financing subsidiary to "
              f"the parent.\n    Usually right — the parent guarantees the debt "
              f"— but it can mask structural\n    subordination, so those rows "
              f"carry issuer_is_finance_sub.")

    # -- the largest misses, which is where curation pays ------------------
    missed = sorted(
        ((groups[c]['held'], c, groups[c]['names'][:1])
         for c, r in resolutions.items()
         if not r.get('key') and c in groups), reverse=True)[:12]
    if missed:
        print(f"\n  LARGEST UNRESOLVED ISSUERS (each is one override entry)")
        for held, prefix, names in missed:
            name = names[0][:46] if names else '?'
            print(f"    {prefix}  ${held / 1e9:>6.2f}bn  {name}")

    # -- fundamentals sparsity ---------------------------------------------
    if with_fund:
        print(f"\n  FUNDAMENTALS COVERAGE among the {len(with_fund):,} matched bonds")
        for field in ('issuer_int_cov', 'issuer_nd_ebitda', 'issuer_altman_z',
                      'issuer_piotroski', 'issuer_fcf_to_debt',
                      'issuer_debt_maturity_wall_yrs'):
            n = sum(1 for r in with_fund if r.get(field) is not None)
            print(f"    {field:<34}{n:>8,}{100.0 * n / len(with_fund):>7.1f}%")
        banks = [r for r in with_fund
                 if r.get('issuer_sector') == 'Financial Services']
        if banks:
            cet1 = sum(1 for r in banks if r.get('issuer_cet1_ratio') is not None)
            print(f"    {'issuer_cet1_ratio (banks only)':<34}{cet1:>8,}"
                  f"{100.0 * cet1 / len(banks):>7.1f}%  of {len(banks):,} bank bonds")
            if cet1 < 0.5 * len(banks):
                print(f"\n    Most bank bonds have no CET1 or NPL. Those gates "
                      f"apply but have no data,\n    so they score zero — the "
                      f"house rule for missing data. Bank composites are\n"
                      f"    therefore understated until the FDIC enrichment "
                      f"reaches more of them.")

    if audit:
        print(f"\n  RANDOM MATCH AUDIT ({audit} rows) — check these by hand")
        import random
        random.seed(42)
        sample = random.sample([r for r in rows if r.get('issuer_ticker')],
                               min(audit, len(with_fund)))
        print(f"    {'cusip':<11}{'n-port issuer name':<38}"
              f"{'->':<3}{'ticker':<8}{'conf':>6}  method")
        for r in sample:
            print(f"    {r['cusip']:<11}{(r.get('issuer_name') or '')[:36]:<38}"
                  f"{'->':<3}{r.get('issuer_ticker', ''):<8}"
                  f"{r.get('cusip_match_confidence', 0):>6.2f}  "
                  f"{r.get('cusip_match_method', '')}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quarter', default='2026q2')
    ap.add_argument('--as-of', type=lambda s: datetime.strptime(s, '%Y-%m-%d').date(),
                    default=date.today())
    ap.add_argument('--min-funds', type=int, default=MIN_FUNDS_HOLDING)
    ap.add_argument('--min-held', type=float, default=MIN_TOTAL_HELD_USD)
    ap.add_argument('--audit', type=int, default=0,
                    help='print N random matches for hand-checking')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()

    rows = build(args.quarter, args.as_of, args.min_funds, args.min_held,
                 audit=args.audit)
    if not rows:
        raise SystemExit('[fatal] empty universe')

    if not args.no_write:
        import pandas as pd
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f'universe_{args.quarter}.parquet')
        pd.DataFrame(rows).to_parquet(path, index=False)
        log.info('Wrote %s (%d rows, %.1f MB)', os.path.basename(path),
                 len(rows), os.path.getsize(path) / 1e6)
    return 0


if __name__ == '__main__':
    sys.exit(main())
