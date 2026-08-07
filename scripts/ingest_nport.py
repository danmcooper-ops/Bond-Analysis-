#!/usr/bin/env python3
"""Download a quarterly N-PORT data set and build consensus marks.

    python scripts/ingest_nport.py --quarter 2026q2
    python scripts/ingest_nport.py --latest
    python scripts/ingest_nport.py --list

Writes data/cache/nport/{quarter}_marks.parquet — one row per CUSIP-month,
which is the durable artifact. The 440 MB ZIP can be dropped afterwards with
--drop-zip, though keeping it makes a re-parse free if the consensus logic
changes.

Runtime is dominated by the download (~4 min) and the streaming parse of a
910 MB table (~2 min). Idempotent: a cached ZIP is reused.
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.logging_setup import get_logger
from data.nport_client import NPORTClient
from data.nport_consensus import consensus_mark

log = get_logger('ingest_nport')


def marks_path(client, quarter):
    return os.path.join(client.cache_dir, f'{quarter}_marks.parquet')


def ingest(quarter, force=False, drop_zip=False):
    client = NPORTClient()
    if client.download_quarter(quarter, force=force) is None:
        return None

    holdings = client.build_holdings(quarter)
    if not holdings:
        log.error('%s: no usable debt holdings', quarter)
        return None

    marks = consensus_mark(holdings)
    if not marks:
        log.error('%s: no consensus marks', quarter)
        return None

    import pandas as pd
    frame = pd.DataFrame(marks)
    path = marks_path(client, quarter)
    frame.to_parquet(path, index=False)
    log.info('Wrote %s (%d rows, %.1f MB)', os.path.basename(path),
             len(frame), os.path.getsize(path) / 1e6)

    _summarise(frame)

    if drop_zip:
        zip_path = client.zip_path(quarter)
        if os.path.exists(zip_path):
            os.remove(zip_path)
            log.info('Removed %s', os.path.basename(zip_path))
    return path


def _summarise(frame):
    print(f"\n{'=' * 72}")
    print(f"  N-PORT CONSENSUS MARKS  —  {len(frame):,} CUSIP-months")
    print(f"{'=' * 72}")

    by_month = Counter(frame['report_date'].astype(str))
    print("\n  BY REPORT MONTH (the mark date, not the fund's fiscal year end)")
    for month, n in sorted(by_month.items()):
        print(f"    {month}   {n:>7,}")

    print("\n  BY ISSUER TYPE")
    for issuer_type, n in Counter(frame['issuer_type'].fillna('?')).most_common():
        print(f"    {issuer_type:<8} {n:>7,}")

    funds = frame['n_funds']
    print(f"\n  FUND BREADTH   1 fund: {(funds == 1).sum():,}   "
          f"2: {(funds == 2).sum():,}   3-9: {funds.between(3, 9).sum():,}   "
          f"10+: {(funds >= 10).sum():,}   max: {funds.max()}")

    prices = frame['clean_price_marked']
    print(f"  PRICE          p5 {prices.quantile(0.05):.2f}   "
          f"median {prices.median():.2f}   p95 {prices.quantile(0.95):.2f}")

    # Dispersion is what justifies trusting a consensus at all.
    multi = frame[frame['n_funds'] >= 3]
    if len(multi):
        disp = multi['price_dispersion'].dropna()
        print(f"  DISPERSION     (n>=3 funds, {len(multi):,} rows)  "
              f"median {disp.median():.3%}   p90 {disp.quantile(0.90):.3%}   "
              f"above 2%: {(disp > 0.02).sum():,}")

    flags = {'default': frame['is_default'].sum(),
             'in arrears': frame['in_arrears'].sum(),
             'PIK': frame['is_paid_kind'].sum(),
             'convertible': frame['is_convertible'].sum(),
             'term conflict': frame['_identity_conflict'].sum(),
             'level 3': (frame['fair_value_level'] == 3).sum()}
    print("\n  FLAGS          " + '   '.join(f'{k}: {v:,}' for k, v in flags.items()))
    print()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quarter', help='e.g. 2026q2')
    ap.add_argument('--latest', action='store_true',
                    help='ingest the most recent published quarter')
    ap.add_argument('--list', action='store_true',
                    help='list published quarters and exit')
    ap.add_argument('--force', action='store_true', help='re-download')
    ap.add_argument('--drop-zip', action='store_true',
                    help='delete the ZIP once the marks are built')
    args = ap.parse_args()

    client = NPORTClient()
    if args.list:
        quarters = client.list_available_quarters()
        print(f"\n  {len(quarters)} quarters published:\n")
        for i in range(0, len(quarters), 6):
            print('    ' + '  '.join(quarters[i:i + 6]))
        cached = sorted(f.split('_')[0] for f in os.listdir(client.cache_dir)
                        if f.endswith('_nport.zip')) if os.path.isdir(
            client.cache_dir) else []
        print(f"\n  cached locally: {', '.join(cached) if cached else 'none'}\n")
        return 0

    quarter = args.quarter
    if args.latest or not quarter:
        quarters = client.list_available_quarters()
        if not quarters:
            raise SystemExit('[fatal] could not list published quarters')
        quarter = quarters[-1]
        log.info('Latest published quarter: %s', quarter)

    return 0 if ingest(quarter, force=args.force,
                       drop_zip=args.drop_zip) else 1


if __name__ == '__main__':
    sys.exit(main())
