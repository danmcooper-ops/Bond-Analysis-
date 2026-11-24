#!/usr/bin/env python3
"""Build bond price history: three series, each labelled for what it is.

    python scripts/build_price_history.py                    # all three
    python scripts/build_price_history.py --treasury --since 2019-01-01
    python scripts/build_price_history.py --show 912810TM0   # one CUSIP, every series

Writes to output/history/:

  marks_monthly.parquet     OBSERVED, monthly, ~60-day lag. N-PORT consensus
                            marks (median across funds) for every CUSIP-month
                            in every ingested quarter. Two quarters can report
                            the same CUSIP-month; the one with more funds wins.
                            Corporates, agencies and Treasuries alike.
  treasury_daily.parquet    OBSERVED, daily. FedInvest end-of-day prices for
                            every marketable Treasury. TIPS are real prices.
  estimates_daily.parquet   MODELLED, daily. clean_price_est from every
                            output/results_<date>.parquet: the last mark aged
                            forward on the bucket-OAS move. Never call it a
                            price. That is the first honest limit in CLAUDE.md.

Each file is rebuilt in full from its sources on every run. The sources are
the N-PORT marks cache, the FedInvest day cache and the results snapshots, so
a rebuild is idempotent and costs seconds once the caches are warm. Only
--treasury touches the network, and only for weekdays not yet cached.

Deeper monthly history means ingesting older N-PORT quarters first
(scripts/ingest_nport.py --quarter 2019q4 ...); 2019q4 is the first published.
"""

import argparse
import glob
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.fedinvest_client import FedInvestClient, parse_csv
from data.logging_setup import get_logger
from data.nport_client import NPORTClient

log = get_logger('price_history')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(ROOT, 'output')
HISTORY_DIR = os.path.join(OUTPUT_DIR, 'history')
DEFAULT_TREASURY_SINCE = date(2019, 1, 1)    # a year before N-PORT starts

MARK_COLUMNS = ['cusip', 'report_date', 'clean_price_marked', 'price_basis',
                'price_min', 'price_max', 'price_dispersion', 'n_funds',
                'total_held_usd', 'issuer_type', 'issuer_name',
                'title_of_issue', 'maturity_date', 'annualized_rate',
                'coupon_type', 'is_default', 'in_arrears', 'is_convertible']

ESTIMATE_COLUMNS = ['cusip', 'asset_class', 'issuer_name', 'maturity_date',
                    'clean_price_est', 'price_source', 'clean_price_marked',
                    'mark_date', 'mark_age_days', 'ytm', 'z_spread',
                    'model_version']


def _write(frame, name):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, name)
    tmp = path + '.tmp'
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


# -- 1. monthly marks ---------------------------------------------------------

def marks_frame(cache_dir):
    import pandas as pd

    names = sorted(n for n in os.listdir(cache_dir)
                   if n.endswith('_marks.parquet')) \
        if os.path.isdir(cache_dir) else []
    frames = []
    for name in names:
        frame = pd.read_parquet(os.path.join(cache_dir, name))
        frame = frame[[c for c in MARK_COLUMNS if c in frame.columns]].copy()
        frame['source_quarter'] = name.split('_')[0]
        frames.append(frame)
    if not frames:
        return None
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel['clean_price_marked'].notna()].copy()
    # Funds date the same month-end differently (2025-05-30, the last
    # business day, vs 2025-05-31), so one month can arrive as two rows, the
    # odd one often from one to three funds. Key on the month and keep the
    # widest consensus; report_date stays the chosen row's own date.
    panel['month'] = pd.to_datetime(panel['report_date']).dt.to_period('M') \
        .dt.to_timestamp('M')
    panel = (panel.sort_values(['n_funds', 'source_quarter'],
                               ascending=[False, False])
             .drop_duplicates(subset=['cusip', 'month'], keep='first')
             .sort_values(['cusip', 'month'])
             .reset_index(drop=True))
    return panel


def build_marks():
    client = NPORTClient()
    panel = marks_frame(client.cache_dir)
    if panel is None:
        log.warning('No N-PORT marks cached; run scripts/ingest_nport.py')
        return None
    path = _write(panel, 'marks_monthly.parquet')
    print(f'marks_monthly: {len(panel):,} CUSIP-months, '
          f'{panel["cusip"].nunique():,} CUSIPs, '
          f'{panel["month"].nunique()} months '
          f'({panel["month"].min():%Y-%m} to '
          f'{panel["month"].max():%Y-%m}) -> {path}')
    return panel


# -- 2. daily Treasuries ------------------------------------------------------

def treasury_frame(client, days):
    import pandas as pd

    rows = []
    for day in days:
        with open(client.cache_path(day), encoding='utf-8') as fh:
            rows.extend(parse_csv(fh.read(), day))
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    for col in ('price_date', 'maturity_date', 'call_date'):
        frame[col] = pd.to_datetime(frame[col])
    return frame.sort_values(['cusip', 'price_date']).reset_index(drop=True)


def build_treasury(since, until, fetch=True):
    client = FedInvestClient()
    if fetch:
        fetched, failed = client.backfill(since, until)
        print(f'FedInvest: fetched {fetched} new day(s) '
              f'{since} to {until}' + (f'; {len(failed)} failed '
                                      f'(first {failed[0]}), rerun to retry'
                                      if failed else ''))
    frame = treasury_frame(client, client.cached_days())
    if frame is None:
        log.warning('No FedInvest prices cached')
        return None
    path = _write(frame, 'treasury_daily.parquet')
    print(f'treasury_daily: {len(frame):,} rows, '
          f'{frame["cusip"].nunique():,} CUSIPs, '
          f'{frame["price_date"].nunique():,} days '
          f'({frame["price_date"].min():%Y-%m-%d} to '
          f'{frame["price_date"].max():%Y-%m-%d}) -> {path}')
    return frame


# -- 3. daily estimates -------------------------------------------------------

def estimates_frame(paths):
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for path in paths:
        stamp = os.path.basename(path)[len('results_'):-len('.parquet')]
        try:
            run_date = datetime.strptime(stamp, '%Y-%m-%d')
        except ValueError:
            continue
        have = set(pq.read_schema(path).names)
        cols = [c for c in ESTIMATE_COLUMNS if c in have]
        frame = pd.read_parquet(path, columns=cols)
        frame.insert(0, 'run_date', run_date)
        frames.append(frame)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame['clean_price_est'].notna()]
    return frame.sort_values(['cusip', 'run_date']).reset_index(drop=True)


def build_estimates():
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'results_*.parquet')))
    frame = estimates_frame(paths)
    if frame is None:
        log.warning('No output/results_*.parquet snapshots')
        return None
    path = _write(frame, 'estimates_daily.parquet')
    print(f'estimates_daily: {len(frame):,} rows, '
          f'{frame["cusip"].nunique():,} CUSIPs, '
          f'{frame["run_date"].nunique()} run days '
          f'({frame["run_date"].min():%Y-%m-%d} to '
          f'{frame["run_date"].max():%Y-%m-%d}) -> {path}')
    return frame


# -- lookup -------------------------------------------------------------------

def show(cusip):
    import pandas as pd

    cusip = cusip.upper()
    for name, date_col, price_cols in (
            ('marks_monthly.parquet', 'month',
             ['clean_price_marked', 'price_min', 'price_max', 'n_funds']),
            ('treasury_daily.parquet', 'price_date',
             ['end_of_day', 'buy', 'sell', 'security_type']),
            ('estimates_daily.parquet', 'run_date',
             ['clean_price_est', 'price_source', 'mark_date'])):
        path = os.path.join(HISTORY_DIR, name)
        if not os.path.exists(path):
            print(f'\n{name}: not built')
            continue
        frame = pd.read_parquet(path, filters=[('cusip', '==', cusip)])
        print(f'\n{name}: {len(frame)} rows')
        if len(frame):
            print(frame[[date_col] + price_cols].to_string(index=False,
                                                          max_rows=40))


def _date(text):
    return datetime.strptime(text, '%Y-%m-%d').date()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--marks', action='store_true', help='monthly N-PORT marks')
    ap.add_argument('--treasury', action='store_true',
                    help='daily FedInvest Treasury prices')
    ap.add_argument('--estimates', action='store_true',
                    help='daily modelled clean_price_est')
    ap.add_argument('--since', type=_date, default=None,
                    help='FedInvest backfill start (default: '
                         f'{DEFAULT_TREASURY_SINCE}; cached days are skipped)')
    ap.add_argument('--until', type=_date, default=None,
                    help='FedInvest backfill end (default: yesterday)')
    ap.add_argument('--no-fetch', action='store_true',
                    help='build treasury_daily from the cache only')
    ap.add_argument('--show', metavar='CUSIP',
                    help='print every series for one CUSIP and exit')
    args = ap.parse_args()

    if args.show:
        show(args.show)
        return

    every = not (args.marks or args.treasury or args.estimates)
    if every or args.marks:
        build_marks()
    if every or args.treasury:
        until = args.until or date.today() - timedelta(days=1)
        # Always walk from the start: cached days cost no request, so this
        # retries any day that failed before and any recent day that was
        # still empty (not yet published, so never cached).
        build_treasury(args.since or DEFAULT_TREASURY_SINCE, until,
                       fetch=not args.no_fetch)
    if every or args.estimates:
        build_estimates()


if __name__ == '__main__':
    main()
