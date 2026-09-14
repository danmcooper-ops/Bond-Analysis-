#!/usr/bin/env python3
"""Health checks for the scheduled daily run. Report-only; writes nothing.

    python scripts/daily_checks.py gaps                     # before analysis
    python scripts/daily_checks.py ratings --run-date 2026-09-14
    python scripts/daily_checks.py nport                    # network: DERA index

Each check prints its findings and exits 0; lines starting with "ALERT" are the
ones the run summary must surface. They exist because every failure below
happened silently: weekday runs that never fired (08-25, 09-10, 09-11), a month
of reports with zero corporate BUYs that the "moved more than 3pp" drift check
could not see, and marks ageing past a new N-PORT quarter nobody ingested.
"""

import argparse
import glob
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')

CLASSES = ('CORP_IG', 'CORP_HY', 'TREASURY')
RATINGS = ('BUY', 'LEAN BUY', 'HOLD', 'PASS')
BUY_DRIFT_PP = 3.0
MAX_CAPPED_SHARE = 0.90
NPORT_VINTAGE_WARN_DAYS = 150


def _dates(pattern):
    """Dates stamped into output/<pattern> filenames, sorted."""
    prefix, suffix = pattern.split('*')
    out = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
        stamp = os.path.basename(path)[len(prefix):-len(suffix)]
        try:
            out.append(datetime.strptime(stamp, '%Y-%m-%d').date())
        except ValueError:
            continue
    return sorted(out)


def missed_weekdays(completed, today):
    """Weekdays strictly between the last completed run and today with no run.

    Holidays are reported too: a run is expected on them (the Treasury curve
    falls back a day), and the scheduler does fire on them.
    """
    if not completed:
        return []
    day, out = completed[-1] + timedelta(days=1), []
    done = set(completed)
    while day < today:
        if day.weekday() < 5 and day not in done:
            out.append(day)
        day += timedelta(days=1)
    return out


def check_gaps(today):
    completed = _dates('run_meta_*.json')
    if not completed:
        print('ALERT no completed runs found in output/')
        return
    print(f'Last completed run: {completed[-1]}')
    missed = missed_weekdays(completed, today)
    if missed:
        print(f'ALERT missed scheduled runs: {", ".join(map(str, missed))}')
    rendered = set(_dates('bond_analysis_*.html'))
    unrendered = [d for d in completed[-10:] if d not in rendered]
    if unrendered:
        print(f'ALERT completed but never rendered: '
              f'{", ".join(map(str, unrendered))}')


def rating_mix(rows):
    """{asset_class: {rating: share}} plus the overall capped share."""
    from collections import Counter
    by_class = {}
    for cls in CLASSES:
        members = [r for r in rows if r.get('asset_class') == cls]
        counts = Counter(r.get('rating') for r in members)
        by_class[cls] = {k: (counts.get(k, 0) / len(members) if members else 0.0)
                         for k in RATINGS}
    capped = sum(1 for r in rows if r.get('_rating_cap_reasons') is not None
                 and len(r['_rating_cap_reasons']) > 0)
    return by_class, (capped / len(rows) if rows else 0.0)


def rating_alerts(mix, capped, prior_mix=None):
    alerts = []
    for cls in ('CORP_IG', 'CORP_HY'):
        if mix[cls]['BUY'] == 0 and mix[cls]['LEAN BUY'] == 0:
            alerts.append(f'{cls} has zero BUY and zero LEAN BUY — a cap or '
                          f'calibration fault, not a market view')
    if mix['CORP_HY']['BUY'] > mix['CORP_IG']['BUY']:
        alerts.append('HY BUY share exceeds IG BUY share')
    if capped > MAX_CAPPED_SHARE:
        alerts.append(f'{capped:.0%} of rows are capped')
    if prior_mix:
        for cls in CLASSES:
            move = 100 * (mix[cls]['BUY'] - prior_mix[cls]['BUY'])
            if abs(move) > BUY_DRIFT_PP:
                alerts.append(f'{cls} BUY share moved {move:+.1f}pp vs prior run')
    return alerts


def _load(stamp):
    import pandas as pd
    return pd.read_parquet(
        os.path.join(OUTPUT_DIR, f'results_{stamp}.parquet')).to_dict('records')


def comparable_prior(run_date):
    """(prior_date, note): the latest earlier run from the SAME model.

    Drift across a model change is the model moving, not the market; the
    09-13 recalibration against a 09-09 snapshot raised exactly that false
    alert. Returns (None, note) when the newest earlier run is a different
    model, so the check is skipped rather than silently compared further back.
    """
    from scripts.model_version import snapshot_version
    current = snapshot_version(_meta_path(run_date))
    prior = [d for d in _dates('results_*.parquet') if d.isoformat() < run_date]
    if not prior:
        return None, None
    last = prior[-1]
    last_version = snapshot_version(_meta_path(last.isoformat()))
    if last_version != current:
        return None, (f'model version changed ({last_version} -> {current}); '
                      f'drift check skipped')
    return last, None


def _meta_path(stamp):
    return os.path.join(OUTPUT_DIR, f'run_meta_{stamp}.json')


def check_ratings(run_date):
    rows = _load(run_date)
    mix, capped = rating_mix(rows)
    prior, note = comparable_prior(run_date)
    prior_mix = rating_mix(_load(prior.isoformat()))[0] if prior else None

    print(f'Rating mix {run_date} (prior: {prior or "none"})')
    if note:
        print(f'NOTE {note}')
    for cls in CLASSES:
        print(f'  {cls:<9} ' + '  '.join(f'{k} {100 * mix[cls][k]:5.1f}%'
                                         for k in RATINGS))
    print(f'  capped {100 * capped:.1f}%')
    for alert in rating_alerts(mix, capped, prior_mix):
        print(f'ALERT {alert}')


def check_nport(today):
    import json
    from data.nport_client import NPORTClient
    client = NPORTClient()
    ingested = sorted(f.split('_')[0] for f in os.listdir(client.cache_dir)
                      if f.endswith('_marks.parquet')) \
        if os.path.isdir(client.cache_dir) else []
    published = client.list_available_quarters()
    newest = ingested[-1] if ingested else None
    print(f'N-PORT ingested: {newest or "none"}; newest published: '
          f'{published[-1] if published else "unknown (index unreadable)"}')
    if published and (newest is None or published[-1] > newest):
        print(f'ALERT N-PORT {published[-1]} is published but not ingested — '
              f'run scripts/ingest_nport.py --quarter {published[-1]}, then '
              f'build_universe.py and the calibrations')

    metas = _dates('run_meta_*.json')
    if metas:
        with open(os.path.join(OUTPUT_DIR, f'run_meta_{metas[-1]}.json'),
                  encoding='utf-8') as fh:
            age = json.load(fh).get('data_vintage_age_days')
        if age is not None and age > NPORT_VINTAGE_WARN_DAYS:
            print(f'ALERT newest N-PORT mark is {age} days old')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('check', choices=['gaps', 'ratings', 'nport'])
    ap.add_argument('--run-date', default=None)
    args = ap.parse_args()
    today = date.today()

    if args.check == 'gaps':
        check_gaps(today)
    elif args.check == 'ratings':
        check_ratings(args.run_date or today.isoformat())
    else:
        check_nport(today)
    return 0


if __name__ == '__main__':
    sys.exit(main())
