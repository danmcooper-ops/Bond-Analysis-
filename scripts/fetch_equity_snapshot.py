#!/usr/bin/env python3
"""Fetch an equity snapshot into $EQUITY_SNAPSHOT_DIR for a universe build.

    python scripts/fetch_equity_snapshot.py --as-of 2026-09-30   # quarter's last mark
    python scripts/fetch_equity_snapshot.py                      # newest <= today

The credit gates come from the stock model's `results_<date>.json[.gz]`, read
by `build_universe.py` (data/issuer_fundamentals.py) AS OF the quarter's newest
N-PORT mark and baked into `output/universe_<quarter>.parquet`; the daily
`analyze_bonds.py` never opens a snapshot. So run this before building a new
quarter, with `--as-of` at that quarter's newest mark date.

That pipeline now runs in the cloud and archives each night to the public
`data/snapshots` branch of stock-analysis-model, so the local copy the bond
model used to share stopped advancing on 2026-09-14 and nothing said so: a
build for 2026q3 marks would have read fundamentals two weeks short of its
mark date. This pulls the archived `.gz` straight from GitHub (no credentials,
~13-18 MB), which is also what lets the bond model live on a machine with no
stock-model checkout at all.

The `.gz` form stores `edgar_history` as `{"$blob": ...}` references; nothing
here reads that key, so the `blobs/` directory is not needed.

Exits 0 when the newest snapshot is present (fetched or already there), 1 when
it could not be fetched. An "ALERT" line means the newest snapshot available is
stale, i.e. the cloud stock pipeline has stopped publishing.
"""

import argparse
import glob
import gzip
import os
import re
import sys
import zlib
from datetime import date, datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.http import download_atomic, get_json  # noqa: E402

REPO = 'danmcooper-ops/stock-analysis-model'
BRANCH = 'data/snapshots'
TREE_URL = f'https://api.github.com/repos/{REPO}/git/trees/{BRANCH}'
RAW_URL = f'https://raw.githubusercontent.com/{REPO}/{BRANCH}/{{name}}'
NAME_RE = re.compile(r'^results_(\d{4}-\d{2}-\d{2})\.json\.gz$')
STALE_DAYS = 5


def published(as_of):
    """[(date, name, size)] on the branch at or before as_of, oldest first."""
    tree = get_json(TREE_URL, headers={'Accept': 'application/vnd.github+json'})
    if not tree or 'tree' not in tree:
        return None
    out = []
    for entry in tree['tree']:
        m = NAME_RE.match(entry.get('path', ''))
        if entry.get('type') != 'blob' or not m:
            continue
        day = datetime.strptime(m.group(1), '%Y-%m-%d').date()
        if day <= as_of:
            out.append((day, entry['path'], entry.get('size')))
    return sorted(out)


def gzip_intact(path):
    """True when the whole stream decompresses (catches a truncated file)."""
    try:
        with gzip.open(path, 'rb') as fh:
            while fh.read(1 << 20):
                pass
        return True
    except (OSError, EOFError, zlib.error):
        return False


def prune(snapshot_dir, keep, selected):
    """Drop all but the newest `keep` .json.gz snapshots, never `selected` (a
    back-dated --as-of must not delete what it just fetched). Plain .json is
    left alone: on a machine that shares the stock model's output those are
    its."""
    paths = sorted(glob.glob(os.path.join(snapshot_dir, 'results_*.json.gz')),
                   key=os.path.basename)
    for path in paths[:-keep] if keep > 0 else []:
        if os.path.basename(path) == os.path.basename(selected):
            continue
        os.remove(path)
        print(f'Pruned {os.path.basename(path)}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--as-of', type=date.fromisoformat, default=date.today(),
                    help='newest snapshot at or before this date (default today)')
    ap.add_argument('--dir', default=None,
                    help='destination (default $EQUITY_SNAPSHOT_DIR)')
    ap.add_argument('--keep', type=int, default=0,
                    help='keep only the newest N .json.gz files (0 = keep all)')
    args = ap.parse_args(argv)

    snapshot_dir = args.dir or os.environ.get('EQUITY_SNAPSHOT_DIR')
    if not snapshot_dir:
        print('ALERT EQUITY_SNAPSHOT_DIR is not set (.env) — credit gates '
              'will be off')
        return 1

    found = published(args.as_of)
    if not found:
        print(f'ALERT could not list {BRANCH} on {REPO}')
        return 1
    day, name, size = found[-1]

    dest = os.path.join(snapshot_dir, name)
    plain = dest[:-len('.gz')]
    if os.path.exists(plain) or (os.path.exists(dest)
                                 and os.path.getsize(dest) == size):
        print(f'Equity snapshot {day} already present')
    else:
        if download_atomic(RAW_URL.format(name=name), dest) is None:
            print(f'ALERT download of {name} failed')
            return 1
        if not gzip_intact(dest):
            os.remove(dest)
            print(f'ALERT {name} arrived corrupt; removed')
            return 1
        print(f'Fetched {name} ({os.path.getsize(dest) / 1e6:.1f} MB)')

    age = (args.as_of - day).days
    if age > STALE_DAYS:
        print(f'ALERT newest equity snapshot is {day} ({age} days old) — '
              f'the cloud stock pipeline may have stopped publishing')
    if args.keep:
        prune(snapshot_dir, args.keep, dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
