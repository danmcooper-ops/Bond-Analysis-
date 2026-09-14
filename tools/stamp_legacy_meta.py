#!/usr/bin/env python3
"""Stamp model_version onto run_meta files written before versioning existed.

    python tools/stamp_legacy_meta.py --through 2026-09-09          # dry run
    python tools/stamp_legacy_meta.py --through 2026-09-09 --apply

Adds ONE key, "model_version": "pre-2026.09.13", to every run_meta dated on or
before --through that has none. Nothing else in the file changes, and a file
that already carries a version is never touched.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.model_version import LEGACY_VERSION, OUTPUT_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--through', required=True, metavar='YYYY-MM-DD')
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    stamped = 0
    for path in sorted(glob.glob(os.path.join(args.output_dir, 'run_meta_*.json'))):
        stamp = os.path.basename(path)[len('run_meta_'):-len('.json')]
        if stamp > args.through:
            continue
        with open(path, encoding='utf-8') as fh:
            meta = json.load(fh)
        if 'model_version' in meta:
            continue
        print(f'  {"stamp" if args.apply else "would stamp"} {os.path.basename(path)}')
        stamped += 1
        if args.apply:
            meta['model_version'] = LEGACY_VERSION
            with open(path + '.tmp', 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, indent=2, default=str)
            os.replace(path + '.tmp', path)
    print(f'\n  {stamped} file(s) {"stamped" if args.apply else "to stamp"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
