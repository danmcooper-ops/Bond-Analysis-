#!/usr/bin/env python3
"""Quantile-match rating thresholds per asset class against a run's own scores.

    python scripts/calibrate_thresholds.py output/results_2026-08-06.parquet
    python scripts/calibrate_thresholds.py output/results_2026-08-06.parquet --apply

WHAT THIS IS, AND WHAT IT IS NOT
---------------------------------
This is a DISTRIBUTIONAL calibration: it finds the cutpoints that make each
asset class produce a sensible mix of ratings. It says nothing about whether
the ranking predicts returns — only a backtest can say that, and that is M8.

It exists because a composite is not comparable across asset classes. A
Treasury scores 7 gates across 4 categories; a corporate will score up to 26
across 5. Worse, two of a Treasury's four scoring categories are near-constant
(Analyzability is 100 for every Treasury by construction, and issue size is
large for almost all of them), so its composite sits high and compressed. Run
against thresholds quantile-matched on an equity universe, 65% of Treasuries
came out BUY — which is not a screen, it is a list.

The equity model's own 57/39/25 began life exactly this way, quantile-matched
against a snapshot, and were only later checked against forward returns. Same
sequence here: make the labels mean something now, validate at M8, and keep
saying which of the two has happened.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.param_set import ASSET_CLASSES

# Target mix. BUY is deliberately scarce: the point of the exercise is a short
# actionable list, not a ranking with a generous top band.
DEFAULT_TARGET = {'BUY': 0.03, 'LEAN BUY': 0.22, 'HOLD': 0.50, 'PASS': 0.25}

MIN_ROWS = 30


def _quantile(sorted_values, q):
    """Value at quantile q of an ascending list, linearly interpolated."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def thresholds_for(scores, target=None):
    """Cutpoints reproducing `target` on this score distribution.

    Returns None when there are too few rows for a quantile to mean anything —
    better no threshold, and a fall back to the base ones, than a cutpoint
    fitted to nine bonds.
    """
    target = target or DEFAULT_TARGET
    values = sorted(s for s in scores if s is not None)
    if len(values) < MIN_ROWS:
        return None

    buy = target['BUY']
    lean = buy + target['LEAN BUY']
    hold = lean + target['HOLD']
    return {
        'buy': round(_quantile(values, 1.0 - buy), 1),
        'lean': round(_quantile(values, 1.0 - lean), 1),
        'pass': round(_quantile(values, 1.0 - hold), 1),
    }


def _class_key(asset_class):
    """Map an asset class onto its param-set suffix."""
    cls = (asset_class or '').lower()
    if cls.startswith('treasury'):
        return 'treasury'
    if cls == 'agency':
        return 'agency'
    if cls in ('corp_ig', 'corp_hy'):
        return cls
    return None


def load_rows(path):
    if path.endswith('.parquet'):
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    with open(path, encoding='utf-8') as fh:
        payload = json.load(fh)
    return payload.get('results', payload)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('snapshot')
    ap.add_argument('--apply', action='store_true',
                    help='write the thresholds into scripts/config.py')
    ap.add_argument('--buy', type=float, default=DEFAULT_TARGET['BUY'])
    ap.add_argument('--lean', type=float, default=DEFAULT_TARGET['LEAN BUY'])
    ap.add_argument('--hold', type=float, default=DEFAULT_TARGET['HOLD'])
    args = ap.parse_args()

    target = {'BUY': args.buy, 'LEAN BUY': args.lean, 'HOLD': args.hold,
              'PASS': 1.0 - args.buy - args.lean - args.hold}
    if target['PASS'] < 0:
        raise SystemExit('[fatal] BUY + LEAN + HOLD exceeds 100%')

    rows = load_rows(args.snapshot)
    by_class = {}
    for row in rows:
        key = _class_key(row.get('asset_class'))
        if key:
            by_class.setdefault(key, []).append(row.get('_composite_score'))

    print(f"\n  Target mix: BUY {target['BUY']:.0%} / LEAN {target['LEAN BUY']:.0%}"
          f" / HOLD {target['HOLD']:.0%} / PASS {target['PASS']:.0%}")
    print(f"  Source: {os.path.basename(args.snapshot)}  ({len(rows)} rows)\n")

    results = {}
    for key in ASSET_CLASSES:
        scores = by_class.get(key, [])
        if not scores:
            continue
        cuts = thresholds_for(scores, target)
        if cuts is None:
            print(f"  {key:<10} {len(scores):>5} rows — too few to calibrate "
                  f"(need {MIN_ROWS}); falls back to the base thresholds")
            continue
        results[key] = cuts
        valid = [s for s in scores if s is not None]
        counts = Counter(
            'BUY' if s >= cuts['buy'] else
            'LEAN BUY' if s >= cuts['lean'] else
            'HOLD' if s >= cuts['pass'] else 'PASS'
            for s in valid)
        n = len(valid)
        mix = '  '.join(f"{lab}={counts.get(lab, 0)} "
                        f"({100.0 * counts.get(lab, 0) / n:.0f}%)"
                        for lab in ('BUY', 'LEAN BUY', 'HOLD', 'PASS'))
        print(f"  {key:<10} {n:>5} rows   "
              f"buy>={cuts['buy']}  lean>={cuts['lean']}  pass>={cuts['pass']}")
        print(f"  {'':<10} {'':>5}       {mix}")
        print(f"  {'':<10} {'':>5}       score range "
              f"{min(valid):.1f}-{max(valid):.1f}, median "
              f"{sorted(valid)[len(valid) // 2]:.1f}\n")

    if not results:
        raise SystemExit('[fatal] nothing calibrated')

    if args.apply:
        _apply(results)
    else:
        print("  Re-run with --apply to write these into scripts/config.py.")
        print("  These are DISTRIBUTIONAL cutpoints: they make the labels "
              "sensible, and say\n  nothing yet about whether the ranking "
              "predicts returns. That is M8.\n")
    return 0


def _apply(results):
    """Rewrite RATING_THRESHOLDS_BY_CLASS in config.py, in place."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config.py')
    with open(config_path, encoding='utf-8') as fh:
        source = fh.read()

    body = ['RATING_THRESHOLDS_BY_CLASS = {']
    for key in ASSET_CLASSES:
        cuts = results.get(key)
        upper = key.upper()
        if cuts:
            body.append(f"    '{upper}': {{'buy': {cuts['buy']}, "
                        f"'lean': {cuts['lean']}, 'pass': {cuts['pass']}}},")
        else:
            body.append(f"    '{upper}': {{}},")
    # TREASURY_BILL shares the Treasury scale: same gates, same construction.
    if 'treasury' in results:
        cuts = results['treasury']
        body.append(f"    'TREASURY_BILL': {{'buy': {cuts['buy']}, "
                    f"'lean': {cuts['lean']}, 'pass': {cuts['pass']}}},")
    body.append('}')
    replacement = '\n'.join(body)

    start = source.index('RATING_THRESHOLDS_BY_CLASS = {')
    end = source.index('\n}', start) + 2
    updated = source[:start] + replacement + source[end:]

    with open(config_path, 'w', encoding='utf-8') as fh:
        fh.write(updated)
    print(f"  Wrote {len(results)} class thresholds into scripts/config.py")
    print("  Provisional: quantile-matched, not yet validated against "
          "forward returns.\n")


if __name__ == '__main__':
    sys.exit(main())
