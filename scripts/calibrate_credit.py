#!/usr/bin/env python3
"""Align the credit scorecard's bucket labels with the market's.

    python scripts/calibrate_credit.py output/results_2026-08-06.parquet
    python scripts/calibrate_credit.py output/results_2026-08-06.parquet --apply

The scorecard RANKS credit risk correctly — median observed spread rises
monotonically across its buckets and the score-to-spread rank correlation is
-0.43. What was wrong was where the cutpoints sat: the seed values assigned
51% of the universe to high yield when the market prices 23% there, and put
258 bonds in CCC where the market saw three.

That is not cosmetic. fair_spread multiplies by the bucket's index OAS, so a
mislabelled BBB is handed a CCC's 1023bp fair spread and reads as absurdly
rich. Every valuation downstream inherits the label.
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.credit import CREDIT_BUCKETS, CUTPOINT_PARAMS, calibrate_cutpoints

ORDER = list(CREDIT_BUCKETS)
INDEX_OAS_BP = {'AAA': 38, 'AA': 56, 'A': 64, 'BBB': 96,
                'BB': 165, 'B': 290, 'CCC': 1023}


def load(path):
    if path.endswith('.parquet'):
        import pandas as pd
        frame = pd.read_parquet(path)
        return frame.astype(object).where(pd.notna(frame), None).to_dict('records')
    import json
    with open(path, encoding='utf-8') as fh:
        payload = json.load(fh)
    return payload.get('results', payload)


def _mix(rows, field):
    counts = Counter(r.get(field) for r in rows if r.get(field))
    total = sum(counts.values()) or 1
    return counts, total


def report(rows, cuts):
    scored = [r for r in rows if r.get('issuer_credit_score') is not None
              and r.get('market_bucket')]
    before, total = _mix(scored, 'implied_bucket')
    market, _ = _mix(scored, 'market_bucket')

    # Re-bucket every score under the new cutpoints.
    from models.credit import bucket_from_score
    after = Counter(bucket_from_score(r['issuer_credit_score'], cuts)
                    for r in scored)

    print(f"\n{'=' * 72}")
    print(f"  CREDIT CUTPOINT CALIBRATION  —  {len(scored):,} scored bonds")
    print(f"{'=' * 72}")
    print(f"\n  {'bucket':<7}{'index OAS':>11}{'before':>9}{'market':>9}"
          f"{'after':>9}   median observed spread")
    for bucket in ORDER:
        med = None
        subset = [r['z_spread'] for r in scored
                  if r.get('implied_bucket') == bucket and r.get('z_spread')]
        if subset:
            med = sorted(subset)[len(subset) // 2]
        med_txt = f'{med * 10000:.0f}bp' if med else ''
        print(f"  {bucket:<7}{INDEX_OAS_BP[bucket]:>9}bp"
              f"{before.get(bucket, 0):>9}{market.get(bucket, 0):>9}"
              f"{after.get(bucket, 0):>9}{med_txt:>15}")

    hy = ('BB', 'B', 'CCC')
    print(f"\n  high yield share   before {sum(before.get(b, 0) for b in hy) / total:>6.0%}"
          f"   market {sum(market.get(b, 0) for b in hy) / total:>6.0%}"
          f"   after {sum(after.get(b, 0) for b in hy) / total:>6.0%}")

    print(f"\n  CUTPOINTS")
    from scripts.config import (CREDIT_CUT_A, CREDIT_CUT_AA, CREDIT_CUT_AAA,
                                CREDIT_CUT_B, CREDIT_CUT_BB, CREDIT_CUT_BBB)
    seeds = dict(zip(CUTPOINT_PARAMS, [CREDIT_CUT_AAA, CREDIT_CUT_AA,
                                       CREDIT_CUT_A, CREDIT_CUT_BBB,
                                       CREDIT_CUT_BB, CREDIT_CUT_B]))
    for param in CUTPOINT_PARAMS:
        new = cuts.get(param)
        print(f"    {param:<18}{seeds[param]:>6}  ->  "
              f"{(f'{new}' if new is not None else 'unchanged'):>8}")


def apply(cuts):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    with open(path, encoding='utf-8') as fh:
        source = fh.read()
    names = {'credit_cut_aaa': 'CREDIT_CUT_AAA', 'credit_cut_aa': 'CREDIT_CUT_AA',
             'credit_cut_a': 'CREDIT_CUT_A', 'credit_cut_bbb': 'CREDIT_CUT_BBB',
             'credit_cut_bb': 'CREDIT_CUT_BB', 'credit_cut_b': 'CREDIT_CUT_B'}
    import re
    for param, constant in names.items():
        if param not in cuts:
            continue
        source = re.sub(rf'^{constant} = .*$', f'{constant} = {cuts[param]}',
                        source, flags=re.M)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(source)
    print(f"\n  Wrote {len(cuts)} cutpoints into scripts/config.py")
    print("  Market-anchored: the labels now mean what the market means by")
    print("  them. Whether the RANKING predicts returns is a separate question")
    print("  the backtest asks.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('snapshot')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    rows = load(args.snapshot)
    cuts = calibrate_cutpoints(rows)
    if not cuts:
        raise SystemExit('[fatal] not enough scored rows to calibrate')
    report(rows, cuts)
    if args.apply:
        apply(cuts)
    else:
        print("\n  Re-run with --apply to write these into scripts/config.py.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
