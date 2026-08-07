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

import json

from models.credit import (CREDIT_BUCKETS, CUTPOINT_PARAMS, calibrate_cutpoints,
                           fit_bucket_anchors)
from scripts.fit_term_structure import load_fitted, load_tiered

ANCHOR_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output', 'credit_anchors.json')

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


def fit_anchors(rows):
    """Median spread of each bucket's own members, de-termed to 5 years."""
    return fit_bucket_anchors(rows, term_points=load_fitted(),
                              term_by_bucket=load_tiered())


def report_anchors(anchors):
    meta = anchors.get('_meta', {})
    print(f"\n  FAIR-SPREAD ANCHORS — the model's own buckets, not an index")
    print(f"    {'bucket':<7}{'n':>7}{'anchor':>10}{'index OAS':>12}{'gap':>9}")
    for bucket in ORDER:
        info = meta.get(bucket)
        if not info:
            continue
        if not info.get('used'):
            reason = info.get('dropped', f"only {info['n']} bonds")
            print(f"    {bucket:<7}{info['n']:>7}{'dropped':>10}"
                  f"{INDEX_OAS_BP[bucket]:>10}bp   ({reason})")
            continue
        value = info['anchor'] * 10000
        gap = value - INDEX_OAS_BP[bucket]
        print(f"    {bucket:<7}{info['n']:>7}{value:>8.0f}bp"
              f"{INDEX_OAS_BP[bucket]:>10}bp{gap:>+8.0f}")
    print(f"\n    Investment grade anchors sit WIDER than the index and high")
    print(f"    yield far TIGHTER, because the model's buckets are not the")
    print(f"    index's constituents. Pricing one population off the other is")
    print(f"    what gave Meta's 39-year bond a 41bp fair spread against a")
    print(f"    143bp market spread.")


def write_anchors(anchors):
    os.makedirs(os.path.dirname(ANCHOR_PATH), exist_ok=True)
    payload = {k: v for k, v in anchors.items() if not k.startswith('_')}
    with open(ANCHOR_PATH, 'w', encoding='utf-8') as fh:
        json.dump({'anchors': payload, 'meta': anchors.get('_meta', {})},
                  fh, indent=2)
    print(f"\n  Wrote {len(payload)} bucket anchors to "
          f"{os.path.basename(ANCHOR_PATH)}")


def load_anchors(path=None):
    """{bucket: anchor_spread}, or None. Used by the pipeline."""
    path = path or ANCHOR_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    anchors = payload.get('anchors') or {}
    return {k: float(v) for k, v in anchors.items()} or None


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

    # Anchors are fitted with the NEW cutpoints in force, since the bucket a
    # bond lands in decides which anchor it contributes to.
    from models.credit import bucket_from_score
    rebucketed = []
    for row in rows:
        score = row.get('issuer_credit_score')
        if score is None:
            continue
        rebucketed.append({**row,
                           'implied_bucket': bucket_from_score(score, cuts)})
    anchors = fit_anchors(rebucketed)
    report_anchors(anchors)

    if args.apply:
        apply(cuts)
        write_anchors(anchors)
    else:
        print("\n  Re-run with --apply to write these into scripts/config.py.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
