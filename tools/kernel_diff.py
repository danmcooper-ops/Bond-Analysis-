#!/usr/bin/env python3
"""Report drift between vendored kernel code and its upstream source.

This repo deliberately COPIES rather than symlinks the generic machinery from
stock-analysis-model. A symlink would mean an edit over there silently changes
bond ratings, and a fresh clone or CI checkout would be broken. The cost of
copying is drift; this script is how we see it.

It never writes anything. It prints a report and exits non-zero if anything
drifted, so it can run in CI as a reminder rather than a gate.

Because the bond repo restructured the kernel (upstream's single
scripts/scoring.py became scripts/scoring_kernel.py plus a separate gates
module), a line-by-line file diff would be pure noise. Instead this compares
FUNCTION BY FUNCTION: it parses both files with `ast`, pulls out the source of
each named function, normalises away comments/docstrings/blank lines, and
reports only functions whose logic actually differs.

Usage:
    python tools/kernel_diff.py [--equity-root ..] [--verbose]
"""

import argparse
import ast
import difflib
import io
import os
import sys
import tokenize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# (vendored path, upstream path, functions to compare, expected-divergence notes)
#
# A function listed in `diverged` is one we have deliberately changed; it is
# still diffed, but reported as EXPECTED rather than DRIFT. Keep the reason
# current — an expected divergence with a stale reason is how a real
# regression hides.
MANIFEST = [
    (
        'scripts/scoring_kernel.py',
        'scripts/scoring.py',
        [
            '_gate_short', '_gate_key', '_gp_key', '_score_key',
            '_gate_applicable', '_cap_rating',
            '_score_linear', '_ranked_percentiles',
            'apply_screening_matrix', 'compute_continuous_scores',
            'rating_from_composite', 'apply_rating_caps',
            'gate_metadata', '_purge_stale_gate_fields', 'score_and_rate',
        ],
        {
            'apply_screening_matrix':
                'takes a ScoringSpec instead of the module-level GATES; '
                'also records _gates_applicable',
            'compute_continuous_scores':
                'PEER_KEY/MIN_PEER_SCORING replace sector; relative_mode '
                "'peer' replaces 'sector'; MC penalty generalised to "
                'spec.uncertainty_field',
            'rating_from_composite':
                'consults per-asset-class thresholds before the base ones',
            'apply_rating_caps':
                'cap function arrives on the spec; passes asset_class through',
            'gate_metadata':
                'display maps and category order arrive on the spec',
            '_purge_stale_gate_fields': 'takes a spec instead of module GATES',
            'score_and_rate': 'threads the spec through',
        },
    ),
    (
        'data/provenance.py', 'data/provenance.py', None, {},
    ),
    (
        'data/snapshot_cache.py', 'data/snapshot_cache.py', None, {},
    ),
    (
        'data/time_slice.py', 'data/time_slice.py', None, {},
    ),
]


def _strip_noise(src):
    """Remove comments, docstrings and blank lines so cosmetic edits are quiet.

    Comments are where the vendor headers and most prose edits live; without
    stripping them every vendored file would report as drifted forever and the
    report would train us to ignore it.
    """
    out = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
        prev_type = tokenize.INDENT
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                continue
            # A STRING in statement position is a docstring.
            if (tok.type == tokenize.STRING
                    and prev_type in (tokenize.INDENT, tokenize.NEWLINE,
                                      tokenize.NL, tokenize.DEDENT)):
                continue
            if tok.type in (tokenize.NL,):
                continue
            out.append(tok.string)
            if tok.type not in (tokenize.NL, tokenize.COMMENT):
                prev_type = tok.type
    except (tokenize.TokenError, IndentationError):
        # Fall back to a crude strip rather than failing the whole report.
        return [ln.strip() for ln in src.splitlines()
                if ln.strip() and not ln.strip().startswith('#')]
    text = ' '.join(out)
    return [ln for ln in (s.strip() for s in text.split('\n')) if ln]


def _functions(path):
    """Return {name: source} for every top-level function in a file."""
    with open(path, encoding='utf-8') as fh:
        src = fh.read()
    tree = ast.parse(src)
    lines = src.splitlines()
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = '\n'.join(lines[node.lineno - 1:node.end_lineno])
    return out, src


def _compare(name, mine, theirs, verbose):
    a, b = _strip_noise(theirs), _strip_noise(mine)
    if a == b:
        return None
    diff = list(difflib.unified_diff(a, b, fromfile=f'upstream/{name}',
                                     tofile=f'vendored/{name}', lineterm='',
                                     n=1 if not verbose else 3))
    return diff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--equity-root', default='..',
                    help='path to the stock-analysis-model checkout (default: ..)')
    ap.add_argument('--verbose', action='store_true',
                    help='show full diffs, including for expected divergences')
    args = ap.parse_args()

    equity_root = os.path.abspath(os.path.join(REPO_ROOT, args.equity_root))
    if not os.path.isdir(equity_root):
        print(f"[error] equity root not found: {equity_root}", file=sys.stderr)
        return 2

    drift = 0
    missing = 0
    for rel_mine, rel_theirs, fn_names, diverged in MANIFEST:
        mine_path = os.path.join(REPO_ROOT, rel_mine)
        theirs_path = os.path.join(equity_root, rel_theirs)
        print(f"\n=== {rel_mine}  <-  {rel_theirs} ===")
        if not os.path.exists(theirs_path):
            print(f"  [skip] upstream missing: {theirs_path}")
            missing += 1
            continue
        if not os.path.exists(mine_path):
            print(f"  [skip] vendored file missing: {mine_path}")
            missing += 1
            continue

        mine_fns, mine_src = _functions(mine_path)
        theirs_fns, theirs_src = _functions(theirs_path)

        if fn_names is None:
            # Whole-file comparison for verbatim vendored modules.
            d = _compare(rel_mine, mine_src, theirs_src, args.verbose)
            if d is None:
                print("  ok — identical (ignoring comments/docstrings)")
            else:
                drift += 1
                print("  DRIFT — file differs from upstream:")
                for line in d[:60]:
                    print(f"    {line}")
                if len(d) > 60:
                    print(f"    ... {len(d) - 60} more diff lines")
            continue

        for name in fn_names:
            if name not in theirs_fns:
                print(f"  [gone]  {name}: no longer exists upstream "
                      f"(was it renamed? refactored away?)")
                drift += 1
                continue
            if name not in mine_fns:
                print(f"  [absent] {name}: present upstream, missing here")
                drift += 1
                continue
            d = _compare(name, mine_fns[name], theirs_fns[name], args.verbose)
            if d is None:
                print(f"  ok      {name}")
            elif name in diverged:
                print(f"  expect  {name}  ({diverged[name]})")
                if args.verbose:
                    for line in d:
                        print(f"            {line}")
            else:
                drift += 1
                print(f"  DRIFT   {name}  — upstream changed, we did not:")
                for line in d[:30]:
                    print(f"            {line}")
                if len(d) > 30:
                    print(f"            ... {len(d) - 30} more diff lines")

        undeclared = [n for n in diverged if n not in fn_names]
        if undeclared:
            print(f"  [warn] divergence notes for uncompared functions: "
                  f"{', '.join(undeclared)}")

    print(f"\n{'=' * 60}")
    if drift:
        print(f"{drift} drifted item(s). Review whether the upstream change "
              f"should be ported here, then update MANIFEST notes.")
    else:
        print("No unexpected drift.")
    if missing:
        print(f"{missing} item(s) skipped (file not found).")
    return 1 if drift else 0


if __name__ == '__main__':
    sys.exit(main())
