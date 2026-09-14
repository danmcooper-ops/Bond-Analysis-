"""Which model produced a snapshot.

The snapshot corpus is the long-term point of the daily run, and a corpus that
mixes models without saying so is worse than a short one: a backtest or a
day-over-day drift check reads a model change as a market move. Every
run_meta therefore records the model version, the code revision, and hashes
of the parameters and calibration files that shaped its ratings.

Bump MODEL_VERSION whenever a change alters ratings for unchanged inputs —
gates, caps, scoring, calibration method — and note it below.

    pre-2026.09.13  everything through the 2026-09-09 run. Stamped onto those
                    run_meta files after the fact by tools/stamp_legacy_meta.py.
    2026.09.13      the post-review model: stale marks judged against the
                    data vintage, field-based coupon units, point-in-time
                    fundamentals, MSPD tranches, vintage-masked issuer gates.
    2026.09.14      bills calibrated on their own scale; drift cap on aged
                    prices duration cannot explain; credit cutpoints from the
                    index rating mix with a spread-order guard, issuer-floored
                    anchors, derived CCC anchor, geometric market buckets.
    2026.09.15      Altman Z distress cap no longer applied to financials.
"""

import hashlib
import json
import os
import subprocess

MODEL_VERSION = '2026.09.15'
LEGACY_VERSION = 'pre-2026.09.13'

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')
CALIBRATION_FILES = ('term_structure.json', 'credit_anchors.json')


def git_revision(repo_root=REPO_ROOT):
    """HEAD SHA, suffixed '-dirty' when tracked files have local changes."""
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return None
        status = subprocess.run(['git', 'status', '--porcelain',
                                 '--untracked-files=no'], cwd=repo_root,
                                capture_output=True, text=True, timeout=10)
        dirty = status.returncode == 0 and status.stdout.strip()
        return sha.stdout.strip() + ('-dirty' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def params_hash(params):
    """Hash of everything that turns scores into ratings."""
    from scripts import config
    payload = {
        'params': params,
        'thresholds_by_class': config.RATING_THRESHOLDS_BY_CLASS,
        'credit_cuts': {k: getattr(config, k) for k in dir(config)
                        if k.startswith('CREDIT_CUT_')},
    }
    return _sha256(json.dumps(payload, sort_keys=True, default=str).encode())


def calibration_fingerprint(output_dir=OUTPUT_DIR):
    """{file: {sha256, mtime}} for the fitted files a run reads."""
    out = {}
    for name in CALIBRATION_FILES:
        path = os.path.join(output_dir, name)
        try:
            with open(path, 'rb') as fh:
                out[name] = {'sha256': _sha256(fh.read()),
                             'mtime': int(os.path.getmtime(path))}
        except OSError:
            out[name] = None
    return out


def run_stamp(params):
    """The version block written into every run_meta."""
    from data.provenance import library_versions
    return {
        'model_version': MODEL_VERSION,
        'git_sha': git_revision(),
        'params_hash': params_hash(params),
        'calibration_files': calibration_fingerprint(),
        'libraries': library_versions(),
    }


def snapshot_version(meta_path):
    """model_version recorded in a run_meta, LEGACY_VERSION if absent."""
    try:
        with open(meta_path, encoding='utf-8') as fh:
            return json.load(fh).get('model_version') or LEGACY_VERSION
    except (OSError, ValueError):
        return None
