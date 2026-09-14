"""Snapshot versioning: every run says which model produced it. Offline."""

import json
import os
import subprocess
import sys
from datetime import date

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_run_stamp_carries_version_revision_and_hashes():
    from scripts.model_version import MODEL_VERSION, run_stamp
    from scripts.param_set import default_params
    stamp = run_stamp(default_params())
    assert stamp['model_version'] == MODEL_VERSION
    assert stamp['git_sha'] is None or len(stamp['git_sha']) >= 40
    assert len(stamp['params_hash']) == 64
    assert set(stamp['calibration_files']) == {'term_structure.json',
                                               'credit_anchors.json'}
    assert 'python' in stamp['libraries']


def test_params_hash_moves_when_a_threshold_moves(monkeypatch):
    from scripts import config
    from scripts.model_version import params_hash
    before = params_hash({'a': 1})
    changed = {**config.RATING_THRESHOLDS_BY_CLASS,
               'CORP_IG': {'buy': 99.0, 'lean': 50.0, 'pass': 30.0}}
    monkeypatch.setattr(config, 'RATING_THRESHOLDS_BY_CLASS', changed)
    assert params_hash({'a': 1}) != before


def _write(dirpath, stamp, version=None):
    import pandas as pd
    meta = {'run_date': stamp}
    if version:
        meta['model_version'] = version
    (dirpath / f'run_meta_{stamp}.json').write_text(json.dumps(meta))
    pd.DataFrame([{'asset_class': 'CORP_IG', 'rating': 'BUY',
                   '_rating_cap_reasons': []}]).to_parquet(
        dirpath / f'results_{stamp}.parquet')


def test_drift_check_skips_across_a_model_change(tmp_path, monkeypatch):
    import scripts.daily_checks as dc
    monkeypatch.setattr(dc, 'OUTPUT_DIR', str(tmp_path))
    _write(tmp_path, '2026-09-09')                          # legacy, unstamped
    _write(tmp_path, '2026-09-13', '2026.09.13')
    _write(tmp_path, '2026-09-14', '2026.09.13')
    prior, note = dc.comparable_prior('2026-09-13')
    assert prior is None and 'pre-2026.09.13 -> 2026.09.13' in note
    prior, note = dc.comparable_prior('2026-09-14')
    assert prior == date(2026, 9, 13) and note is None


def test_legacy_stamper_adds_one_key_and_is_idempotent(tmp_path):
    (tmp_path / 'run_meta_2026-09-09.json').write_text(
        json.dumps({'run_date': '2026-09-09', 'count': 5}))
    (tmp_path / 'run_meta_2026-09-13.json').write_text(
        json.dumps({'run_date': '2026-09-13', 'model_version': '2026.09.13'}))
    cmd = [sys.executable, 'tools/stamp_legacy_meta.py', '--through',
           '2026-09-12', '--output-dir', str(tmp_path), '--apply']
    for _ in range(2):
        assert subprocess.run(cmd, cwd=REPO, capture_output=True).returncode == 0
    old = json.loads((tmp_path / 'run_meta_2026-09-09.json').read_text())
    assert old == {'run_date': '2026-09-09', 'count': 5,
                   'model_version': 'pre-2026.09.13'}
    new = json.loads((tmp_path / 'run_meta_2026-09-13.json').read_text())
    assert new['model_version'] == '2026.09.13'
