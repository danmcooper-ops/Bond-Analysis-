"""Old N-PORT ZIP pruning after an ingest. Offline."""

import os

from scripts.ingest_nport import prune_old_zips


def make(cache, *names):
    for name in names:
        (cache / name).write_bytes(b'x')


def listing(cache):
    return sorted(os.listdir(cache))


def test_keeps_newest_zip_and_removes_ingested_older_ones(tmp_path):
    make(tmp_path, '2025q4_nport.zip', '2025q4_marks.parquet',
         '2026q1_nport.zip', '2026q1_marks.parquet',
         '2026q2_nport.zip', '2026q2_marks.parquet')
    assert prune_old_zips(tmp_path) == ['2025q4_nport.zip', '2026q1_nport.zip']
    assert listing(tmp_path) == ['2025q4_marks.parquet', '2026q1_marks.parquet',
                                 '2026q2_marks.parquet', '2026q2_nport.zip']


def test_keeps_older_zip_that_was_never_ingested(tmp_path):
    make(tmp_path, '2026q1_nport.zip', '2026q2_nport.zip', '2026q2_marks.parquet')
    assert prune_old_zips(tmp_path) == []
    assert '2026q1_nport.zip' in listing(tmp_path)


def test_reingesting_an_old_quarter_removes_its_zip(tmp_path):
    make(tmp_path, '2025q3_nport.zip', '2025q3_marks.parquet',
         '2026q2_nport.zip', '2026q2_marks.parquet')
    assert prune_old_zips(tmp_path, ingested='2025q3') == ['2025q3_nport.zip']
    assert '2026q2_nport.zip' in listing(tmp_path)


def test_dropped_newest_zip_does_not_protect_older_ones(tmp_path):
    # --drop-zip already removed 2026q2_nport.zip before pruning runs.
    make(tmp_path, '2026q1_nport.zip', '2026q1_marks.parquet', '2026q2_marks.parquet')
    assert prune_old_zips(tmp_path, ingested='2026q2') == ['2026q1_nport.zip']


def test_ignores_unrelated_files_and_missing_dir(tmp_path):
    make(tmp_path, 'notes_nport.zip', '2026q1_nport.zip.tmp', '2026q1_marks.parquet')
    assert prune_old_zips(tmp_path, ingested='2026q2') == []
    assert prune_old_zips(tmp_path / 'absent') == []
