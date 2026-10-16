"""fetch_equity_snapshot: point-in-time selection and pruning, offline."""

import gzip
import os
from datetime import date

from scripts import fetch_equity_snapshot as fes


def _tree(*names):
    return {'tree': [{'type': 'blob', 'path': n, 'size': 1} for n in names]
            + [{'type': 'tree', 'path': 'blobs'}]}


def test_published_never_looks_forward(monkeypatch):
    monkeypatch.setattr(fes, 'get_json', lambda *a, **k: _tree(
        'results_2026-09-29.json.gz', 'results_2026-09-30.json.gz',
        'results_2026-10-01.json.gz', 'portfolio_nav.json', 'rating_history.json'))
    found = fes.published(date(2026, 9, 30))
    assert [name for _, name, _ in found] == ['results_2026-09-29.json.gz',
                                               'results_2026-09-30.json.gz']


def test_published_none_when_listing_fails(monkeypatch):
    monkeypatch.setattr(fes, 'get_json', lambda *a, **k: None)
    assert fes.published(date(2026, 9, 30)) is None


def test_prune_keeps_newest_and_selected(tmp_path):
    for day in ('2026-06-30', '2026-09-30', '2026-10-14', '2026-10-15'):
        (tmp_path / f'results_{day}.json.gz').write_bytes(b'x')
    (tmp_path / 'results_2026-01-02.json').write_bytes(b'x')     # stock model's own
    fes.prune(str(tmp_path), 1, str(tmp_path / 'results_2026-09-30.json.gz'))
    assert sorted(os.listdir(tmp_path)) == ['results_2026-01-02.json',
                                            'results_2026-09-30.json.gz',
                                            'results_2026-10-15.json.gz']


def test_gzip_intact_rejects_truncated(tmp_path):
    good = tmp_path / 'good.json.gz'
    with gzip.open(good, 'wt', encoding='utf-8') as fh:
        fh.write('{"results": []}' * 1000)
    bad = tmp_path / 'bad.json.gz'
    bad.write_bytes(good.read_bytes()[:-20])
    assert fes.gzip_intact(str(good))
    assert not fes.gzip_intact(str(bad))
