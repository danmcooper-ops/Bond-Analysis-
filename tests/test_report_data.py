"""The report as data (bond_report_<date>.json): what another report draws
this page from. Offline."""

import json
import os
from datetime import date

from scripts.report_html import (REPORT_DATA_VERSION, report_data, render,
                                 write_report_data)

META = {'run_date': '2099-01-02', 'curve_date': '2099-01-02',
        'par_curve': {'3M': 0.042, '10Y': 0.05}, 'data_vintage': '2098-07-01',
        'data_vintage_age_days': 185}
TERM = {'by_tier': {'mid': {'points': [[2.0, 0.7], [10.0, 1.1]]}}}


def _rows():
    return [
        {'cusip': 'A', 'issuer_name': 'ALPHA', 'asset_class': 'CORP_IG',
         'rating': 'BUY', 'implied_bucket': 'A', 'mark_age_days': 180,
         'maturity_date': date(2030, 1, 1), 'z_spread': float('nan'),
         '_rating_cap_reasons': [], '_score_valuation': 60.0},
        {'cusip': 'B', 'issuer_name': 'BETA', 'asset_class': 'TREASURY',
         'rating': 'HOLD', 'mark_age_days': 170,
         '_rating_cap_reasons': ['stale mark (210d old)'],
         '_score_valuation': 40.0},
    ]


def test_report_data_carries_the_page():
    d = report_data(_rows(), META, TERM)
    assert d['version'] == REPORT_DATA_VERSION
    assert d['run_date'] == '2099-01-02' and d['total'] == 2
    assert d['capped_pct'] == 50 and d['mark_age'] == 180
    assert d['vintage'] == {'date': '2098-07-01', 'age_days': 185}
    assert [c['k'] for c in d['cards']][:2] == ['Instruments', 'BUY']
    assert d['caveats'][2]['lead'] == '50% of rows carry a rating cap'
    assert d['ratings'] == [['BUY', 1], ['HOLD', 1]]
    assert d['curve'] == [[0.25, 0.042], [10.0, 0.05]]
    assert [s['name'] for s in d['term']] == ['mid', 'FRED slices']
    assert d['cap_reasons'] == [['stale mark', 1]]
    assert set(d['tips']) == {'ratings', 'buckets', 'curve', 'term', 'caps',
                              'categories'}
    cols = [c['k'] for c in d['columns']]
    row = dict(zip(cols, d['rows'][0]))
    assert row['maturity_date'] == '2030-01-01' and row['z_spread'] is None
    assert dict(zip(cols, d['rows'][1]))['_caps'] == 'stale mark (210d old)'


def test_no_vintage_warning_while_marks_are_fresh():
    assert report_data(_rows(), dict(META, data_vintage_age_days=90), TERM)['vintage'] is None


def test_render_writes_the_data_beside_the_page(tmp_path):
    render(_rows(), META, TERM, str(tmp_path / 'page.html'))
    data = json.loads((tmp_path / 'bond_report_2099-01-02.json').read_text(encoding='utf-8'))
    assert data['total'] == 2
    html = (tmp_path / 'page.html').read_text(encoding='utf-8')
    # The page and the data are drawn from the same caveats and tiles.
    assert data['caveats'][0]['text'] in html
    assert '{' + 'tip_' not in html


def test_write_is_strict_json(tmp_path):
    path = write_report_data({'a': [1, None]}, str(tmp_path / 'x.json'))
    assert json.loads(open(path, encoding='utf-8').read()) == {'a': [1, None]}
    assert not os.path.exists(path + '.tmp')
