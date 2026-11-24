"""Price history: FedInvest parsing and caching, marks and estimates assembly. Offline."""

from datetime import date, timedelta

import pandas as pd

from data.fedinvest_client import FedInvestClient, parse_csv
from scripts.build_price_history import estimates_frame, marks_frame

SAMPLE = (
    '912797VY0,MARKET BASED BILL,0.0,11/24/2026,,0.000000,99.956778,99.989194\n'
    '91282CLY5,MARKET BASED NOTE,0.0425,11/30/2026,,0.000000,100.000000,100.000000\n'
    '912828V49,TIPS,0.00375,01/15/2027,,0.000000,99.531250,99.531250\n'
    'garbage line\n'
    '912810XX0,MARKET BASED BOND,0.045,not-a-date,,1,1,1\n'
)


def test_parse_maps_types_and_turns_zero_into_none():
    rows = parse_csv(SAMPLE, date(2026, 11, 20))
    assert [r['cusip'] for r in rows] == ['912797VY0', '91282CLY5', '912828V49']
    bill, note, tips = rows
    assert bill['security_type'] == 'BILL' and bill['buy'] is None
    assert bill['end_of_day'] == 99.989194
    assert note['coupon'] == 0.0425
    assert note['maturity_date'] == date(2026, 11, 30)
    assert tips['security_type'] == 'TIPS'
    assert all(r['price_date'] == date(2026, 11, 20) for r in rows)


def test_parse_of_empty_or_none_is_empty():
    assert parse_csv('', date(2026, 11, 22)) == []
    assert parse_csv(None, date(2026, 11, 22)) == []


class StubClient(FedInvestClient):
    def __init__(self, cache_dir, answers):
        super().__init__(cache_dir=str(cache_dir))
        self.answers, self.calls = answers, []

    def _post(self, day):
        self.calls.append(day)
        return self.answers.get(day, '')


def test_old_empty_day_is_cached_but_recent_empty_day_is_not(tmp_path):
    old = date(2020, 1, 1)
    recent = date.today() - timedelta(days=1)
    client = StubClient(tmp_path, {})
    assert client.raw(old) == '' and client.is_cached(old)
    assert client.raw(recent) == '' and not client.is_cached(recent)


def test_failed_fetch_is_not_cached(tmp_path):
    client = StubClient(tmp_path, {})
    client._post = lambda day: None
    assert client.raw(date(2020, 1, 2)) is None
    assert not client.is_cached(date(2020, 1, 2))


def test_backfill_skips_weekends_and_cached_days(tmp_path):
    monday = date(2020, 1, 6)
    client = StubClient(tmp_path, {monday: SAMPLE})
    fetched, failed = client.backfill(date(2020, 1, 4), date(2020, 1, 8))
    assert client.calls == [monday, date(2020, 1, 7), date(2020, 1, 8)]
    assert (fetched, failed) == (3, [])
    client.calls = []
    client.backfill(date(2020, 1, 4), date(2020, 1, 8))
    assert client.calls == []
    assert client.cached_days() == [monday, date(2020, 1, 7), date(2020, 1, 8)]


def _marks(rows):
    return pd.DataFrame([{
        'cusip': c, 'report_date': pd.Timestamp(d), 'clean_price_marked': p,
        'n_funds': n, 'issuer_type': 'CORP'} for c, d, p, n in rows])


def test_marks_collapse_month_end_variants_to_the_widest_consensus(tmp_path):
    _marks([('A', '2025-05-30', 99.0, 2),
            ('A', '2025-05-31', 98.5, 30),
            ('A', '2025-06-30', 98.0, 25)]).to_parquet(
        tmp_path / '2025q3_marks.parquet')
    _marks([('A', '2025-06-30', 97.0, 10),
            ('B', '2025-06-30', None, 5)]).to_parquet(
        tmp_path / '2025q4_marks.parquet')
    panel = marks_frame(str(tmp_path))
    assert list(panel['cusip']) == ['A', 'A']          # B had no price
    may, june = panel.to_dict('records')
    assert (may['clean_price_marked'], may['n_funds']) == (98.5, 30)
    assert may['report_date'] == pd.Timestamp('2025-05-31')
    assert (june['clean_price_marked'], june['source_quarter']) == (98.0, '2025q3')


def test_marks_frame_without_files_is_none(tmp_path):
    assert marks_frame(str(tmp_path)) is None
    assert marks_frame(str(tmp_path / 'missing')) is None


def test_estimates_keep_only_priced_rows_and_tolerate_missing_columns(tmp_path):
    pd.DataFrame({'cusip': ['A', 'B'], 'clean_price_est': [99.1, None],
                  'asset_class': ['IG', 'HY']}).to_parquet(
        tmp_path / 'results_2026-08-06.parquet')
    pd.DataFrame({'cusip': ['A'], 'clean_price_est': [99.3],
                  'model_version': ['2026.11.10']}).to_parquet(
        tmp_path / 'results_2026-11-20.parquet')
    frame = estimates_frame(sorted(str(p) for p in tmp_path.glob('*.parquet')))
    assert list(frame['clean_price_est']) == [99.1, 99.3]
    assert list(frame['run_date']) == [pd.Timestamp('2026-08-06'),
                                       pd.Timestamp('2026-11-20')]
