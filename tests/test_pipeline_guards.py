"""Guards that stop a failed or partial run from publishing. Offline."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args):
    return subprocess.run([sys.executable, *args], cwd=REPO,
                          capture_output=True, text=True, timeout=60)


def test_report_refuses_a_run_date_with_no_snapshot():
    result = _run('scripts/report_html.py', '--run-date', '2099-01-01')
    assert result.returncode != 0
    assert 'did not complete' in result.stderr


def test_publish_refuses_a_run_date_with_no_report():
    result = _run('scripts/publish.py', '--run-date', '2099-01-01', '--dry-run')
    assert result.returncode != 0
    assert 'refusing to publish an older report' in result.stderr


def test_report_refuses_a_snapshot_without_run_meta(tmp_path):
    import pandas as pd
    snap = tmp_path / 'results_2099-01-02.parquet'
    pd.DataFrame([{'cusip': 'X', 'rating': 'HOLD'}]).to_parquet(snap)
    result = _run('scripts/report_html.py', str(snap), '-o',
                  str(tmp_path / 'out.html'))
    assert result.returncode != 0
    assert 'run_meta_2099-01-02.json' in result.stderr
    assert not (tmp_path / 'out.html').exists()


# --- daily health checks ---------------------------------------------------

def test_missed_weekdays_reports_the_silent_gaps():
    from datetime import date

    from scripts.daily_checks import missed_weekdays
    completed = [date(2026, 9, 8), date(2026, 9, 9)]
    # 09-10 Thu and 09-11 Fri never ran; 09-12/13 are a weekend.
    assert missed_weekdays(completed, date(2026, 9, 14)) == \
        [date(2026, 9, 10), date(2026, 9, 11)]
    assert missed_weekdays(completed, date(2026, 9, 10)) == []


def _rows(cls, ratings, capped=False):
    return [{'asset_class': cls, 'rating': r,
             '_rating_cap_reasons': ['stale mark'] if capped else []}
            for r in ratings]


def test_zero_corporate_buys_raise_an_alert_the_drift_check_cannot():
    from scripts.daily_checks import rating_alerts, rating_mix
    rows = (_rows('CORP_IG', ['HOLD'] * 9 + ['PASS'], capped=True)
            + _rows('CORP_HY', ['HOLD'] * 10, capped=True)
            + _rows('TREASURY', ['BUY', 'HOLD']))
    mix, capped = rating_mix(rows)
    # Identical to yesterday, so a pure drift check stays quiet.
    alerts = rating_alerts(mix, capped, prior_mix=mix)
    assert any('CORP_IG has zero BUY' in a for a in alerts)
    assert any('CORP_HY has zero BUY' in a for a in alerts)
    assert any('capped' in a for a in alerts)


def test_healthy_mix_raises_nothing():
    from scripts.daily_checks import rating_alerts, rating_mix
    rows = (_rows('CORP_IG', ['BUY', 'LEAN BUY'] + ['HOLD'] * 8)
            + _rows('CORP_HY', ['LEAN BUY'] + ['HOLD'] * 9)
            + _rows('TREASURY', ['BUY'] + ['HOLD'] * 9))
    mix, capped = rating_mix(rows)
    assert rating_alerts(mix, capped, prior_mix=mix) == []
