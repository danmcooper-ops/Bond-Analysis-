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


def test_vintage_banner_appears_only_once_the_dataset_is_old():
    from scripts.report_html import vintage_banner
    assert vintage_banner({'data_vintage_age_days': 120}) == ''
    assert vintage_banner({}) == ''
    html = vintage_banner({'data_vintage_age_days': 171,
                           'data_vintage': '2026-04-30'})
    assert '171 days old' in html and '2026-04-30' in html


# --- calibration hygiene ------------------------------------------------------

def test_thresholds_ignore_nan_scores():
    from scripts.calibrate_thresholds import thresholds_for
    clean = [float(i) for i in range(200)]
    with_nan = clean + [float('nan')] * 5
    assert thresholds_for(with_nan) == thresholds_for(clean)
    cuts = thresholds_for(with_nan)
    assert cuts['buy'] > cuts['lean'] > cuts['pass']


def test_term_fit_counts_each_observation_once():
    from scripts.fit_term_structure import BUCKETS, fit
    label = BUCKETS[3][2]
    buckets = {label: [0.01] * 600, f'mid|{label}': [0.01] * 600}
    assert fit(buckets)['n_observations'] == 600


def test_backtest_anchors_come_from_same_date_peers(monkeypatch):
    from datetime import date

    import scripts.backtest as bt
    pit = bt.PointInTime()
    monkeypatch.setattr(pit, 'term_points', lambda when: None)
    monkeypatch.setattr(pit, 'bucket_oas', lambda when: {})     # stay offline
    monkeypatch.setattr('scripts.fit_term_structure.load_tiered', lambda: None)
    when_a, when_b = date(2025, 4, 30), date(2026, 4, 30)
    rows = ([{'cusip': f'{i:06d}AA1', 'report_date': when_a} for i in range(60)]
            + [{'cusip': f'{i:06d}BB1', 'report_date': when_b} for i in range(60)])
    spreads = {when_a: 0.010, when_b: 0.020}
    monkeypatch.setattr(bt, '_compute_base_signal', lambda row, p, params: {
        'implied_bucket': 'BBB', 'z_spread': spreads[row['report_date']],
        'years_to_maturity': 5.0})
    # 60 distinct CUSIP prefixes per date: enough issuers to anchor.
    pit.register(rows)
    assert pit.anchors(when_a, {})['BBB'] == 0.010
    assert pit.anchors(when_b, {})['BBB'] == 0.020


# --- bills calibrate on their own scale ---------------------------------------

def test_bills_and_coupon_treasuries_calibrate_independently():
    from scripts.calibrate_thresholds import CLASS_TARGETS, _class_key, thresholds_for
    assert _class_key('TREASURY_BILL') == 'treasury_bill'
    assert _class_key('TREASURY') == 'treasury'
    bills = [40.0 + i * 0.1 for i in range(50)]
    cuts = thresholds_for(bills, CLASS_TARGETS['treasury_bill'])
    assert cuts['buy'] > max(bills)                       # no bill can be BUY
    assert cuts['buy'] > cuts['lean'] > cuts['pass']


def test_a_bill_param_override_reaches_bill_rows():
    from scripts.param_set import merge_params
    from scripts.scoring_kernel import rating_from_composite
    params = merge_params({'rating_threshold_buy_treasury_bill': 10.0,
                           'rating_threshold_lean_treasury_bill': 5.0,
                           'rating_threshold_pass_treasury_bill': 1.0})
    assert rating_from_composite(12.0, params, asset_class='TREASURY_BILL') == 'BUY'
    assert rating_from_composite(12.0, params, asset_class='TREASURY') != 'BUY'


def test_tier_comparison_flags_a_long_end_gap():
    from scripts.fit_term_structure import compare_tier_assignment
    labels = ['0-3y', '3-5y', '5-7y', '7-10y', '10-15y', '15-20y']
    buckets = {}
    for i, label in enumerate(labels):
        buckets[f'wide|{label}'] = [0.020] * 60                 # flat by spread
        buckets[f'bucket:wide|{label}'] = [0.020 * (1 + 0.05 * i)] * 60
    rows, worst = compare_tier_assignment(buckets)
    assert worst > 0.10                                          # 15-20y: 1.20x vs 1.00x
    buckets = {k: ([0.020] * 60) for k in buckets}
    assert compare_tier_assignment(buckets)[1] == 0.0


def test_fit_prefers_bucket_assigned_tiers_per_tier():
    from scripts.fit_term_structure import BUCKETS, fit
    buckets = {}
    for _lo, _hi, label, _mid in BUCKETS[:6]:
        buckets[label] = [0.01] * 600
        buckets[f'tight|{label}'] = [0.006] * 60
        buckets[f'wide|{label}'] = [0.02] * 60
        buckets[f'bucket:tight|{label}'] = [0.006] * 60   # tight fits by bucket
    tiers = fit(buckets)['by_tier']
    assert tiers['tight']['assigned_by'] == 'implied_bucket'
    assert tiers['wide']['assigned_by'] == 'observed_spread'
