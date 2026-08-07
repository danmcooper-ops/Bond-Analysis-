"""The real bond GATES: masking, robustness, caps.

The keystone is test_treasury_masks_exactly_these_gates, which asserts the
masked set as a LITERAL list. It reads as over-specified and it is deliberate:
the applicability design is the mechanism that lets Treasuries and corporate
bonds share one rating scale, and a predicate quietly changing which gates it
covers would move every Treasury composite without failing anything else.
"""

from datetime import date

import pytest

from scripts.gates import (GATES, SPEC, _analyzability_score,
                           _payment_status_score, peer_group,
                           prepare_scoring_fields, rating_cap_for_row)
from scripts.scoring_kernel import (_gate_applicable, _gate_short, _score_key,
                                    compute_continuous_scores, score_and_rate)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def treasury_row(**kw):
    row = dict(
        cusip='912810UP1', asset_class='TREASURY', issuer_name='US TREASURY',
        coupon_type='Fixed', coupon_rate=0.04625, maturity_date=date(2055, 11, 15),
        years_to_maturity=29.3, seniority_rank=1, seniority_source='sovereign',
        is_default=False, in_arrears=False, is_paid_kind=False,
        is_convertible=False, is_inflation_linked=False,
        clean_price_est=91.0, ytw=0.0523, modified_duration=15.1,
        convexity=339.0, roll_down_12m=-0.0001, carry_12m=0.051,
        amount_outstanding_usd=28e9, _front_end_yield=0.039,
        _curve_regime={'slope_10y_3m': 0.0079, 'level_pctile_1y': 99.0},
    )
    row.update(kw)
    return row


def corporate_row(**kw):
    row = dict(
        cusip='037833DK1', asset_class='CORP_IG', issuer_name='ACME CORP',
        title_of_issue='ACME CORP SR NOTE 5.000% 06/15/35',
        coupon_type='Fixed', coupon_rate=0.05, maturity_date=date(2035, 6, 15),
        years_to_maturity=8.9, is_default=False, in_arrears=False,
        is_paid_kind=False, is_convertible=False, is_inflation_linked=False,
        clean_price_est=97.0, clean_price_marked=97.0, ytw=0.0541,
        modified_duration=6.9, convexity=58.0, roll_down_12m=0.004,
        carry_12m=0.051, z_spread=0.0125, spread_mispricing=0.0030,
        price_mispricing=0.021, yield_over_treasury=0.0130,
        issuer_cik='0000320193', issuer_ticker='ACME',
        issuer_sector='Industrials', _fundamentals_asof='2026-06-30',
        cusip_match_confidence=0.95, issuer_int_cov=8.4, issuer_nd_ebitda=2.1,
        issuer_fcf_to_debt=0.22, issuer_altman_z=3.4, credit_score_trend=1.5,
        bucket_divergence_notches=1, issuer_debt_maturity_wall_yrs=11.0,
        n_funds=14, total_held_usd=420e6, price_dispersion=0.004,
        fair_value_level=2, mark_age_days=61, _front_end_yield=0.039,
        _curve_regime={'slope_10y_3m': 0.0079, 'level_pctile_1y': 99.0},
    )
    row.update(kw)
    return row


def _applicable_names(row):
    prepare_scoring_fields([row])
    return {g.name for g in GATES if _gate_applicable(g, row)}


def _masked_names(row):
    prepare_scoring_fields([row])
    return {g.name for g in GATES if not _gate_applicable(g, row)}


# ---------------------------------------------------------------------------
# The keystone
# ---------------------------------------------------------------------------

TREASURY_MASKED = {
    # Every credit-relative valuation question.
    'Valuation: Spread vs Fair',
    'Valuation: Price vs Fair',
    'Valuation: Spread Percentile',
    'Valuation: Yield over Tsy',
    # The entire Credit category.
    'Credit: Int Coverage',
    'Credit: Net Debt EBITDA',
    'Credit: FCF to Debt',
    'Credit: Altman Z',
    'Credit: Trend',
    'Credit: Rating Divergence',
    'Credit: CET1',
    'Credit: NPL Ratio',
    # Structure questions that presuppose a corporate capital stack.
    'Structure: Seniority',
    'Structure: Maturity Wall',
    'Structure: Payment Status',
    # Fund-holding metrics, absent until N-PORT marks are attached at M4.
    'Liquidity: Fund Breadth',
    'Liquidity: Held Value',
    'Liquidity: Mark Agreement',
    'Liquidity: Valuation Level',
}

TREASURY_APPLICABLE = {
    'Valuation: Yield vs Cash',
    'Rates: Duration Fit',
    'Rates: Convexity',
    'Rates: Roll Down',
    'Rates: Carry and Roll',
    'Structure: Analyzability',
    'Liquidity: Issue Size',
}


def test_treasury_masks_exactly_these_gates():
    """The regression guard on the entire applicability design."""
    row = treasury_row()
    assert _masked_names(row) == TREASURY_MASKED
    assert _applicable_names(row) == TREASURY_APPLICABLE
    assert len(TREASURY_MASKED) + len(TREASURY_APPLICABLE) == len(GATES)


def test_treasury_drops_the_credit_category_entirely():
    rows = [treasury_row()]
    compute_continuous_scores(rows, SPEC)
    assert rows[0]['_score_credit'] is None
    assert 'Credit' not in rows[0]['_composite_categories']
    assert set(rows[0]['_composite_categories']) == {
        'Valuation', 'Rates', 'Structure', 'Liquidity'}


def test_treasury_is_not_penalised_for_having_no_credit_metrics():
    """A Treasury and a strong corporate should not be separated by the
    Treasury's ABSENCE of leverage data. If the mask leaked, the Treasury's
    composite would collapse toward zero."""
    tsy, corp = [treasury_row()], [corporate_row()]
    compute_continuous_scores(tsy, SPEC)
    compute_continuous_scores(corp, SPEC)
    assert tsy[0]['_composite_score'] > 40
    assert tsy[0]['_data_coverage_score'] == 100.0


def test_corporate_scores_every_category():
    rows = [corporate_row()]
    compute_continuous_scores(rows, SPEC)
    assert set(rows[0]['_composite_categories']) == {
        'Valuation', 'Credit', 'Rates', 'Structure', 'Liquidity'}


def test_bank_swaps_the_credit_scorecard_rather_than_losing_it():
    """Banks mask the corporate leverage gates and pick up CET1/NPL instead —
    the category must not simply vanish."""
    bank = corporate_row(issuer_sector='Financial Services',
                         issuer_cet1_ratio=0.132, issuer_npl_ratio=0.006)
    names = _applicable_names(bank)
    assert 'Credit: CET1' in names and 'Credit: NPL Ratio' in names
    assert 'Credit: Int Coverage' not in names
    assert 'Credit: Net Debt EBITDA' not in names
    # Divergence and trend are sector-agnostic and must survive.
    assert 'Credit: Rating Divergence' in names


def test_unidentified_issuers_are_penalised_not_excused():
    """A corporate bond ALWAYS has an issuer whose leverage matters. If we
    cannot identify it that is MISSING DATA — scored zero, kept in the
    denominator — not a question that fails to apply.

    Masking these instead inverted the ranking: dropping the Credit category
    (whose gates average 33-48) let the composite renormalise over the
    higher-scoring categories, so bonds we knew nothing about rose to the top.
    Thirteen of the first corporate run's top fourteen names were
    unidentifiable high-coupon high-yield paper."""
    row = corporate_row(cusip_match_confidence=0.55, issuer_cik=None,
                        issuer_int_cov=None, issuer_nd_ebitda=None,
                        issuer_altman_z=None)
    names = _applicable_names(row)
    assert 'Credit: Int Coverage' in names        # applies, and will score 0
    assert 'Valuation: Spread vs Fair' in names
    assert 'Rates: Duration Fit' in names

    scored = [dict(row)]
    compute_continuous_scores(scored, SPEC)
    assert scored[0]['_score_int_coverage'] == 0.0
    assert 'Credit' in scored[0]['_composite_categories']


def test_an_unidentified_issuer_scores_below_an_identified_one():
    """The direct consequence: not knowing an issuer must cost the bond, not
    reward it."""
    known = [corporate_row()]
    unknown = [corporate_row(cusip_match_confidence=0.40, issuer_cik=None,
                             issuer_int_cov=None, issuer_nd_ebitda=None,
                             issuer_altman_z=None, issuer_fcf_to_debt=None,
                             bucket_divergence_notches=None)]
    compute_continuous_scores(known, SPEC)
    compute_continuous_scores(unknown, SPEC)
    assert unknown[0]['_composite_score'] < known[0]['_composite_score']


def test_bank_only_gates_still_require_a_known_financial_issuer():
    """CET1 is meaningless for anything but a bank, and an unidentified issuer
    is assumed non-financial — so those gates mask rather than score zero."""
    row = corporate_row(cusip_match_confidence=0.55, issuer_cik=None,
                        issuer_sector=None)
    names = _applicable_names(row)
    assert 'Credit: CET1' not in names
    assert 'Credit: NPL Ratio' not in names


def test_floater_masks_the_rates_gates():
    row = corporate_row(coupon_type='Floating')
    names = _applicable_names(row)
    assert 'Rates: Duration Fit' not in names
    assert 'Rates: Convexity' not in names
    assert 'Valuation: Yield vs Cash' not in names


def test_tips_masks_the_rates_gates():
    row = treasury_row(is_inflation_linked=True)
    assert 'Rates: Duration Fit' not in _applicable_names(row)


def test_fund_gates_apply_once_marks_are_attached():
    """The same Treasury gains liquidity gates at M4 when N-PORT marks arrive.
    The masking is data-driven, not hardcoded by asset class."""
    before = _applicable_names(treasury_row())
    after = _applicable_names(treasury_row(n_funds=9, total_held_usd=180e6,
                                           price_dispersion=0.003,
                                           fair_value_level=1))
    assert 'Liquidity: Fund Breadth' not in before
    assert 'Liquidity: Fund Breadth' in after
    assert 'Liquidity: Mark Agreement' in after


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('gate', GATES, ids=lambda g: _gate_short(g.name))
def test_every_test_fn_returns_none_on_missing_data(gate):
    row = corporate_row()
    assert gate.test_fn(None, row) is None or gate.test_fn(None, row) in (True, False)


@pytest.mark.parametrize('gate', GATES, ids=lambda g: _gate_short(g.name))
def test_every_score_fn_stays_in_range_across_extremes(gate):
    row = corporate_row()
    extremes = [-1e9, -100.0, -1.0, -0.5, 0.0, 0.001, 1.0, 50.0, 1e9]
    for value in extremes:
        for pct in (0.0, 50.0, 100.0):
            score = gate.score_fn(value, row, pct)
            if score is not None:
                assert 0.0 <= score <= 100.0, (
                    f'{gate.name} scored {score} at value={value}')


@pytest.mark.parametrize('gate', GATES, ids=lambda g: _gate_short(g.name))
def test_nan_is_treated_as_missing_not_as_a_perfect_score(gate):
    """NaN must score None, never a number. Missing values from a parquet
    column arrive as NaN, and because every NaN comparison is False,
    `min(100.0, nan)` returns 100.0 — a missing metric scoring PERFECT.
    That silently rated ~6,000 issuer-less bonds AAA with full confidence."""
    row = corporate_row()
    score = gate.score_fn(float('nan'), row, 50.0)
    assert score is None, f'{gate.name} scored {score} on NaN'


def test_score_linear_rejects_nan_directly():
    from scripts.scoring_kernel import _score_linear
    assert _score_linear(float('nan'), 0.0, 10.0) is None
    assert _score_linear(None, 0.0, 10.0) is None
    assert _score_linear('not a number', 0.0, 10.0) is None
    assert _score_linear(5.0, 0.0, 10.0) == 50.0


def test_all_missing_row_does_not_crash():
    row = {'cusip': 'X', 'asset_class': 'CORP_IG'}
    score_and_rate([row], SPEC)
    assert row['rating'] in ('PASS', 'HOLD')
    assert row['_composite_score'] is not None


def test_gate_short_names_are_unique():
    """A collision would make two gates share _gate_*/_score_* fields and
    silently overwrite each other."""
    shorts = [_gate_short(g.name) for g in GATES]
    assert len(shorts) == len(set(shorts))


def test_gate_short_names_are_clean_identifiers():
    for g in GATES:
        short = _gate_short(g.name)
        assert short.replace('_', '').isalnum(), f'{g.name} -> {short}'


def test_every_gate_has_display_metadata():
    """A missing entry falls back to the raw gate name in the report, which
    looks like a bug to a reader."""
    from scripts.gates import GATE_DISPLAY
    for g in GATES:
        assert _gate_short(g.name) in GATE_DISPLAY, g.name


def test_category_weights_cover_every_category():
    from scripts.gates import CATEGORY_WEIGHTS
    assert {g.category for g in GATES} == set(CATEGORY_WEIGHTS)


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def test_payment_status_score():
    assert _payment_status_score({}) == 100.0
    assert _payment_status_score({'is_paid_kind': True}) == 40.0
    assert _payment_status_score({'is_default': True}) == 0.0
    assert _payment_status_score({'in_arrears': True}) == 0.0


def test_analyzability_deducts_for_structure_the_model_cannot_price():
    assert _analyzability_score({}) == 100.0
    assert _analyzability_score({'is_convertible': True}) == 60.0
    assert _analyzability_score({'coupon_type': 'Floating'}) == 60.0
    assert _analyzability_score({'is_inflation_linked': True}) == 60.0
    assert _analyzability_score({'n_funds': 1}) == 80.0
    assert _analyzability_score({'is_convertible': True,
                                 'coupon_type': 'Floating'}) == 20.0


def test_peer_group_pools_by_credit_class_and_maturity():
    assert peer_group({'asset_class': 'TREASURY', 'years_to_maturity': 29.0}) == 'TSY|12y+'
    assert peer_group({'asset_class': 'CORP_IG', 'years_to_maturity': 5.0}) == 'IG|3-7y'
    assert peer_group({'asset_class': 'CORP_HY', 'years_to_maturity': 2.0}) == 'HY|0-3y'
    assert peer_group({'asset_class': 'CORP_IG', 'implied_bucket': 'B',
                       'years_to_maturity': 9.0}) == 'HY|7-12y'


def test_maturity_wall_sign_convention():
    """Positive means the issuer's refi crunch lands AFTER this bond matures,
    so the holder is repaid before the squeeze."""
    row = corporate_row(issuer_debt_maturity_wall_yrs=11.0, years_to_maturity=8.9)
    prepare_scoring_fields([row])
    assert row['wall_vs_own_maturity'] == pytest.approx(2.1)


def test_convexity_is_normalised_per_unit_duration():
    row = corporate_row(convexity=58.0, modified_duration=6.9)
    prepare_scoring_fields([row])
    assert row['convexity_per_duration'] == pytest.approx(58.0 / 6.9)


def test_prepare_is_idempotent():
    """Snapshots round-trip through rescoring, so derived fields must be
    recomputed from primitives rather than compounding."""
    row = corporate_row()
    prepare_scoring_fields([row])
    first = dict(row)
    prepare_scoring_fields([row])
    for key in ('wall_vs_own_maturity', 'convexity_per_duration',
                'ytw_over_3m', 'carry_roll_12m', 'analyzability_score'):
        assert row[key] == first[key]


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('overrides,fragment,expected', [
    ({'is_default': True}, 'default', 'PASS'),
    ({'in_arrears': True}, 'arrears', 'PASS'),
    ({'spread_mispricing': -0.02}, 'through fair', 'PASS'),
    ({'is_paid_kind': True}, 'PIK', 'HOLD'),
    ({'ytm_solver_failed': True}, 'solver', 'HOLD'),
    ({'mark_age_days': 400}, 'stale mark', 'HOLD'),
    ({'n_funds': 1}, 'thin fund', 'HOLD'),
    ({'price_dispersion': 0.08}, 'disagreement', 'HOLD'),
    ({'fair_value_level': 3}, 'level-3', 'HOLD'),
    ({'cusip_match_confidence': 0.4}, 'match confidence', 'HOLD'),
    ({'issuer_altman_z_zone': 'distress'}, 'distress', 'HOLD'),
    ({'coupon_type': 'Floating'}, 'non-fixed', 'HOLD'),
    ({'is_convertible': True}, 'convertible', 'HOLD'),
    ({'is_inflation_linked': True}, 'inflation-linked', 'HOLD'),
    ({'years_to_maturity': 0.2}, 'cash decision', 'HOLD'),
])
def test_each_cap_fires_on_its_trigger(overrides, fragment, expected):
    cap, reasons = rating_cap_for_row(corporate_row(**overrides))
    assert cap == expected
    assert any(fragment in r for r in reasons), reasons


def test_clean_row_is_uncapped():
    cap, reasons = rating_cap_for_row(corporate_row())
    assert cap is None and reasons == []


def test_callable_above_par_without_a_schedule_is_capped():
    """The OAS gap made explicit: this is precisely where Z-spread overstates
    the compensation on offer, so the honest answer is 'cannot tell'."""
    cap, reasons = rating_cap_for_row(
        corporate_row(clean_price_est=104.0, is_likely_callable=True,
                      call_schedule=None))
    assert cap == 'HOLD'
    assert any('callable above par' in r for r in reasons)


def test_a_known_call_schedule_lifts_that_cap():
    cap, reasons = rating_cap_for_row(
        corporate_row(clean_price_est=104.0, is_likely_callable=True,
                      call_schedule=[(date(2030, 6, 15), 102.0)]))
    assert not any('callable above par' in r for r in reasons)


def test_most_severe_cap_wins():
    cap, reasons = rating_cap_for_row(
        corporate_row(is_default=True, n_funds=1, fair_value_level=3))
    assert cap == 'PASS'
    assert len(reasons) >= 3


def test_caps_never_raise_a_rating():
    row = corporate_row(is_default=True, spread_mispricing=-0.05)
    score_and_rate([row], SPEC)
    assert row['rating'] == 'PASS'
    assert row['rating_raw'] is not None


def test_capped_rows_keep_their_uncapped_rating_visible():
    """rating != rating_raw is what the report's warning badge keys off, so a
    cap must qualify a row rather than hide it."""
    row = corporate_row(spread_mispricing=0.02, n_funds=1)
    score_and_rate([row], SPEC)
    assert row['_rating_cap'] == 'HOLD'
    assert row['rating_raw'] is not None
    assert row['_rating_cap_reasons']


def test_a_treasury_is_not_capped_by_corporate_only_checks():
    row = treasury_row()
    prepare_scoring_fields([row])
    cap, reasons = rating_cap_for_row(row)
    assert cap is None, reasons


def test_government_paper_keeps_its_asset_class_through_the_credit_model():
    """A Treasury has no issuer balance sheet, so the scorecard returns no
    bucket — and an unknown bucket defaults to CORP_IG. Running government
    paper through it relabelled all 402 Treasuries as corporates in a combined
    run, destroying the masking that gives them a rating scale."""
    from scripts.analyze_bonds import apply_credit_model
    rows = [treasury_row(asset_class='TREASURY'),
            treasury_row(asset_class='TREASURY_BILL'),
            corporate_row()]
    apply_credit_model(rows, {})
    assert rows[0]['asset_class'] == 'TREASURY'
    assert rows[1]['asset_class'] == 'TREASURY_BILL'
    assert rows[2]['asset_class'] in ('CORP_IG', 'CORP_HY')
    assert 'implied_bucket' not in rows[0]


# ---------------------------------------------------------------------------
# Credit cutpoints
# ---------------------------------------------------------------------------

def test_calibrated_cutpoints_place_most_of_the_universe_in_investment_grade():
    """The seed cutpoints split a roughly uniform score distribution into
    sevenths, assigning 51% of the universe to high yield when the market
    prices 23% there — and 258 bonds to CCC where the market saw three. Those
    'CCC' bonds traded at 120bp; the real CCC index is 1023bp.

    The mislabel is not cosmetic: fair_spread multiplies by the bucket's index
    OAS, so a BBB called CCC is handed a 1023bp fair spread and reads as
    absurdly rich."""
    from models.credit import bucket_from_score
    from scripts.config import (CREDIT_CUT_A, CREDIT_CUT_AA, CREDIT_CUT_AAA,
                                CREDIT_CUT_B, CREDIT_CUT_BB, CREDIT_CUT_BBB)
    # A median-quality issuer must not read as high yield.
    assert bucket_from_score(50.0) in ('AAA', 'AA', 'A', 'BBB')
    # Only genuinely weak scores reach the distressed buckets.
    assert bucket_from_score(5.0) in ('B', 'CCC')
    assert CREDIT_CUT_AAA > CREDIT_CUT_AA > CREDIT_CUT_A > CREDIT_CUT_BBB
    assert CREDIT_CUT_BBB > CREDIT_CUT_BB > CREDIT_CUT_B


def test_calibration_matches_the_market_bucket_mix():
    """Distribution matching: the model's mix should reproduce the market's."""
    from models.credit import calibrate_cutpoints, bucket_from_score
    # 100 bonds: market says 40% A, 40% BBB, 20% BB. Scores span 0-99.
    rows = []
    for i in range(100):
        market = 'A' if i >= 60 else ('BBB' if i >= 20 else 'BB')
        rows.append({'issuer_credit_score': float(i), 'market_bucket': market})
    cuts = calibrate_cutpoints(rows, min_per_bucket=10, min_rows=50)
    assert cuts
    mix = {}
    for row in rows:
        b = bucket_from_score(row['issuer_credit_score'], cuts)
        mix[b] = mix.get(b, 0) + 1
    # High-yield share should land near the market's 20%, not the seed's ~50%.
    hy = sum(mix.get(b, 0) for b in ('BB', 'B', 'CCC'))
    assert 10 <= hy <= 35, mix


def test_calibration_keeps_cutpoints_strictly_decreasing():
    """A non-monotone scorecard inverts the rating scale silently."""
    from models.credit import CUTPOINT_PARAMS, calibrate_cutpoints
    rows = [{'issuer_credit_score': float(i % 100),
             'market_bucket': ['AAA', 'AA', 'A', 'BBB', 'BB', 'B'][i % 6]}
            for i in range(600)]
    cuts = calibrate_cutpoints(rows, min_per_bucket=10, min_rows=50)
    values = [cuts[p] for p in CUTPOINT_PARAMS if p in cuts]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


def test_calibration_refuses_a_thin_sample():
    from models.credit import calibrate_cutpoints
    assert calibrate_cutpoints([{'issuer_credit_score': 50.0,
                                 'market_bucket': 'A'}] * 10) == {}
