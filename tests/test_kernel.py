"""Kernel machinery tests, using a toy gate set rather than the real bond one.

The point of a toy set is that the expected numbers are hand-computable, so
these tests pin the machinery itself: the applicable-vs-missing distinction,
category renormalisation, percentile pooling, cap precedence, and — the one
that matters most for this project — that a row masking an entire category
drops that category out of the composite and renormalises over the rest.

The real bond GATES get their own test_gate_logic.py at M6.
"""

import pytest

from scripts.param_set import default_params, merge_params, validate_params
from scripts.scoring_kernel import (
    Gate, ScoringSpec, RATING_RANK,
    _cap_rating, _gate_key, _gp_key, _score_key, _gate_short,
    _ranked_percentiles, _score_linear,
    apply_screening_matrix, compute_continuous_scores, gate_metadata,
    rating_from_composite, score_and_rate,
)


# ---------------------------------------------------------------------------
# Toy spec: two categories, one of which is inapplicable to "TREASURY" rows.
# ---------------------------------------------------------------------------

def _appl_credit(r):
    return r.get('asset_class') != 'TREASURY'


TOY_GATES = [
    Gate('Valuation: Cheapness', 'cheap',
         lambda v, r: v > 0.10 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.10, 0.40)),
    Gate('Valuation: Peer Spread', 'spread',
         lambda v, r: v > 0.02 if v is not None else None,
         lambda v, r, pct: pct,
         relative_mode='peer', higher_better=True),
    Gate('Credit: Coverage', 'cov',
         lambda v, r: v > 3.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 1.0, 15.0),
         applicable=_appl_credit),
]

TOY_SPEC = ScoringSpec(
    gates=TOY_GATES,
    category_weights={'Valuation': ('score_weight_valuation', 0.60),
                      'Credit': ('score_weight_credit', 0.40)},
    category_order=['Valuation', 'Credit'],
    gate_display={'cheapness': {'label': 'Cheap', 'threshold': '> 10%', 'fmt': 'pct1'}},
    category_display={'Valuation': {'dark': '#2F5496', 'light': '#D6E4F0'}},
)


def _corp(**kw):
    row = dict(asset_class='CORP_IG', peer_group='IG|3-7y',
               cheap=0.20, spread=0.03, cov=9.0)
    row.update(kw)
    return row


def _tsy(**kw):
    row = dict(asset_class='TREASURY', peer_group='TSY|3-7y',
               cheap=0.20, spread=0.03)
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def test_score_linear_clamps_and_maps():
    assert _score_linear(0.0, 0.0, 1.0) == 0.0
    assert _score_linear(1.0, 0.0, 1.0) == 100.0
    assert _score_linear(0.25, 0.0, 1.0) == 25.0
    # Clamped outside the range, not extrapolated.
    assert _score_linear(-5.0, 0.0, 1.0) == 0.0
    assert _score_linear(5.0, 0.0, 1.0) == 100.0
    # Lower-is-better: worst > best.
    assert _score_linear(0.0, 1.0, 0.0) == 100.0
    assert _score_linear(1.0, 1.0, 0.0) == 0.0
    # Degenerate range is a coin flip, not a divide-by-zero.
    assert _score_linear(3.0, 2.0, 2.0) == 50.0


def test_score_linear_distinguishes_missing_from_zero():
    """None must survive as None. If it collapsed to 0.0 the caller could no
    longer tell 'no data' from 'scored worst'."""
    assert _score_linear(None, 0.0, 1.0) is None
    assert _score_linear(0.0, 0.0, 1.0) == 0.0


def test_ranked_percentiles_ties_share_a_score():
    pct = _ranked_percentiles([(0, 1.0), (1, 2.0), (2, 2.0), (3, 3.0)])
    assert pct[0] == 0.0
    assert pct[1] == pct[2] == 50.0     # average rank 1.5 of 3
    assert pct[3] == 100.0


def test_ranked_percentiles_direction_and_degenerate_pools():
    assert _ranked_percentiles([]) == {}
    assert _ranked_percentiles([(7, 1.0)]) == {7: 50.0}
    lo = _ranked_percentiles([(0, 1.0), (1, 9.0)], higher_better=False)
    assert lo[0] == 100.0 and lo[1] == 0.0


def test_gate_key_helpers_are_stable():
    assert _gate_short('Valuation: Spread vs Fair') == 'spread_vs_fair'
    assert _gate_short('Credit: Net Debt/EBITDA') == 'net_debt_ebitda'
    assert _gate_key('Credit: Coverage') == '_gate_coverage'
    assert _gp_key('Credit: Coverage') == '_gp_coverage'
    assert _score_key('Credit: Coverage') == '_score_coverage'
    assert TOY_GATES[2].category == 'Credit'


def test_cap_rating_only_lowers():
    assert _cap_rating('BUY', 'HOLD') == 'HOLD'
    assert _cap_rating('PASS', 'HOLD') == 'PASS'      # never raises
    assert _cap_rating('BUY', None) == 'BUY'
    assert _cap_rating(None, 'HOLD') is None


# ---------------------------------------------------------------------------
# Applicable vs missing
# ---------------------------------------------------------------------------

def test_missing_data_fails_but_stays_in_denominator():
    rows = [_corp(cov=None)]
    apply_screening_matrix(rows, TOY_SPEC)
    r = rows[0]
    assert r['_gp_coverage'] is None            # renders N/A
    assert r['_gates_applicable'] == 3          # still counted
    assert r['_gates_passed'] == '2/3'
    assert r['_gates_inapplicable'] == 0


def test_inapplicable_gate_leaves_the_denominator_entirely():
    rows = [_tsy()]
    apply_screening_matrix(rows, TOY_SPEC)
    r = rows[0]
    assert r['_gp_coverage'] is None
    assert r['_gates_inapplicable'] == 1
    assert r['_gates_applicable'] == 2
    assert r['_gates_passed'] == '2/2'          # not penalised for having no credit


def test_missing_scores_zero_while_inapplicable_scores_none():
    rows = [_corp(cov=None), _tsy()]
    compute_continuous_scores(rows, TOY_SPEC)
    assert rows[0]['_score_coverage'] == 0.0    # missing -> worst
    assert rows[1]['_score_coverage'] is None   # inapplicable -> N/A


# ---------------------------------------------------------------------------
# The Treasury path: whole-category dropout
# ---------------------------------------------------------------------------

def test_treasury_drops_credit_and_renormalises_over_valuation_alone():
    """The load-bearing behaviour for this entire project.

    A Treasury masks every Credit gate, so Credit's applicable weight is 0,
    its average is None, and the composite renormalises over Valuation's
    weight alone — the composite must equal the Valuation score exactly, not
    be dragged toward zero by an absent Credit category.
    """
    rows = [_tsy(cheap=0.40, spread=0.03)]
    compute_continuous_scores(rows, TOY_SPEC)
    r = rows[0]
    assert r['_score_credit'] is None
    assert r['_composite_categories'] == ['Valuation']
    assert r['_score_valuation'] is not None
    assert r['_composite_score'] == pytest.approx(r['_score_valuation'], abs=0.05)


def test_treasury_coverage_ignores_inapplicable_gates():
    """A Treasury must not read as low-coverage merely because credit gates
    that cannot describe it are absent."""
    rows = [_tsy()]
    compute_continuous_scores(rows, TOY_SPEC)
    assert rows[0]['_data_coverage_score'] == 100.0


def test_corporate_composite_uses_both_categories():
    rows = [_corp(cheap=0.40, cov=15.0)]
    compute_continuous_scores(rows, TOY_SPEC)
    r = rows[0]
    assert r['_composite_categories'] == ['Valuation', 'Credit']
    assert r['_score_credit'] == 100.0


def test_composite_is_invariant_to_gate_ordering():
    spec_reordered = TOY_SPEC._replace(gates=list(reversed(TOY_GATES)))
    a, b = [_corp()], [_corp()]
    compute_continuous_scores(a, TOY_SPEC)
    compute_continuous_scores(b, spec_reordered)
    assert a[0]['_composite_score'] == pytest.approx(b[0]['_composite_score'])


def test_all_missing_row_scores_zero_rather_than_crashing():
    rows = [_corp(cheap=None, spread=None, cov=None)]
    score_and_rate(rows, TOY_SPEC)
    assert rows[0]['_composite_score'] == 0.0
    assert rows[0]['rating'] == 'PASS'
    assert rows[0]['_data_coverage_score'] == 0.0


# ---------------------------------------------------------------------------
# Peer pooling
# ---------------------------------------------------------------------------

def test_peer_percentiles_pool_within_peer_group():
    """Treasury and corporate spreads must be ranked against their own kind.
    The corporate here has the LOWEST spread of all six rows but the highest
    within IG, so a correct peer pooling scores it top and a broken global
    pooling scores it bottom."""
    rows = ([_corp(spread=s, peer_group='IG|3-7y') for s in
             (0.010, 0.012, 0.014, 0.016, 0.018)]
            + [_tsy(spread=s, peer_group='TSY|3-7y') for s in
               (0.20, 0.22, 0.24, 0.26, 0.28)])
    compute_continuous_scores(rows, TOY_SPEC)
    ig_scores = [r['_score_peer_spread'] for r in rows[:5]]
    assert ig_scores == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_small_peer_pool_falls_back_to_global():
    """Below MIN_PEER_SCORING a lone peer group must borrow the global pool
    rather than score everyone at 50."""
    rows = ([_corp(spread=s, peer_group='IG|3-7y') for s in
             (0.010, 0.012, 0.014, 0.016, 0.018)]
            + [_corp(spread=0.030, peer_group='HY|12y+')])
    compute_continuous_scores(rows, TOY_SPEC)
    # Widest spread in the whole set, ranked globally -> top score.
    assert rows[5]['_score_peer_spread'] == 100.0


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

def test_rating_from_composite_boundaries_are_inclusive():
    assert rating_from_composite(57) == 'BUY'
    assert rating_from_composite(56.9) == 'LEAN BUY'
    assert rating_from_composite(39) == 'LEAN BUY'
    assert rating_from_composite(38.9) == 'HOLD'
    assert rating_from_composite(25) == 'HOLD'
    assert rating_from_composite(24.9) == 'PASS'
    assert rating_from_composite(None) is None


def test_per_asset_class_thresholds_win_over_base():
    """Treasury composites are computed over a different category set, so they
    need their own scale. A class override must beat the base threshold.

    Uses AGENCY as the control because every other class carries calibrated
    thresholds in config, and a test that reads live config is testing the
    calibration rather than the precedence logic."""
    params = {'rating_threshold_buy': 57, 'rating_threshold_lean': 39,
              'rating_threshold_buy_treasury': 70}
    assert rating_from_composite(65, params, asset_class='AGENCY') == 'BUY'
    assert rating_from_composite(65, params, asset_class='TREASURY') == 'LEAN BUY'
    assert rating_from_composite(72, params, asset_class='TREASURY') == 'BUY'


def test_unset_class_override_falls_back_to_base():
    """AGENCY deliberately: TREASURY carries calibrated thresholds in config,
    and a test that silently tracks whatever calibration last ran is testing
    the config file rather than the fallback logic."""
    params = default_params()          # every per-class param is None
    params['rating_threshold_buy'] = 57
    assert rating_from_composite(60, params, asset_class='AGENCY') == 'BUY'


def test_config_class_thresholds_are_consulted_when_params_are_silent():
    """The calibrated thresholds in config.py must actually take effect, not
    just sit there — params override config, config overrides the base."""
    from scripts.config import RATING_THRESHOLDS_BY_CLASS
    cuts = RATING_THRESHOLDS_BY_CLASS.get('TREASURY') or {}
    if not cuts:
        pytest.skip('no calibrated Treasury thresholds in config')
    params = default_params()
    params['rating_threshold_buy'] = 57
    just_under = cuts['buy'] - 0.5
    assert just_under > 57, 'test assumes a calibrated cut above the base'
    assert rating_from_composite(just_under, params,
                                 asset_class='TREASURY') != 'BUY'
    assert rating_from_composite(cuts['buy'], params,
                                 asset_class='TREASURY') == 'BUY'


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

def _toy_cap_fn(row, params=None):
    reasons = []
    cap = None

    def add(new_cap, reason):
        nonlocal cap
        if cap is None or RATING_RANK[new_cap] < RATING_RANK[cap]:
            cap = new_cap
        reasons.append(reason)

    if row.get('is_default'):
        add('PASS', 'issuer in default')
    if row.get('n_funds', 99) < 3:
        add('HOLD', 'thin fund coverage')
    return cap, reasons


CAP_SPEC = TOY_SPEC._replace(cap_fn=_toy_cap_fn)


def test_cap_lowers_rating_and_records_reason():
    rows = [_corp(cheap=0.40, cov=15.0, n_funds=1)]
    score_and_rate(rows, CAP_SPEC)
    assert rows[0]['rating_raw'] == 'BUY'
    assert rows[0]['rating'] == 'HOLD'
    assert rows[0]['_rating_cap_reasons'] == ['thin fund coverage']


def test_most_severe_cap_wins_regardless_of_order():
    rows = [_corp(cheap=0.40, cov=15.0, n_funds=1, is_default=True)]
    score_and_rate(rows, CAP_SPEC)
    assert rows[0]['_rating_cap'] == 'PASS'
    assert len(rows[0]['_rating_cap_reasons']) == 2


def test_caps_never_raise_a_rating():
    rows = [_corp(cheap=-0.10, spread=0.001, cov=1.0, n_funds=1)]
    score_and_rate(rows, CAP_SPEC)
    assert rows[0]['rating_raw'] == 'PASS'
    assert rows[0]['rating'] == 'PASS'


# ---------------------------------------------------------------------------
# Snapshot hygiene + metadata
# ---------------------------------------------------------------------------

def test_stale_gate_fields_are_purged_on_rescore():
    """Snapshots round-trip through rescoring; fields from retired gates must
    not persist forever and leak into report payloads."""
    rows = [_corp()]
    rows[0]['_gate_retired_metric'] = 1.0
    rows[0]['_gp_retired_metric'] = True
    rows[0]['_score_retired_metric'] = 80.0
    score_and_rate(rows, TOY_SPEC)
    assert '_gate_retired_metric' not in rows[0]
    assert '_gp_retired_metric' not in rows[0]
    assert '_score_retired_metric' not in rows[0]
    assert rows[0]['_gate_coverage'] == 9.0        # live gates survive


def test_gate_metadata_exposes_labels_weights_and_categories():
    meta = gate_metadata(TOY_SPEC)
    by_key = {g['key']: g for g in meta['gates']}
    assert by_key['_gate_cheapness']['label'] == 'Cheap'
    assert by_key['_gate_cheapness']['threshold'] == '> 10%'
    # Falls back to the gate name when no display entry exists.
    assert by_key['_gate_coverage']['label'] == 'Coverage'
    assert [c['name'] for c in meta['categories']] == ['Valuation', 'Credit']
    assert meta['categories'][0]['weight'] == 0.60
    assert meta['categories'][0]['scoreKey'] == '_score_valuation'


def test_gate_metadata_weights_follow_params():
    meta = gate_metadata(TOY_SPEC, {'score_weight_valuation': 0.75})
    assert meta['categories'][0]['weight'] == 0.75


# ---------------------------------------------------------------------------
# Param set
# ---------------------------------------------------------------------------

def test_default_params_validate_clean():
    assert validate_params(default_params()) == []


def test_merge_params_rejects_unknown_keys():
    """A typo'd parameter in a sweep would otherwise be silently ignored and
    the sweep would conclude the parameter had no effect."""
    with pytest.raises(ValueError, match='Unknown parameter'):
        merge_params({'score_weight_valuatoin': 0.5})


def test_validate_catches_weights_that_do_not_sum_to_one():
    errors = validate_params(merge_params({'score_weight_valuation': 0.50}))
    assert any('sum to' in e for e in errors)


def test_validate_catches_non_monotone_credit_cutpoints():
    """A non-monotone scorecard inverts the rating scale silently — the worst
    failure a calibration sweep could introduce."""
    errors = validate_params(merge_params({'credit_cut_bbb': 20}))
    assert any('strictly decrease' in e for e in errors)


def test_validate_catches_misordered_rating_thresholds():
    errors = validate_params(merge_params({'rating_threshold_lean': 80}))
    assert any('buy > lean > pass' in e for e in errors)


def test_validate_checks_per_class_thresholds_too():
    errors = validate_params(merge_params({
        'rating_threshold_buy_treasury': 30,
        'rating_threshold_lean_treasury': 40,
        'rating_threshold_pass_treasury': 20,
    }))
    assert any('treasury' in e and 'buy > lean > pass' in e for e in errors)
