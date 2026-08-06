# scripts/param_set.py
#
# Shape VENDORED from stock-analysis-model @ 168c17a on 2026-08-06
# (default_params / merge_params / validate_params, unknown-key rejection,
# error-list return rather than raising). Contents are all bond.
"""ParamSet: overridable parameter sets for the bond analysis pipeline.

A ParamSet is a plain dict of {param_name: value}. ``default_params()``
returns the config.py constants; ``merge_params(overrides)`` layers user
overrides on top and rejects unknown keys; ``validate_params(params)``
returns a list of problems (empty when valid) rather than raising, so a
calibration sweep can score a candidate set and move on.

With no overrides the pipeline behaves identically to the constant path.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.config import (
    SCORE_WEIGHT_VALUATION, SCORE_WEIGHT_CREDIT, SCORE_WEIGHT_RATES,
    SCORE_WEIGHT_STRUCTURE, SCORE_WEIGHT_LIQUIDITY,
    RATING_THRESHOLD_BUY, RATING_THRESHOLD_LEAN, RATING_THRESHOLD_PASS,
    CREDIT_CUT_AAA, CREDIT_CUT_AA, CREDIT_CUT_A, CREDIT_CUT_BBB,
    CREDIT_CUT_BB, CREDIT_CUT_B,
    STALE_MARK_DAYS, MIN_FUNDS_FOR_BUY, MAX_PRICE_DISPERSION,
    MIN_CUSIP_MATCH_CONFIDENCE, MAX_FUNDAMENTALS_AGE_DAYS,
    MIN_FUNDS_HOLDING, MIN_TOTAL_HELD_USD, MIN_YEARS_TO_MATURITY,
    CONSENSUS_MAD_K,
)

CATEGORY_WEIGHT_KEYS = (
    'score_weight_valuation', 'score_weight_credit', 'score_weight_rates',
    'score_weight_structure', 'score_weight_liquidity',
)

ASSET_CLASSES = ('treasury', 'agency', 'corp_ig', 'corp_hy')

# Highest to lowest. Used by validate_params to enforce monotone cutpoints —
# a non-monotone credit scorecard would silently invert the rating scale.
CREDIT_CUT_KEYS = ('credit_cut_aaa', 'credit_cut_aa', 'credit_cut_a',
                   'credit_cut_bbb', 'credit_cut_bb', 'credit_cut_b')


def default_params():
    """Return the full parameter set at its configured defaults."""
    params = {
        # Composite category weights
        'score_weight_valuation': SCORE_WEIGHT_VALUATION,
        'score_weight_credit': SCORE_WEIGHT_CREDIT,
        'score_weight_rates': SCORE_WEIGHT_RATES,
        'score_weight_structure': SCORE_WEIGHT_STRUCTURE,
        'score_weight_liquidity': SCORE_WEIGHT_LIQUIDITY,

        # Base rating thresholds
        'rating_threshold_buy': RATING_THRESHOLD_BUY,
        'rating_threshold_lean': RATING_THRESHOLD_LEAN,
        'rating_threshold_pass': RATING_THRESHOLD_PASS,

        # Credit scorecard cutpoints (calibrated against market spreads)
        'credit_cut_aaa': CREDIT_CUT_AAA,
        'credit_cut_aa': CREDIT_CUT_AA,
        'credit_cut_a': CREDIT_CUT_A,
        'credit_cut_bbb': CREDIT_CUT_BBB,
        'credit_cut_bb': CREDIT_CUT_BB,
        'credit_cut_b': CREDIT_CUT_B,

        # Fair-spread term structure. 1.0 = use the FRED IG maturity slices
        # as-is; the sweep can damp or amplify the term effect.
        'fair_spread_term_beta': 1.0,

        # Data-quality caps
        'stale_mark_days': STALE_MARK_DAYS,
        'min_funds_for_buy': MIN_FUNDS_FOR_BUY,
        'max_price_dispersion': MAX_PRICE_DISPERSION,
        'min_cusip_match_confidence': MIN_CUSIP_MATCH_CONFIDENCE,
        'max_fundamentals_age_days': MAX_FUNDAMENTALS_AGE_DAYS,

        # Universe construction
        'min_funds_holding': MIN_FUNDS_HOLDING,
        'min_total_held_usd': MIN_TOTAL_HELD_USD,
        'min_years_to_maturity': MIN_YEARS_TO_MATURITY,
        'consensus_mad_k': CONSENSUS_MAD_K,
    }
    # Per-asset-class threshold overrides. Absent by default so
    # rating_from_composite falls through to the base thresholds; calibrate.py
    # populates them once there is a backtest to calibrate against.
    for cls in ASSET_CLASSES:
        for name in ('buy', 'lean', 'pass'):
            params[f'rating_threshold_{name}_{cls}'] = None
    return params


def merge_params(overrides=None):
    """Return default params with optional overrides applied.

    Unknown keys raise ValueError — a typo'd parameter in a sweep would
    otherwise be silently ignored and the sweep would report that the
    parameter had no effect.
    """
    params = default_params()
    if overrides:
        for k, v in overrides.items():
            if k not in params:
                raise ValueError(f"Unknown parameter: '{k}'")
            params[k] = v
    return params


def active_params(params):
    """Strip None-valued per-class overrides so `key in params` reads true
    only for thresholds that were actually set."""
    return {k: v for k, v in params.items() if v is not None}


def validate_params(params):
    """Return a list of validation error strings (empty if valid)."""
    errors = []

    # Category weights must sum to 1.0
    sw = sum(params.get(k, 0) for k in CATEGORY_WEIGHT_KEYS)
    if abs(sw - 1.0) > 0.01:
        errors.append(f"Category weights sum to {sw:.3f}, expected 1.0")

    for key in CATEGORY_WEIGHT_KEYS:
        v = params.get(key, 0)
        if v < 0.05:
            errors.append(f"{key} = {v:.3f} is below minimum 0.05")

    # Rating thresholds: in range and strictly ordered, per class and base.
    def check_thresholds(suffix, label):
        vals = {}
        for name in ('buy', 'lean', 'pass'):
            v = params.get(f'rating_threshold_{name}{suffix}')
            if v is None:
                return          # unset class overrides fall back to base
            if not (0 <= v <= 100):
                errors.append(f"{label} {name} threshold {v} outside [0, 100]")
            vals[name] = v
        if len(vals) == 3 and not (vals['buy'] > vals['lean'] > vals['pass']):
            errors.append(
                f"{label} rating thresholds must satisfy buy > lean > pass "
                f"(got {vals['buy']} / {vals['lean']} / {vals['pass']})")

    check_thresholds('', 'base')
    for cls in ASSET_CLASSES:
        check_thresholds(f'_{cls}', cls)

    # Credit cutpoints must be strictly decreasing AAA -> B. A non-monotone
    # scorecard inverts the rating scale silently, which is the worst possible
    # failure mode for a calibration sweep to introduce.
    cuts = [params.get(k) for k in CREDIT_CUT_KEYS]
    if all(c is not None for c in cuts):
        for i in range(len(cuts) - 1):
            if cuts[i] <= cuts[i + 1]:
                errors.append(
                    f"Credit cutpoints must strictly decrease: "
                    f"{CREDIT_CUT_KEYS[i]}={cuts[i]} <= "
                    f"{CREDIT_CUT_KEYS[i + 1]}={cuts[i + 1]}")
        for k, c in zip(CREDIT_CUT_KEYS, cuts):
            if not (0 <= c <= 100):
                errors.append(f"{k} = {c} outside [0, 100]")

    beta = params.get('fair_spread_term_beta', 1.0)
    if not (0.0 <= beta <= 3.0):
        errors.append(f"fair_spread_term_beta {beta} outside [0.0, 3.0]")

    mad_k = params.get('consensus_mad_k', 0)
    if mad_k <= 0:
        errors.append(f"consensus_mad_k = {mad_k} must be positive")

    conf = params.get('min_cusip_match_confidence', 0)
    if not (0.0 <= conf <= 1.0):
        errors.append(f"min_cusip_match_confidence {conf} outside [0.0, 1.0]")

    disp = params.get('max_price_dispersion', 0)
    if not (0.0 < disp <= 1.0):
        errors.append(f"max_price_dispersion {disp} outside (0.0, 1.0]")

    stale = params.get('stale_mark_days', 0)
    if stale <= 0:
        errors.append(f"stale_mark_days = {stale} must be positive")

    return errors
