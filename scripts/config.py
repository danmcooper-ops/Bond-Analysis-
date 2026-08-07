# scripts/config.py
"""Tunable constants for the bond analysis pipeline.

Shape borrowed from the equity model's scripts/config.py (flat module of
constants + a keyed dict with a default fallback), content is all bond.
Anything a calibration sweep should be able to move lives in param_set.py
as well; this module holds the defaults those params fall back to.
"""

# ---------------------------------------------------------------------------
# Composite category weights (must sum to 1.0 — validate_params enforces it)
# ---------------------------------------------------------------------------
SCORE_WEIGHT_VALUATION = 0.32
SCORE_WEIGHT_CREDIT = 0.28
SCORE_WEIGHT_RATES = 0.16
SCORE_WEIGHT_STRUCTURE = 0.12
SCORE_WEIGHT_LIQUIDITY = 0.12

# ---------------------------------------------------------------------------
# Rating thresholds
# ---------------------------------------------------------------------------
# Base thresholds, mirroring the equity model's quantile-matched 57/39/25.
# These are PROVISIONAL for bonds — they get calibrated per asset class at M8
# against the marked-to-marked backtest, because a Treasury scores 10 gates
# across 3 categories while a corporate scores 25 across 5, so their
# composites are not on one scale (see RATING_THRESHOLDS_BY_CLASS).
RATING_THRESHOLD_BUY = 57
RATING_THRESHOLD_LEAN = 39
RATING_THRESHOLD_PASS = 25

# Per-asset-class overrides. Empty values fall back to the base thresholds.
# Populated by calibrate.py; kept explicit from v1 so the comparability
# problem can never be silently ignored.
RATING_THRESHOLDS_BY_CLASS = {
    'TREASURY': {'buy': 72.5, 'lean': 62.6, 'pass': 51.0},
    'AGENCY': {},
    'CORP_IG': {'buy': 67.6, 'lean': 43.4, 'pass': 30.2},
    'CORP_HY': {'buy': 62.8, 'lean': 45.5, 'pass': 33.2},
    'TREASURY_BILL': {'buy': 72.5, 'lean': 62.6, 'pass': 51.0},
}

# ---------------------------------------------------------------------------
# Scoring machinery
# ---------------------------------------------------------------------------
# Minimum population for a peer-relative percentile pool before it falls back
# to the global pool. Equity used MIN_SECTOR_SCORING = 5 for sectors; bond
# peer groups ({TSY|IG|HY} x {0-3y,3-7y,7-12y,12y+}) are far larger, so this
# should essentially never bind — it is a guard, not a tuning knob.
MIN_PEER_SCORING = 5

# ---------------------------------------------------------------------------
# Universe filters (scripts/build_universe.py)
# ---------------------------------------------------------------------------
MIN_FUNDS_HOLDING = 2
MIN_TOTAL_HELD_USD = 10e6
MIN_YEARS_TO_MATURITY = 0.5
PRICE_SANITY_MIN = 20.0
PRICE_SANITY_MAX = 200.0

# ---------------------------------------------------------------------------
# Data-quality gates on the N-PORT mark
# ---------------------------------------------------------------------------
STALE_MARK_DAYS = 100          # older than this -> HOLD cap
MIN_FUNDS_FOR_BUY = 3          # thinner fund coverage -> HOLD cap
MAX_PRICE_DISPERSION = 0.02    # cross-fund MAD/median above this -> HOLD cap
MIN_CUSIP_MATCH_CONFIDENCE = 0.80
MAX_FUNDAMENTALS_AGE_DAYS = 400

# Consensus-mark construction (data/nport_consensus.py)
CONSENSUS_MAD_K = 3.0          # reject marks beyond k MADs from the median
IMPLIED_PRICE_FLOOR = 1.0
IMPLIED_PRICE_CEIL = 250.0

# ---------------------------------------------------------------------------
# Credit scorecard (models/credit.py)
# ---------------------------------------------------------------------------
CREDIT_BUCKETS = ('AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC')

# Score cutpoints, high to low. Calibrated monthly against the market: the
# cutpoints are chosen so each implied bucket's median observed Z-spread lines
# up with the FRED bucket OAS + fitted wedge. These are the seed values.
CREDIT_CUT_AAA = 81.9
CREDIT_CUT_AA = 69.0
CREDIT_CUT_A = 49.8
CREDIT_CUT_BBB = 34.7
CREDIT_CUT_BB = 22.5
CREDIT_CUT_B = 10.4

# Non-financial issuer scorecard: (field, worst, best, weight)
CREDIT_FACTORS_CORPORATE = (
    ('int_cov',       0.5,  15.0, 0.25),
    ('nd_ebitda',     7.0,   0.0, 0.25),
    ('fcf_to_debt',  -0.05,  0.35, 0.15),
    ('altman_z',      1.1,   5.0, 0.15),
    ('log_revenue',   8.0,  11.0, 0.10),
    ('piotroski',     0.0,   9.0, 0.10),
)

# Banks and insurers: operating cash flow reflects deposit/loan movements and
# EBITDA is not meaningful, so the corporate scorecard cannot describe them.
# Same reasoning as the equity model's _appl_non_financial mask.
CREDIT_FACTORS_FINANCIAL = (
    ('cet1_ratio',  0.06, 0.16, 0.35),
    ('npl_ratio',   0.05, 0.003, 0.25),
    ('log_revenue', 8.0, 11.0, 0.20),
    ('piotroski',   0.0,  9.0, 0.20),
)

FINANCIAL_SECTOR_NAME = 'Financial Services'

# ---------------------------------------------------------------------------
# Peer grouping for relative_mode='peer'
# ---------------------------------------------------------------------------
# (label, upper bound in years). Last bucket is open-ended.
MATURITY_BUCKETS = (('0-3y', 3.0), ('3-7y', 7.0), ('7-12y', 12.0), ('12y+', None))

# ---------------------------------------------------------------------------
# Curve construction (models/curve.py)
# ---------------------------------------------------------------------------
CURVE_INTERP_METHOD = 'monotone_cubic'   # Fritsch-Carlson on zero rates
CURVE_EXTRAPOLATION = 'flat'             # never extrapolate a slope

# Cross-check tolerance between the Treasury par XML and FRED DGS* series.
CURVE_CROSSCHECK_TOL_BP = 2.0
