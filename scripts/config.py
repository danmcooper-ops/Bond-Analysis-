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
    'CORP_IG': {'buy': 65.4, 'lean': 41.7, 'pass': 29.8},
    'CORP_HY': {'buy': 66.3, 'lean': 54.0, 'pass': 37.3},
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
CREDIT_CUT_AAA = 65.4
CREDIT_CUT_AA = 55.5
CREDIT_CUT_A = 45.1
CREDIT_CUT_BBB = 40.8
CREDIT_CUT_BB = 34.7
CREDIT_CUT_B = 9.4

# Issuer scorecard: (field, worst, best, weight).
#
# WEIGHTS ARE MEASURED, NOT CHOSEN. Each factor's rank correlation against the
# de-termed observed spread, over 443 issuers:
#
#     log_mcap          -0.755      <- strongest by far, and was ABSENT
#     log_revenue       -0.585         (dropped: redundant with log_mcap)
#     mcap_to_debt      -0.509      <- market leverage, was absent
#     altman_z          -0.405
#     int_cov           -0.356
#     fcf_to_debt       -0.296
#     nd_ebitda         +0.179      <- was weighted 0.25, second-highest
#     piotroski         -0.122      <- dropped, weakest of all
#
# The previous weights were close to backwards: the two heaviest (0.25 each)
# were int_cov and nd_ebitda, the latter among the weakest, while the single
# strongest factor was not in the scorecard at all and the second strongest
# carried the minimum weight. Rebuilding on the measurement lifts the
# scorecard's rank correlation from -0.42 to -0.55.
#
# THE SCORECARD IS NOW HYBRID, NOT PURELY ACCOUNTING-BASED, and that is a real
# change in what it claims. Market capitalisation is the equity market's view
# of the cushion sitting beneath the debt — the structural (Merton) measure of
# solvency, and the basis of every commercial default model. It is public,
# point-in-time, and by far the best predictor available.
#
# The cost is conceptual: the divergence signal was framed as "fundamentals
# versus the market", and with mcap in the score both sides now carry market
# information. It remains a legitimate comparison — equity-market and
# bond-market disagreement is a well-documented signal — but it is no longer
# fundamentals against price.
#
# A sector adjustment was fitted and DISCARDED: the sector residual looked
# substantial (Consumer Cyclical and Utilities wide, Technology tight) but
# adding it moved the rank correlation from -0.548 to -0.549, i.e. nowhere.
# The factors already capture it.
CREDIT_FACTORS_CORPORATE = (
    ('log_mcap',      8.5,  12.0, 0.22),
    ('mcap_to_debt', -0.6,   1.5, 0.20),
    ('int_cov',       0.5,  15.0, 0.18),
    ('altman_z',      1.1,   5.0, 0.15),
    ('fcf_to_debt',  -0.05,  0.35, 0.15),
    ('nd_ebitda',     7.0,   0.0, 0.10),
)

# Banks and insurers: operating cash flow reflects deposit and loan movements
# and EBITDA is not a meaningful denominator, so the leverage and coverage
# factors above cannot describe them. Scale and market leverage still can, and
# they carry most of the weight because CET1 and NPL are populated for only a
# minority of bank issuers — the reweighting over present factors then leans
# on the two that are always there.
CREDIT_FACTORS_FINANCIAL = (
    ('log_mcap',      8.5,  12.0, 0.30),
    ('mcap_to_debt', -0.6,   1.5, 0.25),
    ('cet1_ratio',   0.06,  0.16, 0.25),
    ('npl_ratio',    0.05, 0.003, 0.20),
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
