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
    'TREASURY': {'buy': 73.0, 'lean': 65.4, 'pass': 55.9},
    'TREASURY_BILL': {'buy': 57.3, 'lean': 53.0, 'pass': 44.8},
    'AGENCY': {},
    'CORP_IG': {'buy': 65.9, 'lean': 44.3, 'pass': 32.0},
    'CORP_HY': {'buy': 67.6, 'lean': 54.7, 'pass': 37.0},
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
# Staleness is judged against the DATA VINTAGE, not the wall clock. N-PORT
# publishes quarterly with a ~60-day lag, so the newest mark in the freshest
# available dataset is already ~100 days old on release day and ~190 days old
# just before the next one. A wall-clock cap at 100 days therefore capped every
# row in the universe (8,909 of 8,968 on 2026-09-09). What the cap should catch
# is a bond whose OWN mark lags the rest of the dataset — no fund has priced it
# since — plus a hard backstop for when the whole dataset has gone stale.
STALE_MARK_LAG_DAYS = 35       # mark older than the dataset's newest -> HOLD
HARD_STALE_MARK_DAYS = 200     # any mark older than this -> HOLD
MIN_FUNDS_FOR_BUY = 3          # thinner fund coverage -> HOLD cap
MAX_PRICE_DISPERSION = 0.02    # cross-fund MAD/median above this -> HOLD cap

# Ageing a mark onto today's curve holds its Z-spread fixed. For ordinary paper
# the resulting price move is the rate move times duration, and it is large
# for long bonds: on 2026-09-13 absolute drift ran median 2.6pt, p99 5.4pt,
# alike for IG, HY and Treasuries, so a flat points cap would demote long IG
# for a normal rates rally. The failure is different: a short, distressed bond
# whose huge spread, held constant, manufactures a price (PDVSA marked 33.48,
# aged to 53.5). Drift PER YEAR OF DURATION separates the two cleanly — p99
# 1.4, p99.9 5.4, and the tail is all distressed short paper. Both conditions
# must hold; 16 rows on 2026-09-13, none rated above HOLD.
MIN_MARK_DRIFT_PTS = 3.0
MAX_MARK_DRIFT_PER_DURATION = 3.0
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

# Score cutpoints, high to low. Written by scripts/calibrate_credit.py: they are
# the credit-score quantiles that reproduce INDEX_RATING_MIX, subject to each
# bucket's median observed spread widening from AAA to CCC.
CREDIT_CUT_AAA = 99.0
CREDIT_CUT_AA = 83.8
CREDIT_CUT_A = 51.1
CREDIT_CUT_BBB = 32.9
CREDIT_CUT_BB = 23.9
CREDIT_CUT_B = 17.9

# The rating mix the cutpoints reproduce. EXTERNAL on purpose: the previous
# target was the model's own market_bucket counts, which came from anchors
# fitted on the model's own buckets, so calibration chased its own tail and
# piled ~950 bonds into B (median 113bp against a 290bp index) with 4 in CCC.
#
# Within-class shares: credit-quality breakdown of the two largest index
# trackers, as of 2026-09-10, renormalised over rated bonds only.
#   IG  iShares LQD: AAA 0.98, AA 12.00, A 46.58, BBB 39.96 (cash 0.48)
#   HY  iShares HYG: BB 57.74, B 31.93, CCC 7.58 + CC 0.35 + D 0.46
#       (BBB 0.99, NR 0.59, cash 0.36 excluded)
# The IG/HY split is NOT taken from the index: this universe is what funds
# hold, not what the index holds. It is measured from observed de-termed
# spreads against the BBB/BB index OAS boundary (models/credit.py).
INDEX_RATING_MIX = {
    'IG': {'AAA': 0.0098, 'AA': 0.1206, 'A': 0.4681, 'BBB': 0.4015},
    'HY': {'BB': 0.5888, 'B': 0.3256, 'CCC': 0.0856},
}

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
