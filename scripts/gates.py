"""Bond gate definitions: the asset-class-specific half of the scoring model.

Everything here is content; the machinery lives in scoring_kernel.py. One
GATES list drives both the pass/fail matrix and the continuous composite, so a
gate's threshold and its scoring range cannot drift apart.

THE APPLICABILITY PREDICATES ARE THE DESIGN
-------------------------------------------
A gate that returns False from `applicable` is excluded from the numerator AND
the denominator — the instrument is not penalised for failing a question that
cannot describe it. A gate that applies but has no data scores 0 and stays in
the denominator, so sparse rows are penalised.

That distinction is what lets Treasuries and corporate bonds share one rating
scale. A Treasury masks every credit gate, the whole Credit category loses its
applicable weight, and the composite renormalises over the categories that
remain. Ask a Treasury about its interest coverage and the answer is not
"bad", it is "wrong question".

The predicates are also DATA-driven, not just class-driven: a bond with no
fund-holding data has no measurable fund breadth, so those gates go
inapplicable rather than scoring zero. That matters at M3, where Treasuries
arrive from TreasuryDirect with no N-PORT marks attached yet.
"""

import math

from models.bond_types import infer_seniority
from models.schedule import years_to_maturity
from scripts.config import (FINANCIAL_SECTOR_NAME, MATURITY_BUCKETS,
                            MAX_FUNDAMENTALS_AGE_DAYS, MAX_PRICE_DISPERSION,
                            MIN_CUSIP_MATCH_CONFIDENCE, MIN_FUNDS_FOR_BUY,
                            SCORE_WEIGHT_CREDIT, SCORE_WEIGHT_LIQUIDITY,
                            SCORE_WEIGHT_RATES, SCORE_WEIGHT_STRUCTURE,
                            SCORE_WEIGHT_VALUATION, STALE_MARK_DAYS)
from scripts.scoring_kernel import (RATING_RANK, Gate, ScoringSpec,
                                    _score_linear)


# ---------------------------------------------------------------------------
# Applicability predicates
# ---------------------------------------------------------------------------

def _appl_credit(r):
    """Credit questions are meaningless for sovereign and agency paper.

    Not "the Treasury has excellent credit metrics" — it has none, because
    there is no issuer balance sheet whose leverage bears on repayment. This
    single predicate masks 15 of 26 gates for a Treasury.
    """
    return (r.get('asset_class') or '') not in ('TREASURY', 'TREASURY_BILL',
                                                'AGENCY')


def _appl_has_issuer(r):
    """Issuer fundamentals present AND confidently matched to this CUSIP.

    A low-confidence crosswalk match is worse than no match: it attaches some
    other company's leverage to this bond and produces a confidently wrong
    answer. Below the confidence floor the credit gates go inapplicable and a
    rating cap independently demotes the row.
    """
    conf = r.get('cusip_match_confidence')
    return (conf is not None and conf >= MIN_CUSIP_MATCH_CONFIDENCE
            and bool(r.get('_fundamentals_ok')))


def _appl_credit_fund(r):
    """Credit gates apply to any CORPORATE bond, whether or not we can price
    its issuer.

    This deliberately does NOT require the issuer to be identified, and the
    distinction is the whole applicable-vs-missing doctrine applied honestly.
    A Treasury has no issuer balance sheet — the question is meaningless, so
    the gate masks. A corporate bond ALWAYS has an issuer whose leverage
    matters; if we cannot identify it, that is missing data, and missing data
    scores zero and stays in the denominator.

    Getting this backwards inverted the model. Masking credit gates for
    unidentified issuers dropped the whole Credit category, whose gates
    average 33-48, and the composite renormalised over the remaining
    higher-scoring categories. Bonds we knew NOTHING about therefore rose to
    the top of the ranking — thirteen of the top fourteen names, all
    high-coupon high-yield paper with an unidentifiable issuer. The caps
    caught them, so nothing wrong shipped, but the ranking beneath the caps
    was upside down.
    """
    return _appl_credit(r)


def _appl_corp_nonfin(r):
    """Banks and insurers need the financial scorecard, not this one: their
    operating cash flow reflects deposit and loan movements, and EBITDA is not
    a meaningful denominator. Same reasoning as the equity model's
    _appl_non_financial mask.

    An unidentified issuer has no known sector and lands here, the modal case,
    where its missing metrics score zero rather than escaping assessment."""
    return (_appl_credit_fund(r)
            and r.get('issuer_sector') != FINANCIAL_SECTOR_NAME)


def _appl_corp_financial(r):
    # Requires a KNOWN financial issuer: CET1 and NPL are only meaningful for
    # a bank, and an unidentified issuer is assumed non-financial above.
    return (_appl_credit_fund(r) and _appl_has_issuer(r)
            and r.get('issuer_sector') == FINANCIAL_SECTOR_NAME)


def _appl_fixed_coupon(r):
    """Rates analytics need a fixed, known cashflow stream.

    A floater's duration is near zero by construction and its YTM is undefined
    without projecting a forward index; a TIPS coupon is a real rate with no
    inflation curve here to discount it. Scoring either on nominal duration
    would be arithmetic without meaning.
    """
    if r.get('is_inflation_linked') or r.get('is_convertible'):
        return False
    ctype = (r.get('coupon_type') or 'Fixed').strip().lower()
    return ctype in ('fixed', 'none', '')


def _appl_issuer_field(row, *fields):
    """Does this row's data vintage carry every field the gate needs?

    Distinct from the value being absent for one issuer. A field the snapshot
    never had is structurally unmeasurable — the 2026-04 equity vintage has no
    cet1_ratio at all — so scoring every matched issuer zero on it would drag
    the whole Credit category down while discriminating between nobody.
    """
    available = row.get('_issuer_fields')
    if available is None:
        return True          # unknown vintage: assume present, score normally
    # A tuple written to parquet comes back as a numpy array, where truthiness
    # raises rather than returning False. Normalise before testing.
    try:
        available = set(available)
    except TypeError:
        return True
    if not available:
        return True
    return all(f in available for f in fields)


def _appl_fcf_to_debt(r):
    return _appl_corp_nonfin(r) and _appl_issuer_field(r, 'fcf', 'total_debt')


def _appl_maturity_wall(r):
    return (_appl_credit_fund(r)
            and _appl_issuer_field(r, 'debt_maturity_wall_yrs'))


def _appl_cet1(r):
    return _appl_corp_financial(r) and _appl_issuer_field(r, 'cet1_ratio')


def _appl_npl(r):
    return _appl_corp_financial(r) and _appl_issuer_field(r, 'npl_ratio')


def _appl_issuer_trend(r):
    return _appl_credit_fund(r) and r.get('credit_score_trend') is not None


def _appl_fund_data(r):
    """Fund-holding metrics require fund holdings.

    A Treasury sourced from TreasuryDirect has no N-PORT marks attached, so
    its fund breadth is not zero — it is unmeasured. Scoring it zero would
    rank the most liquid instrument on earth as illiquid.
    """
    return r.get('n_funds') is not None


def _appl_mark_agreement(r):
    """Cross-fund mark dispersion needs several funds to disagree."""
    return (r.get('n_funds') or 0) >= 3


def _appl_amount_outstanding(r):
    return r.get('amount_outstanding_usd') is not None


# ---------------------------------------------------------------------------
# Custom score functions
# ---------------------------------------------------------------------------

def _score_duration_vs_regime(v, r, pct):
    """Is the market paying you to take duration right now?

    Deliberately NOT a rate forecast. A momentum rule ("yields fell, so buy
    duration") is a directional bet dressed as a screen, and this model has no
    business making one. Instead this asks whether duration is currently cheap
    on the same logic the valuation gates use:

      * a steep curve pays you to extend — you earn term premium and roll-down;
      * a high yield level relative to its own recent range means more room to
        fall than to rise, so the asymmetry favours duration.

    Those combine into an "appetite" in [0, 1]. At appetite 0.5 every duration
    scores 50 and the gate is silent, which is the correct behaviour when the
    curve is not saying anything.
    """
    if v is None or (isinstance(v, float) and v != v):
        return None
    regime = r.get('_curve_regime') or {}
    slope = regime.get('slope_10y_3m')
    level_pct = regime.get('level_pctile_1y')

    signals = []
    if slope is not None:
        signals.append(_score_linear(slope, -0.010, 0.020) / 100.0)
    if level_pct is not None:
        signals.append(level_pct / 100.0)
    appetite = sum(signals) / len(signals) if signals else 0.5

    # Normalise duration onto [0, 1]; 15 years is about the long end of what
    # a retail buyer holds.
    d = max(0.0, min(1.0, v / 15.0))
    return 100.0 * (d * appetite + (1.0 - d) * (1.0 - appetite))


def _percentile(v, r, pct):
    """For peer-ranked gates: the percentile is the score, but only when the
    underlying value actually exists. A NaN value must not inherit a rank."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    return pct


def _pass_through(v, r, pct):
    """For fields already expressed on the 0-100 scale.

    Clamped rather than returned raw. The kernel contract is that a score_fn
    yields 0-100, and a pass-through that trusts its input silently breaks
    that the moment upstream hands it something unexpected — the composite
    then goes out of range with nothing to show where it came from.
    """
    if v is None:
        return None
    try:
        value = float(v)
    except (TypeError, ValueError):
        return None
    if value != value:            # NaN is missing, not a perfect score
        return None
    return max(0.0, min(100.0, value))


def _score_carry_and_roll(v, r, pct):
    """Carry plus roll-down, benchmarked against just holding cash.

    The comparison matters: 400bp of carry is unimpressive when the 3-month
    bill pays 390bp. The gate scores the excess.
    """
    if v is None or (isinstance(v, float) and v != v):
        return None
    front = r.get('_front_end_yield')
    excess = v - front if front is not None else v
    return _score_linear(excess, -0.01, 0.03)


# ---------------------------------------------------------------------------
# GATES
# ---------------------------------------------------------------------------

_bp = 1e-4

GATES = [
    # ---- Valuation (0.32) ------------------------------------------------
    # Spread vs fair is the bond analogue of margin of safety and carries
    # double weight so it is not diluted to a fifth of the category.
    Gate('Valuation: Spread vs Fair', 'spread_mispricing',
         lambda v, r: v > 25 * _bp if v is not None else None,
         lambda v, r, pct: _score_linear(v, -150 * _bp, 250 * _bp),
         weight=2.0, applicable=_appl_credit_fund),
    Gate('Valuation: Price vs Fair', 'price_mispricing',
         lambda v, r: v > 0.02 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.08, 0.12),
         applicable=_appl_credit_fund),
    Gate('Valuation: Spread Percentile', 'z_spread',
         lambda v, r: v is not None and v > 0,
         _percentile,
         relative_mode='peer', higher_better=True, applicable=_appl_credit),
    Gate('Valuation: Yield over Tsy', 'yield_over_treasury',
         lambda v, r: v > 100 * _bp if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.0, 400 * _bp),
         applicable=_appl_credit),
    # Applies to Treasuries too: "am I paid to extend past cash" is a fair
    # question to ask a government bond.
    Gate('Valuation: Yield vs Cash', 'ytw_over_3m',
         lambda v, r: v > 0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -150 * _bp, 300 * _bp),
         applicable=_appl_fixed_coupon),

    # ---- Credit (0.28) ---------------------------------------------------
    Gate('Credit: Int Coverage', 'issuer_int_cov',
         lambda v, r: v > 3.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 1.0, 15.0),
         weight=1.5, applicable=_appl_corp_nonfin),
    Gate('Credit: Net Debt EBITDA', 'issuer_nd_ebitda',
         lambda v, r: v <= 3.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 6.0, 0.0),
         weight=1.5, applicable=_appl_corp_nonfin),
    Gate('Credit: FCF to Debt', 'issuer_fcf_to_debt',
         lambda v, r: v > 0.10 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.05, 0.35),
         applicable=_appl_fcf_to_debt),
    Gate('Credit: Altman Z', 'issuer_altman_z',
         lambda v, r: v > 2.6 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 1.1, 5.0),
         applicable=_appl_corp_nonfin),
    # Direction of travel beats level for a bondholder: a BBB deteriorating
    # toward BB loses more than a stable BB.
    Gate('Credit: Trend', 'credit_score_trend',
         lambda v, r: v >= 0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -12.0, 8.0),
         weight=1.5, applicable=_appl_issuer_trend),
    # The headline signal: fundamentals vs where the market prices the credit.
    Gate('Credit: Rating Divergence', 'bucket_divergence_notches',
         lambda v, r: v >= 1 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -2.0, 2.0),
         weight=1.5, applicable=_appl_credit_fund),
    Gate('Credit: CET1', 'issuer_cet1_ratio',
         lambda v, r: v > 0.11 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.06, 0.16),
         weight=1.5, applicable=_appl_cet1),
    Gate('Credit: NPL Ratio', 'issuer_npl_ratio',
         lambda v, r: v < 0.01 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.05, 0.003),
         weight=1.5, applicable=_appl_npl),

    # ---- Rates (0.16) ----------------------------------------------------
    Gate('Rates: Duration Fit', 'modified_duration',
         lambda v, r: v is not None,
         _score_duration_vs_regime,
         weight=1.5, applicable=_appl_fixed_coupon),
    # Per unit of duration, so a 30-year does not win on term alone.
    Gate('Rates: Convexity', 'convexity_per_duration',
         lambda v, r: v is not None and v > 0,
         _percentile,
         relative_mode='peer', higher_better=True, applicable=_appl_fixed_coupon),
    Gate('Rates: Roll Down', 'roll_down_12m',
         lambda v, r: v > 0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.01, 0.02),
         applicable=_appl_fixed_coupon),
    Gate('Rates: Carry and Roll', 'carry_roll_12m',
         lambda v, r: v > (r.get('_front_end_yield') or 0)
         if v is not None else None,
         _score_carry_and_roll,
         applicable=_appl_fixed_coupon),

    # ---- Structure (0.12) ------------------------------------------------
    Gate('Structure: Seniority', 'seniority_rank',
         lambda v, r: v <= 2 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 5.0, 1.0),
         applicable=_appl_credit),
    # Positive = the issuer's refinancing crunch lands AFTER this bond
    # matures, so you are repaid before the squeeze.
    Gate('Structure: Maturity Wall', 'wall_vs_own_maturity',
         lambda v, r: v > 0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -2.0, 3.0),
         applicable=_appl_maturity_wall),
    Gate('Structure: Payment Status', 'payment_status_score',
         lambda v, r: v >= 100 if v is not None else None,
         _pass_through,
         weight=2.0, applicable=_appl_credit),
    # Applies to everything, including Treasuries: "can this model honestly
    # price this instrument" is always a fair question.
    Gate('Structure: Analyzability', 'analyzability_score',
         lambda v, r: v >= 80 if v is not None else None,
         _pass_through),

    # ---- Liquidity (0.12) ------------------------------------------------
    Gate('Liquidity: Fund Breadth', 'n_funds',
         lambda v, r: v >= MIN_FUNDS_FOR_BUY if v is not None else None,
         lambda v, r, pct: _score_linear(v, 1.0, 30.0),
         weight=1.5, applicable=_appl_fund_data),
    Gate('Liquidity: Held Value', 'total_held_usd',
         lambda v, r: v is not None and v > 50e6,
         _percentile,
         relative_mode='peer', higher_better=True, applicable=_appl_fund_data),
    Gate('Liquidity: Mark Agreement', 'price_dispersion',
         lambda v, r: v < MAX_PRICE_DISPERSION if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.03, 0.0),
         applicable=_appl_mark_agreement),
    Gate('Liquidity: Valuation Level', 'fair_value_level',
         lambda v, r: v <= 2 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 3.0, 1.0),
         applicable=_appl_fund_data),
    # Amount outstanding: a genuine liquidity measure available for Treasuries
    # from TreasuryDirect, and the only one they have before N-PORT marks are
    # attached. log10 because the range spans $100m to $100bn.
    Gate('Liquidity: Issue Size', 'amount_outstanding_usd',
         lambda v, r: v is not None and v > 1e9,
         lambda v, r, pct: _score_linear(math.log10(max(v, 1.0)), 8.0, 11.0),
         applicable=_appl_amount_outstanding),
]


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def _maturity_bucket(years):
    if years is None:
        return 'unknown'
    for label, upper in MATURITY_BUCKETS:
        if upper is None or years < upper:
            return label
    return MATURITY_BUCKETS[-1][0]


def _credit_class(row):
    """Coarse class for peer pooling: TSY, IG or HY."""
    asset_class = row.get('asset_class') or ''
    if asset_class.startswith('TREASURY') or asset_class == 'AGENCY':
        return 'TSY'
    bucket = row.get('implied_bucket') or row.get('market_bucket')
    if bucket in ('BB', 'B', 'CCC'):
        return 'HY'
    if asset_class == 'CORP_HY':
        return 'HY'
    return 'IG'


def peer_group(row):
    """Pool label for relative_mode='peer'.

    Coarse on purpose. A finer key (sector x rating x maturity) would be more
    homogeneous but would leave many pools below the population needed for a
    percentile to mean anything, and the kernel would fall back to the global
    pool anyway. Twelve large stable pools beat two hundred sparse ones.
    """
    return f"{_credit_class(row)}|{_maturity_bucket(row.get('years_to_maturity'))}"


def _payment_status_score(row):
    """100 clean, 40 on a PIK toggle, 0 in default or arrears."""
    if row.get('is_default') or row.get('in_arrears'):
        return 0.0
    if row.get('is_paid_kind'):
        return 40.0
    return 100.0


def _analyzability_score(row):
    """How much of this instrument the model can honestly price.

    Deductions are for structure the model does not handle, not for the bond
    being bad. A convertible scores low here because the model cannot value
    the equity option — which is a statement about the model.
    """
    score = 100.0
    if row.get('is_inflation_linked'):
        score -= 40.0
    ctype = (row.get('coupon_type') or 'Fixed').strip().lower()
    if ctype not in ('fixed', 'none', ''):
        score -= 40.0
    if row.get('is_convertible'):
        score -= 40.0
    if row.get('ytm_solver_failed'):
        score -= 30.0
    # Terms sourced from a single fund with nothing to corroborate them.
    if row.get('n_funds') == 1 or row.get('_identity_conflict'):
        score -= 20.0
    if row.get('is_likely_callable') and not row.get('call_schedule'):
        score -= 10.0
    return max(0.0, score)


def prepare_scoring_fields(results):
    """Derive the shared fields the gates read. Idempotent.

    Runs before every scoring pass, including rescores of a stored snapshot,
    so it must recompute from primitives rather than trusting a stored derived
    value that may predate a change to this function.
    """
    for r in results:
        r['peer_group'] = peer_group(r)
        r['payment_status_score'] = _payment_status_score(r)
        r['analyzability_score'] = _analyzability_score(r)

        if r.get('seniority_rank') is None:
            rank, source = infer_seniority(r.get('title_of_issue'),
                                           r.get('payoff_profile'),
                                           r.get('issuer_cat'))
            r['seniority_rank'] = rank
            r['seniority_source'] = source

        # Convexity per unit duration, so term alone does not win the gate.
        dur, cvx = r.get('modified_duration'), r.get('convexity')
        r['convexity_per_duration'] = (cvx / dur if dur and cvx is not None
                                       and dur > 0.1 else None)

        # Positive means the issuer's refi wall lands after this bond matures.
        wall, ttm = r.get('issuer_debt_maturity_wall_yrs'), r.get('years_to_maturity')
        r['wall_vs_own_maturity'] = (wall - ttm if wall is not None
                                     and ttm is not None else None)

        # Yield pickup over cash.
        ytw, front = r.get('ytw'), r.get('_front_end_yield')
        r['ytw_over_3m'] = (ytw - front if ytw is not None
                            and front is not None else None)

        carry, roll = r.get('carry_12m'), r.get('roll_down_12m')
        r['carry_roll_12m'] = (carry + roll if carry is not None
                               and roll is not None else None)

        r['_fundamentals_ok'] = bool(
            r.get('issuer_cik') and r.get('_fundamentals_asof'))


# ---------------------------------------------------------------------------
# Rating caps
# ---------------------------------------------------------------------------

def rating_cap_for_row(row, params=None):
    """Investability caps. Returns (cap, reasons).

    Deliberately a longer list than the equity model's. Free bond data has
    more ways to be quietly wrong, and the entire point of the exercise is
    that the BUY list be actionable. A third to a half of rows carrying at
    least one cap is the expected outcome, not a bug — a capped row still
    shows its uncapped rating_raw, so nothing is hidden, it is qualified.
    """
    p = params or {}
    reasons = []
    cap = None

    def add(new_cap, reason):
        nonlocal cap
        if cap is None or RATING_RANK[new_cap] < RATING_RANK[cap]:
            cap = new_cap
        reasons.append(reason)

    # -- the issuer is not paying ------------------------------------------
    if row.get('is_default'):
        add('PASS', 'issuer in default (N-PORT)')
    if row.get('in_arrears'):
        add('PASS', 'interest payments in arrears')

    # The analogue of the equity model's MoS <= -20% cap.
    mis = row.get('spread_mispricing')
    if mis is not None and mis <= -150 * _bp:
        add('PASS', 'trading 150bp+ through fair spread')

    if row.get('is_paid_kind'):
        add('HOLD', 'PIK toggle — coupon may be paid in kind')

    # -- the model cannot price it -----------------------------------------
    if row.get('clean_price_est') is None and row.get('clean_price_marked') is None:
        add('HOLD', 'no usable price mark')
    if row.get('ytm_solver_failed'):
        add('HOLD', 'yield solver did not converge')
    if row.get('is_inflation_linked'):
        add('HOLD', 'inflation-linked — priced nominally, understated')
    ctype = (row.get('coupon_type') or 'Fixed').strip().lower()
    if ctype not in ('fixed', 'none', ''):
        add('HOLD', f'non-fixed coupon ({ctype})')
    if row.get('is_convertible'):
        add('HOLD', 'convertible — equity option not modelled')

    # The OAS gap made explicit. A premium-priced probable callable is exactly
    # where Z-spread overstates compensation, so the honest answer is "cannot
    # tell" rather than a confident BUY.
    price = row.get('clean_price_est') or row.get('clean_price_marked')
    if (row.get('is_likely_callable') and not row.get('call_schedule')
            and price is not None and price > 100.5):
        add('HOLD', 'callable above par with unknown call schedule')

    # -- the data is too old or too thin -----------------------------------
    age = row.get('mark_age_days')
    if age is not None and age > p.get('stale_mark_days', STALE_MARK_DAYS):
        add('HOLD', f'stale mark ({age}d)')

    n_funds = row.get('n_funds')
    if n_funds is not None and n_funds < p.get('min_funds_for_buy',
                                               MIN_FUNDS_FOR_BUY):
        add('HOLD', f'thin fund coverage ({n_funds})')

    disp = row.get('price_dispersion')
    if disp is not None and disp > p.get('max_price_dispersion',
                                         MAX_PRICE_DISPERSION):
        add('HOLD', f'cross-fund mark disagreement ({disp:.1%})')

    if row.get('fair_value_level') == 3:
        add('HOLD', 'level-3 (unobservable) valuation')

    # -- the issuer attribution is doubtful --------------------------------
    conf = row.get('cusip_match_confidence')
    if conf is not None and conf < p.get('min_cusip_match_confidence',
                                         MIN_CUSIP_MATCH_CONFIDENCE):
        add('HOLD', f'low issuer match confidence ({conf:.2f})')

    fund_age = row.get('_fundamentals_age_days')
    if fund_age is not None and fund_age > p.get('max_fundamentals_age_days',
                                                 MAX_FUNDAMENTALS_AGE_DAYS):
        add('HOLD', f'stale issuer fundamentals ({fund_age}d)')

    if row.get('issuer_altman_z_zone') == 'distress':
        add('HOLD', 'Altman Z distress zone')

    coverage = row.get('_data_coverage_score')
    if coverage is not None and coverage < 25:
        add('HOLD', 'low scoring data coverage')

    # -- not really a bond decision any more -------------------------------
    ttm = row.get('years_to_maturity')
    if ttm is not None and ttm < 0.5:
        add('HOLD', 'inside 6 months — this is a cash decision')

    return cap, reasons


# ---------------------------------------------------------------------------
# Display metadata
# ---------------------------------------------------------------------------

GATE_DISPLAY = {
    'spread_vs_fair': {'label': 'Spread vs Fair', 'threshold': '> +25bp', 'fmt': 'bp'},
    'price_vs_fair': {'label': 'Price vs Fair', 'threshold': '> +2%', 'fmt': 'pct1'},
    'spread_percentile': {'label': 'Spread %ile', 'threshold': 'vs peers', 'fmt': 'bp'},
    'yield_over_tsy': {'label': 'Yield o/ Tsy', 'threshold': '> 100bp', 'fmt': 'bp'},
    'yield_vs_cash': {'label': 'Yield vs Cash', 'threshold': '> 3m bill', 'fmt': 'bp'},
    'int_coverage': {'label': 'Int Cov', 'threshold': '> 3x', 'fmt': 'ratio'},
    'net_debt_ebitda': {'label': 'ND/EBITDA', 'threshold': '<= 3x', 'fmt': 'ratio'},
    'fcf_to_debt': {'label': 'FCF/Debt', 'threshold': '> 10%', 'fmt': 'pct1'},
    'altman_z': {'label': 'Altman Z', 'threshold': '> 2.6', 'fmt': 'n2'},
    'trend': {'label': 'Credit Trend', 'threshold': 'not deteriorating', 'fmt': 'n1'},
    'rating_divergence': {'label': 'Divergence', 'threshold': '>= +1 notch', 'fmt': 'n1'},
    'cet1': {'label': 'CET1', 'threshold': '> 11%', 'fmt': 'pct1'},
    'npl_ratio': {'label': 'NPL', 'threshold': '< 1%', 'fmt': 'pct1'},
    'duration_fit': {'label': 'Duration Fit', 'threshold': 'vs curve regime', 'fmt': 'n2'},
    'convexity': {'label': 'Convexity/Dur', 'threshold': 'vs peers', 'fmt': 'n2'},
    'roll_down': {'label': 'Roll-Down', 'threshold': '> 0', 'fmt': 'bp'},
    'carry_and_roll': {'label': 'Carry + Roll', 'threshold': '> 3m bill', 'fmt': 'bp'},
    'seniority': {'label': 'Seniority', 'threshold': 'senior or better', 'fmt': 'int'},
    'maturity_wall': {'label': 'Wall vs Mat', 'threshold': '> 0 yrs', 'fmt': 'n1'},
    'payment_status': {'label': 'Pmt Status', 'threshold': 'current', 'fmt': 'int'},
    'analyzability': {'label': 'Analyzable', 'threshold': '>= 80', 'fmt': 'int'},
    'fund_breadth': {'label': 'Funds', 'threshold': f'>= {MIN_FUNDS_FOR_BUY}', 'fmt': 'int'},
    'held_value': {'label': 'Held $', 'threshold': '> $50m', 'fmt': 'ds'},
    'mark_agreement': {'label': 'Mark Agree', 'threshold': '< 2% MAD', 'fmt': 'pct1'},
    'valuation_level': {'label': 'FV Level', 'threshold': '<= 2', 'fmt': 'int'},
    'issue_size': {'label': 'Issue Size', 'threshold': '> $1bn', 'fmt': 'ds'},
}

CATEGORY_DISPLAY = {
    'Valuation': {'dark': '#2F5496', 'light': '#D6E4F0'},
    'Credit': {'dark': '#C55A11', 'light': '#FCE4CC'},
    'Rates': {'dark': '#548235', 'light': '#E2EFDA'},
    'Structure': {'dark': '#7030A0', 'light': '#E4CCEF'},
    'Liquidity': {'dark': '#BF8F00', 'light': '#FFF2CC'},
}

CATEGORY_WEIGHTS = {
    'Valuation': ('score_weight_valuation', SCORE_WEIGHT_VALUATION),
    'Credit': ('score_weight_credit', SCORE_WEIGHT_CREDIT),
    'Rates': ('score_weight_rates', SCORE_WEIGHT_RATES),
    'Structure': ('score_weight_structure', SCORE_WEIGHT_STRUCTURE),
    'Liquidity': ('score_weight_liquidity', SCORE_WEIGHT_LIQUIDITY),
}

CATEGORY_ORDER = ['Valuation', 'Credit', 'Rates', 'Structure', 'Liquidity']

SPEC = ScoringSpec(
    gates=GATES,
    category_weights=CATEGORY_WEIGHTS,
    category_order=CATEGORY_ORDER,
    gate_display=GATE_DISPLAY,
    category_display=CATEGORY_DISPLAY,
    prepare_fn=prepare_scoring_fields,
    cap_fn=rating_cap_for_row,
)
