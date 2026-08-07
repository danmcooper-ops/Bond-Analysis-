"""Issuer credit quality, fair spread, and the divergence signal.

This is where a bond gets a valuation. A stock has an intrinsic value you can
discount cash flows toward; a bond's intrinsic value is par plus its coupons,
and the entire question is whether the spread compensates for the chance of
not being paid. So "fair value" here means FAIR SPREAD, and the model is a
relative-value one: given this issuer's financial condition, what spread
should this bond trade at, and what does it actually trade at?

THE CHAIN
---------
    issuer financials -> credit score -> implied rating bucket
    bucket + maturity -> fair spread   (FRED bucket OAS x term factor + wedge)
    fair spread       -> fair price
    observed - fair   -> the mispricing signal

A TRANSPARENT SCORECARD, NOT A MODEL YOU CANNOT ARGUE WITH. Six weighted
factors, each mapped 0-100 by the same `_score_linear` the gates use. The
point is that a disagreement is checkable: you can look at a BBB call and see
it came from 4.2x coverage and 2.1x leverage, rather than from a fitted
surface nobody can inspect.

CALIBRATED AGAINST THE MARKET, NOT AGAINST AGENCY RATINGS, because there are
no free agency ratings. The cutpoints are chosen so each implied bucket's
median observed spread lines up with the published bucket OAS. That makes the
implied bucket mean "where the market prices issuers that look like this",
which is exactly the reference the mispricing signal needs — and it is
self-consistent rather than borrowed.

WHAT DIVERGENCE IS, AND ITS FAILURE MODE
-----------------------------------------
Divergence compares the bucket the fundamentals imply against the bucket the
bond's own spread implies. Positive means the market prices the credit worse
than the financials suggest (a rising star); negative means better (fallen
angel risk).

The failure mode is unavoidable and must be stated: with a monthly mark
arriving ~60 days late, "the market has not caught up to the deterioration" is
frequently "OUR DATA has not caught up". A divergence is only reported as a
fallen-angel signal when the fundamentals predate the mark — otherwise we are
comparing a fresh balance sheet against a stale price and calling the lag a
signal.
"""

import math

from scripts.config import (CREDIT_BUCKETS, CREDIT_FACTORS_CORPORATE,
                            CREDIT_FACTORS_FINANCIAL, FINANCIAL_SECTOR_NAME)
from scripts.scoring_kernel import _score_linear

BUCKET_RANK = {b: i for i, b in enumerate(CREDIT_BUCKETS)}   # 0 = best
RANK_BUCKET = {i: b for b, i in BUCKET_RANK.items()}

# Buckets at or below this rank are high yield.
HY_FLOOR_RANK = BUCKET_RANK['BB']

CUTPOINT_PARAMS = ('credit_cut_aaa', 'credit_cut_aa', 'credit_cut_a',
                   'credit_cut_bbb', 'credit_cut_bb', 'credit_cut_b')

# A scorecard resting on one or two factors is not a credit opinion.
MIN_FACTOR_COVERAGE = 0.50


def _factor_value(field, fundamentals):
    """Read one scorecard input, deriving the two that are not stored."""
    if field == 'fcf_to_debt':
        fcf, debt = fundamentals.get('fcf'), fundamentals.get('total_debt')
        if fcf is None or not debt or debt <= 0:
            return None
        return fcf / debt
    if field == 'log_revenue':
        revenue = fundamentals.get('revenue')
        if revenue is None or revenue <= 0:
            return None
        return math.log10(revenue)
    return fundamentals.get(field)


def credit_score(fundamentals, sector=None):
    """Weighted 0-100 credit score. Returns the score and its provenance.

    Reweights over the factors that are PRESENT rather than scoring a missing
    factor as zero. Missing leverage data is not evidence of bad leverage, and
    treating it as such would rank every thinly-covered issuer as distressed.
    The `coverage` figure carries that uncertainty forward instead, and the
    caller refuses to call a bucket below MIN_FACTOR_COVERAGE.
    """
    if not fundamentals:
        return {'score': None, 'coverage': 0.0, 'factors': {},
                'scorecard': None}

    is_financial = (sector or fundamentals.get('sector')) == FINANCIAL_SECTOR_NAME
    spec = CREDIT_FACTORS_FINANCIAL if is_financial else CREDIT_FACTORS_CORPORATE

    total_weight = sum(w for _, _, _, w in spec)
    used_weight = 0.0
    weighted = 0.0
    detail = {}

    for field, worst, best, weight in spec:
        value = _factor_value(field, fundamentals)
        if value is None:
            detail[field] = None
            continue
        points = _score_linear(value, worst, best)
        if points is None:
            detail[field] = None
            continue
        detail[field] = {'value': value, 'score': round(points, 1),
                         'weight': weight}
        weighted += points * weight
        used_weight += weight

    if used_weight <= 0:
        return {'score': None, 'coverage': 0.0, 'factors': detail,
                'scorecard': 'financial' if is_financial else 'corporate'}

    return {
        'score': round(weighted / used_weight, 1),
        'coverage': round(used_weight / total_weight, 3),
        'factors': detail,
        'scorecard': 'financial' if is_financial else 'corporate',
    }


def bucket_from_score(score, params=None):
    """Map a 0-100 credit score onto a rating bucket."""
    if score is None:
        return None
    p = params or {}
    cuts = [p.get(key) for key in CUTPOINT_PARAMS]
    from scripts.config import (CREDIT_CUT_A, CREDIT_CUT_AA, CREDIT_CUT_AAA,
                                CREDIT_CUT_B, CREDIT_CUT_BB, CREDIT_CUT_BBB)
    defaults = [CREDIT_CUT_AAA, CREDIT_CUT_AA, CREDIT_CUT_A, CREDIT_CUT_BBB,
                CREDIT_CUT_BB, CREDIT_CUT_B]
    cuts = [c if c is not None else d for c, d in zip(cuts, defaults)]

    for bucket, cut in zip(CREDIT_BUCKETS[:-1], cuts):
        if score >= cut:
            return bucket
    return CREDIT_BUCKETS[-1]


def implied_bucket(fundamentals, sector=None, params=None):
    """Fundamentals -> rating bucket, with the score and coverage behind it."""
    result = credit_score(fundamentals, sector=sector)
    coverage = result.get('coverage') or 0.0
    if result['score'] is None or coverage < MIN_FACTOR_COVERAGE:
        result['bucket'] = None
        result['confident'] = False
        return result
    result['bucket'] = bucket_from_score(result['score'], params)
    result['confident'] = coverage >= 0.75
    return result


def is_high_yield(bucket):
    return bucket is not None and BUCKET_RANK[bucket] >= HY_FLOOR_RANK


def asset_class_for(bucket):
    """IG or HY, defaulting to IG when the credit is unknown.

    Defaulting to IG is the conservative choice for peer pooling: it puts an
    unknown credit among tighter-spread names, so a wide spread stands out as
    unusual rather than blending into a high-yield pool.
    """
    return 'CORP_HY' if is_high_yield(bucket) else 'CORP_IG'


# ---------------------------------------------------------------------------
# Fair spread
# ---------------------------------------------------------------------------

def fair_spread(bucket, maturity_years, bucket_oas, term_points=None,
                wedge=None, beta=1.0, term_by_bucket=None):
    """The spread this bond should trade at, per the market's own pricing.

        fair = bucket OAS  x  term factor(maturity)  +  wedge(bucket)

    The TERM FACTOR matters and is often skipped: the published bucket OAS is
    a whole-index number with a duration around seven years, so using it flat
    tells you a 2-year BBB and a 30-year BBB deserve the same spread. They do
    not — the observed IG term structure runs 0.59x at two years to 1.27x at
    thirty. Skipping it makes every short bond look rich and every long bond
    cheap, which is a term-structure artifact dressed as a credit signal.

    THE TERM FACTOR IS MEASURED, NOT BORROWED. `term_points` should come from
    scripts/fit_term_structure.py, which fits the shape from ~130,000 observed
    investment-grade spreads in the N-PORT panel. It falls back to FRED's IG
    maturity slices when no fit exists.

    The distinction matters more than it sounds. FRED's slices stop at "15y+",
    so the factor was flat-extrapolated past twenty years, and they are
    SUB-INDICES WITH DIFFERENT CONSTITUENTS — only the strongest issuers sell
    forty-year paper, so ratio-ing one slice to another silently compares
    different populations. Measured across a single population the curve is
    nearly flat beyond seven years (1.08-1.20x), where FRED climbs to 1.27x.
    The two agree to within 9% over three to ten years, which is what makes
    the divergence past that believable rather than just our own pricing noise.

    The old assumption therefore OVERSTATED long-dated fair spreads by up to
    17%, making long bonds look richer than they were — the opposite of the
    bias originally suspected.

    The WEDGE corrects Z-spread against OAS. We compute Z-spreads and compare
    them to an OAS index; for callable paper Z exceeds OAS by roughly the
    value of the call. The wedge is FITTED from observed history by
    spreads.fit_z_oas_wedge, never assumed.
    """
    if bucket is None or maturity_years is None:
        return None
    base = (bucket_oas or {}).get(bucket)
    if base is None:
        return None

    # A per-bucket curve when one has been fitted, else the shared one. The
    # distinction is large: measured across 72,000 observations, tight and mid
    # credits rise with maturity while WIDE ones invert (0.95x short to 0.84x
    # long), because a struggling issuer's problem is the next maturity rather
    # than the one in twenty years. One shared rising curve gets high yield
    # backwards by roughly 47%.
    points = None
    if term_by_bucket:
        points = term_by_bucket.get(bucket)
    if points is None:
        points = term_points

    factor = 1.0
    if points:
        from data.fred_client import term_factor_at
        factor = term_factor_at(points, maturity_years, beta=beta)

    adjustment = 0.0
    if wedge:
        entry = wedge.get(bucket)
        if isinstance(entry, dict):
            adjustment = entry.get('wedge') or 0.0
        elif entry is not None:
            adjustment = entry
    return base * factor + adjustment


def fair_price(flows, settle, curve, fair_z):
    """Dirty price implied by the fair spread."""
    if fair_z is None:
        return None
    from models.spreads import spread_to_price
    return spread_to_price(fair_z, flows, settle, curve)


def spread_mispricing(observed_z, fair_z):
    """Observed minus fair spread. POSITIVE MEANS CHEAP.

    The sign convention is the opposite of the price one and worth stating:
    a wider spread than deserved is a bond you are overpaid to own.
    """
    if observed_z is None or fair_z is None:
        return None
    return observed_z - fair_z


def price_mispricing(observed_clean, fair_clean):
    """Fair over observed, minus one. Positive means cheap, as with equities."""
    if observed_clean is None or fair_clean is None or observed_clean <= 0:
        return None
    return fair_clean / observed_clean - 1.0


def market_implied_bucket(observed_z, maturity_years, bucket_oas,
                          term_points=None, wedge=None, beta=1.0,
                          term_by_bucket=None):
    """Which bucket's fair spread best explains this bond's actual spread?

    The inverse of fair_spread: instead of asking what an A-rated issuer
    should pay, ask what rating the market is charging this bond for. Compared
    against the fundamental bucket, the gap is the divergence signal.
    """
    if observed_z is None:
        return None
    best, best_gap = None, None
    for bucket in CREDIT_BUCKETS:
        implied = fair_spread(bucket, maturity_years, bucket_oas,
                              term_points=term_points, wedge=wedge, beta=beta,
                              term_by_bucket=term_by_bucket)
        if implied is None:
            continue
        gap = abs(observed_z - implied)
        if best_gap is None or gap < best_gap:
            best, best_gap = bucket, gap
    return best


def divergence(fundamental_bucket, market_bucket, fundamentals_asof=None,
               mark_date=None):
    """Notches between where fundamentals and where the market place a credit.

        positive  market prices it WORSE than the financials  -> rising star
        negative  market prices it BETTER                     -> fallen angel

    `stale_risk` is set when the fundamentals are NEWER than the mark. In that
    case the two sides are not describing the same moment: a fresh balance
    sheet against a two-month-old price will show apparent divergence purely
    from the lag, and calling that a fallen angel is reading our own data
    latency as a market signal. The flag does not suppress the number — it
    tells the gate layer and the reader not to trust it as a credit call.
    """
    if fundamental_bucket is None or market_bucket is None:
        return {'notches': None, 'label': None, 'stale_risk': False}

    notches = BUCKET_RANK[market_bucket] - BUCKET_RANK[fundamental_bucket]
    if notches >= 1:
        label = 'rising_star'
    elif notches <= -1:
        label = 'fallen_angel_risk'
    else:
        label = 'aligned'

    stale_risk = False
    if fundamentals_asof and mark_date:
        try:
            from datetime import datetime
            asof = datetime.strptime(str(fundamentals_asof)[:10], '%Y-%m-%d').date()
            stale_risk = asof > mark_date
        except (ValueError, TypeError):
            stale_risk = False

    return {'notches': notches, 'label': label, 'stale_risk': stale_risk}


def bucket_trend(score_now, score_prior, years=1.0):
    """Change in credit score per year. Negative is deterioration.

    Direction of travel beats level for a bondholder: a BBB sliding toward BB
    loses more than a stable BB, because the loss comes from repricing rather
    than from carry.
    """
    if score_now is None or score_prior is None or years <= 0:
        return None
    return (score_now - score_prior) / years


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_cutpoints(rows, bucket_oas, term_points=None, wedge=None,
                        beta=1.0, min_per_bucket=25):
    """Choose cutpoints so each implied bucket's spread matches the market's.

    Rather than fitting six free parameters, this assigns each bond the bucket
    whose fair spread its OWN observed spread is closest to, then reads off
    the credit-score quantiles those assignments imply. The result is a
    scorecard whose buckets mean what the market means by them.

    Returns {param_name: cutpoint} for the buckets with enough population, and
    leaves the rest at their defaults — a cutpoint fitted to nine bonds is
    worse than the seed value.
    """
    by_bucket = {}
    for row in rows:
        score = row.get('issuer_credit_score')
        z = row.get('z_spread')
        ttm = row.get('years_to_maturity')
        if score is None or z is None or ttm is None:
            continue
        market = market_implied_bucket(z, ttm, bucket_oas,
                                       term_points=term_points, wedge=wedge,
                                       beta=beta)
        if market:
            by_bucket.setdefault(market, []).append(score)

    out = {}
    # Walk best to worst; each cutpoint is the score below which the market
    # stops treating an issuer as that bucket.
    for bucket, param in zip(CREDIT_BUCKETS[:-1], CUTPOINT_PARAMS):
        scores = sorted(by_bucket.get(bucket, []))
        if len(scores) < min_per_bucket:
            continue
        out[param] = round(scores[len(scores) // 10], 1)   # 10th percentile

    # Cutpoints must strictly decrease or the scale inverts. Enforce here
    # rather than letting validate_params reject the whole set later.
    ordered = {}
    ceiling = 100.0
    for param in CUTPOINT_PARAMS:
        value = out.get(param)
        if value is None:
            continue
        value = min(value, ceiling - 1.0)
        ordered[param] = round(value, 1)
        ceiling = value
    return ordered
