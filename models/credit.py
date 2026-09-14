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
no free agency ratings. The cutpoints reproduce the index rating mix
(config.INDEX_RATING_MIX within IG and HY, with this universe's own IG/HY
split measured from spreads), and calibration refuses a set whose buckets do
not widen in median spread from AAA to CCC. The target is external on
purpose: it used to be the model's own market buckets, which made calibration
circular.

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
    """Read one scorecard input, deriving those that are not stored directly."""
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
    if field == 'log_mcap':
        mcap = fundamentals.get('mcap')
        if mcap is None or mcap <= 0:
            return None
        return math.log10(mcap)
    if field == 'mcap_to_debt':
        # The structural (Merton) leverage measure: how much equity cushion
        # sits above the debt, at market value rather than book. Logged
        # because the ratio spans several orders of magnitude — a debt-free
        # issuer and a barely-solvent one are not two points on a linear
        # scale. A debt-free issuer has no leverage story at all rather than
        # an infinitely good one, so it returns None.
        mcap, debt = fundamentals.get('mcap'), fundamentals.get('total_debt')
        if mcap is None or debt is None or mcap <= 0 or debt <= 0:
            return None
        return math.log10(mcap / debt)
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
                wedge=None, beta=1.0, term_by_bucket=None,
                bucket_anchors=None):
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

    # THE ANCHOR IS THE MODEL'S OWN BUCKET, NOT AN EXTERNAL INDEX.
    #
    # FRED's AAA index is a handful of genuinely AAA issuers trading at 38bp.
    # This model's AAA bucket is several hundred merely-excellent ones trading
    # at 53bp. Pricing the second population off the first is a population
    # mismatch — the same error found in FRED's maturity slices, one level up —
    # and it is severe in high yield, where the model's B bucket trades at
    # 142bp against an index at 290bp.
    #
    # Anchoring each bucket on the de-termed median spread of ITS OWN members
    # makes the question self-consistent: not "is this bond cheap against an
    # index of different bonds", but "is it cheap against bonds the model
    # judges to be of the same quality and tenor". That is the relative-value
    # question the model is actually equipped to answer.
    #
    # The cost is that absolute level information is gone: by construction
    # roughly half of each bucket is cheap and half rich. The signal becomes
    # purely cross-sectional, which is honest about what it measures. Thin
    # buckets fall back to the index, since a median over three bonds is not
    # an anchor.
    base = (bucket_anchors or {}).get(bucket)
    if base is None:
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
                          term_by_bucket=None, bucket_anchors=None):
    """Which bucket's fair spread best explains this bond's actual spread?

    The inverse of fair_spread: instead of asking what an A-rated issuer
    should pay, ask what rating the market is charging this bond for. Compared
    against the fundamental bucket, the gap is the divergence signal.
    """
    if observed_z is None:
        return None
    levels = []
    for bucket in CREDIT_BUCKETS:
        implied = fair_spread(bucket, maturity_years, bucket_oas,
                              term_points=term_points, wedge=wedge, beta=beta,
                              term_by_bucket=term_by_bucket,
                              bucket_anchors=bucket_anchors)
        if implied is None or implied <= 0:
            continue
        # Per-tier term curves can cross at long tenors (a mid-tier BBB above
        # a wide-tier BB); a running maximum keeps the ladder monotone.
        if levels:
            implied = max(implied, levels[-1][1])
        levels.append((bucket, implied))
    if not levels:
        return None
    # Boundaries at the GEOMETRIC midpoint between neighbouring buckets.
    # Nearest absolute distance put the B/CCC boundary halfway between 129bp
    # and 1023bp, so everything up to ~575bp read as B; spreads are
    # multiplicative, and the geometric midpoint splits the ladder evenly.
    for (bucket, level), (_, next_level) in zip(levels, levels[1:]):
        if observed_z <= math.sqrt(level * next_level):
            return bucket
    return levels[-1][0]


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

def _determed(row, term_points=None, term_by_bucket=None, bucket=None):
    """Observed Z-spread divided by its term factor, or None."""
    spread, tenor = row.get('z_spread'), row.get('years_to_maturity')
    if spread is None or tenor is None:
        return None
    points = (term_by_bucket or {}).get(bucket) if bucket else None
    points = points or term_points
    if not points:
        return spread
    from data.fred_client import term_factor_at
    return spread / max(term_factor_at(points, tenor), 0.2)


def target_rating_mix(rows, bucket_oas, term_points=None, index_mix=None):
    """{bucket: share} this universe should reproduce.

    Within investment grade and within high yield the shares are the index's
    (INDEX_RATING_MIX). The IG/HY SPLIT is this universe's own: the share of
    scored bonds whose de-termed spread sits beyond the BBB/BB boundary, the
    geometric midpoint of the two index OAS levels. Taking the split from the
    index would impose the index's composition on a fund-held universe that is
    measurably different; taking the within-class shares from our own buckets
    is the circularity this replaces.
    """
    from scripts.config import INDEX_RATING_MIX
    index_mix = index_mix or INDEX_RATING_MIX
    bbb, bb = (bucket_oas or {}).get('BBB'), (bucket_oas or {}).get('BB')
    if not bbb or not bb:
        return {}
    boundary = math.sqrt(bbb * bb)
    spreads = [_determed(r, term_points) for r in rows
               if r.get('issuer_credit_score') is not None]
    spreads = [x for x in spreads if x is not None]
    if not spreads:
        return {}
    hy_share = sum(1 for x in spreads if x > boundary) / len(spreads)
    mix = {b: w * (1.0 - hy_share) for b, w in index_mix['IG'].items()}
    mix.update({b: w * hy_share for b, w in index_mix['HY'].items()})
    return mix


def calibrate_cutpoints(rows, target_mix, min_rows=300):
    """Place the cutpoints so the model's bucket mix reproduces `target_mix`.

    The scorecard RANKS issuers; the cutpoints only decide where the labels
    fall. Each cutpoint is the credit-score quantile at the cumulative target
    share, walking best to worst, so the implied mix matches the target by
    construction.

    THE TARGET IS EXTERNAL. It used to be the mix of each bond's
    market_bucket — read off anchors fitted on the model's own implied
    buckets, which were set by the previous cutpoints. Calibration therefore
    chased its own tail: ~950 bonds ended in B at a 113bp median against a
    290bp index, and CCC held four. See target_rating_mix().

    Deliberately NOT fitted to forward returns: this aligns labels, and a
    return-fitted cutpoint would smuggle in look-ahead.
    """
    scores = sorted(r['issuer_credit_score'] for r in rows
                    if r.get('issuer_credit_score') is not None)
    if len(scores) < min_rows or not target_mix:
        return {}
    total = sum(target_mix.get(b, 0.0) for b in CREDIT_BUCKETS) or 1.0

    out, cumulative = {}, 0.0
    for bucket, param in zip(CREDIT_BUCKETS[:-1], CUTPOINT_PARAMS):
        cumulative += target_mix.get(bucket, 0.0) / total
        index = int(round((1.0 - cumulative) * (len(scores) - 1)))
        out[param] = round(scores[max(0, min(index, len(scores) - 1))], 1)

    # Cutpoints must strictly decrease or the scale inverts. Enforce here
    # rather than letting validate_params reject the whole set later.
    ordered, ceiling = {}, 100.0
    for param in CUTPOINT_PARAMS:
        value = min(out[param], ceiling - 0.1)
        ordered[param] = round(value, 1)
        ceiling = value
    return ordered


def _issuer_key(row):
    return row.get('issuer_ticker') or (row.get('cusip') or '')[:6] or None


def bucket_spread_medians(rows, cuts, term_points=None, min_n=10,
                          min_issuers=1):
    """{bucket: (n_bonds, median de-termed spread, n_issuers)} under `cuts`.

    Buckets with fewer than `min_n` bonds or `min_issuers` distinct issuers
    are omitted.
    """
    by_bucket, issuers = {}, {}
    for row in rows:
        score = row.get('issuer_credit_score')
        if score is None:
            continue
        spread = _determed(row, term_points)
        if spread is None:
            continue
        bucket = bucket_from_score(score, cuts)
        by_bucket.setdefault(bucket, []).append(spread)
        issuers.setdefault(bucket, set()).add(_issuer_key(row))
    out = {}
    for bucket in CREDIT_BUCKETS:
        values = sorted(by_bucket.get(bucket, []))
        n_issuers = len(issuers.get(bucket, set()) - {None})
        if len(values) >= min_n and n_issuers >= min_issuers:
            out[bucket] = (len(values), values[len(values) // 2], n_issuers)
    return out


def check_bucket_spread_order(rows, cuts, term_points=None, min_n=10,
                              min_issuers=10):
    """Adjacent bucket pairs whose median spread does not widen. [] is healthy.

    A label mix can match the index while the scorecard fails to separate two
    grades; this is the check that the labels still MEAN something about risk.

    Only buckets with at least `min_issuers` distinct issuers are compared. A
    median over a handful of issuers measures those issuers: on 2026-09-13
    the AAA bucket was four names (Amazon, Meta, Nvidia, Microsoft), trading
    wide on heavy AI-capex issuance, and "AAA wider than AA" was an issuer
    effect, not a scorecard failure.
    """
    medians = bucket_spread_medians(rows, cuts, term_points, min_n, min_issuers)
    present = [b for b in CREDIT_BUCKETS if b in medians]
    return [(a, b, medians[a][1], medians[b][1])
            for a, b in zip(present, present[1:])
            if not medians[b][1] > medians[a][1]]


def fit_bucket_anchors(rows, term_points=None, term_by_bucket=None,
                       min_per_bucket=50, reference_years=5.0,
                       min_per_bucket_ccc=15, bucket_oas=None,
                       min_issuers=0):
    """Median spread of each bucket's own members, de-termed to a reference tenor.

    Each observed spread is divided by its own term factor before the median
    is taken, so a bucket whose members skew long does not inherit a wider
    anchor purely from tenor. The result is a level per bucket that
    `fair_spread` then re-terms for the bond being priced.

    Buckets thinner than `min_per_bucket` are omitted rather than fitted: a
    median over three bonds is noise, and `fair_spread` falls back to the
    published index for those.

    Returns {bucket: anchor_spread}, plus diagnostics under '_meta'.
    """
    from data.fred_client import term_factor_at

    by_bucket, issuers = {}, {}
    for row in rows:
        bucket = row.get('implied_bucket')
        spread = row.get('z_spread')
        tenor = row.get('years_to_maturity')
        if not bucket or spread is None or tenor is None:
            continue
        if not (0.0 < spread < 0.50):
            continue
        points = (term_by_bucket or {}).get(bucket) or term_points
        factor = term_factor_at(points, tenor) if points else 1.0
        # A near-zero factor would explode the de-termed value; the floor is a
        # guard against a degenerate fitted curve, not a tuning knob.
        by_bucket.setdefault(bucket, []).append(spread / max(factor, 0.2))
        issuers.setdefault(bucket, set()).add(_issuer_key(row))

    anchors, meta = {}, {}
    for bucket, values in by_bucket.items():
        # CCC is thin by nature (~8% of high yield), so it gets a lower floor.
        floor = min_per_bucket_ccc if bucket == 'CCC' else min_per_bucket
        if len(values) < floor:
            meta[bucket] = {'n': len(values), 'used': False}
            continue
        # A median over a few issuers anchors those issuers, not the grade.
        n_issuers = len(issuers.get(bucket, set()) - {None})
        if n_issuers < min_issuers:
            meta[bucket] = {'n': len(values), 'used': False,
                            'dropped': f'only {n_issuers} issuers'}
            continue
        values.sort()
        anchors[bucket] = round(values[len(values) // 2], 6)
        meta[bucket] = {'n': len(values), 'used': True,
                        'anchor': anchors[bucket]}

    # Anchors must rise as credit worsens, or the scale inverts. A bucket that
    # breaks the ordering is dropped rather than forced: it means the scorecard
    # is not separating those two grades, and inventing a gap would hide that.
    ordered, floor = {}, 0.0
    for bucket in CREDIT_BUCKETS:
        value = anchors.get(bucket)
        if value is None:
            continue
        if value <= floor:
            meta[bucket]['used'] = False
            meta[bucket]['dropped'] = 'not wider than the better bucket'
            continue
        ordered[bucket] = value
        floor = value

    # No fitted CCC: derive it from the B anchor at the index's own CCC/B
    # ratio. Falling back to the raw 1023bp index left a ~900bp gap above B,
    # and every spread in it was labelled B.
    if ('CCC' not in ordered and ordered.get('B') and bucket_oas
            and bucket_oas.get('B') and bucket_oas.get('CCC')):
        derived = round(ordered['B'] * bucket_oas['CCC'] / bucket_oas['B'], 6)
        if derived > ordered['B']:
            ordered['CCC'] = derived
            meta['CCC'] = {**meta.get('CCC', {'n': 0}), 'used': True,
                           'anchor': derived, 'derived': 'index_ratio'}

    ordered['_meta'] = meta
    ordered['_reference_years'] = reference_years
    return ordered
