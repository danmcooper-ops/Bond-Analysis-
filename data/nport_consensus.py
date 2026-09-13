"""Collapse many funds' marks on the same CUSIP into one consensus price.

Funds value the same bond independently, at slightly different times, using
different pricing services. The disagreement is signal as well as noise: a
bond that ten funds mark within 20bp of each other is liquid and observable; a
bond that two funds mark 8% apart is neither, and the model should decline to
rate it rather than average the two and pretend.

MEDIAN, NOT MEAN, and outliers rejected on a MEDIAN ABSOLUTE DEVIATION test
rather than a standard-deviation one. Standard deviation is computed from the
very outliers it is meant to catch — one fat-finger at 10x inflates it enough
to bring itself inside three sigma. MAD has a breakdown point of 50%: half the
marks would have to be wrong before it moves.
"""

import re
import statistics
from collections import Counter, defaultdict

from data.logging_setup import get_logger
from scripts.config import (CONSENSUS_MAD_K, IMPLIED_PRICE_CEIL,
                            IMPLIED_PRICE_FLOOR)

log = get_logger('nport_consensus')

# Scale factor making MAD a consistent estimator of sigma for normal data, so
# mad_k reads on the same intuitive scale as a sigma multiple.
MAD_TO_SIGMA = 1.4826

# Absolute floor on the outlier band, in price points. Fund marks cluster far
# more tightly than a MAD test assumes, so without this the band collapses to
# a fraction of a cent and rejects ordinary pricing-service noise.
MIN_OUTLIER_BAND = 0.50

# Relative floor, for deep-discount paper where half a point is a large move.
MIN_OUTLIER_REL = 0.005

# Coupons differing by less than this are the same bond reported to different
# precision (4.125 vs 4.13), not a disagreement. Expressed in percent, matching
# ANNUALIZED_RATE's units.
COUPON_CONFLICT_TOLERANCE = 0.02


def _median(values):
    return statistics.median(values) if values else None


def _mad(values, centre):
    """Median absolute deviation about `centre`."""
    if not values:
        return None
    return statistics.median([abs(v - centre) for v in values])


def _modal(values):
    """Most common non-null value, or None."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return Counter(present).most_common(1)[0][0]


def reject_outliers(prices, mad_k=CONSENSUS_MAD_K, floor=MIN_OUTLIER_BAND):
    """Return (kept, rejected) using a MAD test about the median.

    The band has an ABSOLUTE FLOOR, and it is load-bearing. Fund marks agree
    far more tightly than a statistical test expects — the observed median
    cross-fund dispersion is 0.000% and the 90th percentile is 0.07%, because
    most funds price off the same handful of evaluation services. With a MAD
    of a fraction of a cent, `mad_k * MAD` is a band of a fraction of a cent,
    and a fund quoting 99.66 against a median of 99.625 gets thrown out as an
    outlier. That does not move the median much, but it deflates n_funds,
    which feeds both the Fund Breadth gate and the minimum-funds rating cap —
    so a bond held by forty funds could be capped for "thin coverage".

    Below the floor no rejection happens at all: marks within half a point of
    each other are pricing-service noise, not disagreement.
    """
    if len(prices) < 3:
        return list(prices), []
    centre = _median(prices)
    # MAD is 0 whenever most funds report the same price — the COMMON case —
    # and returning early then kept any outlier: [99.5 x4, 60] kept the 60.
    # The absolute and relative floors still define a sensible band.
    mad = _mad(prices, centre) or 0.0
    limit = max(mad_k * mad * MAD_TO_SIGMA, floor, abs(centre) * MIN_OUTLIER_REL)
    kept = [p for p in prices if abs(p - centre) <= limit]
    rejected = [p for p in prices if abs(p - centre) > limit]
    # Never reject everything: if the test is that aggressive the distribution
    # is not what the test assumes, and the median is still the best guess.
    return (kept, rejected) if kept else (list(prices), [])


# Coupons below this in a wholly-low group can only be decimal fractions: no
# real bond pays 0.05%, so 0.05 is a 5% coupon written as a fraction.
DECIMAL_ONLY_BELOW = 0.10

# Between DECIMAL_ONLY_BELOW and 0.5 a value is genuinely ambiguous: 0.25 is
# a 0.25% convertible or a 25% coupon written as a fraction. Plausible band
# for an approximate yield when using the price to tell them apart.
# The floor matters: a sub-1% yield is not something a non-convertible bond
# prices to, which is what rules out "0.125% at par" in favour of 12.5%.
PLAUSIBLE_YIELD = (0.01, 0.15)

_TITLE_PCT = re.compile(r'(\d{1,2}(?:\.\d+)?)\s*%')
# Bloomberg-style ticker titles: "EEFT 0 5/8 10/01/30", "LABL 10.5 07/15/27",
# and a bare fraction, "ACAFP 1/2 07/21/27".
_TITLE_TICKER = re.compile(
    r'^[A-Z0-9&.\-]+\s+(?:(\d{1,2}(?:\.\d+)?)(?:\s+(\d)/(\d{1,2}))?'
    r'|(\d)/(\d{1,2}))\s+\d{1,2}/\d{1,2}/\d{2,4}\b')

# Government paper has not carried a double-digit coupon in decades.
GOVERNMENT_ISSUER_TYPES = ('UST', 'USGA', 'USGSE')


def coupon_from_title(title):
    """The coupon a title states, in percent, or None."""
    text = (title or '').strip().upper()
    if not text:
        return None
    m = _TITLE_PCT.search(text)
    if m:
        return float(m.group(1))
    m = _TITLE_TICKER.match(text)
    if m:
        if m.group(1) is None:
            return int(m.group(4)) / int(m.group(5))
        value = float(m.group(1))
        if m.group(2):
            value += int(m.group(2)) / int(m.group(3))
        return value
    return None


def _approx_yield(coupon_pct, price, years):
    """Textbook YTM approximation; enough to rule an interpretation out."""
    if not price or not years or years <= 0:
        return None
    return ((coupon_pct + (100.0 - price) / years)
            / ((100.0 + price) / 2.0))


def _resolve_ambiguous(value, context):
    """Percent or fraction, for a coupon in [DECIMAL_ONLY_BELOW, 0.5).

    Returns (percent_value, ambiguous). Evidence, strongest first: what the
    title says, a convertible flag (low coupons are what convertibles pay),
    then which reading gives a yield a bond at this price could have.
    """
    ctx = context or {}
    stated = coupon_from_title(ctx.get('title_of_issue'))
    if stated is not None:
        if abs(stated - value) <= 0.02:
            return value, False
        if abs(stated - value * 100.0) <= max(0.05, 0.02 * stated):
            return value * 100.0, False
    if ctx.get('is_convertible') or ctx.get('issuer_type') in GOVERNMENT_ISSUER_TYPES:
        return value, False

    price, years = ctx.get('price'), ctx.get('years')
    as_pct = _approx_yield(value, price, years)
    as_frac = _approx_yield(value * 100.0, price, years)
    lo, hi = PLAUSIBLE_YIELD
    pct_ok = as_pct is not None and lo <= as_pct <= hi
    frac_ok = as_frac is not None and lo <= as_frac <= hi
    if pct_ok and not frac_ok:
        return value, False
    if frac_ok and not pct_ok:
        return value * 100.0, False
    # No evidence either way. Keep the smaller reading and say so: a coupon
    # inflated 100x produced a 2,005bp Z-spread on a 0.25% note and ranked it.
    return value, True


def normalise_coupon_units(coupons, context=None):
    """Put every reported coupon on the same scale — percent.

    ANNUALIZED_RATE IS MIXED-UNITS IN THE SOURCE DATA. Some funds report a
    5% coupon as 5.0 and others as 0.05, in the same file, for the same CUSIP:
    SBA Communications came back as [0.0312, 3.125], PBF Holding as
    [0.0788, 7.875, 7.88]. Roughly 17,000 CUSIPs in 2026Q2 span both
    conventions.

    Where a group spans both conventions the percent-scale values identify the
    truth and the decimals are rescaled to match. Where a group is entirely
    below 0.5, values under 0.10 are fractions (no bond pays 0.05%), but 0.10
    to 0.5 is genuinely ambiguous — 0.25 is a 0.25% convertible as often as a
    25% coupon — and is resolved from `context` (title_of_issue,
    is_convertible, price, years). Exactly zero is left alone: that is a real
    zero-coupon bond.

    Use coupon_units_ambiguous() to learn whether the evidence ran out.
    """
    return _normalise(coupons, context)[0]


def coupon_units_ambiguous(coupons, context=None):
    return _normalise(coupons, context)[1]


def _normalise(coupons, context):
    present = [c for c in coupons if c is not None]
    if not present:
        return [], False

    pct_like = [c for c in present if c >= 0.5]
    dec_like = [c for c in present if 0 < c < 0.5]
    if not dec_like:
        return present, False

    if pct_like:
        # The group tells us its own scale: rescale only the decimals that
        # land on the percent-scale consensus, leaving a genuinely odd value
        # alone rather than forcing it to fit.
        reference = _median(pct_like)
        tolerance = max(0.05, 0.02 * reference)
        return [c * 100.0 if (0 < c < 0.5 and abs(c * 100.0 - reference) <= tolerance)
                else c for c in present], False

    small = [c for c in dec_like if c < DECIMAL_ONLY_BELOW]
    mid = [c for c in dec_like if c >= DECIMAL_ONLY_BELOW]
    if not mid:
        return [c * 100.0 if 0 < c < 0.5 else c for c in present], False

    # A fraction in the group that lands on the mid value once rescaled means
    # the mid value was already percent: [0.125, 0.00125] is one 0.125% bond.
    if any(abs(s * 100.0 - m) <= 0.02 for s in small for m in mid):
        scale_mid, ambiguous = 1.0, False
    else:
        resolved, ambiguous = _resolve_ambiguous(_median(mid), context)
        scale_mid = 100.0 if resolved != _median(mid) else 1.0

    out = []
    for c in present:
        if 0 < c < DECIMAL_ONLY_BELOW:
            out.append(c * 100.0)
        elif DECIMAL_ONLY_BELOW <= c < 0.5:
            out.append(c * scale_mid)
        else:
            out.append(c)
    return out, ambiguous


def _coupons_conflict(coupons, tolerance=COUPON_CONFLICT_TOLERANCE):
    """Do these reported coupons describe different bonds?

    NOT an equality test. Funds report the same coupon to different precision
    — a 4.125% bond appears as both '4.125' and '4.13', and 5.4 alongside
    5.401. Treating those as disagreement flagged a quarter of all CUSIPs as
    having conflicting terms, which would have driven a large slice of the
    universe into a lower analyzability score for a rounding convention.
    Real conflicts (4.125 vs 5.5) are far larger than reporting precision.
    """
    present = [c for c in coupons if c is not None]
    if len(present) < 2:
        return False
    return (max(present) - min(present)) > tolerance


def consensus_mark(holdings, mad_k=CONSENSUS_MAD_K,
                   price_floor=IMPLIED_PRICE_FLOOR,
                   price_ceil=IMPLIED_PRICE_CEIL):
    """One row per (CUSIP, report month) from many fund holdings.

    Returns a list of dicts carrying the consensus price, the dispersion that
    justifies trusting it, and the security terms — with any cross-fund
    disagreement about those terms flagged rather than silently resolved.
    """
    groups = defaultdict(list)
    for h in holdings:
        price = h.get('implied_price')
        if price is None or not (price_floor <= price <= price_ceil):
            continue
        report_date = h.get('report_date')
        if report_date is None:
            continue
        groups[(h['cusip'], report_date)].append(h)

    out = []
    for (cusip, report_date), rows in groups.items():
        prices = [r['implied_price'] for r in rows]
        kept, rejected = reject_outliers(prices, mad_k=mad_k)
        centre = _median(kept)
        if centre is None:
            continue

        mad = _mad(kept, centre) or 0.0
        maturities = [r.get('maturity_date') for r in rows]
        # Reconcile the mixed percent/decimal encoding BEFORE anything reads
        # a coupon — the modal pick and the conflict test both depend on it.
        raw_coupons = [r.get('annualized_rate') for r in rows]
        maturity = _modal(maturities)
        years = None
        if maturity is not None and hasattr(report_date, 'year'):
            try:
                years = (_as_date(maturity) - _as_date(report_date)).days / 365.25
            except TypeError:
                years = None
        coupon_context = {
            'title_of_issue': _modal([r.get('title_of_issue') for r in rows]),
            'is_convertible': any(r.get('is_convertible') for r in rows),
            'issuer_type': _modal([r.get('issuer_type') for r in rows]),
            'price': centre, 'years': years}
        coupons, coupon_ambiguous = _normalise(raw_coupons, coupon_context)

        # Two funds reporting materially different terms for one CUSIP means
        # at least one is wrong, and there is no way to tell which. Flagged so
        # the analyzability score and the caps can act on it.
        distinct_maturities = {m for m in maturities if m is not None}
        identity_conflict = (len(distinct_maturities) > 1
                             or _coupons_conflict(coupons))

        levels = [r.get('fair_value_level') for r in rows
                  if r.get('fair_value_level')]

        out.append({
            'cusip': cusip,
            'report_date': report_date,
            'clean_price_marked': centre,
            'price_basis': 'nport_implied_clean',
            'price_min': min(kept),
            'price_max': max(kept),
            # MAD relative to the level, so it reads as a percentage and is
            # comparable between a 30-priced distressed bond and a 105 premium.
            'price_dispersion': (mad / centre) if centre else None,
            'n_funds': len(kept),
            'n_funds_rejected': len(rejected),
            'total_held_usd': sum(r.get('value_usd') or 0 for r in rows),
            'max_pct_of_nav': max((r.get('pct_of_nav') or 0) for r in rows),
            # Conservative: one fund calling it level 3 is the signal.
            'fair_value_level': max(levels) if levels else None,
            'issuer_name': _modal([r.get('issuer_name') for r in rows]),
            'title_of_issue': _modal([r.get('title_of_issue') for r in rows]),
            'issuer_type': _modal([r.get('issuer_type') for r in rows]),
            'payoff_profile': _modal([r.get('payoff_profile') for r in rows]),
            'maturity_date': _modal(maturities),
            'annualized_rate': _modal(coupons),
            'coupon_type': _modal([r.get('coupon_type') for r in rows]),
            # Any fund flagging trouble flags the bond. These are assertions
            # about the issuer, not opinions about price: one fund reporting a
            # default is not outvoted by nine that have not updated.
            'is_default': any(r.get('is_default') for r in rows),
            'in_arrears': any(r.get('in_arrears') for r in rows),
            'is_paid_kind': any(r.get('is_paid_kind') for r in rows),
            'is_convertible': any(r.get('is_convertible') for r in rows),
            '_identity_conflict': identity_conflict,
            '_coupon_unit_ambiguous': coupon_ambiguous,
        })

    if out:
        singles = sum(1 for r in out if r['n_funds'] == 1)
        conflicts = sum(1 for r in out if r['_identity_conflict'])
        rejected_total = sum(r['n_funds_rejected'] for r in out)
        log.info('Consensus: %d CUSIP-months, %d single-fund, %d term '
                 'conflicts, %d outlier marks rejected',
                 len(out), singles, conflicts, rejected_total)
    return out


def _as_date(value):
    from datetime import date, datetime
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, 'date'):
        return value.date()
    if isinstance(value, str):
        return datetime.strptime(value[:10], '%Y-%m-%d').date()
    return value


def latest_marks(marks):
    """Keep only the most recent report_date per CUSIP."""
    best = {}
    for row in marks:
        prior = best.get(row['cusip'])
        if prior is None or row['report_date'] > prior['report_date']:
            best[row['cusip']] = row
    return list(best.values())
