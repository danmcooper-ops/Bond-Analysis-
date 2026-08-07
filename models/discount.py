"""Discount instruments: bills, and anything with a single cashflow.

WHY THESE CANNOT GO THROUGH THE COUPON-BOND PATH
-------------------------------------------------
The street pricing formula discounts the k-th remaining cashflow by
(1 + y/m)^(w + k - 1) — it indexes by COUPON PERIOD, not by time. That is
correct and exact for a coupon bond, and silently wrong for an instrument with
no coupon schedule: with frequency 0 there is exactly one cashflow at k=1, so
it gets discounted over exactly one period no matter when it actually matures.

The symptom is dramatic once you look: a 5-day bill at 99.95 came back with a
0.05% yield instead of ~3.9%, and a 10-year zero at 62 came back at 60%
instead of ~4.8%. Duration collapsed to 1.0 year for every one of them.

So discount instruments get their own closed forms here. They are short
because a single cashflow makes everything analytic:

    P = F / (1 + y*t)                 money-market (simple) convention
    D_mac = t                         exactly, always
    D_mod = t / (1 + y*t)
    C     = 2*t^2 / (1 + y*t)^2

LONG zeros are a different case again. A 10-year STRIP is conventionally
quoted on a semiannual bond-equivalent basis, not money-market, so it belongs
on the coupon-bond path with frequency=2 and a zero coupon — which generates
the right number of periods and reuses the tested machinery. The split is by
tenor, not by coupon: see conventions.py.
"""

from models.daycount import ACT_360, year_fraction

# Below this, quote money-market simple. Above it, bond-equivalent semiannual.
# One year is the conventional dividing line and also where simple and
# compound interest start to diverge materially.
MONEY_MARKET_MAX_YEARS = 1.0


def time_to_maturity(settle, maturity, convention=ACT_360):
    """Year fraction to the single cashflow, on the instrument's own basis."""
    return year_fraction(settle, maturity, convention)


def price_from_simple_yield(face, ytm, t_years):
    """P = F / (1 + y*t). None outside the domain."""
    if t_years is None or t_years <= 0:
        return None
    denom = 1.0 + ytm * t_years
    if denom <= 0:
        return None
    return face / denom


def simple_yield_from_price(price, face, t_years):
    """y = (F/P - 1) / t. The money-market convention.

    Deliberately NOT the discount-rate convention ((F-P)/F/t) that bills are
    auctioned on: that quotes a lower number for the same economics and would
    not be comparable to the bond-equivalent yields everything else in this
    model carries. Comparability across the universe is the whole point of a
    single rating scale.
    """
    if price is None or price <= 0 or t_years is None or t_years <= 0:
        return None
    return (face / price - 1.0) / t_years


def discount_rate_from_price(price, face, t_years):
    """The auction convention, for display only: d = (F-P)/F / t."""
    if price is None or face <= 0 or t_years is None or t_years <= 0:
        return None
    return (face - price) / face / t_years


def macaulay_duration(t_years):
    """A single cashflow's PV-weighted average time IS its maturity."""
    return t_years


def modified_duration(ytm, t_years):
    """-1/P dP/dy for P = F/(1+yt), which is t/(1+yt)."""
    if t_years is None or ytm is None:
        return None
    denom = 1.0 + ytm * t_years
    if denom <= 0:
        return None
    return t_years / denom


def convexity(ytm, t_years):
    """1/P d2P/dy2 for P = F/(1+yt), which is 2t^2/(1+yt)^2."""
    if t_years is None or ytm is None:
        return None
    denom = 1.0 + ytm * t_years
    if denom <= 0:
        return None
    return 2.0 * t_years * t_years / (denom * denom)


def bond_equivalent_yield(price, face, settle, maturity):
    """The bill's yield restated on a 365-day basis, for comparability.

    Bills are quoted ACT/360 and coupon bonds on a 365-ish bond basis, so the
    same economics produce a numerically LOWER number for the bill —
    a 5-day bill priced off the curve at the same rate as the 1-month point
    reports 3.71% against the note's 3.90%. Ranking the two against each other
    on those raw numbers would penalise every bill by roughly 5bp of pure
    convention.

    This model's entire premise is one comparable rating scale, so the gates
    consume this and the money-market yield is kept for display only.

    BEY = (F/P - 1) x 365/days. Exact for bills of 182 days or fewer, which is
    all of them; beyond that the convention adds a semiannual compounding
    term, and `discount.analyze` flags when that approximation is in use.
    """
    days = (maturity - settle).days
    if price is None or price <= 0 or days <= 0:
        return None
    return (face / price - 1.0) * 365.0 / days


def analyze(face, price, settle, maturity, convention=ACT_360):
    """Full analytics for a discount instrument from its price.

    Returns the same field names the coupon-bond path produces, so downstream
    scoring does not need to know which path a row came from. `ytm`/`ytw` are
    on the comparable bond-equivalent basis; `ytm_money_market` keeps the
    ACT/360 quote convention for display.
    """
    t = time_to_maturity(settle, maturity, convention)
    if t is None or t <= 0:
        return None
    mm_yield = simple_yield_from_price(price, face, t)
    bey = bond_equivalent_yield(price, face, settle, maturity)
    if mm_yield is None or bey is None:
        return None

    days = (maturity - settle).days
    # Duration and convexity are risk measures, so they use the yield the
    # price actually responds to — the money-market one.
    return {
        'ytm': bey,
        'ytw': bey,
        'ytm_money_market': mm_yield,
        'discount_rate': discount_rate_from_price(price, face, t),
        'macaulay_duration': macaulay_duration(t),
        'modified_duration': modified_duration(mm_yield, t),
        'convexity': convexity(mm_yield, t),
        'time_to_maturity': t,
        'bey_is_approximate': days > 182,
    }
