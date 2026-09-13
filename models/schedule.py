"""Coupon schedules, cashflows and accrued interest.

Schedules are generated BACKWARD from maturity, which is how bonds actually
work: the maturity date is contractual and any irregularity lands in the first
(stub) period, not the last.

The end-of-month rule is inferred from the maturity date when not stated. A
bond maturing on the 31st pays on the last day of each month (including
February's 28th/29th); one maturing on the 15th pays on the 15th. Getting this
wrong shifts every accrual by a day or two, which is small per bond and
systematic across the whole book.
"""

import calendar
from datetime import date

from models.daycount import D30_360, accrual_fraction


def _last_day(year, month):
    return calendar.monthrange(year, month)[1]


def add_months(d, months, eom=False):
    """Add months to a date, clamping the day to the target month's length.

    With eom=True the result is pinned to the last day of the target month,
    which is the correct behaviour for a bond whose maturity is month-end.
    """
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    if eom:
        return date(year, month, _last_day(year, month))
    return date(year, month, min(d.day, _last_day(year, month)))


def infer_eom(maturity):
    """True when the maturity falls on the last day of its month."""
    return maturity.day == _last_day(maturity.year, maturity.month)


def coupon_dates(maturity, frequency=2, settle=None, dated_date=None,
                 eom=None, max_periods=2000):
    """Coupon dates from (but excluding) the period containing `settle`
    through maturity, generated backward from maturity.

    Returns [] for frequency 0 (a discount instrument: a bill or a true zero
    has no coupon dates, only a redemption).

    `dated_date` stops the backward walk at issuance, so a bond issued
    part-way through a period gets a short first period rather than a phantom
    coupon before it existed.
    """
    if not frequency:
        return []
    if eom is None:
        eom = infer_eom(maturity)
    step = 12 // frequency

    dates = [maturity]
    d = maturity
    for _ in range(max_periods):
        d = add_months(maturity, -step * len(dates), eom=eom)
        if dated_date is not None and d <= dated_date:
            break
        if settle is not None and d <= settle:
            # One more back-step gives the period start bracketing settle.
            dates.append(d)
            break
        dates.append(d)
    dates.sort()
    return dates


def _notional_period(settle, maturity, frequency=2, dated_date=None, eom=None):
    """(accrual_start, notional_start, next_coupon) for the period holding settle.

    notional_start is where a REGULAR period would begin. It differs from
    accrual_start only in a short first period (dated date mid-period), and
    that difference matters: ACT/ACT divides by the regular period's length,
    and the stub factor w is measured against it. Using the dated date for
    both overstated accrued interest by ~50% on such a bond.
    """
    prev, nxt = previous_next_coupon(settle, maturity, frequency, eom=eom)
    start = prev
    if dated_date is not None and prev < dated_date:
        start = dated_date
    return start, prev, nxt


def _resolved_eom(maturity, eom):
    return infer_eom(maturity) if eom is None else bool(eom)


def previous_next_coupon(settle, maturity, frequency=2, dated_date=None,
                         eom=None):
    """Return (previous_coupon_date, next_coupon_date) bracketing `settle`.

    On a coupon date exactly, `previous` IS settle and `next` is one period
    later — so accrued interest is zero and the stub factor w is 1. That is
    what makes the par identity come out to exactly 100.
    """
    if not frequency:
        return (dated_date or settle), maturity
    if eom is None:
        eom = infer_eom(maturity)
    step = 12 // frequency

    # Walk back from maturity until we land on or before settle.
    nxt = maturity
    prev = add_months(maturity, -step, eom=eom)
    guard = 0
    while prev > settle and guard < 2000:
        nxt = prev
        prev = add_months(maturity, -step * (guard + 2), eom=eom)
        guard += 1
    if dated_date is not None and prev < dated_date:
        prev = dated_date
    return prev, nxt


def cashflows(face, coupon_rate, maturity, frequency=2, settle=None,
              dated_date=None, eom=None, convention=D30_360):
    """Remaining cashflows as [(date, amount), ...], strictly after `settle`.

    A coupon falling exactly on the settlement date belongs to the seller and
    is excluded — the buyer's first cashflow is the next one.

    For frequency 0 the only flow is the redemption at maturity.
    """
    if settle is None:
        settle = date.today()
    if maturity <= settle:
        return []
    if not frequency:
        return [(maturity, float(face))]

    dates = [d for d in coupon_dates(maturity, frequency, settle=settle,
                                     dated_date=dated_date, eom=eom)
             if d > settle]
    if not dates:
        return [(maturity, float(face))]

    cpn = float(face) * float(coupon_rate) / frequency
    flows = [(d, cpn) for d in dates]

    # A short first period pays only the coupon accrued since the dated date.
    if dated_date is not None:
        step = 12 // frequency
        first = flows[0][0]
        k = round(((first.year - maturity.year) * 12
                   + first.month - maturity.month) / -step) + 1
        notional_start = add_months(maturity, -step * k,
                                    eom=_resolved_eom(maturity, eom))
        if notional_start < dated_date < first:
            frac = accrual_fraction(dated_date, first, first, convention,
                                    frequency=frequency,
                                    eom=_resolved_eom(maturity, eom),
                                    notional_start=notional_start)
            flows[0] = (first, cpn * frac)
    # Redemption rides along with the final coupon.
    flows[-1] = (flows[-1][0], flows[-1][1] + float(face))
    return flows


def accrued_interest(settle, coupon_rate, maturity, frequency=2, face=100.0,
                     convention=D30_360, dated_date=None, eom=None):
    """Accrued interest per `face` at settlement.

    Zero on a coupon date; one full coupon less one day's worth the day before
    the next. A discount instrument (frequency 0) accrues nothing.
    """
    if not frequency or not coupon_rate:
        return 0.0
    start, notional, nxt = _notional_period(settle, maturity, frequency,
                                            dated_date=dated_date, eom=eom)
    frac = accrual_fraction(start, settle, nxt, convention, frequency=frequency,
                            eom=_resolved_eom(maturity, eom),
                            notional_start=notional)
    return float(face) * float(coupon_rate) / frequency * frac


def stub_factor(settle, maturity, frequency=2, convention=D30_360,
                dated_date=None, eom=None):
    """w = the fraction of the current coupon period still to run, in (0, 1].

    The street pricing formula discounts the k-th remaining cashflow by
    (1 + y/m)^(w + k - 1). On a coupon date w == 1 and the formula collapses
    to the textbook one.
    """
    if not frequency:
        return 1.0
    _, notional, nxt = _notional_period(settle, maturity, frequency,
                                        dated_date=dated_date, eom=eom)
    # Measured from the notional start: w is the share of a REGULAR period
    # still to run, which is what the (1 + y/m)^(w + k - 1) formula assumes.
    return 1.0 - accrual_fraction(notional, settle, nxt, convention,
                                 frequency=frequency,
                                 eom=_resolved_eom(maturity, eom))


def dirty_price(clean, accrued):
    """Dirty (invoice) price = clean + accrued."""
    if clean is None or accrued is None:
        return None
    return clean + accrued


def clean_price(dirty, accrued):
    """Clean (quoted) price = dirty - accrued."""
    if dirty is None or accrued is None:
        return None
    return dirty - accrued


def years_to_maturity(settle, maturity):
    """Calendar years to maturity. Used for bucketing and curve lookups, not
    for discounting — discounting uses the coupon-period count."""
    return (maturity - settle).days / 365.25
