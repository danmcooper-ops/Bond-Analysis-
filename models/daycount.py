"""Day-count conventions.

Four conventions cover the universe this model touches:

    30/360    US corporates and agencies (30U/360, "bond basis")
    ACT/ACT   US Treasury notes and bonds (ICMA)
    ACT/360   Treasury bills and money-market instruments
    ACT/365F  occasional non-US paper

ACT/ACT needs the *enclosing coupon period* to know what a year is, which is
why year_fraction takes optional period_start/period_end/frequency arguments
the other conventions ignore. Callers that only ever touch 30/360 can leave
them out; callers that might see a Treasury must pass them.
"""

import calendar
from datetime import date

D30_360 = '30/360'
ACT_ACT = 'ACT/ACT'
ACT_360 = 'ACT/360'
ACT_365F = 'ACT/365F'

CONVENTIONS = (D30_360, ACT_ACT, ACT_360, ACT_365F)


def _is_last_of_february(d):
    return d.month == 2 and d.day == calendar.monthrange(d.year, 2)[1]


def days_30_360(d1, d2, eom=True):
    """Day count under 30U/360 (SIA "bond basis").

    The four adjustment rules, in the order the SIA specifies them:

      1. (EOM only) last-of-Feb to last-of-Feb   -> D2 = 30
      2. (EOM only) D1 is last of Feb            -> D1 = 30
      3. D2 == 31 and D1 in (30, 31)             -> D2 = 30
      4. D1 == 31                                -> D1 = 30

    Rules 1 and 2 are the EOM refinement; `eom=False` gives plain 30/360,
    which some issuers use. The ordering matters: rule 3 tests D1 *before*
    rule 4 rewrites it, so a 31 -> 31 pair becomes 30 -> 30 and a 30 -> 31
    pair also becomes 30 -> 30, but a 29 -> 31 pair stays 29 -> 31.
    """
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day

    if eom:
        if _is_last_of_february(d1) and _is_last_of_february(d2):
            dd2 = 30
        if _is_last_of_february(d1):
            dd1 = 30

    if dd2 == 31 and dd1 in (30, 31):
        dd2 = 30
    if dd1 == 31:
        dd1 = 30

    return 360 * (y2 - y1) + 30 * (m2 - m1) + (dd2 - dd1)


def _act_act_icma(start, end, period_start, period_end, frequency):
    """ACT/ACT (ICMA): actual days over the actual length of the enclosing
    coupon period, annualised by the coupon frequency.

    This is the convention that makes a Treasury's accrued interest exact:
    every coupon period is worth exactly 1/frequency of a year regardless of
    whether it happens to contain 181 days or 184.
    """
    if period_start is None or period_end is None or not frequency:
        # No enclosing period supplied — fall back to ACT/365F rather than
        # silently returning a wrong number. Callers pricing Treasuries must
        # pass the period; this keeps a careless caller merely approximate
        # instead of badly wrong.
        return (end - start).days / 365.0
    period_days = (period_end - period_start).days
    if period_days <= 0:
        return 0.0
    return (end - start).days / (period_days * frequency)


def year_fraction(start, end, convention, period_start=None, period_end=None,
                  frequency=2, eom=True):
    """Return the year fraction between two dates under `convention`.

    Args:
        start, end: dates. `end` before `start` yields a negative fraction.
        convention: one of the module constants.
        period_start, period_end, frequency: the enclosing coupon period.
            Required for ACT/ACT; ignored by the others.
        eom: apply the end-of-month refinement to 30/360.
    """
    if convention == D30_360:
        return days_30_360(start, end, eom=eom) / 360.0
    if convention == ACT_360:
        return (end - start).days / 360.0
    if convention == ACT_365F:
        return (end - start).days / 365.0
    if convention == ACT_ACT:
        return _act_act_icma(start, end, period_start, period_end, frequency)
    raise ValueError(f"Unknown day-count convention: {convention!r}")


def day_count(start, end, convention, eom=True):
    """Return the day count (not a fraction) between two dates.

    For 30/360 this is the adjusted 30-day-month count; for every ACT
    convention it is the plain calendar difference.
    """
    if convention == D30_360:
        return days_30_360(start, end, eom=eom)
    if convention in (ACT_360, ACT_365F, ACT_ACT):
        return (end - start).days
    raise ValueError(f"Unknown day-count convention: {convention!r}")


def accrual_fraction(period_start, settle, period_end, convention,
                     frequency=2, eom=True):
    """Fraction of the current coupon period that has accrued, in [0, 1].

    Defined once here so accrued interest, the pricing formula's stub factor
    w = 1 - accrual_fraction, and duration all agree by construction. A
    mismatch between those three is the classic source of a bond model that
    is nearly right.
    """
    if period_end <= period_start:
        return 0.0
    frac = year_fraction(period_start, settle, convention,
                         period_start=period_start, period_end=period_end,
                         frequency=frequency, eom=eom) * frequency
    return max(0.0, min(1.0, frac))
