"""Price / yield conversion.

Uses the street (ISMA) convention: the k-th remaining cashflow is discounted
by (1 + y/m)^(w + k - 1), where w is the fraction of the current coupon period
still to run. On a coupon date w == 1 and this collapses to the textbook
formula, which is what makes the par identity exact — a bond priced at a yield
equal to its coupon prices to precisely 100.000000, any frequency, any
convention.

`comp` is always an explicit argument. An implicit compounding basis is how a
bond model ends up 30bp wrong in a way nobody notices.
"""

from models.daycount import D30_360
from models.schedule import (accrued_interest, cashflows, previous_next_coupon,
                             stub_factor)
from models.solver import solve

COMP_SEMIANNUAL = 'semiannual'   # (1 + y/m)^(-n), the bond convention
COMP_SIMPLE = 'simple'           # 1 / (1 + y*t), the money-market convention


def price_from_yield(flows, ytm, frequency=2, w=1.0, comp=COMP_SEMIANNUAL,
                     t_years=None):
    """Dirty price per the face implied by `flows`, at yield `ytm`.

    Args:
        flows: [(date, amount), ...] in date order. Only the amounts and their
            ordinal position matter here; the dates are the caller's record.
        ytm: annual yield, as a decimal.
        frequency: coupons per year (m).
        w: stub factor in (0, 1] — the fraction of the current period left.
        comp: COMP_SEMIANNUAL or COMP_SIMPLE.
        t_years: required for COMP_SIMPLE (time to the single cashflow).

    Returns None when the yield is outside the domain (1 + y/m <= 0).
    """
    if not flows:
        return 0.0
    if comp == COMP_SIMPLE:
        if t_years is None:
            raise ValueError("COMP_SIMPLE requires t_years")
        denom = 1.0 + ytm * t_years
        if denom <= 0:
            return None
        return sum(amt for _, amt in flows) / denom

    m = frequency if frequency else 1
    base = 1.0 + ytm / m
    if base <= 0:
        return None
    total = 0.0
    for k, (_, amt) in enumerate(flows, start=1):
        total += amt / base ** (w + k - 1)
    return total


def _price_derivative(flows, ytm, frequency=2, w=1.0):
    """dP/dy for the street formula — the Newton derivative.

    dP/dy = -(1/m) * sum (w + k - 1) * CF_k * (1 + y/m)^-(w + k)
    """
    m = frequency if frequency else 1
    base = 1.0 + ytm / m
    if base <= 0:
        return None
    total = 0.0
    for k, (_, amt) in enumerate(flows, start=1):
        n = w + k - 1
        total += n * amt / base ** (n + 1)
    return -total / m


def yield_from_price(dirty, flows, frequency=2, w=1.0, comp=COMP_SEMIANNUAL,
                     guess=None, t_years=None):
    """Solve for the yield that reproduces `dirty`. None if it will not solve.

    Never raises. A non-converging bond gets None here and a
    `ytm_solver_failed` flag upstream, which a rating cap turns into HOLD.
    """
    if dirty is None or dirty <= 0 or not flows:
        return None

    if comp == COMP_SIMPLE:
        if t_years is None or t_years <= 0:
            return None
        total = sum(amt for _, amt in flows)
        return (total / dirty - 1.0) / t_years

    m = frequency if frequency else 1

    def f(y):
        p = price_from_yield(flows, y, frequency=m, w=w, comp=comp)
        return None if p is None else p - dirty

    def fprime(y):
        return _price_derivative(flows, y, frequency=m, w=w)

    if guess is None:
        # Current-yield-ish seed: total return over remaining life. Good enough
        # that Newton usually lands in two or three steps.
        total = sum(amt for _, amt in flows)
        n_periods = len(flows)
        guess = ((total / dirty) ** (1.0 / max(n_periods, 1)) - 1.0) * m

    # Bracket keeps 1 + y/m > 0 for any m >= 1, and spans deep-discount
    # distressed paper as well as the negative-yield case.
    return solve(f, fprime=fprime, guess=guess, bracket=(-0.99, 5.0))


# ---------------------------------------------------------------------------
# Bond-level convenience wrappers
# ---------------------------------------------------------------------------

def bond_flows_and_stub(coupon_rate, maturity, settle, frequency=2,
                        convention=D30_360, face=100.0, dated_date=None,
                        eom=None):
    """Return (flows, w) for a bond — the two inputs every pricing call needs."""
    flows = cashflows(face, coupon_rate, maturity, frequency=frequency,
                      settle=settle, dated_date=dated_date, eom=eom)
    w = stub_factor(settle, maturity, frequency=frequency,
                    convention=convention, dated_date=dated_date, eom=eom)
    return flows, w


def price_bond(coupon_rate, maturity, settle, ytm, frequency=2,
               convention=D30_360, face=100.0, dated_date=None, eom=None,
               comp=COMP_SEMIANNUAL):
    """Return (clean, dirty, accrued) for a bond at a given yield."""
    flows, w = bond_flows_and_stub(coupon_rate, maturity, settle,
                                   frequency=frequency, convention=convention,
                                   face=face, dated_date=dated_date, eom=eom)
    dirty = price_from_yield(flows, ytm, frequency=frequency, w=w, comp=comp)
    if dirty is None:
        return None, None, None
    accrued = accrued_interest(settle, coupon_rate, maturity,
                               frequency=frequency, face=face,
                               convention=convention, dated_date=dated_date,
                               eom=eom)
    return dirty - accrued, dirty, accrued


def yield_to_maturity(clean, coupon_rate, maturity, settle, frequency=2,
                      convention=D30_360, face=100.0, dated_date=None,
                      eom=None, comp=COMP_SEMIANNUAL):
    """Yield to maturity from a CLEAN price. None if it will not solve."""
    if clean is None:
        return None
    flows, w = bond_flows_and_stub(coupon_rate, maturity, settle,
                                   frequency=frequency, convention=convention,
                                   face=face, dated_date=dated_date, eom=eom)
    if not flows:
        return None
    accrued = accrued_interest(settle, coupon_rate, maturity,
                               frequency=frequency, face=face,
                               convention=convention, dated_date=dated_date,
                               eom=eom)
    return yield_from_price(clean + accrued, flows, frequency=frequency, w=w,
                            comp=comp)


def yield_to_call(clean, coupon_rate, call_date, call_price, settle,
                  frequency=2, convention=D30_360, face=100.0,
                  dated_date=None, eom=None):
    """Yield to a specific call date and price.

    Modelled as a bond redeeming early at `call_price` instead of par: same
    coupon stream, truncated, with the call price as the redemption.
    """
    if clean is None or call_date is None or call_date <= settle:
        return None
    flows, w = bond_flows_and_stub(coupon_rate, call_date, settle,
                                   frequency=frequency, convention=convention,
                                   face=face, dated_date=dated_date, eom=eom)
    if not flows:
        return None
    # Swap par redemption for the call price on the final flow.
    last_date, last_amt = flows[-1]
    flows[-1] = (last_date, last_amt - face + call_price)
    accrued = accrued_interest(settle, coupon_rate, call_date,
                               frequency=frequency, face=face,
                               convention=convention, dated_date=dated_date,
                               eom=eom)
    return yield_from_price(clean + accrued, flows, frequency=frequency, w=w)


def yield_to_worst(clean, coupon_rate, maturity, settle, call_schedule=None,
                   frequency=2, convention=D30_360, face=100.0,
                   dated_date=None, eom=None):
    """Lowest of yield-to-maturity and every yield-to-call.

    Returns a dict rather than a bare number so the caller can tell "worst is
    maturity because there are no calls" from "worst is maturity because we
    have no call data". That distinction drives a rating cap: a bond priced
    above par that is probably callable, with no schedule to prove it, is a
    HOLD rather than a confident BUY, because its Z-spread overstates the
    compensation on offer.

        call_schedule: [(date, price), ...] or None.
    """
    ytm = yield_to_maturity(clean, coupon_rate, maturity, settle,
                            frequency=frequency, convention=convention,
                            face=face, dated_date=dated_date, eom=eom)
    result = {'ytw': ytm, 'to_date': maturity, 'to_type': 'maturity',
              'call_data_available': bool(call_schedule)}
    if not call_schedule or ytm is None:
        return result

    for cd, cp in call_schedule:
        if cd is None or cd <= settle:
            continue
        ytc = yield_to_call(clean, coupon_rate, cd, cp, settle,
                            frequency=frequency, convention=convention,
                            face=face, dated_date=dated_date, eom=eom)
        if ytc is not None and ytc < result['ytw']:
            result.update(ytw=ytc, to_date=cd, to_type='call')
    return result


def current_yield(clean, coupon_rate, face=100.0):
    """Annual coupon over clean price. None when the price is unusable."""
    if clean is None or clean <= 0:
        return None
    return float(face) * float(coupon_rate) / clean
