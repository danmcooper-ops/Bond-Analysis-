"""Total return and its decomposition.

Bond performance is total return — price change PLUS coupon income — not price
return. Backtesting a bond model on price change alone systematically
understates every high-coupon bond and would rank the book roughly by coupon,
backwards.

The decomposition attributes a realised return to carry, roll-down, the rate
move, the spread move, and convexity. It also reports a RESIDUAL, and that
residual is the honest part: attribution is a first- and second-order
approximation, so the pieces do not sum exactly to the total. Silently folding
the gap into one of the components (usually "spread", since it is the least
observable) would make the attribution look precise while quietly lying. Here
the gap is reported, and a large one means the decomposition should not be
trusted for that bond.
"""

from models.daycount import D30_360, year_fraction
from models.schedule import accrued_interest, cashflows


def carry(coupon_rate, dirty, days, face=100.0, basis=360.0):
    """Coupon income over `days`, as a fraction of the dirty price."""
    if dirty is None or dirty <= 0 or coupon_rate is None:
        return None
    return (float(face) * float(coupon_rate) * days / basis) / dirty


def roll_down(curve, maturity_years, horizon_years, mod_duration):
    """Return from the bond ageing down a static curve.

    Holding the curve fixed, a bond that starts at tenor T sits at tenor
    T - h after h years. On an upward-sloping curve that means a lower yield,
    hence a higher price. Positive = the roll helps.

    Returns None past the point where the bond matures inside the horizon.
    """
    if mod_duration is None or maturity_years is None:
        return None
    t_end = maturity_years - horizon_years
    if t_end <= 0:
        return None
    y0, y1 = curve.par(maturity_years), curve.par(t_end)
    if y0 is None or y1 is None:
        return None
    return -mod_duration * (y1 - y0)


def coupons_between(coupon_rate, maturity, d0, d1, frequency=2, face=100.0,
                    dated_date=None, eom=None):
    """Coupons actually paid in (d0, d1].

    Computed from the real schedule rather than pro-rated from the coupon
    rate, because whether a payment date falls inside the window is a discrete
    fact that a pro-rata estimate gets wrong at exactly the moments that
    matter for a monthly backtest.
    """
    if not frequency or not coupon_rate:
        return 0.0
    flows = cashflows(face, coupon_rate, maturity, frequency=frequency,
                      settle=d0, dated_date=dated_date, eom=eom)
    total = 0.0
    for dt, amt in flows:
        if d0 < dt <= d1:
            # Strip the redemption from the final flow: principal returned is
            # not income.
            total += amt - face if dt == maturity else amt
    return total


def realized_total_return(p0_clean, p1_clean, a0, a1, coupons_paid):
    """((P1 + A1 + coupons) - (P0 + A0)) / (P0 + A0).

    Denominated on the DIRTY price, because that is what the buyer actually
    paid. Using the clean price as the base overstates the return of any bond
    bought mid-coupon-period.
    """
    if p0_clean is None or p1_clean is None:
        return None
    a0 = a0 or 0.0
    a1 = a1 or 0.0
    base = p0_clean + a0
    if base <= 0:
        return None
    return ((p1_clean + a1 + (coupons_paid or 0.0)) - base) / base


def decompose_total_return(bond, p0_clean, p1_clean, settle0, settle1,
                           curve0, curve1, ytm0, mod_duration, convexity_,
                           z_spread0, z_spread1, spread_dur=None):
    """Attribute a realised total return to its drivers.

    Returns a dict with total, income, roll_down, rate, spread, convexity and
    residual — all as fractions of the starting dirty price.

    residual = total - sum(components). Reported, never absorbed.
    """
    horizon_days = (settle1 - settle0).days
    if horizon_days <= 0:
        return None

    a0 = accrued_interest(settle0, bond.coupon_rate, bond.maturity,
                          frequency=bond.frequency, face=bond.face,
                          convention=bond.convention,
                          dated_date=bond.dated_date, eom=bond.eom)
    a1 = accrued_interest(settle1, bond.coupon_rate, bond.maturity,
                          frequency=bond.frequency, face=bond.face,
                          convention=bond.convention,
                          dated_date=bond.dated_date, eom=bond.eom)
    coupons = coupons_between(bond.coupon_rate, bond.maturity, settle0,
                              settle1, frequency=bond.frequency,
                              face=bond.face, dated_date=bond.dated_date,
                              eom=bond.eom)

    total = realized_total_return(p0_clean, p1_clean, a0, a1, coupons)
    if total is None:
        return None

    base = p0_clean + a0
    horizon_years = horizon_days / 365.25

    # Income: accrual change plus coupons received.
    income = ((a1 - a0) + coupons) / base

    t0 = bond.years_to_maturity(settle0)
    t1 = bond.years_to_maturity(settle1)

    # Rate: duration times the move in the benchmark yield at CONSTANT
    # maturity, so the ageing effect does not get double-counted here and in
    # roll-down.
    rate = None
    if mod_duration is not None:
        y0 = curve0.par(t0)
        y1_same_tenor = curve1.par(t0)
        if y0 is not None and y1_same_tenor is not None:
            dy = y1_same_tenor - y0
            rate = -mod_duration * dy
            cvx = (0.5 * convexity_ * dy * dy) if convexity_ is not None else 0.0
        else:
            cvx = 0.0
    else:
        cvx = 0.0

    roll = roll_down(curve0, t0, t0 - t1, mod_duration)

    spread = None
    if z_spread0 is not None and z_spread1 is not None:
        sd = spread_dur if spread_dur is not None else mod_duration
        if sd is not None:
            spread = -sd * (z_spread1 - z_spread0)

    parts = {'income': income, 'roll_down': roll, 'rate': rate,
             'spread': spread, 'convexity': cvx}
    explained = sum(v for v in parts.values() if v is not None)

    out = {'total': total, 'residual': total - explained,
           'horizon_days': horizon_days, 'horizon_years': horizon_years,
           'coupons_paid': coupons, 'accrued_start': a0, 'accrued_end': a1}
    out.update(parts)
    return out


def duration_matched_treasury_return(mod_duration, convexity_, curve0, curve1,
                                     maturity_years, horizon_days):
    """Return of a duration-matched Treasury over the same window.

    This — not an ETF — is the primary alpha benchmark, because it isolates
    credit selection skill from the duration bet. Beating LQD by being long
    duration in a rally is not credit skill; beating a duration-matched
    Treasury is.

    income + (-D * dy) + 0.5 * C * dy^2, with income taken from the starting
    curve yield at the matched tenor.
    """
    if mod_duration is None or maturity_years is None:
        return None
    y0 = curve0.par(maturity_years)
    y1 = curve1.par(maturity_years)
    if y0 is None or y1 is None:
        return None
    dy = y1 - y0
    income = y0 * (horizon_days / 365.25)
    price_move = -mod_duration * dy
    if convexity_ is not None:
        price_move += 0.5 * convexity_ * dy * dy
    return income + price_move
