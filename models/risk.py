"""Interest-rate risk: duration, convexity, DV01, key-rate durations.

Durations here are computed analytically from the cashflow set, and there is a
numerical counterpart (`effective_duration`) computed by bumping the yield.
Testing the two against each other proves both at once: an error in the
analytic weights and an identical error in the bumping would have to conspire,
which they will not.
"""

from models.pricing import _price_derivative, price_from_yield


def _pv_weights(flows, ytm, frequency=2, w=1.0):
    """Yield (t_years, pv) per cashflow under the street formula.

    t is measured in years as (w + k - 1)/m, matching the discounting exactly
    — so duration and price cannot disagree about where a cashflow sits.
    """
    m = frequency if frequency else 1
    base = 1.0 + ytm / m
    if base <= 0:
        return None
    out = []
    for k, (_, amt) in enumerate(flows, start=1):
        n = w + k - 1
        out.append((n / m, amt / base ** n))
    return out


def macaulay_duration(flows, ytm, frequency=2, w=1.0):
    """PV-weighted average time to cashflow, in years.

    For a zero this is exactly the time to maturity — the cleanest available
    check that the weighting is right.
    """
    parts = _pv_weights(flows, ytm, frequency=frequency, w=w)
    if not parts:
        return None
    pv_total = sum(pv for _, pv in parts)
    if pv_total <= 0:
        return None
    return sum(t * pv for t, pv in parts) / pv_total


def modified_duration(macaulay, ytm, frequency=2):
    """Macaulay / (1 + y/m): the percentage price move per unit yield move."""
    if macaulay is None:
        return None
    m = frequency if frequency else 1
    base = 1.0 + ytm / m
    if base <= 0:
        return None
    return macaulay / base


def convexity(flows, ytm, frequency=2, w=1.0):
    """Second-order price sensitivity, in years squared.

    sum t(t + 1/m) * PV / (P * (1 + y/m)^2)
    """
    parts = _pv_weights(flows, ytm, frequency=frequency, w=w)
    if not parts:
        return None
    m = frequency if frequency else 1
    base = 1.0 + ytm / m
    if base <= 0:
        return None
    pv_total = sum(pv for _, pv in parts)
    if pv_total <= 0:
        return None
    num = sum(t * (t + 1.0 / m) * pv for t, pv in parts)
    return num / (pv_total * base ** 2)


def effective_duration(flows, ytm, frequency=2, w=1.0, bump=0.0001):
    """Duration by central difference on the yield.

    (P(y-h) - P(y+h)) / (2 * P(y) * h). For an option-free bond this must
    agree with `modified_duration` to well within a basis point; the two
    disagreeing is the signature of a bad discount exponent.
    """
    p0 = price_from_yield(flows, ytm, frequency=frequency, w=w)
    p_up = price_from_yield(flows, ytm + bump, frequency=frequency, w=w)
    p_dn = price_from_yield(flows, ytm - bump, frequency=frequency, w=w)
    if not p0 or p_up is None or p_dn is None or p0 <= 0:
        return None
    return (p_dn - p_up) / (2.0 * p0 * bump)


def effective_convexity(flows, ytm, frequency=2, w=1.0, bump=0.0001):
    """Convexity by central second difference: (P+ + P- - 2P) / (P * h^2)."""
    p0 = price_from_yield(flows, ytm, frequency=frequency, w=w)
    p_up = price_from_yield(flows, ytm + bump, frequency=frequency, w=w)
    p_dn = price_from_yield(flows, ytm - bump, frequency=frequency, w=w)
    if not p0 or p_up is None or p_dn is None or p0 <= 0:
        return None
    return (p_up + p_dn - 2.0 * p0) / (p0 * bump ** 2)


def dv01(dirty, mod_duration, face=100.0):
    """Dollar value of one basis point, per `face`.

    D * P / 10000. A 100-face, duration-5 bond trading at par has a DV01 of
    exactly 0.05.
    """
    if dirty is None or mod_duration is None:
        return None
    return mod_duration * dirty / 10000.0


def spread_duration(flows, settle, z_spread, curve, bump=0.0001):
    """Sensitivity to a 1bp move in the credit spread, holding rates fixed.

    For an option-free bond this is numerically close to modified duration,
    but they are conceptually different quantities and the Rates gates need
    the spread one: it is what tells you how much of a bond's risk is credit
    rather than duration.
    """
    from models.spreads import price_from_zero_curve
    p0 = price_from_zero_curve(flows, settle, curve, spread=z_spread)
    p_up = price_from_zero_curve(flows, settle, curve, spread=z_spread + bump)
    p_dn = price_from_zero_curve(flows, settle, curve, spread=z_spread - bump)
    if not p0 or p_up is None or p_dn is None or p0 <= 0:
        return None
    return (p_dn - p_up) / (2.0 * p0 * bump)


def key_rate_durations(flows, settle, curve, key_tenors=(0.5, 2, 5, 10, 30),
                       bump=0.0001, spread=0.0):
    """Sensitivity to a 1bp bump at each key tenor, with a triangular shock.

    Each bump is a tent function peaking at the key tenor and decaying to zero
    at the neighbouring ones, so the individual KRDs sum to approximately the
    total duration rather than multiply-counting the curve.
    """
    from models.spreads import price_from_zero_curve

    p0 = price_from_zero_curve(flows, settle, curve, spread=spread)
    if not p0 or p0 <= 0:
        return {}

    tenors = sorted(key_tenors)
    out = {}
    for i, kt in enumerate(tenors):
        lo = tenors[i - 1] if i > 0 else 0.0
        hi = tenors[i + 1] if i < len(tenors) - 1 else tenors[-1] * 2

        def shocked(sign):
            def zero_fn(t):
                z = curve.zero(t)
                if z is None:
                    return None
                if t <= lo or t >= hi:
                    weight = 0.0
                elif t <= kt:
                    weight = (t - lo) / (kt - lo) if kt > lo else 1.0
                else:
                    weight = (hi - t) / (hi - kt) if hi > kt else 1.0
                return z + sign * bump * weight
            return zero_fn

        p_up = price_from_zero_curve(flows, settle, curve, spread=spread,
                                     zero_override=shocked(+1))
        p_dn = price_from_zero_curve(flows, settle, curve, spread=spread,
                                     zero_override=shocked(-1))
        if p_up is None or p_dn is None:
            continue
        out[f'krd_{kt:g}y'] = (p_dn - p_up) / (2.0 * p0 * bump)
    return out


def price_change_estimate(mod_duration, convexity_, dy):
    """Second-order price change estimate: -D*dy + 0.5*C*dy^2, as a fraction."""
    if mod_duration is None:
        return None
    est = -mod_duration * dy
    if convexity_ is not None:
        est += 0.5 * convexity_ * dy * dy
    return est
