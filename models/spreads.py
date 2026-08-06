"""Spread measures.

WHY THERE IS NO OAS HERE, AND WHAT WE DO INSTEAD
------------------------------------------------
A true option-adjusted spread requires two things this model cannot get for
free:

  (a) the security's full call/put schedule — which lives only as prose in
      424B prospectuses and EX-4 indentures, never as structured data; and
  (b) an arbitrage-free short-rate model (Hull-White, BDT) calibrated to a
      swaption volatility surface, valued on a lattice or by Monte Carlo —
      and there is no free swaption vol surface.

So this module computes **Z-spread** and the pipeline compares it against the
FRED ICE BofA bucket **OAS** index. Those are not the same quantity. For a
callable bond the Z-spread exceeds the OAS by roughly the value of the
embedded call — on the order of 10-60bp for investment grade, materially more
for high yield trading near a call date. Treating one as the other would make
every callable look systematically cheap.

Three mitigations, none of which is "assume it away":

  1. `is_likely_callable` flags the suspects from the data we do have.
  2. `fit_z_oas_wedge` FITS the Z-minus-OAS gap per rating bucket by
     regressing this model's own aggregate Z-spreads against the published
     bucket OAS across months. The wedge is measured, not assumed, and it is
     refit monthly.
  3. A bond that is priced above par, is probably callable, and has no call
     schedule gets a HOLD cap — because that is precisely the case where the
     Z-spread lies worst, and the honest answer is "we cannot tell".

Asset-swap spread is out of scope for v1: it needs a SOFR OIS curve.
"""

from models.schedule import years_to_maturity
from models.solver import brent


def price_from_zero_curve(flows, settle, curve, spread=0.0,
                          zero_override=None):
    """PV of `flows` off the zero curve, with a constant spread added.

    This is the Z-spread pricing kernel: every cashflow is discounted at the
    zero rate for its own maturity plus one common spread — as opposed to the
    YTM formula, which discounts everything at a single rate.

    `zero_override` lets key_rate_durations shock the curve at one tenor.
    """
    if not flows:
        return 0.0
    m = curve.frequency
    total = 0.0
    for dt, amt in flows:
        t = years_to_maturity(settle, dt)
        if t <= 0:
            continue
        z = zero_override(t) if zero_override else curve.zero(t)
        if z is None:
            return None
        base = 1.0 + (z + spread) / m
        if base <= 0:
            return None
        total += amt * base ** (-t * m)
    return total


def z_spread(dirty, flows, settle, curve, lo=-0.02, hi=0.50):
    """Constant spread over the zero curve that reprices `flows` to `dirty`.

    A bond priced exactly off the curve has a Z-spread of zero — which is the
    definitive joint test of the curve, the bootstrap, and this function.

    Returns None rather than raising when it will not solve; the caller sets a
    diagnostic and a rating cap handles the row.
    """
    if dirty is None or dirty <= 0 or not flows:
        return None

    def err(s):
        p = price_from_zero_curve(flows, settle, curve, spread=s)
        return None if p is None else p - dirty

    return brent(err, lo, hi, tol=1e-14)


def spread_to_price(spread, flows, settle, curve):
    """Inverse of z_spread: the dirty price implied by a given Z-spread.

    Used by the daily mark-to-curve overlay — the observed spread is aged
    forward on the bucket-OAS move, then converted back to a price.
    """
    return price_from_zero_curve(flows, settle, curve, spread=spread)


def nominal_spread(ytm, curve, maturity_years):
    """YTM minus the PAR yield at the same maturity. The quoted convention."""
    if ytm is None:
        return None
    p = curve.par(maturity_years)
    return None if p is None else ytm - p


def g_spread(ytm, curve, maturity_years):
    """YTM minus the interpolated government yield at the same maturity.

    Identical arithmetic to nominal_spread against a government curve; kept as
    a separate name because the report labels them differently and a reader
    should not have to guess which benchmark was used.
    """
    return nominal_spread(ytm, curve, maturity_years)


def yield_over_treasury(ytm, curve, maturity_years):
    """Alias used by the Valuation gates."""
    return nominal_spread(ytm, curve, maturity_years)


def is_likely_callable(row):
    """Heuristic: is this bond probably callable, given only free data?

    None of these signals is conclusive — that is the point. A True here does
    not assert a call feature, it flags that the Z-spread may be overstating
    compensation and that the row should not be trusted for a BUY without a
    schedule.
    """
    if row.get('call_schedule'):
        return True
    if row.get('is_callable') is True:
        return True
    # Corporate paper of 5y+ original tenor is very often callable; a price
    # meaningfully above par is where that optionality starts to bite.
    asset_class = row.get('asset_class') or ''
    if asset_class.startswith('CORP'):
        price = row.get('clean_price_est') or row.get('clean_price_marked')
        if price is not None and price > 100.5:
            return True
    return False


def fit_z_oas_wedge(model_z_by_bucket, fred_oas_by_bucket, min_months=3):
    """Fit the Z-minus-OAS wedge per rating bucket from observed history.

    Args:
        model_z_by_bucket: {bucket: {month: median model Z-spread}}
        fred_oas_by_bucket: {bucket: {month: published bucket OAS}}
        min_months: below this, report the wedge but mark it low-confidence.

    Returns:
        {bucket: {'wedge': float, 'n_months': int, 'confident': bool}}

    The wedge is the median of (model Z - published OAS) over the months where
    both exist. Median rather than mean because a single month of bad marks —
    a quarter where a large fund restated, say — should not move the fitted
    wedge for every bond in the bucket.
    """
    out = {}
    for bucket, z_months in model_z_by_bucket.items():
        oas_months = fred_oas_by_bucket.get(bucket) or {}
        diffs = sorted(z_months[mth] - oas_months[mth]
                       for mth in z_months
                       if mth in oas_months
                       and z_months[mth] is not None
                       and oas_months[mth] is not None)
        if not diffs:
            out[bucket] = {'wedge': 0.0, 'n_months': 0, 'confident': False}
            continue
        n = len(diffs)
        median = (diffs[n // 2] if n % 2
                  else 0.5 * (diffs[n // 2 - 1] + diffs[n // 2]))
        out[bucket] = {'wedge': median, 'n_months': n,
                       'confident': n >= min_months}
    return out
