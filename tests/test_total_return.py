"""Total return and its decomposition.

The decomposition's honesty check is the residual. On a scenario constructed
so every component is exactly zero except income, the residual must be zero
too — if it is not, some component is absorbing error it should be reporting.
"""

from datetime import date

import pytest

from models.bond_types import Bond
from models.curve import YieldCurve
from models.daycount import D30_360
from models.pricing import bond_flows_and_stub
from models.risk import convexity, macaulay_duration, modified_duration
from models.schedule import accrued_interest
from models.total_return import (carry, coupons_between,
                                 decompose_total_return,
                                 duration_matched_treasury_return,
                                 realized_total_return, roll_down)

MATURITY = date(2036, 6, 15)


def _bond(coupon=0.05, maturity=MATURITY):
    return Bond(cusip='000000AA1', issuer_name='ACME CORP', coupon_rate=coupon,
                maturity=maturity, frequency=2, convention=D30_360,
                comp='semiannual', asset_class='CORP_IG', face=100.0)


# ---------------------------------------------------------------------------
# Realised total return
# ---------------------------------------------------------------------------

def test_realized_total_return_counts_coupons_and_accrual():
    # Bought at 98 clean, sold at 99 clean, one 2.5 coupon collected.
    r = realized_total_return(98.0, 99.0, 0.0, 0.0, 2.5)
    assert r == pytest.approx((99.0 + 2.5 - 98.0) / 98.0)


def test_return_is_denominated_on_the_dirty_price():
    """The buyer paid clean + accrued. Using the clean price as the base
    overstates the return of any bond bought mid-period."""
    on_dirty = realized_total_return(98.0, 98.0, 1.5, 1.5, 0.0)
    assert on_dirty == pytest.approx(0.0)
    r = realized_total_return(98.0, 99.0, 1.5, 0.0, 2.5)
    assert r == pytest.approx((99.0 + 0.0 + 2.5 - 99.5) / 99.5)


def test_price_return_alone_would_understate_a_high_coupon_bond():
    """The reason this module exists: ranking on price change alone sorts the
    book roughly by coupon, backwards."""
    price_only = (100.0 - 100.0) / 100.0
    total = realized_total_return(100.0, 100.0, 0.0, 0.0, 5.0)
    assert price_only == 0.0
    assert total == pytest.approx(0.05)


def test_realized_total_return_guards_bad_input():
    assert realized_total_return(None, 99.0, 0, 0, 0) is None
    assert realized_total_return(98.0, None, 0, 0, 0) is None
    assert realized_total_return(0.0, 99.0, 0.0, 0, 0) is None


# ---------------------------------------------------------------------------
# Coupons in a window
# ---------------------------------------------------------------------------

def test_coupons_between_counts_actual_payment_dates():
    """Computed from the real schedule, not pro-rated — whether a payment date
    falls inside the window is a discrete fact a pro-rata estimate gets wrong
    exactly when it matters."""
    assert coupons_between(0.05, MATURITY, date(2026, 6, 15),
                           date(2026, 12, 15)) == pytest.approx(2.5)
    # A window that just misses the payment date collects nothing.
    assert coupons_between(0.05, MATURITY, date(2026, 6, 15),
                           date(2026, 12, 14)) == pytest.approx(0.0)
    # A full year collects two.
    assert coupons_between(0.05, MATURITY, date(2026, 6, 15),
                           date(2027, 6, 15)) == pytest.approx(5.0)


def test_coupons_between_excludes_the_principal_repayment():
    """Principal returned is not income; counting it would show a 100% return
    in the month a bond matures."""
    total = coupons_between(0.05, date(2026, 12, 15), date(2026, 6, 15),
                            date(2027, 1, 15))
    assert total == pytest.approx(2.5)


def test_coupons_between_is_zero_for_a_discount_instrument():
    assert coupons_between(0.0, MATURITY, date(2026, 6, 15), date(2027, 6, 15)) == 0.0


# ---------------------------------------------------------------------------
# Carry and roll-down
# ---------------------------------------------------------------------------

def test_carry_is_coupon_income_over_the_dirty_price():
    assert carry(0.05, 100.0, 180) == pytest.approx(0.025, abs=1e-9)
    assert carry(0.05, None, 180) is None


def test_roll_down_is_positive_on_an_upward_sloping_curve(sample_par_curve):
    """A bond ages into a lower yield, so its price rises."""
    curve = YieldCurve.from_par_dict(date(2026, 8, 5), sample_par_curve)
    assert roll_down(curve, maturity_years=10.0, horizon_years=1.0,
                     mod_duration=7.5) > 0


def test_roll_down_is_zero_on_a_flat_curve(flat_par_curve):
    curve = YieldCurve.from_par_dict(date(2026, 8, 5), flat_par_curve)
    assert roll_down(curve, 10.0, 1.0, 7.5) == pytest.approx(0.0, abs=1e-9)


def test_roll_down_is_negative_on_an_inverted_curve():
    curve = YieldCurve.from_par_dict(date(2026, 8, 5), {
        '6M': 0.055, '1Y': 0.052, '2Y': 0.048, '5Y': 0.044,
        '10Y': 0.042, '30Y': 0.040})
    assert roll_down(curve, 10.0, 1.0, 7.5) < 0


def test_roll_down_returns_none_past_maturity(sample_par_curve):
    curve = YieldCurve.from_par_dict(date(2026, 8, 5), sample_par_curve)
    assert roll_down(curve, 0.5, 1.0, 0.5) is None
    assert roll_down(curve, 10.0, 1.0, None) is None


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def test_residual_is_zero_when_only_income_is_in_play(flat_par_curve):
    """Flat curve, unchanged rates, unchanged spread, exactly one coupon
    period, same clean price at both ends. Every component except income is
    identically zero, so the residual must be too — a non-zero residual here
    would mean a component is absorbing error rather than reporting it."""
    as_of0, as_of1 = date(2026, 6, 15), date(2026, 12, 15)
    curve0 = YieldCurve.from_par_dict(as_of0, flat_par_curve)
    curve1 = YieldCurve.from_par_dict(as_of1, flat_par_curve)
    bond = _bond(coupon=0.05)

    flows, w = bond_flows_and_stub(0.05, MATURITY, as_of0)
    mac = macaulay_duration(flows, 0.05, frequency=2, w=w)
    mod = modified_duration(mac, 0.05, frequency=2)
    cvx = convexity(flows, 0.05, frequency=2, w=w)

    out = decompose_total_return(
        bond, p0_clean=100.0, p1_clean=100.0, settle0=as_of0, settle1=as_of1,
        curve0=curve0, curve1=curve1, ytm0=0.05, mod_duration=mod,
        convexity_=cvx, z_spread0=0.01, z_spread1=0.01)

    assert out['coupons_paid'] == pytest.approx(2.5)
    assert out['total'] == pytest.approx(0.025)
    assert out['income'] == pytest.approx(0.025)
    assert out['rate'] == pytest.approx(0.0, abs=1e-12)
    assert out['spread'] == pytest.approx(0.0, abs=1e-12)
    assert out['roll_down'] == pytest.approx(0.0, abs=1e-9)
    assert out['residual'] == pytest.approx(0.0, abs=1e-9)


def test_components_are_reported_even_when_they_do_not_sum(sample_par_curve):
    """A real scenario leaves a residual. It has to be present and finite —
    the point is that it is visible, not that it is zero."""
    as_of0, as_of1 = date(2026, 6, 15), date(2026, 12, 15)
    curve0 = YieldCurve.from_par_dict(as_of0, sample_par_curve)
    shifted = {k: v + 0.005 for k, v in sample_par_curve.items()}
    curve1 = YieldCurve.from_par_dict(as_of1, shifted)
    bond = _bond(coupon=0.05)

    flows, w = bond_flows_and_stub(0.05, MATURITY, as_of0)
    mac = macaulay_duration(flows, 0.05, frequency=2, w=w)
    mod = modified_duration(mac, 0.05, frequency=2)
    cvx = convexity(flows, 0.05, frequency=2, w=w)

    out = decompose_total_return(
        bond, p0_clean=100.0, p1_clean=96.5, settle0=as_of0, settle1=as_of1,
        curve0=curve0, curve1=curve1, ytm0=0.05, mod_duration=mod,
        convexity_=cvx, z_spread0=0.010, z_spread1=0.013)

    for key in ('total', 'income', 'roll_down', 'rate', 'spread',
                'convexity', 'residual'):
        assert out[key] is not None
    assert out['rate'] < 0                      # yields rose
    assert out['spread'] < 0                    # spread widened
    explained = sum(out[k] for k in
                    ('income', 'roll_down', 'rate', 'spread', 'convexity'))
    assert out['residual'] == pytest.approx(out['total'] - explained, abs=1e-12)


def test_decomposition_rejects_a_non_positive_horizon(sample_par_curve):
    d = date(2026, 6, 15)
    curve = YieldCurve.from_par_dict(d, sample_par_curve)
    assert decompose_total_return(_bond(), 100.0, 100.0, d, d, curve, curve,
                                  0.05, 7.0, 60.0, 0.01, 0.01) is None


# ---------------------------------------------------------------------------
# Duration-matched Treasury benchmark
# ---------------------------------------------------------------------------

def test_duration_matched_treasury_earns_carry_when_rates_are_unchanged(
        sample_par_curve):
    curve = YieldCurve.from_par_dict(date(2026, 8, 5), sample_par_curve)
    r = duration_matched_treasury_return(7.5, 60.0, curve, curve,
                                         maturity_years=10.0, horizon_days=365)
    # Year fractions here use 365.25 days, so a 365-day horizon earns very
    # slightly less than the full annual yield.
    assert r == pytest.approx(curve.par(10.0) * 365 / 365.25, abs=1e-12)
    assert r == pytest.approx(curve.par(10.0), rel=1e-3)


def test_duration_matched_treasury_loses_when_yields_rise(sample_par_curve):
    """The benchmark that isolates credit skill from the duration bet: beating
    LQD by being long duration in a rally is not credit skill."""
    curve0 = YieldCurve.from_par_dict(date(2026, 8, 5), sample_par_curve)
    curve1 = curve0.shift(100)
    r = duration_matched_treasury_return(7.5, 60.0, curve0, curve1, 10.0, 365)
    assert r < 0


def test_duration_matched_treasury_guards_missing_inputs(sample_par_curve):
    curve = YieldCurve.from_par_dict(date(2026, 8, 5), sample_par_curve)
    assert duration_matched_treasury_return(None, 60.0, curve, curve, 10.0, 365) is None
    assert duration_matched_treasury_return(7.5, 60.0, curve, curve, None, 365) is None
