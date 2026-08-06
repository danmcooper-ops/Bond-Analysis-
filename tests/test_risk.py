"""Duration, convexity and DV01.

The keystone test here is analytic-versus-numerical: `modified_duration` is
computed from PV-weighted cashflow times, `effective_duration` by bumping the
yield and repricing. They share no code beyond the pricer, so agreement to
1e-4 proves both implementations at once. An error would have to appear
identically in the weights and in the discounting to survive it.
"""

from datetime import date

import pytest

from models.pricing import bond_flows_and_stub, price_from_yield
from models.risk import (convexity, dv01, effective_convexity,
                         effective_duration, macaulay_duration,
                         modified_duration, price_change_estimate)
from models.schedule import cashflows


def _bond(coupon=0.05, years=10, y=0.05, settle=date(2026, 6, 15), frequency=2):
    maturity = date(2026 + years, 6, 15)
    flows, w = bond_flows_and_stub(coupon, maturity, settle, frequency=frequency)
    return flows, w, y, frequency


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def test_macaulay_duration_of_a_zero_equals_its_maturity():
    """The cleanest possible check that the PV weighting is right: a zero has
    exactly one cashflow, so its PV-weighted average time IS its maturity."""
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    flows = cashflows(100.0, 0.0, maturity, frequency=2, settle=settle)
    assert macaulay_duration(flows, 0.05, frequency=2, w=1.0) == pytest.approx(10.0, abs=1e-12)


@pytest.mark.parametrize('years', [1, 3, 5, 7, 10, 30])
def test_zero_duration_scales_with_maturity(years):
    settle = date(2026, 6, 15)
    maturity = date(2026 + years, 6, 15)
    flows = cashflows(100.0, 0.0, maturity, frequency=2, settle=settle)
    assert macaulay_duration(flows, 0.04, frequency=2) == pytest.approx(years, abs=1e-12)


def test_modified_is_macaulay_over_one_plus_y_over_m():
    flows, w, y, m = _bond()
    mac = macaulay_duration(flows, y, frequency=m, w=w)
    assert modified_duration(mac, y, frequency=m) == pytest.approx(mac / (1 + y / m), abs=1e-14)


def test_dv01_of_a_five_year_zero_at_zero_yield_is_exactly_five_cents():
    """At y=0 a 5y zero prices to 100 with modified duration 5, so DV01 is
    5 * 100 / 10000 = 0.05 on the nose."""
    settle, maturity = date(2026, 6, 15), date(2031, 6, 15)
    flows = cashflows(100.0, 0.0, maturity, frequency=2, settle=settle)
    price = price_from_yield(flows, 0.0, frequency=2, w=1.0)
    mac = macaulay_duration(flows, 0.0, frequency=2, w=1.0)
    mod = modified_duration(mac, 0.0, frequency=2)
    assert price == pytest.approx(100.0, abs=1e-12)
    assert mod == pytest.approx(5.0, abs=1e-12)
    assert dv01(price, mod) == pytest.approx(0.05, abs=1e-14)


# ---------------------------------------------------------------------------
# Analytic vs numerical — the keystone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('coupon,years,y', [
    (0.00, 10, 0.05), (0.05, 10, 0.05), (0.08, 5, 0.03),
    (0.02, 30, 0.06), (0.10, 2, 0.01), (0.045, 7, 0.0),
])
def test_analytic_and_numerical_duration_agree(coupon, years, y):
    flows, w, _, m = _bond(coupon=coupon, years=years, frequency=2)
    mac = macaulay_duration(flows, y, frequency=m, w=w)
    mod = modified_duration(mac, y, frequency=m)
    eff = effective_duration(flows, y, frequency=m, w=w, bump=1e-4)
    assert eff == pytest.approx(mod, abs=1e-4)


@pytest.mark.parametrize('coupon,years,y', [
    (0.00, 10, 0.05), (0.05, 10, 0.05), (0.08, 5, 0.03), (0.02, 30, 0.06),
])
def test_analytic_and_numerical_convexity_agree(coupon, years, y):
    """Not a tautology: the analytic form sums t(t + 1/m) weights while the
    numerical one takes a second difference of the price function."""
    flows, w, _, m = _bond(coupon=coupon, years=years, frequency=2)
    cvx = convexity(flows, y, frequency=m, w=w)
    eff = effective_convexity(flows, y, frequency=m, w=w, bump=1e-4)
    assert eff == pytest.approx(cvx, rel=1e-3)


def test_agreement_holds_off_a_coupon_date():
    """The stub factor has to appear identically in the pricer and in the
    duration weights, or these diverge for every real settlement date."""
    settle, maturity = date(2026, 8, 6), date(2034, 11, 15)
    flows, w = bond_flows_and_stub(0.0425, maturity, settle)
    assert 0 < w < 1
    mac = macaulay_duration(flows, 0.045, frequency=2, w=w)
    mod = modified_duration(mac, 0.045, frequency=2)
    eff = effective_duration(flows, 0.045, frequency=2, w=w)
    assert eff == pytest.approx(mod, abs=1e-4)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_duration_falls_as_coupon_rises():
    """A higher coupon returns more cash sooner, so the weighted average time
    to payment is shorter."""
    durations = []
    for coupon in (0.0, 0.02, 0.05, 0.08, 0.12):
        flows, w, y, m = _bond(coupon=coupon, years=20, y=0.05)
        durations.append(macaulay_duration(flows, y, frequency=m, w=w))
    assert durations == sorted(durations, reverse=True)


def test_duration_rises_with_maturity():
    durations = []
    for years in (1, 3, 5, 10, 20, 30):
        flows, w, y, m = _bond(coupon=0.05, years=years, y=0.05)
        durations.append(macaulay_duration(flows, y, frequency=m, w=w))
    assert durations == sorted(durations)


def test_convexity_rises_with_maturity():
    cvx = []
    for years in (2, 5, 10, 20, 30):
        flows, w, y, m = _bond(coupon=0.05, years=years, y=0.05)
        cvx.append(convexity(flows, y, frequency=m, w=w))
    assert cvx == sorted(cvx)


def test_convexity_is_positive_for_an_option_free_bond():
    flows, w, y, m = _bond(coupon=0.05, years=10)
    assert convexity(flows, y, frequency=m, w=w) > 0


# ---------------------------------------------------------------------------
# Estimation and failure behaviour
# ---------------------------------------------------------------------------

def test_second_order_estimate_beats_first_order_on_a_large_move():
    """Convexity earns its place: over a 100bp move the duration-only estimate
    is visibly short, and adding the convexity term closes most of the gap."""
    flows, w, y, m = _bond(coupon=0.05, years=30, y=0.05)
    p0 = price_from_yield(flows, y, frequency=m, w=w)
    dy = -0.01
    actual = price_from_yield(flows, y + dy, frequency=m, w=w) / p0 - 1.0

    mac = macaulay_duration(flows, y, frequency=m, w=w)
    mod = modified_duration(mac, y, frequency=m)
    cvx = convexity(flows, y, frequency=m, w=w)

    first_order = price_change_estimate(mod, None, dy)
    second_order = price_change_estimate(mod, cvx, dy)
    assert abs(second_order - actual) < abs(first_order - actual)
    # A 30y bond over 100bp leaves ~17bp of genuine third-order truncation;
    # the tolerance is set to that, not tighter, because a tighter bound would
    # be testing the Taylor series rather than the implementation.
    assert second_order == pytest.approx(actual, abs=2e-3)


def test_duration_convexity_error_is_third_order():
    """The real check on the expansion: halving the yield move must shrink the
    leftover error by roughly 8x. A residual that shrinks only 4x would mean
    the convexity term is wrong and is being absorbed as second-order error."""
    flows, w, y, m = _bond(coupon=0.05, years=30, y=0.05)
    p0 = price_from_yield(flows, y, frequency=m, w=w)
    mac = macaulay_duration(flows, y, frequency=m, w=w)
    mod = modified_duration(mac, y, frequency=m)
    cvx = convexity(flows, y, frequency=m, w=w)

    def residual(dy):
        actual = price_from_yield(flows, y + dy, frequency=m, w=w) / p0 - 1.0
        return abs(price_change_estimate(mod, cvx, dy) - actual)

    big, small = residual(-0.01), residual(-0.005)
    assert small > 0
    assert 6.0 < big / small < 10.0


def test_risk_measures_return_none_on_empty_flows():
    assert macaulay_duration([], 0.05) is None
    assert convexity([], 0.05) is None
    assert effective_duration([], 0.05) is None
    assert modified_duration(None, 0.05) is None
    assert dv01(None, 5.0) is None
    assert dv01(100.0, None) is None
    assert price_change_estimate(None, None, 0.01) is None
