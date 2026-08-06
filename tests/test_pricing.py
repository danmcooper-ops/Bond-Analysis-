"""Pricing golden values.

Bond math has published answers, so these are not smoke tests — every
expected number below is either a closed form or a textbook figure, and a
failure means the arithmetic is wrong, not that a fixture drifted.

The par identity is the single highest-value test in the suite: a bond priced
at a yield equal to its coupon must price to exactly 100 on a coupon date, at
any frequency and under any convention. Almost every sign error, off-by-one in
the discount exponent, and stub-factor mistake breaks it.
"""

from datetime import date

import pytest

from models.daycount import ACT_ACT, D30_360
from models.pricing import (COMP_SEMIANNUAL, bond_flows_and_stub,
                            current_yield, price_bond, price_from_yield,
                            yield_from_price, yield_to_call,
                            yield_to_maturity, yield_to_worst)
from models.schedule import accrued_interest, cashflows, clean_price, dirty_price


# ---------------------------------------------------------------------------
# The par identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('frequency', [1, 2, 4])
@pytest.mark.parametrize('convention', [D30_360, ACT_ACT])
@pytest.mark.parametrize('coupon', [0.02, 0.05, 0.0875])
def test_par_identity(frequency, convention, coupon):
    """Priced at its own coupon, on a coupon date, a bond is worth exactly par."""
    settle = date(2026, 6, 15)
    maturity = date(2036, 6, 15)
    clean, dirty, accrued = price_bond(coupon, maturity, settle, coupon,
                                       frequency=frequency,
                                       convention=convention)
    assert accrued == pytest.approx(0.0, abs=1e-12)
    assert dirty == pytest.approx(100.0, abs=1e-9)
    assert clean == pytest.approx(100.0, abs=1e-9)


def test_par_identity_holds_for_a_short_bond_too():
    settle, maturity = date(2026, 6, 15), date(2027, 6, 15)
    clean, _, _ = price_bond(0.05, maturity, settle, 0.05)
    assert clean == pytest.approx(100.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

def test_zero_coupon_closed_form():
    """A 10y zero at 5% semiannual is 100 / 1.025^20 = 61.02709...

    Built as a 5%-frequency schedule with a zero coupon so the discount
    exponent path is the same one a real bond takes.
    """
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    flows = cashflows(100.0, 0.0, maturity, frequency=2, settle=settle)
    price = price_from_yield(flows, 0.05, frequency=2, w=1.0)
    assert price == pytest.approx(100.0 / 1.025 ** 20, abs=1e-9)
    assert price == pytest.approx(61.027094, abs=1e-6)


def test_textbook_annuity_discount():
    """8% coupon, 5 years, semiannual, priced at a 10% yield -> 92.2783."""
    settle, maturity = date(2026, 6, 15), date(2031, 6, 15)
    clean, _, _ = price_bond(0.08, maturity, settle, 0.10)
    assert clean == pytest.approx(92.2783, abs=1e-4)


def test_textbook_annuity_premium():
    """Same bond at a 6% yield -> 108.5302."""
    settle, maturity = date(2026, 6, 15), date(2031, 6, 15)
    clean, _, _ = price_bond(0.08, maturity, settle, 0.06)
    assert clean == pytest.approx(108.5302, abs=1e-4)


def test_price_moves_inversely_to_yield():
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    prices = [price_bond(0.05, maturity, settle, y)[0]
              for y in (0.03, 0.04, 0.05, 0.06, 0.07)]
    assert prices == sorted(prices, reverse=True)
    assert prices[2] == pytest.approx(100.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('y', [-0.005, 0.001, 0.02, 0.05, 0.10, 0.30])
def test_yield_price_round_trip(y):
    """price(yield(price)) is the identity to 1e-10 — INCLUDING a negative
    yield, because the solver bracket has to reach below zero and a bracket
    that starts at 0 silently fails on any negative-yielding paper."""
    settle, maturity = date(2026, 3, 10), date(2033, 9, 15)
    flows, w = bond_flows_and_stub(0.045, maturity, settle)
    price = price_from_yield(flows, y, frequency=2, w=w)
    recovered = yield_from_price(price, flows, frequency=2, w=w)
    assert recovered is not None
    assert recovered == pytest.approx(y, abs=1e-10)


def test_round_trip_survives_a_mid_period_settlement():
    """Off a coupon date the stub factor is in play; the round trip has to
    hold there too or every real settlement date is slightly wrong."""
    settle, maturity = date(2026, 8, 6), date(2034, 11, 15)
    flows, w = bond_flows_and_stub(0.0425, maturity, settle)
    assert 0 < w < 1
    price = price_from_yield(flows, 0.0475, frequency=2, w=w)
    assert yield_from_price(price, flows, frequency=2, w=w) == pytest.approx(0.0475, abs=1e-10)


# ---------------------------------------------------------------------------
# Failure behaviour
# ---------------------------------------------------------------------------

def test_solver_returns_none_rather_than_raising():
    """A 30,000-row run cannot afford one pathological bond to raise."""
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    flows, w = bond_flows_and_stub(0.05, maturity, settle)
    assert yield_from_price(0.0, flows, frequency=2, w=w) is None
    assert yield_from_price(-5.0, flows, frequency=2, w=w) is None
    assert yield_from_price(None, flows, frequency=2, w=w) is None
    assert yield_from_price(100.0, [], frequency=2, w=w) is None


def test_matured_bond_has_no_cashflows():
    assert cashflows(100.0, 0.05, date(2020, 1, 1), settle=date(2026, 6, 15)) == []
    assert yield_to_maturity(99.0, 0.05, date(2020, 1, 1), date(2026, 6, 15)) is None


def test_price_from_yield_returns_none_outside_its_domain():
    """1 + y/m <= 0 has no meaning; it must not raise a complex or a ZeroDivision."""
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    flows, w = bond_flows_and_stub(0.05, maturity, settle)
    assert price_from_yield(flows, -3.0, frequency=2, w=w) is None


# ---------------------------------------------------------------------------
# Accrued interest
# ---------------------------------------------------------------------------

def test_accrued_is_zero_on_a_coupon_date():
    assert accrued_interest(date(2026, 6, 15), 0.05, date(2036, 6, 15)) == pytest.approx(0.0, abs=1e-12)


def test_accrued_approaches_a_full_coupon_before_the_next_payment():
    """One day before the next coupon, 179 of 180 days have accrued."""
    a = accrued_interest(date(2026, 12, 14), 0.05, date(2036, 6, 15))
    assert a == pytest.approx(2.5 * 179 / 180, abs=1e-9)


def test_accrued_is_half_a_coupon_at_mid_period():
    """30/360 makes this exact: 90 of 180 days."""
    a = accrued_interest(date(2026, 9, 15), 0.05, date(2036, 6, 15))
    assert a == pytest.approx(1.25, abs=1e-9)


def test_dirty_minus_clean_is_identically_accrued():
    settle, maturity = date(2026, 8, 6), date(2034, 11, 15)
    clean, dirty, accrued = price_bond(0.0425, maturity, settle, 0.0475)
    assert dirty - clean == pytest.approx(accrued, abs=1e-12)
    assert dirty_price(clean, accrued) == pytest.approx(dirty)
    assert clean_price(dirty, accrued) == pytest.approx(clean)


def test_zero_coupon_never_accrues():
    assert accrued_interest(date(2026, 8, 6), 0.0, date(2030, 1, 1)) == 0.0


# ---------------------------------------------------------------------------
# Yield variants
# ---------------------------------------------------------------------------

def test_current_yield():
    assert current_yield(80.0, 0.05) == pytest.approx(0.0625)
    assert current_yield(None, 0.05) is None
    assert current_yield(0.0, 0.05) is None


def test_yield_to_call_exceeds_ytm_for_a_discount_bond_called_at_par():
    """Bought below par and redeemed early at par, the pull-to-par is
    compressed into fewer years, so YTC > YTM."""
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    call_date = date(2029, 6, 15)
    ytm = yield_to_maturity(90.0, 0.05, maturity, settle)
    ytc = yield_to_call(90.0, 0.05, call_date, 100.0, settle)
    assert ytc > ytm


def test_yield_to_worst_picks_the_lowest_and_says_where_it_came_from():
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    schedule = [(date(2029, 6, 15), 102.0), (date(2031, 6, 15), 100.0)]
    res = yield_to_worst(110.0, 0.06, maturity, settle, call_schedule=schedule)
    assert res['to_type'] == 'call'
    assert res['call_data_available'] is True
    ytm = yield_to_maturity(110.0, 0.06, maturity, settle)
    assert res['ytw'] < ytm


def test_yield_to_worst_without_a_schedule_is_explicit_about_it():
    """The distinction between 'no calls exist' and 'we have no call data' is
    what a rating cap keys off — it must not be silently collapsed."""
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    res = yield_to_worst(105.0, 0.06, maturity, settle, call_schedule=None)
    assert res['to_type'] == 'maturity'
    assert res['call_data_available'] is False
    assert res['ytw'] == pytest.approx(
        yield_to_maturity(105.0, 0.06, maturity, settle))


def test_past_call_dates_are_ignored():
    settle, maturity = date(2026, 6, 15), date(2036, 6, 15)
    stale = [(date(2024, 6, 15), 102.0)]
    res = yield_to_worst(110.0, 0.06, maturity, settle, call_schedule=stale)
    assert res['to_type'] == 'maturity'
