"""Spread measures.

The definitive test: a bond priced exactly off the zero curve has a Z-spread
of exactly zero. That is a joint check on the bootstrap, the interpolator, the
discounting and the solver — all four have to be right for it to hold.
"""

from datetime import date

import pytest

from models.curve import YieldCurve
from models.pricing import bond_flows_and_stub, yield_to_maturity
from models.schedule import accrued_interest
from models.spreads import (fit_z_oas_wedge, is_likely_callable,
                            nominal_spread, price_from_zero_curve,
                            spread_to_price, z_spread)

AS_OF = date(2026, 8, 5)
SETTLE = date(2026, 8, 5)
MATURITY = date(2034, 11, 15)


@pytest.fixture
def curve(sample_par_curve):
    return YieldCurve.from_par_dict(AS_OF, sample_par_curve)


@pytest.fixture
def flat_curve(flat_par_curve):
    return YieldCurve.from_par_dict(AS_OF, flat_par_curve)


# ---------------------------------------------------------------------------
# Z-spread
# ---------------------------------------------------------------------------

def test_bond_priced_off_the_curve_has_zero_z_spread(curve):
    flows, _ = bond_flows_and_stub(0.045, MATURITY, SETTLE)
    dirty = price_from_zero_curve(flows, SETTLE, curve, spread=0.0)
    assert z_spread(dirty, flows, SETTLE, curve) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize('spread', [-0.005, 0.0, 0.0025, 0.01, 0.05, 0.20])
def test_z_spread_recovers_the_spread_it_was_priced_at(curve, spread):
    flows, _ = bond_flows_and_stub(0.05, MATURITY, SETTLE)
    dirty = price_from_zero_curve(flows, SETTLE, curve, spread=spread)
    assert z_spread(dirty, flows, SETTLE, curve) == pytest.approx(spread, abs=1e-10)


def test_spread_to_price_inverts_z_spread(curve):
    flows, _ = bond_flows_and_stub(0.05, MATURITY, SETTLE)
    dirty = 96.5
    z = z_spread(dirty, flows, SETTLE, curve)
    assert spread_to_price(z, flows, SETTLE, curve) == pytest.approx(dirty, abs=1e-9)


def test_cheaper_price_means_wider_spread(curve):
    flows, _ = bond_flows_and_stub(0.05, MATURITY, SETTLE)
    wide = z_spread(90.0, flows, SETTLE, curve)
    tight = z_spread(105.0, flows, SETTLE, curve)
    assert wide > tight


def test_z_spread_equals_nominal_spread_on_a_flat_curve(flat_curve):
    """With no curve shape there is nothing for a Z-spread to capture that a
    nominal spread misses, so the two must agree closely."""
    flows, w = bond_flows_and_stub(0.06, MATURITY, SETTLE)
    dirty = 98.0
    accrued = accrued_interest(SETTLE, 0.06, MATURITY)
    z = z_spread(dirty, flows, SETTLE, flat_curve)
    ytm = yield_to_maturity(dirty - accrued, 0.06, MATURITY, SETTLE)
    nominal = nominal_spread(ytm, flat_curve, (MATURITY - SETTLE).days / 365.25)
    assert z == pytest.approx(nominal, abs=5e-4)


def test_z_and_nominal_spread_diverge_on_a_steep_curve(curve, flat_curve):
    """On a sloped curve they must NOT agree — a Z-spread that tracked the
    nominal spread everywhere would mean the term structure is being ignored."""
    flows, _ = bond_flows_and_stub(0.06, MATURITY, SETTLE)
    dirty = 98.0
    accrued = accrued_interest(SETTLE, 0.06, MATURITY)
    z = z_spread(dirty, flows, SETTLE, curve)
    ytm = yield_to_maturity(dirty - accrued, 0.06, MATURITY, SETTLE)
    nominal = nominal_spread(ytm, curve, (MATURITY - SETTLE).days / 365.25)
    assert abs(z - nominal) > 1e-5


def test_z_spread_returns_none_on_unusable_input(curve):
    flows, _ = bond_flows_and_stub(0.05, MATURITY, SETTLE)
    assert z_spread(None, flows, SETTLE, curve) is None
    assert z_spread(0.0, flows, SETTLE, curve) is None
    assert z_spread(98.0, [], SETTLE, curve) is None


def test_price_from_zero_curve_ignores_flows_at_or_before_settle(curve):
    flows = [(date(2020, 1, 1), 2.5), (MATURITY, 102.5)]
    only_future = price_from_zero_curve([(MATURITY, 102.5)], SETTLE, curve)
    assert price_from_zero_curve(flows, SETTLE, curve) == pytest.approx(only_future)


# ---------------------------------------------------------------------------
# Nominal / G spread
# ---------------------------------------------------------------------------

def test_nominal_spread_is_yield_over_the_matched_government_point(curve):
    assert nominal_spread(0.0625, curve, 10.0) == pytest.approx(0.0625 - curve.par(10.0))
    assert nominal_spread(None, curve, 10.0) is None


# ---------------------------------------------------------------------------
# Callability heuristic
# ---------------------------------------------------------------------------

def test_explicit_call_data_wins():
    assert is_likely_callable({'call_schedule': [(date(2030, 1, 1), 100.0)]}) is True
    assert is_likely_callable({'is_callable': True}) is True


def test_premium_priced_corporate_is_flagged_as_probably_callable():
    """Not an assertion that it IS callable — a flag that the Z-spread may be
    overstating compensation and the row should not be trusted for a BUY."""
    assert is_likely_callable(
        {'asset_class': 'CORP_HY', 'clean_price_est': 104.0}) is True
    assert is_likely_callable(
        {'asset_class': 'CORP_IG', 'clean_price_est': 98.0}) is False


def test_treasuries_are_not_flagged_on_price_alone():
    """Treasuries have not been issued callable since the 1980s; a premium
    price must not drag them into the callable cap."""
    assert is_likely_callable(
        {'asset_class': 'TREASURY', 'clean_price_est': 108.0}) is False


# ---------------------------------------------------------------------------
# The fitted Z-minus-OAS wedge
# ---------------------------------------------------------------------------

def test_wedge_is_the_median_gap_per_bucket():
    model_z = {'BBB': {'2026-01': 0.0180, '2026-02': 0.0190, '2026-03': 0.0200}}
    fred = {'BBB': {'2026-01': 0.0150, '2026-02': 0.0155, '2026-03': 0.0165}}
    out = fit_z_oas_wedge(model_z, fred)
    assert out['BBB']['wedge'] == pytest.approx(0.0035)
    assert out['BBB']['n_months'] == 3
    assert out['BBB']['confident'] is True


def test_wedge_uses_the_median_so_one_bad_month_cannot_move_it():
    """A quarter where a large fund restates its marks should not reprice the
    fair spread of every bond in the bucket."""
    model_z = {'BB': {'m1': 0.030, 'm2': 0.031, 'm3': 0.032,
                      'm4': 0.033, 'm5': 0.500}}
    fred = {'BB': {'m1': 0.025, 'm2': 0.026, 'm3': 0.027,
                   'm4': 0.028, 'm5': 0.029}}
    assert fit_z_oas_wedge(model_z, fred)['BB']['wedge'] == pytest.approx(0.005)


def test_wedge_reports_low_confidence_on_thin_history():
    out = fit_z_oas_wedge({'A': {'m1': 0.012}}, {'A': {'m1': 0.010}})
    assert out['A']['n_months'] == 1
    assert out['A']['confident'] is False


def test_wedge_defaults_to_zero_when_there_is_no_overlap():
    out = fit_z_oas_wedge({'AAA': {'m1': 0.006}}, {'AAA': {'m9': 0.005}})
    assert out['AAA'] == {'wedge': 0.0, 'n_months': 0, 'confident': False}


def test_wedge_skips_months_with_missing_values():
    model_z = {'A': {'m1': 0.012, 'm2': None, 'm3': 0.014}}
    fred = {'A': {'m1': 0.010, 'm2': 0.011, 'm3': 0.011}}
    out = fit_z_oas_wedge(model_z, fred)
    assert out['A']['n_months'] == 2
