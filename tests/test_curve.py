"""Curve bootstrapping and interpolation.

The definitive test is the bootstrap round-trip: take a par curve, bootstrap
zeros from it, then reprice each quoted par bond off those zeros. Every one
must come back to exactly 100. That single check exercises the bootstrap, the
interpolator, and the discounting together — if any of the three is wrong, it
fails.
"""

from datetime import date

import pytest

from models.curve import (TENOR_YEARS, YieldCurve, bootstrap_zero_curve,
                          interpolate)

AS_OF = date(2026, 8, 5)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_round_trip_reprices_every_par_bond_to_100(sample_par_curve):
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    assert curve.repricing_error() < 1e-8


def test_bootstrap_round_trip_on_a_steep_curve():
    """A steeper curve stresses the interpolator harder — the gap between
    quoted tenors is where a bad interpolation shows up."""
    par = {'6M': 0.02, '1Y': 0.025, '2Y': 0.032, '3Y': 0.038, '5Y': 0.046,
           '7Y': 0.052, '10Y': 0.058, '20Y': 0.066, '30Y': 0.070}
    curve = YieldCurve.from_par_dict(AS_OF, par)
    assert curve.repricing_error() < 1e-8


def test_bootstrap_round_trip_on_an_inverted_curve():
    par = {'3M': 0.055, '6M': 0.054, '1Y': 0.051, '2Y': 0.047, '3Y': 0.045,
           '5Y': 0.043, '7Y': 0.0425, '10Y': 0.042, '30Y': 0.041}
    curve = YieldCurve.from_par_dict(AS_OF, par)
    assert curve.repricing_error() < 1e-8


def test_flat_par_gives_flat_zeros_and_flat_forwards(flat_par_curve):
    """On a flat curve, par == zero == forward. Any deviation means the
    bootstrap is introducing structure that is not in the data."""
    curve = YieldCurve.from_par_dict(AS_OF, flat_par_curve)
    for t in (0.5, 1, 2, 3.7, 5, 8.25, 10, 20, 30):
        assert curve.zero(t) == pytest.approx(0.05, abs=1e-9)
        assert curve.par(t) == pytest.approx(0.05, abs=1e-12)
    for t1, t2 in ((1, 2), (2, 5), (5, 10), (10, 30)):
        assert curve.forward(t1, t2) == pytest.approx(0.05, abs=1e-8)


def test_upward_sloping_par_puts_zeros_above_par_at_the_long_end(sample_par_curve):
    """Standard result: when the par curve slopes up, the zero curve sits
    above it, because early coupons are discounted at lower short rates."""
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    for t in (5, 10, 20, 30):
        assert curve.zero(t) > curve.par(t)


def test_inverted_par_puts_zeros_below_par_at_the_long_end():
    par = {'6M': 0.055, '1Y': 0.052, '2Y': 0.048, '5Y': 0.044,
           '10Y': 0.042, '30Y': 0.040}
    curve = YieldCurve.from_par_dict(AS_OF, par)
    for t in (5, 10, 30):
        assert curve.zero(t) < curve.par(t)


def test_bootstrap_handles_a_single_short_tenor():
    zeros = bootstrap_zero_curve([0.25], [0.043])
    assert zeros == [(0.25, 0.043)]


def test_bootstrap_skips_non_positive_tenors():
    zeros = bootstrap_zero_curve([0.0, -1.0, 5.0], [0.04, 0.04, 0.05])
    assert [t for t, _ in zeros] == [5.0]


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def test_interpolation_passes_through_every_knot(sample_par_curve):
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    for label, y in sample_par_curve.items():
        assert curve.par(TENOR_YEARS[label]) == pytest.approx(y, abs=1e-12)


def test_extrapolation_is_flat_not_sloped(sample_par_curve):
    """Extrapolating the long-end slope produces absurd 40-year discount
    factors, and there is nothing to calibrate an extrapolation against."""
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    assert curve.par(50.0) == pytest.approx(curve.par(30.0))
    assert curve.par(100.0) == pytest.approx(curve.par(30.0))
    assert curve.par(0.001) == pytest.approx(sample_par_curve['1M'])
    assert curve.zero(45.0) == pytest.approx(curve.zero(30.0))


def test_monotone_cubic_preserves_monotonicity():
    """The whole reason for choosing PCHIP: it cannot invent a wiggle between
    two monotone knots. A cubic spline would overshoot here."""
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [0.0, 0.9, 0.95, 0.96, 0.99, 1.0]        # monotone but sharply kinked
    prev = None
    for i in range(501):
        x = i * 5.0 / 500
        v = interpolate(x, xs, ys)
        if prev is not None:
            assert v >= prev - 1e-12, f"overshoot at x={x}"
        prev = v


def test_monotone_cubic_stays_within_the_data_range():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 1.0, 0.0]
    for i in range(301):
        x = i * 3.0 / 300
        assert -1e-12 <= interpolate(x, xs, ys) <= 1.0 + 1e-12


def test_monotone_cubic_flattens_at_a_local_extremum():
    """At a turning point the tangent is pinned to zero, so the interpolant
    does not shoot past the peak."""
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 0.0]
    assert interpolate(1.0, xs, ys) == pytest.approx(1.0)
    assert max(interpolate(i / 100, xs, ys) for i in range(201)) <= 1.0 + 1e-12


def test_linear_method_still_available():
    assert interpolate(0.5, [0.0, 1.0], [0.0, 10.0], method='linear') == pytest.approx(5.0)


def test_interpolation_degenerate_inputs():
    assert interpolate(1.0, [], []) is None
    assert interpolate(1.0, [2.0], [0.05]) == 0.05
    assert interpolate(0.5, [0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Curve object
# ---------------------------------------------------------------------------

def test_discount_factors_are_positive_and_decreasing(sample_par_curve):
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    dfs = [curve.discount(t) for t in (0.5, 1, 2, 5, 10, 20, 30)]
    assert all(d > 0 for d in dfs)
    assert dfs == sorted(dfs, reverse=True)


def test_discount_accepts_a_spread(sample_par_curve):
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    assert curve.discount(10.0, spread=0.01) < curve.discount(10.0)


def test_shift_moves_the_whole_curve(sample_par_curve):
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    up = curve.shift(25)
    for t in (1, 5, 10, 30):
        assert up.par(t) == pytest.approx(curve.par(t) + 0.0025, abs=1e-12)
    assert up.repricing_error() < 1e-8


def test_forward_rates_exceed_spot_on_an_upward_curve(sample_par_curve):
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    assert curve.forward(5, 10) > curve.zero(5)


def test_forward_rejects_a_reversed_or_degenerate_interval(sample_par_curve):
    curve = YieldCurve.from_par_dict(AS_OF, sample_par_curve)
    assert curve.forward(10, 5) is None
    assert curve.forward(5, 5) is None


def test_from_par_dict_ignores_unknown_tenors_and_nulls():
    curve = YieldCurve.from_par_dict(
        AS_OF, {'2Y': 0.045, '10Y': 0.051, '17Y': 0.055, '30Y': None})
    assert curve.par_t == [2.0, 10.0]


def test_from_par_dict_rejects_an_empty_curve():
    with pytest.raises(ValueError, match='No recognisable tenors'):
        YieldCurve.from_par_dict(AS_OF, {'nonsense': 0.05})
