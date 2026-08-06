"""Day-count conventions, table-driven against the published rules."""

from datetime import date

import pytest

from models.daycount import (ACT_360, ACT_365F, ACT_ACT, D30_360,
                             accrual_fraction, day_count, days_30_360,
                             year_fraction)


# ---------------------------------------------------------------------------
# 30U/360, including the end-of-month refinements
# ---------------------------------------------------------------------------

# (start, end, expected days, note)
CASES_30_360 = [
    ((2026, 1, 15), (2026, 7, 15), 180, 'plain semiannual period'),
    ((2026, 1, 1), (2027, 1, 1), 360, 'a full year'),
    ((2026, 1, 30), (2026, 2, 28), 28, 'no EOM rule fires on the start date'),
    # Rule 4: D1 == 31 -> 30.
    ((2026, 1, 31), (2026, 2, 28), 28, 'D1 31 becomes 30'),
    # Rule 3: D2 == 31 with D1 in (30, 31) -> D2 = 30.
    ((2026, 1, 30), (2026, 3, 31), 60, 'D2 31 becomes 30 because D1 is 30'),
    ((2026, 1, 31), (2026, 3, 31), 60, 'both become 30'),
    # Rule 3 must NOT fire when D1 is neither 30 nor 31.
    ((2026, 1, 29), (2026, 3, 31), 62, 'D2 stays 31 because D1 is 29'),
    # Rule 2: D1 is last-of-Feb -> 30.
    ((2026, 2, 28), (2026, 8, 31), 180, 'last-of-Feb start becomes the 30th'),
    # Rule 1: last-of-Feb to last-of-Feb -> both 30.
    ((2026, 2, 28), (2027, 2, 28), 360, 'Feb to Feb is a clean year'),
    ((2024, 2, 29), (2025, 2, 28), 360, 'leap-year Feb to Feb'),
]


@pytest.mark.parametrize('start,end,expected,note', CASES_30_360)
def test_days_30_360(start, end, expected, note):
    assert days_30_360(date(*start), date(*end)) == expected, note


def test_30_360_eom_flag_can_be_switched_off():
    """Without the EOM refinement, last-of-Feb is not promoted to the 30th."""
    d1, d2 = date(2026, 2, 28), date(2026, 8, 31)
    assert days_30_360(d1, d2, eom=True) == 180
    assert days_30_360(d1, d2, eom=False) == 183


def test_30_360_is_antisymmetric():
    d1, d2 = date(2026, 3, 15), date(2029, 9, 15)
    assert days_30_360(d1, d2) == -days_30_360(d2, d1)


def test_30_360_year_fraction():
    assert year_fraction(date(2026, 1, 15), date(2026, 7, 15), D30_360) == pytest.approx(0.5)
    assert year_fraction(date(2026, 1, 1), date(2027, 1, 1), D30_360) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ACT conventions
# ---------------------------------------------------------------------------

def test_act_360_on_a_91_day_bill_is_exact():
    """91/360 exactly — no rounding, no approximation."""
    start, end = date(2026, 1, 1), date(2026, 4, 2)
    assert (end - start).days == 91
    assert year_fraction(start, end, ACT_360) == pytest.approx(91 / 360, abs=1e-15)


def test_act_365f():
    start, end = date(2026, 1, 1), date(2027, 1, 1)
    assert year_fraction(start, end, ACT_365F) == pytest.approx(365 / 365, abs=1e-15)
    # A leap year is 366/365 under ACT/365 Fixed — that is the point of "Fixed".
    assert year_fraction(date(2024, 1, 1), date(2025, 1, 1), ACT_365F) == \
        pytest.approx(366 / 365, abs=1e-15)


def test_act_act_icma_uses_the_enclosing_period():
    """A full coupon period is worth exactly 1/frequency of a year, whether it
    happens to hold 181 days or 184. That is what makes Treasury accrual exact."""
    ps, pe = date(2026, 5, 15), date(2026, 11, 15)     # 184 days
    assert year_fraction(ps, pe, ACT_ACT, period_start=ps, period_end=pe,
                         frequency=2) == pytest.approx(0.5, abs=1e-15)

    ps2, pe2 = date(2026, 11, 15), date(2027, 5, 15)   # 181 days
    assert (pe2 - ps2).days == 181
    assert year_fraction(ps2, pe2, ACT_ACT, period_start=ps2, period_end=pe2,
                         frequency=2) == pytest.approx(0.5, abs=1e-15)


def test_act_act_short_first_coupon():
    """A stub period accrues in proportion to the days elapsed."""
    ps, pe = date(2026, 5, 15), date(2026, 11, 15)
    settle = date(2026, 8, 6)                          # 83 of 184 days
    frac = year_fraction(ps, settle, ACT_ACT, period_start=ps, period_end=pe,
                         frequency=2)
    assert frac == pytest.approx(83 / (184 * 2), abs=1e-15)


def test_act_act_without_a_period_degrades_rather_than_lying():
    """A caller that forgets the enclosing period gets an ACT/365 approximation,
    not a silently wrong exact-looking number."""
    start, end = date(2026, 1, 1), date(2027, 1, 1)
    assert year_fraction(start, end, ACT_ACT) == pytest.approx(365 / 365)


def test_unknown_convention_raises():
    with pytest.raises(ValueError, match='Unknown day-count'):
        year_fraction(date(2026, 1, 1), date(2026, 2, 1), 'ACT/EVERY-OTHER-TUESDAY')


# ---------------------------------------------------------------------------
# day_count and accrual_fraction
# ---------------------------------------------------------------------------

def test_day_count_matches_the_convention():
    d1, d2 = date(2026, 1, 31), date(2026, 3, 31)
    assert day_count(d1, d2, D30_360) == 60         # adjusted
    assert day_count(d1, d2, ACT_360) == 59         # calendar
    assert day_count(d1, d2, ACT_ACT) == 59


def test_accrual_fraction_spans_zero_to_one():
    ps, pe = date(2026, 6, 15), date(2026, 12, 15)
    assert accrual_fraction(ps, ps, pe, D30_360) == pytest.approx(0.0)
    assert accrual_fraction(ps, pe, pe, D30_360) == pytest.approx(1.0)
    assert accrual_fraction(ps, date(2026, 9, 15), pe, D30_360) == pytest.approx(0.5)


def test_accrual_fraction_is_clamped():
    """A settlement outside the period is a data error upstream; clamping keeps
    it from producing a negative accrual or one above a full coupon."""
    ps, pe = date(2026, 6, 15), date(2026, 12, 15)
    assert accrual_fraction(ps, date(2026, 1, 1), pe, D30_360) == 0.0
    assert accrual_fraction(ps, date(2027, 6, 1), pe, D30_360) == 1.0


def test_accrual_fraction_on_a_degenerate_period():
    d = date(2026, 6, 15)
    assert accrual_fraction(d, d, d, D30_360) == 0.0
