"""Bond construction from raw rows, conventions, and seniority inference.

`from_row` is the boundary between messy feed data and the analytics. Its job
is to reject rather than guess: a row it cannot price must not enter the
pipeline with a fabricated coupon and go on to produce a confident wrong
spread. Every rejection carries a reason so a run can report why rows dropped.
"""

from datetime import date

import pytest

from models.bond_types import (SENIORITY_JUNIOR, SENIORITY_SENIOR_SECURED,
                               SENIORITY_SENIOR_SUB, SENIORITY_SENIOR_UNSECURED,
                               SENIORITY_SUB, Bond, from_row, infer_seniority)
from models.conventions import (classify_by_cusip, conventions_for,
                                is_analyzable)
from models.daycount import ACT_360, ACT_ACT, D30_360

SETTLE = date(2026, 8, 6)


def _row(**kw):
    row = dict(cusip='000000AA1', issuer_name='ACME CORP',
               maturity_date='2035-06-15', annualized_rate=5.0,
               coupon_type='Fixed')
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_builds_a_bond_from_a_well_formed_row():
    bond, reason = from_row(_row(), settle=SETTLE)
    assert reason is None
    assert bond.cusip == '000000AA1'
    assert bond.coupon_rate == pytest.approx(0.05)
    assert bond.maturity == date(2035, 6, 15)
    assert bond.frequency == 2
    assert bond.convention == D30_360


def test_percentage_coupons_are_normalised_to_decimals():
    """N-PORT reports 5.0 for a 5% coupon; other sources use 0.05. A feed that
    switches units silently is far more likely than a real 100%+ coupon."""
    assert from_row(_row(annualized_rate=5.0), SETTLE)[0].coupon_rate == pytest.approx(0.05)
    assert from_row(_row(annualized_rate=0.05), SETTLE)[0].coupon_rate == pytest.approx(0.05)
    assert from_row(_row(annualized_rate=12.5), SETTLE)[0].coupon_rate == pytest.approx(0.125)


def test_explicit_coupon_rate_wins_over_annualized_rate():
    bond, _ = from_row(_row(coupon_rate=0.0375, annualized_rate=9.9), SETTLE)
    assert bond.coupon_rate == pytest.approx(0.0375)


@pytest.mark.parametrize('value,expected', [
    ('2035-06-15', date(2035, 6, 15)),
    ('06/15/2035', date(2035, 6, 15)),
    (date(2035, 6, 15), date(2035, 6, 15)),
    ('2035-06-15T00:00:00', date(2035, 6, 15)),
])
def test_maturity_date_parsing(value, expected):
    bond, reason = from_row(_row(maturity_date=value), SETTLE)
    assert reason is None
    assert bond.maturity == expected


# ---------------------------------------------------------------------------
# Rejection, with reasons
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('overrides,fragment', [
    ({'cusip': ''}, 'cusip'),
    ({'maturity_date': None}, 'maturity'),
    ({'maturity_date': 'not-a-date'}, 'maturity'),
    ({'maturity_date': '2020-01-01'}, 'matured'),
    ({'annualized_rate': None}, 'coupon'),
    ({'annualized_rate': 'n/a'}, 'coupon'),
    ({'annualized_rate': 95.0}, 'coupon'),           # 95% coupon: bad data
    ({'coupon_type': 'Floating'}, 'fixed-rate'),
    ({'is_convertible': True}, 'fixed-rate'),
])
def test_bad_rows_are_rejected_with_a_reason(overrides, fragment):
    bond, reason = from_row(_row(**overrides), settle=SETTLE)
    assert bond is None
    assert fragment in reason


def test_floaters_are_rejected_rather_than_mispriced():
    """A floater's YTM is undefined without projecting a forward index.
    Feeding it through the fixed-rate machinery would produce a
    confident-looking wrong number."""
    bond, reason = from_row(_row(coupon_type='Floating'), SETTLE)
    assert bond is None and 'floater' in reason


# ---------------------------------------------------------------------------
# Asset-class inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cusip,expected', [
    ('912828XY5', 'TREASURY'),
    ('91282CLW9', 'TREASURY'),
    ('912810QT8', 'TREASURY'),
    ('912796YZ1', 'TREASURY_BILL'),
    ('3135G0X24', 'AGENCY'),
    ('3137EAEZ8', 'AGENCY'),
    ('037833DK1', None),          # Apple: a corporate prefix says nothing
    ('', None),
    ('123', None),
])
def test_classify_by_cusip(cusip, expected):
    assert classify_by_cusip(cusip) == expected


def test_treasury_row_gets_treasury_conventions():
    bond, _ = from_row(_row(cusip='91282CLW9', annualized_rate=4.25), SETTLE)
    assert bond.asset_class == 'TREASURY'
    assert bond.convention == ACT_ACT


def test_explicit_asset_class_wins_over_cusip_inference():
    bond, _ = from_row(_row(cusip='912828XY5', asset_class='CORP_HY'), SETTLE)
    assert bond.asset_class == 'CORP_HY'


def test_unrecognised_cusip_defaults_to_investment_grade_corporate():
    bond, _ = from_row(_row(cusip='037833DK1'), SETTLE)
    assert bond.asset_class == 'CORP_IG'


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

def test_conventions_table():
    assert conventions_for('TREASURY')['convention'] == ACT_ACT
    assert conventions_for('CORP_IG')['convention'] == D30_360
    assert conventions_for('TREASURY_BILL')['frequency'] == 0
    assert conventions_for('TREASURY_BILL')['convention'] == ACT_360
    assert conventions_for('SOMETHING_NEW')['convention'] == D30_360


def test_explicit_frequency_overrides_the_table():
    """Issuers do pay annually and quarterly; the source data wins."""
    assert conventions_for('CORP_IG', frequency=4)['frequency'] == 4
    assert conventions_for('CORP_IG', frequency=1)['frequency'] == 1


def test_zero_coupon_forces_a_discount_instrument():
    conv = conventions_for('CORP_IG', coupon_rate=0.0)
    assert conv['frequency'] == 0
    assert conv['comp'] == 'simple'


def test_is_analyzable():
    assert is_analyzable(coupon_type='Fixed') is True
    assert is_analyzable(coupon_type=None) is True
    assert is_analyzable(coupon_type='Floating') is False
    assert is_analyzable(coupon_type='Variable') is False
    assert is_analyzable(is_convertible=True) is False
    assert is_analyzable(coupon_rate=0.95) is False
    assert is_analyzable(coupon_rate=-0.01) is False


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('title,rank', [
    ('ACME CORP SR NOTE 5.000% 06/15/35', SENIORITY_SENIOR_UNSECURED),
    ('ACME CORP SENIOR NOTES', SENIORITY_SENIOR_UNSECURED),
    ('ACME CORP SR SECURED NOTE', SENIORITY_SENIOR_SECURED),
    ('ACME 1ST LIEN TERM LOAN', SENIORITY_SENIOR_SECURED),
    ('ACME CORP 2ND LIEN NOTES', SENIORITY_SENIOR_SECURED),
    ('ACME CORP SENIOR SUBORDINATED NOTES', SENIORITY_SENIOR_SUB),
    ('ACME CORP SUBORDINATED DEB', SENIORITY_SUB),
    ('ACME CAP TRUST JR SUBORDINATED', SENIORITY_JUNIOR),
    ('ACME CORP PFD SERIES A', SENIORITY_JUNIOR),
])
def test_infer_seniority_from_title(title, rank):
    assert infer_seniority(title)[0] == rank


def test_seniority_marks_when_it_guessed():
    """A guessed seniority must never silently drive a rating — the Structure
    gate and the report both key off this marker."""
    assert infer_seniority('ACME CORP SR NOTE')[1] == 'title'
    assert infer_seniority('ACME CORP 5% 2035')[1] == 'default'
    assert infer_seniority('')[1] == 'default'
    assert infer_seniority(None) == (SENIORITY_SENIOR_UNSECURED, 'default')


def test_more_specific_seniority_patterns_win():
    """'SR SECURED' must beat the bare 'SR', and 'SENIOR SUBORDINATED' must
    beat both 'SENIOR' and 'SUBORDINATED'."""
    assert infer_seniority('ACME SR SECURED NOTES')[0] == SENIORITY_SENIOR_SECURED
    assert infer_seniority('ACME SENIOR SUBORDINATED NOTES')[0] == SENIORITY_SENIOR_SUB


def test_bond_years_to_maturity():
    bond = Bond(cusip='X', issuer_name='Y', coupon_rate=0.05,
                maturity=date(2036, 8, 6), frequency=2, convention=D30_360,
                comp='semiannual', asset_class='CORP_IG')
    assert bond.years_to_maturity(date(2026, 8, 6)) == pytest.approx(10.0, abs=0.01)
