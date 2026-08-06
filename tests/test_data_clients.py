"""Data-client parsing and lookup logic. Entirely offline.

Feed samples below are trimmed copies of real responses captured on
2026-08-06, so a shape change upstream shows up here as a failing parse rather
than as an empty curve in production.
"""

from datetime import date

import pytest

from data.fred_client import FREDClient, term_factor_at
from data.treasury_curve_client import (SANE_MAX, SANE_MIN, _parse_csv,
                                        _parse_xml)
from data.treasury_direct_client import TreasuryDirectClient

# --- captured feed samples -------------------------------------------------

TREASURY_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
<content type="application/xml"><m:properties>
<d:Id m:type="Edm.Int32">288</d:Id>
<d:NEW_DATE m:type="Edm.DateTime">2026-08-05T00:00:00</d:NEW_DATE>
<d:BC_1MONTH m:type="Edm.Double">3.77</d:BC_1MONTH>
<d:BC_1_5MONTH m:type="Edm.Double">3.78</d:BC_1_5MONTH>
<d:BC_2MONTH m:type="Edm.Double">3.82</d:BC_2MONTH>
<d:BC_3MONTH m:type="Edm.Double">3.89</d:BC_3MONTH>
<d:BC_4MONTH m:type="Edm.Double">3.91</d:BC_4MONTH>
<d:BC_6MONTH m:type="Edm.Double">3.98</d:BC_6MONTH>
<d:BC_1YEAR m:type="Edm.Double">4.03</d:BC_1YEAR>
<d:BC_2YEAR m:type="Edm.Double">4.18</d:BC_2YEAR>
<d:BC_3YEAR m:type="Edm.Double">4.24</d:BC_3YEAR>
<d:BC_5YEAR m:type="Edm.Double">4.33</d:BC_5YEAR>
<d:BC_7YEAR m:type="Edm.Double">4.47</d:BC_7YEAR>
<d:BC_10YEAR m:type="Edm.Double">4.63</d:BC_10YEAR>
<d:BC_20YEAR m:type="Edm.Double">5.18</d:BC_20YEAR>
<d:BC_30YEAR m:type="Edm.Double">5.17</d:BC_30YEAR>
<d:BC_30YEARDISPLAY m:type="Edm.Double">5.17</d:BC_30YEARDISPLAY>
</m:properties></content>
</entry>
<entry>
<content type="application/xml"><m:properties>
<d:NEW_DATE m:type="Edm.DateTime">2026-08-06T00:00:00</d:NEW_DATE>
<d:BC_1MONTH m:type="Edm.Double">3.80</d:BC_1MONTH>
<d:BC_10YEAR m:type="Edm.Double">4.69</d:BC_10YEAR>
<d:BC_20YEAR m:type="Edm.Double"></d:BC_20YEAR>
<d:BC_30YEAR m:type="Edm.Double">5.22</d:BC_30YEAR>
</m:properties></content>
</entry>
</feed>"""

TREASURY_CSV = ("Date,1 Mo,3 Mo,6 Mo,1 Yr,2 Yr,10 Yr,30 Yr\n"
                "08/05/2026,3.77,3.89,3.98,4.03,4.18,4.63,5.17\n"
                "08/04/2026,3.75,3.88,3.97,4.02,4.20,4.63,5.16\n")

TREASURY_DIRECT_RECORDS = [
    {'cusip': '91282CLW9', 'securityType': 'Note', 'securityTerm': '10-Year',
     'interestRate': '4.250000', 'maturityDate': '2034-11-15T00:00:00',
     'issueDate': '2024-11-15T00:00:00', 'datedDate': '2024-11-15T00:00:00',
     'interestPaymentFrequency': 'Semi-Annual'},
    {'cusip': '912796YZ1', 'securityType': 'Bill', 'securityTerm': '26-Week',
     'interestRate': '', 'maturityDate': '2027-02-04T00:00:00',
     'issueDate': '2026-08-06T00:00:00', 'datedDate': '2026-08-06T00:00:00',
     'interestPaymentFrequency': 'None'},
    {'cusip': '912810TT8', 'securityType': 'TIPS', 'securityTerm': '30-Year',
     'interestRate': '2.125000', 'maturityDate': '2055-02-15T00:00:00',
     'issueDate': '2025-02-28T00:00:00', 'datedDate': '2025-02-15T00:00:00',
     'interestPaymentFrequency': 'Semi-Annual'},
    {'cusip': '91282CAB1', 'securityType': 'Note', 'securityTerm': '2-Year',
     'interestRate': '3.500000', 'maturityDate': '2020-01-31T00:00:00',
     'issueDate': '2018-01-31T00:00:00', 'datedDate': '2018-01-31T00:00:00',
     'interestPaymentFrequency': 'Semi-Annual'},
]


# ---------------------------------------------------------------------------
# Treasury par curve XML
# ---------------------------------------------------------------------------

def test_xml_parses_all_fourteen_tenors():
    curves = _parse_xml(TREASURY_XML)
    day = curves[date(2026, 8, 5)]
    assert len(day) == 14
    assert day['10Y'] == pytest.approx(0.0463)
    assert day['1.5M'] == pytest.approx(0.0378)
    assert day['30Y'] == pytest.approx(0.0517)


def test_xml_percentages_become_decimals():
    """The feed publishes 4.63 for 4.63%; every consumer works in decimals."""
    assert _parse_xml(TREASURY_XML)[date(2026, 8, 5)]['10Y'] == pytest.approx(0.0463)


def test_display_duplicate_of_the_thirty_year_is_excluded():
    """BC_30YEARDISPLAY is a formatting copy of BC_30YEAR. Including it would
    double-weight the long end of the bootstrap."""
    day = _parse_xml(TREASURY_XML)[date(2026, 8, 5)]
    assert '30Y' in day
    assert not any('DISPLAY' in k.upper() for k in day)
    assert sum(1 for v in day.values() if v == pytest.approx(0.0517)) == 1


def test_empty_tenor_is_skipped_not_zeroed():
    """A blank tenor means 'not quoted today'. Reading it as 0% would put a
    fabricated point in the curve and wreck the bootstrap."""
    day = _parse_xml(TREASURY_XML)[date(2026, 8, 6)]
    assert '20Y' not in day
    assert day['30Y'] == pytest.approx(0.0522)


def test_out_of_band_yields_are_dropped():
    bad = TREASURY_XML.replace('<d:BC_10YEAR m:type="Edm.Double">4.63',
                               '<d:BC_10YEAR m:type="Edm.Double">463.0')
    day = _parse_xml(bad)[date(2026, 8, 5)]
    assert '10Y' not in day
    assert '30Y' in day             # the rest of the curve survives
    assert SANE_MIN < day['30Y'] < SANE_MAX


def test_malformed_feed_returns_empty_rather_than_raising():
    assert _parse_xml('') == {}
    assert _parse_xml('<html>404 not found</html>') == {}
    assert _parse_xml('<feed><entry>truncated') == {}


def test_csv_fallback_parses_the_same_shape():
    curves = _parse_csv(TREASURY_CSV)
    assert set(curves) == {date(2026, 8, 5), date(2026, 8, 4)}
    assert curves[date(2026, 8, 5)]['10Y'] == pytest.approx(0.0463)
    assert curves[date(2026, 8, 4)]['2Y'] == pytest.approx(0.0420)


def test_xml_and_csv_agree_on_the_same_day():
    """The fallback must not quietly produce a different curve."""
    from_xml = _parse_xml(TREASURY_XML)[date(2026, 8, 5)]
    from_csv = _parse_csv(TREASURY_CSV)[date(2026, 8, 5)]
    for tenor in set(from_xml) & set(from_csv):
        assert from_xml[tenor] == pytest.approx(from_csv[tenor])


# ---------------------------------------------------------------------------
# FRED lookup semantics
# ---------------------------------------------------------------------------

def test_as_of_value_falls_back_to_the_most_recent_earlier_observation():
    """FRED skips weekends and holidays and publishes with a lag, so an
    exact-date lookup would fail on most days."""
    obs = {date(2026, 8, 3): 0.046, date(2026, 8, 4): 0.0463,
           date(2026, 8, 5): 0.0463}
    d, v = FREDClient._as_of_value(obs, date(2026, 8, 8))
    assert d == date(2026, 8, 5) and v == pytest.approx(0.0463)


def test_as_of_value_never_looks_forward():
    """Using an observation published after the as-of date would be
    look-ahead bias, straight into the backtest."""
    obs = {date(2026, 8, 5): 0.0463, date(2026, 8, 6): 0.0469}
    d, v = FREDClient._as_of_value(obs, date(2026, 8, 5))
    assert d == date(2026, 8, 5) and v == pytest.approx(0.0463)


def test_as_of_value_respects_the_lookback_limit():
    obs = {date(2026, 1, 2): 0.04}
    assert FREDClient._as_of_value(obs, date(2026, 8, 5)) == (None, None)
    assert FREDClient._as_of_value({}, date(2026, 8, 5)) == (None, None)


# ---------------------------------------------------------------------------
# Spread term factors
# ---------------------------------------------------------------------------

# Captured 2026-08-06: IG index 78bp, slices 46/67/80/95/92/99 bp.
TERM_POINTS = [(2.0, 0.59), (4.0, 0.86), (6.0, 1.03), (8.5, 1.22),
               (12.5, 1.18), (20.0, 1.27)]


def test_term_factor_interpolates_between_slices():
    assert term_factor_at(TERM_POINTS, 2.0) == pytest.approx(0.59, abs=1e-9)
    mid = term_factor_at(TERM_POINTS, 5.0)
    assert 0.86 < mid < 1.03


def test_term_factor_extrapolates_flat():
    assert term_factor_at(TERM_POINTS, 30.0) == pytest.approx(1.27, abs=1e-9)
    assert term_factor_at(TERM_POINTS, 0.5) == pytest.approx(0.59, abs=1e-9)


def test_term_beta_damps_the_structure():
    """beta is a calibration knob: 0 flattens the term effect away entirely,
    1 uses the observed ratios as-is."""
    assert term_factor_at(TERM_POINTS, 20.0, beta=0.0) == pytest.approx(1.0)
    assert term_factor_at(TERM_POINTS, 20.0, beta=1.0) == pytest.approx(1.27)
    assert term_factor_at(TERM_POINTS, 20.0, beta=0.5) == pytest.approx(1.135)


def test_term_factor_without_data_is_neutral():
    assert term_factor_at([], 10.0) == 1.0


def test_short_paper_gets_a_tighter_fair_spread_than_long():
    """The whole point of the term structure: a 2y BBB and a 30y BBB must not
    be assigned the same fair spread by the whole-index OAS."""
    bbb = 0.0096
    assert bbb * term_factor_at(TERM_POINTS, 2.0) < bbb * term_factor_at(TERM_POINTS, 30.0)


# ---------------------------------------------------------------------------
# TreasuryDirect normalisation
# ---------------------------------------------------------------------------

def test_to_bond_rows_normalises_a_note():
    rows = TreasuryDirectClient.to_bond_rows(TREASURY_DIRECT_RECORDS,
                                             as_of=date(2026, 8, 6))
    note = next(r for r in rows if r['cusip'] == '91282CLW9')
    assert note['coupon_rate'] == pytest.approx(0.0425)
    assert note['maturity_date'] == date(2034, 11, 15)
    assert note['frequency'] == 2
    assert note['asset_class'] == 'TREASURY'
    assert note['dated_date'] == date(2024, 11, 15)


def test_bills_get_zero_coupon_and_zero_frequency():
    """A bill's interestRate field is blank, not '0'. Reading blank as missing
    and defaulting the frequency to semiannual would invent coupons."""
    rows = TreasuryDirectClient.to_bond_rows(TREASURY_DIRECT_RECORDS,
                                             as_of=date(2026, 8, 6))
    bill = next(r for r in rows if r['cusip'] == '912796YZ1')
    assert bill['coupon_rate'] == 0.0
    assert bill['frequency'] == 0
    assert bill['asset_class'] == 'TREASURY_BILL'


def test_matured_securities_are_dropped():
    rows = TreasuryDirectClient.to_bond_rows(TREASURY_DIRECT_RECORDS,
                                             as_of=date(2026, 8, 6))
    assert not any(r['cusip'] == '91282CAB1' for r in rows)


def test_tips_are_flagged_because_the_model_prices_them_nominally():
    """The model ignores the inflation accrual, which understates TIPS. They
    are flagged so the gate layer can mark them inapplicable rather than rate
    them wrongly."""
    rows = TreasuryDirectClient.to_bond_rows(TREASURY_DIRECT_RECORDS,
                                             as_of=date(2026, 8, 6))
    tips = next(r for r in rows if r['cusip'] == '912810TT8')
    assert tips['is_inflation_linked'] is True
    assert all(r['is_inflation_linked'] is False
               for r in rows if r['cusip'] != '912810TT8')


# --- the two type fields disagree, and the boolean fields are strings ------

# Real shape, captured 2026-08-06. securityType says 'Note'; the security is a
# TIPS. Every yes/no flag is a STRING.
TIPS_RECORD = {
    'cusip': '91282CJY8', 'securityType': 'Note', 'type': 'TIPS',
    'tips': 'Yes', 'floatingRate': 'No', 'callable': 'No', 'reopening': 'Yes',
    'securityTerm': '9-Year 8-Month', 'originalSecurityTerm': '10-Year',
    'interestRate': '1.750000', 'maturityDate': '2034-01-15T00:00:00',
    'issueDate': '2024-05-31T00:00:00',
    'originalIssueDate': '2024-01-31T00:00:00',
    'datedDate': '2024-01-15T00:00:00',
    'originalDatedDate': '2024-01-15T00:00:00',
    'interestPaymentFrequency': 'Semi-Annual',
    'indexRatioOnIssueDate': '1.015860', 'refCpiOnDatedDate': '307.391000',
    'cpiBaseReferencePeriod': '1982-1984=100',
}

NOMINAL_RECORD = {
    'cusip': '91282CLW9', 'securityType': 'Note', 'type': 'Note',
    'tips': 'No', 'floatingRate': 'No', 'callable': 'No', 'reopening': 'No',
    'securityTerm': '10-Year', 'interestRate': '4.250000',
    'maturityDate': '2034-11-15T00:00:00',
    'issueDate': '2024-11-15T00:00:00', 'datedDate': '2024-11-15T00:00:00',
    'interestPaymentFrequency': 'Semi-Annual',
}


def test_security_type_alone_would_misclassify_every_tips():
    """TreasuryDirect carries TWO type fields and they disagree: for a 10-year
    TIPS, securityType is 'Note' (the auction format) while type is 'TIPS'.
    Reading securityType alone classifies every TIPS as a nominal note."""
    from data.treasury_direct_client import instrument_type, is_inflation_linked
    assert TIPS_RECORD['securityType'] == 'Note'      # the trap
    assert instrument_type(TIPS_RECORD) == 'TIPS'
    assert is_inflation_linked(TIPS_RECORD) is True
    assert instrument_type(NOMINAL_RECORD) == 'Note'
    assert is_inflation_linked(NOMINAL_RECORD) is False


def test_yes_no_fields_are_strings_so_truthiness_is_always_true():
    """Every boolean in the feed is the literal string 'Yes' or 'No', and
    bool('No') is True. Testing truthiness marks every security as having
    every flag — which classified all 6 of 6 nominal notes as floaters."""
    from data.treasury_direct_client import _yes, is_floating_rate
    assert bool(NOMINAL_RECORD['floatingRate']) is True     # the trap
    assert _yes(NOMINAL_RECORD['floatingRate']) is False
    assert _yes(TIPS_RECORD['tips']) is True
    assert is_floating_rate(NOMINAL_RECORD) is False


def test_a_plain_note_survives_construction():
    """Regression guard on the bug above: a 4.25% nominal note must reach the
    analytics, with Treasury conventions."""
    from models.bond_types import from_row
    row = TreasuryDirectClient.to_bond_rows([NOMINAL_RECORD],
                                            as_of=date(2026, 8, 6))[0]
    assert row['coupon_type'] == 'Fixed'
    bond, reason = from_row(row, settle=date(2026, 8, 6))
    assert reason is None
    assert bond.convention == 'ACT/ACT'


def test_tips_are_rejected_with_an_accurate_reason():
    """Rejected, not silently priced: a 1.750% REAL coupon compared against a
    4.6% nominal curve would look like a wildly off-market bond."""
    from models.bond_types import from_row
    row = TreasuryDirectClient.to_bond_rows([TIPS_RECORD],
                                            as_of=date(2026, 8, 6))[0]
    assert row['coupon_type'] == 'Inflation-Linked'
    bond, reason = from_row(row, settle=date(2026, 8, 6))
    assert bond is None
    assert 'inflation-linked' in reason


def test_reopening_prefers_the_original_auction_fields():
    rows = TreasuryDirectClient.to_bond_rows([TIPS_RECORD],
                                             as_of=date(2026, 8, 6))
    assert rows[0]['security_term'] == '10-Year'          # not '9-Year 8-Month'
    assert rows[0]['issue_date'] == date(2024, 1, 31)     # not 2024-05-31


def test_reopenings_of_the_same_cusip_collapse_to_one_row():
    """TreasuryDirect returns one record per AUCTION, and Treasury reopens
    issues repeatedly — the same CUSIP came back three times for the 2034
    maturity year in the live feed. Without deduping, that security enters the
    universe three times and triple-weights every percentile pool it sits in.
    """
    reopenings = [
        {'cusip': '91282CLW9', 'securityType': 'Note',
         'securityTerm': '10-Year', 'interestRate': '4.250000',
         'maturityDate': '2034-11-15T00:00:00',
         'issueDate': '2024-11-15T00:00:00',
         'datedDate': '2024-11-15T00:00:00',
         'interestPaymentFrequency': 'Semi-Annual'},
        {'cusip': '91282CLW9', 'securityType': 'Note',
         'securityTerm': '9-Year 11-Month', 'interestRate': '4.250000',
         'maturityDate': '2034-11-15T00:00:00',
         'issueDate': '2024-12-16T00:00:00',
         'datedDate': '2024-11-15T00:00:00',
         'interestPaymentFrequency': 'Semi-Annual'},
        {'cusip': '91282CLW9', 'securityType': 'Note',
         'securityTerm': '9-Year 10-Month', 'interestRate': '4.250000',
         'maturityDate': '2034-11-15T00:00:00',
         'issueDate': '2025-01-15T00:00:00',
         'datedDate': '2024-11-15T00:00:00',
         'interestPaymentFrequency': 'Semi-Annual'},
    ]
    rows = TreasuryDirectClient.to_bond_rows(reopenings, as_of=date(2026, 8, 6))
    assert len(rows) == 1
    # The original auction wins; pricing fields are identical either way.
    assert rows[0]['security_term'] == '10-Year'
    assert rows[0]['dated_date'] == date(2024, 11, 15)


def test_dedupe_keeps_distinct_cusips():
    rows = TreasuryDirectClient.to_bond_rows(TREASURY_DIRECT_RECORDS,
                                             as_of=date(2026, 8, 6))
    cusips = [r['cusip'] for r in rows]
    assert len(cusips) == len(set(cusips))
    assert set(cusips) == {'91282CLW9', '912796YZ1', '912810TT8'}


def test_normalised_rows_feed_straight_into_bond_construction():
    """The Treasury path and the corporate path must converge on one schema."""
    from models.bond_types import from_row
    rows = TreasuryDirectClient.to_bond_rows(TREASURY_DIRECT_RECORDS,
                                             as_of=date(2026, 8, 6))
    note = next(r for r in rows if r['cusip'] == '91282CLW9')
    bond, reason = from_row(note, settle=date(2026, 8, 6))
    assert reason is None
    assert bond.convention == 'ACT/ACT'      # Treasury conventions applied
    assert bond.coupon_rate == pytest.approx(0.0425)
