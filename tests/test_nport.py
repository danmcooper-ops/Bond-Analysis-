"""N-PORT parsing and consensus marks. Offline.

Every fixture below reproduces a shape actually observed in the 2026Q2 data
set, including three encodings that are not documented anywhere and each of
which silently corrupted the pipeline before being found.
"""

from datetime import date

import pytest

from data.nport_client import (NO_CUSIP_SENTINELS, implied_price,
                               is_valid_cusip, parse_sec_date, parse_yn)
from data.nport_consensus import (COUPON_CONFLICT_TOLERANCE, MIN_OUTLIER_BAND,
                                  consensus_mark, latest_marks,
                                  normalise_coupon_units, reject_outliers)


def holding(**kw):
    row = dict(cusip='06051GHD4', report_date=date(2026, 4, 30),
               issuer_name='BANK OF AMERICA CORP',
               title_of_issue='BANK OF AMERICA CORP SR NOTE',
               implied_price=98.35, value_usd=7_297_417.0, balance=7_420_000.0,
               pct_of_nav=0.31, payoff_profile='Long', issuer_type='CORP',
               fair_value_level=2.0, maturity_date=date(2028, 12, 20),
               annualized_rate=3.42, coupon_type='Fixed', is_default=False,
               in_arrears=False, is_paid_kind=False, is_convertible=False)
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

def test_parse_sec_date():
    assert parse_sec_date('01-NOV-2042') == date(2042, 11, 1)
    assert parse_sec_date('28-FEB-2026') == date(2026, 2, 28)
    assert parse_sec_date('') is None
    assert parse_sec_date('2042-11-01') is None      # wrong format, not a guess
    assert parse_sec_date('31-XXX-2026') is None


def test_parse_yn():
    assert parse_yn('Y') is True
    assert parse_yn('N') is False
    assert parse_yn('') is False
    assert parse_yn(None) is False


# ---------------------------------------------------------------------------
# CUSIP validity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cusip', [
    '06051GHD4',    # Bank of America note
    '91282CLW9',    # US Treasury note
    '037833100',    # Apple common
    '92343VHC1',    # Verizon
])
def test_real_cusips_validate(cusip):
    assert is_valid_cusip(cusip)


@pytest.mark.parametrize('cusip', [
    '999999999',    # documented no-CUSIP sentinel
    '000000000',    # UNDOCUMENTED sentinel — 5,109 holdings in 2026Q2
    '06051GHD5',    # right shape, wrong check digit
    '06051GHD',     # truncated
    '', None, 'N/A',
])
def test_sentinels_and_malformed_cusips_are_rejected(cusip):
    assert not is_valid_cusip(cusip)


def test_zero_cusip_needs_the_sentinel_list_not_just_the_checksum():
    """'000000000' has a legitimate check digit of 0, so the checksum alone
    accepts it. It accumulated 5,109 holdings priced from 0 to 1052 — it would
    have become the single most widely held 'bond' in the universe."""
    assert '000000000' in NO_CUSIP_SENTINELS
    assert not is_valid_cusip('000000000')


# ---------------------------------------------------------------------------
# Implied price
# ---------------------------------------------------------------------------

def test_implied_price_for_usd():
    assert implied_price(7_420_000, 7_297_417) == pytest.approx(98.3479, abs=1e-4)


def test_non_usd_needs_the_exchange_rate():
    """CURRENCY_VALUE is in USD while BALANCE is local face, so the naive
    ratio is FX-contaminated. Verified on a real AUD holding: 974.14 face
    valued at 701.32 USD at 1.389 is a bond at exactly par, which the naive
    ratio reports as 71.99."""
    naive = 701.32 / 974.14 * 100
    assert naive == pytest.approx(71.99, abs=0.01)
    corrected = implied_price(974.14, 701.32, exchange_rate=1.389,
                              currency_code='AUD')
    assert corrected == pytest.approx(100.0, abs=0.05)


def test_non_usd_without_a_rate_refuses_rather_than_guesses():
    assert implied_price(1000, 900, exchange_rate=None, currency_code='EUR') is None
    assert implied_price(1000, 900, exchange_rate=0, currency_code='EUR') is None


def test_implied_price_guards_bad_balances():
    assert implied_price(0, 100) is None
    assert implied_price(-100, 100) is None
    assert implied_price(1000, None) is None


# ---------------------------------------------------------------------------
# Coupon units — the mixed-encoding bug
# ---------------------------------------------------------------------------

def test_coupon_units_are_mixed_in_the_source_and_get_reconciled():
    """ANNUALIZED_RATE is percent for some funds and a decimal fraction for
    others, in the same file for the same CUSIP: SBA Communications came back
    as [0.0312, 3.125]. Unreconciled, a 7.875% bond could enter the model with
    a 0.0788% coupon and every yield, duration and spread derived from it."""
    assert normalise_coupon_units([0.0312, 3.125]) == pytest.approx([3.12, 3.125])
    assert normalise_coupon_units([0.0788, 7.875, 7.88]) == pytest.approx(
        [7.88, 7.875, 7.88])


def test_wholly_decimal_group_is_rescaled():
    """A single-fund CUSIP has no percent-scale sibling to calibrate against.
    A genuine corporate coupon below half a percent is close to nonexistent;
    this encoding demonstrably is not."""
    assert normalise_coupon_units([0.05]) == pytest.approx([5.0])


def test_genuine_zero_coupon_is_left_alone():
    assert normalise_coupon_units([0.0, 0.0]) == [0.0, 0.0]


def test_percent_scale_groups_pass_through_untouched():
    assert normalise_coupon_units([4.125, 4.13]) == pytest.approx([4.125, 4.13])


def test_an_odd_decimal_is_not_forced_to_fit():
    """Rescaling only applies to decimals that land on the group's consensus;
    a value that does not match is left alone rather than bent into place."""
    out = normalise_coupon_units([5.0, 5.0, 0.019])
    assert 0.019 in out


# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------

def test_median_and_mad_survive_a_ten_times_fat_finger():
    """Standard deviation is computed from the very outlier it should catch —
    one 10x mark inflates it enough to bring itself inside three sigma."""
    prices = [98.5, 98.6, 98.55, 98.52, 985.0]
    kept, rejected = reject_outliers(prices)
    assert rejected == [985.0]
    assert len(kept) == 4


def test_only_the_outlier_is_removed():
    prices = [99.0, 99.1, 99.05, 99.2, 60.0]
    kept, rejected = reject_outliers(prices)
    assert rejected == [60.0]
    assert sorted(kept) == [99.0, 99.05, 99.1, 99.2]


def test_the_band_has_a_floor_so_tight_agreement_is_not_punished():
    """Fund marks agree far more tightly than a MAD test expects — observed
    median dispersion is 0.002%. Without a floor the band collapses to a
    fraction of a cent and rejects ordinary pricing-service noise, deflating
    n_funds and tripping the thin-coverage rating cap on a bond held by
    forty funds."""
    prices = [99.625] * 40 + [99.66, 99.59]
    kept, rejected = reject_outliers(prices)
    assert rejected == []
    assert len(kept) == 42


def test_floor_is_relative_for_deep_discount_paper():
    """Half a point is a large move on a bond trading at 30."""
    prices = [30.0] * 10 + [30.1]
    _, rejected = reject_outliers(prices)
    assert rejected == []


def test_too_few_marks_to_test():
    assert reject_outliers([98.0, 120.0]) == ([98.0, 120.0], [])


def test_rejection_never_empties_the_set():
    kept, rejected = reject_outliers([10.0, 200.0, 50.0])
    assert kept


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

def test_consensus_takes_the_median_across_funds():
    rows = [holding(implied_price=p) for p in (98.30, 98.35, 98.40)]
    out = consensus_mark(rows)
    assert len(out) == 1
    assert out[0]['clean_price_marked'] == pytest.approx(98.35)
    assert out[0]['n_funds'] == 3


def test_consensus_groups_by_cusip_and_month():
    rows = [holding(report_date=date(2026, 3, 31), implied_price=97.0),
            holding(report_date=date(2026, 4, 30), implied_price=98.0)]
    out = sorted(consensus_mark(rows), key=lambda r: r['report_date'])
    assert len(out) == 2
    assert out[0]['clean_price_marked'] == pytest.approx(97.0)
    assert out[1]['clean_price_marked'] == pytest.approx(98.0)


def test_fair_value_level_takes_the_worst_across_funds():
    """One fund calling it level 3 is the signal, not the outvoted opinion."""
    rows = [holding(fair_value_level=2.0), holding(fair_value_level=2.0),
            holding(fair_value_level=3.0)]
    assert consensus_mark(rows)[0]['fair_value_level'] == 3.0


def test_trouble_flags_are_any_not_majority():
    """A default is an assertion about the issuer, not an opinion about price:
    one fund reporting it is not outvoted by nine that have not updated."""
    rows = [holding(), holding(), holding(is_default=True)]
    out = consensus_mark(rows)[0]
    assert out['is_default'] is True

    rows = [holding(), holding(in_arrears=True)]
    assert consensus_mark(rows)[0]['in_arrears'] is True


def test_price_dispersion_is_relative_so_it_compares_across_price_levels():
    rows = [holding(implied_price=p) for p in (98.0, 99.0, 100.0, 101.0, 102.0)]
    out = consensus_mark(rows)[0]
    assert out['price_dispersion'] == pytest.approx(1.0 / 100.0, abs=1e-6)


def test_implausible_prices_are_dropped_before_the_median():
    rows = [holding(implied_price=p) for p in (98.0, 98.5, 1e7, 0.0)]
    out = consensus_mark(rows)[0]
    assert out['n_funds'] == 2
    assert out['clean_price_marked'] == pytest.approx(98.25)


def test_coupon_rounding_is_not_a_term_conflict():
    """4.125 and 4.13 are the same bond reported to different precision.
    Treating that as disagreement flagged a quarter of all CUSIPs."""
    rows = [holding(annualized_rate=4.125), holding(annualized_rate=4.13)]
    assert consensus_mark(rows)[0]['_identity_conflict'] is False


def test_a_real_coupon_disagreement_is_flagged():
    rows = [holding(annualized_rate=4.125), holding(annualized_rate=5.5)]
    assert consensus_mark(rows)[0]['_identity_conflict'] is True


def test_maturity_disagreement_is_flagged():
    rows = [holding(maturity_date=date(2028, 12, 20)),
            holding(maturity_date=date(2029, 6, 15))]
    assert consensus_mark(rows)[0]['_identity_conflict'] is True


def test_mixed_unit_coupons_do_not_read_as_a_conflict():
    rows = [holding(annualized_rate=0.0342), holding(annualized_rate=3.42)]
    out = consensus_mark(rows)[0]
    assert out['_identity_conflict'] is False
    assert out['annualized_rate'] == pytest.approx(3.42)


def test_total_held_sums_across_funds():
    rows = [holding(value_usd=1e6), holding(value_usd=2e6), holding(value_usd=3e6)]
    assert consensus_mark(rows)[0]['total_held_usd'] == pytest.approx(6e6)


def test_marks_carry_their_basis():
    """Validated empirically against 1,155 Treasury observations: the residual
    against curve-implied clean prices regresses on accrued interest with a
    slope of -0.016, so the marks exclude accrued interest."""
    assert consensus_mark([holding()])[0]['price_basis'] == 'nport_implied_clean'


def test_latest_marks_keeps_the_most_recent_month():
    marks = consensus_mark([holding(report_date=date(2026, 3, 31)),
                            holding(report_date=date(2026, 4, 30))])
    latest = latest_marks(marks)
    assert len(latest) == 1
    assert latest[0]['report_date'] == date(2026, 4, 30)


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------

def test_wide_credit_term_curves_invert_where_tight_ones_rise():
    """The measured shapes are not one curve. Tight and mid credits widen with
    maturity; WIDE ones narrow, because a struggling issuer's problem is
    refinancing the next maturity rather than the one in twenty years.
    A single rising curve gets high yield backwards by roughly 47%."""
    from data.fred_client import term_factor_at
    tight = [(1.5, 0.86), (4.0, 1.00), (8.5, 1.08), (22.5, 1.12), (38.0, 1.07)]
    wide = [(1.5, 0.95), (4.0, 1.00), (8.5, 0.87), (22.5, 0.81), (38.0, 0.84)]
    assert term_factor_at(tight, 25) > term_factor_at(tight, 5)
    assert term_factor_at(wide, 25) < term_factor_at(wide, 5)


def test_fair_spread_prefers_the_bucket_specific_curve():
    from models.credit import fair_spread
    oas = {'AAA': 0.0038, 'B': 0.029}
    shared = [(4.0, 1.0), (30.0, 1.2)]
    by_bucket = {'B': [(4.0, 1.0), (30.0, 0.84)]}
    # B rides its own inverting curve; AAA has none and falls back to shared.
    assert fair_spread('B', 30, oas, term_points=shared,
                       term_by_bucket=by_bucket) < fair_spread(
        'B', 30, oas, term_points=shared)
    assert fair_spread('AAA', 30, oas, term_points=shared,
                       term_by_bucket=by_bucket) == pytest.approx(
        fair_spread('AAA', 30, oas, term_points=shared))


def test_fair_spread_without_any_term_data_is_the_flat_index():
    from models.credit import fair_spread
    assert fair_spread('BBB', 10, {'BBB': 0.0096}) == pytest.approx(0.0096)
