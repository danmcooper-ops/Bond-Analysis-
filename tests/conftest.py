"""Shared fixtures. Every test in this suite runs offline."""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_par_curve():
    """A flat 5% par curve. Flat is the case with closed-form answers:
    zeros equal pars, forwards equal both, and roll-down is exactly zero."""
    return {'3M': 0.05, '6M': 0.05, '1Y': 0.05, '2Y': 0.05, '3Y': 0.05,
            '5Y': 0.05, '7Y': 0.05, '10Y': 0.05, '20Y': 0.05, '30Y': 0.05}


@pytest.fixture
def sample_par_curve():
    """An upward-sloping par curve, shaped like a normal Treasury curve.
    Used wherever the test needs zeros to sit above pars at the long end."""
    return {'1M': 0.0425, '3M': 0.0430, '6M': 0.0435, '1Y': 0.0440,
            '2Y': 0.0450, '3Y': 0.0460, '5Y': 0.0480, '7Y': 0.0495,
            '10Y': 0.0510, '20Y': 0.0540, '30Y': 0.0550}


# ---------------------------------------------------------------------------
# Bonds
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bond():
    """5% semiannual 30/360 corporate maturing 2035-06-15."""
    return dict(cusip='000000AA1', issuer_name='ACME CORP',
                coupon_rate=0.05, maturity=date(2035, 6, 15),
                frequency=2, convention='30/360', comp='semiannual',
                face=100.0, asset_class='CORP_IG')


@pytest.fixture
def sample_treasury():
    """4.25% semiannual ACT/ACT Treasury note maturing 2034-11-15."""
    return dict(cusip='91282CLW9', issuer_name='UNITED STATES TREASURY',
                coupon_rate=0.0425, maturity=date(2034, 11, 15),
                frequency=2, convention='ACT/ACT', comp='semiannual',
                face=100.0, asset_class='TREASURY')


# ---------------------------------------------------------------------------
# N-PORT
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_nport_rows():
    """Five funds holding one CUSIP, seeded with the failure modes the
    consensus builder has to survive: a 10x fat-finger outlier, a non-USD
    row, a zero-balance row, and a fund disagreeing on the maturity."""
    base = dict(cusip='000000AA1', issuer_name='ACME CORP',
                title_of_issue='ACME CORP SR NOTE 5.000% 06/15/35',
                maturity_date='2035-06-15', annualized_rate=5.0,
                coupon_type='Fixed', currency='USD', asset_cat='DBT',
                issuer_cat='CORP', fair_value_level=2,
                is_default=False, in_arrears=False, is_paid_kind=False,
                is_convertible=False, payoff_profile='Long')
    return [
        {**base, 'accession': 'a1', 'balance': 1_000_000, 'value_usd': 985_000},
        {**base, 'accession': 'a2', 'balance': 2_000_000, 'value_usd': 1_972_000},
        {**base, 'accession': 'a3', 'balance': 500_000, 'value_usd': 492_000},
        # 10x fat finger: implied price 9,850 rather than 98.50.
        {**base, 'accession': 'a4', 'balance': 100_000, 'value_usd': 9_850_000},
        # Non-USD — must be dropped, not FX-converted on a guess.
        {**base, 'accession': 'a5', 'currency': 'EUR',
         'balance': 1_000_000, 'value_usd': 1_070_000},
        # Zero balance — implied price undefined.
        {**base, 'accession': 'a6', 'balance': 0, 'value_usd': 0},
        # Disagrees on maturity -> _identity_conflict.
        {**base, 'accession': 'a7', 'maturity_date': '2035-06-30',
         'balance': 800_000, 'value_usd': 788_000, 'fair_value_level': 3},
    ]


# ---------------------------------------------------------------------------
# Issuer fundamentals
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_issuer_fundamentals():
    """Shaped exactly like a row the equity model already computes."""
    return dict(cik='0000000001', ticker='ACME', sector='Industrials',
                int_cov=8.4, nd_ebitda=2.1, total_debt=4.2e9, net_debt=3.5e9,
                fcf=9.1e8, altman_z=3.4, altman_z_zone='safe', piotroski=7,
                revenue=1.4e10, debt_maturity_wall_yrs=4.5,
                cet1_ratio=None, npl_ratio=None,
                _fundamentals_asof='2026-06-30')
