"""SEC XBRL derivation: point-in-time, tag fallbacks, TTM roll-forward. Offline."""

from datetime import date

import pytest

from data.sec_xbrl_client import altman_zone, derive, instant_at, trim, ttm_at


def _f(start, end, val, filed, form='10-K'):
    return {'start': start, 'end': end, 'val': val, 'filed': filed, 'form': form}


def _payload(us_gaap, dei=None, name='Acme Corp'):
    return {'entityName': name,
            'facts': {'us-gaap': {tag: {'USD': rows} for tag, rows in us_gaap.items()},
                      'dei': dei or {}}}


FY25 = ('2025-01-01', '2025-12-31')


def test_a_fact_filed_after_as_of_is_invisible():
    facts = _payload({'LongTermDebt': [
        _f(None, '2025-12-31', 900.0, '2026-02-20'),
        _f(None, '2025-09-30', 800.0, '2025-11-01', '10-Q')]})['facts']
    assert instant_at(facts, ('LongTermDebt',), date(2026, 1, 15)) == \
        (800.0, date(2025, 9, 30))
    assert instant_at(facts, ('LongTermDebt',), date(2026, 3, 1))[0] == 900.0


def test_tag_fallback_uses_the_first_tag_present():
    facts = _payload({'RevenueFromContractWithCustomerExcludingAssessedTax': [
        _f(*FY25, 5000.0, '2026-02-20')]})['facts']
    from data.sec_xbrl_client import FLOW_TAGS
    assert ttm_at(facts, FLOW_TAGS['revenue'], date(2026, 3, 1))[0] == 5000.0


def test_ttm_rolls_the_annual_forward_by_year_to_date():
    facts = _payload({'Revenues': [
        _f(*FY25, 1000.0, '2026-02-20'),
        _f('2025-01-01', '2025-06-30', 450.0, '2025-08-01', '10-Q'),
        _f('2026-01-01', '2026-06-30', 550.0, '2026-08-01', '10-Q'),
    ]})['facts']
    value, end = ttm_at(facts, ('Revenues',), date(2026, 9, 1))
    assert value == pytest.approx(1100.0) and end == date(2026, 6, 30)
    # Before the 10-Q was filed, only the annual is known.
    assert ttm_at(facts, ('Revenues',), date(2026, 7, 1))[0] == 1000.0


def test_missing_interest_expense_omits_coverage():
    payload = _payload({'OperatingIncomeLoss': [_f(*FY25, 300.0, '2026-02-20')],
                        'LongTermDebt': [_f(None, '2025-12-31', 900.0, '2026-02-20')]})
    out = derive(payload, date(2026, 3, 1))
    assert 'int_cov' not in out
    assert out['total_debt'] == 900.0
    assert out['_fundamentals_source'] == 'sec_xbrl'
    assert out['_fundamentals_asof'] == '2025-12-31'


def test_full_derivation_matches_the_equity_conventions():
    filed = '2026-02-20'
    payload = _payload({
        'Revenues': [_f(*FY25, 10_000.0, filed)],
        'OperatingIncomeLoss': [_f(*FY25, 1_500.0, filed)],
        'InterestExpense': [_f(*FY25, 300.0, filed)],
        'DepreciationDepletionAndAmortization': [_f(*FY25, 500.0, filed)],
        'NetCashProvidedByUsedInOperatingActivities': [_f(*FY25, 1_800.0, filed)],
        'PaymentsToAcquirePropertyPlantAndEquipment': [_f(*FY25, 600.0, filed)],
        'LongTermDebt': [_f(None, '2025-12-31', 4_000.0, filed)],
        'CashAndCashEquivalentsAtCarryingValue': [_f(None, '2025-12-31', 1_000.0, filed)],
        'StockholdersEquity': [_f(None, '2025-12-31', 5_000.0, filed)],
        'Assets': [_f(None, '2025-12-31', 12_000.0, filed)],
        'Liabilities': [_f(None, '2025-12-31', 7_000.0, filed)],
        'AssetsCurrent': [_f(None, '2025-12-31', 3_000.0, filed)],
        'LiabilitiesCurrent': [_f(None, '2025-12-31', 2_000.0, filed)],
        'RetainedEarningsAccumulatedDeficit': [_f(None, '2025-12-31', 2_500.0, filed)],
    }, dei={'EntityCommonStockSharesOutstanding': {'shares': [
        {'end': '2026-02-01', 'val': 100.0, 'filed': filed}]}})
    out = derive(payload, date(2026, 3, 1), price=50.0)
    assert out['int_cov'] == pytest.approx(5.0)
    assert out['nd_ebitda'] == pytest.approx(3000.0 / 2000.0)
    assert out['fcf'] == pytest.approx(1200.0)
    assert out['de'] == pytest.approx(0.8)
    assert out['mcap'] == pytest.approx(5000.0)
    z = (1.2 * 1000 / 12000 + 1.4 * 2500 / 12000 + 3.3 * 1500 / 12000
         + 0.6 * 5000 / 7000 + 10000 / 12000)
    assert out['altman_z'] == pytest.approx(z)
    assert out['altman_z_zone'] == altman_zone(z)
    assert 'sector' not in out
    # No price: no mcap, and no market-value Altman term either.
    bare = derive(payload, date(2026, 3, 1))
    assert 'mcap' not in bare and 'altman_z' not in bare


def test_banks_are_routed_to_the_financial_scorecard():
    payload = _payload({'Deposits': [_f(None, '2025-12-31', 1e9, '2026-02-20')]})
    assert derive(payload, date(2026, 3, 1))['sector'] == 'Financial Services'


def test_trim_keeps_only_used_tags():
    raw = {'entityName': 'X', 'facts': {'us-gaap': {
        'Revenues': {'label': 'l', 'units': {'USD': [
            {'start': '2025-01-01', 'end': '2025-12-31', 'val': 1, 'filed': '2026-02-01',
             'form': '10-K', 'accn': 'drop-me', 'frame': 'CY2025'}]}},
        'SomethingUnused': {'units': {'USD': [{'val': 1}]}}}}}
    out = trim(raw)
    assert list(out['facts']['us-gaap']) == ['Revenues']
    assert 'accn' not in out['facts']['us-gaap']['Revenues']['USD'][0]


def test_backend_reports_only_derived_fields_and_skips_unknown_ciks():
    from data.issuer_fundamentals import SECXBRLBackend

    class Client:
        def companyfacts(self, cik):
            return _payload({'LongTermDebt': [
                _f(None, '2025-12-31', 900.0, '2026-02-20')]}) if cik == 1 else None

    backend = SECXBRLBackend(tickers=['ACME', 'NOPE', 'GONE'],
                             cik_map={'ACME': 1, 'GONE': 2},
                             as_of=date(2026, 3, 1), client=Client(),
                             price_fn=lambda t, d: None)
    rows = backend.load()
    assert list(rows) == ['ACME']
    assert rows['ACME']['total_debt'] == 900.0
    assert backend.schema_fields() == {'total_debt', 'company_name'}
