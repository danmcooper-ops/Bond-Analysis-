"""SEC XBRL company facts, reduced to what the credit scorecard needs.

    client = SECXBRLClient()
    facts = client.companyfacts(320193)          # CIK as int or str

One request per issuer to data.sec.gov/api/xbrl/companyfacts. The raw payload
runs to megabytes per company; the cache keeps only the tags derive() reads,
so a few hundred issuers cost megabytes, not a gigabyte.

Everything here is POINT-IN-TIME by the `filed` date, not the period end: a
10-K for December is not information on January 15th if it was filed on
February 20th. Using period ends would leak roughly six weeks of hindsight
into every divergence signal, the exact failure the equity path guards against.
"""

import json
import os
import time
from datetime import date, datetime

from data.http import get_json
from data.logging_setup import get_logger

log = get_logger('sec_xbrl')

FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'

# Bump when TAGS or the trimming changes, so stale trimmed caches are refetched.
CACHE_VERSION = 1

# Tag fallbacks, first present wins. Duration (flow) and instant (stock)
# concepts are kept apart because they are aggregated differently.
FLOW_TAGS = {
    'revenue': ('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
                'SalesRevenueNet'),
    'operating_income': ('OperatingIncomeLoss',),
    'interest_expense': ('InterestExpense', 'InterestExpenseNonoperating',
                         'InterestExpenseDebt'),
    'dna': ('DepreciationDepletionAndAmortization', 'DepreciationAndAmortization',
            'DepreciationAmortizationAndAccretionNet'),
    'cfo': ('NetCashProvidedByUsedInOperatingActivities',),
    'capex': ('PaymentsToAcquirePropertyPlantAndEquipment',),
}
INSTANT_TAGS = {
    'long_term_debt': ('LongTermDebt',),
    'long_term_debt_noncurrent': ('LongTermDebtNoncurrent',),
    'debt_current': ('DebtCurrent', 'LongTermDebtCurrent'),
    'short_term_borrowings': ('ShortTermBorrowings', 'CommercialPaper'),
    'cash': ('CashAndCashEquivalentsAtCarryingValue',
             'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'),
    'assets': ('Assets',),
    'liabilities': ('Liabilities',),
    'assets_current': ('AssetsCurrent',),
    'liabilities_current': ('LiabilitiesCurrent',),
    'retained_earnings': ('RetainedEarningsAccumulatedDeficit',),
    'equity': ('StockholdersEquity',),
    'deposits': ('Deposits',),
}
SHARES_TAGS = (('dei', 'EntityCommonStockSharesOutstanding'),
               ('us-gaap', 'CommonStockSharesOutstanding'))

KEEP_KEYS = ('start', 'end', 'val', 'filed', 'form')


def _all_tags():
    tags = {t for group in FLOW_TAGS.values() for t in group}
    tags |= {t for group in INSTANT_TAGS.values() for t in group}
    return tags


def trim(payload):
    """Reduce a companyfacts payload to the facts derive() reads."""
    facts = (payload or {}).get('facts') or {}
    wanted = _all_tags()
    out = {'us-gaap': {}, 'dei': {}}
    for tag, body in (facts.get('us-gaap') or {}).items():
        if tag in wanted or tag == 'CommonStockSharesOutstanding':
            out['us-gaap'][tag] = {unit: [{k: f.get(k) for k in KEEP_KEYS}
                                          for f in rows]
                                   for unit, rows in (body.get('units') or {}).items()}
    for namespace, tag in SHARES_TAGS:
        if namespace == 'dei':
            body = (facts.get('dei') or {}).get(tag)
            if body:
                out['dei'][tag] = {unit: [{k: f.get(k) for k in KEEP_KEYS}
                                          for f in rows]
                                   for unit, rows in (body.get('units') or {}).items()}
    return {'entityName': (payload or {}).get('entityName'), 'facts': out}


class SECXBRLClient:
    """Fetches and caches trimmed company facts per CIK."""

    def __init__(self, cache_dir=None, max_age_days=30):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cache', 'sec', 'companyfacts')
        self.max_age_days = max_age_days
        self._memo = {}

    def _path(self, cik):
        return os.path.join(self.cache_dir, f'CIK{int(cik):010d}.json')

    def _load_cache(self, cik):
        path = self._path(cik)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) >= self.max_age_days * 86400:
            return None
        try:
            with open(path, encoding='utf-8') as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            return None
        return payload if payload.get('version') == CACHE_VERSION else None

    def _save_cache(self, cik, payload):
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._path(cik)
        tmp = f'{path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def companyfacts(self, cik, force=False):
        """Trimmed facts for one CIK, or None when unavailable."""
        try:
            cik = int(cik)
        except (TypeError, ValueError):
            return None
        if not force:
            if cik in self._memo:
                return self._memo[cik]
            cached = self._load_cache(cik)
            if cached is not None:
                self._memo[cik] = cached
                return cached
        raw = get_json(FACTS_URL.format(cik=cik), timeout=60)
        if not raw or 'facts' not in raw:
            log.warning('No XBRL company facts for CIK %s', cik)
            self._memo[cik] = None
            return None
        payload = {'version': CACHE_VERSION, **trim(raw)}
        self._memo[cik] = payload
        self._save_cache(cik, payload)
        return payload


# ---------------------------------------------------------------------------
# Point-in-time derivation
# ---------------------------------------------------------------------------

def _parse(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def _facts_for(facts, tags, unit='USD', namespace='us-gaap'):
    """Facts of the first tag present, as (start, end, val, filed) tuples."""
    for tag in tags:
        rows = ((facts.get(namespace) or {}).get(tag) or {}).get(unit)
        if rows:
            out = []
            for f in rows:
                end, filed = _parse(f.get('end')), _parse(f.get('filed'))
                if end is None or filed is None or f.get('val') is None:
                    continue
                out.append((_parse(f.get('start')), end, float(f['val']), filed))
            if out:
                return out
    return []


def _known_by(rows, as_of):
    """Keep facts filed on or before as_of; latest filing wins per period."""
    best = {}
    for start, end, val, filed in rows:
        if filed > as_of:
            continue
        key = (start, end)
        if key not in best or filed > best[key][1]:
            best[key] = (val, filed)
    return {k: v[0] for k, v in best.items()}


def instant_at(facts, tags, as_of):
    """(value, period_end) of the latest instant fact known by as_of."""
    known = _known_by(_facts_for(facts, tags), as_of)
    if not known:
        return None, None
    (_, end), val = max(known.items(), key=lambda kv: kv[0][1])
    return val, end


def ttm_at(facts, tags, as_of):
    """(trailing-twelve-month value, period_end) known by as_of.

    The latest annual figure, rolled forward by the current fiscal year-to-date
    less the same span a year earlier when a later 10-Q exists. Falls back to
    the annual figure alone.
    """
    known = _known_by(_facts_for(facts, tags), as_of)
    durations = {(s, e): v for (s, e), v in known.items() if s is not None}
    annual = [(e, v) for (s, e), v in durations.items()
              if 330 <= (e - s).days <= 400]
    if not annual:
        return None, None
    a_end, a_val = max(annual)
    ytd = [((e - s).days, e, v) for (s, e), v in durations.items()
           if e > a_end and 60 <= (e - s).days <= 300]
    if not ytd:
        return a_val, a_end
    y_end = max(e for _, e, _ in ytd)
    span, _, y_val = max(t for t in ytd if t[1] == y_end)
    for (s, e), v in durations.items():
        if (abs((e - s).days - span) <= 20
                and abs((y_end - e).days - 365) <= 20):
            return a_val + y_val - v, y_end
    return a_val, a_end


def shares_at(facts, as_of):
    for namespace, tag in SHARES_TAGS:
        rows = _facts_for(facts, (tag,), unit='shares', namespace=namespace)
        known = _known_by(rows, as_of)
        if known:
            (_, end), val = max(known.items(), key=lambda kv: kv[0][1])
            return val
    return None


def altman_zone(z):
    if z is None:
        return None
    if z > 2.99:
        return 'safe'
    if z >= 1.81:
        return 'grey'
    return 'distress'


def derive(payload, as_of, price=None):
    """Scorecard fields known by `as_of`, in the equity snapshot's conventions.

    `price` is the share price on or before as_of, for market capitalisation;
    without it mcap and the market-value Altman term are omitted rather than
    guessed. Returns a dict with only the fields that could be derived, plus
    _fundamentals_source and _fundamentals_asof; None if nothing could be.
    """
    facts = (payload or {}).get('facts') or {}
    flow = {k: ttm_at(facts, tags, as_of) for k, tags in FLOW_TAGS.items()}
    inst = {k: instant_at(facts, tags, as_of) for k, tags in INSTANT_TAGS.items()}
    fv = {k: v[0] for k, v in flow.items()}
    iv = {k: v[0] for k, v in inst.items()}

    out = {}
    if iv['long_term_debt'] is not None:
        debt = iv['long_term_debt'] + (iv['short_term_borrowings'] or 0.0)
    elif iv['long_term_debt_noncurrent'] is not None or iv['debt_current'] is not None:
        debt = ((iv['long_term_debt_noncurrent'] or 0.0) + (iv['debt_current'] or 0.0)
                + (iv['short_term_borrowings'] or 0.0))
    else:
        debt = None
    if debt is not None:
        out['total_debt'] = debt
        if iv['cash'] is not None:
            out['net_debt'] = debt - iv['cash']
        if iv['equity']:
            out['de'] = debt / iv['equity']

    if fv['revenue'] is not None:
        out['revenue'] = fv['revenue']
    ebit = fv['operating_income']
    if ebit is not None and fv['interest_expense']:
        out['int_cov'] = ebit / fv['interest_expense']
    if ebit is not None and fv['dna'] is not None and 'net_debt' in out:
        ebitda = ebit + fv['dna']
        if ebitda > 0:
            out['nd_ebitda'] = out['net_debt'] / ebitda
    if fv['cfo'] is not None and fv['capex'] is not None:
        out['fcf'] = fv['cfo'] - fv['capex']

    shares = shares_at(facts, as_of)
    if shares and price:
        out['mcap'] = shares * price

    needed = ('assets_current', 'liabilities_current', 'retained_earnings',
              'assets', 'liabilities')
    if (all(iv[k] is not None for k in needed) and iv['assets']
            and iv['liabilities'] and ebit is not None
            and fv['revenue'] is not None and out.get('mcap')):
        ta = iv['assets']
        z = (1.2 * (iv['assets_current'] - iv['liabilities_current']) / ta
             + 1.4 * iv['retained_earnings'] / ta
             + 3.3 * ebit / ta
             + 0.6 * out['mcap'] / iv['liabilities']
             + 1.0 * fv['revenue'] / ta)
        out['altman_z'] = z
        out['altman_z_zone'] = altman_zone(z)

    # Banks report deposits; their cash flow and EBITDA are not a leverage
    # story, so they must reach the financial scorecard.
    if iv['deposits']:
        out['sector'] = 'Financial Services'

    if not out:
        return None
    ends = [v[1] for v in list(flow.values()) + list(inst.values()) if v[1]]
    out['company_name'] = (payload or {}).get('entityName')
    out['_fundamentals_source'] = 'sec_xbrl'
    out['_fundamentals_asof'] = max(ends).isoformat() if ends else None
    return out
