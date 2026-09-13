"""Monthly Statement of the Public Debt: amount outstanding per CUSIP.

Why this exists rather than reusing TreasuryDirect's `currentlyOutstanding`:
that field is populated for only about 40% of securities (97 of 111 bonds but
just 40 of 241 notes), and the gap is a characteristic of the feed, not of the
bonds. Scoring a liquidity gate off it made data availability the single
largest driver of the rating — securities where the field happened to be
filled averaged a 64.5 composite with 78% BUY, against 49.9 and 9% for the
rest. A gate that measures whether a field was populated is worse than no gate.

MSPD covers every marketable security, `issued_amt` is populated for all 874 of
them, it is free and keyless, and it is the authoritative source. Published
monthly, which is ample for a number that changes only at auction.

Amounts are in MILLIONS of dollars in the feed and are converted to dollars
here, because a silent unit mismatch in a log-scaled gate is invisible.
"""

import os
from datetime import date, datetime

from data.http import get_json
from data.logging_setup import get_logger

log = get_logger('mspd')

BASE = ('https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/'
        'debt/mspd/mspd_table_3_market')

# Rows that are totals or non-marketable rather than individual securities.
NON_SECURITY_CLASSES = {'Total Marketable', 'Federal Financing Bank'}

MILLIONS = 1e6

# Bump whenever amounts_from_records changes what it derives.
# 2: outstanding row wins over tranche rows (was: last row wins).
CACHE_VERSION = 2


def amounts_from_records(records):
    """({cusip: dollars}, {source: count}) from MSPD table-3 records.

    A reopened security has ONE row per tranche, and only one of those rows
    carries outstanding_amt — the total. Taking the last row per CUSIP, as
    this used to, usually kept a single tranche's issued_amt: 912828ZQ6 came
    out at $29.4bn against a true $109.7bn, for 227 of 462 CUSIPs.

    So: an outstanding_amt row always wins; failing that, tranches are summed
    (issued net of redeemed where both are present).
    """
    outstanding, tranche_sum, tranche_source = {}, {}, {}
    for rec in records:
        if rec.get('security_class1_desc') in NON_SECURITY_CLASSES:
            continue
        cusip = str(rec.get('security_class2_desc') or '').strip().upper()
        if len(cusip) != 9:
            continue            # totals and subtotals carry a label here

        total = _amount(rec.get('outstanding_amt'))
        issued = _amount(rec.get('issued_amt'))
        redeemed = _amount(rec.get('redeemed_amt'))

        if total is not None:
            outstanding[cusip] = max(total, outstanding.get(cusip, 0.0))
        elif issued is not None:
            net = issued + redeemed if redeemed is not None else issued
            tranche_sum[cusip] = tranche_sum.get(cusip, 0.0) + net
            if redeemed is None or tranche_source.get(cusip) == 'issued':
                tranche_source[cusip] = 'issued'
            else:
                tranche_source.setdefault(cusip, 'issued_net')

    amounts, sources = {}, {'outstanding': 0, 'issued_net': 0, 'issued': 0}
    for cusip in set(outstanding) | set(tranche_sum):
        if cusip in outstanding:
            value, source = outstanding[cusip], 'outstanding'
        else:
            value, source = tranche_sum[cusip], tranche_source[cusip]
        if value <= 0:
            continue
        amounts[cusip] = value * MILLIONS
        sources[source] += 1
    return amounts, sources


def _amount(value):
    if value in (None, '', 'null'):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MSPDClient:
    """Amount outstanding per Treasury CUSIP, from the monthly statement."""

    def __init__(self, cache_dir=None, max_age_days=25):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cache', 'mspd')
        # Published monthly, so a cache a little under a month old is fine.
        self.max_age_days = max_age_days
        self._memo = None

    def _cache_path(self):
        return os.path.join(self.cache_dir, 'amounts_outstanding.json')

    def _load_cache(self):
        path = self._cache_path()
        if not os.path.exists(path):
            return None
        age = (date.today() - date.fromtimestamp(os.path.getmtime(path))).days
        if age >= self.max_age_days:
            return None
        import json
        try:
            with open(path, encoding='utf-8') as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            return None
        # The cache holds DERIVED amounts, so a fix to the derivation must
        # invalidate it; otherwise the old numbers persist for the full TTL.
        if payload.get('version') != CACHE_VERSION:
            return None
        return payload

    def _save_cache(self, payload):
        import json
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path()
        tmp = f'{path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def latest_record_date(self):
        payload = get_json(BASE, params={'page[size]': 1, 'sort': '-record_date',
                                         'fields': 'record_date'})
        if not payload or not payload.get('data'):
            return None
        return payload['data'][0]['record_date']

    def fetch_amounts(self, force=False):
        """Return {cusip: amount_outstanding_usd}. {} on failure.

        Prefers `outstanding_amt` where present; otherwise falls back to
        issued plus redeemed (redemptions are reported negative), and finally
        to issued alone. Issued is populated for every security, which is what
        makes the coverage complete.
        """
        if not force:
            if self._memo is not None:
                return self._memo['amounts']
            cached = self._load_cache()
            if cached:
                self._memo = cached
                return cached['amounts']

        record_date = self.latest_record_date()
        if not record_date:
            log.error('MSPD: could not determine the latest record date')
            return {}

        payload = get_json(BASE, params={'filter': f'record_date:eq:{record_date}',
                                         'page[size]': 10000})
        if not payload or not payload.get('data'):
            log.error('MSPD: no data for %s', record_date)
            return {}

        amounts, sources = amounts_from_records(payload['data'])

        payload_out = {'version': CACHE_VERSION, 'record_date': record_date,
                       'amounts': amounts, 'sources': sources}
        self._memo = payload_out
        self._save_cache(payload_out)
        log.info('MSPD %s: %d CUSIPs (%s)', record_date, len(amounts),
                 ', '.join(f'{k}={v}' for k, v in sources.items() if v))
        return amounts

    def attach(self, rows, field='amount_outstanding_usd'):
        """Stamp amounts onto rows by CUSIP. Returns the number matched."""
        amounts = self.fetch_amounts()
        if not amounts:
            return 0
        matched = 0
        for row in rows:
            value = amounts.get((row.get('cusip') or '').strip().upper())
            if value is not None:
                row[field] = value
                row['amount_outstanding_source'] = 'mspd'
                matched += 1
        log.info('MSPD: matched %d of %d rows', matched, len(rows))
        return matched
