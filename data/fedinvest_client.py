"""TreasuryDirect FedInvest: daily end-of-day prices for every marketable Treasury.

The only free source of *observed* daily prices in this model. Fiscal Service
publishes a price for every marketable bill, note, bond, TIPS and FRN each
business day (buy, sell, end-of-day, per 100 face). Federal trust funds
use these prices to value their holdings. History goes back to at least 2008
(~220 securities a day then, ~460 now). A weekend or holiday returns an empty
file.

The site is a form behind a Spring CSRF token, so a session first GETs the
date-picker page for its token and cookie. One token covers many POSTs, and
an expired one is refreshed once.

Prices are clean, per 100 face. TIPS are quoted on the real (unindexed)
price, which is not the same thing as a nominal clean price. A 0.000000 means
no quote and becomes None, never zero.

Raw CSVs are cached one file per day under data/cache/fedinvest/<year>/, so a
backfill can be resumed and a rebuild does not refetch. An empty answer is
cached only once the date is RECENT_DAYS old, because before then it can just
mean "not published yet" and caching it would hide that day for good.
"""

import os
import re
from datetime import date, datetime, timedelta

import requests

from data.http import _LIMITER, user_agent
from data.logging_setup import get_logger

log = get_logger('fedinvest')

FORM_URL = 'https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate'
CSV_URL = 'https://www.treasurydirect.gov/GA-FI/FedInvest/securityPriceDetail'
HOST = 'www.treasurydirect.gov'

RECENT_DAYS = 7
_CSRF_RE = re.compile(r'name="_csrf" value="([^"]+)"')

# FedInvest security type -> the label used in our history files.
TYPE_MAP = {
    'MARKET BASED BILL': 'BILL',
    'MARKET BASED NOTE': 'NOTE',
    'MARKET BASED BOND': 'BOND',
    'MARKET BASED FRN': 'FRN',
    'TIPS': 'TIPS',
}


def _price(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def parse_csv(text, price_date):
    """FedInvest CSV -> list of row dicts. Bad lines are skipped, never raised.

    Columns: CUSIP, SECURITY TYPE, RATE, MATURITY DATE, CALL DATE, BUY, SELL,
    END OF DAY. No header row.
    """
    rows = []
    for line in (text or '').splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 8 or len(parts[0]) != 9:
            continue
        cusip, sec_type, rate, maturity, call, buy, sell, eod = parts[:8]
        try:
            maturity_date = datetime.strptime(maturity, '%m/%d/%Y').date()
        except ValueError:
            continue
        try:
            coupon = float(rate)
        except ValueError:
            coupon = None
        try:
            call_date = datetime.strptime(call, '%m/%d/%Y').date() if call else None
        except ValueError:
            call_date = None
        rows.append({
            'price_date': price_date,
            'cusip': cusip,
            'security_type': TYPE_MAP.get(sec_type, sec_type),
            'coupon': coupon,
            'maturity_date': maturity_date,
            'call_date': call_date,
            'buy': _price(buy),
            'sell': _price(sell),
            'end_of_day': _price(eod),
        })
    return rows


class FedInvestClient:
    def __init__(self, cache_dir=None, timeout=30):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = cache_dir or os.path.join(root, 'data', 'cache',
                                                   'fedinvest')
        self.timeout = timeout
        self._session = None
        self._token = None

    # -- cache ---------------------------------------------------------------

    def cache_path(self, day):
        return os.path.join(self.cache_dir, str(day.year), f'{day.isoformat()}.csv')

    def is_cached(self, day):
        return os.path.exists(self.cache_path(day))

    def _write_cache(self, day, text):
        path = self.cache_path(day)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp, path)

    # -- network -------------------------------------------------------------

    def _open(self):
        session = requests.Session()
        session.headers['User-Agent'] = user_agent()
        _LIMITER.wait(HOST)
        resp = session.get(FORM_URL, timeout=self.timeout)
        match = _CSRF_RE.search(resp.text or '') if resp.ok else None
        if not match:
            log.warning('FedInvest form returned no CSRF token (HTTP %s)',
                        resp.status_code)
            return False
        self._session, self._token = session, match.group(1)
        return True

    def _post(self, day):
        """Raw CSV text for *day*: '' for no prices, None on failure."""
        for attempt in range(2):
            if self._session is None and not self._open():
                return None
            _LIMITER.wait(HOST)
            try:
                resp = self._session.post(CSV_URL, timeout=self.timeout, data={
                    'priceDateDay': day.day, 'priceDateMonth': day.month,
                    'priceDateYear': day.year, 'fileType': 'csv',
                    'csv': 'CSV FORMAT', '_csrf': self._token})
            except requests.RequestException as exc:
                log.warning('FedInvest %s: %s', day, exc)
                self._session = None
                continue
            if resp.status_code == 200 and 'csv' in resp.headers.get(
                    'content-type', ''):
                return resp.content.decode('iso-8859-1')
            # An expired session answers 403 or an HTML page: re-open once.
            log.info('FedInvest %s: HTTP %s %s; refreshing session', day,
                     resp.status_code, resp.headers.get('content-type'))
            self._session = None
        return None

    def raw(self, day, refresh=False):
        """Cached-or-fetched CSV text. None means the fetch failed."""
        path = self.cache_path(day)
        if not refresh and os.path.exists(path):
            with open(path, encoding='utf-8') as fh:
                return fh.read()
        text = self._post(day)
        if text is None:
            return None
        if text.strip() or (date.today() - day).days >= RECENT_DAYS:
            self._write_cache(day, text)
        return text

    def prices_on(self, day, refresh=False):
        text = self.raw(day, refresh=refresh)
        return None if text is None else parse_csv(text, day)

    def backfill(self, start, end, refresh=False, progress_every=100):
        """Fetch every weekday in [start, end] not already cached.

        Returns (fetched, failed). Weekends are skipped without a request;
        holidays are fetched once and cached empty.
        """
        fetched, failed = 0, []
        day = start
        while day <= end:
            if day.weekday() < 5 and (refresh or not self.is_cached(day)):
                if self.raw(day, refresh=refresh) is None:
                    failed.append(day)
                else:
                    fetched += 1
                    if fetched % progress_every == 0:
                        log.info('FedInvest backfill: %d days fetched, at %s',
                                 fetched, day)
            day += timedelta(days=1)
        return fetched, failed

    def cached_days(self):
        days = []
        if not os.path.isdir(self.cache_dir):
            return days
        for year in sorted(os.listdir(self.cache_dir)):
            folder = os.path.join(self.cache_dir, year)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if name.endswith('.csv'):
                    try:
                        days.append(date.fromisoformat(name[:-4]))
                    except ValueError:
                        pass
        return days
