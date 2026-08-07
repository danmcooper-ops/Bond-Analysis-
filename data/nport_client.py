"""SEC Form N-PORT bulk data sets: the per-CUSIP price source.

Every registered fund and ETF discloses its monthly holdings on Form N-PORT-P,
and DERA republishes them as flattened quarterly TSVs. Because many funds hold
the same bond, the cross-fund median of their carrying values is a defensible
consensus mark — which is what makes a corporate bond model possible without a
TRACE licence.

FIELD NAMES, VERIFIED AGAINST 2026Q2 RATHER THAN ASSUMED
---------------------------------------------------------
Several differ from what the SEC's own prose documentation implies:

  ISSUER_CUSIP    the CUSIP (not `cusip`), with '999999999' as the sentinel
                  for holdings that have none — private placements, FX
                  forwards, cash. Roughly a third of all rows.
  CURRENCY_VALUE  the value IN USD (not `VALUE_USD`), even when CURRENCY_CODE
                  is not USD. BALANCE, however, is face in the LOCAL currency.
                  So the naive CURRENCY_VALUE/BALANCE*100 silently returns an
                  FX-contaminated price for non-USD paper: an AUD bond at par
                  came out at 71.99. The local-currency price needs the
                  EXCHANGE_RATE multiplier, and EXCHANGE_RATE is blank on
                  every USD row.
  PERCENTAGE      percent of fund NAV (not `pct_val`).
  ISSUER_TYPE     CORP / UST / USGSE / MUN / ... (not `issuer_cat`).
  ISSUER_TITLE    the title of issue, which carries seniority wording.

Dates are DD-MON-YYYY. Booleans are 'Y'/'N'.

REPORT_DATE vs REPORT_ENDING_PERIOD is the trap in SUBMISSION.tsv:

  REPORT_DATE           the month-end the HOLDINGS are as of  <- the mark date
  REPORT_ENDING_PERIOD  the fund's own fiscal period end

They routinely differ by months — a filing with REPORT_DATE 28-FEB-2026 can
carry REPORT_ENDING_PERIOD 30-NOV-2026. Marking a bond to its fund's fiscal
year-end instead of the holdings date would misdate the price by up to a year.

Files are streamed out of the ZIP, never extracted: FUND_REPORTED_HOLDING.tsv
alone is 910 MB uncompressed.
"""

import csv
import io
import os
import re
import zipfile
from datetime import date, datetime

from data.http import download_atomic, get
from data.logging_setup import get_logger

log = get_logger('nport')

INDEX_URL = ('https://www.sec.gov/data-research/sec-markets-data/'
             'form-n-port-data-sets')
ZIP_URL = ('https://www.sec.gov/files/dera/data/form-n-port-data-sets/'
           '{quarter}_nport.zip')

HOLDINGS_TABLE = 'FUND_REPORTED_HOLDING.tsv'
DEBT_TABLE = 'DEBT_SECURITY.tsv'
SUBMISSION_TABLE = 'SUBMISSION.tsv'

# Holdings with no CUSIP use a sentinel rather than leaving the field blank,
# and there is more than one. '999999999' is documented; '000000000' is not,
# and it silently accumulated 5,109 holdings in 2026Q2 with implied prices
# from 0 to 1052 — a catch-all bucket that would otherwise have become the
# single most widely "held" bond in the universe.
NO_CUSIP_SENTINELS = {'999999999', '000000000', 'N/A', 'NA', 'NONE',
                      'UNKNOWN', 'XXXXXXXXX'}

# CUSIP check-digit alphabet: digits are face value, letters A-Z are 10-35,
# and the three special characters continue the sequence.
_CUSIP_VALUES = {**{str(d): d for d in range(10)},
                 **{chr(ord('A') + i): 10 + i for i in range(26)},
                 '*': 36, '@': 37, '#': 38}


def is_valid_cusip(cusip):
    """Validate a CUSIP by its check digit.

    Catches transpositions and truncations that a length check misses. Note
    it does NOT catch '000000000', whose check digit is legitimately 0 — hence
    the sentinel set as well. Two filters because neither alone is enough.
    """
    text = (cusip or '').strip().upper()
    if len(text) != 9 or text in NO_CUSIP_SENTINELS:
        return False
    total = 0
    for i, char in enumerate(text[:8]):
        value = _CUSIP_VALUES.get(char)
        if value is None:
            return False
        if i % 2:                      # double every second character
            value *= 2
        total += value // 10 + value % 10
    try:
        return (10 - (total % 10)) % 10 == int(text[8])
    except ValueError:
        return False

# UNIT codes. PA is principal amount, which is the only one that makes
# CURRENCY_VALUE/BALANCE a price; NS is number of shares, NC number of
# contracts, OU some other unit the fund describes in free text.
UNIT_PRINCIPAL = 'PA'

DEBT_ASSET_CATS = ('DBT',)
TARGET_ISSUER_TYPES = ('CORP', 'UST', 'USGSE', 'USGA')

_MONTHS = {m: i for i, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], start=1)}

# Required columns per table. An upstream rename must fail loudly here rather
# than silently yielding None for every row of a 900 MB file.
REQUIRED_COLUMNS = {
    HOLDINGS_TABLE: ('ACCESSION_NUMBER', 'HOLDING_ID', 'ISSUER_NAME',
                     'ISSUER_TITLE', 'ISSUER_CUSIP', 'BALANCE', 'UNIT',
                     'CURRENCY_CODE', 'CURRENCY_VALUE', 'EXCHANGE_RATE',
                     'PERCENTAGE', 'PAYOFF_PROFILE', 'ASSET_CAT',
                     'ISSUER_TYPE', 'FAIR_VALUE_LEVEL'),
    DEBT_TABLE: ('HOLDING_ID', 'MATURITY_DATE', 'COUPON_TYPE',
                 'ANNUALIZED_RATE', 'IS_DEFAULT', 'ARE_ANY_INTEREST_PAYMENT',
                 'IS_ANY_PORTION_INTEREST_PAID'),
    SUBMISSION_TABLE: ('ACCESSION_NUMBER', 'SUB_TYPE', 'REPORT_DATE',
                       'REPORT_ENDING_PERIOD'),
}


def parse_sec_date(text):
    """Parse DD-MON-YYYY, the format used throughout these files."""
    if not text:
        return None
    parts = str(text).strip().upper().split('-')
    if len(parts) != 3:
        return None
    try:
        return date(int(parts[2]), _MONTHS[parts[1]], int(parts[0]))
    except (KeyError, ValueError):
        return None


def parse_yn(value):
    return str(value or '').strip().upper() == 'Y'


def _number(value):
    if value in (None, '', 'N/A'):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def implied_price(balance, currency_value, exchange_rate=None,
                  currency_code='USD'):
    """Price per 100 face, in the bond's OWN currency.

    CURRENCY_VALUE is in USD while BALANCE is local face, so a non-USD bond
    needs the exchange-rate multiplier to bring the two onto the same basis.
    Verified on an AUD holding: 974.14 face valued at 701.32 USD with a rate
    of 1.389 is a bond at exactly par, which the naive ratio reports as 71.99.
    """
    bal, val = _number(balance), _number(currency_value)
    if bal is None or val is None or bal <= 0:
        return None
    price = val / bal * 100.0
    if currency_code and currency_code != 'USD':
        fx = _number(exchange_rate)
        if not fx or fx <= 0:
            return None          # cannot convert; refuse rather than guess
        price *= fx
    return price


class NPORTClient:
    """Downloads and streams the quarterly N-PORT data sets."""

    def __init__(self, cache_dir=None, keep_zip=True):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cache', 'nport')
        self.keep_zip = keep_zip

    # -- discovery ----------------------------------------------------------

    def list_available_quarters(self):
        """Quarters published by DERA, oldest first. [] on failure."""
        html = get(INDEX_URL, timeout=60)
        if not html:
            log.error('Could not read the N-PORT data-sets index')
            return []
        found = re.findall(r'form-n-port-data-sets/(\d{4}q\d)_nport\.zip', html)
        return sorted(set(found))

    def zip_path(self, quarter):
        return os.path.join(self.cache_dir, f'{quarter}_nport.zip')

    def parquet_path(self, quarter):
        return os.path.join(self.cache_dir, f'{quarter}_debt_holdings.parquet')

    def download_quarter(self, quarter, force=False):
        """Fetch one quarterly ZIP. Returns the path, or None."""
        dest = self.zip_path(quarter)
        if os.path.exists(dest) and not force:
            if zipfile.is_zipfile(dest):
                log.info('%s already cached (%.0f MB)', quarter,
                         os.path.getsize(dest) / 1e6)
                return dest
            log.warning('%s is cached but not a valid ZIP — refetching', quarter)

        url = ZIP_URL.format(quarter=quarter)
        log.info('Downloading %s ...', url)
        path = download_atomic(url, dest, timeout=1800,
                               progress_every=100 * (1 << 20))
        if path is None:
            log.error('Download failed: %s', url)
            return None
        if not zipfile.is_zipfile(path):
            # A truncated ZIP that looks complete is worse than none: the next
            # run would read it and report a data problem rather than a
            # download problem.
            log.error('%s downloaded but is not a valid ZIP — removing', quarter)
            os.remove(path)
            return None
        log.info('%s: %.0f MB', quarter, os.path.getsize(path) / 1e6)
        return path

    # -- streaming ----------------------------------------------------------

    def iter_table(self, quarter, table, columns=None):
        """Yield dict rows from one table, streamed out of the ZIP.

        Never extracts: FUND_REPORTED_HOLDING.tsv is 910 MB uncompressed and
        several tables are larger than the compressed archive.
        """
        path = self.zip_path(quarter)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{quarter} not downloaded — run download_quarter first')

        with zipfile.ZipFile(path) as archive:
            if table not in archive.namelist():
                raise KeyError(f'{table} not in {quarter}: '
                               f'{sorted(archive.namelist())[:8]}...')
            with archive.open(table) as raw:
                stream = io.TextIOWrapper(raw, encoding='utf-8',
                                          errors='replace', newline='')
                reader = csv.DictReader(stream, delimiter='\t')
                self._check_columns(table, reader.fieldnames, quarter)
                for row in reader:
                    yield row

    @staticmethod
    def _check_columns(table, fieldnames, quarter):
        required = REQUIRED_COLUMNS.get(table)
        if not required:
            return
        missing = [c for c in required if c not in (fieldnames or ())]
        if missing:
            raise ValueError(
                f'{table} in {quarter} is missing required columns {missing}. '
                f'The SEC schema changed; update REQUIRED_COLUMNS and the '
                f'field mapping rather than letting these read as null.')

    # -- assembly -----------------------------------------------------------

    def report_dates(self, quarter):
        """{accession: report_date}. REPORT_DATE, not REPORT_ENDING_PERIOD."""
        out = {}
        for row in self.iter_table(quarter, SUBMISSION_TABLE):
            when = parse_sec_date(row.get('REPORT_DATE'))
            if when:
                out[row['ACCESSION_NUMBER']] = when
        log.info('%s: %d submissions', quarter, len(out))
        return out

    def build_holdings(self, quarter, issuer_types=TARGET_ISSUER_TYPES,
                       min_balance=1000.0, usd_only=True):
        """Debt holdings with a usable implied price, as a list of dicts.

        Filtered on the way through rather than after, because the unfiltered
        table is 910 MB and roughly a third of rows carry no CUSIP at all.

        min_balance excludes rows reporting a token BALANCE of 1 — private
        loans and participations where the fund is not reporting a principal
        amount at all, and where the implied price comes out in the millions.
        """
        submissions = self.report_dates(quarter)

        kept, stats = {}, {'rows': 0, 'no_cusip': 0, 'not_debt': 0,
                           'not_principal': 0, 'non_usd': 0, 'tiny': 0,
                           'bad_price': 0, 'no_submission': 0, 'wrong_issuer': 0}
        for row in self.iter_table(quarter, HOLDINGS_TABLE):
            stats['rows'] += 1
            if row.get('ASSET_CAT') not in DEBT_ASSET_CATS:
                stats['not_debt'] += 1
                continue
            cusip = (row.get('ISSUER_CUSIP') or '').strip().upper()
            if not is_valid_cusip(cusip):
                stats['no_cusip'] += 1
                continue
            if row.get('UNIT') != UNIT_PRINCIPAL:
                stats['not_principal'] += 1
                continue
            currency = (row.get('CURRENCY_CODE') or 'USD').strip().upper()
            if usd_only and currency != 'USD':
                stats['non_usd'] += 1
                continue
            if issuer_types and row.get('ISSUER_TYPE') not in issuer_types:
                stats['wrong_issuer'] += 1
                continue
            balance = _number(row.get('BALANCE'))
            if balance is None or balance < min_balance:
                stats['tiny'] += 1
                continue
            price = implied_price(balance, row.get('CURRENCY_VALUE'),
                                  row.get('EXCHANGE_RATE'), currency)
            if price is None:
                stats['bad_price'] += 1
                continue
            report_date = submissions.get(row.get('ACCESSION_NUMBER'))
            if report_date is None:
                stats['no_submission'] += 1
                continue

            kept[row['HOLDING_ID']] = {
                'holding_id': row['HOLDING_ID'],
                'accession': row['ACCESSION_NUMBER'],
                'report_date': report_date,
                'cusip': cusip,
                'issuer_name': (row.get('ISSUER_NAME') or '').strip(),
                'title_of_issue': (row.get('ISSUER_TITLE') or '').strip(),
                'balance': balance,
                'value_usd': _number(row.get('CURRENCY_VALUE')),
                'implied_price': price,
                'pct_of_nav': _number(row.get('PERCENTAGE')),
                'payoff_profile': (row.get('PAYOFF_PROFILE') or '').strip(),
                'issuer_type': row.get('ISSUER_TYPE'),
                'fair_value_level': _number(row.get('FAIR_VALUE_LEVEL')),
                'currency': currency,
            }

        log.info('%s: %d debt holdings kept from %d rows (%s)', quarter,
                 len(kept), stats['rows'],
                 ', '.join(f'{k}={v}' for k, v in stats.items()
                           if k != 'rows' and v))

        self._attach_debt_terms(quarter, kept)
        return list(kept.values())

    def _attach_debt_terms(self, quarter, kept):
        """Join DEBT_SECURITY onto the kept holdings by HOLDING_ID.

        Streamed and filtered against the already-selected ids rather than
        loaded whole: the table has millions of rows and only the retained
        fraction is needed.
        """
        matched = 0
        for row in self.iter_table(quarter, DEBT_TABLE):
            entry = kept.get(row.get('HOLDING_ID'))
            if entry is None:
                continue
            matched += 1
            entry.update({
                'maturity_date': parse_sec_date(row.get('MATURITY_DATE')),
                'coupon_type': (row.get('COUPON_TYPE') or '').strip() or None,
                'annualized_rate': _number(row.get('ANNUALIZED_RATE')),
                'is_default': parse_yn(row.get('IS_DEFAULT')),
                'in_arrears': parse_yn(row.get('ARE_ANY_INTEREST_PAYMENT')),
                'is_paid_kind': parse_yn(row.get('IS_ANY_PORTION_INTEREST_PAID')),
                'is_convertible': (parse_yn(row.get('IS_CONVTIBLE_MANDATORY'))
                                   or parse_yn(row.get('IS_CONVTIBLE_CONTINGENT'))),
            })
        log.info('%s: debt terms attached to %d of %d holdings',
                 quarter, matched, len(kept))

        # A holding with no DEBT_SECURITY row cannot be priced as a bond.
        for holding_id in [k for k, v in kept.items()
                           if v.get('maturity_date') is None]:
            del kept[holding_id]
