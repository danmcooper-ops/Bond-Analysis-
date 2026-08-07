"""TreasuryDirect: authoritative reference data for Treasury securities.

The curve gives yields at standard tenors; this gives the exact coupon,
maturity, dated date and payment frequency for a specific CUSIP. That is what
lets the model price an actual Treasury rather than a curve point — an
off-the-run 4.25% of Nov-2034 is a different instrument from "the 8-year
point", and holders own the former.

Also the reference set the N-PORT clean-vs-dirty validation harness leans on:
Treasury prices are derivable exactly from the curve plus this coupon and
maturity, so any systematic gap against the N-PORT-implied price falsifies
the assumption that fund marks are clean.
"""

import os
from datetime import date, datetime

from data.http import get_json
from data.logging_setup import get_logger

log = get_logger('treasury_direct')

SEARCH_URL = 'https://www.treasurydirect.gov/TA_WS/securities/search'

FREQUENCY_MAP = {
    'semi-annual': 2, 'semiannual': 2, 'annual': 1, 'quarterly': 4,
    'monthly': 12, 'none': 0, 'zero-coupon': 0, 'at maturity': 0,
}

SECURITY_TYPE_TO_CLASS = {
    'Bill': 'TREASURY_BILL',
    'Note': 'TREASURY',
    'Bond': 'TREASURY',
    'TIPS': 'TREASURY',       # nominal pricing only — see instrument_type()
    'FRN': 'TREASURY',        # floater — rejected downstream as unanalyzable
    'CMB': 'TREASURY_BILL',
}


def _yes(value):
    """Parse TreasuryDirect's boolean fields, which are 'Yes'/'No' STRINGS.

    Every flag in the feed — floatingRate, tips, reopening, strippable,
    callable — comes back as the literal string 'Yes' or 'No'. `bool('No')`
    is True, so testing truthiness marks every security as having every flag.
    That is how a plain 4.25% note briefly got classified as a floater and
    rejected from the universe: `bool(rec.get('floatingRate'))` was True for
    all 6 of 6 securities.
    """
    return str(value).strip().lower() in ('yes', 'true', 'y', '1')


def instrument_type(rec):
    """The instrument's real type, which is NOT rec['securityType'].

    TreasuryDirect carries two type fields and they disagree. For a 10-year
    TIPS, `securityType` is 'Note' — describing the auction format — while
    `type` is 'TIPS' and a separate `tips` field says 'Yes'. Reading
    securityType alone silently classifies every TIPS as a nominal note.

    That is not cosmetic. A TIPS coupon is a REAL rate: the 1.750% of
    Jan-2034 priced against a 4.6% nominal curve looks like a wildly
    off-market bond, when in fact its principal accretes with CPI and its
    yield is not comparable to a nominal yield at all. Two of the six
    Treasuries maturing in 2034 are TIPS, so this is a third of that maturity
    bucket, not an edge case.

    Precedence: the explicit `tips` flag, then `type`, then `securityType`.
    """
    if _yes(rec.get('tips')):
        return 'TIPS'
    if _yes(rec.get('floatingRate')):
        return 'FRN'
    for field in ('type', 'securityType'):
        value = (rec.get(field) or '').strip()
        if value:
            return value
    return ''


def is_inflation_linked(rec):
    """True for TIPS, checked several ways because one flag can be absent."""
    if instrument_type(rec) == 'TIPS':
        return True
    # Corroborating fields present only on inflation-linked issues.
    return any(rec.get(f) for f in
               ('indexRatioOnIssueDate', 'refCpiOnDatedDate',
                'cpiBaseReferencePeriod', 'tiinConversionFactorPer1000'))


def is_floating_rate(rec):
    return instrument_type(rec) == 'FRN' or _yes(rec.get('floatingRate'))


def is_callable(rec):
    """Modern Treasuries are not callable, but the feed carries the flag and
    the older long bonds it applies to are still outstanding."""
    return _yes(rec.get('callable'))


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def _parse_rate(value):
    """TreasuryDirect returns rates as percent strings ('4.250000')."""
    if value in (None, ''):
        return None
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def _parse_amount(value):
    """Dollar amounts arrive as strings ('33999000000.000000')."""
    if value in (None, ''):
        return None
    try:
        amount = float(str(value).replace(',', ''))
    except (TypeError, ValueError):
        return None
    return amount if amount > 0 else None


class TreasuryDirectClient:
    """Fetches auctioned Treasury reference data, cached to disk."""

    def __init__(self, cache_dir=None, max_age_days=7):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cache', 'treasury_direct')
        self.max_age_days = max_age_days
        self._memo = {}

    def _cache_path(self, key):
        return os.path.join(self.cache_dir, f'{key}.json')

    def _load_cache(self, key):
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        age = (date.today() - date.fromtimestamp(os.path.getmtime(path))).days
        if age >= self.max_age_days:
            return None
        import json
        try:
            with open(path, encoding='utf-8') as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _save_cache(self, key, payload):
        import json
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path(key)
        tmp = f'{path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def fetch_by_maturity_range(self, start, end, security_type=None,
                                force=False):
        """Securities maturing in [start, end].

        Queried by maturity rather than issue date because the model cares
        about what is outstanding now, not what was auctioned when — a 30-year
        bond issued in 2006 is very much part of today's universe.
        """
        key = (f'mat_{start:%Y%m%d}_{end:%Y%m%d}'
               f'_{security_type or "all"}')
        if not force:
            if key in self._memo:
                return self._memo[key]
            cached = self._load_cache(key)
            if cached is not None:
                self._memo[key] = cached
                return cached

        params = {'format': 'json', 'dateFieldName': 'maturityDate',
                  'startDate': start.isoformat(), 'endDate': end.isoformat()}
        if security_type:
            params['type'] = security_type

        payload = get_json(SEARCH_URL, params=params)
        if payload is None:
            log.error('TreasuryDirect search failed for %s..%s', start, end)
            return []
        if not isinstance(payload, list):
            log.error('Unexpected TreasuryDirect payload type: %s',
                      type(payload).__name__)
            return []

        self._memo[key] = payload
        self._save_cache(key, payload)
        log.info('TreasuryDirect: %d securities maturing %s..%s',
                 len(payload), start, end)
        return payload

    def fetch_outstanding(self, as_of=None, max_years=31, force=False):
        """Every Treasury maturing between `as_of` and max_years out.

        Chunked one year at a time. The API caps a single response, and a
        single 31-year query silently truncates rather than erroring — which
        would quietly drop the long end of the universe.
        """
        target = as_of or date.today()
        rows, seen = [], set()
        for offset in range(max_years + 1):
            y0 = date(target.year + offset, 1, 1)
            y1 = date(target.year + offset, 12, 31)
            if y1 < target:
                continue
            for row in self.fetch_by_maturity_range(max(y0, target), y1,
                                                     force=force):
                cusip = row.get('cusip')
                if cusip and cusip not in seen:
                    seen.add(cusip)
                    rows.append(row)
        log.info('TreasuryDirect: %d distinct outstanding securities', len(rows))
        return rows

    @staticmethod
    def to_bond_rows(records, as_of=None):
        """Normalise TreasuryDirect records into pipeline rows.

        Emits the field names models.bond_types.from_row expects, so the
        Treasury path and the corporate path converge on one schema.

        DEDUPED BY CUSIP. TreasuryDirect returns one record per AUCTION, and
        Treasury reopens issues repeatedly — 91282CLW9 comes back three times
        for the 2034 maturity year, once per reopening, identical except for
        securityTerm and issueDate. Without deduping, that security would
        enter the universe three times, triple-weighting it in every
        peer-relative percentile pool and in any aggregate. The earliest
        issueDate wins, since that is the original auction and the fields that
        matter for pricing (coupon, maturity, dated date) are identical
        across reopenings.

        TIPS are marked but NOT specially handled: this model prices them on
        their nominal coupon and ignores the inflation accrual, which
        understates them. They are flagged `is_inflation_linked` so the gate
        layer can mark them inapplicable rather than rate them wrongly.
        """
        target = as_of or date.today()

        by_cusip = {}
        for rec in records:
            cusip = (rec.get('cusip') or '').strip().upper()
            if not cusip:
                continue
            # originalIssueDate is present on reopenings and points at the
            # first auction; issueDate is this particular reopening's.
            issued = (_parse_date(rec.get('originalIssueDate'))
                      or _parse_date(rec.get('issueDate')) or date.max)
            prior = by_cusip.get(cusip)
            if prior is None or issued < prior[0]:
                by_cusip[cusip] = (issued, rec)
        deduped = [rec for _, rec in by_cusip.values()]
        if len(deduped) < len(records):
            log.info('TreasuryDirect: %d auction records -> %d distinct CUSIPs '
                     '(reopenings collapsed)', len(records), len(deduped))

        out = []
        skipped_unissued = 0
        skipped_no_coupon = 0
        for rec in deduped:
            maturity = _parse_date(rec.get('maturityDate'))
            if maturity is None or maturity <= target:
                continue
            sec_type = instrument_type(rec)
            inflation_linked = is_inflation_linked(rec)
            floating = is_floating_rate(rec)
            # Not yet issued: an announced auction that has not settled is not
            # something anyone can own. TreasuryDirect returns forthcoming
            # auctions alongside outstanding paper, with a blank interestRate
            # because the coupon is set AT the auction. Reading that blank as
            # a 0% coupon produced a "3-Year Note" yielding 13.8% and a
            # "10-Year" yielding 60% at the top of the BUY list.
            issue_date = (_parse_date(rec.get('originalIssueDate'))
                          or _parse_date(rec.get('issueDate')))
            if issue_date is not None and issue_date > target:
                skipped_unissued += 1
                continue

            coupon = _parse_rate(rec.get('interestRate'))
            freq_raw = (rec.get('interestPaymentFrequency') or '').strip().lower()
            frequency = FREQUENCY_MAP.get(freq_raw)

            is_discount = sec_type in ('Bill', 'CMB')
            if coupon is None:
                if is_discount:
                    # Bills genuinely have no coupon; the field is blank.
                    coupon = 0.0
                    frequency = 0
                else:
                    # A coupon-bearing security with no coupon is unusable.
                    # Leave it None so from_row rejects it with a reason
                    # rather than silently inventing a 0% bond.
                    skipped_no_coupon += 1
                    continue
            if frequency is None:
                frequency = 0 if is_discount else 2

            # A TIPS coupon is a REAL rate and its principal accretes with
            # CPI. This model has no inflation curve, so it cannot price one.
            # Marking it 'Inflation-Linked' routes it through the same
            # is_analyzable rejection as a floater instead of letting a 1.75%
            # real coupon be compared against a 4.6% nominal curve.
            if inflation_linked:
                coupon_type = 'Inflation-Linked'
            elif floating:
                coupon_type = 'Floating'
            else:
                coupon_type = 'Fixed'

            out.append({
                'cusip': (rec.get('cusip') or '').strip().upper(),
                'issuer_name': 'UNITED STATES TREASURY',
                'title_of_issue': f"US TREASURY {sec_type.upper()} "
                                  f"{coupon * 100:.3f}% {maturity:%m/%d/%y}",
                'coupon_rate': coupon,
                'maturity_date': maturity,
                'frequency': frequency,
                'dated_date': (_parse_date(rec.get('originalDatedDate'))
                               or _parse_date(rec.get('datedDate'))),
                'issue_date': (_parse_date(rec.get('originalIssueDate'))
                               or _parse_date(rec.get('issueDate'))),
                'asset_class': SECURITY_TYPE_TO_CLASS.get(sec_type, 'TREASURY'),
                'security_type': sec_type,
                'security_term': (rec.get('originalSecurityTerm')
                                  or rec.get('securityTerm')),
                'coupon_type': coupon_type,
                'is_inflation_linked': inflation_linked,
                'is_callable': is_callable(rec),
                'is_convertible': False,
                'is_default': False,
                'in_arrears': False,
                'is_paid_kind': False,
                'seniority_rank': 1,
                'seniority_source': 'sovereign',
                'issuer_cat': 'UST',
                # Amount outstanding is a real liquidity measure and the only
                # one a Treasury has before N-PORT marks are attached. Note it
                # is a TRUE outstanding figure, unlike the corporate side
                # where fund holdings are only a lower bound on issue size.
                'amount_outstanding_usd': _parse_amount(
                    rec.get('currentlyOutstanding')),
                'reference_source': 'treasury_direct',
            })
        if skipped_unissued or skipped_no_coupon:
            log.info('TreasuryDirect: skipped %d not-yet-issued and %d '
                     'without a coupon (forthcoming auctions)',
                     skipped_unissued, skipped_no_coupon)
        return out
