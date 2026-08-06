"""The Bond value object and strict construction from a raw data row.

`from_row` is deliberately strict and returns None rather than a
half-populated Bond. A row missing a coupon or a maturity cannot be priced,
and the honest outcome is that it never enters the analytics at all — not
that it enters with a guessed coupon and produces a confident wrong spread.
The reason for each rejection is recorded so the run can report *why* rows
dropped out rather than just how many.
"""

from dataclasses import dataclass, field
from datetime import date, datetime

from models.conventions import (classify_by_cusip, conventions_for,
                                is_analyzable)

SENIORITY_SENIOR_SECURED = 1
SENIORITY_SENIOR_UNSECURED = 2
SENIORITY_SENIOR_SUB = 3
SENIORITY_SUB = 4
SENIORITY_JUNIOR = 5


@dataclass(frozen=True)
class Bond:
    cusip: str
    issuer_name: str
    coupon_rate: float                 # decimal, e.g. 0.05 for a 5% coupon
    maturity: date
    frequency: int
    convention: str
    comp: str
    asset_class: str
    face: float = 100.0
    dated_date: date = None
    eom: bool = None
    seniority_rank: int = SENIORITY_SENIOR_UNSECURED
    seniority_source: str = 'default'
    is_callable: bool = False
    call_schedule: tuple = field(default=None)

    def years_to_maturity(self, settle):
        return (self.maturity - settle).days / 365.25


def _parse_date(value):
    """Accept a date, a datetime, or an ISO-ish string. None on anything else."""
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()[:10]
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y'):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_coupon(value):
    """Return a decimal coupon rate, or None.

    N-PORT reports ANNUALIZED_RATE as a percentage (5.0 for a 5% coupon), but
    other sources use decimals. Anything above 1.0 is read as a percentage —
    a genuine 100%+ coupon does not exist in this universe, whereas a feed
    that switches units silently very much does.
    """
    if value is None or value == '':
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if rate != rate:                       # NaN
        return None
    if abs(rate) > 1.0:
        rate = rate / 100.0
    if rate < 0 or rate > 0.40:
        return None
    return rate


def from_row(row, settle=None):
    """Build a Bond from a raw pipeline row, or return None.

    Returns (bond, reason). `reason` is None on success and a short string
    naming the first disqualifying problem otherwise, so the caller can
    aggregate drop reasons across the run.
    """
    cusip = (row.get('cusip') or '').strip().upper()
    if not cusip:
        return None, 'missing cusip'

    maturity = _parse_date(row.get('maturity_date') or row.get('maturity'))
    if maturity is None:
        return None, 'missing or unparseable maturity'
    if settle is not None and maturity <= settle:
        return None, 'already matured'

    coupon = _parse_coupon(row.get('coupon_rate')
                           if row.get('coupon_rate') is not None
                           else row.get('annualized_rate'))
    if coupon is None:
        return None, 'missing or implausible coupon'

    if not is_analyzable(coupon_type=row.get('coupon_type'),
                         is_convertible=bool(row.get('is_convertible')),
                         coupon_rate=coupon):
        return None, 'not fixed-rate (floater or convertible)'

    asset_class = (row.get('asset_class')
                   or classify_by_cusip(cusip)
                   or 'CORP_IG')
    conv = conventions_for(asset_class, coupon_rate=coupon,
                           frequency=row.get('frequency'))

    return Bond(
        cusip=cusip,
        issuer_name=(row.get('issuer_name') or '').strip(),
        coupon_rate=coupon,
        maturity=maturity,
        frequency=conv['frequency'],
        convention=conv['convention'],
        comp=conv['comp'],
        asset_class=asset_class,
        face=float(row.get('face') or 100.0),
        dated_date=_parse_date(row.get('dated_date')),
        eom=row.get('eom'),
        seniority_rank=int(row.get('seniority_rank')
                           or SENIORITY_SENIOR_UNSECURED),
        seniority_source=row.get('seniority_source') or 'default',
        is_callable=bool(row.get('is_callable')),
        call_schedule=tuple(row['call_schedule']) if row.get('call_schedule') else None,
    ), None


_SENIORITY_PATTERNS = (
    # Ordered most-specific first: "SR SECURED" must beat the bare "SR".
    (SENIORITY_JUNIOR, ('JR SUBORDINATED', 'JUNIOR SUBORDINATED', 'JR SUB',
                        'HYBRID', 'PFD', 'PREFERRED', 'CAPITAL SECURITIES')),
    (SENIORITY_SENIOR_SUB, ('SENIOR SUBORDINATED', 'SR SUBORDINATED', 'SR SUB')),
    (SENIORITY_SUB, ('SUBORDINATED', 'SUB NOTE', 'SUB DEB')),
    (SENIORITY_SENIOR_SECURED, ('1ST LIEN', 'FIRST LIEN', '2ND LIEN',
                                'SECOND LIEN', 'SR SECURED', 'SENIOR SECURED',
                                'SECURED')),
)


def infer_seniority(title_of_issue, payoff_profile=None, issuer_cat=None):
    """Infer a seniority rank from the N-PORT title text.

    Returns (rank, source). `source` is 'title' when the text actually said
    something and 'default' when it did not — and the default is senior
    unsecured, the modal case. That marker matters: a guessed seniority must
    never silently drive a rating, so the Structure gate and the report both
    key off it.
    """
    text = (title_of_issue or '').upper()
    if text:
        for rank, patterns in _SENIORITY_PATTERNS:
            if any(p in text for p in patterns):
                return rank, 'title'
        if 'SR ' in text or 'SENIOR' in text:
            return SENIORITY_SENIOR_UNSECURED, 'title'
    return SENIORITY_SENIOR_UNSECURED, 'default'
