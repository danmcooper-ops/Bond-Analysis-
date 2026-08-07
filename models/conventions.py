"""Market conventions per asset class.

Floating-rate notes and convertibles are DELIBERATELY absent from this table.
A floater's yield to maturity is undefined without projecting a forward index,
and a convertible's price is dominated by an equity option this model has no
way to value. Rather than feed either one through the fixed-rate machinery and
produce a confident-looking wrong number, the pipeline marks them
structurally inapplicable (`_appl_fixed_coupon`) and says so on the report.
"""

from models.daycount import ACT_360, ACT_ACT, ACT_365F, D30_360
from models.discount import MONEY_MARKET_MAX_YEARS

COMP_SEMIANNUAL = 'semiannual'
COMP_SIMPLE = 'simple'

CONVENTIONS = {
    # Bills are discount instruments: no coupon, money-market compounding.
    'TREASURY_BILL': dict(frequency=0, convention=ACT_360, comp=COMP_SIMPLE),
    'TREASURY':      dict(frequency=2, convention=ACT_ACT, comp=COMP_SEMIANNUAL),
    'AGENCY':        dict(frequency=2, convention=D30_360, comp=COMP_SEMIANNUAL),
    'CORP_IG':       dict(frequency=2, convention=D30_360, comp=COMP_SEMIANNUAL),
    'CORP_HY':       dict(frequency=2, convention=D30_360, comp=COMP_SEMIANNUAL),
    'MUNI':          dict(frequency=2, convention=D30_360, comp=COMP_SEMIANNUAL),
}

DEFAULT_CONVENTION = dict(frequency=2, convention=D30_360,
                          comp=COMP_SEMIANNUAL)

# CUSIP issuer prefixes that identify US government paper without any name
# matching at all. These are stable and authoritative, which is why the
# crosswalk consults them before it ever tries to match an issuer name.
TREASURY_CUSIP_PREFIXES = {
    '912810': 'TREASURY',        # bonds
    '912828': 'TREASURY',        # notes
    '91282C': 'TREASURY',        # notes (current series)
    '912796': 'TREASURY_BILL',   # bills
    '912820': 'TREASURY',        # STRIPS
    '912803': 'TREASURY',        # STRIPS principal
    '9128 ': 'TREASURY',         # defensive: malformed feed rows
}

AGENCY_CUSIP_PREFIX_ROOTS = ('3133', '3134', '3135', '3136', '3137')


def classify_by_cusip(cusip):
    """Return an asset class from the CUSIP alone, or None.

    Only government and agency paper can be identified this way; a corporate
    CUSIP prefix identifies the ISSUER, not the credit quality, so IG vs HY
    still has to come from the credit model.
    """
    if not cusip or len(cusip) < 6:
        return None
    prefix6 = cusip[:6].upper()
    if prefix6 in TREASURY_CUSIP_PREFIXES:
        return TREASURY_CUSIP_PREFIXES[prefix6]
    if prefix6[:4] in AGENCY_CUSIP_PREFIX_ROOTS:
        return 'AGENCY'
    return None


def conventions_for(asset_class, coupon_rate=None, frequency=None,
                    years_to_maturity=None):
    """Return {frequency, convention, comp} for an asset class.

    An explicit `frequency` from the source data wins over the table — issuers
    do occasionally pay annually or quarterly.

    A ZERO COUPON SPLITS BY TENOR, NOT BY COUPON. The obvious rule — "no
    coupon means frequency 0 and simple compounding" — is wrong for long
    zeros. A 10-year STRIP is conventionally quoted on a semiannual
    bond-equivalent basis, and forcing it to frequency 0 sends it down the
    single-period discount path where it prices as though it matured in six
    months. Under a year, money-market simple is right; beyond that,
    bond-equivalent semiannual is, and a zero-coupon schedule at frequency 2
    generates the correct number of periods for the ordinary machinery.
    """
    base = dict(CONVENTIONS.get(asset_class or '', DEFAULT_CONVENTION))
    if frequency is not None:
        base['frequency'] = frequency

    if coupon_rate is not None and coupon_rate == 0:
        short = (years_to_maturity is not None
                 and years_to_maturity <= MONEY_MARKET_MAX_YEARS)
        # A declared bill is always money-market, whatever its remaining life.
        if short or asset_class == 'TREASURY_BILL':
            base['frequency'] = 0
            base['comp'] = COMP_SIMPLE
        else:
            base['frequency'] = base['frequency'] or 2
            base['comp'] = COMP_SEMIANNUAL
    return base


def is_analyzable(coupon_type=None, is_convertible=False, coupon_rate=None):
    """Can the fixed-rate machinery honestly price this?

    False for floaters (no forward index projection), convertibles (equity
    option dominates), and anything with an unusable coupon.
    """
    if is_convertible:
        return False
    if coupon_type and str(coupon_type).strip().lower() not in ('fixed', 'none', ''):
        return False
    if coupon_rate is not None and (coupon_rate < 0 or coupon_rate > 0.40):
        return False
    return True
