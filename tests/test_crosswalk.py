"""CUSIP -> issuer resolution.

The negative tests matter more than the positive ones here. A missed match
costs a bond its credit gates, which the model handles; a WRONG match attaches
another company's balance sheet to a bond and produces a confidently wrong BUY
that looks exactly like a right one. So this file spends most of its effort on
what must NOT match.
"""

import pytest

from data.cusip_crosswalk import (AMBIGUITY_MARGIN, MIN_TOKEN_SCORE,
                                  is_security_listing,
                                  CusipCrosswalk,
                                  is_finance_vehicle, name_variants,
                                  normalise_issuer_name, token_score)

# A small index standing in for the fundamentals universe.
INDEX = [
    ('BAC', 'Bank of America Corporation'),
    ('JPM', 'JPMorgan Chase & Co.'),
    ('GS', 'The Goldman Sachs Group, Inc.'),
    ('F', 'Ford Motor Company'),
    ('GM', 'General Motors Company'),
    ('GIS', 'General Mills, Inc.'),
    ('BA', 'Boeing Company (The)'),
    ('AMZN', 'Amazon.com, Inc.'),
    ('KO', 'The Coca-Cola Company'),
    ('CHTR', 'Charter Communications, Inc.'),
    ('TMUS', 'T-Mobile US, Inc.'),
    ('PM', 'Philip Morris International Inc.'),
    ('COF', 'CAPITAL ONE FINANCIAL CORPORATI'),   # truncated, as stored
    ('X', 'United States Steel Corporation'),
    ('CACC', 'Credit Acceptance Corporation'),
]


@pytest.fixture
def crosswalk():
    return CusipCrosswalk(index=INDEX)


def resolve(crosswalk, name, cusip='037833AA0'):
    return crosswalk.resolve(cusip, [name])


# ---------------------------------------------------------------------------
# THE NEGATIVE TESTS
# ---------------------------------------------------------------------------

def test_general_mills_must_not_match_general_motors(crosswalk):
    """The single most important test in this file. Two large, real issuers
    sharing one generic token. A bare Jaccard on two-token names puts them at
    0.33 — uncomfortably close to a threshold — so the matcher additionally
    requires a shared token of real length, and 'GENERAL' alone cannot carry a
    match."""
    mills = resolve(crosswalk, 'GENERAL MILLS INC')
    motors = resolve(crosswalk, 'GENERAL MOTORS CO')
    assert mills['key'] == 'GIS'
    assert motors['key'] == 'GM'
    assert mills['key'] != motors['key']


def test_a_generic_shared_token_alone_never_matches(crosswalk):
    """One shared token out of three or four is far below the match threshold,
    so an unrelated company sharing a common word cannot resolve."""
    assert token_score('GENERAL FOODS', 'GENERAL MILLS') < MIN_TOKEN_SCORE
    assert token_score('GENERAL MOTORS', 'GENERAL MILLS') < MIN_TOKEN_SCORE
    assert resolve(crosswalk, 'GENERAL DYNAMICS CORP')['key'] is None


def test_close_candidates_return_unmatched_rather_than_a_coin_flip():
    """When two candidates score within the ambiguity margin, guessing has a
    coin-flip chance of attaching the wrong balance sheet. Refusing costs the
    bond its credit gates, which is recoverable; guessing is not."""
    ambiguous = CusipCrosswalk(index=[
        ('AAA', 'Pacific Northern Energy Corporation'),
        ('BBB', 'Pacific Northern Energy Company'),
    ])
    out = ambiguous.resolve('123456', ['PACIFIC NORTHERN ENERGY HOLDINGS'])
    assert out['key'] is None
    assert out['method'] in ('unmatched', 'ambiguous', 'ambiguous_variant')


def test_two_companies_sharing_a_name_variant_never_match():
    """A collision in the index is not a match. Resolving it by whichever
    company happened to be indexed first is exactly the coin flip the
    ambiguity guard exists to prevent, just reached by a different path."""
    colliding = CusipCrosswalk(index=[
        ('AAA', 'Sterling Industries Inc.'),
        ('BBB', 'Sterling Industries Corporation'),
    ])
    assert 'STERLING INDUSTRIES' in colliding.colliding
    out = colliding.resolve('123456', ['STERLING INDUSTRIES LLC'])
    assert out['key'] is None


def test_a_name_reduced_to_a_fragment_is_not_indexed():
    """Stripping geography from the end must not turn 'BANK OF AMERICA' into
    'BANK OF' and then match anything beginning with 'BANK'."""
    assert 'BANK OF' not in name_variants('BANK OF AMERICA CORP')
    assert name_variants('BANK OF AMERICA CORP')[-1] == 'BANK OF AMERICA'


@pytest.mark.parametrize('name,expected_tail', [
    ('US STEEL CORP', 'US STEEL'),
    ('AMERICAN EXPRESS CO', 'AMERICAN EXPRESS'),
    ('CREDIT ACCEPTANCE CORP', 'CREDIT ACCEPTANCE'),
    ('CAPITAL ONE FINANCIAL CORP', 'CAPITAL ONE'),
])
def test_stripping_does_not_destroy_names_that_contain_stripped_words(
        name, expected_tail):
    """'CREDIT', 'CAPITAL', 'US' and 'AMERICAN' are all strippable in some
    positions and load-bearing in others."""
    assert expected_tail in name_variants(name)


def test_unknown_issuer_returns_unmatched_with_its_candidates(crosswalk):
    out = resolve(crosswalk, 'SOME PRIVATE HOLDCO LLC')
    assert out['key'] is None
    assert out['confidence'] == 0.0
    assert out['candidates']


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('raw,expected', [
    ('BOEING CO/THE', 'THE BOEING CO'),
    ('AMAZON.COM INC', 'AMAZON COM INC'),
    ('WELLS FARGO & COM V/R 04/22/28', 'WELLS FARGO AND COM'),
    ('JPMORGAN CHASE & CO', 'JPMORGAN CHASE AND CO'),
    ('PHILIP MORRIS INTL INC', 'PHILIP MORRIS INTERNATIONAL INC'),
    ('CHARTER COMM OPT LLC', 'CHARTER COMMUNICATIONS OPERATING LLC'),
])
def test_normalisation_expands_the_bond_dialect(raw, expected):
    assert normalise_issuer_name(raw) == expected


def test_instrument_detail_is_stripped_before_the_coissuer_split():
    """Funds append rate and date fragments containing slashes. Splitting on
    '/' first turns 'WELLS FARGO & COM V/R 04/22/28' into 'WELLS FARGO AND
    COM V' instead of a company name."""
    # Split-first would yield 'WELLS FARGO AND COM V', keeping the stray 'V'
    # and blocking the reduction to a company name.
    assert normalise_issuer_name('WELLS FARGO & COM V/R 04/22/28') == \
        'WELLS FARGO AND COM'
    assert name_variants('WELLS FARGO & COM V/R 04/22/28')[-1] == 'WELLS FARGO' 


def test_truncated_legal_suffixes_are_recognised():
    """The equity model stores company_name truncated to 30 characters, so
    'Capital One Financial Corporation' arrives as '...CORPORATI'. Left
    unhandled, Capital One goes unmatched with a company sitting in the index."""
    assert 'CAPITAL ONE FINANCIAL' in name_variants('CAPITAL ONE FINANCIAL CORPORATI')


def test_variants_run_specific_to_general():
    assert name_variants('FORD MOTOR CREDIT COMPANY LLC') == [
        'FORD MOTOR CREDIT COMPANY LLC', 'FORD MOTOR CREDIT', 'FORD MOTOR']


def test_empty_and_junk_names():
    assert name_variants('') == []
    assert name_variants(None) == []
    assert normalise_issuer_name('   ') == ''


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name,key', [
    ('BANK OF AMERICA CORP', 'BAC'),
    ('Bank of America Corp.', 'BAC'),
    ('JPMORGAN CHASE & CO', 'JPM'),
    ('GOLDMAN SACHS GROUP INC (THE)', 'GS'),
    ('BOEING CO/THE', 'BA'),
    ('AMAZON.COM INC', 'AMZN'),
    ('COCA-COLA CO/THE', 'KO'),
    ('PHILIP MORRIS INTL INC', 'PM'),
    ('CAPITAL ONE FINANCIAL CO', 'COF'),
])
def test_real_bond_names_resolve(crosswalk, name, key):
    assert resolve(crosswalk, name)['key'] == key


def test_subsidiaries_resolve_to_the_parent(crosswalk):
    assert resolve(crosswalk, 'CHARTER COMMUNICATIONS OPERATING, LLC')['key'] == 'CHTR'
    assert resolve(crosswalk, 'T-MOBILE USA INC')['key'] == 'TMUS'


def test_finance_subsidiaries_resolve_but_are_flagged(crosswalk):
    """Ford Motor Credit's debt is Ford's credit — usually. It can also sit
    structurally junior to the parent's own debt, so the row is marked rather
    than treated as a clean parent match."""
    out = resolve(crosswalk, 'FORD MOTOR CREDIT CO LLC')
    assert out['key'] == 'F'
    assert out['is_finance_sub'] is True
    assert is_finance_vehicle('FORD MOTOR CREDIT CO LLC')
    assert not is_finance_vehicle('BANK OF AMERICA CORP')


def test_match_depth_is_recorded_in_the_method(crosswalk):
    """A parent-level match is a weaker claim than an exact one and must not
    be reported as the same thing."""
    exact = resolve(crosswalk, 'Bank of America Corporation')
    parent = resolve(crosswalk, 'FORD MOTOR CREDIT CO LLC')
    assert exact['method'] == 'name_variant_0'
    assert exact['confidence'] > parent['confidence']
    assert parent['method'].startswith('name_variant_')


# ---------------------------------------------------------------------------
# Government paper and overrides
# ---------------------------------------------------------------------------

def test_treasury_cusips_bypass_name_matching_entirely(crosswalk):
    out = crosswalk.resolve('912828XY5', ['US TREASURY N/B'])
    assert out['method'] == 'government_prefix'
    assert out['confidence'] == 1.0
    assert out['asset_class'] == 'TREASURY'


def test_agency_cusips_are_recognised(crosswalk):
    assert crosswalk.resolve('3135G0X24', ['FANNIE MAE'])['asset_class'] == 'AGENCY'


def test_an_override_wins_and_carries_full_confidence():
    xw = CusipCrosswalk(index=INDEX)
    xw.overrides['404119'] = 'HCA'
    out = xw.resolve('404119AB1', ['HCA INC'])
    assert out['key'] == 'HCA'
    assert out['confidence'] == 1.0
    assert out['method'] == 'override'


def test_a_null_override_means_deliberately_unresolved():
    """Supranationals file no US financials. A null entry documents that as a
    decision rather than leaving the heuristics to guess at 'INTERNATIONAL
    BANK FOR RECONSTRUCTION AND DEVELOPMENT'."""
    xw = CusipCrosswalk(index=INDEX)
    xw.overrides['459058'] = None
    out = xw.resolve('459058KH1', ['INTERNATIONAL BANK FOR RECONSTRUCTION'])
    assert out['key'] is None
    assert out['method'] == 'override_no_fundamentals'


# ---------------------------------------------------------------------------
# The consistency vote
# ---------------------------------------------------------------------------

def test_spellings_are_weighted_by_the_dollars_behind_them(crosswalk):
    """The same issuer appears under many spellings across funds. A spelling
    backed by $8bn is better evidence than one backed by $2m."""
    out = crosswalk.resolve(
        '060505AA1',
        ['BANK OF AMERICA CORP', 'SOME UNRELATED HOLDCO'],
        held_by_name={'BANK OF AMERICA CORP': 8e9, 'SOME UNRELATED HOLDCO': 2e6})
    assert out['key'] == 'BAC'


def test_contested_prefixes_lose_confidence(crosswalk):
    """Names under one CUSIP6 resolving to DIFFERENT issuers means at least
    one is wrong. The weighted winner is usually right but must not read as
    certain."""
    out = crosswalk.resolve('123456', ['BANK OF AMERICA CORP',
                                       'GENERAL MILLS INC'],
                            held_by_name={'BANK OF AMERICA CORP': 5e9,
                                          'GENERAL MILLS INC': 1e9})
    assert out['key'] == 'BAC'
    assert '+contested' in out['method']
    assert out['confidence'] < 0.95


def test_resolve_all_returns_one_entry_per_prefix(crosswalk):
    out = crosswalk.resolve_all({
        '060505': {'names': ['BANK OF AMERICA CORP'], 'held_by_name': {}},
        '345397': {'names': ['FORD MOTOR CREDIT CO LLC'], 'held_by_name': {}},
    })
    assert out['060505']['key'] == 'BAC'
    assert out['345397']['key'] == 'F'


def test_no_names_is_not_an_error(crosswalk):
    assert crosswalk.resolve('123456', [])['method'] == 'no_name'
    assert crosswalk.resolve('123456', None)['method'] == 'no_name'


def test_listed_bonds_and_preferreds_are_not_indexed_as_issuers():
    """The equity universe carries exchange-traded baby bonds under their own
    tickers: 'AT&T Inc. 5.350% Global Notes' sits beside 'AT&T Inc.' as TBB.
    Indexed as an issuer it collides with its own parent, and the collision
    guard then refuses to match AT&T at all — $2.5bn of bonds lost to a
    security listing masquerading as a company."""
    xw = CusipCrosswalk(index=[
        ('T', 'AT&T Inc.'),
        ('TBB', 'AT&T Inc. 5.350% Global Notes d'),
    ])
    assert is_security_listing('AT&T Inc. 5.350% Global Notes d')
    assert not is_security_listing('AT&T Inc.')
    assert 'AT AND T' not in xw.colliding
    out = xw.resolve('00206R', ['AT&T INC'])
    assert out['key'] == 'T'
    assert out['confidence'] == 0.95


@pytest.mark.parametrize('name', [
    'Acme Corp 6.25% Notes due 2049',
    'Acme Series A Preferred',
    'Acme Cumulative Redeemable Pfd',
    'Acme Depositary Shares',
])
def test_security_listing_patterns(name):
    assert is_security_listing(name)


@pytest.mark.parametrize('name', [
    'Amazon.com, Inc.', 'The Coca-Cola Company', 'Ford Motor Company',
    'Bank of America Corporation', 'Charter Communications, Inc.',
])
def test_real_company_names_are_not_mistaken_for_securities(name):
    assert not is_security_listing(name)
