"""Map a bond's CUSIP to the issuer whose financials describe its credit.

THIS IS THE HIGHEST-RISK COMPONENT IN THE MODEL. A wrong match does not
degrade gracefully — it attaches some other company's leverage to a bond and
produces a confidently wrong BUY, indistinguishable from a right one. So every
resolution records HOW it was reached and how much to trust it, and the model
would rather return `unmatched` than guess between two plausible candidates.

The naive baseline is 12%: exact string equality between N-PORT issuer names
and the equity universe's company names matched 37 of the top 300 issuers.
Bond issuer names are a different dialect from company names —

    N-PORT                        equity universe
    BOEING CO/THE                 The Boeing Company
    AMAZON.COM INC                Amazon.com, Inc.
    CITIGROUP INC                 Citigroup Inc.
    PHILIP MORRIS INTL INC        Philip Morris International Inc.
    CHARTER COMM OPT LLC/CAP      Charter Communications, Inc.
    FORD MOTOR CREDIT CO LLC      Ford Motor Company

— so matching works on progressively stripped VARIANTS rather than one
canonical form: try the most specific first, fall back to the more general,
and record which rung of the ladder actually matched.

The financing-subsidiary case is deliberately allowed to fall through to the
parent (Ford Motor Credit -> Ford Motor Company). That is usually right — the
parent guarantees the debt — but it can mask structural subordination, so
those rows are flagged `issuer_is_finance_sub` rather than treated as clean.
"""

import json
import os
import re
from collections import Counter, defaultdict

from data.logging_setup import get_logger
from models.conventions import classify_by_cusip

log = get_logger('crosswalk')

# --- confidence by method --------------------------------------------------
CONF_OVERRIDE = 1.00
CONF_GOVERNMENT = 1.00
CONF_EXACT = 0.95
CONF_VARIANT = 0.90
CONF_TOKEN_MAX = 0.85
CONF_VOTE_PENALTY = 0.10      # subtracted when a CUSIP6's names disagree

# A runner-up within this of the best is a coin flip, not a match.
AMBIGUITY_MARGIN = 0.05
MIN_TOKEN_SCORE = 0.70
MIN_SHARED_TOKEN_LEN = 5

# Legal-form suffixes, stripped from the end repeatedly.
LEGAL_SUFFIXES = {
    'INC', 'INCORPORATED', 'CORP', 'CORPORATION', 'CO', 'COMPANY', 'COMPANIES',
    'COM', 'LLC', 'LLP', 'LP', 'PLC', 'LTD', 'LIMITED', 'NV', 'SA', 'AG', 'SE', 'AB',
    'AS', 'ASA', 'OYJ', 'SPA', 'BV', 'GMBH', 'KGAA', 'PTE', 'PT', 'TRUST',
    'THE', 'GROUP', 'HOLDING', 'HOLDINGS', 'HLDGS', 'HLDG',
}

# Financing vehicles: strip to reach the operating parent. This is the ladder's
# last rung and the one that needs flagging, because a bond issued by a finance
# subsidiary may sit structurally junior to the parent's own debt.
FINANCE_VEHICLE_WORDS = {
    'CAPITAL', 'CAP', 'FINANCE', 'FINANCIAL', 'FINL', 'FUNDING', 'CREDIT',
    'ESCROW', 'ISSUER', 'MERGER', 'SUB', 'ACQUISITION', 'TREASURY',
    'OVERSEAS', 'DELAWARE',
}

# Operating and geographic subsidiary markers, stripped under a STRICTER
# guard than the financing words. 'CHARTER COMMUNICATIONS OPERATING LLC' is
# Charter and 'T-MOBILE USA' is T-Mobile US, so these have to go — but
# stripping them blindly from the end destroys names that legitimately END in
# a geography: 'BANK OF AMERICA' would become 'BANK OF', and 'US STEEL' would
# lose the wrong half. Hence the two-token minimum and the preposition guard.
SUBSIDIARY_MARKERS = {
    'OPERATING', 'OPERATIONS', 'USA', 'US', 'AMERICAS', 'WORLDWIDE',
    'INTERNATIONAL', 'GLOBAL', 'ENTERPRISES',
}

# A name ending in one of these after a strip is a fragment, not a company.
_DANGLING = {'OF', 'THE', 'AND', 'FOR', 'DE', 'DU', 'LA', 'EL', 'NEW'}

# Trailing bond-specific noise: coupon/rate/date fragments that funds append
# to the issuer name ('WELLS FARGO & COM V/R 04/22/28').
_BOND_NOISE = re.compile(
    r'\b(V/?R|FLT|FLOAT|FXD|STEP|MTN|SR|JR|NOTE[S]?|BOND[S]?|DEB|CV|'
    r'\d{1,2}/\d{1,2}/\d{2,4}|\d+\.\d+%?)\b.*$')

# Roman numerals used to distinguish serial financing vehicles (Capital II).
ROMAN = {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'}

# Bond-name abbreviations. Expanded rather than removed so the token overlap
# with a full company name is preserved.
ABBREVIATIONS = {
    'INTL': 'INTERNATIONAL', 'INTERNATL': 'INTERNATIONAL',
    'COMM': 'COMMUNICATIONS', 'COMMUN': 'COMMUNICATIONS',
    'COMMS': 'COMMUNICATIONS', 'CMNTY': 'COMMUNITY',
    'FINL': 'FINANCIAL', 'FIN': 'FINANCIAL',
    'HLDGS': 'HOLDINGS', 'HLDG': 'HOLDINGS', 'HLD': 'HOLDINGS',
    'TECH': 'TECHNOLOGY', 'TECHS': 'TECHNOLOGIES',
    'SVCS': 'SERVICES', 'SVC': 'SERVICE', 'SERV': 'SERVICES',
    'MTG': 'MORTGAGE', 'PPTYS': 'PROPERTIES', 'PPTY': 'PROPERTY',
    'RLTY': 'REALTY', 'RES': 'RESOURCES', 'IND': 'INDUSTRIES',
    'INDS': 'INDUSTRIES', 'MFG': 'MANUFACTURING', 'PHARMA': 'PHARMACEUTICALS',
    'PHARM': 'PHARMACEUTICALS', 'LAB': 'LABORATORIES', 'LABS': 'LABORATORIES',
    'NATL': 'NATIONAL', 'AMER': 'AMERICAN', 'ELEC': 'ELECTRIC',
    'ENGY': 'ENERGY', 'ENRGY': 'ENERGY', 'PWR': 'POWER', 'PETE': 'PETROLEUM',
    'TRANSN': 'TRANSPORTATION', 'SYS': 'SYSTEMS', 'GP': 'GROUP',
    'STS': 'STATES', 'ST': 'STATES', 'DEPT': 'DEPARTMENT', 'STR': 'STORES',
    'OPT': 'OPERATING', 'OPER': 'OPERATING', 'ENTMT': 'ENTERTAINMENT',
    'MGMT': 'MANAGEMENT', 'INVT': 'INVESTMENT', 'INVTS': 'INVESTMENTS',
    'BK': 'BANK', 'BKS': 'BANKS', 'INS': 'INSURANCE', 'ASSUR': 'ASSURANCE',
    'AUTOMOT': 'AUTOMOTIVE', 'RY': 'RAILWAY', 'RR': 'RAILROAD',
}


# Listed securities that are not companies. The equity universe includes
# exchange-traded baby bonds and preferreds under their own tickers — 'AT&T
# Inc. 5.350% Global Notes due 2066' sits alongside 'AT&T Inc.' as ticker TBB.
# Indexed as issuers they collide with their own parent, and the collision
# guard then refuses to match AT&T at all: $2.5bn of bonds lost to a security
# listing masquerading as a company.
_SECURITY_LISTING = re.compile(
    r'\b(NOTE[S]?|DEBENTURE[S]?|PREFERRED|PFD|DEPOSITARY|SUBORDINATED|'
    r'CUMULATIVE|REDEEMABLE|SERIES\s+[A-Z]\b|DUE\s+\d{4}|\d+\.\d+\s*%)',
    re.IGNORECASE)


def is_security_listing(name):
    """True when a 'company name' is really a listed bond or preferred."""
    return bool(name and _SECURITY_LISTING.search(str(name)))


def _tokens(text):
    return [t for t in text.split() if t]


def _is_truncated_suffix(token):
    """Is this a legal suffix cut off mid-word?

    The equity model stores company_name truncated to 30 characters, so
    'Capital One Financial Corporation' arrives as 'CAPITAL ONE FINANCIAL
    CORPORATI'. That trailing fragment is not in LEGAL_SUFFIXES, nothing
    strips it, and Capital One goes unmatched despite $1.6bn of bonds and a
    company sitting right there in the index. Any token that is a proper
    prefix of a known suffix, and long enough not to be a real word, is
    treated as that suffix.
    """
    if len(token) < 4 or token in LEGAL_SUFFIXES:
        return False
    return any(suffix.startswith(token) and len(suffix) > len(token)
               for suffix in LEGAL_SUFFIXES)


def normalise_issuer_name(name):
    """Canonical form: uppercase, punctuation gone, abbreviations expanded.

    Deliberately does NOT strip legal suffixes — that is what the variant
    ladder does, one rung at a time, so a match can record how far it had to
    go.
    """
    if not name:
        return ''
    text = str(name).upper()

    # Instrument detail is stripped BEFORE the co-issuer split, not after.
    # Funds append rate and date fragments to the issuer name, and several of
    # those contain slashes: splitting first turns 'WELLS FARGO & COM
    # V/R 04/22/28' into 'WELLS FARGO AND COM V' rather than 'WELLS FARGO'.
    text = _BOND_NOISE.sub('', text)

    # 'BOEING CO/THE' is Bloomberg style for 'The Boeing Company'. Co-issuer
    # pairs ('CHARTER COMM OPT LLC/CAP CORP') take the first issuer, which is
    # the operating entity.
    if '/' in text:
        head, _, tail = text.partition('/')
        text = head if tail.strip().rstrip('.') not in ('THE',) else f'THE {head}'

    text = text.replace('&', ' AND ')
    text = re.sub(r"[^A-Z0-9 ]+", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return ' '.join(ABBREVIATIONS.get(t, t) for t in _tokens(text))


def name_variants(name):
    """Progressively stripped forms, most specific first.

    Each rung removes one more layer of corporate packaging:

        'FORD MOTOR CREDIT CO LLC'
          -> FORD MOTOR CREDIT COMPANY LLC   (normalised)
          -> FORD MOTOR CREDIT               (legal suffixes gone)
          -> FORD MOTOR                      (financing vehicle gone)

    The caller tries them in order and records which one hit, so a
    parent-level match is never mistaken for an exact one.
    """
    base = normalise_issuer_name(name)
    if not base:
        return []

    variants = [base]

    tokens = _tokens(base)
    # 'THE X' and 'X THE' are the same company.
    if tokens and tokens[0] == 'THE':
        tokens = tokens[1:]
    while tokens and tokens[-1] == 'THE':
        tokens = tokens[:-1]

    # Strip legal suffixes and trailing roman numerals from the end.
    stripped = list(tokens)
    while stripped and (stripped[-1] in LEGAL_SUFFIXES
                        or stripped[-1] in ROMAN
                        or _is_truncated_suffix(stripped[-1])):
        stripped.pop()
    if stripped and stripped != tokens:
        candidate = ' '.join(stripped)
        if candidate not in variants:
            variants.append(candidate)
    elif stripped:
        candidate = ' '.join(stripped)
        if candidate not in variants:
            variants.append(candidate)

    # Strip trailing financing-vehicle and subsidiary words to reach the
    # operating parent. Financing words may strip down to a single token;
    # subsidiary markers need two survivors, because they are far more likely
    # to be a legitimate part of the name.
    parent = list(stripped)
    changed = True
    while changed and len(parent) > 1:
        changed = False
        last = parent[-1]
        if last in FINANCE_VEHICLE_WORDS and len(parent) > 1:
            parent.pop()
            changed = True
        elif last in SUBSIDIARY_MARKERS and len(parent) > 2:
            parent.pop()
            changed = True
        if changed:
            while len(parent) > 1 and (parent[-1] in LEGAL_SUFFIXES
                                       or parent[-1] in ROMAN):
                parent.pop()

    # A name reduced to a dangling preposition is a fragment; discard the rung
    # rather than index 'BANK OF'.
    while parent and parent[-1] in _DANGLING:
        parent.pop()

    if len(parent) >= 1 and parent != stripped:
        candidate = ' '.join(parent)
        if candidate and candidate not in variants:
            variants.append(candidate)

    return variants


def is_finance_vehicle(name):
    """Did reaching the parent require stripping a financing vehicle?"""
    variants = name_variants(name)
    if len(variants) < 3:
        return False
    return _tokens(variants[-2]) != _tokens(variants[-1])


def token_score(a, b):
    """Jaccard overlap with a longest-shared-token requirement.

    The length requirement is what stops 'GENERAL MILLS' matching 'GENERAL
    MOTORS': they share only the generic token 'GENERAL', and a bare Jaccard
    of 0.33 on a two-token name is uncomfortably close to a threshold.
    """
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    shared = ta & tb
    if not shared:
        return 0.0
    if max(len(t) for t in shared) < MIN_SHARED_TOKEN_LEN:
        return 0.0
    return len(shared) / len(ta | tb)


class CusipCrosswalk:
    """Resolves CUSIP -> issuer, recording method and confidence."""

    def __init__(self, overrides_path=None, index=None):
        self.overrides = self._load_overrides(overrides_path)
        # {normalised name: key}, plus a token index for fuzzy fallback.
        self.exact = {}
        # Variants claimed by more than one company. Two different issuers
        # reducing to the same string is not a match, it is a collision, and
        # resolving it by whichever was indexed first is exactly the coin flip
        # the ambiguity guard exists to prevent.
        self.colliding = set()
        self.by_token = defaultdict(set)
        self._names = {}
        if index:
            self.build_index(index)

    @staticmethod
    def _load_overrides(path):
        path = path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'cusip_issuer_overrides.json')
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding='utf-8') as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            log.warning('Could not read overrides at %s', path)
            return {}
        # A null value is a DELIBERATE non-match: a documented decision that
        # this issuer has no usable fundamentals (supranationals, mostly), so
        # it must stay unresolved rather than be guessed at by the heuristics.
        return {k.strip().upper(): v for k, v in raw.items()
                if not k.startswith('_')}

    def build_index(self, entries):
        """entries: iterable of (key, display_name).

        Every variant of every known company name is indexed, so a bond name
        that is already stripped ('ORACLE') finds a company name that is not
        ('Oracle Corporation') without the caller stripping first.
        """
        skipped = 0
        for key, name in entries:
            if not name:
                continue
            if is_security_listing(name):
                # A listed bond or preferred, not an issuer. Indexing it
                # collides with its own parent and costs the parent its match.
                skipped += 1
                continue
            self._names[key] = name
            for variant in name_variants(name):
                existing = self.exact.get(variant)
                if existing is not None and existing != key:
                    self.colliding.add(variant)
                self.exact.setdefault(variant, key)
                for token in set(_tokens(variant)):
                    if len(token) >= MIN_SHARED_TOKEN_LEN:
                        self.by_token[token].add(key)
        if skipped:
            log.info('Skipped %d listed securities (bonds and preferreds '
                     'carrying their own ticker) that are not issuers', skipped)
        if self.colliding:
            log.info('%d name variants are claimed by more than one company '
                     'and will never match', len(self.colliding))
        log.info('Crosswalk index: %d companies, %d name variants',
                 len(self._names), len(self.exact))

    # -- resolution ---------------------------------------------------------

    def _match_name(self, name):
        """Try the variant ladder, then a guarded token match.

        Returns (key, confidence, method) or (None, 0.0, reason).
        """
        variants = name_variants(name)
        for depth, variant in enumerate(variants):
            if variant in self.colliding:
                return None, 0.0, 'ambiguous_variant'
            key = self.exact.get(variant)
            if key:
                # Depth 0 is the name as written; deeper rungs matched only
                # after stripping, which is a weaker claim.
                conf = CONF_EXACT if depth == 0 else CONF_VARIANT
                return key, conf, f'name_variant_{depth}'

        # Token fallback, over candidates sharing a substantial token.
        target = variants[-1] if variants else ''
        if not target:
            return None, 0.0, 'empty_name'
        candidates = set()
        for token in set(_tokens(target)):
            if len(token) >= MIN_SHARED_TOKEN_LEN:
                candidates |= self.by_token.get(token, set())
        if not candidates:
            return None, 0.0, 'no_candidates'

        scored = sorted(
            ((token_score(target, normalise_issuer_name(self._names[k])), k)
             for k in candidates), reverse=True)
        best_score, best_key = scored[0]
        if best_score < MIN_TOKEN_SCORE:
            return None, 0.0, 'below_token_threshold'
        # Never choose between two close candidates. A coin flip here attaches
        # the wrong company's balance sheet to a bond.
        if len(scored) > 1 and (best_score - scored[1][0]) < AMBIGUITY_MARGIN:
            return None, 0.0, 'ambiguous'
        return best_key, min(best_score, CONF_TOKEN_MAX), 'token_overlap'

    def resolve(self, cusip, issuer_names, held_by_name=None):
        """Resolve one CUSIP6 to an issuer key.

        Args:
            cusip: full CUSIP or 6-character issuer prefix.
            issuer_names: every name any fund used for this CUSIP6.
            held_by_name: optional {name: held_usd} for the consistency vote.

        Returns a dict with key, confidence, method, matched_name, candidates.
        """
        prefix = (cusip or '')[:6].upper()
        blank = {'key': None, 'confidence': 0.0, 'matched_name': None,
                 'is_finance_sub': False, 'candidates': []}

        if prefix in self.overrides:
            override = self.overrides[prefix]
            if override is None:
                return {**blank, 'method': 'override_no_fundamentals'}
            return {**blank, 'key': override, 'confidence': CONF_OVERRIDE,
                    'method': 'override', 'matched_name': override}

        asset_class = classify_by_cusip(cusip)
        if asset_class:
            return {**blank, 'key': None, 'confidence': CONF_GOVERNMENT,
                    'method': 'government_prefix', 'asset_class': asset_class}

        names = [n for n in (issuer_names or []) if n]
        if not names:
            return {**blank, 'method': 'no_name'}

        # Every spelling gets a vote, weighted by the dollars behind it — the
        # same issuer appears under five or ten spellings across funds, and
        # their agreement is real information.
        weights = held_by_name or {}
        tally = defaultdict(float)
        methods, matched_names = {}, {}
        for name in set(names):
            key, conf, method = self._match_name(name)
            if key is None:
                continue
            weight = weights.get(name, 1.0) or 1.0
            tally[key] += weight * conf
            if key not in methods or conf > methods[key][0]:
                methods[key] = (conf, method)
                matched_names[key] = name

        if not tally:
            return {**blank, 'method': 'unmatched',
                    'candidates': sorted(set(names))[:4]}

        ranked = sorted(tally.items(), key=lambda kv: kv[1], reverse=True)
        best_key, best_weight = ranked[0]
        confidence, method = methods[best_key]

        # Names under one CUSIP6 that resolve to DIFFERENT issuers mean at
        # least one is wrong. Penalise rather than discard: the weighted
        # winner is usually right, but it should not read as certain.
        if len(ranked) > 1:
            confidence = max(0.0, confidence - CONF_VOTE_PENALTY)
            method += '+contested'

        return {
            'key': best_key,
            'confidence': round(confidence, 3),
            'method': method,
            'matched_name': matched_names.get(best_key),
            'is_finance_sub': is_finance_vehicle(matched_names.get(best_key, '')),
            'candidates': [k for k, _ in ranked[:4]],
        }

    def resolve_all(self, groups):
        """Resolve many CUSIP6 groups at once.

        groups: {cusip6: {'names': [...], 'held_by_name': {...}}}
        Returns {cusip6: resolution}.
        """
        out = {}
        for prefix, payload in groups.items():
            out[prefix] = self.resolve(prefix, payload.get('names'),
                                       payload.get('held_by_name'))
        methods = Counter(r.get('method', '?') for r in out.values())
        log.info('Resolved %d CUSIP6 prefixes: %s', len(out),
                 ', '.join(f'{k}={v}' for k, v in methods.most_common()))
        return out
