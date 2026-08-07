"""Issuer financials for the credit scorecard, from whatever source has them.

Kept at arm's length from the equity model on purpose: this reads that
model's OUTPUT DIRECTORY, never its code. Neither repo imports the other, so
an equity refactor cannot break bond ratings, and the bond model still runs —
thinner, never broken — when the equity output is absent.

COVERAGE IS THE BINDING CONSTRAINT, NOT NAME MATCHING. Measured on 2026Q2
against $2,502bn of held corporate bonds across 7,350 issuers:

    43.8% of held value   matched to an issuer WITH fundamentals
    19.5%                 matched to a real SEC filer, no fundamentals yet
    36.7%                 name unresolved

The middle band is what a SEC XBRL backend would recover — those issuers file,
we simply have not fetched them. The bottom band is the hard residual: private
issuers, foreign banks, and financing structures with no US filing behind them.

A row with no fundamentals is not a failure. Its credit gates go structurally
inapplicable, the Credit category drops from its composite, and the remaining
categories renormalise — the same mechanism that lets Treasuries be rated. The
bond is still scored on valuation, rates, structure and liquidity; it simply
cannot be scored on credit, and the report says so.
"""

import glob
import json
import os
from datetime import date, datetime

from data.logging_setup import get_logger

log = get_logger('issuer_fundamentals')

# Fields the credit scorecard and the Credit gates read. Names on the left are
# what this model uses; the equity snapshot happens to use the same ones.
WANTED_FIELDS = (
    'int_cov', 'nd_ebitda', 'de', 'total_debt', 'net_debt', 'fcf',
    'altman_z', 'altman_z_zone', 'piotroski', 'revenue',
    'debt_maturity_wall_yrs', 'cet1_ratio', 'npl_ratio', 'sector',
    'company_name', 'mcap',
)


class EquitySnapshotBackend:
    """Reads the sibling equity model's newest results_*.json.

    Zero network, and no dependency on that model's code — just its output
    convention, which has been stable across every snapshot in its history.
    """

    def __init__(self, snapshot_dir=None, as_of=None):
        """as_of pins the snapshot to a point in time.

        POINT-IN-TIME MATTERS HERE, it is not hygiene. The divergence signal
        compares a bond's market-implied credit against its fundamentals, and
        N-PORT marks are ~98 days old. Reading TODAY's balance sheet against
        an April price does not measure divergence, it measures our own data
        lag — and the stale-risk guard correctly suppressed the signal for
        every bond in the universe until this existed. With 69 historical
        equity snapshots available, the fundamentals can simply be read as
        they stood when the price was struck.
        """
        self.as_of = as_of
        self.snapshot_dir = snapshot_dir or os.environ.get(
            'EQUITY_SNAPSHOT_DIR',
            os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), 'output'))
        self._rows = None
        self._as_of = None
        self._schema = set()

    @property
    def available(self):
        return bool(self._snapshot_path())

    def _snapshot_path(self):
        if not os.path.isdir(self.snapshot_dir):
            return None
        paths = sorted(glob.glob(os.path.join(self.snapshot_dir,
                                              'results_*.json')))
        if not paths:
            return None
        if self.as_of is None:
            return paths[-1]
        # Newest snapshot at or before as_of. Never look forward: a snapshot
        # published after the mark date would put information into the model
        # that nobody had when the price was struck.
        stamp = self.as_of.isoformat()
        eligible = [p for p in paths
                    if os.path.basename(p)[8:18] <= stamp]
        if not eligible:
            log.warning('No equity snapshot at or before %s; oldest is %s',
                        stamp, os.path.basename(paths[0])[8:18])
            return None
        return eligible[-1]

    def load(self):
        """{ticker: fundamentals}. Empty when the equity output is absent."""
        if self._rows is not None:
            return self._rows

        path = self._snapshot_path()
        if not path:
            log.warning('No equity snapshot under %s — every issuer will '
                        'score without credit gates', self.snapshot_dir)
            self._rows = {}
            return self._rows

        try:
            with open(path, encoding='utf-8') as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            log.error('Could not read %s: %s', path, exc)
            self._rows = {}
            return self._rows

        rows = payload.get('results', payload) or []
        self._as_of = payload.get('date') or os.path.basename(path)[8:18]

        # Which fields this VINTAGE carries at all. Older snapshots predate
        # fields the equity model added later — the 2026-04 snapshot has no
        # total_debt, debt_maturity_wall_yrs, cet1_ratio or npl_ratio. A field
        # the data never carried is structurally unmeasurable for that vintage,
        # not a bad issuer, so the gates that read it must go INAPPLICABLE
        # rather than score every matched issuer zero.
        self._schema = {f for f in WANTED_FIELDS
                        if any(r.get(f) is not None for r in rows)}
        missing = sorted(set(WANTED_FIELDS) - self._schema)
        if missing:
            log.info('Snapshot %s predates these fields: %s',
                     os.path.basename(path), ', '.join(missing))

        out = {}
        for row in rows:
            ticker = (row.get('ticker') or '').strip().upper()
            if not ticker:
                continue
            entry = {k: row.get(k) for k in WANTED_FIELDS}
            entry['_fundamentals_source'] = 'equity_snapshot'
            entry['_fundamentals_asof'] = self._as_of
            out[ticker] = entry

        log.info('Equity snapshot %s: %d issuers with fundamentals',
                 os.path.basename(path), len(out))
        self._rows = out
        return out

    def names(self):
        """[(ticker, company_name)] for the crosswalk index."""
        return [(t, v.get('company_name')) for t, v in self.load().items()
                if v.get('company_name')]

    def schema_fields(self):
        self.load()
        return set(self._schema)


class SECXBRLBackend:
    """Placeholder for issuers that file with the SEC but are outside the
    equity universe — 19.5% of held value on 2026Q2.

    Deliberately NOT half-implemented. Fetching company facts per CIK and
    deriving coverage, leverage and Altman-Z from raw XBRL is the equity
    model's `sec_xbrl_client` plus a slice of its analytics, and a partial
    version would silently produce fundamentals of a different quality from
    the equity path while looking identical downstream. Until it exists these
    issuers correctly report no fundamentals and lose their credit gates.
    """

    available = False

    def load(self):
        return {}

    def names(self):
        return []

    def schema_fields(self):
        return set()


class IssuerFundamentals:
    """Resolves an issuer key to fundamentals, trying each backend in order."""

    def __init__(self, backends=None, as_of=None):
        self.backends = backends or [EquitySnapshotBackend(as_of=as_of),
                                     SECXBRLBackend()]
        self._merged = None

    def load(self):
        if self._merged is None:
            merged = {}
            for backend in self.backends:
                for key, value in backend.load().items():
                    merged.setdefault(key, value)
            self._merged = merged
        return self._merged

    def names(self):
        seen, out = set(), []
        for backend in self.backends:
            for key, name in backend.names():
                if key not in seen:
                    seen.add(key)
                    out.append((key, name))
        return out

    def schema_fields(self):
        """Union of fields any backend's data vintage actually carries."""
        fields = set()
        for backend in self.backends:
            fields |= backend.schema_fields()
        return fields

    def get(self, key):
        return self.load().get((key or '').strip().upper())

    def age_days(self, entry, as_of):
        """Days since the fundamentals were struck. None if undatable."""
        raw = (entry or {}).get('_fundamentals_asof')
        if not raw:
            return None
        try:
            when = datetime.strptime(str(raw)[:10], '%Y-%m-%d').date()
        except ValueError:
            return None
        return (as_of - when).days

    def attach(self, row, resolution, as_of):
        """Stamp issuer fields onto a bond row from a crosswalk resolution.

        Every field is prefixed `issuer_` so a bond's own attributes can never
        be confused with its issuer's — `int_cov` belongs to a company,
        `z_spread` to a bond, and a flat namespace would eventually collide.
        """
        row['cusip_match_method'] = resolution.get('method')
        row['cusip_match_confidence'] = resolution.get('confidence', 0.0)
        row['cusip_match_name'] = resolution.get('matched_name')
        row['issuer_is_finance_sub'] = bool(resolution.get('is_finance_sub'))

        key = resolution.get('key')
        if not key:
            # Resolved against the full SEC filer list but not the
            # fundamentals index: a real filer whose financials we simply have
            # not fetched. Distinct from an unidentifiable issuer, and the
            # coverage report keeps them apart.
            if resolution.get('filer_key'):
                row['issuer_ticker'] = resolution['filer_key']
                row['_fundamentals_missing'] = True
            return False
        entry = self.get(key)
        if not entry:
            # Matched a name but hold no financials for it — a real and
            # distinct outcome from "could not identify the issuer", and the
            # coverage report separates the two.
            row['issuer_ticker'] = key
            row['_fundamentals_missing'] = True
            return False

        row['issuer_ticker'] = key
        # Which issuer fields this data vintage carries, so a gate reading a
        # field the snapshot never had can mask itself instead of scoring 0.
        row['_issuer_fields'] = tuple(sorted(self.schema_fields()))
        row['issuer_sector'] = entry.get('sector')
        row['issuer_company_name'] = entry.get('company_name')
        row['_fundamentals_source'] = entry.get('_fundamentals_source')
        row['_fundamentals_asof'] = entry.get('_fundamentals_asof')
        row['_fundamentals_age_days'] = self.age_days(entry, as_of)
        # The gate layer keys applicability off issuer_cik being present;
        # the equity snapshot has no CIK, so the ticker stands in as the
        # issuer identity.
        row['issuer_cik'] = key

        for field in ('int_cov', 'nd_ebitda', 'altman_z', 'altman_z_zone',
                      'piotroski', 'revenue', 'debt_maturity_wall_yrs',
                      'cet1_ratio', 'npl_ratio', 'mcap', 'fcf'):
            row[f'issuer_{field}'] = entry.get(field)

        # FCF-to-debt is the scorecard's cash-generation factor and is derived
        # rather than stored. Guard the denominator: a debt-free issuer is not
        # infinitely creditworthy, it is simply not a leverage story.
        fcf, debt = entry.get('fcf'), entry.get('total_debt')
        row['issuer_fcf_to_debt'] = (fcf / debt if fcf is not None
                                     and debt not in (None, 0) and debt > 0
                                     else None)
        row['issuer_total_debt'] = debt
        return True
