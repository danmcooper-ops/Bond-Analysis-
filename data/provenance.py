# VENDORED VERBATIM from stock-analysis-model @ 168c17a792f28ec56af874fa2de1be5c3ae92db7
# on 2026-08-06 (data/provenance.py).
# Kernel module — keep structurally in sync; run `python tools/kernel_diff.py`.
# Local divergences: none
"""Record-level data provenance for the daily analysis pipeline.

Two halves:

1. ``ProvenanceRecorder`` — a per-run accumulator used by
   scripts/analyze_stock.py. Phase 1 records which source supplied each
   data slot (statements / market_data / beta / fx) per ticker plus any
   noteworthy events (source fallbacks, conflicts, FX failures); Phase 2
   attaches the per-ticker ``_provenance`` block to the output row and
   the run-level ``provenance`` block to the results JSON.

2. Stateless helpers for the post-run enrichment scripts (enrich_fdic,
   enrich_pipeline, enrich_xbrl), which run as separate processes and
   mutate the results JSON in place. ``attach_enrichment`` /
   ``strip_enrichment`` are overwrite-by-key so re-runs stay idempotent;
   ``append_events`` writes each script's events into its own section of
   the ``output/events_{date}.json`` sidecar (section replacement, also
   idempotent).

Events live in the sidecar, NOT the results JSON: results files are
committed daily and loaded wholesale by replay/calibrate, while the
event log is unbounded diagnostic data (it includes tickers that never
qualify for the results list).

All recorder methods swallow exceptions internally — they run inside
the per-ticker try in analyze_stock.py Phase 1, and a provenance bug
must never turn into a silently skipped ticker.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# A "daily" pipeline serving cached data older than this is worth an event.
STALE_CACHE_DAYS = 7

EVENT_TYPES = (
    'source_fallback',
    'cross_source_conflict',
    'stale_cache',
    'fx_fetch_failed',
    'enrichment_skipped',
)


def now_iso():
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def library_versions():
    """Versions of the libraries whose output lands in the results JSON."""
    versions = {'python': '%d.%d.%d' % sys.version_info[:3]}
    for name in ('yfinance', 'pandas', 'numpy'):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, '__version__', None)
        except Exception:
            versions[name] = None
    return versions


def cache_age_days(path):
    """Age of a cache file in days from its mtime, or None if missing."""
    try:
        return (time.time() - os.path.getmtime(path)) / 86400.0
    except OSError:
        return None


def make_event(etype, ticker=None, source=None, detail=None):
    return {'ts': now_iso(), 'type': etype, 'ticker': ticker,
            'source': source, 'detail': detail}


class ProvenanceRecorder:
    """Per-run provenance accumulator for analyze_stock.py."""

    def __init__(self, run_date):
        self.run_date = (run_date.isoformat()
                         if hasattr(run_date, 'isoformat') else str(run_date))
        self.started_at = now_iso()
        self.finished_at = None
        self.events = []
        self._sources = {}  # ticker -> {slot: {...}}

    def record_source(self, ticker, slot, source,
                      cache_hit=None, cache_age_days=None, **extra):
        """Record which source supplied a data slot for a ticker.

        slot is one of 'statements', 'market_data', 'beta', 'fx'.
        Extra keyword values of None are dropped so blocks stay compact.
        """
        try:
            entry = {'source': source, 'fetched_at': now_iso()}
            if cache_hit is not None:
                entry['cache_hit'] = bool(cache_hit)
            if cache_age_days is not None:
                entry['cache_age_days'] = round(float(cache_age_days), 2)
            for k, v in extra.items():
                if v is not None:
                    entry[k] = v
            self._sources.setdefault(ticker, {})[slot] = entry
        except Exception:
            pass

    def record_event(self, etype, ticker=None, source=None, detail=None):
        try:
            self.events.append(make_event(etype, ticker, source, detail))
        except Exception:
            pass

    def ticker_block(self, ticker):
        """Per-record ``_provenance`` dict for the output row."""
        try:
            return {
                'schema_version': SCHEMA_VERSION,
                'run_date': self.run_date,
                'sources': dict(self._sources.get(ticker) or {}),
            }
        except Exception:
            return {'schema_version': SCHEMA_VERSION,
                    'run_date': self.run_date, 'sources': {}}

    def event_counts(self):
        counts = {}
        for e in self.events:
            counts[e['type']] = counts.get(e['type'], 0) + 1
        return counts

    def run_block(self, results_rows=None):
        """Top-level ``provenance`` dict for the results JSON.

        statement_sources is tallied from results_rows (the rows that
        actually made it into the output) when given; the internal
        Phase-1 tally also covers screened-out tickers and would
        overstate coverage.
        """
        try:
            if self.finished_at is None:
                self.finished_at = now_iso()
            stmt_counts = {}
            if results_rows is not None:
                for r in results_rows:
                    src = r.get('data_source')
                    if src:
                        stmt_counts[src] = stmt_counts.get(src, 0) + 1
            else:
                for slots in self._sources.values():
                    src = (slots.get('statements') or {}).get('source')
                    if src:
                        stmt_counts[src] = stmt_counts.get(src, 0) + 1
            return {
                'schema_version': SCHEMA_VERSION,
                'run_started_at': self.started_at,
                'run_finished_at': self.finished_at,
                'versions': library_versions(),
                'statement_sources': stmt_counts,
                'event_counts': self.event_counts(),
                'events_file': 'events_%s.json' % self.run_date,
            }
        except Exception:
            return {'schema_version': SCHEMA_VERSION}

    def write_events(self, output_dir, section='analyze_stock'):
        try:
            append_events(output_dir, self.run_date, section, self.events)
        except Exception as e:
            print('[warn] provenance events write failed: %s' % e)


# ----------------------------------------------------------------------
# Stateless helpers for the enrichment scripts (separate processes)
# ----------------------------------------------------------------------

def enrichment_block(applied=True, cache_hit=None, cache_age_days=None, **extra):
    """Build an enrichment provenance block. None extras are dropped."""
    block = {'applied': bool(applied), 'fetched_at': now_iso()}
    if cache_hit is not None:
        block['cache_hit'] = bool(cache_hit)
    if cache_age_days is not None:
        block['cache_age_days'] = round(float(cache_age_days), 2)
    for k, v in extra.items():
        if v is not None:
            block[k] = v
    return block


def attach_enrichment(record, name, block):
    """Attach an enrichment block under _provenance.enrichments[name].

    Overwrite-by-key, so re-running an enrichment script is idempotent.
    Creates a minimal _provenance scaffold on records that predate
    provenance (old results files).
    """
    try:
        prov = record.setdefault('_provenance', {'schema_version': SCHEMA_VERSION})
        prov.setdefault('enrichments', {})[name] = block
    except Exception:
        pass


def strip_enrichment(record, name):
    """Remove an enrichment's provenance block if present. Never raises."""
    try:
        prov = record.get('_provenance')
        if isinstance(prov, dict):
            (prov.get('enrichments') or {}).pop(name, None)
    except Exception:
        pass


def append_events(output_dir, run_date, section, events):
    """Write a script's events into its section of events_{date}.json.

    The section is replaced wholesale (not appended), so re-running a
    script never duplicates its events. Other sections are preserved.
    """
    path = os.path.join(output_dir, 'events_%s.json' % run_date)
    data = {'date': str(run_date), 'schema_version': SCHEMA_VERSION, 'sections': {}}
    try:
        with open(path) as f:
            existing = json.load(f)
        if isinstance(existing, dict) and isinstance(existing.get('sections'), dict):
            data = existing
    except (OSError, ValueError):
        pass
    data['sections'][section] = {'written_at': now_iso(), 'events': list(events)}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    return path
