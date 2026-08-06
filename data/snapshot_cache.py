# VENDORED VERBATIM from stock-analysis-model @ 168c17a792f28ec56af874fa2de1be5c3ae92db7
# on 2026-08-06 (data/snapshot_cache.py).
# Kernel module — keep structurally in sync; run `python tools/kernel_diff.py`.
# Local divergences: none yet. Cache keys are bare strings, so CUSIPs work unchanged; the financials_{date}.json filename prefix is the only equity-flavoured detail and is left alone until something actually needs renaming.
# data/snapshot_cache.py
"""Persistent disk cache for yfinance financial snapshots.

Stores financial data (balance sheets, income statements, cash flows, info dicts)
as dated JSON files on disk so that past analyses can be replayed without
re-fetching from yfinance.

Directory structure:
    {cache_dir}/{TICKER}/financials_{YYYY-MM-DD}.json
"""

import os
import json
from datetime import date, datetime

import numpy as np
import pandas as pd

from data.provenance import library_versions as _lib_versions
from data.provenance import now_iso as _now_iso


def _yf_version():
    return _lib_versions().get('yfinance')


class SnapshotCache:
    """Read/write financial snapshots to disk, keyed by (ticker, date)."""

    def __init__(self, cache_dir='data/cache'):
        self._cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, ticker, financials, as_of=None):
        """Persist a financials dict to disk.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL').
            financials: Dict with keys 'balance_sheet', 'income_statement',
                        'cash_flow', 'info', and optionally 'growth_estimates'
                        and 'earnings_history'.
            as_of: Date to tag the snapshot. Defaults to today.

        Returns:
            str: File path of the saved snapshot.
        """
        as_of = as_of or date.today()
        ticker_dir = os.path.join(self._cache_dir, ticker.upper())
        os.makedirs(ticker_dir, exist_ok=True)
        file_path = os.path.join(ticker_dir, f'financials_{as_of.isoformat()}.json')
        payload = self._serialize_financials(financials)
        payload['_meta'] = {'ticker': ticker.upper(), 'date': as_of.isoformat(),
                            'fetched_at': _now_iso(), 'yfinance_version': _yf_version()}
        # Write-then-rename so a crash/watchdog kill mid-write can't leave a
        # truncated JSON that breaks every future replay load for this ticker.
        tmp_path = f'{file_path}.tmp.{os.getpid()}'
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, default=_json_default)
        os.replace(tmp_path, file_path)
        return file_path

    def load(self, ticker, as_of):
        """Load the snapshot closest to (but not after) *as_of*.

        Args:
            ticker: Stock ticker symbol.
            as_of: Date to load.  Returns the most recent snapshot on or
                   before this date.

        Returns:
            dict or None: Deserialized financials dict, or None if no
            snapshot exists on or before *as_of*.
        """
        dates = self.available_dates(ticker)
        if not dates:
            return None
        # Find closest date <= as_of
        candidates = [d for d in dates if d <= as_of]
        if not candidates:
            return None
        best = max(candidates)
        file_path = os.path.join(
            self._cache_dir, ticker.upper(),
            f'financials_{best.isoformat()}.json')
        try:
            with open(file_path, 'r') as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A corrupt/truncated snapshot must not crash every replay —
            # treat it as missing (and say so).
            print(f'WARNING: unreadable snapshot {file_path}: {e}')
            return None
        return self._deserialize_financials(raw)

    def available_dates(self, ticker):
        """Return sorted list of snapshot dates for a ticker.

        Returns:
            list[date]: Ascending-sorted dates with available snapshots.
        """
        ticker_dir = os.path.join(self._cache_dir, ticker.upper())
        if not os.path.isdir(ticker_dir):
            return []
        dates = []
        for fname in os.listdir(ticker_dir):
            if fname.startswith('financials_') and fname.endswith('.json'):
                date_str = fname.replace('financials_', '').replace('.json', '')
                try:
                    dates.append(date.fromisoformat(date_str))
                except ValueError:
                    continue
        return sorted(dates)

    def has_snapshot(self, ticker, as_of):
        """Check if a snapshot exists for this ticker on this exact date."""
        return as_of in self.available_dates(ticker)

    def all_tickers(self):
        """Return list of ticker symbols that have at least one snapshot."""
        if not os.path.isdir(self._cache_dir):
            return []
        return sorted([
            d for d in os.listdir(self._cache_dir)
            if os.path.isdir(os.path.join(self._cache_dir, d))
            and not d.startswith('.')
        ])

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_financials(self, financials):
        """Convert a financials dict to a JSON-safe dict.

        DataFrames are stored as ``{_df: True, ...}`` using
        ``DataFrame.to_dict(orient='split')``.  Plain dicts, None, and
        scalar values pass through unchanged.
        """
        out = {}
        for key, value in financials.items():
            if isinstance(value, pd.DataFrame):
                d = value.to_dict(orient='split')
                # Convert Timestamps and numpy types in index/columns
                d['index'] = [_to_serializable(v) for v in d['index']]
                d['columns'] = [_to_serializable(v) for v in d['columns']]
                d['data'] = [[_to_serializable(cell) for cell in row]
                             for row in d['data']]
                d['_df'] = True
                out[key] = d
            elif isinstance(value, pd.Series):
                d = {
                    '_series': True,
                    'index': [_to_serializable(v) for v in value.index],
                    'data': [_to_serializable(v) for v in value.values],
                    'name': _to_serializable(value.name),
                }
                out[key] = d
            elif isinstance(value, dict):
                out[key] = _sanitize_dict(value)
            else:
                out[key] = _to_serializable(value)
        return out

    def _deserialize_financials(self, raw):
        """Reconstruct DataFrames from a JSON-safe dict."""
        out = {}
        for key, value in raw.items():
            if key == '_meta':
                continue
            if isinstance(value, dict) and value.get('_df'):
                cols = [_parse_timestamp(c) for c in value['columns']]
                idx = [_parse_timestamp(i) for i in value['index']]
                df = pd.DataFrame(value['data'], index=idx, columns=cols)
                out[key] = df
            elif isinstance(value, dict) and value.get('_series'):
                idx = [_parse_timestamp(i) for i in value['index']]
                out[key] = pd.Series(value['data'], index=idx,
                                     name=value.get('name'))
            else:
                out[key] = value
        return out


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _to_serializable(obj):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if obj is None:
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _json_default(obj):
    """Fallback handler for json.dump."""
    return _to_serializable(obj)


def _sanitize_dict(d):
    """Recursively convert numpy/pandas types in a dict."""
    out = {}
    for k, v in d.items():
        k = str(k)
        if isinstance(v, dict):
            out[k] = _sanitize_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_to_serializable(i) for i in v]
        else:
            out[k] = _to_serializable(v)
    return out


def _parse_timestamp(val):
    """Attempt to reconstruct a Timestamp from an ISO string."""
    if isinstance(val, str):
        try:
            return pd.Timestamp(val)
        except (ValueError, TypeError):
            return val
    return val
