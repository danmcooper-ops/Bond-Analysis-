"""FRED: constant-maturity Treasury yields and ICE BofA credit spreads.

The bucket OAS series are the anchor of the whole valuation model — they are
what turns an issuer's credit score into a fair spread. Without them there is
no benchmark to call a bond cheap against.

TWO ACCESS PATHS, AND THE DIFFERENCE MATTERS
--------------------------------------------
Keyed (api.stlouisfed.org, free key) returns full history — the ICE BofA
series go back to 1996.

Keyless (fredgraph.csv) needs no key and works fine for daily pulls, but
measured on 2026-08-06 it returns exactly 796 rows starting 2023-08-07 for
EVERY ICE BofA series, and ignores the cosd start-date parameter entirely.
That is a rolling ~3-year window on the licensed series; DGS* is unaffected
and comes back to 1962 keyless.

So the key is not a nice-to-have. Without it the walk-forward calibration has
~3 years of spread history instead of ~30, which is the difference between
calibrating across two credit cycles and calibrating across none. The client
records `history_source` and the actual first observation on every fetch so a
truncated backfill is visible in the run log rather than silently shrinking
the training window.
"""

import csv
import io
import os
from datetime import date, datetime

from data.http import get, get_json, load_dotenv
from data.logging_setup import get_logger

log = get_logger('fred')

API_URL = 'https://api.stlouisfed.org/fred/series/observations'
CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'

# ICE BofA option-adjusted spreads by rating bucket. These are OAS, not
# effective yield — the *EY-suffixed series are yields and would be wrong here
# by the entire level of the risk-free curve.
OAS_SERIES = {
    'AAA': 'BAMLC0A1CAAA',
    'AA': 'BAMLC0A2CAA',
    'A': 'BAMLC0A3CA',
    'BBB': 'BAMLC0A4CBBB',
    'BB': 'BAMLH0A1HYBB',
    'B': 'BAMLH0A2HYB',
    'CCC': 'BAMLH0A3HYC',
}

IG_ALL = 'BAMLC0A0CM'
HY_ALL = 'BAMLH0A0HYM2'

# IG index sliced by maturity. Ratio-normalised against IG_ALL these give a
# real multiplicative term factor: a 2-year BBB and a 30-year BBB do not
# deserve the same fair spread, and the whole-index OAS cannot tell them apart.
TERM_SERIES = {
    '1-3y': 'BAMLC1A0C13Y',
    '3-5y': 'BAMLC2A0C35Y',
    '5-7y': 'BAMLC3A0C57Y',
    '7-10y': 'BAMLC4A0C710Y',
    '10-15y': 'BAMLC7A0C1015Y',
    '15y+': 'BAMLC8A0C15PY',
}

TERM_MIDPOINTS = {'1-3y': 2.0, '3-5y': 4.0, '5-7y': 6.0, '7-10y': 8.5,
                  '10-15y': 12.5, '15y+': 20.0}

# Constant-maturity Treasury yields. A second, independent read on the curve —
# used to cross-check the Treasury XML feed and to backfill history far past
# what that feed conveniently exposes.
CMT_SERIES = {
    '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO', '1Y': 'DGS1',
    '2Y': 'DGS2', '3Y': 'DGS3', '5Y': 'DGS5', '7Y': 'DGS7',
    '10Y': 'DGS10', '20Y': 'DGS20', '30Y': 'DGS30',
}


class FREDClient:
    """Fetches FRED series with an on-disk parquet-free JSON cache."""

    def __init__(self, api_key=None, cache_dir=None, max_age_days=1):
        load_dotenv()
        self.api_key = api_key or os.environ.get('FRED_API_KEY') or None
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cache', 'fred')
        self.max_age_days = max_age_days
        self._memo = {}
        self.history_source = 'keyed' if self.api_key else 'keyless'
        if not self.api_key:
            log.warning(
                'No FRED_API_KEY: falling back to the keyless endpoint, which '
                'caps ICE BofA credit-spread series at a rolling ~3-year '
                'window. Daily scoring is unaffected; the walk-forward '
                'calibration will train on a much shorter history.')

    # -- fetching -----------------------------------------------------------

    def _cache_path(self, series_id):
        return os.path.join(self.cache_dir, f'{series_id}.json')

    def _load_cache(self, series_id):
        path = self._cache_path(series_id)
        if not os.path.exists(path):
            return None
        age = (date.today() - date.fromtimestamp(os.path.getmtime(path))).days
        if age >= self.max_age_days:
            return None
        import json
        try:
            with open(path, encoding='utf-8') as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return None
        return {date.fromisoformat(k): v for k, v in raw['obs'].items()}

    def _save_cache(self, series_id, obs):
        import json
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path(series_id)
        tmp = f'{path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump({'source': self.history_source,
                           'obs': {k.isoformat(): v for k, v in obs.items()}},
                          fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def _fetch_keyed(self, series_id, start=None, end=None):
        params = {'series_id': series_id, 'api_key': self.api_key,
                  'file_type': 'json'}
        if start:
            params['observation_start'] = start.isoformat()
        if end:
            params['observation_end'] = end.isoformat()
        payload = get_json(API_URL, params=params)
        if not payload or 'observations' not in payload:
            return None
        out = {}
        for row in payload['observations']:
            raw = row.get('value')
            if raw in (None, '', '.'):        # FRED marks holidays with '.'
                continue
            try:
                out[date.fromisoformat(row['date'])] = float(raw) / 100.0
            except (ValueError, KeyError):
                continue
        return out

    def _fetch_keyless(self, series_id, start=None):
        params = {'id': series_id}
        if start:
            params['cosd'] = start.isoformat()
        text = get(CSV_URL, params=params)
        if not text or 'observation_date' not in text[:200]:
            return None
        out = {}
        for row in csv.DictReader(io.StringIO(text)):
            raw_date = row.get('observation_date')
            raw_val = row.get(series_id)
            if not raw_date or raw_val in (None, '', '.'):
                continue
            try:
                out[date.fromisoformat(raw_date)] = float(raw_val) / 100.0
            except ValueError:
                continue
        return out

    def fetch_series(self, series_id, start=None, end=None, force=False):
        """Return {date: decimal value} for a FRED series. {} on failure.

        Values are divided by 100: FRED publishes both yields and OAS in
        percent, and every consumer in this codebase works in decimals.
        """
        if not force and series_id in self._memo:
            return self._memo[series_id]
        if not force:
            cached = self._load_cache(series_id)
            if cached is not None:
                self._memo[series_id] = cached
                return cached

        obs = None
        if self.api_key:
            obs = self._fetch_keyed(series_id, start=start, end=end)
            if obs is None:
                log.warning('%s: keyed fetch failed, trying keyless', series_id)
        if obs is None:
            obs = self._fetch_keyless(series_id, start=start)

        if not obs:
            log.error('%s: no observations from FRED', series_id)
            self._memo[series_id] = {}
            return {}

        if start and min(obs) > start:
            log.warning('%s: requested from %s but earliest available is %s '
                        '(%d obs, source=%s)', series_id, start, min(obs),
                        len(obs), self.history_source)

        self._memo[series_id] = obs
        self._save_cache(series_id, obs)
        return obs

    # -- convenience --------------------------------------------------------

    @staticmethod
    def _as_of_value(obs, as_of, max_lookback_days=10):
        """Most recent observation at or before as_of. FRED lags a day or two
        and skips holidays, so an exact-date lookup would fail routinely."""
        if not obs:
            return None, None
        target = as_of or date.today()
        candidates = [d for d in obs if d <= target
                      and (target - d).days <= max_lookback_days]
        if not candidates:
            return None, None
        best = max(candidates)
        return best, obs[best]

    def fetch_bucket_oas(self, as_of=None):
        """{bucket: OAS} for the seven rating buckets, as decimals."""
        out = {}
        for bucket, series_id in OAS_SERIES.items():
            obs_date, value = self._as_of_value(self.fetch_series(series_id), as_of)
            if value is not None:
                out[bucket] = value
            else:
                log.warning('No OAS for %s near %s', bucket, as_of)
        return out

    def fetch_bucket_oas_history(self, start, end):
        """{bucket: {date: OAS}} across a range — the calibration input."""
        return {bucket: {d: v for d, v in
                         self.fetch_series(sid, start=start, end=end).items()
                         if start <= d <= end}
                for bucket, sid in OAS_SERIES.items()}

    def fetch_term_factors(self, as_of=None):
        """Multiplicative spread term factors, normalised to the IG index.

        A factor of 1.22 at 7-10y means paper of that tenor trades at 122% of
        the whole-index spread. Applied to a bucket OAS this gives a fair
        spread that respects the term structure instead of pretending a 2-year
        and a 30-year BBB are the same credit risk.

        Returns {'factors': {label: ratio}, 'points': [(years, ratio)],
                 'ig_all': base_oas}.
        """
        _, base = self._as_of_value(self.fetch_series(IG_ALL), as_of)
        if not base:
            log.warning('No IG index OAS near %s; term factors unavailable', as_of)
            return {'factors': {}, 'points': [], 'ig_all': None}

        factors, points = {}, []
        for label, series_id in TERM_SERIES.items():
            _, value = self._as_of_value(self.fetch_series(series_id), as_of)
            if value is None:
                continue
            ratio = value / base
            factors[label] = ratio
            points.append((TERM_MIDPOINTS[label], ratio))
        points.sort()
        return {'factors': factors, 'points': points, 'ig_all': base}

    def fetch_cmt_curve(self, as_of=None, exact_date=False, with_dates=False):
        """Constant-maturity Treasury curve — the cross-check on the XML feed.

        `exact_date=True` returns only observations stamped exactly `as_of`,
        with no fallback to an earlier one. That matters specifically for the
        cross-check: FRED publishes about a day behind Treasury's own feed, so
        a nearest-available lookup silently compares today's par curve against
        yesterday's CMT series. On 2026-08-06 that produced a uniform 5-7bp
        "disagreement" between two feeds that, compared same-date, agree to
        0.0bp — an overnight rate move being misread as a data-quality
        problem. Any cross-check must pin the date.
        """
        out = {}
        for tenor, series_id in CMT_SERIES.items():
            obs = self.fetch_series(series_id)
            if exact_date:
                target = as_of or date.today()
                if target not in obs:
                    continue
                obs_date, value = target, obs[target]
            else:
                obs_date, value = self._as_of_value(obs, as_of)
            if value is not None:
                out[tenor] = (obs_date, value) if with_dates else value
        return out

    def latest_common_date(self, series_ids=None, as_of=None,
                           max_lookback_days=10):
        """Most recent date on or before `as_of` present in EVERY series.

        Used to pick a comparison date the two feeds can both honour, rather
        than assuming they publish in lockstep.
        """
        ids = series_ids or list(CMT_SERIES.values())
        target = as_of or date.today()
        common = None
        for sid in ids:
            dates = {d for d in self.fetch_series(sid)
                     if d <= target and (target - d).days <= max_lookback_days}
            common = dates if common is None else (common & dates)
            if not common:
                return None
        return max(common) if common else None

    def coverage(self):
        """Report what history is actually available, per fetched series.

        Called after a backfill so the run log records the training window
        rather than leaving it to be discovered when a calibration silently
        trains on three years.
        """
        return {sid: {'first': min(obs).isoformat(), 'last': max(obs).isoformat(),
                      'n': len(obs)}
                for sid, obs in self._memo.items() if obs}


def term_factor_at(points, maturity_years, beta=1.0):
    """Interpolate a term factor at a given maturity.

    `beta` damps or amplifies the term effect and is a calibration parameter:
    beta=0 flattens the structure to 1.0 everywhere, beta=1 uses the observed
    ratios as-is. Flat extrapolation outside the observed range, for the same
    reason the yield curve does not extrapolate a slope.
    """
    if not points:
        return 1.0
    from models.curve import interpolate
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    raw = interpolate(maturity_years, xs, ys)
    if raw is None:
        return 1.0
    return 1.0 + beta * (raw - 1.0)
