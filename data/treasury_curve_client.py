"""Daily Treasury par yield curve.

Source: home.treasury.gov's Atom XML feed, which needs no key and publishes
all 14 quoted tenors. Yields are par yields on the most recently auctioned
issues, taken from NY Fed bid quotes around 3:30pm ET, so today's curve
appears after that — a morning run gets yesterday's, which is correct and not
an error.

The feed is per calendar year, so one fetch covers a whole year and gets
cached; only the current year is ever refetched.
"""

import os
import re
from datetime import date, datetime

from data.http import get
from data.logging_setup import get_logger

log = get_logger('treasury_curve')

XML_URL = ('https://home.treasury.gov/resource-center/data-chart-center/'
           'interest-rates/pages/xml')
CSV_URL = ('https://home.treasury.gov/resource-center/data-chart-center/'
           'interest-rates/daily-treasury-rates.csv/{year}/all')

# Feed field -> our tenor label. BC_30YEARDISPLAY is a formatting duplicate of
# BC_30YEAR and is deliberately excluded; including it would double-weight the
# long end of the curve.
FIELD_TO_TENOR = {
    'BC_1MONTH': '1M', 'BC_1_5MONTH': '1.5M', 'BC_2MONTH': '2M',
    'BC_3MONTH': '3M', 'BC_4MONTH': '4M', 'BC_6MONTH': '6M',
    'BC_1YEAR': '1Y', 'BC_2YEAR': '2Y', 'BC_3YEAR': '3Y', 'BC_5YEAR': '5Y',
    'BC_7YEAR': '7Y', 'BC_10YEAR': '10Y', 'BC_20YEAR': '20Y',
    'BC_30YEAR': '30Y',
}

# Yields outside this band are a feed error, not a market event.
SANE_MIN, SANE_MAX = -0.02, 0.25

_ENTRY_RE = re.compile(r'<entry>(.*?)</entry>', re.S)
_PROP_RE = re.compile(r'<d:([A-Z_0-9]+)[^>]*>([^<]*)</d:\1>')


def _parse_xml(text):
    """Parse the Atom feed into {date: {tenor: decimal yield}}.

    Regex rather than an XML parser on purpose: the feed carries two
    namespaces and Microsoft ADO type annotations, the shape has been stable
    for years, and a regex degrades to "found nothing" rather than raising on
    a malformed byte in a 230KB document.
    """
    out = {}
    for body in _ENTRY_RE.findall(text or ''):
        props = dict(_PROP_RE.findall(body))
        raw_date = props.get('NEW_DATE')
        if not raw_date:
            continue
        try:
            as_of = datetime.strptime(raw_date[:10], '%Y-%m-%d').date()
        except ValueError:
            continue
        curve = {}
        for field, tenor in FIELD_TO_TENOR.items():
            raw = props.get(field, '').strip()
            if not raw:
                continue                     # tenor not quoted that day
            try:
                pct = float(raw)
            except ValueError:
                continue
            value = pct / 100.0
            if SANE_MIN <= value <= SANE_MAX:
                curve[tenor] = value
            else:
                log.warning('%s %s = %s%% outside sane band, dropped',
                            as_of, tenor, raw)
        if curve:
            out[as_of] = curve
    return out


def _parse_csv(text):
    """Parse the CSV fallback. Header labels differ from the XML field names."""
    import csv
    import io

    label_to_tenor = {
        '1 Mo': '1M', '1.5 Month': '1.5M', '2 Mo': '2M', '3 Mo': '3M',
        '4 Mo': '4M', '6 Mo': '6M', '1 Yr': '1Y', '2 Yr': '2Y', '3 Yr': '3Y',
        '5 Yr': '5Y', '7 Yr': '7Y', '10 Yr': '10Y', '20 Yr': '20Y',
        '30 Yr': '30Y',
    }
    out = {}
    for row in csv.DictReader(io.StringIO(text or '')):
        raw_date = (row.get('Date') or '').strip()
        as_of = None
        for fmt in ('%m/%d/%Y', '%Y-%m-%d'):
            try:
                as_of = datetime.strptime(raw_date, fmt).date()
                break
            except ValueError:
                continue
        if as_of is None:
            continue
        curve = {}
        for label, tenor in label_to_tenor.items():
            raw = (row.get(label) or '').strip()
            if not raw:
                continue
            try:
                value = float(raw) / 100.0
            except ValueError:
                continue
            if SANE_MIN <= value <= SANE_MAX:
                curve[tenor] = value
        if curve:
            out[as_of] = curve
    return out


class TreasuryCurveClient:
    """Fetches and caches daily par yield curves, one JSON file per year."""

    def __init__(self, cache_dir=None, max_age_days=1):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cache', 'treasury')
        self.max_age_days = max_age_days
        self._memo = {}

    # -- fetching -----------------------------------------------------------

    def _cache_path(self, year):
        return os.path.join(self.cache_dir, f'par_curve_{year}.json')

    def _load_cache(self, year):
        path = self._cache_path(year)
        if not os.path.exists(path):
            return None
        # The current year goes stale daily; closed years never change.
        if year == date.today().year:
            age_days = (date.today() - date.fromtimestamp(
                os.path.getmtime(path))).days
            if age_days >= self.max_age_days:
                return None
        import json
        try:
            with open(path, encoding='utf-8') as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return None
        return {date.fromisoformat(k): v for k, v in raw.items()}

    def _save_cache(self, year, curves):
        import json
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path(year)
        tmp = f'{path}.tmp.{os.getpid()}'
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                json.dump({k.isoformat(): v for k, v in curves.items()}, fh)
            os.replace(tmp, path)
        except OSError:
            pass

    def fetch_year(self, year, force=False):
        """Return {date: {tenor: yield}} for a calendar year."""
        if not force:
            if year in self._memo:
                return self._memo[year]
            cached = self._load_cache(year)
            if cached:
                self._memo[year] = cached
                return cached

        text = get(XML_URL, params={'data': 'daily_treasury_yield_curve',
                                    'field_tdr_date_value': str(year)})
        curves = _parse_xml(text) if text else {}

        if not curves:
            log.warning('XML feed empty for %s, trying the CSV fallback', year)
            text = get(CSV_URL.format(year=year),
                       params={'type': 'daily_treasury_yield_curve',
                               'field_tdr_date_value': str(year),
                               '_format': 'csv'})
            curves = _parse_csv(text) if text else {}

        if curves:
            self._memo[year] = curves
            self._save_cache(year, curves)
            log.info('Treasury curve %s: %d business days', year, len(curves))
        else:
            log.error('No Treasury curve data for %s from either source', year)
        return curves

    # -- lookups ------------------------------------------------------------

    def fetch_par_curve(self, as_of=None, max_lookback_days=10):
        """Par curve for `as_of`, or the most recent one at or before it.

        Falling back to an earlier date is correct behaviour, not a
        workaround: weekends, holidays, and a morning run before the 3:30pm
        publication all legitimately have no curve for today. The returned
        date says which one you actually got, and the caller stamps it on
        every row so a stale curve is visible rather than silent.

        Returns (curve_date, {tenor: yield}) or (None, None).
        """
        target = as_of or date.today()
        curves = self.fetch_year(target.year)

        # A date early in January may need last year's feed.
        if not any(d <= target for d in curves):
            curves = {**self.fetch_year(target.year - 1), **curves}

        candidates = [d for d in curves if d <= target
                      and (target - d).days <= max_lookback_days]
        if not candidates:
            log.error('No Treasury curve within %d days of %s',
                      max_lookback_days, target)
            return None, None
        best = max(candidates)
        if best != target:
            log.info('No curve for %s; using %s (%d days back)',
                     target, best, (target - best).days)
        return best, curves[best]

    def latest(self):
        return self.fetch_par_curve(date.today())

    def fetch_curve_history(self, start, end):
        """{date: curve} across a date range, spanning years as needed."""
        out = {}
        for year in range(start.year, end.year + 1):
            for d, curve in self.fetch_year(year).items():
                if start <= d <= end:
                    out[d] = curve
        return out

    # -- regime -------------------------------------------------------------

    def regime(self, as_of=None, lookback_days=365):
        """Curve shape and direction, feeding the Rates gates.

        Returns slope, level, a 1-year level percentile, 3-month momentum, and
        labels for shape and direction. The Duration Fit gate is
        regime-CONDITIONAL, not directional: it rewards long duration when the
        curve says the market is pricing easing and short duration when it is
        not. It is a screen, not a rate call, and must never encode a view.
        """
        target = as_of or date.today()
        curve_date, curve = self.fetch_par_curve(target)
        if not curve:
            return None

        def y(tenor):
            return curve.get(tenor)

        slope_10y_3m = (y('10Y') - y('3M')) if y('10Y') and y('3M') else None
        slope_10y_2y = (y('10Y') - y('2Y')) if y('10Y') and y('2Y') else None
        level_10y = y('10Y')

        start = date(curve_date.year - 1, curve_date.month, curve_date.day) \
            if curve_date.month != 2 or curve_date.day != 29 \
            else date(curve_date.year - 1, 2, 28)
        history = self.fetch_curve_history(start, curve_date)

        tens = sorted(c['10Y'] for c in history.values() if c.get('10Y'))
        level_pctile = None
        if len(tens) >= 30 and level_10y is not None:
            below = sum(1 for v in tens if v <= level_10y)
            level_pctile = 100.0 * below / len(tens)

        momentum_3m = None
        cutoff = date.fromordinal(curve_date.toordinal() - 91)
        earlier = [(d, c['10Y']) for d, c in history.items()
                   if c.get('10Y') and d <= cutoff]
        if earlier and level_10y is not None:
            momentum_3m = level_10y - max(earlier)[1]

        if slope_10y_3m is None:
            shape = 'unknown'
        elif slope_10y_3m < -0.001:
            shape = 'inverted'
        elif slope_10y_3m < 0.005:
            shape = 'flat'
        else:
            shape = 'steep'

        # Direction combines the level move with the slope move: a bull
        # steepener (yields down, curve steeper) and a bear flattener are very
        # different environments for duration, and the level alone cannot
        # distinguish them.
        direction = 'unknown'
        if momentum_3m is not None and slope_10y_3m is not None:
            prior_slopes = [c['10Y'] - c['3M'] for d, c in history.items()
                            if c.get('10Y') and c.get('3M') and d <= cutoff]
            if prior_slopes:
                d_slope = slope_10y_3m - (sum(prior_slopes) / len(prior_slopes))
                falling = momentum_3m < 0
                steepening = d_slope > 0
                direction = (('bull_' if falling else 'bear_')
                             + ('steepener' if steepening else 'flattener'))

        return {
            'curve_date': curve_date,
            'slope_10y_3m': slope_10y_3m,
            'slope_10y_2y': slope_10y_2y,
            'level_10y': level_10y,
            'level_pctile_1y': level_pctile,
            'momentum_3m': momentum_3m,
            'shape': shape,
            'direction': direction,
            'history_days': len(history),
        }
