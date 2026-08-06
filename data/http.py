"""One HTTP layer: throttling, retries, wall-clock timeouts, atomic downloads.

The equity model duplicates this logic four times (yfinance_client, fmp_client,
sec_xbrl_client, macro_client), each with slightly different retry semantics.
Consolidated here so there is one place to fix a rate-limit bug.

One deliberate divergence from upstream: `retry_on_timeout` defaults to True.
The equity model never retries a timeout, with good reason — a hung yfinance
call retried is just more orphaned threads and sockets in CLOSE_WAIT. But this
model's largest job is pulling multi-gigabyte SEC ZIPs, where a timeout is
usually transient congestion and not retrying means losing a 40-minute
download. Callers wrapping a flaky-hang API should pass retry_on_timeout=False.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import requests

DEFAULT_USER_AGENT = os.environ.get(
    'SEC_USER_AGENT', 'BondAnalysisModel/1.0 (contact: set SEC_USER_AGENT)')

# Shared pool for wall-clock deadlines. Module-level and bounded so a run
# cannot spawn an unbounded number of timeout threads.
_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=4,
                                       thread_name_prefix='http-timeout')


class RateLimiter:
    """Per-host minimum spacing between requests. Thread-safe.

    Keyed by host so a slow SEC crawl does not throttle FRED calls, and so
    each host's published limit can be respected independently. SEC asks for
    <=10 requests/second; Treasury and FRED publish no limit but a courteous
    default costs nothing.
    """

    def __init__(self, default_interval=0.12):
        self._default = default_interval
        self._intervals = {}
        self._last = {}
        self._lock = threading.Lock()

    def set_interval(self, host, seconds):
        with self._lock:
            self._intervals[host] = seconds

    def wait(self, host):
        with self._lock:
            interval = self._intervals.get(host, self._default)
            elapsed = time.time() - self._last.get(host, 0.0)
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last[host] = time.time()


_LIMITER = RateLimiter()
_LIMITER.set_interval('www.sec.gov', 0.12)
_LIMITER.set_interval('data.sec.gov', 0.12)
_LIMITER.set_interval('home.treasury.gov', 0.30)
_LIMITER.set_interval('www.treasurydirect.gov', 0.30)
_LIMITER.set_interval('fred.stlouisfed.org', 0.20)
_LIMITER.set_interval('api.stlouisfed.org', 0.20)


def _host_of(url):
    try:
        return url.split('//', 1)[1].split('/', 1)[0]
    except IndexError:
        return url


def run_with_timeout(fn, seconds):
    """Run fn() under a hard wall-clock deadline. None on timeout.

    Belt-and-braces over the socket timeout: a library that manages its own
    connections can hang past any timeout it was configured with, and an
    unattended daily run cannot afford to block forever on one URL.
    """
    future = _TIMEOUT_EXECUTOR.submit(fn)
    try:
        return future.result(timeout=seconds)
    except FutureTimeout:
        future.cancel()
        return None
    except Exception:
        return None


def get(url, *, params=None, headers=None, timeout=30, max_retries=3,
        retry_on_timeout=True, binary=False, session=None):
    """GET with throttling and backoff. Returns text, bytes, or None.

    Retries on 429/5xx with linear backoff, honouring Retry-After when the
    server sends one. A 4xx other than 429 is not retried — it will not
    succeed on a second attempt and retrying only burns the rate limit.
    """
    host = _host_of(url)
    hdrs = {'User-Agent': DEFAULT_USER_AGENT}
    if headers:
        hdrs.update(headers)
    caller = session or requests

    for attempt in range(max_retries + 1):
        _LIMITER.wait(host)
        try:
            resp = caller.get(url, params=params, headers=hdrs, timeout=timeout)
        except requests.Timeout:
            if not retry_on_timeout or attempt >= max_retries:
                return None
            time.sleep(1.0 * (attempt + 1))
            continue
        except requests.RequestException:
            if attempt >= max_retries:
                return None
            time.sleep(1.0 * (attempt + 1))
            continue

        if resp.status_code == 200:
            return resp.content if binary else resp.text

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt >= max_retries:
                return None
            wait = resp.headers.get('Retry-After')
            try:
                delay = float(wait) if wait else 2.0 * (attempt + 1)
            except ValueError:
                delay = 2.0 * (attempt + 1)
            time.sleep(min(delay, 60.0))
            continue

        return None            # 4xx: will not succeed on a retry


def get_json(url, **kwargs):
    """GET and parse JSON. None on any failure, including malformed JSON."""
    import json
    text = get(url, **kwargs)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def download_atomic(url, dest, *, headers=None, timeout=600, chunk=1 << 20,
                    progress_every=None):
    """Stream a URL to `dest` via a temp file, then os.replace.

    Atomic because a partially-written multi-gigabyte N-PORT ZIP that looks
    complete is worse than no file: the next run would read it, fail to parse
    somewhere in the middle, and report a data problem rather than a download
    problem. Returns the destination path, or None.
    """
    host = _host_of(url)
    hdrs = {'User-Agent': DEFAULT_USER_AGENT}
    if headers:
        hdrs.update(headers)

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or '.', exist_ok=True)
    tmp = f'{dest}.tmp.{os.getpid()}'
    _LIMITER.wait(host)
    try:
        with requests.get(url, headers=hdrs, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return None
            written = 0
            with open(tmp, 'wb') as fh:
                for block in r.iter_content(chunk_size=chunk):
                    if not block:
                        continue
                    fh.write(block)
                    written += len(block)
                    if progress_every and written % progress_every < chunk:
                        print(f'  ... {written / 1e6:.0f} MB', flush=True)
        os.replace(tmp, dest)
        return dest
    except (requests.RequestException, OSError):
        return None
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_dotenv(path=None):
    """Minimal .env loader: KEY=VALUE lines, no export, no interpolation.

    Uses setdefault so a real environment variable always wins over the file
    — which is what you want when a scheduled job supplies secrets directly.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(key.strip(), value)
    return True
