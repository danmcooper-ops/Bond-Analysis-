"""Structured logging.

The equity model has none — every diagnostic is a bare print(), which is fine
when a human is watching a terminal and useless when a scheduled job fails at
04:00 and you want to know which of 2,300 tickers was in flight. A pipeline
that runs unattended daily needs a log file with timestamps.

Console output stays terse (INFO and above, no timestamps) so an interactive
run still reads cleanly; the file gets everything with full context.
"""

import logging
import os
import sys
from datetime import date

_CONFIGURED = False
_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'logs')


def configure(level=logging.INFO, log_dir=None, run_date=None, quiet=False):
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger('bond')

    root = logging.getLogger('bond')
    root.setLevel(logging.DEBUG)
    root.propagate = False

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(console)

    directory = log_dir or _LOG_DIR
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = (run_date or date.today()).isoformat()
        handler = logging.FileHandler(
            os.path.join(directory, f'run_{stamp}.log'), encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-7s %(name)-22s %(message)s'))
        root.addHandler(handler)
    except OSError:
        # A read-only or missing output dir must never stop an analysis run;
        # console logging still works.
        pass

    _CONFIGURED = True
    return root


def get_logger(name):
    """Return a namespaced logger, configuring the root on first use."""
    configure()
    return logging.getLogger(f'bond.{name}')
