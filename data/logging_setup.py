"""Structured logging.

The equity model has none — every diagnostic is a bare print(), which is fine
when a human is watching a terminal and useless when a scheduled job fails at
04:00 and you want to know which of 2,300 tickers was in flight. A pipeline
that runs unattended daily needs a log file with timestamps.

Console output stays terse (INFO and above, no timestamps) so an interactive
run still reads cleanly; the file gets everything with full context.

The file lives in $BOND_LOG_DIR when set, else output/logs. The test suite
sets it to a temp dir: tests used to append fixture chatter to the production
run log, which is also the record used to count analysis runs per day.
"""

import logging
import os
import sys
from datetime import date

_CONFIGURED = False
_FILE_HANDLER = None
_DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'logs')


def _log_dir(log_dir=None):
    return log_dir or os.environ.get('BOND_LOG_DIR') or _DEFAULT_LOG_DIR


def _attach_file_handler(root, directory, run_date):
    """(Re)point the file handler at run_<date>.log in `directory`."""
    global _FILE_HANDLER
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = (run_date or date.today()).isoformat()
        # delay=True: the file is created on the first record, not at import.
        # Merely importing a data module used to leave an empty run log.
        handler = logging.FileHandler(
            os.path.join(directory, f'run_{stamp}.log'), encoding='utf-8',
            delay=True)
    except OSError:
        # A read-only or missing output dir must never stop an analysis run;
        # console logging still works.
        return
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-7s %(name)-22s %(message)s'))
    if _FILE_HANDLER is not None:
        root.removeHandler(_FILE_HANDLER)
        _FILE_HANDLER.close()
    root.addHandler(handler)
    _FILE_HANDLER = handler


def configure(level=logging.INFO, log_dir=None, run_date=None, quiet=False):
    """Configure root logging. Safe to call repeatedly.

    The first call installs the handlers. A later call that passes `log_dir`
    or `run_date` re-points the file handler, so an entry point can stamp the
    log with its run-start date even though every module configured logging
    at import time with today's date.
    """
    global _CONFIGURED
    root = logging.getLogger('bond')
    if _CONFIGURED:
        if log_dir is not None or run_date is not None:
            _attach_file_handler(root, _log_dir(log_dir), run_date)
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False

    if not quiet:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(console)

    _attach_file_handler(root, _log_dir(log_dir), run_date)
    _CONFIGURED = True
    return root


def get_logger(name):
    """Return a namespaced logger, configuring the root on first use."""
    configure()
    return logging.getLogger(f'bond.{name}')
