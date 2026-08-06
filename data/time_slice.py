# VENDORED VERBATIM from stock-analysis-model @ 168c17a792f28ec56af874fa2de1be5c3ae92db7
# on 2026-08-06 (data/time_slice.py).
# Kernel module — keep structurally in sync; run `python tools/kernel_diff.py`.
# Local divergences: none. Load-bearing for the M8 historical backfill — it is what keeps the reconstructed issuer fundamentals free of look-ahead bias.
# data/time_slice.py
"""Point-in-time slicing of financial data to prevent look-ahead bias.

When replaying the analysis pipeline for a historical date, we must ensure
that only financial data that *would have been publicly available* on that
date is used.  A fiscal year ending 2024-12-31 is typically not filed until
60-90 days later (10-K deadline), so the default reporting lag is 90 days.
"""

from datetime import date, timedelta

import pandas as pd


def slice_financials_as_of(financials, as_of, reporting_lag_days=90):
    """Return a copy of *financials* with only data available before *as_of*.

    For DataFrames (balance_sheet, income_statement, cash_flow):
      - Keep only columns whose fiscal-period end date + *reporting_lag_days*
        is on or before *as_of*.

    For everything else (info dict, growth_estimates, earnings_history):
      - Passed through as-is.  These represent the snapshot-date reality
        and were captured by the cache at save time.

    Args:
        financials: Full financials dict from yfinance / snapshot cache.
        as_of: Date to slice to.
        reporting_lag_days: Days after fiscal period end before the data
            is considered public (default 90).

    Returns:
        New financials dict with time-sliced DataFrames.  The original
        dict is not mutated.
    """
    if financials is None:
        return None

    out = {}
    df_keys = ('balance_sheet', 'income_statement', 'cash_flow')
    for key, value in financials.items():
        if key in df_keys and isinstance(value, pd.DataFrame):
            out[key] = _slice_dataframe(value, as_of, reporting_lag_days)
        else:
            out[key] = value
    return out


def _slice_dataframe(df, as_of, reporting_lag_days):
    """Keep only columns whose fiscal date would have been published by *as_of*.

    Args:
        df: DataFrame whose columns are Timestamps representing fiscal period
            end dates (e.g. 2024-12-31, 2023-12-31).
        as_of: The point-in-time date.
        reporting_lag_days: Grace period for publication.

    Returns:
        DataFrame with only the available columns.  If no columns survive,
        returns an empty DataFrame with the same index.
    """
    if df is None or df.empty:
        return df

    keep = []
    for col in df.columns:
        if _is_available(col, as_of, reporting_lag_days):
            keep.append(col)

    if not keep:
        return pd.DataFrame(index=df.index)
    return df[keep]


def _is_available(column_date, as_of, reporting_lag_days):
    """Check if a fiscal-period column would have been published by *as_of*.

    Args:
        column_date: The fiscal period end date (Timestamp or date).
        as_of: Point-in-time date.
        reporting_lag_days: Publication delay.

    Returns:
        bool
    """
    # Normalise to date objects
    if hasattr(column_date, 'date'):
        col_date = column_date.date()
    elif isinstance(column_date, date):
        col_date = column_date
    else:
        # Try parsing a string
        try:
            col_date = pd.Timestamp(column_date).date()
        except (ValueError, TypeError):
            return False

    publication_date = col_date + timedelta(days=reporting_lag_days)
    return publication_date <= as_of
