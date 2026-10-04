"""Exact GC Globex trading-date calendar primitives.

The trading day is defined in Chicago civil time: it opens at 17:00 on the
previous local calendar day, closes at 16:00, and is unavailable during the
16:00--17:00 maintenance interval.  Civil date/minute arithmetic is
intentional.  Subtracting an absolute 17-hour timedelta is wrong across DST.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd


GLOBEX_TIME_ZONE = "America/Chicago"
GLOBEX_SESSION_OPEN_MINUTE = 17 * 60
GLOBEX_MAINTENANCE_START_MINUTE = 16 * 60


def _local_trading_date(timestamp: pd.Timestamp) -> str:
    local = timestamp.tz_convert(GLOBEX_TIME_ZONE)
    minute = int(local.hour) * 60 + int(local.minute)
    if GLOBEX_MAINTENANCE_START_MINUTE <= minute < GLOBEX_SESSION_OPEN_MINUTE:
        raise ValueError("clock falls inside the registered Globex maintenance break")
    trading_date: date = local.date()
    if minute >= GLOBEX_SESSION_OPEN_MINUTE:
        trading_date += timedelta(days=1)
    return trading_date.isoformat()


def globex_session_date_at_ns(time_ns: int) -> str:
    """Return the trading date of one exact UTC instant."""

    clock = int(time_ns)
    if clock <= 0:
        raise ValueError("time_ns must be a positive UTC nanosecond clock")
    return _local_trading_date(pd.Timestamp(clock, unit="ns", tz="UTC"))


def globex_session_date_for_bar_end_ns(bar_end_time_ns: int) -> str:
    """Return the trading date of a right-labelled completed bar."""

    boundary = int(bar_end_time_ns)
    if boundary <= 0:
        raise ValueError("bar_end_time_ns must be a positive UTC nanosecond clock")
    return _local_trading_date(pd.Timestamp(boundary - 1, unit="ns", tz="UTC"))


def globex_session_dates_at_ns(values: Iterable[int]) -> pd.Series:
    return pd.Series([globex_session_date_at_ns(int(value)) for value in values], dtype="object")


def globex_session_dates_for_bar_end_ns(values: Iterable[int]) -> pd.Series:
    return pd.Series(
        [globex_session_date_for_bar_end_ns(int(value)) for value in values],
        dtype="object",
    )


def require_globex_dates_at_ns(
    labelled_dates: Iterable[object], values: Iterable[int], *, label: str
) -> pd.Series:
    normalized = pd.to_datetime(pd.Series(list(labelled_dates)), errors="raise").dt.date.astype(str)
    derived = globex_session_dates_at_ns(values)
    if len(normalized) != len(derived) or not normalized.reset_index(drop=True).equals(
        derived.reset_index(drop=True)
    ):
        raise ValueError(f"{label} date disagrees with the registered Globex trading date")
    return normalized.reset_index(drop=True)


def require_globex_dates_for_bar_end_ns(
    labelled_dates: Iterable[object], values: Iterable[int], *, label: str
) -> pd.Series:
    normalized = pd.to_datetime(pd.Series(list(labelled_dates)), errors="raise").dt.date.astype(str)
    derived = globex_session_dates_for_bar_end_ns(values)
    if len(normalized) != len(derived) or not normalized.reset_index(drop=True).equals(
        derived.reset_index(drop=True)
    ):
        raise ValueError(f"{label} date disagrees with the registered Globex trading date")
    return normalized.reset_index(drop=True)


__all__ = [
    "GLOBEX_MAINTENANCE_START_MINUTE",
    "GLOBEX_SESSION_OPEN_MINUTE",
    "GLOBEX_TIME_ZONE",
    "globex_session_date_at_ns",
    "globex_session_date_for_bar_end_ns",
    "globex_session_dates_at_ns",
    "globex_session_dates_for_bar_end_ns",
    "require_globex_dates_at_ns",
    "require_globex_dates_for_bar_end_ns",
]
