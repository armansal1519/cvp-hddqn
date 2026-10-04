from __future__ import annotations

from typing import Iterable

import pandas as pd

from kx_mbo_core.globex import (
    GLOBEX_MAINTENANCE_START_MINUTE,
    GLOBEX_SESSION_OPEN_MINUTE,
    GLOBEX_TIME_ZONE,
    globex_session_date_for_bar_end_ns,
    globex_session_dates_for_bar_end_ns,
    require_globex_dates_for_bar_end_ns,
)


_MINUTE_NS = 60 * 1_000_000_000


def globex_session_date(bar_end_time_ns: int) -> str:
    """Derive the registered GC trading date from a completed bar boundary.

    GC's Globex trading day opens at 17:00 America/Chicago and carries the next
    local calendar date.  The interval from 16:00 through 16:59:59 is the daily
    maintenance break.  Bars are right-labelled, so the final nanosecond of the
    completed interval, rather than the next interval, determines its session.
    """

    return globex_session_date_for_bar_end_ns(bar_end_time_ns)


def globex_session_dates(bar_end_time_ns: Iterable[int]) -> pd.Series:
    """Vector-shaped wrapper preserving input order and failing on breaks."""

    return globex_session_dates_for_bar_end_ns(bar_end_time_ns)


def require_exact_globex_session_dates(
    labelled_dates: Iterable[object],
    bar_end_time_ns: Iterable[int],
    *,
    label: str,
) -> pd.Series:
    """Normalize and prove labels equal the registered exchange trading date."""

    return require_globex_dates_for_bar_end_ns(
        labelled_dates, bar_end_time_ns, label=label
    )


def is_registered_globex_bar_step(
    previous_bar_end_ns: int, current_bar_end_ns: int, minutes: int
) -> bool:
    """Accept a normal bar step or the single registered daily break."""

    previous = int(previous_bar_end_ns)
    current = int(current_bar_end_ns)
    interval = int(minutes) * _MINUTE_NS
    if current - previous == interval:
        return True
    if current - previous != interval + 60 * _MINUTE_NS:
        return False
    previous_local = pd.Timestamp(previous, unit="ns", tz="UTC").tz_convert(
        GLOBEX_TIME_ZONE
    )
    current_local = pd.Timestamp(current, unit="ns", tz="UTC").tz_convert(
        GLOBEX_TIME_ZONE
    )
    expected_current_minute = GLOBEX_SESSION_OPEN_MINUTE + int(minutes)
    return (
        previous_local.date() == current_local.date()
        and previous_local.hour * 60 + previous_local.minute
        == GLOBEX_MAINTENANCE_START_MINUTE
        and current_local.hour * 60 + current_local.minute == expected_current_minute
        and previous_local.second == current_local.second == 0
        and previous_local.nanosecond == current_local.nanosecond == 0
    )
