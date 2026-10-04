from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


OUTCOME_START = date(2025, 1, 1)
OUTCOME_END = date(2025, 12, 31)


_CAUSAL_CLOCK_PATTERN = re.compile(
    r"^(?:ts_(?:event|recv)|(?:available|feature|decision|arrival|submission|"
    r"acknowledgement|termination|terminal|landmark|entry|exit|event)_time)_ns$"
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {source}")
    if "archive_root" in value:
        value["archive_root"] = os.environ.get("KX_MBO_ROOT", value["archive_root"])
    return value


def resolve_archive_root(config: dict[str, Any]) -> Path:
    root = Path(os.environ.get("KX_MBO_ROOT", str(config["archive_root"]))).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"MBO archive root does not exist: {root}")
    return root.resolve()


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def enforce_outcome_dates(
    start: Any,
    end: Any | None = None,
    *,
    minimum: date = OUTCOME_START,
    maximum: date = OUTCOME_END,
) -> None:
    start_date = _as_date(start)
    end_date = _as_date(end if end is not None else start)
    if start_date > end_date:
        raise ValueError(f"start date {start_date} is after end date {end_date}")
    if start_date < minimum:
        raise PermissionError(
            f"outcome-bearing viability work is restricted to {minimum} through "
            f"{maximum}; requested {start_date}"
        )
    if end_date > maximum:
        raise PermissionError(
            f"outcome-bearing viability work is sealed after {maximum}; requested {end_date}"
        )


def _datetime_series(values: Any, *, column: str) -> pd.DatetimeIndex:
    """Convert a pandas/polars clock column without silently guessing ns integers."""

    if hasattr(values, "to_pandas"):
        values = values.to_pandas()
    series = pd.Series(values)
    non_null = series.dropna()
    if non_null.empty:
        return pd.DatetimeIndex([], tz="UTC")
    if pd.api.types.is_numeric_dtype(non_null):
        converted = pd.to_datetime(non_null, unit="ns", utc=True, errors="raise")
    else:
        converted = pd.to_datetime(non_null, utc=True, errors="raise")
    try:
        return pd.DatetimeIndex(converted)
    except (TypeError, ValueError) as error:  # pragma: no cover - defensive
        raise ValueError(f"cannot interpret causal clock column {column!r}") from error


def causal_clock_columns(frame: Any) -> tuple[str, ...]:
    """Return absolute causal clocks that must obey the outcome-year seal.

    Duration columns such as ``lifetime_ns`` are intentionally excluded.  The
    accepted names are explicit absolute timestamps used throughout the MBO
    contracts.  This prevents a 2026 row from being relabelled with a 2025
    ``date`` value and passing the seal.
    """

    return tuple(
        str(column)
        for column in frame.columns
        if _CAUSAL_CLOCK_PATTERN.fullmatch(str(column)) is not None
    )


def enforce_2025_frame(
    frame: Any,
    date_column: str = "date",
    *,
    clock_columns: tuple[str, ...] | list[str] | None = None,
) -> Any:
    if date_column not in frame.columns:
        raise KeyError(f"missing date column {date_column!r}")
    dates = _datetime_series(frame[date_column], column=date_column)
    if len(dates):
        minimum = dates.min().date()
        maximum = dates.max().date()
        if minimum < OUTCOME_START or maximum > OUTCOME_END:
            raise PermissionError(
                "frame contains outcomes outside the permitted 2025 window: "
                f"{minimum} through {maximum}"
            )
    clocks = causal_clock_columns(frame) if clock_columns is None else tuple(clock_columns)
    for column in clocks:
        if column not in frame.columns:
            raise KeyError(f"missing declared causal clock column {column!r}")
        clock = _datetime_series(frame[column], column=column)
        if not len(clock):
            continue
        minimum = clock.min().date()
        maximum = clock.max().date()
        if minimum < OUTCOME_START or maximum > OUTCOME_END:
            raise PermissionError(
                "frame contains causal clocks outside the permitted 2025 window "
                f"in {column!r}: {minimum} through {maximum}"
            )
    return frame


def validate_registered_partitions(
    frame: Any,
    split_config: dict[str, Any],
    *,
    date_column: str = "date",
    partition_column: str = "partition",
    development_label: str = "development",
    locked_label: str = "locked_validation",
) -> None:
    """Bind caller labels to the frozen chronological ranges.

    Development rows may use the pre-fold warm-up period, but may not extend
    past the final registered development fold. Locked rows must fall wholly
    inside the untouched locked range. Calibration dates cannot be relabelled
    as either partition.
    """

    if partition_column not in frame.columns:
        raise KeyError(f"missing partition column {partition_column!r}")
    dates = pd.Series(
        _datetime_series(frame[date_column], column=date_column),
        index=getattr(frame, "index", None),
    )
    partitions = pd.Series(frame[partition_column]).astype(str)
    folds = split_config.get("development_folds", ())
    if not folds:
        raise ValueError("registered split configuration has no development folds")
    development_end = max(_as_date(item["validation_end"]) for item in folds)
    locked = split_config.get("locked_viability") or {}
    locked_start = _as_date(locked["start"])
    locked_end = _as_date(locked["end"])
    allowed = {development_label, locked_label}
    unexpected = sorted(set(partitions.dropna()) - allowed)
    if unexpected:
        raise ValueError(f"unregistered partition labels: {unexpected}")
    calendar_dates = dates.dt.date
    dev_bad = partitions.eq(development_label) & (calendar_dates > development_end)
    locked_bad = partitions.eq(locked_label) & (
        (calendar_dates < locked_start) | (calendar_dates > locked_end)
    )
    if bool(dev_bad.any()):
        raise ValueError(
            f"development rows extend beyond registered cutoff {development_end}"
        )
    if bool(locked_bad.any()):
        raise ValueError(
            "locked-validation labels do not match the registered locked range "
            f"{locked_start} through {locked_end}"
        )


def validate_registered_folds(
    frame: Any,
    split_config: dict[str, Any],
    *,
    date_column: str = "date",
    fold_column: str = "fold",
) -> tuple[str, ...]:
    """Validate fold labels/dates and return their registered chronology."""

    if fold_column not in frame.columns:
        raise KeyError(f"missing fold column {fold_column!r}")
    dates = pd.Series(
        _datetime_series(frame[date_column], column=date_column),
        index=getattr(frame, "index", None),
    ).dt.date
    registered = tuple(split_config.get("development_folds", ()))
    order = tuple(str(item["name"]) for item in registered)
    ranges = {
        str(item["name"]): (
            _as_date(item["validation_start"]),
            _as_date(item["validation_end"]),
        )
        for item in registered
    }
    labels = pd.Series(frame[fold_column]).astype(str)
    unexpected = sorted(set(labels.dropna()) - set(order))
    if unexpected:
        raise ValueError(f"unregistered chronological fold labels: {unexpected}")
    for name, (start, end) in ranges.items():
        selected = labels.eq(name)
        if not bool(selected.any()):
            raise ValueError(f"registered fold {name!r} is absent")
        bad = selected & ((dates < start) | (dates > end))
        if bool(bad.any()):
            raise ValueError(
                f"fold {name!r} rows fall outside {start} through {end}"
            )
    return order


def validate_ordered_split(
    frame: Any,
    *,
    split_column: str,
    time_column: str,
    train_label: str,
    validation_label: str,
    purge_hours: float = 72.0,
) -> None:
    """Require a strict, purged train-then-validation temporal split."""

    for column in (split_column, time_column):
        if column not in frame.columns:
            raise KeyError(f"missing ordered-split column {column!r}")
    labels = pd.Series(frame[split_column]).astype(str)
    unexpected = sorted(set(labels) - {str(train_label), str(validation_label)})
    if unexpected:
        raise ValueError(f"unregistered ordered-split labels: {unexpected}")
    clock = pd.Series(
        _datetime_series(frame[time_column], column=time_column),
        index=getattr(frame, "index", None),
    )
    train = clock.loc[labels.eq(str(train_label))]
    validation = clock.loc[labels.eq(str(validation_label))]
    if train.empty or validation.empty:
        raise ValueError("ordered split requires nonempty train and validation rows")
    boundary = validation.min() - pd.to_timedelta(float(purge_hours), unit="h")
    if train.max() >= boundary:
        raise ValueError(
            "training rows violate the registered purge before validation: "
            f"train max {train.max()}, validation min {validation.min()}"
        )
