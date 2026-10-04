from __future__ import annotations

from datetime import date, datetime
from enum import Enum


class LockedSplitError(PermissionError):
    pass


class Split(str, Enum):
    FINETUNE = "finetune"
    SELECTION = "selection"
    OPERATIONAL_GATE = "operational_gate"
    LOCKED_TEST = "locked_test"


def _date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value[:10])


def classify_time(value: str | date | datetime) -> Split:
    day = _date(value)
    if date(2025, 1, 1) <= day < date(2025, 7, 1):
        return Split.FINETUNE
    if day < date(2025, 9, 1) and day >= date(2025, 7, 1):
        return Split.SELECTION
    if day < date(2025, 11, 1) and day >= date(2025, 9, 1):
        return Split.OPERATIONAL_GATE
    if day < date(2026, 1, 1) and day >= date(2025, 11, 1):
        return Split.LOCKED_TEST
    raise ValueError("timestamp is outside the registered 2025 GC study")


def require_development_access(value: str | date | datetime, *, purpose: str) -> Split:
    split = classify_time(value)
    if split in {Split.OPERATIONAL_GATE, Split.LOCKED_TEST} and purpose in {
        "fit", "select", "tune", "feature_engineering", "diagnostic"
    }:
        raise LockedSplitError(f"{purpose} is forbidden on {split.value}")
    return split
