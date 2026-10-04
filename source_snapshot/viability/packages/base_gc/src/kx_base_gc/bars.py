from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


BAR_FEATURES = ("open", "high", "low", "close", "volume")
BOUNDARY_COLUMNS = ("raw_symbol", "continuity_segment_id", "state_segment_id")
FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000


def _bar_end_ns(frame: pd.DataFrame) -> pd.Series:
    if "bar_end_ns" in frame:
        return pd.to_numeric(frame["bar_end_ns"], errors="raise").astype("int64")
    if "bar_end" not in frame:
        raise KeyError("bars require bar_end_ns or bar_end")
    return pd.to_datetime(frame["bar_end"], utc=True, errors="raise").astype("int64")


def _required_boundaries(frame: pd.DataFrame) -> tuple[str, ...]:
    missing = [column for column in BOUNDARY_COLUMNS if column not in frame]
    if missing:
        raise KeyError(f"bars are missing causal boundary columns: {missing}")
    return BOUNDARY_COLUMNS


def validate_m5_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonically order completed M5 bars.

    ``state_segment_id`` is intentionally separate from a trading episode.  It
    must change at every source discontinuity, contract roll, or other boundary
    across which historical state is forbidden.
    """

    boundaries = _required_boundaries(bars)
    required = {*BAR_FEATURES, "available_time_ns"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise KeyError(f"bars are missing required columns: {missing}")
    frame = bars.copy()
    frame["bar_end_ns"] = _bar_end_ns(frame)
    frame["available_time_ns"] = pd.to_numeric(
        frame["available_time_ns"], errors="raise"
    ).astype("int64")
    if bool((frame["available_time_ns"] < frame["bar_end_ns"]).any()):
        raise ValueError("a bar cannot be available before its completed boundary")
    for feature in BAR_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float)
    if bool((frame[["open", "high", "low", "close"]] <= 0).any().any()):
        raise ValueError("price inputs must be strictly positive")
    if bool((frame["volume"] < 0).any()):
        raise ValueError("volume cannot be negative")
    key = [*boundaries, "bar_end_ns"]
    if bool(frame.duplicated(key).any()):
        raise ValueError("duplicate M5 boundary within a causal state segment")
    frame = frame.sort_values(key, kind="mergesort").reset_index(drop=True)
    return frame


def _complete_rollup(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes not in {15, 60}:
        raise ValueError("only M15 and H1 may be derived from M5")
    factor = minutes // 5
    interval_ns = minutes * 60 * 1_000_000_000
    source = bars.copy()
    source["target_end_ns"] = (
        ((source["bar_end_ns"] - 1) // interval_ns + 1) * interval_ns
    ).astype("int64")
    rows: list[dict[str, Any]] = []
    group_columns = [*BOUNDARY_COLUMNS, "target_end_ns"]
    for key, group in source.groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values("bar_end_ns", kind="mergesort")
        ends = group["bar_end_ns"].to_numpy(dtype=np.int64)
        target_end = int(key[-1])
        expected = target_end - np.arange(factor - 1, -1, -1, dtype=np.int64) * FIVE_MINUTES_NS
        if len(group) != factor or not np.array_equal(ends, expected):
            # Missing or duplicated components invalidate the coarse bar rather
            # than being forward-filled or silently bridged.
            continue
        record = {
            column: key[index] for index, column in enumerate(BOUNDARY_COLUMNS)
        }
        record.update(
            {
                "bar_end_ns": target_end,
                "available_time_ns": int(
                    max(target_end, int(group["available_time_ns"].max()))
                ),
                "open": float(group.iloc[0]["open"]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": float(group.iloc[-1]["close"]),
                "volume": float(group["volume"].sum()),
                "component_first_end_ns": int(ends[0]),
                "component_last_end_ns": int(ends[-1]),
                "component_count": factor,
            }
        )
        if "active_contract_map_hash" in group:
            values = group["active_contract_map_hash"].dropna().astype(str).unique()
            if len(values) != 1:
                raise ValueError("a derived bar spans multiple active-contract maps")
            record["active_contract_map_hash"] = values[0]
        rows.append(record)
    return pd.DataFrame(rows)


def build_causal_timeframes(bars_5m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return M5, M15, and H1 bars built only from completed M5 components."""

    m5 = validate_m5_bars(bars_5m)
    return {"M5": m5, "M15": _complete_rollup(m5, 15), "H1": _complete_rollup(m5, 60)}


def assert_prefix_invariant(
    full: pd.DataFrame,
    prefix_lengths: Iterable[int],
) -> None:
    """Diagnostic used by tests/audits to prove future tails cannot alter bars."""

    full_result = build_causal_timeframes(full)
    for length in prefix_lengths:
        prefix = full.iloc[: int(length)].copy()
        cutoff = int(_bar_end_ns(prefix).max())
        partial_result = build_causal_timeframes(prefix)
        for name in ("M5", "M15", "H1"):
            expected = full_result[name].loc[
                full_result[name]["bar_end_ns"] <= cutoff
            ].reset_index(drop=True)
            observed = partial_result[name].reset_index(drop=True)
            if expected.empty and observed.empty:
                # A prefix shorter than three/twelve M5 components has no
                # coarse bar.  The empty-frame column layout is an
                # implementation detail, not a causal-prefix difference.
                continue
            pd.testing.assert_frame_equal(observed, expected)
