from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .bars import BAR_FEATURES, BOUNDARY_COLUMNS
from .contract import ACTIONS, AgentSpec
from .sessions import is_registered_globex_bar_step


@dataclass(frozen=True)
class StateRecord:
    agent: str
    raw_symbol: str
    continuity_segment_id: int
    state_segment_id: str
    bar_end_ns: int
    available_time_ns: int
    observation: tuple[float, ...]
    observation_sha256: str


def observation_dimension(spec: AgentSpec) -> int:
    return spec.observation_dimension


def _stable_array_hash(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def encode_window(
    window: pd.DataFrame,
    *,
    inherited_positions: Sequence[int] = (),
) -> np.ndarray:
    """Encode a completed OHLCV window without using out-of-window statistics.

    Prices are log returns against the preceding close (the first open for the
    first row). Log-volume is standardized inside the already completed window.
    This is a registered reconstruction default, not a detail supplied by the
    base paper.
    """

    if window.empty:
        raise ValueError("cannot encode an empty window")
    values = window.loc[:, BAR_FEATURES].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("state window contains non-finite OHLCV")
    if np.any(values[:, :4] <= 0) or np.any(values[:, 4] < 0):
        raise ValueError("state window contains invalid price or volume")
    previous_close = np.concatenate(([values[0, 0]], values[:-1, 3]))
    prices = np.log(values[:, :4] / previous_close[:, None])
    log_volume = np.log1p(values[:, 4])
    scale = float(log_volume.std(ddof=0))
    volume = (
        np.zeros_like(log_volume)
        if scale <= np.finfo(np.float64).eps
        else (log_volume - float(log_volume.mean())) / scale
    )
    encoded = np.column_stack([prices, volume]).reshape(-1)
    inherited = np.asarray(tuple(inherited_positions), dtype=np.float64)
    if inherited.size and not np.isin(inherited, ACTIONS).all():
        raise ValueError(f"inherited positions must be in {ACTIONS}")
    return np.concatenate([encoded, inherited]).astype(np.float32)


def _consecutive(window: pd.DataFrame, minutes: int) -> bool:
    ends = window["bar_end_ns"].to_numpy(dtype=np.int64)
    if len(ends) <= 1:
        return True
    return all(
        is_registered_globex_bar_step(previous, current, minutes)
        for previous, current in zip(ends[:-1], ends[1:], strict=True)
    )


def build_state_records(
    bars: pd.DataFrame,
    spec: AgentSpec,
    *,
    inherited: Mapping[str, pd.Series | Sequence[int]] | None = None,
) -> list[StateRecord]:
    """Construct fully warmed causal states inside explicit state boundaries.

    Inherited position arrays must already have been backward-as-of aligned to
    the bar's ``available_time_ns``.  Their names and order are checked against
    the agent contract.
    """

    required = {*BOUNDARY_COLUMNS, "bar_end_ns", "available_time_ns", *BAR_FEATURES}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise KeyError(f"state bars are missing columns: {missing}")
    inherited = {} if inherited is None else dict(inherited)
    if tuple(inherited) != spec.inherited_positions:
        raise ValueError(
            f"{spec.name} inherited fields must be ordered {spec.inherited_positions}; "
            f"got {tuple(inherited)}"
        )
    aligned: dict[str, np.ndarray] = {}
    for name, values in inherited.items():
        series = pd.Series(values, index=bars.index)
        if series.isna().any():
            raise ValueError(f"{name} has unavailable positions; states must fail closed")
        array = series.to_numpy(dtype=np.int8)
        if not np.isin(array, ACTIONS).all():
            raise ValueError(f"{name} contains an invalid inherited position")
        aligned[name] = array

    frame = bars.copy()
    frame["bar_end_ns"] = pd.to_numeric(frame["bar_end_ns"], errors="raise").astype("int64")
    frame["available_time_ns"] = pd.to_numeric(
        frame["available_time_ns"], errors="raise"
    ).astype("int64")
    if bool((frame["available_time_ns"] < frame["bar_end_ns"]).any()):
        raise ValueError("a state bar cannot be available before it completes")
    if bool(frame.duplicated([*BOUNDARY_COLUMNS, "bar_end_ns"]).any()):
        raise ValueError("duplicate agent bar inside a state boundary")
    frame["__source_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values([*BOUNDARY_COLUMNS, "bar_end_ns"], kind="mergesort")
    records: list[StateRecord] = []
    for _, group in frame.groupby(list(BOUNDARY_COLUMNS), sort=False, dropna=False):
        group = group.reset_index(drop=True)
        for end in range(spec.lookback - 1, len(group)):
            window = group.iloc[end - spec.lookback + 1 : end + 1]
            if not _consecutive(window, spec.minutes):
                continue
            source_index = int(group.iloc[end]["__source_index"])
            positions = tuple(int(aligned[name][source_index]) for name in spec.inherited_positions)
            observation = encode_window(window, inherited_positions=positions)
            if observation.size != observation_dimension(spec):
                raise AssertionError("state encoder dimension differs from contract")
            row = group.iloc[end]
            records.append(
                StateRecord(
                    agent=spec.name,
                    raw_symbol=str(row["raw_symbol"]),
                    continuity_segment_id=int(row["continuity_segment_id"]),
                    state_segment_id=str(row["state_segment_id"]),
                    bar_end_ns=int(row["bar_end_ns"]),
                    available_time_ns=int(window["available_time_ns"].max()),
                    observation=tuple(float(v) for v in observation),
                    observation_sha256=_stable_array_hash(observation),
                )
            )
    return records


def build_decision_calendar(timeframes: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Order completed-bar events with deterministic top-down same-clock priority."""

    if set(timeframes) != {"H1", "M15", "M5"}:
        raise ValueError("decision calendar requires exactly H1, M15, and M5 frames")
    priority = {"H1": 0, "M15": 1, "M5": 2}
    parts: list[pd.DataFrame] = []
    for agent in ("H1", "M15", "M5"):
        frame = timeframes[agent]
        required = {*BOUNDARY_COLUMNS, "bar_end_ns", "available_time_ns"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{agent} calendar bars are missing columns: {missing}")
        part = frame.loc[:, [*BOUNDARY_COLUMNS, "bar_end_ns", "available_time_ns"]].copy()
        part["agent"] = agent
        part["agent_priority"] = priority[agent]
        parts.append(part)
    calendar = pd.concat(parts, ignore_index=True)
    return calendar.sort_values(
        ["available_time_ns", "agent_priority", "bar_end_ns", "raw_symbol"],
        kind="mergesort",
    ).reset_index(drop=True)


def align_inherited_position(
    decision_bars: pd.DataFrame,
    slower_trace: pd.DataFrame,
    *,
    position_column: str,
) -> pd.Series:
    """Backward-as-of align one slower position inside matching boundaries."""

    required_trace = {*BOUNDARY_COLUMNS, "available_time_ns", position_column}
    missing = sorted(required_trace - set(slower_trace.columns))
    if missing:
        raise KeyError(f"slower trace is missing columns: {missing}")
    result = pd.Series(np.nan, index=decision_bars.index, dtype=float)
    grouped = slower_trace.groupby(list(BOUNDARY_COLUMNS), sort=False, dropna=False)
    lookup = {key if isinstance(key, tuple) else (key,): value for key, value in grouped}
    for key, decisions in decision_bars.groupby(
        list(BOUNDARY_COLUMNS), sort=False, dropna=False
    ):
        tuple_key = key if isinstance(key, tuple) else (key,)
        source = lookup.get(tuple_key)
        if source is None:
            continue
        source = source.sort_values("available_time_ns", kind="mergesort")
        source_times = source["available_time_ns"].to_numpy(dtype=np.int64)
        source_values = source[position_column].to_numpy(dtype=np.int8)
        for index, available in decisions["available_time_ns"].items():
            location = int(np.searchsorted(source_times, int(available), side="right") - 1)
            if location >= 0:
                result.at[index] = int(source_values[location])
    return result
