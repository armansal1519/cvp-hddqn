from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import enforce_2025_frame
from .globex import globex_session_date_at_ns


def assign_globex_session(ts_event: pd.Series) -> pd.Series:
    utc = pd.to_datetime(ts_event, utc=True)
    values: list[object] = []
    for timestamp in utc:
        try:
            values.append(date.fromisoformat(globex_session_date_at_ns(int(timestamp.value))))
        except ValueError as error:
            if "maintenance break" not in str(error):
                raise
            values.append(None)
    return pd.Series(values, index=ts_event.index, dtype="object")


def daily_contract_volume(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    timestamp = frame["ts_event"] if "ts_event" in frame else pd.to_datetime(frame["ts_event_ns"], unit="ns", utc=True)
    frame["session_date"] = assign_globex_session(timestamp)
    frame = frame[frame["session_date"].notna()].copy()
    return (
        frame.groupby(["session_date", "raw_symbol"], as_index=False)["size"]
        .sum()
        .rename(columns={"size": "executed_volume"})
        .sort_values(["session_date", "raw_symbol"], kind="mergesort")
    )


def select_active_contracts(
    volume: pd.DataFrame,
    *,
    lookback_sessions: int = 5,
    challenger_ratio: float = 1.2,
) -> pd.DataFrame:
    pivot = volume.pivot(index="session_date", columns="raw_symbol", values="executed_volume").fillna(0).sort_index()
    trailing = pivot.shift(1).rolling(lookback_sessions, min_periods=lookback_sessions).sum()
    incumbent: str | None = None
    rows: list[dict[str, Any]] = []
    for session_date, scores in trailing.iterrows():
        if not np.isfinite(scores.max()) or scores.max() <= 0:
            continue
        leader = str(scores.idxmax())
        if incumbent is None or scores.get(incumbent, 0.0) <= 0:
            selected = leader
        elif leader != incumbent and scores[leader] >= challenger_ratio * scores[incumbent]:
            selected = leader
        else:
            selected = incumbent
        rows.append(
            {
                "date": str(session_date),
                "raw_symbol": selected,
                "leader": leader,
                "selected_trailing_volume": float(scores[selected]),
                "leader_trailing_volume": float(scores[leader]),
                "roll": incumbent is not None and selected != incumbent,
            }
        )
        incumbent = selected
    return pd.DataFrame(rows)


def build_trade_bars(trades: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes not in {5, 15, 60}:
        raise ValueError("GC-lite supports only 5, 15, and 60 minute bars")
    frame = trades.copy()
    frame["event_time"] = pd.to_datetime(frame["ts_event_ns"], unit="ns", utc=True)
    frame["receive_time"] = pd.to_datetime(frame["ts_recv_ns"], unit="ns", utc=True)
    frame = frame.sort_values(["raw_symbol", "event_time", "global_record_ordinal"], kind="mergesort")
    grouped = frame.set_index("event_time").groupby("raw_symbol", sort=False)
    bars = grouped["price_ticks"].resample(f"{minutes}min", label="right", closed="left").ohlc()
    volume = grouped["size"].resample(f"{minutes}min", label="right", closed="left").sum().rename("volume")
    available = grouped["receive_time"].resample(f"{minutes}min", label="right", closed="left").max().rename("last_receive")
    result = pd.concat([bars, volume, available], axis=1).dropna(subset=["close"]).reset_index()
    result["bar_end"] = result["event_time"]
    boundary_ns = result["bar_end"].astype("int64")
    receive_ns = result["last_receive"].astype("int64")
    result["available_time_ns"] = np.maximum(boundary_ns, receive_ns) + 1_000_000
    result["date"] = result["bar_end"].dt.date.astype(str)
    return enforce_2025_frame(result)


def _signal(frame: pd.DataFrame, fast: int, slow: int, name: str) -> pd.DataFrame:
    keys = ["raw_symbol"]
    if "continuity_segment_id" in frame:
        keys.append("continuity_segment_id")
    result = frame[[*keys, "bar_end", "available_time_ns", "close"]].copy()
    fast_ema = result.groupby(keys, sort=False)["close"].transform(lambda values: values.ewm(span=fast, adjust=False, min_periods=slow).mean())
    slow_ema = result.groupby(keys, sort=False)["close"].transform(lambda values: values.ewm(span=slow, adjust=False, min_periods=slow).mean())
    result[name] = np.sign(fast_ema - slow_ema).fillna(0).astype(np.int8)
    return result


def generate_gc_lite_trace(
    bars_5m: pd.DataFrame,
    bars_15m: pd.DataFrame,
    bars_1h: pd.DataFrame,
    *,
    maximum_holding_m5_bars: int = 20,
    m5_fast_ema: int = 5,
    m5_slow_ema: int = 13,
    m15_fast_ema: int = 8,
    m15_slow_ema: int = 21,
    h1_fast_ema: int = 12,
    h1_slow_ema: int = 36,
) -> pd.DataFrame:
    spans = {
        "m5": (m5_fast_ema, m5_slow_ema),
        "m15": (m15_fast_ema, m15_slow_ema),
        "h1": (h1_fast_ema, h1_slow_ema),
    }
    for timeframe, (fast, slow) in spans.items():
        if not 1 <= int(fast) < int(slow):
            raise ValueError(
                f"{timeframe} EMA spans must satisfy 1 <= fast < slow; got {fast}, {slow}"
            )
    by = ["raw_symbol"]
    if "continuity_segment_id" in bars_5m:
        by.append("continuity_segment_id")
    sort_keys = ["available_time_ns", *by]
    m5 = _signal(bars_5m, m5_fast_ema, m5_slow_ema, "m5_signal").sort_values(
        sort_keys, kind="mergesort"
    )
    m15 = _signal(bars_15m, m15_fast_ema, m15_slow_ema, "m15_signal").sort_values(
        sort_keys, kind="mergesort"
    )
    h1 = _signal(bars_1h, h1_fast_ema, h1_slow_ema, "h1_signal").sort_values(
        sort_keys, kind="mergesort"
    )
    merged = pd.merge_asof(
        m5,
        m15[[*by, "available_time_ns", "m15_signal"]],
        on="available_time_ns",
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    merged = pd.merge_asof(
        merged,
        h1[[*by, "available_time_ns", "h1_signal"]],
        on="available_time_ns",
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    aligned = (
        merged["m5_signal"].ne(0)
        & merged["m5_signal"].eq(merged["m15_signal"])
        & merged["m5_signal"].eq(merged["h1_signal"])
    )
    desired = merged["m5_signal"].where(aligned, 0).fillna(0).astype(int)
    positions: dict[tuple[str, int | None], int] = {}
    holding_bars: dict[tuple[str, int | None], int] = {}
    targets: list[int] = []
    segments = (
        merged["continuity_segment_id"]
        if "continuity_segment_id" in merged
        else pd.Series([None] * len(merged), index=merged.index)
    )
    for symbol, segment, value in zip(
        merged["raw_symbol"], segments, desired, strict=True
    ):
        key = (str(symbol), None if segment is None else int(segment))
        position = positions.get(key, 0)
        held = holding_bars.get(key, 0)
        if position == 0:
            position = int(value)
            held = 1 if position else 0
        else:
            held += 1
            if value != position or held >= maximum_holding_m5_bars:
                position = 0
                held = 0
        positions[key] = position
        holding_bars[key] = held
        targets.append(position)
    merged["target"] = targets
    merged["date"] = pd.to_datetime(merged["bar_end"], utc=True).dt.date.astype(str)
    merged["decision_id"] = [f"gc-lite-{index:09d}" for index in range(len(merged))]
    return enforce_2025_frame(merged)


def extract_gc_lite_episodes(trace: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}
    previous: dict[str, int] = {}
    episode_counter = 0
    for row in trace.itertuples(index=False):
        symbol = str(row.raw_symbol)
        segment = int(getattr(row, "continuity_segment_id", 0))
        state_key = f"{symbol}|{segment}"
        target = int(row.target)
        prior = previous.get(state_key, 0)
        if prior == 0 and target != 0:
            active[state_key] = {
                "opportunity_id": f"episode-{episode_counter:08d}",
                "date": str(row.date),
                "raw_symbol": symbol,
                "continuity_segment_id": segment,
                "direction": target,
                "start_available_time_ns": int(row.available_time_ns),
                "start_decision_id": str(row.decision_id),
            }
            episode_counter += 1
        elif prior != 0 and target == 0 and state_key in active:
            episode = active.pop(state_key)
            episode["end_available_time_ns"] = int(row.available_time_ns)
            episode["end_decision_id"] = str(row.decision_id)
            rows.append(episode)
        previous[state_key] = target
    return pd.DataFrame(rows)
