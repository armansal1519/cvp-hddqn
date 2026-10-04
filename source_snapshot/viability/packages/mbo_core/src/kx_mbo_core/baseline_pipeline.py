from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd

from .bars import (
    assign_globex_session,
    generate_gc_lite_trace,
    select_active_contracts,
)
from .config import OUTCOME_END, OUTCOME_START, enforce_2025_frame, enforce_outcome_dates


SCREENING_LABEL = "SCREENING_ONLY"
DEVELOPMENT_LABEL = "DEVELOPMENT_ONLY"
DEFAULT_DEVELOPMENT_END = date(2025, 8, 31)
FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000
ONE_MILLISECOND_NS = 1_000_000
SESSION_FLATTEN_RULE_ID = "fixed_research_cutoff_1225_ct_next_5m_bar_v1"


@dataclass(frozen=True)
class ScreeningResult:
    """A bounded baseline result; it is never a confirmatory viability verdict."""

    report: dict[str, Any]
    trace: pd.DataFrame
    episodes: pd.DataFrame


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _globex_session_close_utc(session_date: date) -> pd.Timestamp:
    local = pd.Timestamp(f"{session_date} 16:00:00", tz="America/Chicago")
    return local.tz_convert("UTC")


def _globex_session_open_utc(session_date: date) -> pd.Timestamp:
    prior_date = session_date - timedelta(days=1)
    local = pd.Timestamp(f"{prior_date} 17:00:00", tz="America/Chicago")
    return local.tz_convert("UTC")


def _session_flattening_spec(gc_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed research cutoff known before every 2025 session."""

    nested = gc_config.get("session_flattening", {})
    if not isinstance(nested, Mapping):
        raise TypeError("session_flattening must be a mapping")
    rule_id = str(nested.get("rule_id", SESSION_FLATTEN_RULE_ID))
    timezone = str(nested.get("timezone", "America/Chicago"))
    decision_cutoff = str(nested.get("decision_cutoff", "12:25"))
    proxy_fill_cutoff = str(nested.get("proxy_fill_cutoff", "12:30"))
    if rule_id != SESSION_FLATTEN_RULE_ID:
        raise ValueError(
            f"session flattening rule_id must be {SESSION_FLATTEN_RULE_ID!r}"
        )
    try:
        decision = pd.Timestamp(f"2025-01-02 {decision_cutoff}", tz=timezone)
        proxy_fill = pd.Timestamp(f"2025-01-02 {proxy_fill_cutoff}", tz=timezone)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid fixed session flattening cutoff") from exc
    if int(proxy_fill.value - decision.value) != FIVE_MINUTES_NS:
        raise ValueError(
            "proxy fill cutoff must be exactly five minutes after decision cutoff"
        )
    return {
        "rule_id": rule_id,
        "timezone": timezone,
        "decision_cutoff": decision_cutoff,
        "proxy_fill_cutoff": proxy_fill_cutoff,
        "basis": "conservative_research_cutoff_not_exchange_close",
    }


def _research_session_cutoffs_utc(
    session_date: Any,
    *,
    session_flattening: Mapping[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = _as_date(session_date)
    timezone = str(session_flattening["timezone"])
    decision = pd.Timestamp(
        f"{day} {session_flattening['decision_cutoff']}", tz=timezone
    )
    proxy_fill = pd.Timestamp(
        f"{day} {session_flattening['proxy_fill_cutoff']}", tz=timezone
    )
    return decision.tz_convert("UTC"), proxy_fill.tz_convert("UTC")


def development_end_from_config(config: dict[str, Any]) -> date:
    folds = config.get("development_folds", [])
    if not folds:
        raise ValueError("split config has no development_folds")
    boundary = max(_as_date(fold["validation_end"]) for fold in folds)
    enforce_outcome_dates(OUTCOME_START, boundary)
    calibration = config.get("calibration")
    if calibration and boundary >= _as_date(calibration["start"]):
        raise ValueError("development boundary overlaps calibration")
    return boundary


def sealed_l1_paths(archive_root: str | Path, *, through: Any) -> list[Path]:
    """Return only explicit 2025 L1 files through a UTC partition date.

    The function deliberately starts at ``year=2025`` instead of globbing the
    layer root.  It therefore cannot discover, pass to DuckDB, or scan a sealed
    2026 file.
    """

    through_date = _as_date(through)
    enforce_outcome_dates(OUTCOME_START, through_date)
    year_root = Path(archive_root) / "l1_trades" / "year=2025"
    if not year_root.is_dir():
        raise FileNotFoundError(f"missing sealed L1 root: {year_root}")

    paths: list[Path] = []
    for path in year_root.glob("month=*/day=*/part-*.parquet"):
        try:
            month = int(path.parent.parent.name.removeprefix("month="))
            day = int(path.parent.name.removeprefix("day="))
            partition_date = date(2025, month, day)
        except ValueError as exc:
            raise ValueError(f"invalid L1 partition path: {path}") from exc
        if partition_date <= through_date:
            paths.append(path.resolve())
    if not paths:
        raise FileNotFoundError(f"no 2025 L1 trade files found through {through_date}")
    paths.sort()
    if any("year=2025" not in path.parts for path in paths):
        raise PermissionError("sealed-path invariant failed: non-2025 input discovered")
    return paths


def aggregate_l1_to_5m(
    paths: Iterable[str | Path],
    *,
    through: Any,
    excluded_utc_dates: Iterable[Any] = (),
) -> pd.DataFrame:
    """Aggregate L1 trades inside DuckDB and materialize bars, not trade rows."""

    through_date = _as_date(through)
    enforce_outcome_dates(OUTCOME_START, through_date)
    sources = [str(Path(path)) for path in paths]
    if not sources:
        raise ValueError("at least one 2025 L1 path is required")
    if any("year=2025" not in Path(path).parts for path in sources):
        raise PermissionError("aggregate_l1_to_5m accepts only explicit year=2025 paths")

    start_ns = int(pd.Timestamp(OUTCOME_START, tz="UTC").value)
    # A Globex session labelled D closes at 16:00 Chicago on D.  UTC midnight
    # would leak the first two or three hours of session D+1 into a D screen.
    end_ns = int(_globex_session_close_utc(through_date).value)
    query = f"""
        WITH eligible AS (
            SELECT
                raw_symbol,
                continuity_segment_id,
                ts_event_ns,
                ts_recv_ns,
                global_record_ordinal,
                price_ticks,
                size,
                ts_event_ns - (ts_event_ns % {FIVE_MINUTES_NS}) AS bucket_start_ns
            FROM read_parquet(?)
            WHERE ts_event_ns >= ?
              AND ts_event_ns < ?
              AND price_ticks IS NOT NULL
              AND size > 0
              AND NOT is_bad_ts_recv
        )
        SELECT
            raw_symbol,
            continuity_segment_id,
            bucket_start_ns + {FIVE_MINUTES_NS} AS bar_end_ns,
            first(price_ticks ORDER BY ts_event_ns, global_record_ordinal) AS open,
            max(price_ticks) AS high,
            min(price_ticks) AS low,
            last(price_ticks ORDER BY ts_event_ns, global_record_ordinal) AS close,
            sum(size) AS volume,
            max(ts_recv_ns) AS last_receive_ns
        FROM eligible
        GROUP BY raw_symbol, continuity_segment_id, bucket_start_ns
        ORDER BY bar_end_ns, raw_symbol, continuity_segment_id
    """
    connection = duckdb.connect()
    try:
        bars = connection.execute(query, [sources, start_ns, end_ns]).fetchdf()
    finally:
        connection.close()
    if bars.empty:
        raise ValueError("the selected 2025 L1 inputs contain no eligible trades")

    bars["bar_end"] = pd.to_datetime(bars.pop("bar_end_ns"), unit="ns", utc=True)
    boundary_ns = bars["bar_end"].astype("int64")
    receive_ns = bars.pop("last_receive_ns").astype("int64")
    bars["available_time_ns"] = np.maximum(boundary_ns, receive_ns) + ONE_MILLISECOND_NS
    bars["date"] = (bars["bar_end"] - pd.Timedelta(1, unit="ns")).dt.date.astype(str)

    excluded = {str(_as_date(value)) for value in excluded_utc_dates}
    if excluded:
        bars = bars.loc[~bars["date"].isin(excluded)].copy()
    for column in ("open", "high", "low", "close", "volume"):
        bars[column] = pd.to_numeric(bars[column], errors="raise")
    return enforce_2025_frame(bars).reset_index(drop=True)


def _rollup_bars(bars_5m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes not in {15, 60}:
        raise ValueError("rollup supports only 15- and 60-minute bars")
    interval_ns = minutes * 60 * 1_000_000_000
    source = bars_5m.copy()
    end_ns = source["bar_end"].astype("int64")
    # The minus-one maps a right-labelled 5m bar to the interval it closes.
    source["target_end_ns"] = ((end_ns - 1) // interval_ns + 1) * interval_ns
    source = source.sort_values(
        ["raw_symbol", "continuity_segment_id", "bar_end"], kind="mergesort"
    )
    result = (
        source.groupby(
            ["raw_symbol", "continuity_segment_id", "target_end_ns"],
            sort=False,
            as_index=False,
        )
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            source_available_time_ns=("available_time_ns", "max"),
        )
    )
    result["bar_end"] = pd.to_datetime(result.pop("target_end_ns"), unit="ns", utc=True)
    boundary_ns = result["bar_end"].astype("int64")
    result["available_time_ns"] = np.maximum(
        boundary_ns, result.pop("source_available_time_ns").astype("int64")
    )
    result["date"] = (result["bar_end"] - pd.Timedelta(1, unit="ns")).dt.date.astype(str)
    return enforce_2025_frame(result).sort_values(
        ["bar_end", "raw_symbol", "continuity_segment_id"], kind="mergesort"
    ).reset_index(drop=True)


def _volume_and_active_contracts(
    bars_5m: pd.DataFrame,
    *,
    lookback_sessions: int,
    challenger_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = bars_5m.copy()
    representative_time = frame["bar_end"] - pd.Timedelta(1, unit="ns")
    frame["session_date"] = assign_globex_session(representative_time)
    frame = frame.loc[frame["session_date"].notna()].copy()
    frame = frame.loc[
        frame["session_date"].map(lambda value: OUTCOME_START <= value <= OUTCOME_END)
    ].copy()
    volume = (
        frame.groupby(["session_date", "raw_symbol"], as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "executed_volume"})
        .sort_values(["session_date", "raw_symbol"], kind="mergesort")
    )
    active = select_active_contracts(
        volume,
        lookback_sessions=lookback_sessions,
        challenger_ratio=challenger_ratio,
    )
    return volume, active


def _only_active_bars(bars: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["session_date"] = assign_globex_session(
        frame["bar_end"] - pd.Timedelta(1, unit="ns")
    )
    chosen = active[["date", "raw_symbol"]].rename(
        columns={"date": "session_date", "raw_symbol": "active_raw_symbol"}
    )
    chosen["session_date"] = chosen["session_date"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame = frame.merge(chosen, on="session_date", how="inner", validate="many_to_one")
    frame = frame.loc[frame["raw_symbol"] == frame["active_raw_symbol"]].drop(
        columns="active_raw_symbol"
    )
    return frame.sort_values(["bar_end", "raw_symbol"], kind="mergesort").reset_index(drop=True)


def _apply_session_flat_positions(
    trace: pd.DataFrame,
    *,
    maximum_holding_m5_bars: int,
    session_flattening: Mapping[str, Any] | None = None,
    require_complete_sessions: bool = True,
) -> pd.DataFrame:
    """Rebuild targets and flatten at a fixed cutoff known before the session.

    The former implementation selected the penultimate *observed* row after the
    full session had been loaded.  That made a truncated or gapped feed change an
    earlier decision.  Here the decision is fixed at 12:25 America/Chicago with
    a common 12:30 proxy-fill horizon.  This conservative research cutoff is not
    represented as an official exchange close.
    """

    result = trace.sort_values(
        ["available_time_ns", "raw_symbol", "continuity_segment_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    aligned = (
        result["m5_signal"].ne(0)
        & result["m5_signal"].eq(result["m15_signal"])
        & result["m5_signal"].eq(result["h1_signal"])
    )
    desired = result["m5_signal"].where(aligned, 0).fillna(0).astype(int)
    positions: list[int] = []
    position = 0
    held = 0
    prior_session: str | None = None
    prior_segment: int | None = None
    for session, segment, signal in zip(
        result["session_date"],
        result["continuity_segment_id"],
        desired,
        strict=True,
    ):
        session = str(session)
        segment = int(segment)
        if prior_session is not None and (
            session != prior_session or segment != prior_segment
        ):
            position = 0
            held = 0
        if position == 0:
            position = int(signal)
            held = 1 if position else 0
        else:
            held += 1
            if signal != position or held >= maximum_holding_m5_bars:
                position = 0
                held = 0
        positions.append(position)
        prior_session = session
        prior_segment = segment

    result["target"] = positions
    flattening = dict(session_flattening or _session_flattening_spec({}))
    cutoffs_by_session = {
        str(session): _research_session_cutoffs_utc(
            session,
            session_flattening=flattening,
        )
        for session in result["session_date"].astype(str).unique()
    }
    flatten_bar_end_ns = result["session_date"].astype(str).map(
        lambda session: int(cutoffs_by_session[session][0].value)
    )
    proxy_fill_cutoff_ns = result["session_date"].astype(str).map(
        lambda session: int(cutoffs_by_session[session][1].value)
    )
    observed_bar_end_ns = result["bar_end"].astype("int64")
    if require_complete_sessions:
        for session, (decision, proxy_fill) in cutoffs_by_session.items():
            group = result.loc[result["session_date"].astype(str).eq(session)]
            ends = set(group["bar_end"].astype("int64"))
            required = {int(decision.value), int(proxy_fill.value)}
            missing = sorted(required.difference(ends))
            if missing:
                rendered = [
                    str(pd.Timestamp(value, unit="ns", tz="UTC"))
                    for value in missing
                ]
                raise ValueError(
                    f"session {session} lacks required ex-ante flatten/fill bars: {rendered}"
                )
    result.loc[observed_bar_end_ns >= flatten_bar_end_ns, "target"] = 0
    result["scheduled_flatten_bar_end_ns"] = flatten_bar_end_ns.astype("int64")
    result["research_proxy_fill_cutoff_time_ns"] = proxy_fill_cutoff_ns.astype(
        "int64"
    )
    result["session_flatten_rule_id"] = SESSION_FLATTEN_RULE_ID
    result["date"] = result["session_date"].astype(str)
    return enforce_2025_frame(result)


def _proxy_episodes(
    trace: pd.DataFrame,
    *,
    adverse_slippage_ticks_per_side: int,
    tick_value_usd: float,
    fee_usd_per_side: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    prior_target = 0
    ordered = trace.sort_values(
        ["available_time_ns", "raw_symbol", "continuity_segment_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    for index, row in ordered.iterrows():
        target = int(row["target"])
        if target == prior_target:
            continue
        if index + 1 >= len(ordered):
            break
        next_row = ordered.iloc[index + 1]
        same_instrument_window = (
            str(next_row["raw_symbol"]) == str(row["raw_symbol"])
            and str(next_row["session_date"]) == str(row["session_date"])
            and int(next_row["continuity_segment_id"])
            == int(row["continuity_segment_id"])
        )
        if not same_instrument_window:
            prior_target = target
            active = None
            continue
        # Waiting for the subsequent bar to complete makes the proxy timestamp
        # strictly later than the decision's availability timestamp.  The close
        # is still only a trade-price proxy, not a claim of executable liquidity.
        next_price = float(next_row["close"])
        if prior_target == 0 and target != 0:
            active = {
                "opportunity_id": f"gc-lite-screen-{len(rows):08d}",
                "date": str(row["date"]),
                "raw_symbol": str(row["raw_symbol"]),
                "continuity_segment_id": int(row["continuity_segment_id"]),
                "direction": target,
                "entry_decision_available_time_ns": int(row["available_time_ns"]),
                "entry_fill_time_ns": int(next_row["available_time_ns"]),
                "entry_proxy_ticks": next_price + target * adverse_slippage_ticks_per_side,
            }
        elif prior_target != 0 and target == 0 and active is not None:
            direction = int(active["direction"])
            exit_ticks = next_price - direction * adverse_slippage_ticks_per_side
            pnl_ticks = direction * (exit_ticks - float(active["entry_proxy_ticks"]))
            slippage_adjusted_usd = pnl_ticks * tick_value_usd
            net_usd = slippage_adjusted_usd - 2.0 * fee_usd_per_side
            active.update(
                {
                    "exit_decision_available_time_ns": int(row["available_time_ns"]),
                    "exit_fill_time_ns": int(next_row["available_time_ns"]),
                    "exit_proxy_ticks": exit_ticks,
                    "pnl_ticks_after_slippage": pnl_ticks,
                    "pnl_usd_after_slippage": slippage_adjusted_usd,
                    "fees_usd": 2.0 * fee_usd_per_side,
                    "net_proxy_pnl_usd": net_usd,
                    "evidence_class": SCREENING_LABEL,
                }
            )
            rows.append(active)
            active = None
        prior_target = target
    columns = [
        "opportunity_id",
        "date",
        "raw_symbol",
        "continuity_segment_id",
        "direction",
        "entry_decision_available_time_ns",
        "entry_fill_time_ns",
        "entry_proxy_ticks",
        "exit_decision_available_time_ns",
        "exit_fill_time_ns",
        "exit_proxy_ticks",
        "pnl_ticks_after_slippage",
        "pnl_usd_after_slippage",
        "fees_usd",
        "net_proxy_pnl_usd",
        "evidence_class",
    ]
    return pd.DataFrame(rows, columns=columns)


def _pnl_summary(episodes: pd.DataFrame, session_dates: pd.Series) -> dict[str, Any]:
    if episodes.empty:
        return {
            "completed_trades": 0,
            "total_net_proxy_pnl_usd": 0.0,
            "mean_net_proxy_pnl_usd": None,
            "win_rate": None,
            "profit_factor": None,
            "annualized_session_sharpe_proxy": None,
            "maximum_drawdown_usd": 0.0,
        }
    net = episodes["net_proxy_pnl_usd"].astype(float)
    positive = float(net.loc[net > 0].sum())
    negative = float(-net.loc[net < 0].sum())
    daily = episodes.groupby("date")["net_proxy_pnl_usd"].sum()
    session_index = pd.Index(sorted(set(session_dates.astype(str))), dtype="object")
    daily = daily.reindex(session_index, fill_value=0.0).astype(float)
    standard_deviation = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = (
        float(np.sqrt(252.0) * daily.mean() / standard_deviation)
        if standard_deviation > 0
        else None
    )
    equity = net.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    return {
        "completed_trades": int(len(episodes)),
        "total_net_proxy_pnl_usd": float(net.sum()),
        "mean_net_proxy_pnl_usd": float(net.mean()),
        "win_rate": float((net > 0).mean()),
        "profit_factor": positive / negative if negative > 0 else None,
        "annualized_session_sharpe_proxy": sharpe,
        "maximum_drawdown_usd": float(drawdown.min()),
    }


def run_gc_lite_screening(
    archive_root: str | Path,
    *,
    start: Any,
    end: Any,
    gc_config: dict[str, Any],
    execution_config: dict[str, Any],
    degraded_dates: Iterable[Any] = (),
    development_end: Any = DEFAULT_DEVELOPMENT_END,
) -> ScreeningResult:
    """Run deterministic, non-confirmatory GC-lite screening on sealed 2025 L1."""

    start_date = _as_date(start)
    end_date = _as_date(end)
    enforce_outcome_dates(start_date, end_date)
    development_end_date = _as_date(development_end)
    if end_date > development_end_date:
        raise PermissionError(
            "GC-lite screening is development-only; requested end "
            f"{end_date} is after {development_end_date}. Calibration and locked "
            "viability outcomes must remain unopened."
        )
    paths = sealed_l1_paths(archive_root, through=end_date)
    session_flattening = _session_flattening_spec(gc_config)
    bars_5m = aggregate_l1_to_5m(
        paths,
        through=end_date,
        excluded_utc_dates=degraded_dates,
    )
    bars_15m = _rollup_bars(bars_5m, 15)
    bars_1h = _rollup_bars(bars_5m, 60)

    roll = gc_config["roll"]
    _, active = _volume_and_active_contracts(
        bars_5m,
        lookback_sessions=int(roll["volume_lookback_sessions"]),
        challenger_ratio=float(roll["challenger_ratio"]),
    )
    if active.empty:
        raise ValueError("insufficient sessions to select an active contract")
    signals = gc_config["signals"]
    signal_parameters = {
        "h1_fast_ema": int(signals["h1_fast_ema"]),
        "h1_slow_ema": int(signals["h1_slow_ema"]),
        "m15_fast_ema": int(signals["m15_fast_ema"]),
        "m15_slow_ema": int(signals["m15_slow_ema"]),
        "m5_fast_ema": int(signals["m5_fast_ema"]),
        "m5_slow_ema": int(signals["m5_slow_ema"]),
        "maximum_holding_m5_bars": int(signals["maximum_holding_m5_bars"]),
    }
    trace = generate_gc_lite_trace(
        bars_5m,
        bars_15m,
        bars_1h,
        **signal_parameters,
    )
    # Indicators are warmed causally for every listed contract before applying
    # the lagged-volume active map.  Otherwise an incoming contract would lose
    # its observable pre-roll history and spuriously need 36 active H1 bars.
    trace = _only_active_bars(trace, active)
    if trace.empty:
        raise ValueError("active-contract selection produced an empty trace")
    trace = trace.merge(
        bars_5m[
            [
                "raw_symbol",
                "continuity_segment_id",
                "bar_end",
                "open",
                "high",
                "low",
                "volume",
            ]
        ],
        on=["raw_symbol", "continuity_segment_id", "bar_end"],
        how="left",
        validate="one_to_one",
    )
    trace = _apply_session_flat_positions(
        trace,
        maximum_holding_m5_bars=signal_parameters["maximum_holding_m5_bars"],
        session_flattening=session_flattening,
    )
    trace = trace.loc[
        trace["session_date"].astype(str).between(str(start_date), str(end_date))
    ].copy()
    trace["evidence_class"] = SCREENING_LABEL
    trace = enforce_2025_frame(trace).reset_index(drop=True)

    fill_config = execution_config["fills"]
    cost_config = execution_config["costs"]
    instrument_config = execution_config["instrument"]
    episodes = _proxy_episodes(
        trace,
        adverse_slippage_ticks_per_side=int(fill_config["aggressive_stress_slippage_ticks"]),
        tick_value_usd=float(instrument_config["tick_value_usd"]),
        fee_usd_per_side=float(cost_config["fee_usd_per_contract_side"]),
    )
    if not episodes.empty:
        episodes = enforce_2025_frame(episodes)

    entry_count = int(
        ((trace["target"] != 0) & (trace["target"].shift(fill_value=0) == 0)).sum()
    )
    active_in_range = active.loc[active["date"].between(str(start_date), str(end_date))]
    report = {
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "warning": (
            "GC-lite uses aggregated L1 trade prices and a next-completed-bar proxy; "
            "it is not Base-GC, MBO execution evidence, or a novelty verdict."
        ),
        "source_layer": "l1_trades",
        "aggregation_pushdown": "duckdb_5m_ohlcv",
        "sealed_2026_read": False,
        "source_partition_years": [2025],
        "scan_start": str(OUTCOME_START),
        "requested_start": str(start_date),
        "requested_end": str(end_date),
        "development_end": str(development_end_date),
        "event_time_end_exclusive_utc": str(_globex_session_close_utc(end_date)),
        "input_parquet_files": len(paths),
        "excluded_degraded_utc_dates": sorted(
            str(_as_date(value))
            for value in degraded_dates
            if _as_date(value) <= end_date
        ),
        "aggregate_5m_bars": int(len(bars_5m)),
        "active_sessions": int(active_in_range["date"].nunique()),
        "active_contracts": sorted(active_in_range["raw_symbol"].astype(str).unique().tolist()),
        "continuity_segments": sorted(
            int(value) for value in trace["continuity_segment_id"].unique()
        ),
        "episodes_cross_continuity": False,
        "roll_count": int(active_in_range["roll"].sum()) if not active_in_range.empty else 0,
        "signal_parameters": signal_parameters,
        "trace_rows": int(len(trace)),
        "entry_opportunities": entry_count,
        "long_entries": int(
            ((trace["target"] == 1) & (trace["target"].shift(fill_value=0) == 0)).sum()
        ),
        "short_entries": int(
            ((trace["target"] == -1) & (trace["target"].shift(fill_value=0) == 0)).sum()
        ),
        "proxy_assumptions": {
            "contracts": int(gc_config["position"]["contracts"]),
            "fill_price": "next completed active-contract 5m bar close",
            "adverse_slippage_ticks_per_side": int(
                fill_config["aggressive_stress_slippage_ticks"]
            ),
            "fee_usd_per_contract_side": float(cost_config["fee_usd_per_contract_side"]),
            "tick_value_usd": float(instrument_config["tick_value_usd"]),
            "force_flat_on_session_end": True,
            "force_flat_before_roll": True,
            "session_flatten_rule_id": SESSION_FLATTEN_RULE_ID,
            "session_flattening": session_flattening,
        },
        "proxy_performance": _pnl_summary(episodes, trace["session_date"]),
    }
    return ScreeningResult(report=report, trace=trace, episodes=episodes)
