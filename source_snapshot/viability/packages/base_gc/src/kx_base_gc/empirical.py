from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd

from kx_mbo_core.active_contract_v2 import verify_active_contract_certificate
from kx_mbo_core.baseline_pipeline import aggregate_l1_to_5m

from .bars import BOUNDARY_COLUMNS, assert_prefix_invariant, build_causal_timeframes
from .contract import BaselineContract, registered_contract
from .prepared import (
    PREPARED_ARTIFACT_TYPE,
    PREPARED_PRICE_UNIT,
    load_certified_prepared_product,
    sha256_file,
)
from .sessions import globex_session_dates
from .state import build_state_records


DEVELOPMENT_END = date(2025, 8, 31)
ONE_MILLISECOND_NS = 1_000_000
SESSION_CUTOFF = "12:25"
SESSION_TIME_ZONE = "America/Chicago"


class EmpiricalPreparedError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedBuildResult:
    output_directory: Path
    bars_path: Path
    manifest_path: Path
    split_path: Path
    report_path: Path
    report: Mapping[str, Any]


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, date)):
        return str(value)
    return value


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(handle)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _iso_date(value: Any) -> date:
    result = pd.Timestamp(value).date()
    if result.year != 2025 or result > DEVELOPMENT_END:
        raise PermissionError(
            f"Base-GC empirical preparation is restricted to Jan-Aug 2025: {result}"
        )
    return result


def _session_cutoff_ns(session_date: str) -> int:
    local = pd.Timestamp(f"{session_date} {SESSION_CUTOFF}:00", tz=SESSION_TIME_ZONE)
    return int(local.tz_convert("UTC").value)


def _resolve_active_map(
    certificate_path: str | Path,
    *,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, Mapping[str, Any], Path]:
    source = Path(certificate_path).expanduser().resolve()
    certificate = verify_active_contract_certificate(source)
    if (
        certificate.get("passed") is not True
        or certificate.get("past_only") is not True
        or int(certificate.get("minimum_lag_sessions", 0)) < 1
        or certificate.get("calibration_or_locked_outcomes_read") is not False
        or certificate.get("sealed_2026_read") is not False
    ):
        raise EmpiricalPreparedError("active-contract-v2 certificate is not development-safe")
    artifacts = certificate.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("active_contract_map"), Mapping
    ):
        raise EmpiricalPreparedError("active-contract-v2 certificate lacks its map")
    declaration = artifacts["active_contract_map"]
    map_path = Path(str(declaration.get("path", ""))).expanduser().resolve()
    if not map_path.is_file() or sha256_file(map_path) != str(
        declaration.get("sha256", "")
    ):
        raise EmpiricalPreparedError("active-contract-v2 map binding failed")
    active = pd.read_parquet(map_path)
    active["date"] = pd.to_datetime(active["date"], errors="raise").dt.date.astype(str)
    active = active.loc[active["date"].between(str(start), str(end))].copy()
    if active.empty or active["date"].duplicated().any():
        raise EmpiricalPreparedError("requested period has no unique active assignments")
    expected = [
        value
        for value in certificate.get("assignments", ())
        if str(start) <= str(value.get("date", "")) <= str(end)
    ]
    observed = [
        {"date": str(row.date), "raw_symbol": str(row.selected_raw_symbol)}
        for row in active.itertuples(index=False)
    ]
    if observed != expected:
        raise EmpiricalPreparedError("active map differs from authenticated assignments")
    return active.reset_index(drop=True), certificate, source


def _source_l1_paths(certificate: Mapping[str, Any], *, end: date) -> list[Path]:
    bindings = certificate.get("source_l1_files")
    if not isinstance(bindings, list) or not bindings:
        raise EmpiricalPreparedError("active certificate has no authenticated L1 sources")
    paths: list[Path] = []
    for item in bindings:
        if not isinstance(item, Mapping):
            raise EmpiricalPreparedError("active certificate has an invalid L1 binding")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(item.get("sha256", "")):
            raise EmpiricalPreparedError(f"authenticated L1 source changed: {path}")
        if "year=2025" not in path.parts or "l1_trades" not in path.parts:
            raise PermissionError("an authenticated L1 source is outside year=2025")
        paths.append(path)
    if not paths:
        raise EmpiricalPreparedError(f"no L1 sources are available through {end}")
    return sorted(paths)


def _state_segments(active: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    assignment_order = active.sort_values("date", kind="mergesort")
    assignment_ids: dict[str, int] = {}
    prior_symbol: str | None = None
    segment = -1
    for row in assignment_order.itertuples(index=False):
        symbol = str(row.selected_raw_symbol)
        if symbol != prior_symbol:
            segment += 1
        assignment_ids[str(row.date)] = segment
        prior_symbol = symbol
    values = []
    continuity_seen: dict[tuple[int, int], str] = {}
    for row in bars.itertuples(index=False):
        roll_segment = assignment_ids[str(row.date)]
        continuity = int(row.continuity_segment_id)
        key = (roll_segment, continuity)
        if key not in continuity_seen:
            continuity_seen[key] = f"base-gc-state-{roll_segment:03d}-{continuity:06d}"
        values.append(continuity_seen[key])
    return pd.Series(values, index=bars.index, dtype="object")


def build_empirical_timeframes(
    *,
    archive_root: str | Path,
    active_contract_certificate_path: str | Path,
    start: Any,
    end: Any,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, Mapping[str, Any], Path]:
    """Build exact active-contract H1/M15/M5 OHLCV without opening later data."""

    start_date = _iso_date(start)
    end_date = _iso_date(end)
    if start_date > end_date:
        raise ValueError("prepared period is reversed")
    active, certificate, certificate_path = _resolve_active_map(
        active_contract_certificate_path, start=start_date, end=end_date
    )
    l1_paths = _source_l1_paths(certificate, end=end_date)
    bars = aggregate_l1_to_5m(l1_paths, through=end_date)
    bars["bar_end_ns"] = pd.to_datetime(bars["bar_end"], utc=True).astype("int64")
    bars["date"] = globex_session_dates(bars["bar_end_ns"]).to_numpy()
    selected = active[["date", "selected_raw_symbol"]].rename(
        columns={"selected_raw_symbol": "selected_symbol"}
    )
    bars = bars.merge(selected, on="date", how="inner", validate="many_to_one")
    bars = bars.loc[bars["raw_symbol"].astype(str).eq(bars["selected_symbol"].astype(str))].copy()
    if bars.empty:
        raise EmpiricalPreparedError("authenticated active selection produced no L1 bars")
    bars = bars.sort_values(
        ["bar_end_ns", "raw_symbol", "continuity_segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    bars["state_segment_id"] = _state_segments(active, bars)
    m5 = bars[
        [
            "date",
            "raw_symbol",
            "continuity_segment_id",
            "state_segment_id",
            "bar_end_ns",
            "available_time_ns",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()
    timeframes = build_causal_timeframes(m5)
    assert_prefix_invariant(m5, [1, max(1, len(m5) // 3), max(1, 2 * len(m5) // 3), len(m5)])
    return timeframes, active, certificate, certificate_path


def _active_assignment_ledger(
    active: pd.DataFrame,
    *,
    source_certificate_path: Path,
) -> dict[str, Any]:
    assignments = []
    for row in active.itertuples(index=False):
        source_through = str(row.prior_completed_session)
        if pd.Timestamp(source_through).date() >= pd.Timestamp(row.date).date():
            raise EmpiricalPreparedError("active assignment uses same/future session volume")
        assignments.append(
            {
                "date": str(row.date),
                "raw_symbol": str(row.selected_raw_symbol),
                "source_sessions_through": source_through,
                "selection_available_time_ns": int(row.decision_time_utc_ns),
                "volume_cutoff_available_time_ns": int(
                    row.volume_cutoff_available_time_ns
                ),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "BASE_GC_ACTIVE_CONTRACT_ASSIGNMENT_LEDGER",
        "passed": True,
        "past_only": True,
        "minimum_lag_sessions": 1,
        "source_active_contract_v2_certificate": {
            "path": str(source_certificate_path),
            "sha256": sha256_file(source_certificate_path),
        },
        "assignments": assignments,
    }


def _decorate_timeframes(
    timeframes: Mapping[str, pd.DataFrame],
    *,
    active_map_hash: str,
    contract: BaselineContract,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    route_latency_ns = int(contract.execution.route_latency_ms * 1_000_000)
    decision_latency_ns = int(contract.execution.decision_latency_ms * 1_000_000)
    for name in ("H1", "M15", "M5"):
        frame = timeframes[name].copy()
        frame["date"] = globex_session_dates(frame["bar_end_ns"]).to_numpy()
        cutoffs = frame["date"].map(_session_cutoff_ns).astype("int64")
        frame["scheduled_flatten_time_ns"] = cutoffs
        frame["decision_eligible"] = frame["bar_end_ns"].astype("int64") <= cutoffs
        frame["reward_boundary_id"] = [
            f"base-gc|{session}|{'active' if eligible else 'post-cutoff'}"
            for session, eligible in zip(
                frame["date"], frame["decision_eligible"], strict=True
            )
        ]
        frame["decision_time_ns"] = (
            frame["available_time_ns"].astype("int64") + decision_latency_ns
        )
        frame["submission_time_ns"] = frame["decision_time_ns"]
        frame["arrival_time_ns"] = frame["submission_time_ns"] + route_latency_ns
        frame["active_contract_map_hash"] = active_map_hash
        frame["timeframe"] = name
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["arrival_time_ns", "timeframe", "raw_symbol", "bar_end_ns"],
        kind="mergesort",
    ).reset_index(drop=True)


def scan_timeframe_support(
    timeframes: Mapping[str, pd.DataFrame],
    *,
    contract: BaselineContract,
    training_cutoff_ns: int,
) -> dict[str, Any]:
    """Count the maximum registered renewal samples without training a model."""

    return scan_timeframe_support_cutoffs(
        timeframes,
        contract=contract,
        training_cutoffs_ns={"requested": int(training_cutoff_ns)},
    )["requested"]


def scan_timeframe_support_cutoffs(
    timeframes: Mapping[str, pd.DataFrame],
    *,
    contract: BaselineContract,
    training_cutoffs_ns: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Count multiple cutoffs while constructing each causal state only once."""

    cutoffs = {str(name): int(value) for name, value in training_cutoffs_ns.items()}
    if not cutoffs or any(value <= 0 for value in cutoffs.values()):
        raise ValueError("support scan requires positive named cutoffs")
    outputs: dict[str, dict[str, Any]] = {name: {} for name in cutoffs}

    for name in ("H1", "M15", "M5"):
        frame = timeframes[name].reset_index(drop=True)
        inherited: dict[str, Iterable[int]]
        if name == "H1":
            inherited = {}
        elif name == "M15":
            inherited = {"H1": [0] * len(frame)}
        else:
            inherited = {"H1": [0] * len(frame), "M15": [0] * len(frame)}
        records = build_state_records(
            frame, contract.agent(name), inherited=inherited
        )
        metadata = {
            (
                str(row.raw_symbol),
                int(row.continuity_segment_id),
                str(row.state_segment_id),
                int(row.bar_end_ns),
            ): row
            for row in frame.itertuples(index=False)
        }
        annotated: list[tuple[int, bool, str]] = []
        for record in records:
            row = metadata[
                (
                    record.raw_symbol,
                    record.continuity_segment_id,
                    record.state_segment_id,
                    record.bar_end_ns,
                )
            ]
            annotated.append(
                (
                    int(record.available_time_ns),
                    bool(getattr(row, "decision_eligible", True)),
                    str(row.reward_boundary_id),
                )
            )
        horizon = contract.agent(name).action_lock_bars
        for cutoff_name, cutoff in cutoffs.items():
            boundaries: dict[str, int] = {}
            warmed_state_count = 0
            for available, eligible, boundary in annotated:
                if available <= cutoff:
                    warmed_state_count += 1
                    if eligible:
                        boundaries[boundary] = boundaries.get(boundary, 0) + 1
            state_count = sum(boundaries.values())
            decisions = sum(
                (count + horizon - 1) // horizon for count in boundaries.values()
            )
            matured = sum(
                sum(index + horizon < count for index in range(0, count, horizon))
                for count in boundaries.values()
            )
            outputs[cutoff_name][name] = {
                "agent_bars_before_cutoff": int(
                    (frame["available_time_ns"].astype("int64") <= cutoff).sum()
                ),
                "fully_warmed_training_states": warmed_state_count,
                "warmed_training_states_before_1225_ct": state_count,
                "maximum_training_decisions": decisions,
                "maximum_matured_training_transitions": matured,
                "required_matured_training_transitions": contract.optimization.replay_warmup,
                "support_gate_passed": matured >= contract.optimization.replay_warmup,
            }
    for cutoff_name in cutoffs:
        outputs[cutoff_name]["all_agents_support_gate_passed"] = all(
            bool(outputs[cutoff_name][name]["support_gate_passed"])
            for name in ("H1", "M15", "M5")
        )
    return outputs


def _manifest_for_date(archive_root: Path, utc_date: str) -> tuple[Path, Mapping[str, Any]]:
    if not utc_date.startswith("2025-"):
        raise PermissionError(f"refusing non-2025 source manifest: {utc_date}")
    path = (archive_root / "_manifests" / f"{utc_date}.json").resolve()
    if not path.is_file():
        raise EmpiricalPreparedError(f"source manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("status") != "passed"
        or str(payload.get("date", "")) != utc_date
    ):
        raise EmpiricalPreparedError(f"source manifest did not pass: {path}")
    stats = payload.get("stats")
    if not isinstance(stats, Mapping):
        raise EmpiricalPreparedError(f"source manifest lacks replay stats: {path}")
    zero_checks = (
        "incomplete_mutating_events",
        "degraded_snapshot_resyncs",
        "ts_recv_reversals",
        "non_integral_tick_prices",
    )
    violations = {name: int(stats.get(name, -1)) for name in zero_checks}
    if any(value != 0 for value in violations.values()):
        raise EmpiricalPreparedError(
            f"source manifest has unresolved replay violations {utc_date}: {violations}"
        )
    return path, payload


def _layer_bindings(
    archive_root: Path,
    manifest: Mapping[str, Any],
    *,
    layer: str,
    hash_cache: dict[Path, str],
) -> list[dict[str, Any]]:
    outputs = [
        item
        for item in manifest.get("outputs", ())
        if isinstance(item, Mapping) and str(item.get("layer", "")) == layer
    ]
    if not outputs:
        raise EmpiricalPreparedError(
            f"manifest {manifest.get('date')} has no {layer} outputs"
        )
    bindings = []
    for item in outputs:
        relative = Path(str(item.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise EmpiricalPreparedError("source output path escapes the archive")
        path = (archive_root / relative).resolve()
        if not path.is_relative_to(archive_root) or not path.is_file():
            raise EmpiricalPreparedError(f"source output is missing: {path}")
        observed = hash_cache.setdefault(path, sha256_file(path))
        if observed != str(item.get("sha256", "")):
            raise EmpiricalPreparedError(f"source output hash mismatch: {path}")
        bindings.append(
            {
                "path": str(path),
                "relative_path": str(relative),
                "sha256": observed,
                "rows": int(item.get("rows", -1)),
                "size_bytes": int(item.get("size", -1)),
            }
        )
    return bindings


def _latest_best(heap: list[int], levels: Mapping[int, int], *, bid: bool) -> int:
    while heap:
        encoded = heap[0]
        price = -encoded if bid else encoded
        if price in levels:
            return price
        heapq.heappop(heap)
    raise EmpiricalPreparedError("causal book state has no executable best price")


def _replay_symbol_quotes(
    l2: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    source_tape_sha256: str,
) -> list[dict[str, Any]]:
    if l2.empty:
        raise EmpiricalPreparedError("selected L2 source contains no rows")
    l2 = l2.sort_values(
        ["event_index", "event_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    request_rows = requests.sort_values("arrival_time_ns", kind="mergesort")
    event_index = l2["event_index"].to_numpy(dtype=np.int64)
    receive = l2["ts_recv_ns"].to_numpy(dtype=np.int64)
    ordinals = l2["global_record_ordinal_end"].to_numpy(dtype=np.int64)
    operations = l2["operation"].astype(str).to_numpy()
    sides = l2["side"].astype(str).to_numpy()
    prices = pd.to_numeric(l2["price_ticks"], errors="coerce").to_numpy(dtype=float)
    sizes = pd.to_numeric(l2["level_size"], errors="coerce").to_numpy(dtype=float)
    bid_levels: dict[int, int] = {}
    ask_levels: dict[int, int] = {}
    bid_heap: list[int] = []
    ask_heap: list[int] = []
    offset = 0
    latest_available = 0
    latest_ordinal = 0
    initialized = False
    results: list[dict[str, Any]] = []
    for request in request_rows.itertuples(index=False):
        arrival = int(request.arrival_time_ns)
        while offset < len(l2):
            stop = offset + 1
            while stop < len(l2) and event_index[stop] == event_index[offset]:
                stop += 1
            group_available = int(receive[offset:stop].max()) + ONE_MILLISECOND_NS
            if group_available > arrival:
                break
            for position in range(offset, stop):
                operation = operations[position]
                side = sides[position]
                if operation == "clear":
                    bid_levels.clear()
                    ask_levels.clear()
                    bid_heap.clear()
                    ask_heap.clear()
                    initialized = True
                    continue
                if side not in {"B", "A"} or not np.isfinite(prices[position]):
                    raise EmpiricalPreparedError("L2 replay contains an invalid level operation")
                price = int(prices[position])
                levels = bid_levels if side == "B" else ask_levels
                heap = bid_heap if side == "B" else ask_heap
                if operation == "delete" or not np.isfinite(sizes[position]) or int(sizes[position]) <= 0:
                    levels.pop(price, None)
                elif operation == "upsert":
                    levels[price] = int(sizes[position])
                    heapq.heappush(heap, -price if side == "B" else price)
                else:
                    raise EmpiricalPreparedError(f"unregistered L2 operation {operation!r}")
            latest_available = group_available
            latest_ordinal = int(ordinals[offset:stop].max())
            offset = stop
        if not initialized or latest_available <= 0 or latest_ordinal <= 0:
            raise EmpiricalPreparedError("arrival precedes an authenticated L2 snapshot")
        bid = _latest_best(bid_heap, bid_levels, bid=True)
        ask = _latest_best(ask_heap, ask_levels, bid=False)
        if bid >= ask or latest_available > arrival:
            raise EmpiricalPreparedError("arrival-aligned L2 state is crossed or future-dated")
        results.append(
            {
                "request_id": str(request.request_id),
                "aggressive_sell_fill_ticks": bid,
                "aggressive_buy_fill_ticks": ask,
                "aggressive_fill_available_time_ns": latest_available,
                "aggressive_fill_source_ordinal": latest_ordinal,
                "source_tape_sha256": source_tape_sha256,
            }
        )
    return results


def replay_arrival_aligned_fills(
    prepared: pd.DataFrame,
    *,
    archive_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay authenticated L2 level updates to every registered venue arrival."""

    root = Path(archive_root).expanduser().resolve()
    requests = prepared[
        [
            "timeframe",
            *BOUNDARY_COLUMNS,
            "bar_end_ns",
            "arrival_time_ns",
        ]
    ].copy()
    requests["request_id"] = [f"base-fill-{index:09d}" for index in range(len(requests))]
    requests["utc_date"] = pd.to_datetime(
        requests["arrival_time_ns"], unit="ns", utc=True
    ).dt.date.astype(str)
    hash_cache: dict[Path, str] = {}
    fill_rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    all_l0: list[dict[str, Any]] = []
    all_l2: list[dict[str, Any]] = []
    connection = duckdb.connect()
    try:
        for utc_date, date_requests in requests.groupby("utc_date", sort=True):
            manifest_path, manifest = _manifest_for_date(root, str(utc_date))
            l0_bindings = _layer_bindings(
                root, manifest, layer="l0", hash_cache=hash_cache
            )
            l2_bindings = _layer_bindings(
                root, manifest, layer="l2_depth", hash_cache=hash_cache
            )
            source_manifests.append(
                {
                    "date": str(utc_date),
                    "path": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                    "status": "passed",
                    # Incomplete non-mutating status/trade groups cannot alter
                    # the reconstructed L2 book.  They remain quantified here
                    # instead of being silently relabelled as complete.
                    "incomplete_non_mutating_events": int(
                        manifest["stats"].get("incomplete_non_mutating_events", 0)
                    ),
                }
            )
            all_l0.extend({"date": str(utc_date), **item} for item in l0_bindings)
            all_l2.extend({"date": str(utc_date), **item} for item in l2_bindings)
            tape_hash = _canonical_json_sha256(
                [{"path": item["relative_path"], "sha256": item["sha256"]} for item in l2_bindings]
            )
            paths = [item["path"] for item in l2_bindings]
            symbols = sorted(date_requests["raw_symbol"].astype(str).unique())
            continuities = sorted(
                int(value) for value in date_requests["continuity_segment_id"].unique()
            )
            l2 = connection.execute(
                """
                SELECT raw_symbol, continuity_segment_id, ts_recv_ns,
                       event_index, event_row_index, global_record_ordinal_end,
                       operation, side, price_ticks, level_size
                FROM read_parquet(?)
                WHERE raw_symbol IN (SELECT * FROM unnest(?))
                  AND continuity_segment_id IN (SELECT * FROM unnest(?))
                ORDER BY raw_symbol, continuity_segment_id, event_index, event_row_index
                """,
                [paths, symbols, continuities],
            ).fetchdf()
            for (symbol, continuity), group_requests in date_requests.groupby(
                ["raw_symbol", "continuity_segment_id"], sort=False
            ):
                source = l2.loc[
                    l2["raw_symbol"].astype(str).eq(str(symbol))
                    & l2["continuity_segment_id"].astype(int).eq(int(continuity))
                ]
                boundary_checks = [
                    value
                    for value in manifest["stats"].get("boundary_checks", ())
                    if isinstance(value, Mapping)
                    and str(value.get("raw_symbol", "")) == str(symbol)
                ]
                if not boundary_checks or not any(
                    str(value.get("status", ""))
                    in {"initial_snapshot", "exact_snapshot_match", "reset"}
                    for value in boundary_checks
                ):
                    raise EmpiricalPreparedError(
                        f"{utc_date} {symbol} lacks an audited replay boundary"
                    )
                fill_rows.extend(
                    _replay_symbol_quotes(
                        source,
                        group_requests,
                        source_tape_sha256=tape_hash,
                    )
                )
    finally:
        connection.close()
    fills = pd.DataFrame(fill_rows)
    if len(fills) != len(requests) or fills["request_id"].duplicated().any():
        raise EmpiricalPreparedError("arrival-fill replay did not cover every prepared row")
    joined = requests.merge(fills, on="request_id", how="left", validate="one_to_one")
    ledger = joined[
        [
            "timeframe",
            *BOUNDARY_COLUMNS,
            "bar_end_ns",
            "aggressive_sell_fill_ticks",
            "aggressive_buy_fill_ticks",
            "aggressive_fill_available_time_ns",
            "aggressive_fill_source_ordinal",
            "source_tape_sha256",
        ]
    ].copy()
    source_audit = {
        "schema_version": 1,
        "artifact_type": "BASE_GC_CAUSAL_FIFO_REPLAY_CERTIFICATE",
        "passed": True,
        "complete_atomic_groups": True,
        "complete_book_mutating_atomic_groups": True,
        "incomplete_non_mutating_events": int(
            sum(item["incomplete_non_mutating_events"] for item in source_manifests)
        ),
        "unresolved_sequence_gaps": 0,
        "derivation": "authenticated_l0_to_l2_fifo_then_arrival_asof_v1",
        "availability_rule": "complete_l2_event_ts_recv_ns_plus_1ms",
        "source_daily_manifests": source_manifests,
        "source_l0_partitions": all_l0,
        "source_l2_partitions": all_l2,
        "source_manifest_set_sha256": _canonical_json_sha256(source_manifests),
        "source_l0_partition_set_sha256": _canonical_json_sha256(all_l0),
        "source_l2_partition_set_sha256": _canonical_json_sha256(all_l2),
        "sealed_2026_read": False,
        "sep_dec_2025_read": False,
    }
    return ledger, source_audit


def produce_empirical_prepared_product(
    *,
    archive_root: str | Path,
    active_contract_certificate_path: str | Path,
    start: Any,
    end: Any,
    training_cutoff_ns: int,
    evaluation_partition_id: str,
    evaluation_start_time_ns: int,
    evaluation_end_time_ns: int,
    output_directory: str | Path,
) -> PreparedBuildResult:
    """Create and self-load the smallest authenticated Base-GC product."""

    started = time.perf_counter()
    start_date = _iso_date(start)
    end_date = _iso_date(end)
    if not evaluation_partition_id:
        raise ValueError("evaluation_partition_id cannot be empty")
    if not (
        0 < int(training_cutoff_ns) < int(evaluation_start_time_ns) < int(evaluation_end_time_ns)
    ):
        raise ValueError("registered training/evaluation clocks are invalid")
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace a prepared product: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        timeframes, active, active_source, active_source_path = build_empirical_timeframes(
            archive_root=archive_root,
            active_contract_certificate_path=active_contract_certificate_path,
            start=start_date,
            end=end_date,
        )
        active_ledger = _active_assignment_ledger(
            active, source_certificate_path=active_source_path
        )
        active_ledger_path = temporary / "active_contract_assignment_ledger.json"
        _write_json(active_ledger_path, active_ledger)
        active_map_hash = sha256_file(active_ledger_path)
        prepared = _decorate_timeframes(
            timeframes,
            active_map_hash=active_map_hash,
            contract=registered_contract("base_strong_gc"),
        )
        fill_ledger, replay_certificate = replay_arrival_aligned_fills(
            prepared, archive_root=archive_root
        )
        join_keys = ["timeframe", *BOUNDARY_COLUMNS, "bar_end_ns"]
        prepared = prepared.merge(
            fill_ledger,
            on=join_keys,
            how="left",
            validate="one_to_one",
        )
        if prepared[
            [
                "aggressive_sell_fill_ticks",
                "aggressive_buy_fill_ticks",
                "aggressive_fill_available_time_ns",
                "aggressive_fill_source_ordinal",
            ]
        ].isna().any().any():
            raise EmpiricalPreparedError("prepared rows are missing arrival-fill labels")
        prepared_path = temporary / "prepared_hierarchy_bars.parquet"
        fill_ledger_path = temporary / "arrival_fill_ledger.parquet"
        replay_path = temporary / "causal_fifo_replay_certificate.json"
        _write_parquet(prepared_path, prepared)
        _write_parquet(
            fill_ledger_path,
            prepared[
                [
                    *join_keys,
                    "decision_time_ns",
                    "submission_time_ns",
                    "arrival_time_ns",
                    "aggressive_sell_fill_ticks",
                    "aggressive_buy_fill_ticks",
                    "aggressive_fill_available_time_ns",
                    "aggressive_fill_source_ordinal",
                    "source_tape_sha256",
                ]
            ],
        )
        _write_json(replay_path, replay_certificate)
        bars_hash = sha256_file(prepared_path)

        split = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_REGISTERED_SPLIT",
            "evidence_class": "DEVELOPMENT_ONLY",
            "passed": True,
            "training_cutoff_ns": int(training_cutoff_ns),
            "evaluation": {
                "partition_id": evaluation_partition_id,
                "start_time_ns": int(evaluation_start_time_ns),
                "end_time_ns": int(evaluation_end_time_ns),
            },
            "period": {"start": str(start_date), "end": str(end_date)},
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
        }
        split_path = temporary / "registered_split.json"
        _write_json(split_path, split)
        causal_certificate = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_CAUSAL_COMPLETED_BARS_CERTIFICATE",
            "claim": "causal_completed_bars",
            "passed": True,
            "prepared_bars_sha256": bars_hash,
            "price_unit": PREPARED_PRICE_UNIT,
            "clock_contract": "bar_end<=available<=decision<=submission<=arrival",
            "timeframes": ["H1", "M15", "M5"],
            "rollup_rule": "exact_complete_m5_components_v1",
            "session_calendar": "gc_globex_chicago_1700_1600_v1",
            "session_flattening": {
                "decision_cutoff": SESSION_CUTOFF,
                "timezone": SESSION_TIME_ZONE,
                "post_cutoff_bars": "observation_only_flat",
            },
            "row_counts": {
                name: int(prepared["timeframe"].eq(name).sum())
                for name in ("H1", "M15", "M5")
            },
        }
        causal_path = temporary / "causal_completed_bars_certificate.json"
        _write_json(causal_path, causal_certificate)
        active_certificate = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_ACTIVE_CONTRACT_CERTIFICATE",
            "claim": "active_contract_map",
            "passed": True,
            "prepared_bars_sha256": bars_hash,
            "map_sha256": active_map_hash,
            "map_ledger": {
                "artifact": active_ledger_path.name,
                "sha256": active_map_hash,
            },
            "source_active_contract_v2_certificate": {
                "path": str(active_source_path),
                "sha256": sha256_file(active_source_path),
            },
            "source_active_contract_v2_map_sha256": active_source["map_sha256"],
        }
        active_path = temporary / "active_contract_certificate.json"
        _write_json(active_path, active_certificate)
        arrival_certificate = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_ARRIVAL_FILL_CERTIFICATE",
            "claim": "arrival_aligned_aggressive_fills",
            "passed": True,
            "prepared_bars_sha256": bars_hash,
            "price_unit": PREPARED_PRICE_UNIT,
            "fill_clock_rule": "fill_available_time_ns<=arrival_time_ns",
            "derivation": "causal_fifo_asof_registered_arrival_v1",
            "required_columns": [
                "aggressive_sell_fill_ticks",
                "aggressive_buy_fill_ticks",
                "aggressive_fill_available_time_ns",
                "aggressive_fill_source_ordinal",
                "decision_time_ns",
                "submission_time_ns",
                "arrival_time_ns",
            ],
            "fill_ledger": {
                "artifact": fill_ledger_path.name,
                "sha256": sha256_file(fill_ledger_path),
            },
            "source_replay_certificate": {
                "artifact": replay_path.name,
                "sha256": sha256_file(replay_path),
            },
        }
        arrival_path = temporary / "arrival_fill_certificate.json"
        _write_json(arrival_path, arrival_certificate)
        manifest = {
            "schema_version": 1,
            "artifact_type": PREPARED_ARTIFACT_TYPE,
            "evidence_class": "DEVELOPMENT_ONLY",
            "bars": {
                "filename": prepared_path.name,
                "sha256": bars_hash,
                "row_count": len(prepared),
                "price_unit": PREPARED_PRICE_UNIT,
            },
            "period": {
                "start": str(prepared["date"].min()),
                "end": str(prepared["date"].max()),
            },
            "registered_split": {
                "artifact": split_path.name,
                "sha256": sha256_file(split_path),
            },
            "claims": {
                "causal_completed_bars": {
                    "passed": True,
                    "artifact": causal_path.name,
                    "sha256": sha256_file(causal_path),
                },
                "active_contract_map": {
                    "passed": True,
                    "artifact": active_path.name,
                    "sha256": sha256_file(active_path),
                },
                "arrival_aligned_aggressive_fills": {
                    "passed": True,
                    "artifact": arrival_path.name,
                    "sha256": sha256_file(arrival_path),
                },
            },
            "source_scope": {
                "source_partition_years": [2025],
                "sealed_2026_read": False,
                "sep_dec_2025_read": False,
            },
        }
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)

        paper = load_certified_prepared_product(
            prepared_path,
            manifest_path,
            registered_contract("base_paper_gc"),
            allowed_end_date=str(DEVELOPMENT_END),
        )
        strong = load_certified_prepared_product(
            prepared_path,
            manifest_path,
            registered_contract("base_strong_gc"),
            allowed_end_date=str(DEVELOPMENT_END),
        )
        support = {
            variant: scan_timeframe_support(
                product.timeframes,
                contract=registered_contract(variant),
                training_cutoff_ns=int(training_cutoff_ns),
            )
            for variant, product in (
                ("base_paper_gc", paper),
                ("base_strong_gc", strong),
            )
        }
        report = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_EMPIRICAL_PREPARED_BUILD_REPORT",
            "evidence_class": "DEVELOPMENT_ONLY",
            "passed": True,
            "period": manifest["period"],
            "prepared_manifest_sha256": sha256_file(manifest_path),
            "prepared_bars_sha256": bars_hash,
            "active_contract_v2_certificate_sha256": sha256_file(active_source_path),
            "active_sessions": int(active["date"].nunique()),
            "active_symbols": sorted(active["selected_raw_symbol"].astype(str).unique()),
            "roll_events": int(active["rolled"].astype(bool).sum()),
            "row_counts": causal_certificate["row_counts"],
            "decision_eligible_rows": int(prepared["decision_eligible"].sum()),
            "spread_ticks": {
                "minimum": int(
                    (
                        prepared["aggressive_buy_fill_ticks"]
                        - prepared["aggressive_sell_fill_ticks"]
                    ).min()
                ),
                "median": float(
                    (
                        prepared["aggressive_buy_fill_ticks"]
                        - prepared["aggressive_sell_fill_ticks"]
                    ).median()
                ),
                "maximum": int(
                    (
                        prepared["aggressive_buy_fill_ticks"]
                        - prepared["aggressive_sell_fill_ticks"]
                    ).max()
                ),
            },
            "support": support,
            "five_seed_training_executable": all(
                bool(support[variant]["all_agents_support_gate_passed"])
                for variant in ("base_paper_gc", "base_strong_gc")
            ),
            "load_certified_prepared_product": {
                "base_paper_gc": True,
                "base_strong_gc": True,
            },
            "runtime_seconds": time.perf_counter() - started,
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
            "claim_scope": (
                "Authenticated prepared inputs and count-only support; no model was "
                "trained and no profitability or HMADRL claim is made."
            ),
        }
        report_path = temporary / "build_report.json"
        _write_json(report_path, report)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    final_manifest = output / "manifest.json"
    # Reopen after the atomic directory move so the returned product paths are
    # exactly the immutable paths downstream training will resolve.
    load_certified_prepared_product(
        output / "prepared_hierarchy_bars.parquet",
        final_manifest,
        registered_contract("base_strong_gc"),
        allowed_end_date=str(DEVELOPMENT_END),
    )
    final_report_path = output / "build_report.json"
    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))
    return PreparedBuildResult(
        output_directory=output,
        bars_path=output / "prepared_hierarchy_bars.parquet",
        manifest_path=final_manifest,
        split_path=output / "registered_split.json",
        report_path=final_report_path,
        report=final_report,
    )
