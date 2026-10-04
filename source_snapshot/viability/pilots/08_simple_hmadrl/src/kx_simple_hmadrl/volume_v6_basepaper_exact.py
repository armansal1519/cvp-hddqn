"""Paper-faithful BasePaper comparator for the closed V6 experiment.

This module is intentionally separate from :mod:`volume_v6_baseline`.  It
reconstructs the published Shavandi--Khedmati DQN mechanics without changing
the already-frozen V6 candidate or its historical artifacts.  In particular,
observations are raw, row-indexed OHLCV windows; a held action produces one
transition on every eligible bar; and a direct reversal closes and opens the
opposite position at the same decision boundary.

The paper did not publish every operational hyperparameter.  The reconstruction
uses a batch of 64 and target refresh period C=1,000 environment bars, records
those choices in every artifact, and makes no claim of numerical EUR/USD
replication.  Production training starts from seeded random weights on GC, as
the paper describes for its single-instrument experiment.  The XAU loader is
retained only as an unused research/data-audit utility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import heapq
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from kx_base_gc.contract import ACTIONS, AGENT_ORDER, AgentSpec, registered_contract
from kx_base_gc.models import ReplayBuffer, ReplayItem, VanillaQNetwork

from .environment import EconomicLedger, ExecutionEconomics
from .volume_v6_contract import SEEDS
from .volume_v6_env import (
    V6SessionResult,
    _resolve_route,
    _validate_raw,
    _visible_book,
    uses_legacy_arrival_boundary,
)


NS_PER_MINUTE = 60_000_000_000
MAX_PHASE_GAP_DAYS = 7
PAPER_BATCH_SIZE = 64
PAPER_TARGET_PERIOD = 1_000
PAPER_LEARNING_RATE = 0.00025
PAPER_EXACT_NAME = "BasePaper-Described-GC-FromScratch-v1"
PAPER_PROTOCOL_VERSION = "basepaper_described_gc_from_scratch_v1"
# The already-running GC preparation process loaded the preceding module version
# before the training-lineage correction.  Its bars are unchanged and publish
# this historical *data-product* identity.  Training/checkpoint artifacts must
# never use it.
PREPARED_DATA_NAME = "BasePaper-Described-XAU-Pretrained-GC-v1"
PAPER_TRAINING_STAGE = "gc_from_scratch"
PAPER_INITIALIZATION = "seeded_random_from_scratch"
SCENARIOS = (
    "base",
    "double_commission",
    "20ms_route_latency",
    "one_adverse_tick_per_filled_side",
)
SESSION_START_MINUTE = 7 * 60 + 20
SESSION_END_MINUTE = 12 * 60 + 25
CHICAGO_TZ = "America/Chicago"
EXPECTED_XAU_MANIFEST_SHA256 = (
    "eadf18c857a2269f7d90ca3a3bcf56105138568bdf984068b988f3910138dcfb"
)
EXPECTED_GC_MANIFEST_SHA256 = {
    "train": "cdf3e233789a5370d02e1189bcc83e2dc10be4c23e1e92b311f2d4642f862503",
    "validation": "f68bc69dd63524a59a65b58ee15a87fd5bd8dfb3dfe25d9a98b46a2d0772ef52",
}
EXPECTED_PHASE_DATE_SHA256 = {
    "train": "f0d2dd2d3cd4f5c59b96e17548d27d832175e4c7d37b0f29b50e0968a103edff",
    "validation": "7b8d154c7b201a6d665ab9c60dd55e023c5c2476f77ab016b157e2c011c5446d",
}
EXPECTED_GC_TRAIN_BARS = {"H1": 1_324, "M15": 6_289, "M5": 20_189}
EXPECTED_GC_TRAIN_TRANSITIONS = {"H1": 816, "M15": 4_617, "M5": 17_637}
EXPECTED_GC_TRAIN_UPDATES = {"H1": 753, "M15": 4_554, "M5": 17_574}
EXPECTED_GC_TRAIN_RENEWALS = {"H1": 166, "M15": 464, "M5": 888}
SESSION_FILTER_DESCRIPTION = "07:20_inclusive_to_12:25_exclusive_America_Chicago"
TIMEFRAME_DESCRIPTION = (
    "direct_from_hash_bound_monthly_XAU_1m_or_manifest_listed_full_day_GC_L1;"
    "complete_conventional_wall_clock_intervals_inside_session_only;"
    "no_M5_to_coarse_rollup"
)


def _agent_stream_seed(seed: int, name: str) -> int:
    """Derive independent deterministic agent RNG streams from an ensemble seed."""

    if name not in AGENT_ORDER:
        raise ValueError("paper agent stream name is invalid")
    return int(seed) * 10 + AGENT_ORDER.index(name) + 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_torch(path: Path, value: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent,
    )
    os.close(descriptor)
    try:
        torch.save(dict(value), temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _event_ns(frame: pd.DataFrame) -> np.ndarray:
    if "event_time" not in frame:
        raise ValueError("paper frame lacks event_time")
    timestamps = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    values = timestamps.array.asi8.astype(np.int64, copy=False)
    if len(values) and (np.diff(values) <= 0).any():
        raise ValueError("paper rows must be strictly chronological")
    return values


def paper_observation(
    frame: pd.DataFrame,
    index: int,
    lookback: int,
    parent_positions: Sequence[int | float] = (),
) -> np.ndarray:
    """Return raw OHLCV newest-to-oldest, followed by scalar parent positions."""

    index, lookback = int(index), int(lookback)
    if lookback < 1 or index < lookback - 1 or index >= len(frame):
        raise IndexError("paper observation has insufficient lookback")
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"paper frame lacks {missing}")
    window = frame.iloc[index - lookback + 1:index + 1].loc[:, required]
    market = window.to_numpy(dtype=np.float32, copy=True)[::-1].reshape(-1)
    parents = np.asarray(tuple(parent_positions), dtype=np.float32)
    observation = np.concatenate((market, parents)).astype(np.float32, copy=False)
    if not np.isfinite(observation).all():
        raise ValueError("paper observation contains a non-finite value")
    return observation


def _boundary_values(frame: pd.DataFrame) -> np.ndarray:
    """Return the explicit symbol/continuity identity for every supplied row."""

    columns = [
        name for name in (
            "raw_symbol", "state_id", "continuity_segment",
            "continuity_segment_id",
        )
        if name in frame
    ]
    if not columns:
        return np.full(len(frame), "continuous", dtype=object)
    values: list[str] = []
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values.append("\x1f".join(str(value) for value in row))
    return np.asarray(values, dtype=object)


def paper_segments(frame: pd.DataFrame, max_gap_days: int = MAX_PHASE_GAP_DAYS) -> np.ndarray:
    """Segment only on symbol/continuity changes or a gap longer than seven days.

    Ordinary market closures and absent quote bars do not reset a row-indexed
    paper window.  This is deliberately different from the old exact-clock
    continuity filter that eliminated almost all M15 pretraining support.
    """

    if int(max_gap_days) < 1:
        raise ValueError("max_gap_days must be positive")
    times = _event_ns(frame)
    if not len(times):
        return np.empty(0, dtype=np.int64)
    identity = _boundary_values(frame)
    reset = np.zeros(len(frame), dtype=bool)
    reset[0] = True
    reset[1:] = identity[1:] != identity[:-1]
    reset[1:] |= np.diff(times) > int(max_gap_days) * 24 * 60 * NS_PER_MINUTE
    return np.cumsum(reset, dtype=np.int64) - 1


def paper_origins(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    """Return every row with lookback, t+1 successor, and t+tw reward support."""

    segments = paper_segments(frame)
    if not len(segments):
        return np.empty(0, dtype=np.int64)
    lookback = int(spec.lookback)
    horizon = int(spec.action_lock_bars)
    candidates = np.arange(lookback - 1, len(frame) - horizon, dtype=np.int64)
    if not len(candidates):
        return candidates
    start = candidates - lookback + 1
    future = candidates + horizon
    valid = (
        (segments[start] == segments[candidates])
        & (segments[candidates + 1] == segments[candidates])
        & (segments[future] == segments[candidates])
    )
    return candidates[valid]


@dataclass(frozen=True)
class LoadedPaperData:
    frames: Mapping[str, pd.DataFrame]
    source_digest: str
    source_manifest_sha256: str
    dates: tuple[str, ...]
    phase: str
    source_records: tuple[Mapping[str, object], ...]
    smoke_only: bool = False


def _session_minute_and_date(timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    local = pd.to_datetime(timestamps, utc=True, errors="raise").dt.tz_convert(CHICAGO_TZ)
    minute = (local.dt.hour * 60 + local.dt.minute).to_numpy(dtype=np.int64)
    days = local.dt.strftime("%Y-%m-%d").to_numpy(dtype=str)
    return minute, days


def _filter_common_session(frame: pd.DataFrame) -> pd.DataFrame:
    minute, days = _session_minute_and_date(frame["event_time"])
    keep = (minute >= SESSION_START_MINUTE) & (minute < SESSION_END_MINUTE)
    result = frame.loc[keep].copy()
    result["session_date"] = days[keep]
    return result.reset_index(drop=True)


def _session_bounds_ns(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(str(value) for value in days))
    bounds: dict[str, tuple[int, int]] = {}
    for day in unique:
        midnight = pd.Timestamp(day, tz=CHICAGO_TZ)
        start = midnight + pd.Timedelta(minutes=SESSION_START_MINUTE)
        end = midnight + pd.Timedelta(minutes=SESSION_END_MINUTE)
        bounds[day] = (int(start.tz_convert("UTC").value), int(end.tz_convert("UTC").value))
    return (
        np.asarray([bounds[str(day)][0] for day in days], dtype=np.int64),
        np.asarray([bounds[str(day)][1] for day in days], dtype=np.int64),
    )


def _finalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_time", "open", "high", "low", "close", "volume",
        "raw_symbol", "state_id", "session_date",
    ]
    result = frame.loc[:, columns].sort_values("event_time", kind="stable").reset_index(drop=True)
    _event_ns(result)
    market = result.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(
        dtype=np.float64,
    )
    if (
        not len(result) or not np.isfinite(market).all()
        or (market[:, :4] <= 0).any() or (market[:, 4] < 0).any()
    ):
        raise ValueError("paper timeframe contains invalid OHLCV")
    return result


def _aggregate_xau_minutes(one: pd.DataFrame, minutes: int) -> pd.DataFrame:
    source = _filter_common_session(one)
    event_ns = pd.to_datetime(source["event_time"], utc=True).array.asi8
    interval = int(minutes) * NS_PER_MINUTE
    target = (event_ns // interval + 1) * interval  # one-minute rows are open-labelled
    source = source.assign(__target=target, __event=event_ns)
    grouped = source.groupby(["session_date", "__target"], sort=True, observed=True)
    result = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        count=("__event", "size"), first=("__event", "min"), last=("__event", "max"),
    ).reset_index()
    starts, ends = _session_bounds_ns(result["session_date"].to_numpy(dtype=str))
    target_values = result["__target"].to_numpy(dtype=np.int64)
    valid = (
        (target_values - interval >= starts)
        & (target_values <= ends)
        & (result["count"].to_numpy(dtype=np.int64) == minutes)
        & (result["first"].to_numpy(dtype=np.int64) == target_values - interval)
        & (result["last"].to_numpy(dtype=np.int64) == target_values - NS_PER_MINUTE)
    )
    result = result.loc[valid].copy()
    result["event_time"] = pd.to_datetime(result["__target"], utc=True)
    result["raw_symbol"] = "XAUUSD"
    result["state_id"] = "XAUUSD:quote-continuity"
    return _finalize_bars(result)


def _aggregate_gc_ticks(ticks: pd.DataFrame, minutes: int) -> pd.DataFrame:
    source = _filter_common_session(ticks)
    event_ns = pd.to_datetime(source["event_time"], utc=True).array.asi8
    interval = int(minutes) * NS_PER_MINUTE
    # Trades at an exact completed boundary belong to the bar ending there.
    target = ((event_ns - 1) // interval + 1) * interval
    source = source.assign(__target=target)
    grouped = source.groupby(
        ["session_date", "raw_symbol", "__target"], sort=True, observed=True,
    )
    result = grouped.agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), volume=("size", "sum"),
    ).reset_index()
    starts, ends = _session_bounds_ns(result["session_date"].to_numpy(dtype=str))
    target_values = result["__target"].to_numpy(dtype=np.int64)
    result = result.loc[
        (target_values - interval >= starts) & (target_values <= ends)
    ].copy()
    result["event_time"] = pd.to_datetime(result["__target"], utc=True)
    result["state_id"] = result["raw_symbol"].astype(str) + ":audited-continuity"
    return _finalize_bars(result)


def _validate_xau_root(
    root: Path, *, maximum_months: int | None = None,
) -> tuple[list[tuple[str, Path]], str]:
    root = Path(root).resolve()
    manifest_path, receipt_path = root / "manifest.json", root / "manifest.sha256"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ValueError("XAU source lacks immutable manifest receipt")
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != EXPECTED_XAU_MANIFEST_SHA256:
        raise ValueError("XAU source is not the frozen described-BasePaper manifest")
    if receipt_path.read_text(encoding="utf-8") != f"{manifest_sha}  manifest.json\n":
        raise ValueError("XAU source manifest receipt differs")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("source_kind") != "Dukascopy OTC tick-by-tick quotes"
        or value.get("instrument") != "XAUUSD"
        or value.get("locked_period_downloaded") is not False
    ):
        raise ValueError("XAU source identity/lock disclosure differs")
    semantics = value.get("volume_semantics")
    if not isinstance(semantics, dict) or "proxy" not in str(semantics.get("quote_count", "")).lower():
        raise ValueError("XAU quote-count proxy disclosure is absent")
    selected = [
        row for row in value.get("months", [])
        if isinstance(row, dict) and "2012-01" <= str(row.get("month")) <= "2024-12"
    ]
    selected.sort(key=lambda row: str(row["month"]))
    expected = [str(value)[:7] for value in pd.period_range("2012-01", "2024-12", freq="M")]
    months = [str(row["month"]) for row in selected]
    if maximum_months is None and months != expected:
        raise ValueError("paper XAU pretraining requires every 2012--2024 month")
    if maximum_months is not None:
        if not 1 <= int(maximum_months) <= len(selected):
            raise ValueError("maximum_months is outside available XAU coverage")
        selected = selected[:int(maximum_months)]
    paths: list[tuple[str, Path]] = []
    for row in selected:
        path = (root / str(row["one_minute"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("XAU product escapes its source root") from error
        if not path.is_file() or _sha256_file(path) != row.get("one_minute_sha256"):
            raise ValueError(f"XAU monthly product hash differs for {row['month']}")
        paths.append((str(row["month"]), path))
    return paths, manifest_sha


def load_xau_timeframes(
    direct_xau_root: Path, *, maximum_months: int | None = None,
) -> LoadedPaperData:
    paths, manifest_sha = _validate_xau_root(
        Path(direct_xau_root), maximum_months=maximum_months,
    )
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in AGENT_ORDER}
    records: list[Mapping[str, object]] = []
    for month, path in paths:
        frame = pd.read_parquet(
            path, columns=["event_time", "open", "high", "low", "close", "quote_count"],
        ).rename(columns={"quote_count": "volume"})
        for name in AGENT_ORDER:
            minutes = registered_contract("base_paper_gc").agent(name).minutes
            pieces[name].append(_aggregate_xau_minutes(frame, minutes))
        records.append({"month": month, "path": str(path), "sha256": _sha256_file(path)})
    frames = {
        name: _finalize_bars(pd.concat(pieces[name], ignore_index=True))
        for name in AGENT_ORDER
    }
    return LoadedPaperData(
        frames=frames,
        source_digest=_canonical_sha256({"manifest": manifest_sha, "records": records}),
        source_manifest_sha256=manifest_sha,
        dates=tuple(month for month, _ in paths), phase="xau_pretrain",
        source_records=tuple(records),
        smoke_only=maximum_months is not None,
    )


def _verify_described_freeze(protocol: Path, receipt: Path) -> str:
    protocol, receipt = Path(protocol), Path(receipt)
    if not protocol.is_file() or not receipt.is_file():
        raise PermissionError("May access requires the described-BasePaper freeze receipt")
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    digest = _canonical_sha256(protocol_value)
    if (
        receipt_value.get("protocol_sha256") != digest
        or receipt_value.get("validator_result") != "PASS"
        or protocol_value.get("strategy_version")
        not in {PAPER_PROTOCOL_VERSION, PAPER_EXACT_NAME}
    ):
        raise PermissionError("described-BasePaper protocol is not validly frozen")
    return digest


def _checked_gc_manifest(
    path: Path, *, phase: str,
    protocol: Path | None = None, freeze_receipt: Path | None = None,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if phase not in {"train", "validation"}:
        raise PermissionError("described BasePaper can access only train or May validation")
    if phase == "validation":
        if protocol is None or freeze_receipt is None:
            raise PermissionError("May loader requires an explicit frozen protocol receipt")
        _verify_described_freeze(protocol, freeze_receipt)
    manifest_sha = _sha256_file(Path(path))
    if manifest_sha != EXPECTED_GC_MANIFEST_SHA256[phase]:
        raise ValueError("GC source is not the frozen described-BasePaper phase manifest")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        value.get("artifact_type") != "V6_GC_CAUSAL_SESSION_DATASET"
        or value.get("phase") != phase
        or value.get("classification") != "DEVELOPMENT"
    ):
        raise ValueError("GC source manifest identity differs")
    sessions = value.get("sessions")
    if not isinstance(sessions, dict) or not sessions:
        raise ValueError("GC source manifest has no sessions")
    dates = tuple(sorted(str(day) for day in sessions))
    if _canonical_sha256(list(dates)) != EXPECTED_PHASE_DATE_SHA256[phase]:
        raise ValueError("GC phase session opportunity set differs from its frozen list")
    if phase == "train":
        allowed = all("2025-01-01" <= day <= "2026-04-30" for day in dates)
    else:
        allowed = all("2026-05-01" <= day <= "2026-05-31" for day in dates)
    if not allowed or any(day > "2026-05-31" for day in dates):
        raise PermissionError("described BasePaper hard-blocks every date after 2026-05-31")
    if int(value.get("session_count", -1)) != len(dates):
        raise ValueError("GC source manifest session count differs")
    return value, dates


def load_gc_timeframes(
    dataset_manifest: Path, *, phase: str = "train",
    protocol: Path | None = None, freeze_receipt: Path | None = None,
) -> LoadedPaperData:
    manifest_path = Path(dataset_manifest).resolve()
    value, dates = _checked_gc_manifest(
        manifest_path, phase=phase, protocol=protocol, freeze_receipt=freeze_receipt,
    )
    global_sources = value.get("source_files")
    sessions = value["sessions"]
    if not isinstance(global_sources, dict) or not isinstance(sessions, dict):
        raise ValueError("GC source file inventory is absent")
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in AGENT_ORDER}
    evidence: list[Mapping[str, object]] = []
    for day in dates:
        record = sessions[day]
        if not isinstance(record, dict):
            raise ValueError("GC source session record is malformed")
        symbol = str(record.get("raw_symbol", ""))
        sources = record.get("sources")
        if not symbol or not isinstance(sources, dict):
            raise ValueError("GC source session lacks symbol/file evidence")
        l1 = [
            (Path(str(path)).resolve(), proof) for path, proof in sources.items()
            if "/l1_trades/" in str(path).replace("\\", "/")
        ]
        if not l1:
            raise ValueError(f"{day} lacks manifest-listed full-day L1 trades")
        tick_parts: list[pd.DataFrame] = []
        for path, proof in sorted(l1, key=lambda item: str(item[0])):
            if not isinstance(proof, dict) or global_sources.get(str(path)) != proof:
                raise ValueError("GC per-session/global source evidence differs")
            if (
                not path.is_file() or path.stat().st_size != int(proof.get("bytes", -1))
                or _sha256_file(path) != proof.get("sha256")
            ):
                raise ValueError(f"GC full-day L1 source hash differs for {day}")
            part = pd.read_parquet(
                path,
                columns=[
                    "ts_recv_ns", "global_record_ordinal", "price_ticks", "size",
                    "is_bad_ts_recv", "raw_symbol",
                ],
                filters=[("raw_symbol", "==", symbol)],
            )
            tick_parts.append(part)
            evidence.append({
                "date": day, "symbol": symbol, "path": str(path),
                "sha256": str(proof["sha256"]), "bytes": int(proof["bytes"]),
            })
        ticks = pd.concat(tick_parts, ignore_index=True)
        ticks = ticks.loc[
            (~ticks["is_bad_ts_recv"].astype(bool)) & (ticks["raw_symbol"].astype(str) == symbol)
        ].sort_values(["ts_recv_ns", "global_record_ordinal"], kind="stable")
        if ticks.empty:
            raise ValueError(f"{day} {symbol} has no valid L1 trades")
        ticks = ticks.assign(
            event_time=pd.to_datetime(ticks["ts_recv_ns"].to_numpy(dtype=np.int64), utc=True),
            price=ticks["price_ticks"].to_numpy(dtype=np.float64) * 0.1,
            raw_symbol=symbol,
        )
        for name in AGENT_ORDER:
            minutes = registered_contract("base_paper_gc").agent(name).minutes
            pieces[name].append(_aggregate_gc_ticks(ticks, minutes))
    frames = {
        name: _finalize_bars(pd.concat(pieces[name], ignore_index=True))
        for name in AGENT_ORDER
    }
    manifest_sha = _sha256_file(manifest_path)
    return LoadedPaperData(
        frames=frames,
        source_digest=_canonical_sha256({"manifest": manifest_sha, "evidence": evidence}),
        source_manifest_sha256=manifest_sha,
        dates=dates, phase=phase, source_records=tuple(evidence),
        smoke_only=False,
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part.parquet", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_prepared(data: LoadedPaperData, output: Path) -> Path:
    output = Path(output).resolve()
    if output.exists():
        existing, manifest = verify_prepared_data(output, phase=data.phase)
        identity = (
            existing.source_digest, existing.source_manifest_sha256, existing.dates,
        )
        requested = (data.source_digest, data.source_manifest_sha256, data.dates)
        if identity != requested:
            raise ValueError("existing prepared paper data request identity differs")
        return manifest
    staging = output.with_name(f".{output.name}.part")
    if staging.exists():
        raise FileExistsError(f"stale paper preparation directory exists: {staging}")
    staging.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []
    try:
        for name in AGENT_ORDER:
            path = staging / f"{name.lower()}.parquet"
            _atomic_parquet(path, data.frames[name])
            artifacts.append({
                "agent": name, "path": path.name, "sha256": _sha256_file(path),
                "rows": len(data.frames[name]),
            })
        manifest: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "BASEPAPER_DESCRIBED_PREPARED_BARS",
            "name": PREPARED_DATA_NAME,
            "complete": True,
            "phase": data.phase,
            "dates": list(data.dates),
            "source_digest": data.source_digest,
            "source_manifest_sha256": data.source_manifest_sha256,
            "source_records_sha256": _canonical_sha256(list(data.source_records)),
            "session_filter": SESSION_FILTER_DESCRIPTION,
            "timeframe_construction": TIMEFRAME_DESCRIPTION,
            "volume_semantics": (
                "XAU_quote_count_proxy_not_exchange_volume"
                if data.phase == "xau_pretrain" else "GC_exchange_executed_trade_size"
            ),
            "smoke_only": (
                data.phase == "xau_pretrain"
                and list(data.dates)
                != [str(value)[:7] for value in pd.period_range("2012-01", "2024-12", freq="M")]
            ),
            "artifacts": artifacts,
        }
        _atomic_json(staging / "manifest.json", manifest)
        _atomic_json(staging / "manifest.receipt.json", {
            "manifest_sha256": _sha256_file(staging / "manifest.json"),
        })
        os.replace(staging, output)
    finally:
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
    return output / "manifest.json"


def prepare_xau(
    direct_xau_root: Path, output: Path, *, maximum_months: int | None = None,
) -> Path:
    return _write_prepared(
        load_xau_timeframes(direct_xau_root, maximum_months=maximum_months), output,
    )


def prepare_gc(
    dataset_manifest: Path, output: Path, *, phase: str = "train",
    protocol: Path | None = None, freeze_receipt: Path | None = None,
) -> Path:
    return _write_prepared(
        load_gc_timeframes(
            dataset_manifest, phase=phase, protocol=protocol,
            freeze_receipt=freeze_receipt,
        ),
        output,
    )


def verify_prepared_data(
    root: Path, *, phase: str | None = None,
) -> tuple[LoadedPaperData, Path]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    receipt_path = root / "manifest.receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ValueError("prepared paper data lacks its manifest receipt")
    manifest_sha = _sha256_file(manifest_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        receipt.get("manifest_sha256") != manifest_sha
        or value.get("artifact_type") != "BASEPAPER_DESCRIBED_PREPARED_BARS"
        or value.get("name") != PREPARED_DATA_NAME
        or value.get("complete") is not True
    ):
        raise ValueError("prepared paper data identity/hash differs")
    expected_volume = (
        "XAU_quote_count_proxy_not_exchange_volume"
        if value.get("phase") == "xau_pretrain" else "GC_exchange_executed_trade_size"
    )
    if (
        value.get("session_filter") != SESSION_FILTER_DESCRIPTION
        or value.get("timeframe_construction") != TIMEFRAME_DESCRIPTION
        or value.get("volume_semantics") != expected_volume
        or not isinstance(value.get("smoke_only"), bool)
    ):
        raise ValueError("prepared paper observation contract differs")
    actual_phase = str(value.get("phase"))
    if phase is not None and actual_phase != phase:
        raise ValueError("prepared paper data phase differs")
    dates = tuple(str(item) for item in value.get("dates", []))
    if actual_phase == "train" and any(day > "2026-04-30" for day in dates):
        raise PermissionError("prepared train data crosses its development bound")
    if actual_phase == "validation" and any(
        not "2026-05-01" <= day <= "2026-05-31" for day in dates
    ):
        raise PermissionError("prepared validation data is not wholly May 2026")
    if actual_phase not in {"xau_pretrain", "train", "validation"}:
        raise PermissionError("prepared data phase is unauthorized")
    frames: dict[str, pd.DataFrame] = {}
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(AGENT_ORDER):
        raise ValueError("prepared paper data inventory is incomplete")
    by_name = {str(record.get("agent")): record for record in artifacts if isinstance(record, dict)}
    if set(by_name) != set(AGENT_ORDER):
        raise ValueError("prepared paper timeframe inventory differs")
    for name in AGENT_ORDER:
        record = by_name[name]
        path = (root / str(record.get("path"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("prepared paper product escapes its root") from error
        if not path.is_file() or _sha256_file(path) != record.get("sha256"):
            raise ValueError("prepared paper timeframe hash differs")
        frame = pd.read_parquet(path)
        if len(frame) != int(record.get("rows", -1)):
            raise ValueError("prepared paper timeframe row count differs")
        frames[name] = _finalize_bars(frame)
    data = LoadedPaperData(
        frames=frames, source_digest=str(value["source_digest"]),
        source_manifest_sha256=str(value["source_manifest_sha256"]),
        dates=dates, phase=actual_phase,
        source_records=({"prepared_manifest_sha256": manifest_sha},),
        smoke_only=bool(value["smoke_only"]),
    )
    return data, manifest_path


def _load_stage_data(source: Path, *, phase: str) -> LoadedPaperData:
    source = Path(source)
    if source.is_dir() and (source / "manifest.receipt.json").is_file():
        return verify_prepared_data(source, phase=phase)[0]
    if phase == "xau_pretrain":
        return load_xau_timeframes(source)
    return load_gc_timeframes(source, phase=phase)


def _parent_positions(frame: pd.DataFrame, spec: AgentSpec, index: int) -> tuple[int, ...]:
    result: list[int] = []
    for parent in spec.inherited_positions:
        candidates = (parent, f"{parent}_position", f"{parent.lower()}_position")
        column = next((name for name in candidates if name in frame), None)
        if column is None:
            raise ValueError(f"paper frame lacks current {parent} position")
        value = int(frame.iloc[int(index)][column])
        if value not in ACTIONS:
            raise ValueError(f"paper parent {parent} position is invalid")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class PaperEventReport:
    transitions: int
    optimizer_updates: int
    action_renewals: int
    environment_steps: int
    reward_sum: float
    mean_loss: float | None


class PaperDQNTrainer:
    """The paper's plain Tanh DQN with distinct action and bar clocks.

    Epsilon is ``decay ** environment_steps`` with no floor.  The target is
    copied every 1,000 supplied environment bars, not every 1,000 optimizer
    updates.  No gradient clipping, Double-DQN, dueling head, or shaped reward
    is present.
    """

    def __init__(
        self, spec: AgentSpec, *, seed: int, device: str = "cpu",
        stream_seed: int | None = None,
    ) -> None:
        self.spec = spec
        self.agent = spec  # compatibility with the reusable trainer surface
        self.seed = int(seed)
        self.stream_seed = int(self.seed if stream_seed is None else stream_seed)
        self.device = torch.device(device)
        random.seed(self.stream_seed)
        np.random.seed(self.stream_seed)
        torch.manual_seed(self.stream_seed)
        if self.device.type == "cpu":
            raw_threads = os.environ.get("KX_BASEPAPER_TORCH_THREADS", "1")
            threads = int(raw_threads)
            if threads < 1:
                raise ValueError("KX_BASEPAPER_TORCH_THREADS must be positive")
            torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        self.online = VanillaQNetwork(
            spec.observation_dimension, spec.hidden_sizes, len(ACTIONS),
            activation="tanh",
        ).to(self.device)
        self.target = VanillaQNetwork(
            spec.observation_dimension, spec.hidden_sizes, len(ACTIONS),
            activation="tanh",
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.RMSprop(
            self.online.parameters(), lr=PAPER_LEARNING_RATE,
        )
        self.replay = ReplayBuffer(spec.replay_capacity, self.stream_seed)
        self._rng = np.random.default_rng(self.stream_seed)
        self.environment_steps = 0
        self.action_selections = 0
        self.updates = 0
        self.target_synchronizations = 0

    @property
    def epsilon(self) -> float:
        return float(self.spec.epsilon_decay ** self.environment_steps)

    def _array(self, observation: Sequence[float] | np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.spec.observation_dimension,):
            raise ValueError("paper DQN observation has the wrong dimension")
        if not np.isfinite(value).all():
            raise ValueError("paper DQN observation is non-finite")
        return value

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        value = self._array(observation)
        if explore and self._rng.random() < self.epsilon:
            action_index = int(self._rng.integers(0, len(ACTIONS)))
        else:
            with torch.no_grad():
                q = self.online(
                    torch.as_tensor(value, device=self.device).unsqueeze(0),
                )
                action_index = int(q.argmax(dim=1).item())
        self.action_selections += 1
        return int(ACTIONS[action_index])

    def append(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminal: bool = False,
    ) -> None:
        if int(action) not in ACTIONS:
            raise ValueError("paper DQN replay action is invalid")
        self.replay.append(ReplayItem(
            self._array(observation).copy(), ACTIONS.index(int(action)), float(reward),
            self._array(next_observation).copy(), bool(terminal),
        ))

    def update(self) -> float | None:
        if len(self.replay) < PAPER_BATCH_SIZE:
            return None
        batch = self.replay.sample(PAPER_BATCH_SIZE)
        observations = torch.as_tensor(
            np.stack([item.observation for item in batch]), device=self.device,
        )
        actions = torch.as_tensor(
            [item.action_index for item in batch], dtype=torch.long, device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device,
        )
        next_observations = torch.as_tensor(
            np.stack([item.next_observation for item in batch]), device=self.device,
        )
        terminals = torch.as_tensor(
            [item.terminal for item in batch], dtype=torch.float32, device=self.device,
        )
        predicted = self.online(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            continuation = self.target(next_observations).max(dim=1).values
            target = rewards + self.spec.gamma * (1.0 - terminals) * continuation
        loss = F.mse_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach().cpu().item())

    def advance_environment_step(self) -> None:
        self.environment_steps += 1
        if self.environment_steps % PAPER_TARGET_PERIOD == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_synchronizations += 1

    def greedy(self, observation: np.ndarray) -> int:
        before = self.action_selections
        result = self.act(observation, explore=False)
        self.action_selections = before
        return result

    def save_checkpoint(
        self, path: Path, *, stage: str, request_sha256: str,
        report: Mapping[str, object] | None = None,
    ) -> Path:
        if str(stage) != PAPER_TRAINING_STAGE:
            raise ValueError("production paper checkpoints are GC-from-scratch only")
        if self.stream_seed != _agent_stream_seed(self.seed, self.spec.name):
            raise ValueError("production paper checkpoint lacks its independent agent RNG stream")
        payload: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": "BASEPAPER_DESCRIBED_AGENT_CHECKPOINT",
            "name": PAPER_EXACT_NAME,
            "stage": str(stage),
            "initialization": PAPER_INITIALIZATION,
            "warm_start": False,
            "parent_manifest_sha256": None,
            "seed": self.seed,
            "stream_seed": self.stream_seed,
            "agent": self.spec.name,
            "request_sha256": str(request_sha256),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "action_selections": self.action_selections,
            "updates": self.updates,
            "target_synchronizations": self.target_synchronizations,
            "parameter_sha256": _model_sha256(self.online),
            "report": dict(report or {}),
        }
        return _atomic_torch(Path(path), payload)


def _model_sha256(model: nn.Module) -> str:
    return _state_dict_sha256(model.state_dict())


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def verify_training_artifact(
    root: Path,
    *,
    seed: int | None = None,
    stage: str | None = None,
    request_sha256: str | None = None,
) -> dict[str, object]:
    """Deep-verify a completed immutable paper training manifest."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("paper training manifest is absent")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("artifact_type") != "BASEPAPER_DESCRIBED_TRAINING"
        or value.get("name") != PAPER_EXACT_NAME
        or value.get("complete") is not True
        or value.get("stage") != PAPER_TRAINING_STAGE
        or value.get("initialization") != PAPER_INITIALIZATION
        or value.get("warm_start") is not False
        or value.get("parent_manifest_sha256") is not None
        or "weights_only_xau_to_gc_reset" in value
        or not isinstance(value.get("smoke_only"), bool)
    ):
        raise ValueError("paper training manifest identity differs")
    request = value.get("request")
    if (
        not isinstance(request, dict)
        or _canonical_sha256(request) != value.get("request_sha256")
        or request.get("name") != PAPER_EXACT_NAME
        or request.get("stage") != PAPER_TRAINING_STAGE
        or request.get("initialization") != PAPER_INITIALIZATION
        or request.get("warm_start") is not False
        or request.get("parent_manifest_sha256") is not None
        or "pretrained_ensemble" in request
        or int(request.get("seed", -1)) != int(value.get("seed", -2))
        or request.get("source_digest") != value.get("source_digest")
        or request.get("source_manifest_sha256")
        != value.get("source_manifest_sha256")
        or request.get("input_records_sha256") != value.get("input_records_sha256")
        or request.get("dates") != value.get("dates")
        or request.get("maximum_events_smoke_limit")
        != value.get("maximum_events_smoke_limit")
    ):
        raise ValueError("paper training request digest differs")
    if value["smoke_only"] is False:
        reports = value.get("agents")
        if (
            value.get("maximum_events_smoke_limit") is not None
            or value.get("source_manifest_sha256")
            != EXPECTED_GC_MANIFEST_SHA256["train"]
            or _canonical_sha256(list(value.get("dates", [])))
            != EXPECTED_PHASE_DATE_SHA256["train"]
            or not isinstance(reports, dict)
            or set(reports) != set(AGENT_ORDER)
            or any(not isinstance(reports[name], dict) for name in AGENT_ORDER)
            or {
                name: int(reports.get(name, {}).get("transitions", -1))
                for name in AGENT_ORDER
            }
            != EXPECTED_GC_TRAIN_TRANSITIONS
            or {
                name: int(reports.get(name, {}).get("optimizer_updates", -1))
                for name in AGENT_ORDER
            }
            != EXPECTED_GC_TRAIN_UPDATES
            or {
                name: int(reports.get(name, {}).get("action_renewals", -1))
                for name in AGENT_ORDER
            }
            != EXPECTED_GC_TRAIN_RENEWALS
        ):
            raise ValueError("full paper training report/source support differs")
    if seed is not None and int(value.get("seed", -1)) != int(seed):
        raise ValueError("paper training seed differs")
    if stage is not None and value.get("stage") != stage:
        raise ValueError("paper training stage differs")
    if request_sha256 is not None and value.get("request_sha256") != request_sha256:
        raise ValueError("paper training request hash differs")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(AGENT_ORDER):
        raise ValueError("paper training artifact inventory is incomplete")
    resolved = root.resolve()
    reports = value.get("agents")
    if not isinstance(reports, dict) or set(reports) != set(AGENT_ORDER):
        raise ValueError("paper training report support is incomplete")
    seen: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError("paper training artifact record is malformed")
        path = (resolved / str(record["path"])).resolve()
        try:
            path.relative_to(resolved)
        except ValueError as error:
            raise ValueError("paper training artifact escapes its root") from error
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise ValueError("paper training checkpoint hash differs")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        agent = str(checkpoint.get("agent", ""))
        seen.add(agent)
        report = checkpoint.get("report")
        declared_report = reports.get(agent)
        if (
            checkpoint.get("artifact_type") != "BASEPAPER_DESCRIBED_AGENT_CHECKPOINT"
            or checkpoint.get("name") != PAPER_EXACT_NAME
            or checkpoint.get("stage") != value.get("stage")
            or checkpoint.get("initialization") != PAPER_INITIALIZATION
            or checkpoint.get("warm_start") is not False
            or checkpoint.get("parent_manifest_sha256") is not None
            or int(checkpoint.get("seed", -1)) != int(value.get("seed", -2))
            or int(checkpoint.get("stream_seed", -1))
            != _agent_stream_seed(int(value.get("seed", -2)), agent)
            or checkpoint.get("request_sha256") != value.get("request_sha256")
            or int(checkpoint.get("updates", 0)) <= 0
        ):
            raise ValueError("paper training checkpoint identity/support differs")
        if (
            agent not in AGENT_ORDER or path.stem != agent.lower()
            or not isinstance(report, dict) or report != declared_report
            or int(report.get("transitions", 0))
            != int(report.get("environment_steps", -1))
            or int(report.get("environment_steps", 0))
            != int(checkpoint.get("environment_steps", -1))
            or int(report.get("optimizer_updates", 0))
            != int(checkpoint.get("updates", -1))
            or int(report.get("action_renewals", 0))
            != int(checkpoint.get("action_selections", -1))
            or int(report.get("transitions", 0)) <= 0
            or int(report.get("optimizer_updates", 0)) <= 0
        ):
            raise ValueError("paper training checkpoint report/support differs")
        online = checkpoint.get("online")
        if (
            not isinstance(online, dict)
            or _state_dict_sha256(online) != checkpoint.get("parameter_sha256")
        ):
            raise ValueError("paper training online parameter hash differs")
    if seen != set(AGENT_ORDER):
        raise ValueError("paper training checkpoint agent support differs")
    return value


def run_paper_agent_events(
    trainer: Any,
    frame: pd.DataFrame,
    spec: AgentSpec,
) -> PaperEventReport:
    """Replay Algorithm 1: held action, transition/update/clock on every bar."""

    origins = paper_origins(frame, spec)
    if not len(origins):
        raise ValueError(f"{spec.name} has zero transition support")
    segments = paper_segments(frame)
    current_action: int | None = None
    current_segment: int | None = None
    held_bars = 0
    renewals = updates = 0
    losses: list[float] = []
    rewards: list[float] = []
    for raw_origin in origins:
        origin = int(raw_origin)
        segment = int(segments[origin])
        if segment != current_segment:
            current_action = None
            current_segment = segment
            held_bars = 0
        current_parents = _parent_positions(frame, spec, origin)
        observation = paper_observation(
            frame, origin, spec.lookback, current_parents,
        )
        if current_action is None or held_bars == int(spec.action_lock_bars):
            current_action = int(trainer.act(observation, explore=True))
            if current_action not in ACTIONS:
                raise ValueError("paper trainer selected an invalid action")
            held_bars = 0
            renewals += 1
        following_parents = _parent_positions(frame, spec, origin + 1)
        next_observation = paper_observation(
            frame, origin + 1, spec.lookback, following_parents,
        )
        current_close = float(frame.iloc[origin]["close"])
        future_close = float(frame.iloc[origin + int(spec.action_lock_bars)]["close"])
        if not np.isfinite((current_close, future_close)).all() or current_close <= 0:
            raise ValueError("paper reward price is invalid")
        reward = float(current_action) * (future_close / current_close - 1.0)
        trainer.append(
            observation, current_action, reward, next_observation, terminal=False,
        )
        loss = trainer.update()
        if loss is not None:
            updates += 1
            losses.append(float(loss))
        trainer.advance_environment_step()
        rewards.append(reward)
        held_bars += 1
    if len(rewards) != len(origins):
        raise AssertionError("paper transition count differs from eligible origins")
    if updates == 0:
        raise ValueError(f"{spec.name} has zero optimizer update support")
    return PaperEventReport(
        transitions=len(rewards), optimizer_updates=updates,
        action_renewals=renewals, environment_steps=len(rewards),
        reward_sum=float(np.sum(rewards)),
        mean_loss=float(np.mean(losses)) if losses else None,
    )


@dataclass
class _PendingPaperTransition:
    next_index: int
    observation: np.ndarray
    action: int
    reward: float


def _trainer_spec(trainer: Any) -> AgentSpec:
    spec = getattr(trainer, "spec", getattr(trainer, "agent", None))
    if spec is None or not isinstance(spec, AgentSpec):
        raise TypeError("joint paper trainer must expose an AgentSpec as .spec or .agent")
    return spec


def _row_symbol(frame: pd.DataFrame, index: int) -> str | None:
    return str(frame.iloc[int(index)]["raw_symbol"]) if "raw_symbol" in frame else None


def run_joint_paper_pass(
    trainers: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, PaperEventReport]:
    """Train H1, M15, and M5 concurrently in chronological hierarchy order.

    At a shared completed-bar timestamp H1 is processed before M15, which is
    processed before M5.  Thus each lower state contains the current exploratory
    parent positions, not a post-training greedy trace.  A transition is
    finalized at its actual t+1 row, then its agent selects/reuses the action
    for the new row.
    """

    if set(trainers) != set(AGENT_ORDER) or set(frames) != set(AGENT_ORDER):
        raise ValueError(f"joint paper pass requires exactly {AGENT_ORDER}")
    specs = {name: _trainer_spec(trainers[name]) for name in AGENT_ORDER}
    if any(specs[name].name != name for name in AGENT_ORDER):
        raise ValueError("joint paper trainer names differ from hierarchy order")

    times = {name: _event_ns(frames[name]) for name in AGENT_ORDER}
    segments = {name: paper_segments(frames[name]) for name in AGENT_ORDER}
    origins = {name: paper_origins(frames[name], specs[name]) for name in AGENT_ORDER}
    eligible = {
        name: np.isin(np.arange(len(frames[name]), dtype=np.int64), origins[name])
        for name in AGENT_ORDER
    }
    if any(not len(origins[name]) for name in AGENT_ORDER):
        absent = [name for name in AGENT_ORDER if not len(origins[name])]
        raise ValueError(f"joint paper pass has zero transition support for {absent}")

    # Merge all supplied rows without materializing a large timestamp->event map.
    heap: list[tuple[int, int, int]] = []
    pointers = {name: 0 for name in AGENT_ORDER}
    for ordinal, name in enumerate(AGENT_ORDER):
        if len(times[name]):
            heapq.heappush(heap, (int(times[name][0]), ordinal, 0))

    current_actions: dict[str, int | None] = {name: None for name in AGENT_ORDER}
    positions = {name: 0 for name in AGENT_ORDER}
    held_bars = {name: 0 for name in AGENT_ORDER}
    last_segments: dict[str, int | None] = {name: None for name in AGENT_ORDER}
    pending: dict[str, _PendingPaperTransition | None] = {
        name: None for name in AGENT_ORDER
    }
    transition_counts = {name: 0 for name in AGENT_ORDER}
    update_counts = {name: 0 for name in AGENT_ORDER}
    renewal_counts = {name: 0 for name in AGENT_ORDER}
    rewards: dict[str, list[float]] = {name: [] for name in AGENT_ORDER}
    losses: dict[str, list[float]] = {name: [] for name in AGENT_ORDER}
    global_symbol: str | None = None

    while heap:
        event_time = heap[0][0]
        rows: dict[str, int] = {}
        while heap and heap[0][0] == event_time:
            _, ordinal, index = heapq.heappop(heap)
            name = AGENT_ORDER[ordinal]
            rows[name] = index
            following = index + 1
            pointers[name] = following
            if following < len(times[name]):
                heapq.heappush(
                    heap, (int(times[name][following]), ordinal, following),
                )

        # A contract roll resets all communicated positions before any lower
        # policy at that timestamp can observe a stale parent.
        visible_symbols = [
            _row_symbol(frames[name], rows[name]) for name in AGENT_ORDER if name in rows
        ]
        visible_symbols = [value for value in visible_symbols if value is not None]
        if visible_symbols and len(set(visible_symbols)) != 1:
            raise ValueError("paper hierarchy has conflicting symbols at one timestamp")
        if visible_symbols:
            symbol = visible_symbols[0]
            if global_symbol is not None and symbol != global_symbol:
                current_actions = {name: None for name in AGENT_ORDER}
                positions = {name: 0 for name in AGENT_ORDER}
                held_bars = {name: 0 for name in AGENT_ORDER}
            global_symbol = symbol

        for name in AGENT_ORDER:
            if name not in rows:
                continue
            index = rows[name]
            spec = specs[name]
            segment = int(segments[name][index])
            if last_segments[name] is None or segment != last_segments[name]:
                current_actions[name] = None
                positions[name] = 0
                held_bars[name] = 0
                last_segments[name] = segment

            parent_values = tuple(positions[parent] for parent in spec.inherited_positions)
            old = pending[name]
            if old is not None and old.next_index == index:
                next_observation = paper_observation(
                    frames[name], index, spec.lookback, parent_values,
                )
                trainer = trainers[name]
                trainer.append(
                    old.observation, old.action, old.reward, next_observation,
                    terminal=False,
                )
                loss = trainer.update()
                if loss is not None:
                    update_counts[name] += 1
                    losses[name].append(float(loss))
                trainer.advance_environment_step()
                transition_counts[name] += 1
                rewards[name].append(old.reward)
                pending[name] = None

            if not bool(eligible[name][index]):
                continue
            observation = paper_observation(
                frames[name], index, spec.lookback, parent_values,
            )
            if (
                current_actions[name] is None
                or held_bars[name] == int(spec.action_lock_bars)
            ):
                selected = int(trainers[name].act(observation, explore=True))
                if selected not in ACTIONS:
                    raise ValueError("joint paper trainer selected an invalid action")
                current_actions[name] = selected
                positions[name] = selected
                held_bars[name] = 0
                renewal_counts[name] += 1
            action = int(current_actions[name])
            current_close = float(frames[name].iloc[index]["close"])
            future_close = float(
                frames[name].iloc[index + int(spec.action_lock_bars)]["close"]
            )
            reward = action * (future_close / current_close - 1.0)
            pending[name] = _PendingPaperTransition(
                next_index=index + 1, observation=observation,
                action=action, reward=float(reward),
            )
            held_bars[name] += 1

    reports: dict[str, PaperEventReport] = {}
    for name in AGENT_ORDER:
        expected = len(origins[name])
        if pending[name] is not None or transition_counts[name] != expected:
            raise ValueError(f"{name} transition support did not replay exactly")
        if update_counts[name] == 0:
            raise ValueError(f"{name} has zero optimizer update support")
        reports[name] = PaperEventReport(
            transitions=transition_counts[name], optimizer_updates=update_counts[name],
            action_renewals=renewal_counts[name], environment_steps=transition_counts[name],
            reward_sum=float(np.sum(rewards[name])),
            mean_loss=float(np.mean(losses[name])) if losses[name] else None,
        )
    return reports


def _capped_frames(
    frames: Mapping[str, pd.DataFrame], maximum_events: int | None,
) -> dict[str, pd.DataFrame]:
    if maximum_events is None:
        return {name: frames[name] for name in AGENT_ORDER}
    if int(maximum_events) < PAPER_BATCH_SIZE:
        raise ValueError(
            f"maximum_events must be at least {PAPER_BATCH_SIZE} for optimizer support"
        )
    contract = registered_contract("base_paper_gc")
    result: dict[str, pd.DataFrame] = {}
    for name in AGENT_ORDER:
        frame = frames[name]
        origins = paper_origins(frame, contract.agent(name))
        if len(origins) < int(maximum_events):
            raise ValueError(f"{name} lacks requested capped transition support")
        last = (
            int(origins[int(maximum_events) - 1])
            + int(contract.agent(name).action_lock_bars)
        )
        result[name] = frame.iloc[:last + 1].reset_index(drop=True)
    return result


def paper_effective_contract() -> dict[str, object]:
    """Return the local effective contract, superseding old reconstruction defaults."""

    source = registered_contract("base_paper_gc")
    return {
        "contract_id": PAPER_PROTOCOL_VERSION,
        "name": PAPER_EXACT_NAME,
        "agents": {
            name: {
                "minutes": source.agent(name).minutes,
                "lookback": source.agent(name).lookback,
                "action_lock_bars": source.agent(name).action_lock_bars,
                "gamma": source.agent(name).gamma,
                "epsilon_decay": source.agent(name).epsilon_decay,
                "replay_capacity": source.agent(name).replay_capacity,
                "hidden_sizes": list(source.agent(name).hidden_sizes),
                "inherited_positions": list(source.agent(name).inherited_positions),
            }
            for name in AGENT_ORDER
        },
        "observation": "raw_OHLCV_newest_to_oldest_plus_current_scalar_parents",
        "actions": list(ACTIONS),
        "reward": "action*(close[t+tw]/close[t]-1)",
        "network": "plain_Tanh_DQN",
        "optimizer": "RMSprop",
        "learning_rate": PAPER_LEARNING_RATE,
        "loss": "MSE",
        "target_rule": "vanilla_DQN",
        "batch_size_unpublished_default": PAPER_BATCH_SIZE,
        "target_C_unpublished_default": PAPER_TARGET_PERIOD,
        "target_C_clock": "environment_bar",
        "epsilon_start_unpublished_default": 1.0,
        "epsilon_floor": None,
        "gradient_clipping": None,
        "passes_per_stage": 1,
        "training_instrument": "GC",
        "initialization": PAPER_INITIALIZATION,
        "agent_rng_streams": "master_seed_times_10_plus_H1_1_M15_2_M5_3",
        "warm_start": False,
        "pretraining": None,
        "session_filter": SESSION_FILTER_DESCRIPTION,
        "timeframe_construction": TIMEFRAME_DESCRIPTION,
        "full_gc_support": {
            "bars": EXPECTED_GC_TRAIN_BARS,
            "transitions": EXPECTED_GC_TRAIN_TRANSITIONS,
            "optimizer_updates": EXPECTED_GC_TRAIN_UPDATES,
            "action_renewals": EXPECTED_GC_TRAIN_RENEWALS,
        },
    }


def _stage_request(
    data: LoadedPaperData, *, seed: int, maximum_events: int | None,
) -> tuple[dict[str, object], str]:
    contract = registered_contract("base_paper_gc")
    effective = paper_effective_contract()
    request: dict[str, object] = {
        "name": PAPER_EXACT_NAME,
        "stage": PAPER_TRAINING_STAGE,
        "seed": int(seed),
        "source_digest": data.source_digest,
        "source_manifest_sha256": data.source_manifest_sha256,
        "input_records_sha256": _canonical_sha256(list(data.source_records)),
        "dates": list(data.dates),
        "initialization": PAPER_INITIALIZATION,
        "warm_start": False,
        "parent_manifest_sha256": None,
        "maximum_events_smoke_limit": maximum_events,
        "code_sha256": _sha256_file(Path(__file__)),
        "agent_spec_source_contract_sha256": contract.sha256,
        "effective_contract": effective,
        "effective_contract_sha256": _canonical_sha256(effective),
        "published": {
            "actions": list(ACTIONS), "agent_order": list(AGENT_ORDER),
            "learning_rate": PAPER_LEARNING_RATE, "activation": "tanh",
            "optimizer": "RMSprop", "loss": "MSE", "target_rule": "vanilla",
            "network_widths": {
                name: list(contract.agent(name).hidden_sizes) for name in AGENT_ORDER
            },
            "lookbacks": {name: contract.agent(name).lookback for name in AGENT_ORDER},
            "locks": {name: contract.agent(name).action_lock_bars for name in AGENT_ORDER},
            "gammas": {name: contract.agent(name).gamma for name in AGENT_ORDER},
            "epsilon_decays": {
                name: contract.agent(name).epsilon_decay for name in AGENT_ORDER
            },
            "replay_capacities": {
                name: contract.agent(name).replay_capacity for name in AGENT_ORDER
            },
        },
        "reconstruction_defaults": {
            "batch_size": PAPER_BATCH_SIZE,
            "target_period_environment_bars": PAPER_TARGET_PERIOD,
            "epsilon_start": 1.0, "epsilon_floor": None,
            "gradient_clipping": None,
        },
        "training": (
            "seeded_random_from_scratch_on_GC;"
            "one_chronological_joint_H1_then_M15_then_M5_pass;"
            "one_transition_and_adaptation_attempt_per_eligible_bar;"
            "current_exploratory_parent_positions;offline_t_plus_lock_reward_"
            "attached_when_t_plus_one_successor_is_processed_as_in_paper_algorithm"
        ),
    }
    return request, _canonical_sha256(request)


def _train_stage(
    data: LoadedPaperData,
    output: Path,
    seed: int,
    *,
    maximum_events: int | None = None,
) -> Path:
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if data.phase != "train":
        raise ValueError("paper training data phase differs from stage")
    if maximum_events is None:
        if data.smoke_only:
            raise ValueError("full paper training refuses prepared smoke coverage")
        if (
            data.source_manifest_sha256 != EXPECTED_GC_MANIFEST_SHA256["train"]
            or _canonical_sha256(list(data.dates)) != EXPECTED_PHASE_DATE_SHA256["train"]
        ):
            raise ValueError("full paper GC training opportunity set differs")
        actual_bars = {name: len(data.frames[name]) for name in AGENT_ORDER}
        actual_transitions = {
            name: len(paper_origins(
                data.frames[name], registered_contract("base_paper_gc").agent(name),
            ))
            for name in AGENT_ORDER
        }
        if (
            actual_bars != EXPECTED_GC_TRAIN_BARS
            or actual_transitions != EXPECTED_GC_TRAIN_TRANSITIONS
        ):
            raise ValueError("full paper GC prepared bar/transition support differs")

    request, request_sha = _stage_request(
        data, seed=seed, maximum_events=maximum_events,
    )
    output = Path(output).resolve()
    if (output / "manifest.json").is_file():
        verify_training_artifact(
            output, seed=seed, stage=PAPER_TRAINING_STAGE,
            request_sha256=request_sha,
        )
        return output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("incomplete paper training output is not reusable")
    output.mkdir(parents=True, exist_ok=True)

    contract = registered_contract("base_paper_gc")
    trainers = {
        name: PaperDQNTrainer(
            contract.agent(name), seed=seed, device="cpu",
            stream_seed=_agent_stream_seed(seed, name),
        )
        for name in AGENT_ORDER
    }
    selected_frames = _capped_frames(data.frames, maximum_events)
    reports = run_joint_paper_pass(trainers, selected_frames)
    if maximum_events is None:
        actual_updates = {
            name: reports[name].optimizer_updates for name in AGENT_ORDER
        }
        actual_renewals = {name: reports[name].action_renewals for name in AGENT_ORDER}
        if (
            actual_updates != EXPECTED_GC_TRAIN_UPDATES
            or actual_renewals != EXPECTED_GC_TRAIN_RENEWALS
        ):
            raise AssertionError("full paper GC update/renewal support differs")
    artifacts: list[dict[str, str]] = []
    for name in AGENT_ORDER:
        if reports[name].transitions != len(
            paper_origins(selected_frames[name], contract.agent(name))
        ):
            raise AssertionError("paper stage transition count differs from exact support")
        checkpoint = output / f"{name.lower()}.pt"
        trainers[name].save_checkpoint(
            checkpoint, stage=PAPER_TRAINING_STAGE, request_sha256=request_sha,
            report=asdict(reports[name]),
        )
        artifacts.append({"path": checkpoint.name, "sha256": _sha256_file(checkpoint)})
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "BASEPAPER_DESCRIBED_TRAINING",
        "name": PAPER_EXACT_NAME,
        "complete": True,
        "stage": PAPER_TRAINING_STAGE,
        "seed": int(seed),
        "request": request,
        "request_sha256": request_sha,
        "source_digest": data.source_digest,
        "source_manifest_sha256": data.source_manifest_sha256,
        "input_records_sha256": _canonical_sha256(list(data.source_records)),
        "dates": list(data.dates),
        "initialization": PAPER_INITIALIZATION,
        "warm_start": False,
        "parent_manifest_sha256": None,
        "passes": 1,
        "observation": "raw_OHLCV_newest_to_oldest_plus_scalar_current_parent_positions",
        "reward": "action_times_close_t_plus_tw_over_close_t_minus_one_gross",
        "session_filter": "07:20_inclusive_to_12:25_exclusive_America_Chicago",
        "smoke_only": maximum_events is not None,
        "maximum_events_smoke_limit": maximum_events,
        "agents": {name: asdict(reports[name]) for name in AGENT_ORDER},
        "artifacts": artifacts,
    }
    _atomic_json(output / "manifest.json", manifest)
    verify_training_artifact(
        output, seed=seed, stage=PAPER_TRAINING_STAGE, request_sha256=request_sha,
    )
    return output / "manifest.json"


def train_gc_from_scratch(
    gc_train_or_prepared: Path,
    output: Path,
    seed: int,
    *,
    maximum_events: int | None = None,
) -> Path:
    data = _load_stage_data(gc_train_or_prepared, phase="train")
    return _train_stage(
        data, output, seed, maximum_events=maximum_events,
    )


def _seed_job(
    source: str, output: str, seed: int, maximum_events: int | None,
) -> str:
    os.environ.setdefault("KX_BASEPAPER_TORCH_THREADS", "1")
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    destination = Path(output) / f"seed_{seed}"
    result = train_gc_from_scratch(
        Path(source), destination, seed, maximum_events=maximum_events,
    )
    return str(result)


def verify_ensemble_training(
    root: Path, *, stage: str | None = None,
    request_sha256: str | None = None,
) -> dict[str, object]:
    root = Path(root).resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("paper ensemble training manifest is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    request = value.get("request")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != "BASEPAPER_DESCRIBED_ENSEMBLE_TRAINING"
        or value.get("name") != PAPER_EXACT_NAME
        or value.get("complete") is not True
        or value.get("stage") != PAPER_TRAINING_STAGE
        or value.get("initialization") != PAPER_INITIALIZATION
        or value.get("warm_start") is not False
        or value.get("pretrained_ensemble") is not None
        or not isinstance(request, dict)
        or _canonical_sha256(request) != value.get("request_sha256")
        or request.get("name") != PAPER_EXACT_NAME
        or request.get("stage") != PAPER_TRAINING_STAGE
        or request.get("initialization") != PAPER_INITIALIZATION
        or request.get("warm_start") is not False
        or request.get("pretrained_ensemble") is not None
    ):
        raise ValueError("paper ensemble training identity/request differs")
    if stage is not None and value.get("stage") != stage:
        raise ValueError("paper ensemble training stage differs")
    if request_sha256 is not None and value.get("request_sha256") != request_sha256:
        raise ValueError("paper ensemble training request hash differs")
    members = value.get("members")
    if not isinstance(members, list) or [row.get("seed") for row in members] != list(SEEDS):
        raise ValueError("paper ensemble seed inventory differs")
    for row in members:
        if not isinstance(row, dict) or set(row) != {"seed", "path", "sha256"}:
            raise ValueError("paper ensemble member record is malformed")
        member_path = (root / str(row["path"])).resolve()
        try:
            member_path.relative_to(root)
        except ValueError as error:
            raise ValueError("paper ensemble member escapes its root") from error
        if not member_path.is_file() or _sha256_file(member_path) != row["sha256"]:
            raise ValueError("paper ensemble member manifest hash differs")
        verify_training_artifact(
            member_path.parent, seed=int(row["seed"]), stage=str(value["stage"]),
        )
    return value


def _request_source_identity(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    manifest = path / "manifest.json" if path.is_dir() else path
    if not manifest.is_file():
        raise ValueError("paper five-seed source manifest is absent")
    return {"path": str(path), "manifest_sha256": _sha256_file(manifest)}


def train_five_seeds(
    stage: str,
    source: Path,
    output: Path,
    *,
    workers: int = 5,
    maximum_events: int | None = None,
) -> Path:
    if stage != PAPER_TRAINING_STAGE:
        raise ValueError("five-seed paper production stage must be GC from scratch")
    if not 1 <= int(workers) <= 12:
        raise ValueError("paper worker count must be between 1 and 12")
    source = Path(source).resolve()
    if not source.is_dir() or not (source / "manifest.receipt.json").is_file():
        raise ValueError("five-seed paper training requires prepared GC bars")
    verify_prepared_data(source, phase="train")
    output = Path(output).resolve()
    request: dict[str, object] = {
        "name": PAPER_EXACT_NAME,
        "stage": PAPER_TRAINING_STAGE,
        "source": _request_source_identity(source),
        "initialization": PAPER_INITIALIZATION,
        "warm_start": False,
        "pretrained_ensemble": None,
        "workers": int(workers),
        "maximum_events_smoke_limit": maximum_events,
        "seeds": list(SEEDS),
        "code_sha256": _sha256_file(Path(__file__)),
        "effective_contract_sha256": _canonical_sha256(paper_effective_contract()),
    }
    request_sha = _canonical_sha256(request)
    if (output / "manifest.json").is_file():
        verify_ensemble_training(
            output, stage=stage, request_sha256=request_sha,
        )
        return output / "manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(source), str(output), seed, maximum_events,
        )
        for seed in SEEDS
    ]
    manifests: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(jobs)), mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_seed_job, *job): job[-2] for job in jobs}
        for future in as_completed(futures):
            manifests.append(Path(future.result()))
    manifests.sort(key=lambda path: int(path.parent.name.split("_")[-1]))
    if tuple(int(path.parent.name.split("_")[-1]) for path in manifests) != SEEDS:
        raise AssertionError("paper five-seed result set differs")
    ensemble: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "BASEPAPER_DESCRIBED_ENSEMBLE_TRAINING",
        "name": PAPER_EXACT_NAME,
        "complete": True,
        "stage": stage,
        "initialization": PAPER_INITIALIZATION,
        "warm_start": False,
        "pretrained_ensemble": None,
        "request": request,
        "request_sha256": request_sha,
        "seeds": list(SEEDS),
        "workers": int(workers),
        "maximum_events_smoke_limit": maximum_events,
        "members": [
            {
                "seed": int(path.parent.name.split("_")[-1]),
                "path": os.path.relpath(path, start=output),
                "sha256": _sha256_file(path),
            }
            for path in manifests
        ],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("paper ensemble manifest appeared during training")
    result = _atomic_json(manifest_path, ensemble)
    verify_ensemble_training(output, stage=stage, request_sha256=request_sha)
    return result


@dataclass(frozen=True)
class PaperLedgerReplay:
    result: V6SessionResult
    attributed_reward_cents: tuple[int, ...]
    direct_reversal_decisions: int


def _causal_indexes(
    raw: Mapping[str, np.ndarray], session_start_ns: int, session_end_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    boundaries = np.arange(
        int(session_start_ns) + 5 * NS_PER_MINUTE,
        int(session_end_ns), 5 * NS_PER_MINUTE, dtype=np.int64,
    )
    indexes = np.searchsorted(clock, boundaries, side="right") - 1
    indexes = np.unique(indexes[indexes >= 0]).astype(np.int64)
    valid = np.asarray(raw["valid"], dtype=bool)
    indexes = indexes[valid[indexes]]
    terminal = len(clock) - 1
    if not len(indexes) or indexes[-1] >= terminal or not bool(valid[terminal]):
        raise ValueError("paper replay lacks causal decisions or terminal book")
    chain = np.r_[indexes, terminal]
    continuity = np.asarray(raw["continuity_segment"])[chain]
    if len(set(int(value) for value in continuity)) != 1:
        raise ValueError("paper replay crosses a continuity segment")
    return indexes, chain


def replay_paper_targets(
    raw: Mapping[str, np.ndarray],
    *,
    session_date: str,
    session_start_ns: int,
    session_end_ns: int,
    signal_times_ns: np.ndarray,
    signal_actions: np.ndarray,
    hold_minutes: int,
    economics: ExecutionEconomics | None = None,
) -> PaperLedgerReplay:
    """Replay M5 targets with the paper's immediate close-and-reverse rule."""

    economics = economics or ExecutionEconomics.from_basepaper()
    _validate_raw(raw, economics)
    times = np.asarray(signal_times_ns, dtype=np.int64)
    actions = np.asarray(signal_actions, dtype=np.int8)
    if times.ndim != 1 or times.shape != actions.shape or (
        len(times) > 1 and (np.diff(times) <= 0).any()
    ):
        raise ValueError("paper signal times/actions must be aligned and ordered")
    if not np.isin(actions, ACTIONS).all() or int(hold_minutes) <= 0:
        raise ValueError("paper trace contains an invalid action or holding time")
    indexes, chain = _causal_indexes(raw, session_start_ns, session_end_ns)
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    ledger = EconomicLedger(economics)
    attributed = np.zeros(len(times), dtype=np.int64)
    rewards: list[int] = []
    fills: list[Any] = []
    rejected = reversals = 0
    expiries = times + int(hold_minutes) * NS_PER_MINUTE
    previous_equity = 0
    last_owner: int | None = None

    def route_to(index: int, target: int, *, forced: bool = False) -> bool:
        nonlocal rejected
        if target == ledger.inventory:
            return True
        side = 1 if target > ledger.inventory else -1
        route = _resolve_route(raw, int(index), side, economics)
        if route.rejected or route.book is None:
            rejected += 1
            return False
        required_size = abs(int(target) - int(ledger.inventory))
        displayed_size = route.book.ask_size if side > 0 else route.book.bid_size
        if displayed_size < required_size:
            # A reversal is one atomic target command but two filled sides.  Do
            # not close the old leg unless the same arrival book can also open
            # the opposite leg.
            rejected += 1
            return False
        fills.extend(ledger.move_to_target(target, route.book, forced=forced))
        return True

    for ordinal, raw_index in enumerate(indexes):
        index = int(raw_index)
        boundary = int(clock[index])
        signal = int(np.searchsorted(times, boundary, side="right") - 1)
        active = signal >= 0 and boundary < int(expiries[signal])
        desired = int(actions[signal]) if active else 0
        if ledger.inventory and desired == -ledger.inventory:
            reversals += 1
        route_to(index, desired)
        terminal = ordinal == len(indexes) - 1
        next_index = int(chain[ordinal + 1])
        if terminal and ledger.inventory and not route_to(next_index, 0, forced=True):
            raise RuntimeError("mandatory paper terminal liquidation was rejected")
        equity = int(ledger.cash_cents) if terminal else int(
            ledger.value_cents(_visible_book(raw, next_index))
        )
        reward = equity - previous_equity
        previous_equity = equity
        rewards.append(reward)
        owner: int | None = signal if active else None
        if owner is None:
            prior = int(np.searchsorted(times, boundary, side="left") - 1)
            if prior >= 0 and boundary <= int(expiries[prior]):
                owner = prior
            elif last_owner is not None and reward != 0:
                owner = last_owner
        if owner is not None:
            attributed[owner] += reward
            last_owner = owner
        elif reward != 0:
            raise AssertionError("paper ledger reward has no causal signal owner")
    if ledger.inventory != 0:
        raise AssertionError("paper replay retained terminal inventory")
    net = int(ledger.cash_cents)
    if sum(rewards) != net or int(attributed.sum()) != net:
        raise AssertionError("paper replay rewards do not telescope to terminal cash")
    if len(fills) % 2:
        raise AssertionError("paper replay retained an unmatched fill")
    result = V6SessionResult(
        session_date=str(session_date), net_pnl_cents=net,
        raw_rewards_cents=tuple(rewards), completed_round_trips=len(fills) // 2,
        rejected_commands=rejected, fills=tuple(fills), transitions=(),
        terminal_reward_cents=rewards[-1] if rewards else 0,
        unsafe_legacy_arrival_boundary=uses_legacy_arrival_boundary(raw, economics),
    )
    return PaperLedgerReplay(
        result=result,
        attributed_reward_cents=tuple(int(value) for value in attributed),
        direct_reversal_decisions=reversals,
    )


__all__ = [
    "PAPER_EXACT_NAME",
    "PAPER_INITIALIZATION",
    "PAPER_PROTOCOL_VERSION",
    "PAPER_TRAINING_STAGE",
    "PREPARED_DATA_NAME",
    "PaperEventReport",
    "PaperLedgerReplay",
    "PaperDQNTrainer",
    "LoadedPaperData",
    "EXPECTED_GC_MANIFEST_SHA256",
    "EXPECTED_GC_TRAIN_BARS",
    "EXPECTED_GC_TRAIN_RENEWALS",
    "EXPECTED_GC_TRAIN_TRANSITIONS",
    "EXPECTED_GC_TRAIN_UPDATES",
    "EXPECTED_XAU_MANIFEST_SHA256",
    "SESSION_FILTER_DESCRIPTION",
    "TIMEFRAME_DESCRIPTION",
    "load_gc_timeframes",
    "load_xau_timeframes",
    "paper_observation",
    "paper_effective_contract",
    "paper_origins",
    "paper_segments",
    "prepare_gc",
    "prepare_xau",
    "replay_paper_targets",
    "run_paper_agent_events",
    "run_joint_paper_pass",
    "train_five_seeds",
    "train_gc_from_scratch",
    "verify_ensemble_training",
    "verify_prepared_data",
    "verify_training_artifact",
]
