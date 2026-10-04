from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from kx_base_gc.contract import registered_contract

from .splits import require_development_access


CHICAGO = ZoneInfo("America/Chicago")
DECODE_NS = 1_000_000
STALE_NS = 1_000_000_000
TICK_SIZE = 0.10
LEVELS = 10


@dataclass(frozen=True)
class SessionAudit:
    session_date: str
    raw_symbol: str
    l2_source_rows: int
    trade_source_rows: int
    clocks: int
    valid_clocks: int
    valid_fraction: float
    five_minute_coverage: float
    continuity_segments: int
    crossed_rejections: int
    stale_rejections: int
    incomplete_rejections: int
    unknown_operations: int
    status: str


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _session_clocks(day: str) -> np.ndarray:
    start = pd.Timestamp(f"{day} 07:20", tz=CHICAGO).tz_convert("UTC")
    end = pd.Timestamp(f"{day} 12:25", tz=CHICAGO).tz_convert("UTC")
    return pd.date_range(start, end, freq="250ms").asi8


def _decision_clocks(day: str) -> np.ndarray:
    start = pd.Timestamp(f"{day} 07:25", tz=CHICAGO).tz_convert("UTC")
    end = pd.Timestamp(f"{day} 12:25", tz=CHICAGO).tz_convert("UTC")
    return pd.date_range(start, end, freq="5min").asi8


def _ofi(bid: float, bid_size: float, ask: float, ask_size: float,
         previous: tuple[float, float, float, float] | None) -> float:
    if previous is None:
        return 0.0
    old_bid, old_bid_size, old_ask, old_ask_size = previous
    return float(
        (bid_size if bid >= old_bid else 0.0)
        - (old_bid_size if bid <= old_bid else 0.0)
        - (ask_size if ask <= old_ask else 0.0)
        + (old_ask_size if ask >= old_ask else 0.0)
    )


def reconstruct_l2(
    rows: pd.DataFrame,
    clocks_ns: np.ndarray,
    *,
    include_features: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Reconstruct causal top-ten observations from complete receive-time event groups."""
    required = {
        "event_index", "event_row_index", "ts_recv_ns", "continuity_segment_id",
        "operation", "side", "price_ticks", "level_size",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"L2 rows missing {sorted(missing)}")
    rows = rows.sort_values(["event_index", "event_row_index"], kind="stable")
    count = len(clocks_ns)
    level_values = np.zeros((count, LEVELS, 4), dtype=np.float32) if include_features else None
    engineered = np.zeros((count, 4), dtype=np.float32) if include_features else None
    best_bid = np.full(count, np.nan, dtype=np.float64)
    best_ask = np.full(count, np.nan, dtype=np.float64)
    best_bid_ticks = np.zeros(count, dtype=np.int64)
    best_ask_ticks = np.zeros(count, dtype=np.int64)
    best_bid_size = np.zeros(count, dtype=np.uint64)
    best_ask_size = np.zeros(count, dtype=np.uint64)
    available = np.full(count, -1, dtype=np.int64)
    continuity = np.zeros(count, dtype=np.uint32)
    valid = np.zeros(count, dtype=np.bool_)
    books: dict[str, dict[int, int]] = {"B": {}, "A": {}}
    state_available = -1
    state_segment = 0
    clock_index = 0
    previous_bbo: tuple[float, float, float, float] | None = None
    rejected = {"crossed": 0, "stale": 0, "incomplete": 0, "unknown": 0}

    def sample(index: int) -> None:
        nonlocal previous_bbo
        clock = int(clocks_ns[index])
        available[index] = state_available
        continuity[index] = state_segment
        if state_available < 0 or clock - state_available > STALE_NS:
            rejected["stale"] += 1
            return
        bids = sorted(books["B"].items(), reverse=True)[:LEVELS]
        asks = sorted(books["A"].items())[:LEVELS]
        if len(bids) < LEVELS or len(asks) < LEVELS:
            rejected["incomplete"] += 1
            return
        if bids[0][0] >= asks[0][0]:
            rejected["crossed"] += 1
            return
        bid = bids[0][0] * TICK_SIZE
        ask = asks[0][0] * TICK_SIZE
        mid = (bid + ask) / 2.0
        bid_size, ask_size = float(bids[0][1]), float(asks[0][1])
        if include_features:
            assert level_values is not None and engineered is not None
            for level, ((bid_tick, bsize), (ask_tick, asize)) in enumerate(zip(bids, asks, strict=True)):
                level_values[index, level] = (
                    np.float32((mid - bid_tick * TICK_SIZE) / TICK_SIZE), np.float32(np.log1p(bsize)),
                    np.float32((ask_tick * TICK_SIZE - mid) / TICK_SIZE), np.float32(np.log1p(asize)),
                )
            total_bid = float(sum(size for _, size in bids))
            total_ask = float(sum(size for _, size in asks))
            microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
            engineered[index] = (
                np.float32((ask - bid) / TICK_SIZE), np.float32((microprice - mid) / TICK_SIZE),
                np.float32((total_bid - total_ask) / (total_bid + total_ask)),
                np.float32(_ofi(bid, bid_size, ask, ask_size, previous_bbo)),
            )
        best_bid[index], best_ask[index], valid[index] = bid, ask, True
        best_bid_ticks[index], best_ask_ticks[index] = bids[0][0], asks[0][0]
        best_bid_size[index], best_ask_size[index] = bids[0][1], asks[0][1]
        previous_bbo = (bid, bid_size, ask, ask_size)

    event_index = rows["event_index"].to_numpy(np.uint64)
    event_starts = np.flatnonzero(np.r_[True, event_index[1:] != event_index[:-1]])
    event_ends = np.r_[event_starts[1:], len(rows)]
    receive = rows["ts_recv_ns"].to_numpy(np.int64)
    segments = rows["continuity_segment_id"].to_numpy(np.uint32)
    operations = rows["operation"].astype(str).to_numpy()
    sides = rows["side"].astype(str).to_numpy()
    prices = rows["price_ticks"].to_numpy(np.int64)
    sizes = rows["level_size"].to_numpy(np.uint64)
    for start, end in zip(event_starts, event_ends, strict=True):
        group_available = int(receive[start:end].max()) + DECODE_NS
        while clock_index < count and int(clocks_ns[clock_index]) < group_available:
            sample(clock_index)
            clock_index += 1
        segment = int(segments[start])
        if state_available >= 0 and segment != state_segment:
            books = {"B": {}, "A": {}}
            previous_bbo = None
        state_segment = segment
        for row_index in range(int(start), int(end)):
            operation, side = operations[row_index], sides[row_index]
            if operation == "clear":
                # Databento's depth clear is book-wide and carries a null side.
                books = {"B": {}, "A": {}}
                previous_bbo = None
                continue
            if operation not in {"upsert", "delete"} or side not in {"A", "B"}:
                rejected["unknown"] += 1
                continue
            if operation == "delete":
                books[side].pop(int(prices[row_index]), None)
            else:
                size = int(sizes[row_index])
                if size > 0:
                    books[side][int(prices[row_index])] = size
                else:
                    books[side].pop(int(prices[row_index]), None)
        state_available = group_available
    while clock_index < count:
        sample(clock_index)
        clock_index += 1
    result = {
        "clock_ns": clocks_ns.astype(np.int64), "state_available_ns": available,
        "continuity_segment": continuity, "valid": valid, "levels": level_values,
        "engineered": engineered, "best_bid": best_bid, "best_ask": best_ask,
        "best_bid_ticks": best_bid_ticks, "best_ask_ticks": best_ask_ticks,
        "best_bid_size": best_bid_size, "best_ask_size": best_ask_size,
    }
    if not include_features:
        result.pop("levels")
        result.pop("engineered")
    return result, rejected


def reconstruct_arrival_bbo(
    rows: pd.DataFrame, observation_clocks_ns: np.ndarray, latency_ms: int
) -> dict[str, np.ndarray]:
    """Materialize the causal BBO visible when an order reaches the venue."""

    shifted = observation_clocks_ns + int(latency_ms) * 1_000_000
    state, _ = reconstruct_l2(rows, shifted, include_features=False)
    return {
        f"arrival_{latency_ms}ms_clock_ns": state["clock_ns"],
        f"arrival_{latency_ms}ms_state_available_ns": state["state_available_ns"],
        f"arrival_{latency_ms}ms_continuity_segment": state["continuity_segment"],
        f"arrival_{latency_ms}ms_valid": state["valid"],
        f"arrival_{latency_ms}ms_bid_ticks": state["best_bid_ticks"],
        f"arrival_{latency_ms}ms_ask_ticks": state["best_ask_ticks"],
        f"arrival_{latency_ms}ms_bid_size": state["best_bid_size"],
        f"arrival_{latency_ms}ms_ask_size": state["best_ask_size"],
    }


def build_trade_features(rows: pd.DataFrame, l2: dict[str, np.ndarray], day: str) -> dict[str, np.ndarray]:
    rows = rows.loc[~rows["is_bad_ts_recv"].astype(bool)].sort_values(["ts_recv_ns", "global_record_ordinal"], kind="stable")
    available_ns = rows["ts_recv_ns"].to_numpy(np.int64) + DECODE_NS
    price_ticks = rows["price_ticks"].to_numpy(np.int64)
    sizes = rows["size"].to_numpy(np.float64)
    side = rows["aggressor_side"].map({"B": 1.0, "A": -1.0, "buy": 1.0, "sell": -1.0,
                                        "BUY": 1.0, "SELL": -1.0}).fillna(0.0).to_numpy(np.float64)
    l2_index = np.searchsorted(l2["clock_ns"], available_ns, side="right") - 1
    safe_index = np.maximum(l2_index, 0)
    valid_mid = (l2_index >= 0) & l2["valid"][safe_index]
    mid_ticks = np.where(valid_mid, (l2["best_bid"][safe_index] + l2["best_ask"][safe_index]) / (2 * TICK_SIZE), price_ticks)
    delta_price = np.diff(price_ticks, prepend=price_ticks[0]).astype(np.float64)
    interarrival = np.diff(available_ns, prepend=available_ns[0]).clip(min=0).astype(np.float64) / 1_000.0
    start_ns, end_ns = _session_clocks(day)[[0, -1]]
    progress = np.clip((available_ns - start_ns) / (end_ns - start_ns), 0.0, 1.0)
    features = np.column_stack((
        delta_price, price_ticks - mid_ticks, side, np.log1p(sizes), np.log1p(interarrival), progress,
        np.zeros(len(rows), dtype=np.float64),
    )).astype(np.float32)
    decisions = _decision_clocks(day)
    return {
        "trade_available_ns": available_ns, "trade_features": features,
        "decision_clock_ns": decisions,
        "decision_trade_end": np.searchsorted(available_ns, decisions, side="right").astype(np.int64),
        "decision_l2_end": np.searchsorted(l2["clock_ns"], decisions, side="right").astype(np.int64),
    }


def _read_day(root: Path, layer: str, day: str, symbol: str, columns: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    stamp = date.fromisoformat(day)
    folder = root / layer / f"year={stamp.year:04d}" / f"month={stamp.month:02d}" / f"day={stamp.day:02d}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {layer} files for {day}")
    table = ds.dataset(folder, format="parquet").to_table(columns=columns, filter=ds.field("raw_symbol") == symbol)
    return table.to_pandas(), files


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as target:
            np.savez_compressed(target, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(archive_root: Path, active_map: Path, output: Path, start: str, end: str,
          limit_sessions: int | None = None) -> Path:
    if output.exists():
        raise FileExistsError("GC dataset output is immutable")
    require_development_access(start, purpose="fit")
    if end > "2025-09-01":
        raise PermissionError("the development builder may not inspect September-December 2025")
    mapping = pq.read_table(active_map).to_pandas()
    mapping = mapping[(mapping["session_date"] >= start) & (mapping["session_date"] < end)]
    if limit_sessions is not None:
        mapping = mapping.head(limit_sessions)
    if mapping.empty:
        raise ValueError("no active-contract sessions in requested interval")
    staging = output.with_name(output.name + ".part")
    if staging.exists():
        raise FileExistsError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    audits: list[SessionAudit] = []
    sources: dict[str, dict[str, object]] = {}
    try:
        for item in mapping.itertuples(index=False):
            day, symbol = str(item.session_date), str(item.selected_raw_symbol)
            l2_rows, l2_files = _read_day(archive_root, "l2_depth", day, symbol, [
                "event_index", "event_row_index", "ts_recv_ns", "continuity_segment_id",
                "operation", "side", "price_ticks", "level_size", "raw_symbol",
            ])
            trade_rows, trade_files = _read_day(archive_root, "l1_trades", day, symbol, [
                "ts_recv_ns", "global_record_ordinal", "price_ticks", "size", "aggressor_side",
                "is_bad_ts_recv", "raw_symbol",
            ])
            if l2_rows.empty or trade_rows.empty:
                raise ValueError(f"{day} {symbol}: empty active-contract source")
            segment_values = l2_rows["continuity_segment_id"]
            continuity_segments = int(segment_values.nunique(dropna=False))
            if segment_values.isna().any() or continuity_segments != 1:
                raise ValueError(
                    f"{day} {symbol}: registered session must contain exactly one continuity segment"
                )
            l2, rejected = reconstruct_l2(l2_rows, _session_clocks(day))
            arrival_5ms = reconstruct_arrival_bbo(l2_rows, l2["clock_ns"], 5)
            arrival_20ms = reconstruct_arrival_bbo(l2_rows, l2["clock_ns"], 20)
            trades = build_trade_features(trade_rows, l2, day)
            arrays = {
                **l2, **arrival_5ms, **arrival_20ms, **trades,
                "session_date": np.array(day), "raw_symbol": np.array(symbol),
            }
            shard = staging / "sessions" / f"{day}.npz"
            _atomic_npz(shard, arrays)
            for source in l2_files + trade_files:
                key = str(source.resolve())
                if key not in sources:
                    sources[key] = {"sha256": sha256(source), "bytes": source.stat().st_size}
            valid_count = int(l2["valid"].sum())
            decisions = trades["decision_l2_end"]
            interval_covered = []
            previous = 0
            for endpoint in decisions:
                endpoint = int(endpoint)
                interval_covered.append(bool(l2["valid"][previous:endpoint].any()))
                previous = endpoint
            five_minute_coverage = float(np.mean(interval_covered))
            audits.append(SessionAudit(
                day, symbol, len(l2_rows), len(trade_rows), len(l2["valid"]), valid_count,
                valid_count / len(l2["valid"]), five_minute_coverage,
                continuity_segments,
                rejected["crossed"], rejected["stale"], rejected["incomplete"], rejected["unknown"],
                "PASS" if five_minute_coverage >= 0.95 and rejected["unknown"] == 0 else "FAIL",
            ))
        if any(a.status != "PASS" for a in audits):
            raise ValueError("one or more GC sessions failed the frozen 95% quality threshold")
        map_hash = sha256(active_map)
        products = {str(path.relative_to(staging)): {"sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in sorted((staging / "sessions").glob("*.npz"))}
        manifest = {
            "schema_version": 2, "instrument": "GC_COMEX_FUTURES", "start": start, "end": end,
            "available_time_rule": "event_max_ts_recv_ns_plus_1ms",
            "execution_arrivals_ms": [5, 20],
            "basepaper_execution_contract_sha256": registered_contract("base_paper_gc").sha256,
            "active_contract_map": {"path": str(active_map.resolve()), "sha256": map_hash},
            "contracts": {
                name: sha256(Path(__file__).parents[2] / "contracts" / name)
                for name in (
                    "GC_FEATURE_REGISTRY.json", "SIMULATOR_ASSUMPTIONS.json",
                    "COMMUNICATION_REGISTRY.json", "ENVIRONMENT_CONTRACT_V2.json",
                )
            },
            "sources": sources, "products": products, "audits": [asdict(a) for a in audits],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        (staging / "manifest.sha256").write_text(f"{sha256(manifest_path)}  manifest.json\n")
        (staging / "data_dictionary.json").write_text(json.dumps({
            "levels": ["bid_distance_ticks", "log1p_bid_size", "ask_distance_ticks", "log1p_ask_size"],
            "engineered": ["spread_ticks", "microprice_displacement_ticks", "top10_imbalance", "top_of_book_ofi"],
            "trade_features": ["price_change_ticks", "price_displacement_from_available_mid_ticks", "signed_aggressor", "log1p_size", "log1p_interarrival_us", "session_progress", "padding_mask"],
            "window_materialization": "on_demand; never crosses session shard",
        }, sort_keys=True, indent=2) + "\n")
        os.replace(staging, output)
    except Exception:
        # Preserve staging for forensic diagnosis; a later run must choose or clear it explicitly.
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build causal 250ms GC L2 and trade-session shards")
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--active-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-07-01")
    parser.add_argument("--limit-sessions", type=int)
    args = parser.parse_args(argv)
    result = build(args.archive_root, args.active_map, args.output, args.start, args.end, args.limit_sessions)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
