"""Causal volume-context sidecars derived from immutable GC environment-v2 shards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


DEVELOPMENT_END_EXCLUSIVE = "2025-09-01"
TOP_DECISION_INDEX = np.arange(5, 60, 6, dtype=np.int64)
MIDDLE_DECISION_INDEX = np.arange(5, 60, dtype=np.int64)
TOP_HISTORY = 12
MIDDLE_HISTORY = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def multi_level_ofi(
    levels: np.ndarray, valid: np.ndarray, continuity: np.ndarray,
    best_bid_ticks: np.ndarray, best_ask_ticks: np.ndarray,
) -> np.ndarray:
    """Cont-style OFI at each rank, computed only from consecutive causal books."""
    if levels.ndim != 3 or levels.shape[1:] != (10, 4):
        raise ValueError("levels must have shape [time,10,4]")
    count = len(levels)
    if any(len(value) != count for value in (valid, continuity, best_bid_ticks, best_ask_ticks)):
        raise ValueError("book arrays must align")
    midpoint_twice = best_bid_ticks.astype(np.float64) + best_ask_ticks.astype(np.float64)
    bid_price_twice = midpoint_twice[:, None] - 2.0 * levels[:, :, 0]
    ask_price_twice = midpoint_twice[:, None] + 2.0 * levels[:, :, 2]
    bid_size = np.rint(np.expm1(levels[:, :, 1].astype(np.float64)))
    ask_size = np.rint(np.expm1(levels[:, :, 3].astype(np.float64)))
    output = np.zeros((count, 10), dtype=np.float32)
    causal = valid[1:] & valid[:-1] & (continuity[1:] == continuity[:-1])
    bid = (
        bid_size[1:] * (bid_price_twice[1:] >= bid_price_twice[:-1])
        - bid_size[:-1] * (bid_price_twice[1:] <= bid_price_twice[:-1])
    )
    ask = (
        -ask_size[1:] * (ask_price_twice[1:] <= ask_price_twice[:-1])
        + ask_size[:-1] * (ask_price_twice[1:] >= ask_price_twice[:-1])
    )
    output[1:] = ((bid + ask) * causal[:, None]).astype(np.float32)
    return output


def _five_minute_bars(trades: np.ndarray, endpoints: np.ndarray) -> np.ndarray:
    """Return 61 completed-interval rows without consulting later intervals."""
    sizes = np.expm1(trades[:, 3].astype(np.float64)).clip(min=0.0)
    signs = trades[:, 2].astype(np.float64)
    deltas = trades[:, 0].astype(np.float64)
    relative_price = np.cumsum(deltas)
    rows = np.zeros((len(endpoints), 11), dtype=np.float32)
    start = 0
    for slot, raw_end in enumerate(endpoints):
        end = int(raw_end)
        size = sizes[start:end]
        sign = signs[start:end]
        price = relative_price[start:end]
        delta = deltas[start:end]
        volume = float(size.sum())
        buy = float(size[sign > 0].sum())
        sell = float(size[sign < 0].sum())
        signed = buy - sell
        imbalance = signed / volume if volume > 0 else 0.0
        if len(price):
            opening = float(relative_price[start - 1]) if start else 0.0
            ending = float(price[-1])
            vwap = float(np.average(price, weights=size)) if volume > 0 else ending
            ret = ending - opening
            price_range = float(price.max() - price.min())
            realized = float(np.sqrt(np.square(delta).sum()))
            response = ret * np.sign(signed) / np.sqrt(max(volume, 1.0))
        else:
            vwap = ret = price_range = realized = response = ending = 0.0
        # Column five is raw volume at build time. Fold preparation replaces it
        # with volume / the training-fold median for this clock slot.
        rows[slot] = (volume, buy, sell, imbalance, end - start, volume,
                      ending - vwap, ret, price_range, realized, response)
        start = end
    return rows


def top_windows(bars: np.ndarray, indexes: np.ndarray = TOP_DECISION_INDEX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = np.zeros((len(indexes), TOP_HISTORY, bars.shape[1] + 1), dtype=np.float32)
    slots = np.full((len(indexes), TOP_HISTORY), -1, dtype=np.int16)
    clocks = np.asarray(indexes, dtype=np.int64)
    for output_index, decision_index in enumerate(clocks):
        available = bars[: decision_index + 1]
        selected = available[-TOP_HISTORY:]
        pad = TOP_HISTORY - len(selected)
        windows[output_index, pad:, :-1] = selected
        windows[output_index, :pad, -1] = 1.0
        slots[output_index, pad:] = np.arange(decision_index + 1 - len(selected), decision_index + 1)
    return windows, clocks, slots


def middle_trade_raw(trades: np.ndarray) -> np.ndarray:
    if trades.ndim != 2 or trades.shape[1] < 7:
        raise ValueError("v2 trade features are missing")
    size = np.expm1(trades[:, 3].astype(np.float64)).astype(np.float32)
    signed_size = trades[:, 2] * size
    return np.column_stack((trades[:, :6], signed_size, trades[:, 6])).astype(np.float32)


def materialize_middle_window(raw: np.ndarray, end: int, large_trade_threshold: float) -> np.ndarray:
    if end < 0 or end > len(raw):
        raise IndexError(end)
    selected = raw[max(0, end - MIDDLE_HISTORY):end]
    output = np.zeros((MIDDLE_HISTORY, 10), dtype=np.float32)
    pad = MIDDLE_HISTORY - len(selected)
    if pad:
        output[:pad, -1] = 1.0
    if not len(selected):
        return output
    output[pad:, :7] = selected[:, :7]
    size = np.expm1(selected[:, 3]).clip(min=0.0)
    cumulative_total = np.cumsum(size)
    output[pad:, 7] = np.divide(
        np.cumsum(selected[:, 6]), cumulative_total,
        out=np.zeros_like(cumulative_total), where=cumulative_total > 0,
    )
    output[pad:, 8] = size >= float(large_trade_threshold)
    output[pad:, 9] = selected[:, 7]
    return output


def large_trade_threshold(shards: list[Path]) -> float:
    """Fit the registered threshold from explicitly supplied training shards only."""
    sizes: list[np.ndarray] = []
    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            sizes.append(np.expm1(data["trade_features"][:, 3].astype(np.float64)))
    if not sizes:
        raise ValueError("no training shards supplied")
    return float(np.quantile(np.concatenate(sizes), 0.90))


@dataclass(frozen=True)
class DerivedAudit:
    session_date: str
    source_sha256: str
    top_decisions: int
    middle_trades: int
    bottom_clocks: int
    causal_mlofi_rows: int


def derive_shard(source: Path, output: Path) -> DerivedAudit:
    day = source.stem
    if day >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError(f"development v3 builder refuses holdout session {day}")
    with np.load(source, allow_pickle=False) as data:
        required = {
            "levels", "valid", "continuity_segment", "best_bid_ticks", "best_ask_ticks",
            "trade_features", "decision_trade_end", "decision_clock_ns", "clock_ns",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{source}: missing {sorted(missing)}")
        bars = _five_minute_bars(data["trade_features"], data["decision_trade_end"])
        top, indexes, top_slots = top_windows(bars)
        mlofi = multi_level_ofi(
            data["levels"], data["valid"], data["continuity_segment"],
            data["best_bid_ticks"], data["best_ask_ticks"],
        )
        arrays = {
            "session_date": np.array(day),
            "top_windows_raw": top,
            "top_five_minute_bars_raw": bars,
            "top_window_bar_slot": top_slots,
            "top_decision_index": indexes,
            "top_decision_clock_ns": data["decision_clock_ns"][indexes],
            "middle_trade_raw": middle_trade_raw(data["trade_features"]),
            "middle_decision_clock_ns": data["decision_clock_ns"][MIDDLE_DECISION_INDEX],
            "middle_decision_trade_end": data["decision_trade_end"][MIDDLE_DECISION_INDEX],
            "middle_decision_l2_end": data["decision_l2_end"][MIDDLE_DECISION_INDEX],
            "mlofi": mlofi,
        }
        _atomic_npz(output, arrays)
        causal = data["valid"][1:] & data["valid"][:-1]
    return DerivedAudit(day, sha256(source), len(indexes), len(arrays["middle_trade_raw"]),
                        len(mlofi), int(causal.sum()))


def build(source_roots: list[Path], output: Path) -> Path:
    if output.exists():
        raise FileExistsError("v3 volume sidecar output is immutable")
    staging = output.with_name(output.name + ".part")
    if staging.exists():
        raise FileExistsError(f"staging output exists: {staging}")
    sources: list[tuple[Path, Path]] = []
    source_manifests: list[dict[str, str]] = []
    for root in source_roots:
        manifest = root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        metadata = json.loads(manifest.read_text())
        if metadata.get("end", DEVELOPMENT_END_EXCLUSIVE) > DEVELOPMENT_END_EXCLUSIVE:
            raise PermissionError("source manifest crosses the frozen development boundary")
        source_manifests.append({"path": str(manifest.resolve()), "sha256": sha256(manifest)})
        sources.extend((path, root) for path in sorted((root / "sessions").glob("*.npz")))
    if not sources:
        raise ValueError("no v2 source shards")
    if len({path.stem for path, _ in sources}) != len(sources):
        raise ValueError("duplicate session dates across source roots")
    staging.mkdir(parents=True)
    audits: list[DerivedAudit] = []
    for source, _ in sorted(sources):
        audits.append(derive_shard(source, staging / "sessions" / source.name))
    products = {
        str(path.relative_to(staging)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted((staging / "sessions").glob("*.npz"))
    }
    manifest_data = {
        "schema_version": 3,
        "artifact_type": "GC_VOLUME_CONTEXT_CAUSAL_SIDECARS",
        "development_only": True,
        "development_end_exclusive": DEVELOPMENT_END_EXCLUSIVE,
        "source_manifests": source_manifests,
        "source_shards": {audit.session_date: audit.source_sha256 for audit in audits},
        "products": products,
        "audits": [audit.__dict__ for audit in audits],
        "fold_fitted_values_materialized": False,
        "large_trade_threshold": "fit per training fold only",
        "relative_volume_baseline": "fit per training fold and clock slot only",
    }
    manifest = staging / "manifest.json"
    manifest.write_text(json.dumps(manifest_data, sort_keys=True, indent=2) + "\n")
    (staging / "manifest.sha256").write_text(f"{sha256(manifest)}  manifest.json\n")
    os.replace(staging, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(build(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
