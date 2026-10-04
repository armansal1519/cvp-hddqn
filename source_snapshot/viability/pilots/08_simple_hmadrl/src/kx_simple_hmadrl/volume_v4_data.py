"""Causal volume-event observations for VE-HMADRL v4.

The immutable GC environment-v2 shards contain a complete UTC trade day and a
07:20--12:25 Chicago L2 session.  V3 accidentally started its first aggregate
at the beginning of the UTC trade file.  V4 first slices trades to the L2
session and only then constructs time bars, volume bars, tape windows, and
executed-volume-at-price profiles.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, Mapping

import numpy as np

from .volume_v3_data import multi_level_ofi


DEVELOPMENT_END_EXCLUSIVE = "2025-09-01"
TOP_HISTORY = 12
MIDDLE_BAR_HISTORY = 32
TRADE_HISTORY = 256
L2_HISTORY = 40
PROFILE_BINS = 65
PROFILE_CHANNELS = 3
BAR_FEATURES = 18
BAR_INPUT_FEATURES = BAR_FEATURES + 1  # explicit padding mask

ClockKind = Literal["volume_event", "time"]
BarLevel = Literal["top", "middle"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as target:
            np.savez_compressed(target, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _absolute_trade_prices(raw: Mapping[str, np.ndarray]) -> np.ndarray:
    """Recover integer trade ticks from the registered causal mid displacement.

    The v2 shard stores price changes and trade-to-mid displacement rather than
    the absolute trade price.  For valid 250 ms books, midpoint + displacement
    identifies the absolute trade tick.  A single robust offset then fills any
    trades whose preceding sampled book was invalid without consulting a later
    observation.
    """

    trade_clock = np.asarray(raw["trade_available_ns"], dtype=np.int64)
    features = np.asarray(raw["trade_features"], dtype=np.float64)
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    indexes = np.searchsorted(clock, trade_clock, side="right") - 1
    safe = np.maximum(indexes, 0)
    valid = (indexes >= 0) & np.asarray(raw["valid"], dtype=bool)[safe]
    midpoint = (
        np.asarray(raw["best_bid_ticks"], dtype=np.float64)[safe]
        + np.asarray(raw["best_ask_ticks"], dtype=np.float64)[safe]
    ) / 2.0
    inferred = np.rint(midpoint + features[:, 1])
    relative = np.cumsum(features[:, 0], dtype=np.float64)
    if not valid.any():
        raise ValueError("session has no valid causal book with which to anchor trades")
    offsets = inferred[valid] - relative[valid]
    offset = float(np.median(offsets))
    prices = np.rint(relative + offset).astype(np.int64)
    # Every causally anchored value must agree to within rounding tolerance.
    if np.max(np.abs(prices[valid].astype(np.float64) - inferred[valid])) > 1.0:
        raise ValueError("trade price reconstruction is inconsistent with causal mid displacement")
    return prices


@dataclass(frozen=True)
class DerivedAudit:
    session_date: str
    source_sha256: str
    session_trade_start: int
    session_trade_end: int
    session_trades: int
    session_volume: float
    causal_mlofi_rows: int


def derive_shard(source: Path, output: Path) -> DerivedAudit:
    """Create one immutable v4 sidecar without reading a locked session."""

    day = source.stem
    if day >= DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError(f"v4 development builder refuses locked session {day}")
    with np.load(source, allow_pickle=False) as file:
        raw = {name: file[name] for name in file.files}
    required = {
        "clock_ns", "continuity_segment", "valid", "levels", "engineered",
        "best_bid_ticks", "best_ask_ticks", "trade_available_ns", "trade_features",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"environment-v2 shard is missing {sorted(missing)}")
    if str(raw.get("session_date", day)) != day:
        raise ValueError("source filename and embedded session date differ")
    start_clock = int(raw["clock_ns"][0])
    end_clock = int(raw["clock_ns"][-1])
    trade_clock = np.asarray(raw["trade_available_ns"], dtype=np.int64)
    start = int(np.searchsorted(trade_clock, start_clock, side="left"))
    end = int(np.searchsorted(trade_clock, end_clock, side="right"))
    if end <= start:
        raise ValueError("session contains no causally available trades")
    absolute = _absolute_trade_prices(raw)[start:end]
    features = np.asarray(raw["trade_features"][start:end], dtype=np.float32).copy()
    sizes = np.expm1(features[:, 3].astype(np.float64)).clip(min=0.0)
    signs = features[:, 2].astype(np.float32)
    mlofi = multi_level_ofi(
        np.asarray(raw["levels"]), np.asarray(raw["valid"]),
        np.asarray(raw["continuity_segment"]), np.asarray(raw["best_bid_ticks"]),
        np.asarray(raw["best_ask_ticks"]),
    )
    arrays = {
        "session_date": np.asarray(day),
        "source_sha256": np.asarray(sha256_file(source)),
        "session_start_ns": np.asarray(start_clock, dtype=np.int64),
        "session_end_ns": np.asarray(end_clock, dtype=np.int64),
        "source_trade_start": np.asarray(start, dtype=np.int64),
        "source_trade_end": np.asarray(end, dtype=np.int64),
        "trade_available_ns": trade_clock[start:end],
        "trade_price_ticks": absolute,
        "trade_size": sizes.astype(np.float32),
        "trade_sign": signs,
        "trade_features": features,
        "mlofi": mlofi.astype(np.float32),
        "session_volume": np.asarray(float(sizes.sum()), dtype=np.float64),
    }
    _atomic_npz(output, arrays)
    return DerivedAudit(
        day, str(arrays["source_sha256"]), start, end, end - start,
        float(sizes.sum()), len(mlofi),
    )


def build_dataset(sources: list[Path], output: Path) -> Path:
    if output.exists():
        raise FileExistsError("v4 sidecar product is immutable")
    output.mkdir(parents=True)
    audits: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in sorted(sources, key=lambda item: item.stem):
        if source.stem in seen:
            raise ValueError(f"duplicate source session {source.stem}")
        seen.add(source.stem)
        destination = output / "sessions" / source.name
        audit = derive_shard(source, destination)
        audits.append(asdict(audit) | {
            "sidecar": str(destination.relative_to(output)),
            "sidecar_sha256": sha256_file(destination),
        })
    if not audits:
        raise ValueError("no source shards supplied")
    manifest = {
        "schema_version": 4,
        "artifact_type": "VE_HMADRL_CAUSAL_VOLUME_SIDECARS",
        "development_end_exclusive": DEVELOPMENT_END_EXCLUSIVE,
        "session_count": len(audits),
        "first_session": audits[0]["session_date"],
        "last_session": audits[-1]["session_date"],
        "trade_slice": "session_start_ns <= trade_available_ns <= session_end_ns",
        "sessions": audits,
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


@dataclass(frozen=True)
class VolumeClockSpec:
    reference_session_volume: float
    top_quantum: int
    middle_quantum: int
    top_timeout_seconds: int = 3_600
    middle_timeout_seconds: int = 600

    def __post_init__(self) -> None:
        if self.reference_session_volume <= 0:
            raise ValueError("reference session volume must be positive")
        if self.top_quantum < 1 or self.middle_quantum < 1:
            raise ValueError("volume quanta must be positive")

    @classmethod
    def fit(cls, sidecars: list[Path]) -> "VolumeClockSpec":
        if not sidecars:
            raise ValueError("volume clock fitting requires training sidecars")
        volumes: list[float] = []
        for path in sidecars:
            if path.stem >= DEVELOPMENT_END_EXCLUSIVE:
                raise PermissionError("volume clock cannot fit on locked data")
            with np.load(path, allow_pickle=False) as data:
                volumes.append(float(data["session_volume"]))
        reference = float(np.median(np.asarray(volumes, dtype=np.float64)))
        return cls(reference, max(1, int(round(reference / 10.0))),
                   max(1, int(round(reference / 60.0))))


def _volume_intervals(
    times: np.ndarray, sizes: np.ndarray, session_start: int, session_end: int,
    quantum: int, timeout_seconds: int,
) -> list[tuple[int, int, int, int, bool]]:
    timeout_ns = int(timeout_seconds) * 1_000_000_000
    intervals: list[tuple[int, int, int, int, bool]] = []
    cursor = 0
    bar_start = int(session_start)
    while bar_start < session_end:
        deadline = min(session_end, bar_start + timeout_ns)
        trade_start = cursor
        volume = 0.0
        crossed = False
        close = deadline
        while cursor < len(times) and int(times[cursor]) <= deadline:
            volume += float(sizes[cursor])
            cursor += 1
            if volume >= quantum:
                close = max(bar_start + 1, int(times[cursor - 1]))
                crossed = True
                break
        intervals.append((bar_start, close, trade_start, cursor, not crossed))
        bar_start = close
    return intervals


def _time_intervals(
    times: np.ndarray, session_start: int, session_end: int, period_seconds: int,
) -> list[tuple[int, int, int, int, bool]]:
    period_ns = int(period_seconds) * 1_000_000_000
    intervals: list[tuple[int, int, int, int, bool]] = []
    cursor = 0
    start = int(session_start)
    while start < session_end:
        end = min(session_end, start + period_ns)
        trade_start = cursor
        cursor = int(np.searchsorted(times, end, side="right"))
        # A scheduled fixed-time close is not a volume timeout. Only a final
        # interval shortened by the session boundary exposes the flag.
        intervals.append((start, end, trade_start, cursor, end - start < period_ns))
        start = end
    return intervals


def _book_features(
    raw: Mapping[str, np.ndarray], mlofi: np.ndarray, start_ns: int, end_ns: int,
) -> tuple[float, float, float, float, float, float]:
    clocks = np.asarray(raw["clock_ns"], dtype=np.int64)
    left = int(np.searchsorted(clocks, start_ns, side="right"))
    right = int(np.searchsorted(clocks, end_ns, side="right"))
    if right <= left:
        right = min(len(clocks), left + 1)
    valid = np.asarray(raw["valid"], dtype=bool)[left:right]
    if not valid.any():
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    levels = np.asarray(raw["levels"], dtype=np.float64)[left:right][valid]
    flow = np.asarray(mlofi, dtype=np.float64)[left:right][valid]
    bid_size = np.expm1(levels[:, :, 1]).clip(min=0.0)
    ask_size = np.expm1(levels[:, :, 3]).clip(min=0.0)
    depth = bid_size + ask_size

    def normalized_flow(first: int, last: int) -> float:
        numerator = float(flow[:, first:last].sum())
        denominator = float(depth[:, first:last].sum())
        return numerator / denominator if denominator > 0 else 0.0

    engineered = np.asarray(raw["engineered"], dtype=np.float64)[left:right][valid]
    return (
        normalized_flow(0, 1), normalized_flow(1, 3), normalized_flow(3, 10),
        float(engineered[:, 2].mean()), float(depth.sum(axis=1).mean()),
        float(engineered[:, 0].mean()),
    )


def build_bars(
    raw: Mapping[str, np.ndarray], side: Mapping[str, np.ndarray],
    clock_spec: VolumeClockSpec, level: BarLevel, clock_kind: ClockKind,
) -> dict[str, np.ndarray]:
    if level not in ("top", "middle") or clock_kind not in ("volume_event", "time"):
        raise ValueError("unknown bar level or clock kind")
    times = np.asarray(side["trade_available_ns"], dtype=np.int64)
    prices = np.asarray(side["trade_price_ticks"], dtype=np.int64)
    sizes = np.asarray(side["trade_size"], dtype=np.float64)
    signs = np.asarray(side["trade_sign"], dtype=np.float64)
    session_start = int(side["session_start_ns"])
    session_end = int(side["session_end_ns"])
    quantum = clock_spec.top_quantum if level == "top" else clock_spec.middle_quantum
    if clock_kind == "volume_event":
        timeout = (clock_spec.top_timeout_seconds if level == "top"
                   else clock_spec.middle_timeout_seconds)
        intervals = _volume_intervals(times, sizes, session_start, session_end, quantum, timeout)
    else:
        intervals = _time_intervals(times, session_start, session_end,
                                    1_800 if level == "top" else 300)
    features = np.zeros((len(intervals), BAR_FEATURES), dtype=np.float32)
    starts = np.zeros(len(intervals), dtype=np.int64)
    ends = np.zeros(len(intervals), dtype=np.int64)
    trade_starts = np.zeros(len(intervals), dtype=np.int64)
    trade_ends = np.zeros(len(intervals), dtype=np.int64)
    for row, (start_ns, end_ns, trade_start, trade_end, timed_out) in enumerate(intervals):
        starts[row], ends[row] = start_ns, end_ns
        trade_starts[row], trade_ends[row] = trade_start, trade_end
        one_price = prices[trade_start:trade_end].astype(np.float64)
        one_size = sizes[trade_start:trade_end]
        one_sign = signs[trade_start:trade_end]
        volume = float(one_size.sum())
        signed = float((one_size * one_sign).sum())
        count = trade_end - trade_start
        if count:
            ending = float(one_price[-1])
            vwap = float(np.average(one_price, weights=one_size)) if volume > 0 else ending
            return_ticks = ending - float(one_price[0])
            range_ticks = float(one_price.max() - one_price.min())
            deltas = np.diff(one_price, prepend=one_price[0])
            realized = float(np.sqrt(np.square(deltas).sum()))
            vwap_displacement = ending - vwap
            response = return_ticks * np.sign(signed) / np.sqrt(max(volume, 1.0))
            mean_size = volume / count
        else:
            mean_size = vwap_displacement = return_ticks = range_ticks = realized = response = 0.0
        book = _book_features(raw, np.asarray(side["mlofi"]), start_ns, end_ns)
        features[row] = (
            (end_ns - start_ns) / 1_000_000_000.0,
            volume / quantum,
            float(timed_out),
            np.log1p(count),
            mean_size,
            signed / volume if volume > 0 else 0.0,
            vwap_displacement,
            return_ticks,
            range_ticks,
            realized,
            response,
            max(0.0, volume - quantum) / quantum,
            *book,
        )
    return {
        "features": features,
        "start_ns": starts,
        "end_ns": ends,
        "trade_start": trade_starts,
        "trade_end": trade_ends,
    }


def bar_windows(features: np.ndarray, history: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != BAR_FEATURES:
        raise ValueError(f"bar features must have width {BAR_FEATURES}")
    windows = np.zeros((len(features), history, BAR_INPUT_FEATURES), dtype=np.float32)
    windows[:, :, -1] = 1.0
    for index in range(len(features)):
        selected = features[max(0, index - history + 1):index + 1]
        windows[index, -len(selected):, :BAR_FEATURES] = selected
        windows[index, -len(selected):, -1] = 0.0
    return windows


def materialize_trade_window(
    side: Mapping[str, np.ndarray], end: int, large_trade_threshold: float,
) -> np.ndarray:
    features = np.asarray(side["trade_features"], dtype=np.float32)
    sizes = np.asarray(side["trade_size"], dtype=np.float32)
    signs = np.asarray(side["trade_sign"], dtype=np.float32)
    if end < 0 or end > len(features):
        raise IndexError(end)
    start = max(0, end - TRADE_HISTORY)
    output = np.zeros((TRADE_HISTORY, 10), dtype=np.float32)
    output[:, 9] = 1.0
    selected = features[start:end]
    if not len(selected):
        return output
    pad = TRADE_HISTORY - len(selected)
    selected_size = sizes[start:end]
    selected_sign = signs[start:end]
    output[pad:, :6] = selected[:, :6]
    output[pad:, 6] = selected_size * selected_sign
    cumulative = np.cumsum(selected_size)
    output[pad:, 7] = np.divide(
        np.cumsum(selected_size * selected_sign), cumulative,
        out=np.zeros_like(cumulative), where=cumulative > 0,
    )
    output[pad:, 8] = selected_size >= float(large_trade_threshold)
    output[pad:, 9] = 0.0
    return output


def _point_of_control(volume: np.ndarray) -> int:
    maximum = float(volume.max(initial=0.0))
    candidates = np.flatnonzero(volume == maximum)
    if not len(candidates):
        return PROFILE_BINS // 2
    center = PROFILE_BINS // 2
    distance = np.abs(candidates - center)
    return int(candidates[distance == distance.min()].min())


def _value_area(volume: np.ndarray, poc: int) -> tuple[int, int]:
    total = float(volume.sum())
    if total <= 0:
        return poc, poc
    lower = upper = int(poc)
    covered = float(volume[poc])
    while covered < 0.70 * total and (lower > 0 or upper < len(volume) - 1):
        left = float(volume[lower - 1]) if lower > 0 else -1.0
        right = float(volume[upper + 1]) if upper < len(volume) - 1 else -1.0
        if left >= right:  # deterministic lower-price tie break
            lower -= 1
            covered += max(left, 0.0)
        else:
            upper += 1
            covered += max(right, 0.0)
    return lower, upper


def volume_profile(
    prices: np.ndarray, sizes: np.ndarray, signs: np.ndarray, *, center_ticks: float,
    bin_width_ticks: int, expected_volume: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    profile = np.zeros((PROFILE_BINS, PROFILE_CHANNELS), dtype=np.float32)
    metrics = np.zeros(5, dtype=np.float32)
    if not len(prices) or float(np.sum(sizes)) <= 0:
        return profile, metrics, False
    raw_offset = np.floor((np.asarray(prices, dtype=np.float64) - center_ticks)
                          / float(bin_width_ticks) + 0.5).astype(np.int64)
    half = PROFILE_BINS // 2
    outside = (raw_offset < -half) | (raw_offset > half)
    indexes = np.clip(raw_offset + half, 0, PROFILE_BINS - 1)
    total_by_bin = np.bincount(indexes, weights=sizes, minlength=PROFILE_BINS).astype(np.float64)
    signed_by_bin = np.bincount(indexes, weights=sizes * signs, minlength=PROFILE_BINS).astype(np.float64)
    count_by_bin = np.bincount(indexes, minlength=PROFILE_BINS).astype(np.float64)
    total = float(total_by_bin.sum())
    profile[:, 0] = (total_by_bin / total).astype(np.float32)
    profile[:, 1] = (signed_by_bin / total).astype(np.float32)
    profile[:, 2] = (count_by_bin / max(1.0, count_by_bin.sum())).astype(np.float32)
    poc = _point_of_control(total_by_bin)
    value_low, value_high = _value_area(total_by_bin, poc)
    metrics[:] = (
        (poc - half) * bin_width_ticks,
        (value_low - half) * bin_width_ticks,
        (value_high - half) * bin_width_ticks,
        float(np.sum(np.asarray(sizes)[outside])) / total,
        total / max(float(expected_volume), 1.0),
    )
    return profile, metrics, True


def middle_profiles(
    raw: Mapping[str, np.ndarray], side: Mapping[str, np.ndarray], middle_bars: Mapping[str, np.ndarray],
    index: int, clock_spec: VolumeClockSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if index < 0 or index >= len(middle_bars["end_ns"]):
        raise IndexError(index)
    decision_ns = int(middle_bars["end_ns"][index])
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    book_index = int(np.searchsorted(clock, decision_ns, side="right") - 1)
    if book_index < 0 or not bool(raw["valid"][book_index]):
        return (np.zeros((2, PROFILE_BINS, PROFILE_CHANNELS), np.float32),
                np.zeros((2, 5), np.float32), np.zeros(2, bool))
    center = (float(raw["best_bid_ticks"][book_index]) + float(raw["best_ask_ticks"][book_index])) / 2.0
    local_bar = max(0, index - 5)
    local_start = int(middle_bars["start_ns"][local_bar])
    session_start = int(side["session_start_ns"])
    trade_clock = np.asarray(side["trade_available_ns"], dtype=np.int64)
    end = int(np.searchsorted(trade_clock, decision_ns, side="right"))
    starts = (
        int(np.searchsorted(trade_clock, local_start, side="left")),
        int(np.searchsorted(trade_clock, session_start, side="left")),
    )
    elapsed_fraction = max(
        (decision_ns - session_start) / max(1.0, int(side["session_end_ns"]) - session_start),
        1.0 / 61.0,
    )
    expected = (6.0 * clock_spec.middle_quantum,
                elapsed_fraction * clock_spec.reference_session_volume)
    widths = (1, 4)
    profiles = np.zeros((2, PROFILE_BINS, PROFILE_CHANNELS), dtype=np.float32)
    metrics = np.zeros((2, 5), dtype=np.float32)
    masks = np.zeros(2, dtype=bool)
    for horizon, (start, width, expected_volume) in enumerate(zip(starts, widths, expected, strict=True)):
        profile, one_metrics, valid = volume_profile(
            np.asarray(side["trade_price_ticks"])[start:end],
            np.asarray(side["trade_size"])[start:end],
            np.asarray(side["trade_sign"])[start:end],
            center_ticks=center, bin_width_ticks=width, expected_volume=expected_volume,
        )
        profiles[horizon], metrics[horizon], masks[horizon] = profile, one_metrics, valid
    return profiles, metrics, masks


def rolling_l2(
    raw: Mapping[str, np.ndarray], mlofi: np.ndarray, end_index: int,
    length: int = L2_HISTORY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an explicit rolling sequence confined to one continuity segment."""

    if end_index < 0 or end_index >= len(raw["clock_ns"]):
        raise IndexError(end_index)
    valid = np.asarray(raw["valid"], dtype=bool)
    segment = np.asarray(raw["continuity_segment"])
    selected: list[int] = []
    cursor = int(end_index)
    expected_segment = segment[cursor]
    while cursor >= 0 and len(selected) < length:
        if segment[cursor] != expected_segment or not valid[cursor]:
            break
        selected.append(cursor)
        cursor -= 1
    selected.reverse()
    levels = np.zeros((length, 10, 5), dtype=np.float32)
    engineered = np.zeros((length, 4), dtype=np.float32)
    mask = np.zeros(length, dtype=bool)
    if selected:
        pad = length - len(selected)
        level_values = np.asarray(raw["levels"], dtype=np.float32)[selected]
        levels[pad:, :, :4] = level_values
        levels[pad:, :, 4] = np.arcsinh(np.asarray(mlofi, dtype=np.float32)[selected])
        engineered[pad:] = np.asarray(raw["engineered"], dtype=np.float32)[selected]
        mask[pad:] = True
    return levels, engineered, mask


def _source_paths(roots: list[Path], start: str, end: str) -> list[Path]:
    paths = sorted(
        path for root in roots for path in (root / "sessions").glob("*.npz")
        if start <= path.stem < end
    )
    if not paths or len(paths) != len({path.stem for path in paths}):
        raise ValueError("source roots contain no sessions or duplicate dates")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=DEVELOPMENT_END_EXCLUSIVE)
    args = parser.parse_args(argv)
    if args.end > DEVELOPMENT_END_EXCLUSIVE:
        raise PermissionError("v4 development sidecars are physically truncated before September")
    manifest = build_dataset(_source_paths(args.source, args.start, args.end), args.output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
