"""Causal feature construction for the two-level V6 GC hierarchy.

V2 shards contain both observation-time books and simulator-only arrival
books.  This module intentionally names every observation input and never
indexes an ``arrival_*`` member.  All fitted quantities are carried by
``V6FeatureScaler`` and can only be fitted from registered training dates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .volume_v6_contract import (
    ACTIONS,
    BASE_CONTEXT_WIDTH,
    BOTTOM_BASE_FIELDS,
    BOTTOM_BASE_WIDTH,
    BOTTOM_HISTORY,
    BOTTOM_MINUTES,
    BOTTOM_RICH_FIELDS,
    BOTTOM_RICH_WIDTH,
    CONTEXT_FIELDS,
    CONTEXT_WIDTH,
    DataPhase,
    L2_ENGINEERED_FIELDS,
    L2_ENGINEERED_WIDTH,
    L2_HISTORY,
    L2_LEVEL_FIELDS,
    L2_LEVELS,
    L2_LEVEL_WIDTH,
    RegimeMessage,
    STATE_WIDTH,
    TOP_BASE_FIELDS,
    TOP_BASE_WIDTH,
    TOP_HISTORY,
    TOP_MINUTES,
    TOP_RICH_FIELDS,
    TOP_RICH_WIDTH,
    TRAIN_END_EXCLUSIVE,
    V6Arm,
    assert_phase_date,
    validate_arm,
)


NS_PER_SECOND = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_SECOND
SESSION_TERMINAL_MINUTES = 5
TOP_CLOCK_SLOTS = 20
BOTTOM_CLOCK_SLOTS = 60

# This allow-list documents all V2 arrays that can affect an observation.
# Simulator-only arrival arrays are absent by construction.
OBSERVATION_RAW_FIELDS = frozenset({
    "clock_ns",
    "state_available_ns",
    "continuity_segment",
    "valid",
    "levels",
    "engineered",
    "best_bid_ticks",
    "best_ask_ticks",
    "best_bid_size",
    "best_ask_size",
    "session_date",
    "raw_symbol",
})
OBSERVATION_SIDECAR_FIELDS = frozenset({
    "session_date",
    "session_start_ns",
    "session_end_ns",
    "trade_available_ns",
    "trade_price_ticks",
    "trade_size",
    "trade_sign",
    "mlofi",
    "source_sha256",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected scalar metadata")
    return str(array.item())


def _validate_pair(
    raw: Mapping[str, np.ndarray], side: Mapping[str, np.ndarray],
) -> tuple[str, str]:
    missing_raw = sorted(OBSERVATION_RAW_FIELDS.difference(raw))
    missing_side = sorted(OBSERVATION_SIDECAR_FIELDS.difference(side))
    if missing_raw or missing_side:
        raise KeyError(f"V6 source lacks raw={missing_raw}, sidecar={missing_side}")
    day = _scalar_text(raw["session_date"])
    if _scalar_text(side["session_date"]) != day:
        raise ValueError("raw and sidecar session dates differ")
    symbol = _scalar_text(raw["raw_symbol"])
    if not symbol:
        raise ValueError("raw symbol is empty")

    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    valid = np.asarray(raw["valid"], dtype=bool)
    segment = np.asarray(raw["continuity_segment"])
    levels = np.asarray(raw["levels"])
    if clock.ndim != 1 or not len(clock) or np.any(np.diff(clock) <= 0):
        raise ValueError("book clocks must be nonempty, unique, and ordered")
    if valid.shape != clock.shape or segment.shape != clock.shape:
        raise ValueError("book validity and continuity arrays must align")
    if levels.shape != (len(clock), L2_LEVELS, 4):
        raise ValueError("V2 levels must have shape [clock,10,4]")
    if np.asarray(raw["engineered"]).shape != (len(clock), 4):
        raise ValueError("V2 engineered book fields must have width four")
    for name in (
        "state_available_ns", "best_bid_ticks", "best_ask_ticks",
        "best_bid_size", "best_ask_size",
    ):
        if np.asarray(raw[name]).shape != clock.shape:
            raise ValueError(f"{name} must align with book clocks")

    times = np.asarray(side["trade_available_ns"], dtype=np.int64)
    prices = np.asarray(side["trade_price_ticks"], dtype=np.float64)
    sizes = np.asarray(side["trade_size"], dtype=np.float64)
    signs = np.asarray(side["trade_sign"], dtype=np.float64)
    if times.ndim != 1 or np.any(np.diff(times) < 0):
        raise ValueError("trade availability clocks must be ordered")
    if any(value.shape != times.shape for value in (prices, sizes, signs)):
        raise ValueError("trade arrays must align")
    if not (np.isfinite(prices).all() and np.isfinite(sizes).all() and np.isfinite(signs).all()):
        raise ValueError("trade arrays contain non-finite values")
    if (prices <= 0).any() or (sizes < 0).any():
        raise ValueError("trade prices and sizes are invalid")
    if not np.isin(signs, (-1.0, 0.0, 1.0)).all():
        raise ValueError("trade aggressor sign must be -1, 0, or +1")
    if np.asarray(side["mlofi"]).shape != (len(clock), L2_LEVELS):
        raise ValueError("sidecar MLOFI must align with ten-level book clocks")
    start = int(np.asarray(side["session_start_ns"]).item())
    end = int(np.asarray(side["session_end_ns"]).item())
    if start < int(clock[0]) or end > int(clock[-1]) or end <= start:
        raise ValueError("sidecar session bounds do not fit the book clock")
    return day, symbol


def load_npz_pair(
    v2_path: Path,
    sidecar_path: Path,
    *,
    phase: DataPhase = "train",
    validate_source_hash: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read one aligned pair only after enforcing its registered phase."""

    if v2_path.stem != sidecar_path.stem:
        raise ValueError("V6 raw and sidecar filenames must align")
    assert_phase_date(v2_path.stem, phase)
    with np.load(v2_path, allow_pickle=False) as handle:
        raw = {name: handle[name] for name in handle.files}
    with np.load(sidecar_path, allow_pickle=False) as handle:
        side = {name: handle[name] for name in handle.files}
    _validate_pair(raw, side)
    if validate_source_hash and _scalar_text(side["source_sha256"]) != _sha256(v2_path):
        raise ValueError("V6 sidecar source hash differs from its raw shard")
    return raw, side


@dataclass(frozen=True)
class V6Calibration:
    """Training-only nonlinear feature constants."""

    top_relative_volume_curve: tuple[float, ...]
    bottom_relative_volume_curve: tuple[float, ...]
    large_trade_threshold: float
    training_dates: tuple[str, ...]

    def __post_init__(self) -> None:
        top = np.asarray(self.top_relative_volume_curve, dtype=np.float64)
        bottom = np.asarray(self.bottom_relative_volume_curve, dtype=np.float64)
        if top.shape != (TOP_CLOCK_SLOTS,) or bottom.shape != (BOTTOM_CLOCK_SLOTS,):
            raise ValueError("relative-volume curves have the wrong clock width")
        if (top <= 0).any() or (bottom <= 0).any() or not np.isfinite(top).all() or not np.isfinite(bottom).all():
            raise ValueError("relative-volume curves must be finite and positive")
        if not np.isfinite(self.large_trade_threshold) or self.large_trade_threshold <= 0:
            raise ValueError("large-trade threshold must be finite and positive")
        if not self.training_dates or len(self.training_dates) != len(set(self.training_dates)):
            raise ValueError("calibration requires unique training dates")
        if any(day >= TRAIN_END_EXCLUSIVE for day in self.training_dates):
            raise PermissionError("calibration contains validation or test dates")

    @classmethod
    def fit(
        cls,
        pairs: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]],
    ) -> "V6Calibration":
        if not pairs:
            raise ValueError("calibration requires training sessions")
        dates: list[str] = []
        sizes: list[np.ndarray] = []
        top_slots: list[list[float]] = [[] for _ in range(TOP_CLOCK_SLOTS)]
        bottom_slots: list[list[float]] = [[] for _ in range(BOTTOM_CLOCK_SLOTS)]
        for raw, side in pairs:
            day, _ = _validate_pair(raw, side)
            assert_phase_date(day, "train")
            dates.append(day)
            one_size = np.asarray(side["trade_size"], dtype=np.float64)
            sizes.append(one_size[one_size > 0])
            for minutes, target in ((TOP_MINUTES, top_slots), (BOTTOM_MINUTES, bottom_slots)):
                bars = _aggregate_bars(raw, side, minutes)
                for slot, (volume, valid) in enumerate(zip(bars.volume, bars.valid, strict=True)):
                    if slot < len(target) and valid and volume > 0:
                        target[slot].append(float(volume))
        if len(dates) != len(set(dates)):
            raise ValueError("calibration sessions contain duplicate dates")
        positive_sizes = np.concatenate([value for value in sizes if len(value)]) if any(map(len, sizes)) else np.empty(0)
        if not len(positive_sizes):
            raise ValueError("calibration sessions contain no positive trade size")

        def curve(slots: list[list[float]]) -> tuple[float, ...]:
            observed = np.asarray([value for slot in slots for value in slot], dtype=np.float64)
            fallback = float(np.median(observed)) if len(observed) else 1.0
            return tuple(
                max(1e-8, float(np.median(slot)) if slot else fallback)
                for slot in slots
            )

        return cls(
            curve(top_slots),
            curve(bottom_slots),
            float(np.quantile(positive_sizes, 0.90)),
            tuple(sorted(dates)),
        )

    @classmethod
    def fit_paths(
        cls,
        path_pairs: Sequence[tuple[Path, Path]],
        *,
        validate_source_hash: bool = True,
    ) -> "V6Calibration":
        """Fit nonlinear constants one shard at a time.

        Trade sizes in GC are discrete.  Counting unique observed sizes keeps
        the exact NumPy-style linear 90th percentile without retaining the
        multi-month tape in memory.
        """

        pairs = tuple(path_pairs)
        if not pairs:
            raise ValueError("calibration requires training paths")
        dates: list[str] = []
        size_counts: dict[float, int] = {}
        top_slots: list[list[float]] = [[] for _ in range(TOP_CLOCK_SLOTS)]
        bottom_slots: list[list[float]] = [[] for _ in range(BOTTOM_CLOCK_SLOTS)]
        for raw_path, side_path in pairs:
            raw, side = load_npz_pair(
                raw_path, side_path, phase="train",
                validate_source_hash=validate_source_hash,
            )
            day = _scalar_text(raw["session_date"]); dates.append(day)
            positive = np.asarray(side["trade_size"], dtype=np.float64)
            positive = positive[positive > 0]
            unique, counts = np.unique(positive, return_counts=True)
            for value, count in zip(unique, counts, strict=True):
                key = float(value); size_counts[key] = size_counts.get(key, 0) + int(count)
            for minutes, target in ((TOP_MINUTES, top_slots), (BOTTOM_MINUTES, bottom_slots)):
                bars = _aggregate_bars(raw, side, minutes)
                for slot, (volume, valid) in enumerate(zip(bars.volume, bars.valid, strict=True)):
                    if slot < len(target) and valid and volume > 0:
                        target[slot].append(float(volume))
            del raw, side
        if len(dates) != len(set(dates)):
            raise ValueError("calibration paths contain duplicate dates")
        if not size_counts:
            raise ValueError("calibration paths contain no positive trade size")

        def curve(slots: list[list[float]]) -> tuple[float, ...]:
            all_values = [value for slot in slots for value in slot]
            fallback = float(np.median(all_values)) if all_values else 1.0
            return tuple(
                max(1e-8, float(np.median(slot)) if slot else fallback)
                for slot in slots
            )

        ordered = sorted(size_counts.items())
        total = sum(count for _, count in ordered)
        rank = 0.90 * (total - 1)
        lower_rank = int(np.floor(rank)); upper_rank = int(np.ceil(rank))

        def order_value(target: int) -> float:
            cumulative = 0
            for value, count in ordered:
                cumulative += count
                if target < cumulative:
                    return value
            return ordered[-1][0]

        lower = order_value(lower_rank); upper = order_value(upper_rank)
        quantile = lower + (rank - lower_rank) * (upper - lower)
        return cls(curve(top_slots), curve(bottom_slots), quantile, tuple(sorted(dates)))


@dataclass(frozen=True)
class RobustBlockScaler:
    """Per-field median/MAD scaler fitted with an explicit element mask."""

    fields: tuple[str, ...]
    median: tuple[float, ...]
    scale: tuple[float, ...]
    observed: tuple[int, ...]
    clip: float = 10.0

    @classmethod
    def fit(
        cls, fields: tuple[str, ...], values: np.ndarray, mask: np.ndarray,
    ) -> "RobustBlockScaler":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 2 or array.shape[-1] != len(fields):
            raise ValueError("robust-scaler values disagree with the feature registry")
        present = np.asarray(mask, dtype=bool)
        if present.shape == array.shape[:-1]:
            present = np.broadcast_to(present[..., None], array.shape)
        if present.shape != array.shape:
            raise ValueError("robust-scaler mask must be row- or element-aligned")
        medians: list[float] = []
        scales: list[float] = []
        counts: list[int] = []
        flat = array.reshape(-1, len(fields))
        available = present.reshape(-1, len(fields))
        for column in range(len(fields)):
            selected = flat[available[:, column], column]
            selected = selected[np.isfinite(selected)]
            counts.append(int(len(selected)))
            if not len(selected):
                medians.append(0.0); scales.append(1.0)
                continue
            median = float(np.median(selected))
            mad = float(np.median(np.abs(selected - median)) * 1.4826)
            medians.append(median)
            scales.append(mad if mad > 1e-12 else 1.0)
        return cls(fields, tuple(medians), tuple(scales), tuple(counts))

    def apply(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.fields):
            raise ValueError("scaler feature width differs")
        present = np.asarray(mask, dtype=bool)
        if present.shape == array.shape[:-1]:
            present = np.broadcast_to(present[..., None], array.shape)
        if present.shape != array.shape:
            raise ValueError("scaler mask differs")
        output = np.clip(
            (array - np.asarray(self.median)) / np.asarray(self.scale),
            -float(self.clip), float(self.clip),
        )
        output = np.where(present, output, 0.0).astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("scaled observation contains non-finite values")
        return output


class _DiskBlockAccumulator:
    """Temporary, row-streamed matrix used for exact robust fitting."""

    def __init__(self, root: Path, name: str, fields: tuple[str, ...]) -> None:
        self.path = root / f"{name}.float64"
        self.fields = fields
        self.rows = 0

    def append(self, values: np.ndarray, mask: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != len(self.fields):
            raise ValueError("disk accumulator feature width differs")
        present = np.asarray(mask, dtype=bool)
        if present.shape == array.shape[:-1]:
            present = np.broadcast_to(present[..., None], array.shape)
        if present.shape != array.shape:
            raise ValueError("disk accumulator mask differs")
        flattened = np.where(present, array, np.nan).reshape(-1, len(self.fields))
        with self.path.open("ab") as target:
            flattened.tofile(target)
        self.rows += len(flattened)

    def finish(self) -> RobustBlockScaler:
        if self.rows <= 0:
            raise ValueError("cannot fit an empty disk-backed feature block")
        values = np.memmap(
            self.path, dtype=np.float64, mode="r", shape=(self.rows, len(self.fields)),
        )
        medians: list[float] = []; scales: list[float] = []; counts: list[int] = []
        for column in range(len(self.fields)):
            selected = np.asarray(values[:, column])
            selected = selected[np.isfinite(selected)]
            counts.append(int(len(selected)))
            if not len(selected):
                medians.append(0.0); scales.append(1.0)
                continue
            median = float(np.median(selected))
            mad = float(np.median(np.abs(selected - median)) * 1.4826)
            medians.append(median); scales.append(mad if mad > 1e-12 else 1.0)
        del values
        return RobustBlockScaler(
            self.fields, tuple(medians), tuple(scales), tuple(counts),
        )


@dataclass(frozen=True)
class V6FeatureScaler:
    calibration: V6Calibration
    top_base: RobustBlockScaler
    top_rich: RobustBlockScaler
    top_context: RobustBlockScaler
    bottom_base: RobustBlockScaler
    bottom_rich: RobustBlockScaler
    bottom_context: RobustBlockScaler
    l2_levels: RobustBlockScaler
    l2_engineered: RobustBlockScaler

    @classmethod
    def fit(
        cls,
        pairs: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]],
    ) -> "V6FeatureScaler":
        calibration = V6Calibration.fit(pairs)
        sessions = [build_session_features(raw, side, calibration) for raw, side in pairs]

        def rows(attribute: str) -> np.ndarray:
            return np.concatenate([np.asarray(getattr(item, attribute)) for item in sessions])

        def masks(attribute: str) -> np.ndarray:
            return np.concatenate([np.asarray(getattr(item, attribute)) for item in sessions])

        return cls(
            calibration,
            RobustBlockScaler.fit(TOP_BASE_FIELDS, rows("top_base"), masks("top_valid")),
            RobustBlockScaler.fit(TOP_RICH_FIELDS, rows("top_rich"), masks("top_valid")),
            RobustBlockScaler.fit(CONTEXT_FIELDS, rows("top_context"), masks("top_context_mask")),
            RobustBlockScaler.fit(BOTTOM_BASE_FIELDS, rows("bottom_base"), masks("bottom_valid")),
            RobustBlockScaler.fit(BOTTOM_RICH_FIELDS, rows("bottom_rich"), masks("bottom_valid")),
            RobustBlockScaler.fit(CONTEXT_FIELDS, rows("bottom_context"), masks("bottom_context_mask")),
            RobustBlockScaler.fit(
                L2_LEVEL_FIELDS,
                rows("bottom_levels").reshape(-1, L2_LEVEL_WIDTH),
                masks("bottom_l2_mask")[:, :, None, None]
                .repeat(L2_LEVELS, axis=2)
                .repeat(L2_LEVEL_WIDTH, axis=3)
                .reshape(-1, L2_LEVEL_WIDTH),
            ),
            RobustBlockScaler.fit(
                L2_ENGINEERED_FIELDS,
                rows("bottom_l2_engineered").reshape(-1, L2_ENGINEERED_WIDTH),
                masks("bottom_l2_mask").reshape(-1),
            ),
        )

    @classmethod
    def fit_paths(
        cls,
        path_pairs: Sequence[tuple[Path, Path]],
        *,
        temporary_directory: Path | None = None,
        validate_source_hash: bool = True,
    ) -> "V6FeatureScaler":
        """Laptop-safe two-pass fit that never retains raw sessions in memory."""

        pairs = tuple(path_pairs)
        calibration = V6Calibration.fit_paths(
            pairs, validate_source_hash=validate_source_hash,
        )
        parent = None if temporary_directory is None else str(temporary_directory)
        with tempfile.TemporaryDirectory(prefix="kx-v6-scale-", dir=parent) as temporary:
            root = Path(temporary)
            accumulators = {
                "top_base": _DiskBlockAccumulator(root, "top_base", TOP_BASE_FIELDS),
                "top_rich": _DiskBlockAccumulator(root, "top_rich", TOP_RICH_FIELDS),
                "top_context": _DiskBlockAccumulator(root, "top_context", CONTEXT_FIELDS),
                "bottom_base": _DiskBlockAccumulator(root, "bottom_base", BOTTOM_BASE_FIELDS),
                "bottom_rich": _DiskBlockAccumulator(root, "bottom_rich", BOTTOM_RICH_FIELDS),
                "bottom_context": _DiskBlockAccumulator(root, "bottom_context", CONTEXT_FIELDS),
                "l2_levels": _DiskBlockAccumulator(root, "l2_levels", L2_LEVEL_FIELDS),
                "l2_engineered": _DiskBlockAccumulator(
                    root, "l2_engineered", L2_ENGINEERED_FIELDS,
                ),
            }
            for raw_path, side_path in pairs:
                raw, side = load_npz_pair(
                    raw_path, side_path, phase="train",
                    validate_source_hash=validate_source_hash,
                )
                session = build_session_features(raw, side, calibration)
                accumulators["top_base"].append(session.top_base, session.top_valid)
                accumulators["top_rich"].append(session.top_rich, session.top_valid)
                accumulators["top_context"].append(
                    session.top_context, session.top_context_mask,
                )
                accumulators["bottom_base"].append(session.bottom_base, session.bottom_valid)
                accumulators["bottom_rich"].append(session.bottom_rich, session.bottom_valid)
                accumulators["bottom_context"].append(
                    session.bottom_context, session.bottom_context_mask,
                )
                level_mask = np.broadcast_to(
                    session.bottom_l2_mask[:, :, None],
                    session.bottom_levels.shape[:-1],
                )
                accumulators["l2_levels"].append(session.bottom_levels, level_mask)
                accumulators["l2_engineered"].append(
                    session.bottom_l2_engineered, session.bottom_l2_mask,
                )
                del raw, side, session
            fitted = {name: block.finish() for name, block in accumulators.items()}
        return cls(
            calibration, fitted["top_base"], fitted["top_rich"], fitted["top_context"],
            fitted["bottom_base"], fitted["bottom_rich"], fitted["bottom_context"],
            fitted["l2_levels"], fitted["l2_engineered"],
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "V6FeatureScaler":
        calibration = V6Calibration(**value["calibration"])  # type: ignore[arg-type]
        names = (
            "top_base", "top_rich", "top_context", "bottom_base", "bottom_rich",
            "bottom_context", "l2_levels", "l2_engineered",
        )
        blocks = [RobustBlockScaler(**value[name]) for name in names]  # type: ignore[arg-type]
        return cls(calibration, *blocks)

    def canonical_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def save_json(self, path: Path) -> Path:
        """Atomically persist the complete training-only scaler contract."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(self.to_dict(), handle, sort_keys=True, indent=2)
            handle.write("\n")
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load_json(cls, path: Path) -> "V6FeatureScaler":
        return cls.from_dict(json.loads(path.read_text()))


@dataclass(frozen=True)
class _BarAggregates:
    clock_ns: np.ndarray
    segment: np.ndarray
    segment_start_ns: np.ndarray
    valid: np.ndarray
    trade_start: np.ndarray
    trade_end: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    signed_volume: np.ndarray
    known_volume: np.ndarray
    count: np.ndarray
    mean_size: np.ndarray
    large_share: np.ndarray
    vwap: np.ndarray
    realized: np.ndarray


def _scheduled_ends(start_ns: int, end_ns: int, minutes: int) -> np.ndarray:
    terminal = int(end_ns) - SESSION_TERMINAL_MINUTES * NS_PER_MINUTE
    values = np.arange(
        int(start_ns) + int(minutes) * NS_PER_MINUTE,
        terminal + 1,
        int(minutes) * NS_PER_MINUTE,
        dtype=np.int64,
    )
    maximum = TOP_CLOCK_SLOTS if minutes == TOP_MINUTES else BOTTOM_CLOCK_SLOTS
    if len(values) > maximum:
        raise ValueError("session contains more decision slots than the V6 contract")
    return values


def _trade_segments(raw: Mapping[str, np.ndarray], trade_times: np.ndarray) -> np.ndarray:
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    index = np.searchsorted(clock, trade_times, side="right") - 1
    output = np.full(len(trade_times), -1, dtype=np.int64)
    eligible = index >= 0
    output[eligible] = np.asarray(raw["continuity_segment"], dtype=np.int64)[index[eligible]]
    return output


def _one_minute_realized(times: np.ndarray, prices: np.ndarray, opening: float) -> float:
    if not len(prices):
        return 0.0
    minute = times // NS_PER_MINUTE
    ends = np.r_[np.flatnonzero(minute[1:] != minute[:-1]), len(minute) - 1]
    closes = prices[ends].astype(np.float64)
    sequence = np.r_[float(opening), closes]
    returns = np.diff(np.log(sequence))
    return float(np.sqrt(np.square(returns).sum()))


def _aggregate_bars(
    raw: Mapping[str, np.ndarray],
    side: Mapping[str, np.ndarray],
    minutes: int,
    *,
    large_trade_threshold: float = np.inf,
) -> _BarAggregates:
    _validate_pair(raw, side)
    start_ns = int(np.asarray(side["session_start_ns"]).item())
    end_ns = int(np.asarray(side["session_end_ns"]).item())
    ends = _scheduled_ends(start_ns, end_ns, minutes)
    n = len(ends)
    trade_times = np.asarray(side["trade_available_ns"], dtype=np.int64)
    prices = np.asarray(side["trade_price_ticks"], dtype=np.float64)
    sizes = np.asarray(side["trade_size"], dtype=np.float64)
    signs = np.asarray(side["trade_sign"], dtype=np.float64)
    trade_segment = _trade_segments(raw, trade_times)
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    raw_segment = np.asarray(raw["continuity_segment"], dtype=np.int64)
    raw_valid = np.asarray(raw["valid"], dtype=bool)

    segment = np.full(n, -1, dtype=np.int64)
    segment_start = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    trade_start = np.zeros(n, dtype=np.int64)
    trade_end = np.zeros(n, dtype=np.int64)
    output = [np.zeros(n, dtype=np.float64) for _ in range(12)]
    (opening, high, low, close, volume, signed, known, count, mean_size,
     large_share, vwap, realized) = output

    period = int(minutes) * NS_PER_MINUTE
    for row, decision in enumerate(ends):
        left = int(np.searchsorted(trade_times, int(decision) - period, side="left"))
        right = int(np.searchsorted(trade_times, int(decision), side="left"))
        trade_start[row], trade_end[row] = left, right
        book_index = int(np.searchsorted(clock, decision, side="right") - 1)
        if book_index < 0:
            continue
        one_segment = int(raw_segment[book_index])
        segment[row] = one_segment
        cursor = book_index
        while cursor > 0 and raw_segment[cursor - 1] == one_segment:
            cursor -= 1
        segment_start[row] = int(clock[cursor])
        if not raw_valid[book_index] or right <= left:
            continue
        if not np.all(trade_segment[left:right] == one_segment):
            continue
        one_price = prices[left:right]
        one_size = sizes[left:right]
        one_sign = signs[left:right]
        one_volume = float(one_size.sum())
        if one_volume <= 0:
            continue
        opening[row], high[row], low[row], close[row] = (
            float(one_price[0]), float(one_price.max()), float(one_price.min()), float(one_price[-1])
        )
        volume[row] = one_volume
        is_known = one_sign != 0
        known[row] = float(one_size[is_known].sum())
        signed[row] = float(np.sum(one_size[is_known] * one_sign[is_known]))
        count[row] = float(len(one_size))
        mean_size[row] = one_volume / len(one_size)
        large_volume = float(one_size[one_size >= large_trade_threshold].sum())
        large_share[row] = large_volume / one_volume
        vwap[row] = float(np.average(one_price, weights=one_size))
        realized[row] = _one_minute_realized(trade_times[left:right], one_price, opening[row])
        valid[row] = True
    return _BarAggregates(
        ends, segment, segment_start, valid, trade_start, trade_end,
        opening, high, low, close, volume, signed, known, count, mean_size,
        large_share, vwap, realized,
    )


def _contiguous_start(bars: _BarAggregates, index: int) -> int:
    if not bool(bars.valid[index]):
        return index + 1
    segment = int(bars.segment[index])
    cursor = int(index)
    while cursor > 0 and bars.valid[cursor - 1] and int(bars.segment[cursor - 1]) == segment:
        cursor -= 1
    return cursor


def _lag_log_return(bars: _BarAggregates, index: int, lag: int) -> tuple[float, bool]:
    prior = index - int(lag)
    start = _contiguous_start(bars, index)
    if prior < start or prior < 0 or not bars.valid[index]:
        return 0.0, False
    return float(np.log(bars.close[index] / bars.close[prior])), True


def _profile_summary(
    prices: np.ndarray, sizes: np.ndarray, current_price: float,
) -> tuple[np.ndarray, bool]:
    output = np.zeros(8, dtype=np.float64)
    if not len(prices) or float(sizes.sum()) <= 0:
        return output, False
    integer_price = np.rint(prices).astype(np.int64)
    minimum, maximum = int(integer_price.min()), int(integer_price.max())
    dense = np.bincount(integer_price - minimum, weights=sizes, minlength=maximum - minimum + 1)
    total = float(dense.sum())
    maximum_mass = float(dense.max(initial=0.0))
    candidates = np.flatnonzero(dense == maximum_mass) + minimum
    distance = np.abs(candidates.astype(np.float64) - current_price)
    poc = int(candidates[distance == distance.min()].min())
    poc_index = poc - minimum
    lower = upper = poc_index
    covered = float(dense[poc_index])
    while covered < 0.70 * total and (lower > 0 or upper < len(dense) - 1):
        left = float(dense[lower - 1]) if lower > 0 else -1.0
        right = float(dense[upper + 1]) if upper < len(dense) - 1 else -1.0
        if left >= right:
            lower -= 1; covered += max(left, 0.0)
        else:
            upper += 1; covered += max(right, 0.0)
    probabilities = dense[dense > 0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    if len(dense) > 1:
        entropy /= float(np.log(len(dense)))
    below = float(sizes[prices < current_price].sum())
    above = float(sizes[prices > current_price].sum())
    vwap = float(np.average(prices, weights=sizes))
    output[:] = (
        vwap, float(poc), float(minimum + lower), float(minimum + upper),
        float(upper - lower), entropy, maximum_mass / total, (below - above) / total,
    )
    return output, True


def _volume_context(
    raw: Mapping[str, np.ndarray],
    side: Mapping[str, np.ndarray],
    bars: _BarAggregates,
    index: int,
    volume_curve: np.ndarray,
    atr: float,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.zeros(12, dtype=np.float64)
    mask = np.zeros(12, dtype=bool)
    if not bars.valid[index]:
        return value, mask
    times = np.asarray(side["trade_available_ns"], dtype=np.int64)
    prices = np.asarray(side["trade_price_ticks"], dtype=np.float64)
    sizes = np.asarray(side["trade_size"], dtype=np.float64)
    signs = np.asarray(side["trade_sign"], dtype=np.float64)
    segments = _trade_segments(raw, times)
    decision = int(bars.clock_ns[index])
    segment_start = int(bars.segment_start_ns[index])
    selected = (
        (times >= segment_start) & (times < decision)
        & (segments == int(bars.segment[index]))
    )
    one_price, one_size, one_sign = prices[selected], sizes[selected], signs[selected]
    summary, profile_valid = _profile_summary(one_price, one_size, bars.close[index])
    if not profile_valid:
        return value, mask
    start_bar = _contiguous_start(bars, index)
    expected = max(float(volume_curve[start_bar:index + 1].sum()), 1e-8)
    actual = float(one_size.sum())
    known = one_sign != 0
    known_volume = float(one_size[known].sum())
    signed = float(np.sum(one_size[known] * one_sign[known]))
    vwap, poc, val, vah, width, entropy, concentration, mass = summary
    scale = max(float(atr), 1e-8)
    value[:10] = (
        actual / expected - 1.0,
        signed / known_volume if known_volume > 0 else 0.0,
        (bars.close[index] - vwap) / scale,
        (bars.close[index] - poc) / scale,
        (bars.close[index] - vah) / scale,
        (bars.close[index] - val) / scale,
        width / scale,
        entropy,
        concentration,
        mass,
    )
    mask[:10] = True
    mask[1] = known_volume > 0

    past_end = decision - 30 * NS_PER_MINUTE
    old_selected = selected & (times < past_end)
    old_summary, old_valid = _profile_summary(
        prices[old_selected], sizes[old_selected], bars.close[index],
    )
    if old_valid and past_end > segment_start:
        value[10] = (poc - old_summary[1]) / scale
        value[11] = (vwap - old_summary[0]) / scale
        mask[10:] = True
    return value, mask


def _feature_rows(
    raw: Mapping[str, np.ndarray],
    side: Mapping[str, np.ndarray],
    bars: _BarAggregates,
    volume_curve: np.ndarray,
    *,
    top: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_width = TOP_BASE_WIDTH if top else BOTTOM_BASE_WIDTH
    rich_width = TOP_RICH_WIDTH if top else BOTTOM_RICH_WIDTH
    base = np.zeros((len(bars.clock_ns), base_width), dtype=np.float64)
    rich = np.zeros((len(bars.clock_ns), rich_width), dtype=np.float64)
    context = np.zeros((len(bars.clock_ns), CONTEXT_WIDTH), dtype=np.float64)
    context_mask = np.zeros_like(context, dtype=bool)
    session_start = int(np.asarray(side["session_start_ns"]).item())
    session_end = int(np.asarray(side["session_end_ns"]).item())
    true_ranges = np.zeros(len(bars.clock_ns), dtype=np.float64)
    atrs = np.ones(len(bars.clock_ns), dtype=np.float64)
    for index in range(len(bars.clock_ns)):
        if not bars.valid[index]:
            continue
        start = _contiguous_start(bars, index)
        previous_close = bars.close[index - 1] if index > start else bars.open[index]
        reference = max(float(previous_close), 1e-8)
        high_low = float(bars.high[index] - bars.low[index])
        true_range = max(
            high_low,
            abs(float(bars.high[index] - previous_close)),
            abs(float(bars.low[index] - previous_close)),
        )
        true_ranges[index] = true_range
        atr = float(np.mean(true_ranges[max(start, index - 13):index + 1]))
        atr = max(atr, 1.0)
        atrs[index] = atr
        base[index, :TOP_BASE_WIDTH] = (
            np.log(bars.open[index] / reference),
            np.log(bars.high[index] / reference),
            np.log(bars.low[index] / reference),
            np.log(bars.close[index] / reference),
            true_range / atr,
            (bars.close[index] - bars.open[index]) / max(high_low, 1.0),
            (bars.close[index] - bars.low[index]) / max(high_low, 1.0) if high_low > 0 else 0.5,
            bars.realized[index],
        )
        known_imbalance = (
            bars.signed_volume[index] / bars.known_volume[index]
            if bars.known_volume[index] > 0 else 0.0
        )
        common_rich = (
            np.log1p(bars.volume[index]),
            bars.volume[index] / max(float(volume_curve[index]), 1e-8),
            known_imbalance,
            bars.known_volume[index] / bars.volume[index],
            np.log1p(bars.count[index]),
            bars.mean_size[index],
            bars.large_share[index],
        )
        volume_values, volume_mask = _volume_context(raw, side, bars, index, volume_curve, atr)
        close_vwap = (bars.close[index] - bars.vwap[index]) / atr
        if top:
            rich[index] = (*common_rich, close_vwap)
        else:
            expected = max(float(volume_curve[index]), 1e-8)
            rich[index] = (
                *common_rich,
                bars.signed_volume[index] / expected,
                close_vwap,
                *volume_values,
            )

        # Current segment price regime.
        segment_open = float(bars.open[start])
        segment_high = float(np.max(bars.high[start:index + 1]))
        segment_low = float(np.min(bars.low[start:index + 1]))
        progress = np.clip(
            (int(bars.clock_ns[index]) - session_start) / max(1, session_end - session_start),
            0.0, 1.0,
        )
        angle = 2.0 * np.pi * progress
        if top:
            momentum_lags = (2, 4, 8)
        else:
            momentum_lags = (6, 12, 24)
        momentums = [_lag_log_return(bars, index, lag) for lag in momentum_lags]
        short_lag, long_lag = ((2, 8) if top else (6, 24))
        short_returns = np.diff(np.log(bars.close[max(start, index - short_lag):index + 1]))
        long_returns = np.diff(np.log(bars.close[max(start, index - long_lag):index + 1]))
        vol_ratio_valid = index - start >= long_lag and len(short_returns) and len(long_returns)
        short_vol = float(np.sqrt(np.mean(np.square(short_returns)))) if len(short_returns) else 0.0
        long_vol = float(np.sqrt(np.mean(np.square(long_returns)))) if len(long_returns) else 0.0
        base_context = (
            momentums[0][0], momentums[1][0], momentums[2][0],
            (bars.close[index] - segment_open) / atr,
            (bars.close[index] - segment_high) / atr,
            (bars.close[index] - segment_low) / atr,
            short_vol / max(long_vol, 1e-12) if vol_ratio_valid else 0.0,
            np.sin(angle), np.cos(angle), 1.0 - progress,
        )
        context[index] = (*base_context, *volume_values)
        context_mask[index, :BASE_CONTEXT_WIDTH] = (
            momentums[0][1], momentums[1][1], momentums[2][1],
            True, True, True, bool(vol_ratio_valid), True, True, True,
        )
        context_mask[index, BASE_CONTEXT_WIDTH:] = volume_mask

        if not top:
            bottom_momentum = [_lag_log_return(bars, index, lag) for lag in (3, 6, 12)]
            base[index, TOP_BASE_WIDTH:] = (
                bottom_momentum[0][0], bottom_momentum[1][0], bottom_momentum[2][0],
                (bars.close[index] - segment_open) / atr,
                (bars.close[index] - segment_high) / atr,
                (bars.close[index] - segment_low) / atr,
                np.sin(angle), np.cos(angle), 1.0 - progress,
            )
    for array in (base, rich, context):
        if not np.isfinite(array).all():
            raise ValueError("V6 bar feature construction produced non-finite values")
    return base.astype(np.float32), rich.astype(np.float32), context.astype(np.float32), context_mask


def _l2_window(
    raw: Mapping[str, np.ndarray], side: Mapping[str, np.ndarray], decision_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    end = int(np.searchsorted(clock, int(decision_ns), side="right") - 1)
    levels_out = np.zeros((L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH), dtype=np.float32)
    engineered_out = np.zeros((L2_HISTORY, L2_ENGINEERED_WIDTH), dtype=np.float32)
    mask = np.zeros(L2_HISTORY, dtype=bool)
    if end < 0:
        return levels_out, engineered_out, mask
    valid = np.asarray(raw["valid"], dtype=bool)
    segment = np.asarray(raw["continuity_segment"])
    if not valid[end]:
        return levels_out, engineered_out, mask
    expected_segment = segment[end]
    selected: list[int] = []
    cursor = end
    while cursor >= 0 and len(selected) < L2_HISTORY:
        if not valid[cursor] or segment[cursor] != expected_segment:
            break
        selected.append(cursor); cursor -= 1
    selected.reverse()
    pad = L2_HISTORY - len(selected)
    raw_levels = np.asarray(raw["levels"], dtype=np.float64)
    flow = np.asarray(side["mlofi"], dtype=np.float64)
    state_available = np.asarray(raw["state_available_ns"], dtype=np.int64)
    best_bid = np.asarray(raw["best_bid_ticks"], dtype=np.float64)
    best_ask = np.asarray(raw["best_ask_ticks"], dtype=np.float64)

    for output_index, source_index in enumerate(selected, start=pad):
        one = raw_levels[source_index]
        bid_depth = np.expm1(one[:, 1]).clip(min=0.0)
        ask_depth = np.expm1(one[:, 3]).clip(min=0.0)
        one_flow = flow[source_index]
        levels_out[output_index, :, :4] = one.astype(np.float32)
        levels_out[output_index, :, 4] = np.arcsinh(one_flow).astype(np.float32)
        spread = float(best_ask[source_index] - best_bid[source_index])
        midpoint = (best_bid[source_index] + best_ask[source_index]) / 2.0
        l1_total = float(bid_depth[0] + ask_depth[0])
        microprice = (
            (best_ask[source_index] * bid_depth[0] + best_bid[source_index] * ask_depth[0])
            / l1_total if l1_total > 0 else midpoint
        )

        def imbalance(depth: int) -> float:
            bid = float(bid_depth[:depth].sum()); ask = float(ask_depth[:depth].sum())
            return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0

        def normalized_flow(depth: int) -> float:
            denominator = float(bid_depth[:depth].sum() + ask_depth[:depth].sum())
            return float(one_flow[:depth].sum()) / denominator if denominator > 0 else 0.0

        def mid_return(lag: int) -> float:
            prior = source_index - lag
            if prior < 0 or not valid[prior] or segment[prior] != expected_segment:
                return 0.0
            prior_mid = (best_bid[prior] + best_ask[prior]) / 2.0
            return midpoint - prior_mid

        engineered_out[output_index] = (
            spread,
            microprice - midpoint,
            imbalance(1), imbalance(3), imbalance(10),
            normalized_flow(1), normalized_flow(3), normalized_flow(10),
            float(bid_depth.sum()), float(ask_depth.sum()),
            float(np.polyfit(np.arange(L2_LEVELS), one[:, 1], 1)[0]),
            float(np.polyfit(np.arange(L2_LEVELS), one[:, 3], 1)[0]),
            mid_return(1), mid_return(4), mid_return(20),
            max(0.0, (clock[source_index] - state_available[source_index]) / 1_000_000.0),
        )
        mask[output_index] = True
    if not np.isfinite(levels_out).all() or not np.isfinite(engineered_out).all():
        raise ValueError("V6 L2 construction produced non-finite values")
    return levels_out, engineered_out, mask


@dataclass(frozen=True)
class SessionFeatures:
    session_date: str
    raw_symbol: str
    top_clock_ns: np.ndarray
    top_segment: np.ndarray
    top_valid: np.ndarray
    top_base: np.ndarray
    top_rich: np.ndarray
    top_context: np.ndarray
    top_context_mask: np.ndarray
    bottom_clock_ns: np.ndarray
    bottom_segment: np.ndarray
    bottom_valid: np.ndarray
    bottom_base: np.ndarray
    bottom_rich: np.ndarray
    bottom_context: np.ndarray
    bottom_context_mask: np.ndarray
    bottom_levels: np.ndarray
    bottom_l2_engineered: np.ndarray
    bottom_l2_mask: np.ndarray

    def validate(self) -> None:
        top = len(self.top_clock_ns); bottom = len(self.bottom_clock_ns)
        expected = {
            "top_segment": (top,), "top_valid": (top,),
            "top_base": (top, TOP_BASE_WIDTH), "top_rich": (top, TOP_RICH_WIDTH),
            "top_context": (top, CONTEXT_WIDTH), "top_context_mask": (top, CONTEXT_WIDTH),
            "bottom_segment": (bottom,), "bottom_valid": (bottom,),
            "bottom_base": (bottom, BOTTOM_BASE_WIDTH),
            "bottom_rich": (bottom, BOTTOM_RICH_WIDTH),
            "bottom_context": (bottom, CONTEXT_WIDTH),
            "bottom_context_mask": (bottom, CONTEXT_WIDTH),
            "bottom_levels": (bottom, L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH),
            "bottom_l2_engineered": (bottom, L2_HISTORY, L2_ENGINEERED_WIDTH),
            "bottom_l2_mask": (bottom, L2_HISTORY),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in ("top_valid", "top_context_mask", "bottom_valid", "bottom_context_mask", "bottom_l2_mask"):
            if np.asarray(getattr(self, name)).dtype != np.bool_:
                raise ValueError(f"{name} must be boolean")


def build_session_features(
    raw: Mapping[str, np.ndarray],
    side: Mapping[str, np.ndarray],
    calibration: V6Calibration,
) -> SessionFeatures:
    day, symbol = _validate_pair(raw, side)
    top_bars = _aggregate_bars(
        raw, side, TOP_MINUTES, large_trade_threshold=calibration.large_trade_threshold,
    )
    bottom_bars = _aggregate_bars(
        raw, side, BOTTOM_MINUTES, large_trade_threshold=calibration.large_trade_threshold,
    )
    top_base, top_rich, top_context, top_context_mask = _feature_rows(
        raw, side, top_bars, np.asarray(calibration.top_relative_volume_curve), top=True,
    )
    bottom_base, bottom_rich, bottom_context, bottom_context_mask = _feature_rows(
        raw, side, bottom_bars, np.asarray(calibration.bottom_relative_volume_curve), top=False,
    )
    levels = np.zeros(
        (len(bottom_bars.clock_ns), L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH), np.float32,
    )
    engineered = np.zeros(
        (len(bottom_bars.clock_ns), L2_HISTORY, L2_ENGINEERED_WIDTH), np.float32,
    )
    l2_mask = np.zeros((len(bottom_bars.clock_ns), L2_HISTORY), bool)
    for index, decision in enumerate(bottom_bars.clock_ns):
        levels[index], engineered[index], l2_mask[index] = _l2_window(raw, side, int(decision))
    result = SessionFeatures(
        day, symbol,
        top_bars.clock_ns, top_bars.segment, top_bars.valid,
        top_base, top_rich, top_context, top_context_mask,
        bottom_bars.clock_ns, bottom_bars.segment, bottom_bars.valid,
        bottom_base, bottom_rich, bottom_context, bottom_context_mask,
        levels, engineered, l2_mask,
    )
    result.validate()
    return result


def _window_indexes(
    valid: np.ndarray, segment: np.ndarray, index: int, history: int,
) -> np.ndarray:
    if index < 0 or index >= len(valid):
        raise IndexError(index)
    if not bool(valid[index]):
        raise ValueError("cannot act on an invalid decision bar")
    expected = segment[index]
    selected: list[int] = []
    cursor = int(index)
    while cursor >= 0 and len(selected) < history:
        if not bool(valid[cursor]) or segment[cursor] != expected:
            break
        selected.append(cursor); cursor -= 1
    return np.asarray(selected[::-1], dtype=np.int64)


@dataclass(frozen=True)
class RegimeObservation:
    base: np.ndarray
    rich: np.ndarray
    sequence_mask: np.ndarray
    context: np.ndarray
    context_mask: np.ndarray
    decision_time_ns: int

    def arrays(self, arm: V6Arm = "rich_volume") -> dict[str, np.ndarray]:
        validate_arm(arm)
        rich = self.rich.copy()
        context = self.context.copy()
        context_mask = self.context_mask.copy()
        if arm == "base_ohlcv":
            # The registered OHLCV control keeps only ordinary log volume.
            rich[:, 1:] = 0.0
            context[BASE_CONTEXT_WIDTH:] = 0.0
            context_mask[BASE_CONTEXT_WIDTH:] = False
        result = {
            "base": self.base.copy(), "rich": rich,
            "sequence_mask": self.sequence_mask.copy(),
            "context": context, "context_mask": context_mask,
        }
        if any(name.startswith("arrival_") for name in result):
            raise AssertionError("simulator arrival field entered a V6 observation")
        return result


@dataclass(frozen=True)
class TraderObservation:
    base: np.ndarray
    rich: np.ndarray
    sequence_mask: np.ndarray
    context: np.ndarray
    context_mask: np.ndarray
    levels: np.ndarray
    l2_engineered: np.ndarray
    l2_mask: np.ndarray
    message: np.ndarray
    state: np.ndarray
    decision_time_ns: int

    def arrays(self, arm: V6Arm = "rich_volume") -> dict[str, np.ndarray]:
        validate_arm(arm)
        rich = self.rich.copy(); context = self.context.copy()
        context_mask = self.context_mask.copy(); levels = self.levels.copy()
        engineered = self.l2_engineered.copy()
        if arm == "base_ohlcv":
            rich[:, 1:] = 0.0
            context[BASE_CONTEXT_WIDTH:] = 0.0
            context_mask[BASE_CONTEXT_WIDTH:] = False
            # Retain book price distances and executable spread, not depth/OFI.
            levels[:, :, (1, 3, 4)] = 0.0
            engineered[:, 1:] = 0.0
        result = {
            "base": self.base.copy(), "rich": rich,
            "sequence_mask": self.sequence_mask.copy(),
            "context": context, "context_mask": context_mask,
            "levels": levels, "l2_engineered": engineered,
            "l2_mask": self.l2_mask.copy(), "message": self.message.copy(),
            "state": self.state.copy(),
        }
        if any(name.startswith("arrival_") for name in result):
            raise AssertionError("simulator arrival field entered a V6 observation")
        return result


def top_observation_at(
    session: SessionFeatures, index: int, scaler: V6FeatureScaler,
) -> RegimeObservation:
    indexes = _window_indexes(session.top_valid, session.top_segment, index, TOP_HISTORY)
    base = np.zeros((TOP_HISTORY, TOP_BASE_WIDTH), np.float32)
    rich = np.zeros((TOP_HISTORY, TOP_RICH_WIDTH), np.float32)
    sequence_mask = np.zeros(TOP_HISTORY, bool)
    pad = TOP_HISTORY - len(indexes)
    base[pad:] = scaler.top_base.apply(session.top_base[indexes], np.ones(len(indexes), bool))
    rich[pad:] = scaler.top_rich.apply(session.top_rich[indexes], np.ones(len(indexes), bool))
    sequence_mask[pad:] = True
    context_mask = session.top_context_mask[index]
    context = scaler.top_context.apply(session.top_context[index:index + 1], context_mask[None])[0]
    result = RegimeObservation(
        base, rich, sequence_mask, context, context_mask.copy(),
        int(session.top_clock_ns[index]),
    )
    if base.shape != (TOP_HISTORY, TOP_BASE_WIDTH) or rich.shape != (TOP_HISTORY, TOP_RICH_WIDTH):
        raise AssertionError("top observation shape changed")
    return result


def bottom_observation_at(
    session: SessionFeatures,
    index: int,
    scaler: V6FeatureScaler,
    message: RegimeMessage,
    state: np.ndarray,
) -> TraderObservation:
    indexes = _window_indexes(
        session.bottom_valid, session.bottom_segment, index, BOTTOM_HISTORY,
    )
    base = np.zeros((BOTTOM_HISTORY, BOTTOM_BASE_WIDTH), np.float32)
    rich = np.zeros((BOTTOM_HISTORY, BOTTOM_RICH_WIDTH), np.float32)
    sequence_mask = np.zeros(BOTTOM_HISTORY, bool)
    pad = BOTTOM_HISTORY - len(indexes)
    base[pad:] = scaler.bottom_base.apply(
        session.bottom_base[indexes], np.ones(len(indexes), bool),
    )
    rich[pad:] = scaler.bottom_rich.apply(
        session.bottom_rich[indexes], np.ones(len(indexes), bool),
    )
    sequence_mask[pad:] = True
    context_mask = session.bottom_context_mask[index]
    context = scaler.bottom_context.apply(
        session.bottom_context[index:index + 1], context_mask[None],
    )[0]
    l2_mask = session.bottom_l2_mask[index]
    level_mask = np.broadcast_to(
        l2_mask[:, None], (L2_HISTORY, L2_LEVELS),
    )
    levels = scaler.l2_levels.apply(session.bottom_levels[index], level_mask)
    engineered = scaler.l2_engineered.apply(
        session.bottom_l2_engineered[index], l2_mask,
    )
    state_value = np.asarray(state, dtype=np.float32)
    if state_value.shape != (STATE_WIDTH,) or not np.isfinite(state_value).all():
        raise ValueError(f"lower-agent state must be a finite width-{STATE_WIDTH} vector")
    decision = int(session.bottom_clock_ns[index])
    result = TraderObservation(
        base, rich, sequence_mask, context, context_mask.copy(), levels, engineered,
        l2_mask.copy(), message.encode(decision), state_value.copy(), decision,
    )
    if levels.shape != (L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH):
        raise AssertionError("bottom L2 observation shape changed")
    return result


def encode_agent_state(
    *,
    inventory: int,
    current_target: int,
    liquidation_pnl_ticks: float,
    signed_entry_distance_ticks: float,
    holding_age_hours: float,
    session_remaining: float,
    visible_spread_ticks: float,
    round_trip_fee_ticks: float,
    previous_command_rejected: bool,
) -> np.ndarray:
    if inventory not in ACTIONS or current_target not in ACTIONS:
        raise ValueError("inventory and target must be SHORT, FLAT, or LONG")
    continuous = np.asarray((
        liquidation_pnl_ticks, signed_entry_distance_ticks, holding_age_hours,
        session_remaining, visible_spread_ticks, round_trip_fee_ticks,
    ), dtype=np.float64)
    if not np.isfinite(continuous).all() or holding_age_hours < 0:
        raise ValueError("agent state contains invalid continuous values")
    if not 0.0 <= session_remaining <= 1.0 or visible_spread_ticks < 0 or round_trip_fee_ticks < 0:
        raise ValueError("agent state has an invalid fraction or cost")
    output = np.zeros(STATE_WIDTH, dtype=np.float32)
    output[inventory + 1] = 1.0
    output[3 + current_target + 1] = 1.0
    output[6:] = (
        float(liquidation_pnl_ticks), float(signed_entry_distance_ticks),
        float(holding_age_hours), float(session_remaining), float(visible_spread_ticks),
        float(round_trip_fee_ticks), float(bool(previous_command_rejected)),
    )
    return output
