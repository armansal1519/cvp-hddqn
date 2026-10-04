"""Causal XAUUSD price-only pretraining for the V6 base encoders.

This module is intentionally separate from GC fine-tuning.  It accepts only a
hash-valid direct Dukascopy product, loads only pre-2025 midpoint bars, and
trains auxiliary Double-DQN heads on signed next-interval midpoint returns.
The auxiliary heads are discarded: the resulting :class:`V6ModelBundle`
contains learned weights only in ``regime.base`` and ``trader.base``.

``quote_count`` is checked and disclosed as an OTC quote-count proxy.  It is
never an input to either learner and therefore cannot initialize a V6 rich
volume, profile, order-flow, message, state, or L2 encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from .downloader import (
    sha256_file,
    validate_manifest,
    validate_manifest_digest,
)
from .volume_v6 import (
    ACTION_COUNT,
    RegimeQNetwork,
    TraderQNetwork,
    V6ModelBundle,
)
from .volume_v6_contract import (
    ACTIONS,
    BOTTOM_BASE_FIELDS,
    BOTTOM_BASE_WIDTH,
    BOTTOM_HISTORY,
    BOTTOM_RICH_WIDTH,
    CONTEXT_WIDTH,
    L2_ENGINEERED_WIDTH,
    L2_HISTORY,
    L2_LEVELS,
    L2_LEVEL_WIDTH,
    MESSAGE_WIDTH,
    STATE_WIDTH,
    TOP_BASE_FIELDS,
    TOP_BASE_WIDTH,
    TOP_HISTORY,
    TOP_RICH_WIDTH,
)
from .volume_v6_eval import atomic_json, frozen_protocol
from .volume_v6_training import (
    DEFAULT_DDQN_CONFIG,
    TRAINER_ARTIFACT,
    V6DDQNConfig,
    V6DDQNTrainer,
    file_sha256,
    load_model_bundle,
    model_forward,
    save_model_bundle,
    seed_everything,
)


CHICAGO = ZoneInfo("America/Chicago")
PRETRAIN_START = "2012-01-01"
PRETRAIN_END_EXCLUSIVE = "2025-01-01"
EXPECTED_PRETRAIN_MONTHS = tuple(
    f"{year:04d}-{month:02d}"
    for year in range(2012, 2025)
    for month in range(1, 13)
)
SESSION_START = time(7, 20)
SESSION_END = time(12, 25)
SESSION_MINUTES = 305
MINIMUM_SESSION_MINUTES = math.ceil(0.95 * SESSION_MINUTES)
XAU_TICK_SIZE = 0.10
TOP_MINUTES = 15
BOTTOM_MINUTES = 5
TOP_SLOTS = 20
BOTTOM_SLOTS = 61
MINIMUM_FULL_SESSIONS = 2_500
RETURN_REWARD_DIVISOR = 1e-3
RESUME_ARTIFACT = "V6_XAU_PRICE_PRETRAIN_RESUME"
FINAL_ARTIFACT = "V6_XAU_PRICE_PRETRAINING"


class XAUPretrainingError(ValueError):
    """Raised when the direct product or price-only training contract fails."""


class XAUPretrainingInterrupted(RuntimeError):
    """Test hook indicating that an atomic monthly checkpoint was written."""


@dataclass(frozen=True)
class DirectXAUProduct:
    root: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    pretraining_months: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class XAUPriceScaler:
    """Training-period robust scalers for the two common price encoders."""

    top_median: tuple[float, ...]
    top_scale: tuple[float, ...]
    bottom_median: tuple[float, ...]
    bottom_scale: tuple[float, ...]
    clip: float = 10.0

    def __post_init__(self) -> None:
        fields = (
            (self.top_median, TOP_BASE_WIDTH, "top median"),
            (self.top_scale, TOP_BASE_WIDTH, "top scale"),
            (self.bottom_median, BOTTOM_BASE_WIDTH, "bottom median"),
            (self.bottom_scale, BOTTOM_BASE_WIDTH, "bottom scale"),
        )
        for values, width, name in fields:
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise XAUPretrainingError(f"{name} has the wrong width or non-finite values")
            if name.endswith("scale") and bool((array <= 0).any()):
                raise XAUPretrainingError(f"{name} must be positive")
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise XAUPretrainingError("scaler clipping bound must be positive")

    @staticmethod
    def _fit_block(values: np.ndarray, valid: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
        selected = np.asarray(values, dtype=np.float64)[np.asarray(valid, dtype=bool)]
        if selected.ndim != 2 or not len(selected) or not np.isfinite(selected).all():
            raise XAUPretrainingError("cannot fit an empty or non-finite XAU price block")
        median = np.median(selected, axis=0)
        scale = 1.4826 * np.median(np.abs(selected - median), axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return tuple(map(float, median)), tuple(map(float, scale))

    @classmethod
    def fit(cls, sessions: Sequence["XAUPriceSession"]) -> "XAUPriceScaler":
        if not sessions:
            raise XAUPretrainingError("XAU scaler requires at least one session")
        top = np.concatenate([session.top_base for session in sessions])
        top_valid = np.concatenate([session.top_valid for session in sessions])
        bottom = np.concatenate([session.bottom_base for session in sessions])
        bottom_valid = np.concatenate([session.bottom_valid for session in sessions])
        top_median, top_scale = cls._fit_block(top, top_valid)
        bottom_median, bottom_scale = cls._fit_block(bottom, bottom_valid)
        return cls(top_median, top_scale, bottom_median, bottom_scale)

    def apply_top(self, values: np.ndarray) -> np.ndarray:
        return self._apply(values, self.top_median, self.top_scale, TOP_BASE_WIDTH)

    def apply_bottom(self, values: np.ndarray) -> np.ndarray:
        return self._apply(values, self.bottom_median, self.bottom_scale, BOTTOM_BASE_WIDTH)

    def _apply(
        self,
        values: np.ndarray,
        median: tuple[float, ...],
        scale: tuple[float, ...],
        width: int,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
            raise XAUPretrainingError("XAU price feature block is malformed")
        output = np.clip(
            (array - np.asarray(median)) / np.asarray(scale), -self.clip, self.clip,
        ).astype(np.float32)
        if not np.isfinite(output).all():
            raise XAUPretrainingError("scaled XAU feature block is non-finite")
        return output

    def canonical_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class XAUPriceSession:
    session_date: str
    month: str
    top_base: np.ndarray
    top_close: np.ndarray
    top_valid: np.ndarray
    bottom_base: np.ndarray
    bottom_close: np.ndarray
    bottom_valid: np.ndarray
    quote_count_proxy_total: int
    observed_minutes: int

    def __post_init__(self) -> None:
        if not PRETRAIN_START <= self.session_date < PRETRAIN_END_EXCLUSIVE:
            raise XAUPretrainingError("XAU session is outside 2012-2024 pretraining")
        if self.month != self.session_date[:7]:
            raise XAUPretrainingError("XAU month and session date differ")
        expected = (
            (self.top_base, (TOP_SLOTS, TOP_BASE_WIDTH), "top_base"),
            (self.top_close, (TOP_SLOTS,), "top_close"),
            (self.top_valid, (TOP_SLOTS,), "top_valid"),
            (self.bottom_base, (BOTTOM_SLOTS, BOTTOM_BASE_WIDTH), "bottom_base"),
            (self.bottom_close, (BOTTOM_SLOTS,), "bottom_close"),
            (self.bottom_valid, (BOTTOM_SLOTS,), "bottom_valid"),
        )
        for value, shape, name in expected:
            if np.asarray(value).shape != shape:
                raise XAUPretrainingError(f"{name} has shape {np.asarray(value).shape}, expected {shape}")
        for value in (self.top_base, self.top_close, self.bottom_base, self.bottom_close):
            if not np.isfinite(np.asarray(value)).all():
                raise XAUPretrainingError("XAU session contains non-finite price features")
        if self.quote_count_proxy_total < 1 or self.observed_minutes < MINIMUM_SESSION_MINUTES:
            raise XAUPretrainingError("XAU session does not meet source coverage")


class PriceOnlyQNetwork(nn.Module):
    """Auxiliary Q head around one V6 base encoder.

    Only ``base`` and ``sequence_mask`` are accepted.  This makes it
    structurally impossible for Dukascopy quote counts or neutral rich/L2
    tensors to update another V6 branch.
    """

    action_count = ACTION_COUNT

    def __init__(self, encoder: nn.Module, *, history: int, width: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.history = int(history)
        self.width = int(width)
        self.head = nn.Sequential(
            nn.Linear(24, 32), nn.GELU(), nn.Linear(32, ACTION_COUNT),
        )

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        if set(observation) != {"base", "sequence_mask"}:
            raise XAUPretrainingError("price-only learner received a non-base observation field")
        base = observation["base"]
        mask = observation["sequence_mask"]
        if base.ndim != 3 or tuple(base.shape[1:]) != (self.history, self.width):
            raise XAUPretrainingError("price-only base tensor has the wrong shape")
        if mask.shape != base.shape[:2] or mask.dtype != torch.bool:
            raise XAUPretrainingError("price-only sequence mask is malformed")
        return self.head(self.encoder(base, mask))


def load_direct_xau_product(root: str | Path) -> DirectXAUProduct:
    """Validate the immutable direct BI5 manifest before returning any paths."""

    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XAUPretrainingError("direct XAU manifest is absent or malformed") from error
    if not isinstance(manifest, dict):
        raise XAUPretrainingError("direct XAU manifest must be a JSON object")
    if (
        manifest.get("source_kind") != "Dukascopy OTC tick-by-tick quotes"
        or manifest.get("instrument") != "XAUUSD"
        or manifest.get("locked_period_downloaded") is not False
        or not validate_manifest_digest(directory)
        or not validate_manifest(directory, manifest)
    ):
        raise XAUPretrainingError("XAU pretraining requires a hash-valid direct BI5 product")
    semantics = manifest.get("volume_semantics")
    if not isinstance(semantics, dict) or "proxy" not in str(semantics.get("quote_count", "")).lower():
        raise XAUPretrainingError("direct XAU product lacks quote-count proxy disclosure")
    raw_months = manifest.get("months")
    if not isinstance(raw_months, list):
        raise XAUPretrainingError("direct XAU manifest months are malformed")
    months = tuple(
        month for month in raw_months
        if isinstance(month, dict) and PRETRAIN_START[:7] <= str(month.get("month")) < PRETRAIN_END_EXCLUSIVE[:7]
    )
    month_names = [str(month["month"]) for month in months]
    if tuple(month_names) != EXPECTED_PRETRAIN_MONTHS:
        raise XAUPretrainingError(
            "direct XAU product must contain every month from 2012-01 through 2024-12 exactly once"
        )
    accepted: list[int] = []
    for month in months:
        count = month.get("accepted_sessions")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise XAUPretrainingError(
                "every 2012-2024 XAU month must declare nonempty accepted session support"
            )
        accepted.append(count)
    if manifest.get("pretraining_sessions") != sum(accepted) or sum(accepted) < MINIMUM_FULL_SESSIONS:
        raise XAUPretrainingError(
            "direct XAU manifest must reconcile at least 2,500 accepted pretraining sessions"
        )
    return DirectXAUProduct(directory, manifest, sha256_file(manifest_path), months)


def _clock_minutes(local: pd.Series) -> np.ndarray:
    return (
        local.dt.hour.to_numpy(np.int64) * 60
        + local.dt.minute.to_numpy(np.int64)
        - (SESSION_START.hour * 60 + SESSION_START.minute)
    )


def _bar_arrays(
    frame: pd.DataFrame,
    minute_slot: np.ndarray,
    minutes: int,
    slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    ohlc = np.zeros((slots, 4), dtype=np.float64)
    close = np.zeros(slots, dtype=np.float64)
    valid = np.zeros(slots, dtype=bool)
    minute_closes: list[np.ndarray] = []
    for index in range(slots):
        selected = (minute_slot >= index * minutes) & (minute_slot < (index + 1) * minutes)
        part = frame.loc[selected]
        minute_closes.append(part["close"].to_numpy(np.float64))
        if len(part) != minutes:
            continue
        ohlc[index] = (
            float(part.iloc[0]["open"]), float(part["high"].max()),
            float(part["low"].min()), float(part.iloc[-1]["close"]),
        )
        close[index] = ohlc[index, 3]
        valid[index] = True
    return ohlc, close, valid, minute_closes


def _contiguous_start(valid: np.ndarray, index: int) -> int:
    cursor = int(index)
    while cursor > 0 and bool(valid[cursor - 1]):
        cursor -= 1
    return cursor


def _lag_return(close: np.ndarray, valid: np.ndarray, index: int, lag: int) -> float:
    start = _contiguous_start(valid, index)
    prior = index - int(lag)
    if prior < start or prior < 0 or not bool(valid[index]):
        return 0.0
    return float(np.log(close[index] / close[prior]))


def _price_features(
    ohlc_dollars: np.ndarray,
    close_dollars: np.ndarray,
    valid: np.ndarray,
    minute_closes_dollars: Sequence[np.ndarray],
    *,
    bottom: bool,
) -> np.ndarray:
    width = BOTTOM_BASE_WIDTH if bottom else TOP_BASE_WIDTH
    output = np.zeros((len(valid), width), dtype=np.float64)
    # Express range features in GC-compatible 0.1-dollar ticks.  Fractional
    # log returns are unchanged by this unit conversion.
    ohlc = np.asarray(ohlc_dollars, dtype=np.float64) / XAU_TICK_SIZE
    closes = np.asarray(close_dollars, dtype=np.float64) / XAU_TICK_SIZE
    true_ranges = np.zeros(len(valid), dtype=np.float64)
    for index in range(len(valid)):
        if not bool(valid[index]):
            continue
        start = _contiguous_start(valid, index)
        opening, high, low, close = ohlc[index]
        previous = closes[index - 1] if index > start else opening
        reference = max(float(previous), 1e-8)
        high_low = float(high - low)
        true_range = max(high_low, abs(float(high - previous)), abs(float(low - previous)))
        true_ranges[index] = true_range
        atr = max(float(np.mean(true_ranges[max(start, index - 13):index + 1])), 1.0)
        one_closes = np.asarray(minute_closes_dollars[index], dtype=np.float64)
        sequence = np.r_[opening * XAU_TICK_SIZE, one_closes]
        realized = float(np.sqrt(np.square(np.diff(np.log(sequence))).sum())) if len(one_closes) else 0.0
        output[index, :TOP_BASE_WIDTH] = (
            np.log(opening / reference),
            np.log(high / reference),
            np.log(low / reference),
            np.log(close / reference),
            true_range / atr,
            (close - opening) / max(high_low, 1.0),
            (close - low) / max(high_low, 1.0) if high_low > 0 else 0.5,
            realized,
        )
        if bottom:
            segment_open = float(ohlc[start, 0])
            segment_high = float(np.max(ohlc[start:index + 1, 1]))
            segment_low = float(np.min(ohlc[start:index + 1, 2]))
            progress = ((index + 1) * BOTTOM_MINUTES) / SESSION_MINUTES
            angle = 2.0 * np.pi * progress
            output[index, TOP_BASE_WIDTH:] = (
                _lag_return(closes, valid, index, 3),
                _lag_return(closes, valid, index, 6),
                _lag_return(closes, valid, index, 12),
                (close - segment_open) / atr,
                (close - segment_high) / atr,
                (close - segment_low) / atr,
                np.sin(angle), np.cos(angle), max(0.0, 1.0 - progress),
            )
    if not np.isfinite(output).all():
        raise XAUPretrainingError("XAU price feature calculation produced non-finite values")
    return output.astype(np.float32)


def build_xau_price_sessions(one_minute: pd.DataFrame) -> tuple[XAUPriceSession, ...]:
    """Build session-local completed 15-minute/5-minute midpoint features."""

    required = {"event_time", "open", "high", "low", "close", "quote_count"}
    if missing := required.difference(one_minute.columns):
        raise XAUPretrainingError(f"one-minute XAU product lacks {sorted(missing)}")
    frame = one_minute.loc[:, sorted(required)].copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    frame = frame.sort_values("event_time", kind="stable")
    if frame["event_time"].duplicated().any():
        raise XAUPretrainingError("one-minute XAU product contains duplicate timestamps")
    numeric = frame[["open", "high", "low", "close", "quote_count"]].to_numpy(np.float64)
    if not np.isfinite(numeric).all() or bool((numeric[:, :4] <= 0).any()):
        raise XAUPretrainingError("one-minute XAU midpoint bars are invalid")
    if bool((frame["quote_count"].to_numpy(np.float64) < 1).any()):
        raise XAUPretrainingError("quote_count proxy is absent or invalid")
    local = frame["event_time"].dt.tz_convert(CHICAGO)
    frame["session_date"] = local.dt.strftime("%Y-%m-%d")
    frame["minute_slot"] = _clock_minutes(local)
    frame = frame.loc[
        (local.dt.weekday < 5)
        & (frame["session_date"] >= PRETRAIN_START)
        & (frame["session_date"] < PRETRAIN_END_EXCLUSIVE)
        & (frame["minute_slot"] >= 0)
        & (frame["minute_slot"] < SESSION_MINUTES)
    ]
    sessions: list[XAUPriceSession] = []
    for day, group in frame.groupby("session_date", sort=True):
        group = group.sort_values("minute_slot", kind="stable")
        minute_slot = group["minute_slot"].to_numpy(np.int64)
        if len(group) < MINIMUM_SESSION_MINUTES or len(np.unique(minute_slot)) != len(group):
            continue
        top_ohlc, top_close, top_valid, top_minutes = _bar_arrays(
            group, minute_slot, TOP_MINUTES, TOP_SLOTS,
        )
        bottom_ohlc, bottom_close, bottom_valid, bottom_minutes = _bar_arrays(
            group, minute_slot, BOTTOM_MINUTES, BOTTOM_SLOTS,
        )
        if int(top_valid.sum()) < 2 or int(bottom_valid.sum()) < 2:
            continue
        sessions.append(XAUPriceSession(
            str(day), str(day)[:7],
            _price_features(top_ohlc, top_close, top_valid, top_minutes, bottom=False),
            top_close.astype(np.float64), top_valid,
            _price_features(bottom_ohlc, bottom_close, bottom_valid, bottom_minutes, bottom=True),
            bottom_close.astype(np.float64), bottom_valid,
            int(group["quote_count"].sum()), int(len(group)),
        ))
    return tuple(sessions)


def _history_observation(
    features: np.ndarray,
    valid: np.ndarray,
    index: int,
    history: int,
    scaler: XAUPriceScaler,
    *,
    top: bool,
) -> dict[str, np.ndarray]:
    if not 0 <= index < len(valid) or not bool(valid[index]):
        raise XAUPretrainingError("price observation requires a valid completed bar")
    start = _contiguous_start(valid, index)
    indexes = np.arange(max(start, index - history + 1), index + 1)
    raw = np.asarray(features[indexes], dtype=np.float32)
    scaled = scaler.apply_top(raw) if top else scaler.apply_bottom(raw)
    width = TOP_BASE_WIDTH if top else BOTTOM_BASE_WIDTH
    output = np.zeros((history, width), dtype=np.float32)
    mask = np.zeros(history, dtype=bool)
    output[-len(indexes):] = scaled
    mask[-len(indexes):] = True
    return {"base": output, "sequence_mask": mask}


def regime_price_observation(
    session: XAUPriceSession, index: int, scaler: XAUPriceScaler,
) -> dict[str, np.ndarray]:
    return _history_observation(
        session.top_base, session.top_valid, index, TOP_HISTORY, scaler, top=True,
    )


def trader_price_observation(
    session: XAUPriceSession, index: int, scaler: XAUPriceScaler,
) -> dict[str, np.ndarray]:
    return _history_observation(
        session.bottom_base, session.bottom_valid, index, BOTTOM_HISTORY, scaler, top=False,
    )


def neutral_regime_observation(price: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Expand a price-only observation to the exact upper V6 input contract."""

    return {
        "base": np.asarray(price["base"], dtype=np.float32),
        "rich": np.zeros((TOP_HISTORY, TOP_RICH_WIDTH), np.float32),
        "sequence_mask": np.asarray(price["sequence_mask"], dtype=bool),
        "context": np.zeros(CONTEXT_WIDTH, np.float32),
        "context_mask": np.zeros(CONTEXT_WIDTH, bool),
    }


def neutral_trader_observation(price: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Expand a price-only observation to the exact lower V6 input contract."""

    return {
        "base": np.asarray(price["base"], dtype=np.float32),
        "rich": np.zeros((BOTTOM_HISTORY, BOTTOM_RICH_WIDTH), np.float32),
        "sequence_mask": np.asarray(price["sequence_mask"], dtype=bool),
        "context": np.zeros(CONTEXT_WIDTH, np.float32),
        "context_mask": np.zeros(CONTEXT_WIDTH, bool),
        "levels": np.zeros((L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH), np.float32),
        "l2_engineered": np.zeros((L2_HISTORY, L2_ENGINEERED_WIDTH), np.float32),
        "l2_mask": np.zeros(L2_HISTORY, bool),
        "message": np.zeros(MESSAGE_WIDTH, np.float32),
        "state": np.zeros(STATE_WIDTH, np.float32),
    }


def _month_sessions(
    product: DirectXAUProduct, *, require_complete_monthly_support: bool = False,
) -> tuple[tuple[str, tuple[XAUPriceSession, ...]], ...]:
    result: list[tuple[str, tuple[XAUPriceSession, ...]]] = []
    seen: set[str] = set()
    cached: dict[Path, tuple[XAUPriceSession, ...]] = {}
    for record in product.pretraining_months:
        month = str(record["month"])
        path = (product.root / str(record["one_minute"])).resolve()
        if path not in cached:
            frame = pd.read_parquet(
                path, columns=["event_time", "open", "high", "low", "close", "quote_count"],
            )
            cached[path] = build_xau_price_sessions(frame)
        sessions = cached[path]
        if any(session.month != month for session in sessions):
            raise XAUPretrainingError("monthly XAU product contains another local month")
        if any(session.session_date in seen for session in sessions):
            raise XAUPretrainingError("XAU product repeats a pretraining session")
        seen.update(session.session_date for session in sessions)
        if require_complete_monthly_support and not sessions:
            raise XAUPretrainingError(
                "registered XAU pretraining requires nonempty accepted support in every month"
            )
        result.append((month, sessions))
    ordered = [session.session_date for _, sessions in result for session in sessions]
    if ordered != sorted(ordered):
        raise XAUPretrainingError("XAU sessions are not chronological")
    return tuple(result)


def _pairs(valid: np.ndarray) -> list[tuple[int, bool]]:
    result: list[tuple[int, bool]] = []
    for index in range(len(valid) - 1):
        if not (bool(valid[index]) and bool(valid[index + 1])):
            continue
        continuation = index + 2 < len(valid) and bool(valid[index + 2])
        result.append((index, not continuation))
    return result


def _train_session(
    trainer: V6DDQNTrainer,
    session: XAUPriceSession,
    scaler: XAUPriceScaler,
    *,
    top: bool,
) -> None:
    features = session.top_base if top else session.bottom_base
    close = session.top_close if top else session.bottom_close
    valid = session.top_valid if top else session.bottom_valid
    observation = regime_price_observation if top else trader_price_observation
    legal = np.ones(ACTION_COUNT, dtype=bool)
    for index, terminal in _pairs(valid):
        current = observation(session, index, scaler)
        following = observation(session, index + 1, scaler)
        action = trainer.act(current, legal=legal, explore=True)
        raw_return = float(close[index + 1] / close[index] - 1.0)
        trainer.append(
            current, action, action * raw_return, following,
            terminal=terminal, legal=legal,
            next_legal=np.zeros(ACTION_COUNT, dtype=bool) if terminal else legal,
        )
        trainer.update()


def _atomic_torch(path: Path, value: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _cap_sessions(
    monthly: Sequence[tuple[str, tuple[XAUPriceSession, ...]]],
    maximum_sessions: int | None,
) -> tuple[tuple[str, tuple[XAUPriceSession, ...]], ...]:
    if maximum_sessions is None:
        return tuple(monthly)
    if maximum_sessions < 1:
        raise XAUPretrainingError("maximum_sessions must be positive")
    remaining = int(maximum_sessions)
    result: list[tuple[str, tuple[XAUPriceSession, ...]]] = []
    for month, sessions in monthly:
        selected = sessions[:remaining]
        if selected:
            result.append((month, selected))
            remaining -= len(selected)
        if remaining <= 0:
            break
    return tuple(result)


def _request_identity(
    product: DirectXAUProduct,
    *,
    seed: int,
    passes: int,
    maximum_sessions: int | None,
    minimum_sessions: int,
    config: V6DDQNConfig,
    regime_replay_capacity: int,
    trader_replay_capacity: int,
    regime_replay_warmup: int,
    trader_replay_warmup: int,
    protocol_sha256: str,
) -> dict[str, object]:
    return {
        "protocol_sha256": protocol_sha256,
        "source_manifest_sha256": product.manifest_sha256,
        "seed": int(seed),
        "passes": int(passes),
        "maximum_sessions": maximum_sessions,
        "minimum_sessions": int(minimum_sessions),
        "config": asdict(config),
        "regime_replay_capacity": int(regime_replay_capacity),
        "trader_replay_capacity": int(trader_replay_capacity),
        "regime_replay_warmup": int(regime_replay_warmup),
        "trader_replay_warmup": int(trader_replay_warmup),
        "return_reward_divisor": RETURN_REWARD_DIVISOR,
    }


def _bound_output_file(
    output: Path, record: object, *, label: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise XAUPretrainingError(f"XAU final manifest lacks {label}")
    product = (output / str(record.get("path", ""))).resolve()
    try:
        product.relative_to(output.resolve())
    except ValueError as error:
        raise XAUPretrainingError(f"XAU {label} escapes its output root") from error
    if not product.is_file() or file_sha256(product) != record.get("sha256"):
        raise XAUPretrainingError(f"XAU {label} failed its immutable hash")
    return product


def _load_final(
    output: Path,
    protocol_sha256: str,
    *,
    expected_request: Mapping[str, object] | None = None,
) -> Path:
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise XAUPretrainingError("existing XAU pretraining output is incomplete") from error
    if (
        manifest.get("artifact_type") != FINAL_ARTIFACT
        or manifest.get("protocol_sha256") != protocol_sha256
        or manifest.get("stage") != "xau_price_pretraining"
        or manifest.get("gc_outcomes_accessed") is not False
        or manifest.get("gc_parameters_trained") is not False
        or not isinstance(manifest.get("request_identity"), dict)
        or not isinstance(manifest.get("training_identity"), dict)
    ):
        raise XAUPretrainingError("existing XAU pretraining output failed its hashes")
    request = manifest["request_identity"]
    if expected_request is not None and request != dict(expected_request):
        raise XAUPretrainingError("existing XAU pretraining output belongs to another request")
    if (
        manifest.get("seed") != request.get("seed")
        or manifest.get("passes") != request.get("passes")
        or manifest.get("source_manifest_sha256") != request.get("source_manifest_sha256")
        or manifest.get("smoke_session_limit") != request.get("maximum_sessions")
    ):
        raise XAUPretrainingError("XAU final metadata differs from its request identity")
    source_record = manifest.get("source_manifest")
    if not isinstance(source_record, Mapping):
        raise XAUPretrainingError("XAU final manifest lacks its source receipt")
    source_path = Path(str(source_record.get("path", "")))
    source_path = (
        source_path.resolve()
        if source_path.is_absolute() else (output / source_path).resolve()
    )
    if (
        source_path.name != "manifest.json"
        or not source_path.is_file()
        or file_sha256(source_path) != source_record.get("sha256")
        or source_path.stat().st_size != source_record.get("bytes")
    ):
        raise XAUPretrainingError("XAU source manifest bound by final output is absent or changed")
    source_product = load_direct_xau_product(source_path.parent)
    if source_product.manifest_sha256 != request.get("source_manifest_sha256"):
        raise XAUPretrainingError("XAU final output is bound to another direct source")
    if request.get("maximum_sessions") is None:
        source_monthly = _month_sessions(
            source_product, require_complete_monthly_support=True,
        )
        if sum(len(sessions) for _, sessions in source_monthly) < MINIMUM_FULL_SESSIONS:
            raise XAUPretrainingError("registered XAU source has fewer than 2,500 valid sessions")

    model = _bound_output_file(output, manifest.get("model_bundle"), label="model bundle")
    resume = _bound_output_file(
        output, manifest.get("resume_checkpoint"), label="resume checkpoint",
    )
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != RESUME_ARTIFACT
        or payload.get("identity") != manifest.get("training_identity")
        or payload.get("next_pass") != manifest.get("passes")
        or payload.get("next_month") != 0
    ):
        raise XAUPretrainingError("XAU final resume checkpoint differs from its training identity")
    for key, expected_name in (
        ("regime", "xau_regime_price"), ("trader", "xau_trader_price"),
    ):
        state = payload.get(key)
        if (
            not isinstance(state, Mapping)
            or state.get("artifact_type") != TRAINER_ARTIFACT
            or state.get("protocol_id") != protocol_sha256
            or state.get("name") != expected_name
            or state.get("seed") != request.get("seed")
            or state.get("config") != request.get("config")
            or state.get("reward_divisor") != RETURN_REWARD_DIVISOR
            or state.get("reward_unit") != "xau_fractional_midpoint_return"
            or not isinstance(state.get("metadata"), Mapping)
            or state["metadata"].get("stage") != "xau_price_pretraining"  # type: ignore[index]
        ):
            raise XAUPretrainingError("XAU trainer resume state differs from its request identity")
    _, metadata = load_model_bundle(model, protocol_id=protocol_sha256)
    if any(
        metadata.get(name) != manifest.get(name)
        for name in (
            "stage", "seed", "source_manifest_sha256", "protocol_sha256",
            "period", "session_count", "passes", "gc_outcomes_accessed",
            "gc_parameters_trained", "gc_transfer_eligible",
        )
    ):
        raise XAUPretrainingError("XAU model bundle metadata differs from its final manifest")
    return manifest_path


def pretrain_xau_price_encoders(
    direct_xau: str | Path,
    output: str | Path,
    *,
    seed: int,
    passes: int = 1,
    maximum_sessions: int | None = None,
    minimum_sessions: int = MINIMUM_FULL_SESSIONS,
    config: V6DDQNConfig = DEFAULT_DDQN_CONFIG,
    regime_replay_capacity: int = 20_000,
    trader_replay_capacity: int = 40_000,
    regime_replay_warmup: int = 256,
    trader_replay_warmup: int = 512,
    stop_after_month_checkpoints_for_test: int | None = None,
) -> Path:
    """Pretrain both common base encoders and atomically publish their bundle.

    A hidden ``.<output>.part/resume.pt`` is replaced after every month.  A
    rerun with the same source, seed, hyperparameters, and frozen protocol
    resumes at the next month.  The test-only stop hook raises immediately
    after a durable checkpoint and is never recorded as a successful run.
    """

    if passes < 1 or minimum_sessions < 1:
        raise XAUPretrainingError("passes and minimum_sessions must be positive")
    if seed < 0:
        raise XAUPretrainingError("seed must be nonnegative")
    if stop_after_month_checkpoints_for_test is not None and stop_after_month_checkpoints_for_test < 1:
        raise XAUPretrainingError("test checkpoint stop must be positive")
    _, protocol_sha256 = frozen_protocol()
    destination = Path(output).resolve()
    product = load_direct_xau_product(direct_xau)
    request_identity = _request_identity(
        product,
        seed=seed,
        passes=passes,
        maximum_sessions=maximum_sessions,
        minimum_sessions=minimum_sessions,
        config=config,
        regime_replay_capacity=regime_replay_capacity,
        trader_replay_capacity=trader_replay_capacity,
        regime_replay_warmup=regime_replay_warmup,
        trader_replay_warmup=trader_replay_warmup,
        protocol_sha256=protocol_sha256,
    )
    if destination.exists():
        return _load_final(
            destination, protocol_sha256, expected_request=request_identity,
        )
    complete_monthly = _month_sessions(
        product, require_complete_monthly_support=maximum_sessions is None,
    )
    if (
        maximum_sessions is None
        and sum(len(values) for _, values in complete_monthly) < MINIMUM_FULL_SESSIONS
    ):
        raise XAUPretrainingError("registered XAU source has fewer than 2,500 valid sessions")
    monthly = _cap_sessions(complete_monthly, maximum_sessions)
    sessions = tuple(session for _, values in monthly for session in values)
    if len(sessions) < minimum_sessions:
        raise XAUPretrainingError(
            f"XAU pretraining has {len(sessions)} valid sessions; requires {minimum_sessions}"
        )
    if not monthly:
        raise XAUPretrainingError("XAU pretraining has no usable monthly sessions")
    scaler = XAUPriceScaler.fit(sessions)
    scaler_sha256 = scaler.canonical_sha256()
    top_transitions = sum(len(_pairs(session.top_valid)) for session in sessions) * passes
    bottom_transitions = sum(len(_pairs(session.bottom_valid)) for session in sessions) * passes
    if top_transitions < regime_replay_warmup or bottom_transitions < trader_replay_warmup:
        raise XAUPretrainingError("XAU support cannot reach the configured replay warmups")
    if not (config.batch_size <= regime_replay_warmup <= regime_replay_capacity):
        raise XAUPretrainingError("regime replay settings are inconsistent")
    if not (config.batch_size <= trader_replay_warmup <= trader_replay_capacity):
        raise XAUPretrainingError("trader replay settings are inconsistent")

    identity = {
        **request_identity,
        "session_dates": [sessions[0].session_date, sessions[-1].session_date],
        "session_count": len(sessions),
        "scaler_sha256": scaler_sha256,
    }
    work = destination.with_name(f".{destination.name}.part")
    work.mkdir(parents=True, exist_ok=True)
    resume_path = work / "resume.pt"

    seed_everything(seed)
    models = V6ModelBundle.create()
    regime_aux = PriceOnlyQNetwork(models.regime.base, history=TOP_HISTORY, width=TOP_BASE_WIDTH)
    trader_aux = PriceOnlyQNetwork(models.trader.base, history=BOTTOM_HISTORY, width=BOTTOM_BASE_WIDTH)
    regime = V6DDQNTrainer(
        "xau_regime_price", regime_aux, seed=seed,
        replay_capacity=regime_replay_capacity, replay_warmup=regime_replay_warmup,
        epsilon_decay_decisions=max(1, top_transitions // 2),
        reward_divisor=RETURN_REWARD_DIVISOR, reward_unit="xau_fractional_midpoint_return",
        config=config, forward=model_forward, protocol_id=protocol_sha256,
    )
    trader = V6DDQNTrainer(
        "xau_trader_price", trader_aux, seed=seed,
        replay_capacity=trader_replay_capacity, replay_warmup=trader_replay_warmup,
        epsilon_decay_decisions=max(1, bottom_transitions // 2),
        reward_divisor=RETURN_REWARD_DIVISOR, reward_unit="xau_fractional_midpoint_return",
        config=config, forward=model_forward, protocol_id=protocol_sha256,
    )
    start_pass = start_month = 0
    if resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("artifact_type") != RESUME_ARTIFACT
            or payload.get("identity") != identity
        ):
            raise XAUPretrainingError("XAU resume checkpoint identity differs")
        regime.load_checkpoint_state(payload["regime"])
        trader.load_checkpoint_state(payload["trader"])
        start_pass = int(payload["next_pass"])
        start_month = int(payload["next_month"])

    checkpoint_count = 0
    for pass_index in range(start_pass, passes):
        first_month = start_month if pass_index == start_pass else 0
        for month_index in range(first_month, len(monthly)):
            _, month_values = monthly[month_index]
            for session in month_values:
                _train_session(regime, session, scaler, top=True)
                _train_session(trader, session, scaler, top=False)
            if month_index + 1 < len(monthly):
                next_pass, next_month = pass_index, month_index + 1
            else:
                next_pass, next_month = pass_index + 1, 0
            _atomic_torch(resume_path, {
                "schema_version": 1,
                "artifact_type": RESUME_ARTIFACT,
                "identity": identity,
                "next_pass": next_pass,
                "next_month": next_month,
                "regime": regime.checkpoint_state({"stage": "xau_price_pretraining"}),
                "trader": trader.checkpoint_state({"stage": "xau_price_pretraining"}),
            })
            checkpoint_count += 1
            if (
                stop_after_month_checkpoints_for_test is not None
                and checkpoint_count >= stop_after_month_checkpoints_for_test
                and (next_pass < passes)
            ):
                raise XAUPretrainingInterrupted("stopped after an atomic XAU monthly checkpoint")

    metadata: dict[str, object] = {
        "stage": "xau_price_pretraining",
        "seed": int(seed),
        "source_manifest_sha256": product.manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "period": [sessions[0].session_date, sessions[-1].session_date],
        "session_count": len(sessions),
        "passes": passes,
        "gc_outcomes_accessed": False,
        "gc_parameters_trained": False,
        "reward": "action_sign_times_next_completed_interval_midpoint_fractional_return",
        "learner_reward_divisor": RETURN_REWARD_DIVISOR,
        "quote_count": "validated quality/disclosure proxy only; absent from every learner observation",
        "source_volume_kind": "OTC quote count proxy; not exchange-traded volume",
        "transferrable_state": ["regime.base", "trader.base"],
        "discarded_state": ["xau_regime_auxiliary_q_head", "xau_trader_auxiliary_q_head"],
        "neutral_or_absent_during_learning": [
            "rich_volume", "volume_profile", "context", "L2", "message", "inventory_state",
        ],
        "feature_names": {
            "regime.base": list(TOP_BASE_FIELDS),
            "trader.base": list(BOTTOM_BASE_FIELDS),
        },
        "scaler": {**asdict(scaler), "sha256": scaler_sha256},
        "gc_transfer_eligible": maximum_sessions is None and len(sessions) >= MINIMUM_FULL_SESSIONS,
        "smoke_session_limit": maximum_sessions,
        "trainers": {"regime": regime.report(), "trader": trader.report()},
        "request_identity": request_identity,
        "training_identity": identity,
    }
    model_path = save_model_bundle(
        models, work / "model_bundle.pt", metadata, protocol_id=protocol_sha256,
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": FINAL_ARTIFACT,
        **metadata,
        "source_manifest": {
            "path": str((product.root / "manifest.json").resolve()),
            "sha256": product.manifest_sha256,
            "bytes": (product.root / "manifest.json").stat().st_size,
        },
        "model_bundle": {"path": model_path.name, "sha256": file_sha256(model_path)},
        "resume_checkpoint": {"path": resume_path.name, "sha256": file_sha256(resume_path)},
        "atomic_monthly_resume": True,
    }
    atomic_json(work / "manifest.json", manifest)
    os.replace(work, destination)
    return _load_final(
        destination, protocol_sha256, expected_request=request_identity,
    )


def load_xau_pretrained_bundle(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    require_gc_transfer_eligible: bool = True,
) -> tuple[V6ModelBundle, dict[str, object]]:
    """Load a hash-checked pretraining bundle for base-encoder transfer."""

    root = Path(directory).resolve()
    _, protocol_sha256 = frozen_protocol()
    manifest_path = _load_final(root, protocol_sha256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_gc_transfer_eligible and manifest.get("gc_transfer_eligible") is not True:
        raise XAUPretrainingError("smoke XAU pretraining is not eligible for GC transfer")
    models, metadata = load_model_bundle(
        root / str(manifest["model_bundle"]["path"]),
        device=device, protocol_id=protocol_sha256,
    )
    if (
        metadata.get("stage") != "xau_price_pretraining"
        or metadata.get("seed") != manifest.get("seed")
        or metadata.get("source_manifest_sha256") != manifest.get("source_manifest_sha256")
        or metadata.get("gc_outcomes_accessed") is not False
    ):
        raise XAUPretrainingError("XAU model metadata differs from its manifest")
    return models, metadata


__all__ = [
    "DirectXAUProduct", "EXPECTED_PRETRAIN_MONTHS", "MINIMUM_FULL_SESSIONS", "PriceOnlyQNetwork",
    "RETURN_REWARD_DIVISOR", "XAUPriceScaler", "XAUPriceSession",
    "XAUPretrainingError", "XAUPretrainingInterrupted", "build_xau_price_sessions",
    "load_direct_xau_product", "load_xau_pretrained_bundle",
    "neutral_regime_observation", "neutral_trader_observation",
    "pretrain_xau_price_encoders", "regime_price_observation",
    "trader_price_observation",
]
