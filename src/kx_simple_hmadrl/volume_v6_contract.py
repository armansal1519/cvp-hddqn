"""Decision-complete public contract for the two-level V6 experiment.

The tuples in this module are the feature registry.  Their order is part of
the experiment contract: training, checkpoints, and evaluation must all use
these exact positions.  V6 deliberately communicates semantic Q advantages
rather than seed-specific hidden coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


V6Arm = Literal["rich_volume", "base_ohlcv"]
DataPhase = Literal["train", "validation", "test"]

ACTIONS = (-1, 0, 1)
ACTION_NAMES = ("SHORT", "FLAT", "LONG")
SEEDS = (1103, 2207, 3301, 4409, 5501)

TRAIN_START = "2025-01-01"
TRAIN_END_EXCLUSIVE = "2026-05-01"
VALIDATION_START = "2026-05-01"
VALIDATION_END_EXCLUSIVE = "2026-06-01"
TEST_START = "2026-06-01"
TEST_END_EXCLUSIVE = "2026-09-01"

TOP_MINUTES = 15
TOP_HISTORY = 8
BOTTOM_MINUTES = 5
BOTTOM_HISTORY = 24
L2_HISTORY = 20
L2_LEVELS = 10
MESSAGE_WIDTH = 5


TOP_BASE_FIELDS = (
    "open_log_return",
    "high_log_return",
    "low_log_return",
    "close_log_return",
    "true_range_atr",
    "signed_body_over_range",
    "close_location",
    "realized_volatility_1m",
)

TOP_RICH_FIELDS = (
    "log1p_executed_volume",
    "clock_relative_volume",
    "known_signed_volume_imbalance",
    "known_aggressor_fraction",
    "log1p_trade_count",
    "mean_trade_size",
    "large_trade_volume_share",
    "close_minus_trade_vwap_atr",
)

BASE_CONTEXT_FIELDS = (
    "momentum_30m",
    "momentum_60m",
    "momentum_120m",
    "distance_session_open_atr",
    "distance_session_high_atr",
    "distance_session_low_atr",
    "realized_volatility_ratio_30m_120m",
    "time_of_day_sin",
    "time_of_day_cos",
    "session_remaining",
)

VOLUME_CONTEXT_FIELDS = (
    "cumulative_relative_volume_surprise",
    "cumulative_known_signed_volume_imbalance",
    "distance_session_vwap_atr",
    "distance_poc_atr",
    "distance_vah_atr",
    "distance_val_atr",
    "value_area_width_atr",
    "profile_entropy",
    "profile_concentration",
    "profile_mass_below_minus_above",
    "poc_drift_30m_atr",
    "vwap_drift_30m_atr",
)

CONTEXT_FIELDS = BASE_CONTEXT_FIELDS + VOLUME_CONTEXT_FIELDS

BOTTOM_BASE_FIELDS = TOP_BASE_FIELDS + (
    "momentum_15m",
    "momentum_30m",
    "momentum_60m",
    "distance_session_open_atr",
    "distance_session_high_atr",
    "distance_session_low_atr",
    "time_of_day_sin",
    "time_of_day_cos",
    "session_remaining",
)

BOTTOM_RICH_FIELDS = (
    "log1p_executed_volume",
    "clock_relative_volume",
    "known_signed_volume_imbalance",
    "known_aggressor_fraction",
    "log1p_trade_count",
    "mean_trade_size",
    "large_trade_volume_share",
    "cvd_change_over_expected_volume",
    "close_minus_trade_vwap_atr",
) + VOLUME_CONTEXT_FIELDS

L2_LEVEL_FIELDS = (
    "bid_distance_from_mid_ticks",
    "log1p_bid_size",
    "ask_distance_from_mid_ticks",
    "log1p_ask_size",
    "asinh_mlofi",
)

L2_ENGINEERED_FIELDS = (
    "spread_ticks",
    "microprice_displacement_ticks",
    "depth_imbalance_l1",
    "depth_imbalance_l3",
    "depth_imbalance_l10",
    "normalized_mlofi_l1",
    "normalized_mlofi_l3",
    "normalized_mlofi_l10",
    "total_bid_depth",
    "total_ask_depth",
    "bid_log_depth_slope",
    "ask_log_depth_slope",
    "mid_return_250ms_ticks",
    "mid_return_1s_ticks",
    "mid_return_5s_ticks",
    "book_age_ms",
)

STATE_FIELDS = (
    "inventory_short",
    "inventory_flat",
    "inventory_long",
    "target_short",
    "target_flat",
    "target_long",
    "liquidation_pnl_ticks",
    "signed_entry_distance_ticks",
    "holding_age_hours",
    "session_remaining",
    "visible_spread_ticks",
    "round_trip_fee_ticks",
    "previous_command_rejected",
)

TOP_BASE_WIDTH = len(TOP_BASE_FIELDS)
TOP_RICH_WIDTH = len(TOP_RICH_FIELDS)
CONTEXT_WIDTH = len(CONTEXT_FIELDS)
BASE_CONTEXT_WIDTH = len(BASE_CONTEXT_FIELDS)
BOTTOM_BASE_WIDTH = len(BOTTOM_BASE_FIELDS)
BOTTOM_RICH_WIDTH = len(BOTTOM_RICH_FIELDS)
L2_LEVEL_WIDTH = len(L2_LEVEL_FIELDS)
L2_ENGINEERED_WIDTH = len(L2_ENGINEERED_FIELDS)
STATE_WIDTH = len(STATE_FIELDS)


def validate_arm(arm: str) -> V6Arm:
    if arm not in {"rich_volume", "base_ohlcv"}:
        raise ValueError(f"unknown V6 arm {arm!r}")
    return arm  # type: ignore[return-value]


def assert_phase_date(day: str, phase: DataPhase = "train") -> None:
    """Fail closed at the physical data-access boundary for each phase."""

    if phase == "train":
        allowed = TRAIN_START <= day < TRAIN_END_EXCLUSIVE
    elif phase == "validation":
        allowed = VALIDATION_START <= day < VALIDATION_END_EXCLUSIVE
    elif phase == "test":
        allowed = TEST_START <= day < TEST_END_EXCLUSIVE
    else:
        raise ValueError(f"unknown data phase {phase!r}")
    if not allowed:
        raise PermissionError(f"{day} is outside the registered V6 {phase} interval")


@dataclass(frozen=True)
class RegimeMessage:
    """Five-number, seed-stable top-to-bottom communication message."""

    normalized_advantages: tuple[float, float, float]
    decision_time_ns: int
    expiry_time_ns: int

    def __post_init__(self) -> None:
        advantages = np.asarray(self.normalized_advantages, dtype=np.float64)
        if advantages.shape != (3,) or not np.isfinite(advantages).all():
            raise ValueError("regime advantages must be three finite values")
        if np.max(np.abs(advantages), initial=0.0) > 1.0 + 1e-6:
            raise ValueError("regime advantages must be normalized to [-1,1]")
        if not np.isclose(float(advantages.mean()), 0.0, atol=1e-5):
            raise ValueError("regime advantages must be centered")
        if self.decision_time_ns < 0 or self.expiry_time_ns <= self.decision_time_ns:
            raise ValueError("message expiry must follow its nonnegative decision time")

    @classmethod
    def from_q_values(
        cls, q_values: np.ndarray | tuple[float, float, float],
        decision_time_ns: int, expiry_time_ns: int,
    ) -> "RegimeMessage":
        q = np.asarray(q_values, dtype=np.float64)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError("Q values must be a finite three-vector")
        centered = q - float(q.mean())
        denominator = max(float(np.max(np.abs(centered))), 1e-6)
        advantages = tuple(float(value) for value in centered / denominator)
        return cls(advantages, int(decision_time_ns), int(expiry_time_ns))

    def encode(self, now_ns: int) -> np.ndarray:
        if now_ns < self.decision_time_ns:
            raise ValueError("cannot consume a regime message before publication")
        if now_ns >= self.expiry_time_ns:
            return np.zeros(MESSAGE_WIDTH, dtype=np.float32)
        duration = max(1, self.expiry_time_ns - self.decision_time_ns)
        remaining = (self.expiry_time_ns - int(now_ns)) / duration
        return np.asarray(
            (*self.normalized_advantages, 1.0, np.clip(remaining, 0.0, 1.0)),
            dtype=np.float32,
        )


def neutral_regime_message(now_ns: int, duration_ns: int = 1) -> RegimeMessage:
    return RegimeMessage((0.0, 0.0, 0.0), int(now_ns), int(now_ns) + max(1, int(duration_ns)))


if not (
    TOP_BASE_WIDTH == 8
    and TOP_RICH_WIDTH == 8
    and CONTEXT_WIDTH == 22
    and BASE_CONTEXT_WIDTH == 10
    and BOTTOM_BASE_WIDTH == 17
    and BOTTOM_RICH_WIDTH == 21
    and L2_LEVEL_WIDTH == 5
    and L2_ENGINEERED_WIDTH == 16
    and STATE_WIDTH == 13
):
    raise AssertionError("the registered V6 feature widths changed")
