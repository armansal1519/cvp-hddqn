"""Frozen one-lot capacity contract for the RENT confirmatory experiment.

Capacity is evaluated at the *actual arrival* of an exposure-increasing order.
The API deliberately has no terminal/exit-book argument: future liquidity may
never license an entry.  A delayed branch must therefore call :func:`evaluate`
again with the book state available at its delayed arrival.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from numbers import Integral
from typing import Any

from .artifacts import sha256_json


CAPACITY_V2_SCHEMA_VERSION = 1
ONE_LOT_QUANTITY = 1
MIN_BEST_OPPOSITE_DEPTH = 2
MIN_OPPOSITE_TOP3_DEPTH = 20
MIN_TRAILING_AGGRESSIVE_VOLUME_5M = 200
TRAILING_WINDOW_NS = 300_000_000_000


class CapacityV2Error(ValueError):
    """Raised when an arrival state cannot support a causal capacity decision."""


class TransitionKind(StrEnum):
    SKIP = "skip"
    RISK_REDUCE = "risk_reduce"
    EXPOSURE_INCREASE = "exposure_increase"


@dataclass(frozen=True)
class CapacityV2Config:
    """Immutable thresholds for the prospective one-contract estimand."""

    schema_version: int = CAPACITY_V2_SCHEMA_VERSION
    quantity: int = ONE_LOT_QUANTITY
    min_best_opposite_depth: int = MIN_BEST_OPPOSITE_DEPTH
    min_opposite_top3_depth: int = MIN_OPPOSITE_TOP3_DEPTH
    min_trailing_aggressive_volume_5m: int = MIN_TRAILING_AGGRESSIVE_VOLUME_5M
    trailing_window_ns: int = TRAILING_WINDOW_NS

    def __post_init__(self) -> None:
        if self.schema_version != CAPACITY_V2_SCHEMA_VERSION:
            raise CapacityV2Error("unsupported capacity-v2 schema version")
        if self.quantity != ONE_LOT_QUANTITY:
            raise CapacityV2Error("capacity-v2 is frozen to exactly one contract")
        for name in (
            "min_best_opposite_depth",
            "min_opposite_top3_depth",
            "min_trailing_aggressive_volume_5m",
            "trailing_window_ns",
        ):
            if int(getattr(self, name)) <= 0:
                raise CapacityV2Error(f"{name} must be positive")

    @property
    def content_sha256(self) -> str:
        return sha256_json(asdict(self))


FROZEN_CAPACITY_V2 = CapacityV2Config()


@dataclass(frozen=True)
class ArrivalCapacityState:
    """Book and tape quantities known no later than one order arrival.

    ``trailing_window_end_ns`` is the availability-time boundary of a causal
    ``(arrival - 5 minutes, arrival]`` executed-volume window.  The caller must
    pass the state for the same symbol as the candidate order.
    """

    raw_symbol: str
    # Evaluation/window time is exactly the hypothetical order arrival.  The
    # book itself may be the last complete state before arrival and therefore
    # carries a separate immutable clock and ordinal.
    available_time_ns: int
    book_available_time_ns: int
    book_last_ordinal: int
    best_bid_depth: int
    best_ask_depth: int
    bid_top3_depth: int
    ask_top3_depth: int
    trailing_aggressive_volume_5m: int
    trailing_window_start_ns: int
    trailing_window_end_ns: int
    trailing_window_complete: bool
    source_tape_hash: str = ""

    def validate_for(self, *, raw_symbol: str, arrival_time_ns: int) -> None:
        if not self.raw_symbol or self.raw_symbol != raw_symbol:
            raise CapacityV2Error("capacity state is not for the candidate symbol")
        if len(self.source_tape_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_tape_hash
        ):
            raise CapacityV2Error("capacity state must bind a lowercase source-tape SHA-256")
        clocks = (
            self.available_time_ns,
            self.book_available_time_ns,
            self.trailing_window_start_ns,
            self.trailing_window_end_ns,
            arrival_time_ns,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in clocks):
            raise CapacityV2Error("capacity clocks must be integer nanoseconds")
        if self.trailing_window_start_ns > self.trailing_window_end_ns:
            raise CapacityV2Error("capacity window clocks are reversed")
        if (
            self.available_time_ns > arrival_time_ns
            or self.trailing_window_end_ns > arrival_time_ns
            or self.book_available_time_ns > arrival_time_ns
        ):
            raise CapacityV2Error("capacity state uses information after order arrival")
        if (
            self.trailing_window_end_ns != arrival_time_ns
            or self.available_time_ns != arrival_time_ns
        ):
            raise CapacityV2Error(
                "capacity evaluation/window must use the actual order arrival"
            )
        if (
            isinstance(self.book_last_ordinal, bool)
            or not isinstance(self.book_last_ordinal, Integral)
            or self.book_last_ordinal < 0
        ):
            raise CapacityV2Error("book_last_ordinal must be a nonnegative integer")
        if self.trailing_window_end_ns != arrival_time_ns:
            raise CapacityV2Error("trailing-volume window must end at actual order arrival")
        if self.trailing_window_end_ns - self.trailing_window_start_ns != TRAILING_WINDOW_NS:
            raise CapacityV2Error("trailing-volume window must be exactly five minutes")
        if not isinstance(self.trailing_window_complete, bool):
            raise CapacityV2Error("trailing_window_complete must be boolean")
        for name in (
            "best_bid_depth",
            "best_ask_depth",
            "bid_top3_depth",
            "ask_top3_depth",
            "trailing_aggressive_volume_5m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise CapacityV2Error(f"{name} must be a nonnegative integer")
        if self.best_bid_depth > self.bid_top3_depth:
            raise CapacityV2Error("best bid depth exceeds bid top-three depth")
        if self.best_ask_depth > self.ask_top3_depth:
            raise CapacityV2Error("best ask depth exceeds ask top-three depth")


@dataclass(frozen=True)
class CapacityDecision:
    feasible: bool
    transition_kind: TransitionKind
    reason: str
    arrival_time_ns: int
    evaluated_state_time_ns: int | None
    direction: int
    quantity: int
    opposite_side: str
    best_opposite_depth: int | None
    opposite_top3_depth: int | None
    trailing_aggressive_volume_5m: int | None
    capacity_config_sha256: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["transition_kind"] = self.transition_kind.value
        return value


def classify_transition(position_before: int, position_after: int) -> TransitionKind:
    """Classify a target transition without treating reversals as reductions."""

    for name, value in (
        ("position_before", position_before),
        ("position_after", position_after),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise CapacityV2Error(f"{name} must be an integer contract position")
        if value not in {-1, 0, 1}:
            raise CapacityV2Error("capacity-v2 positions are restricted to {-1, 0, 1}")
    if position_after == position_before:
        return TransitionKind.SKIP
    if position_before != 0 and (
        position_after == 0
        or (
            (position_before > 0) == (position_after > 0)
            and abs(position_after) < abs(position_before)
        )
    ):
        return TransitionKind.RISK_REDUCE
    return TransitionKind.EXPOSURE_INCREASE


def evaluate_capacity_v2(
    *,
    raw_symbol: str,
    arrival_time_ns: int,
    position_before: int,
    position_after: int,
    state: ArrivalCapacityState | None,
    config: CapacityV2Config = FROZEN_CAPACITY_V2,
) -> CapacityDecision:
    """Evaluate one transition using only information available at arrival.

    Skip and strict risk-reduction transitions are always feasible, including
    when no usable book state exists.  Opening, adding, and reversing require a
    fresh same-symbol arrival state.  For a delayed branch, ``arrival_time_ns``
    and ``state`` must both be the delayed values; reusing a future or terminal
    snapshot is rejected by the clock checks.
    """

    if isinstance(arrival_time_ns, bool) or not isinstance(arrival_time_ns, Integral):
        raise CapacityV2Error("arrival_time_ns must be an integer")
    if not raw_symbol:
        raise CapacityV2Error("raw_symbol cannot be empty")
    kind = classify_transition(position_before, position_after)
    if position_before * position_after == -1:
        raise CapacityV2Error(
            "direct reversal is forbidden; use a two-leg decomposition with a risk-reducing close then a separately checked open"
        )
    direction = 0 if kind is not TransitionKind.EXPOSURE_INCREASE else (1 if position_after > 0 else -1)
    opposite_side = "" if direction == 0 else ("A" if direction > 0 else "B")
    common = {
        "transition_kind": kind,
        "arrival_time_ns": arrival_time_ns,
        "direction": direction,
        "quantity": config.quantity,
        "opposite_side": opposite_side,
        "capacity_config_sha256": config.content_sha256,
    }
    if kind is TransitionKind.SKIP:
        return CapacityDecision(
            feasible=True,
            reason="skip_always_feasible",
            evaluated_state_time_ns=None,
            best_opposite_depth=None,
            opposite_top3_depth=None,
            trailing_aggressive_volume_5m=None,
            **common,
        )
    if kind is TransitionKind.RISK_REDUCE:
        return CapacityDecision(
            feasible=True,
            reason="risk_reduction_always_feasible",
            evaluated_state_time_ns=None,
            best_opposite_depth=None,
            opposite_top3_depth=None,
            trailing_aggressive_volume_5m=None,
            **common,
        )
    if state is None:
        return CapacityDecision(
            feasible=False,
            reason="missing_arrival_capacity_state",
            evaluated_state_time_ns=None,
            best_opposite_depth=None,
            opposite_top3_depth=None,
            trailing_aggressive_volume_5m=None,
            **common,
        )

    state.validate_for(raw_symbol=raw_symbol, arrival_time_ns=arrival_time_ns)
    best_depth = state.best_ask_depth if direction > 0 else state.best_bid_depth
    top3_depth = state.ask_top3_depth if direction > 0 else state.bid_top3_depth
    failures: list[str] = []
    if best_depth < config.min_best_opposite_depth:
        failures.append("best_opposite_depth_below_2")
    if top3_depth < config.min_opposite_top3_depth:
        failures.append("opposite_top3_depth_below_20")
    if state.trailing_aggressive_volume_5m < config.min_trailing_aggressive_volume_5m:
        failures.append("trailing_aggressive_volume_5m_below_200")
    if not state.trailing_window_complete:
        failures.append("trailing_aggressive_volume_5m_not_warm")
    return CapacityDecision(
        feasible=not failures,
        reason="eligible" if not failures else ";".join(failures),
        evaluated_state_time_ns=state.available_time_ns,
        best_opposite_depth=best_depth,
        opposite_top3_depth=top3_depth,
        trailing_aggressive_volume_5m=state.trailing_aggressive_volume_5m,
        **common,
    )


__all__ = [
    "CAPACITY_V2_SCHEMA_VERSION",
    "FROZEN_CAPACITY_V2",
    "MIN_BEST_OPPOSITE_DEPTH",
    "MIN_OPPOSITE_TOP3_DEPTH",
    "MIN_TRAILING_AGGRESSIVE_VOLUME_5M",
    "ONE_LOT_QUANTITY",
    "TRAILING_WINDOW_NS",
    "ArrivalCapacityState",
    "CapacityDecision",
    "CapacityV2Config",
    "CapacityV2Error",
    "TransitionKind",
    "classify_transition",
    "evaluate_capacity_v2",
]
