"""Causal RENT feature accumulation and immutable feature products.

This module has two deliberately separate responsibilities:

* :class:`RentFeatureAccumulator` computes book/flow features incrementally on
  the available-time clock and fails closed across resets or unusable clocks.
* the product helpers validate, hash, atomically write, and reload feature rows
  without trusting caller-supplied hashes.

The implementation never fits a scaler or threshold.  Those transformations
belong inside chronological training folds and must bind this product hash.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file, sha256_json, write_json
from .book import FIFOBook
from .replay import GroupResult
from .capacity_v2 import ArrivalCapacityState, TRAILING_WINDOW_NS
from .feature_registry import (
    FEATURE_METADATA_COLUMNS,
    L2_MASK_COLUMNS,
    L3_MASK_COLUMNS,
    RENT_FEATURE_REGISTRY,
    WINDOWS_NS,
    FeatureRegistry,
    registry_from_dict,
)


FEATURE_PRODUCT_SCHEMA_VERSION = 1
FEATURE_PRODUCT_TYPE = "kx_rent_mbo_feature_product"
FEATURE_FILE_NAME = "features.parquet"
MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_CHECKSUM_FILE_NAME = "manifest.sha256"
HASH_COLUMNS = ("l2_feature_hash", "l3_feature_hash")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_EVIDENCE_CLASSES = frozenset(
    {"DEVELOPMENT_ONLY", "LOCKED_EMPIRICAL", "SYNTHETIC_ONLY"}
)


class FeatureProductError(ValueError):
    """Raised when feature causality, schema, or artifact integrity fails."""


@dataclass(frozen=True)
class FlowEvent:
    """One causally available, side-specific aggregate flow observation.

    Quantities describe displayed resting-side changes.  For example, a market
    buy that removes asks is supplied as ``side="A", execute_quantity=q``.
    Replenishment/depletion are explicit replay labels; they are not inferred
    from a completed future interval.
    """

    event_time_ns: int
    available_time_ns: int | None
    side: str
    publisher_id: int
    instrument_id: int
    raw_symbol: str
    continuity_segment_id: int
    tape_last_ordinal: int
    source_tape_hash: str
    add_quantity: int = 0
    cancel_quantity: int = 0
    execute_quantity: int = 0
    add_events: int = 0
    cancel_events: int = 0
    execute_events: int = 0
    priority_reset_quantity: int = 0
    replenishment_quantity: int = 0
    depletion_quantity: int = 0

    def __post_init__(self) -> None:
        if self.side not in {"B", "A"}:
            raise FeatureProductError("flow-event side must be 'B' or 'A'")
        if not self.raw_symbol:
            raise FeatureProductError("flow-event raw_symbol cannot be empty")
        for name in ("publisher_id", "instrument_id", "continuity_segment_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise FeatureProductError(f"flow-event {name} must be an integer")
        if isinstance(self.event_time_ns, bool) or not isinstance(self.event_time_ns, int):
            raise FeatureProductError("event_time_ns must be an integer")
        if self.available_time_ns is not None and (
            isinstance(self.available_time_ns, bool)
            or not isinstance(self.available_time_ns, int)
        ):
            raise FeatureProductError("available_time_ns must be an integer or None")
        if (
            isinstance(self.tape_last_ordinal, bool)
            or not isinstance(self.tape_last_ordinal, int)
            or self.tape_last_ordinal < 0
        ):
            raise FeatureProductError("tape_last_ordinal must be a nonnegative integer")
        if not SHA256_PATTERN.fullmatch(self.source_tape_hash):
            raise FeatureProductError("source_tape_hash must be a lowercase SHA-256")
        for name in (
            "add_quantity",
            "cancel_quantity",
            "execute_quantity",
            "add_events",
            "cancel_events",
            "execute_events",
            "priority_reset_quantity",
            "replenishment_quantity",
            "depletion_quantity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FeatureProductError(f"{name} must be a nonnegative integer")


def flow_events_from_group(
    result: GroupResult,
    book: FIFOBook,
    *,
    source_tape_hash: str,
) -> tuple[FlowEvent, FlowEvent]:
    """Derive the only certifiable flow observations from a completed replay group.

    Callers cannot provide empirical add/cancel/execute labels directly.  This
    adapter checks the post-group book cursor and derives quantities from the
    replay engine's mutation decomposition. Snapshot payload adds are excluded.
    """

    if result.available_time_ns is None or not result.cause_labels_eligible:
        raise FeatureProductError("ambiguous/unavailable group cannot produce certified flow")
    if (
        book.publisher_id != result.publisher_id
        or book.instrument_id != result.instrument_id
        or book.raw_symbol != result.raw_symbol
        or book.continuity_segment_id != result.continuity_segment_id
    ):
        raise FeatureProductError("completed group and FIFO book identities differ")
    if (
        not book.causal_state_valid
        or book.last_group_ordinal != result.last_ordinal
        or book.last_group_available_ns != result.available_time_ns
        or book.source_tape_hash != source_tape_hash
    ):
        raise FeatureProductError("FIFO book is not the exact post-group state")
    totals: dict[str, dict[str, int]] = {
        side: {field: 0 for field in _FLOW_FIELDS} for side in ("B", "A")
    }
    if not result.snapshot:
        for mutation in result.mutations:
            if mutation.side not in {"B", "A"}:
                continue
            values = totals[mutation.side]
            if mutation.added_quantity:
                values["add_quantity"] += int(mutation.added_quantity)
                values["add_events"] += 1
            if mutation.executed_removed_quantity:
                values["execute_quantity"] += int(
                    mutation.executed_removed_quantity
                )
                values["execute_events"] += 1
            if mutation.withdrawn_removed_quantity:
                values["cancel_quantity"] += int(
                    mutation.withdrawn_removed_quantity
                )
                values["cancel_events"] += 1
            if mutation.priority_reset:
                values["priority_reset_quantity"] += int(
                    mutation.removed_quantity
                )
            values["depletion_quantity"] += int(
                mutation.executed_removed_quantity
                + mutation.withdrawn_removed_quantity
            )
        # The publisher group is atomic at the decision clock: row ordering
        # inside it is not actionable.  This is therefore a conservative
        # *co-occurring refill* proxy, not a claim that the add followed the
        # removal or occurred at the best price.
        for side in ("B", "A"):
            if totals[side]["depletion_quantity"]:
                totals[side]["replenishment_quantity"] = totals[side][
                    "add_quantity"
                ]
    common = {
        "event_time_ns": int(result.event_time_ns),
        "available_time_ns": int(result.available_time_ns),
        "publisher_id": int(result.publisher_id),
        "instrument_id": int(result.instrument_id),
        "raw_symbol": str(result.raw_symbol),
        "continuity_segment_id": int(result.continuity_segment_id),
        "tape_last_ordinal": int(result.last_ordinal),
        "source_tape_hash": source_tape_hash,
    }
    return tuple(
        FlowEvent(side=side, **common, **totals[side]) for side in ("B", "A")
    )  # type: ignore[return-value]


_FLOW_FIELDS = (
    "add_quantity",
    "cancel_quantity",
    "execute_quantity",
    "add_events",
    "cancel_events",
    "execute_events",
    "priority_reset_quantity",
    "replenishment_quantity",
    "depletion_quantity",
)


class _RollingWindow:
    """Incremental exact ``(t-window, t]`` aggregation."""

    def __init__(self, window_ns: int) -> None:
        self.window_ns = int(window_ns)
        self.events: deque[FlowEvent] = deque()
        self.totals: dict[tuple[str, str], int] = {
            (side, field): 0 for side in ("B", "A") for field in _FLOW_FIELDS
        }

    def clear(self) -> None:
        self.events.clear()
        for key in self.totals:
            self.totals[key] = 0

    def add(self, event: FlowEvent) -> None:
        if event.available_time_ns is None:
            raise FeatureProductError("an unavailable event cannot enter a rolling window")
        self.events.append(event)
        for field in _FLOW_FIELDS:
            self.totals[(event.side, field)] += int(getattr(event, field))
        self.advance(event.available_time_ns)

    def advance(self, now_ns: int) -> None:
        boundary = int(now_ns) - self.window_ns
        while self.events and int(self.events[0].available_time_ns) <= boundary:
            expired = self.events.popleft()
            for field in _FLOW_FIELDS:
                key = (expired.side, field)
                self.totals[key] -= int(getattr(expired, field))

    def get(self, side: str, field: str) -> int:
        return self.totals[(side, field)]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _book_feature_values(book: FIFOBook, available_time_ns: int) -> dict[str, float | int]:
    bid = book.best_price("B")
    ask = book.best_price("A")
    values: dict[str, float | int] = {
        "best_bid_ticks": float(bid) if bid is not None else float("nan"),
        "best_ask_ticks": float(ask) if ask is not None else float("nan"),
        "spread_ticks": float(ask - bid) if bid is not None and ask is not None else float("nan"),
        "mid_ticks": float((bid + ask) / 2) if bid is not None and ask is not None else float("nan"),
    }
    for depth in (1, 3, 5):
        bid_depth = sum(quantity for _, quantity in book.price_levels("B", depth))
        ask_depth = sum(quantity for _, quantity in book.price_levels("A", depth))
        values[f"bid_depth_l{depth}"] = int(bid_depth)
        values[f"ask_depth_l{depth}"] = int(ask_depth)
        values[f"imbalance_l{depth}"] = _safe_ratio(
            bid_depth - ask_depth, bid_depth + ask_depth
        )
    bid_one = int(values["bid_depth_l1"])
    ask_one = int(values["ask_depth_l1"])
    if bid is None or ask is None or bid_one + ask_one == 0:
        microprice = float("nan")
    else:
        microprice = (ask * bid_one + bid * ask_one) / (bid_one + ask_one)
    values["microprice_ticks"] = float(microprice)
    values["microprice_displacement_ticks"] = (
        float(microprice - values["mid_ticks"])
        if not math.isnan(microprice) and not math.isnan(float(values["mid_ticks"]))
        else float("nan")
    )

    orders = list(book.iter_orders())
    for native_side, side in (("B", "bid"), ("A", "ask")):
        side_orders = [order for order in orders if order.side == native_side]
        sizes = np.asarray([order.size for order in side_orders], dtype=float)
        total_size = float(sizes.sum()) if sizes.size else 0.0
        values[f"{side}_order_count"] = len(side_orders)
        values[f"{side}_size_hhi"] = (
            float(np.sum((sizes / total_size) ** 2)) if total_size else float("nan")
        )
        values[f"{side}_top_order_share"] = (
            float(sizes.max() / total_size) if total_size else float("nan")
        )
        values[f"{side}_left_censored_share"] = (
            float(np.mean([order.left_censored for order in side_orders]))
            if side_orders
            else float("nan")
        )
        uncensored_ages = np.asarray(
            [
                max(0, available_time_ns - order.birth_available_ns) / 1e9
                for order in side_orders
                if not order.left_censored
            ],
            dtype=float,
        )
        for quantile in (25, 50, 75):
            values[f"{side}_age_q{quantile}_s"] = (
                float(np.quantile(uncensored_ages, quantile / 100))
                if uncensored_ages.size
                else float("nan")
            )
        front_quantity = 0
        top3_depth = 0
        for price, depth in book.price_levels(native_side, 3):
            top3_depth += depth
            keys = book.ordered_keys(native_side, price)
            if keys:
                live = book.orders.get(keys[0].order_id)
                if live is not None and live.key == keys[0]:
                    front_quantity += live.size
        values[f"{side}_front_concentration_l3"] = _safe_ratio(
            front_quantity, top3_depth
        )
    return values


def _window_values(
    get: Any,
    *,
    label: str,
    window_ns: int,
    book_values: Mapping[str, Any],
    include_l3: bool,
) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    add_bid = get("B", "add_quantity")
    add_ask = get("A", "add_quantity")
    cancel_bid = get("B", "cancel_quantity")
    cancel_ask = get("A", "cancel_quantity")
    execute_bid = get("B", "execute_quantity")
    execute_ask = get("A", "execute_quantity")
    output[f"ofi_qty_{label}"] = (
        add_bid - cancel_bid - execute_bid - add_ask + cancel_ask + execute_ask
    )
    for kind in ("add", "cancel", "execute"):
        output[f"{kind}_bid_qty_{label}"] = get("B", f"{kind}_quantity")
        output[f"{kind}_ask_qty_{label}"] = get("A", f"{kind}_quantity")
        count = get("B", f"{kind}_events") + get("A", f"{kind}_events")
        output[f"{kind}_intensity_per_s_{label}"] = count / (window_ns / 1e9)
    if include_l3:
        for native_side, side in (("B", "bid"), ("A", "ask")):
            add = get(native_side, "add_quantity")
            cancel = get(native_side, "cancel_quantity")
            execute = get(native_side, "execute_quantity")
            reset = get(native_side, "priority_reset_quantity")
            output[f"priority_reset_{side}_qty_{label}"] = reset
            live_depth = sum(
                int(book_values[f"{side}_depth_l{depth}"])
                for depth in (5,)
            )
            output[f"churn_{side}_ratio_{label}"] = (add + cancel + reset) / max(
                live_depth, 1
            )
            output[f"replenishment_{side}_qty_{label}"] = get(
                native_side, "replenishment_quantity"
            )
            output[f"depletion_{side}_qty_{label}"] = get(
                native_side, "depletion_quantity"
            )
            output[f"execution_removal_{side}_share_{label}"] = _safe_ratio(
                execute, execute + cancel
            )
    return output


class RentFeatureAccumulator:
    """Available-time incremental state for one symbol/continuity segment."""

    def __init__(
        self,
        *,
        publisher_id: int,
        instrument_id: int,
        raw_symbol: str,
        continuity_segment_id: int,
        segment_start_available_ns: int,
        source_tape_hash: str,
        initial_tape_ordinal: int,
        book_epoch: int = 0,
        registry: FeatureRegistry = RENT_FEATURE_REGISTRY,
    ) -> None:
        if not raw_symbol:
            raise FeatureProductError("raw_symbol cannot be empty")
        if not SHA256_PATTERN.fullmatch(source_tape_hash):
            raise FeatureProductError("source_tape_hash must be a lowercase SHA-256")
        self.publisher_id = int(publisher_id)
        self.instrument_id = int(instrument_id)
        self.raw_symbol = raw_symbol
        self.continuity_segment_id = int(continuity_segment_id)
        self.registry = registry
        self.windows = {label: _RollingWindow(window) for label, window in WINDOWS_NS}
        self.capacity_window = _RollingWindow(TRAILING_WINDOW_NS)
        self.clean_clock = True
        self.segment_start_available_ns = int(segment_start_available_ns)
        self.last_available_time_ns = int(segment_start_available_ns)
        self.last_event_time_ns = int(segment_start_available_ns)
        self.book_epoch = int(book_epoch)
        self.source_tape_hash = source_tape_hash
        if initial_tape_ordinal < 0:
            raise FeatureProductError("initial_tape_ordinal must be nonnegative")
        self.last_tape_ordinal = int(initial_tape_ordinal)

    def reset(
        self,
        *,
        available_time_ns: int,
        event_time_ns: int,
        book_epoch: int,
        tape_last_ordinal: int,
    ) -> None:
        """Start a new causally isolated epoch and discard every rolling value."""

        if event_time_ns > available_time_ns:
            raise FeatureProductError("reset event is not causally available")
        if available_time_ns < self.last_available_time_ns:
            raise FeatureProductError("reset availability clock moved backwards")
        for window in (*self.windows.values(), self.capacity_window):
            window.clear()
        self.clean_clock = True
        self.segment_start_available_ns = int(available_time_ns)
        self.last_available_time_ns = int(available_time_ns)
        self.last_event_time_ns = int(event_time_ns)
        self.book_epoch = int(book_epoch)
        if tape_last_ordinal < self.last_tape_ordinal:
            raise FeatureProductError("reset tape ordinal moved backwards")
        self.last_tape_ordinal = int(tape_last_ordinal)

    def invalidate_bad_clock(self, *, event_time_ns: int) -> None:
        """Fail closed until an explicit certified reset/snapshot boundary."""

        for window in (*self.windows.values(), self.capacity_window):
            window.clear()
        self.clean_clock = False
        # The unusable event is not represented as an available input.  Retain
        # the last safe exchange clock so the product's causal clock chain does
        # not pretend that the bad event was consumed.

    def update(self, event: FlowEvent) -> None:
        if (
            event.publisher_id != self.publisher_id
            or event.instrument_id != self.instrument_id
            or event.raw_symbol != self.raw_symbol
            or event.continuity_segment_id != self.continuity_segment_id
        ):
            raise FeatureProductError("flow event and feature accumulator identities differ")
        if event.source_tape_hash != self.source_tape_hash:
            raise FeatureProductError("flow event belongs to a different source tape")
        if event.tape_last_ordinal < self.last_tape_ordinal:
            raise FeatureProductError("flow-event tape ordinal moved backwards")
        self.last_tape_ordinal = int(event.tape_last_ordinal)
        if event.available_time_ns is None or event.event_time_ns > event.available_time_ns:
            self.invalidate_bad_clock(event_time_ns=event.event_time_ns)
            return
        if event.available_time_ns < self.last_available_time_ns:
            raise FeatureProductError("available-time flow events are not monotone")
        self.last_available_time_ns = int(event.available_time_ns)
        self.last_event_time_ns = max(self.last_event_time_ns, int(event.event_time_ns))
        if not self.clean_clock:
            return
        for window in (*self.windows.values(), self.capacity_window):
            window.add(event)

    def _validate_book(self, book: FIFOBook) -> None:
        if (
            book.publisher_id != self.publisher_id
            or book.instrument_id != self.instrument_id
            or book.raw_symbol != self.raw_symbol
            or book.continuity_segment_id != self.continuity_segment_id
        ):
            raise FeatureProductError("feature accumulator and FIFO book identities differ")
        if book.book_epoch != self.book_epoch:
            raise FeatureProductError("book epoch changed without an accumulator reset")
        if book.source_tape_hash != self.source_tape_hash:
            raise FeatureProductError("feature accumulator and FIFO book tapes differ")
        if not book.causal_state_valid or book.last_group_available_ns is None:
            raise FeatureProductError("FIFO book is not bound to a causal complete group")
        if book.last_group_ordinal is None:
            raise FeatureProductError("FIFO book has no completed-group ordinal")
        if self.last_tape_ordinal != book.last_group_ordinal:
            raise FeatureProductError("flow accumulator and FIFO book cursors differ")

    def snapshot(
        self,
        book: FIFOBook,
        *,
        feature_join_key: str,
        feature_time_ns: int,
        decision_time_ns: int,
        tape_last_ordinal: int,
        source_tape_hash: str,
    ) -> dict[str, Any]:
        self._validate_book(book)
        if not feature_join_key:
            raise FeatureProductError("feature_join_key cannot be empty")
        if not SHA256_PATTERN.fullmatch(source_tape_hash):
            raise FeatureProductError("source_tape_hash must be a lowercase SHA-256")
        if source_tape_hash != self.source_tape_hash:
            raise FeatureProductError("snapshot source tape differs from accumulator")
        if int(tape_last_ordinal) != book.last_group_ordinal:
            raise FeatureProductError("snapshot ordinal does not match FIFO book state")
        if book.last_group_available_ns > feature_time_ns:
            raise FeatureProductError("FIFO book state is not available at feature time")
        if not (
            self.last_available_time_ns <= feature_time_ns <= decision_time_ns
        ):
            raise FeatureProductError("feature/decision clocks are not causal")
        for window in (*self.windows.values(), self.capacity_window):
            window.advance(feature_time_ns)
        metadata = {
            "feature_join_key": feature_join_key,
            "publisher_id": self.publisher_id,
            "instrument_id": self.instrument_id,
            "raw_symbol": self.raw_symbol,
            "continuity_segment_id": self.continuity_segment_id,
            "book_epoch": self.book_epoch,
            "tape_last_ordinal": int(tape_last_ordinal),
            "event_time_ns": self.last_event_time_ns,
            "available_time_ns": self.last_available_time_ns,
            "feature_time_ns": int(feature_time_ns),
            "decision_time_ns": int(decision_time_ns),
            "source_tape_hash": source_tape_hash,
            "feature_registry_sha256": self.registry.content_sha256,
        }
        warm = {
            label: self.clean_clock
            and feature_time_ns - self.segment_start_available_ns >= window_ns
            for label, window_ns in WINDOWS_NS
        }
        best_bid = book.best_price("B")
        best_ask = book.best_price("A")
        two_sided = (
            best_bid is not None and best_ask is not None and best_bid < best_ask
        )
        masks = {
            "mask_clean_clock": self.clean_clock,
            "mask_two_sided_book": self.clean_clock and two_sided,
            "mask_warm_250ms": warm["250ms"],
            "mask_warm_1s": warm["1s"],
            "mask_warm_5s": warm["5s"],
            "mask_l3_lifecycle": self.clean_clock,
            "mask_l3_age_bid": self.clean_clock
            and any(order.side == "B" and not order.left_censored for order in book.iter_orders()),
            "mask_l3_age_ask": self.clean_clock
            and any(order.side == "A" and not order.left_censored for order in book.iter_orders()),
        }
        if not self.clean_clock:
            values = {column: float("nan") for column in self.registry.feature_columns("l3")}
        else:
            values = _book_feature_values(book, feature_time_ns)
            for label, window_ns in WINDOWS_NS:
                rolling = self.windows[label]
                values.update(
                    _window_values(
                        rolling.get,
                        label=label,
                        window_ns=window_ns,
                        book_values=values,
                        include_l3=label in {"1s", "5s"},
                    )
                )
        row = {**metadata, **masks, **values}
        row["l2_feature_hash"] = feature_view_content_sha256(
            row, view="l2", registry=self.registry
        )
        row["l3_feature_hash"] = feature_view_content_sha256(
            row, view="l3", registry=self.registry
        )
        return row

    def capacity_state(
        self, book: FIFOBook, *, arrival_time_ns: int, source_tape_hash: str = ""
    ) -> ArrivalCapacityState:
        """Return a five-minute volume/depth state evaluated at actual arrival."""

        self._validate_book(book)
        if not self.clean_clock:
            raise FeatureProductError("bad-clock state cannot support capacity")
        if arrival_time_ns < self.last_available_time_ns:
            raise FeatureProductError("capacity arrival precedes available state")
        if book.last_group_available_ns > arrival_time_ns:
            raise FeatureProductError("FIFO book is not available at capacity arrival")
        if source_tape_hash and source_tape_hash != self.source_tape_hash:
            raise FeatureProductError("capacity source tape differs from accumulator")
        self.capacity_window.advance(arrival_time_ns)
        return ArrivalCapacityState(
            raw_symbol=self.raw_symbol,
            available_time_ns=int(arrival_time_ns),
            book_available_time_ns=int(book.last_group_available_ns),
            book_last_ordinal=int(book.last_group_ordinal),
            best_bid_depth=sum(quantity for _, quantity in book.price_levels("B", 1)),
            best_ask_depth=sum(quantity for _, quantity in book.price_levels("A", 1)),
            bid_top3_depth=sum(quantity for _, quantity in book.price_levels("B", 3)),
            ask_top3_depth=sum(quantity for _, quantity in book.price_levels("A", 3)),
            trailing_aggressive_volume_5m=(
                self.capacity_window.get("B", "execute_quantity")
                + self.capacity_window.get("A", "execute_quantity")
            ),
            trailing_window_start_ns=int(arrival_time_ns - TRAILING_WINDOW_NS),
            trailing_window_end_ns=int(arrival_time_ns),
            trailing_window_complete=(
                arrival_time_ns - self.segment_start_available_ns >= TRAILING_WINDOW_NS
            ),
            source_tape_hash=self.source_tape_hash,
        )


def reference_feature_values(
    book: FIFOBook,
    events: Sequence[FlowEvent],
    *,
    feature_time_ns: int,
    segment_start_available_ns: int,
) -> dict[str, float | int | bool]:
    """Slow reference formulas for production-accumulator parity tests."""

    available_clocks = [
        int(event.available_time_ns)
        for event in events
        if event.available_time_ns is not None
    ]
    clean = all(
        event.available_time_ns is not None
        and event.event_time_ns <= int(event.available_time_ns)
        and int(event.available_time_ns) <= feature_time_ns
        for event in events
    ) and available_clocks == sorted(available_clocks)
    if not clean:
        return {
            **{column: float("nan") for column in RENT_FEATURE_REGISTRY.feature_columns("l3")},
            "mask_clean_clock": False,
        }
    values = _book_feature_values(book, feature_time_ns)
    for label, window_ns in WINDOWS_NS:
        selected = [
            event
            for event in events
            if feature_time_ns - window_ns < int(event.available_time_ns) <= feature_time_ns
        ]

        def get(side: str, field: str) -> int:
            return sum(int(getattr(event, field)) for event in selected if event.side == side)

        values.update(
            _window_values(
                get,
                label=label,
                window_ns=window_ns,
                book_values=values,
                include_l3=label in {"1s", "5s"},
            )
        )
    values["mask_clean_clock"] = True
    values.update(
        {
            f"mask_warm_{label}": feature_time_ns - segment_start_available_ns >= window_ns
            for label, window_ns in WINDOWS_NS
        }
    )
    return values


def _canonical_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return None
        if not math.isfinite(numeric):
            raise FeatureProductError("feature hashes cannot contain infinity")
        return 0.0 if numeric == 0.0 else numeric
    return str(value)


def feature_view_content_sha256(
    row: Mapping[str, Any],
    *,
    view: str,
    registry: FeatureRegistry = RENT_FEATURE_REGISTRY,
) -> str:
    """Hash one feature view, its masks, causal clocks, and source identity."""

    columns = (
        *FEATURE_METADATA_COLUMNS,
        *registry.mask_columns(view),
        *registry.feature_columns(view),
    )
    missing = sorted(set(columns) - set(row.keys()))
    if missing:
        raise FeatureProductError(f"cannot hash {view}: missing columns {missing}")
    payload = {
        "feature_product_schema_version": FEATURE_PRODUCT_SCHEMA_VERSION,
        "view": view,
        "columns": [(column, _canonical_scalar(row[column])) for column in columns],
    }
    return sha256_json(payload)


def _strict_bool(value: Any, *, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise FeatureProductError(f"{column} contains a non-boolean value {value!r}")


def _validate_sha(value: Any, *, column: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise FeatureProductError(f"{column} must contain lowercase SHA-256 values")
    return text


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    registry: FeatureRegistry = RENT_FEATURE_REGISTRY,
) -> pd.DataFrame:
    """Return a canonical, key-sorted frame after fail-closed validation."""

    required = (
        *FEATURE_METADATA_COLUMNS,
        *L3_MASK_COLUMNS,
        *registry.feature_columns("l3"),
        *HASH_COLUMNS,
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise FeatureProductError(f"feature frame is missing columns: {missing}")
    if frame.empty:
        raise FeatureProductError("feature frame cannot be empty")
    output = frame.loc[:, list(required)].copy()
    for column in (
        "feature_join_key",
        "raw_symbol",
    ):
        if bool(output[column].isna().any()) or bool(output[column].astype(str).str.strip().eq("").any()):
            raise FeatureProductError(f"{column} cannot be empty")
    if output.duplicated("feature_join_key").any():
        raise FeatureProductError("feature_join_key must be unique")
    for column in (
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "book_epoch",
        "tape_last_ordinal",
        "event_time_ns",
        "available_time_ns",
        "feature_time_ns",
        "decision_time_ns",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    bad_clock = ~(
        (output["event_time_ns"] <= output["available_time_ns"])
        & (output["available_time_ns"] <= output["feature_time_ns"])
        & (output["feature_time_ns"] <= output["decision_time_ns"])
    )
    # A bad receipt invalidates features, but its last safe input clock still
    # has to be monotone.  Rows never encode an unavailable event as available.
    if bool(bad_clock.any()):
        raise FeatureProductError("feature clocks are not causally ordered")
    for column in L3_MASK_COLUMNS:
        output[column] = output[column].map(lambda value: _strict_bool(value, column=column)).astype(bool)
    dirty = ~output["mask_clean_clock"]
    downstream_masks = tuple(
        column for column in L3_MASK_COLUMNS if column != "mask_clean_clock"
    )
    if bool(output.loc[dirty, list(downstream_masks)].any(axis=None)):
        raise FeatureProductError("bad-clock rows must fail every downstream validity mask")
    numeric_columns = list(registry.feature_columns("l3"))
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="raise")
    values = output[numeric_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise FeatureProductError("features may be NaN but cannot be infinite")
    if bool(output.loc[dirty, numeric_columns].notna().any(axis=None)):
        raise FeatureProductError("bad-clock rows must not expose feature values")
    registry_hash = registry.content_sha256
    if not output["feature_registry_sha256"].astype(str).eq(registry_hash).all():
        raise FeatureProductError("feature row is not bound to the supplied registry")
    for column in ("source_tape_hash", "feature_registry_sha256", *HASH_COLUMNS):
        output[column] = output[column].map(lambda value: _validate_sha(value, column=column))
    for index, row in output.iterrows():
        for view, hash_column in (("l2", "l2_feature_hash"), ("l3", "l3_feature_hash")):
            observed = feature_view_content_sha256(row, view=view, registry=registry)
            if observed != row[hash_column]:
                raise FeatureProductError(
                    f"{hash_column} mismatch for feature_join_key={row['feature_join_key']!r}"
                )
    return output.sort_values(
        ["decision_time_ns", "feature_join_key"], kind="stable"
    ).reset_index(drop=True)


def feature_frame_content_sha256(
    frame: pd.DataFrame, *, registry: FeatureRegistry = RENT_FEATURE_REGISTRY
) -> str:
    canonical = validate_feature_frame(frame, registry=registry)
    return sha256_json(
        {
            "schema_version": FEATURE_PRODUCT_SCHEMA_VERSION,
            "columns": [(column, str(canonical[column].dtype)) for column in canonical.columns],
            "rows": [
                {
                    "feature_join_key": row.feature_join_key,
                    "l2_feature_hash": row.l2_feature_hash,
                    "l3_feature_hash": row.l3_feature_hash,
                }
                for row in canonical.itertuples(index=False)
            ],
        }
    )


@dataclass(frozen=True)
class FeatureProduct:
    features: pd.DataFrame
    manifest: Mapping[str, Any]
    registry: FeatureRegistry
    # Populated only by create_empirical_feature_product.  Keeping this payload
    # out of the ordinary constructor preserves lightweight synthetic fixtures,
    # while empirical consumers require the serialized/recomputed certificate.
    derivation: Any | None = None


def create_feature_product(
    features: pd.DataFrame,
    *,
    product_id: str,
    data_version: str,
    source_manifest_sha256: str,
    evidence_class: str = "DEVELOPMENT_ONLY",
    registry: FeatureRegistry = RENT_FEATURE_REGISTRY,
) -> FeatureProduct:
    if not product_id or not data_version:
        raise FeatureProductError("product_id and data_version cannot be empty")
    source_manifest_sha256 = _validate_sha(
        source_manifest_sha256, column="source_manifest_sha256"
    )
    evidence_class = evidence_class.upper()
    if evidence_class not in ALLOWED_EVIDENCE_CLASSES:
        raise FeatureProductError(f"unsupported evidence_class {evidence_class!r}")
    canonical = validate_feature_frame(features, registry=registry)
    manifest: dict[str, Any] = {
        "schema_version": FEATURE_PRODUCT_SCHEMA_VERSION,
        "product_type": FEATURE_PRODUCT_TYPE,
        "product_id": product_id,
        "data_version": data_version,
        "evidence_class": evidence_class,
        "source_manifest_sha256": source_manifest_sha256,
        "registry": registry.to_dict(),
        "feature_registry_sha256": registry.content_sha256,
        "feature_frame_content_sha256": feature_frame_content_sha256(
            canonical, registry=registry
        ),
        "row_count": len(canonical),
        "first_decision_time_ns": int(canonical["decision_time_ns"].min()),
        "last_decision_time_ns": int(canonical["decision_time_ns"].max()),
    }
    return FeatureProduct(canonical, manifest, registry)


def write_feature_product(path: str | Path, product: FeatureProduct) -> Path:
    """Atomically create a new immutable directory; overwrites are forbidden."""

    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"immutable feature product already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = validate_feature_frame(product.features, registry=product.registry)
    expected_digest = feature_frame_content_sha256(canonical, registry=product.registry)
    if product.manifest.get("feature_frame_content_sha256") != expected_digest:
        raise FeatureProductError("feature product changed after creation")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        feature_path = staging / FEATURE_FILE_NAME
        canonical.to_parquet(feature_path, index=False)
        if product.derivation is not None:
            from .feature_derivation import write_feature_derivation_artifacts

            derivation_declaration: Mapping[str, Any] = (
                write_feature_derivation_artifacts(
                    staging,
                    product.derivation,
                    feature_frame_sha256=expected_digest,
                )
            )
        elif str(product.manifest.get("evidence_class", "")).upper() == "SYNTHETIC_ONLY":
            derivation_declaration = {
                "status": "NOT_REQUIRED_SYNTHETIC",
            }
        else:
            # A caller-supplied empirical frame may still be materialized for
            # debugging, but it is explicitly uncertified and every empirical
            # BranchProduct/training boundary rejects it.
            derivation_declaration = {
                "status": "UNCERTIFIED_CALLER_FRAME",
            }
        manifest = {
            **dict(product.manifest),
            "feature_derivation": dict(derivation_declaration),
            "artifacts": {
                "features": {
                    "path": FEATURE_FILE_NAME,
                    "sha256": sha256_file(feature_path),
                }
            },
        }
        manifest_path = write_json(staging / MANIFEST_FILE_NAME, manifest)
        (staging / MANIFEST_CHECKSUM_FILE_NAME).write_text(
            f"{sha256_file(manifest_path)}  {MANIFEST_FILE_NAME}\n", encoding="utf-8"
        )
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_feature_product(path: str | Path) -> FeatureProduct:
    source = Path(path).expanduser().resolve()
    manifest_path = source / MANIFEST_FILE_NAME
    checksum_path = source / MANIFEST_CHECKSUM_FILE_NAME
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FeatureProductError("feature product is missing its manifest/checksum")
    tokens = checksum_path.read_text(encoding="utf-8").strip().split()
    if len(tokens) != 2 or tokens[1] != MANIFEST_FILE_NAME:
        raise FeatureProductError("invalid manifest checksum file")
    if sha256_file(manifest_path) != tokens[0]:
        raise FeatureProductError("feature-product manifest checksum mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FeatureProductError("invalid feature-product manifest JSON") from error
    if (
        int(manifest.get("schema_version", 0)) != FEATURE_PRODUCT_SCHEMA_VERSION
        or manifest.get("product_type") != FEATURE_PRODUCT_TYPE
    ):
        raise FeatureProductError("unsupported feature-product manifest")
    if str(manifest.get("evidence_class", "")) not in ALLOWED_EVIDENCE_CLASSES:
        raise FeatureProductError("unsupported feature-product evidence class")
    if not str(manifest.get("product_id", "")).strip() or not str(
        manifest.get("data_version", "")
    ).strip():
        raise FeatureProductError("feature-product identity fields cannot be empty")
    _validate_sha(
        manifest.get("source_manifest_sha256"), column="source_manifest_sha256"
    )
    registry = registry_from_dict(manifest.get("registry"))
    if sha256_json(registry.to_dict()) != sha256_json(manifest.get("registry")):
        raise FeatureProductError("serialized feature registry is not canonical")
    if registry.content_sha256 != manifest.get("feature_registry_sha256"):
        raise FeatureProductError("serialized feature-registry hash mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("features"), dict):
        raise FeatureProductError("manifest does not bind its feature artifact")
    declaration = artifacts["features"]
    if declaration.get("path") != FEATURE_FILE_NAME:
        raise FeatureProductError("feature artifact path is not canonical")
    feature_path = source / FEATURE_FILE_NAME
    if not feature_path.is_file() or sha256_file(feature_path) != declaration.get("sha256"):
        raise FeatureProductError("feature artifact hash mismatch")
    features = validate_feature_frame(pd.read_parquet(feature_path), registry=registry)
    if len(features) != int(manifest.get("row_count", -1)):
        raise FeatureProductError("feature artifact row count mismatch")
    observed_digest = feature_frame_content_sha256(features, registry=registry)
    if observed_digest != manifest.get("feature_frame_content_sha256"):
        raise FeatureProductError("feature frame content hash mismatch")
    if int(features["decision_time_ns"].min()) != int(
        manifest.get("first_decision_time_ns", -1)
    ) or int(features["decision_time_ns"].max()) != int(
        manifest.get("last_decision_time_ns", -1)
    ):
        raise FeatureProductError("feature decision-clock bounds mismatch")
    return FeatureProduct(features, manifest, registry)


def verify_feature_product(
    path: str | Path,
    *,
    require_empirical_derivation: bool = True,
) -> dict[str, Any]:
    """Load all bytes and return concise verified provenance."""

    product = load_feature_product(path)
    source = Path(path).expanduser().resolve()
    evidence_class = str(product.manifest["evidence_class"])
    derivation: dict[str, Any] | None = None
    if evidence_class != "SYNTHETIC_ONLY":
        declaration = product.manifest.get("feature_derivation")
        recomputable = (
            isinstance(declaration, Mapping)
            and declaration.get("status") == "RECOMPUTABLE"
        )
        if require_empirical_derivation or recomputable:
            from .feature_derivation import verify_feature_derivation

            derivation = verify_feature_derivation(source, product=product)
    return {
        "product_id": product.manifest["product_id"],
        "evidence_class": evidence_class,
        "row_count": len(product.features),
        "feature_registry_sha256": product.registry.content_sha256,
        "feature_frame_content_sha256": product.manifest[
            "feature_frame_content_sha256"
        ],
        "manifest_sha256": sha256_file(source / MANIFEST_FILE_NAME),
        "features_sha256": sha256_file(source / FEATURE_FILE_NAME),
        "derivation_verified": derivation is not None,
        "feature_derivation": derivation,
    }


__all__ = [
    "ALLOWED_EVIDENCE_CLASSES",
    "FEATURE_FILE_NAME",
    "FEATURE_PRODUCT_SCHEMA_VERSION",
    "FEATURE_PRODUCT_TYPE",
    "HASH_COLUMNS",
    "MANIFEST_CHECKSUM_FILE_NAME",
    "MANIFEST_FILE_NAME",
    "FeatureProduct",
    "FeatureProductError",
    "FlowEvent",
    "RentFeatureAccumulator",
    "create_feature_product",
    "feature_frame_content_sha256",
    "feature_view_content_sha256",
    "flow_events_from_group",
    "load_feature_product",
    "reference_feature_values",
    "validate_feature_frame",
    "verify_feature_product",
    "write_feature_product",
]
