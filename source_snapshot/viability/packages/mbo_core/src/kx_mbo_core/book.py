from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .types import EventRecord, LifeKey


@dataclass
class OrderState:
    key: LifeKey
    side: str
    price_ticks: int
    size: int
    birth_event_ns: int
    birth_available_ns: int
    priority_ordinal: int
    left_censored: bool = False
    executed_quantity: int = 0
    withdrawn_quantity: int = 0
    modify_count: int = 0


@dataclass(frozen=True)
class Mutation:
    action: str
    order_key: LifeKey | None = None
    removed_quantity: int = 0
    added_quantity: int = 0
    terminal: bool = False
    priority_reset: bool = False
    duplicate_noop: bool = False
    recovery_left_censored: bool = False
    detail: str = ""
    anomaly_code: str | None = None
    side: str = ""
    executed_removed_quantity: int = 0
    withdrawn_removed_quantity: int = 0


class FIFOBook:
    """One-contract deterministic price-time book.

    The implementation favors auditability over cleverness. Per-price queues are
    ordered dictionaries, so insertion/deletion is deterministic and queue-ahead
    queries are exact for pilot-sized landmarks.
    """

    def __init__(
        self,
        publisher_id: int,
        raw_symbol: str,
        continuity_segment_id: int,
        *,
        instrument_id: int = 0,
        source_tape_hash: str = "",
    ) -> None:
        self.publisher_id = publisher_id
        self.raw_symbol = raw_symbol
        self.continuity_segment_id = continuity_segment_id
        self.instrument_id = int(instrument_id)
        self.book_epoch = 0
        self.orders: dict[int, OrderState] = {}
        self.levels: dict[str, dict[int, OrderedDict[LifeKey, None]]] = {
            "B": defaultdict(OrderedDict),
            "A": defaultdict(OrderedDict),
        }
        self.life_instances: Counter[int] = Counter()
        self.priority_counter = 0
        self.anomalies: Counter[str] = Counter()
        # These fields bind a mutable book to the last *complete* atomic event
        # that produced it.  Feature builders must never attach caller-supplied
        # clocks or ordinals to an otherwise unbound/stale book.
        self.source_tape_hash = str(source_tape_hash)
        self.last_group_ordinal: int | None = None
        self.last_group_available_ns: int | None = None
        self.causal_state_valid = True

    def bind_completed_group(
        self,
        *,
        last_ordinal: int,
        available_time_ns: int | None,
        source_tape_hash: str,
        restores_causal_state: bool = False,
    ) -> None:
        """Bind book state to one complete publisher event.

        A group with no trustworthy receive clock invalidates the state until a
        certified snapshot/reset explicitly restores it.  Equal ordinals are
        not permitted because a book is bound once, after the whole group.
        """

        ordinal = int(last_ordinal)
        if self.last_group_ordinal is not None and ordinal <= self.last_group_ordinal:
            raise ValueError("completed-group ordinal did not advance")
        tape_hash = str(source_tape_hash)
        if self.source_tape_hash and tape_hash and tape_hash != self.source_tape_hash:
            raise ValueError("completed group belongs to a different source tape")
        if tape_hash:
            self.source_tape_hash = tape_hash
        self.last_group_ordinal = ordinal
        self.last_group_available_ns = (
            None if available_time_ns is None else int(available_time_ns)
        )
        if available_time_ns is None:
            self.causal_state_valid = False
        elif restores_causal_state:
            self.causal_state_valid = True

    def _new_key(self, order_id: int, priority_epoch: int = 0) -> LifeKey:
        return LifeKey(
            self.publisher_id,
            self.raw_symbol,
            self.continuity_segment_id,
            self.book_epoch,
            int(order_id),
            int(self.life_instances[order_id]),
            int(priority_epoch),
        )

    def _append(self, order: OrderState) -> None:
        self.orders[order.key.order_id] = order
        self.levels[order.side][order.price_ticks][order.key] = None
        self.priority_counter = max(self.priority_counter, order.priority_ordinal)

    def _unlink(self, order: OrderState) -> None:
        queue = self.levels[order.side].get(order.price_ticks)
        if queue is not None:
            queue.pop(order.key, None)
            if not queue:
                self.levels[order.side].pop(order.price_ticks, None)
        self.orders.pop(order.key.order_id, None)

    def seed_order(
        self,
        *,
        order_id: int,
        side: str,
        price_ticks: int,
        size: int,
        priority_ordinal: int,
        birth_event_ns: int,
        birth_available_ns: int | None = None,
        left_censored: bool = True,
    ) -> OrderState:
        if order_id in self.orders:
            raise ValueError(f"duplicate seed order {order_id}")
        self.life_instances[order_id] += 1
        key = self._new_key(order_id)
        order = OrderState(
            key=key,
            side=side,
            price_ticks=int(price_ticks),
            size=int(size),
            birth_event_ns=int(birth_event_ns),
            birth_available_ns=int(birth_available_ns or birth_event_ns),
            priority_ordinal=int(priority_ordinal),
            left_censored=left_censored,
        )
        self._append(order)
        return order

    def add(self, record: EventRecord, available_time_ns: int, *, left_censored: bool = False) -> Mutation:
        existing = self.orders.get(record.order_id)
        anomaly_code: str | None = None
        if existing is not None:
            if (
                existing.side == record.side
                and existing.price_ticks == record.price_ticks
                and existing.size == record.size
            ):
                self.anomalies["idempotent_duplicate_add"] += 1
                return Mutation("A", existing.key, duplicate_noop=True, side=existing.side)
            self.anomalies["conflicting_duplicate_add"] += 1
            anomaly_code = "conflicting_duplicate_add"
            self._unlink(existing)
        self.life_instances[record.order_id] += 1
        self.priority_counter += 1
        key = self._new_key(record.order_id)
        order = OrderState(
            key=key,
            side=record.side,
            price_ticks=record.price_ticks,
            size=record.size,
            birth_event_ns=record.ts_event_ns,
            birth_available_ns=available_time_ns,
            priority_ordinal=self.priority_counter,
            left_censored=left_censored,
        )
        self._append(order)
        return Mutation(
            "A",
            key,
            added_quantity=record.size,
            recovery_left_censored=left_censored,
            detail="conflicting duplicate add replaced conservatively" if anomaly_code else "",
            anomaly_code=anomaly_code,
            side=record.side,
        )

    def cancel(self, record: EventRecord) -> Mutation:
        existing = self.orders.get(record.order_id)
        if existing is None:
            self.anomalies["cancel_missing_order"] += 1
            return Mutation(
                "C", detail="missing order", anomaly_code="cancel_missing_order", side=record.side
            )
        removed = min(int(record.size), existing.size)
        anomaly_code: str | None = None
        if record.size > existing.size:
            self.anomalies["cancel_exceeds_remainder"] += 1
            anomaly_code = "cancel_exceeds_remainder"
        existing.size -= removed
        terminal = existing.size == 0
        key = existing.key
        if terminal:
            self._unlink(existing)
        return Mutation(
            "C",
            key,
            removed_quantity=removed,
            terminal=terminal,
            detail="cancel quantity exceeded live remainder" if anomaly_code else "",
            anomaly_code=anomaly_code,
            side=existing.side,
        )

    def modify(self, record: EventRecord, available_time_ns: int) -> Mutation:
        existing = self.orders.get(record.order_id)
        if existing is None:
            self.anomalies["modify_missing_order"] += 1
            added = self.add(record, available_time_ns, left_censored=True)
            return Mutation(
                "M",
                added.order_key,
                added_quantity=added.added_quantity,
                recovery_left_censored=True,
                detail="orphan modify represented as censored recovery",
                anomaly_code="modify_missing_order",
                side=record.side,
            )
        same_location = existing.side == record.side and existing.price_ticks == record.price_ticks
        if same_location and record.size <= existing.size:
            removed = existing.size - record.size
            existing.size = record.size
            existing.modify_count += 1
            key = existing.key
            terminal = existing.size == 0
            if terminal:
                self._unlink(existing)
            return Mutation(
                "M", key, removed_quantity=removed, terminal=terminal, side=existing.side
            )

        old = existing
        old_key = old.key
        old_size = old.size
        self._unlink(old)
        self.priority_counter += 1
        key = LifeKey(
            old_key.publisher_id,
            old_key.raw_symbol,
            old_key.continuity_segment_id,
            old_key.book_epoch,
            old_key.order_id,
            old_key.life_instance,
            old_key.priority_epoch + 1,
        )
        replacement = OrderState(
            key=key,
            side=record.side,
            price_ticks=record.price_ticks,
            size=record.size,
            birth_event_ns=record.ts_event_ns,
            birth_available_ns=available_time_ns,
            priority_ordinal=self.priority_counter,
            left_censored=old.left_censored,
            executed_quantity=old.executed_quantity,
            withdrawn_quantity=old.withdrawn_quantity,
            modify_count=old.modify_count + 1,
        )
        self._append(replacement)
        return Mutation(
            "M",
            key,
            removed_quantity=old_size,
            added_quantity=record.size,
            terminal=True,
            priority_reset=True,
            detail=f"old_priority_key={old_key}",
            side=old.side,
        )

    def clear(self) -> tuple[OrderState, ...]:
        terminated = tuple(self.orders.values())
        self.orders.clear()
        self.levels = {"B": defaultdict(OrderedDict), "A": defaultdict(OrderedDict)}
        self.book_epoch += 1
        self.priority_counter = 0
        return terminated

    def apply(self, record: EventRecord, available_time_ns: int) -> Mutation:
        if record.action == "A":
            return self.add(record, available_time_ns, left_censored=record.is_snapshot)
        if record.action == "C":
            return self.cancel(record)
        if record.action == "M":
            return self.modify(record, available_time_ns)
        if record.action == "R":
            count = len(self.orders)
            self.clear()
            return Mutation("R", removed_quantity=count, terminal=True)
        if record.action in {"T", "F", "N"}:
            return Mutation(record.action)
        self.anomalies["unknown_action"] += 1
        return Mutation(
            record.action, detail="unknown action", anomaly_code="unknown_action"
        )

    def best_price(self, side: str) -> int | None:
        if not self.levels[side]:
            return None
        return max(self.levels[side]) if side == "B" else min(self.levels[side])

    def level_depth(self, side: str, price_ticks: int) -> int:
        queue = self.levels[side].get(price_ticks, {})
        return int(sum(self.orders[key.order_id].size for key in queue if key.order_id in self.orders))

    def ordered_keys(self, side: str, price_ticks: int) -> tuple[LifeKey, ...]:
        return tuple(self.levels[side].get(price_ticks, {}).keys())

    def queue_ahead(self, key: LifeKey) -> int:
        order = self.orders.get(key.order_id)
        if order is None or order.key != key:
            raise KeyError(key)
        total = 0
        for candidate in self.ordered_keys(order.side, order.price_ticks):
            if candidate == key:
                return total
            live = self.orders.get(candidate.order_id)
            if live is not None and live.key == candidate:
                total += live.size
        raise KeyError(key)

    def rank(self, key: LifeKey) -> int:
        for index, candidate in enumerate(self.ordered_keys(self.orders[key.order_id].side, self.orders[key.order_id].price_ticks), start=1):
            if candidate == key:
                return index
        raise KeyError(key)

    def price_levels(self, side: str, limit: int | None = None) -> list[tuple[int, int]]:
        prices = sorted(self.levels[side], reverse=side == "B")
        if limit is not None:
            prices = prices[:limit]
        return [(price, self.level_depth(side, price)) for price in prices]

    def iter_orders(self) -> Iterable[OrderState]:
        return self.orders.values()
