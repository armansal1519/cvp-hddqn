from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

from .book import FIFOBook, Mutation, OrderState
from .types import EventRecord, LifeKey


@dataclass(frozen=True)
class Termination:
    key: LifeKey
    time_ns: int
    available_time_ns: int
    cause: str
    quantity: int
    side: str
    price_ticks: int
    age_ns: int
    rank: int | None
    queue_ahead: int | None
    left_censored: bool
    cause_label_eligible: bool = True
    tombstone_eligible: bool = False
    ineligibility_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupResult:
    publisher_id: int
    instrument_id: int
    raw_symbol: str
    continuity_segment_id: int
    first_ordinal: int
    last_ordinal: int
    event_time_ns: int
    available_time_ns: int | None
    records: int
    mutations: tuple[Mutation, ...]
    terminations: tuple[Termination, ...]
    executed_quantity: int
    implied_execution_quantity: int
    withdrawn_quantity: int
    aggressor_remainder_quantity: int
    priority_reset_quantity: int
    ambiguous_cause: bool
    cause_labels_eligible: bool
    ineligibility_reasons: tuple[str, ...]
    snapshot: bool
    snapshot_reconciled: bool
    snapshot_reloaded: bool


class AtomicGroupBuilder:
    def __init__(self) -> None:
        self.buffers: dict[tuple[int, int], list[EventRecord]] = defaultdict(list)
        self.last_ordinal = -1

    def push(self, record: EventRecord) -> tuple[EventRecord, ...] | None:
        if record.global_record_ordinal <= self.last_ordinal:
            raise ValueError("records are not in strictly increasing global order")
        self.last_ordinal = record.global_record_ordinal
        key = (record.publisher_id, record.instrument_id)
        self.buffers[key].append(record)
        if not record.is_last:
            return None
        return tuple(self.buffers.pop(key))

    def unfinished(self) -> dict[tuple[int, int], tuple[EventRecord, ...]]:
        return {key: tuple(value) for key, value in self.buffers.items()}


class ReplayEngine:
    def __init__(
        self, *, decode_latency_ns: int = 1_000_000, source_tape_hash: str = ""
    ) -> None:
        self.decode_latency_ns = int(decode_latency_ns)
        self.source_tape_hash = str(source_tape_hash)
        self.group_builder = AtomicGroupBuilder()
        self.books: dict[tuple[int, int, str, int], FIFOBook] = {}
        self.anomalies: Counter[str] = Counter()

    def book_for(self, record: EventRecord) -> FIFOBook:
        key = (
            record.publisher_id,
            record.instrument_id,
            record.raw_symbol,
            record.continuity_segment_id,
        )
        if key not in self.books:
            self.books[key] = FIFOBook(
                record.publisher_id,
                record.raw_symbol,
                record.continuity_segment_id,
                instrument_id=record.instrument_id,
                source_tape_hash=self.source_tape_hash,
            )
        return self.books[key]

    @staticmethod
    def _pre_state(book: FIFOBook, order_id: int) -> tuple[OrderState | None, int | None, int | None]:
        order = book.orders.get(order_id)
        if order is None:
            return None, None, None
        try:
            return order, book.rank(order.key), book.queue_ahead(order.key)
        except KeyError:
            return order, None, None

    @staticmethod
    def _snapshot_matches(book: FIFOBook, records: tuple[EventRecord, ...]) -> bool:
        additions = [record for record in records if record.action == "A"]
        if not additions or any(record.action not in {"R", "A", "N"} for record in records):
            return False
        expected = {
            record.order_id: (record.side, record.price_ticks, record.size)
            for record in additions
        }
        observed = {
            order_id: (order.side, order.price_ticks, order.size)
            for order_id, order in book.orders.items()
        }
        if expected != observed:
            return False
        expected_queues: dict[tuple[str, int], list[int]] = defaultdict(list)
        for record in additions:
            expected_queues[(record.side, record.price_ticks)].append(record.order_id)
        for (side, price), order_ids in expected_queues.items():
            live_ids = [key.order_id for key in book.ordered_keys(side, price)]
            if live_ids != order_ids:
                return False
        return True

    def apply_group(self, records: tuple[EventRecord, ...]) -> GroupResult:
        if not records or not records[-1].is_last:
            raise ValueError("group must be nonempty and F_LAST-terminated")
        exemplar = records[-1]
        identity = {
            (
                record.publisher_id,
                record.instrument_id,
                record.raw_symbol,
                record.continuity_segment_id,
            )
            for record in records
        }
        if len(identity) != 1:
            raise ValueError("an atomic group crossed instrument, symbol, or continuity identity")
        book = self.book_for(exemplar)
        snapshot = any(record.is_snapshot for record in records)
        # Synthetic snapshot rows carry F_BAD_TS_RECV because their exchange
        # timestamps belong to the original orders. Their delivered snapshot
        # receipt timestamp is nevertheless the causal availability boundary.
        if snapshot:
            valid_receipts = [record.ts_recv_ns for record in records]
        else:
            # An atomic event is unavailable until its closing F_LAST record.
            # A bad closing receipt cannot be replaced by an earlier member's
            # clock without exposing a partially delivered event.
            valid_receipts = [] if exemplar.is_bad_ts_recv else [exemplar.ts_recv_ns]
        available = max(valid_receipts) + self.decode_latency_ns if valid_receipts else None
        apply_time = available if available is not None else max(record.ts_event_ns for record in records)
        snapshot_reconciled = snapshot and self._snapshot_matches(book, records)
        if snapshot_reconciled:
            book.bind_completed_group(
                last_ordinal=records[-1].global_record_ordinal,
                available_time_ns=available,
                source_tape_hash=self.source_tape_hash,
                restores_causal_state=available is not None,
            )
            return GroupResult(
                publisher_id=exemplar.publisher_id,
                instrument_id=exemplar.instrument_id,
                raw_symbol=exemplar.raw_symbol,
                continuity_segment_id=exemplar.continuity_segment_id,
                first_ordinal=records[0].global_record_ordinal,
                last_ordinal=records[-1].global_record_ordinal,
                event_time_ns=max(record.ts_event_ns for record in records),
                available_time_ns=available,
                records=len(records),
                mutations=tuple(
                    Mutation(record.action, duplicate_noop=True, detail="reconciled administrative snapshot")
                    for record in records
                ),
                terminations=(),
                executed_quantity=0,
                implied_execution_quantity=0,
                withdrawn_quantity=0,
                aggressor_remainder_quantity=0,
                priority_reset_quantity=0,
                ambiguous_cause=False,
                cause_labels_eligible=True,
                ineligibility_reasons=(),
                snapshot=True,
                snapshot_reconciled=True,
                snapshot_reloaded=False,
            )
        trade_records = [record for record in records if record.action == "T"]
        aggressor_ids = {
            record.order_id
            for record in trade_records
            if record.order_id
        }
        trade_sides = {record.side for record in trade_records}
        valid_aggressor_side = (
            next(iter(trade_sides))
            if len(trade_sides) == 1 and next(iter(trade_sides)) in {"B", "A"}
            else None
        )
        expected_resting_side = (
            "A" if valid_aggressor_side == "B" else "B"
            if valid_aggressor_side == "A"
            else None
        )
        passive_fills: Counter[int] = Counter()
        implied_fills: Counter[int] = Counter()
        for record in records:
            if record.action != "F" or record.order_id in aggressor_ids:
                continue
            if record.side == expected_resting_side:
                passive_fills[record.order_id] += record.size
            else:
                implied_fills[record.order_id] += record.size
        pre_group_live_ids = set(book.orders)
        mutation_ids = {
            record.order_id for record in records if record.action in {"C", "M"}
        }
        unmatched_ids = {
            order_id
            for order_id in passive_fills
            if order_id not in mutation_ids or order_id not in pre_group_live_ids
        }

        # Fill records describe gross execution but do not mutate the displayed
        # book. Account for them before a following M can copy lifecycle totals
        # into a refreshed/native-iceberg priority epoch.
        for order_id, quantity in passive_fills.items():
            live = book.orders.get(order_id)
            if live is not None:
                live.executed_quantity += quantity

        mutations: list[Mutation] = []
        terminations: list[Termination] = []
        executed = sum(passive_fills.values())
        implied_executed = sum(implied_fills.values())
        withdrawn = 0
        aggressor_remainder = 0
        priority_reset = 0
        fill_removal_budget = passive_fills.copy()
        execution_removed_by_order: Counter[int] = Counter()
        withdrawal_removed_by_order: Counter[int] = Counter()
        for record in records:
            pre, pre_rank, pre_ahead = self._pre_state(book, record.order_id)
            old_size = pre.size if pre is not None else 0
            mutation = book.apply(record, apply_time)
            removed = mutation.removed_quantity
            execution_part = 0
            withdrawal_part = 0
            if record.action in {"C", "M"} and removed:
                execution_part = min(fill_removal_budget[record.order_id], removed)
                fill_removal_budget[record.order_id] -= execution_part
                if mutation.priority_reset:
                    priority_reset += removed
                elif record.order_id in aggressor_ids:
                    aggressor_remainder += removed
                else:
                    withdrawal_part = removed - execution_part
                    executed_removal = execution_part
                    withdrawn += withdrawal_part
                    execution_removed_by_order[record.order_id] += executed_removal
                    withdrawal_removed_by_order[record.order_id] += withdrawal_part
                if pre is not None:
                    if not mutation.priority_reset and record.order_id not in aggressor_ids:
                        pre.withdrawn_quantity += removed - execution_part
            mutation = replace(
                mutation,
                side=(mutation.side or (pre.side if pre is not None else record.side)),
                executed_removed_quantity=execution_part,
                withdrawn_removed_quantity=withdrawal_part,
            )
            mutations.append(mutation)
            terminal_old = mutation.terminal and pre is not None
            if terminal_old:
                if mutation.priority_reset:
                    cause = "priority_reset"
                elif record.order_id in aggressor_ids:
                    cause = "aggressor_remainder"
                elif (
                    execution_removed_by_order[record.order_id] > 0
                    and withdrawal_removed_by_order[record.order_id] > 0
                ):
                    cause = "mixed"
                elif execution_removed_by_order[record.order_id] > 0:
                    cause = "execution"
                else:
                    cause = "withdrawal"
                terminations.append(
                    Termination(
                        key=pre.key,
                        time_ns=record.ts_event_ns,
                        available_time_ns=apply_time,
                        cause=cause,
                        quantity=old_size,
                        side=pre.side,
                        price_ticks=pre.price_ticks,
                        age_ns=max(0, apply_time - pre.birth_available_ns),
                        rank=pre_rank,
                        queue_ahead=pre_ahead,
                        left_censored=pre.left_censored,
                    )
                )
        unmatched_fills = sum(passive_fills[order_id] for order_id in unmatched_ids)
        ineligibility_reasons = {
            mutation.anomaly_code
            for mutation in mutations
            if mutation.anomaly_code is not None
        }
        if unmatched_fills:
            ineligibility_reasons.add("passive_fill_without_state_record")
        if implied_executed:
            ineligibility_reasons.add("implied_execution_quantity")
        if available is None:
            ineligibility_reasons.add("no_valid_group_receipt")
        ordered_ineligibility_reasons = tuple(sorted(ineligibility_reasons))
        ambiguous = bool(ordered_ineligibility_reasons)
        if unmatched_fills:
            self.anomalies["passive_fill_without_state_record"] += unmatched_fills
        if implied_executed:
            self.anomalies["implied_execution_quantity"] += implied_executed
        for reason in sorted(
            reason
            for reason in ineligibility_reasons
            if reason
            not in {"passive_fill_without_state_record", "implied_execution_quantity"}
        ):
            self.anomalies[f"ambiguous_group:{reason}"] += 1

        labelled_terminations: list[Termination] = []
        for termination in terminations:
            reasons = set(ordered_ineligibility_reasons)
            if termination.cause == "mixed":
                reasons.add("mixed_execution_withdrawal")
            if termination.left_censored:
                reasons.add("left_censored_life")
            termination_reasons = tuple(sorted(reasons))
            pure_cause_eligible = not termination_reasons
            labelled_terminations.append(
                replace(
                    termination,
                    cause_label_eligible=pure_cause_eligible,
                    tombstone_eligible=(
                        pure_cause_eligible and termination.cause == "withdrawal"
                    ),
                    ineligibility_reasons=termination_reasons,
                )
            )
        book.bind_completed_group(
            last_ordinal=records[-1].global_record_ordinal,
            available_time_ns=available,
            source_tape_hash=self.source_tape_hash,
            restores_causal_state=snapshot and available is not None,
        )
        return GroupResult(
            publisher_id=exemplar.publisher_id,
            instrument_id=exemplar.instrument_id,
            raw_symbol=exemplar.raw_symbol,
            continuity_segment_id=exemplar.continuity_segment_id,
            first_ordinal=records[0].global_record_ordinal,
            last_ordinal=records[-1].global_record_ordinal,
            event_time_ns=max(record.ts_event_ns for record in records),
            available_time_ns=available,
            records=len(records),
            mutations=tuple(mutations),
            terminations=tuple(labelled_terminations),
            executed_quantity=executed,
            implied_execution_quantity=implied_executed,
            withdrawn_quantity=withdrawn,
            aggressor_remainder_quantity=aggressor_remainder,
            priority_reset_quantity=priority_reset,
            ambiguous_cause=ambiguous,
            cause_labels_eligible=not ambiguous,
            ineligibility_reasons=ordered_ineligibility_reasons,
            snapshot=snapshot,
            snapshot_reconciled=False,
            snapshot_reloaded=snapshot,
        )

    def push(self, record: EventRecord) -> GroupResult | None:
        group = self.group_builder.push(record)
        return self.apply_group(group) if group is not None else None

    def replay(self, records: Iterable[EventRecord]) -> Iterable[GroupResult]:
        for record in records:
            result = self.push(record)
            if result is not None:
                yield result
