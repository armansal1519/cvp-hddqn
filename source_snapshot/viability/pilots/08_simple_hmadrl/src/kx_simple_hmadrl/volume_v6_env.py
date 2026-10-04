"""Causal, inventory-aware economic environment for the two-level V6 model.

Only the lower agent touches this ledger.  A target action is a desired
one-contract position, not an order opcode.  Direct reversals are absent from
the legal mask and are rejected rather than translated into a flatten.  Legal
changes route as deterministic aggressive orders against the materialized
five-millisecond arrival BBO.  When that BBO is unavailable, the order remains
eligible for the first executable arrival in the same continuity segment for
at most five seconds; otherwise it is explicitly rejected.

Rewards are the exact change in integer-valued, executable net-liquidation
equity.  Open positions are marked on the liquidation side of the visible BBO
and reserve the future exit commission.  Consequently the undiscounted raw
rewards telescope cent-for-cent to terminal cash after the mandatory final
flatten.  No activity, Sharpe, drawdown, waiting, or other shaped reward is
present in this module.

The upper agent has no access to the real ledger.  ``BiasRewardLabeler``
provides its separate training-only target: a hypothetical after-cost round
trip over an exact, non-overlapping fifteen-minute lifetime using the same
route semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Mapping

import numpy as np

from .environment import EconomicLedger, ExecutionEconomics, Fill, TopOfBook
from .volume_v6_contract import ACTIONS, STATE_FIELDS, STATE_WIDTH


NANOSECONDS_PER_MILLISECOND = 1_000_000
NANOSECONDS_PER_MINUTE = 60 * 1_000_000_000
TOP_REWARD_LIFETIME_NS = 15 * NANOSECONDS_PER_MINUTE


class TargetPosition(IntEnum):
    """Registered semantic action values shared by both V6 agents."""

    SHORT = -1
    FLAT = 0
    LONG = 1


AGENT_STATE_FIELDS = STATE_FIELDS
AGENT_STATE_WIDTH = STATE_WIDTH


def position_for_action(action: int | TargetPosition) -> int:
    """Map the registered three-way Q action to ``{-1, 0, +1}``."""

    return int(TargetPosition(int(action)))


def action_for_position(position: int) -> TargetPosition:
    if int(position) not in ACTIONS:
        raise ValueError("position must be -1, 0, or +1")
    return TargetPosition(int(position))


def action_index(action: int | TargetPosition) -> int:
    """Map a semantic action value to its SHORT/FLAT/LONG Q-column index."""

    return int(TargetPosition(int(action))) + 1


def legal_target_mask(inventory: int) -> np.ndarray:
    """Return legal desired positions without reversal aliasing.

    An open position may be maintained or flattened.  Entering the opposite
    side requires a later decision after a successful flatten.
    """

    if int(inventory) not in ACTIONS:
        raise ValueError("inventory must be -1, 0, or +1")
    if int(inventory) < 0:
        return np.asarray((True, True, False), dtype=bool)
    if int(inventory) > 0:
        return np.asarray((False, True, True), dtype=bool)
    return np.ones(3, dtype=bool)


@dataclass(frozen=True)
class RouteResolution:
    """Observable outcome of one deterministic aggressive route."""

    decision_index: int
    decision_time_ns: int
    deadline_time_ns: int
    side: int
    fill_index: int | None
    searched_snapshots: int
    rejected: bool
    book: TopOfBook | None = None
    fill: Fill | None = None


@dataclass(frozen=True)
class V6Transition:
    """One lower-agent transition from a decision to its next valuation."""

    index: int
    next_index: int
    decision_time_ns: int
    next_time_ns: int
    action: int
    target_position: int
    inventory_before: int
    inventory_after: int
    current_state: tuple[float, ...]
    next_state: tuple[float, ...]
    current_legal_mask: tuple[bool, bool, bool]
    next_legal_mask: tuple[bool, bool, bool]
    reward_cents: int
    reward_ticks: float
    command_created: bool
    command_rejected: bool
    route: RouteResolution | None
    terminal_route: RouteResolution | None
    done: bool


@dataclass(frozen=True)
class V6SessionResult:
    session_date: str
    net_pnl_cents: int
    raw_rewards_cents: tuple[int, ...]
    completed_round_trips: int
    rejected_commands: int
    fills: tuple[Fill, ...]
    transitions: tuple[V6Transition, ...]
    terminal_reward_cents: int
    unsafe_legacy_arrival_boundary: bool = False

    @property
    def net_pnl_ticks(self) -> float:
        return self.net_pnl_cents / 1_000.0


@dataclass(frozen=True)
class BiasRewardInterval:
    """Three upper-agent labels in SHORT/FLAT/LONG action order."""

    decision_index: int
    expiry_index: int
    decision_time_ns: int
    expiry_time_ns: int
    rewards_cents: tuple[int, int, int]
    valid_mask: tuple[bool, bool, bool]
    entry_routes: tuple[RouteResolution | None, RouteResolution | None, RouteResolution | None]
    exit_routes: tuple[RouteResolution | None, RouteResolution | None, RouteResolution | None]
    unsafe_legacy_arrival_boundary: bool = False

    @property
    def rewards_ticks(self) -> tuple[float, float, float]:
        return tuple(value / 1_000.0 for value in self.rewards_cents)

    def is_valid(self, action: int | TargetPosition) -> bool:
        return self.valid_mask[action_index(action)]

    def reward_cents_for(self, action: int | TargetPosition) -> int:
        if not self.is_valid(action):
            raise ValueError("requested upper reward is invalid and must be skipped")
        return self.rewards_cents[action_index(action)]

    def reward_ticks_for(self, action: int | TargetPosition) -> float:
        return self.reward_cents_for(action) / 1_000.0


_BASE_KEYS = {
    "clock_ns",
    "continuity_segment",
    "valid",
    "best_bid_ticks",
    "best_ask_ticks",
    "best_bid_size",
    "best_ask_size",
}


def _arrival_keys(latency_ms: int) -> set[str]:
    prefix = f"arrival_{int(latency_ms)}ms_"
    return {
        prefix + "clock_ns",
        prefix + "state_available_ns",
        prefix + "continuity_segment",
        prefix + "valid",
        prefix + "bid_ticks",
        prefix + "ask_ticks",
        prefix + "bid_size",
        prefix + "ask_size",
    }


def uses_legacy_arrival_boundary(
    raw: Mapping[str, np.ndarray], economics: ExecutionEconomics,
) -> bool:
    prefix = f"arrival_{economics.route_latency_ms}ms_"
    return (
        prefix + "state_available_ns" not in raw
        or prefix + "continuity_segment" not in raw
    )


def _validate_raw(
    raw: Mapping[str, np.ndarray], economics: ExecutionEconomics, *,
    allow_legacy_arrival_segment: bool = False,
) -> int:
    if economics.route_latency_ms not in (5, 20):
        raise ValueError("GC shards materialize only 5 ms and 20 ms arrival books")
    arrival_keys = _arrival_keys(economics.route_latency_ms)
    boundary_keys = {
        f"arrival_{economics.route_latency_ms}ms_state_available_ns",
        f"arrival_{economics.route_latency_ms}ms_continuity_segment",
    }
    required = _BASE_KEYS | (
        arrival_keys - boundary_keys if allow_legacy_arrival_segment else arrival_keys
    )
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"V6 environment shard is missing {sorted(missing)}")
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    if clock.ndim != 1 or len(clock) < 2 or (np.diff(clock) <= 0).any():
        raise ValueError("visible clocks must be a strictly increasing vector")
    length = len(clock)
    for key in required - {"clock_ns"}:
        if np.asarray(raw[key]).shape != (length,):
            raise ValueError(f"{key} must align one-to-one with clock_ns")
    prefix = f"arrival_{economics.route_latency_ms}ms_"
    arrival_clock = np.asarray(raw[prefix + "clock_ns"], dtype=np.int64)
    legacy = uses_legacy_arrival_boundary(raw, economics)
    if legacy and not allow_legacy_arrival_segment:
        raise ValueError("arrival book lacks materialized availability/continuity evidence")
    arrival_available = np.asarray(
        raw.get(prefix + "state_available_ns", arrival_clock), dtype=np.int64,
    )
    arrival_segment = np.asarray(
        raw.get(prefix + "continuity_segment", raw["continuity_segment"]), dtype=np.int64,
    )
    if arrival_available.shape != (length,) or arrival_segment.shape != (length,):
        raise ValueError("arrival availability/continuity must align one-to-one with clock_ns")
    expected = clock + economics.route_latency_ms * NANOSECONDS_PER_MILLISECOND
    if not np.array_equal(arrival_clock, expected):
        raise ValueError("arrival clocks must equal visible clocks plus registered route latency")
    valid = np.asarray(raw["valid"], dtype=bool)
    if (np.asarray(raw["best_bid_ticks"])[valid] >
            np.asarray(raw["best_ask_ticks"])[valid]).any():
        raise ValueError("visible shard contains a crossed BBO")
    arrival_valid = np.asarray(raw[prefix + "valid"], dtype=bool)
    if (
        bool((arrival_available[arrival_valid] < 0).any())
        or bool((arrival_available[arrival_valid] > arrival_clock[arrival_valid]).any())
    ):
        raise ValueError("arrival state was unavailable at its materialized route clock")
    if bool((arrival_segment[arrival_valid] < 0).any()):
        raise ValueError("valid arrival book lacks a continuity segment")
    if (np.asarray(raw[prefix + "bid_ticks"])[arrival_valid] >
            np.asarray(raw[prefix + "ask_ticks"])[arrival_valid]).any():
        raise ValueError("arrival shard contains a crossed BBO")
    if (np.asarray(raw["best_bid_size"]) < 0).any() or (
            np.asarray(raw["best_ask_size"]) < 0).any():
        raise ValueError("visible BBO contains negative displayed size")
    if (np.asarray(raw[prefix + "bid_size"]) < 0).any() or (
            np.asarray(raw[prefix + "ask_size"]) < 0).any():
        raise ValueError("arrival BBO contains negative displayed size")
    return length


def _visible_book(raw: Mapping[str, np.ndarray], index: int) -> TopOfBook:
    return TopOfBook(
        timestamp_ns=int(raw["clock_ns"][index]),
        bid_ticks=int(raw["best_bid_ticks"][index]),
        ask_ticks=int(raw["best_ask_ticks"][index]),
        bid_size=int(raw["best_bid_size"][index]),
        ask_size=int(raw["best_ask_size"][index]),
        valid=bool(raw["valid"][index]),
    )


def _arrival_book(
    raw: Mapping[str, np.ndarray], index: int, latency_ms: int,
) -> TopOfBook:
    prefix = f"arrival_{int(latency_ms)}ms_"
    return TopOfBook(
        timestamp_ns=int(raw[prefix + "clock_ns"][index]),
        bid_ticks=int(raw[prefix + "bid_ticks"][index]),
        ask_ticks=int(raw[prefix + "ask_ticks"][index]),
        bid_size=int(raw[prefix + "bid_size"][index]),
        ask_size=int(raw[prefix + "ask_size"][index]),
        valid=bool(raw[prefix + "valid"][index]),
    )


def _resolve_route(
    raw: Mapping[str, np.ndarray], index: int, side: int,
    economics: ExecutionEconomics, *, allow_legacy_arrival_segment: bool = False,
) -> RouteResolution:
    if side not in (-1, 1):
        raise ValueError("route side must be -1 or +1")
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    segment = np.asarray(raw["continuity_segment"])
    decision_time = int(clock[index])
    deadline = decision_time + economics.deadline_ms * NANOSECONDS_PER_MILLISECOND
    prefix = f"arrival_{economics.route_latency_ms}ms_"
    arrivals = np.asarray(raw[prefix + "clock_ns"], dtype=np.int64)
    if uses_legacy_arrival_boundary(raw, economics) and not allow_legacy_arrival_segment:
        raise ValueError("arrival book lacks materialized availability/continuity evidence")
    arrival_available = np.asarray(
        raw.get(prefix + "state_available_ns", arrivals), dtype=np.int64,
    )
    arrival_segment = np.asarray(
        raw.get(prefix + "continuity_segment", raw["continuity_segment"]), dtype=np.int64,
    )
    arrival_valid = np.asarray(raw[prefix + "valid"], dtype=bool)
    attempted = 0
    origin_segment = int(segment[index])
    for candidate in range(int(index), len(clock)):
        if int(segment[candidate]) != origin_segment:
            break
        if int(arrivals[candidate]) > deadline:
            break
        attempted += 1
        # A reset between the visible decision and its shifted route clock is a
        # hard boundary, not an executable opportunity in the old contract.
        if (
            bool(arrival_valid[candidate])
            and (
                int(arrival_segment[candidate]) != origin_segment
                or int(arrival_available[candidate]) > int(arrivals[candidate])
            )
        ):
            break
        book = _arrival_book(raw, candidate, economics.route_latency_ms)
        displayed = book.ask_size if side > 0 else book.bid_size
        if book.valid and displayed >= 1:
            return RouteResolution(
                int(index), decision_time, deadline, int(side), int(candidate), attempted,
                False, book, None,
            )
    return RouteResolution(
        int(index), decision_time, deadline, int(side), None, attempted, True, None, None,
    )


def _tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values, dtype=np.float32))


def _mask_tuple(values: np.ndarray) -> tuple[bool, bool, bool]:
    result = tuple(bool(value) for value in np.asarray(values, dtype=bool))
    if len(result) != 3:
        raise AssertionError("target mask width changed")
    return result  # type: ignore[return-value]


@dataclass
class V6TradingEnvironment:
    """Exact one-contract lower-agent environment.

    ``step(index, action, next_index)`` follows the conventional transition
    direction: it observes and acts at ``index``, then assigns the change in
    executable NLV measured at ``next_index`` to that action.  Consecutive
    calls must therefore use the previous ``next_index`` as their new
    ``index``.  Set ``terminal=True`` only when ``next_index`` is the final
    (12:25 CT in a canonical shard) clock.
    """

    raw: Mapping[str, np.ndarray]
    session_date: str
    economics: ExecutionEconomics = field(default_factory=ExecutionEconomics.from_basepaper)
    allow_legacy_arrival_segment: bool = False
    ledger: EconomicLedger = field(init=False)
    current_target: int = field(init=False, default=0)
    fills: list[Fill] = field(init=False, default_factory=list)
    rewards_cents: list[int] = field(init=False, default_factory=list)
    transitions: list[V6Transition] = field(init=False, default_factory=list)
    rejected_commands: int = field(init=False, default=0)
    holding_since_ns: int | None = field(init=False, default=None)
    entry_price_ticks: int | None = field(init=False, default=None)
    previous_command_rejected: bool = field(init=False, default=False)
    _length: int = field(init=False)
    _equity_cents: int = field(init=False, default=0)
    _ready_index: int | None = field(init=False, default=None)
    _finished: bool = field(init=False, default=False)
    _terminal_reward_cents: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._length = _validate_raw(
            self.raw, self.economics,
            allow_legacy_arrival_segment=self.allow_legacy_arrival_segment,
        )
        if self.economics.maximum_inventory != 1:
            raise ValueError("V6 is registered for exactly one contract")
        self.ledger = EconomicLedger(self.economics)

    @property
    def inventory(self) -> int:
        return int(self.ledger.inventory)

    def legal_actions(self) -> np.ndarray:
        return legal_target_mask(self.inventory)

    def _check_index(self, index: int) -> int:
        result = int(index)
        if result < 0 or result >= self._length:
            raise IndexError("environment index is outside the session shard")
        return result

    def _valuation_book(self, index: int) -> TopOfBook:
        book = _visible_book(self.raw, self._check_index(index))
        if not book.valid:
            raise ValueError("an economic decision/valuation requires a valid visible BBO")
        displayed = book.bid_size if self.inventory > 0 else book.ask_size
        if self.inventory and displayed < 1:
            raise ValueError("open inventory has no executable visible liquidation quote")
        return book

    def agent_state(self, index: int) -> np.ndarray:
        """Return only causal decision-time ledger and visible-book fields."""

        index = self._check_index(index)
        book = _visible_book(self.raw, index)
        if not book.valid:
            raise ValueError("agent state requires a valid visible BBO")
        inventory_hot = np.zeros(3, dtype=np.float32)
        inventory_hot[action_index(action_for_position(self.inventory))] = 1.0
        target_hot = np.zeros(3, dtype=np.float32)
        target_hot[action_index(action_for_position(self.current_target))] = 1.0
        midpoint = 0.5 * (book.bid_ticks + book.ask_ticks)
        liquidation = 0.0
        entry_distance = 0.0
        holding_age = 0.0
        if self.inventory and self.entry_price_ticks is not None:
            exit_ticks = book.bid_ticks if self.inventory > 0 else book.ask_ticks
            gross = self.inventory * (exit_ticks - self.entry_price_ticks)
            liquidation = (
                gross
                - 2.0 * self.economics.fee_cents_per_side
                / self.economics.tick_value_cents
            )
            entry_distance = self.inventory * (midpoint - self.entry_price_ticks)
            if self.holding_since_ns is not None:
                holding_age = max(
                    0.0,
                    (int(self.raw["clock_ns"][index]) - self.holding_since_ns)
                    / (60.0 * NANOSECONDS_PER_MINUTE),
                )
        start = int(self.raw["clock_ns"][0])
        end = int(self.raw["clock_ns"][-1])
        remaining = float(np.clip(
            (end - int(self.raw["clock_ns"][index])) / max(1, end - start), 0.0, 1.0,
        ))
        state = np.concatenate((
            inventory_hot,
            target_hot,
            np.asarray((
                liquidation,
                entry_distance,
                holding_age,
                remaining,
                float(book.ask_ticks - book.bid_ticks),
                2.0 * self.economics.fee_cents_per_side
                / self.economics.tick_value_cents,
                float(self.previous_command_rejected),
            ), dtype=np.float32),
        )).astype(np.float32, copy=False)
        if state.shape != (AGENT_STATE_WIDTH,) or not np.isfinite(state).all():
            raise AssertionError("V6 agent-state contract changed or became non-finite")
        return state

    def _execute_target(self, index: int, target: int, *, forced: bool = False) -> RouteResolution | None:
        if target == self.inventory:
            return None
        delta = int(target) - self.inventory
        if delta not in (-1, 1):
            raise AssertionError("a legal V6 target change must be exactly one contract")
        route = _resolve_route(
            self.raw, index, delta, self.economics,
            allow_legacy_arrival_segment=self.allow_legacy_arrival_segment,
        )
        if route.rejected or route.book is None:
            self.rejected_commands += 1
            self.previous_command_rejected = True
            self.current_target = self.inventory
            return route
        before = self.inventory
        fill = self.ledger.fill(delta, route.book, forced=forced)
        self.fills.append(fill)
        route = replace(route, fill=fill)
        self.previous_command_rejected = False
        self.current_target = self.inventory
        if before == 0 and self.inventory != 0:
            self.entry_price_ticks = int(fill.price_ticks)
            self.holding_since_ns = int(fill.timestamp_ns)
        elif self.inventory == 0:
            self.entry_price_ticks = None
            self.holding_since_ns = None
        return route

    def step(
        self,
        index: int,
        action: int | TargetPosition,
        next_index: int,
        *,
        terminal: bool = False,
    ) -> V6Transition:
        if self._finished:
            raise RuntimeError("session environment is already finished")
        index, next_index = self._check_index(index), self._check_index(next_index)
        if next_index <= index:
            raise ValueError("next valuation must follow the current decision")
        if self._ready_index is not None and index != self._ready_index:
            raise ValueError("consecutive transitions must share their boundary index")
        if terminal and next_index != self._length - 1:
            raise ValueError("terminal transition must end on the shard's 12:25 boundary")
        segment = np.asarray(self.raw["continuity_segment"])
        if int(segment[index]) != int(segment[next_index]):
            raise ValueError("a V6 transition may not cross a continuity boundary")

        selected = TargetPosition(int(action))
        current_mask = self.legal_actions()
        if not bool(current_mask[action_index(selected)]):
            raise ValueError("illegal target action; direct reversals are never reinterpreted")
        target = position_for_action(selected)
        current_state = self.agent_state(index)
        before = self.ledger.value_cents(self._valuation_book(index))
        if self._ready_index is not None and before != self._equity_cents:
            raise AssertionError("transition boundary NLV failed exact reconciliation")
        inventory_before = self.inventory
        route = self._execute_target(index, target)
        if route is not None and route.fill is not None:
            if route.fill.timestamp_ns > int(self.raw["clock_ns"][next_index]):
                raise ValueError("next decision precedes completion of the deterministic route")

        terminal_route: RouteResolution | None = None
        if terminal:
            if self.inventory:
                terminal_route = self._execute_target(next_index, 0, forced=True)
                if terminal_route is None or terminal_route.rejected:
                    raise RuntimeError("mandatory 12:25 liquidation has no executable arrival BBO")
            after = int(self.ledger.cash_cents)
            self._finished = True
        else:
            after = self.ledger.value_cents(self._valuation_book(next_index))
        reward = int(after - before)
        self.rewards_cents.append(reward)
        self._equity_cents = after
        self._ready_index = next_index
        if terminal:
            self._terminal_reward_cents = reward
        next_state = self.agent_state(next_index)
        next_mask = self.legal_actions()
        transition = V6Transition(
            index=index,
            next_index=next_index,
            decision_time_ns=int(self.raw["clock_ns"][index]),
            next_time_ns=int(self.raw["clock_ns"][next_index]),
            action=int(selected),
            target_position=target,
            inventory_before=inventory_before,
            inventory_after=self.inventory,
            current_state=_tuple(current_state),
            next_state=_tuple(next_state),
            current_legal_mask=_mask_tuple(current_mask),
            next_legal_mask=_mask_tuple(next_mask),
            reward_cents=reward,
            reward_ticks=reward / 1_000.0,
            command_created=route is not None,
            command_rejected=bool(route is not None and route.rejected),
            route=route,
            terminal_route=terminal_route,
            done=bool(terminal),
        )
        self.transitions.append(transition)
        return transition

    def finish(self) -> V6SessionResult:
        """Force-flat at the shard end if needed, reconcile, and report.

        The canonical data contract makes the final shard clock 12:25 CT.  A
        direct call is useful when a collector does not encode the terminal
        boundary as its last transition; using ``terminal=True`` in ``step``
        instead attaches this same reward to the final action.
        """

        if not self._finished:
            terminal_index = self._length - 1
            terminal_route: RouteResolution | None = None
            if self.inventory:
                terminal_route = self._execute_target(terminal_index, 0, forced=True)
                if terminal_route is None or terminal_route.rejected:
                    raise RuntimeError("mandatory 12:25 liquidation has no executable arrival BBO")
            reward = int(self.ledger.cash_cents - self._equity_cents)
            self.rewards_cents.append(reward)
            self._equity_cents = int(self.ledger.cash_cents)
            self._ready_index = terminal_index
            self._terminal_reward_cents = reward
            self._finished = True
        if self.inventory != 0:
            raise AssertionError("finished V6 session retained inventory")
        net = int(self.ledger.cash_cents)
        if sum(self.rewards_cents) != net:
            raise AssertionError(
                f"V6 reward/ledger mismatch: {sum(self.rewards_cents)} != {net}"
            )
        if len(self.fills) % 2:
            raise AssertionError("finished one-contract session has an unmatched fill")
        return V6SessionResult(
            session_date=str(self.session_date),
            net_pnl_cents=net,
            raw_rewards_cents=tuple(int(value) for value in self.rewards_cents),
            completed_round_trips=len(self.fills) // 2,
            rejected_commands=int(self.rejected_commands),
            fills=tuple(self.fills),
            transitions=tuple(self.transitions),
            terminal_reward_cents=int(self._terminal_reward_cents),
            unsafe_legacy_arrival_boundary=uses_legacy_arrival_boundary(
                self.raw, self.economics,
            ),
        )


@dataclass
class BiasRewardLabeler:
    """Generate non-overlapping exact fifteen-minute upper-agent rewards."""

    raw: Mapping[str, np.ndarray]
    economics: ExecutionEconomics = field(default_factory=ExecutionEconomics.from_basepaper)
    allow_legacy_arrival_segment: bool = False
    _last_expiry_ns: int | None = field(init=False, default=None)
    _length: int = field(init=False)

    def __post_init__(self) -> None:
        self._length = _validate_raw(
            self.raw, self.economics,
            allow_legacy_arrival_segment=self.allow_legacy_arrival_segment,
        )

    def label(self, decision_index: int, expiry_index: int) -> BiasRewardInterval:
        decision_index, expiry_index = int(decision_index), int(expiry_index)
        if not (0 <= decision_index < expiry_index < self._length):
            raise IndexError("bias reward interval indexes are invalid")
        clock = np.asarray(self.raw["clock_ns"], dtype=np.int64)
        decision_time = int(clock[decision_index])
        expiry_time = int(clock[expiry_index])
        if expiry_time - decision_time != TOP_REWARD_LIFETIME_NS:
            raise ValueError("upper reward lifetime must be exactly fifteen minutes")
        if self._last_expiry_ns is not None and decision_time < self._last_expiry_ns:
            raise ValueError("upper reward intervals must not overlap")
        segment = np.asarray(self.raw["continuity_segment"])
        if int(segment[decision_index]) != int(segment[expiry_index]):
            raise ValueError("upper reward interval may not cross a continuity boundary")

        rewards = [0, 0, 0]
        valid = [False, True, False]
        entries: list[RouteResolution | None] = [None, None, None]
        exits: list[RouteResolution | None] = [None, None, None]
        for selected in (TargetPosition.SHORT, TargetPosition.LONG):
            position = position_for_action(selected)
            selected_index = action_index(selected)
            ledger = EconomicLedger(self.economics)
            entry = _resolve_route(
                self.raw, decision_index, position, self.economics,
                allow_legacy_arrival_segment=self.allow_legacy_arrival_segment,
            )
            entries[selected_index] = entry
            if entry.rejected or entry.book is None:
                continue
            entry_fill = ledger.fill(position, entry.book)
            entry = replace(entry, fill=entry_fill)
            entries[selected_index] = entry
            exit_route = _resolve_route(
                self.raw, expiry_index, -position, self.economics,
                allow_legacy_arrival_segment=self.allow_legacy_arrival_segment,
            )
            exits[selected_index] = exit_route
            if exit_route.rejected or exit_route.book is None:
                continue
            exit_fill = ledger.fill(-position, exit_route.book)
            exits[selected_index] = replace(exit_route, fill=exit_fill)
            if ledger.inventory != 0:
                raise AssertionError("hypothetical upper round trip retained inventory")
            rewards[selected_index] = int(ledger.cash_cents)
            valid[selected_index] = True

        self._last_expiry_ns = expiry_time
        return BiasRewardInterval(
            decision_index=decision_index,
            expiry_index=expiry_index,
            decision_time_ns=decision_time,
            expiry_time_ns=expiry_time,
            rewards_cents=tuple(rewards),  # type: ignore[arg-type]
            valid_mask=tuple(valid),  # type: ignore[arg-type]
            entry_routes=tuple(entries),  # type: ignore[arg-type]
            exit_routes=tuple(exits),  # type: ignore[arg-type]
            unsafe_legacy_arrival_boundary=uses_legacy_arrival_boundary(
                self.raw, self.economics,
            ),
        )


__all__ = [
    "ACTIONS",
    "AGENT_STATE_FIELDS",
    "AGENT_STATE_WIDTH",
    "BiasRewardInterval",
    "BiasRewardLabeler",
    "RouteResolution",
    "TargetPosition",
    "TOP_REWARD_LIFETIME_NS",
    "V6SessionResult",
    "V6TradingEnvironment",
    "V6Transition",
    "action_index",
    "action_for_position",
    "legal_target_mask",
    "position_for_action",
    "uses_legacy_arrival_boundary",
]
