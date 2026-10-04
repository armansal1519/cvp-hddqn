"""Causal, integer-valued trading economics shared by every HMADRL agent.

Prices are represented in GC ticks and cash in cents.  Keeping the accounting
integer-valued makes the training reward, replay ledger, and acceptance report
agree exactly instead of accumulating floating-point drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Side = Literal[-1, 1]


@dataclass(frozen=True)
class ExecutionEconomics:
    """Execution assumptions bound to the registered BasePaper contract."""

    tick_size: float = 0.1
    tick_value_cents: int = 1_000
    fee_cents_per_side: int = 300
    route_latency_ms: int = 5
    maximum_inventory: int = 1
    deadline_ms: int = 5_000
    adverse_ticks: int = 0

    @classmethod
    def from_basepaper(
        cls,
        *,
        fee_multiplier: int = 1,
        route_latency_ms: int | None = None,
        adverse_ticks: int = 0,
    ) -> "ExecutionEconomics":
        """Load costs from BasePaper rather than maintaining a second default."""

        from kx_base_gc.contract import registered_contract

        execution = registered_contract("base_paper_gc").execution
        return cls(
            # BasePaper registers GC tick value but its bar contract does not
            # duplicate the instrument's exchange tick size.
            tick_size=0.1,
            tick_value_cents=int(round(execution.tick_value_usd * 100)),
            fee_cents_per_side=int(
                round(execution.fee_usd_per_filled_side * 100 * fee_multiplier)
            ),
            route_latency_ms=(
                int(execution.route_latency_ms)
                if route_latency_ms is None
                else int(route_latency_ms)
            ),
            maximum_inventory=int(execution.quantity),
            deadline_ms=5_000,
            adverse_ticks=int(adverse_ticks),
        )

    def price_to_ticks(self, price: float) -> int:
        return int(round(float(price) / self.tick_size))

    def ticks_to_price(self, ticks: int) -> float:
        return int(ticks) * self.tick_size


@dataclass(frozen=True)
class TopOfBook:
    timestamp_ns: int
    bid_ticks: int
    ask_ticks: int
    bid_size: int = 1
    ask_size: int = 1
    valid: bool = True

    def __post_init__(self) -> None:
        if self.valid and self.bid_ticks > self.ask_ticks:
            raise ValueError("crossed top of book")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("negative displayed size")


@dataclass(frozen=True)
class Fill:
    timestamp_ns: int
    side: Side
    price_ticks: int
    fee_cents: int
    forced: bool = False


@dataclass
class EconomicLedger:
    """One-contract cash/inventory ledger with conservative liquidation marks."""

    economics: ExecutionEconomics
    cash_cents: int = 0
    inventory: int = 0

    def value_cents(self, book: TopOfBook, *, reserve_exit_fee: bool = True) -> int:
        if not book.valid:
            raise ValueError("cannot mark against an invalid book")
        if self.inventory > 0:
            marked = self.inventory * book.bid_ticks * self.economics.tick_value_cents
        elif self.inventory < 0:
            marked = self.inventory * book.ask_ticks * self.economics.tick_value_cents
        else:
            marked = 0
        exit_fee = (
            abs(self.inventory) * self.economics.fee_cents_per_side
            if reserve_exit_fee
            else 0
        )
        return int(self.cash_cents + marked - exit_fee)

    def fill(self, side: Side, book: TopOfBook, *, forced: bool = False) -> Fill:
        if side not in (-1, 1):
            raise ValueError("side must be -1 or +1")
        if not book.valid:
            raise ValueError("cannot fill against an invalid book")
        next_inventory = self.inventory + int(side)
        if abs(next_inventory) > self.economics.maximum_inventory:
            raise ValueError("fill would violate the inventory limit")
        displayed = book.ask_size if side > 0 else book.bid_size
        if displayed < 1:
            raise ValueError("insufficient displayed top-level size")
        adverse = self.economics.adverse_ticks * int(side)
        price_ticks = (book.ask_ticks if side > 0 else book.bid_ticks) + adverse
        notional_cents = price_ticks * self.economics.tick_value_cents
        self.cash_cents -= int(side) * notional_cents
        self.cash_cents -= self.economics.fee_cents_per_side
        self.inventory = next_inventory
        return Fill(
            timestamp_ns=book.timestamp_ns,
            side=side,
            price_ticks=price_ticks,
            fee_cents=self.economics.fee_cents_per_side,
            forced=forced,
        )

    def move_to_target(
        self, target: int, book: TopOfBook, *, forced: bool = False
    ) -> tuple[Fill, ...]:
        if target not in (-1, 0, 1):
            raise ValueError("target must be -1, 0, or +1")
        if abs(target) > self.economics.maximum_inventory:
            raise ValueError("target violates the inventory limit")
        fills: list[Fill] = []
        while self.inventory != target:
            side: Side = 1 if self.inventory < target else -1
            fills.append(self.fill(side, book, forced=forced))
        return tuple(fills)


def economic_reward_cents(
    ledger: EconomicLedger,
    before_book: TopOfBook,
    after_book: TopOfBook,
    target: int | None = None,
    *,
    forced: bool = False,
) -> tuple[int, tuple[Fill, ...]]:
    """Apply an optional target change and return the exact change in value."""

    before = ledger.value_cents(before_book)
    fills = () if target is None else ledger.move_to_target(target, before_book, forced=forced)
    after = ledger.value_cents(after_book)
    return after - before, fills


def ppo_reward(economic_cents: int, step: int, gamma: float = 0.99) -> float:
    """Compensate PPO discounting so its return equals undiscounted GC ticks."""

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    return (float(economic_cents) / 1_000.0) / (gamma**int(step))


def discounted_economic_ticks(
    rewards: np.ndarray | list[float], gamma: float = 0.99
) -> float:
    values = np.asarray(rewards, dtype=np.float64)
    return float(np.sum(values * np.power(gamma, np.arange(values.size))))


def execution_savings_ticks(
    *,
    side: Side,
    immediate_price_ticks: int,
    actual_price_ticks: int,
    tick_value_cents: int = 1_000,
) -> int:
    """Paired price improvement versus forced-immediate execution in GC ticks."""

    cents = int(side) * (int(immediate_price_ticks) - int(actual_price_ticks))
    cents *= int(tick_value_cents)
    return int(cents // tick_value_cents)


@dataclass
class ExecutionEpisode:
    """WAIT/CROSS episode for a fixed, immutable target delta."""

    side: Side
    immediate_book: TopOfBook
    economics: ExecutionEconomics
    elapsed_ms: int = 0
    done: bool = False

    def legal_actions(self) -> tuple[bool, bool]:
        # WAIT becomes illegal exactly at the five-second deadline.
        return (not self.done and self.elapsed_ms < self.economics.deadline_ms, not self.done)

    def step(
        self, action: Literal[0, 1], book: TopOfBook, *, advance_ms: int = 250
    ) -> tuple[int, Fill | None, bool]:
        if self.done:
            raise RuntimeError("episode already completed")
        wait_legal, cross_legal = self.legal_actions()
        if action == 0 and wait_legal:
            self.elapsed_ms += int(advance_ms)
            return 0, None, False
        if action not in (0, 1) or not cross_legal:
            raise ValueError("illegal execution action")
        forced = self.elapsed_ms >= self.economics.deadline_ms
        actual = book.ask_ticks if self.side > 0 else book.bid_ticks
        immediate = (
            self.immediate_book.ask_ticks
            if self.side > 0
            else self.immediate_book.bid_ticks
        )
        actual += self.economics.adverse_ticks * int(self.side)
        reward_ticks = execution_savings_ticks(
            side=self.side,
            immediate_price_ticks=immediate,
            actual_price_ticks=actual,
            tick_value_cents=self.economics.tick_value_cents,
        )
        self.done = True
        return reward_ticks, Fill(
            timestamp_ns=book.timestamp_ns,
            side=self.side,
            price_ticks=actual,
            fee_cents=self.economics.fee_cents_per_side,
            forced=forced,
        ), True


def top_of_book_from_shard(
    data: object,
    index: int,
    *,
    latency_ms: int = 0,
    require_arrival_alignment: bool = True,
) -> TopOfBook:
    """Read a BBO without silently treating a decision book as an arrival book."""

    prefix = f"arrival_{int(latency_ms)}ms_" if latency_ms else "best_"
    files = set(getattr(data, "files"))
    if latency_ms:
        required = {
            prefix + "clock_ns", prefix + "valid", prefix + "bid_ticks",
            prefix + "ask_ticks", prefix + "bid_size", prefix + "ask_size",
        }
        if not required.issubset(files):
            if require_arrival_alignment:
                raise ValueError(
                    f"dataset lacks causal {latency_ms} ms arrival snapshots; rebuild with schema v2"
                )
            latency_ms = 0
            prefix = "best_"
    if latency_ms:
        return TopOfBook(
            int(data[prefix + "clock_ns"][index]),
            int(data[prefix + "bid_ticks"][index]),
            int(data[prefix + "ask_ticks"][index]),
            int(data[prefix + "bid_size"][index]),
            int(data[prefix + "ask_size"][index]),
            bool(data[prefix + "valid"][index]),
        )
    if {"best_bid_ticks", "best_ask_ticks", "best_bid_size", "best_ask_size"}.issubset(files):
        bid_ticks = int(data["best_bid_ticks"][index])
        ask_ticks = int(data["best_ask_ticks"][index])
        bid_size = int(data["best_bid_size"][index])
        ask_size = int(data["best_ask_size"][index])
    else:
        # Explicit compatibility path for forensic v1 replay only.
        bid_ticks = int(round(float(data["best_bid"][index]) / 0.1))
        ask_ticks = int(round(float(data["best_ask"][index]) / 0.1))
        bid_size = ask_size = 1
    return TopOfBook(
        int(data["clock_ns"][index]), bid_ticks, ask_ticks, bid_size, ask_size,
        bool(data["valid"][index]),
    )
