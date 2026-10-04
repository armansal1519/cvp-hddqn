from __future__ import annotations

from dataclasses import dataclass, field

from .book import FIFOBook
from .types import Fill


@dataclass
class SelfFinancingLedger:
    tick_value_usd: float = 10.0
    cash_usd: float = 0.0
    position: int = 0
    fills: list[Fill] = field(default_factory=list)

    def apply_fill(self, fill: Fill) -> None:
        self.cash_usd -= fill.signed_quantity * fill.price_ticks * self.tick_value_usd
        self.cash_usd -= fill.fee_usd
        self.position += fill.signed_quantity
        self.fills.append(fill)

    def equity(self, mark_ticks: float) -> float:
        return self.cash_usd + self.position * mark_ticks * self.tick_value_usd

    @property
    def total_fees(self) -> float:
        return float(sum(fill.fee_usd for fill in self.fills))


def execute_ioc(
    book: FIFOBook,
    *,
    direction: int,
    quantity: int,
    time_ns: int,
    fee_per_contract: float,
    slippage_ticks: int = 0,
) -> tuple[Fill, ...]:
    if direction not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")
    if quantity <= 0:
        return ()
    resting_side = "A" if direction > 0 else "B"
    levels = book.price_levels(resting_side)
    remaining = quantity
    fills: list[Fill] = []
    for price, depth in levels:
        take = min(remaining, depth)
        if take <= 0:
            continue
        fill_price = price + direction * int(slippage_ticks)
        fills.append(
            Fill(
                time_ns=int(time_ns),
                signed_quantity=direction * take,
                price_ticks=fill_price,
                fee_usd=fee_per_contract * take,
                liquidity="aggressive",
            )
        )
        remaining -= take
        if remaining == 0:
            break
    return tuple(fills)


@dataclass
class PassiveShadowOrder:
    direction: int
    quantity: int
    price_ticks: int
    arrival_time_ns: int
    queue_ahead: int
    queue_buffer: int
    filled: int = 0
    canceled: bool = False

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled

    def cancel(self) -> None:
        self.canceled = True

    def ahead_removed(self, quantity: int) -> None:
        self.queue_ahead = max(0, self.queue_ahead - max(0, int(quantity)))

    def observe_execution(self, quantity: int, *, event_time_ns: int) -> int:
        if self.canceled or event_time_ns <= self.arrival_time_ns or quantity <= 0:
            return 0
        available = int(quantity)
        consumed_ahead = min(available, self.queue_ahead)
        self.queue_ahead -= consumed_ahead
        available -= consumed_ahead
        consumed_buffer = min(available, self.queue_buffer)
        self.queue_buffer -= consumed_buffer
        available -= consumed_buffer
        fill = min(self.remaining, max(0, available))
        self.filled += fill
        return fill


def submit_passive(
    book: FIFOBook,
    *,
    direction: int,
    quantity: int,
    price_ticks: int,
    arrival_time_ns: int,
    buffer_fraction: float = 0.05,
    minimum_buffer: int = 1,
) -> PassiveShadowOrder:
    side = "B" if direction > 0 else "A"
    ahead = book.level_depth(side, price_ticks)
    buffer = max(int(minimum_buffer), int((ahead * buffer_fraction) + 0.999999))
    return PassiveShadowOrder(direction, quantity, price_ticks, arrival_time_ns, ahead, buffer)


def implementation_shortfall_ticks(
    fills: tuple[Fill, ...] | list[Fill],
    *,
    direction: int,
    arrival_mid_ticks: float,
    requested_quantity: int,
    forced_price_ticks: int,
) -> float:
    filled = sum(abs(fill.signed_quantity) for fill in fills)
    total = sum(abs(fill.signed_quantity) * direction * (fill.price_ticks - arrival_mid_ticks) for fill in fills)
    residual = max(0, requested_quantity - filled)
    total += residual * direction * (forced_price_ticks - arrival_mid_ticks)
    return float(total / requested_quantity) if requested_quantity else 0.0
