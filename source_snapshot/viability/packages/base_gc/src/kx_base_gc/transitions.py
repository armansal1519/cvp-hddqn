from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np

from .contract import ACTIONS, AgentSpec, BaselineContract


@dataclass(frozen=True)
class Transition:
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    decision_time_ns: int
    maturity_time_ns: int
    boundary_id: str
    gross_reward: float
    cost_usd: float
    entry_arrival_time_ns: int | None
    exit_arrival_time_ns: int | None
    entry_fill_source_ordinal: int | None
    exit_fill_source_ordinal: int | None


class ActionLock:
    """Freeze a chosen action for exactly the registered number of agent bars."""

    def __init__(self, hold_bars: int) -> None:
        if hold_bars <= 0:
            raise ValueError("hold_bars must be positive")
        self.hold_bars = int(hold_bars)
        self.current_action = 0
        self.remaining_bars = 0
        self.boundary_id: Hashable | None = None

    def reset(self) -> None:
        self.current_action = 0
        self.remaining_bars = 0
        self.boundary_id = None

    def step(self, proposed_action: int, *, boundary_id: Hashable) -> int:
        if proposed_action not in ACTIONS:
            raise ValueError(f"action must be in {ACTIONS}")
        if self.boundary_id != boundary_id:
            self.current_action = 0
            self.remaining_bars = 0
            self.boundary_id = boundary_id
        if self.remaining_bars == 0:
            self.current_action = int(proposed_action)
            self.remaining_bars = self.hold_bars
        executed = self.current_action
        self.remaining_bars -= 1
        return executed


@dataclass
class _Pending:
    index: int
    observation: tuple[float, ...]
    action: int
    close_ticks: float
    aggressive_sell_fill_ticks: float | None
    aggressive_buy_fill_ticks: float | None
    decision_time_ns: int
    arrival_time_ns: int | None
    fill_source_ordinal: int | None


class DelayedTransitionBuilder:
    """Reveal horizon rewards only when their future observation is available."""

    def __init__(
        self,
        horizon_bars: int,
        *,
        reward_mode: Literal["gross_fraction", "net_usd"] = "gross_fraction",
        tick_value_usd: float = 10.0,
        fee_usd_per_side: float = 3.0,
        quantity: int = 1,
        decision_latency_ns: int = 0,
        route_latency_ns: int = 0,
    ) -> None:
        if horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if tick_value_usd <= 0 or fee_usd_per_side < 0 or quantity <= 0:
            raise ValueError("invalid reward economics")
        self.horizon_bars = int(horizon_bars)
        self.reward_mode = reward_mode
        self.tick_value_usd = float(tick_value_usd)
        self.fee_usd_per_side = float(fee_usd_per_side)
        self.quantity = int(quantity)
        if decision_latency_ns < 0 or route_latency_ns < 0:
            raise ValueError("registered latencies cannot be negative")
        self.decision_latency_ns = int(decision_latency_ns)
        self.route_latency_ns = int(route_latency_ns)
        self._boundary: str | None = None
        self._index = -1
        self._pending: list[_Pending] = []
        self._last_available_time_ns: int | None = None
        self.discarded_at_boundaries = 0

    def reset(self) -> None:
        self._boundary = None
        self._index = -1
        self._pending.clear()
        self._last_available_time_ns = None

    @staticmethod
    def _observation(values: Sequence[float]) -> tuple[float, ...]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("observation must be a finite vector")
        return tuple(float(value) for value in array)

    def observe(
        self,
        observation: Sequence[float],
        *,
        action: int,
        price_ticks: float,
        available_time_ns: int,
        boundary_id: Hashable,
        aggressive_sell_fill_ticks: float | None = None,
        aggressive_buy_fill_ticks: float | None = None,
        decision_time_ns: int | None = None,
        submission_time_ns: int | None = None,
        arrival_time_ns: int | None = None,
        aggressive_fill_available_time_ns: int | None = None,
        aggressive_fill_source_ordinal: int | None = None,
        record_decision: bool = True,
    ) -> list[Transition]:
        if action not in ACTIONS:
            raise ValueError(f"action must be in {ACTIONS}")
        if not np.isfinite(price_ticks) or price_ticks <= 0:
            raise ValueError("price_ticks must be finite and positive exchange ticks")
        sell_fill: float | None = None
        buy_fill: float | None = None
        if self.reward_mode == "net_usd":
            if aggressive_sell_fill_ticks is None or aggressive_buy_fill_ticks is None:
                raise ValueError(
                    "net_usd rewards require causal arrival-aligned aggressive "
                    "sell/buy fill ticks"
                )
            sell_fill = float(aggressive_sell_fill_ticks)
            buy_fill = float(aggressive_buy_fill_ticks)
            if (
                not np.isfinite(sell_fill)
                or not np.isfinite(buy_fill)
                or sell_fill <= 0
                or buy_fill <= 0
                or sell_fill > buy_fill
            ):
                raise ValueError("aggressive fill ticks must satisfy 0 < sell <= buy")
            clocks = (
                decision_time_ns,
                submission_time_ns,
                arrival_time_ns,
                aggressive_fill_available_time_ns,
                aggressive_fill_source_ordinal,
            )
            if any(value is None for value in clocks):
                raise ValueError(
                    "net_usd rewards require decision/submission/arrival/fill clocks "
                    "and a positive source ordinal"
                )
            decision_clock = int(decision_time_ns)  # type: ignore[arg-type]
            submission_clock = int(submission_time_ns)  # type: ignore[arg-type]
            arrival_clock = int(arrival_time_ns)  # type: ignore[arg-type]
            fill_clock = int(aggressive_fill_available_time_ns)  # type: ignore[arg-type]
            fill_ordinal = int(aggressive_fill_source_ordinal)  # type: ignore[arg-type]
            if decision_clock < int(available_time_ns) + self.decision_latency_ns:
                raise ValueError("decision clock violates registered decision latency")
            if submission_clock < decision_clock:
                raise ValueError("submission clock precedes decision")
            if arrival_clock < submission_clock + self.route_latency_ns:
                raise ValueError("arrival clock violates registered route latency")
            if fill_clock > arrival_clock:
                raise ValueError("aggressive fill state was unavailable at venue arrival")
            if fill_ordinal <= 0:
                raise ValueError("aggressive fill source ordinal must be positive")
        else:
            arrival_clock = None
            fill_ordinal = None
        state = self._observation(observation)
        boundary = str(boundary_id)
        if self._boundary != boundary:
            self.discarded_at_boundaries += len(self._pending)
            self._pending.clear()
            self._boundary = boundary
            self._index = 0
            self._last_available_time_ns = None
        else:
            if (
                self._last_available_time_ns is not None
                and int(available_time_ns) <= self._last_available_time_ns
            ):
                raise ValueError("observations must arrive in strict causal order")
            self._index += 1
        self._last_available_time_ns = int(available_time_ns)

        matured: list[Transition] = []
        still_pending: list[_Pending] = []
        for pending in self._pending:
            if self._index - pending.index != self.horizon_bars:
                still_pending.append(pending)
                continue
            fractional = (
                (float(price_ticks) - pending.close_ticks)
                / pending.close_ticks
                * pending.action
            )
            if self.reward_mode == "net_usd" and pending.action != 0:
                if (
                    pending.aggressive_sell_fill_ticks is None
                    or pending.aggressive_buy_fill_ticks is None
                    or sell_fill is None
                    or buy_fill is None
                ):
                    raise AssertionError("validated fill ticks became unavailable")
                entry_ticks = (
                    pending.aggressive_buy_fill_ticks
                    if pending.action > 0
                    else pending.aggressive_sell_fill_ticks
                )
                exit_ticks = sell_fill if pending.action > 0 else buy_fill
                gross_usd = (
                    (exit_ticks - entry_ticks)
                    * pending.action
                    * self.tick_value_usd
                    * self.quantity
                )
            else:
                gross_usd = 0.0
            cost = (
                2.0 * self.fee_usd_per_side * self.quantity
                if pending.action != 0 and self.reward_mode == "net_usd"
                else 0.0
            )
            gross = fractional if self.reward_mode == "gross_fraction" else gross_usd
            matured.append(
                Transition(
                    observation=pending.observation,
                    action=pending.action,
                    reward=float(gross - cost),
                    next_observation=state,
                    decision_time_ns=pending.decision_time_ns,
                    maturity_time_ns=int(available_time_ns),
                    boundary_id=boundary,
                    gross_reward=float(gross),
                    cost_usd=float(cost),
                    entry_arrival_time_ns=pending.arrival_time_ns,
                    exit_arrival_time_ns=arrival_clock,
                    entry_fill_source_ordinal=pending.fill_source_ordinal,
                    exit_fill_source_ordinal=fill_ordinal,
                )
            )
        self._pending = still_pending
        if record_decision:
            self._pending.append(
                _Pending(
                    index=self._index,
                    observation=state,
                    action=int(action),
                    close_ticks=float(price_ticks),
                    aggressive_sell_fill_ticks=sell_fill,
                    aggressive_buy_fill_ticks=buy_fill,
                    decision_time_ns=(
                        int(decision_time_ns)
                        if decision_time_ns is not None
                        else int(available_time_ns)
                    ),
                    arrival_time_ns=arrival_clock,
                    fill_source_ordinal=fill_ordinal,
                )
            )
        return matured

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def transition_builder_for(
    contract: BaselineContract,
    agent: AgentSpec,
) -> DelayedTransitionBuilder:
    """Bind delayed reward semantics to a registered Base-GC variant."""

    try:
        registered_agent = contract.agent(agent.name)
    except KeyError as error:
        raise ValueError("agent is not part of the supplied Base-GC contract") from error
    if agent != registered_agent:
        raise ValueError("agent specification differs from the supplied Base-GC contract")
    reward_mode = "net_usd" if contract.variant == "base_strong_gc" else "gross_fraction"
    return DelayedTransitionBuilder(
        agent.action_lock_bars,
        reward_mode=reward_mode,
        tick_value_usd=contract.execution.tick_value_usd,
        fee_usd_per_side=contract.execution.fee_usd_per_filled_side,
        quantity=contract.execution.quantity,
        decision_latency_ns=int(contract.execution.decision_latency_ms * 1_000_000),
        route_latency_ns=int(contract.execution.route_latency_ms * 1_000_000),
    )
