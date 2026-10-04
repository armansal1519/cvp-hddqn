from __future__ import annotations

import numpy as np
import pytest

from kx_simple_hmadrl.volume_v6_env import (
    AGENT_STATE_FIELDS,
    AGENT_STATE_WIDTH,
    BiasRewardLabeler,
    TargetPosition,
    V6TradingEnvironment,
    action_for_position,
    action_index,
    legal_target_mask,
    position_for_action,
)


STEP_NS = 250_000_000


def _raw(seconds: int = 10, *, rising_at_second: int | None = None) -> dict[str, np.ndarray]:
    count = seconds * 4 + 1
    clock = np.arange(count, dtype=np.int64) * STEP_NS
    bid = np.full(count, 20_000, dtype=np.int64)
    if rising_at_second is not None:
        bid[clock >= rising_at_second * 1_000_000_000] += 3
    ask = bid + 1
    return {
        "clock_ns": clock,
        "continuity_segment": np.ones(count, dtype=np.int32),
        "valid": np.ones(count, dtype=bool),
        "best_bid_ticks": bid,
        "best_ask_ticks": ask,
        "best_bid_size": np.full(count, 10, dtype=np.int64),
        "best_ask_size": np.full(count, 10, dtype=np.int64),
        "arrival_5ms_clock_ns": clock + 5_000_000,
        "arrival_5ms_state_available_ns": clock + 4_000_000,
        "arrival_5ms_continuity_segment": np.ones(count, dtype=np.int32),
        "arrival_5ms_valid": np.ones(count, dtype=bool),
        "arrival_5ms_bid_ticks": bid.copy(),
        "arrival_5ms_ask_ticks": ask.copy(),
        "arrival_5ms_bid_size": np.full(count, 10, dtype=np.int64),
        "arrival_5ms_ask_size": np.full(count, 10, dtype=np.int64),
    }


def _copy(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in raw.items()}


def test_semantic_actions_and_inventory_masks_have_one_unambiguous_mapping() -> None:
    assert tuple(int(value) for value in TargetPosition) == (-1, 0, 1)
    assert [position_for_action(value) for value in TargetPosition] == [-1, 0, 1]
    assert [action_index(value) for value in TargetPosition] == [0, 1, 2]
    assert [action_for_position(value) for value in (-1, 0, 1)] == list(TargetPosition)
    assert legal_target_mask(-1).tolist() == [True, True, False]
    assert legal_target_mask(0).tolist() == [True, True, True]
    assert legal_target_mask(1).tolist() == [False, True, True]
    with pytest.raises(ValueError, match="inventory"):
        legal_target_mask(2)


def test_direct_reversal_is_rejected_before_any_environment_mutation() -> None:
    env = V6TradingEnvironment(_raw(), "synthetic-development")
    opened = env.step(0, TargetPosition.LONG, 24)
    assert opened.inventory_before == 0 and opened.inventory_after == 1
    assert opened.current_legal_mask == (True, True, True)
    assert opened.next_legal_mask == (False, True, True)
    assert opened.reward_cents == -1_600
    assert opened.reward_ticks == -1.6
    snapshot = (env.ledger.cash_cents, tuple(env.rewards_cents), tuple(env.fills))
    with pytest.raises(ValueError, match="direct reversals"):
        env.step(24, TargetPosition.SHORT, 28)
    assert (env.ledger.cash_cents, tuple(env.rewards_cents), tuple(env.fills)) == snapshot

    flattened = env.step(24, TargetPosition.FLAT, 40, terminal=True)
    assert flattened.inventory_after == 0 and flattened.done
    assert flattened.reward_cents == 0
    result = env.finish()
    assert result.net_pnl_cents == -1_600
    assert sum(result.raw_rewards_cents) == result.net_pnl_cents
    assert result.completed_round_trips == 1


def test_route_uses_first_causal_executable_arrival_within_five_seconds() -> None:
    raw = _raw(seconds=7)
    raw["arrival_5ms_valid"][:] = False
    raw["arrival_5ms_valid"][2] = True
    raw["arrival_5ms_valid"][-1] = True  # mandatory terminal liquidation book
    env = V6TradingEnvironment(raw, "synthetic-development")
    transition = env.step(0, TargetPosition.LONG, 24)
    assert transition.route is not None and not transition.route.rejected
    assert transition.route.fill_index == 2
    assert transition.route.searched_snapshots == 3
    assert transition.route.fill is not None
    assert transition.route.fill.timestamp_ns == 505_000_000
    assert not transition.command_rejected and env.inventory == 1
    result = env.finish()
    assert result.fills[-1].forced
    assert sum(result.raw_rewards_cents) == result.net_pnl_cents


def test_route_rejects_after_deadline_and_preserves_actual_inventory() -> None:
    raw = _raw(seconds=7)
    raw["arrival_5ms_valid"][:] = False
    # This is the first valid arrival, but 5.005 seconds is outside the
    # registered five-second route deadline.
    raw["arrival_5ms_valid"][20] = True
    env = V6TradingEnvironment(raw, "synthetic-development")
    transition = env.step(0, TargetPosition.LONG, 24)
    assert transition.command_created and transition.command_rejected
    assert transition.route is not None and transition.route.rejected
    assert transition.route.searched_snapshots == 20
    assert transition.inventory_after == env.inventory == 0
    assert transition.reward_cents == 0
    assert transition.next_state[-1] == 1.0
    result = env.finish()
    assert result.rejected_commands == 1 and result.fills == ()
    assert sum(result.raw_rewards_cents) == result.net_pnl_cents == 0


def test_terminal_transition_reward_equals_after_cost_cash_cent_for_cent() -> None:
    raw = _raw(seconds=10, rising_at_second=10)
    env = V6TradingEnvironment(raw, "synthetic-development")
    transition = env.step(0, TargetPosition.LONG, len(raw["clock_ns"]) - 1, terminal=True)
    assert transition.route is not None and transition.route.fill is not None
    assert transition.terminal_route is not None and transition.terminal_route.fill is not None
    assert transition.terminal_route.fill.forced
    # Buy 20001, sell 20003: +2 ticks gross and $6 round-trip fees.
    assert transition.reward_cents == 1_400
    assert transition.reward_ticks == 1.4
    result = env.finish()
    assert result.raw_rewards_cents == (1_400,)
    assert sum(result.raw_rewards_cents) == result.net_pnl_cents == 1_400
    assert result.completed_round_trips == 1


def test_agent_state_is_finite_inventory_aware_and_does_not_read_arrival_arrays() -> None:
    original = _raw()
    changed = _copy(original)
    changed["arrival_5ms_bid_ticks"][:] = 10_000
    changed["arrival_5ms_ask_ticks"][:] = 30_000
    first = V6TradingEnvironment(original, "synthetic-development").agent_state(0)
    second = V6TradingEnvironment(changed, "synthetic-development").agent_state(0)
    assert np.array_equal(first, second)
    assert first.shape == (AGENT_STATE_WIDTH,) == (len(AGENT_STATE_FIELDS),)
    assert np.isfinite(first).all()
    assert first[:3].tolist() == [0.0, 1.0, 0.0]
    assert first[3:6].tolist() == [0.0, 1.0, 0.0]
    assert first[10] == 1.0 and first[11] == pytest.approx(0.6)

    env = V6TradingEnvironment(original, "synthetic-development")
    opened = env.step(0, TargetPosition.LONG, 24)
    state = np.asarray(opened.next_state)
    assert state[:3].tolist() == [0.0, 0.0, 1.0]
    assert state[3:6].tolist() == [0.0, 0.0, 1.0]
    assert state[6] == pytest.approx(-1.6)
    assert state[7] == pytest.approx(-0.5)
    assert state[8] > 0


def test_transition_rejects_cross_segment_and_misaligned_arrival_data() -> None:
    raw = _raw()
    raw["continuity_segment"][30:] = 2
    env = V6TradingEnvironment(raw, "synthetic-development")
    with pytest.raises(ValueError, match="continuity"):
        env.step(0, TargetPosition.FLAT, 40)

    malformed = _raw()
    malformed["arrival_5ms_clock_ns"][3] += 1
    with pytest.raises(ValueError, match="registered route latency"):
        V6TradingEnvironment(malformed, "synthetic-development")


def test_route_rejects_arrival_from_another_continuity_segment() -> None:
    raw = _raw(seconds=7)
    raw["arrival_5ms_continuity_segment"][0] = 2
    env = V6TradingEnvironment(raw, "synthetic-development")
    transition = env.step(0, TargetPosition.LONG, 24)
    assert transition.command_rejected
    assert transition.route is not None and transition.route.rejected
    assert env.inventory == 0

    unavailable = _raw()
    unavailable["arrival_5ms_state_available_ns"][0] = (
        unavailable["arrival_5ms_clock_ns"][0] + 1
    )
    with pytest.raises(ValueError, match="unavailable"):
        V6TradingEnvironment(unavailable, "synthetic-development")


def test_legacy_arrival_boundary_is_explicit_smoke_only_and_reported() -> None:
    raw = _raw()
    raw.pop("arrival_5ms_state_available_ns")
    raw.pop("arrival_5ms_continuity_segment")
    with pytest.raises(ValueError, match="missing"):
        V6TradingEnvironment(raw, "synthetic-development")
    env = V6TradingEnvironment(
        raw, "synthetic-smoke", allow_legacy_arrival_segment=True,
    )
    env.step(0, TargetPosition.FLAT, len(raw["clock_ns"]) - 1, terminal=True)
    assert env.finish().unsafe_legacy_arrival_boundary is True


def test_upper_bias_reward_is_nonoverlapping_exact_fifteen_minute_round_trip() -> None:
    raw = _raw(seconds=30 * 60, rising_at_second=15 * 60)
    labeler = BiasRewardLabeler(raw)
    first = labeler.label(0, 15 * 60 * 4)
    assert first.valid_mask == (True, True, True)
    assert first.rewards_cents == (-4_600, 0, 1_400)
    assert first.rewards_ticks == (-4.6, 0.0, 1.4)
    assert all(route is None for route in (first.entry_routes[1], first.exit_routes[1]))
    assert first.entry_routes[0] is not None and first.exit_routes[0] is not None
    assert first.entry_routes[2] is not None and first.exit_routes[2] is not None

    # An exact fifteen-minute interval that begins ten minutes into the first
    # label is forbidden even though both endpoint indexes are otherwise valid.
    with pytest.raises(ValueError, match="must not overlap"):
        labeler.label(10 * 60 * 4, 25 * 60 * 4)

    second = labeler.label(15 * 60 * 4, 30 * 60 * 4)
    assert second.decision_time_ns == first.expiry_time_ns
    assert second.valid_mask == (True, True, True)


def test_upper_bias_reward_masks_direction_when_either_route_is_unexecutable() -> None:
    raw = _raw(seconds=15 * 60)
    expiry = 15 * 60 * 4
    # Long entries require displayed ask size.  Short entry and both exits
    # remain executable, so validity is action-specific rather than invented.
    raw["arrival_5ms_ask_size"][:20] = 0
    result = BiasRewardLabeler(raw).label(0, expiry)
    assert result.valid_mask == (True, True, False)
    assert result.rewards_cents[1:] == (0, 0)
    assert result.entry_routes[2] is not None and result.entry_routes[2].rejected
