from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kx_simple_hmadrl.volume_v6_baseline import (
    BASELINE_NAMES,
    BASE_PAPER_VARIANT,
    BASE_STRONG_VARIANT,
    SessionPair,
    assert_training_sessions,
    baseline_contract,
    evaluate_ensemble,
    economics_for_scenario,
    fine_tune_gc,
    majority_target_vote,
    replay_target_trace,
    sha256_file,
    verify_artifact_manifest,
)
from kx_simple_hmadrl.volume_v6_baseline import (
    _StrongSessionCursor,
    _build_timeframes,
    _contiguous_state_indexes,
    _existing_complete,
    _gc_decision_indexes,
    _gc_indexes,
    _run_interleaved_training_pass,
)


NS_PER_MINUTE = 60_000_000_000


def _raw_session() -> dict[str, np.ndarray]:
    clock = np.asarray([0, 5, 10, 15], dtype=np.int64) * NS_PER_MINUTE
    arrival = clock + 5_000_000
    return {
        "clock_ns": clock,
        "continuity_segment": np.zeros(len(clock), dtype=np.uint32),
        "valid": np.ones(len(clock), dtype=bool),
        "best_bid_ticks": np.full(len(clock), 1_000, dtype=np.int64),
        "best_ask_ticks": np.full(len(clock), 1_001, dtype=np.int64),
        "best_bid_size": np.ones(len(clock), dtype=np.int64),
        "best_ask_size": np.ones(len(clock), dtype=np.int64),
        "arrival_5ms_clock_ns": arrival,
        "arrival_5ms_state_available_ns": arrival - 1_000_000,
        "arrival_5ms_continuity_segment": np.zeros(len(clock), dtype=np.uint32),
        "arrival_5ms_valid": np.ones(len(clock), dtype=bool),
        "arrival_5ms_bid_ticks": np.full(len(clock), 1_000, dtype=np.int64),
        "arrival_5ms_ask_ticks": np.full(len(clock), 1_001, dtype=np.int64),
        "arrival_5ms_bid_size": np.ones(len(clock), dtype=np.int64),
        "arrival_5ms_ask_size": np.ones(len(clock), dtype=np.int64),
        "arrival_5ms_state_available_ns": arrival,
        "arrival_5ms_continuity_segment": np.zeros(len(clock), dtype=np.uint32),
    }


def test_registered_baseline_contracts_keep_paper_clocks_but_fix_strong_sync() -> None:
    paper = baseline_contract(BASE_PAPER_VARIANT)
    strong = baseline_contract(BASE_STRONG_VARIANT)
    paper_structure = [
        (agent.name, agent.minutes, agent.lookback, agent.action_lock_bars,
         agent.gamma, agent.hidden_sizes, agent.inherited_positions)
        for agent in paper.agents
    ]
    strong_structure = [
        (agent.name, agent.minutes, agent.lookback, agent.action_lock_bars,
         agent.gamma, agent.hidden_sizes, agent.inherited_positions)
        for agent in strong.agents
    ]
    assert paper_structure == strong_structure
    assert paper.optimization.target_sync_updates == 1_000
    assert paper.optimization.optimizer == "rmsprop"
    assert paper.optimization.loss == "mse"
    assert paper.optimization.target_rule == "vanilla"
    assert strong.optimization.target_sync_updates == 100
    assert strong.optimization.optimizer == "adam"
    assert strong.optimization.loss == "huber"
    assert strong.optimization.target_rule == "double"
    assert strong.optimization.network_head == "dueling"
    assert "exact_common_ledger" in strong.reward


def test_semantic_five_seed_vote_is_flat_on_any_tie() -> None:
    traces = np.asarray([
        [-1, -1, 1],
        [-1, -1, 1],
        [0, 0, 1],
        [1, 1, 0],
        [1, 1, -1],
    ], dtype=np.int8)
    # The first two columns tie SHORT/LONG 2--2.  The last has a clear LONG vote.
    assert majority_target_vote(traces).tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match="five aligned"):
        majority_target_vote(traces[:4])


def test_training_date_guard_runs_before_missing_file_access(tmp_path: Path) -> None:
    locked = SessionPair(
        tmp_path / "2026-05-01.npz",
        tmp_path / "sidecars" / "2026-05-01.npz",
    )
    with pytest.raises(PermissionError, match="training request"):
        assert_training_sessions([locked])


def test_test_evaluation_requires_unlock_before_checkpoint_or_shard_read(tmp_path: Path) -> None:
    locked = SessionPair(
        tmp_path / "2026-06-01.npz",
        tmp_path / "sidecars" / "2026-06-01.npz",
    )
    with pytest.raises(PermissionError, match="unlock receipt"):
        evaluate_ensemble(
            [tmp_path / f"missing_seed_{seed}" for seed in range(5)],
            [locked], [locked], variant=BASE_PAPER_VARIANT, phase="test",
        )


def test_validation_requires_full_proof_chain_before_checkpoint_or_shard_read(
    tmp_path: Path,
) -> None:
    may = SessionPair(
        tmp_path / "2026-05-01.npz",
        tmp_path / "sidecars" / "2026-05-01.npz",
    )
    with pytest.raises(PermissionError, match="complete phase proof chain"):
        evaluate_ensemble(
            [tmp_path / f"missing_seed_{seed}" for seed in range(5)],
            [may], [may], variant=BASE_PAPER_VARIANT, phase="validation",
        )


def test_registered_fine_tune_requires_dataset_proof_before_any_input_read(
    tmp_path: Path,
) -> None:
    train = SessionPair(
        tmp_path / "2025-01-02.npz",
        tmp_path / "sidecars" / "2025-01-02.npz",
    )
    with pytest.raises(PermissionError, match="training dataset manifest"):
        fine_tune_gc(
            [train], tmp_path / "missing-pretrain", tmp_path / "output",
            seed=1103, variant=BASE_PAPER_VARIANT, phase="train",
        )


def test_common_ledger_reversal_flattens_then_waits_for_later_m5_decision() -> None:
    replay = replay_target_trace(
        _raw_session(),
        session_date="2025-01-09",
        session_start_ns=0,
        session_end_ns=15 * NS_PER_MINUTE,
        signal_times_ns=np.asarray([5, 10], dtype=np.int64) * NS_PER_MINUTE,
        signal_actions=np.asarray([1, -1], dtype=np.int8),
        hold_minutes=100,
    )
    assert replay.reversal_flatten_decisions == 1
    assert replay.result.completed_round_trips == 1
    assert [fill.side for fill in replay.result.fills] == [1, -1]
    assert replay.result.net_pnl_cents == -1_600
    assert sum(replay.attributed_reward_cents) == replay.result.net_pnl_cents


def test_invalid_five_minute_endpoint_is_skipped_without_forward_fill() -> None:
    raw = _raw_session()
    raw["valid"][2] = False
    replay = replay_target_trace(
        raw,
        session_date="2025-01-09",
        session_start_ns=0,
        session_end_ns=15 * NS_PER_MINUTE,
        signal_times_ns=np.asarray([5, 10], dtype=np.int64) * NS_PER_MINUTE,
        signal_actions=np.asarray([1, -1], dtype=np.int8),
        hold_minutes=100,
    )
    # The invalid 10-minute decision never transforms LONG into FLAT/SHORT.
    # The existing LONG target persists until deterministic terminal liquidation.
    assert replay.reversal_flatten_decisions == 0
    assert [fill.side for fill in replay.result.fills] == [1, -1]
    assert [transition.index for transition in replay.result.transitions] == [1]


def test_gc_reward_horizon_cannot_jump_to_the_next_session() -> None:
    base = baseline_contract(BASE_PAPER_VARIANT).agent("M5")
    spec = replace(base, lookback=2, action_lock_bars=1)
    next_day = 24 * 60 * NS_PER_MINUTE
    frame = pd.DataFrame({
        "event_time": pd.to_datetime(
            [0, 5 * NS_PER_MINUTE, next_day, next_day + 5 * NS_PER_MINUTE], utc=True,
        ),
        "state_id": ["GCG5:0"] * 4,
        "session_date": ["2025-01-09", "2025-01-09", "2025-01-10", "2025-01-10"],
        "session_end_ns": [30 * NS_PER_MINUTE, 30 * NS_PER_MINUTE,
                           next_day + 30 * NS_PER_MINUTE, next_day + 30 * NS_PER_MINUTE],
    })
    # Candidate index 1 would cross overnight and is rejected.  Index 2 has an
    # exact five-minute same-session future and remains eligible.
    assert _gc_indexes(frame, spec).tolist() == [2]


def test_inference_keeps_late_renewals_and_resets_lock_each_session() -> None:
    base = baseline_contract(BASE_PAPER_VARIANT).agent("M5")
    spec = replace(base, lookback=2, action_lock_bars=2)
    next_day = 24 * 60 * NS_PER_MINUTE
    event_ns = np.asarray(
        [0, 5, 10, 15, 20, 25, 0, 5, 10, 15, 20, 25], dtype=np.int64,
    ) * NS_PER_MINUTE
    event_ns[6:] += next_day
    frame = pd.DataFrame({
        "event_time": pd.to_datetime(event_ns, utc=True),
        "state_id": ["GC:0"] * len(event_ns),
        "session_date": ["2025-01-09"] * 6 + ["2025-01-10"] * 6,
        "session_end_ns": [30 * NS_PER_MINUTE] * 6
        + [next_day + 30 * NS_PER_MINUTE] * 6,
    })
    assert _gc_decision_indexes(frame, spec).tolist() == [1, 3, 5, 6, 8, 10]
    # The final renewal in each session remains an inference publication but
    # lacks its exact same-session reward horizon.
    assert _gc_indexes(frame, spec).tolist() == [1, 3, 6, 8]


def test_missing_intraday_step_invalidates_the_entire_lookback() -> None:
    base = baseline_contract(BASE_PAPER_VARIANT).agent("M5")
    spec = replace(base, lookback=3, action_lock_bars=1)
    event_ns = np.asarray([0, 5, 15, 20, 25], dtype=np.int64) * NS_PER_MINUTE
    frame = pd.DataFrame({
        "event_time": pd.to_datetime(event_ns, utc=True),
        "state_id": ["GC:0"] * len(event_ns),
        "session_date": ["2025-01-09"] * len(event_ns),
        "session_end_ns": [30 * NS_PER_MINUTE] * len(event_ns),
    })
    assert _contiguous_state_indexes(frame, spec).tolist() == [4]


def test_month_gap_resets_state_without_requiring_unauthorized_intervening_data() -> None:
    base = baseline_contract(BASE_PAPER_VARIANT).agent("M5")
    spec = replace(base, lookback=2, action_lock_bars=1)
    april = pd.Timestamp("2026-04-30 17:00", tz="UTC").value
    june = pd.Timestamp("2026-06-01 12:00", tz="UTC").value
    event_ns = np.asarray([
        april, april + 5 * NS_PER_MINUTE,
        june, june + 5 * NS_PER_MINUTE,
    ], dtype=np.int64)
    frame = pd.DataFrame({
        "event_time": pd.to_datetime(event_ns, utc=True),
        "state_id": ["GC:0"] * 4,
        "session_date": ["2026-04-30", "2026-04-30", "2026-06-01", "2026-06-01"],
        "session_end_ns": [event_ns[1] + 5 * NS_PER_MINUTE] * 2
        + [event_ns[3] + 5 * NS_PER_MINUTE] * 2,
    })
    # June's first bar cannot inherit April state.  The next exact June bar
    # becomes valid using June data alone; May is neither opened nor required.
    assert _contiguous_state_indexes(frame, spec).tolist() == [1, 3]


def test_h1_and_m15_are_exact_rollups_of_complete_m5_bars() -> None:
    event_ns = np.arange(60, dtype=np.int64) * NS_PER_MINUTE
    one = pd.DataFrame({
        "event_time": pd.to_datetime(event_ns, utc=True),
        "open": np.arange(60, dtype=float) + 1,
        "high": np.arange(60, dtype=float) + 2,
        "low": np.arange(60, dtype=float) + 0.5,
        "close": np.arange(60, dtype=float) + 1.5,
        "volume": np.ones(60),
        "state_id": ["GC:0"] * 60,
        "session_date": ["2025-01-09"] * 60,
        "raw_symbol": ["GCG5"] * 60,
        "session_end_ns": [65 * NS_PER_MINUTE] * 60,
    })
    frames = _build_timeframes(one)
    assert len(frames["M5"]) == 12
    assert len(frames["M15"]) == 4
    assert len(frames["H1"]) == 1
    assert frames["H1"].iloc[0]["component_count"] == 12
    assert frames["H1"].iloc[0]["volume"] == 60
    incomplete = _build_timeframes(one.drop(index=7))
    assert len(incomplete["M5"]) == 11
    assert len(incomplete["H1"]) == 0


def test_training_acts_before_same_state_reward_matures_and_updates() -> None:
    events: list[str] = []

    class Trainer:
        def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
            events.append(f"act:{int(observation[0])}")
            return 1

        def append(
            self, observation: np.ndarray, action: int, reward: float,
            next_observation: np.ndarray, *, terminal: bool = False,
        ) -> None:
            events.append(f"append:{int(observation[0])}->{int(next_observation[0])}")

        def update(self) -> float:
            events.append("update")
            return 0.25

    _, rewards, losses = _run_interleaved_training_pass(
        Trainer(),  # type: ignore[arg-type]
        event_indexes=np.asarray([0, 2], dtype=np.int64),
        decision_indexes=np.asarray([0, 2], dtype=np.int64),
        maturities={0: 2},
        observation_at=lambda index: {
            0: np.asarray([0], dtype=np.float32),
            2: np.asarray([2], dtype=np.float32),
        }[index],
        reward_for=lambda origin, future, action: float(future - origin) * action,
    )
    assert events == ["act:0", "act:2", "append:0->2", "update"]
    assert rewards == [2.0]
    assert losses == [0.25]


def test_online_strong_reward_cursor_matches_batch_common_ledger() -> None:
    raw = _raw_session()
    signal_times = np.asarray([5, 10], dtype=np.int64) * NS_PER_MINUTE
    signal_actions = np.asarray([1, -1], dtype=np.int8)
    batch = replay_target_trace(
        raw, session_date="2025-01-09", session_start_ns=0,
        session_end_ns=15 * NS_PER_MINUTE, signal_times_ns=signal_times,
        signal_actions=signal_actions, hold_minutes=100,
    )
    economics = economics_for_scenario("base")
    cursor = _StrongSessionCursor(
        raw, session_date="2025-01-09", session_start_ns=0,
        session_end_ns=15 * NS_PER_MINUTE, hold_minutes=100,
        economics=economics,
    )
    cursor.advance_before(int(signal_times[0])); cursor.register(0, int(signal_times[0]), 1)
    cursor.advance_before(int(signal_times[1])); cursor.register(1, int(signal_times[1]), -1)
    result = cursor.finish()
    assert result.net_pnl_cents == batch.result.net_pnl_cents
    assert tuple(cursor.rewards[index] for index in range(2)) == batch.attributed_reward_cents


def test_scenario_label_has_one_exact_economic_contract() -> None:
    assert economics_for_scenario("base").fee_cents_per_side == 300
    assert economics_for_scenario("double_commission").fee_cents_per_side == 600
    assert economics_for_scenario("20ms_route_latency").route_latency_ms == 20
    assert economics_for_scenario("one_adverse_tick_per_filled_side").adverse_ticks == 1
    with pytest.raises(ValueError, match="scenario"):
        economics_for_scenario("custom")


def test_completed_manifest_hashes_are_verified(tmp_path: Path) -> None:
    product = tmp_path / "seed_1103_m5_gc.pt"
    product.write_bytes(b"frozen-checkpoint")
    manifest = {
        "schema_version": 1,
        "artifact_type": "V6_BASELINE_GC_FINETUNE",
        "complete": True,
        "name": BASELINE_NAMES[BASE_PAPER_VARIANT],
        "variant": BASE_PAPER_VARIANT,
        "seed": 1103,
        "artifacts": [{"path": product.name, "sha256": sha256_file(product)}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = verify_artifact_manifest(
        tmp_path,
        artifact_type="V6_BASELINE_GC_FINETUNE",
        variant=BASE_PAPER_VARIANT,
        seed=1103,
    )
    assert loaded["complete"] is True
    product.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        verify_artifact_manifest(tmp_path)


def test_completed_resume_refuses_a_different_bound_run_identity(tmp_path: Path) -> None:
    product = tmp_path / "seed_1103_m5_gc.pt"
    product.write_bytes(b"frozen-checkpoint")
    manifest = {
        "schema_version": 1,
        "artifact_type": "V6_BASELINE_GC_FINETUNE",
        "complete": True,
        "variant": BASE_PAPER_VARIANT,
        "seed": 1103,
        "run_identity_sha256": "a" * 64,
        "source_digest": "b" * 64,
        "passes": 3,
        "artifacts": [{"path": product.name, "sha256": sha256_file(product)}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _existing_complete(
        tmp_path, artifact_type="V6_BASELINE_GC_FINETUNE",
        variant=BASE_PAPER_VARIANT, seed=1103,
        expected_identity={"run_identity_sha256": "a" * 64, "passes": 3},
    ) == tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="run_identity_sha256"):
        _existing_complete(
            tmp_path, artifact_type="V6_BASELINE_GC_FINETUNE",
            variant=BASE_PAPER_VARIANT, seed=1103,
            expected_identity={"run_identity_sha256": "c" * 64},
        )
