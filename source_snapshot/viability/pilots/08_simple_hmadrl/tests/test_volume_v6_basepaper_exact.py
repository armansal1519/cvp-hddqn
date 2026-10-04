from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest
import torch

from kx_base_gc.contract import registered_contract
from kx_simple_hmadrl import volume_v6_basepaper_exact as exact
from kx_simple_hmadrl.volume_v6_basepaper_exact import (
    PAPER_EXACT_NAME,
    PAPER_INITIALIZATION,
    PAPER_TRAINING_STAGE,
    PaperDQNTrainer,
    paper_observation,
    paper_origins,
    paper_segments,
    replay_paper_targets,
    run_joint_paper_pass,
    run_paper_agent_events,
    verify_training_artifact,
)


NS_PER_MINUTE = 60_000_000_000


class _RecordingTrainer:
    """Small injectable trainer that exposes the paper's two distinct clocks."""

    def __init__(
        self,
        selected_actions: tuple[int, ...] = (1, -1, 0),
        *,
        update_result: float | None = 0.25,
        spec=None,
    ) -> None:
        self.selected_actions = selected_actions
        self.update_result = update_result
        self.selected_observations: list[np.ndarray] = []
        self.transitions: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.update_calls = 0
        self.environment_steps = 0
        self.events: list[str] = []
        self.spec = spec
        self.agent = spec

    def act(self, observation: np.ndarray, *, explore: bool = True) -> int:
        ordinal = len(self.selected_observations)
        if ordinal >= len(self.selected_actions):
            raise AssertionError("test trainer received an unexpected lock renewal")
        self.selected_observations.append(np.asarray(observation).copy())
        self.events.append(f"act:{ordinal}")
        return self.selected_actions[ordinal]

    def append(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        *,
        terminal: bool = False,
    ) -> None:
        self.transitions.append(
            (
                np.asarray(observation).copy(),
                int(action),
                float(reward),
                np.asarray(next_observation).copy(),
                bool(terminal),
            )
        )
        self.events.append(f"append:{len(self.transitions) - 1}")

    def update(self) -> float | None:
        self.update_calls += 1
        self.events.append(f"update:{self.update_calls - 1}")
        return self.update_result

    def advance_environment_step(self) -> None:
        """Advance epsilon and target-sync clocks without selecting an action."""

        self.environment_steps += 1
        self.events.append(f"advance:{self.environment_steps - 1}")


def _bars(times: pd.DatetimeIndex) -> pd.DataFrame:
    ordinal = np.arange(len(times), dtype=np.float64)
    return pd.DataFrame(
        {
            "event_time": times,
            "open": 100.0 + ordinal,
            "high": 101.0 + ordinal,
            "low": 99.0 + ordinal,
            "close": 100.0 + ordinal,
            "volume": 1_000.0 + ordinal,
            "state_id": "one-source-segment",
        }
    )


def _paper_m5_spec(*, lookback: int = 2, lock: int = 2):
    return replace(
        registered_contract("base_paper_gc").agent("M5"),
        lookback=lookback,
        action_lock_bars=lock,
        inherited_positions=(),
    )


def _raw_session(*, displayed_size: int = 2) -> dict[str, np.ndarray]:
    clock = np.asarray([0, 5, 10, 15], dtype=np.int64) * NS_PER_MINUTE
    arrival = clock + 5_000_000
    return {
        "clock_ns": clock,
        "continuity_segment": np.zeros(len(clock), dtype=np.uint32),
        "valid": np.ones(len(clock), dtype=bool),
        "best_bid_ticks": np.full(len(clock), 1_000, dtype=np.int64),
        "best_ask_ticks": np.full(len(clock), 1_001, dtype=np.int64),
        "best_bid_size": np.full(len(clock), displayed_size, dtype=np.int64),
        "best_ask_size": np.full(len(clock), displayed_size, dtype=np.int64),
        "arrival_5ms_clock_ns": arrival,
        "arrival_5ms_state_available_ns": arrival,
        "arrival_5ms_continuity_segment": np.zeros(len(clock), dtype=np.uint32),
        "arrival_5ms_valid": np.ones(len(clock), dtype=bool),
        "arrival_5ms_bid_ticks": np.full(len(clock), 1_000, dtype=np.int64),
        "arrival_5ms_ask_ticks": np.full(len(clock), 1_001, dtype=np.int64),
        "arrival_5ms_bid_size": np.full(len(clock), displayed_size, dtype=np.int64),
        "arrival_5ms_ask_size": np.full(len(clock), displayed_size, dtype=np.int64),
    }


def test_paper_observation_is_raw_newest_to_oldest_ohlcv_plus_parents() -> None:
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2024-01-02 00:00Z", "2024-01-02 00:05Z", "2024-01-02 00:10Z"]
            ),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [10.5, 20.5, 30.5],
            "volume": [100.0, 200.0, 300.0],
        }
    )
    actual = paper_observation(
        frame, index=2, lookback=2, parent_positions=(1, -1),
    )
    expected = np.asarray(
        [
            30.0, 31.0, 29.0, 30.5, 300.0,
            20.0, 21.0, 19.0, 20.5, 200.0,
            1.0, -1.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_paper_segments_keep_row_continuity_across_market_and_missing_bar_gaps() -> None:
    pieces: list[pd.DatetimeIndex] = []
    for day in pd.date_range("2024-01-02", "2024-01-04", freq="D", tz="UTC"):
        pieces.append(pd.date_range(day, day + timedelta(hours=22), freq="15min"))
        pieces.append(
            pd.date_range(
                day + timedelta(hours=23, minutes=15),
                day + timedelta(hours=23, minutes=45),
                freq="15min",
            )
        )
    frame = _bars(pd.DatetimeIndex(np.concatenate([piece.values for piece in pieces]), tz="UTC"))
    assert len(frame) == 276
    assert np.unique(paper_segments(frame)).tolist() == [0]

    spec = replace(
        registered_contract("base_paper_gc").agent("M15"),
        lookback=200,
        action_lock_bars=10,
    )
    np.testing.assert_array_equal(paper_origins(frame, spec), np.arange(199, 266))

    # The paper indexes supplied rows. A missing quote bar is recorded by the
    # acquisition audit, but it does not turn all subsequent rows into a new MDP.
    missing = frame.drop(index=frame.index[100]).reset_index(drop=True)
    assert np.unique(paper_segments(missing)).tolist() == [0]
    np.testing.assert_array_equal(paper_origins(missing, spec), np.arange(199, 265))


def test_paper_segments_reset_only_at_explicit_or_long_source_boundaries() -> None:
    frame = _bars(
        pd.to_datetime(
            [
                "2024-01-02 21:45Z",
                "2024-01-02 23:15Z",
                "2024-01-11 00:00Z",
                "2024-01-11 00:15Z",
            ]
        )
    )
    assert paper_segments(frame, max_gap_days=7).tolist() == [0, 0, 1, 1]
    changed = frame.copy()
    changed.loc[3, "state_id"] = "new-source-segment"
    assert paper_segments(changed, max_gap_days=7).tolist() == [0, 0, 1, 2]


def test_paper_loop_records_each_bar_with_held_action_and_t_plus_one_successor() -> None:
    frame = _bars(pd.date_range("2024-01-02", periods=7, freq="5min", tz="UTC"))
    spec = _paper_m5_spec()
    trainer = _RecordingTrainer((1, -1))

    run_paper_agent_events(trainer, frame, spec)

    assert len(trainer.selected_observations) == 2
    assert len(trainer.transitions) == 4
    assert trainer.update_calls == 4
    assert trainer.environment_steps == 4
    assert [transition[1] for transition in trainer.transitions] == [1, 1, -1, -1]
    assert all(transition[4] is False for transition in trainer.transitions)

    origins = paper_origins(frame, spec)
    assert origins.tolist() == [1, 2, 3, 4]
    for transition, origin in zip(trainer.transitions, origins, strict=True):
        observation, action, reward, next_observation, _ = transition
        np.testing.assert_array_equal(
            observation, paper_observation(frame, int(origin), spec.lookback),
        )
        np.testing.assert_array_equal(
            next_observation,
            paper_observation(frame, int(origin) + 1, spec.lookback),
        )
        expected_reward = action * (
            float(frame.iloc[int(origin) + spec.action_lock_bars]["close"])
            / float(frame.iloc[int(origin)]["close"])
            - 1.0
        )
        assert reward == pytest.approx(expected_reward)

    # Algorithm 1 orders each eligible bar as transition, update, then epsilon/
    # target-clock advancement. Only lock renewals contain an action selection.
    assert trainer.events == [
        "act:0", "append:0", "update:0", "advance:0",
        "append:1", "update:1", "advance:1",
        "act:1", "append:2", "update:2", "advance:2",
        "append:3", "update:3", "advance:3",
    ]


def test_h1_lock_and_reward_horizon_count_rows_across_gc_sessions() -> None:
    timestamps: list[pd.Timestamp] = []
    for day in pd.date_range("2025-01-06", periods=4, freq="D", tz="UTC"):
        timestamps.extend(day + pd.to_timedelta(np.arange(8, 13), unit="h"))
    frame = _bars(pd.DatetimeIndex(timestamps))
    spec = replace(
        registered_contract("base_paper_gc").agent("H1"),
        lookback=2,
        action_lock_bars=5,
    )
    trainer = _RecordingTrainer((1, -1, 0))

    assert paper_origins(frame, spec).tolist() == list(range(1, 15))
    run_paper_agent_events(trainer, frame, spec)

    # No UTC-midnight/session reset: renewals are exactly five supplied H1 bars
    # apart, and each eligible H1 origin produces an adaptation transition.
    assert [values[0] for values in trainer.selected_observations] == [101.0, 106.0, 111.0]
    assert len(trainer.transitions) == 14
    assert trainer.update_calls == 14
    assert trainer.environment_steps == 14


def test_current_pandas_numpy_xau_midnight_rebuild_does_not_raise() -> None:
    frame = _bars(
        pd.to_datetime(["2024-01-01 23:45Z", "2024-01-02 00:00Z"])
    ).drop(columns="state_id")
    assert paper_segments(frame).tolist() == [0, 0]


def test_registered_training_fails_closed_on_zero_transition_or_update_support() -> None:
    short = _bars(pd.date_range("2024-01-02", periods=3, freq="5min", tz="UTC"))
    with pytest.raises(ValueError, match="transition support"):
        run_paper_agent_events(_RecordingTrainer(), short, _paper_m5_spec())

    supported = _bars(pd.date_range("2024-01-02", periods=7, freq="5min", tz="UTC"))
    with pytest.raises(ValueError, match="optimizer update support"):
        run_paper_agent_events(
            _RecordingTrainer((1, -1), update_result=None),
            supported,
            _paper_m5_spec(),
        )


def test_paper_direct_reversal_closes_and_opens_opposite_at_same_decision() -> None:
    replay = replay_paper_targets(
        _raw_session(displayed_size=2),
        session_date="2025-01-09",
        session_start_ns=0,
        session_end_ns=15 * NS_PER_MINUTE,
        signal_times_ns=np.asarray([5, 10], dtype=np.int64) * NS_PER_MINUTE,
        signal_actions=np.asarray([1, -1], dtype=np.int8),
        hold_minutes=100,
    )
    assert replay.direct_reversal_decisions == 1
    assert replay.result.completed_round_trips == 2
    assert [fill.side for fill in replay.result.fills] == [1, -1, -1, 1]
    assert replay.result.fills[1].timestamp_ns == replay.result.fills[2].timestamp_ns
    assert replay.result.net_pnl_cents == -3_200
    assert sum(replay.attributed_reward_cents) == replay.result.net_pnl_cents


def _joint_fixture():
    frame = _bars(pd.date_range("2024-01-02", periods=7, freq="5min", tz="UTC"))
    registered = registered_contract("base_paper_gc")
    specs = {
        "H1": replace(
            registered.agent("H1"), lookback=2, action_lock_bars=2,
        ),
        "M15": replace(
            registered.agent("M15"), lookback=2, action_lock_bars=2,
        ),
        "M5": replace(
            registered.agent("M5"), lookback=2, action_lock_bars=2,
        ),
    }
    frames = {name: frame.copy() for name in specs}
    trainers = {
        "H1": _RecordingTrainer((1, -1), spec=specs["H1"]),
        "M15": _RecordingTrainer((-1, 1), spec=specs["M15"]),
        "M5": _RecordingTrainer((0, 1), spec=specs["M5"]),
    }
    return specs, frames, trainers


def test_joint_hierarchy_uses_current_exploratory_parent_positions() -> None:
    specs, frames, trainers = _joint_fixture()
    reports = run_joint_paper_pass(trainers, frames)

    assert list(reports) == ["H1", "M15", "M5"]
    # At the first common completed-bar event, H1 publishes LONG, M15 observes
    # that current exploratory position and publishes SHORT, then M5 observes
    # both current positions. These are behavior-policy messages, not greedy
    # traces produced after a parent was trained to completion.
    assert trainers["M15"].selected_observations[0][-1] == 1
    np.testing.assert_array_equal(
        trainers["M5"].selected_observations[0][-2:],
        np.asarray([1, -1], dtype=np.float32),
    )
    for name, report in reports.items():
        expected = len(paper_origins(frames[name], specs[name]))
        assert report.transitions == expected == 4
        assert report.optimizer_updates == expected
        assert report.environment_steps == expected


def test_joint_stage_fails_closed_when_one_agent_has_zero_stage_updates() -> None:
    _, frames, trainers = _joint_fixture()
    trainers["M15"].update_result = None
    with pytest.raises(ValueError, match="M15.*zero optimizer update support"):
        run_joint_paper_pass(trainers, frames)


def _small_neural_spec(name: str):
    registered = registered_contract("base_paper_gc").agent(name)
    return replace(
        registered,
        lookback=2,
        action_lock_bars=2,
        replay_capacity=128,
        hidden_sizes=(8,),
    )


def _populate_trainer(trainer: PaperDQNTrainer, transitions: int = 64) -> None:
    observation = np.zeros(trainer.spec.observation_dimension, dtype=np.float32)
    for ordinal in range(transitions):
        if ordinal % trainer.spec.action_lock_bars == 0:
            trainer.act(observation, explore=True)
        trainer.append(observation, 0, 0.0, observation, terminal=False)
        trainer.update()
        trainer.advance_environment_step()


def test_production_training_has_seeded_random_initialization_and_no_parent_api(
    tmp_path,
) -> None:
    spec = _small_neural_spec("M5")
    first = PaperDQNTrainer(spec, seed=1103, device="cpu")
    second = PaperDQNTrainer(spec, seed=1103, device="cpu")
    for name, value in first.online.state_dict().items():
        torch.testing.assert_close(value, second.online.state_dict()[name])
        torch.testing.assert_close(value, first.target.state_dict()[name])
    assert len(first.replay) == 0
    assert len(first.optimizer.state) == 0
    assert first.environment_steps == first.action_selections == first.updates == 0
    assert exact.paper_effective_contract()["initialization"] == PAPER_INITIALIZATION
    assert exact.paper_effective_contract()["warm_start"] is False
    assert "pretrained_root" not in inspect.signature(exact.train_gc_from_scratch).parameters
    assert "pretrained_ensemble_root" not in inspect.signature(exact.train_five_seeds).parameters
    assert "load_pretrained_weights_only" not in exact.__all__
    assert len({exact._agent_stream_seed(1103, name) for name in ("H1", "M15", "M5")}) == 3
    with pytest.raises(ValueError, match="from-scratch"):
        first.save_checkpoint(
            tmp_path / "forbidden_xau.pt",
            stage="xau_pretrain",
            request_sha256="a" * 64,
        )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_sha256(request: dict[str, object]) -> str:
    payload = json.dumps(
        request, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _supported_request(marker: str) -> dict[str, object]:
    return {
        "synthetic_test": marker,
        "name": PAPER_EXACT_NAME,
        "stage": PAPER_TRAINING_STAGE,
        "seed": 1103,
        "initialization": PAPER_INITIALIZATION,
        "warm_start": False,
        "parent_manifest_sha256": None,
        "maximum_events_smoke_limit": 64,
        "source_digest": "d" * 64,
        "source_manifest_sha256": "e" * 64,
        "input_records_sha256": "f" * 64,
        "dates": ["2025-01-02"],
    }


def _write_supported_training_artifact(
    root, *, request: dict[str, object] | None = None,
) -> str:
    request = request or _supported_request("original")
    request_sha256 = _request_sha256(request)
    root.mkdir()
    reports = {}
    artifacts = []
    for name in ("H1", "M15", "M5"):
        trainer = PaperDQNTrainer(
            _small_neural_spec(name), seed=1103, device="cpu",
            stream_seed=exact._agent_stream_seed(1103, name),
        )
        _populate_trainer(trainer)
        report = {
            "transitions": 64,
            "optimizer_updates": trainer.updates,
            "action_renewals": trainer.action_selections,
            "environment_steps": trainer.environment_steps,
            "reward_sum": 0.0,
            "mean_loss": 0.0,
        }
        checkpoint = root / f"{name.lower()}.pt"
        trainer.save_checkpoint(
            checkpoint,
            stage=PAPER_TRAINING_STAGE,
            request_sha256=request_sha256,
            report=report,
        )
        reports[name] = report
        artifacts.append({"path": checkpoint.name, "sha256": _sha256(checkpoint)})
    manifest = {
        "schema_version": 1,
        "artifact_type": "BASEPAPER_DESCRIBED_TRAINING",
        "name": PAPER_EXACT_NAME,
        "complete": True,
        "stage": PAPER_TRAINING_STAGE,
        "seed": 1103,
        "initialization": PAPER_INITIALIZATION,
        "warm_start": False,
        "parent_manifest_sha256": None,
        "source_digest": request["source_digest"],
        "source_manifest_sha256": request["source_manifest_sha256"],
        "input_records_sha256": request["input_records_sha256"],
        "dates": request["dates"],
        "smoke_only": True,
        "maximum_events_smoke_limit": 64,
        "request": request,
        "request_sha256": request_sha256,
        "agents": reports,
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return request_sha256


def test_completed_checkpoint_cannot_be_reused_for_a_different_request_hash(
    tmp_path,
) -> None:
    root = tmp_path / "training"
    original_request = _write_supported_training_artifact(root)
    replacement_payload = _supported_request("replacement")
    replacement_request = _request_sha256(replacement_payload)
    assert verify_training_artifact(
        root, seed=1103, stage=PAPER_TRAINING_STAGE,
        request_sha256=original_request,
    )["complete"] is True
    with pytest.raises(ValueError, match="request hash differs"):
        verify_training_artifact(
            root, seed=1103, stage=PAPER_TRAINING_STAGE,
            request_sha256=replacement_request,
        )

    # Even if a caller rewrites the outer request, the bound checkpoints retain
    # their original request identity and must make reuse fail closed.
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request"] = replacement_payload
    manifest["request_sha256"] = replacement_request
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checkpoint identity/support differs"):
        verify_training_artifact(
            root, seed=1103, stage=PAPER_TRAINING_STAGE,
            request_sha256=replacement_request,
        )


def test_manifest_stage_support_must_match_each_bound_checkpoint(tmp_path) -> None:
    root = tmp_path / "training"
    request_sha256 = _write_supported_training_artifact(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["agents"]["H1"]["transitions"] = 63
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="report|support"):
        verify_training_artifact(
            root, seed=1103, stage=PAPER_TRAINING_STAGE,
            request_sha256=request_sha256,
        )


def test_production_artifact_rejects_any_parent_or_warm_start_lineage(tmp_path) -> None:
    root = tmp_path / "training"
    _write_supported_training_artifact(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_manifest_sha256"] = "a" * 64
    manifest["warm_start"] = True
    manifest["request"]["parent_manifest_sha256"] = "a" * 64
    manifest["request"]["warm_start"] = True
    manifest["request_sha256"] = _request_sha256(manifest["request"])
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity|request"):
        verify_training_artifact(
            root, seed=1103, stage=PAPER_TRAINING_STAGE,
        )


def test_actual_gc_smoke_stage_publishes_from_scratch_lineage(tmp_path) -> None:
    contract = registered_contract("base_paper_gc")
    frames = {}
    for name in ("H1", "M15", "M5"):
        spec = contract.agent(name)
        rows = spec.lookback + 64 + spec.action_lock_bars + 2
        frames[name] = _bars(
            pd.date_range(
                "2025-01-02", periods=rows, freq=f"{spec.minutes}min", tz="UTC",
            )
        )
    data = exact.LoadedPaperData(
        frames=frames,
        source_digest="d" * 64,
        source_manifest_sha256="e" * 64,
        dates=("2025-01-02",),
        phase="train",
        source_records=({"synthetic_smoke": True},),
        smoke_only=True,
    )
    manifest_path = exact._train_stage(
        data, tmp_path / "gc_from_scratch", 1103, maximum_events=64,
    )
    manifest = verify_training_artifact(
        manifest_path.parent, seed=1103, stage=PAPER_TRAINING_STAGE,
    )
    assert manifest["initialization"] == PAPER_INITIALIZATION
    assert manifest["warm_start"] is False
    assert manifest["parent_manifest_sha256"] is None
    assert manifest["request"]["parent_manifest_sha256"] is None
    assert "weights_only_xau_to_gc_reset" not in manifest
    assert all(row["transitions"] == 64 for row in manifest["agents"].values())
    assert all(row["optimizer_updates"] == 1 for row in manifest["agents"].values())


def _write_protocol_and_receipt(root):
    protocol_value = {
        "strategy_version": PAPER_EXACT_NAME,
        "z_format_independent": [3, 2, 1],
        "a_nested": {"right": True, "left": False},
    }
    protocol = root / "protocol.json"
    # Deliberately noncanonical whitespace/key order: the guard validator emits
    # the digest of parsed canonical JSON, not the raw file-byte SHA-256.
    protocol.write_text(json.dumps(protocol_value, indent=4) + "\n", encoding="utf-8")
    canonical = _request_sha256(protocol_value)
    assert _sha256(protocol) != canonical
    receipt = root / "freeze_receipt.json"
    receipt.write_text(
        json.dumps({"protocol_sha256": canonical, "validator_result": "PASS"}) + "\n",
        encoding="utf-8",
    )
    return protocol, receipt, canonical


def test_freeze_receipt_uses_validator_canonical_protocol_digest(tmp_path) -> None:
    protocol, receipt, canonical = _write_protocol_and_receipt(tmp_path)
    assert exact._verify_described_freeze(protocol, receipt) == canonical


def _prepared_data(*, source_digest: str, dates=("2024-01",)):
    frames = {}
    for name in ("H1", "M15", "M5"):
        frame = _bars(pd.date_range("2024-01-02", periods=20, freq="5min", tz="UTC"))
        frame["raw_symbol"] = "XAUUSD"
        frame["session_date"] = "2024-01-02"
        frames[name] = frame
    return exact.LoadedPaperData(
        frames=frames,
        source_digest=source_digest,
        source_manifest_sha256="c" * 64,
        dates=tuple(dates),
        phase="xau_pretrain",
        source_records=({"synthetic": True},),
    )


def test_prepared_xau_utility_cannot_enter_production_gc_training(
    tmp_path, monkeypatch,
) -> None:
    prepared = tmp_path / "prepared"
    exact._write_prepared(_prepared_data(source_digest="a" * 64), prepared)

    def unexpected_training(*args, **kwargs):
        raise AssertionError("capped prepared XAU reached model training")

    monkeypatch.setattr(exact, "_train_stage", unexpected_training)
    with pytest.raises(ValueError, match="phase differs"):
        exact.train_gc_from_scratch(prepared, tmp_path / "training", 1103)


def test_stale_prepared_output_rejects_changed_source_digest(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    exact._write_prepared(_prepared_data(source_digest="a" * 64), prepared)
    with pytest.raises(ValueError, match="request identity differs"):
        exact._write_prepared(_prepared_data(source_digest="b" * 64), prepared)


def test_capped_frames_retain_every_registered_t_plus_lock_reward_horizon() -> None:
    contract = registered_contract("base_paper_gc")
    frames = {}
    for name in ("H1", "M15", "M5"):
        spec = contract.agent(name)
        count = spec.lookback + 64 + spec.action_lock_bars + 5
        frames[name] = _bars(
            pd.date_range("2024-01-02", periods=count, freq=f"{spec.minutes}min", tz="UTC")
        )
    capped = exact._capped_frames(frames, 64)
    for name in ("H1", "M15", "M5"):
        spec = contract.agent(name)
        origins = paper_origins(capped[name], spec)
        assert len(origins) == 64
        assert int(origins[-1]) + spec.action_lock_bars < len(capped[name])


@pytest.mark.parametrize("phase", ["train", "validation"])
def test_wrong_frozen_gc_phase_manifest_hash_is_rejected_before_child_read(
    tmp_path, monkeypatch, phase,
) -> None:
    day = "2025-01-09" if phase == "train" else "2026-05-01"
    child = tmp_path / "must_not_be_read.parquet"
    manifest = tmp_path / f"wrong_{phase}.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "V6_GC_CAUSAL_SESSION_DATASET",
                "phase": phase,
                "classification": "DEVELOPMENT",
                "session_count": 1,
                "sessions": {
                    day: {
                        "raw_symbol": "GCG5",
                        "sources": {str(child): {"bytes": 1, "sha256": "d" * 64}},
                    }
                },
                "source_files": {str(child): {"bytes": 1, "sha256": "d" * 64}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a child product was opened before manifest rejection")
        ),
    )
    kwargs = {}
    if phase == "validation":
        protocol, receipt, _ = _write_protocol_and_receipt(tmp_path)
        kwargs = {"protocol": protocol, "freeze_receipt": receipt}
    with pytest.raises(ValueError, match="frozen described-BasePaper phase manifest"):
        exact.load_gc_timeframes(manifest, phase=phase, **kwargs)
