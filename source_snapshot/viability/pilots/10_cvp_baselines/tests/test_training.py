from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kx_simple_hmadrl.volume_v6_contract import (
    BOTTOM_BASE_WIDTH,
    BOTTOM_HISTORY,
    BOTTOM_RICH_WIDTH,
    CONTEXT_WIDTH,
    L2_ENGINEERED_WIDTH,
    L2_HISTORY,
    L2_LEVELS,
    L2_LEVEL_WIDTH,
    MESSAGE_WIDTH,
    STATE_WIDTH,
    TOP_BASE_WIDTH,
    TOP_HISTORY,
    TOP_RICH_WIDTH,
)
from kx_simple_hmadrl.volume_v6_data import RegimeObservation, TraderObservation

from kx_cvp_baselines import training
from kx_cvp_baselines.artifacts import file_sha256
from kx_cvp_baselines.artifacts import workspace_root
from kx_cvp_baselines.contract import CANONICAL_OUTPUT_ROOT, LEARNER_ARMS, PROTOCOL_ID, SEEDS
from kx_cvp_baselines.fqi import FQICorpus, FLAT_STATE_WIDTH, MESSAGE_SLICE
from kx_cvp_baselines.protocol import load_authorized_dataset


def _trader(decision: int, *, message_value: float = 1.0) -> TraderObservation:
    return TraderObservation(
        np.zeros((BOTTOM_HISTORY, BOTTOM_BASE_WIDTH), np.float32),
        np.zeros((BOTTOM_HISTORY, BOTTOM_RICH_WIDTH), np.float32),
        np.ones(BOTTOM_HISTORY, np.bool_),
        np.zeros(CONTEXT_WIDTH, np.float32),
        np.ones(CONTEXT_WIDTH, np.bool_),
        np.zeros((L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH), np.float32),
        np.zeros((L2_HISTORY, L2_ENGINEERED_WIDTH), np.float32),
        np.ones(L2_HISTORY, np.bool_),
        np.full(MESSAGE_WIDTH, message_value, np.float32),
        np.zeros(STATE_WIDTH, np.float32),
        int(decision),
    )


def _top(decision: int) -> RegimeObservation:
    return RegimeObservation(
        np.zeros((TOP_HISTORY, TOP_BASE_WIDTH), np.float32),
        np.zeros((TOP_HISTORY, TOP_RICH_WIDTH), np.float32),
        np.ones(TOP_HISTORY, np.bool_),
        np.zeros(CONTEXT_WIDTH, np.float32),
        np.ones(CONTEXT_WIDTH, np.bool_),
        int(decision),
    )


def test_flat_parent_epsilon_schedule_is_distinct_from_observed_interactions() -> None:
    assert training.FLAT_EPSILON_DECAY_DECISIONS == 29_293
    settings = training._algorithm_settings(
        "flat_multiclock_ddqn",
        expected_decisions=training.FLAT_EPSILON_DECAY_DECISIONS,
    )
    assert settings["epsilon_decay_decisions"] == 29_293
    assert settings["environment_interactions"] == 57_726
    assert settings["optimizer_updates_total"] == 74_735


def test_canonical_output_root_is_workspace_relative_not_caller_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert training._resolve_output_root(CANONICAL_OUTPUT_ROOT) == (
        workspace_root() / CANONICAL_OUTPUT_ROOT
    ).resolve()
    assert training._resolve_output_root("explicit-relative") == (
        workspace_root() / "explicit-relative"
    ).resolve()
    with pytest.raises(PermissionError, match="inside the workspace"):
        training._resolve_output_root(tmp_path / "outside-production")


def test_bottom_builder_replaces_neutral_validity_with_exact_float32_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _trader(10, message_value=1.0)
    monkeypatch.setattr(training, "bottom_observation_at", lambda *args: source)
    session = SimpleNamespace(bottom_clock_ns=np.asarray([10], dtype=np.int64))
    result = training.make_zero_message_observation(
        session, 0, object(), np.zeros(STATE_WIDTH, np.float32),
    )
    assert result.message.dtype == np.float32
    assert np.array_equal(result.message, np.zeros(MESSAGE_WIDTH, np.float32))
    # The immutable source observation was not modified in place.
    assert np.array_equal(source.message, np.ones(MESSAGE_WIDTH, np.float32))


def test_flat_alignment_uses_explicit_unavailable_then_latest_completed_top(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minute = 60 * 1_000_000_000
    session = SimpleNamespace(
        bottom_clock_ns=np.asarray([5 * minute, 15 * minute], np.int64),
        bottom_segment=np.asarray([7, 7], np.int64),
        top_clock_ns=np.asarray([15 * minute, 30 * minute], np.int64),
        top_segment=np.asarray([7, 7], np.int64),
        top_valid=np.asarray([True, True], np.bool_),
    )
    monkeypatch.setattr(
        training, "top_observation_at", lambda one, index, scaler: _top(
            int(one.top_clock_ns[index]),
        ),
    )
    unavailable = training.aligned_flat_observation(
        session, 0, object(), _trader(5 * minute, message_value=0.0),
    )
    available = training.aligned_flat_observation(
        session, 1, object(), _trader(15 * minute, message_value=0.0),
    )
    assert unavailable.top_available is False
    assert unavailable.top_decision_time_ns == -1
    assert not unavailable.top_sequence_mask.any()
    assert available.top_available is True
    assert available.top_decision_time_ns == 15 * minute
    assert available.top_segment == available.bottom_segment == 7


def test_flat_alignment_marks_invalid_latest_top_unavailable_without_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minute = 60 * 1_000_000_000
    session = SimpleNamespace(
        bottom_clock_ns=np.asarray([30 * minute], np.int64),
        bottom_segment=np.asarray([2], np.int64),
        top_clock_ns=np.asarray([15 * minute, 30 * minute], np.int64),
        top_segment=np.asarray([2, 2], np.int64),
        top_valid=np.asarray([True, False], np.bool_),
    )
    touched = False

    def forbidden(*args):
        nonlocal touched
        touched = True
        raise AssertionError("an older valid top row must not be read")

    monkeypatch.setattr(training, "top_observation_at", forbidden)
    observation = training.aligned_flat_observation(
        session, 0, object(), _trader(30 * minute, message_value=0.0),
    )
    assert observation.top_available is False
    assert observation.top_decision_time_ns == -1
    assert not observation.top_sequence_mask.any()
    assert touched is False


def _corpus(seed: int, day: str, pass_index: int) -> FQICorpus:
    state = np.zeros((1, FLAT_STATE_WIDTH), np.float32)
    return FQICorpus(
        seed=seed,
        training_dates=(day,),
        states=state,
        actions=np.asarray([0], np.int8),
        rewards=np.asarray([3.0], np.float64),
        next_states=state.copy(),
        terminals=np.asarray([True], np.bool_),
        legal=np.asarray([[True, True, True]], np.bool_),
        next_legal=np.asarray([[False, False, False]], np.bool_),
        session_dates=np.asarray([day], dtype="U10"),
        pass_indexes=np.asarray([pass_index], np.int8),
        decision_ordinals=np.asarray([0], np.int16),
    )


def test_fqi_pass_combination_preserves_seed_dates_and_chronology() -> None:
    combined = training._combine_fqi_parts(
        [_corpus(1103, "2025-01-02", 1), _corpus(1103, "2025-01-02", 2)],
        seed=1103,
        dates=("2025-01-02",),
    )
    assert combined.transitions == 2
    assert combined.seed == 1103
    assert combined.training_dates == ("2025-01-02",)
    assert combined.pass_indexes.tolist() == [1, 2]
    assert not combined.states[:, MESSAGE_SLICE].any()


def test_train_member_rejects_nonlearners_and_unregistered_seed_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    touched = False

    def forbidden(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("dataset loader should not be reached")

    monkeypatch.setattr(training, "load_authorized_dataset", forbidden)
    with pytest.raises(ValueError, match="only the three registered learners"):
        training.train_member("always_flat", 1103, tmp_path)
    with pytest.raises(ValueError, match="learner seed"):
        training.train_member("recurrent_ppo", 999, tmp_path)
    assert touched is False


def test_common_replay_smoke_uses_only_zero_message_development_transitions() -> None:
    dataset = load_authorized_dataset(
        training.TRAIN_DATASET_PATH,
        purpose="train",
        period_id="development",
        maximum_sessions=1,
    )
    reference = training._rich_reference(1103)
    observed = 0

    def act(observation, legal):
        assert np.asarray(legal, np.bool_)[1]
        assert not np.asarray(observation["message"]).any()
        return 0

    def record(observation, receipt, reward, following, **kwargs):
        nonlocal observed
        assert receipt == 0
        assert np.isfinite(reward)
        assert not np.asarray(observation["message"]).any()
        assert not np.asarray(following["message"]).any()
        if kwargs["terminal"]:
            assert not np.asarray(kwargs["next_legal"]).any()
        observed += 1

    report = training._pass_replay(
        dataset=dataset,
        scaler=reference.scaler,
        pass_index=1,
        arm="lightgbm_fitted_q",
        act=act,
        record=record,
    )
    assert observed == report["transitions"] > 0
    assert report["sessions"] == 1
    assert report["skipped_sessions"] == 0
    assert report["exact_zero_messages"] == report["transitions"]
    assert report["unsafe_legacy_arrival_boundary"] is False


def test_complete_manifest_rejects_anything_but_exact_fifteen_jobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = {"test": True}
    implementation = {"aggregate_sha256": "a" * 64}
    monkeypatch.setattr(training, "protocol_binding", lambda: protocol)
    monkeypatch.setattr(training, "verify_implementation_binding", lambda value: None)
    value = {
        "schema_version": 1,
        "artifact_type": training.COMPLETE_TRAINING_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol": protocol,
        "implementation": implementation,
        "dataset_manifest": {"sha256": "d" * 64},
        "required_jobs": 15,
        "members": [],
        "status": "COMPLETE",
        "classification": "DEVELOPMENT",
        "all_registered_members_retained": True,
        "economic_selection_used": False,
        "holdout_read": False,
        "evaluation_accessed": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="15 jobs"):
        training.verify_complete_training_manifest(path)


def test_complete_manifest_accepts_exact_cartesian_product_and_shared_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = {"test": True}
    implementation = {"aggregate_sha256": "b" * 64}
    dataset = {"sha256": "d" * 64}
    members = []
    path_map: dict[str, Path] = {}
    for arm in LEARNER_ARMS:
        for seed in SEEDS:
            member = tmp_path / f"{arm}_{seed}.json"
            member.write_text("{}\n", encoding="utf-8")
            key = f"members/{arm}/{seed}"
            path_map[key] = member
            members.append({
                "arm": arm,
                "seed": seed,
                "manifest": key,
                "manifest_sha256": file_sha256(member),
                "status": "COMPLETE",
            })
    value = {
        "schema_version": 1,
        "artifact_type": training.COMPLETE_TRAINING_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol": protocol,
        "implementation": implementation,
        "dataset_manifest": dataset,
        "required_jobs": 15,
        "members": members,
        "status": "COMPLETE",
        "classification": "DEVELOPMENT",
        "all_registered_members_retained": True,
        "economic_selection_used": False,
        "holdout_read": False,
        "evaluation_accessed": False,
    }
    manifest = tmp_path / "complete.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(training, "protocol_binding", lambda: protocol)
    monkeypatch.setattr(training, "verify_implementation_binding", lambda one: None)
    monkeypatch.setattr(training, "resolve_project_path", lambda one: path_map[str(one)])
    monkeypatch.setattr(
        training,
        "verify_training_member",
        lambda *args, **kwargs: {
            "implementation": implementation,
            "dataset_manifest": dataset,
        },
    )
    checked = training.verify_complete_training_manifest(manifest)
    assert len(checked["members"]) == 15
