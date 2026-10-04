"""Trader-only DDQN training for the three registered CVP ablations."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
import torch

from kx_simple_hmadrl.environment import ExecutionEconomics
from kx_simple_hmadrl.volume_v6 import TraderQNetwork, parameter_count
from kx_simple_hmadrl.volume_v6_contract import (
    ACTIONS,
    MESSAGE_WIDTH,
    neutral_regime_message,
)
from kx_simple_hmadrl.volume_v6_env import V6TradingEnvironment, uses_legacy_arrival_boundary
from kx_simple_hmadrl.volume_v6_training import (
    DEFAULT_DDQN_CONFIG,
    V6DDQNTrainer,
    load_trainer_checkpoint,
    reconcile_raw_rewards,
    save_trainer_checkpoint,
    trader_forward,
)

from .artifacts import (
    atomic_json,
    atomic_torch_save,
    canonical_json_sha256,
    file_sha256,
    project_relative,
    read_json_object,
    relative_path,
    resolve_project_path,
    torch_state_sha256,
    verify_file_record,
)
from .contract import PROTOCOL_ID, PUBLISHER_PARENT_ARM, SEEDS, AblationArm, validate_arm
from .frozen_v6 import (
    FrozenPublisher,
    V6FeatureScaler,
    bottom_observation_at,
    build_session_features,
    latest_message,
    load_frozen_publisher,
    load_frozen_reference,
    load_npz_pair,
    publisher_messages,
    raw_index,
    valid_feature_indexes,
)
from .initialization import load_trader_initialization, verify_initialization
from .protocol import (
    AuthorizedDataset, assert_registered_parent_root, implementation_binding,
    load_authorized_dataset, protocol_binding, verify_implementation_binding,
)
from .representation import RepresentationMask


REGISTERED_PASSES = 3
REPLAY_CAPACITY = 120_000
REPLAY_WARMUP = 512
REWARD_DIVISOR = 1_000.0
REWARD_UNIT = "gc_ticks"
TRAINING_ARTIFACT = "CVP_ABLATION_TRADER_TRAINING"
TRADER_MODEL_ARTIFACT = "CVP_ABLATION_TRADER_MODEL"
TRAIN_DATASET_PATH = (
    "viability/artifacts/development/"
    "v6_registered_train_dataset_terminal_eligible/manifest.json"
)


def _atomic_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _reference_member(
    root: Path, *, arm: AblationArm, seed: int,
) -> FrozenPublisher:
    expected = PUBLISHER_PARENT_ARM[arm.value]
    # no_publisher still needs the frozen rich scaler as an exact observation
    # transform, but its publisher network must never be instantiated.
    parent_arm = "rich_volume" if expected is None else str(expected)
    assert_registered_parent_root(Path(root), parent_arm=parent_arm, seed=seed)
    loader = load_frozen_reference if expected is None else load_frozen_publisher
    return loader(Path(root), expected_parent_arm=parent_arm, expected_seed=seed)


def registered_trainer_settings(
    *, expected_decisions: int, passes: int = REGISTERED_PASSES,
) -> dict[str, object]:
    """Return the complete immutable trainer configuration for a job."""

    return {
        "algorithm": "dueling_double_dqn",
        "optimizer": "Adam",
        "loss": "Huber",
        "ddqn_config": asdict(DEFAULT_DDQN_CONFIG),
        "action_values": list(ACTIONS),
        "replay_capacity": REPLAY_CAPACITY,
        "replay_warmup": REPLAY_WARMUP,
        "replay_sampling": "uniform",
        "chronological_transition_ingestion": True,
        "chronological_session_passes": True,
        "passes": int(passes),
        "epsilon_decay_decisions": int(expected_decisions),
        "reward_divisor": REWARD_DIVISOR,
        "reward_unit": REWARD_UNIT,
        "reward_clipping": False,
        "legal_masked_behavior": True,
        "legal_masked_bootstrap": True,
        "device": "cpu",
    }


def _trainer_from_initialization(
    state: Mapping[str, torch.Tensor], *, arm: AblationArm, seed: int,
    expected_decisions: int,
) -> V6DDQNTrainer:
    model = TraderQNetwork()
    model.load_state_dict(state)
    return V6DDQNTrainer(
        f"trader_{arm.value}", model, seed=seed, replay_capacity=REPLAY_CAPACITY,
        replay_warmup=REPLAY_WARMUP,
        epsilon_decay_decisions=max(1, expected_decisions),
        reward_divisor=REWARD_DIVISOR, reward_unit=REWARD_UNIT,
        config=DEFAULT_DDQN_CONFIG, forward=trader_forward, device="cpu",
        protocol_id=PROTOCOL_ID,
    )


def _progress_identity(
    *, arm: AblationArm, seed: int, dataset: AuthorizedDataset,
    initialization_manifest: Path, initialization_sha256: str,
    reference: FrozenPublisher,
    smoke_only: bool, maximum_sessions: int | None, expected_decisions: int,
    passes: int,
) -> dict[str, object]:
    try:
        initialization_record = {
            "path": project_relative(initialization_manifest),
            "path_scope": "project_root",
            "sha256": file_sha256(initialization_manifest),
        }
    except ValueError:
        if not smoke_only:
            raise
        initialization_record = {
            "path": str(initialization_manifest.resolve()),
            "path_scope": "smoke_absolute",
            "sha256": file_sha256(initialization_manifest),
        }
    return {
        "schema_version": 1,
        "artifact_type": "CVP_ABLATION_TRAINING_PROGRESS",
        "experiment_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "arm": arm.value,
        "seed": seed,
        "passes_required": int(passes),
        "smoke_only": bool(smoke_only),
        "maximum_sessions": maximum_sessions,
        "dataset_manifest": dataset.binding(),
        "training_dates": list(dataset.dates),
        "input_digest": canonical_json_sha256({
            "manifest_sha256": dataset.manifest_sha256,
            "dates": list(dataset.dates),
        }),
        "initialization_manifest": initialization_record,
        "initialization_sha256": initialization_sha256,
        "reference_member_manifest": {
            "path": project_relative(reference.root / "manifest.json"),
            "path_scope": "project_root",
            "sha256": reference.manifest_sha256,
        },
        "publisher_parent_arm": PUBLISHER_PARENT_ARM[arm.value],
        "reference_scaler_sha256": reference.scaler.canonical_sha256(),
        "representation": RepresentationMask.for_arm(arm).registry_record(),
        "trainer_settings": registered_trainer_settings(
            expected_decisions=expected_decisions, passes=passes,
        ),
    }


def _make_observation(
    *, session: object, feature_index: int, scaler: V6FeatureScaler,
    representation: RepresentationMask, message: object, state: np.ndarray,
) -> dict[str, np.ndarray]:
    observation = bottom_observation_at(
        session, int(feature_index), scaler, message, state,
    )
    arrays = representation.trader_arrays(observation)
    if representation.message_mode == "exact_zero" and not np.array_equal(
        arrays["message"], np.zeros(MESSAGE_WIDTH, dtype=np.float32),
    ):
        raise AssertionError("no-publisher training consumed a nonzero message")
    return arrays


def train_trader(
    *, arm: str | AblationArm, seed: int, initialization_manifest: Path,
    reference_member_root: Path, output: Path,
    dataset_manifest: str = TRAIN_DATASET_PATH,
    smoke_only: bool = False, maximum_sessions: int | None = None,
    passes: int = REGISTERED_PASSES,
) -> Path:
    """Train/resume one trader; production calls are registry- and hash-bound."""

    selected_arm = validate_arm(arm)
    seed = int(seed)
    if seed not in SEEDS and not smoke_only:
        raise ValueError(f"registered ablation seed must be one of {SEEDS}")
    if passes != REGISTERED_PASSES and not smoke_only:
        raise ValueError("registered ablation training uses exactly three passes")
    if maximum_sessions is not None and not smoke_only:
        raise PermissionError("session caps are permitted only for SMOKE_ONLY jobs")
    dataset = load_authorized_dataset(
        dataset_manifest, purpose="train", period_id="development",
        maximum_sessions=maximum_sessions,
    )
    if not smoke_only and dataset.dates != tuple(dataset.dates):
        raise AssertionError("registered training date order changed")

    initialization_path = Path(initialization_manifest).resolve()
    initial_state, initialization = load_trader_initialization(
        initialization_path, expected_seed=seed, allow_smoke=smoke_only,
    )
    initialization_sha = str(initialization["trader_state_sha256"])
    if torch_state_sha256(initial_state) != initialization_sha:
        raise AssertionError("loaded trader initialization digest changed")
    reference = _reference_member(
        Path(reference_member_root), arm=selected_arm, seed=seed,
    )
    if not smoke_only and tuple(reference.manifest.get("training_dates", [])) != dataset.dates:
        raise ValueError("new ablation training dates differ from frozen parent dates")
    representation = RepresentationMask.for_arm(selected_arm)
    if representation.publisher_parent_arm != PUBLISHER_PARENT_ARM[selected_arm.value]:
        raise AssertionError("representation and publisher-source registries differ")

    destination = Path(output).resolve()
    if (destination / "manifest.json").is_file():
        verify_training_member(
            destination / "manifest.json", expected_arm=selected_arm.value,
            expected_seed=seed, allow_smoke=smoke_only,
        )
        return destination / "manifest.json"
    if destination.exists():
        raise FileExistsError("ablation training destination exists without a manifest")
    work = destination.with_name(f".{destination.name}.part")
    work.mkdir(parents=True, exist_ok=True)
    # A crash after the immutable manifest was written but before the directory
    # rename is a completed job, not a reason to retrain. Verify it in place,
    # discard only the named resumable scratch files, and finish the rename.
    if (work / "manifest.json").is_file():
        verify_training_member(
            work / "manifest.json", expected_arm=selected_arm.value,
            expected_seed=seed, allow_smoke=smoke_only,
        )
        (work / "progress.json").unlink(missing_ok=True)
        (work / "trader_resume.pt").unlink(missing_ok=True)
        os.replace(work, destination)
        return destination / "manifest.json"
    expected_decisions = max(1, len(dataset.pairs) * 59 * passes // 2)
    identity = _progress_identity(
        arm=selected_arm, seed=seed, dataset=dataset,
        initialization_manifest=initialization_path,
        initialization_sha256=initialization_sha, reference=reference,
        smoke_only=smoke_only, maximum_sessions=maximum_sessions,
        expected_decisions=expected_decisions, passes=passes,
    )
    progress_path = work / "progress.json"
    if progress_path.is_file():
        progress = read_json_object(progress_path)
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError("interrupted ablation training identity differs")
        trainer = _trainer_from_initialization(
            initial_state, arm=selected_arm, seed=seed,
            expected_decisions=expected_decisions,
        )
        resume = work / "trader_resume.pt"
        if int(progress.get("passes_complete", 0)) and not resume.is_file():
            raise ValueError("training progress exists without its resume checkpoint")
        if resume.is_file():
            metadata = load_trainer_checkpoint(trainer, resume)
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise ValueError("resume checkpoint identity differs")
            recorded_passes = int(progress.get("passes_complete", 0))
            checkpoint_passes = int(metadata.get("passes_complete", -1))
            if not recorded_passes <= checkpoint_passes <= passes:
                raise ValueError("resume checkpoint pass counter is inconsistent")
            progress["passes_complete"] = checkpoint_passes
            # The checkpoint rename can complete immediately before a crash and
            # the progress JSON rename. In that window checkpoint metadata is
            # the authoritative copy of all pass-level reports and audit flags.
            for key in (
                "pass_reports", "trainer_report", "unsafe_legacy_arrival_boundary",
            ):
                if key in metadata:
                    progress[key] = metadata[key]
    else:
        trainer = _trainer_from_initialization(
            initial_state, arm=selected_arm, seed=seed,
            expected_decisions=expected_decisions,
        )
        if torch_state_sha256(trainer.online.state_dict()) != initialization_sha:
            raise AssertionError("trader did not start from registered cloned initialization")
        progress = identity | {"passes_complete": 0, "pass_reports": []}
        atomic_json(progress_path, progress)

    scaler_path = work / "scaler.json"
    if not scaler_path.is_file():
        source_record = reference.manifest.get("scaler")
        source = verify_file_record(
            reference.root, source_record, require_within=reference.root,
        )
        _atomic_copy(source, scaler_path)
    scaler = V6FeatureScaler.load_json(scaler_path)
    if scaler.canonical_sha256() != reference.scaler.canonical_sha256():
        raise ValueError("copied ablation scaler differs from frozen reference")

    start_pass = int(progress.get("passes_complete", 0))
    unsafe_legacy_arrival_boundary = bool(
        progress.get("unsafe_legacy_arrival_boundary", False),
    )
    for pass_index in range(start_pass, passes):
        losses: list[float] = []
        transitions = skipped_sessions = round_trips = rejected = filled_sides = 0
        reward_total_cents = 0
        exact_zero_messages = 0
        for raw_path, side_path in dataset.pairs:
            raw, side = load_npz_pair(raw_path, side_path, phase="train")
            session = build_session_features(raw, side, scaler.calibration)
            selected = valid_feature_indexes(session, top=False)
            if not len(selected):
                skipped_sessions += 1
                continue
            unsafe_legacy_arrival_boundary |= uses_legacy_arrival_boundary(
                raw, ExecutionEconomics.from_basepaper(),
            )
            messages = (
                () if representation.publisher_parent_arm is None
                else publisher_messages(reference, session, representation)
            )
            env = V6TradingEnvironment(raw, raw_path.stem)
            session_rewards: list[int] = []
            for ordinal, feature_index_value in enumerate(selected):
                feature_index = int(feature_index_value)
                decision = int(session.bottom_clock_ns[feature_index])
                current_raw = raw_index(raw, decision)
                terminal = ordinal == len(selected) - 1
                if terminal:
                    next_feature_index = feature_index
                    next_raw = len(np.asarray(raw["clock_ns"])) - 1
                else:
                    next_feature_index = int(selected[ordinal + 1])
                    next_raw = raw_index(
                        raw, int(session.bottom_clock_ns[next_feature_index]),
                    )
                message = (
                    neutral_regime_message(decision)
                    if representation.publisher_parent_arm is None
                    else latest_message(messages, decision)
                )
                observation = _make_observation(
                    session=session, feature_index=feature_index, scaler=scaler,
                    representation=representation, message=message,
                    state=env.agent_state(current_raw),
                )
                if representation.message_mode == "exact_zero":
                    exact_zero_messages += 1
                legal = env.legal_actions()
                action = trainer.act(observation, legal=legal, explore=True)
                transition = env.step(current_raw, action, next_raw, terminal=terminal)
                if terminal:
                    following = {
                        name: np.asarray(value).copy() for name, value in observation.items()
                    }
                    following["state"] = np.asarray(
                        transition.next_state, dtype=np.float32,
                    )
                    next_legal = np.zeros(3, dtype=bool)
                else:
                    next_decision = int(session.bottom_clock_ns[next_feature_index])
                    next_message = (
                        neutral_regime_message(next_decision)
                        if representation.publisher_parent_arm is None
                        else latest_message(messages, next_decision)
                    )
                    following = _make_observation(
                        session=session, feature_index=next_feature_index, scaler=scaler,
                        representation=representation, message=next_message,
                        state=np.asarray(transition.next_state, dtype=np.float32),
                    )
                    next_legal = np.asarray(transition.next_legal_mask, dtype=bool)
                trainer.append(
                    observation, action, float(transition.reward_cents), following,
                    terminal=terminal, legal=legal, next_legal=next_legal,
                )
                loss = trainer.update()
                if loss is not None:
                    losses.append(loss)
                session_rewards.append(transition.reward_cents)
                transitions += 1
            result = env.finish()
            reconcile_raw_rewards(session_rewards, result.net_pnl_cents)
            reward_total_cents += result.net_pnl_cents
            round_trips += result.completed_round_trips
            rejected += result.rejected_commands
            filled_sides += len(result.fills)
        if unsafe_legacy_arrival_boundary and not smoke_only:
            raise ValueError("registered ablation training found legacy arrival boundaries")
        report = {
            "pass": pass_index + 1,
            "transitions": transitions,
            "skipped_sessions": skipped_sessions,
            "net_training_reward_cents": reward_total_cents,
            "completed_round_trips": round_trips,
            "filled_sides": filled_sides,
            "rejected_commands": rejected,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "exact_zero_messages": exact_zero_messages,
        }
        progress["passes_complete"] = pass_index + 1
        progress["pass_reports"] = list(progress.get("pass_reports", [])) + [report]
        progress["trainer_report"] = trainer.report()
        progress["unsafe_legacy_arrival_boundary"] = bool(
            unsafe_legacy_arrival_boundary
        )
        save_trainer_checkpoint(trainer, work / "trader_resume.pt", progress)
        atomic_json(progress_path, progress, immutable=False)

    trainer.audit_rewards()
    model_path = work / "trader_model.pt"
    final_state_sha = torch_state_sha256(trainer.online.state_dict())
    model_payload = {
        "schema_version": 1,
        "artifact_type": TRADER_MODEL_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "arm": selected_arm.value,
        "seed": seed,
        "initialization_sha256": initialization_sha,
        "trader_state_sha256": final_state_sha,
        "trader": trainer.online.state_dict(),
    }
    if model_path.is_file():
        existing_model = torch.load(model_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(existing_model, Mapping)
            or existing_model.get("artifact_type") != TRADER_MODEL_ARTIFACT
            or existing_model.get("arm") != selected_arm.value
            or int(existing_model.get("seed", -1)) != seed
            or existing_model.get("initialization_sha256") != initialization_sha
            or existing_model.get("trader_state_sha256") != final_state_sha
            or not isinstance(existing_model.get("trader"), Mapping)
            or torch_state_sha256(existing_model["trader"]) != final_state_sha
        ):
            raise ValueError("interrupted final trader model differs from resumed state")
    else:
        atomic_torch_save(model_path, model_payload)
    trainer_path = work / "trader_trainer.pt"
    if trainer_path.is_file():
        if file_sha256(trainer_path) != file_sha256(work / "trader_resume.pt"):
            raise ValueError("interrupted final trainer differs from resume checkpoint")
    else:
        _atomic_copy(work / "trader_resume.pt", trainer_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": TRAINING_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "arm": selected_arm.value,
        "seed": seed,
        "status": "COMPLETE",
        "phase": "train",
        "classification": "DEVELOPMENT",
        "passes": passes,
        "training_dates": list(dataset.dates),
        "dataset_manifest": dataset.binding(),
        "initialization_manifest": identity["initialization_manifest"],
        "initialization_sha256": initialization_sha,
        "reference_member_manifest": identity["reference_member_manifest"],
        "publisher_parent_arm": representation.publisher_parent_arm,
        "publisher_reused_same_seed": representation.publisher_parent_arm is not None,
        "publisher_frozen_before_trader": representation.publisher_parent_arm is not None,
        "publisher_training_permitted": False,
        "message_mode": representation.message_mode,
        "representation": representation.registry_record(),
        "scaler": {
            "path": scaler_path.name,
            "sha256": file_sha256(scaler_path),
            "canonical_sha256": scaler.canonical_sha256(),
        },
        "trader_model": {"path": model_path.name, "sha256": file_sha256(model_path)},
        "trader_state_sha256": final_state_sha,
        "trader_trainer": {
            "path": "trader_trainer.pt",
            "sha256": file_sha256(trainer_path),
        },
        "trader_parameters": parameter_count(trainer.online),
        "trainer": trainer.report(),
        "trainer_settings": identity["trainer_settings"],
        "pass_reports": progress.get("pass_reports", []),
        "chronological_transition_ingestion": True,
        "uniform_experience_replay": True,
        "same_training_budget_as_v6_trader": True,
        "smoke_only": bool(smoke_only),
        "holdout_read": False,
        "unsafe_legacy_arrival_boundary": bool(unsafe_legacy_arrival_boundary),
    }
    atomic_json(work / "manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    (work / "trader_resume.pt").unlink(missing_ok=True)
    os.replace(work, destination)
    verify_training_member(
        destination / "manifest.json", expected_arm=selected_arm.value,
        expected_seed=seed, allow_smoke=smoke_only,
    )
    return destination / "manifest.json"


def verify_training_member(
    manifest_path: Path, *, expected_arm: str | None = None,
    expected_seed: int | None = None, allow_smoke: bool = False,
) -> dict[str, object]:
    path = Path(manifest_path).resolve()
    value = read_json_object(path)
    arm = validate_arm(str(value.get("arm", "")))
    seed = int(value.get("seed", -1))
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != TRAINING_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("phase") != "train"
        or value.get("classification") != "DEVELOPMENT"
        or value.get("passes") != REGISTERED_PASSES
        or value.get("publisher_training_permitted") is not False
        or value.get("holdout_read") is not False
        or (value.get("smoke_only") is True and not allow_smoke)
    ):
        raise ValueError("ablation trader training manifest is invalid")
    if expected_arm is not None and arm.value != expected_arm:
        raise ValueError("ablation training arm differs")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError("ablation training seed differs")
    verify_implementation_binding(value.get("implementation"))
    if value.get("protocol") != protocol_binding():
        raise ValueError("ablation training protocol binding differs")
    if seed not in SEEDS and not allow_smoke:
        raise ValueError("ablation training uses an unregistered seed")
    representation = RepresentationMask.for_arm(arm)
    if (
        value.get("publisher_parent_arm") != PUBLISHER_PARENT_ARM[arm.value]
        or value.get("message_mode") != representation.message_mode
        or value.get("representation") != representation.registry_record()
    ):
        raise ValueError("ablation training publisher lineage differs")
    publisher_present = representation.publisher_parent_arm is not None
    if (
        value.get("publisher_reused_same_seed") is not publisher_present
        or value.get("publisher_frozen_before_trader") is not publisher_present
    ):
        raise ValueError("ablation publisher reuse/freeze attestation differs")
    initialization_record = value.get("initialization_manifest")
    if not isinstance(initialization_record, dict):
        raise ValueError("ablation initialization manifest binding is malformed")
    if initialization_record.get("path_scope") == "project_root":
        initialization_path = resolve_project_path(initialization_record["path"])
    elif allow_smoke and initialization_record.get("path_scope") == "smoke_absolute":
        initialization_path = Path(str(initialization_record["path"])).resolve()
    else:
        raise ValueError("ablation initialization path scope is invalid")
    if (
        not initialization_path.is_file()
        or file_sha256(initialization_path) != initialization_record.get("sha256")
    ):
        raise ValueError("ablation initialization manifest lineage differs")
    initialization = verify_initialization(
        initialization_path, expected_seed=seed, allow_smoke=allow_smoke,
    )
    if initialization.get("trader_state_sha256") != value.get("initialization_sha256"):
        raise ValueError("ablation member initialization tensor digest differs")
    dates = value.get("training_dates")
    dataset_record = value.get("dataset_manifest")
    if (
        not isinstance(dates, list)
        or not dates
        or dates != sorted(set(str(day) for day in dates))
        or not isinstance(dataset_record, dict)
        or dataset_record.get("path_scope") != "project_root"
        or dataset_record.get("path") != TRAIN_DATASET_PATH
        or dataset_record.get("purpose") != "train"
        or dataset_record.get("period_id") != "development"
        or dataset_record.get("dates") != dates
    ):
        raise ValueError("ablation member training dataset binding differs")
    dataset_path = resolve_project_path(dataset_record.get("path"))
    if not dataset_path.is_file() or file_sha256(dataset_path) != dataset_record.get(
        "sha256"
    ):
        raise ValueError("ablation member training dataset hash differs")
    dataset_value = read_json_object(dataset_path)
    dataset_sessions = dataset_value.get("sessions")
    if not isinstance(dataset_sessions, dict) or any(day not in dataset_sessions for day in dates):
        raise ValueError("ablation training dates are absent from the frozen dataset")
    smoke_only = bool(value.get("smoke_only"))
    if not smoke_only and sorted(dataset_sessions) != dates:
        raise ValueError("production ablation training did not use every registered session")
    scaler_path = verify_file_record(path.parent, value.get("scaler"), require_within=path.parent)
    scaler = V6FeatureScaler.load_json(scaler_path)
    scaler_record = value.get("scaler")
    assert isinstance(scaler_record, Mapping)
    if scaler.canonical_sha256() != scaler_record.get("canonical_sha256"):
        raise ValueError("ablation member scaler canonical digest differs")
    model_path = verify_file_record(
        path.parent, value.get("trader_model"), require_within=path.parent,
    )
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != TRADER_MODEL_ARTIFACT
        or payload.get("experiment_id") != PROTOCOL_ID
        or payload.get("arm") != arm.value
        or int(payload.get("seed", -1)) != seed
        or payload.get("initialization_sha256") != value.get("initialization_sha256")
        or not isinstance(payload.get("trader"), Mapping)
    ):
        raise ValueError("ablation trader model payload is invalid")
    digest = torch_state_sha256(payload["trader"])
    if digest != payload.get("trader_state_sha256") or digest != value.get(
        "trader_state_sha256"
    ):
        raise ValueError("ablation final trader tensor digest differs")
    if any(
        not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all())
        for tensor in payload["trader"].values()
    ):
        raise ValueError("ablation final trader state is nonfinite or malformed")
    audited_model = TraderQNetwork()
    audited_model.load_state_dict(payload["trader"], strict=True)
    if (
        parameter_count(audited_model) != int(value.get("trader_parameters", -1))
        or parameter_count(audited_model) != 21_144
    ):
        raise ValueError("ablation trader architecture/parameter count differs")
    verify_file_record(
        path.parent, value.get("trader_trainer"), require_within=path.parent,
    )
    parent = value.get("reference_member_manifest")
    if not isinstance(parent, dict):
        raise ValueError("ablation member lacks frozen-parent lineage")
    if parent.get("path_scope") != "project_root":
        raise ValueError("ablation frozen-parent path scope differs")
    parent_path = resolve_project_path(parent.get("path"))
    if not parent_path.is_file() or file_sha256(parent_path) != parent.get("sha256"):
        raise ValueError("ablation frozen-parent lineage changed")
    parent_arm = PUBLISHER_PARENT_ARM[arm.value]
    parent_arm = "rich_volume" if parent_arm is None else parent_arm
    assert_registered_parent_root(parent_path.parent, parent_arm=parent_arm, seed=seed)
    parent_manifest = read_json_object(parent_path)
    parent_scaler = parent_manifest.get("scaler")
    if (
        not isinstance(parent_scaler, dict)
        or parent_scaler.get("canonical_sha256") != scaler.canonical_sha256()
    ):
        raise ValueError("ablation scaler differs from its exact frozen parent")
    expected_decisions = max(1, len(dates) * 59 * REGISTERED_PASSES // 2)
    expected_settings = registered_trainer_settings(
        expected_decisions=expected_decisions, passes=REGISTERED_PASSES,
    )
    if value.get("trainer_settings") != expected_settings:
        raise ValueError("ablation trader settings differ from the registered V6 budget")
    if (
        value.get("chronological_transition_ingestion") is not True
        or value.get("uniform_experience_replay") is not True
        or value.get("same_training_budget_as_v6_trader") is not True
        or (value.get("unsafe_legacy_arrival_boundary") is True and not smoke_only)
    ):
        raise ValueError("ablation training-method attestation differs")
    reports = value.get("pass_reports")
    if not isinstance(reports, list) or len(reports) != REGISTERED_PASSES:
        raise ValueError("ablation pass report set is incomplete")
    total_transitions = total_reward = 0
    for pass_index, report in enumerate(reports, start=1):
        if (
            not isinstance(report, dict)
            or report.get("pass") != pass_index
            or not isinstance(report.get("transitions"), int)
            or int(report["transitions"]) < 1
            or not isinstance(report.get("skipped_sessions"), int)
            or not 0 <= int(report["skipped_sessions"]) <= len(dates)
            or not isinstance(report.get("net_training_reward_cents"), int)
            or not isinstance(report.get("completed_round_trips"), int)
            or not isinstance(report.get("filled_sides"), int)
            or int(report["filled_sides"]) != 2 * int(report["completed_round_trips"])
            or not isinstance(report.get("rejected_commands"), int)
            or int(report["rejected_commands"]) < 0
        ):
            raise ValueError("ablation pass report is malformed")
        expected_zero = int(report["transitions"]) if arm is AblationArm.NO_PUBLISHER else 0
        if report.get("exact_zero_messages") != expected_zero:
            raise ValueError("ablation pass message audit differs")
        mean_loss = report.get("mean_loss")
        if mean_loss is not None and not np.isfinite(float(mean_loss)):
            raise ValueError("ablation pass mean loss is nonfinite")
        total_transitions += int(report["transitions"])
        total_reward += int(report["net_training_reward_cents"])
    trainer = value.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("ablation trainer report is absent")
    action_counts = trainer.get("action_counts")
    if (
        trainer.get("name") != f"trader_{arm.value}"
        or int(trainer.get("seed", -1)) != seed
        or int(trainer.get("parameters", -1)) != int(value.get("trader_parameters", -1))
        or int(trainer.get("decisions", -1)) != total_transitions
        or int(trainer.get("transitions", -1)) != total_transitions
        or int(trainer.get("optimization_calls", -1)) != total_transitions
        or int(trainer.get("replay_capacity", -1)) != REPLAY_CAPACITY
        or int(trainer.get("replay_size", -1)) != min(total_transitions, REPLAY_CAPACITY)
        or float(trainer.get("reward_divisor", -1)) != REWARD_DIVISOR
        or trainer.get("reward_unit") != REWARD_UNIT
        or not np.isclose(float(trainer.get("raw_reward_total", np.nan)), total_reward,
                          rtol=0.0, atol=0.0)
        or not isinstance(action_counts, dict)
        or set(action_counts) != {str(action) for action in ACTIONS}
        or sum(int(count) for count in action_counts.values()) != total_transitions
    ):
        raise ValueError("ablation trainer counters/settings do not reconcile")
    expected_updates = max(0, total_transitions - REPLAY_WARMUP + 1)
    epsilon_fraction = min(1.0, total_transitions / expected_decisions)
    expected_epsilon = DEFAULT_DDQN_CONFIG.epsilon_start + epsilon_fraction * (
        DEFAULT_DDQN_CONFIG.epsilon_end - DEFAULT_DDQN_CONFIG.epsilon_start
    )
    if (
        int(trainer.get("updates", -1)) != expected_updates
        or int(trainer.get("target_sync_updates", -1))
        != DEFAULT_DDQN_CONFIG.target_sync_updates
        or int(trainer.get("target_synchronizations", -1))
        != expected_updates // DEFAULT_DDQN_CONFIG.target_sync_updates
        or not np.isclose(
            float(trainer.get("epsilon", np.nan)), expected_epsilon,
            rtol=0.0, atol=1e-15,
        )
    ):
        raise ValueError("ablation optimizer/target-sync counters do not reconcile")
    return value


def load_training_member(
    manifest_path: Path, *, expected_arm: str, expected_seed: int,
) -> tuple[TraderQNetwork, V6FeatureScaler, dict[str, object]]:
    manifest = verify_training_member(
        manifest_path, expected_arm=expected_arm, expected_seed=expected_seed,
    )
    path = Path(manifest_path).resolve()
    model_path = verify_file_record(
        path.parent, manifest["trader_model"], require_within=path.parent,
    )
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TraderQNetwork()
    model.load_state_dict(payload["trader"])
    model.eval()
    scaler_path = verify_file_record(
        path.parent, manifest["scaler"], require_within=path.parent,
    )
    return model, V6FeatureScaler.load_json(scaler_path), manifest


__all__ = [
    "REGISTERED_PASSES", "TRAINING_ARTIFACT", "TRAIN_DATASET_PATH",
    "TRADER_MODEL_ARTIFACT", "load_training_member", "train_trader",
    "registered_trainer_settings", "verify_training_member",
]
