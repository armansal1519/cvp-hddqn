"""Development-only training orchestration for the three Pilot 10 learners.

This module is deliberately narrower than an experiment runner.  It can open
only the exact registered development manifest through :mod:`protocol`, never
discovers evaluation data, and exports the final member for every registered
arm/seed rather than selecting a checkpoint from economic outcomes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from kx_cvp_ablations.artifacts import torch_state_sha256
from kx_cvp_ablations.frozen_v6 import (
    FrozenPublisher,
    V6FeatureScaler,
    bottom_observation_at,
    build_session_features,
    load_frozen_reference,
    load_npz_pair,
    load_xau_price_bundle,
    raw_index,
    top_observation_at,
    valid_feature_indexes,
)
from kx_simple_hmadrl.environment import ExecutionEconomics
from kx_simple_hmadrl.volume_v6 import parameter_count
from kx_simple_hmadrl.volume_v6_contract import (
    ACTIONS,
    CONTEXT_WIDTH,
    MESSAGE_WIDTH,
    TOP_BASE_WIDTH,
    TOP_HISTORY,
    TOP_RICH_WIDTH,
    neutral_regime_message,
)
from kx_simple_hmadrl.volume_v6_data import SessionFeatures, TraderObservation
from kx_simple_hmadrl.volume_v6_env import V6TradingEnvironment, uses_legacy_arrival_boundary
from kx_simple_hmadrl.volume_v6_training import (
    load_trainer_checkpoint,
    reconcile_raw_rewards,
    save_trainer_checkpoint,
    seed_everything,
)

from .artifacts import (
    atomic_json,
    atomic_torch_save,
    canonical_json_sha256,
    file_sha256,
    project_relative,
    read_json_object,
    resolve_project_path,
    workspace_root,
)
from .contract import (
    CANONICAL_OUTPUT_ROOT,
    DATASET_MANIFESTS,
    DEVELOPMENT_PERIOD_ID,
    FLAT_ENV_INTERACTIONS_PER_SEED,
    LEARNER_ARMS,
    PROTOCOL_ID,
    REGISTERED_PASSES,
    REQUIRED_LEARNER_TRAINING_JOBS,
    SEEDS,
    SUPPLEMENTAL_UPDATES_BY_PASS,
    BaselineArm,
    validate_arm,
)
from .flat_ddqn import (
    FlatMultiClockObservation,
    FlatMultiClockQNetwork,
    apply_supplemental_replay_updates,
    assert_training_report_budget,
    audit_registered_parent_budget,
    create_trainer,
    registered_trainer_settings,
    transfer_price_encoders,
)
from .fqi import (
    FQICorpus,
    FQICorpusCollector,
    FQI_ITERATIONS,
    TRANSITIONS_PER_PASS,
    archive_corpus,
    fit_fitted_q,
    load_corpus,
    registered_fqi_settings,
)
from .ppo import (
    PPO_OBSERVATION_KEYS,
    PPO_OPTIMIZATION_CALLS_PER_SEED,
    PPO_OPTIMIZER_UPDATES_PER_SEED,
    create_ppo_trainer,
    load_ppo_model,
    registered_ppo_settings,
    restore_ppo_trainer,
    save_ppo_checkpoint,
    save_ppo_model,
)
from .protocol import (
    AuthorizedDataset,
    implementation_binding,
    load_authorized_dataset,
    protocol_binding,
    reject_forbidden_text,
    validate_workspace_output,
    verify_implementation_binding,
)


TRAINING_ARTIFACT = "CVP_BASELINE_LEARNER_TRAINING"
COMPLETE_TRAINING_ARTIFACT = "CVP_BASELINE_COMPLETE_LEARNER_TRAINING"
FLAT_MODEL_ARTIFACT = "CVP_BASELINE_FLAT_DDQN_MODEL"
TRAIN_DATASET_PATH = str(DATASET_MANIFESTS[DEVELOPMENT_PERIOD_ID]["path"])
RICH_REFERENCE_ROOT = (
    "viability/artifacts/development/"
    "v6_registered_training_terminal_eligible/rich_volume/seed_{seed}"
)
XAU_REFERENCE_ROOT = (
    "viability/artifacts/development/"
    "v6_registered_xau_pretraining/v6_price/seed_{seed}"
)
REGISTERED_DEVELOPMENT_SESSIONS = 331

# Process-local only: the first verification remains deep and hash-bound. A
# replay worker may then reuse the verified immutable member rather than
# reloading multi-gigabyte fitted-Q corpora for every scenario cell.
_VERIFIED_MEMBER_CACHE: dict[tuple[str, str, bool, str], dict[str, object]] = {}
REGISTERED_DECISIONS_PER_NOMINAL_SESSION = 59
# This is the frozen parent schedule, distinct from observed interactions.
FLAT_EPSILON_DECAY_DECISIONS = (
    REGISTERED_DEVELOPMENT_SESSIONS
    * REGISTERED_DECISIONS_PER_NOMINAL_SESSION
    * REGISTERED_PASSES
    // 2
)
if FLAT_EPSILON_DECAY_DECISIONS != 29_293:
    raise AssertionError("registered flat-DDQN epsilon schedule changed")


def _atomic_copy(source: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _file_record(path: Path, *, anchor: Path | None = None) -> dict[str, object]:
    return {
        "path": path.name if anchor is not None and path.parent == anchor else project_relative(path),
        "path_scope": "member" if anchor is not None and path.parent == anchor else "project_root",
        "sha256": file_sha256(path),
    }


def _verify_member_file(root: Path, record: object) -> Path:
    if not isinstance(record, Mapping) or record.get("path_scope") != "member":
        raise ValueError("member artifact record is malformed")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ValueError("member artifact path is not a direct portable child")
    path = root / relative
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("member artifact hash differs")
    return path


def _rich_reference(seed: int) -> FrozenPublisher:
    return load_frozen_reference(
        workspace_root() / RICH_REFERENCE_ROOT.format(seed=int(seed)),
        expected_parent_arm="rich_volume",
        expected_seed=int(seed),
    )


def _reference_binding(reference: FrozenPublisher) -> dict[str, object]:
    return {
        "path": project_relative(reference.root / "manifest.json"),
        "path_scope": "project_root",
        "sha256": reference.manifest_sha256,
        "parent_arm": reference.parent_arm,
        "seed": reference.seed,
        "scaler_canonical_sha256": reference.scaler.canonical_sha256(),
    }


def _load_xau(seed: int) -> tuple[object, dict[str, object]]:
    root = workspace_root() / XAU_REFERENCE_ROOT.format(seed=int(seed))
    models, metadata, manifest = load_xau_price_bundle(root, expected_seed=int(seed))
    manifest_path = root / "manifest.json"
    model_record = manifest.get("model_bundle")
    if not isinstance(model_record, Mapping):
        raise ValueError("XAU warm start lacks its model record")
    binding = {
        "manifest": {
            "path": project_relative(manifest_path),
            "path_scope": "project_root",
            "sha256": file_sha256(manifest_path),
        },
        "model_bundle": {
            "path": str(model_record.get("path")),
            "sha256": str(model_record.get("sha256")),
        },
        "seed": int(seed),
        "gc_transfer_eligible": manifest.get("gc_transfer_eligible") is True,
        "gc_outcomes_accessed": manifest.get("gc_outcomes_accessed") is True,
        "stage": metadata.get("stage"),
        "transferred_tensors": "price_base_encoders_only",
    }
    if binding["gc_outcomes_accessed"] is not False:
        raise ValueError("XAU warm start accessed GC outcomes")
    return models, binding


def _assert_lineage(dataset: AuthorizedDataset, reference: FrozenPublisher, *, smoke_only: bool) -> None:
    calibration_dates = tuple(reference.scaler.calibration.training_dates)
    parent_dates = tuple(str(day) for day in reference.manifest.get("training_dates", ()))
    if smoke_only:
        if not set(dataset.dates).issubset(parent_dates) or not set(dataset.dates).issubset(calibration_dates):
            raise ValueError("smoke dates are absent from the frozen scaler lineage")
        return
    if len(dataset.dates) != REGISTERED_DEVELOPMENT_SESSIONS:
        raise ValueError("production training requires exactly 331 development sessions")
    if parent_dates != dataset.dates or calibration_dates != dataset.dates:
        raise ValueError("training dates differ from the frozen rich/scaler lineage")


def make_zero_message_observation(
    session: SessionFeatures,
    feature_index: int,
    scaler: V6FeatureScaler,
    state: np.ndarray,
) -> TraderObservation:
    """Build the full rich trader state and replace advice with exact zeros."""

    decision = int(session.bottom_clock_ns[int(feature_index)])
    source = bottom_observation_at(
        session,
        int(feature_index),
        scaler,
        neutral_regime_message(decision),
        np.asarray(state, dtype=np.float32),
    )
    result = replace(source, message=np.zeros(MESSAGE_WIDTH, dtype=np.float32))
    arrays = result.arrays("rich_volume")
    if arrays["message"].dtype != np.float32 or not np.array_equal(
        arrays["message"], np.zeros(MESSAGE_WIDTH, dtype=np.float32),
    ):
        raise AssertionError("single-agent training consumed a publisher message")
    return result


def aligned_flat_observation(
    session: SessionFeatures,
    feature_index: int,
    scaler: V6FeatureScaler,
    bottom: TraderObservation,
) -> FlatMultiClockObservation:
    """Align the current scheduled top slot, representing gaps explicitly.

    The latest scheduled slot at or before the five-minute decision is the
    only eligible top input.  If it is absent, invalid, or belongs to another
    continuity segment, all top arrays/masks are explicit unavailable values;
    an older valid top is never carried through the gap.
    """

    bottom_index = int(feature_index)
    decision = int(session.bottom_clock_ns[bottom_index])
    segment = int(session.bottom_segment[bottom_index])
    clocks = np.asarray(session.top_clock_ns, dtype=np.int64)
    segments = np.asarray(session.top_segment, dtype=np.int64)
    valid = np.asarray(session.top_valid, dtype=np.bool_)
    at_or_before = np.flatnonzero(clocks <= decision)
    if len(at_or_before):
        top_index = int(at_or_before[-1])
        if bool(valid[top_index]) and int(segments[top_index]) == segment:
            return FlatMultiClockObservation.from_v6(
                top_observation_at(session, top_index, scaler),
                bottom,
                top_segment=segment,
                bottom_segment=segment,
            )
    # Construct missingness from current information only.  In particular, do
    # not inspect/search future top rows to decide which five-minute rows exist.
    return FlatMultiClockObservation(
        top_base=np.zeros((TOP_HISTORY, TOP_BASE_WIDTH), np.float32),
        top_rich=np.zeros((TOP_HISTORY, TOP_RICH_WIDTH), np.float32),
        top_sequence_mask=np.zeros(TOP_HISTORY, np.bool_),
        top_context=np.zeros(CONTEXT_WIDTH, np.float32),
        top_context_mask=np.zeros(CONTEXT_WIDTH, np.bool_),
        base=np.asarray(bottom.base, dtype=np.float32).copy(),
        rich=np.asarray(bottom.rich, dtype=np.float32).copy(),
        sequence_mask=np.asarray(bottom.sequence_mask, dtype=np.bool_).copy(),
        context=np.asarray(bottom.context, dtype=np.float32).copy(),
        context_mask=np.asarray(bottom.context_mask, dtype=np.bool_).copy(),
        levels=np.asarray(bottom.levels, dtype=np.float32).copy(),
        l2_engineered=np.asarray(bottom.l2_engineered, dtype=np.float32).copy(),
        l2_mask=np.asarray(bottom.l2_mask, dtype=np.bool_).copy(),
        state=np.asarray(bottom.state, dtype=np.float32).copy(),
        top_decision_time_ns=-1,
        decision_time_ns=decision,
        top_segment=-1,
        bottom_segment=segment,
        top_available=False,
    )


def _bottom_arrays(observation: TraderObservation) -> dict[str, np.ndarray]:
    arrays = observation.arrays("rich_volume")
    if not np.array_equal(arrays["message"], np.zeros(MESSAGE_WIDTH, np.float32)):
        raise AssertionError("zero-message invariant changed")
    return arrays


def _terminal_following(
    observation: Mapping[str, np.ndarray], next_state: object,
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value).copy() for name, value in observation.items()}
    result["state"] = np.asarray(next_state, dtype=np.float32).copy()
    return result


def _pass_replay(
    *,
    dataset: AuthorizedDataset,
    scaler: V6FeatureScaler,
    pass_index: int,
    arm: str,
    act: Callable[[Mapping[str, np.ndarray], np.ndarray], object],
    record: Callable[..., object],
) -> dict[str, object]:
    transitions = skipped = reward_total = round_trips = rejected = filled = 0
    zero_messages = top_available = top_unavailable = 0
    unsafe = False
    losses: list[float] = []
    economics = ExecutionEconomics.from_basepaper()
    for raw_path, side_path in dataset.pairs:
        raw, side = load_npz_pair(raw_path, side_path, phase="train")
        session = build_session_features(raw, side, scaler.calibration)
        selected = valid_feature_indexes(session, top=False)
        if not len(selected):
            skipped += 1
            continue
        unsafe |= uses_legacy_arrival_boundary(raw, economics)
        environment = V6TradingEnvironment(raw, raw_path.stem, economics=economics)
        session_rewards: list[int] = []
        for ordinal, feature_value in enumerate(selected):
            feature_index = int(feature_value)
            decision = int(session.bottom_clock_ns[feature_index])
            current_raw = raw_index(raw, decision)
            terminal = ordinal == len(selected) - 1
            if terminal:
                next_feature_index = feature_index
                next_raw = len(np.asarray(raw["clock_ns"])) - 1
            else:
                next_feature_index = int(selected[ordinal + 1])
                next_raw = raw_index(raw, int(session.bottom_clock_ns[next_feature_index]))
            bottom = make_zero_message_observation(
                session, feature_index, scaler, environment.agent_state(current_raw),
            )
            bottom_values = _bottom_arrays(bottom)
            zero_messages += 1
            if arm == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
                flat = aligned_flat_observation(session, feature_index, scaler, bottom)
                observation = flat.arrays()
                top_available += int(flat.top_available)
                top_unavailable += int(not flat.top_available)
            elif arm == BaselineArm.RECURRENT_PPO.value:
                observation = {name: bottom_values[name] for name in PPO_OBSERVATION_KEYS}
            else:
                observation = bottom_values
            legal = np.asarray(environment.legal_actions(), dtype=np.bool_)
            receipt = act(observation, legal)
            action = int(getattr(receipt, "action", receipt))
            transition = environment.step(current_raw, action, next_raw, terminal=terminal)
            if terminal:
                following = _terminal_following(observation, transition.next_state)
                next_legal = np.zeros(len(ACTIONS), dtype=np.bool_)
            else:
                next_bottom = make_zero_message_observation(
                    session,
                    next_feature_index,
                    scaler,
                    np.asarray(transition.next_state, dtype=np.float32),
                )
                next_bottom_values = _bottom_arrays(next_bottom)
                if arm == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
                    following = aligned_flat_observation(
                        session, next_feature_index, scaler, next_bottom,
                    ).arrays()
                elif arm == BaselineArm.RECURRENT_PPO.value:
                    following = {
                        name: next_bottom_values[name] for name in PPO_OBSERVATION_KEYS
                    }
                else:
                    following = next_bottom_values
                next_legal = np.asarray(transition.next_legal_mask, dtype=np.bool_)
            outcome = record(
                observation,
                receipt,
                float(transition.reward_cents),
                following,
                terminal=terminal,
                legal=legal,
                next_legal=next_legal,
                session_id=raw_path.stem,
                pass_index=int(pass_index),
                decision_ordinal=int(ordinal),
            )
            if outcome is not None:
                try:
                    losses.append(float(outcome))
                except (TypeError, ValueError):
                    pass
            session_rewards.append(int(transition.reward_cents))
            transitions += 1
        result = environment.finish()
        reconcile_raw_rewards(session_rewards, result.net_pnl_cents)
        reward_total += int(result.net_pnl_cents)
        round_trips += int(result.completed_round_trips)
        rejected += int(result.rejected_commands)
        filled += len(result.fills)
    return {
        "pass": int(pass_index),
        "sessions": len(dataset.pairs) - skipped,
        "skipped_sessions": skipped,
        "transitions": transitions,
        "net_training_reward_cents": reward_total,
        "completed_round_trips": round_trips,
        "filled_sides": filled,
        "rejected_commands": rejected,
        "mean_online_loss": float(np.mean(losses)) if losses else None,
        "exact_zero_messages": zero_messages,
        "top_available_observations": top_available,
        "top_unavailable_observations": top_unavailable,
        "unsafe_legacy_arrival_boundary": bool(unsafe),
    }


def _member_directory(output_root: Path | str, arm: str, seed: int) -> Path:
    return Path(output_root).resolve() / arm / f"seed_{int(seed)}"


def _resolve_output_root(
    value: Path | str, *, allow_temporary: bool = False,
) -> Path:
    return validate_workspace_output(value, allow_temporary=allow_temporary)


def _algorithm_settings(arm: str, *, expected_decisions: int) -> dict[str, object]:
    if arm == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
        return registered_trainer_settings(expected_decisions=expected_decisions)
    if arm == BaselineArm.RECURRENT_PPO.value:
        return registered_ppo_settings()
    return registered_fqi_settings()


def _reward_contract(arm: str) -> dict[str, object]:
    if arm == BaselineArm.LIGHTGBM_FITTED_Q.value:
        return {
            "environment_reward": "executable_liquidation_nlv_change",
            "raw_unit": "cents",
            "learner_transform": "identity",
            "learner_unit": "cents",
            "clipping": False,
            "shaping": False,
        }
    return {
        "environment_reward": "executable_liquidation_nlv_change",
        "raw_unit": "cents",
        "learner_transform": "raw_cents_div_1000",
        "learner_unit": "gc_ticks",
        "clipping": False,
        "shaping": False,
    }


def _progress_identity(
    *, arm: str, seed: int, dataset: AuthorizedDataset, reference: FrozenPublisher,
    xau_binding: Mapping[str, object] | None, smoke_only: bool,
    maximum_sessions: int | None, passes: int, expected_decisions: int,
    initialization_state_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "CVP_BASELINE_TRAINING_PROGRESS",
        "experiment_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "arm": arm,
        "seed": int(seed),
        "phase": "train",
        "classification": "DEVELOPMENT",
        "passes_required": int(passes),
        "smoke_only": bool(smoke_only),
        "maximum_sessions": maximum_sessions,
        "dataset_manifest": dataset.binding(),
        "training_dates": list(dataset.dates),
        "input_digest": canonical_json_sha256(dataset.binding()),
        "rich_reference": _reference_binding(reference),
        "xau_price_warm_start": None if xau_binding is None else dict(xau_binding),
        "initialization_state_sha256": initialization_state_sha256,
        "algorithm_settings": _algorithm_settings(
            arm, expected_decisions=expected_decisions,
        ),
        "reward_contract": _reward_contract(arm),
        "epsilon_decay_decisions_scheduled": (
            expected_decisions if arm == BaselineArm.FLAT_MULTICLOCK_DDQN.value else None
        ),
        "environment_interactions_observed_required": (
            FLAT_ENV_INTERACTIONS_PER_SEED if not smoke_only else None
        ),
        "holdout_read": False,
        "evaluation_accessed": False,
    }


def _make_flat_trainer(seed: int, xau_models: object, expected_decisions: int):
    # The seed call must immediately precede construction; warm-start loading
    # happened earlier and cannot perturb fresh GC tensor initialization.
    seed_everything(int(seed))
    model = FlatMultiClockQNetwork()
    transfer_price_encoders(xau_models, model)  # type: ignore[arg-type]
    digest = torch_state_sha256(model.state_dict())
    return create_trainer(
        model, seed=int(seed), expected_decisions=int(expected_decisions),
    ), digest


def _make_ppo_trainer(seed: int, xau_models: object):
    base_state = xau_models.trader.base.state_dict()  # type: ignore[attr-defined]
    trainer = create_ppo_trainer(
        seed=int(seed), authorized_xau_base_state=base_state, protocol_id=PROTOCOL_ID,
    )
    return trainer, torch_state_sha256(trainer.model.state_dict())


def _write_progress(path: Path, progress: Mapping[str, object]) -> None:
    atomic_json(path, progress, immutable=False)


def _require_pass_report(
    report: Mapping[str, object], *, smoke_only: bool, expected_sessions: int,
) -> None:
    if bool(report.get("unsafe_legacy_arrival_boundary")) and not smoke_only:
        raise ValueError("production training found legacy arrival boundaries")
    if int(report.get("skipped_sessions", -1)) != 0:
        raise ValueError("registered development replay skipped a session")
    if int(report.get("sessions", -1)) != int(expected_sessions):
        raise ValueError("development replay session count differs")
    if not smoke_only and int(report.get("transitions", -1)) != TRANSITIONS_PER_PASS:
        raise ValueError("registered pass must contain exactly 19,242 transitions")
    if int(report.get("exact_zero_messages", -1)) != int(report.get("transitions", -2)):
        raise ValueError("every economic transition must use an exact-zero message")


def _combine_fqi_parts(parts: Sequence[FQICorpus], *, seed: int, dates: tuple[str, ...]) -> FQICorpus:
    if not parts:
        raise ValueError("cannot combine an empty FQI corpus registry")
    fields = (
        "states", "actions", "rewards", "next_states", "terminals", "legal",
        "next_legal", "session_dates", "pass_indexes", "decision_ordinals",
    )
    return FQICorpus(
        seed=int(seed),
        training_dates=dates,
        **{
            name: np.concatenate([np.asarray(getattr(part, name)) for part in parts], axis=0)
            for name in fields
        },
    )


def _directory_files_record(directory: Path, *, member_root: Path) -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(member_root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return {
        "path": directory.relative_to(member_root).as_posix(),
        "files": files,
        "file_count": len(files),
        "aggregate_sha256": canonical_json_sha256(files),
    }


def _complete_fqi_attempt(directory: Path, corpus: FQICorpus) -> bool:
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return False
    value = read_json_object(manifest)
    return (
        value.get("arm") == BaselineArm.LIGHTGBM_FITTED_Q.value
        and int(value.get("seed", -1)) == corpus.seed
        and value.get("corpus_sha256") == corpus.canonical_sha256()
        and isinstance(value.get("iterations"), list)
        and len(value["iterations"]) == FQI_ITERATIONS
        and (directory / "model.txt").is_file()
        and file_sha256(directory / "model.txt") == value.get("final_model_sha256")
    )
def train_member(
    arm: str | BaselineArm,
    seed: int,
    output_root: Path | str = CANONICAL_OUTPUT_ROOT,
    *,
    resume: bool = True,
    smoke_only: bool = False,
    maximum_sessions: int | None = None,
    passes: int = REGISTERED_PASSES,
) -> Path:
    """Train or resume one registered learner on development data only.

    ``output_root`` is an aggregate root; the immutable member is written to
    ``<root>/<arm>/seed_<seed>``.  Smoke caps are deliberately marked and may
    not be written below the canonical production root.
    """

    selected = validate_arm(arm).value
    if selected not in LEARNER_ARMS:
        raise ValueError("train_member accepts only the three registered learners")
    one_seed = int(seed)
    if one_seed not in SEEDS:
        raise ValueError(f"learner seed must be one of {SEEDS}")
    if passes != REGISTERED_PASSES:
        raise ValueError("Pilot 10 learner training requires exactly three passes")
    if maximum_sessions is not None and not smoke_only:
        raise PermissionError("session caps are permitted only for SMOKE_ONLY training")
    root = _resolve_output_root(output_root, allow_temporary=smoke_only)
    canonical = (workspace_root() / CANONICAL_OUTPUT_ROOT).resolve()
    if smoke_only and (root == canonical or canonical in root.parents):
        raise PermissionError("SMOKE_ONLY artifacts cannot enter the canonical production root")
    dataset = load_authorized_dataset(
        TRAIN_DATASET_PATH,
        purpose="train",
        period_id=DEVELOPMENT_PERIOD_ID,
        maximum_sessions=maximum_sessions,
    )
    reference = _rich_reference(one_seed)
    _assert_lineage(dataset, reference, smoke_only=smoke_only)
    expected_decisions = (
        FLAT_EPSILON_DECAY_DECISIONS
        if not smoke_only
        else max(1, len(dataset.dates) * REGISTERED_DECISIONS_PER_NOMINAL_SESSION * passes // 2)
    )
    xau_models = None
    xau_binding: dict[str, object] | None = None
    if selected != BaselineArm.LIGHTGBM_FITTED_Q.value:
        xau_models, xau_binding = _load_xau(one_seed)

    if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
        if not smoke_only:
            audit_registered_parent_budget()
        trainer, initialization_digest = _make_flat_trainer(
            one_seed, xau_models, expected_decisions,
        )
        assert xau_binding is not None
        xau_binding["transferred_state_sha256"] = canonical_json_sha256({
            "top_base": torch_state_sha256(xau_models.regime.base.state_dict()),
            "bottom_base": torch_state_sha256(xau_models.trader.base.state_dict()),
        })
    elif selected == BaselineArm.RECURRENT_PPO.value:
        trainer, initialization_digest = _make_ppo_trainer(one_seed, xau_models)
        assert xau_binding is not None
        xau_binding["transferred_state_sha256"] = torch_state_sha256(
            xau_models.trader.base.state_dict(),
        )
    else:
        trainer = None
        initialization_digest = None

    identity = _progress_identity(
        arm=selected,
        seed=one_seed,
        dataset=dataset,
        reference=reference,
        xau_binding=xau_binding,
        smoke_only=smoke_only,
        maximum_sessions=maximum_sessions,
        passes=passes,
        expected_decisions=expected_decisions,
        initialization_state_sha256=initialization_digest,
    )
    destination = _member_directory(root, selected, one_seed)
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        verify_training_member(
            manifest_path,
            expected_arm=selected,
            expected_seed=one_seed,
            allow_smoke=smoke_only,
        )
        return manifest_path
    if destination.exists():
        raise FileExistsError("training member destination exists without a manifest")
    work = destination.with_name(f".{destination.name}.part")
    if work.exists() and not resume:
        raise FileExistsError("resumable training state exists but resume=False")
    if work.exists() and any(work.iterdir()) and not (
        (work / "progress.json").is_file() or (work / "manifest.json").is_file()
    ):
        raise FileExistsError("training work directory has no recognized resume identity")
    work.mkdir(parents=True, exist_ok=True)
    progress_path = work / "progress.json"
    if (work / "manifest.json").is_file():
        verify_training_member(
            work / "manifest.json", expected_arm=selected,
            expected_seed=one_seed, allow_smoke=smoke_only,
        )
        progress_path.unlink(missing_ok=True)
        (work / "trainer_resume.pt").unlink(missing_ok=True)
        os.replace(work, destination)
        return destination / "manifest.json"

    if progress_path.is_file():
        progress = read_json_object(progress_path)
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError("interrupted learner training identity differs")
    else:
        progress = identity | {
            "stage": "REPLAY",
            "passes_complete": 0,
            "pass_reports": [],
            "fqi_pass_corpora": [],
        }
        _write_progress(progress_path, progress)

    resume_path = work / "trainer_resume.pt"
    if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value and (
        int(progress["passes_complete"]) or resume_path.is_file()
    ):
        if not resume_path.is_file():
            raise ValueError("flat progress exists without its trainer checkpoint")
        assert trainer is not None
        metadata = load_trainer_checkpoint(trainer, resume_path)
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("flat resume checkpoint identity differs")
        recorded = int(progress["passes_complete"])
        checkpoint_pass = int(metadata.get("passes_complete", -1))
        if checkpoint_pass not in {recorded, recorded + 1}:
            raise ValueError("flat progress/checkpoint pass counters differ")
        if checkpoint_pass == recorded + 1:
            reports = metadata.get("pass_reports")
            if not isinstance(reports, list) or len(reports) != checkpoint_pass:
                raise ValueError("ahead flat checkpoint lacks authoritative pass reports")
            progress["passes_complete"] = checkpoint_pass
            progress["pass_reports"] = reports
            _write_progress(progress_path, progress)
    elif selected == BaselineArm.RECURRENT_PPO.value and (
        int(progress["passes_complete"]) or resume_path.is_file()
    ):
        if not resume_path.is_file():
            raise ValueError("PPO progress exists without its trainer checkpoint")
        assert xau_models is not None
        restored, metadata = restore_ppo_trainer(
            resume_path,
            authorized_xau_base_state=xau_models.trader.base.state_dict(),
            protocol_id=PROTOCOL_ID,
        )
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise ValueError("PPO resume checkpoint identity differs")
        recorded = int(progress["passes_complete"])
        checkpoint_pass = int(metadata.get("passes_complete", -1))
        if checkpoint_pass not in {recorded, recorded + 1}:
            raise ValueError("PPO progress/checkpoint pass counters differ")
        if checkpoint_pass == recorded + 1:
            reports = metadata.get("pass_reports")
            if not isinstance(reports, list) or len(reports) != checkpoint_pass:
                raise ValueError("ahead PPO checkpoint lacks authoritative pass reports")
            progress["passes_complete"] = checkpoint_pass
            progress["pass_reports"] = reports
            _write_progress(progress_path, progress)
        trainer = restored

    scaler_path = work / "scaler.json"
    if not scaler_path.is_file():
        source_record = reference.manifest.get("scaler")
        if not isinstance(source_record, Mapping):
            raise ValueError("rich reference lacks its scaler record")
        source = reference.root / str(source_record.get("path", ""))
        if not source.is_file() or file_sha256(source) != source_record.get("sha256"):
            raise ValueError("rich reference scaler hash differs")
        _atomic_copy(source, scaler_path)
    scaler = V6FeatureScaler.load_json(scaler_path)
    if scaler.canonical_sha256() != reference.scaler.canonical_sha256():
        raise ValueError("local scaler differs from frozen rich reference")

    start_pass = int(progress.get("passes_complete", 0))
    for pass_index in range(start_pass + 1, passes + 1):
        if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
            assert trainer is not None

            def flat_act(obs: Mapping[str, np.ndarray], legal: np.ndarray):
                return trainer.act(obs, legal=legal, explore=True)

            def flat_record(obs, receipt, reward, following, **kwargs):
                trainer.append(
                    obs, int(receipt), reward, following,
                    terminal=kwargs["terminal"], legal=kwargs["legal"],
                    next_legal=kwargs["next_legal"],
                )
                return trainer.update()

            report = _pass_replay(
                dataset=dataset, scaler=scaler, pass_index=pass_index, arm=selected,
                act=flat_act, record=flat_record,
            )
            _require_pass_report(
                report, smoke_only=smoke_only, expected_sessions=len(dataset.pairs),
            )
            supplemental = (
                0 if smoke_only else apply_supplemental_replay_updates(
                    trainer, pass_number=pass_index,
                )
            )
            report["supplemental_replay_updates"] = supplemental
            progress["passes_complete"] = pass_index
            progress["pass_reports"] = list(progress["pass_reports"]) + [report]
            checkpoint_metadata = identity | {
                "passes_complete": pass_index,
                "pass_reports": progress["pass_reports"],
            }
            save_trainer_checkpoint(trainer, resume_path, checkpoint_metadata)
        elif selected == BaselineArm.RECURRENT_PPO.value:
            assert trainer is not None
            active = trainer.begin_pass()
            if active != pass_index:
                raise AssertionError("PPO pass counter diverged")

            def ppo_act(obs: Mapping[str, np.ndarray], legal: np.ndarray):
                return trainer.act(obs, legal=legal, deterministic=False)

            def ppo_record(obs, receipt, reward, following, **kwargs):
                return trainer.record(
                    obs, receipt, reward, following,
                    terminal=kwargs["terminal"], legal=kwargs["legal"],
                    next_legal=kwargs["next_legal"],
                    session_id=kwargs["session_id"],
                )

            report = _pass_replay(
                dataset=dataset, scaler=scaler, pass_index=pass_index, arm=selected,
                act=ppo_act, record=ppo_record,
            )
            _require_pass_report(
                report, smoke_only=smoke_only, expected_sessions=len(dataset.pairs),
            )
            flushed = trainer.finish_pass()
            report["end_of_pass_rollout_flushed"] = flushed is not None
            progress["passes_complete"] = pass_index
            progress["pass_reports"] = list(progress["pass_reports"]) + [report]
            checkpoint_metadata = identity | {
                "passes_complete": pass_index,
                "pass_reports": progress["pass_reports"],
            }
            save_ppo_checkpoint(trainer, resume_path, checkpoint_metadata)
        else:
            collector = FQICorpusCollector(seed=one_seed, training_dates=dataset.dates)
            rng_state = progress.get("fqi_behavior_rng_state")
            if rng_state is not None:
                collector.behavior_policy.rng.bit_generator.state = rng_state  # type: ignore[assignment]

            def fqi_act(obs: Mapping[str, np.ndarray], legal: np.ndarray):
                return collector.act(obs, legal=legal)

            def fqi_record(obs, receipt, reward, following, **kwargs):
                collector.append(
                    obs, int(receipt), reward, following,
                    terminal=kwargs["terminal"], legal=kwargs["legal"],
                    next_legal=kwargs["next_legal"],
                    session_date=kwargs["session_id"],
                    pass_index=kwargs["pass_index"],
                    decision_ordinal=kwargs["decision_ordinal"],
                )
                return None

            report = _pass_replay(
                dataset=dataset, scaler=scaler, pass_index=pass_index, arm=selected,
                act=fqi_act, record=fqi_record,
            )
            _require_pass_report(
                report, smoke_only=smoke_only, expected_sessions=len(dataset.pairs),
            )
            corpus = collector.finalize(require_registered=False)
            part_path = work / f"corpus_pass_{pass_index:02d}.npz"
            if part_path.is_file():
                existing = load_corpus(part_path, require_registered=False)
                if existing.canonical_sha256() != corpus.canonical_sha256():
                    raise ValueError("orphaned FQI pass corpus differs on deterministic replay")
                part_record = {
                    "path": part_path.name,
                    "sha256": file_sha256(part_path),
                    "canonical_sha256": corpus.canonical_sha256(),
                    "rows": corpus.transitions,
                }
            else:
                part_record = archive_corpus(corpus, part_path, require_registered=False)
            part_record["path_scope"] = "member"
            progress["passes_complete"] = pass_index
            progress["pass_reports"] = list(progress["pass_reports"]) + [report]
            progress["fqi_pass_corpora"] = list(progress["fqi_pass_corpora"]) + [part_record]
            progress["fqi_behavior_rng_state"] = collector.behavior_policy.rng.bit_generator.state
        _write_progress(progress_path, progress)

    pass_reports = progress.get("pass_reports")
    if not isinstance(pass_reports, list) or len(pass_reports) != passes:
        raise ValueError("training progress lacks one report per chronological pass")
    total_transitions = sum(int(report["transitions"]) for report in pass_reports)
    if not smoke_only and total_transitions != FLAT_ENV_INTERACTIONS_PER_SEED:
        raise ValueError("production learner did not observe exactly 57,726 interactions")

    if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
        assert trainer is not None
        trainer.audit_rewards()
        trainer_report = trainer.report()
        if not smoke_only:
            assert_training_report_budget(trainer_report)
            if int(trainer_report.get("decisions", -1)) != FLAT_ENV_INTERACTIONS_PER_SEED:
                raise ValueError("flat decisions differ from observed interactions")
            if int(trainer_report.get("optimization_calls", -1)) != 75_246:
                raise ValueError("flat optimization-call budget differs")
        state = trainer.online.state_dict()
        model_path = work / "model.pt"
        model_payload = {
            "schema_version": 1,
            "artifact_type": FLAT_MODEL_ARTIFACT,
            "experiment_id": PROTOCOL_ID,
            "arm": selected,
            "seed": one_seed,
            "parameters": parameter_count(trainer.online),
            "state_sha256": torch_state_sha256(state),
            "model": state,
        }
        if not model_path.exists():
            atomic_torch_save(model_path, model_payload)
        else:
            existing = torch.load(model_path, map_location="cpu", weights_only=False)
            if (
                not isinstance(existing, Mapping)
                or existing.get("artifact_type") != FLAT_MODEL_ARTIFACT
                or existing.get("state_sha256") != model_payload["state_sha256"]
                or not isinstance(existing.get("model"), Mapping)
                or torch_state_sha256(existing["model"]) != model_payload["state_sha256"]
            ):
                raise ValueError("interrupted flat final model differs from resumed trainer")
        model_record = _file_record(model_path, anchor=work) | {
            "state_sha256": model_payload["state_sha256"],
        }
        training_record: dict[str, object] = {"trainer": trainer_report}
    elif selected == BaselineArm.RECURRENT_PPO.value:
        assert trainer is not None
        trainer_report = trainer.report()
        if not smoke_only and (
            not trainer.training_complete
            or int(trainer_report["optimization_calls"]) != PPO_OPTIMIZATION_CALLS_PER_SEED
            or int(trainer_report["updates"]) != PPO_OPTIMIZER_UPDATES_PER_SEED
        ):
            raise ValueError("PPO training budget differs from the registered schedule")
        model_path = work / "model.pt"
        if not model_path.exists():
            if smoke_only:
                atomic_torch_save(model_path, {
                    "schema_version": 1,
                    "artifact_type": "CVP_BASELINE_RECURRENT_PPO_SMOKE_MODEL",
                    "arm": selected,
                    "protocol_id": PROTOCOL_ID,
                    "seed": one_seed,
                    "parameters": parameter_count(trainer.model),
                    "model": trainer.model.state_dict(),
                })
            else:
                save_ppo_model(trainer, model_path, {"training_complete": True})
        else:
            if smoke_only:
                existing = torch.load(model_path, map_location="cpu", weights_only=False)
                if (
                    not isinstance(existing, Mapping)
                    or existing.get("artifact_type") != "CVP_BASELINE_RECURRENT_PPO_SMOKE_MODEL"
                    or not isinstance(existing.get("model"), Mapping)
                    or torch_state_sha256(existing["model"])
                    != torch_state_sha256(trainer.model.state_dict())
                ):
                    raise ValueError("interrupted PPO smoke model differs")
            else:
                existing_model, existing_seed, _ = load_ppo_model(
                    model_path, protocol_id=PROTOCOL_ID,
                )
                if (
                    existing_seed != one_seed
                    or torch_state_sha256(existing_model.state_dict())
                    != torch_state_sha256(trainer.model.state_dict())
                ):
                    raise ValueError("interrupted PPO final model differs from resumed trainer")
        model_record = _file_record(model_path, anchor=work) | {
            "state_sha256": torch_state_sha256(trainer.model.state_dict()),
        }
        training_record = {
            "trainer": trainer_report,
            "rollout_updates": [asdict(report) for report in trainer.update_reports],
        }
    else:
        corpus_records = progress.get("fqi_pass_corpora")
        if not isinstance(corpus_records, list) or len(corpus_records) != passes:
            raise ValueError("FQI progress lacks its per-pass corpus registry")
        parts = []
        for record in corpus_records:
            path = _verify_member_file(work, record)
            parts.append(load_corpus(
                path,
                expected_sha256=str(record["sha256"]),
                expected_canonical_sha256=str(record["canonical_sha256"]),
                require_registered=False,
            ))
        corpus = _combine_fqi_parts(parts, seed=one_seed, dates=dataset.dates)
        del parts
        if not smoke_only:
            corpus.validate_registered()
        attempts = sorted(work.glob("fqi_model_attempt_*"))
        completed = [path for path in attempts if _complete_fqi_attempt(path, corpus)]
        if len(completed) > 1:
            raise ValueError("multiple complete deterministic FQI attempts are ambiguous")
        abandoned = [
            _directory_files_record(path, member_root=work)
            for path in attempts if path not in completed
        ]
        if completed:
            fqi_dir = completed[0]
            result_record = read_json_object(fqi_dir / "manifest.json") | {
                "rows": corpus.transitions,
                "behavior_decisions": corpus.transitions,
            }
        else:
            fqi_dir = work / f"fqi_model_attempt_{len(attempts) + 1:02d}"
            result = fit_fitted_q(
                corpus,
                require_registered_corpus=not smoke_only,
                archive_directory=fqi_dir,
            )
            result_record = result.audit_record() | {
                "rows": corpus.transitions,
                "behavior_decisions": corpus.transitions,
            }
        model_path = fqi_dir / "model.txt"
        model_record = {
            "path": model_path.relative_to(work).as_posix(),
            "path_scope": "member",
            "sha256": file_sha256(model_path),
            "state_sha256": file_sha256(model_path),
        }
        training_record = {
            "corpus": result_record,
            "corpus_audit": corpus.audit_record(),
            "fqi_archive_manifest": {
                "path": (fqi_dir / "manifest.json").relative_to(work).as_posix(),
                "path_scope": "member",
                "sha256": file_sha256(fqi_dir / "manifest.json"),
            },
            "fqi_pass_corpora": corpus_records,
            "abandoned_fqi_attempts": abandoned,
        }

    scaler_record = _file_record(scaler_path, anchor=work) | {
        "canonical_sha256": scaler.canonical_sha256(),
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": TRAINING_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "arm": selected,
        "seed": one_seed,
        "status": "COMPLETE",
        "phase": "train",
        "classification": "DEVELOPMENT",
        "passes": passes,
        "training_dates": list(dataset.dates),
        "dataset_manifest": dataset.binding(),
        "rich_reference": identity["rich_reference"],
        "xau_price_warm_start": identity["xau_price_warm_start"],
        "initialization_state_sha256": initialization_digest,
        "scaler": scaler_record,
        "model": model_record,
        "algorithm_settings": identity["algorithm_settings"],
        "reward_contract": identity["reward_contract"],
        "epsilon_decay_decisions_scheduled": identity["epsilon_decay_decisions_scheduled"],
        "environment_interactions_observed": total_transitions,
        "pass_reports": pass_reports,
        "training": training_record,
        "smoke_only": bool(smoke_only),
        "holdout_read": False,
        "evaluation_accessed": False,
        "selection_metric": None,
        "all_registered_members_retained": True,
    }
    atomic_json(work / "manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    resume_path.unlink(missing_ok=True)
    os.replace(work, destination)
    verify_training_member(
        destination / "manifest.json",
        expected_arm=selected,
        expected_seed=one_seed,
        allow_smoke=smoke_only,
    )
    return destination / "manifest.json"


def _resolve_member_record(root: Path, record: object) -> Path:
    if not isinstance(record, Mapping) or record.get("path_scope") != "member":
        raise ValueError("member file record is malformed")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("member file path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("member file escapes its member root") from error
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("member file hash differs")
    return path


def _verify_project_record(record: object) -> Path:
    if not isinstance(record, Mapping) or record.get("path_scope") != "project_root":
        raise ValueError("project-root artifact binding is malformed")
    path = resolve_project_path(record.get("path"))
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("project-root artifact hash differs")
    return path


def verify_training_member(
    manifest_path: Path | str,
    *,
    expected_arm: str | None = None,
    expected_seed: int | None = None,
    allow_smoke: bool = False,
) -> dict[str, object]:
    """Verify identity, lineage, exact budgets, and every member file hash."""

    reject_forbidden_text(manifest_path)
    path = Path(manifest_path).resolve()
    manifest_digest = file_sha256(path)
    implementation_digest = canonical_json_sha256(implementation_binding())
    cache_key = (
        path.as_posix(), manifest_digest, bool(allow_smoke), implementation_digest,
    )
    cached = _VERIFIED_MEMBER_CACHE.get(cache_key)
    if cached is not None:
        selected = str(cached["arm"])
        seed = int(cached["seed"])
        if expected_arm is not None and selected != str(expected_arm):
            raise ValueError("training member arm differs")
        if expected_seed is not None and seed != int(expected_seed):
            raise ValueError("training member seed differs")
        return cached
    value = read_json_object(path)
    selected = validate_arm(str(value.get("arm", ""))).value
    seed = int(value.get("seed", -1))
    smoke = bool(value.get("smoke_only"))
    if (
        selected not in LEARNER_ARMS
        or value.get("schema_version") != 1
        or value.get("artifact_type") != TRAINING_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("phase") != "train"
        or value.get("classification") != "DEVELOPMENT"
        or value.get("passes") != REGISTERED_PASSES
        or value.get("holdout_read") is not False
        or value.get("evaluation_accessed") is not False
        or value.get("selection_metric") is not None
        or value.get("all_registered_members_retained") is not True
        or (smoke and not allow_smoke)
    ):
        raise ValueError("learner training manifest identity differs")
    if expected_arm is not None and selected != str(expected_arm):
        raise ValueError("learner training arm differs")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError("learner training seed differs")
    if seed not in SEEDS:
        raise ValueError("learner training seed is not registered")
    if value.get("protocol") != protocol_binding():
        raise ValueError("learner training protocol binding differs")
    verify_implementation_binding(value.get("implementation"))

    dates = value.get("training_dates")
    dataset = value.get("dataset_manifest")
    if (
        not isinstance(dates, list)
        or not dates
        or dates != sorted(set(str(day) for day in dates))
        or not isinstance(dataset, Mapping)
        or dataset.get("path_scope") != "project_root"
        or dataset.get("path") != TRAIN_DATASET_PATH
        or dataset.get("purpose") != "train"
        or dataset.get("period_id") != DEVELOPMENT_PERIOD_ID
        or dataset.get("classification") != "DEVELOPMENT"
        or dataset.get("dates") != dates
    ):
        raise ValueError("learner development dataset binding differs")
    dataset_path = _verify_project_record(dataset)
    dataset_value = read_json_object(dataset_path)
    sessions = dataset_value.get("sessions")
    if not isinstance(sessions, Mapping) or any(day not in sessions for day in dates):
        raise ValueError("member dates are absent from the registered dataset")
    if not smoke and (len(dates) != REGISTERED_DEVELOPMENT_SESSIONS or sorted(sessions) != dates):
        raise ValueError("production member did not use all 331 registered dates")

    rich = value.get("rich_reference")
    rich_manifest_path = _verify_project_record(rich)
    rich_manifest = read_json_object(rich_manifest_path)
    if (
        not isinstance(rich, Mapping)
        or rich.get("parent_arm") != "rich_volume"
        or int(rich.get("seed", -1)) != seed
        or rich_manifest.get("arm") != "rich_volume"
        or int(rich_manifest.get("seed", -1)) != seed
    ):
        raise ValueError("rich scaler lineage differs")
    reference = load_frozen_reference(
        rich_manifest_path.parent,
        expected_parent_arm="rich_volume",
        expected_seed=seed,
    )
    if reference.scaler.canonical_sha256() != rich.get("scaler_canonical_sha256"):
        raise ValueError("rich scaler canonical digest differs")
    if not smoke and (
        tuple(reference.manifest.get("training_dates", ())) != tuple(dates)
        or tuple(reference.scaler.calibration.training_dates) != tuple(dates)
    ):
        raise ValueError("rich scaler dates differ from production dates")

    xau = value.get("xau_price_warm_start")
    if selected == BaselineArm.LIGHTGBM_FITTED_Q.value:
        if xau is not None or value.get("initialization_state_sha256") is not None:
            raise ValueError("fitted-Q must not use XAU initialization")
    else:
        if not isinstance(xau, Mapping):
            raise ValueError("neural learner lacks XAU initialization lineage")
        xau_manifest = xau.get("manifest")
        xau_path = _verify_project_record(xau_manifest)
        xau_models, _, xau_value = load_xau_price_bundle(
            xau_path.parent, expected_seed=seed,
        )
        model_binding = xau.get("model_bundle")
        if (
            xau.get("gc_transfer_eligible") is not True
            or xau.get("gc_outcomes_accessed") is not False
            or xau.get("transferred_tensors") != "price_base_encoders_only"
            or xau_value.get("gc_outcomes_accessed") is not False
            or not isinstance(model_binding, Mapping)
            or model_binding != xau_value.get("model_bundle")
            or not isinstance(value.get("initialization_state_sha256"), str)
        ):
            raise ValueError("XAU price-only initialization binding differs")
        if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
            initialized, initialization_sha = _make_flat_trainer(
                seed, xau_models, FLAT_EPSILON_DECAY_DECISIONS,
            )
            del initialized
            transfer_sha = canonical_json_sha256({
                "top_base": torch_state_sha256(xau_models.regime.base.state_dict()),
                "bottom_base": torch_state_sha256(xau_models.trader.base.state_dict()),
            })
        else:
            initialized, initialization_sha = _make_ppo_trainer(seed, xau_models)
            del initialized
            transfer_sha = torch_state_sha256(xau_models.trader.base.state_dict())
        if (
            initialization_sha != value.get("initialization_state_sha256")
            or transfer_sha != xau.get("transferred_state_sha256")
        ):
            raise ValueError("neural initialization tensors differ from XAU lineage")

    scaler_record = value.get("scaler")
    scaler_path = _resolve_member_record(path.parent, scaler_record)
    scaler = V6FeatureScaler.load_json(scaler_path)
    if (
        not isinstance(scaler_record, Mapping)
        or scaler.canonical_sha256() != scaler_record.get("canonical_sha256")
        or scaler.canonical_sha256() != reference.scaler.canonical_sha256()
    ):
        raise ValueError("member scaler differs from frozen reference")

    reports = value.get("pass_reports")
    if not isinstance(reports, list) or len(reports) != REGISTERED_PASSES:
        raise ValueError("member lacks three pass reports")
    for index, report in enumerate(reports, start=1):
        if not isinstance(report, Mapping) or int(report.get("pass", -1)) != index:
            raise ValueError("member pass reports are not chronological")
        _require_pass_report(
            report, smoke_only=smoke, expected_sessions=len(dates),
        )
        top_total = int(report.get("top_available_observations", -1)) + int(
            report.get("top_unavailable_observations", -1)
        )
        if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
            if top_total != int(report.get("transitions", -2)):
                raise ValueError("flat top availability does not cover every transition")
        elif top_total != 0:
            raise ValueError("bottom-only learner unexpectedly recorded top observations")
    total = sum(int(report["transitions"]) for report in reports)
    if int(value.get("environment_interactions_observed", -1)) != total:
        raise ValueError("member interaction count does not reconcile")
    if not smoke and total != FLAT_ENV_INTERACTIONS_PER_SEED:
        raise ValueError("member interaction budget differs")

    expected_decisions = (
        FLAT_EPSILON_DECAY_DECISIONS
        if not smoke else max(1, len(dates) * REGISTERED_DECISIONS_PER_NOMINAL_SESSION * REGISTERED_PASSES // 2)
    )
    if value.get("algorithm_settings") != _algorithm_settings(
        selected, expected_decisions=expected_decisions,
    ) or value.get("reward_contract") != _reward_contract(selected):
        raise ValueError("member algorithm settings differ")
    model_path = _resolve_member_record(path.parent, value.get("model"))
    training = value.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("member training record is malformed")
    if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        report = training.get("trainer")
        if (
            not isinstance(payload, Mapping)
            or payload.get("artifact_type") != FLAT_MODEL_ARTIFACT
            or payload.get("arm") != selected
            or int(payload.get("seed", -1)) != seed
            or not isinstance(payload.get("model"), Mapping)
            or torch_state_sha256(payload["model"]) != payload.get("state_sha256")
            or payload.get("state_sha256") != value["model"].get("state_sha256")
            or not isinstance(report, Mapping)
        ):
            raise ValueError("flat model/trainer record differs")
        model = FlatMultiClockQNetwork()
        model.load_state_dict(payload["model"], strict=True)
        if parameter_count(model) != int(payload.get("parameters", -1)):
            raise ValueError("flat parameter count differs")
        if not smoke:
            assert_training_report_budget(report)
            action_counts = report.get("action_counts")
            supplemental = [
                int(one.get("supplemental_replay_updates", -1)) for one in reports
            ]
            if (
                int(report.get("decisions", -1)) != FLAT_ENV_INTERACTIONS_PER_SEED
                or int(report.get("optimization_calls", -1)) != 75_246
                or int(report.get("target_synchronizations", -1)) != 747
                or int(report.get("replay_size", -1)) != FLAT_ENV_INTERACTIONS_PER_SEED
                or not np.isclose(float(report.get("epsilon", np.nan)), 0.05, rtol=0.0, atol=1e-15)
                or not isinstance(action_counts, Mapping)
                or sum(int(count) for count in action_counts.values())
                != FLAT_ENV_INTERACTIONS_PER_SEED
                or supplemental != list(SUPPLEMENTAL_UPDATES_BY_PASS)
                or int(report.get("updates", -1)) - sum(supplemental) != 57_215
                or int(value.get("epsilon_decay_decisions_scheduled", -1))
                != FLAT_EPSILON_DECAY_DECISIONS
            ):
                raise ValueError("flat exact training counters differ")
    elif selected == BaselineArm.RECURRENT_PPO.value:
        report = training.get("trainer")
        if not isinstance(report, Mapping):
            raise ValueError("PPO trainer record is malformed")
        if not smoke:
            model, loaded_seed, _ = load_ppo_model(model_path, protocol_id=PROTOCOL_ID)
            action_counts = report.get("action_counts")
            rollouts = training.get("rollout_updates")
            expected_rollout_steps = ([2_048] * 9 + [810]) * REGISTERED_PASSES
            if (
                loaded_seed != seed
                or parameter_count(model) != int(report.get("parameters", -1))
                or torch_state_sha256(model.state_dict())
                != value["model"].get("state_sha256")
            ):
                raise ValueError("PPO model identity differs")
            if (
                int(report.get("decisions", -1)) != FLAT_ENV_INTERACTIONS_PER_SEED
                or int(report.get("transitions", -1)) != FLAT_ENV_INTERACTIONS_PER_SEED
                or int(report.get("optimization_calls", -1)) != PPO_OPTIMIZATION_CALLS_PER_SEED
                or int(report.get("updates", -1)) != PPO_OPTIMIZER_UPDATES_PER_SEED
                or int(report.get("passes_complete", -1)) != REGISTERED_PASSES
                or report.get("active_pass") is not None
                or int(report.get("rollout_size", -1)) != 0
                or not isinstance(action_counts, Mapping)
                or sum(int(count) for count in action_counts.values())
                != FLAT_ENV_INTERACTIONS_PER_SEED
                or not isinstance(rollouts, list)
                or [int(one.get("rollout_steps", -1)) for one in rollouts]
                != expected_rollout_steps
                or sum(int(one.get("minibatch_updates", -1)) for one in rollouts)
                != PPO_OPTIMIZER_UPDATES_PER_SEED
            ):
                raise ValueError("PPO exact training counters differ")
    else:
        corpus = training.get("corpus")
        corpus_audit = training.get("corpus_audit")
        archive_manifest = _resolve_member_record(
            path.parent, training.get("fqi_archive_manifest"),
        )
        archive = read_json_object(archive_manifest)
        iterations = corpus.get("iterations") if isinstance(corpus, Mapping) else None
        if (
            not isinstance(corpus, Mapping)
            or int(corpus.get("rows", -1)) != total
            or int(corpus.get("behavior_decisions", -1)) != total
            or not isinstance(iterations, list)
            or len(iterations) != FQI_ITERATIONS
            or archive.get("corpus_sha256") != corpus.get("corpus_sha256")
            or archive.get("config_sha256") != corpus.get("config_sha256")
            or archive.get("final_model_sha256") != corpus.get("final_model_sha256")
            or archive.get("iterations") != iterations
            or not isinstance(corpus_audit, Mapping)
            or int(corpus_audit.get("sessions", -1)) != len(dates)
            or int(corpus_audit.get("passes", -1)) != REGISTERED_PASSES
            or int(corpus_audit.get("transitions", -1)) != total
            or int(corpus_audit.get("terminal_transitions", -1))
            != len(dates) * REGISTERED_PASSES
            or corpus_audit.get("evaluation_accessed") is not False
            or corpus_audit.get("canonical_sha256") != corpus.get("corpus_sha256")
            or not isinstance(corpus_audit.get("action_counts"), Mapping)
            or sum(int(count) for count in corpus_audit["action_counts"].values()) != total
        ):
            raise ValueError("fitted-Q training record differs")
        if (
            str(corpus.get("final_model_sha256")) != file_sha256(model_path)
            or value["model"].get("state_sha256") != file_sha256(model_path)
        ):
            raise ValueError("fitted-Q final model hash differs")
        if iterations[-1].get("model_sha256") != corpus.get("final_model_sha256"):
            raise ValueError("fitted-Q iteration chain does not end at final model")
        previous_hash: str | None = None
        for index, record in enumerate(iterations, start=1):
            if (
                not isinstance(record, Mapping)
                or int(record.get("iteration", -1)) != index
                or not all(
                    isinstance(record.get(name), str) and len(str(record[name])) == 64
                    for name in ("target_sha256", "model_sha256", "chain_sha256")
                )
                or record.get("previous_model_sha256") != previous_hash
            ):
                raise ValueError("fitted-Q iteration hash record is incomplete")
            iteration_path = archive_manifest.parent / f"iteration_{index:02d}.txt"
            if (
                not iteration_path.is_file()
                or file_sha256(iteration_path) != record.get("model_sha256")
            ):
                raise ValueError("fitted-Q iteration model hash differs")
            expected_chain = canonical_json_sha256({
                "iteration": index,
                "target_sha256": record["target_sha256"],
                "model_sha256": record["model_sha256"],
                "previous_model_sha256": previous_hash,
                "corpus_sha256": corpus["corpus_sha256"],
                "config_sha256": corpus["config_sha256"],
            })
            if expected_chain != record.get("chain_sha256"):
                raise ValueError("fitted-Q iteration chain hash differs")
            previous_hash = str(record["model_sha256"])
        combined_path = archive_manifest.parent / "corpus.npz"
        combined = load_corpus(
            combined_path,
            expected_sha256=str(archive.get("corpus_file_sha256")),
            expected_canonical_sha256=str(corpus.get("corpus_sha256")),
            require_registered=not smoke,
        )
        if combined.transitions != total:
            raise ValueError("fitted-Q archived corpus row count differs")
        del combined
        part_records = training.get("fqi_pass_corpora")
        if not isinstance(part_records, list) or len(part_records) != REGISTERED_PASSES:
            raise ValueError("fitted-Q pass-corpus registry differs")
        for pass_index, record in enumerate(part_records, start=1):
            part_path = _resolve_member_record(path.parent, record)
            part = load_corpus(
                part_path,
                expected_sha256=str(record.get("sha256")),
                expected_canonical_sha256=str(record.get("canonical_sha256")),
                require_registered=False,
            )
            if (
                int(record.get("rows", -1)) != part.transitions
                or not np.all(part.pass_indexes == pass_index)
            ):
                raise ValueError("fitted-Q pass corpus identity differs")
        abandoned = training.get("abandoned_fqi_attempts")
        if not isinstance(abandoned, list):
            raise ValueError("fitted-Q abandoned-attempt ledger is absent")
        for attempt in abandoned:
            if not isinstance(attempt, Mapping) or not isinstance(attempt.get("files"), list):
                raise ValueError("fitted-Q abandoned-attempt record is malformed")
            files = attempt["files"]
            if (
                int(attempt.get("file_count", -1)) != len(files)
                or canonical_json_sha256(files) != attempt.get("aggregate_sha256")
            ):
                raise ValueError("fitted-Q abandoned-attempt digest differs")
            for record in files:
                _resolve_member_record(
                    path.parent,
                    {"path": record.get("path"), "path_scope": "member", "sha256": record.get("sha256")},
                )
        if not smoke and total != FLAT_ENV_INTERACTIONS_PER_SEED:
            raise ValueError("fitted-Q corpus is not registered size")
    _VERIFIED_MEMBER_CACHE[cache_key] = value
    return value


def load_training_member(
    manifest_path: Path | str,
    *,
    expected_arm: str,
    expected_seed: int,
) -> tuple[object, V6FeatureScaler, dict[str, object]]:
    """Load one verified inference policy plus its frozen rich scaler."""

    reject_forbidden_text(manifest_path)
    path = Path(manifest_path).resolve()
    value = verify_training_member(
        path, expected_arm=expected_arm, expected_seed=expected_seed,
    )
    model_path = _resolve_member_record(path.parent, value["model"])
    selected = str(value["arm"])
    if selected == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        model = FlatMultiClockQNetwork()
        model.load_state_dict(payload["model"], strict=True)
        policy: object = model.eval()
    elif selected == BaselineArm.RECURRENT_PPO.value:
        policy, _, _ = load_ppo_model(model_path, protocol_id=PROTOCOL_ID)
    else:
        from .fqi import load_fitted_q_policy

        policy = load_fitted_q_policy(
            model_path, seed=int(value["seed"]), expected_sha256=file_sha256(model_path),
        )
    scaler_path = _resolve_member_record(path.parent, value["scaler"])
    return policy, V6FeatureScaler.load_json(scaler_path), value


load_trained_policy = load_training_member


def build_complete_training_manifest(
    training_root: Path | str,
    output: Path | str | None = None,
) -> Path:
    """Create the immutable, complete 3-arm x 5-seed production registry."""

    reject_forbidden_text(training_root, output or "")
    root = _resolve_output_root(training_root)
    destination = root / "complete_training_manifest.json" if output is None else Path(output).resolve()
    if destination != root / "complete_training_manifest.json":
        raise ValueError("complete training manifest must reside at the registered root")
    if destination.is_file():
        verify_complete_training_manifest(destination)
        return destination
    records: list[dict[str, object]] = []
    shared_dataset: object | None = None
    shared_implementation: object | None = None
    for arm in LEARNER_ARMS:
        for seed in SEEDS:
            member = root / arm / f"seed_{seed}" / "manifest.json"
            value = verify_training_member(
                member, expected_arm=arm, expected_seed=seed,
            )
            if shared_dataset is None:
                shared_dataset = value["dataset_manifest"]
                shared_implementation = value["implementation"]
            elif (
                value["dataset_manifest"] != shared_dataset
                or value["implementation"] != shared_implementation
            ):
                raise ValueError("learner members do not share one dataset/implementation binding")
            records.append({
                "arm": arm,
                "seed": seed,
                "manifest": project_relative(member),
                "manifest_sha256": file_sha256(member),
                "status": "COMPLETE",
            })
    if len(records) != REQUIRED_LEARNER_TRAINING_JOBS:
        raise AssertionError("complete learner registry is not 15 jobs")
    manifest = {
        "schema_version": 1,
        "artifact_type": COMPLETE_TRAINING_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": shared_implementation,
        "dataset_manifest": shared_dataset,
        "required_jobs": REQUIRED_LEARNER_TRAINING_JOBS,
        "members": records,
        "status": "COMPLETE",
        "classification": "DEVELOPMENT",
        "all_registered_members_retained": True,
        "economic_selection_used": False,
        "holdout_read": False,
        "evaluation_accessed": False,
    }
    atomic_json(destination, manifest)
    verify_complete_training_manifest(destination)
    return destination


create_complete_training_manifest = build_complete_training_manifest


def verify_complete_training_manifest(path: Path | str) -> dict[str, object]:
    reject_forbidden_text(path)
    source = Path(path).resolve()
    value = read_json_object(source)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != COMPLETE_TRAINING_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol") != protocol_binding()
        or value.get("required_jobs") != REQUIRED_LEARNER_TRAINING_JOBS
        or value.get("status") != "COMPLETE"
        or value.get("classification") != "DEVELOPMENT"
        or value.get("all_registered_members_retained") is not True
        or value.get("economic_selection_used") is not False
        or value.get("holdout_read") is not False
        or value.get("evaluation_accessed") is not False
    ):
        raise ValueError("complete learner training manifest identity differs")
    verify_implementation_binding(value.get("implementation"))
    records = value.get("members")
    if not isinstance(records, list) or len(records) != REQUIRED_LEARNER_TRAINING_JOBS:
        raise ValueError("complete learner training manifest must contain 15 jobs")
    observed: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("complete learner member record is malformed")
        identity = (str(record.get("arm")), int(record.get("seed", -1)))
        if identity in observed:
            raise ValueError("complete learner manifest duplicates a job")
        observed.add(identity)
        member = resolve_project_path(record.get("manifest"))
        if not member.is_file() or file_sha256(member) != record.get("manifest_sha256"):
            raise ValueError("complete learner member hash differs")
        one = verify_training_member(
            member, expected_arm=identity[0], expected_seed=identity[1],
        )
        if (
            one.get("implementation") != value.get("implementation")
            or one.get("dataset_manifest") != value.get("dataset_manifest")
        ):
            raise ValueError("complete learner member shares neither dataset nor implementation")
    expected = {(arm, seed) for arm in LEARNER_ARMS for seed in SEEDS}
    if observed != expected:
        raise ValueError("complete learner training Cartesian product differs")
    return value


def _train_one_job(args: tuple[str, int, str]) -> str:
    arm, seed, root = args
    return str(train_member(arm, seed, root))


def train_all_learners(
    output_root: Path | str = CANONICAL_OUTPUT_ROOT,
    *,
    workers: int = 1,
) -> Path:
    """Train every registered member and then create the 15-job receipt."""

    count = int(workers)
    if count < 1:
        raise ValueError("workers must be positive")
    root = _resolve_output_root(output_root)
    jobs = [(arm, seed, str(root)) for arm in LEARNER_ARMS for seed in SEEDS]
    if count == 1:
        for job in jobs:
            _train_one_job(job)
    else:
        # Explicit opt-in only: each FQI corpus and flat replay is large.
        with ProcessPoolExecutor(max_workers=count) as executor:
            list(executor.map(_train_one_job, jobs))
    return build_complete_training_manifest(root)


train_all = train_all_learners


def training_status(
    output_root: Path | str = CANONICAL_OUTPUT_ROOT,
) -> tuple[dict[str, object], ...]:
    """Return result-blind artifact status for the exact 15 registered jobs."""

    root = _resolve_output_root(output_root)
    records: list[dict[str, object]] = []
    for arm in LEARNER_ARMS:
        for seed in SEEDS:
            member = root / arm / f"seed_{seed}" / "manifest.json"
            work = member.parent.with_name(f".{member.parent.name}.part")
            if member.is_file():
                value = verify_training_member(
                    member, expected_arm=arm, expected_seed=seed,
                )
                status = "COMPLETE"
                manifest_sha = file_sha256(member)
                passes_complete = int(value.get("passes", -1))
            elif (work / "progress.json").is_file():
                progress = read_json_object(work / "progress.json")
                if progress.get("arm") != arm or int(progress.get("seed", -1)) != seed:
                    raise ValueError("training progress identity differs from its job path")
                status = "IN_PROGRESS"
                manifest_sha = None
                passes_complete = int(progress.get("passes_complete", 0))
            else:
                status = "NOT_STARTED"
                manifest_sha = None
                passes_complete = 0
            records.append({
                "arm": arm,
                "seed": seed,
                "status": status,
                "passes_complete": passes_complete,
                "manifest": project_relative(member),
                "manifest_sha256": manifest_sha,
                "economic_result_read": False,
            })
    return tuple(records)


__all__ = [
    "COMPLETE_TRAINING_ARTIFACT",
    "FLAT_EPSILON_DECAY_DECISIONS",
    "FLAT_MODEL_ARTIFACT",
    "RICH_REFERENCE_ROOT",
    "TRAINING_ARTIFACT",
    "TRAIN_DATASET_PATH",
    "XAU_REFERENCE_ROOT",
    "aligned_flat_observation",
    "build_complete_training_manifest",
    "create_complete_training_manifest",
    "load_trained_policy",
    "load_training_member",
    "make_zero_message_observation",
    "train_all",
    "train_all_learners",
    "train_member",
    "training_status",
    "verify_complete_training_manifest",
    "verify_training_member",
]
