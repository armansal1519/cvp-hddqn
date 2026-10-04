"""End-to-end training and deterministic replay for the frozen V6 hierarchy."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np

from .environment import ExecutionEconomics
from .volume_v6 import V6ModelBundle, regime_message, transfer_base_encoders
from .volume_v6_contract import (
    ACTIONS, BOTTOM_MINUTES, SEEDS, TOP_MINUTES, RegimeMessage, neutral_regime_message,
    validate_arm,
)
from .volume_v6_data import (
    SessionFeatures, V6FeatureScaler, bottom_observation_at, build_session_features,
    load_npz_pair, top_observation_at,
)
from .volume_v6_env import (
    BiasRewardLabeler, V6TradingEnvironment, legal_target_mask,
    uses_legacy_arrival_boundary,
)
from .volume_v6_eval import (
    assert_test_unlocked, atomic_json, classify_dates, daily_metrics, file_sha256,
    flat_on_vote_tie, frozen_protocol,
)
from .volume_v6_training import (
    V6TrainerBundle, create_trainers, load_model_bundle, load_trainer_checkpoint,
    reconcile_raw_rewards, save_model_bundle, save_trainer_checkpoint,
)


REGISTERED_GC_PASSES = 3
TOP_LIFETIME_NS = TOP_MINUTES * 60 * 1_000_000_000


def scenario_economics(name: str) -> ExecutionEconomics:
    try:
        return {
            "base": ExecutionEconomics.from_basepaper(),
            "double_commission": ExecutionEconomics.from_basepaper(fee_multiplier=2),
            "20ms_route_latency": ExecutionEconomics.from_basepaper(route_latency_ms=20),
            "one_adverse_tick_per_filled_side": ExecutionEconomics.from_basepaper(
                adverse_ticks=1,
            ),
        }[name]
    except KeyError as error:
        raise ValueError("unknown V6 execution scenario") from error


def aligned_pairs(
    raw_paths: Sequence[Path], sidecar_paths: Sequence[Path], *, phase: str,
    maximum_sessions: int | None = None,
) -> tuple[tuple[Path, Path], ...]:
    raw = sorted(map(Path, raw_paths), key=lambda path: path.stem)
    side = sorted(map(Path, sidecar_paths), key=lambda path: path.stem)
    if not raw or [path.stem for path in raw] != [path.stem for path in side]:
        raise ValueError("V6 raw and sidecar session sets are absent or misaligned")
    dates = [path.stem for path in raw]
    access_phase = "smoke" if phase == "smoke" else phase
    classify_dates(dates, phase=access_phase)
    if maximum_sessions is not None:
        if phase != "smoke" or maximum_sessions < 1:
            raise ValueError("a positive session cap is allowed only for smoke")
        raw = raw[:maximum_sessions]
        side = side[:maximum_sessions]
    return tuple(zip(raw, side, strict=True))


def _exact_paths(
    pairs: Sequence[tuple[Path, Path]],
) -> tuple[tuple[Path, Path], ...]:
    return tuple((Path(raw).resolve(), Path(side).resolve()) for raw, side in pairs)


def _registered_training_dataset(
    pairs: Sequence[tuple[Path, Path]], dataset_manifest: Path | None,
) -> dict[str, object]:
    """Deep-verify and bind a registered training call to its exact phase build."""

    if dataset_manifest is None:
        raise PermissionError("registered V6 training requires the train dataset manifest")
    from .volume_v6_orchestration import load_phase_dataset
    dataset = load_phase_dataset(Path(dataset_manifest), phase="train")
    if _exact_paths(pairs) != _exact_paths(dataset.pairs):
        raise ValueError("V6 training paths differ from the registered train dataset")
    return {
        "path": str(dataset.manifest_path),
        "sha256": dataset.manifest_sha256,
    }


def _registered_evaluation_chain(
    pairs: Sequence[tuple[Path, Path]], checkpoint_roots: Sequence[Path], *,
    arm: str, phase: str, evaluation_dataset_manifest: Path | None,
    complete_training_manifest: Path | None, unlock_receipt: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Deep-verify evaluation data and the exact immutable checkpoint ensemble."""

    if evaluation_dataset_manifest is None or complete_training_manifest is None:
        raise PermissionError(
            "registered V6 evaluation requires dataset and complete-training manifests",
        )
    from .volume_v6_orchestration import (
        load_phase_dataset, verify_complete_training_manifest,
    )
    complete_path = Path(complete_training_manifest).resolve()
    complete = verify_complete_training_manifest(complete_path)
    dataset = load_phase_dataset(
        Path(evaluation_dataset_manifest), phase=phase,
        complete_training_manifest=complete_path if phase == "validation" else None,
        unlock_receipt=unlock_receipt,
    )
    if _exact_paths(pairs) != _exact_paths(dataset.pairs):
        raise ValueError("V6 evaluation paths differ from the registered phase dataset")
    public_arm = "V6-RichVolume" if arm == "rich_volume" else "V6-BaseOHLCV"
    records = complete.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("complete V6 training manifest has no checkpoint records")
    expected: list[tuple[int, Path]] = []
    for record in records:
        if not isinstance(record, dict) or record.get("arm") != public_arm:
            continue
        manifest = Path(str(record.get("manifest", "")))
        if not manifest.is_absolute():
            manifest = (complete_path.parent / manifest).resolve()
        expected.append((int(record.get("seed", -1)), manifest.parent))
    expected_roots = tuple(path for _, path in sorted(expected))
    supplied_roots = tuple(
        path for _, path in sorted(
            (int(json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))["seed"]),
             Path(root).resolve())
            for root in checkpoint_roots
        )
    )
    if expected_roots != supplied_roots:
        raise ValueError("V6 evaluation checkpoint roots differ from the complete receipt")
    return (
        {"path": str(dataset.manifest_path), "sha256": dataset.manifest_sha256},
        {"path": str(complete_path), "sha256": file_sha256(complete_path)},
    )


def _load_pair(
    pair: tuple[Path, Path], *, phase: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Smoke dates are a subset of the training-access boundary.
    data_phase = "train" if phase == "smoke" else phase
    return load_npz_pair(pair[0], pair[1], phase=data_phase)  # type: ignore[arg-type]


def _assert_registered_dataset_access(
    pairs: Sequence[tuple[Path, Path]], *, phase: str,
    unlock_receipt: Path | None = None,
) -> None:
    """Verify authorization before opening a registered validation/test shard."""

    if phase not in {"validation", "test"}:
        return
    if phase == "test":
        if unlock_receipt is None:
            raise PermissionError("registered test replay requires its unlock receipt")
        assert_test_unlocked(unlock_receipt)
    _, protocol_digest = frozen_protocol()
    manifests: dict[Path, dict[str, object]] = {}
    for raw, side in pairs:
        manifest_path = raw.parent.parent / "manifest.json"
        if side.parent.parent != raw.parent.parent or not manifest_path.is_file():
            raise PermissionError("registered V6 replay requires paired phase-built data")
        if manifest_path not in manifests:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            proof = value.get("access_authorization")
            expected_kind = (
                "test_unlock_receipt" if phase == "test"
                else "complete_training_manifest"
            )
            if (
                value.get("artifact_type") != "V6_GC_CAUSAL_SESSION_DATASET"
                or value.get("protocol_sha256") != protocol_digest
                or value.get("phase") != phase
                or value.get("classification") != (
                    "FRESH_OOS" if phase == "test" else "DEVELOPMENT"
                )
                or not isinstance(proof, dict)
                or proof.get("kind") != expected_kind
            ):
                raise PermissionError("V6 phase dataset lacks its frozen access proof")
            proof_path = Path(str(proof.get("path", "")))
            if not proof_path.is_absolute():
                proof_path = (manifest_path.parent / proof_path).resolve()
            if not proof_path.is_file() or file_sha256(proof_path) != proof.get("sha256"):
                raise PermissionError("V6 phase dataset access proof is absent or changed")
            if phase == "test":
                assert_test_unlocked(proof_path)
                if unlock_receipt is None or proof_path.resolve() != Path(unlock_receipt).resolve():
                    raise PermissionError("test dataset and replay use different unlock receipts")
            else:
                from .volume_v6_orchestration import verify_complete_training_manifest
                verify_complete_training_manifest(proof_path)
            manifests[manifest_path] = value
        record = manifests[manifest_path].get("sessions", {})
        if not isinstance(record, dict) or raw.stem not in record:
            raise ValueError("V6 phase manifest does not bind a requested session")
        one = record[raw.stem]
        if (
            not isinstance(one, dict)
            or one.get("session_sha256") != file_sha256(raw)
            or one.get("sidecar_sha256") != file_sha256(side)
        ):
            raise ValueError("V6 phase session hash differs from its manifest")


def _raw_index(raw: Mapping[str, np.ndarray], clock_ns: int) -> int:
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    index = int(np.searchsorted(clock, int(clock_ns), side="right") - 1)
    if index < 0 or int(clock[index]) != int(clock_ns):
        raise ValueError("feature decision clock is absent from the raw 250 ms grid")
    return index


def _valid_feature_indexes(session: SessionFeatures, *, top: bool) -> np.ndarray:
    valid = session.top_valid if top else session.bottom_valid
    segment = session.top_segment if top else session.bottom_segment
    indexes = np.flatnonzero(np.asarray(valid, dtype=bool))
    if not len(indexes):
        return indexes
    # A session with a mid-window source discontinuity is omitted from economic
    # rollouts rather than joining two distinct market states or forward-filling.
    if len(set(int(segment[index]) for index in indexes)) != 1:
        return np.empty(0, dtype=np.int64)
    return indexes.astype(np.int64)


def _top_examples(
    raw: Mapping[str, np.ndarray], session: SessionFeatures,
    scaler: V6FeatureScaler, arm: str, *,
    allow_legacy_arrival_segment: bool = False,
    label_audit: dict[str, int] | None = None,
) -> list[tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, bool,
    dict[str, np.ndarray], np.ndarray,
]]:
    valid = set(int(index) for index in _valid_feature_indexes(session, top=True))
    if not valid:
        return []
    labeler = BiasRewardLabeler(
        raw, allow_legacy_arrival_segment=allow_legacy_arrival_segment,
    )
    pending: list[tuple[
        int, dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, np.ndarray],
    ]] = []
    for index in range(len(session.top_clock_ns) - 1):
        if index not in valid or index + 1 not in valid:
            continue
        decision = int(session.top_clock_ns[index])
        expiry = int(session.top_clock_ns[index + 1])
        if expiry - decision != TOP_LIFETIME_NS:
            continue
        if label_audit is not None:
            label_audit["counterfactual_labels_attempted"] = (
                label_audit.get("counterfactual_labels_attempted", 0) + 1
            )
        label = labeler.label(_raw_index(raw, decision), _raw_index(raw, expiry))
        legal = np.asarray(label.valid_mask, dtype=bool)
        # Route success at the future expiry is a reward-label availability
        # condition, never a decision-time action mask.  Drop the whole sample
        # unless every counterfactual is defined, then expose all three actions.
        if not legal.all():
            if label_audit is not None:
                label_audit["counterfactual_labels_dropped"] = (
                    label_audit.get("counterfactual_labels_dropped", 0) + 1
                )
            continue
        if label_audit is not None:
            label_audit["counterfactual_labels_kept"] = (
                label_audit.get("counterfactual_labels_kept", 0) + 1
            )
        legal = np.ones(3, dtype=bool)
        current = top_observation_at(session, index, scaler).arrays(validate_arm(arm))
        following = top_observation_at(session, index + 1, scaler).arrays(validate_arm(arm))
        pending.append((index, current, np.asarray(label.rewards_cents, dtype=np.float64),
                        legal, following))
    examples = []
    for ordinal, (index, current, rewards, legal, following) in enumerate(pending):
        contiguous = (
            ordinal + 1 < len(pending) and pending[ordinal + 1][0] == index + 1
        )
        terminal = not contiguous
        next_legal = np.zeros(3, bool) if terminal else pending[ordinal + 1][3].copy()
        examples.append((current, rewards, legal, terminal, following, next_legal))
    return examples


def _latest_message(messages: Sequence[RegimeMessage], clock_ns: int) -> RegimeMessage:
    if not messages:
        return neutral_regime_message(clock_ns)
    times = np.asarray([message.decision_time_ns for message in messages], dtype=np.int64)
    index = int(np.searchsorted(times, int(clock_ns), side="right") - 1)
    if index < 0 or int(clock_ns) >= messages[index].expiry_time_ns:
        return neutral_regime_message(clock_ns)
    return messages[index]


def publish_regime_messages(
    trainer: object, session: SessionFeatures, scaler: V6FeatureScaler, arm: str,
) -> tuple[RegimeMessage, ...]:
    output: list[RegimeMessage] = []
    valid = set(int(index) for index in _valid_feature_indexes(session, top=True))
    for index in sorted(valid):
        # The publisher is trained only on exact 15-minute round trips.  Do not
        # publish an end-of-session action for which that registered horizon
        # cannot exist (notably the otherwise-valid 12:20 bar).
        if (
            index + 1 >= len(session.top_clock_ns)
            or index + 1 not in valid
            or int(session.top_clock_ns[index + 1])
            - int(session.top_clock_ns[index]) != TOP_LIFETIME_NS
        ):
            continue
        observation = top_observation_at(session, int(index), scaler).arrays(validate_arm(arm))
        q_values = trainer.q_values(observation)  # type: ignore[attr-defined]
        decision = int(session.top_clock_ns[index])
        output.append(RegimeMessage.from_q_values(q_values, decision, decision + TOP_LIFETIME_NS))
    return tuple(output)


def _atomic_copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _progress_identity(
    seed: int, arm: str, phase: str, pairs: Sequence[tuple[Path, Path]],
    protocol_digest: str, dataset_binding: Mapping[str, object] | None,
    xau_warmstart: Mapping[str, object] | None,
) -> dict[str, object]:
    digest = __import__("hashlib").sha256()
    for raw, side in pairs:
        digest.update(raw.stem.encode())
        digest.update(file_sha256(raw).encode())
        digest.update(file_sha256(side).encode())
    return {
        "schema_version": 1,
        "artifact_type": "V6_TRAINING_PROGRESS",
        "protocol_sha256": protocol_digest,
        "seed": int(seed),
        "arm": arm,
        "phase": phase,
        "input_digest": digest.hexdigest(),
        "training_dates": [pair[0].stem for pair in pairs],
        "dataset_manifest": None if dataset_binding is None else dict(dataset_binding),
        "xau_warmstart": None if xau_warmstart is None else dict(xau_warmstart),
    }


def _load_or_initialize(
    work: Path, *, seed: int, arm: str, phase: str,
    pairs: Sequence[tuple[Path, Path]], pretrained_model: Path | None,
    scaler: V6FeatureScaler, protocol_digest: str,
    dataset_binding: Mapping[str, object] | None,
) -> tuple[V6TrainerBundle, dict[str, object]]:
    pretrained: V6ModelBundle | None = None
    transfer: dict[str, object] | None = None
    if phase != "smoke" and pretrained_model is None:
        raise ValueError("registered V6 training requires eligible XAU price pretraining")
    if pretrained_model is not None:
        from .volume_v6_xau import load_xau_pretrained_bundle
        model_path = Path(pretrained_model).resolve()
        pretrained_root = model_path.parent
        pretrained, metadata = load_xau_pretrained_bundle(
            pretrained_root, require_gc_transfer_eligible=phase != "smoke",
        )
        manifest = json.loads((pretrained_root / "manifest.json").read_text(encoding="utf-8"))
        recorded_model = pretrained_root / str(manifest.get("model_bundle", {}).get("path", ""))
        if (
            recorded_model.resolve() != model_path
            or metadata.get("stage") != "xau_price_pretraining"
            or metadata.get("seed") != seed
            or (phase != "smoke" and metadata.get("gc_transfer_eligible") is not True)
        ):
            raise ValueError("V6 XAU price warm-start identity differs")
        transfer = {"path": str(model_path), "sha256": file_sha256(model_path)}
    identity = _progress_identity(
        seed, arm, phase, pairs, protocol_digest, dataset_binding, transfer,
    )
    progress_path = work / "progress.json"
    expected_top = max(1, len(pairs) * 19 * REGISTERED_GC_PASSES // 2)
    expected_bottom = max(1, len(pairs) * 59 * REGISTERED_GC_PASSES // 2)
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError("interrupted V6 training identity differs")
        bundle = create_trainers(
            seed=seed, regime_epsilon_decay_decisions=expected_top,
            trader_epsilon_decay_decisions=expected_bottom, protocol_id=protocol_digest,
        )
        regime_path = work / "regime_resume.pt"
        trader_path = work / "trader_resume.pt"
        regime_metadata: dict[str, object] | None = None
        trader_metadata: dict[str, object] | None = None
        if regime_path.is_file():
            regime_metadata = load_trainer_checkpoint(bundle.regime, regime_path)
        if trader_path.is_file():
            trader_metadata = load_trainer_checkpoint(bundle.trader, trader_path)

        # A process can stop after the durable trainer checkpoint is replaced
        # but before progress.json is replaced.  In that narrow window the
        # checkpoint metadata is authoritative, otherwise the completed pass
        # would be presented twice on resume.
        for name, checkpoint, metadata in (
            ("regime", regime_path, regime_metadata),
            ("trader", trader_path, trader_metadata),
        ):
            key = f"{name}_passes_complete"
            recorded = int(progress.get(key, 0))
            if metadata is None:
                if recorded:
                    raise ValueError(f"V6 resume records {name} progress without its checkpoint")
                continue
            if any(metadata.get(field) != value for field, value in identity.items()):
                raise ValueError(f"V6 {name} checkpoint metadata differs from resume identity")
            completed = int(metadata.get(key, -1))
            if completed < recorded or completed > REGISTERED_GC_PASSES:
                raise ValueError(f"V6 {name} checkpoint pass counter is invalid")
            progress[key] = completed
            for report_key in (f"{name}_report", f"{name}_pass_reports"):
                if report_key in metadata:
                    progress[report_key] = metadata[report_key]
        atomic_json(progress_path, progress, immutable=False)
        return bundle, progress

    models = V6ModelBundle.create()
    if pretrained is not None:
        transfer_base_encoders(pretrained, models)
    bundle = create_trainers(
        seed=seed, regime_epsilon_decay_decisions=expected_top,
        trader_epsilon_decay_decisions=expected_bottom, models=models,
        protocol_id=protocol_digest,
    )
    progress = identity | {
        "regime_passes_complete": 0,
        "trader_passes_complete": 0,
        "scaler_sha256": scaler.canonical_sha256(),
    }
    atomic_json(progress_path, progress, immutable=False)
    return bundle, progress


def train_seed(
    raw_paths: Sequence[Path], sidecar_paths: Sequence[Path], output: Path, *,
    seed: int, arm: str, pretrained_model: Path | None = None,
    phase: str = "train", maximum_sessions: int | None = None,
    passes: int = REGISTERED_GC_PASSES,
    dataset_manifest: Path | None = None,
) -> Path:
    """Train one same-seed publisher/trader pair with pass-level resume."""

    if phase not in {"train", "smoke"}:
        raise ValueError("V6 model training is allowed only in train or smoke phase")
    if passes != REGISTERED_GC_PASSES and phase != "smoke":
        raise ValueError("registered V6 training uses exactly three GC passes")
    if seed not in SEEDS and phase != "smoke":
        raise ValueError(f"registered V6 seed must be one of {SEEDS}")
    arm = validate_arm(arm)
    pairs = aligned_pairs(
        raw_paths, sidecar_paths, phase=phase, maximum_sessions=maximum_sessions,
    )
    dataset_binding = (
        None if phase == "smoke"
        else _registered_training_dataset(pairs, dataset_manifest)
    )
    if output.exists():
        raise FileExistsError("V6 training output is immutable")
    work = output.with_name(f".{output.name}.part")
    work.mkdir(parents=True, exist_ok=True)
    protocol, protocol_digest = frozen_protocol()
    scaler_path = work / "scaler.json"
    if scaler_path.is_file():
        scaler = V6FeatureScaler.load_json(scaler_path)
    else:
        scaler = V6FeatureScaler.fit_paths(
            pairs, temporary_directory=work, validate_source_hash=True,
        )
        scaler.save_json(scaler_path)
    if tuple(scaler.calibration.training_dates) != tuple(pair[0].stem for pair in pairs):
        raise ValueError("V6 scaler dates differ from the requested training sessions")
    trainers, progress = _load_or_initialize(
        work, seed=seed, arm=arm, phase=phase, pairs=pairs,
        pretrained_model=pretrained_model, scaler=scaler,
        protocol_digest=protocol_digest, dataset_binding=dataset_binding,
    )
    if progress.get("scaler_sha256") != scaler.canonical_sha256():
        raise ValueError("resume scaler differs from V6 training progress")

    regime_reports: list[dict[str, object]] = []
    unsafe_legacy_arrival_boundary = bool(
        progress.get("unsafe_legacy_arrival_boundary", False),
    )
    start_regime = int(progress.get("regime_passes_complete", 0))
    for pass_index in range(start_regime, passes):
        losses: list[float] = []; transitions = skipped_sessions = 0
        label_audit = {
            "counterfactual_labels_attempted": 0,
            "counterfactual_labels_dropped": 0,
            "counterfactual_labels_kept": 0,
        }
        for pair in pairs:
            raw, side = _load_pair(pair, phase=phase)
            session = build_session_features(raw, side, scaler.calibration)
            unsafe_legacy_arrival_boundary |= uses_legacy_arrival_boundary(
                raw, scenario_economics("base"),
            )
            examples = _top_examples(
                raw, session, scaler, arm,
                allow_legacy_arrival_segment=phase == "smoke",
                label_audit=label_audit,
            )
            if not examples:
                skipped_sessions += 1
                continue
            for observation, rewards, legal, terminal, following, next_legal in examples:
                action = trainers.regime.act(observation, legal=legal, explore=True)
                column = ACTIONS.index(action)
                trainers.regime.append(
                    observation, action, float(rewards[column]), following,
                    terminal=terminal, legal=legal, next_legal=next_legal,
                )
                loss = trainers.regime.update()
                if loss is not None:
                    losses.append(loss)
                transitions += 1
        regime_reports.append({
            "pass": pass_index + 1, "transitions": transitions,
            "skipped_sessions": skipped_sessions,
            "mean_loss": float(np.mean(losses)) if losses else None,
            **label_audit,
        })
        progress["regime_passes_complete"] = pass_index + 1
        progress["regime_report"] = trainers.regime.report()
        progress["unsafe_legacy_arrival_boundary"] = bool(
            unsafe_legacy_arrival_boundary,
        )
        progress["regime_pass_reports"] = list(progress.get("regime_pass_reports", [])) + [regime_reports[-1]]
        save_trainer_checkpoint(trainers.regime, work / "regime_resume.pt", progress)
        atomic_json(work / "progress.json", progress, immutable=False)

    trainers.models.regime.eval()
    for parameter in trainers.models.regime.parameters():
        parameter.requires_grad_(False)

    trader_reports: list[dict[str, object]] = []
    start_trader = int(progress.get("trader_passes_complete", 0))
    for pass_index in range(start_trader, passes):
        losses: list[float] = []
        transitions = skipped_sessions = round_trips = rejected = 0
        reward_total_cents = 0
        for pair in pairs:
            raw, side = _load_pair(pair, phase=phase)
            session = build_session_features(raw, side, scaler.calibration)
            selected = _valid_feature_indexes(session, top=False)
            if not len(selected):
                skipped_sessions += 1
                continue
            messages = publish_regime_messages(trainers.regime, session, scaler, arm)
            unsafe_legacy_arrival_boundary |= uses_legacy_arrival_boundary(
                raw, scenario_economics("base"),
            )
            env = V6TradingEnvironment(
                raw, pair[0].stem,
                allow_legacy_arrival_segment=phase == "smoke",
            )
            session_rewards: list[int] = []
            for ordinal, feature_index in enumerate(selected):
                current_raw = _raw_index(raw, int(session.bottom_clock_ns[feature_index]))
                last = ordinal == len(selected) - 1
                if last:
                    next_raw = len(np.asarray(raw["clock_ns"])) - 1
                    terminal = True
                    next_feature_index = int(feature_index)
                else:
                    next_feature_index = int(selected[ordinal + 1])
                    next_raw = _raw_index(raw, int(session.bottom_clock_ns[next_feature_index]))
                    terminal = False
                message = _latest_message(messages, int(session.bottom_clock_ns[feature_index]))
                observation = bottom_observation_at(
                    session, int(feature_index), scaler, message, env.agent_state(current_raw),
                ).arrays(arm)
                legal = env.legal_actions()
                action = trainers.trader.act(observation, legal=legal, explore=True)
                one = env.step(current_raw, action, next_raw, terminal=terminal)
                if terminal:
                    following = {name: np.asarray(value).copy() for name, value in observation.items()}
                    following["state"] = np.asarray(one.next_state, dtype=np.float32)
                    next_legal = np.zeros(3, bool)
                else:
                    next_message = _latest_message(
                        messages, int(session.bottom_clock_ns[next_feature_index]),
                    )
                    following = bottom_observation_at(
                        session, next_feature_index, scaler, next_message,
                        np.asarray(one.next_state, dtype=np.float32),
                    ).arrays(arm)
                    next_legal = np.asarray(one.next_legal_mask, dtype=bool)
                trainers.trader.append(
                    observation, action, float(one.reward_cents), following,
                    terminal=terminal, legal=legal, next_legal=next_legal,
                )
                loss = trainers.trader.update()
                if loss is not None:
                    losses.append(loss)
                session_rewards.append(one.reward_cents); transitions += 1
            result = env.finish()
            reconcile_raw_rewards(session_rewards, result.net_pnl_cents)
            reward_total_cents += result.net_pnl_cents
            round_trips += result.completed_round_trips
            rejected += result.rejected_commands
        trader_reports.append({
            "pass": pass_index + 1, "transitions": transitions,
            "skipped_sessions": skipped_sessions,
            "net_training_reward_cents": reward_total_cents,
            "completed_round_trips": round_trips,
            "rejected_commands": rejected,
            "mean_loss": float(np.mean(losses)) if losses else None,
        })
        progress["trader_passes_complete"] = pass_index + 1
        progress["trader_report"] = trainers.trader.report()
        progress["unsafe_legacy_arrival_boundary"] = bool(
            unsafe_legacy_arrival_boundary,
        )
        progress["trader_pass_reports"] = list(progress.get("trader_pass_reports", [])) + [trader_reports[-1]]
        save_trainer_checkpoint(trainers.trader, work / "trader_resume.pt", progress)
        atomic_json(work / "progress.json", progress, immutable=False)

    # Freezing is a training-stage rule, not part of serialized inference state.
    # Re-enable flags so parameter audits continue to report the real capacity.
    for parameter in trainers.models.regime.parameters():
        parameter.requires_grad_(True)
    trainers.regime.audit_rewards(); trainers.trader.audit_rewards()
    model_path = work / "model_bundle.pt"
    metadata = {
        "stage": "gc_two_level_training", "seed": seed, "arm": arm,
        "phase": phase, "passes": passes,
        "training_start": pairs[0][0].stem, "training_end": pairs[-1][0].stem,
        "training_dates": [pair[0].stem for pair in pairs],
        "scaler_sha256": scaler.canonical_sha256(),
        "smoke_only": phase == "smoke",
        "xau_warmstart": progress.get("xau_warmstart"),
        "dataset_manifest": dataset_binding,
        "unsafe_legacy_arrival_boundary": bool(unsafe_legacy_arrival_boundary),
    }
    save_model_bundle(trainers.models, model_path, metadata, protocol_id=protocol_digest)
    _atomic_copy_checkpoint(work / "regime_resume.pt", work / "regime_trainer.pt")
    _atomic_copy_checkpoint(work / "trader_resume.pt", work / "trader_trainer.pt")
    manifest = {
        "schema_version": 1,
        "artifact_type": "V6_GC_TWO_LEVEL_TRAINING",
        "protocol_sha256": protocol_digest,
        "seed": seed,
        "arm": arm,
        "phase": phase,
        "classification": "DEVELOPMENT",
        "passes": passes,
        "training_start": pairs[0][0].stem,
        "training_end": pairs[-1][0].stem,
        "training_dates": [pair[0].stem for pair in pairs],
        "input_digest": progress["input_digest"],
        "dataset_manifest": dataset_binding,
        "scaler": {"path": scaler_path.name, "sha256": file_sha256(scaler_path),
                   "canonical_sha256": scaler.canonical_sha256()},
        "model_bundle": {"path": model_path.name, "sha256": file_sha256(model_path)},
        "regime_trainer": {"path": "regime_trainer.pt", "sha256": file_sha256(work / "regime_trainer.pt")},
        "trader_trainer": {"path": "trader_trainer.pt", "sha256": file_sha256(work / "trader_trainer.pt")},
        "regime": trainers.regime.report(),
        "trader": trainers.trader.report(),
        "regime_pass_reports": progress.get("regime_pass_reports", []),
        "trader_pass_reports": progress.get("trader_pass_reports", []),
        "same_seed_message_pairing": True,
        "upper_frozen_before_trader": True,
        "joint_fine_tune": False,
        "xau_warmstart": progress.get("xau_warmstart"),
        "smoke_only": phase == "smoke",
        "holdout_read": False,
        "unsafe_legacy_arrival_boundary": bool(unsafe_legacy_arrival_boundary),
        "protocol_verdict_at_training": protocol["development_result"],
    }
    atomic_json(work / "manifest.json", manifest, immutable=False)
    (work / "progress.json").unlink(missing_ok=True)
    (work / "regime_resume.pt").unlink(missing_ok=True)
    (work / "trader_resume.pt").unlink(missing_ok=True)
    os.replace(work, output)
    return output / "manifest.json"


def _load_member(root: Path, *, expected_arm: str) -> tuple[V6ModelBundle, V6FeatureScaler, dict[str, object]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, protocol_digest = frozen_protocol()
    if (
        manifest.get("artifact_type") != "V6_GC_TWO_LEVEL_TRAINING"
        or manifest.get("protocol_sha256") != protocol_digest
        or manifest.get("arm") != expected_arm
    ):
        raise ValueError("V6 checkpoint manifest identity differs")
    model_record = manifest.get("model_bundle")
    scaler_record = manifest.get("scaler")
    if not isinstance(model_record, dict) or not isinstance(scaler_record, dict):
        raise ValueError("V6 checkpoint manifest is incomplete")
    model_path = root / str(model_record["path"])
    scaler_path = root / str(scaler_record["path"])
    if file_sha256(model_path) != model_record.get("sha256") or file_sha256(scaler_path) != scaler_record.get("sha256"):
        raise ValueError("V6 checkpoint file hash differs")
    models, metadata = load_model_bundle(model_path, protocol_id=protocol_digest)
    scaler = V6FeatureScaler.load_json(scaler_path)
    if scaler.canonical_sha256() != scaler_record.get("canonical_sha256"):
        raise ValueError("V6 scaler canonical hash differs")
    if metadata.get("seed") != manifest.get("seed") or metadata.get("arm") != expected_arm:
        raise ValueError("V6 model metadata differs from its manifest")
    return models, scaler, manifest


def _member_messages(
    model: V6ModelBundle, session: SessionFeatures, scaler: V6FeatureScaler, arm: str,
) -> tuple[RegimeMessage, ...]:
    class _Inference:
        def q_values(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
            from .volume_v6_training import _tensor_batch, regime_forward
            import torch
            with torch.no_grad():
                return regime_forward(model.regime, _tensor_batch([observation], torch.device("cpu")))[0].numpy()
    return publish_regime_messages(_Inference(), session, scaler, arm)


def evaluate_ensemble(
    checkpoint_roots: Sequence[Path], raw_paths: Sequence[Path],
    sidecar_paths: Sequence[Path], *, arm: str, phase: str,
    economics: ExecutionEconomics | None = None,
    require_registered_seeds: bool = True,
    unlock_receipt: Path | None = None,
    scenario: str = "base",
    evaluation_dataset_manifest: Path | None = None,
    complete_training_manifest: Path | None = None,
) -> dict[str, object]:
    """Replay five paired hierarchies; vote only on final target positions."""

    arm = validate_arm(arm)
    expected_economics = scenario_economics(scenario)
    pairs = aligned_pairs(raw_paths, sidecar_paths, phase=phase)
    evaluation_binding: dict[str, object] | None = None
    training_binding: dict[str, object] | None = None
    if phase in {"validation", "test"}:
        _assert_registered_dataset_access(
            pairs, phase=phase, unlock_receipt=unlock_receipt,
        )
        evaluation_binding, training_binding = _registered_evaluation_chain(
            pairs, checkpoint_roots, arm=arm, phase=phase,
            evaluation_dataset_manifest=evaluation_dataset_manifest,
            complete_training_manifest=complete_training_manifest,
            unlock_receipt=unlock_receipt,
        )
    loaded_with_roots = sorted(
        ((Path(root), *_load_member(Path(root), expected_arm=arm))
         for root in checkpoint_roots),
        key=lambda value: int(value[3]["seed"]),
    )
    canonical_roots = [value[0] for value in loaded_with_roots]
    loaded = [(value[1], value[2], value[3]) for value in loaded_with_roots]
    seeds = tuple(int(value[2]["seed"]) for value in loaded)
    if require_registered_seeds and seeds != SEEDS:
        raise ValueError(f"registered V6 ensemble requires seeds {SEEDS}")
    if require_registered_seeds and any(value[2].get("smoke_only") is not False for value in loaded):
        raise ValueError("registered evaluation refuses smoke checkpoints")
    scaler_hashes = {value[1].canonical_sha256() for value in loaded}
    training_dates = {tuple(value[2].get("training_dates", [])) for value in loaded}
    if len(scaler_hashes) != 1 or len(training_dates) != 1:
        raise ValueError("V6 ensemble members differ in fitted transforms or training dates")
    scaler = loaded[0][1]
    economics = expected_economics if economics is None else economics
    if economics != expected_economics:
        raise ValueError("execution economics do not match the named V6 scenario")
    daily: dict[str, int] = {}
    trips = rejected = fills = 0
    unsafe_legacy_arrival_boundary = False
    for pair in pairs:
        raw, side = _load_pair(pair, phase=phase)
        session = build_session_features(raw, side, scaler.calibration)
        selected = _valid_feature_indexes(session, top=False)
        if not len(selected):
            raise ValueError(f"{pair[0].stem}: no valid V6 trader decisions")
        member_messages = [
            _member_messages(models, session, member_scaler, arm)
            for models, member_scaler, _ in loaded
        ]
        unsafe_legacy_arrival_boundary |= uses_legacy_arrival_boundary(raw, economics)
        env = V6TradingEnvironment(
            raw, pair[0].stem, economics,
            allow_legacy_arrival_segment=phase == "smoke",
        )
        for ordinal, feature_index in enumerate(selected):
            current_raw = _raw_index(raw, int(session.bottom_clock_ns[feature_index]))
            terminal = ordinal == len(selected) - 1
            next_raw = (
                len(np.asarray(raw["clock_ns"])) - 1 if terminal
                else _raw_index(raw, int(session.bottom_clock_ns[selected[ordinal + 1]]))
            )
            state = env.agent_state(current_raw)
            legal = env.legal_actions()
            votes: list[int] = []
            for (models, member_scaler, _), messages in zip(loaded, member_messages, strict=True):
                message = _latest_message(messages, int(session.bottom_clock_ns[feature_index]))
                observation = bottom_observation_at(
                    session, int(feature_index), member_scaler, message, state,
                ).arrays(arm)
                from .volume_v6_training import _tensor_batch, trader_forward
                import torch
                with torch.no_grad():
                    q = trader_forward(
                        models.trader, _tensor_batch([observation], torch.device("cpu")),
                    )[0].numpy()
                q[~legal] = -np.inf
                votes.append(ACTIONS[int(np.argmax(q))])
            action = flat_on_vote_tie(votes) if len(votes) == len(SEEDS) else int(np.median(votes))
            if action not in ACTIONS or not bool(legal[ACTIONS.index(action)]):
                action = 0
            env.step(current_raw, action, next_raw, terminal=terminal)
        result = env.finish()
        daily[pair[0].stem] = result.net_pnl_cents
        trips += result.completed_round_trips; rejected += result.rejected_commands
        fills += len(result.fills)
    _, protocol_digest = frozen_protocol()
    classification = classify_dates([pair[0].stem for pair in pairs], phase=phase)
    arm_name = "V6-RichVolume" if arm == "rich_volume" else "V6-BaseOHLCV"
    checkpoint_set = [
        {"seed": int(manifest["seed"]), "manifest_sha256": file_sha256(root / "manifest.json")}
        for root, (_, _, manifest) in zip(canonical_roots, loaded, strict=True)
    ]
    return {
        "schema_version": 1,
        "artifact_type": "V6_COMMON_LEDGER_REPLAY",
        "protocol_sha256": protocol_digest,
        "arm": arm_name,
        "phase": phase,
        "classification": classification,
        "scenario": scenario,
        "economics": asdict(economics),
        "ensemble_seeds": list(seeds),
        "ensemble_rule": "same_seed_publisher_then_target_majority_flat_on_any_tie_v1",
        "evaluation_dates": [pair[0].stem for pair in pairs],
        "daily_pnl_cents": daily,
        "metrics": daily_metrics(daily, trips),
        "rejected_commands": rejected,
        "fills": fills,
        "checkpoint_set": checkpoint_set,
        "evaluation_dataset_manifest": evaluation_binding,
        "complete_training_manifest": training_binding,
        "registered_evaluation": bool(require_registered_seeds),
        "smoke_only": not require_registered_seeds,
        "holdout_read": phase == "test",
        "unsafe_legacy_arrival_boundary": bool(unsafe_legacy_arrival_boundary),
    }
