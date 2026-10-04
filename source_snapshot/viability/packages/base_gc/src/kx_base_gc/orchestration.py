from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .bars import BOUNDARY_COLUMNS
from .contract import ACTIONS, AgentSpec, BaselineContract
from .models import DQNTrainer
from .prepared import (
    PreparedBarsEvidence,
    PreparedHierarchyProduct,
    reverify_prepared_product,
    sha256_file,
)
from .state import StateRecord, build_state_records
from .trace import ModelManifest, trace_content_sha256, validate_trace_frame
from .transitions import ActionLock, transition_builder_for


ORCHESTRATION_SCHEMA_VERSION = 1


class HierarchyTrainingError(RuntimeError):
    pass


class InsufficientMaturedTransitions(HierarchyTrainingError):
    pass


class UnsafeTrainingCutoff(HierarchyTrainingError):
    pass


class Trainer(Protocol):
    def act(self, observation: Sequence[float], *, explore: bool = True) -> int: ...

    def append(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        *,
        terminal: bool = False,
    ) -> None: ...

    def update(self) -> float | None: ...

    def parameter_sha256(self) -> str: ...


TrainerFactory = Callable[[BaselineContract, AgentSpec, int], Trainer]


@dataclass(frozen=True)
class AgentSupport:
    agent: str
    state_count: int
    training_state_count: int
    evaluation_state_count: int
    decision_count: int
    training_decision_count: int
    matured_training_transitions: int
    matured_evaluation_transitions: int
    discarded_at_reward_boundaries: int
    discarded_at_training_cutoff: int
    pending_at_end: int
    update_count: int
    first_available_time_ns: int
    last_available_time_ns: int


@dataclass(frozen=True)
class SeedHierarchyResult:
    schema_version: int
    variant: str
    evidence_class: str
    seed: int
    contract_sha256: str
    training_cutoff_ns: int
    evidence_manifest_sha256: str
    registered_split_sha256: str
    training_config_sha256: str
    training_config: Mapping[str, Any]
    trace: pd.DataFrame
    model_manifests: tuple[ModelManifest, ...]
    support: tuple[AgentSupport, ...]
    checkpoint_paths: tuple[str, ...]


@dataclass(frozen=True)
class _AgentRun:
    trace: pd.DataFrame
    support: AgentSupport
    trainer: Trainer
    model_manifest: ModelManifest
    checkpoint_path: str | None


def _sha256_json(value: Any) -> str:
    def convert(item: Any) -> Any:
        if hasattr(item, "item"):
            return item.item()
        if isinstance(item, Mapping):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(value) for value in item]
        return item

    payload = json.dumps(
        convert(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_trainer_factory(
    contract: BaselineContract, agent: AgentSpec, seed: int
) -> Trainer:
    return DQNTrainer(contract, agent, seed=seed)


def _asof_within_reward_boundary(
    target: pd.DataFrame,
    source: pd.DataFrame,
    *,
    value_column: str,
    default: Any,
) -> pd.Series:
    """Backward-as-of join that cannot carry a position across a cutoff."""

    grouping = [*BOUNDARY_COLUMNS, "reward_boundary_id"]
    required = {*grouping, "available_time_ns", value_column}
    missing = sorted(required - set(source.columns))
    if missing:
        raise KeyError(f"hierarchy source trace is missing columns: {missing}")
    result = pd.Series([default] * len(target), index=target.index, dtype=object)
    lookup = {
        key if isinstance(key, tuple) else (key,): value.sort_values(
            "available_time_ns", kind="mergesort"
        )
        for key, value in source.groupby(grouping, sort=False, dropna=False)
    }
    for key, rows in target.groupby(grouping, sort=False, dropna=False):
        source_rows = lookup.get(key if isinstance(key, tuple) else (key,))
        if source_rows is None:
            continue
        clocks = source_rows["available_time_ns"].to_numpy(dtype=np.int64)
        values = source_rows[value_column].to_numpy()
        for index, available in rows["available_time_ns"].items():
            location = int(np.searchsorted(clocks, int(available), side="right") - 1)
            if location >= 0:
                result.at[index] = values[location]
    return result


def _metadata_lookup(bars: pd.DataFrame) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    keys = [*BOUNDARY_COLUMNS, "bar_end_ns"]
    if bars.duplicated(keys).any():
        raise HierarchyTrainingError("duplicate agent bar metadata key")
    return {
        tuple(row[column] for column in keys): row
        for row in bars.to_dict(orient="records")
    }


def _checkpoint(
    trainer: Trainer,
    checkpoint_directory: Path | None,
    *,
    agent: str,
    seed: int,
) -> tuple[str | None, str | None]:
    if checkpoint_directory is None:
        return None, None
    save = getattr(trainer, "save_checkpoint", None)
    if save is None:
        raise HierarchyTrainingError("trainer cannot emit a reproducible checkpoint")
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    path = checkpoint_directory / f"seed_{seed}_{agent.lower()}.pt"
    save(path)
    return str(path), sha256_file(path)


def _train_agent(
    bars: pd.DataFrame,
    records: Sequence[StateRecord],
    *,
    contract: BaselineContract,
    agent: AgentSpec,
    seed: int,
    training_cutoff_ns: int,
    minimum_matured_transitions: int,
    trainer_factory: TrainerFactory,
    evidence: PreparedBarsEvidence,
    attempted_trial_id: str,
    training_config_sha256: str,
    checkpoint_directory: Path | None,
) -> _AgentRun:
    if not records:
        raise InsufficientMaturedTransitions(f"{agent.name} has no fully warmed states")
    metadata = _metadata_lookup(bars)
    trainer = trainer_factory(contract, agent, seed)
    lock = ActionLock(agent.action_lock_bars)
    rewards = transition_builder_for(contract, agent)
    rows: list[dict[str, Any]] = []
    matured_training = 0
    matured_evaluation = 0
    updates = 0
    training_decisions = 0
    total_decisions = 0
    training_states = 0
    evaluation_states = 0
    discarded_at_cutoff = 0
    was_training: bool | None = None
    last_training_boundary: str | None = None
    trained_dates: list[str] = []

    ordered_records = sorted(
        records,
        key=lambda record: (
            record.available_time_ns,
            record.raw_symbol,
            record.bar_end_ns,
        ),
    )
    for record in ordered_records:
        key = (
            record.raw_symbol,
            record.continuity_segment_id,
            record.state_segment_id,
            record.bar_end_ns,
        )
        row = metadata.get(key)
        if row is None:
            raise HierarchyTrainingError(f"missing metadata for {agent.name} state {key}")
        available = int(record.available_time_ns)
        if int(row["decision_time_ns"]) < available:
            raise HierarchyTrainingError(
                f"{agent.name} prepared decision precedes complete state availability"
            )
        boundary = str(row["reward_boundary_id"])
        training = available <= training_cutoff_ns
        if was_training is True and not training:
            if boundary == last_training_boundary:
                raise UnsafeTrainingCutoff(
                    f"{agent.name} first evaluation state shares reward boundary "
                    "with training; move the cutoff to a registered boundary"
                )
            discarded_at_cutoff += rewards.pending_count
            rewards.reset()
            lock.reset()
        was_training = training
        if training:
            training_states += 1
            last_training_boundary = boundary
            trained_dates.append(
                pd.Timestamp(available, unit="ns", tz="UTC").date().isoformat()
            )
        else:
            evaluation_states += 1

        decision_eligible = bool(row.get("decision_eligible", True))
        if decision_eligible:
            decision_taken = lock.boundary_id != boundary or lock.remaining_bars == 0
            proposed = (
                int(trainer.act(record.observation, explore=training))
                if decision_taken
                else lock.current_action
            )
            action = lock.step(proposed, boundary_id=boundary)
        else:
            # Prepared empirical products retain post-cutoff bars as causal
            # history, but the registered 12:25 Chicago liquidation boundary
            # makes them observation-only.  Opening a sample here would turn a
            # fixed ex-ante cutoff into an unregistered after-hours strategy.
            if lock.boundary_id != boundary:
                lock.reset()
                lock.boundary_id = boundary
            decision_taken = False
            action = 0
        if action not in ACTIONS:
            raise HierarchyTrainingError("trainer emitted an unregistered action")
        if decision_taken:
            total_decisions += 1
            training_decisions += int(training)

        reward_kwargs: dict[str, Any] = {}
        if contract.variant == "base_strong_gc":
            reward_kwargs = {
                "aggressive_sell_fill_ticks": float(
                    row["aggressive_sell_fill_ticks"]
                ),
                "aggressive_buy_fill_ticks": float(row["aggressive_buy_fill_ticks"]),
                "decision_time_ns": int(row["decision_time_ns"]),
                "submission_time_ns": int(row["submission_time_ns"]),
                "arrival_time_ns": int(row["arrival_time_ns"]),
                "aggressive_fill_available_time_ns": int(
                    row["aggressive_fill_available_time_ns"]
                ),
                "aggressive_fill_source_ordinal": int(
                    row["aggressive_fill_source_ordinal"]
                ),
            }
        matured = rewards.observe(
            record.observation,
            action=action,
            price_ticks=float(row["close"]),
            available_time_ns=available,
            boundary_id=boundary,
            record_decision=decision_taken and decision_eligible,
            **reward_kwargs,
        )
        for transition in matured:
            if training:
                trainer.append(
                    transition.observation,
                    transition.action,
                    transition.reward,
                    transition.next_observation,
                    terminal=False,
                )
                matured_training += 1
                if trainer.update() is not None:
                    updates += 1
            else:
                matured_evaluation += 1
        rows.append(
            {
                "agent": agent.name,
                "policy_seed": seed,
                "date": str(row["date"]),
                "raw_symbol": record.raw_symbol,
                "continuity_segment_id": record.continuity_segment_id,
                "state_segment_id": record.state_segment_id,
                "reward_boundary_id": boundary,
                "bar_end_ns": record.bar_end_ns,
                "available_time_ns": available,
                "decision_time_ns": int(row["decision_time_ns"]),
                "submission_time_ns": int(row["submission_time_ns"]),
                "arrival_time_ns": int(row["arrival_time_ns"]),
                "action": action,
                "position": action,
                "decision_taken": bool(decision_taken),
                "state_sha256": record.observation_sha256,
                "active_contract_map_hash": str(row["active_contract_map_hash"]),
                "phase": "training" if training else "evaluation",
            }
        )

    if matured_training < minimum_matured_transitions:
        raise InsufficientMaturedTransitions(
            f"{agent.name} matured {matured_training} training transitions; "
            f"requires {minimum_matured_transitions}"
        )
    if not trained_dates:
        raise InsufficientMaturedTransitions(f"{agent.name} has no states before cutoff")
    parameter_hash = trainer.parameter_sha256()
    if len(parameter_hash) != 64:
        raise HierarchyTrainingError("trainer returned an invalid parameter SHA-256")
    checkpoint_path, checkpoint_hash = _checkpoint(
        trainer, checkpoint_directory, agent=agent.name, seed=seed
    )
    support = AgentSupport(
        agent=agent.name,
        state_count=len(rows),
        training_state_count=training_states,
        evaluation_state_count=evaluation_states,
        decision_count=total_decisions,
        training_decision_count=training_decisions,
        matured_training_transitions=matured_training,
        matured_evaluation_transitions=matured_evaluation,
        discarded_at_reward_boundaries=rewards.discarded_at_boundaries,
        discarded_at_training_cutoff=discarded_at_cutoff,
        pending_at_end=rewards.pending_count,
        update_count=updates,
        first_available_time_ns=int(rows[0]["available_time_ns"]),
        last_available_time_ns=int(rows[-1]["available_time_ns"]),
    )
    manifest = ModelManifest(
        schema_version=1,
        model_id=f"{contract.variant}-{agent.name.lower()}-seed-{seed}",
        variant=contract.variant,
        agent=agent.name,
        seed=seed,
        state_dimension=agent.observation_dimension,
        action_values=ACTIONS,
        contract_sha256=contract.sha256,
        training_config_sha256=training_config_sha256,
        parameter_sha256=parameter_hash,
        training_data_manifest_sha256=evidence.manifest_sha256,
        evidence_manifest_sha256=evidence.manifest_sha256,
        registered_split_sha256=evidence.registered_split_sha256,
        training_cutoff_ns=training_cutoff_ns,
        trained_through_date=max(trained_dates),
        attempted_trial_id=attempted_trial_id,
        checkpoint_sha256=checkpoint_hash,
    )
    manifest.validate()
    return _AgentRun(
        trace=pd.DataFrame(rows),
        support=support,
        trainer=trainer,
        model_manifest=manifest,
        checkpoint_path=checkpoint_path,
    )


def _cold_state_hash(agent: str) -> str:
    return hashlib.sha256(f"{agent}:cold-flat-state:v1".encode("utf-8")).hexdigest()


def _seed_trace(
    m5: _AgentRun,
    h1: _AgentRun,
    m15: _AgentRun,
    *,
    seed: int,
) -> pd.DataFrame:
    target = m5.trace.copy().reset_index(drop=True)
    h1_position = _asof_within_reward_boundary(
        target, h1.trace, value_column="position", default=0
    ).astype(int)
    m15_position = _asof_within_reward_boundary(
        target, m15.trace, value_column="position", default=0
    ).astype(int)
    h1_hash = _asof_within_reward_boundary(
        target,
        h1.trace,
        value_column="state_sha256",
        default=_cold_state_hash("H1"),
    ).astype(str)
    m15_hash = _asof_within_reward_boundary(
        target,
        m15.trace,
        value_column="state_sha256",
        default=_cold_state_hash("M15"),
    ).astype(str)
    trace = pd.DataFrame(
        {
            "decision_id": [
                "base-gc-"
                + hashlib.sha256(
                    (
                        f"{row.raw_symbol}|{row.continuity_segment_id}|"
                        f"{row.state_segment_id}|{row.reward_boundary_id}|"
                        f"{row.bar_end_ns}|{row.available_time_ns}"
                    ).encode("utf-8")
                ).hexdigest()[:32]
                for row in target.itertuples(index=False)
            ],
            "policy_seed": seed,
            "date": target["date"],
            "raw_symbol": target["raw_symbol"],
            "continuity_segment_id": target["continuity_segment_id"],
            "state_segment_id": target["state_segment_id"],
            "reward_boundary_id": target["reward_boundary_id"],
            "bar_end_time_ns": target["bar_end_ns"],
            "available_time_ns": target["available_time_ns"],
            "decision_time_ns": target["decision_time_ns"],
            "submission_time_ns": target["submission_time_ns"],
            "arrival_time_ns": target["arrival_time_ns"],
            "h1_action": h1_position,
            "m15_action": m15_position,
            "m5_action": target["action"].astype(int),
            "h1_position": h1_position,
            "m15_position": m15_position,
            "final_target": target["position"].astype(int),
            "m5_decision_taken": target["decision_taken"].astype(bool),
            "h1_state_sha256": h1_hash,
            "m15_state_sha256": m15_hash,
            "m5_state_sha256": target["state_sha256"],
            "active_contract_map_hash": target["active_contract_map_hash"],
        }
    )
    return validate_trace_frame(trace)


def train_seed_hierarchy(
    prepared: PreparedHierarchyProduct,
    contract: BaselineContract,
    *,
    seed: int,
    training_cutoff_ns: int,
    attempted_trial_id: str,
    minimum_matured_transitions: int | None = None,
    trainer_factory: TrainerFactory | None = None,
    checkpoint_directory: str | Path | None = None,
) -> SeedHierarchyResult:
    """Causally train one complete H1→M15→M5 hierarchy.

    The function accepts only previously verified/prepared timeframe frames. It
    never scans raw MBO storage and never chooses a split from outcomes.
    """

    contract.validate()
    prepared = reverify_prepared_product(prepared, contract)
    timeframes = prepared.timeframes
    evidence = prepared.evidence
    if set(timeframes) != {"H1", "M15", "M5"}:
        raise HierarchyTrainingError("training requires exactly H1/M15/M5 frames")
    if seed not in contract.seed_ensemble:
        raise HierarchyTrainingError("seed is not in the registered five-seed ensemble")
    if training_cutoff_ns <= 0 or not attempted_trial_id:
        raise HierarchyTrainingError("positive cutoff and attempted_trial_id are required")
    minimum = (
        contract.optimization.replay_warmup
        if minimum_matured_transitions is None
        else int(minimum_matured_transitions)
    )
    if minimum < 1:
        raise HierarchyTrainingError("minimum matured transitions must be positive")
    if (
        evidence.evidence_class != "SYNTHETIC_ONLY"
        and minimum < contract.optimization.replay_warmup
    ):
        raise HierarchyTrainingError(
            "empirical training cannot lower the registered replay-warmup support gate"
        )
    factory = trainer_factory or _default_trainer_factory
    checkpoint_path = (
        None if checkpoint_directory is None else Path(checkpoint_directory)
    )
    if evidence.evidence_class != "SYNTHETIC_ONLY" and checkpoint_path is None:
        raise HierarchyTrainingError(
            "empirical training requires a resolved checkpoint directory"
        )
    training_config = {
        "schema_version": ORCHESTRATION_SCHEMA_VERSION,
        "contract_sha256": contract.sha256,
        "variant": contract.variant,
        "seed": seed,
        "training_cutoff_ns": int(training_cutoff_ns),
        "minimum_matured_transitions": minimum,
        "attempted_trial_id": attempted_trial_id,
        "evidence_class": evidence.evidence_class,
        "evidence_manifest_sha256": evidence.manifest_sha256,
        "registered_split_sha256": evidence.registered_split_sha256,
        "agent_order": ["H1", "M15", "M5"],
        "orchestration": "causal_online_one_seed_hierarchy_v1",
        "cutoff_rule": "freeze_and_reset_only_at_new_reward_boundary_v1",
    }
    training_config_hash = _sha256_json(training_config)

    h1_records = build_state_records(timeframes["H1"], contract.agent("H1"))
    h1 = _train_agent(
        timeframes["H1"],
        h1_records,
        contract=contract,
        agent=contract.agent("H1"),
        seed=seed,
        training_cutoff_ns=training_cutoff_ns,
        minimum_matured_transitions=minimum,
        trainer_factory=factory,
        evidence=evidence,
        attempted_trial_id=attempted_trial_id,
        training_config_sha256=training_config_hash,
        checkpoint_directory=checkpoint_path,
    )

    m15_bars = timeframes["M15"].reset_index(drop=True)
    m15_h1 = _asof_within_reward_boundary(
        m15_bars, h1.trace, value_column="position", default=0
    ).astype(int)
    m15_records = build_state_records(
        m15_bars, contract.agent("M15"), inherited={"H1": m15_h1}
    )
    m15 = _train_agent(
        m15_bars,
        m15_records,
        contract=contract,
        agent=contract.agent("M15"),
        seed=seed,
        training_cutoff_ns=training_cutoff_ns,
        minimum_matured_transitions=minimum,
        trainer_factory=factory,
        evidence=evidence,
        attempted_trial_id=attempted_trial_id,
        training_config_sha256=training_config_hash,
        checkpoint_directory=checkpoint_path,
    )

    m5_bars = timeframes["M5"].reset_index(drop=True)
    m5_h1 = _asof_within_reward_boundary(
        m5_bars, h1.trace, value_column="position", default=0
    ).astype(int)
    m5_m15 = _asof_within_reward_boundary(
        m5_bars, m15.trace, value_column="position", default=0
    ).astype(int)
    m5_records = build_state_records(
        m5_bars,
        contract.agent("M5"),
        inherited={"H1": m5_h1, "M15": m5_m15},
    )
    m5 = _train_agent(
        m5_bars,
        m5_records,
        contract=contract,
        agent=contract.agent("M5"),
        seed=seed,
        training_cutoff_ns=training_cutoff_ns,
        minimum_matured_transitions=minimum,
        trainer_factory=factory,
        evidence=evidence,
        attempted_trial_id=attempted_trial_id,
        training_config_sha256=training_config_hash,
        checkpoint_directory=checkpoint_path,
    )
    trace = _seed_trace(m5, h1, m15, seed=seed)
    runs = (h1, m15, m5)
    return SeedHierarchyResult(
        schema_version=ORCHESTRATION_SCHEMA_VERSION,
        variant=contract.variant,
        evidence_class=evidence.evidence_class,
        seed=seed,
        contract_sha256=contract.sha256,
        training_cutoff_ns=training_cutoff_ns,
        evidence_manifest_sha256=evidence.manifest_sha256,
        registered_split_sha256=evidence.registered_split_sha256,
        training_config_sha256=training_config_hash,
        training_config=training_config,
        trace=trace,
        model_manifests=tuple(run.model_manifest for run in runs),
        support=tuple(run.support for run in runs),
        checkpoint_paths=tuple(
            run.checkpoint_path for run in runs if run.checkpoint_path is not None
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(handle)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_seed_hierarchy_result(
    result: SeedHierarchyResult,
    output_directory: str | Path,
) -> dict[str, Any]:
    if result.schema_version != ORCHESTRATION_SCHEMA_VERSION:
        raise HierarchyTrainingError("unsupported one-seed run schema")
    if result.evidence_class not in {
        "SYNTHETIC_ONLY",
        "DEVELOPMENT_ONLY",
        "LOCKED_EMPIRICAL",
    }:
        raise HierarchyTrainingError("invalid one-seed evidence class")
    trace = validate_trace_frame(result.trace)
    if set(trace["policy_seed"].astype(int)) != {result.seed}:
        raise HierarchyTrainingError("one-seed trace policy differs from run seed")
    if _sha256_json(result.training_config) != result.training_config_sha256:
        raise HierarchyTrainingError("training configuration content hash mismatch")
    expected_config = {
        "contract_sha256": result.contract_sha256,
        "variant": result.variant,
        "seed": result.seed,
        "training_cutoff_ns": result.training_cutoff_ns,
        "evidence_class": result.evidence_class,
        "evidence_manifest_sha256": result.evidence_manifest_sha256,
        "registered_split_sha256": result.registered_split_sha256,
    }
    if any(result.training_config.get(key) != value for key, value in expected_config.items()):
        raise HierarchyTrainingError("training configuration differs from run identity")
    manifests = tuple(result.model_manifests)
    if {(value.seed, value.agent) for value in manifests} != {
        (result.seed, agent) for agent in ("H1", "M15", "M5")
    } or len(manifests) != 3:
        raise HierarchyTrainingError("one-seed run requires exactly H1/M15/M5 manifests")
    for manifest in manifests:
        manifest.validate()
        if (
            manifest.variant != result.variant
            or manifest.contract_sha256 != result.contract_sha256
            or manifest.training_config_sha256 != result.training_config_sha256
            or manifest.evidence_manifest_sha256 != result.evidence_manifest_sha256
            or manifest.training_data_manifest_sha256 != result.evidence_manifest_sha256
            or manifest.registered_split_sha256 != result.registered_split_sha256
            or manifest.training_cutoff_ns != result.training_cutoff_ns
        ):
            raise HierarchyTrainingError("model manifest differs from its one-seed run")
    if {value.agent for value in result.support} != {"H1", "M15", "M5"}:
        raise HierarchyTrainingError("one-seed run requires H1/M15/M5 support records")
    path_count = len(result.checkpoint_paths)
    if result.evidence_class != "SYNTHETIC_ONLY" and path_count != 3:
        raise HierarchyTrainingError("empirical one-seed run requires three checkpoints")
    if path_count not in {0, 3}:
        raise HierarchyTrainingError("checkpoint paths must be absent or complete")

    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    trace_path = target / "seed_decision_trace.parquet"
    manifests_path = target / "model_manifests.json"
    support_path = target / "support.json"
    config_path = target / "training_config.json"
    run_path = target / "run_manifest.json"
    for path in (trace_path, manifests_path, support_path, config_path, run_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing result artifact: {path}")
    checkpoint_artifacts: dict[str, dict[str, str]] = {}
    if path_count:
        for manifest, raw_path in zip(manifests, result.checkpoint_paths, strict=True):
            checkpoint = Path(raw_path).expanduser().resolve()
            try:
                relative = checkpoint.relative_to(target.resolve())
            except ValueError as error:
                raise HierarchyTrainingError(
                    "checkpoints must reside inside their one-seed output directory"
                ) from error
            if not checkpoint.is_file() or manifest.checkpoint_sha256 is None:
                raise HierarchyTrainingError(f"model {manifest.model_id} checkpoint is missing")
            observed = sha256_file(checkpoint)
            if observed != manifest.checkpoint_sha256:
                raise HierarchyTrainingError(f"model {manifest.model_id} checkpoint hash mismatch")
            checkpoint_artifacts[manifest.model_id] = {
                "path": relative.as_posix(),
                "sha256": observed,
            }
    elif any(value.checkpoint_sha256 is not None for value in manifests):
        raise HierarchyTrainingError("model manifests declare unresolved checkpoints")

    _write_parquet(trace_path, trace)
    _write_json(manifests_path, [asdict(value) for value in manifests])
    _write_json(support_path, [asdict(value) for value in result.support])
    _write_json(config_path, result.training_config)
    artifacts = {
        "seed_decision_trace": {
            "path": trace_path.name,
            "sha256": sha256_file(trace_path),
            "content_sha256": trace_content_sha256(result.trace),
        },
        "model_manifests": {
            "path": manifests_path.name,
            "sha256": sha256_file(manifests_path),
        },
        "support": {"path": support_path.name, "sha256": sha256_file(support_path)},
        "training_config": {
            "path": config_path.name,
            "sha256": sha256_file(config_path),
            "content_sha256": result.training_config_sha256,
        },
    }
    run_manifest = {
        "schema_version": result.schema_version,
        "artifact_type": "BASE_GC_ONE_SEED_TRAINING_RUN",
        "claim_scope": (
            "Causal one-seed structural training artifact. Neural rewards are "
            "training labels, not common-ledger execution or profitability evidence."
        ),
        "variant": result.variant,
        "evidence_class": result.evidence_class,
        "seed": result.seed,
        "contract_sha256": result.contract_sha256,
        "training_cutoff_ns": result.training_cutoff_ns,
        "training_config_sha256": result.training_config_sha256,
        "evidence_manifest_sha256": result.evidence_manifest_sha256,
        "registered_split_sha256": result.registered_split_sha256,
        "artifacts": artifacts,
        "checkpoints": checkpoint_artifacts,
    }
    _write_json(run_path, run_manifest)
    return {**run_manifest, "run_manifest_sha256": sha256_file(run_path)}
