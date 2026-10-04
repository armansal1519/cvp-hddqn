from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contract import ACTIONS, BaselineContract, registered_contract
from .prepared import sha256_file
from .sessions import require_exact_globex_session_dates
from .trace import (
    INVARIANT_ENSEMBLE_COLUMNS,
    ModelManifest,
    trace_content_sha256,
    validate_trace_frame,
)


ENSEMBLE_TRACE_COLUMNS = (
    *INVARIANT_ENSEMBLE_COLUMNS,
    "final_target",
    "vote_short",
    "vote_flat",
    "vote_long",
    "vote_vector_sha256",
    "winning_policy_seed",
)
_SHA_COLUMNS = ("vote_vector_sha256", "active_contract_map_hash")


@dataclass(frozen=True)
class EnsembleTraceProduct:
    schema_version: int
    product_id: str
    variant: str
    evidence_class: str
    contract_sha256: str
    registered_split_sha256: str | None
    training_cutoff_ns: int | None
    trace_content_sha256: str
    trace_rows: int
    member_trace_sha256: tuple[tuple[int, str], ...]
    model_manifests: tuple[ModelManifest, ...]
    claim_scope: str
    member_run_sha256: tuple[tuple[int, str], ...] = ()
    evidence_manifest_sha256: str | None = None

    def validate(self) -> None:
        if self.schema_version != 1 or self.trace_rows < 0:
            raise ValueError("invalid ensemble trace product schema/count")
        if self.variant not in {"base_paper_gc", "base_strong_gc"}:
            raise ValueError("invalid ensemble variant")
        if self.evidence_class not in {
            "SYNTHETIC_ONLY",
            "DEVELOPMENT_ONLY",
            "LOCKED_EMPIRICAL",
        }:
            raise ValueError("invalid ensemble evidence class")
        hashes = [self.contract_sha256, self.trace_content_sha256]
        hashes.extend(value for _, value in self.member_trace_sha256)
        if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) for value in hashes):
            raise ValueError("ensemble product contains an invalid SHA-256")
        seeds = tuple(seed for seed, _ in self.member_trace_sha256)
        if len(seeds) != 5 or len(set(seeds)) != 5:
            raise ValueError("ensemble product requires five member traces")
        if self.evidence_class != "SYNTHETIC_ONLY":
            if self.registered_split_sha256 is None or self.training_cutoff_ns is None:
                raise ValueError("empirical ensemble must bind an exact split and cutoff")
            if len(self.model_manifests) != 15:
                raise ValueError("empirical ensemble requires 5 seeds x 3 agent models")
            run_seeds = tuple(seed for seed, _ in self.member_run_sha256)
            if len(run_seeds) != 5 or set(run_seeds) != set(seeds):
                raise ValueError("empirical ensemble requires five verified one-seed runs")
            if any(not _valid_sha(value) for _, value in self.member_run_sha256):
                raise ValueError("empirical ensemble contains an invalid run hash")
            if self.evidence_manifest_sha256 is None or not _valid_sha(
                self.evidence_manifest_sha256
            ):
                raise ValueError("empirical ensemble lacks its exact evidence-manifest hash")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_manifests"] = [asdict(item) for item in self.model_manifests]
        return value


@dataclass(frozen=True)
class VerifiedSeedRun:
    seed: int
    variant: str
    evidence_class: str
    contract_sha256: str
    training_cutoff_ns: int
    evidence_manifest_sha256: str
    registered_split_sha256: str
    training_config_sha256: str
    run_manifest_sha256: str
    trace: pd.DataFrame
    model_manifests: tuple[ModelManifest, ...]
    checkpoint_paths: Mapping[str, Path]


@dataclass(frozen=True)
class LoadedEnsembleBundle:
    """A disk-authenticated ensemble trace and its complete evidence identity."""

    trace: pd.DataFrame
    product: EnsembleTraceProduct
    manifest_path: Path
    manifest_sha256: str
    registered_split_path: Path
    source_run_manifest_paths: Mapping[int, Path]

    def validate(self) -> None:
        self.product.validate()
        if not self.manifest_path.is_file() or sha256_file(self.manifest_path) != self.manifest_sha256:
            raise ValueError("ensemble bundle manifest changed after verification")
        if not self.registered_split_path.is_file():
            raise ValueError("ensemble bundle registered split is missing")
        if set(self.source_run_manifest_paths) != {
            seed for seed, _ in self.product.member_run_sha256
        }:
            raise ValueError("ensemble bundle run bindings differ from its product")
        trace = validate_ensemble_trace_frame(self.trace)
        if (
            len(trace) != self.product.trace_rows
            or ensemble_trace_content_sha256(trace)
            != self.product.trace_content_sha256
        ):
            raise ValueError("ensemble bundle trace differs from its product")


def _valid_sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _resolve_run_artifact(
    declaration: object, *, parent: Path, label: str
) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(declaration, Mapping):
        raise ValueError(f"one-seed run lacks {label} artifact declaration")
    raw_path = declaration.get("path")
    expected = declaration.get("sha256")
    if not isinstance(raw_path, str) or not _valid_sha(expected):
        raise ValueError(f"one-seed run {label} artifact declaration is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"one-seed run {label} path must be run-relative")
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file():
        raise ValueError(f"one-seed run {label} artifact is missing or out of scope")
    if sha256_file(path) != expected:
        raise ValueError(f"one-seed run {label} artifact hash mismatch")
    return path, declaration


def _json_content_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_verified_seed_run(
    run_manifest_path: str | Path,
    *,
    contract: BaselineContract,
    evidence_class: str,
) -> VerifiedSeedRun:
    """Resolve and hash-check a complete one-seed training-run artifact graph."""

    source = Path(run_manifest_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("one-seed run manifest does not exist")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("schema_version", 0)) != 1
        or payload.get("artifact_type") != "BASE_GC_ONE_SEED_TRAINING_RUN"
        or payload.get("evidence_class") != evidence_class
        or payload.get("variant") != contract.variant
        or payload.get("contract_sha256") != contract.sha256
    ):
        raise ValueError("one-seed run typed identity differs from the ensemble contract")
    seed = int(payload.get("seed", -1))
    if seed not in contract.seed_ensemble:
        raise ValueError("one-seed run uses an unregistered policy seed")
    cutoff = int(payload.get("training_cutoff_ns", 0))
    evidence_hash = str(payload.get("evidence_manifest_sha256", ""))
    split_hash = str(payload.get("registered_split_sha256", ""))
    config_hash = str(payload.get("training_config_sha256", ""))
    if cutoff <= 0 or not all(_valid_sha(value) for value in (evidence_hash, split_hash, config_hash)):
        raise ValueError("one-seed run lacks exact cutoff/evidence/split/config bindings")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("one-seed run lacks its artifact graph")

    trace_path, trace_declaration = _resolve_run_artifact(
        artifacts.get("seed_decision_trace"), parent=source.parent, label="trace"
    )
    trace = validate_trace_frame(pd.read_parquet(trace_path))
    if str(trace_declaration.get("content_sha256", "")) != trace_content_sha256(trace):
        raise ValueError("one-seed run trace content hash mismatch")
    if set(trace["policy_seed"].astype(int)) != {seed}:
        raise ValueError("one-seed run trace policy seed differs from its manifest")

    config_path, config_declaration = _resolve_run_artifact(
        artifacts.get("training_config"), parent=source.parent, label="training config"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or _json_content_sha256(config) != config_hash
        or config_declaration.get("content_sha256") != config_hash
    ):
        raise ValueError("one-seed run training configuration hash mismatch")
    expected_config = {
        "contract_sha256": contract.sha256,
        "variant": contract.variant,
        "seed": seed,
        "training_cutoff_ns": cutoff,
        "evidence_class": evidence_class,
        "evidence_manifest_sha256": evidence_hash,
        "registered_split_sha256": split_hash,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("one-seed run configuration differs from its identity")

    manifests_path, _ = _resolve_run_artifact(
        artifacts.get("model_manifests"), parent=source.parent, label="model manifests"
    )
    manifest_payload = json.loads(manifests_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("one-seed model-manifest artifact must be a JSON list")
    manifests: list[ModelManifest] = []
    for item in manifest_payload:
        if not isinstance(item, Mapping):
            raise ValueError("one-seed model-manifest entry is invalid")
        values = dict(item)
        values["action_values"] = tuple(values.get("action_values", ()))
        manifest = ModelManifest(**values)
        manifest.validate()
        if (
            manifest.seed != seed
            or manifest.variant != contract.variant
            or manifest.contract_sha256 != contract.sha256
            or manifest.state_dimension
            != contract.agent(manifest.agent).observation_dimension
            or manifest.training_config_sha256 != config_hash
            or manifest.training_data_manifest_sha256 != evidence_hash
            or manifest.evidence_manifest_sha256 != evidence_hash
            or manifest.registered_split_sha256 != split_hash
            or manifest.training_cutoff_ns != cutoff
        ):
            raise ValueError("one-seed model manifest differs from run/contract evidence")
        manifests.append(manifest)
    if len(manifests) != 3 or {value.agent for value in manifests} != {"H1", "M15", "M5"}:
        raise ValueError("one-seed run requires exactly H1/M15/M5 model manifests")

    support_path, _ = _resolve_run_artifact(
        artifacts.get("support"), parent=source.parent, label="support"
    )
    support = json.loads(support_path.read_text(encoding="utf-8"))
    if (
        not isinstance(support, list)
        or len(support) != 3
        or {str(value.get("agent")) for value in support if isinstance(value, Mapping)}
        != {"H1", "M15", "M5"}
        or any(
            int(value.get("matured_training_transitions", -1))
            < contract.optimization.replay_warmup
            for value in support
            if isinstance(value, Mapping)
        )
    ):
        raise ValueError("one-seed run support artifact is incomplete")

    checkpoint_declarations = payload.get("checkpoints")
    if not isinstance(checkpoint_declarations, Mapping):
        raise ValueError("one-seed run lacks checkpoint declarations")
    checkpoints: dict[str, Path] = {}
    if set(checkpoint_declarations) != {value.model_id for value in manifests}:
        raise ValueError("one-seed run checkpoint set differs from model manifests")
    for manifest in manifests:
        checkpoint, declaration = _resolve_run_artifact(
            checkpoint_declarations.get(manifest.model_id),
            parent=source.parent,
            label=f"checkpoint {manifest.model_id}",
        )
        if manifest.checkpoint_sha256 != declaration.get("sha256"):
            raise ValueError("one-seed checkpoint differs from its model manifest")
        checkpoints[manifest.model_id] = checkpoint
    return VerifiedSeedRun(
        seed=seed,
        variant=contract.variant,
        evidence_class=evidence_class,
        contract_sha256=contract.sha256,
        training_cutoff_ns=cutoff,
        evidence_manifest_sha256=evidence_hash,
        registered_split_sha256=split_hash,
        training_config_sha256=config_hash,
        run_manifest_sha256=sha256_file(source),
        trace=trace,
        model_manifests=tuple(manifests),
        checkpoint_paths=checkpoints,
    )


def majority_action(actions: Sequence[int]) -> int:
    """Return the unique majority; return flat for every tie."""

    if not actions:
        raise ValueError("majority_action requires at least one seed action")
    values = [int(value) for value in actions]
    if any(value not in ACTIONS for value in values):
        raise ValueError(f"ensemble actions must be in {ACTIONS}")
    counts = Counter(values)
    maximum = max(counts.values())
    winners = [action for action, count in counts.items() if count == maximum]
    return winners[0] if len(winners) == 1 else 0


def aggregate_seed_traces(
    seed_traces: Mapping[int, pd.DataFrame],
    *,
    expected_seeds: Sequence[int] = (1103, 2207, 3301, 4409, 5501),
) -> pd.DataFrame:
    """Aggregate seed policies only after proving identical causal calendars."""

    expected = tuple(int(seed) for seed in expected_seeds)
    if len(expected) != 5 or len(set(expected)) != 5:
        raise ValueError("registered ensemble requires five distinct expected seeds")
    if set(int(seed) for seed in seed_traces) != set(expected):
        raise ValueError("seed traces do not match the five registered ensemble seeds")
    ordered = [(int(seed), validate_trace_frame(frame)) for seed, frame in seed_traces.items()]
    ordered.sort(key=lambda item: item[0])
    reference = ordered[0][1].set_index("decision_id", drop=False).sort_index()
    indexed: list[pd.DataFrame] = []
    for seed, trace in ordered:
        candidate = trace.set_index("decision_id", drop=False).sort_index()
        if not candidate.index.equals(reference.index):
            raise ValueError(f"seed {seed} uses a different decision calendar")
        pd.testing.assert_frame_equal(
            candidate.loc[:, INVARIANT_ENSEMBLE_COLUMNS],
            reference.loc[:, INVARIANT_ENSEMBLE_COLUMNS],
            check_dtype=False,
            obj=f"seed {seed} invariant trace",
        )
        indexed.append(candidate)

    rows: list[dict[str, object]] = []
    for index in reference.index:
        votes = {
            seed: int(frame.at[index, "final_target"])
            for (seed, _), frame in zip(ordered, indexed, strict=True)
        }
        final_target = majority_action(list(votes.values()))
        winning_seed = min(seed for seed, action in votes.items() if action == final_target)
        canonical_votes = json.dumps(votes, sort_keys=True, separators=(",", ":"))
        row = {
            column: reference.at[index, column]
            for column in INVARIANT_ENSEMBLE_COLUMNS
        }
        row.update(
            {
                "final_target": final_target,
                "vote_short": sum(action == -1 for action in votes.values()),
                "vote_flat": sum(action == 0 for action in votes.values()),
                "vote_long": sum(action == 1 for action in votes.values()),
                "vote_vector_sha256": hashlib.sha256(
                    canonical_votes.encode("utf-8")
                ).hexdigest(),
                "winning_policy_seed": winning_seed,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows, columns=ENSEMBLE_TRACE_COLUMNS)
    return validate_ensemble_trace_frame(result)


def validate_ensemble_trace_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(ENSEMBLE_TRACE_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"ensemble trace is missing columns: {missing}")
    trace = frame.loc[:, ENSEMBLE_TRACE_COLUMNS].copy()
    if trace["decision_id"].astype(str).duplicated().any():
        raise ValueError("ensemble decision_id must be unique")
    for column in (
        "bar_end_time_ns",
        "available_time_ns",
        "decision_time_ns",
        "submission_time_ns",
        "arrival_time_ns",
    ):
        trace[column] = pd.to_numeric(trace[column], errors="raise").astype("int64")
    clocks = trace[
        [
            "bar_end_time_ns",
            "available_time_ns",
            "decision_time_ns",
            "submission_time_ns",
            "arrival_time_ns",
        ]
    ].to_numpy(dtype=np.int64)
    if bool((np.diff(clocks, axis=1) < 0).any()):
        raise ValueError("ensemble causal clocks are not monotone")
    for column in ("final_target", "vote_short", "vote_flat", "vote_long"):
        trace[column] = pd.to_numeric(trace[column], errors="raise").astype("int64")
    if not trace["final_target"].isin(ACTIONS).all():
        raise ValueError("ensemble final_target contains an invalid action")
    vote_counts = trace[["vote_short", "vote_flat", "vote_long"]]
    if bool((vote_counts < 0).any().any()) or bool(vote_counts.sum(axis=1).ne(5).any()):
        raise ValueError("ensemble vote counts must be nonnegative and sum to five")
    reconstructed = [
        majority_action([-1] * int(row.vote_short) + [0] * int(row.vote_flat) + [1] * int(row.vote_long))
        for row in vote_counts.itertuples(index=False)
    ]
    if not np.array_equal(trace["final_target"].to_numpy(), np.asarray(reconstructed)):
        raise ValueError("ensemble final_target differs from registered vote rule")
    trace["winning_policy_seed"] = pd.to_numeric(
        trace["winning_policy_seed"], errors="raise"
    ).astype("int64")
    if not trace["winning_policy_seed"].isin((1103, 2207, 3301, 4409, 5501)).all():
        raise ValueError("ensemble winning_policy_seed is unregistered")
    for column in _SHA_COLUMNS:
        if not trace[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"ensemble {column} contains an invalid SHA-256")
    trace["m5_decision_taken"] = trace["m5_decision_taken"].astype(bool)
    trace["date"] = require_exact_globex_session_dates(
        trace["date"], trace["bar_end_time_ns"], label="ensemble trace"
    ).to_numpy()
    trace = trace.sort_values(
        ["available_time_ns", "raw_symbol", "decision_id"], kind="mergesort"
    ).reset_index(drop=True)
    for _, group in trace.groupby(
        ["raw_symbol", "continuity_segment_id", "reward_boundary_id"],
        sort=False,
        dropna=False,
    ):
        changed = group["final_target"].ne(group["final_target"].shift())
        if len(changed):
            changed.iloc[0] = False
        if bool((changed & ~group["m5_decision_taken"]).any()):
            raise ValueError("voted target changed outside a registered M5 renewal")
    return trace


def ensemble_trace_content_sha256(frame: pd.DataFrame) -> str:
    trace = validate_ensemble_trace_frame(frame)
    records = []
    for row in trace.to_dict(orient="records"):
        records.append(
            {
                column: (
                    value.item() if hasattr(value, "item") else value
                )
                for column, value in row.items()
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_registered_split(
    path: str | Path,
    *,
    expected_sha256: str,
    evidence_class: str,
    training_cutoff_ns: int,
    trace: pd.DataFrame,
) -> None:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise ValueError("registered split artifact is missing or hash-mismatched")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("schema_version", 0)) != 1
        or payload.get("artifact_type") != "BASE_GC_REGISTERED_SPLIT"
        or payload.get("evidence_class") != evidence_class
        or payload.get("passed") is not True
    ):
        raise ValueError("registered split artifact has the wrong typed contract")
    if int(payload.get("training_cutoff_ns", 0)) != training_cutoff_ns:
        raise ValueError("registered split cutoff differs from trained models")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping) or not evaluation.get("partition_id"):
        raise ValueError("registered split lacks its evaluation partition")
    start = int(evaluation.get("start_time_ns", 0))
    end = int(evaluation.get("end_time_ns", 0))
    observed_start = int(trace["decision_time_ns"].min())
    observed_end = int(trace["arrival_time_ns"].max())
    if not (training_cutoff_ns < start <= observed_start <= observed_end <= end):
        raise ValueError("ensemble clocks fall outside the exact registered evaluation split")


def create_ensemble_trace_product(
    seed_traces: Mapping[int, pd.DataFrame] | None = None,
    *,
    contract: BaselineContract,
    model_manifests: Iterable[ModelManifest] = (),
    evidence_class: str,
    product_id: str,
    registered_split_path: str | Path | None = None,
    checkpoint_paths: Mapping[str, str | Path] | None = None,
    seed_run_manifest_paths: Mapping[int, str | Path] | None = None,
) -> tuple[pd.DataFrame, EnsembleTraceProduct]:
    supplied_manifests = tuple(model_manifests)
    member_runs: tuple[tuple[int, str], ...] = ()
    evidence_hash: str | None = None
    if evidence_class != "SYNTHETIC_ONLY":
        if seed_traces is not None or supplied_manifests or checkpoint_paths is not None:
            raise ValueError(
                "empirical ensembles reject detached traces/models; provide verified "
                "BASE_GC_ONE_SEED_TRAINING_RUN manifests"
            )
        canonical_run_paths = (
            None
            if seed_run_manifest_paths is None
            else {int(seed): path for seed, path in seed_run_manifest_paths.items()}
        )
        if canonical_run_paths is None or set(canonical_run_paths) != set(contract.seed_ensemble):
            raise ValueError("empirical ensemble requires five resolved one-seed run manifests")
        runs = tuple(
            load_verified_seed_run(
                canonical_run_paths[seed],
                contract=contract,
                evidence_class=evidence_class,
            )
            for seed in contract.seed_ensemble
        )
        if tuple(value.seed for value in runs) != contract.seed_ensemble:
            raise ValueError("one-seed run path keys differ from their policy seeds")
        cutoffs = {value.training_cutoff_ns for value in runs}
        evidence_hashes = {value.evidence_manifest_sha256 for value in runs}
        split_hashes = {value.registered_split_sha256 for value in runs}
        if len(cutoffs) != 1 or len(evidence_hashes) != 1 or len(split_hashes) != 1:
            raise ValueError("one-seed runs do not share one cutoff/evidence/split")
        cutoff_value = next(iter(cutoffs))
        seed_traces = {
            value.seed: value.trace.loc[
                value.trace["available_time_ns"] > cutoff_value
            ].reset_index(drop=True)
            for value in runs
        }
        if any(frame.empty for frame in seed_traces.values()):
            raise ValueError("one-seed run lacks decisions after the registered cutoff")
        supplied_manifests = tuple(
            manifest for value in runs for manifest in value.model_manifests
        )
        member_runs = tuple(
            (value.seed, value.run_manifest_sha256) for value in runs
        )
        evidence_hash = next(iter(evidence_hashes))
    else:
        if seed_run_manifest_paths is not None:
            raise ValueError("synthetic ensemble does not consume empirical run manifests")
        if seed_traces is None:
            raise ValueError("synthetic ensemble requires explicit seed traces")

    assert seed_traces is not None
    trace = aggregate_seed_traces(
        seed_traces, expected_seeds=contract.seed_ensemble
    )
    manifests = supplied_manifests
    member_hashes = tuple(
        sorted((int(seed), trace_content_sha256(value)) for seed, value in seed_traces.items())
    )
    split_hash: str | None = None
    cutoff: int | None = None
    if evidence_class != "SYNTHETIC_ONLY":
        expected_keys = {
            (seed, agent)
            for seed in contract.seed_ensemble
            for agent in ("H1", "M15", "M5")
        }
        observed_keys = {(value.seed, value.agent) for value in manifests}
        if observed_keys != expected_keys or len(manifests) != 15:
            raise ValueError("empirical ensemble requires exactly 5 seed x 3 agent manifests")
        for manifest in manifests:
            manifest.validate()
            if manifest.contract_sha256 != contract.sha256 or manifest.variant != contract.variant:
                raise ValueError("ensemble model manifest differs from the frozen contract")
            if manifest.state_dimension != contract.agent(manifest.agent).observation_dimension:
                raise ValueError("ensemble model state dimension differs from contract")
        split_hashes = {value.registered_split_sha256 for value in manifests}
        cutoffs = {value.training_cutoff_ns for value in manifests}
        if len(split_hashes) != 1 or len(cutoffs) != 1:
            raise ValueError("ensemble models do not share one split and cutoff")
        split_hash = next(iter(split_hashes))
        cutoff = next(iter(cutoffs))
        if int(trace["decision_time_ns"].min()) <= cutoff:
            raise ValueError("empirical ensemble trace overlaps its training cutoff")
        if registered_split_path is None:
            raise ValueError("empirical ensemble requires the resolved registered split artifact")
        _verify_registered_split(
            registered_split_path,
            expected_sha256=split_hash,
            evidence_class=evidence_class,
            training_cutoff_ns=cutoff,
            trace=trace,
        )
    product = EnsembleTraceProduct(
        schema_version=1,
        product_id=product_id,
        variant=contract.variant,
        evidence_class=evidence_class,
        contract_sha256=contract.sha256,
        registered_split_sha256=split_hash,
        training_cutoff_ns=cutoff,
        trace_content_sha256=ensemble_trace_content_sha256(trace),
        trace_rows=len(trace),
        member_trace_sha256=member_hashes,
        model_manifests=manifests,
        claim_scope=(
            "Five independent whole-hierarchy seeds vote only on the executable "
            "M5 target. Member internal states/actions remain in their own traces."
        ),
        member_run_sha256=member_runs,
        evidence_manifest_sha256=evidence_hash,
    )
    product.validate()
    return trace, product


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
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


def _ensemble_product_from_mapping(value: Mapping[str, Any]) -> EnsembleTraceProduct:
    raw_manifests = value.get("model_manifests")
    if not isinstance(raw_manifests, list):
        raise ValueError("ensemble bundle product lacks model manifests")
    manifests: list[ModelManifest] = []
    for raw in raw_manifests:
        if not isinstance(raw, Mapping):
            raise ValueError("ensemble bundle contains an invalid model manifest")
        fields = dict(raw)
        fields["action_values"] = tuple(fields.get("action_values", ()))
        manifest = ModelManifest(**fields)
        manifest.validate()
        manifests.append(manifest)
    product = EnsembleTraceProduct(
        schema_version=int(value.get("schema_version", 0)),
        product_id=str(value.get("product_id", "")),
        variant=str(value.get("variant", "")),
        evidence_class=str(value.get("evidence_class", "")),
        contract_sha256=str(value.get("contract_sha256", "")),
        registered_split_sha256=(
            None
            if value.get("registered_split_sha256") is None
            else str(value.get("registered_split_sha256"))
        ),
        training_cutoff_ns=(
            None
            if value.get("training_cutoff_ns") is None
            else int(value.get("training_cutoff_ns"))
        ),
        trace_content_sha256=str(value.get("trace_content_sha256", "")),
        trace_rows=int(value.get("trace_rows", -1)),
        member_trace_sha256=tuple(
            (int(item[0]), str(item[1]))
            for item in value.get("member_trace_sha256", ())
        ),
        model_manifests=tuple(manifests),
        claim_scope=str(value.get("claim_scope", "")),
        member_run_sha256=tuple(
            (int(item[0]), str(item[1]))
            for item in value.get("member_run_sha256", ())
        ),
        evidence_manifest_sha256=(
            None
            if value.get("evidence_manifest_sha256") is None
            else str(value.get("evidence_manifest_sha256"))
        ),
    )
    product.validate()
    return product


def write_empirical_ensemble_bundle(
    trace: pd.DataFrame,
    product: EnsembleTraceProduct,
    output_directory: str | Path,
    *,
    registered_split_path: str | Path,
    seed_run_manifest_paths: Mapping[int, str | Path],
) -> Path:
    """Write an immutable empirical ensemble trace bundle.

    The bundle does not replace the five one-seed artifact graphs.  It binds
    their exact manifest bytes and retains their resolved paths so the loader
    can authenticate every checkpoint/model/trace again before returning the
    ensemble to a downstream RENT producer.
    """

    product.validate()
    if product.evidence_class == "SYNTHETIC_ONLY":
        raise ValueError("the disk ensemble bundle is restricted to empirical products")
    canonical = validate_ensemble_trace_frame(trace)
    if (
        len(canonical) != product.trace_rows
        or ensemble_trace_content_sha256(canonical) != product.trace_content_sha256
    ):
        raise ValueError("ensemble trace differs from the supplied product")
    expected_run_hashes = dict(product.member_run_sha256)
    run_paths = {int(seed): Path(path).expanduser().resolve() for seed, path in seed_run_manifest_paths.items()}
    if set(run_paths) != set(expected_run_hashes):
        raise ValueError("ensemble bundle requires the five product run manifests")
    run_bindings: list[dict[str, object]] = []
    for seed in sorted(run_paths):
        path = run_paths[seed]
        if not path.is_file() or sha256_file(path) != expected_run_hashes[seed]:
            raise ValueError(f"one-seed run {seed} is missing or hash-mismatched")
        run_bindings.append(
            {"seed": seed, "path": str(path), "sha256": expected_run_hashes[seed]}
        )

    split_source = Path(registered_split_path).expanduser().resolve()
    if (
        product.registered_split_sha256 is None
        or not split_source.is_file()
        or sha256_file(split_source) != product.registered_split_sha256
    ):
        raise ValueError("registered split is missing or differs from the ensemble product")
    split = json.loads(split_source.read_text(encoding="utf-8"))
    if not isinstance(split, Mapping):
        raise ValueError("registered split must be a JSON object")
    evaluation = split.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("registered split lacks its evaluation partition")
    evaluation_id = str(evaluation.get("partition_id", ""))
    evaluation_start = int(evaluation.get("start_time_ns", 0))
    evaluation_end = int(evaluation.get("end_time_ns", 0))
    cutoff = int(product.training_cutoff_ns or 0)
    if not evaluation_id or not cutoff < evaluation_start <= evaluation_end:
        raise ValueError("registered evaluation window is invalid")
    if canonical.empty:
        raise ValueError("empirical ensemble bundle cannot contain an empty trace")
    if not (
        evaluation_start <= int(canonical["decision_time_ns"].min())
        and int(canonical["arrival_time_ns"].max()) <= evaluation_end
    ):
        raise ValueError("ensemble trace is outside its registered evaluation window")
    if pd.to_datetime(canonical["date"], errors="raise").dt.date.max() > pd.Timestamp("2025-08-31").date():
        raise PermissionError("development ensemble bundle cannot cross 2025-08-31")

    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace an ensemble bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        trace_path = temporary / "ensemble_trace.parquet"
        split_path = temporary / "registered_split.json"
        _atomic_parquet(trace_path, canonical)
        shutil.copyfile(split_source, split_path)
        manifest = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_ENSEMBLE_TRACE_BUNDLE",
            "evidence_class": product.evidence_class,
            "variant": product.variant,
            "contract_sha256": product.contract_sha256,
            "training_cutoff_ns": cutoff,
            "registered_split_sha256": product.registered_split_sha256,
            "evidence_manifest_sha256": product.evidence_manifest_sha256,
            "evaluation": {
                "partition_id": evaluation_id,
                "start_time_ns": evaluation_start,
                "end_time_ns": evaluation_end,
            },
            "trace": {
                "path": trace_path.name,
                "sha256": sha256_file(trace_path),
                "content_sha256": product.trace_content_sha256,
                "row_count": product.trace_rows,
            },
            "registered_split": {
                "path": split_path.name,
                "sha256": sha256_file(split_path),
            },
            "source_one_seed_runs": run_bindings,
            "product": product.to_dict(),
            "claim_scope": (
                "Authenticated Base-GC ensemble decisions only; execution and "
                "profitability remain properties of the independent common ledger."
            ),
        }
        _atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "manifest.json"


def _bundle_relative_artifact(
    declaration: object, *, parent: Path, label: str
) -> Path:
    if not isinstance(declaration, Mapping):
        raise ValueError(f"ensemble bundle lacks {label} declaration")
    raw_path = declaration.get("path")
    expected = declaration.get("sha256")
    if not isinstance(raw_path, str) or not _valid_sha(expected):
        raise ValueError(f"ensemble bundle {label} declaration is invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"ensemble bundle {label} path must be bundle-relative")
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file():
        raise ValueError(f"ensemble bundle {label} is missing or out of scope")
    if sha256_file(path) != expected:
        raise ValueError(f"ensemble bundle {label} hash mismatch")
    return path


def load_empirical_ensemble_bundle(
    manifest_path: str | Path,
) -> LoadedEnsembleBundle:
    """Authenticate a written ensemble bundle and all five source runs."""

    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("ensemble bundle manifest does not exist")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("schema_version", 0)) != 1
        or payload.get("artifact_type") != "BASE_GC_ENSEMBLE_TRACE_BUNDLE"
    ):
        raise ValueError("unsupported ensemble bundle manifest")
    raw_product = payload.get("product")
    if not isinstance(raw_product, Mapping):
        raise ValueError("ensemble bundle lacks its typed product")
    product = _ensemble_product_from_mapping(raw_product)
    identity = {
        "evidence_class": product.evidence_class,
        "variant": product.variant,
        "contract_sha256": product.contract_sha256,
        "training_cutoff_ns": product.training_cutoff_ns,
        "registered_split_sha256": product.registered_split_sha256,
        "evidence_manifest_sha256": product.evidence_manifest_sha256,
    }
    if any(payload.get(name) != value for name, value in identity.items()):
        raise ValueError("ensemble bundle identity differs from its product")
    contract = registered_contract(product.variant)
    if contract.sha256 != product.contract_sha256:
        raise ValueError("ensemble bundle differs from the registered Base-GC contract")

    trace_path = _bundle_relative_artifact(
        payload.get("trace"), parent=source.parent, label="trace"
    )
    trace_declaration = payload["trace"]
    assert isinstance(trace_declaration, Mapping)
    trace = validate_ensemble_trace_frame(pd.read_parquet(trace_path))
    if (
        int(trace_declaration.get("row_count", -1)) != len(trace)
        or str(trace_declaration.get("content_sha256", ""))
        != ensemble_trace_content_sha256(trace)
        or len(trace) != product.trace_rows
        or ensemble_trace_content_sha256(trace) != product.trace_content_sha256
    ):
        raise ValueError("ensemble bundle trace content/count mismatch")

    split_path = _bundle_relative_artifact(
        payload.get("registered_split"), parent=source.parent, label="registered split"
    )
    if sha256_file(split_path) != product.registered_split_sha256:
        raise ValueError("ensemble bundle split differs from the product")
    _verify_registered_split(
        split_path,
        expected_sha256=str(product.registered_split_sha256),
        evidence_class=product.evidence_class,
        training_cutoff_ns=int(product.training_cutoff_ns or 0),
        trace=trace,
    )
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("ensemble bundle lacks its evaluation identity")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if dict(evaluation) != dict(split_payload.get("evaluation", {})):
        raise ValueError("ensemble bundle evaluation window differs from its split")

    raw_runs = payload.get("source_one_seed_runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != 5:
        raise ValueError("ensemble bundle requires five one-seed run bindings")
    run_paths: dict[int, Path] = {}
    expected_runs = dict(product.member_run_sha256)
    expected_member_traces = dict(product.member_trace_sha256)
    observed_manifests: list[ModelManifest] = []
    for declaration in raw_runs:
        if not isinstance(declaration, Mapping):
            raise ValueError("ensemble bundle contains an invalid run binding")
        seed = int(declaration.get("seed", -1))
        path = Path(str(declaration.get("path", ""))).expanduser().resolve()
        expected = str(declaration.get("sha256", ""))
        if (
            seed in run_paths
            or expected_runs.get(seed) != expected
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"ensemble bundle one-seed run {seed} failed its binding")
        run = load_verified_seed_run(
            path, contract=contract, evidence_class=product.evidence_class
        )
        if (
            run.run_manifest_sha256 != expected
            or trace_content_sha256(
                run.trace.loc[
                    run.trace["available_time_ns"] > int(product.training_cutoff_ns or 0)
                ].reset_index(drop=True)
            )
            != expected_member_traces.get(seed)
        ):
            raise ValueError(f"ensemble bundle one-seed run {seed} content mismatch")
        observed_manifests.extend(run.model_manifests)
        run_paths[seed] = path
    if tuple(observed_manifests) != product.model_manifests:
        raise ValueError("ensemble bundle model manifests differ from source runs")

    bundle = LoadedEnsembleBundle(
        trace=trace,
        product=product,
        manifest_path=source,
        manifest_sha256=sha256_file(source),
        registered_split_path=split_path,
        source_run_manifest_paths=MappingProxyType(dict(sorted(run_paths.items()))),
    )
    bundle.validate()
    return bundle
