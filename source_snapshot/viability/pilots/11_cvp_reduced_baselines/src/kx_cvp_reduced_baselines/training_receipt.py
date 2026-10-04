"""Adopt and verify the ten completed Pilot 10 learner artifacts.

Pilot 11 performs no additional fitting.  Its only admissible learned inputs
are the five completed recurrent-PPO members and the five completed
LightGBM fitted-Q members from Pilot 10.  This module records their exact
manifests and model/scaler hashes in a new immutable receipt before any
historical evaluation data may be opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from kx_cvp_baselines.training import verify_training_member

from .artifacts import (
    atomic_json,
    canonical_json_sha256,
    file_sha256,
    project_file_record,
    project_relative,
    read_json_object,
    resolve_project_path,
    verify_project_file,
)
from .contract import (
    CANONICAL_OUTPUT_ROOT,
    LEARNER_ARMS,
    PROTOCOL_ID,
    REQUIRED_TRAINING_JOBS,
    SEEDS,
)
from .protocol import (
    assert_frozen_contracts,
    implementation_binding,
    protocol_binding,
    reject_forbidden_text,
    validate_workspace_output,
    verify_implementation_binding,
)


ADOPTED_TRAINING_ARTIFACT = "CVP_BASELINE_REDUCED_ADOPTED_TRAINING_RECEIPT"
ADOPTED_TRAINING_FILENAME = "adopted_training_receipt.json"
SOURCE_EXPERIMENT_ID = "CVP_BASELINE_SUITE_V1"
SOURCE_TRAINING_ROOT = (
    "viability/artifacts/development/cvp_baseline_suite_v1"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "31ce433cf5c5e9a401ee500ff7a990caee91f42267083552095e6bd125a182a0"
)
SOURCE_ENVIRONMENT_SHA256 = (
    "cf9f2d1538beeea9423446e54d97e6ce3bbb69ee20aa028d269c207f676089fe"
)


def _output_root(value: Path | str | None) -> Path:
    return validate_workspace_output(
        CANONICAL_OUTPUT_ROOT if value is None else value,
    )


def _source_member_path(arm: str, seed: int) -> Path:
    return resolve_project_path(
        f"{SOURCE_TRAINING_ROOT}/{arm}/seed_{int(seed)}/manifest.json"
    )


def _member_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    model = manifest.get("model")
    scaler = manifest.get("scaler")
    if not isinstance(model, Mapping) or not isinstance(scaler, Mapping):
        raise ValueError("adopted member lacks model/scaler bindings")
    return {
        "arm": str(manifest["arm"]),
        "seed": int(manifest["seed"]),
        "status": "ADOPTED_COMPLETE",
        "source_manifest": project_file_record(path),
        "model_sha256": str(model.get("sha256", "")),
        "model_state_sha256": str(model.get("state_sha256", "")),
        "scaler_sha256": str(scaler.get("sha256", "")),
        "scaler_canonical_sha256": str(scaler.get("canonical_sha256", "")),
    }


def build_adopted_training_receipt(
    output_root: Path | str | None = None,
) -> Path:
    """Create the immutable two-arm x five-seed adoption receipt.

    Contract validation precedes all source-member access.  The receipt is
    result-independent: only development training identities and hashes are
    read, never May/June/July prices, traces, or economic summaries.
    """

    reject_forbidden_text(output_root or "")
    assert_frozen_contracts()
    root = _output_root(output_root)
    destination = root / ADOPTED_TRAINING_FILENAME
    if destination.is_file():
        verify_adopted_training_receipt(destination)
        return destination
    if root.exists() and any(root.iterdir()):
        # New evaluation output may never precede its training receipt.  An
        # empty pre-created root is harmless; any content is not.
        raise FileExistsError(
            "reduced-suite output exists before its adopted-training receipt"
        )

    members: list[dict[str, object]] = []
    common_dataset: object | None = None
    common_source_implementation: object | None = None
    for arm in LEARNER_ARMS:
        for seed in SEEDS:
            path = _source_member_path(arm, seed)
            value = verify_training_member(
                path, expected_arm=arm, expected_seed=seed,
            )
            if (
                value.get("holdout_read") is not False
                or value.get("evaluation_accessed") is not False
            ):
                raise ValueError("adopted training member accessed evaluation data")
            if common_dataset is None:
                common_dataset = value.get("dataset_manifest")
                common_source_implementation = value.get("implementation")
            elif (
                value.get("dataset_manifest") != common_dataset
                or value.get("implementation") != common_source_implementation
            ):
                raise ValueError("adopted members have inconsistent frozen lineage")
            members.append(_member_record(path, value))

    if len(members) != REQUIRED_TRAINING_JOBS:
        raise AssertionError("reduced suite did not adopt exactly ten jobs")
    if not isinstance(common_source_implementation, Mapping):
        raise ValueError("adopted source implementation binding is absent")
    environment = common_source_implementation.get("environment")
    if (
        common_source_implementation.get("aggregate_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or not isinstance(environment, Mapping)
        or environment.get("aggregate_sha256") != SOURCE_ENVIRONMENT_SHA256
    ):
        raise ValueError("Pilot 10 source/environment lineage differs from the freeze")

    value = {
        "schema_version": 1,
        "artifact_type": ADOPTED_TRAINING_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "status": "COMPLETE",
        "classification": "DEVELOPMENT",
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_training_root": SOURCE_TRAINING_ROOT,
        "source_implementation_aggregate_sha256": SOURCE_IMPLEMENTATION_SHA256,
        "source_environment_aggregate_sha256": SOURCE_ENVIRONMENT_SHA256,
        "source_implementation": common_source_implementation,
        "development_dataset": common_dataset,
        "arms": list(LEARNER_ARMS),
        "seeds": list(SEEDS),
        "required_jobs": REQUIRED_TRAINING_JOBS,
        "members": members,
        "member_records_sha256": canonical_json_sha256(members),
        "all_registered_members_retained": True,
        "economic_selection_used": False,
        "adoption_basis": "OWNER_SCOPE_REDUCTION_BEFORE_REPLAY",
        "holdout_read": False,
        "evaluation_accessed": False,
        "source_partial_artifacts_modified": False,
        "new_training_performed": False,
        "august_2026_accessed": False,
    }
    atomic_json(destination, value)
    verify_adopted_training_receipt(destination)
    return destination


def verify_adopted_training_receipt(path: Path | str) -> dict[str, object]:
    """Deeply verify the exact ten-job Pilot 11 training prerequisite."""

    reject_forbidden_text(path)
    assert_frozen_contracts()
    source = Path(path).resolve()
    value = read_json_object(source)
    members = value.get("members")
    if (
        source.name != ADOPTED_TRAINING_FILENAME
        or validate_workspace_output(source.parent) != source.parent
        or value.get("schema_version") != 1
        or value.get("artifact_type") != ADOPTED_TRAINING_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("classification") != "DEVELOPMENT"
        or value.get("source_experiment_id") != SOURCE_EXPERIMENT_ID
        or value.get("source_training_root") != SOURCE_TRAINING_ROOT
        or value.get("source_implementation_aggregate_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or value.get("source_environment_aggregate_sha256")
        != SOURCE_ENVIRONMENT_SHA256
        or value.get("arms") != list(LEARNER_ARMS)
        or value.get("seeds") != list(SEEDS)
        or value.get("required_jobs") != REQUIRED_TRAINING_JOBS
        or value.get("all_registered_members_retained") is not True
        or value.get("economic_selection_used") is not False
        or value.get("adoption_basis") != "OWNER_SCOPE_REDUCTION_BEFORE_REPLAY"
        or value.get("holdout_read") is not False
        or value.get("evaluation_accessed") is not False
        or value.get("source_partial_artifacts_modified") is not False
        or value.get("new_training_performed") is not False
        or value.get("august_2026_accessed") is not False
        or not isinstance(members, list)
        or len(members) != REQUIRED_TRAINING_JOBS
        or value.get("member_records_sha256") != canonical_json_sha256(members)
    ):
        raise ValueError("reduced adopted-training receipt identity differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("reduced adopted-training protocol binding differs")
    verify_implementation_binding(value.get("implementation"))

    source_implementation = value.get("source_implementation")
    if not isinstance(source_implementation, Mapping):
        raise ValueError("source implementation binding is malformed")
    environment = source_implementation.get("environment")
    if (
        source_implementation.get("aggregate_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or not isinstance(environment, Mapping)
        or environment.get("aggregate_sha256") != SOURCE_ENVIRONMENT_SHA256
    ):
        raise ValueError("source implementation/environment digest differs")

    expected = [
        (arm, seed) for arm in LEARNER_ARMS for seed in SEEDS
    ]
    observed: list[tuple[str, int]] = []
    common_dataset: object | None = None
    for record in members:
        if not isinstance(record, Mapping) or set(record) != {
            "arm", "seed", "status", "source_manifest", "model_sha256",
            "model_state_sha256", "scaler_sha256", "scaler_canonical_sha256",
        }:
            raise ValueError("adopted member record schema differs")
        arm = str(record.get("arm"))
        seed = int(record.get("seed", -1))
        observed.append((arm, seed))
        if record.get("status") != "ADOPTED_COMPLETE":
            raise ValueError("adopted member is not complete")
        member_path = verify_project_file(record.get("source_manifest"))
        if member_path != _source_member_path(arm, seed):
            raise ValueError("adopted member path differs from its frozen source")
        member = verify_training_member(
            member_path, expected_arm=arm, expected_seed=seed,
        )
        model = member.get("model")
        scaler = member.get("scaler")
        if not isinstance(model, Mapping) or not isinstance(scaler, Mapping):
            raise ValueError("source training member bindings are malformed")
        if (
            record.get("model_sha256") != model.get("sha256")
            or record.get("model_state_sha256") != model.get("state_sha256")
            or record.get("scaler_sha256") != scaler.get("sha256")
            or record.get("scaler_canonical_sha256")
            != scaler.get("canonical_sha256")
            or member.get("implementation") != source_implementation
        ):
            raise ValueError("adopted member hash/lineage differs")
        if common_dataset is None:
            common_dataset = member.get("dataset_manifest")
        elif member.get("dataset_manifest") != common_dataset:
            raise ValueError("adopted member datasets differ")
    if observed != expected or value.get("development_dataset") != common_dataset:
        raise ValueError("adopted training registry is not the exact ordered product")
    return value


def adopted_member_records(
    receipt: Mapping[str, object], arm: str,
) -> tuple[Mapping[str, object], ...]:
    """Return the verified ordered source records for one registered arm."""

    members = receipt.get("members")
    if not isinstance(members, list):
        raise ValueError("adopted training receipt lacks member records")
    selected = tuple(
        record for record in members
        if isinstance(record, Mapping) and record.get("arm") == arm
    )
    if tuple(int(record.get("seed", -1)) for record in selected) != SEEDS:
        raise ValueError("adopted training receipt lacks an exact five-seed arm")
    return selected


__all__ = [
    "ADOPTED_TRAINING_ARTIFACT",
    "ADOPTED_TRAINING_FILENAME",
    "SOURCE_ENVIRONMENT_SHA256",
    "SOURCE_EXPERIMENT_ID",
    "SOURCE_IMPLEMENTATION_SHA256",
    "SOURCE_TRAINING_ROOT",
    "adopted_member_records",
    "build_adopted_training_receipt",
    "verify_adopted_training_receipt",
]
