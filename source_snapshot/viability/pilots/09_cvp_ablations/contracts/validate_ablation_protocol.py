#!/usr/bin/env python3
"""Fail-closed validator and physical access guard for CVP ablations.

The module deliberately uses only the standard library so it can be executed
before importing Torch, opening a dataset manifest, or constructing a model.
Unregistered and August requests are rejected lexically before any filesystem
listing or file open is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


EXPERIMENT_ID = "CVP_ABLATION_EXTENSION_V1"
PROTOCOL_FILE = "CVP_ABLATION_PROTOCOL.json"
ACCESS_LEDGER_FILE = "CVP_ABLATION_ACCESS_LEDGER.json"
FEATURE_ARM_REGISTRY_FILE = "CVP_ABLATION_FEATURE_ARM_REGISTRY.json"
ACCESS_LEDGER_RECEIPT_FILE = "CVP_ABLATION_ACCESS_LEDGER_RECEIPT.json"
FREEZE_RECEIPT_FILE = "CVP_ABLATION_FREEZE_RECEIPT.json"

ARMS = ("flow_profile_only", "depth_ofi_only", "no_publisher")
SEEDS = (1103, 2207, 3301, 4409, 5501)
PERIODS = ("may_2026", "june_july_2026")
SCENARIOS = ("nominal", "double_commission", "latency_20ms", "adverse_tick")
REGISTERED_CONTRASTS = (
    "cvp_hddqn_minus_restricted_representation",
    "cvp_hddqn_minus_flow_profile_only",
    "cvp_hddqn_minus_depth_ofi_only",
    "flow_profile_only_minus_restricted_representation",
    "depth_ofi_only_minus_restricted_representation",
    "cvp_hddqn_minus_no_publisher",
    "representation_two_by_two_interaction_secondary_diagnostic",
)
TECHNICAL_FAILURE_REASONS = frozenset(
    {
        "HASH_MISMATCH",
        "CAUSALITY_VIOLATION",
        "NONFINITE_STATE_OR_SCORE",
        "INCOMPLETE_TRAINING",
        "REWARD_LEDGER_MISMATCH",
        "ACTION_LEGALITY_VIOLATION",
        "TERMINAL_INVENTORY_NOT_FLAT",
        "UNEXPECTED_POLICY_FALLBACK",
    }
)

DATE_RANGES = {
    "development": (date(2025, 1, 1), date(2026, 4, 30)),
    "may_2026": (date(2026, 5, 1), date(2026, 5, 31)),
    "june_july_2026": (date(2026, 6, 1), date(2026, 7, 31)),
}
FORBIDDEN_START = date(2026, 8, 1)

REGISTERED_INPUTS = {
    ("train", "development"): (
        "viability/artifacts/development/"
        "v6_registered_train_dataset_terminal_eligible/manifest.json",
        "cdf3e233789a5370d02e1189bcc83e2dc10be4c23e1e92b311f2d4642f862503",
    ),
    ("evaluate", "may_2026"): (
        "viability/artifacts/development/"
        "v6_registered_validation_dataset_eligible/manifest.json",
        "f68bc69dd63524a59a65b58ee15a87fd5bd8dfb3dfe25d9a98b46a2d0772ef52",
    ),
    ("evaluate", "june_july_2026"): (
        "paper/sealed_extension/dataset/manifest.json",
        "c50bf2f5a04ecd8fe90c0cc92df7a3edaed879a1c68ac13e36b4588dcfdd018c",
    ),
}


class ContractValidationError(ValueError):
    """Raised when a frozen ablation contract or result matrix is invalid."""


@dataclass(frozen=True)
class ValidatedContracts:
    protocol: Mapping[str, Any]
    access_ledger: Mapping[str, Any]
    feature_arm_registry: Mapping[str, Any]
    protocol_canonical_sha256: str
    protocol_file_sha256: str
    access_ledger_file_sha256: str
    feature_arm_registry_file_sha256: str


@dataclass(frozen=True)
class AuthorizedInput:
    purpose: str
    period_id: str
    relative_path: str
    sha256: str


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractValidationError(f"JSON root must be an object: {path}")
    return value


def _iso_date(value: Any, field: str, errors: list[str]) -> date | None:
    if not isinstance(value, str):
        errors.append(f"{field} must be an ISO date string")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        errors.append(f"{field} is not a valid ISO date: {value!r}")
        return None


def _iso_timestamp(value: Any, field: str, errors: list[str]) -> None:
    if not isinstance(value, str):
        errors.append(f"{field} must be an ISO timestamp")
        return
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{field} must be an ISO timestamp")
        return
    if parsed.tzinfo is None:
        errors.append(f"{field} must include a timezone")


def _same_members(value: Any, expected: Sequence[Any], field: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != len(set(value)) or set(value) != set(expected):
        errors.append(f"{field} must contain exactly {list(expected)!r}")


def _plain_relative_path(value: Any, field: str, errors: list[str]) -> PurePosixPath | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{field} must be a nonempty repository-relative path")
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        errors.append(f"{field} must be a normalized repository-relative path")
        return None
    return path


def _validate_exact_period(
    value: Mapping[str, Any], period_id: str, classification: str, errors: list[str]
) -> None:
    expected_start, expected_end = DATE_RANGES[period_id]
    start = _iso_date(value.get("start_inclusive"), f"{period_id}.start_inclusive", errors)
    end = _iso_date(value.get("end_inclusive"), f"{period_id}.end_inclusive", errors)
    if start is not None and start != expected_start:
        errors.append(f"{period_id} start must be {expected_start.isoformat()}")
    if end is not None and end != expected_end:
        errors.append(f"{period_id} end must be {expected_end.isoformat()}")
    if value.get("label", value.get("classification")) != classification:
        errors.append(f"{period_id} classification must be {classification}")


def validate_contract_values(
    protocol: Mapping[str, Any],
    access_ledger: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for field, value in (
        ("protocol", protocol),
        ("access_ledger", access_ledger),
        ("feature_arm_registry", registry),
    ):
        if value.get("experiment_id") != EXPERIMENT_ID:
            errors.append(f"{field}.experiment_id must be {EXPERIMENT_ID}")
        if value.get("schema_version") != 1:
            errors.append(f"{field}.schema_version must be 1")

    if protocol.get("artifact_type") != "CVP_ABLATION_FREEZE_PROTOCOL":
        errors.append("protocol artifact_type changed")
    if protocol.get("registration_mode") != "APPEND_ONLY_POST_HOC_ABLATION":
        errors.append("protocol must remain append-only and post-hoc")
    if protocol.get("scientific_status") != "DESCRIPTIVE_HISTORICAL_EVIDENCE_ONLY":
        errors.append("protocol scientific status must remain descriptive historical evidence")
    _iso_timestamp(protocol.get("created_at_utc"), "protocol.created_at_utc", errors)

    development = protocol.get("development")
    if not isinstance(development, dict):
        errors.append("protocol.development must be an object")
    else:
        _validate_exact_period(development, "development", "DEVELOPMENT", errors)
        if development.get("previously_inspected") is not True:
            errors.append("development must be marked previously inspected")

    historical = protocol.get("historical_evaluation")
    if not isinstance(historical, list) or len(historical) != 2:
        errors.append("protocol.historical_evaluation must contain exactly two periods")
    else:
        indexed = {
            item.get("period_id"): item for item in historical if isinstance(item, dict)
        }
        if set(indexed) != set(PERIODS):
            errors.append("historical evaluation period IDs changed")
        for period_id in PERIODS:
            value = indexed.get(period_id)
            if value is None:
                continue
            _validate_exact_period(
                value, period_id, "CONTAMINATED_HISTORICAL_HOLDOUT", errors
            )
            if value.get("previously_inspected") is not True:
                errors.append(f"{period_id} must be marked previously inspected")
            if value.get("opened_before_ablation_freeze") is not True:
                errors.append(f"{period_id} must disclose pre-freeze outcome access")

    forbidden = protocol.get("forbidden_period")
    if not isinstance(forbidden, dict):
        errors.append("protocol.forbidden_period must be an object")
    else:
        start = _iso_date(forbidden.get("start_inclusive"), "forbidden.start", errors)
        end = _iso_date(forbidden.get("end_inclusive"), "forbidden.end", errors)
        if start != date(2026, 8, 1) or end != date(2026, 8, 31):
            errors.append("the forbidden interval must be exactly August 2026")
        if forbidden.get("access_authorized") is not False:
            errors.append("August access must remain unauthorized")
        if forbidden.get("previously_inspected") is not False:
            errors.append("August must remain recorded as uninspected")

    design = protocol.get("design")
    if not isinstance(design, dict):
        errors.append("protocol.design must be an object")
    else:
        _same_members(design.get("registered_new_arms"), ARMS, "registered_new_arms", errors)
        if design.get("trial_budget_new_arms") != 3:
            errors.append("trial budget must be three new arms")
        if design.get("seed_budget_per_arm") != 5:
            errors.append("seed budget must be five per arm")
        if design.get("required_trader_jobs") != 15:
            errors.append("required trader job count must be 15")
        if design.get("new_publisher_jobs") != 0:
            errors.append("new publisher job count must be zero")
        if design.get("required_ensemble_evaluations") != 24:
            errors.append("required ensemble evaluation count must be 24")
        selection = str(design.get("result_selection", "")).lower()
        if "every registered" not in selection or "may alter inclusion" not in selection:
            errors.append("result-independent retention language is required")

    training = protocol.get("training")
    if not isinstance(training, dict):
        errors.append("protocol.training must be an object")
    else:
        _same_members(training.get("seeds"), SEEDS, "training.seeds", errors)
        if training.get("training_manifest_required_before_evaluation") is not True:
            errors.append("complete training must precede evaluation")

    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, dict):
        errors.append("protocol.evaluation must be an object")
    else:
        _same_members(evaluation.get("periods"), PERIODS, "evaluation.periods", errors)
        _same_members(evaluation.get("scenarios"), SCENARIOS, "evaluation.scenarios", errors)
        _same_members(evaluation.get("new_arms"), ARMS, "evaluation.new_arms", errors)
    if protocol.get("registered_contrasts") != list(REGISTERED_CONTRASTS):
        errors.append("protocol registered contrasts changed")

    if not protocol.get("development_gates"):
        errors.append("technical development gates must be nonempty")
    if not protocol.get("historical_evaluation_gates"):
        errors.append("technical historical-evaluation gates must be nonempty")
    if "cannot guarantee" not in str(protocol.get("overfit_statement", "")).lower():
        errors.append("protocol must state that overfitting absence cannot be guaranteed")

    if access_ledger.get("artifact_type") != "CVP_ABLATION_ACCESS_LEDGER":
        errors.append("access ledger artifact_type changed")
    ledger_periods = access_ledger.get("periods")
    if not isinstance(ledger_periods, list):
        errors.append("access ledger periods must be a list")
    else:
        by_name = {
            item.get("name"): item for item in ledger_periods if isinstance(item, dict)
        }
        if not {"gc_training", "may_2026", "june_july_2026", "august_2026"}.issubset(by_name):
            errors.append("access ledger omits a required GC period")
        for period_id in PERIODS:
            value = by_name.get(period_id, {})
            if value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT":
                errors.append(f"access ledger misclassifies {period_id}")
            if value.get("previously_inspected") is not True:
                errors.append(f"access ledger must mark {period_id} inspected")
            forbidden_uses = set(value.get("forbidden_uses", []))
            if not {"training", "tuning", "checkpoint_selection", "fresh_oos_claim"}.issubset(forbidden_uses):
                errors.append(f"access ledger does not fully restrict {period_id}")
        august = by_name.get("august_2026", {})
        if august.get("classification") != "FRESH_OOS":
            errors.append("access ledger must classify unopened August as FRESH_OOS")
        if august.get("previously_inspected") is not False:
            errors.append("access ledger must mark August uninspected")
        if set(august.get("permitted_uses", [])):
            errors.append("access ledger must permit no August use")
        if not {"file_listing", "file_open", "conversion", "feature_generation"}.issubset(
            set(august.get("forbidden_uses", []))
        ):
            errors.append("access ledger must reject August before filesystem access")
    if "cannot guarantee" not in str(access_ledger.get("overfit_statement", "")).lower():
        errors.append("access ledger must state that overfitting absence cannot be guaranteed")

    if registry.get("artifact_type") != "CVP_ABLATION_FEATURE_ARM_REGISTRY":
        errors.append("feature-arm registry artifact_type changed")
    _same_members(registry.get("seeds"), SEEDS, "registry.seeds", errors)
    _same_members(registry.get("scenarios"), SCENARIOS, "registry.scenarios", errors)
    if registry.get("required_training_jobs") != 15:
        errors.append("registry required_training_jobs must be 15")
    if registry.get("required_ensemble_evaluations") != 24:
        errors.append("registry required_ensemble_evaluations must be 24")
    parents = registry.get("parent_conditions")
    if not isinstance(parents, dict) or set(parents) != {
        "cvp_hddqn", "restricted_representation"
    }:
        errors.append("registry parent conditions changed")
    elif any(value.get("new_training_permitted") is not False for value in parents.values()):
        errors.append("frozen parent retraining must remain forbidden")
    arms = registry.get("new_arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        errors.append("registry must contain exactly the three new ablation arms")
    else:
        for arm, value in arms.items():
            if value.get("train_new_trader") is not True:
                errors.append(f"{arm} must register one new trader per seed")
            if value.get("publisher_training_permitted") is not False:
                errors.append(f"{arm} must forbid publisher training")
            if value.get("initialization_group") != "shared_new_ablation_trader_by_seed":
                errors.append(f"{arm} initialization group changed")
        if arms["flow_profile_only"].get("publisher_source") != (
            "frozen_same_seed_rich_volume_publisher"
        ):
            errors.append("flow_profile_only publisher source changed")
        if arms["depth_ofi_only"].get("publisher_source") != (
            "frozen_same_seed_base_ohlcv_publisher"
        ):
            errors.append("depth_ofi_only publisher source changed")
        no_publisher = arms["no_publisher"]
        if no_publisher.get("publisher_source") != "NONE":
            errors.append("no_publisher must not have a publisher")
        if no_publisher.get("message_mode") != "EXACT_ZERO_FLOAT32":
            errors.append("no_publisher message must be exact float32 zeros")

    source = registry.get("source_feature_order")
    if not isinstance(source, dict):
        errors.append("source_feature_order must be an object")
    elif isinstance(arms, dict) and set(arms) == set(ARMS):
        flow = arms["flow_profile_only"].get("representation", {})
        depth = arms["depth_ofi_only"].get("representation", {})
        if set(flow.get("l2_level_keep", [])) | set(flow.get("l2_level_zero", [])) != set(
            source.get("l2_level", [])
        ) or set(flow.get("l2_level_keep", [])) & set(flow.get("l2_level_zero", [])):
            errors.append("flow_profile_only L2-level partition is not exact")
        if set(flow.get("l2_engineered_keep", [])) | set(
            flow.get("l2_engineered_zero", [])
        ) != set(source.get("l2_engineered", [])) or set(
            flow.get("l2_engineered_keep", [])
        ) & set(flow.get("l2_engineered_zero", [])):
            errors.append("flow_profile_only engineered-L2 partition is not exact")
        if set(depth.get("top_rich_keep", [])) | set(depth.get("top_rich_zero", [])) != set(
            source.get("top_rich", [])
        ) or set(depth.get("top_rich_keep", [])) & set(depth.get("top_rich_zero", [])):
            errors.append("depth_ofi_only upper flow partition is not exact")
        if set(depth.get("bottom_rich_keep", [])) | set(
            depth.get("bottom_rich_zero", [])
        ) != set(source.get("bottom_rich", [])) or set(
            depth.get("bottom_rich_keep", [])
        ) & set(depth.get("bottom_rich_zero", [])):
            errors.append("depth_ofi_only lower flow partition is not exact")

    reporting = registry.get("reporting_policy")
    if not isinstance(reporting, dict):
        errors.append("registry.reporting_policy must be an object")
    else:
        for key in (
            "include_all_registered_arms",
            "include_all_registered_periods",
            "include_all_registered_scenarios",
            "outcome_dependent_omission_forbidden",
            "technical_failure_must_remain_in_registry",
        ):
            if reporting.get(key) is not True:
                errors.append(f"reporting_policy.{key} must be true")
        if set(reporting.get("allowed_technical_failure_reasons", [])) != set(
            TECHNICAL_FAILURE_REASONS
        ):
            errors.append("allowed technical-failure reasons changed")
    return errors


def _raise_errors(errors: Sequence[str]) -> None:
    if errors:
        raise ContractValidationError("invalid CVP ablation contract:\n- " + "\n- ".join(errors))


def load_and_validate_contracts(
    contract_dir: Path,
    *,
    repo_root: Path | None = None,
    verify_repository_artifacts: bool = True,
) -> ValidatedContracts:
    contract_dir = Path(contract_dir)
    protocol_path = contract_dir / PROTOCOL_FILE
    ledger_path = contract_dir / ACCESS_LEDGER_FILE
    registry_path = contract_dir / FEATURE_ARM_REGISTRY_FILE
    protocol = _read_json(protocol_path)
    ledger = _read_json(ledger_path)
    registry = _read_json(registry_path)
    _raise_errors(validate_contract_values(protocol, ledger, registry))

    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractValidationError("protocol.artifacts must be an object")
    local_records = {
        "access_ledger": (ledger_path, ACCESS_LEDGER_FILE),
        "feature_arm_registry": (registry_path, FEATURE_ARM_REGISTRY_FILE),
    }
    for key, (path, expected_name) in local_records.items():
        record = artifacts.get(key)
        if not isinstance(record, dict) or record.get("path") != expected_name:
            raise ContractValidationError(f"protocol artifact record {key} changed")
        actual = file_sha256(path)
        if record.get("sha256") != actual:
            raise ContractValidationError(f"protocol artifact hash mismatch for {key}")

    if verify_repository_artifacts:
        root = Path(repo_root) if repo_root is not None else contract_dir.parents[3]
        for key, record in artifacts.items():
            if key in local_records:
                continue
            if not isinstance(record, dict):
                raise ContractValidationError(f"protocol artifact record {key} must be an object")
            errors: list[str] = []
            relative = _plain_relative_path(record.get("path"), f"artifacts.{key}.path", errors)
            _raise_errors(errors)
            assert relative is not None
            path = root.joinpath(*relative.parts)
            if file_sha256(path) != record.get("sha256"):
                raise ContractValidationError(f"repository artifact hash mismatch for {key}")

    return ValidatedContracts(
        protocol=protocol,
        access_ledger=ledger,
        feature_arm_registry=registry,
        protocol_canonical_sha256=canonical_json_sha256(protocol),
        protocol_file_sha256=file_sha256(protocol_path),
        access_ledger_file_sha256=file_sha256(ledger_path),
        feature_arm_registry_file_sha256=file_sha256(registry_path),
    )


def assert_registered_date(day: str | date, purpose: str, period_id: str) -> date:
    """Authorize a registered day without touching the filesystem."""

    try:
        parsed = date.fromisoformat(day) if isinstance(day, str) else day
    except ValueError as exc:
        raise PermissionError(f"invalid requested date {day!r}") from exc
    if not isinstance(parsed, date):
        raise PermissionError("requested date must be an ISO date or date object")
    if parsed >= FORBIDDEN_START:
        raise PermissionError("August 2026 and later are forbidden by this ablation protocol")
    key = (purpose, period_id)
    if key not in REGISTERED_INPUTS:
        raise PermissionError(f"unregistered ablation period request {key!r}")
    start, end = DATE_RANGES[period_id]
    if not start <= parsed <= end:
        raise PermissionError(
            f"{parsed.isoformat()} is outside registered {purpose}/{period_id}"
        )
    return parsed


def authorize_manifest_request(
    relative_path: str | Path, purpose: str, period_id: str
) -> AuthorizedInput:
    """Reject unregistered paths lexically before any filesystem operation."""

    raw = str(relative_path).replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise PermissionError("dataset manifest path must be normalized and repository-relative")
    lowered = raw.lower()
    if "2026-08" in lowered or "august" in lowered or "/08/" in lowered:
        raise PermissionError("August paths are forbidden before filesystem access")
    key = (purpose, period_id)
    expected = REGISTERED_INPUTS.get(key)
    if expected is None:
        raise PermissionError(f"unregistered ablation period request {key!r}")
    if raw != expected[0]:
        raise PermissionError("only the exact frozen dataset manifest path is authorized")
    return AuthorizedInput(purpose, period_id, expected[0], expected[1])


def load_registered_manifest(
    repo_root: Path,
    relative_path: str | Path,
    *,
    purpose: str,
    period_id: str,
) -> Mapping[str, Any]:
    """Open a dataset manifest only after the lexical fail-closed guard passes."""

    authorized = authorize_manifest_request(relative_path, purpose, period_id)
    path = Path(repo_root).joinpath(*PurePosixPath(authorized.relative_path).parts)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != authorized.sha256:
        raise ContractValidationError("registered dataset manifest hash mismatch")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractValidationError("registered dataset manifest is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractValidationError("registered dataset manifest root must be an object")
    return value


def expected_training_keys() -> set[tuple[str, int]]:
    return {(arm, seed) for arm in ARMS for seed in SEEDS}


def expected_evaluation_keys() -> set[tuple[str, str, str]]:
    return {
        (arm, period, scenario)
        for arm in ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    }


def validate_complete_training_manifest(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("artifact_type") != "CVP_ABLATION_COMPLETE_TRAINING":
        errors.append("training manifest artifact_type changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("training manifest experiment_id changed")
    if value.get("complete") is not True:
        errors.append("training manifest must be complete")
    if value.get("classification") != "DEVELOPMENT":
        errors.append("training manifest classification must be DEVELOPMENT")
    if value.get("historical_evaluation_read") is not False:
        errors.append("training manifest must attest historical_evaluation_read=false")
    if value.get("august_accessed") is not False:
        errors.append("training manifest must attest august_accessed=false")
    jobs = value.get("jobs")
    if not isinstance(jobs, list):
        errors.append("training manifest jobs must be a list")
        return errors
    keys: list[tuple[Any, Any]] = []
    initialization_by_seed: dict[Any, set[Any]] = {}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            errors.append(f"training job {index} must be an object")
            continue
        keys.append((job.get("arm"), job.get("seed")))
        if job.get("status") != "COMPLETE":
            errors.append(f"training job {index} is not COMPLETE")
        artifact = job.get("manifest")
        if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
            errors.append(f"training job {index} lacks a hash-bound manifest")
        initialization = job.get("initialization_sha256")
        if (
            not isinstance(initialization, str)
            or len(initialization) != 64
            or any(character not in "0123456789abcdef" for character in initialization)
        ):
            errors.append(f"training job {index} lacks a canonical initialization hash")
        initialization_by_seed.setdefault(job.get("seed"), set()).add(initialization)
    if len(keys) != len(set(keys)):
        errors.append("training manifest contains duplicate arm/seed jobs")
    if set(keys) != expected_training_keys():
        errors.append("training manifest must contain the exact 15 arm/seed jobs")
    for seed in SEEDS:
        if len(initialization_by_seed.get(seed, set())) != 1:
            errors.append(f"seed {seed} does not share one initialization across all arms")
    seed_hashes = {
        next(iter(initialization_by_seed[seed]))
        for seed in SEEDS
        if len(initialization_by_seed.get(seed, set())) == 1
    }
    if len(seed_hashes) != len(SEEDS):
        errors.append("different seeds must have distinct seed-specific initializations")
    return errors


def validate_complete_evaluation_manifest(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("artifact_type") != "CVP_ABLATION_COMPLETE_EVALUATION":
        errors.append("evaluation manifest artifact_type changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("evaluation manifest experiment_id changed")
    if value.get("complete") is not True:
        errors.append("evaluation manifest must be complete")
    if value.get("period_classification") != {
        "may_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "june_july_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
    }:
        errors.append("evaluation manifest period classifications changed")
    if value.get("fresh_oos") is not False:
        errors.append("evaluation manifest must attest fresh_oos=false")
    if value.get("august_accessed") is not False:
        errors.append("evaluation manifest must attest august_accessed=false")
    if value.get("result_dependent_omission") is not False:
        errors.append("evaluation manifest must attest result_dependent_omission=false")
    if value.get("registered_contrasts") != list(REGISTERED_CONTRASTS):
        errors.append("evaluation manifest registered contrasts changed")
    parent_references = value.get("parent_reference_reports")
    expected_parent_references = {
        "may_2026": {
            "path": (
                "viability/artifacts/development/"
                "v6_registered_may_descriptive_evaluation/manifest.json"
            ),
            "sha256": "b9dc8928ae24e8e1c44130dee175819cf2456ccd2c09aa7786f76ac2ba137c71",
        },
        "june_july_2026": {
            "path": "paper/sealed_extension/result/manifest.json",
            "sha256": "fb9a8ccbe8d28d64b6dbe70968329d09c103a6c5d773b42eee856a280c2418c1",
        },
    }
    if parent_references != expected_parent_references:
        errors.append("evaluation manifest parent-reference bindings changed")
    reports = value.get("reports")
    if not isinstance(reports, list):
        errors.append("evaluation manifest reports must be a list")
        return errors
    keys: list[tuple[Any, Any, Any]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, dict):
            errors.append(f"evaluation report {index} must be an object")
            continue
        keys.append((report.get("arm"), report.get("period"), report.get("scenario")))
        status = report.get("status")
        if status not in {"COMPLETE", "TECHNICAL_FAILURE"}:
            errors.append(f"evaluation report {index} has invalid status")
        if report.get("included_in_reporting") is not True:
            errors.append(f"evaluation report {index} is omitted from reporting")
        if status == "TECHNICAL_FAILURE":
            if report.get("technical_failure_reason") not in TECHNICAL_FAILURE_REASONS:
                errors.append(f"evaluation report {index} has an unregistered failure reason")
        elif report.get("technical_failure_reason") is not None:
            errors.append(f"evaluation report {index} has a spurious failure reason")
        artifact = (
            report.get("diagnostic")
            if status == "TECHNICAL_FAILURE" and report.get("diagnostic") is not None
            else report.get("report")
        )
        if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
            errors.append(f"evaluation report {index} lacks a hash-bound report")
    if len(keys) != len(set(keys)):
        errors.append("evaluation manifest contains duplicate arm/period/scenario reports")
    if set(keys) != expected_evaluation_keys():
        errors.append("evaluation manifest must contain the exact 24 registered reports")
    return errors


def validate_freeze_receipts(
    contract_dir: Path, contracts: ValidatedContracts
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    ledger_receipt = _read_json(Path(contract_dir) / ACCESS_LEDGER_RECEIPT_FILE)
    freeze_receipt = _read_json(Path(contract_dir) / FREEZE_RECEIPT_FILE)
    if ledger_receipt.get("artifact_type") != "CVP_ABLATION_ACCESS_LEDGER_RECEIPT":
        raise ContractValidationError("access-ledger receipt artifact_type changed")
    if ledger_receipt.get("experiment_id") != EXPERIMENT_ID:
        raise ContractValidationError("access-ledger receipt experiment_id changed")
    if ledger_receipt.get("access_ledger_file_sha256") != contracts.access_ledger_file_sha256:
        raise ContractValidationError("access-ledger receipt hash mismatch")
    classifications = ledger_receipt.get("period_classifications")
    if classifications != {
        "development": "DEVELOPMENT",
        "may_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "june_july_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "august_2026": "FRESH_OOS_UNOPENED_FORBIDDEN",
    }:
        raise ContractValidationError("access-ledger receipt classifications changed")
    if freeze_receipt.get("artifact_type") != "CVP_ABLATION_FREEZE_RECEIPT":
        raise ContractValidationError("freeze receipt artifact_type changed")
    if freeze_receipt.get("experiment_id") != EXPERIMENT_ID:
        raise ContractValidationError("freeze receipt experiment_id changed")
    if freeze_receipt.get("protocol_canonical_sha256") != contracts.protocol_canonical_sha256:
        raise ContractValidationError("freeze receipt canonical protocol digest mismatch")
    if freeze_receipt.get("protocol_file_sha256") != contracts.protocol_file_sha256:
        raise ContractValidationError("freeze receipt protocol file hash mismatch")
    if freeze_receipt.get("access_ledger_file_sha256") != contracts.access_ledger_file_sha256:
        raise ContractValidationError("freeze receipt access-ledger hash mismatch")
    if freeze_receipt.get("feature_arm_registry_file_sha256") != (
        contracts.feature_arm_registry_file_sha256
    ):
        raise ContractValidationError("freeze receipt feature-arm registry hash mismatch")
    expected_receipt_hash = file_sha256(Path(contract_dir) / ACCESS_LEDGER_RECEIPT_FILE)
    if freeze_receipt.get("access_ledger_receipt_file_sha256") != expected_receipt_hash:
        raise ContractValidationError("freeze receipt access-ledger receipt hash mismatch")
    if freeze_receipt.get("training_authorized_after_contract_verification") is not True:
        raise ContractValidationError("freeze receipt does not authorize verified training")
    if freeze_receipt.get("historical_evaluation_is_fresh_oos") is not False:
        raise ContractValidationError("freeze receipt misclassifies historical evaluation")
    if freeze_receipt.get("august_access_authorized") is not False:
        raise ContractValidationError("freeze receipt authorizes forbidden August access")
    return ledger_receipt, freeze_receipt


def assert_frozen(
    contract_dir: Path, *, repo_root: Path | None = None
) -> ValidatedContracts:
    contracts = load_and_validate_contracts(
        contract_dir, repo_root=repo_root, verify_repository_artifacts=True
    )
    validate_freeze_receipts(contract_dir, contracts)
    return contracts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument(
        "--phase", choices=("contracts", "freeze", "training", "evaluation"),
        default="freeze",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        contracts = load_and_validate_contracts(
            args.contract_dir, repo_root=args.repo_root, verify_repository_artifacts=True
        )
        if args.phase == "freeze":
            validate_freeze_receipts(args.contract_dir, contracts)
        elif args.phase in {"training", "evaluation"}:
            if args.manifest is None:
                raise ContractValidationError(f"--manifest is required for {args.phase}")
            value = _read_json(args.manifest)
            errors = (
                validate_complete_training_manifest(value)
                if args.phase == "training"
                else validate_complete_evaluation_manifest(value)
            )
            _raise_errors(errors)
    except ContractValidationError as exc:
        print("FAIL", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    print("PASS")
    print(f"phase={args.phase}")
    print(f"experiment_id={EXPERIMENT_ID}")
    print(f"protocol_canonical_sha256={contracts.protocol_canonical_sha256}")
    print("historical_evaluation_classification=CONTAMINATED_HISTORICAL_HOLDOUT")
    print("august_access_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
