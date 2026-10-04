#!/usr/bin/env python3
"""Standard-library validator for the frozen Pilot 11 reduced-suite contracts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


PROTOCOL_FILE = "CVP_BASELINE_REDUCED_SUITE_PROTOCOL.json"
ACCESS_LEDGER_FILE = "CVP_BASELINE_REDUCED_ACCESS_LEDGER.json"
ARM_REGISTRY_FILE = "CVP_BASELINE_REDUCED_ARM_REGISTRY.json"
CANCELLATION_LEDGER_FILE = "CVP_BASELINE_V1_OWNER_CANCELLATION_LEDGER.json"
FREEZE_RECEIPT_FILE = "CVP_BASELINE_REDUCED_SUITE_FREEZE_RECEIPT.json"
EXPERIMENT_ID = "CVP_BASELINE_REDUCED_SUITE_V2"
SOURCE_EXPERIMENT_ID = "CVP_BASELINE_SUITE_V1"
SEEDS = [1103, 2207, 3301, 4409, 5501]
LEARNER_ARMS = ["recurrent_ppo", "lightgbm_fitted_q"]
REFERENCE_ARMS = [
    "cvp_hddqn",
    "no_publisher_single_agent_ddqn",
    "shavandi_khedmati_reconstruction",
]
PERIODS = ["may_2026", "june_july_2026"]
SCENARIOS = [
    "nominal",
    "double_commission",
    "latency_20ms",
    "adverse_tick",
]
SOURCE_PROTOCOL_CANONICAL_SHA256 = (
    "bc8c0c159e30d996fe542d58c55b4090f6cc70398a28c3773470924a571485b9"
)
SOURCE_IMPLEMENTATION_SHA256 = (
    "31ce433cf5c5e9a401ee500ff7a990caee91f42267083552095e6bd125a182a0"
)
SOURCE_ENVIRONMENT_SHA256 = (
    "cf9f2d1538beeea9423446e54d97e6ce3bbb69ee20aa028d269c207f676089fe"
)


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"contract JSON is not an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, name: str, errors: list[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _safe_repository_path(root: Path, value: object, name: str) -> Path:
    raw = str(value).replace("\\", "/")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or not raw
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{name} is not a portable repository path")
    result = (root / Path(*relative.parts)).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the repository") from error
    return result


def validate_contract_values(
    protocol: Mapping[str, object],
    ledger: Mapping[str, object],
    registry: Mapping[str, object],
    cancellation: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    for name, value in (
        ("protocol", protocol),
        ("access ledger", ledger),
        ("arm registry", registry),
    ):
        if value.get("experiment_id") != EXPERIMENT_ID:
            errors.append(f"{name} experiment_id differs")

    if protocol.get("parent_version") != SOURCE_EXPERIMENT_ID:
        errors.append("source experiment differs")
    if protocol.get("scientific_status") != "DESCRIPTIVE_HISTORICAL_EVIDENCE_ONLY":
        errors.append("scientific status differs")

    development = _mapping(protocol.get("development"), "development", errors)
    if (
        development.get("label") != "DEVELOPMENT"
        or development.get("start_inclusive") != "2025-01-01"
        or development.get("end_inclusive") != "2026-04-30"
        or development.get("previously_inspected") is not True
        or "No Pilot 11 fitting" not in str(development.get("dataset_policy", ""))
    ):
        errors.append("development classification, bounds, or no-refit rule differs")

    holdout = _mapping(protocol.get("holdout"), "holdout", errors)
    expected_periods = {
        "may_2026": ("2026-05-01", "2026-05-31"),
        "june_july_2026": ("2026-06-01", "2026-07-31"),
    }
    observed: dict[str, tuple[object, object]] = {}
    period_values = holdout.get("periods")
    if isinstance(period_values, list):
        for item in period_values:
            if not isinstance(item, Mapping):
                errors.append("holdout period is malformed")
                continue
            period_id = str(item.get("period_id"))
            observed[period_id] = (
                item.get("start_inclusive"),
                item.get("end_inclusive"),
            )
            if (
                item.get("label") != "CONTAMINATED_HISTORICAL_HOLDOUT"
                or item.get("previously_inspected") is not True
                or item.get("opened_before_successor_freeze") is not True
            ):
                errors.append(f"{period_id} contamination classification differs")
    else:
        errors.append("holdout periods must be a list")
    if observed != expected_periods:
        errors.append("historical evaluation bounds differ")
    if (
        holdout.get("label") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or protocol.get("holdout_access_authorized") is not True
        or not protocol.get("holdout_opened_at_utc")
    ):
        errors.append("contaminated historical holdout status differs")

    forbidden = _mapping(protocol.get("forbidden_period"), "forbidden period", errors)
    if (
        forbidden.get("start_inclusive") != "2026-08-01"
        or forbidden.get("end_inclusive") != "2026-08-31"
        or forbidden.get("label") != "FRESH_OOS"
        or forbidden.get("previously_inspected") is not False
        or forbidden.get("access_authorized") is not False
        or "lexically" not in str(forbidden.get("rule", "")).lower()
        or "no unlock" not in str(forbidden.get("rule", "")).lower()
    ):
        errors.append("August must remain unopened, unauthorized, and lexically rejected")

    design = _mapping(protocol.get("design"), "design", errors)
    if design.get("comparators") != LEARNER_ARMS:
        errors.append("reduced comparator registry is not exactly PPO and fitted Q")
    if design.get("primary_arm") != "recurrent_ppo":
        errors.append("reduced primary comparator differs")
    if (
        design.get("trial_budget") != 2
        or design.get("learner_seed_budget_per_arm") != 5
        or design.get("required_learner_training_jobs") != 10
        or design.get("required_new_evaluations_total") != 16
        or design.get("required_frozen_reference_rows") != 20
    ):
        errors.append("reduced trial, training, evaluation, or reference budget differs")
    retention = str(design.get("result_selection", "")).lower()
    if "retain every" not in retention or "whether cvp-hddqn wins" not in retention:
        errors.append("result-independent retention rule differs")
    excluded = _mapping(design.get("excluded_arms"), "excluded arms", errors)
    if (
        "CANCELLED_BY_OWNER" not in str(excluded.get("flat_multiclock_ddqn", ""))
        or "EXCLUDED_BY_OWNER_SCOPE" not in str(excluded.get("deterministic_controls", ""))
    ):
        errors.append("owner scope exclusions differ")

    execution = _mapping(protocol.get("execution"), "execution", errors)
    if execution.get("scenarios") != SCENARIOS:
        errors.append("execution scenarios differ")
    if execution.get("maximum_lot") != 1 or execution.get("actions") != [
        "SHORT", "FLAT", "LONG",
    ]:
        errors.append("execution action/size contract differs")

    training = _mapping(protocol.get("training"), "training", errors)
    if training.get("seeds") != SEEDS:
        errors.append("training seeds differ")
    if (
        training.get("source_experiment") != SOURCE_EXPERIMENT_ID
        or training.get("frozen_before_adopted_training") is not False
        or training.get("source_v1_frozen_before_original_training") is not True
        or training.get("frozen_before_new_reduced_suite_evaluation") is not True
        or training.get("adopted_training_receipt_required_before_evaluation") is not True
    ):
        errors.append("training adoption timing or prerequisite differs")
    if training.get("adopted_training_receipt") != (
        "viability/artifacts/development/cvp_baseline_reduced_suite_v2/"
        "adopted_training_receipt.json"
    ):
        errors.append("adopted training receipt path differs")

    evaluation = _mapping(protocol.get("evaluation"), "evaluation", errors)
    if (
        evaluation.get("periods") != PERIODS
        or evaluation.get("scenarios") != SCENARIOS
        or evaluation.get("new_baseline_arms") != LEARNER_ARMS
        or evaluation.get("new_cells") != 16
    ):
        errors.append("evaluation arm/period/scenario grid differs")
    if "descriptive" not in str(evaluation.get("inferential_limit", "")).lower():
        errors.append("historical inference limit is absent")
    if "recurrent PPO" not in str(protocol.get("primary_metric", "")):
        errors.append("primary descriptive contrast differs")

    development_result = _mapping(
        protocol.get("development_result"), "development result", errors,
    )
    if (
        development_result.get("verdict") != "PASS"
        or development_result.get("failed_gates") != []
        or not development_result.get("evaluated_at_utc")
        or "no new historical evaluation outcome" not in str(
            development_result.get("basis", "")
        )
    ):
        errors.append("artifact-only development result differs")
    if not isinstance(protocol.get("development_gates"), Mapping) or not isinstance(
        protocol.get("holdout_gates"), Mapping
    ):
        errors.append("development and holdout gates must be nonempty objects")

    output = _mapping(protocol.get("output_contract"), "output contract", errors)
    if output != {
        "artifact_root": "viability/artifacts/development/cvp_baseline_reduced_suite_v2",
        "adopted_training_receipt": "adopted_training_receipt.json",
        "adopted_training_artifact_type": "CVP_BASELINE_REDUCED_ADOPTED_TRAINING_RECEIPT",
        "evaluation_manifest": "evaluation/manifest.json",
        "evaluation_artifact_type": "CVP_BASELINE_REDUCED_COMPLETE_EVALUATION_MANIFEST",
        "report_root": "viability/reports/cvp_baseline_reduced_suite_v2",
        "paper_write_authorized": False,
    }:
        errors.append("output contract differs")

    if registry.get("seeds") != SEEDS:
        errors.append("arm registry seeds differ")
    if registry.get("learner_arms") != LEARNER_ARMS:
        errors.append("learner registry must contain exactly two arms")
    if registry.get("deterministic_arms") != []:
        errors.append("deterministic controls must not be registered in Pilot 11")
    if registry.get("reference_arms_not_retrained") != REFERENCE_ARMS:
        errors.append("frozen reference registry differs")
    arms = _mapping(registry.get("arms"), "registered arms", errors)
    if list(arms) != LEARNER_ARMS:
        errors.append("arm definition set or order differs")
    ppo = _mapping(arms.get("recurrent_ppo"), "PPO arm", errors)
    ppo_training = _mapping(ppo.get("training"), "PPO training", errors)
    if (
        ppo_training.get("environment_interactions_per_seed") != 57_726
        or ppo_training.get("optimizer_updates_per_seed") != 912
    ):
        errors.append("PPO training budget differs")
    fitted = _mapping(arms.get("lightgbm_fitted_q"), "fitted-Q arm", errors)
    fitted_training = _mapping(fitted.get("training"), "fitted-Q training", errors)
    if (
        fitted_training.get("fqi_iterations") != 10
        or fitted_training.get("n_estimators") != 300
    ):
        errors.append("fitted-Q training contract differs")
    source_binding = _mapping(
        registry.get("source_training_binding"), "source training binding", errors,
    )
    if (
        source_binding.get("experiment_id") != SOURCE_EXPERIMENT_ID
        or source_binding.get("protocol_canonical_sha256")
        != SOURCE_PROTOCOL_CANONICAL_SHA256
        or source_binding.get("implementation_aggregate_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or source_binding.get("environment_aggregate_sha256")
        != SOURCE_ENVIRONMENT_SHA256
    ):
        errors.append("source training lineage differs")

    member_values = registry.get("adopted_training_members")
    if not isinstance(member_values, list) or len(member_values) != 10:
        errors.append("adopted training registry must contain ten members")
    else:
        expected = [(arm, seed) for arm in LEARNER_ARMS for seed in SEEDS]
        observed_members = [
            (str(item.get("arm")), int(item.get("seed", -1)))
            for item in member_values if isinstance(item, Mapping)
        ]
        if observed_members != expected or len(observed_members) != len(member_values):
            errors.append("adopted member arm/seed order differs")
        for item in member_values:
            if not isinstance(item, Mapping) or item.get("status") != "COMPLETE":
                errors.append("adopted member is malformed or incomplete")

    period_records = ledger.get("periods")
    classifications: dict[str, object] = {}
    if isinstance(period_records, list):
        for item in period_records:
            if not isinstance(item, Mapping):
                errors.append("access-ledger period is malformed")
                continue
            classifications[str(item.get("name"))] = item.get("classification")
            if item.get("name") == "august_2026" and (
                item.get("previously_inspected") is not False
                or item.get("permitted_uses") != []
                or "path_resolution" not in item.get("forbidden_uses", [])
            ):
                errors.append("August access-ledger rule differs")
    else:
        errors.append("access ledger periods must be a list")
    if classifications != {
        "xau_price_encoder_pretraining": "DEVELOPMENT",
        "gc_training": "DEVELOPMENT",
        "may_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "june_july_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "august_2026": "FRESH_OOS",
    }:
        errors.append("access-ledger classifications differ")

    if (
        cancellation.get("source_experiment_id") != SOURCE_EXPERIMENT_ID
        or cancellation.get("successor_experiment_id") != EXPERIMENT_ID
        or cancellation.get("decision") != "OWNER_SCOPE_REDUCTION"
        or cancellation.get("cancelled_arm") != "flat_multiclock_ddqn"
        or cancellation.get("status") != "CANCELLED_BY_OWNER"
        or cancellation.get("technical_failure") is not False
        or cancellation.get("evaluation_eligible") is not False
        or cancellation.get("report_as_baseline_result") is not False
    ):
        errors.append("owner cancellation identity/status differs")
    timing = _mapping(cancellation.get("decision_timing"), "cancellation timing", errors)
    if (
        timing.get("before_any_new_v1_or_v2_historical_evaluation") is not True
        or timing.get("economic_outcomes_accessed_for_cancelled_arm") is not False
        or timing.get("result_based_cancellation") is not False
    ):
        errors.append("cancellation timing/result-independence differs")
    cancelled_seeds = cancellation.get("seed_records")
    if not isinstance(cancelled_seeds, list) or len(cancelled_seeds) != 5:
        errors.append("cancellation ledger must retain five seed records")
    else:
        observed_seeds = [
            int(item.get("seed", -1))
            for item in cancelled_seeds if isinstance(item, Mapping)
        ]
        if observed_seeds != SEEDS or len(observed_seeds) != len(cancelled_seeds):
            errors.append("cancelled seed order differs")
        for item in cancelled_seeds:
            if not isinstance(item, Mapping) or (
                item.get("status") != "CANCELLED_BY_OWNER"
                or item.get("evaluation_eligible") is not False
                or item.get("passes_required") != 3
            ):
                errors.append("cancelled seed status differs")
                continue
            seed = int(item.get("seed", -1))
            if seed == 2207:
                if (
                    item.get("training_state") != "NOT_STARTED"
                    or item.get("passes_complete") != 0
                    or item.get("progress_path") is not None
                    or item.get("progress_sha256") is not None
                ):
                    errors.append("cancelled unstarted seed 2207 differs")
            elif (
                item.get("training_state") != "PARTIAL"
                or item.get("passes_complete") != 1
                or not item.get("progress_path")
                or not item.get("progress_sha256")
            ):
                errors.append(f"cancelled partial seed {seed} differs")

    if protocol.get("strongest_action_justified") != (
        "PAPER_DESCRIPTIVE_ONLY_AND_COLLECT_FRESH_PROSPECTIVE_DATA"
    ):
        errors.append("strongest justified action differs")
    if "cannot guarantee" not in str(protocol.get("overfit_statement", "")).lower():
        errors.append("mandatory overfit limitation is absent")
    return errors


def _verify_member_artifacts(
    registry: Mapping[str, object],
    *,
    repo_root: Path,
) -> None:
    records = registry.get("adopted_training_members")
    if not isinstance(records, list):
        raise ValueError("adopted training member registry is malformed")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("adopted training member record is malformed")
        arm = str(record.get("arm"))
        seed = int(record.get("seed", -1))
        manifest_path = _safe_repository_path(
            repo_root, record.get("manifest_path"), "member manifest",
        )
        if (
            not manifest_path.is_file()
            or _file_sha256(manifest_path) != record.get("manifest_sha256")
        ):
            raise ValueError(f"adopted member manifest differs: {arm}/{seed}")
        manifest = _read_object(manifest_path)
        implementation = manifest.get("implementation")
        if not isinstance(implementation, Mapping):
            raise ValueError(f"source implementation missing: {arm}/{seed}")
        environment = implementation.get("environment")
        protocol = manifest.get("protocol")
        if (
            manifest.get("artifact_type") != "CVP_BASELINE_LEARNER_TRAINING"
            or manifest.get("experiment_id") != SOURCE_EXPERIMENT_ID
            or manifest.get("arm") != arm
            or manifest.get("seed") != seed
            or manifest.get("status") != "COMPLETE"
            or manifest.get("holdout_read") is not False
            or manifest.get("evaluation_accessed") is not False
            or not isinstance(protocol, Mapping)
            or protocol.get("canonical_sha256") != SOURCE_PROTOCOL_CANONICAL_SHA256
            or implementation.get("aggregate_sha256") != SOURCE_IMPLEMENTATION_SHA256
            or not isinstance(environment, Mapping)
            or environment.get("aggregate_sha256") != SOURCE_ENVIRONMENT_SHA256
        ):
            raise ValueError(f"adopted member lineage differs: {arm}/{seed}")
        for label in ("model", "scaler"):
            member_record = manifest.get(label)
            if not isinstance(member_record, Mapping):
                raise ValueError(f"source {label} binding missing: {arm}/{seed}")
            expected_path = str(record.get(f"{label}_path"))
            expected_sha = str(record.get(f"{label}_sha256"))
            if (
                member_record.get("path") != expected_path
                or member_record.get("sha256") != expected_sha
            ):
                raise ValueError(f"registered {label} binding differs: {arm}/{seed}")
            path = _safe_repository_path(
                manifest_path.parent,
                expected_path,
                f"{label} member path",
            )
            if not path.is_file() or _file_sha256(path) != expected_sha:
                raise ValueError(f"adopted {label} file differs: {arm}/{seed}")


def _verify_cancelled_partials(
    cancellation: Mapping[str, object],
    *,
    repo_root: Path,
) -> None:
    records = cancellation.get("seed_records")
    if not isinstance(records, list):
        raise ValueError("cancellation records are malformed")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("cancellation record is malformed")
        seed = int(record.get("seed", -1))
        relative = record.get("progress_path")
        if seed == 2207:
            unexpected = repo_root / (
                "viability/artifacts/development/cvp_baseline_suite_v1/"
                "flat_multiclock_ddqn/.seed_2207.part/progress.json"
            )
            if unexpected.exists():
                raise ValueError("seed 2207 no longer matches NOT_STARTED cancellation")
            continue
        path = _safe_repository_path(repo_root, relative, "cancelled progress path")
        if not path.is_file() or _file_sha256(path) != record.get("progress_sha256"):
            raise ValueError(f"cancelled partial changed after owner stop: seed {seed}")
        progress = _read_object(path)
        if (
            progress.get("experiment_id") != SOURCE_EXPERIMENT_ID
            or progress.get("arm") != "flat_multiclock_ddqn"
            or progress.get("seed") != seed
            or progress.get("passes_complete") != 1
            or progress.get("passes_required") != 3
            or progress.get("holdout_read") is not False
            or progress.get("evaluation_accessed") is not False
        ):
            raise ValueError(f"cancelled partial state differs: seed {seed}")


@dataclass(frozen=True)
class FrozenContracts:
    protocol: dict[str, object]
    ledger: dict[str, object]
    registry: dict[str, object]
    cancellation: dict[str, object]
    protocol_file_sha256: str
    protocol_canonical_sha256: str
    ledger_file_sha256: str
    registry_file_sha256: str
    cancellation_file_sha256: str


def load_and_validate_contracts(
    contract_dir: Path,
    *,
    repo_root: Path,
    verify_repository_artifacts: bool = True,
) -> FrozenContracts:
    directory = Path(contract_dir).resolve()
    root = Path(repo_root).resolve()
    protocol_path = directory / PROTOCOL_FILE
    ledger_path = directory / ACCESS_LEDGER_FILE
    registry_path = directory / ARM_REGISTRY_FILE
    cancellation_path = directory / CANCELLATION_LEDGER_FILE
    protocol = _read_object(protocol_path)
    ledger = _read_object(ledger_path)
    registry = _read_object(registry_path)
    cancellation = _read_object(cancellation_path)
    errors = validate_contract_values(protocol, ledger, registry, cancellation)
    if errors:
        raise ValueError("; ".join(errors))
    result = FrozenContracts(
        protocol=protocol,
        ledger=ledger,
        registry=registry,
        cancellation=cancellation,
        protocol_file_sha256=_file_sha256(protocol_path),
        protocol_canonical_sha256=_canonical_sha256(protocol),
        ledger_file_sha256=_file_sha256(ledger_path),
        registry_file_sha256=_file_sha256(registry_path),
        cancellation_file_sha256=_file_sha256(cancellation_path),
    )
    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("protocol artifacts must be an object")
    local_names = {"access_ledger", "arm_registry", "owner_cancellation_ledger"}
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"artifact binding is malformed: {name}")
        anchor = directory if name in local_names else root
        path = _safe_repository_path(anchor, record.get("path"), name)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"artifact binding escapes repository: {name}") from error
        if verify_repository_artifacts and (
            not path.is_file() or _file_sha256(path) != record.get("sha256")
        ):
            raise ValueError(f"artifact binding differs: {name}")
    if artifacts["access_ledger"].get("sha256") != result.ledger_file_sha256:
        raise ValueError("protocol access-ledger digest differs")
    if artifacts["arm_registry"].get("sha256") != result.registry_file_sha256:
        raise ValueError("protocol arm-registry digest differs")
    if (
        artifacts["owner_cancellation_ledger"].get("sha256")
        != result.cancellation_file_sha256
    ):
        raise ValueError("protocol cancellation-ledger digest differs")
    if verify_repository_artifacts:
        _verify_member_artifacts(registry, repo_root=root)
        _verify_cancelled_partials(cancellation, repo_root=root)
    return result


def validate_freeze_receipt(
    contract_dir: Path,
    contracts: FrozenContracts,
) -> dict[str, object]:
    receipt = _read_object(Path(contract_dir) / FREEZE_RECEIPT_FILE)
    if (
        receipt.get("artifact_type")
        != "CVP_BASELINE_REDUCED_SUITE_FREEZE_RECEIPT"
        or receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("source_experiment_id") != SOURCE_EXPERIMENT_ID
        or receipt.get("protocol_file_sha256") != contracts.protocol_file_sha256
        or receipt.get("protocol_canonical_sha256")
        != contracts.protocol_canonical_sha256
        or receipt.get("access_ledger_file_sha256")
        != contracts.ledger_file_sha256
        or receipt.get("arm_registry_file_sha256")
        != contracts.registry_file_sha256
        or receipt.get("owner_cancellation_ledger_file_sha256")
        != contracts.cancellation_file_sha256
        or receipt.get("frozen_before_adopted_training") is not False
        or receipt.get("source_v1_frozen_before_original_training") is not True
        or receipt.get("frozen_before_new_reduced_suite_evaluation") is not True
        or receipt.get("new_reduced_suite_evaluation_started") is not False
        or receipt.get("result_based_scope_reduction") is not False
        or receipt.get("august_access_authorized") is not False
        or receipt.get("development_verdict") != "PASS"
        or receipt.get("registered_learner_jobs") != 10
        or receipt.get("registered_new_evaluations") != 16
        or receipt.get("registered_frozen_reference_rows") != 20
        or receipt.get("flat_ddqn_status") != "CANCELLED_BY_OWNER"
        or receipt.get("flat_ddqn_evaluation_eligible") is not False
        or receipt.get("strongest_action_justified")
        != "PAPER_DESCRIPTIVE_ONLY_AND_COLLECT_FRESH_PROSPECTIVE_DATA"
    ):
        raise ValueError("reduced-suite freeze receipt differs")
    if receipt.get("period_classifications") != {
        "development": "DEVELOPMENT",
        "may_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "june_july_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "august_2026": "FRESH_OOS_UNOPENED_FORBIDDEN",
    }:
        raise ValueError("freeze-receipt period classifications differ")
    return receipt


def assert_frozen(contract_dir: Path, *, repo_root: Path) -> FrozenContracts:
    contracts = load_and_validate_contracts(
        contract_dir,
        repo_root=repo_root,
        verify_repository_artifacts=True,
    )
    validate_freeze_receipt(contract_dir, contracts)
    return contracts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract_dir", type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--phase", choices=("freeze", "report"), default="freeze")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contracts = assert_frozen(args.contract_dir, repo_root=args.repo_root)
    print(f"PASS phase={args.phase}")
    print(f"experiment_id={EXPERIMENT_ID}")
    print(f"protocol_canonical_sha256={contracts.protocol_canonical_sha256}")
    print("registered_new_baselines=recurrent_ppo,lightgbm_fitted_q")
    print("registered_new_evaluation_cells=16")
    print("flat_multiclock_ddqn=CANCELLED_BY_OWNER_INELIGIBLE")
    print("development_classification=DEVELOPMENT")
    print("historical_evaluation_classification=CONTAMINATED_HISTORICAL_HOLDOUT")
    print("august_access_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
