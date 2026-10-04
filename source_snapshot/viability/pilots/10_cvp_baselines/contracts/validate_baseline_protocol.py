#!/usr/bin/env python3
"""Standard-library validator for the frozen CVP baseline-suite contracts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_FILE = "CVP_BASELINE_SUITE_PROTOCOL.json"
ACCESS_LEDGER_FILE = "CVP_BASELINE_ACCESS_LEDGER.json"
ARM_REGISTRY_FILE = "CVP_BASELINE_ARM_REGISTRY.json"
FREEZE_RECEIPT_FILE = "CVP_BASELINE_SUITE_FREEZE_RECEIPT.json"
EXPERIMENT_ID = "CVP_BASELINE_SUITE_V1"
SEEDS = [1103, 2207, 3301, 4409, 5501]
LEARNER_ARMS = [
    "flat_multiclock_ddqn",
    "recurrent_ppo",
    "lightgbm_fitted_q",
]
DETERMINISTIC_ARMS = [
    "tsmom_60m",
    "vwap_reversion_atr1",
    "always_flat",
    "session_long",
    "session_short",
]
SCENARIOS = [
    "nominal",
    "double_commission",
    "latency_20ms",
    "adverse_tick",
]


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
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, name: str, errors: list[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def validate_contract_values(
    protocol: Mapping[str, object], ledger: Mapping[str, object],
    registry: Mapping[str, object],
) -> list[str]:
    errors: list[str] = []
    for name, value in (
        ("protocol", protocol), ("access ledger", ledger), ("arm registry", registry),
    ):
        if value.get("experiment_id") != EXPERIMENT_ID:
            errors.append(f"{name} experiment_id differs")

    development = _mapping(protocol.get("development"), "development", errors)
    if (
        development.get("label") != "DEVELOPMENT"
        or development.get("start_inclusive") != "2025-01-01"
        or development.get("end_inclusive") != "2026-04-30"
        or development.get("previously_inspected") is not True
    ):
        errors.append("development classification or bounds differ")

    holdout = _mapping(protocol.get("holdout"), "holdout", errors)
    periods = holdout.get("periods")
    expected_periods = {
        "may_2026": ("2026-05-01", "2026-05-31"),
        "june_july_2026": ("2026-06-01", "2026-07-31"),
    }
    observed: dict[str, tuple[object, object]] = {}
    if isinstance(periods, list):
        for item in periods:
            if not isinstance(item, Mapping):
                errors.append("holdout period is malformed")
                continue
            period_id = str(item.get("period_id"))
            observed[period_id] = (
                item.get("start_inclusive"), item.get("end_inclusive"),
            )
            if (
                item.get("label") != "CONTAMINATED_HISTORICAL_HOLDOUT"
                or item.get("previously_inspected") is not True
                or item.get("opened_before_baseline_freeze") is not True
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
        errors.append("historical holdout status differs")

    forbidden = _mapping(protocol.get("forbidden_period"), "forbidden period", errors)
    if (
        forbidden.get("start_inclusive") != "2026-08-01"
        or forbidden.get("end_inclusive") != "2026-08-31"
        or forbidden.get("label") != "FRESH_OOS"
        or forbidden.get("previously_inspected") is not False
        or forbidden.get("access_authorized") is not False
        or "lexically" not in str(forbidden.get("rule", "")).lower()
    ):
        errors.append("August must remain unopened, unauthorized, and lexically rejected")

    design = _mapping(protocol.get("design"), "design", errors)
    if design.get("comparators") != LEARNER_ARMS + DETERMINISTIC_ARMS:
        errors.append("registered comparator arms differ")
    if design.get("primary_arm") != "flat_multiclock_ddqn":
        errors.append("registered primary comparator differs")
    if (
        design.get("trial_budget") != 8
        or design.get("required_learner_training_jobs") != 15
        or design.get("required_new_evaluations_total") != 64
    ):
        errors.append("registered trial/job budget differs")
    retention = str(design.get("result_selection", "")).lower()
    if "retain every" not in retention or "whether cvp-hddqn wins" not in retention:
        errors.append("result-independent retention rule differs")

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
    budget_text = str(training.get("flat_capacity_and_budget", ""))
    if not all(token in budget_text for token in ("36,388", "57,726", "74,735")):
        errors.append("flat same-capacity/combined-update budget differs")

    evaluation = _mapping(protocol.get("evaluation"), "evaluation", errors)
    if evaluation.get("periods") != ["may_2026", "june_july_2026"]:
        errors.append("evaluation periods differ")
    if evaluation.get("scenarios") != SCENARIOS:
        errors.append("evaluation scenarios differ")
    if "descriptive" not in str(evaluation.get("inferential_limit", "")).lower():
        errors.append("historical inference limit is absent")
    if "flat_multiclock_ddqn" not in str(protocol.get("primary_metric", "")):
        errors.append("registered primary metric differs")
    amendments = protocol.get("preproduction_amendments")
    if not isinstance(amendments, list) or len(amendments) != 1:
        errors.append("preproduction amendment registry differs")
    else:
        amendment = amendments[0]
        if not isinstance(amendment, Mapping) or (
            amendment.get("amendment_id")
            != "CVP_BASELINE_SUITE_V1_AMENDMENT_001"
            or amendment.get("prior_protocol_file_sha256")
            != "139347ccd746e8b08ba1414bcc6173cd435d0acf46839ab0e4f520999a8361ab"
            or amendment.get("prior_protocol_canonical_sha256")
            != "eb01d599cc67c68278926b73af8ea5372fa9906c81441e9c7e3bfb7fe2e76fd8"
            or amendment.get("production_training_started_before_amendment") is not False
            or amendment.get("new_baseline_historical_evaluation_started_before_amendment") is not False
            or amendment.get("result_based_change") is not False
        ):
            errors.append("preproduction primary-comparator amendment differs")

    development_result = _mapping(
        protocol.get("development_result"), "development result", errors,
    )
    if development_result.get("verdict") != "PENDING":
        errors.append("development result must remain PENDING at freeze")
    if not isinstance(protocol.get("development_gates"), Mapping) or not isinstance(
        protocol.get("holdout_gates"), Mapping
    ):
        errors.append("development and holdout gates must be nonempty objects")

    if registry.get("seeds") != SEEDS:
        errors.append("arm registry seeds differ")
    if registry.get("learner_arms") != LEARNER_ARMS:
        errors.append("learner arm registry differs")
    if registry.get("deterministic_arms") != DETERMINISTIC_ARMS:
        errors.append("deterministic arm registry differs")
    arms = _mapping(registry.get("arms"), "registered arms", errors)
    if set(arms) != set(LEARNER_ARMS + DETERMINISTIC_ARMS):
        errors.append("arm definition set differs")
    flat = _mapping(arms.get("flat_multiclock_ddqn"), "flat DDQN arm", errors)
    architecture = _mapping(flat.get("architecture"), "flat architecture", errors)
    flat_training = _mapping(flat.get("training"), "flat training", errors)
    if (
        architecture.get("total_parameters") != 36_388
        or architecture.get("all_parameters_jointly_optimized") is not True
        or flat.get("message_interface", "missing") is not None
        or flat.get("publication_or_expiry_interface", "missing") is not None
    ):
        errors.append("flat model capacity/message contract differs")
    if (
        flat_training.get("environment_interactions_per_seed") != 57_726
        or flat_training.get("ordinary_optimizer_updates_per_seed") != 57_215
        or flat_training.get("supplemental_replay_updates_per_seed") != 17_520
        or flat_training.get("optimizer_updates_per_seed") != 74_735
        or flat_training.get("supplemental_replay_updates_by_pass")
        != [5_670, 5_925, 5_925]
    ):
        errors.append("flat exact interaction/update budget differs")
    if "false" not in str(flat.get("top_unavailable", "")).lower():
        errors.append("flat unavailable-top mask contract is absent")

    ppo = _mapping(arms.get("recurrent_ppo"), "PPO arm", errors)
    ppo_training = _mapping(ppo.get("training"), "PPO training", errors)
    if (
        ppo_training.get("environment_interactions_per_seed") != 57_726
        or ppo_training.get("optimizer_updates_per_seed") != 912
    ):
        errors.append("PPO registered exposure/update budget differs")
    fitted_q = _mapping(arms.get("lightgbm_fitted_q"), "fitted-Q arm", errors)
    fitted_training = _mapping(fitted_q.get("training"), "fitted-Q training", errors)
    if fitted_training.get("fqi_iterations") != 10 or fitted_training.get(
        "n_estimators"
    ) != 300:
        errors.append("fitted-Q registered training contract differs")

    periods_value = ledger.get("periods")
    classifications: dict[str, object] = {}
    if isinstance(periods_value, list):
        for item in periods_value:
            if isinstance(item, Mapping):
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
        errors.append("access-ledger period classifications differ")

    if protocol.get("scientific_status") != "DESCRIPTIVE_HISTORICAL_EVIDENCE_ONLY":
        errors.append("scientific status differs")
    if protocol.get("strongest_action_justified") != (
        "PAPER_DESCRIPTIVE_ONLY_AND_COLLECT_FRESH_PROSPECTIVE_DATA"
    ):
        errors.append("strongest justified action differs")
    if "cannot guarantee" not in str(protocol.get("overfit_statement", "")).lower():
        errors.append("mandatory overfit limitation is absent")
    return errors


@dataclass(frozen=True)
class FrozenContracts:
    protocol: dict[str, object]
    ledger: dict[str, object]
    registry: dict[str, object]
    protocol_file_sha256: str
    protocol_canonical_sha256: str
    ledger_file_sha256: str
    registry_file_sha256: str


def load_and_validate_contracts(
    contract_dir: Path, *, repo_root: Path,
    verify_repository_artifacts: bool = True,
) -> FrozenContracts:
    directory = Path(contract_dir).resolve()
    root = Path(repo_root).resolve()
    protocol_path = directory / PROTOCOL_FILE
    ledger_path = directory / ACCESS_LEDGER_FILE
    registry_path = directory / ARM_REGISTRY_FILE
    protocol = _read_object(protocol_path)
    ledger = _read_object(ledger_path)
    registry = _read_object(registry_path)
    errors = validate_contract_values(protocol, ledger, registry)
    if errors:
        raise ValueError("; ".join(errors))
    result = FrozenContracts(
        protocol=protocol,
        ledger=ledger,
        registry=registry,
        protocol_file_sha256=_file_sha256(protocol_path),
        protocol_canonical_sha256=_canonical_sha256(protocol),
        ledger_file_sha256=_file_sha256(ledger_path),
        registry_file_sha256=_file_sha256(registry_path),
    )
    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("protocol artifacts must be an object")
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"artifact binding is malformed: {name}")
        value = Path(str(record.get("path", "")))
        if value.is_absolute() or ".." in value.parts:
            raise ValueError(f"artifact binding escapes its scope: {name}")
        anchor = directory if name in {"access_ledger", "arm_registry"} else root
        path = (anchor / value).resolve()
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
    return result


def validate_freeze_receipt(
    contract_dir: Path, contracts: FrozenContracts,
) -> dict[str, object]:
    receipt = _read_object(Path(contract_dir) / FREEZE_RECEIPT_FILE)
    if (
        receipt.get("experiment_id") != EXPERIMENT_ID
        or receipt.get("protocol_file_sha256") != contracts.protocol_file_sha256
        or receipt.get("protocol_canonical_sha256")
        != contracts.protocol_canonical_sha256
        or receipt.get("access_ledger_file_sha256")
        != contracts.ledger_file_sha256
        or receipt.get("arm_registry_file_sha256")
        != contracts.registry_file_sha256
        or receipt.get("frozen_before_production_training") is not True
        or receipt.get("frozen_before_new_baseline_outcome_access") is not True
        or receipt.get("frozen_before_predecessor_outcome_access") is not False
        or receipt.get("august_access_authorized") is not False
        or receipt.get("implementation_binding_required_before_production") is not True
        or receipt.get("development_verdict") != "PENDING"
        or receipt.get("strongest_action_justified")
        != "PAPER_DESCRIPTIVE_ONLY_AND_COLLECT_FRESH_PROSPECTIVE_DATA"
    ):
        raise ValueError("baseline-suite freeze receipt differs")
    amendments = receipt.get("preproduction_amendments")
    if not isinstance(amendments, list) or amendments != [{
        "amendment_id": "CVP_BASELINE_SUITE_V1_AMENDMENT_001",
        "prior_protocol_file_sha256": "139347ccd746e8b08ba1414bcc6173cd435d0acf46839ab0e4f520999a8361ab",
        "prior_protocol_canonical_sha256": "eb01d599cc67c68278926b73af8ea5372fa9906c81441e9c7e3bfb7fe2e76fd8",
        "primary_arm_before": "recurrent_ppo",
        "primary_arm_after": "flat_multiclock_ddqn",
        "production_training_started_before_amendment": False,
        "new_baseline_historical_evaluation_started_before_amendment": False,
        "result_based_change": False,
    }]:
        raise ValueError("baseline-suite freeze amendment receipt differs")
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
        contract_dir, repo_root=repo_root, verify_repository_artifacts=True,
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
    print("development_classification=DEVELOPMENT")
    print("historical_evaluation_classification=CONTAMINATED_HISTORICAL_HOLDOUT")
    print("august_access_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
