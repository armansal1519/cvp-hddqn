from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


PILOT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PILOT_ROOT / "contracts"
REPO_ROOT = PILOT_ROOT.parents[2]


def _load_validator():
    path = CONTRACT_DIR / "validate_ablation_protocol.py"
    spec = importlib.util.spec_from_file_location("cvp_ablation_contract_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _json(name: str) -> dict[str, object]:
    value = json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _training_manifest() -> dict[str, object]:
    jobs: list[dict[str, object]] = []
    for arm in validator.ARMS:
        for seed in validator.SEEDS:
            initialization = hashlib.sha256(f"initialization/{seed}".encode()).hexdigest()
            jobs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "status": "COMPLETE",
                    "initialization_sha256": initialization,
                    "manifest": {
                        "path": f"training/{arm}/seed_{seed}/manifest.json",
                        "sha256": hashlib.sha256(f"{arm}/{seed}".encode()).hexdigest(),
                    },
                }
            )
    return {
        "artifact_type": "CVP_ABLATION_COMPLETE_TRAINING",
        "experiment_id": validator.EXPERIMENT_ID,
        "complete": True,
        "classification": "DEVELOPMENT",
        "historical_evaluation_read": False,
        "august_accessed": False,
        "jobs": jobs,
    }


def _evaluation_manifest(*, pnl_sign: int = 1) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for arm, period, scenario in sorted(validator.expected_evaluation_keys()):
        reports.append(
            {
                "arm": arm,
                "period": period,
                "scenario": scenario,
                "status": "COMPLETE",
                "included_in_reporting": True,
                "technical_failure_reason": None,
                "net_pnl_usd": pnl_sign * (len(reports) + 1),
                "report": {
                    "path": f"evaluation/{period}/{scenario}/{arm}/report.json",
                    "sha256": hashlib.sha256(
                        f"{arm}/{period}/{scenario}".encode()
                    ).hexdigest(),
                },
            }
        )
    return {
        "artifact_type": "CVP_ABLATION_COMPLETE_EVALUATION",
        "experiment_id": validator.EXPERIMENT_ID,
        "complete": True,
        "period_classification": {
            "may_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
            "june_july_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        },
        "fresh_oos": False,
        "august_accessed": False,
        "result_dependent_omission": False,
        "registered_contrasts": list(validator.REGISTERED_CONTRASTS),
        "parent_reference_reports": {
            "may_2026": {
                "path": (
                    "viability/artifacts/development/"
                    "v6_registered_may_descriptive_evaluation/manifest.json"
                ),
                "sha256": (
                    "b9dc8928ae24e8e1c44130dee175819cf2456ccd2c09aa7786f76ac2ba137c71"
                ),
            },
            "june_july_2026": {
                "path": "paper/sealed_extension/result/manifest.json",
                "sha256": (
                    "fb9a8ccbe8d28d64b6dbe70968329d09c103a6c5d773b42eee856a280c2418c1"
                ),
            },
        },
        "reports": reports,
    }


def test_frozen_contract_and_receipts_validate_against_repository() -> None:
    contracts = validator.assert_frozen(CONTRACT_DIR, repo_root=REPO_ROOT)
    assert contracts.protocol_canonical_sha256 == (
        "2d86687e3c786ca2c30dc39adeb3b28a95ecd684164a097ba8cc4b9f33ad9796"
    )
    assert contracts.protocol["scientific_status"] == (
        "DESCRIPTIVE_HISTORICAL_EVIDENCE_ONLY"
    )


def test_freeze_validator_is_executable() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CONTRACT_DIR / "validate_ablation_protocol.py"),
            str(CONTRACT_DIR),
            "--repo-root",
            str(REPO_ROOT),
            "--phase",
            "freeze",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    assert "historical_evaluation_classification=CONTAMINATED_HISTORICAL_HOLDOUT" in (
        result.stdout
    )
    assert "august_access_authorized=false" in result.stdout


def test_protocol_discloses_access_status_and_all_historical_failed_gates() -> None:
    protocol = _json(validator.PROTOCOL_FILE)
    receipt = _json(validator.FREEZE_RECEIPT_FILE)
    assert protocol["status"] == {
        "implementation": "PENDING_VERIFICATION",
        "development_training": "PENDING",
        "historical_evaluation": "PENDING_NEW_ARM_REPLAY",
        "fresh_holdout": "NOT_REGISTERED_AND_NOT_AUTHORIZED",
    }
    assert receipt["frozen_before_new_ablation_training"] is True
    assert receipt["frozen_before_historical_outcome_access"] is False
    assert receipt["historical_evaluation_opened_by_predecessor"] is True
    assert receipt["new_development_training_verdict"] == "PENDING"
    assert receipt["parent_original_external_gate"] == "FAIL"
    assert set(receipt["parent_failed_external_gates"]) == {
        "20ms_route_latency_nonnegative_vs_external_comparator",
        "double_commission_nonnegative_vs_external_comparator",
        "relative_drawdown_criterion",
        "adverse_tick_nonnegative_vs_external_comparator",
        "paired_bootstrap_lower_bound_vs_external_comparator_positive",
        "sharpe_exceeds_external_comparator",
    }
    assert receipt["strongest_action_justified"] == (
        "PAPER_DESCRIPTIVE_ONLY_AND_COLLECT_FRESH_PROSPECTIVE_DATA"
    )


def test_access_ledger_receipt_binds_exact_ledger_and_classifications() -> None:
    contracts = validator.load_and_validate_contracts(
        CONTRACT_DIR, repo_root=REPO_ROOT, verify_repository_artifacts=True
    )
    ledger_receipt, freeze_receipt = validator.validate_freeze_receipts(
        CONTRACT_DIR, contracts
    )
    assert ledger_receipt["access_ledger_file_sha256"] == (
        contracts.access_ledger_file_sha256
    )
    assert ledger_receipt["period_classifications"] == {
        "development": "DEVELOPMENT",
        "may_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "june_july_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "august_2026": "FRESH_OOS_UNOPENED_FORBIDDEN",
    }
    assert freeze_receipt["august_access_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda p, _l, _r: p["historical_evaluation"][0].update(
                {"label": "FRESH_OOS"}
            ),
            "classification",
        ),
        (
            lambda p, _l, _r: p["historical_evaluation"][1].update(
                {"previously_inspected": False}
            ),
            "previously inspected",
        ),
        (
            lambda p, _l, _r: p["forbidden_period"].update(
                {"access_authorized": True}
            ),
            "unauthorized",
        ),
        (
            lambda p, _l, _r: p["design"].update({"required_trader_jobs": 10}),
            "15",
        ),
        (
            lambda p, _l, _r: p["evaluation"].update(
                {"scenarios": ["nominal"]}
            ),
            "exactly",
        ),
        (
            lambda p, _l, _r: p["design"].update(
                {"result_selection": "Keep only favorable results"}
            ),
            "retention",
        ),
        (
            lambda _p, _l, r: r["parent_conditions"]["cvp_hddqn"].update(
                {"new_training_permitted": True}
            ),
            "retraining",
        ),
        (
            lambda _p, _l, r: r["new_arms"]["no_publisher"].update(
                {"message_mode": "FROZEN_PUBLISHER_MESSAGE"}
            ),
            "exact float32 zeros",
        ),
    ],
)
def test_contract_validator_fails_closed_on_integrity_mutations(mutation, message: str) -> None:
    protocol = _json(validator.PROTOCOL_FILE)
    ledger = _json(validator.ACCESS_LEDGER_FILE)
    registry = _json(validator.FEATURE_ARM_REGISTRY_FILE)
    mutation(protocol, ledger, registry)
    errors = validator.validate_contract_values(protocol, ledger, registry)
    assert errors
    assert message.lower() in "\n".join(errors).lower()


def test_feature_partition_gap_or_overlap_is_rejected() -> None:
    protocol = _json(validator.PROTOCOL_FILE)
    ledger = _json(validator.ACCESS_LEDGER_FILE)
    registry = _json(validator.FEATURE_ARM_REGISTRY_FILE)
    flow = registry["new_arms"]["flow_profile_only"]["representation"]
    flow["l2_level_zero"].remove("asinh_mlofi")
    flow["l2_level_keep"].append("log1p_bid_size")
    errors = validator.validate_contract_values(protocol, ledger, registry)
    assert "flow_profile_only L2-level partition is not exact" in errors


def test_tampered_ledger_breaks_protocol_hash_binding(tmp_path: Path) -> None:
    copied = tmp_path / "contracts"
    shutil.copytree(CONTRACT_DIR, copied)
    ledger_path = copied / validator.ACCESS_LEDGER_FILE
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["strongest_action_justified"] = "LIVE"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(validator.ContractValidationError, match="hash mismatch"):
        validator.load_and_validate_contracts(
            copied, verify_repository_artifacts=False
        )


def test_tampered_freeze_receipt_breaks_canonical_digest_binding(tmp_path: Path) -> None:
    copied = tmp_path / "contracts"
    shutil.copytree(CONTRACT_DIR, copied)
    receipt_path = copied / validator.FREEZE_RECEIPT_FILE
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["protocol_canonical_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    contracts = validator.load_and_validate_contracts(
        copied, verify_repository_artifacts=False
    )
    with pytest.raises(validator.ContractValidationError, match="canonical protocol"):
        validator.validate_freeze_receipts(copied, contracts)


@pytest.mark.parametrize(
    ("day", "purpose", "period"),
    [
        ("2025-01-01", "train", "development"),
        ("2026-04-30", "train", "development"),
        ("2026-05-01", "evaluate", "may_2026"),
        ("2026-05-31", "evaluate", "may_2026"),
        ("2026-06-01", "evaluate", "june_july_2026"),
        ("2026-07-31", "evaluate", "june_july_2026"),
    ],
)
def test_registered_date_boundaries_are_allowed(day: str, purpose: str, period: str) -> None:
    assert validator.assert_registered_date(day, purpose, period).isoformat() == day


@pytest.mark.parametrize(
    ("day", "purpose", "period"),
    [
        ("2024-12-31", "train", "development"),
        ("2026-05-01", "train", "development"),
        ("2026-06-01", "evaluate", "may_2026"),
        ("2026-08-01", "evaluate", "june_july_2026"),
        ("2026-08-31", "evaluate", "june_july_2026"),
        ("2026-09-01", "evaluate", "june_july_2026"),
    ],
)
def test_unregistered_or_august_dates_fail_closed(
    day: str, purpose: str, period: str
) -> None:
    with pytest.raises(PermissionError):
        validator.assert_registered_date(day, purpose, period)


def test_august_path_rejected_before_any_listing_or_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"read": 0, "list": 0, "open": 0}

    def forbidden_read(*_args, **_kwargs):
        calls["read"] += 1
        raise AssertionError("read_bytes must not run")

    def forbidden_list(*_args, **_kwargs):
        calls["list"] += 1
        raise AssertionError("iterdir must not run")

    def forbidden_open(*_args, **_kwargs):
        calls["open"] += 1
        raise AssertionError("open must not run")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "iterdir", forbidden_list)
    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(PermissionError, match="August"):
        validator.load_registered_manifest(
            REPO_ROOT,
            "raw/gc/2026-08-01/manifest.json",
            purpose="evaluate",
            period_id="june_july_2026",
        )
    assert calls == {"read": 0, "list": 0, "open": 0}


def test_unregistered_path_rejected_before_any_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_read(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("read_bytes must not run")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(PermissionError, match="exact frozen"):
        validator.load_registered_manifest(
            REPO_ROOT,
            "paper/sealed_extension/dataset/copy.json",
            purpose="evaluate",
            period_id="june_july_2026",
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("purpose", "period", "path"),
    [
        (
            "train",
            "development",
            "viability/artifacts/development/"
            "v6_registered_train_dataset_terminal_eligible/manifest.json",
        ),
        (
            "evaluate",
            "may_2026",
            "viability/artifacts/development/"
            "v6_registered_validation_dataset_eligible/manifest.json",
        ),
        (
            "evaluate",
            "june_july_2026",
            "paper/sealed_extension/dataset/manifest.json",
        ),
    ],
)
def test_only_exact_hash_bound_manifests_can_be_loaded(
    purpose: str, period: str, path: str
) -> None:
    value = validator.load_registered_manifest(
        REPO_ROOT, path, purpose=purpose, period_id=period
    )
    assert isinstance(value, dict)


def test_expected_matrices_are_exact() -> None:
    training = validator.expected_training_keys()
    evaluation = validator.expected_evaluation_keys()
    assert len(training) == 15
    assert len(evaluation) == 24
    assert training == {
        (arm, seed) for arm in validator.ARMS for seed in validator.SEEDS
    }
    assert evaluation == {
        (arm, period, scenario)
        for arm in validator.ARMS
        for period in validator.PERIODS
        for scenario in validator.SCENARIOS
    }


def test_complete_training_manifest_accepts_exact_15_job_matrix() -> None:
    assert validator.validate_complete_training_manifest(_training_manifest()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("classification", "FRESH_OOS"),
        ("historical_evaluation_read", True),
        ("august_accessed", True),
    ],
)
def test_training_manifest_rejects_contaminated_or_august_access(
    field: str, value: object
) -> None:
    manifest = _training_manifest()
    manifest[field] = value
    assert validator.validate_complete_training_manifest(manifest)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_initialization"])
def test_training_manifest_rejects_incomplete_or_unmatched_jobs(mutation: str) -> None:
    value = _training_manifest()
    jobs = value["jobs"]
    if mutation == "missing":
        jobs.pop()
    elif mutation == "duplicate":
        jobs[-1] = copy.deepcopy(jobs[0])
    else:
        jobs[0]["initialization_sha256"] = "f" * 64
    errors = validator.validate_complete_training_manifest(value)
    assert errors


def test_training_manifest_rejects_same_initialization_across_different_seeds() -> None:
    value = _training_manifest()
    first = value["jobs"][0]["initialization_sha256"]
    for job in value["jobs"]:
        job["initialization_sha256"] = first
    errors = validator.validate_complete_training_manifest(value)
    assert "different seeds must have distinct seed-specific initializations" in errors


@pytest.mark.parametrize("pnl_sign", [-1, 1])
def test_evaluation_manifest_inclusion_is_independent_of_performance(pnl_sign: int) -> None:
    value = _evaluation_manifest(pnl_sign=pnl_sign)
    assert validator.validate_complete_evaluation_manifest(value) == []


def test_evaluation_manifest_rejects_missing_or_hidden_result() -> None:
    value = _evaluation_manifest()
    value["reports"].pop()
    assert any(
        "exact 24" in error
        for error in validator.validate_complete_evaluation_manifest(value)
    )
    value = _evaluation_manifest()
    value["reports"][3]["included_in_reporting"] = False
    assert any(
        "omitted" in error
        for error in validator.validate_complete_evaluation_manifest(value)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fresh_oos", True),
        ("august_accessed", True),
        ("result_dependent_omission", True),
        ("period_classification", {"may_2026": "FRESH_OOS"}),
        ("registered_contrasts", []),
        ("parent_reference_reports", {}),
    ],
)
def test_evaluation_manifest_rejects_false_evidence_disclosures(
    field: str, value: object
) -> None:
    manifest = _evaluation_manifest()
    manifest[field] = value
    assert validator.validate_complete_evaluation_manifest(manifest)


def test_registered_technical_failure_remains_reportable_with_diagnostic() -> None:
    value = _evaluation_manifest()
    failed = value["reports"][0]
    failed["status"] = "TECHNICAL_FAILURE"
    failed["technical_failure_reason"] = "HASH_MISMATCH"
    failed["diagnostic"] = failed.pop("report")
    assert validator.validate_complete_evaluation_manifest(value) == []


def test_unregistered_failure_reason_cannot_hide_an_economic_loss() -> None:
    value = _evaluation_manifest(pnl_sign=-1)
    failed = value["reports"][0]
    failed["status"] = "TECHNICAL_FAILURE"
    failed["technical_failure_reason"] = "NEGATIVE_PNL"
    errors = validator.validate_complete_evaluation_manifest(value)
    assert any("unregistered failure reason" in error for error in errors)


def test_protocol_has_no_new_baseline_arms() -> None:
    registry = _json(validator.FEATURE_ARM_REGISTRY_FILE)
    registered_conditions = set(registry["parent_conditions"]) | set(
        registry["new_arms"]
    )
    assert registered_conditions == {
        "cvp_hddqn",
        "restricted_representation",
        "flow_profile_only",
        "depth_ofi_only",
        "no_publisher",
    }


def test_protocol_freezes_all_seven_ablation_contrasts() -> None:
    protocol = _json(validator.PROTOCOL_FILE)
    assert protocol["registered_contrasts"] == [
        "cvp_hddqn_minus_restricted_representation",
        "cvp_hddqn_minus_flow_profile_only",
        "cvp_hddqn_minus_depth_ofi_only",
        "flow_profile_only_minus_restricted_representation",
        "depth_ofi_only_minus_restricted_representation",
        "cvp_hddqn_minus_no_publisher",
        "representation_two_by_two_interaction_secondary_diagnostic",
    ]
