"""Fail-closed completion manifests for training and historical replay."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping

import numpy as np

from kx_simple_hmadrl.volume_v6_eval import block_bootstrap_lower

from .artifacts import (
    atomic_json,
    file_sha256,
    read_json_object,
    relative_path,
    resolve_bound_path,
    resolve_project_path,
)
from .contract import (
    ABLATION_ARMS,
    PARENT_REFERENCE_REPORTS,
    PERIODS,
    PROTOCOL_ID,
    REGISTERED_CONTRASTS,
    REQUIRED_ENSEMBLE_EVALUATIONS,
    REQUIRED_TRAINING_JOBS,
    SCENARIOS,
    SEEDS,
)
from .initialization import verify_initialization
from .protocol import (
    implementation_binding,
    protocol_binding,
    validate_complete_evaluation_value,
    validate_complete_training_value,
    verify_implementation_binding,
)
from .training import verify_training_member


COMPLETE_TRAINING_ARTIFACT = "CVP_ABLATION_COMPLETE_TRAINING"
COMPLETE_EVALUATION_ARTIFACT = "CVP_ABLATION_COMPLETE_EVALUATION"
REPLAY_ARTIFACT = "CVP_ABLATION_COMMON_LEDGER_REPLAY"
TECHNICAL_FAILURE_ARTIFACT = "CVP_ABLATION_TECHNICAL_FAILURE"


def _verify_parent_reference_reports() -> None:
    for period, record in PARENT_REFERENCE_REPORTS.items():
        path = resolve_project_path(record["path"])
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"frozen parent reference changed for {period}")


def build_complete_training_manifest(output_root: Path) -> Path:
    root = Path(output_root).resolve()
    destination = root / "training" / "manifest.json"
    if destination.is_file():
        verify_complete_training_manifest(destination)
        return destination
    jobs: list[dict[str, object]] = []
    initialization_records: list[dict[str, object]] = []
    per_seed_digest: dict[int, str] = {}
    for seed in SEEDS:
        initialization_path = root / "initializations" / f"seed_{seed}" / "manifest.json"
        initialization = verify_initialization(initialization_path, expected_seed=seed)
        digest = str(initialization["trader_state_sha256"])
        per_seed_digest[seed] = digest
        initialization_records.append({
            "seed": seed,
            "path": relative_path(destination.parent, initialization_path),
            "sha256": file_sha256(initialization_path),
            "trader_state_sha256": digest,
        })
        for arm in ABLATION_ARMS:
            member = root / "training" / arm / f"seed_{seed}" / "manifest.json"
            value = verify_training_member(
                member, expected_arm=arm, expected_seed=seed,
            )
            if value.get("initialization_sha256") != digest:
                raise ValueError("same-seed ablation arms do not share initialization")
            relative = relative_path(destination.parent, member)
            sha256 = file_sha256(member)
            jobs.append({
                "arm": arm,
                "seed": seed,
                "status": "COMPLETE",
                "initialization_sha256": digest,
                "path": relative,
                "sha256": sha256,
                "manifest": {"path": relative, "sha256": sha256},
            })
    if len(set(per_seed_digest.values())) != len(SEEDS):
        raise ValueError("different registered seeds share an initialization digest")
    value = {
        "schema_version": 1,
        "artifact_type": COMPLETE_TRAINING_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "complete": True,
        "arms": list(ABLATION_ARMS),
        "seeds": list(SEEDS),
        "required_training_jobs": REQUIRED_TRAINING_JOBS,
        "classification": "DEVELOPMENT",
        "historical_evaluation_read": False,
        "august_accessed": False,
        "initializations": initialization_records,
        "jobs": jobs,
        "result_dependent_omission_permitted": False,
        "holdout_read": False,
    }
    validate_complete_training_value(value)
    atomic_json(destination, value)
    verify_complete_training_manifest(destination)
    return destination


def verify_complete_training_manifest(path: Path) -> dict[str, object]:
    manifest_path = Path(path).resolve()
    value = read_json_object(manifest_path)
    verify_implementation_binding(value.get("implementation"))
    validate_complete_training_value(value)
    if (
        value.get("required_training_jobs") != REQUIRED_TRAINING_JOBS
        or value.get("arms") != list(ABLATION_ARMS)
        or value.get("seeds") != list(SEEDS)
        or value.get("holdout_read") is not False
        or value.get("historical_evaluation_read") is not False
        or value.get("august_accessed") is not False
        or value.get("result_dependent_omission_permitted") is not False
    ):
        raise ValueError("complete ablation training manifest identity differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("complete ablation training protocol binding differs")
    jobs = value.get("jobs")
    assert isinstance(jobs, list)
    initializations: dict[int, str] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("training job record is malformed")
        record = job.get("manifest")
        if not isinstance(record, dict):
            raise ValueError("training job lacks nested manifest record")
        member = resolve_bound_path(
            manifest_path.parent, record.get("path"),
            require_within=manifest_path.parent,
        )
        if not member.is_file() or file_sha256(member) != record.get("sha256"):
            raise ValueError("complete training member hash differs")
        arm, seed = str(job["arm"]), int(job["seed"])
        product = verify_training_member(
            member, expected_arm=arm, expected_seed=seed,
        )
        digest = str(product["initialization_sha256"])
        if digest != job.get("initialization_sha256"):
            raise ValueError("training job initialization alias differs")
        prior = initializations.setdefault(seed, digest)
        if prior != digest:
            raise ValueError("same-seed training members differ in initialization")
    if len(initializations) != len(SEEDS) or len(set(initializations.values())) != len(SEEDS):
        raise ValueError("registered seed initialization uniqueness failed")
    initialization_records = value.get("initializations")
    if not isinstance(initialization_records, list) or len(initialization_records) != len(SEEDS):
        raise ValueError("complete training initialization set is incomplete")
    observed_initializations: dict[int, str] = {}
    for record in initialization_records:
        if not isinstance(record, dict):
            raise ValueError("complete training initialization record is malformed")
        seed = int(record.get("seed", -1))
        initialization_path = resolve_bound_path(
            manifest_path.parent, record.get("path"),
            require_within=manifest_path.parent.parent,
        )
        if not initialization_path.is_file() or file_sha256(
            initialization_path
        ) != record.get("sha256"):
            raise ValueError("complete training initialization hash differs")
        initialization = verify_initialization(
            initialization_path, expected_seed=seed,
        )
        digest = str(initialization.get("trader_state_sha256"))
        if digest != record.get("trader_state_sha256"):
            raise ValueError("complete training initialization digest differs")
        observed_initializations[seed] = digest
    if observed_initializations != initializations:
        raise ValueError("complete training jobs and initializations differ")
    return value


def _paired_effect(first: Mapping[str, object], second: Mapping[str, object]) -> dict[str, object]:
    left = first.get("daily_pnl_cents")
    right = second.get("daily_pnl_cents")
    if not isinstance(left, dict) or not isinstance(right, dict) or sorted(left) != sorted(right):
        raise ValueError("paired ablation reports have different dates")
    days = sorted(left)
    difference_cents = np.asarray(
        [int(left[day]) - int(right[day]) for day in days], dtype=np.float64,
    )
    difference_usd = difference_cents / 100.0
    return {
        "first_arm": first["arm"],
        "second_arm": second["arm"],
        "sessions": len(days),
        "total_difference_cents": int(difference_cents.sum()),
        "total_difference_usd": float(difference_usd.sum()),
        "mean_daily_difference_usd": float(difference_usd.mean()),
        "five_session_moving_block_bootstrap_90pct_lower_usd": block_bootstrap_lower(
            difference_usd, block=5, confidence=0.90, repetitions=10_000, seed=1103,
        ),
        "inferential_status": "POST_HOC_DESCRIPTIVE",
    }


def build_complete_evaluation_manifest(output_root: Path) -> Path:
    root = Path(output_root).resolve()
    training_manifest = root / "training" / "manifest.json"
    verify_complete_training_manifest(training_manifest)
    _verify_parent_reference_reports()
    destination = root / "evaluation" / "manifest.json"
    if destination.is_file():
        verify_complete_evaluation_manifest(destination)
        return destination
    records: list[dict[str, object]] = []
    reports_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for period in PERIODS:
        for scenario in SCENARIOS:
            for arm in ABLATION_ARMS:
                report_path = root / "evaluation" / period / scenario / arm / "report.json"
                report = read_json_object(report_path)
                artifact_type = report.get("artifact_type")
                if artifact_type == REPLAY_ARTIFACT:
                    from .replay import verify_replay_report

                    report = verify_replay_report(
                        report_path, expected_arm=arm, expected_period=period,
                        expected_scenario=scenario,
                    )
                    status = "COMPLETE"
                    reason = None
                elif artifact_type == TECHNICAL_FAILURE_ARTIFACT:
                    from .replay import verify_technical_failure_report

                    report = verify_technical_failure_report(
                        report_path, expected_arm=arm, expected_period=period,
                        expected_scenario=scenario,
                    )
                    status = "TECHNICAL_FAILURE"
                    reason = report.get("technical_failure_reason")
                else:
                    raise ValueError("registered evaluation report artifact type differs")
                if (
                    report.get("arm") != arm
                    or report.get("period") != period
                    or report.get("scenario") != scenario
                    or report.get("included_in_reporting") is not True
                ):
                    raise ValueError("registered evaluation report identity differs")
                relative = relative_path(destination.parent, report_path)
                sha256 = file_sha256(report_path)
                records.append({
                    "arm": arm,
                    "period": period,
                    "scenario": scenario,
                    "status": status,
                    "path": relative,
                    "sha256": sha256,
                    "report": {"path": relative, "sha256": sha256},
                    "included_in_reporting": True,
                    "technical_failure_reason": reason,
                })
                reports_by_key[(arm, period, scenario)] = report
    paired: list[dict[str, object]] = []
    for period in PERIODS:
        for scenario in SCENARIOS:
            complete = {
                arm: reports_by_key[(arm, period, scenario)]
                for arm in ABLATION_ARMS
                if reports_by_key[(arm, period, scenario)].get("artifact_type")
                == REPLAY_ARTIFACT
            }
            for first_arm, second_arm in combinations(sorted(complete), 2):
                paired.append({
                    "period": period,
                    "scenario": scenario,
                    **_paired_effect(complete[first_arm], complete[second_arm]),
                })
    value = {
        "schema_version": 1,
        "artifact_type": COMPLETE_EVALUATION_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "complete": True,
        "arms": list(ABLATION_ARMS),
        "periods": list(PERIODS),
        "scenarios": list(SCENARIOS),
        "required_ensemble_evaluations": REQUIRED_ENSEMBLE_EVALUATIONS,
        "period_classification": {
            "may_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
            "june_july_2026": "CONTAMINATED_HISTORICAL_HOLDOUT",
        },
        "fresh_oos": False,
        "august_accessed": False,
        "result_dependent_omission": False,
        "training_manifest": {
            "path": relative_path(destination.parent, training_manifest),
            "sha256": file_sha256(training_manifest),
        },
        "registered_contrasts": list(REGISTERED_CONTRASTS),
        "parent_reference_reports": PARENT_REFERENCE_REPORTS,
        "reports": records,
        "paired_effects": paired,
        "result_dependent_omission_permitted": False,
        "scientific_status": "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE",
    }
    validate_complete_evaluation_value(value)
    atomic_json(destination, value)
    verify_complete_evaluation_manifest(destination)
    return destination


def verify_complete_evaluation_manifest(path: Path) -> dict[str, object]:
    manifest_path = Path(path).resolve()
    value = read_json_object(manifest_path)
    verify_implementation_binding(value.get("implementation"))
    validate_complete_evaluation_value(value)
    _verify_parent_reference_reports()
    if (
        value.get("required_ensemble_evaluations") != REQUIRED_ENSEMBLE_EVALUATIONS
        or value.get("arms") != list(ABLATION_ARMS)
        or value.get("periods") != list(PERIODS)
        or value.get("scenarios") != list(SCENARIOS)
        or value.get("result_dependent_omission_permitted") is not False
        or value.get("result_dependent_omission") is not False
        or value.get("fresh_oos") is not False
        or value.get("august_accessed") is not False
        or value.get("scientific_status")
        != "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE"
    ):
        raise ValueError("complete ablation evaluation identity differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("complete ablation evaluation protocol binding differs")
    training = value.get("training_manifest")
    if not isinstance(training, dict):
        raise ValueError("evaluation manifest lacks training lineage")
    training_path = resolve_bound_path(
        manifest_path.parent, training.get("path"),
        require_within=manifest_path.parent.parent,
    )
    if not training_path.is_file() or file_sha256(training_path) != training.get("sha256"):
        raise ValueError("evaluation training-manifest lineage differs")
    verify_complete_training_manifest(training_path)
    reports = value.get("reports")
    assert isinstance(reports, list)
    complete_reports: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in reports:
        assert isinstance(record, dict)
        nested = record.get("report")
        if not isinstance(nested, dict):
            raise ValueError("evaluation record lacks nested report")
        report_path = resolve_bound_path(
            manifest_path.parent, nested.get("path"),
            require_within=manifest_path.parent,
        )
        if not report_path.is_file() or file_sha256(report_path) != nested.get("sha256"):
            raise ValueError("evaluation report hash differs")
        report = read_json_object(report_path)
        if (
            report.get("arm") != record.get("arm")
            or report.get("period") != record.get("period")
            or report.get("scenario") != record.get("scenario")
            or report.get("included_in_reporting") is not True
        ):
            raise ValueError("evaluation report identity differs from suite record")
        if report.get("artifact_type") == REPLAY_ARTIFACT:
            from .replay import verify_replay_report

            verified = verify_replay_report(
                report_path, expected_arm=str(record.get("arm")),
                expected_period=str(record.get("period")),
                expected_scenario=str(record.get("scenario")),
            )
            complete_reports[(
                str(record.get("arm")), str(record.get("period")),
                str(record.get("scenario")),
            )] = verified
        elif report.get("artifact_type") == TECHNICAL_FAILURE_ARTIFACT:
            from .replay import verify_technical_failure_report

            verify_technical_failure_report(
                report_path, expected_arm=str(record.get("arm")),
                expected_period=str(record.get("period")),
                expected_scenario=str(record.get("scenario")),
            )
        else:
            raise ValueError("evaluation report has an unregistered artifact type")
    expected_paired: list[dict[str, object]] = []
    for period in PERIODS:
        for scenario in SCENARIOS:
            complete = {
                arm: complete_reports[(arm, period, scenario)]
                for arm in ABLATION_ARMS
                if (arm, period, scenario) in complete_reports
            }
            for first_arm, second_arm in combinations(sorted(complete), 2):
                expected_paired.append({
                    "period": period,
                    "scenario": scenario,
                    **_paired_effect(complete[first_arm], complete[second_arm]),
                })
    if value.get("paired_effects") != expected_paired:
        raise ValueError("complete evaluation paired effects do not reconcile")
    return value


__all__ = [
    "COMPLETE_EVALUATION_ARTIFACT", "COMPLETE_TRAINING_ARTIFACT", "REPLAY_ARTIFACT",
    "TECHNICAL_FAILURE_ARTIFACT", "build_complete_evaluation_manifest",
    "build_complete_training_manifest", "verify_complete_evaluation_manifest",
    "verify_complete_training_manifest",
]
