"""Hash-bound 16-cell replay adapter for the reduced Pilot 11 suite.

The economic implementation is not re-created here.  The adapter loads the
ten exact Pilot 10 checkpoints named by the Pilot 11 adoption receipt and
delegates observation construction, action masking, five-seed voting,
closed-loop execution, trace writing, and trace reconciliation to the frozen
Pilot 10 runtime.  Pilot 11 supplies a new protocol identity and removes the
unfinished flat-DDQN arm before any holdout result is opened.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path
from typing import Mapping, Sequence

from kx_cvp_baselines.metrics import daily_metrics
from kx_cvp_baselines.protocol import load_authorized_dataset
from kx_cvp_baselines.replay import (
    ENSEMBLE_RULE,
    LearnedEnsemble,
    LearnedMember,
    _assert_source_rows,
    _checkpoint_set,
    _evaluate_learned,
    _per_seed_report,
    _resolve_trace,
    _rows,
    _verify_decisions,
    _verify_reward_sequences,
    _write_traces,
)
from kx_cvp_baselines.training import load_training_member

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
    DATASET_MANIFESTS,
    LEARNER_ARMS,
    PERIOD_CLASSIFICATION,
    PERIODS,
    PROTOCOL_ID,
    REFERENCE_ARMS,
    REQUIRED_EVALUATION_CELLS,
    SCENARIOS,
    SEEDS,
    scenario_economics,
    validate_arm,
    validate_period,
    validate_scenario,
)
from .protocol import (
    assert_frozen_contracts,
    implementation_binding,
    protocol_binding,
    reject_forbidden_text,
    validate_workspace_output,
    verify_implementation_binding,
)
from .training_receipt import (
    ADOPTED_TRAINING_FILENAME,
    SOURCE_ENVIRONMENT_SHA256,
    SOURCE_IMPLEMENTATION_SHA256,
    adopted_member_records,
    verify_adopted_training_receipt,
)


REPLAY_ARTIFACT = "CVP_BASELINE_REDUCED_REPLAY_REPORT"
TECHNICAL_FAILURE_ARTIFACT = "CVP_BASELINE_REDUCED_TECHNICAL_FAILURE"
COMPLETE_EVALUATION_ARTIFACT = (
    "CVP_BASELINE_REDUCED_COMPLETE_EVALUATION_MANIFEST"
)
REFERENCE_REGISTRY_FILENAME = "CVP_BASELINE_REDUCED_ARM_REGISTRY.json"
CANCELLATION_LEDGER_FILENAME = "CVP_BASELINE_V1_OWNER_CANCELLATION_LEDGER.json"
PERIOD_DATASETS = {
    period: str(DATASET_MANIFESTS[period]["path"]) for period in PERIODS
}
TECHNICAL_FAILURE_REASONS = frozenset({
    "HASH_MISMATCH",
    "CAUSALITY_VIOLATION",
    "NONFINITE_STATE_OR_SCORE",
    "POLICY_LOAD_FAILURE",
    "REWARD_LEDGER_MISMATCH",
    "ACTION_LEGALITY_VIOLATION",
    "ROUTE_RECONCILIATION_FAILURE",
    "TERMINAL_INVENTORY_NOT_FLAT",
    "INCOMPLETE_TRACE",
    "UNEXPECTED_POLICY_FALLBACK",
})


def _output_root(value: Path | str | None) -> Path:
    return validate_workspace_output(
        CANONICAL_OUTPUT_ROOT if value is None else value,
    )


def _contract_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts"


def _contract_record(filename: str) -> dict[str, object]:
    path = _contract_dir() / filename
    return project_file_record(path)


def _authorize_dataset_lexically(
    relative_manifest: object, *, period: str,
) -> str:
    """Authorize an exact dataset string before resolving or opening paths."""

    reject_forbidden_text(relative_manifest, period)
    expected = DATASET_MANIFESTS.get(period)
    if (
        not isinstance(expected, Mapping)
        or expected.get("purpose") != "evaluate"
        or str(relative_manifest) != expected.get("path")
    ):
        raise PermissionError("dataset request differs from the reduced protocol")
    return str(relative_manifest)


def _receipt_path(root: Path, supplied: Path | str | None) -> Path:
    registered = root / ADOPTED_TRAINING_FILENAME
    if supplied is not None:
        candidate = Path(supplied)
        candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else resolve_project_path(candidate.as_posix())
        )
        if candidate != registered:
            raise ValueError("evaluation must use the registered adoption receipt")
    verify_adopted_training_receipt(registered)
    return registered


def _load_reduced_ensemble(
    receipt_path: Path, arm: str,
) -> LearnedEnsemble:
    receipt = verify_adopted_training_receipt(receipt_path)
    members: list[LearnedMember] = []
    for record in adopted_member_records(receipt, arm):
        seed = int(record["seed"])
        member_path = verify_project_file(record["source_manifest"])
        policy, scaler, manifest = load_training_member(
            member_path, expected_arm=arm, expected_seed=seed,
        )
        model = manifest.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("adopted training member lacks its model binding")
        if model.get("sha256") != record.get("model_sha256"):
            raise ValueError("adopted model hash differs from its receipt")
        members.append(LearnedMember(
            seed=seed,
            policy=policy,
            scaler=scaler,
            member_manifest_path=member_path,
            member_manifest_sha256=file_sha256(member_path),
            model_sha256=str(model["sha256"]),
        ))
    return LearnedEnsemble(
        arm=arm,
        members=tuple(members),
        training_manifest_path=receipt_path,
        training_manifest_sha256=file_sha256(receipt_path),
        training_manifest=receipt,
    )


def replay_cell(
    *,
    arm: str,
    period: str,
    scenario: str,
    output_root: Path | str | None = None,
    adopted_training_receipt: Path | str | None = None,
    dataset_manifest: str | None = None,
) -> Path:
    """Run one immutable learned-arm/period/scenario replay cell."""

    # All caller-controlled text is rejected before contract, output, receipt,
    # or dataset I/O.  This ordering keeps August 2026 physically unopened.
    reject_forbidden_text(
        arm, period, scenario, output_root or "",
        adopted_training_receipt or "", dataset_manifest or "",
    )
    selected_arm = validate_arm(arm).value
    selected_period = validate_period(period).value
    selected_scenario = validate_scenario(scenario).value
    relative_dataset = (
        PERIOD_DATASETS[selected_period]
        if dataset_manifest is None else str(dataset_manifest)
    )
    _authorize_dataset_lexically(relative_dataset, period=selected_period)
    assert_frozen_contracts()
    root = _output_root(output_root)
    receipt_path = _receipt_path(root, adopted_training_receipt)

    destination = (
        root / "evaluation" / selected_period / selected_scenario / selected_arm
    )
    report_path = destination / "report.json"
    partial_path = destination.with_name(f".{destination.name}.part")
    if partial_path.exists():
        raise FileExistsError("evaluation cell has an unaudited partial artifact")
    if report_path.is_file():
        existing = read_json_object(report_path)
        if existing.get("artifact_type") == REPLAY_ARTIFACT:
            verify_replay_report(
                report_path,
                expected_arm=selected_arm,
                expected_period=selected_period,
                expected_scenario=selected_scenario,
            )
        elif existing.get("artifact_type") == TECHNICAL_FAILURE_ARTIFACT:
            verify_technical_failure_report(
                report_path,
                expected_arm=selected_arm,
                expected_period=selected_period,
                expected_scenario=selected_scenario,
            )
        else:
            raise ValueError("existing reduced evaluation artifact type differs")
        return report_path
    if destination.exists():
        raise FileExistsError("evaluation cell exists without a complete report")

    # The Pilot 11 lexical guard and frozen contract have passed.  The exact
    # inherited loader may now enumerate/open only the registered period.
    dataset = load_authorized_dataset(
        relative_dataset, purpose="evaluate", period_id=selected_period,
    )
    ensemble = _load_reduced_ensemble(receipt_path, selected_arm)
    primary, seed_rows, legacy = _evaluate_learned(
        ensemble, dataset, scenario=selected_scenario,
    )
    if legacy:
        raise ValueError("reduced replay encountered unsafe legacy arrival boundaries")
    if sorted(primary.daily_pnl_cents) != list(dataset.dates):
        raise AssertionError("reduced replay did not retain the exact date set")

    work = partial_path
    if work.exists():
        raise FileExistsError("stale evaluation partial must be audited")
    work.mkdir(parents=True)
    try:
        traces = _write_traces(work, primary=primary, seed=seed_rows)
        metrics = daily_metrics(
            primary.daily_pnl_cents,
            completed_round_trips=primary.completed_round_trips,
        )
        per_seed_daily, per_seed_metrics = _per_seed_report(seed_rows)
        economics = scenario_economics(selected_scenario)
        report: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": REPLAY_ARTIFACT,
            "experiment_id": PROTOCOL_ID,
            "protocol_id": PROTOCOL_ID,
            "protocol": protocol_binding(),
            "implementation": implementation_binding(),
            "inherited_replay_implementation_aggregate_sha256": (
                SOURCE_IMPLEMENTATION_SHA256
            ),
            "inherited_environment_aggregate_sha256": SOURCE_ENVIRONMENT_SHA256,
            "status": "COMPLETE",
            "arm": selected_arm,
            "policy_family": "learned_five_seed",
            "period": selected_period,
            "phase": "validation" if selected_period == "may_2026" else "test",
            "classification": PERIOD_CLASSIFICATION[selected_period],
            "fresh_oos": False,
            "scenario": selected_scenario,
            "economics": asdict(economics),
            "evaluation_dates": list(dataset.dates),
            "daily_pnl_cents": dict(primary.daily_pnl_cents),
            "metrics": metrics,
            "completed_round_trips": primary.completed_round_trips,
            "rejected_commands": primary.rejected_commands,
            "filled_sides": primary.filled_sides,
            "commissions_cents": primary.commissions_cents,
            "explicit_commissions_usd": primary.commissions_cents / 100.0,
            "ensemble_seeds": list(SEEDS),
            "policy_replications": len(SEEDS),
            "ensemble_rule": ENSEMBLE_RULE,
            "per_seed_daily_pnl_cents": per_seed_daily,
            "per_seed_metrics": per_seed_metrics,
            "checkpoint_set": _checkpoint_set(ensemble),
            "adopted_training_receipt": project_file_record(receipt_path),
            "dataset_manifest": dataset.binding(),
            **traces,
            "reward_telescope_verified": True,
            "terminal_inventory_flat": True,
            "action_legality_verified": True,
            "routes_verified": True,
            "unsafe_legacy_arrival_boundary": False,
            "unexpected_policy_fallbacks": 0,
            "per_seed_policy_dependent_replays_complete": True,
            "registered_evaluation": True,
            "holdout_read": True,
            "august_2026_accessed": False,
            "included_in_reporting": True,
            "result_dependent_omission": False,
            "inferential_status": "POST_HOC_DESCRIPTIVE",
        }
        atomic_json(work / "report.json", report)
        os.replace(work, destination)
    except Exception:
        # A failed production attempt remains visible for explicit audit.
        raise
    verify_replay_report(
        destination / "report.json",
        expected_arm=selected_arm,
        expected_period=selected_period,
        expected_scenario=selected_scenario,
    )
    return destination / "report.json"


def _registered_dataset_binding(period: str) -> dict[str, object]:
    record = DATASET_MANIFESTS[period]
    return {
        "path": str(record["path"]),
        "path_scope": "project_root",
        "sha256": str(record["sha256"]),
        "purpose": "evaluate",
        "period_id": period,
        "classification": PERIOD_CLASSIFICATION[period],
    }


def write_technical_failure(
    *,
    arm: str,
    period: str,
    scenario: str,
    reason: str,
    diagnostic: str,
    output_root: Path | str | None = None,
    adopted_training_receipt: Path | str | None = None,
    holdout_read: bool = False,
) -> Path:
    """Retain a predeclared technical failure without fabricating economics."""

    reject_forbidden_text(
        arm, period, scenario, reason, diagnostic, output_root or "",
        adopted_training_receipt or "",
    )
    selected_arm = validate_arm(arm).value
    selected_period = validate_period(period).value
    selected_scenario = validate_scenario(scenario).value
    selected_reason = str(reason)
    message = str(diagnostic).strip()
    if selected_reason not in TECHNICAL_FAILURE_REASONS:
        raise ValueError("technical failure reason is not predeclared")
    if not message:
        raise ValueError("technical failure diagnostic must be nonempty")
    assert_frozen_contracts()
    root = _output_root(output_root)
    receipt_path = _receipt_path(root, adopted_training_receipt)
    destination = (
        root / "evaluation" / selected_period / selected_scenario
        / selected_arm / "report.json"
    )
    partial = destination.parent.with_name(f".{selected_arm}.part")
    if partial.exists():
        raise FileExistsError("technical failure cannot coexist with partial traces")
    if destination.parent.exists():
        raise FileExistsError("registered reduced evaluation result already exists")
    value: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": TECHNICAL_FAILURE_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "status": "TECHNICAL_FAILURE",
        "arm": selected_arm,
        "policy_family": "learned_five_seed",
        "period": selected_period,
        "phase": "validation" if selected_period == "may_2026" else "test",
        "classification": PERIOD_CLASSIFICATION[selected_period],
        "fresh_oos": False,
        "scenario": selected_scenario,
        "economics": asdict(scenario_economics(selected_scenario)),
        "dataset_manifest": _registered_dataset_binding(selected_period),
        "adopted_training_receipt": project_file_record(receipt_path),
        "technical_failure_reason": selected_reason,
        "diagnostic": message,
        "evaluation_dates": None,
        "daily_pnl_cents": None,
        "metrics": None,
        "decision_trace": None,
        "reward_trace": None,
        "fill_trace": None,
        "seed_decision_trace": None,
        "seed_reward_trace": None,
        "seed_fill_trace": None,
        "registered_evaluation": True,
        "holdout_read": bool(holdout_read),
        "august_2026_accessed": False,
        "included_in_reporting": True,
        "result_dependent_omission": False,
        "inferential_status": "TECHNICALLY_INVALID_REPORTED",
    }
    atomic_json(destination, value)
    verify_technical_failure_report(
        destination, expected_arm=selected_arm, expected_period=selected_period,
        expected_scenario=selected_scenario,
    )
    return destination


def verify_technical_failure_report(
    path: Path | str,
    *,
    expected_arm: str | None = None,
    expected_period: str | None = None,
    expected_scenario: str | None = None,
) -> dict[str, object]:
    """Verify one retained technical failure without economic interpretation."""

    reject_forbidden_text(
        path, expected_arm or "", expected_period or "", expected_scenario or "",
    )
    assert_frozen_contracts()
    source = Path(path).resolve()
    value = read_json_object(source)
    absent = (
        "evaluation_dates", "daily_pnl_cents", "metrics", "decision_trace",
        "reward_trace", "fill_trace", "seed_decision_trace",
        "seed_reward_trace", "seed_fill_trace",
    )
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != TECHNICAL_FAILURE_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "TECHNICAL_FAILURE"
        or value.get("policy_family") != "learned_five_seed"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("technical_failure_reason") not in TECHNICAL_FAILURE_REASONS
        or not isinstance(value.get("diagnostic"), str)
        or not str(value.get("diagnostic")).strip()
        or value.get("registered_evaluation") is not True
        or not isinstance(value.get("holdout_read"), bool)
        or value.get("august_2026_accessed") is not False
        or value.get("included_in_reporting") is not True
        or value.get("result_dependent_omission") is not False
        or value.get("inferential_status") != "TECHNICALLY_INVALID_REPORTED"
        or any(value.get(name) is not None for name in absent)
    ):
        raise ValueError("reduced technical-failure report identity differs")
    arm = validate_arm(str(value.get("arm", ""))).value
    period = validate_period(str(value.get("period", ""))).value
    scenario = validate_scenario(str(value.get("scenario", ""))).value
    if expected_arm is not None and arm != str(expected_arm):
        raise ValueError("reduced failure arm differs")
    if expected_period is not None and period != str(expected_period):
        raise ValueError("reduced failure period differs")
    if expected_scenario is not None and scenario != str(expected_scenario):
        raise ValueError("reduced failure scenario differs")
    if (
        value.get("phase") != ("validation" if period == "may_2026" else "test")
        or value.get("economics") != asdict(scenario_economics(scenario))
        or value.get("dataset_manifest") != _registered_dataset_binding(period)
        or value.get("protocol") != protocol_binding()
    ):
        raise ValueError("reduced failure protocol/economics binding differs")
    verify_implementation_binding(value.get("implementation"))
    receipt_path = verify_project_file(value.get("adopted_training_receipt"))
    verify_adopted_training_receipt(receipt_path)
    return value


def _verify_checkpoints(
    checkpoints: object, receipt: Mapping[str, object], arm: str,
) -> None:
    expected: dict[int, tuple[str, str, str]] = {}
    for record in adopted_member_records(receipt, arm):
        binding = record.get("source_manifest")
        if not isinstance(binding, Mapping):
            raise ValueError("adopted source manifest binding is malformed")
        expected[int(record["seed"])] = (
            str(binding.get("path")),
            str(binding.get("sha256")),
            str(record.get("model_sha256")),
        )
    if not isinstance(checkpoints, list) or len(checkpoints) != len(SEEDS):
        raise ValueError("reduced replay checkpoint set is incomplete")
    observed: dict[int, tuple[str, str, str]] = {}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("reduced replay checkpoint binding is malformed")
        binding = checkpoint.get("manifest")
        path = verify_project_file(binding)
        assert isinstance(binding, Mapping)
        seed = int(checkpoint.get("seed", -1))
        observed[seed] = (
            project_relative(path), file_sha256(path),
            str(checkpoint.get("model_sha256")),
        )
    if observed != expected:
        raise ValueError("reduced replay checkpoint set differs from adoption receipt")


def verify_replay_report(
    path: Path | str,
    *,
    expected_arm: str | None = None,
    expected_period: str | None = None,
    expected_scenario: str | None = None,
) -> dict[str, object]:
    """Deeply verify a Pilot 11 report and all economic trace invariants."""

    reject_forbidden_text(
        path, expected_arm or "", expected_period or "", expected_scenario or "",
    )
    assert_frozen_contracts()
    report_path = Path(path).resolve()
    value = read_json_object(report_path)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != REPLAY_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("policy_family") != "learned_five_seed"
        or value.get("ensemble_seeds") != list(SEEDS)
        or value.get("policy_replications") != len(SEEDS)
        or value.get("ensemble_rule") != ENSEMBLE_RULE
        or value.get("reward_telescope_verified") is not True
        or value.get("terminal_inventory_flat") is not True
        or value.get("action_legality_verified") is not True
        or value.get("routes_verified") is not True
        or value.get("unsafe_legacy_arrival_boundary") is not False
        or value.get("unexpected_policy_fallbacks") != 0
        or value.get("per_seed_policy_dependent_replays_complete") is not True
        or value.get("registered_evaluation") is not True
        or value.get("holdout_read") is not True
        or value.get("august_2026_accessed") is not False
        or value.get("included_in_reporting") is not True
        or value.get("result_dependent_omission") is not False
        or value.get("inferential_status") != "POST_HOC_DESCRIPTIVE"
        or value.get("inherited_replay_implementation_aggregate_sha256")
        != SOURCE_IMPLEMENTATION_SHA256
        or value.get("inherited_environment_aggregate_sha256")
        != SOURCE_ENVIRONMENT_SHA256
    ):
        raise ValueError("reduced replay report identity differs")
    arm = validate_arm(str(value.get("arm", ""))).value
    period = validate_period(str(value.get("period", ""))).value
    scenario = validate_scenario(str(value.get("scenario", ""))).value
    if expected_arm is not None and arm != str(expected_arm):
        raise ValueError("reduced replay arm differs")
    if expected_period is not None and period != str(expected_period):
        raise ValueError("reduced replay period differs")
    if expected_scenario is not None and scenario != str(expected_scenario):
        raise ValueError("reduced replay scenario differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("reduced replay protocol binding differs")
    verify_implementation_binding(value.get("implementation"))
    expected_phase = "validation" if period == "may_2026" else "test"
    if value.get("phase") != expected_phase:
        raise ValueError("reduced replay phase differs")
    economics = scenario_economics(scenario)
    if value.get("economics") != asdict(economics):
        raise ValueError("reduced replay economics differ")

    receipt_path = verify_project_file(value.get("adopted_training_receipt"))
    receipt = verify_adopted_training_receipt(receipt_path)
    _verify_checkpoints(value.get("checkpoint_set"), receipt, arm)

    dates = value.get("evaluation_dates")
    daily_raw = value.get("daily_pnl_cents")
    if (
        not isinstance(dates, list)
        or dates != sorted(set(str(day) for day in dates))
        or not isinstance(daily_raw, Mapping)
        or sorted(str(day) for day in daily_raw) != dates
    ):
        raise ValueError("reduced replay date/daily-cash set differs")
    daily = {str(day): int(amount) for day, amount in daily_raw.items()}
    expected_metrics = daily_metrics(
        daily, completed_round_trips=int(value.get("completed_round_trips", -1)),
    )
    if value.get("metrics") != expected_metrics:
        raise ValueError("reduced replay metrics do not reconcile")
    if (
        int(value.get("filled_sides", -1))
        != 2 * int(value.get("completed_round_trips", -1))
        or int(value.get("commissions_cents", -1)) < 0
        or float(value.get("explicit_commissions_usd", -1.0))
        != int(value.get("commissions_cents", -1)) / 100.0
    ):
        raise ValueError("reduced replay activity/commission summary differs")

    dataset = value.get("dataset_manifest")
    expected_dataset = DATASET_MANIFESTS[period]
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("path_scope") != "project_root"
        or dataset.get("path") != expected_dataset["path"]
        or dataset.get("sha256") != expected_dataset["sha256"]
        or dataset.get("purpose") != "evaluate"
        or dataset.get("period_id") != period
        or dataset.get("classification") != PERIOD_CLASSIFICATION[period]
        or dataset.get("dates") != dates
    ):
        raise ValueError("reduced replay dataset binding differs")
    dataset_path = verify_project_file(dataset)
    dataset_value = read_json_object(dataset_path)
    sessions = dataset_value.get("sessions")
    if not isinstance(sessions, Mapping) or sorted(str(day) for day in sessions) != dates:
        raise ValueError("reduced replay dates differ from dataset manifest")

    decisions = _rows(_resolve_trace(report_path, value.get("decision_trace")))
    rewards = _rows(_resolve_trace(report_path, value.get("reward_trace")))
    fills = _rows(_resolve_trace(report_path, value.get("fill_trace")))
    _verify_decisions(
        arm=arm, decisions=decisions, rewards=rewards,
        learned_ensemble=True, independent_seed=False,
    )
    _verify_reward_sequences(
        rewards=rewards, fills=fills, economics=economics, daily=daily,
        expected_rejections=int(value.get("rejected_commands", -1)),
        expected_filled_sides=int(value.get("filled_sides", -1)),
        with_seed=False,
    )
    if sum(int(row["fee_cents"]) for row in fills) != int(
        value.get("commissions_cents", -1)
    ):
        raise ValueError("reduced replay trace fees do not reconcile")

    seed_decisions = _rows(
        _resolve_trace(report_path, value.get("seed_decision_trace"))
    )
    seed_rewards = _rows(
        _resolve_trace(report_path, value.get("seed_reward_trace"))
    )
    seed_fills = _rows(_resolve_trace(report_path, value.get("seed_fill_trace")))
    if len(seed_rewards) != len(SEEDS) * len(rewards):
        raise ValueError("reduced per-seed reward trace is incomplete")
    _verify_decisions(
        arm=arm, decisions=seed_decisions, rewards=seed_rewards,
        learned_ensemble=False, independent_seed=True,
    )
    per_seed_daily = value.get("per_seed_daily_pnl_cents")
    per_seed_metrics = value.get("per_seed_metrics")
    if not isinstance(per_seed_daily, Mapping) or not isinstance(
        per_seed_metrics, Mapping
    ):
        raise ValueError("reduced per-seed diagnostics are absent")
    expected_seed_keys = {str(seed) for seed in SEEDS}
    if set(str(key) for key in per_seed_daily) != expected_seed_keys or set(
        str(key) for key in per_seed_metrics
    ) != expected_seed_keys:
        raise ValueError("reduced per-seed diagnostic identity set differs")
    flat_seed_daily: dict[str, int] = {}
    total_rejections = 0
    total_fills = 0
    total_fees = 0
    for seed in SEEDS:
        one = per_seed_daily.get(str(seed))
        metrics = per_seed_metrics.get(str(seed))
        if (
            not isinstance(one, Mapping)
            or sorted(str(day) for day in one) != dates
            or not isinstance(metrics, Mapping)
        ):
            raise ValueError("reduced per-seed evidence is incomplete")
        one_daily = {str(day): int(amount) for day, amount in one.items()}
        expected = daily_metrics(
            one_daily,
            completed_round_trips=int(metrics.get("completed_round_trips", -1)),
        )
        if any(metrics.get(name) != amount for name, amount in expected.items()):
            raise ValueError("reduced per-seed metrics do not reconcile")
        seed_rows = [
            row for row in seed_fills if int(row.get("seed", -1)) == seed
        ]
        filled = len(seed_rows)
        fees = sum(int(row.get("fee_cents", -1)) for row in seed_rows)
        if (
            int(metrics.get("filled_sides", -1)) != filled
            or int(metrics.get("commissions_cents", -1)) != fees
            or float(metrics.get("explicit_commissions_usd", -1.0)) != fees / 100.0
            or filled != 2 * int(metrics.get("completed_round_trips", -1))
        ):
            raise ValueError("reduced per-seed activity does not reconcile")
        total_rejections += int(metrics.get("rejected_commands", -1))
        total_fills += filled
        total_fees += fees
        for day, amount in one_daily.items():
            flat_seed_daily[f"{seed}:{day}"] = amount
    _verify_reward_sequences(
        rewards=seed_rewards, fills=seed_fills, economics=economics,
        daily=flat_seed_daily, expected_rejections=total_rejections,
        expected_filled_sides=total_fills, with_seed=True,
    )
    if sum(int(row["fee_cents"]) for row in seed_fills) != total_fees:
        raise ValueError("reduced per-seed trace fees do not reconcile")
    return value


def _relative_report(evaluation_root: Path, report_path: Path) -> str:
    relative = report_path.resolve().relative_to(evaluation_root.resolve())
    if len(relative.parts) != 4 or relative.parts[-1] != "report.json":
        raise ValueError("reduced evaluation report path has unexpected layout")
    return relative.as_posix()


def build_complete_evaluation_manifest(
    output_root: Path | str | None = None,
) -> Path:
    """Bind all and only the exact 2 x 2 x 4 reduced replay reports."""

    reject_forbidden_text(output_root or "")
    assert_frozen_contracts()
    root = _output_root(output_root)
    receipt_path = _receipt_path(root, None)
    evaluation_root = root / "evaluation"
    destination = evaluation_root / "manifest.json"
    if destination.is_file():
        verify_complete_evaluation_manifest(destination)
        return destination
    records: list[dict[str, object]] = []
    for arm in LEARNER_ARMS:
        for period in PERIODS:
            for scenario in SCENARIOS:
                report_path = (
                    evaluation_root / period / scenario / arm / "report.json"
                )
                partial = report_path.parent.with_name(f".{arm}.part")
                if partial.exists():
                    raise FileExistsError(
                        "complete reduced manifest cannot certify partial traces"
                    )
                header = read_json_object(report_path)
                if header.get("artifact_type") == REPLAY_ARTIFACT:
                    report = verify_replay_report(
                        report_path, expected_arm=arm, expected_period=period,
                        expected_scenario=scenario,
                    )
                    status = "COMPLETE"
                elif header.get("artifact_type") == TECHNICAL_FAILURE_ARTIFACT:
                    report = verify_technical_failure_report(
                        report_path, expected_arm=arm, expected_period=period,
                        expected_scenario=scenario,
                    )
                    status = "TECHNICAL_FAILURE"
                else:
                    raise ValueError("registered reduced artifact type differs")
                if report.get("included_in_reporting") is not True:
                    raise ValueError("registered reduced result was omitted")
                records.append({
                    "arm": arm,
                    "period": period,
                    "scenario": scenario,
                    "status": status,
                    "path": _relative_report(evaluation_root, report_path),
                    "sha256": file_sha256(report_path),
                })
    if len(records) != REQUIRED_EVALUATION_CELLS:
        raise AssertionError("reduced evaluation registry is not 16 cells")
    value = {
        "schema_version": 1,
        "artifact_type": COMPLETE_EVALUATION_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "status": "COMPLETE",
        "complete": True,
        "arms": list(LEARNER_ARMS),
        "periods": list(PERIODS),
        "scenarios": list(SCENARIOS),
        "required_cells": REQUIRED_EVALUATION_CELLS,
        "complete_cells": sum(
            record["status"] == "COMPLETE" for record in records
        ),
        "technical_failure_cells": sum(
            record["status"] == "TECHNICAL_FAILURE" for record in records
        ),
        "period_classification": {
            period: PERIOD_CLASSIFICATION[period] for period in PERIODS
        },
        "fresh_oos": False,
        "august_2026_accessed": False,
        "all_registered_results_retained": True,
        "result_dependent_omission": False,
        "result_dependent_omission_permitted": False,
        "adopted_training_receipt": project_file_record(receipt_path),
        "reference_arms": list(REFERENCE_ARMS),
        "reference_registry": _contract_record(REFERENCE_REGISTRY_FILENAME),
        "owner_cancellation_ledger": _contract_record(
            CANCELLATION_LEDGER_FILENAME
        ),
        "reports": records,
        "report_records_sha256": canonical_json_sha256(records),
        "scientific_status": "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE",
    }
    atomic_json(destination, value)
    verify_complete_evaluation_manifest(destination)
    return destination


def verify_complete_evaluation_manifest(
    path: Path | str,
) -> dict[str, object]:
    """Verify the exact ordered 16-cell registry and every bound report."""

    reject_forbidden_text(path)
    assert_frozen_contracts()
    manifest_path = Path(path).resolve()
    value = read_json_object(manifest_path)
    records = value.get("reports")
    expected_order = [
        (arm, period, scenario)
        for arm in LEARNER_ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    ]
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != COMPLETE_EVALUATION_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("complete") is not True
        or value.get("arms") != list(LEARNER_ARMS)
        or value.get("periods") != list(PERIODS)
        or value.get("scenarios") != list(SCENARIOS)
        or value.get("required_cells") != REQUIRED_EVALUATION_CELLS
        or not isinstance(value.get("complete_cells"), int)
        or not isinstance(value.get("technical_failure_cells"), int)
        or int(value.get("complete_cells", -1))
        + int(value.get("technical_failure_cells", -1))
        != REQUIRED_EVALUATION_CELLS
        or value.get("period_classification") != {
            period: PERIOD_CLASSIFICATION[period] for period in PERIODS
        }
        or value.get("fresh_oos") is not False
        or value.get("august_2026_accessed") is not False
        or value.get("all_registered_results_retained") is not True
        or value.get("result_dependent_omission") is not False
        or value.get("result_dependent_omission_permitted") is not False
        or value.get("reference_arms") != list(REFERENCE_ARMS)
        or value.get("scientific_status")
        != "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE"
        or not isinstance(records, list)
        or len(records) != REQUIRED_EVALUATION_CELLS
        or value.get("report_records_sha256") != canonical_json_sha256(records)
    ):
        raise ValueError("complete reduced evaluation identity differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("complete reduced evaluation protocol binding differs")
    verify_implementation_binding(value.get("implementation"))
    receipt_path = verify_project_file(value.get("adopted_training_receipt"))
    verify_adopted_training_receipt(receipt_path)
    registry_path = verify_project_file(value.get("reference_registry"))
    cancellation_path = verify_project_file(value.get("owner_cancellation_ledger"))
    if (
        registry_path != _contract_dir() / REFERENCE_REGISTRY_FILENAME
        or cancellation_path != _contract_dir() / CANCELLATION_LEDGER_FILENAME
    ):
        raise ValueError("reduced registry/cancellation binding differs")

    evaluation_root = manifest_path.parent.resolve()
    observed: list[tuple[str, str, str]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "arm", "period", "scenario", "status", "path", "sha256",
        }:
            raise ValueError("reduced evaluation record schema differs")
        arm = validate_arm(str(record.get("arm", ""))).value
        period = validate_period(str(record.get("period", ""))).value
        scenario = validate_scenario(str(record.get("scenario", ""))).value
        observed.append((arm, period, scenario))
        expected_relative = f"{period}/{scenario}/{arm}/report.json"
        if record.get("status") not in {"COMPLETE", "TECHNICAL_FAILURE"} or record.get("path") != expected_relative:
            raise ValueError("reduced evaluation cell identity differs")
        report_path = (evaluation_root / expected_relative).resolve()
        try:
            report_path.relative_to(evaluation_root)
        except ValueError as error:
            raise ValueError("reduced evaluation report escapes its registry") from error
        if not report_path.is_file() or file_sha256(report_path) != record.get("sha256"):
            raise ValueError("reduced evaluation report hash differs")
        if record.get("status") == "COMPLETE":
            report = verify_replay_report(
                report_path, expected_arm=arm, expected_period=period,
                expected_scenario=scenario,
            )
        else:
            report = verify_technical_failure_report(
                report_path, expected_arm=arm, expected_period=period,
                expected_scenario=scenario,
            )
        if report.get("status") != record.get("status"):
            raise ValueError("reduced evaluation report status differs")
        if report.get("included_in_reporting") is not True:
            raise ValueError("reduced evaluation result was omitted")
    if observed != expected_order:
        raise ValueError("reduced evaluation is not the ordered Cartesian product")
    observed_complete = sum(
        record.get("status") == "COMPLETE" for record in records
    )
    observed_failures = REQUIRED_EVALUATION_CELLS - observed_complete
    if (
        value.get("complete_cells") != observed_complete
        or value.get("technical_failure_cells") != observed_failures
    ):
        raise ValueError("reduced evaluation status counts do not reconcile")
    return value


def audit_replay_report_against_source(
    path: Path | str,
    *,
    expected_arm: str | None = None,
    expected_period: str | None = None,
    expected_scenario: str | None = None,
) -> dict[str, object]:
    """Regenerate one successful cell from checkpoints and causal raw data."""

    reject_forbidden_text(
        path, expected_arm or "", expected_period or "", expected_scenario or "",
    )
    report_path = Path(path).resolve()
    report = verify_replay_report(
        report_path, expected_arm=expected_arm, expected_period=expected_period,
        expected_scenario=expected_scenario,
    )
    arm = str(report["arm"])
    period = str(report["period"])
    scenario = str(report["scenario"])
    dataset_record = report.get("dataset_manifest")
    assert isinstance(dataset_record, Mapping)
    dataset = load_authorized_dataset(
        str(dataset_record["path"]), purpose="evaluate", period_id=period,
    )
    if dataset.binding() != dataset_record:
        raise ValueError("reduced source-audit dataset binding changed")
    receipt_path = verify_project_file(report.get("adopted_training_receipt"))
    ensemble = _load_reduced_ensemble(receipt_path, arm)
    primary, seeds, legacy = _evaluate_learned(
        ensemble, dataset, scenario=scenario,
    )
    if legacy:
        raise ValueError("reduced source audit encountered unsafe arrival boundaries")

    audited_rows: dict[str, int] = {}
    for field, expected_rows in (
        ("decision_trace", primary.decisions),
        ("reward_trace", primary.rewards),
        ("fill_trace", primary.fills),
        ("seed_decision_trace", seeds.decisions),
        ("seed_reward_trace", seeds.rewards),
        ("seed_fill_trace", seeds.fills),
    ):
        observed_rows = _rows(_resolve_trace(report_path, report.get(field)))
        _assert_source_rows(field, observed_rows, expected_rows)
        audited_rows[field] = len(observed_rows)

    expected_metrics = daily_metrics(
        primary.daily_pnl_cents,
        completed_round_trips=primary.completed_round_trips,
    )
    per_seed_daily, per_seed_metrics = _per_seed_report(seeds)
    if (
        report.get("daily_pnl_cents") != primary.daily_pnl_cents
        or report.get("metrics") != expected_metrics
        or int(report.get("completed_round_trips", -1))
        != primary.completed_round_trips
        or int(report.get("rejected_commands", -1)) != primary.rejected_commands
        or int(report.get("filled_sides", -1)) != primary.filled_sides
        or int(report.get("commissions_cents", -1)) != primary.commissions_cents
        or report.get("per_seed_daily_pnl_cents") != per_seed_daily
        or report.get("per_seed_metrics") != per_seed_metrics
    ):
        raise ValueError("reduced source replay summary differs from report")
    return {
        "artifact_type": "CVP_BASELINE_REDUCED_SOURCE_REPLAY_AUDIT",
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "period": period,
        "scenario": scenario,
        "report_sha256": file_sha256(report_path),
        "checkpoint_scores_recomputed": True,
        "raw_routes_and_fills_recomputed": True,
        "adverse_tick_price_recomputed": scenario == "adverse_tick",
        "gapped_books_not_carried": True,
        "reward_and_terminal_cash_recomputed": True,
        "trace_rows": audited_rows,
        "status": "PASS",
    }


def audit_complete_evaluation_against_source(
    path: Path | str,
) -> tuple[dict[str, object], ...]:
    """Source-replay every successful cell; structurally verify failures."""

    reject_forbidden_text(path)
    manifest_path = Path(path).resolve()
    manifest = verify_complete_evaluation_manifest(manifest_path)
    records = manifest.get("reports")
    assert isinstance(records, list)
    audits: list[dict[str, object]] = []
    for record in records:
        assert isinstance(record, Mapping)
        report_path = (manifest_path.parent / str(record["path"])).resolve()
        if record["status"] == "COMPLETE":
            audits.append(audit_replay_report_against_source(
                report_path, expected_arm=str(record["arm"]),
                expected_period=str(record["period"]),
                expected_scenario=str(record["scenario"]),
            ))
        else:
            verify_technical_failure_report(
                report_path, expected_arm=str(record["arm"]),
                expected_period=str(record["period"]),
                expected_scenario=str(record["scenario"]),
            )
    return tuple(audits)


def _run_job(args: tuple[str, str, str, str, str]) -> str:
    arm, period, scenario, root, receipt = args
    return str(replay_cell(
        arm=arm, period=period, scenario=scenario,
        output_root=root, adopted_training_receipt=receipt,
    ))


def replay_all(
    output_root: Path | str | None = None,
    adopted_training_receipt: Path | str | None = None,
    *, workers: int = 1,
) -> Path:
    """Run the exact 16-cell matrix, optionally in isolated worker processes."""

    reject_forbidden_text(output_root or "", adopted_training_receipt or "")
    count = int(workers)
    if count < 1:
        raise ValueError("workers must be positive")
    assert_frozen_contracts()
    root = _output_root(output_root)
    receipt = _receipt_path(root, adopted_training_receipt)
    jobs = [
        (arm, period, scenario, str(root), str(receipt))
        for arm in LEARNER_ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    ]
    if len(jobs) != REQUIRED_EVALUATION_CELLS:
        raise AssertionError("reduced replay job matrix is not exactly 16 cells")
    if count == 1:
        for job in jobs:
            _run_job(job)
    else:
        with ProcessPoolExecutor(max_workers=count) as executor:
            list(executor.map(_run_job, jobs))
    return build_complete_evaluation_manifest(root)


def evaluation_status(
    output_root: Path | str | None = None,
) -> dict[str, object]:
    """Return artifact progress only; never expose economic outcomes."""

    reject_forbidden_text(output_root or "")
    root = _output_root(output_root)
    records: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for arm in LEARNER_ARMS:
        for period in PERIODS:
            for scenario in SCENARIOS:
                cell = root / "evaluation" / period / scenario / arm
                report_path = cell / "report.json"
                partial = cell.with_name(f".{arm}.part")
                digest: str | None = None
                if report_path.is_file():
                    try:
                        header = read_json_object(report_path)
                        if (
                            header.get("artifact_type") == REPLAY_ARTIFACT
                            and header.get("status") == "COMPLETE"
                            and header.get("arm") == arm
                            and header.get("period") == period
                            and header.get("scenario") == scenario
                        ):
                            status = "COMPLETE"
                        elif (
                            header.get("artifact_type") == TECHNICAL_FAILURE_ARTIFACT
                            and header.get("status") == "TECHNICAL_FAILURE"
                            and header.get("arm") == arm
                            and header.get("period") == period
                            and header.get("scenario") == scenario
                        ):
                            status = "TECHNICAL_FAILURE"
                        else:
                            status = "INVALID_REPORT"
                        digest = file_sha256(report_path)
                    except (OSError, TypeError, ValueError):
                        status = "INVALID_REPORT"
                    if partial.exists():
                        status = "RECORDED_WITH_PARTIAL_CONFLICT"
                elif partial.exists():
                    status = "PARTIAL_RETAINED"
                elif cell.exists():
                    status = "INCOMPLETE_CELL"
                else:
                    status = "NOT_STARTED"
                counts[status] = counts.get(status, 0) + 1
                records.append({
                    "arm": arm,
                    "period": period,
                    "scenario": scenario,
                    "status": status,
                    "report_path": report_path.as_posix(),
                    "report_sha256": digest,
                })
    return {
        "experiment_id": PROTOCOL_ID,
        "required_cells": REQUIRED_EVALUATION_CELLS,
        "recorded_cells": (
            counts.get("COMPLETE", 0) + counts.get("TECHNICAL_FAILURE", 0)
        ),
        "counts": dict(sorted(counts.items())),
        "records": records,
        "result_blind": True,
        "historical_economic_values_returned": False,
        "market_data_opened": False,
        "august_2026_accessed": False,
    }


__all__ = [
    "COMPLETE_EVALUATION_ARTIFACT",
    "PERIOD_DATASETS",
    "REPLAY_ARTIFACT",
    "TECHNICAL_FAILURE_ARTIFACT",
    "TECHNICAL_FAILURE_REASONS",
    "build_complete_evaluation_manifest",
    "audit_complete_evaluation_against_source",
    "audit_replay_report_against_source",
    "evaluation_status",
    "replay_all",
    "replay_cell",
    "verify_complete_evaluation_manifest",
    "verify_replay_report",
    "verify_technical_failure_report",
    "write_technical_failure",
]
