from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kx_cvp_baselines import __main__ as cli
from kx_cvp_baselines.artifacts import canonical_json_sha256, file_sha256
from kx_cvp_baselines.contract import (
    BASELINE_ARMS,
    DATASET_MANIFESTS,
    LEARNER_ARMS,
    PERIOD_CLASSIFICATION,
    PERIODS,
    PROTOCOL_ID,
    SCENARIOS,
    scenario_economics,
)
from kx_cvp_baselines.portable import (
    BOOTSTRAP_COLUMNS,
    COMPLETE_EVALUATION_ARTIFACT,
    DAILY_COLUMNS,
    DETERMINISTIC_RULE,
    EFFECT_COLUMNS,
    EVIDENCE_ARTIFACT,
    METRICS_COLUMNS,
    PORTABLE_VERIFICATION_LEVEL,
    REPLAY_ARTIFACT,
    REPORT_ARTIFACT,
    TECHNICAL_FAILURE_ARTIFACT,
    portable_protocol_binding,
    verify_portable_evaluation_manifest,
    verify_portable_evidence_manifest,
    verify_portable_replay_report,
    verify_portable_suite,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _implementation() -> dict[str, object]:
    files: list[dict[str, str]] = []
    environment: dict[str, object] = {
        "files": [],
        "python": "3.11.0",
        "python_implementation": "CPython",
        "system": "Synthetic",
        "machine": "test",
        "packages": {"pyarrow": "test"},
    }
    environment["aggregate_sha256"] = canonical_json_sha256(environment)
    return {
        "path_scope": "project_root",
        "files": files,
        "file_count": 0,
        "aggregate_sha256": canonical_json_sha256(files),
        "environment": environment,
    }


def _project_binding(path: str) -> dict[str, object]:
    return {"path": path, "path_scope": "project_root", "sha256": "a" * 64}


def _metric(completed: int = 0) -> dict[str, object]:
    return {
        "sessions": 1,
        "net_pnl_cents": 0,
        "net_pnl_usd": 0.0,
        "mean_daily_pnl_cents": 0.0,
        "mean_daily_pnl_usd": 0.0,
        "annualized_daily_sharpe": 0.0,
        "profit_factor": None,
        "daily_profit_factor": None,
        "positive_day_fraction": 0.0,
        "maximum_drawdown_cents": 0,
        "maximum_drawdown_usd": 0.0,
        "completed_round_trips": completed,
    }


def _trace_record(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": path.name,
        "path_scope": "report_relative",
        "sha256": file_sha256(path),
        "rows": rows,
    }


def _write_complete_rule_report(root: Path) -> Path:
    report = root / "evaluation" / "may_2026" / "nominal" / "always_flat" / "report.json"
    report.parent.mkdir(parents=True)
    decision_row = {
        "session_date": "2026-05-01",
        "decision_ordinal": 0,
        "decision_time_ns": 1,
        "next_time_ns": 2,
        "seed": -1,
        "score_short": 0.0,
        "score_flat": 1.0,
        "score_long": 0.0,
        "legal_short": True,
        "legal_flat": True,
        "legal_long": True,
        "member_vote": 0,
        "desired_action": 0,
        "legality_fallback": False,
        "executed_action": 0,
        "inventory_before": 0,
        "inventory_after": 0,
        "reward_cents": 0,
        "terminal": True,
        "momentum_60m": 0.0,
        "distance_session_vwap_atr": 0.0,
    }
    reward_row = {
        "session_date": "2026-05-01",
        "decision_ordinal": 0,
        "decision_time_ns": 1,
        "next_time_ns": 2,
        "executed_action": 0,
        "legal_short": True,
        "legal_flat": True,
        "legal_long": True,
        "inventory_before": 0,
        "inventory_after": 0,
        "reward_cents": 0,
        "cumulative_executable_nlv_cents": 0,
        "command_created": False,
        "command_rejected": False,
        "terminal": True,
        "route_deadline_ns": None,
        "route_side": None,
        "route_fill_time_ns": None,
        "route_fill_price_ticks": None,
        "route_fill_fee_cents": None,
        "route_displayed_size": None,
        "route_searched_snapshots": None,
        "terminal_route_deadline_ns": None,
        "terminal_route_side": None,
        "terminal_fill_time_ns": None,
        "terminal_fill_price_ticks": None,
        "terminal_fill_fee_cents": None,
        "terminal_displayed_size": None,
    }
    decisions = report.parent / "decisions.parquet"
    rewards = report.parent / "rewards.parquet"
    fills = report.parent / "fills.parquet"
    pq.write_table(pa.Table.from_pylist([decision_row]), decisions)
    pq.write_table(pa.Table.from_pylist([reward_row]), rewards)
    fill_schema = pa.schema([
        ("session_date", pa.string()), ("fill_ordinal", pa.int32()),
        ("timestamp_ns", pa.int64()), ("side", pa.int8()),
        ("price_ticks", pa.int64()), ("fee_cents", pa.int64()),
        ("forced", pa.bool_()),
    ])
    pq.write_table(pa.Table.from_pylist([], schema=fill_schema), fills)
    dataset = DATASET_MANIFESTS["may_2026"]
    value = {
        "schema_version": 1,
        "artifact_type": REPLAY_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": portable_protocol_binding(),
        "implementation": _implementation(),
        "status": "COMPLETE",
        "arm": "always_flat",
        "policy_family": "deterministic_single_path",
        "period": "may_2026",
        "phase": "validation",
        "classification": PERIOD_CLASSIFICATION["may_2026"],
        "fresh_oos": False,
        "scenario": "nominal",
        "economics": asdict(scenario_economics("nominal")),
        "evaluation_dates": ["2026-05-01"],
        "daily_pnl_cents": {"2026-05-01": 0},
        "metrics": _metric(),
        "completed_round_trips": 0,
        "rejected_commands": 0,
        "filled_sides": 0,
        "commissions_cents": 0,
        "explicit_commissions_usd": 0.0,
        "ensemble_seeds": [],
        "policy_replications": 1,
        "ensemble_rule": DETERMINISTIC_RULE,
        "per_seed_daily_pnl_cents": None,
        "per_seed_metrics": None,
        "checkpoint_set": [],
        # All three lineage targets deliberately do not exist. Portable
        # verification must validate only their recorded shapes and digests.
        "suite_training_prerequisite": _project_binding("ignored/training.json"),
        "training_manifest": None,
        "rule_scaler_reference": {
            **_project_binding("ignored/scaler.json"),
            "scaler_canonical_sha256": "b" * 64,
            "policy_uses_scaled_values": False,
        },
        "dataset_manifest": {
            "path": dataset["path"],
            "path_scope": "project_root",
            "sha256": dataset["sha256"],
            "purpose": "evaluate",
            "period_id": "may_2026",
            "classification": PERIOD_CLASSIFICATION["may_2026"],
            "dates": ["2026-05-01"],
        },
        "decision_trace": _trace_record(decisions, 1),
        "reward_trace": _trace_record(rewards, 1),
        "fill_trace": _trace_record(fills, 0),
        "seed_decision_trace": None,
        "seed_reward_trace": None,
        "seed_fill_trace": None,
        "reward_telescope_verified": True,
        "terminal_inventory_flat": True,
        "action_legality_verified": True,
        "routes_verified": True,
        "unsafe_legacy_arrival_boundary": False,
        "legality_fallbacks": 0,
        "unexpected_policy_fallbacks": 0,
        "per_seed_policy_dependent_replays_complete": False,
        "registered_evaluation": True,
        "holdout_read": True,
        "august_accessed": False,
        "included_in_reporting": True,
        "technical_failure_reason": None,
        "inferential_status": "POST_HOC_DESCRIPTIVE",
    }
    _write_json(report, value)
    return report


def _failure_report(
    arm: str, period: str, scenario: str,
    suite: dict[str, object], training: dict[str, object],
) -> dict[str, object]:
    dataset = DATASET_MANIFESTS[period]
    learned = arm in LEARNER_ARMS
    return {
        "schema_version": 1,
        "artifact_type": TECHNICAL_FAILURE_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": portable_protocol_binding(),
        "implementation": _implementation(),
        "status": "TECHNICAL_FAILURE",
        "arm": arm,
        "policy_family": "learned_five_seed" if learned else "deterministic_single_path",
        "period": period,
        "phase": "validation" if period == "may_2026" else "test",
        "classification": PERIOD_CLASSIFICATION[period],
        "fresh_oos": False,
        "scenario": scenario,
        "economics": asdict(scenario_economics(scenario)),
        "dataset_manifest": {
            "path": dataset["path"], "path_scope": "project_root",
            "sha256": dataset["sha256"], "purpose": "evaluate",
            "period_id": period, "classification": PERIOD_CLASSIFICATION[period],
        },
        "suite_training_prerequisite": suite,
        "training_manifest": training if learned else None,
        "training_binding_status": "VERIFIED" if learned else "NOT_APPLICABLE",
        "technical_failure_reason": "HASH_MISMATCH",
        "diagnostic": "synthetic retained integrity failure",
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
        "holdout_read": False,
        "august_accessed": False,
        "included_in_reporting": True,
        "inferential_status": "TECHNICALLY_INVALID_REPORTED",
    }


def _write_failure_evaluation(project: Path, output: Path) -> Path:
    evaluation = output / "evaluation"
    suite = _project_binding("ignored/complete_training_manifest.json")
    training = _project_binding("ignored/complete_training_manifest.json")
    records: list[dict[str, object]] = []
    for arm in BASELINE_ARMS:
        for period in PERIODS:
            for scenario in SCENARIOS:
                relative = f"{period}/{scenario}/{arm}/report.json"
                report = evaluation / relative
                _write_json(report, _failure_report(arm, period, scenario, suite, training))
                records.append({
                    "arm": arm, "period": period, "scenario": scenario,
                    "status": "TECHNICAL_FAILURE", "path": relative,
                    "sha256": file_sha256(report),
                })
    value = {
        "schema_version": 1,
        "artifact_type": COMPLETE_EVALUATION_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": portable_protocol_binding(),
        "implementation": _implementation(),
        "status": "COMPLETE",
        "complete": True,
        "arms": list(BASELINE_ARMS),
        "periods": list(PERIODS),
        "scenarios": list(SCENARIOS),
        "required_cells": 64,
        "complete_cells": 0,
        "technical_failure_cells": 64,
        "period_classification": {period: PERIOD_CLASSIFICATION[period] for period in PERIODS},
        "fresh_oos": False,
        "august_accessed": False,
        "all_registered_results_retained": True,
        "result_dependent_omission": False,
        "result_dependent_omission_permitted": False,
        "training_manifest": training,
        "suite_training_prerequisite": suite,
        "learner_cells_with_verified_training": len(LEARNER_ARMS) * len(PERIODS) * len(SCENARIOS),
        "holdout_read_cells": 0,
        "reports": records,
        "report_records_sha256": canonical_json_sha256(records),
        "scientific_status": "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE",
    }
    path = evaluation / "manifest.json"
    _write_json(path, value)
    return path


def _relative(project: Path, path: Path) -> str:
    return path.relative_to(project).as_posix()


def _real_binding(project: Path, path: Path, *, rows: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": _relative(project, path),
        "path_scope": "project_root",
        "sha256": file_sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _write_empty_csv(path: Path, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(columns) + "\n", encoding="utf-8")


def _write_portable_evidence_and_report(
    project: Path, report_root: Path, evaluation: Path,
) -> None:
    source = report_root / "source_data"
    metrics = source / "metrics.csv"
    daily = source / "daily_pnl.csv"
    effects = source / "paired_effects.csv"
    draws = source / "bootstrap_draws.parquet"
    _write_empty_csv(metrics, METRICS_COLUMNS)
    _write_empty_csv(daily, DAILY_COLUMNS)
    _write_empty_csv(effects, EFFECT_COLUMNS)
    draw_schema = pa.schema([
        ("reference_policy_id", pa.string()),
        ("baseline_policy_id", pa.string()),
        ("period", pa.string()),
        ("scenario", pa.string()),
        ("replicate", pa.int32()),
        ("mean_difference_usd_per_session", pa.float64()),
    ])
    pq.write_table(pa.Table.from_pylist([], schema=draw_schema), draws)
    evidence = source / "manifest.json"
    _write_json(evidence, {
        "schema_version": 1,
        "artifact_type": EVIDENCE_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "complete": True,
        "scientific_status": "CONTAMINATED_HISTORICAL_DESCRIPTIVE",
        "period_classification": PERIOD_CLASSIFICATION,
        "june_july_headline": True,
        "may_appendix": True,
        "august_accessed": False,
        "all_registered_results_retained": True,
        "all_new_baseline_cells_retained": True,
        "registered_result_scope": "eight new baseline arms x two periods x four scenarios",
        "selected_reference_subset": {
            "included": [
                "cvp_hddqn", "no_publisher_single_agent_ddqn",
                "shavandi_khedmati_reconstruction",
            ],
            "excluded": {
                "restricted_representation": (
                    "Existing representation ablation; deliberately kept separate "
                    "from this baseline report."
                )
            },
        },
        "protocol": portable_protocol_binding(),
        "implementation": _implementation(),
        "evaluation_manifest": _real_binding(project, evaluation),
        "reference_sources": [],
        "products": {
            "metrics": _real_binding(project, metrics, rows=0),
            "daily_pnl": _real_binding(project, daily, rows=0),
            "paired_effects": _real_binding(project, effects, rows=0),
            "bootstrap_draws": _real_binding(project, draws, rows=0),
        },
        "content_sha256": "c" * 64,
    })
    markdown = report_root / "baseline_report.md"
    pdf = report_root / "baseline_report.pdf"
    markdown.write_text("# Synthetic portable report\n", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.7\n% synthetic structural fixture\n%%EOF\n")
    _write_json(report_root / "report_manifest.json", {
        "schema_version": 1,
        "artifact_type": REPORT_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "scientific_status": "CONTAMINATED_HISTORICAL_DESCRIPTIVE",
        "paper_modified": False,
        "august_accessed": False,
        "protocol": portable_protocol_binding(),
        "implementation": _implementation(),
        "evidence_manifest": _real_binding(project, evidence),
        "outputs": {
            "markdown": _real_binding(project, markdown),
            "pdf": {**_real_binding(project, pdf), "pages": 6},
        },
    })


def test_portable_complete_report_checks_traces_without_opening_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_cvp_baselines.protocol as strict_protocol

    report = _write_complete_rule_report(tmp_path)
    monkeypatch.setattr(
        strict_protocol,
        "assert_frozen_contracts",
        lambda: (_ for _ in ()).throw(AssertionError("strict artifact audit invoked")),
    )
    result = verify_portable_replay_report(report, project_root=tmp_path)
    assert result["verification_level"] == PORTABLE_VERIFICATION_LEVEL
    assert result["trace_rows"] == {
        "decision_trace": 1, "reward_trace": 1, "fill_trace": 0,
    }

    # A byte change is rejected even though no checkpoint/raw input is present.
    with (report.parent / "fills.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="file/hash differs"):
        verify_portable_replay_report(report, project_root=tmp_path)


def test_portable_report_rejects_unregistered_period_dates(tmp_path: Path) -> None:
    report = _write_complete_rule_report(tmp_path)
    value = json.loads(report.read_text(encoding="utf-8"))
    value["evaluation_dates"] = ["2026-08-01"]
    value["daily_pnl_cents"] = {"2026-08-01": 0}
    value["dataset_manifest"]["dates"] = ["2026-08-01"]
    _write_json(report, value)
    with pytest.raises(PermissionError, match="unopened and unauthorized"):
        verify_portable_replay_report(report, project_root=tmp_path)


def test_portable_evaluation_requires_exact_64_cell_registry(tmp_path: Path) -> None:
    evaluation = _write_failure_evaluation(tmp_path, tmp_path / "artifacts")
    result = verify_portable_evaluation_manifest(
        evaluation, project_root=tmp_path, allowed_root=tmp_path / "artifacts",
    )
    assert result["reports_verified"] == 64
    assert result["technical_failure_reports"] == 64
    assert result["trace_files_verified"] == 0

    value = json.loads(evaluation.read_text(encoding="utf-8"))
    value["reports"] = value["reports"][:-1]
    _write_json(evaluation, value)
    with pytest.raises(ValueError, match="identity"):
        verify_portable_evaluation_manifest(
            evaluation, project_root=tmp_path, allowed_root=tmp_path / "artifacts",
        )


def test_portable_suite_verifies_trackable_layer_and_returns_no_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_cvp_baselines.portable as portable

    project = tmp_path / "checkout"
    output = project / "artifacts"
    report_root = project / "reports"
    evaluation = _write_failure_evaluation(project, output)
    _write_portable_evidence_and_report(project, report_root, evaluation)
    monkeypatch.setattr(portable, "_expected_reference_sources", lambda root: [])
    result = verify_portable_suite(
        output, report_root, require_complete=True, project_root=project,
    )
    assert result["verification_level"] == PORTABLE_VERIFICATION_LEVEL
    assert result["models_or_corpora_opened"] is False
    assert result["raw_market_sessions_opened"] is False
    assert result["economic_values_recomputed"] is False
    assert result["strict_local_verification_performed"] is False
    assert result["source_replay_audit_performed"] is False
    assert set(result["artifacts"]) == {"evaluation", "evidence", "report"}


def test_evidence_rejects_unapproved_paths_before_hashing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_cvp_baselines.portable as portable

    project = tmp_path / "checkout"
    output = project / "artifacts"
    report_root = project / "reports"
    evaluation = _write_failure_evaluation(project, output)
    _write_portable_evidence_and_report(project, report_root, evaluation)
    evidence = report_root / "source_data" / "manifest.json"
    value = json.loads(evidence.read_text(encoding="utf-8"))
    value["reference_sources"] = [{
        "manifest_path": "ignored/checkpoint.pt",
        "manifest_sha256": "d" * 64,
        "report_path": "ignored/model_report.json",
        "report_sha256": "e" * 64,
    }]
    _write_json(evidence, value)

    real_sha256 = portable.file_sha256

    def guarded_sha256(path: Path | str) -> str:
        if "ignored" in Path(path).parts:
            raise AssertionError("portable verifier opened forbidden lineage")
        return real_sha256(path)

    monkeypatch.setattr(portable, "file_sha256", guarded_sha256)
    monkeypatch.setattr(
        portable, "_expected_reference_sources", lambda root: value["reference_sources"],
    )
    with pytest.raises(ValueError, match="frozen allowlist"):
        verify_portable_evidence_manifest(
            evidence, project_root=project, expected_evaluation_path=evaluation,
        )

    value["reference_sources"] = []
    value["products"]["metrics"]["path"] = "ignored/fqi_corpus.parquet"
    _write_json(evidence, value)
    monkeypatch.setattr(portable, "_expected_reference_sources", lambda root: [])
    with pytest.raises(ValueError, match="location differs"):
        verify_portable_evidence_manifest(
            evidence, project_root=project, expected_evaluation_path=evaluation,
        )


def test_evaluation_symlink_escape_is_rejected_before_json_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_cvp_baselines.portable as portable

    project = tmp_path / "checkout"
    output = project / "artifacts"
    outside = tmp_path / "outside" / "evaluation"
    outside.mkdir(parents=True)
    escaped = outside / "manifest.json"
    _write_json(escaped, {"should_not_be_opened": True})
    output.mkdir(parents=True)
    (output / "evaluation").symlink_to(outside, target_is_directory=True)

    real_reader = portable.read_json_object

    def guarded_reader(path: Path | str) -> dict[str, object]:
        if Path(path).resolve() == escaped.resolve():
            raise AssertionError("portable verifier opened a symlink escape")
        return real_reader(path)

    monkeypatch.setattr(portable, "read_json_object", guarded_reader)
    with pytest.raises(ValueError, match="outside its allowed root"):
        verify_portable_evaluation_manifest(
            output / "evaluation" / "manifest.json",
            project_root=project,
            allowed_root=output,
        )


def test_reference_selection_derives_exact_20_registered_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_cvp_baselines.portable as portable

    project = tmp_path / "checkout"

    def record(arm: str, period: str, scenario: str, suffix: str) -> dict[str, object]:
        return {
            "arm": arm,
            "period": period,
            "scenario": scenario,
            "path": f"reports/{period}/{scenario}/{arm}/report.json",
            "sha256": suffix * 64,
        }

    aliases = portable.SCENARIO_ALIASES
    cvp_may_records = [
        record("CVP-HDDQN", "may_2026", aliases[scenario], "a")
        for scenario in SCENARIOS
    ]
    # Same scenario, excluded arm: the portable selector must not substitute it.
    cvp_may_records.append(
        record("V6-Restricted", "may_2026", aliases["nominal"], "b")
    )
    cvp_jj_records = [
        record("CVP-HDDQN", "june_july_2026", aliases[scenario], "b")
        for scenario in SCENARIOS
    ]
    no_publisher_records = [
        record("no_publisher", period, scenario, "c")
        for period in PERIODS for scenario in SCENARIOS
    ]
    base_records = [
        record("basepaper", "may_2026", aliases[scenario], "d")
        for scenario in SCENARIOS
    ]
    manifests = {
        "cvp_may": cvp_may_records,
        "cvp_jj": cvp_jj_records,
        "no_publisher": no_publisher_records,
        "base": base_records,
    }
    registrations: dict[str, tuple[str, str]] = {}
    for name, records in manifests.items():
        path = project / "refs" / name / "manifest.json"
        _write_json(path, {"reports": records})
        registrations[name] = (_relative(project, path), file_sha256(path))
    monkeypatch.setattr(portable, "REFERENCE_MANIFESTS", {
        "cvp_hddqn": {
            "may_2026": registrations["cvp_may"],
            "june_july_2026": registrations["cvp_jj"],
        },
        "no_publisher_single_agent_ddqn": {
            "all": registrations["no_publisher"],
        },
        "shavandi_khedmati_reconstruction": {
            "may_2026": registrations["base"],
        },
    })
    sources = portable._expected_reference_sources(project)
    assert len(sources) == 20
    assert all("V6-Restricted" not in str(source["report_path"]) for source in sources)
    assert sources == sorted(
        sources,
        key=lambda item: (
            item["manifest_path"], item["manifest_sha256"],
            item["report_path"], item["report_sha256"],
        ),
    )


def test_cli_exposes_portable_label_without_invoking_strict_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import kx_cvp_baselines.portable as portable

    monkeypatch.setattr(cli, "_output_root", lambda value: tmp_path / "artifacts")
    monkeypatch.setattr(cli, "_report_root", lambda value: tmp_path / "reports")
    calls: list[tuple[Path, Path, bool]] = []

    def fake_verify(output: Path, reports: Path, *, require_complete: bool) -> dict[str, object]:
        calls.append((output, reports, require_complete))
        return {"status": "PASS", "verification_level": PORTABLE_VERIFICATION_LEVEL}

    monkeypatch.setattr(portable, "verify_portable_suite", fake_verify)
    assert cli.main(["verify-portable", "--require-complete"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["verification_level"] == PORTABLE_VERIFICATION_LEVEL
    assert calls == [(tmp_path / "artifacts", tmp_path / "reports", True)]
