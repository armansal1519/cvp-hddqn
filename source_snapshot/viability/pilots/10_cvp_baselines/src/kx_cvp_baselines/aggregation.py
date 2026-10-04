"""Hash-bound, result-complete evidence products for the baseline suite."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import (
    atomic_json,
    canonical_json_sha256,
    file_sha256,
    project_relative,
    read_json_object,
    resolve_project_path,
    workspace_root,
)
from .contract import (
    BASELINE_ARMS,
    CANONICAL_OUTPUT_ROOT,
    PERIOD_CLASSIFICATION,
    PERIODS,
    PROTOCOL_ID,
    SCENARIOS,
)
from .metrics import (
    BOOTSTRAP_BLOCK,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    daily_metrics,
    moving_block_means,
    paired_effect,
)
from .protocol import (
    implementation_binding,
    protocol_binding,
    reject_forbidden_text,
    validate_workspace_output,
    verify_implementation_binding,
)
from .references import ReferenceReport, load_all_reference_reports


EVIDENCE_ARTIFACT = "CVP_BASELINE_SUITE_EVIDENCE"
DEFAULT_REPORT_ROOT = "viability/reports/cvp_baseline_suite_v1"

PUBLIC_NAMES: dict[str, str] = {
    "cvp_hddqn": "CVP-HDDQN",
    "flat_multiclock_ddqn": "Flat multi-clock DDQN",
    "recurrent_ppo": "Recurrent PPO",
    "lightgbm_fitted_q": "LightGBM fitted Q",
    "no_publisher_single_agent_ddqn": "No-publisher recurrent DDQN",
    "tsmom_60m": "60-minute momentum rule",
    "vwap_reversion_atr1": "VWAP reversion (1 ATR)",
    "always_flat": "Always flat",
    "session_long": "Session long",
    "session_short": "Session short",
    "shavandi_khedmati_reconstruction": "Shavandi-Khedmati reconstruction",
}

ARTIFACT_ARM_IDS: dict[str, str] = {
    "cvp_hddqn": "V6-RichVolume",
    "flat_multiclock_ddqn": "flat_multiclock_ddqn",
    "recurrent_ppo": "recurrent_ppo",
    "lightgbm_fitted_q": "lightgbm_fitted_q",
    "no_publisher_single_agent_ddqn": "no_publisher",
    "tsmom_60m": "tsmom_60m",
    "vwap_reversion_atr1": "vwap_reversion_atr1",
    "always_flat": "always_flat",
    "session_long": "session_long",
    "session_short": "session_short",
    "shavandi_khedmati_reconstruction": "BasePaper-Described-GC-FromScratch-v1",
}

POLICY_ORDER: tuple[str, ...] = (
    "cvp_hddqn",
    "flat_multiclock_ddqn",
    "recurrent_ppo",
    "lightgbm_fitted_q",
    "no_publisher_single_agent_ddqn",
    "tsmom_60m",
    "vwap_reversion_atr1",
    "always_flat",
    "session_long",
    "session_short",
    "shavandi_khedmati_reconstruction",
)

METRIC_FIELDS: tuple[str, ...] = (
    "sessions",
    "net_pnl_cents",
    "net_pnl_usd",
    "mean_daily_pnl_usd",
    "annualized_daily_sharpe",
    "daily_profit_factor",
    "positive_day_fraction",
    "maximum_drawdown_usd",
    "completed_round_trips",
)

METRICS_COLUMNS: tuple[str, ...] = (
    "policy_id",
    "artifact_arm_id",
    "public_name",
    "policy_role",
    "period",
    "classification",
    "source_original_classification",
    "scenario",
    "status",
    *METRIC_FIELDS,
    "filled_sides",
    "explicit_commissions_usd",
    "execution_rejections",
    "technical_failure_reason",
    "technical_failure_diagnostic",
    "source_report_path",
    "source_report_sha256",
)

DAILY_COLUMNS: tuple[str, ...] = (
    "policy_id",
    "artifact_arm_id",
    "public_name",
    "policy_role",
    "period",
    "classification",
    "source_original_classification",
    "scenario",
    "session_date",
    "daily_pnl_cents",
    "daily_pnl_usd",
    "cumulative_pnl_usd",
    "underwater_drawdown_usd",
)

EFFECT_COLUMNS: tuple[str, ...] = (
    "reference_policy_id",
    "baseline_policy_id",
    "baseline_artifact_arm_id",
    "baseline_public_name",
    "period",
    "classification",
    "scenario",
    "status",
    "sessions",
    "total_difference_usd",
    "mean_daily_difference_usd",
    "one_sided_90pct_lower_usd_per_session",
    "block_length_sessions",
    "bootstrap_repetitions",
    "bootstrap_seed",
    "technical_failure_reason",
    "technical_failure_diagnostic",
)


def _scalar(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        unknown = set(row) - set(columns)
        if unknown:
            raise ValueError(f"CSV row contains undeclared fields: {sorted(unknown)}")
        writer.writerow({name: _scalar(row.get(name)) for name in columns})
    return stream.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _report_path(manifest_path: Path, record: Mapping[str, object]) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute():
        raise ValueError("evaluation report path must be relative")
    candidate = (manifest_path.parent / relative).resolve()
    try:
        candidate.relative_to(manifest_path.parent.resolve())
    except ValueError as error:
        raise ValueError("evaluation report path escapes evaluation root") from error
    if not candidate.is_file() or file_sha256(candidate) != record.get("sha256"):
        raise ValueError("evaluation report hash differs")
    return candidate


def _load_new_reports(evaluation_manifest_path: Path) -> list[dict[str, object]]:
    from .replay import (
        verify_complete_evaluation_manifest,
        verify_replay_report,
        verify_technical_failure_report,
    )

    manifest_path = Path(evaluation_manifest_path).resolve()
    manifest = verify_complete_evaluation_manifest(manifest_path)
    records = manifest.get("reports")
    if not isinstance(records, list) or len(records) != 64:
        raise ValueError("baseline evaluation registry is not the exact 64-cell suite")
    output: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("evaluation report record is malformed")
        arm = str(record.get("arm"))
        period = str(record.get("period"))
        scenario = str(record.get("scenario"))
        path = _report_path(manifest_path, record)
        status = str(record.get("status"))
        if status == "COMPLETE":
            report = verify_replay_report(
                path,
                expected_arm=arm,
                expected_period=period,
                expected_scenario=scenario,
            )
        elif status == "TECHNICAL_FAILURE":
            report = verify_technical_failure_report(
                path,
                expected_arm=arm,
                expected_period=period,
                expected_scenario=scenario,
            )
        else:
            raise ValueError("evaluation status is neither complete nor a retained failure")
        output.append({
            "policy_id": arm,
            "artifact_arm_id": ARTIFACT_ARM_IDS[arm],
            "public_name": PUBLIC_NAMES[arm],
            "policy_role": "new_baseline",
            "period": period,
            "scenario": scenario,
            "status": status,
            "classification": PERIOD_CLASSIFICATION[period],
            "source_original_classification": PERIOD_CLASSIFICATION[period],
            "report": report,
            "source_report_path": project_relative(path),
            "source_report_sha256": file_sha256(path),
        })
    expected = {
        (arm, period, scenario)
        for arm in BASELINE_ARMS for period in PERIODS for scenario in SCENARIOS
    }
    observed = {
        (str(item["policy_id"]), str(item["period"]), str(item["scenario"]))
        for item in output
    }
    if observed != expected:
        raise ValueError("evaluation manifest omits or duplicates a registered cell")
    return output


def _reference_row(value: ReferenceReport) -> dict[str, object]:
    return {
        "policy_id": value.policy_id,
        "artifact_arm_id": ARTIFACT_ARM_IDS[value.policy_id],
        "public_name": value.public_name,
        "policy_role": "frozen_reference",
        "period": value.period,
        "scenario": value.scenario,
        "status": "COMPLETE",
        "classification": value.classification_for_baseline_suite,
        "source_original_classification": value.original_classification,
        "report": {
            "daily_pnl_cents": value.daily_pnl_cents,
            "metrics": value.metrics,
            "filled_sides": value.filled_sides,
            "rejected_commands": value.rejected_commands,
            "commissions_cents": value.commissions_cents,
        },
        "source_report_path": value.source_report_path,
        "source_report_sha256": value.source_report_sha256,
        "source_manifest_path": value.source_manifest_path,
        "source_manifest_sha256": value.source_manifest_sha256,
        "original_classification": value.original_classification,
    }


def _normalise_metrics(item: Mapping[str, object]) -> dict[str, object]:
    report = item.get("report")
    if not isinstance(report, Mapping):
        raise ValueError("report payload is malformed")
    status = str(item["status"])
    row: dict[str, object] = {
        "policy_id": item["policy_id"],
        "artifact_arm_id": item["artifact_arm_id"],
        "public_name": item["public_name"],
        "policy_role": item["policy_role"],
        "period": item["period"],
        "classification": item["classification"],
        "source_original_classification": item["source_original_classification"],
        "scenario": item["scenario"],
        "status": status,
        "technical_failure_reason": report.get("technical_failure_reason", ""),
        "technical_failure_diagnostic": report.get("diagnostic", ""),
        "source_report_path": item["source_report_path"],
        "source_report_sha256": item["source_report_sha256"],
    }
    if status != "COMPLETE":
        for name in METRIC_FIELDS:
            row[name] = ""
        row.update({
            "filled_sides": "",
            "explicit_commissions_usd": "",
            "execution_rejections": "",
        })
        return row
    metrics = report.get("metrics")
    daily = report.get("daily_pnl_cents")
    if not isinstance(metrics, Mapping) or not isinstance(daily, Mapping):
        raise ValueError("complete report lacks metrics or daily P&L")
    completed = int(metrics.get("completed_round_trips", -1))
    recomputed = daily_metrics(
        {str(day): int(value) for day, value in daily.items()},
        completed_round_trips=completed,
    )
    for name in METRIC_FIELDS:
        source_name = "profit_factor" if name == "daily_profit_factor" else name
        expected = recomputed.get(source_name)
        observed_name = name if name in metrics else source_name
        if observed_name in metrics:
            observed = metrics.get(observed_name)
            if observed is None or expected is None:
                if observed is not None or expected is not None:
                    raise ValueError(f"metric {name} differs from integer daily cash")
            elif abs(float(observed) - float(expected)) > 1e-9:
                raise ValueError(f"metric {name} differs from integer daily cash")
        row[name] = expected
    filled = int(report.get("filled_sides", report.get("fills", -1)))
    rejected = int(report.get("rejected_commands", -1))
    if filled < 0 or rejected < 0 or filled != 2 * completed:
        raise ValueError("activity fields do not reconcile")
    commissions = report.get("commissions_cents")
    if commissions is None:
        economics = report.get("economics")
        if not isinstance(economics, Mapping):
            raise ValueError("complete report lacks economics")
        commissions = filled * int(economics.get("fee_cents_per_side", -1))
    row.update({
        "filled_sides": filled,
        "explicit_commissions_usd": int(commissions) / 100.0,
        "execution_rejections": rejected,
    })
    return row


def _daily_rows(item: Mapping[str, object]) -> list[dict[str, object]]:
    if item["status"] != "COMPLETE":
        return []
    report = item["report"]
    assert isinstance(report, Mapping)
    daily = report.get("daily_pnl_cents")
    if not isinstance(daily, Mapping) or not daily:
        raise ValueError("complete report lacks daily P&L")
    cumulative = 0.0
    peak = 0.0
    rows: list[dict[str, object]] = []
    for day in sorted(str(value) for value in daily):
        cents = int(daily[day])
        pnl = cents / 100.0
        cumulative += pnl
        peak = max(peak, cumulative)
        rows.append({
            "policy_id": item["policy_id"],
            "artifact_arm_id": item["artifact_arm_id"],
            "public_name": item["public_name"],
            "policy_role": item["policy_role"],
            "period": item["period"],
            "classification": item["classification"],
            "source_original_classification": item["source_original_classification"],
            "scenario": item["scenario"],
            "session_date": day,
            "daily_pnl_cents": cents,
            "daily_pnl_usd": pnl,
            "cumulative_pnl_usd": cumulative,
            "underwater_drawdown_usd": peak - cumulative,
        })
    return rows


def _sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        PERIODS.index(str(row["period"])),
        SCENARIOS.index(str(row["scenario"])),
        POLICY_ORDER.index(str(row["policy_id"])),
    )


def _expected_products(
    evaluation_manifest_path: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    items = _load_new_reports(evaluation_manifest_path)
    references = list(load_all_reference_reports())
    items.extend(_reference_row(value) for value in references)
    items.sort(key=_sort_key)
    metrics_rows = [_normalise_metrics(item) for item in items]
    daily_rows = [row for item in items for row in _daily_rows(item)]

    lookup = {
        (str(item["policy_id"]), str(item["period"]), str(item["scenario"])): item
        for item in items
    }
    effects: list[dict[str, object]] = []
    draws: list[dict[str, object]] = []
    candidates = list(BASELINE_ARMS) + ["no_publisher_single_agent_ddqn"]
    for period in PERIODS:
        if period == "may_2026":
            candidates_for_period = candidates + ["shavandi_khedmati_reconstruction"]
        else:
            candidates_for_period = candidates
        for scenario in SCENARIOS:
            reference = lookup[("cvp_hddqn", period, scenario)]
            reference_report = reference["report"]
            assert isinstance(reference_report, Mapping)
            reference_daily = reference_report.get("daily_pnl_cents")
            assert isinstance(reference_daily, Mapping)
            for policy_id in candidates_for_period:
                baseline = lookup[(policy_id, period, scenario)]
                baseline_report = baseline["report"]
                assert isinstance(baseline_report, Mapping)
                base = {
                    "reference_policy_id": "cvp_hddqn",
                    "baseline_policy_id": policy_id,
                    "baseline_artifact_arm_id": ARTIFACT_ARM_IDS[policy_id],
                    "baseline_public_name": PUBLIC_NAMES[policy_id],
                    "period": period,
                    "classification": PERIOD_CLASSIFICATION[period],
                    "scenario": scenario,
                }
                if baseline["status"] != "COMPLETE":
                    effects.append({
                        **base,
                        "status": "TECHNICAL_FAILURE",
                        "sessions": "",
                        "total_difference_usd": "",
                        "mean_daily_difference_usd": "",
                        "one_sided_90pct_lower_usd_per_session": "",
                        "block_length_sessions": BOOTSTRAP_BLOCK,
                        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "technical_failure_reason": baseline_report.get(
                            "technical_failure_reason", "UNSPECIFIED"
                        ),
                        "technical_failure_diagnostic": baseline_report.get(
                            "diagnostic", "UNSPECIFIED"
                        ),
                    })
                    continue
                baseline_daily = baseline_report.get("daily_pnl_cents")
                if not isinstance(baseline_daily, Mapping):
                    raise ValueError("complete baseline lacks daily cash")
                effect = paired_effect(
                    {str(k): int(v) for k, v in reference_daily.items()},
                    {str(k): int(v) for k, v in baseline_daily.items()},
                )
                effects.append({
                    **base,
                    "status": "COMPLETE",
                    **{name: effect[name] for name in (
                        "sessions", "total_difference_usd",
                        "mean_daily_difference_usd",
                        "one_sided_90pct_lower_usd_per_session",
                        "block_length_sessions", "bootstrap_repetitions",
                        "bootstrap_seed",
                    )},
                    "technical_failure_reason": "",
                    "technical_failure_diagnostic": "",
                })
                days = sorted(str(day) for day in reference_daily)
                difference = np.asarray([
                    (int(reference_daily[day]) - int(baseline_daily[day])) / 100.0
                    for day in days
                ])
                distribution = moving_block_means(difference)
                for replicate, value in enumerate(distribution):
                    draws.append({
                        "reference_policy_id": "cvp_hddqn",
                        "baseline_policy_id": policy_id,
                        "period": period,
                        "scenario": scenario,
                        "replicate": replicate,
                        "mean_difference_usd_per_session": float(value),
                    })
    reference_bindings = sorted(
        {
            (
                value.source_manifest_path,
                value.source_manifest_sha256,
                value.source_report_path,
                value.source_report_sha256,
            )
            for value in references
        }
    )
    binding_rows = [
        {
            "manifest_path": row[0],
            "manifest_sha256": row[1],
            "report_path": row[2],
            "report_sha256": row[3],
        }
        for row in reference_bindings
    ]
    return metrics_rows, daily_rows, effects, draws, binding_rows


def _parquet_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    import io

    schema = pa.schema([
        ("reference_policy_id", pa.string()),
        ("baseline_policy_id", pa.string()),
        ("period", pa.string()),
        ("scenario", pa.string()),
        ("replicate", pa.int32()),
        ("mean_difference_usd_per_session", pa.float64()),
    ])
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="zstd", version="2.6")
    return sink.getvalue()


def _record(path: Path, *, rows: int) -> dict[str, object]:
    return {
        "path": project_relative(path),
        "path_scope": "project_root",
        "sha256": file_sha256(path),
        "rows": int(rows),
    }


def build_evidence_bundle(
    output_root: Path | None = None,
    report_root: Path | None = None,
) -> Path:
    """Build immutable publication-independent evidence from verified reports."""

    reject_forbidden_text(
        output_root if output_root is not None else CANONICAL_OUTPUT_ROOT,
        report_root if report_root is not None else DEFAULT_REPORT_ROOT,
    )
    destination_root = validate_workspace_output(
        DEFAULT_REPORT_ROOT if report_root is None else report_root,
    )
    root = validate_workspace_output(
        CANONICAL_OUTPUT_ROOT if output_root is None else output_root,
    )
    workspace = workspace_root().resolve()
    paper = (workspace / "paper").resolve()
    try:
        destination_root.relative_to(workspace)
    except ValueError as error:
        raise PermissionError("baseline evidence must remain inside the workspace") from error
    try:
        destination_root.relative_to(paper)
    except ValueError:
        pass
    else:
        raise PermissionError("Pilot 10 evidence cannot be written below paper/")
    source = destination_root / "source_data"
    manifest_path = source / "manifest.json"
    if manifest_path.exists():
        verify_evidence_bundle(manifest_path)
        return manifest_path
    evaluation_manifest = root / "evaluation" / "manifest.json"
    metrics_rows, daily_rows, effects, draws, references = _expected_products(
        evaluation_manifest
    )
    metrics_path = source / "metrics.csv"
    daily_path = source / "daily_pnl.csv"
    effects_path = source / "paired_effects.csv"
    draws_path = source / "bootstrap_draws.parquet"
    _atomic_bytes(metrics_path, _csv_bytes(METRICS_COLUMNS, metrics_rows))
    _atomic_bytes(daily_path, _csv_bytes(DAILY_COLUMNS, daily_rows))
    _atomic_bytes(effects_path, _csv_bytes(EFFECT_COLUMNS, effects))
    _atomic_bytes(draws_path, _parquet_bytes(draws))
    value: dict[str, object] = {
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
                "cvp_hddqn",
                "no_publisher_single_agent_ddqn",
                "shavandi_khedmati_reconstruction",
            ],
            "excluded": {
                "restricted_representation": (
                    "Existing representation ablation; deliberately kept separate "
                    "from this baseline report."
                )
            },
        },
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "evaluation_manifest": {
            "path": project_relative(evaluation_manifest),
            "path_scope": "project_root",
            "sha256": file_sha256(evaluation_manifest),
        },
        "reference_sources": references,
        "products": {
            "metrics": _record(metrics_path, rows=len(metrics_rows)),
            "daily_pnl": _record(daily_path, rows=len(daily_rows)),
            "paired_effects": _record(effects_path, rows=len(effects)),
            "bootstrap_draws": _record(draws_path, rows=len(draws)),
        },
        "content_sha256": canonical_json_sha256({
            "metrics": metrics_rows,
            "daily": daily_rows,
            "effects": effects,
            "reference_sources": references,
        }),
    }
    atomic_json(manifest_path, value)
    verify_evidence_bundle(manifest_path)
    return manifest_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_evidence_bundle(path: Path) -> dict[str, object]:
    """Independently reconstruct every evidence row and hash binding."""

    reject_forbidden_text(path)
    manifest_path = Path(path).resolve()
    workspace = workspace_root().resolve()
    paper = (workspace / "paper").resolve()
    try:
        manifest_path.relative_to(workspace)
    except ValueError as error:
        raise ValueError("evidence manifest is outside the workspace") from error
    try:
        manifest_path.relative_to(paper)
    except ValueError:
        pass
    else:
        raise ValueError("Pilot 10 evidence cannot reside below paper/")
    if manifest_path.name != "manifest.json" or manifest_path.parent.name != "source_data":
        raise ValueError("evidence manifest location differs")
    value = read_json_object(manifest_path)
    if (
        value.get("artifact_type") != EVIDENCE_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("complete") is not True
        or value.get("august_accessed") is not False
        or value.get("all_registered_results_retained") is not True
        or value.get("all_new_baseline_cells_retained") is not True
        or value.get("registered_result_scope")
        != "eight new baseline arms x two periods x four scenarios"
        or value.get("period_classification") != PERIOD_CLASSIFICATION
        or value.get("protocol") != protocol_binding()
    ):
        raise ValueError("baseline evidence identity differs")
    if value.get("selected_reference_subset") != {
        "included": [
            "cvp_hddqn",
            "no_publisher_single_agent_ddqn",
            "shavandi_khedmati_reconstruction",
        ],
        "excluded": {
            "restricted_representation": (
                "Existing representation ablation; deliberately kept separate "
                "from this baseline report."
            )
        },
    }:
        raise ValueError("selected reference subset differs")
    verify_implementation_binding(value.get("implementation"))
    evaluation = value.get("evaluation_manifest")
    if not isinstance(evaluation, Mapping):
        raise ValueError("evidence lacks evaluation binding")
    if evaluation.get("path_scope") != "project_root":
        raise ValueError("evidence evaluation path scope differs")
    evaluation_path = resolve_project_path(evaluation.get("path"))
    if not evaluation_path.is_file() or file_sha256(evaluation_path) != evaluation.get("sha256"):
        raise ValueError("evidence evaluation manifest changed")
    expected_metrics, expected_daily, expected_effects, expected_draws, references = (
        _expected_products(evaluation_path)
    )
    if value.get("reference_sources") != references:
        raise ValueError("reference source registry differs")
    products = value.get("products")
    if not isinstance(products, Mapping):
        raise ValueError("evidence product registry is absent")
    expected_payloads = {
        "metrics": _csv_bytes(METRICS_COLUMNS, expected_metrics),
        "daily_pnl": _csv_bytes(DAILY_COLUMNS, expected_daily),
        "paired_effects": _csv_bytes(EFFECT_COLUMNS, expected_effects),
        "bootstrap_draws": _parquet_bytes(expected_draws),
    }
    expected_rows = {
        "metrics": len(expected_metrics),
        "daily_pnl": len(expected_daily),
        "paired_effects": len(expected_effects),
        "bootstrap_draws": len(expected_draws),
    }
    for name, payload in expected_payloads.items():
        record = products.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"evidence product {name} is absent")
        if record.get("path_scope") != "project_root":
            raise ValueError(f"evidence product {name} path scope differs")
        product_path = resolve_project_path(record.get("path"))
        expected_name = {
            "metrics": "metrics.csv",
            "daily_pnl": "daily_pnl.csv",
            "paired_effects": "paired_effects.csv",
            "bootstrap_draws": "bootstrap_draws.parquet",
        }[name]
        if (
            not product_path.is_file()
            or product_path != manifest_path.parent / expected_name
            or file_sha256(product_path) != record.get("sha256")
            or product_path.read_bytes() != payload
            or int(record.get("rows", -1)) != expected_rows[name]
        ):
            raise ValueError(f"evidence product {name} differs")
    content = canonical_json_sha256({
        "metrics": expected_metrics,
        "daily": expected_daily,
        "effects": expected_effects,
        "reference_sources": references,
    })
    if value.get("content_sha256") != content:
        raise ValueError("evidence content digest differs")
    return value


__all__ = [
    "DAILY_COLUMNS", "DEFAULT_REPORT_ROOT", "EFFECT_COLUMNS",
    "EVIDENCE_ARTIFACT", "METRICS_COLUMNS", "POLICY_ORDER", "PUBLIC_NAMES",
    "build_evidence_bundle", "verify_evidence_bundle",
]
