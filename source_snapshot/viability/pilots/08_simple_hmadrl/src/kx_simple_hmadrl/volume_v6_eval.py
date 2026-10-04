"""Deterministic V6 reporting, paired tests, and fail-closed split gates.

This module deliberately contains no model-selection logic.  It evaluates one
frozen candidate and refuses to turn a smoke report or incomplete comparator
set into permission to read the registered test period.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .environment import ExecutionEconomics


SEEDS = (1103, 2207, 3301, 4409, 5501)
VALIDATION_START = "2026-05-01"
VALIDATION_END = "2026-05-31"
TEST_START = "2026-06-01"
TEST_END = "2026-08-31"
TEST_COMPLETE_ON = date(2026, 9, 1)
REQUIRED_ARMS = (
    "V6-RichVolume",
    "V6-BaseOHLCV",
    "BasePaper-XAU-Pretrained-GC",
    "BaseStrong-XAU-Pretrained-GC-v6",
)
STRESS_SCENARIOS = (
    "double_commission", "20ms_route_latency",
    "one_adverse_tick_per_filled_side",
)


def _current_date() -> date:
    """Single private clock boundary; tests monkeypatch this, callers cannot."""

    return date.today()


def expected_economics(scenario: str) -> dict[str, object]:
    try:
        economics = {
            "base": ExecutionEconomics.from_basepaper(),
            "double_commission": ExecutionEconomics.from_basepaper(fee_multiplier=2),
            "20ms_route_latency": ExecutionEconomics.from_basepaper(route_latency_ms=20),
            "one_adverse_tick_per_filled_side": ExecutionEconomics.from_basepaper(
                adverse_ticks=1,
            ),
        }[scenario]
    except KeyError as error:
        raise ValueError("unknown registered execution scenario") from error
    return asdict(economics)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, object], *, immutable: bool = True) -> Path:
    if immutable and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable V6 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".part", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, sort_keys=True, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _protocol_path() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts" / "VOLUME_V6_PROTOCOL.json"


def frozen_protocol() -> tuple[dict[str, object], str]:
    path = _protocol_path()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("strategy_version") != "two_level_volume_publisher_v6":
        raise ValueError("V6 protocol identity differs")
    receipt = json.loads((path.parent / "VOLUME_V6_FREEZE_RECEIPT.json").read_text())
    digest = canonical_json_sha256(value)
    if receipt.get("protocol_sha256") != digest or receipt.get("validator_result") != "PASS":
        raise ValueError("V6 protocol no longer matches its freeze receipt")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("V6 protocol has no bound artifact registry")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"V6 protocol artifact {name!r} is malformed")
        artifact = path.parent / str(record.get("path", ""))
        if not artifact.is_file() or file_sha256(artifact) != record.get("sha256"):
            raise ValueError(f"V6 protocol artifact {name!r} changed after freeze")
    code_record = artifacts.get("implementation_code_manifest")
    if code_record is not None:
        assert isinstance(code_record, dict)
        code_path = path.parent / str(code_record["path"])
        code_manifest = json.loads(code_path.read_text(encoding="utf-8"))
        if (
            not isinstance(code_manifest, dict)
            or code_manifest.get("artifact_type") != "VOLUME_V6_CODE_MANIFEST"
            or code_manifest.get("strategy_version") != "two_level_volume_publisher_v6"
        ):
            raise ValueError("V6 implementation code manifest identity differs")
        sections = ("files", "audited_dependencies")
        declared = 0
        workspace = path.parents[3]
        for section in sections:
            products = code_manifest.get(section)
            if not isinstance(products, dict):
                raise ValueError(f"V6 code manifest lacks {section}")
            declared += len(products)
            for relative, expected in products.items():
                product = (workspace / str(relative)).resolve()
                try:
                    product.relative_to(workspace.resolve())
                except ValueError as error:
                    raise ValueError("V6 code manifest path escapes the workspace") from error
                if not product.is_file() or file_sha256(product) != expected:
                    raise ValueError(f"V6 implementation file changed: {relative}")
        if code_manifest.get("file_count") != declared:
            raise ValueError("V6 implementation code manifest count differs")
    return value, digest


def classify_dates(days: Sequence[str], *, phase: str) -> str:
    """Validate a requested period without touching any market-data path."""

    if not days or len(days) != len(set(days)) or list(days) != sorted(days):
        raise ValueError("session dates must be nonempty, unique, and sorted")
    if phase == "smoke":
        if any(not "2025-01-01" <= day <= "2025-08-31" for day in days):
            raise PermissionError("implementation smoke is limited to contaminated Jan-Aug 2025")
        return "DEVELOPMENT"
    if phase == "train":
        if any(not "2025-01-01" <= day <= "2026-04-30" for day in days):
            raise PermissionError("training request leaves the frozen Jan-2025/Apr-2026 range")
        return "DEVELOPMENT"
    if phase == "validation":
        if any(not VALIDATION_START <= day <= VALIDATION_END for day in days):
            raise PermissionError("validation request is not wholly inside May 2026")
        return "DEVELOPMENT"
    if phase == "test":
        if any(not TEST_START <= day <= TEST_END for day in days):
            raise PermissionError("test request is not wholly inside Jun-Aug 2026")
        return "FRESH_OOS"
    raise ValueError("phase must be smoke, train, validation, or test")


def flat_on_vote_tie(actions: Sequence[int]) -> int:
    values = np.asarray(actions, dtype=np.int8)
    if len(values) != len(SEEDS) or not np.isin(values, (-1, 0, 1)).all():
        raise ValueError("ensemble vote requires five legal seed actions")
    counts = np.asarray([(values == action).sum() for action in (-1, 0, 1)])
    winners = np.flatnonzero(counts == counts.max())
    return 0 if len(winners) != 1 else (-1, 0, 1)[int(winners[0])]


def daily_metrics(
    daily_pnl_cents: Mapping[str, int], completed_round_trips: int,
) -> dict[str, float | int | None]:
    if not daily_pnl_cents:
        raise ValueError("daily metrics require at least one session")
    days = sorted(daily_pnl_cents)
    values = np.asarray([daily_pnl_cents[day] for day in days], dtype=np.float64) / 100.0
    standard = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sharpe = 0.0 if standard < 1e-12 else float(np.sqrt(252.0) * values.mean() / standard)
    equity = np.concatenate(([0.0], np.cumsum(values)))
    drawdown = np.maximum.accumulate(equity) - equity
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return {
        "sessions": len(values),
        "net_pnl_usd": float(values.sum()),
        "mean_daily_pnl_usd": float(values.mean()),
        "annualized_daily_sharpe": sharpe,
        "maximum_drawdown_usd": float(drawdown.max(initial=0.0)),
        "completed_round_trips": int(completed_round_trips),
        "positive_day_fraction": float((values > 0).mean()),
        "profit_factor": gains / losses if losses > 0 else None,
    }


def block_bootstrap_lower(
    values: Sequence[float] | np.ndarray, *, block: int = 5, confidence: float = 0.90,
    repetitions: int = 10_000, seed: int = 1103,
) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or len(sample) < block:
        raise ValueError("registered bootstrap needs at least five paired sessions")
    if block < 1 or not 0.0 < confidence < 1.0 or repetitions < 1:
        raise ValueError("invalid block-bootstrap configuration")
    generator = np.random.default_rng(seed)
    starts = np.arange(len(sample) - block + 1)
    blocks_needed = int(np.ceil(len(sample) / block))
    means = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        chosen = generator.choice(starts, blocks_needed, replace=True)
        draw = np.concatenate([sample[start:start + block] for start in chosen])[:len(sample)]
        means[repetition] = draw.mean()
    return float(np.quantile(means, 1.0 - confidence))


def paired_improvement(
    primary: Mapping[str, object], comparator: Mapping[str, object],
) -> dict[str, float]:
    first = primary.get("daily_pnl_cents")
    second = comparator.get("daily_pnl_cents")
    if not isinstance(first, dict) or not isinstance(second, dict) or sorted(first) != sorted(second):
        raise ValueError("paired reports must contain identical session dates")
    difference = np.asarray(
        [int(first[day]) - int(second[day]) for day in sorted(first)], dtype=np.float64,
    ) / 100.0
    return {
        "total_usd": float(difference.sum()),
        "mean_daily_usd": float(difference.mean()),
        "five_session_block_bootstrap_90pct_lower_usd": block_bootstrap_lower(difference),
    }


def _validate_report_set(reports: Mapping[str, Mapping[str, object]], *, period: str) -> None:
    if set(reports) != set(REQUIRED_ARMS):
        raise ValueError(f"gate requires exactly {REQUIRED_ARMS}")
    primary = reports["V6-RichVolume"]
    identity = (
        primary.get("protocol_sha256"), primary.get("evaluation_dates"),
        primary.get("classification"), primary.get("economics"),
        primary.get("ensemble_seeds"), primary.get("scenario", "base"),
    )
    for arm, report in reports.items():
        if report.get("arm") != arm:
            raise ValueError(f"report is mislabeled for arm {arm}")
        candidate = (
            report.get("protocol_sha256"), report.get("evaluation_dates"),
            report.get("classification"), report.get("economics"),
            report.get("ensemble_seeds"), report.get("scenario", "base"),
        )
        if candidate != identity:
            raise ValueError(f"paired report identity differs for {arm}")
        if report.get("registered_evaluation") is not True or report.get("smoke_only") is not False:
            raise ValueError("registered gate refuses smoke or partial reports")
        scenario = str(report.get("scenario", "base"))
        if report.get("economics") != expected_economics(scenario):
            raise ValueError(f"report economics differ from registered scenario {scenario}")
    expected_class = "DEVELOPMENT" if period == "validation" else "FRESH_OOS"
    if primary.get("classification") != expected_class:
        raise ValueError("report classification differs from gate period")
    dates = primary.get("evaluation_dates")
    if not isinstance(dates, list):
        raise ValueError("evaluation date receipt is absent")
    classify_dates([str(value) for value in dates], phase=period)


def apply_registered_gates(
    nominal: Mapping[str, Mapping[str, object]],
    stresses: Mapping[str, Mapping[str, Mapping[str, object]]], *,
    period: str,
) -> dict[str, object]:
    """Apply every frozen absolute, paired, risk, activity, and stress gate."""

    if period not in {"validation", "test"}:
        raise ValueError("registered gates apply only to validation or test")
    _validate_report_set(nominal, period=period)
    if set(stresses) != set(STRESS_SCENARIOS):
        raise ValueError("all three frozen stress scenarios are required")
    for scenario, reports in stresses.items():
        _validate_report_set(reports, period=period)
        if any(report.get("scenario") != scenario for report in reports.values()):
            raise ValueError(f"stress reports are mislabeled for {scenario}")
        if any(
            report.get("evaluation_dates") != nominal[arm].get("evaluation_dates")
            or report.get("checkpoint_set") != nominal[arm].get("checkpoint_set")
            for arm, report in reports.items()
        ):
            raise ValueError(f"stress {scenario} changes dates or checkpoints")

    primary = nominal["V6-RichVolume"]
    metrics = primary.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("primary metrics are absent")
    comparisons = {
        arm: paired_improvement(primary, nominal[arm]) for arm in REQUIRED_ARMS[1:]
    }
    external = REQUIRED_ARMS[2:]
    external_metrics = []
    for arm in external:
        value = nominal[arm].get("metrics")
        if not isinstance(value, dict):
            raise ValueError(f"metrics are absent for {arm}")
        external_metrics.append(value)

    checks: dict[str, bool] = {
        "positive_net_pnl": float(metrics["net_pnl_usd"]) > 0.0,
        "positive_sharpe": float(metrics["annualized_daily_sharpe"]) > 0.0,
        "minimum_30_round_trips": int(metrics["completed_round_trips"]) >= 30,
        "paired_lb_vs_base_ohlcv_positive": comparisons["V6-BaseOHLCV"]["five_session_block_bootstrap_90pct_lower_usd"] > 0.0,
        "sharpe_exceeds_basepaper": float(metrics["annualized_daily_sharpe"]) > float(external_metrics[0]["annualized_daily_sharpe"]),
        "sharpe_exceeds_basestrong": float(metrics["annualized_daily_sharpe"]) > float(external_metrics[1]["annualized_daily_sharpe"]),
        "paired_lb_vs_basepaper_positive": comparisons[external[0]]["five_session_block_bootstrap_90pct_lower_usd"] > 0.0,
        "paired_lb_vs_basestrong_positive": comparisons[external[1]]["five_session_block_bootstrap_90pct_lower_usd"] > 0.0,
        "activity_at_least_half_active_external": int(metrics["completed_round_trips"]) >= 0.5 * max(int(value["completed_round_trips"]) for value in external_metrics),
        "drawdown_within_10pct_of_better_external": float(metrics["maximum_drawdown_usd"]) <= 1.10 * min(float(value["maximum_drawdown_usd"]) for value in external_metrics),
    }
    stress_deltas: dict[str, dict[str, float]] = {}
    for scenario, reports in stresses.items():
        rich_net = float(reports["V6-RichVolume"]["metrics"]["net_pnl_usd"])  # type: ignore[index]
        stress_deltas[scenario] = {}
        for arm in REQUIRED_ARMS[1:]:
            delta = rich_net - float(reports[arm]["metrics"]["net_pnl_usd"])  # type: ignore[index]
            stress_deltas[scenario][arm] = delta
            checks[f"{scenario}_nonnegative_vs_{arm}"] = delta >= 0.0

    failures = sorted(name for name, passed in checks.items() if not passed)
    _, protocol_digest = frozen_protocol()
    return {
        "schema_version": 1,
        "artifact_type": "V6_REGISTERED_GATE",
        "protocol_sha256": protocol_digest,
        "period": period,
        "classification": "DEVELOPMENT" if period == "validation" else "FRESH_OOS",
        "verdict": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failed_gates": failures,
        "paired_improvements": comparisons,
        "stress_net_pnl_differences_usd": stress_deltas,
        "evaluation_dates": primary["evaluation_dates"],
        "registered_evaluation": True,
        "smoke_only": False,
    }


@dataclass(frozen=True)
class UnlockReceipt:
    protocol_sha256: str
    validation_evaluation_manifest_sha256: str
    validation_gate_sha256: str
    complete_training_manifest_sha256: str
    authorized_at_utc: str
    test_start: str = TEST_START
    test_end: str = TEST_END


def _resolve_bound_path(anchor: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (anchor / path).resolve()


def _validation_unlock_chain(
    validation_evaluation_manifest: Path, protocol_digest: str,
) -> tuple[dict[str, object], Path, Path]:
    """Verify the immutable outer validation -> gate -> training hash chain."""

    manifest_path = Path(validation_evaluation_manifest).resolve()
    # The outer JSON chain is necessary but not sufficient: without reopening
    # its phase dataset, all checkpoint children, all reports, and the
    # recomputed gate, a caller could fabricate a self-consistent profitable
    # chain.  Import lazily to avoid the eval/orchestration module cycle.
    try:
        from .volume_v6_orchestration import verify_evaluation_manifest
        deeply_verified = verify_evaluation_manifest(manifest_path)
    except (OSError, ValueError, PermissionError, KeyError, TypeError) as error:
        raise PermissionError(
            "validation evaluation failed deep dataset/checkpoint/gate verification",
        ) from error
    if deeply_verified.get("phase") != "validation":
        raise PermissionError("only a deeply verified May evaluation can unlock test")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("artifact_type") != "V6_COMPLETE_REGISTERED_EVALUATION"
        or value.get("protocol_sha256") != protocol_digest
        or value.get("complete") is not True
        or value.get("phase") != "validation"
        or value.get("classification") != "DEVELOPMENT"
        or value.get("verdict") != "PASS"
        or value.get("failed_gates") != []
        or value.get("registered_evaluation") is not True
        or value.get("smoke_only") is not False
        or value.get("arms") != list(REQUIRED_ARMS)
        or value.get("scenarios") != ["base", *STRESS_SCENARIOS]
        or value.get("report_count") != len(REQUIRED_ARMS) * (1 + len(STRESS_SCENARIOS))
    ):
        raise PermissionError("only a complete registered May evaluation PASS can unlock test")
    gate_record = value.get("gate")
    training_record = value.get("training_manifest")
    dataset_record = value.get("evaluation_dataset_manifest")
    if (
        not isinstance(gate_record, dict)
        or not isinstance(training_record, dict)
        or not isinstance(dataset_record, dict)
    ):
        raise PermissionError("validation evaluation lacks its gate/training hash chain")
    gate_path = _resolve_bound_path(manifest_path.parent, gate_record.get("path"))
    training_path = _resolve_bound_path(manifest_path.parent, training_record.get("path"))
    dataset_path = _resolve_bound_path(manifest_path.parent, dataset_record.get("path"))
    if (
        not gate_path.is_file()
        or file_sha256(gate_path) != gate_record.get("sha256")
        or not training_path.is_file()
        or file_sha256(training_path) != training_record.get("sha256")
        or not dataset_path.is_file()
        or file_sha256(dataset_path) != dataset_record.get("sha256")
    ):
        raise PermissionError("validation evaluation gate/training artifact is absent or changed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gate.get("artifact_type") != "V6_REGISTERED_GATE"
        or gate.get("period") != "validation"
        or gate.get("classification") != "DEVELOPMENT"
        or gate.get("verdict") != "PASS"
        or gate.get("failed_gates") != []
        or gate.get("protocol_sha256") != protocol_digest
        or gate.get("registered_evaluation") is not True
        or gate.get("smoke_only") is not False
        or gate.get("complete_training_manifest_sha256") != training_record.get("sha256")
    ):
        raise PermissionError("validation gate differs from its complete evaluation receipt")
    reports = value.get("reports")
    if not isinstance(reports, list):
        raise PermissionError("validation evaluation report records are absent")
    identities: set[tuple[str, str]] = set()
    nominal: dict[str, dict[str, object]] = {}
    stresses: dict[str, dict[str, dict[str, object]]] = {
        scenario: {} for scenario in STRESS_SCENARIOS
    }
    for record in reports:
        if not isinstance(record, dict):
            raise PermissionError("validation report record is malformed")
        scenario = str(record.get("scenario")); arm = str(record.get("arm"))
        report_path = _resolve_bound_path(manifest_path.parent, record.get("path"))
        try:
            report_path.relative_to(manifest_path.parent)
        except ValueError as error:
            raise PermissionError("validation report escapes its evaluation root") from error
        if not report_path.is_file() or file_sha256(report_path) != record.get("sha256"):
            raise PermissionError("validation report hash differs")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise PermissionError("validation report is not a JSON object")
        metrics = report.get("metrics")
        daily = report.get("daily_pnl_cents")
        if (
            report.get("arm") != arm
            or report.get("scenario") != scenario
            or report.get("protocol_sha256") != protocol_digest
            or report.get("evaluation_dataset_manifest_sha256") != dataset_record.get("sha256")
            or report.get("complete_training_manifest_sha256") != training_record.get("sha256")
            or not isinstance(metrics, dict)
            or not isinstance(daily, dict)
        ):
            raise PermissionError("validation report identity/source binding differs")
        try:
            normalized = {str(day): int(amount) for day, amount in daily.items()}
            recomputed_metrics = daily_metrics(
                normalized, int(metrics["completed_round_trips"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PermissionError("validation report metrics are malformed") from error
        if normalized != daily or metrics != recomputed_metrics:
            raise PermissionError("validation report metrics do not recompute")
        identities.add((scenario, arm))
        if scenario == "base":
            nominal[arm] = report
        elif scenario in stresses:
            stresses[scenario][arm] = report
    expected = {
        (scenario, arm)
        for scenario in ("base", *STRESS_SCENARIOS)
        for arm in REQUIRED_ARMS
    }
    if identities != expected:
        raise PermissionError("validation evaluation lacks a registered scenario/arm report")
    try:
        recomputed_gate = apply_registered_gates(nominal, stresses, period="validation")
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("validation gate inputs do not recompute") from error
    recomputed_gate["complete_training_manifest"] = str(training_path)
    recomputed_gate["complete_training_manifest_sha256"] = training_record["sha256"]
    if gate != recomputed_gate:
        raise PermissionError("validation gate differs from full report recomputation")
    return value, gate_path, training_path


def authorize_test(
    validation_evaluation_manifest: Path, output: Path,
) -> Path:
    """Unlock test from a complete, hash-bound validation evaluation PASS."""

    protocol, protocol_digest = frozen_protocol()
    current = _current_date()
    if current < TEST_COMPLETE_ON:
        raise PermissionError("the June-August 2026 test cannot open before August is complete")
    if protocol.get("holdout", {}).get("previously_inspected") is not False:  # type: ignore[union-attr]
        raise PermissionError("protocol no longer classifies the test as unopened")
    validation_path = Path(validation_evaluation_manifest).resolve()
    _, gate_path, training_path = _validation_unlock_chain(validation_path, protocol_digest)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    receipt = {
        "schema_version": 1,
        "artifact_type": "V6_TEST_UNLOCK_RECEIPT",
        "strategy_version": "two_level_volume_publisher_v6",
        "protocol_sha256": protocol_digest,
        "validation_evaluation_manifest": str(validation_path),
        "validation_evaluation_manifest_sha256": file_sha256(validation_path),
        "validation_gate": str(gate_path),
        "validation_gate_sha256": file_sha256(gate_path),
        "complete_training_manifest": str(training_path),
        "complete_training_manifest_sha256": file_sha256(training_path),
        "authorized_at_utc": timestamp,
        "test_period": [TEST_START, TEST_END],
        "single_open_only": True,
    }
    return atomic_json(output, receipt)


def assert_test_unlocked(receipt_path: Path) -> dict[str, object]:
    _, protocol_digest = frozen_protocol()
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        value.get("artifact_type") != "V6_TEST_UNLOCK_RECEIPT"
        or value.get("protocol_sha256") != protocol_digest
        or value.get("test_period") != [TEST_START, TEST_END]
        or value.get("single_open_only") is not True
    ):
        raise PermissionError("invalid V6 test unlock receipt")
    validation_path = Path(str(value.get("validation_evaluation_manifest", "")))
    if (
        not validation_path.is_file()
        or file_sha256(validation_path) != value.get("validation_evaluation_manifest_sha256")
    ):
        raise PermissionError("validation evaluation bound by the unlock receipt changed")
    _, gate_path, training_path = _validation_unlock_chain(validation_path, protocol_digest)
    if (
        str(gate_path) != str(value.get("validation_gate"))
        or file_sha256(gate_path) != value.get("validation_gate_sha256")
        or str(training_path) != str(value.get("complete_training_manifest"))
        or file_sha256(training_path) != value.get("complete_training_manifest_sha256")
    ):
        raise PermissionError("unlock receipt no longer matches its validation hash chain")
    return value


def write_report(path: Path, report: Mapping[str, object]) -> Path:
    return atomic_json(path, report)
