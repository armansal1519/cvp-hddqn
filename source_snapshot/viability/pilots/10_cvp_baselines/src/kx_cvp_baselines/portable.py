"""Checkpoint-free structural verification for tracked Pilot 10 artifacts.

This module deliberately verifies a smaller claim than the local production
verifiers.  It authenticates immutable JSON/CSV/Parquet/PDF/Markdown files and
their schemas without resolving model, corpus, scaler, raw-session, or dataset
lineage paths.  It never replays a policy or recomputes an economic result.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import math
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import (
    canonical_json_sha256,
    file_sha256,
    read_json_object,
    resolve_project_path,
    workspace_root,
)
from .contract import (
    BASELINE_ARMS,
    DATASET_MANIFESTS,
    DETERMINISTIC_ARMS,
    LEARNER_ARMS,
    PERIOD_BOUNDS,
    PERIOD_CLASSIFICATION,
    PERIODS,
    PROTOCOL_ID,
    SCENARIOS,
    SEEDS,
    scenario_economics,
)
from .protocol import reject_forbidden_text
from .references import REFERENCE_MANIFESTS, SCENARIO_ALIASES


PORTABLE_VERIFICATION_LEVEL = "PORTABLE_STRUCTURAL"

# Kept local so importing the portable verifier cannot import neural training,
# replay, report rendering, or reference loaders.
REPLAY_ARTIFACT = "CVP_BASELINE_REPLAY_REPORT"
TECHNICAL_FAILURE_ARTIFACT = "CVP_BASELINE_TECHNICAL_FAILURE"
COMPLETE_EVALUATION_ARTIFACT = "CVP_BASELINE_COMPLETE_EVALUATION_MANIFEST"
EVIDENCE_ARTIFACT = "CVP_BASELINE_SUITE_EVIDENCE"
REPORT_ARTIFACT = "CVP_BASELINE_SUITE_STANDALONE_REPORT"

ENSEMBLE_RULE = "five_legal_seed_votes_majority_flat_on_tie_v1"
DETERMINISTIC_RULE = "single_unreplicated_deterministic_path_v1"

PORTABLE_NON_CLAIMS: tuple[str, ...] = (
    "No checkpoint, fitted-Q corpus, replay buffer, or model payload was opened.",
    "No raw market session, dataset manifest, or scaler lineage file was opened.",
    "No policy inference, route replay, reward reconciliation, or metric recomputation was performed.",
    "No causality, economic correctness, source-code identity, or prospective validity claim is made.",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

METRICS_COLUMNS: tuple[str, ...] = (
    "policy_id", "artifact_arm_id", "public_name", "policy_role", "period",
    "classification", "source_original_classification", "scenario", "status",
    "sessions", "net_pnl_cents", "net_pnl_usd", "mean_daily_pnl_usd",
    "annualized_daily_sharpe", "daily_profit_factor", "positive_day_fraction",
    "maximum_drawdown_usd", "completed_round_trips", "filled_sides",
    "explicit_commissions_usd", "execution_rejections",
    "technical_failure_reason", "technical_failure_diagnostic",
    "source_report_path", "source_report_sha256",
)
DAILY_COLUMNS: tuple[str, ...] = (
    "policy_id", "artifact_arm_id", "public_name", "policy_role", "period",
    "classification", "source_original_classification", "scenario",
    "session_date", "daily_pnl_cents", "daily_pnl_usd", "cumulative_pnl_usd",
    "underwater_drawdown_usd",
)
EFFECT_COLUMNS: tuple[str, ...] = (
    "reference_policy_id", "baseline_policy_id", "baseline_artifact_arm_id",
    "baseline_public_name", "period", "classification", "scenario", "status",
    "sessions", "total_difference_usd", "mean_daily_difference_usd",
    "one_sided_90pct_lower_usd_per_session", "block_length_sessions",
    "bootstrap_repetitions", "bootstrap_seed", "technical_failure_reason",
    "technical_failure_diagnostic",
)
BOOTSTRAP_COLUMNS: tuple[str, ...] = (
    "reference_policy_id", "baseline_policy_id", "period", "scenario",
    "replicate", "mean_difference_usd_per_session",
)

_METRIC_KEYS = frozenset({
    "sessions", "net_pnl_cents", "net_pnl_usd", "mean_daily_pnl_cents",
    "mean_daily_pnl_usd", "annualized_daily_sharpe", "profit_factor",
    "daily_profit_factor", "positive_day_fraction", "maximum_drawdown_cents",
    "maximum_drawdown_usd", "completed_round_trips",
})
_PER_SEED_METRIC_KEYS = _METRIC_KEYS | {
    "rejected_commands", "filled_sides", "commissions_cents",
    "explicit_commissions_usd",
}

_TRACE_FIELDS = (
    "decision_trace", "reward_trace", "fill_trace", "seed_decision_trace",
    "seed_reward_trace", "seed_fill_trace",
)
_LEARNED_DECISION_COLUMNS = (
    "session_date", "decision_ordinal", "decision_time_ns", "next_time_ns",
    "seed", "score_short", "score_flat", "score_long", "legal_short",
    "legal_flat", "legal_long", "member_vote", "votes_short", "votes_flat",
    "votes_long", "executed_action", "inventory_before", "inventory_after",
    "reward_cents", "terminal", "top_available",
)
_SEED_DECISION_COLUMNS = (
    "session_date", "decision_ordinal", "decision_time_ns", "next_time_ns",
    "seed", "score_short", "score_flat", "score_long", "legal_short",
    "legal_flat", "legal_long", "member_vote", "executed_action",
    "inventory_before", "inventory_after", "reward_cents", "terminal",
    "top_available",
)
_RULE_DECISION_COLUMNS = (
    "session_date", "decision_ordinal", "decision_time_ns", "next_time_ns",
    "seed", "score_short", "score_flat", "score_long", "legal_short",
    "legal_flat", "legal_long", "member_vote", "desired_action",
    "legality_fallback", "executed_action", "inventory_before",
    "inventory_after", "reward_cents", "terminal", "momentum_60m",
    "distance_session_vwap_atr",
)
_REWARD_COLUMNS = (
    "session_date", "decision_ordinal", "decision_time_ns", "next_time_ns",
    "executed_action", "legal_short", "legal_flat", "legal_long",
    "inventory_before", "inventory_after", "reward_cents",
    "cumulative_executable_nlv_cents", "command_created", "command_rejected",
    "terminal", "route_deadline_ns", "route_side", "route_fill_time_ns",
    "route_fill_price_ticks", "route_fill_fee_cents", "route_displayed_size",
    "route_searched_snapshots", "terminal_route_deadline_ns",
    "terminal_route_side", "terminal_fill_time_ns",
    "terminal_fill_price_ticks", "terminal_fill_fee_cents",
    "terminal_displayed_size",
)
_FILL_COLUMNS = (
    "session_date", "fill_ordinal", "timestamp_ns", "side", "price_ticks",
    "fee_cents", "forced",
)
_SEED_FILL_COLUMNS = (
    "session_date", "seed", "fill_ordinal", "timestamp_ns", "side",
    "price_ticks", "fee_cents", "forced",
)

_COMPLETE_REPORT_KEYS = frozenset({
    "schema_version", "artifact_type", "experiment_id", "protocol_id",
    "protocol", "implementation", "status", "arm", "policy_family", "period",
    "phase", "classification", "fresh_oos", "scenario", "economics",
    "evaluation_dates", "daily_pnl_cents", "metrics", "completed_round_trips",
    "rejected_commands", "filled_sides", "commissions_cents",
    "explicit_commissions_usd", "ensemble_seeds", "policy_replications",
    "ensemble_rule", "per_seed_daily_pnl_cents", "per_seed_metrics",
    "checkpoint_set", "suite_training_prerequisite", "training_manifest",
    "rule_scaler_reference", "dataset_manifest", *_TRACE_FIELDS,
    "reward_telescope_verified", "terminal_inventory_flat",
    "action_legality_verified", "routes_verified",
    "unsafe_legacy_arrival_boundary", "legality_fallbacks",
    "unexpected_policy_fallbacks", "per_seed_policy_dependent_replays_complete",
    "registered_evaluation", "holdout_read", "august_accessed",
    "included_in_reporting", "technical_failure_reason", "inferential_status",
})
_FAILURE_REPORT_KEYS = frozenset({
    "schema_version", "artifact_type", "experiment_id", "protocol_id",
    "protocol", "implementation", "status", "arm", "policy_family", "period",
    "phase", "classification", "fresh_oos", "scenario", "economics",
    "dataset_manifest", "suite_training_prerequisite", "training_manifest",
    "training_binding_status", "technical_failure_reason", "diagnostic",
    "evaluation_dates", "daily_pnl_cents", "metrics", *_TRACE_FIELDS,
    "registered_evaluation", "holdout_read", "august_accessed",
    "included_in_reporting", "inferential_status",
})
_TECHNICAL_FAILURE_REASONS = frozenset({
    "HASH_MISMATCH", "CAUSALITY_VIOLATION", "NONFINITE_STATE_OR_SCORE",
    "POLICY_LOAD_FAILURE", "REWARD_LEDGER_MISMATCH",
    "ACTION_LEGALITY_VIOLATION", "ROUTE_RECONCILIATION_FAILURE",
    "TERMINAL_INVENTORY_NOT_FLAT", "INCOMPLETE_TRACE",
    "UNEXPECTED_POLICY_FALLBACK",
})

_REGISTERED_REFERENCE_MANIFESTS: dict[str, str] = {}
for _reference_periods in REFERENCE_MANIFESTS.values():
    for _reference_path, _reference_sha256 in _reference_periods.values():
        existing = _REGISTERED_REFERENCE_MANIFESTS.setdefault(
            str(_reference_path), str(_reference_sha256),
        )
        if existing != str(_reference_sha256):  # pragma: no cover - frozen constant guard.
            raise AssertionError("reference manifest path has conflicting registered hashes")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} is not an integer >= {minimum}")
    return value


def _require_finite_number(value: object, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric or null")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} is nonfinite")


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    observed = frozenset(str(name) for name in value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} schema differs (missing={missing}, extra={extra})")


def _portable_relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a nonempty path string")
    raw = value.replace("\\", "/")
    reject_forbidden_text(raw)
    path = PurePosixPath(raw)
    if path.is_absolute() or not raw or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is not a safe project-relative path")
    return path.as_posix()


def _binding_shape(
    value: object, label: str, *, exact_keys: frozenset[str] | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} binding is absent")
    if exact_keys is not None:
        _exact_keys(value, exact_keys, label)
    if value.get("path_scope") != "project_root":
        raise ValueError(f"{label} path scope differs")
    _portable_relative(value.get("path"), label)
    _require_sha256(value.get("sha256"), f"{label} hash")
    return value


def _bound_file(
    value: object, label: str, *, root: Path,
    exact_keys: frozenset[str] | None = None,
    expected_path: Path | None = None,
) -> Path:
    record = _binding_shape(value, label, exact_keys=exact_keys)
    path = resolve_project_path(record["path"], root=root)
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"{label} location differs")
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"{label} file/hash differs")
    return path


def _verify_file_registry(value: object, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} file registry is absent")
    records: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} file record {index} is malformed")
        _exact_keys(item, frozenset({"path", "sha256"}), f"{label} file record")
        path = _portable_relative(item.get("path"), f"{label} file path")
        _require_sha256(item.get("sha256"), f"{label} file hash")
        if path in seen:
            raise ValueError(f"{label} file registry duplicates {path}")
        seen.add(path)
        records.append(item)
    return records


def _verify_implementation_shape(value: object) -> None:
    """Validate a recorded implementation receipt without opening its files."""

    if not isinstance(value, Mapping):
        raise ValueError("implementation binding is absent")
    _exact_keys(
        value,
        frozenset({
            "path_scope", "files", "file_count", "aggregate_sha256", "environment",
        }),
        "implementation binding",
    )
    if value.get("path_scope") != "project_root":
        raise ValueError("implementation path scope differs")
    files = _verify_file_registry(value.get("files"), "implementation")
    if _require_int(value.get("file_count"), "implementation file count") != len(files):
        raise ValueError("implementation file count differs")
    if value.get("aggregate_sha256") != canonical_json_sha256(files):
        raise ValueError("implementation registry digest differs")
    environment = value.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("implementation environment is absent")
    _exact_keys(
        environment,
        frozenset({
            "files", "python", "python_implementation", "system", "machine",
            "packages", "aggregate_sha256",
        }),
        "implementation environment",
    )
    _verify_file_registry(environment.get("files"), "environment")
    for name in ("python", "python_implementation", "system", "machine"):
        if not isinstance(environment.get(name), str) or not environment.get(name):
            raise ValueError(f"implementation environment {name} differs")
    packages = environment.get("packages")
    if not isinstance(packages, Mapping) or not packages or any(
        not isinstance(name, str) or not isinstance(version, str) or not version
        for name, version in packages.items()
    ):
        raise ValueError("implementation package registry is malformed")
    expected_environment = dict(environment)
    digest = expected_environment.pop("aggregate_sha256", None)
    _require_sha256(digest, "implementation environment hash")
    if digest != canonical_json_sha256(expected_environment):
        raise ValueError("implementation environment digest differs")


def portable_protocol_binding() -> dict[str, object]:
    """Bind only the two committed contract files needed by this verifier.

    Unlike :func:`kx_cvp_baselines.protocol.protocol_binding`, this helper does
    not invoke the full frozen-contract validator, because that validator also
    audits registered checkpoint and dataset artifacts.
    """

    directory = Path(__file__).resolve().parents[2] / "contracts"
    protocol_path = directory / "CVP_BASELINE_SUITE_PROTOCOL.json"
    receipt_path = directory / "CVP_BASELINE_SUITE_FREEZE_RECEIPT.json"
    protocol = read_json_object(protocol_path)
    receipt = read_json_object(receipt_path)
    protocol_file_hash = file_sha256(protocol_path)
    protocol_canonical_hash = canonical_json_sha256(protocol)
    if (
        protocol.get("experiment_id") != PROTOCOL_ID
        or receipt.get("experiment_id") != PROTOCOL_ID
        or receipt.get("protocol_file_sha256") != protocol_file_hash
        or receipt.get("protocol_canonical_sha256") != protocol_canonical_hash
        or receipt.get("frozen_before_production_training") is not True
        or receipt.get("frozen_before_new_baseline_outcome_access") is not True
        or receipt.get("august_access_authorized") is not False
    ):
        raise ValueError("portable contract/freeze-receipt identity differs")
    root = workspace_root().resolve()
    return {
        "path": protocol_path.resolve().relative_to(root).as_posix(),
        "path_scope": "project_root",
        "sha256": protocol_file_hash,
        "canonical_sha256": protocol_canonical_hash,
        "freeze_receipt": {
            "path": receipt_path.resolve().relative_to(root).as_posix(),
            "path_scope": "project_root",
            "sha256": file_sha256(receipt_path),
        },
    }


def _verify_protocol(value: object) -> None:
    if value != portable_protocol_binding():
        raise ValueError("portable artifact protocol binding differs")


def _verify_metric_shape(value: object, label: str, *, per_seed: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is absent")
    expected = _PER_SEED_METRIC_KEYS if per_seed else _METRIC_KEYS
    _exact_keys(value, frozenset(expected), label)
    for name, item in value.items():
        _require_finite_number(item, f"{label}.{name}")


def _verify_lineage_shapes(
    value: Mapping[str, object], *, arm: str, period: str,
    dates: Sequence[str] | None,
) -> None:
    suite = _binding_shape(
        value.get("suite_training_prerequisite"),
        "suite training prerequisite",
        exact_keys=frozenset({"path", "path_scope", "sha256"}),
    )
    del suite  # Shape only: the ignored training receipt is never opened.

    dataset = value.get("dataset_manifest")
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset binding is absent")
    expected_dataset_keys = {
        "path", "path_scope", "sha256", "purpose", "period_id", "classification",
    }
    if dates is not None:
        expected_dataset_keys.add("dates")
    _exact_keys(dataset, frozenset(expected_dataset_keys), "dataset binding")
    _binding_shape(dataset, "dataset binding")
    registered = DATASET_MANIFESTS[period]
    if (
        dataset.get("path") != registered["path"]
        or dataset.get("sha256") != registered["sha256"]
        or dataset.get("purpose") != "evaluate"
        or dataset.get("period_id") != period
        or dataset.get("classification") != PERIOD_CLASSIFICATION[period]
        or (dates is not None and dataset.get("dates") != list(dates))
    ):
        raise ValueError("dataset lineage record differs from the frozen contract")

    training = value.get("training_manifest")
    if arm in LEARNER_ARMS:
        _binding_shape(
            training, "training manifest",
            exact_keys=frozenset({"path", "path_scope", "sha256"}),
        )
    elif training is not None:
        raise ValueError("deterministic report unexpectedly binds training")


def _trace_path(
    report_path: Path, record: object, field: str, expected_name: str,
) -> tuple[Path, int]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{field} binding is absent")
    _exact_keys(
        record, frozenset({"path", "path_scope", "sha256", "rows"}), field,
    )
    if record.get("path_scope") != "report_relative" or record.get("path") != expected_name:
        raise ValueError(f"{field} path differs")
    _require_sha256(record.get("sha256"), f"{field} hash")
    rows = _require_int(record.get("rows"), f"{field} row count")
    path = (report_path.parent / expected_name).resolve()
    if path.parent != report_path.parent.resolve():
        raise ValueError(f"{field} escapes its report directory")
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"{field} file/hash differs")
    metadata_rows = pq.ParquetFile(path).metadata.num_rows
    if metadata_rows != rows:
        raise ValueError(f"{field} Parquet row count differs")
    return path, rows


def _require_arrow_type(field: pa.Field, family: str, label: str) -> None:
    kind = field.type
    allowed = {
        "string": pa.types.is_string,
        "integer": pa.types.is_integer,
        "floating": pa.types.is_floating,
        "boolean": pa.types.is_boolean,
    }[family]
    if not allowed(kind):
        raise ValueError(f"{label} has Arrow type {kind}, expected {family}")


def _require_nullable_arrow_type(field: pa.Field, family: str, label: str) -> None:
    if pa.types.is_null(field.type):
        return
    _require_arrow_type(field, family, label)


def _verify_trace_schema(path: Path, expected: Sequence[str], label: str) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    if tuple(schema.names) != tuple(expected):
        raise ValueError(f"{label} Parquet columns differ")
    by_name = {field.name: field for field in schema}
    _require_arrow_type(by_name["session_date"], "string", f"{label}.session_date")
    for name in (
        "decision_ordinal", "decision_time_ns", "next_time_ns", "seed",
        "member_vote", "votes_short", "votes_flat", "votes_long",
        "desired_action", "executed_action", "inventory_before", "inventory_after",
        "reward_cents", "cumulative_executable_nlv_cents", "fill_ordinal",
        "timestamp_ns", "side", "price_ticks", "fee_cents",
    ):
        if name in by_name:
            _require_arrow_type(by_name[name], "integer", f"{label}.{name}")
    for name in ("score_short", "score_flat", "score_long"):
        if name in by_name:
            _require_arrow_type(by_name[name], "floating", f"{label}.{name}")
    for name in (
        "legal_short", "legal_flat", "legal_long", "legality_fallback", "terminal",
        "command_created", "command_rejected", "forced",
    ):
        if name in by_name:
            _require_arrow_type(by_name[name], "boolean", f"{label}.{name}")
    for name in (
        "route_deadline_ns", "route_side", "route_fill_time_ns",
        "route_fill_price_ticks", "route_fill_fee_cents", "route_displayed_size",
        "route_searched_snapshots", "terminal_route_deadline_ns",
        "terminal_route_side", "terminal_fill_time_ns", "terminal_fill_price_ticks",
        "terminal_fill_fee_cents", "terminal_displayed_size",
    ):
        if name in by_name:
            _require_nullable_arrow_type(by_name[name], "integer", f"{label}.{name}")
    if "top_available" in by_name:
        _require_nullable_arrow_type(
            by_name["top_available"], "boolean", f"{label}.top_available",
        )
    for name in ("momentum_60m", "distance_session_vwap_atr"):
        if name in by_name:
            _require_nullable_arrow_type(by_name[name], "floating", f"{label}.{name}")


def _verify_report_traces(
    report_path: Path, value: Mapping[str, object], *, learned: bool,
) -> dict[str, int]:
    names = {
        "decision_trace": "decisions.parquet",
        "reward_trace": "rewards.parquet",
        "fill_trace": "fills.parquet",
        "seed_decision_trace": "seed_decisions.parquet",
        "seed_reward_trace": "seed_rewards.parquet",
        "seed_fill_trace": "seed_fills.parquet",
    }
    expected_columns: dict[str, tuple[str, ...]] = {
        "decision_trace": _LEARNED_DECISION_COLUMNS if learned else _RULE_DECISION_COLUMNS,
        "reward_trace": _REWARD_COLUMNS,
        "fill_trace": _FILL_COLUMNS,
        "seed_decision_trace": _SEED_DECISION_COLUMNS,
        "seed_reward_trace": (*_REWARD_COLUMNS, "seed"),
        "seed_fill_trace": _SEED_FILL_COLUMNS,
    }
    counts: dict[str, int] = {}
    required = _TRACE_FIELDS if learned else _TRACE_FIELDS[:3]
    for field in required:
        path, rows = _trace_path(report_path, value.get(field), field, names[field])
        _verify_trace_schema(path, expected_columns[field], field)
        counts[field] = rows
    if not learned and any(value.get(field) is not None for field in _TRACE_FIELDS[3:]):
        raise ValueError("deterministic report contains seed trace bindings")
    return counts


def _verify_complete_report_shape(
    report_path: Path, value: Mapping[str, object], *, expected_arm: str | None,
    expected_period: str | None, expected_scenario: str | None,
) -> dict[str, object]:
    _exact_keys(value, _COMPLETE_REPORT_KEYS, "complete evaluation report")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != REPLAY_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("registered_evaluation") is not True
        or value.get("holdout_read") is not True
        or value.get("august_accessed") is not False
        or value.get("included_in_reporting") is not True
        or value.get("technical_failure_reason") is not None
        or value.get("inferential_status") != "POST_HOC_DESCRIPTIVE"
    ):
        raise ValueError("portable complete-report identity differs")
    arm = str(value.get("arm"))
    period = str(value.get("period"))
    scenario = str(value.get("scenario"))
    if arm not in BASELINE_ARMS or period not in PERIODS or scenario not in SCENARIOS:
        raise ValueError("portable complete-report cell is unregistered")
    if any((expected_arm is not None and arm != expected_arm,
            expected_period is not None and period != expected_period,
            expected_scenario is not None and scenario != expected_scenario)):
        raise ValueError("portable complete-report cell differs from its registry")
    _verify_protocol(value.get("protocol"))
    _verify_implementation_shape(value.get("implementation"))
    if value.get("economics") != asdict(scenario_economics(scenario)):
        raise ValueError("portable complete-report scenario economics differ")
    if value.get("phase") != ("validation" if period == "may_2026" else "test"):
        raise ValueError("portable complete-report phase differs")

    dates = value.get("evaluation_dates")
    if not isinstance(dates, list) or not dates or any(not isinstance(day, str) for day in dates):
        raise ValueError("portable complete-report date schema differs")
    reject_forbidden_text(*dates)
    start, end_exclusive = PERIOD_BOUNDS[period]
    if dates != sorted(set(dates)) or any(
        not start <= str(day) < end_exclusive for day in dates
    ):
        raise ValueError("portable complete-report dates differ from the registered period")
    daily = value.get("daily_pnl_cents")
    if not isinstance(daily, Mapping) or sorted(str(day) for day in daily) != dates or any(
        isinstance(amount, bool) or not isinstance(amount, int) for amount in daily.values()
    ):
        raise ValueError("portable complete-report daily-cash schema differs")
    _verify_metric_shape(value.get("metrics"), "complete report metrics")
    for name in (
        "completed_round_trips", "rejected_commands", "filled_sides",
        "commissions_cents", "legality_fallbacks", "unexpected_policy_fallbacks",
    ):
        _require_int(value.get(name), f"complete report {name}")
    _require_finite_number(value.get("explicit_commissions_usd"), "explicit commissions")
    _verify_lineage_shapes(value, arm=arm, period=period, dates=dates)

    learned = arm in LEARNER_ARMS
    expected_family = "learned_five_seed" if learned else "deterministic_single_path"
    expected_rule = ENSEMBLE_RULE if learned else DETERMINISTIC_RULE
    if (
        value.get("policy_family") != expected_family
        or value.get("ensemble_seeds") != (list(SEEDS) if learned else [])
        or value.get("policy_replications") != (len(SEEDS) if learned else 1)
        or value.get("ensemble_rule") != expected_rule
        or value.get("per_seed_policy_dependent_replays_complete") is not learned
    ):
        raise ValueError("portable complete-report policy schema differs")
    checkpoints = value.get("checkpoint_set")
    if learned:
        if not isinstance(checkpoints, list) or len(checkpoints) != len(SEEDS):
            raise ValueError("portable checkpoint registry is incomplete")
        observed: list[int] = []
        for item in checkpoints:
            if not isinstance(item, Mapping):
                raise ValueError("portable checkpoint record is malformed")
            _exact_keys(item, frozenset({"seed", "manifest", "model_sha256"}), "checkpoint record")
            observed.append(_require_int(item.get("seed"), "checkpoint seed"))
            _binding_shape(
                item.get("manifest"), "checkpoint manifest",
                exact_keys=frozenset({"path", "path_scope", "sha256"}),
            )
            _require_sha256(item.get("model_sha256"), "checkpoint model hash")
        if tuple(observed) != SEEDS or value.get("rule_scaler_reference") is not None:
            raise ValueError("portable checkpoint seed/order registry differs")
        per_seed_daily = value.get("per_seed_daily_pnl_cents")
        per_seed_metrics = value.get("per_seed_metrics")
        if not isinstance(per_seed_daily, Mapping) or not isinstance(per_seed_metrics, Mapping):
            raise ValueError("portable per-seed report schema is absent")
        if tuple(per_seed_daily) != tuple(str(seed) for seed in SEEDS) or tuple(
            per_seed_metrics
        ) != tuple(str(seed) for seed in SEEDS):
            raise ValueError("portable per-seed registry differs")
        for seed in SEEDS:
            one = per_seed_daily[str(seed)]
            if not isinstance(one, Mapping) or sorted(str(day) for day in one) != dates or any(
                isinstance(amount, bool) or not isinstance(amount, int)
                for amount in one.values()
            ):
                raise ValueError("portable per-seed daily schema differs")
            _verify_metric_shape(
                per_seed_metrics[str(seed)], f"per-seed metrics {seed}", per_seed=True,
            )
    else:
        if checkpoints != [] or value.get("per_seed_daily_pnl_cents") is not None or value.get(
            "per_seed_metrics"
        ) is not None:
            raise ValueError("deterministic report contains replicated policy records")
        scaler = _binding_shape(
            value.get("rule_scaler_reference"), "rule scaler reference",
            exact_keys=frozenset({
                "path", "path_scope", "sha256", "scaler_canonical_sha256",
                "policy_uses_scaled_values",
            }),
        )
        if scaler.get("policy_uses_scaled_values") is not False:
            raise ValueError("deterministic rule scaler-use flag differs")
        _require_sha256(scaler.get("scaler_canonical_sha256"), "rule scaler canonical hash")

    for name in (
        "reward_telescope_verified", "terminal_inventory_flat",
        "action_legality_verified", "routes_verified",
    ):
        if value.get(name) is not True:
            raise ValueError(f"portable report flag {name} differs")
    if value.get("unsafe_legacy_arrival_boundary") is not False:
        raise ValueError("portable report legacy-boundary flag differs")
    trace_counts = _verify_report_traces(report_path, value, learned=learned)
    return {
        "artifact_type": REPLAY_ARTIFACT,
        "status": "PASS",
        "arm": arm,
        "period": period,
        "scenario": scenario,
        "trace_rows": trace_counts,
    }


def _verify_failure_report_shape(
    value: Mapping[str, object], *, expected_arm: str | None,
    expected_period: str | None, expected_scenario: str | None,
) -> dict[str, object]:
    _exact_keys(value, _FAILURE_REPORT_KEYS, "technical-failure evaluation report")
    absent = ("evaluation_dates", "daily_pnl_cents", "metrics", *_TRACE_FIELDS)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != TECHNICAL_FAILURE_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "TECHNICAL_FAILURE"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("registered_evaluation") is not True
        or not isinstance(value.get("holdout_read"), bool)
        or value.get("august_accessed") is not False
        or value.get("included_in_reporting") is not True
        or value.get("inferential_status") != "TECHNICALLY_INVALID_REPORTED"
        or value.get("technical_failure_reason") not in _TECHNICAL_FAILURE_REASONS
        or not isinstance(value.get("diagnostic"), str)
        or not str(value.get("diagnostic")).strip()
        or any(value.get(name) is not None for name in absent)
    ):
        raise ValueError("portable technical-failure identity differs")
    arm = str(value.get("arm"))
    period = str(value.get("period"))
    scenario = str(value.get("scenario"))
    if arm not in BASELINE_ARMS or period not in PERIODS or scenario not in SCENARIOS:
        raise ValueError("portable technical-failure cell is unregistered")
    if any((expected_arm is not None and arm != expected_arm,
            expected_period is not None and period != expected_period,
            expected_scenario is not None and scenario != expected_scenario)):
        raise ValueError("portable technical-failure cell differs from its registry")
    _verify_protocol(value.get("protocol"))
    _verify_implementation_shape(value.get("implementation"))
    if (
        value.get("policy_family")
        != ("learned_five_seed" if arm in LEARNER_ARMS else "deterministic_single_path")
        or value.get("phase") != ("validation" if period == "may_2026" else "test")
        or value.get("economics") != asdict(scenario_economics(scenario))
    ):
        raise ValueError("portable technical-failure policy/economics differs")
    _verify_lineage_shapes(value, arm=arm, period=period, dates=None)
    if arm in LEARNER_ARMS:
        if value.get("training_binding_status") != "VERIFIED":
            raise ValueError("learner failure lacks a structurally verified training binding")
    elif value.get("training_binding_status") != "NOT_APPLICABLE":
        raise ValueError("deterministic failure training status differs")
    return {
        "artifact_type": TECHNICAL_FAILURE_ARTIFACT,
        "status": "PASS",
        "arm": arm,
        "period": period,
        "scenario": scenario,
        "trace_rows": {},
    }


def verify_portable_replay_report(
    path: Path | str, *, expected_arm: str | None = None,
    expected_period: str | None = None, expected_scenario: str | None = None,
    project_root: Path | None = None, allowed_root: Path | None = None,
) -> dict[str, object]:
    """Verify one committed report and its Parquet structure, never its lineage."""

    reject_forbidden_text(path, expected_arm or "", expected_period or "", expected_scenario or "")
    root = workspace_root().resolve() if project_root is None else Path(project_root).resolve()
    boundary = root if allowed_root is None else Path(allowed_root).resolve()
    report_path = Path(path).resolve()
    try:
        boundary.relative_to(root)
        report_path.relative_to(boundary)
    except ValueError as error:
        raise ValueError("portable evaluation report is outside its allowed root") from error
    value = read_json_object(report_path)
    artifact = value.get("artifact_type")
    if artifact == REPLAY_ARTIFACT:
        result = _verify_complete_report_shape(
            report_path, value, expected_arm=expected_arm,
            expected_period=expected_period, expected_scenario=expected_scenario,
        )
    elif artifact == TECHNICAL_FAILURE_ARTIFACT:
        result = _verify_failure_report_shape(
            value, expected_arm=expected_arm, expected_period=expected_period,
            expected_scenario=expected_scenario,
        )
    else:
        raise ValueError("portable evaluation-report artifact type differs")
    return {
        "verification_level": PORTABLE_VERIFICATION_LEVEL,
        "report_sha256": file_sha256(report_path),
        **result,
    }


def verify_portable_evaluation_manifest(
    path: Path | str, *, project_root: Path | None = None,
    allowed_root: Path | None = None,
) -> dict[str, object]:
    """Verify the 64-cell registry and only report-local committed artifacts."""

    reject_forbidden_text(path)
    root = workspace_root().resolve() if project_root is None else Path(project_root).resolve()
    boundary = root if allowed_root is None else Path(allowed_root).resolve()
    manifest_path = Path(path).resolve()
    try:
        boundary.relative_to(root)
        manifest_path.relative_to(boundary)
    except ValueError as error:
        raise ValueError("portable evaluation manifest is outside its allowed root") from error
    if manifest_path.name != "manifest.json" or manifest_path.parent.name != "evaluation":
        raise ValueError("portable evaluation-manifest location differs")
    value = read_json_object(manifest_path)
    expected_keys = frozenset({
        "schema_version", "artifact_type", "experiment_id", "protocol_id",
        "protocol", "implementation", "status", "complete", "arms", "periods",
        "scenarios", "required_cells", "complete_cells", "technical_failure_cells",
        "period_classification", "fresh_oos", "august_accessed",
        "all_registered_results_retained", "result_dependent_omission",
        "result_dependent_omission_permitted", "training_manifest",
        "suite_training_prerequisite", "learner_cells_with_verified_training",
        "holdout_read_cells", "reports", "report_records_sha256",
        "scientific_status",
    })
    _exact_keys(value, expected_keys, "complete evaluation manifest")
    records = value.get("reports")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != COMPLETE_EVALUATION_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("complete") is not True
        or value.get("arms") != list(BASELINE_ARMS)
        or value.get("periods") != list(PERIODS)
        or value.get("scenarios") != list(SCENARIOS)
        or value.get("required_cells") != 64
        or value.get("period_classification")
        != {period: PERIOD_CLASSIFICATION[period] for period in PERIODS}
        or value.get("fresh_oos") is not False
        or value.get("august_accessed") is not False
        or value.get("all_registered_results_retained") is not True
        or value.get("result_dependent_omission") is not False
        or value.get("result_dependent_omission_permitted") is not False
        or value.get("scientific_status")
        != "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE"
        or not isinstance(records, list)
        or len(records) != 64
    ):
        raise ValueError("portable evaluation-manifest identity differs")
    _verify_protocol(value.get("protocol"))
    _verify_implementation_shape(value.get("implementation"))
    if value.get("report_records_sha256") != canonical_json_sha256(records):
        raise ValueError("portable evaluation report-record digest differs")
    _binding_shape(
        value.get("suite_training_prerequisite"), "evaluation suite training prerequisite",
        exact_keys=frozenset({"path", "path_scope", "sha256"}),
    )
    if value.get("training_manifest") is not None:
        _binding_shape(
            value.get("training_manifest"), "evaluation training manifest",
            exact_keys=frozenset({"path", "path_scope", "sha256"}),
        )

    expected_order = [
        (arm, period, scenario)
        for arm in BASELINE_ARMS for period in PERIODS for scenario in SCENARIOS
    ]
    statuses: list[str] = []
    training_bindings: list[object] = []
    suite_bindings: list[object] = []
    holdout_reads = 0
    trace_files = 0
    trace_rows = 0
    evaluation_root = manifest_path.parent.resolve()
    for index, (record, cell) in enumerate(zip(records, expected_order, strict=True)):
        if not isinstance(record, Mapping):
            raise ValueError(f"portable evaluation record {index} is malformed")
        _exact_keys(
            record, frozenset({"arm", "period", "scenario", "status", "path", "sha256"}),
            "evaluation report record",
        )
        arm, period, scenario = cell
        expected_relative = f"{period}/{scenario}/{arm}/report.json"
        if (
            record.get("arm") != arm or record.get("period") != period
            or record.get("scenario") != scenario
            or record.get("path") != expected_relative
            or record.get("status") not in {"COMPLETE", "TECHNICAL_FAILURE"}
        ):
            raise ValueError("portable evaluation registry order/cell differs")
        _require_sha256(record.get("sha256"), "evaluation report hash")
        report_path = (manifest_path.parent / expected_relative).resolve()
        try:
            report_path.relative_to(manifest_path.parent.resolve())
        except ValueError as error:
            raise ValueError("portable evaluation report escapes its registry") from error
        if not report_path.is_file() or file_sha256(report_path) != record["sha256"]:
            raise ValueError("portable evaluation report file/hash differs")
        verified = verify_portable_replay_report(
            report_path, expected_arm=arm, expected_period=period,
            expected_scenario=scenario,
            project_root=root, allowed_root=evaluation_root,
        )
        report = read_json_object(report_path)
        if report.get("status") != record["status"]:
            raise ValueError("portable evaluation report status differs from registry")
        statuses.append(str(record["status"]))
        if report.get("training_manifest") is not None:
            training_bindings.append(report.get("training_manifest"))
        suite_bindings.append(report.get("suite_training_prerequisite"))
        holdout_reads += report.get("holdout_read") is True
        counts = verified["trace_rows"]
        assert isinstance(counts, Mapping)
        trace_files += len(counts)
        trace_rows += sum(int(number) for number in counts.values())

    complete = statuses.count("COMPLETE")
    unique_training = {canonical_json_sha256(item): item for item in training_bindings}
    unique_suite = {canonical_json_sha256(item): item for item in suite_bindings}
    if (
        value.get("complete_cells") != complete
        or value.get("technical_failure_cells") != 64 - complete
        or value.get("learner_cells_with_verified_training") != len(training_bindings)
        or value.get("holdout_read_cells") != holdout_reads
        or len(unique_training) > 1
        or value.get("training_manifest") != next(iter(unique_training.values()), None)
        or len(unique_suite) != 1
        or value.get("suite_training_prerequisite")
        != next(iter(unique_suite.values()), None)
    ):
        raise ValueError("portable evaluation summary/lineage registry differs")
    return {
        "verification_level": PORTABLE_VERIFICATION_LEVEL,
        "status": "PASS",
        "manifest_sha256": file_sha256(manifest_path),
        "reports_verified": 64,
        "complete_reports": complete,
        "technical_failure_reports": 64 - complete,
        "trace_files_verified": trace_files,
        "trace_rows_registered": trace_rows,
    }


def _csv_schema_and_rows(path: Path) -> tuple[tuple[str, ...], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"CSV artifact is empty: {path}") from error
        return header, sum(1 for _ in reader)


def _expected_reference_sources(root: Path) -> list[dict[str, object]]:
    """Derive the exact 20 registered result-report bindings from frozen manifests."""

    loaded: dict[str, tuple[Path, Mapping[str, object], str]] = {}

    def load(policy: str, period: str) -> tuple[Path, Mapping[str, object], str]:
        registered = REFERENCE_MANIFESTS[policy].get(period)
        if registered is None:
            registered = REFERENCE_MANIFESTS[policy].get("all")
        if registered is None:
            raise ValueError("registered reference manifest selection differs")
        relative, expected_sha = str(registered[0]), str(registered[1])
        existing = loaded.get(relative)
        if existing is not None:
            return existing
        path = resolve_project_path(relative, root=root)
        if not path.is_file() or file_sha256(path) != expected_sha:
            raise ValueError("registered reference manifest file/hash differs")
        value = read_json_object(path)
        loaded[relative] = (path, value, expected_sha)
        return loaded[relative]

    bindings: set[tuple[str, str, str, str]] = set()
    for period in PERIODS:
        policies = ["cvp_hddqn", "no_publisher_single_agent_ddqn"]
        if period == "may_2026":
            policies.append("shavandi_khedmati_reconstruction")
        for scenario in SCENARIOS:
            for policy in policies:
                manifest_path, manifest, manifest_sha = load(policy, period)
                reports = manifest.get("reports")
                if not isinstance(reports, list):
                    raise ValueError("registered reference manifest lacks reports")
                source_scenario = (
                    scenario
                    if policy == "no_publisher_single_agent_ddqn"
                    else SCENARIO_ALIASES[scenario]
                )
                matches: list[Mapping[str, object]] = []
                for record in reports:
                    if not isinstance(record, Mapping) or record.get("scenario") != source_scenario:
                        continue
                    if policy == "cvp_hddqn" and record.get("arm") not in {
                        "V6-RichVolume", "CVP-HDDQN",
                    }:
                        continue
                    if policy == "no_publisher_single_agent_ddqn" and (
                        record.get("arm") != "no_publisher"
                        or record.get("period") != period
                    ):
                        continue
                    matches.append(record)
                if len(matches) != 1:
                    raise ValueError("registered reference report selection is not unique")
                record = matches[0]
                relative_report = PurePosixPath(str(record.get("path", "")))
                if (
                    relative_report.is_absolute()
                    or not relative_report.parts
                    or any(part in {"", ".", ".."} for part in relative_report.parts)
                ):
                    raise ValueError("registered reference report path is unsafe")
                report_path = (manifest_path.parent / relative_report).resolve()
                try:
                    report_path.relative_to(manifest_path.parent.resolve())
                    report_relative_to_root = report_path.relative_to(root).as_posix()
                except ValueError as error:
                    raise ValueError("registered reference report escapes its root") from error
                report_sha = _require_sha256(
                    record.get("sha256"), "registered reference report hash",
                )
                bindings.add((
                    manifest_path.relative_to(root).as_posix(), manifest_sha,
                    report_relative_to_root, report_sha,
                ))
    rows = [
        {
            "manifest_path": item[0], "manifest_sha256": item[1],
            "report_path": item[2], "report_sha256": item[3],
        }
        for item in sorted(bindings)
    ]
    if len(rows) != 20:
        raise ValueError("portable reference registry is not the exact 20-report set")
    return rows


def _verify_product(
    manifest_path: Path, record: object, *, root: Path, name: str,
    filename: str, columns: Sequence[str], parquet: bool = False,
) -> dict[str, object]:
    expected_path = (manifest_path.parent / filename).resolve()
    path = _bound_file(
        record, f"evidence product {name}", root=root,
        exact_keys=frozenset({"path", "path_scope", "sha256", "rows"}),
        expected_path=expected_path,
    )
    assert isinstance(record, Mapping)
    registered_rows = _require_int(record.get("rows"), f"evidence product {name} rows")
    if parquet:
        parquet_file = pq.ParquetFile(path)
        parquet_schema = parquet_file.schema_arrow
        observed_columns = tuple(parquet_schema.names)
        observed_rows = parquet_file.metadata.num_rows
    else:
        observed_columns, observed_rows = _csv_schema_and_rows(path)
    if observed_columns != tuple(columns):
        raise ValueError(f"evidence product {name} schema differs")
    if observed_rows != registered_rows:
        raise ValueError(f"evidence product {name} row count differs")
    if parquet:
        by_name = {field.name: field for field in parquet_schema}
        for field_name in (
            "reference_policy_id", "baseline_policy_id", "period", "scenario",
        ):
            _require_arrow_type(
                by_name[field_name], "string", f"evidence product {name}.{field_name}",
            )
        _require_arrow_type(
            by_name["replicate"], "integer", f"evidence product {name}.replicate",
        )
        _require_arrow_type(
            by_name["mean_difference_usd_per_session"], "floating",
            f"evidence product {name}.mean_difference_usd_per_session",
        )
    return {"sha256": str(record["sha256"]), "rows": registered_rows}


def verify_portable_evidence_manifest(
    path: Path | str, *, project_root: Path | None = None,
    expected_evaluation_path: Path | None = None,
) -> dict[str, object]:
    """Verify bound evidence bytes/schemas without reconstructing their values."""

    reject_forbidden_text(path)
    root = workspace_root().resolve() if project_root is None else Path(project_root).resolve()
    manifest_path = Path(path).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError("portable evidence manifest is outside the project root") from error
    if manifest_path.name != "manifest.json" or manifest_path.parent.name != "source_data":
        raise ValueError("portable evidence-manifest location differs")
    value = read_json_object(manifest_path)
    expected_keys = frozenset({
        "schema_version", "artifact_type", "experiment_id", "complete",
        "scientific_status", "period_classification", "june_july_headline",
        "may_appendix", "august_accessed", "all_registered_results_retained",
        "all_new_baseline_cells_retained", "registered_result_scope",
        "selected_reference_subset", "protocol", "implementation",
        "evaluation_manifest", "reference_sources", "products", "content_sha256",
    })
    _exact_keys(value, expected_keys, "portable evidence manifest")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != EVIDENCE_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("complete") is not True
        or value.get("scientific_status") != "CONTAMINATED_HISTORICAL_DESCRIPTIVE"
        or value.get("period_classification") != PERIOD_CLASSIFICATION
        or value.get("june_july_headline") is not True
        or value.get("may_appendix") is not True
        or value.get("august_accessed") is not False
        or value.get("all_registered_results_retained") is not True
        or value.get("all_new_baseline_cells_retained") is not True
        or value.get("registered_result_scope")
        != "eight new baseline arms x two periods x four scenarios"
    ):
        raise ValueError("portable evidence identity differs")
    _verify_protocol(value.get("protocol"))
    _verify_implementation_shape(value.get("implementation"))
    _require_sha256(value.get("content_sha256"), "evidence content hash")
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
        raise ValueError("portable evidence reference subset differs")
    evaluation_record = _binding_shape(
        value.get("evaluation_manifest"), "evidence evaluation manifest",
        exact_keys=frozenset({"path", "path_scope", "sha256"}),
    )
    recorded_evaluation_path = resolve_project_path(evaluation_record["path"], root=root)
    authorized_evaluation_path = (
        recorded_evaluation_path
        if expected_evaluation_path is None else Path(expected_evaluation_path).resolve()
    )
    if (
        authorized_evaluation_path.name != "manifest.json"
        or authorized_evaluation_path.parent.name != "evaluation"
    ):
        raise ValueError("evidence evaluation-manifest location is unauthorized")
    evaluation_path = _bound_file(
        evaluation_record, "evidence evaluation manifest", root=root,
        exact_keys=frozenset({"path", "path_scope", "sha256"}),
        expected_path=authorized_evaluation_path,
    )
    evaluation = verify_portable_evaluation_manifest(
        evaluation_path, project_root=root,
        allowed_root=authorized_evaluation_path.parent.parent,
    )

    references = value.get("reference_sources")
    if not isinstance(references, list):
        raise ValueError("portable reference-source registry is absent")
    expected_references = _expected_reference_sources(root)
    if references != expected_references:
        raise ValueError("portable reference-source registry differs")
    seen_references: set[tuple[str, str]] = set()
    for item in references:
        if not isinstance(item, Mapping):
            raise ValueError("portable reference-source record is malformed")
        _exact_keys(
            item,
            frozenset({
                "manifest_path", "manifest_sha256", "report_path", "report_sha256",
            }),
            "reference-source record",
        )
        manifest_path_value = str(item.get("manifest_path"))
        expected_manifest_sha = _REGISTERED_REFERENCE_MANIFESTS.get(
            manifest_path_value,
        )
        if expected_manifest_sha is None or item.get("manifest_sha256") != expected_manifest_sha:
            raise ValueError("reference manifest is not in the frozen allowlist")
        manifest_record = {
            "path": item.get("manifest_path"), "path_scope": "project_root",
            "sha256": item.get("manifest_sha256"),
        }
        report_record = {
            "path": item.get("report_path"), "path_scope": "project_root",
            "sha256": item.get("report_sha256"),
        }
        authorized_manifest = resolve_project_path(manifest_path_value, root=root)
        reference_manifest_path = _bound_file(
            manifest_record, "reference manifest", root=root,
            expected_path=authorized_manifest,
        )
        reference_manifest = read_json_object(reference_manifest_path)
        registered_reports = reference_manifest.get("reports")
        if not isinstance(registered_reports, list):
            raise ValueError("registered reference manifest lacks a report registry")
        report_path_value = _portable_relative(
            item.get("report_path"), "reference report path",
        )
        authorized_report = resolve_project_path(report_path_value, root=root)
        report_is_registered = False
        for registered in registered_reports:
            if not isinstance(registered, Mapping):
                continue
            relative = PurePosixPath(str(registered.get("path", "")))
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                continue
            candidate = (reference_manifest_path.parent / relative).resolve()
            try:
                candidate.relative_to(reference_manifest_path.parent.resolve())
            except ValueError:
                continue
            if (
                candidate == authorized_report
                and registered.get("sha256") == item.get("report_sha256")
            ):
                report_is_registered = True
                break
        if not report_is_registered:
            raise ValueError("reference report is absent from its frozen manifest")
        _bound_file(
            report_record, "reference report", root=root,
            expected_path=authorized_report,
        )
        key = (str(item["manifest_path"]), str(item["report_path"]))
        if key in seen_references:
            raise ValueError("portable reference-source registry duplicates a report")
        seen_references.add(key)

    products = value.get("products")
    if not isinstance(products, Mapping):
        raise ValueError("portable evidence product registry is absent")
    _exact_keys(
        products,
        frozenset({"metrics", "daily_pnl", "paired_effects", "bootstrap_draws"}),
        "portable evidence products",
    )
    verified_products = {
        "metrics": _verify_product(
            manifest_path, products["metrics"], root=root, name="metrics",
            filename="metrics.csv", columns=METRICS_COLUMNS,
        ),
        "daily_pnl": _verify_product(
            manifest_path, products["daily_pnl"], root=root, name="daily_pnl",
            filename="daily_pnl.csv", columns=DAILY_COLUMNS,
        ),
        "paired_effects": _verify_product(
            manifest_path, products["paired_effects"], root=root,
            name="paired_effects", filename="paired_effects.csv",
            columns=EFFECT_COLUMNS,
        ),
        "bootstrap_draws": _verify_product(
            manifest_path, products["bootstrap_draws"], root=root,
            name="bootstrap_draws", filename="bootstrap_draws.parquet",
            columns=BOOTSTRAP_COLUMNS, parquet=True,
        ),
    }
    return {
        "verification_level": PORTABLE_VERIFICATION_LEVEL,
        "status": "PASS",
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_manifest_sha256": evaluation["manifest_sha256"],
        "reference_files_verified": 2 * len(references),
        "products": verified_products,
    }


def verify_portable_report_manifest(
    path: Path | str, *, project_root: Path | None = None,
    expected_evaluation_path: Path | None = None,
) -> dict[str, object]:
    """Verify standalone report outputs without rendering or recomputing them."""

    reject_forbidden_text(path)
    root = workspace_root().resolve() if project_root is None else Path(project_root).resolve()
    manifest_path = Path(path).resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError("portable report manifest is outside the project root") from error
    if manifest_path.name != "report_manifest.json":
        raise ValueError("portable standalone-report manifest filename differs")
    value = read_json_object(manifest_path)
    _exact_keys(
        value,
        frozenset({
            "schema_version", "artifact_type", "experiment_id", "scientific_status",
            "paper_modified", "august_accessed", "protocol", "implementation",
            "evidence_manifest", "outputs",
        }),
        "portable standalone-report manifest",
    )
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != REPORT_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("scientific_status") != "CONTAMINATED_HISTORICAL_DESCRIPTIVE"
        or value.get("paper_modified") is not False
        or value.get("august_accessed") is not False
    ):
        raise ValueError("portable standalone-report identity differs")
    _verify_protocol(value.get("protocol"))
    _verify_implementation_shape(value.get("implementation"))
    expected_evidence_path = (manifest_path.parent / "source_data" / "manifest.json").resolve()
    evidence_path = _bound_file(
        value.get("evidence_manifest"), "standalone-report evidence", root=root,
        exact_keys=frozenset({"path", "path_scope", "sha256"}),
        expected_path=expected_evidence_path,
    )
    evidence = verify_portable_evidence_manifest(
        evidence_path, project_root=root,
        expected_evaluation_path=expected_evaluation_path,
    )
    outputs = value.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("portable standalone-report outputs are absent")
    _exact_keys(outputs, frozenset({"markdown", "pdf"}), "standalone-report outputs")
    markdown_path = _bound_file(
        outputs.get("markdown"), "standalone Markdown", root=root,
        exact_keys=frozenset({"path", "path_scope", "sha256"}),
        expected_path=manifest_path.parent / "baseline_report.md",
    )
    pdf_path = _bound_file(
        outputs.get("pdf"), "standalone PDF", root=root,
        exact_keys=frozenset({"path", "path_scope", "sha256", "pages"}),
        expected_path=manifest_path.parent / "baseline_report.pdf",
    )
    try:
        markdown = markdown_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("standalone Markdown is not UTF-8") from error
    if not markdown.strip():
        raise ValueError("standalone Markdown is empty")
    pdf = pdf_path.read_bytes()
    if not pdf.startswith(b"%PDF-") or b"%%EOF" not in pdf[-1024:]:
        raise ValueError("standalone PDF container signature differs")
    assert isinstance(outputs["pdf"], Mapping)
    pages = _require_int(outputs["pdf"].get("pages"), "standalone PDF page count", minimum=6)
    return {
        "verification_level": PORTABLE_VERIFICATION_LEVEL,
        "status": "PASS",
        "manifest_sha256": file_sha256(manifest_path),
        "evidence_manifest_sha256": evidence["manifest_sha256"],
        "markdown_sha256": file_sha256(markdown_path),
        "pdf_sha256": file_sha256(pdf_path),
        "registered_pdf_pages": pages,
    }


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def verify_portable_suite(
    output_root: Path | str, report_root: Path | str, *,
    require_complete: bool = False, project_root: Path | None = None,
) -> dict[str, object]:
    """Verify the portable committed-artifact layer and return no outcomes."""

    reject_forbidden_text(output_root, report_root)
    root = workspace_root().resolve() if project_root is None else Path(project_root).resolve()
    output = Path(output_root).resolve()
    reports = Path(report_root).resolve()
    for label, path in (("output", output), ("report", reports)):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"portable {label} root is outside the project root") from error
    paths = {
        "evaluation": output / "evaluation" / "manifest.json",
        "evidence": reports / "source_data" / "manifest.json",
        "report": reports / "report_manifest.json",
    }
    verifiers = {
        "evaluation": lambda path: verify_portable_evaluation_manifest(
            path, project_root=root, allowed_root=output,
        ),
        "evidence": lambda path: verify_portable_evidence_manifest(
            path, project_root=root, expected_evaluation_path=paths["evaluation"],
        ),
        "report": lambda path: verify_portable_report_manifest(
            path, project_root=root, expected_evaluation_path=paths["evaluation"],
        ),
    }
    records: dict[str, object] = {}
    for name, path in paths.items():
        if not path.is_file():
            if require_complete:
                raise FileNotFoundError(f"required portable {name} artifact is absent: {path}")
            records[name] = {"status": "ABSENT"}
            continue
        result = verifiers[name](path)
        records[name] = {
            "status": "PASS",
            "path": _display_path(path, root),
            "sha256": file_sha256(path),
            "verification_level": PORTABLE_VERIFICATION_LEVEL,
            "structural_summary": result,
        }
    return {
        "experiment_id": PROTOCOL_ID,
        "status": "PASS",
        "verification_level": PORTABLE_VERIFICATION_LEVEL,
        "artifacts": records,
        "portable_scope": (
            "Committed manifest/file hashes, registry completeness, and JSON/CSV/Parquet/"
            "PDF/Markdown container schemas only."
        ),
        "non_claims": list(PORTABLE_NON_CLAIMS),
        "strict_local_verification_performed": False,
        "source_replay_audit_performed": False,
        "models_or_corpora_opened": False,
        "raw_market_sessions_opened": False,
        "economic_values_recomputed": False,
        "paper_modified": False,
        "august_accessed": False,
    }


__all__ = [
    "PORTABLE_NON_CLAIMS", "PORTABLE_VERIFICATION_LEVEL",
    "portable_protocol_binding",
    "verify_portable_evaluation_manifest", "verify_portable_evidence_manifest",
    "verify_portable_replay_report", "verify_portable_report_manifest",
    "verify_portable_suite",
]
