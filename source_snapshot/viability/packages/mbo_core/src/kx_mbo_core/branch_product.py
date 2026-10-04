from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file, sha256_json, verify_evidence_manifest
from .config import enforce_2025_frame
from .execution import SelfFinancingLedger
from .feature_product import MANIFEST_FILE_NAME as FEATURE_MANIFEST_FILE_NAME
from .feature_product import FeatureProduct, load_feature_product, verify_feature_product
from .rent_counterfactuals import (
    BASE_RENT_ACTION,
    DENSE_RENT_ACTION_IDS,
    DENSE_RENT_PAYOFF_COLUMNS,
    RENT_ACTION_PREFIX,
    REGISTERED_WORKER_SUBSETS,
    DenseRentBranchResult,
    dense_action_binding_hash,
    dense_branch_binding_hash,
    load_dense_rent_branch_result,
    parse_rent_action_id,
    worker_subset_members,
)
from .types import Fill


SCHEMA_VERSION = 2
PRODUCT_TYPE = "kx_common_tape_branch_product"
ALLOWED_EVIDENCE_CLASSES = frozenset(
    {"DEVELOPMENT_ONLY", "LOCKED_EMPIRICAL", "SYNTHETIC_ONLY"}
)
ALLOWED_LIBRARIES = frozenset({"cash", "rent"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REPLAY_ARTIFACT_MANIFEST_VERSION = 1
BRANCH_TRACE_MANIFEST_TYPE = "kx_branch_trace_artifacts"
POLICY_REPLAY_MANIFEST_TYPE = "kx_policy_replay_artifacts"
POLICY_FILL_LEDGER_TYPE = "kx_policy_fill_ledger"
POLICY_FILL_LEDGER_SCHEMA_VERSION = 1

ROOT_REQUIRED_COLUMNS = (
    "opportunity_id",
    "date",
    "partition",
    "fold",
    "decision_panel",
    "calendar_block",
    "eligible",
    "eligibility_reason",
    "base_action",
    "direction",
    "quantity",
    "publisher_id",
    "instrument_id",
    "raw_symbol",
    "continuity_segment_id",
    "decision_time_ns",
    "available_time_ns",
    "submission_time_ns",
    "arrival_time_ns",
    "terminal_time_ns",
    "terminal_arrival_time_ns",
    "tape_first_ordinal",
    "tape_last_ordinal",
    "feature_join_key",
    "l2_feature_hash",
    "l3_feature_hash",
    "active_contract_map_hash",
    "source_tape_hash",
)

PAYOFF_REQUIRED_COLUMNS = (
    "opportunity_id",
    "date",
    "partition",
    "fold",
    "decision_panel",
    "calendar_block",
    "library",
    "action",
    "scenario",
    "feature_view",
    "feasible",
    "infeasible_reason",
    "net_ticks",
    "base_net_ticks",
    "downside_ticks",
    "filled_quantity",
    "forced_quantity",
    "retains_base_trade",
    "worker_subset",
    "passive_fifo_certified",
    "replay_start_time_ns",
    "replay_end_time_ns",
    "source_tape_hash",
    "branch_trace_hash",
)

POLICY_REQUIRED_COLUMNS = (
    "opportunity_id",
    "date",
    "partition",
    "fold",
    "decision_panel",
    "calendar_block",
    "policy_id",
    "scenario",
    "payoff_library",
    "selected_action",
    "feature_view",
    "net_ticks",
    "base_net_ticks",
    "downside_ticks",
    "intervened",
    "base_trade_retained",
    "replay_complete",
    "model_artifact_hash",
    "policy_config_hash",
    "training_cutoff_time_ns",
    "policy_decision_time_ns",
    "replay_start_time_ns",
    "replay_end_time_ns",
    "policy_trace_hash",
    "source_tape_hash",
)

ROOT_KEY = ("opportunity_id",)
PAYOFF_KEY = ("opportunity_id", "library", "action", "scenario", "feature_view")
POLICY_KEY = ("opportunity_id", "policy_id", "scenario")

_ROOT_CLOCKS = (
    "decision_time_ns",
    "available_time_ns",
    "submission_time_ns",
    "arrival_time_ns",
    "terminal_time_ns",
    "terminal_arrival_time_ns",
)
_PAYOFF_CLOCKS = ("replay_start_time_ns", "replay_end_time_ns")
_POLICY_CLOCKS = (
    "training_cutoff_time_ns",
    "policy_decision_time_ns",
    "replay_start_time_ns",
    "replay_end_time_ns",
)
_ROOT_METADATA = (
    "date",
    "partition",
    "fold",
    "decision_panel",
    "calendar_block",
    "source_tape_hash",
)


class BranchProductError(ValueError):
    """Raised when a common-tape product cannot support an empirical claim."""


REQUIRED_EXTERNAL_CLAIMS = (
    "active_contract_certification",
    "causal_fifo_replay",
    "complete_branch_replay",
    "sequential_policy_replay",
    "development_power",
    "generation_config",
    "source_manifest",
)


@dataclass(frozen=True, order=True)
class PayoffCell:
    library: str
    action: str
    scenario: str = "nominal"
    feature_view: str = "l3"

    def __post_init__(self) -> None:
        if self.library not in ALLOWED_LIBRARIES:
            raise BranchProductError(f"unsupported branch library {self.library!r}")
        for name in ("action", "scenario", "feature_view"):
            if not str(getattr(self, name)).strip():
                raise BranchProductError(f"payoff-cell {name} cannot be empty")


@dataclass(frozen=True, order=True)
class PolicyCell:
    policy_id: str
    scenario: str = "nominal"

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.scenario.strip():
            raise BranchProductError("policy-cell fields cannot be empty")


def empty_policies_frame() -> pd.DataFrame:
    """Return the typed empty policy ledger used before sequential replay."""

    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in POLICY_REQUIRED_COLUMNS})
    for column in ("net_ticks", "base_net_ticks", "downside_ticks"):
        frame[column] = frame[column].astype("float64")
    for column in _POLICY_CLOCKS:
        frame[column] = frame[column].astype("int64")
    for column in ("intervened", "base_trade_retained", "replay_complete"):
        frame[column] = frame[column].astype("bool")
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise BranchProductError(f"{name} is missing columns: {missing}")


def _strict_bool(value: Any, *, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "0", "1"}:
        return value.strip().lower() in {"true", "1"}
    raise BranchProductError(f"{column!r} contains a non-boolean value: {value!r}")


def _strict_boolean_series(values: pd.Series, *, column: str) -> pd.Series:
    return values.map(lambda value: _strict_bool(value, column=column)).astype(bool)


def _require_nonempty_text(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        values = frame[column]
        bad = values.isna() | values.astype(str).str.strip().eq("")
        if bool(bad.any()):
            raise BranchProductError(f"{name}.{column} contains an empty value")


def _require_sha256(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        invalid = ~frame[column].astype(str).map(lambda value: bool(SHA256_PATTERN.fullmatch(value)))
        if bool(invalid.any()):
            sample = frame.loc[invalid, column].head().tolist()
            raise BranchProductError(f"{name}.{column} is not SHA-256: {sample}")


def _coerce_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if columns and not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
        raise BranchProductError(f"{name} contains non-finite numeric values")


def _frame_digest(frame: pd.DataFrame) -> str:
    """Hash values, column order, and dtypes without trusting a caller hash."""

    digest = hashlib.sha256()
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    digest.update(json.dumps(schema, separators=(",", ":")).encode())
    if len(frame):
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def policy_ledger_content_sha256(frame: pd.DataFrame) -> str:
    """Hash a policy ledger using the canonical schema coercions and key order."""

    _require_columns(frame, POLICY_REQUIRED_COLUMNS, "policies")
    output = frame.copy()
    _coerce_finite(
        output, ("net_ticks", "base_net_ticks", "downside_ticks"), "policies"
    )
    for column in _POLICY_CLOCKS:
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    for column in ("intervened", "base_trade_retained", "replay_complete"):
        output[column] = _strict_boolean_series(output[column], column=column)
    canonical = output.sort_values(list(POLICY_KEY), kind="stable").reset_index(drop=True)
    return _frame_digest(canonical)


def root_ledger_content_sha256(frame: pd.DataFrame) -> str:
    """Hash a root ledger using the same coercions used by product validation."""

    _require_columns(frame, ROOT_REQUIRED_COLUMNS, "roots")
    output = frame.copy()
    enforce_2025_frame(output, date_column="date", clock_columns=list(_ROOT_CLOCKS))
    output["eligible"] = _strict_boolean_series(output["eligible"], column="eligible")
    for column in (
        "direction",
        "quantity",
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "tape_first_ordinal",
        "tape_last_ordinal",
        *_ROOT_CLOCKS,
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    canonical = output.sort_values(list(ROOT_KEY), kind="stable").reset_index(drop=True)
    return _frame_digest(canonical)


def payoff_ledger_content_sha256(frame: pd.DataFrame) -> str:
    """Hash a payoff ledger using the same coercions used by product validation."""

    _require_columns(frame, PAYOFF_REQUIRED_COLUMNS, "payoffs")
    output = frame.copy()
    enforce_2025_frame(output, date_column="date", clock_columns=list(_PAYOFF_CLOCKS))
    for column in ("feasible", "retains_base_trade", "passive_fifo_certified"):
        output[column] = _strict_boolean_series(output[column], column=column)
    _coerce_finite(
        output, ("net_ticks", "base_net_ticks", "downside_ticks"), "payoffs"
    )
    for column in ("filled_quantity", "forced_quantity", *_PAYOFF_CLOCKS):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    canonical = output.sort_values(list(PAYOFF_KEY), kind="stable").reset_index(drop=True)
    return _frame_digest(canonical)


def _normalise_cells(values: Iterable[PayoffCell | Mapping[str, Any]]) -> tuple[PayoffCell, ...]:
    cells = tuple(
        value if isinstance(value, PayoffCell) else PayoffCell(**dict(value))
        for value in values
    )
    if not cells or len(cells) != len(set(cells)):
        raise BranchProductError("registered payoff cells must be nonempty and unique")
    return tuple(sorted(cells))


def _normalise_policy_cells(
    values: Iterable[PolicyCell | Mapping[str, Any]],
) -> tuple[PolicyCell, ...]:
    cells = tuple(
        value if isinstance(value, PolicyCell) else PolicyCell(**dict(value))
        for value in values
    )
    if len(cells) != len(set(cells)):
        raise BranchProductError("registered policy cells must be unique")
    return tuple(sorted(cells))


def _claim(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    claims = manifest.get("claims")
    if not isinstance(claims, Mapping) or not isinstance(claims.get(name), Mapping):
        raise BranchProductError(f"branch product is missing claim {name!r}")
    return claims[name]


def _claim_passed(manifest: Mapping[str, Any], name: str) -> bool:
    claim = _claim(manifest, name)
    if not isinstance(claim.get("passed"), bool):
        raise BranchProductError(f"claim {name!r} needs a JSON boolean 'passed'")
    return bool(claim["passed"])


def _claim_evidence_hash(manifest: Mapping[str, Any], name: str) -> str:
    value = str(_claim(manifest, name).get("evidence_sha256", ""))
    if not SHA256_PATTERN.fullmatch(value):
        raise BranchProductError(
            f"claim {name!r} must bind an external evidence artifact by evidence_sha256"
        )
    return value


def _require_artifact_manifest_binding(
    claim: Mapping[str, Any], field: str, *, claim_name: str
) -> None:
    binding = claim.get(field)
    if not isinstance(binding, Mapping):
        raise BranchProductError(
            f"claim {claim_name!r} must bind external {field}"
        )
    if not str(binding.get("path", "")).strip():
        raise BranchProductError(
            f"claim {claim_name!r}.{field} needs a path"
        )
    if not SHA256_PATTERN.fullmatch(str(binding.get("sha256", ""))):
        raise BranchProductError(
            f"claim {claim_name!r}.{field} needs a lowercase SHA-256"
        )


def _manifest_has_dense_rent_lattice(manifest: Mapping[str, Any]) -> bool:
    values = manifest.get("registered_payoff_cells")
    if not isinstance(values, list):
        return False
    actions = [
        str(value.get("action", ""))
        for value in values
        if isinstance(value, Mapping) and str(value.get("library", "")) == "rent"
    ]
    return any(action.startswith(f"{RENT_ACTION_PREFIX}|") for action in actions)


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    if int(manifest.get("schema_version", 0)) != SCHEMA_VERSION:
        raise BranchProductError("unsupported common-tape product schema")
    if str(manifest.get("product_type", "")) != PRODUCT_TYPE:
        raise BranchProductError("manifest is not a common-tape branch product")
    if str(manifest.get("evidence_class", "")) not in ALLOWED_EVIDENCE_CLASSES:
        raise BranchProductError(
            "unsupported common-tape branch-product evidence class"
        )
    for field in ("product_id", "data_version", "config_hash", "source_manifest_sha256"):
        value = str(manifest.get(field, ""))
        if not value:
            raise BranchProductError(f"manifest field {field!r} cannot be empty")
    if not SHA256_PATTERN.fullmatch(str(manifest["config_hash"])):
        raise BranchProductError("manifest config_hash must be SHA-256")
    if not SHA256_PATTERN.fullmatch(str(manifest["source_manifest_sha256"])):
        raise BranchProductError("manifest source_manifest_sha256 must be SHA-256")
    for name in REQUIRED_EXTERNAL_CLAIMS[:5]:
        _claim_evidence_hash(manifest, name)
    if not _claim_passed(manifest, "active_contract_certification"):
        raise BranchProductError("active-contract certification did not pass")
    if not _claim_passed(manifest, "causal_fifo_replay"):
        raise BranchProductError("causal FIFO replay did not pass")
    complete = _claim(manifest, "complete_branch_replay")
    if not _claim_passed(manifest, "complete_branch_replay") or complete.get("complete") is not True:
        raise BranchProductError("complete branch replay is absent or incomplete")
    for field in (
        "root_ledger_content_sha256",
        "payoff_ledger_content_sha256",
        "trace_artifact_set_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(complete.get(field, ""))):
            raise BranchProductError(
                f"complete branch replay must bind {field}"
            )
    if _manifest_has_dense_rent_lattice(manifest) and not SHA256_PATTERN.fullmatch(
        str(complete.get("continuation_trace_artifact_set_sha256", ""))
    ):
        raise BranchProductError(
            "dense RENT branch replay must bind continuation_trace_artifact_set_sha256"
        )
    external_artifacts_required = str(manifest["evidence_class"]) != "SYNTHETIC_ONLY"
    if external_artifacts_required:
        _require_artifact_manifest_binding(
            complete,
            "trace_artifact_manifest",
            claim_name="complete_branch_replay",
        )
    sequential = _claim(manifest, "sequential_policy_replay")
    if not isinstance(sequential.get("complete"), bool):
        raise BranchProductError("sequential_policy_replay.complete must be boolean")
    if not isinstance(sequential.get("passed"), bool):
        raise BranchProductError("sequential_policy_replay.passed must be boolean")
    if bool(sequential["passed"]) != bool(sequential["complete"]):
        raise BranchProductError(
            "sequential_policy_replay passed/complete flags must agree"
        )
    power = _claim(manifest, "development_power")
    if not isinstance(power.get("passed"), bool):
        raise BranchProductError("development_power.passed must be boolean")
    value = power.get("value")
    if power["passed"] and value is None:
        raise BranchProductError("passed development power needs a numeric value")
    if value is not None:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise BranchProductError("development power must be numeric") from error
        if not np.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise BranchProductError("development power must lie in [0, 1]")
    if bool(sequential["complete"]):
        sequential_digest_fields = [
            "policy_ledger_content_sha256",
            "trace_artifact_set_sha256",
        ]
        if external_artifacts_required:
            sequential_digest_fields.extend(
                (
                    "model_artifact_set_sha256",
                    "policy_config_artifact_set_sha256",
                )
            )
        for field in sequential_digest_fields:
            if not SHA256_PATTERN.fullmatch(str(sequential.get(field, ""))):
                raise BranchProductError(
                    f"complete sequential replay must bind {field}"
                )
        if external_artifacts_required:
            _require_artifact_manifest_binding(
                sequential,
                "policy_artifact_manifest",
                claim_name="sequential_policy_replay",
            )
        specs = sequential.get("policy_specs")
        if not isinstance(specs, list) or not specs:
            raise BranchProductError(
                "complete sequential replay must register policy provenance specs"
            )
        registered = {
            (str(cell.policy_id), str(cell.scenario))
            for cell in _registered_policy_cells(manifest)
        }
        observed: set[tuple[str, str]] = set()
        registered_payoffs = _registered_payoff_cells(manifest)
        for spec in specs:
            if not isinstance(spec, Mapping):
                raise BranchProductError("each policy provenance spec must be an object")
            key = (str(spec.get("policy_id", "")), str(spec.get("scenario", "")))
            if not all(key):
                raise BranchProductError("policy provenance spec needs policy_id and scenario")
            if key in observed:
                raise BranchProductError(f"duplicate policy provenance spec {key!r}")
            observed.add(key)
            payoff_library = str(spec.get("payoff_library", ""))
            if payoff_library not in ALLOWED_LIBRARIES:
                raise BranchProductError(
                    f"policy provenance spec {key!r} needs a registered payoff_library"
                )
            allowed_actions = spec.get("allowed_actions")
            allowed_views = spec.get("allowed_feature_views")
            if (
                not isinstance(allowed_actions, list)
                or not allowed_actions
                or not all(str(value).strip() for value in allowed_actions)
                or len({str(value) for value in allowed_actions}) != len(allowed_actions)
            ):
                raise BranchProductError(
                    f"policy provenance spec {key!r} needs nonempty allowed_actions"
                )
            if (
                not isinstance(allowed_views, list)
                or not allowed_views
                or not all(str(value).strip() for value in allowed_views)
                or len({str(value) for value in allowed_views}) != len(allowed_views)
            ):
                raise BranchProductError(
                    f"policy provenance spec {key!r} needs nonempty allowed_feature_views"
                )
            supported_pairs = {
                (cell.action, cell.feature_view)
                for cell in registered_payoffs
                if cell.library == payoff_library and cell.scenario == key[1]
            }
            declared_actions = {str(value) for value in allowed_actions}
            declared_views = {str(value) for value in allowed_views}
            unsupported_actions = sorted(
                action
                for action in declared_actions
                if not any(
                    candidate_action == action and candidate_view in declared_views
                    for candidate_action, candidate_view in supported_pairs
                )
            )
            unsupported_views = sorted(
                view
                for view in declared_views
                if not any(
                    candidate_view == view and candidate_action in declared_actions
                    for candidate_action, candidate_view in supported_pairs
                )
            )
            if unsupported_actions or unsupported_views:
                raise BranchProductError(
                    f"policy provenance spec {key!r} is not backed by registered "
                    f"payoff cells; actions={unsupported_actions}, views={unsupported_views}"
                )
        if observed != registered:
            raise BranchProductError(
                "policy provenance specs must exactly match registered policy cells"
            )


def _validate_roots(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    _require_columns(frame, ROOT_REQUIRED_COLUMNS, "roots")
    if frame.empty:
        raise BranchProductError("roots cannot be empty")
    output = frame.copy()
    enforce_2025_frame(output, date_column="date", clock_columns=list(_ROOT_CLOCKS))
    if output.duplicated(list(ROOT_KEY)).any():
        raise BranchProductError("roots must contain exactly one row per opportunity_id")
    _require_nonempty_text(
        output,
        (
            "opportunity_id",
            "date",
            "partition",
            "fold",
            "decision_panel",
            "calendar_block",
            "base_action",
            "raw_symbol",
            "feature_join_key",
        ),
        "roots",
    )
    _require_sha256(
        output,
        ("l2_feature_hash", "l3_feature_hash", "active_contract_map_hash", "source_tape_hash"),
        "roots",
    )
    output["eligible"] = _strict_boolean_series(output["eligible"], column="eligible")
    output["direction"] = pd.to_numeric(output["direction"], errors="raise").astype("int64")
    output["quantity"] = pd.to_numeric(output["quantity"], errors="raise").astype("int64")
    if not output["direction"].isin((-1, 1)).all():
        raise BranchProductError("root direction must be -1 or 1")
    if not output["quantity"].eq(1).all():
        raise BranchProductError("the certified frozen-tape product is restricted to one contract")
    for column in (
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "tape_first_ordinal",
        "tape_last_ordinal",
        *_ROOT_CLOCKS,
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    bad_clock = ~(
        (output["available_time_ns"] <= output["decision_time_ns"])
        & (output["decision_time_ns"] <= output["submission_time_ns"])
        & (output["submission_time_ns"] <= output["arrival_time_ns"])
        & (output["arrival_time_ns"] < output["terminal_time_ns"])
        & (output["terminal_time_ns"] <= output["terminal_arrival_time_ns"])
    )
    if bool(bad_clock.any()):
        raise BranchProductError("root causal clocks are not monotonically ordered")
    if not (output["tape_first_ordinal"] <= output["tape_last_ordinal"]).all():
        raise BranchProductError("root tape ordinal bounds are reversed")
    ineligible_missing = ~output["eligible"] & output["eligibility_reason"].astype(str).str.strip().eq("")
    if bool(ineligible_missing.any()):
        raise BranchProductError("ineligible roots need an auditable eligibility_reason")
    active_claim = _claim(manifest, "active_contract_certification")
    active_hash = str(active_claim.get("map_sha256", ""))
    if not SHA256_PATTERN.fullmatch(active_hash):
        raise BranchProductError("active-contract claim must bind the map by map_sha256")
    if active_claim.get("past_only") is not True:
        raise BranchProductError("active-contract certification must be explicitly past-only")
    try:
        lag_sessions = int(active_claim.get("minimum_lag_sessions", 0))
    except (TypeError, ValueError) as error:
        raise BranchProductError("active-contract minimum_lag_sessions must be an integer") from error
    if lag_sessions < 1:
        raise BranchProductError("active-contract map needs at least one completed-session lag")
    if not output["active_contract_map_hash"].astype(str).eq(active_hash).all():
        raise BranchProductError(
            "a root is not bound to the certified past-only active-contract map"
        )
    assignments = active_claim.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise BranchProductError("active-contract claim must contain dated symbol assignments")
    active_by_date: dict[str, str] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise BranchProductError("active-contract assignment must be an object")
        assignment_date = str(assignment.get("date", ""))
        assignment_symbol = str(assignment.get("raw_symbol", ""))
        if not assignment_date or not assignment_symbol:
            raise BranchProductError("active-contract assignment needs date and raw_symbol")
        if assignment_date in active_by_date:
            raise BranchProductError(f"duplicate active-contract date {assignment_date!r}")
        active_by_date[assignment_date] = assignment_symbol
    for row in output.itertuples(index=False):
        expected_symbol = active_by_date.get(str(row.date))
        if expected_symbol is None or expected_symbol != str(row.raw_symbol):
            raise BranchProductError(
                f"root {row.opportunity_id!r} is not on its certified active contract"
            )

    fifo_claim = _claim(manifest, "causal_fifo_replay")
    _claim_evidence_hash(manifest, "causal_fifo_replay")
    periods = fifo_claim.get("periods")
    if not isinstance(periods, Mapping) or not periods:
        raise BranchProductError("causal FIFO claim must enumerate certified tape periods")
    zero_counters = (
        "sequence_gap_count",
        "ambiguous_group_count",
        "unfinished_group_count",
        "recovery_boundary_count",
    )
    for row in output.itertuples(index=False):
        tape_hash = str(row.source_tape_hash)
        period = periods.get(tape_hash)
        if not isinstance(period, Mapping) or period.get("certified") is not True:
            raise BranchProductError(
                f"root {row.opportunity_id!r} has no certified common-tape period"
            )
        for counter in zero_counters:
            value = period.get(counter)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise BranchProductError(f"certified tape period needs integer {counter}")
            if int(value) != 0:
                raise BranchProductError(
                    f"root {row.opportunity_id!r} crosses an uncertified tape: "
                    f"{counter}={value}"
                )
        try:
            first_ordinal = int(period["first_ordinal"])
            last_ordinal = int(period["last_ordinal"])
            start_time_ns = int(period["start_time_ns"])
            end_time_ns = int(period["end_time_ns"])
        except (KeyError, TypeError, ValueError) as error:
            raise BranchProductError("certified tape period has invalid bounds") from error
        if not (
            first_ordinal <= int(row.tape_first_ordinal)
            <= int(row.tape_last_ordinal)
            <= last_ordinal
        ):
            raise BranchProductError(
                f"root {row.opportunity_id!r} ordinal window exceeds its certified tape"
            )
        if not (
            start_time_ns <= int(row.available_time_ns)
            <= int(row.terminal_arrival_time_ns)
            <= end_time_ns
        ):
            raise BranchProductError(
                f"root {row.opportunity_id!r} clocks exceed its certified tape"
            )
    return output.sort_values(["decision_time_ns", "opportunity_id"], kind="stable").reset_index(drop=True)


def _metadata_matches_roots(
    frame: pd.DataFrame, roots: pd.DataFrame, *, name: str
) -> None:
    root_metadata = roots.set_index("opportunity_id")[list(_ROOT_METADATA)]
    child = frame.set_index("opportunity_id")
    for column in _ROOT_METADATA:
        expected = child.index.map(root_metadata[column])
        observed = child[column].astype(str).to_numpy()
        if not np.array_equal(observed, np.asarray(expected, dtype=str)):
            raise BranchProductError(f"{name}.{column} disagrees with its root")


def _registered_payoff_cells(manifest: Mapping[str, Any]) -> tuple[PayoffCell, ...]:
    values = manifest.get("registered_payoff_cells")
    if not isinstance(values, list):
        raise BranchProductError("manifest registered_payoff_cells must be a list")
    return _normalise_cells(values)


def _registered_policy_cells(manifest: Mapping[str, Any]) -> tuple[PolicyCell, ...]:
    values = manifest.get("registered_policy_cells")
    if not isinstance(values, list):
        raise BranchProductError("manifest registered_policy_cells must be a list")
    return _normalise_policy_cells(values)


def _validate_registered_dense_rent_lattice(manifest: Mapping[str, Any]) -> bool:
    rent_cells = [
        cell for cell in _registered_payoff_cells(manifest) if cell.library == "rent"
    ]
    dense = [
        cell
        for cell in rent_cells
        if cell.action.startswith(f"{RENT_ACTION_PREFIX}|")
    ]
    if not dense:
        return False
    if len(dense) != len(rent_cells):
        raise BranchProductError(
            "a dense RENT product cannot mix legacy and rent.v1 action identifiers"
        )
    expected_actions = set(DENSE_RENT_ACTION_IDS)
    grouped: dict[tuple[str, str], set[str]] = {}
    for cell in dense:
        try:
            parse_rent_action_id(cell.action)
        except ValueError as error:
            raise BranchProductError(str(error)) from error
        grouped.setdefault((cell.scenario, cell.feature_view), set()).add(cell.action)
    for key, observed in grouped.items():
        if observed != expected_actions:
            missing = sorted(expected_actions - observed)
            extra = sorted(observed - expected_actions)
            raise BranchProductError(
                f"registered dense RENT lattice {key!r} is incomplete; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    return True


def _validate_rent_subset_and_action_bindings(
    output: pd.DataFrame,
    roots: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> bool:
    rent = output["library"].astype(str).eq("rent")
    for value in output.loc[rent, "worker_subset"].astype(str):
        try:
            worker_subset_members(value)
        except ValueError as error:
            raise BranchProductError(str(error)) from error
    dense = _validate_registered_dense_rent_lattice(manifest)
    if not dense:
        return False
    _require_columns(output, DENSE_RENT_PAYOFF_COLUMNS, "dense RENT payoffs")
    dense_rows = output.loc[rent]
    _require_sha256(
        dense_rows,
        (
            "action_binding_hash",
            "branch_binding_hash",
            "continuation_trace_hash",
        ),
        "dense RENT payoffs",
    )
    root_index = roots.set_index("opportunity_id")
    for row in dense_rows.itertuples(index=False):
        try:
            contract = parse_rent_action_id(str(row.action))
        except ValueError as error:
            raise BranchProductError(str(error)) from error
        if str(row.worker_subset) != contract.worker_subset:
            raise BranchProductError(
                "dense RENT worker_subset disagrees with its canonical action identifier"
            )
        observed_primitives = (
            str(row.entry_action),
            str(row.execution_action),
            str(row.protection_action),
        )
        expected_primitives = (
            contract.entry_action,
            contract.execution_action,
            contract.protection_action,
        )
        if observed_primitives != expected_primitives:
            raise BranchProductError(
                "dense RENT primitive actions disagree with the action identifier"
            )
        root = root_index.loc[str(row.opportunity_id)]
        expected_action_binding = dense_action_binding_hash(
            opportunity_id=str(row.opportunity_id),
            scenario=str(row.scenario),
            action=str(row.action),
            publisher_id=int(root["publisher_id"]),
            instrument_id=int(root["instrument_id"]),
            raw_symbol=str(root["raw_symbol"]),
            continuity_segment_id=int(root["continuity_segment_id"]),
            source_tape_hash=str(row.source_tape_hash),
            continuation_trace_hash=str(row.continuation_trace_hash),
        )
        if str(row.action_binding_hash) != expected_action_binding:
            raise BranchProductError(
                "dense RENT action binding does not bind its root/subset/actions/continuation"
            )
        expected_branch_binding = dense_branch_binding_hash(
            action_binding_hash=expected_action_binding,
            branch_trace_hash=str(row.branch_trace_hash),
        )
        if str(row.branch_binding_hash) != expected_branch_binding:
            raise BranchProductError(
                "dense RENT branch binding does not bind subset, action, and trace"
            )
    for key, group in dense_rows.groupby(
        ["opportunity_id", "scenario"], sort=False
    ):
        if set(group["worker_subset"].astype(str)) != set(REGISTERED_WORKER_SUBSETS):
            raise BranchProductError(
                f"dense RENT root/scenario {key!r} does not cover all eight subsets"
            )
        if group["continuation_trace_hash"].astype(str).nunique() != 1:
            raise BranchProductError(
                f"dense RENT root/scenario {key!r} does not share one independent continuation"
            )
        action_traces = group.groupby("action", sort=False)["branch_trace_hash"].nunique()
        if not action_traces.eq(1).all():
            raise BranchProductError(
                f"dense RENT root/scenario {key!r} relabels an action across traces"
            )
        unique_action_traces = group.drop_duplicates("action")["branch_trace_hash"]
        if unique_action_traces.nunique() != len(DENSE_RENT_ACTION_IDS):
            raise BranchProductError(
                f"dense RENT root/scenario {key!r} does not bind each action to a distinct trace"
            )
    return True


def _validate_payoffs(
    frame: pd.DataFrame, roots: pd.DataFrame, manifest: Mapping[str, Any]
) -> pd.DataFrame:
    _require_columns(frame, PAYOFF_REQUIRED_COLUMNS, "payoffs")
    if frame.empty:
        raise BranchProductError("payoffs cannot be empty")
    output = frame.copy()
    enforce_2025_frame(output, date_column="date", clock_columns=list(_PAYOFF_CLOCKS))
    if output.duplicated(list(PAYOFF_KEY)).any():
        raise BranchProductError(f"payoffs must be unique by {PAYOFF_KEY}")
    unknown_roots = sorted(set(output["opportunity_id"]) - set(roots["opportunity_id"]))
    if unknown_roots:
        raise BranchProductError(f"payoffs reference unknown roots: {unknown_roots[:5]}")
    _require_nonempty_text(
        output,
        (
            "opportunity_id",
            "library",
            "action",
            "scenario",
            "feature_view",
            "branch_trace_hash",
            "source_tape_hash",
        ),
        "payoffs",
    )
    _require_sha256(output, ("branch_trace_hash", "source_tape_hash"), "payoffs")
    unknown_libraries = sorted(set(output["library"].astype(str)) - ALLOWED_LIBRARIES)
    if unknown_libraries:
        raise BranchProductError(f"unregistered payoff libraries: {unknown_libraries}")
    dense_rent = _validate_rent_subset_and_action_bindings(output, roots, manifest)
    for column in ("feasible", "retains_base_trade", "passive_fifo_certified"):
        output[column] = _strict_boolean_series(output[column], column=column)
    _coerce_finite(output, ("net_ticks", "base_net_ticks", "downside_ticks"), "payoffs")
    expected_downside = np.maximum(
        0.0,
        output["base_net_ticks"].to_numpy(dtype=float)
        - output["net_ticks"].to_numpy(dtype=float),
    )
    if not np.allclose(
        output["downside_ticks"].to_numpy(dtype=float),
        expected_downside,
        rtol=0.0,
        atol=1e-12,
    ):
        raise BranchProductError(
            "payoff downside_ticks must equal max(0, base_net_ticks - net_ticks)"
        )
    for column in ("filled_quantity", "forced_quantity", *_PAYOFF_CLOCKS):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    if (output[["downside_ticks", "filled_quantity", "forced_quantity"]] < 0).any().any():
        raise BranchProductError("payoff downside and quantities must be nonnegative")
    if not (output["replay_start_time_ns"] <= output["replay_end_time_ns"]).all():
        raise BranchProductError("payoff replay clock bounds are reversed")
    root_index = roots.set_index("opportunity_id")
    expected_start = output["opportunity_id"].map(root_index["available_time_ns"])
    expected_end = output["opportunity_id"].map(root_index["terminal_arrival_time_ns"])
    if not output["replay_start_time_ns"].eq(expected_start).all():
        raise BranchProductError(
            "every shadow payoff must restart at its root's available causal state"
        )
    if not output["replay_end_time_ns"].eq(expected_end).all():
        raise BranchProductError(
            "every shadow payoff must stop at its root's common terminal arrival"
        )
    root_quantity = output["opportunity_id"].map(root_index["quantity"])
    if bool((output["forced_quantity"] > root_quantity).any()):
        raise BranchProductError("forced payoff quantity exceeds the frozen root quantity")
    passive = output["action"].astype(str).str.contains("passive|join", case=False, regex=True)
    if bool((output["feasible"] & passive & ~output["passive_fifo_certified"]).any()):
        raise BranchProductError("a feasible passive payoff lacks actual FIFO certification")
    infeasible = ~output["feasible"]
    if bool((infeasible & output["infeasible_reason"].astype(str).str.strip().eq("")).any()):
        raise BranchProductError("infeasible actions need a reason")
    fallback_bad = infeasible & (
        ~np.isclose(output["net_ticks"], output["base_net_ticks"], rtol=0.0, atol=1e-12)
        | ~output["retains_base_trade"]
    )
    if bool(fallback_bad.any()):
        raise BranchProductError("infeasible action rows must be exact Base fallbacks")
    root_eligible = output["opportunity_id"].map(root_index["eligible"]).astype(bool)
    ineligible_bad = ~root_eligible & (
        ~np.isclose(output["net_ticks"], output["base_net_ticks"], rtol=0.0, atol=1e-12)
        | ~output["retains_base_trade"]
    )
    if bool(ineligible_bad.any()):
        raise BranchProductError(
            "all actions at an ineligible root must be exact Base fallbacks"
        )
    root_base_action = output["opportunity_id"].map(root_index["base_action"]).astype(str)
    nonbase_at_ineligible = (
        ~root_eligible
        & ~output["action"].astype(str).eq(root_base_action)
        & ~output["action"].astype(str).eq("base")
        & ~output["action"].astype(str).eq(BASE_RENT_ACTION.action_id)
    )
    if bool((nonbase_at_ineligible & output["feasible"]).any()):
        raise BranchProductError(
            "a non-Base action at an ineligible root must be explicitly infeasible"
        )
    _metadata_matches_roots(output, roots, name="payoffs")
    invariant_fields = [
        "feasible",
        "infeasible_reason",
        "net_ticks",
        "base_net_ticks",
        "downside_ticks",
        "filled_quantity",
        "forced_quantity",
        "retains_base_trade",
        "worker_subset",
        "passive_fifo_certified",
        "replay_start_time_ns",
        "replay_end_time_ns",
        "source_tape_hash",
        "branch_trace_hash",
    ]
    if dense_rent:
        invariant_fields.extend(DENSE_RENT_PAYOFF_COLUMNS)
    for key, action_group in output.groupby(
        ["opportunity_id", "library", "action", "scenario"], sort=False
    ):
        changed = [
            column
            for column in invariant_fields
            if action_group[column].nunique(dropna=False) != 1
        ]
        if changed:
            raise BranchProductError(
                f"feature_view changes counterfactual economics for {key!r}: {changed}"
            )
    expected_cells = {
        (cell.library, cell.action, cell.scenario, cell.feature_view)
        for cell in _registered_payoff_cells(manifest)
    }
    for opportunity_id, group in output.groupby("opportunity_id", sort=False):
        observed_cells = set(
            group[["library", "action", "scenario", "feature_view"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if observed_cells != expected_cells:
            missing = sorted(expected_cells - observed_cells)
            extra = sorted(observed_cells - expected_cells)
            raise BranchProductError(
                f"root {opportunity_id!r} has an incomplete payoff lattice; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        base_action = str(
            roots.loc[roots["opportunity_id"].eq(opportunity_id), "base_action"].iloc[0]
        )
        for scenario, scenario_group in group.groupby("scenario", sort=False):
            if scenario_group["base_net_ticks"].nunique(dropna=False) != 1:
                raise BranchProductError(
                    f"Base payoff changes within root {opportunity_id!r}, scenario {scenario!r}"
                )
            candidates = scenario_group.loc[
                scenario_group["action"].astype(str).isin(
                    (base_action, "base", BASE_RENT_ACTION.action_id)
                )
            ]
            if (
                candidates.empty
                or not np.isclose(
                    candidates["net_ticks"],
                    candidates["base_net_ticks"],
                    rtol=0.0,
                    atol=1e-12,
                ).all()
                or not candidates["feasible"].all()
                or not candidates["retains_base_trade"].all()
            ):
                raise BranchProductError(
                    f"root {opportunity_id!r}, scenario {scenario!r} has no feasible exact Base action row"
                )
    return output.sort_values(list(PAYOFF_KEY), kind="stable").reset_index(drop=True)


def _validate_policies(
    frame: pd.DataFrame,
    roots: pd.DataFrame,
    payoffs: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    _require_columns(frame, POLICY_REQUIRED_COLUMNS, "policies")
    output = frame.copy()
    sequential = _claim(manifest, "sequential_policy_replay")
    complete = bool(sequential["complete"])
    registered = _registered_policy_cells(manifest)
    if not complete:
        if not output.empty:
            raise BranchProductError(
                "partial policy outcomes are forbidden; keep policies empty until independent replay completes"
            )
        return empty_policies_frame()
    if not _claim_passed(manifest, "sequential_policy_replay"):
        raise BranchProductError("complete sequential replay must have passed certification")
    if output.empty or not registered:
        raise BranchProductError("complete sequential replay needs registered policy rows")
    enforce_2025_frame(
        output, date_column="date", clock_columns=list(_POLICY_CLOCKS)
    )
    if output.duplicated(list(POLICY_KEY)).any():
        raise BranchProductError(f"policies must be unique by {POLICY_KEY}")
    unknown_roots = sorted(set(output["opportunity_id"]) - set(roots["opportunity_id"]))
    if unknown_roots:
        raise BranchProductError(f"policies reference unknown roots: {unknown_roots[:5]}")
    _require_nonempty_text(
        output,
        (
            "opportunity_id",
            "policy_id",
            "scenario",
            "payoff_library",
            "selected_action",
            "feature_view",
            "policy_trace_hash",
            "source_tape_hash",
        ),
        "policies",
    )
    _require_sha256(
        output,
        (
            "model_artifact_hash",
            "policy_config_hash",
            "policy_trace_hash",
            "source_tape_hash",
        ),
        "policies",
    )
    _coerce_finite(output, ("net_ticks", "base_net_ticks", "downside_ticks"), "policies")
    expected_downside = np.maximum(
        0.0,
        output["base_net_ticks"].to_numpy(dtype=float)
        - output["net_ticks"].to_numpy(dtype=float),
    )
    if not np.allclose(
        output["downside_ticks"].to_numpy(dtype=float),
        expected_downside,
        rtol=0.0,
        atol=1e-12,
    ):
        raise BranchProductError(
            "policy downside_ticks must equal max(0, base_net_ticks - net_ticks)"
        )
    for column in _POLICY_CLOCKS:
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    if (output["downside_ticks"] < 0).any():
        raise BranchProductError("policy downside must be nonnegative")
    for column in ("intervened", "base_trade_retained", "replay_complete"):
        output[column] = _strict_boolean_series(output[column], column=column)
    if not output["replay_complete"].all():
        raise BranchProductError("every policy row must come from a complete sequential replay")
    _metadata_matches_roots(output, roots, name="policies")
    root_index = roots.set_index("opportunity_id")
    expected_decision = output["opportunity_id"].map(root_index["decision_time_ns"])
    expected_start = output["opportunity_id"].map(root_index["available_time_ns"])
    expected_end = output["opportunity_id"].map(root_index["terminal_arrival_time_ns"])
    if not output["policy_decision_time_ns"].eq(expected_decision).all():
        raise BranchProductError("policy decisions must occur at the frozen root decision clock")
    if not output["replay_start_time_ns"].eq(expected_start).all():
        raise BranchProductError("policy replay must restart at the root causal state")
    if not output["replay_end_time_ns"].eq(expected_end).all():
        raise BranchProductError("policy replay must end at the root common terminal")
    if not (output["training_cutoff_time_ns"] < output["policy_decision_time_ns"]).all():
        raise BranchProductError("policy training cutoff must strictly precede its decision")
    locked = output["partition"].astype(str).eq("locked_validation")
    if bool(locked.any()) and int(output.loc[locked, "training_cutoff_time_ns"].max()) >= int(
        output.loc[locked, "policy_decision_time_ns"].min()
    ):
        raise BranchProductError(
            "all locked-validation policies must be frozen before the first locked decision"
        )

    sequential_specs = {
        (str(spec["policy_id"]), str(spec["scenario"])): spec
        for spec in sequential["policy_specs"]
    }
    payoff_actions = {
        (str(opportunity_id), str(scenario)): group
        for (opportunity_id, scenario), group in payoffs.groupby(
            ["opportunity_id", "scenario"], sort=False
        )
    }
    for row in output.itertuples(index=False):
        key = (str(row.policy_id), str(row.scenario))
        spec = sequential_specs[key]
        if str(row.payoff_library) != str(spec["payoff_library"]):
            raise BranchProductError(
                f"policy {key!r} disagrees with its registered payoff library"
            )
        if str(row.selected_action) not in {
            str(value) for value in spec["allowed_actions"]
        }:
            raise BranchProductError(
                f"policy {key!r} selected unregistered action {row.selected_action!r}"
            )
        if str(row.feature_view) not in {
            str(value) for value in spec["allowed_feature_views"]
        }:
            raise BranchProductError(
                f"policy {key!r} used unregistered feature view {row.feature_view!r}"
            )
        candidates = payoff_actions.get((str(row.opportunity_id), str(row.scenario)))
        if candidates is None:
            raise BranchProductError(
                f"policy {key!r} has no same-root/scenario action-payoff support"
            )
        selected = candidates.loc[
            candidates["library"].astype(str).eq(str(row.payoff_library))
            & candidates["action"].astype(str).eq(str(row.selected_action))
            & candidates["feature_view"].astype(str).eq(str(row.feature_view))
        ]
        if len(selected) != 1:
            raise BranchProductError(
                f"policy {key!r} must resolve to exactly one same-root payoff cell"
            )
        payoff = selected.iloc[0]
        if not bool(payoff["feasible"]) and not (
            np.isclose(float(row.net_ticks), float(row.base_net_ticks), rtol=0.0, atol=1e-12)
            and bool(row.base_trade_retained)
            and not bool(row.intervened)
        ):
            raise BranchProductError(
                "an infeasible selected policy action must execute an exact Base fallback"
            )
        if bool(payoff["feasible"]):
            economic_pairs = (
                (float(row.net_ticks), float(payoff["net_ticks"]), "net_ticks"),
                (
                    float(row.base_net_ticks),
                    float(payoff["base_net_ticks"]),
                    "base_net_ticks",
                ),
                (
                    float(row.downside_ticks),
                    float(payoff["downside_ticks"]),
                    "downside_ticks",
                ),
            )
            for observed, expected, field in economic_pairs:
                if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
                    raise BranchProductError(
                        f"feasible policy {key!r} {field} differs from its selected payoff trace"
                    )
            if bool(row.base_trade_retained) != bool(payoff["retains_base_trade"]):
                raise BranchProductError(
                    f"feasible policy {key!r} retention flag differs from its selected payoff"
                )
            root_base_action = str(root_index.loc[str(row.opportunity_id), "base_action"])
            expected_intervened = str(row.selected_action) not in {
                root_base_action,
                "base",
                BASE_RENT_ACTION.action_id,
            }
            if bool(row.intervened) != expected_intervened:
                raise BranchProductError(
                    f"feasible policy {key!r} intervention flag disagrees with its action"
                )

    root_eligible = output["opportunity_id"].map(root_index["eligible"]).astype(bool)
    ineligible_bad = ~root_eligible & (
        ~np.isclose(output["net_ticks"], output["base_net_ticks"], rtol=0.0, atol=1e-12)
        | output["intervened"]
        | ~output["base_trade_retained"]
    )
    if bool(ineligible_bad.any()):
        raise BranchProductError(
            "every policy at an ineligible root must be an exact non-intervening Base fallback"
        )
    expected_cells = {(cell.policy_id, cell.scenario) for cell in registered}
    for opportunity_id, group in output.groupby("opportunity_id", sort=False):
        observed = set(
            group[["policy_id", "scenario"]].astype(str).itertuples(index=False, name=None)
        )
        if observed != expected_cells:
            raise BranchProductError(
                f"root {opportunity_id!r} has an incomplete sequential-policy lattice"
            )
        for scenario, scenario_group in group.groupby("scenario", sort=False):
            if scenario_group["base_net_ticks"].nunique(dropna=False) != 1:
                raise BranchProductError(
                    f"sequential Base payoff changes within root {opportunity_id!r}, "
                    f"scenario {scenario!r}"
                )
            base = scenario_group.loc[scenario_group["policy_id"].eq("base")]
            if not base.empty and not np.isclose(
                base["net_ticks"], base["base_net_ticks"], rtol=0.0, atol=1e-12
            ).all():
                raise BranchProductError("the sequential Base policy is not an exact Base replay")
    canonical = output.sort_values(list(POLICY_KEY), kind="stable").reset_index(drop=True)
    observed_policy_digest = _frame_digest(canonical)
    if observed_policy_digest != str(sequential["policy_ledger_content_sha256"]):
        raise BranchProductError(
            "sequential policy ledger disagrees with its independent replay certificate"
        )
    return canonical


def validate_branch_product(
    roots: pd.DataFrame,
    payoffs: pd.DataFrame,
    policies: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate all evidence and return canonical defensive frame copies."""

    _validate_manifest_header(manifest)
    clean_roots = _validate_roots(roots, manifest)
    clean_payoffs = _validate_payoffs(payoffs, clean_roots, manifest)
    complete = _claim(manifest, "complete_branch_replay")
    if root_ledger_content_sha256(clean_roots) != str(
        complete["root_ledger_content_sha256"]
    ):
        raise BranchProductError(
            "root ledger disagrees with the complete-branch replay certificate"
        )
    if payoff_ledger_content_sha256(clean_payoffs) != str(
        complete["payoff_ledger_content_sha256"]
    ):
        raise BranchProductError(
            "payoff ledger disagrees with the complete-branch replay certificate"
        )
    clean_policies = _validate_policies(
        policies, clean_roots, clean_payoffs, manifest
    )
    return clean_roots, clean_payoffs, clean_policies


class BranchProduct:
    """Immutable-by-interface common-tape product for CASH and RENT adapters."""

    def __init__(
        self,
        roots: pd.DataFrame,
        payoffs: pd.DataFrame,
        policies: pd.DataFrame,
        manifest: Mapping[str, Any],
        *,
        validate: bool = True,
    ) -> None:
        if validate:
            roots, payoffs, policies = validate_branch_product(
                roots, payoffs, policies, manifest
            )
        self._roots = roots.copy(deep=True)
        self._payoffs = payoffs.copy(deep=True)
        self._policies = policies.copy(deep=True)
        self._manifest = copy.deepcopy(dict(manifest))

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def hashes(self) -> dict[str, str]:
        return copy.deepcopy(dict(self._manifest.get("content_hashes", {})))

    def roots_frame(self) -> pd.DataFrame:
        return self._roots.copy(deep=True)

    def payoffs_frame(self) -> pd.DataFrame:
        return self._payoffs.copy(deep=True)

    def policies_frame(self) -> pd.DataFrame:
        return self._policies.copy(deep=True)


def create_branch_product(
    roots: pd.DataFrame,
    payoffs: pd.DataFrame,
    *,
    product_id: str,
    data_version: str,
    config_hash: str,
    source_manifest_sha256: str,
    claims: Mapping[str, Any],
    registered_payoff_cells: Iterable[PayoffCell | Mapping[str, Any]],
    policies: pd.DataFrame | None = None,
    registered_policy_cells: Iterable[PolicyCell | Mapping[str, Any]] = (),
    evidence_class: str = "DEVELOPMENT_ONLY",
) -> BranchProduct:
    payoff_cells = _normalise_cells(registered_payoff_cells)
    policy_cells = _normalise_policy_cells(registered_policy_cells)
    policy_frame = empty_policies_frame() if policies is None else policies.copy()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "product_type": PRODUCT_TYPE,
        "product_id": str(product_id),
        "evidence_class": str(evidence_class),
        "data_version": str(data_version),
        "config_hash": str(config_hash),
        "source_manifest_sha256": str(source_manifest_sha256),
        "registered_payoff_cells": [asdict(cell) for cell in payoff_cells],
        "registered_policy_cells": [asdict(cell) for cell in policy_cells],
        "claims": copy.deepcopy(dict(claims)),
    }
    clean_roots, clean_payoffs, clean_policies = validate_branch_product(
        roots, payoffs, policy_frame, manifest
    )
    manifest["row_counts"] = {
        "roots": int(len(clean_roots)),
        "payoffs": int(len(clean_payoffs)),
        "policies": int(len(clean_policies)),
    }
    manifest["content_hashes"] = {
        "roots": _frame_digest(clean_roots),
        "payoffs": _frame_digest(clean_payoffs),
        "policies": _frame_digest(clean_policies),
    }
    return BranchProduct(clean_roots, clean_payoffs, clean_policies, manifest)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_branch_product(path: str | Path, product: BranchProduct) -> Path:
    """Atomically create a new product directory; existing products are never overwritten."""

    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"branch product already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        frames = {
            "roots": product.roots_frame(),
            "payoffs": product.payoffs_frame(),
            "policies": product.policies_frame(),
        }
        files: dict[str, dict[str, Any]] = {}
        for name, frame in frames.items():
            filename = f"{name}.parquet"
            file_path = temporary / filename
            frame.to_parquet(file_path, index=False)
            files[name] = {
                "path": filename,
                "sha256": sha256_file(file_path),
                "rows": int(len(frame)),
            }
        manifest = product.manifest
        manifest["files"] = files
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    return target


def load_branch_product(path: str | Path, *, verify: bool = True) -> BranchProduct:
    source = Path(path).expanduser().resolve()
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_header(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise BranchProductError("on-disk product manifest has no files mapping")
    frames: dict[str, pd.DataFrame] = {}
    for name in ("roots", "payoffs", "policies"):
        declaration = files.get(name)
        if not isinstance(declaration, Mapping) or declaration.get("path") != f"{name}.parquet":
            raise BranchProductError(f"invalid fixed product path for {name!r}")
        file_path = source / f"{name}.parquet"
        if not file_path.is_file():
            raise BranchProductError(f"missing product file: {file_path}")
        if verify:
            expected = str(declaration.get("sha256", ""))
            observed = sha256_file(file_path)
            if observed != expected:
                raise BranchProductError(
                    f"product file hash mismatch for {name!r}: expected {expected}, observed {observed}"
                )
        frame = pd.read_parquet(file_path)
        if int(declaration.get("rows", -1)) != len(frame):
            raise BranchProductError(f"product row count mismatch for {name!r}")
        frames[name] = frame
    product = BranchProduct(
        frames["roots"], frames["payoffs"], frames["policies"], manifest
    )
    if verify:
        expected_hashes = manifest.get("content_hashes")
        if not isinstance(expected_hashes, Mapping):
            raise BranchProductError("product manifest has no content hashes")
        observed = {
            "roots": _frame_digest(product.roots_frame()),
            "payoffs": _frame_digest(product.payoffs_frame()),
            "policies": _frame_digest(product.policies_frame()),
        }
        if observed != dict(expected_hashes):
            raise BranchProductError("decoded product content hashes do not match manifest")
    return product


def _resolve_relative_artifact(owner: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = owner.parent / candidate
    return candidate.resolve()


def _verify_replay_artifact_manifest(
    *,
    certificate_path: Path,
    claim: Mapping[str, Any],
    binding_field: str,
    manifest_type: str,
    expected_references: Mapping[str, set[str]],
    digest_fields: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve and hash every byte referenced by a replay certificate.

    Row hashes are treated as file-content hashes.  The externally hashed JSON
    certificate binds a separate artifact manifest; that manifest in turn binds
    every branch trace, policy trace, serialized model, and policy config.  Exact
    set equality prevents an omitted referenced artifact or an unrelated extra
    artifact from being presented as complete replay evidence.
    """

    _require_artifact_manifest_binding(
        claim, binding_field, claim_name=str(manifest_type)
    )
    binding = claim[binding_field]
    manifest_path = _resolve_relative_artifact(
        certificate_path, str(binding["path"])
    )
    if not manifest_path.is_file():
        raise BranchProductError(
            f"replay artifact manifest does not exist: {manifest_path}"
        )
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != str(binding["sha256"]):
        raise BranchProductError(
            f"replay artifact manifest hash mismatch: expected {binding['sha256']}, "
            f"observed {observed_manifest_sha256}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BranchProductError(
            f"replay artifact manifest is not readable JSON: {manifest_path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise BranchProductError("replay artifact manifest must be a JSON object")
    if int(payload.get("schema_version", 0)) != REPLAY_ARTIFACT_MANIFEST_VERSION:
        raise BranchProductError(
            "replay artifact manifest has an unsupported schema_version"
        )
    if str(payload.get("manifest_type", "")) != manifest_type:
        raise BranchProductError(
            f"replay artifact manifest must have manifest_type={manifest_type!r}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BranchProductError("replay artifact manifest needs a nonempty artifacts list")

    expected_kinds = set(expected_references)
    if expected_kinds != set(digest_fields):
        raise AssertionError("artifact reference kinds and digest fields disagree")
    observed: dict[str, set[str]] = {kind: set() for kind in expected_kinds}
    verified: list[dict[str, str]] = []
    observed_paths: set[Path] = set()
    for index, declaration in enumerate(artifacts):
        if not isinstance(declaration, Mapping):
            raise BranchProductError(
                f"replay artifact declaration #{index} must be an object"
            )
        kind = str(declaration.get("kind", ""))
        if kind not in expected_kinds:
            raise BranchProductError(
                f"replay artifact declaration #{index} has unsupported kind {kind!r}"
            )
        raw_path = str(declaration.get("path", "")).strip()
        expected_sha256 = str(declaration.get("sha256", ""))
        if not raw_path or not SHA256_PATTERN.fullmatch(expected_sha256):
            raise BranchProductError(
                f"replay artifact declaration #{index} needs path and SHA-256"
            )
        artifact_path = _resolve_relative_artifact(manifest_path, raw_path)
        if artifact_path in observed_paths:
            raise BranchProductError(
                f"replay artifact manifest repeats path: {artifact_path}"
            )
        if expected_sha256 in observed[kind]:
            raise BranchProductError(
                f"replay artifact manifest repeats {kind} hash: {expected_sha256}"
            )
        if not artifact_path.is_file():
            raise BranchProductError(
                f"referenced replay artifact does not exist: {artifact_path}"
            )
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != expected_sha256:
            raise BranchProductError(
                f"replay artifact hash mismatch for {artifact_path}: "
                f"expected {expected_sha256}, observed {actual_sha256}"
            )
        observed_paths.add(artifact_path)
        observed[kind].add(actual_sha256)
        verified.append(
            {"kind": kind, "path": str(artifact_path), "sha256": actual_sha256}
        )

    for kind, expected_hashes in expected_references.items():
        if observed[kind] != expected_hashes:
            missing = sorted(expected_hashes - observed[kind])
            extra = sorted(observed[kind] - expected_hashes)
            raise BranchProductError(
                f"replay artifact manifest does not exactly cover {kind}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        canonical_set_sha256 = sha256_json(sorted(observed[kind]))
        digest_field = digest_fields[kind]
        if canonical_set_sha256 != str(claim.get(digest_field, "")):
            raise BranchProductError(
                f"replay artifact {kind} set disagrees with {digest_field}"
            )
    return {
        "manifest_type": manifest_type,
        "manifest_path": str(manifest_path),
        "manifest_sha256": observed_manifest_sha256,
        "artifact_count": len(verified),
        "artifacts": sorted(
            verified, key=lambda value: (value["kind"], value["sha256"], value["path"])
        ),
    }


def _verify_feature_product_binding(
    *,
    branch_product: BranchProduct,
    feature_product_path: str | Path,
    require_empirical_derivation: bool = False,
) -> dict[str, Any]:
    """Resolve every root to the exact hash-verified causal feature row."""

    source = Path(feature_product_path).expanduser().resolve()
    feature_product: FeatureProduct = load_feature_product(source)
    branch_manifest = branch_product.manifest
    feature_manifest = dict(feature_product.manifest)
    for field in ("evidence_class", "data_version", "source_manifest_sha256"):
        if str(feature_manifest.get(field, "")) != str(branch_manifest.get(field, "")):
            raise BranchProductError(
                f"feature product {field} disagrees with the branch product"
            )

    roots = branch_product.roots_frame().copy()
    features = feature_product.features.copy()
    root_keys = roots["feature_join_key"].astype(str)
    feature_keys = features["feature_join_key"].astype(str)
    if root_keys.duplicated().any():
        raise BranchProductError("branch roots repeat a feature_join_key")
    if set(root_keys) != set(feature_keys):
        missing = sorted(set(root_keys) - set(feature_keys))
        extra = sorted(set(feature_keys) - set(root_keys))
        raise BranchProductError(
            "feature product and branch roots do not have an exact join-key set; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    feature_index = features.assign(
        feature_join_key=feature_keys
    ).set_index("feature_join_key")
    exact_text = (
        "raw_symbol",
        "source_tape_hash",
        "l2_feature_hash",
        "l3_feature_hash",
    )
    exact_integer = (
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "available_time_ns",
        "decision_time_ns",
    )
    for root in roots.itertuples(index=False):
        key = str(root.feature_join_key)
        feature = feature_index.loc[key]
        if isinstance(feature, pd.DataFrame):
            raise BranchProductError(f"feature join {key!r} is not one-to-one")
        for column in exact_text:
            if str(getattr(root, column)) != str(feature[column]):
                raise BranchProductError(
                    f"root {root.opportunity_id!r} {column} disagrees with its feature row"
                )
        for column in exact_integer:
            if int(getattr(root, column)) != int(feature[column]):
                raise BranchProductError(
                    f"root {root.opportunity_id!r} {column} disagrees with its feature row"
                )
        if int(feature["feature_time_ns"]) > int(root.decision_time_ns):
            raise BranchProductError(
                f"root {root.opportunity_id!r} is bound to a future feature"
            )

    empirical_derivation: Mapping[str, Any] | None = None
    if require_empirical_derivation and str(feature_manifest.get("evidence_class")) != "SYNTHETIC_ONLY":
        try:
            verified_product = verify_feature_product(
                source, require_empirical_derivation=True
            )
        except (OSError, TypeError, ValueError) as error:
            raise BranchProductError(
                f"empirical feature derivation verification failed: {error}"
            ) from error
        candidate = verified_product.get("feature_derivation")
        if not isinstance(candidate, Mapping) or candidate.get("recomputed") is not True:
            raise BranchProductError(
                "empirical feature product lacks recomputed derivation evidence"
            )
        empirical_derivation = candidate

    manifest_path = source / FEATURE_MANIFEST_FILE_NAME
    return {
        "product_id": str(feature_manifest["product_id"]),
        "evidence_class": str(feature_manifest["evidence_class"]),
        "data_version": str(feature_manifest["data_version"]),
        "row_count": int(len(features)),
        "feature_registry_sha256": str(
            feature_manifest["feature_registry_sha256"]
        ),
        "feature_frame_content_sha256": str(
            feature_manifest["feature_frame_content_sha256"]
        ),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "feature_derivation": empirical_derivation,
    }


def _authenticate_active_contract_certificate(
    *,
    certificate: Mapping[str, Any],
    certificate_path: Path,
    evidence_class: str,
) -> dict[str, Any] | None:
    """Route empirical v2 roll evidence through its causal recomputation boundary."""

    declared_v2 = (
        int(certificate.get("schema_version", 0)) == 2
        and str(certificate.get("certificate_type", ""))
        == "PAST_ONLY_ACTIVE_CONTRACT_SELECTION"
    )
    if not declared_v2:
        if evidence_class != "SYNTHETIC_ONLY":
            raise BranchProductError(
                "empirical active-contract evidence must be an authenticated "
                "PAST_ONLY_ACTIVE_CONTRACT_SELECTION v2 certificate"
            )
        return None
    # Import the module at the trust boundary so tests and downstream audit
    # harnesses can prove that the exported verifier was actually invoked.
    from . import active_contract_v2

    try:
        authenticated = active_contract_v2.verify_active_contract_certificate(
            certificate_path
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise BranchProductError(
            f"active-contract v2 authentication failed: {error}"
        ) from error
    if not isinstance(authenticated, Mapping) or authenticated.get("passed") is not True:
        raise BranchProductError("authenticated active-contract v2 certificate did not pass")
    canonical_loaded = json.dumps(
        certificate, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    canonical_authenticated = json.dumps(
        authenticated, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if canonical_loaded != canonical_authenticated:
        raise BranchProductError(
            "active-contract certificate differs from its authenticated recomputation"
        )
    return {
        "certificate_path": str(certificate_path),
        "certificate_sha256": sha256_file(certificate_path),
        "certificate_type": str(authenticated["certificate_type"]),
        "schema_version": int(authenticated["schema_version"]),
        "map_sha256": str(authenticated["map_sha256"]),
    }


def _assert_dense_result_matches_branch_product(
    result: DenseRentBranchResult,
    branch_product: BranchProduct,
) -> None:
    """Prove the typed dense artifact is the RENT portion of this product."""

    branch_roots = branch_product.roots_frame().sort_values(
        ["decision_time_ns", "opportunity_id"], kind="stable"
    ).reset_index(drop=True)
    dense_roots = result.roots.copy()
    missing_root_columns = sorted(set(branch_roots.columns) - set(dense_roots.columns))
    if missing_root_columns:
        raise BranchProductError(
            "typed dense RENT roots do not carry the branch-root provenance: "
            f"{missing_root_columns}"
        )
    dense_roots = dense_roots.loc[:, branch_roots.columns].sort_values(
        ["decision_time_ns", "opportunity_id"], kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            branch_roots,
            dense_roots,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise BranchProductError(
            "typed dense RENT roots differ from the branch-product roots"
        ) from error

    branch_rent = (
        branch_product.payoffs_frame()
        .loc[lambda frame: frame["library"].astype(str).eq("rent")]
        .sort_values(list(PAYOFF_KEY), kind="stable")
        .reset_index(drop=True)
    )
    if branch_rent.empty:
        raise BranchProductError("typed dense RENT evidence has no RENT payoff cells")
    dense_payoffs = result.payoffs.copy()
    missing_payoff_columns = sorted(
        set(branch_rent.columns) - set(dense_payoffs.columns)
    )
    if missing_payoff_columns:
        raise BranchProductError(
            "typed dense RENT payoffs omit branch-product columns: "
            f"{missing_payoff_columns}"
        )
    dense_payoffs = dense_payoffs.loc[:, branch_rent.columns].sort_values(
        list(PAYOFF_KEY), kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            branch_rent,
            dense_payoffs,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise BranchProductError(
            "typed dense RENT payoffs differ from the branch-product RENT cells"
        ) from error


def _verify_complete_branch_artifact_manifest(
    *,
    certificate_path: Path,
    claim: Mapping[str, Any],
    branch_product: BranchProduct,
) -> dict[str, Any]:
    """Verify typed dense RENT frames and any remaining byte-bound traces."""

    _require_artifact_manifest_binding(
        claim,
        "trace_artifact_manifest",
        claim_name="complete_branch_replay",
    )
    binding = claim["trace_artifact_manifest"]
    manifest_path = _resolve_relative_artifact(certificate_path, str(binding["path"]))
    if not manifest_path.is_file():
        raise BranchProductError(
            f"replay artifact manifest does not exist: {manifest_path}"
        )
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != str(binding["sha256"]):
        raise BranchProductError("complete-branch artifact manifest hash mismatch")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BranchProductError(
            "complete-branch artifact manifest must be readable JSON"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("schema_version", 0)) != REPLAY_ARTIFACT_MANIFEST_VERSION
        or str(payload.get("manifest_type", "")) != BRANCH_TRACE_MANIFEST_TYPE
    ):
        raise BranchProductError("unsupported complete-branch artifact manifest")
    declarations = payload.get("artifacts")
    if not isinstance(declarations, list) or not declarations:
        raise BranchProductError("complete-branch artifact manifest is empty")

    payoffs = branch_product.payoffs_frame()
    dense_rent = _manifest_has_dense_rent_lattice(branch_product.manifest)
    dense_hashes = set(
        payoffs.loc[
            payoffs["library"].astype(str).eq("rent"), "branch_trace_hash"
        ].astype(str)
    ) if dense_rent else set()
    remaining_hashes = set(payoffs["branch_trace_hash"].astype(str)) - dense_hashes
    observed_remaining: set[str] = set()
    dense_declarations: list[tuple[Path, str]] = []
    verified: list[dict[str, str]] = []
    observed_paths: set[Path] = set()
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, Mapping):
            raise BranchProductError(f"branch artifact #{index} must be an object")
        kind = str(declaration.get("kind", ""))
        if kind not in {"branch_trace", "dense_rent_result"}:
            raise BranchProductError(
                f"branch artifact #{index} has unsupported kind {kind!r}"
            )
        raw_path = str(declaration.get("path", "")).strip()
        expected_hash = str(declaration.get("sha256", ""))
        if not raw_path or not SHA256_PATTERN.fullmatch(expected_hash):
            raise BranchProductError(
                f"branch artifact #{index} needs a path and SHA-256"
            )
        artifact_path = _resolve_relative_artifact(manifest_path, raw_path)
        if artifact_path in observed_paths:
            raise BranchProductError(f"branch artifact path is repeated: {artifact_path}")
        if not artifact_path.is_file():
            raise BranchProductError(f"branch artifact does not exist: {artifact_path}")
        observed_hash = sha256_file(artifact_path)
        if observed_hash != expected_hash:
            raise BranchProductError(
                f"branch artifact hash mismatch for {artifact_path}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        observed_paths.add(artifact_path)
        verified.append(
            {"kind": kind, "path": str(artifact_path), "sha256": observed_hash}
        )
        if kind == "branch_trace":
            observed_remaining.add(observed_hash)
        else:
            dense_declarations.append((artifact_path, observed_hash))

    if observed_remaining != remaining_hashes:
        raise BranchProductError(
            "complete-branch artifacts do not exactly cover the non-dense traces"
        )
    if dense_rent:
        if len(dense_declarations) != 1:
            raise BranchProductError(
                "dense RENT evidence requires exactly one typed dense_rent_result"
            )
        dense_manifest_path, _ = dense_declarations[0]
        if dense_manifest_path.name != "manifest.json":
            raise BranchProductError(
                "dense_rent_result must reference its manifest.json"
            )
        try:
            dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BranchProductError("dense RENT manifest is not readable JSON") from error
        if str(dense_manifest.get("evidence_class", "")) != str(
            branch_product.manifest["evidence_class"]
        ):
            raise BranchProductError(
                "dense RENT artifact evidence class disagrees with the branch product"
            )
        try:
            dense_result = load_dense_rent_branch_result(dense_manifest_path.parent)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise BranchProductError(f"invalid typed dense RENT artifact: {error}") from error
        _assert_dense_result_matches_branch_product(dense_result, branch_product)
    elif dense_declarations:
        raise BranchProductError(
            "non-dense branch product cannot borrow a dense RENT artifact"
        )

    expected_trace_set_hash = sha256_json(
        sorted(set(payoffs["branch_trace_hash"].astype(str)))
    )
    if expected_trace_set_hash != str(claim.get("trace_artifact_set_sha256", "")):
        raise BranchProductError(
            "complete-branch logical trace set disagrees with its certificate"
        )
    if dense_rent:
        continuation_hash = sha256_json(
            sorted(
                set(
                    payoffs.loc[
                        payoffs["library"].astype(str).eq("rent"),
                        "continuation_trace_hash",
                    ].astype(str)
                )
            )
        )
        if continuation_hash != str(
            claim.get("continuation_trace_artifact_set_sha256", "")
        ):
            raise BranchProductError(
                "dense RENT continuation set disagrees with its certificate"
            )
    return {
        "manifest_type": BRANCH_TRACE_MANIFEST_TYPE,
        "manifest_path": str(manifest_path),
        "manifest_sha256": observed_manifest_sha256,
        "artifact_count": len(verified),
        "typed_dense_rent": dense_rent,
        "artifacts": sorted(
            verified, key=lambda value: (value["kind"], value["sha256"], value["path"])
        ),
    }


def _verify_typed_policy_fill_ledgers(
    policies: pd.DataFrame,
    verified_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse policy traces and recompute their flat-to-flat economics.

    This deliberately rejects plain text and arbitrary model logs.  Empirical
    sequential claims must provide one JSON fill ledger per policy row; until a
    producer emits this schema, the claim remains blocked rather than trusted.
    """

    declarations = [
        item
        for item in verified_manifest.get("artifacts", ())
        if str(item.get("kind", "")) == "policy_trace"
    ]
    by_hash = {str(item["sha256"]): Path(str(item["path"])) for item in declarations}
    if len(by_hash) != len(declarations):
        raise BranchProductError("policy fill-ledger artifact hashes must be unique")
    if policies["policy_trace_hash"].astype(str).duplicated().any():
        raise BranchProductError(
            "typed policy fill ledgers require one trace hash per policy row"
        )
    required_fields = {
        "schema_version",
        "artifact_type",
        "tick_value_usd",
        "opportunity_id",
        "policy_id",
        "scenario",
        "payoff_library",
        "selected_action",
        "feature_view",
        "source_tape_hash",
        "policy_decision_time_ns",
        "replay_start_time_ns",
        "replay_end_time_ns",
        "fills",
    }
    fill_fields = {
        "time_ns",
        "signed_quantity",
        "price_ticks",
        "fee_usd",
        "liquidity",
    }
    verified_rows: list[dict[str, Any]] = []
    persistent: dict[tuple[str, str], SelfFinancingLedger] = {}
    prior_end: dict[tuple[str, str], int] = {}
    ordered = policies.sort_values(
        ["policy_id", "scenario", "policy_decision_time_ns", "opportunity_id"],
        kind="stable",
    )
    for row in ordered.itertuples(index=False):
        trace_hash = str(row.policy_trace_hash)
        artifact_path = by_hash.get(trace_hash)
        if artifact_path is None:
            raise BranchProductError(
                f"policy row {row.opportunity_id!r}/{row.policy_id!r} has no typed fill ledger"
            )
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BranchProductError(
                f"policy trace must be a typed JSON fill ledger: {artifact_path}"
            ) from error
        if not isinstance(payload, Mapping) or set(payload) != required_fields:
            raise BranchProductError(
                f"policy fill ledger has an invalid top-level schema: {artifact_path}"
            )
        if (
            int(payload["schema_version"]) != POLICY_FILL_LEDGER_SCHEMA_VERSION
            or str(payload["artifact_type"]) != POLICY_FILL_LEDGER_TYPE
        ):
            raise BranchProductError("unsupported policy fill-ledger schema")
        exact_text = (
            "opportunity_id",
            "policy_id",
            "scenario",
            "payoff_library",
            "selected_action",
            "feature_view",
            "source_tape_hash",
        )
        for field in exact_text:
            if str(payload[field]) != str(getattr(row, field)):
                raise BranchProductError(
                    f"policy fill ledger {field} disagrees with its policy row"
                )
        exact_clocks = (
            "policy_decision_time_ns",
            "replay_start_time_ns",
            "replay_end_time_ns",
        )
        for field in exact_clocks:
            if int(payload[field]) != int(getattr(row, field)):
                raise BranchProductError(
                    f"policy fill ledger {field} disagrees with its policy row"
                )
        try:
            tick_value_usd = float(payload["tick_value_usd"])
        except (TypeError, ValueError) as error:
            raise BranchProductError("policy tick_value_usd must be numeric") from error
        if not np.isfinite(tick_value_usd) or tick_value_usd <= 0:
            raise BranchProductError("policy tick_value_usd must be finite and positive")
        fills = payload["fills"]
        if not isinstance(fills, list):
            raise BranchProductError("policy fill ledger fills must be a list")
        key = (str(row.policy_id), str(row.scenario))
        start = int(row.replay_start_time_ns)
        decision = int(row.policy_decision_time_ns)
        end = int(row.replay_end_time_ns)
        if key in prior_end and start < prior_end[key]:
            raise BranchProductError(
                "typed sequential policy ledgers overlap within a policy/scenario"
            )
        prior_end[key] = end
        ledger = persistent.setdefault(key, SelfFinancingLedger(tick_value_usd=tick_value_usd))
        if not np.isclose(ledger.tick_value_usd, tick_value_usd, rtol=0.0, atol=0.0):
            raise BranchProductError("policy tick value changes within a sequential ledger")
        before = float(ledger.cash_usd)
        prior_fill_time = decision
        for index, fill_value in enumerate(fills):
            if not isinstance(fill_value, Mapping) or set(fill_value) != fill_fields:
                raise BranchProductError(
                    f"policy fill #{index} has an invalid schema"
                )
            try:
                time_ns = int(fill_value["time_ns"])
                signed_quantity = int(fill_value["signed_quantity"])
                price_ticks = int(fill_value["price_ticks"])
                fee_usd = float(fill_value["fee_usd"])
            except (TypeError, ValueError) as error:
                raise BranchProductError(f"policy fill #{index} is not numeric") from error
            if signed_quantity == 0:
                raise BranchProductError("policy fill quantity cannot be zero")
            if not np.isfinite(fee_usd) or fee_usd < 0:
                raise BranchProductError("policy fill fee must be finite and nonnegative")
            if not decision <= time_ns <= end or time_ns < prior_fill_time:
                raise BranchProductError("policy fill violates its causal replay clock")
            prior_fill_time = time_ns
            liquidity = str(fill_value["liquidity"]).strip()
            if not liquidity:
                raise BranchProductError("policy fill liquidity cannot be empty")
            ledger.apply_fill(
                Fill(
                    time_ns=time_ns,
                    signed_quantity=signed_quantity,
                    price_ticks=price_ticks,
                    fee_usd=fee_usd,
                    liquidity=liquidity,
                )
            )
        if ledger.position != 0:
            raise BranchProductError("policy fill ledger does not restore flat inventory")
        observed_ticks = (float(ledger.cash_usd) - before) / tick_value_usd
        if not np.isclose(observed_ticks, float(row.net_ticks), rtol=0.0, atol=1e-12):
            raise BranchProductError(
                "policy net_ticks differs from its self-financing fill ledger"
            )
        if bool(row.base_trade_retained) != bool(fills):
            raise BranchProductError(
                "policy retention flag disagrees with its typed fill ledger"
            )
        verified_rows.append(
            {
                "opportunity_id": str(row.opportunity_id),
                "policy_id": str(row.policy_id),
                "scenario": str(row.scenario),
                "policy_trace_hash": trace_hash,
                "fill_count": len(fills),
                "net_ticks": observed_ticks,
            }
        )
    return {
        "schema_version": POLICY_FILL_LEDGER_SCHEMA_VERSION,
        "artifact_type": POLICY_FILL_LEDGER_TYPE,
        "row_count": len(verified_rows),
        "rows": verified_rows,
    }


def verify_branch_product_evidence(
    product_path: str | Path,
    evidence_manifest_path: str | Path,
    *,
    pilot_config_path: str | Path,
    split_config_path: str | Path,
    feature_product_path: str | Path,
    product: BranchProduct | None = None,
) -> dict[str, Any]:
    """Bind an empirical branch product and every upstream claim to real bytes.

    The product manifest deliberately stores semantic claims, while the external
    evidence manifest stores the independently addressable artifact paths.  An
    SHA-shaped string inside ``manifest.json`` is not certification: this helper
    resolves and hashes every claim artifact before an empirical pilot can run.
    """

    source = Path(product_path).expanduser().resolve()
    manifest_path = source / "manifest.json"
    disk_product = load_branch_product(source, verify=True)
    if product is not None:
        same_manifest = product.manifest == disk_product.manifest
        same_frames = all(
            _frame_digest(candidate) == _frame_digest(observed)
            for candidate, observed in (
                (product.roots_frame(), disk_product.roots_frame()),
                (product.payoffs_frame(), disk_product.payoffs_frame()),
                (product.policies_frame(), disk_product.policies_frame()),
            )
        )
        if not same_manifest or not same_frames:
            raise BranchProductError(
                "caller-supplied product does not match the hash-verified disk product"
            )
    current = disk_product
    manifest = current.manifest
    feature_source = Path(feature_product_path).expanduser().resolve()
    feature_manifest_path = feature_source / FEATURE_MANIFEST_FILE_NAME
    if not feature_manifest_path.is_file():
        raise BranchProductError(
            "feature_product_path must contain a hash-verified manifest.json"
        )
    evidence = verify_evidence_manifest(
        evidence_manifest_path,
        inputs={
            "branch_product_manifest": manifest_path,
            "feature_product_manifest": feature_manifest_path,
            "pilot_config": pilot_config_path,
            "splits": split_config_path,
        },
        required_claims=REQUIRED_EXTERNAL_CLAIMS,
    )
    if str(evidence.get("evidence_class", "")) != str(manifest["evidence_class"]):
        raise BranchProductError(
            "external evidence class disagrees with the branch product"
        )
    if str(evidence.get("data_version", "")) != str(manifest["data_version"]):
        raise BranchProductError(
            "external evidence data_version disagrees with the branch product"
        )
    verified_feature_product = _verify_feature_product_binding(
        branch_product=current,
        feature_product_path=feature_source,
        require_empirical_derivation=True,
    )

    external_claims = evidence["claims"]
    semantic_fields = {
        "active_contract_certification": (
            "passed",
            "past_only",
            "minimum_lag_sessions",
            "assignments",
            "map_sha256",
        ),
        "causal_fifo_replay": ("passed", "periods"),
        "complete_branch_replay": (
            "passed",
            "complete",
            "root_ledger_content_sha256",
            "payoff_ledger_content_sha256",
            "trace_artifact_set_sha256",
            "continuation_trace_artifact_set_sha256",
            "trace_artifact_manifest",
        ),
        "sequential_policy_replay": (
            "passed",
            "complete",
            "policy_specs",
            "policy_ledger_content_sha256",
            "trace_artifact_set_sha256",
            "model_artifact_set_sha256",
            "policy_config_artifact_set_sha256",
            "policy_artifact_manifest",
        ),
        "development_power": ("passed", "value"),
    }
    certificates: dict[str, tuple[Mapping[str, Any], Path]] = {}
    verified_active_contract: dict[str, Any] | None = None
    for name in REQUIRED_EXTERNAL_CLAIMS[:5]:
        internal = _claim(manifest, name)
        external = external_claims[name]
        if str(internal["evidence_sha256"]) != str(external["sha256"]):
            raise BranchProductError(
                f"branch claim {name!r} is not bound to its verified external artifact"
            )
        if bool(internal["passed"]) != bool(external["passed"]):
            raise BranchProductError(
                f"branch claim {name!r} disagrees with its external certificate"
            )
        for field in ("complete", "value"):
            if field in internal and internal.get(field) != external.get(field):
                raise BranchProductError(
                    f"branch claim {name!r}.{field} disagrees with its external certificate"
                )
        artifact_path = Path(str(external["artifact"]))
        try:
            certificate = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise BranchProductError(
                f"branch claim {name!r} must bind a readable JSON certificate"
            ) from error
        if not isinstance(certificate, Mapping):
            raise BranchProductError(
                f"branch claim {name!r} certificate must be a JSON object"
            )
        certificates[name] = (certificate, artifact_path)
        for field in semantic_fields[name]:
            if internal.get(field) != certificate.get(field):
                raise BranchProductError(
                    f"branch claim {name!r}.{field} disagrees with its hashed certificate"
                )
        if name == "active_contract_certification":
            verified_active_contract = _authenticate_active_contract_certificate(
                certificate=certificate,
                certificate_path=artifact_path,
                evidence_class=str(manifest["evidence_class"]),
            )

    verified_replay_artifacts: dict[str, Any] = {
        "required": str(manifest["evidence_class"]) != "SYNTHETIC_ONLY"
    }
    if verified_replay_artifacts["required"]:
        branch_certificate, branch_certificate_path = certificates[
            "complete_branch_replay"
        ]
        verified_replay_artifacts["complete_branch_replay"] = (
            _verify_complete_branch_artifact_manifest(
                certificate_path=branch_certificate_path,
                claim=branch_certificate,
                branch_product=current,
            )
        )
        sequential = _claim(manifest, "sequential_policy_replay")
        if bool(sequential["complete"]):
            policy_certificate, policy_certificate_path = certificates[
                "sequential_policy_replay"
            ]
            policies = current.policies_frame()
            verified_policy_artifacts = _verify_replay_artifact_manifest(
                    certificate_path=policy_certificate_path,
                    claim=policy_certificate,
                    binding_field="policy_artifact_manifest",
                    manifest_type=POLICY_REPLAY_MANIFEST_TYPE,
                    expected_references={
                        "policy_trace": set(
                            policies["policy_trace_hash"].astype(str)
                        ),
                        "model": set(policies["model_artifact_hash"].astype(str)),
                        "policy_config": set(
                            policies["policy_config_hash"].astype(str)
                        ),
                    },
                    digest_fields={
                        "policy_trace": "trace_artifact_set_sha256",
                        "model": "model_artifact_set_sha256",
                        "policy_config": "policy_config_artifact_set_sha256",
                    },
            )
            verified_policy_artifacts["typed_fill_ledgers"] = (
                _verify_typed_policy_fill_ledgers(
                    policies,
                    verified_policy_artifacts,
                )
            )
            verified_replay_artifacts["sequential_policy_replay"] = (
                verified_policy_artifacts
            )
    if str(external_claims["generation_config"]["sha256"]) != str(
        manifest["config_hash"]
    ):
        raise BranchProductError(
            "branch product config_hash is not bound to the verified generation config"
        )
    if str(external_claims["source_manifest"]["sha256"]) != str(
        manifest["source_manifest_sha256"]
    ):
        raise BranchProductError(
            "branch product source_manifest_sha256 is not externally verified"
        )
    for name in ("generation_config", "source_manifest"):
        if external_claims[name]["passed"] is not True:
            raise BranchProductError(f"external binding {name!r} did not pass")
    return {
        "branch_product": manifest,
        "external_evidence": evidence,
        "verified_active_contract": verified_active_contract,
        "verified_feature_product": verified_feature_product,
        "verified_replay_artifacts": verified_replay_artifacts,
        "branch_product_manifest_sha256": sha256_file(manifest_path),
    }
