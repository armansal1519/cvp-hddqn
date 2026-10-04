from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contract import ACTIONS, BaselineContract, contract_sha256
from .sessions import require_exact_globex_session_dates


TRACE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TRACE_COLUMNS = (
    "decision_id",
    "policy_seed",
    "date",
    "raw_symbol",
    "continuity_segment_id",
    "state_segment_id",
    "reward_boundary_id",
    "bar_end_time_ns",
    "available_time_ns",
    "decision_time_ns",
    "submission_time_ns",
    "arrival_time_ns",
    "h1_action",
    "m15_action",
    "m5_action",
    "h1_position",
    "m15_position",
    "final_target",
    "m5_decision_taken",
    "h1_state_sha256",
    "m15_state_sha256",
    "m5_state_sha256",
    "active_contract_map_hash",
)
INVARIANT_ENSEMBLE_COLUMNS = (
    "decision_id",
    "date",
    "raw_symbol",
    "continuity_segment_id",
    "state_segment_id",
    "reward_boundary_id",
    "bar_end_time_ns",
    "available_time_ns",
    "decision_time_ns",
    "submission_time_ns",
    "arrival_time_ns",
    "m5_decision_taken",
    "active_contract_map_hash",
)
ACTION_COLUMNS = (
    "h1_action",
    "m15_action",
    "m5_action",
    "h1_position",
    "m15_position",
    "final_target",
)


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    variant: str
    agent: str
    seed: int
    state_dimension: int
    action_values: tuple[int, int, int]
    contract_sha256: str
    training_config_sha256: str
    parameter_sha256: str
    training_data_manifest_sha256: str
    evidence_manifest_sha256: str
    registered_split_sha256: str
    training_cutoff_ns: int
    trained_through_date: str
    attempted_trial_id: str
    checkpoint_sha256: str | None = None

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported model-manifest schema")
        if self.variant not in {"base_paper_gc", "base_strong_gc"}:
            raise ValueError("invalid Base-GC variant")
        if self.agent not in {"H1", "M15", "M5"}:
            raise ValueError("invalid Base-GC agent")
        if self.seed < 0 or self.state_dimension < 1 or self.action_values != ACTIONS:
            raise ValueError("invalid model manifest dimensions/actions")
        for name in (
            "contract_sha256",
            "training_config_sha256",
            "parameter_sha256",
            "training_data_manifest_sha256",
            "evidence_manifest_sha256",
            "registered_split_sha256",
        ):
            if not _SHA256.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        pd.Timestamp(self.trained_through_date)
        if self.training_cutoff_ns <= 0:
            raise ValueError("training_cutoff_ns must be positive")
        cutoff_date = pd.Timestamp(self.training_cutoff_ns, unit="ns", tz="UTC").date()
        if pd.Timestamp(self.trained_through_date).date() > cutoff_date:
            raise ValueError("trained_through_date is after training_cutoff_ns")
        if self.checkpoint_sha256 is not None and not _SHA256.fullmatch(
            self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256")
        if not self.attempted_trial_id:
            raise ValueError("attempted_trial_id is required for trial accounting")


@dataclass(frozen=True)
class BaseTraceProduct:
    schema_version: int
    product_id: str
    variant: str
    contract_sha256: str
    trace_content_sha256: str
    trace_rows: int
    ensemble_seeds: tuple[int, ...]
    model_manifests: tuple[ModelManifest, ...]
    evidence_class: str
    claim_scope: str

    def validate(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace-product schema")
        if not _SHA256.fullmatch(self.contract_sha256):
            raise ValueError("invalid contract hash")
        if not _SHA256.fullmatch(self.trace_content_sha256):
            raise ValueError("invalid trace hash")
        if self.trace_rows < 0:
            raise ValueError("trace_rows cannot be negative")
        if self.evidence_class not in {"SYNTHETIC_ONLY", "DEVELOPMENT_ONLY", "LOCKED_EMPIRICAL"}:
            raise ValueError("invalid evidence_class")
        if len(self.ensemble_seeds) != 5 or len(set(self.ensemble_seeds)) != 5:
            raise ValueError("trace product requires the five distinct registered seeds")
        model_keys: list[tuple[int, str]] = []
        for manifest in self.model_manifests:
            manifest.validate()
            if manifest.contract_sha256 != self.contract_sha256:
                raise ValueError("model manifest is bound to a different contract")
            if manifest.variant != self.variant:
                raise ValueError("model manifest variant differs from trace product")
            model_keys.append((manifest.seed, manifest.agent))
        if len(model_keys) != len(set(model_keys)):
            raise ValueError("duplicate seed/agent model manifest")
        if self.evidence_class != "SYNTHETIC_ONLY":
            raise ValueError(
                "BaseTraceProduct is synthetic individual-trace smoke only; "
                "empirical evidence requires EnsembleTraceProduct"
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["model_manifests"] = [asdict(item) for item in self.model_manifests]
        return value


def _scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            raise ValueError("trace contains a non-finite float")
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _scalar(value.item())
    return str(value) if not isinstance(value, (str, bool)) else value


def validate_trace_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(TRACE_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"Base trace is missing columns: {missing}")
    trace = frame.loc[:, TRACE_COLUMNS].copy()
    if trace["decision_id"].astype(str).duplicated().any():
        raise ValueError("decision_id must be unique")
    trace["policy_seed"] = pd.to_numeric(trace["policy_seed"], errors="raise").astype("int64")
    if bool((trace["policy_seed"] < 0).any()) or trace["policy_seed"].nunique() != 1:
        raise ValueError("an individual Base trace must contain one nonnegative policy_seed")
    trace["reward_boundary_id"] = trace["reward_boundary_id"].astype(str)
    if bool(trace["reward_boundary_id"].eq("").any()):
        raise ValueError("reward_boundary_id cannot be empty")
    if not pd.api.types.is_bool_dtype(trace["m5_decision_taken"]):
        values = trace["m5_decision_taken"]
        if not values.isin([True, False]).all():
            raise ValueError("m5_decision_taken must be boolean")
    trace["m5_decision_taken"] = trace["m5_decision_taken"].astype(bool)
    for column in (
        "bar_end_time_ns",
        "available_time_ns",
        "decision_time_ns",
        "submission_time_ns",
        "arrival_time_ns",
    ):
        trace[column] = pd.to_numeric(trace[column], errors="raise").astype("int64")
    clocks = trace[
        [
            "bar_end_time_ns",
            "available_time_ns",
            "decision_time_ns",
            "submission_time_ns",
            "arrival_time_ns",
        ]
    ].to_numpy(dtype=np.int64)
    if bool((np.diff(clocks, axis=1) < 0).any()):
        raise ValueError("Base trace causal clocks are not monotone")
    for column in ACTION_COLUMNS:
        trace[column] = pd.to_numeric(trace[column], errors="raise").astype("int8")
        if not trace[column].isin(ACTIONS).all():
            raise ValueError(f"{column} contains an invalid action")
    for column in (
        "h1_state_sha256",
        "m15_state_sha256",
        "m5_state_sha256",
        "active_contract_map_hash",
    ):
        invalid = ~trace[column].astype(str).str.fullmatch(_SHA256)
        if bool(invalid.any()):
            raise ValueError(f"{column} contains an invalid SHA-256")
    trace["date"] = require_exact_globex_session_dates(
        trace["date"], trace["bar_end_time_ns"], label="trace"
    ).to_numpy()
    trace = trace.sort_values(
        ["available_time_ns", "raw_symbol", "decision_id"], kind="mergesort"
    ).reset_index(drop=True)
    for _, group in trace.groupby(
        ["raw_symbol", "continuity_segment_id", "reward_boundary_id"],
        sort=False,
        dropna=False,
    ):
        changed = group["final_target"].ne(group["final_target"].shift())
        if len(changed):
            changed.iloc[0] = False
        if bool((changed & ~group["m5_decision_taken"]).any()):
            raise ValueError("final_target changed outside an M5 action-lock renewal")
    return trace


def trace_content_sha256(frame: pd.DataFrame) -> str:
    trace = validate_trace_frame(frame)
    rows = [
        {column: _scalar(row[column]) for column in TRACE_COLUMNS}
        for row in trace.to_dict(orient="records")
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_trace_product(
    frame: pd.DataFrame,
    *,
    contract: BaselineContract,
    model_manifests: Iterable[ModelManifest],
    evidence_class: str,
    product_id: str,
) -> BaseTraceProduct:
    trace = validate_trace_frame(frame)
    if evidence_class != "SYNTHETIC_ONLY":
        raise ValueError(
            "empirical products must use the dedicated ensemble trace creator"
        )
    manifests = tuple(model_manifests)
    expected_hash = contract_sha256(contract)
    for manifest in manifests:
        manifest.validate()
        if manifest.contract_sha256 != expected_hash:
            raise ValueError("model manifest is bound to a different contract")
        if manifest.seed not in contract.seed_ensemble:
            raise ValueError("model manifest uses an unregistered seed")
        if manifest.variant != contract.variant:
            raise ValueError("model manifest uses the wrong Base-GC variant")
        if manifest.state_dimension != contract.agent(manifest.agent).observation_dimension:
            raise ValueError("model manifest state dimension differs from contract")
        if pd.Timestamp(manifest.trained_through_date).date() > pd.to_datetime(
            trace["date"], errors="raise"
        ).dt.date.max():
            raise ValueError("model trained-through date is after the trace period")
    product = BaseTraceProduct(
        schema_version=TRACE_SCHEMA_VERSION,
        product_id=product_id,
        variant=contract.variant,
        contract_sha256=expected_hash,
        trace_content_sha256=trace_content_sha256(trace),
        trace_rows=len(trace),
        ensemble_seeds=contract.seed_ensemble,
        model_manifests=manifests,
        evidence_class=evidence_class,
        claim_scope=(
            "Causal Base-GC decision trace bound to registered models and contract; "
            "profitability and exact-paper-replication are not implied."
        ),
    )
    product.validate()
    return product
