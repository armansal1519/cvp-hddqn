"""Authenticated, replay-derived provenance for empirical FeatureProducts.

The ordinary :func:`create_feature_product` helper is intentionally useful for
synthetic fixtures, but a hash of a caller-supplied frame is not evidence that
the frame was derived causally.  This module supplies the empirical trust
boundary: it replays an exact EventRecord artifact, derives FlowEvents only from
completed publisher groups, snapshots the incremental feature accumulator at
registered cursors, and later repeats the whole computation during verification.

The certificate proves reproducibility and byte-level provenance.  It is not a
cryptographic signature and it cannot independently prove that a vendor's raw
bytes are truthful; that remains the role of the bound MBO data-audit artifact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .artifacts import sha256_file, sha256_json
from .feature_product import (
    FeatureProduct,
    FeatureProductError,
    FlowEvent,
    RentFeatureAccumulator,
    create_feature_product,
    feature_frame_content_sha256,
    flow_events_from_group,
    validate_feature_frame,
)
from .feature_registry import FeatureRegistry, RENT_FEATURE_REGISTRY
from .replay import ReplayEngine
from .types import EventRecord


DERIVATION_SCHEMA_VERSION = 1
DERIVATION_CERTIFICATE_TYPE = "kx_mbo_feature_derivation_certificate"
SOURCE_MANIFEST_TYPE = "kx_mbo_feature_source_manifest"
FIFO_AUDIT_CERTIFICATE_TYPE = "MBO_DATA_AUDIT"
DERIVATION_DIRECTORY = "feature_derivation"
DECISIONS_FILE_NAME = "decisions.parquet"
FLOWS_FILE_NAME = "flow_events.parquet"

EVENT_COLUMNS = (
    "global_record_ordinal",
    "continuity_segment_id",
    "publisher_id",
    "instrument_id",
    "raw_symbol",
    "ts_event_ns",
    "ts_recv_ns",
    "action",
    "side",
    "price_ticks",
    "size",
    "order_id",
    "flags",
    "is_last",
    "is_snapshot",
    "is_bad_ts_recv",
)
DECISION_COLUMNS = (
    "feature_join_key",
    "publisher_id",
    "instrument_id",
    "raw_symbol",
    "continuity_segment_id",
    "tape_last_ordinal",
    "feature_time_ns",
    "decision_time_ns",
)
FLOW_COLUMNS = tuple(FlowEvent.__dataclass_fields__)
AUDIT_CHECKS = (
    "deterministic_reconstruction",
    "sequence_continuity",
    "available_time_causality",
    "snapshot_recovery",
)

DERIVATION_ALGORITHM_SHA256 = sha256_json(
    {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "algorithm": "ReplayEngine -> completed GroupResult -> flow_events_from_group "
        "-> available-time RentFeatureAccumulator -> exact cursor snapshot",
        "event_columns": EVENT_COLUMNS,
        "decision_columns": DECISION_COLUMNS,
        "flow_columns": FLOW_COLUMNS,
        "requires_snapshot_before_feature": True,
        "rejects_stale_cursor": True,
        "rejects_ambiguous_group_until_snapshot": True,
    }
)


@dataclass(frozen=True)
class ArtifactBinding:
    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class SourceProvenance:
    manifest_path: str
    manifest_sha256: str
    evidence_class: str
    data_version: str
    source_tape_hash: str
    replay_input: ArtifactBinding
    raw_inputs: tuple[ArtifactBinding, ...]
    fifo_audit: ArtifactBinding


@dataclass(frozen=True)
class FeatureDerivationPayload:
    """Typed in-memory material written beside an empirical FeatureProduct."""

    decisions: pd.DataFrame
    flows: pd.DataFrame
    provenance: SourceProvenance
    decode_latency_ns: int
    coverage: tuple[Mapping[str, Any], ...]
    group_count: int
    flow_eligible_group_count: int
    feature_frame_content_sha256: str


def _require_sha(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FeatureProductError(f"{field} must be a lowercase SHA-256")
    return text


def _strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    # Pandas commonly returns numpy.bool_, whose scalar conversion is exact.
    if type(value).__name__ == "bool_":
        return bool(value)
    raise FeatureProductError(f"{field} must contain JSON/typed booleans")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FeatureProductError(f"{label} is not readable canonical JSON: {path}") from error
    if not isinstance(value, dict):
        raise FeatureProductError(f"{label} must be a JSON object")
    return value


def _resolve(owner: Path, value: Any, *, field: str) -> Path:
    text = str(value).strip()
    if not text:
        raise FeatureProductError(f"{field} artifact path cannot be empty")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = owner.parent / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FeatureProductError(f"{field} artifact does not exist: {resolved}")
    return resolved


def _binding_from_declaration(
    owner: Path,
    declaration: Any,
    *,
    field: str,
    default_name: str,
) -> ArtifactBinding:
    if not isinstance(declaration, Mapping):
        raise FeatureProductError(f"{field} must bind an artifact and sha256")
    path = _resolve(owner, declaration.get("artifact"), field=field)
    expected = _require_sha(declaration.get("sha256"), field=f"{field}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise FeatureProductError(
            f"{field} artifact hash mismatch: expected {expected}, observed {observed}"
        )
    name = str(declaration.get("name", default_name)).strip()
    if not name:
        raise FeatureProductError(f"{field}.name cannot be empty")
    return ArtifactBinding(name=name, path=str(path), sha256=observed)


def raw_input_set_sha256(bindings: Sequence[ArtifactBinding]) -> str:
    return sha256_json(
        sorted(
            ({"name": binding.name, "sha256": binding.sha256} for binding in bindings),
            key=lambda row: (row["name"], row["sha256"]),
        )
    )


def verify_feature_source_manifest(path: str | Path) -> SourceProvenance:
    """Hash every declared byte and authenticate the typed FIFO-audit binding."""

    source = Path(path).expanduser().resolve()
    payload = _read_json_object(source, label="feature source manifest")
    if (
        int(payload.get("schema_version", 0)) != 1
        or payload.get("manifest_type") != SOURCE_MANIFEST_TYPE
    ):
        raise FeatureProductError("unsupported feature source manifest")
    evidence_class = str(payload.get("evidence_class", "")).upper()
    if evidence_class not in {"DEVELOPMENT_ONLY", "LOCKED_EMPIRICAL"}:
        raise FeatureProductError("feature source manifest must be empirical")
    data_version = str(payload.get("data_version", "")).strip()
    if not data_version:
        raise FeatureProductError("feature source manifest data_version cannot be empty")
    source_tape_hash = _require_sha(
        payload.get("source_tape_hash"), field="source_tape_hash"
    )
    replay = _binding_from_declaration(
        source,
        payload.get("replay_input"),
        field="replay_input",
        default_name="canonical_event_records",
    )
    if Path(replay.path).suffix.lower() != ".parquet":
        raise FeatureProductError("replay_input must be a canonical Parquet EventRecord ledger")
    raw_declarations = payload.get("raw_inputs")
    if not isinstance(raw_declarations, list) or not raw_declarations:
        raise FeatureProductError("feature source manifest needs nonempty raw_inputs")
    raw_inputs = tuple(
        _binding_from_declaration(
            source,
            declaration,
            field=f"raw_inputs[{index}]",
            default_name=f"raw-{index}",
        )
        for index, declaration in enumerate(raw_declarations)
    )
    names = [binding.name for binding in raw_inputs]
    paths = [binding.path for binding in raw_inputs]
    if len(names) != len(set(names)) or len(paths) != len(set(paths)):
        raise FeatureProductError("raw_inputs repeat a name or artifact path")
    audit = _binding_from_declaration(
        source,
        payload.get("fifo_audit"),
        field="fifo_audit",
        default_name="mbo_data_audit",
    )
    audit_payload = _read_json_object(Path(audit.path), label="FIFO audit certificate")
    if (
        int(audit_payload.get("schema_version", 0)) != 1
        or audit_payload.get("certificate_type") != FIFO_AUDIT_CERTIFICATE_TYPE
    ):
        raise FeatureProductError("unsupported FIFO audit certificate")
    if str(audit_payload.get("decision", "")).upper() != "PASS":
        raise FeatureProductError("FIFO audit certificate decision is not PASS")
    if str(audit_payload.get("evidence_class", "")).upper() != evidence_class:
        raise FeatureProductError("FIFO audit evidence class disagrees with source manifest")
    if str(audit_payload.get("data_version", "")) != data_version:
        raise FeatureProductError("FIFO audit data_version disagrees with source manifest")
    if _require_sha(
        audit_payload.get("replay_input_sha256"), field="audit replay_input_sha256"
    ) != replay.sha256:
        raise FeatureProductError("FIFO audit is bound to a different replay input")
    observed_raw_set = raw_input_set_sha256(raw_inputs)
    if _require_sha(
        audit_payload.get("raw_input_set_sha256"), field="audit raw_input_set_sha256"
    ) != observed_raw_set:
        raise FeatureProductError("FIFO audit is bound to a different raw input set")
    checks = audit_payload.get("checks")
    if not isinstance(checks, Mapping):
        raise FeatureProductError("FIFO audit certificate needs typed checks")
    for name in AUDIT_CHECKS:
        if checks.get(name) is not True:
            raise FeatureProductError(f"FIFO audit required check did not pass: {name}")
    return SourceProvenance(
        manifest_path=str(source),
        manifest_sha256=sha256_file(source),
        evidence_class=evidence_class,
        data_version=data_version,
        source_tape_hash=source_tape_hash,
        replay_input=replay,
        raw_inputs=raw_inputs,
        fifo_audit=audit,
    )


def validate_replay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the exact EventRecord ledger; opaque/future columns are forbidden."""

    if tuple(frame.columns) != EVENT_COLUMNS:
        missing = sorted(set(EVENT_COLUMNS) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(EVENT_COLUMNS))
        raise FeatureProductError(
            f"replay input must contain exactly EventRecord columns; missing={missing}, extra={extra}"
        )
    if frame.empty:
        raise FeatureProductError("replay input cannot be empty")
    output = frame.copy()
    integer_columns = (
        "global_record_ordinal",
        "continuity_segment_id",
        "publisher_id",
        "instrument_id",
        "ts_event_ns",
        "ts_recv_ns",
        "price_ticks",
        "size",
        "order_id",
        "flags",
    )
    for column in integer_columns:
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    for column in ("global_record_ordinal", "continuity_segment_id", "publisher_id", "instrument_id", "size", "order_id", "flags"):
        if bool(output[column].lt(0).any()):
            raise FeatureProductError(f"replay input {column} must be nonnegative")
    ordinals = output["global_record_ordinal"]
    if bool(ordinals.duplicated().any()) or not bool(ordinals.is_monotonic_increasing):
        raise FeatureProductError("replay input ordinals must be unique and strictly increasing")
    for column in ("raw_symbol", "action", "side"):
        if bool(output[column].isna().any()) or bool(output[column].astype(str).str.strip().eq("").any()):
            raise FeatureProductError(f"replay input {column} cannot be empty")
        output[column] = output[column].astype(str)
    for column in ("is_last", "is_snapshot", "is_bad_ts_recv"):
        output[column] = output[column].map(
            lambda value: _strict_bool(value, field=f"replay input {column}")
        ).astype(bool)
    return output.reset_index(drop=True)


def load_replay_frame(path: str | Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(Path(path))
    except Exception as error:
        raise FeatureProductError("replay input is not readable Parquet") from error
    return validate_replay_frame(frame)


def validate_decision_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != DECISION_COLUMNS:
        missing = sorted(set(DECISION_COLUMNS) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(DECISION_COLUMNS))
        raise FeatureProductError(
            f"decisions must contain exactly causal cursor columns; missing={missing}, extra={extra}"
        )
    if frame.empty:
        raise FeatureProductError("decisions cannot be empty")
    output = frame.copy()
    for column in ("feature_join_key", "raw_symbol"):
        if bool(output[column].isna().any()) or bool(output[column].astype(str).str.strip().eq("").any()):
            raise FeatureProductError(f"decision {column} cannot be empty")
        output[column] = output[column].astype(str)
    if bool(output["feature_join_key"].duplicated().any()):
        raise FeatureProductError("decision feature_join_key must be unique")
    for column in (
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "tape_last_ordinal",
        "feature_time_ns",
        "decision_time_ns",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    for column in ("publisher_id", "instrument_id", "continuity_segment_id", "tape_last_ordinal"):
        if bool(output[column].lt(0).any()):
            raise FeatureProductError(f"decision {column} must be nonnegative")
    if bool((output["feature_time_ns"] > output["decision_time_ns"]).any()):
        raise FeatureProductError("decision feature_time_ns cannot follow decision_time_ns")
    return output.sort_values(
        ["tape_last_ordinal", "feature_time_ns", "feature_join_key"], kind="stable"
    ).reset_index(drop=True)


def _flow_frame(events: Sequence[FlowEvent]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(event) for event in events], columns=FLOW_COLUMNS)
    if frame.empty:
        raise FeatureProductError("causal replay produced no certifiable FlowEvents")
    return frame.sort_values(
        ["tape_last_ordinal", "publisher_id", "instrument_id", "side"], kind="stable"
    ).reset_index(drop=True)


def flow_frame_content_sha256(frame: pd.DataFrame) -> str:
    if tuple(frame.columns) != FLOW_COLUMNS or frame.empty:
        raise FeatureProductError("flow ledger has an unsupported schema or is empty")
    events = [FlowEvent(**row) for row in frame.to_dict("records")]
    canonical = _flow_frame(events)
    return sha256_json(canonical.to_dict("records"))


def decision_frame_content_sha256(frame: pd.DataFrame) -> str:
    canonical = validate_decision_frame(frame)
    return sha256_json(canonical.to_dict("records"))


def _identity_from_mapping(row: Mapping[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["publisher_id"]),
        int(row["instrument_id"]),
        str(row["raw_symbol"]),
        int(row["continuity_segment_id"]),
    )


def _coverage(records: pd.DataFrame, group_rows: Sequence[Mapping[str, Any]], source_tape_hash: str) -> tuple[Mapping[str, Any], ...]:
    groups = pd.DataFrame(group_rows)
    output: list[Mapping[str, Any]] = []
    keys = ["publisher_id", "instrument_id", "raw_symbol", "continuity_segment_id"]
    for identity, selected in records.groupby(keys, sort=True, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        selector = pd.Series(True, index=groups.index)
        for column, value in zip(keys, identity, strict=True):
            selector &= groups[column].eq(value)
        selected_groups = groups.loc[selector]
        output.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "source_tape_hash": source_tape_hash,
                "record_count": int(len(selected)),
                "completed_group_count": int(len(selected_groups)),
                "first_ordinal": int(selected["global_record_ordinal"].min()),
                "last_ordinal": int(selected["global_record_ordinal"].max()),
                "first_event_time_ns": int(selected["ts_event_ns"].min()),
                "last_event_time_ns": int(selected["ts_event_ns"].max()),
                "first_receive_time_ns": int(selected["ts_recv_ns"].min()),
                "last_receive_time_ns": int(selected["ts_recv_ns"].max()),
            }
        )
    return tuple(output)


def derive_feature_rows(
    replay: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    source_tape_hash: str,
    decode_latency_ns: int,
    registry: FeatureRegistry = RENT_FEATURE_REGISTRY,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[Mapping[str, Any], ...], int, int]:
    """Deterministically replay one exact ledger and snapshot registered cursors."""

    records = validate_replay_frame(replay)
    decision_rows = validate_decision_frame(decisions)
    if decode_latency_ns < 0:
        raise FeatureProductError("decode_latency_ns must be nonnegative")
    pending: dict[tuple[tuple[int, int, str, int], int], list[dict[str, Any]]] = {}
    for row in decision_rows.to_dict("records"):
        pending.setdefault((_identity_from_mapping(row), int(row["tape_last_ordinal"])), []).append(row)
    engine = ReplayEngine(
        decode_latency_ns=int(decode_latency_ns), source_tape_hash=source_tape_hash
    )
    accumulators: dict[tuple[int, int, str, int], RentFeatureAccumulator] = {}
    epoch_ready: dict[tuple[int, int, str, int], bool] = {}
    rows: list[dict[str, Any]] = []
    flows: list[FlowEvent] = []
    groups: list[dict[str, Any]] = []
    flow_eligible_groups = 0
    group_available_clocks: dict[tuple[int, int, str, int], list[tuple[int, int | None]]] = {}
    for record_values in records.to_dict("records"):
        record = EventRecord.from_mapping(record_values)
        result = engine.push(record)
        if result is None:
            continue
        identity = (
            result.publisher_id,
            result.instrument_id,
            result.raw_symbol,
            result.continuity_segment_id,
        )
        groups.append(
            {
                "publisher_id": result.publisher_id,
                "instrument_id": result.instrument_id,
                "raw_symbol": result.raw_symbol,
                "continuity_segment_id": result.continuity_segment_id,
                "first_ordinal": result.first_ordinal,
                "last_ordinal": result.last_ordinal,
                "event_time_ns": result.event_time_ns,
                "available_time_ns": result.available_time_ns,
            }
        )
        group_available_clocks.setdefault(identity, []).append(
            (int(result.last_ordinal), None if result.available_time_ns is None else int(result.available_time_ns))
        )
        book = engine.book_for(record)
        group_flows: tuple[FlowEvent, FlowEvent] | None = None
        if result.available_time_ns is not None and result.cause_labels_eligible:
            group_flows = flow_events_from_group(
                result, book, source_tape_hash=source_tape_hash
            )
            flows.extend(group_flows)
            flow_eligible_groups += 1
        if result.snapshot and group_flows is not None:
            accumulator = accumulators.get(identity)
            if accumulator is None:
                accumulator = RentFeatureAccumulator(
                    publisher_id=result.publisher_id,
                    instrument_id=result.instrument_id,
                    raw_symbol=result.raw_symbol,
                    continuity_segment_id=result.continuity_segment_id,
                    segment_start_available_ns=int(result.available_time_ns),
                    source_tape_hash=source_tape_hash,
                    initial_tape_ordinal=result.last_ordinal,
                    book_epoch=book.book_epoch,
                    registry=registry,
                )
                accumulators[identity] = accumulator
            else:
                accumulator.reset(
                    available_time_ns=int(result.available_time_ns),
                    event_time_ns=int(result.event_time_ns),
                    book_epoch=book.book_epoch,
                    tape_last_ordinal=result.last_ordinal,
                )
            epoch_ready[identity] = True
        elif group_flows is None:
            epoch_ready[identity] = False
        accumulator = accumulators.get(identity)
        if accumulator is not None and epoch_ready.get(identity, False) and group_flows is not None:
            for flow in group_flows:
                accumulator.update(flow)
        cursor = (identity, int(result.last_ordinal))
        if cursor not in pending:
            continue
        if accumulator is None or not epoch_ready.get(identity, False):
            raise FeatureProductError(
                "decision cursor is not inside a snapshot-restored, unambiguous replay epoch"
            )
        for decision in sorted(
            pending.pop(cursor),
            key=lambda value: (int(value["feature_time_ns"]), str(value["feature_join_key"])),
        ):
            rows.append(
                accumulator.snapshot(
                    book,
                    feature_join_key=str(decision["feature_join_key"]),
                    feature_time_ns=int(decision["feature_time_ns"]),
                    decision_time_ns=int(decision["decision_time_ns"]),
                    tape_last_ordinal=int(decision["tape_last_ordinal"]),
                    source_tape_hash=source_tape_hash,
                )
            )
    if engine.group_builder.unfinished():
        raise FeatureProductError("replay input ends with unfinished atomic groups")
    if pending:
        example = next(iter(pending))
        raise FeatureProductError(
            f"decision identity/cursor does not resolve to a completed replay group: {example}"
        )
    # The selected cursor must be the latest complete same-identity group that
    # was causally available at feature_time. Otherwise the certificate would
    # bless a stale state that omitted already-known events.
    for decision in decision_rows.to_dict("records"):
        identity = _identity_from_mapping(decision)
        cursor = int(decision["tape_last_ordinal"])
        feature_time = int(decision["feature_time_ns"])
        omitted = [
            ordinal
            for ordinal, available in group_available_clocks.get(identity, [])
            if ordinal > cursor and available is not None and available <= feature_time
        ]
        if omitted:
            raise FeatureProductError(
                "decision cursor omits a later completed group already available at feature_time"
            )
    if not groups:
        raise FeatureProductError("replay input contains no completed publisher groups")
    feature_frame = validate_feature_frame(pd.DataFrame(rows), registry=registry)
    flow_frame = _flow_frame(flows)
    return (
        feature_frame,
        flow_frame,
        _coverage(records, groups, source_tape_hash),
        len(groups),
        flow_eligible_groups,
    )


def create_empirical_feature_product(
    *,
    decisions: pd.DataFrame,
    source_manifest_path: str | Path,
    product_id: str,
    data_version: str,
    evidence_class: str = "DEVELOPMENT_ONLY",
    decode_latency_ns: int = 1_000_000,
    registry: FeatureRegistry = RENT_FEATURE_REGISTRY,
) -> FeatureProduct:
    """Build an empirical product only from authenticated replay inputs."""

    evidence = str(evidence_class).upper()
    if evidence not in {"DEVELOPMENT_ONLY", "LOCKED_EMPIRICAL"}:
        raise FeatureProductError("empirical feature builder requires empirical evidence_class")
    provenance = verify_feature_source_manifest(source_manifest_path)
    if provenance.evidence_class != evidence:
        raise FeatureProductError("feature evidence class disagrees with source manifest")
    if provenance.data_version != str(data_version):
        raise FeatureProductError("feature data_version disagrees with source manifest")
    replay = load_replay_frame(provenance.replay_input.path)
    features, flows, coverage, group_count, eligible_count = derive_feature_rows(
        replay,
        decisions,
        source_tape_hash=provenance.source_tape_hash,
        decode_latency_ns=int(decode_latency_ns),
        registry=registry,
    )
    product = create_feature_product(
        features,
        product_id=product_id,
        data_version=data_version,
        source_manifest_sha256=provenance.manifest_sha256,
        evidence_class=evidence,
        registry=registry,
    )
    derivation = FeatureDerivationPayload(
        decisions=validate_decision_frame(decisions),
        flows=flows,
        provenance=provenance,
        decode_latency_ns=int(decode_latency_ns),
        coverage=coverage,
        group_count=group_count,
        flow_eligible_group_count=eligible_count,
        feature_frame_content_sha256=feature_frame_content_sha256(
            features, registry=registry
        ),
    )
    return FeatureProduct(product.features, product.manifest, product.registry, derivation)


def _binding_dict(binding: ArtifactBinding) -> dict[str, str]:
    return {"name": binding.name, "artifact": binding.path, "sha256": binding.sha256}


def write_feature_derivation_artifacts(
    staging: Path,
    payload: FeatureDerivationPayload,
    *,
    feature_frame_sha256: str,
) -> dict[str, Any]:
    if payload.feature_frame_content_sha256 != feature_frame_sha256:
        raise FeatureProductError("derivation payload belongs to a different feature frame")
    directory = staging / DERIVATION_DIRECTORY
    directory.mkdir(parents=True, exist_ok=False)
    decisions = validate_decision_frame(payload.decisions)
    # Reconstructing typed FlowEvents rejects malformed or caller-invented flow fields.
    flow_digest = flow_frame_content_sha256(payload.flows)
    decisions_path = directory / DECISIONS_FILE_NAME
    flows_path = directory / FLOWS_FILE_NAME
    decisions.to_parquet(decisions_path, index=False)
    payload.flows.loc[:, list(FLOW_COLUMNS)].to_parquet(flows_path, index=False)
    provenance = payload.provenance
    return {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "certificate_type": DERIVATION_CERTIFICATE_TYPE,
        "status": "RECOMPUTABLE",
        "algorithm_sha256": DERIVATION_ALGORITHM_SHA256,
        "decode_latency_ns": int(payload.decode_latency_ns),
        "source_tape_hash": provenance.source_tape_hash,
        "source_manifest": {
            "artifact": provenance.manifest_path,
            "sha256": provenance.manifest_sha256,
        },
        "replay_input": _binding_dict(provenance.replay_input),
        "raw_inputs": [_binding_dict(binding) for binding in provenance.raw_inputs],
        "fifo_audit": _binding_dict(provenance.fifo_audit),
        "coverage": list(payload.coverage),
        "group_count": int(payload.group_count),
        "flow_eligible_group_count": int(payload.flow_eligible_group_count),
        "feature_frame_content_sha256": feature_frame_sha256,
        "decision_frame_content_sha256": decision_frame_content_sha256(decisions),
        "flow_frame_content_sha256": flow_digest,
        "artifacts": {
            "decisions": {
                "path": f"{DERIVATION_DIRECTORY}/{DECISIONS_FILE_NAME}",
                "sha256": sha256_file(decisions_path),
            },
            "flows": {
                "path": f"{DERIVATION_DIRECTORY}/{FLOWS_FILE_NAME}",
                "sha256": sha256_file(flows_path),
            },
        },
    }


def _same_bindings(observed: Sequence[ArtifactBinding], declared: Any) -> bool:
    if not isinstance(declared, list):
        return False
    expected = [_binding_dict(binding) for binding in observed]
    return expected == [dict(value) if isinstance(value, Mapping) else value for value in declared]


def verify_feature_derivation(
    product_path: str | Path,
    *,
    product: FeatureProduct,
) -> dict[str, Any]:
    """Repeat derivation from raw-bound replay bytes and reject any mismatch."""

    root = Path(product_path).expanduser().resolve()
    declaration = product.manifest.get("feature_derivation")
    if not isinstance(declaration, Mapping):
        raise FeatureProductError("empirical FeatureProduct lacks a derivation certificate")
    if (
        int(declaration.get("schema_version", 0)) != DERIVATION_SCHEMA_VERSION
        or declaration.get("certificate_type") != DERIVATION_CERTIFICATE_TYPE
        or declaration.get("status") != "RECOMPUTABLE"
        or declaration.get("algorithm_sha256") != DERIVATION_ALGORITHM_SHA256
    ):
        raise FeatureProductError("unsupported or self-asserted feature derivation certificate")
    source_decl = declaration.get("source_manifest")
    if not isinstance(source_decl, Mapping):
        raise FeatureProductError("derivation certificate does not bind its source manifest")
    source_path = _resolve(root / "manifest.json", source_decl.get("artifact"), field="feature_derivation.source_manifest")
    source_sha = _require_sha(source_decl.get("sha256"), field="feature_derivation.source_manifest.sha256")
    if sha256_file(source_path) != source_sha:
        raise FeatureProductError("feature derivation source-manifest hash mismatch")
    provenance = verify_feature_source_manifest(source_path)
    if provenance.manifest_sha256 != source_sha:
        raise FeatureProductError("feature derivation source manifest changed")
    if provenance.manifest_sha256 != str(product.manifest.get("source_manifest_sha256")):
        raise FeatureProductError("FeatureProduct binds a different source manifest")
    if provenance.evidence_class != str(product.manifest.get("evidence_class")):
        raise FeatureProductError("feature derivation evidence class disagrees with FeatureProduct")
    if provenance.data_version != str(product.manifest.get("data_version")):
        raise FeatureProductError("feature derivation data_version disagrees with FeatureProduct")
    if dict(declaration.get("replay_input", {})) != _binding_dict(provenance.replay_input):
        raise FeatureProductError("derivation certificate replay binding disagrees with source manifest")
    if not _same_bindings(provenance.raw_inputs, declaration.get("raw_inputs")):
        raise FeatureProductError("derivation certificate raw bindings disagree with source manifest")
    if dict(declaration.get("fifo_audit", {})) != _binding_dict(provenance.fifo_audit):
        raise FeatureProductError("derivation certificate audit binding disagrees with source manifest")
    if declaration.get("source_tape_hash") != provenance.source_tape_hash:
        raise FeatureProductError("derivation certificate source identity hash mismatch")
    if not product.features["source_tape_hash"].astype(str).eq(provenance.source_tape_hash).all():
        raise FeatureProductError("FeatureProduct rows use a different source tape")
    artifacts = declaration.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FeatureProductError("derivation certificate does not bind internal artifacts")
    loaded: dict[str, pd.DataFrame] = {}
    expected_paths = {
        "decisions": f"{DERIVATION_DIRECTORY}/{DECISIONS_FILE_NAME}",
        "flows": f"{DERIVATION_DIRECTORY}/{FLOWS_FILE_NAME}",
    }
    for name, expected_relative in expected_paths.items():
        binding = artifacts.get(name)
        if not isinstance(binding, Mapping) or binding.get("path") != expected_relative:
            raise FeatureProductError(f"derivation {name} artifact path is not canonical")
        path = (root / expected_relative).resolve()
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise FeatureProductError(f"derivation {name} artifact hash mismatch")
        loaded[name] = pd.read_parquet(path)
    decisions = validate_decision_frame(loaded["decisions"])
    if decision_frame_content_sha256(decisions) != declaration.get("decision_frame_content_sha256"):
        raise FeatureProductError("derivation decision ledger content hash mismatch")
    if flow_frame_content_sha256(loaded["flows"]) != declaration.get("flow_frame_content_sha256"):
        raise FeatureProductError("derivation flow ledger content hash mismatch")
    recomputed, flows, coverage, groups, eligible_groups = derive_feature_rows(
        load_replay_frame(provenance.replay_input.path),
        decisions,
        source_tape_hash=provenance.source_tape_hash,
        decode_latency_ns=int(declaration.get("decode_latency_ns", -1)),
        registry=product.registry,
    )
    recomputed_feature_sha = feature_frame_content_sha256(
        recomputed, registry=product.registry
    )
    observed_feature_sha = feature_frame_content_sha256(
        product.features, registry=product.registry
    )
    if recomputed_feature_sha != observed_feature_sha:
        raise FeatureProductError("FeatureProduct differs from deterministic replay recomputation")
    if recomputed_feature_sha != declaration.get("feature_frame_content_sha256"):
        raise FeatureProductError("derivation certificate binds a different feature frame")
    if flow_frame_content_sha256(flows) != declaration.get("flow_frame_content_sha256"):
        raise FeatureProductError("stored FlowEvent ledger differs from completed-group replay")
    if list(coverage) != declaration.get("coverage"):
        raise FeatureProductError("derivation source identity/ordinal/time coverage mismatch")
    if groups != int(declaration.get("group_count", -1)) or eligible_groups != int(
        declaration.get("flow_eligible_group_count", -1)
    ):
        raise FeatureProductError("derivation completed-group coverage mismatch")
    return {
        "certificate_type": DERIVATION_CERTIFICATE_TYPE,
        "algorithm_sha256": DERIVATION_ALGORITHM_SHA256,
        "source_manifest_sha256": provenance.manifest_sha256,
        "replay_input_sha256": provenance.replay_input.sha256,
        "raw_input_set_sha256": raw_input_set_sha256(provenance.raw_inputs),
        "fifo_audit_sha256": provenance.fifo_audit.sha256,
        "decision_frame_content_sha256": declaration["decision_frame_content_sha256"],
        "flow_frame_content_sha256": declaration["flow_frame_content_sha256"],
        "feature_frame_content_sha256": recomputed_feature_sha,
        "row_count": len(recomputed),
        "group_count": groups,
        "flow_eligible_group_count": eligible_groups,
        "coverage": list(coverage),
        "recomputed": True,
    }


__all__ = [
    "AUDIT_CHECKS",
    "DECISION_COLUMNS",
    "DERIVATION_ALGORITHM_SHA256",
    "DERIVATION_CERTIFICATE_TYPE",
    "DERIVATION_SCHEMA_VERSION",
    "EVENT_COLUMNS",
    "FIFO_AUDIT_CERTIFICATE_TYPE",
    "SOURCE_MANIFEST_TYPE",
    "ArtifactBinding",
    "FeatureDerivationPayload",
    "SourceProvenance",
    "create_empirical_feature_product",
    "decision_frame_content_sha256",
    "derive_feature_rows",
    "flow_frame_content_sha256",
    "load_replay_frame",
    "raw_input_set_sha256",
    "validate_decision_frame",
    "validate_replay_frame",
    "verify_feature_derivation",
    "verify_feature_source_manifest",
    "write_feature_derivation_artifacts",
]
