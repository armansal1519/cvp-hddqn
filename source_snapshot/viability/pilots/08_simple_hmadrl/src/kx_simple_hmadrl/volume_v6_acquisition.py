"""Resumable, phase-gated construction of V6 GC session and volume sidecars.

The builder reads only explicitly selected dates.  In particular, constructing
training data cannot enumerate or open the registered validation or test
partitions.  A past-only active-contract map is mandatory for registered work.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .gc_data import (
    SessionAudit,
    _atomic_npz,
    _read_day,
    _session_clocks,
    build_trade_features,
    reconstruct_arrival_bbo,
    reconstruct_l2,
    sha256,
)
from .volume_v3_data import multi_level_ofi
from .volume_v4_data import _absolute_trade_prices
from .volume_v6_eval import (
    TEST_COMPLETE_ON,
    assert_test_unlocked,
    atomic_json,
    canonical_json_sha256,
    classify_dates,
    frozen_protocol,
)


ACTIVE_MAP_RECEIPT_ARTIFACT = "V6_PAST_ONLY_ACTIVE_CONTRACT_MAP_RECEIPT"
ACTIVE_MAP_SOURCE_ARTIFACT = "V6_GC_ELIGIBLE_PHASE_UNIVERSE"
ACTIVE_MAP_REQUIRED_COLUMNS = (
    "session_date",
    "selected_raw_symbol",
    "prior_completed_session",
    "same_session_volume_used",
    "source_coverage_complete",
    "volume_cutoff_available_time_ns",
    "decision_time_utc_ns",
)
ACTIVE_MAP_CONSTRUCTION_METHOD = "prior_completed_session_volume_only"
ACTIVE_MAP_AVAILABILITY_RULE = (
    "volume_cutoff_available_time_ns < decision_time_utc_ns"
)
REGISTERED_PHASE_BOUNDS = {
    "train": ("2025-01-01", "2026-05-01"),
    "validation": ("2026-05-01", "2026-06-01"),
    "test": ("2026-06-01", "2026-09-01"),
}


def _date_strings(start: str, end_exclusive: str) -> list[str]:
    first = date.fromisoformat(start)
    end = date.fromisoformat(end_exclusive)
    if first >= end:
        raise ValueError("data-build interval must be nonempty")
    return [str(value.date()) for value in pd.date_range(first, end, inclusive="left", freq="D")]


def _current_date() -> date:
    """Private wall-clock seam; production callers cannot override test timing."""

    return date.today()


def _deep_verify_complete_training_manifest(path: Path) -> dict[str, object]:
    """Resolve the orchestration verifier lazily to keep module imports acyclic."""

    from .volume_v6_orchestration import verify_complete_training_manifest

    return verify_complete_training_manifest(Path(path).resolve())


def _deep_verify_test_unlock(path: Path) -> dict[str, object]:
    """Recompute the unlock, complete training, and May evaluation chains."""

    unlock = assert_test_unlocked(Path(path).resolve())
    from .volume_v6_orchestration import (
        verify_complete_training_manifest,
        verify_evaluation_manifest,
    )

    training = Path(str(unlock.get("complete_training_manifest", ""))).resolve()
    validation = Path(str(unlock.get("validation_evaluation_manifest", ""))).resolve()
    complete = verify_complete_training_manifest(training)
    evaluation = verify_evaluation_manifest(validation)
    if (
        evaluation.get("phase") != "validation"
        or evaluation.get("verdict") != "PASS"
        or evaluation.get("failed_gates") != []
        or evaluation.get("complete") is not True
        or complete.get("complete") is not True
    ):
        raise PermissionError("test unlock does not deeply bind a complete May PASS")
    return unlock


def validate_build_access(
    phase: str, start: str, end_exclusive: str, *,
    training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> None:
    """Authorize a calendar interval before any archive or map is opened."""

    days = _date_strings(start, end_exclusive)
    if phase == "smoke":
        classify_dates(days, phase="smoke")
        return
    if phase == "train":
        classify_dates(days, phase="train")
        return
    if phase == "validation":
        classify_dates(days, phase="validation")
        if training_manifest is None or not training_manifest.is_file():
            raise PermissionError("May build requires the complete frozen training manifest")
        try:
            _deep_verify_complete_training_manifest(training_manifest)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise PermissionError(
                "training manifest failed the complete V6 lineage verification"
            ) from error
        return
    if phase == "test":
        classify_dates(days, phase="test")
        if _current_date() < TEST_COMPLETE_ON:
            raise PermissionError("test construction is forbidden before August 2026 completes")
        if unlock_receipt is None:
            raise PermissionError("test construction requires the immutable unlock receipt")
        try:
            _deep_verify_test_unlock(unlock_receipt)
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise PermissionError("test unlock failed its complete V6 lineage") from error
        return
    raise ValueError("phase must be smoke, train, validation, or test")


def active_map_receipt_path(active_map: Path) -> Path:
    """Return the mandatory immutable proof sidecar for an active-contract map."""

    path = Path(active_map)
    return path.with_name(f"{path.name}.receipt.json")


def _resolved_from(anchor: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _canonical_date(value: object, *, label: str) -> str:
    text = str(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must be canonical YYYY-MM-DD") from error
    if str(parsed) != text:
        raise ValueError(f"{label} must be canonical YYYY-MM-DD")
    return text


def _source_partition(
    archive_root: Path, layer: str, session_date: str,
) -> Path:
    stamp = date.fromisoformat(session_date)
    return (
        archive_root / layer / f"year={stamp.year:04d}"
        / f"month={stamp.month:02d}" / f"day={stamp.day:02d}"
    )


def _archive_session_dates(
    archive_root: Path, start: str, end_exclusive: str,
) -> tuple[str, ...]:
    dates: list[str] = []
    for day in _date_strings(start, end_exclusive):
        l2 = tuple(_source_partition(archive_root, "l2_depth", day).glob("*.parquet"))
        trades = tuple(_source_partition(archive_root, "l1_trades", day).glob("*.parquet"))
        if bool(l2) != bool(trades):
            raise ValueError(f"authoritative archive has one-sided layer coverage for {day}")
        if l2:
            dates.append(day)
    if not dates:
        raise ValueError("authoritative archive contains no complete session partitions")
    return tuple(dates)


def _validate_source_file_evidence(
    source: Path,
    archive_root: Path,
    row: Mapping[str, object],
    *,
    layer: str,
    session_date: str,
    raw_symbol: str,
) -> list[dict[str, object]]:
    key = "l2_depth_files" if layer == "l2_depth" else "l1_trade_files"
    raw_records = row.get(key)
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"source-universe row lacks {key} evidence")
    partition = _source_partition(archive_root, layer, session_date).resolve()
    actual_files = tuple(sorted(path.resolve() for path in partition.glob("*.parquet")))
    records: list[dict[str, object]] = []
    declared_files: list[Path] = []
    symbol_seen = False
    for raw_record in raw_records:
        if not isinstance(raw_record, dict) or set(raw_record) != {"path", "sha256", "bytes"}:
            raise ValueError("source child evidence requires path, sha256, and bytes")
        child = _resolved_from(source.parent, raw_record.get("path"))
        try:
            child.relative_to(archive_root)
        except ValueError as error:
            raise ValueError("source child evidence escapes its authoritative archive") from error
        if (
            child.parent != partition
            or child.suffix != ".parquet"
            or not child.is_file()
            or child.stat().st_size != raw_record.get("bytes")
            or sha256(child) != raw_record.get("sha256")
        ):
            raise ValueError("source child file path/hash/partition evidence differs")
        try:
            symbols = pq.ParquetFile(child).read(columns=["raw_symbol"])[
                "raw_symbol"
            ].to_pylist()
        except (KeyError, OSError) as error:
            raise ValueError("source child lacks authoritative raw_symbol rows") from error
        symbol_seen |= any(str(value) == raw_symbol for value in symbols)
        declared_files.append(child)
        records.append({
            "path": str(raw_record["path"]),
            "sha256": str(raw_record["sha256"]),
            "bytes": int(raw_record["bytes"]),
        })
    if tuple(declared_files) != actual_files:
        raise ValueError("source child evidence is not the exact archive partition inventory")
    if not symbol_seen:
        raise ValueError("source child evidence contains no row for its selected raw symbol")
    return records


def _validate_source_universe(
    source: Path,
    *,
    expected_sha256: object | None = None,
    expected_bytes: object | None = None,
) -> dict[str, object]:
    """Validate the authoritative, complete session/date/symbol universe."""

    source = Path(source).resolve()
    if (
        not source.is_file()
        or (expected_sha256 is not None and sha256(source) != expected_sha256)
        or (expected_bytes is not None and source.stat().st_size != expected_bytes)
    ):
        raise ValueError("active-contract construction source receipt is absent or changed")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("active-contract source universe is malformed") from error
    _, protocol_digest = frozen_protocol()
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("artifact_type") != ACTIVE_MAP_SOURCE_ARTIFACT
        or value.get("protocol_sha256") != protocol_digest
        or value.get("eligible_phase_universe_complete") is not True
        or value.get("source_coverage_complete") is not True
    ):
        raise ValueError("active-contract source lacks the authoritative universe identity")
    phase = str(value.get("phase", ""))
    if phase not in {"smoke", *REGISTERED_PHASE_BOUNDS}:
        raise ValueError("active-contract source universe has an invalid phase")
    start = _canonical_date(value.get("start"), label="source-universe start")
    end_exclusive = _canonical_date(
        value.get("end_exclusive"), label="source-universe end",
    )
    if start >= end_exclusive:
        raise ValueError("active-contract source universe interval must be nonempty")
    if phase in REGISTERED_PHASE_BOUNDS:
        if (start, end_exclusive) != REGISTERED_PHASE_BOUNDS[phase]:
            raise ValueError("registered source universe does not cover its exact frozen phase")
    elif not ("2025-01-01" <= start < end_exclusive <= "2025-09-01"):
        raise ValueError("smoke source universe leaves contaminated Jan-Aug 2025")
    archive_root = _resolved_from(source.parent, value.get("archive_root"))
    if not archive_root.is_dir():
        raise ValueError("source universe lacks its authoritative archive root")
    raw_rows = value.get("sessions")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("active-contract source universe contains no eligible sessions")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict) or set(raw) != {
            "session_date", "raw_symbol", "l2_depth_files", "l1_trade_files",
        }:
            raise ValueError(
                "source-universe rows require date, symbol, and both child-file inventories"
            )
        day = _canonical_date(raw.get("session_date"), label="source session date")
        symbol_value = raw.get("raw_symbol")
        if symbol_value is None or pd.isna(symbol_value) or not str(symbol_value).strip():
            raise ValueError("source-universe row contains an empty raw symbol")
        if not start <= day < end_exclusive:
            raise ValueError("source-universe row lies outside its declared phase")
        symbol = str(symbol_value).strip()
        rows.append({
            "session_date": day,
            "raw_symbol": symbol,
            "l2_depth_files": _validate_source_file_evidence(
                source, archive_root, raw, layer="l2_depth",
                session_date=day, raw_symbol=symbol,
            ),
            "l1_trade_files": _validate_source_file_evidence(
                source, archive_root, raw, layer="l1_trades",
                session_date=day, raw_symbol=symbol,
            ),
        })
    if rows != sorted(rows, key=lambda row: row["session_date"]):
        raise ValueError("source-universe session rows must be chronological")
    dates = [row["session_date"] for row in rows]
    if len(dates) != len(set(dates)):
        raise ValueError("source universe contains duplicate eligible session dates")
    if tuple(dates) != _archive_session_dates(archive_root, start, end_exclusive):
        raise ValueError("source universe is not the exact authoritative archive session inventory")
    rows_sha256 = canonical_json_sha256(rows)
    if (
        value.get("session_count") != len(rows)
        or value.get("session_rows_sha256") != rows_sha256
    ):
        raise ValueError("source-universe row count/digest does not recompute")
    return {
        "phase": phase,
        "start": start,
        "end_exclusive": end_exclusive,
        "session_count": len(rows),
        "session_rows_sha256": rows_sha256,
        "archive_root": str(archive_root),
        "sessions": [
            {"session_date": row["session_date"], "raw_symbol": row["raw_symbol"]}
            for row in rows
        ],
    }


def _source_access_identity(source: Path) -> tuple[str, str, str]:
    """Read only the control-plane fields needed before archive authorization."""

    try:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("active-contract source lacks the authoritative universe identity") from error
    if not isinstance(value, dict) or value.get("artifact_type") != ACTIVE_MAP_SOURCE_ARTIFACT:
        raise ValueError("active-contract source lacks the authoritative universe identity")
    return (
        str(value.get("phase", "")), str(value.get("start", "")),
        str(value.get("end_exclusive", "")),
    )


def _assert_map_equals_source_universe(
    mapping: pd.DataFrame, universe: Mapping[str, object],
) -> None:
    expected = [
        (str(row["session_date"]), str(row["raw_symbol"]))
        for row in universe["sessions"]  # type: ignore[index]
    ]
    selected = list(zip(
        mapping["session_date"].astype(str),
        mapping["selected_raw_symbol"].astype(str).str.strip(),
        strict=True,
    ))
    if selected != expected:
        raise ValueError(
            "active-contract map is not the exact eligible source session/date/symbol universe"
        )


def _load_active_map_receipt(
    active_map: Path,
    receipt_path: Path | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    """Verify the construction receipt before the Parquet map is opened."""

    active_map = Path(active_map).resolve()
    proof_path = (
        active_map_receipt_path(active_map)
        if receipt_path is None else Path(receipt_path).resolve()
    )
    try:
        value = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("active-contract map lacks its construction receipt") from error
    if not isinstance(value, dict):
        raise ValueError("active-contract construction receipt must be a JSON object")
    _, protocol_digest = frozen_protocol()
    construction = value.get("construction")
    map_record = value.get("active_contract_map")
    source_record = value.get("source_manifest")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != ACTIVE_MAP_RECEIPT_ARTIFACT
        or value.get("protocol_sha256") != protocol_digest
        or not isinstance(construction, dict)
        or construction.get("method") != ACTIVE_MAP_CONSTRUCTION_METHOD
        or construction.get("same_session_volume_used") is not False
        or construction.get("source_coverage_complete") is not True
        or construction.get("availability_rule") != ACTIVE_MAP_AVAILABILITY_RULE
        or not isinstance(map_record, dict)
        or not isinstance(source_record, dict)
    ):
        raise ValueError("active-contract construction receipt has the wrong identity")
    registered_columns = value.get("required_columns")
    if (
        not isinstance(registered_columns, list)
        or not set(ACTIVE_MAP_REQUIRED_COLUMNS).issubset(map(str, registered_columns))
    ):
        raise ValueError("active-contract receipt does not bind the mandatory proof schema")
    bound_map = _resolved_from(proof_path.parent, map_record.get("path"))
    if (
        bound_map != active_map
        or not active_map.is_file()
        or map_record.get("sha256") != sha256(active_map)
        or map_record.get("bytes") != active_map.stat().st_size
    ):
        raise ValueError("active-contract map differs from its construction receipt")
    source = _resolved_from(proof_path.parent, source_record.get("path"))
    if source == active_map:
        raise ValueError("active-contract map cannot be its own source universe")
    universe = _validate_source_universe(
        source,
        expected_sha256=source_record.get("sha256"),
        expected_bytes=source_record.get("bytes"),
    )
    if value.get("eligible_phase_universe") != {
        name: universe[name]
        for name in (
            "phase", "start", "end_exclusive", "session_count", "session_rows_sha256",
            "archive_root",
        )
    }:
        raise ValueError("active-contract receipt differs from its eligible source universe")
    return proof_path, value, universe


def _strict_boolean_column(mapping: pd.DataFrame, name: str) -> pd.Series:
    values = mapping[name]
    if values.isna().any() or not bool(values.map(lambda value: isinstance(value, (bool, np.bool_))).all()):
        raise ValueError(f"active-contract map column {name} must contain only booleans")
    return values.astype(bool)


def _validated_active_map_frame(active_map: Path) -> pd.DataFrame:
    mapping = pq.read_table(active_map).to_pandas()
    required = set(ACTIVE_MAP_REQUIRED_COLUMNS)
    if missing := required.difference(mapping.columns):
        raise ValueError(f"active-contract map is missing {sorted(missing)}")
    if mapping.empty:
        raise ValueError("active-contract map has no selected sessions")
    mapping["session_date"] = mapping["session_date"].astype(str)
    for day in mapping["session_date"]:
        try:
            parsed = date.fromisoformat(str(day))
        except ValueError as error:
            raise ValueError("active-contract map contains a malformed session date") from error
        if str(parsed) != str(day):
            raise ValueError("active-contract map session dates must be canonical YYYY-MM-DD")
    if mapping["session_date"].duplicated().any():
        raise ValueError("active-contract map contains duplicate session dates")
    symbols = mapping["selected_raw_symbol"]
    if symbols.isna().any() or not bool(symbols.map(lambda value: bool(str(value).strip())).all()):
        raise ValueError("active-contract map contains an empty selected symbol")
    mapping["selected_raw_symbol"] = symbols.astype(str).str.strip()
    same_session = _strict_boolean_column(mapping, "same_session_volume_used")
    coverage = _strict_boolean_column(mapping, "source_coverage_complete")
    if bool(same_session.any()):
        raise ValueError("active-contract map used current-session volume")
    if not bool(coverage.all()):
        raise ValueError("active-contract map includes incomplete source coverage")
    prior = mapping["prior_completed_session"].astype(str)
    for value in prior:
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("active-contract map prior session is malformed") from error
        if str(parsed) != value:
            raise ValueError("active-contract prior sessions must be canonical YYYY-MM-DD")
    if not bool((prior < mapping["session_date"]).all()):
        raise ValueError("active-contract map is not past-only")
    cutoff = pd.to_numeric(mapping["volume_cutoff_available_time_ns"], errors="coerce")
    decision = pd.to_numeric(mapping["decision_time_utc_ns"], errors="coerce")
    if (
        cutoff.isna().any()
        or decision.isna().any()
        or not bool(np.isfinite(cutoff.to_numpy(np.float64)).all())
        or not bool(np.isfinite(decision.to_numpy(np.float64)).all())
        or bool((cutoff < 0).any())
        or not bool((cutoff < decision).all())
    ):
        raise ValueError("roll information was unavailable at its decision")
    return mapping.sort_values("session_date", kind="stable").reset_index(drop=True)


def create_eligible_phase_universe(
    active_map: Path,
    archive_root: Path,
    output: Path,
    *,
    phase: str,
    start: str,
    end_exclusive: str,
    training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> Path:
    """Hash the exact authorized archive partitions behind an active map.

    This is the required first registration step.  It enumerates only the
    authorized interval, requires both source layers for every included date,
    proves the selected symbol exists in both, and binds every Parquet child by
    relative path, byte length, and SHA-256.
    """

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable V6 artifact: {destination}")
    validate_build_access(
        phase, start, end_exclusive,
        training_manifest=training_manifest, unlock_receipt=unlock_receipt,
    )
    archive_root = Path(archive_root).resolve()
    active_map = Path(active_map).resolve()
    if not archive_root.is_dir() or not active_map.is_file():
        raise ValueError("active map and authoritative archive root must exist")
    mapping = _validated_active_map_frame(active_map)
    if not bool(
        ((mapping["session_date"] >= start) & (mapping["session_date"] < end_exclusive)).all()
    ):
        raise ValueError("active map contains sessions outside the universe interval")
    archive_dates = _archive_session_dates(archive_root, start, end_exclusive)
    if tuple(mapping["session_date"].astype(str)) != archive_dates:
        raise ValueError("active map is not the exact authoritative archive date inventory")

    def evidence(layer: str, day: str, symbol: str) -> list[dict[str, object]]:
        partition = _source_partition(archive_root, layer, day)
        records: list[dict[str, object]] = []
        symbol_seen = False
        for child in sorted(partition.glob("*.parquet")):
            symbols = pq.ParquetFile(child).read(columns=["raw_symbol"])[
                "raw_symbol"
            ].to_pylist()
            symbol_seen |= any(str(value) == symbol for value in symbols)
            records.append({
                "path": os.path.relpath(child.resolve(), start=destination.parent),
                "sha256": sha256(child),
                "bytes": child.stat().st_size,
            })
        if not symbol_seen:
            raise ValueError(f"{day} {symbol}: selected symbol is absent from {layer}")
        return records

    rows: list[dict[str, object]] = []
    for item in mapping.itertuples(index=False):
        day = str(item.session_date)
        symbol = str(item.selected_raw_symbol)
        rows.append({
            "session_date": day,
            "raw_symbol": symbol,
            "l2_depth_files": evidence("l2_depth", day, symbol),
            "l1_trade_files": evidence("l1_trades", day, symbol),
        })
    _, protocol_digest = frozen_protocol()
    value = {
        "schema_version": 1,
        "artifact_type": ACTIVE_MAP_SOURCE_ARTIFACT,
        "protocol_sha256": protocol_digest,
        "phase": phase,
        "start": start,
        "end_exclusive": end_exclusive,
        "archive_root": os.path.relpath(archive_root, start=destination.parent),
        "eligible_phase_universe_complete": True,
        "source_coverage_complete": True,
        "session_count": len(rows),
        "session_rows_sha256": canonical_json_sha256(rows),
        "sessions": rows,
    }
    work = destination.with_name(f".{destination.name}.part")
    if work.exists():
        raise FileExistsError(f"stale universe registration exists: {work}")
    atomic_json(work, value)
    try:
        _validate_source_universe(work)
        os.replace(work, destination)
    finally:
        work.unlink(missing_ok=True)
    _validate_source_universe(destination)
    return destination


def create_active_map_receipt(
    active_map: Path,
    source_manifest: Path,
    output: Path | None = None,
    *,
    training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> Path:
    """Validate and atomically register a past-only active-contract map."""

    active_map = Path(active_map).resolve()
    source_manifest = Path(source_manifest).resolve()
    if source_manifest == active_map:
        raise ValueError("active map and its construction source must be distinct")
    source_phase, source_start, source_end = _source_access_identity(source_manifest)
    validate_build_access(
        source_phase, source_start, source_end,
        training_manifest=training_manifest, unlock_receipt=unlock_receipt,
    )
    if not active_map.is_file():
        raise ValueError("active map must exist after phase authorization")
    universe = _validate_source_universe(source_manifest)
    mapping = _validated_active_map_frame(active_map)
    _assert_map_equals_source_universe(mapping, universe)
    path = (
        active_map_receipt_path(active_map)
        if output is None else Path(output).resolve()
    )
    _, protocol_digest = frozen_protocol()

    def relative(product: Path) -> str:
        return os.path.relpath(product, start=path.parent)

    value = {
        "schema_version": 1,
        "artifact_type": ACTIVE_MAP_RECEIPT_ARTIFACT,
        "protocol_sha256": protocol_digest,
        "required_columns": list(ACTIVE_MAP_REQUIRED_COLUMNS),
        "row_count": len(mapping),
        "first_session": str(mapping.iloc[0]["session_date"]),
        "last_session": str(mapping.iloc[-1]["session_date"]),
        "construction": {
            "method": ACTIVE_MAP_CONSTRUCTION_METHOD,
            "same_session_volume_used": False,
            "source_coverage_complete": True,
            "availability_rule": ACTIVE_MAP_AVAILABILITY_RULE,
        },
        "eligible_phase_universe": {
            name: universe[name]
            for name in (
                "phase", "start", "end_exclusive", "session_count",
                "session_rows_sha256", "archive_root",
            )
        },
        "active_contract_map": {
            "path": relative(active_map),
            "sha256": sha256(active_map),
            "bytes": active_map.stat().st_size,
        },
        "source_manifest": {
            "path": relative(source_manifest),
            "sha256": sha256(source_manifest),
            "bytes": source_manifest.stat().st_size,
        },
    }
    return atomic_json(path, value)


def validate_active_contract_map(
    active_map: Path,
    start: str,
    end_exclusive: str,
    *,
    receipt_path: Path | None = None,
    phase: str,
    training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> pd.DataFrame:
    """Load a receipt-bound map and enforce its row-level past-only evidence."""

    validate_build_access(
        phase, start, end_exclusive,
        training_manifest=training_manifest, unlock_receipt=unlock_receipt,
    )
    active_map = Path(active_map).resolve()
    _, receipt, universe = _load_active_map_receipt(active_map, receipt_path)
    mapping = _validated_active_map_frame(active_map)
    _assert_map_equals_source_universe(mapping, universe)
    if (
        receipt.get("row_count") != len(mapping)
        or receipt.get("first_session") != str(mapping.iloc[0]["session_date"])
        or receipt.get("last_session") != str(mapping.iloc[-1]["session_date"])
    ):
        raise ValueError("active-contract receipt row count differs from its map")
    if universe.get("phase") != phase:
        raise ValueError("active-contract source universe belongs to another phase")
    source_start = str(universe["start"])
    source_end = str(universe["end_exclusive"])
    if phase == "smoke":
        if not source_start <= start < end_exclusive <= source_end:
            raise ValueError("smoke interval leaves its source universe")
    elif (start, end_exclusive) != (source_start, source_end):
        raise ValueError("registered build must consume its exact eligible phase universe")
    mapping = mapping[
        (mapping["session_date"] >= start) & (mapping["session_date"] < end_exclusive)
    ].copy()
    if mapping.empty:
        raise ValueError("active-contract map has no selected sessions")
    return mapping.rename(columns={"session_date": "v6_session_date"}).sort_values(
        "v6_session_date", kind="stable",
    )


def _sidecar_arrays(raw: Mapping[str, np.ndarray], source: Path) -> dict[str, np.ndarray]:
    day = source.stem
    start_clock = int(np.asarray(raw["clock_ns"])[0])
    end_clock = int(np.asarray(raw["clock_ns"])[-1])
    trade_clock = np.asarray(raw["trade_available_ns"], dtype=np.int64)
    start = int(np.searchsorted(trade_clock, start_clock, side="left"))
    end = int(np.searchsorted(trade_clock, end_clock, side="right"))
    if end <= start:
        raise ValueError(f"{day}: session contains no causally available trades")
    features = np.asarray(raw["trade_features"][start:end], dtype=np.float32).copy()
    sizes = np.expm1(features[:, 3].astype(np.float64)).clip(min=0.0)
    signs = features[:, 2].astype(np.float32)
    prices = _absolute_trade_prices(raw)[start:end]
    mlofi = multi_level_ofi(
        np.asarray(raw["levels"]), np.asarray(raw["valid"]),
        np.asarray(raw["continuity_segment"]), np.asarray(raw["best_bid_ticks"]),
        np.asarray(raw["best_ask_ticks"]),
    )
    return {
        "session_date": np.asarray(day),
        "source_sha256": np.asarray(sha256(source)),
        "session_start_ns": np.asarray(start_clock, dtype=np.int64),
        "session_end_ns": np.asarray(end_clock, dtype=np.int64),
        "source_trade_start": np.asarray(start, dtype=np.int64),
        "source_trade_end": np.asarray(end, dtype=np.int64),
        "trade_available_ns": trade_clock[start:end],
        "trade_price_ticks": prices.astype(np.int64),
        "trade_size": sizes.astype(np.float32),
        "trade_sign": signs,
        "trade_features": features,
        "mlofi": mlofi.astype(np.float32),
        "session_volume": np.asarray(float(sizes.sum()), dtype=np.float64),
    }


def _valid_existing_pair(
    work: Path, day: str, symbol: str, record: Mapping[str, object],
) -> bool:
    raw = work / "sessions" / f"{day}.npz"
    side = work / "sidecars" / f"{day}.npz"
    audit = record.get("audit")
    derived_valid = (
        record.get("raw_symbol") == symbol
        and isinstance(audit, dict)
        and audit.get("status") == "PASS"
        and audit.get("continuity_segments") == 1
        and raw.is_file() and side.is_file()
        and record.get("session_sha256") == sha256(raw)
        and record.get("sidecar_sha256") == sha256(side)
    )
    sources = record.get("sources")
    if not derived_valid or not isinstance(sources, dict) or not sources:
        return False
    for name, source_record in sources.items():
        if not isinstance(source_record, dict):
            return False
        source = Path(str(name))
        if (
            not source.is_file()
            or source.stat().st_size != source_record.get("bytes")
            or sha256(source) != source_record.get("sha256")
        ):
            return False
    return True


def build_dataset(
    archive_root: Path, active_map: Path, output: Path, *,
    start: str, end_exclusive: str, phase: str,
    training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
    active_map_receipt: Path | None = None,
    limit_sessions: int | None = None,
) -> Path:
    """Build paired V2-compatible raw shards and V6 volume sidecars."""

    validate_build_access(
        phase, start, end_exclusive, training_manifest=training_manifest,
        unlock_receipt=unlock_receipt,
    )
    if output.exists():
        raise FileExistsError("V6 GC dataset is immutable")
    _, protocol_digest = frozen_protocol()
    archive_root = Path(archive_root).resolve()
    active_map = Path(active_map).resolve()
    receipt_path, _, universe = _load_active_map_receipt(active_map, active_map_receipt)
    if archive_root != Path(str(universe["archive_root"])).resolve():
        raise ValueError("build archive_root differs from the authoritative source universe")
    resume_identity = {
        "protocol_sha256": protocol_digest,
        "phase": phase,
        "start": start,
        "end_exclusive": end_exclusive,
        "archive_root": str(archive_root),
        "active_contract_map": str(active_map),
        "active_contract_map_sha256": sha256(active_map),
        "active_contract_map_receipt": str(receipt_path),
        "active_contract_map_receipt_sha256": sha256(receipt_path),
        "limit_sessions": limit_sessions,
    }
    mapping = validate_active_contract_map(
        active_map, start, end_exclusive, receipt_path=receipt_path, phase=phase,
        training_manifest=training_manifest, unlock_receipt=unlock_receipt,
    )
    if limit_sessions is not None:
        if phase != "smoke" or limit_sessions < 1:
            raise ValueError("a positive session cap is allowed only for smoke")
        mapping = mapping.head(limit_sessions)
    work = output.with_name(f".{output.name}.part")
    work.mkdir(parents=True, exist_ok=True)
    progress_path = work / "progress.json"
    resumed = progress_path.is_file()
    progress: dict[str, object] = (
        json.loads(progress_path.read_text()) if progress_path.is_file() else {
            "schema_version": 1, "artifact_type": "V6_GC_BUILD_PROGRESS",
            "identity": resume_identity, "sessions": {},
        }
    )
    if (
        progress.get("schema_version") != 1
        or progress.get("artifact_type") != "V6_GC_BUILD_PROGRESS"
        or progress.get("identity") != resume_identity
    ):
        raise ValueError("V6 build resume identity differs from the current request")
    records = progress.get("sessions")
    if not isinstance(records, dict):
        raise ValueError("V6 build progress is malformed")
    audits: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    for item in mapping.itertuples(index=False):
        day = str(item.v6_session_date)
        symbol = str(item.selected_raw_symbol)
        old = records.get(day)
        if isinstance(old, dict) and _valid_existing_pair(work, day, symbol, old):
            audits.append(dict(old["audit"]))
            old_sources = old.get("sources", {})
            if isinstance(old_sources, dict):
                sources.update({str(key): dict(value) for key, value in old_sources.items()
                                if isinstance(value, dict)})
            continue
        l2_rows, l2_files = _read_day(archive_root, "l2_depth", day, symbol, [
            "event_index", "event_row_index", "ts_recv_ns", "continuity_segment_id",
            "operation", "side", "price_ticks", "level_size", "raw_symbol",
        ])
        trade_rows, trade_files = _read_day(archive_root, "l1_trades", day, symbol, [
            "ts_recv_ns", "global_record_ordinal", "price_ticks", "size", "aggressor_side",
            "is_bad_ts_recv", "raw_symbol",
        ])
        if l2_rows.empty or trade_rows.empty:
            raise ValueError(f"{day} {symbol}: empty active-contract source")
        segment_values = l2_rows["continuity_segment_id"]
        continuity_segments = int(segment_values.nunique(dropna=False))
        if segment_values.isna().any() or continuity_segments != 1:
            raise ValueError(
                f"{day} {symbol}: registered session must contain exactly one continuity segment"
            )
        l2, rejected = reconstruct_l2(l2_rows, _session_clocks(day))
        arrival_5ms = reconstruct_arrival_bbo(l2_rows, l2["clock_ns"], 5)
        arrival_20ms = reconstruct_arrival_bbo(l2_rows, l2["clock_ns"], 20)
        trades = build_trade_features(trade_rows, l2, day)
        arrays = {
            **l2, **arrival_5ms, **arrival_20ms, **trades,
            "session_date": np.asarray(day), "raw_symbol": np.asarray(symbol),
        }
        raw_path = work / "sessions" / f"{day}.npz"
        _atomic_npz(raw_path, arrays)
        side_path = work / "sidecars" / f"{day}.npz"
        _atomic_npz(side_path, _sidecar_arrays(arrays, raw_path))
        decisions = np.asarray(trades["decision_l2_end"], dtype=np.int64)
        previous = 0
        interval_covered: list[bool] = []
        for endpoint in decisions:
            interval_covered.append(bool(np.asarray(l2["valid"])[previous:int(endpoint)].any()))
            previous = int(endpoint)
        five_minute_coverage = float(np.mean(interval_covered))
        audit = SessionAudit(
            day, symbol, len(l2_rows), len(trade_rows), len(l2["valid"]),
            int(np.asarray(l2["valid"]).sum()),
            float(np.asarray(l2["valid"]).mean()), five_minute_coverage,
            continuity_segments,
            rejected["crossed"], rejected["stale"], rejected["incomplete"], rejected["unknown"],
            "PASS" if five_minute_coverage >= 0.95 and rejected["unknown"] == 0 else "FAIL",
        )
        if audit.status != "PASS":
            raise ValueError(f"{day}: failed the frozen session quality gate")
        audit_value = asdict(audit)
        source_records: dict[str, dict[str, object]] = {}
        for source in (*l2_files, *trade_files):
            absolute = str(source.resolve())
            source_records[absolute] = {
                "sha256": sha256(source), "bytes": source.stat().st_size,
            }
        sources.update(source_records)
        records[day] = {
            "raw_symbol": symbol,
            "session_sha256": sha256(raw_path),
            "sidecar_sha256": sha256(side_path),
            "audit": audit_value,
            "sources": source_records,
        }
        audits.append(audit_value)
        atomic_json(progress_path, progress, immutable=False)
    if not audits:
        raise ValueError("no V6 sessions were built")
    selected_days = tuple(str(value) for value in mapping["v6_session_date"])
    if set(records) != set(selected_days) or len(records) != len(selected_days):
        raise ValueError("V6 build progress contains sessions outside the selected active map")
    manifest = {
        "schema_version": 1,
        "artifact_type": "V6_GC_CAUSAL_SESSION_DATASET",
        "protocol_sha256": protocol_digest,
        "phase": phase,
        "classification": "FRESH_OOS" if phase == "test" else "DEVELOPMENT",
        "start": start,
        "end_exclusive": end_exclusive,
        "session_count": len(audits),
        "active_contract_map": {"path": str(active_map.resolve()), "sha256": sha256(active_map)},
        "active_contract_map_receipt": {
            "path": str(receipt_path), "sha256": sha256(receipt_path),
        },
        "source_files": sources,
        "sessions": dict(sorted(records.items())),
        "audits": sorted(audits, key=lambda value: str(value["session_date"])),
        "observation_excludes": ["arrival_5ms_*", "arrival_20ms_*"],
        "arrival_arrays_role": "simulator_only",
        "access_authorization": (
            {"kind": "complete_training_manifest", "path": str(training_manifest),
             "sha256": sha256(training_manifest)}
            if phase == "validation" and training_manifest is not None else
            {"kind": "test_unlock_receipt", "path": str(unlock_receipt),
             "sha256": sha256(unlock_receipt)}
            if phase == "test" and unlock_receipt is not None else None
        ),
        "resumed": resumed,
    }
    atomic_json(work / "manifest.json", manifest, immutable=False)
    progress_path.unlink(missing_ok=True)
    os.replace(work, output)
    return output / "manifest.json"


__all__ = [
    "ACTIVE_MAP_AVAILABILITY_RULE",
    "ACTIVE_MAP_CONSTRUCTION_METHOD",
    "ACTIVE_MAP_RECEIPT_ARTIFACT",
    "ACTIVE_MAP_REQUIRED_COLUMNS",
    "ACTIVE_MAP_SOURCE_ARTIFACT",
    "active_map_receipt_path",
    "build_dataset",
    "create_active_map_receipt",
    "create_eligible_phase_universe",
    "validate_active_contract_map",
    "validate_build_access",
]
