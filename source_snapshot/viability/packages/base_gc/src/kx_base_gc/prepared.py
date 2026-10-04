from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .bars import BAR_FEATURES, BOUNDARY_COLUMNS, build_causal_timeframes, validate_m5_bars
from .contract import BaselineContract
from .sessions import require_exact_globex_session_dates


PREPARED_SCHEMA_VERSION = 1
PREPARED_ARTIFACT_TYPE = "BASE_GC_PREPARED_HIERARCHY_BARS"
PREPARED_PRICE_UNIT = "exchange_ticks"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_CLASSES = {"SYNTHETIC_ONLY", "DEVELOPMENT_ONLY", "LOCKED_EMPIRICAL"}
_VERIFIED_TOKEN = object()


class PreparedBarsError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedBarsEvidence:
    schema_version: int
    evidence_class: str
    bars_sha256: str
    manifest_sha256: str
    registered_split_sha256: str
    active_contract_map_sha256: str
    minimum_date: str
    maximum_date: str
    declared_rows: int
    price_unit: str
    arrival_fills_certified: bool
    causal_certificate_sha256: str
    active_contract_certificate_sha256: str
    arrival_fill_certificate_sha256: str | None = None
    verified_bars_path: str | None = None
    verified_manifest_path: str | None = None
    active_contract_ledger_path: str | None = None
    arrival_fill_ledger_path: str | None = None
    _verification_token: object | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.schema_version != PREPARED_SCHEMA_VERSION:
            raise PreparedBarsError("unsupported prepared-bars evidence schema")
        if self.evidence_class not in _EVIDENCE_CLASSES:
            raise PreparedBarsError("invalid prepared-bars evidence class")
        for name in (
            "bars_sha256",
            "manifest_sha256",
            "registered_split_sha256",
            "active_contract_map_sha256",
            "causal_certificate_sha256",
            "active_contract_certificate_sha256",
        ):
            if not _SHA256.fullmatch(str(getattr(self, name))):
                raise PreparedBarsError(f"{name} must be a lowercase SHA-256")
        if date.fromisoformat(self.minimum_date) > date.fromisoformat(self.maximum_date):
            raise PreparedBarsError("prepared period is reversed")
        if self.declared_rows < 1:
            raise PreparedBarsError("prepared bar product must contain rows")
        if self.price_unit != PREPARED_PRICE_UNIT:
            raise PreparedBarsError(
                "prepared OHLC/fill values must be explicitly exchange tick counts"
            )
        if (
            self.arrival_fill_certificate_sha256 is not None
            and not _SHA256.fullmatch(self.arrival_fill_certificate_sha256)
        ):
            raise PreparedBarsError("arrival fill certificate hash is invalid")
        if self.arrival_fills_certified != (
            self.arrival_fill_certificate_sha256 is not None
        ):
            raise PreparedBarsError("arrival fill certification fields disagree")
        if self.evidence_class != "SYNTHETIC_ONLY":
            if self._verification_token is not _VERIFIED_TOKEN:
                raise PreparedBarsError(
                    "empirical evidence must come from the verified prepared-product loader"
                )
            for value in (
                self.verified_bars_path,
                self.verified_manifest_path,
                self.active_contract_ledger_path,
            ):
                if value is None or not Path(value).is_file():
                    raise PreparedBarsError("resolved empirical evidence artifact is missing")
            if self.arrival_fills_certified and (
                self.arrival_fill_ledger_path is None
                or not Path(self.arrival_fill_ledger_path).is_file()
            ):
                raise PreparedBarsError("resolved arrival-fill ledger is missing")


@dataclass(frozen=True)
class PreparedHierarchyProduct:
    timeframes: Mapping[str, pd.DataFrame]
    evidence: PreparedBarsEvidence
    allowed_end_date: str
    maximum_rows: int
    _verification_token: object | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        self.evidence.validate()
        if self._verification_token is not _VERIFIED_TOKEN:
            raise PreparedBarsError("prepared hierarchy product was not created by its loader")
        if set(self.timeframes) != {"H1", "M15", "M5"}:
            raise PreparedBarsError("prepared hierarchy product lacks required timeframes")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_claim(
    claims: Mapping[str, Any],
    name: str,
    *,
    manifest_path: Path,
    expected_artifact_type: str,
    bars_sha256: str,
) -> tuple[Mapping[str, Any], str, Path]:
    claim = claims.get(name)
    if not isinstance(claim, Mapping) or claim.get("passed") is not True:
        raise PreparedBarsError(f"prepared manifest requires passed claim {name!r}")
    expected_hash = str(claim.get("sha256", ""))
    raw_artifact = claim.get("artifact")
    if not _SHA256.fullmatch(expected_hash) or not isinstance(raw_artifact, str):
        raise PreparedBarsError(f"claim {name!r} lacks a typed artifact and SHA-256")
    relative = Path(raw_artifact)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparedBarsError(f"claim {name!r} artifact must be manifest-relative")
    artifact = (manifest_path.parent / relative).resolve()
    if not artifact.is_relative_to(manifest_path.parent.resolve()) or not artifact.is_file():
        raise PreparedBarsError(f"claim {name!r} artifact is missing or out of scope")
    observed_hash = sha256_file(artifact)
    if observed_hash != expected_hash:
        raise PreparedBarsError(f"claim {name!r} artifact hash mismatch")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PreparedBarsError(f"claim {name!r} artifact must be a JSON object")
    if (
        int(payload.get("schema_version", 0)) != 1
        or payload.get("artifact_type") != expected_artifact_type
        or payload.get("claim") != name
        or payload.get("passed") is not True
        or payload.get("prepared_bars_sha256") != bars_sha256
    ):
        raise PreparedBarsError(f"claim {name!r} typed artifact contract failed")
    return payload, observed_hash, artifact


def _nested_artifact(
    declaration: Any,
    *,
    parent: Path,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(declaration, Mapping):
        raise PreparedBarsError(f"{label} declaration is missing")
    raw_path = declaration.get("artifact")
    expected_hash = str(declaration.get("sha256", ""))
    if not isinstance(raw_path, str) or not _SHA256.fullmatch(expected_hash):
        raise PreparedBarsError(f"{label} must bind an artifact and SHA-256")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreparedBarsError(f"{label} path must be certificate-relative")
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file():
        raise PreparedBarsError(f"{label} artifact is missing or out of scope")
    if sha256_file(path) != expected_hash:
        raise PreparedBarsError(f"{label} artifact hash mismatch")
    return path, expected_hash


def verify_prepared_manifest(
    bars_path: str | Path,
    manifest_path: str | Path,
    *,
    allowed_end_date: str = "2025-08-31",
) -> PreparedBarsEvidence:
    """Verify all declarations before opening the prepared Parquet product."""

    bars_source = Path(bars_path).expanduser().resolve()
    manifest_source = Path(manifest_path).expanduser().resolve()
    if bars_source.suffix.lower() != ".parquet" or not bars_source.is_file():
        raise PreparedBarsError("--bars must be one explicit prepared Parquet file")
    if not manifest_source.is_file():
        raise PreparedBarsError("prepared evidence manifest does not exist")
    payload = json.loads(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PreparedBarsError("prepared evidence manifest must be a JSON object")
    if int(payload.get("schema_version", 0)) != PREPARED_SCHEMA_VERSION:
        raise PreparedBarsError("unsupported prepared evidence manifest schema")
    if payload.get("artifact_type") != PREPARED_ARTIFACT_TYPE:
        raise PreparedBarsError("manifest is not a Base-GC prepared-bar product")
    evidence_class = str(payload.get("evidence_class", ""))
    if evidence_class not in _EVIDENCE_CLASSES:
        raise PreparedBarsError("invalid prepared evidence class")
    bars = payload.get("bars")
    if not isinstance(bars, Mapping):
        raise PreparedBarsError("manifest must contain a bars declaration")
    expected_bars_hash = str(bars.get("sha256", ""))
    observed_bars_hash = sha256_file(bars_source)
    if expected_bars_hash != observed_bars_hash:
        raise PreparedBarsError("prepared bar Parquet hash mismatch")
    if str(bars.get("filename", bars_source.name)) != bars_source.name:
        raise PreparedBarsError("prepared bar filename differs from its manifest")
    period = payload.get("period")
    if not isinstance(period, Mapping):
        raise PreparedBarsError("manifest must declare its calendar period")
    minimum_date = str(period.get("start", ""))
    maximum_date = str(period.get("end", ""))
    try:
        minimum = date.fromisoformat(minimum_date)
        maximum = date.fromisoformat(maximum_date)
        allowed = date.fromisoformat(allowed_end_date)
    except ValueError as error:
        raise PreparedBarsError("manifest contains an invalid ISO date") from error
    if minimum > maximum or maximum > allowed:
        raise PreparedBarsError(
            f"prepared period {minimum} through {maximum} exceeds allowed end {allowed}"
        )
    split = payload.get("registered_split")
    if not isinstance(split, Mapping) or not _SHA256.fullmatch(str(split.get("sha256", ""))):
        raise PreparedBarsError("manifest must bind a registered split SHA-256")
    claims = payload.get("claims")
    if not isinstance(claims, Mapping):
        raise PreparedBarsError("manifest must declare certification claims")
    causal_claim, causal_certificate_hash, _ = _required_claim(
        claims,
        "causal_completed_bars",
        manifest_path=manifest_source,
        expected_artifact_type="BASE_GC_CAUSAL_COMPLETED_BARS_CERTIFICATE",
        bars_sha256=observed_bars_hash,
    )
    if (
        causal_claim.get("price_unit") != PREPARED_PRICE_UNIT
        or causal_claim.get("clock_contract")
        != "bar_end<=available<=decision<=submission<=arrival"
        or set(causal_claim.get("timeframes", ())) != {"H1", "M15", "M5"}
    ):
        raise PreparedBarsError("causal completed-bar certificate semantics are invalid")
    active_claim, active_certificate_hash, active_certificate_path = _required_claim(
        claims,
        "active_contract_map",
        manifest_path=manifest_source,
        expected_artifact_type="BASE_GC_ACTIVE_CONTRACT_CERTIFICATE",
        bars_sha256=observed_bars_hash,
    )
    active_hash = str(active_claim.get("map_sha256", ""))
    if not _SHA256.fullmatch(active_hash):
        raise PreparedBarsError("active-contract claim lacks its map SHA-256")
    active_ledger_path, active_ledger_hash = _nested_artifact(
        active_claim.get("map_ledger"),
        parent=active_certificate_path.parent,
        label="active-contract assignment ledger",
    )
    if active_hash != active_ledger_hash:
        raise PreparedBarsError("active map hash must equal the resolved assignment ledger hash")
    active_ledger = json.loads(active_ledger_path.read_text(encoding="utf-8"))
    if (
        not isinstance(active_ledger, Mapping)
        or int(active_ledger.get("schema_version", 0)) != 1
        or active_ledger.get("artifact_type")
        != "BASE_GC_ACTIVE_CONTRACT_ASSIGNMENT_LEDGER"
        or active_ledger.get("passed") is not True
        or active_ledger.get("past_only") is not True
        or int(active_ledger.get("minimum_lag_sessions", 0)) < 1
        or not isinstance(active_ledger.get("assignments"), list)
    ):
        raise PreparedBarsError("active-contract assignment ledger is not past-only certified")
    for assignment in active_ledger["assignments"]:
        if not isinstance(assignment, Mapping):
            raise PreparedBarsError("active-contract ledger contains an invalid assignment")
        session_date = date.fromisoformat(str(assignment.get("date", "")))
        source_through = date.fromisoformat(str(assignment.get("source_sessions_through", "")))
        if source_through >= session_date or not assignment.get("raw_symbol"):
            raise PreparedBarsError("active-contract assignment violates its past-only lag")
    arrival_claim = claims.get("arrival_aligned_aggressive_fills")
    arrival_certified = isinstance(arrival_claim, Mapping) and arrival_claim.get("passed") is True
    arrival_certificate_hash: str | None = None
    arrival_ledger_path: Path | None = None
    if arrival_certified:
        arrival_artifact, arrival_certificate_hash, arrival_certificate_path = _required_claim(
            claims,
            "arrival_aligned_aggressive_fills",
            manifest_path=manifest_source,
            expected_artifact_type="BASE_GC_ARRIVAL_FILL_CERTIFICATE",
            bars_sha256=observed_bars_hash,
        )
        required_fill_columns = {
            "aggressive_sell_fill_ticks",
            "aggressive_buy_fill_ticks",
            "aggressive_fill_available_time_ns",
            "aggressive_fill_source_ordinal",
            "decision_time_ns",
            "submission_time_ns",
            "arrival_time_ns",
        }
        if (
            arrival_artifact.get("price_unit") != PREPARED_PRICE_UNIT
            or arrival_artifact.get("fill_clock_rule")
            != "fill_available_time_ns<=arrival_time_ns"
            or set(arrival_artifact.get("required_columns", ())) != required_fill_columns
        ):
            raise PreparedBarsError("arrival-fill certificate semantics are invalid")
        if arrival_artifact.get("derivation") != "causal_fifo_asof_registered_arrival_v1":
            raise PreparedBarsError("arrival-fill certificate lacks the registered derivation")
        arrival_ledger_path, _ = _nested_artifact(
            arrival_artifact.get("fill_ledger"),
            parent=arrival_certificate_path.parent,
            label="arrival-fill ledger",
        )
        replay_path, _ = _nested_artifact(
            arrival_artifact.get("source_replay_certificate"),
            parent=arrival_certificate_path.parent,
            label="source FIFO replay certificate",
        )
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if (
            not isinstance(replay, Mapping)
            or int(replay.get("schema_version", 0)) != 1
            or replay.get("artifact_type") != "BASE_GC_CAUSAL_FIFO_REPLAY_CERTIFICATE"
            or replay.get("passed") is not True
            or replay.get("complete_atomic_groups") is not True
            or int(replay.get("unresolved_sequence_gaps", -1)) != 0
        ):
            raise PreparedBarsError("arrival fills lack a complete causal FIFO replay certificate")
    evidence = PreparedBarsEvidence(
        schema_version=PREPARED_SCHEMA_VERSION,
        evidence_class=evidence_class,
        bars_sha256=observed_bars_hash,
        manifest_sha256=sha256_file(manifest_source),
        registered_split_sha256=str(split["sha256"]),
        active_contract_map_sha256=active_hash,
        minimum_date=minimum_date,
        maximum_date=maximum_date,
        declared_rows=int(bars.get("row_count", 0)),
        price_unit=str(bars.get("price_unit", "")),
        arrival_fills_certified=arrival_certified,
        causal_certificate_sha256=causal_certificate_hash,
        active_contract_certificate_sha256=active_certificate_hash,
        arrival_fill_certificate_sha256=arrival_certificate_hash,
        verified_bars_path=str(bars_source),
        verified_manifest_path=str(manifest_source),
        active_contract_ledger_path=str(active_ledger_path),
        arrival_fill_ledger_path=(
            None if arrival_ledger_path is None else str(arrival_ledger_path)
        ),
        _verification_token=_VERIFIED_TOKEN,
    )
    evidence.validate()
    return evidence


def _validate_agent_frame(
    frame: pd.DataFrame,
    *,
    contract: BaselineContract,
    require_fills: bool,
) -> pd.DataFrame:
    required = {
        "date",
        "reward_boundary_id",
        "active_contract_map_hash",
        "decision_time_ns",
        "submission_time_ns",
        "arrival_time_ns",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PreparedBarsError(f"prepared agent bars are missing columns: {missing}")
    result = validate_m5_bars(frame)
    for column in ("decision_time_ns", "submission_time_ns", "arrival_time_ns"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    decision_floor = result["available_time_ns"] + int(
        contract.execution.decision_latency_ms * 1_000_000
    )
    arrival_floor = result["submission_time_ns"] + int(
        contract.execution.route_latency_ms * 1_000_000
    )
    if bool((result["decision_time_ns"] < decision_floor).any()):
        raise PreparedBarsError("prepared decision clock violates registered latency")
    if bool((result["submission_time_ns"] < result["decision_time_ns"]).any()):
        raise PreparedBarsError("prepared submission precedes decision")
    if bool((result["arrival_time_ns"] < arrival_floor).any()):
        raise PreparedBarsError("prepared arrival violates registered route latency")
    if bool(result["reward_boundary_id"].astype(str).eq("").any()):
        raise PreparedBarsError("reward_boundary_id cannot be empty")
    if require_fills:
        fill_columns = {
            "aggressive_sell_fill_ticks",
            "aggressive_buy_fill_ticks",
            "aggressive_fill_available_time_ns",
            "aggressive_fill_source_ordinal",
        }
        missing_fills = sorted(fill_columns - set(result.columns))
        if missing_fills:
            raise PreparedBarsError(
                f"BaseStrong prepared bars lack certified fill columns: {missing_fills}"
            )
        for column in fill_columns:
            result[column] = pd.to_numeric(result[column], errors="raise")
        invalid = (
            (result["aggressive_sell_fill_ticks"] <= 0)
            | (result["aggressive_buy_fill_ticks"] <= 0)
            | (
                result["aggressive_sell_fill_ticks"]
                > result["aggressive_buy_fill_ticks"]
            )
            | (
                result["aggressive_fill_available_time_ns"]
                > result["arrival_time_ns"]
            )
            | (result["aggressive_fill_source_ordinal"] <= 0)
        )
        if bool(invalid.any()):
            raise PreparedBarsError("prepared aggressive fill clocks/ticks are not causal")
    return result


def _ledger_frame(path: Path, *, artifact_type: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("schema_version", 0)) != 1
        or payload.get("artifact_type") != artifact_type
        or payload.get("passed") is not True
        or not isinstance(payload.get("rows"), list)
    ):
        raise PreparedBarsError(f"typed ledger {artifact_type} is invalid")
    return pd.DataFrame(payload["rows"])


def _crosscheck_evidence_ledgers(
    frame: pd.DataFrame,
    evidence: PreparedBarsEvidence,
    *,
    require_fills: bool,
) -> None:
    if evidence.active_contract_ledger_path is None:
        if evidence.evidence_class == "SYNTHETIC_ONLY":
            return
        raise PreparedBarsError("active-contract ledger was not resolved")
    active_payload = json.loads(
        Path(evidence.active_contract_ledger_path).read_text(encoding="utf-8")
    )
    assignments = active_payload.get("assignments", [])
    by_date: dict[str, Mapping[str, Any]] = {}
    for assignment in assignments:
        key = str(assignment["date"])
        if key in by_date:
            raise PreparedBarsError("active-contract ledger has duplicate session dates")
        by_date[key] = assignment
    for session_date, rows in frame.groupby(frame["date"].astype(str), sort=False):
        assignment = by_date.get(str(session_date))
        if assignment is None:
            raise PreparedBarsError(f"active-contract ledger lacks date {session_date}")
        if set(rows["raw_symbol"].astype(str)) != {str(assignment["raw_symbol"])}:
            raise PreparedBarsError("prepared symbol differs from past-only active assignment")
        selection_available = int(assignment.get("selection_available_time_ns", 0))
        if selection_available <= 0 or selection_available >= int(
            rows["decision_time_ns"].min()
        ):
            raise PreparedBarsError(
                "active-contract selection was not available before the session decisions"
            )
    if not require_fills:
        return
    if evidence.arrival_fill_ledger_path is None:
        raise PreparedBarsError("BaseStrong arrival-fill ledger was not resolved")
    ledger = _ledger_frame(
        Path(evidence.arrival_fill_ledger_path),
        artifact_type="BASE_GC_ARRIVAL_FILL_LEDGER",
    )
    keys = ["timeframe", *BOUNDARY_COLUMNS, "bar_end_ns"]
    values = [
        "decision_time_ns",
        "submission_time_ns",
        "arrival_time_ns",
        "aggressive_sell_fill_ticks",
        "aggressive_buy_fill_ticks",
        "aggressive_fill_available_time_ns",
        "aggressive_fill_source_ordinal",
    ]
    missing = sorted(set(keys + values + ["source_tape_sha256"]) - set(ledger.columns))
    if missing:
        raise PreparedBarsError(f"arrival-fill ledger lacks columns: {missing}")
    if len(ledger) != len(frame) or ledger.duplicated(keys).any() or frame.duplicated(keys).any():
        raise PreparedBarsError("arrival-fill ledger keys/count do not match prepared bars")
    if not ledger["source_tape_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise PreparedBarsError("arrival-fill ledger has an invalid source-tape hash")
    left = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = ledger.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if left[keys].astype(str).to_dict("records") != right[keys].astype(str).to_dict("records"):
        raise PreparedBarsError("arrival-fill ledger decision keys differ from prepared bars")
    for column in values:
        if not np.allclose(
            pd.to_numeric(left[column], errors="raise").to_numpy(dtype=float),
            pd.to_numeric(right[column], errors="raise").to_numpy(dtype=float),
            rtol=0,
            atol=0,
        ):
            raise PreparedBarsError(f"arrival-fill ledger differs in {column}")


def validate_prepared_hierarchy_frame(
    frame: pd.DataFrame,
    evidence: PreparedBarsEvidence,
    contract: BaselineContract,
    *,
    maximum_rows: int = 1_000_000,
) -> dict[str, pd.DataFrame]:
    evidence.validate()
    if len(frame) != evidence.declared_rows:
        raise PreparedBarsError("prepared row count differs from its manifest")
    if len(frame) > maximum_rows:
        raise PreparedBarsError(f"prepared product exceeds bounded limit {maximum_rows}")
    if "timeframe" not in frame:
        raise PreparedBarsError("prepared product requires a timeframe column")
    if "date" not in frame:
        raise PreparedBarsError("prepared product requires an exchange-session date")
    if set(frame["timeframe"].astype(str)) != {"H1", "M15", "M5"}:
        raise PreparedBarsError("prepared product must contain exactly H1/M15/M5 rows")
    try:
        exact_dates = require_exact_globex_session_dates(
            frame["date"], frame["bar_end_ns"], label="prepared bar"
        )
    except ValueError as error:
        raise PreparedBarsError(str(error)) from error
    frame = frame.copy()
    frame["date"] = exact_dates.to_numpy()
    if evidence.evidence_class != "SYNTHETIC_ONLY":
        cutoff_columns = {"scheduled_flatten_time_ns", "decision_eligible"}
        missing_cutoff = sorted(cutoff_columns - set(frame.columns))
        if missing_cutoff:
            raise PreparedBarsError(
                f"empirical prepared bars lack fixed-cutoff fields: {missing_cutoff}"
            )
        eligibility = frame["decision_eligible"]
        if not eligibility.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise PreparedBarsError("decision_eligible must be a strict boolean")
        declared_cutoffs = pd.to_numeric(
            frame["scheduled_flatten_time_ns"], errors="raise"
        ).astype("int64")
        expected_cutoffs = frame["date"].map(
            lambda value: int(
                pd.Timestamp(
                    f"{value} 12:25:00", tz="America/Chicago"
                ).tz_convert("UTC").value
            )
        ).astype("int64")
        if not np.array_equal(declared_cutoffs.to_numpy(), expected_cutoffs.to_numpy()):
            raise PreparedBarsError("prepared flatten clock is not exactly 12:25 Chicago")
        expected_eligible = frame["bar_end_ns"].astype("int64") <= expected_cutoffs
        if not np.array_equal(
            eligibility.astype(bool).to_numpy(), expected_eligible.to_numpy()
        ):
            raise PreparedBarsError("decision eligibility differs from the fixed cutoff")
        expected_boundaries = [
            f"base-gc|{session}|{'active' if eligible else 'post-cutoff'}"
            for session, eligible in zip(
                frame["date"], expected_eligible, strict=True
            )
        ]
        if frame["reward_boundary_id"].astype(str).tolist() != expected_boundaries:
            raise PreparedBarsError("reward boundary differs from fixed cutoff semantics")
    actual_dates = pd.to_datetime(frame["date"], errors="raise").dt.date
    if actual_dates.min().isoformat() != evidence.minimum_date:
        raise PreparedBarsError("prepared minimum date differs from its manifest")
    if actual_dates.max().isoformat() != evidence.maximum_date:
        raise PreparedBarsError("prepared maximum date differs from its manifest")
    active_hashes = set(frame["active_contract_map_hash"].astype(str))
    if active_hashes != {evidence.active_contract_map_sha256}:
        raise PreparedBarsError("prepared rows are bound to a different active-contract map")
    require_fills = contract.variant == "base_strong_gc"
    if require_fills and not evidence.arrival_fills_certified:
        raise PreparedBarsError(
            "BaseStrong requires a passed arrival_aligned_aggressive_fills claim"
        )
    _crosscheck_evidence_ledgers(frame, evidence, require_fills=require_fills)
    result = {
        name: _validate_agent_frame(
            frame.loc[frame["timeframe"].astype(str).eq(name)].drop(columns="timeframe"),
            contract=contract,
            require_fills=require_fills,
        ).reset_index(drop=True)
        for name in ("H1", "M15", "M5")
    }

    derived = build_causal_timeframes(result["M5"])
    keys = [*BOUNDARY_COLUMNS, "bar_end_ns"]
    for name in ("M15", "H1"):
        expected = derived[name].sort_values(keys, kind="mergesort").reset_index(drop=True)
        observed = result[name].sort_values(keys, kind="mergesort").reset_index(drop=True)
        if expected[keys].astype(str).to_dict("records") != observed[keys].astype(str).to_dict("records"):
            raise PreparedBarsError(f"prepared {name} calendar differs from causal M5 rollup")
        for column in (*BAR_FEATURES, "available_time_ns"):
            if not np.allclose(
                expected[column].to_numpy(dtype=float),
                observed[column].to_numpy(dtype=float),
                rtol=0,
                atol=0,
            ):
                raise PreparedBarsError(
                    f"prepared {name} {column} differs from causal M5 rollup"
                )
    return result


def load_certified_prepared_bars(
    bars_path: str | Path,
    manifest_path: str | Path,
    contract: BaselineContract,
    *,
    allowed_end_date: str = "2025-08-31",
    maximum_rows: int = 1_000_000,
) -> tuple[dict[str, pd.DataFrame], PreparedBarsEvidence]:
    evidence = verify_prepared_manifest(
        bars_path, manifest_path, allowed_end_date=allowed_end_date
    )
    frame = pd.read_parquet(Path(bars_path).expanduser().resolve())
    return (
        validate_prepared_hierarchy_frame(
            frame, evidence, contract, maximum_rows=maximum_rows
        ),
        evidence,
    )


def load_certified_prepared_product(
    bars_path: str | Path,
    manifest_path: str | Path,
    contract: BaselineContract,
    *,
    allowed_end_date: str = "2025-08-31",
    maximum_rows: int = 1_000_000,
) -> PreparedHierarchyProduct:
    timeframes, evidence = load_certified_prepared_bars(
        bars_path,
        manifest_path,
        contract,
        allowed_end_date=allowed_end_date,
        maximum_rows=maximum_rows,
    )
    product = PreparedHierarchyProduct(
        timeframes=timeframes,
        evidence=evidence,
        allowed_end_date=allowed_end_date,
        maximum_rows=maximum_rows,
        _verification_token=_VERIFIED_TOKEN,
    )
    product.validate()
    return product


def synthetic_prepared_product(
    timeframes: Mapping[str, pd.DataFrame],
    evidence: PreparedBarsEvidence,
) -> PreparedHierarchyProduct:
    if evidence.evidence_class != "SYNTHETIC_ONLY":
        raise PreparedBarsError(
            "detached in-memory prepared products are restricted to SYNTHETIC_ONLY"
        )
    product = PreparedHierarchyProduct(
        timeframes=dict(timeframes),
        evidence=evidence,
        allowed_end_date=evidence.maximum_date,
        maximum_rows=sum(len(value) for value in timeframes.values()),
        _verification_token=_VERIFIED_TOKEN,
    )
    product.validate()
    return product


def reverify_prepared_product(
    product: PreparedHierarchyProduct,
    contract: BaselineContract,
) -> PreparedHierarchyProduct:
    product.validate()
    if product.evidence.evidence_class == "SYNTHETIC_ONLY":
        return product
    if (
        product.evidence.verified_bars_path is None
        or product.evidence.verified_manifest_path is None
    ):
        raise PreparedBarsError("empirical product lacks resolved immutable paths")
    refreshed = load_certified_prepared_product(
        product.evidence.verified_bars_path,
        product.evidence.verified_manifest_path,
        contract,
        allowed_end_date=product.allowed_end_date,
        maximum_rows=product.maximum_rows,
    )
    if (
        refreshed.evidence.bars_sha256 != product.evidence.bars_sha256
        or refreshed.evidence.manifest_sha256 != product.evidence.manifest_sha256
    ):
        raise PreparedBarsError("prepared product changed between load and training")
    return refreshed
