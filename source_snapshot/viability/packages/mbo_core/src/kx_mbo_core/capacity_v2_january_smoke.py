"""Authenticated January-only capacity-v2 screening smoke.

This module is intentionally narrow.  It reuses the immutable corrected-roll
GC-lite L2 aggressive preflight roots and evaluates the frozen one-contract
capacity-v2 rule with trailing volume reconstructed from the *exact* L0 files
declared by each selected daily manifest.

Trade volume is available only after an instrument's complete ``F_LAST``
group has arrived plus the fixed decode latency.  The rolling interval is
``(arrival - five minutes, arrival]``.  No discovery glob is used at the raw
data trust boundary, and every consumed manifest and L0 part is re-hashed.

The output is ``SCREENING_ONLY``.  It is not Base-GC, RENT, HMADRL,
profitability, novelty, locked-validation, or final-OOS evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd

from .artifacts import sha256_file, sha256_json, write_json
from .capacity_v2 import (
    FROZEN_CAPACITY_V2,
    TRAILING_WINDOW_NS,
    ArrivalCapacityState,
    evaluate_capacity_v2,
)
from .config import load_yaml
from .l2_aggressive_preflight import (
    DEVELOPMENT_LABEL,
    NONFLAT_ACTION_IDS,
    PREFLIGHT_ID,
    SCREENING_LABEL,
)


SMOKE_ID = "gc_lite_v2_capacity_v2_january_l0_atomic_smoke_v2"
SUPERSEDED_SMOKE_ID = "gc_lite_v2_capacity_v2_january_l0_atomic_smoke_v1"
SMOKE_SCHEMA_VERSION = 1
JANUARY_START = date(2025, 1, 1)
JANUARY_END = date(2025, 1, 31)
DECODE_LATENCY_NS = 1_000_000

CAPACITY_ROW_REQUIRED_COLUMNS = {
    "root_id",
    "action_id",
    "observation_id",
    "date",
    "raw_symbol",
    "continuity_segment_id",
    "direction",
    "arrival_time_ns",
    "state_available",
    "best_bid_size",
    "best_ask_size",
    "bid_top3_depth",
    "ask_top3_depth",
}


class CapacityV2SmokeError(ValueError):
    """Raised when the January capacity smoke cannot prove its inputs."""


@dataclass(frozen=True)
class CapacityV2JanuarySmokeResult:
    report: dict[str, Any]
    input_certificate: dict[str, Any]
    capacity_rows: pd.DataFrame
    atomic_trade_groups: pd.DataFrame
    source_l0_paths: tuple[Path, ...]


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _require_january_date(value: Any, *, name: str) -> str:
    observed = _as_date(value)
    if not JANUARY_START <= observed <= JANUARY_END:
        raise PermissionError(
            f"{name} escaped the January-2025 screening seal: {observed}"
        )
    return str(observed)


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CapacityV2SmokeError(f"{name} is missing columns: {missing}")


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CapacityV2SmokeError(f"{name} must be a JSON object: {path}")
    return payload


def _declared_artifact_path(
    upstream_dir: Path,
    declaration: Mapping[str, Any],
    *,
    expected_name: str,
) -> Path:
    declared = Path(str(declaration.get("path", ""))).expanduser()
    if not declared.is_absolute():
        declared = upstream_dir / declared
    declared = declared.resolve()
    expected = (upstream_dir / expected_name).resolve()
    if declared != expected:
        raise CapacityV2SmokeError(
            f"upstream manifest path for {expected_name} is not the exact artifact: "
            f"{declared} != {expected}"
        )
    if not declared.is_file():
        raise FileNotFoundError(f"missing upstream artifact: {declared}")
    expected_sha256 = str(declaration.get("sha256", ""))
    observed_sha256 = sha256_file(declared)
    if observed_sha256 != expected_sha256:
        raise CapacityV2SmokeError(
            f"upstream artifact hash mismatch for {expected_name}: "
            f"{observed_sha256} != {expected_sha256}"
        )
    return declared


def load_authenticated_upstream(
    upstream_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify and load only the two upstream frames needed by this smoke."""

    source = Path(upstream_dir).expanduser().resolve()
    manifest_path = source / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing upstream run manifest: {manifest_path}")
    manifest = _load_json_object(manifest_path, name="upstream run manifest")
    if int(manifest.get("schema_version", 0)) != 1:
        raise CapacityV2SmokeError("unsupported upstream run-manifest schema")
    if str(manifest.get("preflight_id", "")) != PREFLIGHT_ID:
        raise CapacityV2SmokeError("upstream artifact is not the corrected L2 preflight")
    if str(manifest.get("evidence_class", "")) != SCREENING_LABEL:
        raise CapacityV2SmokeError("upstream artifact is not SCREENING_ONLY")
    if str(manifest.get("split_class", "")) != DEVELOPMENT_LABEL:
        raise CapacityV2SmokeError("upstream artifact is not DEVELOPMENT_ONLY")
    if manifest.get("sealed_2026_read") is not False:
        raise PermissionError("upstream preflight opened sealed 2026 data")
    if manifest.get("sep_dec_2025_read") is not False:
        raise PermissionError("upstream preflight opened Sep-Dec 2025 data")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CapacityV2SmokeError("upstream manifest has no artifacts mapping")
    if sha256_json(dict(artifacts)) != str(manifest.get("artifact_set_sha256", "")):
        raise CapacityV2SmokeError("upstream artifact-set hash mismatch")

    names = {
        "l2_observations": "l2_observations.parquet",
        "payoffs_long": "payoffs_long.parquet",
        "input_certificate": "input_certificate.json",
        "screening_report": "SCREENING_ONLY.json",
    }
    paths: dict[str, Path] = {}
    for key, filename in names.items():
        declaration = artifacts.get(key)
        if not isinstance(declaration, Mapping):
            raise CapacityV2SmokeError(f"upstream manifest is missing {key!r}")
        paths[key] = _declared_artifact_path(
            source, declaration, expected_name=filename
        )

    certificate = _load_json_object(
        paths["input_certificate"], name="upstream input certificate"
    )
    if certificate.get("passed") is not True:
        raise CapacityV2SmokeError("upstream input certificate did not pass")
    if str(certificate.get("certificate_id", "")) != PREFLIGHT_ID:
        raise CapacityV2SmokeError("upstream input certificate has the wrong ID")
    if str(certificate.get("evidence_class", "")) != SCREENING_LABEL:
        raise CapacityV2SmokeError("upstream input certificate is not SCREENING_ONLY")

    report = _load_json_object(paths["screening_report"], name="upstream report")
    if report.get("sealed_2026_read") is not False:
        raise PermissionError("upstream report opened sealed 2026 data")
    if report.get("sep_dec_2025_read") is not False:
        raise PermissionError("upstream report opened Sep-Dec 2025 data")

    observations = pd.read_parquet(paths["l2_observations"])
    payoffs = pd.read_parquet(paths["payoffs_long"])
    return observations, payoffs, certificate, manifest, {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for name, path in {"run_manifest": manifest_path, **paths}.items()
    }


def build_capacity_requests(
    observations: pd.DataFrame, payoffs: pd.DataFrame
) -> pd.DataFrame:
    """Join all 138 x 3 non-flat roots to their exact L2 arrival states."""

    _require_columns(
        observations,
        {
            "observation_id",
            "arrival_time_ns",
            "date",
            "raw_symbol",
            "continuity_segment_id",
            "state_available",
            "state_available_time_ns",
            "state_global_record_ordinal",
            "best_bid_size",
            "best_ask_size",
            "bid_top3_depth",
            "ask_top3_depth",
            "evidence_class",
            "split_class",
        },
        name="L2 observations",
    )
    _require_columns(
        payoffs,
        {
            "root_id",
            "action_id",
            "date",
            "raw_symbol",
            "continuity_segment_id",
            "direction",
            "entry_arrival_time_ns",
            "evidence_class",
            "split_class",
        },
        name="upstream payoffs",
    )
    if set(observations["evidence_class"].astype(str)) != {SCREENING_LABEL}:
        raise CapacityV2SmokeError("L2 observations are not SCREENING_ONLY")
    if set(observations["split_class"].astype(str)) != {DEVELOPMENT_LABEL}:
        raise CapacityV2SmokeError("L2 observations are not DEVELOPMENT_ONLY")
    if set(payoffs["evidence_class"].astype(str)) != {SCREENING_LABEL}:
        raise CapacityV2SmokeError("upstream payoffs are not SCREENING_ONLY")
    if set(payoffs["split_class"].astype(str)) != {DEVELOPMENT_LABEL}:
        raise CapacityV2SmokeError("upstream payoffs are not DEVELOPMENT_ONLY")

    action_rows = payoffs.loc[
        payoffs["action_id"].astype(str).isin(NONFLAT_ACTION_IDS)
    ].copy()
    if action_rows.empty:
        raise CapacityV2SmokeError("upstream payoffs contain no non-flat actions")
    expected_actions = set(NONFLAT_ACTION_IDS)
    observed_actions = set(action_rows["action_id"].astype(str))
    if observed_actions != expected_actions:
        raise CapacityV2SmokeError(
            f"upstream action lattice changed: {sorted(observed_actions)}"
        )
    action_rows["root_id"] = action_rows["root_id"].astype(str)
    action_rows["action_id"] = action_rows["action_id"].astype(str)
    action_rows["observation_id"] = (
        action_rows["root_id"] + "::entry::" + action_rows["action_id"]
    )
    if action_rows[["root_id", "action_id"]].duplicated().any():
        raise CapacityV2SmokeError("upstream payoffs repeat a root/action")
    root_action_counts = action_rows.groupby("root_id")["action_id"].nunique()
    if not bool((root_action_counts == len(NONFLAT_ACTION_IDS)).all()):
        raise CapacityV2SmokeError("some roots lack the complete non-flat action lattice")

    selected_observations = observations.loc[
        observations["observation_id"].astype(str).isin(
            set(action_rows["observation_id"])
        )
    ].copy()
    if selected_observations["observation_id"].astype(str).duplicated().any():
        raise CapacityV2SmokeError("L2 observations repeat an entry observation")
    joined = action_rows.merge(
        selected_observations,
        on="observation_id",
        how="left",
        suffixes=("_payoff", "_l2"),
        validate="one_to_one",
    )
    if len(joined) != len(action_rows):
        raise AssertionError("capacity request join changed row count")
    required_l2 = [
        "arrival_time_ns",
        "raw_symbol_l2",
        "continuity_segment_id_l2",
        "date_l2",
    ]
    if joined[required_l2].isna().any().any():
        raise CapacityV2SmokeError("an entry action has no exact L2 observation")
    if not bool(
        (
            joined["entry_arrival_time_ns"].astype("int64")
            == joined["arrival_time_ns"].astype("int64")
        ).all()
    ):
        raise CapacityV2SmokeError("payoff and L2 arrival clocks disagree")
    for payoff_name, l2_name in (
        ("raw_symbol_payoff", "raw_symbol_l2"),
        ("continuity_segment_id_payoff", "continuity_segment_id_l2"),
    ):
        if not bool(joined[payoff_name].astype(str).eq(joined[l2_name].astype(str)).all()):
            raise CapacityV2SmokeError(
                f"payoff and L2 observations disagree on {payoff_name}"
            )

    output = pd.DataFrame(
        {
            "root_id": joined["root_id"].astype(str),
            "action_id": joined["action_id"].astype(str),
            "observation_id": joined["observation_id"].astype(str),
            "date": joined["date_l2"].astype(str),
            "session_date": joined["date_payoff"].astype(str),
            "raw_symbol": joined["raw_symbol_l2"].astype(str),
            "continuity_segment_id": joined[
                "continuity_segment_id_l2"
            ].astype("int64"),
            "direction": joined["direction"].astype("int64"),
            "arrival_time_ns": joined["arrival_time_ns"].astype("int64"),
            "state_available": joined["state_available"].astype(bool),
            "state_available_time_ns": joined[
                "state_available_time_ns"
            ].astype("Int64"),
            "state_global_record_ordinal": joined[
                "state_global_record_ordinal"
            ].astype("Int64"),
            "best_bid_size": joined["best_bid_size"].astype("Int64"),
            "best_ask_size": joined["best_ask_size"].astype("Int64"),
            "bid_top3_depth": joined["bid_top3_depth"].astype("int64"),
            "ask_top3_depth": joined["ask_top3_depth"].astype("int64"),
        }
    )
    for value in output["date"]:
        _require_january_date(value, name="L2 observation date")
    arrival_dates = pd.to_datetime(
        output["arrival_time_ns"], unit="ns", utc=True
    ).dt.date.astype(str)
    if not bool(arrival_dates.eq(output["date"]).all()):
        raise CapacityV2SmokeError("L2 observation date is not its UTC arrival date")
    if not set(output["direction"].astype(int)).issubset({-1, 1}):
        raise CapacityV2SmokeError("capacity requests contain an invalid direction")
    output["trailing_window_start_ns"] = (
        output["arrival_time_ns"] - TRAILING_WINDOW_NS
    )
    return output.sort_values(
        ["arrival_time_ns", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_capacity_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = load_yaml(source)
    expected = {
        "rule_id": "gc_one_lot_causal_capacity_v2",
        "quantity_contracts": 1,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise CapacityV2SmokeError(f"capacity config changed {key!r}")
    exposure = payload.get("exposure_increasing")
    if not isinstance(exposure, Mapping):
        raise CapacityV2SmokeError("capacity config lacks exposure_increasing")
    trailing = exposure.get("trailing_executed_volume")
    if not isinstance(trailing, Mapping):
        raise CapacityV2SmokeError("capacity config lacks trailing-volume rules")
    checks = {
        "minimum_opposite_best_depth_contracts": 2,
        "minimum_opposite_top3_depth_contracts": 20,
        "maximum_top3_fraction": 0.05,
    }
    for key, value in checks.items():
        if exposure.get(key) != value:
            raise CapacityV2SmokeError(f"capacity config changed {key!r}")
    trailing_checks = {
        "window_seconds": 300,
        "minimum_contracts": 200,
        "maximum_fraction": 0.005,
        "source": "same_symbol_aggressive_trade_records",
    }
    for key, value in trailing_checks.items():
        if trailing.get(key) != value:
            raise CapacityV2SmokeError(f"capacity config changed trailing {key!r}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "rule_id": str(payload["rule_id"]),
        "capacity_contract_sha256": FROZEN_CAPACITY_V2.content_sha256,
    }


def resolve_manifest_declared_l0(
    certificate: Mapping[str, Any],
    requests: pd.DataFrame,
) -> tuple[
    dict[str, tuple[Path, ...]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Resolve exact January L0 parts without discovering a raw directory."""

    requested_dates = set(requests["date"].astype(str))
    if not requested_dates:
        raise CapacityV2SmokeError("capacity request date set is empty")
    for value in requested_dates:
        _require_january_date(value, name="requested raw date")
    bindings = certificate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise CapacityV2SmokeError("upstream certificate has no bindings")
    declarations = bindings.get("selected_daily_health_manifests")
    if not isinstance(declarations, list):
        raise CapacityV2SmokeError("upstream certificate has no daily manifests")
    by_date: dict[str, Mapping[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise CapacityV2SmokeError("invalid daily-manifest declaration")
        declared_date = _require_january_date(
            declaration.get("date"), name="declared manifest date"
        )
        if declared_date in by_date:
            raise CapacityV2SmokeError(f"duplicate daily manifest: {declared_date}")
        by_date[declared_date] = declaration
    if set(by_date) != requested_dates:
        raise CapacityV2SmokeError(
            "daily-manifest dates do not exactly match capacity arrival dates: "
            f"declared={sorted(by_date)}, requested={sorted(requested_dates)}"
        )

    l0_by_date: dict[str, tuple[Path, ...]] = {}
    manifest_bindings: list[dict[str, Any]] = []
    l0_bindings: list[dict[str, Any]] = []
    health_by_date: dict[str, dict[str, Any]] = {}
    for selected_date in sorted(by_date):
        declaration = by_date[selected_date]
        manifest_path = Path(str(declaration.get("path", ""))).expanduser().resolve()
        expected_name = f"{selected_date}.json"
        if manifest_path.name != expected_name or manifest_path.parent.name != "_manifests":
            raise PermissionError(f"daily manifest escaped its exact path: {manifest_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing daily manifest: {manifest_path}")
        manifest_sha = sha256_file(manifest_path)
        if manifest_sha != str(declaration.get("sha256", "")):
            raise CapacityV2SmokeError(
                f"daily manifest hash mismatch for {selected_date}"
            )
        payload = _load_json_object(manifest_path, name="daily health manifest")
        if str(payload.get("date", "")) != selected_date:
            raise CapacityV2SmokeError("daily manifest date mismatch")
        if str(payload.get("status", "")) != "passed":
            raise CapacityV2SmokeError(f"daily manifest did not pass: {selected_date}")
        stats = payload.get("stats")
        if not isinstance(stats, Mapping):
            raise CapacityV2SmokeError("daily manifest lacks health stats")
        if int(stats.get("continuity_segment_id", -1)) != 0:
            raise CapacityV2SmokeError("unexpected January continuity segment")
        if int(stats.get("incomplete_mutating_events", -1)) != 0:
            raise CapacityV2SmokeError("daily manifest reports incomplete mutating events")
        if int(stats.get("incomplete_non_mutating_events", -1)) != 0:
            raise CapacityV2SmokeError(
                "daily manifest reports incomplete non-mutating events"
            )
        if int(stats.get("ts_recv_reversals", -1)) != 0:
            raise CapacityV2SmokeError("daily manifest reports receive-clock reversals")

        archive_root = manifest_path.parent.parent.resolve()
        outputs = payload.get("outputs")
        if not isinstance(outputs, list):
            raise CapacityV2SmokeError("daily manifest has no output list")
        selected_outputs = [
            item
            for item in outputs
            if isinstance(item, Mapping) and str(item.get("layer", "")) == "l0"
        ]
        if not selected_outputs:
            raise CapacityV2SmokeError(f"daily manifest binds no L0: {selected_date}")
        paths: list[Path] = []
        daily_l0: list[dict[str, Any]] = []
        for item in selected_outputs:
            if str(item.get("date", "")) != selected_date:
                raise CapacityV2SmokeError("L0 output date differs from manifest date")
            relative = Path(str(item.get("relative_path", "")))
            expected_parent = Path(
                f"l0/year=2025/month=01/day={selected_date[-2:]}"
            )
            if relative.parent != expected_parent or not relative.name.startswith("part-"):
                raise PermissionError(f"manifest L0 path escaped January: {relative}")
            if relative.suffix != ".parquet" or relative.is_absolute() or ".." in relative.parts:
                raise PermissionError(f"invalid manifest L0 path: {relative}")
            source_path = (archive_root / relative).resolve()
            if source_path.parent != (archive_root / expected_parent).resolve():
                raise PermissionError("resolved L0 path escaped its exact partition")
            if not source_path.is_file():
                raise FileNotFoundError(f"manifest-declared L0 part is missing: {source_path}")
            observed_size = int(source_path.stat().st_size)
            if observed_size != int(item.get("size", -1)):
                raise CapacityV2SmokeError(f"L0 size mismatch: {source_path}")
            observed_sha = sha256_file(source_path)
            if observed_sha != str(item.get("sha256", "")):
                raise CapacityV2SmokeError(f"L0 hash mismatch: {source_path}")
            binding = {
                "date": selected_date,
                "path": str(source_path),
                "relative_path": str(relative),
                "rows": int(item.get("rows", -1)),
                "size_bytes": observed_size,
                "sha256": observed_sha,
            }
            paths.append(source_path)
            daily_l0.append(binding)
            l0_bindings.append(binding)
        if len(paths) != len(set(paths)):
            raise CapacityV2SmokeError("daily manifest repeats an L0 part")
        daily_l0.sort(key=lambda item: item["path"])
        declared_set_hash = sha256_json(daily_l0)
        l0_by_date[selected_date] = tuple(sorted(paths))
        manifest_bindings.append(
            {
                "date": selected_date,
                "path": str(manifest_path),
                "sha256": manifest_sha,
                "l0_parts": len(paths),
                "l0_output_set_sha256": declared_set_hash,
            }
        )
        health_by_date[selected_date] = {
            "continuity_segment_id": int(stats["continuity_segment_id"]),
            "first_ts_recv_ns": int(stats["first_ts_recv_ns"]),
            "last_ts_recv_ns": int(stats["last_ts_recv_ns"]),
            "incomplete_mutating_events": 0,
            "incomplete_non_mutating_events": 0,
            "ts_recv_reversals": 0,
            "status": "passed",
        }
    manifest_bindings.sort(key=lambda item: item["date"])
    l0_bindings.sort(key=lambda item: item["path"])
    return l0_by_date, manifest_bindings, l0_bindings, health_by_date


def _atomic_trade_groups_for_date(
    paths: Iterable[str | Path],
    requests: pd.DataFrame,
    *,
    decode_latency_ns: int = DECODE_LATENCY_NS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build exact per-instrument F_LAST groups and selected trade groups.

    The returned coverage frame has one row per requested symbol/continuity
    identity.  The trade frame contains only complete, causally available
    groups that fall inside at least one requested five-minute window.
    """

    if requests.empty:
        raise CapacityV2SmokeError("per-date request frame cannot be empty")
    _require_columns(
        requests,
        {
            "observation_id",
            "raw_symbol",
            "continuity_segment_id",
            "arrival_time_ns",
            "trailing_window_start_ns",
        },
        name="per-date requests",
    )
    source_paths = [str(Path(path).resolve()) for path in paths]
    if not source_paths:
        raise CapacityV2SmokeError("per-date L0 path set cannot be empty")
    if decode_latency_ns < 0:
        raise CapacityV2SmokeError("decode latency cannot be negative")
    connection = duckdb.connect()
    try:
        connection.register(
            "request_frame",
            requests[
                [
                    "observation_id",
                    "raw_symbol",
                    "continuity_segment_id",
                    "arrival_time_ns",
                    "trailing_window_start_ns",
                ]
            ],
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE atomic_groups AS
            WITH selected_source AS (
                SELECT
                    global_record_ordinal,
                    publisher_id,
                    instrument_id,
                    raw_symbol,
                    continuity_segment_id,
                    ts_event_ns,
                    ts_recv_ns,
                    action,
                    side,
                    size,
                    is_last,
                    is_snapshot,
                    is_bad_ts_recv,
                    coalesce(
                        sum(CASE WHEN is_last THEN 1 ELSE 0 END) OVER (
                            PARTITION BY
                                publisher_id,
                                instrument_id,
                                continuity_segment_id
                            ORDER BY global_record_ordinal
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        ),
                        0
                    ) AS atomic_group_index
                FROM read_parquet(?) AS source
                WHERE EXISTS (
                    SELECT 1
                    FROM request_frame AS request
                    WHERE request.raw_symbol = source.raw_symbol
                      AND request.continuity_segment_id = source.continuity_segment_id
                )
            )
            SELECT
                publisher_id,
                instrument_id,
                raw_symbol,
                continuity_segment_id,
                atomic_group_index,
                min(global_record_ordinal) AS first_global_record_ordinal,
                max(global_record_ordinal) AS last_global_record_ordinal,
                count(*) AS group_records,
                max(ts_event_ns) AS event_time_ns,
                bool_or(is_snapshot) AS is_snapshot_group,
                bool_or(action = 'R') AS has_reset,
                arg_max(is_last, global_record_ordinal) AS closes_with_f_last,
                arg_max(is_bad_ts_recv, global_record_ordinal) AS closing_bad_ts_recv,
                CASE
                    WHEN bool_or(is_snapshot)
                        THEN max(ts_recv_ns) + {int(decode_latency_ns)}
                    WHEN arg_max(is_bad_ts_recv, global_record_ordinal)
                        THEN NULL
                    ELSE arg_max(ts_recv_ns, global_record_ordinal)
                        + {int(decode_latency_ns)}
                END AS available_time_ns,
                count(*) FILTER (WHERE action = 'T') AS aggressive_trade_records,
                coalesce(sum(size) FILTER (WHERE action = 'T'), 0)
                    AS aggressive_volume_contracts,
                count(*) FILTER (WHERE action = 'T' AND side = 'B')
                    AS buy_aggressor_trade_records,
                count(*) FILTER (WHERE action = 'T' AND side = 'A')
                    AS sell_aggressor_trade_records,
                count(*) FILTER (WHERE action = 'T' AND side = 'N')
                    AS neutral_aggressor_trade_records
            FROM selected_source
            GROUP BY
                publisher_id,
                instrument_id,
                raw_symbol,
                continuity_segment_id,
                atomic_group_index
            """,
            [source_paths],
        )
        coverage = connection.execute(
            """
            SELECT
                identity.raw_symbol,
                identity.continuity_segment_id,
                count(DISTINCT atomic.publisher_id) AS publisher_count,
                count(DISTINCT atomic.instrument_id) AS instrument_count,
                min(atomic.publisher_id) AS publisher_id,
                min(atomic.instrument_id) AS instrument_id,
                count(*) AS atomic_groups,
                count(*) FILTER (WHERE atomic.closes_with_f_last)
                    AS complete_atomic_groups,
                count(*) FILTER (WHERE NOT atomic.closes_with_f_last)
                    AS incomplete_atomic_groups,
                count(*) FILTER (
                    WHERE atomic.closes_with_f_last
                      AND atomic.available_time_ns IS NULL
                ) AS unavailable_complete_groups,
                count(*) FILTER (
                    WHERE atomic.aggressive_volume_contracts > 0
                      AND NOT atomic.closes_with_f_last
                ) AS incomplete_trade_groups,
                count(*) FILTER (
                    WHERE atomic.aggressive_volume_contracts > 0
                      AND atomic.closes_with_f_last
                      AND atomic.available_time_ns IS NULL
                ) AS unavailable_trade_groups,
                min(atomic.available_time_ns) FILTER (
                    WHERE atomic.closes_with_f_last
                ) AS first_complete_available_time_ns,
                max(atomic.available_time_ns) FILTER (
                    WHERE atomic.closes_with_f_last
                ) AS last_complete_available_time_ns,
                min(atomic.first_global_record_ordinal) AS first_source_ordinal,
                max(atomic.last_global_record_ordinal) AS last_source_ordinal
            FROM (
                SELECT DISTINCT raw_symbol, continuity_segment_id
                FROM request_frame
            ) AS identity
            LEFT JOIN atomic_groups AS atomic
              ON atomic.raw_symbol = identity.raw_symbol
             AND atomic.continuity_segment_id = identity.continuity_segment_id
            GROUP BY identity.raw_symbol, identity.continuity_segment_id
            ORDER BY identity.raw_symbol, identity.continuity_segment_id
            """
        ).fetchdf()
        trades = connection.execute(
            """
            SELECT atomic.*
            FROM atomic_groups AS atomic
            WHERE atomic.closes_with_f_last
              AND atomic.available_time_ns IS NOT NULL
              AND atomic.aggressive_volume_contracts > 0
              AND EXISTS (
                    SELECT 1
                    FROM request_frame AS request
                    WHERE request.raw_symbol = atomic.raw_symbol
                      AND request.continuity_segment_id = atomic.continuity_segment_id
                      AND atomic.available_time_ns > request.trailing_window_start_ns
                      AND atomic.available_time_ns <= request.arrival_time_ns
              )
            ORDER BY
                atomic.available_time_ns,
                atomic.last_global_record_ordinal
            """
        ).fetchdf()
        resets = connection.execute(
            """
            SELECT
                request.observation_id,
                max(atomic.available_time_ns) AS last_reset_available_time_ns
            FROM request_frame AS request
            LEFT JOIN atomic_groups AS atomic
              ON atomic.raw_symbol = request.raw_symbol
             AND atomic.continuity_segment_id = request.continuity_segment_id
             AND atomic.closes_with_f_last
             AND atomic.available_time_ns IS NOT NULL
             AND atomic.has_reset
             AND atomic.available_time_ns <= request.arrival_time_ns
            GROUP BY request.observation_id
            ORDER BY request.observation_id
            """
        ).fetchdf()
    finally:
        connection.close()
    if len(resets) != len(requests):
        raise AssertionError("reset query did not return one row per request")
    return coverage, trades, resets


def reconstruct_trailing_volume(
    requests: pd.DataFrame,
    l0_by_date: Mapping[str, Iterable[str | Path]],
    health_by_date: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Attach exact available-time trade volume and reset-aware warm-up."""

    pieces: list[pd.DataFrame] = []
    trade_pieces: list[pd.DataFrame] = []
    coverage_ledger: list[dict[str, Any]] = []
    for selected_date, date_requests in requests.groupby("date", sort=True):
        if selected_date not in l0_by_date or selected_date not in health_by_date:
            raise CapacityV2SmokeError(f"missing certified L0 inputs for {selected_date}")
        coverage, trades, resets = _atomic_trade_groups_for_date(
            l0_by_date[selected_date], date_requests
        )
        if bool((coverage["publisher_count"].astype(int) != 1).any()):
            raise CapacityV2SmokeError("requested symbol has ambiguous publisher identity")
        if bool((coverage["instrument_count"].astype(int) != 1).any()):
            raise CapacityV2SmokeError("requested symbol has ambiguous instrument identity")
        if coverage[
            ["first_complete_available_time_ns", "last_complete_available_time_ns"]
        ].isna().any().any():
            raise CapacityV2SmokeError("requested symbol has no complete F_LAST coverage")
        if bool((coverage["incomplete_atomic_groups"].astype(int) != 0).any()):
            raise CapacityV2SmokeError("requested symbol has an unfinished atomic group")
        if bool((coverage["unavailable_complete_groups"].astype(int) != 0).any()):
            raise CapacityV2SmokeError(
                "requested symbol has a complete group without causal availability"
            )
        if bool((coverage["incomplete_trade_groups"].astype(int) != 0).any()):
            raise CapacityV2SmokeError("an aggressive trade is in an unfinished F_LAST group")
        if bool((coverage["unavailable_trade_groups"].astype(int) != 0).any()):
            raise CapacityV2SmokeError("an aggressive trade group has no causal receipt clock")

        selected = date_requests.merge(
            resets, on="observation_id", how="left", validate="one_to_one"
        )
        identity_coverage = coverage[
            [
                "raw_symbol",
                "continuity_segment_id",
                "publisher_id",
                "instrument_id",
                "first_complete_available_time_ns",
                "last_complete_available_time_ns",
                "first_source_ordinal",
                "last_source_ordinal",
            ]
        ].copy()
        selected = selected.merge(
            identity_coverage,
            on=["raw_symbol", "continuity_segment_id"],
            how="left",
            validate="many_to_one",
        )
        if selected[
            [
                "publisher_id",
                "instrument_id",
                "first_complete_available_time_ns",
                "last_complete_available_time_ns",
            ]
        ].isna().any().any():
            raise CapacityV2SmokeError("request is not covered by an exact L0 identity")
        if bool(
            (
                selected["first_complete_available_time_ns"].astype("int64")
                > selected["arrival_time_ns"].astype("int64")
            ).any()
        ):
            raise CapacityV2SmokeError("identity coverage begins after an order arrival")
        if bool(
            (
                selected["last_complete_available_time_ns"].astype("int64")
                < selected["arrival_time_ns"].astype("int64")
            ).any()
        ):
            raise CapacityV2SmokeError("identity coverage ends before an order arrival")
        if trades.empty:
            volumes = pd.DataFrame(
                {
                    "observation_id": selected["observation_id"],
                    "trailing_aggressive_volume_5m": 0,
                    "trailing_trade_groups_5m": 0,
                    "trailing_trade_records_5m": 0,
                    "trailing_neutral_trade_records_5m": 0,
                }
            )
        else:
            joined = selected[
                [
                    "observation_id",
                    "raw_symbol",
                    "continuity_segment_id",
                    "arrival_time_ns",
                    "trailing_window_start_ns",
                ]
            ].merge(
                trades,
                on=["raw_symbol", "continuity_segment_id"],
                how="left",
            )
            in_window = joined.loc[
                joined["available_time_ns"].notna()
                & (
                    joined["available_time_ns"].astype("int64")
                    > joined["trailing_window_start_ns"].astype("int64")
                )
                & (
                    joined["available_time_ns"].astype("int64")
                    <= joined["arrival_time_ns"].astype("int64")
                )
            ].copy()
            volumes = (
                in_window.groupby("observation_id", as_index=False)
                .agg(
                    trailing_aggressive_volume_5m=(
                        "aggressive_volume_contracts",
                        "sum",
                    ),
                    trailing_trade_groups_5m=("atomic_group_index", "size"),
                    trailing_trade_records_5m=(
                        "aggressive_trade_records",
                        "sum",
                    ),
                    trailing_neutral_trade_records_5m=(
                        "neutral_aggressor_trade_records",
                        "sum",
                    ),
                )
            )
            volumes = selected[["observation_id"]].merge(
                volumes, on="observation_id", how="left", validate="one_to_one"
            )
            for column in (
                "trailing_aggressive_volume_5m",
                "trailing_trade_groups_5m",
                "trailing_trade_records_5m",
                "trailing_neutral_trade_records_5m",
            ):
                volumes[column] = volumes[column].fillna(0).astype("int64")

        selected = selected.merge(
            volumes, on="observation_id", how="left", validate="one_to_one"
        )
        source_health = health_by_date[selected_date]
        certified_start = int(source_health["first_ts_recv_ns"]) + DECODE_LATENCY_NS
        selected["certified_source_start_available_time_ns"] = certified_start
        selected["effective_warmup_start_available_time_ns"] = selected[
            "last_reset_available_time_ns"
        ].fillna(certified_start)
        selected["effective_warmup_start_available_time_ns"] = selected[
            [
                "effective_warmup_start_available_time_ns",
                "certified_source_start_available_time_ns",
                "first_complete_available_time_ns",
            ]
        ].max(axis=1).astype("int64")
        selected["trailing_window_complete"] = (
            selected["effective_warmup_start_available_time_ns"].astype("int64")
            <= selected["trailing_window_start_ns"].astype("int64")
        )
        pieces.append(selected)
        if not trades.empty:
            stamped = trades.copy()
            stamped.insert(0, "date", selected_date)
            trade_pieces.append(stamped)
        coverage_ledger.extend(
            {"date": selected_date, **row}
            for row in coverage.to_dict(orient="records")
        )
    output = pd.concat(pieces, ignore_index=True).sort_values(
        ["arrival_time_ns", "observation_id"], kind="mergesort"
    )
    if len(output) != len(requests):
        raise AssertionError("trailing-volume reconstruction changed request count")
    atomic = (
        pd.concat(trade_pieces, ignore_index=True)
        if trade_pieces
        else pd.DataFrame(
            columns=[
                "date",
                "publisher_id",
                "instrument_id",
                "raw_symbol",
                "continuity_segment_id",
                "atomic_group_index",
                "first_global_record_ordinal",
                "last_global_record_ordinal",
                "group_records",
                "event_time_ns",
                "is_snapshot_group",
                "has_reset",
                "closes_with_f_last",
                "closing_bad_ts_recv",
                "available_time_ns",
                "aggressive_trade_records",
                "aggressive_volume_contracts",
                "buy_aggressor_trade_records",
                "sell_aggressor_trade_records",
                "neutral_aggressor_trade_records",
            ]
        )
    )
    return output.reset_index(drop=True), atomic, coverage_ledger


def evaluate_capacity_rows(frame: pd.DataFrame, *, source_set_sha256: str) -> pd.DataFrame:
    """Apply the frozen one-lot capacity contract independently per arrival."""

    _require_columns(
        frame,
        CAPACITY_ROW_REQUIRED_COLUMNS
        | {
            "trailing_aggressive_volume_5m",
            "trailing_window_start_ns",
            "trailing_window_complete",
        },
        name="capacity frame",
    )
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        state: ArrivalCapacityState | None = None
        if bool(row.state_available):
            state = ArrivalCapacityState(
                raw_symbol=str(row.raw_symbol),
                available_time_ns=int(row.arrival_time_ns),
                book_available_time_ns=int(row.state_available_time_ns),
                book_last_ordinal=int(row.state_global_record_ordinal),
                best_bid_depth=int(row.best_bid_size),
                best_ask_depth=int(row.best_ask_size),
                bid_top3_depth=int(row.bid_top3_depth),
                ask_top3_depth=int(row.ask_top3_depth),
                trailing_aggressive_volume_5m=int(
                    row.trailing_aggressive_volume_5m
                ),
                trailing_window_start_ns=int(row.trailing_window_start_ns),
                trailing_window_end_ns=int(row.arrival_time_ns),
                trailing_window_complete=bool(row.trailing_window_complete),
                source_tape_hash=source_set_sha256,
            )
        decision = evaluate_capacity_v2(
            raw_symbol=str(row.raw_symbol),
            arrival_time_ns=int(row.arrival_time_ns),
            position_before=0,
            position_after=int(row.direction),
            state=state,
        )
        rows.append(
            {
                **row._asdict(),
                "book_available_time_ns": (
                    int(row.state_available_time_ns)
                    if bool(row.state_available)
                    else None
                ),
                "book_last_ordinal": (
                    int(row.state_global_record_ordinal)
                    if bool(row.state_available)
                    else None
                ),
                "capacity_state_source_sha256": source_set_sha256,
                **decision.as_dict(),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(frame):
        raise AssertionError("capacity evaluation changed row count")
    if set(result["quantity"].astype(int)) != {1}:
        raise AssertionError("capacity smoke escaped the one-contract estimand")
    return result


def _failure_counts(group: pd.DataFrame) -> dict[str, int]:
    reasons = group["reason"].astype(str)
    return {
        "missing_arrival_capacity_state": int(
            reasons.str.contains("missing_arrival_capacity_state", regex=False).sum()
        ),
        "best_opposite_depth_below_2": int(
            reasons.str.contains("best_opposite_depth_below_2", regex=False).sum()
        ),
        "opposite_top3_depth_below_20": int(
            reasons.str.contains("opposite_top3_depth_below_20", regex=False).sum()
        ),
        "trailing_aggressive_volume_5m_below_200": int(
            reasons.str.contains(
                "trailing_aggressive_volume_5m_below_200", regex=False
            ).sum()
        ),
        "trailing_aggressive_volume_5m_not_warm": int(
            reasons.str.contains(
                "trailing_aggressive_volume_5m_not_warm", regex=False
            ).sum()
        ),
    }


def run_capacity_v2_january_smoke(
    *,
    upstream_dir: str | Path,
    capacity_config_path: str | Path,
) -> CapacityV2JanuarySmokeResult:
    """Run the authenticated January-only capacity mechanism smoke."""

    config_binding = _validate_capacity_config(capacity_config_path)
    observations, payoffs, certificate, upstream_manifest, upstream_bindings = (
        load_authenticated_upstream(upstream_dir)
    )
    requests = build_capacity_requests(observations, payoffs)
    l0_by_date, manifest_bindings, l0_bindings, health_by_date = (
        resolve_manifest_declared_l0(certificate, requests)
    )
    source_identity = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in l0_bindings
    ]
    source_set_sha256 = sha256_json(source_identity)
    capacity_state_source_sha256 = sha256_json(
        {
            "source_l0_partition_set_sha256": source_set_sha256,
            "source_l2_partition_set_sha256": str(
                upstream_manifest["source_l2_partition_set_sha256"]
            ),
            "l2_observations_sha256": str(
                upstream_bindings["l2_observations"]["sha256"]
            ),
        }
    )
    with_volume, atomic_groups, coverage = reconstruct_trailing_volume(
        requests, l0_by_date, health_by_date
    )
    capacity_rows = evaluate_capacity_rows(
        with_volume, source_set_sha256=capacity_state_source_sha256
    )
    root_count = int(capacity_rows["root_id"].nunique())
    if root_count * len(NONFLAT_ACTION_IDS) != len(capacity_rows):
        raise CapacityV2SmokeError("capacity result is not a dense root/action lattice")
    action_reports: dict[str, Any] = {}
    for action_id in NONFLAT_ACTION_IDS:
        group = capacity_rows.loc[
            capacity_rows["action_id"].astype(str) == action_id
        ]
        eligible = int(group["feasible"].astype(bool).sum())
        action_reports[action_id] = {
            "roots": int(len(group)),
            "eligible": eligible,
            "ineligible": int(len(group) - eligible),
            "eligible_rate": float(eligible / len(group)) if len(group) else None,
            "failure_counts_nonexclusive": _failure_counts(group),
            "trailing_volume_min": int(group["trailing_aggressive_volume_5m"].min()),
            "trailing_volume_median": float(
                group["trailing_aggressive_volume_5m"].median()
            ),
            "trailing_volume_max": int(group["trailing_aggressive_volume_5m"].max()),
        }

    coverage_frame = pd.DataFrame(coverage)
    report = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "smoke_id": SMOKE_ID,
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "status": "CAPACITY_V2_MECHANISM_SCREEN_COMPLETE",
        "warning": (
            "January corrected-roll GC-lite roots are a development-only capacity "
            "mechanism smoke. This is not Base-GC, RENT, HMADRL, profitability, "
            "novelty, locked-validation, or final-OOS evidence."
        ),
        "sealed_2026_read": False,
        "sep_dec_2025_read": False,
        "january_only": True,
        "supersedes": {
            "smoke_id": SUPERSEDED_SMOKE_ID,
            "reason": (
                "v1 used partition-wide rather than per-symbol first-complete "
                "availability for five-minute warm-up"
            ),
        },
        "source_date_min": str(min(requests["date"])),
        "source_date_max": str(max(requests["date"])),
        "root_source": "corrected-roll GC-lite v2 L2 aggressive preflight",
        "root_count": root_count,
        "capacity_rows": int(len(capacity_rows)),
        "action_set": list(NONFLAT_ACTION_IDS),
        "action_reports": action_reports,
        "capacity_contract": {
            "rule_id": config_binding["rule_id"],
            "quantity_contracts": 1,
            "minimum_opposite_best_depth_contracts": 2,
            "minimum_opposite_top3_depth_contracts": 20,
            "trailing_window": "(arrival - 5 minutes, arrival]",
            "minimum_trailing_aggressive_volume_contracts": 200,
            "delayed_action_rechecked_at_actual_arrival": True,
            "future_exit_liquidity_used": False,
            "capacity_contract_sha256": FROZEN_CAPACITY_V2.content_sha256,
        },
        "atomic_availability": {
            "source_layer": "l0",
            "event_group_key": (
                "publisher_id,instrument_id,continuity_segment_id,F_LAST sequence"
            ),
            "complete_group_rule": "closing record has F_LAST",
            "available_time_rule": "closing F_LAST ts_recv_ns + 1ms decode latency",
            "snapshot_available_time_rule": "max group ts_recv_ns + 1ms",
            "trade_volume_rule": "sum size once over action=T records per complete group",
            "left_boundary_excluded": True,
            "right_boundary_included": True,
            "manifest_declared_exact_paths_only": True,
            "source_l0_parts": len(l0_bindings),
            "source_l0_set_sha256": source_set_sha256,
            "combined_l2_l0_capacity_state_sha256": capacity_state_source_sha256,
            "selected_atomic_trade_groups": int(len(atomic_groups)),
            "selected_aggressive_volume_contracts": int(
                atomic_groups["aggressive_volume_contracts"].sum()
            )
            if not atomic_groups.empty
            else 0,
            "incomplete_trade_groups": int(
                coverage_frame["incomplete_trade_groups"].astype(int).sum()
            ),
            "unavailable_trade_groups": int(
                coverage_frame["unavailable_trade_groups"].astype(int).sum()
            ),
            "incomplete_atomic_groups": int(
                coverage_frame["incomplete_atomic_groups"].astype(int).sum()
            ),
            "unavailable_complete_groups": int(
                coverage_frame["unavailable_complete_groups"].astype(int).sum()
            ),
            "per_symbol_first_complete_warmup_bound": True,
            "identity_coverage_spans_every_arrival": True,
        },
        "limitations": [
            "Roots are GC-lite screening proxies, not the registered Base-GC paper port.",
            "The smoke tests entry feasibility, not net profit or Sharpe improvement.",
            "No RENT license, residual-worker value, or HMADRL learning claim is tested.",
            "Only one-lot aggressive entry capacity is evaluated; passive FIFO is absent.",
            "The same historical tape is used for a retrospective January screen only.",
            "September-December 2025 and all 2026 outcomes remain unopened here.",
        ],
        "advancement_claim": (
            "The result may inform engineering readiness only. It cannot authorize "
            "RENT-HMADRL training or support a publication claim."
        ),
    }
    input_certificate = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "certificate_type": "CAPACITY_V2_JANUARY_SMOKE_INPUTS",
        "certificate_id": SMOKE_ID,
        "passed": True,
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "sealed_2026_read": False,
        "sep_dec_2025_read": False,
        "checks": {
            "upstream_preflight_id": str(upstream_manifest["preflight_id"]),
            "upstream_authenticated": True,
            "dense_three_action_lattice": True,
            "january_arrival_dates_only": True,
            "exact_manifest_declared_l0_paths": True,
            "l0_sizes_and_hashes_recomputed": True,
            "daily_health_passed": True,
            "per_symbol_first_complete_warmup_bound": True,
            "identity_coverage_spans_every_arrival": True,
            "incomplete_atomic_groups": 0,
            "unavailable_complete_groups": 0,
            "incomplete_trade_groups": 0,
            "unavailable_trade_groups": 0,
            "event_atomic_f_last_availability": True,
            "future_exit_eligibility_used": False,
        },
        "bindings": {
            "capacity_config": config_binding,
            "upstream": upstream_bindings,
            "daily_manifests": manifest_bindings,
            "daily_manifest_set_sha256": sha256_json(manifest_bindings),
            "source_l0_partitions": l0_bindings,
            "source_l0_partition_set_sha256": source_set_sha256,
            "combined_l2_l0_capacity_state_sha256": capacity_state_source_sha256,
        },
        "claim_scope": (
            "This certificate binds a January-only GC-lite capacity-v2 mechanism "
            "screen. It makes no Base-GC, RENT, HMADRL, profit, Sharpe, novelty, "
            "locked-validation, final-OOS, or deployment claim."
        ),
    }
    return CapacityV2JanuarySmokeResult(
        report=report,
        input_certificate=input_certificate,
        capacity_rows=capacity_rows,
        atomic_trade_groups=atomic_groups,
        source_l0_paths=tuple(
            sorted(path for paths in l0_by_date.values() for path in paths)
        ),
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def write_capacity_v2_january_smoke(
    output_dir: str | Path,
    result: CapacityV2JanuarySmokeResult,
) -> dict[str, Any]:
    """Atomically create an immutable evidence directory; never overwrite."""

    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"screening output is create-new-only; choose a new path: {target}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent)
    )
    try:
        paths = {
            "capacity_rows": _atomic_parquet(
                staging / "capacity_rows.parquet", result.capacity_rows
            ),
            "atomic_trade_groups": _atomic_parquet(
                staging / "atomic_trade_groups.parquet",
                result.atomic_trade_groups,
            ),
            "input_certificate": write_json(
                staging / "input_certificate.json", result.input_certificate
            ),
            "screening_report": write_json(
                staging / "SCREENING_ONLY.json", result.report
            ),
        }
        artifact_bindings = {
            name: {
                "path": str(target / path.name),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        }
        source_bindings = [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in result.source_l0_paths
        ]
        manifest = {
            "schema_version": SMOKE_SCHEMA_VERSION,
            "smoke_id": SMOKE_ID,
            "evidence_class": SCREENING_LABEL,
            "split_class": DEVELOPMENT_LABEL,
            "artifacts": artifact_bindings,
            "artifact_set_sha256": sha256_json(artifact_bindings),
            "source_l0_partitions": source_bindings,
            "source_l0_partition_set_sha256": sha256_json(source_bindings),
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
            "claim_scope": result.input_certificate["claim_scope"],
        }
        staging_manifest = write_json(staging / "run_manifest.json", manifest)
        if target.exists():
            raise FileExistsError(
                f"screening output appeared during staging; refusing overwrite: {target}"
            )
        os.rename(staging, target)
        manifest_path = target / staging_manifest.name
        return {
            **manifest,
            "run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
