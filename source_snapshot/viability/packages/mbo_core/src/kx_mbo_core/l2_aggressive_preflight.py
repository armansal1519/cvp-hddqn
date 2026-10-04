from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd

from .active_contract_v2 import (
    ROLL_RULE_ID,
    ROLL_RULE_VERSION,
    verify_active_contract_certificate,
)
from .artifacts import sha256_file, sha256_json, write_json
from .config import enforce_2025_frame, load_yaml, resolve_archive_root


SCREENING_LABEL = "SCREENING_ONLY"
DEVELOPMENT_LABEL = "DEVELOPMENT_ONLY"
PREFLIGHT_ID = "gc_lite_v2_l2_aggressive_branch_preflight_v1"
DEVELOPMENT_END = date(2025, 8, 31)

ONE_MILLISECOND_NS = 1_000_000
FIVE_MILLISECONDS_NS = 5_000_000
TWO_HUNDRED_FIFTY_MILLISECONDS_NS = 250_000_000
ONE_SECOND_NS = 1_000_000_000

CONTRACTS = 1
TICK_VALUE_USD = 10.0
FEE_USD_PER_SIDE = 3.0
AGGRESSIVE_STRESS_TICKS_PER_FILLED_SIDE = 1
PRIMARY_CAPACITY_FRACTION = 0.01
CAPACITY_FRACTIONS = (0.0025, 0.005, 0.01, 0.02, 0.05)

ACTION_SPECS: tuple[tuple[str, int | None], ...] = (
    ("flat", None),
    ("gc_lite_screen_follow_now", FIVE_MILLISECONDS_NS),
    (
        "wait_250ms",
        FIVE_MILLISECONDS_NS + TWO_HUNDRED_FIFTY_MILLISECONDS_NS,
    ),
    ("wait_1s", FIVE_MILLISECONDS_NS + ONE_SECOND_NS),
)
ACTION_IDS = tuple(action_id for action_id, _ in ACTION_SPECS)
NONFLAT_ACTION_IDS = ACTION_IDS[1:]

EPISODE_REQUIRED_COLUMNS = {
    "opportunity_id",
    "date",
    "raw_symbol",
    "continuity_segment_id",
    "direction",
    "entry_decision_available_time_ns",
    "exit_decision_available_time_ns",
    "evidence_class",
    "roll_rule_id",
    "roll_rule_version",
}
ACTIVE_MAP_REQUIRED_COLUMNS = {
    "date",
    "selected_raw_symbol",
    "volume_information_cutoff_session",
    "same_session_volume_used",
    "rollback_allowed",
    "roll_rule_id",
    "roll_rule_version",
}
OBSERVATION_REQUIRED_COLUMNS = {
    "observation_id",
    "arrival_time_ns",
    "raw_symbol",
    "continuity_segment_id",
    "state_available",
    "best_bid_ticks",
    "best_ask_ticks",
    "best_bid_size",
    "best_ask_size",
    "bid_top3_depth",
    "ask_top3_depth",
}


@dataclass(frozen=True)
class L2AggressivePreflightResult:
    report: dict[str, Any]
    input_certificate: dict[str, Any]
    observations: pd.DataFrame
    payoffs: pd.DataFrame
    root_headroom: pd.DataFrame
    capacity_sensitivity: pd.DataFrame
    source_paths: tuple[Path, ...]


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _utc_date_from_ns(value: int) -> date:
    return pd.Timestamp(int(value), unit="ns", tz="UTC").date()


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _development_clock_guard(values: Iterable[int], *, name: str) -> None:
    series = pd.Series(list(values), dtype="int64")
    if series.empty:
        raise ValueError(f"{name} cannot be empty")
    clocks = pd.to_datetime(series, unit="ns", utc=True, errors="raise")
    minimum = clocks.min().date()
    maximum = clocks.max().date()
    if minimum < date(2025, 1, 1) or maximum > DEVELOPMENT_END:
        raise PermissionError(
            f"{name} is development-only and may use only 2025-01-01 through "
            f"{DEVELOPMENT_END}; observed {minimum} through {maximum}"
        )


def validate_screening_episodes(episodes: pd.DataFrame) -> pd.DataFrame:
    """Validate corrected-roll GC-lite v2 flat-to-flat screening episodes."""

    _require_columns(episodes, EPISODE_REQUIRED_COLUMNS, name="episodes")
    if episodes.empty:
        raise ValueError("episodes cannot be empty")
    frame = episodes.copy()
    frame["opportunity_id"] = frame["opportunity_id"].astype(str)
    if frame["opportunity_id"].duplicated().any():
        raise ValueError("episode opportunity_id values must be unique")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.date.astype(str)
    if set(frame["evidence_class"].astype(str)) != {SCREENING_LABEL}:
        raise ValueError("episodes must be explicitly SCREENING_ONLY")
    if set(frame["roll_rule_id"].astype(str)) != {ROLL_RULE_ID}:
        raise ValueError("episodes are not bound to corrected-roll GC-lite v2")
    if set(frame["roll_rule_version"].astype(int)) != {ROLL_RULE_VERSION}:
        raise ValueError("episodes have the wrong corrected-roll version")
    if not set(frame["direction"].astype(int)).issubset({-1, 1}):
        raise ValueError("episode direction must be -1 or +1")
    entry = pd.to_numeric(
        frame["entry_decision_available_time_ns"], errors="raise"
    ).astype("int64")
    exit_ = pd.to_numeric(
        frame["exit_decision_available_time_ns"], errors="raise"
    ).astype("int64")
    if bool((entry >= exit_).any()):
        raise ValueError("every episode must have entry decision before exit decision")
    _development_clock_guard(entry, name="entry decision clocks")
    _development_clock_guard(exit_ + FIVE_MILLISECONDS_NS, name="exit arrival clocks")
    _development_clock_guard(
        entry + FIVE_MILLISECONDS_NS + ONE_SECOND_NS,
        name="latest entry arrival clocks",
    )
    if any(_as_date(value) > DEVELOPMENT_END for value in frame["date"]):
        raise PermissionError("episode labels include Sep-Dec 2025 or sealed outcomes")

    ordered = frame.assign(
        _entry=entry,
        _exit=exit_,
        _earliest_entry_arrival=entry + FIVE_MILLISECONDS_NS,
        _latest_entry_arrival=entry + FIVE_MILLISECONDS_NS + ONE_SECOND_NS,
        _exit_arrival=exit_ + FIVE_MILLISECONDS_NS,
    ).sort_values(
        ["raw_symbol", "continuity_segment_id", "_entry"], kind="mergesort"
    )
    if bool((ordered["_latest_entry_arrival"] >= ordered["_exit_arrival"]).any()):
        raise ValueError("a delayed entry reaches at or after its frozen exit arrival")
    for _, group in ordered.groupby(
        ["raw_symbol", "continuity_segment_id"], sort=False
    ):
        prior_exit_arrival = group["_exit_arrival"].shift()
        overlap = group["_earliest_entry_arrival"] < prior_exit_arrival
        if bool(overlap.fillna(False).any()):
            raise ValueError(
                "episode IOC arrivals overlap; input is not executable flat-to-flat"
            )
    return enforce_2025_frame(frame).reset_index(drop=True)


def validate_active_contract_inputs(
    episodes: pd.DataFrame,
    active_contract_map: pd.DataFrame,
    certificate: Mapping[str, Any],
    *,
    episodes_sha256: str,
    active_map_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless the exact episodes and map have a past-only v2 certificate."""

    _require_columns(
        active_contract_map, ACTIVE_MAP_REQUIRED_COLUMNS, name="active contract map"
    )
    active = active_contract_map.copy()
    active["date"] = pd.to_datetime(active["date"], errors="raise").dt.date.astype(str)
    if active.empty or active["date"].duplicated().any():
        raise ValueError("active contract map needs one unique symbol per date")
    if any(_as_date(value) > DEVELOPMENT_END for value in active["date"]):
        raise PermissionError("active map includes Sep-Dec 2025 or sealed outcomes")
    if bool(active["same_session_volume_used"].astype(bool).any()):
        raise ValueError("active map consumed same-session volume")
    if bool(active["rollback_allowed"].astype(bool).any()):
        raise ValueError("active map permits maturity rollback")
    if set(active["roll_rule_id"].astype(str)) != {ROLL_RULE_ID}:
        raise ValueError("active map does not use the corrected v2 roll rule")
    if set(active["roll_rule_version"].astype(int)) != {ROLL_RULE_VERSION}:
        raise ValueError("active map has the wrong roll-rule version")
    selected_dates = pd.to_datetime(active["date"]).dt.date
    cutoffs = pd.to_datetime(active["volume_information_cutoff_session"]).dt.date
    if bool((cutoffs >= selected_dates).any()):
        raise ValueError("active map uses same-session or future volume")

    if str(certificate.get("certificate_type", "")) != "PAST_ONLY_ACTIVE_CONTRACT_SELECTION":
        raise ValueError("wrong active-contract certificate type")
    if str(certificate.get("certificate_id", "")) != ROLL_RULE_ID:
        raise ValueError("active-contract certificate has the wrong rule ID")
    if certificate.get("passed") is not True:
        raise ValueError("active-contract certificate did not pass")
    if certificate.get("past_only") is not True:
        raise ValueError("active-contract certificate is not explicitly past-only")
    if int(certificate.get("minimum_lag_sessions", 0)) < 1:
        raise ValueError("active-contract certificate lacks a completed-session lag")
    if str(certificate.get("evidence_class", "")) != SCREENING_LABEL:
        raise ValueError("active-contract certificate must be SCREENING_ONLY")
    if certificate.get("sealed_2026_read") is not False:
        raise ValueError("active-contract certificate opened sealed 2026 outcomes")
    if certificate.get("calibration_or_locked_outcomes_read") is not False:
        raise ValueError("active-contract certificate opened non-development outcomes")
    checks = certificate.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("active-contract certificate has no checks mapping")
    if checks.get("unique_date_symbol") is not True:
        raise ValueError("active-contract certificate did not prove unique date/symbol rows")
    if int(checks.get("same_or_future_session_volume_rows", -1)) != 0:
        raise ValueError("active-contract certificate reports causal roll violations")
    if int(checks.get("maturity_rollback_count", -1)) != 0:
        raise ValueError("active-contract certificate reports maturity rollback")
    if checks.get("causal_session_flattening_recomputed") is not True:
        raise ValueError(
            "active-contract certificate did not recompute causal session flattening"
        )
    if checks.get("causal_volume_availability_recomputed") is not True:
        raise ValueError(
            "active-contract certificate did not recompute causal volume availability"
        )
    if checks.get("complete_candidate_coverage_recomputed") is not True:
        raise ValueError(
            "active-contract certificate did not recompute complete candidate coverage"
        )
    if int(checks.get("volume_availability_cutoff_violations", -1)) != 0:
        raise ValueError(
            "active-contract certificate reports volume-availability cutoff violations"
        )
    if int(checks.get("incomplete_source_coverage_rows", -1)) != 0:
        raise ValueError(
            "active-contract certificate reports incomplete source coverage"
        )
    if int(checks.get("unexpected_missing_volume_policy_rows", -1)) != 0:
        raise ValueError(
            "active-contract certificate reports an unexpected missing-volume policy"
        )
    flattening = certificate.get("session_flattening")
    if not isinstance(flattening, Mapping) or str(flattening.get("rule_id", "")) != (
        "fixed_research_cutoff_1225_ct_next_5m_bar_v1"
    ):
        raise ValueError("active-contract certificate has the wrong flattening rule")

    declared_map_hashes = {
        str(certificate.get("sha256", "")),
        str(certificate.get("map_sha256", "")),
        str(certificate.get("artifact_sha256", "")),
    }
    artifacts = certificate.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("active-contract certificate has no artifact bindings")
    for declaration in artifacts.values():
        if isinstance(declaration, Mapping):
            declared_map_hashes.add(str(declaration.get("sha256", "")))
    if active_map_sha256 not in declared_map_hashes:
        raise ValueError("active-contract certificate does not bind the selected map hash")
    episodes_declaration = artifacts.get("episodes")
    if not isinstance(episodes_declaration, Mapping) or str(
        episodes_declaration.get("sha256", "")
    ) != episodes_sha256:
        raise ValueError("active-contract certificate does not bind the episode hash")

    map_assignments = {
        (str(row.date), str(row.selected_raw_symbol))
        for row in active[["date", "selected_raw_symbol"]].itertuples(index=False)
    }
    episode_assignments = set(
        zip(episodes["date"].astype(str), episodes["raw_symbol"].astype(str))
    )
    unexpected = sorted(episode_assignments.difference(map_assignments))
    if unexpected:
        raise ValueError(
            f"episodes do not follow the certified active-contract map: {unexpected[:5]}"
        )
    certificate_assignments = certificate.get("assignments") or certificate.get(
        "date_symbol_assignments"
    )
    if not isinstance(certificate_assignments, list):
        raise ValueError("active-contract certificate has no dated assignments")
    certified_pairs = {
        (str(item.get("date")), str(item.get("raw_symbol")))
        for item in certificate_assignments
        if isinstance(item, Mapping)
    }
    if certified_pairs != map_assignments:
        raise ValueError("certificate assignments do not exactly match the map")
    return {
        "passed": True,
        "past_only": True,
        "minimum_lag_sessions": int(certificate["minimum_lag_sessions"]),
        "causal_session_flattening_recomputed": True,
        "causal_volume_availability_recomputed": True,
        "complete_candidate_coverage_recomputed": True,
        "session_flattening_rule_id": str(flattening["rule_id"]),
        "episode_assignments": len(episode_assignments),
        "active_map_rows": int(len(active)),
        "episodes_sha256": episodes_sha256,
        "active_map_sha256": active_map_sha256,
    }


def build_observation_requests(episodes: pd.DataFrame) -> pd.DataFrame:
    """Freeze three entry IOC clocks and one shared exit IOC clock per root."""

    frame = validate_screening_episodes(episodes)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        common = {
            "root_id": str(row.opportunity_id),
            "session_date": str(row.date),
            "raw_symbol": str(row.raw_symbol),
            "continuity_segment_id": int(row.continuity_segment_id),
            "direction": int(row.direction),
        }
        entry_decision = int(row.entry_decision_available_time_ns)
        exit_decision = int(row.exit_decision_available_time_ns)
        for action_id, delay_ns in ACTION_SPECS[1:]:
            assert delay_ns is not None
            arrival = entry_decision + int(delay_ns)
            if arrival >= exit_decision + FIVE_MILLISECONDS_NS:
                raise ValueError(f"entry for {action_id!r} reaches after the common exit")
            rows.append(
                {
                    **common,
                    "observation_id": f"{row.opportunity_id}::entry::{action_id}",
                    "phase": "entry",
                    "action_id": action_id,
                    "decision_time_ns": entry_decision,
                    "arrival_time_ns": arrival,
                }
            )
        rows.append(
            {
                **common,
                "observation_id": f"{row.opportunity_id}::exit::common",
                "phase": "exit",
                "action_id": "common_exit",
                "decision_time_ns": exit_decision,
                "arrival_time_ns": exit_decision + FIVE_MILLISECONDS_NS,
            }
        )
    requests = pd.DataFrame(rows)
    requests["date"] = requests["arrival_time_ns"].map(
        lambda value: str(_utc_date_from_ns(int(value)))
    )
    _development_clock_guard(requests["arrival_time_ns"], name="L2 observation clocks")
    return enforce_2025_frame(requests).sort_values(
        ["arrival_time_ns", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)


def _partition_paths(
    archive_root: str | Path,
    *,
    utc_date: date,
) -> list[Path]:
    if utc_date.year != 2025 or utc_date > DEVELOPMENT_END:
        raise PermissionError(
            "L2 aggressive preflight may discover only Jan-Aug 2025 partitions"
        )
    folder = (
        Path(archive_root)
        / "l2_depth"
        / "year=2025"
        / f"month={utc_date.month:02d}"
        / f"day={utc_date.day:02d}"
    )
    if not folder.is_dir():
        raise FileNotFoundError(f"missing L2 partition: {folder}")
    paths = sorted(path.resolve() for path in folder.glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"L2 partition has no Parquet parts: {folder}")
    if any("year=2025" not in path.parts or "l2_depth" not in path.parts for path in paths):
        raise PermissionError("L2 path seal failed")
    return paths


def validate_health_and_continuity(
    health: Mapping[str, Any], requests: pd.DataFrame
) -> dict[str, Any]:
    daily = health.get("daily")
    if not isinstance(daily, list):
        raise ValueError("archive health report has no daily ledger")
    by_date: dict[str, Mapping[str, Any]] = {}
    for row in daily:
        if isinstance(row, Mapping):
            row_date = str(row.get("date", ""))
            if row_date in by_date:
                raise ValueError(f"health report repeats date {row_date}")
            by_date[row_date] = row
    unhealthy = set(
        str(value)
        for value in (health.get("coverage") or {}).get(
            "unhealthy_condition_days", {}
        )
    )
    checked: list[dict[str, Any]] = []
    for (utc_date, continuity), group in requests.groupby(
        ["date", "continuity_segment_id"], sort=True
    ):
        if _as_date(utc_date) > DEVELOPMENT_END:
            raise PermissionError("health guard reached Sep-Dec 2025 or sealed data")
        row = by_date.get(str(utc_date))
        if row is None:
            raise ValueError(f"health report has no row for {utc_date}")
        if str(row.get("status", "")).lower() != "passed":
            raise ValueError(f"L2 source date {utc_date} did not pass archive health")
        if str(utc_date) in unhealthy:
            raise ValueError(f"L2 source date {utc_date} is in the unhealthy-date ledger")
        if int(row.get("continuity_segment_id", -1)) != int(continuity):
            raise ValueError(
                f"continuity mismatch on {utc_date}: episodes={continuity}, "
                f"health={row.get('continuity_segment_id')}"
            )
        if int(row.get("incomplete_mutating_events", -1)) != 0:
            raise ValueError(f"incomplete mutating events on {utc_date}")
        if int(row.get("ts_recv_reversals", -1)) != 0:
            raise ValueError(f"receive-clock reversal on {utc_date}")
        if int(row.get("l2_rows", 0)) <= 0:
            raise ValueError(f"no certified L2 rows on {utc_date}")
        checked.append(
            {
                "utc_date": str(utc_date),
                "continuity_segment_id": int(continuity),
                "requests": int(len(group)),
                "status": "passed",
            }
        )
    return {
        "passed": True,
        "certificate_source": str(health.get("verdict", "")),
        "checked_partitions": checked,
        "unhealthy_selected_partitions": 0,
        "continuity_crossings": 0,
    }


def load_selected_daily_health_manifests(
    archive_root: str | Path,
    requests: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, tuple[Path, ...]],
]:
    """Read only the daily health manifests needed by the development requests.

    The archive-wide health report also describes later sealed dates. This
    loader deliberately never opens it: each selected Jan-Aug 2025 UTC date is
    certified from its own ``_manifests/YYYY-MM-DD.json`` artifact.
    """

    selected_dates = sorted({_as_date(value) for value in requests["date"]})
    if not selected_dates:
        raise ValueError("cannot certify health for an empty request set")
    if selected_dates[0] < date(2025, 1, 1) or selected_dates[-1] > DEVELOPMENT_END:
        raise PermissionError("daily health discovery is restricted to Jan-Aug 2025")
    root = Path(archive_root)
    daily: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    certified_paths_by_date: dict[str, tuple[Path, ...]] = {}
    for selected_date in selected_dates:
        path = root / "_manifests" / f"{selected_date}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing selected daily manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("date", "")) != str(selected_date):
            raise ValueError(f"daily manifest date mismatch: {path}")
        if str(payload.get("status", "")).lower() != "passed":
            raise ValueError(f"daily manifest did not pass: {path}")
        stats = payload.get("stats")
        if not isinstance(stats, Mapping):
            raise ValueError(f"daily manifest has no health stats: {path}")
        stats_row = dict(stats)
        stats_row["date"] = str(selected_date)
        stats_row["continuity_segment_id"] = int(
            payload.get("continuity_segment_id", stats.get("continuity_segment_id", -1))
        )
        stats_row["status"] = str(payload["status"])
        daily.append(stats_row)
        l2_outputs = [
            item
            for item in payload.get("outputs", ())
            if isinstance(item, Mapping) and str(item.get("layer")) == "l2_depth"
        ]
        if not l2_outputs:
            raise ValueError(f"daily manifest binds no L2 outputs: {path}")
        declared_l2: list[dict[str, Any]] = []
        for item in l2_outputs:
            relative = Path(str(item.get("relative_path", "")))
            if "year=2025" not in relative.parts:
                raise PermissionError("daily manifest references a non-2025 L2 output")
            output_path = (root / relative).resolve()
            if not output_path.is_file():
                raise FileNotFoundError(f"daily manifest L2 output is missing: {output_path}")
            if int(item.get("size", -1)) != int(output_path.stat().st_size):
                raise ValueError(f"daily manifest L2 size mismatch: {output_path}")
            expected_sha256 = str(item.get("sha256", ""))
            observed_sha256 = sha256_file(output_path)
            if not expected_sha256 or observed_sha256 != expected_sha256:
                raise ValueError(f"daily manifest L2 hash mismatch: {output_path}")
            declared_l2.append(
                {
                    "path": str(output_path),
                    "size_bytes": int(output_path.stat().st_size),
                    "sha256": observed_sha256,
                }
            )
        bindings.append(
            {
                "date": str(selected_date),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "l2_output_parts": int(len(l2_outputs)),
                "l2_output_set_sha256": sha256_json(declared_l2),
            }
        )
        certified_paths_by_date[str(selected_date)] = tuple(
            Path(item["path"]) for item in declared_l2
        )
    health = {
        "verdict": "SELECTED_DAILY_MANIFESTS_ONLY",
        "daily": daily,
        "coverage": {"unhealthy_condition_days": {}},
    }
    return health, bindings, certified_paths_by_date


def _query_l2_partition(
    paths: list[Path],
    requests: pd.DataFrame,
    *,
    raw_symbol: str,
    continuity_segment_id: int,
) -> pd.DataFrame:
    """Resolve price-level states with interval joins over atomic L2 events."""

    selected_requests = requests[
        ["observation_id", "arrival_time_ns"]
    ].sort_values("arrival_time_ns", kind="mergesort")
    maximum_arrival = int(selected_requests["arrival_time_ns"].max())
    sources = [str(path) for path in paths]
    query = f"""
        WITH raw_source AS (
            SELECT
                event_index,
                event_row_index,
                global_record_ordinal_end,
                ts_event_ns,
                ts_recv_ns,
                is_snapshot,
                operation,
                side,
                price_ticks,
                level_size
            FROM read_parquet(?)
            WHERE raw_symbol = ?
              AND continuity_segment_id = ?
        ),
        event_completion AS (
            SELECT
                event_index,
                global_record_ordinal_end,
                max(ts_recv_ns) AS complete_ts_recv_ns,
                max(ts_recv_ns) + {ONE_MILLISECOND_NS} AS complete_available_time_ns
            FROM raw_source
            GROUP BY event_index, global_record_ordinal_end
        ),
        selected AS (
            SELECT
                raw_source.*,
                event_completion.complete_ts_recv_ns,
                event_completion.complete_available_time_ns AS available_time_ns
            FROM raw_source
            JOIN event_completion
              USING (event_index, global_record_ordinal_end)
            WHERE event_completion.complete_available_time_ns <= ?
        ),
        eventized AS (
            SELECT
                *,
                sum(CASE WHEN operation = 'clear' THEN 1 ELSE 0 END) OVER (
                    ORDER BY global_record_ordinal_end, event_row_index
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS book_epoch
            FROM selected
        ),
        epoch_starts AS (
            SELECT book_epoch, min(available_time_ns) AS epoch_start_ns
            FROM eventized
            WHERE book_epoch > 0
            GROUP BY book_epoch
        ),
        epoch_bounds AS (
            SELECT
                book_epoch,
                lead(epoch_start_ns) OVER (ORDER BY book_epoch) AS epoch_end_ns
            FROM epoch_starts
        ),
        raw_level_ops AS (
            SELECT
                event_index,
                event_row_index,
                global_record_ordinal_end,
                available_time_ns,
                book_epoch,
                operation,
                side,
                price_ticks,
                level_size,
                lead(available_time_ns) OVER (
                    PARTITION BY book_epoch, side, price_ticks
                    ORDER BY global_record_ordinal_end, event_row_index
                ) AS next_level_change_ns
            FROM eventized
            WHERE book_epoch > 0
              AND side IN ('B', 'A')
              AND operation IN ('upsert', 'delete')
              AND price_ticks IS NOT NULL
        ),
        intervals AS (
            SELECT
                level_ops.*,
                CASE
                    WHEN next_level_change_ns IS NULL THEN epoch_end_ns
                    WHEN epoch_end_ns IS NULL THEN next_level_change_ns
                    ELSE least(next_level_change_ns, epoch_end_ns)
                END AS valid_to_ns
            FROM raw_level_ops AS level_ops
            JOIN epoch_bounds USING (book_epoch)
            WHERE operation = 'upsert' AND level_size > 0
        ),
        active_levels AS (
            SELECT
                request.observation_id,
                request.arrival_time_ns,
                intervals.side,
                intervals.price_ticks,
                intervals.level_size,
                row_number() OVER (
                    PARTITION BY request.observation_id, intervals.side
                    ORDER BY CASE
                        WHEN intervals.side = 'B' THEN -intervals.price_ticks
                        ELSE intervals.price_ticks
                    END
                ) AS depth_rank
            FROM request_frame AS request
            JOIN intervals
              ON intervals.available_time_ns <= request.arrival_time_ns
             AND (
                    intervals.valid_to_ns IS NULL
                    OR request.arrival_time_ns < intervals.valid_to_ns
                 )
        ),
        book_state AS (
            SELECT
                observation_id,
                max(CASE WHEN side = 'B' AND depth_rank = 1 THEN price_ticks END)
                    AS best_bid_ticks,
                max(CASE WHEN side = 'A' AND depth_rank = 1 THEN price_ticks END)
                    AS best_ask_ticks,
                max(CASE WHEN side = 'B' AND depth_rank = 1 THEN level_size END)
                    AS best_bid_size,
                max(CASE WHEN side = 'A' AND depth_rank = 1 THEN level_size END)
                    AS best_ask_size,
                sum(CASE WHEN side = 'B' AND depth_rank <= 3 THEN level_size ELSE 0 END)
                    AS bid_top3_depth,
                sum(CASE WHEN side = 'A' AND depth_rank <= 3 THEN level_size ELSE 0 END)
                    AS ask_top3_depth,
                count(CASE WHEN side = 'B' AND depth_rank <= 3 THEN 1 END)
                    AS bid_levels_observed,
                count(CASE WHEN side = 'A' AND depth_rank <= 3 THEN 1 END)
                    AS ask_levels_observed
            FROM active_levels
            WHERE depth_rank <= 3
            GROUP BY observation_id
        ),
        event_headers_raw AS (
            SELECT
                event_index,
                global_record_ordinal_end,
                max(ts_event_ns) AS ts_event_ns,
                max(complete_ts_recv_ns) AS ts_recv_ns,
                max(available_time_ns) AS available_time_ns,
                max(book_epoch) AS book_epoch,
                bool_or(is_snapshot) AS is_snapshot
            FROM eventized
            GROUP BY event_index, global_record_ordinal_end
        ),
        event_headers AS (
            SELECT * EXCLUDE (same_clock_rank)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY available_time_ns
                        ORDER BY global_record_ordinal_end DESC
                    ) AS same_clock_rank
                FROM event_headers_raw
            )
            WHERE same_clock_rank = 1
        ),
        anchors AS (
            SELECT
                request.observation_id,
                header.event_index AS state_event_index,
                header.global_record_ordinal_end AS state_global_record_ordinal,
                header.ts_event_ns AS state_ts_event_ns,
                header.ts_recv_ns AS state_ts_recv_ns,
                header.available_time_ns AS state_available_time_ns,
                header.book_epoch,
                header.is_snapshot AS state_event_is_snapshot
            FROM request_frame AS request
            ASOF LEFT JOIN event_headers AS header
              ON request.arrival_time_ns >= header.available_time_ns
        )
        SELECT
            request.observation_id,
            request.arrival_time_ns,
            anchors.state_event_index,
            anchors.state_global_record_ordinal,
            anchors.state_ts_event_ns,
            anchors.state_ts_recv_ns,
            anchors.state_available_time_ns,
            anchors.book_epoch,
            anchors.state_event_is_snapshot,
            book_state.best_bid_ticks,
            book_state.best_ask_ticks,
            book_state.best_bid_size,
            book_state.best_ask_size,
            coalesce(book_state.bid_top3_depth, 0) AS bid_top3_depth,
            coalesce(book_state.ask_top3_depth, 0) AS ask_top3_depth,
            coalesce(book_state.bid_levels_observed, 0) AS bid_levels_observed,
            coalesce(book_state.ask_levels_observed, 0) AS ask_levels_observed
        FROM request_frame AS request
        LEFT JOIN anchors USING (observation_id)
        LEFT JOIN book_state USING (observation_id)
        ORDER BY request.arrival_time_ns, request.observation_id
    """
    connection = duckdb.connect()
    try:
        connection.register("request_frame", selected_requests)
        result = connection.execute(
            query,
            [sources, raw_symbol, int(continuity_segment_id), maximum_arrival],
        ).fetchdf()
    finally:
        connection.close()
    if len(result) != len(selected_requests):
        raise AssertionError("L2 query did not return exactly one state per request")
    return result


def extract_l2_observations(
    archive_root: str | Path,
    requests: pd.DataFrame,
    *,
    certified_paths_by_date: Mapping[str, Iterable[str | Path]] | None = None,
) -> tuple[pd.DataFrame, tuple[Path, ...]]:
    _require_columns(
        requests,
        {
            "observation_id",
            "date",
            "raw_symbol",
            "continuity_segment_id",
            "arrival_time_ns",
        },
        name="requests",
    )
    if requests["observation_id"].astype(str).duplicated().any():
        raise ValueError("observation IDs must be unique")
    _development_clock_guard(requests["arrival_time_ns"], name="L2 request clocks")
    pieces: list[pd.DataFrame] = []
    used_paths: set[Path] = set()
    requested_dates = set(requests["date"].astype(str))
    if certified_paths_by_date is not None:
        certified_dates = {str(value) for value in certified_paths_by_date}
        if certified_dates != requested_dates:
            raise ValueError(
                "certified L2 date set must exactly match the observation dates"
            )
    group_keys = ["date", "raw_symbol", "continuity_segment_id"]
    for (date_value, raw_symbol, continuity), group in requests.groupby(
        group_keys, sort=True
    ):
        utc_date = _as_date(date_value)
        if certified_paths_by_date is None:
            paths = _partition_paths(archive_root, utc_date=utc_date)
        else:
            expected_folder = (
                Path(archive_root).resolve()
                / "l2_depth"
                / "year=2025"
                / f"month={utc_date.month:02d}"
                / f"day={utc_date.day:02d}"
            )
            paths = sorted(
                Path(value).expanduser().resolve()
                for value in certified_paths_by_date[str(utc_date)]
            )
            if not paths:
                raise ValueError(f"certified L2 path set is empty for {utc_date}")
            if len(paths) != len(set(paths)):
                raise ValueError(f"certified L2 path set has duplicates for {utc_date}")
            for path in paths:
                if path.parent != expected_folder or not path.name.startswith("part-"):
                    raise PermissionError(
                        f"certified L2 path escaped selected 2025 partition: {path}"
                    )
                if path.suffix != ".parquet" or not path.is_file():
                    raise FileNotFoundError(f"certified L2 part is missing: {path}")
        used_paths.update(paths)
        observed = _query_l2_partition(
            paths,
            group,
            raw_symbol=str(raw_symbol),
            continuity_segment_id=int(continuity),
        )
        observed["date"] = str(utc_date)
        observed["raw_symbol"] = str(raw_symbol)
        observed["continuity_segment_id"] = int(continuity)
        pieces.append(observed)
    observations = pd.concat(pieces, ignore_index=True)
    observations["state_available"] = (
        observations["state_available_time_ns"].notna()
        & observations["book_epoch"].fillna(0).astype(int).gt(0)
        & observations["best_bid_ticks"].notna()
        & observations["best_ask_ticks"].notna()
    )
    observations["book_uncrossed"] = (
        observations["best_bid_ticks"].isna()
        | observations["best_ask_ticks"].isna()
        | observations["best_bid_ticks"].lt(observations["best_ask_ticks"])
    )
    if bool((observations["state_available_time_ns"] > observations["arrival_time_ns"]).fillna(False).any()):
        raise AssertionError("an L2 observation used a future complete event")
    available = observations["state_available_time_ns"].dropna().astype("int64")
    received = observations.loc[available.index, "state_ts_recv_ns"].astype("int64")
    if bool((available - received).ne(ONE_MILLISECOND_NS).any()):
        raise AssertionError("L2 state availability is not exactly ts_recv + 1ms")
    if bool((observations["state_available"] & ~observations["book_uncrossed"]).any()):
        raise ValueError("a selected complete-event L2 state is locked or crossed")
    observations["evidence_class"] = SCREENING_LABEL
    observations["split_class"] = DEVELOPMENT_LABEL
    observations = enforce_2025_frame(observations)
    return (
        observations.sort_values(
            ["arrival_time_ns", "observation_id"], kind="mergesort"
        ).reset_index(drop=True),
        tuple(sorted(used_paths)),
    )


def _side_values(observation: pd.Series, side: str) -> tuple[float, int, int]:
    if side == "buy":
        return (
            float(observation["best_ask_ticks"]),
            int(observation["best_ask_size"]),
            int(observation["ask_top3_depth"]),
        )
    if side == "sell":
        return (
            float(observation["best_bid_ticks"]),
            int(observation["best_bid_size"]),
            int(observation["bid_top3_depth"]),
        )
    raise ValueError(f"unsupported aggressive side: {side!r}")


def build_long_payoffs(
    episodes: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Create four retrospective screening payoffs per root, including flat."""

    episodes = validate_screening_episodes(episodes)
    _require_columns(observations, OBSERVATION_REQUIRED_COLUMNS, name="observations")
    if observations["observation_id"].astype(str).duplicated().any():
        raise ValueError("observations must be unique by observation_id")
    state = observations.set_index("observation_id", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        root_id = str(episode.opportunity_id)
        direction = int(episode.direction)
        base = {
            "root_id": root_id,
            "date": str(episode.date),
            "raw_symbol": str(episode.raw_symbol),
            "continuity_segment_id": int(episode.continuity_segment_id),
            "direction": direction,
            "contracts": CONTRACTS,
            "order_type": "IOC",
            "fee_usd_per_side": FEE_USD_PER_SIDE,
            "tick_value_usd": TICK_VALUE_USD,
            "evidence_class": SCREENING_LABEL,
            "split_class": DEVELOPMENT_LABEL,
            "retrospective_screening_only": True,
        }
        rows.append(
            {
                **base,
                "action_id": "flat",
                "order_type": "NONE",
                "entry_delay_ns": pd.NA,
                "entry_arrival_time_ns": int(episode.entry_decision_available_time_ns),
                "exit_arrival_time_ns": int(episode.exit_decision_available_time_ns),
                "entry_side": "none",
                "exit_side": "none",
                "entry_state_available": True,
                "exit_state_available": True,
                "entry_state_available_time_ns": pd.NA,
                "exit_state_available_time_ns": pd.NA,
                "entry_fill_ticks": np.nan,
                "exit_fill_ticks": np.nan,
                "entry_best_level_size": 0,
                "exit_best_level_size": 0,
                "entry_top3_depth": 0,
                "exit_top3_depth": 0,
                "entry_ioc_filled": False,
                "exit_ioc_filled": False,
                "outcome_complete": True,
                "gross_ticks": 0.0,
                "gross_pnl_usd": 0.0,
                "fees_usd": 0.0,
                "nominal_screening_payoff_usd": 0.0,
                "stress_adverse_ticks_per_filled_side": (
                    AGGRESSIVE_STRESS_TICKS_PER_FILLED_SIDE
                ),
                "stress_screening_payoff_usd": 0.0,
                "screening_payoff_usd": 0.0,
                "capacity_lots_1pct": math.inf,
                "capacity_eligible_1pct": True,
                "capacity_fallback_flat_nominal_payoff_usd": 0.0,
                "capacity_fallback_flat_stress_payoff_usd": 0.0,
                "capacity_fallback_flat_payoff_usd": 0.0,
                "incomplete_reason": None,
            }
        )
        exit_id = f"{root_id}::exit::common"
        if exit_id not in state.index:
            raise ValueError(f"missing common exit observation for {root_id}")
        exit_observation = state.loc[exit_id]
        for action_id, delay_ns in ACTION_SPECS[1:]:
            assert delay_ns is not None
            entry_id = f"{root_id}::entry::{action_id}"
            if entry_id not in state.index:
                raise ValueError(f"missing entry observation {entry_id}")
            entry_observation = state.loc[entry_id]
            entry_side = "buy" if direction > 0 else "sell"
            exit_side = "sell" if direction > 0 else "buy"
            entry_available = bool(entry_observation["state_available"])
            exit_available = bool(exit_observation["state_available"])
            entry_price = np.nan
            exit_price = np.nan
            entry_best_size = 0
            exit_best_size = 0
            entry_depth = 0
            exit_depth = 0
            if entry_available:
                entry_price, entry_best_size, entry_depth = _side_values(
                    entry_observation, entry_side
                )
            if exit_available:
                exit_price, exit_best_size, exit_depth = _side_values(
                    exit_observation, exit_side
                )
            entry_filled = entry_available and entry_best_size >= CONTRACTS
            exit_filled = entry_filled and exit_available and exit_best_size >= CONTRACTS
            incomplete_reason: str | None = None
            if not entry_available:
                incomplete_reason = "entry_l2_state_unavailable"
            elif entry_filled and not exit_available:
                incomplete_reason = "exit_l2_state_unavailable"
            elif entry_filled and not exit_filled:
                incomplete_reason = "common_exit_ioc_unfilled"
            outcome_complete = incomplete_reason is None
            if not entry_filled and entry_available:
                gross_ticks = 0.0
                gross_usd = 0.0
                fees_usd = 0.0
                nominal_payoff = 0.0
                stress_payoff = 0.0
            elif outcome_complete:
                gross_ticks = direction * (float(exit_price) - float(entry_price))
                gross_usd = gross_ticks * TICK_VALUE_USD
                fees_usd = 2.0 * FEE_USD_PER_SIDE
                nominal_payoff = gross_usd - fees_usd
                stress_payoff = nominal_payoff - (
                    2.0
                    * AGGRESSIVE_STRESS_TICKS_PER_FILLED_SIDE
                    * TICK_VALUE_USD
                )
            else:
                gross_ticks = np.nan
                gross_usd = np.nan
                fees_usd = np.nan
                nominal_payoff = np.nan
                stress_payoff = np.nan
            capacity_lots = min(
                math.floor(PRIMARY_CAPACITY_FRACTION * entry_depth),
                math.floor(PRIMARY_CAPACITY_FRACTION * exit_depth),
            )
            capacity_eligible = (
                outcome_complete
                and entry_filled
                and exit_filled
                and capacity_lots >= CONTRACTS
            )
            capacity_nominal_payoff = (
                float(nominal_payoff)
                if capacity_eligible and pd.notna(nominal_payoff)
                else 0.0
            )
            capacity_stress_payoff = (
                float(stress_payoff)
                if capacity_eligible and pd.notna(stress_payoff)
                else 0.0
            )
            rows.append(
                {
                    **base,
                    "action_id": action_id,
                    "entry_delay_ns": int(delay_ns),
                    "entry_arrival_time_ns": int(entry_observation["arrival_time_ns"]),
                    "exit_arrival_time_ns": int(exit_observation["arrival_time_ns"]),
                    "entry_side": entry_side,
                    "exit_side": exit_side,
                    "entry_state_available": entry_available,
                    "exit_state_available": exit_available,
                    "entry_state_available_time_ns": entry_observation[
                        "state_available_time_ns"
                    ],
                    "exit_state_available_time_ns": exit_observation[
                        "state_available_time_ns"
                    ],
                    "entry_fill_ticks": entry_price,
                    "exit_fill_ticks": exit_price,
                    "entry_best_level_size": entry_best_size,
                    "exit_best_level_size": exit_best_size,
                    "entry_top3_depth": entry_depth,
                    "exit_top3_depth": exit_depth,
                    "entry_ioc_filled": entry_filled,
                    "exit_ioc_filled": exit_filled,
                    "outcome_complete": outcome_complete,
                    "gross_ticks": gross_ticks,
                    "gross_pnl_usd": gross_usd,
                    "fees_usd": fees_usd,
                    "nominal_screening_payoff_usd": nominal_payoff,
                    "stress_adverse_ticks_per_filled_side": (
                        AGGRESSIVE_STRESS_TICKS_PER_FILLED_SIDE
                    ),
                    "stress_screening_payoff_usd": stress_payoff,
                    # The conservative stress is deliberately the primary
                    # economic screening column. Nominal remains alongside it.
                    "screening_payoff_usd": stress_payoff,
                    "capacity_lots_1pct": int(capacity_lots),
                    "capacity_eligible_1pct": capacity_eligible,
                    "capacity_fallback_flat_nominal_payoff_usd": (
                        capacity_nominal_payoff
                    ),
                    "capacity_fallback_flat_stress_payoff_usd": (
                        capacity_stress_payoff
                    ),
                    "capacity_fallback_flat_payoff_usd": capacity_stress_payoff,
                    "incomplete_reason": incomplete_reason,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != len(episodes) * len(ACTION_IDS):
        raise AssertionError("payoff product is not dense over the frozen action set")
    observed_actions = tuple(
        action for action in ACTION_IDS if action in set(result["action_id"])
    )
    if observed_actions != ACTION_IDS:
        raise AssertionError("payoff action set changed")
    return enforce_2025_frame(result).reset_index(drop=True)


def build_root_headroom(payoffs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    action_rank = {action: index for index, action in enumerate(ACTION_IDS)}
    for root_id, group in payoffs.groupby("root_id", sort=False):
        complete = group.loc[group["outcome_complete"].astype(bool)].copy()
        if "flat" not in set(complete["action_id"]):
            raise ValueError(f"root {root_id} has no complete flat action")
        complete_actions = set(complete["action_id"].astype(str))
        if len(complete) != len(ACTION_IDS) or complete_actions != set(ACTION_IDS):
            missing = sorted(set(ACTION_IDS).difference(complete_actions))
            raise ValueError(
                "economic headroom requires every registered action to have a "
                f"complete flat-to-flat outcome; root {root_id} is missing {missing}"
            )
        complete["_rank"] = complete["action_id"].map(action_rank)
        raw = complete.sort_values(
            ["screening_payoff_usd", "_rank"],
            ascending=[False, True],
            kind="mergesort",
        )
        raw_best = raw.iloc[0]
        nominal = complete.sort_values(
            ["nominal_screening_payoff_usd", "_rank"],
            ascending=[False, True],
            kind="mergesort",
        )
        nominal_best = nominal.iloc[0]
        constrained = complete.copy()
        constrained["_constrained_payoff"] = np.where(
            constrained["capacity_eligible_1pct"].astype(bool),
            constrained["screening_payoff_usd"].astype(float),
            0.0,
        )
        constrained = constrained.sort_values(
            ["_constrained_payoff", "_rank"],
            ascending=[False, True],
            kind="mergesort",
        )
        constrained_best = constrained.iloc[0]
        follow = complete.loc[
            complete["action_id"] == "gc_lite_screen_follow_now"
        ]
        follow_payoff = (
            float(follow.iloc[0]["screening_payoff_usd"])
            if not follow.empty
            else np.nan
        )
        follow_capacity_payoff = (
            float(follow.iloc[0]["capacity_fallback_flat_payoff_usd"])
            if not follow.empty
            else np.nan
        )
        follow_nominal_payoff = (
            float(follow.iloc[0]["nominal_screening_payoff_usd"])
            if not follow.empty
            else np.nan
        )
        first = group.iloc[0]
        rows.append(
            {
                "root_id": str(root_id),
                "date": str(first["date"]),
                "raw_symbol": str(first["raw_symbol"]),
                "continuity_segment_id": int(first["continuity_segment_id"]),
                "direction": int(first["direction"]),
                "complete_actions": int(len(complete)),
                "primary_payoff_scenario": "adverse_1tick_entry_and_exit",
                "raw_best_action_id": str(raw_best["action_id"]),
                "raw_best_payoff_usd": float(raw_best["screening_payoff_usd"]),
                "follow_now_payoff_usd": follow_payoff,
                "raw_headroom_vs_follow_now_usd": (
                    float(raw_best["screening_payoff_usd"]) - follow_payoff
                ),
                "stress_best_action_id": str(raw_best["action_id"]),
                "stress_best_payoff_usd": float(
                    raw_best["stress_screening_payoff_usd"]
                ),
                "stress_headroom_vs_follow_now_usd": float(
                    raw_best["stress_screening_payoff_usd"]
                )
                - follow_payoff,
                "nominal_best_action_id": str(nominal_best["action_id"]),
                "nominal_best_payoff_usd": float(
                    nominal_best["nominal_screening_payoff_usd"]
                ),
                "follow_now_nominal_payoff_usd": follow_nominal_payoff,
                "nominal_headroom_vs_follow_now_usd": float(
                    nominal_best["nominal_screening_payoff_usd"]
                )
                - follow_nominal_payoff,
                "capacity_best_action_id_1pct": str(constrained_best["action_id"]),
                "capacity_best_payoff_usd_1pct": float(
                    constrained_best["_constrained_payoff"]
                ),
                "follow_now_capacity_fallback_payoff_usd_1pct": follow_capacity_payoff,
                "capacity_headroom_vs_follow_now_usd_1pct": float(
                    constrained_best["_constrained_payoff"]
                )
                - follow_capacity_payoff,
                "retrospective_only": True,
                "evidence_class": SCREENING_LABEL,
                "split_class": DEVELOPMENT_LABEL,
            }
        )
    return enforce_2025_frame(pd.DataFrame(rows)).reset_index(drop=True)


def build_capacity_sensitivity(payoffs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nonflat = payoffs.loc[payoffs["action_id"].isin(NONFLAT_ACTION_IDS)].copy()
    for fraction in CAPACITY_FRACTIONS:
        frame = nonflat.copy()
        lots = np.minimum(
            np.floor(frame["entry_top3_depth"].astype(float) * fraction),
            np.floor(frame["exit_top3_depth"].astype(float) * fraction),
        )
        frame["capacity_lots"] = lots.astype(int)
        frame["eligible"] = (
            frame["outcome_complete"].astype(bool)
            & frame["entry_ioc_filled"].astype(bool)
            & frame["exit_ioc_filled"].astype(bool)
            & frame["capacity_lots"].ge(CONTRACTS)
        )
        frame["fallback_flat_payoff"] = np.where(
            frame["eligible"], frame["screening_payoff_usd"].fillna(0.0), 0.0
        )
        frame["fallback_flat_nominal_payoff"] = np.where(
            frame["eligible"],
            frame["nominal_screening_payoff_usd"].fillna(0.0),
            0.0,
        )
        for action_id, group in frame.groupby("action_id", sort=False):
            eligible = group.loc[group["eligible"]]
            rows.append(
                {
                    "capacity_fraction": float(fraction),
                    "frozen_primary": bool(
                        math.isclose(fraction, PRIMARY_CAPACITY_FRACTION)
                    ),
                    "action_id": str(action_id),
                    "roots": int(len(group)),
                    "complete_outcomes": int(group["outcome_complete"].sum()),
                    "capacity_eligible_roots": int(group["eligible"].sum()),
                    "capacity_eligible_rate": float(group["eligible"].mean()),
                    "minimum_capacity_lots": int(group["capacity_lots"].min()),
                    "median_capacity_lots": float(group["capacity_lots"].median()),
                    "maximum_capacity_lots": int(group["capacity_lots"].max()),
                    "eligible_total_screening_payoff_usd": float(
                        eligible["screening_payoff_usd"].sum()
                    ),
                    "eligible_total_stress_payoff_usd": float(
                        eligible["stress_screening_payoff_usd"].sum()
                    ),
                    "eligible_total_nominal_payoff_usd": float(
                        eligible["nominal_screening_payoff_usd"].sum()
                    ),
                    "fallback_flat_total_screening_payoff_usd": float(
                        group["fallback_flat_payoff"].sum()
                    ),
                    "fallback_flat_total_stress_payoff_usd": float(
                        group["fallback_flat_payoff"].sum()
                    ),
                    "fallback_flat_total_nominal_payoff_usd": float(
                        group["fallback_flat_nominal_payoff"].sum()
                    ),
                    "evidence_class": SCREENING_LABEL,
                    "split_class": DEVELOPMENT_LABEL,
                }
            )
    return pd.DataFrame(rows)


def _concentration(values: pd.DataFrame, *, value_column: str) -> dict[str, Any]:
    daily = values.groupby("date")[value_column].sum().astype(float).sort_index()
    if daily.empty:
        return {
            "session_days": 0,
            "total_usd": 0.0,
            "daily": [],
            "warning": "no complete outcomes",
        }
    total = float(daily.sum())
    positive = daily.loc[daily > 0]
    absolute = daily.abs()
    absolute_total = float(absolute.sum())
    best_date = str(positive.idxmax()) if not positive.empty else None
    best_value = float(positive.max()) if not positive.empty else None
    without_best = total - best_value if best_value is not None else None
    warning = None
    if total > 0 and without_best is not None and without_best <= 0:
        warning = "positive total becomes nonpositive without its best day"
    elif absolute_total > 0 and float(absolute.max()) / absolute_total >= 0.5:
        warning = "at least half of absolute daily payoff is concentrated in one day"
    standard_deviation = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    return {
        "session_days": int(len(daily)),
        "total_usd": total,
        "mean_daily_usd": float(daily.mean()),
        "annualized_session_sharpe_diagnostic": (
            float(np.sqrt(252.0) * daily.mean() / standard_deviation)
            if standard_deviation > 0
            else None
        ),
        "positive_days": int((daily > 0).sum()),
        "negative_days": int((daily < 0).sum()),
        "largest_positive_day": best_date,
        "largest_positive_day_usd": best_value,
        "net_without_largest_positive_day_usd": without_best,
        "largest_absolute_day_share": (
            float(absolute.max()) / absolute_total if absolute_total > 0 else None
        ),
        "absolute_daily_hhi": (
            float(np.square(absolute / absolute_total).sum())
            if absolute_total > 0
            else None
        ),
        "warning": warning,
        "daily": [
            {"date": str(day), "payoff_usd": float(value)}
            for day, value in daily.items()
        ],
    }


def stability_concentration_report(
    payoffs: pd.DataFrame,
    root_headroom: pd.DataFrame,
) -> dict[str, Any]:
    action_reports: dict[str, Any] = {}
    for action_id in ACTION_IDS:
        group = payoffs.loc[payoffs["action_id"] == action_id].copy()
        complete = group.loc[group["outcome_complete"].astype(bool)].copy()
        entry_filled = group.loc[group["entry_ioc_filled"].astype(bool)]
        capacity = group.copy()
        capacity["capacity_stress_value"] = capacity[
            "capacity_fallback_flat_stress_payoff_usd"
        ].astype(float)
        capacity["capacity_nominal_value"] = capacity[
            "capacity_fallback_flat_nominal_payoff_usd"
        ].astype(float)
        action_reports[action_id] = {
            "roots": int(len(group)),
            "complete_outcomes": int(len(complete)),
            "entry_ioc_fill_rate": (
                float(group["entry_ioc_filled"].mean()) if action_id != "flat" else None
            ),
            "exit_ioc_fill_rate_given_entry_fill": (
                float(entry_filled["exit_ioc_filled"].mean())
                if action_id != "flat" and not entry_filled.empty
                else None
            ),
            "capacity_eligible_rate_1pct": float(
                group["capacity_eligible_1pct"].mean()
            ),
            "primary_stress_complete": _concentration(
                complete, value_column="stress_screening_payoff_usd"
            ),
            "nominal_complete_diagnostic": _concentration(
                complete, value_column="nominal_screening_payoff_usd"
            ),
            "capacity_fallback_flat_stress_1pct": _concentration(
                capacity, value_column="capacity_stress_value"
            ),
            "capacity_fallback_flat_nominal_1pct": _concentration(
                capacity, value_column="capacity_nominal_value"
            ),
        }
    raw_headroom = root_headroom["raw_headroom_vs_follow_now_usd"].astype(float)
    capacity_headroom = root_headroom[
        "capacity_headroom_vs_follow_now_usd_1pct"
    ].astype(float)
    nominal_headroom = root_headroom[
        "nominal_headroom_vs_follow_now_usd"
    ].astype(float)
    return {
        "schema_version": 1,
        "preflight_id": PREFLIGHT_ID,
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "action_reports": action_reports,
        "headroom": {
            "primary_scenario": "adverse_1tick_entry_and_exit",
            "roots": int(len(root_headroom)),
            "stress_total_headroom_vs_follow_now_usd": float(raw_headroom.sum()),
            "stress_mean_headroom_vs_follow_now_usd": float(raw_headroom.mean()),
            "stress_roots_with_positive_headroom": int((raw_headroom > 0).sum()),
            "nominal_total_headroom_vs_follow_now_usd": float(
                nominal_headroom.sum()
            ),
            "nominal_roots_with_positive_headroom": int(
                (nominal_headroom > 0).sum()
            ),
            "capacity_total_headroom_vs_follow_now_usd_1pct": float(
                capacity_headroom.sum()
            ),
            "capacity_roots_with_positive_headroom_1pct": int(
                (capacity_headroom > 0).sum()
            ),
            "stress_best_action_counts": {
                str(key): int(value)
                for key, value in root_headroom["stress_best_action_id"]
                .value_counts()
                .items()
            },
            "nominal_best_action_counts": {
                str(key): int(value)
                for key, value in root_headroom["nominal_best_action_id"]
                .value_counts()
                .items()
            },
            "capacity_best_action_counts_1pct": {
                str(key): int(value)
                for key, value in root_headroom["capacity_best_action_id_1pct"]
                .value_counts()
                .items()
            },
        },
        "warning": (
            "Economic screening uses the adverse-one-tick-per-filled-side stress; "
            "nominal values are secondary diagnostics. Retrospective branch headroom "
            "is an upper bound. It is not "
            "a learned policy, Base-GC comparison, CASH/RENT product, or deployable result."
        ),
    }


def run_l2_aggressive_preflight(
    archive_root: str | Path,
    *,
    episodes: pd.DataFrame,
    active_contract_map: pd.DataFrame,
    active_contract_certificate: Mapping[str, Any],
    health: Mapping[str, Any],
    daily_health_bindings: Iterable[Mapping[str, Any]],
    certified_l2_paths_by_date: Mapping[str, Iterable[str | Path]],
    episodes_sha256: str,
    active_map_sha256: str,
    active_certificate_sha256: str,
) -> L2AggressivePreflightResult:
    episodes = validate_screening_episodes(episodes)
    active_checks = validate_active_contract_inputs(
        episodes,
        active_contract_map,
        active_contract_certificate,
        episodes_sha256=episodes_sha256,
        active_map_sha256=active_map_sha256,
    )
    requests = build_observation_requests(episodes)
    health_checks = validate_health_and_continuity(health, requests)
    health_bindings = [dict(value) for value in daily_health_bindings]
    if not health_bindings:
        raise ValueError("selected daily health bindings cannot be empty")
    observations, source_paths = extract_l2_observations(
        archive_root,
        requests,
        certified_paths_by_date=certified_l2_paths_by_date,
    )
    payoffs = build_long_payoffs(episodes, observations)
    root_headroom = build_root_headroom(payoffs)
    sensitivity = build_capacity_sensitivity(payoffs)
    stability = stability_concentration_report(payoffs, root_headroom)
    nonflat = payoffs.loc[payoffs["action_id"].isin(NONFLAT_ACTION_IDS)].copy()
    bottleneck_depth = np.minimum(
        nonflat["entry_top3_depth"].astype(int),
        nonflat["exit_top3_depth"].astype(int),
    )
    licensed_counts = {
        action_id: int(
            nonflat.loc[
                nonflat["action_id"].eq(action_id), "capacity_eligible_1pct"
            ].sum()
        )
        for action_id in NONFLAT_ACTION_IDS
    }
    minimum_capacity_passed = all(value > 0 for value in licensed_counts.values())
    frozen_capacity_gate = {
        "rule": "floor(0.01 * min(entry opposite top3, exit opposite top3)) >= 1",
        "minimum_required_bottleneck_top3_depth": 100,
        "one_lot_licensed_roots_by_action": licensed_counts,
        "all_nonflat_actions_have_at_least_one_licensed_root": minimum_capacity_passed,
        "minimum_feasibility_status": (
            "PASS_MINIMUM_ONE_LOT_CAPACITY"
            if minimum_capacity_passed
            else "STOP_NO_ONE_LOT_CAPACITY"
        ),
        "observed_bottleneck_top3_depth": {
            "minimum": int(bottleneck_depth.min()),
            "median": float(bottleneck_depth.median()),
            "mean": float(bottleneck_depth.mean()),
            "maximum": int(bottleneck_depth.max()),
        },
        "scope": (
            "A failed minimum capacity screen stops this frozen preflight only; "
            "it is not a rejection of CASH, RENT, HMADRL, or the thesis mechanism."
        ),
    }
    input_certificate = {
        "schema_version": 1,
        "certificate_type": "L2_AGGRESSIVE_PREFLIGHT_INPUTS",
        "certificate_id": PREFLIGHT_ID,
        "passed": True,
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "checks": {
            "corrected_roll_gc_lite_v2": active_checks,
            "health_and_continuity": health_checks,
            "flat_to_flat_episodes": True,
            "complete_registered_action_set_for_every_root": True,
            "development_end": str(DEVELOPMENT_END),
            "sep_dec_2025_read": False,
            "sealed_2026_read": False,
        },
        "bindings": {
            "episodes_sha256": episodes_sha256,
            "active_contract_map_sha256": active_map_sha256,
            "active_contract_certificate_sha256": active_certificate_sha256,
            "selected_daily_health_manifests": health_bindings,
            "selected_daily_health_manifest_set_sha256": sha256_json(
                health_bindings
            ),
        },
        "claim_scope": (
            "This certificate binds inputs for a development-only L2 aggressive "
            "screen. It makes no Base-GC, FIFO, CASH/RENT, policy, profitability, "
            "novelty, or deployment claim."
        ),
    }
    state_coverage = float(observations["state_available"].mean())
    report = {
        **stability,
        "screen_status": frozen_capacity_gate["minimum_feasibility_status"],
        "pipeline_completed": True,
        "economic_conclusion_suppressed": not minimum_capacity_passed,
        "advancement_decision": (
            "ADVANCE_TO_NEXT_GATE"
            if minimum_capacity_passed
            else "DO_NOT_ADVANCE_UNDER_FROZEN_1PCT_CAP"
        ),
        "warning": (
            "SCREENING_ONLY and DEVELOPMENT_ONLY. This is a cheap L2 aggressive "
            "branch preflight, not Base-GC, FIFO execution, a CASH/RENT branch "
            "product, a deployable policy, final OOS evidence, or a novelty verdict."
        ),
        "source_layer": "l2_depth",
        "l2_state_availability": "complete event at ts_recv_ns + 1ms",
        "action_set": list(ACTION_IDS),
        "entry_arrivals": {
            "gc_lite_screen_follow_now": "decision + 5ms",
            "wait_250ms": "decision + 255ms",
            "wait_1s": "decision + 1005ms",
        },
        "common_exit_arrival": "frozen exit decision + 5ms",
        "order_type": "aggressive IOC",
        "contracts": CONTRACTS,
        "fee_usd_per_side": FEE_USD_PER_SIDE,
        "tick_value_usd": TICK_VALUE_USD,
        "economic_screening_primary_scenario": "adverse_1tick_entry_and_exit",
        "execution_scenarios": {
            "nominal_l2_cross": (
                "one lot at contemporaneous L2 best price plus $3 per filled side"
            ),
            "adverse_1tick_entry_and_exit": (
                "nominal less one adverse GC tick at entry and one at exit "
                "($20 per filled round trip), plus $3 per filled side"
            ),
        },
        "primary_capacity_rule": "floor(0.01 * opposite-side displayed top-3 depth)",
        "frozen_capacity_gate": frozen_capacity_gate,
        "capacity_sensitivity_fractions": list(CAPACITY_FRACTIONS),
        "episodes": int(len(episodes)),
        "long_payoff_rows": int(len(payoffs)),
        "expected_long_payoff_rows": int(len(episodes) * len(ACTION_IDS)),
        "observation_requests": int(len(observations)),
        "complete_action_set_roots": int(len(root_headroom)),
        "incomplete_action_set_roots": 0,
        "state_available_rate": state_coverage,
        "state_unavailable_requests": int((~observations["state_available"]).sum()),
        "source_l2_parts": int(len(source_paths)),
        "source_partition_years": [2025],
        "latest_outcome_clock_date": str(
            pd.to_datetime(observations["arrival_time_ns"], unit="ns", utc=True)
            .max()
            .date()
        ),
        "sep_dec_2025_read": False,
        "sealed_2026_read": False,
        "input_certificate_passed": True,
    }
    return L2AggressivePreflightResult(
        report=report,
        input_certificate=input_certificate,
        observations=observations,
        payoffs=payoffs,
        root_headroom=root_headroom,
        capacity_sensitivity=sensitivity,
        source_paths=source_paths,
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


def write_l2_aggressive_preflight(
    output_dir: str | Path,
    result: L2AggressivePreflightResult,
    *,
    supersedes: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"evidence output is create-new-only; choose a new versioned path: {target}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent)
    )
    report = dict(result.report)
    supersession: dict[str, Any] | None = None
    if supersedes is not None:
        supersession = {
            "path": str(Path(supersedes).expanduser().resolve()),
            "reason": "replaces an upstream-invalidated provisional preflight",
        }
        report["supersedes"] = supersession
    try:
        paths = {
            "payoffs_long": _atomic_parquet(
                staging / "payoffs_long.parquet", result.payoffs
            ),
            "root_headroom": _atomic_parquet(
                staging / "root_headroom.parquet", result.root_headroom
            ),
            "capacity_sensitivity": _atomic_parquet(
                staging / "capacity_sensitivity.parquet",
                result.capacity_sensitivity,
            ),
            "l2_observations": _atomic_parquet(
                staging / "l2_observations.parquet", result.observations
            ),
            "input_certificate": write_json(
                staging / "input_certificate.json", result.input_certificate
            ),
            "screening_report": write_json(
                staging / "SCREENING_ONLY.json", report
            ),
        }
        artifact_bindings = {
            name: {
                "path": str(target / path.name),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        }
        source_identity = [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in result.source_paths
        ]
        manifest = {
            "schema_version": 1,
            "preflight_id": PREFLIGHT_ID,
            "evidence_class": SCREENING_LABEL,
            "split_class": DEVELOPMENT_LABEL,
            "artifacts": artifact_bindings,
            "artifact_set_sha256": sha256_json(artifact_bindings),
            "source_l2_partitions": source_identity,
            "source_l2_partition_set_sha256": sha256_json(source_identity),
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
            "claim_scope": result.input_certificate["claim_scope"],
        }
        if supersession is not None:
            manifest["supersedes"] = supersession
        staging_manifest = write_json(staging / "run_manifest.json", manifest)
        if target.exists():
            raise FileExistsError(
                f"evidence output appeared during staging; refusing overwrite: {target}"
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


def run_l2_aggressive_preflight_from_paths(
    *,
    archive_root: str | Path,
    episodes_path: str | Path,
    active_contract_map_path: str | Path,
    active_contract_certificate_path: str | Path,
) -> L2AggressivePreflightResult:
    episodes_source = Path(episodes_path).expanduser().resolve()
    map_source = Path(active_contract_map_path).expanduser().resolve()
    certificate_source = Path(active_contract_certificate_path).expanduser().resolve()
    # This is the disk trust boundary: never accept a self-asserted certificate.
    # The verifier re-hashes every bound input and recomputes the full causal
    # active-contract screen before returning the canonical certificate.
    certificate = verify_active_contract_certificate(certificate_source)
    configs = certificate.get("configs")
    data_binding = configs.get("data") if isinstance(configs, Mapping) else None
    if not isinstance(data_binding, Mapping) or not data_binding.get("path"):
        raise ValueError("verified certificate has no bound data configuration")
    certified_archive_root = resolve_archive_root(load_yaml(data_binding["path"]))
    supplied_archive_root = Path(archive_root).expanduser().resolve()
    if supplied_archive_root != certified_archive_root:
        raise ValueError(
            "L2 archive root does not match the archive bound by the verified "
            f"active-contract certificate: {supplied_archive_root} != "
            f"{certified_archive_root}"
        )
    episodes = pd.read_parquet(episodes_source)
    active = pd.read_parquet(map_source)
    requests = build_observation_requests(validate_screening_episodes(episodes))
    health, health_bindings, certified_l2_paths = load_selected_daily_health_manifests(
        archive_root, requests
    )
    return run_l2_aggressive_preflight(
        archive_root,
        episodes=episodes,
        active_contract_map=active,
        active_contract_certificate=certificate,
        health=health,
        daily_health_bindings=health_bindings,
        certified_l2_paths_by_date=certified_l2_paths,
        episodes_sha256=sha256_file(episodes_source),
        active_map_sha256=sha256_file(map_source),
        active_certificate_sha256=sha256_file(certificate_source),
    )
