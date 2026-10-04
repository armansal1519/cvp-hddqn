from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from kx_mbo_core.active_contract_v2 import (
    _partition_date_from_l1_path,
    _session_contract_volume,
    build_past_only_active_contract_map,
)
from kx_mbo_core.baseline_pipeline import aggregate_l1_to_5m, sealed_l1_paths
from kx_mbo_core.config import load_yaml, resolve_archive_root

from .bars import build_causal_timeframes
from .contract import registered_contract
from .empirical import (
    DEVELOPMENT_END,
    _canonical_json_sha256,
    _decorate_timeframes,
    _state_segments,
    _write_json,
    _write_parquet,
    scan_timeframe_support_cutoffs,
)
from .prepared import sha256_file
from .sessions import globex_session_dates


@dataclass(frozen=True)
class SupportScanResult:
    output_directory: Path
    run_manifest_path: Path
    report_path: Path
    run_manifest_sha256: str
    report_sha256: str


def _training_session_close_cutoff(train_end: str) -> int:
    # One second clears the registered +1 ms completed-bar decode latency while
    # remaining inside the closed session and before the next 17:00 CT open.
    local = pd.Timestamp(f"{train_end} 16:00:01", tz="America/Chicago")
    return int(local.tz_convert("UTC").value)


def _verify_l1_source_chain(
    archive_root: Path, paths: Sequence[Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    l1_bindings: list[dict[str, Any]] = []
    manifests: dict[str, dict[str, Any]] = {}
    for path in paths:
        source_date = _partition_date_from_l1_path(path)
        manifest_path = archive_root / "_manifests" / f"{source_date}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing daily source manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or payload.get("status") != "passed":
            raise ValueError(f"daily source manifest did not pass: {manifest_path}")
        relative = str(path.relative_to(archive_root))
        outputs = [
            item
            for item in payload.get("outputs", ())
            if isinstance(item, Mapping)
            and item.get("layer") == "l1_trades"
            and str(item.get("relative_path", "")) == relative
        ]
        if len(outputs) != 1:
            raise ValueError(f"manifest does not uniquely bind L1 source: {path}")
        observed = sha256_file(path)
        if observed != str(outputs[0].get("sha256", "")):
            raise ValueError(f"manifest-to-L1 hash mismatch: {path}")
        l1_bindings.append(
            {
                "date": source_date,
                "path": str(path),
                "sha256": observed,
                "rows": int(outputs[0].get("rows", -1)),
                "size_bytes": int(outputs[0].get("size", -1)),
            }
        )
        manifests[source_date] = {
            "date": source_date,
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "status": "passed",
        }
    return l1_bindings, [manifests[key] for key in sorted(manifests)]


def run_development_support_scan(
    *,
    data_config_path: str | Path,
    roll_config_path: str | Path,
    split_config_path: str | Path,
    output_directory: str | Path,
    command: Sequence[str] = (),
) -> SupportScanResult:
    """Persist an authenticated upper-bound support audit without model training."""

    started = time.perf_counter()
    data_path = Path(data_config_path).expanduser().resolve()
    roll_path = Path(roll_config_path).expanduser().resolve()
    split_path = Path(split_config_path).expanduser().resolve()
    data_config = load_yaml(data_path)
    roll_config = load_yaml(roll_path)
    split_config = load_yaml(split_path)
    archive_root = resolve_archive_root(data_config)
    paths = sealed_l1_paths(archive_root, through=DEVELOPMENT_END)
    l1_bindings, manifest_bindings = _verify_l1_source_chain(archive_root, paths)
    bars = aggregate_l1_to_5m(paths, through=DEVELOPMENT_END)
    volume = _session_contract_volume(
        bars,
        source_partition_dates=(
            _partition_date_from_l1_path(path) for path in paths
        ),
        degraded_utc_dates=data_config.get("degraded_dates", ()),
    )
    active = build_past_only_active_contract_map(
        volume, roll_config=roll_config
    )

    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace support scan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        active_path = temporary / "active_contract_map.parquet"
        _write_parquet(active_path, active)
        active_hash = sha256_file(active_path)
        bars["bar_end_ns"] = pd.to_datetime(bars["bar_end"], utc=True).astype("int64")
        bars["date"] = globex_session_dates(bars["bar_end_ns"]).to_numpy()
        bars = bars.merge(
            active[["date", "selected_raw_symbol"]].rename(
                columns={"selected_raw_symbol": "selected_symbol"}
            ),
            on="date",
            how="inner",
            validate="many_to_one",
        )
        bars = bars.loc[
            bars["raw_symbol"].astype(str).eq(bars["selected_symbol"].astype(str))
        ].sort_values("bar_end_ns", kind="mergesort").reset_index(drop=True)
        bars["state_segment_id"] = _state_segments(active, bars)
        m5 = bars[
            [
                "date",
                "raw_symbol",
                "continuity_segment_id",
                "state_segment_id",
                "bar_end_ns",
                "available_time_ns",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        raw_timeframes = build_causal_timeframes(m5)
        decorated = _decorate_timeframes(
            raw_timeframes,
            active_map_hash=active_hash,
            contract=registered_contract("base_strong_gc"),
        )
        timeframes = {
            name: decorated.loc[decorated["timeframe"].eq(name)]
            .drop(columns="timeframe")
            .reset_index(drop=True)
            for name in ("H1", "M15", "M5")
        }
        cutoffs: dict[str, int] = {}
        fold_identity: dict[str, dict[str, Any]] = {}
        for fold in split_config.get("development_folds", ()):
            name = str(fold["name"])
            train_end = str(fold["train_end"])
            cutoffs[name] = _training_session_close_cutoff(train_end)
            fold_identity[name] = {
                "train_end_trading_date": train_end,
                "training_cutoff_ns": cutoffs[name],
                "validation_start": str(fold["validation_start"]),
                "validation_end": str(fold["validation_end"]),
            }
        cutoffs["all_development_through_august"] = _training_session_close_cutoff(
            str(DEVELOPMENT_END)
        )
        fold_identity["all_development_through_august"] = {
            "train_end_trading_date": str(DEVELOPMENT_END),
            "training_cutoff_ns": cutoffs["all_development_through_august"],
            "validation_start": None,
            "validation_end": None,
        }
        support = scan_timeframe_support_cutoffs(
            timeframes,
            contract=registered_contract("base_paper_gc"),
            training_cutoffs_ns=cutoffs,
        )
        for name, identity in fold_identity.items():
            identity["support"] = support[name]

        source_bindings = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_SUPPORT_SCAN_SOURCE_BINDINGS",
            "source_l1_files": l1_bindings,
            "source_daily_manifests": manifest_bindings,
            "source_l1_set_sha256": _canonical_json_sha256(l1_bindings),
            "source_manifest_set_sha256": _canonical_json_sha256(manifest_bindings),
            "configs": {
                "data": {"path": str(data_path), "sha256": sha256_file(data_path)},
                "roll": {"path": str(roll_path), "sha256": sha256_file(roll_path)},
                "splits": {"path": str(split_path), "sha256": sha256_file(split_path)},
            },
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
        }
        source_path = temporary / "source_bindings.json"
        _write_json(source_path, source_bindings)
        first_supported = {}
        ordered_cutoffs = [
            str(value["name"]) for value in split_config.get("development_folds", ())
        ] + ["all_development_through_august"]
        for agent in ("H1", "M15", "M5"):
            first_supported[agent] = next(
                (
                    name
                    for name in ordered_cutoffs
                    if support[name][agent]["support_gate_passed"]
                ),
                None,
            )
        report = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_DEVELOPMENT_SUPPORT_SCAN",
            "evidence_class": "DEVELOPMENT_ONLY_COUNT_ONLY",
            "passed": True,
            "period": {"start": "2025-01-01", "end": str(DEVELOPMENT_END)},
            "active_period": {
                "start": str(active["date"].min()),
                "end": str(active["date"].max()),
                "sessions": int(active["date"].nunique()),
                "roll_events": int(active["rolled"].astype(bool).sum()),
            },
            "source_l1_files": len(paths),
            "active_timeframe_rows": {
                name: len(timeframes[name]) for name in ("H1", "M15", "M5")
            },
            "rules": {
                "state_segments": (
                    "cross the registered daily maintenance break; change on active "
                    "contract roll or source continuity segment"
                ),
                "window_continuity": (
                    "exact timeframe step or registered 16:00-17:00 CT break only; "
                    "weekends, holidays, missing bars, and rolls cannot be bridged"
                ),
                "decision_cutoff": "right-labelled completed bar end <=12:25 America/Chicago",
                "sample_open": "only at registered action-lock renewal",
                "maturity": "exactly tw agent bars later inside the same active reward boundary",
                "support_floor": 512,
                "purge_embargo_treatment": (
                    "not subtracted: these are optimistic upper bounds; the registered "
                    "72h purge/one-session embargo can only reduce support"
                ),
            },
            "folds": fold_identity,
            "first_supported_cutoff_by_agent": first_supported,
            "earliest_all_agent_supported_cutoff": next(
                (
                    name
                    for name in ordered_cutoffs
                    if support[name]["all_agents_support_gate_passed"]
                ),
                None,
            ),
            "five_fold_protocol_executable": False,
            "five_seed_training_executable": False,
            "terminal_reason": (
                "H1 never reaches 512 matured renewal transitions: the maximum "
                f"through August is {support['all_development_through_august']['H1']['maximum_matured_training_transitions']}."
            ),
            "why_naive_hour_counts_are_wrong": (
                "Raw H1 bars are not training samples. A 60-bar causal lookback must "
                "warm after every non-bridgeable weekend/holiday/missing-bar gap and "
                "roll; only bars through 12:25 CT are decision-eligible; H1 opens one "
                "sample per five-bar action lock; and that sample needs five later H1 "
                "states in the same reward boundary to mature."
            ),
            "required_remediation": (
                "Acquire and audit pre-2025 history, rebuild the past-only roll map and "
                "prepared product, and rerun this exact scan. Do not lower 512."
            ),
            "command": list(command),
            "runtime_seconds": time.perf_counter() - started,
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
            "model_training_performed": False,
            "profitability_evaluated": False,
        }
        report_path = temporary / "support_report.json"
        _write_json(report_path, report)
        run_manifest = {
            "schema_version": 1,
            "artifact_type": "BASE_GC_DEVELOPMENT_SUPPORT_SCAN_RUN",
            "evidence_class": "DEVELOPMENT_ONLY_COUNT_ONLY",
            "artifacts": {
                "support_report": {
                    "path": report_path.name,
                    "sha256": sha256_file(report_path),
                },
                "source_bindings": {
                    "path": source_path.name,
                    "sha256": sha256_file(source_path),
                },
                "active_contract_map": {
                    "path": active_path.name,
                    "sha256": active_hash,
                },
            },
            "source_l1_set_sha256": source_bindings["source_l1_set_sha256"],
            "source_manifest_set_sha256": source_bindings[
                "source_manifest_set_sha256"
            ],
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
            "claim_scope": (
                "Count-only optimistic support upper bound; no model, performance, "
                "profitability, RENT, or HMADRL claim."
            ),
        }
        run_path = temporary / "run_manifest.json"
        _write_json(run_path, run_manifest)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    final_run = output / "run_manifest.json"
    final_report = output / "support_report.json"
    return SupportScanResult(
        output_directory=output,
        run_manifest_path=final_run,
        report_path=final_report,
        run_manifest_sha256=sha256_file(final_run),
        report_sha256=sha256_file(final_report),
    )
