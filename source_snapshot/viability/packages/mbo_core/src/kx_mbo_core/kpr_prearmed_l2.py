"""Causal L2 evidence product for pre-armed kinetic proofreading.

The product deliberately keeps the four pre-decision proof observations apart
from the immediate-follow economic label.  Proof features are available no
later than the GC-lite decision clock; the one-lot capacity decision and payoff
are outcomes bound at the frozen ``decision + 5ms`` action arrival.
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

import numpy as np
import pandas as pd

from .active_contract_v2 import verify_active_contract_certificate
from .artifacts import sha256_file, sha256_json, write_json
from .capacity_v2 import FROZEN_CAPACITY_V2, TRAILING_WINDOW_NS
from .capacity_v2_january_smoke import (
    evaluate_capacity_rows,
    reconstruct_trailing_volume,
)
from .config import load_yaml, resolve_archive_root
from .l2_aggressive_preflight import (
    DEVELOPMENT_END,
    DEVELOPMENT_LABEL,
    FIVE_MILLISECONDS_NS,
    PREFLIGHT_ID,
    SCREENING_LABEL,
    extract_l2_observations,
    load_selected_daily_health_manifests,
    validate_active_contract_inputs,
    validate_health_and_continuity,
    validate_screening_episodes,
)


PRODUCT_SCHEMA_VERSION = 1
PRODUCT_ID = "kpr_prearmed_l2_screening_v1"
IMMEDIATE_ACTION_ID = "gc_lite_screen_follow_now"
CAPACITY_QUANTITY = 1

MILLISECOND_NS = 1_000_000
PROOF_CLOCKS: tuple[tuple[str, int], ...] = (
    ("t_minus_750ms", -750 * MILLISECOND_NS),
    ("t_minus_500ms", -500 * MILLISECOND_NS),
    ("t_minus_250ms", -250 * MILLISECOND_NS),
    ("t_0ms", 0),
)
PROOF_CLOCK_IDS = tuple(clock_id for clock_id, _ in PROOF_CLOCKS)


FEATURE_REGISTRY: dict[str, Any] = {
    "schema_version": 1,
    "registry_id": "kpr_prearmed_directional_l2_features_v1",
    "information_clock": (
        "Each row uses the last complete L2 event with ts_recv_ns + 1ms no "
        "later than its proof_time_ns. No interpolation or forward fill is used."
    ),
    "direction_convention": "+1=long candidate; -1=short candidate",
    "features": [
        {
            "feature_id": "spread_ticks",
            "formula": "best_ask_ticks - best_bid_ticks",
            "role": "quantity/control",
        },
        {
            "feature_id": "mid_ticks",
            "formula": "(best_ask_ticks + best_bid_ticks) / 2",
            "role": "price-state control",
        },
        {
            "feature_id": "directional_support_best_depth",
            "formula": "best_bid_size if direction=+1 else best_ask_size",
            "role": "same-direction displayed support",
        },
        {
            "feature_id": "directional_opposite_best_depth",
            "formula": "best_ask_size if direction=+1 else best_bid_size",
            "role": "immediate aggressive-side dose",
        },
        {
            "feature_id": "directional_support_top3_depth",
            "formula": "bid_top3_depth if direction=+1 else ask_top3_depth",
            "role": "same-direction displayed dose",
        },
        {
            "feature_id": "directional_opposite_top3_depth",
            "formula": "ask_top3_depth if direction=+1 else bid_top3_depth",
            "role": "aggressive-side displayed dose",
        },
        {
            "feature_id": "directional_top3_imbalance",
            "formula": (
                "direction * (bid_top3_depth - ask_top3_depth) / "
                "(bid_top3_depth + ask_top3_depth); missing when denominator=0"
            ),
            "role": "signed proof signal",
        },
        {
            "feature_id": "directional_microprice_displacement_ticks",
            "formula": (
                "direction * (((ask*bid_size + bid*ask_size) / "
                "(bid_size+ask_size)) - mid); missing when denominator=0"
            ),
            "role": "signed touch pressure",
        },
        {
            "feature_id": "directional_mid_change_from_previous_ticks",
            "formula": "direction * (mid_t - mid_previous_proof_clock)",
            "role": "one-step proof transition",
        },
        {
            "feature_id": "directional_mid_change_from_first_ticks",
            "formula": "direction * (mid_t - mid_t_minus_750ms)",
            "role": "cumulative proof transition",
        },
        {
            "feature_id": "support_top3_change_from_previous",
            "formula": "support_top3_t - support_top3_previous_proof_clock",
            "role": "support persistence transition",
        },
        {
            "feature_id": "opposite_top3_change_from_previous",
            "formula": "opposite_top3_t - opposite_top3_previous_proof_clock",
            "role": "opposition persistence transition",
        },
        {
            "feature_id": "directional_imbalance_change_from_previous",
            "formula": "directional_top3_imbalance_t - previous_proof_clock",
            "role": "signed proof transition",
        },
    ],
    "forbidden_feature_columns": [
        "label_nominal_payoff_usd",
        "label_stress_payoff_usd",
        "label_capacity_v2_eligible_q1",
        "label_capacity_v2_reason",
    ],
}


class KPRPrearmedError(ValueError):
    """Raised when the KPR pre-armed evidence contract is violated."""


@dataclass(frozen=True)
class KPRPrearmedL2Result:
    report: dict[str, Any]
    input_certificate: dict[str, Any]
    proof_requests: pd.DataFrame
    observations: pd.DataFrame
    features: pd.DataFrame
    root_labels: pd.DataFrame
    capacity_rows: pd.DataFrame
    atomic_trade_groups: pd.DataFrame
    session_quarantine: pd.DataFrame
    feature_registry: dict[str, Any]
    source_paths: tuple[Path, ...]
    source_l0_paths: tuple[Path, ...]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KPRPrearmedError(f"{name} is missing columns: {missing}")


def _as_date(value: Any) -> date:
    return pd.Timestamp(value).date()


def _utc_date_from_ns(value: int) -> str:
    return str(pd.Timestamp(int(value), unit="ns", tz="UTC").date())


def _development_clock_guard(values: Iterable[int], *, name: str) -> None:
    clocks = pd.to_datetime(
        pd.Series(list(values), dtype="int64"), unit="ns", utc=True, errors="raise"
    )
    if clocks.empty:
        raise KPRPrearmedError(f"{name} cannot be empty")
    minimum = clocks.min().date()
    maximum = clocks.max().date()
    if minimum < date(2025, 1, 1) or maximum > DEVELOPMENT_END:
        raise PermissionError(
            f"{name} is restricted to 2025-01-01 through {DEVELOPMENT_END}; "
            f"observed {minimum} through {maximum}"
        )


def build_proof_requests(episodes: pd.DataFrame) -> pd.DataFrame:
    """Build exactly four pre-armed observations for every retained root."""

    frame = validate_screening_episodes(episodes)
    rows: list[dict[str, Any]] = []
    for episode in frame.itertuples(index=False):
        decision = int(episode.entry_decision_available_time_ns)
        for proof_step, (clock_id, offset_ns) in enumerate(PROOF_CLOCKS):
            proof_time = decision + int(offset_ns)
            rows.append(
                {
                    "root_id": str(episode.opportunity_id),
                    "session_date": str(episode.date),
                    "raw_symbol": str(episode.raw_symbol),
                    "continuity_segment_id": int(episode.continuity_segment_id),
                    "direction": int(episode.direction),
                    "decision_time_ns": decision,
                    "proof_step": proof_step,
                    "proof_clock_id": clock_id,
                    "proof_offset_ns": int(offset_ns),
                    "proof_time_ns": proof_time,
                    "arrival_time_ns": proof_time,
                    "observation_id": (
                        f"{episode.opportunity_id}::kpr_prearmed::{clock_id}"
                    ),
                }
            )
    requests = pd.DataFrame(rows)
    if len(requests) != len(frame) * len(PROOF_CLOCKS):
        raise AssertionError("proof request construction changed the frozen lattice")
    if requests["observation_id"].duplicated().any():
        raise KPRPrearmedError("proof observation IDs are not unique")
    counts = requests.groupby("root_id", sort=False)["proof_clock_id"].agg(list)
    expected = list(PROOF_CLOCK_IDS)
    if any(value != expected for value in counts):
        raise AssertionError("a root does not have the ordered four-clock lattice")
    requests["date"] = requests["proof_time_ns"].map(_utc_date_from_ns)
    _development_clock_guard(requests["proof_time_ns"], name="KPR proof clocks")
    return requests.sort_values(
        ["arrival_time_ns", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)


def _empty_quarantine() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": pd.Series(dtype="object"),
            "reason": pd.Series(dtype="object"),
            "selection_basis": pd.Series(dtype="object"),
            "source_reference": pd.Series(dtype="object"),
            "outcome_fields_read": pd.Series(dtype="bool"),
            "quarantined_roots": pd.Series(dtype="int64"),
        }
    )


def load_session_quarantine(
    path: str | Path | None,
    *,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load an explicit calendar-only early-close quarantine ledger.

    Quarantine selection must be made without opening payoff or capacity rows.
    The narrow schema prevents outcome-conditioned removal from being disguised
    as a data-quality exclusion.
    """

    if path is None:
        frame = _empty_quarantine()
        return frame, {
            "path": None,
            "sha256": sha256_json([]),
            "entries": 0,
            "selection_basis": "NO_QUARANTINE",
        }
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise KPRPrearmedError("session quarantine must be a JSON object")
    if int(payload.get("schema_version", 0)) != 1:
        raise KPRPrearmedError("unsupported session quarantine schema")
    if str(payload.get("artifact_type", "")) != "KPR_PREARMED_SESSION_QUARANTINE":
        raise KPRPrearmedError("wrong session quarantine artifact type")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise KPRPrearmedError("session quarantine has no entries list")
    rows: list[dict[str, Any]] = []
    episode_dates = episodes["date"].astype(str)
    for value in entries:
        if not isinstance(value, Mapping):
            raise KPRPrearmedError("session quarantine entries must be objects")
        session_date = str(_as_date(value.get("session_date")))
        if _as_date(session_date) > DEVELOPMENT_END:
            raise PermissionError("session quarantine reached Sep-Dec 2025 or 2026")
        if str(value.get("reason", "")) != "early_close":
            raise KPRPrearmedError("only predeclared early_close quarantine is allowed")
        if str(value.get("selection_basis", "")) != "PRE_OUTCOME_EXCHANGE_CALENDAR":
            raise KPRPrearmedError("early-close quarantine is not calendar-only")
        if value.get("outcome_fields_read") is not False:
            raise KPRPrearmedError("quarantine selection read outcome fields")
        source_reference = str(value.get("source_reference", "")).strip()
        if not source_reference:
            raise KPRPrearmedError("quarantine entry lacks a source reference")
        root_count = int(episode_dates.eq(session_date).sum())
        rows.append(
            {
                "session_date": session_date,
                "reason": "early_close",
                "selection_basis": "PRE_OUTCOME_EXCHANGE_CALENDAR",
                "source_reference": source_reference,
                "outcome_fields_read": False,
                "quarantined_roots": root_count,
            }
        )
    frame = pd.DataFrame(rows) if rows else _empty_quarantine()
    if frame["session_date"].duplicated().any():
        raise KPRPrearmedError("session quarantine repeats a date")
    frame = frame.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    return frame, {
        "path": str(source),
        "sha256": sha256_file(source),
        "entries": int(len(frame)),
        "selection_basis": "PRE_OUTCOME_EXCHANGE_CALENDAR",
    }


def derive_directional_proof_features(
    requests: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """Derive transparent directional L2 proof features without outcomes."""

    _require_columns(
        requests,
        {
            "observation_id",
            "root_id",
            "proof_step",
            "proof_clock_id",
            "proof_offset_ns",
            "proof_time_ns",
            "direction",
        },
        name="proof requests",
    )
    _require_columns(
        observations,
        {
            "observation_id",
            "arrival_time_ns",
            "state_available",
            "book_uncrossed",
            "book_epoch",
            "state_available_time_ns",
            "best_bid_ticks",
            "best_ask_ticks",
            "best_bid_size",
            "best_ask_size",
            "bid_top3_depth",
            "ask_top3_depth",
        },
        name="L2 observations",
    )
    request_frame = requests.copy()
    observed = observations.copy()
    duplicate_request_columns = [
        column
        for column in observed.columns
        if column != "observation_id" and column in request_frame.columns
    ]
    observed = observed.drop(columns=duplicate_request_columns)
    frame = request_frame.merge(
        observed, on="observation_id", how="left", validate="one_to_one"
    )
    if len(frame) != len(request_frame):
        raise AssertionError("L2 merge changed proof request count")
    if frame["state_available"].isna().any():
        raise KPRPrearmedError("an L2 request has no extraction result")
    frame["proof_state_usable"] = (
        frame["state_available"].astype(bool)
        & frame["book_uncrossed"].astype(bool)
        & frame["state_available_time_ns"].le(frame["proof_time_ns"])
    )

    numeric = (
        "best_bid_ticks",
        "best_ask_ticks",
        "best_bid_size",
        "best_ask_size",
        "bid_top3_depth",
        "ask_top3_depth",
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~frame["proof_state_usable"], column] = np.nan
    direction = frame["direction"].astype(int)
    frame["spread_ticks"] = frame["best_ask_ticks"] - frame["best_bid_ticks"]
    frame["mid_ticks"] = (
        frame["best_ask_ticks"] + frame["best_bid_ticks"]
    ) / 2.0
    long_mask = direction.eq(1)
    frame["directional_support_best_depth"] = np.where(
        long_mask, frame["best_bid_size"], frame["best_ask_size"]
    )
    frame["directional_opposite_best_depth"] = np.where(
        long_mask, frame["best_ask_size"], frame["best_bid_size"]
    )
    frame["directional_support_top3_depth"] = np.where(
        long_mask, frame["bid_top3_depth"], frame["ask_top3_depth"]
    )
    frame["directional_opposite_top3_depth"] = np.where(
        long_mask, frame["ask_top3_depth"], frame["bid_top3_depth"]
    )
    top3_total = frame["bid_top3_depth"] + frame["ask_top3_depth"]
    frame["directional_top3_imbalance"] = np.where(
        top3_total.gt(0),
        direction
        * (frame["bid_top3_depth"] - frame["ask_top3_depth"])
        / top3_total,
        np.nan,
    )
    best_total = frame["best_bid_size"] + frame["best_ask_size"]
    microprice = np.where(
        best_total.gt(0),
        (
            frame["best_ask_ticks"] * frame["best_bid_size"]
            + frame["best_bid_ticks"] * frame["best_ask_size"]
        )
        / best_total,
        np.nan,
    )
    frame["directional_microprice_displacement_ticks"] = direction * (
        microprice - frame["mid_ticks"]
    )
    frame = frame.sort_values(["root_id", "proof_step"], kind="mergesort")
    grouped = frame.groupby("root_id", sort=False)
    frame["directional_mid_change_from_previous_ticks"] = direction * (
        frame["mid_ticks"] - grouped["mid_ticks"].shift()
    )
    first_mid = grouped["mid_ticks"].transform("first")
    frame["directional_mid_change_from_first_ticks"] = direction * (
        frame["mid_ticks"] - first_mid
    )
    frame["support_top3_change_from_previous"] = (
        frame["directional_support_top3_depth"]
        - grouped["directional_support_top3_depth"].shift()
    )
    frame["opposite_top3_change_from_previous"] = (
        frame["directional_opposite_top3_depth"]
        - grouped["directional_opposite_top3_depth"].shift()
    )
    frame["directional_imbalance_change_from_previous"] = (
        frame["directional_top3_imbalance"]
        - grouped["directional_top3_imbalance"].shift()
    )

    root_usable = grouped["proof_state_usable"].transform("all")
    root_count = grouped["proof_step"].transform("size").eq(len(PROOF_CLOCKS))
    epoch_count = grouped["book_epoch"].transform(
        lambda values: values.dropna().nunique()
    )
    frame["proof_window_single_epoch"] = epoch_count.eq(1)
    frame["proof_window_eligible"] = (
        root_usable & root_count & frame["proof_window_single_epoch"]
    )
    frame["evidence_class"] = SCREENING_LABEL
    frame["split_class"] = DEVELOPMENT_LABEL
    if any(column.startswith("label_") for column in frame.columns):
        raise AssertionError("outcome labels leaked into the proof feature table")
    _development_clock_guard(frame["proof_time_ns"], name="derived proof clocks")
    return frame.sort_values(
        ["proof_time_ns", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise KPRPrearmedError(f"{name} must be a JSON object")
    return payload


def _verify_product_directory(
    directory: str | Path,
    *,
    required_artifacts: Mapping[str, str],
    name: str,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, str]]]:
    source = Path(directory).expanduser().resolve()
    manifest_path = source / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {name} run manifest: {manifest_path}")
    manifest = _load_json_object(manifest_path, name=f"{name} run manifest")
    if int(manifest.get("schema_version", 0)) != 1:
        raise KPRPrearmedError(f"unsupported {name} run-manifest schema")
    if str(manifest.get("evidence_class", "")) != SCREENING_LABEL:
        raise KPRPrearmedError(f"{name} is not SCREENING_ONLY")
    if str(manifest.get("split_class", "")) != DEVELOPMENT_LABEL:
        raise KPRPrearmedError(f"{name} is not DEVELOPMENT_ONLY")
    if manifest.get("sealed_2026_read") is not False:
        raise PermissionError(f"{name} opened sealed 2026 data")
    if manifest.get("sep_dec_2025_read") is not False:
        raise PermissionError(f"{name} opened Sep-Dec 2025 data")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise KPRPrearmedError(f"{name} manifest has no artifacts mapping")
    if sha256_json(dict(artifacts)) != str(manifest.get("artifact_set_sha256", "")):
        raise KPRPrearmedError(f"{name} artifact-set hash mismatch")
    paths: dict[str, Path] = {}
    bindings: dict[str, dict[str, str]] = {
        "run_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    }
    for artifact_name, expected_filename in required_artifacts.items():
        declaration = artifacts.get(artifact_name)
        if not isinstance(declaration, Mapping):
            raise KPRPrearmedError(f"{name} is missing artifact {artifact_name!r}")
        declared = Path(str(declaration.get("path", ""))).expanduser()
        if not declared.is_absolute():
            declared = source / declared
        path = declared.resolve()
        if path.parent != source or path.name != expected_filename:
            raise KPRPrearmedError(
                f"{name} artifact {artifact_name!r} escaped its immutable directory"
            )
        if not path.is_file():
            raise FileNotFoundError(f"missing {name} artifact: {path}")
        observed = sha256_file(path)
        if observed != str(declaration.get("sha256", "")):
            raise KPRPrearmedError(f"{name} artifact hash mismatch: {artifact_name}")
        paths[artifact_name] = path
        bindings[artifact_name] = {"path": str(path), "sha256": observed}
    return manifest, paths, bindings


def load_authenticated_payoffs(
    payoff_product_dir: str | Path,
    *,
    episodes_sha256: str,
    active_map_sha256: str,
    active_certificate_sha256: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    manifest, paths, artifact_bindings = _verify_product_directory(
        payoff_product_dir,
        required_artifacts={
            "l2_observations": "l2_observations.parquet",
            "payoffs_long": "payoffs_long.parquet",
            "input_certificate": "input_certificate.json",
            "screening_report": "SCREENING_ONLY.json",
        },
        name="L2 payoff product",
    )
    if str(manifest.get("preflight_id", "")) != PREFLIGHT_ID:
        raise KPRPrearmedError("payoff product has the wrong L2 preflight ID")
    certificate = _load_json_object(
        paths["input_certificate"], name="L2 payoff input certificate"
    )
    if certificate.get("passed") is not True:
        raise KPRPrearmedError("L2 payoff input certificate did not pass")
    if str(certificate.get("certificate_id", "")) != PREFLIGHT_ID:
        raise KPRPrearmedError("L2 payoff certificate has the wrong ID")
    bindings = certificate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise KPRPrearmedError("L2 payoff certificate has no bindings")
    expected = {
        "episodes_sha256": episodes_sha256,
        "active_contract_map_sha256": active_map_sha256,
        "active_contract_certificate_sha256": active_certificate_sha256,
    }
    for key, digest in expected.items():
        if str(bindings.get(key, "")) != digest:
            raise KPRPrearmedError(f"L2 payoff certificate does not bind exact {key}")
    report = _load_json_object(paths["screening_report"], name="L2 payoff report")
    if report.get("sealed_2026_read") is not False:
        raise PermissionError("L2 payoff report opened sealed 2026 data")
    if report.get("sep_dec_2025_read") is not False:
        raise PermissionError("L2 payoff report opened Sep-Dec 2025 data")
    return (
        pd.read_parquet(paths["payoffs_long"]),
        pd.read_parquet(paths["l2_observations"]),
        certificate,
        manifest,
        artifact_bindings,
    )


def _validate_capacity_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = load_yaml(source)
    if payload.get("rule_id") != "gc_one_lot_causal_capacity_v2":
        raise KPRPrearmedError("capacity config has the wrong rule ID")
    if payload.get("quantity_contracts") != CAPACITY_QUANTITY:
        raise KPRPrearmedError("capacity config escaped the Q=1 estimand")
    exposure = payload.get("exposure_increasing")
    if not isinstance(exposure, Mapping):
        raise KPRPrearmedError("capacity config lacks exposure-increasing rules")
    expected_exposure = {
        "minimum_opposite_best_depth_contracts": 2,
        "minimum_opposite_top3_depth_contracts": 20,
        "maximum_top3_fraction": 0.05,
    }
    for key, expected in expected_exposure.items():
        if exposure.get(key) != expected:
            raise KPRPrearmedError(f"capacity config changed {key!r}")
    trailing = exposure.get("trailing_executed_volume")
    if not isinstance(trailing, Mapping):
        raise KPRPrearmedError("capacity config lacks trailing-volume rules")
    expected_trailing = {
        "window_seconds": 300,
        "minimum_contracts": 200,
        "maximum_fraction": 0.005,
        "source": "same_symbol_aggressive_trade_records",
    }
    for key, expected in expected_trailing.items():
        if trailing.get(key) != expected:
            raise KPRPrearmedError(f"capacity config changed trailing {key!r}")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "rule_id": str(payload["rule_id"]),
        "capacity_contract_sha256": FROZEN_CAPACITY_V2.content_sha256,
    }


def build_immediate_capacity_requests(
    observations: pd.DataFrame,
    payoffs: pd.DataFrame,
) -> pd.DataFrame:
    """Build the exact Q=1 request at the frozen immediate-action arrival."""

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
        name="payoff-product L2 observations",
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
        name="payoff-product rows",
    )
    action_rows = payoffs.loc[
        payoffs["action_id"].astype(str).eq(IMMEDIATE_ACTION_ID)
    ].copy()
    if action_rows.empty:
        raise KPRPrearmedError("payoff product has no immediate-follow rows")
    action_rows["root_id"] = action_rows["root_id"].astype(str)
    action_rows["observation_id"] = (
        action_rows["root_id"] + "::entry::" + IMMEDIATE_ACTION_ID
    )
    if action_rows["root_id"].duplicated().any():
        raise KPRPrearmedError("payoff product repeats an immediate-follow root")
    selected = observations.loc[
        observations["observation_id"].astype(str).isin(
            set(action_rows["observation_id"])
        )
    ].copy()
    if selected["observation_id"].astype(str).duplicated().any():
        raise KPRPrearmedError("L2 product repeats an immediate-follow state")
    joined = action_rows.merge(
        selected,
        on="observation_id",
        how="left",
        suffixes=("_payoff", "_l2"),
        validate="one_to_one",
    )
    required_joined = [
        "arrival_time_ns",
        "date_l2",
        "raw_symbol_l2",
        "continuity_segment_id_l2",
    ]
    if joined[required_joined].isna().any().any():
        raise KPRPrearmedError("an immediate action lacks its exact L2 state row")
    if not bool(
        joined["entry_arrival_time_ns"].astype("int64").eq(
            joined["arrival_time_ns"].astype("int64")
        ).all()
    ):
        raise KPRPrearmedError("payoff and L2 immediate-arrival clocks disagree")
    if not bool(
        joined["raw_symbol_payoff"].astype(str).eq(
            joined["raw_symbol_l2"].astype(str)
        ).all()
    ):
        raise KPRPrearmedError("payoff and L2 symbols disagree")
    if not bool(
        joined["continuity_segment_id_payoff"].astype(int).eq(
            joined["continuity_segment_id_l2"].astype(int)
        ).all()
    ):
        raise KPRPrearmedError("payoff and L2 continuity segments disagree")
    if set(joined["evidence_class_payoff"].astype(str)) != {SCREENING_LABEL}:
        raise KPRPrearmedError("payoff rows are not SCREENING_ONLY")
    if set(joined["evidence_class_l2"].astype(str)) != {SCREENING_LABEL}:
        raise KPRPrearmedError("L2 rows are not SCREENING_ONLY")
    if set(joined["split_class_payoff"].astype(str)) != {DEVELOPMENT_LABEL}:
        raise KPRPrearmedError("payoff rows are not DEVELOPMENT_ONLY")
    if set(joined["split_class_l2"].astype(str)) != {DEVELOPMENT_LABEL}:
        raise KPRPrearmedError("L2 rows are not DEVELOPMENT_ONLY")
    output = pd.DataFrame(
        {
            "root_id": joined["root_id"].astype(str),
            "action_id": IMMEDIATE_ACTION_ID,
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
    arrival_dates = pd.to_datetime(
        output["arrival_time_ns"], unit="ns", utc=True
    ).dt.date.astype(str)
    if not bool(arrival_dates.eq(output["date"]).all()):
        raise KPRPrearmedError("L2 row date is not its UTC immediate-arrival date")
    if any(_as_date(value) > DEVELOPMENT_END for value in output["date"]):
        raise PermissionError("capacity requests reached Sep-Dec 2025 or 2026")
    output["trailing_window_start_ns"] = (
        output["arrival_time_ns"] - TRAILING_WINDOW_NS
    )
    return output.sort_values(
        ["arrival_time_ns", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)


def resolve_development_manifest_declared_l0(
    certificate: Mapping[str, Any],
    requests: pd.DataFrame,
    *,
    archive_root: str | Path,
) -> tuple[
    dict[str, tuple[Path, ...]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Resolve only manifest-declared Jan-Aug L0 parts for requested dates."""

    requested_dates = set(requests["date"].astype(str))
    if not requested_dates:
        raise KPRPrearmedError("capacity request date set is empty")
    if min(_as_date(value) for value in requested_dates) < date(2025, 1, 1):
        raise PermissionError("capacity request predates the development window")
    if max(_as_date(value) for value in requested_dates) > DEVELOPMENT_END:
        raise PermissionError("capacity request reached Sep-Dec 2025 or 2026")
    bindings = certificate.get("bindings")
    declarations = (
        bindings.get("selected_daily_health_manifests")
        if isinstance(bindings, Mapping)
        else None
    )
    if not isinstance(declarations, list):
        raise KPRPrearmedError("payoff certificate has no daily-manifest ledger")
    by_date: dict[str, Mapping[str, Any]] = {}
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise KPRPrearmedError("daily-manifest declaration is malformed")
        declared_date = str(_as_date(declaration.get("date")))
        if _as_date(declared_date) > DEVELOPMENT_END:
            raise PermissionError("payoff certificate opened a post-August manifest")
        if declared_date in by_date:
            raise KPRPrearmedError(f"duplicate daily manifest: {declared_date}")
        by_date[declared_date] = declaration
    missing = sorted(requested_dates.difference(by_date))
    if missing:
        raise KPRPrearmedError(f"capacity dates lack certified manifests: {missing}")

    certified_root = Path(archive_root).expanduser().resolve()
    l0_by_date: dict[str, tuple[Path, ...]] = {}
    manifest_bindings: list[dict[str, Any]] = []
    l0_bindings: list[dict[str, Any]] = []
    health_by_date: dict[str, dict[str, Any]] = {}
    for selected_date in sorted(requested_dates):
        declaration = by_date[selected_date]
        manifest_path = Path(str(declaration.get("path", ""))).expanduser().resolve()
        expected_manifest = certified_root / "_manifests" / f"{selected_date}.json"
        if manifest_path != expected_manifest:
            raise PermissionError("daily manifest escaped the certified archive root")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing daily manifest: {manifest_path}")
        manifest_sha = sha256_file(manifest_path)
        if manifest_sha != str(declaration.get("sha256", "")):
            raise KPRPrearmedError(f"daily manifest hash mismatch: {selected_date}")
        payload = _load_json_object(manifest_path, name="daily health manifest")
        if str(payload.get("date", "")) != selected_date:
            raise KPRPrearmedError("daily manifest date mismatch")
        if str(payload.get("status", "")).lower() != "passed":
            raise KPRPrearmedError(f"daily manifest did not pass: {selected_date}")
        stats = payload.get("stats")
        if not isinstance(stats, Mapping):
            raise KPRPrearmedError("daily manifest lacks health stats")
        requested_continuities = set(
            requests.loc[
                requests["date"].astype(str).eq(selected_date),
                "continuity_segment_id",
            ].astype(int)
        )
        if requested_continuities != {int(stats.get("continuity_segment_id", -1))}:
            raise KPRPrearmedError("daily manifest continuity does not match requests")
        # An incomplete mutating event or receipt reversal invalidates the book.
        # Partition-wide incomplete non-mutating groups may belong to unrelated
        # instruments at a file boundary. They are recorded below, while the
        # exact selected-symbol F_LAST reconstruction independently requires
        # zero incomplete trade groups and zero unavailable trade groups.
        for field in ("incomplete_mutating_events", "ts_recv_reversals"):
            if int(stats.get(field, -1)) != 0:
                raise KPRPrearmedError(
                    f"daily manifest has nonzero {field}: {selected_date}"
                )
        outputs = payload.get("outputs")
        if not isinstance(outputs, list):
            raise KPRPrearmedError("daily manifest has no outputs ledger")
        selected_outputs = [
            item
            for item in outputs
            if isinstance(item, Mapping) and str(item.get("layer", "")) == "l0"
        ]
        if not selected_outputs:
            raise KPRPrearmedError(f"daily manifest binds no L0: {selected_date}")
        parsed = _as_date(selected_date)
        expected_parent = Path(
            f"l0/year=2025/month={parsed.month:02d}/day={parsed.day:02d}"
        )
        paths: list[Path] = []
        daily_l0: list[dict[str, Any]] = []
        for item in selected_outputs:
            if str(item.get("date", "")) != selected_date:
                raise KPRPrearmedError("L0 output date differs from manifest date")
            relative = Path(str(item.get("relative_path", "")))
            if (
                relative.parent != expected_parent
                or not relative.name.startswith("part-")
                or relative.suffix != ".parquet"
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise PermissionError(f"manifest L0 path escaped its date: {relative}")
            source_path = (certified_root / relative).resolve()
            if source_path.parent != (certified_root / expected_parent).resolve():
                raise PermissionError("resolved L0 path escaped its exact partition")
            if not source_path.is_file():
                raise FileNotFoundError(f"manifest-declared L0 is missing: {source_path}")
            observed_size = int(source_path.stat().st_size)
            if observed_size != int(item.get("size", -1)):
                raise KPRPrearmedError(f"L0 size mismatch: {source_path}")
            observed_sha = sha256_file(source_path)
            if observed_sha != str(item.get("sha256", "")):
                raise KPRPrearmedError(f"L0 hash mismatch: {source_path}")
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
            raise KPRPrearmedError("daily manifest repeats an L0 part")
        daily_l0.sort(key=lambda item: item["path"])
        l0_by_date[selected_date] = tuple(sorted(paths))
        manifest_bindings.append(
            {
                "date": selected_date,
                "path": str(manifest_path),
                "sha256": manifest_sha,
                "l0_parts": len(paths),
                "l0_output_set_sha256": sha256_json(daily_l0),
                "partition_incomplete_non_mutating_events": int(
                    stats.get("incomplete_non_mutating_events", 0)
                ),
            }
        )
        health_by_date[selected_date] = {
            "continuity_segment_id": int(stats["continuity_segment_id"]),
            "first_ts_recv_ns": int(stats["first_ts_recv_ns"]),
            "last_ts_recv_ns": int(stats["last_ts_recv_ns"]),
            "incomplete_mutating_events": 0,
            "incomplete_non_mutating_events": int(
                stats.get("incomplete_non_mutating_events", 0)
            ),
            "ts_recv_reversals": 0,
            "status": "passed",
        }
    manifest_bindings.sort(key=lambda item: item["date"])
    l0_bindings.sort(key=lambda item: item["path"])
    return l0_by_date, manifest_bindings, l0_bindings, health_by_date


def compute_exact_immediate_capacity_v2(
    *,
    archive_root: str | Path,
    observations: pd.DataFrame,
    payoffs: pd.DataFrame,
    payoff_certificate: Mapping[str, Any],
    payoff_manifest: Mapping[str, Any],
    payoff_artifact_bindings: Mapping[str, Mapping[str, str]],
    capacity_config_path: str | Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    tuple[Path, ...],
]:
    """Evaluate exact capacity-v2 from manifest-declared L0 for one action."""

    config_binding = _validate_capacity_config(capacity_config_path)
    requests = build_immediate_capacity_requests(observations, payoffs)
    l0_by_date, manifest_bindings, l0_bindings, health_by_date = (
        resolve_development_manifest_declared_l0(
            payoff_certificate, requests, archive_root=archive_root
        )
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
                payoff_manifest["source_l2_partition_set_sha256"]
            ),
            "l2_observations_sha256": str(
                payoff_artifact_bindings["l2_observations"]["sha256"]
            ),
        }
    )
    with_volume, atomic_groups, coverage = reconstruct_trailing_volume(
        requests, l0_by_date, health_by_date
    )
    capacity_rows = evaluate_capacity_rows(
        with_volume, source_set_sha256=capacity_state_source_sha256
    )
    if len(capacity_rows) != capacity_rows["root_id"].astype(str).nunique():
        raise KPRPrearmedError("exact capacity result is not one row per root")
    if set(capacity_rows["action_id"].astype(str)) != {IMMEDIATE_ACTION_ID}:
        raise KPRPrearmedError("exact capacity result escaped the immediate action")
    coverage_frame = pd.DataFrame(coverage)
    for field in (
        "incomplete_atomic_groups",
        "unavailable_complete_groups",
        "incomplete_trade_groups",
        "unavailable_trade_groups",
    ):
        if int(coverage_frame[field].astype(int).sum()) != 0:
            raise KPRPrearmedError(f"exact capacity L0 coverage failed: {field}")
    bindings = {
        "mode": "EXACT_MANIFEST_DECLARED_L0_IN_PRODUCT",
        "capacity_config": config_binding,
        "payoff_product_run_manifest": payoff_artifact_bindings["run_manifest"],
        "daily_manifests": manifest_bindings,
        "daily_manifest_set_sha256": sha256_json(manifest_bindings),
        "source_l0_partitions": l0_bindings,
        "source_l0_partition_set_sha256": source_set_sha256,
        "combined_l2_l0_capacity_state_sha256": capacity_state_source_sha256,
        "event_atomic_f_last_availability": True,
        "future_exit_eligibility_used": False,
    }
    source_paths = tuple(
        sorted(path for paths in l0_by_date.values() for path in paths)
    )
    return capacity_rows, atomic_groups, bindings, source_paths


def load_authenticated_capacity(
    capacity_product_dir: str | Path,
    *,
    payoff_manifest_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    manifest, paths, artifact_bindings = _verify_product_directory(
        capacity_product_dir,
        required_artifacts={
            "capacity_rows": "capacity_rows.parquet",
            "atomic_trade_groups": "atomic_trade_groups.parquet",
            "input_certificate": "input_certificate.json",
            "screening_report": "SCREENING_ONLY.json",
        },
        name="capacity-v2 product",
    )
    certificate = _load_json_object(
        paths["input_certificate"], name="capacity-v2 input certificate"
    )
    if certificate.get("passed") is not True:
        raise KPRPrearmedError("capacity-v2 input certificate did not pass")
    bindings = certificate.get("bindings")
    if not isinstance(bindings, Mapping):
        raise KPRPrearmedError("capacity-v2 certificate has no bindings")
    config = bindings.get("capacity_config")
    if not isinstance(config, Mapping):
        raise KPRPrearmedError("capacity-v2 certificate has no frozen config binding")
    if str(config.get("capacity_contract_sha256", "")) != FROZEN_CAPACITY_V2.content_sha256:
        raise KPRPrearmedError("capacity product does not use the frozen capacity-v2 contract")
    upstream = bindings.get("upstream")
    upstream_manifest = upstream.get("run_manifest") if isinstance(upstream, Mapping) else None
    if not isinstance(upstream_manifest, Mapping):
        raise KPRPrearmedError("capacity product has no upstream payoff-manifest binding")
    if str(upstream_manifest.get("sha256", "")) != payoff_manifest_sha256:
        raise KPRPrearmedError("capacity product is not bound to the exact payoff product")
    for item in manifest.get("source_l0_partitions", []):
        if not isinstance(item, Mapping):
            raise KPRPrearmedError("capacity source ledger is malformed")
        source_date = item.get("date")
        if source_date is not None and _as_date(source_date) > DEVELOPMENT_END:
            raise PermissionError("capacity product opened Sep-Dec 2025 or 2026 L0")
    report = _load_json_object(paths["screening_report"], name="capacity-v2 report")
    if report.get("sealed_2026_read") is not False:
        raise PermissionError("capacity-v2 report opened sealed 2026 data")
    if report.get("sep_dec_2025_read") is not False:
        raise PermissionError("capacity-v2 report opened Sep-Dec 2025 data")
    return (
        pd.read_parquet(paths["capacity_rows"]),
        pd.read_parquet(paths["atomic_trade_groups"]),
        manifest,
        artifact_bindings,
    )


def build_root_labels(
    episodes: pd.DataFrame,
    payoffs: pd.DataFrame,
    capacity_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Bind exact immediate-follow payoff and Q=1 capacity as non-features."""

    roots = validate_screening_episodes(episodes)[
        [
            "opportunity_id",
            "date",
            "raw_symbol",
            "continuity_segment_id",
            "direction",
            "entry_decision_available_time_ns",
            "exit_decision_available_time_ns",
        ]
    ].rename(columns={"opportunity_id": "root_id", "date": "session_date"})
    _require_columns(
        payoffs,
        {
            "root_id",
            "action_id",
            "date",
            "raw_symbol",
            "continuity_segment_id",
            "direction",
            "entry_delay_ns",
            "entry_arrival_time_ns",
            "exit_arrival_time_ns",
            "outcome_complete",
            "nominal_screening_payoff_usd",
            "stress_screening_payoff_usd",
            "incomplete_reason",
            "evidence_class",
            "split_class",
        },
        name="payoff rows",
    )
    _require_columns(
        capacity_rows,
        {
            "root_id",
            "action_id",
            "raw_symbol",
            "continuity_segment_id",
            "direction",
            "arrival_time_ns",
            "feasible",
            "reason",
            "quantity",
            "capacity_config_sha256",
        },
        name="capacity-v2 rows",
    )
    payoff = payoffs.loc[payoffs["action_id"].eq(IMMEDIATE_ACTION_ID)].copy()
    capacity = capacity_rows.loc[
        capacity_rows["action_id"].eq(IMMEDIATE_ACTION_ID)
    ].copy()
    for name, frame in (("payoff", payoff), ("capacity", capacity)):
        if frame["root_id"].astype(str).duplicated().any():
            raise KPRPrearmedError(f"immediate-follow {name} rows repeat roots")
    expected_roots = set(roots["root_id"].astype(str))
    if set(payoff["root_id"].astype(str)) != expected_roots:
        raise KPRPrearmedError("payoff product lacks exact retained-root coverage")
    if set(capacity["root_id"].astype(str)) != expected_roots:
        raise KPRPrearmedError("capacity-v2 product lacks exact retained-root coverage")
    if set(payoff["evidence_class"].astype(str)) != {SCREENING_LABEL}:
        raise KPRPrearmedError("payoff rows are not SCREENING_ONLY")
    if set(payoff["split_class"].astype(str)) != {DEVELOPMENT_LABEL}:
        raise KPRPrearmedError("payoff rows are not DEVELOPMENT_ONLY")
    if set(pd.to_numeric(capacity["quantity"], errors="raise").astype(int)) != {
        CAPACITY_QUANTITY
    }:
        raise KPRPrearmedError("capacity rows escaped the Q=1 estimand")
    if set(capacity["capacity_config_sha256"].astype(str)) != {
        FROZEN_CAPACITY_V2.content_sha256
    }:
        raise KPRPrearmedError("capacity rows have the wrong frozen contract hash")

    payoff_columns = [
        "root_id",
        "date",
        "raw_symbol",
        "continuity_segment_id",
        "direction",
        "entry_delay_ns",
        "entry_arrival_time_ns",
        "exit_arrival_time_ns",
        "outcome_complete",
        "nominal_screening_payoff_usd",
        "stress_screening_payoff_usd",
        "incomplete_reason",
    ]
    labels = roots.merge(
        payoff[payoff_columns], on="root_id", how="left", validate="one_to_one",
        suffixes=("", "_payoff"),
    )
    identity_checks = (
        (labels["session_date"].astype(str) == labels["date"].astype(str))
        & (labels["raw_symbol"].astype(str) == labels["raw_symbol_payoff"].astype(str))
        & (
            labels["continuity_segment_id"].astype(int)
            == labels["continuity_segment_id_payoff"].astype(int)
        )
        & (labels["direction"].astype(int) == labels["direction_payoff"].astype(int))
    )
    if not bool(identity_checks.all()):
        raise KPRPrearmedError("payoff identity does not match certified episodes")
    expected_arrival = (
        labels["entry_decision_available_time_ns"].astype("int64")
        + FIVE_MILLISECONDS_NS
    )
    if not bool(labels["entry_delay_ns"].astype("int64").eq(FIVE_MILLISECONDS_NS).all()):
        raise KPRPrearmedError("immediate payoff does not use the frozen +5ms delay")
    if not bool(labels["entry_arrival_time_ns"].astype("int64").eq(expected_arrival).all()):
        raise KPRPrearmedError("immediate payoff has the wrong entry-arrival clock")

    capacity_columns = [
        "root_id",
        "raw_symbol",
        "continuity_segment_id",
        "direction",
        "arrival_time_ns",
        "feasible",
        "reason",
        "quantity",
        "capacity_config_sha256",
    ]
    labels = labels.merge(
        capacity[capacity_columns], on="root_id", how="left", validate="one_to_one",
        suffixes=("", "_capacity"),
    )
    capacity_identity = (
        (labels["raw_symbol"].astype(str) == labels["raw_symbol_capacity"].astype(str))
        & (
            labels["continuity_segment_id"].astype(int)
            == labels["continuity_segment_id_capacity"].astype(int)
        )
        & (labels["direction"].astype(int) == labels["direction_capacity"].astype(int))
    )
    if not bool(capacity_identity.all()):
        raise KPRPrearmedError("capacity identity does not match certified episodes")
    if not bool(labels["arrival_time_ns"].astype("int64").eq(expected_arrival).all()):
        raise KPRPrearmedError("capacity-v2 was not evaluated at immediate +5ms arrival")

    output = pd.DataFrame(
        {
            "root_id": labels["root_id"].astype(str),
            "session_date": labels["session_date"].astype(str),
            "raw_symbol": labels["raw_symbol"].astype(str),
            "continuity_segment_id": labels["continuity_segment_id"].astype(int),
            "direction": labels["direction"].astype(int),
            "label_action_id": IMMEDIATE_ACTION_ID,
            "label_entry_arrival_time_ns": labels["entry_arrival_time_ns"].astype("int64"),
            "label_exit_arrival_time_ns": labels["exit_arrival_time_ns"].astype("int64"),
            "label_outcome_complete": labels["outcome_complete"].astype(bool),
            "label_nominal_payoff_usd": labels["nominal_screening_payoff_usd"].astype(float),
            "label_stress_payoff_usd": labels["stress_screening_payoff_usd"].astype(float),
            "label_incomplete_reason": labels["incomplete_reason"],
            "label_capacity_v2_eligible_q1": labels["feasible"].astype(bool),
            "label_capacity_v2_reason": labels["reason"].astype(str),
            "label_capacity_v2_quantity": labels["quantity"].astype(int),
            "label_capacity_v2_config_sha256": labels[
                "capacity_config_sha256"
            ].astype(str),
            "evidence_class": SCREENING_LABEL,
            "split_class": DEVELOPMENT_LABEL,
            "feature_use_forbidden": True,
        }
    )
    _development_clock_guard(
        output["label_entry_arrival_time_ns"], name="immediate label entry clocks"
    )
    _development_clock_guard(
        output["label_exit_arrival_time_ns"], name="immediate label exit clocks"
    )
    return output.sort_values("root_id", kind="mergesort").reset_index(drop=True)


def run_kpr_prearmed_l2_from_paths(
    *,
    archive_root: str | Path,
    episodes_path: str | Path,
    active_contract_map_path: str | Path,
    active_contract_certificate_path: str | Path,
    payoff_product_dir: str | Path,
    capacity_config_path: str | Path,
    capacity_product_dir: str | Path | None = None,
    session_quarantine_path: str | Path | None = None,
) -> KPRPrearmedL2Result:
    """Build the authenticated pre-armed KPR screening payload."""

    episodes_source = Path(episodes_path).expanduser().resolve()
    map_source = Path(active_contract_map_path).expanduser().resolve()
    certificate_source = Path(active_contract_certificate_path).expanduser().resolve()
    episodes_sha256 = sha256_file(episodes_source)
    active_map_sha256 = sha256_file(map_source)
    active_certificate_sha256 = sha256_file(certificate_source)
    certificate = verify_active_contract_certificate(certificate_source)
    configs = certificate.get("configs")
    data_binding = configs.get("data") if isinstance(configs, Mapping) else None
    if not isinstance(data_binding, Mapping) or not data_binding.get("path"):
        raise KPRPrearmedError("verified active-contract certificate has no data config")
    certified_archive_root = resolve_archive_root(load_yaml(data_binding["path"]))
    supplied_archive_root = Path(archive_root).expanduser().resolve()
    if supplied_archive_root != certified_archive_root:
        raise KPRPrearmedError(
            "archive root does not match the verified active-contract certificate"
        )

    all_episodes = validate_screening_episodes(pd.read_parquet(episodes_source))
    active_map = pd.read_parquet(map_source)
    active_checks = validate_active_contract_inputs(
        all_episodes,
        active_map,
        certificate,
        episodes_sha256=episodes_sha256,
        active_map_sha256=active_map_sha256,
    )
    quarantine, quarantine_binding = load_session_quarantine(
        session_quarantine_path, episodes=all_episodes
    )
    excluded_dates = set(quarantine["session_date"].astype(str))
    episodes = all_episodes.loc[
        ~all_episodes["date"].astype(str).isin(excluded_dates)
    ].reset_index(drop=True)
    if episodes.empty:
        raise KPRPrearmedError("early-close quarantine removed every episode")

    requests = build_proof_requests(episodes)
    health, health_bindings, certified_l2_paths = load_selected_daily_health_manifests(
        supplied_archive_root, requests
    )
    health_checks = validate_health_and_continuity(health, requests)
    observations, source_paths = extract_l2_observations(
        supplied_archive_root,
        requests,
        certified_paths_by_date=certified_l2_paths,
    )
    features = derive_directional_proof_features(requests, observations)

    retained_roots = set(episodes["opportunity_id"].astype(str))
    (
        all_payoffs,
        payoff_observations,
        payoff_certificate,
        payoff_manifest,
        payoff_bindings,
    ) = load_authenticated_payoffs(
        payoff_product_dir,
        episodes_sha256=episodes_sha256,
        active_map_sha256=active_map_sha256,
        active_certificate_sha256=active_certificate_sha256,
    )
    payoffs = all_payoffs.loc[
        all_payoffs["root_id"].astype(str).isin(retained_roots)
    ].reset_index(drop=True)
    capacity_manifest: dict[str, Any] | None = None
    atomic_trade_groups = pd.DataFrame()
    source_l0_paths: tuple[Path, ...] = ()
    if capacity_product_dir is None:
        selected_observation_ids = {
            f"{root_id}::entry::{IMMEDIATE_ACTION_ID}" for root_id in retained_roots
        }
        payoff_observations = payoff_observations.loc[
            payoff_observations["observation_id"].astype(str).isin(
                selected_observation_ids
            )
        ].reset_index(drop=True)
        (
            capacity_rows,
            atomic_trade_groups,
            capacity_bindings,
            source_l0_paths,
        ) = compute_exact_immediate_capacity_v2(
            archive_root=supplied_archive_root,
            observations=payoff_observations,
            payoffs=payoffs,
            payoff_certificate=payoff_certificate,
            payoff_manifest=payoff_manifest,
            payoff_artifact_bindings=payoff_bindings,
            capacity_config_path=capacity_config_path,
        )
    else:
        _validate_capacity_config(capacity_config_path)
        capacity_rows, atomic_trade_groups, capacity_manifest, capacity_bindings = (
            load_authenticated_capacity(
                capacity_product_dir,
                payoff_manifest_sha256=payoff_bindings["run_manifest"]["sha256"],
            )
        )
    capacity = capacity_rows.loc[
        capacity_rows["root_id"].astype(str).isin(retained_roots)
    ].reset_index(drop=True)
    root_labels = build_root_labels(episodes, payoffs, capacity)

    root_window = features.groupby("root_id", sort=False)[
        "proof_window_eligible"
    ].first()
    capacity_eligible = root_labels["label_capacity_v2_eligible_q1"].astype(bool)
    outcome_complete = root_labels["label_outcome_complete"].astype(bool)
    eligible_by_direction = {
        str(int(direction)): int(group["label_capacity_v2_eligible_q1"].astype(bool).sum())
        for direction, group in root_labels.groupby("direction", sort=True)
    }
    latest_clock = max(
        int(features["proof_time_ns"].max()),
        int(root_labels["label_exit_arrival_time_ns"].max()),
    )
    _development_clock_guard([latest_clock], name="latest KPR product clock")
    input_certificate = {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "certificate_type": "KPR_PREARMED_L2_INPUTS",
        "certificate_id": PRODUCT_ID,
        "passed": True,
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "checks": {
            "corrected_roll_gc_lite_v2": active_checks,
            "health_and_continuity": health_checks,
            "four_exact_pre_root_clocks": list(PROOF_CLOCK_IDS),
            "proof_clock_offsets_ns": [offset for _, offset in PROOF_CLOCKS],
            "proof_state_availability": "complete_event_ts_recv_plus_1ms",
            "immediate_action_arrival_offset_ns": FIVE_MILLISECONDS_NS,
            "capacity_quantity": CAPACITY_QUANTITY,
            "capacity_contract_sha256": FROZEN_CAPACITY_V2.content_sha256,
            "payoff_and_capacity_separated_from_features": True,
            "early_close_quarantine_pre_outcome": True,
            "development_end": str(DEVELOPMENT_END),
            "sep_dec_2025_read": False,
            "sealed_2026_read": False,
        },
        "bindings": {
            "episodes": {"path": str(episodes_source), "sha256": episodes_sha256},
            "active_contract_map": {
                "path": str(map_source),
                "sha256": active_map_sha256,
            },
            "active_contract_certificate": {
                "path": str(certificate_source),
                "sha256": active_certificate_sha256,
            },
            "session_quarantine": quarantine_binding,
            "payoff_product": payoff_bindings,
            "capacity_v2": capacity_bindings,
            "selected_daily_health_manifests": health_bindings,
            "selected_daily_health_manifest_set_sha256": sha256_json(
                health_bindings
            ),
            "payoff_product_artifact_set_sha256": payoff_manifest[
                "artifact_set_sha256"
            ],
        },
        "claim_scope": (
            "This immutable development product binds causal pre-armed L2 features "
            "to exact immediate-follow Q=1 eligibility and payoff labels. It makes "
            "no HMADRL, policy, Base-GC superiority, profitability, novelty, locked "
            "validation, final-OOS, or deployment claim."
        ),
    }
    if capacity_manifest is not None:
        input_certificate["bindings"]["capacity_product_artifact_set_sha256"] = (
            capacity_manifest["artifact_set_sha256"]
        )
    report = {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "product_id": PRODUCT_ID,
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "screen_status": "AUTHENTICATED_INPUT_PRODUCT_COMPLETE",
        "performance_conclusion": "NOT_EVALUATED",
        "authenticated_episode_roots": int(len(all_episodes)),
        "early_close_quarantined_sessions": int(len(quarantine)),
        "early_close_quarantined_roots": int(quarantine["quarantined_roots"].sum()),
        "retained_roots": int(len(episodes)),
        "proof_requests": int(len(requests)),
        "expected_proof_requests": int(len(episodes) * len(PROOF_CLOCKS)),
        "proof_state_available_rate": float(features["proof_state_usable"].mean()),
        "proof_window_eligible_roots": int(root_window.sum()),
        "proof_window_ineligible_roots": int((~root_window).sum()),
        "immediate_follow_outcome_complete_roots": int(outcome_complete.sum()),
        "capacity_v2_q1_eligible_roots": int(capacity_eligible.sum()),
        "capacity_v2_q1_eligible_rate": float(capacity_eligible.mean()),
        "capacity_v2_q1_eligible_roots_by_direction": eligible_by_direction,
        "proof_clocks": {
            clock_id: offset for clock_id, offset in PROOF_CLOCKS
        },
        "immediate_action_id": IMMEDIATE_ACTION_ID,
        "immediate_action_arrival_offset_ns": FIVE_MILLISECONDS_NS,
        "feature_table_contains_outcomes": False,
        "root_labels_feature_use_forbidden": True,
        "latest_clock_date": _utc_date_from_ns(latest_clock),
        "sep_dec_2025_read": False,
        "sealed_2026_read": False,
        "warning": input_certificate["claim_scope"],
    }
    return KPRPrearmedL2Result(
        report=report,
        input_certificate=input_certificate,
        proof_requests=requests,
        observations=observations,
        features=features,
        root_labels=root_labels,
        capacity_rows=capacity,
        atomic_trade_groups=atomic_trade_groups,
        session_quarantine=quarantine,
        feature_registry=FEATURE_REGISTRY,
        source_paths=source_paths,
        source_l0_paths=source_l0_paths,
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


def write_kpr_prearmed_l2(
    output_dir: str | Path,
    result: KPRPrearmedL2Result,
) -> dict[str, Any]:
    """Atomically create a new immutable SCREENING_ONLY evidence directory."""

    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(
            f"KPR evidence is create-new-only; choose a new path: {target}"
        )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent)
    )
    try:
        paths = {
            "proof_requests": _atomic_parquet(
                staging / "proof_requests.parquet", result.proof_requests
            ),
            "l2_observations": _atomic_parquet(
                staging / "l2_observations.parquet", result.observations
            ),
            "prearmed_features": _atomic_parquet(
                staging / "prearmed_features.parquet", result.features
            ),
            "root_labels": _atomic_parquet(
                staging / "root_labels.parquet", result.root_labels
            ),
            "capacity_rows": _atomic_parquet(
                staging / "capacity_rows.parquet", result.capacity_rows
            ),
            "atomic_trade_groups": _atomic_parquet(
                staging / "atomic_trade_groups.parquet",
                result.atomic_trade_groups,
            ),
            "session_quarantine": _atomic_parquet(
                staging / "session_quarantine.parquet", result.session_quarantine
            ),
            "feature_registry": write_json(
                staging / "feature_registry.json", result.feature_registry
            ),
            "input_certificate": write_json(
                staging / "input_certificate.json", result.input_certificate
            ),
            "screening_report": write_json(
                staging / "SCREENING_ONLY.json", result.report
            ),
        }
        artifact_bindings = {
            name: {"path": str(target / path.name), "sha256": sha256_file(path)}
            for name, path in paths.items()
        }
        source_bindings = [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in result.source_paths
        ]
        source_l0_bindings = [
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in result.source_l0_paths
        ]
        manifest = {
            "schema_version": PRODUCT_SCHEMA_VERSION,
            "product_id": PRODUCT_ID,
            "evidence_class": SCREENING_LABEL,
            "split_class": DEVELOPMENT_LABEL,
            "artifacts": artifact_bindings,
            "artifact_set_sha256": sha256_json(artifact_bindings),
            "source_l2_partitions": source_bindings,
            "source_l2_partition_set_sha256": sha256_json(source_bindings),
            "source_l0_partitions": source_l0_bindings,
            "source_l0_partition_set_sha256": sha256_json(source_l0_bindings),
            "sealed_2026_read": False,
            "sep_dec_2025_read": False,
            "claim_scope": result.input_certificate["claim_scope"],
        }
        staging_manifest = write_json(staging / "run_manifest.json", manifest)
        if target.exists():
            raise FileExistsError(
                f"KPR evidence appeared during staging; refusing overwrite: {target}"
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


def _verified_source_paths(
    manifest: Mapping[str, Any],
    *,
    ledger_name: str,
    set_hash_name: str,
    verify_source_bytes: bool,
) -> tuple[Path, ...]:
    ledger = manifest.get(ledger_name)
    if not isinstance(ledger, list):
        raise KPRPrearmedError(f"KPR manifest has no {ledger_name} ledger")
    if sha256_json(ledger) != str(manifest.get(set_hash_name, "")):
        raise KPRPrearmedError(f"KPR manifest {ledger_name} set hash mismatch")
    paths: list[Path] = []
    for item in ledger:
        if not isinstance(item, Mapping):
            raise KPRPrearmedError(f"KPR manifest {ledger_name} row is malformed")
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        digest = str(item.get("sha256", ""))
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise KPRPrearmedError(f"KPR manifest {ledger_name} has an invalid SHA-256")
        if verify_source_bytes:
            if not path.is_file():
                raise FileNotFoundError(f"KPR source artifact is missing: {path}")
            if int(item.get("size_bytes", -1)) != int(path.stat().st_size):
                raise KPRPrearmedError(f"KPR source size changed: {path}")
            if sha256_file(path) != digest:
                raise KPRPrearmedError(f"KPR source hash changed: {path}")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise KPRPrearmedError(f"KPR manifest {ledger_name} repeats a path")
    return tuple(paths)


def load_kpr_prearmed_l2(
    product_dir: str | Path,
    *,
    verify_source_bytes: bool = False,
) -> KPRPrearmedL2Result:
    """Typed, fail-closed loader for every downstream empirical consumer.

    Output artifact bytes are always re-hashed. ``verify_source_bytes=True``
    additionally re-hashes every bound raw L2/L0 partition; the default still
    validates their immutable source ledgers and set hashes without repeatedly
    scanning the multi-gigabyte archive.
    """

    manifest, paths, _ = _verify_product_directory(
        product_dir,
        required_artifacts={
            "proof_requests": "proof_requests.parquet",
            "l2_observations": "l2_observations.parquet",
            "prearmed_features": "prearmed_features.parquet",
            "root_labels": "root_labels.parquet",
            "capacity_rows": "capacity_rows.parquet",
            "atomic_trade_groups": "atomic_trade_groups.parquet",
            "session_quarantine": "session_quarantine.parquet",
            "feature_registry": "feature_registry.json",
            "input_certificate": "input_certificate.json",
            "screening_report": "SCREENING_ONLY.json",
        },
        name="KPR pre-armed product",
    )
    if str(manifest.get("product_id", "")) != PRODUCT_ID:
        raise KPRPrearmedError("KPR manifest has the wrong product ID")
    certificate = _load_json_object(
        paths["input_certificate"], name="KPR input certificate"
    )
    if certificate.get("passed") is not True:
        raise KPRPrearmedError("KPR input certificate did not pass")
    if str(certificate.get("certificate_type", "")) != "KPR_PREARMED_L2_INPUTS":
        raise KPRPrearmedError("KPR input certificate has the wrong type")
    if str(certificate.get("certificate_id", "")) != PRODUCT_ID:
        raise KPRPrearmedError("KPR input certificate has the wrong ID")
    checks = certificate.get("checks")
    if not isinstance(checks, Mapping):
        raise KPRPrearmedError("KPR input certificate has no checks mapping")
    required_true = (
        "payoff_and_capacity_separated_from_features",
        "early_close_quarantine_pre_outcome",
    )
    if any(checks.get(name) is not True for name in required_true):
        raise KPRPrearmedError("KPR certificate lacks a required causal separation check")
    if checks.get("sep_dec_2025_read") is not False:
        raise PermissionError("KPR certificate opened Sep-Dec 2025")
    if checks.get("sealed_2026_read") is not False:
        raise PermissionError("KPR certificate opened sealed 2026")
    registry = _load_json_object(paths["feature_registry"], name="KPR feature registry")
    if sha256_json(registry) != sha256_json(FEATURE_REGISTRY):
        raise KPRPrearmedError("KPR feature registry changed")
    report = _load_json_object(paths["screening_report"], name="KPR screening report")
    if report.get("sep_dec_2025_read") is not False:
        raise PermissionError("KPR screening report opened Sep-Dec 2025")
    if report.get("sealed_2026_read") is not False:
        raise PermissionError("KPR screening report opened sealed 2026")

    proof_requests = pd.read_parquet(paths["proof_requests"])
    observations = pd.read_parquet(paths["l2_observations"])
    features = pd.read_parquet(paths["prearmed_features"])
    root_labels = pd.read_parquet(paths["root_labels"])
    capacity_rows = pd.read_parquet(paths["capacity_rows"])
    atomic_trade_groups = pd.read_parquet(paths["atomic_trade_groups"])
    session_quarantine = pd.read_parquet(paths["session_quarantine"])
    _require_columns(
        proof_requests,
        {
            "root_id",
            "observation_id",
            "proof_step",
            "proof_clock_id",
            "proof_offset_ns",
            "proof_time_ns",
            "decision_time_ns",
        },
        name="loaded proof requests",
    )
    if proof_requests["observation_id"].astype(str).duplicated().any():
        raise KPRPrearmedError("loaded proof requests repeat observation IDs")
    expected_clock_map = dict(PROOF_CLOCKS)
    if not proof_requests["proof_clock_id"].astype(str).map(expected_clock_map).eq(
        proof_requests["proof_offset_ns"].astype("int64")
    ).all():
        raise KPRPrearmedError("loaded proof requests changed a frozen clock")
    if not proof_requests["proof_time_ns"].astype("int64").eq(
        proof_requests["decision_time_ns"].astype("int64")
        + proof_requests["proof_offset_ns"].astype("int64")
    ).all():
        raise KPRPrearmedError("loaded proof request clocks are not root-relative")
    request_counts = proof_requests.groupby("root_id")["proof_step"].agg(list)
    if any(sorted(values) != list(range(len(PROOF_CLOCKS))) for values in request_counts):
        raise KPRPrearmedError("loaded proof lattice is not exactly four clocks per root")
    _require_columns(
        features,
        {
            "root_id",
            "observation_id",
            "proof_step",
            "proof_state_usable",
            "proof_window_eligible",
            "directional_support_top3_depth",
            "directional_opposite_top3_depth",
            "directional_top3_imbalance",
            "spread_ticks",
            "state_global_record_ordinal",
        },
        name="loaded KPR features",
    )
    if len(features) != len(proof_requests):
        raise KPRPrearmedError("loaded KPR feature/request row counts differ")
    if set(features["observation_id"].astype(str)) != set(
        proof_requests["observation_id"].astype(str)
    ):
        raise KPRPrearmedError("loaded KPR feature/request observation sets differ")
    if any(column.startswith("label_") for column in features.columns):
        raise KPRPrearmedError("loaded KPR feature table contains outcome labels")
    recomputed_features = derive_directional_proof_features(
        proof_requests, observations
    )
    loaded_canonical = features.sort_values(
        ["proof_time_ns", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)
    if list(loaded_canonical.columns) != list(recomputed_features.columns):
        raise KPRPrearmedError(
            "loaded KPR feature schema differs from deterministic recomputation"
        )
    try:
        pd.testing.assert_frame_equal(
            loaded_canonical,
            recomputed_features,
            check_dtype=False,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as error:
        raise KPRPrearmedError(
            "loaded KPR features differ from deterministic causal recomputation"
        ) from error
    _require_columns(
        root_labels,
        {
            "root_id",
            "label_action_id",
            "label_entry_arrival_time_ns",
            "label_exit_arrival_time_ns",
            "label_outcome_complete",
            "label_stress_payoff_usd",
            "label_capacity_v2_eligible_q1",
            "label_capacity_v2_config_sha256",
            "feature_use_forbidden",
        },
        name="loaded KPR root labels",
    )
    if root_labels["root_id"].astype(str).duplicated().any():
        raise KPRPrearmedError("loaded KPR labels repeat roots")
    feature_roots = set(features["root_id"].astype(str))
    if set(root_labels["root_id"].astype(str)) != feature_roots:
        raise KPRPrearmedError("loaded KPR labels do not cover the exact feature roots")
    if set(root_labels["label_action_id"].astype(str)) != {IMMEDIATE_ACTION_ID}:
        raise KPRPrearmedError("loaded KPR labels changed the immediate action")
    if not root_labels["feature_use_forbidden"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).all():
        raise KPRPrearmedError("loaded KPR labels are not feature-use forbidden")
    if set(root_labels["label_capacity_v2_config_sha256"].astype(str)) != {
        FROZEN_CAPACITY_V2.content_sha256
    }:
        raise KPRPrearmedError("loaded KPR labels changed the capacity-v2 contract")
    _require_columns(
        capacity_rows,
        {
            "root_id",
            "action_id",
            "arrival_time_ns",
            "quantity",
            "feasible",
            "capacity_config_sha256",
        },
        name="loaded exact capacity rows",
    )
    if capacity_rows["root_id"].astype(str).duplicated().any():
        raise KPRPrearmedError("loaded exact capacity rows repeat roots")
    if set(capacity_rows["root_id"].astype(str)) != feature_roots:
        raise KPRPrearmedError("loaded exact capacity rows do not cover feature roots")
    if set(capacity_rows["action_id"].astype(str)) != {IMMEDIATE_ACTION_ID}:
        raise KPRPrearmedError("loaded capacity rows changed the immediate action")
    if set(capacity_rows["quantity"].astype(int)) != {CAPACITY_QUANTITY}:
        raise KPRPrearmedError("loaded capacity rows escaped Q=1")
    label_capacity = root_labels.set_index("root_id")[
        "label_capacity_v2_eligible_q1"
    ].astype(bool)
    row_capacity = capacity_rows.set_index("root_id")["feasible"].astype(bool)
    if not label_capacity.sort_index().equals(row_capacity.sort_index()):
        raise KPRPrearmedError("loaded label/capacity eligibility differs")
    _development_clock_guard(
        proof_requests["proof_time_ns"], name="loaded KPR proof clocks"
    )
    _development_clock_guard(
        root_labels["label_exit_arrival_time_ns"], name="loaded KPR outcome clocks"
    )
    source_paths = _verified_source_paths(
        manifest,
        ledger_name="source_l2_partitions",
        set_hash_name="source_l2_partition_set_sha256",
        verify_source_bytes=verify_source_bytes,
    )
    source_l0_paths = _verified_source_paths(
        manifest,
        ledger_name="source_l0_partitions",
        set_hash_name="source_l0_partition_set_sha256",
        verify_source_bytes=verify_source_bytes,
    )
    return KPRPrearmedL2Result(
        report=report,
        input_certificate=certificate,
        proof_requests=proof_requests,
        observations=observations,
        features=features,
        root_labels=root_labels,
        capacity_rows=capacity_rows,
        atomic_trade_groups=atomic_trade_groups,
        session_quarantine=session_quarantine,
        feature_registry=registry,
        source_paths=source_paths,
        source_l0_paths=source_l0_paths,
    )


def build_pilot04_prearmed_frame(product: KPRPrearmedL2Result) -> pd.DataFrame:
    """Adapt the typed product to pilot04's immutable four-clock frame contract.

    The proof inputs are derived exclusively from the causal feature table.
    Economic and capacity fields enter only after that derivation and retain
    their explicit label/filter roles.
    """

    features = product.features.sort_values(
        ["root_id", "proof_step"], kind="mergesort"
    ).copy()
    labels = product.root_labels.copy()
    if any(column.startswith("label_") for column in features.columns):
        raise KPRPrearmedError("pilot adapter received leaked proof features")
    valid_roots = set(
        features.groupby("root_id")["proof_window_eligible"]
        .first()
        .loc[lambda values: values.astype(bool)]
        .index.astype(str)
    )
    complete_roots = set(
        labels.loc[
            labels["label_outcome_complete"].astype(bool), "root_id"
        ].astype(str)
    )
    all_roots = set(labels["root_id"].astype(str))
    if valid_roots != all_roots:
        missing = sorted(all_roots.difference(valid_roots))
        raise KPRPrearmedError(
            "pilot04 requires every frozen root to have a valid proof window; "
            f"ineligible roots cannot be silently filtered: {missing[:5]}"
        )
    if complete_roots != all_roots:
        missing = sorted(all_roots.difference(complete_roots))
        raise KPRPrearmedError(
            "pilot04 requires every frozen root to have a complete outcome; "
            f"incomplete roots cannot be silently filtered: {missing[:5]}"
        )
    support = pd.to_numeric(
        features["directional_support_top3_depth"], errors="coerce"
    )
    opposite = pd.to_numeric(
        features["directional_opposite_top3_depth"], errors="coerce"
    )
    if support.isna().any() or opposite.isna().any():
        raise KPRPrearmedError("pilot04 adapter found missing proof depth")
    if bool((support < 0).any() or (opposite < 0).any()):
        raise KPRPrearmedError("pilot04 adapter found negative proof depth")
    # Persistent quality is the supporting quantity that survived every proof
    # landmark observed so far. It is a transparent L2 proxy, not an L3 order-
    # lifecycle claim; a future L3 stage must replace it explicitly.
    features["_supportive_dose"] = support.astype(float)
    features["_lifecycle_quality"] = features.groupby("root_id", sort=False)[
        "_supportive_dose"
    ].cummin()
    features["_contradiction_dose"] = np.maximum(
        opposite.to_numpy(dtype=float) - support.to_numpy(dtype=float), 0.0
    )
    features["_slow_context"] = pd.to_numeric(
        features["spread_ticks"], errors="coerce"
    ).astype(float)
    if not np.isfinite(
        features[
            [
                "_supportive_dose",
                "_lifecycle_quality",
                "_contradiction_dose",
                "_slow_context",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise KPRPrearmedError("pilot04 adapter produced non-finite proof inputs")
    label_index = labels.set_index(labels["root_id"].astype(str), verify_integrity=True)
    root_ids = features["root_id"].astype(str)
    matched = label_index.loc[root_ids]
    capacity_eligible = matched["label_capacity_v2_eligible_q1"].astype(bool)
    effective_base_payoff = np.where(
        capacity_eligible.to_numpy(),
        matched["label_stress_payoff_usd"].astype(float).to_numpy(),
        0.0,
    )
    state_ordinals = pd.to_numeric(
        features["state_global_record_ordinal"], errors="coerce"
    )
    if state_ordinals.isna().any() or bool((state_ordinals < 0).any()):
        raise KPRPrearmedError("pilot04 adapter lacks causal tape ordinals")
    output = pd.DataFrame(
        {
            "opportunity_id": root_ids.to_numpy(),
            "date": matched["session_date"].astype(str).to_numpy(),
            "clock_index": features["proof_step"].astype("int64").to_numpy(),
            "feature_time_ns": features["proof_time_ns"].astype("int64").to_numpy(),
            "available_time_ns": features["proof_time_ns"].astype("int64").to_numpy(),
            "decision_time_ns": features["decision_time_ns"].astype("int64").to_numpy(),
            "episode_end_time_ns": matched[
                "label_exit_arrival_time_ns"
            ].astype("int64").to_numpy(),
            "direction": features["direction"].astype("int64").to_numpy(),
            "raw_symbol": features["raw_symbol"].astype(str).to_numpy(),
            "continuity_segment_id": features[
                "continuity_segment_id"
            ].astype("int64").to_numpy(),
            "tape_last_ordinal": state_ordinals.astype("int64").to_numpy(),
            "supportive_dose": features["_supportive_dose"].to_numpy(float),
            "lifecycle_quality": features["_lifecycle_quality"].to_numpy(float),
            "contradiction_dose": features["_contradiction_dose"].to_numpy(float),
            "slow_context": features["_slow_context"].to_numpy(float),
            "eligible": capacity_eligible.to_numpy(),
            # A capacity-ineligible candidate cannot take exposure. The raw
            # counterfactual follow payoff remains only in root_labels, whose
            # feature_use_forbidden flag is authenticated by the typed loader.
            "base_follow_net_profit": effective_base_payoff,
            "veto_net_profit": 0.0,
            "causal_replay_valid": True,
            "capacity_state_valid": True,
            "branch_replay_valid": True,
        }
    )
    if output.groupby("opportunity_id")["clock_index"].size().ne(
        len(PROOF_CLOCKS)
    ).any():
        raise AssertionError("pilot04 adapter changed the four-clock lattice")
    return output.sort_values(
        ["opportunity_id", "clock_index"], kind="mergesort"
    ).reset_index(drop=True)


def load_pilot04_prearmed_frame(
    product_dir: str | Path,
    *,
    verify_source_bytes: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Authenticate a KPR product and return pilot04's exact input frame."""

    product = load_kpr_prearmed_l2(
        product_dir, verify_source_bytes=verify_source_bytes
    )
    frame = build_pilot04_prearmed_frame(product)
    manifest_path = Path(product_dir).expanduser().resolve() / "run_manifest.json"
    provenance = {
        "product_id": PRODUCT_ID,
        "product_dir": str(Path(product_dir).expanduser().resolve()),
        "run_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "rows": int(len(frame)),
        "roots": int(frame["opportunity_id"].nunique()),
        "proof_input_layer": "causal_complete_event_L2",
        "quality_proxy": "running_min_directional_support_top3_depth",
        "label_payoff": "adverse_one_tick_per_filled_side_stress_usd",
        "capacity_estimand": "frozen_capacity_v2_Q1_at_root_plus_5ms",
        "generic_claim_boolean_authorization_used": False,
    }
    return frame, provenance


__all__ = [
    "CAPACITY_QUANTITY",
    "FEATURE_REGISTRY",
    "IMMEDIATE_ACTION_ID",
    "KPRPrearmedError",
    "KPRPrearmedL2Result",
    "PRODUCT_ID",
    "PROOF_CLOCKS",
    "PROOF_CLOCK_IDS",
    "build_proof_requests",
    "build_immediate_capacity_requests",
    "build_pilot04_prearmed_frame",
    "build_root_labels",
    "compute_exact_immediate_capacity_v2",
    "derive_directional_proof_features",
    "load_authenticated_capacity",
    "load_authenticated_payoffs",
    "load_kpr_prearmed_l2",
    "load_pilot04_prearmed_frame",
    "load_session_quarantine",
    "run_kpr_prearmed_l2_from_paths",
    "resolve_development_manifest_declared_l0",
    "write_kpr_prearmed_l2",
]
