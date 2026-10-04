"""Fail-closed May evaluation for the corrected paper-described comparator.

This module is deliberately isolated from the frozen V6 implementation and
from :mod:`volume_v6_basepaper_exact`.  It adds only the post-training access
receipt and deterministic evaluation path required by the comparator
correction.  June--August 2026 is not an accepted input to any API here.

The access order is part of the contract: the canonical correction freeze,
the complete five-seed exact training chain, the exact May source manifest,
the immutable historical V6 report manifest, and both source-code hashes are
verified before a May market-data or report child is opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from kx_base_gc.contract import ACTIONS, AGENT_ORDER, AgentSpec, registered_contract

from .environment import ExecutionEconomics
from . import volume_v6_basepaper_exact as exact
from .volume_v6_contract import SEEDS
from .volume_v6_eval import (
    atomic_json,
    daily_metrics,
    expected_economics,
    file_sha256,
    flat_on_vote_tie,
    frozen_protocol,
    paired_improvement,
)


SCENARIOS = tuple(exact.SCENARIOS)
MAY_START = "2026-05-01"
MAY_END = "2026-05-31"
MAY_END_EXCLUSIVE = "2026-06-01"
CORRECTION_STRATEGY_VERSION = exact.PAPER_PROTOCOL_VERSION
EXACT_TRAINING_STAGE = exact.PAPER_TRAINING_STAGE
EXACT_INITIALIZATION = exact.PAPER_INITIALIZATION
EVALUATION_INITIALIZATION = (
    "May_boundary_FLAT_policy_state_with_pre_May_OHLCV_lookback_only_v1"
)
CORRECTION_CLASSIFICATION = "DEVELOPMENT_CONTAMINATED"
FREEZE_ARTIFACT = "BASEPAPER_DESCRIBED_FREEZE_RECEIPT"
ACCESS_ARTIFACT = "BASEPAPER_DESCRIBED_MAY_ACCESS_RECEIPT"
EVALUATION_ARTIFACT = "BASEPAPER_DESCRIBED_MAY_EVALUATION"
REPORT_ARTIFACT = "BASEPAPER_DESCRIBED_COMMON_LEDGER_REPLAY"
EXPECTED_MAY_DATASET_SHA256 = exact.EXPECTED_GC_MANIFEST_SHA256["validation"]
EXPECTED_V6_MAY_EVALUATION_SHA256 = (
    "b9dc8928ae24e8e1c44130dee175819cf2456ccd2c09aa7786f76ac2ba137c71"
)
V6_COMPARISON_ARMS = ("V6-RichVolume", "V6-BaseOHLCV")

PILOT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PILOT_ROOT / "contracts" / "BASEPAPER_DESCRIBED_CORRECTION_V1_PROTOCOL.json"
DEFAULT_FREEZE_RECEIPT = (
    PILOT_ROOT / "contracts" / "BASEPAPER_DESCRIBED_CORRECTION_V1_FREEZE_RECEIPT.json"
)


def canonical_json_sha256(value: object) -> str:
    """Return the validator-style digest of parsed canonical JSON."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _resolve(anchor: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _manifest_path(value: Path, *, label: str) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{label} manifest is absent: {path}")
    return path


def _record(path: Path, *, digest: str | None = None) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved) if digest is None else str(digest),
    }


def _bound_file(
    receipt_path: Path, value: object, *, label: str,
) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise PermissionError(f"May access receipt lacks {label} binding")
    digest = value.get("sha256")
    if not _is_sha256(digest):
        raise PermissionError(f"May access receipt has invalid {label} hash")
    path = _resolve(receipt_path.parent, value.get("path"))
    if not path.is_file() or file_sha256(path) != digest:
        raise PermissionError(f"May access receipt {label} is absent or changed")
    return path, str(digest)


def _assert_may_dates(days: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(str(day) for day in days)
    if not normalized or normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{label} dates must be nonempty, unique, and sorted")
    for day in normalized:
        try:
            parsed = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError as error:
            raise ValueError(f"{label} contains a noncanonical date") from error
        if parsed.isoformat() != day:
            raise ValueError(f"{label} contains a noncanonical date")
        if not MAY_START <= day <= MAY_END:
            raise PermissionError(
                "corrected BasePaper evaluation hard-blocks every date after "
                "2026-05-31 and every date outside May 2026"
            )
    return normalized


def verify_corrected_freeze(
    protocol_path: Path = DEFAULT_PROTOCOL,
    freeze_receipt_path: Path = DEFAULT_FREEZE_RECEIPT,
) -> tuple[dict[str, object], str]:
    """Authenticate the parsed canonical correction protocol and PASS receipt."""

    protocol_path = Path(protocol_path).resolve()
    receipt_path = Path(freeze_receipt_path).resolve()
    if not protocol_path.is_file() or not receipt_path.is_file():
        raise PermissionError("May access requires the corrected protocol freeze receipt")
    protocol = _read_object(protocol_path)
    receipt = _read_object(receipt_path)
    digest = canonical_json_sha256(protocol)
    bound_protocol = _resolve(receipt_path.parent, receipt.get("protocol_path"))
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != FREEZE_ARTIFACT
        or receipt.get("validator_result") != "PASS"
        or receipt.get("protocol_sha256") != digest
        or bound_protocol != protocol_path
        or not isinstance(receipt.get("created_at_utc"), str)
        or protocol.get("strategy_version") != CORRECTION_STRATEGY_VERSION
    ):
        raise PermissionError("corrected BasePaper protocol is not canonically frozen")
    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise PermissionError("corrected BasePaper protocol lacks frozen artifacts")
    verified: dict[str, Path] = {}
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise PermissionError(f"corrected BasePaper artifact {name} is malformed")
        declared = record.get("sha256")
        if not _is_sha256(declared):
            raise PermissionError(f"corrected BasePaper artifact {name} hash is invalid")
        path = _resolve(protocol_path.parent, record.get("path"))
        if not path.is_file() or file_sha256(path) != declared:
            raise PermissionError(
                f"corrected BasePaper artifact {name} is absent or changed"
            )
        verified[str(name)] = path
    required_code = {
        "trainer_code": Path(exact.__file__).resolve(),
        "evaluator_code": Path(__file__).resolve(),
    }
    for name, expected in required_code.items():
        if verified.get(name) != expected:
            raise PermissionError(
                f"corrected BasePaper protocol does not bind the current {name}"
            )
    return protocol, digest


def _source_identity_manifest(
    identity: Mapping[str, object], *, anchor: Path, label: str,
) -> tuple[Path, str]:
    declared = identity.get("manifest_sha256")
    if not _is_sha256(declared):
        raise ValueError(f"{label} source identity lacks a SHA-256")
    source = _resolve(anchor, identity.get("path"))
    manifest = source / "manifest.json" if source.is_dir() else source
    if not manifest.is_file() or file_sha256(manifest) != declared:
        raise ValueError(f"{label} source manifest is absent or changed")
    return manifest.resolve(), str(declared)


def _validate_training_source_outer(path: Path) -> dict[str, object]:
    """Validate the exact prepared-GC source without opening a data child."""

    value = _read_object(path)
    expected_raw = exact.EXPECTED_GC_MANIFEST_SHA256["train"]
    dates = tuple(str(day) for day in value.get("dates", []))
    if (
        value.get("artifact_type") != "BASEPAPER_DESCRIBED_PREPARED_BARS"
        or value.get("name") != exact.PREPARED_DATA_NAME
        or value.get("phase") != "train"
        or value.get("complete") is not True
        or value.get("smoke_only") is not False
        or value.get("source_manifest_sha256") != expected_raw
        or canonical_json_sha256(list(dates))
        != exact.EXPECTED_PHASE_DATE_SHA256["train"]
        or any(day > "2026-04-30" for day in dates)
        or not _is_sha256(value.get("source_digest"))
        or not _is_sha256(value.get("source_records_sha256"))
    ):
        raise ValueError("full exact prepared-GC source identity differs")
    receipt_path = path.parent / "manifest.receipt.json"
    if (
        not receipt_path.is_file()
        or _read_object(receipt_path).get("manifest_sha256") != file_sha256(path)
    ):
        raise ValueError("full exact prepared-GC source receipt differs")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(AGENT_ORDER):
        raise ValueError("full exact prepared-GC product inventory differs")
    by_agent = {
        str(record.get("agent")): record
        for record in artifacts if isinstance(record, dict)
    }
    if set(by_agent) != set(AGENT_ORDER):
        raise ValueError("full exact prepared-GC product inventory differs")
    for name in AGENT_ORDER:
        record = by_agent[name]
        product = _resolve(path.parent, record.get("path"))
        try:
            product.relative_to(path.parent.resolve())
        except ValueError as error:
            raise ValueError("full exact prepared-GC product escapes its root") from error
        if (
            not _is_sha256(record.get("sha256"))
            or not product.is_file()
            or file_sha256(product) != record.get("sha256")
            or type(record.get("rows")) is not int
            or int(record["rows"]) != exact.EXPECTED_GC_TRAIN_BARS[name]
        ):
            raise ValueError(f"full exact prepared-GC {name} product differs")
    return value


@dataclass(frozen=True)
class ExactTrainingBinding:
    manifest_path: Path
    manifest_sha256: str
    value: Mapping[str, object]
    training_source_manifest: Path
    training_source_manifest_sha256: str


def _member_values(
    root: Path, value: Mapping[str, object], *, stage: str,
) -> dict[int, tuple[Path, dict[str, object]]]:
    rows = value.get("members")
    if not isinstance(rows, list):
        raise ValueError("exact ensemble member inventory is absent")
    result: dict[int, tuple[Path, dict[str, object]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("exact ensemble member record is malformed")
        seed = int(row.get("seed", -1))
        path = _resolve(root, row.get("path"))
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("exact ensemble member escapes its root") from error
        if path.name != "manifest.json" or file_sha256(path) != row.get("sha256"):
            raise ValueError("exact ensemble member manifest hash differs")
        member = exact.verify_training_artifact(path.parent, seed=seed, stage=stage)
        result[seed] = (path, member)
    if tuple(sorted(result)) != tuple(SEEDS):
        raise ValueError(f"exact ensemble requires seeds {SEEDS}")
    return result


def _assert_full_member(
    member: Mapping[str, object], *, trainer_sha256: str,
) -> None:
    request = member.get("request")
    if (
        not isinstance(request, dict)
        or request.get("code_sha256") != trainer_sha256
        or request.get("effective_contract_sha256")
        != canonical_json_sha256(exact.paper_effective_contract())
        or request.get("initialization") != EXACT_INITIALIZATION
        or "parent_manifest_sha256" not in request
        or request.get("parent_manifest_sha256") is not None
        or member.get("stage") != EXACT_TRAINING_STAGE
        or member.get("initialization") != EXACT_INITIALIZATION
        or "parent_manifest_sha256" not in member
        or member.get("parent_manifest_sha256") is not None
        or member.get("source_manifest_sha256")
        != exact.EXPECTED_GC_MANIFEST_SHA256["train"]
        or member.get("smoke_only") is not False
        or member.get("maximum_events_smoke_limit") is not None
        or member.get("passes") != 1
        or "pretrained_ensemble" in request
        or "weights_only_xau_to_gc_reset" in member
        or "weights_only_xau_to_gc_reset" in request
    ):
        raise ValueError(
            "exact GC-from-scratch member has a parent, transfer state, smoke cap, "
            "or stale contract/code"
        )
    dates = tuple(str(day) for day in member.get("dates", []))
    if (
        canonical_json_sha256(list(dates))
        != exact.EXPECTED_PHASE_DATE_SHA256["train"]
        or any(day > "2026-04-30" for day in dates)
    ):
        raise ValueError("exact GC-from-scratch member coverage differs")


def verify_exact_training_chain(training_manifest: Path) -> ExactTrainingBinding:
    """Deep-verify five full GC-from-scratch seeds, source, and current code."""

    manifest_path = _manifest_path(Path(training_manifest), label="exact training")
    root = manifest_path.parent
    value = exact.verify_ensemble_training(root, stage=EXACT_TRAINING_STAGE)
    trainer_path = Path(exact.__file__).resolve()
    trainer_sha = file_sha256(trainer_path)
    request = value.get("request")
    if (
        not isinstance(request, dict)
        or value.get("maximum_events_smoke_limit") is not None
        or request.get("maximum_events_smoke_limit") is not None
        or request.get("code_sha256") != trainer_sha
        or request.get("effective_contract_sha256")
        != canonical_json_sha256(exact.paper_effective_contract())
        or request.get("initialization") != EXACT_INITIALIZATION
        or "pretrained_ensemble" not in request
        or request.get("pretrained_ensemble") is not None
        or value.get("initialization") != EXACT_INITIALIZATION
        or "pretrained_ensemble" not in value
        or value.get("pretrained_ensemble") is not None
        or "weights_only_xau_to_gc_reset" in request
        or "weights_only_xau_to_gc_reset" in value
    ):
        raise ValueError(
            "exact GC ensemble is not full, current-code, parentless, and from scratch"
        )
    source_identity = request.get("source")
    if not isinstance(source_identity, dict):
        raise ValueError("exact GC-from-scratch ensemble lacks its source identity")
    source_path, source_sha = _source_identity_manifest(
        source_identity, anchor=root, label="GC training",
    )
    source_value = _validate_training_source_outer(source_path)
    members = _member_values(root, value, stage=EXACT_TRAINING_STAGE)
    for _, member in members.values():
        _assert_full_member(member, trainer_sha256=trainer_sha)
    gc_source_digests = {str(member.get("source_digest")) for _, member in members.values()}
    input_record_digests = {
        str(member.get("input_records_sha256")) for _, member in members.values()
    }
    expected_input_records = canonical_json_sha256([
        {"prepared_manifest_sha256": source_sha},
    ])
    if (
        gc_source_digests != {str(source_value.get("source_digest"))}
        or input_record_digests != {expected_input_records}
    ):
        raise ValueError("exact GC ensemble members differ from their bound source digest")
    return ExactTrainingBinding(
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        value=value,
        training_source_manifest=source_path,
        training_source_manifest_sha256=source_sha,
    )


def _verify_may_dataset_outer(
    path: Path, *, protocol_path: Path, freeze_receipt_path: Path,
) -> tuple[dict[str, object], tuple[str, ...]]:
    path = _manifest_path(path, label="May dataset")
    if file_sha256(path) != EXPECTED_MAY_DATASET_SHA256:
        raise ValueError("May source is not the exact frozen dataset manifest")
    # This verifier parses only the outer manifest; it does not open a source
    # Parquet, session NPZ, sidecar, or report child.
    value, dates = exact._checked_gc_manifest(
        path, phase="validation", protocol=protocol_path,
        freeze_receipt=freeze_receipt_path,
    )
    _assert_may_dates(dates, label="May source")
    return value, dates


def _verify_v6_outer(path: Path) -> dict[str, object]:
    path = _manifest_path(path, label="immutable V6 May evaluation")
    if file_sha256(path) != EXPECTED_V6_MAY_EVALUATION_SHA256:
        raise ValueError("immutable V6 May evaluation manifest hash differs")
    value = _read_object(path)
    dataset_record = value.get("evaluation_dataset_manifest")
    if (
        not isinstance(dataset_record, dict)
        or value.get("artifact_type") != "V6_COMPLETE_REGISTERED_EVALUATION"
        or value.get("phase") != "validation"
        or value.get("classification") != "DEVELOPMENT"
        or value.get("holdout_read") is not False
        or value.get("complete") is not True
        or dataset_record.get("sha256") != EXPECTED_MAY_DATASET_SHA256
    ):
        raise ValueError("immutable V6 May evaluation identity differs")
    return value


def authorize_may(
    protocol_path: Path,
    freeze_receipt_path: Path,
    training_manifest: Path,
    may_dataset_manifest: Path,
    v6_evaluation_manifest: Path,
    output: Path,
) -> Path:
    """Create the immutable post-training receipt that alone permits May reads."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite May access receipt: {output}")
    protocol_path = Path(protocol_path).resolve()
    freeze_receipt_path = Path(freeze_receipt_path).resolve()
    _, protocol_sha = verify_corrected_freeze(protocol_path, freeze_receipt_path)
    training = verify_exact_training_chain(training_manifest)
    may_path = _manifest_path(may_dataset_manifest, label="May dataset")
    _, dates = _verify_may_dataset_outer(
        may_path, protocol_path=protocol_path,
        freeze_receipt_path=freeze_receipt_path,
    )
    v6_path = _manifest_path(v6_evaluation_manifest, label="immutable V6 May evaluation")
    _verify_v6_outer(v6_path)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z",
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": ACCESS_ARTIFACT,
        "strategy_version": CORRECTION_STRATEGY_VERSION,
        "created_at_utc": timestamp,
        "classification": CORRECTION_CLASSIFICATION,
        "period": [MAY_START, MAY_END],
        "holdout_opened": False,
        "training_stage": EXACT_TRAINING_STAGE,
        "training_initialization": EXACT_INITIALIZATION,
        "pretrained_ensemble": None,
        "protocol": {
            "path": str(protocol_path),
            "canonical_sha256": protocol_sha,
        },
        "protocol_sha256": protocol_sha,
        "freeze_receipt": _record(freeze_receipt_path),
        "exact_gc_training_manifest": _record(
            training.manifest_path, digest=training.manifest_sha256,
        ),
        "exact_training_deep_verifier_result": "PASS",
        "training_source_manifest": _record(
            training.training_source_manifest,
            digest=training.training_source_manifest_sha256,
        ),
        "may_dataset_manifest": _record(may_path),
        "may_dates_sha256": canonical_json_sha256(list(dates)),
        "immutable_v6_evaluation_manifest": _record(v6_path),
        "source_code": {
            "evaluator": _record(Path(__file__)),
            "trainer": _record(Path(exact.__file__)),
        },
        "access_rule": (
            "verify_this_complete_hash_chain_before_any_May_market_data_or_report_child"
        ),
    }
    return atomic_json(output, receipt)


@dataclass(frozen=True)
class MayAccessAuthorization:
    receipt_path: Path
    receipt_sha256: str
    protocol_path: Path
    freeze_receipt_path: Path
    protocol_sha256: str
    training: ExactTrainingBinding
    may_dataset_manifest: Path
    may_dataset_sha256: str
    may_dates: tuple[str, ...]
    v6_evaluation_manifest: Path
    v6_evaluation_manifest_sha256: str


def verify_may_access_receipt(receipt_path: Path) -> MayAccessAuthorization:
    """Recompute the entire pre-May authorization chain without child reads."""

    receipt_path = Path(receipt_path).resolve()
    if not receipt_path.is_file():
        raise PermissionError("May evaluation requires its access receipt")
    value = _read_object(receipt_path)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != ACCESS_ARTIFACT
        or value.get("strategy_version") != CORRECTION_STRATEGY_VERSION
        or value.get("classification") != CORRECTION_CLASSIFICATION
        or value.get("period") != [MAY_START, MAY_END]
        or value.get("holdout_opened") is not False
        or value.get("training_stage") != EXACT_TRAINING_STAGE
        or value.get("training_initialization") != EXACT_INITIALIZATION
        or "pretrained_ensemble" not in value
        or value.get("pretrained_ensemble") is not None
        or value.get("exact_training_deep_verifier_result") != "PASS"
    ):
        raise PermissionError("May access receipt identity or classification differs")
    protocol_record = value.get("protocol")
    if not isinstance(protocol_record, dict) or not _is_sha256(
        protocol_record.get("canonical_sha256")
    ):
        raise PermissionError("May access receipt lacks its canonical protocol binding")
    protocol_path = _resolve(receipt_path.parent, protocol_record.get("path"))
    freeze_path, _ = _bound_file(
        receipt_path, value.get("freeze_receipt"), label="freeze receipt",
    )
    _, protocol_sha = verify_corrected_freeze(protocol_path, freeze_path)
    if (
        protocol_sha != protocol_record.get("canonical_sha256")
        or value.get("protocol_sha256") != protocol_sha
    ):
        raise PermissionError("May access receipt canonical protocol hash differs")

    training_path, training_sha = _bound_file(
        receipt_path, value.get("exact_gc_training_manifest"),
        label="exact GC training manifest",
    )
    training = verify_exact_training_chain(training_path)
    if training.manifest_sha256 != training_sha:
        raise PermissionError("May access receipt exact training hash differs")
    source_path, source_sha = _bound_file(
        receipt_path, value.get("training_source_manifest"),
        label="training source manifest",
    )
    if (
        source_path != training.training_source_manifest
        or source_sha != training.training_source_manifest_sha256
    ):
        raise PermissionError("May access receipt training source binding differs")

    may_path, may_sha = _bound_file(
        receipt_path, value.get("may_dataset_manifest"), label="May dataset manifest",
    )
    if may_sha != EXPECTED_MAY_DATASET_SHA256:
        raise PermissionError("May access receipt does not bind the exact May source")
    _, dates = _verify_may_dataset_outer(
        may_path, protocol_path=protocol_path, freeze_receipt_path=freeze_path,
    )
    if value.get("may_dates_sha256") != canonical_json_sha256(list(dates)):
        raise PermissionError("May access receipt date-set hash differs")

    v6_path, v6_sha = _bound_file(
        receipt_path, value.get("immutable_v6_evaluation_manifest"),
        label="immutable V6 evaluation manifest",
    )
    if v6_sha != EXPECTED_V6_MAY_EVALUATION_SHA256:
        raise PermissionError("May access receipt does not bind immutable V6 reports")
    _verify_v6_outer(v6_path)

    source_code = value.get("source_code")
    if not isinstance(source_code, dict):
        raise PermissionError("May access receipt lacks source-code bindings")
    evaluator_path, evaluator_sha = _bound_file(
        receipt_path, source_code.get("evaluator"), label="evaluator source code",
    )
    trainer_path, trainer_sha = _bound_file(
        receipt_path, source_code.get("trainer"), label="trainer source code",
    )
    if (
        evaluator_path != Path(__file__).resolve()
        or evaluator_sha != file_sha256(Path(__file__).resolve())
        or trainer_path != Path(exact.__file__).resolve()
        or trainer_sha != file_sha256(Path(exact.__file__).resolve())
    ):
        raise PermissionError("May access receipt source code is stale or substituted")
    return MayAccessAuthorization(
        receipt_path=receipt_path,
        receipt_sha256=file_sha256(receipt_path),
        protocol_path=protocol_path,
        freeze_receipt_path=freeze_path,
        protocol_sha256=protocol_sha,
        training=training,
        may_dataset_manifest=may_path,
        may_dataset_sha256=may_sha,
        may_dates=dates,
        v6_evaluation_manifest=v6_path,
        v6_evaluation_manifest_sha256=v6_sha,
    )


@dataclass(frozen=True)
class AgentGreedyTrace:
    times_ns: np.ndarray
    actions: np.ndarray
    renewal_mask: np.ndarray


@dataclass(frozen=True)
class JointGreedyTrace:
    agents: Mapping[str, AgentGreedyTrace]


def _trainer_spec(policy: Any) -> AgentSpec:
    spec = getattr(policy, "spec", getattr(policy, "agent", None))
    if not isinstance(spec, AgentSpec):
        raise TypeError("exact inference policy must expose an AgentSpec")
    return spec


def _inference_indexes(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    """Return causal lookback-supported states, including unmatured late states."""

    segments = exact.paper_segments(frame)
    if len(frame) < int(spec.lookback):
        return np.empty(0, dtype=np.int64)
    candidates = np.arange(int(spec.lookback) - 1, len(frame), dtype=np.int64)
    starts = candidates - int(spec.lookback) + 1
    return candidates[segments[starts] == segments[candidates]]


def _greedy(policy: Any, observation: np.ndarray) -> int:
    if callable(getattr(policy, "greedy", None)):
        action = int(policy.greedy(observation))
    else:
        action = int(policy.act(observation, explore=False))
    if action not in ACTIONS:
        raise ValueError("exact greedy policy selected an invalid action")
    return action


def greedy_joint_inference(
    policies: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    *,
    evaluation_start_indexes: Mapping[str, int] | None = None,
) -> JointGreedyTrace:
    """Run one seeded H1/M15/M5 hierarchy in deterministic chronological order.

    The returned trace contains the held target at every eligible completed
    bar.  Only rows marked by ``renewal_mask`` invoke a network.  Publishing
    the held M5 target every bar preserves the paper's row-count action lock
    across ordinary session closures when the common ledger is session-flat.

    Rows before ``evaluation_start_indexes`` may supply raw OHLCV lookback but
    are never processed by a policy.  Consequently every agent's position,
    action, and lock clock starts FLAT at the evaluation boundary.  This is the
    frozen local held-out reconstruction convention; it is not a published
    evaluation rule from the paper.
    """

    if set(policies) != set(AGENT_ORDER) or set(frames) != set(AGENT_ORDER):
        raise ValueError(f"joint exact inference requires exactly {AGENT_ORDER}")
    specs = {name: _trainer_spec(policies[name]) for name in AGENT_ORDER}
    if any(specs[name].name != name for name in AGENT_ORDER):
        raise ValueError("exact inference policy names differ from hierarchy order")
    if evaluation_start_indexes is None:
        starts = {name: 0 for name in AGENT_ORDER}
    else:
        if set(evaluation_start_indexes) != set(AGENT_ORDER):
            raise ValueError(
                f"evaluation starts must be supplied for exactly {AGENT_ORDER}"
            )
        starts = {name: int(evaluation_start_indexes[name]) for name in AGENT_ORDER}
        if any(not 0 <= starts[name] < len(frames[name]) for name in AGENT_ORDER):
            raise ValueError("evaluation start index is outside its agent frame")
    times = {name: exact._event_ns(frames[name]) for name in AGENT_ORDER}
    segments = {name: exact.paper_segments(frames[name]) for name in AGENT_ORDER}
    supported = {
        name: _inference_indexes(frames[name], specs[name]) for name in AGENT_ORDER
    }
    indexes = {
        name: supported[name][supported[name] >= starts[name]] for name in AGENT_ORDER
    }
    eligible = {
        name: np.isin(np.arange(len(frames[name]), dtype=np.int64), indexes[name])
        for name in AGENT_ORDER
    }
    if any(not len(indexes[name]) for name in AGENT_ORDER):
        absent = [name for name in AGENT_ORDER if not len(indexes[name])]
        raise ValueError(f"joint exact inference has zero support for {absent}")

    heap: list[tuple[int, int, int]] = []
    for ordinal, name in enumerate(AGENT_ORDER):
        if starts[name] < len(times[name]):
            heapq.heappush(
                heap, (int(times[name][starts[name]]), ordinal, starts[name]),
            )
    current: dict[str, int | None] = {name: None for name in AGENT_ORDER}
    positions = {name: 0 for name in AGENT_ORDER}
    held = {name: 0 for name in AGENT_ORDER}
    last_segment: dict[str, int | None] = {name: None for name in AGENT_ORDER}
    global_symbol: str | None = None
    trace_times: dict[str, list[int]] = {name: [] for name in AGENT_ORDER}
    trace_actions: dict[str, list[int]] = {name: [] for name in AGENT_ORDER}
    trace_renewals: dict[str, list[bool]] = {name: [] for name in AGENT_ORDER}

    while heap:
        timestamp = heap[0][0]
        rows: dict[str, int] = {}
        while heap and heap[0][0] == timestamp:
            _, ordinal, index = heapq.heappop(heap)
            name = AGENT_ORDER[ordinal]
            rows[name] = index
            following = index + 1
            if following < len(times[name]):
                heapq.heappush(
                    heap, (int(times[name][following]), ordinal, following),
                )
        visible = [
            exact._row_symbol(frames[name], rows[name])
            for name in AGENT_ORDER if name in rows
        ]
        symbols = [value for value in visible if value is not None]
        if symbols and len(set(symbols)) != 1:
            raise ValueError("exact hierarchy has conflicting symbols at one timestamp")
        if symbols:
            symbol = str(symbols[0])
            if global_symbol is not None and symbol != global_symbol:
                current = {name: None for name in AGENT_ORDER}
                positions = {name: 0 for name in AGENT_ORDER}
                held = {name: 0 for name in AGENT_ORDER}
            global_symbol = symbol

        for name in AGENT_ORDER:
            if name not in rows:
                continue
            index = rows[name]
            segment = int(segments[name][index])
            if last_segment[name] is None or segment != last_segment[name]:
                current[name] = None
                positions[name] = 0
                held[name] = 0
                last_segment[name] = segment
            if not bool(eligible[name][index]):
                continue
            spec = specs[name]
            parents = tuple(positions[parent] for parent in spec.inherited_positions)
            observation = exact.paper_observation(
                frames[name], index, spec.lookback, parents,
            )
            renewal = current[name] is None or held[name] == int(spec.action_lock_bars)
            if renewal:
                current[name] = _greedy(policies[name], observation)
                positions[name] = int(current[name])
                held[name] = 0
            trace_times[name].append(timestamp)
            trace_actions[name].append(int(current[name]))
            trace_renewals[name].append(bool(renewal))
            held[name] += 1

    return JointGreedyTrace(agents={
        name: AgentGreedyTrace(
            times_ns=np.asarray(trace_times[name], dtype=np.int64),
            actions=np.asarray(trace_actions[name], dtype=np.int8),
            renewal_mask=np.asarray(trace_renewals[name], dtype=bool),
        )
        for name in AGENT_ORDER
    })


@dataclass(frozen=True)
class ExactEnsembleTrace:
    times_ns: np.ndarray
    actions: np.ndarray
    member_actions: np.ndarray
    renewal_mask: np.ndarray


def majority_flat_trace(member_traces: Sequence[AgentGreedyTrace]) -> ExactEnsembleTrace:
    """Align five seed traces and vote FLAT on every multi-action tie."""

    if len(member_traces) != len(SEEDS):
        raise ValueError("exact ensemble vote requires five seed traces")
    first = member_traces[0]
    for trace in member_traces[1:]:
        if (
            not np.array_equal(trace.times_ns, first.times_ns)
            or not np.array_equal(trace.renewal_mask, first.renewal_mask)
        ):
            raise ValueError("exact ensemble seed clocks or renewal schedules differ")
    values = np.stack([trace.actions for trace in member_traces]).astype(np.int8)
    if not np.isin(values, ACTIONS).all():
        raise ValueError("exact ensemble contains an invalid target")
    voted = np.asarray(
        [flat_on_vote_tie(values[:, column]) for column in range(values.shape[1])],
        dtype=np.int8,
    )
    return ExactEnsembleTrace(
        times_ns=first.times_ns.copy(), actions=voted,
        member_actions=values, renewal_mask=first.renewal_mask.copy(),
    )


def _checkpoint_path(
    root: Path, manifest: Mapping[str, object], *, agent: str,
) -> Path:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("exact member checkpoint inventory is absent")
    matches = [
        row for row in records
        if isinstance(row, dict) and Path(str(row.get("path", ""))).stem == agent.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"exact member lacks unique {agent} checkpoint")
    path = (root / str(matches[0]["path"])).resolve()
    if not path.is_file() or file_sha256(path) != matches[0].get("sha256"):
        raise ValueError("exact member checkpoint hash differs")
    return path


def load_exact_policies(
    training: ExactTrainingBinding,
) -> list[tuple[int, Mapping[str, exact.PaperDQNTrainer]]]:
    """Load inference-only online weights from the deeply verified GC ensemble."""

    root = training.manifest_path.parent
    members = _member_values(root, training.value, stage=EXACT_TRAINING_STAGE)
    contract = registered_contract("base_paper_gc")
    loaded: list[tuple[int, Mapping[str, exact.PaperDQNTrainer]]] = []
    for seed in SEEDS:
        member_path, member = members[seed]
        policies: dict[str, exact.PaperDQNTrainer] = {}
        for name in AGENT_ORDER:
            policy = exact.PaperDQNTrainer(contract.agent(name), seed=seed, device="cpu")
            checkpoint_path = _checkpoint_path(member_path.parent, member, agent=name)
            checkpoint = torch.load(
                checkpoint_path, map_location=policy.device, weights_only=False,
            )
            if (
                checkpoint.get("artifact_type")
                != "BASEPAPER_DESCRIBED_AGENT_CHECKPOINT"
                or checkpoint.get("name") != exact.PAPER_EXACT_NAME
                or checkpoint.get("stage") != EXACT_TRAINING_STAGE
                or checkpoint.get("initialization") != EXACT_INITIALIZATION
                or checkpoint.get("warm_start") is not False
                or "parent_manifest_sha256" not in checkpoint
                or checkpoint.get("parent_manifest_sha256") is not None
                or checkpoint.get("agent") != name
                or int(checkpoint.get("seed", -1)) != seed
            ):
                raise ValueError("exact inference checkpoint identity differs")
            policy.online.load_state_dict(checkpoint["online"])
            if exact._model_sha256(policy.online) != checkpoint.get("parameter_sha256"):
                raise ValueError("exact inference online parameter hash differs")
            policy.online.eval()
            policies[name] = policy
        loaded.append((seed, policies))
    return loaded


def _load_context_source(training: ExactTrainingBinding) -> exact.LoadedPaperData:
    path = training.training_source_manifest
    value = _read_object(path)
    if value.get("artifact_type") == "BASEPAPER_DESCRIBED_PREPARED_BARS":
        return exact.verify_prepared_data(path.parent, phase="train")[0]
    return exact.load_gc_timeframes(path, phase="train")


def _combined_frames(
    context: exact.LoadedPaperData, validation: exact.LoadedPaperData,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Build validation frames with market-only pre-May lookback tails.

    At most ``lookback - 1`` rows from the training phase are retained per
    agent.  Their sole purpose is to complete a first-May raw OHLCV window;
    ``evaluation_start_indexes`` prevents policy calls or state evolution on
    those rows.
    """

    if context.phase != "train" or validation.phase != "validation":
        raise ValueError("exact inference context/validation phase differs")
    if any(day > "2026-04-30" for day in context.dates):
        raise PermissionError("exact inference context reaches beyond April 2026")
    _assert_may_dates(validation.dates, label="exact validation bars")
    frames: dict[str, pd.DataFrame] = {}
    evaluation_start_indexes: dict[str, int] = {}
    contract = registered_contract("base_paper_gc")
    for name in AGENT_ORDER:
        prior = context.frames[name]
        current = validation.frames[name]
        prior_times = exact._event_ns(prior)
        current_times = exact._event_ns(current)
        if not len(current_times):
            raise ValueError(f"exact validation has zero {name} bars")
        if len(prior_times) and int(prior_times[-1]) >= int(current_times[0]):
            raise ValueError("exact inference phases overlap or are out of order")
        context_rows = max(0, int(contract.agent(name).lookback) - 1)
        tail = prior.iloc[-context_rows:].copy() if context_rows else prior.iloc[:0].copy()
        evaluation_start_indexes[name] = len(tail)
        combined = pd.concat(
            [tail, current], ignore_index=True,
        ).reset_index(drop=True)
        exact._event_ns(combined)
        if "session_date" in combined and any(
            str(day) > MAY_END for day in combined["session_date"].astype(str)
        ):
            raise PermissionError("exact inference frame crosses 2026-05-31")
        frames[name] = combined
    return frames, evaluation_start_indexes


@dataclass(frozen=True)
class ImmutableV6Reports:
    reports: Mapping[str, Mapping[str, Mapping[str, object]]]
    records: Mapping[str, Mapping[str, Mapping[str, str]]]
    complete_training_manifest: Path
    complete_training_manifest_sha256: str


def load_immutable_v6_reports(
    manifest_path: Path, *, expected_dates: Sequence[str],
) -> ImmutableV6Reports:
    """Open only hash-recorded RichVolume/BaseOHLCV May report children."""

    path = _manifest_path(manifest_path, label="immutable V6 May evaluation")
    value = _verify_v6_outer(path)
    dates = _assert_may_dates(expected_dates, label="immutable V6 comparison")
    rows = value.get("reports")
    training_record = value.get("training_manifest")
    if not isinstance(rows, list) or not isinstance(training_record, dict):
        raise ValueError("immutable V6 evaluation lacks reports/training binding")
    training_path = _resolve(path.parent, training_record.get("path"))
    training_sha = training_record.get("sha256")
    if (
        not _is_sha256(training_sha)
        or not training_path.is_file()
        or file_sha256(training_path) != training_sha
    ):
        raise ValueError("immutable V6 training receipt is absent or changed")
    selected = {
        (str(row.get("scenario")), str(row.get("arm"))): row
        for row in rows if isinstance(row, dict)
        and str(row.get("arm")) in V6_COMPARISON_ARMS
    }
    expected = {(scenario, arm) for scenario in SCENARIOS for arm in V6_COMPARISON_ARMS}
    if set(selected) != expected:
        raise ValueError("immutable V6 comparison report inventory differs")
    protocol, v6_protocol_sha = frozen_protocol()
    del protocol
    reports: dict[str, dict[str, Mapping[str, object]]] = {
        scenario: {} for scenario in SCENARIOS
    }
    records: dict[str, dict[str, Mapping[str, str]]] = {
        scenario: {} for scenario in SCENARIOS
    }
    root = path.parent.resolve()
    for scenario, arm in sorted(expected):
        row = selected[(scenario, arm)]
        child = _resolve(root, row.get("path"))
        try:
            child.relative_to(root)
        except ValueError as error:
            raise ValueError("immutable V6 report escapes its evaluation root") from error
        if not child.is_file() or file_sha256(child) != row.get("sha256"):
            raise ValueError("immutable V6 report child hash differs")
        report = _read_object(child)
        daily = report.get("daily_pnl_cents")
        metrics = report.get("metrics")
        if not isinstance(daily, dict) or not isinstance(metrics, dict):
            raise ValueError("immutable V6 report lacks daily P&L/metrics")
        normalized = {str(day): int(amount) for day, amount in daily.items()}
        report_dates = _assert_may_dates(
            [str(day) for day in report.get("evaluation_dates", [])],
            label="immutable V6 report",
        )
        recomputed = daily_metrics(
            normalized, int(metrics.get("completed_round_trips", -1)),
        )
        if (
            report.get("artifact_type") != "V6_COMMON_LEDGER_REPLAY"
            or report.get("arm") != arm
            or report.get("scenario") != scenario
            or report.get("phase") != "validation"
            or report.get("classification") != "DEVELOPMENT"
            or report.get("registered_evaluation") is not True
            or report.get("smoke_only") is not False
            or report.get("holdout_read") is not False
            or report.get("protocol_sha256") != v6_protocol_sha
            or report.get("evaluation_dataset_manifest_sha256")
            != EXPECTED_MAY_DATASET_SHA256
            or report.get("complete_training_manifest_sha256") != training_sha
            or report.get("economics") != expected_economics(scenario)
            or report_dates != dates
            or tuple(sorted(normalized)) != dates
            or normalized != daily
            or recomputed != metrics
            or int(report.get("fills", -1))
            != 2 * int(metrics.get("completed_round_trips", -1))
        ):
            raise ValueError("immutable V6 report content does not recompute")
        reports[scenario][arm] = report
        records[scenario][arm] = {
            "path": str(child), "sha256": str(row["sha256"]),
        }
    return ImmutableV6Reports(
        reports=reports,
        records=records,
        complete_training_manifest=training_path,
        complete_training_manifest_sha256=str(training_sha),
    )


def _scenario_economics(name: str) -> ExecutionEconomics:
    try:
        return {
            "base": ExecutionEconomics.from_basepaper(),
            "double_commission": ExecutionEconomics.from_basepaper(fee_multiplier=2),
            "20ms_route_latency": ExecutionEconomics.from_basepaper(route_latency_ms=20),
            "one_adverse_tick_per_filled_side": ExecutionEconomics.from_basepaper(
                adverse_ticks=1,
            ),
        }[name]
    except KeyError as error:
        raise ValueError("unknown corrected BasePaper scenario") from error


def _economic_raw(path: Path, economics: ExecutionEconomics) -> dict[str, np.ndarray]:
    prefix = f"arrival_{int(economics.route_latency_ms)}ms_"
    required = {
        "clock_ns", "continuity_segment", "valid",
        "best_bid_ticks", "best_ask_ticks", "best_bid_size", "best_ask_size",
        prefix + "clock_ns", prefix + "valid", prefix + "bid_ticks",
        prefix + "ask_ticks", prefix + "bid_size", prefix + "ask_size",
        prefix + "state_available_ns", prefix + "continuity_segment",
    }
    with np.load(path, allow_pickle=False) as source:
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"May economic shard lacks {sorted(missing)}")
        return {name: source[name] for name in required}


def _side_bounds(path: Path, *, day: str) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as source:
        required = {"session_date", "session_start_ns", "session_end_ns"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"May sidecar lacks {sorted(missing)}")
        if str(np.asarray(source["session_date"]).item()) != day:
            raise ValueError("May sidecar date differs from its manifest key")
        return int(source["session_start_ns"].item()), int(source["session_end_ns"].item())


def _trace_sha256(times: np.ndarray, actions: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(times, dtype=np.int64).tobytes(order="C"))
    digest.update(np.asarray(actions, dtype=np.int8).tobytes(order="C"))
    return digest.hexdigest()


def _exact_scenario_report(
    trace: ExactEnsembleTrace,
    pairs: Sequence[tuple[Path, Path]],
    *,
    scenario: str,
    authorization: MayAccessAuthorization,
    training_checkpoint_set: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    economics = _scenario_economics(scenario)
    if asdict(economics) != expected_economics(scenario):
        raise AssertionError("corrected evaluator scenario registry drifted from V6")
    daily: dict[str, int] = {}
    trips = rejected = fills = reversals = 0
    unsafe = False
    hold_minutes = (
        registered_contract("base_paper_gc").agent("M5").minutes
        * registered_contract("base_paper_gc").agent("M5").action_lock_bars
    )
    for raw_path, sidecar_path in pairs:
        day = raw_path.stem
        _assert_may_dates([day], label="common-ledger replay")
        start_ns, end_ns = _side_bounds(sidecar_path, day=day)
        raw = _economic_raw(raw_path, economics)
        replay = exact.replay_paper_targets(
            raw,
            session_date=day,
            session_start_ns=start_ns,
            session_end_ns=end_ns,
            signal_times_ns=trace.times_ns,
            signal_actions=trace.actions,
            hold_minutes=hold_minutes,
            economics=economics,
        )
        daily[day] = int(replay.result.net_pnl_cents)
        trips += int(replay.result.completed_round_trips)
        rejected += int(replay.result.rejected_commands)
        fills += len(replay.result.fills)
        reversals += int(replay.direct_reversal_decisions)
        unsafe |= bool(replay.result.unsafe_legacy_arrival_boundary)
    dates = _assert_may_dates(sorted(daily), label="common-ledger result")
    return {
        "schema_version": 1,
        "artifact_type": REPORT_ARTIFACT,
        "strategy_version": CORRECTION_STRATEGY_VERSION,
        "name": exact.PAPER_EXACT_NAME,
        "arm": exact.PAPER_EXACT_NAME,
        "phase": "validation",
        "classification": CORRECTION_CLASSIFICATION,
        "training_stage": EXACT_TRAINING_STAGE,
        "training_initialization": EXACT_INITIALIZATION,
        "pretraining": None,
        "scenario": scenario,
        "economics": asdict(economics),
        "ensemble_seeds": list(SEEDS),
        "ensemble_rule": "five_seed_final_M5_majority_FLAT_on_every_tie_v1",
        "inference": "joint_greedy_H1_then_M15_then_M5_current_seed_parents_v1",
        "evaluation_initialization": EVALUATION_INITIALIZATION,
        "pre_may_policy_state_replayed": False,
        "pre_may_context_authority": "raw_OHLCV_lookback_only",
        "evaluation_convention_scope": (
            "local_frozen_heldout_reconstruction_not_paper_described_evaluation_rule"
        ),
        "only_m5_controls_common_ledger_inventory": True,
        "h1_m15_authority": "information_only",
        "direct_reversal_rule": "atomic_close_and_open_opposite_if_two_side_size_available",
        "evaluation_dates": list(dates),
        "daily_pnl_cents": daily,
        "metrics": daily_metrics(daily, trips),
        "fills": fills,
        "rejected_commands": rejected,
        "direct_reversal_decisions": reversals,
        "unsafe_legacy_arrival_boundary": unsafe,
        "checkpoint_set": list(training_checkpoint_set),
        "exact_gc_training_manifest_sha256": authorization.training.manifest_sha256,
        "evaluation_dataset_manifest_sha256": authorization.may_dataset_sha256,
        "may_access_receipt_sha256": authorization.receipt_sha256,
        "protocol_sha256": authorization.protocol_sha256,
        "signal_count": int(len(trace.times_ns)),
        "renewal_count": int(trace.renewal_mask.sum()),
        "ensemble_target_counts": {
            str(action): int((trace.actions == action).sum()) for action in ACTIONS
        },
        "target_trace_sha256": _trace_sha256(trace.times_ns, trace.actions),
        "member_target_trace_sha256": [
            _trace_sha256(trace.times_ns, member) for member in trace.member_actions
        ],
        "registered_evaluation": True,
        "smoke_only": False,
        "holdout_read": False,
    }


def evaluate_may(access_receipt: Path, output: Path) -> Path:
    """Run the one corrected deterministic May replay and immutable comparison.

    This function has no phase/date argument by design.  It authenticates the
    complete access receipt before loading a May Parquet, NPZ, sidecar, model,
    or historical V6 report child.
    """

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite corrected May evaluation: {output}")
    authorization = verify_may_access_receipt(access_receipt)

    # The authorization above is the mandatory no-child-read boundary.
    comparisons = load_immutable_v6_reports(
        authorization.v6_evaluation_manifest,
        expected_dates=authorization.may_dates,
    )
    from .volume_v6_orchestration import load_phase_dataset
    dataset = load_phase_dataset(
        authorization.may_dataset_manifest,
        phase="validation",
        complete_training_manifest=comparisons.complete_training_manifest,
    )
    if dataset.manifest_sha256 != authorization.may_dataset_sha256:
        raise ValueError("deep May phase dataset differs from its access receipt")
    if dataset.dates != authorization.may_dates:
        raise ValueError("deep May phase dataset date set differs")

    context = _load_context_source(authorization.training)
    validation = exact.load_gc_timeframes(
        authorization.may_dataset_manifest,
        phase="validation",
        protocol=authorization.protocol_path,
        freeze_receipt=authorization.freeze_receipt_path,
    )
    frames, evaluation_starts = _combined_frames(context, validation)
    policies = load_exact_policies(authorization.training)
    member_traces: list[AgentGreedyTrace] = []
    checkpoint_set: list[dict[str, object]] = []
    training_members = _member_values(
        authorization.training.manifest_path.parent,
        authorization.training.value,
        stage=EXACT_TRAINING_STAGE,
    )
    for seed, seeded_policies in policies:
        joint = greedy_joint_inference(
            seeded_policies,
            frames,
            evaluation_start_indexes=evaluation_starts,
        )
        member_traces.append(joint.agents["M5"])
        checkpoint_set.append({
            "seed": seed,
            "manifest_sha256": file_sha256(training_members[seed][0]),
        })
    trace = majority_flat_trace(member_traces)
    scenario_reports = {
        scenario: _exact_scenario_report(
            trace, dataset.pairs, scenario=scenario, authorization=authorization,
            training_checkpoint_set=checkpoint_set,
        )
        for scenario in SCENARIOS
    }
    paired: dict[str, dict[str, Mapping[str, float]]] = {}
    for scenario in SCENARIOS:
        exact_report = scenario_reports[scenario]
        rich = comparisons.reports[scenario]["V6-RichVolume"]
        base = comparisons.reports[scenario]["V6-BaseOHLCV"]
        paired[scenario] = {
            "V6-RichVolume_minus_corrected_BasePaper": paired_improvement(
                rich, exact_report,
            ),
            "V6-BaseOHLCV_minus_corrected_BasePaper": paired_improvement(
                base, exact_report,
            ),
            "corrected_BasePaper_minus_V6-RichVolume": paired_improvement(
                exact_report, rich,
            ),
            "corrected_BasePaper_minus_V6-BaseOHLCV": paired_improvement(
                exact_report, base,
            ),
        }

    output.mkdir(parents=True, exist_ok=False)
    report_records: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        report_path = output / f"{scenario}.json"
        atomic_json(report_path, scenario_reports[scenario])
        report_records.append({
            "scenario": scenario, "path": report_path.name,
            "sha256": file_sha256(report_path),
        })
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": EVALUATION_ARTIFACT,
        "strategy_version": CORRECTION_STRATEGY_VERSION,
        "name": exact.PAPER_EXACT_NAME,
        "complete": True,
        "phase": "validation",
        "classification": CORRECTION_CLASSIFICATION,
        "training_stage": EXACT_TRAINING_STAGE,
        "training_initialization": EXACT_INITIALIZATION,
        "pretraining": None,
        "evaluation_initialization": EVALUATION_INITIALIZATION,
        "pre_may_policy_state_replayed": False,
        "pre_may_context_authority": "raw_OHLCV_lookback_only",
        "evaluation_convention_scope": (
            "local_frozen_heldout_reconstruction_not_paper_described_evaluation_rule"
        ),
        "evaluation_dates": list(authorization.may_dates),
        "scenarios": list(SCENARIOS),
        "reports": report_records,
        "metrics": {
            scenario: scenario_reports[scenario]["metrics"] for scenario in SCENARIOS
        },
        "paired_comparisons": paired,
        "immutable_v6_reports": comparisons.records,
        "may_access_receipt": _record(
            authorization.receipt_path, digest=authorization.receipt_sha256,
        ),
        "protocol": {
            "path": str(authorization.protocol_path),
            "canonical_sha256": authorization.protocol_sha256,
        },
        "exact_gc_training_manifest": _record(
            authorization.training.manifest_path,
            digest=authorization.training.manifest_sha256,
        ),
        "may_dataset_manifest": _record(
            authorization.may_dataset_manifest,
            digest=authorization.may_dataset_sha256,
        ),
        "immutable_v6_evaluation_manifest": _record(
            authorization.v6_evaluation_manifest,
            digest=authorization.v6_evaluation_manifest_sha256,
        ),
        "verdict": "DESCRIPTIVE_ONLY",
        "holdout_opened": False,
        "holdout_read": False,
        "strongest_justified_action": (
            "report corrected contaminated-development comparison; do not open holdout"
        ),
    }
    return atomic_json(output / "manifest.json", manifest)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    authorize = commands.add_parser(
        "authorize-may",
        help="freeze exact post-training/source/report hashes before any May child read",
    )
    authorize.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    authorize.add_argument("--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT)
    authorize.add_argument("--training-manifest", type=Path, required=True)
    authorize.add_argument("--may-dataset-manifest", type=Path, required=True)
    authorize.add_argument("--v6-evaluation-manifest", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser(
        "evaluate-may",
        aliases=["evaluate"],
        help="run deterministic corrected May replay from one access receipt",
    )
    evaluate.add_argument("--access-receipt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser(
        "verify-access", help="recompute the May access chain without reading May children",
    )
    verify.add_argument("--access-receipt", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "authorize-may":
        print(authorize_may(
            args.protocol, args.freeze_receipt, args.training_manifest,
            args.may_dataset_manifest, args.v6_evaluation_manifest, args.output,
        ))
        return 0
    if args.command in {"evaluate-may", "evaluate"}:
        print(evaluate_may(args.access_receipt, args.output))
        return 0
    if args.command == "verify-access":
        authorization = verify_may_access_receipt(args.access_receipt)
        print(json.dumps({
            "artifact_type": ACCESS_ARTIFACT,
            "status": "PASS",
            "receipt_sha256": authorization.receipt_sha256,
            "classification": CORRECTION_CLASSIFICATION,
            "holdout_opened": False,
        }, sort_keys=True, indent=2))
        return 0
    raise AssertionError("argparse accepted an unknown exact-evaluation command")


__all__ = [
    "ACCESS_ARTIFACT",
    "CORRECTION_CLASSIFICATION",
    "CORRECTION_STRATEGY_VERSION",
    "EVALUATION_INITIALIZATION",
    "EVALUATION_ARTIFACT",
    "EXACT_INITIALIZATION",
    "EXACT_TRAINING_STAGE",
    "AgentGreedyTrace",
    "ExactEnsembleTrace",
    "ExactTrainingBinding",
    "JointGreedyTrace",
    "MayAccessAuthorization",
    "authorize_may",
    "canonical_json_sha256",
    "evaluate_may",
    "greedy_joint_inference",
    "load_exact_policies",
    "load_immutable_v6_reports",
    "majority_flat_trace",
    "verify_corrected_freeze",
    "verify_exact_training_chain",
    "verify_may_access_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
