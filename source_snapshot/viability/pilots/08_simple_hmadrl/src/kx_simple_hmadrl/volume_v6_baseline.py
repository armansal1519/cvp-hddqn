"""Frozen paper-style comparators for the two-level V6 GC experiment.

This module deliberately lives beside, rather than modifies, the closed V5
implementation.  It preserves the published H1/M15/M5 observation clocks,
lookbacks, action locks, discounts, and MLP widths.  The two registered arms
are:

``base_paper_gc``
    Vanilla DQN, RMSProp, MSE, the registered 1,000-update target clock, and
    the paper's gross fractional horizon-return training reward.

``base_strong_gc_v6``
    Parameter-identical clocks and observations with Double-DQN, a dueling
    head, Adam, Huber loss, a reachable 100-update target clock, and exact
    after-cost rewards from the same V6 one-contract execution ledger.

Both arms are pretrained from 2012--2024 Dukascopy XAUUSD midpoint OHLC and a
fifth ``quote_count`` channel.  That channel is always disclosed as an OTC
quote-count proxy, never exchange-traded volume.  GC fine-tuning is physically
date-truncated at 2026-04-30 before any shard is opened.  Evaluation of the
June--August test requires the immutable V6 unlock receipt before any market
data or checkpoint is read.

Only the M5 target trace is replayed into the economic ledger.  H1 and M15 are
information publishers exactly as in the paper-style hierarchy.  Direct M5
reversals are handled transparently: the first decision requests FLAT and a
new opposite position can be entered only at a later completed five-minute
decision.  No execution DRL is present.

Registered status is never inferred from a date-shaped list of files.  GC
training and evaluation must be bound to the immutable phase-dataset and
complete-training manifests emitted by ``volume_v6_orchestration``; direct
raw-path calls without those proofs are accepted only under explicit
``phase='smoke'`` on contaminated dates and are permanently smoke-only.

Three deliberate V6 comparator adaptations are disclosed in every GC
manifest: downstream training consumes a post-training greedy parent trace,
BaseStrong uses common-ledger GC-tick reward despite paper observations having
no inventory field, and an opposite target is adapted to FLAT until a later
renewal.  They are frozen differences, not claims about the source paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from kx_base_gc.contract import (
    ACTIONS,
    AGENT_ORDER,
    AgentSpec,
    BaselineContract,
    registered_contract,
)
from kx_base_gc.models import DQNTrainer, ReplayItem
from kx_base_gc.state import encode_window

from .environment import ExecutionEconomics
from .volume_v6_contract import SEEDS, assert_phase_date
from .volume_v6_env import V6SessionResult, V6TradingEnvironment
from .volume_v6_eval import (
    assert_test_unlocked,
    classify_dates,
    daily_metrics,
    flat_on_vote_tie,
    frozen_protocol,
)


BaselineVariant = Literal["base_paper_gc", "base_strong_gc_v6"]
EvaluationPhase = Literal["smoke", "validation", "test"]

BASE_PAPER_VARIANT: BaselineVariant = "base_paper_gc"
BASE_STRONG_VARIANT: BaselineVariant = "base_strong_gc_v6"
BASELINE_NAMES: dict[BaselineVariant, str] = {
    BASE_PAPER_VARIANT: "BasePaper-XAU-Pretrained-GC",
    BASE_STRONG_VARIANT: "BaseStrong-XAU-Pretrained-GC-v6",
}
REGISTERED_GC_PASSES = 3
XAU_START_MONTH = "2012-01"
XAU_END_MONTH_EXCLUSIVE = "2025-01"
GC_TRAIN_START = "2025-01-01"
GC_TRAIN_END = "2026-04-30"
NS_PER_MINUTE = 60_000_000_000
MAX_RESTRICTED_SESSION_SEAM_DAYS = 7
WARMSTART_REPLAY_ITEMS = 512
REGISTERED_SCENARIOS = (
    "base", "double_commission", "20ms_route_latency",
    "one_adverse_tick_per_filled_side",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_torch(path: Path, value: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent,
    )
    os.close(descriptor)
    try:
        torch.save(dict(value), temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def _normalise_variant(variant: str) -> BaselineVariant:
    # ``base_strong_gc`` is accepted only as a convenience for old pipeline
    # callers; every V6 artifact records the unambiguous v6 name.
    if variant == "base_strong_gc":
        return BASE_STRONG_VARIANT
    if variant not in BASELINE_NAMES:
        raise ValueError(f"unknown V6 baseline variant {variant!r}")
    return variant  # type: ignore[return-value]


def baseline_contract(variant: str) -> BaselineContract:
    """Return the exact frozen optimizer/model contract for one comparator."""

    selected = _normalise_variant(variant)
    if selected == BASE_PAPER_VARIANT:
        contract = registered_contract("base_paper_gc")
    else:
        parent = registered_contract("base_strong_gc")
        optimization = replace(parent.optimization, target_sync_updates=100)
        contract = replace(
            parent,
            contract_id="base_strong_gc_v6_structural_v1",
            optimization=optimization,
            reward="v6_exact_common_ledger_after_cost_reward_ticks_v1",
            registered_defaults=parent.registered_defaults + (
                "V6 target network synchronization every 100 optimizer updates",
                "raw common-ledger reward cents divided by the fixed 1000-cents-per-GC-tick divisor",
                "M5 alone controls common-ledger inventory; H1 and M15 publish positions",
            ),
        )
    contract.validate()
    return contract


def economics_for_scenario(scenario: str) -> ExecutionEconomics:
    """Return the one exact registered ledger configuration for a report label."""

    if scenario == "base":
        return ExecutionEconomics.from_basepaper()
    if scenario == "double_commission":
        return ExecutionEconomics.from_basepaper(fee_multiplier=2)
    if scenario == "20ms_route_latency":
        return ExecutionEconomics.from_basepaper(route_latency_ms=20)
    if scenario == "one_adverse_tick_per_filled_side":
        return ExecutionEconomics.from_basepaper(adverse_ticks=1)
    raise ValueError(f"scenario must be one of {REGISTERED_SCENARIOS}")


def registered_comparator_contracts() -> dict[str, BaselineContract]:
    return {variant: baseline_contract(variant) for variant in BASELINE_NAMES}


@dataclass(frozen=True)
class SessionPair:
    """Paths for one aligned economic V2 shard and V6 trade sidecar."""

    v2_path: Path
    sidecar_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "v2_path", Path(self.v2_path))
        object.__setattr__(self, "sidecar_path", Path(self.sidecar_path))
        if self.v2_path.stem != self.sidecar_path.stem:
            raise ValueError("baseline raw and sidecar filenames must share a date stem")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.v2_path.stem) is None:
            raise ValueError("baseline session filenames must use YYYY-MM-DD")
        try:
            pd.Timestamp(self.v2_path.stem)
        except ValueError as error:
            raise ValueError("baseline session filenames must use YYYY-MM-DD") from error

    @property
    def day(self) -> str:
        return self.v2_path.stem


def _pairs(values: Sequence[SessionPair | tuple[Path, Path]]) -> list[SessionPair]:
    result = [value if isinstance(value, SessionPair) else SessionPair(*value) for value in values]
    result.sort(key=lambda value: value.day)
    days = [value.day for value in result]
    if not result or len(days) != len(set(days)):
        raise ValueError("baseline sessions must be nonempty with unique dates")
    return result


def assert_training_sessions(values: Sequence[SessionPair | tuple[Path, Path]]) -> list[SessionPair]:
    """Validate every filename before opening any training input."""

    result = _pairs(values)
    classify_dates([value.day for value in result], phase="train")
    return result


def assert_evaluation_sessions(
    values: Sequence[SessionPair | tuple[Path, Path]],
    *,
    phase: EvaluationPhase,
    unlock_receipt: Path | None = None,
) -> list[SessionPair]:
    """Validate an evaluation request before any input or model is opened."""

    result = _pairs(values)
    classify_dates([value.day for value in result], phase=phase)
    if phase == "smoke" and any(
        not "2025-01-01" <= value.day <= "2025-08-31" for value in result
    ):
        raise PermissionError("baseline smoke evaluation is limited to contaminated Jan-Aug 2025")
    if phase == "test":
        if unlock_receipt is None:
            raise PermissionError("V6 test evaluation requires its immutable unlock receipt")
        assert_test_unlocked(Path(unlock_receipt))
    return result


def _resolved_pair_paths(pairs: Sequence[SessionPair]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(pair.v2_path.resolve()), str(pair.sidecar_path.resolve()))
        for pair in pairs
    )


def _verify_phase_dataset_binding(
    manifest_path: Path,
    pairs: Sequence[SessionPair],
    *,
    phase: Literal["train", "validation", "test"],
    complete_training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> dict[str, object]:
    """Bind direct APIs to the exact phase-built dataset orchestration proof."""

    # Lazy import avoids a module cycle: orchestration imports this comparator
    # only inside its worker/evaluation functions.
    from .volume_v6_orchestration import load_phase_dataset

    dataset = load_phase_dataset(
        Path(manifest_path), phase=phase,
        complete_training_manifest=complete_training_manifest,
        unlock_receipt=unlock_receipt,
    )
    expected = tuple(
        (str(raw.resolve()), str(side.resolve()))
        for raw, side in dataset.pairs
    )
    if _resolved_pair_paths(pairs) != expected:
        raise ValueError("direct baseline paths differ from the exact phase dataset manifest")
    return {
        "path": str(Path(manifest_path).resolve()),
        "sha256": dataset.manifest_sha256,
        "phase": phase,
        "dates": list(dataset.dates),
        "session_count": len(dataset.dates),
        "coverage_digest": _canonical_digest({
            "manifest_sha256": dataset.manifest_sha256,
            "dates": list(dataset.dates),
            "raw_paths": [str(path.resolve()) for path in dataset.raw_paths],
            "sidecar_paths": [str(path.resolve()) for path in dataset.sidecar_paths],
        }),
    }


def _verify_complete_checkpoint_binding(
    complete_training_manifest: Path,
    checkpoint_roots: Sequence[Path],
    *,
    selected: BaselineVariant,
    training_dataset_sha256: str,
) -> tuple[str, dict[str, object]]:
    """Prove the five roots are exactly those in the complete training receipt."""

    from .volume_v6_orchestration import verify_complete_training_manifest

    path = Path(complete_training_manifest).resolve()
    value = verify_complete_training_manifest(path)
    dataset_record = value.get("dataset_manifest")
    if (
        not isinstance(dataset_record, dict)
        or dataset_record.get("sha256") != training_dataset_sha256
    ):
        raise ValueError("complete training receipt binds a different training dataset")
    records = value.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("complete training receipt has no checkpoint records")
    expected: list[tuple[int, str]] = []
    arm = BASELINE_NAMES[selected]
    for record in records:
        if isinstance(record, dict) and record.get("arm") == arm:
            manifest = Path(str(record.get("manifest", "")))
            if not manifest.is_absolute():
                manifest = (path.parent / manifest).resolve()
            expected.append((int(record.get("seed", -1)), str(manifest.parent.resolve())))
    expected.sort()
    # Callers may supply roots in any order.  Read their manifest seed rather
    # than relying on positional correspondence for the final comparison.
    observed = []
    for root in checkpoint_roots:
        manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
        observed.append((int(manifest.get("seed", -1)), str(Path(root).resolve())))
    observed.sort()
    if expected != observed:
        raise ValueError("checkpoint roots differ from the complete training receipt")
    return sha256_file(path), value


def _verify_xau_parent_binding(
    complete_training_manifest: Path,
    complete: Mapping[str, object],
    records: Sequence[tuple[int, Path, Mapping[str, object]]],
    *,
    selected: BaselineVariant,
) -> None:
    aggregate_record = complete.get("pretraining_manifest")
    if not isinstance(aggregate_record, dict):
        raise ValueError("complete training receipt lacks its XAU parent set")
    aggregate_path = Path(str(aggregate_record.get("path", "")))
    complete_path = Path(complete_training_manifest).resolve()
    if not aggregate_path.is_absolute():
        aggregate_path = (complete_path.parent / aggregate_path).resolve()
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("complete XAU pretraining receipt lacks component artifacts")
    expected: dict[int, tuple[Path, str]] = {}
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("kind") == selected:
            path = Path(str(artifact.get("manifest", "")))
            if not path.is_absolute():
                path = (aggregate_path.parent / path).resolve()
            expected[int(artifact.get("seed", -1))] = (
                path, str(artifact.get("manifest_sha256", "")),
            )
    if set(expected) != set(SEEDS):
        raise ValueError("complete XAU receipt lacks the comparator's five seeded parents")
    for seed, _, manifest in records:
        parent_path, parent_sha = expected[seed]
        recorded_path = Path(str(manifest.get("pretraining_manifest", "")))
        if (
            recorded_path.resolve() != parent_path.resolve()
            or manifest.get("pretraining_manifest_sha256") != parent_sha
        ):
            raise ValueError(f"seed {seed} comparator binds a different XAU parent")


def _access_phase(day: str) -> Literal["train", "validation", "test"]:
    if day <= GC_TRAIN_END:
        return "train"
    if day < "2026-06-01":
        return "validation"
    return "test"


_BASELINE_RAW_FIELDS = (
    "clock_ns", "continuity_segment", "raw_symbol", "session_date",
)
_BASELINE_SIDECAR_FIELDS = (
    "session_date", "session_start_ns", "session_end_ns", "trade_available_ns",
    "trade_price_ticks", "trade_size", "source_sha256",
)


def _load_feature_pair(
    pair: SessionPair,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Load only baseline features, never MBP-10 or simulator arrivals."""

    assert_phase_date(pair.day, _access_phase(pair.day))
    raw_digest = sha256_file(pair.v2_path)
    side_digest = sha256_file(pair.sidecar_path)
    with np.load(pair.v2_path, allow_pickle=False) as source:
        missing = set(_BASELINE_RAW_FIELDS).difference(source.files)
        if missing:
            raise ValueError(f"baseline raw shard lacks {sorted(missing)}")
        raw = {name: source[name] for name in _BASELINE_RAW_FIELDS}
    with np.load(pair.sidecar_path, allow_pickle=False) as source:
        missing = set(_BASELINE_SIDECAR_FIELDS).difference(source.files)
        if missing:
            raise ValueError(f"baseline sidecar lacks {sorted(missing)}")
        side = {name: source[name] for name in _BASELINE_SIDECAR_FIELDS}
    raw_day = str(np.asarray(raw["session_date"]).item())
    side_day = str(np.asarray(side["session_date"]).item())
    if raw_day != pair.day or side_day != pair.day:
        raise ValueError("baseline filename/raw/sidecar session dates differ")
    if str(np.asarray(side["source_sha256"]).item()) != raw_digest:
        raise ValueError("baseline sidecar source hash differs from its raw shard")
    record = {
        "date": pair.day,
        "v2_path": str(pair.v2_path),
        "v2_sha256": raw_digest,
        "sidecar_path": str(pair.sidecar_path),
        "sidecar_sha256": side_digest,
    }
    return raw, side, record


def _load_side_bounds(pair: SessionPair) -> tuple[int, int]:
    with np.load(pair.sidecar_path, allow_pickle=False) as source:
        required = {"session_start_ns", "session_end_ns"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"baseline sidecar lacks {sorted(missing)}")
        return int(source["session_start_ns"].item()), int(source["session_end_ns"].item())


def majority_target_vote(member_actions: np.ndarray) -> np.ndarray:
    """Vote on semantic target positions; return FLAT on every tie."""

    values = np.asarray(member_actions, dtype=np.int8)
    if values.ndim != 2 or values.shape[0] != len(SEEDS):
        raise ValueError("V6 baseline ensemble requires five aligned seed traces")
    if not np.isin(values, ACTIONS).all():
        raise ValueError("baseline ensemble contains an invalid target action")
    return np.asarray(
        [flat_on_vote_tie(values[:, column]) for column in range(values.shape[1])],
        dtype=np.int8,
    )


_BAR_VALUES = ("open", "high", "low", "close", "volume")
_BAR_METADATA = ("state_id", "session_date", "raw_symbol", "session_end_ns")


def _vector_complete_rollup(
    source: pd.DataFrame,
    *,
    target_end_ns: np.ndarray,
    component_minutes: int,
    component_count: int,
    source_is_open_labelled: bool,
) -> pd.DataFrame:
    """Vectorized exact-component OHLCV rollup for multi-year XAU history."""

    frame = source.copy()
    frame["__target_end_ns"] = np.asarray(target_end_ns, dtype=np.int64)
    frame["__event_ns"] = frame["event_time"].array.asi8
    grouped = frame.groupby("__target_end_ns", sort=True, observed=True)
    aggregation: dict[str, str] = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }
    for column in _BAR_METADATA:
        if column in frame:
            aggregation[column] = "first"
    result = grouped.agg(aggregation)
    count = grouped.size()
    first = grouped["__event_ns"].min()
    last = grouped["__event_ns"].max()
    interval = component_minutes * NS_PER_MINUTE
    target = result.index.to_numpy(dtype=np.int64)
    expected_first = target - component_count * interval if source_is_open_labelled else (
        target - (component_count - 1) * interval
    )
    expected_last = target - interval if source_is_open_labelled else target
    valid = (
        (count.to_numpy(dtype=np.int64) == component_count)
        & (first.to_numpy(dtype=np.int64) == expected_first)
        & (last.to_numpy(dtype=np.int64) == expected_last)
    )
    for column in _BAR_METADATA:
        if column in frame:
            valid &= grouped[column].nunique(dropna=False).to_numpy(dtype=np.int64) == 1
    result = result.loc[valid].copy()
    result.insert(
        0, "event_time",
        pd.to_datetime(result.index.to_numpy(dtype=np.int64), utc=True),
    )
    result["component_count"] = component_count
    return result.reset_index(drop=True)


def _complete_m5_from_one_minute(one: pd.DataFrame) -> pd.DataFrame:
    """Build right-labelled M5 bars from five exact one-minute open stamps."""

    source = one.copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    source = source.sort_values("event_time", kind="stable").reset_index(drop=True)
    if source["event_time"].duplicated().any():
        raise ValueError("one-minute baseline input contains duplicate timestamps")
    event_ns = source["event_time"].array.asi8
    if np.any(event_ns % NS_PER_MINUTE):
        raise ValueError("one-minute baseline timestamps must be exact minute opens")
    interval = 5 * NS_PER_MINUTE
    # One-minute products are open-labelled.  An observation stamped exactly
    # 08:00 therefore belongs to the M5 bar completed at 08:05.
    target = (event_ns // interval + 1) * interval
    return _vector_complete_rollup(
        source, target_end_ns=target, component_minutes=1,
        component_count=5, source_is_open_labelled=True,
    )


def _complete_rollup_from_m5(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Build canonical M15/H1 bars from exact completed M5 components."""

    if minutes not in {15, 60}:
        raise ValueError("baseline coarse rollups must be M15 or H1")
    source = m5.copy()
    source["event_time"] = pd.to_datetime(source["event_time"], utc=True, errors="raise")
    source = source.sort_values("event_time", kind="stable").reset_index(drop=True)
    event_ns = source["event_time"].array.asi8
    if np.any(event_ns % (5 * NS_PER_MINUTE)):
        raise ValueError("M5 baseline timestamps must be exact completed boundaries")
    interval = minutes * NS_PER_MINUTE
    factor = minutes // 5
    # Completed M5 timestamps on an exact coarse boundary belong to that
    # boundary, matching kx_base_gc.bars._complete_rollup.
    target = ((event_ns - 1) // interval + 1) * interval
    return _vector_complete_rollup(
        source, target_end_ns=target, component_minutes=5,
        component_count=factor, source_is_open_labelled=False,
    )


def _build_timeframes(one: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return M5 plus exact 3/12-component M15/H1 canonical rollups."""

    m5 = _complete_m5_from_one_minute(one)
    return {
        "H1": _complete_rollup_from_m5(m5, 60),
        "M15": _complete_rollup_from_m5(m5, 15),
        "M5": m5,
    }


def _validate_xau_manifest(root: Path, *, smoke_only: bool) -> tuple[list[Path], str, list[str]]:
    manifest_path = root / "manifest.json"
    digest_path = root / "manifest.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise ValueError("direct XAU product lacks its immutable manifest receipt")
    if digest_path.read_text(encoding="utf-8") != f"{sha256_file(manifest_path)}  manifest.json\n":
        raise ValueError("direct XAU top-level manifest digest differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_kind") != "Dukascopy OTC tick-by-tick quotes":
        raise ValueError("baseline pretraining requires direct Dukascopy quote history")
    semantics = manifest.get("volume_semantics")
    if not isinstance(semantics, dict) or "proxy" not in str(semantics.get("quote_count", "")).lower():
        raise ValueError("XAU source does not disclose quote_count proxy semantics")
    months = manifest.get("months")
    if not isinstance(months, list):
        raise ValueError("direct XAU manifest lacks monthly products")
    selected = [
        value for value in months
        if isinstance(value, dict)
        and XAU_START_MONTH <= str(value.get("month", "")) < XAU_END_MONTH_EXCLUSIVE
    ]
    selected.sort(key=lambda value: str(value["month"]))
    month_names = [str(value["month"]) for value in selected]
    if not selected or len(month_names) != len(set(month_names)):
        raise ValueError("direct XAU products have missing or duplicate pretraining months")
    if not smoke_only:
        expected = [str(value)[:7] for value in pd.period_range("2012-01", "2024-12", freq="M")]
        if month_names != expected:
            raise ValueError("registered XAU pretraining requires every month from 2012 through 2024")
    paths: list[Path] = []
    for record in selected:
        path = root / str(record["one_minute"])
        if not path.is_file() or sha256_file(path) != record.get("one_minute_sha256"):
            raise ValueError(f"XAU one-minute product hash differs for {record['month']}")
        paths.append(path)
    return paths, sha256_file(manifest_path), month_names


def _load_xau_one_minute(
    root: Path, *, smoke_only: bool,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    paths, manifest_sha, months = _validate_xau_manifest(root, smoke_only=smoke_only)
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(
            path, columns=["event_time", "open", "high", "low", "close", "quote_count"],
        )
        if frame["quote_count"].isna().any() or (frame["quote_count"] < 0).any():
            raise ValueError("XAU quote_count proxy is missing or negative")
        frames.append(frame.rename(columns={"quote_count": "volume"}))
    one = pd.concat(frames, ignore_index=True).sort_values("event_time", kind="stable")
    one["event_time"] = pd.to_datetime(one["event_time"], utc=True)
    if one["event_time"].duplicated().any():
        raise ValueError("XAU one-minute products contain duplicate timestamps")
    values = one.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("XAU one-minute OHLC/quote-count values are invalid")
    source_digest = _canonical_digest({"manifest_sha256": manifest_sha, "months": months})
    return one, source_digest, manifest_sha, months


def _segment_for_trades(raw: Mapping[str, np.ndarray], times: np.ndarray) -> np.ndarray:
    clocks = np.asarray(raw["clock_ns"], dtype=np.int64)
    indexes = np.searchsorted(clocks, times, side="right") - 1
    indexes = np.clip(indexes, 0, len(clocks) - 1)
    return np.asarray(raw["continuity_segment"])[indexes]


def _load_gc_one_minute(
    pairs: Sequence[SessionPair],
) -> tuple[pd.DataFrame, str, list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, str]] = []
    for pair in pairs:
        raw, side, record = _load_feature_pair(pair)
        records.append(record)
        times = np.asarray(side["trade_available_ns"], dtype=np.int64)
        prices = np.asarray(side["trade_price_ticks"], dtype=np.float64) * 0.1
        sizes = np.asarray(side["trade_size"], dtype=np.float64)
        if not len(times):
            raise ValueError(f"GC baseline session {pair.day} contains no trades")
        symbol = str(np.asarray(raw["raw_symbol"]).item())
        segments = _segment_for_trades(raw, times)
        ticks = pd.DataFrame({
            "event_time": pd.to_datetime(times, utc=True),
            "price": prices,
            "volume": sizes,
            "state_id": [f"{symbol}:{int(value)}" for value in segments],
        }).set_index("event_time")
        bars = ticks.resample("1min").agg(
            open=("price", "first"), high=("price", "max"),
            low=("price", "min"), close=("price", "last"),
            volume=("volume", "sum"), state_id=("state_id", "first"),
            state_last=("state_id", "last"), state_count=("state_id", "nunique"),
        ).dropna(subset=["open", "high", "low", "close"])
        bars = bars.loc[(bars["state_count"] == 1) & (bars["state_id"] == bars["state_last"])]
        bars = bars.drop(columns=["state_last", "state_count"])
        bars["session_date"] = pair.day
        bars["raw_symbol"] = symbol
        bars["session_end_ns"] = int(np.asarray(side["session_end_ns"]).item())
        frames.append(bars.reset_index())
    one = pd.concat(frames, ignore_index=True).sort_values("event_time", kind="stable")
    source_digest = _canonical_digest(records)
    return one, source_digest, records


def _session_boundaries(frame: pd.DataFrame) -> np.ndarray:
    if "session_date" in frame:
        return frame["session_date"].astype(str).to_numpy()
    # Direct XAU products do not label sessions.  UTC calendar days are an
    # explicit operational boundary for action-lock/reward accounting; state
    # history may still cross an exact midnight bar step.
    return frame["event_time"].dt.strftime("%Y-%m-%d").to_numpy(dtype=str)


def _state_ids(frame: pd.DataFrame) -> np.ndarray:
    if "state_id" in frame:
        return frame["state_id"].astype(str).to_numpy()
    return np.full(len(frame), "XAUUSD:continuous", dtype=object)


def _contiguous_state_indexes(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    """Return indices whose full published lookback has no hidden missing bar.

    GC inputs deliberately contain only the registered intraday window.  A
    state may span adjacent supplied sessions, but every step *inside* a
    session must be exact.  The session seam itself is explicit and never used
    by a reward horizon or an action lock.
    """

    if len(frame) < spec.lookback:
        return np.empty(0, dtype=np.int64)
    times = frame["event_time"].array.asi8
    sessions = _session_boundaries(frame)
    states = _state_ids(frame)
    exact = spec.minutes * NS_PER_MINUTE
    valid_step = np.ones(len(frame), dtype=bool)
    for index in range(1, len(frame)):
        if states[index] != states[index - 1]:
            valid_step[index] = False
        elif sessions[index] == sessions[index - 1]:
            valid_step[index] = int(times[index] - times[index - 1]) == exact
        else:
            # Restricted-session inputs omit the overnight interval by
            # contract.  Ordinary weekends/holidays may join causal history,
            # but a missing phase or month is a hard state reset.  This keeps
            # Jan--Apr context useful without requiring May to test in June.
            previous_day = pd.Timestamp(sessions[index - 1])
            current_day = pd.Timestamp(sessions[index])
            elapsed_days = int((current_day - previous_day).days)
            valid_step[index] = 0 < elapsed_days <= MAX_RESTRICTED_SESSION_SEAM_DAYS
    invalid_prefix = np.cumsum(~valid_step)
    result: list[int] = []
    for index in range(spec.lookback - 1, len(frame)):
        start = index - spec.lookback + 1
        invalid = int(invalid_prefix[index] - invalid_prefix[start])
        if invalid == 0:
            result.append(index)
    return np.asarray(result, dtype=np.int64)


def _decision_indexes(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    """Return every causal action-lock renewal, including unmatured late ones."""

    eligible = _contiguous_state_indexes(frame, spec)
    if not len(eligible):
        return eligible
    times = frame["event_time"].array.asi8
    sessions = _session_boundaries(frame)
    states = _state_ids(frame)
    session_end = (
        frame["session_end_ns"].to_numpy(dtype=np.int64)
        if "session_end_ns" in frame else None
    )
    decisions: list[int] = []
    remaining = 0
    previous: int | None = None
    for raw_index in eligible:
        index = int(raw_index)
        if session_end is not None and int(times[index]) > int(session_end[index]) - 5 * NS_PER_MINUTE:
            continue
        reset = (
            previous is None
            or sessions[index] != sessions[previous]
            or states[index] != states[previous]
            or index != previous + 1
            or int(times[index] - times[previous]) != spec.minutes * NS_PER_MINUTE
        )
        if reset:
            remaining = 0
        if remaining == 0:
            decisions.append(index)
            remaining = spec.action_lock_bars
        remaining -= 1
        previous = index
    return np.asarray(decisions, dtype=np.int64)


def _reward_maturities(
    frame: pd.DataFrame, spec: AgentSpec, decision_indexes: np.ndarray,
) -> dict[int, int]:
    """Map reward-eligible renewals to exact same-boundary future states."""

    valid_state = np.zeros(len(frame), dtype=bool)
    valid_state[_contiguous_state_indexes(frame, spec)] = True
    times = frame["event_time"].array.asi8
    sessions = _session_boundaries(frame)
    states = _state_ids(frame)
    exact = spec.action_lock_bars * spec.minutes * NS_PER_MINUTE
    result: dict[int, int] = {}
    targets = times[decision_indexes] + exact
    locations = np.searchsorted(times, targets, side="left")
    for raw_index, target, raw_future in zip(
        decision_indexes, targets, locations, strict=True,
    ):
        index = int(raw_index)
        future = int(raw_future)
        if (
            future < len(times)
            and int(times[future]) == int(target)
            and bool(valid_state[future])
            and sessions[future] == sessions[index]
            and states[future] == states[index]
        ):
            result[index] = int(future)
    return result


def _paper_pretrain_decision_indexes(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    return _decision_indexes(frame, spec)


def _paper_pretrain_indexes(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    """Compatibility helper: return reward-bearing XAU renewal indices."""

    decisions = _paper_pretrain_decision_indexes(frame, spec)
    return np.asarray(list(_reward_maturities(frame, spec, decisions)), dtype=np.int64)


def _gc_decision_indexes(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    return _decision_indexes(frame, spec)


def _gc_indexes(frame: pd.DataFrame, spec: AgentSpec) -> np.ndarray:
    """Compatibility helper: return reward-bearing GC renewal indices."""

    decisions = _gc_decision_indexes(frame, spec)
    return np.asarray(list(_reward_maturities(frame, spec, decisions)), dtype=np.int64)


def _inherited_at(
    times: np.ndarray, slower_times: np.ndarray, positions: np.ndarray,
    *, hold_ns: int | None = None,
) -> np.ndarray:
    indexes = np.searchsorted(slower_times, times, side="right") - 1
    safe = np.maximum(indexes, 0)
    valid = indexes >= 0
    if hold_ns is not None:
        valid &= times < slower_times[safe] + int(hold_ns)
    return np.where(valid, positions[safe], 0).astype(np.int8)


def _parent_trace_digest(
    traces: Mapping[str, tuple[np.ndarray, np.ndarray]], parents: Sequence[str],
) -> str:
    payload = {
        name: {
            "times_ns": np.asarray(traces[name][0], dtype=np.int64).tolist(),
            "actions": np.asarray(traces[name][1], dtype=np.int8).tolist(),
        }
        for name in parents
    }
    return _canonical_digest(payload)


def _greedy_actions(trainer: DQNTrainer, observations: Iterable[np.ndarray]) -> np.ndarray:
    output: list[int] = []
    trainer.online.eval()
    with torch.no_grad():
        for observation in observations:
            values = trainer.online(
                torch.as_tensor(observation, dtype=torch.float32, device=trainer.device).unsqueeze(0)
            )
            output.append(int(ACTIONS[int(values.argmax(dim=1).item())]))
    trainer.online.train()
    return np.asarray(output, dtype=np.int8)


def _observation_at(
    frame: pd.DataFrame,
    spec: AgentSpec,
    index: int,
    inherited_arrays: Sequence[np.ndarray],
) -> np.ndarray:
    inherited = tuple(int(value[index]) for value in inherited_arrays)
    return encode_window(
        frame.iloc[index - spec.lookback + 1:index + 1],
        inherited_positions=inherited,
    )


def _observations(
    frame: pd.DataFrame,
    spec: AgentSpec,
    indexes: np.ndarray,
    inherited_arrays: Sequence[np.ndarray],
    *,
    maturities: Mapping[int, int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    current: list[np.ndarray] = []
    following: list[np.ndarray] = []
    for raw_index in indexes:
        index = int(raw_index)
        next_index = index if maturities is None else int(maturities[index])
        current.append(_observation_at(frame, spec, index, inherited_arrays))
        following.append(_observation_at(frame, spec, next_index, inherited_arrays))
    return current, following


def _run_interleaved_training_pass(
    trainer: DQNTrainer,
    *,
    event_indexes: np.ndarray,
    decision_indexes: np.ndarray,
    maturities: Mapping[int, int],
    observation_at: Callable[[int], np.ndarray],
    reward_for: Callable[[int, int, int], float],
    before_event: Callable[[int], None] | None = None,
    after_action: Callable[[int, int], None] | None = None,
) -> tuple[dict[int, int], list[float], list[float]]:
    """Act, then reveal matured rewards and update at each causal state event."""

    decisions = set(int(value) for value in decision_indexes)
    matured_at: dict[int, list[int]] = {}
    for origin, future in maturities.items():
        matured_at.setdefault(int(future), []).append(int(origin))
    actions: dict[int, int] = {}
    decision_observations: dict[int, np.ndarray] = {}
    rewards: list[float] = []
    losses: list[float] = []
    for raw_index in event_indexes:
        index = int(raw_index)
        if before_event is not None:
            before_event(index)
        current_observation = observation_at(index)
        if index in decisions:
            action = int(trainer.act(current_observation, explore=True))
            actions[index] = action
            decision_observations[index] = current_observation
            if after_action is not None:
                after_action(index, action)
        for origin in matured_at.get(index, []):
            action = actions[origin]
            reward = float(reward_for(origin, index, action))
            trainer.append(
                decision_observations[origin], action, reward,
                current_observation, terminal=False,
            )
            loss = trainer.update()
            if loss is not None:
                losses.append(float(loss))
            rewards.append(reward)
    return actions, rewards, losses


def _serialize_replay(trainer: DQNTrainer) -> list[dict[str, object]]:
    items = list(trainer.replay._items)[-WARMSTART_REPLAY_ITEMS:]  # noqa: SLF001
    return [{
        "observation": item.observation,
        "action_index": int(item.action_index),
        "reward": float(item.reward),
        "next_observation": item.next_observation,
        "terminal": bool(item.terminal),
    } for item in items]


def _save_agent_state(
    path: Path,
    trainer: DQNTrainer,
    *,
    stage: str,
    variant: BaselineVariant,
    source_digest: str,
    trace_times: np.ndarray,
    trace_actions: np.ndarray,
    report: Mapping[str, object],
    run_identity_sha256: str,
    parent_trace_sha256: str,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "V6_BASELINE_AGENT_STATE",
        "complete": True,
        "stage": stage,
        "variant": variant,
        "seed": trainer.seed,
        "agent": trainer.agent.name,
        "contract_sha256": trainer.contract.sha256,
        "source_digest": source_digest,
        "run_identity_sha256": run_identity_sha256,
        "parent_trace_sha256": parent_trace_sha256,
        "online": trainer.online.state_dict(),
        "target": trainer.target.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "updates": trainer.updates,
        "decisions": trainer.decisions,
        "replay_items": _serialize_replay(trainer),
        "replay_population": min(len(trainer.replay), WARMSTART_REPLAY_ITEMS),
        "trainer_rng": trainer._rng.bit_generator.state,  # noqa: SLF001
        "replay_rng": trainer.replay._rng.bit_generator.state,  # noqa: SLF001
        "trace_times_ns": np.asarray(trace_times, dtype=np.int64),
        "trace_actions": np.asarray(trace_actions, dtype=np.int8),
        "report": dict(report),
    }
    return _atomic_torch(path, payload)


def _load_agent_state(
    path: Path,
    trainer: DQNTrainer,
    *,
    stage: str,
    variant: BaselineVariant,
    source_digest: str,
    restore_training_state: bool,
    run_identity_sha256: str,
    parent_trace_sha256: str,
) -> dict[str, object]:
    value = torch.load(path, map_location=trainer.device, weights_only=False)
    expected = (
        "V6_BASELINE_AGENT_STATE", True, stage, variant, trainer.seed,
        trainer.agent.name, trainer.contract.sha256, source_digest,
        run_identity_sha256, parent_trace_sha256,
    )
    actual = (
        value.get("artifact_type"), value.get("complete"), value.get("stage"),
        value.get("variant"), value.get("seed"), value.get("agent"),
        value.get("contract_sha256"), value.get("source_digest"),
        value.get("run_identity_sha256"), value.get("parent_trace_sha256"),
    )
    if actual != expected:
        raise ValueError(f"V6 baseline agent-state identity differs for {path}")
    trainer.online.load_state_dict(value["online"])
    trainer.target.load_state_dict(value["target"])
    if restore_training_state:
        trainer.optimizer.load_state_dict(value["optimizer"])
        trainer.updates = int(value["updates"])
        trainer.decisions = int(value["decisions"])
        trainer.replay._items.clear()  # noqa: SLF001
        for item in value["replay_items"]:
            trainer.replay.append(ReplayItem(
                np.asarray(item["observation"], dtype=np.float32),
                int(item["action_index"]), float(item["reward"]),
                np.asarray(item["next_observation"], dtype=np.float32),
                bool(item["terminal"]),
            ))
        trainer._rng.bit_generator.state = value["trainer_rng"]  # noqa: SLF001
        trainer.replay._rng.bit_generator.state = value["replay_rng"]  # noqa: SLF001
    trace_times = np.asarray(value["trace_times_ns"], dtype=np.int64)
    trace_actions = np.asarray(value["trace_actions"], dtype=np.int8)
    if trace_times.shape != trace_actions.shape or not np.isin(trace_actions, ACTIONS).all():
        raise ValueError("saved V6 baseline policy trace is malformed")
    return value


def _manifest_artifacts(root: Path, agents: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for record in agents:
        relative = str(record["state_path"])
        result.append({"path": relative, "sha256": sha256_file(root / relative)})
    return result


def verify_artifact_manifest(
    root: Path,
    *,
    artifact_type: str | None = None,
    variant: str | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Verify identity and every immutable product hash in a completed run."""

    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"V6 baseline manifest is absent at {root}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("complete") is not True:
        raise ValueError("V6 baseline manifest is incomplete or has the wrong schema")
    if artifact_type is not None and value.get("artifact_type") != artifact_type:
        raise ValueError("V6 baseline artifact type differs")
    if variant is not None and value.get("variant") != _normalise_variant(variant):
        raise ValueError("V6 baseline variant differs")
    if seed is not None and int(value.get("seed", -1)) != int(seed):
        raise ValueError("V6 baseline seed differs")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("V6 baseline manifest has no bound products")
    resolved_root = Path(root).resolve()
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("V6 baseline artifact record is malformed")
        path = (resolved_root / str(record.get("path", ""))).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("V6 baseline manifest path escapes its artifact root") from error
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"V6 baseline artifact hash differs for {path.name}")
    return value


def _existing_complete(
    output: Path, *, artifact_type: str, variant: BaselineVariant, seed: int,
    expected_identity: Mapping[str, object],
) -> Path | None:
    manifest = output / "manifest.json"
    if not manifest.exists():
        return None
    value = verify_artifact_manifest(
        output, artifact_type=artifact_type, variant=variant, seed=seed,
    )
    for key, expected in expected_identity.items():
        if value.get(key) != expected:
            raise ValueError(f"completed V6 baseline identity differs for {key}")
    return manifest


def _agent_report(value: Mapping[str, object], path: Path) -> dict[str, object]:
    report = dict(value["report"])  # type: ignore[arg-type]
    report["state_path"] = path.name
    report["state_sha256"] = sha256_file(path)
    return report


def pretrain_xau(
    direct_xau_root: Path,
    output: Path,
    seed: int,
    variant: str,
    maximum_decisions: int | None = None,
) -> Path:
    """Pretrain all three paper clocks on direct 2012--2024 XAU quotes.

    ``maximum_decisions`` is implementation-smoke-only.  A capped artifact is
    permanently marked ineligible for registered GC fine-tuning.
    """

    selected = _normalise_variant(variant)
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if maximum_decisions is not None and maximum_decisions < 1:
        raise ValueError("maximum_decisions must be positive")
    smoke_only = maximum_decisions is not None
    one, source_digest, source_manifest_sha256, months = _load_xau_one_minute(
        Path(direct_xau_root), smoke_only=smoke_only,
    )
    contract = baseline_contract(selected)
    _, protocol_sha = frozen_protocol()
    run_identity = _canonical_digest({
        "artifact_type": "V6_BASELINE_XAU_PRETRAIN",
        "variant": selected,
        "seed": seed,
        "contract_sha256": contract.sha256,
        "protocol_sha256": protocol_sha,
        "source_digest": source_digest,
        "source_manifest_sha256": source_manifest_sha256,
        "pretraining_months": months,
        "maximum_decisions_smoke_limit": maximum_decisions,
    })
    completed = _existing_complete(
        Path(output), artifact_type="V6_BASELINE_XAU_PRETRAIN", variant=selected, seed=seed,
        expected_identity={
            "run_identity_sha256": run_identity,
            "source_digest": source_digest,
            "source_manifest_sha256": source_manifest_sha256,
            "pretraining_months": months,
            "maximum_decisions_smoke_limit": maximum_decisions,
        },
    )
    if completed is not None:
        return completed
    frames = _build_timeframes(one)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reports: list[dict[str, object]] = []
    for name in AGENT_ORDER:
        spec = contract.agent(name)
        frame = frames[name]
        trainer = DQNTrainer(contract, spec, seed=seed, device="cpu")
        state_path = output / f"seed_{seed}_{name.lower()}_xau.pt"
        parent_digest = _parent_trace_digest(traces, spec.inherited_positions)
        if state_path.exists():
            saved = _load_agent_state(
                state_path, trainer, stage="xau_pretrain", variant=selected,
                source_digest=source_digest, restore_training_state=True,
                run_identity_sha256=run_identity, parent_trace_sha256=parent_digest,
            )
            traces[name] = (
                np.asarray(saved["trace_times_ns"], dtype=np.int64),
                np.asarray(saved["trace_actions"], dtype=np.int8),
            )
            reports.append(_agent_report(saved, state_path))
            continue
        times = frame["event_time"].array.asi8
        inherited_arrays = [
            _inherited_at(times, traces[parent][0], traces[parent][1])
            for parent in spec.inherited_positions
        ]
        indexes = _paper_pretrain_decision_indexes(frame, spec)
        if maximum_decisions is not None:
            indexes = indexes[:maximum_decisions]
        if not len(indexes):
            raise ValueError(f"insufficient direct XAU history for {name}")
        maturities = _reward_maturities(frame, spec, indexes)
        required_indexes = np.asarray(
            sorted(set(int(value) for value in indexes).union(maturities.values())),
            dtype=np.int64,
        )
        def gross_reward(origin: int, future_index: int, action: int) -> float:
            current = float(frame.iloc[origin]["close"])
            future = float(frame.iloc[future_index]["close"])
            return action * (future / current - 1.0)

        _, rewards, losses = _run_interleaved_training_pass(
            trainer, event_indexes=required_indexes, decision_indexes=indexes,
            maturities=maturities,
            observation_at=lambda index: _observation_at(
                frame, spec, index, inherited_arrays,
            ),
            reward_for=gross_reward,
        )
        greedy = _greedy_actions(
            trainer,
            (_observation_at(frame, spec, int(index), inherited_arrays) for index in indexes),
        )
        trace_times = times[indexes].astype(np.int64)
        report: dict[str, object] = {
            "agent": name,
            "minutes": spec.minutes,
            "lookback": spec.lookback,
            "action_lock_bars": spec.action_lock_bars,
            "decisions": len(indexes),
            "matured_transitions": len(rewards),
            "updates": trainer.updates,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "gross_fractional_reward_sum": float(np.sum(rewards)),
            "replay_population": min(len(trainer.replay), WARMSTART_REPLAY_ITEMS),
            "trace_policy": "frozen_greedy_causal",
            "training_event_order": "act_then_mature_append_update_at_each_causal_state",
            "parent_trace_sha256": parent_digest,
        }
        _save_agent_state(
            state_path, trainer, stage="xau_pretrain", variant=selected,
            source_digest=source_digest, trace_times=trace_times,
            trace_actions=greedy, report=report,
            run_identity_sha256=run_identity, parent_trace_sha256=parent_digest,
        )
        saved = _load_agent_state(
            state_path, trainer, stage="xau_pretrain", variant=selected,
            source_digest=source_digest, restore_training_state=False,
            run_identity_sha256=run_identity, parent_trace_sha256=parent_digest,
        )
        traces[name] = trace_times, greedy
        reports.append(_agent_report(saved, state_path))
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "V6_BASELINE_XAU_PRETRAIN",
        "complete": True,
        "strategy_version": "two_level_volume_publisher_v6",
        "protocol_sha256": protocol_sha,
        "name": BASELINE_NAMES[selected],
        "variant": selected,
        "seed": seed,
        "contract_sha256": contract.sha256,
        "run_identity_sha256": run_identity,
        "source_digest": source_digest,
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest": str(Path(direct_xau_root) / "manifest.json"),
        "pretraining_months": months,
        "pretraining_range": ["2012-01-01", "2024-12-31"],
        "source_kind": "Dukascopy OTC tick-by-tick midpoint quotes",
        "fifth_channel": "quote_count_proxy_not_exchange_traded_volume",
        "proxy_disclosure": (
            "quote_count counts Dukascopy OTC quote updates; it is not COMEX executed volume "
            "and is used only as the paper-style fifth OHLCV channel"
        ),
        "pretraining_reward": "signed gross fractional midpoint horizon return",
        "timeframe_construction": "complete_M5_then_exact_3x_M15_and_12x_H1_rollups_v1",
        "lookback_continuity": "exact_steps_with_UTC_day_action_boundaries_v1",
        "decision_schedule": "per_UTC_day_action_lock_reset_all_renewals_v1",
        "parent_trace_training_difference": (
            "downstream agents receive a post-training greedy causal parent trace, "
            "not the exploratory behavior trace used while the parent learned"
        ),
        "maximum_decisions_smoke_limit": maximum_decisions,
        "smoke_only": smoke_only,
        "agents": reports,
    }
    result["artifacts"] = _manifest_artifacts(output, reports)
    return _atomic_json(output / "manifest.json", result)


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
            raise ValueError(f"economic GC shard lacks {sorted(missing)}")
        return {name: source[name] for name in required}


@dataclass(frozen=True)
class LedgerReplay:
    result: V6SessionResult
    attributed_reward_cents: tuple[int, ...]
    reversal_flatten_decisions: int


def _causal_replay_indexes(
    raw: Mapping[str, np.ndarray], *, session_start_ns: int, session_end_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    boundaries = np.arange(
        int(session_start_ns) + 5 * NS_PER_MINUTE,
        int(session_end_ns),
        5 * NS_PER_MINUTE,
        dtype=np.int64,
    )
    indexes = np.searchsorted(clock, boundaries, side="right") - 1
    indexes = np.unique(indexes[indexes >= 0].astype(np.int64))
    visible_valid = np.asarray(raw["valid"], dtype=bool)
    indexes = indexes[visible_valid[indexes]]
    terminal_index = len(clock) - 1
    if not len(indexes) or int(indexes[-1]) >= terminal_index:
        raise ValueError("session has no causal five-minute target boundaries")
    if not bool(visible_valid[terminal_index]):
        raise ValueError("mandatory terminal liquidation endpoint is not executable")
    chain = np.r_[indexes, terminal_index]
    segments = np.asarray(raw["continuity_segment"])[chain]
    if len(set(int(value) for value in segments)) != 1:
        raise ValueError("baseline replay requires a pre-split continuity segment")
    return indexes, chain


def replay_target_trace(
    raw: Mapping[str, np.ndarray],
    *,
    session_date: str,
    session_start_ns: int,
    session_end_ns: int,
    signal_times_ns: np.ndarray,
    signal_actions: np.ndarray,
    hold_minutes: int,
    economics: ExecutionEconomics | None = None,
) -> LedgerReplay:
    """Replay one target trace through the common V6 ledger.

    Rewards are attributed to the active signal for strong-baseline training;
    the returned session result remains the authoritative economic record.
    """

    times = np.asarray(signal_times_ns, dtype=np.int64)
    actions = np.asarray(signal_actions, dtype=np.int8)
    if times.ndim != 1 or times.shape != actions.shape or np.any(np.diff(times) <= 0):
        raise ValueError("baseline signal times/actions must be aligned and ordered")
    if not np.isin(actions, ACTIONS).all() or hold_minutes <= 0:
        raise ValueError("baseline trace has an invalid action or hold duration")
    economics = economics or ExecutionEconomics.from_basepaper()
    clock = np.asarray(raw["clock_ns"], dtype=np.int64)
    indexes, chain = _causal_replay_indexes(
        raw, session_start_ns=session_start_ns, session_end_ns=session_end_ns,
    )
    env = V6TradingEnvironment(raw, session_date, economics)
    attributed = np.zeros(len(times), dtype=np.int64)
    expiries = times + int(hold_minutes) * NS_PER_MINUTE
    reversal_flats = 0
    last_owner: int | None = None
    for ordinal, index in enumerate(indexes):
        boundary = int(clock[int(index)])
        signal = int(np.searchsorted(times, boundary, side="right") - 1)
        active = signal >= 0 and boundary < int(expiries[signal])
        desired = int(actions[signal]) if active else 0
        if env.inventory and desired == -env.inventory:
            applied = 0
            reversal_flats += 1
        else:
            applied = desired
        next_index = int(chain[ordinal + 1])
        terminal = ordinal == len(indexes) - 1
        transition = env.step(int(index), applied, next_index, terminal=terminal)
        # The action published at this boundary owns the following economic
        # interval.  At an exact expiry with no replacement signal, the prior
        # action owns its flatten.  ``last_owner`` also keeps a late successful
        # flatten attributable after an earlier route rejection.
        owner: int | None = signal if active else None
        if owner is None:
            prior = int(np.searchsorted(times, boundary, side="left") - 1)
            if prior >= 0 and boundary <= int(expiries[prior]):
                owner = prior
            elif last_owner is not None and (
                transition.inventory_before != 0
                or transition.inventory_after != 0
                or transition.reward_cents != 0
            ):
                owner = last_owner
        if owner is not None:
            attributed[owner] += int(transition.reward_cents)
            last_owner = owner
        elif transition.reward_cents != 0:
            raise AssertionError("nonzero baseline ledger reward has no causal signal owner")
    result = env.finish()
    if int(attributed.sum()) != int(result.net_pnl_cents):
        raise AssertionError("strong-baseline rewards do not reconcile to common-ledger P&L")
    return LedgerReplay(
        result=result,
        attributed_reward_cents=tuple(int(value) for value in attributed),
        reversal_flatten_decisions=reversal_flats,
    )


class _StrongSessionCursor:
    """Incremental form of ``replay_target_trace`` for online DQN updates."""

    def __init__(
        self,
        raw: Mapping[str, np.ndarray],
        *,
        session_date: str,
        session_start_ns: int,
        session_end_ns: int,
        hold_minutes: int,
        economics: ExecutionEconomics,
    ) -> None:
        self.raw = raw
        self.clock = np.asarray(raw["clock_ns"], dtype=np.int64)
        self.indexes, self.chain = _causal_replay_indexes(
            raw, session_start_ns=session_start_ns, session_end_ns=session_end_ns,
        )
        self.env = V6TradingEnvironment(raw, session_date, economics)
        self.hold_ns = int(hold_minutes) * NS_PER_MINUTE
        self.pointer = 0
        self.signals: list[tuple[int, int, int]] = []
        self.rewards: dict[int, int] = {}
        self.last_owner: int | None = None
        self.reversal_flats = 0
        self.result: V6SessionResult | None = None

    def register(self, owner: int, time_ns: int, action: int) -> None:
        if self.result is not None or action not in ACTIONS:
            raise ValueError("cannot register a signal on a finished strong-reward cursor")
        if self.signals and int(time_ns) <= self.signals[-1][1]:
            raise ValueError("strong-reward signals must be strictly ordered")
        self.signals.append((int(owner), int(time_ns), int(action)))
        self.rewards.setdefault(int(owner), 0)

    def _active(self, boundary: int) -> tuple[int | None, int]:
        if not self.signals:
            return None, 0
        signal = self.signals[-1]
        if signal[1] <= boundary < signal[1] + self.hold_ns:
            return signal[0], signal[2]
        return None, 0

    def _step(self) -> None:
        ordinal = self.pointer
        index = int(self.indexes[ordinal])
        boundary = int(self.clock[index])
        owner, desired = self._active(boundary)
        if self.env.inventory and desired == -self.env.inventory:
            applied = 0
            self.reversal_flats += 1
        else:
            applied = desired
        terminal = ordinal == len(self.indexes) - 1
        transition = self.env.step(
            index, applied, int(self.chain[ordinal + 1]), terminal=terminal,
        )
        if owner is None:
            # Match batch replay attribution at exact expiry and after a route
            # rejection: the most recent causal signal owns its eventual exit.
            prior = self.signals[-1] if self.signals else None
            if prior is not None and boundary <= prior[1] + self.hold_ns:
                owner = prior[0]
            elif self.last_owner is not None and (
                transition.inventory_before != 0
                or transition.inventory_after != 0
                or transition.reward_cents != 0
            ):
                owner = self.last_owner
        if owner is not None:
            self.rewards[owner] = self.rewards.get(owner, 0) + int(transition.reward_cents)
            self.last_owner = owner
        elif transition.reward_cents != 0:
            raise AssertionError("nonzero online strong reward has no causal owner")
        self.pointer += 1
        if terminal:
            self.result = self.env.finish()

    def advance_before(self, time_ns: int) -> None:
        while (
            self.pointer < len(self.indexes)
            and int(self.clock[int(self.indexes[self.pointer])]) < int(time_ns)
        ):
            self._step()

    def finish(self) -> V6SessionResult:
        while self.pointer < len(self.indexes):
            self._step()
        if self.result is None:
            self.result = self.env.finish()
        if sum(self.rewards.values()) != int(self.result.net_pnl_cents):
            raise AssertionError("online strong rewards do not reconcile to common-ledger P&L")
        return self.result


class _StrongRewardPass:
    """Session-bound online ledger used by one agent training presentation."""

    def __init__(
        self,
        pairs: Sequence[SessionPair],
        *,
        hold_minutes: int,
        economics: ExecutionEconomics,
    ) -> None:
        self.pairs = {pair.day: pair for pair in pairs}
        self.hold_minutes = int(hold_minutes)
        self.economics = economics
        self.current_day: str | None = None
        self.current: _StrongSessionCursor | None = None
        self.net_pnl_cents = 0
        self.reversal_flats = 0

    def _finish_current(self) -> None:
        if self.current is None:
            return
        self.net_pnl_cents += int(self.current.finish().net_pnl_cents)
        self.reversal_flats += self.current.reversal_flats
        self.current = None

    def _cursor(self, day: str) -> _StrongSessionCursor:
        if self.current_day == day and self.current is not None:
            return self.current
        if self.current_day is not None and day <= self.current_day:
            raise ValueError("strong-reward sessions must be consumed in date order")
        self._finish_current()
        pair = self.pairs[day]
        start_ns, end_ns = _load_side_bounds(pair)
        raw = _economic_raw(pair.v2_path, self.economics)
        self.current_day = day
        self.current = _StrongSessionCursor(
            raw, session_date=day, session_start_ns=start_ns,
            session_end_ns=end_ns, hold_minutes=self.hold_minutes,
            economics=self.economics,
        )
        return self.current

    def before_decision(self, day: str, time_ns: int) -> None:
        self._cursor(day).advance_before(time_ns)

    def register(self, day: str, owner: int, time_ns: int, action: int) -> None:
        self._cursor(day).register(owner, time_ns, action)

    def reward_ticks(self, day: str, owner: int) -> float:
        return float(self._cursor(day).rewards.get(owner, 0)) / 1_000.0

    def finish(self) -> tuple[int, int]:
        self._finish_current()
        return self.net_pnl_cents, self.reversal_flats


def _after_cost_rewards(
    pairs: Sequence[SessionPair],
    signal_times: np.ndarray,
    actions: np.ndarray,
    *,
    hold_minutes: int,
    economics: ExecutionEconomics,
) -> np.ndarray:
    rewards = np.zeros(len(signal_times), dtype=np.int64)
    for pair in pairs:
        # The date guard ran before this function and before either file read.
        start_ns, end_ns = _load_side_bounds(pair)
        economic = _economic_raw(pair.v2_path, economics)
        replay = replay_target_trace(
            economic,
            session_date=pair.day,
            session_start_ns=start_ns,
            session_end_ns=end_ns,
            signal_times_ns=signal_times,
            signal_actions=actions,
            hold_minutes=hold_minutes,
            economics=economics,
        )
        rewards += np.asarray(replay.attributed_reward_cents, dtype=np.int64)
    return rewards.astype(np.float64) / 1_000.0


def _load_pretrained_agent(
    pretrained_root: Path,
    manifest: Mapping[str, object],
    trainer: DQNTrainer,
    *,
    selected: BaselineVariant,
) -> dict[str, object]:
    record = {str(value["agent"]): value for value in manifest["agents"]}[trainer.agent.name]  # type: ignore[index]
    path = pretrained_root / str(record["state_path"])
    value = _load_agent_state(
        path, trainer, stage="xau_pretrain", variant=selected,
        source_digest=str(manifest["source_digest"]), restore_training_state=False,
        run_identity_sha256=str(manifest["run_identity_sha256"]),
        parent_trace_sha256=str(record["parent_trace_sha256"]),
    )
    trainer.online.load_state_dict(value["online"])
    trainer.target.load_state_dict(trainer.online.state_dict())
    # Every GC stage receives a fresh registered optimizer, exploration clock,
    # and RNG.  Only the paper arm may keep 512 same-unit gross XAU transitions.
    optimizer_type = (
        torch.optim.RMSprop if trainer.contract.optimization.optimizer == "rmsprop"
        else torch.optim.Adam
    )
    trainer.optimizer = optimizer_type(
        trainer.online.parameters(), lr=trainer.contract.optimization.learning_rate,
    )
    trainer.updates = 0
    trainer.decisions = 0
    trainer.replay._items.clear()  # noqa: SLF001
    if selected == BASE_PAPER_VARIANT:
        for item in value["replay_items"]:
            trainer.replay.append(ReplayItem(
                np.asarray(item["observation"], dtype=np.float32),
                int(item["action_index"]), float(item["reward"]),
                np.asarray(item["next_observation"], dtype=np.float32),
                bool(item["terminal"]),
            ))
    return value


def fine_tune_gc(
    sessions: Sequence[SessionPair | tuple[Path, Path]],
    pretrained_root: Path,
    output: Path,
    seed: int,
    variant: str,
    phase: str = "train",
    passes: int = REGISTERED_GC_PASSES,
    maximum_decisions: int | None = None,
    dataset_manifest: Path | None = None,
) -> Path:
    """Fine-tune one seeded comparator on date-truncated GC sessions."""

    if phase not in {"train", "smoke"}:
        raise PermissionError("baseline fine-tuning is restricted to train or contaminated smoke")
    if phase == "train" and dataset_manifest is None:
        raise PermissionError(
            "registered baseline fine-tuning requires the exact training dataset manifest"
        )
    if phase == "smoke" and dataset_manifest is not None:
        raise ValueError("smoke fine-tuning must not claim a registered dataset proof")
    if phase == "smoke" and maximum_decisions is None:
        raise ValueError("unproved smoke fine-tuning requires a decision cap")
    selected = _normalise_variant(variant)
    if phase == "train":
        pairs = assert_training_sessions(sessions)  # Must precede every file read.
    else:
        pairs = _pairs(sessions)
        classify_dates([pair.day for pair in pairs], phase="smoke")
        if any(not "2025-01-01" <= pair.day <= "2025-08-31" for pair in pairs):
            raise PermissionError("baseline smoke fine-tuning is limited to contaminated Jan-Aug 2025")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if passes != REGISTERED_GC_PASSES and maximum_decisions is None:
        raise ValueError("registered V6 comparator fine-tuning uses exactly three passes")
    if passes < 1 or (maximum_decisions is not None and maximum_decisions < 1):
        raise ValueError("fine-tuning passes/decision cap must be positive")
    dataset_proof = (
        _verify_phase_dataset_binding(Path(dataset_manifest), pairs, phase="train")
        if dataset_manifest is not None else None
    )
    pretrained_root = Path(pretrained_root)
    pretrained = verify_artifact_manifest(
        pretrained_root, artifact_type="V6_BASELINE_XAU_PRETRAIN",
        variant=selected, seed=seed,
    )
    if pretrained.get("smoke_only") is True and maximum_decisions is None:
        raise ValueError("registered GC training refuses a capped XAU smoke warm-start")
    one, source_digest, source_records = _load_gc_one_minute(pairs)
    contract = baseline_contract(selected)
    _, protocol_sha = frozen_protocol()
    pretraining_manifest_sha256 = sha256_file(pretrained_root / "manifest.json")
    run_identity = _canonical_digest({
        "artifact_type": "V6_BASELINE_GC_FINETUNE",
        "variant": selected,
        "seed": seed,
        "contract_sha256": contract.sha256,
        "protocol_sha256": protocol_sha,
        "source_digest": source_digest,
        "training_dates": [pair.day for pair in pairs],
        "phase": phase,
        "passes": passes,
        "maximum_decisions_smoke_limit": maximum_decisions,
        "pretraining_manifest_sha256": pretraining_manifest_sha256,
        "pretraining_source_digest": pretrained.get("source_digest"),
        "pretraining_source_manifest_sha256": pretrained.get("source_manifest_sha256"),
        "training_dataset_proof": dataset_proof,
    })
    completed = _existing_complete(
        Path(output), artifact_type="V6_BASELINE_GC_FINETUNE", variant=selected, seed=seed,
        expected_identity={
            "run_identity_sha256": run_identity,
            "source_digest": source_digest,
            "gc_training_dates": [pair.day for pair in pairs],
            "phase": phase,
            "passes": passes,
            "maximum_decisions_smoke_limit": maximum_decisions,
            "pretraining_manifest_sha256": pretraining_manifest_sha256,
            "training_dataset_proof": dataset_proof,
        },
    )
    if completed is not None:
        return completed
    frames = _build_timeframes(one)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    economics = ExecutionEconomics.from_basepaper()
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reports: list[dict[str, object]] = []
    for name in AGENT_ORDER:
        spec = contract.agent(name)
        frame = frames[name]
        trainer = DQNTrainer(contract, spec, seed=seed, device="cpu")
        state_path = output / f"seed_{seed}_{name.lower()}_gc.pt"
        parent_digest = _parent_trace_digest(traces, spec.inherited_positions)
        if state_path.exists():
            saved = _load_agent_state(
                state_path, trainer, stage="gc_finetune", variant=selected,
                source_digest=source_digest, restore_training_state=True,
                run_identity_sha256=run_identity, parent_trace_sha256=parent_digest,
            )
            traces[name] = (
                np.asarray(saved["trace_times_ns"], dtype=np.int64),
                np.asarray(saved["trace_actions"], dtype=np.int8),
            )
            reports.append(_agent_report(saved, state_path))
            continue
        _load_pretrained_agent(pretrained_root, pretrained, trainer, selected=selected)
        times = frame["event_time"].array.asi8
        inherited_arrays = [
            _inherited_at(
                times, traces[parent][0], traces[parent][1],
                hold_ns=contract.agent(parent).minutes
                * contract.agent(parent).action_lock_bars * NS_PER_MINUTE,
            )
            for parent in spec.inherited_positions
        ]
        indexes = _gc_decision_indexes(frame, spec)
        if maximum_decisions is not None:
            indexes = indexes[:maximum_decisions]
        if not len(indexes):
            raise ValueError(f"insufficient GC training history for {name}")
        maturities = _reward_maturities(frame, spec, indexes)
        required_indexes = np.asarray(
            sorted(set(int(value) for value in indexes).union(maturities.values())),
            dtype=np.int64,
        )
        decision_times = times[indexes].astype(np.int64)
        losses: list[float] = []
        presentations: list[dict[str, object]] = []
        decision_ordinal = {int(index): ordinal for ordinal, index in enumerate(indexes)}
        sessions_for_frame = _session_boundaries(frame)
        for pass_index in range(passes):
            before = trainer.updates
            strong = (
                _StrongRewardPass(
                    pairs, hold_minutes=spec.minutes * spec.action_lock_bars,
                    economics=economics,
                )
                if selected == BASE_STRONG_VARIANT else None
            )
            def before_event(index: int) -> None:
                if strong is not None:
                    strong.before_decision(
                        str(sessions_for_frame[index]), int(times[index]),
                    )

            def after_action(index: int, action: int) -> None:
                if strong is not None:
                    strong.register(
                        str(sessions_for_frame[index]), decision_ordinal[index],
                        int(times[index]), action,
                    )

            def training_reward(origin: int, future_index: int, action: int) -> float:
                if strong is not None:
                    return strong.reward_ticks(
                        str(sessions_for_frame[future_index]), decision_ordinal[origin],
                    )
                current = float(frame.iloc[origin]["close"])
                future = float(frame.iloc[future_index]["close"])
                return action * (future / current - 1.0)

            _, pass_rewards, pass_losses = _run_interleaved_training_pass(
                trainer, event_indexes=required_indexes, decision_indexes=indexes,
                maturities=maturities,
                observation_at=lambda index: _observation_at(
                    frame, spec, index, inherited_arrays,
                ),
                reward_for=training_reward, before_event=before_event,
                after_action=after_action,
            )
            losses.extend(pass_losses)
            ledger_net_cents = reversal_flats = None
            if strong is not None:
                ledger_net_cents, reversal_flats = strong.finish()
                reward_unit = "exact_common_ledger_after_cost_GC_ticks"
            else:
                reward_unit = "paper_gross_fractional_horizon_return"
            presentations.append({
                "pass": pass_index + 1,
                "decisions": len(indexes),
                "transitions": len(pass_rewards),
                "reward_sum": float(np.sum(pass_rewards)),
                "updates": trainer.updates - before,
                "common_ledger_net_pnl_cents": ledger_net_cents,
                "reversal_flatten_decisions": reversal_flats,
            })
        greedy = _greedy_actions(
            trainer,
            (_observation_at(frame, spec, int(index), inherited_arrays) for index in indexes),
        )
        report = {
            "agent": name,
            "minutes": spec.minutes,
            "lookback": spec.lookback,
            "action_lock_bars": spec.action_lock_bars,
            "unique_gc_decisions": len(indexes),
            "unique_gc_transitions": len(maturities),
            "gc_transition_presentations": len(maturities) * passes,
            "updates": trainer.updates,
            "target_sync_optimizer_steps": contract.optimization.target_sync_updates,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "reward_unit": reward_unit,
            "presentations": presentations,
            "trace_policy": "frozen_greedy_causal",
            "parent_trace_sha256": parent_digest,
            "training_event_order": "act_then_mature_append_update_at_each_causal_state",
        }
        _save_agent_state(
            state_path, trainer, stage="gc_finetune", variant=selected,
            source_digest=source_digest, trace_times=decision_times,
            trace_actions=greedy, report=report,
            run_identity_sha256=run_identity, parent_trace_sha256=parent_digest,
        )
        saved = _load_agent_state(
            state_path, trainer, stage="gc_finetune", variant=selected,
            source_digest=source_digest, restore_training_state=False,
            run_identity_sha256=run_identity, parent_trace_sha256=parent_digest,
        )
        traces[name] = decision_times, greedy
        reports.append(_agent_report(saved, state_path))
    smoke_only = (
        maximum_decisions is not None
        or passes != REGISTERED_GC_PASSES
        or dataset_proof is None
        or phase != "train"
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "V6_BASELINE_GC_FINETUNE",
        "complete": True,
        "strategy_version": "two_level_volume_publisher_v6",
        "protocol_sha256": protocol_sha,
        "name": BASELINE_NAMES[selected],
        "variant": selected,
        "seed": seed,
        "contract_sha256": contract.sha256,
        "run_identity_sha256": run_identity,
        "source_digest": source_digest,
        "source_records": source_records,
        "training_dataset_proof": dataset_proof,
        "gc_training_dates": [pair.day for pair in pairs],
        "phase": phase,
        "gc_training_range_limit": [GC_TRAIN_START, GC_TRAIN_END],
        "passes": passes,
        "optimizer": contract.optimization.optimizer,
        "loss": contract.optimization.loss,
        "target_rule": contract.optimization.target_rule,
        "network_head": contract.optimization.network_head,
        "target_sync_optimizer_steps": contract.optimization.target_sync_updates,
        "fine_tuning_reward": contract.reward,
        "pretraining_manifest": str((pretrained_root / "manifest.json").resolve()),
        "pretraining_manifest_sha256": pretraining_manifest_sha256,
        "pretraining_source_digest": pretrained.get("source_digest"),
        "pretraining_source_manifest_sha256": pretrained.get("source_manifest_sha256"),
        "pretraining_fifth_channel": "quote_count_proxy_not_exchange_traded_volume",
        "only_m5_controls_common_ledger_inventory": True,
        "h1_m15_authority": "information_only",
        "timeframe_construction": "complete_M5_then_exact_3x_M15_and_12x_H1_rollups_v1",
        "lookback_continuity": (
            "exact_intraday_steps_with_at_most_7_calendar_day_restricted_session_seams_v1"
        ),
        "decision_schedule": "per_session_action_lock_reset_all_renewals_including_unmatured_late_v1",
        "parent_trace_training_difference": (
            "downstream agents receive a post-training greedy causal parent trace, "
            "not the exploratory behavior trace used while the parent learned"
        ),
        "strong_reward_difference": (
            "BaseStrong-v6 uses stateful common-ledger reward in fixed GC-tick units; "
            "inventory is intentionally absent from the paper observation"
        ) if selected == BASE_STRONG_VARIANT else None,
        "reversal_adapter_difference": (
            "an opposite M5 target is translated to FLAT; opposite entry waits for a later renewal"
        ),
        "ensemble_eligible": not smoke_only,
        "maximum_decisions_smoke_limit": maximum_decisions,
        "smoke_only": smoke_only,
        "holdout_read": False,
        "agents": reports,
    }
    result["artifacts"] = _manifest_artifacts(output, reports)
    return _atomic_json(output / "manifest.json", result)


def _load_gc_policy(
    root: Path,
    manifest: Mapping[str, object],
    *,
    seed: int,
    name: str,
    selected: BaselineVariant,
) -> DQNTrainer:
    contract = baseline_contract(selected)
    trainer = DQNTrainer(contract, contract.agent(name), seed=seed, device="cpu")
    record = {str(value["agent"]): value for value in manifest["agents"]}[name]  # type: ignore[index]
    state = _load_agent_state(
        root / str(record["state_path"]), trainer, stage="gc_finetune",
        variant=selected, source_digest=str(manifest["source_digest"]),
        restore_training_state=False,
        run_identity_sha256=str(manifest["run_identity_sha256"]),
        parent_trace_sha256=str(record["parent_trace_sha256"]),
    )
    trainer.online.load_state_dict(state["online"])
    trainer.target.load_state_dict(state["target"])
    trainer.online.eval()
    return trainer


def _seed_trace(
    root: Path,
    manifest: Mapping[str, object],
    frames: Mapping[str, pd.DataFrame],
    *,
    seed: int,
    selected: BaselineVariant,
) -> tuple[np.ndarray, np.ndarray]:
    contract = baseline_contract(selected)
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in AGENT_ORDER:
        spec = contract.agent(name)
        frame = frames[name]
        policy = _load_gc_policy(
            root, manifest, seed=seed, name=name, selected=selected,
        )
        times = frame["event_time"].array.asi8
        inherited_arrays = [
            _inherited_at(
                times, traces[parent][0], traces[parent][1],
                hold_ns=contract.agent(parent).minutes
                * contract.agent(parent).action_lock_bars * NS_PER_MINUTE,
            )
            for parent in spec.inherited_positions
        ]
        indexes = _gc_decision_indexes(frame, spec)
        observations, _ = _observations(frame, spec, indexes, inherited_arrays)
        actions = _greedy_actions(policy, observations)
        traces[name] = times[indexes].astype(np.int64), actions
    return traces["M5"]


def _context_guard(
    context: Sequence[SessionPair], evaluation: Sequence[SessionPair], phase: EvaluationPhase,
) -> None:
    evaluation_days = {value.day for value in evaluation}
    context_days = {value.day for value in context}
    if not evaluation_days.issubset(context_days):
        raise ValueError("every evaluated session must be included in causal baseline context")
    latest = {"smoke": "2025-08-31", "validation": "2026-05-31",
              "test": "2026-08-31"}[phase]
    if any(value.day > latest for value in context):
        raise PermissionError("baseline context reaches beyond the authorized evaluation phase")


def evaluate_ensemble(
    checkpoint_roots: Sequence[Path],
    context_sessions: Sequence[SessionPair | tuple[Path, Path]],
    evaluation_sessions: Sequence[SessionPair | tuple[Path, Path]],
    *,
    variant: str,
    phase: EvaluationPhase,
    unlock_receipt: Path | None = None,
    economics: ExecutionEconomics | None = None,
    scenario: str = "base",
    allow_smoke_checkpoints: bool = False,
    training_dataset_manifest: Path | None = None,
    evaluation_dataset_manifest: Path | None = None,
    complete_training_manifest: Path | None = None,
) -> dict[str, object]:
    """Evaluate the five-seed M5 ensemble on the common V6 ledger."""

    selected = _normalise_variant(variant)
    if phase not in {"smoke", "validation", "test"}:
        raise PermissionError("direct comparator diagnostics must use phase='smoke'")
    # Enforce the market-data boundary and unlock before reading any checkpoint
    # or shard.  This ordering is intentional and covered by tests.
    evaluation = assert_evaluation_sessions(
        evaluation_sessions, phase=phase, unlock_receipt=unlock_receipt,
    )
    proof_arguments = (
        training_dataset_manifest,
        evaluation_dataset_manifest,
        complete_training_manifest,
    )
    supplied_proofs = [value is not None for value in proof_arguments]
    if phase in {"validation", "test"} and not all(supplied_proofs):
        raise PermissionError(
            "validation/test baseline evaluation requires the complete phase proof chain"
        )
    if phase == "smoke" and any(supplied_proofs):
        raise ValueError("contaminated smoke evaluation must not claim registered proofs")
    if phase in {"validation", "test"} and allow_smoke_checkpoints:
        raise ValueError("validation/test evaluation refuses smoke checkpoints")
    registered_proofs = phase in {"validation", "test"} and all(supplied_proofs)
    context = _pairs(context_sessions)
    _context_guard(context, evaluation, phase)
    training_proof: dict[str, object] | None = None
    evaluation_proof: dict[str, object] | None = None
    complete_sha256: str | None = None
    complete_value: dict[str, object] | None = None
    if registered_proofs:
        evaluation_days = {pair.day for pair in evaluation}
        training_context = [pair for pair in context if pair.day not in evaluation_days]
        if len(training_context) + len(evaluation) != len(context):
            raise ValueError("baseline context has an ambiguous phase overlap")
        assert training_dataset_manifest is not None
        assert evaluation_dataset_manifest is not None
        assert complete_training_manifest is not None
        training_proof = _verify_phase_dataset_binding(
            training_dataset_manifest, training_context, phase="train",
        )
        evaluation_proof = _verify_phase_dataset_binding(
            evaluation_dataset_manifest, evaluation, phase=phase,
            complete_training_manifest=complete_training_manifest,
            unlock_receipt=unlock_receipt,
        )
    registered_economics = economics_for_scenario(scenario)
    if economics is None:
        economics = registered_economics
    elif asdict(economics) != asdict(registered_economics):
        raise ValueError("scenario label and ExecutionEconomics differ from the registry")
    if len(checkpoint_roots) != len(SEEDS):
        raise ValueError("baseline evaluation requires exactly five checkpoint roots")
    records: list[tuple[int, Path, dict[str, object]]] = []
    for root_value in checkpoint_roots:
        root = Path(root_value)
        value = verify_artifact_manifest(
            root, artifact_type="V6_BASELINE_GC_FINETUNE", variant=selected,
        )
        seed = int(value.get("seed", -1))
        records.append((seed, root, value))
    records.sort(key=lambda value: value[0])
    if tuple(value[0] for value in records) != tuple(SEEDS):
        raise ValueError("baseline checkpoint ensemble seeds differ from the registry")
    if not allow_smoke_checkpoints and any(value[2].get("smoke_only") is True for value in records):
        raise ValueError("registered baseline evaluation refuses smoke checkpoints")
    contract_hashes = {str(value[2].get("contract_sha256")) for value in records}
    protocols = {str(value[2].get("protocol_sha256")) for value in records}
    if contract_hashes != {baseline_contract(selected).sha256} or len(protocols) != 1:
        raise ValueError("baseline ensemble contract/protocol identity differs")
    ensemble_identity_fields = (
        "source_digest", "gc_training_dates", "passes",
        "pretraining_source_digest", "pretraining_source_manifest_sha256",
        "training_dataset_proof",
    )
    identities = {
        _canonical_digest({field: value[2].get(field) for field in ensemble_identity_fields})
        for value in records
    }
    if len(identities) != 1:
        raise ValueError(
            "baseline ensemble members differ in source, dates, passes, XAU source, or coverage"
        )
    if training_proof is not None:
        for _, _, manifest in records:
            if (
                manifest.get("gc_training_dates") != training_proof["dates"]
                or manifest.get("training_dataset_proof") != training_proof
                or manifest.get("passes") != REGISTERED_GC_PASSES
                or manifest.get("ensemble_eligible") is not True
            ):
                raise ValueError("baseline checkpoint does not bind the registered training coverage")
        assert complete_training_manifest is not None
        complete_sha256, complete_value = _verify_complete_checkpoint_binding(
            complete_training_manifest, [value[1] for value in records],
            selected=selected,
            training_dataset_sha256=str(training_proof["sha256"]),
        )
        _verify_xau_parent_binding(
            complete_training_manifest, complete_value, records, selected=selected,
        )
    one, _, _ = _load_gc_one_minute(context)
    frames = _build_timeframes(one)
    common_times: np.ndarray | None = None
    member_actions: list[np.ndarray] = []
    for seed, root, manifest in records:
        times, actions = _seed_trace(
            root, manifest, frames, seed=seed, selected=selected,
        )
        if common_times is not None and not np.array_equal(times, common_times):
            raise ValueError("baseline seed M5 clocks differ")
        common_times = times
        member_actions.append(actions)
    if common_times is None:
        raise AssertionError("baseline ensemble produced no M5 trace")
    ensemble = majority_target_vote(np.stack(member_actions))
    hold_minutes = (
        baseline_contract(selected).agent("M5").minutes
        * baseline_contract(selected).agent("M5").action_lock_bars
    )
    daily: dict[str, int] = {}
    round_trips = rejected = reversal_flats = 0
    evaluation_lookup = {value.day: value for value in evaluation}
    for day in sorted(evaluation_lookup):
        pair = evaluation_lookup[day]
        start_ns, end_ns = _load_side_bounds(pair)
        raw = _economic_raw(pair.v2_path, economics)
        replay = replay_target_trace(
            raw,
            session_date=pair.day,
            session_start_ns=start_ns,
            session_end_ns=end_ns,
            signal_times_ns=common_times,
            signal_actions=ensemble,
            hold_minutes=hold_minutes,
            economics=economics,
        )
        daily[day] = int(replay.result.net_pnl_cents)
        round_trips += int(replay.result.completed_round_trips)
        rejected += int(replay.result.rejected_commands)
        reversal_flats += int(replay.reversal_flatten_decisions)
    classification = "FRESH_OOS" if phase == "test" else "DEVELOPMENT"
    registered = (
        phase in {"validation", "test"}
        and not allow_smoke_checkpoints
        and registered_proofs
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "V6_BASELINE_COMMON_LEDGER_REPLAY",
        "protocol_sha256": next(iter(protocols)),
        "arm": BASELINE_NAMES[selected],
        "name": BASELINE_NAMES[selected],
        "variant": selected,
        "ensemble_seeds": list(SEEDS),
        "ensemble_rule": "majority_target_vote_flat_on_every_tie_v1",
        "only_m5_controls_common_ledger_inventory": True,
        "h1_m15_authority": "information_only",
        "direct_reversal_rule": "flatten_now_then_wait_for_a_later_completed_M5_decision",
        "context_gap_rule": (
            "lookbacks reset across session gaps greater than 7 calendar days; "
            "test does not request or synthesize May context"
        ),
        "reversal_flatten_decisions": reversal_flats,
        "evaluation_dates": sorted(daily),
        "classification": classification,
        "phase": phase,
        "checkpoint_set": [{
            "seed": seed,
            "manifest_sha256": sha256_file(root / "manifest.json"),
        } for seed, root, _ in records],
        "training_dataset_proof": training_proof,
        "evaluation_dataset_proof": evaluation_proof,
        "complete_training_manifest_sha256": complete_sha256,
        "access_precondition": (
            "all five final seeded comparator manifests and artifact hashes were verified "
            "before evaluation market data was opened"
        ),
        "economics": asdict(economics),
        "scenario": scenario,
        "daily_pnl_cents": daily,
        "metrics": daily_metrics(daily, round_trips),
        "rejected_commands": rejected,
        "registered_evaluation": registered,
        "smoke_only": not registered,
        "holdout_read": phase == "test",
        "unlock_receipt": str(unlock_receipt) if phase == "test" else None,
    }
    return report


# Pipeline-friendly names used by the V6 orchestrator.
pretrain = pretrain_xau
fine_tune = fine_tune_gc
evaluate = evaluate_ensemble


__all__ = [
    "BASELINE_NAMES",
    "BASE_PAPER_VARIANT",
    "BASE_STRONG_VARIANT",
    "BaselineVariant",
    "EvaluationPhase",
    "GC_TRAIN_END",
    "GC_TRAIN_START",
    "LedgerReplay",
    "REGISTERED_GC_PASSES",
    "REGISTERED_SCENARIOS",
    "SessionPair",
    "assert_evaluation_sessions",
    "assert_training_sessions",
    "baseline_contract",
    "evaluate",
    "evaluate_ensemble",
    "economics_for_scenario",
    "fine_tune",
    "fine_tune_gc",
    "majority_target_vote",
    "pretrain",
    "pretrain_xau",
    "registered_comparator_contracts",
    "replay_target_trace",
    "sha256_file",
    "verify_artifact_manifest",
]
