"""Fail-closed orchestration for the registered two-level V6 experiment.

The functions in this module are intentionally boring.  They bind every data
shard and checkpoint through an immutable manifest, schedule the registered
five-seed arms, and make it impossible for a convenient directory glob to
cross the May-validation or June--August-test boundary.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import importlib
import json
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .volume_v6_contract import SEEDS
from .volume_v6_eval import (
    REQUIRED_ARMS,
    STRESS_SCENARIOS,
    TEST_COMPLETE_ON,
    apply_registered_gates,
    assert_test_unlocked,
    atomic_json,
    authorize_test,
    daily_metrics,
    file_sha256,
    frozen_protocol,
)


TRAIN_START = "2025-01-01"
TRAIN_END_EXCLUSIVE = "2026-05-01"
VALIDATION_START = "2026-05-01"
VALIDATION_END_EXCLUSIVE = "2026-06-01"
TEST_START = "2026-06-01"
TEST_END_EXCLUSIVE = "2026-09-01"
SCENARIOS = ("base",) + tuple(STRESS_SCENARIOS)

ARM_KEYS: dict[str, str] = {
    "V6-RichVolume": "rich_volume",
    "V6-BaseOHLCV": "base_ohlcv",
    "BasePaper-XAU-Pretrained-GC": "base_paper_gc",
    "BaseStrong-XAU-Pretrained-GC-v6": "base_strong_gc_v6",
}
HIERARCHY_ARMS = ("V6-RichVolume", "V6-BaseOHLCV")
BASELINE_ARMS = REQUIRED_ARMS[2:]
COMPLETE_TRAINING_ARTIFACT = "V6_COMPLETE_TRAINING_RUN"
COMPLETE_EVALUATION_ARTIFACT = "V6_COMPLETE_REGISTERED_EVALUATION"


@dataclass(frozen=True)
class PhaseDataset:
    manifest_path: Path
    phase: str
    dates: tuple[str, ...]
    raw_paths: tuple[Path, ...]
    sidecar_paths: tuple[Path, ...]
    manifest_sha256: str

    @property
    def pairs(self) -> tuple[tuple[Path, Path], ...]:
        return tuple(zip(self.raw_paths, self.sidecar_paths, strict=True))


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _resolve(anchor: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (anchor / path).resolve()


def _relative(anchor: Path, path: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=Path(anchor).resolve())


def _registered_bounds(phase: str) -> tuple[str, str]:
    try:
        return {
            "train": (TRAIN_START, TRAIN_END_EXCLUSIVE),
            "validation": (VALIDATION_START, VALIDATION_END_EXCLUSIVE),
            "test": (TEST_START, TEST_END_EXCLUSIVE),
        }[phase]
    except KeyError as error:
        raise ValueError("registered phase must be train, validation, or test") from error


def _current_date() -> date:
    """Private clock boundary; public access APIs cannot inject future dates."""

    return date.today()


def _assert_test_date() -> None:
    if _current_date() < TEST_COMPLETE_ON:
        raise PermissionError("the June-August 2026 test cannot open before 2026-09-01")


def load_phase_dataset(
    manifest_path: Path,
    *,
    phase: str,
    complete_training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> PhaseDataset:
    """Load only session names listed in one authorized phase manifest.

    Authorization deliberately happens before the requested dataset manifest
    is opened.  This ordering prevents even a malformed or absent locked shard
    from becoming an accidental probe of the validation/test filesystem.
    """

    if phase == "validation":
        if complete_training_manifest is None or not Path(complete_training_manifest).is_file():
            raise PermissionError("May access requires the complete V6 training manifest")
        verify_complete_training_manifest(complete_training_manifest)
    elif phase == "test":
        _assert_test_date()
        if unlock_receipt is None:
            raise PermissionError("test access requires the immutable V6 unlock receipt")
        unlock = assert_test_unlocked(unlock_receipt)
        bound_training = Path(str(unlock.get("complete_training_manifest", "")))
        _assert_test_chain_bound(unlock, bound_training)
    elif phase != "train":
        raise ValueError("phase must be train, validation, or test")

    path = Path(manifest_path).resolve()
    value = _read_object(path)
    _, protocol_sha256 = frozen_protocol()
    start, end_exclusive = _registered_bounds(phase)
    classification = "FRESH_OOS" if phase == "test" else "DEVELOPMENT"
    if (
        value.get("artifact_type") != "V6_GC_CAUSAL_SESSION_DATASET"
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("phase") != phase
        or value.get("classification") != classification
        or value.get("start") != start
        or value.get("end_exclusive") != end_exclusive
    ):
        raise ValueError("V6 dataset manifest does not match the registered phase")

    if phase == "validation":
        proof = value.get("access_authorization")
        if not isinstance(proof, dict) or proof.get("kind") != "complete_training_manifest":
            raise PermissionError("May dataset lacks its complete-training authorization")
        assert complete_training_manifest is not None
        if proof.get("sha256") != file_sha256(complete_training_manifest):
            raise PermissionError("May dataset was authorized by a different training receipt")
    elif phase == "test":
        proof = value.get("access_authorization")
        if not isinstance(proof, dict) or proof.get("kind") != "test_unlock_receipt":
            raise PermissionError("test dataset lacks its unlock authorization")
        assert unlock_receipt is not None
        if proof.get("sha256") != file_sha256(unlock_receipt):
            raise PermissionError("test dataset was authorized by a different unlock receipt")

    records = value.get("sessions")
    if not isinstance(records, dict) or not records:
        raise ValueError("V6 dataset manifest has no bound sessions")
    dates = tuple(sorted(str(day) for day in records))
    if len(dates) != len(set(dates)) or value.get("session_count") != len(dates):
        raise ValueError("V6 dataset manifest contains duplicate dates")
    map_record = value.get("active_contract_map")
    receipt_record = value.get("active_contract_map_receipt")
    if not isinstance(map_record, dict) or not isinstance(receipt_record, dict):
        raise ValueError("V6 dataset lacks its receipt-bound active-contract map")
    active_map = _resolve(path.parent, map_record.get("path"))
    active_receipt = _resolve(path.parent, receipt_record.get("path"))
    if (
        not active_map.is_file()
        or file_sha256(active_map) != map_record.get("sha256")
        or not active_receipt.is_file()
        or file_sha256(active_receipt) != receipt_record.get("sha256")
    ):
        raise ValueError("V6 dataset active-contract map/receipt hash differs")
    from .volume_v6_acquisition import validate_active_contract_map
    selected_map = validate_active_contract_map(
        active_map, start, end_exclusive, receipt_path=active_receipt, phase=phase,
        training_manifest=complete_training_manifest, unlock_receipt=unlock_receipt,
    )
    selected_rows = tuple(zip(
        (str(day) for day in selected_map["v6_session_date"]),
        (str(symbol) for symbol in selected_map["selected_raw_symbol"]),
        strict=True,
    ))
    manifest_rows = tuple(
        (day, str(records[day].get("raw_symbol", "")))
        if isinstance(records[day], dict) else (day, "")
        for day in dates
    )
    if selected_rows != manifest_rows:
        raise ValueError(
            "V6 dataset sessions/symbols are not the exact receipt-bound phase universe"
        )
    audits = value.get("audits")
    if (
        not isinstance(audits, list)
        or len(audits) != len(dates)
        or tuple(sorted(str(audit.get("session_date")) for audit in audits if isinstance(audit, dict)))
        != dates
        or any(
            not isinstance(audit, dict)
            or audit.get("status") != "PASS"
            or audit.get("continuity_segments") != 1
            for audit in audits
        )
    ):
        raise ValueError("V6 dataset session-quality receipts are incomplete")
    raw_paths: list[Path] = []
    sidecar_paths: list[Path] = []
    for day in dates:
        try:
            parsed = date.fromisoformat(day)
        except ValueError as error:
            raise ValueError("V6 dataset session key is not YYYY-MM-DD") from error
        if str(parsed) != day or not start <= day < end_exclusive:
            raise ValueError("V6 dataset session lies outside its registered phase")
        record = records[day]
        if not isinstance(record, dict):
            raise ValueError("V6 dataset session record is malformed")
        raw = path.parent / "sessions" / f"{day}.npz"
        side = path.parent / "sidecars" / f"{day}.npz"
        if (
            not raw.is_file()
            or not side.is_file()
            or file_sha256(raw) != record.get("session_sha256")
            or file_sha256(side) != record.get("sidecar_sha256")
        ):
            raise ValueError(f"V6 phase manifest hash differs for {day}")
        raw_paths.append(raw)
        sidecar_paths.append(side)
    if value.get("session_count") != len(dates):
        raise ValueError("V6 dataset session count differs from its records")
    return PhaseDataset(
        path, phase, dates, tuple(raw_paths), tuple(sidecar_paths), file_sha256(path),
    )


def build_phase_dataset(
    archive_root: Path,
    active_map: Path,
    output: Path,
    *,
    phase: str,
    active_map_receipt: Path | None = None,
    complete_training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> Path:
    """Build one exact registered phase without listing another phase."""

    start, end_exclusive = _registered_bounds(phase)
    if phase == "validation":
        if complete_training_manifest is None or not Path(complete_training_manifest).is_file():
            raise PermissionError("May build requires the complete V6 training manifest")
        verify_complete_training_manifest(complete_training_manifest)
    if phase == "test":
        _assert_test_date()
        if unlock_receipt is None:
            raise PermissionError("test build requires the immutable V6 unlock receipt")
        unlock = assert_test_unlocked(unlock_receipt)
        bound_training = Path(str(unlock.get("complete_training_manifest", "")))
        _assert_test_chain_bound(unlock, bound_training)
    from .volume_v6_acquisition import build_dataset
    return build_dataset(
        Path(archive_root), Path(active_map), Path(output), start=start,
        end_exclusive=end_exclusive, phase=phase,
        active_map_receipt=active_map_receipt,
        training_manifest=complete_training_manifest, unlock_receipt=unlock_receipt,
    )


def register_active_map(
    active_map: Path, source_manifest: Path, output: Path | None = None, *,
    complete_training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> Path:
    """Validate and sign the past-only active-contract map used by V6 build."""

    from .volume_v6_acquisition import create_active_map_receipt
    return create_active_map_receipt(
        Path(active_map), Path(source_manifest), None if output is None else Path(output),
        training_manifest=complete_training_manifest, unlock_receipt=unlock_receipt,
    )


def create_active_map_universe(
    active_map: Path,
    archive_root: Path,
    output: Path,
    *,
    phase: str,
    start: str,
    end_exclusive: str,
    complete_training_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> Path:
    """Create the authoritative child-hash inventory required before map signing."""

    from .volume_v6_acquisition import create_eligible_phase_universe

    return create_eligible_phase_universe(
        Path(active_map), Path(archive_root), Path(output), phase=phase,
        start=start, end_exclusive=end_exclusive,
        training_manifest=complete_training_manifest, unlock_receipt=unlock_receipt,
    )


def _pretrain_job(
    kind: str, direct_xau: str, output: str, seed: int,
    maximum_sessions: int | None, maximum_decisions: int | None,
) -> str:
    from .volume_v6_parallel import configure_worker
    configure_worker(1)
    destination = Path(output)
    if kind == "v6_price":
        try:
            module = importlib.import_module(".volume_v6_xau", package=__package__)
        except ImportError as error:
            raise RuntimeError("the optional V6 XAU price-pretraining module is unavailable") from error
        function = getattr(module, "pretrain_xau", None)
        if function is None:
            function = getattr(module, "pretrain_xau_price_encoders", None)
        if function is None:
            raise RuntimeError("volume_v6_xau exports no supported pretraining entry point")
        result = function(
            Path(direct_xau), destination, seed=seed, passes=1,
            maximum_sessions=maximum_sessions,
            minimum_sessions=(
                1 if maximum_sessions is not None
                else int(getattr(module, "MINIMUM_FULL_SESSIONS", 2_500))
            ),
        )
    else:
        from .volume_v6_baseline import pretrain_xau
        result = pretrain_xau(
            Path(direct_xau), destination, seed=seed, variant=kind,
            maximum_decisions=maximum_decisions,
        )
    return str(result)


def _run_jobs(
    jobs: Sequence[tuple[object, ...]], worker: Callable[..., str], workers: int,
) -> list[str]:
    if not 1 <= workers <= 12:
        raise ValueError("V6 laptop worker count must be between 1 and 12")
    results: list[str] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {executor.submit(worker, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    return sorted(results)


def pretrain_all_xau(
    direct_xau: Path,
    output: Path,
    *,
    workers: int = 8,
    maximum_sessions: int | None = None,
    maximum_decisions: int | None = None,
) -> Path:
    """Pretrain V6 base encoders and both paper-style comparators for all seeds."""

    if not 1 <= workers <= 12:
        raise ValueError("V6 laptop worker count must be between 1 and 12")
    smoke_only = maximum_sessions is not None or maximum_decisions is not None
    if (maximum_sessions is None) != (maximum_decisions is None):
        raise ValueError("XAU smoke caps must be supplied together")
    direct_manifest = Path(direct_xau).resolve() / "manifest.json"
    if not direct_manifest.is_file():
        raise ValueError("direct XAU product manifest is absent")
    direct_manifest_sha256 = file_sha256(direct_manifest)
    root = Path(output).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        verify_pretraining_manifest(
            manifest_path, allow_smoke=smoke_only,
            expected_source_manifest_sha256=direct_manifest_sha256,
        )
        return manifest_path
    root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[object, ...]] = []
    for kind in ("v6_price", "base_paper_gc", "base_strong_gc_v6"):
        for seed in SEEDS:
            jobs.append((
                kind, str(Path(direct_xau).resolve()), str(root / kind / f"seed_{seed}"),
                seed, maximum_sessions, maximum_decisions,
            ))
    _run_jobs(jobs, _pretrain_job, workers)
    _, protocol_sha256 = frozen_protocol()
    records: list[dict[str, object]] = []
    for kind in ("v6_price", "base_paper_gc", "base_strong_gc_v6"):
        for seed in SEEDS:
            one = root / kind / f"seed_{seed}" / "manifest.json"
            if not one.is_file():
                raise RuntimeError(f"XAU pretraining did not produce {kind} seed {seed}")
            records.append({
                "kind": kind, "seed": seed,
                "manifest": _relative(root, one), "manifest_sha256": file_sha256(one),
            })
    value = {
        "schema_version": 1,
        "artifact_type": "V6_COMPLETE_XAU_PRETRAINING",
        "protocol_sha256": protocol_sha256,
        "complete": True,
        "smoke_only": smoke_only,
        "seeds": list(SEEDS),
        "components": ["v6_price", "base_paper_gc", "base_strong_gc_v6"],
        "direct_xau_manifest": {
            "path": str(direct_manifest), "sha256": direct_manifest_sha256,
        },
        "artifacts": records,
    }
    atomic_json(manifest_path, value)
    verify_pretraining_manifest(
        manifest_path, allow_smoke=smoke_only,
        expected_source_manifest_sha256=direct_manifest_sha256,
    )
    return manifest_path


def verify_pretraining_manifest(
    path: Path, *, allow_smoke: bool = False,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, object]:
    value = _read_object(Path(path))
    _, protocol_sha256 = frozen_protocol()
    if (
        value.get("artifact_type") != "V6_COMPLETE_XAU_PRETRAINING"
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("complete") is not True
        or value.get("seeds") != list(SEEDS)
        or value.get("components") != ["v6_price", "base_paper_gc", "base_strong_gc_v6"]
        or (value.get("smoke_only") is True and not allow_smoke)
    ):
        raise ValueError("V6 complete XAU pretraining manifest is invalid")
    records = value.get("artifacts")
    if not isinstance(records, list) or len(records) != 3 * len(SEEDS):
        raise ValueError("V6 XAU pretraining manifest is incomplete")
    source_record = value.get("direct_xau_manifest")
    if not isinstance(source_record, dict):
        raise ValueError("V6 XAU pretraining lacks its direct source manifest")
    source_path = _resolve(Path(path).parent, source_record.get("path"))
    source_sha256 = str(source_record.get("sha256", ""))
    if (
        not source_path.is_file()
        or file_sha256(source_path) != source_sha256
        or (
            expected_source_manifest_sha256 is not None
            and source_sha256 != expected_source_manifest_sha256
        )
    ):
        raise ValueError("V6 XAU pretraining binds a different direct source product")
    identities: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("V6 XAU pretraining record is malformed")
        identity = (str(record.get("kind")), int(record.get("seed", -1)))
        identities.add(identity)
        one = _resolve(Path(path).parent, record.get("manifest"))
        if not one.is_file() or file_sha256(one) != record.get("manifest_sha256"):
            raise ValueError("V6 XAU pretraining artifact hash differs")
        if identity[0] == "v6_price":
            product = _read_object(one)
            if (
                product.get("artifact_type") != "V6_XAU_PRICE_PRETRAINING"
                or product.get("protocol_sha256") != protocol_sha256
                or product.get("seed") != identity[1]
                or product.get("source_manifest_sha256") != source_sha256
                or (
                    not allow_smoke
                    and product.get("gc_transfer_eligible") is not True
                )
            ):
                raise ValueError("V6 price-only XAU pretraining identity differs")
            for name in ("model_bundle", "resume_checkpoint"):
                artifact = product.get(name)
                if not isinstance(artifact, dict):
                    raise ValueError("V6 price-only XAU manifest is incomplete")
                payload = (one.parent / str(artifact.get("path", ""))).resolve()
                try:
                    payload.relative_to(one.parent.resolve())
                except ValueError as error:
                    raise ValueError("V6 XAU artifact escapes its root") from error
                if not payload.is_file() or file_sha256(payload) != artifact.get("sha256"):
                    raise ValueError("V6 price-only XAU artifact hash differs")
        else:
            from .volume_v6_baseline import verify_artifact_manifest
            product = verify_artifact_manifest(
                one.parent, artifact_type="V6_BASELINE_XAU_PRETRAIN",
                variant=identity[0], seed=identity[1],
            )
            if product.get("protocol_sha256") != protocol_sha256:
                raise ValueError("V6 baseline XAU pretraining protocol differs")
            if product.get("source_manifest_sha256") != source_sha256:
                raise ValueError("V6 baseline XAU pretraining source differs")
            if product.get("smoke_only") is True and not allow_smoke:
                raise ValueError("registered V6 pretraining refuses a capped baseline artifact")
    expected = {
        (kind, seed)
        for kind in ("v6_price", "base_paper_gc", "base_strong_gc_v6")
        for seed in SEEDS
    }
    if identities != expected:
        raise ValueError("V6 XAU pretraining identities are incomplete")
    return value


def _baseline_train_job(
    pairs: tuple[tuple[str, str], ...], pretrained: str, output: str,
    seed: int, variant: str, dataset_manifest: str,
) -> str:
    from .volume_v6_parallel import configure_worker
    configure_worker(1)
    from .volume_v6_baseline import fine_tune_gc
    result = fine_tune_gc(
        [(Path(raw), Path(side)) for raw, side in pairs], Path(pretrained), Path(output),
        seed=seed, variant=variant, passes=3,
        dataset_manifest=Path(dataset_manifest),
    )
    return str(result)


def train_all(
    dataset_manifest: Path,
    pretraining_manifest: Path,
    output: Path,
    *,
    workers: int = 8,
) -> Path:
    """Train all 20 registered GC models and publish one completeness receipt."""

    if not 1 <= workers <= 12:
        raise ValueError("V6 laptop worker count must be between 1 and 12")
    dataset = load_phase_dataset(dataset_manifest, phase="train")
    pretraining = verify_pretraining_manifest(pretraining_manifest)
    pretrain_root = Path(pretraining_manifest).resolve().parent
    root = Path(output).resolve()
    complete = root / "complete_training_manifest.json"
    if complete.is_file():
        existing = verify_complete_training_manifest(complete)
        dataset_record = existing.get("dataset_manifest")
        pretraining_record = existing.get("pretraining_manifest")
        if (
            not isinstance(dataset_record, dict)
            or not isinstance(pretraining_record, dict)
            or _resolve(complete.parent, dataset_record.get("path"))
            != dataset.manifest_path
            or dataset_record.get("sha256") != dataset.manifest_sha256
            or _resolve(complete.parent, pretraining_record.get("path"))
            != Path(pretraining_manifest).resolve()
            or pretraining_record.get("sha256") != file_sha256(pretraining_manifest)
        ):
            raise ValueError("existing V6 training output belongs to a different request")
        return complete
    root.mkdir(parents=True, exist_ok=True)

    from .volume_v6_parallel import train_parallel
    train_parallel(
        dataset.raw_paths, dataset.sidecar_paths, root,
        pretrained_root=pretrain_root / "v6_price", workers=workers,
        phase="train", passes=3, seeds=SEEDS,
        arms=("rich_volume", "base_ohlcv"),
        dataset_manifest=dataset.manifest_path,
    )

    serial_pairs = tuple((str(raw), str(side)) for raw, side in dataset.pairs)
    jobs: list[tuple[object, ...]] = []
    for variant in ("base_paper_gc", "base_strong_gc_v6"):
        for seed in SEEDS:
            jobs.append((
                serial_pairs, str(pretrain_root / variant / f"seed_{seed}"),
                str(root / variant / f"seed_{seed}"), seed, variant,
                str(dataset.manifest_path),
            ))
    _run_jobs(jobs, _baseline_train_job, workers)
    return create_complete_training_manifest(
        root, dataset, Path(pretraining_manifest).resolve(), complete,
    )


def _artifact_root(training_root: Path, arm: str, seed: int) -> Path:
    return training_root / ARM_KEYS[arm] / f"seed_{seed}"


def _expected_pretraining_parent(
    aggregate_manifest: Path, *, kind: str, seed: int,
) -> tuple[Path, str]:
    aggregate_path = Path(aggregate_manifest).resolve()
    value = _read_object(aggregate_path)
    records = value.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("complete XAU parent records are absent")
    matches = [
        record for record in records
        if isinstance(record, dict)
        and record.get("kind") == kind
        and record.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"complete XAU parent is absent for {kind} seed {seed}")
    manifest = _resolve(aggregate_path.parent, matches[0].get("manifest"))
    digest = str(matches[0].get("manifest_sha256", ""))
    if not manifest.is_file() or file_sha256(manifest) != digest:
        raise ValueError("complete XAU parent manifest hash differs")
    return manifest, digest


def _verify_training_artifact(
    root: Path, *, arm: str, seed: int, dates: Sequence[str],
    pretraining_manifest: Path | None = None,
    dataset_manifest: Path | None = None,
) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing V6 training manifest for {arm} seed {seed}")
    _, protocol_sha256 = frozen_protocol()
    if arm in HIERARCHY_ARMS:
        value = _read_object(manifest_path)
        expected_arm = ARM_KEYS[arm]
        if (
            value.get("artifact_type") != "V6_GC_TWO_LEVEL_TRAINING"
            or value.get("protocol_sha256") != protocol_sha256
            or value.get("arm") != expected_arm
            or value.get("seed") != seed
            or value.get("phase") != "train"
            or value.get("passes") != 3
            or value.get("smoke_only") is not False
            or value.get("holdout_read") is not False
            or value.get("training_dates") != list(dates)
        ):
            raise ValueError(f"invalid V6 hierarchy training artifact for {arm} seed {seed}")
        dataset_proof = value.get("dataset_manifest")
        if dataset_manifest is not None and (
            not isinstance(dataset_proof, dict)
            or Path(str(dataset_proof.get("path", ""))).resolve()
            != Path(dataset_manifest).resolve()
            or dataset_proof.get("sha256") != file_sha256(dataset_manifest)
        ):
            raise ValueError("V6 hierarchy training dataset proof differs")
        products: dict[str, Path] = {}
        for name in ("model_bundle", "regime_trainer", "trader_trainer", "scaler"):
            record = value.get(name)
            if not isinstance(record, dict):
                raise ValueError("V6 hierarchy training manifest is incomplete")
            product = (root / str(record.get("path", ""))).resolve()
            try:
                product.relative_to(root.resolve())
            except ValueError as error:
                raise ValueError("V6 hierarchy product escapes its artifact root") from error
            if not product.is_file() or file_sha256(product) != record.get("sha256"):
                raise ValueError(f"V6 hierarchy product hash differs: {name}")
            products[name] = product
        from .volume_v6_training import load_model_bundle
        _, metadata = load_model_bundle(products["model_bundle"], protocol_id=protocol_sha256)
        warmstart = metadata.get("xau_warmstart")
        if (
            metadata.get("stage") != "gc_two_level_training"
            or metadata.get("seed") != seed
            or metadata.get("arm") != expected_arm
            or not isinstance(warmstart, dict)
        ):
            raise ValueError("V6 hierarchy lacks its registered XAU base-encoder warm-start")
        source = Path(str(warmstart.get("path", "")))
        if not source.is_file() or file_sha256(source) != warmstart.get("sha256"):
            raise ValueError("V6 hierarchy XAU warm-start hash differs")
        if pretraining_manifest is not None:
            parent_manifest, _ = _expected_pretraining_parent(
                pretraining_manifest, kind="v6_price", seed=seed,
            )
            parent = _read_object(parent_manifest)
            parent_model = parent.get("model_bundle")
            if not isinstance(parent_model, dict):
                raise ValueError("V6 price-only parent lacks its model bundle")
            expected_model = (parent_manifest.parent / str(parent_model.get("path", ""))).resolve()
            if (
                source.resolve() != expected_model
                or warmstart.get("sha256") != parent_model.get("sha256")
            ):
                raise ValueError("V6 hierarchy binds a different XAU parent")
        return value

    from .volume_v6_baseline import verify_artifact_manifest
    value = verify_artifact_manifest(
        root, artifact_type="V6_BASELINE_GC_FINETUNE",
        variant=ARM_KEYS[arm], seed=seed,
    )
    if (
        value.get("protocol_sha256") != protocol_sha256
        or value.get("passes") != 3
        or value.get("gc_training_dates") != list(dates)
        or value.get("ensemble_eligible") is not True
        or value.get("smoke_only") is not False
        or value.get("holdout_read") is not False
    ):
        raise ValueError(f"invalid V6 comparator training artifact for {arm} seed {seed}")
    dataset_proof = value.get("training_dataset_proof")
    if dataset_manifest is not None and (
        not isinstance(dataset_proof, dict)
        or Path(str(dataset_proof.get("path", ""))).resolve()
        != Path(dataset_manifest).resolve()
        or dataset_proof.get("sha256") != file_sha256(dataset_manifest)
        or dataset_proof.get("dates") != list(dates)
    ):
        raise ValueError("V6 comparator training dataset proof differs")
    pretrain_path = Path(str(value.get("pretraining_manifest", "")))
    if (
        not pretrain_path.is_file()
        or file_sha256(pretrain_path) != value.get("pretraining_manifest_sha256")
    ):
        raise ValueError("V6 comparator pretraining receipt is absent or changed")
    pretrain = verify_artifact_manifest(
        pretrain_path.parent, artifact_type="V6_BASELINE_XAU_PRETRAIN",
        variant=ARM_KEYS[arm], seed=seed,
    )
    if pretrain.get("smoke_only") is not False:
        raise ValueError("registered V6 comparator used capped XAU pretraining")
    if pretraining_manifest is not None:
        expected_parent, expected_sha = _expected_pretraining_parent(
            pretraining_manifest, kind=ARM_KEYS[arm], seed=seed,
        )
        if pretrain_path.resolve() != expected_parent or file_sha256(pretrain_path) != expected_sha:
            raise ValueError("V6 comparator binds a different XAU parent")
    return value


def create_complete_training_manifest(
    training_root: Path,
    dataset: PhaseDataset,
    pretraining_manifest: Path,
    output: Path,
) -> Path:
    """Hash all 5 x 4 final checkpoints; any missing member is a hard failure."""

    if dataset.phase != "train":
        raise ValueError("complete training receipt requires the registered training dataset")
    verify_pretraining_manifest(pretraining_manifest)
    path = Path(output).resolve()
    if path.is_file():
        verify_complete_training_manifest(path)
        return path
    training_root = Path(training_root).resolve()
    records: list[dict[str, object]] = []
    for arm in REQUIRED_ARMS:
        for seed in SEEDS:
            root = _artifact_root(training_root, arm, seed)
            _verify_training_artifact(
                root, arm=arm, seed=seed, dates=dataset.dates,
                pretraining_manifest=pretraining_manifest,
                dataset_manifest=dataset.manifest_path,
            )
            manifest = root / "manifest.json"
            records.append({
                "arm": arm, "seed": seed,
                "manifest": _relative(path.parent, manifest),
                "manifest_sha256": file_sha256(manifest),
            })
    _, protocol_sha256 = frozen_protocol()
    value = {
        "schema_version": 1,
        "artifact_type": COMPLETE_TRAINING_ARTIFACT,
        "protocol_sha256": protocol_sha256,
        "complete": True,
        "classification": "DEVELOPMENT",
        "training_start": TRAIN_START,
        "training_end": "2026-04-30",
        "training_dates": list(dataset.dates),
        "seeds": list(SEEDS),
        "arms": list(REQUIRED_ARMS),
        "dataset_manifest": {
            "path": _relative(path.parent, dataset.manifest_path),
            "sha256": dataset.manifest_sha256,
        },
        "pretraining_manifest": {
            "path": _relative(path.parent, pretraining_manifest),
            "sha256": file_sha256(pretraining_manifest),
        },
        "checkpoints": records,
        "checkpoint_count": len(records),
        "all_registered_training_complete": True,
        "holdout_read": False,
    }
    atomic_json(path, value)
    verify_complete_training_manifest(path)
    return path


def verify_complete_training_manifest(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    value = _read_object(path)
    _, protocol_sha256 = frozen_protocol()
    if (
        value.get("artifact_type") != COMPLETE_TRAINING_ARTIFACT
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("complete") is not True
        or value.get("classification") != "DEVELOPMENT"
        or value.get("training_start") != TRAIN_START
        or value.get("training_end") != "2026-04-30"
        or value.get("seeds") != list(SEEDS)
        or value.get("arms") != list(REQUIRED_ARMS)
        or value.get("checkpoint_count") != len(SEEDS) * len(REQUIRED_ARMS)
        or value.get("all_registered_training_complete") is not True
        or value.get("holdout_read") is not False
    ):
        raise ValueError("complete V6 training manifest has the wrong identity")
    dates = value.get("training_dates")
    if not isinstance(dates, list) or not dates or dates != sorted(set(map(str, dates))):
        raise ValueError("complete V6 training manifest has invalid dates")
    for name in ("dataset_manifest", "pretraining_manifest"):
        record = value.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"complete V6 training manifest lacks {name}")
        product = _resolve(path.parent, record.get("path"))
        if not product.is_file() or file_sha256(product) != record.get("sha256"):
            raise ValueError(f"complete V6 training input hash differs: {name}")
    dataset_path = _resolve(path.parent, value["dataset_manifest"]["path"])  # type: ignore[index]
    dataset = load_phase_dataset(dataset_path, phase="train")
    if dataset.dates != tuple(str(day) for day in dates):
        raise ValueError("complete V6 training dates differ from the phase-built dataset")
    verify_pretraining_manifest(
        _resolve(path.parent, value["pretraining_manifest"]["path"]),  # type: ignore[index]
    )
    pretraining_path = _resolve(
        path.parent, value["pretraining_manifest"]["path"],  # type: ignore[index]
    )
    records = value.get("checkpoints")
    if not isinstance(records, list) or len(records) != len(SEEDS) * len(REQUIRED_ARMS):
        raise ValueError("complete V6 training manifest does not contain 20 checkpoints")
    identities: set[tuple[str, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("complete V6 training checkpoint record is malformed")
        arm = str(record.get("arm")); seed = int(record.get("seed", -1))
        identities.add((arm, seed))
        manifest = _resolve(path.parent, record.get("manifest"))
        if not manifest.is_file() or file_sha256(manifest) != record.get("manifest_sha256"):
            raise ValueError(f"V6 training checkpoint manifest hash differs for {arm} {seed}")
        _verify_training_artifact(
            manifest.parent, arm=arm, seed=seed, dates=dates,
            pretraining_manifest=pretraining_path,
            dataset_manifest=dataset_path,
        )
    expected = {(arm, seed) for arm in REQUIRED_ARMS for seed in SEEDS}
    if identities != expected:
        raise ValueError("complete V6 training manifest is missing a seed/arm identity")
    return value


def _training_roots(
    complete_manifest_path: Path, complete: Mapping[str, object], arm: str,
) -> tuple[Path, ...]:
    records = complete.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("complete training checkpoint records are absent")
    chosen: list[tuple[int, Path]] = []
    for record in records:
        if isinstance(record, dict) and record.get("arm") == arm:
            manifest = _resolve(Path(complete_manifest_path).parent, record.get("manifest"))
            chosen.append((int(record.get("seed", -1)), manifest.parent))
    chosen.sort()
    if tuple(seed for seed, _ in chosen) != tuple(SEEDS):
        raise ValueError(f"complete training receipt lacks the five {arm} seeds")
    return tuple(root for _, root in chosen)


def _write_or_check(path: Path, value: Mapping[str, object]) -> Path:
    if path.is_file():
        old = _read_object(path)
        if old != dict(value):
            raise FileExistsError(f"refusing to replace differing V6 report: {path}")
        return path
    return atomic_json(path, value)


def _assert_test_chain_bound(
    unlock: Mapping[str, object], complete_training_manifest: Path,
) -> None:
    """Deeply revalidate May and require its exact Jan--Apr model receipt."""

    validation_path = Path(str(unlock.get("validation_evaluation_manifest", "")))
    if (
        not validation_path.is_file()
        or file_sha256(validation_path)
        != unlock.get("validation_evaluation_manifest_sha256")
    ):
        raise PermissionError("test unlock no longer binds a complete May evaluation")
    verified = verify_evaluation_manifest(validation_path)
    expected = unlock.get("complete_training_manifest_sha256")
    training_path = Path(str(unlock.get("complete_training_manifest", "")))
    if (
        verified.get("phase") != "validation"
        or verified.get("verdict") != "PASS"
        or not Path(complete_training_manifest).is_file()
        or expected != file_sha256(complete_training_manifest)
        or training_path.resolve() != Path(complete_training_manifest).resolve()
    ):
        raise PermissionError(
            "test checkpoints differ from the immutable Jan-Apr training receipt used for May"
        )


def _claim_test_output(unlock_receipt: Path, output: Path) -> Path:
    """Bind a one-shot unlock to one output path while allowing safe resume."""

    receipt_path = Path(unlock_receipt).resolve()
    receipt = _read_object(receipt_path)
    validation_path = Path(str(receipt.get("validation_evaluation_manifest", ""))).resolve()
    if not validation_path.is_file():
        raise PermissionError("test unlock lacks its validation evaluation anchor")
    # Anchor consumption beside the one May evaluation, not beside an unlock
    # filename; issuing another unlock cannot create another test output.
    claim_path = validation_path.with_name("test_output_consumption.json")
    _, protocol_sha256 = frozen_protocol()
    expected = {
        "schema_version": 1,
        "artifact_type": "V6_TEST_OUTPUT_CONSUMPTION",
        "protocol_sha256": protocol_sha256,
        "unlock_receipt": str(receipt_path),
        "unlock_receipt_sha256": file_sha256(receipt_path),
        "validation_evaluation_manifest": str(validation_path),
        "validation_evaluation_manifest_sha256": file_sha256(validation_path),
        "test_output": str(Path(output).resolve()),
        "resume_same_output_only": True,
    }
    if claim_path.is_file():
        if _read_object(claim_path) != expected:
            raise PermissionError("V6 test unlock was already consumed by another output")
        return claim_path
    atomic_json(claim_path, expected)
    return claim_path


def _evaluate_one_arm(
    arm: str,
    roots: Sequence[Path],
    training: PhaseDataset,
    evaluation: PhaseDataset,
    *,
    phase: str,
    scenario: str,
    unlock_receipt: Path | None,
    complete_training_manifest: Path,
) -> dict[str, object]:
    if arm in HIERARCHY_ARMS:
        from .volume_v6_pipeline import evaluate_ensemble
        return evaluate_ensemble(
            roots, evaluation.raw_paths, evaluation.sidecar_paths,
            arm=ARM_KEYS[arm], phase=phase, scenario=scenario,
            require_registered_seeds=True, unlock_receipt=unlock_receipt,
            evaluation_dataset_manifest=evaluation.manifest_path,
            complete_training_manifest=complete_training_manifest,
        )
    from .volume_v6_baseline import evaluate_ensemble, SessionPair
    training_pairs = [SessionPair(raw, side) for raw, side in training.pairs]
    evaluation_pairs = [SessionPair(raw, side) for raw, side in evaluation.pairs]
    return evaluate_ensemble(
        roots, training_pairs + evaluation_pairs, evaluation_pairs,
        variant=ARM_KEYS[arm], phase=phase, scenario=scenario,
        unlock_receipt=unlock_receipt, allow_smoke_checkpoints=False,
        training_dataset_manifest=training.manifest_path,
        evaluation_dataset_manifest=evaluation.manifest_path,
        complete_training_manifest=complete_training_manifest,
    )


def evaluate_registered_period(
    complete_training_manifest: Path,
    evaluation_dataset_manifest: Path,
    output: Path,
    *,
    phase: str,
    unlock_receipt: Path | None = None,
) -> Path:
    """Replay all arms under nominal plus all stresses, then apply all gates."""

    if phase not in {"validation", "test"}:
        raise ValueError("registered evaluation phase must be validation or test")
    if phase == "test":
        # This must remain the first filesystem-independent guard.
        _assert_test_date()
        if unlock_receipt is None:
            raise PermissionError("test evaluation requires the immutable unlock receipt")
        unlock = assert_test_unlocked(unlock_receipt)
        _assert_test_chain_bound(unlock, complete_training_manifest)
        consumption_claim = _claim_test_output(unlock_receipt, output)
    else:
        consumption_claim = None
    complete = verify_complete_training_manifest(complete_training_manifest)
    training_record = complete["dataset_manifest"]
    training_manifest_path = _resolve(
        Path(complete_training_manifest).resolve().parent,
        training_record["path"],  # type: ignore[index]
    )
    training = load_phase_dataset(training_manifest_path, phase="train")
    evaluation = load_phase_dataset(
        evaluation_dataset_manifest, phase=phase,
        complete_training_manifest=complete_training_manifest,
        unlock_receipt=unlock_receipt,
    )

    destination = Path(output).resolve()
    final_manifest = destination / "manifest.json"
    if final_manifest.is_file():
        existing = verify_evaluation_manifest(final_manifest)
        existing_training = existing.get("training_manifest")
        existing_evaluation = existing.get("evaluation_dataset_manifest")
        existing_unlock = existing.get("unlock_receipt")
        expected_unlock = (
            None if unlock_receipt is None else {
                "path": str(Path(unlock_receipt).resolve()),
                "sha256": file_sha256(unlock_receipt),
            }
        )
        if (
            existing.get("phase") != phase
            or not isinstance(existing_training, dict)
            or existing_training.get("path")
            != str(Path(complete_training_manifest).resolve())
            or existing_training.get("sha256")
            != file_sha256(complete_training_manifest)
            or not isinstance(existing_evaluation, dict)
            or existing_evaluation.get("path")
            != str(Path(evaluation_dataset_manifest).resolve())
            or existing_evaluation.get("sha256") != evaluation.manifest_sha256
            or existing_unlock != expected_unlock
        ):
            raise ValueError("existing V6 evaluation output belongs to a different request")
        return final_manifest
    if destination.exists():
        raise FileExistsError("registered evaluation output already exists but is incomplete")
    work = destination.with_name(f".{destination.name}.part")
    work.mkdir(parents=True, exist_ok=True)
    nominal: dict[str, dict[str, object]] = {}
    stresses: dict[str, dict[str, dict[str, object]]] = {
        scenario: {} for scenario in STRESS_SCENARIOS
    }
    input_records: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for arm in REQUIRED_ARMS:
            roots = _training_roots(Path(complete_training_manifest), complete, arm)
            report = _evaluate_one_arm(
                arm, roots, training, evaluation, phase=phase,
                scenario=scenario, unlock_receipt=unlock_receipt,
                complete_training_manifest=Path(complete_training_manifest).resolve(),
            )
            report = dict(report)
            report["evaluation_dataset_manifest_path"] = str(evaluation.manifest_path)
            report["evaluation_dataset_manifest_sha256"] = evaluation.manifest_sha256
            report["complete_training_manifest_path"] = str(
                Path(complete_training_manifest).resolve()
            )
            report["complete_training_manifest_sha256"] = file_sha256(
                complete_training_manifest
            )
            directory = work / scenario
            slug = ARM_KEYS[arm]
            report_path = _write_or_check(directory / f"{slug}.json", report)
            input_records.append({
                "scenario": scenario, "arm": arm,
                "path": str(report_path.relative_to(work)),
                "sha256": file_sha256(report_path),
            })
            if scenario == "base":
                nominal[arm] = report
            else:
                stresses[scenario][arm] = report
    gate = apply_registered_gates(nominal, stresses, period=phase)
    # The May gate (and therefore its later unlock receipt) cryptographically
    # binds the exact Jan--April model set.  Test cannot replace, refit, or
    # selectively omit a checkpoint after seeing validation.
    gate["complete_training_manifest"] = str(Path(complete_training_manifest).resolve())
    gate["complete_training_manifest_sha256"] = file_sha256(complete_training_manifest)
    gate_path = _write_or_check(work / "gate.json", gate)
    _, protocol_sha256 = frozen_protocol()
    value = {
        "schema_version": 1,
        "artifact_type": COMPLETE_EVALUATION_ARTIFACT,
        "protocol_sha256": protocol_sha256,
        "complete": True,
        "phase": phase,
        "classification": "DEVELOPMENT" if phase == "validation" else "FRESH_OOS",
        "training_manifest": {
            "path": str(Path(complete_training_manifest).resolve()),
            "sha256": file_sha256(complete_training_manifest),
        },
        "evaluation_dataset_manifest": {
            "path": str(Path(evaluation_dataset_manifest).resolve()),
            "sha256": evaluation.manifest_sha256,
        },
        "unlock_receipt": (
            {
                "path": str(Path(unlock_receipt).resolve()),
                "sha256": file_sha256(unlock_receipt),
            }
            if unlock_receipt is not None else None
        ),
        "test_output_consumption": (
            {
                "path": str(consumption_claim),
                "sha256": file_sha256(consumption_claim),
            }
            if consumption_claim is not None else None
        ),
        "reports": input_records,
        "report_count": len(input_records),
        "scenarios": list(SCENARIOS),
        "arms": list(REQUIRED_ARMS),
        "gate": {"path": gate_path.name, "sha256": file_sha256(gate_path)},
        "verdict": gate["verdict"],
        "failed_gates": gate["failed_gates"],
        "registered_evaluation": True,
        "smoke_only": False,
        "holdout_read": phase == "test",
    }
    atomic_json(work / "manifest.json", value)
    os.replace(work, destination)
    verify_evaluation_manifest(final_manifest)
    return final_manifest


def verify_evaluation_manifest(path: Path) -> dict[str, object]:
    """Deeply reproduce a registered evaluation from its immutable receipts."""

    path = Path(path).resolve()
    value = _read_object(path)
    _, protocol_sha256 = frozen_protocol()
    phase = str(value.get("phase"))
    expected_classification = "DEVELOPMENT" if phase == "validation" else "FRESH_OOS"
    if (
        value.get("artifact_type") != COMPLETE_EVALUATION_ARTIFACT
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("complete") is not True
        or phase not in {"validation", "test"}
        or value.get("classification") != expected_classification
        or value.get("scenarios") != list(SCENARIOS)
        or value.get("arms") != list(REQUIRED_ARMS)
        or value.get("report_count") != len(SCENARIOS) * len(REQUIRED_ARMS)
        or value.get("registered_evaluation") is not True
        or value.get("smoke_only") is not False
        or value.get("holdout_read") is not (phase == "test")
    ):
        raise ValueError("registered V6 evaluation manifest is invalid")

    training_source = value.get("training_manifest")
    evaluation_source = value.get("evaluation_dataset_manifest")
    if not isinstance(training_source, dict) or not isinstance(evaluation_source, dict):
        raise ValueError("registered V6 evaluation lacks its training/data inputs")
    training_path = _resolve(path.parent, training_source.get("path"))
    evaluation_path = _resolve(path.parent, evaluation_source.get("path"))
    if (
        not training_path.is_file()
        or file_sha256(training_path) != training_source.get("sha256")
        or not evaluation_path.is_file()
        or file_sha256(evaluation_path) != evaluation_source.get("sha256")
    ):
        raise ValueError("registered V6 evaluation input hash differs")

    unlock_path: Path | None = None
    if phase == "test":
        unlock_record = value.get("unlock_receipt")
        claim_record = value.get("test_output_consumption")
        if not isinstance(unlock_record, dict) or not isinstance(claim_record, dict):
            raise PermissionError("test evaluation lacks unlock/one-output receipts")
        unlock_path = _resolve(path.parent, unlock_record.get("path"))
        claim_path = _resolve(path.parent, claim_record.get("path"))
        if (
            not unlock_path.is_file()
            or file_sha256(unlock_path) != unlock_record.get("sha256")
            or not claim_path.is_file()
            or file_sha256(claim_path) != claim_record.get("sha256")
        ):
            raise PermissionError("test unlock/one-output receipt changed")
        unlock = assert_test_unlocked(unlock_path)
        _assert_test_chain_bound(unlock, training_path)
        claim = _read_object(claim_path)
        if (
            claim.get("artifact_type") != "V6_TEST_OUTPUT_CONSUMPTION"
            or claim.get("protocol_sha256") != protocol_sha256
            or claim.get("unlock_receipt_sha256") != file_sha256(unlock_path)
            or claim.get("validation_evaluation_manifest_sha256")
            != unlock.get("validation_evaluation_manifest_sha256")
            or Path(str(claim.get("validation_evaluation_manifest", ""))).resolve()
            != Path(str(unlock.get("validation_evaluation_manifest", ""))).resolve()
            or Path(str(claim.get("test_output", ""))).resolve() != path.parent
            or claim.get("resume_same_output_only") is not True
        ):
            raise PermissionError("test output consumption receipt is invalid")
    elif value.get("unlock_receipt") is not None or value.get("test_output_consumption") is not None:
        raise ValueError("validation evaluation must not contain test receipts")

    complete = verify_complete_training_manifest(training_path)
    evaluation = load_phase_dataset(
        evaluation_path, phase=phase,
        complete_training_manifest=training_path,
        unlock_receipt=unlock_path,
    )
    expected_checkpoints: dict[str, list[dict[str, object]]] = {}
    complete_records = complete.get("checkpoints")
    if not isinstance(complete_records, list):
        raise ValueError("training receipt checkpoint records are absent")
    for arm in REQUIRED_ARMS:
        selected = sorted(
            (
                {
                    "seed": int(record["seed"]),
                    "manifest_sha256": str(record["manifest_sha256"]),
                }
                for record in complete_records
                if isinstance(record, dict) and record.get("arm") == arm
            ),
            key=lambda record: int(record["seed"]),
        )
        if [record["seed"] for record in selected] != list(SEEDS):
            raise ValueError(f"training receipt lacks checkpoint identities for {arm}")
        expected_checkpoints[arm] = selected

    records = value.get("reports")
    if not isinstance(records, list) or len(records) != len(SCENARIOS) * len(REQUIRED_ARMS):
        raise ValueError("registered V6 evaluation reports are incomplete")
    identities: set[tuple[str, str]] = set()
    nominal: dict[str, dict[str, object]] = {}
    stresses: dict[str, dict[str, dict[str, object]]] = {
        scenario: {} for scenario in STRESS_SCENARIOS
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("registered V6 report record is malformed")
        scenario = str(record.get("scenario")); arm = str(record.get("arm"))
        identities.add((scenario, arm))
        report_path = _resolve(path.parent, record.get("path"))
        try:
            report_path.relative_to(path.parent)
        except ValueError as error:
            raise ValueError("registered V6 report path escapes its result root") from error
        if not report_path.is_file() or file_sha256(report_path) != record.get("sha256"):
            raise ValueError("registered V6 evaluation report hash differs")
        report = _read_object(report_path)
        expected_type = (
            "V6_COMMON_LEDGER_REPLAY" if arm in HIERARCHY_ARMS
            else "V6_BASELINE_COMMON_LEDGER_REPLAY"
        )
        if (
            report.get("artifact_type") != expected_type
            or report.get("protocol_sha256") != protocol_sha256
            or report.get("arm") != arm
            or report.get("scenario") != scenario
            or report.get("phase") != phase
            or report.get("classification") != expected_classification
            or report.get("evaluation_dates") != list(evaluation.dates)
            or report.get("ensemble_seeds") != list(SEEDS)
            or report.get("checkpoint_set") != expected_checkpoints.get(arm)
            or report.get("registered_evaluation") is not True
            or report.get("smoke_only") is not False
            or report.get("holdout_read") is not (phase == "test")
            or report.get("evaluation_dataset_manifest_sha256")
            != evaluation.manifest_sha256
            or report.get("complete_training_manifest_sha256")
            != training_source.get("sha256")
            or Path(str(report.get("evaluation_dataset_manifest_path", ""))).resolve()
            != evaluation_path
            or Path(str(report.get("complete_training_manifest_path", ""))).resolve()
            != training_path
        ):
            raise ValueError(f"registered V6 report identity differs for {scenario}/{arm}")
        daily = report.get("daily_pnl_cents")
        metrics = report.get("metrics")
        if not isinstance(daily, dict) or not isinstance(metrics, dict):
            raise ValueError("registered V6 report lacks daily P&L/metrics")
        try:
            normalized_daily = {str(day): int(amount) for day, amount in daily.items()}
            trips = int(metrics["completed_round_trips"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("registered V6 report metrics are malformed") from error
        if normalized_daily != daily or sorted(normalized_daily) != list(evaluation.dates):
            raise ValueError("registered V6 daily P&L is not paired to the phase dataset")
        if metrics != daily_metrics(normalized_daily, trips):
            raise ValueError("registered V6 metrics do not recompute from daily P&L")
        if scenario == "base":
            nominal[arm] = report
        elif scenario in stresses:
            stresses[scenario][arm] = report
    expected = {(scenario, arm) for scenario in SCENARIOS for arm in REQUIRED_ARMS}
    if identities != expected:
        raise ValueError("registered V6 evaluation is missing a scenario/arm report")

    gate_record = value.get("gate")
    if not isinstance(gate_record, dict):
        raise ValueError("registered V6 evaluation gate is absent")
    gate_path = _resolve(path.parent, gate_record.get("path"))
    if not gate_path.is_file() or file_sha256(gate_path) != gate_record.get("sha256"):
        raise ValueError("registered V6 gate hash differs")
    recomputed_gate = apply_registered_gates(nominal, stresses, period=phase)
    recomputed_gate["complete_training_manifest"] = str(training_path)
    recomputed_gate["complete_training_manifest_sha256"] = training_source["sha256"]
    gate_value = _read_object(gate_path)
    if gate_value != recomputed_gate:
        raise ValueError("stored V6 gate differs from a full recomputation")
    if (
        value.get("verdict") != gate_value.get("verdict")
        or value.get("failed_gates") != gate_value.get("failed_gates")
    ):
        raise ValueError("evaluation summary differs from its recomputed gate")
    return value


def unlock_test(
    validation_evaluation_manifest: Path, output: Path,
) -> Path:
    _assert_test_date()
    validation_path = Path(validation_evaluation_manifest).resolve()
    verified = verify_evaluation_manifest(validation_path)
    if verified.get("phase") != "validation" or verified.get("verdict") != "PASS":
        raise PermissionError("only the deeply verified complete May PASS can unlock test")
    return authorize_test(validation_path, output)


def status(
    *,
    complete_training_manifest: Path | None = None,
    validation_evaluation_manifest: Path | None = None,
    unlock_receipt: Path | None = None,
) -> dict[str, object]:
    """Return control-plane status without enumerating a market-data root."""

    protocol, protocol_sha256 = frozen_protocol()
    current = _current_date()
    return {
        "strategy_version": protocol["strategy_version"],
        "protocol_sha256": protocol_sha256,
        "development_verdict": protocol["development_result"]["verdict"],  # type: ignore[index]
        "training_manifest_exists": bool(
            complete_training_manifest is not None and Path(complete_training_manifest).is_file()
        ),
        "validation_evaluation_manifest_exists": bool(
            validation_evaluation_manifest is not None
            and Path(validation_evaluation_manifest).is_file()
        ),
        "unlock_receipt_exists": bool(unlock_receipt is not None and Path(unlock_receipt).is_file()),
        "test_calendar_complete": current >= TEST_COMPLETE_ON,
        "test_earliest_open_date": str(TEST_COMPLETE_ON),
        "test_checkpoint_policy": "reuse_exact_hash_bound_Jan-Apr_selection_checkpoints",
        "holdout_access_authorized_in_frozen_protocol": protocol["holdout_access_authorized"],
        "market_data_roots_enumerated": False,
    }


__all__ = [
    "ARM_KEYS", "COMPLETE_EVALUATION_ARTIFACT", "COMPLETE_TRAINING_ARTIFACT",
    "PhaseDataset", "SCENARIOS", "build_phase_dataset",
    "create_active_map_universe", "create_complete_training_manifest", "evaluate_registered_period",
    "load_phase_dataset", "pretrain_all_xau", "register_active_map", "status", "train_all",
    "unlock_test", "verify_complete_training_manifest",
    "verify_evaluation_manifest", "verify_pretraining_manifest",
]
