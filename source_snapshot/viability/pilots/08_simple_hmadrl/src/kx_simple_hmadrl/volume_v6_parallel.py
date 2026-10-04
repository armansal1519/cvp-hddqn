"""Eight-process, hash-resumable launcher for V6 laptop training."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import os
from pathlib import Path
import time
from typing import Sequence

from .volume_v6_contract import SEEDS
from .volume_v6_eval import file_sha256, frozen_protocol


def configure_worker(threads: int = 1) -> None:
    if threads < 1:
        raise ValueError("worker thread count must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Inter-op threads can be configured only once in an interpreter.
        pass


def _input_digest(raw_paths: Sequence[Path], side_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for raw, side in zip(raw_paths, side_paths, strict=True):
        digest.update(Path(raw).stem.encode())
        digest.update(file_sha256(Path(raw)).encode())
        digest.update(file_sha256(Path(side)).encode())
    return digest.hexdigest()


def _valid_training_output(
    path: Path, *, seed: int, arm: str, smoke: bool, phase: str,
    passes: int, input_digest: str, pretrained_model: Path | None,
    dataset_manifest: Path | None,
) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        _, digest = frozen_protocol()
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        value.get("artifact_type") != "V6_GC_TWO_LEVEL_TRAINING"
        or value.get("protocol_sha256") != digest
        or value.get("seed") != seed
        or value.get("arm") != arm
        or value.get("smoke_only") is not smoke
        or value.get("phase") != phase
        or value.get("passes") != passes
        or value.get("input_digest") != input_digest
    ):
        return False
    expected_dataset = None if dataset_manifest is None else {
        "path": str(Path(dataset_manifest).resolve()),
        "sha256": file_sha256(Path(dataset_manifest)),
    }
    if value.get("dataset_manifest") != expected_dataset:
        return False
    warmstart = value.get("model_bundle")
    if not isinstance(warmstart, dict):
        return False
    # The immutable model metadata is checked by the loader.  At the launcher
    # boundary, also bind reuse to the exact requested XAU file (or its absence).
    expected_xau = None if pretrained_model is None else {
        "path": str(Path(pretrained_model).resolve()),
        "sha256": file_sha256(Path(pretrained_model)),
    }
    recorded_xau = value.get("xau_warmstart")
    if recorded_xau != expected_xau:
        return False
    for name in ("model_bundle", "regime_trainer", "trader_trainer", "scaler"):
        record = value.get(name)
        if not isinstance(record, dict):
            return False
        artifact = path / str(record.get("path", ""))
        if not artifact.is_file() or file_sha256(artifact) != record.get("sha256"):
            return False
    return True


def _job(
    raw_paths: tuple[str, ...], side_paths: tuple[str, ...], output: str,
    seed: int, arm: str, pretrain: str | None, phase: str,
    maximum_sessions: int | None, passes: int, dataset_manifest: str | None,
) -> str:
    configure_worker(1)
    destination = Path(output) / arm / f"seed_{seed}"
    raw = tuple(Path(value) for value in raw_paths)
    side = tuple(Path(value) for value in side_paths)
    pretrained_model = None if pretrain is None else Path(pretrain)
    dataset_path = None if dataset_manifest is None else Path(dataset_manifest)
    if _valid_training_output(
        destination, seed=seed, arm=arm, smoke=phase == "smoke", phase=phase,
        passes=passes, input_digest=_input_digest(raw, side),
        pretrained_model=pretrained_model, dataset_manifest=dataset_path,
    ):
        return f"SKIP:{arm}:seed_{seed}:{destination}"
    if destination.exists():
        os.replace(
            destination,
            destination.with_name(f".{destination.name}.invalid.{time.time_ns()}"),
        )
    from .volume_v6_pipeline import train_seed
    manifest = train_seed(
        raw, side,
        destination, seed=seed, arm=arm,
        pretrained_model=pretrained_model,
        phase=phase, maximum_sessions=maximum_sessions, passes=passes,
        dataset_manifest=dataset_path,
    )
    return f"DONE:{arm}:seed_{seed}:{manifest}"


def train_parallel(
    raw_paths: Sequence[Path], side_paths: Sequence[Path], output: Path, *,
    pretrained_root: Path | None = None,
    workers: int = 8,
    phase: str = "train",
    maximum_sessions: int | None = None,
    passes: int = 3,
    seeds: Sequence[int] = SEEDS,
    arms: Sequence[str] = ("rich_volume", "base_ohlcv"),
    dataset_manifest: Path | None = None,
) -> list[str]:
    """Run independent arm/seed jobs; each hierarchy stays same-seed internally."""

    if not 1 <= workers <= 12:
        raise ValueError("V6 laptop worker count must be between 1 and 12")
    selected_seeds = tuple(int(value) for value in seeds)
    if phase != "smoke" and selected_seeds != SEEDS:
        raise ValueError("registered parallel training requires all five seeds")
    if phase != "smoke" and pretrained_root is None:
        raise ValueError("registered parallel training requires V6 XAU pretraining")
    if phase != "smoke" and dataset_manifest is None:
        raise ValueError("registered parallel training requires the train dataset manifest")
    jobs = []
    raw = tuple(str(Path(value)) for value in raw_paths)
    side = tuple(str(Path(value)) for value in side_paths)
    for arm in arms:
        if arm not in {"rich_volume", "base_ohlcv"}:
            raise ValueError(f"unknown V6 arm {arm!r}")
        for seed in selected_seeds:
            pretrain = (
                None if pretrained_root is None
                else str(pretrained_root / f"seed_{seed}" / "model_bundle.pt")
            )
            jobs.append((raw, side, str(output), seed, arm, pretrain, phase,
                         maximum_sessions, passes,
                         None if dataset_manifest is None else str(Path(dataset_manifest).resolve())))
    results: list[str] = []
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        initializer=configure_worker,
        initargs=(1,),
    ) as executor:
        futures = {executor.submit(_job, *job): (job[4], job[3]) for job in jobs}
        for future in as_completed(futures):
            arm, seed = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"V6 parallel job failed for {arm} seed {seed}") from error
    return sorted(results)


__all__ = ["configure_worker", "train_parallel"]
