"""Job-oriented command line interface for the frozen ablation matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Sequence

from .aggregation import (
    build_complete_evaluation_manifest,
    build_complete_training_manifest,
    verify_complete_evaluation_manifest,
    verify_complete_training_manifest,
)
from .artifacts import workspace_root
from .contract import (
    ABLATION_ARMS,
    CANONICAL_OUTPUT_ROOT,
    PERIODS,
    PUBLISHER_PARENT_ARM,
    SCENARIOS,
    SEEDS,
)
from .initialization import create_trader_initialization, verify_initialization
from .protocol import assert_frozen_contracts, protocol_binding
from .replay import replay_ensemble, write_technical_failure
from .training import train_trader, verify_training_member


def canonical_output_root() -> Path:
    return workspace_root() / CANONICAL_OUTPUT_ROOT


def _output_root(value: Path | None, *, smoke_only: bool) -> Path:
    root = canonical_output_root() if value is None else Path(value).resolve()
    if not smoke_only and root != canonical_output_root().resolve():
        raise PermissionError("production ablations require the canonical output root")
    return root


def _xau_root(seed: int) -> Path:
    return (
        workspace_root() / "viability/artifacts/development/"
        "v6_registered_xau_pretraining/v6_price" / f"seed_{seed}"
    )


def _reference_root(arm: str, seed: int) -> Path:
    parent = PUBLISHER_PARENT_ARM[arm]
    parent = "rich_volume" if parent is None else parent
    return (
        workspace_root() / "viability/artifacts/development/"
        "v6_registered_training_terminal_eligible" / str(parent) / f"seed_{seed}"
    )


def _initialize_one(seed: int, output_root: str, smoke_only: bool) -> str:
    from kx_simple_hmadrl.volume_v6_parallel import configure_worker

    configure_worker(1)
    root = Path(output_root)
    result = create_trader_initialization(
        seed=seed, xau_pretraining_root=_xau_root(seed),
        output=root / "initializations" / f"seed_{seed}", smoke_only=smoke_only,
    )
    return str(result)


def _train_one(
    arm: str, seed: int, output_root: str, smoke_only: bool,
    maximum_sessions: int | None,
) -> str:
    from kx_simple_hmadrl.volume_v6_parallel import configure_worker

    configure_worker(1)
    root = Path(output_root)
    result = train_trader(
        arm=arm,
        seed=seed,
        initialization_manifest=root / "initializations" / f"seed_{seed}" / "manifest.json",
        reference_member_root=_reference_root(arm, seed),
        output=root / "training" / arm / f"seed_{seed}",
        smoke_only=smoke_only,
        maximum_sessions=maximum_sessions,
    )
    return str(result)


def _replay_one(
    arm: str, period: str, scenario: str, output_root: str,
) -> str:
    from kx_simple_hmadrl.volume_v6_parallel import configure_worker

    configure_worker(1)
    root = Path(output_root)
    return str(replay_ensemble(
        arm=arm, period=period, scenario=scenario,
        training_manifest=root / "training" / "manifest.json",
        output_root=root,
    ))


def _parallel(jobs: list[tuple[object, ...]], worker: object, workers: int) -> list[str]:
    if not 1 <= workers <= 12:
        raise ValueError("worker count must be between 1 and 12")
    results: list[str] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = {executor.submit(worker, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                results.append(str(future.result()))
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
    return sorted(results)


def initialize_all(root: Path, *, workers: int, smoke_only: bool) -> list[str]:
    assert_frozen_contracts()
    jobs = [(seed, str(root), smoke_only) for seed in SEEDS]
    return _parallel(jobs, _initialize_one, workers)


def train_all(
    root: Path, *, workers: int, smoke_only: bool,
    maximum_sessions: int | None,
) -> list[str]:
    assert_frozen_contracts()
    for seed in SEEDS:
        verify_initialization(
            root / "initializations" / f"seed_{seed}" / "manifest.json",
            expected_seed=seed, allow_smoke=smoke_only,
        )
    jobs = [
        (arm, seed, str(root), smoke_only, maximum_sessions)
        for arm in ABLATION_ARMS for seed in SEEDS
    ]
    return _parallel(jobs, _train_one, workers)


def replay_all(root: Path, *, workers: int) -> list[str]:
    assert_frozen_contracts()
    verify_complete_training_manifest(root / "training" / "manifest.json")
    jobs = [
        (arm, period, scenario, str(root))
        for arm in ABLATION_ARMS for period in PERIODS for scenario in SCENARIOS
    ]
    return _parallel(jobs, _replay_one, workers)


def status(root: Path) -> dict[str, object]:
    initializations = sum(
        (root / "initializations" / f"seed_{seed}" / "manifest.json").is_file()
        for seed in SEEDS
    )
    training = sum(
        (root / "training" / arm / f"seed_{seed}" / "manifest.json").is_file()
        for arm in ABLATION_ARMS for seed in SEEDS
    )
    evaluations = sum(
        (root / "evaluation" / period / scenario / arm / "report.json").is_file()
        for arm in ABLATION_ARMS for period in PERIODS for scenario in SCENARIOS
    )
    return {
        "experiment_id": "CVP_ABLATION_EXTENSION_V1",
        "output_root": str(root),
        "initializations": {"complete": initializations, "required": len(SEEDS)},
        "training_jobs": {"complete": training, "required": 15},
        "ensemble_evaluations": {"complete": evaluations, "required": 24},
        "complete_training_manifest": (root / "training" / "manifest.json").is_file(),
        "complete_evaluation_manifest": (root / "evaluation" / "manifest.json").is_file(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-contracts")
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--seed", type=int, required=True, choices=SEEDS)
    initialize_all_parser = subparsers.add_parser("initialize-all")
    initialize_all_parser.add_argument("--workers", type=int, default=5)
    train = subparsers.add_parser("train")
    train.add_argument("--arm", choices=ABLATION_ARMS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--maximum-sessions", type=int)
    train_all_parser = subparsers.add_parser("train-all")
    train_all_parser.add_argument("--workers", type=int, default=6)
    train_all_parser.add_argument("--maximum-sessions", type=int)
    full_training = subparsers.add_parser("all-training")
    full_training.add_argument("--workers", type=int, default=6)
    full_training.add_argument("--maximum-sessions", type=int)
    subparsers.add_parser("aggregate-training")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--arm", choices=ABLATION_ARMS, required=True)
    replay.add_argument("--period", choices=PERIODS, required=True)
    replay.add_argument("--scenario", choices=SCENARIOS, required=True)
    replay_all_parser = subparsers.add_parser("replay-all")
    replay_all_parser.add_argument("--workers", type=int, default=4)
    subparsers.add_parser("aggregate-evaluation")
    failure = subparsers.add_parser("record-technical-failure")
    failure.add_argument("--arm", choices=ABLATION_ARMS, required=True)
    failure.add_argument("--period", choices=PERIODS, required=True)
    failure.add_argument("--scenario", choices=SCENARIOS, required=True)
    failure.add_argument("--reason", required=True)
    failure.add_argument("--diagnostic", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--require-evaluation", action="store_true")
    subparsers.add_parser("status")
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--training-workers", type=int, default=6)
    all_parser.add_argument("--replay-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _output_root(args.output_root, smoke_only=args.smoke_only)
    command = args.command
    if command == "verify-contracts":
        result: object = protocol_binding()
    elif command == "initialize":
        result = _initialize_one(args.seed, str(root), args.smoke_only)
    elif command == "initialize-all":
        result = initialize_all(root, workers=args.workers, smoke_only=args.smoke_only)
    elif command == "train":
        result = _train_one(
            args.arm, args.seed, str(root), args.smoke_only, args.maximum_sessions,
        )
    elif command == "train-all":
        result = train_all(
            root, workers=args.workers, smoke_only=args.smoke_only,
            maximum_sessions=args.maximum_sessions,
        )
    elif command == "all-training":
        initialize_all(root, workers=min(args.workers, 5), smoke_only=args.smoke_only)
        train_all(
            root, workers=args.workers, smoke_only=args.smoke_only,
            maximum_sessions=args.maximum_sessions,
        )
        if args.smoke_only:
            result = status(root)
        else:
            result = str(build_complete_training_manifest(root))
    elif command == "aggregate-training":
        if args.smoke_only:
            raise PermissionError("SMOKE_ONLY runs cannot create a complete training receipt")
        result = str(build_complete_training_manifest(root))
    elif command == "replay":
        if args.smoke_only:
            raise PermissionError("registered historical replay refuses smoke mode")
        result = _replay_one(args.arm, args.period, args.scenario, str(root))
    elif command == "replay-all":
        if args.smoke_only:
            raise PermissionError("registered historical replay refuses smoke mode")
        result = replay_all(root, workers=args.workers)
    elif command == "aggregate-evaluation":
        if args.smoke_only:
            raise PermissionError("SMOKE_ONLY runs cannot create an evaluation receipt")
        result = str(build_complete_evaluation_manifest(root))
    elif command == "record-technical-failure":
        result = str(write_technical_failure(
            arm=args.arm, period=args.period, scenario=args.scenario,
            reason=args.reason, diagnostic=args.diagnostic, output_root=root,
        ))
    elif command == "verify":
        assert_frozen_contracts()
        result = {"training": str(verify_complete_training_manifest(
            root / "training" / "manifest.json"
        )["artifact_type"])}
        if args.require_evaluation:
            result["evaluation"] = str(verify_complete_evaluation_manifest(
                root / "evaluation" / "manifest.json"
            )["artifact_type"])
    elif command == "status":
        result = status(root)
    elif command == "all":
        if args.smoke_only:
            raise PermissionError("full registered workflow refuses smoke mode")
        initialize_all(root, workers=min(args.training_workers, 5), smoke_only=False)
        train_all(
            root, workers=args.training_workers, smoke_only=False,
            maximum_sessions=None,
        )
        build_complete_training_manifest(root)
        replay_all(root, workers=args.replay_workers)
        result = str(build_complete_evaluation_manifest(root))
    else:  # pragma: no cover - argparse guarantees a known command
        raise AssertionError(command)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
