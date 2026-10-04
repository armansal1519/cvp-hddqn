"""CLI for the frozen two-level V6 GC volume experiment.

Typical registered order::

    python -m kx_simple_hmadrl.volume_v6_cli create-active-map-universe ...
    python -m kx_simple_hmadrl.volume_v6_cli register-active-map ...
    python -m kx_simple_hmadrl.volume_v6_cli build --phase train ...
    python -m kx_simple_hmadrl.volume_v6_cli pretrain-xau ...
    python -m kx_simple_hmadrl.volume_v6_cli train ...
    python -m kx_simple_hmadrl.volume_v6_cli build --phase validation ...
    python -m kx_simple_hmadrl.volume_v6_cli validate ...

``unlock-test`` and ``test`` remain calendar-locked until 2026-09-01.  The May
gate binds the exact Jan--April checkpoint receipt, so test cannot substitute,
refit, or select a different model after validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .volume_v6_orchestration import (
    build_phase_dataset,
    create_active_map_universe,
    evaluate_registered_period,
    pretrain_all_xau,
    register_active_map,
    status as experiment_status,
    train_all,
    unlock_test,
)


def _workers(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 12:
        raise argparse.ArgumentTypeError("workers must be between 1 and 12")
    return parsed


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-manifest", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="show split/control-plane status only")
    status.add_argument("--training-manifest", type=Path)
    status.add_argument("--validation-evaluation-manifest", type=Path)
    status.add_argument("--unlock-receipt", type=Path)

    build = commands.add_parser("build", help="build one exact phase-bound GC dataset")
    build.add_argument("--phase", required=True, choices=("train", "validation", "test"))
    build.add_argument("--archive-root", required=True, type=Path)
    build.add_argument("--active-map", required=True, type=Path)
    build.add_argument("--active-map-receipt", type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--training-manifest", type=Path)
    build.add_argument("--unlock-receipt", type=Path)

    register = commands.add_parser(
        "register-active-map", help="validate and hash-bind the strict past-only roll map",
    )
    register.add_argument("--active-map", required=True, type=Path)
    register.add_argument("--source-manifest", required=True, type=Path)
    register.add_argument("--output", type=Path)
    register.add_argument("--training-manifest", type=Path)
    register.add_argument("--unlock-receipt", type=Path)

    universe = commands.add_parser(
        "create-active-map-universe",
        help="authorize and hash every archive child behind the exact phase roll map",
    )
    universe.add_argument(
        "--phase", required=True, choices=("smoke", "train", "validation", "test"),
    )
    universe.add_argument("--archive-root", required=True, type=Path)
    universe.add_argument("--active-map", required=True, type=Path)
    universe.add_argument("--start", required=True)
    universe.add_argument("--end-exclusive", required=True)
    universe.add_argument("--output", required=True, type=Path)
    universe.add_argument("--training-manifest", type=Path)
    universe.add_argument("--unlock-receipt", type=Path)

    pretrain = commands.add_parser(
        "pretrain-xau", help="pretrain all five seeds for V6 and both paper comparators",
    )
    pretrain.add_argument("--direct-xau", required=True, type=Path)
    pretrain.add_argument("--output", required=True, type=Path)
    pretrain.add_argument("--workers", type=_workers, default=8)
    pretrain.add_argument(
        "--maximum-sessions", type=int,
        help="contaminated implementation smoke only; requires --maximum-decisions",
    )
    pretrain.add_argument(
        "--maximum-decisions", type=int,
        help="contaminated implementation smoke only; requires --maximum-sessions",
    )

    train = commands.add_parser(
        "train", help="train five seeds of both V6 arms and both baselines",
    )
    train.add_argument("--dataset-manifest", required=True, type=Path)
    train.add_argument("--pretraining-manifest", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--workers", type=_workers, default=8)

    for name in ("evaluate", "replay"):
        evaluate = commands.add_parser(
            name, help="registered all-arm nominal-plus-stress replay and gate",
        )
        evaluate.add_argument("--phase", required=True, choices=("validation", "test"))
        _add_evaluation_arguments(evaluate)
        evaluate.add_argument("--unlock-receipt", type=Path)

    validate = commands.add_parser(
        "validate", help="open May once and run the complete registered development gate",
    )
    _add_evaluation_arguments(validate)

    unlock = commands.add_parser(
        "unlock-test", help="authorize the untouched test after a complete May PASS",
    )
    unlock.add_argument("--validation-evaluation-manifest", required=True, type=Path)
    unlock.add_argument("--output", required=True, type=Path)

    test = commands.add_parser(
        "test", help="one-shot test using the exact May-bound Jan-Apr checkpoints",
    )
    _add_evaluation_arguments(test)
    test.add_argument("--unlock-receipt", required=True, type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "status":
        value = experiment_status(
            complete_training_manifest=args.training_manifest,
            validation_evaluation_manifest=args.validation_evaluation_manifest,
            unlock_receipt=args.unlock_receipt,
        )
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    if args.command == "build":
        result = build_phase_dataset(
            args.archive_root, args.active_map, args.output, phase=args.phase,
            active_map_receipt=args.active_map_receipt,
            complete_training_manifest=args.training_manifest,
            unlock_receipt=args.unlock_receipt,
        )
        print(result)
        return 0
    if args.command == "register-active-map":
        print(register_active_map(
            args.active_map, args.source_manifest, args.output,
            complete_training_manifest=args.training_manifest,
            unlock_receipt=args.unlock_receipt,
        ))
        return 0
    if args.command == "create-active-map-universe":
        print(create_active_map_universe(
            args.active_map, args.archive_root, args.output, phase=args.phase,
            start=args.start, end_exclusive=args.end_exclusive,
            complete_training_manifest=args.training_manifest,
            unlock_receipt=args.unlock_receipt,
        ))
        return 0
    if args.command == "pretrain-xau":
        if args.maximum_sessions is not None and args.maximum_sessions < 1:
            raise ValueError("maximum-sessions must be positive")
        if args.maximum_decisions is not None and args.maximum_decisions < 1:
            raise ValueError("maximum-decisions must be positive")
        result = pretrain_all_xau(
            args.direct_xau, args.output, workers=args.workers,
            maximum_sessions=args.maximum_sessions,
            maximum_decisions=args.maximum_decisions,
        )
        print(result)
        return 0
    if args.command == "train":
        result = train_all(
            args.dataset_manifest, args.pretraining_manifest, args.output,
            workers=args.workers,
        )
        print(result)
        return 0
    if args.command in {"evaluate", "replay"}:
        if args.phase == "test" and args.unlock_receipt is None:
            raise PermissionError("test replay requires --unlock-receipt")
        if args.phase == "validation" and args.unlock_receipt is not None:
            raise ValueError("validation replay must not use a test unlock receipt")
        result = evaluate_registered_period(
            args.training_manifest, args.dataset_manifest, args.output,
            phase=args.phase, unlock_receipt=args.unlock_receipt,
        )
        print(result)
        return 0
    if args.command == "validate":
        result = evaluate_registered_period(
            args.training_manifest, args.dataset_manifest, args.output,
            phase="validation",
        )
        print(result)
        return 0
    if args.command == "unlock-test":
        print(unlock_test(args.validation_evaluation_manifest, args.output))
        return 0
    if args.command == "test":
        result = evaluate_registered_period(
            args.training_manifest, args.dataset_manifest, args.output,
            phase="test", unlock_receipt=args.unlock_receipt,
        )
        print(result)
        return 0
    raise AssertionError("argparse accepted an unknown V6 command")


if __name__ == "__main__":
    raise SystemExit(main())
