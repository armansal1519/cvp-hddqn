"""Command-line orchestration for the frozen Pilot 10 baseline suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import file_sha256, workspace_root
from .contract import (
    BASELINE_ARMS,
    CANONICAL_OUTPUT_ROOT,
    DETERMINISTIC_ARMS,
    LEARNER_ARMS,
    PERIOD_CLASSIFICATION,
    PERIODS,
    PROTOCOL_ID,
    SCENARIOS,
    SEEDS,
)
from .protocol import assert_frozen_contracts, validate_workspace_output


def _output_root(value: str | None) -> Path:
    return validate_workspace_output(
        CANONICAL_OUTPUT_ROOT if value is None else value,
    )


def _report_root(value: str | None) -> Path:
    from .aggregation import DEFAULT_REPORT_ROOT

    return validate_workspace_output(
        DEFAULT_REPORT_ROOT if value is None else value,
    )


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        help="Project-root-relative or absolute artifact root; defaults to the frozen canonical root.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kx-cvp-baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-contracts")
    sub.add_parser("describe")

    status = sub.add_parser("status")
    _add_output(status)
    status.add_argument("--verbose", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("arm", choices=LEARNER_ARMS)
    train.add_argument("seed", type=int, choices=SEEDS)
    _add_output(train)
    train.add_argument("--no-resume", action="store_true")

    train_all = sub.add_parser("train-all")
    _add_output(train_all)
    train_all.add_argument("--workers", type=int, default=1)

    verify_training = sub.add_parser("verify-training")
    _add_output(verify_training)

    replay = sub.add_parser("replay")
    replay.add_argument("arm", choices=BASELINE_ARMS)
    replay.add_argument("period", choices=PERIODS)
    replay.add_argument("scenario", choices=SCENARIOS)
    _add_output(replay)

    replay_all = sub.add_parser("replay-all")
    _add_output(replay_all)
    replay_all.add_argument("--workers", type=int, default=1)

    aggregate = sub.add_parser("aggregate-evaluation")
    _add_output(aggregate)

    audit = sub.add_parser("audit-source")
    _add_output(audit)

    evidence = sub.add_parser("build-evidence")
    _add_output(evidence)
    evidence.add_argument("--report-root")

    report = sub.add_parser("build-report")
    report.add_argument("--report-root")

    verify = sub.add_parser("verify")
    _add_output(verify)
    verify.add_argument("--report-root")
    verify.add_argument("--require-complete", action="store_true")
    verify.add_argument("--audit-source", action="store_true")

    portable = sub.add_parser(
        "verify-portable",
        help=(
            "Verify committed hashes and artifact schemas without checkpoints, "
            "raw sessions, replay, or economic recomputation."
        ),
    )
    _add_output(portable)
    portable.add_argument("--report-root")
    portable.add_argument("--require-complete", action="store_true")
    return parser


def _describe() -> dict[str, object]:
    contracts = assert_frozen_contracts()
    return {
        "experiment_id": PROTOCOL_ID,
        "protocol_canonical_sha256": contracts.protocol_canonical_sha256,
        "learner_arms": list(LEARNER_ARMS),
        "deterministic_arms": list(DETERMINISTIC_ARMS),
        "all_new_arms": list(BASELINE_ARMS),
        "registered_primary_comparator": "flat_multiclock_ddqn",
        "seeds": list(SEEDS),
        "periods": list(PERIODS),
        "scenarios": list(SCENARIOS),
        "period_classification": PERIOD_CLASSIFICATION,
        "required_training_jobs": 15,
        "required_evaluation_cells": 64,
        "august_access_authorized": False,
        "paper_integration": False,
    }


def _status(root: Path, *, verbose: bool) -> dict[str, object]:
    from .replay import evaluation_status
    from .training import training_status

    training = list(training_status(root))
    training_counts: dict[str, int] = {}
    for record in training:
        status = str(record["status"])
        training_counts[status] = training_counts.get(status, 0) + 1
    evaluation = evaluation_status(root)
    value: dict[str, object] = {
        "experiment_id": PROTOCOL_ID,
        "training_counts": dict(sorted(training_counts.items())),
        "evaluation_counts": evaluation["counts"],
        "result_blind": True,
        "economic_values_returned": False,
        "august_accessed": False,
    }
    if verbose:
        value["training_jobs"] = training
        value["evaluation_cells"] = evaluation["records"]
    return value


def _verify(root: Path, report_root: Path, *, require: bool, audit: bool) -> dict[str, object]:
    from .aggregation import verify_evidence_bundle
    from .replay import (
        audit_complete_evaluation_against_source,
        verify_complete_evaluation_manifest,
    )
    from .reporting import verify_report
    from .training import verify_complete_training_manifest

    assert_frozen_contracts()
    paths = {
        "training": root / "complete_training_manifest.json",
        "evaluation": root / "evaluation" / "manifest.json",
        "evidence": report_root / "source_data" / "manifest.json",
        "report": report_root / "report_manifest.json",
    }
    verifiers = {
        "training": verify_complete_training_manifest,
        "evaluation": verify_complete_evaluation_manifest,
        "evidence": verify_evidence_bundle,
        "report": verify_report,
    }
    records: dict[str, object] = {}
    for name, path in paths.items():
        if not path.is_file():
            if require:
                raise FileNotFoundError(f"required {name} artifact is absent: {path}")
            records[name] = {"status": "ABSENT"}
            continue
        verifiers[name](path)
        records[name] = {
            "status": "PASS",
            "path": path.relative_to(workspace_root()).as_posix(),
            "sha256": file_sha256(path),
        }
    if audit:
        if not paths["evaluation"].is_file():
            raise FileNotFoundError("source audit requires the complete evaluation manifest")
        audited = audit_complete_evaluation_against_source(paths["evaluation"])
        records["source_replay_audit"] = {
            "status": "PASS",
            "successful_cells_replayed": len(audited),
            "technical_failures_structurally_verified": 64 - len(audited),
        }
    return {
        "experiment_id": PROTOCOL_ID,
        "status": "PASS",
        "artifacts": records,
        "paper_modified": False,
        "august_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.command
    if command == "verify-contracts":
        contracts = assert_frozen_contracts()
        result: object = {
            "experiment_id": PROTOCOL_ID,
            "protocol_canonical_sha256": contracts.protocol_canonical_sha256,
            "status": "PASS",
        }
    elif command == "describe":
        result = _describe()
    elif command == "status":
        result = _status(_output_root(args.output_root), verbose=args.verbose)
    elif command == "train":
        from .training import train_member

        path = train_member(
            args.arm, args.seed, _output_root(args.output_root),
            resume=not args.no_resume,
        )
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif command == "train-all":
        from .training import train_all_learners

        path = train_all_learners(_output_root(args.output_root), workers=args.workers)
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif command == "verify-training":
        from .training import verify_complete_training_manifest

        path = _output_root(args.output_root) / "complete_training_manifest.json"
        verify_complete_training_manifest(path)
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif command == "replay":
        from .replay import replay_cell
        from .training import verify_complete_training_manifest

        root = _output_root(args.output_root)
        training = root / "complete_training_manifest.json"
        verify_complete_training_manifest(training)
        path = replay_cell(
            arm=args.arm, period=args.period, scenario=args.scenario,
            output_root=root,
            training_manifest=(
                training
                if args.arm in LEARNER_ARMS else None
            ),
        )
        result = {"status": "PASS", "report": str(path), "sha256": file_sha256(path)}
    elif command == "replay-all":
        from .replay import replay_all

        root = _output_root(args.output_root)
        path = replay_all(
            root, root / "complete_training_manifest.json", workers=args.workers,
        )
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif command == "aggregate-evaluation":
        from .replay import build_complete_evaluation_manifest

        path = build_complete_evaluation_manifest(_output_root(args.output_root))
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif command == "audit-source":
        from .replay import audit_complete_evaluation_against_source

        path = _output_root(args.output_root) / "evaluation" / "manifest.json"
        audited = audit_complete_evaluation_against_source(path)
        result = {
            "status": "PASS",
            "manifest": str(path),
            "successful_cells_replayed": len(audited),
            "technical_failures_structurally_verified": 64 - len(audited),
        }
    elif command == "build-evidence":
        from .aggregation import build_evidence_bundle

        path = build_evidence_bundle(
            _output_root(args.output_root), _report_root(args.report_root),
        )
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif command == "build-report":
        from .reporting import build_report

        root = _report_root(args.report_root)
        evidence = root / "source_data" / "manifest.json"
        path = build_report(evidence, root)
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif command == "verify":
        result = _verify(
            _output_root(args.output_root), _report_root(args.report_root),
            require=args.require_complete, audit=args.audit_source,
        )
    elif command == "verify-portable":
        from .portable import verify_portable_suite

        result = verify_portable_suite(
            _output_root(args.output_root), _report_root(args.report_root),
            require_complete=args.require_complete,
        )
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(f"unhandled command {command}")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
