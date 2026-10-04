"""Result-blind command line for the Pilot 11 reduced suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import file_sha256
from .contract import (
    CANONICAL_OUTPUT_ROOT,
    LEARNER_ARMS,
    PERIOD_CLASSIFICATION,
    PERIODS,
    PROTOCOL_ID,
    REFERENCE_ARMS,
    REQUIRED_EVALUATION_CELLS,
    REQUIRED_TRAINING_JOBS,
    SCENARIOS,
    SEEDS,
)
from .protocol import assert_frozen_contracts, validate_workspace_output


def _root(value: str | None) -> Path:
    return validate_workspace_output(
        CANONICAL_OUTPUT_ROOT if value is None else value,
    )


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kx-cvp-reduced-baselines")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-contracts")
    sub.add_parser("describe")
    status = sub.add_parser("status")
    _add_root(status)
    receipt = sub.add_parser("build-training-receipt")
    _add_root(receipt)
    verify_receipt = sub.add_parser("verify-training-receipt")
    _add_root(verify_receipt)
    replay = sub.add_parser("replay")
    replay.add_argument("arm", choices=LEARNER_ARMS)
    replay.add_argument("period", choices=PERIODS)
    replay.add_argument("scenario", choices=SCENARIOS)
    _add_root(replay)
    replay_all = sub.add_parser("replay-all")
    _add_root(replay_all)
    replay_all.add_argument("--workers", type=int, default=1)
    failure = sub.add_parser("record-technical-failure")
    failure.add_argument("arm", choices=LEARNER_ARMS)
    failure.add_argument("period", choices=PERIODS)
    failure.add_argument("scenario", choices=SCENARIOS)
    failure.add_argument("reason")
    failure.add_argument("diagnostic")
    failure.add_argument("--holdout-read", action="store_true")
    _add_root(failure)
    aggregate = sub.add_parser("aggregate-evaluation")
    _add_root(aggregate)
    audit = sub.add_parser("audit-source")
    _add_root(audit)
    verify = sub.add_parser("verify")
    _add_root(verify)
    verify.add_argument("--require-complete", action="store_true")
    verify.add_argument("--audit-source", action="store_true")
    return parser


def _describe() -> dict[str, object]:
    frozen = assert_frozen_contracts()
    return {
        "experiment_id": PROTOCOL_ID,
        "protocol_canonical_sha256": frozen.protocol_canonical_sha256,
        "learner_arms": list(LEARNER_ARMS),
        "reference_arms": list(REFERENCE_ARMS),
        "seeds": list(SEEDS),
        "periods": list(PERIODS),
        "scenarios": list(SCENARIOS),
        "period_classification": PERIOD_CLASSIFICATION,
        "adopted_training_jobs": REQUIRED_TRAINING_JOBS,
        "required_evaluation_cells": REQUIRED_EVALUATION_CELLS,
        "new_training_performed": False,
        "flat_ddqn_included": False,
        "result_dependent_selection": False,
        "august_2026_access_authorized": False,
        "paper_integration": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _root(getattr(args, "output_root", None)) if hasattr(
        args, "output_root"
    ) else None
    if args.command == "verify-contracts":
        frozen = assert_frozen_contracts()
        result: object = {
            "experiment_id": PROTOCOL_ID,
            "protocol_canonical_sha256": frozen.protocol_canonical_sha256,
            "status": "PASS",
        }
    elif args.command == "describe":
        result = _describe()
    elif args.command == "status":
        from .replay import evaluation_status

        result = evaluation_status(root)
    elif args.command == "build-training-receipt":
        from .training_receipt import build_adopted_training_receipt

        path = build_adopted_training_receipt(root)
        result = {"status": "PASS", "receipt": str(path), "sha256": file_sha256(path)}
    elif args.command == "verify-training-receipt":
        from .training_receipt import (
            ADOPTED_TRAINING_FILENAME,
            verify_adopted_training_receipt,
        )

        assert root is not None
        path = root / ADOPTED_TRAINING_FILENAME
        verify_adopted_training_receipt(path)
        result = {"status": "PASS", "receipt": str(path), "sha256": file_sha256(path)}
    elif args.command == "replay":
        from .replay import replay_cell

        path = replay_cell(
            arm=args.arm, period=args.period, scenario=args.scenario,
            output_root=root,
        )
        result = {"status": "PASS", "report": str(path), "sha256": file_sha256(path)}
    elif args.command == "replay-all":
        from .replay import replay_all

        path = replay_all(root, workers=args.workers)
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif args.command == "record-technical-failure":
        from .replay import write_technical_failure

        path = write_technical_failure(
            arm=args.arm, period=args.period, scenario=args.scenario,
            reason=args.reason, diagnostic=args.diagnostic,
            output_root=root, holdout_read=args.holdout_read,
        )
        result = {
            "status": "PASS", "technical_failure": str(path),
            "sha256": file_sha256(path),
        }
    elif args.command == "aggregate-evaluation":
        from .replay import build_complete_evaluation_manifest

        path = build_complete_evaluation_manifest(root)
        result = {"status": "PASS", "manifest": str(path), "sha256": file_sha256(path)}
    elif args.command == "audit-source":
        from .replay import audit_complete_evaluation_against_source

        assert root is not None
        path = root / "evaluation" / "manifest.json"
        audits = audit_complete_evaluation_against_source(path)
        result = {
            "status": "PASS",
            "manifest": str(path),
            "successful_cells_replayed": len(audits),
            "technical_failures_structurally_verified": (
                REQUIRED_EVALUATION_CELLS - len(audits)
            ),
        }
    else:
        from .replay import verify_complete_evaluation_manifest
        from .training_receipt import (
            ADOPTED_TRAINING_FILENAME,
            verify_adopted_training_receipt,
        )

        assert root is not None
        receipt = root / ADOPTED_TRAINING_FILENAME
        evaluation = root / "evaluation" / "manifest.json"
        artifacts: dict[str, object] = {}
        if receipt.is_file():
            verify_adopted_training_receipt(receipt)
            artifacts["training_receipt"] = {
                "status": "PASS", "path": str(receipt),
                "sha256": file_sha256(receipt),
            }
        elif args.require_complete:
            raise FileNotFoundError(receipt)
        else:
            artifacts["training_receipt"] = {"status": "ABSENT"}
        if evaluation.is_file():
            verify_complete_evaluation_manifest(evaluation)
            artifacts["evaluation"] = {
                "status": "PASS", "path": str(evaluation),
                "sha256": file_sha256(evaluation),
            }
        elif args.require_complete:
            raise FileNotFoundError(evaluation)
        else:
            artifacts["evaluation"] = {"status": "ABSENT"}
        source_audit: dict[str, object] | None = None
        if args.audit_source:
            if not evaluation.is_file():
                raise FileNotFoundError(
                    "source audit requires the complete evaluation manifest"
                )
            from .replay import audit_complete_evaluation_against_source

            audits = audit_complete_evaluation_against_source(evaluation)
            source_audit = {
                "status": "PASS",
                "successful_cells_replayed": len(audits),
                "technical_failures_structurally_verified": (
                    REQUIRED_EVALUATION_CELLS - len(audits)
                ),
            }
        result = {
            "experiment_id": PROTOCOL_ID,
            "status": "PASS",
            "artifacts": artifacts,
            "paper_modified": False,
            "august_2026_accessed": False,
        }
        if source_audit is not None:
            result["source_replay_audit"] = source_audit
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
