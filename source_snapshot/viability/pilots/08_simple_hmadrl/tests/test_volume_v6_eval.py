from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from kx_simple_hmadrl.volume_v6_eval import (
    REQUIRED_ARMS,
    STRESS_SCENARIOS,
    apply_registered_gates,
    assert_test_unlocked,
    authorize_test,
    classify_dates,
    daily_metrics,
    expected_economics,
    flat_on_vote_tie,
)


def _report(arm: str, pnl: dict[str, int], trips: int, *, scenario: str = "base") -> dict[str, object]:
    return {
        "arm": arm,
        "protocol_sha256": "same",
        "evaluation_dates": sorted(pnl),
        "classification": "DEVELOPMENT",
        "economics": expected_economics(scenario),
        "ensemble_seeds": [1103, 2207, 3301, 4409, 5501],
        "scenario": scenario,
        "daily_pnl_cents": pnl,
        "metrics": daily_metrics(pnl, trips),
        "checkpoint_set": [arm],
        "registered_evaluation": True,
        "smoke_only": False,
    }


def test_period_guard_refuses_locked_test_as_smoke() -> None:
    assert classify_dates(["2025-08-29"], phase="smoke") == "DEVELOPMENT"
    with pytest.raises(PermissionError):
        classify_dates(["2026-06-01"], phase="smoke")


def test_vote_is_per_action_and_flat_on_tie() -> None:
    assert flat_on_vote_tie([-1, -1, 0, 1, -1]) == -1
    assert flat_on_vote_tie([-1, -1, 0, 1, 1]) == 0


def test_registered_gate_can_pass_a_strong_synthetic_case(monkeypatch: pytest.MonkeyPatch) -> None:
    days = [f"2026-05-{day:02d}" for day in range(1, 21)]
    rich = {day: (10_000 if index % 2 else 30_000) for index, day in enumerate(days)}
    nominal = {
        REQUIRED_ARMS[0]: _report(REQUIRED_ARMS[0], rich, 60),
        REQUIRED_ARMS[1]: _report(REQUIRED_ARMS[1], {day: 1_000 for day in days}, 60),
        REQUIRED_ARMS[2]: _report(REQUIRED_ARMS[2], {day: -2_000 for day in days}, 50),
        REQUIRED_ARMS[3]: _report(REQUIRED_ARMS[3], {day: -1_000 for day in days}, 55),
    }
    stresses: dict[str, dict[str, dict[str, object]]] = {}
    for scenario in STRESS_SCENARIOS:
        stresses[scenario] = {
            arm: _report(arm, dict(report["daily_pnl_cents"]), int(report["metrics"]["completed_round_trips"]), scenario=scenario)  # type: ignore[index,arg-type]
            for arm, report in nominal.items()
        }
        for arm in REQUIRED_ARMS:
            stresses[scenario][arm]["checkpoint_set"] = nominal[arm]["checkpoint_set"]
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_eval.frozen_protocol", lambda: ({}, "same")
    )
    result = apply_registered_gates(nominal, stresses, period="validation")
    assert result["verdict"] == "PASS"
    assert result["failed_gates"] == []


def test_unlock_refuses_before_august_is_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "artifact_type": "V6_REGISTERED_GATE",
        "period": "validation",
        "classification": "DEVELOPMENT",
        "verdict": "PASS",
        "failed_gates": [],
        "protocol_sha256": "same",
        "registered_evaluation": True,
        "smoke_only": False,
    }))
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_eval.frozen_protocol",
        lambda: ({"holdout": {"previously_inspected": False}}, "same"),
    )
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_eval._current_date", lambda: date(2026, 8, 23),
    )
    with pytest.raises(PermissionError, match="before August is complete"):
        authorize_test(gate, tmp_path / "unlock.json")


def test_unlock_rejects_self_consistent_but_fabricated_validation_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import kx_simple_hmadrl.volume_v6_eval as module
    import kx_simple_hmadrl.volume_v6_orchestration as orchestration

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_sha = "same"
    frozen = lambda: ({"holdout": {"previously_inspected": False}}, protocol_sha)
    monkeypatch.setattr(module, "frozen_protocol", frozen)
    monkeypatch.setattr(orchestration, "frozen_protocol", frozen)
    training = tmp_path / "complete-training.json"
    training.write_text(json.dumps({
        "artifact_type": "V6_COMPLETE_TRAINING_RUN",
        "protocol_sha256": protocol_sha,
        "complete": True,
        "checkpoint_count": 20,
        # Deliberately no deeply verifiable dataset, XAU parent, or children.
        "checkpoints": [],
    }))
    dataset = tmp_path / "may-dataset.json"
    dataset.write_text(json.dumps({
        "artifact_type": "V6_GC_CAUSAL_SESSION_DATASET",
        "protocol_sha256": protocol_sha,
        "phase": "validation",
        "classification": "DEVELOPMENT",
        "sessions": {},
    }))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "artifact_type": "V6_REGISTERED_GATE",
        "period": "validation",
        "classification": "DEVELOPMENT",
        "verdict": "PASS",
        "failed_gates": [],
        "protocol_sha256": protocol_sha,
        "registered_evaluation": True,
        "smoke_only": False,
        "complete_training_manifest_sha256": digest(training),
    }))
    reports: list[dict[str, object]] = []
    for scenario in ("base", *STRESS_SCENARIOS):
        for arm in REQUIRED_ARMS:
            report = tmp_path / "reports" / scenario / f"{len(reports)}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({
                "arm": arm,
                "scenario": scenario,
                "protocol_sha256": protocol_sha,
                "evaluation_dataset_manifest_sha256": digest(dataset),
                "complete_training_manifest_sha256": digest(training),
                "daily_pnl_cents": {"2026-05-01": 100},
                "metrics": daily_metrics({"2026-05-01": 100}, 1),
            }))
            reports.append({
                "scenario": scenario,
                "arm": arm,
                "path": str(report.relative_to(tmp_path)),
                "sha256": digest(report),
            })
    validation = tmp_path / "manifest.json"
    validation.write_text(json.dumps({
        "artifact_type": "V6_COMPLETE_REGISTERED_EVALUATION",
        "protocol_sha256": protocol_sha,
        "complete": True,
        "phase": "validation",
        "classification": "DEVELOPMENT",
        "verdict": "PASS",
        "failed_gates": [],
        "registered_evaluation": True,
        "smoke_only": False,
        "holdout_read": False,
        "arms": list(REQUIRED_ARMS),
        "scenarios": ["base", *STRESS_SCENARIOS],
        "report_count": len(reports),
        "reports": reports,
        "gate": {"path": str(gate), "sha256": digest(gate)},
        "training_manifest": {"path": str(training), "sha256": digest(training)},
        "evaluation_dataset_manifest": {
            "path": str(dataset), "sha256": digest(dataset),
        },
        "unlock_receipt": None,
        "test_output_consumption": None,
    }))
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_eval._current_date", lambda: date(2026, 9, 1),
    )
    with pytest.raises(PermissionError, match="deep dataset/checkpoint/gate verification"):
        module._validation_unlock_chain(validation, protocol_sha)
    unlock = tmp_path / "unlock.json"
    with pytest.raises(PermissionError, match="deep dataset/checkpoint/gate verification"):
        authorize_test(validation, unlock)
    assert not unlock.exists()
