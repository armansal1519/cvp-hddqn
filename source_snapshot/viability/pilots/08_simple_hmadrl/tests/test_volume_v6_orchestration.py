from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from kx_simple_hmadrl.volume_v6_cli import parser
from kx_simple_hmadrl.volume_v6_contract import SEEDS
from kx_simple_hmadrl.volume_v6_eval import (
    REQUIRED_ARMS,
    STRESS_SCENARIOS,
    apply_registered_gates,
    assert_test_unlocked,
    authorize_test,
    daily_metrics,
    expected_economics,
    frozen_protocol,
)
from kx_simple_hmadrl.volume_v6_orchestration import (
    ARM_KEYS,
    PhaseDataset,
    create_complete_training_manifest,
    evaluate_registered_period,
    load_phase_dataset,
    status,
    verify_evaluation_manifest,
    verify_pretraining_manifest,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_status_does_not_enumerate_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_orchestration._current_date",
        lambda: date(2026, 8, 23),
    )
    value = status()
    assert value["market_data_roots_enumerated"] is False
    assert value["test_calendar_complete"] is False
    assert value["test_earliest_open_date"] == "2026-09-01"


def test_cli_rejects_more_than_twelve_workers_before_running() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args([
            "train", "--dataset-manifest", "data.json",
            "--pretraining-manifest", "pretrain.json", "--output", "out",
            "--workers", "13",
        ])


def test_cli_exposes_authoritative_universe_before_map_registration() -> None:
    args = parser().parse_args([
        "create-active-map-universe", "--phase", "train",
        "--archive-root", "archive", "--active-map", "active.parquet",
        "--start", "2025-01-01", "--end-exclusive", "2026-05-01",
        "--output", "eligible-universe.json",
    ])
    assert args.command == "create-active-map-universe"
    assert args.phase == "train"
    register = parser().parse_args([
        "register-active-map", "--active-map", "active.parquet",
        "--source-manifest", "eligible-universe.json",
    ])
    assert register.source_manifest == Path("eligible-universe.json")


def test_may_guard_runs_before_dataset_manifest_access(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="complete V6 training"):
        load_phase_dataset(
            tmp_path / "locked-may" / "manifest.json", phase="validation",
            complete_training_manifest=tmp_path / "missing-training.json",
        )


def test_test_calendar_guard_runs_before_any_artifact_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    monkeypatch.setattr(module, "_current_date", lambda: date(2026, 8, 31))
    with pytest.raises(PermissionError, match="2026-09-01"):
        evaluate_registered_period(
            tmp_path / "missing-training.json", tmp_path / "locked-test.json",
            tmp_path / "result", phase="test",
            unlock_receipt=tmp_path / "missing-unlock.json",
        )


def test_test_cannot_substitute_or_refit_the_may_bound_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    selected = tmp_path / "jan-april-selected.json"
    selected.write_bytes(b"frozen checkpoint receipt")
    substituted = tmp_path / "post-validation-substitute.json"
    substituted.write_bytes(b"different or retrained checkpoints")
    validation = tmp_path / "complete-may-evaluation.json"
    validation.write_bytes(b"deeply verified May evaluation")
    monkeypatch.setattr(
        module, "assert_test_unlocked", lambda path: {
            "validation_evaluation_manifest": str(validation),
            "validation_evaluation_manifest_sha256": _sha(validation),
            "complete_training_manifest": str(selected),
            "complete_training_manifest_sha256": _sha(selected),
        },
    )
    monkeypatch.setattr(
        module, "verify_evaluation_manifest",
        lambda path: {"phase": "validation", "verdict": "PASS"},
    )
    monkeypatch.setattr(module, "_current_date", lambda: date(2026, 9, 1))
    with pytest.raises(PermissionError, match="differ from the immutable Jan-Apr"):
        evaluate_registered_period(
            substituted,
            tmp_path / "locked-test.json", tmp_path / "result", phase="test",
            unlock_receipt=tmp_path / "synthetic-unlock.json",
        )


def test_test_unlock_consumption_allows_resume_only_to_same_output(
    tmp_path: Path,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    validation = tmp_path / "validation" / "manifest.json"
    validation.parent.mkdir()
    validation.write_bytes(b"May evaluation")
    unlock = tmp_path / "unlock.json"
    _write_json(unlock, {"validation_evaluation_manifest": str(validation)})
    first = module._claim_test_output(unlock, tmp_path / "test-result")
    assert module._claim_test_output(unlock, tmp_path / "test-result") == first
    with pytest.raises(PermissionError, match="another output"):
        module._claim_test_output(unlock, tmp_path / "selected-after-test")


def test_phase_loader_uses_only_manifest_bound_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "train"
    raw = root / "sessions" / "2025-01-02.npz"
    side = root / "sidecars" / "2025-01-02.npz"
    raw.parent.mkdir(parents=True); side.parent.mkdir(parents=True)
    raw.write_bytes(b"registered raw")
    side.write_bytes(b"registered sidecar")
    # This deliberately exists beside the registered shard.  A glob-based
    # loader would discover it; the V6 loader must never inspect or hash it.
    (root / "sessions" / "2026-06-01.npz").write_bytes(b"locked decoy")
    active_map = root / "active.parquet"
    active_receipt = root / "active.parquet.receipt.json"
    active_map.write_bytes(b"strict map")
    active_receipt.write_bytes(b"map receipt")

    class _Selected:
        def __getitem__(self, name: str) -> list[str]:
            return {
                "v6_session_date": ["2025-01-02"],
                "selected_raw_symbol": ["GCG5"],
            }[name]

    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_acquisition.validate_active_contract_map",
        lambda *args, **kwargs: _Selected(),
    )
    _, protocol_sha = frozen_protocol()
    manifest = {
        "schema_version": 1,
        "artifact_type": "V6_GC_CAUSAL_SESSION_DATASET",
        "protocol_sha256": protocol_sha,
        "phase": "train",
        "classification": "DEVELOPMENT",
        "start": "2025-01-01",
        "end_exclusive": "2026-05-01",
        "session_count": 1,
        "active_contract_map": {"path": str(active_map), "sha256": _sha(active_map)},
        "active_contract_map_receipt": {
            "path": str(active_receipt), "sha256": _sha(active_receipt),
        },
        "sessions": {
            "2025-01-02": {
                "raw_symbol": "GCG5",
                "session_sha256": _sha(raw), "sidecar_sha256": _sha(side),
            }
        },
        "audits": [{
            "session_date": "2025-01-02", "status": "PASS", "continuity_segments": 1,
        }],
        "access_authorization": None,
    }
    _write_json(root / "manifest.json", manifest)
    loaded = load_phase_dataset(root / "manifest.json", phase="train")
    assert loaded.dates == ("2025-01-02",)
    assert loaded.raw_paths == (raw,)
    manifest["sessions"]["2025-01-02"]["raw_symbol"] = "GCJ5"  # type: ignore[index]
    _write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="exact receipt-bound phase universe"):
        load_phase_dataset(root / "manifest.json", phase="train")


def test_complete_training_receipt_requires_twenty_seed_arm_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    dataset_manifest = tmp_path / "dataset" / "manifest.json"
    dataset_manifest.parent.mkdir(parents=True)
    dataset_manifest.write_bytes(b"dataset receipt")
    pretraining = tmp_path / "pretrain" / "manifest.json"
    pretraining.parent.mkdir(parents=True)
    pretraining.write_bytes(b"pretraining receipt")
    training_root = tmp_path / "training"
    for arm in REQUIRED_ARMS:
        for seed in SEEDS:
            one = training_root / ARM_KEYS[arm] / f"seed_{seed}" / "manifest.json"
            _write_json(one, {"arm": arm, "seed": seed})

    dataset = PhaseDataset(
        dataset_manifest, "train", ("2025-01-02",), (), (), _sha(dataset_manifest),
    )
    monkeypatch.setattr(module, "verify_pretraining_manifest", lambda *a, **k: {})
    monkeypatch.setattr(module, "_verify_training_artifact", lambda *a, **k: {})
    monkeypatch.setattr(module, "load_phase_dataset", lambda *a, **k: dataset)
    result = create_complete_training_manifest(
        training_root, dataset, pretraining,
        training_root / "complete_training_manifest.json",
    )
    value = json.loads(result.read_text())
    assert value["checkpoint_count"] == 20
    assert {(one["arm"], one["seed"]) for one in value["checkpoints"]} == {
        (arm, seed) for arm in REQUIRED_ARMS for seed in SEEDS
    }


def test_complete_xau_receipt_requires_one_common_direct_source(tmp_path: Path) -> None:
    _, protocol_sha = frozen_protocol()
    direct = tmp_path / "direct" / "manifest.json"
    _write_json(direct, {"source": "Dukascopy"})
    direct_sha = _sha(direct)
    root = tmp_path / "pretrain"
    records: list[dict[str, object]] = []
    for kind in ("v6_price", "base_paper_gc", "base_strong_gc_v6"):
        for seed in SEEDS:
            child = root / kind / f"seed_{seed}"
            product = child / "weights.pt"
            product.parent.mkdir(parents=True, exist_ok=True)
            product.write_bytes(f"{kind}:{seed}".encode())
            manifest_path = child / "manifest.json"
            if kind == "v6_price":
                resume = child / "resume.pt"
                resume.write_bytes(b"resume")
                value = {
                    "artifact_type": "V6_XAU_PRICE_PRETRAINING",
                    "protocol_sha256": protocol_sha,
                    "seed": seed,
                    "source_manifest_sha256": direct_sha,
                    "gc_transfer_eligible": True,
                    "model_bundle": {"path": product.name, "sha256": _sha(product)},
                    "resume_checkpoint": {"path": resume.name, "sha256": _sha(resume)},
                }
            else:
                value = {
                    "schema_version": 1,
                    "artifact_type": "V6_BASELINE_XAU_PRETRAIN",
                    "complete": True,
                    "protocol_sha256": protocol_sha,
                    "variant": kind,
                    "seed": seed,
                    "source_manifest_sha256": direct_sha,
                    "smoke_only": False,
                    "artifacts": [{"path": product.name, "sha256": _sha(product)}],
                }
            _write_json(manifest_path, value)
            records.append({
                "kind": kind, "seed": seed,
                "manifest": str(manifest_path.relative_to(root)),
                "manifest_sha256": _sha(manifest_path),
            })
    aggregate = root / "manifest.json"
    value = {
        "artifact_type": "V6_COMPLETE_XAU_PRETRAINING",
        "protocol_sha256": protocol_sha,
        "complete": True,
        "smoke_only": False,
        "seeds": list(SEEDS),
        "components": ["v6_price", "base_paper_gc", "base_strong_gc_v6"],
        "direct_xau_manifest": {"path": str(direct), "sha256": direct_sha},
        "artifacts": records,
    }
    _write_json(aggregate, value)
    assert verify_pretraining_manifest(aggregate)["complete"] is True

    child = root / "base_paper_gc" / f"seed_{SEEDS[0]}" / "manifest.json"
    changed = json.loads(child.read_text())
    changed["source_manifest_sha256"] = "0" * 64
    _write_json(child, changed)
    next(record for record in records if record["kind"] == "base_paper_gc" and record["seed"] == SEEDS[0])["manifest_sha256"] = _sha(child)
    _write_json(aggregate, value)
    with pytest.raises(ValueError, match="source differs"):
        verify_pretraining_manifest(aggregate)


def test_registered_replay_runs_all_arms_and_stresses_through_one_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    training_manifest = tmp_path / "training.json"
    training_manifest.write_bytes(b"complete training")
    training_data_manifest = tmp_path / "train-data.json"
    training_data_manifest.write_bytes(b"train data")
    evaluation_data_manifest = tmp_path / "may-data.json"
    evaluation_data_manifest.write_bytes(b"May data")
    complete = {
        "dataset_manifest": {"path": str(training_data_manifest), "sha256": _sha(training_data_manifest)},
        "checkpoints": [],
    }
    train_data = PhaseDataset(
        training_data_manifest, "train", ("2025-01-02",), (), (), _sha(training_data_manifest),
    )
    eval_data = PhaseDataset(
        evaluation_data_manifest, "validation", ("2026-05-01",), (), (), _sha(evaluation_data_manifest),
    )
    monkeypatch.setattr(module, "verify_complete_training_manifest", lambda path: complete)
    monkeypatch.setattr(
        module, "load_phase_dataset",
        lambda path, **kwargs: train_data if kwargs["phase"] == "train" else eval_data,
    )
    monkeypatch.setattr(module, "_training_roots", lambda *a, **k: tuple(Path(str(seed)) for seed in SEEDS))

    calls: list[tuple[str, str]] = []

    def fake_evaluate(arm: str, *args: object, scenario: str, **kwargs: object) -> dict[str, object]:
        calls.append((scenario, arm))
        return {"arm": arm, "scenario": scenario}

    captured: dict[str, object] = {}

    def fake_gate(nominal: object, stresses: object, *, period: str) -> dict[str, object]:
        captured.update({"nominal": nominal, "stresses": stresses, "period": period})
        return {"verdict": "FAIL", "failed_gates": ["synthetic"]}

    monkeypatch.setattr(module, "_evaluate_one_arm", fake_evaluate)
    monkeypatch.setattr(module, "apply_registered_gates", fake_gate)
    # Final verification is separately unit-tested by its hash contract; avoid
    # teaching this scheduling test the entire report schema.
    monkeypatch.setattr(module, "verify_evaluation_manifest", lambda path: {})
    result = evaluate_registered_period(
        training_manifest, evaluation_data_manifest, tmp_path / "evaluation",
        phase="validation",
    )
    assert result.is_file()
    assert len(calls) == 4 * 4
    assert set(calls) == {
        (scenario, arm)
        for scenario in ("base", *STRESS_SCENARIOS)
        for arm in REQUIRED_ARMS
    }
    assert set(captured["nominal"]) == set(REQUIRED_ARMS)  # type: ignore[arg-type]
    assert set(captured["stresses"]) == set(STRESS_SCENARIOS)  # type: ignore[arg-type]
    assert captured["period"] == "validation"


def test_complete_evaluation_deeply_recomputes_metrics_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    _, protocol_sha = frozen_protocol()
    training = tmp_path / "complete-training.json"
    training.write_bytes(b"training receipt")
    dataset_manifest = tmp_path / "may-data.json"
    dataset_manifest.write_bytes(b"May dataset receipt")
    days = (
        "2026-05-01", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-07",
    )
    dataset = PhaseDataset(
        dataset_manifest, "validation", days, (), (), _sha(dataset_manifest),
    )
    checkpoint_records = [
        {"arm": arm, "seed": seed, "manifest_sha256": f"{arm}:{seed}"}
        for arm in REQUIRED_ARMS for seed in SEEDS
    ]
    complete = {"checkpoints": checkpoint_records}
    monkeypatch.setattr(module, "verify_complete_training_manifest", lambda path: complete)
    monkeypatch.setattr(module, "load_phase_dataset", lambda *a, **k: dataset)

    result_root = tmp_path / "validation-result"
    nominal: dict[str, dict[str, object]] = {}
    stresses: dict[str, dict[str, dict[str, object]]] = {
        scenario: {} for scenario in STRESS_SCENARIOS
    }
    report_records: list[dict[str, object]] = []
    training_sha = _sha(training)
    for scenario in ("base", *STRESS_SCENARIOS):
        for arm in REQUIRED_ARMS:
            checkpoints = [
                {"seed": seed, "manifest_sha256": f"{arm}:{seed}"} for seed in SEEDS
            ]
            if arm == "V6-RichVolume":
                daily = {day: 10_000 + index * 1_000 for index, day in enumerate(days)}
                trips = 60
            elif arm == "V6-BaseOHLCV":
                daily = {day: 0 for day in days}
                trips = 60
            elif arm == "BasePaper-XAU-Pretrained-GC":
                daily = {day: -2_000 + (index % 2) * 500 for index, day in enumerate(days)}
                trips = 50
            else:
                daily = {day: -1_000 + (index % 2) * 250 for index, day in enumerate(days)}
                trips = 55
            report: dict[str, object] = {
                "artifact_type": (
                    "V6_COMMON_LEDGER_REPLAY" if arm.startswith("V6-")
                    else "V6_BASELINE_COMMON_LEDGER_REPLAY"
                ),
                "protocol_sha256": protocol_sha,
                "arm": arm,
                "phase": "validation",
                "classification": "DEVELOPMENT",
                "scenario": scenario,
                "economics": expected_economics(scenario),
                "ensemble_seeds": list(SEEDS),
                "evaluation_dates": list(days),
                "daily_pnl_cents": daily,
                "metrics": daily_metrics(daily, trips),
                "checkpoint_set": checkpoints,
                "registered_evaluation": True,
                "smoke_only": False,
                "holdout_read": False,
                "evaluation_dataset_manifest_sha256": _sha(dataset_manifest),
                "complete_training_manifest_sha256": training_sha,
                "evaluation_dataset_manifest_path": str(dataset_manifest.resolve()),
                "complete_training_manifest_path": str(training.resolve()),
            }
            report_path = result_root / scenario / f"{ARM_KEYS[arm]}.json"
            _write_json(report_path, report)
            report_records.append({
                "scenario": scenario, "arm": arm,
                "path": str(report_path.relative_to(result_root)),
                "sha256": _sha(report_path),
            })
            if scenario == "base":
                nominal[arm] = report
            else:
                stresses[scenario][arm] = report
    gate = apply_registered_gates(nominal, stresses, period="validation")
    gate["complete_training_manifest"] = str(training.resolve())
    gate["complete_training_manifest_sha256"] = training_sha
    gate_path = result_root / "gate.json"
    _write_json(gate_path, gate)
    manifest = {
        "artifact_type": "V6_COMPLETE_REGISTERED_EVALUATION",
        "protocol_sha256": protocol_sha,
        "complete": True,
        "phase": "validation",
        "classification": "DEVELOPMENT",
        "scenarios": ["base", *STRESS_SCENARIOS],
        "arms": list(REQUIRED_ARMS),
        "report_count": 16,
        "registered_evaluation": True,
        "smoke_only": False,
        "holdout_read": False,
        "training_manifest": {"path": str(training), "sha256": training_sha},
        "evaluation_dataset_manifest": {
            "path": str(dataset_manifest), "sha256": _sha(dataset_manifest),
        },
        "unlock_receipt": None,
        "test_output_consumption": None,
        "reports": report_records,
        "gate": {"path": "gate.json", "sha256": _sha(gate_path)},
        "verdict": gate["verdict"],
        "failed_gates": gate["failed_gates"],
    }
    manifest_path = result_root / "manifest.json"
    _write_json(manifest_path, manifest)
    assert verify_evaluation_manifest(manifest_path)["verdict"] == gate["verdict"]
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_eval._current_date", lambda: date(2026, 9, 1),
    )
    unlock = authorize_test(
        manifest_path, tmp_path / "unlock.json",
    )
    assert assert_test_unlocked(unlock)["validation_evaluation_manifest"] == str(
        manifest_path.resolve()
    )

    first = result_root / report_records[0]["path"]  # type: ignore[operator]
    changed = json.loads(first.read_text())
    changed["metrics"]["net_pnl_usd"] = 999.0
    _write_json(first, changed)
    report_records[0]["sha256"] = _sha(first)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="metrics do not recompute"):
        verify_evaluation_manifest(manifest_path)
    with pytest.raises(PermissionError):
        assert_test_unlocked(unlock)
