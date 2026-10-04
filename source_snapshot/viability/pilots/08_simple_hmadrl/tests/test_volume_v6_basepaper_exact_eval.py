from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kx_base_gc.contract import registered_contract
from kx_simple_hmadrl import volume_v6_basepaper_exact_eval as evaluation
from kx_simple_hmadrl.volume_v6_basepaper_exact_eval import (
    AgentGreedyTrace,
    greedy_joint_inference,
    majority_flat_trace,
    verify_corrected_freeze,
)


class _GreedyPolicy:
    def __init__(self, spec, actions):
        self.spec = spec
        self.actions = iter(actions)
        self.observations: list[np.ndarray] = []

    def greedy(self, observation: np.ndarray) -> int:
        self.observations.append(np.asarray(observation).copy())
        return int(next(self.actions))


def _bars() -> pd.DataFrame:
    count = 7
    ordinal = np.arange(count, dtype=np.float64)
    return pd.DataFrame({
        "event_time": pd.date_range(
            "2026-05-01 12:25Z", periods=count, freq="5min",
        ),
        "open": 100.0 + ordinal,
        "high": 101.0 + ordinal,
        "low": 99.0 + ordinal,
        "close": 100.5 + ordinal,
        "volume": 10.0 + ordinal,
        "raw_symbol": "GCM6",
        "state_id": "GCM6:one",
        "session_date": "2026-05-01",
    })


def _policies_and_frames():
    contract = registered_contract("base_paper_gc")
    specs = {
        name: replace(contract.agent(name), lookback=2, action_lock_bars=2)
        for name in ("H1", "M15", "M5")
    }
    policies = {
        "H1": _GreedyPolicy(specs["H1"], (1, -1, 0)),
        "M15": _GreedyPolicy(specs["M15"], (-1, 1, 0)),
        "M5": _GreedyPolicy(specs["M5"], (0, 1, -1)),
    }
    frames = {name: _bars() for name in specs}
    return policies, frames


def test_joint_greedy_inference_publishes_current_parents_and_held_m5_target() -> None:
    policies, frames = _policies_and_frames()
    trace = greedy_joint_inference(policies, frames)

    # H1 acts first, M15 consumes that current H1 position, then M5 consumes
    # both current positions at the same completed-bar timestamp.
    assert policies["M15"].observations[0][-1] == 1
    np.testing.assert_array_equal(
        policies["M5"].observations[0][-2:],
        np.asarray([1, -1], dtype=np.float32),
    )
    m5 = trace.agents["M5"]
    assert m5.actions.tolist() == [0, 0, 1, 1, -1, -1]
    assert m5.renewal_mask.tolist() == [True, False, True, False, True, False]
    assert len(policies["M5"].observations) == 3


def test_pre_may_rows_are_market_lookback_only_and_policy_state_starts_flat() -> None:
    policies, frames = _policies_and_frames()
    context_times = pd.to_datetime(
        ["2026-04-30 12:20Z", "2026-04-30 12:25Z"], utc=True,
    )
    validation_times = pd.date_range(
        "2026-05-01 12:25Z", periods=5, freq="5min",
    )
    times = context_times.append(validation_times)
    for name in frames:
        frames[name] = frames[name].copy()
        frames[name]["event_time"] = times
        frames[name]["session_date"] = [
            "2026-04-30", "2026-04-30", *(["2026-05-01"] * 5),
        ]
    starts = {name: 2 for name in frames}

    trace = greedy_joint_inference(
        policies, frames, evaluation_start_indexes=starts,
    )

    expected_times = validation_times.array.asi8
    for name in ("H1", "M15", "M5"):
        np.testing.assert_array_equal(trace.agents[name].times_ns, expected_times)
        assert trace.agents[name].renewal_mask[0]
        # No April row consumed a policy decision or advanced its lock clock.
        assert len(policies[name].observations) == 3
    assert trace.agents["M5"].actions.tolist() == [0, 0, 1, 1, -1]

    changed_policies, changed_frames = _policies_and_frames()
    for name in changed_frames:
        changed_frames[name] = changed_frames[name].copy()
        changed_frames[name]["event_time"] = times
        changed_frames[name]["session_date"] = frames[name]["session_date"]
        # This April row lies outside the two-row first-May market window.
        changed_frames[name].loc[0, ["open", "high", "low", "close", "volume"]] = 9e6
    changed = greedy_joint_inference(
        changed_policies, changed_frames, evaluation_start_indexes=starts,
    )
    np.testing.assert_array_equal(
        changed.agents["M5"].actions, trace.agents["M5"].actions,
    )
    for name in ("H1", "M15", "M5"):
        np.testing.assert_array_equal(
            changed_policies[name].observations[0], policies[name].observations[0],
        )


def test_five_seed_vote_is_flat_on_a_two_two_one_tie() -> None:
    times = np.asarray([1, 2], dtype=np.int64)
    renewals = np.asarray([True, False])
    actions = (
        (-1, 1),
        (-1, 1),
        (0, 1),
        (0, -1),
        (1, -1),
    )
    traces = [
        AgentGreedyTrace(
            times_ns=times,
            actions=np.asarray(value, dtype=np.int8),
            renewal_mask=renewals,
        )
        for value in actions
    ]
    voted = majority_flat_trace(traces)
    assert voted.actions.tolist() == [0, 1]
    assert voted.member_actions.shape == (5, 2)


def test_corrected_freeze_uses_canonical_parsed_protocol_digest(tmp_path) -> None:
    trainer = Path(evaluation.exact.__file__).resolve()
    evaluator = Path(evaluation.__file__).resolve()
    protocol_value = {
        "strategy_version": evaluation.CORRECTION_STRATEGY_VERSION,
        "z": [3, 2, 1],
        "a": {"right": True, "left": False},
        "artifacts": {
            "trainer_code": {
                "path": str(trainer),
                "sha256": evaluation.file_sha256(trainer),
            },
            "evaluator_code": {
                "path": str(evaluator),
                "sha256": evaluation.file_sha256(evaluator),
            },
        },
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_value, indent=4) + "\n", encoding="utf-8")
    canonical = evaluation.canonical_json_sha256(protocol_value)
    raw = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert raw != canonical
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": evaluation.FREEZE_ARTIFACT,
        "protocol_path": protocol.name,
        "protocol_sha256": canonical,
        "validator_result": "PASS",
        "created_at_utc": "2026-08-24T00:00:00Z",
    }) + "\n", encoding="utf-8")

    _, actual = verify_corrected_freeze(protocol, receipt)
    assert actual == canonical


def test_corrected_freeze_rejects_changed_bound_artifact(tmp_path) -> None:
    trainer = Path(evaluation.exact.__file__).resolve()
    evaluator = Path(evaluation.__file__).resolve()
    bound = tmp_path / "bound.txt"
    bound.write_text("frozen\n", encoding="utf-8")
    protocol_value = {
        "strategy_version": evaluation.CORRECTION_STRATEGY_VERSION,
        "artifacts": {
            "trainer_code": {
                "path": str(trainer),
                "sha256": evaluation.file_sha256(trainer),
            },
            "evaluator_code": {
                "path": str(evaluator),
                "sha256": evaluation.file_sha256(evaluator),
            },
            "bound_input": {
                "path": str(bound),
                "sha256": evaluation.file_sha256(bound),
            },
        },
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(protocol_value) + "\n", encoding="utf-8")
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": evaluation.FREEZE_ARTIFACT,
        "protocol_path": protocol.name,
        "protocol_sha256": evaluation.canonical_json_sha256(protocol_value),
        "validator_result": "PASS",
        "created_at_utc": "2026-08-24T00:00:00Z",
    }) + "\n", encoding="utf-8")
    bound.write_text("changed\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="absent or changed"):
        verify_corrected_freeze(protocol, receipt)


def test_prepared_gc_source_chain_verifies_receipt_and_every_product_hash(
    tmp_path, monkeypatch,
) -> None:
    dates = ["2026-01-02"]
    monkeypatch.setitem(
        evaluation.exact.EXPECTED_PHASE_DATE_SHA256,
        "train",
        evaluation.canonical_json_sha256(dates),
    )
    artifacts = []
    for name in ("H1", "M15", "M5"):
        product = tmp_path / f"{name.lower()}.parquet"
        product.write_bytes(f"synthetic-{name}".encode())
        artifacts.append({
            "agent": name,
            "path": product.name,
            "sha256": evaluation.file_sha256(product),
            "rows": evaluation.exact.EXPECTED_GC_TRAIN_BARS[name],
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "artifact_type": "BASEPAPER_DESCRIBED_PREPARED_BARS",
        "name": evaluation.exact.PREPARED_DATA_NAME,
        "phase": "train",
        "complete": True,
        "smoke_only": False,
        "dates": dates,
        "source_manifest_sha256": (
            evaluation.exact.EXPECTED_GC_MANIFEST_SHA256["train"]
        ),
        "source_digest": "1" * 64,
        "source_records_sha256": "2" * 64,
        "artifacts": artifacts,
    }) + "\n", encoding="utf-8")
    receipt = tmp_path / "manifest.receipt.json"
    receipt.write_text(json.dumps({
        "manifest_sha256": evaluation.file_sha256(manifest),
    }) + "\n", encoding="utf-8")

    evaluation._validate_training_source_outer(manifest)
    (tmp_path / "m5.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="M5 product differs"):
        evaluation._validate_training_source_outer(manifest)


def test_future_date_is_hard_blocked() -> None:
    with pytest.raises(PermissionError, match="hard-blocks every date after"):
        evaluation._assert_may_dates(
            ["2026-05-29", "2026-06-01"], label="synthetic request",
        )


@pytest.mark.parametrize("omit_parent_marker", [False, True])
def test_exact_chain_rejects_any_pretrained_parent_or_missing_null_marker(
    tmp_path, monkeypatch, omit_parent_marker,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    request = {
        "code_sha256": evaluation.file_sha256(Path(evaluation.exact.__file__)),
        "effective_contract_sha256": evaluation.canonical_json_sha256(
            evaluation.exact.paper_effective_contract()
        ),
        "initialization": evaluation.EXACT_INITIALIZATION,
    }
    if not omit_parent_marker:
        request["pretrained_ensemble"] = {
            "path": "/forbidden/xau/manifest.json",
            "sha256": "0" * 64,
        }
    outer = {
        "request": request,
        "initialization": evaluation.EXACT_INITIALIZATION,
        "pretrained_ensemble": None,
        "maximum_events_smoke_limit": None,
    }
    monkeypatch.setattr(
        evaluation.exact, "verify_ensemble_training", lambda *args, **kwargs: outer,
    )

    with pytest.raises(ValueError, match="parentless"):
        evaluation.verify_exact_training_chain(manifest)


def test_evaluate_checks_access_receipt_before_any_may_child(tmp_path, monkeypatch) -> None:
    events: list[str] = []

    def reject_receipt(path):
        events.append("receipt")
        raise PermissionError("receipt rejected")

    def unexpected_child(*args, **kwargs):
        events.append("child")
        raise AssertionError("a May child was opened before authorization")

    monkeypatch.setattr(evaluation, "verify_may_access_receipt", reject_receipt)
    monkeypatch.setattr(evaluation, "load_immutable_v6_reports", unexpected_child)
    with pytest.raises(PermissionError, match="receipt rejected"):
        evaluation.evaluate_may(tmp_path / "bad.json", tmp_path / "output")
    assert events == ["receipt"]


def test_existing_output_is_rejected_before_access_receipt_read(tmp_path, monkeypatch) -> None:
    output = tmp_path / "already_exists"
    output.mkdir()
    monkeypatch.setattr(
        evaluation,
        "verify_may_access_receipt",
        lambda path: (_ for _ in ()).throw(AssertionError("receipt should not be opened")),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluation.evaluate_may(tmp_path / "receipt.json", output)
