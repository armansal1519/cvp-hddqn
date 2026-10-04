from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kx_simple_hmadrl.volume_v6_contract import ACTIONS

from kx_cvp_baselines import replay
from kx_cvp_baselines.artifacts import canonical_json_sha256, workspace_root
from kx_cvp_baselines.contract import (
    BASELINE_ARMS,
    CANONICAL_OUTPUT_ROOT,
    PERIODS,
    SCENARIOS,
    scenario_economics,
)


def _mock_training_prerequisite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import kx_cvp_baselines.training as training

    bound = {
        "path": "synthetic/complete_training_manifest.json",
        "path_scope": "project_root",
        "sha256": "f" * 64,
    }
    dummy = tmp_path / "complete_training_manifest.json"
    dummy.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        replay, "_suite_training_prerequisite",
        lambda root, supplied: (dummy, bound),
    )
    monkeypatch.setattr(
        replay, "_verify_suite_training_prerequisite", lambda record: dummy,
    )
    monkeypatch.setattr(replay, "_verify_binding_file", lambda record: dummy)
    monkeypatch.setattr(
        replay, "_verified_training_binding",
        lambda arm, manifest, reason: (
            (bound, "VERIFIED")
            if arm in replay.LEARNER_ARMS else (None, "NOT_APPLICABLE")
        ),
    )
    monkeypatch.setattr(
        training, "verify_complete_training_manifest",
        lambda path: {"status": "COMPLETE"},
    )


def _absent_routes() -> dict[str, object]:
    return {
        "route_deadline_ns": None,
        "route_side": None,
        "route_fill_time_ns": None,
        "route_fill_price_ticks": None,
        "route_fill_fee_cents": None,
        "route_displayed_size": None,
        "route_searched_snapshots": None,
        "terminal_route_deadline_ns": None,
        "terminal_route_side": None,
        "terminal_fill_time_ns": None,
        "terminal_fill_price_ticks": None,
        "terminal_fill_fee_cents": None,
        "terminal_displayed_size": None,
    }


def _flat_reward(ordinal: int, *, terminal: bool) -> dict[str, object]:
    decision = 100 + 100 * ordinal
    return {
        "session_date": "2026-05-01",
        "decision_ordinal": ordinal,
        "decision_time_ns": decision,
        "next_time_ns": decision + 100,
        "executed_action": 0,
        "legal_short": True,
        "legal_flat": True,
        "legal_long": True,
        "inventory_before": 0,
        "inventory_after": 0,
        "reward_cents": 0,
        "cumulative_executable_nlv_cents": 0,
        "command_created": False,
        "command_rejected": False,
        "terminal": terminal,
        **_absent_routes(),
    }


def _flat_rule_decision(reward: dict[str, object]) -> dict[str, object]:
    return {
        **{
            name: reward[name]
            for name in (
                "session_date", "decision_ordinal", "decision_time_ns",
                "next_time_ns", "executed_action", "inventory_before",
                "inventory_after", "reward_cents", "terminal",
            )
        },
        "seed": -1,
        "score_short": 0.0,
        "score_flat": 1.0,
        "score_long": 0.0,
        "legal_short": True,
        "legal_flat": True,
        "legal_long": True,
        "member_vote": 0,
        "desired_action": 0,
        "legality_fallback": False,
    }


def test_legal_masked_vote_masks_reversal_and_rejects_bad_inputs() -> None:
    assert replay.legal_masked_vote([9.0, 2.0, 3.0], [False, True, True]) == 1
    assert replay.legal_masked_vote([1.0, 7.0, 2.0], [True, True, True]) == 0
    with pytest.raises(ValueError, match="finite"):
        replay.legal_masked_vote([0.0, np.nan, 1.0], [True, True, True])
    with pytest.raises(ValueError, match="nonempty"):
        replay.legal_masked_vote([0.0, 1.0, 2.0], [False, False, False])


def test_canonical_replay_output_is_workspace_relative_from_any_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = (workspace_root() / CANONICAL_OUTPUT_ROOT).resolve()
    assert replay._resolve_output_root(None) == expected
    assert replay._resolve_output_root(CANONICAL_OUTPUT_ROOT) == expected
    # All relative replay roots are intentionally anchored to the thesis root.
    assert replay._resolve_output_root("relative/evidence") == (
        workspace_root() / "relative/evidence"
    ).resolve()


def test_august_replay_is_rejected_before_any_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("replay performed I/O before the lexical August guard")

    monkeypatch.setattr(replay, "authorize_manifest_request", forbidden)
    monkeypatch.setattr(replay, "_resolve_output_root", forbidden)
    monkeypatch.setattr(replay, "load_authorized_dataset", forbidden)
    with pytest.raises(PermissionError, match="August"):
        replay.replay_cell(
            arm="always_flat",
            period="august_2026",
            scenario="nominal",
            output_root="results/2026-08",
        )


def test_deterministic_trace_deep_verifier_reconciles_telescope_and_flatness() -> None:
    rewards = [_flat_reward(0, terminal=False), _flat_reward(1, terminal=True)]
    decisions = [_flat_rule_decision(row) for row in rewards]
    replay._verify_decisions(
        arm="always_flat",
        decisions=decisions,
        rewards=rewards,
        learned_ensemble=False,
        independent_seed=False,
    )
    replay._verify_reward_sequences(
        rewards=rewards,
        fills=[],
        economics=scenario_economics("nominal"),
        daily={"2026-05-01": 0},
        expected_rejections=0,
        expected_filled_sides=0,
        with_seed=False,
    )
    changed = [dict(row) for row in rewards]
    changed[1]["cumulative_executable_nlv_cents"] = 1
    with pytest.raises(ValueError, match="telescope"):
        replay._verify_reward_sequences(
            rewards=changed,
            fills=[],
            economics=scenario_economics("nominal"),
            daily={"2026-05-01": 0},
            expected_rejections=0,
            expected_filled_sides=0,
            with_seed=False,
        )


def test_five_seed_vote_trace_uses_flat_on_tied_plurality() -> None:
    reward = _flat_reward(0, terminal=True)
    votes = (-1, -1, 0, 1, 1)
    rows = []
    for seed, vote in zip(replay.SEEDS, votes, strict=True):
        scores = {
            -1: (3.0, 0.0, 0.0),
            0: (0.0, 3.0, 0.0),
            1: (0.0, 0.0, 3.0),
        }[vote]
        rows.append({
            **_flat_rule_decision(reward),
            "seed": seed,
            "score_short": scores[0],
            "score_flat": scores[1],
            "score_long": scores[2],
            "member_vote": vote,
            "votes_short": 2,
            "votes_flat": 1,
            "votes_long": 2,
        })
    replay._verify_decisions(
        arm="recurrent_ppo",
        decisions=rows,
        rewards=[reward],
        learned_ensemble=True,
        independent_seed=False,
    )
    rows[0]["votes_short"] = 1
    with pytest.raises(ValueError, match="vote-count"):
        replay._verify_decisions(
            arm="recurrent_ppo",
            decisions=rows,
            rewards=[reward],
            learned_ensemble=True,
            independent_seed=False,
        )


def test_route_verifier_reconciles_latency_deadline_fee_and_fill() -> None:
    economics = scenario_economics("nominal")
    decision = 1_000_000_000
    deadline = decision + economics.deadline_ms * 1_000_000
    row = {
        "decision_time_ns": decision,
        "next_time_ns": decision + 60_000_000_000,
        "route_deadline_ns": deadline,
        "route_side": 1,
        "route_fill_time_ns": decision + economics.route_latency_ms * 1_000_000,
        "route_fill_price_ticks": 2_000,
        "route_fill_fee_cents": economics.fee_cents_per_side,
        "route_displayed_size": 3,
        "command_rejected": False,
    }
    assert replay._validate_route(row, economics=economics, terminal=False) == (
        row["route_fill_time_ns"], 1, 2_000,
        economics.fee_cents_per_side, False,
    )
    row["route_fill_time_ns"] = decision + 1
    with pytest.raises(ValueError, match="latency"):
        replay._validate_route(row, economics=economics, terminal=False)


def test_technical_failure_is_hash_bound_and_contains_no_partial_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(replay, "_resolve_output_root", lambda value: tmp_path)
    _mock_training_prerequisite(monkeypatch, tmp_path)
    path = replay.write_technical_failure(
        arm="always_flat",
        period="may_2026",
        scenario="nominal",
        reason="HASH_MISMATCH",
        diagnostic="Synthetic missing source hash for unit test.",
        output_root=tmp_path,
        holdout_read=False,
    )
    value = replay.verify_technical_failure_report(
        path,
        expected_arm="always_flat",
        expected_period="may_2026",
        expected_scenario="nominal",
    )
    assert value["daily_pnl_cents"] is None
    assert value["metrics"] is None
    assert value["training_binding_status"] == "NOT_APPLICABLE"
    monkeypatch.setattr(
        replay, "load_authorized_dataset",
        lambda *args, **kwargs: pytest.fail(
            "idempotent retained failure reopened historical data"
        ),
    )
    assert replay.replay_cell(
        arm="always_flat",
        period="may_2026",
        scenario="nominal",
        output_root=tmp_path,
    ) == path
    with pytest.raises(FileExistsError):
        replay.write_technical_failure(
            arm="always_flat",
            period="may_2026",
            scenario="nominal",
            reason="HASH_MISMATCH",
            diagnostic="Cannot overwrite retained failure.",
            output_root=tmp_path,
        )


def test_complete_evaluation_manifest_requires_and_retains_exact_64_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(replay, "_resolve_output_root", lambda value: tmp_path)
    _mock_training_prerequisite(monkeypatch, tmp_path)
    for arm in BASELINE_ARMS:
        for period in PERIODS:
            for scenario in SCENARIOS:
                replay.write_technical_failure(
                    arm=arm,
                    period=period,
                    scenario=scenario,
                    reason="HASH_MISMATCH",
                    diagnostic="Synthetic retained failure; no historical data opened.",
                    output_root=tmp_path,
                    holdout_read=False,
                )
    path = replay.build_complete_evaluation_manifest(tmp_path)
    value = replay.verify_complete_evaluation_manifest(path)
    assert value["required_cells"] == 64
    assert value["complete_cells"] == 0
    assert value["technical_failure_cells"] == 64
    assert value["holdout_read_cells"] == 0
    assert value["training_manifest"] == value["suite_training_prerequisite"]
    assert value["learner_cells_with_verified_training"] == 24
    assert len(value["reports"]) == 8 * 2 * 4
    assert all(
        set(record) == {"arm", "period", "scenario", "status", "path", "sha256"}
        for record in value["reports"]
    )
    assert value["report_records_sha256"] == canonical_json_sha256(value["reports"])
    expected = [
        (arm, period, scenario)
        for arm in BASELINE_ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    ]
    assert [
        (record["arm"], record["period"], record["scenario"])
        for record in value["reports"]
    ] == expected


def test_source_audit_recomputes_policy_trace_and_detects_fabricated_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    rewards = [_flat_reward(0, terminal=False), _flat_reward(1, terminal=True)]
    decisions = [_flat_rule_decision(row) for row in rewards]
    primary = replay.RolloutRows.empty()
    primary.decisions = decisions
    primary.rewards = rewards
    primary.fills = []
    primary.daily_pnl_cents = {"2026-05-01": 0}

    dataset_binding = {
        "path": "synthetic/evaluation/manifest.json",
        "path_scope": "project_root",
        "sha256": "d" * 64,
        "purpose": "evaluate",
        "period_id": "may_2026",
        "classification": "CONTAMINATED_HISTORICAL_HOLDOUT",
        "dates": ["2026-05-01"],
    }

    class FakeDataset:
        def binding(self) -> dict[str, object]:
            return dict(dataset_binding)

    class FakeScaler:
        @staticmethod
        def canonical_sha256() -> str:
            return "s" * 64

    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    report_value = {
        "arm": "always_flat",
        "period": "may_2026",
        "scenario": "nominal",
        "dataset_manifest": dataset_binding,
        "training_manifest": None,
        "daily_pnl_cents": {"2026-05-01": 0},
        "metrics": replay.daily_metrics(
            {"2026-05-01": 0}, completed_round_trips=0,
        ),
        "completed_round_trips": 0,
        "rejected_commands": 0,
        "filled_sides": 0,
        "commissions_cents": 0,
        "decision_trace": {"id": "decisions"},
        "reward_trace": {"id": "rewards"},
        "fill_trace": {"id": "fills"},
        "rule_scaler_reference": {"scaler_canonical_sha256": "s" * 64},
    }
    rows = {
        "decisions": [dict(row) for row in decisions],
        "rewards": [dict(row) for row in rewards],
        "fills": [],
    }
    monkeypatch.setattr(replay, "verify_replay_report", lambda *args, **kwargs: report_value)
    monkeypatch.setattr(replay, "load_authorized_dataset", lambda *args, **kwargs: FakeDataset())
    monkeypatch.setattr(
        replay, "_evaluate_rule",
        lambda *args, **kwargs: (primary, False, FakeScaler()),
    )
    monkeypatch.setattr(
        replay, "_resolve_trace",
        lambda report, record: Path(str(record["id"])),
    )
    monkeypatch.setattr(replay, "_rows", lambda path: rows[path.name])

    audit = replay.audit_replay_report_against_source(report_path)
    assert audit["status"] == "PASS"
    assert audit["rule_signals_recomputed"] is True
    assert audit["raw_routes_and_fills_recomputed"] is True
    rows["decisions"][0]["score_long"] = 99.0
    with pytest.raises(ValueError, match="checkpoint/raw-source"):
        replay.audit_replay_report_against_source(report_path)


def test_replay_all_runs_exact_matrix_after_training_receipt_without_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import kx_cvp_baselines.training as training

    training_path = tmp_path / "complete_training_manifest.json"
    verified: list[Path] = []
    calls: list[dict[str, object]] = []
    destination = tmp_path / "evaluation" / "manifest.json"
    monkeypatch.setattr(replay, "_resolve_output_root", lambda value: tmp_path)
    monkeypatch.setattr(
        training,
        "verify_complete_training_manifest",
        lambda path: verified.append(Path(path)) or {"status": "COMPLETE"},
    )

    def fake_cell(**kwargs):
        calls.append(dict(kwargs))
        return tmp_path / "unused.json"

    monkeypatch.setattr(replay, "replay_cell", fake_cell)
    monkeypatch.setattr(
        replay, "build_complete_evaluation_manifest", lambda root: destination,
    )
    assert replay.replay_all(tmp_path, workers=1) == destination
    assert verified == [training_path]
    assert len(calls) == 64
    assert [
        (call["arm"], call["period"], call["scenario"])
        for call in calls
    ] == [
        (arm, period, scenario)
        for arm in BASELINE_ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    ]
    assert all(
        call["training_manifest"] == str(training_path)
        for call in calls if call["arm"] in replay.LEARNER_ARMS
    )
    assert all(
        call["training_manifest"] is None
        for call in calls if call["arm"] in replay.DETERMINISTIC_ARMS
    )


def test_evaluation_status_is_exact_and_does_not_return_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(replay, "_resolve_output_root", lambda value: tmp_path)
    cell = tmp_path / "evaluation" / "may_2026" / "nominal" / "always_flat"
    cell.mkdir(parents=True)
    (cell / "report.json").write_text(json.dumps({
        "artifact_type": replay.REPLAY_ARTIFACT,
        "status": "COMPLETE",
        "arm": "always_flat",
        "period": "may_2026",
        "scenario": "nominal",
        "daily_pnl_cents": {"2026-05-01": 987_654_321},
    }), encoding="utf-8")
    partial = (
        tmp_path / "evaluation" / "june_july_2026" / "adverse_tick"
        / ".session_short.part"
    )
    partial.mkdir(parents=True)
    status = replay.evaluation_status(tmp_path)
    assert status["required_cells"] == 64
    assert status["recorded_cells"] == 1
    assert status["counts"]["COMPLETE"] == 1
    assert status["counts"]["PARTIAL_RETAINED"] == 1
    assert status["counts"]["NOT_STARTED"] == 62
    assert status["historical_economic_values_returned"] is False
    assert status["market_data_opened"] is False
    assert "987654321" not in json.dumps(status)


def test_failure_writer_rejects_weak_performance_as_technical_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not predeclared"):
        replay.write_technical_failure(
            arm="session_short",
            period="june_july_2026",
            scenario="nominal",
            reason="NEGATIVE_PNL",
            diagnostic="Economic underperformance is not technical invalidity.",
            output_root=tmp_path,
        )


def test_incomplete_training_cannot_masquerade_as_an_evaluation_cell(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="suite-level pre-evaluation failure"):
        replay.write_technical_failure(
            arm="session_short",
            period="june_july_2026",
            scenario="nominal",
            reason="INCOMPLETE_TRAINING",
            diagnostic="Synthetic pre-evaluation condition.",
            output_root=tmp_path,
        )


def test_manifest_mutation_is_detected_even_when_json_remains_well_formed(
    tmp_path: Path,
) -> None:
    # A minimal registry cannot masquerade as a complete suite.
    path = tmp_path / "evaluation" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_type": replay.COMPLETE_EVALUATION_ARTIFACT,
        "reports": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        replay.verify_complete_evaluation_manifest(path)


def test_action_domain_used_by_test_matches_short_flat_long_semantics() -> None:
    assert tuple(ACTIONS) == (-1, 0, 1)
