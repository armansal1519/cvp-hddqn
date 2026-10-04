from __future__ import annotations

from pathlib import Path

import pytest

from kx_cvp_reduced_baselines import replay
from kx_cvp_reduced_baselines.contract import (
    LEARNER_ARMS,
    PERIODS,
    REFERENCE_ARMS,
    REQUIRED_EVALUATION_CELLS,
    REQUIRED_TRAINING_JOBS,
    SCENARIOS,
    SEEDS,
    scenario_economics,
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


def _flat_reward() -> dict[str, object]:
    return {
        "session_date": "2026-05-01",
        "decision_ordinal": 0,
        "decision_time_ns": 100,
        "next_time_ns": 200,
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
        "terminal": True,
        **_absent_routes(),
    }


def _decision(reward: dict[str, object], seed: int, vote: int) -> dict[str, object]:
    scores = {
        -1: (3.0, 0.0, 0.0),
        0: (0.0, 3.0, 0.0),
        1: (0.0, 0.0, 3.0),
    }[vote]
    return {
        **{
            name: reward[name]
            for name in (
                "session_date", "decision_ordinal", "decision_time_ns",
                "next_time_ns", "executed_action", "inventory_before",
                "inventory_after", "reward_cents", "terminal",
            )
        },
        "seed": seed,
        "score_short": scores[0],
        "score_flat": scores[1],
        "score_long": scores[2],
        "legal_short": True,
        "legal_flat": True,
        "legal_long": True,
        "member_vote": vote,
        "votes_short": 2,
        "votes_flat": 1,
        "votes_long": 2,
    }


def test_registered_matrix_is_exactly_two_by_two_by_four() -> None:
    assert LEARNER_ARMS == ("recurrent_ppo", "lightgbm_fitted_q")
    assert len(SEEDS) == 5
    assert REQUIRED_TRAINING_JOBS == 10
    assert REQUIRED_EVALUATION_CELLS == 16
    assert [
        (arm, period, scenario)
        for arm in LEARNER_ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    ] == [
        (arm, period, scenario)
        for arm in ("recurrent_ppo", "lightgbm_fitted_q")
        for period in ("may_2026", "june_july_2026")
        for scenario in (
            "nominal", "double_commission", "latency_20ms", "adverse_tick",
        )
    ]
    assert REFERENCE_ARMS == (
        "cvp_hddqn",
        "no_publisher_single_agent_ddqn",
        "shavandi_khedmati_reconstruction",
    )


def test_august_is_rejected_before_contract_or_output_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("I/O occurred before the August lexical guard")

    monkeypatch.setattr(replay, "assert_frozen_contracts", forbidden)
    monkeypatch.setattr(replay, "_output_root", forbidden)
    monkeypatch.setattr(replay, "load_authorized_dataset", forbidden)
    with pytest.raises(PermissionError, match="August"):
        replay.replay_cell(
            arm="recurrent_ppo",
            period="august_2026",
            scenario="nominal",
            output_root="viability/artifacts/2026-08",
        )


def test_result_blind_status_enumerates_only_sixteen_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(replay, "_output_root", lambda value: tmp_path)
    value = replay.evaluation_status(tmp_path)
    assert value["required_cells"] == 16
    assert value["recorded_cells"] == 0
    assert value["counts"] == {"NOT_STARTED": 16}
    assert value["result_blind"] is True
    assert value["historical_economic_values_returned"] is False
    assert len(value["records"]) == 16
    assert all("metrics" not in record for record in value["records"])


def test_inherited_vote_and_ledger_checks_cover_reduced_arms() -> None:
    reward = _flat_reward()
    votes = (-1, -1, 0, 1, 1)
    decisions = [
        _decision(reward, seed, vote)
        for seed, vote in zip(SEEDS, votes, strict=True)
    ]
    replay._verify_decisions(
        arm="recurrent_ppo",
        decisions=decisions,
        rewards=[reward],
        learned_ensemble=True,
        independent_seed=False,
    )
    replay._verify_reward_sequences(
        rewards=[reward],
        fills=[],
        economics=scenario_economics("nominal"),
        daily={"2026-05-01": 0},
        expected_rejections=0,
        expected_filled_sides=0,
        with_seed=False,
    )
    decisions[0]["votes_short"] = 1
    with pytest.raises(ValueError, match="vote-count"):
        replay._verify_decisions(
            arm="recurrent_ppo",
            decisions=decisions,
            rewards=[reward],
            learned_ensemble=True,
            independent_seed=False,
        )


def test_unregistered_failure_reason_cannot_be_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unregistered failure reached contract or artifact I/O")

    monkeypatch.setattr(replay, "assert_frozen_contracts", forbidden)
    monkeypatch.setattr(replay, "_output_root", forbidden)
    with pytest.raises(ValueError, match="predeclared"):
        replay.write_technical_failure(
            arm="recurrent_ppo",
            period="may_2026",
            scenario="nominal",
            reason="WEAK_PERFORMANCE",
            diagnostic="Synthetic test.",
            output_root=tmp_path,
        )


def test_worker_matrix_has_no_flat_or_deterministic_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    receipt = tmp_path / "adopted_training_receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    seen: list[tuple[str, str, str]] = []
    monkeypatch.setattr(replay, "assert_frozen_contracts", lambda: object())
    monkeypatch.setattr(replay, "_output_root", lambda value: tmp_path)
    monkeypatch.setattr(replay, "_receipt_path", lambda root, supplied: receipt)
    monkeypatch.setattr(
        replay,
        "_run_job",
        lambda job: seen.append((job[0], job[1], job[2])) or "synthetic",
    )
    sentinel = tmp_path / "evaluation" / "manifest.json"
    monkeypatch.setattr(
        replay, "build_complete_evaluation_manifest", lambda root: sentinel,
    )
    assert replay.replay_all(tmp_path, workers=1) == sentinel
    assert len(seen) == 16
    assert {arm for arm, _, _ in seen} == set(LEARNER_ARMS)
    assert "flat_multiclock_ddqn" not in {arm for arm, _, _ in seen}
