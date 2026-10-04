from __future__ import annotations

import json

import pytest

from kx_cvp_baselines.__main__ import main


def test_describe_exposes_frozen_primary_and_no_paper(capsys) -> None:
    assert main(["describe"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["registered_primary_comparator"] == "flat_multiclock_ddqn"
    assert value["required_training_jobs"] == 15
    assert value["required_evaluation_cells"] == 64
    assert value["paper_integration"] is False
    assert value["august_access_authorized"] is False


def test_status_is_result_blind(capsys) -> None:
    assert main(["status"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["result_blind"] is True
    assert value["economic_values_returned"] is False
    assert sum(value["training_counts"].values()) == 15
    assert sum(value["evaluation_counts"].values()) == 64


def test_cli_rejects_report_destination_below_paper() -> None:
    with pytest.raises(PermissionError, match="cannot be written below paper"):
        main(["build-report", "--report-root", "paper/forbidden"])


def test_cli_rejects_artifact_destination_below_paper() -> None:
    with pytest.raises(PermissionError, match="cannot be written below paper"):
        main(["status", "--output-root", "paper/forbidden"])


def test_cli_rejects_august_lexically_before_artifact_io() -> None:
    with pytest.raises(PermissionError, match="August"):
        main(["status", "--output-root", "viability/artifacts/2026-08"])
