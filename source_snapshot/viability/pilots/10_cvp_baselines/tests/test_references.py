from __future__ import annotations

import pytest

from kx_cvp_baselines.references import load_reference_report


def test_reference_loader_reconciles_existing_nominal_reports() -> None:
    cvp = load_reference_report("cvp_hddqn", "june_july_2026", "nominal")
    no_publisher = load_reference_report(
        "no_publisher_single_agent_ddqn", "june_july_2026", "nominal",
    )
    shavandi = load_reference_report(
        "shavandi_khedmati_reconstruction", "may_2026", "nominal",
    )
    assert cvp.metrics["net_pnl_usd"] == 14_496.0
    assert no_publisher.metrics["net_pnl_usd"] == -956.0
    assert shavandi.metrics["net_pnl_usd"] == -1_608.0
    assert cvp.classification_for_baseline_suite == "CONTAMINATED_HISTORICAL_HOLDOUT"


def test_shavandi_extension_is_forbidden() -> None:
    with pytest.raises(PermissionError):
        load_reference_report(
            "shavandi_khedmati_reconstruction", "june_july_2026", "nominal",
        )
