from __future__ import annotations

import json
from pathlib import Path

import pytest

from kx_cvp_baselines.contract import (
    BASELINE_ARMS,
    DATASET_MANIFESTS,
    DETERMINISTIC_ARMS,
    FLAT_ENV_INTERACTIONS_PER_SEED,
    FLAT_OPTIMIZER_UPDATES_PER_SEED,
    FLAT_PARAMETER_COUNT,
    FLAT_PRIMARY_UPDATES_PER_SEED,
    FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED,
    LEARNER_ARMS,
    PERIOD_CLASSIFICATION,
    PROTOCOL_ID,
    SEEDS,
    SUPPLEMENTAL_UPDATES_BY_PASS,
    assert_registered_dates,
)
from kx_cvp_baselines.flat_ddqn import audit_registered_parent_budget
from kx_cvp_baselines.protocol import (
    assert_frozen_contracts,
    authorize_manifest_request,
    implementation_binding,
    verify_implementation_binding,
)


PILOT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PILOT_ROOT / "contracts"
REPO_ROOT = PILOT_ROOT.parents[2]


def _validator():
    import kx_cvp_baselines.protocol as protocol

    return protocol._validator()


def test_registered_arm_partitions_and_flat_budget_are_exact() -> None:
    assert PROTOCOL_ID == "CVP_BASELINE_SUITE_V1"
    assert LEARNER_ARMS == (
        "flat_multiclock_ddqn",
        "recurrent_ppo",
        "lightgbm_fitted_q",
    )
    assert DETERMINISTIC_ARMS == (
        "tsmom_60m",
        "vwap_reversion_atr1",
        "always_flat",
        "session_long",
        "session_short",
    )
    assert BASELINE_ARMS == LEARNER_ARMS + DETERMINISTIC_ARMS
    assert SEEDS == (1103, 2207, 3301, 4409, 5501)
    assert FLAT_PARAMETER_COUNT == 36_388
    assert FLAT_ENV_INTERACTIONS_PER_SEED == 57_726
    assert FLAT_PRIMARY_UPDATES_PER_SEED == 57_215
    assert FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED == 17_520
    assert FLAT_OPTIMIZER_UPDATES_PER_SEED == 74_735
    assert SUPPLEMENTAL_UPDATES_BY_PASS == (5_670, 5_925, 5_925)


def test_frozen_contracts_and_receipt_validate_against_repository() -> None:
    contracts = assert_frozen_contracts()
    assert contracts.protocol_canonical_sha256 == (
        "bc8c0c159e30d996fe542d58c55b4090f6cc70398a28c3773470924a571485b9"
    )
    assert contracts.protocol["scientific_status"] == (
        "DESCRIPTIVE_HISTORICAL_EVIDENCE_ONLY"
    )


def test_contract_validator_rejects_result_selection_and_fresh_labels() -> None:
    validator = _validator()
    protocol = json.loads(
        (CONTRACT_DIR / validator.PROTOCOL_FILE).read_text(encoding="utf-8")
    )
    ledger = json.loads(
        (CONTRACT_DIR / validator.ACCESS_LEDGER_FILE).read_text(encoding="utf-8")
    )
    registry = json.loads(
        (CONTRACT_DIR / validator.ARM_REGISTRY_FILE).read_text(encoding="utf-8")
    )
    protocol["holdout"]["periods"][0]["label"] = "FRESH_OOS"
    protocol["design"]["result_selection"] = "retain only baselines CVP beats"
    errors = validator.validate_contract_values(protocol, ledger, registry)
    joined = "\n".join(errors).lower()
    assert "contamination" in joined
    assert "retention" in joined


def test_manifest_authorization_is_exact_and_classified_before_io() -> None:
    for period_id, record in DATASET_MANIFESTS.items():
        authorization = authorize_manifest_request(
            record["path"], purpose=record["purpose"], period_id=period_id,
        )
        assert authorization.sha256 == record["sha256"]
        assert authorization.classification == PERIOD_CLASSIFICATION[period_id]
    with pytest.raises(PermissionError, match="differs"):
        authorize_manifest_request(
            "viability/artifacts/development/not_registered/manifest.json",
            purpose="train",
            period_id="development",
        )


@pytest.mark.parametrize(
    ("path", "period"),
    [
        ("paper/august_2026/dataset/manifest.json", "june_july_2026"),
        ("paper/2026-08/manifest.json", "june_july_2026"),
        ("paper/sealed_extension/dataset/manifest.json", "august_2026"),
    ],
)
def test_august_is_rejected_lexically_without_contract_or_path_io(
    monkeypatch: pytest.MonkeyPatch, path: str, period: str,
) -> None:
    import kx_cvp_baselines.protocol as protocol

    monkeypatch.setattr(
        protocol, "_validator", lambda: pytest.fail("validator loaded before rejection"),
    )
    monkeypatch.setattr(
        protocol, "workspace_root", lambda: pytest.fail("path resolved before rejection"),
    )
    with pytest.raises(PermissionError, match="August"):
        authorize_manifest_request(path, purpose="evaluate", period_id=period)


def test_historical_date_guard_returns_only_contaminated_classification() -> None:
    assert assert_registered_dates(["2026-05-01", "2026-05-31"], "may_2026") == (
        "CONTAMINATED_HISTORICAL_HOLDOUT"
    )
    assert assert_registered_dates(
        ["2026-06-01", "2026-07-31"], "june_july_2026",
    ) == "CONTAMINATED_HISTORICAL_HOLDOUT"
    with pytest.raises(PermissionError):
        assert_registered_dates(["2026-08-01"], "june_july_2026")


def test_dynamic_implementation_binding_contains_runtime_source_not_reports() -> None:
    binding = implementation_binding()
    paths = {record["path"] for record in binding["files"]}
    assert (
        "viability/pilots/10_cvp_baselines/contracts/validate_baseline_protocol.py"
        in paths
    )
    assert "viability/pilots/10_cvp_baselines/src/kx_cvp_baselines/flat_ddqn.py" in paths
    assert all("artifacts/development" not in path for path in paths)
    verify_implementation_binding(binding)
    changed = dict(binding)
    changed["aggregate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="implementation binding"):
        verify_implementation_binding(changed)


def test_parent_combined_budget_is_verified_for_every_registered_seed() -> None:
    audit = audit_registered_parent_budget()
    assert audit["all_registered_seeds_verified"] is True
    assert [member["seed"] for member in audit["members"]] == list(SEEDS)
    assert all(member["combined_updates"] == 74_735 for member in audit["members"])
    assert all(member["trader_interactions"] == 57_726 for member in audit["members"])
