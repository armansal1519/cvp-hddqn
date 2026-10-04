from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kx_cvp_ablations.artifacts import atomic_parquet
from kx_cvp_ablations.contract import (
    ABLATION_ARMS,
    PERIODS,
    REQUIRED_ENSEMBLE_EVALUATIONS,
    REQUIRED_TRAINING_JOBS,
    SCENARIOS,
    SEEDS,
    assert_registered_dates,
)
from kx_cvp_ablations.representation import RepresentationMask
from kx_cvp_ablations.protocol import (
    assert_frozen_contracts,
    implementation_binding,
    verify_implementation_binding,
)
from kx_simple_hmadrl.volume_v6_contract import (
    BASE_CONTEXT_FIELDS,
    BOTTOM_BASE_FIELDS,
    BOTTOM_RICH_FIELDS,
    CONTEXT_FIELDS,
    L2_ENGINEERED_FIELDS,
    L2_LEVEL_FIELDS,
    STATE_FIELDS,
    TOP_BASE_FIELDS,
    TOP_RICH_FIELDS,
    VOLUME_CONTEXT_FIELDS,
)
from kx_simple_hmadrl.volume_v6_data import RegimeObservation, TraderObservation


PILOT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads(
    (PILOT_ROOT / "contracts/CVP_ABLATION_FEATURE_ARM_REGISTRY.json").read_text(
        encoding="utf-8"
    )
)


def _regime_observation() -> RegimeObservation:
    return RegimeObservation(
        base=np.arange(8 * len(TOP_BASE_FIELDS), dtype=np.float32).reshape(
            8, len(TOP_BASE_FIELDS)
        ) + 1,
        rich=np.arange(8 * len(TOP_RICH_FIELDS), dtype=np.float32).reshape(
            8, len(TOP_RICH_FIELDS)
        ) + 101,
        sequence_mask=np.ones(8, dtype=bool),
        context=np.arange(len(CONTEXT_FIELDS), dtype=np.float32) + 201,
        context_mask=np.ones(len(CONTEXT_FIELDS), dtype=bool),
        decision_time_ns=1,
    )


def _trader_observation() -> TraderObservation:
    return TraderObservation(
        base=np.arange(24 * len(BOTTOM_BASE_FIELDS), dtype=np.float32).reshape(
            24, len(BOTTOM_BASE_FIELDS)
        ) + 1,
        rich=np.arange(24 * len(BOTTOM_RICH_FIELDS), dtype=np.float32).reshape(
            24, len(BOTTOM_RICH_FIELDS)
        ) + 1001,
        sequence_mask=np.ones(24, dtype=bool),
        context=np.arange(len(CONTEXT_FIELDS), dtype=np.float32) + 2001,
        context_mask=np.ones(len(CONTEXT_FIELDS), dtype=bool),
        levels=np.arange(20 * 10 * len(L2_LEVEL_FIELDS), dtype=np.float32).reshape(
            20, 10, len(L2_LEVEL_FIELDS)
        ) + 3001,
        l2_engineered=np.arange(
            20 * len(L2_ENGINEERED_FIELDS), dtype=np.float32
        ).reshape(20, len(L2_ENGINEERED_FIELDS)) + 4001,
        l2_mask=np.ones(20, dtype=bool),
        message=np.asarray([-0.5, 0.0, 0.5, 1.0, 0.25], dtype=np.float32),
        state=np.arange(len(STATE_FIELDS), dtype=np.float32) + 5001,
        decision_time_ns=1,
    )


def test_core_constants_match_frozen_feature_arm_registry() -> None:
    assert ABLATION_ARMS == tuple(REGISTRY["new_arms"])
    assert SEEDS == tuple(REGISTRY["seeds"])
    assert SCENARIOS == tuple(REGISTRY["scenarios"])
    assert PERIODS == tuple(REGISTRY["evaluation_periods"])
    assert REQUIRED_TRAINING_JOBS == REGISTRY["required_training_jobs"] == 15
    assert REQUIRED_ENSEMBLE_EVALUATIONS == (
        REGISTRY["required_ensemble_evaluations"]
    ) == 24


def test_core_protocol_bridge_verifies_freeze_receipt() -> None:
    contracts = assert_frozen_contracts()
    assert contracts.protocol_canonical_sha256 == (
        "2d86687e3c786ca2c30dc39adeb3b28a95ecd684164a097ba8cc4b9f33ad9796"
    )


def test_runtime_implementation_binding_hashes_every_source_and_validator() -> None:
    binding = implementation_binding()
    assert binding["file_count"] == len(binding["files"])
    paths = {record["path"] for record in binding["files"]}
    assert "viability/pilots/09_cvp_ablations/contracts/validate_ablation_protocol.py" in paths
    assert "viability/pilots/09_cvp_ablations/src/kx_cvp_ablations/representation.py" in paths
    verify_implementation_binding(binding)
    altered = dict(binding)
    altered["aggregate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="implementation binding"):
        verify_implementation_binding(altered)


def test_registry_source_field_order_matches_frozen_v6_contract() -> None:
    source = REGISTRY["source_feature_order"]
    assert source["top_base"] == list(TOP_BASE_FIELDS)
    assert source["top_rich"] == list(TOP_RICH_FIELDS)
    assert source["base_context"] == list(BASE_CONTEXT_FIELDS)
    assert source["volume_context"] == list(VOLUME_CONTEXT_FIELDS)
    assert source["bottom_base"] == list(BOTTOM_BASE_FIELDS)
    assert source["bottom_rich"] == list(BOTTOM_RICH_FIELDS)
    assert source["l2_level"] == list(L2_LEVEL_FIELDS)
    assert source["l2_engineered"] == list(L2_ENGINEERED_FIELDS)


def test_flow_profile_publisher_is_bitwise_full_parent_path() -> None:
    observation = _regime_observation()
    mask = RepresentationMask.for_arm("flow_profile_only")
    actual = mask.publisher_arrays(observation)
    expected = observation.arrays("rich_volume")
    for key in expected:
        assert np.array_equal(actual[key], expected[key])


def test_depth_ofi_publisher_is_bitwise_restricted_parent_path() -> None:
    observation = _regime_observation()
    mask = RepresentationMask.for_arm("depth_ofi_only")
    actual = mask.publisher_arrays(observation)
    expected = observation.arrays("base_ohlcv")
    for key in expected:
        assert np.array_equal(actual[key], expected[key])


def test_no_publisher_cannot_construct_a_publisher_observation() -> None:
    with pytest.raises(ValueError, match="no-publisher"):
        RepresentationMask.for_arm("no_publisher").publisher_arrays(
            _regime_observation()
        )


def test_flow_profile_only_preserves_flow_and_suppresses_depth_ofi() -> None:
    observation = _trader_observation()
    before = observation.arrays("rich_volume")
    result = RepresentationMask.for_arm("flow_profile_only").trader_arrays(
        observation
    )
    assert np.array_equal(result["base"], before["base"])
    assert np.array_equal(result["rich"], before["rich"])
    assert np.array_equal(result["context"], before["context"])
    assert np.array_equal(result["context_mask"], before["context_mask"])
    assert np.array_equal(result["message"], before["message"])
    assert np.array_equal(result["l2_mask"], before["l2_mask"])
    for name in ("bid_distance_from_mid_ticks", "ask_distance_from_mid_ticks"):
        index = L2_LEVEL_FIELDS.index(name)
        assert np.array_equal(result["levels"][:, :, index], before["levels"][:, :, index])
    for name in ("log1p_bid_size", "log1p_ask_size", "asinh_mlofi"):
        assert not np.any(result["levels"][:, :, L2_LEVEL_FIELDS.index(name)])
    spread = L2_ENGINEERED_FIELDS.index("spread_ticks")
    assert np.array_equal(
        result["l2_engineered"][:, spread], before["l2_engineered"][:, spread]
    )
    for index, name in enumerate(L2_ENGINEERED_FIELDS):
        if name != "spread_ticks":
            assert not np.any(result["l2_engineered"][:, index])


def test_depth_ofi_only_retains_ordinary_volume_and_full_l2() -> None:
    observation = _trader_observation()
    before = observation.arrays("rich_volume")
    result = RepresentationMask.for_arm("depth_ofi_only").trader_arrays(observation)
    ordinary = BOTTOM_RICH_FIELDS.index("log1p_executed_volume")
    assert np.array_equal(result["rich"][:, ordinary], before["rich"][:, ordinary])
    assert not np.any(np.delete(result["rich"], ordinary, axis=1))
    split = len(BASE_CONTEXT_FIELDS)
    assert np.array_equal(result["context"][:split], before["context"][:split])
    assert not np.any(result["context"][split:])
    assert np.array_equal(result["context_mask"][:split], before["context_mask"][:split])
    assert not np.any(result["context_mask"][split:])
    assert np.array_equal(result["levels"], before["levels"])
    assert np.array_equal(result["l2_engineered"], before["l2_engineered"])
    assert np.array_equal(result["l2_mask"], before["l2_mask"])


def test_no_publisher_is_full_trader_state_with_exact_zero_float32_message() -> None:
    observation = _trader_observation()
    before = observation.arrays("rich_volume")
    result = RepresentationMask.for_arm("no_publisher").trader_arrays(observation)
    for key in before:
        if key == "message":
            assert result[key].dtype == np.float32
            assert result[key].shape == (5,)
            assert np.array_equal(result[key], np.zeros(5, dtype=np.float32))
        else:
            assert np.array_equal(result[key], before[key])


@pytest.mark.parametrize("arm", ABLATION_ARMS)
def test_representation_mask_never_mutates_source_observation(arm: str) -> None:
    observation = _trader_observation()
    before = observation.arrays("rich_volume")
    RepresentationMask.for_arm(arm).trader_arrays(observation)
    after = observation.arrays("rich_volume")
    for key in before:
        assert np.array_equal(before[key], after[key])


def test_mask_audit_detects_nonzero_suppressed_channel() -> None:
    mask = RepresentationMask.for_arm("flow_profile_only")
    arrays = mask.trader_arrays(_trader_observation())
    arrays["levels"][0, 0, L2_LEVEL_FIELDS.index("log1p_bid_size")] = 1.0
    with pytest.raises(AssertionError, match="depth/OFI"):
        mask.audit(arrays)


@pytest.mark.parametrize(
    ("period", "days"),
    [
        ("may_2026", ["2026-05-01", "2026-05-31"]),
        ("june_july_2026", ["2026-06-01", "2026-07-31"]),
    ],
)
def test_core_date_guard_returns_contaminated_classification(
    period: str, days: list[str]
) -> None:
    assert assert_registered_dates(days, period) == (
        "CONTAMINATED_HISTORICAL_HOLDOUT"
    )


@pytest.mark.parametrize("day", ["2026-08-01", "2026-08-15", "2026-08-31"])
def test_core_date_guard_rejects_every_august_date(day: str) -> None:
    with pytest.raises(PermissionError):
        assert_registered_dates([day], "june_july_2026")


def test_core_date_guard_rejects_unsorted_duplicate_or_empty_requests() -> None:
    with pytest.raises(ValueError):
        assert_registered_dates([], "may_2026")
    with pytest.raises(ValueError):
        assert_registered_dates(["2026-05-02", "2026-05-01"], "may_2026")
    with pytest.raises(ValueError):
        assert_registered_dates(["2026-05-01", "2026-05-01"], "may_2026")


def test_zero_fill_trace_is_a_valid_economic_outcome(tmp_path: Path) -> None:
    schema = pa.schema([("fee_cents", pa.int64())])
    path = atomic_parquet(tmp_path / "fills.parquet", [], schema=schema)
    assert pq.ParquetFile(path).metadata.num_rows == 0
    assert pq.read_schema(path) == schema
