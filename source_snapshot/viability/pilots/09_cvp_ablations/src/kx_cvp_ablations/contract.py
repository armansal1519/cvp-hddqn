"""Public, outcome-independent contract for the ablation extension."""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum
from typing import Final

from kx_simple_hmadrl.environment import ExecutionEconomics
from kx_simple_hmadrl.volume_v6_contract import SEEDS as V6_SEEDS


PROTOCOL_ID: Final = "CVP_ABLATION_EXTENSION_V1"
CANONICAL_OUTPUT_ROOT: Final = (
    "viability/artifacts/development/cvp_ablation_extension_v1"
)


class AblationArm(StrEnum):
    FLOW_PROFILE_ONLY = "flow_profile_only"
    DEPTH_OFI_ONLY = "depth_ofi_only"
    NO_PUBLISHER = "no_publisher"


class EvaluationPeriod(StrEnum):
    MAY_2026 = "may_2026"
    JUNE_JULY_2026 = "june_july_2026"


class BenchmarkScenario(StrEnum):
    NOMINAL = "nominal"
    DOUBLE_COMMISSION = "double_commission"
    LATENCY_20MS = "latency_20ms"
    ADVERSE_TICK = "adverse_tick"


ABLATION_ARMS: Final = tuple(arm.value for arm in AblationArm)
PERIODS: Final = tuple(period.value for period in EvaluationPeriod)
SCENARIOS: Final = tuple(scenario.value for scenario in BenchmarkScenario)
SEEDS: Final = tuple(int(seed) for seed in V6_SEEDS)
REQUIRED_TRAINING_JOBS: Final = len(ABLATION_ARMS) * len(SEEDS)
REQUIRED_ENSEMBLE_EVALUATIONS: Final = (
    len(ABLATION_ARMS) * len(PERIODS) * len(SCENARIOS)
)
REGISTERED_CONTRASTS: Final = (
    "cvp_hddqn_minus_restricted_representation",
    "cvp_hddqn_minus_flow_profile_only",
    "cvp_hddqn_minus_depth_ofi_only",
    "flow_profile_only_minus_restricted_representation",
    "depth_ofi_only_minus_restricted_representation",
    "cvp_hddqn_minus_no_publisher",
    "representation_two_by_two_interaction_secondary_diagnostic",
)
PARENT_REFERENCE_REPORTS: Final = {
    "may_2026": {
        "path": (
            "viability/artifacts/development/"
            "v6_registered_may_descriptive_evaluation/manifest.json"
        ),
        "sha256": "b9dc8928ae24e8e1c44130dee175819cf2456ccd2c09aa7786f76ac2ba137c71",
    },
    "june_july_2026": {
        "path": "paper/sealed_extension/result/manifest.json",
        "sha256": "fb9a8ccbe8d28d64b6dbe70968329d09c103a6c5d773b42eee856a280c2418c1",
    },
}

PERIOD_BOUNDS: Final = {
    EvaluationPeriod.MAY_2026.value: ("2026-05-01", "2026-06-01"),
    EvaluationPeriod.JUNE_JULY_2026.value: ("2026-06-01", "2026-08-01"),
}
PERIOD_CLASSIFICATION: Final = {
    period: "CONTAMINATED_HISTORICAL_HOLDOUT" for period in PERIODS
}

PUBLISHER_PARENT_ARM: Final = {
    AblationArm.FLOW_PROFILE_ONLY.value: "rich_volume",
    AblationArm.DEPTH_OFI_ONLY.value: "base_ohlcv",
    AblationArm.NO_PUBLISHER.value: None,
}


def validate_arm(value: str | AblationArm) -> AblationArm:
    try:
        return AblationArm(str(value))
    except ValueError as error:
        raise ValueError(f"unknown ablation arm {value!r}") from error


def validate_period(value: str | EvaluationPeriod) -> EvaluationPeriod:
    try:
        return EvaluationPeriod(str(value))
    except ValueError as error:
        raise ValueError(f"unknown evaluation period {value!r}") from error


def validate_scenario(value: str | BenchmarkScenario) -> BenchmarkScenario:
    try:
        return BenchmarkScenario(str(value))
    except ValueError as error:
        raise ValueError(f"unknown benchmark scenario {value!r}") from error


def assert_registered_dates(days: list[str] | tuple[str, ...], period: str) -> str:
    """Reject nonregistered dates before any dataset path is opened."""

    selected = validate_period(period).value
    if not days or tuple(days) != tuple(sorted(set(days))):
        raise ValueError("evaluation dates must be nonempty, unique, and sorted")
    start, end_exclusive = PERIOD_BOUNDS[selected]
    if any(not start <= str(day) < end_exclusive for day in days):
        raise PermissionError(f"requested session is outside registered {selected}")
    # August is outside both registered bounds and therefore cannot be listed or opened.
    return PERIOD_CLASSIFICATION[selected]


def scenario_economics(value: str | BenchmarkScenario) -> ExecutionEconomics:
    scenario = validate_scenario(value)
    if scenario is BenchmarkScenario.NOMINAL:
        return ExecutionEconomics.from_basepaper()
    if scenario is BenchmarkScenario.DOUBLE_COMMISSION:
        return ExecutionEconomics.from_basepaper(fee_multiplier=2)
    if scenario is BenchmarkScenario.LATENCY_20MS:
        return ExecutionEconomics.from_basepaper(route_latency_ms=20)
    return ExecutionEconomics.from_basepaper(adverse_ticks=1)


def economics_record(value: str | BenchmarkScenario) -> dict[str, object]:
    return asdict(scenario_economics(value))


__all__ = [
    "ABLATION_ARMS", "CANONICAL_OUTPUT_ROOT", "PERIODS", "PERIOD_BOUNDS",
    "PERIOD_CLASSIFICATION", "PROTOCOL_ID", "PUBLISHER_PARENT_ARM",
    "PARENT_REFERENCE_REPORTS", "REGISTERED_CONTRASTS",
    "REQUIRED_ENSEMBLE_EVALUATIONS", "REQUIRED_TRAINING_JOBS", "SCENARIOS",
    "SEEDS", "AblationArm", "BenchmarkScenario", "EvaluationPeriod",
    "assert_registered_dates", "economics_record", "scenario_economics",
    "validate_arm", "validate_period", "validate_scenario",
]
