"""Public, result-independent contract for the Pilot 11 reduced suite."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from kx_cvp_baselines.contract import scenario_economics as _v1_scenario_economics


PROTOCOL_ID: Final = "CVP_BASELINE_REDUCED_SUITE_V2"
CANONICAL_OUTPUT_ROOT: Final = (
    "viability/artifacts/development/cvp_baseline_reduced_suite_v2"
)
CANONICAL_REPORT_ROOT: Final = "viability/reports/cvp_baseline_reduced_suite_v2"


class BaselineArm(StrEnum):
    RECURRENT_PPO = "recurrent_ppo"
    LIGHTGBM_FITTED_Q = "lightgbm_fitted_q"


class EvaluationPeriod(StrEnum):
    MAY_2026 = "may_2026"
    JUNE_JULY_2026 = "june_july_2026"


class BenchmarkScenario(StrEnum):
    NOMINAL = "nominal"
    DOUBLE_COMMISSION = "double_commission"
    LATENCY_20MS = "latency_20ms"
    ADVERSE_TICK = "adverse_tick"


LEARNER_ARMS: Final = tuple(arm.value for arm in BaselineArm)
PERIODS: Final = tuple(period.value for period in EvaluationPeriod)
SCENARIOS: Final = tuple(scenario.value for scenario in BenchmarkScenario)
SEEDS: Final = (1103, 2207, 3301, 4409, 5501)
REFERENCE_ARMS: Final = (
    "cvp_hddqn",
    "no_publisher_single_agent_ddqn",
    "shavandi_khedmati_reconstruction",
)

DEVELOPMENT_PERIOD_ID: Final = "development"
DEVELOPMENT_BOUNDS: Final = ("2025-01-01", "2026-05-01")
PERIOD_BOUNDS: Final = {
    EvaluationPeriod.MAY_2026.value: ("2026-05-01", "2026-06-01"),
    EvaluationPeriod.JUNE_JULY_2026.value: ("2026-06-01", "2026-08-01"),
}
PERIOD_CLASSIFICATION: Final = {
    DEVELOPMENT_PERIOD_ID: "DEVELOPMENT",
    EvaluationPeriod.MAY_2026.value: "CONTAMINATED_HISTORICAL_HOLDOUT",
    EvaluationPeriod.JUNE_JULY_2026.value: "CONTAMINATED_HISTORICAL_HOLDOUT",
}
DATASET_MANIFESTS: Final = {
    DEVELOPMENT_PERIOD_ID: {
        "path": (
            "viability/artifacts/development/"
            "v6_registered_train_dataset_terminal_eligible/manifest.json"
        ),
        "sha256": (
            "cdf3e233789a5370d02e1189bcc83e2dc10be4c23e1e92b311f2d4642f862503"
        ),
        "purpose": "train",
    },
    EvaluationPeriod.MAY_2026.value: {
        "path": (
            "viability/artifacts/development/"
            "v6_registered_validation_dataset_eligible/manifest.json"
        ),
        "sha256": (
            "f68bc69dd63524a59a65b58ee15a87fd5bd8dfb3dfe25d9a98b46a2d0772ef52"
        ),
        "purpose": "evaluate",
    },
    EvaluationPeriod.JUNE_JULY_2026.value: {
        "path": "paper/sealed_extension/dataset/manifest.json",
        "sha256": (
            "c50bf2f5a04ecd8fe90c0cc92df7a3edaed879a1c68ac13e36b4588dcfdd018c"
        ),
        "purpose": "evaluate",
    },
}

REQUIRED_TRAINING_JOBS: Final = len(LEARNER_ARMS) * len(SEEDS)
REQUIRED_EVALUATION_CELLS: Final = (
    len(LEARNER_ARMS) * len(PERIODS) * len(SCENARIOS)
)
if REQUIRED_TRAINING_JOBS != 10 or REQUIRED_EVALUATION_CELLS != 16:
    raise AssertionError("reduced-suite budget changed")


def validate_arm(value: str | BaselineArm) -> BaselineArm:
    try:
        return BaselineArm(str(value))
    except ValueError as error:
        raise ValueError(f"unknown reduced baseline arm {value!r}") from error


def validate_period(value: str | EvaluationPeriod) -> EvaluationPeriod:
    try:
        return EvaluationPeriod(str(value))
    except ValueError as error:
        raise ValueError(f"unknown reduced evaluation period {value!r}") from error


def validate_scenario(value: str | BenchmarkScenario) -> BenchmarkScenario:
    try:
        return BenchmarkScenario(str(value))
    except ValueError as error:
        raise ValueError(f"unknown reduced benchmark scenario {value!r}") from error


def assert_registered_dates(
    days: list[str] | tuple[str, ...],
    period: str | EvaluationPeriod,
) -> str:
    """Validate an already enumerated set without broadening its date range."""

    selected = validate_period(period).value
    normalized = tuple(str(day) for day in days)
    if not normalized or normalized != tuple(sorted(set(normalized))):
        raise ValueError("evaluation dates must be nonempty, unique, and sorted")
    start, end_exclusive = PERIOD_BOUNDS[selected]
    if any(not start <= day < end_exclusive for day in normalized):
        raise PermissionError(f"requested session is outside registered {selected}")
    return PERIOD_CLASSIFICATION[selected]


def scenario_economics(value: str | BenchmarkScenario) -> object:
    """Delegate exact economics to the hash-bound Pilot 10 implementation."""

    return _v1_scenario_economics(validate_scenario(value).value)


__all__ = [
    "BaselineArm",
    "BenchmarkScenario",
    "CANONICAL_OUTPUT_ROOT",
    "CANONICAL_REPORT_ROOT",
    "DATASET_MANIFESTS",
    "DEVELOPMENT_BOUNDS",
    "DEVELOPMENT_PERIOD_ID",
    "EvaluationPeriod",
    "LEARNER_ARMS",
    "PERIODS",
    "PERIOD_BOUNDS",
    "PERIOD_CLASSIFICATION",
    "PROTOCOL_ID",
    "REFERENCE_ARMS",
    "REQUIRED_EVALUATION_CELLS",
    "REQUIRED_TRAINING_JOBS",
    "SCENARIOS",
    "SEEDS",
    "assert_registered_dates",
    "scenario_economics",
    "validate_arm",
    "validate_period",
    "validate_scenario",
]
