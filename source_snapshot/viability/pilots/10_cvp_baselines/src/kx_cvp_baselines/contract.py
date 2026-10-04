"""Public, result-independent contract for the CVP baseline suite."""

from __future__ import annotations

from dataclasses import asdict
from enum import StrEnum
from typing import Final

from kx_simple_hmadrl.environment import ExecutionEconomics
from kx_simple_hmadrl.volume_v6_contract import SEEDS as V6_SEEDS


PROTOCOL_ID: Final = "CVP_BASELINE_SUITE_V1"
CANONICAL_OUTPUT_ROOT: Final = (
    "viability/artifacts/development/cvp_baseline_suite_v1"
)


class BaselineArm(StrEnum):
    FLAT_MULTICLOCK_DDQN = "flat_multiclock_ddqn"
    RECURRENT_PPO = "recurrent_ppo"
    LIGHTGBM_FITTED_Q = "lightgbm_fitted_q"
    TSMOM_60M = "tsmom_60m"
    VWAP_REVERSION_ATR1 = "vwap_reversion_atr1"
    ALWAYS_FLAT = "always_flat"
    SESSION_LONG = "session_long"
    SESSION_SHORT = "session_short"


class EvaluationPeriod(StrEnum):
    MAY_2026 = "may_2026"
    JUNE_JULY_2026 = "june_july_2026"


class BenchmarkScenario(StrEnum):
    NOMINAL = "nominal"
    DOUBLE_COMMISSION = "double_commission"
    LATENCY_20MS = "latency_20ms"
    ADVERSE_TICK = "adverse_tick"


LEARNER_ARMS: Final = (
    BaselineArm.FLAT_MULTICLOCK_DDQN.value,
    BaselineArm.RECURRENT_PPO.value,
    BaselineArm.LIGHTGBM_FITTED_Q.value,
)
DETERMINISTIC_ARMS: Final = (
    BaselineArm.TSMOM_60M.value,
    BaselineArm.VWAP_REVERSION_ATR1.value,
    BaselineArm.ALWAYS_FLAT.value,
    BaselineArm.SESSION_LONG.value,
    BaselineArm.SESSION_SHORT.value,
)
BASELINE_ARMS: Final = LEARNER_ARMS + DETERMINISTIC_ARMS
PERIODS: Final = tuple(period.value for period in EvaluationPeriod)
SCENARIOS: Final = tuple(scenario.value for scenario in BenchmarkScenario)
SEEDS: Final = tuple(int(seed) for seed in V6_SEEDS)

# These frozen artifacts may be compared or reused read-only. They are not new
# Pilot 10 arms and therefore do not count against its registered trial budget.
REFERENCE_ARMS: Final = (
    "cvp_hddqn",
    "restricted_representation",
    "no_publisher_single_agent_ddqn",
    "basepaper_described_gc_from_scratch_v1",
)

DEVELOPMENT_PERIOD_ID: Final = "development"
DEVELOPMENT_BOUNDS: Final = ("2025-01-01", "2026-05-01")
PERIOD_BOUNDS: Final = {
    EvaluationPeriod.MAY_2026.value: ("2026-05-01", "2026-06-01"),
    EvaluationPeriod.JUNE_JULY_2026.value: ("2026-06-01", "2026-08-01"),
}
PERIOD_CLASSIFICATION: Final = {
    DEVELOPMENT_PERIOD_ID: "DEVELOPMENT",
    **{
        period: "CONTAMINATED_HISTORICAL_HOLDOUT"
        for period in PERIODS
    },
}
FORBIDDEN_START: Final = "2026-08-01"

DATASET_MANIFESTS: Final = {
    DEVELOPMENT_PERIOD_ID: {
        "path": (
            "viability/artifacts/development/"
            "v6_registered_train_dataset_terminal_eligible/manifest.json"
        ),
        "sha256": "cdf3e233789a5370d02e1189bcc83e2dc10be4c23e1e92b311f2d4642f862503",
        "purpose": "train",
    },
    EvaluationPeriod.MAY_2026.value: {
        "path": (
            "viability/artifacts/development/"
            "v6_registered_validation_dataset_eligible/manifest.json"
        ),
        "sha256": "f68bc69dd63524a59a65b58ee15a87fd5bd8dfb3dfe25d9a98b46a2d0772ef52",
        "purpose": "evaluate",
    },
    EvaluationPeriod.JUNE_JULY_2026.value: {
        "path": "paper/sealed_extension/dataset/manifest.json",
        "sha256": "c50bf2f5a04ecd8fe90c0cc92df7a3edaed879a1c68ac13e36b4588dcfdd018c",
        "purpose": "evaluate",
    },
}

# Exact, seed-invariant counts verified from all five frozen rich-parent
# training receipts. The flat DDQN uses the same number of 5-minute economic
# interactions and the combined publisher-plus-trader optimizer budget.
FLAT_ENV_INTERACTIONS_PER_SEED: Final = 57_726
FLAT_PRIMARY_UPDATES_PER_SEED: Final = 57_215
FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED: Final = 17_520
FLAT_OPTIMIZER_UPDATES_PER_SEED: Final = 74_735
FLAT_PARAMETER_COUNT: Final = 36_388
REGISTERED_PASSES: Final = 3
REPLAY_CAPACITY: Final = 120_000
REPLAY_WARMUP: Final = 512
SUPPLEMENTAL_UPDATES_BY_PASS: Final = (5_670, 5_925, 5_925)

REQUIRED_LEARNER_TRAINING_JOBS: Final = len(LEARNER_ARMS) * len(SEEDS)
REQUIRED_ENSEMBLE_EVALUATIONS: Final = (
    len(LEARNER_ARMS) * len(PERIODS) * len(SCENARIOS)
)
REQUIRED_DETERMINISTIC_EVALUATIONS: Final = (
    len(DETERMINISTIC_ARMS) * len(PERIODS) * len(SCENARIOS)
)


def validate_arm(value: str | BaselineArm) -> BaselineArm:
    try:
        return BaselineArm(str(value))
    except ValueError as error:
        raise ValueError(f"unknown baseline arm {value!r}") from error


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


def assert_registered_dates(
    days: list[str] | tuple[str, ...], period: str | EvaluationPeriod,
) -> str:
    """Validate already-enumerated historical dates against a frozen period."""

    selected = validate_period(period).value
    if not days or tuple(days) != tuple(sorted(set(str(day) for day in days))):
        raise ValueError("evaluation dates must be nonempty, unique, and sorted")
    start, end_exclusive = PERIOD_BOUNDS[selected]
    if any(not start <= str(day) < end_exclusive for day in days):
        raise PermissionError(f"requested session is outside registered {selected}")
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


if FLAT_PRIMARY_UPDATES_PER_SEED + FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED != (
    FLAT_OPTIMIZER_UPDATES_PER_SEED
):
    raise AssertionError("flat-DDQN optimizer budget does not reconcile")
if sum(SUPPLEMENTAL_UPDATES_BY_PASS) != FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED:
    raise AssertionError("flat-DDQN supplemental update schedule does not reconcile")


__all__ = [
    "BASELINE_ARMS",
    "CANONICAL_OUTPUT_ROOT",
    "DATASET_MANIFESTS",
    "DETERMINISTIC_ARMS",
    "DEVELOPMENT_BOUNDS",
    "DEVELOPMENT_PERIOD_ID",
    "FLAT_ENV_INTERACTIONS_PER_SEED",
    "FLAT_OPTIMIZER_UPDATES_PER_SEED",
    "FLAT_PARAMETER_COUNT",
    "FLAT_PRIMARY_UPDATES_PER_SEED",
    "FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED",
    "FORBIDDEN_START",
    "LEARNER_ARMS",
    "PERIODS",
    "PERIOD_BOUNDS",
    "PERIOD_CLASSIFICATION",
    "PROTOCOL_ID",
    "REFERENCE_ARMS",
    "REGISTERED_PASSES",
    "REPLAY_CAPACITY",
    "REPLAY_WARMUP",
    "REQUIRED_DETERMINISTIC_EVALUATIONS",
    "REQUIRED_ENSEMBLE_EVALUATIONS",
    "REQUIRED_LEARNER_TRAINING_JOBS",
    "SCENARIOS",
    "SEEDS",
    "SUPPLEMENTAL_UPDATES_BY_PASS",
    "BaselineArm",
    "BenchmarkScenario",
    "EvaluationPeriod",
    "assert_registered_dates",
    "economics_record",
    "scenario_economics",
    "validate_arm",
    "validate_period",
    "validate_scenario",
]
