"""Registered, outcome-independent baselines for CVP-HDDQN."""

from .contract import (
    BASELINE_ARMS,
    DETERMINISTIC_ARMS,
    LEARNER_ARMS,
    PERIODS,
    PROTOCOL_ID,
    SCENARIOS,
    SEEDS,
    BaselineArm,
    BenchmarkScenario,
    EvaluationPeriod,
)


__all__ = [
    "BASELINE_ARMS",
    "DETERMINISTIC_ARMS",
    "LEARNER_ARMS",
    "PERIODS",
    "PROTOCOL_ID",
    "SCENARIOS",
    "SEEDS",
    "BaselineArm",
    "BenchmarkScenario",
    "EvaluationPeriod",
]
