from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable

import numpy as np


class ContractError(ValueError):
    pass


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


def _probabilities(values: Iterable[float], length: int) -> tuple[float, ...]:
    result = tuple(float(x) for x in values)
    if len(result) != length or any(not math.isfinite(x) or x < 0 for x in result):
        raise ContractError("policy probabilities must be finite, nonnegative, and correctly sized")
    if not math.isclose(sum(result), 1.0, abs_tol=1e-6):
        raise ContractError("policy probabilities must sum to one")
    return result


@dataclass(frozen=True)
class MacroMessage:
    bias: Direction
    probabilities: tuple[float, float, float]
    decision_time_ns: int
    expiry_time_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bias", Direction(self.bias))
        object.__setattr__(self, "probabilities", _probabilities(self.probabilities, 3))
        if self.decision_time_ns >= self.expiry_time_ns:
            raise ContractError("macro message must expire after its decision")

    def active_at(self, time_ns: int) -> bool:
        return self.decision_time_ns <= time_ns < self.expiry_time_ns


@dataclass(frozen=True)
class TargetMessage:
    target_position: int
    probabilities: tuple[float, float]
    decision_time_ns: int
    target_change: bool

    def __post_init__(self) -> None:
        if self.target_position not in {-1, 0, 1}:
            raise ContractError("target position exceeds the one-contract limit")
        object.__setattr__(self, "probabilities", _probabilities(self.probabilities, 2))


@dataclass(frozen=True)
class ExecutionCommand:
    target_delta: int
    inventory: int
    deadline_ns: int
    remaining_ns: int

    def __post_init__(self) -> None:
        if self.inventory not in {-1, 0, 1} or self.inventory + self.target_delta not in {-1, 0, 1}:
            raise ContractError("execution command violates one-contract inventory")
        if self.remaining_ns < 0:
            raise ContractError("remaining time cannot be negative")


def legal_trade_mask(macro_bias: Direction) -> np.ndarray:
    """Mask positions ordered SHORT, FLAT, LONG."""
    bias = Direction(macro_bias)
    return np.asarray([bias == Direction.SHORT, True, bias == Direction.LONG], dtype=bool)


def masked_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    legal = np.asarray(mask, dtype=bool)
    if values.shape != legal.shape or not legal.any():
        raise ContractError("action mask must align and retain at least one action")
    return np.where(legal, values, -np.inf)


def film(latent: np.ndarray, message: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Bounded FiLM: y=(1+0.5*tanh(gamma))*x+0.5*tanh(beta)."""
    x = np.asarray(latent, dtype=float)
    m = np.asarray(message, dtype=float)
    w = np.asarray(weight, dtype=float)
    b = np.asarray(bias, dtype=float)
    if w.shape != (2 * x.shape[-1], m.shape[-1]) or b.shape != (2 * x.shape[-1],):
        raise ContractError("FiLM parameter shapes do not match latent and message")
    gamma, beta = np.split(w @ m + b, 2)
    return (1.0 + 0.5 * np.tanh(gamma)) * x + 0.5 * np.tanh(beta)
