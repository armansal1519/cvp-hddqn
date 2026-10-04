"""Registered deterministic Pilot 10 policies.

The signal rules deliberately read the *unscaled* causal ``SessionFeatures``
rows at the current completed-bar decision index.  They do not consume a
publisher message, a learned representation, execution outcomes, or any
mutable policy state.  Consequently the same input and legal mask always
produce the same target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kx_simple_hmadrl.volume_v6_contract import (
    ACTIONS,
    BOTTOM_BASE_FIELDS,
    CONTEXT_FIELDS,
)
from kx_simple_hmadrl.volume_v6_data import SessionFeatures

from .contract import DETERMINISTIC_ARMS, BaselineArm


SHORT, FLAT, LONG = ACTIONS
RULE_ARMS = DETERMINISTIC_ARMS
MOMENTUM_60M_INDEX = BOTTOM_BASE_FIELDS.index("momentum_60m")
VWAP_DISTANCE_INDEX = CONTEXT_FIELDS.index("distance_session_vwap_atr")


def _optional_finite(value: object) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != ():
        return None
    result = float(array)
    return result if np.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class RuleSignal:
    """The two independent, unscaled values used by the registered rules."""

    momentum_60m: float | None = None
    distance_session_vwap_atr: float | None = None

    @classmethod
    def from_session(cls, session: SessionFeatures, index: int) -> "RuleSignal":
        """Extract only causal unscaled values at one completed-bar decision.

        ``SessionFeatures.bottom_base`` and ``bottom_context`` precede robust
        scaling.  An invalid row or unavailable context field is represented
        as missing and therefore produces ``FLAT`` downstream.
        """

        position = int(index)
        valid = np.asarray(session.bottom_valid, dtype=np.bool_)
        if position < 0 or position >= len(valid):
            raise IndexError(position)
        if not bool(valid[position]):
            return cls()

        base = np.asarray(session.bottom_base)
        context = np.asarray(session.bottom_context)
        context_mask = np.asarray(session.bottom_context_mask, dtype=np.bool_)
        if base.ndim != 2 or base.shape[0] != len(valid):
            raise ValueError("unscaled bottom-base rows do not align with decisions")
        if context.ndim != 2 or context.shape[0] != len(valid):
            raise ValueError("unscaled bottom-context rows do not align with decisions")
        if context_mask.shape != context.shape:
            raise ValueError("bottom-context availability mask does not align")
        if base.shape[1] <= MOMENTUM_60M_INDEX:
            raise ValueError("unscaled bottom-base row lacks momentum_60m")
        if context.shape[1] <= VWAP_DISTANCE_INDEX:
            raise ValueError("unscaled bottom-context row lacks VWAP distance")

        momentum = _optional_finite(base[position, MOMENTUM_60M_INDEX])
        vwap_distance = (
            _optional_finite(context[position, VWAP_DISTANCE_INDEX])
            if bool(context_mask[position, VWAP_DISTANCE_INDEX])
            else None
        )
        return cls(momentum, vwap_distance)

    @classmethod
    def coerce(cls, value: "RuleSignal | Mapping[str, object] | None") -> "RuleSignal":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("rule observation must be RuleSignal, a mapping, or None")
        return cls(
            _optional_finite(value.get("momentum_60m")),
            _optional_finite(value.get("distance_session_vwap_atr")),
        )


def _legal_mask(value: object) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    if mask.shape != (len(ACTIONS),):
        raise ValueError("legal-action mask must have SHORT/FLAT/LONG width")
    if not bool(mask.any()):
        raise ValueError("legal-action mask cannot be empty")
    return mask.copy()


def _legalize(desired: int, legal: object) -> int:
    mask = _legal_mask(legal)
    desired_index = ACTIONS.index(int(desired))
    if bool(mask[desired_index]):
        return int(desired)
    # Under the frozen inventory contract a prohibited reversal must first
    # flatten at an earlier completed-bar decision.
    flat_index = ACTIONS.index(FLAT)
    if not bool(mask[flat_index]):
        raise ValueError("registered inventory mask must always permit FLAT")
    return int(FLAT)


@dataclass(frozen=True, slots=True)
class DeterministicRulePolicy:
    """Stateless policy exposing the common replay ``q_values``/``act`` API."""

    arm: str

    def __post_init__(self) -> None:
        if self.arm not in RULE_ARMS:
            raise ValueError(f"unknown deterministic baseline arm {self.arm!r}")

    def desired_action(
        self, observation: RuleSignal | Mapping[str, object] | None = None,
    ) -> int:
        signal = RuleSignal.coerce(observation)
        if self.arm == "always_flat":
            return int(FLAT)
        if self.arm == "session_long":
            return int(LONG)
        if self.arm == "session_short":
            return int(SHORT)
        if self.arm == "tsmom_60m":
            value = signal.momentum_60m
            if value is None or value == 0.0:
                return int(FLAT)
            return int(LONG if value > 0.0 else SHORT)

        value = signal.distance_session_vwap_atr
        if value is None:
            return int(FLAT)
        # Positive distance means price is above VWAP, hence the contrarian
        # target is SHORT.  The registered one-ATR boundary is inclusive.
        if value >= 1.0:
            return int(SHORT)
        if value <= -1.0:
            return int(LONG)
        return int(FLAT)

    def q_values(
        self, observation: RuleSignal | Mapping[str, object] | None = None,
    ) -> np.ndarray:
        """Return deterministic preference scores in SHORT/FLAT/LONG order."""

        desired = self.desired_action(observation)
        values = np.zeros(len(ACTIONS), dtype=np.float64)
        values[ACTIONS.index(desired)] = 1.0
        return values

    def act(
        self,
        observation: RuleSignal | Mapping[str, object] | None = None,
        *,
        legal: object,
    ) -> int:
        return _legalize(self.desired_action(observation), legal)

    def act_session(self, session: SessionFeatures, index: int, *, legal: object) -> int:
        return self.act(RuleSignal.from_session(session, index), legal=legal)


def rule_policy(arm: str) -> DeterministicRulePolicy:
    """Construct one registered policy without retaining cross-call state."""

    return DeterministicRulePolicy(str(arm))


def registered_rule_policies() -> tuple[DeterministicRulePolicy, ...]:
    return tuple(rule_policy(arm) for arm in RULE_ARMS)


def registered_rule_settings() -> dict[str, dict[str, object]]:
    """Return the frozen, result-independent rule registry."""

    shared = {
        "source": "unscaled_causal_SessionFeatures_at_decision_index",
        "stateful": False,
        "legal_mask": "desired_if_legal_else_FLAT",
    }
    return {
        BaselineArm.TSMOM_60M.value: shared | {
            "field": "bottom_base.momentum_60m",
            "missing_or_nonfinite": "FLAT",
            "long": "value > 0",
            "short": "value < 0",
            "flat": "value == 0 or missing",
        },
        BaselineArm.VWAP_REVERSION_ATR1.value: shared | {
            "field": "bottom_context.distance_session_vwap_atr",
            "missing_or_nonfinite": "FLAT",
            "short": "value >= +1.0",
            "long": "value <= -1.0",
            "flat": "-1.0 < value < +1.0 or missing",
        },
        BaselineArm.ALWAYS_FLAT.value: shared | {
            "signal_required": False, "target": "FLAT",
        },
        BaselineArm.SESSION_LONG.value: shared | {
            "signal_required": False, "target": "LONG",
        },
        BaselineArm.SESSION_SHORT.value: shared | {
            "signal_required": False, "target": "SHORT",
        },
    }


__all__ = [
    "ACTIONS",
    "DeterministicRulePolicy",
    "FLAT",
    "LONG",
    "MOMENTUM_60M_INDEX",
    "RULE_ARMS",
    "RuleSignal",
    "SHORT",
    "VWAP_DISTANCE_INDEX",
    "registered_rule_policies",
    "registered_rule_settings",
    "rule_policy",
]
