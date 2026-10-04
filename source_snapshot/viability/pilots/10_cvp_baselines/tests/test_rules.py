from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kx_simple_hmadrl.volume_v6_contract import (
    BOTTOM_BASE_WIDTH,
    CONTEXT_WIDTH,
)
from kx_cvp_baselines.rules import (
    MOMENTUM_60M_INDEX,
    RULE_ARMS,
    RuleSignal,
    VWAP_DISTANCE_INDEX,
    registered_rule_policies,
    rule_policy,
)


ALL_LEGAL = np.asarray([True, True, True])


def _session(*, momentum: float = 0.0, distance: float = 0.0, available: bool = True):
    base = np.zeros((2, BOTTOM_BASE_WIDTH), dtype=np.float32)
    context = np.zeros((2, CONTEXT_WIDTH), dtype=np.float32)
    mask = np.ones_like(context, dtype=bool)
    base[1, MOMENTUM_60M_INDEX] = momentum
    context[1, VWAP_DISTANCE_INDEX] = distance
    mask[1, VWAP_DISTANCE_INDEX] = available
    return SimpleNamespace(
        bottom_valid=np.asarray([True, True]),
        bottom_base=base,
        bottom_context=context,
        bottom_context_mask=mask,
    )


def test_signal_extracts_unscaled_session_rows_and_respects_availability() -> None:
    signal = RuleSignal.from_session(_session(momentum=2.5, distance=-1.25), 1)
    assert signal == RuleSignal(momentum_60m=2.5, distance_session_vwap_atr=-1.25)

    missing = RuleSignal.from_session(
        _session(momentum=-3.0, distance=9.0, available=False), 1,
    )
    assert missing.momentum_60m == -3.0
    assert missing.distance_session_vwap_atr is None


def test_tsmom_60m_sign_and_missing_are_registered_targets() -> None:
    policy = rule_policy("tsmom_60m")
    assert policy.act(RuleSignal(momentum_60m=0.25), legal=ALL_LEGAL) == 1
    assert policy.act(RuleSignal(momentum_60m=-0.25), legal=ALL_LEGAL) == -1
    assert policy.act(RuleSignal(momentum_60m=0.0), legal=ALL_LEGAL) == 0
    assert policy.act(RuleSignal(momentum_60m=None), legal=ALL_LEGAL) == 0
    assert policy.act({"momentum_60m": np.nan}, legal=ALL_LEGAL) == 0


def test_vwap_reversion_uses_inclusive_one_atr_boundaries() -> None:
    policy = rule_policy("vwap_reversion_atr1")
    assert policy.act(RuleSignal(distance_session_vwap_atr=1.0), legal=ALL_LEGAL) == -1
    assert policy.act(RuleSignal(distance_session_vwap_atr=-1.0), legal=ALL_LEGAL) == 1
    assert policy.act(RuleSignal(distance_session_vwap_atr=0.999), legal=ALL_LEGAL) == 0
    assert policy.act(RuleSignal(distance_session_vwap_atr=None), legal=ALL_LEGAL) == 0


def test_sanity_policies_and_illegal_reversal_flatten_first() -> None:
    assert rule_policy("always_flat").act(None, legal=ALL_LEGAL) == 0
    assert rule_policy("session_long").act(None, legal=ALL_LEGAL) == 1
    assert rule_policy("session_short").act(None, legal=ALL_LEGAL) == -1

    # A short inventory makes LONG illegal; the rule must request FLAT rather
    # than aliasing the reversal or selecting a different arbitrary action.
    assert rule_policy("session_long").act(None, legal=[True, True, False]) == 0
    assert rule_policy("session_short").act(None, legal=[False, True, True]) == 0


def test_rules_are_stateless_and_common_replay_scores_have_action_order() -> None:
    policy = rule_policy("tsmom_60m")
    observation = RuleSignal(momentum_60m=1.0)
    assert policy.act(observation, legal=ALL_LEGAL) == 1
    assert policy.act(observation, legal=ALL_LEGAL) == 1
    assert policy.q_values(observation).tolist() == [0.0, 0.0, 1.0]
    assert tuple(item.arm for item in registered_rule_policies()) == RULE_ARMS


def test_rule_masks_and_names_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown deterministic"):
        rule_policy("unregistered")
    with pytest.raises(ValueError, match="width"):
        rule_policy("always_flat").act(None, legal=[True, True])
    with pytest.raises(ValueError, match="empty"):
        rule_policy("always_flat").act(None, legal=[False, False, False])
    with pytest.raises(ValueError, match="always permit FLAT"):
        rule_policy("session_long").act(None, legal=[True, False, False])

