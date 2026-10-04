from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from kx_cvp_baselines.contract import FLAT_PARAMETER_COUNT
from kx_cvp_baselines.flat_ddqn import (
    FLAT_OBSERVATION_KEYS,
    FlatMultiClockObservation,
    FlatMultiClockQNetwork,
    assert_training_report_budget,
    registered_trainer_settings,
)
from kx_simple_hmadrl.volume_v6 import parameter_count
from kx_simple_hmadrl.volume_v6_contract import (
    BOTTOM_BASE_WIDTH,
    BOTTOM_HISTORY,
    BOTTOM_RICH_WIDTH,
    CONTEXT_WIDTH,
    L2_ENGINEERED_WIDTH,
    L2_HISTORY,
    L2_LEVELS,
    L2_LEVEL_WIDTH,
    MESSAGE_WIDTH,
    STATE_WIDTH,
    TOP_BASE_WIDTH,
    TOP_HISTORY,
    TOP_RICH_WIDTH,
)
from kx_simple_hmadrl.volume_v6_data import RegimeObservation, TraderObservation


FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000


def _top(*, decision: int = 3 * FIVE_MINUTES_NS) -> RegimeObservation:
    return RegimeObservation(
        base=np.linspace(0.1, 1.0, TOP_HISTORY * TOP_BASE_WIDTH, dtype=np.float32).reshape(
            TOP_HISTORY, TOP_BASE_WIDTH
        ),
        rich=np.linspace(1.1, 2.0, TOP_HISTORY * TOP_RICH_WIDTH, dtype=np.float32).reshape(
            TOP_HISTORY, TOP_RICH_WIDTH
        ),
        sequence_mask=np.ones(TOP_HISTORY, dtype=np.bool_),
        context=np.linspace(0.1, 1.0, CONTEXT_WIDTH, dtype=np.float32),
        context_mask=np.ones(CONTEXT_WIDTH, dtype=np.bool_),
        decision_time_ns=decision,
    )


def _bottom(*, decision: int = 4 * FIVE_MINUTES_NS, message: float = 7.0) -> TraderObservation:
    return TraderObservation(
        base=np.linspace(
            0.1, 1.0, BOTTOM_HISTORY * BOTTOM_BASE_WIDTH, dtype=np.float32,
        ).reshape(BOTTOM_HISTORY, BOTTOM_BASE_WIDTH),
        rich=np.linspace(
            1.1, 2.0, BOTTOM_HISTORY * BOTTOM_RICH_WIDTH, dtype=np.float32,
        ).reshape(BOTTOM_HISTORY, BOTTOM_RICH_WIDTH),
        sequence_mask=np.ones(BOTTOM_HISTORY, dtype=np.bool_),
        context=np.linspace(0.2, 1.2, CONTEXT_WIDTH, dtype=np.float32),
        context_mask=np.ones(CONTEXT_WIDTH, dtype=np.bool_),
        levels=np.linspace(
            0.1,
            1.0,
            L2_HISTORY * L2_LEVELS * L2_LEVEL_WIDTH,
            dtype=np.float32,
        ).reshape(L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH),
        l2_engineered=np.linspace(
            0.1, 1.0, L2_HISTORY * L2_ENGINEERED_WIDTH, dtype=np.float32,
        ).reshape(L2_HISTORY, L2_ENGINEERED_WIDTH),
        l2_mask=np.ones(L2_HISTORY, dtype=np.bool_),
        message=np.full(MESSAGE_WIDTH, message, dtype=np.float32),
        state=np.linspace(0.1, 1.0, STATE_WIDTH, dtype=np.float32),
        decision_time_ns=decision,
    )


def _tensor_batch(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(
            value[None],
            dtype=torch.bool if value.dtype == np.bool_ else torch.float32,
        )
        for name, value in arrays.items()
    }


def test_message_is_absent_and_cannot_affect_flat_observation() -> None:
    first = FlatMultiClockObservation.from_v6(
        _top(), _bottom(message=-100.0), top_segment=9, bottom_segment=9,
    ).arrays()
    second = FlatMultiClockObservation.from_v6(
        _top(), _bottom(message=100.0), top_segment=9, bottom_segment=9,
    ).arrays()
    assert tuple(first) == FLAT_OBSERVATION_KEYS
    assert "message" not in first
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_unavailable_top_is_explicit_zero_false_and_preserves_decision() -> None:
    bottom = _bottom(decision=FIVE_MINUTES_NS)
    observation = FlatMultiClockObservation.before_first_top(
        bottom,
        first_top_decision_time_ns=3 * FIVE_MINUTES_NS,
        bottom_segment=4,
    )
    arrays = observation.arrays()
    assert observation.top_available is False
    assert observation.top_decision_time_ns == -1
    assert observation.decision_time_ns == FIVE_MINUTES_NS
    assert arrays["top_base"].dtype == np.float32
    assert arrays["top_rich"].dtype == np.float32
    assert not arrays["top_base"].any()
    assert not arrays["top_rich"].any()
    assert arrays["top_sequence_mask"].dtype == np.bool_
    assert not arrays["top_sequence_mask"].any()
    assert not arrays["top_context_mask"].any()
    with pytest.raises(ValueError, match="before the first top"):
        FlatMultiClockObservation.before_first_top(
            _bottom(decision=3 * FIVE_MINUTES_NS),
            first_top_decision_time_ns=3 * FIVE_MINUTES_NS,
            bottom_segment=4,
        )


def test_available_top_must_be_causal_fresh_and_same_segment() -> None:
    with pytest.raises(ValueError, match="available"):
        FlatMultiClockObservation.from_v6(
            _top(decision=5 * FIVE_MINUTES_NS),
            _bottom(decision=4 * FIVE_MINUTES_NS),
            top_segment=1,
            bottom_segment=1,
        )
    with pytest.raises(ValueError, match="stale"):
        FlatMultiClockObservation.from_v6(
            _top(decision=FIVE_MINUTES_NS),
            _bottom(decision=4 * FIVE_MINUTES_NS),
            top_segment=1,
            bottom_segment=1,
        )
    with pytest.raises(ValueError, match="segment"):
        FlatMultiClockObservation.from_v6(
            _top(), _bottom(), top_segment=1, bottom_segment=2,
        )


def test_network_is_exact_capacity_and_every_parameter_is_on_loss_path() -> None:
    torch.manual_seed(1103)
    model = FlatMultiClockQNetwork()
    assert parameter_count(model.top) == 15_244
    assert parameter_count(model.bottom) == 21_144
    assert parameter_count(model) == FLAT_PARAMETER_COUNT == 36_388
    batch = _tensor_batch(
        FlatMultiClockObservation.from_v6(
            _top(), _bottom(), top_segment=7, bottom_segment=7,
        ).arrays()
    )
    q_values = model(batch)
    assert q_values.shape == (1, 3)
    q_values.square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_network_accepts_explicit_unavailable_top_but_rejects_message_channel() -> None:
    model = FlatMultiClockQNetwork().eval()
    arrays = FlatMultiClockObservation.before_first_top(
        _bottom(decision=FIVE_MINUTES_NS),
        first_top_decision_time_ns=3 * FIVE_MINUTES_NS,
        bottom_segment=1,
    ).arrays()
    batch = _tensor_batch(arrays)
    assert torch.isfinite(model(batch)).all()
    batch["message"] = torch.zeros((1, MESSAGE_WIDTH), dtype=torch.float32)
    with pytest.raises(ValueError, match="no publisher/message"):
        model(batch)


def test_registered_flat_settings_and_report_audit_reconcile() -> None:
    settings = registered_trainer_settings(expected_decisions=29_293)
    assert settings["passes"] == 3
    assert settings["environment_interactions"] == 57_726
    assert settings["ordinary_optimizer_updates"] == 57_215
    assert settings["supplemental_replay_updates"] == 17_520
    assert settings["optimizer_updates_total"] == 74_735
    assert settings["parameter_count"] == 36_388
    assert settings["message_interface"] is None
    assert_training_report_budget(
        {"parameters": 36_388, "transitions": 57_726, "updates": 74_735}
    )
    with pytest.raises(ValueError, match="registered budget"):
        assert_training_report_budget(
            {"parameters": 36_388, "transitions": 57_725, "updates": 74_735}
        )
