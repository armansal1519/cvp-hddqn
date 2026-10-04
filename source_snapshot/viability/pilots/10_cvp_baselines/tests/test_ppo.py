from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from kx_cvp_baselines.contract import FLAT_ENV_INTERACTIONS_PER_SEED, PROTOCOL_ID
from kx_cvp_baselines.ppo import (
    DEFAULT_PPO_CONFIG,
    PPOAction,
    PPO_OBSERVATION_KEYS,
    PPO_OPTIMIZATION_CALLS_PER_SEED,
    PPO_OPTIMIZER_UPDATES_PER_SEED,
    PPO_PARAMETER_COUNT,
    PPO_TRANSITIONS_PER_PASS,
    RecurrentPPOTrader,
    compute_gae,
    create_ppo_trainer,
    deterministic_masked_argmax,
    initialize_ppo_trader,
    load_ppo_checkpoint,
    masked_logits,
    ppo_loss,
    registered_ppo_settings,
    restore_ppo_trainer,
    save_ppo_checkpoint,
    save_ppo_model,
)
from kx_simple_hmadrl.volume_v6 import TraderQNetwork, parameter_count
from kx_simple_hmadrl.volume_v6_contract import (
    BOTTOM_BASE_WIDTH,
    BOTTOM_HISTORY,
    BOTTOM_RICH_WIDTH,
    CONTEXT_WIDTH,
    L2_ENGINEERED_WIDTH,
    L2_HISTORY,
    L2_LEVEL_WIDTH,
    L2_LEVELS,
    MESSAGE_WIDTH,
    STATE_WIDTH,
)
from kx_simple_hmadrl.volume_v6_training import seed_everything


def _observation(value: float = 0.0) -> dict[str, np.ndarray]:
    result = {
        "base": np.full(
            (BOTTOM_HISTORY, BOTTOM_BASE_WIDTH), value, dtype=np.float32,
        ),
        "rich": np.full(
            (BOTTOM_HISTORY, BOTTOM_RICH_WIDTH), value, dtype=np.float32,
        ),
        "sequence_mask": np.ones(BOTTOM_HISTORY, dtype=np.bool_),
        "context": np.full(CONTEXT_WIDTH, value, dtype=np.float32),
        "context_mask": np.ones(CONTEXT_WIDTH, dtype=np.bool_),
        "levels": np.full(
            (L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH), value, dtype=np.float32,
        ),
        "l2_engineered": np.full(
            (L2_HISTORY, L2_ENGINEERED_WIDTH), value, dtype=np.float32,
        ),
        "l2_mask": np.ones(L2_HISTORY, dtype=np.bool_),
        "state": np.full(STATE_WIDTH, value, dtype=np.float32),
    }
    assert set(result) == PPO_OBSERVATION_KEYS
    return result


def _tensor_observation(batch: int = 2) -> dict[str, torch.Tensor]:
    observation = _observation()
    return {
        name: torch.as_tensor(np.stack([value] * batch))
        for name, value in observation.items()
    }


def _xau_base_state() -> dict[str, torch.Tensor]:
    seed_everything(90_001)
    return {
        name: value.detach().clone()
        for name, value in TraderQNetwork().base.state_dict().items()
    }


def _state_clone(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def test_registered_configuration_and_capacity_are_exact() -> None:
    settings = registered_ppo_settings()
    assert DEFAULT_PPO_CONFIG.gamma == 1.0
    assert DEFAULT_PPO_CONFIG.gae_lambda == 0.95
    assert DEFAULT_PPO_CONFIG.clip_ratio == 0.20
    assert DEFAULT_PPO_CONFIG.learning_rate == 2.5e-4
    assert DEFAULT_PPO_CONFIG.entropy_coefficient == 0.01
    assert DEFAULT_PPO_CONFIG.value_coefficient == 0.5
    assert DEFAULT_PPO_CONFIG.gradient_clip_norm == 0.5
    assert DEFAULT_PPO_CONFIG.epochs == 4
    assert DEFAULT_PPO_CONFIG.rollout_size == 2_048
    assert DEFAULT_PPO_CONFIG.minibatch_size == 256
    assert DEFAULT_PPO_CONFIG.chronological_passes == 3
    assert settings["parameters"] == PPO_PARAMETER_COUNT == 21_144
    assert settings["environment_interactions_per_seed"] == (
        FLAT_ENV_INTERACTIONS_PER_SEED
    )
    assert PPO_TRANSITIONS_PER_PASS == 19_242
    assert PPO_OPTIMIZATION_CALLS_PER_SEED == 30
    assert PPO_OPTIMIZER_UPDATES_PER_SEED == 912
    assert settings["message_mode"] == "internal_exact_float32_zero"
    assert settings["cross_decision_hidden_state"] is False
    assert parameter_count(RecurrentPPOTrader()) == 21_144


def test_actor_critic_is_a_seed_matched_head_substitution() -> None:
    seed = 1_103
    seed_everything(seed)
    ddqn = TraderQNetwork()
    seed_everything(seed)
    ppo = RecurrentPPOTrader()

    for branch in ("base", "rich", "l2", "message_state", "fusion"):
        left = getattr(ddqn, branch).state_dict()
        right = getattr(ppo, branch).state_dict()
        assert set(left) == set(right)
        assert all(torch.equal(left[name], right[name]) for name in left)
    for q_name, ppo_name in (("value", "critic"), ("advantage", "actor")):
        left = getattr(ddqn.head, q_name).state_dict()
        right = getattr(ppo.head, ppo_name).state_dict()
        assert set(left) == set(right)
        assert all(torch.equal(left[name], right[name]) for name in left)


def test_warm_start_changes_only_the_authorized_price_encoder() -> None:
    seed = 2_207
    seed_everything(seed)
    fresh = RecurrentPPOTrader()
    untouched = {
        branch: _state_clone(getattr(fresh, branch))
        for branch in ("rich", "l2", "message_state", "fusion", "head")
    }
    source = _xau_base_state()
    initialized = initialize_ppo_trader(
        seed=seed,
        authorized_xau_base_state=source,
    )
    assert all(
        torch.equal(initialized.base.state_dict()[name], value)
        for name, value in source.items()
    )
    for branch, expected in untouched.items():
        actual = getattr(initialized, branch).state_dict()
        assert all(torch.equal(actual[name], value) for name, value in expected.items())

    malformed = dict(source)
    malformed["not_a_base_weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="exactly the price base encoder"):
        initialize_ppo_trader(seed=seed, authorized_xau_base_state=malformed)


def test_network_injects_zero_message_and_rejects_every_extra_field() -> None:
    model = RecurrentPPOTrader()
    captured: list[torch.Tensor] = []

    def capture_message(
        _module: torch.nn.Module, arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(arguments[0].detach().clone())

    handle = model.message_state.register_forward_pre_hook(capture_message)
    try:
        logits, values = model(_tensor_observation(batch=2))
    finally:
        handle.remove()
    assert logits.shape == (2, 3)
    assert values.shape == (2,)
    assert len(captured) == 1
    assert captured[0].shape == (2, MESSAGE_WIDTH)
    assert captured[0].dtype == torch.float32
    assert torch.equal(captured[0], torch.zeros_like(captured[0]))

    for name in ("message", "top_base"):
        invalid = _tensor_observation(batch=1)
        invalid[name] = torch.zeros((1, MESSAGE_WIDTH), dtype=torch.float32)
        with pytest.raises(ValueError, match="extra"):
            model(invalid)


def test_legal_mask_controls_sampling_argmax_and_objective() -> None:
    logits = torch.tensor([[100.0, 2.0, 3.0], [1.0, 4.0, 9.0]])
    legal = torch.tensor([[False, True, True], [True, True, False]])
    selected = masked_logits(logits, legal)
    assert torch.isneginf(selected[0, 0])
    assert torch.isneginf(selected[1, 2])
    assert deterministic_masked_argmax(logits, legal).tolist() == [2, 1]
    with pytest.raises(ValueError, match="at least one legal"):
        masked_logits(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))

    objective_logits = torch.zeros(1, 3, requires_grad=True)
    objective_legal = torch.ones(1, 3, dtype=torch.bool)
    current_log_probability = -float(np.log(3.0))
    loss = ppo_loss(
        logits=objective_logits,
        values=torch.zeros(1, requires_grad=True),
        legal=objective_legal,
        action_indexes=torch.tensor([0], dtype=torch.long),
        old_log_probabilities=torch.tensor(
            [current_log_probability - float(np.log(2.0))],
        ),
        advantages=torch.ones(1),
        returns=torch.zeros(1),
    )
    assert float(loss.actor.detach()) == pytest.approx(-1.2, abs=1e-6)
    assert float(loss.clip_fraction.detach()) == pytest.approx(1.0)
    loss.total.backward()
    assert torch.isfinite(objective_logits.grad).all()

    with pytest.raises(ValueError, match="illegal behavior action"):
        ppo_loss(
            logits=torch.zeros(1, 3),
            values=torch.zeros(1),
            legal=torch.tensor([[False, True, True]]),
            action_indexes=torch.tensor([0]),
            old_log_probabilities=torch.zeros(1),
            advantages=torch.ones(1),
            returns=torch.zeros(1),
        )


def test_gae_stops_at_session_terminal_and_bootstraps_only_open_cutoff() -> None:
    advantages, returns = compute_gae(
        torch.tensor([1.0, 1.0, 10.0]),
        torch.zeros(3),
        torch.tensor([False, True, True]),
        last_value=999.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(advantages, torch.tensor([2.0, 1.0, 10.0]))
    assert torch.equal(advantages, returns)

    cutoff, _ = compute_gae(
        torch.tensor([1.0, 1.0]),
        torch.zeros(2),
        torch.tensor([False, False]),
        last_value=5.0,
        gamma=1.0,
        gae_lambda=0.95,
    )
    assert cutoff.tolist() == pytest.approx([6.7, 6.0], abs=1e-6)


def test_trainer_masks_actions_and_enforces_chronological_session_resets() -> None:
    trainer = create_ppo_trainer(
        seed=3_301,
        authorized_xau_base_state=_xau_base_state(),
        protocol_id=PROTOCOL_ID,
    )
    observation = _observation()
    trainer.begin_pass()
    action = trainer.act(
        observation,
        legal=np.array([False, True, False]),
        deterministic=False,
    )
    assert action.action == 0
    trainer.record(
        observation,
        action,
        100.0,
        observation,
        terminal=False,
        legal=np.array([False, True, False]),
        next_legal=np.array([True, True, True]),
        session_id="2025-01-02",
    )
    next_action = trainer.act(
        observation,
        legal=np.array([True, True, True]),
        deterministic=True,
    )
    with pytest.raises(ValueError, match="without a terminal reset"):
        trainer.record(
            observation,
            next_action,
            0.0,
            observation,
            terminal=True,
            legal=np.array([True, True, True]),
            next_legal=np.zeros(3, dtype=np.bool_),
            session_id="2025-01-03",
        )

    trainer.record(
        observation,
        next_action,
        -50.0,
        observation,
        terminal=True,
        legal=np.array([True, True, True]),
        next_legal=np.zeros(3, dtype=np.bool_),
        session_id="2025-01-02",
    )
    report = trainer.finish_pass()
    assert report is not None
    assert report.rollout_steps == 2
    assert report.epochs == 4
    assert report.minibatch_updates == 4
    assert trainer.passes_complete == 1
    assert trainer.transitions == 2
    assert trainer.raw_reward_total == 50.0
    assert len(trainer.rollout) == 0


def test_partial_rollout_checkpoint_resume_is_exact_and_hash_bound(
    tmp_path: Path,
) -> None:
    source = _xau_base_state()
    trainer = create_ppo_trainer(
        seed=4_409,
        authorized_xau_base_state=source,
        protocol_id=PROTOCOL_ID,
    )
    observation = _observation(0.25)
    trainer.begin_pass()
    decision = trainer.act(
        observation,
        legal=np.array([True, True, True]),
        deterministic=True,
    )
    trainer.record(
        observation,
        decision,
        125.0,
        observation,
        terminal=False,
        legal=np.array([True, True, True]),
        next_legal=np.array([True, True, True]),
        session_id="2025-02-03",
    )
    before = _state_clone(trainer.model)
    checkpoint = save_ppo_checkpoint(
        trainer,
        tmp_path / "ppo_resume.pt",
        {"passes_required": 3, "classification": "DEVELOPMENT"},
    )
    restored, metadata = restore_ppo_trainer(
        checkpoint,
        authorized_xau_base_state=source,
        protocol_id=PROTOCOL_ID,
    )
    assert metadata == {"passes_required": 3, "classification": "DEVELOPMENT"}
    assert restored.active_pass == 1
    assert restored.current_session_id == "2025-02-03"
    assert restored.last_transition_terminal is False
    assert len(restored.rollout) == 1
    assert restored.report() == trainer.report()
    assert all(
        torch.equal(restored.model.state_dict()[name], value)
        for name, value in before.items()
    )

    direct = create_ppo_trainer(
        seed=4_409,
        authorized_xau_base_state=source,
        protocol_id=PROTOCOL_ID,
    )
    loaded_metadata = load_ppo_checkpoint(direct, checkpoint)
    assert loaded_metadata == metadata
    assert direct.active_pass == restored.active_pass
    assert len(direct.rollout) == len(restored.rollout)

    corrupted = copy.deepcopy(trainer.checkpoint_state())
    corrupted["protocol_id"] = "ANOTHER_PROTOCOL"
    with pytest.raises(ValueError, match="identity differs"):
        direct.load_checkpoint_state(corrupted)


def test_incomplete_training_cannot_export_an_inference_model(tmp_path: Path) -> None:
    trainer = create_ppo_trainer(
        seed=5_501,
        authorized_xau_base_state=_xau_base_state(),
        protocol_id=PROTOCOL_ID,
    )
    with pytest.raises(ValueError, match="incomplete PPO model"):
        save_ppo_model(trainer, tmp_path / "ppo_model.pt")


def test_action_receipt_cannot_bypass_legality() -> None:
    trainer = create_ppo_trainer(
        seed=1_103,
        authorized_xau_base_state=_xau_base_state(),
        protocol_id=PROTOCOL_ID,
    )
    observation = _observation()
    trainer.begin_pass()
    fabricated = PPOAction(-1, 0, 0.0, 0.0)
    with pytest.raises(ValueError, match="illegal behavior action"):
        trainer.record(
            observation,
            fabricated,
            0.0,
            observation,
            terminal=True,
            legal=np.array([False, True, True]),
            next_legal=np.zeros(3, dtype=np.bool_),
            session_id="2025-03-03",
        )
