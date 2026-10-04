from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from kx_simple_hmadrl.volume_v6 import (
    BOTTOM_PARAMETER_LIMIT,
    TOP_PARAMETER_LIMIT,
    V6ModelBundle,
    parameter_count,
    regime_message,
    transfer_base_encoders,
)
from kx_simple_hmadrl.volume_v6_contract import ACTIONS, RegimeMessage
from kx_simple_hmadrl.volume_v6_data import RegimeObservation, TraderObservation
from kx_simple_hmadrl.volume_v6_training import (
    DEFAULT_DDQN_CONFIG,
    V6DDQNConfig,
    V6DDQNTrainer,
    file_sha256,
    load_model_bundle,
    load_trainer_checkpoint,
    reconcile_raw_rewards,
    save_model_bundle,
)


def _top(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "base": torch.randn(batch, 8, 8),
        "rich": torch.randn(batch, 8, 8),
        "sequence_mask": torch.ones(batch, 8, dtype=torch.bool),
        "context": torch.randn(batch, 22),
        "context_mask": torch.ones(batch, 22, dtype=torch.bool),
    }


def _bottom(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "base": torch.randn(batch, 24, 17),
        "rich": torch.randn(batch, 24, 21),
        "sequence_mask": torch.ones(batch, 24, dtype=torch.bool),
        "context": torch.randn(batch, 22),
        "context_mask": torch.ones(batch, 22, dtype=torch.bool),
        "levels": torch.randn(batch, 20, 10, 5),
        "l2_engineered": torch.randn(batch, 20, 16),
        "l2_mask": torch.ones(batch, 20, dtype=torch.bool),
        "message": torch.randn(batch, 5),
        "state": torch.randn(batch, 13),
    }


def test_regime_message_is_semantic_centered_and_expires_to_zero() -> None:
    message = regime_message(torch.tensor((1.0, 2.0, 4.0)), decision_time_ns=10, expiry_time_ns=20)
    assert isinstance(message, RegimeMessage)
    assert np.allclose(message.normalized_advantages, (-0.8, -0.2, 1.0))
    assert np.allclose(message.encode(10), (-0.8, -0.2, 1.0, 1.0, 1.0))
    assert np.allclose(message.encode(15), (-0.8, -0.2, 1.0, 1.0, 0.5))
    assert np.array_equal(message.encode(20), np.zeros(5, np.float32))
    with pytest.raises(ValueError, match="before publication"):
        message.encode(9)
    equal = regime_message(torch.ones(3), decision_time_ns=1, expiry_time_ns=2)
    assert equal.normalized_advantages == (0.0, 0.0, 0.0)


def test_two_level_models_match_registered_shapes_and_parameter_limits() -> None:
    torch.manual_seed(3)
    models = V6ModelBundle.create()
    top_q = models.regime(_top())
    bottom_q = models.trader(_bottom())
    assert top_q.shape == bottom_q.shape == (2, 3)
    assert torch.isfinite(top_q).all() and torch.isfinite(bottom_q).all()
    assert parameter_count(models.regime) <= TOP_PARAMETER_LIMIT
    assert parameter_count(models.trader) <= BOTTOM_PARAMETER_LIMIT
    assert parameter_count(models.regime) == 15_244
    assert parameter_count(models.trader) == 21_144


def test_xau_transfer_copies_only_common_base_encoders() -> None:
    torch.manual_seed(101)
    source = V6ModelBundle.create()
    torch.manual_seed(202)
    destination = V6ModelBundle.create()
    rich_before = {
        name: value.clone() for name, value in destination.regime.rich.state_dict().items()
    }
    head_before = {
        name: value.clone() for name, value in destination.trader.head.state_dict().items()
    }
    transfer_base_encoders(source, destination)
    for name, value in source.regime.base.state_dict().items():
        assert torch.equal(value, destination.regime.base.state_dict()[name])
    for name, value in source.trader.base.state_dict().items():
        assert torch.equal(value, destination.trader.base.state_dict()[name])
    assert all(
        torch.equal(value, destination.regime.rich.state_dict()[name])
        for name, value in rich_before.items()
    )
    assert all(
        torch.equal(value, destination.trader.head.state_dict()[name])
        for name, value in head_before.items()
    )


def test_padding_values_are_inert_but_valid_semantic_inputs_matter() -> None:
    torch.manual_seed(7)
    models = V6ModelBundle.create().eval()
    top = _top(1)
    top["sequence_mask"][:, :4] = False
    top_changed = {name: value.clone() for name, value in top.items()}
    top_changed["base"][:, :4] = 1_000_000
    top_changed["rich"][:, :4] = -1_000_000
    assert torch.allclose(models.regime(top), models.regime(top_changed), atol=1e-7)

    bottom = _bottom(1)
    bottom["l2_mask"][:, :10] = False
    bottom_changed = {name: value.clone() for name, value in bottom.items()}
    bottom_changed["levels"][:, :10] = 1_000_000
    bottom_changed["l2_engineered"][:, :10] = -1_000_000
    assert torch.allclose(models.trader(bottom), models.trader(bottom_changed), atol=1e-7)
    different_message = {name: value.clone() for name, value in bottom.items()}
    different_message["message"] += 5.0
    assert not torch.allclose(models.trader(bottom), models.trader(different_message))


def test_data_observation_arrays_feed_both_trainers_without_adapters() -> None:
    models = V6ModelBundle.create()
    top = _top(1)
    top_observation = RegimeObservation(
        top["base"][0].numpy(), top["rich"][0].numpy(),
        top["sequence_mask"][0].numpy(), top["context"][0].numpy(),
        top["context_mask"][0].numpy(), 1,
    )
    bottom = _bottom(1)
    trader_observation = TraderObservation(
        bottom["base"][0].numpy(), bottom["rich"][0].numpy(),
        bottom["sequence_mask"][0].numpy(), bottom["context"][0].numpy(),
        bottom["context_mask"][0].numpy(), bottom["levels"][0].numpy(),
        bottom["l2_engineered"][0].numpy(), bottom["l2_mask"][0].numpy(),
        bottom["message"][0].numpy(), bottom["state"][0].numpy(), 1,
    )
    regime_trainer = V6DDQNTrainer(
        "regime", models.regime, seed=1, replay_capacity=64, replay_warmup=64,
        epsilon_decay_decisions=10,
    )
    trader_trainer = V6DDQNTrainer(
        "trader", models.trader, seed=1, replay_capacity=64, replay_warmup=64,
        epsilon_decay_decisions=10,
    )
    assert regime_trainer.q_values(top_observation.arrays("rich_volume")).shape == (3,)
    assert trader_trainer.q_values(trader_observation.arrays("rich_volume")).shape == (3,)
    assert regime_trainer.q_values(top_observation.arrays("base_ohlcv")).shape == (3,)
    assert trader_trainer.q_values(trader_observation.arrays("base_ohlcv")).shape == (3,)


def test_model_rejects_wrong_shapes_dtypes_and_nonfinite_values() -> None:
    models = V6ModelBundle.create()
    top = _top(1)
    top["sequence_mask"] = top["sequence_mask"].float()
    with pytest.raises(ValueError, match="boolean"):
        models.regime(top)
    bottom = _bottom(1)
    bottom["state"] = torch.zeros(1, 12)
    with pytest.raises(ValueError, match="state"):
        models.trader(bottom)
    bottom = _bottom(1)
    bottom["levels"][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        models.trader(bottom)


class _TinyQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.tensor((1.0, 10.0, 2.0)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.q.unsqueeze(0).expand(len(batch["x"]), -1)


def _array_observation(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"x": np.asarray((value,), dtype=np.float32)}


def _tiny_trainer(*, seed: int = 11, target_sync: int = 2) -> V6DDQNTrainer:
    config = V6DDQNConfig(batch_size=2, target_sync_updates=target_sync)
    return V6DDQNTrainer(
        "tiny", _TinyQ(), seed=seed, replay_capacity=8, replay_warmup=2,
        epsilon_decay_decisions=4, config=config, reward_divisor=10.0,
        reward_unit="raw_cents",
    )


def test_ddqn_masks_behaviour_replay_and_bootstrap_actions() -> None:
    trainer = _tiny_trainer()
    observation = _array_observation()
    only_long = np.asarray((False, False, True))
    for _ in range(10):
        assert trainer.act(observation, legal=only_long, explore=True) == ACTIONS[2]
    with pytest.raises(ValueError, match="illegal behaviour"):
        trainer.append(
            observation, ACTIONS[1], 0.0, observation, terminal=False,
            legal=np.asarray((True, False, False)), next_legal=np.ones(3, bool),
        )

    legal = np.ones(3, bool)
    next_legal = np.asarray((True, False, False))
    for _ in range(2):
        trainer.append(
            observation, ACTIONS[0], 0.0, observation, terminal=False,
            legal=legal, next_legal=next_legal,
        )
    with torch.no_grad():
        trainer.target.q.copy_(torch.tensor((100.0, 3.0, 7.0)))
    # Online argmax is action 1, but it is illegal in next_legal.  Correct DDQN
    # therefore selects action 0 and bootstraps the target value 100.
    assert trainer.update() == pytest.approx(98.5)


def test_target_sync_counts_actual_optimizer_updates_and_rewards_reconcile() -> None:
    assert DEFAULT_DDQN_CONFIG.target_sync_updates == 100
    trainer = _tiny_trainer(target_sync=2)
    observation = _array_observation()
    legal = np.ones(3, bool)
    for reward in (10.0, -5.0):
        trainer.append(
            observation, ACTIONS[0], reward, observation, terminal=False,
            legal=legal, next_legal=legal,
        )
    trainer.audit_rewards()
    assert trainer.replay.items[0].learner_reward == 1.0
    assert trainer.raw_reward_total == 5.0
    assert trainer.update() is not None
    assert trainer.target_synchronizations == 0
    assert trainer.update() is not None
    assert trainer.updates == 2 and trainer.target_synchronizations == 1
    report = trainer.report()
    assert report["action_counts"] == {"-1": 2, "0": 0, "1": 0}
    assert report["legal_exposure_counts"] == {"-1": 2, "0": 2, "1": 2}
    assert report["target_synchronizations"] == 1
    for name, value in trainer.online.state_dict().items():
        assert torch.equal(value, trainer.target.state_dict()[name])
    reconcile_raw_rewards((10.0, -5.0), 5.0)
    with pytest.raises(AssertionError, match="terminal net P&L"):
        reconcile_raw_rewards((10.0, -5.0), 4.0)


def test_atomic_full_trainer_checkpoint_restores_replay_rng_and_counters(tmp_path: Path) -> None:
    trainer = _tiny_trainer(seed=23)
    observation = _array_observation()
    legal = np.ones(3, bool)
    for index in range(3):
        trainer.append(
            observation, ACTIONS[index], float(index), _array_observation(index + 1),
            terminal=index == 2, legal=legal,
            next_legal=np.zeros(3, bool) if index == 2 else legal,
        )
        trainer.update()
    path = trainer.save(tmp_path / "trainer.pt", {"fold": 1})
    assert path.is_file() and len(file_sha256(path)) == 64
    assert not list(tmp_path.glob("*.part"))

    restored = _tiny_trainer(seed=23)
    metadata = load_trainer_checkpoint(restored, path)
    assert metadata == {"fold": 1}
    assert len(restored.replay) == 3
    assert (restored.decisions, restored.transitions, restored.updates) == (
        trainer.decisions, trainer.transitions, trainer.updates,
    )
    assert restored.raw_reward_total == trainer.raw_reward_total
    assert restored.replay.rng.bit_generator.state == trainer.replay.rng.bit_generator.state
    for name, value in restored.online.state_dict().items():
        assert torch.equal(value, trainer.online.state_dict()[name])


def test_atomic_inference_bundle_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(31)
    models = V6ModelBundle.create().eval()
    top = _top(1)
    bottom = _bottom(1)
    expected_top = models.regime(top)
    expected_bottom = models.trader(bottom)
    protocol_sha256 = "a" * 64
    path = save_model_bundle(
        models, tmp_path / "models.pt", {"seed": 31}, protocol_id=protocol_sha256,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["protocol_id"] == protocol_sha256
    assert set(payload["models"]) == {"regime", "trader"}
    loaded, metadata = load_model_bundle(path, protocol_id=protocol_sha256)
    assert metadata == {"seed": 31}
    assert torch.equal(expected_top, loaded.regime(top))
    assert torch.equal(expected_bottom, loaded.trader(bottom))
    with pytest.raises(ValueError, match="identity"):
        load_model_bundle(path, protocol_id="b" * 64)
