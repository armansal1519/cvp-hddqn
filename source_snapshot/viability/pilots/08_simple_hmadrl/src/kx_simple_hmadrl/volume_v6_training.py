"""Replay, masked Double-DQN, and atomic checkpoints for V6.

The trainer keeps the raw economic reward beside the fixed learner-unit
conversion, applies legal-action masks during behaviour and bootstrapping, and
stores the complete replay/RNG state required for an exact laptop resume.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Callable, Mapping, Sequence, TypeAlias

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .volume_v6 import ACTION_COUNT, RegimeQNetwork, TraderQNetwork, V6ModelBundle
from .volume_v6 import parameter_count
from .volume_v6_contract import ACTIONS


ArrayValue: TypeAlias = np.ndarray | np.generic | float | int | bool
ArrayObservation: TypeAlias = Mapping[str, ArrayValue]
StoredObservation: TypeAlias = dict[str, np.ndarray]
TensorBatch: TypeAlias = dict[str, Tensor]
ForwardFunction: TypeAlias = Callable[[nn.Module, Mapping[str, Tensor]], Tensor]

CHECKPOINT_SCHEMA_VERSION = 1
TRAINER_ARTIFACT = "V6_FULL_DDQN_CHECKPOINT"
MODEL_ARTIFACT = "V6_INFERENCE_MODEL_BUNDLE"
DEFAULT_PROTOCOL_ID = "V6_DEVELOPMENT_UNFROZEN"
DEFAULT_ACTION_VALUES = ACTIONS


@dataclass(frozen=True)
class V6DDQNConfig:
    learning_rate: float = 2.5e-4
    batch_size: int = 64
    gamma: float = 1.0
    gradient_clip_norm: float = 10.0
    target_sync_updates: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.target_sync_updates < 1:
            raise ValueError("batch_size and target_sync_updates must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0,1]")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon bounds are invalid")


DEFAULT_DDQN_CONFIG = V6DDQNConfig()


def seed_everything(seed: int) -> None:
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)


def model_forward(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    value = model(batch)
    if not isinstance(value, Tensor):
        raise TypeError("V6 model forward must return a Q-value tensor")
    return value


def regime_forward(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    if not isinstance(model, RegimeQNetwork):
        raise TypeError("regime adapter received another model type")
    return model(batch)


def trader_forward(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    if not isinstance(model, TraderQNetwork):
        raise TypeError("trader adapter received another model type")
    return model(batch)


def _copy_observation(observation: ArrayObservation) -> StoredObservation:
    if not observation:
        raise ValueError("observation cannot be empty")
    result: StoredObservation = {}
    for name, value in observation.items():
        if not isinstance(name, str) or not name:
            raise ValueError("observation keys must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.kind in {"O", "S", "U", "V"}:
            raise ValueError(f"observation {name!r} has an unsupported dtype")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"observation {name!r} contains a non-finite value")
        if array.dtype == np.bool_:
            result[name] = np.ascontiguousarray(array, dtype=np.bool_).copy()
        elif np.issubdtype(array.dtype, np.number):
            result[name] = np.ascontiguousarray(array, dtype=np.float32).copy()
        else:
            raise ValueError(f"observation {name!r} is not numeric")
    return result


def _tensor_batch(
    observations: Sequence[ArrayObservation],
    device: torch.device,
) -> TensorBatch:
    if not observations:
        raise ValueError("cannot tensorize an empty observation batch")
    keys = tuple(sorted(observations[0]))
    if any(tuple(sorted(observation)) != keys for observation in observations[1:]):
        raise ValueError("observation keys differ within a replay batch")
    result: TensorBatch = {}
    for name in keys:
        try:
            array = np.stack([np.asarray(observation[name]) for observation in observations])
        except ValueError as error:
            raise ValueError(f"observation field {name!r} has inconsistent shapes") from error
        is_mask = name.endswith("_mask") or array.dtype == np.bool_
        dtype = torch.bool if is_mask else torch.float32
        result[name] = torch.as_tensor(array, dtype=dtype, device=device)
    return result


def _validate_action_mask(mask: np.ndarray, width: int, *, allow_empty: bool = False) -> np.ndarray:
    value = np.asarray(mask, dtype=np.bool_)
    if value.shape != (width,):
        raise ValueError(f"legal-action mask must have shape ({width},)")
    if not allow_empty and not bool(value.any()):
        raise ValueError("legal-action mask cannot be empty")
    return value.copy()


@dataclass(frozen=True)
class V6Transition:
    observation: StoredObservation
    action_index: int
    raw_reward: float
    learner_reward: float
    next_observation: StoredObservation
    terminal: bool
    legal: np.ndarray
    next_legal: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if int(capacity) < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.items: deque[V6Transition] = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.items)

    def append(self, transition: V6Transition) -> None:
        self.items.append(transition)

    def sample(self, size: int) -> list[V6Transition]:
        if int(size) > len(self.items):
            raise ValueError("replay sample exceeds its population")
        indexes = self.rng.choice(len(self.items), size=int(size), replace=False)
        values = list(self.items)
        return [values[int(index)] for index in indexes]

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "items": list(self.items),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("checkpoint replay capacity differs")
        items = state.get("items")
        if not isinstance(items, list) or len(items) > self.capacity:
            raise ValueError("checkpoint replay payload is invalid")
        if any(not isinstance(item, V6Transition) for item in items):
            raise ValueError("checkpoint replay contains an invalid transition")
        self.items.clear()
        self.items.extend(items)
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


class V6DDQNTrainer:
    """Legal-mask-aware Double-DQN with a complete resumable state."""

    def __init__(
        self,
        name: str,
        online: nn.Module,
        *,
        seed: int,
        replay_capacity: int,
        replay_warmup: int,
        epsilon_decay_decisions: int,
        action_values: Sequence[int] = DEFAULT_ACTION_VALUES,
        reward_divisor: float = 1.0,
        reward_unit: str = "raw",
        config: V6DDQNConfig = DEFAULT_DDQN_CONFIG,
        forward: ForwardFunction = model_forward,
        device: str | torch.device = "cpu",
        protocol_id: str = DEFAULT_PROTOCOL_ID,
    ) -> None:
        seed_everything(seed)
        if not name:
            raise ValueError("trainer name cannot be empty")
        if replay_warmup < config.batch_size or replay_warmup > replay_capacity:
            raise ValueError("replay warmup must be between batch size and capacity")
        if epsilon_decay_decisions < 1:
            raise ValueError("epsilon_decay_decisions must be positive")
        if not np.isfinite(reward_divisor) or reward_divisor <= 0.0:
            raise ValueError("reward_divisor must be one finite positive constant")
        action_values = tuple(int(value) for value in action_values)
        if len(action_values) != ACTION_COUNT or len(set(action_values)) != ACTION_COUNT:
            raise ValueError("V6 trainer requires three unique action values")
        self.name = str(name)
        self.seed = int(seed)
        self.config = config
        self.action_values = action_values
        self.reward_divisor = float(reward_divisor)
        self.reward_unit = str(reward_unit)
        self.replay_warmup = int(replay_warmup)
        self.epsilon_decay_decisions = int(epsilon_decay_decisions)
        self.forward = forward
        self.device = torch.device(device)
        self.protocol_id = str(protocol_id)
        self.online = online.to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device).eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.config.learning_rate,
        )
        self.replay = ReplayBuffer(replay_capacity, self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.decisions = 0
        self.transitions = 0
        self.optimization_calls = 0
        self.updates = 0
        self.target_synchronizations = 0
        self.raw_reward_total = 0.0
        self.action_counts = np.zeros(len(self.action_values), dtype=np.int64)
        self.legal_exposure_counts = np.zeros(len(self.action_values), dtype=np.int64)
        self.next_legal_exposure_counts = np.zeros(len(self.action_values), dtype=np.int64)

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.decisions / self.epsilon_decay_decisions)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def q_values(self, observation: ArrayObservation) -> np.ndarray:
        batch = _tensor_batch([observation], self.device)
        with torch.no_grad():
            q_values = self.forward(self.online, batch)
        if q_values.shape != (1, len(self.action_values)) or not bool(torch.isfinite(q_values).all()):
            raise ValueError("network returned invalid Q values")
        return q_values[0].detach().cpu().numpy().copy()

    def act(
        self,
        observation: ArrayObservation,
        *,
        legal: np.ndarray,
        explore: bool = True,
        count_decision: bool | None = None,
    ) -> int:
        legal_mask = _validate_action_mask(legal, len(self.action_values))
        if explore and self.rng.random() < self.epsilon:
            choices = np.flatnonzero(legal_mask)
            action_index = int(choices[int(self.rng.integers(len(choices)))])
        else:
            q_values = self.q_values(observation)
            q_values[~legal_mask] = -np.inf
            action_index = int(np.argmax(q_values))
        if count_decision if count_decision is not None else explore:
            self.decisions += 1
        return self.action_values[action_index]

    def append(
        self,
        observation: ArrayObservation,
        action: int,
        raw_reward: float,
        next_observation: ArrayObservation,
        *,
        terminal: bool,
        legal: np.ndarray,
        next_legal: np.ndarray,
    ) -> None:
        if int(action) not in self.action_values:
            raise ValueError("transition action is not registered")
        if not np.isfinite(raw_reward):
            raise ValueError("raw reward must be finite")
        legal_mask = _validate_action_mask(legal, len(self.action_values))
        next_mask = _validate_action_mask(
            next_legal, len(self.action_values), allow_empty=bool(terminal),
        )
        action_index = self.action_values.index(int(action))
        if not bool(legal_mask[action_index]):
            raise ValueError("replay transition contains an illegal behaviour action")
        transition = V6Transition(
            _copy_observation(observation), action_index, float(raw_reward),
            float(raw_reward) / self.reward_divisor, _copy_observation(next_observation),
            bool(terminal), legal_mask, next_mask,
        )
        self.replay.append(transition)
        self.transitions += 1
        self.raw_reward_total += float(raw_reward)
        self.action_counts[action_index] += 1
        self.legal_exposure_counts += legal_mask.astype(np.int64)
        self.next_legal_exposure_counts += next_mask.astype(np.int64)

    def update(self) -> float | None:
        self.optimization_calls += 1
        if len(self.replay) < self.replay_warmup:
            return None
        items = self.replay.sample(self.config.batch_size)
        observation = _tensor_batch([item.observation for item in items], self.device)
        next_observation = _tensor_batch(
            [item.next_observation for item in items], self.device,
        )
        actions = torch.as_tensor(
            [item.action_index for item in items], dtype=torch.long, device=self.device,
        )
        rewards = torch.as_tensor(
            [item.learner_reward for item in items], dtype=torch.float32, device=self.device,
        )
        terminal = torch.as_tensor(
            [item.terminal for item in items], dtype=torch.bool, device=self.device,
        )
        next_legal = torch.as_tensor(
            np.stack([item.next_legal for item in items]), dtype=torch.bool,
            device=self.device,
        )
        predicted = self.forward(self.online, observation).gather(
            1, actions[:, None],
        ).squeeze(1)
        with torch.no_grad():
            bootstrap_legal = next_legal.clone()
            bootstrap_legal[terminal] = True
            online_next = self.forward(self.online, next_observation).masked_fill(
                ~bootstrap_legal, -torch.inf,
            )
            selected = online_next.argmax(dim=1, keepdim=True)
            continuation = self.forward(self.target, next_observation).gather(
                1, selected,
            ).squeeze(1)
            expected = rewards + self.config.gamma * (~terminal).to(rewards.dtype) * continuation
        loss = F.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.target_synchronizations += 1
        return float(loss.detach().cpu())

    def audit_rewards(self) -> None:
        total = 0.0
        for item in self.replay.items:
            expected = item.raw_reward / self.reward_divisor
            if not np.isclose(item.learner_reward, expected, rtol=0.0, atol=1e-12):
                raise AssertionError("learner reward no longer reconciles to raw reward")
            total += item.raw_reward
        if self.transitions == len(self.replay) and not np.isclose(
            total, self.raw_reward_total, rtol=0.0, atol=1e-9,
        ):
            raise AssertionError("raw reward running total differs from replay")

    def report(self) -> dict[str, object]:
        """Return counters required by every V6 training-job receipt."""

        return {
            "name": self.name,
            "seed": self.seed,
            "parameters": parameter_count(self.online),
            "decisions": self.decisions,
            "transitions": self.transitions,
            "optimization_calls": self.optimization_calls,
            "updates": self.updates,
            "target_sync_updates": self.config.target_sync_updates,
            "target_synchronizations": self.target_synchronizations,
            "replay_size": len(self.replay),
            "replay_capacity": self.replay.capacity,
            "epsilon": self.epsilon,
            "raw_reward_total": self.raw_reward_total,
            "reward_divisor": self.reward_divisor,
            "reward_unit": self.reward_unit,
            "action_counts": {
                str(action): int(self.action_counts[index])
                for index, action in enumerate(self.action_values)
            },
            "legal_exposure_counts": {
                str(action): int(self.legal_exposure_counts[index])
                for index, action in enumerate(self.action_values)
            },
            "next_legal_exposure_counts": {
                str(action): int(self.next_legal_exposure_counts[index])
                for index, action in enumerate(self.action_values)
            },
        }

    def checkpoint_state(self, metadata: Mapping[str, object] | None = None) -> dict[str, object]:
        self.audit_rewards()
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "artifact_type": TRAINER_ARTIFACT,
            "protocol_id": self.protocol_id,
            "name": self.name,
            "seed": self.seed,
            "config": asdict(self.config),
            "action_values": self.action_values,
            "reward_divisor": self.reward_divisor,
            "reward_unit": self.reward_unit,
            "replay_warmup": self.replay_warmup,
            "epsilon_decay_decisions": self.epsilon_decay_decisions,
            "decisions": self.decisions,
            "transitions": self.transitions,
            "optimization_calls": self.optimization_calls,
            "updates": self.updates,
            "target_synchronizations": self.target_synchronizations,
            "raw_reward_total": self.raw_reward_total,
            "action_counts": self.action_counts.copy(),
            "legal_exposure_counts": self.legal_exposure_counts.copy(),
            "next_legal_exposure_counts": self.next_legal_exposure_counts.copy(),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "trainer_rng": copy.deepcopy(self.rng.bit_generator.state),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "metadata": {} if metadata is None else dict(metadata),
        }

    def load_checkpoint_state(self, state: Mapping[str, object]) -> dict[str, object]:
        identity = (
            state.get("schema_version"), state.get("artifact_type"),
            state.get("protocol_id"), state.get("name"), int(state.get("seed", -1)),
            state.get("config"), tuple(state.get("action_values", ())),
            int(state.get("replay_warmup", -1)),
            int(state.get("epsilon_decay_decisions", -1)), state.get("reward_unit"),
        )
        expected = (
            CHECKPOINT_SCHEMA_VERSION, TRAINER_ARTIFACT, self.protocol_id, self.name,
            self.seed, asdict(self.config), self.action_values, self.replay_warmup,
            self.epsilon_decay_decisions, self.reward_unit,
        )
        if identity != expected or not np.isclose(
            float(state.get("reward_divisor", -1.0)), self.reward_divisor,
        ):
            raise ValueError("V6 trainer checkpoint identity differs")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.target.eval()
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.decisions = int(state["decisions"])
        self.transitions = int(state["transitions"])
        self.optimization_calls = int(state["optimization_calls"])
        self.updates = int(state["updates"])
        self.target_synchronizations = int(state["target_synchronizations"])
        self.raw_reward_total = float(state["raw_reward_total"])
        for name in (
            "action_counts", "legal_exposure_counts", "next_legal_exposure_counts",
        ):
            value = np.asarray(state[name], dtype=np.int64)
            if value.shape != (len(self.action_values),) or (value < 0).any():
                raise ValueError(f"checkpoint {name} is invalid")
            setattr(self, name, value.copy())
        self.rng.bit_generator.state = copy.deepcopy(state["trainer_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        self.audit_rewards()
        metadata = state.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("checkpoint metadata is not a mapping")
        return dict(metadata)

    def save(
        self,
        path: str | os.PathLike[str],
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        return _atomic_torch_save(self.checkpoint_state(metadata), Path(path))


@dataclass
class V6TrainerBundle:
    models: V6ModelBundle
    regime: V6DDQNTrainer
    trader: V6DDQNTrainer


def create_trainers(
    *,
    seed: int,
    regime_epsilon_decay_decisions: int,
    trader_epsilon_decay_decisions: int,
    models: V6ModelBundle | None = None,
    regime_replay_warmup: int = 256,
    trader_replay_warmup: int = 512,
    reward_divisor: float = 1_000.0,
    reward_unit: str = "gc_ticks",
    config: V6DDQNConfig = DEFAULT_DDQN_CONFIG,
    device: str | torch.device = "cpu",
    protocol_id: str = DEFAULT_PROTOCOL_ID,
) -> V6TrainerBundle:
    """Create the paired upper/lower trainers with registered V6 defaults."""

    seed_everything(seed)
    model_bundle = V6ModelBundle.create() if models is None else models
    regime = V6DDQNTrainer(
        "regime", model_bundle.regime, seed=seed, replay_capacity=50_000,
        replay_warmup=regime_replay_warmup,
        epsilon_decay_decisions=regime_epsilon_decay_decisions,
        reward_divisor=reward_divisor, reward_unit=reward_unit, config=config,
        forward=regime_forward, device=device, protocol_id=protocol_id,
    )
    trader = V6DDQNTrainer(
        "trader", model_bundle.trader, seed=seed, replay_capacity=120_000,
        replay_warmup=trader_replay_warmup,
        epsilon_decay_decisions=trader_epsilon_decay_decisions,
        reward_divisor=reward_divisor, reward_unit=reward_unit, config=config,
        forward=trader_forward, device=device, protocol_id=protocol_id,
    )
    return V6TrainerBundle(model_bundle, regime, trader)


def reconcile_raw_rewards(
    rewards: Sequence[float],
    terminal_net_pnl: float,
    *,
    atol: float = 1e-9,
) -> None:
    """Fail closed unless transition rewards equal terminal economic P&L."""

    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(terminal_net_pnl):
        raise ValueError("reward reconciliation inputs must be finite one-dimensional values")
    if not np.isclose(values.sum(), float(terminal_net_pnl), rtol=0.0, atol=atol):
        raise AssertionError("raw transition rewards do not reconcile to terminal net P&L")


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent,
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # Some filesystems do not support directory fsync; rename remains atomic.
            pass
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def save_trainer_checkpoint(
    trainer: V6DDQNTrainer,
    path: str | os.PathLike[str],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    return trainer.save(path, metadata)


def load_trainer_checkpoint(
    trainer: V6DDQNTrainer,
    path: str | os.PathLike[str],
) -> dict[str, object]:
    state = torch.load(Path(path), map_location=trainer.device, weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError("V6 trainer checkpoint is not a mapping")
    return trainer.load_checkpoint_state(state)


def save_model_bundle(
    models: V6ModelBundle,
    path: str | os.PathLike[str],
    metadata: Mapping[str, object] | None = None,
    *,
    protocol_id: str = DEFAULT_PROTOCOL_ID,
) -> Path:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": MODEL_ARTIFACT,
        "protocol_id": str(protocol_id),
        "models": models.state_dict(),
        "metadata": {} if metadata is None else dict(metadata),
    }
    return _atomic_torch_save(payload, Path(path))


def load_model_bundle(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    protocol_id: str = DEFAULT_PROTOCOL_ID,
) -> tuple[V6ModelBundle, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("artifact_type") != MODEL_ARTIFACT
        or payload.get("protocol_id") != str(protocol_id)
    ):
        raise ValueError("V6 model checkpoint identity differs")
    models = V6ModelBundle.create().to(device)
    models.load_state_dict(payload["models"])
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("model checkpoint metadata is not a mapping")
    return models.eval(), dict(metadata)


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ArrayObservation", "CHECKPOINT_SCHEMA_VERSION", "DEFAULT_ACTION_VALUES",
    "DEFAULT_DDQN_CONFIG", "DEFAULT_PROTOCOL_ID", "MODEL_ARTIFACT", "ReplayBuffer",
    "TRAINER_ARTIFACT", "V6DDQNConfig", "V6DDQNTrainer", "V6TrainerBundle",
    "V6Transition", "create_trainers", "file_sha256", "load_model_bundle",
    "load_trainer_checkpoint", "model_forward", "reconcile_raw_rewards",
    "regime_forward", "save_model_bundle", "save_trainer_checkpoint", "seed_everything",
    "trader_forward",
]
