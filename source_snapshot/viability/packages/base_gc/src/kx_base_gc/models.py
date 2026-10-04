from __future__ import annotations

import hashlib
import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .contract import ACTIONS, AgentSpec, BaselineContract


try:  # Keep causal preprocessing usable before the reusable train extra is installed.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in the no-torch environment
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


TORCH_INSTALL_MESSAGE = (
    "PyTorch is required for Base-GC training. Install the reusable optional "
    "dependency with `uv sync --all-packages --extra train` (or install "
    "kx-base-gc[train])."
)


def torch_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(TORCH_INSTALL_MESSAGE)


if nn is not None:

    class VanillaQNetwork(nn.Module):
        def __init__(
            self,
            input_dimension: int,
            hidden_sizes: Sequence[int],
            action_count: int = 3,
            *,
            activation: str = "tanh",
        ) -> None:
            super().__init__()
            activation_type = nn.Tanh if activation == "tanh" else nn.ReLU
            layers: list[nn.Module] = []
            width = int(input_dimension)
            for hidden in hidden_sizes:
                layers.extend([nn.Linear(width, int(hidden)), activation_type()])
                width = int(hidden)
            layers.append(nn.Linear(width, int(action_count)))
            self.network = nn.Sequential(*layers)

        def forward(self, observation: Any) -> Any:
            return self.network(observation)


    class DuelingQNetwork(nn.Module):
        def __init__(
            self,
            input_dimension: int,
            hidden_sizes: Sequence[int],
            action_count: int = 3,
            *,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if not hidden_sizes:
                raise ValueError("dueling network requires hidden layers")
            activation_type = nn.ReLU if activation == "relu" else nn.Tanh
            layers: list[nn.Module] = []
            width = int(input_dimension)
            for hidden in hidden_sizes:
                layers.extend([nn.Linear(width, int(hidden)), activation_type()])
                width = int(hidden)
            self.trunk = nn.Sequential(*layers)
            self.value = nn.Linear(width, 1)
            self.advantage = nn.Linear(width, int(action_count))

        def forward(self, observation: Any) -> Any:
            latent = self.trunk(observation)
            advantage = self.advantage(latent)
            return self.value(latent) + advantage - advantage.mean(dim=-1, keepdim=True)

else:

    class VanillaQNetwork:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class DuelingQNetwork:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def configure_determinism(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        raw_threads = os.environ.get("KX_BASE_GC_TORCH_NUM_THREADS", "1")
        try:
            threads = int(raw_threads)
        except ValueError as error:
            raise ValueError(
                "KX_BASE_GC_TORCH_NUM_THREADS must be a positive integer"
            ) from error
        if threads < 1:
            raise ValueError("KX_BASE_GC_TORCH_NUM_THREADS must be a positive integer")
        torch.set_num_threads(threads)
        torch.manual_seed(int(seed))
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)


def build_q_network(contract: BaselineContract, agent: AgentSpec) -> Any:
    _require_torch()
    kwargs = {
        "input_dimension": agent.observation_dimension,
        "hidden_sizes": agent.hidden_sizes,
        "action_count": len(ACTIONS),
        "activation": contract.optimization.activation,
    }
    if contract.optimization.network_head == "dueling":
        return DuelingQNetwork(**kwargs)
    return VanillaQNetwork(**kwargs)


def model_parameter_sha256(model: Any) -> str:
    _require_torch()
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayItem:
    observation: np.ndarray
    action_index: int
    reward: float
    next_observation: np.ndarray
    terminal: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._items: deque[ReplayItem] = deque(maxlen=int(capacity))
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self._items)

    def append(self, item: ReplayItem) -> None:
        self._items.append(item)

    def sample(self, size: int) -> list[ReplayItem]:
        if size > len(self._items):
            raise ValueError("sample exceeds replay population")
        indexes = self._rng.choice(len(self._items), size=int(size), replace=False)
        items = list(self._items)
        return [items[int(index)] for index in indexes]


class DQNTrainer:
    """Matched vanilla-DQN / Double-Dueling trainer for registered comparisons."""

    def __init__(
        self,
        contract: BaselineContract,
        agent: AgentSpec,
        *,
        seed: int,
        device: str | None = None,
        replay_warmup_override: int | None = None,
    ) -> None:
        _require_torch()
        configure_determinism(seed)
        self.contract = contract
        self.agent = agent
        self.seed = int(seed)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.online = build_q_network(contract, agent).to(self.device)
        self.target = build_q_network(contract, agent).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        optimization = contract.optimization
        self.replay_warmup = (optimization.replay_warmup if replay_warmup_override is None
                              else int(replay_warmup_override))
        if self.replay_warmup < optimization.batch_size:
            raise ValueError("replay warmup cannot be smaller than the registered batch")
        optimizer_type = torch.optim.RMSprop if optimization.optimizer == "rmsprop" else torch.optim.Adam
        self.optimizer = optimizer_type(self.online.parameters(), lr=optimization.learning_rate)
        self.replay = ReplayBuffer(agent.replay_capacity, seed)
        self.updates = 0
        self.decisions = 0
        self._rng = np.random.default_rng(seed)

    @property
    def epsilon(self) -> float:
        spec = self.contract.optimization
        return max(
            spec.epsilon_end,
            spec.epsilon_start * self.agent.epsilon_decay**self.decisions,
        )

    def act(self, observation: Sequence[float], *, explore: bool = True) -> int:
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.agent.observation_dimension,):
            raise ValueError("observation has the wrong registered dimension")
        if explore and self._rng.random() < self.epsilon:
            action_index = int(self._rng.integers(0, len(ACTIONS)))
        else:
            with torch.no_grad():
                values = self.online(torch.as_tensor(array, device=self.device).unsqueeze(0))
                action_index = int(values.argmax(dim=1).item())
        self.decisions += 1
        return ACTIONS[action_index]

    def append(
        self,
        observation: Sequence[float],
        action: int,
        reward: float,
        next_observation: Sequence[float],
        *,
        terminal: bool = False,
    ) -> None:
        if action not in ACTIONS:
            raise ValueError("invalid replay action")
        self.replay.append(
            ReplayItem(
                np.asarray(observation, dtype=np.float32),
                ACTIONS.index(action),
                float(reward),
                np.asarray(next_observation, dtype=np.float32),
                bool(terminal),
            )
        )

    def update(self) -> float | None:
        spec = self.contract.optimization
        if len(self.replay) < self.replay_warmup:
            return None
        batch = self.replay.sample(spec.batch_size)
        observations = torch.as_tensor(
            np.stack([item.observation for item in batch]), device=self.device
        )
        actions = torch.as_tensor(
            [item.action_index for item in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        next_observations = torch.as_tensor(
            np.stack([item.next_observation for item in batch]), device=self.device
        )
        terminals = torch.as_tensor(
            [item.terminal for item in batch], dtype=torch.float32, device=self.device
        )
        predicted = self.online(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            if spec.target_rule == "double":
                selected = self.online(next_observations).argmax(dim=1, keepdim=True)
                continuation = self.target(next_observations).gather(1, selected).squeeze(1)
            else:
                continuation = self.target(next_observations).max(dim=1).values
            target = rewards + self.agent.gamma * (1.0 - terminals) * continuation
        loss = F.mse_loss(predicted, target) if spec.loss == "mse" else F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), spec.gradient_clip_norm)
        self.optimizer.step()
        self.updates += 1
        if self.updates % spec.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def parameter_sha256(self) -> str:
        return model_parameter_sha256(self.online)

    def save_checkpoint(self, path: str | Any) -> None:
        """Atomically persist frozen model/optimizer state for artifact binding.

        The replay population and RNG stream are intentionally not serialized;
        this artifact certifies the frozen model and is not a mid-run resume
        promise.
        """

        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(handle)
        try:
            torch.save(
                {
                    "contract_sha256": self.contract.sha256,
                    "variant": self.contract.variant,
                    "agent": self.agent.name,
                    "seed": self.seed,
                    "updates": self.updates,
                    "decisions": self.decisions,
                    "replay_warmup": self.replay_warmup,
                    "parameter_sha256": self.parameter_sha256(),
                    "online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                temporary_name,
            )
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
