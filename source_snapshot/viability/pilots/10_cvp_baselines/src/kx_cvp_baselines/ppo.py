"""Matched recurrent PPO policy and trainer for Pilot 10.

The policy deliberately reuses the frozen V6 trader encoder layout.  Its GRUs
encode the same causal 24-bar and 20-book-state windows as the no-publisher
Double-DQN comparator; there is no carried hidden state between decisions.
Session isolation is instead enforced by the upstream observation masks and by
terminal boundaries in this module's rollout/GAE ledger.

The caller cannot supply a publisher message.  A five-field float32 zero vector
is constructed inside the network immediately before the message/state encoder.
This makes ``recurrent_ppo`` an information-matched single-agent comparator.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kx_simple_hmadrl.volume_v6 import (
    ACTION_COUNT,
    L2Encoder,
    MaskedGRUEncoder,
    MessageStateEncoder,
    parameter_count,
)
from kx_simple_hmadrl.volume_v6_contract import (
    ACTIONS,
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

from .contract import (
    FLAT_ENV_INTERACTIONS_PER_SEED,
    PROTOCOL_ID,
    REGISTERED_PASSES,
    BaselineArm,
)


ArrayValue: TypeAlias = np.ndarray | np.generic | float | int | bool
ArrayObservation: TypeAlias = Mapping[str, ArrayValue]
StoredObservation: TypeAlias = dict[str, np.ndarray]
TensorObservation: TypeAlias = Mapping[str, Tensor]

PPO_ARM = BaselineArm.RECURRENT_PPO.value
PPO_CHECKPOINT_SCHEMA_VERSION = 1
PPO_TRAINER_ARTIFACT = "CVP_BASELINE_RECURRENT_PPO_TRAINER"
PPO_MODEL_ARTIFACT = "CVP_BASELINE_RECURRENT_PPO_MODEL"
PPO_PARAMETER_COUNT = 21_144
PPO_REWARD_DIVISOR = 1_000.0
PPO_REWARD_UNIT = "gc_ticks"
DEFAULT_PROTOCOL_ID = PROTOCOL_ID
PPO_TRANSITIONS_PER_PASS = FLAT_ENV_INTERACTIONS_PER_SEED // REGISTERED_PASSES

PPO_OBSERVATION_KEYS = frozenset({
    "base",
    "rich",
    "sequence_mask",
    "context",
    "context_mask",
    "levels",
    "l2_engineered",
    "l2_mask",
    "state",
})


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """The single registered PPO configuration.

    These values are intentionally explicit in checkpoints and manifests.  A
    Pilot 10 trainer refuses a different configuration rather than silently
    creating another historical trial.
    """

    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    learning_rate: float = 2.5e-4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    gradient_clip_norm: float = 0.5
    epochs: int = 4
    rollout_size: int = 2_048
    minibatch_size: int = 256
    chronological_passes: int = REGISTERED_PASSES

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0,1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("GAE lambda must lie in [0,1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("PPO clip ratio must lie in (0,1)")
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive")
        if self.entropy_coefficient < 0.0 or self.value_coefficient < 0.0:
            raise ValueError("PPO loss coefficients must be nonnegative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")
        if min(
            self.epochs,
            self.rollout_size,
            self.minibatch_size,
            self.chronological_passes,
        ) < 1:
            raise ValueError("PPO counts must be positive")
        if self.minibatch_size > self.rollout_size:
            raise ValueError("PPO minibatch cannot exceed its rollout")


DEFAULT_PPO_CONFIG = PPOConfig()
PPO_FULL_ROLLOUTS_PER_PASS = (
    PPO_TRANSITIONS_PER_PASS // DEFAULT_PPO_CONFIG.rollout_size
)
PPO_FINAL_ROLLOUT_STEPS_PER_PASS = (
    PPO_TRANSITIONS_PER_PASS % DEFAULT_PPO_CONFIG.rollout_size
)
PPO_OPTIMIZATION_CALLS_PER_SEED = (
    PPO_FULL_ROLLOUTS_PER_PASS + 1
) * REGISTERED_PASSES
PPO_OPTIMIZER_UPDATES_PER_SEED = REGISTERED_PASSES * DEFAULT_PPO_CONFIG.epochs * (
    PPO_FULL_ROLLOUTS_PER_PASS
    * (DEFAULT_PPO_CONFIG.rollout_size // DEFAULT_PPO_CONFIG.minibatch_size)
    + int(np.ceil(PPO_FINAL_ROLLOUT_STEPS_PER_PASS / DEFAULT_PPO_CONFIG.minibatch_size))
)


def registered_ppo_settings() -> dict[str, object]:
    """Return the complete algorithm record for a training receipt."""

    return {
        "algorithm": "recurrent_ppo",
        "optimizer": "Adam",
        "config": asdict(DEFAULT_PPO_CONFIG),
        "action_values": list(ACTIONS),
        "observation_keys": sorted(PPO_OBSERVATION_KEYS),
        "message_mode": "internal_exact_float32_zero",
        "message_width": MESSAGE_WIDTH,
        "parameters": PPO_PARAMETER_COUNT,
        "environment_interactions_per_seed": FLAT_ENV_INTERACTIONS_PER_SEED,
        "environment_interactions_per_pass": PPO_TRANSITIONS_PER_PASS,
        "full_rollouts_per_pass": PPO_FULL_ROLLOUTS_PER_PASS,
        "final_rollout_steps_per_pass": PPO_FINAL_ROLLOUT_STEPS_PER_PASS,
        "optimization_calls_per_seed": PPO_OPTIMIZATION_CALLS_PER_SEED,
        "optimizer_updates_per_seed": PPO_OPTIMIZER_UPDATES_PER_SEED,
        "chronological_transition_ingestion": True,
        "chronological_session_passes": True,
        "cross_decision_hidden_state": False,
        "advantage_normalization": "whole_rollout_population_std",
        "reward_divisor": PPO_REWARD_DIVISOR,
        "reward_unit": PPO_REWARD_UNIT,
        "reward_clipping": False,
        "legal_masked_sampling": True,
        "legal_masked_objective": True,
        "deterministic_inference": "masked_argmax",
        "device": "cpu",
    }


def _require_exact_observation_keys(observation: Mapping[str, object]) -> None:
    keys = set(observation)
    if keys != PPO_OBSERVATION_KEYS:
        missing = sorted(PPO_OBSERVATION_KEYS - keys)
        extra = sorted(keys - PPO_OBSERVATION_KEYS)
        raise ValueError(
            "PPO observation keys differ from the matched bottom-state contract; "
            f"missing={missing}, extra={extra}"
        )


def _require_tensor(
    observation: TensorObservation,
    name: str,
    *,
    rank: int,
    tail: tuple[int, ...],
    boolean: bool = False,
) -> Tensor:
    try:
        value = observation[name]
    except KeyError as error:
        raise ValueError(f"observation is missing {name!r}") from error
    if not isinstance(value, Tensor):
        raise TypeError(f"observation field {name!r} must be a torch.Tensor")
    if value.ndim != rank or tuple(value.shape[-len(tail):]) != tail:
        expected = "batch," + ",".join(str(item) for item in tail)
        raise ValueError(f"{name} must have shape [{expected}]")
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    elif not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not boolean and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _same_batch_device(values: Mapping[str, Tensor]) -> None:
    if len({int(value.shape[0]) for value in values.values()}) != 1:
        raise ValueError("observation fields have inconsistent batch sizes")
    if len({value.device for value in values.values()}) != 1:
        raise ValueError("observation fields must be on one device")


class PPOActorCriticHead(nn.Module):
    """Actor and critic branches with the exact V6 dueling-head capacity."""

    def __init__(self, input_width: int = 64, hidden_width: int = 32) -> None:
        super().__init__()
        # Critic first, then actor, preserves the V6 value/advantage parameter
        # initialization order for a seed-matched capacity substitution.
        self.critic = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, ACTION_COUNT),
        )

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.actor(latent)
        values = self.critic(latent).squeeze(-1)
        return logits, values


class RecurrentPPOTrader(nn.Module):
    """Information- and parameter-matched no-publisher actor--critic."""

    action_count = ACTION_COUNT

    def __init__(self) -> None:
        super().__init__()
        self.base = MaskedGRUEncoder(BOTTOM_BASE_WIDTH, 24)
        self.rich = MaskedGRUEncoder(BOTTOM_RICH_WIDTH, 24)
        self.l2 = L2Encoder(
            level_features=L2_LEVEL_WIDTH,
            engineered_features=L2_ENGINEERED_WIDTH,
        )
        self.message_state = MessageStateEncoder(
            context_features=CONTEXT_WIDTH,
            state_features=STATE_WIDTH,
        )
        self.fusion = nn.Sequential(
            nn.Linear(24 + 24 + 16 + 24, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.head = PPOActorCriticHead(64)
        if parameter_count(self) != PPO_PARAMETER_COUNT:
            raise ValueError("PPO trader no longer has the matched 21,144 parameters")

    def forward(self, observation: TensorObservation) -> tuple[Tensor, Tensor]:
        _require_exact_observation_keys(observation)
        base = _require_tensor(
            observation,
            "base",
            rank=3,
            tail=(BOTTOM_HISTORY, BOTTOM_BASE_WIDTH),
        )
        rich = _require_tensor(
            observation,
            "rich",
            rank=3,
            tail=(BOTTOM_HISTORY, BOTTOM_RICH_WIDTH),
        )
        sequence_mask = _require_tensor(
            observation,
            "sequence_mask",
            rank=2,
            tail=(BOTTOM_HISTORY,),
            boolean=True,
        )
        context = _require_tensor(
            observation,
            "context",
            rank=2,
            tail=(CONTEXT_WIDTH,),
        )
        context_mask = _require_tensor(
            observation,
            "context_mask",
            rank=2,
            tail=(CONTEXT_WIDTH,),
            boolean=True,
        )
        levels = _require_tensor(
            observation,
            "levels",
            rank=4,
            tail=(L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH),
        )
        engineered = _require_tensor(
            observation,
            "l2_engineered",
            rank=3,
            tail=(L2_HISTORY, L2_ENGINEERED_WIDTH),
        )
        l2_mask = _require_tensor(
            observation,
            "l2_mask",
            rank=2,
            tail=(L2_HISTORY,),
            boolean=True,
        )
        state = _require_tensor(
            observation,
            "state",
            rank=2,
            tail=(STATE_WIDTH,),
        )
        values = {
            "base": base,
            "rich": rich,
            "sequence_mask": sequence_mask,
            "context": context,
            "context_mask": context_mask,
            "levels": levels,
            "l2_engineered": engineered,
            "l2_mask": l2_mask,
            "state": state,
        }
        _same_batch_device(values)
        # Constructed here, never accepted from caller data.
        message = torch.zeros(
            (len(base), MESSAGE_WIDTH), dtype=torch.float32, device=base.device,
        )
        latent = self.fusion(torch.cat((
            self.base(base, sequence_mask),
            self.rich(rich, sequence_mask),
            self.l2(levels, engineered, l2_mask),
            self.message_state(message, state, context, context_mask),
        ), dim=-1))
        logits, critic = self.head(latent)
        if (
            logits.shape != (len(base), ACTION_COUNT)
            or critic.shape != (len(base),)
            or not bool(torch.isfinite(logits).all())
            or not bool(torch.isfinite(critic).all())
        ):
            raise ValueError("PPO trader produced invalid actor or critic values")
        return logits, critic


def _clone_tensor_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("authorized XAU base-encoder state must be a nonempty mapping")
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("authorized XAU base-encoder state is malformed")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("authorized XAU base-encoder state is non-finite")
        result[name] = value.detach().cpu().clone()
    return result


def initialize_ppo_trader(
    *,
    seed: int,
    authorized_xau_base_state: Mapping[str, Tensor],
    device: str | torch.device = "cpu",
) -> RecurrentPPOTrader:
    """Create a seed-matched PPO trader and transfer only the XAU price GRU.

    Artifact/date authorization is deliberately a caller responsibility.  This
    narrow API accepts only the already-authorized base-encoder state, making it
    impossible to initialize rich, flow, L2, state, actor, or critic branches
    through the warm-start path.
    """

    seed_everything(int(seed))
    model = RecurrentPPOTrader()
    source = _clone_tensor_state(authorized_xau_base_state)
    expected_keys = set(model.base.state_dict())
    if set(source) != expected_keys:
        raise ValueError("authorized XAU state is not exactly the price base encoder")
    model.base.load_state_dict(source, strict=True)
    return model.to(device)


def _copy_observation(observation: ArrayObservation) -> StoredObservation:
    _require_exact_observation_keys(observation)
    result: StoredObservation = {}
    for name in sorted(PPO_OBSERVATION_KEYS):
        array = np.asarray(observation[name])
        if array.dtype.kind in {"O", "S", "U", "V"}:
            raise ValueError(f"observation {name!r} has an unsupported dtype")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"observation {name!r} contains a non-finite value")
        if array.dtype == np.bool_ or name.endswith("_mask"):
            result[name] = np.ascontiguousarray(array, dtype=np.bool_).copy()
        elif np.issubdtype(array.dtype, np.number):
            result[name] = np.ascontiguousarray(array, dtype=np.float32).copy()
        else:
            raise ValueError(f"observation {name!r} is not numeric")
    return result


def _tensor_batch(
    observations: Sequence[ArrayObservation],
    device: torch.device,
) -> dict[str, Tensor]:
    if not observations:
        raise ValueError("cannot tensorize an empty PPO observation batch")
    for observation in observations:
        _require_exact_observation_keys(observation)
    result: dict[str, Tensor] = {}
    for name in sorted(PPO_OBSERVATION_KEYS):
        try:
            array = np.stack([np.asarray(observation[name]) for observation in observations])
        except ValueError as error:
            raise ValueError(f"observation field {name!r} has inconsistent shapes") from error
        dtype = torch.bool if name.endswith("_mask") else torch.float32
        result[name] = torch.as_tensor(array, dtype=dtype, device=device)
    return result


def _legal_array(mask: object, *, allow_empty: bool = False) -> np.ndarray:
    value = np.asarray(mask, dtype=np.bool_)
    if value.shape != (ACTION_COUNT,):
        raise ValueError(f"legal-action mask must have shape ({ACTION_COUNT},)")
    if not allow_empty and not bool(value.any()):
        raise ValueError("legal-action mask cannot be empty")
    return value.copy()


def masked_logits(logits: Tensor, legal: Tensor) -> Tensor:
    """Mask illegal actions before sampling, argmax, entropy, or log-probability."""

    if logits.ndim != 2 or logits.shape[-1] != ACTION_COUNT:
        raise ValueError("actor logits must have shape [batch,3]")
    if legal.shape != logits.shape or legal.dtype != torch.bool:
        raise ValueError("legal mask must be boolean and align with actor logits")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("actor logits contain a non-finite value")
    if not bool(legal.any(dim=1).all()):
        raise ValueError("each policy row must expose at least one legal action")
    return logits.masked_fill(~legal, -torch.inf)


def deterministic_masked_argmax(logits: Tensor, legal: Tensor) -> Tensor:
    """Return the stable first-index argmax over legal actions only."""

    return masked_logits(logits, legal).argmax(dim=-1)


def _log_probabilities_and_entropy(
    logits: Tensor, legal: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    selected = masked_logits(logits, legal)
    log_probabilities = F.log_softmax(selected, dim=-1)
    probabilities = log_probabilities.exp()
    safe_log_probabilities = torch.where(
        legal, log_probabilities, torch.zeros_like(log_probabilities),
    )
    entropy = -(probabilities * safe_log_probabilities).sum(dim=-1)
    if not bool(torch.isfinite(entropy).all()):
        raise ValueError("masked actor entropy is non-finite")
    return selected, log_probabilities, entropy


@dataclass(frozen=True, slots=True)
class PPOLoss:
    total: Tensor
    actor: Tensor
    critic: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def ppo_loss(
    *,
    logits: Tensor,
    values: Tensor,
    legal: Tensor,
    action_indexes: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    returns: Tensor,
    config: PPOConfig = DEFAULT_PPO_CONFIG,
) -> PPOLoss:
    """Compute the registered clipped PPO objective for one minibatch."""

    if config != DEFAULT_PPO_CONFIG:
        raise ValueError("Pilot 10 PPO loss refuses an unregistered configuration")
    batch = logits.shape[0] if logits.ndim else -1
    for name, value in (
        ("values", values),
        ("old_log_probabilities", old_log_probabilities),
        ("advantages", advantages),
        ("returns", returns),
    ):
        if value.shape != (batch,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be one finite value per policy row")
    if action_indexes.shape != (batch,) or action_indexes.dtype != torch.long:
        raise ValueError("action indexes must be int64 with one value per policy row")
    if bool(((action_indexes < 0) | (action_indexes >= ACTION_COUNT)).any()):
        raise ValueError("action index lies outside SHORT/FLAT/LONG")
    _, log_probabilities, entropy_rows = _log_probabilities_and_entropy(logits, legal)
    chosen_is_legal = legal.gather(1, action_indexes[:, None]).squeeze(1)
    if not bool(chosen_is_legal.all()):
        raise ValueError("PPO objective contains an illegal behavior action")
    current = log_probabilities.gather(1, action_indexes[:, None]).squeeze(1)
    ratio = torch.exp(current - old_log_probabilities)
    unclipped = ratio * advantages
    clipped = ratio.clamp(
        1.0 - config.clip_ratio, 1.0 + config.clip_ratio,
    ) * advantages
    actor = -torch.minimum(unclipped, clipped).mean()
    critic = F.mse_loss(values, returns)
    entropy = entropy_rows.mean()
    total = (
        actor
        + config.value_coefficient * critic
        - config.entropy_coefficient * entropy
    )
    approximate_kl = (old_log_probabilities - current).mean()
    clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).to(values.dtype).mean()
    for name, value in (
        ("total", total),
        ("actor", actor),
        ("critic", critic),
        ("entropy", entropy),
        ("approximate_kl", approximate_kl),
        ("clip_fraction", clip_fraction),
    ):
        if value.ndim != 0 or not bool(torch.isfinite(value)):
            raise ValueError(f"PPO {name} loss statistic is invalid")
    return PPOLoss(total, actor, critic, entropy, approximate_kl, clip_fraction)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    terminals: Tensor,
    *,
    last_value: Tensor | float,
    gamma: float = DEFAULT_PPO_CONFIG.gamma,
    gae_lambda: float = DEFAULT_PPO_CONFIG.gae_lambda,
) -> tuple[Tensor, Tensor]:
    """Compute one chronological rollout's GAE without crossing terminals."""

    if rewards.ndim != 1 or values.shape != rewards.shape or terminals.shape != rewards.shape:
        raise ValueError("GAE inputs must be aligned one-dimensional tensors")
    if terminals.dtype != torch.bool:
        raise ValueError("GAE terminal markers must be boolean")
    if not rewards.is_floating_point() or not values.is_floating_point():
        raise ValueError("GAE rewards and values must be floating point")
    if not bool(torch.isfinite(rewards).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError("GAE inputs contain a non-finite value")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("GAE gamma/lambda must lie in [0,1]")
    bootstrap = torch.as_tensor(last_value, dtype=values.dtype, device=values.device)
    if bootstrap.shape != () or not bool(torch.isfinite(bootstrap)):
        raise ValueError("GAE bootstrap must be one finite scalar")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=values.dtype, device=values.device)
    following = bootstrap
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = (~terminals[index]).to(values.dtype)
        delta = rewards[index] + gamma * following * nonterminal - values[index]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
        following = values[index]
    returns = advantages + values
    return advantages, returns


@dataclass(frozen=True, slots=True)
class PPOAction:
    action: int
    action_index: int
    log_probability: float
    value: float


@dataclass(frozen=True, slots=True)
class PPORolloutStep:
    observation: StoredObservation
    action_index: int
    raw_reward: float
    learner_reward: float
    terminal: bool
    legal: np.ndarray
    old_log_probability: float
    old_value: float
    session_id: str


class PPORolloutBuffer:
    """Chronological on-policy storage with exact resume serialization."""

    def __init__(self, capacity: int = DEFAULT_PPO_CONFIG.rollout_size) -> None:
        if int(capacity) < 1:
            raise ValueError("PPO rollout capacity must be positive")
        self.capacity = int(capacity)
        self.items: list[PPORolloutStep] = []

    def __len__(self) -> int:
        return len(self.items)

    @property
    def full(self) -> bool:
        return len(self.items) == self.capacity

    def append(self, step: PPORolloutStep) -> None:
        if len(self.items) >= self.capacity:
            raise ValueError("PPO rollout is full")
        if not isinstance(step, PPORolloutStep):
            raise TypeError("PPO rollout accepts only validated steps")
        self.items.append(step)

    def clear(self) -> None:
        self.items.clear()

    def state_dict(self) -> dict[str, object]:
        return {"capacity": self.capacity, "items": copy.deepcopy(self.items)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("checkpoint PPO rollout capacity differs")
        items = state.get("items")
        if not isinstance(items, list) or len(items) > self.capacity:
            raise ValueError("checkpoint PPO rollout payload is invalid")
        if any(not isinstance(item, PPORolloutStep) for item in items):
            raise ValueError("checkpoint PPO rollout contains an invalid step")
        self.items = copy.deepcopy(items)


@dataclass(frozen=True, slots=True)
class PPOUpdateReport:
    rollout_steps: int
    epochs: int
    minibatch_updates: int
    mean_total_loss: float
    mean_actor_loss: float
    mean_critic_loss: float
    mean_entropy: float
    mean_approximate_kl: float
    mean_clip_fraction: float
    mean_gradient_norm: float


class PPOTrainer:
    """Legal-mask-aware, exact-resume trainer for one registered seed."""

    def __init__(
        self,
        model: RecurrentPPOTrader,
        *,
        seed: int,
        config: PPOConfig = DEFAULT_PPO_CONFIG,
        action_values: Sequence[int] = ACTIONS,
        reward_divisor: float = PPO_REWARD_DIVISOR,
        reward_unit: str = PPO_REWARD_UNIT,
        device: str | torch.device = "cpu",
        protocol_id: str = DEFAULT_PROTOCOL_ID,
    ) -> None:
        if config != DEFAULT_PPO_CONFIG:
            raise ValueError("Pilot 10 PPO trainer refuses an unregistered configuration")
        if not isinstance(model, RecurrentPPOTrader):
            raise TypeError("PPO trainer requires RecurrentPPOTrader")
        if tuple(int(value) for value in action_values) != tuple(ACTIONS):
            raise ValueError("PPO action values must be SHORT/FLAT/LONG")
        if not np.isfinite(reward_divisor) or float(reward_divisor) <= 0.0:
            raise ValueError("PPO reward divisor must be finite and positive")
        if not str(protocol_id):
            raise ValueError("PPO protocol id cannot be empty")
        seed_everything(int(seed))
        self.seed = int(seed)
        self.config = config
        self.action_values = tuple(int(value) for value in action_values)
        self.reward_divisor = float(reward_divisor)
        self.reward_unit = str(reward_unit)
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("registered Pilot 10 PPO training is CPU-only")
        self.protocol_id = str(protocol_id)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate,
        )
        self.rollout = PPORolloutBuffer(self.config.rollout_size)
        self.rng = np.random.default_rng(self.seed)
        self.decisions = 0
        self.transitions = 0
        self.optimization_calls = 0
        self.updates = 0
        self.passes_complete = 0
        self.active_pass: int | None = None
        self.current_session_id: str | None = None
        self.last_transition_terminal = True
        self.sessions_started = 0
        self.raw_reward_total = 0.0
        self.action_counts = np.zeros(ACTION_COUNT, dtype=np.int64)
        self.legal_exposure_counts = np.zeros(ACTION_COUNT, dtype=np.int64)
        self.update_reports: list[PPOUpdateReport] = []

    def begin_pass(self) -> int:
        if self.active_pass is not None:
            raise ValueError("a chronological PPO pass is already active")
        if self.passes_complete >= self.config.chronological_passes:
            raise ValueError("all registered PPO passes are already complete")
        if len(self.rollout):
            raise ValueError("a new PPO pass cannot begin with a pending rollout")
        self.active_pass = self.passes_complete + 1
        self.current_session_id = None
        self.last_transition_terminal = True
        self.sessions_started = 0
        return self.active_pass

    def _policy(
        self, observation: ArrayObservation,
    ) -> tuple[Tensor, Tensor]:
        batch = _tensor_batch([observation], self.device)
        logits, values = self.model(batch)
        return logits, values

    def act(
        self,
        observation: ArrayObservation,
        *,
        legal: object,
        deterministic: bool = False,
        count_decision: bool = True,
    ) -> PPOAction:
        legal_array = _legal_array(legal)
        legal_tensor = torch.as_tensor(
            legal_array[None, :], dtype=torch.bool, device=self.device,
        )
        with torch.no_grad():
            logits, values = self._policy(observation)
            selected, log_probabilities, _ = _log_probabilities_and_entropy(
                logits, legal_tensor,
            )
            if deterministic:
                action_index = int(selected.argmax(dim=-1).item())
            else:
                probabilities = selected.softmax(dim=-1)
                action_index = int(torch.multinomial(probabilities[0], 1).item())
            log_probability = float(log_probabilities[0, action_index].cpu())
            value = float(values[0].cpu())
        if not bool(legal_array[action_index]):
            raise AssertionError("masked PPO policy selected an illegal action")
        if count_decision:
            self.decisions += 1
            self.action_counts[action_index] += 1
            self.legal_exposure_counts += legal_array.astype(np.int64)
        return PPOAction(
            self.action_values[action_index], action_index, log_probability, value,
        )

    def _accept_session(self, session_id: str) -> str:
        value = str(session_id)
        if not value:
            raise ValueError("PPO session id cannot be empty")
        if self.active_pass is None:
            raise ValueError("begin_pass must precede PPO transitions")
        if self.current_session_id is None:
            self.current_session_id = value
            self.sessions_started = 1
        elif value != self.current_session_id:
            if not self.last_transition_terminal:
                raise ValueError("PPO session changed without a terminal reset")
            if value <= self.current_session_id:
                raise ValueError("PPO sessions are not strictly chronological")
            self.current_session_id = value
            self.sessions_started += 1
        elif self.last_transition_terminal:
            raise ValueError("PPO received another step after a terminal session step")
        return value

    def record(
        self,
        observation: ArrayObservation,
        decision: PPOAction,
        raw_reward: float,
        next_observation: ArrayObservation,
        *,
        terminal: bool,
        legal: object,
        next_legal: object,
        session_id: str,
    ) -> PPOUpdateReport | None:
        if not isinstance(decision, PPOAction):
            raise TypeError("PPO transition requires the action receipt returned by act")
        if not np.isfinite(raw_reward):
            raise ValueError("PPO raw reward must be finite")
        if (
            decision.action_index < 0
            or decision.action_index >= ACTION_COUNT
            or decision.action != self.action_values[decision.action_index]
            or not np.isfinite(decision.log_probability)
            or not np.isfinite(decision.value)
        ):
            raise ValueError("PPO action receipt is invalid")
        legal_array = _legal_array(legal)
        next_legal_array = _legal_array(next_legal, allow_empty=bool(terminal))
        if not bool(legal_array[decision.action_index]):
            raise ValueError("PPO transition contains an illegal behavior action")
        if bool(terminal) and bool(next_legal_array.any()):
            raise ValueError("terminal PPO transition must expose no next legal action")
        if not bool(terminal) and not bool(next_legal_array.any()):
            raise ValueError("nonterminal PPO transition needs a next legal action")
        accepted_session = self._accept_session(session_id)
        current = _copy_observation(observation)
        following = _copy_observation(next_observation)
        learner_reward = float(raw_reward) / self.reward_divisor
        self.rollout.append(PPORolloutStep(
            current,
            int(decision.action_index),
            float(raw_reward),
            learner_reward,
            bool(terminal),
            legal_array,
            float(decision.log_probability),
            float(decision.value),
            accepted_session,
        ))
        self.transitions += 1
        self.raw_reward_total += float(raw_reward)
        self.last_transition_terminal = bool(terminal)
        if not self.rollout.full:
            return None
        last_value = 0.0 if terminal else self.value(following)
        return self._optimize(last_value=last_value)

    def value(self, observation: ArrayObservation) -> float:
        with torch.no_grad():
            _, values = self._policy(observation)
        return float(values[0].cpu())

    def _optimize(self, *, last_value: float) -> PPOUpdateReport:
        if not len(self.rollout):
            raise ValueError("cannot optimize an empty PPO rollout")
        items = tuple(self.rollout.items)
        rewards = torch.as_tensor(
            [item.learner_reward for item in items], dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.as_tensor(
            [item.old_value for item in items], dtype=torch.float32,
            device=self.device,
        )
        terminals = torch.as_tensor(
            [item.terminal for item in items], dtype=torch.bool, device=self.device,
        )
        advantages, returns = compute_gae(
            rewards,
            old_values,
            terminals,
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        # Population standard deviation is deterministic for every rollout
        # size, including a one-step partial rollout at a pass boundary.
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        old_log_probabilities = torch.as_tensor(
            [item.old_log_probability for item in items],
            dtype=torch.float32,
            device=self.device,
        )
        action_indexes = torch.as_tensor(
            [item.action_index for item in items], dtype=torch.long, device=self.device,
        )
        legal = torch.as_tensor(
            np.stack([item.legal for item in items]),
            dtype=torch.bool,
            device=self.device,
        )
        observations = [item.observation for item in items]
        statistics: dict[str, list[float]] = {
            "total": [],
            "actor": [],
            "critic": [],
            "entropy": [],
            "kl": [],
            "clip": [],
            "gradient": [],
        }
        self.model.train()
        minibatch_updates = 0
        for _ in range(self.config.epochs):
            order = np.arange(len(items), dtype=np.int64)
            self.rng.shuffle(order)
            for start in range(0, len(order), self.config.minibatch_size):
                selected_indexes = order[start : start + self.config.minibatch_size]
                index = torch.as_tensor(
                    selected_indexes, dtype=torch.long, device=self.device,
                )
                batch = _tensor_batch(
                    [observations[int(position)] for position in selected_indexes],
                    self.device,
                )
                logits, predicted_values = self.model(batch)
                loss = ppo_loss(
                    logits=logits,
                    values=predicted_values,
                    legal=legal.index_select(0, index),
                    action_indexes=action_indexes.index_select(0, index),
                    old_log_probabilities=old_log_probabilities.index_select(0, index),
                    advantages=advantages.index_select(0, index),
                    returns=returns.index_select(0, index),
                    config=self.config,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm,
                )
                if not bool(torch.isfinite(gradient)):
                    raise ValueError("PPO gradient norm is non-finite")
                self.optimizer.step()
                minibatch_updates += 1
                self.updates += 1
                statistics["total"].append(float(loss.total.detach().cpu()))
                statistics["actor"].append(float(loss.actor.detach().cpu()))
                statistics["critic"].append(float(loss.critic.detach().cpu()))
                statistics["entropy"].append(float(loss.entropy.detach().cpu()))
                statistics["kl"].append(float(loss.approximate_kl.detach().cpu()))
                statistics["clip"].append(float(loss.clip_fraction.detach().cpu()))
                statistics["gradient"].append(float(gradient.detach().cpu()))
        self.optimization_calls += 1
        report = PPOUpdateReport(
            rollout_steps=len(items),
            epochs=self.config.epochs,
            minibatch_updates=minibatch_updates,
            mean_total_loss=float(np.mean(statistics["total"])),
            mean_actor_loss=float(np.mean(statistics["actor"])),
            mean_critic_loss=float(np.mean(statistics["critic"])),
            mean_entropy=float(np.mean(statistics["entropy"])),
            mean_approximate_kl=float(np.mean(statistics["kl"])),
            mean_clip_fraction=float(np.mean(statistics["clip"])),
            mean_gradient_norm=float(np.mean(statistics["gradient"])),
        )
        self.update_reports.append(report)
        self.rollout.clear()
        return report

    def finish_pass(self) -> PPOUpdateReport | None:
        if self.active_pass is None:
            raise ValueError("no chronological PPO pass is active")
        if self.current_session_id is None:
            raise ValueError("a chronological PPO pass cannot be empty")
        if not self.last_transition_terminal:
            raise ValueError("PPO pass ended before the session terminal reset")
        report = self._optimize(last_value=0.0) if len(self.rollout) else None
        self.passes_complete += 1
        if self.passes_complete != self.active_pass:
            raise AssertionError("PPO chronological pass counter diverged")
        self.active_pass = None
        self.current_session_id = None
        self.last_transition_terminal = True
        self.sessions_started = 0
        return report

    def report(self) -> dict[str, object]:
        return {
            "arm": PPO_ARM,
            "seed": self.seed,
            "parameters": parameter_count(self.model),
            "decisions": self.decisions,
            "transitions": self.transitions,
            "optimization_calls": self.optimization_calls,
            "updates": self.updates,
            "passes_complete": self.passes_complete,
            "active_pass": self.active_pass,
            "rollout_size": len(self.rollout),
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
            "settings": registered_ppo_settings(),
        }

    @property
    def training_complete(self) -> bool:
        """Whether the exact registered exposure and update budget is complete."""

        return (
            self.passes_complete == self.config.chronological_passes
            and self.active_pass is None
            and len(self.rollout) == 0
            and self.transitions == FLAT_ENV_INTERACTIONS_PER_SEED
            and self.optimization_calls == PPO_OPTIMIZATION_CALLS_PER_SEED
            and self.updates == PPO_OPTIMIZER_UPDATES_PER_SEED
        )

    def checkpoint_state(
        self, metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": PPO_CHECKPOINT_SCHEMA_VERSION,
            "artifact_type": PPO_TRAINER_ARTIFACT,
            "arm": PPO_ARM,
            "protocol_id": self.protocol_id,
            "seed": self.seed,
            "config": asdict(self.config),
            "action_values": self.action_values,
            "reward_divisor": self.reward_divisor,
            "reward_unit": self.reward_unit,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rollout": self.rollout.state_dict(),
            "decisions": self.decisions,
            "transitions": self.transitions,
            "optimization_calls": self.optimization_calls,
            "updates": self.updates,
            "passes_complete": self.passes_complete,
            "active_pass": self.active_pass,
            "current_session_id": self.current_session_id,
            "last_transition_terminal": self.last_transition_terminal,
            "sessions_started": self.sessions_started,
            "raw_reward_total": self.raw_reward_total,
            "action_counts": self.action_counts.copy(),
            "legal_exposure_counts": self.legal_exposure_counts.copy(),
            "update_reports": copy.deepcopy(self.update_reports),
            "trainer_rng": copy.deepcopy(self.rng.bit_generator.state),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "metadata": {} if metadata is None else dict(metadata),
        }

    def load_checkpoint_state(self, state: Mapping[str, object]) -> dict[str, object]:
        identity = (
            state.get("schema_version"),
            state.get("artifact_type"),
            state.get("arm"),
            state.get("protocol_id"),
            int(state.get("seed", -1)),
            state.get("config"),
            tuple(state.get("action_values", ())),
            state.get("reward_unit"),
        )
        expected = (
            PPO_CHECKPOINT_SCHEMA_VERSION,
            PPO_TRAINER_ARTIFACT,
            PPO_ARM,
            self.protocol_id,
            self.seed,
            asdict(self.config),
            self.action_values,
            self.reward_unit,
        )
        if identity != expected or not np.isclose(
            float(state.get("reward_divisor", -1.0)), self.reward_divisor,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("PPO trainer checkpoint identity differs")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rollout.load_state_dict(state["rollout"])
        self.decisions = int(state["decisions"])
        self.transitions = int(state["transitions"])
        self.optimization_calls = int(state["optimization_calls"])
        self.updates = int(state["updates"])
        self.passes_complete = int(state["passes_complete"])
        active = state.get("active_pass")
        self.active_pass = None if active is None else int(active)
        current = state.get("current_session_id")
        self.current_session_id = None if current is None else str(current)
        self.last_transition_terminal = bool(state["last_transition_terminal"])
        self.sessions_started = int(state["sessions_started"])
        self.raw_reward_total = float(state["raw_reward_total"])
        for name in ("action_counts", "legal_exposure_counts"):
            value = np.asarray(state[name], dtype=np.int64)
            if value.shape != (ACTION_COUNT,) or bool((value < 0).any()):
                raise ValueError(f"checkpoint {name} is invalid")
            setattr(self, name, value.copy())
        reports = state.get("update_reports")
        if not isinstance(reports, list) or any(
            not isinstance(report, PPOUpdateReport) for report in reports
        ):
            raise ValueError("checkpoint PPO update reports are invalid")
        self.update_reports = copy.deepcopy(reports)
        if not 0 <= self.passes_complete <= self.config.chronological_passes:
            raise ValueError("checkpoint PPO pass counter is invalid")
        if self.active_pass is not None and self.active_pass != self.passes_complete + 1:
            raise ValueError("checkpoint active PPO pass is inconsistent")
        if self.current_session_id is None and len(self.rollout):
            raise ValueError("checkpoint pending rollout lacks its session identity")
        self.rng.bit_generator.state = copy.deepcopy(state["trainer_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        metadata = state.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("PPO checkpoint metadata is not a mapping")
        return dict(metadata)


def create_ppo_trainer(
    *,
    seed: int,
    authorized_xau_base_state: Mapping[str, Tensor],
    device: str | torch.device = "cpu",
    protocol_id: str = DEFAULT_PROTOCOL_ID,
) -> PPOTrainer:
    """Initialize the registered model and its exact fixed-config trainer."""

    model = initialize_ppo_trader(
        seed=seed,
        authorized_xau_base_state=authorized_xau_base_state,
        device=device,
    )
    return PPOTrainer(
        model,
        seed=seed,
        device=device,
        protocol_id=protocol_id,
    )


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
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def save_ppo_checkpoint(
    trainer: PPOTrainer,
    path: str | os.PathLike[str],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    return _atomic_torch_save(trainer.checkpoint_state(metadata), Path(path))


def load_ppo_checkpoint(
    trainer: PPOTrainer,
    path: str | os.PathLike[str],
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location=trainer.device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("PPO trainer checkpoint is not a mapping")
    return trainer.load_checkpoint_state(payload)


def restore_ppo_trainer(
    path: str | os.PathLike[str],
    *,
    authorized_xau_base_state: Mapping[str, Tensor],
    device: str | torch.device = "cpu",
    protocol_id: str = DEFAULT_PROTOCOL_ID,
) -> tuple[PPOTrainer, dict[str, object]]:
    """Construct and restore one trainer without trusting checkpoint identity."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("PPO trainer checkpoint is not a mapping")
    trainer = create_ppo_trainer(
        seed=int(payload.get("seed", -1)),
        authorized_xau_base_state=authorized_xau_base_state,
        device=device,
        protocol_id=protocol_id,
    )
    metadata = trainer.load_checkpoint_state(payload)
    return trainer, metadata


def save_ppo_model(
    trainer: PPOTrainer,
    path: str | os.PathLike[str],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    if not trainer.training_complete:
        raise ValueError(
            "refusing to export an incomplete PPO model or mismatched training budget"
        )
    payload = {
        "schema_version": PPO_CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": PPO_MODEL_ARTIFACT,
        "arm": PPO_ARM,
        "protocol_id": trainer.protocol_id,
        "seed": trainer.seed,
        "config": asdict(trainer.config),
        "parameters": parameter_count(trainer.model),
        "model": trainer.model.state_dict(),
        "metadata": {} if metadata is None else dict(metadata),
    }
    return _atomic_torch_save(payload, Path(path))


def load_ppo_model(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    protocol_id: str = DEFAULT_PROTOCOL_ID,
) -> tuple[RecurrentPPOTrader, int, dict[str, object]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != PPO_CHECKPOINT_SCHEMA_VERSION
        or payload.get("artifact_type") != PPO_MODEL_ARTIFACT
        or payload.get("arm") != PPO_ARM
        or payload.get("protocol_id") != str(protocol_id)
        or payload.get("config") != asdict(DEFAULT_PPO_CONFIG)
        or int(payload.get("parameters", -1)) != PPO_PARAMETER_COUNT
    ):
        raise ValueError("PPO inference model identity differs")
    model = RecurrentPPOTrader().to(device)
    model.load_state_dict(payload["model"], strict=True)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("PPO inference metadata is not a mapping")
    return model.eval(), int(payload["seed"]), dict(metadata)


__all__ = [
    "ArrayObservation",
    "DEFAULT_PPO_CONFIG",
    "DEFAULT_PROTOCOL_ID",
    "PPOAction",
    "PPOActorCriticHead",
    "PPOConfig",
    "PPOLoss",
    "PPOUpdateReport",
    "PPOTrainer",
    "PPO_ARM",
    "PPO_CHECKPOINT_SCHEMA_VERSION",
    "PPO_MODEL_ARTIFACT",
    "PPO_OBSERVATION_KEYS",
    "PPO_PARAMETER_COUNT",
    "PPO_FINAL_ROLLOUT_STEPS_PER_PASS",
    "PPO_FULL_ROLLOUTS_PER_PASS",
    "PPO_OPTIMIZATION_CALLS_PER_SEED",
    "PPO_OPTIMIZER_UPDATES_PER_SEED",
    "PPO_REWARD_DIVISOR",
    "PPO_REWARD_UNIT",
    "PPO_TRAINER_ARTIFACT",
    "PPO_TRANSITIONS_PER_PASS",
    "RecurrentPPOTrader",
    "compute_gae",
    "create_ppo_trainer",
    "deterministic_masked_argmax",
    "initialize_ppo_trader",
    "load_ppo_checkpoint",
    "load_ppo_model",
    "masked_logits",
    "ppo_loss",
    "registered_ppo_settings",
    "restore_ppo_trainer",
    "save_ppo_checkpoint",
    "save_ppo_model",
]
