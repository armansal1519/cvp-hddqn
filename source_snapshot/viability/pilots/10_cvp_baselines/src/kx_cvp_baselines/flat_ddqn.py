"""Same-information, same-capacity flat multi-clock Double-DQN baseline.

The model has no publisher, publication clock, expiry, or message channel. At
each five-minute decision it evaluates the latest completed causal 15-minute
observation and the full causal five-minute/L2/portfolio observation in one
jointly optimized network. Independent top and bottom action values meet only
at fixed, parameter-free late fusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from kx_simple_hmadrl.volume_v6 import (
    ACTION_COUNT,
    BOTTOM_BASE_FEATURES,
    BOTTOM_RICH_FEATURES,
    CONTEXT_FEATURES,
    L2_ENGINEERED_FEATURES,
    L2_LEVEL_FEATURES,
    L2Encoder,
    MaskedGRUEncoder,
    RegimeQNetwork,
    STATE_FEATURES,
    TOP_BASE_FEATURES,
    TOP_RICH_FEATURES,
    V6ModelBundle,
    parameter_count,
)
from kx_simple_hmadrl.volume_v6_contract import (
    ACTIONS,
    BOTTOM_HISTORY,
    L2_HISTORY,
    L2_LEVELS,
    TOP_HISTORY,
)
from kx_simple_hmadrl.volume_v6_data import RegimeObservation, TraderObservation
from kx_simple_hmadrl.volume_v6_training import (
    DEFAULT_DDQN_CONFIG,
    V6DDQNTrainer,
)

from .artifacts import file_sha256, read_json_object, workspace_root
from .contract import (
    FLAT_ENV_INTERACTIONS_PER_SEED,
    FLAT_OPTIMIZER_UPDATES_PER_SEED,
    FLAT_PARAMETER_COUNT,
    FLAT_PRIMARY_UPDATES_PER_SEED,
    FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED,
    PROTOCOL_ID,
    REGISTERED_PASSES,
    REPLAY_CAPACITY,
    REPLAY_WARMUP,
    SEEDS,
    SUPPLEMENTAL_UPDATES_BY_PASS,
)


TOP_PARAMETER_COUNT = 15_244
BOTTOM_PARAMETER_COUNT = 21_144
REWARD_DIVISOR = 1_000.0
REWARD_UNIT = "gc_ticks"
PARENT_COMPLETE_TRAINING_MANIFEST = (
    "viability/artifacts/development/"
    "v6_registered_training_terminal_eligible/complete_training_manifest.json"
)
PARENT_COMPLETE_TRAINING_SHA256 = (
    "3206e019c3d80d43dbc447137597eb2c2eab78ac723eca7f49c8610445fab13b"
)

TOP_KEYS = (
    "top_base",
    "top_rich",
    "top_sequence_mask",
    "top_context",
    "top_context_mask",
)
BOTTOM_KEYS = (
    "base",
    "rich",
    "sequence_mask",
    "context",
    "context_mask",
    "levels",
    "l2_engineered",
    "l2_mask",
    "state",
)
FLAT_OBSERVATION_KEYS = TOP_KEYS + BOTTOM_KEYS


def _copy_array(value: np.ndarray, *, boolean: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if boolean:
        return np.ascontiguousarray(array, dtype=np.bool_).copy()
    result = np.ascontiguousarray(array, dtype=np.float32).copy()
    if not np.isfinite(result).all():
        raise ValueError("flat observation contains a non-finite value")
    return result


@dataclass(frozen=True)
class FlatMultiClockObservation:
    """A message-free copy of one aligned V6 top and bottom observation."""

    top_base: np.ndarray
    top_rich: np.ndarray
    top_sequence_mask: np.ndarray
    top_context: np.ndarray
    top_context_mask: np.ndarray
    base: np.ndarray
    rich: np.ndarray
    sequence_mask: np.ndarray
    context: np.ndarray
    context_mask: np.ndarray
    levels: np.ndarray
    l2_engineered: np.ndarray
    l2_mask: np.ndarray
    state: np.ndarray
    top_decision_time_ns: int
    decision_time_ns: int
    top_segment: int
    bottom_segment: int
    top_available: bool

    @classmethod
    def from_v6(
        cls, top: RegimeObservation, bottom: TraderObservation, *,
        top_segment: int, bottom_segment: int,
    ) -> "FlatMultiClockObservation":
        top_time = int(top.decision_time_ns)
        decision = int(bottom.decision_time_ns)
        age = decision - top_time
        if top_time < 0 or decision < 0 or age < 0:
            raise ValueError("15-minute observation must be available by the 5-minute decision")
        if age >= 15 * 60 * 1_000_000_000:
            raise ValueError("15-minute observation is stale at the 5-minute decision")
        if int(top_segment) != int(bottom_segment):
            raise ValueError("15-minute and 5-minute observations cross a segment boundary")
        # Deliberately do not read bottom.message. Mutating the publisher message
        # in a source TraderObservation cannot affect this baseline.
        return cls(
            top_base=_copy_array(top.base),
            top_rich=_copy_array(top.rich),
            top_sequence_mask=_copy_array(top.sequence_mask, boolean=True),
            top_context=_copy_array(top.context),
            top_context_mask=_copy_array(top.context_mask, boolean=True),
            base=_copy_array(bottom.base),
            rich=_copy_array(bottom.rich),
            sequence_mask=_copy_array(bottom.sequence_mask, boolean=True),
            context=_copy_array(bottom.context),
            context_mask=_copy_array(bottom.context_mask, boolean=True),
            levels=_copy_array(bottom.levels),
            l2_engineered=_copy_array(bottom.l2_engineered),
            l2_mask=_copy_array(bottom.l2_mask, boolean=True),
            state=_copy_array(bottom.state),
            top_decision_time_ns=top_time,
            decision_time_ns=decision,
            top_segment=int(top_segment),
            bottom_segment=int(bottom_segment),
            top_available=True,
        )

    @classmethod
    def before_first_top(
        cls, bottom: TraderObservation, *, first_top_decision_time_ns: int,
        bottom_segment: int,
    ) -> "FlatMultiClockObservation":
        """Represent explicit top-clock unavailability without forward filling.

        This path is valid only when the five-minute decision strictly precedes
        the session/segment's first completed, available 15-minute observation.
        """

        decision = int(bottom.decision_time_ns)
        first_top = int(first_top_decision_time_ns)
        if decision < 0 or first_top < 0 or decision >= first_top:
            raise ValueError(
                "unavailable-top path is restricted to decisions before the first top state"
            )
        return cls(
            top_base=np.zeros((TOP_HISTORY, TOP_BASE_FEATURES), dtype=np.float32),
            top_rich=np.zeros((TOP_HISTORY, TOP_RICH_FEATURES), dtype=np.float32),
            top_sequence_mask=np.zeros(TOP_HISTORY, dtype=np.bool_),
            top_context=np.zeros(CONTEXT_FEATURES, dtype=np.float32),
            top_context_mask=np.zeros(CONTEXT_FEATURES, dtype=np.bool_),
            base=_copy_array(bottom.base),
            rich=_copy_array(bottom.rich),
            sequence_mask=_copy_array(bottom.sequence_mask, boolean=True),
            context=_copy_array(bottom.context),
            context_mask=_copy_array(bottom.context_mask, boolean=True),
            levels=_copy_array(bottom.levels),
            l2_engineered=_copy_array(bottom.l2_engineered),
            l2_mask=_copy_array(bottom.l2_mask, boolean=True),
            state=_copy_array(bottom.state),
            top_decision_time_ns=-1,
            decision_time_ns=decision,
            top_segment=-1,
            bottom_segment=int(bottom_segment),
            top_available=False,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            name: _copy_array(
                getattr(self, name),
                boolean=name.endswith("_mask"),
            )
            for name in FLAT_OBSERVATION_KEYS
        }


def _require_tensor(
    observation: Mapping[str, Tensor], name: str, *, rank: int,
    tail: tuple[int, ...], boolean: bool = False,
) -> Tensor:
    value = observation.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"observation field {name!r} must be a torch.Tensor")
    if value.ndim != rank or tuple(value.shape[-len(tail):]) != tail:
        expected = "batch," + ",".join(str(item) for item in tail)
        raise ValueError(f"{name} must have shape [{expected}]")
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    elif not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite floating point")
    return value


def _same_batch_device(values: Mapping[str, Tensor]) -> None:
    if len({int(value.shape[0]) for value in values.values()}) != 1:
        raise ValueError("flat observation fields have inconsistent batch sizes")
    if len({value.device for value in values.values()}) != 1:
        raise ValueError("flat observation fields must be on one device")


class PortfolioContextEncoder(nn.Module):
    """Message-free portfolio/context encoder with capacity reallocated to width 25."""

    def __init__(
        self, *, context_features: int = CONTEXT_FEATURES,
        state_features: int = STATE_FEATURES,
    ) -> None:
        super().__init__()
        self.context_features = int(context_features)
        self.state_features = int(state_features)
        input_width = self.state_features + self.context_features * 2
        self.network = nn.Sequential(
            nn.Linear(input_width, 25),
            nn.GELU(),
            nn.LayerNorm(25),
        )

    def forward(
        self, state: Tensor, context: Tensor, context_mask: Tensor,
    ) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != self.state_features:
            raise ValueError(f"state must have shape [batch,{self.state_features}]")
        if context.ndim != 2 or context.shape[-1] != self.context_features:
            raise ValueError(f"context must have shape [batch,{self.context_features}]")
        if context_mask.shape != context.shape or context_mask.dtype != torch.bool:
            raise ValueError("context_mask must be boolean and align with context")
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(context).all()):
            raise ValueError("portfolio/context input contains a non-finite value")
        valid = context_mask.to(context.dtype)
        return self.network(torch.cat((state, context * valid, valid), dim=-1))


class BiaslessDuelingHead(nn.Module):
    """V6 dueling head with four output biases reallocated upstream."""

    def __init__(self, input_width: int = 64, hidden_width: int = 32) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1, bias=False),
        )
        self.advantage = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, ACTION_COUNT, bias=False),
        )

    def forward(self, latent: Tensor) -> Tensor:
        advantage = self.advantage(latent)
        return self.value(latent) + advantage - advantage.mean(dim=-1, keepdim=True)


class FlatTraderBranch(nn.Module):
    """Full five-minute V6 trader state with no message input."""

    action_count = ACTION_COUNT

    def __init__(self) -> None:
        super().__init__()
        self.base = MaskedGRUEncoder(BOTTOM_BASE_FEATURES, 24)
        self.rich = MaskedGRUEncoder(BOTTOM_RICH_FEATURES, 24)
        self.l2 = L2Encoder(
            level_features=L2_LEVEL_FEATURES,
            engineered_features=L2_ENGINEERED_FEATURES,
        )
        self.portfolio_context = PortfolioContextEncoder()
        self.fusion = nn.Sequential(
            nn.Linear(24 + 24 + 16 + 25, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )
        self.head = BiaslessDuelingHead(64)
        if parameter_count(self) != BOTTOM_PARAMETER_COUNT:
            raise AssertionError("flat bottom branch is not exactly 21,144 parameters")

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        base = _require_tensor(
            observation, "base", rank=3,
            tail=(BOTTOM_HISTORY, BOTTOM_BASE_FEATURES),
        )
        rich = _require_tensor(
            observation, "rich", rank=3,
            tail=(BOTTOM_HISTORY, BOTTOM_RICH_FEATURES),
        )
        sequence_mask = _require_tensor(
            observation, "sequence_mask", rank=2,
            tail=(BOTTOM_HISTORY,), boolean=True,
        )
        context = _require_tensor(
            observation, "context", rank=2, tail=(CONTEXT_FEATURES,),
        )
        context_mask = _require_tensor(
            observation, "context_mask", rank=2,
            tail=(CONTEXT_FEATURES,), boolean=True,
        )
        levels = _require_tensor(
            observation, "levels", rank=4,
            tail=(L2_HISTORY, L2_LEVELS, L2_LEVEL_FEATURES),
        )
        engineered = _require_tensor(
            observation, "l2_engineered", rank=3,
            tail=(L2_HISTORY, L2_ENGINEERED_FEATURES),
        )
        l2_mask = _require_tensor(
            observation, "l2_mask", rank=2,
            tail=(L2_HISTORY,), boolean=True,
        )
        state = _require_tensor(
            observation, "state", rank=2, tail=(STATE_FEATURES,),
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
        latent = self.fusion(
            torch.cat(
                (
                    self.base(base, sequence_mask),
                    self.rich(rich, sequence_mask),
                    self.l2(levels, engineered, l2_mask),
                    self.portfolio_context(state, context, context_mask),
                ),
                dim=-1,
            )
        )
        result = self.head(latent)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("flat bottom branch produced non-finite Q values")
        return result


class FlatMultiClockQNetwork(nn.Module):
    """One end-to-end DDQN with parameter-free late fusion across two clocks."""

    action_count = ACTION_COUNT

    def __init__(self) -> None:
        super().__init__()
        self.top = RegimeQNetwork()
        self.bottom = FlatTraderBranch()
        if parameter_count(self.top) != TOP_PARAMETER_COUNT:
            raise AssertionError("flat top branch is not exactly 15,244 parameters")
        if parameter_count(self) != FLAT_PARAMETER_COUNT:
            raise AssertionError("flat multi-clock model is not exactly 36,388 parameters")

    def forward(self, observation: Mapping[str, Tensor]) -> Tensor:
        keys = set(observation)
        if "message" in keys or any(name.startswith("publisher_") for name in keys):
            raise ValueError("flat multi-clock DDQN has no publisher/message interface")
        if keys != set(FLAT_OBSERVATION_KEYS):
            missing = sorted(set(FLAT_OBSERVATION_KEYS).difference(keys))
            extra = sorted(keys.difference(FLAT_OBSERVATION_KEYS))
            raise ValueError(f"flat observation keys differ; missing={missing}, extra={extra}")
        top = {
            "base": observation["top_base"],
            "rich": observation["top_rich"],
            "sequence_mask": observation["top_sequence_mask"],
            "context": observation["top_context"],
            "context_mask": observation["top_context_mask"],
        }
        bottom = {name: observation[name] for name in BOTTOM_KEYS}
        top_q = self.top(top)
        bottom_q = self.bottom(bottom)
        if top_q.shape != bottom_q.shape:
            raise ValueError("flat clock branches produced incompatible Q shapes")
        # Fixed scaling preserves the magnitude of two similarly distributed
        # branches without adding a learned publisher/message bottleneck.
        result = (top_q + bottom_q) / math.sqrt(2.0)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("flat multi-clock DDQN produced non-finite Q values")
        return result


def flat_forward(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    if not isinstance(model, FlatMultiClockQNetwork):
        raise TypeError("flat adapter received another model type")
    return model(batch)


@torch.no_grad()
def transfer_price_encoders(
    source: V6ModelBundle, destination: FlatMultiClockQNetwork,
) -> None:
    """Transfer only the registered top/bottom OHLCV base encoders."""

    destination.top.base.load_state_dict(source.regime.base.state_dict())
    destination.bottom.base.load_state_dict(source.trader.base.state_dict())


def create_trainer(
    model: FlatMultiClockQNetwork, *, seed: int, expected_decisions: int,
) -> V6DDQNTrainer:
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"registered flat-DDQN seed must be one of {SEEDS}")
    return V6DDQNTrainer(
        "flat_multiclock_ddqn",
        model,
        seed=seed,
        replay_capacity=REPLAY_CAPACITY,
        replay_warmup=REPLAY_WARMUP,
        epsilon_decay_decisions=max(1, int(expected_decisions)),
        reward_divisor=REWARD_DIVISOR,
        reward_unit=REWARD_UNIT,
        config=DEFAULT_DDQN_CONFIG,
        forward=flat_forward,
        device="cpu",
        protocol_id=PROTOCOL_ID,
    )


def registered_trainer_settings(*, expected_decisions: int) -> dict[str, object]:
    return {
        "algorithm": "dueling_double_dqn_joint_end_to_end_late_fusion",
        "optimizer": "Adam",
        "loss": "Huber",
        "ddqn_config": asdict(DEFAULT_DDQN_CONFIG),
        "action_values": list(ACTIONS),
        "replay_capacity": REPLAY_CAPACITY,
        "replay_warmup": REPLAY_WARMUP,
        "replay_sampling": "uniform",
        "chronological_transition_ingestion": True,
        "chronological_session_passes": True,
        "passes": REGISTERED_PASSES,
        "epsilon_decay_decisions": int(expected_decisions),
        "reward_divisor": REWARD_DIVISOR,
        "reward_unit": REWARD_UNIT,
        "reward_clipping": False,
        "legal_masked_behavior": True,
        "legal_masked_bootstrap": True,
        "environment_interactions": FLAT_ENV_INTERACTIONS_PER_SEED,
        "ordinary_optimizer_updates": FLAT_PRIMARY_UPDATES_PER_SEED,
        "supplemental_replay_updates": FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED,
        "supplemental_updates_by_pass": list(SUPPLEMENTAL_UPDATES_BY_PASS),
        "optimizer_updates_total": FLAT_OPTIMIZER_UPDATES_PER_SEED,
        "parameter_count": FLAT_PARAMETER_COUNT,
        "message_interface": None,
        "late_fusion": "fixed_sum_div_sqrt_2",
        "device": "cpu",
    }


def apply_supplemental_replay_updates(
    trainer: V6DDQNTrainer, *, pass_number: int,
) -> int:
    """Perform the frozen extra replay updates that match the parent total."""

    if not 1 <= int(pass_number) <= REGISTERED_PASSES:
        raise ValueError("pass_number is outside the registered three passes")
    required = SUPPLEMENTAL_UPDATES_BY_PASS[int(pass_number) - 1]
    completed = 0
    for _ in range(required):
        loss = trainer.update()
        if loss is None:
            raise RuntimeError("supplemental update attempted before replay warmup")
        completed += 1
    return completed


def audit_registered_parent_budget(
    path: Path | None = None,
) -> dict[str, object]:
    """Verify exact counts for every frozen rich-parent seed, not one exemplar."""

    expected = workspace_root() / PARENT_COMPLETE_TRAINING_MANIFEST
    supplied = expected if path is None else Path(path).resolve()
    if supplied != expected.resolve():
        raise ValueError("parent training receipt is not the registered complete manifest")
    if not supplied.is_file() or file_sha256(supplied) != PARENT_COMPLETE_TRAINING_SHA256:
        raise ValueError("registered complete-training manifest changed")
    complete = read_json_object(supplied)
    records = complete.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("parent complete-training manifest lacks checkpoints")
    rich = [
        record for record in records
        if isinstance(record, dict) and record.get("arm") == "V6-RichVolume"
    ]
    if sorted(int(record.get("seed", -1)) for record in rich) != list(SEEDS):
        raise ValueError("parent rich members do not match all registered seeds")
    verified: list[dict[str, int]] = []
    for record in sorted(rich, key=lambda item: int(item["seed"])):
        member = supplied.parent / str(record.get("manifest", ""))
        if not member.is_file() or file_sha256(member) != record.get("manifest_sha256"):
            raise ValueError("parent rich member manifest changed")
        value = read_json_object(member)
        regime = value.get("regime")
        trader = value.get("trader")
        if not isinstance(regime, dict) or not isinstance(trader, dict):
            raise ValueError("parent member lacks trainer reports")
        seed = int(value.get("seed", -1))
        actual = {
            "seed": seed,
            "publisher_updates": int(regime.get("updates", -1)),
            "trader_updates": int(trader.get("updates", -1)),
            "trader_interactions": int(trader.get("transitions", -1)),
            "combined_updates": int(regime.get("updates", -1))
            + int(trader.get("updates", -1)),
        }
        if actual != {
            "seed": seed,
            "publisher_updates": FLAT_SUPPLEMENTAL_REPLAY_UPDATES_PER_SEED,
            "trader_updates": FLAT_PRIMARY_UPDATES_PER_SEED,
            "trader_interactions": FLAT_ENV_INTERACTIONS_PER_SEED,
            "combined_updates": FLAT_OPTIMIZER_UPDATES_PER_SEED,
        }:
            raise ValueError(f"parent update budget differs for seed {seed}")
        verified.append(actual)
    return {
        "manifest_path": PARENT_COMPLETE_TRAINING_MANIFEST,
        "manifest_sha256": PARENT_COMPLETE_TRAINING_SHA256,
        "all_registered_seeds_verified": True,
        "members": verified,
    }


def assert_training_report_budget(report: Mapping[str, object]) -> None:
    if (
        int(report.get("parameters", -1)) != FLAT_PARAMETER_COUNT
        or int(report.get("transitions", -1)) != FLAT_ENV_INTERACTIONS_PER_SEED
        or int(report.get("updates", -1)) != FLAT_OPTIMIZER_UPDATES_PER_SEED
    ):
        raise ValueError("flat-DDQN training report violates its registered budget")


__all__ = [
    "BOTTOM_KEYS",
    "BOTTOM_PARAMETER_COUNT",
    "FLAT_OBSERVATION_KEYS",
    "FlatMultiClockObservation",
    "FlatMultiClockQNetwork",
    "FlatTraderBranch",
    "PARENT_COMPLETE_TRAINING_MANIFEST",
    "PARENT_COMPLETE_TRAINING_SHA256",
    "TOP_KEYS",
    "TOP_PARAMETER_COUNT",
    "apply_supplemental_replay_updates",
    "assert_training_report_budget",
    "audit_registered_parent_budget",
    "create_trainer",
    "flat_forward",
    "registered_trainer_settings",
    "transfer_price_encoders",
]
