"""Compact two-level PyTorch models for the V6 volume hierarchy.

The upper network publishes a semantic value message.  Only the lower network
selects a position in the economic trading environment.  Inputs are mappings
so the data builder can remain independent of PyTorch, while every public
forward pass still validates the frozen tensor shapes explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import torch
from torch import Tensor, nn

from .volume_v6_contract import (
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
    TOP_BASE_WIDTH,
    TOP_HISTORY,
    TOP_RICH_WIDTH,
    RegimeMessage,
)


ACTION_COUNT = 3
TOP_BASE_FEATURES = TOP_BASE_WIDTH
TOP_RICH_FEATURES = TOP_RICH_WIDTH
BOTTOM_BASE_FEATURES = BOTTOM_BASE_WIDTH
BOTTOM_RICH_FEATURES = BOTTOM_RICH_WIDTH
CONTEXT_FEATURES = CONTEXT_WIDTH
L2_LEVEL_FEATURES = L2_LEVEL_WIDTH
L2_ENGINEERED_FEATURES = L2_ENGINEERED_WIDTH
STATE_FEATURES = STATE_WIDTH

TOP_PARAMETER_LIMIT = 25_000
BOTTOM_PARAMETER_LIMIT = 60_000

TensorObservation: TypeAlias = Mapping[str, Tensor]


def parameter_count(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


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
    batches = {int(value.shape[0]) for value in values.values()}
    devices = {value.device for value in values.values()}
    if len(batches) != 1:
        raise ValueError("observation fields have inconsistent batch sizes")
    if len(devices) != 1:
        raise ValueError("observation fields must be on one device")


def _compact_valid(sequence: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Left-compact valid rows before packing, preserving temporal order."""

    lengths = mask.sum(dim=1)
    compact = torch.zeros_like(sequence)
    for row in range(len(sequence)):
        valid = sequence[row, mask[row]]
        if len(valid):
            compact[row, : len(valid)] = valid
    return compact, lengths


class MaskedGRUEncoder(nn.Module):
    """A small GRU that is invariant to values in padded time steps."""

    def __init__(self, feature_width: int, hidden_width: int = 24) -> None:
        super().__init__()
        self.feature_width = int(feature_width)
        self.hidden_width = int(hidden_width)
        self.input_norm = nn.LayerNorm(self.feature_width)
        self.gru = nn.GRU(self.feature_width, self.hidden_width, batch_first=True)
        self.output_norm = nn.LayerNorm(self.hidden_width)

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.feature_width:
            raise ValueError(
                f"GRU sequence must have shape [batch,time,{self.feature_width}]"
            )
        if mask.shape != sequence.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("GRU mask must be boolean and align with [batch,time]")
        if not bool(torch.isfinite(sequence).all()):
            raise ValueError("GRU sequence contains a non-finite value")
        normalized = self.input_norm(sequence) * mask.unsqueeze(-1).to(sequence.dtype)
        compact, lengths = _compact_valid(normalized, mask)
        encoded = sequence.new_zeros(len(sequence), self.hidden_width)
        nonempty = lengths > 0
        if bool(nonempty.any()):
            packed = nn.utils.rnn.pack_padded_sequence(
                compact[nonempty], lengths[nonempty].cpu(), batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            encoded[nonempty] = hidden[-1]
        return self.output_norm(encoded)


class MaskedContextEncoder(nn.Module):
    def __init__(self, feature_width: int = CONTEXT_FEATURES, output_width: int = 24) -> None:
        super().__init__()
        self.feature_width = int(feature_width)
        self.network = nn.Sequential(
            nn.Linear(self.feature_width * 2, output_width),
            nn.GELU(),
            nn.LayerNorm(output_width),
        )

    def forward(self, context: Tensor, mask: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[-1] != self.feature_width:
            raise ValueError(f"context must have shape [batch,{self.feature_width}]")
        if mask.shape != context.shape or mask.dtype != torch.bool:
            raise ValueError("context_mask must be boolean and align with context")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("context contains a non-finite value")
        valid = mask.to(context.dtype)
        return self.network(torch.cat((context * valid, valid), dim=-1))


class DuelingHead(nn.Module):
    def __init__(self, input_width: int = 64, hidden_width: int = 32) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(input_width, hidden_width), nn.ReLU(), nn.Linear(hidden_width, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(input_width, hidden_width), nn.ReLU(),
            nn.Linear(hidden_width, ACTION_COUNT),
        )

    def forward(self, latent: Tensor) -> Tensor:
        advantage = self.advantage(latent)
        return self.value(latent) + advantage - advantage.mean(dim=-1, keepdim=True)


class RegimeQNetwork(nn.Module):
    """15-minute publisher: two GRUs, masked context, and a dueling head."""

    action_count = ACTION_COUNT

    def __init__(
        self,
        *,
        base_features: int = TOP_BASE_FEATURES,
        rich_features: int = TOP_RICH_FEATURES,
        context_features: int = CONTEXT_FEATURES,
    ) -> None:
        super().__init__()
        self.base_features = int(base_features)
        self.rich_features = int(rich_features)
        self.context_features = int(context_features)
        self.base = MaskedGRUEncoder(self.base_features, 24)
        self.rich = MaskedGRUEncoder(self.rich_features, 24)
        self.context = MaskedContextEncoder(self.context_features, 24)
        self.fusion = nn.Sequential(
            nn.Linear(72, 64), nn.GELU(), nn.LayerNorm(64),
        )
        self.head = DuelingHead(64)
        if parameter_count(self) > TOP_PARAMETER_LIMIT:
            raise ValueError("upper V6 network exceeds its 25,000-parameter contract")

    def forward(self, observation: TensorObservation) -> Tensor:
        base = _require_tensor(
            observation, "base", rank=3, tail=(TOP_HISTORY, self.base_features),
        )
        rich = _require_tensor(
            observation, "rich", rank=3, tail=(TOP_HISTORY, self.rich_features),
        )
        mask = _require_tensor(
            observation, "sequence_mask", rank=2, tail=(TOP_HISTORY,), boolean=True,
        )
        context = _require_tensor(
            observation, "context", rank=2, tail=(self.context_features,),
        )
        context_mask = _require_tensor(
            observation, "context_mask", rank=2, tail=(self.context_features,), boolean=True,
        )
        values = {
            "base": base, "rich": rich, "sequence_mask": mask,
            "context": context, "context_mask": context_mask,
        }
        _same_batch_device(values)
        latent = self.fusion(torch.cat((
            self.base(base, mask), self.rich(rich, mask),
            self.context(context, context_mask),
        ), dim=-1))
        q_values = self.head(latent)
        if not bool(torch.isfinite(q_values).all()):
            raise ValueError("upper network produced non-finite Q values")
        return q_values


class L2Encoder(nn.Module):
    """Price-level CNN-16 followed by a masked temporal GRU-16."""

    def __init__(
        self,
        *,
        level_features: int = L2_LEVEL_FEATURES,
        engineered_features: int = L2_ENGINEERED_FEATURES,
    ) -> None:
        super().__init__()
        self.level_features = int(level_features)
        self.engineered_features = int(engineered_features)
        self.spatial = nn.Sequential(
            nn.Conv1d(self.level_features, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.temporal = MaskedGRUEncoder(16 + self.engineered_features, 16)

    def forward(self, levels: Tensor, engineered: Tensor, mask: Tensor) -> Tensor:
        if levels.ndim != 4 or tuple(levels.shape[1:]) != (
            L2_HISTORY, L2_LEVELS, self.level_features,
        ):
            raise ValueError(
                "levels must have shape "
                f"[batch,{L2_HISTORY},{L2_LEVELS},{self.level_features}]"
            )
        if engineered.ndim != 3 or tuple(engineered.shape[1:]) != (
            L2_HISTORY, self.engineered_features,
        ):
            raise ValueError(
                f"l2_engineered must have shape [batch,{L2_HISTORY},"
                f"{self.engineered_features}]"
            )
        if mask.shape != levels.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("l2_mask must be boolean [batch,20]")
        if not bool(torch.isfinite(levels).all()) or not bool(torch.isfinite(engineered).all()):
            raise ValueError("L2 observation contains a non-finite value")
        batch = len(levels)
        spatial = self.spatial(
            levels.reshape(batch * L2_HISTORY, L2_LEVELS, self.level_features).transpose(1, 2)
        ).squeeze(-1).reshape(batch, L2_HISTORY, 16)
        return self.temporal(torch.cat((spatial, engineered), dim=-1), mask)


class MessageStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        context_features: int = CONTEXT_FEATURES,
        state_features: int = STATE_FEATURES,
    ) -> None:
        super().__init__()
        self.context_features = int(context_features)
        self.state_features = int(state_features)
        width = MESSAGE_WIDTH + self.state_features + self.context_features * 2
        self.network = nn.Sequential(nn.Linear(width, 24), nn.GELU(), nn.LayerNorm(24))

    def forward(
        self,
        message: Tensor,
        state: Tensor,
        context: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        if message.ndim != 2 or message.shape[-1] != MESSAGE_WIDTH:
            raise ValueError("message must have shape [batch,5]")
        if state.ndim != 2 or state.shape[-1] != self.state_features:
            raise ValueError(f"state must have shape [batch,{self.state_features}]")
        if context.ndim != 2 or context.shape[-1] != self.context_features:
            raise ValueError(f"context must have shape [batch,{self.context_features}]")
        if context_mask.shape != context.shape or context_mask.dtype != torch.bool:
            raise ValueError("context_mask must be boolean and align with context")
        for name, value in (("message", message), ("state", state), ("context", context)):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating point")
        valid = context_mask.to(context.dtype)
        return self.network(torch.cat((
            message, state, context * valid, valid,
        ), dim=-1))


class TraderQNetwork(nn.Module):
    """Five-minute inventory-aware trader conditioned by regime and L2."""

    action_count = ACTION_COUNT

    def __init__(
        self,
        *,
        base_features: int = BOTTOM_BASE_FEATURES,
        rich_features: int = BOTTOM_RICH_FEATURES,
        context_features: int = CONTEXT_FEATURES,
        level_features: int = L2_LEVEL_FEATURES,
        engineered_features: int = L2_ENGINEERED_FEATURES,
        state_features: int = STATE_FEATURES,
    ) -> None:
        super().__init__()
        self.base_features = int(base_features)
        self.rich_features = int(rich_features)
        self.context_features = int(context_features)
        self.level_features = int(level_features)
        self.engineered_features = int(engineered_features)
        self.state_features = int(state_features)
        self.base = MaskedGRUEncoder(self.base_features, 24)
        self.rich = MaskedGRUEncoder(self.rich_features, 24)
        self.l2 = L2Encoder(
            level_features=self.level_features, engineered_features=self.engineered_features,
        )
        self.message_state = MessageStateEncoder(
            context_features=self.context_features, state_features=self.state_features,
        )
        self.fusion = nn.Sequential(
            nn.Linear(24 + 24 + 16 + 24, 64), nn.GELU(), nn.LayerNorm(64),
        )
        self.head = DuelingHead(64)
        if parameter_count(self) > BOTTOM_PARAMETER_LIMIT:
            raise ValueError("lower V6 network exceeds its 60,000-parameter contract")

    def forward(self, observation: TensorObservation) -> Tensor:
        base = _require_tensor(
            observation, "base", rank=3, tail=(BOTTOM_HISTORY, self.base_features),
        )
        rich = _require_tensor(
            observation, "rich", rank=3, tail=(BOTTOM_HISTORY, self.rich_features),
        )
        sequence_mask = _require_tensor(
            observation, "sequence_mask", rank=2, tail=(BOTTOM_HISTORY,), boolean=True,
        )
        context = _require_tensor(
            observation, "context", rank=2, tail=(self.context_features,),
        )
        context_mask = _require_tensor(
            observation, "context_mask", rank=2, tail=(self.context_features,), boolean=True,
        )
        levels = _require_tensor(
            observation, "levels", rank=4,
            tail=(L2_HISTORY, L2_LEVELS, self.level_features),
        )
        engineered = _require_tensor(
            observation, "l2_engineered", rank=3,
            tail=(L2_HISTORY, self.engineered_features),
        )
        l2_mask = _require_tensor(
            observation, "l2_mask", rank=2, tail=(L2_HISTORY,), boolean=True,
        )
        message = _require_tensor(
            observation, "message", rank=2, tail=(MESSAGE_WIDTH,),
        )
        state = _require_tensor(
            observation, "state", rank=2, tail=(self.state_features,),
        )
        values = {
            "base": base, "rich": rich, "sequence_mask": sequence_mask,
            "context": context, "context_mask": context_mask, "levels": levels,
            "l2_engineered": engineered, "l2_mask": l2_mask,
            "message": message, "state": state,
        }
        _same_batch_device(values)
        latent = self.fusion(torch.cat((
            self.base(base, sequence_mask),
            self.rich(rich, sequence_mask),
            self.l2(levels, engineered, l2_mask),
            self.message_state(message, state, context, context_mask),
        ), dim=-1))
        q_values = self.head(latent)
        if not bool(torch.isfinite(q_values).all()):
            raise ValueError("lower network produced non-finite Q values")
        return q_values


@dataclass
class V6ModelBundle:
    regime: RegimeQNetwork
    trader: TraderQNetwork

    @classmethod
    def create(cls) -> "V6ModelBundle":
        return cls(RegimeQNetwork(), TraderQNetwork())

    def modules(self) -> tuple[nn.Module, nn.Module]:
        return self.regime, self.trader

    def to(self, device: torch.device | str) -> "V6ModelBundle":
        for module in self.modules():
            module.to(device)
        return self

    def eval(self) -> "V6ModelBundle":
        for module in self.modules():
            module.eval()
        return self

    def state_dict(self) -> dict[str, Mapping[str, Tensor]]:
        return {
            "regime": self.regime.state_dict(),
            "trader": self.trader.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Mapping[str, Tensor]]) -> None:
        if set(state) != {"regime", "trader"}:
            raise ValueError("V6 model bundle state must contain regime and trader")
        self.regime.load_state_dict(state["regime"])
        self.trader.load_state_dict(state["trader"])


@torch.no_grad()
def regime_message(
    q_values: Tensor,
    *,
    decision_time_ns: int,
    expiry_time_ns: int,
) -> RegimeMessage:
    """Convert one upper-agent Q vector to the registered semantic message."""

    if q_values.shape == (3,):
        q_values = q_values.unsqueeze(0)
    if q_values.shape != (1, ACTION_COUNT) or not bool(torch.isfinite(q_values).all()):
        raise ValueError("message construction expects one finite three-action Q vector")
    return RegimeMessage.from_q_values(
        q_values[0].cpu().numpy(), int(decision_time_ns), int(expiry_time_ns),
    )


message_from_q_values = regime_message


@torch.no_grad()
def transfer_base_encoders(source: V6ModelBundle, destination: V6ModelBundle) -> None:
    """Transfer only common OHLCV encoders from XAU pretraining into fresh GC models.

    Rich volume/L2 branches and both Q heads deliberately remain at their fresh
    GC initialization.  Replay, targets, optimizer and exploration state live
    in the trainer and therefore cannot leak through this function.
    """

    destination.regime.base.load_state_dict(source.regime.base.state_dict())
    destination.trader.base.load_state_dict(source.trader.base.state_dict())


__all__ = [
    "ACTION_COUNT", "BOTTOM_BASE_FEATURES", "BOTTOM_HISTORY",
    "BOTTOM_PARAMETER_LIMIT", "BOTTOM_RICH_FEATURES", "CONTEXT_FEATURES",
    "L2_ENGINEERED_FEATURES", "L2_HISTORY", "L2_LEVEL_FEATURES", "L2_LEVELS",
    "MESSAGE_WIDTH", "RegimeMessage", "RegimeQNetwork", "STATE_FEATURES",
    "TOP_BASE_FEATURES", "TOP_HISTORY", "TOP_PARAMETER_LIMIT", "TOP_RICH_FEATURES",
    "TraderQNetwork", "V6ModelBundle", "message_from_q_values", "parameter_count",
    "regime_message", "transfer_base_encoders",
]
