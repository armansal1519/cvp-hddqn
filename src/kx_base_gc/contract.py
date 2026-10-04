from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


ACTIONS = (-1, 0, 1)
AGENT_ORDER = ("H1", "M15", "M5")
CONTRACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentSpec:
    name: Literal["H1", "M15", "M5"]
    minutes: int
    lookback: int
    action_lock_bars: int
    gamma: float
    epsilon_decay: float
    replay_capacity: int
    hidden_sizes: tuple[int, ...]
    inherited_positions: tuple[str, ...]

    @property
    def observation_dimension(self) -> int:
        return self.lookback * 5 + len(self.inherited_positions)

    def validate(self) -> None:
        if self.name not in AGENT_ORDER:
            raise ValueError(f"unknown agent {self.name!r}")
        if self.minutes <= 0 or self.lookback <= 1 or self.action_lock_bars <= 0:
            raise ValueError(f"invalid timing contract for {self.name}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1] for {self.name}")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError(f"epsilon_decay must be in (0, 1] for {self.name}")
        if self.replay_capacity < 1 or not self.hidden_sizes:
            raise ValueError(f"invalid model capacity for {self.name}")


@dataclass(frozen=True)
class OptimizationSpec:
    learning_rate: float
    batch_size: int
    replay_warmup: int
    target_sync_updates: int
    epsilon_start: float
    epsilon_end: float
    gradient_clip_norm: float
    optimizer: Literal["rmsprop", "adam"]
    loss: Literal["mse", "huber"]
    target_rule: Literal["vanilla", "double"]
    network_head: Literal["plain", "dueling"]
    activation: Literal["tanh", "relu"]

    def validate(self) -> None:
        if self.learning_rate <= 0 or self.batch_size < 1:
            raise ValueError("invalid optimizer scale")
        if self.replay_warmup < self.batch_size:
            raise ValueError("replay_warmup must be at least batch_size")
        if self.target_sync_updates < 1:
            raise ValueError("target_sync_updates must be positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon bounds must satisfy 0 <= end <= start <= 1")


@dataclass(frozen=True)
class ExecutionSpec:
    decision_latency_ms: float = 0.0
    route_latency_ms: float = 5.0
    fee_usd_per_filled_side: float = 3.0
    tick_value_usd: float = 10.0
    quantity: int = 1
    force_flat_at_boundary: bool = True

    def validate(self) -> None:
        if self.decision_latency_ms < 0 or self.route_latency_ms < 0:
            raise ValueError("latencies cannot be negative")
        if self.fee_usd_per_filled_side < 0 or self.tick_value_usd <= 0:
            raise ValueError("invalid execution economics")
        if self.quantity != 1:
            raise ValueError("Base-GC v1 is registered for exactly one contract")


@dataclass(frozen=True)
class BaselineContract:
    schema_version: int
    contract_id: str
    variant: Literal["base_paper_gc", "base_strong_gc"]
    claim_scope: str
    agents: tuple[AgentSpec, ...]
    optimization: OptimizationSpec
    execution: ExecutionSpec
    action_values: tuple[int, int, int]
    bar_source: str
    bar_availability_rule: str
    normalization: str
    reward: str
    seed_ensemble: tuple[int, ...]
    ensemble_rule: str
    registered_defaults: tuple[str, ...]
    paper_supplied_fields: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported Base-GC contract schema")
        if tuple(agent.name for agent in self.agents) != AGENT_ORDER:
            raise ValueError(f"agents must be ordered {AGENT_ORDER}")
        for agent in self.agents:
            agent.validate()
        self.optimization.validate()
        self.execution.validate()
        if self.action_values != ACTIONS:
            raise ValueError(f"actions must be {ACTIONS}")
        if len(self.seed_ensemble) != 5 or len(set(self.seed_ensemble)) != 5:
            raise ValueError("the registered ensemble requires five distinct seeds")
        if self.ensemble_rule != "majority_flat_on_any_tie_v1":
            raise ValueError("unregistered ensemble rule")
        if not self.registered_defaults:
            raise ValueError("under-specified registered defaults must be explicit")

    def agent(self, name: str) -> AgentSpec:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return contract_sha256(self)


_AGENTS = (
    AgentSpec("H1", 60, 60, 5, 0.45, 0.999, 10_000, (300, 150), ()),
    AgentSpec("M15", 15, 200, 10, 0.50, 0.9999, 40_000, (300, 200, 100), ("H1",)),
    AgentSpec("M5", 5, 300, 20, 0.60, 0.99995, 120_000, (400, 200, 100), ("H1", "M15")),
)

_PAPER_FIELDS = (
    "H1/M15/M5 hierarchy and top-down position inputs",
    "OHLCV observations",
    "actions {-1,0,1}",
    "lookbacks 60/200/300",
    "action locks 5/10/20",
    "discounts 0.45/0.50/0.60",
    "epsilon decays 0.999/0.9999/0.99995",
    "replay capacities 10000/40000/120000",
    "hidden layers 300-150, 300-200-100, 400-200-100",
    "tanh activation and learning rate 0.00025",
    "gross horizon-return reward",
)

_REGISTERED_DEFAULTS = (
    "causal_return_volume_v1 normalization",
    "batch size 64",
    "replay warm-up 512",
    "target synchronization every 1000 updates",
    "epsilon starts at 1.0 and floors at 0.05",
    "gradient clipping at norm 10",
    "RMSProp and MSE for BasePaper-GC",
    "five fixed seeds 1103/2207/3301/4409/5501",
    "majority ensemble with flat on every tie",
    "completed-bar availability is max(bar boundary, component receives)",
    "one-contract aggressive execution and common forced liquidation",
)


def registered_contract(variant: str = "base_paper_gc") -> BaselineContract:
    """Return a fully specified, immutable reconstruction contract.

    ``base_paper_gc`` preserves published architectural choices and fills only
    unpublished operational details. ``base_strong_gc`` changes the registered
    optimizer/head/reward, but not observations, actions, hierarchy, clocks, or
    execution opportunity set.
    """

    if variant == "base_paper_gc":
        optimization = OptimizationSpec(
            learning_rate=0.00025,
            batch_size=64,
            replay_warmup=512,
            target_sync_updates=1_000,
            epsilon_start=1.0,
            epsilon_end=0.05,
            gradient_clip_norm=10.0,
            optimizer="rmsprop",
            loss="mse",
            target_rule="vanilla",
            network_head="plain",
            activation="tanh",
        )
        reward = "paper_gross_fractional_horizon_return_v1"
        defaults = _REGISTERED_DEFAULTS
    elif variant == "base_strong_gc":
        optimization = OptimizationSpec(
            learning_rate=0.00025,
            batch_size=64,
            replay_warmup=512,
            target_sync_updates=1_000,
            epsilon_start=1.0,
            epsilon_end=0.05,
            gradient_clip_norm=10.0,
            optimizer="adam",
            loss="huber",
            target_rule="double",
            network_head="dueling",
            activation="relu",
        )
        reward = "arrival_fill_spread_fee_aware_training_net_usd_v2"
        defaults = _REGISTERED_DEFAULTS + (
            "Double-DQN targets, dueling head, ReLU, Adam, and Huber loss",
            "net-dollar training reward requires causal arrival-aligned aggressive bid/ask fills",
            "one reward sample is opened only at each action-lock renewal",
            "training reward is not executable fill-ledger evidence",
        )
    else:
        raise ValueError(f"unknown Base-GC variant {variant!r}")

    contract = BaselineContract(
        schema_version=CONTRACT_SCHEMA_VERSION,
        contract_id=f"{variant}_structural_v1",
        variant=variant,  # type: ignore[arg-type]
        claim_scope=(
            "Registered structural GC reconstruction; not an exact numerical "
            "replication of Shavandi and Khedmati (2022)."
        ),
        agents=_AGENTS,
        optimization=optimization,
        execution=ExecutionSpec(),
        action_values=ACTIONS,
        bar_source="audited_active_contract_trade_bars_v2",
        bar_availability_rule="max_completed_boundary_and_component_receive_v1",
        normalization="causal_return_volume_v1",
        reward=reward,
        seed_ensemble=(1103, 2207, 3301, 4409, 5501),
        ensemble_rule="majority_flat_on_any_tie_v1",
        registered_defaults=defaults,
        paper_supplied_fields=_PAPER_FIELDS,
    )
    contract.validate()
    return contract


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def contract_sha256(contract: BaselineContract) -> str:
    contract.validate()
    payload = json.dumps(
        _jsonable(contract.to_dict()), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("expected a sequence")
    return tuple(value)


def contract_from_mapping(value: Mapping[str, Any]) -> BaselineContract:
    agents = tuple(
        AgentSpec(
            name=str(item["name"]),  # type: ignore[arg-type]
            minutes=int(item["minutes"]),
            lookback=int(item["lookback"]),
            action_lock_bars=int(item["action_lock_bars"]),
            gamma=float(item["gamma"]),
            epsilon_decay=float(item["epsilon_decay"]),
            replay_capacity=int(item["replay_capacity"]),
            hidden_sizes=tuple(int(v) for v in _tuple(item["hidden_sizes"])),
            inherited_positions=tuple(str(v) for v in item.get("inherited_positions", ())),
        )
        for item in _tuple(value["agents"])
    )
    optimization = OptimizationSpec(**dict(value["optimization"]))
    execution = ExecutionSpec(**dict(value["execution"]))
    contract = BaselineContract(
        schema_version=int(value["schema_version"]),
        contract_id=str(value["contract_id"]),
        variant=str(value["variant"]),  # type: ignore[arg-type]
        claim_scope=str(value["claim_scope"]),
        agents=agents,
        optimization=optimization,
        execution=execution,
        action_values=tuple(int(v) for v in value["action_values"]),  # type: ignore[arg-type]
        bar_source=str(value["bar_source"]),
        bar_availability_rule=str(value["bar_availability_rule"]),
        normalization=str(value["normalization"]),
        reward=str(value["reward"]),
        seed_ensemble=tuple(int(v) for v in value["seed_ensemble"]),
        ensemble_rule=str(value["ensemble_rule"]),
        registered_defaults=tuple(str(v) for v in value["registered_defaults"]),
        paper_supplied_fields=tuple(str(v) for v in value["paper_supplied_fields"]),
    )
    contract.validate()
    return contract


def load_contract_yaml(path: str | Path) -> tuple[BaselineContract, BaselineContract]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Base-GC config must be a mapping")
    variants = payload.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("Base-GC config must contain variants")
    paper = contract_from_mapping(variants["base_paper_gc"])
    strong = contract_from_mapping(variants["base_strong_gc"])
    return paper, strong
