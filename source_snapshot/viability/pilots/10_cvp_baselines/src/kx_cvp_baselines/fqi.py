"""Development-only LightGBM fitted-Q baseline for Pilot 10.

This module has no dataset discovery or evaluation entry point.  A caller must
feed it already-causal trader observations and exact economic transitions from
the common replay.  The registered corpus is produced by a seed-specific legal
random policy over the 331 development sessions, in chronological order, for
three passes.  May 2026 and later dates are rejected at the row boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from kx_simple_hmadrl.volume_v6_contract import (
    ACTIONS,
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
)

from .contract import (
    DEVELOPMENT_BOUNDS,
    FLAT_ENV_INTERACTIONS_PER_SEED,
    REGISTERED_PASSES,
    SEEDS,
    BaselineArm,
)


ARM = BaselineArm.LIGHTGBM_FITTED_Q.value
REGISTERED_SESSIONS = 331
CORPUS_PASSES = REGISTERED_PASSES
TRANSITIONS_PER_PASS = 19_242
TRANSITIONS_PER_SEED = FLAT_ENV_INTERACTIONS_PER_SEED
FQI_ITERATIONS = 10
GAMMA = 1.0

FLATTEN_ORDER = (
    "base",
    "rich",
    "sequence_mask",
    "context",
    "context_mask",
    "levels",
    "l2_engineered",
    "l2_mask",
    "message",
    "state",
)
TRADER_ARRAY_SHAPES: dict[str, tuple[int, ...]] = {
    "base": (BOTTOM_HISTORY, BOTTOM_BASE_WIDTH),
    "rich": (BOTTOM_HISTORY, BOTTOM_RICH_WIDTH),
    "sequence_mask": (BOTTOM_HISTORY,),
    "context": (CONTEXT_WIDTH,),
    "context_mask": (CONTEXT_WIDTH,),
    "levels": (L2_HISTORY, L2_LEVELS, L2_LEVEL_WIDTH),
    "l2_engineered": (L2_HISTORY, L2_ENGINEERED_WIDTH),
    "l2_mask": (L2_HISTORY,),
    "message": (MESSAGE_WIDTH,),
    "state": (STATE_WIDTH,),
}
MASK_FIELDS = frozenset({"sequence_mask", "context_mask", "l2_mask"})
FLAT_STATE_WIDTH = sum(int(np.prod(TRADER_ARRAY_SHAPES[name])) for name in FLATTEN_ORDER)
MESSAGE_OFFSET = sum(
    int(np.prod(TRADER_ARRAY_SHAPES[name]))
    for name in FLATTEN_ORDER[: FLATTEN_ORDER.index("message")]
)
MESSAGE_SLICE = slice(MESSAGE_OFFSET, MESSAGE_OFFSET + MESSAGE_WIDTH)


class PredictModel(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _development_day(value: object) -> str:
    day = str(value)
    try:
        parsed = date.fromisoformat(day)
    except ValueError as error:
        raise ValueError(f"invalid ISO session date {day!r}") from error
    if parsed.isoformat() != day:
        raise ValueError(f"session date is not canonical ISO text: {day!r}")
    start, end_exclusive = DEVELOPMENT_BOUNDS
    if not start <= day < end_exclusive:
        raise PermissionError(
            f"{day} is outside Pilot 10 development; evaluation access is forbidden"
        )
    return day


def flatten_trader_state(
    observation: Mapping[str, np.ndarray], *, strict_keys: bool = True,
) -> np.ndarray:
    """Flatten the complete causal trader observation in a frozen field order.

    The publisher message is required to be exact float32 zero.  This makes the
    baseline a single-agent action-value model rather than a consumer of the
    CVP publisher.  Simulator-only arrival fields and shape drift fail closed.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("trader observation must be a mapping of named arrays")
    keys = set(observation)
    missing = sorted(set(FLATTEN_ORDER).difference(keys))
    if missing:
        raise KeyError(f"trader observation lacks fields: {missing}")
    forbidden = sorted(name for name in keys if str(name).startswith("arrival_"))
    if forbidden:
        raise ValueError(f"simulator-only fields entered FQI state: {forbidden}")
    extra = sorted(keys.difference(FLATTEN_ORDER))
    if strict_keys and extra:
        raise KeyError(f"unregistered trader observation fields: {extra}")

    flattened: list[np.ndarray] = []
    for name in FLATTEN_ORDER:
        raw = np.asarray(observation[name])
        expected = TRADER_ARRAY_SHAPES[name]
        if raw.shape != expected:
            raise ValueError(f"trader field {name!r} must have shape {expected}")
        if name in MASK_FIELDS:
            if raw.dtype != np.bool_:
                raise TypeError(f"trader mask {name!r} must be boolean")
        else:
            if raw.dtype.kind not in "fiu":
                raise TypeError(f"trader field {name!r} must be numeric")
            if not np.isfinite(raw).all():
                raise ValueError(f"trader field {name!r} contains non-finite values")
        if name == "message":
            if raw.dtype != np.float32:
                raise TypeError("FQI publisher message must be exact float32")
            if not np.array_equal(raw, np.zeros(MESSAGE_WIDTH, dtype=np.float32)):
                raise ValueError("FQI publisher message must be exact zero")
        flattened.append(np.asarray(raw, dtype=np.float32).reshape(-1))

    result = np.ascontiguousarray(np.concatenate(flattened), dtype=np.float32)
    if result.shape != (FLAT_STATE_WIDTH,) or not np.isfinite(result).all():
        raise AssertionError("flattened FQI trader-state contract changed")
    if not np.array_equal(result[MESSAGE_SLICE], np.zeros(MESSAGE_WIDTH, np.float32)):
        raise AssertionError("flattened FQI state contains a nonzero publisher message")
    return result


def coerce_flat_states(
    value: Mapping[str, np.ndarray] | np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return a finite ``[rows, FLAT_STATE_WIDTH]`` float32 matrix."""

    if isinstance(value, Mapping):
        result = flatten_trader_state(value)[None, :]
    else:
        result = np.asarray(value, dtype=np.float32)
        if result.ndim == 1:
            result = result[None, :]
    if result.ndim != 2 or result.shape[1] != FLAT_STATE_WIDTH:
        raise ValueError(f"FQI state matrix must have width {FLAT_STATE_WIDTH}")
    if not np.isfinite(result).all():
        raise ValueError("FQI state matrix contains non-finite values")
    if np.any(result[:, MESSAGE_SLICE] != 0.0):
        raise ValueError("FQI state matrix contains a nonzero publisher message")
    return np.ascontiguousarray(result, dtype=np.float32)


def _legal_mask(value: object, *, allow_empty: bool = False) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    if mask.shape != (len(ACTIONS),):
        raise ValueError("legal-action mask must have SHORT/FLAT/LONG width")
    if not allow_empty and not bool(mask.any()):
        raise ValueError("legal-action mask cannot be empty")
    return mask.copy()


class LegalRandomPolicy:
    """Seed-specific uniform behavior policy over the current legal targets."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.decisions = 0

    def act(
        self,
        observation: Mapping[str, np.ndarray] | np.ndarray | None = None,
        *,
        legal: object,
    ) -> int:
        # The behavior distribution intentionally depends only on the legal
        # action set.  Observation acceptance preserves the common replay API.
        del observation
        mask = _legal_mask(legal)
        choices = np.flatnonzero(mask)
        index = int(choices[int(self.rng.integers(len(choices)))])
        self.decisions += 1
        return int(ACTIONS[index])

    def audit_record(self) -> dict[str, object]:
        state = self.rng.bit_generator.state
        return {
            "policy": "uniform_legal_random",
            "seed": self.seed,
            "decisions": self.decisions,
            "rng_state_sha256": canonical_json_sha256(state),
        }


@dataclass(frozen=True, slots=True)
class FQICorpus:
    seed: int
    training_dates: tuple[str, ...]
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminals: np.ndarray
    legal: np.ndarray
    next_legal: np.ndarray
    session_dates: np.ndarray
    pass_indexes: np.ndarray
    decision_ordinals: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))
        dates = tuple(_development_day(day) for day in self.training_dates)
        if not dates or dates != tuple(sorted(set(dates))):
            raise ValueError("FQI training dates must be nonempty, unique, and sorted")
        object.__setattr__(self, "training_dates", dates)

        specifications: dict[str, tuple[np.dtype[Any], int]] = {
            "states": (np.dtype(np.float32), 2),
            "actions": (np.dtype(np.int8), 1),
            "rewards": (np.dtype(np.float64), 1),
            "next_states": (np.dtype(np.float32), 2),
            "terminals": (np.dtype(np.bool_), 1),
            "legal": (np.dtype(np.bool_), 2),
            "next_legal": (np.dtype(np.bool_), 2),
            "session_dates": (np.dtype("U10"), 1),
            "pass_indexes": (np.dtype(np.int8), 1),
            "decision_ordinals": (np.dtype(np.int16), 1),
        }
        for name, (dtype, dimensions) in specifications.items():
            source = np.asarray(getattr(self, name))
            if name in {"terminals", "legal", "next_legal"}:
                if source.dtype != np.bool_:
                    raise TypeError(f"FQI corpus field {name!r} must be boolean")
            elif name in {"actions", "pass_indexes", "decision_ordinals"}:
                if source.dtype.kind not in "iu" or source.dtype == np.bool_:
                    raise TypeError(f"FQI corpus field {name!r} must be integral")
            elif name in {"states", "rewards", "next_states"}:
                if source.dtype.kind not in "fiu" or source.dtype == np.bool_:
                    raise TypeError(f"FQI corpus field {name!r} must be numeric")
            array = np.ascontiguousarray(source, dtype=dtype)
            if array.ndim != dimensions:
                raise ValueError(f"FQI corpus field {name!r} has wrong rank")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        self.validate()

    @property
    def transitions(self) -> int:
        return int(len(self.actions))

    def validate(self) -> None:
        rows = self.transitions
        vectors = (self.actions, self.rewards, self.terminals, self.session_dates,
                   self.pass_indexes, self.decision_ordinals)
        if rows < 1 or any(len(value) != rows for value in vectors):
            raise ValueError("FQI corpus columns do not share a nonzero row count")
        if self.states.shape != (rows, FLAT_STATE_WIDTH):
            raise ValueError("FQI current-state matrix has wrong shape")
        if self.next_states.shape != self.states.shape:
            raise ValueError("FQI next-state matrix has wrong shape")
        if self.legal.shape != (rows, len(ACTIONS)) or self.next_legal.shape != self.legal.shape:
            raise ValueError("FQI legal masks have wrong shape")
        if not (np.isfinite(self.states).all() and np.isfinite(self.next_states).all()
                and np.isfinite(self.rewards).all()):
            raise ValueError("FQI corpus contains non-finite numeric values")
        if np.any(self.states[:, MESSAGE_SLICE] != 0.0) or np.any(
            self.next_states[:, MESSAGE_SLICE] != 0.0
        ):
            raise ValueError("FQI corpus contains a nonzero publisher message")
        if not np.isin(self.actions, ACTIONS).all():
            raise ValueError("FQI corpus contains an unregistered action")
        if np.any(~self.legal.any(axis=1)):
            raise ValueError("FQI corpus contains an empty behavior legal mask")
        action_indexes = self.actions.astype(np.int16) + 1
        if np.any(~self.legal[np.arange(rows), action_indexes]):
            raise ValueError("FQI corpus contains an illegal behavior action")
        if np.any(self.next_legal[self.terminals]):
            raise ValueError("terminal FQI rows must have zero continuation mask")
        if np.any(~self.next_legal[~self.terminals].any(axis=1)):
            raise ValueError("nonterminal FQI rows need a legal continuation")
        if np.any(self.pass_indexes < 1) or np.any(self.decision_ordinals < 0):
            raise ValueError("FQI corpus pass/decision indexes are invalid")
        observed_days = tuple(str(day) for day in self.session_dates)
        if any(_development_day(day) not in self.training_dates for day in observed_days):
            raise ValueError("FQI corpus row is outside its training-date manifest")

    def validate_registered(self) -> None:
        """Enforce the frozen 331-session, three-pass, 57,726-row contract."""

        self.validate()
        if self.seed not in SEEDS:
            raise ValueError("FQI corpus seed is outside the five registered seeds")
        if len(self.training_dates) != REGISTERED_SESSIONS:
            raise ValueError("FQI corpus must bind exactly 331 development sessions")
        if self.transitions != TRANSITIONS_PER_SEED:
            raise ValueError("FQI corpus must contain exactly 57,726 transitions")
        if set(int(value) for value in np.unique(self.pass_indexes)) != set(
            range(1, CORPUS_PASSES + 1)
        ):
            raise ValueError("FQI corpus must contain exactly passes 1, 2, and 3")

        reference_support: tuple[tuple[str, int], ...] | None = None
        for pass_index in range(1, CORPUS_PASSES + 1):
            selected = np.flatnonzero(self.pass_indexes == pass_index)
            if len(selected) != TRANSITIONS_PER_PASS:
                raise ValueError("each FQI pass must contain exactly 19,242 transitions")
            pairs = tuple(
                (str(self.session_dates[row]), int(self.decision_ordinals[row]))
                for row in selected
            )
            if len(set(pairs)) != len(pairs):
                raise ValueError("FQI pass repeats a session decision")
            if pairs != tuple(sorted(pairs)):
                raise ValueError("FQI pass is not chronological by session and decision")
            ordered_days = tuple(dict.fromkeys(day for day, _ in pairs))
            if ordered_days != self.training_dates:
                raise ValueError("FQI pass does not traverse all 331 sessions in order")
            if reference_support is None:
                reference_support = pairs
            elif pairs != reference_support:
                raise ValueError("FQI passes do not share identical decision support")

            for day in self.training_dates:
                day_rows = selected[self.session_dates[selected] == day]
                ordinals = self.decision_ordinals[day_rows]
                if not np.array_equal(ordinals, np.arange(len(ordinals), dtype=ordinals.dtype)):
                    raise ValueError("FQI session decision ordinals are not contiguous")
                terminal = self.terminals[day_rows]
                if terminal.sum() != 1 or not bool(terminal[-1]):
                    raise ValueError("FQI session must have one final terminal transition")

    def canonical_sha256(self) -> str:
        payload = {
            "schema": "cvp_lightgbm_fqi_corpus_v1",
            "seed": self.seed,
            "training_dates": list(self.training_dates),
            "flatten_order": list(FLATTEN_ORDER),
            "flat_state_width": FLAT_STATE_WIDTH,
            "arrays": {
                name: array_sha256(np.asarray(getattr(self, name)))
                for name in (
                    "states", "actions", "rewards", "next_states", "terminals",
                    "legal", "next_legal", "session_dates", "pass_indexes",
                    "decision_ordinals",
                )
            },
        }
        return canonical_json_sha256(payload)

    def audit_record(self) -> dict[str, object]:
        return {
            "schema": "cvp_lightgbm_fqi_corpus_v1",
            "arm": ARM,
            "seed": self.seed,
            "sessions": len(self.training_dates),
            "passes": int(len(np.unique(self.pass_indexes))),
            "transitions": self.transitions,
            "state_width": FLAT_STATE_WIDTH,
            "action_counts": {
                str(action): int(np.count_nonzero(self.actions == action))
                for action in ACTIONS
            },
            "terminal_transitions": int(self.terminals.sum()),
            "training_start": self.training_dates[0],
            "training_end": self.training_dates[-1],
            "evaluation_accessed": False,
            "canonical_sha256": self.canonical_sha256(),
        }


class FQICorpusCollector:
    """Append-only adapter used by the common development replay."""

    def __init__(self, *, seed: int, training_dates: Sequence[str]) -> None:
        self.seed = int(seed)
        self.training_dates = tuple(_development_day(day) for day in training_dates)
        if self.training_dates != tuple(sorted(set(self.training_dates))):
            raise ValueError("collector training dates must be unique and sorted")
        self.behavior_policy = LegalRandomPolicy(self.seed)
        self._rows: dict[str, list[object]] = {
            name: [] for name in (
                "states", "actions", "rewards", "next_states", "terminals",
                "legal", "next_legal", "session_dates", "pass_indexes",
                "decision_ordinals",
            )
        }

    def act(
        self, observation: Mapping[str, np.ndarray] | np.ndarray | None = None,
        *, legal: object,
    ) -> int:
        return self.behavior_policy.act(observation, legal=legal)

    def append(
        self,
        observation: Mapping[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: Mapping[str, np.ndarray],
        *,
        terminal: bool,
        legal: object,
        next_legal: object,
        session_date: str,
        pass_index: int,
        decision_ordinal: int,
    ) -> None:
        day = _development_day(session_date)
        if day not in self.training_dates:
            raise ValueError("collector row date is absent from its frozen manifest")
        action_value = int(action)
        if action_value not in ACTIONS:
            raise ValueError("collector action is unregistered")
        current_mask = _legal_mask(legal)
        if not bool(current_mask[ACTIONS.index(action_value)]):
            raise ValueError("collector refuses an illegal behavior action")
        following_mask = _legal_mask(next_legal, allow_empty=bool(terminal))
        if bool(terminal) and bool(following_mask.any()):
            raise ValueError("terminal collector row must have zero next-legal mask")
        if int(pass_index) < 1 or int(decision_ordinal) < 0:
            raise ValueError("collector pass/decision index is invalid")
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("collector reward must be finite")

        values: dict[str, object] = {
            "states": flatten_trader_state(observation),
            "actions": action_value,
            "rewards": reward_value,
            "next_states": flatten_trader_state(next_observation),
            "terminals": bool(terminal),
            "legal": current_mask,
            "next_legal": following_mask,
            "session_dates": day,
            "pass_indexes": int(pass_index),
            "decision_ordinals": int(decision_ordinal),
        }
        for name, value in values.items():
            self._rows[name].append(value)

    record_transition = append

    def finalize(self, *, require_registered: bool = True) -> FQICorpus:
        if not self._rows["actions"]:
            raise ValueError("cannot finalize an empty FQI corpus")
        corpus = FQICorpus(
            seed=self.seed,
            training_dates=self.training_dates,
            **{name: np.asarray(value) for name, value in self._rows.items()},
        )
        if require_registered:
            corpus.validate_registered()
        return corpus


def action_conditioned_features(
    states: Mapping[str, np.ndarray] | np.ndarray | Sequence[float],
    actions: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Append SHORT/FLAT/LONG one-hot indicators to flattened states."""

    matrix = coerce_flat_states(states)
    selected = np.asarray(actions, dtype=np.int8)
    if selected.ndim == 0:
        selected = selected[None]
    if selected.shape != (len(matrix),) or not np.isin(selected, ACTIONS).all():
        raise ValueError("action-conditioned rows need one registered action each")
    one_hot = np.zeros((len(matrix), len(ACTIONS)), dtype=np.float32)
    one_hot[np.arange(len(matrix)), selected.astype(np.int16) + 1] = 1.0
    return np.ascontiguousarray(np.concatenate((matrix, one_hot), axis=1), dtype=np.float32)


def predict_q_matrix(model: PredictModel, states: np.ndarray | Sequence[float]) -> np.ndarray:
    matrix = coerce_flat_states(states)
    result = np.empty((len(matrix), len(ACTIONS)), dtype=np.float64)
    for column, action in enumerate(ACTIONS):
        conditioned = action_conditioned_features(
            matrix, np.full(len(matrix), action, dtype=np.int8),
        )
        predicted = np.asarray(model.predict(conditioned), dtype=np.float64)
        if predicted.shape != (len(matrix),) or not np.isfinite(predicted).all():
            raise ValueError("FQI model returned invalid action values")
        result[:, column] = predicted
    return result


def fitted_q_targets(
    corpus: FQICorpus,
    previous_model: PredictModel | None,
    *,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Compute one legal-mask-aware fitted-Q target vector."""

    if not np.isfinite(gamma) or float(gamma) != GAMMA:
        raise ValueError("registered fitted-Q discount must equal one")
    target = corpus.rewards.astype(np.float64, copy=True)
    if previous_model is None:
        return target
    values = predict_q_matrix(previous_model, corpus.next_states)
    nonterminal = ~corpus.terminals
    masked = np.where(corpus.next_legal[nonterminal], values[nonterminal], -np.inf)
    continuation = np.max(masked, axis=1)
    if not np.isfinite(continuation).all():
        raise ValueError("nonterminal fitted-Q row has no finite legal continuation")
    target[nonterminal] += float(gamma) * continuation
    if not np.isfinite(target).all():
        raise ValueError("fitted-Q target contains non-finite values")
    return target


@dataclass(frozen=True, slots=True)
class FQIConfig:
    iterations: int = FQI_ITERATIONS
    gamma: float = GAMMA
    objective: str = "huber"
    trees: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 100
    row_fraction: float = 0.8
    column_fraction: float = 0.8
    deterministic: bool = True
    force_col_wise: bool = True
    n_jobs: int = 1

    def assert_registered(self) -> None:
        expected = FQIConfig()
        if self != expected:
            raise ValueError("fitted-Q configuration differs from the frozen registry")

    def model_parameters(self, seed: int) -> dict[str, object]:
        self.assert_registered()
        return {
            "objective": self.objective,
            "n_estimators": self.trees,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.row_fraction,
            "subsample_freq": 1,
            "colsample_bytree": self.column_fraction,
            "random_state": int(seed),
            "n_jobs": self.n_jobs,
            "deterministic": self.deterministic,
            "force_col_wise": self.force_col_wise,
            "verbosity": -1,
        }

    def canonical_sha256(self) -> str:
        return canonical_json_sha256(asdict(self))


def registered_fqi_settings() -> dict[str, object]:
    """Return the complete result-independent algorithm record for receipts."""

    config = FQIConfig()
    return {
        "arm": ARM,
        "algorithm": "action_conditioned_fitted_q_iteration",
        "state": {
            "flatten_order": list(FLATTEN_ORDER),
            "flat_width": FLAT_STATE_WIDTH,
            "action_encoding": "SHORT_FLAT_LONG_one_hot",
            "message_mode": "exact_float32_zero",
        },
        "behavior_policy": "uniform_over_current_legal_targets",
        "environment_interactions_per_seed": TRANSITIONS_PER_SEED,
        "sessions": REGISTERED_SESSIONS,
        "chronological_passes": CORPUS_PASSES,
        "transitions_per_pass": TRANSITIONS_PER_PASS,
        "iterations": config.iterations,
        "target": "reward_plus_max_legal_previous_Q",
        "gamma": config.gamma,
        "terminal_continuation": 0.0,
        "reward_clipping": False,
        "model_parameters": config.model_parameters(0) | {
            "random_state": "seed",
        },
        "deterministic_inference": "legal_masked_argmax",
        "archive_hashes": [
            "corpus_file_and_canonical",
            "iteration_target_model_and_chain",
            "final_model",
        ],
        "evaluation_accessed": False,
    }


def _new_regressor(config: FQIConfig, seed: int) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "LightGBM is required for the Pilot 10 fitted-Q baseline"
        ) from error
    return LGBMRegressor(**config.model_parameters(seed))


def _model_text(model: object) -> str:
    booster = getattr(model, "booster_", model)
    serializer = getattr(booster, "model_to_string", None)
    if not callable(serializer):
        raise TypeError("FQI model does not expose deterministic LightGBM text")
    text = str(serializer(num_iteration=-1))
    if not text:
        raise ValueError("FQI LightGBM serialization is empty")
    return text.replace("\r\n", "\n")


def model_sha256(model: object) -> str:
    return hashlib.sha256(_model_text(model).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FQIIterationRecord:
    iteration: int
    target_sha256: str
    model_sha256: str
    previous_model_sha256: str | None
    chain_sha256: str


@dataclass(frozen=True, slots=True)
class FittedQPolicy:
    model: PredictModel
    seed: int
    model_sha256: str

    def q_values(
        self, observation: Mapping[str, np.ndarray] | np.ndarray | Sequence[float],
    ) -> np.ndarray:
        values = predict_q_matrix(self.model, coerce_flat_states(observation))
        if len(values) != 1:
            raise ValueError("single-policy q_values requires one observation")
        return values[0]

    def act(
        self,
        observation: Mapping[str, np.ndarray] | np.ndarray | Sequence[float],
        *,
        legal: object,
    ) -> int:
        mask = _legal_mask(legal)
        values = self.q_values(observation)
        values[~mask] = -np.inf
        return int(ACTIONS[int(np.argmax(values))])


@dataclass(frozen=True, slots=True)
class FQITrainingResult:
    policy: FittedQPolicy
    corpus_sha256: str
    config_sha256: str
    iterations: tuple[FQIIterationRecord, ...]

    def audit_record(self) -> dict[str, object]:
        return {
            "schema": "cvp_lightgbm_fqi_training_v1",
            "arm": ARM,
            "seed": self.policy.seed,
            "algorithm": "action_conditioned_fitted_q_iteration",
            "corpus_sha256": self.corpus_sha256,
            "config_sha256": self.config_sha256,
            "final_model_sha256": self.policy.model_sha256,
            "iterations": [asdict(record) for record in self.iterations],
            "evaluation_accessed": False,
        }


def _atomic_text(path: Path, text: str) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def archive_corpus(
    corpus: FQICorpus, path: str | os.PathLike[str], *, require_registered: bool = True,
) -> dict[str, object]:
    """Write one immutable NPZ plus both file and canonical-content hashes."""

    if require_registered:
        corpus.validate_registered()
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                seed=np.asarray(corpus.seed, dtype=np.int64),
                training_dates=np.asarray(corpus.training_dates, dtype="U10"),
                states=corpus.states,
                actions=corpus.actions,
                rewards=corpus.rewards,
                next_states=corpus.next_states,
                terminals=corpus.terminals,
                legal=corpus.legal,
                next_legal=corpus.next_legal,
                session_dates=corpus.session_dates,
                pass_indexes=corpus.pass_indexes,
                decision_ordinals=corpus.decision_ordinals,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": destination.name,
        "sha256": file_sha256(destination),
        "canonical_sha256": corpus.canonical_sha256(),
        "rows": corpus.transitions,
    }


def load_corpus(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    expected_canonical_sha256: str | None = None,
    require_registered: bool = True,
) -> FQICorpus:
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256:
        raise ValueError("archived FQI corpus file hash differs")
    with np.load(source, allow_pickle=False) as values:
        corpus = FQICorpus(
            seed=int(np.asarray(values["seed"]).item()),
            training_dates=tuple(str(day) for day in values["training_dates"]),
            states=values["states"],
            actions=values["actions"],
            rewards=values["rewards"],
            next_states=values["next_states"],
            terminals=values["terminals"],
            legal=values["legal"],
            next_legal=values["next_legal"],
            session_dates=values["session_dates"],
            pass_indexes=values["pass_indexes"],
            decision_ordinals=values["decision_ordinals"],
        )
    if require_registered:
        corpus.validate_registered()
    if (
        expected_canonical_sha256 is not None
        and corpus.canonical_sha256() != expected_canonical_sha256
    ):
        raise ValueError("archived FQI corpus canonical hash differs")
    return corpus


def fit_fitted_q(
    corpus: FQICorpus,
    *,
    config: FQIConfig = FQIConfig(),
    require_registered_corpus: bool = True,
    archive_directory: str | os.PathLike[str] | None = None,
) -> FQITrainingResult:
    """Run the ten registered fitted-Q iterations and return an audited policy."""

    config.assert_registered()
    if require_registered_corpus:
        corpus.validate_registered()
    else:
        corpus.validate()
    conditioned = action_conditioned_features(corpus.states, corpus.actions)
    corpus_hash = corpus.canonical_sha256()
    config_hash = config.canonical_sha256()
    archive = None if archive_directory is None else Path(archive_directory)
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        archive_corpus(
            corpus, archive / "corpus.npz", require_registered=require_registered_corpus,
        )

    previous: PredictModel | None = None
    previous_hash: str | None = None
    records: list[FQIIterationRecord] = []
    for iteration in range(1, config.iterations + 1):
        target = fitted_q_targets(corpus, previous, gamma=config.gamma)
        target_hash = array_sha256(target)
        model = _new_regressor(config, corpus.seed)
        model.fit(conditioned, target)
        serialized = _model_text(model)
        one_model_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        chain_hash = canonical_json_sha256({
            "iteration": iteration,
            "target_sha256": target_hash,
            "model_sha256": one_model_hash,
            "previous_model_sha256": previous_hash,
            "corpus_sha256": corpus_hash,
            "config_sha256": config_hash,
        })
        record = FQIIterationRecord(
            iteration, target_hash, one_model_hash, previous_hash, chain_hash,
        )
        records.append(record)
        if archive is not None:
            path = _atomic_text(archive / f"iteration_{iteration:02d}.txt", serialized)
            if file_sha256(path) != one_model_hash:
                raise AssertionError("archived fitted-Q model hash changed")
        previous = model
        previous_hash = one_model_hash

    if previous is None or previous_hash is None:
        raise AssertionError("registered fitted-Q loop produced no model")
    result = FQITrainingResult(
        policy=FittedQPolicy(previous, corpus.seed, previous_hash),
        corpus_sha256=corpus_hash,
        config_sha256=config_hash,
        iterations=tuple(records),
    )
    if len(result.iterations) != FQI_ITERATIONS:
        raise AssertionError("registered fitted-Q iteration count changed")
    if archive is not None:
        final_text = _model_text(previous)
        final_path = _atomic_text(archive / "model.txt", final_text)
        if file_sha256(final_path) != previous_hash:
            raise AssertionError("archived final fitted-Q model hash changed")
        manifest = result.audit_record() | {
            "corpus_file_sha256": file_sha256(archive / "corpus.npz"),
            "model_file_sha256": file_sha256(final_path),
        }
        _atomic_text(
            archive / "manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        )
    return result


def load_fitted_q_policy(
    path: str | os.PathLike[str], *, seed: int, expected_sha256: str,
) -> FittedQPolicy:
    source = Path(path)
    if file_sha256(source) != expected_sha256:
        raise ValueError("archived fitted-Q model hash differs")
    try:
        from lightgbm import Booster
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("LightGBM is required to load the fitted-Q policy") from error
    model = Booster(model_file=str(source))
    if model_sha256(model) != expected_sha256:
        raise ValueError("loaded fitted-Q model canonical hash differs")
    return FittedQPolicy(model, int(seed), expected_sha256)


__all__ = [
    "ACTIONS",
    "ARM",
    "CORPUS_PASSES",
    "FLATTEN_ORDER",
    "FLAT_STATE_WIDTH",
    "FQIConfig",
    "FQICorpus",
    "FQICorpusCollector",
    "FQIIterationRecord",
    "FQITrainingResult",
    "FQI_ITERATIONS",
    "FittedQPolicy",
    "GAMMA",
    "LegalRandomPolicy",
    "MESSAGE_SLICE",
    "REGISTERED_SESSIONS",
    "TRANSITIONS_PER_PASS",
    "TRANSITIONS_PER_SEED",
    "TRADER_ARRAY_SHAPES",
    "action_conditioned_features",
    "archive_corpus",
    "array_sha256",
    "canonical_json_sha256",
    "coerce_flat_states",
    "file_sha256",
    "fit_fitted_q",
    "fitted_q_targets",
    "flatten_trader_state",
    "load_corpus",
    "load_fitted_q_policy",
    "model_sha256",
    "predict_q_matrix",
    "registered_fqi_settings",
]
