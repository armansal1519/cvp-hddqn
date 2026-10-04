"""Hash-bound closed-loop evaluation for all eight registered Pilot 10 arms.

This module has no import-time execution. Production entry points first pass
caller-controlled period/path text through Pilot 10's lexical access guard,
then load exact training and dataset bindings. Learned policies are evaluated
both as a five-vote ensemble and in five independent policy-dependent ledgers;
deterministic rules have exactly one unreplicated ledger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kx_simple_hmadrl.volume_v6_contract import ACTIONS
from kx_simple_hmadrl.volume_v6_data import SessionFeatures
from kx_simple_hmadrl.volume_v6_env import (
    V6TradingEnvironment,
    V6Transition,
    legal_target_mask,
    uses_legacy_arrival_boundary,
)
from kx_simple_hmadrl.volume_v6_eval import flat_on_vote_tie

from kx_cvp_ablations.frozen_v6 import (
    build_session_features,
    load_frozen_reference,
    load_npz_pair,
    raw_index,
    valid_feature_indexes,
)

from .artifacts import (
    atomic_json,
    atomic_parquet,
    canonical_json_sha256,
    file_sha256,
    project_relative,
    read_json_object,
    resolve_project_path,
)
from .contract import (
    BASELINE_ARMS,
    CANONICAL_OUTPUT_ROOT,
    DATASET_MANIFESTS,
    DETERMINISTIC_ARMS,
    LEARNER_ARMS,
    PERIODS,
    PERIOD_CLASSIFICATION,
    PROTOCOL_ID,
    SCENARIOS,
    SEEDS,
    BaselineArm,
    scenario_economics,
    validate_arm,
    validate_period,
    validate_scenario,
)
from .flat_ddqn import FlatMultiClockQNetwork
from .metrics import daily_metrics
from .protocol import (
    AuthorizedDataset,
    authorize_manifest_request,
    implementation_binding,
    load_authorized_dataset,
    protocol_binding,
    verify_implementation_binding,
)
from .rules import RuleSignal, rule_policy


REPLAY_ARTIFACT = "CVP_BASELINE_REPLAY_REPORT"
TECHNICAL_FAILURE_ARTIFACT = "CVP_BASELINE_TECHNICAL_FAILURE"
COMPLETE_EVALUATION_ARTIFACT = "CVP_BASELINE_COMPLETE_EVALUATION_MANIFEST"
ENSEMBLE_RULE = "five_legal_seed_votes_majority_flat_on_tie_v1"
DETERMINISTIC_RULE = "single_unreplicated_deterministic_path_v1"

PERIOD_DATASETS = {
    period: str(DATASET_MANIFESTS[period]["path"]) for period in PERIODS
}

TECHNICAL_FAILURE_REASONS = frozenset({
    "HASH_MISMATCH",
    "CAUSALITY_VIOLATION",
    "NONFINITE_STATE_OR_SCORE",
    "INCOMPLETE_TRAINING",
    "POLICY_LOAD_FAILURE",
    "REWARD_LEDGER_MISMATCH",
    "ACTION_LEGALITY_VIOLATION",
    "ROUTE_RECONCILIATION_FAILURE",
    "TERMINAL_INVENTORY_NOT_FLAT",
    "INCOMPLETE_TRACE",
    "UNEXPECTED_POLICY_FALLBACK",
})

FILL_SCHEMA = pa.schema([
    ("session_date", pa.string()),
    ("fill_ordinal", pa.int32()),
    ("timestamp_ns", pa.int64()),
    ("side", pa.int8()),
    ("price_ticks", pa.int64()),
    ("fee_cents", pa.int64()),
    ("forced", pa.bool_()),
])
SEED_FILL_SCHEMA = pa.schema([
    ("session_date", pa.string()),
    ("seed", pa.int32()),
    ("fill_ordinal", pa.int32()),
    ("timestamp_ns", pa.int64()),
    ("side", pa.int8()),
    ("price_ticks", pa.int64()),
    ("fee_cents", pa.int64()),
    ("forced", pa.bool_()),
])


@dataclass(frozen=True)
class LearnedMember:
    seed: int
    policy: object
    scaler: object
    member_manifest_path: Path
    member_manifest_sha256: str
    model_sha256: str


@dataclass(frozen=True)
class LearnedEnsemble:
    arm: str
    members: tuple[LearnedMember, ...]
    training_manifest_path: Path
    training_manifest_sha256: str
    training_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.arm not in LEARNER_ARMS:
            raise ValueError("learned ensemble has a nonlearner arm")
        if tuple(member.seed for member in self.members) != SEEDS:
            raise ValueError("learned ensemble must contain all five ordered seeds")
        if len({member.member_manifest_sha256 for member in self.members}) != len(SEEDS):
            raise ValueError("learned ensemble member manifests are duplicated")


@dataclass
class RolloutRows:
    decisions: list[dict[str, object]]
    rewards: list[dict[str, object]]
    fills: list[dict[str, object]]
    daily_pnl_cents: dict[str, int]
    completed_round_trips: int = 0
    rejected_commands: int = 0
    filled_sides: int = 0
    commissions_cents: int = 0
    per_seed_daily_pnl_cents: dict[int, dict[str, int]] | None = None
    per_seed_round_trips: dict[int, int] | None = None
    per_seed_rejections: dict[int, int] | None = None
    per_seed_filled_sides: dict[int, int] | None = None
    per_seed_commissions_cents: dict[int, int] | None = None

    @classmethod
    def empty(cls) -> "RolloutRows":
        return cls(
            [], [], [], {}, 0, 0, 0, 0,
            {seed: {} for seed in SEEDS},
            {seed: 0 for seed in SEEDS},
            {seed: 0 for seed in SEEDS},
            {seed: 0 for seed in SEEDS},
            {seed: 0 for seed in SEEDS},
        )


def _trace_record(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": path.name,
        "path_scope": "report_relative",
        "sha256": file_sha256(path),
        "rows": int(rows),
    }


def _resolve_trace(report_path: Path, record: object) -> Path:
    if not isinstance(record, Mapping) or record.get("path_scope") != "report_relative":
        raise ValueError("trace binding is malformed")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != str(relative):
        raise ValueError("trace path is not report-relative")
    path = (report_path.parent / relative).resolve()
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("trace hash differs")
    rows = pq.ParquetFile(path).metadata.num_rows
    if rows != int(record.get("rows", -1)):
        raise ValueError("trace row count differs")
    return path


def _rows(path: Path) -> list[dict[str, object]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _tensor_batch(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    import torch

    result: dict[str, object] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        dtype = torch.bool if array.dtype == np.bool_ else torch.float32
        result[name] = torch.as_tensor(array[None], dtype=dtype, device="cpu")
    return result


def _policy_scores(
    arm: str, policy: object, observation: Mapping[str, np.ndarray],
) -> np.ndarray:
    if arm == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
        import torch

        if not isinstance(policy, FlatMultiClockQNetwork):
            raise TypeError("flat DDQN member has another policy type")
        with torch.no_grad():
            value = policy(_tensor_batch(observation))[0].detach().cpu().numpy()
    elif arm == BaselineArm.RECURRENT_PPO.value:
        import torch

        with torch.no_grad():
            output = policy(_tensor_batch(observation))  # type: ignore[operator]
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("PPO member did not return actor logits and critic value")
        value = output[0][0].detach().cpu().numpy()
    elif arm == BaselineArm.LIGHTGBM_FITTED_Q.value:
        value = policy.q_values(observation)  # type: ignore[attr-defined]
    else:
        raise ValueError("score adapter received a nonlearner arm")
    scores = np.asarray(value, dtype=np.float64)
    if scores.shape != (len(ACTIONS),) or not np.isfinite(scores).all():
        raise ValueError("policy scores must be one finite SHORT/FLAT/LONG vector")
    return scores.copy()


def legal_masked_vote(scores: object, legal: object) -> int:
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(legal, dtype=np.bool_)
    if values.shape != (len(ACTIONS),) or not np.isfinite(values).all():
        raise ValueError("scores must be one finite SHORT/FLAT/LONG vector")
    if mask.shape != (len(ACTIONS),) or not bool(mask.any()):
        raise ValueError("legal mask must be one nonempty SHORT/FLAT/LONG vector")
    masked = values.copy()
    masked[~mask] = -np.inf
    return int(ACTIONS[int(np.argmax(masked))])


def _bottom_arrays(
    session: SessionFeatures, feature_index: int, scaler: object,
    state: np.ndarray,
) -> dict[str, np.ndarray]:
    # Reuse the production training constructor. A merely neutral V6 message
    # has zero advantages but validity/lifetime fields of one, whereas these
    # single-agent baselines require all five fields to be exact float32 zero.
    from .training import make_zero_message_observation

    return make_zero_message_observation(
        session,
        int(feature_index),
        scaler,  # type: ignore[arg-type]
        state,
    ).arrays("rich_volume")


def _flat_arrays(
    session: SessionFeatures, feature_index: int, scaler: object,
    state: np.ndarray,
) -> dict[str, np.ndarray]:
    from .training import aligned_flat_observation, make_zero_message_observation

    bottom = make_zero_message_observation(
        session,
        int(feature_index),
        scaler,  # type: ignore[arg-type]
        state,
    )
    return aligned_flat_observation(
        session,
        int(feature_index),
        scaler,  # type: ignore[arg-type]
        bottom,
    ).arrays()


def _learner_observation(
    arm: str, session: SessionFeatures, feature_index: int, scaler: object,
    state: np.ndarray,
) -> dict[str, np.ndarray]:
    if arm == BaselineArm.FLAT_MULTICLOCK_DDQN.value:
        return _flat_arrays(session, feature_index, scaler, state)
    bottom = _bottom_arrays(session, feature_index, scaler, state)
    if arm == BaselineArm.RECURRENT_PPO.value:
        bottom.pop("message")
    return bottom


def _route_fields(transition: V6Transition) -> dict[str, object]:
    route = transition.route
    terminal = transition.terminal_route

    def displayed(value: object) -> int | None:
        if value is None or value.book is None:  # type: ignore[attr-defined]
            return None
        return int(
            value.book.ask_size if value.side > 0 else value.book.bid_size  # type: ignore[attr-defined]
        )

    return {
        "route_deadline_ns": None if route is None else route.deadline_time_ns,
        "route_side": None if route is None else route.side,
        "route_fill_time_ns": (
            None if route is None or route.fill is None else route.fill.timestamp_ns
        ),
        "route_fill_price_ticks": (
            None if route is None or route.fill is None else route.fill.price_ticks
        ),
        "route_fill_fee_cents": (
            None if route is None or route.fill is None else route.fill.fee_cents
        ),
        "route_displayed_size": displayed(route),
        "route_searched_snapshots": None if route is None else route.searched_snapshots,
        "terminal_route_deadline_ns": (
            None if terminal is None else terminal.deadline_time_ns
        ),
        "terminal_route_side": None if terminal is None else terminal.side,
        "terminal_fill_time_ns": (
            None if terminal is None or terminal.fill is None
            else terminal.fill.timestamp_ns
        ),
        "terminal_fill_price_ticks": (
            None if terminal is None or terminal.fill is None
            else terminal.fill.price_ticks
        ),
        "terminal_fill_fee_cents": (
            None if terminal is None or terminal.fill is None
            else terminal.fill.fee_cents
        ),
        "terminal_displayed_size": displayed(terminal),
    }


def _reward_row(
    *, day: str, ordinal: int, action: int, transition: V6Transition,
    cumulative: int, seed: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "session_date": day,
        "decision_ordinal": int(ordinal),
        "decision_time_ns": transition.decision_time_ns,
        "next_time_ns": transition.next_time_ns,
        "executed_action": int(action),
        "legal_short": bool(transition.current_legal_mask[0]),
        "legal_flat": bool(transition.current_legal_mask[1]),
        "legal_long": bool(transition.current_legal_mask[2]),
        "inventory_before": transition.inventory_before,
        "inventory_after": transition.inventory_after,
        "reward_cents": transition.reward_cents,
        "cumulative_executable_nlv_cents": int(cumulative),
        "command_created": transition.command_created,
        "command_rejected": transition.command_rejected,
        "terminal": transition.done,
        **_route_fields(transition),
    }
    if seed is not None:
        result["seed"] = int(seed)
    return result


def _fill_rows(day: str, fills: Sequence[object], *, seed: int | None = None) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for ordinal, fill in enumerate(fills):
        row: dict[str, object] = {
            "session_date": day,
            "fill_ordinal": ordinal,
            "timestamp_ns": int(fill.timestamp_ns),  # type: ignore[attr-defined]
            "side": int(fill.side),  # type: ignore[attr-defined]
            "price_ticks": int(fill.price_ticks),  # type: ignore[attr-defined]
            "fee_cents": int(fill.fee_cents),  # type: ignore[attr-defined]
            "forced": bool(fill.forced),  # type: ignore[attr-defined]
        }
        if seed is not None:
            row["seed"] = int(seed)
        output.append(row)
    return output


def _record_session_result(
    target: RolloutRows, *, day: str, result: object, cumulative: int,
    seed: int | None = None,
) -> None:
    if int(cumulative) != int(result.net_pnl_cents):  # type: ignore[attr-defined]
        raise AssertionError("raw rewards do not telescope to terminal cash")
    transitions = result.transitions  # type: ignore[attr-defined]
    if not transitions or int(transitions[-1].inventory_after) != 0:
        raise AssertionError("terminal inventory is not flat")
    fills = result.fills  # type: ignore[attr-defined]
    if len(fills) % 2:
        raise AssertionError("one-contract replay has an unmatched fill")
    fees = sum(int(fill.fee_cents) for fill in fills)
    if seed is None:
        target.daily_pnl_cents[day] = int(result.net_pnl_cents)  # type: ignore[attr-defined]
        target.completed_round_trips += int(result.completed_round_trips)  # type: ignore[attr-defined]
        target.rejected_commands += int(result.rejected_commands)  # type: ignore[attr-defined]
        target.filled_sides += len(fills)
        target.commissions_cents += fees
    else:
        assert target.per_seed_daily_pnl_cents is not None
        assert target.per_seed_round_trips is not None
        assert target.per_seed_rejections is not None
        assert target.per_seed_filled_sides is not None
        assert target.per_seed_commissions_cents is not None
        target.per_seed_daily_pnl_cents[int(seed)][day] = int(result.net_pnl_cents)  # type: ignore[attr-defined]
        target.per_seed_round_trips[int(seed)] += int(result.completed_round_trips)  # type: ignore[attr-defined]
        target.per_seed_rejections[int(seed)] += int(result.rejected_commands)  # type: ignore[attr-defined]
        target.per_seed_filled_sides[int(seed)] += len(fills)
        target.per_seed_commissions_cents[int(seed)] += fees
    target.fills.extend(_fill_rows(day, fills, seed=seed))


def _replay_learned_ensemble_session(
    *, arm: str, members: tuple[LearnedMember, ...], raw: Mapping[str, np.ndarray],
    session: SessionFeatures, selected: np.ndarray, day: str, economics: object,
    output: RolloutRows,
) -> None:
    env = V6TradingEnvironment(raw, day, economics)  # type: ignore[arg-type]
    cumulative = 0
    for ordinal in range(len(selected)):
        feature_index = int(selected[ordinal])
        current = raw_index(raw, int(session.bottom_clock_ns[feature_index]))
        terminal = ordinal == len(selected) - 1
        following = (
            len(np.asarray(raw["clock_ns"])) - 1
            if terminal
            else raw_index(raw, int(session.bottom_clock_ns[int(selected[ordinal + 1])]))
        )
        state = env.agent_state(current)
        legal = env.legal_actions()
        seed_values: list[tuple[LearnedMember, np.ndarray, int, bool]] = []
        votes: list[int] = []
        for member in members:
            observation = _learner_observation(
                arm, session, feature_index, member.scaler, state,
            )
            scores = _policy_scores(arm, member.policy, observation)
            vote = legal_masked_vote(scores, legal)
            votes.append(vote)
            seed_values.append((
                member,
                scores,
                vote,
                bool(np.asarray(observation.get("top_sequence_mask", [])).any()),
            ))
        action = flat_on_vote_tie(votes)
        if action not in ACTIONS or not bool(legal[ACTIONS.index(action)]):
            raise AssertionError("five-vote ensemble produced an illegal target")
        transition = env.step(current, action, following, terminal=terminal)
        cumulative += int(transition.reward_cents)
        counts = {candidate: votes.count(candidate) for candidate in ACTIONS}
        for member, scores, vote, top_available in seed_values:
            output.decisions.append({
                "session_date": day,
                "decision_ordinal": ordinal,
                "decision_time_ns": transition.decision_time_ns,
                "next_time_ns": transition.next_time_ns,
                "seed": member.seed,
                "score_short": float(scores[0]),
                "score_flat": float(scores[1]),
                "score_long": float(scores[2]),
                "legal_short": bool(legal[0]),
                "legal_flat": bool(legal[1]),
                "legal_long": bool(legal[2]),
                "member_vote": int(vote),
                "votes_short": int(counts[-1]),
                "votes_flat": int(counts[0]),
                "votes_long": int(counts[1]),
                "executed_action": int(action),
                "inventory_before": transition.inventory_before,
                "inventory_after": transition.inventory_after,
                "reward_cents": transition.reward_cents,
                "terminal": transition.done,
                "top_available": top_available if arm == "flat_multiclock_ddqn" else None,
            })
        output.rewards.append(_reward_row(
            day=day,
            ordinal=ordinal,
            action=action,
            transition=transition,
            cumulative=cumulative,
        ))
    result = env.finish()
    _record_session_result(output, day=day, result=result, cumulative=cumulative)


def _replay_member_session(
    *, arm: str, member: LearnedMember, raw: Mapping[str, np.ndarray],
    session: SessionFeatures, selected: np.ndarray, day: str, economics: object,
    output: RolloutRows,
) -> None:
    env = V6TradingEnvironment(raw, day, economics)  # type: ignore[arg-type]
    cumulative = 0
    for ordinal in range(len(selected)):
        feature_index = int(selected[ordinal])
        current = raw_index(raw, int(session.bottom_clock_ns[feature_index]))
        terminal = ordinal == len(selected) - 1
        following = (
            len(np.asarray(raw["clock_ns"])) - 1
            if terminal
            else raw_index(raw, int(session.bottom_clock_ns[int(selected[ordinal + 1])]))
        )
        state = env.agent_state(current)
        legal = env.legal_actions()
        observation = _learner_observation(
            arm, session, feature_index, member.scaler, state,
        )
        scores = _policy_scores(arm, member.policy, observation)
        action = legal_masked_vote(scores, legal)
        transition = env.step(current, action, following, terminal=terminal)
        cumulative += int(transition.reward_cents)
        output.decisions.append({
            "session_date": day,
            "decision_ordinal": ordinal,
            "decision_time_ns": transition.decision_time_ns,
            "next_time_ns": transition.next_time_ns,
            "seed": member.seed,
            "score_short": float(scores[0]),
            "score_flat": float(scores[1]),
            "score_long": float(scores[2]),
            "legal_short": bool(legal[0]),
            "legal_flat": bool(legal[1]),
            "legal_long": bool(legal[2]),
            "member_vote": int(action),
            "executed_action": int(action),
            "inventory_before": transition.inventory_before,
            "inventory_after": transition.inventory_after,
            "reward_cents": transition.reward_cents,
            "terminal": transition.done,
            "top_available": (
                bool(np.asarray(observation.get("top_sequence_mask", [])).any())
                if arm == "flat_multiclock_ddqn" else None
            ),
        })
        output.rewards.append(_reward_row(
            day=day,
            ordinal=ordinal,
            action=action,
            transition=transition,
            cumulative=cumulative,
            seed=member.seed,
        ))
    result = env.finish()
    _record_session_result(
        output,
        day=day,
        result=result,
        cumulative=cumulative,
        seed=member.seed,
    )


def _replay_rule_session(
    *, arm: str, raw: Mapping[str, np.ndarray], session: SessionFeatures,
    selected: np.ndarray, day: str, economics: object, output: RolloutRows,
) -> None:
    policy = rule_policy(arm)
    env = V6TradingEnvironment(raw, day, economics)  # type: ignore[arg-type]
    cumulative = 0
    for ordinal in range(len(selected)):
        feature_index = int(selected[ordinal])
        current = raw_index(raw, int(session.bottom_clock_ns[feature_index]))
        terminal = ordinal == len(selected) - 1
        following = (
            len(np.asarray(raw["clock_ns"])) - 1
            if terminal
            else raw_index(raw, int(session.bottom_clock_ns[int(selected[ordinal + 1])]))
        )
        legal = env.legal_actions()
        signal = RuleSignal.from_session(session, feature_index)
        scores = policy.q_values(signal)
        desired = policy.desired_action(signal)
        action = policy.act(signal, legal=legal)
        transition = env.step(current, action, following, terminal=terminal)
        cumulative += int(transition.reward_cents)
        output.decisions.append({
            "session_date": day,
            "decision_ordinal": ordinal,
            "decision_time_ns": transition.decision_time_ns,
            "next_time_ns": transition.next_time_ns,
            "seed": -1,
            "score_short": float(scores[0]),
            "score_flat": float(scores[1]),
            "score_long": float(scores[2]),
            "legal_short": bool(legal[0]),
            "legal_flat": bool(legal[1]),
            "legal_long": bool(legal[2]),
            "member_vote": int(action),
            "desired_action": int(desired),
            "legality_fallback": bool(desired != action),
            "executed_action": int(action),
            "inventory_before": transition.inventory_before,
            "inventory_after": transition.inventory_after,
            "reward_cents": transition.reward_cents,
            "terminal": transition.done,
            "momentum_60m": signal.momentum_60m,
            "distance_session_vwap_atr": signal.distance_session_vwap_atr,
        })
        output.rewards.append(_reward_row(
            day=day,
            ordinal=ordinal,
            action=action,
            transition=transition,
            cumulative=cumulative,
        ))
    result = env.finish()
    _record_session_result(output, day=day, result=result, cumulative=cumulative)


def _scaler_digest(scaler: object) -> str:
    digest = getattr(scaler, "canonical_sha256", None)
    if not callable(digest):
        raise TypeError("registered V6 scaler lacks its canonical digest")
    value = str(digest())
    if len(value) != 64:
        raise ValueError("registered V6 scaler digest is malformed")
    return value


def _evaluate_learned(
    ensemble: LearnedEnsemble, dataset: AuthorizedDataset, *, scenario: str,
) -> tuple[RolloutRows, RolloutRows, bool]:
    scaler_hashes = {_scaler_digest(member.scaler) for member in ensemble.members}
    if len(scaler_hashes) != 1:
        raise ValueError("ensemble members use different causal transforms")
    economics = scenario_economics(scenario)
    phase = "validation" if dataset.period_id == "may_2026" else "test"
    ensemble_rows = RolloutRows.empty()
    seed_rows = RolloutRows.empty()
    legacy = False
    for raw_path, side_path in dataset.pairs:
        raw, side = load_npz_pair(raw_path, side_path, phase=phase)
        common_scaler = ensemble.members[0].scaler
        session = build_session_features(raw, side, common_scaler.calibration)
        selected = valid_feature_indexes(session, top=False)
        if not len(selected):
            raise ValueError(f"{raw_path.stem}: no valid five-minute decisions")
        legacy |= uses_legacy_arrival_boundary(raw, economics)
        _replay_learned_ensemble_session(
            arm=ensemble.arm,
            members=ensemble.members,
            raw=raw,
            session=session,
            selected=selected,
            day=raw_path.stem,
            economics=economics,
            output=ensemble_rows,
        )
        for member in ensemble.members:
            _replay_member_session(
                arm=ensemble.arm,
                member=member,
                raw=raw,
                session=session,
                selected=selected,
                day=raw_path.stem,
                economics=economics,
                output=seed_rows,
            )
    return ensemble_rows, seed_rows, legacy


RULE_SCALER_ROOT = (
    "viability/artifacts/development/"
    "v6_registered_training_terminal_eligible/rich_volume/seed_1103"
)


def _registered_rule_scaler() -> object:
    reference = load_frozen_reference(
        resolve_project_path(RULE_SCALER_ROOT),
        expected_parent_arm="rich_volume",
        expected_seed=1103,
    )
    return reference.scaler


def _evaluate_rule(
    arm: str, dataset: AuthorizedDataset, *, scenario: str,
) -> tuple[RolloutRows, bool, object]:
    scaler = _registered_rule_scaler()
    economics = scenario_economics(scenario)
    phase = "validation" if dataset.period_id == "may_2026" else "test"
    rows = RolloutRows.empty()
    legacy = False
    for raw_path, side_path in dataset.pairs:
        raw, side = load_npz_pair(raw_path, side_path, phase=phase)
        session = build_session_features(raw, side, scaler.calibration)
        selected = valid_feature_indexes(session, top=False)
        if not len(selected):
            raise ValueError(f"{raw_path.stem}: no valid five-minute decisions")
        legacy |= uses_legacy_arrival_boundary(raw, economics)
        _replay_rule_session(
            arm=arm,
            raw=raw,
            session=session,
            selected=selected,
            day=raw_path.stem,
            economics=economics,
            output=rows,
        )
    return rows, legacy, scaler


def _load_learned_ensemble(
    training_manifest: Path, arm: str,
) -> LearnedEnsemble:
    from .training import load_training_member, verify_complete_training_manifest

    path = Path(training_manifest).resolve()
    complete = verify_complete_training_manifest(path)
    records = complete.get("members")
    if not isinstance(records, list):
        raise ValueError("complete training manifest lacks member records")
    selected: dict[int, Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("arm") != arm:
            continue
        seed = int(record.get("seed", -1))
        if seed in selected:
            raise ValueError("complete training manifest duplicates an ensemble seed")
        selected[seed] = record
    if tuple(sorted(selected)) != SEEDS:
        raise ValueError("complete training manifest lacks the exact five-seed ensemble")

    members: list[LearnedMember] = []
    for seed in SEEDS:
        record = selected[seed]
        member_path = resolve_project_path(record.get("manifest"))
        if not member_path.is_file() or file_sha256(member_path) != record.get(
            "manifest_sha256"
        ):
            raise ValueError("training member manifest hash differs")
        policy, scaler, manifest = load_training_member(
            member_path, expected_arm=arm, expected_seed=seed,
        )
        model = manifest.get("model")
        if not isinstance(model, Mapping) or not isinstance(model.get("sha256"), str):
            raise ValueError("training member lacks a model hash")
        members.append(LearnedMember(
            seed=seed,
            policy=policy,
            scaler=scaler,
            member_manifest_path=member_path,
            member_manifest_sha256=str(record["manifest_sha256"]),
            model_sha256=str(model["sha256"]),
        ))
    return LearnedEnsemble(
        arm=arm,
        members=tuple(members),
        training_manifest_path=path,
        training_manifest_sha256=file_sha256(path),
        training_manifest=complete,
    )


def _write_traces(
    work: Path, *, primary: RolloutRows, seed: RolloutRows | None,
) -> dict[str, object]:
    decision = atomic_parquet(work / "decisions.parquet", primary.decisions)
    reward = atomic_parquet(work / "rewards.parquet", primary.rewards)
    fill = atomic_parquet(
        work / "fills.parquet", primary.fills, schema=FILL_SCHEMA,
    )
    result: dict[str, object] = {
        "decision_trace": _trace_record(decision, len(primary.decisions)),
        "reward_trace": _trace_record(reward, len(primary.rewards)),
        "fill_trace": _trace_record(fill, len(primary.fills)),
    }
    if seed is not None:
        seed_decision = atomic_parquet(
            work / "seed_decisions.parquet", seed.decisions,
        )
        seed_reward = atomic_parquet(work / "seed_rewards.parquet", seed.rewards)
        seed_fill = atomic_parquet(
            work / "seed_fills.parquet", seed.fills, schema=SEED_FILL_SCHEMA,
        )
        result.update({
            "seed_decision_trace": _trace_record(
                seed_decision, len(seed.decisions),
            ),
            "seed_reward_trace": _trace_record(seed_reward, len(seed.rewards)),
            "seed_fill_trace": _trace_record(seed_fill, len(seed.fills)),
        })
    else:
        result.update({
            "seed_decision_trace": None,
            "seed_reward_trace": None,
            "seed_fill_trace": None,
        })
    return result


def _checkpoint_set(ensemble: LearnedEnsemble) -> list[dict[str, object]]:
    return [
        {
            "seed": member.seed,
            "manifest": {
                "path": project_relative(member.member_manifest_path),
                "path_scope": "project_root",
                "sha256": member.member_manifest_sha256,
            },
            "model_sha256": member.model_sha256,
        }
        for member in ensemble.members
    ]


def _per_seed_report(seed_rows: RolloutRows) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, object]]]:
    assert seed_rows.per_seed_daily_pnl_cents is not None
    assert seed_rows.per_seed_round_trips is not None
    assert seed_rows.per_seed_rejections is not None
    assert seed_rows.per_seed_filled_sides is not None
    assert seed_rows.per_seed_commissions_cents is not None
    daily = {
        str(seed): dict(seed_rows.per_seed_daily_pnl_cents[seed]) for seed in SEEDS
    }
    metrics: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        one = daily_metrics(
            daily[str(seed)],
            completed_round_trips=seed_rows.per_seed_round_trips[seed],
        )
        metrics[str(seed)] = dict(one) | {
            "rejected_commands": seed_rows.per_seed_rejections[seed],
            "filled_sides": seed_rows.per_seed_filled_sides[seed],
            "commissions_cents": seed_rows.per_seed_commissions_cents[seed],
            "explicit_commissions_usd": (
                seed_rows.per_seed_commissions_cents[seed] / 100.0
            ),
        }
    return daily, metrics


def _reject_august_lexically(*values: object) -> None:
    markers = ("august", "2026-08", "2026_08", "202608", "aug_2026", "aug-2026")
    if any(
        marker in str(value).replace("\\", "/").casefold()
        for value in values for marker in markers
    ):
        raise PermissionError("August 2026 is unopened and unauthorized")


def _resolve_output_root(value: Path | str | None) -> Path:
    """Interpret relative output roots from the thesis root, never caller cwd."""

    from .protocol import validate_workspace_output

    return validate_workspace_output(
        CANONICAL_OUTPUT_ROOT if value is None else value,
    )


def replay_cell(
    *, arm: str | BaselineArm, period: str, scenario: str,
    output_root: Path | str | None = None,
    training_manifest: Path | str | None = None,
    dataset_manifest: str | None = None,
    ensemble_loader: Callable[[Path, str], LearnedEnsemble] | None = None,
) -> Path:
    """Evaluate one registered arm/period/scenario cell exactly once.

    This is the production entry point. It is deliberately job-sized and never
    catches integrity errors as economic results; callers may record only a
    predeclared technical-failure reason with :func:`write_technical_failure`.
    """

    _reject_august_lexically(
        arm, period, scenario, output_root or "", training_manifest or "",
        dataset_manifest or "",
    )
    selected_arm = validate_arm(arm).value
    selected_period = validate_period(period).value
    selected_scenario = validate_scenario(scenario).value
    relative_dataset = (
        PERIOD_DATASETS[selected_period]
        if dataset_manifest is None else str(dataset_manifest)
    )
    # Pure lexical authorization precedes output, training, contract, or data I/O.
    authorize_manifest_request(
        relative_dataset, purpose="evaluate", period_id=selected_period,
    )
    root = _resolve_output_root(output_root)
    registered_training, suite_training_binding = _suite_training_prerequisite(
        root, training_manifest,
    )
    destination = (
        root
        / "evaluation"
        / selected_period
        / selected_scenario
        / selected_arm
    )
    report_path = destination / "report.json"
    partial_path = destination.with_name(f".{destination.name}.part")
    if partial_path.exists():
        raise FileExistsError("evaluation cell has an unaudited partial artifact")
    if report_path.is_file():
        existing = read_json_object(report_path)
        if existing.get("artifact_type") == REPLAY_ARTIFACT:
            verify_replay_report(
                report_path,
                expected_arm=selected_arm,
                expected_period=selected_period,
                expected_scenario=selected_scenario,
            )
        elif existing.get("artifact_type") == TECHNICAL_FAILURE_ARTIFACT:
            verify_technical_failure_report(
                report_path,
                expected_arm=selected_arm,
                expected_period=selected_period,
                expected_scenario=selected_scenario,
            )
        else:
            raise ValueError("existing evaluation report artifact type differs")
        return report_path
    if destination.exists():
        raise FileExistsError("evaluation cell exists without a complete report")

    dataset = load_authorized_dataset(
        relative_dataset, purpose="evaluate", period_id=selected_period,
    )
    economics = scenario_economics(selected_scenario)
    ensemble: LearnedEnsemble | None = None
    training_binding: dict[str, object] | None = None
    checkpoints: list[dict[str, object]] = []
    if selected_arm in LEARNER_ARMS:
        loader = _load_learned_ensemble if ensemble_loader is None else ensemble_loader
        training_path = registered_training
        ensemble = loader(training_path, selected_arm)
        training_binding = {
            "path": project_relative(ensemble.training_manifest_path),
            "path_scope": "project_root",
            "sha256": ensemble.training_manifest_sha256,
        }
        checkpoints = _checkpoint_set(ensemble)
        primary, seed_rows, legacy = _evaluate_learned(
            ensemble, dataset, scenario=selected_scenario,
        )
        rule_scaler_binding = None
    else:
        primary, legacy, scaler = _evaluate_rule(
            selected_arm, dataset, scenario=selected_scenario,
        )
        seed_rows = None
        reference_path = resolve_project_path(RULE_SCALER_ROOT) / "manifest.json"
        rule_scaler_binding = {
            "path": project_relative(reference_path),
            "path_scope": "project_root",
            "sha256": file_sha256(reference_path),
            "scaler_canonical_sha256": _scaler_digest(scaler),
            "policy_uses_scaled_values": False,
        }
    if legacy:
        raise ValueError("registered replay encountered unsafe legacy arrival boundaries")
    if sorted(primary.daily_pnl_cents) != list(dataset.dates):
        raise AssertionError("replay did not retain the exact evaluation date set")

    work = destination.with_name(f".{destination.name}.part")
    if work.exists():
        raise FileExistsError("stale evaluation part directory must be audited")
    work.mkdir(parents=True)
    try:
        traces = _write_traces(work, primary=primary, seed=seed_rows)
        metrics = daily_metrics(
            primary.daily_pnl_cents,
            completed_round_trips=primary.completed_round_trips,
        )
        if seed_rows is None:
            per_seed_daily: dict[str, dict[str, int]] | None = None
            per_seed_metrics: dict[str, dict[str, object]] | None = None
        else:
            per_seed_daily, per_seed_metrics = _per_seed_report(seed_rows)
        report: dict[str, object] = {
            "schema_version": 1,
            "artifact_type": REPLAY_ARTIFACT,
            "experiment_id": PROTOCOL_ID,
            "protocol_id": PROTOCOL_ID,
            "protocol": protocol_binding(),
            "implementation": implementation_binding(),
            "status": "COMPLETE",
            "arm": selected_arm,
            "policy_family": (
                "learned_five_seed" if selected_arm in LEARNER_ARMS
                else "deterministic_single_path"
            ),
            "period": selected_period,
            "phase": "validation" if selected_period == "may_2026" else "test",
            "classification": PERIOD_CLASSIFICATION[selected_period],
            "fresh_oos": False,
            "scenario": selected_scenario,
            "economics": asdict(economics),
            "evaluation_dates": list(dataset.dates),
            "daily_pnl_cents": dict(primary.daily_pnl_cents),
            "metrics": metrics,
            "completed_round_trips": primary.completed_round_trips,
            "rejected_commands": primary.rejected_commands,
            "filled_sides": primary.filled_sides,
            "commissions_cents": primary.commissions_cents,
            "explicit_commissions_usd": primary.commissions_cents / 100.0,
            "ensemble_seeds": list(SEEDS) if selected_arm in LEARNER_ARMS else [],
            "policy_replications": len(SEEDS) if selected_arm in LEARNER_ARMS else 1,
            "ensemble_rule": (
                ENSEMBLE_RULE if selected_arm in LEARNER_ARMS else DETERMINISTIC_RULE
            ),
            "per_seed_daily_pnl_cents": per_seed_daily,
            "per_seed_metrics": per_seed_metrics,
            "checkpoint_set": checkpoints,
            "suite_training_prerequisite": suite_training_binding,
            "training_manifest": training_binding,
            "rule_scaler_reference": rule_scaler_binding,
            "dataset_manifest": dataset.binding(),
            **traces,
            "reward_telescope_verified": True,
            "terminal_inventory_flat": True,
            "action_legality_verified": True,
            "routes_verified": True,
            "unsafe_legacy_arrival_boundary": False,
            "legality_fallbacks": sum(
                1 for row in primary.decisions if bool(row.get("legality_fallback"))
            ),
            "unexpected_policy_fallbacks": 0,
            "per_seed_policy_dependent_replays_complete": (
                selected_arm in LEARNER_ARMS
            ),
            "registered_evaluation": True,
            "holdout_read": True,
            "august_accessed": False,
            "included_in_reporting": True,
            "technical_failure_reason": None,
            "inferential_status": "POST_HOC_DESCRIPTIVE",
        }
        atomic_json(work / "report.json", report)
        os.replace(work, destination)
    except Exception:
        # Preserve the immutable partial directory for explicit audit. Never
        # erase or relabel a failed production attempt automatically.
        raise
    verify_replay_report(
        destination / "report.json",
        expected_arm=selected_arm,
        expected_period=selected_period,
        expected_scenario=selected_scenario,
    )
    return destination / "report.json"


def _validate_route(
    row: Mapping[str, object], *, economics: object, terminal: bool,
) -> tuple[int, int, int, int, bool] | None:
    decision_name = "next_time_ns" if terminal else "decision_time_ns"
    deadline_name = "terminal_route_deadline_ns" if terminal else "route_deadline_ns"
    side_name = "terminal_route_side" if terminal else "route_side"
    fill_name = "terminal_fill_time_ns" if terminal else "route_fill_time_ns"
    price_name = "terminal_fill_price_ticks" if terminal else "route_fill_price_ticks"
    fee_name = "terminal_fill_fee_cents" if terminal else "route_fill_fee_cents"
    size_name = "terminal_displayed_size" if terminal else "route_displayed_size"
    deadline = row.get(deadline_name)
    side = row.get(side_name)
    fill_time = row.get(fill_name)
    price = row.get(price_name)
    fee = row.get(fee_name)
    displayed = row.get(size_name)
    if deadline is None:
        if any(value is not None for value in (side, fill_time, price, fee, displayed)):
            raise ValueError("absent route is partially populated")
        return None
    if side is None or int(side) not in (-1, 1):
        raise ValueError("created route lacks a valid side")
    decision = int(row.get(decision_name, -1))
    expected_deadline = decision + int(economics.deadline_ms) * 1_000_000  # type: ignore[attr-defined]
    if int(deadline) != expected_deadline:
        raise ValueError("route deadline differs from scenario economics")
    if fill_time is None:
        if any(value is not None for value in (price, fee, displayed)):
            raise ValueError("unfilled route has partial fill fields")
        if terminal:
            raise ValueError("mandatory terminal route is unfilled")
        if not terminal and row.get("command_rejected") is not True:
            raise ValueError("unfilled decision route is not recorded as rejected")
        return None
    if price is None or fee is None or displayed is None:
        raise ValueError("filled route is incomplete")
    fill_ns = int(fill_time)
    arrival = decision + int(economics.route_latency_ms) * 1_000_000  # type: ignore[attr-defined]
    if not arrival <= fill_ns <= int(deadline) or int(displayed) < 1:
        raise ValueError("fill violates route latency/deadline/displayed-size contract")
    if int(fee) != int(economics.fee_cents_per_side):  # type: ignore[attr-defined]
        raise ValueError("fill commission differs from scenario economics")
    return fill_ns, int(side), int(price), int(fee), bool(terminal)


def _reward_index(
    rows: Sequence[Mapping[str, object]], *, with_seed: bool,
) -> dict[tuple[object, ...], Mapping[str, object]]:
    result: dict[tuple[object, ...], Mapping[str, object]] = {}
    for row in rows:
        key: tuple[object, ...] = (
            str(row.get("session_date")),
            *((int(row.get("seed", -1)),) if with_seed else ()),
            int(row.get("decision_ordinal", -1)),
        )
        if key in result:
            raise ValueError("reward trace duplicates a transition key")
        result[key] = row
    return result


def _decision_groups(
    rows: Sequence[Mapping[str, object]], *, with_seed: bool,
) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    result: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key: tuple[object, ...] = (
            str(row.get("session_date")),
            *((int(row.get("seed", -1)),) if with_seed else ()),
            int(row.get("decision_ordinal", -1)),
        )
        result.setdefault(key, []).append(row)
    return result


def _verify_decisions(
    *, arm: str, decisions: Sequence[Mapping[str, object]],
    rewards: Sequence[Mapping[str, object]], learned_ensemble: bool,
    independent_seed: bool,
) -> None:
    groups = _decision_groups(decisions, with_seed=independent_seed)
    reward_map = _reward_index(rewards, with_seed=independent_seed)
    if set(groups) != set(reward_map):
        raise ValueError("decision and reward trace keys differ")
    expected_rows = len(SEEDS) if learned_ensemble else 1
    for key, rows in groups.items():
        reward = reward_map[key]
        if len(rows) != expected_rows:
            raise ValueError("decision trace has an incorrect policy multiplicity")
        legal = np.asarray([
            reward.get("legal_short"), reward.get("legal_flat"),
            reward.get("legal_long"),
        ], dtype=np.bool_)
        action = int(reward.get("executed_action", 99))
        if legal.shape != (3,) or not bool(legal.any()) or action not in ACTIONS:
            raise ValueError("reward trace contains an invalid legal/action state")
        if not bool(legal[ACTIONS.index(action)]):
            raise ValueError("recorded action is illegal")
        votes: list[int] = []
        seeds: set[int] = set()
        for row in rows:
            scores = np.asarray([
                row.get("score_short"), row.get("score_flat"), row.get("score_long"),
            ], dtype=np.float64)
            row_legal = np.asarray([
                row.get("legal_short"), row.get("legal_flat"), row.get("legal_long"),
            ], dtype=np.bool_)
            if not np.array_equal(row_legal, legal):
                raise ValueError("decision policies disagree on shared legal state")
            if independent_seed or learned_ensemble:
                expected_vote = legal_masked_vote(scores, legal)
                vote = int(row.get("member_vote", 99))
                if vote != expected_vote:
                    raise ValueError("learned policy vote does not reconcile to scores")
            else:
                desired = int(row.get("desired_action", 99))
                if desired not in ACTIONS or int(ACTIONS[int(np.argmax(scores))]) != desired:
                    raise ValueError("deterministic desired action does not reconcile to scores")
                expected_action = desired if bool(legal[ACTIONS.index(desired)]) else 0
                vote = int(row.get("member_vote", 99))
                if vote != expected_action or bool(row.get("legality_fallback")) != (
                    desired != expected_action
                ):
                    raise ValueError("deterministic legality fallback differs")
            votes.append(vote)
            if learned_ensemble:
                seeds.add(int(row.get("seed", -1)))
            for field in (
                "decision_time_ns", "next_time_ns", "executed_action",
                "inventory_before", "inventory_after", "reward_cents", "terminal",
            ):
                if row.get(field) != reward.get(field):
                    raise ValueError("decision and reward transition fields differ")
        if learned_ensemble:
            if seeds != set(SEEDS) or flat_on_vote_tie(votes) != action:
                raise ValueError("ensemble action does not reconcile to five seed votes")
            counts = {candidate: votes.count(candidate) for candidate in ACTIONS}
            for row in rows:
                if (
                    int(row.get("votes_short", -1)) != counts[-1]
                    or int(row.get("votes_flat", -1)) != counts[0]
                    or int(row.get("votes_long", -1)) != counts[1]
                ):
                    raise ValueError("ensemble vote-count trace differs")
        elif action != votes[0]:
            raise ValueError("single policy action differs from its trace")


def _verify_reward_sequences(
    *, rewards: Sequence[Mapping[str, object]], fills: Sequence[Mapping[str, object]],
    economics: object, daily: Mapping[str, int], expected_rejections: int,
    expected_filled_sides: int, with_seed: bool,
) -> None:
    grouped: dict[tuple[str, int | None], list[Mapping[str, object]]] = {}
    for row in rewards:
        key = (
            str(row.get("session_date")),
            int(row.get("seed", -1)) if with_seed else None,
        )
        grouped.setdefault(key, []).append(row)
    generated: list[tuple[object, ...]] = []
    rejected = 0
    for (day, seed), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row.get("decision_ordinal", -1)))
        if [int(row.get("decision_ordinal", -1)) for row in ordered] != list(
            range(len(ordered))
        ):
            raise ValueError("reward ordinals are incomplete")
        cumulative = 0
        previous_next: int | None = None
        previous_inventory = 0
        for ordinal, row in enumerate(ordered):
            decision = int(row.get("decision_time_ns", -1))
            following = int(row.get("next_time_ns", -1))
            if following <= decision or (
                previous_next is not None and decision != previous_next
            ):
                raise ValueError("reward transition clocks are not contiguous")
            before = int(row.get("inventory_before", 99))
            after = int(row.get("inventory_after", 99))
            if before != previous_inventory:
                raise ValueError("inventory trace is discontinuous")
            legal = np.asarray([
                row.get("legal_short"), row.get("legal_flat"), row.get("legal_long"),
            ], dtype=np.bool_)
            if not np.array_equal(legal, legal_target_mask(before)):
                raise ValueError("legal mask differs from recorded inventory")
            action = int(row.get("executed_action", 99))
            if action not in ACTIONS or not bool(legal[ACTIONS.index(action)]):
                raise ValueError("reward trace action is illegal")
            terminal = bool(row.get("terminal"))
            if terminal != (ordinal == len(ordered) - 1):
                raise ValueError("terminal flag is misplaced")
            cumulative += int(row.get("reward_cents", 0))
            if cumulative != int(row.get("cumulative_executable_nlv_cents", 1)):
                raise ValueError("executable NLV does not telescope")
            main = _validate_route(row, economics=economics, terminal=False)
            forced = _validate_route(row, economics=economics, terminal=True)
            main_created = row.get("route_deadline_ns") is not None
            if bool(row.get("command_created")) != main_created:
                raise ValueError("command-created flag differs from route trace")
            if bool(row.get("command_rejected")) != (
                main_created and main is None
            ):
                raise ValueError("command-rejected flag differs from route outcome")
            # Independently reconcile semantic target, route side, rejection,
            # and the inventory that exists just before any forced flatten.
            if main_created:
                expected_side = action - before
                if expected_side not in (-1, 1) or int(
                    row.get("route_side", 0)
                ) != expected_side:
                    raise ValueError("decision route side differs from target delta")
                mid_inventory = action if main is not None else before
            else:
                if action != before:
                    raise ValueError("target change has no recorded route")
                mid_inventory = before
            if terminal:
                if forced is None:
                    if mid_inventory != 0:
                        raise ValueError("open terminal inventory lacks forced route")
                elif (
                    mid_inventory == 0
                    or int(row.get("terminal_route_side", 0)) != -mid_inventory
                ):
                    raise ValueError("forced route side differs from terminal inventory")
                expected_after = 0
            else:
                if forced is not None or row.get("terminal_route_deadline_ns") is not None:
                    raise ValueError("nonterminal transition contains a forced route")
                expected_after = mid_inventory
            if after != expected_after:
                raise ValueError("inventory outcome differs from route outcomes")
            rejected += int(bool(row.get("command_rejected")))
            prefix: tuple[object, ...] = (day, *((seed,) if with_seed else ()))
            if main is not None:
                generated.append((*prefix, *main))
            if forced is not None:
                generated.append((*prefix, *forced))
            previous_next = following
            previous_inventory = after
        if previous_inventory != 0:
            raise ValueError("terminal reward trace retains inventory")
        expected_cash = (
            int(daily[day]) if not with_seed
            else int(daily[f"{seed}:{day}"])
        )
        if cumulative != expected_cash:
            raise ValueError("terminal executable NLV differs from daily cash")

    recorded: list[tuple[object, ...]] = []
    ordinal_groups: dict[tuple[object, ...], list[int]] = {}
    for row in fills:
        day = str(row.get("session_date"))
        seed = int(row.get("seed", -1)) if with_seed else None
        key: tuple[object, ...] = (day, *((seed,) if with_seed else ()))
        ordinal_groups.setdefault(key, []).append(int(row.get("fill_ordinal", -1)))
        recorded.append((
            *key,
            int(row.get("timestamp_ns", -1)),
            int(row.get("side", 0)),
            int(row.get("price_ticks", 0)),
            int(row.get("fee_cents", -1)),
            bool(row.get("forced")),
        ))
    if any(sorted(values) != list(range(len(values))) for values in ordinal_groups.values()):
        raise ValueError("fill ordinals are incomplete")
    if sorted(recorded) != sorted(generated):
        raise ValueError("fill trace does not reconcile to route outcomes")
    if rejected != int(expected_rejections):
        raise ValueError("rejection count does not reconcile")
    if len(recorded) != int(expected_filled_sides):
        raise ValueError("filled-side count does not reconcile")


def _verify_binding_file(record: object) -> Path:
    if not isinstance(record, Mapping) or record.get("path_scope") != "project_root":
        raise ValueError("project-root binding is malformed")
    path = resolve_project_path(record.get("path"))
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("project-root binding hash differs")
    return path


def verify_replay_report(
    path: Path, *, expected_arm: str | None = None,
    expected_period: str | None = None, expected_scenario: str | None = None,
) -> dict[str, object]:
    _reject_august_lexically(
        path, expected_arm or "", expected_period or "", expected_scenario or "",
    )
    report_path = Path(path).resolve()
    value = read_json_object(report_path)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != REPLAY_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("reward_telescope_verified") is not True
        or value.get("terminal_inventory_flat") is not True
        or value.get("action_legality_verified") is not True
        or value.get("routes_verified") is not True
        or value.get("unsafe_legacy_arrival_boundary") is not False
        or value.get("unexpected_policy_fallbacks") != 0
        or value.get("registered_evaluation") is not True
        or value.get("holdout_read") is not True
        or value.get("august_accessed") is not False
        or value.get("included_in_reporting") is not True
        or value.get("technical_failure_reason") is not None
        or value.get("inferential_status") != "POST_HOC_DESCRIPTIVE"
    ):
        raise ValueError("baseline replay report identity differs")
    arm = validate_arm(str(value.get("arm", ""))).value
    period = validate_period(str(value.get("period", ""))).value
    scenario = validate_scenario(str(value.get("scenario", ""))).value
    if expected_arm is not None and arm != str(expected_arm):
        raise ValueError("baseline replay arm differs")
    if expected_period is not None and period != str(expected_period):
        raise ValueError("baseline replay period differs")
    if expected_scenario is not None and scenario != str(expected_scenario):
        raise ValueError("baseline replay scenario differs")
    learned = arm in LEARNER_ARMS
    if (
        value.get("policy_family")
        != ("learned_five_seed" if learned else "deterministic_single_path")
        or value.get("ensemble_seeds") != (list(SEEDS) if learned else [])
        or int(value.get("policy_replications", -1)) != (len(SEEDS) if learned else 1)
        or value.get("ensemble_rule") != (ENSEMBLE_RULE if learned else DETERMINISTIC_RULE)
        or value.get("per_seed_policy_dependent_replays_complete") is not learned
    ):
        raise ValueError("baseline replay policy/ensemble contract differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("baseline replay protocol binding differs")
    verify_implementation_binding(value.get("implementation"))
    _verify_suite_training_prerequisite(value.get("suite_training_prerequisite"))
    economics = scenario_economics(scenario)
    if value.get("economics") != asdict(economics):
        raise ValueError("baseline replay scenario economics differ")
    expected_phase = "validation" if period == "may_2026" else "test"
    if value.get("phase") != expected_phase:
        raise ValueError("baseline replay phase differs")

    dates = value.get("evaluation_dates")
    daily_raw = value.get("daily_pnl_cents")
    if (
        not isinstance(dates, list)
        or dates != sorted(set(str(day) for day in dates))
        or not isinstance(daily_raw, Mapping)
        or sorted(str(day) for day in daily_raw) != dates
    ):
        raise ValueError("baseline replay date/daily-cash set differs")
    daily = {str(day): int(amount) for day, amount in daily_raw.items()}
    expected_metrics = daily_metrics(
        daily, completed_round_trips=int(value.get("completed_round_trips", -1)),
    )
    if value.get("metrics") != expected_metrics:
        raise ValueError("baseline replay metrics do not reconcile")
    if (
        int(value.get("filled_sides", -1))
        != 2 * int(value.get("completed_round_trips", -1))
        or int(value.get("commissions_cents", -1)) < 0
        or float(value.get("explicit_commissions_usd", -1.0))
        != int(value.get("commissions_cents", -1)) / 100.0
    ):
        raise ValueError("baseline replay activity/commission summary differs")

    dataset = value.get("dataset_manifest")
    expected_dataset = DATASET_MANIFESTS[period]
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("path_scope") != "project_root"
        or dataset.get("path") != expected_dataset["path"]
        or dataset.get("sha256") != expected_dataset["sha256"]
        or dataset.get("purpose") != "evaluate"
        or dataset.get("period_id") != period
        or dataset.get("classification") != PERIOD_CLASSIFICATION[period]
        or dataset.get("dates") != dates
    ):
        raise ValueError("baseline replay dataset binding differs")
    dataset_path = _verify_binding_file(dataset)
    dataset_value = read_json_object(dataset_path)
    sessions = dataset_value.get("sessions")
    if not isinstance(sessions, Mapping) or sorted(str(day) for day in sessions) != dates:
        raise ValueError("baseline replay dates differ from dataset manifest")

    checkpoints = value.get("checkpoint_set")
    if learned:
        from .training import verify_complete_training_manifest

        training_path = _verify_binding_file(value.get("training_manifest"))
        complete = verify_complete_training_manifest(training_path)
        records = complete.get("members")
        assert isinstance(records, list)
        expected_members: dict[int, tuple[str, str]] = {}
        for record in records:
            if not isinstance(record, Mapping) or record.get("arm") != arm:
                continue
            member_path = resolve_project_path(record.get("manifest"))
            manifest = read_json_object(member_path)
            model = manifest.get("model")
            if not isinstance(model, Mapping):
                raise ValueError("training member model binding is malformed")
            expected_members[int(record["seed"])] = (
                str(record["manifest_sha256"]), str(model.get("sha256")),
            )
        if set(expected_members) != set(SEEDS) or not isinstance(
            checkpoints, list
        ) or len(checkpoints) != len(SEEDS):
            raise ValueError("baseline replay checkpoint set is incomplete")
        observed: dict[int, tuple[str, str]] = {}
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping):
                raise ValueError("baseline replay checkpoint record is malformed")
            manifest_path = _verify_binding_file(checkpoint.get("manifest"))
            seed = int(checkpoint.get("seed", -1))
            observed[seed] = (
                file_sha256(manifest_path), str(checkpoint.get("model_sha256")),
            )
        if observed != expected_members or value.get("rule_scaler_reference") is not None:
            raise ValueError("baseline replay checkpoint set differs from training")
    else:
        if value.get("training_manifest") is not None or checkpoints != []:
            raise ValueError("deterministic replay unexpectedly binds training")
        reference = value.get("rule_scaler_reference")
        reference_path = _verify_binding_file(reference)
        if (
            not isinstance(reference, Mapping)
            or project_relative(reference_path)
            != f"{RULE_SCALER_ROOT}/manifest.json"
            or reference.get("policy_uses_scaled_values") is not False
            or not isinstance(reference.get("scaler_canonical_sha256"), str)
        ):
            raise ValueError("deterministic rule scaler lineage differs")

    decisions_path = _resolve_trace(report_path, value.get("decision_trace"))
    rewards_path = _resolve_trace(report_path, value.get("reward_trace"))
    fills_path = _resolve_trace(report_path, value.get("fill_trace"))
    decisions = _rows(decisions_path)
    rewards = _rows(rewards_path)
    fills = _rows(fills_path)
    _verify_decisions(
        arm=arm,
        decisions=decisions,
        rewards=rewards,
        learned_ensemble=learned,
        independent_seed=False,
    )
    _verify_reward_sequences(
        rewards=rewards,
        fills=fills,
        economics=economics,
        daily=daily,
        expected_rejections=int(value.get("rejected_commands", -1)),
        expected_filled_sides=int(value.get("filled_sides", -1)),
        with_seed=False,
    )
    trace_fee = sum(int(row["fee_cents"]) for row in fills)
    if trace_fee != int(value.get("commissions_cents", -1)):
        raise ValueError("baseline replay fill fees do not reconcile")

    if learned:
        seed_decisions_path = _resolve_trace(
            report_path, value.get("seed_decision_trace"),
        )
        seed_rewards_path = _resolve_trace(report_path, value.get("seed_reward_trace"))
        seed_fills_path = _resolve_trace(report_path, value.get("seed_fill_trace"))
        seed_decisions = _rows(seed_decisions_path)
        seed_rewards = _rows(seed_rewards_path)
        seed_fills = _rows(seed_fills_path)
        if len(seed_rewards) != len(SEEDS) * len(rewards):
            raise ValueError("independent per-seed reward trace is incomplete")
        _verify_decisions(
            arm=arm,
            decisions=seed_decisions,
            rewards=seed_rewards,
            learned_ensemble=False,
            independent_seed=True,
        )
        per_seed_daily_raw = value.get("per_seed_daily_pnl_cents")
        per_seed_metrics = value.get("per_seed_metrics")
        if not isinstance(per_seed_daily_raw, Mapping) or not isinstance(
            per_seed_metrics, Mapping
        ):
            raise ValueError("per-seed economic diagnostics are absent")
        flat_seed_daily: dict[str, int] = {}
        total_seed_rejections = 0
        total_seed_fills = 0
        total_seed_fees = 0
        for seed in SEEDS:
            one = per_seed_daily_raw.get(str(seed))
            metrics = per_seed_metrics.get(str(seed))
            if not isinstance(one, Mapping) or sorted(str(day) for day in one) != dates:
                raise ValueError("per-seed daily cash is incomplete")
            if not isinstance(metrics, Mapping):
                raise ValueError("per-seed metrics are incomplete")
            one_daily = {str(day): int(amount) for day, amount in one.items()}
            expected = daily_metrics(
                one_daily,
                completed_round_trips=int(metrics.get("completed_round_trips", -1)),
            )
            if any(metrics.get(name) != amount for name, amount in expected.items()):
                raise ValueError("per-seed daily metrics do not reconcile")
            filled = sum(1 for row in seed_fills if int(row.get("seed", -1)) == seed)
            fees = sum(
                int(row.get("fee_cents", -1))
                for row in seed_fills if int(row.get("seed", -1)) == seed
            )
            if (
                int(metrics.get("filled_sides", -1)) != filled
                or int(metrics.get("commissions_cents", -1)) != fees
                or float(metrics.get("explicit_commissions_usd", -1.0)) != fees / 100.0
                or filled != 2 * int(metrics.get("completed_round_trips", -1))
            ):
                raise ValueError("per-seed activity/commissions do not reconcile")
            total_seed_rejections += int(metrics.get("rejected_commands", -1))
            total_seed_fills += filled
            total_seed_fees += fees
            for day, amount in one_daily.items():
                flat_seed_daily[f"{seed}:{day}"] = amount
        _verify_reward_sequences(
            rewards=seed_rewards,
            fills=seed_fills,
            economics=economics,
            daily=flat_seed_daily,
            expected_rejections=total_seed_rejections,
            expected_filled_sides=total_seed_fills,
            with_seed=True,
        )
        if sum(int(row["fee_cents"]) for row in seed_fills) != total_seed_fees:
            raise ValueError("per-seed trace fees do not reconcile")
    else:
        if any(value.get(name) is not None for name in (
            "seed_decision_trace", "seed_reward_trace", "seed_fill_trace",
        )) or value.get("per_seed_daily_pnl_cents") is not None or value.get(
            "per_seed_metrics"
        ) is not None:
            raise ValueError("deterministic replay contains replicated seed evidence")
        fallbacks = sum(1 for row in decisions if bool(row.get("legality_fallback")))
        if int(value.get("legality_fallbacks", -1)) != fallbacks:
            raise ValueError("deterministic legality-fallback count differs")
    return value


def _registered_dataset_binding(
    period: str, dataset_manifest: str | None = None,
) -> dict[str, object]:
    """Return the frozen evaluation binding without enumerating/opening data."""

    relative = PERIOD_DATASETS[period] if dataset_manifest is None else str(
        dataset_manifest
    )
    authorization = authorize_manifest_request(
        relative, purpose="evaluate", period_id=period,
    )
    return {
        "path": authorization.relative_path,
        "path_scope": "project_root",
        "sha256": authorization.sha256,
        "purpose": authorization.purpose,
        "period_id": authorization.period_id,
        "classification": authorization.classification,
    }


def _verified_training_binding(
    arm: str, training_manifest: Path | str | None, reason: str,
) -> tuple[dict[str, object] | None, str]:
    """Bind a complete learner registry or explicitly retain why it is absent."""

    if arm in DETERMINISTIC_ARMS:
        if training_manifest is not None:
            raise ValueError("deterministic failure records must not bind training")
        return None, "NOT_APPLICABLE"
    if training_manifest is None:
        if reason not in {
            "INCOMPLETE_TRAINING", "POLICY_LOAD_FAILURE", "HASH_MISMATCH",
        }:
            raise ValueError(
                "this learner failure requires its verified training manifest"
            )
        return None, "UNAVAILABLE_DUE_TO_FAILURE"

    from .training import verify_complete_training_manifest

    path = Path(training_manifest).resolve()
    verify_complete_training_manifest(path)
    return {
        "path": project_relative(path),
        "path_scope": "project_root",
        "sha256": file_sha256(path),
    }, "VERIFIED"


def _suite_training_prerequisite(
    root: Path, supplied: Path | str | None,
) -> tuple[Path, dict[str, object]]:
    """Verify the exact 15-job receipt required before every evaluation cell."""

    registered = root / "complete_training_manifest.json"
    if supplied is not None:
        candidate = Path(supplied)
        candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else resolve_project_path(candidate.as_posix())
        )
        if candidate != registered:
            raise ValueError("evaluation prerequisite must be the registered manifest")
    from .training import verify_complete_training_manifest

    verify_complete_training_manifest(registered)
    return registered, {
        "path": project_relative(registered),
        "path_scope": "project_root",
        "sha256": file_sha256(registered),
    }


def _verify_suite_training_prerequisite(record: object) -> Path:
    from .training import verify_complete_training_manifest

    path = _verify_binding_file(record)
    verify_complete_training_manifest(path)
    return path


def write_technical_failure(
    *, arm: str | BaselineArm, period: str, scenario: str, reason: str,
    diagnostic: str, output_root: Path | str | None = None,
    training_manifest: Path | str | None = None,
    dataset_manifest: str | None = None, holdout_read: bool = False,
) -> Path:
    """Persist one predeclared integrity failure as a retained suite result.

    Economic underperformance, inactivity, or outperforming CVP-HDDQN are not
    accepted failure reasons.  This writer never fabricates partial metrics or
    relabels an exception automatically.
    """

    # All caller-controlled strings pass the lexical unopened-period guard
    # before path resolution, contract reads, output creation, or other I/O.
    _reject_august_lexically(
        arm, period, scenario, reason, diagnostic, output_root or "",
        training_manifest or "", dataset_manifest or "",
    )
    selected_arm = validate_arm(arm).value
    selected_period = validate_period(period).value
    selected_scenario = validate_scenario(scenario).value
    selected_reason = str(reason)
    if selected_reason not in TECHNICAL_FAILURE_REASONS:
        raise ValueError("technical failure reason is not predeclared")
    if selected_reason == "INCOMPLETE_TRAINING":
        raise ValueError(
            "incomplete training is a suite-level pre-evaluation failure, not a replay cell"
        )
    message = str(diagnostic).strip()
    if not message:
        raise ValueError("technical failure diagnostic must be nonempty")

    root = _resolve_output_root(output_root)
    registered_training, suite_training_binding = _suite_training_prerequisite(
        root, training_manifest,
    )
    dataset_binding = _registered_dataset_binding(
        selected_period, dataset_manifest,
    )
    training_binding, training_status = _verified_training_binding(
        selected_arm,
        registered_training if selected_arm in LEARNER_ARMS else None,
        selected_reason,
    )
    destination = (
        root
        / "evaluation"
        / selected_period
        / selected_scenario
        / selected_arm
        / "report.json"
    )
    partial_path = destination.parent.with_name(f".{destination.parent.name}.part")
    if partial_path.exists():
        raise FileExistsError("technical failure cannot coexist with partial traces")
    if destination.parent.exists():
        raise FileExistsError("registered evaluation result already exists")
    value: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": TECHNICAL_FAILURE_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "status": "TECHNICAL_FAILURE",
        "arm": selected_arm,
        "policy_family": (
            "learned_five_seed" if selected_arm in LEARNER_ARMS
            else "deterministic_single_path"
        ),
        "period": selected_period,
        "phase": "validation" if selected_period == "may_2026" else "test",
        "classification": PERIOD_CLASSIFICATION[selected_period],
        "fresh_oos": False,
        "scenario": selected_scenario,
        "economics": asdict(scenario_economics(selected_scenario)),
        "dataset_manifest": dataset_binding,
        "suite_training_prerequisite": suite_training_binding,
        "training_manifest": training_binding,
        "training_binding_status": training_status,
        "technical_failure_reason": selected_reason,
        "diagnostic": message,
        "evaluation_dates": None,
        "daily_pnl_cents": None,
        "metrics": None,
        "decision_trace": None,
        "reward_trace": None,
        "fill_trace": None,
        "seed_decision_trace": None,
        "seed_reward_trace": None,
        "seed_fill_trace": None,
        "registered_evaluation": True,
        "holdout_read": bool(holdout_read),
        "august_accessed": False,
        "included_in_reporting": True,
        "inferential_status": "TECHNICALLY_INVALID_REPORTED",
    }
    atomic_json(destination, value)
    verify_technical_failure_report(
        destination,
        expected_arm=selected_arm,
        expected_period=selected_period,
        expected_scenario=selected_scenario,
    )
    return destination


def verify_technical_failure_report(
    path: Path | str, *, expected_arm: str | None = None,
    expected_period: str | None = None, expected_scenario: str | None = None,
) -> dict[str, object]:
    """Deep-verify one retained failure without interpreting it economically."""

    _reject_august_lexically(
        path, expected_arm or "", expected_period or "", expected_scenario or "",
    )
    report_path = Path(path).resolve()
    value = read_json_object(report_path)
    absent = (
        "evaluation_dates", "daily_pnl_cents", "metrics", "decision_trace",
        "reward_trace", "fill_trace", "seed_decision_trace",
        "seed_reward_trace", "seed_fill_trace",
    )
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != TECHNICAL_FAILURE_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "TECHNICAL_FAILURE"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("registered_evaluation") is not True
        or not isinstance(value.get("holdout_read"), bool)
        or value.get("august_accessed") is not False
        or value.get("included_in_reporting") is not True
        or value.get("inferential_status") != "TECHNICALLY_INVALID_REPORTED"
        or value.get("technical_failure_reason") not in TECHNICAL_FAILURE_REASONS
        or not isinstance(value.get("diagnostic"), str)
        or not str(value.get("diagnostic")).strip()
        or any(value.get(name) is not None for name in absent)
    ):
        raise ValueError("baseline technical-failure report identity differs")
    arm = validate_arm(str(value.get("arm", ""))).value
    period = validate_period(str(value.get("period", ""))).value
    scenario = validate_scenario(str(value.get("scenario", ""))).value
    if expected_arm is not None and arm != str(expected_arm):
        raise ValueError("baseline technical-failure arm differs")
    if expected_period is not None and period != str(expected_period):
        raise ValueError("baseline technical-failure period differs")
    if expected_scenario is not None and scenario != str(expected_scenario):
        raise ValueError("baseline technical-failure scenario differs")
    if (
        value.get("policy_family")
        != ("learned_five_seed" if arm in LEARNER_ARMS else "deterministic_single_path")
        or value.get("phase") != ("validation" if period == "may_2026" else "test")
        or value.get("economics") != asdict(scenario_economics(scenario))
        or value.get("protocol") != protocol_binding()
    ):
        raise ValueError("baseline technical-failure protocol/economics differ")
    verify_implementation_binding(value.get("implementation"))
    _verify_suite_training_prerequisite(value.get("suite_training_prerequisite"))
    expected_dataset = _registered_dataset_binding(period)
    if value.get("dataset_manifest") != expected_dataset:
        raise ValueError("baseline technical-failure dataset binding differs")

    training = value.get("training_manifest")
    training_status = value.get("training_binding_status")
    reason = str(value["technical_failure_reason"])
    if arm in LEARNER_ARMS:
        from .training import verify_complete_training_manifest

        if training is None or training_status != "VERIFIED":
            raise ValueError("learner failure omits verified training lineage")
        verify_complete_training_manifest(_verify_binding_file(training))
    elif training is not None or training_status != "NOT_APPLICABLE":
        raise ValueError("deterministic failure unexpectedly binds training")
    if reason == "INCOMPLETE_TRAINING":
        raise ValueError("incomplete training cannot be registered as an evaluation cell")
    return value


def _relative_evaluation_report(
    evaluation_root: Path, report_path: Path,
) -> str:
    root = evaluation_root.resolve()
    resolved = report_path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("evaluation report escapes the registered root") from error
    if relative.parts[-1:] != ("report.json",) or len(relative.parts) != 4:
        raise ValueError("evaluation report path has an unexpected layout")
    return relative.as_posix()


def build_complete_evaluation_manifest(
    output_root: Path | str | None = None, output: Path | str | None = None,
) -> Path:
    """Build the immutable, result-complete registry of all 64 cells."""

    _reject_august_lexically(output_root or "", output or "")
    root = _resolve_output_root(output_root)
    evaluation_root = root / "evaluation"
    destination = (
        evaluation_root / "manifest.json"
        if output is None else Path(output).resolve()
    )
    if destination.is_file():
        verify_complete_evaluation_manifest(destination)
        return destination
    if destination.parent.resolve() != evaluation_root.resolve():
        raise ValueError("complete evaluation manifest must reside in evaluation root")

    records: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    for arm in BASELINE_ARMS:
        for period in PERIODS:
            for scenario in SCENARIOS:
                report_path = (
                    evaluation_root / period / scenario / arm / "report.json"
                )
                partial_path = report_path.parent.with_name(f".{arm}.part")
                if partial_path.exists():
                    raise FileExistsError(
                        "complete evaluation registry cannot certify partial traces"
                    )
                value = read_json_object(report_path)
                artifact = value.get("artifact_type")
                if artifact == REPLAY_ARTIFACT:
                    verified = verify_replay_report(
                        report_path, expected_arm=arm, expected_period=period,
                        expected_scenario=scenario,
                    )
                    status = "COMPLETE"
                elif artifact == TECHNICAL_FAILURE_ARTIFACT:
                    verified = verify_technical_failure_report(
                        report_path, expected_arm=arm, expected_period=period,
                        expected_scenario=scenario,
                    )
                    status = "TECHNICAL_FAILURE"
                else:
                    raise ValueError("registered evaluation report artifact type differs")
                if verified.get("included_in_reporting") is not True:
                    raise ValueError("registered evaluation result was omitted")
                record = {
                    "arm": arm,
                    "period": period,
                    "scenario": scenario,
                    "status": status,
                    "path": _relative_evaluation_report(
                        evaluation_root, report_path,
                    ),
                    "sha256": file_sha256(report_path),
                }
                records.append(record)
                reports.append(verified)

    expected_cells = len(BASELINE_ARMS) * len(PERIODS) * len(SCENARIOS)
    if len(records) != expected_cells or expected_cells != 64:
        raise AssertionError("complete evaluation registry is not 64 cells")
    nonnull_training = [
        report.get("training_manifest") for report in reports
        if report.get("training_manifest") is not None
    ]
    unique_training = {
        canonical_json_sha256(binding): binding for binding in nonnull_training
    }
    if len(unique_training) > 1:
        raise ValueError("evaluation cells bind different learner training manifests")
    common_training = next(iter(unique_training.values()), None)
    suite_bindings = {
        canonical_json_sha256(report.get("suite_training_prerequisite")):
        report.get("suite_training_prerequisite")
        for report in reports
    }
    if len(suite_bindings) != 1:
        raise ValueError("evaluation cells bind different suite training prerequisites")
    suite_training = next(iter(suite_bindings.values()))
    complete_cells = sum(record["status"] == "COMPLETE" for record in records)
    failure_cells = len(records) - complete_cells
    value = {
        "schema_version": 1,
        "artifact_type": COMPLETE_EVALUATION_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "status": "COMPLETE",
        "complete": True,
        "arms": list(BASELINE_ARMS),
        "periods": list(PERIODS),
        "scenarios": list(SCENARIOS),
        "required_cells": 64,
        "complete_cells": complete_cells,
        "technical_failure_cells": failure_cells,
        "period_classification": {
            period: PERIOD_CLASSIFICATION[period] for period in PERIODS
        },
        "fresh_oos": False,
        "august_accessed": False,
        "all_registered_results_retained": True,
        "result_dependent_omission": False,
        "result_dependent_omission_permitted": False,
        "training_manifest": common_training,
        "suite_training_prerequisite": suite_training,
        "learner_cells_with_verified_training": len(nonnull_training),
        "holdout_read_cells": sum(
            report.get("holdout_read") is True for report in reports
        ),
        "reports": records,
        "report_records_sha256": canonical_json_sha256(records),
        "scientific_status": "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE",
    }
    atomic_json(destination, value)
    verify_complete_evaluation_manifest(destination)
    return destination


def verify_complete_evaluation_manifest(
    path: Path | str,
) -> dict[str, object]:
    """Verify the exact 8 x 2 x 4 result registry and every bound report."""

    _reject_august_lexically(path)
    manifest_path = Path(path).resolve()
    value = read_json_object(manifest_path)
    expected_order = [
        (arm, period, scenario)
        for arm in BASELINE_ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    ]
    records = value.get("reports")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != COMPLETE_EVALUATION_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("complete") is not True
        or value.get("arms") != list(BASELINE_ARMS)
        or value.get("periods") != list(PERIODS)
        or value.get("scenarios") != list(SCENARIOS)
        or value.get("required_cells") != 64
        or value.get("period_classification") != {
            period: PERIOD_CLASSIFICATION[period] for period in PERIODS
        }
        or value.get("fresh_oos") is not False
        or value.get("august_accessed") is not False
        or value.get("all_registered_results_retained") is not True
        or value.get("result_dependent_omission") is not False
        or value.get("result_dependent_omission_permitted") is not False
        or value.get("scientific_status")
        != "POST_HOC_DESCRIPTIVE_HISTORICAL_EVIDENCE"
        or not isinstance(records, list)
        or len(records) != 64
    ):
        raise ValueError("complete baseline evaluation identity differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("complete baseline evaluation protocol binding differs")
    verify_implementation_binding(value.get("implementation"))
    if value.get("report_records_sha256") != canonical_json_sha256(records):
        raise ValueError("complete evaluation report-record digest differs")

    observed_order: list[tuple[str, str, str]] = []
    verified_reports: list[dict[str, object]] = []
    evaluation_root = manifest_path.parent.resolve()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "arm", "period", "scenario", "status", "path", "sha256",
        }:
            raise ValueError("evaluation registry record schema differs")
        arm = validate_arm(str(record.get("arm", ""))).value
        period = validate_period(str(record.get("period", ""))).value
        scenario = validate_scenario(str(record.get("scenario", ""))).value
        observed_order.append((arm, period, scenario))
        expected_relative = f"{period}/{scenario}/{arm}/report.json"
        if record.get("path") != expected_relative:
            raise ValueError("evaluation report path differs from its cell")
        report_path = (evaluation_root / expected_relative).resolve()
        try:
            report_path.relative_to(evaluation_root)
        except ValueError as error:
            raise ValueError("evaluation report path escapes its registry") from error
        if not report_path.is_file() or file_sha256(report_path) != record.get("sha256"):
            raise ValueError("evaluation report hash differs")
        status = record.get("status")
        if status == "COMPLETE":
            report = verify_replay_report(
                report_path, expected_arm=arm, expected_period=period,
                expected_scenario=scenario,
            )
        elif status == "TECHNICAL_FAILURE":
            report = verify_technical_failure_report(
                report_path, expected_arm=arm, expected_period=period,
                expected_scenario=scenario,
            )
        else:
            raise ValueError("evaluation cell status is unregistered")
        if report.get("status") != status or report.get("included_in_reporting") is not True:
            raise ValueError("evaluation cell status/reporting flag differs")
        verified_reports.append(report)
    if observed_order != expected_order:
        raise ValueError("evaluation registry is not the exact ordered Cartesian product")

    complete_cells = sum(report.get("status") == "COMPLETE" for report in verified_reports)
    nonnull_training = [
        report.get("training_manifest") for report in verified_reports
        if report.get("training_manifest") is not None
    ]
    unique_training = {
        canonical_json_sha256(binding): binding for binding in nonnull_training
    }
    suite_bindings = {
        canonical_json_sha256(report.get("suite_training_prerequisite")):
        report.get("suite_training_prerequisite")
        for report in verified_reports
    }
    if (
        value.get("complete_cells") != complete_cells
        or value.get("technical_failure_cells") != 64 - complete_cells
        or value.get("holdout_read_cells")
        != sum(report.get("holdout_read") is True for report in verified_reports)
        or value.get("learner_cells_with_verified_training") != len(nonnull_training)
        or len(unique_training) > 1
        or value.get("training_manifest") != next(iter(unique_training.values()), None)
        or len(suite_bindings) != 1
        or value.get("suite_training_prerequisite")
        != next(iter(suite_bindings.values()), None)
    ):
        raise ValueError("complete evaluation summary/lineage differs")
    return value


def _canonical_trace_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(name): _canonical_trace_value(item)
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_trace_value(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("source replay contains a nonfinite trace value")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported source trace value {type(value).__name__}")


def _assert_source_rows(
    name: str, observed: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
) -> None:
    observed_value = _canonical_trace_value(list(observed))
    expected_value = _canonical_trace_value(list(expected))
    if canonical_json_sha256(observed_value) != canonical_json_sha256(expected_value):
        raise ValueError(
            f"{name} differs from deterministic checkpoint/raw-source replay"
        )


def audit_replay_report_against_source(
    path: Path | str, *, expected_arm: str | None = None,
    expected_period: str | None = None, expected_scenario: str | None = None,
) -> dict[str, object]:
    """Rerun one complete cell read-only from bound models and raw sessions.

    This is intentionally stronger than the self-consistency verifier.  It
    regenerates policy scores, seed votes, legal actions, routes, first causal
    executable fills (including displayed size and adverse-tick prices), raw
    rewards, and terminal cash through the frozen V6 environment, then compares
    every regenerated row with the immutable Parquet traces.  Empty/gapped book
    states are therefore resolved afresh rather than trusted from report fields.
    """

    _reject_august_lexically(
        path, expected_arm or "", expected_period or "", expected_scenario or "",
    )
    report_path = Path(path).resolve()
    report = verify_replay_report(
        report_path,
        expected_arm=expected_arm,
        expected_period=expected_period,
        expected_scenario=expected_scenario,
    )
    arm = str(report["arm"])
    period = str(report["period"])
    scenario = str(report["scenario"])
    dataset_record = report.get("dataset_manifest")
    assert isinstance(dataset_record, Mapping)
    dataset = load_authorized_dataset(
        str(dataset_record["path"]), purpose="evaluate", period_id=period,
    )
    if dataset.binding() != dataset_record:
        raise ValueError("source audit dataset binding changed")

    if arm in LEARNER_ARMS:
        training_path = _verify_binding_file(report.get("training_manifest"))
        ensemble = _load_learned_ensemble(training_path, arm)
        primary, seeds, legacy = _evaluate_learned(
            ensemble, dataset, scenario=scenario,
        )
        expected_seed: RolloutRows | None = seeds
        expected_rule_scaler = None
    else:
        primary, legacy, scaler = _evaluate_rule(arm, dataset, scenario=scenario)
        expected_seed = None
        expected_rule_scaler = _scaler_digest(scaler)
    if legacy:
        raise ValueError("source audit encountered unsafe legacy arrival boundaries")

    trace_names = (
        ("decision_trace", primary.decisions),
        ("reward_trace", primary.rewards),
        ("fill_trace", primary.fills),
    )
    audited_rows: dict[str, int] = {}
    for field, expected_rows in trace_names:
        observed_rows = _rows(_resolve_trace(report_path, report.get(field)))
        _assert_source_rows(field, observed_rows, expected_rows)
        audited_rows[field] = len(observed_rows)
    if expected_seed is not None:
        for field, expected_rows in (
            ("seed_decision_trace", expected_seed.decisions),
            ("seed_reward_trace", expected_seed.rewards),
            ("seed_fill_trace", expected_seed.fills),
        ):
            observed_rows = _rows(_resolve_trace(report_path, report.get(field)))
            _assert_source_rows(field, observed_rows, expected_rows)
            audited_rows[field] = len(observed_rows)

    expected_metrics = daily_metrics(
        primary.daily_pnl_cents,
        completed_round_trips=primary.completed_round_trips,
    )
    if (
        report.get("daily_pnl_cents") != primary.daily_pnl_cents
        or report.get("metrics") != expected_metrics
        or int(report.get("completed_round_trips", -1))
        != primary.completed_round_trips
        or int(report.get("rejected_commands", -1)) != primary.rejected_commands
        or int(report.get("filled_sides", -1)) != primary.filled_sides
        or int(report.get("commissions_cents", -1)) != primary.commissions_cents
    ):
        raise ValueError("source replay economic summary differs from report")
    if expected_seed is not None:
        per_seed_daily, per_seed_metrics = _per_seed_report(expected_seed)
        if (
            report.get("per_seed_daily_pnl_cents") != per_seed_daily
            or report.get("per_seed_metrics") != per_seed_metrics
        ):
            raise ValueError("source replay per-seed economics differ from report")
    else:
        reference = report.get("rule_scaler_reference")
        if (
            not isinstance(reference, Mapping)
            or reference.get("scaler_canonical_sha256") != expected_rule_scaler
        ):
            raise ValueError("source rule scaler differs from report lineage")
    return {
        "artifact_type": "CVP_BASELINE_SOURCE_REPLAY_AUDIT",
        "protocol_id": PROTOCOL_ID,
        "arm": arm,
        "period": period,
        "scenario": scenario,
        "report_sha256": file_sha256(report_path),
        "checkpoint_scores_recomputed": arm in LEARNER_ARMS,
        "rule_signals_recomputed": arm in DETERMINISTIC_ARMS,
        "raw_routes_and_fills_recomputed": True,
        "adverse_tick_price_recomputed": scenario == "adverse_tick",
        "gapped_books_not_carried": True,
        "reward_and_terminal_cash_recomputed": True,
        "trace_rows": audited_rows,
        "status": "PASS",
    }


def audit_complete_evaluation_against_source(
    path: Path | str,
) -> tuple[dict[str, object], ...]:
    """Run the read-only model/raw-source audit for every successful cell."""

    _reject_august_lexically(path)
    manifest_path = Path(path).resolve()
    manifest = verify_complete_evaluation_manifest(manifest_path)
    records = manifest.get("reports")
    assert isinstance(records, list)
    audits: list[dict[str, object]] = []
    for record in records:
        assert isinstance(record, Mapping)
        relative = str(record["path"])
        report_path = (manifest_path.parent / relative).resolve()
        if record["status"] == "COMPLETE":
            audits.append(audit_replay_report_against_source(
                report_path,
                expected_arm=str(record["arm"]),
                expected_period=str(record["period"]),
                expected_scenario=str(record["scenario"]),
            ))
        else:
            verify_technical_failure_report(
                report_path,
                expected_arm=str(record["arm"]),
                expected_period=str(record["period"]),
                expected_scenario=str(record["scenario"]),
            )
    return tuple(audits)


def _run_replay_job(
    args: tuple[str, str, str, str, str | None],
) -> str:
    arm, period, scenario, output_root, training_manifest = args
    return str(replay_cell(
        arm=arm,
        period=period,
        scenario=scenario,
        output_root=output_root,
        training_manifest=(
            training_manifest if arm in LEARNER_ARMS else None
        ),
    ))


def replay_all(
    output_root: Path | str | None = None,
    training_manifest: Path | str | None = None,
    workers: int = 1,
) -> Path:
    """Idempotently run the exact 64 registered cells, then bind the suite.

    Existing complete or explicit technical-failure reports are verified and
    retained. Exceptions are never auto-converted into failures; the caller
    must use :func:`write_technical_failure` with a predeclared cause.
    """

    _reject_august_lexically(output_root or "", training_manifest or "")
    count = int(workers)
    if count < 1:
        raise ValueError("workers must be positive")
    root = _resolve_output_root(output_root)
    if training_manifest is None:
        training_path = root / "complete_training_manifest.json"
    else:
        supplied = Path(training_manifest)
        training_path = (
            supplied.resolve() if supplied.is_absolute()
            else resolve_project_path(supplied.as_posix())
        )

    # The frozen protocol requires all 15 development jobs to be complete and
    # bound before *any* new historical evaluation, including deterministic
    # rules. This check opens no May/June/July session data.
    from .training import verify_complete_training_manifest

    verify_complete_training_manifest(training_path)
    jobs = [
        (arm, period, scenario, str(root), str(training_path))
        for arm in BASELINE_ARMS
        for period in PERIODS
        for scenario in SCENARIOS
    ]
    if len(jobs) != 64:
        raise AssertionError("registered replay job matrix is not 64 cells")
    if count == 1:
        for job in jobs:
            _run_replay_job(job)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=count) as executor:
            list(executor.map(_run_replay_job, jobs))
    return build_complete_evaluation_manifest(root)


def evaluation_status(
    output_root: Path | str | None = None,
) -> dict[str, object]:
    """Return result-blind artifact status for all 64 registered cells.

    The function reports only identity, artifact state, hashes, and counts. It
    never returns P&L, ranks, actions, scores, fills, or any other outcome used
    for economic selection, and it does not open a market-data manifest.
    """

    _reject_august_lexically(output_root or "")
    root = _resolve_output_root(output_root)
    records: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for arm in BASELINE_ARMS:
        for period in PERIODS:
            for scenario in SCENARIOS:
                cell = root / "evaluation" / period / scenario / arm
                report_path = cell / "report.json"
                partial_path = cell.with_name(f".{cell.name}.part")
                artifact_type: str | None = None
                report_sha256: str | None = None
                if report_path.is_file():
                    try:
                        header = read_json_object(report_path)
                        artifact_type = str(header.get("artifact_type", ""))
                        identity_ok = (
                            header.get("arm") == arm
                            and header.get("period") == period
                            and header.get("scenario") == scenario
                        )
                        if (
                            identity_ok
                            and artifact_type == REPLAY_ARTIFACT
                            and header.get("status") == "COMPLETE"
                        ):
                            status = "COMPLETE"
                        elif (
                            identity_ok
                            and artifact_type == TECHNICAL_FAILURE_ARTIFACT
                            and header.get("status") == "TECHNICAL_FAILURE"
                        ):
                            status = "TECHNICAL_FAILURE"
                        else:
                            status = "INVALID_REPORT"
                        report_sha256 = file_sha256(report_path)
                    except (OSError, TypeError, ValueError):
                        status = "INVALID_REPORT"
                    if partial_path.exists():
                        status = "RECORDED_WITH_PARTIAL_CONFLICT"
                elif partial_path.exists():
                    status = "PARTIAL_RETAINED"
                elif cell.exists():
                    status = "INCOMPLETE_CELL"
                else:
                    status = "NOT_STARTED"
                counts[status] = counts.get(status, 0) + 1
                records.append({
                    "arm": arm,
                    "period": period,
                    "scenario": scenario,
                    "status": status,
                    "report_path": report_path.as_posix(),
                    "report_sha256": report_sha256,
                    "artifact_type": artifact_type,
                })
    recorded = counts.get("COMPLETE", 0) + counts.get("TECHNICAL_FAILURE", 0)
    return {
        "experiment_id": PROTOCOL_ID,
        "required_cells": 64,
        "recorded_cells": recorded,
        "counts": dict(sorted(counts.items())),
        "records": records,
        "result_blind": True,
        "historical_economic_values_returned": False,
        "market_data_opened": False,
        "august_accessed": False,
    }


__all__ = [
    "COMPLETE_EVALUATION_ARTIFACT",
    "DETERMINISTIC_RULE",
    "ENSEMBLE_RULE",
    "LearnedEnsemble",
    "LearnedMember",
    "PERIOD_DATASETS",
    "REPLAY_ARTIFACT",
    "TECHNICAL_FAILURE_ARTIFACT",
    "TECHNICAL_FAILURE_REASONS",
    "audit_complete_evaluation_against_source",
    "audit_replay_report_against_source",
    "build_complete_evaluation_manifest",
    "evaluation_status",
    "legal_masked_vote",
    "replay_cell",
    "replay_all",
    "verify_complete_evaluation_manifest",
    "verify_replay_report",
    "verify_technical_failure_report",
    "write_technical_failure",
]
