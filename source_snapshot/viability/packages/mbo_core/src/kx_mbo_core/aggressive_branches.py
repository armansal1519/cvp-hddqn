from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json
from .config import enforce_2025_frame
from .execution import SelfFinancingLedger
from .types import Fill


BASE_ACTION = "base_follow_ioc"
SKIP_ACTION = "skip"
WAIT_250MS_ACTION = "wait_250ms_ioc"
WAIT_1S_ACTION = "wait_1s_ioc"
AGGRESSIVE_ACTIONS = (
    BASE_ACTION,
    SKIP_ACTION,
    WAIT_250MS_ACTION,
    WAIT_1S_ACTION,
)


@dataclass(frozen=True)
class BranchScenario:
    name: str
    fee_usd_per_side: float
    adverse_ticks_per_filled_side: int = 0
    route_latency_ns: int = 5_000_000

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name cannot be empty")
        if (
            self.fee_usd_per_side < 0
            or self.adverse_ticks_per_filled_side < 0
            or self.route_latency_ns < 0
        ):
            raise ValueError("scenario costs cannot be negative")


NOMINAL_SCENARIO = BranchScenario("nominal", 3.0, 0, 5_000_000)
ADVERSE_FILL_SCENARIO = BranchScenario("adverse_fill", 3.0, 1, 5_000_000)
JOINT_STRESS_SCENARIO = BranchScenario("joint_stress", 5.0, 1, 20_000_000)
DEFAULT_SCENARIOS = (
    NOMINAL_SCENARIO,
    ADVERSE_FILL_SCENARIO,
    JOINT_STRESS_SCENARIO,
)


ROOT_COLUMNS = {
    "opportunity_id",
    "date",
    "partition",
    "fold",
    "decision_panel",
    "calendar_block",
    "direction",
    "quantity",
    "decision_time_ns",
    "available_time_ns",
    "terminal_arrival_time_ns",
    "terminal_time_ns",
    "raw_symbol",
    "publisher_id",
    "instrument_id",
    "continuity_segment_id",
    "tape_first_ordinal",
    "tape_last_ordinal",
    "source_tape_hash",
}
OBSERVATION_COLUMNS = {
    "opportunity_id",
    "scenario",
    "action",
    "phase",
    "arrival_time_ns",
    "state_available_time_ns",
    "state_global_record_ordinal",
    "next_group_available_time_ns",
    "next_group_first_ordinal",
    "event_group_complete",
    "publisher_id",
    "instrument_id",
    "raw_symbol",
    "continuity_segment_id",
    "source_tape_hash",
    "state_available",
    "capacity_permitted",
    "best_bid_ticks",
    "best_ask_ticks",
    "best_bid_size",
    "best_ask_size",
}
TRACE_COLUMNS = (
    "opportunity_id",
    "scenario",
    "action",
    "fill_index",
    "fill_role",
    "time_ns",
    "signed_quantity",
    "price_ticks",
    "fee_usd",
    "liquidity",
    "source_tape_hash",
)


@dataclass(frozen=True)
class AggressiveBranchResult:
    payoffs: pd.DataFrame
    traces: pd.DataFrame


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def _strict_bool(value: Any, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{column} is not boolean: {value!r}")


def _hash_records(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(records), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _fill_price(row: pd.Series, *, direction: int, adverse_ticks: int) -> int:
    if direction > 0:
        price = int(row["best_ask_ticks"])
        size = int(row["best_ask_size"])
    else:
        price = int(row["best_bid_ticks"])
        size = int(row["best_bid_size"])
    if size < 1:
        raise ValueError("one-lot IOC has no displayed best-level quantity")
    return price + direction * int(adverse_ticks)


def _trace_records(
    root: pd.Series,
    observations: pd.DataFrame,
    *,
    action: str,
    scenario: BranchScenario,
) -> tuple[list[dict[str, Any]], bool, str]:
    if action == SKIP_ACTION:
        return [], True, "registered_no_order_veto"
    entry = observations.loc[
        observations["phase"].astype(str).eq("entry")
        & observations["action"].astype(str).eq(action)
    ]
    terminal = observations.loc[observations["phase"].astype(str).eq("terminal")]
    if len(entry) != 1 or len(terminal) != 1:
        raise ValueError(f"{action!r} needs one entry and one terminal observation")
    entry_row = entry.iloc[0]
    terminal_row = terminal.iloc[0]
    for label, row in (("entry", entry_row), ("terminal", terminal_row)):
        if str(row["raw_symbol"]) != str(root["raw_symbol"]):
            raise ValueError(f"{label} observation is for the wrong symbol")
        if int(row["continuity_segment_id"]) != int(root["continuity_segment_id"]):
            raise ValueError(f"{label} observation crosses continuity")
        if int(row["publisher_id"]) != int(root["publisher_id"]):
            raise ValueError(f"{label} observation is for the wrong publisher")
        if int(row["instrument_id"]) != int(root["instrument_id"]):
            raise ValueError(f"{label} observation is for the wrong instrument")
        if str(row["source_tape_hash"]) != str(root["source_tape_hash"]):
            raise ValueError(f"{label} observation is from the wrong source tape")
        state_time = int(row["state_available_time_ns"])
        arrival = int(row["arrival_time_ns"])
        if state_time > arrival:
            raise ValueError(f"{label} observation uses a future book state")
        ordinal = int(row["state_global_record_ordinal"])
        if not int(root["tape_first_ordinal"]) <= ordinal <= int(root["tape_last_ordinal"]):
            raise ValueError(f"{label} observation ordinal exceeds the root tape")
        if not _strict_bool(row["event_group_complete"], "event_group_complete"):
            raise ValueError(f"{label} observation is not a complete atomic group")
        next_time = int(row["next_group_available_time_ns"])
        next_ordinal = int(row["next_group_first_ordinal"])
        if not (state_time <= arrival < next_time):
            raise ValueError(f"{label} observation is not the latest state at arrival")
        if not (ordinal < next_ordinal <= int(root["tape_last_ordinal"])):
            raise ValueError(f"{label} next-group cursor is outside the root tape")
    entry_delay = {
        BASE_ACTION: 0,
        WAIT_250MS_ACTION: 250_000_000,
        WAIT_1S_ACTION: 1_000_000_000,
    }[action]
    expected_entry_arrival = (
        int(root["decision_time_ns"]) + entry_delay + scenario.route_latency_ns
    )
    if int(entry_row["arrival_time_ns"]) != expected_entry_arrival:
        raise ValueError(f"{action!r} arrival violates its registered latency")
    expected_terminal_arrival = int(root["terminal_time_ns"]) + scenario.route_latency_ns
    if int(terminal_row["arrival_time_ns"]) != expected_terminal_arrival:
        raise ValueError("terminal observation violates the scenario route latency")
    if expected_terminal_arrival > int(root["terminal_arrival_time_ns"]):
        raise ValueError("scenario terminal exceeds the frozen replay window")
    if not _strict_bool(entry_row["state_available"], "state_available"):
        return [], False, "entry_state_unavailable"
    if action != BASE_ACTION and not _strict_bool(
        entry_row["capacity_permitted"], "capacity_permitted"
    ):
        # The registered delay policy remains flat when its future arrival
        # shield fails; this is part of the action, not a retroactive filter.
        return [], True, "arrival_capacity_shield_flat"
    direction = int(root["direction"])
    quantity = int(root["quantity"])
    if direction not in (-1, 1) or quantity != 1:
        raise ValueError("aggressive RENT-lite supports one signed GC contract")
    entry_price = _fill_price(
        entry_row,
        direction=direction,
        adverse_ticks=scenario.adverse_ticks_per_filled_side,
    )
    if not _strict_bool(terminal_row["state_available"], "state_available"):
        raise ValueError("risk-reducing terminal state is unavailable")
    exit_price = _fill_price(
        terminal_row,
        direction=-direction,
        adverse_ticks=scenario.adverse_ticks_per_filled_side,
    )
    common = {
        "opportunity_id": str(root["opportunity_id"]),
        "scenario": scenario.name,
        "action": action,
        "source_tape_hash": str(root["source_tape_hash"]),
        "liquidity": "aggressive",
    }
    records = [
        {
            **common,
            "fill_index": 0,
            "fill_role": "entry",
            "time_ns": int(entry_row["arrival_time_ns"]),
            "signed_quantity": direction,
            "price_ticks": entry_price,
            "fee_usd": float(scenario.fee_usd_per_side),
        },
        {
            **common,
            "fill_index": 1,
            "fill_role": "terminal_exit",
            "time_ns": int(terminal_row["arrival_time_ns"]),
            "signed_quantity": -direction,
            "price_ticks": exit_price,
            "fee_usd": float(scenario.fee_usd_per_side),
        },
    ]
    if records[0]["time_ns"] >= records[1]["time_ns"]:
        raise ValueError("entry reaches at or after the common terminal")
    return records, True, "filled_and_forced_flat"


def _net_ticks(records: Sequence[Mapping[str, Any]], *, tick_value_usd: float) -> float:
    ledger = SelfFinancingLedger(tick_value_usd=tick_value_usd)
    for row in records:
        ledger.apply_fill(
            Fill(
                time_ns=int(row["time_ns"]),
                signed_quantity=int(row["signed_quantity"]),
                price_ticks=int(row["price_ticks"]),
                fee_usd=float(row["fee_usd"]),
                liquidity=str(row["liquidity"]),
            )
        )
    if ledger.position != 0:
        raise AssertionError("complete aggressive branch does not end flat")
    return float(ledger.cash_usd / tick_value_usd)


def build_aggressive_branches(
    roots: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    scenarios: Sequence[BranchScenario] = DEFAULT_SCENARIOS,
    tick_value_usd: float = 10.0,
) -> AggressiveBranchResult:
    """Create dense, complete one-lot branch traces on a prepared causal tape."""

    _require(roots, ROOT_COLUMNS, "roots")
    _require(observations, OBSERVATION_COLUMNS, "observations")
    root_frame = enforce_2025_frame(roots.copy(), date_column="date")
    observation_frame = observations.copy()
    if root_frame["opportunity_id"].duplicated().any():
        raise ValueError("root opportunity IDs must be unique")
    if set(observation_frame["opportunity_id"]) - set(root_frame["opportunity_id"]):
        raise ValueError("observations reference unknown roots")
    hashes = root_frame["source_tape_hash"].astype(str)
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("root source_tape_hash must be a lowercase SHA-256")
    for column in (
        "available_time_ns",
        "decision_time_ns",
        "terminal_time_ns",
        "terminal_arrival_time_ns",
        "tape_first_ordinal",
        "tape_last_ordinal",
    ):
        root_frame[column] = pd.to_numeric(root_frame[column], errors="raise").astype("int64")
    valid_root_clock = (
        (root_frame["available_time_ns"] <= root_frame["decision_time_ns"])
        & (root_frame["decision_time_ns"] < root_frame["terminal_time_ns"])
        & (root_frame["terminal_time_ns"] <= root_frame["terminal_arrival_time_ns"])
        & (root_frame["tape_first_ordinal"] <= root_frame["tape_last_ordinal"])
    )
    if not bool(valid_root_clock.all()):
        raise ValueError("root causal clock or tape bounds are invalid")
    trace_rows: list[dict[str, Any]] = []
    payoff_rows: list[dict[str, Any]] = []
    staged: dict[tuple[str, str, str], tuple[list[dict[str, Any]], bool, str]] = {}
    for _, root in root_frame.iterrows():
        root_observations = observation_frame.loc[
            observation_frame["opportunity_id"].astype(str).eq(str(root["opportunity_id"]))
        ]
        for scenario in scenarios:
            selected = root_observations.loc[
                root_observations["scenario"].astype(str).eq(scenario.name)
            ]
            for action in AGGRESSIVE_ACTIONS:
                staged[(str(root["opportunity_id"]), scenario.name, action)] = _trace_records(
                    root, selected, action=action, scenario=scenario
                )
    for _, root in root_frame.iterrows():
        opportunity_id = str(root["opportunity_id"])
        for scenario in scenarios:
            base_records, base_feasible, _ = staged[(opportunity_id, scenario.name, BASE_ACTION)]
            if not base_feasible:
                raise ValueError("the frozen Base trace must be executable")
            base_ticks = _net_ticks(base_records, tick_value_usd=tick_value_usd)
            for action in AGGRESSIVE_ACTIONS:
                records, feasible, reason = staged[(opportunity_id, scenario.name, action)]
                if not feasible:
                    records = [dict(row, action=action) for row in base_records]
                    reason = f"exact_base_fallback:{reason}"
                ticks = _net_ticks(records, tick_value_usd=tick_value_usd)
                trace_hash = _hash_records(records or [{"action": action, "reason": reason}])
                trace_rows.extend(records)
                payoff_rows.append(
                    {
                        "opportunity_id": opportunity_id,
                        "date": root["date"],
                        "partition": str(root["partition"]),
                        "fold": str(root["fold"]),
                        "decision_panel": str(root["decision_panel"]),
                        "calendar_block": str(root["calendar_block"]),
                        "library": "rent",
                        "action": action,
                        "scenario": scenario.name,
                        "feature_view": "common_tape",
                        "feasible": bool(feasible),
                        "infeasible_reason": "" if feasible else reason,
                        "net_ticks": ticks,
                        "base_net_ticks": base_ticks,
                        "downside_ticks": max(0.0, base_ticks - ticks),
                        "filled_quantity": int(bool(records)),
                        "forced_quantity": int(bool(records)),
                        "retains_base_trade": action not in {SKIP_ACTION} and bool(records),
                        "worker_subset": (
                            "empty"
                            if action == BASE_ACTION
                            else "entry"
                            if action == SKIP_ACTION
                            else "execution"
                        ),
                        "passive_fifo_certified": False,
                        "replay_start_time_ns": int(root["available_time_ns"]),
                        "replay_end_time_ns": int(root["terminal_arrival_time_ns"]),
                        "source_tape_hash": str(root["source_tape_hash"]),
                        "branch_trace_hash": trace_hash,
                        "branch_outcome_reason": reason,
                    }
                )
    traces = pd.DataFrame(trace_rows, columns=TRACE_COLUMNS)
    payoffs = pd.DataFrame(payoff_rows)
    expected = len(root_frame) * len(scenarios) * len(AGGRESSIVE_ACTIONS)
    if len(payoffs) != expected:
        raise AssertionError("aggressive branch lattice is incomplete")
    return AggressiveBranchResult(payoffs=payoffs, traces=traces)


def replay_policy_plan(
    roots: pd.DataFrame,
    branch_result: AggressiveBranchResult,
    plan: pd.DataFrame,
    *,
    policy_id: str,
    model_artifact_hash: str,
    policy_config_hash: str,
    tick_value_usd: float = 10.0,
) -> pd.DataFrame:
    """Replay selected fill traces through one persistent chronological ledger."""

    required_plan = {
        "opportunity_id",
        "scenario",
        "selected_action",
        "feature_view",
        "training_cutoff_time_ns",
        "policy_decision_time_ns",
    }
    _require(plan, required_plan, "policy plan")
    root_frame = enforce_2025_frame(roots.copy(), date_column="date")
    if plan.duplicated(["opportunity_id", "scenario"]).any():
        raise ValueError("policy plan repeats a root/scenario")
    root_index = root_frame.set_index("opportunity_id")
    payoff_index = branch_result.payoffs.set_index(["opportunity_id", "scenario", "action"])
    trace_groups = {
        key: group.sort_values("fill_index", kind="stable")
        for key, group in branch_result.traces.groupby(
            ["opportunity_id", "scenario", "action"], sort=False
        )
    }
    selected = plan.copy()
    selected["_decision"] = selected["opportunity_id"].map(root_index["decision_time_ns"])
    selected = selected.sort_values(["_decision", "opportunity_id", "scenario"], kind="stable")
    ledger = SelfFinancingLedger(tick_value_usd=tick_value_usd)
    rows: list[dict[str, Any]] = []
    prior_end_by_scenario: dict[str, int] = {}
    for item in selected.itertuples(index=False):
        root = root_index.loc[str(item.opportunity_id)]
        scenario = str(item.scenario)
        action = str(item.selected_action)
        key = (str(item.opportunity_id), scenario, action)
        if key not in payoff_index.index:
            raise ValueError(f"policy selected an unregistered branch: {key}")
        payoff = payoff_index.loc[key]
        if isinstance(payoff, pd.DataFrame):
            raise ValueError(f"branch payoff is duplicated: {key}")
        if int(item.policy_decision_time_ns) != int(root["decision_time_ns"]):
            raise ValueError("policy decision clock differs from the frozen root")
        if int(item.training_cutoff_time_ns) >= int(item.policy_decision_time_ns):
            raise ValueError("policy training cutoff must precede decision")
        start = int(root["available_time_ns"])
        end = int(root["terminal_arrival_time_ns"])
        if scenario in prior_end_by_scenario and start < prior_end_by_scenario[scenario]:
            raise ValueError("policy roots overlap within a scenario")
        prior_end_by_scenario[scenario] = end
        before = ledger.cash_usd
        trace = trace_groups.get(key, pd.DataFrame(columns=TRACE_COLUMNS))
        for fill_row in trace.itertuples(index=False):
            ledger.apply_fill(
                Fill(
                    time_ns=int(fill_row.time_ns),
                    signed_quantity=int(fill_row.signed_quantity),
                    price_ticks=int(fill_row.price_ticks),
                    fee_usd=float(fill_row.fee_usd),
                    liquidity=str(fill_row.liquidity),
                )
            )
        if ledger.position != 0:
            raise AssertionError("selected policy branch did not restore flat inventory")
        net_ticks = float((ledger.cash_usd - before) / tick_value_usd)
        policy_trace_hash = _hash_records(
            trace.to_dict("records")
            or [{"opportunity_id": str(item.opportunity_id), "action": action, "flat": True}]
        )
        rows.append(
            {
                "opportunity_id": str(item.opportunity_id),
                "date": root["date"],
                "partition": str(root["partition"]),
                "fold": str(root["fold"]),
                "decision_panel": str(root["decision_panel"]),
                "calendar_block": str(root["calendar_block"]),
                "policy_id": str(policy_id),
                "scenario": scenario,
                "payoff_library": "rent",
                "selected_action": action,
                "feature_view": str(item.feature_view),
                "net_ticks": net_ticks,
                "base_net_ticks": float(payoff["base_net_ticks"]),
                "downside_ticks": max(0.0, float(payoff["base_net_ticks"]) - net_ticks),
                "intervened": bool(payoff["feasible"]) and action != BASE_ACTION,
                "base_trade_retained": (
                    True
                    if not bool(payoff["feasible"])
                    else action not in {SKIP_ACTION} and bool(len(trace))
                ),
                "replay_complete": True,
                "model_artifact_hash": str(model_artifact_hash),
                "policy_config_hash": str(policy_config_hash),
                "training_cutoff_time_ns": int(item.training_cutoff_time_ns),
                "policy_decision_time_ns": int(item.policy_decision_time_ns),
                "replay_start_time_ns": start,
                "replay_end_time_ns": end,
                "policy_trace_hash": policy_trace_hash,
                "source_tape_hash": str(root["source_tape_hash"]),
            }
        )
    return pd.DataFrame(rows)


def write_aggressive_branch_result(
    path: str | Path,
    result: AggressiveBranchResult,
    *,
    roots_sha256: str,
    observations_sha256: str,
    evidence_class: str = "DEVELOPMENT_ONLY",
) -> dict[str, Any]:
    """Atomically persist prepared branch evidence without claiming certification."""

    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"aggressive branch output already exists: {target}")
    if evidence_class not in {"DEVELOPMENT_ONLY", "SYNTHETIC_ONLY"}:
        raise ValueError("prepared branch writer cannot create locked/final evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        payoffs_path = temporary / "payoffs.parquet"
        traces_path = temporary / "traces.parquet"
        result.payoffs.to_parquet(payoffs_path, index=False)
        result.traces.to_parquet(traces_path, index=False)
        manifest = {
            "schema_version": 1,
            "product_type": "kx_prepared_aggressive_branch_result",
            "evidence_class": evidence_class,
            "claim_status": "UNCERTIFIED_PREPARED_PRODUCT",
            "inputs": {
                "roots_sha256": str(roots_sha256),
                "observations_sha256": str(observations_sha256),
            },
            "files": {
                "payoffs": {
                    "path": "payoffs.parquet",
                    "rows": int(len(result.payoffs)),
                    "sha256": sha256_file(payoffs_path),
                },
                "traces": {
                    "path": "traces.parquet",
                    "rows": int(len(result.traces)),
                    "sha256": sha256_file(traces_path),
                },
            },
        }
        write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    return manifest
