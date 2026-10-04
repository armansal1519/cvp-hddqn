from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .aggressive_branches import BranchScenario, DEFAULT_SCENARIOS
from .artifacts import sha256_file, write_json
from .config import enforce_2025_frame
from .execution import SelfFinancingLedger
from .globex import require_globex_dates_at_ns
from .types import Fill


RENT_WORKERS = ("entry", "execution", "protection")
EMPTY_WORKER_SUBSET = "empty"
ENTRY_ACTIONS = ("follow_base", "skip")
EXECUTION_ACTIONS = ("submit", "wait_250ms_ioc", "wait_1s_ioc", "abort")
PROTECTION_ACTIONS = ("hold", "flatten_early")
BASE_ENTRY_ACTION = ENTRY_ACTIONS[0]
BASE_EXECUTION_ACTION = EXECUTION_ACTIONS[0]
BASE_PROTECTION_ACTION = PROTECTION_ACTIONS[0]
RENT_ACTION_PREFIX = "rent.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_worker_subset(workers: str | Iterable[str]) -> str:
    """Return the sole accepted spelling for one of the eight manager actions."""

    if isinstance(workers, str):
        text = workers.strip().lower()
        if text == EMPTY_WORKER_SUBSET:
            values: set[str] = set()
        elif not text:
            raise ValueError("worker_subset cannot be empty; use 'empty' for Base")
        else:
            parts = text.split("+")
            if any(not part for part in parts):
                raise ValueError(f"invalid worker_subset {workers!r}")
            values = set(parts)
            if len(values) != len(parts):
                raise ValueError(f"worker_subset repeats a worker: {workers!r}")
    else:
        values = {str(worker).strip().lower() for worker in workers}
        if "" in values:
            raise ValueError("worker_subset contains an empty worker")
    unknown = sorted(values - set(RENT_WORKERS))
    if unknown:
        raise ValueError(f"unknown RENT workers: {unknown}")
    return "+".join(worker for worker in RENT_WORKERS if worker in values) or EMPTY_WORKER_SUBSET


def worker_subset_members(value: str) -> tuple[str, ...]:
    canonical = canonical_worker_subset(value)
    if canonical != str(value).strip().lower():
        raise ValueError(
            f"worker_subset is not canonical: {value!r}; expected {canonical!r}"
        )
    if canonical == EMPTY_WORKER_SUBSET:
        return ()
    return tuple(canonical.split("+"))


REGISTERED_WORKER_SUBSETS = tuple(
    canonical_worker_subset(
        worker for bit, worker in enumerate(RENT_WORKERS) if mask & (1 << bit)
    )
    for mask in range(8)
)


@dataclass(frozen=True, order=True)
class RentActionContract:
    worker_subset: str
    entry_action: str
    execution_action: str
    protection_action: str

    def __post_init__(self) -> None:
        members = set(worker_subset_members(self.worker_subset))
        if self.entry_action not in ENTRY_ACTIONS:
            raise ValueError(f"unknown entry action {self.entry_action!r}")
        if self.execution_action not in EXECUTION_ACTIONS:
            raise ValueError(f"unknown execution action {self.execution_action!r}")
        if self.protection_action not in PROTECTION_ACTIONS:
            raise ValueError(f"unknown protection action {self.protection_action!r}")
        defaults = {
            "entry": (self.entry_action, BASE_ENTRY_ACTION),
            "execution": (self.execution_action, BASE_EXECUTION_ACTION),
            "protection": (self.protection_action, BASE_PROTECTION_ACTION),
        }
        for worker, (observed, default) in defaults.items():
            if worker not in members and observed != default:
                raise ValueError(
                    f"inactive {worker} worker must use Base action {default!r}"
                )

    @property
    def action_id(self) -> str:
        return (
            f"{RENT_ACTION_PREFIX}|subset={self.worker_subset}"
            f"|entry={self.entry_action}|execution={self.execution_action}"
            f"|protection={self.protection_action}"
        )


def _registered_action_contracts() -> tuple[RentActionContract, ...]:
    contracts: list[RentActionContract] = []
    for subset in REGISTERED_WORKER_SUBSETS:
        active = set(worker_subset_members(subset))
        entries = ENTRY_ACTIONS if "entry" in active else (BASE_ENTRY_ACTION,)
        executions = (
            EXECUTION_ACTIONS if "execution" in active else (BASE_EXECUTION_ACTION,)
        )
        protections = (
            PROTECTION_ACTIONS if "protection" in active else (BASE_PROTECTION_ACTION,)
        )
        contracts.extend(
            RentActionContract(subset, entry, execution, protection)
            for entry, execution, protection in product(entries, executions, protections)
        )
    return tuple(contracts)


DENSE_RENT_ACTION_CONTRACTS = _registered_action_contracts()
DENSE_RENT_ACTION_IDS = tuple(contract.action_id for contract in DENSE_RENT_ACTION_CONTRACTS)
_CONTRACT_BY_ACTION = {contract.action_id: contract for contract in DENSE_RENT_ACTION_CONTRACTS}
BASE_RENT_ACTION = RentActionContract(
    EMPTY_WORKER_SUBSET,
    BASE_ENTRY_ACTION,
    BASE_EXECUTION_ACTION,
    BASE_PROTECTION_ACTION,
)


def parse_rent_action_id(value: str) -> RentActionContract:
    text = str(value).strip().lower()
    contract = _CONTRACT_BY_ACTION.get(text)
    if contract is None:
        raise ValueError(f"unregistered dense RENT action {value!r}")
    if contract.action_id != text:
        raise AssertionError("registered action identifier is not canonical")
    return contract


ROOT_COLUMNS = {
    "opportunity_id",
    "date",
    "partition",
    "fold",
    "decision_panel",
    "calendar_block",
    "direction",
    "quantity",
    "available_time_ns",
    "decision_time_ns",
    "protection_time_ns",
    "terminal_time_ns",
    "terminal_arrival_time_ns",
    "publisher_id",
    "instrument_id",
    "raw_symbol",
    "continuity_segment_id",
    "tape_first_ordinal",
    "tape_last_ordinal",
    "source_tape_hash",
}
OBSERVATION_COLUMNS = {
    "opportunity_id",
    "scenario",
    "phase",
    "primitive_action",
    "arrival_time_ns",
    "state_as_of_time_ns",
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
    "prepared_observation_hash",
}
CONTINUATION_COLUMNS = {
    "opportunity_id",
    "scenario",
    "continuation_start_time_ns",
    "continuation_end_time_ns",
    "first_ordinal",
    "last_ordinal",
    "publisher_id",
    "instrument_id",
    "raw_symbol",
    "continuity_segment_id",
    "source_tape_hash",
    "complete",
    "independent_of_action",
    "continuation_trace_hash",
}

_OBSERVATION_HASH_COLUMNS = (
    "opportunity_id",
    "scenario",
    "phase",
    "primitive_action",
    "arrival_time_ns",
    "state_as_of_time_ns",
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
)

TRACE_COLUMNS = (
    "opportunity_id",
    "scenario",
    "action",
    "worker_subset",
    "entry_action",
    "execution_action",
    "protection_action",
    "action_binding_hash",
    "continuation_trace_hash",
    "event_index",
    "event_kind",
    "event_role",
    "time_ns",
    "signed_quantity",
    "price_ticks",
    "fee_usd",
    "liquidity",
    "source_tape_hash",
)

DENSE_RENT_PAYOFF_COLUMNS = (
    "entry_action",
    "execution_action",
    "protection_action",
    "action_binding_hash",
    "branch_binding_hash",
    "continuation_trace_hash",
)


@dataclass(frozen=True)
class DenseRentBranchResult:
    roots: pd.DataFrame
    observations: pd.DataFrame
    continuations: pd.DataFrame
    payoffs: pd.DataFrame
    traces: pd.DataFrame


def _require(frame: pd.DataFrame, columns: set[str] | Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{name} is missing columns: {missing}")


def _strict_bool(value: Any, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{column} is not a strict boolean: {value!r}")


def _json_hash(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _observation_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    integers = {
        "arrival_time_ns",
        "state_as_of_time_ns",
        "state_available_time_ns",
        "state_global_record_ordinal",
        "next_group_available_time_ns",
        "next_group_first_ordinal",
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "best_bid_ticks",
        "best_ask_ticks",
        "best_bid_size",
        "best_ask_size",
    }
    booleans = {"event_group_complete", "state_available", "capacity_permitted"}
    payload: dict[str, Any] = {}
    for column in _OBSERVATION_HASH_COLUMNS:
        value = row[column]
        if column in integers:
            payload[column] = int(value)
        elif column in booleans:
            payload[column] = _strict_bool(value, column)
        else:
            payload[column] = str(value)
    return payload


def prepared_observation_hash(row: Mapping[str, Any]) -> str:
    """Hash the causal observation and all of its tape/source coordinates."""

    return _json_hash(_observation_payload(row))


def dense_action_binding_hash(
    *,
    opportunity_id: str,
    scenario: str,
    action: str,
    publisher_id: int,
    instrument_id: int,
    raw_symbol: str,
    continuity_segment_id: int,
    source_tape_hash: str,
    continuation_trace_hash: str,
) -> str:
    contract = parse_rent_action_id(action)
    payload = {
        "action": contract.action_id,
        "continuation_trace_hash": str(continuation_trace_hash),
        "continuity_segment_id": int(continuity_segment_id),
        "instrument_id": int(instrument_id),
        "opportunity_id": str(opportunity_id),
        "publisher_id": int(publisher_id),
        "raw_symbol": str(raw_symbol),
        "scenario": str(scenario),
        "source_tape_hash": str(source_tape_hash),
        "worker_subset": contract.worker_subset,
    }
    return _json_hash(payload)


def dense_branch_binding_hash(*, action_binding_hash: str, branch_trace_hash: str) -> str:
    return _json_hash(
        {
            "action_binding_hash": str(action_binding_hash),
            "branch_trace_hash": str(branch_trace_hash),
        }
    )


def _normalise_trace_records(records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(records, pd.DataFrame):
        rows = records.sort_values("event_index", kind="stable").to_dict("records")
    else:
        rows = sorted(records, key=lambda row: int(row["event_index"]))
    integer_columns = {"event_index", "time_ns", "signed_quantity", "price_ticks"}
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for column in TRACE_COLUMNS:
            value = row[column]
            if column in integer_columns:
                item[column] = int(value)
            elif column == "fee_usd":
                item[column] = float(value)
            else:
                item[column] = str(value)
        output.append(item)
    return output


def dense_trace_content_hash(records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> str:
    return _json_hash(_normalise_trace_records(records))


def _validate_roots(roots: pd.DataFrame) -> pd.DataFrame:
    _require(roots, ROOT_COLUMNS, "dense RENT roots")
    output = enforce_2025_frame(
        roots.copy(),
        date_column="date",
        clock_columns=[
            "available_time_ns",
            "decision_time_ns",
            "protection_time_ns",
            "terminal_time_ns",
            "terminal_arrival_time_ns",
        ],
    )
    if output.empty or output["opportunity_id"].duplicated().any():
        raise ValueError("dense RENT roots must be nonempty and unique")
    for column in (
        "direction",
        "quantity",
        "available_time_ns",
        "decision_time_ns",
        "protection_time_ns",
        "terminal_time_ns",
        "terminal_arrival_time_ns",
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "tape_first_ordinal",
        "tape_last_ordinal",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    if not output["direction"].isin((-1, 1)).all() or not output["quantity"].eq(1).all():
        raise ValueError("dense RENT replay supports exactly one signed contract")
    clock_ok = (
        (output["available_time_ns"] <= output["decision_time_ns"])
        & (output["decision_time_ns"] < output["protection_time_ns"])
        & (output["protection_time_ns"] < output["terminal_time_ns"])
        & (output["terminal_time_ns"] <= output["terminal_arrival_time_ns"])
        & (output["tape_first_ordinal"] <= output["tape_last_ordinal"])
    )
    if not bool(clock_ok.all()):
        raise ValueError("dense RENT root causal clocks or tape bounds are invalid")
    output["date"] = require_globex_dates_at_ns(
        output["date"], output["decision_time_ns"], label="dense RENT root"
    ).to_numpy()
    if not output["source_tape_hash"].astype(str).str.fullmatch(SHA256_PATTERN).all():
        raise ValueError("dense RENT roots need lowercase source-tape SHA-256 values")
    for column in ("opportunity_id", "raw_symbol"):
        if output[column].isna().any() or output[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"dense RENT roots contain empty {column}")
    return output


def _validate_observation_row(root: pd.Series, row: pd.Series) -> None:
    for column in ("publisher_id", "instrument_id", "continuity_segment_id"):
        if int(row[column]) != int(root[column]):
            raise ValueError(f"prepared observation {column} disagrees with its root")
    for column in ("raw_symbol", "source_tape_hash"):
        if str(row[column]) != str(root[column]):
            raise ValueError(f"prepared observation {column} disagrees with its root")
    as_of = int(row["state_as_of_time_ns"])
    available = int(row["state_available_time_ns"])
    arrival = int(row["arrival_time_ns"])
    next_available = int(row["next_group_available_time_ns"])
    ordinal = int(row["state_global_record_ordinal"])
    next_ordinal = int(row["next_group_first_ordinal"])
    if not (as_of <= available <= arrival < next_available):
        raise ValueError("prepared observation violates as-of/availability/arrival causality")
    if not (
        int(root["tape_first_ordinal"])
        <= ordinal
        < next_ordinal
        <= int(root["tape_last_ordinal"])
    ):
        raise ValueError("prepared observation cursor exceeds its root tape")
    if not _strict_bool(row["event_group_complete"], "event_group_complete"):
        raise ValueError("prepared observation ends inside an atomic event group")
    expected_hash = prepared_observation_hash(row)
    if str(row["prepared_observation_hash"]) != expected_hash:
        raise ValueError("prepared observation hash does not bind its causal contents")


def _validate_observations(
    roots: pd.DataFrame,
    observations: pd.DataFrame,
    scenarios: Sequence[BranchScenario],
) -> pd.DataFrame:
    _require(observations, OBSERVATION_COLUMNS, "dense RENT observations")
    output = observations.copy()
    if output.empty or output.duplicated(
        ["opportunity_id", "scenario", "phase", "primitive_action"]
    ).any():
        raise ValueError("prepared observations must be nonempty and uniquely keyed")
    for column in (
        "arrival_time_ns",
        "state_as_of_time_ns",
        "state_available_time_ns",
        "state_global_record_ordinal",
        "next_group_available_time_ns",
        "next_group_first_ordinal",
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
        "best_bid_ticks",
        "best_ask_ticks",
        "best_bid_size",
        "best_ask_size",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    if len(scenario_by_name) != len(tuple(scenarios)):
        raise ValueError("dense RENT scenarios must have unique names")
    root_index = roots.set_index("opportunity_id")
    unknown_roots = set(output["opportunity_id"].astype(str)) - set(root_index.index.astype(str))
    if unknown_roots:
        raise ValueError(f"prepared observations reference unknown roots: {sorted(unknown_roots)[:5]}")
    required_cells = {
        *(('entry', action) for action in EXECUTION_ACTIONS if action != "abort"),
        ("protection", "flatten_early"),
        ("terminal", "hold"),
    }
    for opportunity_id, root in root_index.iterrows():
        for scenario_name, scenario in scenario_by_name.items():
            group = output.loc[
                output["opportunity_id"].astype(str).eq(str(opportunity_id))
                & output["scenario"].astype(str).eq(scenario_name)
            ]
            observed_cells = set(
                group[["phase", "primitive_action"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
            if observed_cells != required_cells:
                raise ValueError(
                    f"root {opportunity_id!r}, scenario {scenario_name!r} has an "
                    f"incomplete prepared-observation lattice"
                )
            expected_arrivals = {
                ("entry", "submit"): int(root["decision_time_ns"]) + scenario.route_latency_ns,
                ("entry", "wait_250ms_ioc"): int(root["decision_time_ns"])
                + 250_000_000
                + scenario.route_latency_ns,
                ("entry", "wait_1s_ioc"): int(root["decision_time_ns"])
                + 1_000_000_000
                + scenario.route_latency_ns,
                ("protection", "flatten_early"): int(root["protection_time_ns"])
                + scenario.route_latency_ns,
                ("terminal", "hold"): int(root["terminal_time_ns"])
                + scenario.route_latency_ns,
            }
            for _, row in group.iterrows():
                _validate_observation_row(root, row)
                key = (str(row["phase"]), str(row["primitive_action"]))
                if int(row["arrival_time_ns"]) != expected_arrivals[key]:
                    raise ValueError(f"prepared observation {key!r} violates registered latency")
                if int(row["arrival_time_ns"]) > int(root["terminal_arrival_time_ns"]):
                    raise ValueError("prepared observation exceeds the common replay window")
    return output


def _validate_continuations(
    roots: pd.DataFrame,
    observations: pd.DataFrame,
    continuations: pd.DataFrame,
    scenarios: Sequence[BranchScenario],
) -> pd.DataFrame:
    _require(continuations, CONTINUATION_COLUMNS, "dense RENT continuations")
    output = continuations.copy()
    if output.empty or output.duplicated(["opportunity_id", "scenario"]).any():
        raise ValueError("continuations must contain one unique row per root/scenario")
    for column in (
        "continuation_start_time_ns",
        "continuation_end_time_ns",
        "first_ordinal",
        "last_ordinal",
        "publisher_id",
        "instrument_id",
        "continuity_segment_id",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
    expected_keys = {
        (str(root), scenario.name)
        for root in roots["opportunity_id"]
        for scenario in scenarios
    }
    observed_keys = set(
        output[["opportunity_id", "scenario"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise ValueError("independent continuation lattice is incomplete")
    root_index = roots.set_index("opportunity_id")
    obs_index = observations.set_index(
        ["opportunity_id", "scenario", "phase", "primitive_action"]
    )
    for row in output.itertuples(index=False):
        root = root_index.loc[str(row.opportunity_id)]
        for column in ("publisher_id", "instrument_id", "continuity_segment_id"):
            if int(getattr(row, column)) != int(root[column]):
                raise ValueError(f"continuation {column} disagrees with its root")
        for column in ("raw_symbol", "source_tape_hash"):
            if str(getattr(row, column)) != str(root[column]):
                raise ValueError(f"continuation {column} disagrees with its root")
        if not _strict_bool(row.complete, "complete"):
            raise ValueError("continuation trace is incomplete")
        if not _strict_bool(row.independent_of_action, "independent_of_action"):
            raise ValueError("continuation trace is policy/action contaminated")
        if not SHA256_PATTERN.fullmatch(str(row.continuation_trace_hash)):
            raise ValueError("continuation trace hash must be lowercase SHA-256")
        protection = obs_index.loc[
            (str(row.opportunity_id), str(row.scenario), "protection", "flatten_early")
        ]
        terminal = obs_index.loc[
            (str(row.opportunity_id), str(row.scenario), "terminal", "hold")
        ]
        if int(row.continuation_start_time_ns) != int(protection["arrival_time_ns"]):
            raise ValueError("continuation does not start at the protection arrival")
        if int(row.continuation_end_time_ns) != int(terminal["arrival_time_ns"]):
            raise ValueError("continuation does not end at the common terminal arrival")
        if int(row.first_ordinal) != int(protection["next_group_first_ordinal"]):
            raise ValueError("continuation first ordinal skips or reuses a protection group")
        if int(row.last_ordinal) != int(terminal["state_global_record_ordinal"]):
            raise ValueError("continuation last ordinal does not bind the terminal state")
        if not int(row.first_ordinal) <= int(row.last_ordinal):
            raise ValueError("continuation ordinal bounds are reversed")
    return output


def _fill_price(row: pd.Series, *, direction: int, adverse_ticks: int) -> int:
    if direction > 0:
        price, size = int(row["best_ask_ticks"]), int(row["best_ask_size"])
    else:
        price, size = int(row["best_bid_ticks"]), int(row["best_bid_size"])
    if size < 1:
        raise ValueError("one-lot IOC has no displayed best-level quantity")
    return price + direction * int(adverse_ticks)


def _fill_records(
    *,
    root: pd.Series,
    scenario: BranchScenario,
    contract: RentActionContract,
    action_binding: str,
    continuation_hash: str,
    observations: pd.DataFrame,
) -> tuple[list[dict[str, Any]], bool, str]:
    common = {
        "opportunity_id": str(root["opportunity_id"]),
        "scenario": scenario.name,
        "action": contract.action_id,
        "worker_subset": contract.worker_subset,
        "entry_action": contract.entry_action,
        "execution_action": contract.execution_action,
        "protection_action": contract.protection_action,
        "action_binding_hash": action_binding,
        "continuation_trace_hash": continuation_hash,
        "source_tape_hash": str(root["source_tape_hash"]),
    }
    records: list[dict[str, Any]] = [
        {
            **common,
            "event_index": 0,
            "event_kind": "decision",
            "event_role": "branch_contract",
            "time_ns": int(root["decision_time_ns"]),
            "signed_quantity": 0,
            "price_ticks": 0,
            "fee_usd": 0.0,
            "liquidity": "none",
        }
    ]
    if contract.entry_action == "skip":
        return records, True, "entry_skip_flat"
    if contract.execution_action == "abort":
        return records, True, "execution_abort_flat"
    entry = observations.loc[
        observations["phase"].astype(str).eq("entry")
        & observations["primitive_action"].astype(str).eq(contract.execution_action)
    ].iloc[0]
    if not _strict_bool(entry["state_available"], "state_available"):
        return records, False, "entry_state_unavailable"
    if contract.execution_action != "submit" and not _strict_bool(
        entry["capacity_permitted"], "capacity_permitted"
    ):
        return records, True, "arrival_capacity_shield_flat"
    if contract.protection_action == "flatten_early":
        exit_row = observations.loc[
            observations["phase"].astype(str).eq("protection")
        ].iloc[0]
        exit_role = "protection_flatten"
    else:
        exit_row = observations.loc[observations["phase"].astype(str).eq("terminal")].iloc[0]
        exit_role = "terminal_exit"
    if not _strict_bool(exit_row["state_available"], "state_available"):
        return records, False, f"{exit_role}_state_unavailable"
    if int(entry["arrival_time_ns"]) >= int(exit_row["arrival_time_ns"]):
        return records, False, "entry_not_before_exit"
    direction = int(root["direction"])
    entry_price = _fill_price(
        entry,
        direction=direction,
        adverse_ticks=scenario.adverse_ticks_per_filled_side,
    )
    exit_price = _fill_price(
        exit_row,
        direction=-direction,
        adverse_ticks=scenario.adverse_ticks_per_filled_side,
    )
    records.extend(
        [
            {
                **common,
                "event_index": 1,
                "event_kind": "fill",
                "event_role": "entry",
                "time_ns": int(entry["arrival_time_ns"]),
                "signed_quantity": direction,
                "price_ticks": entry_price,
                "fee_usd": float(scenario.fee_usd_per_side),
                "liquidity": "aggressive",
            },
            {
                **common,
                "event_index": 2,
                "event_kind": "fill",
                "event_role": exit_role,
                "time_ns": int(exit_row["arrival_time_ns"]),
                "signed_quantity": -direction,
                "price_ticks": exit_price,
                "fee_usd": float(scenario.fee_usd_per_side),
                "liquidity": "aggressive",
            },
        ]
    )
    return records, True, "filled_and_flattened" if exit_role == "protection_flatten" else "filled_and_held"


def _retarget_base_fallback(
    base_records: Sequence[Mapping[str, Any]],
    *,
    contract: RentActionContract,
    action_binding: str,
    continuation_hash: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, source in enumerate(base_records):
        row = dict(source)
        row.update(
            {
                "action": contract.action_id,
                "worker_subset": contract.worker_subset,
                "entry_action": contract.entry_action,
                "execution_action": contract.execution_action,
                "protection_action": contract.protection_action,
                "action_binding_hash": action_binding,
                "continuation_trace_hash": continuation_hash,
                "event_index": index,
            }
        )
        if index == 0:
            row["event_role"] = "exact_base_fallback_contract"
        records.append(row)
    return records


def _net_ticks(records: Sequence[Mapping[str, Any]], *, tick_value_usd: float) -> float:
    ledger = SelfFinancingLedger(tick_value_usd=tick_value_usd)
    for row in records:
        if str(row["event_kind"]) != "fill":
            continue
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
        raise AssertionError("dense RENT branch does not restore flat inventory")
    return float(ledger.cash_usd / tick_value_usd)


def build_dense_rent_branches(
    roots: pd.DataFrame,
    observations: pd.DataFrame,
    continuations: pd.DataFrame,
    *,
    scenarios: Sequence[BranchScenario] = DEFAULT_SCENARIOS,
    tick_value_usd: float = 10.0,
) -> DenseRentBranchResult:
    """Replay all 45 legal subset/action contracts on each causal root/scenario.

    This produces an uncertified prepared product. It does not inspect a sealed
    outcome period and it does not establish that an HMADRL policy is profitable.
    """

    clean_roots = _validate_roots(roots)
    clean_observations = _validate_observations(clean_roots, observations, scenarios)
    clean_continuations = _validate_continuations(
        clean_roots, clean_observations, continuations, scenarios
    )
    continuation_index = clean_continuations.set_index(["opportunity_id", "scenario"])
    trace_rows: list[dict[str, Any]] = []
    payoff_rows: list[dict[str, Any]] = []
    for _, root in clean_roots.iterrows():
        opportunity_id = str(root["opportunity_id"])
        for scenario in scenarios:
            selected = clean_observations.loc[
                clean_observations["opportunity_id"].astype(str).eq(opportunity_id)
                & clean_observations["scenario"].astype(str).eq(scenario.name)
            ]
            continuation = continuation_index.loc[(opportunity_id, scenario.name)]
            continuation_hash = str(continuation["continuation_trace_hash"])
            staged: dict[str, tuple[list[dict[str, Any]], bool, str, str]] = {}
            for contract in DENSE_RENT_ACTION_CONTRACTS:
                action_binding = dense_action_binding_hash(
                    opportunity_id=opportunity_id,
                    scenario=scenario.name,
                    action=contract.action_id,
                    publisher_id=int(root["publisher_id"]),
                    instrument_id=int(root["instrument_id"]),
                    raw_symbol=str(root["raw_symbol"]),
                    continuity_segment_id=int(root["continuity_segment_id"]),
                    source_tape_hash=str(root["source_tape_hash"]),
                    continuation_trace_hash=continuation_hash,
                )
                records, feasible, reason = _fill_records(
                    root=root,
                    scenario=scenario,
                    contract=contract,
                    action_binding=action_binding,
                    continuation_hash=continuation_hash,
                    observations=selected,
                )
                staged[contract.action_id] = (records, feasible, reason, action_binding)
            base_records, base_feasible, _, _ = staged[BASE_RENT_ACTION.action_id]
            if not base_feasible or len(base_records) != 3:
                raise ValueError("frozen Base branch is not executable through the common terminal")
            base_ticks = _net_ticks(base_records, tick_value_usd=tick_value_usd)
            for contract in DENSE_RENT_ACTION_CONTRACTS:
                records, feasible, reason, action_binding = staged[contract.action_id]
                if not feasible:
                    records = _retarget_base_fallback(
                        base_records,
                        contract=contract,
                        action_binding=action_binding,
                        continuation_hash=continuation_hash,
                    )
                    reason = f"exact_base_fallback:{reason}"
                ticks = _net_ticks(records, tick_value_usd=tick_value_usd)
                trace_hash = dense_trace_content_hash(records)
                branch_binding = dense_branch_binding_hash(
                    action_binding_hash=action_binding,
                    branch_trace_hash=trace_hash,
                )
                trace_rows.extend(records)
                filled = any(
                    str(record["event_kind"]) == "fill"
                    and str(record["event_role"]) == "entry"
                    for record in records
                )
                payoff_rows.append(
                    {
                        "opportunity_id": opportunity_id,
                        "date": root["date"],
                        "partition": str(root["partition"]),
                        "fold": str(root["fold"]),
                        "decision_panel": str(root["decision_panel"]),
                        "calendar_block": str(root["calendar_block"]),
                        "library": "rent",
                        "action": contract.action_id,
                        "scenario": scenario.name,
                        "feature_view": "common_tape",
                        "feasible": bool(feasible),
                        "infeasible_reason": "" if feasible else reason,
                        "net_ticks": ticks,
                        "base_net_ticks": base_ticks,
                        "downside_ticks": max(0.0, base_ticks - ticks),
                        "filled_quantity": int(filled),
                        "forced_quantity": int(filled),
                        "retains_base_trade": bool(filled),
                        "worker_subset": contract.worker_subset,
                        "passive_fifo_certified": False,
                        "replay_start_time_ns": int(root["available_time_ns"]),
                        "replay_end_time_ns": int(root["terminal_arrival_time_ns"]),
                        "source_tape_hash": str(root["source_tape_hash"]),
                        "branch_trace_hash": trace_hash,
                        "entry_action": contract.entry_action,
                        "execution_action": contract.execution_action,
                        "protection_action": contract.protection_action,
                        "action_binding_hash": action_binding,
                        "branch_binding_hash": branch_binding,
                        "continuation_trace_hash": continuation_hash,
                        "branch_outcome_reason": reason,
                    }
                )
    result = DenseRentBranchResult(
        roots=clean_roots,
        observations=clean_observations,
        continuations=clean_continuations,
        payoffs=pd.DataFrame(payoff_rows),
        traces=pd.DataFrame(trace_rows, columns=TRACE_COLUMNS),
    )
    validate_dense_rent_branch_result(result, scenarios=scenarios, tick_value_usd=tick_value_usd)
    return result


def validate_dense_rent_branch_result(
    result: DenseRentBranchResult,
    *,
    scenarios: Sequence[BranchScenario] = DEFAULT_SCENARIOS,
    tick_value_usd: float = 10.0,
) -> None:
    roots = _validate_roots(result.roots)
    observations = _validate_observations(roots, result.observations, scenarios)
    continuations = _validate_continuations(
        roots, observations, result.continuations, scenarios
    )
    payoffs = result.payoffs.copy()
    traces = result.traces.copy()
    _require(
        payoffs,
        {
            "opportunity_id",
            "date",
            "partition",
            "fold",
            "decision_panel",
            "calendar_block",
            "scenario",
            "library",
            "action",
            "feature_view",
            "worker_subset",
            "net_ticks",
            "base_net_ticks",
            "downside_ticks",
            "feasible",
            "infeasible_reason",
            "filled_quantity",
            "forced_quantity",
            "retains_base_trade",
            "passive_fifo_certified",
            "replay_start_time_ns",
            "replay_end_time_ns",
            "branch_trace_hash",
            "source_tape_hash",
            *DENSE_RENT_PAYOFF_COLUMNS,
        },
        "dense RENT payoffs",
    )
    _require(traces, TRACE_COLUMNS, "dense RENT traces")
    if payoffs.duplicated(["opportunity_id", "scenario", "action"]).any():
        raise ValueError("dense RENT payoff keys are duplicated")
    expected_keys = {
        (str(root), scenario.name, action)
        for root in roots["opportunity_id"]
        for scenario in scenarios
        for action in DENSE_RENT_ACTION_IDS
    }
    observed_keys = set(
        payoffs[["opportunity_id", "scenario", "action"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise ValueError("dense RENT payoff lattice is incomplete or contains extra cells")
    continuation_index = continuations.set_index(["opportunity_id", "scenario"])
    root_index = roots.set_index("opportunity_id")
    trace_groups = {
        tuple(str(value) for value in key): group
        for key, group in traces.groupby(
            ["opportunity_id", "scenario", "action"], sort=False
        )
    }
    if set(trace_groups) != expected_keys:
        raise ValueError("dense RENT trace lattice is incomplete or contains extra cells")
    for row in payoffs.itertuples(index=False):
        contract = parse_rent_action_id(str(row.action))
        root = root_index.loc[str(row.opportunity_id)]
        if str(row.library) != "rent":
            raise ValueError("dense RENT payoff contains a non-RENT library row")
        if str(row.worker_subset) != contract.worker_subset:
            raise ValueError("payoff worker_subset disagrees with its action identifier")
        if (
            str(row.entry_action),
            str(row.execution_action),
            str(row.protection_action),
        ) != (
            contract.entry_action,
            contract.execution_action,
            contract.protection_action,
        ):
            raise ValueError("payoff primitive actions disagree with its action identifier")
        continuation_hash = str(
            continuation_index.loc[
                (str(row.opportunity_id), str(row.scenario)), "continuation_trace_hash"
            ]
        )
        if str(row.continuation_trace_hash) != continuation_hash:
            raise ValueError("payoff is not bound to its independent continuation trace")
        for column in ("date", "partition", "fold", "decision_panel", "calendar_block"):
            if str(getattr(row, column)) != str(root[column]):
                raise ValueError(f"dense RENT payoff {column} disagrees with its root")
        if str(row.source_tape_hash) != str(root["source_tape_hash"]):
            raise ValueError("dense RENT payoff source tape disagrees with its root")
        if int(row.replay_start_time_ns) != int(root["available_time_ns"]) or int(
            row.replay_end_time_ns
        ) != int(root["terminal_arrival_time_ns"]):
            raise ValueError("dense RENT payoff replay window disagrees with its root")
        economics = np.asarray(
            [row.net_ticks, row.base_net_ticks, row.downside_ticks], dtype=float
        )
        if not np.isfinite(economics).all():
            raise ValueError("dense RENT payoff economics must be finite")
        expected_downside = max(0.0, float(row.base_net_ticks) - float(row.net_ticks))
        if not np.isclose(
            float(row.downside_ticks), expected_downside, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "dense RENT downside_ticks must equal max(0, Base minus branch)"
            )
        expected_action_binding = dense_action_binding_hash(
            opportunity_id=str(row.opportunity_id),
            scenario=str(row.scenario),
            action=str(row.action),
            publisher_id=int(root_index.loc[str(row.opportunity_id), "publisher_id"]),
            instrument_id=int(root_index.loc[str(row.opportunity_id), "instrument_id"]),
            raw_symbol=str(root_index.loc[str(row.opportunity_id), "raw_symbol"]),
            continuity_segment_id=int(
                root_index.loc[str(row.opportunity_id), "continuity_segment_id"]
            ),
            source_tape_hash=str(row.source_tape_hash),
            continuation_trace_hash=continuation_hash,
        )
        if str(row.action_binding_hash) != expected_action_binding:
            raise ValueError("payoff action binding is invalid")
        trace = trace_groups[(str(row.opportunity_id), str(row.scenario), str(row.action))]
        if not trace["action_binding_hash"].astype(str).eq(expected_action_binding).all():
            raise ValueError("trace is not bound to its payoff action")
        if not trace["worker_subset"].astype(str).eq(contract.worker_subset).all():
            raise ValueError("trace worker_subset disagrees with its action")
        expected_trace_fields = {
            "entry_action": contract.entry_action,
            "execution_action": contract.execution_action,
            "protection_action": contract.protection_action,
            "continuation_trace_hash": continuation_hash,
            "source_tape_hash": str(root["source_tape_hash"]),
        }
        for column, expected in expected_trace_fields.items():
            if not trace[column].astype(str).eq(expected).all():
                raise ValueError(f"trace {column} disagrees with its branch contract")
        indices = pd.to_numeric(trace["event_index"], errors="raise").astype("int64")
        if indices.tolist() != list(range(len(trace))):
            raise ValueError("dense RENT trace event indices are not complete and ordered")
        header = trace.iloc[0]
        if (
            str(header["event_kind"]) != "decision"
            or int(header["time_ns"]) != int(root["decision_time_ns"])
        ):
            raise ValueError("dense RENT trace lacks its causal root-decision header")
        fill_trace = trace.loc[trace["event_kind"].astype(str).eq("fill")]
        fill_times = pd.to_numeric(fill_trace["time_ns"], errors="raise").astype("int64")
        if len(fill_times) and (
            not fill_times.is_monotonic_increasing
            or int(fill_times.min()) <= int(root["decision_time_ns"])
            or int(fill_times.max()) > int(root["terminal_arrival_time_ns"])
        ):
            raise ValueError("dense RENT fill trace violates the root replay clock")
        observed_trace_hash = dense_trace_content_hash(trace)
        if str(row.branch_trace_hash) != observed_trace_hash:
            raise ValueError("branch_trace_hash does not bind the complete trace")
        expected_branch_binding = dense_branch_binding_hash(
            action_binding_hash=expected_action_binding,
            branch_trace_hash=observed_trace_hash,
        )
        if str(row.branch_binding_hash) != expected_branch_binding:
            raise ValueError("branch binding does not bind subset, action, and trace")
        observed_ticks = _net_ticks(trace.to_dict("records"), tick_value_usd=tick_value_usd)
        if not np.isclose(observed_ticks, float(row.net_ticks), rtol=0.0, atol=1e-12):
            raise ValueError("payoff differs from its self-financing branch trace")
        entry_count = int(fill_trace["event_role"].astype(str).eq("entry").sum())
        exit_count = int(
            fill_trace["event_role"]
            .astype(str)
            .isin(("protection_flatten", "terminal_exit"))
            .sum()
        )
        if int(row.filled_quantity) != entry_count or int(row.forced_quantity) != exit_count:
            raise ValueError("dense RENT payoff quantities disagree with its fill trace")
        if _strict_bool(row.retains_base_trade, "retains_base_trade") != bool(entry_count):
            raise ValueError("dense RENT retention flag disagrees with its fill trace")
        if not _strict_bool(row.feasible, "feasible"):
            if not np.isclose(
                float(row.net_ticks), float(row.base_net_ticks), rtol=0.0, atol=1e-12
            ) or not _strict_bool(row.retains_base_trade, "retains_base_trade"):
                raise ValueError("infeasible dense action is not an exact Base fallback")
    for (opportunity_id, scenario), group in payoffs.groupby(
        ["opportunity_id", "scenario"], sort=False
    ):
        if set(group["worker_subset"].astype(str)) != set(REGISTERED_WORKER_SUBSETS):
            raise ValueError("dense RENT result does not cover all eight manager subsets")
        if group["base_net_ticks"].nunique(dropna=False) != 1:
            raise ValueError("dense RENT Base payoff changes within a root/scenario")
        base = group.loc[group["action"].astype(str).eq(BASE_RENT_ACTION.action_id)]
        if len(base) != 1 or not bool(base.iloc[0]["feasible"]):
            raise ValueError(f"root {opportunity_id!r}/{scenario!r} has no executable Base")
        if not np.isclose(
            float(base.iloc[0]["net_ticks"]),
            float(base.iloc[0]["base_net_ticks"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("dense RENT Base payoff is not exact")


def _revert_worker(contract: RentActionContract, worker: str) -> RentActionContract:
    members = set(worker_subset_members(contract.worker_subset))
    if worker not in members:
        raise ValueError(f"cannot revert inactive worker {worker!r}")
    members.remove(worker)
    return RentActionContract(
        canonical_worker_subset(members),
        BASE_ENTRY_ACTION if worker == "entry" else contract.entry_action,
        BASE_EXECUTION_ACTION if worker == "execution" else contract.execution_action,
        BASE_PROTECTION_ACTION if worker == "protection" else contract.protection_action,
    )


def build_dense_worker_rent_labels(
    result: DenseRentBranchResult,
    *,
    scenarios: Sequence[BranchScenario] = DEFAULT_SCENARIOS,
    tick_value_usd: float = 10.0,
) -> pd.DataFrame:
    """Pair each active worker with its exact one-worker Base reversion.

    Other workers' primitive actions, the root, scenario, feature view, source
    tape, and independent continuation are held fixed. This is deliberately an
    action-level label: collapsing the 45 cells to one arbitrary row per subset
    before reversion can misattribute another worker's action change as rent.
    """

    validate_dense_rent_branch_result(
        result, scenarios=scenarios, tick_value_usd=tick_value_usd
    )
    payoffs = result.payoffs.copy()
    index = payoffs.set_index(
        ["opportunity_id", "scenario", "feature_view", "action"]
    )
    rows: list[dict[str, Any]] = []
    for full in payoffs.itertuples(index=False):
        contract = parse_rent_action_id(str(full.action))
        for worker in worker_subset_members(contract.worker_subset):
            reverted_contract = _revert_worker(contract, worker)
            key = (
                str(full.opportunity_id),
                str(full.scenario),
                str(full.feature_view),
                reverted_contract.action_id,
            )
            if key not in index.index:
                raise ValueError(f"dense RENT action has no exact {worker!r} reversion")
            reverted = index.loc[key]
            if isinstance(reverted, pd.DataFrame):
                raise ValueError("dense RENT action reversion is not unique")
            invariants = (
                "date",
                "partition",
                "fold",
                "decision_panel",
                "calendar_block",
                "base_net_ticks",
                "source_tape_hash",
                "continuation_trace_hash",
                "replay_start_time_ns",
                "replay_end_time_ns",
            )
            changed = [
                column
                for column in invariants
                if str(getattr(full, column)) != str(reverted[column])
            ]
            if changed:
                raise ValueError(
                    f"worker-rent reversion changes common branch coordinates: {changed}"
                )
            for other_worker in set(RENT_WORKERS) - {worker}:
                field = f"{other_worker}_action"
                if str(getattr(full, field)) != str(reverted[field]):
                    raise ValueError(
                        f"{worker!r} reversion changes {other_worker!r} primitive action"
                    )
            full_net = float(full.net_ticks)
            reverted_net = float(reverted["net_ticks"])
            pair_payload = {
                "continuation_trace_hash": str(full.continuation_trace_hash),
                "full_action_binding_hash": str(full.action_binding_hash),
                "full_branch_trace_hash": str(full.branch_trace_hash),
                "reverted_action_binding_hash": str(reverted["action_binding_hash"]),
                "reverted_branch_trace_hash": str(reverted["branch_trace_hash"]),
                "worker": worker,
            }
            rows.append(
                {
                    "opportunity_id": str(full.opportunity_id),
                    "date": full.date,
                    "partition": str(full.partition),
                    "fold": str(full.fold),
                    "decision_panel": str(full.decision_panel),
                    "calendar_block": str(full.calendar_block),
                    "scenario": str(full.scenario),
                    "feature_view": str(full.feature_view),
                    "worker_subset": contract.worker_subset,
                    "reverted_worker_subset": reverted_contract.worker_subset,
                    "worker": worker,
                    "full_action": contract.action_id,
                    "reverted_action": reverted_contract.action_id,
                    "full_net_ticks": full_net,
                    "worker_reverted_net_ticks": reverted_net,
                    "worker_rent_ticks": full_net - reverted_net,
                    "base_net_ticks": float(full.base_net_ticks),
                    "joint_delta_to_base_ticks": full_net - float(full.base_net_ticks),
                    "source_tape_hash": str(full.source_tape_hash),
                    "continuation_trace_hash": str(full.continuation_trace_hash),
                    "full_action_binding_hash": str(full.action_binding_hash),
                    "reverted_action_binding_hash": str(reverted["action_binding_hash"]),
                    "full_branch_trace_hash": str(full.branch_trace_hash),
                    "reverted_branch_trace_hash": str(reverted["branch_trace_hash"]),
                    "worker_rent_pair_hash": _json_hash(pair_payload),
                }
            )
    labels = pd.DataFrame(rows)
    expected_per_root_scenario_view = sum(
        len(worker_subset_members(contract.worker_subset))
        for contract in DENSE_RENT_ACTION_CONTRACTS
    )
    if expected_per_root_scenario_view != 96:
        raise AssertionError("registered dense RENT worker-rent count changed")
    expected = (
        payoffs[["opportunity_id", "scenario", "feature_view"]]
        .drop_duplicates()
        .shape[0]
        * expected_per_root_scenario_view
    )
    if len(labels) != expected:
        raise AssertionError("dense RENT worker-rent lattice is incomplete")
    if labels.duplicated(
        ["opportunity_id", "scenario", "feature_view", "full_action", "worker"]
    ).any():
        raise AssertionError("dense RENT worker-rent pair keys are duplicated")
    return labels


def write_dense_rent_branch_result(
    path: str | Path,
    result: DenseRentBranchResult,
    *,
    evidence_class: str = "DEVELOPMENT_ONLY",
    scenarios: Sequence[BranchScenario] = DEFAULT_SCENARIOS,
    tick_value_usd: float = 10.0,
) -> dict[str, Any]:
    """Atomically persist a validated but explicitly uncertified dense lattice."""

    if evidence_class not in {"DEVELOPMENT_ONLY", "SYNTHETIC_ONLY"}:
        raise ValueError("prepared dense RENT writer cannot create locked/final evidence")
    if not np.isfinite(tick_value_usd) or tick_value_usd <= 0:
        raise ValueError("tick_value_usd must be finite and positive")
    validate_dense_rent_branch_result(
        result, scenarios=scenarios, tick_value_usd=tick_value_usd
    )
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"dense RENT output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        files: dict[str, dict[str, Any]] = {}
        for name in ("roots", "observations", "continuations", "payoffs", "traces"):
            frame = getattr(result, name)
            file_path = temporary / f"{name}.parquet"
            frame.to_parquet(file_path, index=False)
            files[name] = {
                "path": file_path.name,
                "rows": int(len(frame)),
                "sha256": sha256_file(file_path),
            }
        manifest = {
            "schema_version": 1,
            "product_type": "kx_prepared_dense_rent_counterfactuals",
            "evidence_class": evidence_class,
            "claim_status": "UNCERTIFIED_PREPARED_PRODUCT",
            "registered_worker_subsets": list(REGISTERED_WORKER_SUBSETS),
            "registered_action_count": len(DENSE_RENT_ACTION_CONTRACTS),
            "registered_scenarios": [scenario.name for scenario in scenarios],
            "tick_value_usd": float(tick_value_usd),
            "files": files,
        }
        write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, target)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    return manifest


def load_dense_rent_branch_result(
    path: str | Path,
    *,
    scenarios: Sequence[BranchScenario] = DEFAULT_SCENARIOS,
) -> DenseRentBranchResult:
    """Load, byte-verify, and economically replay a typed dense RENT artifact.

    A collection of files whose names merely look like branch hashes is not a
    replay certificate.  This loader accepts only the fixed dense-result
    product written above, re-hashes every Parquet frame, then delegates to
    :func:`validate_dense_rent_branch_result`, which rebuilds every trace hash
    and every flat-to-flat :class:`SelfFinancingLedger` payoff.
    """

    source = Path(path).expanduser().resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("dense RENT artifact is missing manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("dense RENT artifact manifest is not readable JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("dense RENT artifact manifest must be a JSON object")
    if (
        int(manifest.get("schema_version", 0)) != 1
        or str(manifest.get("product_type", ""))
        != "kx_prepared_dense_rent_counterfactuals"
    ):
        raise ValueError("unsupported dense RENT artifact manifest")
    if str(manifest.get("evidence_class", "")) not in {
        "DEVELOPMENT_ONLY",
        "LOCKED_EMPIRICAL",
        "SYNTHETIC_ONLY",
    }:
        raise ValueError("unsupported dense RENT artifact evidence class")
    if str(manifest.get("claim_status", "")) != "UNCERTIFIED_PREPARED_PRODUCT":
        raise ValueError("dense RENT artifact has an unsupported claim status")
    if tuple(manifest.get("registered_worker_subsets", ())) != tuple(
        REGISTERED_WORKER_SUBSETS
    ):
        raise ValueError("dense RENT artifact worker-subset registry changed")
    if int(manifest.get("registered_action_count", -1)) != len(
        DENSE_RENT_ACTION_CONTRACTS
    ):
        raise ValueError("dense RENT artifact action registry changed")
    expected_scenarios = tuple(scenario.name for scenario in scenarios)
    if tuple(manifest.get("registered_scenarios", ())) != expected_scenarios:
        raise ValueError("dense RENT artifact scenario registry changed")
    try:
        tick_value_usd = float(manifest["tick_value_usd"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dense RENT artifact must bind tick_value_usd") from error
    if not np.isfinite(tick_value_usd) or tick_value_usd <= 0:
        raise ValueError("dense RENT artifact tick_value_usd must be finite and positive")

    expected_files = ("roots", "observations", "continuations", "payoffs", "traces")
    declarations = manifest.get("files")
    if not isinstance(declarations, Mapping) or set(declarations) != set(expected_files):
        raise ValueError("dense RENT artifact must bind exactly its five typed frames")
    frames: dict[str, pd.DataFrame] = {}
    for name in expected_files:
        declaration = declarations[name]
        if not isinstance(declaration, Mapping):
            raise ValueError(f"dense RENT {name!r} declaration must be an object")
        expected_path = f"{name}.parquet"
        if str(declaration.get("path", "")) != expected_path:
            raise ValueError(f"dense RENT {name!r} artifact path is not canonical")
        expected_hash = str(declaration.get("sha256", ""))
        if not SHA256_PATTERN.fullmatch(expected_hash):
            raise ValueError(f"dense RENT {name!r} artifact needs a SHA-256")
        frame_path = source / expected_path
        if not frame_path.is_file():
            raise ValueError(f"dense RENT artifact is missing {expected_path}")
        observed_hash = sha256_file(frame_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"dense RENT {name!r} artifact hash mismatch: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
        frame = pd.read_parquet(frame_path)
        if len(frame) != int(declaration.get("rows", -1)):
            raise ValueError(f"dense RENT {name!r} artifact row count changed")
        frames[name] = frame

    result = DenseRentBranchResult(
        roots=frames["roots"],
        observations=frames["observations"],
        continuations=frames["continuations"],
        payoffs=frames["payoffs"],
        traces=frames["traces"],
    )
    validate_dense_rent_branch_result(
        result,
        scenarios=scenarios,
        tick_value_usd=tick_value_usd,
    )
    return result
