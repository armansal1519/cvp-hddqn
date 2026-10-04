"""Five-seed ablation replay through the unchanged V6 economic ledger."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kx_simple_hmadrl.volume_v6_contract import ACTIONS, MESSAGE_WIDTH, neutral_regime_message
from kx_simple_hmadrl.volume_v6_env import (
    V6TradingEnvironment, legal_target_mask, uses_legacy_arrival_boundary,
)
from kx_simple_hmadrl.volume_v6_eval import flat_on_vote_tie

from .aggregation import (
    REPLAY_ARTIFACT,
    TECHNICAL_FAILURE_ARTIFACT,
    verify_complete_training_manifest,
)
from .artifacts import (
    atomic_json,
    atomic_parquet,
    file_sha256,
    project_relative,
    read_json_object,
    resolve_bound_path,
    resolve_project_path,
)
from .contract import (
    PERIOD_CLASSIFICATION,
    PROTOCOL_ID,
    PUBLISHER_PARENT_ARM,
    SEEDS,
    AblationArm,
    validate_arm,
    validate_period,
    validate_scenario,
    scenario_economics,
)
from .frozen_v6 import (
    FrozenPublisher,
    bottom_observation_at,
    build_session_features,
    latest_message,
    load_frozen_publisher,
    load_npz_pair,
    publisher_messages,
    raw_index,
    trader_q_values,
    valid_feature_indexes,
)
from .protocol import (
    implementation_binding, load_authorized_dataset, protocol_binding,
    verify_implementation_binding,
)
from .representation import RepresentationMask
from .training import load_training_member


PERIOD_DATASETS = {
    "may_2026": (
        "viability/artifacts/development/"
        "v6_registered_validation_dataset_eligible/manifest.json"
    ),
    "june_july_2026": "paper/sealed_extension/dataset/manifest.json",
}
TECHNICAL_FAILURE_REASONS = frozenset({
    "HASH_MISMATCH",
    "CAUSALITY_VIOLATION",
    "NONFINITE_STATE_OR_SCORE",
    "INCOMPLETE_TRAINING",
    "REWARD_LEDGER_MISMATCH",
    "ACTION_LEGALITY_VIOLATION",
    "TERMINAL_INVENTORY_NOT_FLAT",
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


def _training_members(
    training_manifest_path: Path, arm: AblationArm,
) -> list[Path]:
    value = verify_complete_training_manifest(training_manifest_path)
    jobs = value.get("jobs")
    assert isinstance(jobs, list)
    selected: dict[int, Path] = {}
    for job in jobs:
        assert isinstance(job, dict)
        if job.get("arm") != arm.value:
            continue
        seed = int(job["seed"])
        record = job.get("manifest")
        assert isinstance(record, dict)
        path = resolve_bound_path(training_manifest_path.parent, record.get("path"))
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError("ablation ensemble member hash differs")
        selected[seed] = path
    if tuple(sorted(selected)) != SEEDS:
        raise ValueError("ablation ensemble is not the exact five-seed set")
    return [selected[seed] for seed in SEEDS]


def _publisher_from_member(
    manifest: Mapping[str, object], *, arm: AblationArm, seed: int,
) -> FrozenPublisher | None:
    expected = PUBLISHER_PARENT_ARM[arm.value]
    if expected is None:
        return None
    record = manifest.get("reference_member_manifest")
    if not isinstance(record, dict):
        raise ValueError("ablation member lacks publisher-source lineage")
    from .artifacts import resolve_project_path

    manifest_path = resolve_project_path(record.get("path"))
    if not manifest_path.is_file() or file_sha256(manifest_path) != record.get("sha256"):
        raise ValueError("ablation publisher-source manifest changed")
    return load_frozen_publisher(
        manifest_path.parent, expected_parent_arm=str(expected), expected_seed=seed,
    )


def _metrics(daily: Mapping[str, int], completed_round_trips: int) -> dict[str, object]:
    if not daily:
        raise ValueError("ablation replay has no daily P&L")
    cents = np.asarray([int(daily[day]) for day in sorted(daily)], dtype=np.float64)
    usd = cents / 100.0
    standard = float(usd.std(ddof=1)) if len(usd) > 1 else 0.0
    sharpe = 0.0 if standard < 1e-12 else float(np.sqrt(252.0) * usd.mean() / standard)
    equity_cents = np.concatenate(([0.0], np.cumsum(cents)))
    drawdown_cents = np.maximum.accumulate(equity_cents) - equity_cents
    gains = float(usd[usd > 0].sum())
    losses = float(-usd[usd < 0].sum())
    profit_factor = gains / losses if losses > 0 else None
    return {
        "sessions": len(cents),
        "net_pnl_cents": int(cents.sum()),
        "net_pnl_usd": float(usd.sum()),
        "mean_daily_pnl_cents": float(cents.mean()),
        "mean_daily_pnl_usd": float(usd.mean()),
        "annualized_daily_sharpe": sharpe,
        "maximum_drawdown_cents": int(drawdown_cents.max(initial=0.0)),
        "maximum_drawdown_usd": float(drawdown_cents.max(initial=0.0) / 100.0),
        "completed_round_trips": int(completed_round_trips),
        "positive_day_fraction": float((cents > 0).mean()),
        "profit_factor": profit_factor,
        "daily_profit_factor": profit_factor,
    }


def _trace_record(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": path.name,
        "path_scope": "report_relative",
        "sha256": file_sha256(path),
        "rows": int(rows),
    }


def _verify_trace(report_path: Path, record: object) -> Path:
    if not isinstance(record, dict):
        raise ValueError("replay trace record is malformed")
    path = (report_path.parent / str(record.get("path", ""))).resolve()
    try:
        path.relative_to(report_path.parent.resolve())
    except ValueError as error:
        raise ValueError("replay trace escapes its report root") from error
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("replay trace hash differs")
    if pq.ParquetFile(path).metadata.num_rows != int(record.get("rows", -1)):
        raise ValueError("replay trace row count differs")
    return path


def _trace_rows(path: Path) -> list[dict[str, object]]:
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _require_exact_message(row: Mapping[str, object], *, arm: str) -> None:
    names = (
        "message_short", "message_flat", "message_long",
        "message_valid", "message_lifetime",
    )
    values = np.asarray([row.get(name) for name in names], dtype=np.float64)
    if values.shape != (MESSAGE_WIDTH,) or not np.isfinite(values).all():
        raise ValueError("ablation decision trace contains an invalid message")
    if arm == AblationArm.NO_PUBLISHER.value:
        if not np.array_equal(values, np.zeros(MESSAGE_WIDTH, dtype=np.float64)):
            raise ValueError("no-publisher trace contains a nonzero message")
        return
    publication = int(row.get("message_publication_time_ns", -1))
    expiry = int(row.get("message_expiry_time_ns", -1))
    decision = int(row.get("decision_time_ns", -1))
    if not publication <= decision < expiry:
        raise ValueError("publisher message was consumed outside its causal lifetime")
    if values[3] != 1.0 or not 0.0 <= values[4] <= 1.0:
        raise ValueError("publisher validity/lifetime encoding is inconsistent")


def _validate_route_row(
    row: Mapping[str, object], *, economics: object, terminal: bool,
) -> tuple[int, int, int, int, bool] | None:
    decision_field = "next_time_ns" if terminal else "decision_time_ns"
    fill_field = "terminal_fill_time_ns" if terminal else "route_fill_time_ns"
    price_field = "terminal_fill_price_ticks" if terminal else "route_fill_price_ticks"
    fee_field = "terminal_fill_fee_cents" if terminal else "route_fill_fee_cents"
    side_field = "terminal_route_side" if terminal else "route_side"
    size_field = "terminal_displayed_size" if terminal else "route_displayed_size"
    deadline_field = "terminal_route_deadline_ns" if terminal else "route_deadline_ns"
    fill_time = row.get(fill_field)
    deadline = row.get(deadline_field)
    side = row.get(side_field)
    displayed = row.get(size_field)
    price = row.get(price_field)
    fee = row.get(fee_field)
    if deadline is None:
        if any(value is not None for value in (fill_time, side, displayed, price, fee)):
            raise ValueError("absent route trace is partially populated")
        return None
    if side is None:
        raise ValueError("created route trace lacks its side")
    route_side = int(side)
    decision = int(row[decision_field])
    deadline_ns = int(deadline)
    expected_deadline = decision + int(economics.deadline_ms) * 1_000_000
    if route_side not in (-1, 1) or deadline_ns != expected_deadline:
        raise ValueError("created route side/deadline differs from the contract")
    if fill_time is None:
        if terminal or any(value is not None for value in (displayed, price, fee)):
            raise ValueError("rejected or terminal route trace is inconsistent")
        return None
    if any(value is None for value in (displayed, price, fee)):
        raise ValueError("filled route trace is incomplete")
    fill_ns = int(fill_time)
    arrival_ns = decision + int(economics.route_latency_ms) * 1_000_000
    if int(displayed) < 1 or not arrival_ns <= fill_ns <= deadline_ns:
        raise ValueError("fill lacks displayed size or falls outside its route interval")
    if int(fee) != int(economics.fee_cents_per_side):
        raise ValueError("fill trace commission differs from scenario economics")
    return fill_ns, route_side, int(price), int(fee), bool(terminal)


def _verify_ensemble_traces(
    *, report: Mapping[str, object], arm: str, economics: object,
    decision_path: Path, reward_path: Path, fill_path: Path,
) -> None:
    decision_rows = _trace_rows(decision_path)
    reward_rows = _trace_rows(reward_path)
    fill_rows = _trace_rows(fill_path)
    dates = [str(day) for day in report["evaluation_dates"]]
    daily = report["daily_pnl_cents"]
    assert isinstance(daily, dict)

    decisions_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in decision_rows:
        key = (str(row.get("session_date")), int(row.get("decision_ordinal", -1)))
        decisions_by_key.setdefault(key, []).append(row)
    rewards_by_key = {
        (str(row.get("session_date")), int(row.get("decision_ordinal", -1))): row
        for row in reward_rows
    }
    if len(rewards_by_key) != len(reward_rows) or set(decisions_by_key) != set(
        rewards_by_key
    ):
        raise ValueError("ensemble decision and reward trace keys differ")
    for key, rows in decisions_by_key.items():
        if len(rows) != len(SEEDS) or {int(row.get("seed", -1)) for row in rows} != set(SEEDS):
            raise ValueError("ensemble decision trace is not one row per seed")
        reward = rewards_by_key[key]
        votes: list[int] = []
        common: tuple[object, ...] | None = None
        for row in rows:
            _require_exact_message(row, arm=arm)
            q = np.asarray(
                [row.get("q_short"), row.get("q_flat"), row.get("q_long")],
                dtype=np.float64,
            )
            legal = np.asarray(
                [row.get("legal_short"), row.get("legal_flat"), row.get("legal_long")],
                dtype=np.bool_,
            )
            if not np.isfinite(q).all() or legal.shape != (3,) or not legal.any():
                raise ValueError("ensemble decision trace contains invalid Q/legal state")
            inventory_before = int(row.get("inventory_before", 99))
            if not np.array_equal(legal, legal_target_mask(inventory_before)):
                raise ValueError("ensemble legal mask differs from inventory state")
            masked = q.copy(); masked[~legal] = -np.inf
            expected_vote = ACTIONS[int(np.argmax(masked))]
            vote = int(row.get("seed_vote", 99))
            if vote != expected_vote or not bool(legal[ACTIONS.index(vote)]):
                raise ValueError("seed vote does not reconcile to masked Q values")
            votes.append(vote)
            one = (
                int(row.get("decision_time_ns", -1)), int(row.get("next_time_ns", -1)),
                tuple(bool(value) for value in legal),
                int(row.get("ensemble_target", 99)), int(row.get("inventory_before", 99)),
                int(row.get("inventory_after", 99)), bool(row.get("terminal")),
            )
            if common is None:
                common = one
            elif one != common:
                raise ValueError("seed rows disagree on shared ensemble transition state")
        expected_target = flat_on_vote_tie(votes)
        if expected_target != int(reward.get("ensemble_target", 99)):
            raise ValueError("ensemble target does not reconcile to five seed votes")
        for row in rows:
            for field in (
                "decision_time_ns", "next_time_ns", "ensemble_target",
                "inventory_before", "inventory_after", "command_created",
                "command_rejected", "reward_cents", "terminal",
            ):
                if row.get(field) != reward.get(field):
                    raise ValueError(
                        "ensemble decision/reward transition fields differ"
                    )
        counts = {action: votes.count(action) for action in ACTIONS}
        for row in rows:
            if (
                int(row.get("votes_short", -1)) != counts[-1]
                or int(row.get("votes_flat", -1)) != counts[0]
                or int(row.get("votes_long", -1)) != counts[1]
            ):
                raise ValueError("recorded ensemble vote counts differ")

    generated_fills: list[tuple[str, int, int, int, int, bool]] = []
    rejected = 0
    for day in dates:
        rows = sorted(
            (row for row in reward_rows if str(row.get("session_date")) == day),
            key=lambda row: int(row.get("decision_ordinal", -1)),
        )
        if not rows or [int(row["decision_ordinal"]) for row in rows] != list(range(len(rows))):
            raise ValueError("ensemble reward ordinals are incomplete")
        cumulative = 0
        prior_inventory = 0
        for ordinal, row in enumerate(rows):
            decision = int(row["decision_time_ns"]); next_time = int(row["next_time_ns"])
            if next_time <= decision:
                raise ValueError("reward interval is not forward in time")
            if ordinal and decision != int(rows[ordinal - 1]["next_time_ns"]):
                raise ValueError("reward intervals are not chronologically contiguous")
            before = int(row["inventory_before"]); after = int(row["inventory_after"])
            if before != prior_inventory or before not in ACTIONS or after not in ACTIONS:
                raise ValueError("ensemble inventory trace is discontinuous")
            prior_inventory = after
            terminal = bool(row["terminal"])
            if terminal != (ordinal == len(rows) - 1):
                raise ValueError("ensemble terminal flag is misplaced")
            cumulative += int(row["reward_cents"])
            if cumulative != int(row["cumulative_executable_nlv_cents"]):
                raise ValueError("ensemble cumulative executable NLV does not telescope")
            command_created = bool(row["command_created"])
            command_rejected = bool(row["command_rejected"])
            target = int(row["ensemble_target"])
            if target not in ACTIONS or not bool(
                legal_target_mask(before)[ACTIONS.index(target)]
            ):
                raise ValueError("ensemble target is illegal for recorded inventory")
            main = _validate_route_row(row, economics=economics, terminal=False)
            if command_created != (target != before) or command_created != (
                row.get("route_deadline_ns") is not None
            ):
                raise ValueError("command-created flag differs from route trace")
            if command_rejected != (command_created and main is None):
                raise ValueError("command-rejection flag differs from route outcome")
            if command_created and int(row["route_side"]) != target - before:
                raise ValueError("route side differs from the legal target change")
            post_main = before if (not command_created or command_rejected) else target
            terminal_route_exists = row.get("terminal_route_deadline_ns") is not None
            if terminal_route_exists != (terminal and post_main != 0):
                raise ValueError("terminal liquidation route presence differs from inventory")
            if terminal_route_exists and int(row["terminal_route_side"]) != -post_main:
                raise ValueError("terminal liquidation side differs from open inventory")
            if not terminal and after != post_main:
                raise ValueError("post-route inventory differs from route outcome")
            rejected += int(command_rejected)
            if main is not None:
                generated_fills.append((day, *main))
            terminal_fill = _validate_route_row(row, economics=economics, terminal=True)
            if terminal_fill is not None:
                generated_fills.append((day, *terminal_fill))
        if prior_inventory != 0 or cumulative != int(daily[day]):
            raise ValueError("terminal inventory or reward-to-cash reconciliation failed")

    recorded_fills: list[tuple[str, int, int, int, int, bool]] = []
    per_day_ordinals: dict[str, list[int]] = {day: [] for day in dates}
    for row in fill_rows:
        day = str(row.get("session_date"))
        per_day_ordinals.setdefault(day, []).append(int(row.get("fill_ordinal", -1)))
        recorded_fills.append((
            day, int(row["timestamp_ns"]), int(row["side"]), int(row["price_ticks"]),
            int(row["fee_cents"]), bool(row["forced"]),
        ))
    if any(sorted(values) != list(range(len(values))) for values in per_day_ordinals.values()):
        raise ValueError("fill ordinals are incomplete")
    if sorted(recorded_fills) != sorted(generated_fills):
        raise ValueError("ensemble fill and route traces do not reconcile")
    if rejected != int(report.get("rejected_commands", -1)):
        raise ValueError("ensemble rejection count does not reconcile")
    if len(recorded_fills) != int(report.get("filled_sides", -1)):
        raise ValueError("ensemble filled-side count does not reconcile")
    if int(report.get("completed_round_trips", -1)) * 2 != len(recorded_fills):
        raise ValueError("ensemble round trips do not reconcile to filled sides")


def _verify_seed_traces(
    *, report: Mapping[str, object], arm: str, economics: object,
    decision_path: Path, reward_path: Path, fill_path: Path,
) -> None:
    decision_rows = _trace_rows(decision_path)
    reward_rows = _trace_rows(reward_path)
    fill_rows = _trace_rows(fill_path)
    decision_index = {
        (str(row.get("session_date")), int(row.get("seed", -1)),
         int(row.get("decision_ordinal", -1))): row
        for row in decision_rows
    }
    reward_index = {
        (str(row.get("session_date")), int(row.get("seed", -1)),
         int(row.get("decision_ordinal", -1))): row
        for row in reward_rows
    }
    if (
        len(decision_index) != len(decision_rows)
        or len(reward_index) != len(reward_rows)
        or set(decision_index) != set(reward_index)
    ):
        raise ValueError("per-seed decision and reward trace keys differ")
    for key, decision in decision_index.items():
        _require_exact_message(decision, arm=arm)
        q = np.asarray(
            [decision.get("q_short"), decision.get("q_flat"), decision.get("q_long")],
            dtype=np.float64,
        )
        legal = np.asarray(
            [decision.get("legal_short"), decision.get("legal_flat"),
             decision.get("legal_long")], dtype=np.bool_,
        )
        masked = q.copy(); masked[~legal] = -np.inf
        expected = ACTIONS[int(np.argmax(masked))]
        if (
            not np.isfinite(q).all()
            or not legal.any()
            or not np.array_equal(
                legal, legal_target_mask(int(decision.get("inventory_before", 99))),
            )
            or int(decision.get("action", 99)) != expected
        ):
            raise ValueError("per-seed action does not reconcile to masked Q values")
        reward = reward_index[key]
        for field in (
            "decision_time_ns", "next_time_ns", "action", "inventory_before",
            "inventory_after", "command_created", "command_rejected", "terminal",
        ):
            if decision.get(field) != reward.get(field):
                raise ValueError("per-seed decision/reward transition fields differ")

    seed_daily = report["per_seed_daily_pnl_cents"]
    seed_metrics = report["per_seed_metrics"]
    assert isinstance(seed_daily, dict) and isinstance(seed_metrics, dict)
    generated_fills: list[tuple[str, int, int, int, int, int, bool]] = []
    for seed in SEEDS:
        metrics = seed_metrics[str(seed)]
        assert isinstance(metrics, dict)
        rejected = 0; filled = 0
        for day in report["evaluation_dates"]:
            rows = sorted(
                (row for row in reward_rows
                 if int(row.get("seed", -1)) == seed
                 and str(row.get("session_date")) == str(day)),
                key=lambda row: int(row.get("decision_ordinal", -1)),
            )
            if not rows or [int(row["decision_ordinal"]) for row in rows] != list(range(len(rows))):
                raise ValueError("per-seed reward ordinals are incomplete")
            cumulative = 0; prior_inventory = 0
            for ordinal, row in enumerate(rows):
                if ordinal and int(row["decision_time_ns"]) != int(rows[ordinal - 1]["next_time_ns"]):
                    raise ValueError("per-seed reward intervals are not contiguous")
                if int(row["next_time_ns"]) <= int(row["decision_time_ns"]):
                    raise ValueError("per-seed reward interval is not forward in time")
                before = int(row["inventory_before"]); after = int(row["inventory_after"])
                if before != prior_inventory or before not in ACTIONS or after not in ACTIONS:
                    raise ValueError("per-seed inventory trace is discontinuous")
                prior_inventory = after
                if bool(row["terminal"]) != (ordinal == len(rows) - 1):
                    raise ValueError("per-seed terminal flag is misplaced")
                cumulative += int(row["reward_cents"])
                if cumulative != int(row["cumulative_executable_nlv_cents"]):
                    raise ValueError("per-seed cumulative executable NLV does not telescope")
                main = _validate_route_row(row, economics=economics, terminal=False)
                created = bool(row["command_created"]); was_rejected = bool(row["command_rejected"])
                target = int(row["action"])
                if target not in ACTIONS or not bool(
                    legal_target_mask(before)[ACTIONS.index(target)]
                ):
                    raise ValueError("per-seed action is illegal for recorded inventory")
                if created != (target != before) or created != (
                    row.get("route_deadline_ns") is not None
                ):
                    raise ValueError("per-seed command-created flag differs from route")
                if was_rejected != (created and main is None):
                    raise ValueError("per-seed command-rejection flag differs from route")
                if created and int(row["route_side"]) != target - before:
                    raise ValueError("per-seed route side differs from target change")
                post_main = before if (not created or was_rejected) else target
                terminal_route_exists = row.get("terminal_route_deadline_ns") is not None
                if terminal_route_exists != (bool(row["terminal"]) and post_main != 0):
                    raise ValueError("per-seed terminal route presence differs from inventory")
                if terminal_route_exists and int(row["terminal_route_side"]) != -post_main:
                    raise ValueError("per-seed terminal liquidation side differs")
                if not bool(row["terminal"]) and after != post_main:
                    raise ValueError("per-seed post-route inventory differs")
                rejected += int(was_rejected)
                if main is not None:
                    generated_fills.append((str(day), seed, *main)); filled += 1
                terminal_fill = _validate_route_row(row, economics=economics, terminal=True)
                if terminal_fill is not None:
                    generated_fills.append((str(day), seed, *terminal_fill)); filled += 1
            if prior_inventory != 0 or cumulative != int(seed_daily[str(seed)][str(day)]):
                raise ValueError("per-seed terminal reward/cash reconciliation failed")
        if rejected != int(metrics.get("rejected_commands", -1)):
            raise ValueError("per-seed rejection count does not reconcile")
        if filled != int(metrics.get("filled_sides", -1)):
            raise ValueError("per-seed filled-side count does not reconcile")
        if 2 * int(metrics.get("completed_round_trips", -1)) != filled:
            raise ValueError("per-seed round trips do not reconcile to filled sides")

    recorded_fills: list[tuple[str, int, int, int, int, int, bool]] = []
    fill_ordinals: dict[tuple[str, int], list[int]] = {}
    for row in fill_rows:
        day = str(row["session_date"]); seed = int(row["seed"])
        fill_ordinals.setdefault((day, seed), []).append(
            int(row.get("fill_ordinal", -1))
        )
        recorded_fills.append((
            day, seed, int(row["timestamp_ns"]), int(row["side"]),
            int(row["price_ticks"]), int(row["fee_cents"]), bool(row["forced"]),
        ))
    if any(
        sorted(ordinals) != list(range(len(ordinals)))
        for ordinals in fill_ordinals.values()
    ):
        raise ValueError("per-seed fill ordinals are incomplete")
    if sorted(recorded_fills) != sorted(generated_fills):
        raise ValueError("per-seed fill and route traces do not reconcile")


def replay_ensemble(
    *, arm: str | AblationArm, period: str, scenario: str,
    training_manifest: Path, output_root: Path,
    dataset_manifest: str | None = None,
) -> Path:
    """Replay one registered arm/period/scenario and retain full compact traces."""

    selected_arm = validate_arm(arm)
    selected_period = validate_period(period).value
    selected_scenario = validate_scenario(scenario).value
    relative_dataset = (
        PERIOD_DATASETS[selected_period]
        if dataset_manifest is None else str(dataset_manifest)
    )
    # The physical guard lexically rejects an unregistered/August path before
    # opening the training manifest or any market-data path.
    dataset = load_authorized_dataset(
        relative_dataset, purpose="evaluate", period_id=selected_period,
    )
    training_path = Path(training_manifest).resolve()
    member_paths = _training_members(training_path, selected_arm)
    representation = RepresentationMask.for_arm(selected_arm)
    output = (
        Path(output_root).resolve() / "evaluation" / selected_period /
        selected_scenario / selected_arm.value
    )
    report_path = output / "report.json"
    if report_path.is_file():
        verify_replay_report(
            report_path, expected_arm=selected_arm.value,
            expected_period=selected_period, expected_scenario=selected_scenario,
        )
        return report_path
    if output.exists():
        raise FileExistsError("evaluation output exists without a complete report")
    work = output.with_name(f".{output.name}.part")
    if work.exists():
        raise FileExistsError("stale evaluation part directory must be audited")
    work.mkdir(parents=True)

    members = []
    for seed, manifest_path in zip(SEEDS, member_paths, strict=True):
        trader, scaler, manifest = load_training_member(
            manifest_path, expected_arm=selected_arm.value, expected_seed=seed,
        )
        publisher = _publisher_from_member(manifest, arm=selected_arm, seed=seed)
        if publisher is not None and (
            publisher.scaler.canonical_sha256() != scaler.canonical_sha256()
        ):
            raise ValueError("same-seed publisher and trader scalers differ")
        members.append((seed, manifest_path, trader, scaler, manifest, publisher))
    scaler_hashes = {member[3].canonical_sha256() for member in members}
    training_dates = {tuple(member[4].get("training_dates", [])) for member in members}
    initialization_hashes = {str(member[4].get("initialization_sha256")) for member in members}
    if len(scaler_hashes) != 1 or len(training_dates) != 1:
        raise ValueError("ensemble members differ in transforms or training dates")
    if len(initialization_hashes) != len(SEEDS):
        # Each seed has its own initialization; equality is required only across
        # arms, which the complete-training manifest already verifies.
        raise ValueError("ensemble seed initializations are not unique")
    scaler = members[0][3]
    economics = scenario_economics(selected_scenario)
    phase = "validation" if selected_period == "may_2026" else "test"

    daily: dict[str, int] = {}
    decision_rows: list[dict[str, object]] = []
    reward_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    seed_decision_rows: list[dict[str, object]] = []
    seed_reward_rows: list[dict[str, object]] = []
    seed_fill_rows: list[dict[str, object]] = []
    seed_daily: dict[str, dict[str, int]] = {str(seed): {} for seed in SEEDS}
    seed_trips: dict[int, int] = {seed: 0 for seed in SEEDS}
    seed_rejections: dict[int, int] = {seed: 0 for seed in SEEDS}
    seed_filled_sides: dict[int, int] = {seed: 0 for seed in SEEDS}
    seed_commissions_cents: dict[int, int] = {seed: 0 for seed in SEEDS}
    trips = rejected = filled_sides = commissions_cents = 0
    unsafe_legacy_arrival_boundary = False
    for raw_path, side_path in dataset.pairs:
        raw, side = load_npz_pair(raw_path, side_path, phase=phase)
        session = build_session_features(raw, side, scaler.calibration)
        selected = valid_feature_indexes(session, top=False)
        if not len(selected):
            raise ValueError(f"{raw_path.stem}: no valid ablation trader decisions")
        member_messages = [
            () if publisher is None else publisher_messages(
                publisher, session, representation,
            )
            for _, _, _, _, _, publisher in members
        ]
        unsafe_legacy_arrival_boundary |= uses_legacy_arrival_boundary(raw, economics)
        env = V6TradingEnvironment(raw, raw_path.stem, economics)
        cumulative_nlv_cents = 0
        for ordinal, feature_index_value in enumerate(selected):
            feature_index = int(feature_index_value)
            decision = int(session.bottom_clock_ns[feature_index])
            current_raw = raw_index(raw, decision)
            terminal = ordinal == len(selected) - 1
            if terminal:
                next_raw = len(np.asarray(raw["clock_ns"])) - 1
            else:
                next_raw = raw_index(
                    raw, int(session.bottom_clock_ns[int(selected[ordinal + 1])]),
                )
            state = env.agent_state(current_raw)
            legal = env.legal_actions()
            seed_records: list[
                tuple[int, np.ndarray, int, np.ndarray, int, int]
            ] = []
            votes: list[int] = []
            for member, messages in zip(members, member_messages, strict=True):
                seed, _, trader, member_scaler, _, _ = member
                message = (
                    neutral_regime_message(decision)
                    if representation.publisher_parent_arm is None
                    else latest_message(messages, decision)
                )
                observation = representation.trader_arrays(bottom_observation_at(
                    session, feature_index, member_scaler, message, state,
                ))
                if representation.message_mode == "exact_zero" and not np.array_equal(
                    observation["message"], np.zeros(MESSAGE_WIDTH, np.float32),
                ):
                    raise AssertionError("no-publisher replay consumed a nonzero message")
                q = trader_q_values(trader, observation)
                masked = q.copy(); masked[~legal] = -np.inf
                vote = ACTIONS[int(np.argmax(masked))]
                votes.append(vote)
                seed_records.append((
                    seed, q, vote, observation["message"].copy(),
                    int(message.decision_time_ns), int(message.expiry_time_ns),
                ))
            ensemble_target = flat_on_vote_tie(votes)
            if ensemble_target not in ACTIONS or not bool(legal[ACTIONS.index(ensemble_target)]):
                raise AssertionError("ensemble produced an illegal target")
            transition = env.step(
                current_raw, ensemble_target, next_raw, terminal=terminal,
            )
            cumulative_nlv_cents += transition.reward_cents
            counts = {action: votes.count(action) for action in ACTIONS}
            for seed, q, vote, message, publication_ns, expiry_ns in seed_records:
                decision_rows.append({
                    "session_date": raw_path.stem,
                    "decision_ordinal": ordinal,
                    "decision_time_ns": transition.decision_time_ns,
                    "next_time_ns": transition.next_time_ns,
                    "seed": seed,
                    "q_short": float(q[0]),
                    "q_flat": float(q[1]),
                    "q_long": float(q[2]),
                    "legal_short": bool(legal[0]),
                    "legal_flat": bool(legal[1]),
                    "legal_long": bool(legal[2]),
                    "seed_vote": int(vote),
                    "votes_short": counts[-1],
                    "votes_flat": counts[0],
                    "votes_long": counts[1],
                    "ensemble_target": int(ensemble_target),
                    "message_short": float(message[0]),
                    "message_flat": float(message[1]),
                    "message_long": float(message[2]),
                    "message_valid": float(message[3]),
                    "message_lifetime": float(message[4]),
                    "message_publication_time_ns": publication_ns,
                    "message_expiry_time_ns": expiry_ns,
                    "inventory_before": transition.inventory_before,
                    "inventory_after": transition.inventory_after,
                    "command_created": transition.command_created,
                    "command_rejected": transition.command_rejected,
                    "reward_cents": transition.reward_cents,
                    "terminal": transition.done,
                })
            route = transition.route
            terminal_route = transition.terminal_route
            reward_rows.append({
                "session_date": raw_path.stem,
                "decision_ordinal": ordinal,
                "decision_time_ns": transition.decision_time_ns,
                "next_time_ns": transition.next_time_ns,
                "ensemble_target": int(ensemble_target),
                "inventory_before": transition.inventory_before,
                "inventory_after": transition.inventory_after,
                "reward_cents": transition.reward_cents,
                "cumulative_executable_nlv_cents": cumulative_nlv_cents,
                "command_created": transition.command_created,
                "command_rejected": transition.command_rejected,
                "route_deadline_ns": None if route is None else route.deadline_time_ns,
                "route_side": None if route is None else route.side,
                "route_fill_time_ns": (
                    None if route is None or route.fill is None
                    else route.fill.timestamp_ns
                ),
                "route_fill_price_ticks": (
                    None if route is None or route.fill is None
                    else route.fill.price_ticks
                ),
                "route_fill_fee_cents": (
                    None if route is None or route.fill is None
                    else route.fill.fee_cents
                ),
                "route_displayed_size": (
                    None if route is None or route.book is None
                    else (
                        route.book.ask_size if route.side > 0
                        else route.book.bid_size
                    )
                ),
                "route_searched_snapshots": (
                    None if route is None else route.searched_snapshots
                ),
                "terminal_route_deadline_ns": (
                    None if terminal_route is None
                    else terminal_route.deadline_time_ns
                ),
                "terminal_route_side": (
                    None if terminal_route is None else terminal_route.side
                ),
                "terminal_fill_time_ns": (
                    None if terminal_route is None or terminal_route.fill is None
                    else terminal_route.fill.timestamp_ns
                ),
                "terminal_fill_price_ticks": (
                    None if terminal_route is None or terminal_route.fill is None
                    else terminal_route.fill.price_ticks
                ),
                "terminal_fill_fee_cents": (
                    None if terminal_route is None or terminal_route.fill is None
                    else terminal_route.fill.fee_cents
                ),
                "terminal_displayed_size": (
                    None if terminal_route is None or terminal_route.book is None
                    else (
                        terminal_route.book.ask_size if terminal_route.side > 0
                        else terminal_route.book.bid_size
                    )
                ),
                "terminal": transition.done,
            })
        result = env.finish()
        if cumulative_nlv_cents != result.net_pnl_cents:
            raise AssertionError("reward trace does not telescope to terminal cash")
        if result.transitions[-1].inventory_after != 0:
            raise AssertionError("terminal ablation inventory is not flat")
        daily[raw_path.stem] = result.net_pnl_cents
        trips += result.completed_round_trips
        rejected += result.rejected_commands
        filled_sides += len(result.fills)
        commissions_cents += sum(fill.fee_cents for fill in result.fills)
        for fill_ordinal, fill in enumerate(result.fills):
            fill_rows.append({
                "session_date": raw_path.stem,
                "fill_ordinal": fill_ordinal,
                "timestamp_ns": fill.timestamp_ns,
                "side": fill.side,
                "price_ticks": fill.price_ticks,
                "fee_cents": fill.fee_cents,
                "forced": fill.forced,
            })
        # Independent member replays are diagnostics only. Each member owns its
        # own policy-dependent inventory and execution state; no seed P&L is
        # inferred from its vote inside the ensemble environment.
        for member, messages in zip(members, member_messages, strict=True):
            seed, _, trader, member_scaler, _, _ = member
            seed_env = V6TradingEnvironment(raw, raw_path.stem, economics)
            seed_cumulative = 0
            for ordinal, feature_index_value in enumerate(selected):
                feature_index = int(feature_index_value)
                decision = int(session.bottom_clock_ns[feature_index])
                current_raw = raw_index(raw, decision)
                terminal = ordinal == len(selected) - 1
                next_raw = (
                    len(np.asarray(raw["clock_ns"])) - 1
                    if terminal else raw_index(
                        raw, int(session.bottom_clock_ns[int(selected[ordinal + 1])]),
                    )
                )
                state = seed_env.agent_state(current_raw)
                legal = seed_env.legal_actions()
                message = (
                    neutral_regime_message(decision)
                    if representation.publisher_parent_arm is None
                    else latest_message(messages, decision)
                )
                observation = representation.trader_arrays(bottom_observation_at(
                    session, feature_index, member_scaler, message, state,
                ))
                q = trader_q_values(trader, observation)
                masked = q.copy(); masked[~legal] = -np.inf
                action = ACTIONS[int(np.argmax(masked))]
                transition = seed_env.step(
                    current_raw, action, next_raw, terminal=terminal,
                )
                seed_cumulative += transition.reward_cents
                seed_decision_rows.append({
                    "session_date": raw_path.stem,
                    "decision_ordinal": ordinal,
                    "decision_time_ns": transition.decision_time_ns,
                    "next_time_ns": transition.next_time_ns,
                    "seed": seed,
                    "q_short": float(q[0]),
                    "q_flat": float(q[1]),
                    "q_long": float(q[2]),
                    "legal_short": bool(legal[0]),
                    "legal_flat": bool(legal[1]),
                    "legal_long": bool(legal[2]),
                    "action": int(action),
                    "message_short": float(observation["message"][0]),
                    "message_flat": float(observation["message"][1]),
                    "message_long": float(observation["message"][2]),
                    "message_valid": float(observation["message"][3]),
                    "message_lifetime": float(observation["message"][4]),
                    "message_publication_time_ns": int(message.decision_time_ns),
                    "message_expiry_time_ns": int(message.expiry_time_ns),
                    "inventory_before": transition.inventory_before,
                    "inventory_after": transition.inventory_after,
                    "command_created": transition.command_created,
                    "command_rejected": transition.command_rejected,
                    "terminal": transition.done,
                })
                route = transition.route
                terminal_route = transition.terminal_route
                seed_reward_rows.append({
                    "session_date": raw_path.stem,
                    "decision_ordinal": ordinal,
                    "decision_time_ns": transition.decision_time_ns,
                    "next_time_ns": transition.next_time_ns,
                    "seed": seed,
                    "action": int(action),
                    "inventory_before": transition.inventory_before,
                    "inventory_after": transition.inventory_after,
                    "reward_cents": transition.reward_cents,
                    "cumulative_executable_nlv_cents": seed_cumulative,
                    "command_created": transition.command_created,
                    "command_rejected": transition.command_rejected,
                    "route_fill_time_ns": (
                        None if route is None or route.fill is None
                        else route.fill.timestamp_ns
                    ),
                    "route_deadline_ns": (
                        None if route is None else route.deadline_time_ns
                    ),
                    "route_side": None if route is None else route.side,
                    "route_fill_price_ticks": (
                        None if route is None or route.fill is None
                        else route.fill.price_ticks
                    ),
                    "route_fill_fee_cents": (
                        None if route is None or route.fill is None
                        else route.fill.fee_cents
                    ),
                    "route_displayed_size": (
                        None if route is None or route.book is None
                        else (
                            route.book.ask_size if route.side > 0
                            else route.book.bid_size
                        )
                    ),
                    "terminal_fill_time_ns": (
                        None if terminal_route is None or terminal_route.fill is None
                        else terminal_route.fill.timestamp_ns
                    ),
                    "terminal_route_deadline_ns": (
                        None if terminal_route is None
                        else terminal_route.deadline_time_ns
                    ),
                    "terminal_route_side": (
                        None if terminal_route is None else terminal_route.side
                    ),
                    "terminal_fill_price_ticks": (
                        None if terminal_route is None or terminal_route.fill is None
                        else terminal_route.fill.price_ticks
                    ),
                    "terminal_fill_fee_cents": (
                        None if terminal_route is None or terminal_route.fill is None
                        else terminal_route.fill.fee_cents
                    ),
                    "terminal_displayed_size": (
                        None if terminal_route is None or terminal_route.book is None
                        else (
                            terminal_route.book.ask_size if terminal_route.side > 0
                            else terminal_route.book.bid_size
                        )
                    ),
                    "terminal": transition.done,
                })
            seed_result = seed_env.finish()
            if seed_cumulative != seed_result.net_pnl_cents:
                raise AssertionError("per-seed rewards do not telescope to terminal cash")
            if seed_result.transitions[-1].inventory_after != 0:
                raise AssertionError("per-seed terminal inventory is not flat")
            seed_daily[str(seed)][raw_path.stem] = seed_result.net_pnl_cents
            seed_trips[seed] += seed_result.completed_round_trips
            seed_rejections[seed] += seed_result.rejected_commands
            seed_filled_sides[seed] += len(seed_result.fills)
            seed_commissions_cents[seed] += sum(
                fill.fee_cents for fill in seed_result.fills
            )
            for fill_ordinal, fill in enumerate(seed_result.fills):
                seed_fill_rows.append({
                    "session_date": raw_path.stem,
                    "seed": seed,
                    "fill_ordinal": fill_ordinal,
                    "timestamp_ns": fill.timestamp_ns,
                    "side": fill.side,
                    "price_ticks": fill.price_ticks,
                    "fee_cents": fill.fee_cents,
                    "forced": fill.forced,
                })
    if unsafe_legacy_arrival_boundary:
        raise ValueError("registered ablation replay found legacy arrival boundaries")

    decision_path = atomic_parquet(work / "ensemble_decisions.parquet", decision_rows)
    reward_path = atomic_parquet(work / "ensemble_rewards.parquet", reward_rows)
    fill_path = atomic_parquet(work / "ensemble_fills.parquet", fill_rows, schema=FILL_SCHEMA)
    seed_decision_path = atomic_parquet(
        work / "seed_decisions.parquet", seed_decision_rows,
    )
    seed_reward_path = atomic_parquet(
        work / "seed_rewards.parquet", seed_reward_rows,
    )
    seed_fill_path = atomic_parquet(
        work / "seed_fills.parquet", seed_fill_rows, schema=SEED_FILL_SCHEMA,
    )
    checkpoint_set = [
        {
            "seed": seed,
            "path": project_relative(manifest_path),
            "path_scope": "project_root",
            "sha256": file_sha256(manifest_path),
            "trader_state_sha256": manifest["trader_state_sha256"],
            "initialization_sha256": manifest["initialization_sha256"],
        }
        for seed, manifest_path, _, _, manifest, _ in members
    ]
    report = {
        "schema_version": 1,
        "artifact_type": REPLAY_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "status": "COMPLETE",
        "arm": selected_arm.value,
        "period": selected_period,
        "phase": phase,
        "classification": PERIOD_CLASSIFICATION[selected_period],
        "fresh_oos": False,
        "scenario": selected_scenario,
        "economics": asdict(economics),
        "ensemble_seeds": list(SEEDS),
        "ensemble_rule": "legal_seed_argmax_then_target_majority_flat_on_tie_v1",
        "evaluation_dates": list(dataset.dates),
        "daily_pnl_cents": daily,
        "metrics": _metrics(daily, trips),
        "per_seed_daily_pnl_cents": seed_daily,
        "per_seed_metrics": {
            str(seed): _metrics(seed_daily[str(seed)], seed_trips[seed])
            | {
                "rejected_commands": seed_rejections[seed],
                "filled_sides": seed_filled_sides[seed],
                "commissions_cents": seed_commissions_cents[seed],
                "explicit_commissions_usd": seed_commissions_cents[seed] / 100.0,
            }
            for seed in SEEDS
        },
        "completed_round_trips": trips,
        "rejected_commands": rejected,
        "fills": filled_sides,
        "filled_sides": filled_sides,
        "commissions_cents": commissions_cents,
        "explicit_commissions_usd": commissions_cents / 100.0,
        "checkpoint_set": checkpoint_set,
        "dataset_manifest": dataset.binding(),
        "training_manifest": {
            "path": project_relative(training_path),
            "path_scope": "project_root",
            "sha256": file_sha256(training_path),
        },
        "decision_trace": _trace_record(decision_path, len(decision_rows)),
        "reward_trace": _trace_record(reward_path, len(reward_rows)),
        "fill_trace": _trace_record(fill_path, len(fill_rows)),
        "seed_decision_trace": _trace_record(
            seed_decision_path, len(seed_decision_rows),
        ),
        "seed_reward_trace": _trace_record(seed_reward_path, len(seed_reward_rows)),
        "seed_fill_trace": _trace_record(seed_fill_path, len(seed_fill_rows)),
        "per_seed_decision_trace_complete": True,
        "per_seed_policy_dependent_replays_complete": True,
        "ensemble_reward_trace_complete": True,
        "reward_telescope_verified": True,
        "terminal_inventory_flat": True,
        "policy_fallbacks": 0,
        "registered_evaluation": True,
        "smoke_only": False,
        "holdout_read": True,
        "august_accessed": False,
        "unsafe_legacy_arrival_boundary": False,
        "included_in_reporting": True,
        "technical_failure_reason": None,
        "inferential_status": "POST_HOC_DESCRIPTIVE",
    }
    atomic_json(work / "report.json", report)
    work.rename(output)
    verify_replay_report(
        output / "report.json", expected_arm=selected_arm.value,
        expected_period=selected_period, expected_scenario=selected_scenario,
    )
    return output / "report.json"


def verify_replay_report(
    path: Path, *, expected_arm: str | None = None,
    expected_period: str | None = None, expected_scenario: str | None = None,
) -> dict[str, object]:
    report_path = Path(path).resolve()
    value = read_json_object(report_path)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != REPLAY_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("status") != "COMPLETE"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("ensemble_seeds") != list(SEEDS)
        or value.get("registered_evaluation") is not True
        or value.get("smoke_only") is not False
        or value.get("holdout_read") is not True
        or value.get("august_accessed") is not False
        or value.get("included_in_reporting") is not True
        or value.get("technical_failure_reason") is not None
        or value.get("policy_fallbacks") != 0
        or value.get("terminal_inventory_flat") is not True
        or value.get("reward_telescope_verified") is not True
        or value.get("per_seed_decision_trace_complete") is not True
        or value.get("per_seed_policy_dependent_replays_complete") is not True
        or value.get("ensemble_reward_trace_complete") is not True
        or value.get("ensemble_rule")
        != "legal_seed_argmax_then_target_majority_flat_on_tie_v1"
        or value.get("unsafe_legacy_arrival_boundary") is not False
        or value.get("inferential_status") != "POST_HOC_DESCRIPTIVE"
    ):
        raise ValueError("ablation replay report is invalid")
    arm = validate_arm(str(value.get("arm", ""))).value
    verify_implementation_binding(value.get("implementation"))
    period = validate_period(str(value.get("period", ""))).value
    scenario = validate_scenario(str(value.get("scenario", ""))).value
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("ablation replay arm differs")
    if expected_period is not None and period != expected_period:
        raise ValueError("ablation replay period differs")
    if expected_scenario is not None and scenario != expected_scenario:
        raise ValueError("ablation replay scenario differs")
    economics = scenario_economics(scenario)
    if value.get("economics") != asdict(economics):
        raise ValueError("ablation replay economics differ from named scenario")
    if value.get("protocol") != protocol_binding():
        raise ValueError("ablation replay protocol binding differs")
    expected_phase = "validation" if period == "may_2026" else "test"
    if value.get("phase") != expected_phase:
        raise ValueError("ablation replay phase differs from its registered period")
    dates = value.get("evaluation_dates")
    daily = value.get("daily_pnl_cents")
    if not isinstance(dates, list) or not isinstance(daily, dict) or sorted(daily) != dates:
        raise ValueError("ablation replay daily dates are incomplete")
    if list(dates) != sorted(set(str(day) for day in dates)):
        raise ValueError("ablation replay dates are not unique and sorted")
    expected_metrics = _metrics(
        {str(day): int(amount) for day, amount in daily.items()},
        int(value.get("completed_round_trips", -1)),
    )
    if value.get("metrics") != expected_metrics:
        raise ValueError("ablation replay metrics do not reconcile to daily P&L")
    dataset_record = value.get("dataset_manifest")
    if not isinstance(dataset_record, dict) or (
        dataset_record.get("path_scope") != "project_root"
        or dataset_record.get("path") != PERIOD_DATASETS[period]
        or dataset_record.get("purpose") != "evaluate"
        or dataset_record.get("period_id") != period
        or dataset_record.get("dates") != dates
    ):
        raise ValueError("ablation replay dataset binding differs")
    dataset_path = resolve_project_path(dataset_record.get("path"))
    if not dataset_path.is_file() or file_sha256(dataset_path) != dataset_record.get(
        "sha256"
    ):
        raise ValueError("ablation replay dataset manifest hash differs")
    dataset_value = read_json_object(dataset_path)
    sessions = dataset_value.get("sessions")
    if not isinstance(sessions, dict) or sorted(sessions) != dates:
        raise ValueError("ablation replay dates differ from its dataset manifest")
    training_record = value.get("training_manifest")
    if not isinstance(training_record, dict) or training_record.get(
        "path_scope"
    ) != "project_root":
        raise ValueError("ablation replay training binding is malformed")
    training_path = resolve_project_path(training_record.get("path"))
    if not training_path.is_file() or file_sha256(training_path) != training_record.get(
        "sha256"
    ):
        raise ValueError("ablation replay training-manifest hash differs")
    training_value = verify_complete_training_manifest(training_path)
    jobs = training_value.get("jobs")
    assert isinstance(jobs, list)
    expected_checkpoints: set[tuple[int, str, str, str, str]] = set()
    for job in jobs:
        if not isinstance(job, dict) or job.get("arm") != arm:
            continue
        member_path = resolve_bound_path(
            training_path.parent, job["manifest"]["path"],
        )
        member = read_json_object(member_path)
        expected_checkpoints.add((
            int(job["seed"]), str(member_path), str(job["manifest"]["sha256"]),
            str(job["initialization_sha256"]),
            str(member.get("trader_state_sha256")),
        ))
    checkpoints = value.get("checkpoint_set")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(SEEDS):
        raise ValueError("ablation replay checkpoint set is absent")
    observed_checkpoints: set[tuple[int, str, str, str, str]] = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or checkpoint.get(
            "path_scope"
        ) != "project_root":
            raise ValueError("ablation replay checkpoint binding is malformed")
        checkpoint_path = resolve_project_path(checkpoint.get("path"))
        if not checkpoint_path.is_file() or file_sha256(
            checkpoint_path
        ) != checkpoint.get("sha256"):
            raise ValueError("ablation replay checkpoint hash differs")
        observed_checkpoints.add((
            int(checkpoint.get("seed", -1)), str(checkpoint_path),
            str(checkpoint.get("sha256")),
            str(checkpoint.get("initialization_sha256")),
            str(checkpoint.get("trader_state_sha256")),
        ))
    if len(observed_checkpoints) != len(SEEDS) or observed_checkpoints != expected_checkpoints:
        raise ValueError("ablation replay checkpoint set differs from complete training")
    decision = _verify_trace(report_path, value.get("decision_trace"))
    rewards = _verify_trace(report_path, value.get("reward_trace"))
    fills = _verify_trace(report_path, value.get("fill_trace"))
    seed_decisions = _verify_trace(report_path, value.get("seed_decision_trace"))
    seed_rewards = _verify_trace(report_path, value.get("seed_reward_trace"))
    seed_fills = _verify_trace(report_path, value.get("seed_fill_trace"))
    if pq.ParquetFile(decision).metadata.num_rows != 5 * pq.ParquetFile(
        rewards
    ).metadata.num_rows:
        raise ValueError("ablation per-seed decision trace is incomplete")
    if (
        pq.ParquetFile(seed_decisions).metadata.num_rows
        != pq.ParquetFile(seed_rewards).metadata.num_rows
        or pq.ParquetFile(seed_rewards).metadata.num_rows
        != 5 * pq.ParquetFile(rewards).metadata.num_rows
    ):
        raise ValueError("independent per-seed replay traces are incomplete")
    fill_table = pq.read_table(fills, columns=["fee_cents"])
    fee_total = int(sum(fill_table.column("fee_cents").to_pylist()))
    if (
        pq.ParquetFile(fills).metadata.num_rows != int(value.get("filled_sides", -1))
        or fee_total != int(value.get("commissions_cents", -1))
        or value.get("fills") != value.get("filled_sides")
        or value.get("explicit_commissions_usd") != fee_total / 100.0
    ):
        raise ValueError("ablation fill trace does not reconcile")
    seed_daily = value.get("per_seed_daily_pnl_cents")
    seed_metrics = value.get("per_seed_metrics")
    if not isinstance(seed_daily, dict) or not isinstance(seed_metrics, dict):
        raise ValueError("per-seed economic diagnostics are absent")
    seed_fill_table = pq.read_table(seed_fills, columns=["seed", "fee_cents"])
    for seed in SEEDS:
        one = seed_daily.get(str(seed))
        metrics = seed_metrics.get(str(seed))
        if not isinstance(one, dict) or not isinstance(metrics, dict) or sorted(one) != dates:
            raise ValueError("per-seed daily P&L is incomplete")
        expected = _metrics(
            {str(day): int(amount) for day, amount in one.items()},
            int(metrics.get("completed_round_trips", -1)),
        )
        if any(metrics.get(key) != expected_value for key, expected_value in expected.items()):
            raise ValueError("per-seed metrics do not reconcile")
        seed_values = seed_fill_table.column("seed").to_pylist()
        fee_values = seed_fill_table.column("fee_cents").to_pylist()
        filled = sum(1 for value in seed_values if int(value) == seed)
        fees = sum(
            int(fee) for value, fee in zip(seed_values, fee_values, strict=True)
            if int(value) == seed
        )
        if (
            metrics.get("filled_sides") != filled
            or metrics.get("commissions_cents") != fees
            or metrics.get("explicit_commissions_usd") != fees / 100.0
        ):
            raise ValueError("per-seed fills and commissions do not reconcile")
    if pq.ParquetFile(seed_fills).metadata.num_rows != len(
        seed_fill_table.column("seed")
    ):
        raise ValueError("per-seed fill trace is malformed")
    _verify_ensemble_traces(
        report=value, arm=arm, economics=economics,
        decision_path=decision, reward_path=rewards, fill_path=fills,
    )
    _verify_seed_traces(
        report=value, arm=arm, economics=economics,
        decision_path=seed_decisions, reward_path=seed_rewards,
        fill_path=seed_fills,
    )
    return value


def verify_technical_failure_report(
    path: Path, *, expected_arm: str | None = None,
    expected_period: str | None = None, expected_scenario: str | None = None,
) -> dict[str, object]:
    report_path = Path(path).resolve()
    value = read_json_object(report_path)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != TECHNICAL_FAILURE_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status") != "TECHNICAL_FAILURE"
        or value.get("classification") != "CONTAMINATED_HISTORICAL_HOLDOUT"
        or value.get("fresh_oos") is not False
        or value.get("included_in_reporting") is not True
        or value.get("august_accessed") is not False
        or value.get("inferential_status") != "TECHNICALLY_INVALID_REPORTED"
        or not isinstance(value.get("diagnostic"), str)
        or not str(value.get("diagnostic")).strip()
        or value.get("technical_failure_reason") not in TECHNICAL_FAILURE_REASONS
    ):
        raise ValueError("ablation technical-failure report is invalid")
    arm = validate_arm(str(value.get("arm", ""))).value
    period = validate_period(str(value.get("period", ""))).value
    scenario = validate_scenario(str(value.get("scenario", ""))).value
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("ablation technical-failure arm differs")
    if expected_period is not None and period != expected_period:
        raise ValueError("ablation technical-failure period differs")
    if expected_scenario is not None and scenario != expected_scenario:
        raise ValueError("ablation technical-failure scenario differs")
    if value.get("protocol") != protocol_binding():
        raise ValueError("ablation technical-failure protocol binding differs")
    verify_implementation_binding(value.get("implementation"))
    return value


def write_technical_failure(
    *, arm: str, period: str, scenario: str, reason: str,
    diagnostic: str, output_root: Path,
) -> Path:
    selected_arm = validate_arm(arm).value
    selected_period = validate_period(period).value
    selected_scenario = validate_scenario(scenario).value
    if reason not in TECHNICAL_FAILURE_REASONS:
        raise ValueError("technical failure reason is not predeclared")
    destination = (
        Path(output_root).resolve() / "evaluation" / selected_period /
        selected_scenario / selected_arm / "report.json"
    )
    if destination.exists():
        raise FileExistsError("registered evaluation result already exists")
    value = {
        "schema_version": 1,
        "artifact_type": TECHNICAL_FAILURE_ARTIFACT,
        "experiment_id": PROTOCOL_ID,
        "protocol_id": PROTOCOL_ID,
        "protocol": protocol_binding(),
        "implementation": implementation_binding(),
        "status": "TECHNICAL_FAILURE",
        "arm": selected_arm,
        "period": selected_period,
        "scenario": selected_scenario,
        "classification": PERIOD_CLASSIFICATION[selected_period],
        "fresh_oos": False,
        "technical_failure_reason": reason,
        "diagnostic": str(diagnostic),
        "included_in_reporting": True,
        "august_accessed": False,
        "inferential_status": "TECHNICALLY_INVALID_REPORTED",
    }
    atomic_json(destination, value)
    verify_technical_failure_report(
        destination, expected_arm=selected_arm, expected_period=selected_period,
        expected_scenario=selected_scenario,
    )
    return destination


__all__ = [
    "PERIOD_DATASETS", "TECHNICAL_FAILURE_REASONS", "replay_ensemble",
    "verify_replay_report", "verify_technical_failure_report",
    "write_technical_failure",
]
