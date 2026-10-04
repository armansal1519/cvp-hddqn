"""Hash-bound adapters for the three pre-existing benchmark references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .artifacts import file_sha256, read_json_object, workspace_root
from .contract import PERIOD_CLASSIFICATION, SCENARIOS
from .metrics import daily_metrics
from .protocol import reject_forbidden_text


REFERENCE_MANIFESTS = {
    "cvp_hddqn": {
        "may_2026": (
            "viability/artifacts/development/"
            "v6_registered_may_descriptive_evaluation/manifest.json",
            "b9dc8928ae24e8e1c44130dee175819cf2456ccd2c09aa7786f76ac2ba137c71",
        ),
        "june_july_2026": (
            "paper/sealed_extension/result/manifest.json",
            "fb9a8ccbe8d28d64b6dbe70968329d09c103a6c5d773b42eee856a280c2418c1",
        ),
    },
    "no_publisher_single_agent_ddqn": {
        "all": (
            "viability/artifacts/development/"
            "cvp_ablation_extension_v1/evaluation/manifest.json",
            "457161ba1febb172da45b3a719cb02a814049f3fff11c86e0826266803d3afc3",
        ),
    },
    "shavandi_khedmati_reconstruction": {
        "may_2026": (
            "viability/artifacts/development/"
            "basepaper_described_gc_from_scratch_v1/may_descriptive/manifest.json",
            "449ff3f5537a11816b411f8d66eef927ae12a49561cd277e106cabbdcf32f299",
        ),
    },
}

PUBLIC_NAMES = {
    "cvp_hddqn": "CVP-HDDQN",
    "no_publisher_single_agent_ddqn": "No-publisher recurrent DDQN",
    "shavandi_khedmati_reconstruction": "Shavandi-Khedmati reconstruction",
}

SCENARIO_ALIASES = {
    "nominal": "base",
    "double_commission": "double_commission",
    "latency_20ms": "20ms_route_latency",
    "adverse_tick": "one_adverse_tick_per_filled_side",
}


@dataclass(frozen=True)
class ReferenceReport:
    policy_id: str
    public_name: str
    period: str
    scenario: str
    daily_pnl_cents: dict[str, int]
    metrics: dict[str, object]
    filled_sides: int
    rejected_commands: int
    commissions_cents: int
    source_manifest_path: str
    source_manifest_sha256: str
    source_report_path: str
    source_report_sha256: str
    original_classification: str
    classification_for_baseline_suite: str


def _manifest(policy_id: str, period: str) -> tuple[Path, dict[str, object], str]:
    record = REFERENCE_MANIFESTS[policy_id].get(period)
    if record is None:
        record = REFERENCE_MANIFESTS[policy_id].get("all")
    if record is None:
        raise PermissionError(f"{policy_id} has no registered {period} evidence")
    relative, expected_sha = record
    path = workspace_root() / relative
    if not path.is_file() or file_sha256(path) != expected_sha:
        raise ValueError(f"reference manifest changed: {relative}")
    return path, read_json_object(path), expected_sha


def _report_record(
    policy_id: str, period: str, scenario: str, manifest: Mapping[str, object],
) -> Mapping[str, object]:
    reports = manifest.get("reports")
    if not isinstance(reports, list):
        raise ValueError("reference manifest lacks its report registry")
    source_scenario = (
        scenario if policy_id == "no_publisher_single_agent_ddqn"
        else SCENARIO_ALIASES[scenario]
    )
    matches: list[Mapping[str, object]] = []
    for record in reports:
        if not isinstance(record, Mapping) or record.get("scenario") != source_scenario:
            continue
        if policy_id == "cvp_hddqn" and record.get("arm") not in {
            "V6-RichVolume", "CVP-HDDQN",
        }:
            continue
        if policy_id == "no_publisher_single_agent_ddqn":
            if record.get("arm") != "no_publisher" or record.get("period") != period:
                continue
        matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"reference report is absent or duplicated: {policy_id}/{period}/{scenario}"
        )
    return matches[0]


def load_reference_report(policy_id: str, period: str, scenario: str) -> ReferenceReport:
    """Load one exact frozen reference without admitting invalidated arms."""

    reject_forbidden_text(policy_id, period, scenario)
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if policy_id == "shavandi_khedmati_reconstruction" and period != "may_2026":
        raise PermissionError("the Shavandi-Khedmati reconstruction is May-only")
    path, manifest, manifest_sha = _manifest(policy_id, period)
    record = _report_record(policy_id, period, scenario, manifest)
    relative = record.get("path")
    if policy_id == "no_publisher_single_agent_ddqn":
        report_path = path.parent / str(record.get("path"))
        expected_report_sha = str(record.get("sha256"))
        from kx_cvp_ablations.replay import verify_replay_report

        report = verify_replay_report(
            report_path,
            expected_arm="no_publisher",
            expected_period=period,
            expected_scenario=scenario,
        )
    else:
        report_path = path.parent / str(relative)
        expected_report_sha = str(record.get("sha256"))
        if not report_path.is_file() or file_sha256(report_path) != expected_report_sha:
            raise ValueError("reference report hash differs")
        report = read_json_object(report_path)
    if not report_path.is_file() or file_sha256(report_path) != expected_report_sha:
        raise ValueError("reference report hash differs after verification")
    daily_raw = report.get("daily_pnl_cents")
    metrics_raw = report.get("metrics")
    if not isinstance(daily_raw, Mapping) or not isinstance(metrics_raw, Mapping):
        raise ValueError("reference report lacks daily cash or metrics")
    daily = {str(day): int(value) for day, value in daily_raw.items()}
    filled_sides = int(report.get("filled_sides", report.get("fills", -1)))
    rejected = int(report.get("rejected_commands", -1))
    completed = int(metrics_raw.get("completed_round_trips", -1))
    if filled_sides < 0 or filled_sides != 2 * completed or rejected < 0:
        raise ValueError("reference activity does not reconcile")
    recomputed = daily_metrics(daily, completed_round_trips=completed)
    for key in (
        "sessions", "net_pnl_usd", "mean_daily_pnl_usd",
        "annualized_daily_sharpe", "maximum_drawdown_usd",
        "positive_day_fraction", "profit_factor", "completed_round_trips",
    ):
        observed = metrics_raw.get(key)
        expected = recomputed[key]
        if observed is None or expected is None:
            if observed is not expected:
                raise ValueError(f"reference metric {key} differs")
        elif abs(float(observed) - float(expected)) > 1e-9:
            raise ValueError(f"reference metric {key} differs")
    economics = report.get("economics")
    if not isinstance(economics, Mapping):
        raise ValueError("reference economics record is absent")
    fee = int(economics.get("fee_cents_per_side", -1))
    if fee < 0:
        raise ValueError("reference commission is invalid")
    return ReferenceReport(
        policy_id=policy_id,
        public_name=PUBLIC_NAMES[policy_id],
        period=period,
        scenario=scenario,
        daily_pnl_cents=daily,
        metrics=dict(metrics_raw),
        filled_sides=filled_sides,
        rejected_commands=rejected,
        commissions_cents=filled_sides * fee,
        source_manifest_path=path.relative_to(workspace_root()).as_posix(),
        source_manifest_sha256=manifest_sha,
        source_report_path=report_path.relative_to(workspace_root()).as_posix(),
        source_report_sha256=expected_report_sha,
        original_classification=str(report.get("classification", "NOT_REPORTED")),
        classification_for_baseline_suite=PERIOD_CLASSIFICATION[period],
    )


def load_all_reference_reports() -> tuple[ReferenceReport, ...]:
    records: list[ReferenceReport] = []
    for period in ("may_2026", "june_july_2026"):
        for scenario in SCENARIOS:
            records.append(load_reference_report("cvp_hddqn", period, scenario))
            records.append(load_reference_report(
                "no_publisher_single_agent_ddqn", period, scenario,
            ))
            if period == "may_2026":
                records.append(load_reference_report(
                    "shavandi_khedmati_reconstruction", period, scenario,
                ))
    return tuple(records)


__all__ = [
    "PUBLIC_NAMES", "REFERENCE_MANIFESTS", "ReferenceReport",
    "SCENARIO_ALIASES", "load_all_reference_reports", "load_reference_report",
]
