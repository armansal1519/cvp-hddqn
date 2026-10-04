from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .bars import assign_globex_session, generate_gc_lite_trace
from .baseline_pipeline import (
    DEVELOPMENT_LABEL,
    SCREENING_LABEL,
    SESSION_FLATTEN_RULE_ID,
    _as_date,
    _globex_session_close_utc,
    _globex_session_open_utc,
    _only_active_bars,
    _pnl_summary,
    _proxy_episodes,
    _rollup_bars,
    _apply_session_flat_positions,
    _session_flattening_spec,
    aggregate_l1_to_5m,
    development_end_from_config,
    sealed_l1_paths,
)
from .artifacts import sha256_file
from .config import (
    OUTCOME_START,
    enforce_2025_frame,
    enforce_outcome_dates,
    load_yaml,
    resolve_archive_root,
)


ROLL_RULE_ID = "gc_prior_session_volume_crossover_v2"
ROLL_RULE_VERSION = 2
INITIAL_SELECTION_RULE = (
    "highest_cumulative_executed_volume_over_completed_warmup_sessions"
)
DECISION_CLOCK_RULE = "immediately_prior_completed_globex_session"
TIE_BREAK_RULE = "highest_volume_then_nearest_later_maturity"
MISSING_VOLUME_POLICY = (
    "complete_source_partitions_absence_is_observed_zero"
)
GC_MONTH_CODES = {code: month for month, code in enumerate("FGHJKMNQUVXZ", start=1)}
GC_OUTRIGHT_PATTERN = re.compile(r"^GC([FGHJKMNQUVXZ])(\d)$")


@dataclass(frozen=True)
class PastOnlyScreeningResult:
    report: dict[str, Any]
    trace: pd.DataFrame
    episodes: pd.DataFrame
    active_contract_map: pd.DataFrame
    source_paths: tuple[Path, ...]


def gc_contract_maturity(raw_symbol: str, *, year_decade: int = 2020) -> tuple[int, int]:
    """Return the listed maturity for the one-digit GC symbols in this archive."""

    match = GC_OUTRIGHT_PATTERN.fullmatch(str(raw_symbol))
    if match is None:
        raise ValueError(f"unsupported GC outright symbol: {raw_symbol!r}")
    if year_decade % 10:
        raise ValueError("year_decade must end in zero")
    month_code, year_digit = match.groups()
    return int(year_decade) + int(year_digit), GC_MONTH_CODES[month_code]


def _validate_roll_config(config: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = int(config.get("schema_version", 0))
    rule_id = str(config.get("roll_rule_id", ""))
    warmup = int(config.get("warmup_completed_sessions", 0))
    year_decade = int(config.get("contract_year_decade", 0))
    if schema_version != ROLL_RULE_VERSION:
        raise ValueError(f"roll config must have schema_version={ROLL_RULE_VERSION}")
    if rule_id != ROLL_RULE_ID:
        raise ValueError(f"roll config must use roll_rule_id={ROLL_RULE_ID!r}")
    if warmup < 1:
        raise ValueError("warmup_completed_sessions must be positive")
    if year_decade % 10:
        raise ValueError("contract_year_decade must end in zero")
    if bool(config.get("allow_rollback", True)):
        raise ValueError("v2 active-contract selection forbids rollback")
    if str(config.get("roll_comparison", "")) != "strictly_greater":
        raise ValueError("v2 roll_comparison must be 'strictly_greater'")
    initial_selection = str(config.get("initial_selection", ""))
    if initial_selection != INITIAL_SELECTION_RULE:
        raise ValueError(
            f"v2 initial_selection must be {INITIAL_SELECTION_RULE!r}"
        )
    decision_clock = str(config.get("decision_clock", ""))
    if decision_clock != DECISION_CLOCK_RULE:
        raise ValueError(f"v2 decision_clock must be {DECISION_CLOCK_RULE!r}")
    tie_break = str(config.get("tie_break", ""))
    if tie_break != TIE_BREAK_RULE:
        raise ValueError(f"v2 tie_break must be {TIE_BREAK_RULE!r}")
    return {
        "schema_version": schema_version,
        "roll_rule_id": rule_id,
        "warmup_completed_sessions": warmup,
        "contract_year_decade": year_decade,
        "initial_selection": initial_selection,
        "decision_clock": decision_clock,
        "roll_comparison": "strictly_greater",
        "allow_rollback": False,
        "tie_break": tie_break,
    }


def _volume_winner(
    values: pd.Series,
    *,
    symbols: Iterable[str],
    year_decade: int,
) -> str:
    candidates = [str(symbol) for symbol in symbols]
    if not candidates:
        raise ValueError("a volume winner requires at least one candidate")
    # Highest volume wins. An exact tie chooses the nearest listed maturity so
    # the rule cannot jump farther down the curve for an arbitrary string order.
    return min(
        candidates,
        key=lambda symbol: (
            -float(values.get(symbol, 0.0)),
            gc_contract_maturity(symbol, year_decade=year_decade),
            symbol,
        ),
    )


def build_past_only_active_contract_map(
    session_volume: pd.DataFrame,
    *,
    roll_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Select GC contracts without consuming current-session or future volume.

    The first eligible session follows a completed-session warm-up and uses the
    cumulative warm-up winner. Later decisions use only the immediately prior
    completed Globex session. A contract may change only to a later maturity
    whose prior-session volume strictly exceeds the incumbent's volume.
    """

    rule = _validate_roll_config(roll_config)
    required = {
        "session_date",
        "raw_symbol",
        "executed_volume",
        "available_time_ns",
        "session_source_coverage_complete",
    }
    missing = sorted(required.difference(session_volume.columns))
    if missing:
        raise ValueError(f"session volume is missing columns: {missing}")
    frame = session_volume.loc[:, sorted(required)].copy()
    frame["session_date"] = pd.to_datetime(
        frame["session_date"], errors="raise"
    ).dt.date
    frame["raw_symbol"] = frame["raw_symbol"].astype(str)
    frame["executed_volume"] = pd.to_numeric(
        frame["executed_volume"], errors="raise"
    ).astype(float)
    available = pd.to_numeric(frame["available_time_ns"], errors="raise")
    if bool(available.isna().any()) or bool((available <= 0).any()):
        raise ValueError("available_time_ns must contain positive integers")
    if bool((available.astype("int64").astype(float) != available.astype(float)).any()):
        raise ValueError("available_time_ns must contain positive integers")
    frame["available_time_ns"] = available.astype("int64")
    coverage_values = frame["session_source_coverage_complete"]
    if not bool(
        coverage_values.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    ):
        raise TypeError("session_source_coverage_complete must be boolean")
    frame["session_source_coverage_complete"] = coverage_values.astype(bool)
    if frame.empty:
        raise ValueError("session volume cannot be empty")
    if bool((frame["executed_volume"] < 0).any()):
        raise ValueError("executed volume cannot be negative")
    for symbol in frame["raw_symbol"].unique():
        gc_contract_maturity(symbol, year_decade=rule["contract_year_decade"])

    coverage_cardinality = frame.groupby("session_date")[
        "session_source_coverage_complete"
    ].nunique()
    if bool((coverage_cardinality != 1).any()):
        raise ValueError("source coverage flag is inconsistent within a session")
    coverage_by_session = frame.groupby("session_date")[
        "session_source_coverage_complete"
    ].first()
    observed_sessions = sorted(coverage_by_session.index)
    first_complete_offset = next(
        (
            index
            for index, session in enumerate(observed_sessions)
            if bool(coverage_by_session.loc[session])
        ),
        None,
    )
    if first_complete_offset is None:
        raise ValueError("no session has complete source partition coverage")
    leading_incomplete_sessions = observed_sessions[:first_complete_offset]
    incomplete_after_start = [
        str(session)
        for session in observed_sessions[first_complete_offset:]
        if not bool(coverage_by_session.loc[session])
    ]
    if incomplete_after_start:
        raise ValueError(
            "incomplete source partition coverage for roll information sessions: "
            f"{incomplete_after_start}"
        )
    if leading_incomplete_sessions:
        frame = frame.loc[
            ~frame["session_date"].isin(leading_incomplete_sessions)
        ].copy()
        coverage_by_session = coverage_by_session.drop(
            index=leading_incomplete_sessions
        )
    frame = frame.sort_values(
        ["session_date", "available_time_ns", "raw_symbol"], kind="mergesort"
    ).reset_index(drop=True)
    sessions = sorted(frame["session_date"].unique())
    warmup_count = int(rule["warmup_completed_sessions"])
    if len(sessions) <= warmup_count:
        raise ValueError(
            f"need more than {warmup_count} completed sessions; observed {len(sessions)}"
        )

    warmup_sessions = sessions[:warmup_count]
    first_selected_session = sessions[warmup_count]
    first_decision_time_ns = int(
        _globex_session_open_utc(first_selected_session).value
    )
    incomplete_warmup = [
        str(session)
        for session in warmup_sessions
        if not bool(coverage_by_session.loc[session])
    ]
    if incomplete_warmup:
        raise ValueError(
            "incomplete source partition coverage for roll information sessions: "
            f"{incomplete_warmup}"
        )
    warmup_all = frame.loc[frame["session_date"].isin(warmup_sessions)]
    warmup_rows = warmup_all.loc[
        warmup_all["available_time_ns"] < first_decision_time_ns
    ]
    if warmup_rows.empty:
        raise ValueError("no warm-up volume was available before the first decision")
    warmup_volume = warmup_rows.groupby("raw_symbol")["executed_volume"].sum()
    incumbent = _volume_winner(
        warmup_volume,
        symbols=warmup_volume.index,
        year_decade=rule["contract_year_decade"],
    )
    initial_warmup_volume = float(warmup_volume[incumbent])
    rows: list[dict[str, Any]] = []

    for offset, selected_session in enumerate(sessions[warmup_count:]):
        session_index = warmup_count + offset
        prior_session = sessions[session_index - 1]
        decision_time_ns = int(_globex_session_open_utc(selected_session).value)
        information_sessions = warmup_sessions if offset == 0 else [prior_session]
        incomplete_sessions = [
            str(session)
            for session in information_sessions
            if not bool(coverage_by_session.loc[session])
        ]
        if incomplete_sessions:
            raise ValueError(
                "incomplete source partition coverage for roll information sessions: "
                f"{incomplete_sessions}"
            )
        information_all = frame.loc[
            frame["session_date"].isin(information_sessions)
        ]
        information_rows = information_all.loc[
            information_all["available_time_ns"] < decision_time_ns
        ]
        late_information_rows = information_all.loc[
            information_all["available_time_ns"] >= decision_time_ns
        ]
        volume_cutoff_available_time_ns = (
            int(information_rows["available_time_ns"].max())
            if not information_rows.empty
            else 0
        )
        if volume_cutoff_available_time_ns >= decision_time_ns:
            raise AssertionError(
                "roll volume availability cutoff is not before its decision"
            )

        prior_rows = frame.loc[
            frame["session_date"].eq(prior_session)
            & frame["available_time_ns"].lt(decision_time_ns)
        ]
        known_rows = frame.loc[
            frame["session_date"].le(prior_session)
            & frame["available_time_ns"].lt(decision_time_ns)
        ]
        eligible_symbols = sorted(known_rows["raw_symbol"].unique().tolist())
        if not eligible_symbols:
            raise ValueError("no contract was observable before the roll decision")
        # A missing contract row is converted to zero only after the complete
        # source-partition check above.  The eligible universe contains only
        # symbols observed by this decision, never symbols first seen later.
        prior_observed = prior_rows.groupby("raw_symbol")["executed_volume"].sum()
        prior_scores = prior_observed.reindex(eligible_symbols, fill_value=0.0)
        incumbent_before = incumbent
        incumbent_maturity = gc_contract_maturity(
            incumbent_before, year_decade=rule["contract_year_decade"]
        )
        prior_incumbent_volume = float(prior_scores.get(incumbent_before, 0.0))
        later_symbols = [
            str(symbol)
            for symbol in prior_scores.index
            if gc_contract_maturity(
                str(symbol), year_decade=rule["contract_year_decade"]
            )
            > incumbent_maturity
        ]
        candidate = (
            _volume_winner(
                prior_scores,
                symbols=later_symbols,
                year_decade=rule["contract_year_decade"],
            )
            if later_symbols
            else None
        )
        candidate_volume = (
            float(prior_scores.get(candidate, 0.0)) if candidate is not None else None
        )
        rolled = False
        # The warm-up establishes the first incumbent. Every selected session,
        # including the first, then applies the immediately prior completed
        # session's crossover information before the session begins.
        if candidate is not None and candidate_volume is not None:
            if candidate_volume > prior_incumbent_volume:
                incumbent = candidate
                rolled = True

        selected_year, selected_month = gc_contract_maturity(
            incumbent, year_decade=rule["contract_year_decade"]
        )
        rows.append(
            {
                "date": str(selected_session),
                "session_date": str(selected_session),
                "raw_symbol": incumbent,
                "selected_raw_symbol": incumbent,
                "selected_maturity": f"{selected_year:04d}-{selected_month:02d}",
                "incumbent_before": incumbent_before,
                "prior_completed_session": str(prior_session),
                "volume_information_cutoff_session": str(prior_session),
                "decision_time_utc_ns": decision_time_ns,
                "volume_cutoff_available_time_ns": (
                    volume_cutoff_available_time_ns
                ),
                "prior_incumbent_volume": prior_incumbent_volume,
                "prior_later_candidate": candidate,
                "prior_later_candidate_volume": candidate_volume,
                "rolled": rolled,
                "selection_basis": (
                    "completed_warmup_then_prior_session_crossover"
                    if offset == 0
                    else "prior_session_crossover"
                ),
                "completed_sessions_before_decision": session_index,
                "warmup_completed_sessions": warmup_count,
                "initial_warmup_incumbent_volume": initial_warmup_volume,
                "source_coverage_complete": True,
                "leading_incomplete_sessions_quarantined": len(
                    leading_incomplete_sessions
                ),
                "leading_incomplete_session_labels": ",".join(
                    str(session) for session in leading_incomplete_sessions
                ),
                "source_coverage_session_count": len(information_sessions),
                "eligible_contract_count": len(eligible_symbols),
                "prior_observed_contract_count": int(prior_observed.index.nunique()),
                "prior_absent_contracts_treated_as_zero": int(
                    len(set(eligible_symbols).difference(prior_observed.index))
                ),
                "missing_volume_policy": MISSING_VOLUME_POLICY,
                "volume_information_rows_used": int(len(information_rows)),
                "volume_rows_excluded_after_decision": int(
                    len(late_information_rows)
                ),
                "volume_excluded_after_decision": float(
                    late_information_rows["executed_volume"].sum()
                ),
                "same_session_volume_used": False,
                "rollback_allowed": False,
                "roll_rule_id": ROLL_RULE_ID,
                "roll_rule_version": ROLL_RULE_VERSION,
            }
        )

    result = pd.DataFrame(rows)
    selected_maturities = [
        gc_contract_maturity(
            symbol, year_decade=rule["contract_year_decade"]
        )
        for symbol in result["selected_raw_symbol"]
    ]
    if any(right < left for left, right in zip(selected_maturities, selected_maturities[1:])):
        raise AssertionError("past-only roll map attempted a maturity rollback")
    selected_dates = pd.to_datetime(result["session_date"]).dt.date
    cutoffs = pd.to_datetime(result["volume_information_cutoff_session"]).dt.date
    if bool((cutoffs >= selected_dates).any()):
        raise AssertionError("roll map consumed same-session or future volume")
    decision_times = pd.to_numeric(
        result["decision_time_utc_ns"], errors="raise"
    ).astype("int64")
    volume_cutoff_times = pd.to_numeric(
        result["volume_cutoff_available_time_ns"], errors="raise"
    ).astype("int64")
    if bool((volume_cutoff_times >= decision_times).any()):
        raise AssertionError(
            "roll map consumed volume unavailable at its decision time"
        )
    if not bool(result["source_coverage_complete"].astype(bool).all()):
        raise AssertionError("roll map contains incomplete source coverage")
    return enforce_2025_frame(result, date_column="date").reset_index(drop=True)


def _session_contract_volume(
    bars_5m: pd.DataFrame,
    *,
    source_partition_dates: Iterable[Any],
    degraded_utc_dates: Iterable[Any] = (),
) -> pd.DataFrame:
    """Expose bar-level volume with the availability and source-coverage proof.

    Deliberately retaining the 5-minute observations lets the roll builder
    exclude late or recovery data at each decision instead of certifying a
    session total whose components were not all available at that time.
    """

    required = {"bar_end", "raw_symbol", "volume", "available_time_ns"}
    missing = sorted(required.difference(bars_5m.columns))
    if missing:
        raise ValueError(f"5-minute bars are missing volume evidence: {missing}")
    frame = bars_5m.copy()
    frame["session_date"] = assign_globex_session(
        frame["bar_end"] - pd.Timedelta(1, unit="ns")
    )
    frame = frame.loc[frame["session_date"].notna()].copy()
    present_dates = {_as_date(value) for value in source_partition_dates}
    degraded_dates = {_as_date(value) for value in degraded_utc_dates}
    if not present_dates:
        raise ValueError("source partition dates cannot be empty")

    coverage: dict[date, tuple[bool, str]] = {}
    for session in sorted(frame["session_date"].unique()):
        open_time = _globex_session_open_utc(session)
        close_time = _globex_session_close_utc(session)
        required_dates = {
            timestamp.date()
            for timestamp in pd.date_range(
                open_time.normalize(),
                (close_time - pd.Timedelta(1, unit="ns")).normalize(),
                freq="D",
            )
        }
        complete = required_dates.issubset(present_dates) and not bool(
            required_dates.intersection(degraded_dates)
        )
        coverage[session] = (
            complete,
            ",".join(str(value) for value in sorted(required_dates)),
        )

    frame["executed_volume"] = pd.to_numeric(frame.pop("volume"), errors="raise")
    frame["available_time_ns"] = pd.to_numeric(
        frame["available_time_ns"], errors="raise"
    ).astype("int64")
    frame["session_source_coverage_complete"] = frame["session_date"].map(
        lambda value: coverage[value][0]
    )
    frame["required_source_utc_dates"] = frame["session_date"].map(
        lambda value: coverage[value][1]
    )
    return frame[
        [
            "session_date",
            "raw_symbol",
            "executed_volume",
            "available_time_ns",
            "session_source_coverage_complete",
            "required_source_utc_dates",
        ]
    ].sort_values(
        ["session_date", "available_time_ns", "raw_symbol"], kind="mergesort"
    ).reset_index(drop=True)


def screening_concentration_diagnostics(episodes: pd.DataFrame) -> dict[str, Any]:
    if episodes.empty:
        return {
            "session_days": 0,
            "total_net_proxy_pnl_usd": 0.0,
            "largest_positive_day": None,
            "largest_positive_day_net_proxy_pnl_usd": None,
            "net_without_largest_positive_day_usd": None,
            "largest_positive_day_share_of_total_positive_pnl": None,
            "largest_absolute_day_share_of_total_absolute_daily_pnl": None,
            "absolute_daily_pnl_hhi": None,
            "positive_days": 0,
            "negative_days": 0,
            "concentration_warning": "no completed proxy episodes",
            "daily_net_proxy_pnl_usd": [],
        }
    daily = (
        episodes.groupby("date")["net_proxy_pnl_usd"]
        .sum()
        .astype(float)
        .sort_index()
    )
    total = float(daily.sum())
    positive = daily.loc[daily > 0]
    absolute = daily.abs()
    absolute_total = float(absolute.sum())
    largest_absolute_date = str(absolute.idxmax())
    if positive.empty:
        largest_positive_date = None
        largest_positive_value = None
        without_largest = None
        positive_share = None
    else:
        largest_positive_date = str(positive.idxmax())
        largest_positive_value = float(positive.max())
        without_largest = total - largest_positive_value
        positive_share = largest_positive_value / float(positive.sum())
    absolute_share = (
        float(absolute.max()) / absolute_total if absolute_total > 0 else None
    )
    absolute_hhi = (
        float(np.square(absolute / absolute_total).sum())
        if absolute_total > 0
        else None
    )
    warning = None
    if largest_positive_value is not None and total > 0 and without_largest <= 0:
        warning = "positive aggregate proxy P&L becomes nonpositive without its best day"
    elif absolute_share is not None and absolute_share >= 0.50:
        warning = "at least half of absolute daily proxy P&L is concentrated in one day"
    return {
        "session_days": int(len(daily)),
        "total_net_proxy_pnl_usd": total,
        "largest_positive_day": largest_positive_date,
        "largest_positive_day_net_proxy_pnl_usd": largest_positive_value,
        "largest_absolute_day": largest_absolute_date,
        "largest_absolute_day_net_proxy_pnl_usd": float(daily.loc[largest_absolute_date]),
        "net_without_largest_positive_day_usd": without_largest,
        "largest_positive_day_share_of_total_positive_pnl": positive_share,
        "largest_positive_day_share_of_total_net_pnl": (
            largest_positive_value / total
            if largest_positive_value is not None and total != 0
            else None
        ),
        "largest_absolute_day_share_of_total_absolute_daily_pnl": absolute_share,
        "absolute_daily_pnl_hhi": absolute_hhi,
        "positive_days": int((daily > 0).sum()),
        "negative_days": int((daily < 0).sum()),
        "zero_days": int((daily == 0).sum()),
        "concentration_warning": warning,
        "daily_net_proxy_pnl_usd": [
            {"date": str(day), "net_proxy_pnl_usd": float(value)}
            for day, value in daily.items()
        ],
    }


def run_gc_lite_past_only_screening(
    archive_root: str | Path,
    *,
    start: Any,
    end: Any,
    gc_config: Mapping[str, Any],
    roll_config: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    degraded_dates: Iterable[Any] = (),
    development_end: Any = date(2025, 8, 31),
) -> PastOnlyScreeningResult:
    """Run the versioned past-only GC-lite screen without opening later outcomes."""

    start_date = _as_date(start)
    end_date = _as_date(end)
    development_end_date = _as_date(development_end)
    enforce_outcome_dates(start_date, end_date)
    if end_date > development_end_date:
        raise PermissionError(
            "past-only GC-lite screening is development-only; requested end "
            f"{end_date} is after {development_end_date}"
        )
    rule = _validate_roll_config(roll_config)
    session_flattening = _session_flattening_spec(gc_config)
    paths = sealed_l1_paths(archive_root, through=end_date)
    bars_5m = aggregate_l1_to_5m(
        paths,
        through=end_date,
        excluded_utc_dates=degraded_dates,
    )
    bars_15m = _rollup_bars(bars_5m, 15)
    bars_1h = _rollup_bars(bars_5m, 60)
    session_volume = _session_contract_volume(
        bars_5m,
        source_partition_dates=(
            _partition_date_from_l1_path(Path(path)) for path in paths
        ),
        degraded_utc_dates=degraded_dates,
    )
    active = build_past_only_active_contract_map(
        session_volume,
        roll_config=roll_config,
    )

    signals = gc_config["signals"]
    signal_parameters = {
        "h1_fast_ema": int(signals["h1_fast_ema"]),
        "h1_slow_ema": int(signals["h1_slow_ema"]),
        "m15_fast_ema": int(signals["m15_fast_ema"]),
        "m15_slow_ema": int(signals["m15_slow_ema"]),
        "m5_fast_ema": int(signals["m5_fast_ema"]),
        "m5_slow_ema": int(signals["m5_slow_ema"]),
        "maximum_holding_m5_bars": int(signals["maximum_holding_m5_bars"]),
    }
    trace = generate_gc_lite_trace(
        bars_5m,
        bars_15m,
        bars_1h,
        **signal_parameters,
    )
    trace = _only_active_bars(trace, active[["date", "raw_symbol"]])
    if trace.empty:
        raise ValueError("past-only active-contract selection produced an empty trace")
    trace = trace.merge(
        bars_5m[
            [
                "raw_symbol",
                "continuity_segment_id",
                "bar_end",
                "open",
                "high",
                "low",
                "volume",
            ]
        ],
        on=["raw_symbol", "continuity_segment_id", "bar_end"],
        how="left",
        validate="one_to_one",
    )
    trace = _apply_session_flat_positions(
        trace,
        maximum_holding_m5_bars=signal_parameters["maximum_holding_m5_bars"],
        session_flattening=session_flattening,
    )
    trace = trace.loc[
        trace["session_date"].astype(str).between(str(start_date), str(end_date))
    ].copy()
    trace["roll_rule_id"] = ROLL_RULE_ID
    trace["roll_rule_version"] = ROLL_RULE_VERSION
    trace["evidence_class"] = SCREENING_LABEL
    trace = enforce_2025_frame(trace).reset_index(drop=True)

    fill_config = execution_config["fills"]
    cost_config = execution_config["costs"]
    instrument_config = execution_config["instrument"]
    episodes = _proxy_episodes(
        trace,
        adverse_slippage_ticks_per_side=int(
            fill_config["aggressive_stress_slippage_ticks"]
        ),
        tick_value_usd=float(instrument_config["tick_value_usd"]),
        fee_usd_per_side=float(cost_config["fee_usd_per_contract_side"]),
    )
    if not episodes.empty:
        episodes["roll_rule_id"] = ROLL_RULE_ID
        episodes["roll_rule_version"] = ROLL_RULE_VERSION
        episodes = enforce_2025_frame(episodes)

    active_in_range = active.loc[
        active["date"].between(str(start_date), str(end_date))
    ].copy()
    entry_count = int(
        ((trace["target"] != 0) & (trace["target"].shift(fill_value=0) == 0)).sum()
    )
    concentration = screening_concentration_diagnostics(episodes)
    report = {
        "schema_version": ROLL_RULE_VERSION,
        "screen_id": "gc_lite_past_only_roll_v2",
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "warning": (
            "This is a corrected development-only GC-lite proxy using aggregated "
            "L1 trade prices and next-completed-bar fills. It is not Base-GC, "
            "queue-realizable performance, a novelty verdict, or final OOS evidence."
        ),
        "source_layer": "l1_trades",
        "aggregation_pushdown": "duckdb_5m_ohlcv",
        "sealed_2026_read": False,
        "calibration_or_locked_outcomes_read": False,
        "source_partition_years": [2025],
        "scan_start": str(OUTCOME_START),
        "requested_start": str(start_date),
        "requested_end": str(end_date),
        "development_end": str(development_end_date),
        "event_time_end_exclusive_utc": str(
            _globex_session_close_utc(end_date)
        ),
        "input_parquet_files": len(paths),
        "excluded_degraded_utc_dates": sorted(
            str(_as_date(value))
            for value in degraded_dates
            if _as_date(value) <= end_date
        ),
        "aggregate_5m_bars": int(len(bars_5m)),
        "active_sessions": int(active_in_range["date"].nunique()),
        "active_contracts": sorted(
            active_in_range["raw_symbol"].astype(str).unique().tolist()
        ),
        "active_contract_map_rows": int(len(active_in_range)),
        "roll_rule": rule,
        "session_flattening": {
            "rule_id": SESSION_FLATTEN_RULE_ID,
            **session_flattening,
            "fill_horizon_bars": 1,
            "bar_minutes": 5,
        },
        "roll_events": active_in_range.loc[
            active_in_range["rolled"],
            [
                "date",
                "incumbent_before",
                "selected_raw_symbol",
                "prior_completed_session",
                "prior_incumbent_volume",
                "prior_later_candidate_volume",
            ],
        ].to_dict(orient="records"),
        "same_session_or_future_volume_used": False,
        "volume_availability_cutoff_violations": int(
            (
                active_in_range["volume_cutoff_available_time_ns"].astype("int64")
                >= active_in_range["decision_time_utc_ns"].astype("int64")
            ).sum()
        ),
        "incomplete_source_coverage_rows": int(
            (~active_in_range["source_coverage_complete"].astype(bool)).sum()
        ),
        "late_or_recovery_volume_rows_excluded": int(
            active_in_range["volume_rows_excluded_after_decision"]
            .astype(int)
            .sum()
        ),
        "missing_volume_policy": MISSING_VOLUME_POLICY,
        "rollback_count": 0,
        "continuity_segments": sorted(
            int(value) for value in trace["continuity_segment_id"].unique()
        ),
        "episodes_cross_continuity": False,
        "flat_to_direction_entries": entry_count,
        **_pnl_summary(episodes, trace["session_date"]),
        "concentration_diagnostics": concentration,
    }
    return PastOnlyScreeningResult(
        report=report,
        trace=trace,
        episodes=episodes.reset_index(drop=True),
        active_contract_map=active_in_range.reset_index(drop=True),
        source_paths=tuple(Path(path) for path in paths),
    )


def _verify_file_binding(
    name: str,
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, str]]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{name} binding must be a mapping")
    raw_path = str(binding.get("path", "")).strip()
    expected = str(binding.get("sha256", "")).strip()
    if not raw_path:
        raise ValueError(f"{name} binding must contain a path")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{name} binding must contain a lowercase SHA-256")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} bound file does not exist: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{name} hash mismatch: expected {expected}, observed {observed}"
        )
    return path, {"path": str(path), "sha256": observed}


def _verify_named_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
    *,
    required: set[str],
    kind: str,
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    if not isinstance(bindings, Mapping):
        raise ValueError(f"{kind} bindings must be a mapping")
    missing = sorted(required.difference(bindings))
    if missing:
        raise ValueError(f"{kind} bindings are missing: {missing}")
    paths: dict[str, Path] = {}
    verified: dict[str, dict[str, str]] = {}
    for name in sorted(bindings):
        path, normalized = _verify_file_binding(
            f"{kind} {name!r}", bindings[name]
        )
        paths[name] = path
        verified[name] = normalized
    return paths, verified


def _assert_bound_frame_matches(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    name: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError(f"{name} does not match causal recomputation: {exc}") from exc


def _partition_date_from_l1_path(path: Path) -> str:
    try:
        year = int(path.parent.parent.parent.name.removeprefix("year="))
        month = int(path.parent.parent.name.removeprefix("month="))
        day = int(path.parent.name.removeprefix("day="))
        return str(date(year, month, day))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"invalid bound L1 partition path: {path}") from exc


def _verify_source_chain(
    *,
    archive_root: Path,
    through: date,
    source_l1_bindings: Iterable[Mapping[str, Any]],
    source_manifest_bindings: Iterable[Mapping[str, Any]],
) -> tuple[list[Path], list[dict[str, str]], list[dict[str, str]]]:
    declared_l1 = list(source_l1_bindings)
    declared_manifests = list(source_manifest_bindings)
    if not declared_l1 or not declared_manifests:
        raise ValueError("certificate requires nonempty L1 and manifest bindings")

    verified_l1: list[dict[str, str]] = []
    l1_paths: list[Path] = []
    for index, binding in enumerate(declared_l1):
        path, normalized = _verify_file_binding(f"source L1 #{index}", binding)
        l1_paths.append(path)
        verified_l1.append(normalized)
    expected_paths = sealed_l1_paths(archive_root, through=through)
    if sorted(l1_paths) != sorted(expected_paths):
        raise ValueError(
            "bound L1 files are not the complete sealed source set through "
            f"{through}: expected {len(expected_paths)}, observed {len(l1_paths)}"
        )

    manifest_by_date: dict[
        str, tuple[Path, dict[str, str], Mapping[str, Any]]
    ] = {}
    for index, binding in enumerate(declared_manifests):
        if not isinstance(binding, Mapping):
            raise ValueError(f"source manifest #{index} binding must be a mapping")
        path, normalized = _verify_file_binding(
            f"source manifest #{index}", binding
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"source manifest is not an object: {path}")
        declared_date = str(binding.get("date", payload.get("date", "")))
        if declared_date != str(payload.get("date", "")):
            raise ValueError(f"source manifest date mismatch: {path}")
        if declared_date in manifest_by_date:
            raise ValueError(f"duplicate source manifest date: {declared_date}")
        if str(payload.get("status", "")) != "passed":
            raise ValueError(
                f"source manifest {declared_date} is not healthy: "
                f"{payload.get('status')!r}"
            )
        expected_manifest = (
            archive_root / "_manifests" / f"{declared_date}.json"
        ).resolve()
        if path != expected_manifest:
            raise ValueError(
                f"source manifest path mismatch for {declared_date}: {path}"
            )
        manifest_by_date[declared_date] = (path, normalized, payload)

    expected_dates = {_partition_date_from_l1_path(path) for path in expected_paths}
    if set(manifest_by_date) != expected_dates:
        raise ValueError(
            "source manifest dates do not exactly cover bound L1 partitions: "
            f"expected {sorted(expected_dates)}, observed {sorted(manifest_by_date)}"
        )

    l1_hash_by_path = {
        Path(binding["path"]).resolve(): binding["sha256"] for binding in verified_l1
    }
    for source_path in expected_paths:
        source_date = _partition_date_from_l1_path(source_path)
        payload = manifest_by_date[source_date][2]
        relative = str(source_path.relative_to(archive_root))
        matches = [
            item
            for item in payload.get("outputs", [])
            if isinstance(item, Mapping)
            and str(item.get("relative_path", "")) == relative
            and str(item.get("layer", "")) == "l1_trades"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"manifest {source_date} does not bind exactly one L1 output {relative}"
            )
        if str(matches[0].get("sha256", "")) != l1_hash_by_path[source_path]:
            raise ValueError(f"manifest-to-L1 hash mismatch for {source_path}")

    verified_manifest_bindings = [
        {
            "date": source_date,
            **manifest_by_date[source_date][1],
        }
        for source_date in sorted(manifest_by_date)
    ]
    verified_l1.sort(key=lambda value: value["path"])
    return sorted(expected_paths), verified_l1, verified_manifest_bindings


def _validate_recomputed_session_flattening(trace: pd.DataFrame) -> None:
    required = {
        "bar_end",
        "target",
        "scheduled_flatten_bar_end_ns",
        "research_proxy_fill_cutoff_time_ns",
        "session_flatten_rule_id",
    }
    missing = sorted(required.difference(trace.columns))
    if missing:
        raise ValueError(f"recomputed trace lacks flattening evidence: {missing}")
    if set(trace["session_flatten_rule_id"].astype(str)) != {
        SESSION_FLATTEN_RULE_ID
    }:
        raise ValueError("recomputed trace uses an unexpected session flattening rule")
    bar_end_ns = trace["bar_end"].astype("int64")
    flatten_ns = trace["scheduled_flatten_bar_end_ns"].astype("int64")
    if bool(trace.loc[bar_end_ns >= flatten_ns, "target"].astype(int).ne(0).any()):
        raise ValueError("recomputed trace is not flat at its ex-ante session cutoff")


def build_active_contract_certificate(
    *,
    active_contract_map: pd.DataFrame,
    roll_config: Mapping[str, Any],
    artifact_bindings: Mapping[str, Mapping[str, str]],
    config_bindings: Mapping[str, Mapping[str, str]],
    source_l1_bindings: Iterable[Mapping[str, str]],
    source_manifest_bindings: Iterable[Mapping[str, str]],
) -> dict[str, Any]:
    rule = _validate_roll_config(roll_config)
    if active_contract_map.empty:
        raise ValueError("cannot certify an empty active-contract map")

    artifact_paths, verified_artifacts = _verify_named_bindings(
        artifact_bindings,
        required={"active_contract_map", "trace", "episodes"},
        kind="artifact",
    )
    config_paths, verified_configs = _verify_named_bindings(
        config_bindings,
        required={
            "data",
            "execution",
            "gc_lite_v1_signals",
            "past_only_roll_v2",
            "splits",
        },
        kind="config",
    )
    bound_roll_config = load_yaml(config_paths["past_only_roll_v2"])
    if bound_roll_config != dict(roll_config):
        raise ValueError("roll_config does not match its bound configuration artifact")

    bound_map = pd.read_parquet(artifact_paths["active_contract_map"])
    _assert_bound_frame_matches(
        bound_map,
        active_contract_map,
        name="active-contract map argument",
    )
    map_evidence_columns = {
        "decision_time_utc_ns",
        "volume_cutoff_available_time_ns",
        "source_coverage_complete",
        "eligible_contract_count",
        "prior_observed_contract_count",
        "prior_absent_contracts_treated_as_zero",
        "missing_volume_policy",
        "volume_rows_excluded_after_decision",
    }
    missing_map_evidence = sorted(
        map_evidence_columns.difference(active_contract_map.columns)
    )
    if missing_map_evidence:
        raise ValueError(
            "active-contract map lacks causal volume evidence: "
            f"{missing_map_evidence}"
        )
    selected_dates = pd.to_datetime(active_contract_map["date"]).dt.date
    cutoff_dates = pd.to_datetime(
        active_contract_map["volume_information_cutoff_session"]
    ).dt.date
    maturities = [
        gc_contract_maturity(
            symbol, year_decade=rule["contract_year_decade"]
        )
        for symbol in active_contract_map["selected_raw_symbol"]
    ]
    same_or_future = int((cutoff_dates >= selected_dates).sum())
    decision_times_ns = pd.to_numeric(
        active_contract_map["decision_time_utc_ns"], errors="raise"
    ).astype("int64")
    volume_cutoffs_ns = pd.to_numeric(
        active_contract_map["volume_cutoff_available_time_ns"], errors="raise"
    ).astype("int64")
    volume_availability_violations = int(
        (volume_cutoffs_ns >= decision_times_ns).sum()
    )
    coverage_values = active_contract_map["source_coverage_complete"]
    if not bool(
        coverage_values.map(lambda value: isinstance(value, (bool, np.bool_))).all()
    ):
        raise TypeError("active-contract source coverage evidence must be boolean")
    incomplete_coverage_rows = int((~coverage_values.astype(bool)).sum())
    unexpected_missing_policy_rows = int(
        active_contract_map["missing_volume_policy"]
        .astype(str)
        .ne(MISSING_VOLUME_POLICY)
        .sum()
    )
    rollback_count = sum(
        right < left for left, right in zip(maturities, maturities[1:])
    )
    warmup_ok = bool(
        (
            active_contract_map["completed_sessions_before_decision"].astype(int)
            >= int(rule["warmup_completed_sessions"])
        ).all()
    )
    assignment_frame = active_contract_map[["date", "selected_raw_symbol"]].copy()
    assignment_frame["date"] = assignment_frame["date"].astype(str)
    assignment_frame["selected_raw_symbol"] = assignment_frame[
        "selected_raw_symbol"
    ].astype(str)
    unique_date_symbol = bool(
        not assignment_frame["date"].duplicated().any()
        and assignment_frame["date"].str.strip().ne("").all()
        and assignment_frame["selected_raw_symbol"].str.strip().ne("").all()
    )
    assignments = [
        {"date": str(row.date), "raw_symbol": str(row.selected_raw_symbol)}
        for row in assignment_frame.itertuples(index=False)
    ]
    first_selected = min(selected_dates)
    last_selected = max(selected_dates)
    data_config = load_yaml(config_paths["data"])
    archive_root = resolve_archive_root(data_config)
    verified_source_paths, verified_l1, verified_manifests = _verify_source_chain(
        archive_root=archive_root,
        through=last_selected,
        source_l1_bindings=source_l1_bindings,
        source_manifest_bindings=source_manifest_bindings,
    )
    recomputed = run_gc_lite_past_only_screening(
        archive_root,
        start=first_selected,
        end=last_selected,
        gc_config=load_yaml(config_paths["gc_lite_v1_signals"]),
        roll_config=bound_roll_config,
        execution_config=load_yaml(config_paths["execution"]),
        degraded_dates=data_config.get("degraded_dates", ()),
        development_end=development_end_from_config(
            load_yaml(config_paths["splits"])
        ),
    )
    if sorted(path.resolve() for path in recomputed.source_paths) != sorted(
        path.resolve() for path in verified_source_paths
    ):
        raise ValueError(
            "causal recomputation did not consume the complete bound source set"
        )
    _assert_bound_frame_matches(
        recomputed.active_contract_map,
        bound_map,
        name="bound active-contract map",
    )
    bound_trace = pd.read_parquet(artifact_paths["trace"])
    bound_episodes = pd.read_parquet(artifact_paths["episodes"])
    _assert_bound_frame_matches(recomputed.trace, bound_trace, name="bound trace")
    _assert_bound_frame_matches(
        recomputed.episodes,
        bound_episodes,
        name="bound episodes",
    )
    _validate_recomputed_session_flattening(recomputed.trace)

    map_sha256 = verified_artifacts["active_contract_map"]["sha256"]
    passed = (
        same_or_future == 0
        and volume_availability_violations == 0
        and incomplete_coverage_rows == 0
        and unexpected_missing_policy_rows == 0
        and rollback_count == 0
        and warmup_ok
        and unique_date_symbol
    )
    return {
        "schema_version": ROLL_RULE_VERSION,
        "certificate_type": "PAST_ONLY_ACTIVE_CONTRACT_SELECTION",
        "certificate_id": ROLL_RULE_ID,
        "evidence_class": SCREENING_LABEL,
        "split_class": DEVELOPMENT_LABEL,
        "passed": passed,
        # These fields intentionally match the BranchProduct active-contract
        # claim contract so this certificate can be embedded without caller
        # assertions or an ad-hoc translation layer.
        "method": "past-only prior-completed-session volume crossover",
        "past_only": True,
        "minimum_lag_sessions": 1,
        "unique_date_symbol": unique_date_symbol,
        "sha256": map_sha256,
        "artifact_sha256": map_sha256,
        "map_sha256": map_sha256,
        "artifact": verified_artifacts["active_contract_map"]["path"],
        "assignments": assignments,
        "date_symbol_assignments": assignments,
        "claim_scope": (
            "The daily GC contract map is selected from completed past volume only. "
            "This certificate makes no profitability, Base-GC, execution, or novelty claim."
        ),
        "rule": rule,
        "checks": {
            "five_completed_session_warmup_passed": warmup_ok,
            "same_or_future_session_volume_rows": same_or_future,
            "volume_availability_cutoff_violations": (
                volume_availability_violations
            ),
            "incomplete_source_coverage_rows": incomplete_coverage_rows,
            "unexpected_missing_volume_policy_rows": (
                unexpected_missing_policy_rows
            ),
            "maturity_rollback_count": int(rollback_count),
            "unique_date_symbol": unique_date_symbol,
            "daily_map_rows": int(len(active_contract_map)),
            "first_selected_session": str(active_contract_map.iloc[0]["date"]),
            "last_selected_session": str(active_contract_map.iloc[-1]["date"]),
            "roll_events": int(active_contract_map["rolled"].astype(bool).sum()),
            "bound_file_hashes_verified": True,
            "source_manifest_chain_verified": True,
            "full_screen_recomputed_from_bound_inputs": True,
            "causal_session_flattening_recomputed": True,
            "causal_volume_availability_recomputed": True,
            "complete_candidate_coverage_recomputed": True,
        },
        "volume_availability": {
            "decision_time_field": "decision_time_utc_ns",
            "cutoff_field": "volume_cutoff_available_time_ns",
            "required_relation": "volume_cutoff_available_time_ns < decision_time_utc_ns",
            "decision_timezone": "America/Chicago",
            "late_or_recovery_rows_are_excluded": True,
            "missing_volume_policy": MISSING_VOLUME_POLICY,
            "active_contract_map_sha256": map_sha256,
        },
        "session_flattening": {
            **_session_flattening_spec(
                load_yaml(config_paths["gc_lite_v1_signals"])
            ),
            "trace_sha256": verified_artifacts["trace"]["sha256"],
            "episodes_sha256": verified_artifacts["episodes"]["sha256"],
        },
        "artifacts": verified_artifacts,
        "configs": verified_configs,
        "source_l1_files": verified_l1,
        "source_daily_manifests": verified_manifests,
        "implementation": {
            "active_contract_v2": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "baseline_pipeline": {
                "path": str(
                    Path(
                        _apply_session_flat_positions.__code__.co_filename
                    ).resolve()
                ),
                "sha256": sha256_file(
                    Path(_apply_session_flat_positions.__code__.co_filename).resolve()
                ),
            },
            "bars": {
                "path": str(Path(generate_gc_lite_trace.__code__.co_filename).resolve()),
                "sha256": sha256_file(
                    Path(generate_gc_lite_trace.__code__.co_filename).resolve()
                ),
            },
        },
        "sealed_2026_read": False,
        "calibration_or_locked_outcomes_read": False,
    }


def verify_active_contract_certificate(
    path: str | Path,
) -> dict[str, Any]:
    """Authenticate a certificate by rebuilding it from every bound input.

    This is the disk-loading trust boundary for downstream consumers.  No
    self-reported boolean, embedded digest, source list, or implementation hash
    is accepted until the bound screen has been rerun and the resulting
    canonical certificate is exactly equal to the loaded JSON object.
    """

    certificate_path = Path(path).expanduser().resolve()
    if not certificate_path.is_file():
        raise FileNotFoundError(
            f"active-contract certificate does not exist: {certificate_path}"
        )
    try:
        payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"active-contract certificate is not valid JSON: {certificate_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("active-contract certificate must be a JSON object")

    artifact_bindings = payload.get("artifacts")
    config_bindings = payload.get("configs")
    source_l1_bindings = payload.get("source_l1_files")
    source_manifest_bindings = payload.get("source_daily_manifests")
    if not isinstance(artifact_bindings, Mapping):
        raise ValueError("certificate artifacts must be a mapping")
    if not isinstance(config_bindings, Mapping):
        raise ValueError("certificate configs must be a mapping")
    if not isinstance(source_l1_bindings, list):
        raise ValueError("certificate source_l1_files must be a list")
    if not isinstance(source_manifest_bindings, list):
        raise ValueError("certificate source_daily_manifests must be a list")
    if "active_contract_map" not in artifact_bindings:
        raise ValueError("certificate does not bind an active-contract map")
    if "past_only_roll_v2" not in config_bindings:
        raise ValueError("certificate does not bind the v2 roll config")

    map_path, _ = _verify_file_binding(
        "certificate active-contract map",
        artifact_bindings["active_contract_map"],
    )
    roll_config_path, _ = _verify_file_binding(
        "certificate v2 roll config",
        config_bindings["past_only_roll_v2"],
    )
    active_contract_map = pd.read_parquet(map_path)
    roll_config = load_yaml(roll_config_path)
    rebuilt = build_active_contract_certificate(
        active_contract_map=active_contract_map,
        roll_config=roll_config,
        artifact_bindings=artifact_bindings,
        config_bindings=config_bindings,
        source_l1_bindings=source_l1_bindings,
        source_manifest_bindings=source_manifest_bindings,
    )
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    canonical_rebuilt = json.dumps(
        rebuilt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical_payload != canonical_rebuilt:
        raise ValueError(
            "active-contract certificate does not equal its canonical causal "
            "recomputation: loaded_sha256="
            f"{sha256(canonical_payload).hexdigest()}, rebuilt_sha256="
            f"{sha256(canonical_rebuilt).hexdigest()}"
        )
    if rebuilt.get("passed") is not True:
        raise ValueError("canonical active-contract certificate did not pass")
    return rebuilt
