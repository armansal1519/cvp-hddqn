from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kx_simple_hmadrl.volume_v6_contract import (
    BASE_CONTEXT_WIDTH,
    BOTTOM_BASE_FIELDS,
    BOTTOM_HISTORY,
    BOTTOM_RICH_FIELDS,
    CONTEXT_FIELDS,
    L2_ENGINEERED_FIELDS,
    L2_HISTORY,
    L2_LEVEL_FIELDS,
    RegimeMessage,
    STATE_FIELDS,
    TOP_BASE_FIELDS,
    TOP_HISTORY,
    TOP_RICH_FIELDS,
    assert_phase_date,
)
from kx_simple_hmadrl.volume_v6_data import (
    V6Calibration,
    V6FeatureScaler,
    bottom_observation_at,
    build_session_features,
    encode_agent_state,
    top_observation_at,
)


NS = 1_000_000_000


def _session(
    day: str = "2025-01-09", *, segment_break_minute: int | None = None,
    unknown_signs: bool = False, volume_multiplier: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Sixty-five minutes yields four 15-minute and twelve 5-minute decisions;
    # the last five minutes remain reserved for terminal liquidation.
    start = 1_700_000_000 * NS
    book_count = 65 * 60 * 4 + 1
    clock = start + np.arange(book_count, dtype=np.int64) * 250_000_000
    seconds = np.arange(book_count, dtype=np.int64) // 4
    bid = 20_000 + seconds // 60
    ask = bid + 1
    segment = np.ones(book_count, dtype=np.uint32)
    if segment_break_minute is not None:
        segment[clock >= start + segment_break_minute * 60 * NS] = 2
    levels = np.zeros((book_count, 10, 4), np.float32)
    for depth in range(10):
        levels[:, depth, 0] = depth + 0.5
        levels[:, depth, 1] = np.log1p(10 + depth + seconds % 3)
        levels[:, depth, 2] = depth + 0.5
        levels[:, depth, 3] = np.log1p(12 + depth + seconds % 5)
    raw: dict[str, np.ndarray] = {
        "clock_ns": clock,
        "state_available_ns": clock - 1_000_000,
        "continuity_segment": segment,
        "valid": np.ones(book_count, bool),
        "levels": levels,
        "engineered": np.zeros((book_count, 4), np.float32),
        "best_bid_ticks": bid,
        "best_ask_ticks": ask,
        "best_bid_size": np.full(book_count, 10, np.uint64),
        "best_ask_size": np.full(book_count, 12, np.uint64),
        "session_date": np.asarray(day),
        "raw_symbol": np.asarray("GCG5"),
        # Deliberately absurd simulator-only fields: observations must ignore them.
        "arrival_5ms_bid_ticks": np.full(book_count, -999_999, np.int64),
        "arrival_5ms_ask_ticks": np.full(book_count, 999_999, np.int64),
    }
    trade_count = 65 * 60
    trade_time = start + np.arange(trade_count, dtype=np.int64) * NS + 100_000_000
    book_index = np.searchsorted(clock, trade_time, side="right") - 1
    trade_price = bid[book_index] + (np.arange(trade_count) % 2)
    size = volume_multiplier * (1.0 + np.arange(trade_count) % 10)
    signs = np.zeros(trade_count, np.float32) if unknown_signs else np.asarray(
        (-1, 0, 1), np.float32,
    )[np.arange(trade_count) % 3]
    mlofi = np.sin(np.arange(book_count)[:, None] / 17 + np.arange(10)[None]).astype(np.float32)
    side: dict[str, np.ndarray] = {
        "session_date": np.asarray(day),
        "session_start_ns": np.asarray(clock[0], np.int64),
        "session_end_ns": np.asarray(clock[-1], np.int64),
        "trade_available_ns": trade_time,
        "trade_price_ticks": trade_price.astype(np.int64),
        "trade_size": size.astype(np.float32),
        "trade_sign": signs,
        "mlofi": mlofi,
        "source_sha256": np.asarray("synthetic"),
    }
    return raw, side


def _fit(raw: dict[str, np.ndarray], side: dict[str, np.ndarray]) -> V6FeatureScaler:
    return V6FeatureScaler.fit(((raw, side),))


def _state() -> np.ndarray:
    return encode_agent_state(
        inventory=1,
        current_target=1,
        liquidation_pnl_ticks=2.5,
        signed_entry_distance_ticks=3.0,
        holding_age_hours=0.25,
        session_remaining=0.5,
        visible_spread_ticks=1.0,
        round_trip_fee_ticks=0.6,
        previous_command_rejected=False,
    )


def test_registered_feature_order_and_observation_shapes_are_exact() -> None:
    assert tuple(map(len, (
        TOP_BASE_FIELDS, TOP_RICH_FIELDS, CONTEXT_FIELDS, BOTTOM_BASE_FIELDS,
        BOTTOM_RICH_FIELDS, L2_LEVEL_FIELDS, L2_ENGINEERED_FIELDS, STATE_FIELDS,
    ))) == (8, 8, 22, 17, 21, 5, 16, 13)
    raw, side = _session()
    scaler = _fit(raw, side)
    session = build_session_features(raw, side, scaler.calibration)
    top = top_observation_at(session, 3, scaler).arrays("rich_volume")
    assert top["base"].shape == (TOP_HISTORY, 8)
    assert top["rich"].shape == (TOP_HISTORY, 8)
    assert top["sequence_mask"].shape == (TOP_HISTORY,)
    assert top["context"].shape == top["context_mask"].shape == (22,)
    message = RegimeMessage.from_q_values(
        (-2.0, 0.0, 4.0), int(session.bottom_clock_ns[-1]),
        int(session.bottom_clock_ns[-1]) + 15 * 60 * NS,
    )
    bottom = bottom_observation_at(session, 11, scaler, message, _state()).arrays()
    assert bottom["base"].shape == (BOTTOM_HISTORY, 17)
    assert bottom["rich"].shape == (BOTTOM_HISTORY, 21)
    assert bottom["levels"].shape == (L2_HISTORY, 10, 5)
    assert bottom["l2_engineered"].shape == (L2_HISTORY, 16)
    assert bottom["message"].shape == (5,) and bottom["state"].shape == (13,)
    assert not any(name.startswith("arrival_") for name in top | bottom)
    assert scaler.canonical_sha256() == V6FeatureScaler.from_dict(
        scaler.to_dict(),
    ).canonical_sha256()


def test_future_and_arrival_perturbations_cannot_change_earlier_observation() -> None:
    raw, side = _session()
    scaler = _fit(raw, side)
    original = build_session_features(raw, side, scaler.calibration)
    changed_raw = {name: value.copy() for name, value in raw.items()}
    changed_side = {name: value.copy() for name, value in side.items()}
    decision = int(original.bottom_clock_ns[5])
    future_trades = changed_side["trade_available_ns"] >= decision
    changed_side["trade_price_ticks"][future_trades] += 50_000
    changed_side["trade_size"][future_trades] *= 10_000
    changed_side["trade_sign"][future_trades] *= -1
    future_books = changed_raw["clock_ns"] > decision
    changed_raw["levels"][future_books, :, 0] += 100
    changed_raw["levels"][future_books, :, 2] += 100
    changed_raw["levels"][future_books, :, 1] += 5
    changed_raw["levels"][future_books, :, 3] += 5
    changed_raw["best_bid_ticks"][future_books] += 90_000
    changed_raw["best_ask_ticks"][future_books] += 90_000
    changed_raw["arrival_5ms_bid_ticks"][:] = 123_456_789
    changed_raw["arrival_5ms_ask_ticks"][:] = -123_456_789
    changed = build_session_features(changed_raw, changed_side, scaler.calibration)
    publication = int(original.bottom_clock_ns[5])
    message = RegimeMessage((0.0, 0.0, 0.0), publication, publication + 15 * 60 * NS)
    first = bottom_observation_at(original, 5, scaler, message, _state()).arrays()
    second = bottom_observation_at(changed, 5, scaler, message, _state()).arrays()
    for name in first:
        assert np.array_equal(first[name], second[name]), name


def test_continuity_break_and_invalid_gap_reset_both_windows() -> None:
    raw, side = _session(segment_break_minute=35)
    scaler = _fit(raw, side)
    session = build_session_features(raw, side, scaler.calibration)
    # The 30--45m top bar crosses the segment boundary and is rejected.  The
    # 45--60m observation may therefore contain only its own segment/bar.
    assert session.top_valid.tolist() == [True, True, False, True]
    top = top_observation_at(session, 3, scaler)
    assert int(top.sequence_mask.sum()) == 1
    # Bottom's 30--35m bar crosses the same boundary; five later bars survive.
    bottom_message = RegimeMessage(
        (0.0, 0.0, 0.0), int(session.bottom_clock_ns[-1]),
        int(session.bottom_clock_ns[-1]) + NS,
    )
    bottom = bottom_observation_at(session, 11, scaler, bottom_message, _state())
    assert int(bottom.sequence_mask.sum()) == 5

    gap_raw, gap_side = _session()
    gap_time = int(gap_side["session_start_ns"]) + 55 * 60 * NS
    gap_index = int(np.searchsorted(gap_raw["clock_ns"], gap_time, side="left"))
    gap_raw["valid"][gap_index] = False
    gap_scaler = _fit(gap_raw, gap_side)
    gap_session = build_session_features(gap_raw, gap_side, gap_scaler.calibration)
    assert not gap_session.bottom_valid[10]
    gap_message = RegimeMessage(
        (0.0, 0.0, 0.0), int(gap_session.bottom_clock_ns[-1]),
        int(gap_session.bottom_clock_ns[-1]) + NS,
    )
    gap_observation = bottom_observation_at(
        gap_session, 11, gap_scaler, gap_message, _state(),
    )
    assert int(gap_observation.sequence_mask.sum()) == 1


def test_unknown_aggressors_are_masked_and_control_keeps_only_registered_ohlcv() -> None:
    raw, side = _session(unknown_signs=True)
    scaler = _fit(raw, side)
    session = build_session_features(raw, side, scaler.calibration)
    # Rich field two is imbalance, field three is known-aggressor fraction.
    assert np.allclose(session.top_rich[:, 2:4], 0.0)
    assert not session.top_context_mask[:, BASE_CONTEXT_WIDTH + 1].any()
    decision = int(session.bottom_clock_ns[-1])
    message = RegimeMessage((0.0, 0.0, 0.0), decision, decision + NS)
    observation = bottom_observation_at(session, 11, scaler, message, _state())
    rich = observation.arrays("rich_volume")
    control = observation.arrays("base_ohlcv")
    assert np.array_equal(control["rich"][:, 0], rich["rich"][:, 0])
    assert np.count_nonzero(control["rich"][:, 1:]) == 0
    assert np.count_nonzero(control["context"][BASE_CONTEXT_WIDTH:]) == 0
    assert not control["context_mask"][BASE_CONTEXT_WIDTH:].any()
    assert np.array_equal(control["levels"][:, :, (0, 2)], rich["levels"][:, :, (0, 2)])
    assert np.count_nonzero(control["levels"][:, :, (1, 3, 4)]) == 0
    assert np.array_equal(control["l2_engineered"][:, 0], rich["l2_engineered"][:, 0])
    assert np.count_nonzero(control["l2_engineered"][:, 1:]) == 0


def test_calibration_is_training_only_clock_specific_and_large_trade_p90() -> None:
    first = _session("2025-01-09", volume_multiplier=1.0)
    second = _session("2025-01-10", volume_multiplier=2.0)
    calibration = V6Calibration.fit((first, second))
    assert calibration.training_dates == ("2025-01-09", "2025-01-10")
    sizes = np.concatenate((first[1]["trade_size"], second[1]["trade_size"]))
    assert calibration.large_trade_threshold == pytest.approx(np.quantile(sizes, 0.90))
    # Each five-minute slot has identical construction, so every fitted slot is
    # the median of the two sessions rather than a statistic from later data.
    expected = 1.5 * float(first[1]["trade_size"][:300].sum())
    assert np.allclose(calibration.bottom_relative_volume_curve, expected)
    assert_phase_date("2026-04-30", "train")
    with pytest.raises(PermissionError):
        assert_phase_date("2026-05-01", "train")
    locked_raw, locked_side = _session("2026-05-01")
    with pytest.raises(PermissionError):
        V6Calibration.fit(((locked_raw, locked_side),))


def test_message_expiry_and_inventory_state_are_semantic() -> None:
    message = RegimeMessage.from_q_values((1.0, 2.0, 4.0), 10, 20)
    encoded = message.encode(10)
    assert encoded.shape == (5,) and encoded[3] == 1 and encoded[4] == 1
    assert np.array_equal(message.encode(20), np.zeros(5, np.float32))
    with pytest.raises(ValueError):
        message.encode(9)
    state = _state()
    assert np.array_equal(state[:3], (0.0, 0.0, 1.0))
    assert np.array_equal(state[3:6], (0.0, 0.0, 1.0))
    assert state[6] == 2.5 and state[-1] == 0.0


def test_path_scaler_is_streamed_exact_and_serializable(tmp_path: Path) -> None:
    raw, side = _session()
    raw_path = tmp_path / "raw" / "2025-01-09.npz"
    side_path = tmp_path / "side" / "2025-01-09.npz"
    raw_path.parent.mkdir(); side_path.parent.mkdir()
    np.savez_compressed(raw_path, **raw)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    side["source_sha256"] = np.asarray(digest)
    np.savez_compressed(side_path, **side)
    expected = V6FeatureScaler.fit(((raw, side),))
    streamed = V6FeatureScaler.fit_paths(
        ((raw_path, side_path),), temporary_directory=tmp_path,
    )
    assert streamed.canonical_sha256() == expected.canonical_sha256()
    destination = streamed.save_json(tmp_path / "scaler.json")
    assert V6FeatureScaler.load_json(destination).canonical_sha256() == streamed.canonical_sha256()
    assert not list(tmp_path.glob("kx-v6-scale-*"))
