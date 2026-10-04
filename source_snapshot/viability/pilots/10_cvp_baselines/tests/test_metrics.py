from __future__ import annotations

import numpy as np
import pytest

from kx_cvp_baselines.metrics import daily_metrics, moving_block_means, paired_effect


def test_daily_metrics_use_session_end_cash_drawdown() -> None:
    result = daily_metrics(
        {"2026-01-01": 10000, "2026-01-02": -15000, "2026-01-03": 5000},
        completed_round_trips=3,
    )
    assert result["net_pnl_cents"] == 0
    assert result["maximum_drawdown_cents"] == 15000
    assert result["completed_round_trips"] == 3


def test_moving_block_is_deterministic_and_supports_43_sessions() -> None:
    values = np.arange(43, dtype=float)
    first = moving_block_means(values)
    second = moving_block_means(values)
    assert first.shape == (10_000,)
    assert np.array_equal(first, second)


def test_paired_effect_is_reference_minus_baseline() -> None:
    days = [f"2026-06-{index:02d}" for index in range(1, 6)]
    result = paired_effect(
        dict(zip(days, [200] * 5, strict=True)),
        dict(zip(days, [100] * 5, strict=True)),
    )
    assert result["total_difference_cents"] == 500
    assert result["mean_daily_difference_usd"] == pytest.approx(1.0)
    assert result["one_sided_90pct_lower_usd_per_session"] == pytest.approx(1.0)


def test_bootstrap_rejects_short_or_nonfinite_input() -> None:
    with pytest.raises(ValueError):
        moving_block_means([1.0, 2.0])
    with pytest.raises(ValueError):
        moving_block_means([1.0, 2.0, 3.0, 4.0, float("nan")])
