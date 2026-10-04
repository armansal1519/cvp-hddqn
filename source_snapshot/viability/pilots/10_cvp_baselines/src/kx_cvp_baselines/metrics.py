"""Exact daily-dollar metrics and descriptive paired uncertainty summaries."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


BOOTSTRAP_BLOCK = 5
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 1103


def daily_metrics(
    daily_pnl_cents: Mapping[str, int], *, completed_round_trips: int,
) -> dict[str, float | int | None]:
    """Recompute the frozen V6 economic summaries from integer daily cash."""

    days = sorted(str(day) for day in daily_pnl_cents)
    if not days or days != sorted(set(days)):
        raise ValueError("daily P&L must contain unique dated observations")
    cents = np.asarray([int(daily_pnl_cents[day]) for day in days], dtype=np.float64)
    usd = cents / 100.0
    standard = float(usd.std(ddof=1)) if len(usd) > 1 else 0.0
    sharpe = 0.0 if standard < 1e-12 else float(math.sqrt(252.0) * usd.mean() / standard)
    gains = float(usd[usd > 0].sum())
    losses = float(-usd[usd < 0].sum())
    equity = np.concatenate(([0.0], np.cumsum(usd)))
    drawdown = np.maximum.accumulate(equity) - equity
    return {
        "sessions": int(len(usd)),
        "net_pnl_cents": int(cents.sum()),
        "net_pnl_usd": float(usd.sum()),
        "mean_daily_pnl_cents": float(cents.mean()),
        "mean_daily_pnl_usd": float(usd.mean()),
        "annualized_daily_sharpe": sharpe,
        "profit_factor": None if losses == 0.0 else gains / losses,
        "daily_profit_factor": None if losses == 0.0 else gains / losses,
        "positive_day_fraction": float(np.mean(usd > 0.0)),
        "maximum_drawdown_cents": int(round(float(drawdown.max(initial=0.0)) * 100.0)),
        "maximum_drawdown_usd": float(drawdown.max(initial=0.0)),
        "completed_round_trips": int(completed_round_trips),
    }


def moving_block_means(
    differences_usd: Sequence[float] | np.ndarray,
    *, block: int = BOOTSTRAP_BLOCK,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Overlapping non-circular moving-block means in chronological order."""

    values = np.asarray(differences_usd, dtype=np.float64)
    if values.ndim != 1 or len(values) < int(block):
        raise ValueError("bootstrap needs one finite vector and at least one block")
    if not np.isfinite(values).all() or block < 1 or repetitions < 1:
        raise ValueError("bootstrap inputs must be finite and positive")
    blocks = np.asarray(
        [values[index:index + block] for index in range(len(values) - block + 1)],
        dtype=np.float64,
    )
    draws = math.ceil(len(values) / block)
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(blocks), size=(int(repetitions), draws))
    return blocks[selected].reshape(int(repetitions), -1)[:, :len(values)].mean(axis=1)


def paired_effect(
    reference_daily_cents: Mapping[str, int],
    baseline_daily_cents: Mapping[str, int],
) -> dict[str, float | int]:
    """Return the registered CVP-minus-baseline descriptive paired summary."""

    days = sorted(str(day) for day in reference_daily_cents)
    if days != sorted(str(day) for day in baseline_daily_cents) or not days:
        raise ValueError("paired policies must contain the same nonempty date set")
    difference_cents = np.asarray(
        [int(reference_daily_cents[day]) - int(baseline_daily_cents[day]) for day in days],
        dtype=np.int64,
    )
    difference_usd = difference_cents.astype(np.float64) / 100.0
    distribution = moving_block_means(difference_usd)
    return {
        "sessions": len(days),
        "total_difference_cents": int(difference_cents.sum()),
        "total_difference_usd": float(difference_usd.sum()),
        "mean_daily_difference_usd": float(difference_usd.mean()),
        "one_sided_90pct_lower_usd_per_session": float(
            np.quantile(distribution, 0.10)
        ),
        "block_length_sessions": BOOTSTRAP_BLOCK,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


__all__ = [
    "BOOTSTRAP_BLOCK", "BOOTSTRAP_REPETITIONS", "BOOTSTRAP_SEED",
    "daily_metrics", "moving_block_means", "paired_effect",
]
