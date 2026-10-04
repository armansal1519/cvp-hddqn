from __future__ import annotations

import json
from pathlib import Path

import pytest

from kx_simple_hmadrl.volume_v6_pipeline import (
    aligned_pairs,
    evaluate_ensemble,
    train_seed,
)


def test_pair_selection_refuses_a_locked_training_date() -> None:
    with pytest.raises(PermissionError):
        aligned_pairs(
            [Path("2026-06-01.npz")], [Path("2026-06-01.npz")], phase="train",
        )


def test_one_known_session_trains_and_replays_end_to_end(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    raw = root / "artifacts/development/simple_hmadrl_gc_jan_jun_env_v2/sessions/2025-01-09.npz"
    side = root / "artifacts/development/simple_hmadrl_volume_v4_sidecars/sessions/2025-01-09.npz"
    if not raw.is_file() or not side.is_file():
        pytest.skip("the optional contaminated GC integration fixture is absent")
    output = tmp_path / "seed_1103"
    manifest_path = train_seed(
        [raw], [side], output, seed=1103, arm="rich_volume",
        phase="smoke", maximum_sessions=1, passes=1,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["same_seed_message_pairing"] is True
    assert manifest["upper_frozen_before_trader"] is True
    assert manifest["regime_pass_reports"]
    for report in manifest["regime_pass_reports"]:
        assert {
            "counterfactual_labels_attempted",
            "counterfactual_labels_kept",
            "counterfactual_labels_dropped",
        } <= report.keys()
        assert report["counterfactual_labels_attempted"] == (
            report["counterfactual_labels_kept"]
            + report["counterfactual_labels_dropped"]
        )
    assert manifest["regime"]["parameters"] > 0
    assert manifest["trader"]["parameters"] > 0
    report = evaluate_ensemble(
        [output], [raw], [side], arm="rich_volume", phase="smoke",
        require_registered_seeds=False,
    )
    assert report["evaluation_dates"] == ["2025-01-09"]
    assert report["metrics"]["sessions"] == 1
    assert report["holdout_read"] is False
