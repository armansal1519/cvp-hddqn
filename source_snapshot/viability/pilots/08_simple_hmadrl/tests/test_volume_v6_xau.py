from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from kx_simple_hmadrl.downloader import sha256_file
from kx_simple_hmadrl.volume_v6 import RegimeQNetwork, TraderQNetwork, V6ModelBundle
from kx_simple_hmadrl.volume_v6_training import V6DDQNConfig, seed_everything
from kx_simple_hmadrl.volume_v6_xau import (
    EXPECTED_PRETRAIN_MONTHS,
    XAUPriceScaler,
    XAUPretrainingError,
    XAUPretrainingInterrupted,
    build_xau_price_sessions,
    load_direct_xau_product,
    load_xau_pretrained_bundle,
    neutral_regime_observation,
    neutral_trader_observation,
    pretrain_xau_price_encoders,
    regime_price_observation,
    trader_price_observation,
)


CHICAGO = "America/Chicago"


def _one_session(day: str, *, drift: float = 0.02) -> pd.DataFrame:
    event_time = pd.date_range(
        f"{day} 07:20", periods=305, freq="1min", tz=CHICAGO,
    ).tz_convert("UTC")
    index = np.arange(len(event_time), dtype=np.float64)
    base = 1_600.0 + drift * index + 0.08 * np.sin(index / 7.0)
    close = base + 0.015 * np.cos(index / 5.0)
    return pd.DataFrame({
        "event_time": event_time,
        "open": base,
        "high": np.maximum(base, close) + 0.04,
        "low": np.minimum(base, close) - 0.04,
        "close": close,
        "quote_count": (2 + (index.astype(np.int64) % 5)).astype(np.int64),
        "bid_quote_volume": np.ones(len(index), dtype=np.float32),
        "ask_quote_volume": np.ones(len(index), dtype=np.float32),
    })


def _direct_product(root: Path, months: dict[str, list[pd.DataFrame]]) -> Path:
    records: list[dict[str, object]] = []
    declared_counts = {
        month: 16 + int(index < 4)
        for index, month in enumerate(EXPECTED_PRETRAIN_MONTHS)
    }
    empty = _one_session("2012-01-03").iloc[:0]
    empty_one = root / "monthly" / "1min" / "empty.parquet"
    empty_five = root / "monthly" / "5min" / "empty.parquet"
    empty_source = root / "bi5" / "empty.bi5"
    empty_one.parent.mkdir(parents=True, exist_ok=True)
    empty_five.parent.mkdir(parents=True, exist_ok=True)
    empty_source.parent.mkdir(parents=True, exist_ok=True)
    empty.to_parquet(empty_one, index=False)
    empty.to_parquet(empty_five, index=False)
    empty_source.write_bytes(b"synthetic-empty-month")
    for month in EXPECTED_PRETRAIN_MONTHS:
        frames = months.get(month, [])
        one = pd.concat(frames, ignore_index=True) if frames else empty
        one_path = (
            root / "monthly" / "1min" / f"XAUUSD_{month}.parquet"
            if frames else empty_one
        )
        five_path = (
            root / "monthly" / "5min" / f"XAUUSD_{month}.parquet"
            if frames else empty_five
        )
        source_path = root / "bi5" / month / "sample.bi5" if frames else empty_source
        one_path.parent.mkdir(parents=True, exist_ok=True)
        five_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            one.to_parquet(one_path, index=False)
            one.iloc[::5].to_parquet(five_path, index=False)
            source_path.write_bytes(f"synthetic-{month}".encode())
        records.append({
            "schema": "kx_dukascopy_xauusd_monthly_v1",
            "month": month,
            "one_minute": str(one_path.relative_to(root)),
            "one_minute_sha256": sha256_file(one_path),
            "five_minute": str(five_path.relative_to(root)),
            "five_minute_sha256": sha256_file(five_path),
            "sources": [{
                "path": str(source_path.relative_to(root)),
                "sha256": sha256_file(source_path),
                "bytes": source_path.stat().st_size,
            }],
            "accepted_sessions": declared_counts[month],
        })
    manifest = {
        "schema": "kx_dukascopy_xauusd_monthly_v1",
        "instrument": "XAUUSD",
        "source_kind": "Dukascopy OTC tick-by-tick quotes",
        "start": "2012-01-01",
        "end": "2025-09-01",
        "locked_period_downloaded": False,
        "price_scale": 1_000.0,
        "pretraining_sessions": sum(declared_counts.values()),
        "volume_semantics": {
            "quote_count": "proxy channel; not exchange-traded volume",
            "bid_quote_volume": "vendor quote volume",
            "ask_quote_volume": "vendor quote volume",
        },
        "months": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (root / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n"
    )
    return root


def _batch(observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value[None], dtype=torch.bool if name.endswith("mask") else torch.float32)
        for name, value in observation.items()
    }


def test_direct_product_must_be_direct_hash_valid_and_disclose_proxy(tmp_path: Path) -> None:
    root = _direct_product(tmp_path / "direct", {"2012-01": [_one_session("2012-01-03")]})
    product = load_direct_xau_product(root)
    assert product.manifest["source_kind"] == "Dukascopy OTC tick-by-tick quotes"
    assert tuple(month["month"] for month in product.pretraining_months) == EXPECTED_PRETRAIN_MONTHS
    source = root / "bi5" / "2012-01" / "sample.bi5"
    source.write_bytes(b"tampered")
    with pytest.raises(XAUPretrainingError, match="hash-valid direct"):
        load_direct_xau_product(root)


def test_direct_product_requires_every_2012_2024_month(tmp_path: Path) -> None:
    root = _direct_product(tmp_path / "direct", {"2012-01": [_one_session("2012-01-03")]})
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["months"] = manifest["months"][:-1]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (root / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n"
    )
    with pytest.raises(XAUPretrainingError, match="every month"):
        load_direct_xau_product(root)


def test_registered_xau_requires_nonempty_support_in_every_month(tmp_path: Path) -> None:
    root = _direct_product(tmp_path / "direct", {
        "2012-01": [_one_session("2012-01-03")],
    })
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    removed = int(manifest["months"][1]["accepted_sessions"])
    manifest["months"][1]["accepted_sessions"] = 0
    manifest["pretraining_sessions"] -= removed
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    (root / "manifest.sha256").write_text(
        f"{sha256_file(manifest_path)}  manifest.json\n"
    )
    with pytest.raises(XAUPretrainingError, match="every 2012-2024 XAU month"):
        load_direct_xau_product(root)

    # A declared label is not enough for registered transfer: the actual
    # monthly products must each materialize at least one accepted session.
    root = _direct_product(tmp_path / "sparse", {
        "2012-01": [_one_session("2012-01-03")],
    })
    with pytest.raises(XAUPretrainingError, match="nonempty accepted support"):
        _pretrain(root, tmp_path / "registered", maximum_sessions=None, minimum_sessions=1)


def test_price_features_are_completed_causal_and_quote_count_is_not_an_input() -> None:
    frame = _one_session("2012-01-03")
    sessions = build_xau_price_sessions(frame)
    assert len(sessions) == 1
    session = sessions[0]
    assert session.top_base.shape == (20, 8)
    assert session.bottom_base.shape == (61, 17)
    assert session.top_valid.all() and session.bottom_valid.all()
    assert session.observed_minutes == 305 and session.quote_count_proxy_total > 0

    future = frame.copy()
    future.loc[future.index >= 120, ["open", "high", "low", "close"]] += 500.0
    changed = build_xau_price_sessions(future)[0]
    # The fifth completed 15-minute bar ends before minute 75.
    assert np.array_equal(session.top_base[:5], changed.top_base[:5])
    # Quote-count proxy mutations cannot alter any price/base feature.
    proxy = frame.copy()
    proxy["quote_count"] *= 10_000
    proxy_session = build_xau_price_sessions(proxy)[0]
    assert np.array_equal(session.top_base, proxy_session.top_base)
    assert np.array_equal(session.bottom_base, proxy_session.bottom_base)


def test_neutral_expansion_matches_both_v6_network_contracts() -> None:
    session = build_xau_price_sessions(_one_session("2012-01-03"))[0]
    scaler = XAUPriceScaler.fit((session,))
    top_price = regime_price_observation(session, 7, scaler)
    bottom_price = trader_price_observation(session, 23, scaler)
    top = neutral_regime_observation(top_price)
    bottom = neutral_trader_observation(bottom_price)
    assert set(top_price) == set(bottom_price) == {"base", "sequence_mask"}
    assert np.count_nonzero(top["rich"]) == 0
    assert not top["context_mask"].any()
    assert np.count_nonzero(bottom["rich"]) == 0
    assert np.count_nonzero(bottom["levels"]) == 0
    assert np.count_nonzero(bottom["l2_engineered"]) == 0
    assert not bottom["l2_mask"].any()
    assert np.count_nonzero(bottom["message"]) == 0
    assert np.count_nonzero(bottom["state"]) == 0
    assert RegimeQNetwork()(_batch(top)).shape == (1, 3)
    assert TraderQNetwork()(_batch(bottom)).shape == (1, 3)


def _tiny_config() -> V6DDQNConfig:
    return V6DDQNConfig(
        learning_rate=2.5e-4,
        batch_size=8,
        gamma=1.0,
        gradient_clip_norm=10.0,
        target_sync_updates=5,
        epsilon_start=1.0,
        epsilon_end=0.05,
    )


def _pretrain(root: Path, output: Path, **extra: object) -> Path:
    arguments: dict[str, object] = {
        "seed": 1103,
        "passes": 1,
        "minimum_sessions": 2,
        "maximum_sessions": 2,
        "config": _tiny_config(),
        "regime_replay_capacity": 64,
        "trader_replay_capacity": 128,
        "regime_replay_warmup": 8,
        "trader_replay_warmup": 8,
    }
    arguments.update(extra)
    return pretrain_xau_price_encoders(root, output, **arguments)  # type: ignore[arg-type]


def test_pretraining_changes_only_base_encoders_and_binds_metadata(tmp_path: Path) -> None:
    root = _direct_product(tmp_path / "direct", {
        "2012-01": [_one_session("2012-01-03"), _one_session("2012-01-04", drift=-0.01)],
    })
    seed_everything(1103)
    initial = V6ModelBundle.create()
    manifest_path = _pretrain(root, tmp_path / "trained")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["stage"] == "xau_price_pretraining"
    assert manifest["seed"] == 1103
    assert manifest["gc_outcomes_accessed"] is False
    assert manifest["gc_transfer_eligible"] is False
    assert "proxy" in manifest["quote_count"]
    trained, metadata = load_xau_pretrained_bundle(
        tmp_path / "trained", require_gc_transfer_eligible=False,
    )
    assert metadata["stage"] == "xau_price_pretraining"

    def different(first: torch.nn.Module, second: torch.nn.Module) -> bool:
        return any(
            not torch.equal(left, right)
            for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True)
        )

    def equal(first: torch.nn.Module, second: torch.nn.Module) -> bool:
        return all(
            torch.equal(left, right)
            for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True)
        )

    assert different(initial.regime.base, trained.regime.base)
    assert different(initial.trader.base, trained.trader.base)
    assert equal(initial.regime.rich, trained.regime.rich)
    assert equal(initial.trader.rich, trained.trader.rich)
    assert equal(initial.trader.l2, trained.trader.l2)
    assert equal(initial.regime.head, trained.regime.head)
    assert equal(initial.trader.head, trained.trader.head)
    with pytest.raises(XAUPretrainingError, match="not eligible"):
        load_xau_pretrained_bundle(tmp_path / "trained")


def test_monthly_checkpoint_resumes_to_identical_weights(tmp_path: Path) -> None:
    root = _direct_product(tmp_path / "direct", {
        "2012-01": [_one_session("2012-01-03")],
        "2012-02": [_one_session("2012-02-02", drift=-0.015)],
    })
    resumed = tmp_path / "resumed"
    with pytest.raises(XAUPretrainingInterrupted):
        _pretrain(root, resumed, stop_after_month_checkpoints_for_test=1)
    assert not resumed.exists()
    assert (tmp_path / ".resumed.part" / "resume.pt").is_file()
    _pretrain(root, resumed)
    _pretrain(root, tmp_path / "uninterrupted")
    first, _ = load_xau_pretrained_bundle(resumed, require_gc_transfer_eligible=False)
    second, _ = load_xau_pretrained_bundle(
        tmp_path / "uninterrupted", require_gc_transfer_eligible=False,
    )
    for branch in ("regime", "trader"):
        left = getattr(first, branch).state_dict()
        right = getattr(second, branch).state_dict()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[name], right[name]) for name in left)


def test_existing_xau_output_is_bound_to_request_source_and_resume_hash(tmp_path: Path) -> None:
    root = _direct_product(tmp_path / "direct", {
        "2012-01": [_one_session("2012-01-03"), _one_session("2012-01-04")],
    })
    output = tmp_path / "trained"
    _pretrain(root, output)
    # An immutable destination cannot be silently reused for another seed or
    # another minimum-support request.
    with pytest.raises(XAUPretrainingError, match="another request"):
        _pretrain(root, output, seed=2207)
    with pytest.raises(XAUPretrainingError, match="another request"):
        _pretrain(root, output, minimum_sessions=1)

    manifest = json.loads((output / "manifest.json").read_text())
    resume = output / manifest["resume_checkpoint"]["path"]
    resume.write_bytes(resume.read_bytes() + b"tamper")
    with pytest.raises(XAUPretrainingError, match="resume checkpoint"):
        load_xau_pretrained_bundle(output, require_gc_transfer_eligible=False)
