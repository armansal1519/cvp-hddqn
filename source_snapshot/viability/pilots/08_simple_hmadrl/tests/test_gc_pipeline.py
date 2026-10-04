from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
import torch

from kx_simple_hmadrl.checkpoint import load_checkpoint, save_checkpoint
from kx_simple_hmadrl.basepaper_pretrain import _load_direct_bars
from kx_simple_hmadrl.datasets import SessionShard
from kx_simple_hmadrl.gate import verify_unlock
from kx_simple_hmadrl.gc_data import DECODE_NS, reconstruct_arrival_bbo, reconstruct_l2
from kx_simple_hmadrl.networks import MonolithicPPO
from kx_simple_hmadrl.runner import matched_monolithic_hidden
from kx_simple_hmadrl.torch_ppo import RolloutBatch, masked_distribution, ppo_update


def _book_rows(event: int, recv: int, *, shift: int = 0) -> list[dict[str, object]]:
    result = []
    row = 0
    for side, prices in (("B", range(2000 + shift, 1990 + shift, -1)), ("A", range(2001 + shift, 2011 + shift))):
        for price in prices:
            result.append({"event_index": event, "event_row_index": row, "ts_recv_ns": recv,
                           "continuity_segment_id": 1, "operation": "upsert", "side": side,
                           "price_ticks": price, "level_size": 10 + row})
            row += 1
    return result


def test_l2_available_time_and_prefix_invariance() -> None:
    rows = pd.DataFrame(_book_rows(1, 1_000_000))
    clocks = np.array([1_500_000, 2_000_000, 3_000_000], dtype=np.int64)
    first, _ = reconstruct_l2(rows, clocks)
    assert not first["valid"][0]  # complete event is unavailable until recv + 1 ms
    assert first["valid"][1:].all()
    future = pd.concat((rows, pd.DataFrame(_book_rows(2, 4_000_000, shift=1))), ignore_index=True)
    second, _ = reconstruct_l2(future, clocks)
    np.testing.assert_array_equal(first["valid"], second["valid"])
    np.testing.assert_allclose(first["levels"], second["levels"])
    assert first["best_bid_ticks"][1] == 2000
    assert first["best_ask_ticks"][1] == 2001
    assert first["best_bid_size"][1] > 0 and first["best_ask_size"][1] > 0
    assert DECODE_NS == 1_000_000


def test_arrival_snapshot_uses_exact_shifted_clock_and_is_prefix_invariant() -> None:
    rows = pd.DataFrame(_book_rows(1, 1_000_000))
    clocks = np.array([2_000_000, 3_000_000], dtype=np.int64)
    first = reconstruct_arrival_bbo(rows, clocks, 5)
    assert first["arrival_5ms_clock_ns"].tolist() == [7_000_000, 8_000_000]
    assert first["arrival_5ms_state_available_ns"].tolist() == [2_000_000, 2_000_000]
    assert first["arrival_5ms_continuity_segment"].tolist() == [1, 1]
    future = pd.concat((rows, pd.DataFrame(_book_rows(2, 9_000_000, shift=1))), ignore_index=True)
    second = reconstruct_arrival_bbo(future, clocks, 5)
    np.testing.assert_array_equal(first["arrival_5ms_bid_ticks"], second["arrival_5ms_bid_ticks"])


def test_session_windows_pad_but_never_cross(tmp_path: Path) -> None:
    shard = tmp_path / "2025-01-09.npz"
    trade_features = np.ones((2, 7)); trade_features[:, -1] = 0
    np.savez(shard, clock_ns=np.arange(3), levels=np.ones((3, 10, 4)), engineered=np.ones((3, 4)),
             valid=np.ones(3, bool), trade_available_ns=np.arange(2), trade_features=trade_features)
    with SessionShard(shard) as data:
        levels, _, valid = data.l2_window(2)
        trades, mask = data.trade_window(1)
    assert levels.shape == (40, 10, 4) and valid.sum() == 2
    assert trades.shape == (256, 7) and mask.sum() == 1
    assert np.all(trades[:-1, -1] == 1)


def test_masked_ppo_updates_only_legal_actions() -> None:
    torch.manual_seed(7)
    model = MonolithicPPO(3, 8, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    observations = torch.randn(8, 3)
    with torch.no_grad():
        logits, values = model(observations)
        mask = torch.tensor([[True, False]] * 8)
        distribution = masked_distribution(logits, mask)
        actions = distribution.sample()
        logs = distribution.log_prob(actions)
    before = next(model.parameters()).detach().clone()
    batch = RolloutBatch((observations,), actions, logs, torch.linspace(-1, 1, 8),
                         values.squeeze(-1) + 0.5, mask)
    ppo_update(model, optimizer, batch)
    assert actions.eq(0).all()
    assert not torch.equal(before, next(model.parameters()).detach())


def test_checkpoint_restores_rng_and_parameter_match(tmp_path: Path) -> None:
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    model = MonolithicPPO(4, 8, 2)
    optimizer = torch.optim.Adam(model.parameters())
    path = save_checkpoint(tmp_path / "model.pt", model, optimizer, step=3, metadata={"trial": "x"})
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    state = load_checkpoint(path, model, optimizer)
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert state["step"] == 3 and actual[0] == expected[0] and actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
    target = sum(p.numel() for p in MonolithicPPO(12, 64, 2).parameters())
    _, difference = matched_monolithic_hidden(12, 2, target)
    assert difference <= 0.01


def test_signed_unlock_rejects_tampering(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_bytes(b"test-secret")
    body = {"purpose": "one_shot_locked_gc_test", "split": "2025-11-01/2026-01-01", "artifacts": {}}
    signature = hmac.new(key.read_bytes(), json.dumps(body, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
    unlock = tmp_path / "unlock.json"
    unlock.write_text(json.dumps({**body, "signature_hmac_sha256": signature}))
    assert verify_unlock(unlock, key)["purpose"] == "one_shot_locked_gc_test"
    unlock.write_text(json.dumps({**body, "purpose": "diagnostic", "signature_hmac_sha256": signature}))
    with pytest.raises(PermissionError):
        verify_unlock(unlock, key)


def test_basepaper_pretraining_rejects_ohlc_mirror(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({
        "source_kind": "monthly Git mirror", "months": [],
        "volume_semantics": {"quote_count": "unavailable"},
    }))
    with pytest.raises(ValueError, match="direct BI5"):
        _load_direct_bars(tmp_path)
