#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Inspect an untrained model using deterministic synthetic inputs only."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    source = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source))
    import torch
    from kx_simple_hmadrl.volume_v6 import (
        V6ModelBundle, parameter_count, regime_message,
    )

    torch.manual_seed(1103)
    torch.set_num_threads(1)
    models = V6ModelBundle.create().eval()
    publisher = {
        "base": torch.zeros(1, 8, 8),
        "rich": torch.zeros(1, 8, 8),
        "sequence_mask": torch.ones(1, 8, dtype=torch.bool),
        "context": torch.zeros(1, 22),
        "context_mask": torch.ones(1, 22, dtype=torch.bool),
    }
    trader = {
        "base": torch.zeros(1, 24, 17),
        "rich": torch.zeros(1, 24, 21),
        "sequence_mask": torch.ones(1, 24, dtype=torch.bool),
        "context": torch.zeros(1, 22),
        "context_mask": torch.ones(1, 22, dtype=torch.bool),
        "levels": torch.zeros(1, 20, 10, 5),
        "l2_engineered": torch.zeros(1, 20, 16),
        "l2_mask": torch.ones(1, 20, dtype=torch.bool),
        "message": torch.zeros(1, 5),
        "state": torch.zeros(1, 13),
    }
    with torch.no_grad():
        publisher_values = models.regime(publisher)
        trader_values = models.trader(trader)
        message = regime_message(
            torch.tensor([1.0, 2.0, 4.0]), decision_time_ns=10, expiry_time_ns=20,
        )
    assert publisher_values.shape == trader_values.shape == (1, 3)
    assert torch.isfinite(publisher_values).all() and torch.isfinite(trader_values).all()
    print(json.dumps({
        "scope": "Untrained synthetic forward pass; not a trading or performance result",
        "seed": 1103,
        "publisher": {"parameters": parameter_count(models.regime),
                      "output_shape": list(publisher_values.shape)},
        "trader": {"parameters": parameter_count(models.trader),
                   "output_shape": list(trader_values.shape)},
        "message_example": {
            "action_order": ["SHORT", "FLAT", "LONG"],
            "synthetic_q_values": [1.0, 2.0, 4.0],
            "at_publication": message.encode(10).tolist(),
            "halfway_to_expiry": message.encode(15).tolist(),
            "at_expiry": message.encode(20).tolist(),
        },
        "optimizer_updates": 0,
        "market_data_loaded": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
