#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run eight existing synthetic checks without training or market-data replay."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


RELEASE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = (
    RELEASE_ROOT / "source_snapshot" / "viability" / "pilots"
    / "08_simple_hmadrl" / "tests"
)
SELECTED_TESTS = {
    "test_volume_v6_model.py": (
        "test_regime_message_is_semantic_centered_and_expires_to_zero",
        "test_two_level_models_match_registered_shapes_and_parameter_limits",
        "test_xau_transfer_copies_only_common_base_encoders",
        "test_padding_values_are_inert_but_valid_semantic_inputs_matter",
        "test_model_rejects_wrong_shapes_dtypes_and_nonfinite_values",
    ),
    "test_volume_v6_env.py": (
        "test_semantic_actions_and_inventory_masks_have_one_unambiguous_mapping",
        "test_direct_reversal_is_rejected_before_any_environment_mutation",
        "test_terminal_transition_reward_equals_after_cost_cash_cent_for_cent",
    ),
}
CORE_MODULES = (
    "kx_simple_hmadrl.volume_v6",
    "kx_simple_hmadrl.volume_v6_contract",
    "kx_simple_hmadrl.volume_v6_training",
    "kx_simple_hmadrl.volume_v6_data",
    "kx_simple_hmadrl.volume_v6_env",
    "kx_simple_hmadrl.environment",
    "kx_base_gc.contract",
)


def assert_package_origins(expected_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(("kx_simple_hmadrl", "kx_base_gc")):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None or not Path(filename).resolve().is_relative_to(expected_root):
            raise AssertionError(f"Unexpected package origin for {name}: {filename}")


def forbid_optimizer_step(*args: object, **kwargs: object) -> None:
    raise AssertionError("Optimizer updates are forbidden in the synthetic core check")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-package-root", type=Path,
        help=("Directory containing an installed wheel's packages; use this instead "
              "of the release src directory and verify every core import originates there."),
    )
    args = parser.parse_args()
    expected_root = (args.expect_package_root or RELEASE_ROOT / "src").resolve()
    if not expected_root.is_dir():
        parser.error(f"Package root does not exist: {expected_root}")
    sys.path.insert(0, str(expected_root))

    import numpy as np
    import torch
    import yaml

    torch.set_num_threads(1)
    for name in CORE_MODULES:
        __import__(name, fromlist=["*"])
    assert_package_origins(expected_root)
    passed: list[str] = []
    with ExitStack() as guards, torch.no_grad():
        # These selected tests do not optimize; fail explicitly if that changes.
        for candidate in vars(torch.optim).values():
            if isinstance(candidate, type) and issubclass(candidate, torch.optim.Optimizer):
                guards.enter_context(patch.object(candidate, "step", forbid_optimizer_step))
        for filename, names in SELECTED_TESTS.items():
            path = TEST_ROOT / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing original synthetic-test source: {path}")
            spec = importlib.util.spec_from_file_location(
                f"_cvp_release_{path.stem}", path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load test source: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in names:
                getattr(module, name)()
                passed.append(name)
    assert len(passed) == 8
    assert_package_origins(expected_root)
    print(json.dumps({
        "status": "PASS",
        "scope": "Selected synthetic unit checks; no empirical performance claim",
        "package_root": str(expected_root),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "numpy": np.__version__, "PyYAML": yaml.__version__},
        "checks_passed": len(passed),
        "checks": passed,
        "optimizer_updates": 0,
        "market_data_loaded": False,
        "financial_replay_executed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
