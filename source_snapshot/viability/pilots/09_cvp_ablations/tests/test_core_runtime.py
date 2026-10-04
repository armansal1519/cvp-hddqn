from __future__ import annotations

from pathlib import Path

from kx_cvp_ablations.artifacts import read_json_object, workspace_root
from kx_cvp_ablations.contract import AblationArm
from kx_cvp_ablations.initialization import _registered_xau_lineage
from kx_cvp_ablations.training import _reference_member, registered_trainer_settings


def test_no_publisher_reference_loads_only_the_registered_scaler() -> None:
    root = (
        workspace_root()
        / "viability/artifacts/development/"
        "v6_registered_training_terminal_eligible/rich_volume/seed_1103"
    )
    reference = _reference_member(
        root, arm=AblationArm.NO_PUBLISHER, seed=1103,
    )
    assert reference.model is None
    assert reference.parent_arm == "rich_volume"
    assert reference.seed == 1103


def test_registered_xau_lineage_matches_frozen_parent_model_hash() -> None:
    root = (
        workspace_root()
        / "viability/artifacts/development/"
        "v6_registered_xau_pretraining/v6_price/seed_1103"
    )
    manifest = read_json_object(root / "manifest.json")
    lineage = _registered_xau_lineage(
        seed=1103, xau_root=root, xau_manifest=manifest, smoke_only=False,
    )
    assert lineage["status"] == "REGISTERED_PARENT_MATCH"
    assert lineage["model_sha256"] == manifest["model_bundle"]["sha256"]


def test_registered_trainer_settings_bind_the_full_training_budget() -> None:
    settings = registered_trainer_settings(expected_decisions=12_345)
    assert settings["algorithm"] == "dueling_double_dqn"
    assert settings["replay_sampling"] == "uniform"
    assert settings["chronological_transition_ingestion"] is True
    assert settings["chronological_session_passes"] is True
    assert settings["reward_clipping"] is False
    assert settings["passes"] == 3
    assert settings["epsilon_decay_decisions"] == 12_345

