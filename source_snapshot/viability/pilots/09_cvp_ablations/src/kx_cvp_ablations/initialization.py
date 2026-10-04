"""One seed-specific trader initialization cloned across all new arms."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import torch

from kx_simple_hmadrl.volume_v6 import V6ModelBundle, transfer_base_encoders
from kx_simple_hmadrl.volume_v6_training import seed_everything

from .artifacts import (
    atomic_json,
    atomic_torch_save,
    file_sha256,
    project_relative,
    read_json_object,
    torch_state_sha256,
    verify_file_record,
    resolve_project_path,
)
from .contract import ABLATION_ARMS, PROTOCOL_ID, SEEDS
from .frozen_v6 import load_xau_price_bundle
from .protocol import (
    implementation_binding,
    registered_parent_member,
    verify_implementation_binding,
)


INITIALIZATION_ARTIFACT = "CVP_ABLATION_TRADER_INITIALIZATION"
INITIALIZATION_PAYLOAD = "CVP_ABLATION_TRADER_INITIAL_STATE"
REGISTERED_XAU_ROOT = (
    "viability/artifacts/development/"
    "v6_registered_xau_pretraining/v6_price"
)


def _registered_xau_lineage(
    *, seed: int, xau_root: Path, xau_manifest: Mapping[str, object],
    smoke_only: bool,
) -> dict[str, object]:
    expected_root = resolve_project_path(f"{REGISTERED_XAU_ROOT}/seed_{seed}")
    if xau_root != expected_root:
        if not smoke_only:
            raise ValueError("supplied XAU root is not the registered seed member")
        return {
            "status": "SMOKE_UNREGISTERED",
            "registered_parent_required": False,
        }
    parent_path, parent_record = registered_parent_member("rich_volume", seed)
    parent = read_json_object(parent_path)
    parent_warmstart = parent.get("xau_warmstart")
    model_record = xau_manifest.get("model_bundle")
    if not isinstance(parent_warmstart, dict) or not isinstance(model_record, dict):
        raise ValueError("registered parent or XAU member lacks warm-start lineage")
    if parent_warmstart.get("sha256") != model_record.get("sha256"):
        raise ValueError("XAU model differs from the frozen parent warm start")
    return {
        "status": "REGISTERED_PARENT_MATCH",
        "registered_parent_required": True,
        "parent_arm": "rich_volume",
        "parent_member": {
            "path": project_relative(parent_path),
            "path_scope": "project_root",
            "sha256": file_sha256(parent_path),
        },
        "parent_complete_record": {
            "arm": parent_record.get("arm"),
            "seed": int(parent_record.get("seed", -1)),
            "manifest_sha256": parent_record.get("manifest_sha256"),
        },
        "model_sha256": str(model_record.get("sha256")),
    }


def create_trader_initialization(
    *, seed: int, xau_pretraining_root: Path, output: Path, smoke_only: bool = False,
) -> Path:
    """Create once, then reuse the exact trader tensor state for all three arms."""

    seed = int(seed)
    if seed not in SEEDS and not smoke_only:
        raise ValueError(f"registered initialization seed must be one of {SEEDS}")
    destination = Path(output).resolve()
    if (destination / "manifest.json").is_file():
        verify_initialization(
            destination / "manifest.json", expected_seed=seed, allow_smoke=smoke_only,
        )
        return destination / "manifest.json"
    if destination.exists():
        raise FileExistsError("initialization output exists without a valid manifest")
    work = destination.with_name(f".{destination.name}.part")
    if work.exists():
        raise FileExistsError("stale initialization part directory must be audited")
    work.mkdir(parents=True)
    try:
        xau_root = Path(xau_pretraining_root).resolve()
        xau_manifest_path = xau_root / "manifest.json"
        xau_manifest = read_json_object(xau_manifest_path)
        if int(xau_manifest.get("seed", -1)) != seed:
            raise ValueError("XAU warm-start seed differs from initialization seed")
        parent_lineage = _registered_xau_lineage(
            seed=seed, xau_root=xau_root, xau_manifest=xau_manifest,
            smoke_only=smoke_only,
        )
        pretrained, metadata, verified_manifest = load_xau_price_bundle(
            xau_root, expected_seed=seed, allow_smoke=smoke_only,
        )
        if verified_manifest != xau_manifest:
            raise AssertionError("XAU manifest changed during initialization")
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError("loaded XAU warm-start metadata has another seed")

        # The seed is set immediately before construction. Nothing outcome-derived
        # is consulted between seeding and the fresh GC initialization.
        seed_everything(seed)
        fresh = V6ModelBundle.create()
        transfer_base_encoders(pretrained, fresh)
        trader_state = fresh.trader.state_dict()
        state_digest = torch_state_sha256(trader_state)
        implementation = implementation_binding()
        payload_path = work / "trader_init.pt"
        atomic_torch_save(payload_path, {
            "schema_version": 1,
            "artifact_type": INITIALIZATION_PAYLOAD,
            "protocol_id": PROTOCOL_ID,
            "implementation": implementation,
            "seed": seed,
            "trader_state_sha256": state_digest,
            "trader": trader_state,
        })
        manifest = {
            "schema_version": 1,
            "artifact_type": INITIALIZATION_ARTIFACT,
            "experiment_id": PROTOCOL_ID,
            "protocol_id": PROTOCOL_ID,
            "implementation": implementation,
            "seed": seed,
            "eligible_arms": list(ABLATION_ARMS),
            "construction_order": "seed_then_construct_then_xau_price_encoder_transfer",
            "xau_warmstart": {
                "path": project_relative(xau_manifest_path),
                "path_scope": "project_root",
                "sha256": file_sha256(xau_manifest_path),
                "model_sha256": str(xau_manifest["model_bundle"]["sha256"]),
            },
            "parent_warmstart_lineage": parent_lineage,
            "trader_state_sha256": state_digest,
            "payload": {"path": payload_path.name, "sha256": file_sha256(payload_path)},
            "smoke_only": bool(smoke_only),
            "holdout_read": False,
        }
        atomic_json(work / "manifest.json", manifest)
        os.replace(work, destination)
    except Exception:
        # Keep the part directory for resume/audit; initialization is small and
        # intentionally has no automatic destructive cleanup.
        raise
    verify_initialization(destination / "manifest.json", expected_seed=seed, allow_smoke=smoke_only)
    return destination / "manifest.json"


def verify_initialization(
    manifest_path: Path, *, expected_seed: int | None = None, allow_smoke: bool = False,
) -> dict[str, object]:
    path = Path(manifest_path).resolve()
    value = read_json_object(path)
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != INITIALIZATION_ARTIFACT
        or value.get("experiment_id") != PROTOCOL_ID
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("eligible_arms") != list(ABLATION_ARMS)
        or value.get("construction_order")
        != "seed_then_construct_then_xau_price_encoder_transfer"
        or (value.get("smoke_only") is True and not allow_smoke)
        or value.get("holdout_read") is not False
    ):
        raise ValueError("ablation trader initialization manifest is invalid")
    seed = int(value.get("seed", -1))
    verify_implementation_binding(value.get("implementation"))
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError("ablation initialization seed differs")
    if seed not in SEEDS and not allow_smoke:
        raise ValueError("unregistered ablation initialization seed")
    payload_path = verify_file_record(path.parent, value.get("payload"), require_within=path.parent)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != INITIALIZATION_PAYLOAD
        or payload.get("protocol_id") != PROTOCOL_ID
        or int(payload.get("seed", -1)) != seed
        or not isinstance(payload.get("trader"), Mapping)
    ):
        raise ValueError("ablation trader initialization payload is invalid")
    verify_implementation_binding(payload.get("implementation"))
    digest = torch_state_sha256(payload["trader"])
    if digest != value.get("trader_state_sha256") or digest != payload.get(
        "trader_state_sha256"
    ):
        raise ValueError("ablation trader initialization tensor hash differs")
    xau = value.get("xau_warmstart")
    if not isinstance(xau, dict):
        raise ValueError("initialization lacks XAU lineage")
    xau_path = resolve_project_path(xau.get("path"))
    if not xau_path.is_file() or file_sha256(xau_path) != xau.get("sha256"):
        raise ValueError("initialization XAU lineage changed")
    if xau.get("path_scope") != "project_root":
        raise ValueError("initialization XAU path scope differs")
    xau_manifest = read_json_object(xau_path)
    model_path = verify_file_record(
        xau_path.parent, xau_manifest.get("model_bundle"),
        require_within=xau_path.parent,
    )
    if (
        file_sha256(model_path) != xau.get("model_sha256")
        or int(xau_manifest.get("seed", -1)) != seed
    ):
        raise ValueError("initialization XAU model lineage changed")
    expected_parent_lineage = _registered_xau_lineage(
        seed=seed, xau_root=xau_path.parent, xau_manifest=xau_manifest,
        smoke_only=bool(value.get("smoke_only")),
    )
    if value.get("parent_warmstart_lineage") != expected_parent_lineage:
        raise ValueError("initialization parent warm-start lineage changed")
    return value


def load_trader_initialization(
    manifest_path: Path, *, expected_seed: int, allow_smoke: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    value = verify_initialization(
        manifest_path, expected_seed=expected_seed, allow_smoke=allow_smoke,
    )
    path = Path(manifest_path).resolve()
    payload_path = verify_file_record(path.parent, value["payload"], require_within=path.parent)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    return dict(payload["trader"]), value


__all__ = [
    "INITIALIZATION_ARTIFACT", "INITIALIZATION_PAYLOAD", "create_trader_initialization",
    "REGISTERED_XAU_ROOT", "load_trader_initialization", "verify_initialization",
]
