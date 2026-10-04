"""Bridge to the standard-library freeze validator and physical data guard."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from .artifacts import (
    canonical_json_sha256, file_sha256, project_relative, read_json_object,
    resolve_project_path, workspace_root,
)


def contract_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts"


def _validator() -> ModuleType:
    path = contract_dir() / "validate_ablation_protocol.py"
    name = "kx_cvp_ablations._frozen_protocol_validator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen ablation protocol validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_frozen_contracts() -> object:
    return _validator().assert_frozen(contract_dir(), repo_root=workspace_root())


def protocol_binding() -> dict[str, object]:
    contracts = assert_frozen_contracts()
    return {
        "path": project_relative(contract_dir() / "CVP_ABLATION_PROTOCOL.json"),
        "path_scope": "project_root",
        "sha256": contracts.protocol_file_sha256,
        "canonical_sha256": contracts.protocol_canonical_sha256,
        "freeze_receipt": {
            "path": project_relative(contract_dir() / "CVP_ABLATION_FREEZE_RECEIPT.json"),
            "path_scope": "project_root",
            "sha256": file_sha256(contract_dir() / "CVP_ABLATION_FREEZE_RECEIPT.json"),
        },
    }


def implementation_binding() -> dict[str, object]:
    source = Path(__file__).resolve().parent
    paths = sorted(source.glob("*.py")) + [contract_dir() / "validate_ablation_protocol.py"]
    files = [
        {"path": project_relative(path), "sha256": file_sha256(path)} for path in paths
    ]
    return {
        "path_scope": "project_root",
        "files": files,
        "file_count": len(files),
        "aggregate_sha256": canonical_json_sha256(files),
    }


def verify_implementation_binding(value: object) -> None:
    if not isinstance(value, dict) or value != implementation_binding():
        raise ValueError("ablation implementation binding changed")


def registered_parent_member(parent_arm: str, seed: int) -> tuple[Path, dict[str, object]]:
    """Resolve only the parent member frozen into the exact complete receipt."""

    contracts = assert_frozen_contracts()
    artifacts = contracts.protocol.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("ablation protocol lacks artifact bindings")
    record = artifacts.get("parent_complete_training_manifest")
    if not isinstance(record, dict):
        raise ValueError("ablation protocol lacks parent training binding")
    complete_path = resolve_project_path(record.get("path"))
    if not complete_path.is_file() or file_sha256(complete_path) != record.get("sha256"):
        raise ValueError("frozen parent complete-training manifest changed")
    complete = read_json_object(complete_path)
    public_arm = {
        "rich_volume": "V6-RichVolume",
        "base_ohlcv": "V6-BaseOHLCV",
    }.get(parent_arm)
    if public_arm is None:
        raise ValueError("unregistered frozen parent arm")
    checkpoints = complete.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise ValueError("parent complete-training receipt lacks checkpoints")
    matched = [
        value for value in checkpoints
        if isinstance(value, dict)
        and value.get("arm") == public_arm
        and int(value.get("seed", -1)) == int(seed)
    ]
    if len(matched) != 1:
        raise ValueError("parent complete-training member is absent or duplicated")
    one = matched[0]
    member = (complete_path.parent / str(one.get("manifest", ""))).resolve()
    if not member.is_file() or file_sha256(member) != one.get("manifest_sha256"):
        raise ValueError("frozen parent member manifest changed")
    return member, one


def assert_registered_parent_root(root: Path, *, parent_arm: str, seed: int) -> Path:
    member, _ = registered_parent_member(parent_arm, seed)
    supplied = Path(root).resolve() / "manifest.json"
    if supplied != member:
        raise ValueError("supplied parent root is not the frozen registered member")
    return member


@dataclass(frozen=True)
class AuthorizedDataset:
    purpose: str
    period_id: str
    manifest_path: Path
    manifest_sha256: str
    dates: tuple[str, ...]
    pairs: tuple[tuple[Path, Path], ...]

    def binding(self) -> dict[str, object]:
        return {
            "path": project_relative(self.manifest_path),
            "path_scope": "project_root",
            "sha256": self.manifest_sha256,
            "purpose": self.purpose,
            "period_id": self.period_id,
            "dates": list(self.dates),
        }


def load_authorized_dataset(
    relative_manifest: str, *, purpose: str, period_id: str,
    maximum_sessions: int | None = None,
) -> AuthorizedDataset:
    """Deny an unregistered request lexically before opening any path."""

    validator = _validator()
    # This function is intentionally first: it performs no filesystem I/O.
    authorized = validator.authorize_manifest_request(
        relative_manifest, purpose, period_id,
    )
    assert_frozen_contracts()
    value = validator.load_registered_manifest(
        workspace_root(), authorized.relative_path, purpose=purpose, period_id=period_id,
    )
    records = value.get("sessions")
    if not isinstance(records, dict) or not records:
        raise ValueError("registered dataset manifest has no sessions")
    dates = tuple(sorted(str(day) for day in records))
    if maximum_sessions is not None:
        if purpose != "train" or maximum_sessions < 1:
            raise PermissionError("a positive data cap is allowed only for training smoke")
        dates = dates[:maximum_sessions]
    pairs: list[tuple[Path, Path]] = []
    manifest_path = workspace_root() / authorized.relative_path
    for day in dates:
        validator.assert_registered_date(day, purpose, period_id)
        record = records[day]
        if not isinstance(record, dict):
            raise ValueError("registered dataset session record is malformed")
        raw = manifest_path.parent / "sessions" / f"{day}.npz"
        side = manifest_path.parent / "sidecars" / f"{day}.npz"
        if (
            not raw.is_file()
            or not side.is_file()
            or file_sha256(raw) != record.get("session_sha256")
            or file_sha256(side) != record.get("sidecar_sha256")
        ):
            raise ValueError(f"registered session hash differs for {day}")
        pairs.append((raw, side))
    return AuthorizedDataset(
        purpose, period_id, manifest_path, authorized.sha256, dates, tuple(pairs),
    )


def validate_complete_training_value(value: dict[str, object]) -> None:
    errors = _validator().validate_complete_training_manifest(value)
    if errors:
        raise ValueError("; ".join(errors))


def validate_complete_evaluation_value(value: dict[str, object]) -> None:
    errors = _validator().validate_complete_evaluation_manifest(value)
    if errors:
        raise ValueError("; ".join(errors))


__all__ = [
    "AuthorizedDataset", "assert_frozen_contracts", "contract_dir",
    "assert_registered_parent_root", "implementation_binding", "load_authorized_dataset",
    "protocol_binding", "registered_parent_member",
    "validate_complete_evaluation_value", "validate_complete_training_value",
    "verify_implementation_binding",
]
