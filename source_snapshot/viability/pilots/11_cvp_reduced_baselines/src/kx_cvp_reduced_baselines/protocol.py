"""Frozen-contract bridge and lexical data guard for Pilot 11."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path, PurePosixPath
import sys
from types import ModuleType
from typing import Mapping

from .artifacts import (
    canonical_json_sha256,
    file_sha256,
    project_relative,
    workspace_root,
)
from .contract import DATASET_MANIFESTS, PERIOD_CLASSIFICATION


FORBIDDEN_LEXICAL_MARKERS = (
    "august",
    "2026-08",
    "2026_08",
    "202608",
    "aug_2026",
    "aug-2026",
)


def reject_forbidden_text(*values: object) -> None:
    """Reject unopened August text before any path or contract I/O."""

    for value in values:
        lowered = str(value).replace("\\", "/").casefold()
        if any(marker in lowered for marker in FORBIDDEN_LEXICAL_MARKERS):
            raise PermissionError("August 2026 is unopened and unauthorized")


def validate_workspace_output(
    value: object,
    *,
    allow_temporary: bool = False,
) -> Path:
    """Resolve one output while forbidding paths outside the workspace/paper."""

    reject_forbidden_text(value)
    supplied = Path(str(value))
    root = workspace_root().resolve()
    result = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        if allow_temporary:
            temporary_roots = tuple(
                path.resolve()
                for path in (Path("/private/tmp"), Path("/tmp"))
                if path.exists()
            )
            if any(result == item or item in result.parents for item in temporary_roots):
                return result
        raise PermissionError("Pilot 11 outputs must remain inside the workspace") from error
    paper = (root / "paper").resolve()
    try:
        result.relative_to(paper)
    except ValueError:
        return result
    raise PermissionError("Pilot 11 is not authorized to write below paper/")


def contract_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts"


def _validator() -> ModuleType:
    path = contract_dir() / "validate_reduced_baseline_protocol.py"
    name = "kx_cvp_reduced_baselines._frozen_protocol_validator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Pilot 11 protocol validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_frozen_contracts() -> object:
    return _validator().assert_frozen(contract_dir(), repo_root=workspace_root())


def _contract_file_binding(filename: str) -> dict[str, object]:
    path = contract_dir() / filename
    return {
        "path": project_relative(path),
        "path_scope": "project_root",
        "sha256": file_sha256(path),
    }


def reference_registry_binding() -> dict[str, object]:
    assert_frozen_contracts()
    return _contract_file_binding("CVP_BASELINE_REDUCED_ARM_REGISTRY.json")


def cancellation_ledger_binding() -> dict[str, object]:
    assert_frozen_contracts()
    return _contract_file_binding("CVP_BASELINE_V1_OWNER_CANCELLATION_LEDGER.json")


def protocol_binding() -> dict[str, object]:
    contracts = assert_frozen_contracts()
    result = _contract_file_binding("CVP_BASELINE_REDUCED_SUITE_PROTOCOL.json")
    result["canonical_sha256"] = contracts.protocol_canonical_sha256
    result["freeze_receipt"] = _contract_file_binding(
        "CVP_BASELINE_REDUCED_SUITE_FREEZE_RECEIPT.json"
    )
    result["access_ledger"] = _contract_file_binding(
        "CVP_BASELINE_REDUCED_ACCESS_LEDGER.json"
    )
    result["arm_and_reference_registry"] = reference_registry_binding()
    result["owner_cancellation_ledger"] = cancellation_ledger_binding()
    return result


@dataclass(frozen=True)
class ManifestAuthorization:
    relative_path: str
    sha256: str
    purpose: str
    period_id: str
    classification: str


def _portable_path(value: object) -> str:
    raw = str(value).replace("\\", "/")
    reject_forbidden_text(raw)
    path = PurePosixPath(raw)
    if path.is_absolute() or not raw or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise PermissionError("dataset manifest is not a portable registered path")
    return path.as_posix()


def authorize_manifest_request(
    relative_manifest: object,
    *,
    purpose: str,
    period_id: str,
) -> ManifestAuthorization:
    """Authorize an exact manifest lexically, before filesystem access."""

    path = _portable_path(relative_manifest)
    selected_period = str(period_id).casefold()
    _portable_path(selected_period)
    if purpose not in {"train", "evaluate"}:
        raise PermissionError("unregistered dataset purpose")
    record = DATASET_MANIFESTS.get(selected_period)
    if not isinstance(record, Mapping):
        raise PermissionError("unregistered dataset period")
    if record.get("purpose") != purpose or record.get("path") != path:
        raise PermissionError("dataset request differs from the frozen Pilot 11 manifest")
    return ManifestAuthorization(
        relative_path=path,
        sha256=str(record["sha256"]),
        purpose=purpose,
        period_id=selected_period,
        classification=PERIOD_CLASSIFICATION[selected_period],
    )


def implementation_binding() -> dict[str, object]:
    """Bind the complete Pilot 11 adapter plus inherited frozen lineage."""

    assert_frozen_contracts()
    source = Path(__file__).resolve().parent
    paths = sorted(source.glob("*.py")) + [
        contract_dir() / "validate_reduced_baseline_protocol.py"
    ]
    files = [
        {
            "path": project_relative(path),
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    return {
        "path_scope": "project_root",
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": canonical_json_sha256(files),
        "inherited_pilot10_implementation_aggregate_sha256": (
            "31ce433cf5c5e9a401ee500ff7a990caee91f42267083552095e6bd125a182a0"
        ),
        "inherited_environment_aggregate_sha256": (
            "cf9f2d1538beeea9423446e54d97e6ce3bbb69ee20aa028d269c207f676089fe"
        ),
    }


def verify_implementation_binding(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Pilot 11 implementation binding is malformed")
    expected = implementation_binding()
    observed = dict(value)
    if observed != expected:
        raise ValueError("Pilot 11 implementation binding differs")
    return observed


__all__ = [
    "ManifestAuthorization",
    "assert_frozen_contracts",
    "authorize_manifest_request",
    "cancellation_ledger_binding",
    "contract_dir",
    "implementation_binding",
    "protocol_binding",
    "reference_registry_binding",
    "reject_forbidden_text",
    "validate_workspace_output",
    "verify_implementation_binding",
]
