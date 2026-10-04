"""Frozen-contract bridge and lexical data-access guard for Pilot 10."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import importlib.util
from pathlib import Path, PurePosixPath
import platform
import sys
from types import ModuleType

from .artifacts import (
    canonical_json_sha256,
    file_sha256,
    project_relative,
    read_json_object,
    resolve_project_path,
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
    """Reject unopened August text without resolving, listing, or opening I/O."""

    for value in values:
        lowered = str(value).replace("\\", "/").casefold()
        if any(marker in lowered for marker in FORBIDDEN_LEXICAL_MARKERS):
            raise PermissionError("August 2026 is unopened and unauthorized")


def validate_workspace_output(
    value: object, *, allow_temporary: bool = False,
) -> Path:
    """Resolve an output after lexical checks; reject outside/paper paths."""

    reject_forbidden_text(value)
    supplied = Path(str(value))
    result = (
        supplied.resolve()
        if supplied.is_absolute()
        else resolve_project_path(supplied.as_posix())
    )
    workspace = workspace_root().resolve()
    try:
        result.relative_to(workspace)
    except ValueError as error:
        if allow_temporary:
            temporary_roots = tuple(
                path.resolve() for path in (Path("/private/tmp"), Path("/tmp"))
                if path.exists()
            )
            if any(result == root or root in result.parents for root in temporary_roots):
                return result
        raise PermissionError("Pilot 10 outputs must remain inside the workspace") from error
    paper = (workspace / "paper").resolve()
    try:
        result.relative_to(paper)
    except ValueError:
        return result
    raise PermissionError("Pilot 10 outputs cannot be written below paper/")


def contract_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "contracts"


def _validator() -> ModuleType:
    path = contract_dir() / "validate_baseline_protocol.py"
    name = "kx_cvp_baselines._frozen_protocol_validator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Pilot 10 protocol validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_frozen_contracts() -> object:
    return _validator().assert_frozen(contract_dir(), repo_root=workspace_root())


def protocol_binding() -> dict[str, object]:
    contracts = assert_frozen_contracts()
    protocol_path = contract_dir() / "CVP_BASELINE_SUITE_PROTOCOL.json"
    receipt_path = contract_dir() / "CVP_BASELINE_SUITE_FREEZE_RECEIPT.json"
    return {
        "path": project_relative(protocol_path),
        "path_scope": "project_root",
        "sha256": contracts.protocol_file_sha256,
        "canonical_sha256": contracts.protocol_canonical_sha256,
        "freeze_receipt": {
            "path": project_relative(receipt_path),
            "path_scope": "project_root",
            "sha256": file_sha256(receipt_path),
        },
    }


@dataclass(frozen=True)
class ManifestAuthorization:
    relative_path: str
    sha256: str
    purpose: str
    period_id: str
    classification: str


def _lexical_path(value: object) -> str:
    """Validate a portable path without touching the filesystem."""

    raw = str(value).replace("\\", "/")
    reject_forbidden_text(raw)
    path = PurePosixPath(raw)
    if path.is_absolute() or not raw or any(part in {"", ".", ".."} for part in path.parts):
        raise PermissionError("dataset manifest path is not a portable registered path")
    return path.as_posix()


def authorize_manifest_request(
    relative_manifest: object, *, purpose: str, period_id: str,
) -> ManifestAuthorization:
    """Authorize an exact manifest lexically, before any filesystem operation."""

    # Ordering is an integrity property: inspect all caller-controlled strings
    # before importing/loading contracts or resolving/listing/opening a path.
    path = _lexical_path(relative_manifest)
    selected_period = str(period_id).casefold()
    _lexical_path(selected_period)
    selected_purpose = str(purpose)
    if selected_purpose not in {"train", "evaluate"}:
        raise PermissionError("unregistered dataset purpose")
    record = DATASET_MANIFESTS.get(selected_period)
    if not isinstance(record, dict):
        raise PermissionError("unregistered dataset period")
    if record["purpose"] != selected_purpose or record["path"] != path:
        raise PermissionError("dataset request differs from the registered manifest")
    return ManifestAuthorization(
        relative_path=path,
        sha256=str(record["sha256"]),
        purpose=selected_purpose,
        period_id=selected_period,
        classification=PERIOD_CLASSIFICATION[selected_period],
    )


@dataclass(frozen=True)
class AuthorizedDataset:
    purpose: str
    period_id: str
    classification: str
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
            "classification": self.classification,
            "dates": list(self.dates),
        }


def load_authorized_dataset(
    relative_manifest: object, *, purpose: str, period_id: str,
    maximum_sessions: int | None = None,
) -> AuthorizedDataset:
    """Open only an exact hash-bound manifest after pure lexical authorization."""

    authorization = authorize_manifest_request(
        relative_manifest, purpose=purpose, period_id=period_id,
    )
    assert_frozen_contracts()
    root = workspace_root()
    manifest_path = root / authorization.relative_path
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != authorization.sha256
    ):
        raise ValueError("registered dataset manifest hash differs")
    value = read_json_object(manifest_path)
    records = value.get("sessions")
    if not isinstance(records, dict) or not records:
        raise ValueError("registered dataset manifest has no sessions")
    dates = tuple(sorted(str(day) for day in records))
    if maximum_sessions is not None:
        if purpose != "train" or maximum_sessions < 1:
            raise PermissionError(
                "a positive session cap is permitted only for training smoke tests"
            )
        dates = dates[: int(maximum_sessions)]
    pairs: list[tuple[Path, Path]] = []
    for day in dates:
        if authorization.period_id == "development":
            if not "2025-01-01" <= day < "2026-05-01":
                raise PermissionError("training manifest contains a nondevelopment date")
        else:
            from .contract import assert_registered_dates

            assert_registered_dates([day], authorization.period_id)
        record = records[day]
        if not isinstance(record, dict):
            raise ValueError("registered session record is malformed")
        raw = manifest_path.parent / "sessions" / f"{day}.npz"
        sidecar = manifest_path.parent / "sidecars" / f"{day}.npz"
        if (
            not raw.is_file()
            or not sidecar.is_file()
            or file_sha256(raw) != record.get("session_sha256")
            or file_sha256(sidecar) != record.get("sidecar_sha256")
        ):
            raise ValueError(f"registered session hash differs for {day}")
        pairs.append((raw, sidecar))
    return AuthorizedDataset(
        purpose=authorization.purpose,
        period_id=authorization.period_id,
        classification=authorization.classification,
        manifest_path=manifest_path,
        manifest_sha256=authorization.sha256,
        dates=dates,
        pairs=tuple(pairs),
    )


def implementation_binding() -> dict[str, object]:
    """Bind every runtime module plus the validator immediately before a job."""

    source = Path(__file__).resolve().parent
    root = workspace_root()
    external_sources = (
        root / "viability/pilots/09_cvp_ablations/src/kx_cvp_ablations",
        root / "viability/pilots/08_simple_hmadrl/src/kx_simple_hmadrl",
    )
    paths = (
        sorted(source.glob("*.py"))
        + [contract_dir() / "validate_baseline_protocol.py"]
        + [
            path
            for directory in external_sources
            for path in sorted(directory.rglob("*.py"))
        ]
    )
    files = [
        {"path": project_relative(path), "sha256": file_sha256(path)}
        for path in paths
    ]
    environment_paths = [
        root / "viability/uv.lock",
        root / "viability/pyproject.toml",
        root / "viability/pilots/10_cvp_baselines/pyproject.toml",
        root / "viability/pilots/09_cvp_ablations/pyproject.toml",
        root / "viability/pilots/08_simple_hmadrl/pyproject.toml",
    ]
    environment_files = [
        {"path": project_relative(path), "sha256": file_sha256(path)}
        for path in environment_paths
    ]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("lightgbm", "matplotlib", "numpy", "pyarrow", "pypdf", "torch")
    }
    environment = {
        "files": environment_files,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
    }
    environment["aggregate_sha256"] = canonical_json_sha256(environment)
    return {
        "path_scope": "project_root",
        "files": files,
        "file_count": len(files),
        "aggregate_sha256": canonical_json_sha256(files),
        "environment": environment,
    }


def verify_implementation_binding(value: object) -> None:
    if not isinstance(value, dict) or value != implementation_binding():
        raise ValueError("Pilot 10 implementation binding changed")


__all__ = [
    "AuthorizedDataset",
    "FORBIDDEN_LEXICAL_MARKERS",
    "ManifestAuthorization",
    "assert_frozen_contracts",
    "authorize_manifest_request",
    "contract_dir",
    "implementation_binding",
    "load_authorized_dataset",
    "protocol_binding",
    "reject_forbidden_text",
    "validate_workspace_output",
    "verify_implementation_binding",
]
