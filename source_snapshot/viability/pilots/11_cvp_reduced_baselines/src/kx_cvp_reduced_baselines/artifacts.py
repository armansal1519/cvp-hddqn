"""Small artifact helpers for the Pilot 11 reduced baseline adapter.

The implementation deliberately delegates canonical serialization and path
handling to the already frozen Pilot 10 utility layer.  Pilot 11 binds that
entire inherited implementation by digest in addition to binding these local
adapter sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from kx_cvp_baselines.artifacts import (
    atomic_json,
    canonical_json_sha256,
    file_sha256,
    project_relative,
    read_json_object,
    resolve_project_path,
    workspace_root,
)


def verify_project_file(record: object) -> Path:
    """Resolve and hash-check one portable project-root file binding."""

    if not isinstance(record, Mapping) or record.get("path_scope") != "project_root":
        raise ValueError("project-root artifact binding is malformed")
    path = resolve_project_path(record.get("path"))
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError("project-root artifact binding hash differs")
    return path


def project_file_record(path: Path | str) -> dict[str, object]:
    """Return a portable immutable binding for an existing project file."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": project_relative(source),
        "path_scope": "project_root",
        "sha256": file_sha256(source),
    }


__all__ = [
    "atomic_json",
    "canonical_json_sha256",
    "file_sha256",
    "project_file_record",
    "project_relative",
    "read_json_object",
    "resolve_project_path",
    "verify_project_file",
    "workspace_root",
]
