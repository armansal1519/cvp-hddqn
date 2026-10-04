"""Atomic, hash-bound JSON, Parquet, and PyTorch artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


def workspace_root() -> Path:
    root = Path(__file__).resolve().parents[5]
    if not (root / "viability").is_dir():
        raise RuntimeError("cannot locate thesis workspace root")
    return root


def project_relative(path: Path, *, root: Path | None = None) -> str:
    base = workspace_root() if root is None else Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError as error:
        raise ValueError("source artifact is outside the thesis workspace") from error


def resolve_project_path(value: object, *, root: Path | None = None) -> Path:
    base = workspace_root() if root is None else Path(root).resolve()
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError("portable evidence paths must be project-root-relative")
    result = (base / relative).resolve()
    try:
        result.relative_to(base)
    except ValueError as error:
        raise ValueError("portable evidence path escapes the thesis workspace") from error
    return result


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_object(path: str | os.PathLike[str]) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def atomic_json(
    path: str | os.PathLike[str], value: Mapping[str, object], *, immutable: bool = True,
) -> Path:
    destination = Path(path)
    if immutable and destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.",
        suffix=".part", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(value), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_parquet(
    path: str | os.PathLike[str], rows: Sequence[Mapping[str, object]], *,
    schema: pa.Schema | None = None,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    if not rows and schema is None:
        raise ValueError("Parquet trace cannot be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    temporary = destination.with_name(f".{destination.name}.part")
    try:
        pq.write_table(table, temporary, compression="zstd", version="2.6")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_torch_save(path: str | os.PathLike[str], value: Mapping[str, object]) -> Path:
    import torch

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def relative_path(anchor: Path, path: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=Path(anchor).resolve())


def resolve_bound_path(anchor: Path, value: object, *, require_within: Path | None = None) -> Path:
    path = Path(str(value))
    result = path.resolve() if path.is_absolute() else (Path(anchor) / path).resolve()
    if require_within is not None:
        try:
            result.relative_to(Path(require_within).resolve())
        except ValueError as error:
            raise ValueError("artifact path escapes its registered root") from error
    return result


def verify_file_record(
    anchor: Path, record: object, *, require_within: Path | None = None,
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError("artifact file record is not a mapping")
    path = resolve_bound_path(anchor, record.get("path"), require_within=require_within)
    if not path.is_file() or file_sha256(path) != record.get("sha256"):
        raise ValueError(f"artifact file hash differs: {path}")
    return path


def torch_state_sha256(state: Mapping[str, object]) -> str:
    """Canonical digest independent of the nondeterministic torch container."""

    import numpy as np
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state {name!r} is not a torch tensor")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "atomic_json", "atomic_parquet", "atomic_torch_save", "canonical_json_sha256", "file_sha256",
    "project_relative", "read_json_object", "relative_path", "resolve_bound_path",
    "resolve_project_path", "torch_state_sha256", "verify_file_record", "workspace_root",
]
