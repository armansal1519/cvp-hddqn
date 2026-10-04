#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Verify the exact source-only release inventory and its file hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
IGNORED_DIRECTORIES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist",
}
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".csv", ".tsv", ".parquet",
    ".npy", ".npz", ".dbn", ".pdf", ".png", ".svg", ".jpg", ".jpeg",
    ".sha256",
}
ALLOWED_JSON = {MANIFEST, ".zenodo.json", "provenance/source_files.json"}


def ignored(relative: Path) -> bool:
    return (
        any(part in IGNORED_DIRECTORIES or part.endswith(".egg-info")
            for part in relative.parts)
        or relative.name == ".DS_Store"
        or relative.suffix in {".pyc", ".pyo"}
    )


def main() -> int:
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    expected_files = manifest["files"]
    failures = []
    for name, record in expected_files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name == MANIFEST:
            failures.append({"file": name, "reason": "invalid manifest path"})
            continue
        path = ROOT.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            failures.append({"file": name, "reason": "missing or symbolic link"})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append({"file": name, "reason": "changed"})
    actual = set()
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ignored(relative):
            continue
        if path.is_symlink():
            failures.append({"file": relative.as_posix(), "reason": "symbolic link"})
        elif path.is_file() and relative.as_posix() != MANIFEST:
            actual.add(relative.as_posix())
    for name in sorted(actual - set(expected_files)):
        failures.append({"file": name, "reason": "unlisted file"})
    for name in actual:
        relative = PurePosixPath(name)
        if (
            relative.suffix.lower() in FORBIDDEN_SUFFIXES
            or relative.name == ".env" or relative.name.startswith(".env.")
            or (relative.suffix == ".json" and name not in ALLOWED_JSON)
            or relative.parts[0] in {"analysis", "results", "data", "outputs", "figures"}
        ):
            failures.append({"file": name, "reason": "outside source-only payload"})
        if name.startswith("source_snapshot/"):
            if relative.suffix != ".py" and relative.name != "pyproject.toml":
                failures.append({"file": name, "reason": "non-source snapshot artifact"})
    print(json.dumps({
        "status": "FAIL" if failures else "PASS",
        "scope": "Source-only inventory integrity; not a performance assessment",
        "files_checked": len(expected_files),
        "failures": failures,
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
