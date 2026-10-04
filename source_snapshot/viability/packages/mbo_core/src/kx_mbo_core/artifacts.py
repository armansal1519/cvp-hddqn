from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import enforce_2025_frame
from .types import ViabilityVerdict


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def write_verdict(path: str | Path, verdict: ViabilityVerdict) -> Path:
    return write_json(path, verdict.to_dict())


def read_parquet(path: str | Path, *, date_column: str = "date") -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return enforce_2025_frame(frame, date_column=date_column)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _resolve_manifest_artifact(manifest_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def verify_evidence_manifest(
    path: str | Path,
    *,
    inputs: dict[str, str | Path],
    required_claims: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Verify an auditable evidence manifest and every referenced byte hash.

    A row-level boolean is not certification.  This contract binds each pilot
    input and each upstream claim to immutable artifacts.  It is intentionally
    fail-closed: a missing claim, file, or hash is a data error rather than a
    negative result for the research mechanism.
    """

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("evidence manifest must be a schema_version=1 object")
    evidence_class = str(payload.get("evidence_class", "")).upper()
    if evidence_class not in {
        "DEVELOPMENT_ONLY",
        "LOCKED_EMPIRICAL",
        "SYNTHETIC_ONLY",
    }:
        raise ValueError(f"unsupported evidence_class: {evidence_class!r}")
    declared_inputs = payload.get("inputs")
    if not isinstance(declared_inputs, dict):
        raise ValueError("evidence manifest must declare an inputs mapping")
    verified_inputs: dict[str, str] = {}
    for name, input_path in inputs.items():
        if name not in declared_inputs:
            raise ValueError(f"evidence manifest is missing input {name!r}")
        declaration = declared_inputs[name]
        expected = (
            str(declaration.get("sha256", ""))
            if isinstance(declaration, dict)
            else str(declaration)
        )
        observed = sha256_file(input_path)
        if not expected or observed != expected:
            raise ValueError(
                f"input hash mismatch for {name!r}: expected {expected!r}, observed {observed!r}"
            )
        verified_inputs[name] = observed
    claims = payload.get("claims")
    if not isinstance(claims, dict):
        raise ValueError("evidence manifest must declare a claims mapping")
    verified_claims: dict[str, Any] = {}
    for name in required_claims:
        claim = claims.get(name)
        if not isinstance(claim, dict):
            raise ValueError(f"evidence manifest is missing claim {name!r}")
        artifact_value = claim.get("artifact")
        expected = str(claim.get("sha256", ""))
        if not artifact_value or not expected:
            raise ValueError(f"claim {name!r} must bind an artifact and sha256")
        if not isinstance(claim.get("passed"), bool):
            raise ValueError(f"claim {name!r} must contain a JSON boolean 'passed'")
        artifact = _resolve_manifest_artifact(source, str(artifact_value))
        if not artifact.is_file():
            raise ValueError(f"claim artifact does not exist: {artifact}")
        observed = sha256_file(artifact)
        if observed != expected:
            raise ValueError(
                f"claim artifact hash mismatch for {name!r}: expected {expected!r}, observed {observed!r}"
            )
        verified_claims[name] = {
            **claim,
            "passed": claim["passed"],
            "artifact": str(artifact),
            "sha256": observed,
        }
    return {
        **payload,
        "evidence_class": evidence_class,
        "inputs": verified_inputs,
        "claims": {**claims, **verified_claims},
        "manifest_path": str(source),
        "manifest_sha256": sha256_file(source),
    }
