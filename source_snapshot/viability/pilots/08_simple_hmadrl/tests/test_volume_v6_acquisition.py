from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from kx_simple_hmadrl.volume_v6_acquisition import (
    ACTIVE_MAP_SOURCE_ARTIFACT,
    active_map_receipt_path,
    build_dataset,
    create_active_map_receipt,
    create_eligible_phase_universe,
    validate_active_contract_map,
    validate_build_access,
)
from kx_simple_hmadrl.volume_v6_eval import canonical_json_sha256, frozen_protocol


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_universe(
    path: Path,
    rows: list[dict[str, str]],
    *,
    archive_root: Path | None = None,
    phase: str = "smoke",
    start: str = "2025-01-01",
    end_exclusive: str = "2025-02-01",
) -> Path:
    _, protocol_sha = frozen_protocol()
    archive = (path.parent / "archive" if archive_root is None else archive_root).resolve()
    complete_rows: list[dict[str, object]] = []
    for row in rows:
        day = date.fromisoformat(row["session_date"])
        evidence: dict[str, list[dict[str, object]]] = {}
        for layer, key in (
            ("l2_depth", "l2_depth_files"), ("l1_trades", "l1_trade_files"),
        ):
            child = (
                archive / layer / f"year={day.year:04d}" / f"month={day.month:02d}"
                / f"day={day.day:02d}" / "part.parquet"
            )
            child.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"raw_symbol": [row["raw_symbol"]]}).to_parquet(child, index=False)
            evidence[key] = [{
                "path": os.path.relpath(child, start=path.parent),
                "sha256": _sha(child), "bytes": child.stat().st_size,
            }]
        complete_rows.append({**row, **evidence})
    value = {
        "schema_version": 1,
        "artifact_type": ACTIVE_MAP_SOURCE_ARTIFACT,
        "protocol_sha256": protocol_sha,
        "phase": phase,
        "start": start,
        "end_exclusive": end_exclusive,
        "archive_root": os.path.relpath(archive, start=path.parent),
        "eligible_phase_universe_complete": True,
        "source_coverage_complete": True,
        "session_count": len(complete_rows),
        "session_rows_sha256": canonical_json_sha256(complete_rows),
        "sessions": complete_rows,
    }
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


def _active_map(
    root: Path, *, extra: dict[str, object] | None = None,
    source_rows: list[dict[str, str]] | None = None,
    archive_root: Path | None = None,
) -> Path:
    row: dict[str, object] = {
        "session_date": "2025-01-02",
        "selected_raw_symbol": "GCG5",
        "prior_completed_session": "2024-12-31",
        "same_session_volume_used": False,
        "source_coverage_complete": True,
        "volume_cutoff_available_time_ns": 100,
        "decision_time_utc_ns": 200,
    }
    row.update(extra or {})
    path = root / "active.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(path, index=False)
    source = root / "roll-construction-source.json"
    _source_universe(source, source_rows or [{
        "session_date": "2025-01-02", "raw_symbol": "GCG5",
    }], archive_root=archive_root)
    create_active_map_receipt(path, source)
    return path


def test_train_builder_cannot_cross_into_may() -> None:
    validate_build_access("train", "2025-01-01", "2026-05-01")
    with pytest.raises(PermissionError):
        validate_build_access("train", "2025-01-01", "2026-05-02")


def test_validation_builder_requires_complete_training_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = tmp_path / "training.json"
    incomplete.write_text(json.dumps({"complete": False}))
    with pytest.raises(PermissionError):
        validate_build_access(
            "validation", "2026-05-01", "2026-06-01",
            training_manifest=incomplete,
        )
    incomplete.write_text(json.dumps({"complete": True}))
    verified: list[Path] = []
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_acquisition._deep_verify_complete_training_manifest",
        lambda path: verified.append(Path(path)) or {"complete": True},
    )
    validate_build_access(
        "validation", "2026-05-01", "2026-06-01",
        training_manifest=incomplete,
    )
    assert verified == [incomplete]


def test_test_builder_checks_calendar_before_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_acquisition._current_date",
        lambda: date(2026, 8, 23),
    )
    with pytest.raises(PermissionError, match="before August"):
        validate_build_access(
            "test", "2026-06-01", "2026-09-01",
            unlock_receipt=Path("missing"),
        )


def test_test_builder_invokes_deep_unlock_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    unlock = tmp_path / "unlock.json"
    unlock.write_text("{}")
    verified: list[Path] = []
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_acquisition._current_date",
        lambda: date(2026, 9, 1),
    )
    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_acquisition._deep_verify_test_unlock",
        lambda path: verified.append(Path(path)) or {"complete": True},
    )
    validate_build_access(
        "test", "2026-06-01", "2026-09-01", unlock_receipt=unlock,
    )
    assert verified == [unlock]


def test_active_map_requires_receipt_bound_past_only_evidence(tmp_path: Path) -> None:
    path = _active_map(tmp_path / "valid")
    selected = validate_active_contract_map(
        path, "2025-01-01", "2025-02-01", phase="smoke",
    )
    assert selected["v6_session_date"].tolist() == ["2025-01-02"]

    missing = tmp_path / "missing" / "active.parquet"
    missing.parent.mkdir(parents=True)
    pd.DataFrame([{"session_date": "2025-01-02", "selected_raw_symbol": "GCG5"}]).to_parquet(
        missing, index=False,
    )
    with pytest.raises(ValueError, match="construction receipt"):
        validate_active_contract_map(
            missing, "2025-01-01", "2025-02-01", phase="smoke",
        )

    with pytest.raises(ValueError, match="current-session volume"):
        _active_map(tmp_path / "leaking", extra={"same_session_volume_used": True})

    with pytest.raises(ValueError, match="unavailable"):
        _active_map(
            tmp_path / "unavailable", extra={"volume_cutoff_available_time_ns": 200},
        )


def test_active_map_source_must_be_authoritative_and_exact(tmp_path: Path) -> None:
    root = tmp_path / "arbitrary"
    root.mkdir()
    active = root / "active.parquet"
    pd.DataFrame([{
        "session_date": "2025-01-02", "selected_raw_symbol": "GCG5",
        "prior_completed_session": "2024-12-31", "same_session_volume_used": False,
        "source_coverage_complete": True, "volume_cutoff_available_time_ns": 100,
        "decision_time_utc_ns": 200,
    }]).to_parquet(active, index=False)
    arbitrary = root / "source.json"
    arbitrary.write_text(json.dumps({"source": "not authoritative"}))
    with pytest.raises(ValueError, match="authoritative universe"):
        create_active_map_receipt(active, arbitrary)

    wrong = root / "wrong.json"
    _source_universe(wrong, [{
        "session_date": "2025-01-02", "raw_symbol": "GCJ5",
    }])
    with pytest.raises(ValueError, match="exact eligible source"):
        create_active_map_receipt(active, wrong)

    extra = root / "extra.json"
    _source_universe(extra, [
        {"session_date": "2025-01-02", "raw_symbol": "GCG5"},
        {"session_date": "2025-01-03", "raw_symbol": "GCG5"},
    ])
    with pytest.raises(ValueError, match="exact eligible source"):
        create_active_map_receipt(active, extra)


def test_create_eligible_phase_universe_hashes_exact_archive_children(tmp_path: Path) -> None:
    root = tmp_path / "registration"
    active = root / "active.parquet"
    active.parent.mkdir(parents=True)
    pd.DataFrame([{
        "session_date": "2025-01-02", "selected_raw_symbol": "GCG5",
        "prior_completed_session": "2024-12-31", "same_session_volume_used": False,
        "source_coverage_complete": True, "volume_cutoff_available_time_ns": 100,
        "decision_time_utc_ns": 200,
    }]).to_parquet(active, index=False)
    archive = root / "archive"
    bootstrap = root / "bootstrap.json"
    _source_universe(bootstrap, [{
        "session_date": "2025-01-02", "raw_symbol": "GCG5",
    }], archive_root=archive)
    bootstrap.unlink()
    source = create_eligible_phase_universe(
        active, archive, root / "universe.json", phase="smoke",
        start="2025-01-01", end_exclusive="2025-02-01",
    )
    value = json.loads(source.read_text())
    assert value["session_count"] == 1
    assert len(value["sessions"][0]["l2_depth_files"]) == 1
    assert len(value["sessions"][0]["l1_trade_files"]) == 1
    create_active_map_receipt(active, source)
    child_record = value["sessions"][0]["l2_depth_files"][0]
    child = (source.parent / child_record["path"]).resolve()
    child.write_bytes(child.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="child file"):
        validate_active_contract_map(
            active, "2025-01-01", "2025-02-01", phase="smoke",
        )


def test_active_map_receipt_detects_map_or_source_mutation(tmp_path: Path) -> None:
    path = _active_map(tmp_path)
    source = tmp_path / "roll-construction-source.json"
    source.write_text("changed")
    with pytest.raises(ValueError, match="source receipt"):
        validate_active_contract_map(
            path, "2025-01-01", "2025-02-01", phase="smoke",
        )


def _resume_identity(active_map: Path, archive_root: Path) -> dict[str, object]:
    _, protocol_sha = frozen_protocol()
    receipt = active_map_receipt_path(active_map)
    return {
        "protocol_sha256": protocol_sha,
        "phase": "smoke",
        "start": "2025-01-02",
        "end_exclusive": "2025-01-03",
        "archive_root": str(archive_root.resolve()),
        "active_contract_map": str(active_map.resolve()),
        "active_contract_map_sha256": _sha(active_map),
        "active_contract_map_receipt": str(receipt.resolve()),
        "active_contract_map_receipt_sha256": _sha(receipt),
        "limit_sessions": None,
    }


def test_build_resume_binds_archive_root_and_rejects_extra_records(tmp_path: Path) -> None:
    first_archive = tmp_path / "archive-a"; first_archive.mkdir()
    second_archive = tmp_path / "archive-b"; second_archive.mkdir()
    active_map = _active_map(tmp_path / "map", archive_root=first_archive)
    output = tmp_path / "dataset"
    work = tmp_path / ".dataset.part"
    work.mkdir()
    progress = {
        "schema_version": 1,
        "artifact_type": "V6_GC_BUILD_PROGRESS",
        "identity": _resume_identity(active_map, first_archive),
        "sessions": {},
    }
    (work / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="archive_root"):
        build_dataset(
            second_archive, active_map, output, start="2025-01-02",
            end_exclusive="2025-01-03", phase="smoke",
        )

    source = tmp_path / "source.parquet"; source.write_bytes(b"source")
    raw = work / "sessions" / "2025-01-02.npz"
    side = work / "sidecars" / "2025-01-02.npz"
    raw.parent.mkdir(parents=True); side.parent.mkdir(parents=True)
    raw.write_bytes(b"raw"); side.write_bytes(b"side")
    audit = {
        "session_date": "2025-01-02", "status": "PASS", "continuity_segments": 1,
    }
    source_record = {str(source): {"sha256": _sha(source), "bytes": source.stat().st_size}}
    one = {
        "raw_symbol": "GCG5", "session_sha256": _sha(raw),
        "sidecar_sha256": _sha(side), "audit": audit, "sources": source_record,
    }
    progress["identity"] = _resume_identity(active_map, first_archive)
    progress["sessions"] = {"2025-01-02": one, "2025-01-03": dict(one)}
    (work / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="outside the selected active map"):
        build_dataset(
            first_archive, active_map, output, start="2025-01-02",
            end_exclusive="2025-01-03", phase="smoke",
        )


def test_registered_builder_rejects_multi_continuity_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    active_map = _active_map(tmp_path / "map", archive_root=archive)

    def fake_read_day(
        root: Path, kind: str, day: str, symbol: str, columns: list[str],
    ) -> tuple[pd.DataFrame, list[Path]]:
        del root, day, symbol, columns
        if kind == "l2_depth":
            return pd.DataFrame({"continuity_segment_id": [7, 8]}), []
        return pd.DataFrame({"one_trade": [1]}), []

    monkeypatch.setattr(
        "kx_simple_hmadrl.volume_v6_acquisition._read_day", fake_read_day,
    )
    with pytest.raises(ValueError, match="exactly one continuity segment"):
        build_dataset(
            archive, active_map, tmp_path / "dataset", start="2025-01-02",
            end_exclusive="2025-01-03", phase="smoke",
        )
