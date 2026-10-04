from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kx_simple_hmadrl.volume_v6_contract import SEEDS
from kx_simple_hmadrl.volume_v6_orchestration import PhaseDataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase(path: Path, phase: str) -> PhaseDataset:
    return PhaseDataset(path.resolve(), phase, (), (), (), _sha(path))


@pytest.mark.parametrize("changed", ["dataset", "pretraining"])
def test_existing_complete_training_rejects_a_different_current_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    old_dataset = tmp_path / "old-train-data.json"
    new_dataset = tmp_path / "new-train-data.json"
    old_pretraining = tmp_path / "old-xau.json"
    new_pretraining = tmp_path / "new-xau.json"
    for path, marker in (
        (old_dataset, "old data"), (new_dataset, "new data"),
        (old_pretraining, "old XAU"), (new_pretraining, "new XAU"),
    ):
        path.write_text(marker)
    requested_dataset = new_dataset if changed == "dataset" else old_dataset
    requested_pretraining = (
        new_pretraining if changed == "pretraining" else old_pretraining
    )
    root = tmp_path / "existing-training"
    root.mkdir()
    complete_path = root / "complete_training_manifest.json"
    complete_path.write_text("{}")
    old = {
        "dataset_manifest": {"path": str(old_dataset), "sha256": _sha(old_dataset)},
        "pretraining_manifest": {
            "path": str(old_pretraining), "sha256": _sha(old_pretraining),
        },
    }
    monkeypatch.setattr(module, "load_phase_dataset", lambda *a, **k: _phase(requested_dataset, "train"))
    monkeypatch.setattr(module, "verify_pretraining_manifest", lambda *a, **k: {})
    monkeypatch.setattr(module, "verify_complete_training_manifest", lambda *a, **k: old)

    with pytest.raises(ValueError, match="different request"):
        module.train_all(
            requested_dataset, requested_pretraining, root, workers=1,
        )


@pytest.mark.parametrize(
    "changed", ["phase", "evaluation_dataset", "training_manifest", "unlock_receipt"],
)
def test_existing_registered_evaluation_rejects_a_different_current_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    import kx_simple_hmadrl.volume_v6_orchestration as module

    old_training = tmp_path / "old-training.json"
    new_training = tmp_path / "new-training.json"
    training_data = tmp_path / "training-data.json"
    old_evaluation = tmp_path / "old-evaluation-data.json"
    new_evaluation = tmp_path / "new-evaluation-data.json"
    old_unlock = tmp_path / "old-unlock.json"
    new_unlock = tmp_path / "new-unlock.json"
    for path in (
        old_training, new_training, training_data, old_evaluation,
        new_evaluation, old_unlock, new_unlock,
    ):
        path.write_text(path.name)

    phase = "test" if changed in {"phase", "unlock_receipt"} else "validation"
    requested_training = new_training if changed == "training_manifest" else old_training
    requested_evaluation = (
        new_evaluation if changed == "evaluation_dataset" else old_evaluation
    )
    requested_unlock = new_unlock if phase == "test" else None
    existing_phase = "validation" if changed == "phase" else phase
    existing_training = old_training
    existing_evaluation = old_evaluation
    existing_unlock = (
        {"path": str(old_unlock.resolve()), "sha256": _sha(old_unlock)}
        if existing_phase == "test" else None
    )

    destination = tmp_path / "existing-evaluation"
    destination.mkdir()
    (destination / "manifest.json").write_text("{}")
    complete = {
        "dataset_manifest": {
            "path": str(training_data.resolve()), "sha256": _sha(training_data),
        },
        "checkpoints": [],
    }
    existing = {
        "phase": existing_phase,
        "training_manifest": {
            "path": str(existing_training.resolve()),
            "sha256": _sha(existing_training),
        },
        "evaluation_dataset_manifest": {
            "path": str(existing_evaluation.resolve()),
            "sha256": _sha(existing_evaluation),
        },
        "unlock_receipt": existing_unlock,
    }

    monkeypatch.setattr(module, "verify_complete_training_manifest", lambda *a, **k: complete)

    def fake_load(path: Path, *, phase: str, **kwargs: object) -> PhaseDataset:
        return _phase(training_data if phase == "train" else requested_evaluation, phase)

    monkeypatch.setattr(module, "load_phase_dataset", fake_load)
    monkeypatch.setattr(module, "verify_evaluation_manifest", lambda *a, **k: existing)
    if phase == "test":
        monkeypatch.setattr(module, "_assert_test_date", lambda: None)
        monkeypatch.setattr(module, "assert_test_unlocked", lambda *a, **k: {})
        monkeypatch.setattr(module, "_assert_test_chain_bound", lambda *a, **k: None)
        monkeypatch.setattr(
            module, "_claim_test_output", lambda *a, **k: tmp_path / "claim.json",
        )

    with pytest.raises(ValueError, match="different request"):
        module.evaluate_registered_period(
            requested_training, requested_evaluation, destination,
            phase=phase, unlock_receipt=requested_unlock,
        )


def test_hierarchy_progress_identity_binds_exact_xau_path_and_hash(tmp_path: Path) -> None:
    from kx_simple_hmadrl.volume_v6_pipeline import _progress_identity

    raw = tmp_path / "2025-01-02.npz"
    side = tmp_path / "2025-01-02-sidecar.npz"
    first = tmp_path / "xau-a.pt"
    second = tmp_path / "xau-b.pt"
    raw.write_bytes(b"raw"); side.write_bytes(b"side")
    first.write_bytes(b"same weights"); second.write_bytes(b"same weights")
    warmstart = {"path": str(first.resolve()), "sha256": _sha(first)}
    common = (1103, "rich_volume", "train", ((raw, side),), "protocol", None)

    identity = _progress_identity(*common, warmstart)
    assert identity["xau_warmstart"] == warmstart
    assert _progress_identity(
        *common, {"path": str(second.resolve()), "sha256": _sha(second)},
    ) != identity
    assert _progress_identity(
        *common, {"path": str(first.resolve()), "sha256": "0" * 64},
    ) != identity


def test_upper_counterfactual_label_audit_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kx_simple_hmadrl.volume_v6_pipeline as module

    class Labeler:
        def __init__(self) -> None:
            self.calls = 0

        def label(self, start: int, end: int) -> SimpleNamespace:
            self.calls += 1
            valid = np.asarray(
                [True, True, True] if self.calls == 1 else [True, False, True],
                dtype=bool,
            )
            return SimpleNamespace(
                valid_mask=valid,
                rewards_cents=np.asarray([-10, 0, 10], dtype=np.int64),
            )

    class Observation:
        def arrays(self, arm: str) -> dict[str, np.ndarray]:
            return {"state": np.asarray([1.0], dtype=np.float32)}

    labeler = Labeler()
    monkeypatch.setattr(module, "BiasRewardLabeler", lambda *a, **k: labeler)
    monkeypatch.setattr(module, "_valid_feature_indexes", lambda *a, **k: np.arange(3))
    monkeypatch.setattr(module, "_raw_index", lambda raw, clock: int(clock))
    monkeypatch.setattr(module, "top_observation_at", lambda *a, **k: Observation())
    session = SimpleNamespace(
        top_clock_ns=np.asarray(
            [0, module.TOP_LIFETIME_NS, 2 * module.TOP_LIFETIME_NS], dtype=np.int64,
        ),
    )
    audit: dict[str, int] = {}

    examples = module._top_examples(
        {}, session, object(), "rich_volume", label_audit=audit,
    )

    assert len(examples) == 1
    assert audit == {
        "counterfactual_labels_attempted": 2,
        "counterfactual_labels_kept": 1,
        "counterfactual_labels_dropped": 1,
    }
    assert audit["counterfactual_labels_attempted"] == (
        audit["counterfactual_labels_kept"] + audit["counterfactual_labels_dropped"]
    )
