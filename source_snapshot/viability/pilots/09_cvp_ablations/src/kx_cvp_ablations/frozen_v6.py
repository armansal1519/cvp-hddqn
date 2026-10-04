"""The only module allowed to adapt private frozen-V6 implementation details."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from kx_simple_hmadrl.volume_v6 import RegimeQNetwork
from kx_simple_hmadrl.volume_v6_contract import RegimeMessage
from kx_simple_hmadrl.volume_v6_data import (
    SessionFeatures,
    V6FeatureScaler,
    bottom_observation_at,
    build_session_features,
    load_npz_pair,
    top_observation_at,
)
from kx_simple_hmadrl.volume_v6_pipeline import (
    _latest_message,
    _load_member,
    _raw_index,
    _valid_feature_indexes,
    aligned_pairs,
)
from kx_simple_hmadrl.volume_v6_training import _tensor_batch, regime_forward
from kx_simple_hmadrl.volume_v6_training import load_model_bundle
from kx_simple_hmadrl.volume_v6_eval import frozen_protocol

from .artifacts import file_sha256
from .artifacts import read_json_object
from .representation import RepresentationMask


@dataclass(frozen=True)
class FrozenPublisher:
    model: RegimeQNetwork | None
    scaler: V6FeatureScaler
    manifest: dict[str, object]
    root: Path

    @property
    def seed(self) -> int:
        return int(self.manifest["seed"])

    @property
    def parent_arm(self) -> str:
        return str(self.manifest["arm"])

    @property
    def manifest_sha256(self) -> str:
        return file_sha256(self.root / "manifest.json")


def load_frozen_publisher(
    root: Path, *, expected_parent_arm: str, expected_seed: int,
) -> FrozenPublisher:
    models, scaler, manifest = _load_member(
        Path(root).resolve(), expected_arm=expected_parent_arm,
    )
    if int(manifest.get("seed", -1)) != int(expected_seed):
        raise ValueError("frozen publisher seed differs")
    if manifest.get("smoke_only") is not False:
        raise ValueError("registered ablation refuses a smoke parent publisher")
    models.regime.eval()
    for parameter in models.regime.parameters():
        parameter.requires_grad_(False)
    return FrozenPublisher(models.regime, scaler, manifest, Path(root).resolve())


def load_frozen_reference(
    root: Path, *, expected_parent_arm: str, expected_seed: int,
) -> FrozenPublisher:
    """Load only manifest/scaler lineage; never instantiate a publisher."""

    directory = Path(root).resolve()
    manifest = read_json_object(directory / "manifest.json")
    _, protocol_sha256 = frozen_protocol()
    if (
        manifest.get("artifact_type") != "V6_GC_TWO_LEVEL_TRAINING"
        or manifest.get("protocol_sha256") != protocol_sha256
        or manifest.get("arm") != expected_parent_arm
        or int(manifest.get("seed", -1)) != int(expected_seed)
        or manifest.get("smoke_only") is not False
    ):
        raise ValueError("frozen scaler-reference member identity differs")
    record = manifest.get("scaler")
    if not isinstance(record, dict):
        raise ValueError("frozen scaler-reference member lacks its scaler")
    scaler_path = directory / str(record.get("path", ""))
    if not scaler_path.is_file() or file_sha256(scaler_path) != record.get("sha256"):
        raise ValueError("frozen scaler-reference file hash differs")
    scaler = V6FeatureScaler.load_json(scaler_path)
    if scaler.canonical_sha256() != record.get("canonical_sha256"):
        raise ValueError("frozen scaler-reference canonical hash differs")
    return FrozenPublisher(None, scaler, manifest, directory)


def load_xau_price_bundle(root: Path, *, expected_seed: int, allow_smoke: bool = False):
    """Load a final XAU model without reopening its multi-year raw source."""

    directory = Path(root).resolve()
    manifest = read_json_object(directory / "manifest.json")
    if (
        manifest.get("artifact_type") != "V6_XAU_PRICE_PRETRAINING"
        or int(manifest.get("seed", -1)) != int(expected_seed)
        or (manifest.get("gc_transfer_eligible") is not True and not allow_smoke)
        or manifest.get("gc_outcomes_accessed") is not False
    ):
        raise ValueError("XAU price warm-start manifest identity differs")
    record = manifest.get("model_bundle")
    if not isinstance(record, dict):
        raise ValueError("XAU price warm-start lacks a model bundle")
    model_path = directory / str(record.get("path", ""))
    if not model_path.is_file() or file_sha256(model_path) != record.get("sha256"):
        raise ValueError("XAU price warm-start model hash differs")
    _, protocol_sha256 = frozen_protocol()
    models, metadata = load_model_bundle(model_path, protocol_id=protocol_sha256)
    if (
        metadata.get("stage") != "xau_price_pretraining"
        or int(metadata.get("seed", -1)) != int(expected_seed)
        or metadata.get("source_manifest_sha256") != manifest.get("source_manifest_sha256")
        or metadata.get("gc_outcomes_accessed") is not False
    ):
        raise ValueError("XAU price warm-start model metadata differs")
    return models, metadata, manifest


def publisher_messages(
    publisher: FrozenPublisher,
    session: SessionFeatures,
    representation: RepresentationMask,
) -> tuple[RegimeMessage, ...]:
    import torch

    if representation.publisher_parent_arm != publisher.parent_arm:
        raise ValueError("publisher parent arm differs from representation contract")
    if publisher.model is None:
        raise ValueError("scaler-only reference cannot publish a message")
    output: list[RegimeMessage] = []
    valid = set(int(index) for index in valid_feature_indexes(session, top=True))
    lifetime = 15 * 60 * 1_000_000_000
    for index in sorted(valid):
        if (
            index + 1 >= len(session.top_clock_ns)
            or index + 1 not in valid
            or int(session.top_clock_ns[index + 1]) - int(session.top_clock_ns[index])
            != lifetime
        ):
            continue
        observation = representation.publisher_arrays(
            top_observation_at(session, index, publisher.scaler)
        )
        with torch.no_grad():
            q = regime_forward(
                publisher.model, _tensor_batch([observation], torch.device("cpu")),
            )[0].detach().cpu().numpy()
        decision = int(session.top_clock_ns[index])
        output.append(RegimeMessage.from_q_values(q, decision, decision + lifetime))
    return tuple(output)


def latest_message(messages: Sequence[RegimeMessage], clock_ns: int) -> RegimeMessage:
    return _latest_message(messages, int(clock_ns))


def raw_index(raw: Mapping[str, np.ndarray], clock_ns: int) -> int:
    return _raw_index(raw, int(clock_ns))


def valid_feature_indexes(session: SessionFeatures, *, top: bool) -> np.ndarray:
    return _valid_feature_indexes(session, top=top)


def trader_q_values(model: object, observation: Mapping[str, np.ndarray]) -> np.ndarray:
    import torch
    from kx_simple_hmadrl.volume_v6_training import trader_forward

    with torch.no_grad():
        values = trader_forward(
            model, _tensor_batch([observation], torch.device("cpu")),
        )
    result = values[0].detach().cpu().numpy().copy()
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("frozen trader adapter received invalid Q values")
    return result


__all__ = [
    "FrozenPublisher", "SessionFeatures", "V6FeatureScaler", "aligned_pairs",
    "bottom_observation_at", "build_session_features", "latest_message",
    "load_frozen_publisher", "load_frozen_reference", "load_npz_pair",
    "load_xau_price_bundle",
    "publisher_messages", "raw_index",
    "top_observation_at", "trader_q_values", "valid_feature_indexes",
]
