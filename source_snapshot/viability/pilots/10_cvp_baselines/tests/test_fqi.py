from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import kx_cvp_baselines.fqi as fqi
from kx_cvp_baselines.fqi import (
    ACTIONS,
    FLAT_STATE_WIDTH,
    FQIConfig,
    FQICorpus,
    FQICorpusCollector,
    LegalRandomPolicy,
    MESSAGE_SLICE,
    TRADER_ARRAY_SHAPES,
    action_conditioned_features,
    archive_corpus,
    fitted_q_targets,
    fit_fitted_q,
    flatten_trader_state,
    load_corpus,
)


def _observation() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, shape in TRADER_ARRAY_SHAPES.items():
        dtype = bool if name.endswith("_mask") else np.float32
        result[name] = np.zeros(shape, dtype=dtype)
    result["base"][0, 0] = 2.0
    return result


def _small_corpus() -> FQICorpus:
    states = np.zeros((3, FLAT_STATE_WIDTH), dtype=np.float32)
    next_states = states.copy()
    return FQICorpus(
        seed=1103,
        training_dates=("2025-01-02",),
        states=states,
        actions=np.asarray([-1, 0, 1]),
        rewards=np.asarray([1.0, 2.0, 3.0]),
        next_states=next_states,
        terminals=np.asarray([False, False, True]),
        legal=np.ones((3, 3), dtype=bool),
        next_legal=np.asarray([
            [False, True, True],
            [True, False, False],
            [False, False, False],
        ]),
        session_dates=np.asarray(["2025-01-02"] * 3),
        pass_indexes=np.ones(3, dtype=np.int8),
        decision_ordinals=np.arange(3),
    )


def test_flattened_full_state_has_frozen_order_and_exact_zero_message() -> None:
    observation = _observation()
    flattened = flatten_trader_state(observation)
    assert flattened.dtype == np.float32
    assert flattened.shape == (2338,) == (FLAT_STATE_WIDTH,)
    assert flattened[0] == 2.0
    assert np.array_equal(flattened[MESSAGE_SLICE], np.zeros(5, np.float32))

    nonzero = _observation()
    nonzero["message"][0] = 1.0
    with pytest.raises(ValueError, match="exact zero"):
        flatten_trader_state(nonzero)
    with pytest.raises(ValueError, match="simulator-only"):
        flatten_trader_state(_observation() | {"arrival_best_bid": np.asarray([1.0])})


def test_legal_random_policy_is_seed_specific_deterministic_and_never_illegal() -> None:
    left = LegalRandomPolicy(1103)
    right = LegalRandomPolicy(1103)
    masks = ([True, True, True], [True, True, False], [False, True, True]) * 20
    left_actions = [left.act(legal=mask) for mask in masks]
    right_actions = [right.act(legal=mask) for mask in masks]
    assert left_actions == right_actions
    for action, mask in zip(left_actions, masks, strict=True):
        assert bool(mask[ACTIONS.index(action)])
    assert left.audit_record()["decisions"] == len(masks)


def test_collector_rejects_evaluation_dates_before_accepting_a_row() -> None:
    with pytest.raises(PermissionError, match="evaluation access is forbidden"):
        FQICorpusCollector(seed=1103, training_dates=["2026-05-01"])


def test_action_conditioning_is_explicit_short_flat_long_one_hot() -> None:
    states = np.zeros((3, FLAT_STATE_WIDTH), dtype=np.float32)
    features = action_conditioned_features(states, [-1, 0, 1])
    assert features.shape == (3, FLAT_STATE_WIDTH + 3)
    assert np.array_equal(features[:, -3:], np.eye(3, dtype=np.float32))


class _ActionValueModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, -3:] @ np.asarray([100.0, 5.0, 1.0])


def test_fitted_q_target_masks_illegal_actions_and_zeroes_terminal_continuation() -> None:
    corpus = _small_corpus()
    assert fitted_q_targets(corpus, None).tolist() == [1.0, 2.0, 3.0]
    # Row 0 cannot bootstrap SHORT (100), so it uses FLAT (5).  Row 1 can
    # bootstrap only SHORT.  Row 2 is terminal and receives no continuation.
    assert fitted_q_targets(corpus, _ActionValueModel()).tolist() == [6.0, 102.0, 3.0]


def test_corpus_hash_archive_round_trip_and_registered_count_guard(tmp_path: Path) -> None:
    corpus = _small_corpus()
    with pytest.raises(ValueError, match="331 development sessions"):
        corpus.validate_registered()
    record = archive_corpus(corpus, tmp_path / "corpus.npz", require_registered=False)
    restored = load_corpus(
        tmp_path / "corpus.npz",
        expected_sha256=str(record["sha256"]),
        expected_canonical_sha256=str(record["canonical_sha256"]),
        require_registered=False,
    )
    assert restored.canonical_sha256() == corpus.canonical_sha256()
    assert restored.audit_record()["evaluation_accessed"] is False


class _FakeBooster:
    def __init__(self, serial: str) -> None:
        self.serial = serial

    def model_to_string(self, *, num_iteration: int = -1) -> str:
        assert num_iteration == -1
        return self.serial


class _FakeRegressor:
    calls = 0

    def __init__(self, parameters: dict[str, object]) -> None:
        self.parameters = parameters
        self.ordinal = -1
        self.booster_ = _FakeBooster("")

    def fit(self, features: np.ndarray, target: np.ndarray) -> "_FakeRegressor":
        type(self).calls += 1
        self.ordinal = type(self).calls
        digest = hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()
        self.booster_ = _FakeBooster(f"fake-lightgbm-{self.ordinal}-{digest}\n")
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, -3:] @ np.asarray([-1.0, 0.0, 1.0]) + self.ordinal


def test_ten_iterations_and_all_hashes_are_archived(monkeypatch, tmp_path: Path) -> None:
    _FakeRegressor.calls = 0

    def factory(config: FQIConfig, seed: int) -> _FakeRegressor:
        return _FakeRegressor(config.model_parameters(seed))

    monkeypatch.setattr(fqi, "_new_regressor", factory)
    result = fit_fitted_q(
        _small_corpus(),
        require_registered_corpus=False,
        archive_directory=tmp_path / "archive",
    )
    assert _FakeRegressor.calls == 10
    assert len(result.iterations) == 10
    assert result.iterations[0].previous_model_sha256 is None
    assert result.iterations[-1].model_sha256 == result.policy.model_sha256
    assert len({record.chain_sha256 for record in result.iterations}) == 10
    assert (tmp_path / "archive" / "corpus.npz").is_file()
    assert (tmp_path / "archive" / "iteration_10.txt").is_file()
    assert (tmp_path / "archive" / "model.txt").is_file()
    assert (tmp_path / "archive" / "manifest.json").is_file()
    assert result.audit_record()["evaluation_accessed"] is False


def test_registered_lightgbm_parameters_are_exact_and_immutable() -> None:
    parameters = FQIConfig().model_parameters(2207)
    assert parameters == {
        "objective": "huber",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 2207,
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    with pytest.raises(ValueError, match="frozen registry"):
        FQIConfig(trees=299).assert_registered()
