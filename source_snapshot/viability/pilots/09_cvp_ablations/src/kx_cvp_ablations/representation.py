"""Shape-preserving, field-name-auditable ablation masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kx_simple_hmadrl.volume_v6_contract import (
    BASE_CONTEXT_WIDTH,
    BOTTOM_RICH_FIELDS,
    CONTEXT_FIELDS,
    L2_ENGINEERED_FIELDS,
    L2_LEVEL_FIELDS,
    MESSAGE_WIDTH,
    TOP_RICH_FIELDS,
)
from kx_simple_hmadrl.volume_v6_data import RegimeObservation, TraderObservation

from .contract import AblationArm, validate_arm


ORDINARY_VOLUME_FIELD = "log1p_executed_volume"
L2_GEOMETRY_LEVEL_FIELDS = (
    "bid_distance_from_mid_ticks",
    "ask_distance_from_mid_ticks",
)
L2_GEOMETRY_ENGINEERED_FIELDS = ("spread_ticks",)


def _indexes(fields: tuple[str, ...], kept: tuple[str, ...]) -> tuple[int, ...]:
    unknown = set(kept).difference(fields)
    if unknown:
        raise AssertionError(f"registered ablation field disappeared: {sorted(unknown)}")
    return tuple(fields.index(name) for name in kept)


TOP_ORDINARY_VOLUME_INDEX = TOP_RICH_FIELDS.index(ORDINARY_VOLUME_FIELD)
BOTTOM_ORDINARY_VOLUME_INDEX = BOTTOM_RICH_FIELDS.index(ORDINARY_VOLUME_FIELD)
L2_GEOMETRY_LEVEL_INDEXES = _indexes(L2_LEVEL_FIELDS, L2_GEOMETRY_LEVEL_FIELDS)
L2_GEOMETRY_ENGINEERED_INDEXES = _indexes(
    L2_ENGINEERED_FIELDS, L2_GEOMETRY_ENGINEERED_FIELDS,
)


@dataclass(frozen=True)
class RepresentationMask:
    arm: AblationArm
    keep_flow_profile: bool
    keep_depth_ofi: bool
    publisher_parent_arm: str | None
    message_mode: str

    @classmethod
    def for_arm(cls, value: str | AblationArm) -> "RepresentationMask":
        arm = validate_arm(value)
        if arm is AblationArm.FLOW_PROFILE_ONLY:
            return cls(arm, True, False, "rich_volume", "frozen_publisher")
        if arm is AblationArm.DEPTH_OFI_ONLY:
            return cls(arm, False, True, "base_ohlcv", "frozen_publisher")
        return cls(arm, True, True, None, "exact_zero")

    def publisher_arrays(self, observation: RegimeObservation) -> dict[str, np.ndarray]:
        if self.publisher_parent_arm is None:
            raise ValueError("no-publisher arm has no publisher observation")
        result = observation.arrays("rich_volume")
        if not self.keep_flow_profile:
            result["rich"][:, :] = 0.0
            result["rich"][:, TOP_ORDINARY_VOLUME_INDEX] = observation.rich[
                :, TOP_ORDINARY_VOLUME_INDEX
            ]
            result["context"][BASE_CONTEXT_WIDTH:] = 0.0
            result["context_mask"][BASE_CONTEXT_WIDTH:] = False
        return result

    def trader_arrays(self, observation: TraderObservation) -> dict[str, np.ndarray]:
        result = observation.arrays("rich_volume")
        if not self.keep_flow_profile:
            result["rich"][:, :] = 0.0
            result["rich"][:, BOTTOM_ORDINARY_VOLUME_INDEX] = observation.rich[
                :, BOTTOM_ORDINARY_VOLUME_INDEX
            ]
            result["context"][BASE_CONTEXT_WIDTH:] = 0.0
            result["context_mask"][BASE_CONTEXT_WIDTH:] = False
        if not self.keep_depth_ofi:
            keep_levels = set(L2_GEOMETRY_LEVEL_INDEXES)
            for index in range(len(L2_LEVEL_FIELDS)):
                if index not in keep_levels:
                    result["levels"][:, :, index] = 0.0
            keep_engineered = set(L2_GEOMETRY_ENGINEERED_INDEXES)
            for index in range(len(L2_ENGINEERED_FIELDS)):
                if index not in keep_engineered:
                    result["l2_engineered"][:, index] = 0.0
        if self.message_mode == "exact_zero":
            result["message"] = np.zeros(MESSAGE_WIDTH, dtype=np.float32)
        self.audit(result)
        return result

    def audit(self, arrays: Mapping[str, np.ndarray]) -> None:
        if self.message_mode == "exact_zero":
            message = np.asarray(arrays["message"])
            if message.dtype != np.float32 or message.shape != (MESSAGE_WIDTH,):
                raise AssertionError("no-publisher message must be float32 width five")
            if not np.array_equal(message, np.zeros(MESSAGE_WIDTH, np.float32)):
                raise AssertionError("no-publisher message is not exact zero")
        if not self.keep_flow_profile:
            rich = np.asarray(arrays["rich"])
            nonordinary = np.delete(rich, BOTTOM_ORDINARY_VOLUME_INDEX, axis=1)
            if np.any(nonordinary != 0.0):
                raise AssertionError("flow/profile-suppressed rich field is nonzero")
            context = np.asarray(arrays["context"])
            context_mask = np.asarray(arrays["context_mask"])
            if np.any(context[BASE_CONTEXT_WIDTH:] != 0.0) or np.any(
                context_mask[BASE_CONTEXT_WIDTH:]
            ):
                raise AssertionError("flow/profile context was not suppressed")
        if not self.keep_depth_ofi:
            levels = np.asarray(arrays["levels"])
            engineered = np.asarray(arrays["l2_engineered"])
            for index in set(range(len(L2_LEVEL_FIELDS))).difference(
                L2_GEOMETRY_LEVEL_INDEXES
            ):
                if np.any(levels[:, :, index] != 0.0):
                    raise AssertionError("depth/OFI level field was not suppressed")
            for index in set(range(len(L2_ENGINEERED_FIELDS))).difference(
                L2_GEOMETRY_ENGINEERED_INDEXES
            ):
                if np.any(engineered[:, index] != 0.0):
                    raise AssertionError("depth/OFI engineered field was not suppressed")

    def registry_record(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "keep_flow_profile": self.keep_flow_profile,
            "keep_depth_ofi": self.keep_depth_ofi,
            "publisher_parent_arm": self.publisher_parent_arm,
            "message_mode": self.message_mode,
            "ordinary_volume_field": ORDINARY_VOLUME_FIELD,
            "kept_l2_level_fields_when_suppressed": list(L2_GEOMETRY_LEVEL_FIELDS),
            "kept_l2_engineered_fields_when_suppressed": list(
                L2_GEOMETRY_ENGINEERED_FIELDS
            ),
            "suppressed_context_fields_when_flow_off": list(
                CONTEXT_FIELDS[BASE_CONTEXT_WIDTH:]
            ),
        }


__all__ = [
    "BOTTOM_ORDINARY_VOLUME_INDEX", "L2_GEOMETRY_ENGINEERED_FIELDS",
    "L2_GEOMETRY_ENGINEERED_INDEXES", "L2_GEOMETRY_LEVEL_FIELDS",
    "L2_GEOMETRY_LEVEL_INDEXES", "ORDINARY_VOLUME_FIELD", "RepresentationMask",
    "TOP_ORDINARY_VOLUME_INDEX",
]
