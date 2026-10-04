"""Frozen, auditable L2/L3 feature registry for RENT-HMADRL.

The registry is code rather than an ad-hoc list in a notebook so every product
binds exact formulas, availability clocks, reset behavior, and required masks.
Thresholds and normalizers are intentionally absent: they must be fitted inside
each chronological training fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .artifacts import sha256_json


FEATURE_REGISTRY_SCHEMA_VERSION = 1
RENT_FEATURE_REGISTRY_VERSION = "rent-mbo-v1"
WINDOWS_NS: tuple[tuple[str, int], ...] = (
    ("250ms", 250_000_000),
    ("1s", 1_000_000_000),
    ("5s", 5_000_000_000),
)
L3_WINDOWS_NS: tuple[tuple[str, int], ...] = WINDOWS_NS[1:]

FEATURE_METADATA_COLUMNS = (
    "feature_join_key",
    "publisher_id",
    "instrument_id",
    "raw_symbol",
    "continuity_segment_id",
    "book_epoch",
    "tape_last_ordinal",
    "event_time_ns",
    "available_time_ns",
    "feature_time_ns",
    "decision_time_ns",
    "source_tape_hash",
    "feature_registry_sha256",
)

L2_MASK_COLUMNS = (
    "mask_clean_clock",
    "mask_two_sided_book",
    "mask_warm_250ms",
    "mask_warm_1s",
    "mask_warm_5s",
)
L3_MASK_COLUMNS = L2_MASK_COLUMNS + (
    "mask_l3_lifecycle",
    "mask_l3_age_bid",
    "mask_l3_age_ask",
)


class FeatureRegistryError(ValueError):
    """Raised when a registry is ambiguous or internally inconsistent."""


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    column: str
    layer: str
    family: str
    hypothesis: str
    expected_sign: str
    horizon_ns: int
    formula: str
    units: str
    event_types: tuple[str, ...]
    side_convention: str
    boundary_behavior: str
    inputs: tuple[str, ...]
    missing_policy: str
    reset_policy: str
    warmup_ns: int
    intended_use: tuple[str, ...]
    correlated_family: str
    required_ablation: str
    live_feasible: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        if self.layer not in {"l2", "l3"}:
            raise FeatureRegistryError(f"unsupported layer {self.layer!r}")
        if not self.feature_id or not self.column or not self.formula:
            raise FeatureRegistryError("feature id, column, and formula cannot be empty")
        if self.horizon_ns < 0 or self.warmup_ns < 0:
            raise FeatureRegistryError("feature horizons cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureRegistry:
    version: str
    features: tuple[FeatureSpec, ...]
    schema_version: int = FEATURE_REGISTRY_SCHEMA_VERSION
    clock: str = "available_time_ns"
    window_boundary: str = "(t-window, t]"
    normalization_policy: str = "fit inside each chronological training fold only"

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_REGISTRY_SCHEMA_VERSION:
            raise FeatureRegistryError("unsupported feature-registry schema")
        if not self.version or not self.features:
            raise FeatureRegistryError("registry version and features are required")
        ids = [feature.feature_id for feature in self.features]
        columns = [feature.column for feature in self.features]
        if len(ids) != len(set(ids)):
            raise FeatureRegistryError("feature ids must be unique")
        if len(columns) != len(set(columns)):
            raise FeatureRegistryError("feature columns must be unique")

    @property
    def content_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "clock": self.clock,
            "window_boundary": self.window_boundary,
            "normalization_policy": self.normalization_policy,
            "features": [feature.to_dict() for feature in self.features],
            "metadata_columns": list(FEATURE_METADATA_COLUMNS),
            "l2_mask_columns": list(L2_MASK_COLUMNS),
            "l3_mask_columns": list(L3_MASK_COLUMNS),
        }

    def feature_columns(self, view: str) -> tuple[str, ...]:
        if view == "l2":
            return tuple(feature.column for feature in self.features if feature.layer == "l2")
        if view == "l3":
            return tuple(feature.column for feature in self.features)
        raise FeatureRegistryError(f"unsupported feature view {view!r}")

    def mask_columns(self, view: str) -> tuple[str, ...]:
        if view == "l2":
            return L2_MASK_COLUMNS
        if view == "l3":
            return L3_MASK_COLUMNS
        raise FeatureRegistryError(f"unsupported feature view {view!r}")


def _spec(
    column: str,
    *,
    layer: str,
    family: str,
    formula: str,
    units: str,
    inputs: Iterable[str],
    horizon_ns: int = 0,
    warmup_ns: int = 0,
    event_types: Iterable[str] = ("A", "C", "M", "F", "T", "R"),
    side_convention: str = "native bid/ask; direction alignment is downstream",
    missing_policy: str = "NaN plus explicit validity mask",
    hypothesis: str = "local book state conditions short-horizon execution value and risk",
    expected_sign: str = "context-dependent; never selected on final OOS",
    intended_use: Iterable[str] = ("context", "veto", "execution", "exit"),
) -> FeatureSpec:
    return FeatureSpec(
        feature_id=f"{RENT_FEATURE_REGISTRY_VERSION}:{column}",
        column=column,
        layer=layer,
        family=family,
        hypothesis=hypothesis,
        expected_sign=expected_sign,
        horizon_ns=horizon_ns,
        formula=formula,
        units=units,
        event_types=tuple(event_types),
        side_convention=side_convention,
        boundary_behavior="available-time events in (t-window, t]; same-time events included",
        inputs=tuple(inputs),
        missing_policy=missing_policy,
        reset_policy="clear rolling state at reset/continuity boundary; invalidate on bad clock",
        warmup_ns=warmup_ns,
        intended_use=tuple(intended_use),
        correlated_family=family,
        required_ablation=f"drop {family} family and compare on identical roots",
    )


def _build_specs() -> tuple[FeatureSpec, ...]:
    specs: list[FeatureSpec] = []
    book_inputs = ("FIFOBook.price_levels", "FIFOBook.level_depth")
    for column, formula, units in (
        ("best_bid_ticks", "highest live bid price", "ticks"),
        ("best_ask_ticks", "lowest live ask price", "ticks"),
        ("spread_ticks", "best_ask_ticks - best_bid_ticks", "ticks"),
        ("mid_ticks", "(best_bid_ticks + best_ask_ticks) / 2", "ticks"),
        (
            "microprice_ticks",
            "(best_ask*bid_depth_l1 + best_bid*ask_depth_l1) / (bid_depth_l1+ask_depth_l1)",
            "ticks",
        ),
        ("microprice_displacement_ticks", "microprice_ticks - mid_ticks", "ticks"),
    ):
        specs.append(
            _spec(column, layer="l2", family="book_shape", formula=formula, units=units, inputs=book_inputs)
        )
    for depth in (1, 3, 5):
        for side in ("bid", "ask"):
            specs.append(
                _spec(
                    f"{side}_depth_l{depth}",
                    layer="l2",
                    family="depth",
                    formula=f"sum displayed {side} quantity over best {depth} price levels",
                    units="contracts",
                    inputs=book_inputs,
                )
            )
        specs.append(
            _spec(
                f"imbalance_l{depth}",
                layer="l2",
                family="imbalance",
                formula=f"(bid_depth_l{depth}-ask_depth_l{depth})/(bid_depth_l{depth}+ask_depth_l{depth})",
                units="ratio",
                inputs=(f"bid_depth_l{depth}", f"ask_depth_l{depth}"),
            )
        )

    for label, window_ns in WINDOWS_NS:
        specs.append(
            _spec(
                f"ofi_qty_{label}",
                layer="l2",
                family="order_flow",
                formula="bid_add-bid_cancel-bid_execute-ask_add+ask_cancel+ask_execute",
                units="contracts",
                inputs=("causal displayed-flow events",),
                horizon_ns=window_ns,
                warmup_ns=window_ns,
            )
        )
        for kind in ("add", "cancel", "execute"):
            for side in ("bid", "ask"):
                specs.append(
                    _spec(
                        f"{kind}_{side}_qty_{label}",
                        layer="l2",
                        family=f"{kind}_flow",
                        formula=f"sum {kind} quantity on {side} in (t-{label},t]",
                        units="contracts",
                        inputs=("causal displayed-flow events",),
                        horizon_ns=window_ns,
                        warmup_ns=window_ns,
                    )
                )
            specs.append(
                _spec(
                    f"{kind}_intensity_per_s_{label}",
                    layer="l2",
                    family=f"{kind}_flow",
                    formula=f"count {kind} events in (t-{label},t] divided by {window_ns / 1e9:g}s",
                    units="events/second",
                    inputs=("causal displayed-flow events",),
                    horizon_ns=window_ns,
                    warmup_ns=window_ns,
                )
            )

    for side in ("bid", "ask"):
        for column, formula, units in (
            (f"{side}_order_count", f"number of live {side} order identities", "orders"),
            (f"{side}_size_hhi", "sum((order_size / total_side_size)^2)", "ratio"),
            (f"{side}_top_order_share", "max(order_size) / total_side_size", "ratio"),
            (f"{side}_left_censored_share", "mean(order.left_censored)", "ratio"),
            (
                f"{side}_front_concentration_l3",
                "sum(size of first FIFO order at each best-three level) / top-three displayed depth",
                "ratio",
            ),
        ):
            specs.append(
                _spec(
                    column,
                    layer="l3",
                    family="order_identity",
                    formula=formula,
                    units=units,
                    inputs=("FIFOBook orders and price-time queues",),
                    event_types=("A", "C", "M", "R"),
                )
            )
        for quantile in (25, 50, 75):
            specs.append(
                _spec(
                    f"{side}_age_q{quantile}_s",
                    layer="l3",
                    family="order_age",
                    formula=f"q{quantile} of max(0,t-birth_available_time) over non-left-censored live {side} orders",
                    units="seconds",
                    inputs=("FIFOBook.birth_available_ns", "available_time_ns"),
                    event_types=("A", "C", "M", "R"),
                )
            )

    for label, window_ns in L3_WINDOWS_NS:
        for side in ("bid", "ask"):
            for column, formula, units in (
                (
                    f"priority_reset_{side}_qty_{label}",
                    f"sum quantity losing FIFO priority on {side} in (t-{label},t]",
                    "contracts",
                ),
                (
                    f"churn_{side}_ratio_{label}",
                    f"({side}_add_qty+{side}_cancel_qty+priority_reset_qty)/max(live_{side}_top5_depth,1)",
                    "ratio",
                ),
                (
                    f"replenishment_{side}_qty_{label}",
                    f"sum same-completed-group {side} add quantity in groups that also contain {side} removal in (t-{label},t]; atomic co-occurrence proxy, not within-group ordering or best-level refill",
                    "contracts",
                ),
                (
                    f"depletion_{side}_qty_{label}",
                    f"sum executed plus nonexecution removal quantity across all displayed {side} prices in (t-{label},t]",
                    "contracts",
                ),
                (
                    f"execution_removal_{side}_share_{label}",
                    f"{side}_execute_qty / ({side}_execute_qty+{side}_cancel_qty); NaN at zero denominator",
                    "ratio",
                ),
            ):
                specs.append(
                    _spec(
                        column,
                        layer="l3",
                        family="lifecycle_flow",
                        formula=formula,
                        units=units,
                        inputs=("causal order lifecycle events", "FIFOBook"),
                        horizon_ns=window_ns,
                        warmup_ns=window_ns,
                    )
                )
    return tuple(specs)


RENT_FEATURE_REGISTRY = FeatureRegistry(
    version=RENT_FEATURE_REGISTRY_VERSION,
    features=_build_specs(),
)


def registry_from_dict(value: dict[str, Any]) -> FeatureRegistry:
    """Reconstruct and validate a serialized registry without trusting its hash."""

    if not isinstance(value, dict):
        raise FeatureRegistryError("serialized registry must be an object")
    raw_features = value.get("features")
    if not isinstance(raw_features, list):
        raise FeatureRegistryError("serialized registry needs a feature list")
    try:
        features = tuple(
            FeatureSpec(
                **{
                    **item,
                    "event_types": tuple(item["event_types"]),
                    "inputs": tuple(item["inputs"]),
                    "intended_use": tuple(item["intended_use"]),
                }
            )
            for item in raw_features
        )
        return FeatureRegistry(
            schema_version=int(value["schema_version"]),
            version=str(value["version"]),
            clock=str(value["clock"]),
            window_boundary=str(value["window_boundary"]),
            normalization_policy=str(value["normalization_policy"]),
            features=features,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FeatureRegistryError("invalid serialized feature registry") from error


__all__ = [
    "FEATURE_METADATA_COLUMNS",
    "FEATURE_REGISTRY_SCHEMA_VERSION",
    "L2_MASK_COLUMNS",
    "L3_MASK_COLUMNS",
    "L3_WINDOWS_NS",
    "RENT_FEATURE_REGISTRY",
    "RENT_FEATURE_REGISTRY_VERSION",
    "WINDOWS_NS",
    "FeatureRegistry",
    "FeatureRegistryError",
    "FeatureSpec",
    "registry_from_dict",
]
