from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class VerdictStatus(StrEnum):
    SYNTHETIC_SMOKE_ONLY = "SYNTHETIC_SMOKE_ONLY"
    VIABLE_FOR_HMADRL = "VIABLE_FOR_HMADRL"
    MECHANISM_NOT_SUPPORTED = "MECHANISM_NOT_SUPPORTED"
    INCONCLUSIVE_DATA = "INCONCLUSIVE_DATA"
    INCONCLUSIVE_POWER = "INCONCLUSIVE_POWER"
    INCONCLUSIVE_GC_LITE = "INCONCLUSIVE_GC_LITE"
    INCONCLUSIVE_HISTORICAL_REPLAY = "INCONCLUSIVE_HISTORICAL_REPLAY"
    TECHNICAL_BLOCK = "TECHNICAL_BLOCK"


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool | None
    observed: float | int | str | None = None
    threshold: float | int | str | None = None
    detail: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ViabilityVerdict:
    pilot: str
    status: VerdictStatus
    summary: str
    gates: tuple[GateResult, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    data_version: str = ""
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass(frozen=True, order=True)
class LifeKey:
    publisher_id: int
    raw_symbol: str
    continuity_segment_id: int
    book_epoch: int
    order_id: int
    life_instance: int
    priority_epoch: int


@dataclass(frozen=True)
class EventRecord:
    global_record_ordinal: int
    continuity_segment_id: int
    publisher_id: int
    instrument_id: int
    raw_symbol: str
    ts_event_ns: int
    ts_recv_ns: int
    action: str
    side: str
    price_ticks: int
    size: int
    order_id: int
    flags: int = 0
    is_last: bool = False
    is_snapshot: bool = False
    is_bad_ts_recv: bool = False

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EventRecord":
        def integer(name: str, default: int = 0) -> int:
            value = row.get(name, default)
            return default if value is None else int(value)

        return cls(
            global_record_ordinal=int(row["global_record_ordinal"]),
            continuity_segment_id=int(row["continuity_segment_id"]),
            publisher_id=int(row["publisher_id"]),
            instrument_id=int(row["instrument_id"]),
            raw_symbol=str(row["raw_symbol"]),
            ts_event_ns=int(row["ts_event_ns"]),
            ts_recv_ns=int(row["ts_recv_ns"]),
            action=str(row["action"]),
            side=str(row["side"]),
            price_ticks=integer("price_ticks"),
            size=integer("size"),
            order_id=integer("order_id"),
            flags=integer("flags"),
            is_last=bool(row.get("is_last", False)),
            is_snapshot=bool(row.get("is_snapshot", False)),
            is_bad_ts_recv=bool(row.get("is_bad_ts_recv", False)),
        )


@dataclass(frozen=True)
class BranchPayoff:
    opportunity_id: str
    action: str
    date: str
    net_usd: float
    net_ticks: float
    filled_quantity: int
    forced_quantity: int = 0
    valid: bool = True
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    time_ns: int
    signed_quantity: int
    price_ticks: int
    fee_usd: float
    liquidity: str
