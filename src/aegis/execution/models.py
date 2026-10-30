"""Plain views of broker state. The rest of the package does not import Alpaca types."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class MinuteBar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class AccountView:
    equity: float
    last_equity: float
    buying_power: float
    trading_blocked: bool = False


@dataclass(frozen=True)
class AssetView:
    symbol: str
    tradable: bool
    status: str = "active"


@dataclass(frozen=True)
class PositionView:
    symbol: str
    qty: int
    avg_entry: float = 0.0


@dataclass(frozen=True)
class OrderView:
    id: str
    symbol: str
    qty: int
    side: str
    status: str
    order_class: str = "simple"
    role: str = "unknown"
    client_order_id: str = ""
    legs: tuple[OrderView, ...] = ()
    limit_price: float | None = None
    stop_price: float | None = None


@dataclass(frozen=True)
class BracketOrder:
    symbol: str
    side: str
    quantity: int
    stop_price: float
    target_price: float
    client_order_id: str
    entry_price: float


@dataclass(frozen=True)
class ReconcileReport:
    mismatch: bool
    problems: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return "; ".join(self.problems) if self.problems else "ok"


# Statuses Alpaca still treats as working. A filled or canceled order is not open.
OPEN_ORDER_STATUSES = frozenset(
    {
        "new",
        "accepted",
        "pending_new",
        "partially_filled",
        "held",
        "pending_cancel",
        "pending_replace",
        "pending_review",
        "accepted_for_bidding",
        "stopped",
        "calculated",
    }
)


def status_text(value: object) -> str:
    if value is None:
        return ""
    raw = value.value if hasattr(value, "value") else value
    return str(raw).strip().lower()


def is_open_status(value: object) -> bool:
    return status_text(value) in OPEN_ORDER_STATUSES


def iter_orders(orders: Iterable[OrderView]) -> Iterator[OrderView]:
    for order in orders:
        yield order
        yield from iter_orders(order.legs)
