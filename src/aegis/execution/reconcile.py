"""Compare the local book with the broker. Mismatches are reported, not repaired."""

from __future__ import annotations

from aegis.execution.models import (
    OPEN_ORDER_STATUSES,
    OrderView,
    PositionView,
    ReconcileReport,
    is_open_status,
    iter_orders,
    status_text,
)
from aegis.execution.state import PaperState


def _position_qty(positions: list[PositionView]) -> dict[str, int]:
    quantities: dict[str, int] = {}
    for position in positions:
        if position.qty != 0:
            quantities[position.symbol] = position.qty
    return quantities


def _open_broker_orders(orders: list[OrderView]) -> dict[str, OrderView]:
    found: dict[str, OrderView] = {}
    for order in iter_orders(orders):
        if order.id and is_open_status(order.status):
            found[order.id] = order
    return found


def apply_fill_updates(
    state: PaperState,
    positions: list[PositionView],
    orders: list[OrderView],
) -> list[str]:
    """Record fills we can explain. Unexplained differences stay for :func:`diff_books`."""
    notes: list[str] = []
    broker_open = _open_broker_orders(orders)
    broker_qty = _position_qty(positions)
    entries = [order for order in state.orders.values() if order.get("role") == "entry"]
    for order in entries:
        if status_text(order.get("status")) not in OPEN_ORDER_STATUSES:
            continue
        if order["id"] in broker_open:
            continue
        symbol = str(order["symbol"])
        signed = int(order["qty"]) if status_text(order.get("side")) != "sell" else -int(order["qty"])
        if broker_qty.get(symbol, 0) == signed:
            order["status"] = "filled"
            state.positions[symbol] = {
                "qty": signed,
                "entry": order.get("entry"),
                "stop": order.get("stop"),
                "target": order.get("target"),
                "order_id": order["id"],
            }
            notes.append(f"entry filled {symbol}")
    for order in list(state.orders.values()):
        if order.get("role") not in {"stop", "target"}:
            continue
        if status_text(order.get("status")) not in OPEN_ORDER_STATUSES:
            continue
        if order["id"] in broker_open:
            continue
        symbol = str(order["symbol"])
        if broker_qty.get(symbol, 0) != 0:
            continue
        for sibling in state.orders.values():
            if sibling.get("symbol") == symbol and status_text(sibling.get("status")) in OPEN_ORDER_STATUSES:
                sibling["status"] = "closed"
        state.positions.pop(symbol, None)
        notes.append(f"position closed {symbol}")
    return notes


def diff_books(
    state: PaperState,
    positions: list[PositionView],
    orders: list[OrderView],
) -> ReconcileReport:
    problems: list[str] = []
    broker_qty = _position_qty(positions)
    internal_qty = {
        symbol: int(row["qty"])
        for symbol, row in state.positions.items()
        if int(row.get("qty") or 0) != 0
    }
    for symbol in sorted(set(broker_qty) | set(internal_qty)):
        internal = internal_qty.get(symbol)
        actual = broker_qty.get(symbol)
        if internal is None:
            problems.append(f"broker position {symbol} qty {actual} is not in internal state")
        elif actual is None:
            problems.append(f"internal position {symbol} qty {internal} is not at the broker")
        elif internal != actual:
            problems.append(f"position {symbol} qty internal {internal} broker {actual}")
    broker_open = _open_broker_orders(orders)
    internal_open = {
        order_id: row
        for order_id, row in state.orders.items()
        if status_text(row.get("status")) in OPEN_ORDER_STATUSES
    }
    for order_id in sorted(set(broker_open) - set(internal_open)):
        order = broker_open[order_id]
        problems.append(f"broker order {order_id} ({order.symbol}) is not in internal state")
    for order_id in sorted(set(internal_open) - set(broker_open)):
        row = internal_open[order_id]
        problems.append(f"internal order {order_id} ({row.get('symbol')}) is not open at the broker")
    return ReconcileReport(mismatch=bool(problems), problems=tuple(problems))


def sync_and_reconcile(
    state: PaperState,
    positions: list[PositionView],
    orders: list[OrderView],
) -> ReconcileReport:
    apply_fill_updates(state, positions, orders)
    return diff_books(state, positions, orders)
