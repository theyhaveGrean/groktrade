"""Alpaca paper broker. The trading client is always pointed at the paper host."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrderByIdRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from aegis.config import Settings
from aegis.errors import LiveTradingDisabled
from aegis.execution.guards import PAPER_TRADING_BASE_URL, assert_paper_execution_allowed
from aegis.execution.models import (
    AccountView,
    AssetView,
    BracketOrder,
    OrderView,
    PositionView,
    status_text,
)

log = logging.getLogger(__name__)


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)  # type: ignore[arg-type]


def _required_float(value: object, label: str) -> float:
    parsed = _float_or_none(value)
    if parsed is None:
        raise ValueError(f"Alpaca account is missing {label}.")
    return parsed


def _whole_qty(value: object) -> int:
    number = float(value)  # type: ignore[arg-type]
    rounded = int(round(number))
    if abs(number - rounded) > 1e-6:
        raise ValueError(f"Fractional quantity {value} is not supported.")
    return rounded


def _base_url_text(client: Any) -> str | None:
    observed = getattr(client, "_base_url", None)
    if observed is None:
        return None
    raw = observed.value if hasattr(observed, "value") else observed
    return str(raw).strip().rstrip("/")


class AlpacaPaperBroker:
    """Thin wrapper. Construction fails if the client is not aimed at paper."""

    def __init__(self, client: Any, *, base_url: str = PAPER_TRADING_BASE_URL) -> None:
        configured = base_url.strip().rstrip("/")
        observed = _base_url_text(client)
        if configured != PAPER_TRADING_BASE_URL or (observed is not None and observed != PAPER_TRADING_BASE_URL):
            raise LiveTradingDisabled(
                "Refusing a broker client whose base URL is not "
                f"{PAPER_TRADING_BASE_URL}."
            )
        self._client = client
        self.base_url = PAPER_TRADING_BASE_URL

    def account(self) -> AccountView:
        raw = self._client.get_account()
        equity = _required_float(getattr(raw, "equity", None), "equity")
        last_equity = _float_or_none(getattr(raw, "last_equity", None))
        buying_power = _float_or_none(getattr(raw, "buying_power", None))
        blocked = bool(getattr(raw, "trading_blocked", False) or getattr(raw, "account_blocked", False))
        return AccountView(
            equity=equity,
            last_equity=equity if last_equity is None else last_equity,
            buying_power=0.0 if buying_power is None else buying_power,
            trading_blocked=blocked,
        )

    def asset(self, symbol: str) -> AssetView:
        raw = self._client.get_asset(symbol)
        status = status_text(getattr(raw, "status", ""))
        return AssetView(
            symbol=str(getattr(raw, "symbol", symbol)),
            tradable=bool(getattr(raw, "tradable", False)),
            status=status or "inactive",
        )

    def positions(self) -> list[PositionView]:
        views: list[PositionView] = []
        for raw in self._client.get_all_positions():
            qty = _whole_qty(raw.qty)
            if status_text(getattr(raw, "side", "")) == "short":
                qty = -abs(qty)
            else:
                qty = abs(qty)
            entry = _float_or_none(getattr(raw, "avg_entry_price", None)) or 0.0
            views.append(PositionView(symbol=str(raw.symbol), qty=qty, avg_entry=entry))
        return views

    def open_orders(self) -> list[OrderView]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500)
        raw_orders = self._client.get_orders(filter=request)
        if len(raw_orders) >= 500:
            log.warning("Alpaca returned 500 open orders; reconciliation may be incomplete.")
        return [_order_view(order) for order in raw_orders]

    def submit_bracket(self, order: BracketOrder) -> OrderView:
        side = OrderSide.BUY if order.side == "long" else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=order.target_price),
            stop_loss=StopLossRequest(stop_price=order.stop_price),
            client_order_id=order.client_order_id,
            extended_hours=False,
        )
        submitted = self._client.submit_order(order_data=request)
        if not getattr(submitted, "legs", None):
            submitted = self._reload_nested(submitted)
        view = _order_view(submitted, role="entry")
        return view

    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()

    def flatten_all(self) -> None:
        self.cancel_all_orders()
        self._client.close_all_positions(cancel_orders=True)

    def flatten_symbol(self, symbol: str) -> None:
        for order in self.open_orders():
            if order.symbol == symbol and order.id:
                _cancel_quietly(self._client, order.id)
        try:
            self._client.close_position(symbol)
        except APIError as exc:
            if exc.status_code == 404:
                return
            raise

    def _reload_nested(self, submitted: Any) -> Any:
        order_id = getattr(submitted, "id", None)
        if order_id is None or not hasattr(self._client, "get_order_by_id"):
            return submitted
        try:
            return self._client.get_order_by_id(order_id, GetOrderByIdRequest(nested=True))
        except (APIError, TypeError) as exc:
            log.warning("bracket %s returned without legs (%s)", order_id, exc.__class__.__name__)
            return submitted


def connect_paper_broker(settings: Settings) -> AlpacaPaperBroker:
    """Build the only supported trading client: paper mode, paper host."""
    assert_paper_execution_allowed(settings)
    key, secret = settings.require_alpaca_credentials()
    client = TradingClient(
        api_key=key,
        secret_key=secret,
        paper=True,
        url_override=PAPER_TRADING_BASE_URL,
    )
    return AlpacaPaperBroker(client)


def _cancel_quietly(client: Any, order_id: str) -> None:
    try:
        client.cancel_order_by_id(order_id)
    except APIError as exc:
        if exc.status_code not in {404, 422}:
            raise


def _order_view(raw: Any, *, role: str = "unknown") -> OrderView:
    legs = tuple(_order_view(leg) for leg in (getattr(raw, "legs", None) or []))
    qty_raw = getattr(raw, "qty", None)
    qty = _whole_qty(qty_raw) if qty_raw not in (None, "") else 0
    side = status_text(getattr(raw, "side", "")) or "buy"
    limit_price = _float_or_none(getattr(raw, "limit_price", None))
    stop_price = _float_or_none(getattr(raw, "stop_price", None))
    if role == "unknown":
        if stop_price is not None and limit_price is None:
            role = "stop"
        elif limit_price is not None and stop_price is None:
            role = "target"
    order_id = getattr(raw, "id", "")
    if isinstance(order_id, UUID):
        order_id = str(order_id)
    return OrderView(
        id=str(order_id),
        symbol=str(getattr(raw, "symbol", "") or ""),
        qty=qty,
        side=side,
        status=status_text(getattr(raw, "status", "")) or "accepted",
        order_class=status_text(getattr(raw, "order_class", "")) or "simple",
        role=role,
        client_order_id=str(getattr(raw, "client_order_id", "") or ""),
        legs=legs,
        limit_price=limit_price,
        stop_price=stop_price,
    )
