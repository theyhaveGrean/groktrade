"""Alpaca minute bars on the free IEX feed. SIP is not used."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from aegis.config import Settings
from aegis.errors import AegisError
from aegis.execution.guards import assert_paper_execution_allowed
from aegis.execution.models import MinuteBar

IEX_FEED = DataFeed.IEX


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def minute_bar_from_mapping(
    *,
    symbol: str,
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 0.0,
) -> MinuteBar:
    return MinuteBar(
        symbol=symbol.upper(),
        ts=_aware(ts),
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(volume or 0.0),
    )


def minute_bar_from_alpaca(bar: Any) -> MinuteBar | None:
    symbol = getattr(bar, "symbol", None)
    ts = getattr(bar, "timestamp", None)
    if not symbol or ts is None:
        return None
    return minute_bar_from_mapping(
        symbol=str(symbol),
        ts=ts,
        open_=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=getattr(bar, "volume", 0.0) or 0.0,
    )


class AlpacaIexFeed:
    """Historical warmup and the live minute stream, both forced to IEX."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        stream_factory: Callable[..., Any] | None = None,
        history_factory: Callable[..., Any] | None = None,
    ) -> None:
        if IEX_FEED is not DataFeed.IEX:
            raise AegisError("Paper bars must use the IEX feed.")
        self._stream = (stream_factory or StockDataStream)(api_key, secret_key, feed=DataFeed.IEX)
        self._history = (history_factory or StockHistoricalDataClient)(api_key, secret_key)

    def historical_bars(self, symbols: list[str], start: datetime, end: datetime) -> list[MinuteBar]:
        if not symbols:
            return []
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
            adjustment=Adjustment.RAW,
        )
        result = self._history.get_stock_bars(request)
        bars: list[MinuteBar] = []
        data = getattr(result, "data", None) or {}
        for rows in data.values():
            for row in rows:
                converted = minute_bar_from_alpaca(row)
                if converted is not None:
                    bars.append(converted)
        bars.sort(key=lambda bar: (bar.ts, bar.symbol))
        return bars

    def stream(self, symbols: list[str], on_bar: Callable[[MinuteBar], None]) -> None:
        if not symbols:
            raise AegisError("No symbols to stream.")

        async def handler(bar: Any) -> None:
            converted = minute_bar_from_alpaca(bar)
            if converted is not None:
                on_bar(converted)

        self._stream.subscribe_bars(handler, *symbols)
        self._stream.run()


def connect_iex_feed(settings: Settings) -> AlpacaIexFeed:
    assert_paper_execution_allowed(settings)
    key, secret = settings.require_alpaca_credentials()
    return AlpacaIexFeed(key, secret)
