"""Plain records returned by the Massive client before they hit Parquet."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class MinuteBar:
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    transactions: int | None = None


@dataclass(frozen=True)
class Split:
    symbol: str
    execution_date: date
    split_from: float
    split_to: float


@dataclass(frozen=True)
class Dividend:
    symbol: str
    ex_date: date
    cash_amount: float
    currency: str = "USD"
