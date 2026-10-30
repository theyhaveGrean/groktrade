"""Signal plug-in interface.

A signal module sees one completed bar at a time and returns a
:class:`Signal`. It does not size, place, or cancel orders. The harness
owns those calls and may ignore stop or target hints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

NY = ZoneInfo("America/New_York")
SESSION_OPEN_MINUTE = 9 * 60 + 30


class SignalAction(str, Enum):
    NONE = "none"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"


class Signal(BaseModel):
    """What the harness is allowed to hear from a signal model."""

    model_config = ConfigDict(frozen=True)

    action: SignalAction = SignalAction.NONE
    stop_price: float | None = None
    target_price: float | None = None

    @classmethod
    def none(cls) -> Signal:
        return cls()

    @classmethod
    def enter_long(cls, stop_price: float | None = None, target_price: float | None = None) -> Signal:
        return cls(action=SignalAction.ENTER_LONG, stop_price=stop_price, target_price=target_price)

    @classmethod
    def enter_short(cls, stop_price: float | None = None, target_price: float | None = None) -> Signal:
        return cls(action=SignalAction.ENTER_SHORT, stop_price=stop_price, target_price=target_price)

    @classmethod
    def exit(cls) -> Signal:
        return cls(action=SignalAction.EXIT)


class IndicatorSpec(BaseModel):
    """A price indicator the harness registers with Nautilus before ``on_bar``."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["ema"]
    period: int = Field(ge=2)
    source: Literal["close"] = "close"


class BarView(BaseModel):
    """One completed bar plus indicator values known at that close."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr: float | None = None
    indicators: dict[str, float | None] = Field(default_factory=dict)

    def eastern(self) -> datetime:
        if self.ts.tzinfo is None:
            raise ValueError("Bar timestamps must be timezone-aware.")
        return self.ts.astimezone(NY)

    @property
    def session_date(self):
        return self.eastern().date()

    @property
    def minute_of_day(self) -> int:
        local = self.eastern()
        return local.hour * 60 + local.minute

    def in_opening_range(self, opening_minutes: int) -> bool:
        minute = self.minute_of_day
        return SESSION_OPEN_MINUTE <= minute < SESSION_OPEN_MINUTE + opening_minutes


class SignalModel(ABC):
    """Per-symbol state machine. Construct one instance per symbol."""

    def __init__(self, params) -> None:
        self.params = params

    def indicator_specs(self) -> list[IndicatorSpec]:
        return []

    def warmup_bars(self) -> int:
        """Bars the optimizer should ignore. The harness does not block on this."""
        return 0

    @abstractmethod
    def on_bar(self, bar: BarView) -> Signal:
        """Return the action for this close. Use only ``bar`` and prior calls."""


class EmaTracker:
    """Causal EMA. Seeds with the first value and applies ``2 / (period + 1)``.

    ``initialized`` flips once ``period`` values have been seen, matching
    Nautilus ``ExponentialMovingAverage``. Use this for a derived series
    such as a MACD signal line. Price EMAs that must match the EMA baseline
    are declared as :class:`IndicatorSpec` and updated by the harness.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError("EMA period must be positive.")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.value: float | None = None
        self.count = 0

    @property
    def initialized(self) -> bool:
        return self.count >= self.period and self.value is not None

    def update(self, price: float) -> float:
        self.count += 1
        if self.value is None:
            self.value = price
        else:
            self.value = self.alpha * price + (1.0 - self.alpha) * self.value
        return self.value
