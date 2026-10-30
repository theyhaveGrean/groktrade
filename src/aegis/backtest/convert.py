"""Turn cached Polars bars into Nautilus ``Bar`` objects.

Nautilus 1.231's ``BarDataWrangler`` reads a NumPy view of a pandas frame.
Pandas 3 can hand that view back read-only, so the converter builds ``Bar``
objects directly. The research venue for every U.S. name is ``XNAS``: one
simulated account, with the listing MIC kept on the universe membership.
"""

from __future__ import annotations

from datetime import timezone

import polars as pl
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Price, Quantity

RESEARCH_VENUE = "XNAS"


def equity_instrument(symbol: str, venue_name: str = RESEARCH_VENUE) -> Equity:
    venue = Venue(venue_name)
    return Equity(
        instrument_id=InstrumentId(Symbol(symbol), venue),
        raw_symbol=Symbol(symbol),
        currency=USD,
        price_precision=2,
        price_increment=Price.from_str("0.01"),
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


def bar_type_for(symbol: str, venue_name: str = RESEARCH_VENUE, minutes: int = 1) -> BarType:
    if minutes < 1:
        raise ValueError("Bar size must be at least one minute.")
    return BarType.from_str(f"{symbol}.{venue_name}-{minutes}-MINUTE-LAST-EXTERNAL")


def frame_to_bars(frame: pl.DataFrame, instrument: Equity, bar_type: BarType) -> list[Bar]:
    """Convert one symbol's frame. Rows must already be that symbol, sorted by ``ts``."""
    if frame.is_empty():
        return []
    bars: list[Bar] = []
    for row in frame.iter_rows(named=True):
        ts = row["ts"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        nanos = int(ts.timestamp()) * 1_000_000_000
        volume = max(int(round(float(row["volume"]))), 0)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=instrument.make_price(float(row["open"])),
                high=instrument.make_price(float(row["high"])),
                low=instrument.make_price(float(row["low"])),
                close=instrument.make_price(float(row["close"])),
                volume=instrument.make_qty(volume),
                ts_event=nanos,
                ts_init=nanos,
            )
        )
    return bars
