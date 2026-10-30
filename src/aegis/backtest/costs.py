"""Per-share commission plus deterministic adverse slippage.

Nautilus bar execution fills a market order submitted in ``on_bar`` against
the price prevailing at the next bar. Slippage here is not a random one-tick
coin flip. Every fill pays ``slippage_bps`` of notional on top of the
per-share commission, on both entry and exit. That is the conservative cost
model for liquid U.S. names (default 5 bps and $0.005 per share).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.model.objects import Money


class EquityCommissionSlippage(FeeModel):
    def __init__(self, commission_per_share: float, slippage_bps: float) -> None:
        if commission_per_share < 0 or slippage_bps < 0:
            raise ValueError("Costs cannot be negative.")
        self.commission_per_share = Decimal(str(commission_per_share))
        self.slippage_bps = Decimal(str(slippage_bps))

    def get_commission(self, order, fill_qty, fill_px, instrument) -> Money:
        quantity = fill_qty.as_decimal()
        price = fill_px.as_decimal()
        commission = quantity * self.commission_per_share
        slippage = quantity * price * self.slippage_bps / Decimal("10000")
        total = (commission + slippage).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if total < 0:
            total = Decimal("0.00")
        return Money(total, instrument.quote_currency)
