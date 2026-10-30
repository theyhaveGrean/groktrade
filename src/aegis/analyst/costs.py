"""Rescale the commission and slippage already charged on a trade list.

The backtest fee model adds a per-share commission and a bps charge on each
fill. It does not move the fill price, so a higher multiple does not create a
different path. Net pnl at multiple ``m`` is the recorded 1x pnl minus
``(m - 1)`` times the round-trip cost. R-multiples stay gross.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

from aegis.backtest.report import TradeRecord


def fill_cost(quantity: int, price: float, commission_per_share: float, slippage_bps: float) -> Decimal:
    """One side of the venue fee, rounded to cents the same way the backtest does."""
    if quantity < 0 or price < 0 or commission_per_share < 0 or slippage_bps < 0:
        raise ValueError("Cost inputs cannot be negative.")
    qty = Decimal(quantity)
    commission = qty * Decimal(str(commission_per_share))
    slippage = qty * Decimal(str(price)) * Decimal(str(slippage_bps)) / Decimal("10000")
    total = (commission + slippage).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if total < 0:
        return Decimal("0.00")
    return total


def round_trip_cost(
    quantity: int,
    entry_price: float,
    exit_price: float,
    commission_per_share: float,
    slippage_bps: float,
) -> Decimal:
    """Entry fill plus exit fill."""
    return fill_cost(quantity, entry_price, commission_per_share, slippage_bps) + fill_cost(
        quantity, exit_price, commission_per_share, slippage_bps
    )


def scaled_pnl(
    trade: TradeRecord,
    multiplier: float,
    *,
    commission_per_share: float,
    slippage_bps: float,
) -> float:
    """Net pnl after charging ``multiplier`` times the 1x schedule."""
    if multiplier < 0:
        raise ValueError("Cost multiplier cannot be negative.")
    if multiplier == 1:
        return trade.pnl
    cost = round_trip_cost(
        trade.quantity,
        trade.entry_price,
        trade.exit_price,
        commission_per_share,
        slippage_bps,
    )
    extra = (Decimal(str(multiplier)) - Decimal(1)) * cost
    net = (Decimal(str(trade.pnl)) - extra).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(net)


def scale_trades(
    trades: tuple[TradeRecord, ...] | list[TradeRecord],
    multiplier: float,
    *,
    commission_per_share: float,
    slippage_bps: float,
) -> list[TradeRecord]:
    if multiplier == 1:
        return list(trades)
    scaled: list[TradeRecord] = []
    for trade in trades:
        scaled.append(
            TradeRecord(
                symbol=trade.symbol,
                entry_ts=trade.entry_ts,
                exit_ts=trade.exit_ts,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                stop_price=trade.stop_price,
                target_price=trade.target_price,
                pnl=scaled_pnl(
                    trade,
                    multiplier,
                    commission_per_share=commission_per_share,
                    slippage_bps=slippage_bps,
                ),
                r_multiple=trade.r_multiple,
                exit_reason=trade.exit_reason,
            )
        )
    return scaled


def scale_equity_curve(
    equity_curve: tuple[tuple[datetime, float], ...] | list[tuple[datetime, float]],
    trades: tuple[TradeRecord, ...] | list[TradeRecord],
    multiplier: float,
    *,
    commission_per_share: float,
    slippage_bps: float,
) -> list[tuple[datetime, float]]:
    """Subtract the extra cost of ``multiplier`` from each mark at or after the fill."""
    curve = sorted(equity_curve, key=lambda point: point[0])
    if multiplier == 1 or not curve:
        return curve
    extras: list[tuple[datetime, Decimal]] = []
    factor = Decimal(str(multiplier)) - Decimal(1)
    for trade in trades:
        extras.append(
            (
                trade.entry_ts,
                factor * fill_cost(trade.quantity, trade.entry_price, commission_per_share, slippage_bps),
            )
        )
        extras.append(
            (
                trade.exit_ts,
                factor * fill_cost(trade.quantity, trade.exit_price, commission_per_share, slippage_bps),
            )
        )
    extras.sort(key=lambda item: item[0])
    adjusted: list[tuple[datetime, float]] = []
    cumulative = Decimal("0")
    index = 0
    for ts, equity in curve:
        while index < len(extras) and extras[index][0] <= ts:
            cumulative += extras[index][1]
            index += 1
        adjusted.append((ts, float(Decimal(str(equity)) - cumulative)))
    if index < len(extras):
        while index < len(extras):
            cumulative += extras[index][1]
            last_ts = extras[index][0]
            index += 1
        adjusted.append((last_ts, float(Decimal(str(curve[-1][1])) - cumulative)))
    return adjusted
