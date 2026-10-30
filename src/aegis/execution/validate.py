"""Pre-trade gate. The risk engine decides first; broker checks cannot widen it."""

from __future__ import annotations

from datetime import datetime

from aegis.backtest.risk import PortfolioSnapshot, RiskDecision, RiskEngine
from aegis.execution.models import AssetView


def _deny(reason: str, *, flatten: bool = False) -> RiskDecision:
    return RiskDecision(approved=False, reason=reason, flatten=flatten)


class PreTradeValidator:
    def __init__(self, risk: RiskEngine) -> None:
        self.risk = risk

    def evaluate(
        self,
        *,
        symbol: str,
        ts: datetime,
        side: str,
        entry: float,
        atr: float | None,
        snapshot: PortfolioSnapshot,
        buying_power: float,
        trading_blocked: bool,
        asset: AssetView | None,
        mismatch: bool,
    ) -> RiskDecision:
        decision = self.risk.evaluate_entry(
            symbol=symbol,
            ts=ts,
            side=side,
            entry=entry,
            atr=atr,
            snapshot=snapshot,
        )
        if not decision.approved:
            return decision
        if mismatch:
            return _deny("broker reconciliation mismatch")
        if trading_blocked:
            return _deny("account trading is blocked")
        if asset is None or not asset.tradable:
            return _deny("symbol not tradable")
        if asset.status != "active":
            return _deny("symbol not active")
        if decision.stop_price is None or decision.target_price is None:
            return _deny("bracket missing stop or target")
        risk_per_share = abs(entry - decision.stop_price)
        reward = abs(decision.target_price - entry)
        limits = self.risk.limits
        if risk_per_share <= 0 or reward / risk_per_share < limits.reward_to_risk - 1e-9:
            return _deny("reward-to-risk below 2.5")
        if decision.quantity * risk_per_share > snapshot.equity * limits.max_risk_per_trade + 1e-6:
            return _deny("risk exceeds 1% of equity")
        notional = decision.quantity * entry
        if notional > snapshot.equity * limits.max_position_equity_fraction + 1e-6:
            return _deny("notional exceeds 25% of equity")
        if notional > buying_power + 1e-6:
            return _deny("insufficient buying power")
        return decision
