"""Alpaca paper trading. The deterministic risk engine gates every order.

NautilusTrader 1.231 has no Alpaca adapter, so this package talks to Alpaca
through alpaca-py and reuses :class:`aegis.backtest.risk.RiskEngine` plus the
EMA cross from the backtest strategy. Live capital is not implemented.
"""

from aegis.execution.guards import PAPER_TRADING_BASE_URL, assert_paper_execution_allowed

__all__ = ["PAPER_TRADING_BASE_URL", "assert_paper_execution_allowed"]
