"""Phase 2 backtesting.

The EMA crossover baseline runs inside NautilusTrader. Position size, stops,
the entry window, the flatten time, the daily loss cutoff, and the kill switch
are decided by :class:`aegis.backtest.risk.RiskEngine`, which is plain Python
and is not an LLM tool.
"""
