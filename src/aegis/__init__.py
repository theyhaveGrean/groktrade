"""Aegis: data, backtest, analyst reports, and Alpaca paper execution.

The deterministic risk engine gates backtests and paper orders. ``aegis.analyst``
turns a finished backtest into the Backtesting Analyst report. Later phases add
``agents``, ``ml``, and ``experiments`` without moving that boundary. Live
capital is not implemented.
"""

__version__ = "0.1.0"
