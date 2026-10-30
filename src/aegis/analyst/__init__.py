"""Deterministic backtest report for the Backtesting Analyst.

The package reads a finished result: a trade list, an equity curve, and
optional bars. It does not import a strategy, place an order, or change the
risk engine. Agents call :func:`build_analyst_report` and the ``aegis analyze``
commands.
"""

from aegis.analyst.demo import demo_analysis_input
from aegis.analyst.input import analysis_from_dict, analysis_to_dict, load_analysis_input, load_bars
from aegis.analyst.report import AnalystReport, build_analyst_report, write_analyst_report

__all__ = [
    "AnalystReport",
    "analysis_from_dict",
    "analysis_to_dict",
    "build_analyst_report",
    "demo_analysis_input",
    "load_analysis_input",
    "load_bars",
    "write_analyst_report",
]
