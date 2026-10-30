"""Advisory portfolio analytics for the Risk & Portfolio Analyst.

:func:`analyze_portfolio` reports correlation, concentration, drawdowns,
regime exposure, stress paths, and an inverse-volatility weight suggestion.
It does not place orders and it does not change the risk engine.
"""

from aegis.portfolio.analytics import analyze_portfolio, series_from_report
from aegis.portfolio.models import PortfolioReport, StrategySeries

__all__ = [
    "PortfolioReport",
    "StrategySeries",
    "analyze_portfolio",
    "series_from_report",
]
