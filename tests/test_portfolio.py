from datetime import date, datetime, timezone

import pytest

from aegis.backtest.report import build_report
from aegis.backtest.risk import RiskEngine, RiskLimits
from aegis.data.cache import empty_bars
from aegis.portfolio import analyze_portfolio, series_from_report
from aegis.portfolio.models import StrategySeries


def _series(name: str, returns: list[float]) -> StrategySeries:
    days = [date(2024, 1, index + 2) for index in range(len(returns))]
    return StrategySeries(name=name, dates=days, returns=returns)


def test_portfolio_report_is_advisory_and_does_not_touch_the_risk_engine():
    engine = RiskEngine()
    before = engine.limits
    report = analyze_portfolio(
        [
            _series("slow", [0.01, 0.01, 0.00, 0.01, -0.01, 0.01]),
            _series("fast", [0.04, -0.03, 0.05, -0.04, 0.06, -0.02]),
        ],
        vol_lookback=3,
    )
    assert engine.limits == before
    assert engine.killed is False
    assert RiskLimits() == RiskLimits()
    assert report.advisory_only is True
    assert report.changes_risk_limits is False
    assert report.enforced_by == "RiskEngine"
    assert report.allocation.hard_limits_unchanged is True
    assert sum(report.allocation.weights.values()) == pytest.approx(1.0)
    assert report.allocation.weights["slow"] > report.allocation.weights["fast"]
    assert report.correlations.values[0][0] == pytest.approx(1.0)
    assert report.correlations.values[0][1] is not None
    assert report.concentration.herfindahl == pytest.approx(0.5)
    assert report.drawdowns.max_drawdown >= 0
    assert report.regimes.buckets
    assert {item.name for item in report.stress} == {
        "vol_spike",
        "correlation_one",
        "drop_largest_weight",
        "worst_overlapping_day",
    }


def test_position_cap_is_applied_when_it_is_feasible():
    report = analyze_portfolio(
        [
            _series("quiet", [0.01, 0.01, 0.01, 0.01]),
            _series("loud", [0.05, -0.04, 0.06, -0.05]),
        ],
        position_cap=0.6,
        vol_lookback=2,
    )
    assert report.allocation.cap_feasible is True
    assert max(report.allocation.weights.values()) <= 0.6 + 1e-9
    assert report.concentration.position_cap == pytest.approx(0.6)


def test_series_from_report_reads_daily_returns():
    start = datetime(2024, 1, 3, 20, 0, tzinfo=timezone.utc)
    later = datetime(2024, 1, 4, 20, 0, tzinfo=timezone.utc)
    performance = build_report(
        strategy="ema_crossover",
        symbols=("AAPL",),
        starting_equity=100_000,
        equity_curve=[(start, 101_000), (later, 99_000)],
        trades=[],
        bars=empty_bars(),
    )
    series = series_from_report(performance)
    assert series.name == "ema_crossover"
    assert series.returns[0] == pytest.approx(0.01)
    assert series.returns[1] == pytest.approx(99_000 / 101_000 - 1)
