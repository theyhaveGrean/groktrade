"""Exit reasons follow the order that closed the position.

A walk-forward print labeled a winning WMT trade (entry 85.32, exit 91.95,
about +2.46R net of costs) as ``daily loss cutoff``. The fill was the 2.5R
target. A flatten attempted while the symbol was flat had left that stamp
for the next close.
"""

from __future__ import annotations

from datetime import date

import pytest

from aegis.backtest.risk import RiskEngine, RiskLimits
from aegis.backtest.synthetic import synthetic_bars
from aegis.holdout import DateRange
from aegis.strategies.ema import EmaCrossParams, EmaCrossover
from aegis.strategies.harness import resolve_exit_reason, run_signal_on_bars

HOLD = DateRange(date(2026, 7, 1), date(2026, 10, 31))


def test_wmt_target_fill_is_not_relabeled_daily_loss_cutoff():
    entry = 85.32
    exit_price = 91.95
    stop = 82.67
    gross_r = (exit_price - entry) / (entry - stop)
    assert gross_r == pytest.approx(2.5, rel=0.02)
    reason = resolve_exit_reason(
        stamped="daily loss cutoff",
        closing_tags=("TARGET",),
        exit_price=exit_price,
        stop=stop,
        target=exit_price,
    )
    assert reason == "target"


def test_stale_cutoff_stamp_does_not_label_the_next_winner():
    """Day one trips the cutoff while WMT is flat. Day two's target stays a target."""
    bars = synthetic_bars("WMT", sessions=(date(2024, 1, 3), date(2024, 1, 4)))
    report = run_signal_on_bars(
        bars,
        holdout=HOLD,
        signal_model=EmaCrossover,
        params=EmaCrossParams(fast_period=5, slow_period=12, atr_period=10, bar_timeframe="1min"),
        strategy_name="ema_crossover",
        atr_period=10,
        risk_engine=_FirstSessionCutoff(),
    )
    winners = [trade for trade in report.trades if trade.symbol == "WMT" and (trade.r_multiple or 0) > 2]
    assert winners, report.trades
    assert all(trade.exit_reason != "daily loss cutoff" for trade in winners)
    assert any(trade.exit_reason == "target" for trade in winners)


class _FirstSessionCutoff(RiskEngine):
    """Breach the daily-loss rule on the first session only, including while flat."""

    def __init__(self) -> None:
        super().__init__(RiskLimits())
        self._checks = 0

    def daily_loss_breached(self, snapshot) -> bool:
        self._checks += 1
        # One true result halts that session. Later bars short-circuit, so the
        # next session is the next call and must be allowed to trade.
        return self._checks == 20
