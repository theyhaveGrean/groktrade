from datetime import date, datetime, timezone

import pytest
from nautilus_trader.indicators import ExponentialMovingAverage

from aegis.backtest.engine import run_demo_backtest
from aegis.backtest.risk import RiskEngine, RiskLimits
from aegis.backtest.synthetic import synthetic_bars_for
from aegis.config import Settings
from aegis.holdout import DateRange
from aegis.strategies import get_strategy, list_strategies, parameter_grid
from aegis.strategies.harness import resolve_bracket
from aegis.strategies.policy import check_signal_package, check_signal_source
from aegis.strategies.signal import EmaTracker

HOLD_OUT = DateRange(date(2026, 7, 1), date(2026, 10, 31))
NAMES = ("ema_cross", "macd", "mean_reversion", "opening_range_breakout", "volatility_breakout")


def test_registry_lists_every_starter_and_exposes_search_metadata():
    found = tuple(plugin.name for plugin in list_strategies())
    assert found == NAMES
    for name in NAMES:
        plugin = get_strategy(name)
        points = parameter_grid(plugin.params_model)
        assert points
        assert all("bar_timeframe" in point for point in points)
        assert plugin.params_model().bar_timeframe in {"1min", "5min", "15min", "30min", "60min"}


def test_ema_harness_matches_the_pre_plugin_trade():
    report = run_demo_backtest(Settings(_env_file=None))
    assert report.strategy == "ema_crossover"
    assert len(report.equity_curve) == 780
    assert report.trade_count == 1
    trade = report.trades[0]
    assert trade.symbol == "AAPL"
    assert trade.entry_ts == datetime(2024, 1, 3, 15, 15, tzinfo=timezone.utc)
    assert trade.exit_ts == datetime(2024, 1, 3, 15, 24, tzinfo=timezone.utc)
    assert trade.entry_price == pytest.approx(97.4)
    assert trade.exit_price == pytest.approx(98.35)
    assert trade.quantity == 256
    assert trade.stop_price == pytest.approx(97.02)
    assert trade.target_price == pytest.approx(98.35)
    assert trade.pnl == pytest.approx(215.58)
    assert trade.r_multiple == pytest.approx(2.216077302631523)
    assert trade.exit_reason == "target"
    assert report.ending_equity == pytest.approx(100215.58)
    payload = report.to_dict()
    assert payload["schema_version"] == "1.1"
    assert payload["parameters"]["fast_period"] == 5
    assert payload["risk_limits"]["max_risk_per_trade"] == pytest.approx(0.01)
    assert payload["risk_engine_installed"] is True


@pytest.mark.parametrize("name", ["macd", "opening_range_breakout", "mean_reversion", "volatility_breakout"])
def test_plugins_trade_on_synthetic_bars_and_obey_the_kill_switch(name):
    settings = Settings(_env_file=None)
    report = run_demo_backtest(settings, strategy=name)
    assert report.trade_count >= 1
    assert report.strategy == name
    blocked = run_demo_backtest(Settings(_env_file=None, risk_kill_switch=True), strategy=name)
    assert blocked.trade_count == 0


def test_signal_package_passes_and_order_imports_are_rejected():
    result = check_signal_package()
    assert result.passed, result.violations
    assert len(result.files_checked) == 5
    leaked = check_signal_source(
        "import nautilus_trader\nfrom aegis.strategies.signal import Signal\n\ndef f(self):\n    self.submit_order(1)\n",
        filename="bad.py",
    )
    assert not leaked.passed
    messages = " ".join(item.message for item in leaked.violations)
    assert "nautilus_trader" in messages
    assert "submit_order" in messages


def test_hints_cannot_widen_risk():
    stop, target = resolve_bracket(
        side="long",
        entry=100.0,
        engine_stop=98.0,
        engine_target=105.0,
        hint_stop=None,
        hint_target=None,
        reward_to_risk=2.5,
    )
    assert (stop, target) == (98.0, 105.0)
    tighter, same_target = resolve_bracket(
        side="long",
        entry=100.0,
        engine_stop=98.0,
        engine_target=105.0,
        hint_stop=99.0,
        hint_target=101.0,
        reward_to_risk=2.5,
    )
    assert tighter == pytest.approx(99.0)
    assert same_target == pytest.approx(105.0)
    wider, _target = resolve_bracket(
        side="long",
        entry=100.0,
        engine_stop=98.0,
        engine_target=105.0,
        hint_stop=97.0,
        hint_target=None,
        reward_to_risk=2.5,
    )
    assert wider == pytest.approx(98.0)


def test_signal_ema_matches_nautilus_initialization():
    prices = [10, 11, 12, 11, 10, 13, 14, 12, 15, 16]
    nautilus = ExponentialMovingAverage(5)
    local = EmaTracker(5)
    for price in prices:
        nautilus.update_raw(price)
        local.update(price)
        assert local.initialized is nautilus.initialized
        assert local.value == pytest.approx(nautilus.value)


def test_resampled_timeframe_changes_the_bar_count():
    from aegis.backtest.engine import run_ema_on_bars
    from aegis.backtest.synthetic import synthetic_bars
    from aegis.strategies.resample import resample_bars

    bars = synthetic_bars()
    report = run_ema_on_bars(
        resample_bars(bars, "5min"),
        holdout=HOLD_OUT,
        fast_period=3,
        slow_period=8,
        atr_period=5,
        bar_minutes=5,
        risk_engine=RiskEngine(RiskLimits()),
    )
    assert len(report.equity_curve) == bars.height // 5
    assert report.bar_timeframe == "5min"


def test_optimizer_accepts_a_non_ema_plugin(tmp_path):
    from aegis.backtest.optimize import optimize_on_bars
    from aegis.backtest.synthetic import OPTIMIZE_DEMO_DAYS, synthetic_bars

    report = optimize_on_bars(
        synthetic_bars(sessions=OPTIMIZE_DEMO_DAYS),
        strategy="macd",
        start=date(2024, 1, 3),
        end=date(2024, 1, 15),
        holdout=HOLD_OUT,
        param_overrides={
            "fast_period": [5],
            "slow_period": [12],
            "signal_period": [4],
            "atr_period": [10],
            "atr_multiplier": [2.0],
            "bar_timeframe": ["1min"],
        },
        train_days=3,
        test_days=3,
        step_days=5,
        embargo_days=2,
        workers=1,
        ledger_path=tmp_path / "trials.parquet",
    )
    assert report.strategy == "macd"
    assert report.n_trials >= 1
    assert report.n_parameter_sets == 1


def test_cli_demo_accepts_strategy_name():
    from typer.testing import CliRunner

    from aegis.cli import app

    result = CliRunner().invoke(app, ["backtest", "demo", "--strategy", "opening_range_breakout"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Trade count" in result.stdout
    unknown = CliRunner().invoke(app, ["backtest", "demo", "--strategy", "not_a_strategy"])
    assert unknown.exit_code != 0


def test_opening_range_requires_a_multiple_of_the_bar():
    from aegis.strategies.opening_range import OpeningRangeParams

    OpeningRangeParams(opening_minutes=15, bar_timeframe="5min")
    with pytest.raises(ValueError):
        OpeningRangeParams(opening_minutes=10, bar_timeframe="15min")
    assert synthetic_bars_for("opening_range_breakout").height == 390
