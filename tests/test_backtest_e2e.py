from datetime import date

import pytest
from typer.testing import CliRunner

from aegis.backtest.costs import EquityCommissionSlippage
from aegis.backtest.engine import run_demo_backtest, run_ema_on_bars
from aegis.backtest.risk import RiskEngine, RiskLimits
from aegis.backtest.synthetic import synthetic_bars
from aegis.cli import app
from aegis.config import Settings
from aegis.errors import HoldoutViolation, LiveTradingDisabled
from aegis.holdout import DateRange
from nautilus_trader.model.currencies import USD


def _settings(**overrides) -> Settings:
    base = dict(_env_file=None)
    base.update(overrides)
    return Settings(**base)


def test_fee_model_charges_commission_and_five_bps():
    model = EquityCommissionSlippage(0.005, 5)

    class _Qty:
        def as_decimal(self):
            from decimal import Decimal

            return Decimal("10")

    class _Px:
        def as_decimal(self):
            from decimal import Decimal

            return Decimal("100")

    class _Instrument:
        quote_currency = USD

    commission = model.get_commission(None, _Qty(), _Px(), _Instrument())
    # 10 * 0.005 + 10 * 100 * 5 / 10000 = 0.55
    assert commission.as_double() == pytest.approx(0.55)


def test_synthetic_backtest_produces_the_standard_report():
    settings = _settings()
    report = run_demo_backtest(settings)
    text = report.render()
    assert report.trade_count >= 1
    assert report.strategy == "ema_crossover"
    assert report.symbols == ("AAPL",)
    assert report.starting_equity == 100_000
    assert report.max_drawdown >= 0
    assert report.sharpe is not None
    assert any(trade.pnl > 0 for trade in report.trades)
    for label in (
        "Total return",
        "Sharpe",
        "Max drawdown",
        "Win rate",
        "Avg R-multiple",
        "Trade count",
        "Buy-and-hold return",
        "Excess return",
    ):
        assert label in text
    payload = report.to_dict()
    assert payload["trade_count"] == report.trade_count


def test_kill_switch_and_missing_engine_block_entries():
    settings = _settings(risk_kill_switch=True)
    blocked = run_demo_backtest(settings)
    assert blocked.trade_count == 0
    report = run_ema_on_bars(
        synthetic_bars(),
        holdout=DateRange(date(2026, 7, 1), date(2026, 10, 31)),
        fast_period=5,
        slow_period=12,
        atr_period=10,
        install_risk=False,
    )
    assert report.trade_count == 0
    # A custom engine with zero slots also blocks, which is the same object the strategy must obey.
    capped = run_ema_on_bars(
        synthetic_bars(),
        holdout=DateRange(date(2026, 7, 1), date(2026, 10, 31)),
        fast_period=5,
        slow_period=12,
        atr_period=10,
        risk_engine=RiskEngine(RiskLimits(max_positions=0)),
    )
    assert capped.trade_count == 0


def test_backtest_refuses_holdout_dates_and_live_mode():
    with pytest.raises(HoldoutViolation):
        run_ema_on_bars(
            synthetic_bars(),
            holdout=DateRange(date(2024, 1, 1), date(2024, 1, 31)),
            fast_period=5,
            slow_period=12,
            atr_period=10,
        )
    settings = _settings(trading_mode="live", live_trading_enabled=True)
    with pytest.raises(LiveTradingDisabled):
        run_demo_backtest(settings)


def test_cli_demo_prints_a_report():
    runner = CliRunner()
    result = runner.invoke(app, ["backtest", "demo"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Trade count" in result.stdout
    assert "Sharpe" in result.stdout


def test_settings_hide_the_api_key():
    settings = _settings(massive_api_key="super-secret-key")
    assert "super-secret-key" not in repr(settings)
    assert settings.require_massive_api_key() == "super-secret-key"
