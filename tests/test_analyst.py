"""Analyst report on synthetic results. No strategy class and no API key."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from typer.testing import CliRunner

from aegis.analyst import (
    analysis_from_dict,
    analysis_to_dict,
    build_analyst_report,
    demo_analysis_input,
    load_analysis_input,
)
from aegis.analyst.costs import fill_cost, round_trip_cost
from aegis.analyst.input import AnalysisInput, FoldInput
from aegis.backtest.report import TradeRecord
from aegis.cli import app
from aegis.errors import AegisError, HoldoutViolation
from aegis.holdout import DateRange

NY = ZoneInfo("America/New_York")
HOLD = DateRange(date(2026, 7, 1), date(2026, 10, 31))
RUNNER = CliRunner()


def _ts(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY).astimezone(timezone.utc)


def _trade(day: date, pnl: float, r_multiple: float | None = 1.0, quantity: int = 10) -> TradeRecord:
    return TradeRecord(
        symbol="AAPL",
        entry_ts=_ts(day, 10, 0),
        exit_ts=_ts(day, 15, 0),
        entry_price=100.0,
        exit_price=101.0,
        quantity=quantity,
        stop_price=98.0,
        target_price=105.0,
        pnl=pnl,
        r_multiple=r_multiple,
        exit_reason="target",
    )


def _bars(day: date, closes: list[float]) -> pl.DataFrame:
    rows = []
    start = datetime(day.year, day.month, day.day, 14, 30, tzinfo=timezone.utc)
    for offset, close in enumerate(closes):
        rows.append({"symbol": "AAPL", "ts": start + timedelta(minutes=offset), "close": close})
    return pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))


def test_package_does_not_import_strategy_ml_or_portfolio_modules():
    root = Path(__file__).resolve().parents[1] / "src" / "aegis" / "analyst"
    text = "\n".join(path.read_text() for path in root.glob("*.py"))
    for forbidden in (
        "aegis.backtest.strategy",
        "aegis.strategies",
        "aegis.ml",
        "aegis.portfolio",
        "EmaCross",
        "xgboost",
    ):
        assert forbidden not in text


def test_report_covers_benchmark_walk_forward_costs_and_regimes():
    report = build_analyst_report(demo_analysis_input(), holdout=HOLD)
    markdown = report.to_markdown()
    payload = report.to_dict()
    assert report.strategy == "demo"
    assert payload["schema_version"] == 1
    assert payload["benchmark"]["benchmark_return"] is not None
    assert payload["benchmark"]["excess_return"] == pytest.approx(
        payload["performance"]["total_return"] - payload["benchmark"]["benchmark_return"]
    )
    assert [row["multiplier"] for row in payload["cost_sensitivity"]] == [1.0, 2.0, 3.0]
    endings = [row["ending_equity"] for row in payload["cost_sensitivity"]]
    assert endings[0] > endings[1] > endings[2]
    assert payload["cost_sensitivity"][0]["avg_r_multiple"] == payload["cost_sensitivity"][2]["avg_r_multiple"]
    assert payload["performance"]["ending_equity"] == pytest.approx(endings[0])
    summary = payload["walk_forward"]
    assert len(summary["folds"]) == 3
    assert summary["mean_return"] == pytest.approx((0.01 - 0.005 + 0.0) / 3)
    assert summary["median_return"] == pytest.approx(0.0)
    assert summary["min_return"] == pytest.approx(-0.005)
    assert summary["fraction_positive"] == pytest.approx(1 / 3)
    assert summary["compounded_return"] == pytest.approx(1.01 * 0.995 - 1)
    assert summary["total_trades"] == 2
    assert summary["mean_avg_r"] == pytest.approx((1.5 - 0.4) / 2)
    assert len(payload["regimes"]) >= 2
    for heading in (
        "## Performance",
        "## Benchmark",
        "## Walk-forward out of sample",
        "## Cost and slippage sensitivity",
        "## Regime breakdown",
    ):
        assert heading in markdown
    assert "1x" in markdown and "2x" in markdown and "3x" in markdown
    round_trip = json.loads(report.to_json())
    assert round_trip["strategy"] == "demo"


def test_cost_gap_equals_the_extra_round_trip():
    day = date(2024, 1, 3)
    cost = round_trip_cost(100, 100.0, 101.0, 0.005, 5.0)
    gross = 100.0
    trade = TradeRecord(
        symbol="AAPL",
        entry_ts=_ts(day, 10, 0),
        exit_ts=_ts(day, 15, 0),
        entry_price=100.0,
        exit_price=101.0,
        quantity=100,
        stop_price=98.0,
        target_price=105.0,
        pnl=gross - float(cost),
        r_multiple=0.5,
        exit_reason="target",
    )
    analysis = AnalysisInput(
        strategy="custom_alpha",
        starting_equity=100_000.0,
        trades=(trade,),
        equity_curve=((_ts(day, 16, 0), 100_000.0 + trade.pnl),),
        symbols=("AAPL",),
    )
    report = build_analyst_report(analysis, holdout=HOLD)
    equities = [scenario.ending_equity for scenario in report.cost_sensitivity]
    assert equities[0] - equities[1] == pytest.approx(float(cost), abs=0.02)
    assert equities[1] - equities[2] == pytest.approx(float(cost), abs=0.02)
    assert report.performance["strategy"] == "custom_alpha"
    assert report.benchmark.benchmark_return is None
    assert report.regimes == ()


def test_trade_list_without_a_curve_still_reports_pnl():
    day = date(2024, 1, 3)
    cost = round_trip_cost(100, 100.0, 101.0, 0.005, 5.0)
    trade = TradeRecord(
        symbol="AAPL",
        entry_ts=_ts(day, 10, 0),
        exit_ts=_ts(day, 15, 0),
        entry_price=100.0,
        exit_price=101.0,
        quantity=100,
        stop_price=98.0,
        target_price=105.0,
        pnl=100.0 - float(cost),
        r_multiple=0.5,
        exit_reason="target",
    )
    report = build_analyst_report(
        AnalysisInput(strategy="custom_alpha", starting_equity=100_000.0, trades=(trade,)),
        holdout=HOLD,
    )
    assert report.performance["ending_equity"] == pytest.approx(100_000.0 + trade.pnl, abs=0.02)
    assert report.performance["trade_count"] == 1


def test_fill_cost_matches_the_backtest_fee_model():
    from decimal import Decimal

    from nautilus_trader.model.currencies import USD

    from aegis.backtest.costs import EquityCommissionSlippage

    model = EquityCommissionSlippage(0.005, 5)

    class _Qty:
        def as_decimal(self):
            return Decimal("100")

    class _Px:
        def as_decimal(self):
            return Decimal("101")

    class _Instrument:
        quote_currency = USD

    charged = model.get_commission(None, _Qty(), _Px(), _Instrument())
    assert float(fill_cost(100, 101.0, 0.005, 5.0)) == pytest.approx(charged.as_double())


def test_benchmark_is_equal_weight_close_to_close():
    day = date(2024, 1, 3)
    bars = _bars(day, [100.0, 110.0])
    analysis = AnalysisInput(
        strategy="custom_alpha",
        starting_equity=100_000.0,
        equity_curve=((_ts(day, 16, 0), 101_000.0),),
        symbols=("AAPL",),
        bars=bars,
    )
    report = build_analyst_report(analysis, holdout=HOLD)
    assert report.benchmark.benchmark_return == pytest.approx(0.10)
    assert report.benchmark.strategy_return == pytest.approx(0.01)
    assert report.benchmark.excess_return == pytest.approx(-0.09)


def test_regimes_split_trend_and_volatility():
    up = date(2024, 1, 3)
    flat = date(2024, 1, 4)
    up_bars = _bars(up, [100.0 + 0.05 * step for step in range(20)])
    chop = [100.0 + (2.0 if step % 2 else 0.0) for step in range(20)]
    chop[-1] = 100.0
    bars = pl.concat([up_bars, _bars(flat, chop)])
    analysis = AnalysisInput(
        strategy="custom_alpha",
        starting_equity=100_000.0,
        trades=(_trade(up, pnl=1000.0),),
        equity_curve=((_ts(up, 16, 0), 101_000.0), (_ts(flat, 16, 0), 101_000.0)),
        symbols=("AAPL",),
        bars=bars,
    )
    report = build_analyst_report(analysis, holdout=HOLD)
    names = {regime.regime for regime in report.regimes}
    assert any(name.startswith("trend_up") for name in names)
    assert any(name.startswith("range") for name in names)
    assert any(name.endswith("vol_high") for name in names)
    assert any(name.endswith("vol_low") for name in names)
    trend = next(regime for regime in report.regimes if regime.regime.startswith("trend_up"))
    assert trend.sum_daily_return == pytest.approx(0.01)
    assert trend.trade_pnl == pytest.approx(1000.0)
    assert trend.trade_count == 1
    assert sum(regime.sum_daily_return for regime in report.regimes) == pytest.approx(0.01)


def test_walk_forward_summary_math():
    folds = (
        _fold(0, 110_000.0, (2.0,)),
        _fold(1, 98_000.0, (1.0, 1.0, 1.0)),
        _fold(2, 100_000.0, ()),
    )
    analysis = AnalysisInput(strategy="custom_alpha", starting_equity=50_000.0, folds=folds)
    summary = build_analyst_report(analysis, holdout=HOLD).walk_forward
    assert summary.mean_return == pytest.approx((0.10 - 0.02 + 0.0) / 3)
    assert summary.median_return == pytest.approx(0.0)
    assert summary.min_return == pytest.approx(-0.02)
    assert summary.max_return == pytest.approx(0.10)
    assert summary.fraction_positive == pytest.approx(1 / 3)
    assert summary.compounded_return == pytest.approx(1.10 * 0.98 - 1)
    assert summary.total_trades == 4
    assert summary.mean_avg_r == pytest.approx((2.0 + 1.0 + 1.0 + 1.0) / 4)


def test_holdout_dates_in_the_curve_or_a_fold_are_refused():
    leaked = AnalysisInput(
        strategy="custom_alpha",
        starting_equity=100_000.0,
        equity_curve=((_ts(date(2026, 7, 15), 16, 0), 100_000.0),),
    )
    with pytest.raises(HoldoutViolation):
        build_analyst_report(leaked, holdout=HOLD)
    folded = AnalysisInput(
        strategy="custom_alpha",
        starting_equity=100_000.0,
        equity_curve=((_ts(date(2024, 1, 3), 16, 0), 100_000.0),),
        folds=(
            FoldInput(
                fold_index=0,
                train_start=date(2024, 1, 2),
                train_end=date(2024, 1, 5),
                test_start=date(2026, 7, 1),
                test_end=date(2026, 7, 3),
                starting_equity=100_000.0,
            ),
        ),
    )
    with pytest.raises(HoldoutViolation):
        build_analyst_report(folded, holdout=HOLD)


def test_input_round_trip_and_bad_timestamps():
    analysis = demo_analysis_input()
    restored = analysis_from_dict(analysis_to_dict(analysis), analysis.bars)
    assert restored.strategy == analysis.strategy
    assert len(restored.trades) == len(analysis.trades)
    assert len(restored.folds) == 3
    original = build_analyst_report(analysis, holdout=HOLD)
    again = build_analyst_report(restored, holdout=HOLD)
    assert again.performance["total_return"] == pytest.approx(original.performance["total_return"])
    with pytest.raises(AegisError):
        analysis_from_dict(
            {
                "strategy": "custom_alpha",
                "starting_equity": 1000,
                "trades": [
                    {
                        "symbol": "AAPL",
                        "entry_ts": "2024-01-03T15:00:00",
                        "exit_ts": "2024-01-03T20:00:00+00:00",
                        "entry_price": 1,
                        "exit_price": 1,
                        "quantity": 1,
                        "stop_price": 1,
                        "target_price": 1,
                        "pnl": 0,
                    }
                ],
            }
        )


def test_cli_demo_and_report_file(tmp_path: Path):
    demo = RUNNER.invoke(app, ["analyze", "demo", "--json", "--out-dir", str(tmp_path / "demo")])
    assert demo.exit_code == 0, demo.stdout + demo.stderr
    payload = json.loads(_json_stdout(demo.stdout))
    assert payload["cost_sensitivity"][0]["multiplier"] == 1.0
    assert (tmp_path / "demo" / "report.md").exists()
    assert (tmp_path / "demo" / "report.json").exists()

    analysis = demo_analysis_input()
    source = tmp_path / "result.json"
    bars_path = tmp_path / "bars.parquet"
    source.write_text(json.dumps(analysis_to_dict(analysis)))
    analysis.bars.write_parquet(bars_path)
    reported = RUNNER.invoke(
        app,
        ["analyze", "report", "--input", str(source), "--bars", str(bars_path), "--out-dir", str(tmp_path / "file")],
    )
    assert reported.exit_code == 0, reported.stdout + reported.stderr
    assert "## Benchmark" in reported.stdout
    assert "Buy-and-hold return" in reported.stdout
    written = json.loads((tmp_path / "file" / "report.json").read_text())
    assert written["walk_forward"]["total_trades"] == 2

    source.write_text(
        json.dumps(
            {
                "strategy": "custom_alpha",
                "starting_equity": 100000,
                "equity_curve": [{"ts": "2026-07-02T20:00:00+00:00", "equity": 100000}],
                "trades": [],
            }
        )
    )
    refused = RUNNER.invoke(app, ["analyze", "report", "--input", str(source)])
    assert refused.exit_code == 1
    assert "holdout" in (refused.stderr + refused.stdout).lower()


def test_loaded_file_matches_the_parser(tmp_path: Path):
    analysis = demo_analysis_input()
    path = tmp_path / "result.json"
    path.write_text(json.dumps(analysis_to_dict(analysis)))
    loaded = load_analysis_input(path)
    assert loaded.strategy == "demo"
    assert loaded.bars is None


def _fold(index: int, ending: float, r_values: tuple[float, ...]) -> FoldInput:
    day = date(2024, 1, 3 + index * 7)
    trades = tuple(_trade(day, pnl=0.0, r_multiple=value) for value in r_values)
    return FoldInput(
        fold_index=index,
        train_start=day,
        train_end=day,
        test_start=day,
        test_end=day,
        starting_equity=100_000.0,
        equity_curve=((_ts(day, 16, 0), ending),),
        trades=trades,
        symbols=("AAPL",),
    )


def _json_stdout(text: str) -> str:
    start = text.find("{")
    assert start >= 0, text
    return text[start:]
