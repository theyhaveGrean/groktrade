from datetime import date

import polars as pl
import pytest

from aegis.backtest.deflated_sharpe import _norm_ppf, deflated_sharpe_ratio
from aegis.backtest.optimize import optimize_demo, optimize_on_bars, read_trials
from aegis.backtest.synthetic import synthetic_bars
from aegis.backtest.walkforward import walk_forward_splits
from aegis.config import Settings
from aegis.data.cache import empty_bars
from aegis.errors import HoldoutViolation
from aegis.holdout import DateRange
from aegis.cli import app

HOLD_OUT = DateRange(date(2026, 7, 1), date(2026, 10, 31))


def test_resample_aligns_five_minute_bars_to_the_open():
    from datetime import datetime, timezone

    from aegis.data.models import MinuteBar
    from aegis.data.cache import bars_to_frame
    from aegis.strategies.resample import resample_bars

    start = datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc)  # 09:30 ET
    bars = []
    for offset, close in enumerate([10.0, 11.0, 9.0, 10.5, 12.0, 8.0]):
        ts = start.replace(minute=30 + offset)
        bars.append(MinuteBar("AAPL", ts, close, close + 1, close - 1, close, 100, close, 1))
    frame = bars_to_frame(bars)
    out = resample_bars(frame, "5min")
    assert out.height == 2
    first = out.row(0, named=True)
    assert first["open"] == 10.0
    assert first["high"] == 13.0
    assert first["low"] == 8.0
    assert first["close"] == 12.0
    assert first["volume"] == 500
    assert resample_bars(frame, "1min").height == 6


def test_inverse_normal_and_deflated_sharpe_penalizes_more_trials():
    assert _norm_ppf(0.975) == pytest.approx(1.95996398454, abs=1e-6)
    returns = [0.01, 0.02, -0.005, 0.015, 0.008, -0.002, 0.012, 0.004]
    single = deflated_sharpe_ratio(returns, n_trials=1)
    many = deflated_sharpe_ratio(returns, n_trials=500)
    assert single is not None and many is not None
    assert 0.0 < many < single <= 1.0
    assert deflated_sharpe_ratio(returns[:2], n_trials=10) is None


def test_param_model_grid_drops_invalid_pairs_and_includes_timeframe():
    from aegis.strategies import get_strategy, parameter_grid

    model = get_strategy("ema_cross").params_model
    overrides = {
        "fast_period": [8, 12],
        "slow_period": [10, 30],
        "atr_multiplier": [2.0],
        "atr_period": [14],
        "bar_timeframe": ["1min", "5min"],
    }
    points = parameter_grid(model, overrides)
    pairs = {(point["fast_period"], point["slow_period"]) for point in points}
    assert (12, 10) not in pairs
    assert (8, 10) in pairs
    assert {point["bar_timeframe"] for point in points} == {"1min", "5min"}
    assert len(points) == 6
    first = parameter_grid(model, overrides, method="random", samples=3, seed=7)
    second = parameter_grid(model, overrides, method="random", samples=3, seed=7)
    assert first == second
    assert len(first) == 3
    with pytest.raises(ValueError, match="not one of"):
        parameter_grid(model, {"bar_timeframe": ["2min"]})


def test_anchored_folds_keep_the_training_start():
    rolling = walk_forward_splits(
        date(2024, 1, 1),
        date(2024, 6, 1),
        train_days=30,
        test_days=10,
        embargo_days=1,
        holdout=HOLD_OUT,
        mode="rolling",
    )
    anchored = walk_forward_splits(
        date(2024, 1, 1),
        date(2024, 6, 1),
        train_days=30,
        test_days=10,
        embargo_days=1,
        holdout=HOLD_OUT,
        mode="anchored",
    )
    assert anchored[0] == rolling[0]
    assert anchored[1].train_start == date(2024, 1, 1)
    assert anchored[1].train_end > anchored[0].train_end
    assert rolling[1].train_start > rolling[0].train_start


def test_overlapping_out_of_sample_windows_are_refused():
    with pytest.raises(ValueError, match="overlap"):
        optimize_on_bars(
            empty_bars(),
            strategy="ema_cross",
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            holdout=HOLD_OUT,
            param_overrides={"fast_period": [5], "slow_period": [12], "bar_timeframe": ["1min"]},
            train_days=10,
            test_days=10,
            step_days=5,
            embargo_days=0,
            workers=1,
        )


def test_optimize_refuses_holdout_before_any_trial(tmp_path):
    bars = synthetic_bars()
    with pytest.raises(HoldoutViolation):
        optimize_on_bars(
            bars,
            strategy="ema_cross",
            start=date(2024, 1, 3),
            end=date(2024, 1, 4),
            holdout=DateRange(date(2024, 1, 1), date(2024, 1, 31)),
            param_overrides={"fast_period": [5], "slow_period": [12], "bar_timeframe": ["1min"]},
            train_days=1,
            test_days=1,
            embargo_days=0,
            workers=1,
            ledger_path=tmp_path / "trials.parquet",
        )
    assert not (tmp_path / "trials.parquet").exists()


def test_synthetic_walk_forward_ledger_and_parallel_workers(tmp_path):
    ledger = tmp_path / "trials.parquet"
    overrides = {
        "fast_period": [5, 8],
        "slow_period": [21],
        "atr_multiplier": [2.0],
        "bar_timeframe": ["1min"],
    }
    report = optimize_demo(
        Settings(_env_file=None),
        param_overrides=overrides,
        workers=2,
        ledger_path=ledger,
        study_id="synthetic",
    )
    assert report.strategy == "ema_cross"
    assert report.n_parameter_sets == 2
    assert len(report.folds) == 2
    assert report.n_trials == 4
    assert {choice.params["fast_period"] for choice in report.folds} <= {5, 8}
    assert all(choice.params["bar_timeframe"] == "1min" for choice in report.folds)
    assert report.stitched.trade_count == sum(choice.oos_trade_count for choice in report.folds)
    assert report.stitched.trade_count >= 1
    names = {item.name for item in report.stability}
    assert {"fast_period", "slow_period", "atr_multiplier", "bar_timeframe"} <= names
    timeframe = next(item for item in report.stability if item.name == "bar_timeframe")
    assert timeframe.mode == "1min"
    assert timeframe.mode_fraction == 1.0
    assert report.deflated_sharpe is None or 0.0 <= report.deflated_sharpe <= 1.0
    text = report.render()
    assert "Trials: 4" in text
    assert "Deflated Sharpe" in text
    assert "Stitched OOS Sharpe" in text
    assert "Best fold OOS Sharpe" in text
    frame = read_trials(ledger)
    assert frame.height == 4
    assert set(frame["n_trials"].to_list()) == {4}
    selected = frame.filter(pl.col("selected"))
    assert selected.height == 2
    rejected = frame.filter(~pl.col("selected"))
    assert rejected["oos_trade_count"].null_count() == rejected.height
    assert selected["oos_trade_count"].null_count() == 0
    again = optimize_demo(
        Settings(_env_file=None),
        param_overrides=overrides,
        workers=1,
        ledger_path=ledger,
        study_id="synthetic-repeat",
    )
    assert [choice.params for choice in again.folds] == [choice.params for choice in report.folds]


def test_cli_optimize_demo(tmp_path):
    from typer.testing import CliRunner

    ledger = tmp_path / "trials.parquet"
    result = CliRunner().invoke(
        app,
        [
            "optimize",
            "demo",
            "--set",
            "fast_period=5",
            "--set",
            "slow_period=12",
            "--set",
            "atr_multiplier=2",
            "--set",
            "bar_timeframe=1min",
            "--workers",
            "1",
            "--ledger",
            str(ledger),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Trials:" in result.stdout
    assert "Parameter stability" in result.stdout
    assert read_trials(ledger).height == 2
