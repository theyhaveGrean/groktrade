from datetime import date, datetime, timedelta, timezone

import polars as pl

from aegis.backtest.synthetic import synthetic_bars
from aegis.data.adjustments import apply_dividend_adjustments, apply_split_adjustments
from aegis.data.cache import bars_to_frame
from aegis.data.models import MinuteBar
from aegis.data.validation import validate_bars


def _frame(rows: list[MinuteBar]) -> pl.DataFrame:
    return bars_to_frame(rows)


def test_split_adjustment_is_point_in_time():
    bars = _frame(
        [
            MinuteBar("AAPL", datetime(2020, 8, 28, 14, 30, tzinfo=timezone.utc), 400, 404, 396, 400, 1000, 400),
        ]
    )
    splits = pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "execution_date": [date(2020, 8, 31)],
            "split_from": [1.0],
            "split_to": [4.0],
        }
    ).with_columns(pl.col("execution_date").cast(pl.Date))
    before = apply_split_adjustments(bars, splits, date(2020, 8, 1))
    after = apply_split_adjustments(bars, splits, date(2020, 9, 1))
    assert before["close"].to_list() == [400.0]
    assert after["close"].to_list() == [100.0]
    assert after["volume"].to_list() == [4000.0]


def test_dividend_adjustment_scales_prices_before_the_ex_date():
    day1 = datetime(2024, 1, 3, 20, 0, tzinfo=timezone.utc)
    day2 = datetime(2024, 1, 4, 20, 0, tzinfo=timezone.utc)
    bars = _frame(
        [
            MinuteBar("AAPL", day1, 100, 101, 99, 100, 10),
            MinuteBar("AAPL", day2, 99, 100, 98, 99, 10),
        ]
    )
    dividends = pl.DataFrame(
        {"symbol": ["AAPL"], "ex_date": [date(2024, 1, 4)], "cash_amount": [1.0], "currency": ["USD"]}
    ).with_columns(pl.col("ex_date").cast(pl.Date))
    splits = pl.DataFrame(
        schema={"symbol": pl.String, "execution_date": pl.Date, "split_from": pl.Float64, "split_to": pl.Float64}
    )
    adjusted = apply_dividend_adjustments(bars, dividends, splits, date(2024, 1, 4))
    closes = adjusted.sort("ts")["close"].to_list()
    assert closes[0] == 99.0
    assert closes[1] == 99.0


def test_regular_hours_filter_keeps_a_full_synthetic_session():
    from aegis.data.query import filter_regular_hours

    bars = synthetic_bars()
    assert filter_regular_hours(bars).height == bars.height


def test_validation_flags_bad_bars_and_accepts_a_full_synthetic_session():
    bad = _frame(
        [MinuteBar("AAPL", datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc), 10, 9, 11, 10, 5)]
    )
    report = validate_bars(bad, check_calendar=False)
    assert report.errors
    assert report.errors[0].code == "bad_bar"

    clean = synthetic_bars().filter(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date() == date(2024, 1, 3)
    )
    clean_report = validate_bars(clean)
    assert clean_report.errors == ()
    assert not any(issue.code == "missing_candles" for issue in clean_report.warnings)

    gappy = clean.filter(
        pl.col("ts") != datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc)
    )
    # Drop a few midday minutes.
    drop_from = datetime(2024, 1, 3, 16, 0, tzinfo=timezone.utc)
    gappy = clean.filter(pl.col("ts") < drop_from)
    gap_report = validate_bars(gappy)
    assert any(issue.code == "missing_candles" for issue in gap_report.warnings)


def test_extreme_return_is_suppressed_on_a_split_day():
    first = datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc)
    second = first + timedelta(minutes=1)
    bars = _frame(
        [
            MinuteBar("AAPL", first, 100, 101, 99, 100, 10),
            MinuteBar("AAPL", second, 50, 51, 49, 50, 10),
        ]
    )
    plain = validate_bars(bars, check_calendar=False)
    assert any(issue.code == "extreme_return" for issue in plain.warnings)
    splits = pl.DataFrame(
        {
            "symbol": ["AAPL"],
            "execution_date": [date(2024, 1, 3)],
            "split_from": [1.0],
            "split_to": [2.0],
        }
    ).with_columns(pl.col("execution_date").cast(pl.Date))
    suppressed = validate_bars(bars, splits=splits, check_calendar=False)
    assert not any(issue.code == "extreme_return" for issue in suppressed.warnings)
