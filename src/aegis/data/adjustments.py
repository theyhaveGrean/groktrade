"""Split and dividend adjustments applied at read time.

The cache stores raw prints (``adjusted=false``). A research read multiplies
historical prices by splits whose execution date is after the bar and on or
before ``as_of``. Splits after ``as_of`` are ignored, which is the
point-in-time rule. Dividend adjustment is optional and off by default: it
changes the price level stops are computed from.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def apply_split_adjustments(bars: pl.DataFrame, splits: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """Backward-adjust OHLC and VWAP. Volume is scaled by the inverse factor.

    Massive ``split_from`` / ``split_to``: a 4-for-1 split is ``1 -> 4``.
    Historical prices are multiplied by ``split_from / split_to``.
    """
    if bars.is_empty():
        return bars
    work = bars.with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_session")
    )
    if splits.is_empty():
        return work.drop("_session")
    applicable = splits.filter(pl.col("execution_date") <= as_of)
    if applicable.is_empty():
        return work.drop("_session")
    applicable = applicable.with_columns(
        (pl.col("split_from") / pl.col("split_to")).alias("_px"),
        (pl.col("split_to") / pl.col("split_from")).alias("_vol"),
    )
    joined = work.join(applicable, on="symbol", how="left")
    eligible = (
        pl.col("execution_date").is_not_null()
        & (pl.col("_session") < pl.col("execution_date"))
        & (pl.col("execution_date") <= as_of)
    )
    joined = joined.with_columns(
        pl.when(eligible).then(pl.col("_px")).otherwise(1.0).alias("_px"),
        pl.when(eligible).then(pl.col("_vol")).otherwise(1.0).alias("_vol"),
    )
    factors = joined.group_by(["symbol", "ts"]).agg(
        pl.col("_px").product().alias("_px"),
        pl.col("_vol").product().alias("_vol"),
    )
    out = work.join(factors, on=["symbol", "ts"], how="left")
    out = out.with_columns(
        (pl.col("open") * pl.col("_px")).alias("open"),
        (pl.col("high") * pl.col("_px")).alias("high"),
        (pl.col("low") * pl.col("_px")).alias("low"),
        (pl.col("close") * pl.col("_px")).alias("close"),
        (pl.col("vwap") * pl.col("_px")).alias("vwap"),
        (pl.col("volume") * pl.col("_vol")).alias("volume"),
    )
    return out.drop(["_session", "_px", "_vol"]).sort(["symbol", "ts"])


def apply_dividend_adjustments(
    bars: pl.DataFrame,
    dividends: pl.DataFrame,
    splits: pl.DataFrame,
    as_of: date,
) -> pl.DataFrame:
    """Total-return adjustment using cash dividends, after split adjustment.

    Cash amounts are themselves split-adjusted for splits that land after the
    ex-date and on or before ``as_of``. Each dividend scales prices before the
    ex-date by ``(prev_close - cash) / prev_close``.
    """
    if bars.is_empty() or dividends.is_empty():
        return bars
    work = bars.sort(["symbol", "ts"]).with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_session")
    )
    last_close = (
        work.group_by(["symbol", "_session"])
        .agg(pl.col("close").last().alias("_session_close"))
        .sort(["symbol", "_session"])
    )
    last_close = last_close.with_columns(
        pl.col("_session_close").shift(1).over("symbol").alias("_prev_close")
    )
    divs = dividends.filter(pl.col("ex_date") <= as_of)
    if divs.is_empty():
        return work.drop("_session")
    factors = _dividend_factors(divs, splits, last_close, as_of)
    if factors.is_empty():
        return work.drop("_session")
    joined = work.join(factors, on="symbol", how="left")
    eligible = pl.col("ex_date").is_not_null() & (pl.col("_session") < pl.col("ex_date"))
    joined = joined.with_columns(
        pl.when(eligible).then(pl.col("_div_factor")).otherwise(1.0).alias("_div_factor")
    )
    collapsed = joined.group_by(["symbol", "ts"]).agg(pl.col("_div_factor").product().alias("_div_factor"))
    out = work.join(collapsed, on=["symbol", "ts"], how="left")
    out = out.with_columns(
        (pl.col("open") * pl.col("_div_factor")).alias("open"),
        (pl.col("high") * pl.col("_div_factor")).alias("high"),
        (pl.col("low") * pl.col("_div_factor")).alias("low"),
        (pl.col("close") * pl.col("_div_factor")).alias("close"),
        (pl.col("vwap") * pl.col("_div_factor")).alias("vwap"),
    )
    return out.drop(["_session", "_div_factor"]).sort(["symbol", "ts"])


def _dividend_factors(
    dividends: pl.DataFrame,
    splits: pl.DataFrame,
    session_closes: pl.DataFrame,
    as_of: date,
) -> pl.DataFrame:
    """One row per dividend with a price factor, or empty if none apply."""
    rows: list[dict[str, object]] = []
    split_rows = splits.to_dicts() if not splits.is_empty() else []
    closes = {
        (row["symbol"], row["_session"]): row["_prev_close"]
        for row in session_closes.to_dicts()
    }
    for dividend in dividends.to_dicts():
        symbol = dividend["symbol"]
        ex_date = dividend["ex_date"]
        if ex_date > as_of:
            continue
        cash = float(dividend["cash_amount"])
        for split in split_rows:
            if split["symbol"] != symbol:
                continue
            execution = split["execution_date"]
            if ex_date < execution <= as_of and split["split_to"]:
                cash *= float(split["split_from"]) / float(split["split_to"])
        prev = closes.get((symbol, ex_date))
        if prev is None or prev <= cash or prev <= 0:
            continue
        rows.append({"symbol": symbol, "ex_date": ex_date, "_div_factor": (prev - cash) / prev})
    if not rows:
        return pl.DataFrame(schema={"symbol": pl.String, "ex_date": pl.Date, "_div_factor": pl.Float64})
    return pl.DataFrame(rows).with_columns(pl.col("ex_date").cast(pl.Date))
