"""Point-in-time price, volume, and volatility features.

Every column at a row uses only that row and earlier rows of the same symbol.
Session VWAP resets on the Eastern date and does not see later bars.
"""

from __future__ import annotations

import polars as pl

FEATURE_COLUMNS = (
    "ret_1",
    "ret_5",
    "ret_15",
    "vol_15",
    "vol_30",
    "volume_z_20",
    "atr_pct_14",
    "vwap_dist",
    "bb_z_20",
)


def compute_features(bars: pl.DataFrame) -> pl.DataFrame:
    """Add feature columns. Warmup rows stay null. Input order is restored by time."""
    if bars.is_empty():
        return bars
    group = ["symbol"]
    frame = bars.sort(["symbol", "ts"]).with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("session"),
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_typical"),
    )
    frame = frame.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(group) - 1.0).alias("ret_1"),
        (pl.col("close") / pl.col("close").shift(5).over(group) - 1.0).alias("ret_5"),
        (pl.col("close") / pl.col("close").shift(15).over(group) - 1.0).alias("ret_15"),
        (pl.col("_typical") * pl.col("volume")).cum_sum().over(["symbol", "session"]).alias("_tpv"),
        pl.col("volume").cum_sum().over(["symbol", "session"]).alias("_cum_volume"),
    )
    previous_close = pl.col("close").shift(1).over(group)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )
    rolling_close_mean = pl.col("close").rolling_mean(window_size=20, min_samples=20).over(group)
    rolling_close_std = pl.col("close").rolling_std(window_size=20, min_samples=20).over(group)
    volume_mean = pl.col("volume").rolling_mean(window_size=20, min_samples=20).over(group)
    volume_std = pl.col("volume").rolling_std(window_size=20, min_samples=20).over(group)
    frame = frame.with_columns(
        pl.col("ret_1").rolling_std(window_size=15, min_samples=15).over(group).alias("vol_15"),
        pl.col("ret_1").rolling_std(window_size=30, min_samples=30).over(group).alias("vol_30"),
        pl.when(volume_std > 0)
        .then((pl.col("volume") - volume_mean) / volume_std)
        .when(volume_std.is_not_null())
        .then(0.0)
        .otherwise(None)
        .alias("volume_z_20"),
        (true_range.rolling_mean(window_size=14, min_samples=14).over(group) / pl.col("close")).alias("atr_pct_14"),
        pl.when(pl.col("_cum_volume") > 0)
        .then(pl.col("close") / (pl.col("_tpv") / pl.col("_cum_volume")) - 1.0)
        .otherwise(None)
        .alias("vwap_dist"),
        pl.when(rolling_close_std > 0)
        .then((pl.col("close") - rolling_close_mean) / rolling_close_std)
        .otherwise(None)
        .alias("bb_z_20"),
    )
    return frame.drop(["_typical", "_tpv", "_cum_volume"])
