"""Net realized R labels for candidate long signals.

Entry is the next bar's open, which is the earliest price a market order
submitted on the signal bar can see. The stop and target use Wilder ATR at
the signal bar and the same 2.0 / 2.5 geometry as the risk engine. Costs are
commission per share on both sides plus slippage in basis points of each fill.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from aegis.backtest.risk import AtrState

NY = ZoneInfo("America/New_York")
ENTRY_START = time(9, 35)
ENTRY_END = time(15, 30)
FLATTEN = time(15, 55)


def candidate_signals(bars: pl.DataFrame, *, every_n: int = 15) -> pl.DataFrame:
    """Causal timestamps inside the entry window, one every ``every_n`` minutes."""
    if bars.is_empty():
        return pl.DataFrame(schema={"symbol": pl.String, "ts": bars.schema["ts"]})
    frame = bars.sort(["symbol", "ts"]).with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").alias("_et")
    )
    frame = frame.with_columns(
        (pl.col("_et").dt.hour().cast(pl.Int32) * 60 + pl.col("_et").dt.minute().cast(pl.Int32) - (9 * 60 + 30)).alias(
            "_offset"
        )
    )
    kept = frame.filter(
        (pl.col("_offset") >= 5)
        & (pl.col("_offset") <= (15 * 60 + 30) - (9 * 60 + 30))
        & (pl.col("_offset") % every_n == 0)
    )
    return kept.select(["symbol", "ts"])


def label_signals(
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    reward_to_risk: float = 2.5,
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
) -> pl.DataFrame:
    """One row per signal that has a next bar and a positive ATR stop."""
    schema = {
        "symbol": pl.String,
        "ts": pl.Datetime("us", "UTC"),
        "exit_ts": pl.Datetime("us", "UTC"),
        "session": pl.Date,
        "exit_session": pl.Date,
        "entry_price": pl.Float64,
        "exit_price": pl.Float64,
        "stop_price": pl.Float64,
        "target_price": pl.Float64,
        "net_r": pl.Float64,
        "gross_r": pl.Float64,
        "cost_per_share": pl.Float64,
        "exit_reason": pl.String,
    }
    if bars.is_empty() or signals.is_empty():
        return pl.DataFrame(schema=schema)
    rows: list[dict] = []
    signal_keys = {(row["symbol"], row["ts"]) for row in signals.select(["symbol", "ts"]).iter_rows(named=True)}
    for symbol in bars["symbol"].unique().sort().to_list():
        part = bars.filter(pl.col("symbol") == symbol).sort("ts")
        records = list(part.iter_rows(named=True))
        atr_state = AtrState(atr_period)
        atrs: list[float | None] = []
        for record in records:
            atrs.append(atr_state.update(float(record["high"]), float(record["low"]), float(record["close"])))
        by_ts = {record["ts"]: index for index, record in enumerate(records)}
        for record in records:
            key = (symbol, record["ts"])
            if key not in signal_keys:
                continue
            index = by_ts[record["ts"]]
            labeled = _label_one(
                records,
                atrs,
                index,
                atr_multiplier=atr_multiplier,
                reward_to_risk=reward_to_risk,
                commission_per_share=commission_per_share,
                slippage_bps=slippage_bps,
            )
            if labeled is not None:
                labeled["symbol"] = symbol
                rows.append(labeled)
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")),
        pl.col("exit_ts").cast(pl.Datetime("us", "UTC")),
    )


def _label_one(
    records: list[dict],
    atrs: list[float | None],
    index: int,
    *,
    atr_multiplier: float,
    reward_to_risk: float,
    commission_per_share: float,
    slippage_bps: float,
) -> dict | None:
    signal = records[index]
    local = _eastern(signal["ts"])
    if local.weekday() >= 5 or not (ENTRY_START <= local.time() <= ENTRY_END):
        return None
    atr = atrs[index]
    if atr is None or atr <= 0 or index + 1 >= len(records):
        return None
    entry_bar = records[index + 1]
    entry_local = _eastern(entry_bar["ts"])
    if entry_local.time() >= FLATTEN or entry_local.date() != local.date():
        return None
    entry = float(entry_bar["open"])
    stop = entry - atr_multiplier * atr
    risk = entry - stop
    if risk <= 0.01:
        return None
    target = entry + reward_to_risk * risk
    exit_price = float(entry_bar["close"])
    exit_ts = entry_bar["ts"]
    reason = "flatten"
    for cursor in range(index + 1, len(records)):
        bar = records[cursor]
        bar_local = _eastern(bar["ts"])
        if bar_local.date() != local.date():
            break
        open_ = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        if bar_local.time() >= FLATTEN:
            if low <= stop:
                exit_price = open_ if open_ <= stop else stop
                reason = "stop"
            else:
                exit_price = close
                reason = "flatten"
            exit_ts = bar["ts"]
            break
        if low <= stop:
            exit_price = open_ if open_ <= stop else stop
            reason = "stop"
            exit_ts = bar["ts"]
            break
        if high >= target:
            exit_price = open_ if open_ >= target else target
            reason = "target"
            exit_ts = bar["ts"]
            break
        exit_price = close
        exit_ts = bar["ts"]
        reason = "end"
    slip = slippage_bps / 10_000.0
    cost = 2.0 * commission_per_share + entry * slip + exit_price * slip
    net = (exit_price - entry) - cost
    return {
        "ts": signal["ts"],
        "exit_ts": exit_ts,
        "session": local.date(),
        "exit_session": _eastern(exit_ts).date(),
        "entry_price": entry,
        "exit_price": exit_price,
        "stop_price": stop,
        "target_price": target,
        "net_r": net / risk,
        "gross_r": (exit_price - entry) / risk,
        "cost_per_share": cost,
        "exit_reason": reason,
    }


def _eastern(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=NY)
    return ts.astimezone(NY)
