"""Parquet cache partitioned by symbol and Eastern session date.

A session directory is the resume checkpoint. ``aegis data backfill`` writes a
day only after that day's bars are in memory, via a temp file and ``rename``.
A killed process leaves finished days in place and retries the missing ones.
Raw, unadjusted prints are what gets stored. Adjustment happens at read time.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from aegis.data.models import Dividend, MinuteBar, Split
from aegis.data.sessions import eastern_date
from aegis.data.universe import validate_symbol
from aegis.errors import HoldoutViolation
from aegis.holdout import DateRange, assert_no_holdout_overlap

BAR_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "ts": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "vwap": pl.Float64,
    "transactions": pl.Int64,
}


def empty_bars() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


class BarCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def sessions_present(self, symbol: str) -> set[date]:
        symbol = validate_symbol(symbol)
        directory = self.root / "bars" / f"symbol={symbol}"
        if not directory.exists():
            return set()
        found: set[date] = set()
        for path in directory.glob("date=*/bars.parquet"):
            found.add(date.fromisoformat(path.parent.name.split("=", 1)[1]))
        return found

    def write_session(self, symbol: str, session: date, frame: pl.DataFrame) -> None:
        symbol = validate_symbol(symbol)
        path = self._bars_path(symbol, session)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = _normalize_bars(frame, symbol)
        temporary = path.with_suffix(".parquet.tmp")
        normalized.write_parquet(temporary)
        temporary.replace(path)

    def write_bars(self, bars: list[MinuteBar]) -> int:
        """Write bars into their session partitions. Returns the row count."""
        if not bars:
            return 0
        frame = bars_to_frame(bars)
        count = 0
        for (symbol, session), part in _split_sessions(frame):
            self.write_session(symbol, session, part)
            count += part.height
        return count

    def write_empty_sessions(self, symbol: str, sessions: list[date]) -> None:
        """Mark sessions the API returned nothing for, so a resume does not refetch them."""
        empty = empty_bars()
        for session in sessions:
            if session not in self.sessions_present(symbol):
                self.write_session(symbol, session, empty)

    def read_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        *,
        purpose: str = "research",
        holdout: DateRange | None = None,
    ) -> pl.DataFrame:
        """Read cached bars.

        The default purpose is ``research``, which refuses the holdout. Data
        quality and backfill bookkeeping must pass ``purpose="maintenance"``.
        """
        if start > end:
            raise ValueError("start must be on or before end.")
        if purpose == "research":
            if holdout is None:
                raise HoldoutViolation(
                    "Research reads require the protected holdout window. "
                    "Pass holdout=settings.holdout, or use purpose='maintenance' for data checks."
                )
            assert_no_holdout_overlap(
                DateRange(start, end),
                holdout,
                purpose="research bar read",
            )
        elif purpose != "maintenance":
            raise ValueError("purpose must be 'research' or 'maintenance'.")
        files = self._files_for(symbols, start, end)
        if not files:
            return empty_bars()
        import duckdb

        connection = duckdb.connect()
        try:
            table = connection.sql(
                "SELECT * FROM read_parquet(?) ORDER BY symbol, ts",
                params=[files],
            ).to_arrow_table()
        finally:
            connection.close()
        frame = pl.from_arrow(table)
        if not isinstance(frame, pl.DataFrame):
            frame = pl.DataFrame(frame)
        return _normalize_bars(frame)

    def coverage_end(self, kind: str, symbol: str) -> date | None:
        path = self._coverage_path(kind, symbol)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return date.fromisoformat(payload["end"])

    def mark_coverage(self, kind: str, symbol: str, end: date) -> None:
        path = self._coverage_path(kind, symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"end": end.isoformat()}), encoding="utf-8")

    def write_splits(self, splits: list[Split]) -> None:
        if not splits:
            return
        symbol = validate_symbol(splits[0].symbol)
        frame = pl.DataFrame(
            {
                "symbol": [row.symbol for row in splits],
                "execution_date": [row.execution_date for row in splits],
                "split_from": [row.split_from for row in splits],
                "split_to": [row.split_to for row in splits],
            }
        ).with_columns(pl.col("execution_date").cast(pl.Date))
        path = self._corporate_path("splits", symbol)
        _atomic_write(path, frame)

    def write_dividends(self, dividends: list[Dividend]) -> None:
        if not dividends:
            return
        symbol = validate_symbol(dividends[0].symbol)
        frame = pl.DataFrame(
            {
                "symbol": [row.symbol for row in dividends],
                "ex_date": [row.ex_date for row in dividends],
                "cash_amount": [row.cash_amount for row in dividends],
                "currency": [row.currency for row in dividends],
            }
        ).with_columns(pl.col("ex_date").cast(pl.Date))
        path = self._corporate_path("dividends", symbol)
        _atomic_write(path, frame)

    def read_splits(self, symbols: list[str]) -> pl.DataFrame:
        return self._read_corporate(
            "splits",
            symbols,
            {
                "symbol": pl.String,
                "execution_date": pl.Date,
                "split_from": pl.Float64,
                "split_to": pl.Float64,
            },
        )

    def read_dividends(self, symbols: list[str]) -> pl.DataFrame:
        return self._read_corporate(
            "dividends",
            symbols,
            {
                "symbol": pl.String,
                "ex_date": pl.Date,
                "cash_amount": pl.Float64,
                "currency": pl.String,
            },
        )

    def append_manifest(self, record: dict) -> None:
        path = self.root / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **record}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def _files_for(self, symbols: list[str], start: date, end: date) -> list[str]:
        files: list[str] = []
        for symbol in symbols:
            symbol = validate_symbol(symbol)
            directory = self.root / "bars" / f"symbol={symbol}"
            if not directory.exists():
                continue
            for path in sorted(directory.glob("date=*/bars.parquet")):
                session = date.fromisoformat(path.parent.name.split("=", 1)[1])
                if start <= session <= end:
                    files.append(str(path))
        return files

    def _bars_path(self, symbol: str, session: date) -> Path:
        return self.root / "bars" / f"symbol={symbol}" / f"date={session.isoformat()}" / "bars.parquet"

    def _corporate_path(self, kind: str, symbol: str) -> Path:
        return self.root / "corporate_actions" / kind / f"symbol={symbol}" / f"{kind}.parquet"

    def _coverage_path(self, kind: str, symbol: str) -> Path:
        return self.root / "corporate_actions" / kind / f"symbol={validate_symbol(symbol)}" / "coverage.json"

    def _read_corporate(self, kind: str, symbols: list[str], schema: dict[str, pl.DataType]) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for symbol in symbols:
            path = self._corporate_path(kind, validate_symbol(symbol))
            if path.exists():
                frames.append(pl.read_parquet(path))
        if not frames:
            return pl.DataFrame(schema=schema)
        return pl.concat(frames, how="diagonal_relaxed")


def bars_to_frame(bars: list[MinuteBar]) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "symbol": [bar.symbol for bar in bars],
            "ts": [bar.ts for bar in bars],
            "open": [bar.open for bar in bars],
            "high": [bar.high for bar in bars],
            "low": [bar.low for bar in bars],
            "close": [bar.close for bar in bars],
            "volume": [bar.volume for bar in bars],
            "vwap": [bar.vwap for bar in bars],
            "transactions": [bar.transactions for bar in bars],
        }
    )
    return _normalize_bars(frame)


def _normalize_bars(frame: pl.DataFrame, symbol: str | None = None) -> pl.DataFrame:
    if frame.is_empty():
        return empty_bars()
    work = frame
    if symbol is not None and "symbol" not in work.columns:
        work = work.with_columns(pl.lit(symbol).alias("symbol"))
    if work.schema["ts"].time_zone is None:
        work = work.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    else:
        work = work.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
    work = work.with_columns(
        pl.col("ts").cast(pl.Datetime("us", "UTC")),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("vwap").cast(pl.Float64),
        pl.col("transactions").cast(pl.Int64),
        pl.col("symbol").cast(pl.String),
    )
    work = work.select(list(BAR_SCHEMA))
    work = work.unique(subset=["symbol", "ts"], keep="last").sort(["symbol", "ts"])
    return work


def _split_sessions(frame: pl.DataFrame) -> list[tuple[tuple[str, date], pl.DataFrame]]:
    stamped = frame.with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("_session")
    )
    parts: list[tuple[tuple[str, date], pl.DataFrame]] = []
    for symbol, session in stamped.select(["symbol", "_session"]).unique().iter_rows():
        part = stamped.filter((pl.col("symbol") == symbol) & (pl.col("_session") == session)).drop("_session")
        parts.append(((symbol, session), part))
    return parts


def _atomic_write(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def session_of(ts: datetime) -> date:
    return eastern_date(ts)
