"""Checks for missing minutes, broken bars, and prints outside the regular session.

Structural problems (inverted bars, non-positive prices) are errors. Gaps,
empty sessions, extreme one-minute returns, and extended-hours prints are
warnings. A split on the session suppresses the extreme-return warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timezone
from typing import Literal

import polars as pl

from aegis.data.sessions import expected_minute_opens, is_session

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    symbol: str
    session: date | None
    code: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[Issue, ...]

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def render(self) -> str:
        if not self.issues:
            return "No data-quality issues."
        lines = [f"{len(self.errors)} errors, {len(self.warnings)} warnings."]
        for issue in self.issues:
            session = issue.session.isoformat() if issue.session else "-"
            lines.append(
                f"{issue.severity.upper():7} {issue.symbol:6} {session} {issue.code}: {issue.detail}"
            )
        return "\n".join(lines)


def validate_bars(
    bars: pl.DataFrame,
    *,
    splits: pl.DataFrame | None = None,
    extreme_return: float = 0.10,
    check_calendar: bool = True,
) -> ValidationReport:
    """Validate a frame of 1-minute bars. ``bars`` may already be a maintenance read."""
    issues: list[Issue] = []
    if bars.is_empty():
        return ValidationReport(())
    frame = bars.sort(["symbol", "ts"]).with_columns(
        pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("session"),
        pl.col("ts").dt.convert_time_zone("America/New_York").alias("et"),
    )
    split_days = _split_days(splits)
    symbols = frame["symbol"].unique().to_list()
    for symbol in symbols:
        part = frame.filter(pl.col("symbol") == symbol)
        issues.extend(_structural_issues(symbol, part))
        issues.extend(_return_issues(symbol, part, split_days, extreme_return))
        if check_calendar:
            issues.extend(_session_issues(symbol, part))
    return ValidationReport(tuple(issues))


def _as_utc_minute(ts) -> tuple:
    """Timezone-safe minute key so pandas and Polars stamps compare equal."""
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        raise ValueError("Bar timestamps must be timezone-aware.")
    utc = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return (utc.year, utc.month, utc.day, utc.hour, utc.minute)


def _split_days(splits: pl.DataFrame | None) -> set[tuple[str, date]]:
    if splits is None or splits.is_empty():
        return set()
    return {(row["symbol"], row["execution_date"]) for row in splits.select(["symbol", "execution_date"]).iter_rows(named=True)}


def _structural_issues(symbol: str, part: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    bad = part.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("volume") < 0)
        | ~pl.col("open").is_finite()
        | ~pl.col("close").is_finite()
    )
    for row in bad.iter_rows(named=True):
        issues.append(
            Issue(
                symbol=symbol,
                session=row["session"],
                code="bad_bar",
                severity="error",
                detail=(
                    f"{row['ts'].isoformat()} o={row['open']} h={row['high']} "
                    f"l={row['low']} c={row['close']} v={row['volume']}"
                ),
            )
        )
    return issues


def _return_issues(
    symbol: str,
    part: pl.DataFrame,
    split_days: set[tuple[str, date]],
    extreme_return: float,
) -> list[Issue]:
    issues: list[Issue] = []
    returns = part.with_columns(pl.col("close").pct_change().alias("ret"))
    extreme = returns.filter(pl.col("ret").abs() > extreme_return)
    for row in extreme.iter_rows(named=True):
        if (symbol, row["session"]) in split_days:
            continue
        issues.append(
            Issue(
                symbol=symbol,
                session=row["session"],
                code="extreme_return",
                severity="warning",
                detail=f"{row['ts'].isoformat()} return {row['ret']:.2%}",
            )
        )
    return issues


def _session_issues(symbol: str, part: pl.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    sessions = part["session"].unique().to_list()
    for session in sessions:
        day = part.filter(pl.col("session") == session)
        if not is_session(session):
            issues.append(
                Issue(
                    symbol=symbol,
                    session=session,
                    code="outside_regular_session",
                    severity="warning",
                    detail=f"{day.height} bars on a non-session date.",
                )
            )
            continue
        expected = expected_minute_opens(session)
        expected_set = {_as_utc_minute(item) for item in expected}
        actual = {_as_utc_minute(ts) for ts in day["ts"].to_list()}
        missing = [item for item in expected if _as_utc_minute(item) not in actual]
        outside_count = sum(1 for stamp in actual if stamp not in expected_set)
        if day.height == 0:
            issues.append(
                Issue(
                    symbol=symbol,
                    session=session,
                    code="empty_session",
                    severity="error",
                    detail="Session partition is empty.",
                )
            )
        elif missing:
            issues.append(
                Issue(
                    symbol=symbol,
                    session=session,
                    code="missing_candles",
                    severity="warning",
                    detail=f"{len(missing)} of {len(expected)} regular-session minutes are absent.",
                )
            )
        if outside_count:
            issues.append(
                Issue(
                    symbol=symbol,
                    session=session,
                    code="outside_regular_session",
                    severity="warning",
                    detail=f"{outside_count} bars fall outside the regular session.",
                )
            )
    return issues
