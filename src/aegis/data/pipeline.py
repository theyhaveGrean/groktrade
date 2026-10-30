"""Backfill and incremental update.

Requests run one at a time under the shared rate limiter. Each completed
session is a checkpoint: stopping the process and running the same command
again does not re-download days already in the cache. The in-flight chunk is
the only work that can repeat.

Free-tier defaults refuse today's session and anything older than about two
years. The plan is delayed end-of-day data, not a live feed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from aegis.config import Settings, load_settings
from aegis.data.cache import BarCache
from aegis.data.massive import MassiveClient
from aegis.data.plan import BackfillPlan, assert_chunk_fits_page_limit, group_missing_sessions
from aegis.data.rate_limit import RateLimiter
from aegis.data.sessions import (
    clamp_history_window,
    eastern_date,
    latest_requestable_session,
    sessions_between,
)
from aegis.data.universe import (
    Membership,
    load_universe,
    resolve_universe_path,
    select_as_of,
    symbol_window,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillResult:
    plan: BackfillPlan
    http_requests: int
    rows_written: int
    chunks_downloaded: int
    chunks_skipped: int

    def render(self) -> str:
        return "\n".join(
            [
                self.plan.render(),
                f"Downloaded chunks: {self.chunks_downloaded}",
                f"Skipped chunks: {self.chunks_skipped}",
                f"HTTP requests this run: {self.http_requests}",
                f"Rows written this run: {self.rows_written}",
            ]
        )


def build_plan(
    settings: Settings,
    *,
    start: date | None,
    end: date | None,
    symbols: list[str] | None = None,
    now: datetime | None = None,
    refresh_corporate_actions: bool = True,
) -> BackfillPlan:
    """Describe the work without calling Massive."""
    assert_chunk_fits_page_limit(settings.massive_chunk_calendar_days, settings.massive_page_limit)
    moment = now or datetime.now(timezone.utc)
    latest = latest_requestable_session(moment, plan=settings.massive_plan)
    requested_end = end or latest
    requested_start = start or (latest - timedelta(days=settings.massive_max_lookback_days))
    window_start, window_end, warnings = clamp_history_window(
        requested_start,
        requested_end,
        latest_end=latest,
        lookback_days=settings.massive_max_lookback_days,
    )
    extra = settings.free_tier_rate_warning()
    if extra:
        warnings = (*warnings, extra)
    members = resolve_members(settings, symbols, window_end)
    cache = BarCache(settings.data_cache_dir)
    bar_requests = 0
    cached_sessions = 0
    missing_sessions = 0
    corporate = 0
    for member in members:
        window = symbol_window(member, window_start, window_end)
        if window is None:
            continue
        sessions = sessions_between(window[0], window[1])
        have = cache.sessions_present(member.symbol)
        cached_sessions += sum(1 for day in sessions if day in have)
        missing = [day for day in sessions if day not in have]
        missing_sessions += len(missing)
        bar_requests += len(
            group_missing_sessions(
                sessions,
                have,
                max_calendar_days=settings.massive_chunk_calendar_days,
            )
        )
        if refresh_corporate_actions and _needs_corporate_actions(cache, member.symbol, window[1]):
            corporate += 2
    return BackfillPlan(
        start=window_start,
        end=window_end,
        symbols=tuple(member.symbol for member in members),
        bar_requests=bar_requests,
        cached_sessions=cached_sessions,
        missing_sessions=missing_sessions,
        corporate_action_requests=corporate,
        rate_limit_per_minute=settings.massive_rate_limit_per_minute,
        chunk_calendar_days=settings.massive_chunk_calendar_days,
        page_limit=settings.massive_page_limit,
        warnings=warnings,
    )


def backfill(
    settings: Settings | None = None,
    *,
    start: date,
    end: date | None = None,
    symbols: list[str] | None = None,
    now: datetime | None = None,
    client: MassiveClient | None = None,
    refresh_corporate_actions: bool = True,
) -> BackfillResult:
    """Fill ``[start, end]``, skipping session partitions that already exist."""
    settings = settings or load_settings()
    return _run(
        settings,
        start=start,
        end=end,
        symbols=symbols,
        now=now,
        client=client,
        refresh_corporate_actions=refresh_corporate_actions,
    )


def update(
    settings: Settings | None = None,
    *,
    symbols: list[str] | None = None,
    now: datetime | None = None,
    client: MassiveClient | None = None,
    refresh_corporate_actions: bool = True,
) -> BackfillResult:
    """Extend each cached symbol through the latest requestable session.

    Symbols with an empty cache are skipped. Use ``backfill`` for the first pull.
    """
    settings = settings or load_settings()
    moment = now or datetime.now(timezone.utc)
    latest = latest_requestable_session(moment, plan=settings.massive_plan)
    members = resolve_members(settings, symbols, latest)
    cache = BarCache(settings.data_cache_dir)
    cached = [member for member in members if cache.sessions_present(member.symbol)]
    skipped = [member.symbol for member in members if member not in cached]
    if skipped:
        logger.warning(
            "Skipping %s with no cache. Run `aegis data backfill --start YYYY-MM-DD` first.",
            ", ".join(skipped),
        )
    if not cached:
        plan = build_plan(
            settings,
            start=latest,
            end=latest,
            symbols=[],
            now=moment,
            refresh_corporate_actions=False,
        )
        return BackfillResult(plan, http_requests=0, rows_written=0, chunks_downloaded=0, chunks_skipped=0)
    earliest = min(min(cache.sessions_present(member.symbol)) for member in cached)
    return _run(
        settings,
        start=earliest,
        end=latest,
        symbols=[member.symbol for member in cached],
        now=moment,
        client=client,
        refresh_corporate_actions=refresh_corporate_actions,
    )


def _run(
    settings: Settings,
    *,
    start: date,
    end: date | None,
    symbols: list[str] | None,
    now: datetime | None,
    client: MassiveClient | None,
    refresh_corporate_actions: bool,
) -> BackfillResult:
    plan = build_plan(
        settings,
        start=start,
        end=end,
        symbols=symbols,
        now=now,
        refresh_corporate_actions=refresh_corporate_actions,
    )
    for warning in plan.warnings:
        logger.warning("%s", warning)
    owns_client = client is None
    if client is None:
        client = _build_client(settings)
    calls_before = client.http_calls
    cache = BarCache(settings.data_cache_dir)
    rows = 0
    downloaded = 0
    skipped = 0
    members = {member.symbol: member for member in resolve_members(settings, list(plan.symbols), plan.end)}
    try:
        for symbol in plan.symbols:
            window = symbol_window(members[symbol], plan.start, plan.end)
            if window is None:
                continue
            gained, did, missed = _sync_symbol(
                client,
                cache,
                symbol,
                window[0],
                window[1],
                chunk_days=settings.massive_chunk_calendar_days,
            )
            rows += gained
            downloaded += did
            skipped += missed
            if refresh_corporate_actions:
                _sync_corporate_actions(client, cache, symbol, plan.end)
    finally:
        if owns_client:
            client.close()
    return BackfillResult(
        plan=plan,
        http_requests=client.http_calls - calls_before,
        rows_written=rows,
        chunks_downloaded=downloaded,
        chunks_skipped=skipped,
    )


def _sync_symbol(
    client: MassiveClient,
    cache: BarCache,
    symbol: str,
    start: date,
    end: date,
    *,
    chunk_days: int,
) -> tuple[int, int, int]:
    sessions = sessions_between(start, end)
    have = cache.sessions_present(symbol)
    chunks = group_missing_sessions(sessions, have, max_calendar_days=chunk_days)
    present_chunks = _count_complete_spans(sessions, have, chunk_days)
    rows = 0
    for chunk_start, chunk_end in chunks:
        logger.info("Fetching %s %s..%s", symbol, chunk_start.isoformat(), chunk_end.isoformat())
        bars = client.fetch_minute_bars(symbol, chunk_start, chunk_end)
        chunk_sessions = [day for day in sessions if chunk_start <= day <= chunk_end]
        returned = {eastern_date(bar.ts) for bar in bars}
        to_write = [bar for bar in bars if chunk_start <= eastern_date(bar.ts) <= chunk_end]
        rows += cache.write_bars(to_write)
        empty_days = [day for day in chunk_sessions if day not in returned and day not in have]
        cache.write_empty_sessions(symbol, empty_days)
        cache.append_manifest(
            {
                "kind": "bars",
                "symbol": symbol,
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
                "rows": len(to_write),
                "status": "complete",
            }
        )
    return rows, len(chunks), present_chunks


def _count_complete_spans(sessions: list[date], have: set[date], chunk_days: int) -> int:
    """How many chunk-sized spans are already fully cached. Informational."""
    if not sessions:
        return 0
    missing = group_missing_sessions(sessions, have, max_calendar_days=chunk_days)
    everything = group_missing_sessions(sessions, set(), max_calendar_days=chunk_days)
    return max(0, len(everything) - len(missing))


def _sync_corporate_actions(client: MassiveClient, cache: BarCache, symbol: str, end: date) -> None:
    if not _needs_corporate_actions(cache, symbol, end):
        logger.info("Corporate actions for %s already cover %s.", symbol, end.isoformat())
        return
    splits = client.fetch_splits(symbol)
    dividends = client.fetch_dividends(symbol)
    if splits:
        cache.write_splits(splits)
    if dividends:
        cache.write_dividends(dividends)
    cache.mark_coverage("splits", symbol, end)
    cache.mark_coverage("dividends", symbol, end)
    cache.append_manifest(
        {
            "kind": "corporate_actions",
            "symbol": symbol,
            "end": end.isoformat(),
            "splits": len(splits),
            "dividends": len(dividends),
            "status": "complete",
        }
    )


def _needs_corporate_actions(cache: BarCache, symbol: str, end: date) -> bool:
    splits_end = cache.coverage_end("splits", symbol)
    dividends_end = cache.coverage_end("dividends", symbol)
    if splits_end is None or dividends_end is None:
        return True
    return splits_end < end or dividends_end < end


def resolve_members(settings: Settings, symbols: list[str] | None, as_of: date) -> tuple[Membership, ...]:
    universe = load_universe(resolve_universe_path(settings.universe_path))
    if symbols is not None and not symbols:
        return ()
    if symbols:
        wanted = {symbol.strip().upper() for symbol in symbols if symbol.strip()}
        chosen = tuple(member for member in universe if member.symbol in wanted)
        missing = wanted - {member.symbol for member in chosen}
        if missing:
            raise ValueError(
                "Unknown symbols: " + ", ".join(sorted(missing)) + ". Add them to the universe file."
            )
        return tuple(sorted(chosen, key=lambda member: member.symbol))
    return select_as_of(universe, as_of)


def _build_client(settings: Settings) -> MassiveClient:
    return MassiveClient(
        settings.require_massive_api_key(),
        base_url=settings.massive_base_url,
        rate_limiter=RateLimiter(settings.massive_rate_limit_per_minute),
        max_retries=settings.massive_max_retries,
        timeout=settings.massive_timeout_seconds,
        page_limit=settings.massive_page_limit,
    )
