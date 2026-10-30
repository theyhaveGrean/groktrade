"""Massive (formerly Polygon.io) REST client for 1-minute stock aggregates.

The free tier is 5 requests per minute, about two years of history, and
delayed end-of-day data. This client does not call snapshot, trade, or
websocket endpoints. Every call is rate-limited, retried on 429 and 5xx, and
paged via ``next_url``. Aggregates are requested with ``limit=50000`` and
``adjusted=false`` so the cache keeps raw prints.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import httpx

from aegis.config import MAX_AGGREGATE_PAGE_LIMIT
from aegis.data.models import Dividend, MinuteBar, Split
from aegis.data.rate_limit import RateLimiter
from aegis.errors import MassiveAPIError

logger = logging.getLogger(__name__)

_SECRET_QUERY = re.compile(r"(apiKey|api_key|apikey)=([^&\s]+)", re.IGNORECASE)
_OK_STATUSES = {"OK", "DELAYED"}


def redact_secrets(text: str) -> str:
    return _SECRET_QUERY.sub(r"\1=***", text)


def _epoch_to_utc(raw: int | float) -> datetime:
    value = float(raw)
    if value > 10**16:
        seconds = value / 1e9
    elif value > 10**14:
        seconds = value / 1e6
    else:
        seconds = value / 1e3
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


class MassiveClient:
    """HTTP client. Pass ``transport`` in tests to avoid the network."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.massive.com",
        rate_limiter: RateLimiter | None = None,
        calls_per_minute: float = 5,
        max_retries: int = 5,
        timeout: float = 30.0,
        page_limit: int = MAX_AGGREGATE_PAGE_LIMIT,
        transport: httpx.BaseTransport | None = None,
        jitter_seconds: float = 0.25,
        rng: random.Random | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("MassiveClient requires an API key.")
        if not 1 <= page_limit <= MAX_AGGREGATE_PAGE_LIMIT:
            raise ValueError(f"page_limit must be between 1 and {MAX_AGGREGATE_PAGE_LIMIT}.")
        self.page_limit = page_limit
        self.max_retries = max_retries
        self.http_calls = 0
        self._jitter_seconds = jitter_seconds
        self._rng = rng or random.Random()
        self._rate_limiter = rate_limiter or RateLimiter(
            calls_per_minute,
            sleep=sleep or time.sleep,
        )
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "aegis/0.1.0",
            },
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MassiveClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_minute_bars(self, symbol: str, start: date, end: date) -> list[MinuteBar]:
        """All 1-minute bars in ``[start, end]``, following ``next_url``.

        The caller should keep ``end - start`` inside one chunk (~70 calendar
        days) so this is usually a single HTTP call. Pagination is still
        followed so a long window cannot silently truncate.
        """
        if start > end:
            raise ValueError("start must be on or before end.")
        path = (
            f"/v2/aggs/ticker/{quote(symbol, safe='.')}/range/1/minute/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        params: dict[str, str] | None = {
            "adjusted": "false",
            "sort": "asc",
            "limit": str(self.page_limit),
        }
        bars: list[MinuteBar] = []
        url: str | None = path
        while url:
            payload = self._get_json(url, params)
            bars.extend(_parse_aggregates(payload, symbol))
            url = payload.get("next_url") or None
            params = None
        return bars

    def fetch_splits(self, symbol: str) -> list[Split]:
        return [
            _parse_split(row, symbol)
            for row in self._collect_v3(
                "/v3/reference/splits",
                {"ticker": symbol, "limit": "1000", "sort": "execution_date", "order": "asc"},
            )
        ]

    def fetch_dividends(self, symbol: str) -> list[Dividend]:
        return [
            _parse_dividend(row, symbol)
            for row in self._collect_v3(
                "/v3/reference/dividends",
                {"ticker": symbol, "limit": "1000", "sort": "ex_dividend_date", "order": "asc"},
            )
        ]

    def _collect_v3(self, path: str, params: dict[str, str]) -> list[dict]:
        rows: list[dict] = []
        url: str | None = path
        query: dict[str, str] | None = params
        while url:
            payload = self._get_json(url, query)
            rows.extend(payload.get("results") or [])
            url = payload.get("next_url") or None
            query = None
        return rows

    def _get_json(self, url: str, params: dict[str, str] | None) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._rate_limiter.acquire()
            self.http_calls += 1
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                last_error = exc
                self._backoff(attempt, None)
                continue
            logger.info("GET %s -> %s", redact_secrets(str(response.request.url)), response.status_code)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = MassiveAPIError(
                    f"Massive HTTP {response.status_code} for {redact_secrets(str(response.request.url))}"
                )
                self._backoff(attempt, response)
                continue
            if response.status_code >= 400:
                body = redact_secrets(response.text)[:300]
                raise MassiveAPIError(f"Massive HTTP {response.status_code}: {body}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise MassiveAPIError("Massive returned a non-JSON body.") from exc
            status = str(payload.get("status") or "OK")
            if status not in _OK_STATUSES:
                message = redact_secrets(str(payload.get("error") or payload.get("message") or status))
                if _looks_rate_limited(message) and attempt + 1 < self.max_retries:
                    last_error = MassiveAPIError(message)
                    self._backoff(attempt, response)
                    continue
                raise MassiveAPIError(f"Massive status {status}: {message[:300]}")
            return payload
        raise MassiveAPIError(
            f"Massive request failed after {self.max_retries} attempts."
        ) from last_error

    def _backoff(self, attempt: int, response: httpx.Response | None) -> None:
        retry_after = _retry_after_seconds(response)
        exponential = min(60.0, 2.0**attempt)
        delay = exponential if retry_after is None else retry_after
        delay = max(delay, self._rate_limiter.min_interval)
        delay = min(delay, 120.0)
        if self._jitter_seconds > 0:
            delay += self._rng.random() * self._jitter_seconds
        logger.warning("Backing off %.1fs after a retryable Massive response.", delay)
        self._rate_limiter.penalize(delay)


def _retry_after_seconds(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _looks_rate_limited(message: str) -> bool:
    text = message.lower()
    return "rate" in text or "too many" in text or "limit" in text


def _parse_aggregates(payload: dict, symbol: str) -> list[MinuteBar]:
    rows = payload.get("results") or []
    bars: list[MinuteBar] = []
    for row in rows:
        bars.append(
            MinuteBar(
                symbol=symbol,
                ts=_epoch_to_utc(row["t"]),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row.get("v") or 0.0),
                vwap=float(row["vw"]) if row.get("vw") is not None else None,
                transactions=int(row["n"]) if row.get("n") is not None else None,
            )
        )
    return bars


def _parse_split(row: dict, symbol: str) -> Split:
    return Split(
        symbol=str(row.get("ticker") or symbol),
        execution_date=date.fromisoformat(str(row["execution_date"])),
        split_from=float(row["split_from"]),
        split_to=float(row["split_to"]),
    )


def _parse_dividend(row: dict, symbol: str) -> Dividend:
    ex = row.get("ex_dividend_date") or row.get("ex_date")
    return Dividend(
        symbol=str(row.get("ticker") or symbol),
        ex_date=date.fromisoformat(str(ex)),
        cash_amount=float(row["cash_amount"]),
        currency=str(row.get("currency") or "USD"),
    )


def chunk_bounds(start: date, end: date, *, calendar_days: int) -> list[tuple[date, date]]:
    """Inclusive calendar slices of at most ``calendar_days``. Used by tests and docs."""
    if calendar_days < 1:
        raise ValueError("calendar_days must be positive.")
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=calendar_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks
