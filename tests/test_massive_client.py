from datetime import date

import httpx
import pytest

from aegis.data.massive import MassiveClient, chunk_bounds, redact_secrets
from aegis.data.rate_limit import RateLimiter
from aegis.errors import MassiveAPIError


def _limiter() -> RateLimiter:
    return RateLimiter(100_000, sleep=lambda _seconds: None)


def _client(handler) -> MassiveClient:
    return MassiveClient(
        "super-secret-key",
        rate_limiter=_limiter(),
        max_retries=3,
        jitter_seconds=0,
        transport=httpx.MockTransport(handler),
    )


def _bar(millis: int, close: float = 10.0) -> dict:
    return {"t": millis, "o": close, "h": close + 0.1, "l": close - 0.1, "c": close, "v": 100, "vw": close, "n": 4}


def test_aggregates_use_max_limit_and_follow_next_url(caplog):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "cursor" not in str(request.url):
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [_bar(1_704_277_800_000)],
                    "next_url": "https://api.massive.com/v2/aggs/ticker/AAPL/range/1/minute/2024-01-02/2024-03-11?cursor=abc",
                },
            )
        return httpx.Response(200, json={"status": "DELAYED", "results": [_bar(1_704_277_860_000, 11.0)]})

    with _client(handler) as client:
        bars = client.fetch_minute_bars("AAPL", date(2024, 1, 2), date(2024, 3, 11))

    assert len(bars) == 2
    assert bars[1].close == 11.0
    first = seen[0]
    assert first.url.params["limit"] == "50000"
    assert first.url.params["adjusted"] == "false"
    assert first.url.params["sort"] == "asc"
    assert "/range/1/minute/2024-01-02/2024-03-11" in str(first.url)
    assert first.headers["Authorization"] == "Bearer super-secret-key"
    assert "apiKey" not in str(first.url)
    assert "super-secret-key" not in caplog.text
    assert client.http_calls == 2


def test_retries_429_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"status": "ERROR"})
        return httpx.Response(200, json={"status": "OK", "results": [_bar(1_704_277_800_000)]})

    with _client(handler) as client:
        bars = client.fetch_minute_bars("MSFT", date(2024, 1, 2), date(2024, 1, 5))
    assert len(bars) == 1
    assert attempts["n"] == 2


def test_client_error_is_not_retried_and_redacts_secrets():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"status":"ERROR","error":"bad apiKey=super-secret-key"}')

    with _client(handler) as client:
        with pytest.raises(MassiveAPIError) as raised:
            client.fetch_minute_bars("AAPL", date(2024, 1, 2), date(2024, 1, 2))
    assert "super-secret-key" not in str(raised.value)
    assert client.http_calls == 1


def test_chunk_bounds_cover_the_range_without_overlap():
    chunks = chunk_bounds(date(2024, 1, 1), date(2024, 4, 10), calendar_days=70)
    assert chunks[0] == (date(2024, 1, 1), date(2024, 3, 10))
    assert chunks[-1][1] == date(2024, 4, 10)
    assert all((end - start).days + 1 <= 70 for start, end in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert current[0] == previous[1].fromordinal(previous[1].toordinal() + 1)


def test_redact_secrets():
    assert redact_secrets("https://api.massive.com/x?apiKey=abc123&limit=5") == "https://api.massive.com/x?apiKey=***&limit=5"
