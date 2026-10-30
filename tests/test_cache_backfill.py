from datetime import date, datetime, timezone

import httpx
import pytest

from aegis.config import Settings
from aegis.data.cache import BarCache
from aegis.data.massive import MassiveClient
from aegis.data.models import MinuteBar
from aegis.data.pipeline import backfill
from aegis.data.rate_limit import RateLimiter
from aegis.holdout import DateRange
from aegis.errors import HoldoutViolation, MassiveAPIError


def _bar(symbol: str, ts: datetime, close: float = 10.0) -> dict:
    millis = int(ts.timestamp() * 1000)
    return {"t": millis, "o": close, "h": close + 0.2, "l": close - 0.2, "c": close, "v": 50, "vw": close, "n": 2}


def test_partitioned_roundtrip_and_research_holdout(tmp_path):
    cache = BarCache(tmp_path)
    holdout = DateRange(date(2026, 7, 1), date(2026, 10, 31))
    research_bar = MinuteBar("AAPL", datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc), 10, 11, 9, 10.5, 100)
    holdout_bar = MinuteBar("AAPL", datetime(2026, 7, 15, 14, 30, tzinfo=timezone.utc), 10, 11, 9, 10.5, 100)
    cache.write_bars([research_bar, holdout_bar])
    assert cache.sessions_present("AAPL") == {date(2024, 1, 3), date(2026, 7, 15)}
    maintenance = cache.read_bars(["AAPL"], date(2026, 7, 1), date(2026, 7, 31), purpose="maintenance")
    assert maintenance.height == 1
    with pytest.raises(HoldoutViolation):
        cache.read_bars(["AAPL"], date(2026, 7, 1), date(2026, 7, 31), purpose="research", holdout=holdout)
    allowed = cache.read_bars(["AAPL"], date(2024, 1, 1), date(2024, 1, 31), purpose="research", holdout=holdout)
    assert allowed["close"].to_list() == [10.5]
    with pytest.raises(HoldoutViolation):
        cache.read_bars(["AAPL"], date(2024, 1, 1), date(2024, 1, 31), purpose="research", holdout=None)


def test_backfill_is_resumable_and_uses_one_multiday_request(tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url.path))
        path = request.url.path
        if "/v2/aggs/" in path:
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [
                        _bar("AAPL", datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc), 180),
                        _bar("AAPL", datetime(2024, 1, 4, 14, 31, tzinfo=timezone.utc), 181),
                    ],
                },
            )
        return httpx.Response(200, json={"status": "OK", "results": []})

    client = MassiveClient(
        "test-key",
        rate_limiter=RateLimiter(100_000, sleep=lambda _seconds: None),
        jitter_seconds=0,
        transport=httpx.MockTransport(handler),
    )
    now = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)
    # Universe file is missing, so pass symbols that exist on the packaged universe.
    # Point the settings at the real config by leaving the default if the file exists.
    settings = Settings(
        _env_file=None,
        massive_api_key="test-key",
        data_cache_dir=tmp_path,
        massive_plan="free",
        massive_rate_limit_per_minute=100_000,
        massive_max_lookback_days=3650,
    )
    first = backfill(
        settings,
        start=date(2024, 1, 3),
        end=date(2024, 1, 4),
        symbols=["AAPL"],
        now=now,
        client=client,
    )
    assert first.rows_written == 2
    assert first.chunks_downloaded == 1
    bar_calls = [path for path in calls if "/v2/aggs/" in path]
    assert len(bar_calls) == 1
    assert "/range/1/minute/2024-01-03/2024-01-04" in bar_calls[0]
    assert any("splits" in path for path in calls)
    cached_calls = len(calls)
    second = backfill(
        settings,
        start=date(2024, 1, 3),
        end=date(2024, 1, 4),
        symbols=["AAPL"],
        now=now,
        client=client,
    )
    assert second.rows_written == 0
    assert second.chunks_downloaded == 0
    assert len(calls) == cached_calls
    frame = BarCache(tmp_path).read_bars(["AAPL"], date(2024, 1, 3), date(2024, 1, 4), purpose="maintenance")
    assert frame.height == 2
    assert set(frame["close"].to_list()) == {180.0, 181.0}


def test_failed_chunk_is_retried_and_finished_days_are_kept(tmp_path):
    state = {"aggs": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/v2/aggs/" in request.url.path:
            state["aggs"] += 1
            if state["aggs"] <= 2:
                return httpx.Response(500, text="unavailable")
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [_bar("AAPL", datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc), 50)],
                },
            )
        return httpx.Response(200, json={"status": "OK", "results": []})

    settings = Settings(
        _env_file=None,
        massive_api_key="test-key",
        data_cache_dir=tmp_path,
        massive_plan="free",
        massive_rate_limit_per_minute=100_000,
        massive_max_lookback_days=3650,
        massive_max_retries=2,
    )
    now = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)

    def make_client() -> MassiveClient:
        return MassiveClient(
            "test-key",
            rate_limiter=RateLimiter(100_000, sleep=lambda _seconds: None),
            max_retries=2,
            jitter_seconds=0,
            transport=httpx.MockTransport(handler),
        )

    with pytest.raises(MassiveAPIError):
        backfill(
            settings,
            start=date(2024, 1, 3),
            end=date(2024, 1, 3),
            symbols=["AAPL"],
            now=now,
            client=make_client(),
        )
    assert BarCache(tmp_path).sessions_present("AAPL") == set()
    result = backfill(
        settings,
        start=date(2024, 1, 3),
        end=date(2024, 1, 3),
        symbols=["AAPL"],
        now=now,
        client=make_client(),
    )
    assert result.rows_written == 1
    assert date(2024, 1, 3) in BarCache(tmp_path).sessions_present("AAPL")
