"""Blocking rate limiter shared by every Massive HTTP call.

The free tier allows 5 requests per minute. ``acquire`` spaces call *starts*
by ``60 / rate`` seconds. ``penalize`` pushes the next start further out after
a 429, so the retry loop does not immediately burn another call.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        calls_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive.")
        self.calls_per_minute = float(calls_per_minute)
        self.min_interval = 60.0 / self.calls_per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next - now
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
            self._next = now + self.min_interval

    def penalize(self, seconds: float) -> None:
        """Do not start another call until ``seconds`` from now, at the earliest."""
        if seconds < 0:
            raise ValueError("penalize seconds must be non-negative.")
        with self._lock:
            self._next = max(self._next, self._clock() + seconds)
