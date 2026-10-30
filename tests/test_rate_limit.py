from aegis.data.rate_limit import RateLimiter


def test_five_calls_per_minute_are_spaced_twelve_seconds():
    clock = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    limiter = RateLimiter(5, clock=lambda: clock["t"], sleep=sleep)
    for _ in range(5):
        limiter.acquire()
    assert slept == [12.0, 12.0, 12.0, 12.0]
    assert clock["t"] == 48.0


def test_penalize_delays_the_next_acquire():
    clock = {"t": 0.0}

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    limiter = RateLimiter(60, clock=lambda: clock["t"], sleep=sleep)
    limiter.acquire()
    limiter.penalize(30)
    limiter.acquire()
    assert clock["t"] == 30.0
