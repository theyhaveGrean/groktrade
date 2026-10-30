from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from aegis.backtest.risk import PortfolioSnapshot, RiskEngine, RiskLimits
from aegis.backtest.walkforward import walk_forward_splits
from aegis.errors import HoldoutViolation
from aegis.holdout import DateRange

NY = ZoneInfo("America/New_York")
HOLD_OUT = DateRange(date(2026, 7, 1), date(2026, 10, 31))


def _ts(year, month, day, hour, minute) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY)


def _snapshot(**overrides) -> PortfolioSnapshot:
    base = dict(equity=100_000.0, day_start_equity=100_000.0, open_symbols=frozenset())
    base.update(overrides)
    return PortfolioSnapshot(**base)


def test_position_size_respects_one_percent_risk_and_twenty_five_percent_notional():
    engine = RiskEngine()
    # Wide stop: 1% risk binds (1000 / 10 = 100 shares), notional would allow 250.
    wide = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 10, 0), side="long", entry=100, atr=5, snapshot=_snapshot()
    )
    assert wide.approved
    assert wide.quantity == 100
    assert wide.quantity * abs(100 - wide.stop_price) <= 1000 + 1e-6
    # Tight stop: 25% notional binds.
    tight = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 10, 0), side="long", entry=100, atr=0.05, snapshot=_snapshot()
    )
    assert tight.approved
    assert tight.quantity == 250
    assert tight.quantity * 100 <= 25_000 + 1e-6


def test_reward_to_risk_stays_at_least_two_and_a_half():
    engine = RiskEngine()
    decision = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 10, 0), side="long", entry=50, atr=0.37, snapshot=_snapshot()
    )
    assert decision.approved
    risk = abs(50 - decision.stop_price)
    reward = abs(decision.target_price - 50)
    assert reward / risk >= 2.5 - 1e-9
    short = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 10, 0), side="short", entry=50, atr=0.4, snapshot=_snapshot()
    )
    assert short.approved
    assert short.stop_price > 50
    assert short.target_price < 50


def test_entry_window_flatten_and_weekend_use_eastern_time():
    engine = RiskEngine()
    assert not engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 9, 34), side="long", entry=100, atr=1, snapshot=_snapshot()
    ).approved
    assert engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 9, 35), side="long", entry=100, atr=1, snapshot=_snapshot()
    ).approved
    assert engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 15, 30), side="long", entry=100, atr=1, snapshot=_snapshot()
    ).approved
    late = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 15, 31), side="long", entry=100, atr=1, snapshot=_snapshot()
    )
    assert not late.approved
    assert late.reason == "after entry window"
    flatten = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 15, 55), side="long", entry=100, atr=1, snapshot=_snapshot()
    )
    assert flatten.flatten and not flatten.approved
    assert not engine.must_flatten(_ts(2024, 1, 3, 15, 54))
    assert engine.must_flatten(_ts(2024, 1, 3, 15, 55))
    weekend = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 6, 10, 0), side="long", entry=100, atr=1, snapshot=_snapshot()
    )
    assert weekend.reason == "weekend"
    # July is EDT. 13:35 UTC is 09:35 ET; 13:34 UTC is 09:34 ET.
    assert engine.entry_window_open(datetime(2024, 7, 3, 13, 35, tzinfo=timezone.utc))[0]
    assert not engine.entry_window_open(datetime(2024, 7, 3, 13, 34, tzinfo=timezone.utc))[0]
    with pytest.raises(ValueError):
        engine.entry_window_open(datetime(2024, 1, 3, 15, 0))


def test_max_positions_daily_loss_and_kill_switch_are_separate():
    engine = RiskEngine()
    blocked = engine.evaluate_entry(
        symbol="NEW",
        ts=_ts(2024, 1, 3, 10, 0),
        side="long",
        entry=100,
        atr=1,
        snapshot=_snapshot(open_symbols=frozenset({"A", "B", "C", "D", "E"})),
    )
    assert blocked.reason == "max simultaneous positions"
    already = engine.evaluate_entry(
        symbol="A",
        ts=_ts(2024, 1, 3, 10, 0),
        side="long",
        entry=100,
        atr=1,
        snapshot=_snapshot(open_symbols=frozenset({"A"})),
    )
    assert already.reason == "already in position"
    at_cutoff = _snapshot(equity=95_000, day_start_equity=100_000)
    assert engine.daily_loss_breached(at_cutoff)
    loss = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 10, 0), side="long", entry=100, atr=1, snapshot=at_cutoff
    )
    assert loss.reason == "daily loss cutoff" and loss.flatten
    assert engine.killed is False
    just_inside = _snapshot(equity=95_000.01, day_start_equity=100_000)
    assert not engine.daily_loss_breached(just_inside)
    engine.trip("desk")
    killed = engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 10, 0), side="long", entry=100, atr=1, snapshot=_snapshot()
    )
    assert killed.reason == "desk" and killed.flatten
    engine.reset()
    assert engine.evaluate_entry(
        symbol="AAPL", ts=_ts(2024, 1, 3, 10, 0), side="long", entry=100, atr=1, snapshot=_snapshot()
    ).approved
    armed = RiskEngine(RiskLimits(kill_switch=True))
    assert armed.killed


def test_tiny_account_cannot_round_up_to_a_share():
    engine = RiskEngine()
    decision = engine.evaluate_entry(
        symbol="AAPL",
        ts=_ts(2024, 1, 3, 10, 0),
        side="long",
        entry=80,
        atr=5,
        snapshot=_snapshot(equity=100, day_start_equity=100),
    )
    assert not decision.approved
    assert decision.quantity == 0


def test_walk_forward_refuses_the_holdout_and_builds_embargoed_folds():
    folds = walk_forward_splits(
        date(2024, 1, 1),
        date(2024, 6, 1),
        train_days=30,
        test_days=10,
        embargo_days=1,
        holdout=HOLD_OUT,
    )
    assert folds
    first = folds[0]
    assert first.train_start == date(2024, 1, 1)
    assert (first.train_end - first.train_start).days + 1 == 30
    assert (first.test_start - first.train_end).days == 2  # one embargo day between them
    assert (first.test_end - first.test_start).days + 1 == 10
    assert all(fold.test_end <= date(2024, 6, 1) for fold in folds)
    with pytest.raises(HoldoutViolation):
        walk_forward_splits(
            date(2026, 6, 1),
            date(2026, 8, 1),
            train_days=20,
            test_days=5,
            embargo_days=1,
            holdout=HOLD_OUT,
        )
