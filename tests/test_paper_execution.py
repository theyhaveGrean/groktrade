"""Paper execution with a fake Alpaca client. No test talks to the network."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from aegis.backtest.risk import RiskDecision, RiskLimits, get_risk_engine
from aegis.cli import app
from aegis.config import Settings
from aegis.errors import LiveTradingDisabled
from aegis.execution.broker import AlpacaPaperBroker, connect_paper_broker
from aegis.execution.feed import AlpacaIexFeed
from aegis.execution.guards import PAPER_TRADING_BASE_URL, assert_paper_execution_allowed
from aegis.execution.killswitch import arm_kill_file
from aegis.execution.models import AssetView, BracketOrder, MinuteBar, OrderView, PositionView
from aegis.execution.session import PaperSession, execute_kill
from aegis.execution.validate import PreTradeValidator

NY = ZoneInfo("America/New_York")
runner = CliRunner()


def _ts(hour: int, minute: int, day: int = 3) -> datetime:
    return datetime(2024, 1, day, hour, minute, tzinfo=NY)


def _settings(tmp_path, **overrides) -> Settings:
    values = dict(
        _env_file=None,
        trading_mode="paper",
        live_trading_enabled=False,
        alpaca_base_url=PAPER_TRADING_BASE_URL,
        alpaca_api_key="test-alpaca-key",
        alpaca_secret_key="test-alpaca-secret",
        paper_dir=tmp_path,
        risk_kill_switch=False,
        paper_reconcile_seconds=30,
        paper_poll_seconds=15,
    )
    values.update(overrides)
    return Settings(**values)


class FakeBroker:
    def __init__(self) -> None:
        self.equity = 100_000.0
        self.last_equity = 100_000.0
        self.buying_power = 1_000_000.0
        self.trading_blocked = False
        self.assets: dict[str, AssetView] = {}
        self._positions: list[PositionView] = []
        self._orders: list[OrderView] = []
        self.submitted: list[BracketOrder] = []
        self.cancel_count = 0
        self.flatten_count = 0
        self.flattened_symbols: list[str] = []
        self.fail_submit: Exception | None = None

    def account(self):
        from aegis.execution.models import AccountView

        return AccountView(self.equity, self.last_equity, self.buying_power, self.trading_blocked)

    def asset(self, symbol: str) -> AssetView:
        return self.assets.get(symbol, AssetView(symbol, True, "active"))

    def positions(self) -> list[PositionView]:
        return list(self._positions)

    def open_orders(self) -> list[OrderView]:
        return list(self._orders)

    def submit_bracket(self, order: BracketOrder) -> OrderView:
        if self.fail_submit is not None:
            raise self.fail_submit
        self.submitted.append(order)
        parent_id = f"entry-{len(self.submitted)}"
        stop = OrderView(
            id=f"stop-{parent_id}",
            symbol=order.symbol,
            qty=order.quantity,
            side="sell",
            status="accepted",
            role="stop",
            stop_price=order.stop_price,
            order_class="bracket",
        )
        target = OrderView(
            id=f"target-{parent_id}",
            symbol=order.symbol,
            qty=order.quantity,
            side="sell",
            status="accepted",
            role="target",
            limit_price=order.target_price,
            order_class="bracket",
        )
        side = "buy" if order.side == "long" else "sell"
        parent = OrderView(
            id=parent_id,
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            status="accepted",
            role="entry",
            order_class="bracket",
            client_order_id=order.client_order_id,
            legs=(stop, target),
        )
        self._orders.append(parent)
        return parent

    def cancel_all_orders(self) -> None:
        self.cancel_count += 1
        self._orders.clear()

    def flatten_all(self) -> None:
        self.flatten_count += 1
        self._orders.clear()
        self._positions.clear()

    def flatten_symbol(self, symbol: str) -> None:
        self.flattened_symbols.append(symbol)
        self._orders = [order for order in self._orders if order.symbol != symbol]
        self._positions = [position for position in self._positions if position.symbol != symbol]


def _session(tmp_path, broker: FakeBroker | None = None, **overrides) -> tuple[PaperSession, FakeBroker]:
    broker = broker or FakeBroker()
    clock = overrides.pop("clock", lambda: _ts(10, 0))
    settings = _settings(tmp_path, **overrides)
    session = PaperSession(
        settings,
        broker,
        ["AAPL"],
        clock=clock,
        fast_period=overrides.get("fast_period", 20),
        slow_period=overrides.get("slow_period", 50),
        atr_period=overrides.get("atr_period", 14),
    )
    return session, broker


def _entry(session: PaperSession, **overrides) -> RiskDecision:
    values = dict(symbol="AAPL", ts=_ts(10, 0), side="long", entry=100.0, atr=5.0)
    values.update(overrides)
    return session.consider_entry(**values)


@dataclass
class Refusal:
    reason: str
    flatten: bool = False
    ts: datetime | None = None
    side: str = "long"
    entry: float = 100.0
    atr: float | None = 5.0
    equity: float | None = None
    last_equity: float | None = None
    buying_power: float | None = None
    trading_blocked: bool = False
    asset: AssetView | None = None
    positions: tuple[PositionView, ...] = ()
    orders: tuple[OrderView, ...] = ()
    trip: str | None = None


REFUSALS = [
    Refusal("kill switch", flatten=True, trip="kill switch"),
    Refusal("daily loss cutoff", flatten=True, equity=95_000.0, last_equity=100_000.0),
    Refusal("flatten window", flatten=True, ts=_ts(15, 55)),
    Refusal("before entry window", ts=_ts(9, 34)),
    Refusal("after entry window", ts=_ts(15, 31)),
    Refusal("weekend", ts=_ts(10, 0, day=6)),
    Refusal(
        "already in position",
        positions=(PositionView("AAPL", 10),),
    ),
    Refusal(
        "max simultaneous positions",
        orders=tuple(
            OrderView(id=str(index), symbol=symbol, qty=1, side="buy", status="accepted")
            for index, symbol in enumerate(["A", "B", "C", "D", "E"])
        ),
    ),
    Refusal("atr or entry is not positive", atr=None),
    Refusal("atr or entry is not positive", atr=0),
    Refusal("atr or entry is not positive", entry=0),
    Refusal("atr or entry is not positive", entry=-1),
    Refusal("unsupported side", side="flat"),
    Refusal("stop distance rounds to zero", entry=100, atr=0.001),
    Refusal("position size below 1 share", equity=100, last_equity=100),
    Refusal("insufficient buying power", buying_power=1_000),
    Refusal("symbol not tradable", asset=AssetView("AAPL", False, "active")),
    Refusal("symbol not active", asset=AssetView("AAPL", True, "inactive")),
    Refusal("account trading is blocked", trading_blocked=True),
]


@pytest.mark.parametrize("case", REFUSALS, ids=[case.reason.replace(" ", "-") + f"-{index}" for index, case in enumerate(REFUSALS)])
def test_every_risk_and_broker_refusal_skips_the_order(tmp_path, case: Refusal):
    session, broker = _session(tmp_path)
    if case.trip:
        session.risk.trip(case.trip)
    if case.equity is not None:
        broker.equity = case.equity
    if case.last_equity is not None:
        broker.last_equity = case.last_equity
    if case.buying_power is not None:
        broker.buying_power = case.buying_power
    broker.trading_blocked = case.trading_blocked
    if case.asset is not None:
        broker.assets["AAPL"] = case.asset
    broker._positions.extend(case.positions)
    broker._orders.extend(case.orders)
    decision = _entry(
        session,
        ts=case.ts or _ts(10, 0),
        side=case.side,
        entry=case.entry,
        atr=case.atr,
    )
    assert decision.approved is False
    assert decision.reason == case.reason
    assert decision.flatten is case.flatten
    assert broker.submitted == []
    if case.flatten:
        assert broker.flatten_count >= 1
        assert broker.cancel_count >= 1
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert case.reason in text
    assert "test-alpaca-key" not in text
    assert "test-alpaca-secret" not in text


def test_live_mode_live_url_and_live_flag_refuse_to_start(tmp_path):
    with pytest.raises(LiveTradingDisabled):
        assert_paper_execution_allowed(_settings(tmp_path, trading_mode="live"))
    with pytest.raises(LiveTradingDisabled):
        assert_paper_execution_allowed(_settings(tmp_path, live_trading_enabled=True))
    with pytest.raises(LiveTradingDisabled):
        assert_paper_execution_allowed(_settings(tmp_path, alpaca_base_url="https://api.alpaca.markets"))
    with pytest.raises(LiveTradingDisabled):
        assert_paper_execution_allowed(_settings(tmp_path, alpaca_base_url="https://paper-api.alpaca.markets/v2"))
    with pytest.raises(LiveTradingDisabled):
        assert_paper_execution_allowed(_settings(tmp_path, alpaca_base_url="http://paper-api.alpaca.markets"))
    assert_paper_execution_allowed(_settings(tmp_path, alpaca_base_url="https://paper-api.alpaca.markets/"))


def test_connect_forces_the_paper_client_and_never_builds_one_for_a_live_url(tmp_path, monkeypatch):
    created: list[dict] = []

    class Boom:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            raise AssertionError("client constructed")

    monkeypatch.setattr("aegis.execution.broker.TradingClient", Boom)
    with pytest.raises(LiveTradingDisabled):
        connect_paper_broker(_settings(tmp_path, alpaca_base_url="https://api.alpaca.markets"))
    assert created == []

    class FakeClient:
        def __init__(self, api_key, secret_key, paper=False, url_override=None, **kwargs):
            created.append({"paper": paper, "url": url_override, "key": api_key, "secret": secret_key})

    monkeypatch.setattr("aegis.execution.broker.TradingClient", FakeClient)
    connect_paper_broker(_settings(tmp_path))
    assert created[-1]["paper"] is True
    assert created[-1]["url"] == PAPER_TRADING_BASE_URL
    assert created[-1]["key"] == "test-alpaca-key"


def test_wrapper_rejects_a_client_pointed_at_live_even_if_paper_flag_was_set():
    from alpaca.trading.client import TradingClient

    live = TradingClient("key", "secret", paper=False)
    with pytest.raises(LiveTradingDisabled):
        AlpacaPaperBroker(live)
    overridden = TradingClient("key", "secret", paper=True, url_override="https://api.alpaca.markets")
    with pytest.raises(LiveTradingDisabled):
        AlpacaPaperBroker(overridden)
    paper = TradingClient("key", "secret", paper=True)
    assert AlpacaPaperBroker(paper).base_url == PAPER_TRADING_BASE_URL


def test_bracket_request_is_a_day_market_bracket_with_the_stop_and_target():
    from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce

    class Client:
        def __init__(self) -> None:
            self.order_data = None
            self.nested = None
            self._base_url = PAPER_TRADING_BASE_URL

        def submit_order(self, order_data):
            self.order_data = order_data
            return SimpleNamespace(
                id="parent-1",
                client_order_id=order_data.client_order_id,
                symbol=order_data.symbol,
                qty=order_data.qty,
                side=order_data.side,
                status="accepted",
                order_class=order_data.order_class,
                legs=[],
                limit_price=None,
                stop_price=None,
            )

        def get_order_by_id(self, order_id, request):
            self.nested = request.nested
            stop = SimpleNamespace(
                id="stop-1",
                symbol="AAPL",
                qty="10",
                side=OrderSide.SELL,
                status="held",
                order_class="bracket",
                client_order_id="stop",
                legs=[],
                limit_price=None,
                stop_price="90",
            )
            target = SimpleNamespace(
                id="target-1",
                symbol="AAPL",
                qty="10",
                side=OrderSide.SELL,
                status="held",
                order_class="bracket",
                client_order_id="target",
                legs=[],
                limit_price="125",
                stop_price=None,
            )
            return SimpleNamespace(
                id=order_id,
                client_order_id="aegis-1",
                symbol="AAPL",
                qty="10",
                side=OrderSide.BUY,
                status="accepted",
                order_class=OrderClass.BRACKET,
                legs=[stop, target],
                limit_price=None,
                stop_price=None,
            )

    client = Client()
    broker = AlpacaPaperBroker(client)
    view = broker.submit_bracket(
        BracketOrder(
            symbol="AAPL",
            side="long",
            quantity=10,
            stop_price=90.0,
            target_price=125.0,
            client_order_id="aegis-1",
            entry_price=100.0,
        )
    )
    request = client.order_data
    assert request.order_class == OrderClass.BRACKET
    assert request.type == OrderType.MARKET
    assert request.time_in_force == TimeInForce.DAY
    assert request.side == OrderSide.BUY
    assert request.extended_hours is False
    assert request.stop_loss.stop_price == 90.0
    assert request.take_profit.limit_price == 125.0
    assert client.nested is True
    assert view.id == "parent-1"
    assert {leg.role for leg in view.legs} == {"stop", "target"}


def test_defensive_caps_reject_a_decision_the_engine_should_not_have_approved(tmp_path):
    session, broker = _session(tmp_path)

    class StubRisk:
        limits = RiskLimits()
        killed = False

        def __init__(self, decision: RiskDecision) -> None:
            self.decision = decision

        def evaluate_entry(self, **kwargs):
            return self.decision

    bad_reward = RiskDecision(approved=True, reason="approved", quantity=10, stop_price=90, target_price=91)
    session.validator = PreTradeValidator(StubRisk(bad_reward))  # type: ignore[arg-type]
    assert _entry(session).reason == "reward-to-risk below 2.5"
    too_much_risk = RiskDecision(approved=True, reason="approved", quantity=101, stop_price=90, target_price=125)
    session.validator = PreTradeValidator(StubRisk(too_much_risk))  # type: ignore[arg-type]
    assert _entry(session).reason == "risk exceeds 1% of equity"
    too_much_notional = RiskDecision(approved=True, reason="approved", quantity=300, stop_price=99.9, target_price=100.25)
    session.validator = PreTradeValidator(StubRisk(too_much_notional))  # type: ignore[arg-type]
    assert _entry(session).reason == "notional exceeds 25% of equity"
    assert broker.submitted == []


def test_approved_bracket_uses_broker_equity_and_the_risk_engine_prices(tmp_path):
    session, broker = _session(tmp_path)
    assert get_risk_engine() is session.risk
    decision = _entry(session)
    assert decision.approved
    assert decision.quantity == 100
    assert decision.stop_price == 90
    assert decision.target_price == 125
    order = broker.submitted[0]
    assert order.side == "long"
    assert order.quantity == 100
    assert order.stop_price == 90
    assert order.target_price == 125
    risk = abs(order.entry_price - order.stop_price)
    assert abs(order.target_price - order.entry_price) / risk >= 2.5
    assert order.quantity * risk <= 100_000 * 0.01 + 1e-6
    assert order.quantity * order.entry_price <= 100_000 * 0.25 + 1e-6
    assert session.state.mismatch is False
    assert "entry-1" in session.state.orders
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert '"approved":true' in text
    assert "test-alpaca-key" not in text


def test_daily_loss_uses_broker_equity_and_latches_for_the_session(tmp_path):
    session, broker = _session(tmp_path)
    session.state.session_date = "2024-01-03"
    session.state.day_start_equity = 100_000.0
    broker.equity = 95_000.0
    broker.last_equity = 200_000.0
    first = _entry(session)
    assert first.reason == "daily loss cutoff"
    assert broker.flatten_count >= 1
    broker.equity = 100_000.0
    second = _entry(session)
    assert second.reason == "daily loss cutoff"
    assert broker.submitted == []
    session.state.halted_session = "2024-01-02"
    session.state.session_date = "2024-01-02"
    session.state.day_start_equity = 100_000.0
    broker.last_equity = 100_000.0
    broker.equity = 100_000.0
    assert _entry(session, ts=_ts(10, 0)).approved


def test_kill_file_and_env_flatten_and_block_entries(tmp_path):
    broker = FakeBroker()
    broker._positions.append(PositionView("AAPL", 8))
    settings = _settings(tmp_path)
    arm_kill_file(settings, "operator halt")
    session = PaperSession(settings, broker, ["AAPL"], clock=lambda: _ts(10, 0))
    session.poll(_ts(10, 0))
    assert broker.flatten_count >= 1
    assert broker.cancel_count >= 1
    assert broker.positions() == []
    decision = _entry(session)
    assert decision.reason == "operator halt"
    assert decision.flatten is True
    assert broker.submitted == []

    env_session, env_broker = _session(tmp_path / "env", risk_kill_switch=True)
    killed = _entry(env_session)
    assert killed.reason == "kill switch"
    assert killed.flatten is True
    assert env_broker.submitted == []
    assert env_broker.flatten_count >= 1


def test_adapter_parses_string_account_and_position_fields():
    class Client:
        _base_url = PAPER_TRADING_BASE_URL

        def get_account(self):
            return SimpleNamespace(
                equity="100000.50",
                last_equity="101000",
                buying_power="200000",
                trading_blocked=False,
                account_blocked=True,
            )

        def get_all_positions(self):
            return [
                SimpleNamespace(symbol="AAPL", qty="4", side="long", avg_entry_price="100.5"),
                SimpleNamespace(symbol="MSFT", qty="2", side="short", avg_entry_price="300"),
            ]

        def get_orders(self, filter=None):
            assert filter is not None and filter.nested is True
            return []

    broker = AlpacaPaperBroker(Client())
    account = broker.account()
    assert account.equity == 100000.50
    assert account.last_equity == 101000
    assert account.buying_power == 200000
    assert account.trading_blocked is True
    positions = {position.symbol: position.qty for position in broker.positions()}
    assert positions == {"AAPL": 4, "MSFT": -2}
    assert broker.open_orders() == []


def test_flatten_poll_runs_when_the_account_refresh_fails(tmp_path):
    session, broker = _session(tmp_path, clock=lambda: _ts(15, 55))
    broker._positions.append(PositionView("AAPL", 1))

    def down():
        raise RuntimeError("account down")

    broker.account = down  # type: ignore[method-assign]
    session.poll()
    assert broker.flatten_count >= 1
    assert broker.positions() == []


def test_poll_flattens_at_1555_without_a_new_bar(tmp_path):
    session, broker = _session(tmp_path, clock=lambda: _ts(15, 55))
    broker._positions.append(PositionView("AAPL", 2))
    session.poll()
    assert broker.flatten_count >= 1
    assert broker.cancel_count >= 1
    assert broker.positions() == []
    assert broker.submitted == []


def test_wall_clock_flatten_blocks_a_stale_bar(tmp_path):
    session, broker = _session(tmp_path, clock=lambda: _ts(15, 55))
    broker._positions.append(PositionView("AAPL", 2))
    decision = _entry(session, ts=_ts(10, 0))
    assert decision.reason == "flatten"
    assert broker.submitted == []
    assert broker.flatten_count >= 1
    assert broker.positions() == []


def test_reconciliation_mismatch_alerts_and_blocks_the_next_entry(tmp_path):
    session, broker = _session(tmp_path)
    session.state.positions["AAPL"] = {"qty": 10, "entry": 100, "stop": 90, "target": 125}
    broker._positions.append(PositionView("AAPL", 12))
    report = session.reconcile_now(_ts(10, 0))
    assert report.mismatch
    assert "qty internal 10 broker 12" in report.summary()
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "reconcile_mismatch" in text

    broker._positions.clear()
    broker._positions.append(PositionView("MSFT", 4))
    session.state.positions.clear()
    unknown = session.reconcile_now(_ts(10, 1))
    assert any("MSFT" in problem for problem in unknown.problems)
    blocked = _entry(session)
    assert blocked.reason == "broker reconciliation mismatch"
    assert broker.submitted == []

    broker._positions.clear()
    broker._orders.append(OrderView(id="foreign", symbol="NVDA", qty=1, side="buy", status="new"))
    session.state.orders.clear()
    session.state.positions.clear()
    foreign = session.reconcile_now(_ts(10, 2))
    assert any("foreign" in problem for problem in foreign.problems)

    broker._orders.clear()
    session.state.orders["missing"] = {
        "id": "missing",
        "symbol": "AAPL",
        "qty": 3,
        "side": "buy",
        "status": "accepted",
        "role": "entry",
    }
    missing = session.reconcile_now(_ts(10, 3))
    assert any("missing" in problem for problem in missing.problems)


def test_a_filled_entry_that_matches_broker_qty_is_not_a_mismatch(tmp_path):
    session, broker = _session(tmp_path)
    session.state.orders["entry-1"] = {
        "id": "entry-1",
        "symbol": "AAPL",
        "qty": 10,
        "side": "buy",
        "status": "accepted",
        "role": "entry",
        "entry": 100,
        "stop": 90,
        "target": 125,
    }
    broker._positions.append(PositionView("AAPL", 10))
    report = session.reconcile_now(_ts(10, 0))
    assert report.mismatch is False
    assert session.state.positions["AAPL"]["qty"] == 10
    assert session.state.orders["entry-1"]["status"] == "filled"


def test_broker_rejection_is_audited_without_the_secret(tmp_path):
    session, broker = _session(tmp_path)
    broker.fail_submit = RuntimeError("rejected test-alpaca-key")
    decision = _entry(session)
    assert decision.reason == "broker rejected order"
    assert session.state.orders == {}
    text = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "test-alpaca-key" not in text
    assert "[redacted]" in text


def test_ema_cross_submits_once_and_a_bearish_cross_flattens(tmp_path):
    broker = FakeBroker()
    settings = _settings(tmp_path)
    session = PaperSession(
        settings,
        broker,
        ["AAPL"],
        clock=lambda: _ts(10, 0),
        fast_period=2,
        slow_period=4,
        atr_period=2,
    )
    cursor = _ts(10, 0)
    for _ in range(5):
        session.on_bar(_bar("AAPL", cursor, 100))
        cursor += timedelta(minutes=1)
    assert broker.submitted == []
    session.on_bar(_bar("AAPL", cursor, 110))
    cursor += timedelta(minutes=1)
    assert len(broker.submitted) == 1
    order = broker.submitted[0]
    assert order.side == "long"
    assert order.stop_price < order.entry_price < order.target_price
    risk = order.entry_price - order.stop_price
    assert (order.target_price - order.entry_price) / risk >= 2.5 - 1e-9
    for price in (110, 110, 110):
        session.on_bar(_bar("AAPL", cursor, price))
        cursor += timedelta(minutes=1)
    assert len(broker.submitted) == 1
    session.on_bar(_bar("AAPL", cursor, 80))
    assert broker.flattened_symbols == ["AAPL"]
    assert len(broker.submitted) == 1


def test_warmup_does_not_trade_a_cross_that_already_happened(tmp_path):
    broker = FakeBroker()
    settings = _settings(tmp_path)
    session = PaperSession(
        settings,
        broker,
        ["AAPL"],
        clock=lambda: _ts(10, 0),
        fast_period=2,
        slow_period=4,
        atr_period=2,
    )
    cursor = _ts(10, 0)
    history = []
    for _ in range(5):
        history.append(_bar("AAPL", cursor, 100))
        cursor += timedelta(minutes=1)
    history.append(_bar("AAPL", cursor, 110))
    cursor += timedelta(minutes=1)
    live = _bar("AAPL", cursor, 110)

    class Feed:
        def historical_bars(self, symbols, start, end):
            return history

        def stream(self, symbols, on_bar):
            on_bar(live)

    session.serve(Feed())
    assert broker.submitted == []


def test_iex_feed_is_forced_and_refuses_a_live_trading_url(tmp_path, monkeypatch):
    from alpaca.data.enums import Adjustment, DataFeed

    captured: dict = {}

    class Stream:
        def __init__(self, key, secret, feed=None, **kwargs):
            captured["feed"] = feed

        def subscribe_bars(self, handler, *symbols):
            captured["symbols"] = symbols

        def run(self):
            captured["ran"] = True

    class History:
        def __init__(self, key, secret, **kwargs):
            pass

        def get_stock_bars(self, request):
            captured["request"] = request
            bar = SimpleNamespace(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 3, 15, 0, tzinfo=timezone.utc),
                open=1,
                high=2,
                low=0.5,
                close=1.5,
                volume=10,
            )
            return SimpleNamespace(data={"AAPL": [bar]})

    feed = AlpacaIexFeed("k", "s", stream_factory=Stream, history_factory=History)
    bars = feed.historical_bars(["AAPL"], _ts(10, 0), _ts(11, 0))
    assert captured["request"].feed == DataFeed.IEX
    assert captured["request"].adjustment == Adjustment.RAW
    assert captured["request"].timeframe.amount == 1
    assert bars[0].close == 1.5
    assert bars[0].ts.tzinfo is not None
    feed.stream(["AAPL", "MSFT"], lambda bar: None)
    assert captured["feed"] == DataFeed.IEX
    assert captured["symbols"] == ("AAPL", "MSFT")
    assert captured["ran"] is True

    def boom(*args, **kwargs):
        raise AssertionError("stream constructed")

    monkeypatch.setattr("aegis.execution.feed.StockDataStream", boom)
    from aegis.execution.feed import connect_iex_feed

    with pytest.raises(LiveTradingDisabled):
        connect_iex_feed(_settings(tmp_path, alpaca_base_url="https://api.alpaca.markets"))


def test_cli_run_status_and_kill(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_DIR", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "test-alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-alpaca-secret")
    monkeypatch.setenv("ALPACA_BASE_URL", PAPER_TRADING_BASE_URL)
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("RISK_KILL_SWITCH", "false")

    refused = runner.invoke(app, ["paper", "run", "--strategy", "macd", "--symbols", "AAPL"])
    assert refused.exit_code == 1
    assert "ema_crossover" in (refused.stdout + refused.stderr)

    broker = FakeBroker()
    broker.equity = 101_250.0
    streamed: list[list[str]] = []

    class Feed:
        def historical_bars(self, symbols, start, end):
            return []

        def stream(self, symbols, on_bar):
            streamed.append(list(symbols))

    monkeypatch.setattr("aegis.execution.broker.connect_paper_broker", lambda settings: broker)
    monkeypatch.setattr("aegis.execution.feed.connect_iex_feed", lambda settings: Feed())
    ran = runner.invoke(app, ["paper", "run", "--strategy", "ema_crossover", "--symbols", "AAPL"])
    assert ran.exit_code == 0, ran.stdout + ran.stderr
    assert streamed == [["AAPL"]]

    status = runner.invoke(app, ["paper", "status"])
    assert status.exit_code == 0, status.stdout + status.stderr
    assert "101250.00" in status.stdout or "101250" in status.stdout
    assert "kill_switch: off" in status.stdout

    broker._positions.append(PositionView("MSFT", 4))
    mismatched = runner.invoke(app, ["paper", "status", "--json"])
    assert mismatched.exit_code == 2
    payload = json.loads(mismatched.stdout)
    assert payload["mismatch"] is True
    assert payload["alpaca_base_url"] == PAPER_TRADING_BASE_URL

    broker._positions.clear()
    broker._positions.append(PositionView("AAPL", 6))
    killed = runner.invoke(app, ["paper", "kill"])
    assert killed.exit_code == 0, killed.stdout + killed.stderr
    assert "Kill switch armed" in killed.stdout
    assert broker.flatten_count >= 1
    assert broker.positions() == []
    kill_file = tmp_path / "KILL"
    assert kill_file.exists()
    assert "aegis paper kill" in kill_file.read_text(encoding="utf-8")
    assert "test-alpaca-key" not in (tmp_path / "audit.jsonl").read_text(encoding="utf-8")


def test_cli_kill_refuses_a_live_base_url(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_DIR", str(tmp_path))
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")

    def boom(*args, **kwargs):
        raise AssertionError("broker constructed")

    monkeypatch.setattr("aegis.execution.broker.TradingClient", boom)
    result = runner.invoke(app, ["paper", "kill"])
    assert result.exit_code == 1
    assert "paper-api.alpaca.markets" in (result.stdout + result.stderr)
    assert not (tmp_path / "KILL").exists()


def test_execute_kill_reports_an_incomplete_flatten(tmp_path):
    broker = FakeBroker()

    def leave_a_position() -> None:
        broker.flatten_count += 1

    broker.flatten_all = leave_a_position  # type: ignore[method-assign]
    broker._positions.append(PositionView("AAPL", 1))
    with pytest.raises(Exception, match="incompletely"):
        execute_kill(_settings(tmp_path), broker)
    assert (tmp_path / "KILL").exists()


def _bar(symbol: str, ts: datetime, close: float) -> MinuteBar:
    return MinuteBar(symbol=symbol, ts=ts, open=close, high=close + 0.5, low=close - 0.5, close=close, volume=1_000)
