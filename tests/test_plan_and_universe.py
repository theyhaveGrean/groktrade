from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from aegis.config import Settings
from aegis.data.pipeline import build_plan
from aegis.data.plan import assert_chunk_fits_page_limit, estimated_base_minutes, group_missing_sessions
from aegis.data.sessions import clamp_history_window, latest_requestable_session
from aegis.data.universe import Membership, load_universe, packaged_universe_path, resolve_universe_path, select_as_of
from aegis.errors import PlanWindowError

NY = ZoneInfo("America/New_York")


def test_default_chunk_stays_under_the_scan_cap():
    assert estimated_base_minutes(70) < 50_000
    assert_chunk_fits_page_limit(70)
    with pytest.raises(ValueError):
        assert_chunk_fits_page_limit(90)


def test_missing_sessions_do_not_redownload_cached_days():
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    groups = group_missing_sessions(sessions, {date(2024, 1, 3), date(2024, 1, 4)}, max_calendar_days=70)
    assert groups == [(date(2024, 1, 2), date(2024, 1, 2)), (date(2024, 1, 5), date(2024, 1, 8))]


def test_free_tier_excludes_today_and_clamps_lookback():
    now = datetime(2026, 10, 30, 15, 0, tzinfo=NY)  # Friday, session still open
    latest = latest_requestable_session(now, plan="free")
    assert latest == date(2026, 10, 29)
    start, end, warnings = clamp_history_window(
        date(2020, 1, 1),
        date(2026, 10, 30),
        latest_end=latest,
        lookback_days=730,
    )
    assert end == date(2026, 10, 29)
    assert start == date(2026, 10, 29).fromordinal(date(2026, 10, 29).toordinal() - 730)
    assert len(warnings) == 2


def test_paid_plan_includes_today_after_the_close():
    after_close = datetime(2026, 10, 30, 21, 30, tzinfo=timezone.utc)  # 17:30 ET
    before_close = datetime(2026, 10, 30, 19, 0, tzinfo=timezone.utc)  # 15:00 ET
    assert latest_requestable_session(after_close, plan="paid") == date(2026, 10, 30)
    assert latest_requestable_session(before_close, plan="paid") == date(2026, 10, 29)
    weekend = datetime(2026, 10, 31, 15, 0, tzinfo=NY)
    assert latest_requestable_session(weekend, plan="free") == date(2026, 10, 30)


def test_empty_window_is_an_error():
    with pytest.raises(PlanWindowError):
        clamp_history_window(
            date(2026, 10, 30),
            date(2026, 10, 30),
            latest_end=date(2026, 10, 29),
            lookback_days=730,
        )


def test_full_free_tier_backfill_is_about_fifty_two_minutes(tmp_path):
    now = datetime(2026, 10, 30, 18, 0, tzinfo=NY)
    settings = Settings(_env_file=None, data_cache_dir=tmp_path)
    plan = build_plan(settings, start=None, end=None, now=now)
    assert len(plan.symbols) == 20
    assert plan.start == date(2024, 10, 29)
    assert plan.end == date(2026, 10, 29)
    assert plan.bar_requests == 220
    assert plan.corporate_action_requests == 40
    assert plan.total_requests == 260
    assert plan.estimated_minutes == pytest.approx(52.0)


def test_starter_universe_has_twenty_liquid_names():
    path = resolve_universe_path(Settings(_env_file=None).universe_path)
    members = load_universe(path)
    assert len(members) == 20
    assert {member.symbol for member in members} >= {"AAPL", "MSFT", "NVDA", "JPM", "XOM"}
    assert all(member.end is None for member in members)
    packaged = load_universe(packaged_universe_path())
    assert [member.symbol for member in packaged] == [member.symbol for member in members]


def test_point_in_time_membership_keeps_delisted_names_until_their_end():
    members = (
        Membership("AAPL", date(1980, 12, 12), None),
        Membership("LEH", date(2000, 1, 1), date(2008, 9, 15), listing_mic="XNYS", name="Lehman"),
    )
    assert [member.symbol for member in select_as_of(members, date(2008, 9, 15))] == ["AAPL", "LEH"]
    assert [member.symbol for member in select_as_of(members, date(2008, 9, 16))] == ["AAPL"]
    assert select_as_of(members, date(1970, 1, 1)) == ()
