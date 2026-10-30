"""Runtime configuration loaded from the environment.

Secrets are never hard-coded. A missing ``.env`` is fine: the process
environment is the source of truth. :class:`pydantic.SecretStr` keeps API keys
out of ``repr`` and logs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aegis.errors import LiveTradingDisabled, MissingCredentials
from aegis.holdout import DateRange

FREE_TIER_REQUESTS_PER_MINUTE = 5
FREE_TIER_LOOKBACK_DAYS = 730
# 70 calendar days is about 48 NYSE sessions. Massive scans ~960 extended-hours
# minutes per session, so one chunk stays near 46,000 base minutes, under the
# 50,000 ``limit`` cap. One HTTP call then returns the whole chunk.
DEFAULT_CHUNK_CALENDAR_DAYS = 70
MAX_AGGREGATE_PAGE_LIMIT = 50_000


class Settings(BaseSettings):
    """Environment-backed settings. Field names match the env vars in ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: SecretStr | None = None
    massive_api_key: SecretStr | None = None
    massive_base_url: str = "https://api.massive.com"
    massive_plan: Literal["free", "paid"] = "free"
    massive_rate_limit_per_minute: float = FREE_TIER_REQUESTS_PER_MINUTE
    massive_max_lookback_days: int = FREE_TIER_LOOKBACK_DAYS
    massive_chunk_calendar_days: int = DEFAULT_CHUNK_CALENDAR_DAYS
    massive_page_limit: int = MAX_AGGREGATE_PAGE_LIMIT
    massive_max_retries: int = 5
    massive_timeout_seconds: float = 30.0

    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    trading_mode: Literal["paper", "live"] = "paper"
    live_trading_enabled: bool = False
    paper_dir: Path = Path("data/paper")
    paper_kill_file: Path | None = None
    paper_reconcile_seconds: float = 30.0
    paper_poll_seconds: float = 15.0

    data_cache_dir: Path = Path("data/cache")
    universe_path: Path = Path("config/universe.yaml")

    holdout_start: date = date(2026, 7, 1)
    holdout_end: date = date(2026, 10, 31)

    risk_kill_switch: bool = False
    starting_equity: float = 100_000.0
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0

    nautilus_log_level: str = "ERROR"

    @model_validator(mode="after")
    def _check_bounds(self) -> Settings:
        if self.holdout_start > self.holdout_end:
            raise ValueError("HOLDOUT_START must be on or before HOLDOUT_END.")
        if self.massive_rate_limit_per_minute <= 0:
            raise ValueError("MASSIVE_RATE_LIMIT_PER_MINUTE must be positive.")
        if self.massive_max_lookback_days < 1:
            raise ValueError("MASSIVE_MAX_LOOKBACK_DAYS must be at least 1.")
        if self.massive_chunk_calendar_days < 1:
            raise ValueError("MASSIVE_CHUNK_CALENDAR_DAYS must be at least 1.")
        if not 1 <= self.massive_page_limit <= MAX_AGGREGATE_PAGE_LIMIT:
            raise ValueError(
                f"MASSIVE_PAGE_LIMIT must be between 1 and {MAX_AGGREGATE_PAGE_LIMIT}."
            )
        if self.starting_equity <= 0:
            raise ValueError("STARTING_EQUITY must be positive.")
        if self.commission_per_share < 0 or self.slippage_bps < 0:
            raise ValueError("Commission and slippage cannot be negative.")
        if self.paper_reconcile_seconds <= 0 or self.paper_poll_seconds <= 0:
            raise ValueError("Paper reconcile and poll intervals must be positive.")
        return self

    @property
    def holdout(self) -> DateRange:
        return DateRange(self.holdout_start, self.holdout_end)

    def require_massive_api_key(self) -> str:
        if self.massive_api_key is None:
            raise MissingCredentials(
                "MASSIVE_API_KEY is not set. Add it to the environment or a local .env file. "
                "The synthetic backtest (`aegis backtest demo`) does not need a key."
            )
        value = self.massive_api_key.get_secret_value().strip()
        if not value:
            raise MissingCredentials(
                "MASSIVE_API_KEY is empty. The synthetic backtest does not need a key."
            )
        return value

    def require_alpaca_credentials(self) -> tuple[str, str]:
        """Return the paper key pair. Empty values are treated as missing."""
        if self.alpaca_api_key is None or self.alpaca_secret_key is None:
            raise MissingCredentials(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must both be set. "
                "They are read from the environment only."
            )
        key = self.alpaca_api_key.get_secret_value().strip()
        secret = self.alpaca_secret_key.get_secret_value().strip()
        if not key or not secret:
            raise MissingCredentials(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must both be non-empty."
            )
        return key, secret

    def assert_paper_mode(self) -> None:
        """This build has no live-order path. Paper mode is the only legal state."""
        if self.trading_mode != "paper" or self.live_trading_enabled:
            raise LiveTradingDisabled(
                "Live trading is not implemented. Keep TRADING_MODE=paper and "
                "LIVE_TRADING_ENABLED=false. A later phase requires a separate "
                "human approval before any live capital."
            )

    def free_tier_rate_warning(self) -> str | None:
        if self.massive_plan == "free" and self.massive_rate_limit_per_minute > FREE_TIER_REQUESTS_PER_MINUTE:
            return (
                "MASSIVE_PLAN=free but MASSIVE_RATE_LIMIT_PER_MINUTE is above 5. "
                "The free tier will answer with 429s; the client backs off, which "
                "makes the backfill slower rather than faster."
            )
        return None


def load_settings(env_file: str | None = ".env") -> Settings:
    """Load settings. Pass ``env_file=None`` in tests to ignore a developer ``.env``."""
    if env_file is None:
        return Settings(_env_file=None)
    return Settings(_env_file=env_file)
