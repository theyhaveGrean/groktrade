"""Start-up checks for the paper path.

Nothing in this package may construct a broker client until these pass.
Live trading needs a separate human-approval mechanism that does not exist yet.
"""

from __future__ import annotations

from aegis.config import Settings
from aegis.errors import LiveTradingDisabled

PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets"


def normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def assert_paper_execution_allowed(settings: Settings) -> None:
    """Refuse to start unless this process is locked to the Alpaca paper endpoint."""
    settings.assert_paper_mode()
    url = normalize_base_url(settings.alpaca_base_url)
    if url != PAPER_TRADING_BASE_URL:
        raise LiveTradingDisabled(
            "Refusing to start. ALPACA_BASE_URL must be "
            f"{PAPER_TRADING_BASE_URL} for the paper path. "
            f"Got {url or '(empty)'}."
        )
