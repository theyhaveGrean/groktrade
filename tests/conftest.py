import pytest

from aegis.backtest.risk import clear_risk_engine


@pytest.fixture(autouse=True)
def _reset_risk_engine():
    clear_risk_engine()
    yield
    clear_risk_engine()
