"""Pydantic results for advisory portfolio analytics.

These numbers do not change :class:`aegis.backtest.risk.RiskLimits` and do
not approve orders. The deterministic risk engine remains the enforcer.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrategySeries(BaseModel):
    """Daily simple returns for one strategy."""

    model_config = ConfigDict(frozen=True)

    name: str
    dates: list[date]
    returns: list[float]

    @model_validator(mode="after")
    def _aligned(self) -> StrategySeries:
        if len(self.dates) != len(self.returns):
            raise ValueError("dates and returns must be the same length.")
        if len(self.dates) < 2:
            raise ValueError("A strategy series needs at least two daily returns.")
        if len(set(self.dates)) != len(self.dates):
            raise ValueError("Strategy dates must be unique.")
        return self


class CorrelationMatrix(BaseModel):
    model_config = ConfigDict(frozen=True)

    names: list[str]
    values: list[list[float | None]]
    observations: int


class ConcentrationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    weights: dict[str, float]
    herfindahl: float
    effective_n: float
    max_weight: float
    max_weight_name: str
    above_position_cap: list[str]
    position_cap: float


class DrawdownReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_drawdown: float
    peak_date: date | None
    trough_date: date | None
    duration_days: int
    recovered: bool
    per_strategy_max_drawdown: dict[str, float]
    worst_contributor: str | None


class RegimeBucket(BaseModel):
    model_config = ConfigDict(frozen=True)

    volatility: str
    trend: str
    days: int
    mean_return: float
    strategy_mean_returns: dict[str, float]


class RegimeExposureReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    lookback: int
    buckets: list[RegimeBucket]
    notes: list[str]


class StressScenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    portfolio_return: float | None = None
    max_drawdown: float | None = None
    portfolio_volatility: float | None = None
    detail: dict[str, float] = Field(default_factory=dict)


class AllocationSuggestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    advisory_only: Literal[True] = True
    method: str
    weights: dict[str, float]
    capped_at: float | None
    cap_feasible: bool
    hard_limits_unchanged: Literal[True] = True
    rationale: str


class PortfolioReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    advisory_only: Literal[True] = True
    enforced_by: Literal["RiskEngine"] = "RiskEngine"
    changes_risk_limits: Literal[False] = False
    correlations: CorrelationMatrix
    concentration: ConcentrationReport
    drawdowns: DrawdownReport
    regimes: RegimeExposureReport
    stress: list[StressScenario]
    allocation: AllocationSuggestion
    notes: list[str]
