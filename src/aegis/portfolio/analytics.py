"""Advisory correlation, concentration, drawdown, regime, stress, and allocation.

Weights are suggestions. They are not sent to the broker and they do not
modify the risk engine.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from aegis.backtest.report import PerformanceReport
from aegis.portfolio.models import (
    AllocationSuggestion,
    ConcentrationReport,
    CorrelationMatrix,
    DrawdownReport,
    PortfolioReport,
    RegimeBucket,
    RegimeExposureReport,
    StrategySeries,
    StressScenario,
)


def series_from_report(report: PerformanceReport) -> StrategySeries:
    """Daily returns already on a backtest report."""
    rows = report.to_dict()["daily_returns"]
    if len(rows) < 2:
        raise ValueError(f"{report.strategy} has fewer than two daily returns.")
    return StrategySeries(
        name=report.strategy,
        dates=[date.fromisoformat(row["date"]) for row in rows],
        returns=[float(row["return"]) for row in rows],
    )


def analyze_portfolio(
    strategies: list[StrategySeries],
    *,
    weights: dict[str, float] | None = None,
    position_cap: float = 0.25,
    vol_lookback: int = 5,
) -> PortfolioReport:
    """Summarize a book. ``position_cap`` is the advisory echo of the 25% limit."""
    if not strategies:
        raise ValueError("Portfolio analytics need at least one strategy.")
    if position_cap <= 0 or position_cap > 1:
        raise ValueError("position_cap must sit in (0, 1].")
    names = [item.name for item in strategies]
    if len(set(names)) != len(names):
        raise ValueError("Strategy names must be unique.")
    aligned = _align(strategies)
    book_weights = _normalize(weights, names)
    portfolio_returns = _portfolio_returns(aligned, book_weights)
    correlations = _correlations(aligned)
    concentration = _concentration(book_weights, position_cap)
    drawdowns = _drawdowns(aligned, portfolio_returns, book_weights)
    regimes = _regimes(aligned, portfolio_returns, vol_lookback)
    allocation = _allocation(aligned, position_cap)
    stress = _stress(aligned, portfolio_returns, book_weights)
    notes = [
        "Advisory only. Suggested weights do not approve orders.",
        "The deterministic risk engine still enforces 1% risk, five positions, "
        "25% equity, the daily loss cutoff, and the kill switch.",
        "Regime labels use trailing realized volatility and trailing return. "
        "They describe this sample; they are not a trading signal.",
    ]
    return PortfolioReport(
        correlations=correlations,
        concentration=concentration,
        drawdowns=drawdowns,
        regimes=regimes,
        stress=stress,
        allocation=allocation,
        notes=notes,
    )


def _align(strategies: list[StrategySeries]) -> dict[str, dict[date, float]]:
    return {item.name: dict(zip(item.dates, item.returns, strict=True)) for item in strategies}


def _normalize(weights: dict[str, float] | None, names: list[str]) -> dict[str, float]:
    if weights is None:
        share = 1.0 / len(names)
        return {name: share for name in names}
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"Missing weights for {missing}.")
    extra = [name for name in weights if name not in names]
    if extra:
        raise ValueError(f"Unknown strategies in weights: {extra}.")
    total = sum(weights[name] for name in names)
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    if any(weights[name] < 0 for name in names):
        raise ValueError("Advisory weights are long-only.")
    return {name: weights[name] / total for name in names}


def _common_dates(aligned: dict[str, dict[date, float]]) -> list[date]:
    sets = [set(series) for series in aligned.values()]
    shared = set.intersection(*sets) if sets else set()
    return sorted(shared)


def _portfolio_returns(aligned: dict[str, dict[date, float]], weights: dict[str, float]) -> list[tuple[date, float]]:
    rows = []
    for day in _common_dates(aligned):
        value = sum(weights[name] * aligned[name][day] for name in weights)
        rows.append((day, value))
    if len(rows) < 2:
        raise ValueError("Strategies need at least two dates in common.")
    return rows


def _correlations(aligned: dict[str, dict[date, float]]) -> CorrelationMatrix:
    names = list(aligned)
    days = _common_dates(aligned)
    matrix: list[list[float | None]] = []
    for left in names:
        row: list[float | None] = []
        for right in names:
            if left == right:
                row.append(1.0)
                continue
            x = np.array([aligned[left][day] for day in days], dtype=float)
            y = np.array([aligned[right][day] for day in days], dtype=float)
            row.append(_pearson(x, y))
        matrix.append(row)
    return CorrelationMatrix(names=names, values=matrix, observations=len(days))


def _concentration(weights: dict[str, float], cap: float) -> ConcentrationReport:
    herfindahl = sum(value * value for value in weights.values())
    max_name = max(weights, key=weights.get)
    return ConcentrationReport(
        weights=dict(weights),
        herfindahl=herfindahl,
        effective_n=(1.0 / herfindahl) if herfindahl > 0 else 0.0,
        max_weight=weights[max_name],
        max_weight_name=max_name,
        above_position_cap=[name for name, value in weights.items() if value > cap + 1e-12],
        position_cap=cap,
    )


def _drawdowns(
    aligned: dict[str, dict[date, float]],
    portfolio_returns: list[tuple[date, float]],
    weights: dict[str, float],
) -> DrawdownReport:
    dates = [day for day, _value in portfolio_returns]
    equity = _equity([value for _day, value in portfolio_returns])
    max_dd, peak_i, trough_i = _max_drawdown_span(equity)
    recovered = trough_i < len(equity) - 1 and any(point >= equity[peak_i] for point in equity[trough_i:])
    per_strategy = {}
    worst_name = None
    worst_loss = 0.0
    for name, series in aligned.items():
        ordered = [series[day] for day in dates if day in series]
        per_strategy[name] = _max_drawdown_span(_equity(ordered))[0]
        if trough_i > peak_i:
            loss = sum(series[dates[index]] * weights[name] for index in range(peak_i + 1, trough_i + 1))
            if loss < worst_loss:
                worst_loss = loss
                worst_name = name
    return DrawdownReport(
        max_drawdown=max_dd,
        peak_date=dates[peak_i],
        trough_date=dates[trough_i],
        duration_days=trough_i - peak_i,
        recovered=recovered,
        per_strategy_max_drawdown=per_strategy,
        worst_contributor=worst_name,
    )


def _regimes(
    aligned: dict[str, dict[date, float]],
    portfolio_returns: list[tuple[date, float]],
    lookback: int,
) -> RegimeExposureReport:
    if lookback < 2:
        raise ValueError("vol_lookback must be at least 2.")
    dates = [day for day, _value in portfolio_returns]
    values = [value for _day, value in portfolio_returns]
    vols: list[float | None] = []
    trends: list[str] = []
    for index in range(len(values)):
        window = values[max(0, index - lookback + 1) : index + 1]
        if len(window) < lookback:
            vols.append(None)
            trends.append("unknown")
            continue
        vols.append(float(np.std(np.array(window, dtype=float), ddof=1)))
        drift = sum(window)
        if drift > 1e-12:
            trends.append("up")
        elif drift < -1e-12:
            trends.append("down")
        else:
            trends.append("flat")
    known = [value for value in vols if value is not None]
    low_cut, high_cut = _tercile_cuts(known)
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, (vol, trend) in enumerate(zip(vols, trends, strict=True)):
        if vol is None:
            label = "unknown"
        elif vol <= low_cut:
            label = "low"
        elif vol >= high_cut:
            label = "high"
        else:
            label = "medium"
        buckets.setdefault((label, trend), []).append(index)
    rows = []
    for (vol_label, trend), indexes in sorted(buckets.items()):
        mean_return = float(np.mean([values[index] for index in indexes]))
        per_strategy = {}
        for name, series in aligned.items():
            samples = [series[dates[index]] for index in indexes if dates[index] in series]
            if samples:
                per_strategy[name] = float(np.mean(samples))
        rows.append(
            RegimeBucket(
                volatility=vol_label,
                trend=trend,
                days=len(indexes),
                mean_return=mean_return,
                strategy_mean_returns=per_strategy,
            )
        )
    return RegimeExposureReport(
        lookback=lookback,
        buckets=rows,
        notes=["Volatility terciles are computed on this sample's trailing window."],
    )


def _allocation(aligned: dict[str, dict[date, float]], cap: float) -> AllocationSuggestion:
    days = _common_dates(aligned)
    raw: dict[str, float] = {}
    for name, series in aligned.items():
        values = np.array([series[day] for day in days], dtype=float)
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        raw[name] = 0.0 if sigma <= 1e-12 else 1.0 / sigma
    if sum(raw.values()) <= 0:
        equal = {name: 1.0 / len(raw) for name in raw}
        capped, feasible = _cap_weights(equal, cap)
        return AllocationSuggestion(
            method="equal_weight",
            weights=capped,
            capped_at=cap,
            cap_feasible=feasible,
            rationale="Inverse volatility was undefined, so the suggestion is equal weight.",
        )
    inverse = {name: value / sum(raw.values()) for name, value in raw.items()}
    capped, feasible = _cap_weights(inverse, cap)
    rationale = (
        "Inverse-volatility weights, long only."
        if feasible
        else "Inverse-volatility weights. The position cap cannot be met with this many names, so it was not forced."
    )
    return AllocationSuggestion(
        method="inverse_volatility",
        weights=capped,
        capped_at=cap if feasible else None,
        cap_feasible=feasible,
        rationale=rationale,
    )


def _stress(
    aligned: dict[str, dict[date, float]],
    portfolio_returns: list[tuple[date, float]],
    weights: dict[str, float],
) -> list[StressScenario]:
    values = np.array([value for _day, value in portfolio_returns], dtype=float)
    scaled = values * 2.0
    shocked_equity = _equity(scaled.tolist())
    vol = float(np.std(values, ddof=1))
    stressed_vol = sum(weights[name] * _sigma(aligned[name]) for name in weights)
    worst_index = int(np.argmin(values))
    worst_day = portfolio_returns[worst_index][0]
    detail = {name: aligned[name][worst_day] for name in aligned if worst_day in aligned[name]}
    largest = max(weights, key=weights.get)
    reduced = {name: (0.0 if name == largest else weight) for name, weight in weights.items()}
    reduced_total = sum(reduced.values())
    if reduced_total > 0:
        reduced = {name: value / reduced_total for name, value in reduced.items()}
        reduced_values = [
            sum(reduced[name] * aligned[name][day] for name in reduced) for day, _value in portfolio_returns
        ]
        reduced_return = _compound(reduced_values)
    else:
        reduced_return = 0.0
    return [
        StressScenario(
            name="vol_spike",
            description="Scale every portfolio daily return by 2 and recompute the path.",
            portfolio_return=_compound(scaled.tolist()),
            max_drawdown=_max_drawdown_span(shocked_equity)[0],
            portfolio_volatility=float(np.std(scaled, ddof=1)),
        ),
        StressScenario(
            name="correlation_one",
            description="Portfolio volatility if every strategy moves together.",
            portfolio_volatility=stressed_vol,
            detail={"current_volatility": vol},
        ),
        StressScenario(
            name="drop_largest_weight",
            description=f"Remove {largest} and renormalize the remaining weights.",
            portfolio_return=reduced_return,
            detail={name: reduced[name] for name in reduced},
        ),
        StressScenario(
            name="worst_overlapping_day",
            description=f"Cross-strategy book on {worst_day.isoformat()}, the worst portfolio day.",
            portfolio_return=float(values[worst_index]),
            detail=detail,
        ),
    ]


def _cap_weights(weights: dict[str, float], cap: float) -> tuple[dict[str, float], bool]:
    names = list(weights)
    if cap * len(names) < 1.0 - 1e-9:
        return dict(weights), False
    current = dict(weights)
    for _ in range(len(names) + 2):
        over = {name: value for name, value in current.items() if value > cap + 1e-12}
        if not over:
            break
        excess = sum(value - cap for value in over.values())
        for name in over:
            current[name] = cap
        room = [name for name in current if name not in over and current[name] < cap - 1e-12]
        if not room:
            break
        room_weight = sum(current[name] for name in room)
        if room_weight <= 0:
            share = excess / len(room)
            for name in room:
                current[name] += share
        else:
            for name in room:
                current[name] += excess * (current[name] / room_weight)
    total = sum(current.values())
    if total <= 0:
        return dict(weights), False
    return {name: value / total for name, value in current.items()}, True


def _equity(returns: list[float]) -> list[float]:
    equity = []
    level = 1.0
    for value in returns:
        level *= 1.0 + value
        equity.append(level)
    return equity


def _compound(returns: list[float]) -> float:
    level = 1.0
    for value in returns:
        level *= 1.0 + value
    return level - 1.0


def _max_drawdown_span(equity: list[float]) -> tuple[float, int, int]:
    peak = equity[0]
    peak_index = 0
    worst = 0.0
    worst_peak = 0
    worst_trough = 0
    for index, level in enumerate(equity):
        if level >= peak:
            peak = level
            peak_index = index
        if peak > 0:
            drawdown = (peak - level) / peak
            if drawdown > worst:
                worst = drawdown
                worst_peak = peak_index
                worst_trough = index
    return worst, worst_peak, worst_trough


def _sigma(series: dict[date, float]) -> float:
    values = np.array(list(series.values()), dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    x = left - left.mean()
    y = right - right.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom == 0:
        return None
    return float((x * y).sum() / denom)


def _tercile_cuts(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.array(values, dtype=float)
    return float(np.quantile(array, 1 / 3)), float(np.quantile(array, 2 / 3))
