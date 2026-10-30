"""Result bundle the analyst report consumes.

Schema version 1. Bars stay out of the JSON file; pass them beside it. A
strategy name is a label. This module does not import a strategy class.

.. code-block:: json

   {
     "schema_version": 1,
     "strategy": "macd",
     "symbols": ["AAPL"],
     "starting_equity": 100000,
     "commission_per_share": 0.005,
     "slippage_bps": 5,
     "parameters": {},
     "equity_curve": [{"ts": "2024-01-03T21:00:00+00:00", "equity": 100100.0}],
     "trades": [{
       "symbol": "AAPL",
       "entry_ts": "2024-01-03T15:00:00+00:00",
       "exit_ts": "2024-01-03T20:00:00+00:00",
       "entry_price": 100.0,
       "exit_price": 101.0,
       "quantity": 100,
       "stop_price": 98.0,
       "target_price": 105.0,
       "pnl": 93.45,
       "r_multiple": 0.5,
       "exit_reason": "target"
     }],
     "folds": [{
       "fold_index": 0,
       "train_start": "2024-01-02",
       "train_end": "2024-01-05",
       "test_start": "2024-01-08",
       "test_end": "2024-01-12",
       "starting_equity": 100000,
       "equity_curve": [{"ts": "2024-01-12T21:00:00+00:00", "equity": 101000.0}],
       "trades": []
     }]
   }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from aegis.backtest.report import TradeRecord
from aegis.errors import AegisError

SCHEMA_VERSION = 1


@dataclass
class FoldInput:
    """One out-of-sample window. The curve and trades should already be test-only."""

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    starting_equity: float
    equity_curve: tuple[tuple[datetime, float], ...] = ()
    trades: tuple[TradeRecord, ...] = ()
    symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.train_start > self.train_end:
            raise ValueError(f"Fold {self.fold_index} train start is after train end.")
        if self.test_start > self.test_end:
            raise ValueError(f"Fold {self.fold_index} test start is after test end.")
        if self.starting_equity <= 0:
            raise ValueError(f"Fold {self.fold_index} starting equity must be positive.")
        _check_curve(self.equity_curve, label=f"fold {self.fold_index}")
        _check_trades(self.trades, label=f"fold {self.fold_index}")


@dataclass
class AnalysisInput:
    """A finished backtest plus the optional fold results to summarize."""

    strategy: str
    starting_equity: float
    trades: tuple[TradeRecord, ...] = ()
    equity_curve: tuple[tuple[datetime, float], ...] = ()
    symbols: tuple[str, ...] = ()
    bars: pl.DataFrame | None = None
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0
    parameters: dict = field(default_factory=dict)
    folds: tuple[FoldInput, ...] = ()

    def __post_init__(self) -> None:
        if not self.strategy or not str(self.strategy).strip():
            raise ValueError("strategy name is required.")
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive.")
        if self.commission_per_share < 0 or self.slippage_bps < 0:
            raise ValueError("Commission and slippage cannot be negative.")
        object.__setattr__(self, "strategy", str(self.strategy).strip())
        _check_curve(self.equity_curve, label="equity_curve")
        _check_trades(self.trades, label="trades")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be an object.")


def load_bars(path: Path) -> pl.DataFrame:
    """Read a parquet frame with at least ``symbol``, ``ts``, and ``close``."""
    frame = pl.read_parquet(path)
    missing = {"symbol", "ts", "close"} - set(frame.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise AegisError(f"Bars file is missing column(s): {joined}.")
    timezone_name = getattr(frame.schema["ts"], "time_zone", None)
    if timezone_name is None:
        raise AegisError("Bar timestamps must be timezone-aware. Refusing to guess UTC.")
    return frame


def load_analysis_input(path: Path, bars_path: Path | None = None) -> AnalysisInput:
    """Load schema version 1. ``bars_path`` is the optional parquet sidecar."""
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AegisError(f"Could not parse {path}: {exc}.") from exc
    bars = load_bars(bars_path) if bars_path is not None else None
    try:
        return analysis_from_dict(payload, bars)
    except (AegisError, ValueError, TypeError) as exc:
        if isinstance(exc, AegisError):
            raise
        raise AegisError(str(exc)) from exc


def analysis_from_dict(payload: dict, bars: pl.DataFrame | None = None) -> AnalysisInput:
    if not isinstance(payload, dict):
        raise AegisError("Analysis input must be a JSON object.")
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise AegisError(f"Unsupported analysis schema_version {version}. Expected {SCHEMA_VERSION}.")
    strategy = _required(payload, "strategy")
    starting = _float(_required(payload, "starting_equity"), "starting_equity")
    symbols = tuple(_strings(payload.get("symbols", [])))
    trades = tuple(_parse_trade(item, index) for index, item in enumerate(_list(payload.get("trades", []), "trades")))
    curve = tuple(_parse_point(item, index) for index, item in enumerate(_list(payload.get("equity_curve", []), "equity_curve")))
    folds = tuple(
        _parse_fold(item, index, symbols)
        for index, item in enumerate(_list(payload.get("folds", []), "folds"))
    )
    parameters = payload.get("parameters", {})
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise AegisError("parameters must be a JSON object.")
    _check_parameters(parameters)
    commission = _float(payload.get("commission_per_share", 0.005), "commission_per_share")
    slippage = _float(payload.get("slippage_bps", 5.0), "slippage_bps")
    if not symbols:
        symbols = tuple(sorted({trade.symbol for trade in trades}))
    return AnalysisInput(
        strategy=str(strategy),
        starting_equity=starting,
        trades=trades,
        equity_curve=curve,
        symbols=symbols,
        bars=bars,
        commission_per_share=commission,
        slippage_bps=slippage,
        parameters=parameters,
        folds=folds,
    )


def analysis_to_dict(analysis: AnalysisInput) -> dict:
    """JSON object for ``schema_version`` 1. Bars are not embedded."""
    return {
        "schema_version": SCHEMA_VERSION,
        "strategy": analysis.strategy,
        "symbols": list(analysis.symbols),
        "starting_equity": analysis.starting_equity,
        "commission_per_share": analysis.commission_per_share,
        "slippage_bps": analysis.slippage_bps,
        "parameters": analysis.parameters,
        "equity_curve": [{"ts": _iso(ts), "equity": equity} for ts, equity in analysis.equity_curve],
        "trades": [_trade_dict(trade) for trade in analysis.trades],
        "folds": [
            {
                "fold_index": fold.fold_index,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "starting_equity": fold.starting_equity,
                "symbols": list(fold.symbols),
                "equity_curve": [{"ts": _iso(ts), "equity": equity} for ts, equity in fold.equity_curve],
                "trades": [_trade_dict(trade) for trade in fold.trades],
            }
            for fold in analysis.folds
        ],
    }


def _parse_fold(item: object, index: int, default_symbols: tuple[str, ...]) -> FoldInput:
    if not isinstance(item, dict):
        raise AegisError(f"folds[{index}] must be an object.")
    symbols = tuple(_strings(item["symbols"])) if "symbols" in item else default_symbols
    trades = tuple(
        _parse_trade(trade, trade_index, prefix=f"folds[{index}].trades")
        for trade_index, trade in enumerate(_list(item.get("trades", []), f"folds[{index}].trades"))
    )
    curve = tuple(
        _parse_point(point, point_index, prefix=f"folds[{index}].equity_curve")
        for point_index, point in enumerate(_list(item.get("equity_curve", []), f"folds[{index}].equity_curve"))
    )
    return FoldInput(
        fold_index=_int(item.get("fold_index", index), f"folds[{index}].fold_index"),
        train_start=_date(item.get("train_start"), f"folds[{index}].train_start"),
        train_end=_date(item.get("train_end"), f"folds[{index}].train_end"),
        test_start=_date(item.get("test_start"), f"folds[{index}].test_start"),
        test_end=_date(item.get("test_end"), f"folds[{index}].test_end"),
        starting_equity=_float(item.get("starting_equity"), f"folds[{index}].starting_equity"),
        equity_curve=curve,
        trades=trades,
        symbols=symbols,
    )


def _parse_trade(item: object, index: int, prefix: str = "trades") -> TradeRecord:
    if not isinstance(item, dict):
        raise AegisError(f"{prefix}[{index}] must be an object.")
    r_value = item.get("r_multiple")
    r_multiple = None if r_value is None else _float(r_value, f"{prefix}[{index}].r_multiple")
    try:
        return TradeRecord(
            symbol=str(_required(item, "symbol", f"{prefix}[{index}]")),
            entry_ts=_timestamp(_required(item, "entry_ts", f"{prefix}[{index}]"), f"{prefix}[{index}].entry_ts"),
            exit_ts=_timestamp(_required(item, "exit_ts", f"{prefix}[{index}]"), f"{prefix}[{index}].exit_ts"),
            entry_price=_float(_required(item, "entry_price", f"{prefix}[{index}]"), f"{prefix}[{index}].entry_price"),
            exit_price=_float(_required(item, "exit_price", f"{prefix}[{index}]"), f"{prefix}[{index}].exit_price"),
            quantity=_int(_required(item, "quantity", f"{prefix}[{index}]"), f"{prefix}[{index}].quantity"),
            stop_price=_float(_required(item, "stop_price", f"{prefix}[{index}]"), f"{prefix}[{index}].stop_price"),
            target_price=_float(_required(item, "target_price", f"{prefix}[{index}]"), f"{prefix}[{index}].target_price"),
            pnl=_float(_required(item, "pnl", f"{prefix}[{index}]"), f"{prefix}[{index}].pnl"),
            r_multiple=r_multiple,
            exit_reason=str(item.get("exit_reason") or "unknown"),
        )
    except ValueError as exc:
        raise AegisError(str(exc)) from exc


def _parse_point(item: object, index: int, prefix: str = "equity_curve") -> tuple[datetime, float]:
    if not isinstance(item, dict):
        raise AegisError(f"{prefix}[{index}] must be an object with ts and equity.")
    return (
        _timestamp(_required(item, "ts", f"{prefix}[{index}]"), f"{prefix}[{index}].ts"),
        _float(_required(item, "equity", f"{prefix}[{index}]"), f"{prefix}[{index}].equity"),
    )


def _trade_dict(trade: TradeRecord) -> dict:
    return {
        "symbol": trade.symbol,
        "entry_ts": _iso(trade.entry_ts),
        "exit_ts": _iso(trade.exit_ts),
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,
        "stop_price": trade.stop_price,
        "target_price": trade.target_price,
        "pnl": trade.pnl,
        "r_multiple": trade.r_multiple,
        "exit_reason": trade.exit_reason,
    }


def _check_curve(curve: tuple[tuple[datetime, float], ...], *, label: str) -> None:
    for ts, equity in curve:
        if ts.tzinfo is None:
            raise ValueError(f"{label} has a timestamp with no timezone.")
        if equity != equity or equity in {float("inf"), float("-inf")}:
            raise ValueError(f"{label} has a non-finite equity.")


def _check_trades(trades: tuple[TradeRecord, ...], *, label: str) -> None:
    for trade in trades:
        if trade.entry_ts.tzinfo is None or trade.exit_ts.tzinfo is None:
            raise ValueError(f"{label} timestamps must be timezone-aware.")
        if trade.exit_ts < trade.entry_ts:
            raise ValueError(f"{label} trade on {trade.symbol} exits before it enters.")
        if trade.quantity < 0:
            raise ValueError(f"{label} quantity cannot be negative.")


def _check_parameters(parameters: dict) -> None:
    for key, value in parameters.items():
        if not isinstance(key, str):
            raise AegisError("parameter names must be strings.")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise AegisError(f"parameter {key!r} must be a string, number, boolean, or null.")
        if isinstance(value, bool):
            continue
        if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
            raise AegisError(f"parameter {key!r} is not finite.")


def _required(payload: dict, key: str, label: str | None = None) -> object:
    if key not in payload or payload[key] is None:
        raise AegisError(f"{label or key} is required.")
    return payload[key]


def _list(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise AegisError(f"{label} must be a list.")
    return value


def _strings(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise AegisError("symbols must be a list of non-empty strings.")
    return [item.strip() for item in value]


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AegisError(f"{label} must be a number.")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise AegisError(f"{label} must be finite.")
    return number


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AegisError(f"{label} must be an integer.")
    if int(value) != value:
        raise AegisError(f"{label} must be an integer.")
    return int(value)


def _date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise AegisError(f"{label} must be a YYYY-MM-DD string.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise AegisError(f"{label} must be a YYYY-MM-DD string.") from exc


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise AegisError(f"{label} must be an ISO timestamp with a timezone.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AegisError(f"{label} must be an ISO timestamp with a timezone.") from exc
    if parsed.tzinfo is None:
        raise AegisError(f"{label} is missing a timezone.")
    return parsed


def _iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()
