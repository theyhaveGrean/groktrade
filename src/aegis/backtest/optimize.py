"""Walk-forward search over a registered strategy's parameter model.

The search space comes from ``SearchMeta`` on the strategy's Pydantic params.
CLI overrides replace individual axes. Each fold picks a parameter set on its
in-sample window and is scored only on the following out-of-sample window.
Every in-sample evaluation is a trial in the Parquet ledger. Out-of-sample
columns are filled only for the selected set. The stitched equity curve
compounds those out-of-sample paths and never includes in-sample bars.

The deflated Sharpe uses the trial count as N. Bar timeframe is just another
axis: coarser bars are aggregated from the cached 1-minute prints inside the
strategy plugin.
"""

from __future__ import annotations

import io
import json
import logging
import multiprocessing
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from aegis.backtest.deflated_sharpe import deflated_sharpe_ratio
from aegis.backtest.report import PerformanceReport, TradeRecord, build_report
from aegis.backtest.walkforward import WalkForwardFold, walk_forward_splits
from aegis.config import Settings, load_settings
from aegis.data.cache import BarCache
from aegis.data.query import load_research_bars
from aegis.holdout import DateRange, assert_dates_outside_holdout, assert_no_holdout_overlap
from aegis.strategies import get_strategy, parameter_grid
from aegis.strategies.registry import StrategyPlugin

logger = logging.getLogger(__name__)

_BARS: pl.DataFrame | None = None
_CFG: dict | None = None


@dataclass(frozen=True)
class ParamStability:
    name: str
    selected: tuple[str, ...]
    mode: str
    mode_fraction: float
    mean: float | None
    stdev: float | None
    minimum: str
    maximum: str


@dataclass(frozen=True)
class FoldSelection:
    fold: WalkForwardFold
    params: dict
    is_sharpe: float | None
    is_total_return: float
    is_trade_count: int
    oos_sharpe: float | None
    oos_total_return: float
    oos_trade_count: int


@dataclass(frozen=True)
class OptimizationReport:
    study_id: str
    strategy: str
    mode: str
    method: str
    folds: tuple[FoldSelection, ...]
    n_parameter_sets: int
    n_trials: int
    stitched: PerformanceReport
    stitched_sharpe: float | None
    best_fold_oos_sharpe: float | None
    deflated_sharpe: float | None
    stability: tuple[ParamStability, ...]
    ledger_path: Path | None

    def render(self) -> str:
        lines = [
            f"Walk-forward optimization ({self.strategy}, {self.mode}, {self.method})",
            f"Study: {self.study_id}",
            f"Folds: {len(self.folds)}",
            f"Parameter sets: {self.n_parameter_sets}",
            f"Trials: {self.n_trials}",
            f"Stitched OOS Sharpe: {_num(self.stitched_sharpe)}",
            f"Best fold OOS Sharpe: {_num(self.best_fold_oos_sharpe)}",
            f"Deflated Sharpe: {_prob(self.deflated_sharpe)} "
            f"(probability the stitched OOS Sharpe exceeds 0 after {self.n_trials} trials)",
            "Selected parameters:",
        ]
        for choice in self.folds:
            lines.append(
                f"  fold {choice.fold.fold_index}: "
                f"train {choice.fold.train_start.isoformat()}..{choice.fold.train_end.isoformat()} "
                f"test {choice.fold.test_start.isoformat()}..{choice.fold.test_end.isoformat()} "
                f"{_label(choice.params)} "
                f"IS Sharpe {_num(choice.is_sharpe)} "
                f"OOS Sharpe {_num(choice.oos_sharpe)} "
                f"OOS trades {choice.oos_trade_count}"
            )
        lines.append("Parameter stability (selected value per fold):")
        for item in self.stability:
            lines.append(
                f"  {item.name}: mode={item.mode} "
                f"mode_fraction={item.mode_fraction:.0%} "
                f"mean={_num(item.mean)} std={_num(item.stdev)} "
                f"min={item.minimum} max={item.maximum}"
            )
        lines.append("Stitched out-of-sample report:")
        lines.append(self.stitched.render())
        if self.ledger_path is not None:
            lines.append(f"Trial ledger: {self.ledger_path}")
        return "\n".join(lines)


def default_worker_count() -> int:
    """``min(cpu-1, 6)``, and at least one."""
    cpus = os.cpu_count() or 2
    return max(1, min(cpus - 1, 6))


def parameter_stability(selected: list[dict]) -> tuple[ParamStability, ...]:
    """How often each fold agreed on each selected parameter, including bar size."""
    if not selected:
        return ()
    names = sorted({key for row in selected for key in row})
    report: list[ParamStability] = []
    for name in names:
        values = [row[name] for row in selected]
        mode = _mode(values)
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if len(numeric) == len(values) and numeric:
            mean = sum(numeric) / len(numeric)
            variance = (
                sum((item - mean) ** 2 for item in numeric) / (len(numeric) - 1) if len(numeric) > 1 else 0.0
            )
            stdev: float | None = variance**0.5
            mean_value: float | None = mean
        else:
            mean_value = None
            stdev = None
        rendered = tuple(_format_value(value) for value in values)
        report.append(
            ParamStability(
                name=name,
                selected=rendered,
                mode=_format_value(mode),
                mode_fraction=sum(1 for item in values if item == mode) / len(values),
                mean=mean_value,
                stdev=stdev,
                minimum=_format_value(min(values, key=_format_value)),
                maximum=_format_value(max(values, key=_format_value)),
            )
        )
    return tuple(report)


def optimize_research(
    settings: Settings,
    *,
    strategy: str,
    start: date,
    end: date,
    symbols: list[str] | None = None,
    param_overrides: dict[str, list] | None = None,
    method: str = "grid",
    samples: int = 20,
    seed: int = 7,
    train_days: int = 60,
    test_days: int = 20,
    step_days: int | None = None,
    embargo_days: int = 1,
    mode: str = "rolling",
    workers: int | None = None,
    ledger_path: Path | None = None,
    study_id: str | None = None,
) -> OptimizationReport:
    """Optimize on cached bars. Refuses a window or cache slice that touches the holdout."""
    settings.assert_paper_mode()
    from aegis.data.pipeline import resolve_members

    names = [member.symbol for member in resolve_members(settings, symbols, end)]
    bars = load_research_bars(BarCache(settings.data_cache_dir), names, start, end, settings.holdout)
    return optimize_on_bars(
        bars,
        strategy=strategy,
        start=start,
        end=end,
        holdout=settings.holdout,
        param_overrides=param_overrides,
        method=method,
        samples=samples,
        seed=seed,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        embargo_days=embargo_days,
        mode=mode,
        workers=workers,
        starting_equity=settings.starting_equity,
        commission_per_share=settings.commission_per_share,
        slippage_bps=settings.slippage_bps,
        kill_switch=settings.risk_kill_switch,
        log_level=settings.nautilus_log_level,
        ledger_path=ledger_path,
        study_id=study_id,
    )


def optimize_on_bars(
    bars: pl.DataFrame,
    *,
    strategy: str,
    start: date,
    end: date,
    holdout: DateRange,
    param_overrides: dict[str, list] | None = None,
    method: str = "grid",
    samples: int = 20,
    seed: int = 7,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
    embargo_days: int = 1,
    mode: str = "rolling",
    workers: int | None = None,
    starting_equity: float = 100_000.0,
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
    kill_switch: bool = False,
    log_level: str = "ERROR",
    ledger_path: Path | None = None,
    study_id: str | None = None,
    candidate_source: str | None = None,
    candidate_name: str | None = None,
) -> OptimizationReport:
    """Search ``bars`` inside ``start..end``. In-sample selection, out-of-sample score."""
    if method not in {"grid", "random"}:
        raise ValueError("method must be 'grid' or 'random'.")
    if start > end:
        raise ValueError("start must be on or before end.")
    assert_no_holdout_overlap(DateRange(start, end), holdout, purpose="parameter optimization")
    if not bars.is_empty():
        present = set(
            bars.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()).to_series().to_list()
        )
        assert_dates_outside_holdout(present, holdout, purpose="parameter optimization")
    plugin = get_strategy(strategy)
    points = parameter_grid(
        plugin.params_model,
        param_overrides,
        method=method,
        samples=samples,
        seed=seed,
    )
    folds = walk_forward_splits(
        start,
        end,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        embargo_days=embargo_days,
        holdout=holdout,
        mode=mode,
    )
    if not folds:
        raise ValueError(
            "No complete walk-forward folds in that window. "
            "Shorten train_days or test_days, or widen the date range."
        )
    _assert_oos_disjoint(folds)
    worker_count = default_worker_count() if workers is None else workers
    if worker_count < 1:
        raise ValueError("workers must be at least 1.")
    study = study_id or _new_study_id()
    n_trials = len(folds) * len(points)
    logger.info(
        "Study %s (%s): %s %s folds, %s parameter sets, %s trials, %s workers.",
        study,
        plugin.name,
        len(folds),
        mode,
        len(points),
        n_trials,
        worker_count,
    )
    is_tasks = [
        _task(plugin, fold, point, param_index, "in_sample")
        for fold in folds
        for param_index, point in enumerate(points)
    ]
    pool_workers = min(worker_count, len(is_tasks))
    with _executor(
        bars,
        pool_workers,
        holdout,
        starting_equity,
        commission_per_share,
        slippage_bps,
        kill_switch,
        log_level,
        candidate_source=candidate_source,
        candidate_name=candidate_name,
    ) as pool:
        is_results = list(pool.map(worker_run, is_tasks))
        chosen_tasks = []
        chosen_is: dict[int, dict] = {}
        for fold in folds:
            fold_rows = [row for row in is_results if row["fold_index"] == fold.fold_index]
            winner = max(fold_rows, key=_selection_key)
            chosen_is[fold.fold_index] = winner
            chosen_tasks.append(
                _task(plugin, fold, winner["params"], winner["param_index"], "out_of_sample")
            )
            logger.info(
                "Fold %s selected %s (IS Sharpe %s).",
                fold.fold_index,
                _label(winner["params"]),
                _num(winner["sharpe"]),
            )
        oos_results = list(pool.map(worker_run, chosen_tasks))
    oos_by_fold = {row["fold_index"]: row for row in oos_results}
    selections: list[FoldSelection] = []
    oos_curves: list[list[tuple[datetime, float]]] = []
    oos_trades: list[TradeRecord] = []
    oos_bars: list[pl.DataFrame] = []
    ledger_rows: list[dict] = []
    for row in is_results:
        fold = folds[row["fold_index"]]
        selected = row["param_index"] == chosen_is[fold.fold_index]["param_index"]
        oos = oos_by_fold[fold.fold_index] if selected else None
        if selected and oos is not None:
            oos_curves.append(_parse_equity(oos["equity"]))
            oos_trades.extend(_parse_trades(oos["trades"]))
            oos_bars.append(slice_bars(bars, fold.test_start, fold.test_end))
            selections.append(
                FoldSelection(
                    fold=fold,
                    params=dict(row["params"]),
                    is_sharpe=row["sharpe"],
                    is_total_return=row["total_return"],
                    is_trade_count=row["trade_count"],
                    oos_sharpe=oos["sharpe"],
                    oos_total_return=oos["total_return"],
                    oos_trade_count=oos["trade_count"],
                )
            )
        ledger_rows.append(
            _ledger_row(
                study_id=study,
                method=method,
                mode=mode,
                seed=None if method == "grid" else seed,
                n_trials=n_trials,
                n_parameter_sets=len(points),
                n_folds=len(folds),
                row=row,
                fold=fold,
                selected=selected,
                oos=oos,
            )
        )
    stitched_curve = stitch_equity(oos_curves, starting_equity)
    stitched_bars = pl.concat(oos_bars, how="vertical") if oos_bars else bars.clear()
    symbols = tuple(bars["symbol"].unique().sort().to_list()) if not bars.is_empty() else ()
    stitched = build_report(
        strategy=f"{plugin.name}_walk_forward",
        symbols=symbols,
        starting_equity=starting_equity,
        equity_curve=stitched_curve,
        trades=oos_trades,
        bars=stitched_bars,
    )
    daily = _daily_returns(stitched_curve, starting_equity)
    deflated = deflated_sharpe_ratio(daily, n_trials)
    fold_sharpes = [choice.oos_sharpe for choice in selections if choice.oos_sharpe is not None]
    best_fold = max(fold_sharpes) if fold_sharpes else None
    stability = parameter_stability([choice.params for choice in selections])
    frame = pl.DataFrame(ledger_rows).with_columns(
        pl.col("recorded_at").cast(pl.Datetime("us", "UTC")),
        pl.col("seed").cast(pl.Int64),
        pl.col("is_sharpe").cast(pl.Float64),
        pl.col("oos_sharpe").cast(pl.Float64),
        pl.col("oos_total_return").cast(pl.Float64),
        pl.col("oos_max_drawdown").cast(pl.Float64),
        pl.col("oos_trade_count").cast(pl.Int64),
        pl.col("oos_win_rate").cast(pl.Float64),
        pl.lit(stitched.sharpe).cast(pl.Float64).alias("stitched_oos_sharpe"),
        pl.lit(best_fold).cast(pl.Float64).alias("best_fold_oos_sharpe"),
        pl.lit(deflated).cast(pl.Float64).alias("deflated_sharpe"),
    )
    written = append_trials(ledger_path, frame) if ledger_path is not None else None
    if ledger_path is not None:
        _write_oos_equity(ledger_path, study, selections, oos_curves, starting_equity)
    for row in ledger_rows:
        logger.info(
            "Trial %s/%s fold %s %s selected=%s IS Sharpe %s OOS Sharpe %s",
            row["trial_index"] + 1,
            n_trials,
            row["fold_index"],
            row["params_json"],
            row["selected"],
            _num(row["is_sharpe"]),
            _num(row["oos_sharpe"]),
        )
    return OptimizationReport(
        study_id=study,
        strategy=plugin.name,
        mode=mode,
        method=method,
        folds=tuple(selections),
        n_parameter_sets=len(points),
        n_trials=n_trials,
        stitched=stitched,
        stitched_sharpe=stitched.sharpe,
        best_fold_oos_sharpe=best_fold,
        deflated_sharpe=deflated,
        stability=stability,
        ledger_path=written,
    )


def slice_bars(bars: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    """Bars whose Eastern session date is inside ``[start, end]``."""
    if bars.is_empty():
        return bars
    session = pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()
    return bars.filter((session >= start) & (session <= end)).sort(["symbol", "ts"])


def stitch_equity(
    segments: list[list[tuple[datetime, float]]],
    starting_equity: float,
) -> list[tuple[datetime, float]]:
    """Compound independent out-of-sample curves into one equity path.

    Each segment is a fresh backtest that starts near ``starting_equity``.
    The stitch keeps that segment's relative path and continues from the
    previous segment's ending equity.
    """
    level = starting_equity
    stitched: list[tuple[datetime, float]] = []
    for curve in segments:
        if not curve:
            continue
        origin = curve[0][1]
        if origin <= 0:
            continue
        for ts, equity in curve:
            stitched.append((ts, level * (equity / origin)))
        level = stitched[-1][1]
    return stitched


def read_trials(path: Path) -> pl.DataFrame:
    """Load the trial ledger through DuckDB so later checks can use SQL."""
    import duckdb

    connection = duckdb.connect()
    try:
        table = connection.sql(
            "SELECT * FROM read_parquet(?) ORDER BY study_id, trial_index",
            params=[str(path)],
        ).to_arrow_table()
    finally:
        connection.close()
    frame = pl.from_arrow(table)
    if not isinstance(frame, pl.DataFrame):
        frame = pl.DataFrame(frame)
    return frame


def append_trials(path: Path, frame: pl.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = pl.concat([pl.read_parquet(path), frame], how="vertical_relaxed") if path.exists() else frame
    temporary = path.with_suffix(".parquet.tmp")
    combined.write_parquet(temporary)
    temporary.replace(path)
    return path


def worker_init(blob: bytes, config: dict) -> None:
    """Load the bar cache once per process. Tasks only carry dates and parameters."""
    global _BARS, _CFG
    os.environ["OMP_NUM_THREADS"] = "1"
    _BARS = pl.read_parquet(io.BytesIO(blob))
    _CFG = config
    source = config.get("candidate_source")
    name = config.get("candidate_name")
    if source and name:
        from aegis.agents.evaluate import register_candidate_source

        register_candidate_source(source, name)


def worker_run(task: dict) -> dict:
    """One backtest of whatever strategy the task names."""
    if _BARS is None or _CFG is None:
        raise RuntimeError("Optimizer worker was not initialized.")
    start = date.fromisoformat(task["start"])
    end = date.fromisoformat(task["end"])
    window = slice_bars(_BARS, start, end)
    holdout = DateRange(date.fromisoformat(_CFG["holdout_start"]), date.fromisoformat(_CFG["holdout_end"]))
    plugin = get_strategy(task["strategy"])
    params = plugin.params_model.model_validate(task["params"])
    report = plugin.run(
        window,
        params,
        holdout=holdout,
        starting_equity=_CFG["starting_equity"],
        commission_per_share=_CFG["commission_per_share"],
        slippage_bps=_CFG["slippage_bps"],
        kill_switch=_CFG["kill_switch"],
        log_level=_CFG["log_level"],
    )
    return {
        "fold_index": task["fold_index"],
        "param_index": task["param_index"],
        "role": task["role"],
        "strategy": task["strategy"],
        "params": params.model_dump(),
        "sharpe": report.sharpe,
        "total_return": report.total_return,
        "max_drawdown": report.max_drawdown,
        "trade_count": report.trade_count,
        "win_rate": report.win_rate,
        "ending_equity": report.ending_equity,
        "equity": [(ts.isoformat(), equity) for ts, equity in report.equity_curve],
        "trades": report.to_dict()["trades"],
    }


def _executor(
    bars: pl.DataFrame,
    workers: int,
    holdout: DateRange,
    starting_equity: float,
    commission_per_share: float,
    slippage_bps: float,
    kill_switch: bool,
    log_level: str,
    candidate_source: str | None = None,
    candidate_name: str | None = None,
) -> ProcessPoolExecutor:
    blob = io.BytesIO()
    bars.write_parquet(blob)
    config = {
        "holdout_start": holdout.start.isoformat(),
        "holdout_end": holdout.end.isoformat(),
        "starting_equity": starting_equity,
        "commission_per_share": commission_per_share,
        "slippage_bps": slippage_bps,
        "kill_switch": kill_switch,
        "log_level": log_level,
        "candidate_source": candidate_source,
        "candidate_name": candidate_name,
    }
    context = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=worker_init,
        initargs=(blob.getvalue(), config),
    )


def _task(plugin: StrategyPlugin, fold: WalkForwardFold, params: dict, param_index: int, role: str) -> dict:
    start, end = (fold.train_start, fold.train_end) if role == "in_sample" else (fold.test_start, fold.test_end)
    return {
        "strategy": plugin.name,
        "fold_index": fold.fold_index,
        "param_index": param_index,
        "role": role,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "params": params,
    }


def _selection_key(row: dict) -> tuple:
    """Higher in-sample Sharpe wins. Ties keep the earlier parameter set."""
    sharpe = row["sharpe"] if row["sharpe"] is not None else float("-inf")
    return (sharpe, row["total_return"])


def _ledger_row(
    *,
    study_id: str,
    method: str,
    mode: str,
    seed: int | None,
    n_trials: int,
    n_parameter_sets: int,
    n_folds: int,
    row: dict,
    fold: WalkForwardFold,
    selected: bool,
    oos: dict | None,
) -> dict:
    return {
        "study_id": study_id,
        "recorded_at": datetime.now(timezone.utc),
        "search_method": method,
        "walk_mode": mode,
        "seed": seed,
        "objective": "is_sharpe",
        "trial_index": fold.fold_index * n_parameter_sets + row["param_index"],
        "n_trials": n_trials,
        "n_parameter_sets": n_parameter_sets,
        "n_folds": n_folds,
        "fold_index": fold.fold_index,
        "param_index": row["param_index"],
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "strategy": row["strategy"],
        "params_json": json.dumps(row["params"], sort_keys=True, default=str),
        "selected": selected,
        "is_sharpe": row["sharpe"],
        "is_total_return": row["total_return"],
        "is_max_drawdown": row["max_drawdown"],
        "is_trade_count": row["trade_count"],
        "is_win_rate": row["win_rate"],
        "oos_sharpe": None if oos is None else oos["sharpe"],
        "oos_total_return": None if oos is None else oos["total_return"],
        "oos_max_drawdown": None if oos is None else oos["max_drawdown"],
        "oos_trade_count": None if oos is None else oos["trade_count"],
        "oos_win_rate": None if oos is None else oos["win_rate"],
    }


def _write_oos_equity(
    ledger_path: Path,
    study_id: str,
    selections: list[FoldSelection],
    curves: list[list[tuple[datetime, float]]],
    starting_equity: float,
) -> None:
    rows: list[dict] = []
    level = starting_equity
    for choice, curve in zip(selections, curves, strict=True):
        if not curve or curve[0][1] <= 0:
            continue
        origin = curve[0][1]
        for ts, equity in curve:
            rows.append(
                {
                    "study_id": study_id,
                    "fold_index": choice.fold.fold_index,
                    "ts": ts,
                    "equity": level * (equity / origin),
                }
            )
        level = rows[-1]["equity"]
    if not rows:
        return
    frame = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Datetime("us", "UTC")))
    append_trials(ledger_path.with_name("oos_equity.parquet"), frame)


def _parse_equity(points: list[tuple[str, float]]) -> list[tuple[datetime, float]]:
    return [(datetime.fromisoformat(ts), equity) for ts, equity in points]


def _parse_trades(rows: list[dict]) -> list[TradeRecord]:
    trades: list[TradeRecord] = []
    for row in rows:
        entry_ts = row["entry_ts"]
        exit_ts = row["exit_ts"]
        trades.append(
            TradeRecord(
                symbol=row["symbol"],
                entry_ts=datetime.fromisoformat(entry_ts) if isinstance(entry_ts, str) else entry_ts,
                exit_ts=datetime.fromisoformat(exit_ts) if isinstance(exit_ts, str) else exit_ts,
                entry_price=row["entry_price"],
                exit_price=row["exit_price"],
                quantity=row["quantity"],
                stop_price=row["stop_price"],
                target_price=row["target_price"],
                pnl=row["pnl"],
                r_multiple=row["r_multiple"],
                exit_reason=row["exit_reason"],
            )
        )
    return trades


def _daily_returns(curve: list[tuple[datetime, float]], starting_equity: float) -> list[float]:
    if starting_equity <= 0 or not curve:
        return []
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    by_day: dict[date, float] = {}
    for ts, equity in curve:
        stamp = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
        by_day[stamp.astimezone(eastern).date()] = equity
    previous = starting_equity
    returns: list[float] = []
    for day in sorted(by_day):
        equity = by_day[day]
        returns.append(equity / previous - 1.0)
        previous = equity
    return returns


def _assert_oos_disjoint(folds: list[WalkForwardFold]) -> None:
    for previous, nxt in zip(folds, folds[1:]):
        if nxt.test_start <= previous.test_end:
            raise ValueError(
                "Out-of-sample windows overlap. Set step_days >= test_days so the "
                "stitched curve does not reuse a day."
            )


def _mode(values: list) -> object:
    ranked = sorted(set(values), key=lambda item: (-values.count(item), _format_value(item)))
    return ranked[0]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _label(params: dict) -> str:
    parts = []
    for name in sorted(params):
        parts.append(f"{name}={_format_value(params[name])}")
    return " ".join(parts)


def _new_study_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _prob(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def optimize_demo(
    settings: Settings | None = None,
    *,
    strategy: str = "ema_cross",
    param_overrides: dict[str, list] | None = None,
    method: str = "grid",
    samples: int = 20,
    seed: int = 7,
    workers: int | None = None,
    mode: str = "rolling",
    ledger_path: Path | None = None,
    study_id: str | None = None,
) -> OptimizationReport:
    """Short synthetic walk-forward. No cache and no API key.

    With no overrides this sweeps a small smoke grid (two fast periods, one
    bar size) so the command finishes without a data cache. Pass
    ``param_overrides`` to replace any axis.
    """
    from aegis.backtest.synthetic import OPTIMIZE_DEMO_DAYS, synthetic_bars

    from aegis.strategies.params import search_spec

    settings = settings or load_settings()
    settings.assert_paper_mode()
    plugin = get_strategy(strategy)
    if strategy == "ema_cross":
        smoke = {
            "fast_period": [5, 8],
            "slow_period": [21],
            "atr_period": [10],
            "atr_multiplier": [2.0],
            "bar_timeframe": ["1min"],
        }
    else:
        smoke = {}
        for name, field in plugin.params_model.model_fields.items():
            spec = search_spec(field)
            if spec is None:
                continue
            smoke[name] = ["1min"] if name == "bar_timeframe" else [spec.default_grid[0]]
    if param_overrides:
        smoke.update(param_overrides)
    return optimize_on_bars(
        synthetic_bars(sessions=OPTIMIZE_DEMO_DAYS),
        strategy=strategy,
        start=date(2024, 1, 3),
        end=date(2024, 1, 15),
        holdout=settings.holdout,
        param_overrides=smoke,
        method=method,
        samples=samples,
        seed=seed,
        train_days=3,
        test_days=3,
        step_days=5,
        embargo_days=2,
        mode=mode,
        workers=workers,
        starting_equity=settings.starting_equity,
        commission_per_share=settings.commission_per_share,
        slippage_bps=settings.slippage_bps,
        kill_switch=settings.risk_kill_switch,
        log_level=settings.nautilus_log_level,
        ledger_path=ledger_path,
        study_id=study_id,
    )
