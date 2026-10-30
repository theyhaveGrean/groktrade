"""Command line for backfill, cache checks, and the EMA baseline."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import typer

from aegis.config import load_settings
from aegis.data.pipeline import backfill, build_plan, update
from aegis.data.validation import validate_bars
from aegis.errors import AegisError

app = typer.Typer(no_args_is_help=True, help="Aegis research foundation.")
data_app = typer.Typer(no_args_is_help=True, help="Massive 1-minute cache.")
backtest_app = typer.Typer(no_args_is_help=True, help="Registered strategy backtests.")
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _symbols(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def _fail(exc: AegisError) -> None:
    typer.secho(str(exc), err=True, fg=typer.colors.RED)
    raise typer.Exit(code=1) from exc


@data_app.command("plan")
def data_plan(
    start: str | None = typer.Option(None, help="YYYY-MM-DD. Default: start of the lookback window."),
    end: str | None = typer.Option(None, help="YYYY-MM-DD. Default: latest requestable session."),
    symbols: str | None = typer.Option(None, help="Comma-separated tickers. Default: universe as of the end date."),
) -> None:
    """Show chunk count and free-tier wall time without calling Massive."""
    _configure_logging()
    settings = load_settings()
    try:
        plan = build_plan(settings, start=_parse_date(start), end=_parse_date(end), symbols=_symbols(symbols))
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(plan.render())


@data_app.command("backfill")
def data_backfill(
    start: str = typer.Option(..., help="YYYY-MM-DD."),
    end: str | None = typer.Option(None, help="YYYY-MM-DD. Default: latest requestable session."),
    symbols: str | None = typer.Option(None, help="Comma-separated tickers. Default: the starter universe."),
) -> None:
    """Download 1-minute bars. Safe to stop and run again; cached days are skipped."""
    _configure_logging()
    settings = load_settings()
    try:
        result = backfill(settings, start=date.fromisoformat(start), end=_parse_date(end), symbols=_symbols(symbols))
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(result.render())


@data_app.command("update")
def data_update(
    symbols: str | None = typer.Option(None, help="Comma-separated tickers. Default: cached universe names."),
    skip_corporate_actions: bool = typer.Option(
        False,
        help="Do not refresh splits and dividends. Saves about 40 requests on the free tier.",
    ),
) -> None:
    """Fill holes and extend cached symbols through the latest delayed session."""
    _configure_logging()
    settings = load_settings()
    try:
        result = update(
            settings,
            symbols=_symbols(symbols),
            refresh_corporate_actions=not skip_corporate_actions,
        )
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(result.render())


@data_app.command("validate")
def data_validate(
    start: str = typer.Option(..., help="YYYY-MM-DD."),
    end: str = typer.Option(..., help="YYYY-MM-DD."),
    symbols: str | None = typer.Option(None, help="Comma-separated tickers. Default: the starter universe."),
) -> None:
    """Check cached bars for gaps, bad prints, and extended-hours rows."""
    _configure_logging()
    settings = load_settings()
    from aegis.data.cache import BarCache
    from aegis.data.pipeline import resolve_members
    from aegis.data.query import load_maintenance_bars

    try:
        chosen = resolve_members(settings, _symbols(symbols), date.fromisoformat(end))
        names = [member.symbol for member in chosen]
        cache = BarCache(settings.data_cache_dir)
        bars = load_maintenance_bars(cache, names, date.fromisoformat(start), date.fromisoformat(end))
        splits = cache.read_splits(names)
        report = validate_bars(bars, splits=splits)
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(report.render())
    if report.errors:
        raise typer.Exit(code=1)


@backtest_app.command("demo")
def backtest_demo(
    strategy: str = typer.Option("ema_cross", "--strategy", help="Registered strategy name."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Run a strategy on the built-in synthetic sessions. No API key."""
    _configure_logging()
    from aegis.backtest.engine import run_demo_backtest

    settings = load_settings()
    try:
        report = run_demo_backtest(settings, strategy=strategy)
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(json.dumps(report.to_dict(), indent=2) if as_json else report.render())


@backtest_app.command("run")
def backtest_run(
    start: str = typer.Option(..., help="YYYY-MM-DD."),
    end: str = typer.Option(..., help="YYYY-MM-DD."),
    symbols: str | None = typer.Option(None, help="Comma-separated tickers."),
    strategy: str = typer.Option("ema_cross", "--strategy", help="Registered strategy name."),
    fast: int | None = typer.Option(None, help="Fast period when the strategy has fast_period."),
    slow: int | None = typer.Option(None, help="Slow period when the strategy has slow_period."),
    timeframe: str | None = typer.Option(None, "--timeframe", help="1min, 5min, 15min, 30min, or 60min."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Backtest cached bars. Refuses a range that overlaps the holdout."""
    _configure_logging()
    from aegis.backtest.engine import run_research_backtest

    settings = load_settings()
    try:
        report = run_research_backtest(
            settings,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            symbols=_symbols(symbols),
            strategy=strategy,
            fast_period=fast,
            slow_period=slow,
            bar_timeframe=timeframe,
        )
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(json.dumps(report.to_dict(), indent=2) if as_json else report.render())


@backtest_app.command("signals")
def backtest_signals() -> None:
    """Reject signal modules that import outside the sandbox allowlist."""
    from aegis.strategies.policy import check_signal_package

    result = check_signal_package()
    if result.passed:
        typer.echo(f"Signal policy passed ({len(result.files_checked)} files).")
        return
    for violation in result.violations:
        typer.echo(f"{violation.path}:{violation.line}: {violation.message}", err=True)
    raise typer.Exit(code=1)


@backtest_app.command("walk-forward")
def backtest_walk_forward(
    start: str = typer.Option(..., help="YYYY-MM-DD."),
    end: str = typer.Option(..., help="YYYY-MM-DD."),
    train_days: int = typer.Option(90, help="Calendar days in each training fold."),
    test_days: int = typer.Option(30, help="Calendar days in each test fold."),
    embargo_days: int = typer.Option(1, help="Calendar days dropped between train and test."),
    step_days: int | None = typer.Option(None, help="Calendar days to advance each fold. Default: test-days."),
    mode: str = typer.Option("rolling", help="rolling or anchored."),
) -> None:
    """Print walk-forward folds. Refuses a window that overlaps the holdout."""
    from aegis.backtest.walkforward import walk_forward_splits

    settings = load_settings()
    settings.assert_paper_mode()
    try:
        folds = walk_forward_splits(
            date.fromisoformat(start),
            date.fromisoformat(end),
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            embargo_days=embargo_days,
            holdout=settings.holdout,
            mode=mode,
        )
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    if not folds:
        typer.echo("No complete folds in that window.")
        return
    for fold in folds:
        typer.echo(
            f"{fold.fold_index}: train {fold.train_start.isoformat()}..{fold.train_end.isoformat()} "
            f"test {fold.test_start.isoformat()}..{fold.test_end.isoformat()}"
        )


optimize_app = typer.Typer(no_args_is_help=True, help="Walk-forward parameter search.")
app.add_typer(optimize_app, name="optimize")


def _parse_sets(items: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects name=v1,v2, got {item!r}.")
        name, raw = item.split("=", 1)
        values = [part.strip() for part in raw.split(",") if part.strip()]
        if not name.strip() or not values:
            raise ValueError(f"--set expects name=v1,v2, got {item!r}.")
        parsed[name.strip()] = values
    return parsed


@optimize_app.command("strategies")
def optimize_strategies() -> None:
    """List registered strategies."""
    from aegis.strategies import list_strategies

    for plugin in list_strategies():
        typer.echo(f"{plugin.name}: {plugin.summary}")


@optimize_app.command("space")
def optimize_space(
    strategy: str = typer.Argument("ema_cross", help="Registered strategy name."),
) -> None:
    """Print a strategy's defaults, search axes, and valid grid size."""
    from aegis.strategies import describe_search_space, get_strategy

    try:
        plugin = get_strategy(strategy)
    except ValueError as exc:
        _fail(AegisError(str(exc)))
    typer.echo(f"{plugin.name}: {plugin.summary}")
    typer.echo(describe_search_space(plugin.params_model))


@optimize_app.command("run")
def optimize_run(
    start: str = typer.Option(..., help="YYYY-MM-DD. In-sample and out-of-sample must both sit before the holdout."),
    end: str = typer.Option(..., help="YYYY-MM-DD."),
    strategy: str = typer.Option("ema_cross", help="Registered strategy name. See `aegis optimize space`."),
    symbols: str | None = typer.Option(None, help="Comma-separated tickers. Default: universe as of the end date."),
    set_param: list[str] = typer.Option(
        [],
        "--set",
        help="Override one search axis: name=v1,v2. Repeat for each parameter, including bar_timeframe.",
    ),
    method: str = typer.Option("grid", help="grid or random."),
    samples: int = typer.Option(20, help="Draws for random search. Ignored by grid."),
    seed: int = typer.Option(7, help="Random-search seed."),
    train_days: int = typer.Option(60, help="Calendar days in the first training window."),
    test_days: int = typer.Option(20, help="Calendar days in each out-of-sample window."),
    step_days: int | None = typer.Option(None, help="Advance per fold. Default: test-days. Must be >= test-days."),
    embargo_days: int = typer.Option(1, help="Calendar days dropped between train and test."),
    mode: str = typer.Option("rolling", help="rolling or anchored."),
    workers: int | None = typer.Option(None, help="Processes. Default: min(cpu-1, 6)."),
    ledger: str = typer.Option("data/cache/trials/trials.parquet", help="Parquet ledger of every trial."),
) -> None:
    """Search a strategy's parameter model in-sample. Score only the next out-of-sample window."""
    _configure_logging()
    from aegis.backtest.optimize import optimize_research

    settings = load_settings()
    try:
        report = optimize_research(
            settings,
            strategy=strategy,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            symbols=_symbols(symbols),
            param_overrides=_parse_sets(set_param),
            method=method,
            samples=samples,
            seed=seed,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            embargo_days=embargo_days,
            mode=mode,
            workers=workers,
            ledger_path=Path(ledger),
        )
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(report.render())


@optimize_app.command("demo")
def optimize_demo_cmd(
    strategy: str = typer.Option("ema_cross", help="Registered strategy name."),
    set_param: list[str] = typer.Option(
        [],
        "--set",
        help="Override one smoke-grid axis: name=v1,v2. Repeatable. Includes bar_timeframe.",
    ),
    method: str = typer.Option("grid", help="grid or random."),
    samples: int = typer.Option(4, help="Draws for random search."),
    seed: int = typer.Option(7, help="Random-search seed."),
    mode: str = typer.Option("rolling", help="rolling or anchored."),
    workers: int | None = typer.Option(None, help="Processes. Default: min(cpu-1, 6)."),
    ledger: str = typer.Option("data/cache/trials/trials.parquet", help="Parquet ledger of every trial."),
) -> None:
    """Run the search on a built-in synthetic week. No API key and no cache."""
    _configure_logging()
    from aegis.backtest.optimize import optimize_demo

    settings = load_settings()
    try:
        report = optimize_demo(
            settings,
            strategy=strategy,
            param_overrides=_parse_sets(set_param) or None,
            method=method,
            samples=samples,
            seed=seed,
            mode=mode,
            workers=workers,
            ledger_path=Path(ledger),
        )
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(report.render())


if __name__ == "__main__":
    app()
