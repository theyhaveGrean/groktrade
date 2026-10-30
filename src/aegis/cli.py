"""Command line for backfill, cache checks, the EMA baseline, analyst reports, and paper trading."""

from __future__ import annotations

import json
import logging
from datetime import date

import typer

from aegis.config import load_settings
from aegis.data.pipeline import backfill, build_plan, update
from aegis.data.validation import validate_bars
from aegis.errors import AegisError

app = typer.Typer(no_args_is_help=True, help="Aegis research foundation.")
data_app = typer.Typer(no_args_is_help=True, help="Massive 1-minute cache.")
backtest_app = typer.Typer(no_args_is_help=True, help="EMA crossover baseline.")
paper_app = typer.Typer(no_args_is_help=True, help="Alpaca paper trading. Live capital is refused.")
analyze_app = typer.Typer(no_args_is_help=True, help="Backtest analyst report. Does not run a strategy.")
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(paper_app, name="paper")
app.add_typer(analyze_app, name="analyze")


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
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Run the EMA baseline on the built-in synthetic sessions. No API key."""
    _configure_logging()
    from aegis.backtest.engine import run_demo_backtest

    settings = load_settings()
    try:
        report = run_demo_backtest(settings)
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(json.dumps(report.to_dict(), indent=2) if as_json else report.render())


@backtest_app.command("run")
def backtest_run(
    start: str = typer.Option(..., help="YYYY-MM-DD."),
    end: str = typer.Option(..., help="YYYY-MM-DD."),
    symbols: str | None = typer.Option(None, help="Comma-separated tickers."),
    fast: int = typer.Option(20, help="Fast EMA period."),
    slow: int = typer.Option(50, help="Slow EMA period."),
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
            fast_period=fast,
            slow_period=slow,
        )
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(json.dumps(report.to_dict(), indent=2) if as_json else report.render())


@backtest_app.command("walk-forward")
def backtest_walk_forward(
    start: str = typer.Option(..., help="YYYY-MM-DD."),
    end: str = typer.Option(..., help="YYYY-MM-DD."),
    train_days: int = typer.Option(90, help="Calendar days in each training fold."),
    test_days: int = typer.Option(30, help="Calendar days in each test fold."),
    embargo_days: int = typer.Option(1, help="Calendar days dropped between train and test."),
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
            embargo_days=embargo_days,
            holdout=settings.holdout,
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


def _paper_symbols(symbols: str | None):
    from datetime import datetime, timezone

    from aegis.data.pipeline import resolve_members
    from aegis.data.sessions import eastern_date

    settings = load_settings()
    chosen = resolve_members(settings, _symbols(symbols), eastern_date(datetime.now(timezone.utc)))
    return settings, [member.symbol for member in chosen]


@paper_app.command("run")
def paper_run(
    strategy: str = typer.Option(..., "--strategy", help="Strategy id. This build supports ema_crossover."),
    symbols: str | None = typer.Option(None, help="Comma-separated tickers. Default: the starter universe."),
) -> None:
    """Trade the Alpaca paper account from the IEX minute stream.

    Refuses to start unless TRADING_MODE=paper, LIVE_TRADING_ENABLED=false, and
    ALPACA_BASE_URL is the paper endpoint. Ctrl-C cancels orders and flattens.
    """
    _configure_logging()
    from aegis.execution.broker import connect_paper_broker
    from aegis.execution.feed import connect_iex_feed
    from aegis.execution.session import PaperSession

    try:
        settings, names = _paper_symbols(symbols)
        if strategy != "ema_crossover":
            raise AegisError(f"Unknown strategy {strategy!r}. This paper build runs ema_crossover.")
        broker = connect_paper_broker(settings)
        feed = connect_iex_feed(settings)
        session = PaperSession(settings, broker, names, strategy=strategy)
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    try:
        session.serve(feed)
    except KeyboardInterrupt:
        try:
            session.flatten_all(reason="shutdown")
        except Exception as exc:
            typer.secho(f"Shutdown flatten failed: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=130) from None


@paper_app.command("status")
def paper_status(
    as_json: bool = typer.Option(False, "--json", help="Print the status as JSON."),
) -> None:
    """Show the kill switch, broker equity, positions, and the last reconciliation."""
    _configure_logging()
    import json

    from aegis.execution.broker import connect_paper_broker
    from aegis.execution.session import build_status

    settings = load_settings()
    try:
        report = build_status(settings, connect_paper_broker(settings))
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo(json.dumps(report.to_dict(), indent=2) if as_json else report.render())
    if report.mismatch:
        raise typer.Exit(code=2)


@paper_app.command("kill")
def paper_kill() -> None:
    """Arm the kill file, cancel every open order, and flatten the paper account."""
    _configure_logging()
    from aegis.execution.broker import connect_paper_broker
    from aegis.execution.session import execute_kill

    settings = load_settings()
    try:
        execute_kill(settings, connect_paper_broker(settings), reason="aegis paper kill")
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    typer.echo("Kill switch armed. Canceled open orders and flattened positions.")


def _emit_analysis(report, *, as_json: bool, out_dir: str | None) -> None:
    from pathlib import Path

    from aegis.analyst import write_analyst_report

    if out_dir:
        markdown_path, json_path = write_analyst_report(report, Path(out_dir))
        typer.echo(f"Wrote {markdown_path} and {json_path}")
    typer.echo(report.to_json() if as_json else report.to_markdown())


@analyze_app.command("demo")
def analyze_demo(
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    out_dir: str | None = typer.Option(None, "--out-dir", help="Write report.md and report.json here."),
) -> None:
    """Print a full analyst report from a built-in synthetic result. No API key."""
    from aegis.analyst import build_analyst_report, demo_analysis_input

    settings = load_settings()
    settings.assert_paper_mode()
    try:
        report = build_analyst_report(demo_analysis_input(), holdout=settings.holdout)
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    _emit_analysis(report, as_json=as_json, out_dir=out_dir)


@analyze_app.command("report")
def analyze_report(
    input_path: str = typer.Option(..., "--input", help="Schema v1 JSON: trades, equity curve, optional folds."),
    bars: str | None = typer.Option(None, "--bars", help="Parquet bars (symbol, ts, close) for benchmark and regimes."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    out_dir: str | None = typer.Option(None, "--out-dir", help="Write report.md and report.json here."),
) -> None:
    """Build the analyst report from a result file. Refuses the holdout. Does not run a strategy."""
    from pathlib import Path

    from aegis.analyst import build_analyst_report, load_analysis_input

    settings = load_settings()
    settings.assert_paper_mode()
    try:
        analysis = load_analysis_input(Path(input_path), Path(bars) if bars else None)
        report = build_analyst_report(analysis, holdout=settings.holdout)
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    _emit_analysis(report, as_json=as_json, out_dir=out_dir)


if __name__ == "__main__":
    app()
