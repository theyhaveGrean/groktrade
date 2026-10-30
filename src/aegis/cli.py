"""Command line for backfill, cache checks, and the EMA baseline."""

from __future__ import annotations

import json
import logging
from datetime import date

import typer

from aegis.config import load_settings
from aegis.data.pipeline import backfill, build_plan, update
from aegis.data.validation import validate_bars
from aegis.errors import AegisError, BudgetExceeded

app = typer.Typer(no_args_is_help=True, help="Aegis research foundation.")
data_app = typer.Typer(no_args_is_help=True, help="Massive 1-minute cache.")
backtest_app = typer.Typer(no_args_is_help=True, help="EMA crossover baseline.")
research_app = typer.Typer(no_args_is_help=True, help="Phase-3 research agents.")
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(research_app, name="research")


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


@research_app.command("run")
def research_run(
    objective: str = typer.Option(..., help="Research objective for the orchestrator to assign."),
    max_iterations: int = typer.Option(2, "--max-iterations", min=1, help="Propose/critique rounds before the implementer runs."),
    as_json: bool = typer.Option(False, "--json", help="Print the run record as JSON."),
) -> None:
    """Run steps 1-4: assign, propose, critique, implement. Stops before a backtest."""
    _configure_logging()
    from aegis.agents.workflow import run_research

    settings = load_settings()
    try:
        result = run_research(settings, objective=objective, max_iterations=max_iterations)
    except BudgetExceeded as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        partial = getattr(exc, "partial", None)
        if partial is not None:
            typer.echo(f"audit: {partial.audit_path}")
        raise typer.Exit(code=1) from exc
    except (AegisError, ValueError) as exc:
        _fail(exc if isinstance(exc, AegisError) else AegisError(str(exc)))
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    critique = "none" if result.critique is None else result.critique.verdict
    sandbox = "not run" if result.implementation is None else result.implementation.status
    typer.echo(f"run_id: {result.run_id}")
    typer.echo(f"iterations: {result.iterations}")
    typer.echo(f"proposal: {result.proposal.name if result.proposal else 'none'}")
    typer.echo(f"critique: {critique}")
    typer.echo(f"sandbox: {sandbox}")
    typer.echo(f"content_hash: {result.candidate_hash or 'none'}")
    typer.echo(f"advance_to_backtest: {str(result.advance_to_backtest).lower()}")
    typer.echo(f"backtest_started: {str(result.backtest_started).lower()}")
    typer.echo(f"stop_reason: {result.stop_reason}")
    typer.echo(f"audit: {result.audit_path}")
    if result.implementation is not None:
        typer.echo(f"candidate: {result.implementation.candidate_dir}")


if __name__ == "__main__":
    app()
