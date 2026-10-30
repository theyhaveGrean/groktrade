"""Backtest a sandboxed candidate through the harness optimizer.

The child sandbox never imports this module. It would pull Nautilus in. The
parent calls :func:`walk_forward_candidate` only after the naysayer and the
sandbox have both passed. Trials are appended to the same parquet ledger
``aegis optimize`` writes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import mkdtemp

from aegis.backtest.optimize import OptimizationReport, optimize_on_bars
from aegis.backtest.synthetic import OPTIMIZE_DEMO_DAYS, synthetic_bars
from aegis.config import Settings
from aegis.errors import ResearchError
from aegis.strategies.params import StrategyParams, search_spec
from aegis.strategies.plugin import register_signal
from aegis.strategies.registry import get_strategy, unregister
from aegis.strategies.signal import SignalModel


def register_candidate_source(source: str, name: str):
    """Load ``source`` and register it under ``name``. Re-registering replaces it."""
    if not name.startswith("candidate_"):
        raise ResearchError(f"Candidate strategy names must start with 'candidate_', not {name!r}.")
    module = _load_module(source, f"aegis_candidate_{name}")
    params_cls = getattr(module, "Params", None)
    if not isinstance(params_cls, type) or not issubclass(params_cls, StrategyParams):
        raise ResearchError("Candidate Params must subclass StrategyParams.")
    found = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, SignalModel) and value is not SignalModel
    ]
    if len(found) != 1:
        raise ResearchError(f"Expected one SignalModel subclass, found {len(found)}.")
    unregister(name)
    return register_signal(
        name=name,
        summary=f"Research candidate {name}.",
        params_model=params_cls,
        model=found[0],
    )


def walk_forward_candidate(
    source: str,
    *,
    name: str,
    settings: Settings,
    ledger_path: Path,
) -> OptimizationReport:
    """One-point walk-forward on the synthetic week. No API key and no holdout bars."""
    register_candidate_source(source, name)
    try:
        plugin = get_strategy(name)
        overrides: dict[str, list] = {}
        for field_name, field in plugin.params_model.model_fields.items():
            spec = search_spec(field)
            if spec is None:
                continue
            overrides[field_name] = ["1min"] if field_name == "bar_timeframe" else [spec.default_grid[0]]
        return optimize_on_bars(
            synthetic_bars("AAPL", sessions=OPTIMIZE_DEMO_DAYS),
            strategy=name,
            start=OPTIMIZE_DEMO_DAYS[0],
            end=OPTIMIZE_DEMO_DAYS[-1],
            holdout=settings.holdout,
            param_overrides=overrides,
            method="grid",
            samples=1,
            seed=7,
            train_days=3,
            test_days=3,
            step_days=5,
            embargo_days=2,
            mode="rolling",
            workers=1,
            starting_equity=settings.starting_equity,
            commission_per_share=settings.commission_per_share,
            slippage_bps=settings.slippage_bps,
            kill_switch=settings.risk_kill_switch,
            log_level=settings.nautilus_log_level,
            ledger_path=ledger_path,
            study_id=name,
            candidate_source=source,
            candidate_name=name,
        )
    finally:
        unregister(name)


def _load_module(source: str, module_name: str):
    directory = Path(mkdtemp(prefix="aegis-candidate-"))
    path = directory / "candidate.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ResearchError("Could not load the candidate module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
