"""Train a signal ranker on research bars and refuse the holdout.

The public entry point is :func:`train_signal_ranker`. It returns the fitted
ranker and a :class:`~aegis.ml.models.TrainingResult` the ML Specialist agent
can serialize.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from aegis.errors import AegisError, HoldoutViolation
from aegis.holdout import DateRange, assert_dates_outside_holdout
from aegis.ml.cv import purged_folds
from aegis.ml.features import FEATURE_COLUMNS, compute_features
from aegis.ml.labels import candidate_signals, label_signals
from aegis.ml.leakage import audit_point_in_time, audit_purge
from aegis.ml.metrics import rmse, spearman, top_half_mean
from aegis.ml.models import (
    BaselineComparison,
    FeatureStabilityReport,
    FoldMetric,
    HyperparameterTrial,
    LeakageReport,
    RankerParams,
    TrainingResult,
)
from aegis.ml.ranker import SignalRanker

DEFAULT_GRID = (
    RankerParams(max_depth=2, learning_rate=0.1, n_estimators=40, min_child_weight=1.0),
    RankerParams(max_depth=3, learning_rate=0.05, n_estimators=40, min_child_weight=1.0),
)


def train_signal_ranker(
    bars: pl.DataFrame,
    *,
    holdout: DateRange,
    train_days: int = 4,
    test_days: int = 2,
    embargo_days: int = 1,
    every_n: int = 15,
    grid: list[RankerParams] | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "aegis-signal-ranker",
    baseline_feature: str = "ret_5",
) -> tuple[SignalRanker, TrainingResult]:
    """Fit on research bars. Raises if any bar date sits in ``holdout``."""
    assert_research_bars(bars, holdout, purpose="signal-ranker training")
    lookahead = audit_point_in_time(bars, compute_features, FEATURE_COLUMNS)
    if not lookahead.passed:
        raise AegisError("Feature lookahead check failed. Refusing to train.")
    dataset = build_dataset(bars, holdout=holdout, every_n=every_n)
    if dataset.height < 12:
        raise AegisError(
            f"Only {dataset.height} labeled signals survived warmup. Need at least 12 to train."
        )
    folds = purged_folds(
        dataset,
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
        holdout=holdout,
    )
    usable = [fold for fold in folds if fold.train_rows >= 8 and fold.test_rows >= 2]
    if not usable:
        raise AegisError("Purged walk-forward produced no fold with enough train and test rows.")
    trials: list[HyperparameterTrial] = []
    best_params = (grid or list(DEFAULT_GRID))[0]
    best_score = float("-inf")
    best_folds: list[FoldMetric] = []
    best_oof: pl.DataFrame | None = None
    best_importances: list[dict[str, float]] = []
    for params in grid or list(DEFAULT_GRID):
        metrics, oof, importances = _evaluate(dataset, usable, params)
        score = _mean([item.spearman for item in metrics])
        if score is None:
            score = _mean([item.top_half_mean_r for item in metrics])
        trial_score = float("-inf") if score is None else score
        trials.append(
            HyperparameterTrial(
                params=params,
                mean_spearman=_mean([item.spearman for item in metrics]),
                mean_top_half_r=_mean([item.top_half_mean_r for item in metrics]),
                score=None if score is None else score,
            )
        )
        if trial_score > best_score:
            best_score = trial_score
            best_params = params
            best_folds = metrics
            best_oof = oof
            best_importances = importances
    purge = audit_purge(dataset, usable)
    leakage = LeakageReport(
        passed=lookahead.passed and purge.passed,
        lookahead=lookahead,
        purge=purge,
        notes=["Features were recomputed on truncated history.", "Training rows whose labels overlap a test window are purged."],
    )
    if not leakage.passed:
        raise AegisError("Leakage checks failed. Refusing to train.")
    stability = _stability(list(FEATURE_COLUMNS), best_importances)
    baseline = _baseline(best_oof if best_oof is not None else dataset.clear(), baseline_feature)
    ranker = SignalRanker(best_params, list(FEATURE_COLUMNS)).fit(dataset)
    uri = resolve_tracking_uri(tracking_uri)
    notes = [
        "Metrics are out-of-fold on purged walk-forward test windows.",
        "The returned model is refit on every research sample after the search.",
        "Holdout sessions were refused before features were fit.",
    ]
    result = _log(
        uri=uri,
        experiment_name=experiment_name,
        best_params=best_params,
        trials=trials,
        folds=best_folds,
        leakage=leakage,
        stability=stability,
        baseline=baseline,
        sample_count=dataset.height,
        notes=notes,
    )
    return ranker, result


def build_dataset(bars: pl.DataFrame, *, holdout: DateRange, every_n: int = 15) -> pl.DataFrame:
    """Join point-in-time features to net-R labels. Refuses holdout dates."""
    assert_research_bars(bars, holdout, purpose="signal-ranker dataset")
    features = compute_features(bars)
    signals = candidate_signals(bars, every_n=every_n)
    labels = label_signals(bars, signals)
    if labels.is_empty():
        return labels
    feature_frame = features.select(["symbol", "ts", *FEATURE_COLUMNS])
    return labels.join(feature_frame, on=["symbol", "ts"], how="inner").drop_nulls(list(FEATURE_COLUMNS))


def assert_research_bars(bars: pl.DataFrame, holdout: DateRange, *, purpose: str) -> None:
    if bars.is_empty():
        return
    dates = set(
        bars.select(pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()).to_series().to_list()
    )
    assert_dates_outside_holdout(dates, holdout, purpose=purpose)


def resolve_tracking_uri(value: str | None) -> str:
    if value is None:
        root = Path("mlruns")
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve().as_uri()
    if value.startswith("file:"):
        return value
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve().as_uri()


def _evaluate(
    dataset: pl.DataFrame,
    folds,
    params: RankerParams,
) -> tuple[list[FoldMetric], pl.DataFrame, list[dict[str, float]]]:
    metrics: list[FoldMetric] = []
    oof_parts: list[pl.DataFrame] = []
    importances: list[dict[str, float]] = []
    for fold in folds:
        train = _mask_frame(dataset, fold, role="train")
        test = dataset.filter(
            (pl.col("session") >= fold.test_start) & (pl.col("session") <= fold.test_end)
        )
        metric = FoldMetric(
            fold_index=fold.fold_index,
            train_rows=train.height,
            test_rows=test.height,
            purged_rows=fold.purged_rows,
            rmse=None,
            spearman=None,
            top_half_mean_r=None,
        )
        if train.height >= 8 and test.height >= 2:
            ranker = SignalRanker(params, list(FEATURE_COLUMNS)).fit(train)
            predicted = np.asarray(ranker.predict(test), dtype=float)
            actual = np.asarray(test["net_r"].to_numpy(), dtype=float)
            metric = FoldMetric(
                fold_index=fold.fold_index,
                train_rows=train.height,
                test_rows=test.height,
                purged_rows=fold.purged_rows,
                rmse=rmse(actual, predicted),
                spearman=spearman(predicted, actual),
                top_half_mean_r=top_half_mean(predicted, actual),
            )
            oof_parts.append(test.with_columns(pl.Series("predicted_net_r", predicted)))
            importances.append(ranker.feature_importance())
        metrics.append(metric)
    oof = pl.concat(oof_parts, how="vertical_relaxed") if oof_parts else dataset.clear()
    return metrics, oof, importances


def _mask_frame(dataset: pl.DataFrame, fold, *, role: str) -> pl.DataFrame:
    from aegis.ml.cv import purge_training_mask

    sessions = dataset["session"].to_list()
    exits = dataset["exit_session"].to_list()
    mask = purge_training_mask(
        sessions,
        exits,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        embargo_days=fold.embargo_days,
    )
    if role != "train":
        raise ValueError(role)
    return dataset.with_columns(pl.Series("_keep", mask)).filter(pl.col("_keep")).drop("_keep")


def _stability(names: list[str], importances: list[dict[str, float]]) -> FeatureStabilityReport:
    correlations: list[float | None] = []
    for left, right in zip(importances, importances[1:], strict=False):
        left_values = np.array([left.get(name, 0.0) for name in names], dtype=float)
        right_values = np.array([right.get(name, 0.0) for name in names], dtype=float)
        correlations.append(spearman(left_values, right_values))
    mean = _mean(correlations)
    stable = None if mean is None else mean >= 0.5
    return FeatureStabilityReport(
        feature_names=names,
        importance_by_fold=importances,
        consecutive_spearman=correlations,
        mean_spearman=mean,
        stable=stable,
    )


def _baseline(oof: pl.DataFrame, feature: str) -> BaselineComparison:
    if oof.is_empty() or "predicted_net_r" not in oof.columns or feature not in oof.columns:
        return BaselineComparison(
            ml_top_half_mean_r=None,
            feature_baseline_top_half_mean_r=None,
            feature_baseline_name=feature,
            take_all_mean_r=None,
            lift_vs_take_all=None,
            lift_vs_feature_baseline=None,
            oof_count=0,
        )
    actual = np.asarray(oof["net_r"].to_numpy(), dtype=float)
    predicted = np.asarray(oof["predicted_net_r"].to_numpy(), dtype=float)
    baseline_scores = np.asarray(oof[feature].to_numpy(), dtype=float)
    ml_half = top_half_mean(predicted, actual)
    base_half = top_half_mean(baseline_scores, actual)
    take_all = float(actual.mean()) if len(actual) else None
    return BaselineComparison(
        ml_top_half_mean_r=ml_half,
        feature_baseline_top_half_mean_r=base_half,
        feature_baseline_name=feature,
        take_all_mean_r=take_all,
        lift_vs_take_all=None if ml_half is None or take_all is None else ml_half - take_all,
        lift_vs_feature_baseline=None if ml_half is None or base_half is None else ml_half - base_half,
        oof_count=len(actual),
    )


def _log(
    *,
    uri: str,
    experiment_name: str,
    best_params: RankerParams,
    trials: list[HyperparameterTrial],
    folds: list[FoldMetric],
    leakage: LeakageReport,
    stability: FeatureStabilityReport,
    baseline: BaselineComparison,
    sample_count: int,
    notes: list[str],
) -> TrainingResult:
    import os

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="signal-ranker") as run:
        mlflow.log_params(best_params.model_dump())
        mlflow.log_metric("sample_count", sample_count)
        mlflow.log_metric("holdout_protected", 1.0)
        if baseline.ml_top_half_mean_r is not None:
            mlflow.log_metric("ml_top_half_mean_r", baseline.ml_top_half_mean_r)
        if baseline.take_all_mean_r is not None:
            mlflow.log_metric("take_all_mean_r", baseline.take_all_mean_r)
        if baseline.lift_vs_take_all is not None:
            mlflow.log_metric("lift_vs_take_all", baseline.lift_vs_take_all)
        spearman_mean = _mean([item.spearman for item in folds])
        if spearman_mean is not None:
            mlflow.log_metric("cv_spearman", spearman_mean)
        result = TrainingResult(
            feature_names=list(FEATURE_COLUMNS),
            best_params=best_params,
            trials=trials,
            folds=folds,
            leakage=leakage,
            stability=stability,
            baseline=baseline,
            mlflow_tracking_uri=uri,
            mlflow_run_id=run.info.run_id,
            mlflow_experiment=experiment_name,
            sample_count=sample_count,
            notes=notes,
        )
        mlflow.log_text(result.model_dump_json(indent=2), "training_result.json")
        for index, trial in enumerate(trials):
            with mlflow.start_run(run_name=f"trial-{index}", nested=True):
                mlflow.log_params(trial.params.model_dump())
                if trial.score is not None:
                    mlflow.log_metric("score", trial.score)
    return result


def _mean(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return float(sum(kept) / len(kept))


# Re-export so callers can catch the holdout error from this module.
__all__ = [
    "HoldoutViolation",
    "build_dataset",
    "train_signal_ranker",
]
