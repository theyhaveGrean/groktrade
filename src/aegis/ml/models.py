"""Pydantic results for the signal-ranking pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RankerParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_depth: int = Field(default=3, ge=1, le=8)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    n_estimators: int = Field(default=40, ge=1, le=500)
    min_child_weight: float = Field(default=1.0, ge=0)
    random_state: int = 0


class LookaheadFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    ts: str
    column: str
    full_value: float | None
    prefix_value: float | None


class LookaheadCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    cuts_checked: int
    failures: list[LookaheadFailure] = Field(default_factory=list)


class PurgeCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    folds_checked: int
    violations: list[str] = Field(default_factory=list)


class LeakageReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    lookahead: LookaheadCheck
    purge: PurgeCheck
    notes: list[str] = Field(default_factory=list)


class FoldMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    fold_index: int
    train_rows: int
    test_rows: int
    purged_rows: int
    rmse: float | None
    spearman: float | None
    top_half_mean_r: float | None


class HyperparameterTrial(BaseModel):
    model_config = ConfigDict(frozen=True)

    params: RankerParams
    mean_spearman: float | None
    mean_top_half_r: float | None
    score: float | None


class FeatureStabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature_names: list[str]
    importance_by_fold: list[dict[str, float]]
    consecutive_spearman: list[float | None]
    mean_spearman: float | None
    stable: bool | None


class BaselineComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    ml_top_half_mean_r: float | None
    feature_baseline_top_half_mean_r: float | None
    feature_baseline_name: str
    take_all_mean_r: float | None
    lift_vs_take_all: float | None
    lift_vs_feature_baseline: float | None
    oof_count: int


class TrainingResult(BaseModel):
    """What the ML Specialist agent reads after a research fit.

    Metrics are out-of-fold. The returned ranker is refit on the research
    sample. Holdout dates are refused before any fit.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    model_family: Literal["xgboost_signal_ranker"] = "xgboost_signal_ranker"
    target: Literal["net_realized_r"] = "net_realized_r"
    feature_names: list[str]
    best_params: RankerParams
    trials: list[HyperparameterTrial]
    folds: list[FoldMetric]
    leakage: LeakageReport
    stability: FeatureStabilityReport
    baseline: BaselineComparison
    mlflow_tracking_uri: str
    mlflow_run_id: str | None
    mlflow_experiment: str
    sample_count: int
    holdout_protected: Literal[True] = True
    notes: list[str] = Field(default_factory=list)
