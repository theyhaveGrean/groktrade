"""XGBoost model that predicts net realized R and ranks signals by that prediction."""

from __future__ import annotations

import polars as pl

from aegis.ml.models import RankerParams


class SignalRanker:
    """Fitted on research rows only. ``rank`` sorts by predicted net R, best first."""

    def __init__(self, params: RankerParams, feature_names: list[str]) -> None:
        self.params = params
        self.feature_names = list(feature_names)
        self._model = None

    def fit(self, frame: pl.DataFrame) -> SignalRanker:
        import xgboost as xgb

        model = xgb.XGBRegressor(
            max_depth=self.params.max_depth,
            learning_rate=self.params.learning_rate,
            n_estimators=self.params.n_estimators,
            min_child_weight=self.params.min_child_weight,
            objective="reg:squarederror",
            random_state=self.params.random_state,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(frame.select(self.feature_names).to_numpy(), frame["net_r"].to_numpy())
        self._model = model
        return self

    def predict(self, frame: pl.DataFrame):
        if self._model is None:
            raise RuntimeError("SignalRanker has not been fit.")
        return self._model.predict(frame.select(self.feature_names).to_numpy())

    def rank(self, frame: pl.DataFrame) -> pl.DataFrame:
        scores = self.predict(frame)
        ranked = frame.with_columns(pl.Series("predicted_net_r", scores)).sort("predicted_net_r", descending=True)
        return ranked.with_columns(pl.arange(1, ranked.height + 1).alias("rank"))

    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {name: 0.0 for name in self.feature_names}
        scores = self._model.get_booster().get_score(importance_type="gain")
        importance = {}
        for index, name in enumerate(self.feature_names):
            importance[name] = float(scores.get(f"f{index}", 0.0))
        return importance
