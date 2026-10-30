"""Phase 4 models. Not implemented in this build.

The first model is XGBoost on price, volume, and volatility features, scored
on expected net R-multiples with purged walk-forward validation. Transformers
come later. Feature builds must use the research bar loader so the holdout
stays unread.
"""
