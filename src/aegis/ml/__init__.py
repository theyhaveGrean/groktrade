"""Research models for the ML Specialist.

:func:`train_signal_ranker` builds point-in-time features, labels candidate
signals with net realized R after costs, and fits an XGBoost ranker under
purged walk-forward validation. Runs are logged to a local MLflow file store.
The holdout window is refused. Nothing here places an order.
"""

from aegis.ml.features import FEATURE_COLUMNS, compute_features
from aegis.ml.labels import candidate_signals, label_signals
from aegis.ml.leakage import audit_point_in_time
from aegis.ml.models import TrainingResult
from aegis.ml.pipeline import build_dataset, train_signal_ranker
from aegis.ml.ranker import SignalRanker

__all__ = [
    "FEATURE_COLUMNS",
    "SignalRanker",
    "TrainingResult",
    "audit_point_in_time",
    "build_dataset",
    "candidate_signals",
    "compute_features",
    "label_signals",
    "train_signal_ranker",
]
