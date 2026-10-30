"""Lookahead and purge checks.

:func:`audit_point_in_time` recomputes features on a prefix of the bars and
compares them to the full-sample row. A feature that reads the future changes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl

from aegis.ml.cv import purge_training_mask
from aegis.ml.models import LookaheadCheck, LookaheadFailure, PurgeCheck


def audit_point_in_time(
    bars: pl.DataFrame,
    feature_fn: Callable[[pl.DataFrame], pl.DataFrame],
    columns: tuple[str, ...] | list[str],
    *,
    cuts: int = 3,
) -> LookaheadCheck:
    """Fail when a prefix of history does not reproduce the full-sample features."""
    if bars.is_empty():
        return LookaheadCheck(passed=False, cuts_checked=0, failures=[])
    full = feature_fn(bars).sort(["symbol", "ts"])
    failures: list[LookaheadFailure] = []
    checked = 0
    for symbol in full["symbol"].unique().to_list():
        part = full.filter(pl.col("symbol") == symbol)
        count = part.height
        if count < 8:
            continue
        indexes = _cut_indexes(count, cuts)
        for index in indexes:
            row = part.row(index, named=True)
            ts = row["ts"]
            prefix = feature_fn(bars.filter(pl.col("ts") <= ts))
            match = prefix.filter((pl.col("symbol") == symbol) & (pl.col("ts") == ts))
            checked += 1
            prefix_row = match.row(0, named=True) if match.height else None
            for column in columns:
                full_value = _as_float(row.get(column))
                prefix_value = None if prefix_row is None else _as_float(prefix_row.get(column))
                if not _same(full_value, prefix_value):
                    failures.append(
                        LookaheadFailure(
                            symbol=str(symbol),
                            ts=str(ts),
                            column=column,
                            full_value=full_value,
                            prefix_value=prefix_value,
                        )
                    )
    return LookaheadCheck(passed=checked > 0 and not failures, cuts_checked=checked, failures=failures)


def audit_purge(samples: pl.DataFrame, folds) -> PurgeCheck:
    """Fail when a kept training row's label interval overlaps its test window."""
    if samples.is_empty():
        return PurgeCheck(passed=False, folds_checked=0, violations=["No labeled samples."])
    sessions = samples["session"].to_list()
    exits = samples["exit_session"].to_list()
    violations: list[str] = []
    for fold in folds:
        mask = purge_training_mask(
            sessions,
            exits,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            embargo_days=fold.embargo_days,
        )
        for session, exit_session, kept in zip(sessions, exits, mask, strict=True):
            if not kept:
                continue
            if session <= fold.test_end and exit_session >= fold.test_start:
                violations.append(
                    f"Fold {fold.fold_index} kept a label from {session.isoformat()} "
                    f"that exits {exit_session.isoformat()} inside the test window."
                )
    return PurgeCheck(passed=not violations, folds_checked=len(folds), violations=violations)


def _cut_indexes(count: int, cuts: int) -> list[int]:
    if cuts <= 1:
        return [count // 2]
    indexes = []
    for step in range(1, cuts + 1):
        indexes.append(min(count - 1, max(1, (count * step) // (cuts + 1))))
    return sorted(set(indexes))


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _same(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= 1e-8
