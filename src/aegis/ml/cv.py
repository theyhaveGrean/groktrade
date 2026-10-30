"""Purged and embargoed walk-forward splits for labeled signals.

A training row is dropped when its ``[session, exit_session]`` interval
overlaps the test window, or when the entry falls in the embargo gap
immediately before the test. The date span is refused if it overlaps the
holdout.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from pydantic import BaseModel, ConfigDict

from aegis.backtest.walkforward import walk_forward_splits
from aegis.holdout import DateRange


class PurgedFold(BaseModel):
    model_config = ConfigDict(frozen=True)

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_days: int
    train_rows: int
    purged_rows: int
    test_rows: int


def purge_training_mask(
    sessions: list[date],
    exit_sessions: list[date],
    *,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    embargo_days: int,
) -> list[bool]:
    """True where a row may be used for training."""
    embargo_start = test_start - timedelta(days=embargo_days)
    keep: list[bool] = []
    for session, exit_session in zip(sessions, exit_sessions, strict=True):
        in_train = train_start <= session <= train_end
        overlaps_test = session <= test_end and exit_session >= test_start
        in_embargo = embargo_days > 0 and embargo_start <= session < test_start
        exit_in_embargo = embargo_days > 0 and embargo_start <= exit_session < test_start
        keep.append(in_train and not overlaps_test and not in_embargo and not exit_in_embargo)
    return keep


def purged_folds(
    samples: pl.DataFrame,
    *,
    train_days: int,
    test_days: int,
    embargo_days: int,
    holdout: DateRange,
    step_days: int | None = None,
) -> list[PurgedFold]:
    """Calendar folds over the sample span, with purge counts filled in."""
    if samples.is_empty():
        return []
    start = min(samples["session"].to_list())
    end = max(samples["session"].to_list())
    folds = walk_forward_splits(
        start,
        end,
        train_days=train_days,
        test_days=test_days,
        embargo_days=embargo_days,
        step_days=step_days,
        holdout=holdout,
    )
    sessions = samples["session"].to_list()
    exits = samples["exit_session"].to_list()
    built: list[PurgedFold] = []
    for fold in folds:
        mask = purge_training_mask(
            sessions,
            exits,
            train_start=fold.train_start,
            train_end=fold.train_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            embargo_days=embargo_days,
        )
        in_train_window = [train_start_ok(session, fold.train_start, fold.train_end) for session in sessions]
        purged = sum(1 for window, kept in zip(in_train_window, mask, strict=True) if window and not kept)
        test_rows = sum(1 for session in sessions if fold.test_start <= session <= fold.test_end)
        built.append(
            PurgedFold(
                fold_index=fold.fold_index,
                train_start=fold.train_start,
                train_end=fold.train_end,
                test_start=fold.test_start,
                test_end=fold.test_end,
                embargo_days=embargo_days,
                train_rows=sum(mask),
                purged_rows=purged,
                test_rows=test_rows,
            )
        )
    return built


def train_start_ok(session: date, train_start: date, train_end: date) -> bool:
    return train_start <= session <= train_end
