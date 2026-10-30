"""Calendar walk-forward splits that refuse the protected holdout.

Folds are calendar days, not trading sessions. ``embargo_days`` is the gap
between the last training day and the first test day. Rolling mode slides a
fixed training window. Anchored mode keeps the training start fixed and grows
the training end. The research caller passes the holdout; if the requested
window overlaps it, nothing is returned and
:class:`~aegis.errors.HoldoutViolation` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from aegis.holdout import DateRange, assert_no_holdout_overlap


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date

    @property
    def train(self) -> DateRange:
        return DateRange(self.train_start, self.train_end)

    @property
    def test(self) -> DateRange:
        return DateRange(self.test_start, self.test_end)


def walk_forward_splits(
    start: date,
    end: date,
    *,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
    embargo_days: int = 1,
    holdout: DateRange,
    mode: str = "rolling",
) -> list[WalkForwardFold]:
    """Build folds. Raises if ``start..end`` or any fold touches ``holdout``.

    ``mode="rolling"`` slides the training window by ``step_days``.
    ``mode="anchored"`` keeps ``train_start`` at ``start`` and lengthens the
    training window by ``step_days`` each fold.
    """
    if mode not in {"rolling", "anchored"}:
        raise ValueError("mode must be 'rolling' or 'anchored'.")
    if train_days < 1 or test_days < 1:
        raise ValueError("train_days and test_days must be positive.")
    if embargo_days < 0:
        raise ValueError("embargo_days cannot be negative.")
    step = test_days if step_days is None else step_days
    if step < 1:
        raise ValueError("step_days must be positive.")
    assert_no_holdout_overlap(DateRange(start, end), holdout, purpose="walk-forward split")
    folds: list[WalkForwardFold] = []
    for index in range(10_000):
        if mode == "rolling":
            train_start = start + timedelta(days=index * step)
        else:
            train_start = start
        train_length = train_days if mode == "rolling" else train_days + index * step
        train_end = train_start + timedelta(days=train_length - 1)
        test_start = train_end + timedelta(days=embargo_days + 1)
        test_end = test_start + timedelta(days=test_days - 1)
        if test_end > end:
            break
        fold = WalkForwardFold(index, train_start, train_end, test_start, test_end)
        assert_no_holdout_overlap(fold.train, holdout, purpose="walk-forward train fold")
        assert_no_holdout_overlap(fold.test, holdout, purpose="walk-forward test fold")
        folds.append(fold)
    return folds
