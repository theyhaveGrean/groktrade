"""Small rank and error metrics. No scikit-learn dependency."""

from __future__ import annotations

import math

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, 1-based, stable for ties."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    cursor = 0
    while cursor < len(array):
        end = cursor
        while end + 1 < len(array) and array[order[end + 1]] == array[order[cursor]]:
            end += 1
        average = 0.5 * ((cursor + 1) + (end + 1))
        for index in range(cursor, end + 1):
            ranks[order[index]] = average
        cursor = end + 1
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    return pearson(rankdata(left), rankdata(right))


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if denom == 0:
        return None
    return float((x * y).sum() / denom)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) == 0:
        return None
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(math.sqrt(float((error * error).mean())))


def top_half_mean(scores: np.ndarray, outcomes: np.ndarray) -> float | None:
    if len(scores) == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    keep = max(1, len(order) // 2)
    chosen = np.asarray(outcomes, dtype=float)[order[:keep]]
    return float(chosen.mean())
