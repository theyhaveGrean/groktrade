"""Deflated Sharpe ratio for a walk-forward trial ledger.

Bailey and Lopez de Prado's deflated Sharpe is the probability that the
observed Sharpe is above zero after accounting for selection among N trials
and for skew and kurtosis. The Sharpe passed in is the per-period ratio
(not annualized). N=1 skips the multiple-testing haircut and returns the
probabilistic Sharpe against a zero benchmark. The result is a probability
in ``[0, 1]``, or ``None`` when there are fewer than three returns.
"""

from __future__ import annotations

import math

_EULER = 0.5772156649015329


def deflated_sharpe_ratio(returns: list[float], n_trials: int) -> float | None:
    """Probability that the true per-period Sharpe exceeds zero given ``n_trials``."""
    if n_trials < 1 or len(returns) < 3:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    if variance <= 0:
        return None
    sharpe = mean / math.sqrt(variance)
    skew, kurtosis = _moments(returns)
    scale = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    if scale <= 0:
        return None
    sharpe_std = math.sqrt(scale / (len(returns) - 1))
    if sharpe_std <= 0:
        return None
    benchmark = 0.0 if n_trials == 1 else _expected_max_sharpe(sharpe_std, n_trials)
    return _norm_cdf((sharpe - benchmark) / sharpe_std)


def _expected_max_sharpe(sharpe_std: float, n_trials: int) -> float:
    """Expected maximum Sharpe under a null of zero, for N independent trials."""
    z_one = _norm_ppf(1.0 - 1.0 / n_trials)
    z_two = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sharpe_std * ((1.0 - _EULER) * z_one + _EULER * z_two)


def _moments(returns: list[float]) -> tuple[float, float]:
    """Sample skewness and non-excess (raw) kurtosis."""
    count = len(returns)
    mean = sum(returns) / count
    second = sum((item - mean) ** 2 for item in returns) / count
    if second <= 0:
        return 0.0, 3.0
    third = sum((item - mean) ** 3 for item in returns) / count
    fourth = sum((item - mean) ** 4 for item in returns) / count
    skew = third / (second ** 1.5)
    kurtosis = fourth / (second ** 2)
    return skew, kurtosis


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )
