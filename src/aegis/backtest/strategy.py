"""EMA execution now lives on the harness.

:class:`aegis.strategies.ema.EmaCrossover` is the signal. Orders, sizing,
the entry window, the 15:55 flatten, the daily loss cutoff, and the kill
switch are :class:`aegis.strategies.harness.HarnessStrategy`.

Paper trading still shares :func:`aegis.strategies.ema.detect_ema_cross`
with that signal so a live bar uses the same cross as the backtest.
"""
