"""EMA execution now lives on the harness.

:class:`aegis.strategies.ema.EmaCrossover` is the signal. Orders, sizing,
the entry window, the 15:55 flatten, the daily loss cutoff, and the kill
switch are :class:`aegis.strategies.harness.HarnessStrategy`.
"""
