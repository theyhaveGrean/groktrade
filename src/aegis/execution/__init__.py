"""Phase 6 Alpaca paper trading. Not implemented in this build.

Orders will go to ``ALPACA_BASE_URL`` (paper by default). ``TRADING_MODE``
must stay ``paper`` and ``LIVE_TRADING_ENABLED`` must stay false until a
separate human approval exists. The risk engine still has to approve every
order; the broker adapter does not get a private sizing path.
"""
