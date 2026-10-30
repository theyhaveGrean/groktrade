"""Phase 1 market-data pipeline.

Massive supplies delayed 1-minute bars. The cache stores raw prints. Research
reads go through :func:`aegis.data.query.load_research_bars`, which refuses the
protected holdout.
"""
