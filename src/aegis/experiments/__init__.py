"""Phase 5 experiment memory. Not implemented in this build.

MLflow will track strategy versions and metrics. Postgres will store the
experiment log the Orchestrator reads between runs. Neither store is allowed
to cache holdout labels for an optimizer.
"""
