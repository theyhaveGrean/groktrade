"""Phase 3+ agent roles. Not implemented in this build.

Planned roles: Orchestrator, Researcher, Algorithm Implementer, Naysayer,
Backtesting Analyst, ML Specialist, and Risk & Portfolio Analyst. Each gets
its own system prompt and tools. None of them can change
``aegis.backtest.risk.RiskLimits`` or clear a failed validation gate.
Secrets stay in the environment and never enter a prompt.
"""
