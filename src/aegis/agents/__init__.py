"""Phase 3 research agents.

Orchestrator, Researcher, Algorithm Implementer, and Naysayer. Each has a
versioned system prompt, a private tool list, and a pydantic output. The
implementer may only write a signal plug-in and its tests. The harness, the
risk engine, the data package, and the holdout module stay read-only.

This package stays import-light so the sandbox child can load
``aegis.agents.signal_contract`` without pulling in the OpenAI client.
"""
