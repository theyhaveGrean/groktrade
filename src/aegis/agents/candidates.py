"""Store a signal candidate by the hash of its source, with the proposal and critique."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from aegis.agents.models import Critique, ImplementationResult, SandboxResult, StrategyProposal


def content_hash(source: str) -> str:
    """Identity of a candidate. Newlines are normalized so the hash is stable."""
    normalized = source.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def store_candidate(
    *,
    store_dir: Path,
    run_id: str,
    strategy_name: str,
    signal_source: str,
    test_source: str,
    proposal: StrategyProposal,
    critique: Critique,
    sandbox: SandboxResult,
    policy_implementation: str,
    prompt_version: str,
    notes: str,
) -> ImplementationResult:
    digest = content_hash(signal_source)
    dest = store_dir / run_id / digest
    dest.mkdir(parents=True, exist_ok=True)
    signal_path = dest / "candidate.py"
    test_path = dest / "test_candidate.py"
    proposal_path = dest / "proposal.json"
    critique_path = dest / "critique.json"
    manifest_path = dest / "manifest.json"
    signal_path.write_text(signal_source if signal_source.endswith("\n") else signal_source + "\n", encoding="utf-8")
    test_path.write_text(test_source if test_source.endswith("\n") else test_source + "\n", encoding="utf-8")
    proposal_path.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
    critique_path.write_text(critique.model_dump_json(indent=2), encoding="utf-8")
    manifest = {
        "content_hash": digest,
        "tests_hash": content_hash(test_source),
        "strategy_name": strategy_name,
        "signal_class": sandbox.signal_class,
        "status": sandbox.status,
        "policy_ok": sandbox.policy_ok,
        "policy_implementation": policy_implementation,
        "prompt_version": prompt_version,
        "tests_run": sandbox.tests_run,
        "tests_passed": sandbox.tests_passed,
        "risk_applied": sandbox.risk_applied,
        "risk_quantity": sandbox.risk_quantity,
        "risk_max_risk_per_trade": sandbox.risk_max_risk_per_trade,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "signal": signal_path.name,
            "tests": test_path.name,
            "proposal": proposal_path.name,
            "critique": critique_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log = sandbox.log
    if sandbox.violations:
        log = (log + "\n" if log else "") + "\n".join(sandbox.violations)
    if sandbox.failures:
        log = (log + "\n" if log else "") + "\n".join(sandbox.failures)
    return ImplementationResult(
        strategy_name=strategy_name,
        code_path=str(signal_path),
        test_path=str(test_path),
        status=sandbox.status,
        content_hash=digest,
        tests_run=sandbox.tests_run,
        tests_passed=sandbox.tests_passed,
        sandbox_log=log.strip(),
        notes=notes,
        candidate_dir=str(dest),
    )
