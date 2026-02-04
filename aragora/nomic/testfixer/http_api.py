"""HTTP API helpers for TestFixer integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aragora.nomic.testfixer.analyzer import FailureAnalyzer
from aragora.nomic.testfixer.orchestrator import FixLoopConfig, TestFixerOrchestrator
from aragora.nomic.testfixer.proposer import PatchProposal
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.store import TestFixerAttemptStore
from aragora.nomic.testfixer.generators import AgentCodeGenerator, AgentGeneratorConfig


@dataclass
class TestFixerRunConfig:
    """Config for TestFixer HTTP endpoints."""

    repo_path: Path
    test_command: str
    agents: list[str]
    max_iterations: int = 10
    min_confidence: float = 0.5
    timeout_seconds: float = 300.0
    attempt_store_path: Path | None = None
    artifacts_dir: Path | None = None
    enable_diagnostics: bool = True


async def analyze_failure(repo_path: Path, failure: TestFailure) -> dict[str, Any]:
    analyzer = FailureAnalyzer(repo_path=repo_path)
    analysis = await analyzer.analyze(failure)
    return {
        "analysis": analysis,
        "analysis_dict": {
            "failure": {
                "test_name": failure.test_name,
                "test_file": failure.test_file,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
                "stack_trace": failure.stack_trace,
                "line_number": failure.line_number,
                "relevant_code": failure.relevant_code,
            },
            "category": analysis.category.value,
            "confidence": analysis.confidence,
            "fix_target": analysis.fix_target.value,
            "root_cause": analysis.root_cause,
            "root_cause_file": analysis.root_cause_file,
            "suggested_approach": analysis.suggested_approach,
        },
    }


def _proposal_dir(repo_path: Path) -> Path:
    path = repo_path / ".testfixer" / "proposals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_proposal(repo_path: Path, proposal: PatchProposal) -> Path:
    proposal_path = _proposal_dir(repo_path) / f"{proposal.id}.json"
    data = {
        "id": proposal.id,
        "description": proposal.description,
        "confidence": proposal.post_debate_confidence,
        "patches": [
            {
                "file_path": patch.file_path,
                "patched_content": patch.patched_content,
            }
            for patch in proposal.patches
        ],
    }
    proposal_path.write_text(json.dumps(data, indent=2))
    return proposal_path


def apply_proposal(repo_path: Path, proposal_id: str) -> dict[str, Any]:
    proposal_path = _proposal_dir(repo_path) / f"{proposal_id}.json"
    if not proposal_path.exists():
        return {"status": "error", "message": "proposal not found"}
    data = json.loads(proposal_path.read_text())
    for patch in data.get("patches", []):
        file_path = repo_path / patch["file_path"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(patch["patched_content"])
    return {"status": "applied", "proposal_id": proposal_id}


def _make_generators(agents: list[str]) -> list[AgentCodeGenerator]:
    generators: list[AgentCodeGenerator] = []
    for agent_spec in agents:
        if ":" in agent_spec:
            agent_type, model = agent_spec.split(":", 1)
        else:
            agent_type, model = agent_spec, None
        config = AgentGeneratorConfig(agent_type=agent_type, model=model)
        generators.append(AgentCodeGenerator(config))
    return generators


async def propose_fix(
    repo_path: Path,
    analysis,
    agents: list[str],
) -> PatchProposal:
    from aragora.nomic.testfixer.proposer import PatchProposer

    proposer = PatchProposer(
        repo_path=repo_path,
        generators=_make_generators(agents),
    )
    return await proposer.propose_fix(analysis)


async def run_fix_loop(config: TestFixerRunConfig):
    attempt_store = None
    if config.attempt_store_path:
        attempt_store = TestFixerAttemptStore(config.attempt_store_path)

    loop_config = FixLoopConfig(
        max_iterations=config.max_iterations,
        min_confidence_to_apply=config.min_confidence,
        attempt_store=attempt_store,
        artifacts_dir=config.artifacts_dir,
        enable_diagnostics=config.enable_diagnostics,
    )

    fixer = TestFixerOrchestrator(
        repo_path=config.repo_path,
        test_command=config.test_command,
        config=loop_config,
        generators=_make_generators(config.agents),
        test_timeout=config.timeout_seconds,
    )
    return await fixer.run_fix_loop()
