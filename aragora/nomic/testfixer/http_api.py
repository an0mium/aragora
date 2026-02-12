"""HTTP API helpers for TestFixer integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aragora.nomic.testfixer.analyzer import FailureAnalyzer
from aragora.nomic.testfixer.analyzers import LLMAnalyzerConfig
from aragora.nomic.testfixer.orchestrator import FixLoopConfig, TestFixerOrchestrator
from aragora.nomic.testfixer.proposer import PatchProposal
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.store import TestFixerAttemptStore
from aragora.nomic.testfixer.generators import AgentCodeGenerator, AgentGeneratorConfig
from aragora.nomic.testfixer.proposer import CodeGenerator
from aragora.nomic.testfixer.validators import ArenaValidatorConfig, RedTeamValidatorConfig


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
    use_llm_analyzer: bool = False
    analysis_agents: list[str] | None = None
    analysis_require_consensus: bool = False
    analysis_consensus_threshold: float = 0.7
    arena_validate: bool = False
    arena_agents: list[str] | None = None
    arena_rounds: int = 2
    arena_min_confidence: float = 0.6
    arena_require_consensus: bool = False
    arena_consensus_threshold: float = 0.7
    redteam_validate: bool = False
    redteam_attackers: list[str] | None = None
    redteam_defender: str | None = None
    redteam_rounds: int = 2
    redteam_attacks_per_round: int = 3
    redteam_min_robustness: float = 0.6
    pattern_learning: bool = False
    pattern_store_path: Path | None = None


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


def _make_generators(agents: list[str]) -> list[CodeGenerator]:
    generators: list[CodeGenerator] = []
    for agent_spec in agents:
        if ":" in agent_spec:
            agent_type, model = agent_spec.split(":", 1)
        else:
            agent_type, model = agent_spec, None
        config = AgentGeneratorConfig(agent_type=agent_type, model=model)
        generators.append(AgentCodeGenerator(config))
    return generators


def _parse_agent_specs(
    specs: list[str] | None,
    fallback: list[str],
) -> tuple[list[str], dict[str, str] | None]:
    agent_specs = specs if specs is not None else fallback
    if not agent_specs:
        return ([], None)
    agent_types: list[str] = []
    models: dict[str, str] = {}
    for spec in agent_specs:
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            agent_type, model = spec.split(":", 1)
            models[agent_type] = model
        else:
            agent_type = spec
        agent_types.append(agent_type)
    return agent_types, models or None


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

    analysis_agent_types, analysis_models = _parse_agent_specs(
        config.analysis_agents, config.agents
    )
    llm_analyzer_config = None
    if config.use_llm_analyzer:
        default_analysis = LLMAnalyzerConfig()
        llm_analyzer_config = LLMAnalyzerConfig(
            agent_types=analysis_agent_types or default_analysis.agent_types,
            models=analysis_models,
            require_consensus=config.analysis_require_consensus,
            consensus_threshold=config.analysis_consensus_threshold,
        )

    arena_config = None
    if config.arena_validate:
        arena_agent_types, arena_models = _parse_agent_specs(config.arena_agents, config.agents)
        default_arena = ArenaValidatorConfig()
        arena_config = ArenaValidatorConfig(
            agent_types=arena_agent_types or default_arena.agent_types,
            models=arena_models,
            debate_rounds=config.arena_rounds,
            min_confidence_to_pass=config.arena_min_confidence,
            require_consensus=config.arena_require_consensus,
            consensus_threshold=config.arena_consensus_threshold,
        )

    redteam_config = None
    if config.redteam_validate:
        redteam_attackers, _ = _parse_agent_specs(config.redteam_attackers, config.agents)
        default_redteam = RedTeamValidatorConfig()
        defender = config.redteam_defender or (
            redteam_attackers[0] if redteam_attackers else default_redteam.defender_type
        )
        redteam_config = RedTeamValidatorConfig(
            attacker_types=redteam_attackers or default_redteam.attacker_types,
            defender_type=defender,
            attack_rounds=config.redteam_rounds,
            attacks_per_round=config.redteam_attacks_per_round,
            min_robustness_score=config.redteam_min_robustness,
        )

    loop_config = FixLoopConfig(
        max_iterations=config.max_iterations,
        min_confidence_to_apply=config.min_confidence,
        attempt_store=attempt_store,
        artifacts_dir=config.artifacts_dir,
        enable_diagnostics=config.enable_diagnostics,
        use_llm_analyzer=config.use_llm_analyzer,
        llm_analyzer_config=llm_analyzer_config,
        enable_arena_validation=config.arena_validate,
        arena_validator_config=arena_config,
        enable_redteam_validation=config.redteam_validate,
        redteam_validator_config=redteam_config,
        enable_pattern_learning=config.pattern_learning,
        pattern_store_path=config.pattern_store_path,
    )

    fixer = TestFixerOrchestrator(
        repo_path=config.repo_path,
        test_command=config.test_command,
        config=loop_config,
        generators=_make_generators(config.agents),
        test_timeout=config.timeout_seconds,
    )
    return await fixer.run_fix_loop()
