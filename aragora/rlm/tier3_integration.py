"""
Tier 3 deep RLM integration: Pipeline enrichment, Gauntlet analysis, Trajectory learning.

This module provides deep integration between the RLM system and three key
Aragora subsystems:

1. **Pipeline Enrichment**: Inject RLM helpers into PlanExecutor execution context
   so that decision plans can leverage memory and knowledge navigation.

2. **Gauntlet Adversarial Analysis**: Use RLM to recursively scan debate transcripts
   for vulnerabilities, logical fallacies, and weak arguments.

3. **Trajectory Learning**: Collect RLM trajectory data (JSONL) for the Nomic Loop
   self-improvement pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- Pipeline Enrichment ---


@dataclass
class PipelineRLMContext:
    """RLM context enrichment for PlanExecutor."""

    memory_helpers: dict[str, Any] = field(default_factory=dict)
    knowledge_helpers: dict[str, Any] = field(default_factory=dict)
    debate_summary: str = ""
    enriched: bool = False


def enrich_plan_context(
    rlm: Any,
    *,
    continuum: Any | None = None,
    mound: Any | None = None,
    workspace_id: str = "",
    debate_result: Any | None = None,
) -> PipelineRLMContext:
    """
    Enrich a plan execution context with RLM helpers.

    Injects memory, knowledge, and debate helpers into the context
    so that pipeline steps can leverage TRUE RLM navigation.

    Args:
        rlm: AragoraRLM instance
        continuum: Optional ContinuumMemory for memory helpers
        mound: Optional KnowledgeMound for knowledge helpers
        workspace_id: Workspace ID for knowledge scope
        debate_result: Optional debate result for summary

    Returns:
        PipelineRLMContext with injected helpers
    """
    ctx = PipelineRLMContext()

    # Inject memory helpers
    if continuum is not None and hasattr(rlm, "inject_memory_helpers"):
        try:
            result = rlm.inject_memory_helpers(continuum)
            ctx.memory_helpers = result.get("helpers", {})
        except Exception as e:
            logger.debug("Memory helper injection failed: %s", e)

    # Inject knowledge helpers
    if mound is not None and workspace_id and hasattr(rlm, "inject_knowledge_helpers"):
        try:
            result = rlm.inject_knowledge_helpers(mound, workspace_id)
            ctx.knowledge_helpers = result.get("helpers", {})
        except Exception as e:
            logger.debug("Knowledge helper injection failed: %s", e)

    # Generate debate summary
    if debate_result is not None:
        try:
            from .debate_helpers import load_debate_context

            debate_ctx = load_debate_context(debate_result)
            parts = [f"Debate: {debate_ctx.task}"]
            parts.append(f"Agents: {', '.join(debate_ctx.agent_names)}")
            parts.append(f"Rounds: {debate_ctx.total_rounds}")
            if debate_ctx.consensus_reached:
                parts.append(f"Consensus: {debate_ctx.final_answer or 'reached'}")
            ctx.debate_summary = " | ".join(parts)
        except Exception as e:
            logger.debug("Debate summary generation failed: %s", e)

    ctx.enriched = bool(ctx.memory_helpers or ctx.knowledge_helpers or ctx.debate_summary)
    return ctx


# --- Gauntlet Adversarial Analysis ---


@dataclass
class GauntletRLMFinding:
    """A single finding from gauntlet adversarial analysis."""

    category: str  # "logical_fallacy", "weak_argument", "missing_evidence", etc.
    severity: str  # "low", "medium", "high", "critical"
    description: str
    source_round: int | None = None
    source_agent: str | None = None
    evidence: str = ""


@dataclass
class GauntletRLMAnalysis:
    """Complete gauntlet analysis result."""

    findings: list[GauntletRLMFinding]
    total_findings: int
    severity_counts: dict[str, int] = field(default_factory=dict)
    summary: str = ""
    raw_rlm_answer: str = ""


def analyze_debate_for_gauntlet(
    rlm: Any,
    debate_result: Any,
    *,
    focus_areas: list[str] | None = None,
) -> GauntletRLMAnalysis:
    """
    Use RLM to recursively analyze a debate transcript for vulnerabilities.

    This performs adversarial analysis by scanning for:
    - Logical fallacies (ad hominem, straw man, appeal to authority, etc.)
    - Weak arguments (low evidence, circular reasoning)
    - Missing evidence (claims without support)
    - Consensus manipulation (false consensus, groupthink indicators)

    Args:
        rlm: AragoraRLM instance
        debate_result: DebateResult to analyze
        focus_areas: Optional list of areas to focus on

    Returns:
        GauntletRLMAnalysis with categorized findings
    """
    # Build analysis from debate context
    try:
        from .debate_helpers import load_debate_context

        debate_ctx = load_debate_context(debate_result)
    except Exception as e:
        logger.debug("Failed to load debate context for gauntlet: %s", e)
        return GauntletRLMAnalysis(
            findings=[],
            total_findings=0,
            summary=f"Analysis failed: {e}",
        )

    # Build the adversarial analysis prompt
    focus = focus_areas or [
        "logical_fallacy",
        "weak_argument",
        "missing_evidence",
        "consensus_manipulation",
    ]

    # Analyze each round for findings
    findings: list[GauntletRLMFinding] = []

    for round_num, messages in debate_ctx.rounds.items():
        for msg in messages:
            content = msg.get("content", "")
            agent = msg.get("agent", msg.get("agent_name", "unknown"))

            # Heuristic-based analysis (without actual LLM call)
            round_findings = _analyze_message_heuristic(content, agent, round_num, focus)
            findings.extend(round_findings)

    # Calculate severity counts
    severity_counts: dict[str, int] = {}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

    # Build summary
    summary_parts = [
        f"Analyzed {debate_ctx.total_rounds} rounds, {len(debate_ctx.all_messages)} messages."
    ]
    if findings:
        summary_parts.append(f"Found {len(findings)} potential issue(s).")
        for sev, count in sorted(severity_counts.items()):
            summary_parts.append(f"  {sev}: {count}")
    else:
        summary_parts.append("No issues found.")

    return GauntletRLMAnalysis(
        findings=findings,
        total_findings=len(findings),
        severity_counts=severity_counts,
        summary=" ".join(summary_parts),
    )


def _analyze_message_heuristic(
    content: str,
    agent: str,
    round_num: int,
    focus_areas: list[str],
) -> list[GauntletRLMFinding]:
    """Heuristic analysis of a single message for gauntlet findings."""
    import re

    findings: list[GauntletRLMFinding] = []
    content_lower = content.lower()

    if "logical_fallacy" in focus_areas:
        # Check for ad hominem
        ad_hominem_patterns = [r"you always", r"you never", r"people like you"]
        for pattern in ad_hominem_patterns:
            if re.search(pattern, content_lower):
                findings.append(
                    GauntletRLMFinding(
                        category="logical_fallacy",
                        severity="medium",
                        description=f"Potential ad hominem: '{pattern}' pattern detected",
                        source_round=round_num,
                        source_agent=agent,
                        evidence=content[:200],
                    )
                )
                break

    if "weak_argument" in focus_areas:
        # Check for unsupported assertions
        assertion_patterns = [r"obviously", r"everyone knows", r"it's clear that"]
        for pattern in assertion_patterns:
            if re.search(pattern, content_lower):
                findings.append(
                    GauntletRLMFinding(
                        category="weak_argument",
                        severity="low",
                        description=f"Unsupported assertion: '{pattern}' without evidence",
                        source_round=round_num,
                        source_agent=agent,
                        evidence=content[:200],
                    )
                )
                break

    if "missing_evidence" in focus_areas:
        # Check for claims without citations
        claim_patterns = [r"studies show", r"research proves", r"data indicates"]
        citation_patterns = [r"\d{4}", r"http", r"doi:", r"\[.+\]"]
        for pattern in claim_patterns:
            if re.search(pattern, content_lower):
                has_citation = any(re.search(cp, content) for cp in citation_patterns)
                if not has_citation:
                    findings.append(
                        GauntletRLMFinding(
                            category="missing_evidence",
                            severity="medium",
                            description=f"Claim '{pattern}' without supporting citation",
                            source_round=round_num,
                            source_agent=agent,
                            evidence=content[:200],
                        )
                    )
                    break

    return findings


def _parse_gauntlet_findings(raw_text: str) -> list[GauntletRLMFinding]:
    """Parse raw RLM analysis text into structured findings."""
    findings: list[GauntletRLMFinding] = []

    # Try JSON parsing first
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            for item in data:
                findings.append(
                    GauntletRLMFinding(
                        category=item.get("category", "unknown"),
                        severity=item.get("severity", "low"),
                        description=item.get("description", ""),
                        source_round=item.get("source_round"),
                        source_agent=item.get("source_agent"),
                        evidence=item.get("evidence", ""),
                    )
                )
            return findings
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: parse line-by-line
    import re

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Try to extract category and description
        match = re.match(r"\[(\w+)\]\s*\((\w+)\)\s*(.*)", line)
        if match:
            findings.append(
                GauntletRLMFinding(
                    category=match.group(1),
                    severity=match.group(2),
                    description=match.group(3),
                )
            )

    return findings


# --- Trajectory Learning ---


@dataclass
class TrajectoryLearningData:
    """Trajectory data collected for Nomic Loop self-improvement."""

    query: str
    answer: str
    trajectory_log_path: str | None = None
    rlm_iterations: int = 0
    code_blocks_executed: int = 0
    confidence: float = 0.0
    used_true_rlm: bool = False
    debate_id: str | None = None
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def collect_trajectory_for_learning(
    result: Any,
    *,
    query: str = "",
    debate_id: str | None = None,
) -> TrajectoryLearningData:
    """
    Collect trajectory data from an RLM result for learning.

    Args:
        result: RLMResult from a completed query
        query: The original query
        debate_id: Optional debate ID

    Returns:
        TrajectoryLearningData for storage and analysis
    """
    import datetime

    return TrajectoryLearningData(
        query=query,
        answer=getattr(result, "answer", ""),
        trajectory_log_path=getattr(result, "trajectory_log_path", None),
        rlm_iterations=getattr(result, "rlm_iterations", 0),
        code_blocks_executed=getattr(result, "code_blocks_executed", 0),
        confidence=getattr(result, "confidence", 0.0),
        used_true_rlm=getattr(result, "used_true_rlm", False),
        debate_id=debate_id,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metadata={
            "tokens_processed": getattr(result, "tokens_processed", 0),
            "sub_calls_made": getattr(result, "sub_calls_made", 0),
            "time_seconds": getattr(result, "time_seconds", 0.0),
        },
    )


def store_trajectory_learning(
    data: TrajectoryLearningData,
    output_dir: str | None = None,
) -> str:
    """
    Store trajectory learning data as JSONL for the Nomic Loop.

    Args:
        data: TrajectoryLearningData to store
        output_dir: Directory to write to (defaults to ~/.aragora/trajectories/)

    Returns:
        Path to the JSONL file
    """
    if output_dir is None:
        output_dir = os.path.expanduser("~/.aragora/trajectories")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(output_dir, "rlm_trajectories.jsonl")

    record = {
        "query": data.query,
        "answer": data.answer[:1000],  # Truncate for storage
        "trajectory_log_path": data.trajectory_log_path,
        "rlm_iterations": data.rlm_iterations,
        "code_blocks_executed": data.code_blocks_executed,
        "confidence": data.confidence,
        "used_true_rlm": data.used_true_rlm,
        "debate_id": data.debate_id,
        "timestamp": data.timestamp,
        "metadata": data.metadata,
    }

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return file_path


def load_trajectory_insights(
    trajectory_dir: str | None = None,
    limit: int = 100,
) -> list[TrajectoryLearningData]:
    """
    Load trajectory insights from stored JSONL data.

    Args:
        trajectory_dir: Directory to read from
        limit: Maximum entries to load

    Returns:
        List of TrajectoryLearningData entries
    """
    if trajectory_dir is None:
        trajectory_dir = os.path.expanduser("~/.aragora/trajectories")

    file_path = os.path.join(trajectory_dir, "rlm_trajectories.jsonl")

    if not os.path.exists(file_path):
        return []

    entries: list[TrajectoryLearningData] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    entries.append(
                        TrajectoryLearningData(
                            query=record.get("query", ""),
                            answer=record.get("answer", ""),
                            trajectory_log_path=record.get("trajectory_log_path"),
                            rlm_iterations=record.get("rlm_iterations", 0),
                            code_blocks_executed=record.get("code_blocks_executed", 0),
                            confidence=record.get("confidence", 0.0),
                            used_true_rlm=record.get("used_true_rlm", False),
                            debate_id=record.get("debate_id"),
                            timestamp=record.get("timestamp", ""),
                            metadata=record.get("metadata", {}),
                        )
                    )
                    if len(entries) >= limit:
                        break
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        logger.debug("Failed to load trajectory data: %s", e)

    return entries


__all__ = [
    # Pipeline enrichment
    "PipelineRLMContext",
    "enrich_plan_context",
    # Gauntlet analysis
    "GauntletRLMFinding",
    "GauntletRLMAnalysis",
    "analyze_debate_for_gauntlet",
    # Trajectory learning
    "TrajectoryLearningData",
    "collect_trajectory_for_learning",
    "store_trajectory_learning",
    "load_trajectory_insights",
]
