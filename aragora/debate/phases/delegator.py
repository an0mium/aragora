"""
Debate Delegator - Architectural Delegation Pattern for RLM.

Based on the RLM paper's principle that the main LLM should not process
everything directly but delegate to fresh sub-LLM contexts (arXiv:2512.24601).

Key principles:
1. Main orchestrator coordinates, doesn't process everything
2. Sub-agents get fresh, compressed context windows
3. Parallel execution maximizes throughput
4. Results are synthesized by the orchestrator

Usage:
    from aragora.debate.phases.delegator import DebateDelegator, DelegationConfig

    delegator = DebateDelegator(agent_pool=agents, config=config)

    # Delegate analysis to multiple agents
    analyses = await delegator.delegate_analysis(
        task="Analyze these proposals",
        context=debate_context,
        num_analysts=3,
    )

    # Synthesize results
    synthesis = await delegator.synthesize_results(analyses)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    Generic,
    Optional,
    TypeVar,
)

from aragora.rlm.batch import llm_batch, BatchConfig

if TYPE_CHECKING:
    from aragora.core import Agent

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class DelegationConfig:
    """Configuration for delegation operations."""

    max_concurrent: int = 3
    """Maximum concurrent sub-agent calls."""

    timeout_per_agent: float = 45.0
    """Timeout per agent in seconds."""

    context_token_limit: int = 2000
    """Maximum tokens in compressed context for sub-agents."""

    require_consensus: bool = False
    """Whether to require agreement among sub-agents."""

    consensus_threshold: float = 0.6
    """Fraction of agents that must agree for consensus."""

    retry_on_failure: bool = True
    """Retry failed delegations."""

    max_retries: int = 1
    """Maximum retry attempts."""


@dataclass
class DelegationResult(Generic[R]):
    """Result of a delegation operation."""

    results: list[R]
    """Results from each sub-agent."""

    agent_names: list[str]
    """Names of agents that contributed."""

    consensus_reached: bool = False
    """Whether agents reached consensus."""

    consensus_value: Optional[R] = None
    """Consensus result if reached."""

    duration_seconds: float = 0.0
    """Total delegation time."""


@dataclass
class AnalysisResult:
    """Result of an analysis delegation."""

    agent: str
    """Agent that performed the analysis."""

    summary: str
    """Summary of the analysis."""

    key_points: list[str] = field(default_factory=list)
    """Key points identified."""

    confidence: float = 0.5
    """Confidence in the analysis."""

    raw_response: str = ""
    """Full response from the agent."""


@dataclass
class SynthesisResult:
    """Result of synthesizing multiple analyses."""

    summary: str
    """Synthesized summary."""

    agreements: list[str] = field(default_factory=list)
    """Points of agreement."""

    disagreements: list[str] = field(default_factory=list)
    """Points of disagreement."""

    recommendation: str = ""
    """Final recommendation."""

    confidence: float = 0.5
    """Confidence in the synthesis."""


class DebateDelegator:
    """
    Architectural delegation: main LLM orchestrates, sub-LLMs execute.

    Based on the RLM principle that the main LLM should not process
    everything directly but delegate to fresh sub-LLM contexts. Each
    sub-agent gets a compressed, focused context window.

    Key benefits:
    - Fresh context for each sub-agent (no context pollution)
    - Parallel execution maximizes throughput
    - Orchestrator synthesizes, doesn't process raw data
    - Reduced token usage via compression

    Usage:
        delegator = DebateDelegator(agent_pool=agents)

        # Analyze proposals with multiple agents
        analyses = await delegator.delegate_analysis(
            task="Evaluate the proposals",
            context="<compressed context>",
        )

        # Synthesize the analyses
        synthesis = await delegator.synthesize_results(analyses)
    """

    def __init__(
        self,
        agent_pool: list["Agent"],
        config: Optional[DelegationConfig] = None,
        generate_fn: Optional[Callable[["Agent", str], Awaitable[str]]] = None,
    ):
        """Initialize the delegator.

        Args:
            agent_pool: Pool of available agents for delegation
            config: Delegation configuration
            generate_fn: Optional async function to generate responses
        """
        self._agents = agent_pool
        self._config = config or DelegationConfig()
        self._generate_fn = generate_fn

    async def delegate_analysis(
        self,
        task: str,
        context: str,
        num_analysts: int = 3,
        analysis_prompt: Optional[str] = None,
    ) -> DelegationResult[AnalysisResult]:
        """
        Delegate analysis to multiple sub-agents in parallel.

        Each agent gets a fresh context window to analyze the task
        independently, avoiding context pollution.

        Args:
            task: The analysis task description
            context: Context to analyze (should be pre-compressed)
            num_analysts: Number of agents to use
            analysis_prompt: Custom prompt template (optional)

        Returns:
            DelegationResult containing individual analyses
        """
        import time

        start_time = time.time()

        # Select analysts from pool
        analysts = self._agents[:num_analysts]
        if not analysts:
            logger.warning("No agents available for analysis delegation")
            return DelegationResult(results=[], agent_names=[])

        # Build analysis prompt
        prompt = analysis_prompt or self._build_analysis_prompt(task, context)

        # Delegate in parallel using llm_batch
        async def analyze_with_agent(agent: "Agent") -> AnalysisResult:
            """Execute analysis with a single agent."""
            logger.debug(f"delegation_analysis_start agent={agent.name}")

            if self._generate_fn:
                response = await self._generate_fn(agent, prompt)
            else:
                # Fallback: use agent's default generation
                response = await agent.generate(prompt)

            # Parse response into AnalysisResult
            return self._parse_analysis_response(agent.name, str(response))

        batch_config = BatchConfig(
            max_concurrent=self._config.max_concurrent,
            timeout_per_item=self._config.timeout_per_agent,
            retry_on_error=self._config.retry_on_failure,
            max_retries=self._config.max_retries,
        )

        results = await llm_batch(
            items=analysts,
            process_fn=analyze_with_agent,
            config=batch_config,
        )

        duration = time.time() - start_time

        # Check for consensus if required
        consensus_reached = False
        consensus_value = None
        if self._config.require_consensus and len(results) > 1:
            consensus_reached, consensus_value = self._check_analysis_consensus(results)

        logger.info(
            f"delegation_analysis_complete agents={len(results)} "
            f"consensus={consensus_reached} duration={duration:.2f}s"
        )

        return DelegationResult(
            results=results,
            agent_names=[r.agent for r in results],
            consensus_reached=consensus_reached,
            consensus_value=consensus_value,
            duration_seconds=duration,
        )

    async def delegate_task(
        self,
        task: str,
        context: str,
        agents: Optional[list["Agent"]] = None,
        parse_fn: Optional[Callable[[str, str], R]] = None,
    ) -> DelegationResult[R]:
        """
        Delegate a generic task to multiple agents.

        More flexible than delegate_analysis - allows custom parsing.

        Args:
            task: Task description
            context: Context for the task
            agents: Specific agents to use (default: use pool)
            parse_fn: Function to parse response (agent_name, response) -> R

        Returns:
            DelegationResult with parsed results
        """
        import time

        start_time = time.time()

        target_agents = agents or self._agents[: self._config.max_concurrent]
        if not target_agents:
            return DelegationResult(results=[], agent_names=[])

        prompt = f"""Task: {task}

Context:
{context[: self._config.context_token_limit * 4]}

Please provide your response to the task."""

        async def execute_task(agent: "Agent") -> R:
            if self._generate_fn:
                response = await self._generate_fn(agent, prompt)
            else:
                response = await agent.generate(prompt)

            if parse_fn:
                return parse_fn(agent.name, str(response))
            return str(response)  # type: ignore

        batch_config = BatchConfig(
            max_concurrent=self._config.max_concurrent,
            timeout_per_item=self._config.timeout_per_agent,
        )

        results = await llm_batch(
            items=target_agents,
            process_fn=execute_task,
            config=batch_config,
        )

        return DelegationResult(
            results=results,
            agent_names=[a.name for a in target_agents[: len(results)]],
            duration_seconds=time.time() - start_time,
        )

    async def synthesize_analyses(
        self,
        analyses: list[AnalysisResult],
        synthesizer: Optional["Agent"] = None,
    ) -> SynthesisResult:
        """
        Synthesize multiple analyses into a coherent result.

        The synthesizer (orchestrator) receives only summaries,
        not the full raw data - following RLM principles.

        Args:
            analyses: List of individual analyses to synthesize
            synthesizer: Agent to perform synthesis (default: first in pool)

        Returns:
            SynthesisResult with combined insights
        """
        if not analyses:
            return SynthesisResult(summary="No analyses to synthesize")

        # Select synthesizer
        agent = synthesizer or (self._agents[0] if self._agents else None)
        if not agent:
            return SynthesisResult(summary="No synthesizer available")

        # Build synthesis prompt from summaries only (RLM principle)
        summaries = "\n\n".join(
            f"**{a.agent}** (confidence: {a.confidence:.1%}):\n{a.summary}" for a in analyses
        )

        key_points_combined = []
        for a in analyses:
            key_points_combined.extend(a.key_points)

        prompt = f"""You are synthesizing multiple analyses from different agents.

**Individual Analyses:**
{summaries}

**Key Points Identified:**
{chr(10).join(f"- {p}" for p in key_points_combined[:20])}

Please synthesize these into:
1. A unified summary capturing the main insights
2. Points of agreement among agents
3. Points of disagreement or contention
4. A final recommendation based on the combined analysis

Respond in this format:
SUMMARY: <unified summary>
AGREEMENTS: <comma-separated list>
DISAGREEMENTS: <comma-separated list>
RECOMMENDATION: <final recommendation>
CONFIDENCE: <0.0-1.0>"""

        try:
            if self._generate_fn:
                response = await self._generate_fn(agent, prompt)
            else:
                response = await agent.generate(prompt)

            return self._parse_synthesis_response(str(response))

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return SynthesisResult(
                summary=f"Synthesis failed: {e}",
                confidence=0.0,
            )

    def _build_analysis_prompt(self, task: str, context: str) -> str:
        """Build the analysis prompt for sub-agents."""
        # Truncate context to fit token limit
        max_context_chars = self._config.context_token_limit * 4  # Rough estimate
        truncated_context = context[:max_context_chars]

        return f"""You are analyzing the following for a multi-agent debate.

**Task:** {task}

**Context:**
{truncated_context}

Please provide your analysis including:
1. A brief summary of your assessment
2. Key points or insights (as bullet points)
3. Your confidence level (0.0 to 1.0)

Respond in this format:
SUMMARY: <your summary>
KEY_POINTS:
- Point 1
- Point 2
- Point 3
CONFIDENCE: <0.0-1.0>"""

    def _parse_analysis_response(self, agent_name: str, response: str) -> AnalysisResult:
        """Parse an agent's analysis response."""
        summary = ""
        key_points: list[str] = []
        confidence = 0.5

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.upper().startswith("SUMMARY:"):
                summary = line_stripped.split(":", 1)[-1].strip()
                current_section = "summary"
            elif line_stripped.upper().startswith("KEY_POINTS:"):
                current_section = "key_points"
            elif line_stripped.upper().startswith("CONFIDENCE:"):
                try:
                    conf_str = line_stripped.split(":", 1)[-1].strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
                current_section = None
            elif current_section == "key_points" and line_stripped.startswith("-"):
                key_points.append(line_stripped[1:].strip())
            elif current_section == "summary" and line_stripped:
                summary += " " + line_stripped

        # Fallback if parsing failed
        if not summary:
            summary = response[:500]

        return AnalysisResult(
            agent=agent_name,
            summary=summary.strip(),
            key_points=key_points,
            confidence=confidence,
            raw_response=response,
        )

    def _parse_synthesis_response(self, response: str) -> SynthesisResult:
        """Parse a synthesis response."""
        summary = ""
        agreements: list[str] = []
        disagreements: list[str] = []
        recommendation = ""
        confidence = 0.5

        lines = response.split("\n")

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.upper().startswith("SUMMARY:"):
                summary = line_stripped.split(":", 1)[-1].strip()
            elif line_stripped.upper().startswith("AGREEMENTS:"):
                parts = line_stripped.split(":", 1)[-1].strip()
                agreements = [p.strip() for p in parts.split(",") if p.strip()]
            elif line_stripped.upper().startswith("DISAGREEMENTS:"):
                parts = line_stripped.split(":", 1)[-1].strip()
                disagreements = [p.strip() for p in parts.split(",") if p.strip()]
            elif line_stripped.upper().startswith("RECOMMENDATION:"):
                recommendation = line_stripped.split(":", 1)[-1].strip()
            elif line_stripped.upper().startswith("CONFIDENCE:"):
                try:
                    conf_str = line_stripped.split(":", 1)[-1].strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

        return SynthesisResult(
            summary=summary or response[:500],
            agreements=agreements,
            disagreements=disagreements,
            recommendation=recommendation,
            confidence=confidence,
        )

    def _check_analysis_consensus(
        self, analyses: list[AnalysisResult]
    ) -> tuple[bool, Optional[AnalysisResult]]:
        """Check if analyses have reached consensus."""
        if len(analyses) < 2:
            return False, None

        # Simple consensus: majority of high-confidence analyses agree
        high_confidence = [a for a in analyses if a.confidence >= 0.7]
        if len(high_confidence) < len(analyses) * self._config.consensus_threshold:
            return False, None

        # For now, just check if they exist - semantic similarity
        # would require additional infrastructure
        return True, high_confidence[0] if high_confidence else None


__all__ = [
    "DebateDelegator",
    "DelegationConfig",
    "DelegationResult",
    "AnalysisResult",
    "SynthesisResult",
]
