"""
LLM-powered failure analyzer for TestFixer.

Uses multiple LLM agents to semantically analyze test failures,
cross-validate diagnoses, and synthesize the best root cause analysis.
This goes beyond pattern matching to understand complex failures
that heuristics cannot handle.
"""

from __future__ import annotations

import asyncio
import time
import logging
import re
from dataclasses import dataclass, field

from aragora.agents.base import create_agent
from aragora.core import Agent
from aragora.nomic.testfixer.analyzer import AIAnalyzer, FailureCategory
from aragora.nomic.testfixer.runner import TestFailure

logger = logging.getLogger(__name__)

# Regex patterns for parsing structured LLM output
_ROOT_CAUSE_RE = re.compile(r"<root_cause>(.*?)</root_cause>", re.DOTALL | re.IGNORECASE)
_APPROACH_RE = re.compile(r"<approach>(.*?)</approach>", re.DOTALL | re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"<confidence>(.*?)</confidence>", re.DOTALL | re.IGNORECASE)
_CATEGORY_RE = re.compile(r"<category>(.*?)</category>", re.DOTALL | re.IGNORECASE)
_AGREEMENT_RE = re.compile(r"<agreement>(.*?)</agreement>", re.DOTALL | re.IGNORECASE)
_CRITIQUE_RE = re.compile(r"<critique>(.*?)</critique>", re.DOTALL | re.IGNORECASE)


@dataclass
class LLMAnalyzerConfig:
    """Configuration for LLM-based failure analysis."""

    # Agent types to use for analysis (will use create_agent)
    agent_types: list[str] = field(default_factory=lambda: ["anthropic-api", "openai-api"])

    # Models to use (optional - uses default if None)
    models: dict[str, str] | None = None

    # Analysis settings
    max_context_chars: int = 60_000
    synthesis_threshold: float = 0.6  # Min confidence to use synthesized result
    cross_validate: bool = True  # Have agents validate each other

    # Timeout for each agent (seconds)
    agent_timeout: float = 60.0

    # If True, require consensus between agents
    require_consensus: bool = False
    consensus_threshold: float = 0.7  # Agreement ratio for consensus


@dataclass
class AgentAnalysis:
    """Analysis from a single agent."""

    agent_name: str
    root_cause: str
    approach: str
    confidence: float
    category: str | None = None
    raw_response: str = ""


class LLMFailureAnalyzer(AIAnalyzer):
    """LLM-powered failure analyzer using multiple agents.

    Implements the AIAnalyzer protocol for use with FailureAnalyzer.
    Uses multiple LLMs to analyze failures, cross-validate, and synthesize
    the best diagnosis.

    Example:
        config = LLMAnalyzerConfig(
            agent_types=["anthropic-api", "openai-api", "gemini-api"],
            cross_validate=True,
        )
        llm_analyzer = LLMFailureAnalyzer(config)

        # Use with FailureAnalyzer
        analyzer = FailureAnalyzer(
            repo_path=Path("."),
            ai_analyzer=llm_analyzer,
        )
    """

    def __init__(self, config: LLMAnalyzerConfig | None = None):
        """Initialize the LLM analyzer.

        Args:
            config: Configuration for the analyzer. Uses defaults if None.
        """
        self.config = config or LLMAnalyzerConfig()
        self.agents: list[Agent] = []
        self._initialized = False

    def _ensure_agents(self) -> None:
        """Lazily initialize agents on first use."""
        if self._initialized:
            return

        for agent_type in self.config.agent_types:
            try:
                model = None
                if self.config.models:
                    model = self.config.models.get(agent_type)

                agent = create_agent(
                    model_type=agent_type,  # type: ignore[arg-type]
                    name=f"analyzer_{agent_type}",
                    role="analyst",
                    model=model,
                    timeout=self.config.agent_timeout,
                )
                self.agents.append(agent)
                logger.info("llm_analyzer.agent_created type=%s", agent_type)
            except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
                logger.warning(
                    "llm_analyzer.agent_create_failed type=%s: %s",
                    agent_type,
                    e,
                )

        if not self.agents:
            raise RuntimeError("No agents could be initialized for LLM analysis")

        self._initialized = True

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate text with middle ellipsis."""
        if len(text) <= limit:
            return text
        head = limit // 2 - 100
        tail = limit // 2 - 100
        return f"{text[:head]}\n\n[... {len(text) - limit} chars truncated ...]\n\n{text[-tail:]}"

    def _extract_tag(self, pattern: re.Pattern, text: str) -> str | None:
        """Extract content from a tagged section."""
        match = pattern.search(text)
        if not match:
            return None
        return match.group(1).strip()

    def _parse_confidence(self, text: str, default: float = 0.5) -> float:
        """Parse confidence value from response."""
        raw = self._extract_tag(_CONFIDENCE_RE, text)
        if not raw:
            return default
        try:
            # Handle percentage format
            if "%" in raw:
                raw = raw.replace("%", "")
                return max(0.0, min(1.0, float(raw) / 100))
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return default

    def _build_analysis_prompt(
        self,
        failure: TestFailure,
        code_context: dict[str, str],
    ) -> str:
        """Build prompt for failure analysis."""
        context_str = ""
        if code_context:
            context_parts = []
            for file_path, code in code_context.items():
                truncated = self._truncate(code, self.config.max_context_chars // len(code_context))
                context_parts.append(f"### {file_path}\n```python\n{truncated}\n```")
            context_str = "\n\n".join(context_parts)

        categories = ", ".join(cat.value for cat in FailureCategory)

        return f"""You are an expert test failure analyst. Analyze the following test failure
and determine the root cause and best approach to fix it.

## Test Failure Information

**Test Name:** {failure.test_name}
**Test File:** {failure.test_file}
**Error Type:** {failure.error_type}
**Error Message:** {failure.error_message}

**Stack Trace:**
```
{self._truncate(failure.stack_trace, 10000)}
```

**Involved Files:** {", ".join(failure.involved_files[:10]) if failure.involved_files else "None identified"}

## Code Context
{context_str if context_str else "No code context available."}

## Your Task

Analyze this failure and provide:

1. **Root Cause**: What is the actual underlying issue causing this failure?
   - Go beyond the surface error message
   - Consider: Is the test wrong? Is the implementation wrong? Is it a setup issue?

2. **Category**: Which category best fits this failure?
   Categories: {categories}

3. **Approach**: What specific steps should be taken to fix this?
   - Be concrete and actionable
   - Specify which file(s) need changes
   - Describe what changes are needed

4. **Confidence**: How confident are you in this analysis (0.0 to 1.0)?
   - 0.9+: Very clear root cause, straightforward fix
   - 0.7-0.9: Likely root cause, fix may need adjustment
   - 0.5-0.7: Uncertain, multiple possibilities
   - <0.5: Unclear, needs more investigation

## Response Format

Use these exact tags in your response:

<category>category_name</category>
<root_cause>
Detailed explanation of the root cause...
</root_cause>
<approach>
Step-by-step approach to fix the issue...
</approach>
<confidence>0.X</confidence>
"""

    def _build_validation_prompt(
        self,
        failure: TestFailure,
        original_analysis: AgentAnalysis,
    ) -> str:
        """Build prompt for cross-validation of an analysis."""
        return f"""Review another AI's analysis of a test failure. Assess whether the
root cause identification and proposed approach are correct.

## Test Failure
**Test:** {failure.test_name}
**Error:** {failure.error_type}: {failure.error_message}

## Analysis to Review

**Root Cause:**
{original_analysis.root_cause}

**Proposed Approach:**
{original_analysis.approach}

**Stated Confidence:** {original_analysis.confidence:.2f}

## Your Task

1. Do you agree with the root cause identification?
2. Is the proposed approach likely to fix the issue?
3. Are there any issues or blind spots in this analysis?

## Response Format

<agreement>agree|partial|disagree</agreement>
<critique>
Your assessment of the analysis, including any issues or improvements...
</critique>
<confidence>0.X</confidence>
"""

    async def _get_agent_analysis(
        self,
        agent: Agent,
        prompt: str,
    ) -> AgentAnalysis | None:
        """Get analysis from a single agent with timeout."""
        start_time = time.perf_counter()
        logger.info("llm_analyzer.analysis.start agent=%s", agent.name)
        try:
            response = await asyncio.wait_for(
                agent.generate(prompt),
                timeout=self.config.agent_timeout,
            )
            duration = time.perf_counter() - start_time

            root_cause = self._extract_tag(_ROOT_CAUSE_RE, response)
            approach = self._extract_tag(_APPROACH_RE, response)
            confidence = self._parse_confidence(response)
            category = self._extract_tag(_CATEGORY_RE, response)

            if not root_cause:
                logger.warning(
                    "llm_analyzer.no_root_cause agent=%s",
                    agent.name,
                )
                return None

            return AgentAnalysis(
                agent_name=agent.name,
                root_cause=root_cause,
                approach=approach or "See root cause analysis for guidance.",
                confidence=confidence,
                category=category,
                raw_response=response,
            )

        except asyncio.TimeoutError:
            duration = time.perf_counter() - start_time
            logger.warning(
                "llm_analyzer.timeout agent=%s timeout=%.1f duration=%.2fs",
                agent.name,
                self.config.agent_timeout,
                duration,
            )
            return None
        except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
            duration = time.perf_counter() - start_time
            logger.warning(
                "llm_analyzer.agent_error agent=%s duration=%.2fs: %s",
                agent.name,
                duration,
                e,
            )
            return None

    async def _cross_validate(
        self,
        failure: TestFailure,
        analyses: list[AgentAnalysis],
    ) -> dict[str, list[tuple[str, bool, float]]]:
        """Have agents cross-validate each other's analyses.

        Returns:
            Dict mapping agent_name -> list of (critique, agrees, confidence)
        """
        validations: dict[str, list[tuple[str, bool, float]]] = {a.agent_name: [] for a in analyses}

        # Each agent reviews others' analyses
        validation_tasks = []
        task_mapping = []  # (validator_idx, target_analysis)

        for i, validator in enumerate(self.agents):
            for analysis in analyses:
                if analysis.agent_name == validator.name:
                    continue  # Don't self-validate

                prompt = self._build_validation_prompt(failure, analysis)
                validation_tasks.append(validator.generate(prompt))
                task_mapping.append((i, analysis))

        if not validation_tasks:
            return validations

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*validation_tasks, return_exceptions=True),
                timeout=self.config.agent_timeout * 2,
            )

            for (validator_idx, target_analysis), result in zip(task_mapping, results):
                if isinstance(result, Exception):
                    continue

                response = str(result)
                agreement = self._extract_tag(_AGREEMENT_RE, response) or "partial"
                critique = self._extract_tag(_CRITIQUE_RE, response) or ""
                confidence = self._parse_confidence(response)

                agrees = agreement.lower() == "agree"
                validations[target_analysis.agent_name].append((critique, agrees, confidence))

        except asyncio.TimeoutError:
            logger.warning("llm_analyzer.cross_validation_timeout")

        return validations

    def _synthesize_analyses(
        self,
        analyses: list[AgentAnalysis],
        validations: dict[str, list[tuple[str, bool, float]]] | None = None,
    ) -> tuple[str, str, float]:
        """Synthesize multiple analyses into the best result.

        Uses weighted voting based on:
        - Agent's stated confidence
        - Cross-validation agreement (if available)
        """
        if not analyses:
            return "", "", 0.0

        if len(analyses) == 1:
            a = analyses[0]
            return a.root_cause, a.approach, a.confidence

        # Calculate weighted scores
        scores: list[tuple[AgentAnalysis, float]] = []

        for analysis in analyses:
            base_score = analysis.confidence

            # Boost/penalize based on validation
            if validations and analysis.agent_name in validations:
                validation_list = validations[analysis.agent_name]
                if validation_list:
                    agreement_count = sum(1 for _, agrees, _ in validation_list if agrees)
                    agreement_ratio = agreement_count / len(validation_list)

                    # Boost up to 20% for full agreement, penalize for disagreement
                    validation_modifier = (agreement_ratio - 0.5) * 0.4
                    base_score = min(1.0, max(0.0, base_score + validation_modifier))

            scores.append((analysis, base_score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        best_analysis, best_score = scores[0]

        # Check for consensus
        if self.config.require_consensus and len(scores) > 1:
            second_best = scores[1]
            if best_score - second_best[1] < 0.1:
                # Close scores - check if root causes align
                best_words = set(best_analysis.root_cause.lower().split())
                second_words = set(second_best[0].root_cause.lower().split())
                overlap = len(best_words & second_words) / max(len(best_words | second_words), 1)

                if overlap < 0.5:
                    # Significant disagreement - lower confidence
                    best_score = best_score * 0.8
                    logger.info(
                        "llm_analyzer.low_consensus overlap=%.2f adjusted_confidence=%.2f",
                        overlap,
                        best_score,
                    )

        # Combine approaches from top analyses if they complement each other
        combined_approach = best_analysis.approach
        if len(scores) > 1 and scores[1][1] > self.config.synthesis_threshold:
            second = scores[1][0]
            # Check if second approach adds new information
            if len(second.approach) > 50 and second.approach not in combined_approach:
                combined_approach = (
                    f"{best_analysis.approach}\n\n"
                    f"Additional consideration from {second.agent_name}:\n"
                    f"{second.approach}"
                )

        logger.info(
            "llm_analyzer.synthesis_complete best_agent=%s score=%.2f num_analyses=%d",
            best_analysis.agent_name,
            best_score,
            len(analyses),
        )

        return best_analysis.root_cause, combined_approach, best_score

    async def analyze(
        self,
        failure: TestFailure,
        code_context: dict[str, str],
    ) -> tuple[str, str, float]:
        """Analyze failure using multiple LLM agents.

        Implements the AIAnalyzer protocol.

        Args:
            failure: The test failure to analyze
            code_context: Dict of file_path -> code content

        Returns:
            Tuple of (root_cause, suggested_approach, confidence)
        """
        self._ensure_agents()

        logger.info(
            "llm_analyzer.start test=%s num_agents=%d",
            failure.test_name,
            len(self.agents),
        )

        # Build analysis prompt
        prompt = self._build_analysis_prompt(failure, code_context)

        # Get analyses from all agents in parallel
        analysis_tasks = [self._get_agent_analysis(agent, prompt) for agent in self.agents]

        results = await asyncio.gather(*analysis_tasks)
        analyses = [a for a in results if a is not None]

        if not analyses:
            logger.warning("llm_analyzer.no_valid_analyses")
            return (
                f"Unable to determine root cause: {failure.error_message}",
                "Manual investigation required.",
                0.3,
            )

        logger.info(
            "llm_analyzer.analyses_complete num_valid=%d",
            len(analyses),
        )

        # Cross-validate if enabled and we have multiple analyses
        validations = None
        if self.config.cross_validate and len(analyses) > 1:
            validations = await self._cross_validate(failure, analyses)
            logger.info(
                "llm_analyzer.cross_validation_complete",
            )

        # Synthesize best analysis
        root_cause, approach, confidence = self._synthesize_analyses(
            analyses,
            validations,
        )

        logger.info(
            "llm_analyzer.complete test=%s confidence=%.2f",
            failure.test_name,
            confidence,
        )

        return root_cause, approach, confidence
