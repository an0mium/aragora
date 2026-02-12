"""
ThinkPRM: Process Reward Models That Think.

Based on: https://arxiv.org/abs/2504.16828

Provides verbalized step-wise verification with minimal labels.
Integrates with Aragora's debate rounds as "steps" to verify.

Key features:
- Step-by-step verification of debate reasoning
- Verbalized reasoning for explainable verdicts
- Integration with ASCoT fragility for targeted scrutiny
- Critical error detection in late-stage rounds
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Protocol, Callable
from enum import Enum
import asyncio
import logging
import re
import time

logger = logging.getLogger(__name__)


class StepVerdict(Enum):
    """Verdict for a single verification step."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    NEEDS_REVISION = "needs_revision"


@dataclass
class StepVerification:
    """Verification result for a single debate step."""

    step_id: str
    round_number: int
    agent_id: str
    content_summary: str
    verdict: StepVerdict
    confidence: float
    reasoning: str
    suggested_fix: str | None = None
    dependencies_verified: bool = True
    verification_time_ms: float = 0.0


@dataclass
class ProcessVerificationResult:
    """Full process verification across all debate rounds."""

    debate_id: str
    total_steps: int
    correct_steps: int
    incorrect_steps: int
    uncertain_steps: int
    needs_revision_steps: int
    overall_score: float  # 0.0-1.0
    critical_errors: list[StepVerification] = field(default_factory=list)
    step_results: list[StepVerification] = field(default_factory=list)
    total_time_ms: float = 0.0


@dataclass
class ThinkPRMConfig:
    """Configuration for ThinkPRM verifier."""

    verifier_agent_id: str = "claude"
    max_context_chars: int = 2000
    max_tokens: int = 1000
    critical_round_threshold: float = 0.7  # Last 30% of rounds are "critical"
    cache_verifications: bool = True
    parallel_verification: bool = True
    max_parallel: int = 3
    timeout_seconds: int = 60


class AgentPoolProtocol(Protocol):
    """Protocol for agent pool interaction."""

    async def query(
        self,
        agent_id: str,
        prompt: str,
        max_tokens: int = 1000,
    ) -> str:
        """Query an agent with a prompt."""
        ...


class ThinkPRMVerifier:
    """
    Process Reward Model for debate step verification.

    Uses verbalized reasoning to verify each debate round:
    1. Extract claim from round
    2. Check logical consistency with prior rounds
    3. Verify evidence citations
    4. Assess reasoning validity

    Example:
        verifier = ThinkPRMVerifier()

        # Verify single step
        result = await verifier.verify_step(
            step_content="The temperature rose because of increased CO2...",
            round_number=3,
            agent_id="agent1",
            prior_context="Prior rounds discussed climate factors...",
            dependencies=["CO2 levels increased 40% since 1850"],
            query_fn=agent_pool.query,
        )

        if result.verdict == StepVerdict.INCORRECT:
            print(f"Error found: {result.reasoning}")
            print(f"Suggested fix: {result.suggested_fix}")
    """

    VERIFICATION_PROMPT = """You are a rigorous debate step verifier. Analyze the following debate step and determine if the reasoning is valid.

PRIOR CONTEXT (what came before):
{prior_context}

CURRENT STEP TO VERIFY:
Round {round_number} by {agent_id}:
{step_content}

CLAIMED DEPENDENCIES:
{dependencies}

VERIFICATION TASKS:
1. Is the logical reasoning valid?
2. Are the claimed dependencies actually used correctly?
3. Are there any factual errors or unsupported claims?
4. Does this step build correctly on prior context?

Respond in this exact format:
VERDICT: [CORRECT|INCORRECT|UNCERTAIN|NEEDS_REVISION]
CONFIDENCE: [0.0-1.0]
REASONING: [Your detailed analysis]
SUGGESTED_FIX: [If INCORRECT or NEEDS_REVISION, what should change. Otherwise write "None"]"""

    def __init__(self, config: ThinkPRMConfig | None = None):
        """Initialize the verifier.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or ThinkPRMConfig()
        self._cache: dict[str, StepVerification] = {}
        self._verification_history: list[StepVerification] = []

    async def verify_step(
        self,
        step_content: str,
        round_number: int,
        agent_id: str,
        prior_context: str,
        dependencies: list[str],
        query_fn: Callable,
    ) -> StepVerification:
        """
        Verify a single debate step using verbalized reasoning.

        Args:
            step_content: The content of the step to verify
            round_number: Round number (1-indexed)
            agent_id: ID of the agent who produced this step
            prior_context: Context from previous rounds
            dependencies: List of claimed dependencies/citations
            query_fn: Async function to query the verifier agent

        Returns:
            StepVerification with verdict and reasoning
        """
        start_time = time.time()

        # Generate cache key
        cache_key = f"{round_number}_{agent_id}_{hash(step_content) % 10000}"

        # Check cache
        if self.config.cache_verifications and cache_key in self._cache:
            logger.debug("verification_cache_hit key=%s", cache_key)
            return self._cache[cache_key]

        # Format verification prompt
        prompt = self.VERIFICATION_PROMPT.format(
            prior_context=prior_context[: self.config.max_context_chars],
            round_number=round_number,
            agent_id=agent_id,
            step_content=step_content,
            dependencies="\n".join(f"- {d}" for d in dependencies)
            if dependencies
            else "None claimed",
        )

        # Query verifier agent
        try:
            response = await asyncio.wait_for(
                query_fn(
                    agent_id=self.config.verifier_agent_id,
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                ),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("verification_timeout round=%d agent=%s", round_number, agent_id)
            return StepVerification(
                step_id=f"step_{round_number}_{agent_id}",
                round_number=round_number,
                agent_id=agent_id,
                content_summary=step_content[:200],
                verdict=StepVerdict.UNCERTAIN,
                confidence=0.0,
                reasoning="Verification timed out",
                verification_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error("verification_error round=%d agent=%s error=%s", round_number, agent_id, e)
            return StepVerification(
                step_id=f"step_{round_number}_{agent_id}",
                round_number=round_number,
                agent_id=agent_id,
                content_summary=step_content[:200],
                verdict=StepVerdict.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Verification error: {str(e)}",
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        # Parse response
        result = self._parse_verification_response(
            response=response,
            step_content=step_content,
            round_number=round_number,
            agent_id=agent_id,
        )
        result.verification_time_ms = (time.time() - start_time) * 1000

        # Cache result
        if self.config.cache_verifications:
            self._cache[cache_key] = result

        # Track history
        self._verification_history.append(result)

        logger.debug(
            "step_verified round=%d agent=%s verdict=%s confidence=%.2f time_ms=%.1f",
            round_number,
            agent_id,
            result.verdict.value,
            result.confidence,
            result.verification_time_ms,
        )

        return result

    async def verify_debate_process(
        self,
        debate_rounds: list[dict[str, Any]],
        query_fn: Callable,
        fragility_scores: dict[int, float] | None = None,
    ) -> ProcessVerificationResult:
        """
        Verify entire debate as a sequence of steps.

        Applies fragility-aware verification: later rounds get more scrutiny.

        Args:
            debate_rounds: List of round dicts with 'contributions' list
            query_fn: Async function to query the verifier agent
            fragility_scores: Optional mapping of round_number to fragility score

        Returns:
            ProcessVerificationResult with overall assessment
        """
        start_time = time.time()
        step_results: list[StepVerification] = []
        prior_context = ""

        total_rounds = len(debate_rounds)
        critical_threshold = int(total_rounds * self.config.critical_round_threshold)

        for i, round_data in enumerate(debate_rounds):
            round_number = i + 1
            contributions = round_data.get("contributions", [])

            # Get fragility for this round (if provided)
            # NOTE: Fragility-adjusted verification intensity is a planned enhancement.
            # The fragility score is calculated but not yet used to vary verification depth.
            # Higher fragility rounds could warrant deeper verification or multiple passes.
            # See: https://arxiv.org/abs/2510.12697 for theoretical foundation.
            _round_fragility = (
                fragility_scores.get(round_number, 0.5)
                if fragility_scores
                else round_number / total_rounds
            )

            # Determine if we should verify in parallel
            if self.config.parallel_verification and len(contributions) > 1:
                # Verify contributions in parallel
                tasks = []
                for contribution in contributions:
                    task = self.verify_step(
                        step_content=contribution.get("content", ""),
                        round_number=round_number,
                        agent_id=contribution.get("agent_id", "unknown"),
                        prior_context=prior_context,
                        dependencies=contribution.get("dependencies", []),
                        query_fn=query_fn,
                    )
                    tasks.append(task)

                # Limit parallelism
                for batch_start in range(0, len(tasks), self.config.max_parallel):
                    batch = tasks[batch_start : batch_start + self.config.max_parallel]
                    results = await asyncio.gather(*batch, return_exceptions=True)
                    for r in results:
                        if isinstance(r, BaseException):
                            logger.error("batch_verification_error: %s", r)
                        else:
                            step_results.append(r)  # type: ignore[arg-type]
            else:
                # Sequential verification
                for contribution in contributions:
                    step_result = await self.verify_step(
                        step_content=contribution.get("content", ""),
                        round_number=round_number,
                        agent_id=contribution.get("agent_id", "unknown"),
                        prior_context=prior_context,
                        dependencies=contribution.get("dependencies", []),
                        query_fn=query_fn,
                    )
                    step_results.append(step_result)

            # Update prior context for next round
            prior_context += f"\n\nRound {round_number}:\n"
            prior_context += "\n".join(c.get("content", "")[:500] for c in contributions)

        # Calculate statistics
        correct = sum(1 for s in step_results if s.verdict == StepVerdict.CORRECT)
        incorrect = sum(1 for s in step_results if s.verdict == StepVerdict.INCORRECT)
        uncertain = sum(1 for s in step_results if s.verdict == StepVerdict.UNCERTAIN)
        needs_revision = sum(1 for s in step_results if s.verdict == StepVerdict.NEEDS_REVISION)
        total = len(step_results)

        overall_score = correct / total if total > 0 else 0.0

        # Identify critical errors (errors in late-stage rounds)
        critical_errors = [
            s
            for s in step_results
            if s.verdict in (StepVerdict.INCORRECT, StepVerdict.NEEDS_REVISION)
            and s.round_number >= critical_threshold
        ]

        debate_id = debate_rounds[0].get("debate_id", "unknown") if debate_rounds else "unknown"

        result: ProcessVerificationResult = ProcessVerificationResult(
            debate_id=debate_id,
            total_steps=total,
            correct_steps=correct,
            incorrect_steps=incorrect,
            uncertain_steps=uncertain,
            needs_revision_steps=needs_revision,
            overall_score=overall_score,
            critical_errors=critical_errors,
            step_results=step_results,
            total_time_ms=(time.time() - start_time) * 1000,
        )

        logger.info(
            "debate_verified id=%s steps=%d correct=%d incorrect=%d "
            "critical_errors=%d score=%.2f time_ms=%.1f",
            debate_id,
            total,
            correct,
            incorrect,
            len(critical_errors),
            overall_score,
            result.total_time_ms,
        )

        return result

    def _parse_verification_response(
        self,
        response: str,
        step_content: str,
        round_number: int,
        agent_id: str,
    ) -> StepVerification:
        """Parse LLM verification response into structured result."""
        lines = response.strip().split("\n")

        verdict = StepVerdict.UNCERTAIN
        confidence = 0.5
        reasoning = ""
        suggested_fix = None

        for line in lines:
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                verdict_str = line.split(":", 1)[1].strip().upper()
                verdict_map = {
                    "CORRECT": StepVerdict.CORRECT,
                    "INCORRECT": StepVerdict.INCORRECT,
                    "UNCERTAIN": StepVerdict.UNCERTAIN,
                    "NEEDS_REVISION": StepVerdict.NEEDS_REVISION,
                }
                verdict = verdict_map.get(verdict_str, StepVerdict.UNCERTAIN)

            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":", 1)[1].strip()
                    # Handle formats like "0.8" or "80%" or "0.8 (high)"
                    conf_str = re.sub(r"[^\d.]", "", conf_str.split()[0])
                    confidence = float(conf_str)
                    if confidence > 1.0:
                        confidence = confidence / 100.0  # Convert percentage
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5

            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

            elif line.upper().startswith("SUGGESTED_FIX:"):
                fix = line.split(":", 1)[1].strip()
                if fix.lower() not in ("none", "n/a", ""):
                    suggested_fix = fix

        # If reasoning spans multiple lines, try to capture more
        if not reasoning:
            # Look for reasoning in the full text
            if "REASONING:" in response.upper():
                parts = response.upper().split("REASONING:", 1)
                if len(parts) > 1:
                    reasoning_part = response[len(parts[0]) + len("REASONING:") :]
                    # Take up to SUGGESTED_FIX
                    if "SUGGESTED_FIX:" in reasoning_part.upper():
                        reasoning = reasoning_part.split("SUGGESTED_FIX:")[0].strip()
                    else:
                        reasoning = reasoning_part.strip()

        return StepVerification(
            step_id=f"step_{round_number}_{agent_id}",
            round_number=round_number,
            agent_id=agent_id,
            content_summary=step_content[:200],
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning or "No reasoning provided",
            suggested_fix=suggested_fix,
        )

    def clear_cache(self) -> None:
        """Clear the verification cache."""
        self._cache.clear()

    def reset(self) -> None:
        """Reset verifier state."""
        self._cache.clear()
        self._verification_history.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get verifier metrics for telemetry."""
        if not self._verification_history:
            return {
                "total_verifications": 0,
                "verdict_distribution": {},
                "avg_confidence": 0.0,
                "avg_time_ms": 0.0,
            }

        verdicts = [v.verdict.value for v in self._verification_history]
        verdict_counts: dict[str, int] = {}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        total = len(self._verification_history)

        return {
            "total_verifications": total,
            "verdict_distribution": {k: v / total for k, v in verdict_counts.items()},
            "verdict_counts": verdict_counts,
            "avg_confidence": sum(v.confidence for v in self._verification_history) / total,
            "avg_time_ms": sum(v.verification_time_ms for v in self._verification_history) / total,
            "cache_size": len(self._cache),
        }


# Convenience functions


def create_think_prm_verifier(
    verifier_agent_id: str = "claude",
    parallel: bool = True,
    **kwargs: Any,
) -> ThinkPRMVerifier:
    """Create a ThinkPRM verifier with common configuration.

    Args:
        verifier_agent_id: Agent to use for verification
        parallel: Whether to verify in parallel
        **kwargs: Additional config options

    Returns:
        Configured ThinkPRMVerifier
    """
    config = ThinkPRMConfig(
        verifier_agent_id=verifier_agent_id,
        parallel_verification=parallel,
        **kwargs,
    )
    return ThinkPRMVerifier(config)


async def verify_single_step(
    step_content: str,
    round_number: int,
    agent_id: str,
    query_fn: Callable,
    prior_context: str = "",
    dependencies: list[str] | None = None,
) -> StepVerification:
    """
    Convenience function to verify a single step.

    Args:
        step_content: Content to verify
        round_number: Round number
        agent_id: Agent who produced this step
        query_fn: Function to query verifier agent
        prior_context: Optional context from prior rounds
        dependencies: Optional list of dependencies

    Returns:
        StepVerification result
    """
    verifier = ThinkPRMVerifier()
    return await verifier.verify_step(
        step_content=step_content,
        round_number=round_number,
        agent_id=agent_id,
        prior_context=prior_context,
        dependencies=dependencies or [],
        query_fn=query_fn,
    )
