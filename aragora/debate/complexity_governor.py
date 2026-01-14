"""
Adaptive Complexity Governor - Dynamic context and constraint management.

Monitors system stress and dynamically adjusts:
- Context window size based on agent performance
- Prompt complexity when timeouts occur
- Debate parameters to prevent cascading failures
- Resource allocation across agents

Inspired by nomic loop debate synthesis on system stability.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from aragora.config import AGENT_TIMEOUT_SECONDS
from aragora.core import TaskComplexity

logger = logging.getLogger(__name__)


# Timeout scaling factors based on task complexity
COMPLEXITY_TIMEOUT_FACTORS = {
    TaskComplexity.SIMPLE: 0.5,  # 90s instead of 180s
    TaskComplexity.MODERATE: 1.0,  # 180s (default)
    TaskComplexity.COMPLEX: 1.5,  # 270s
    TaskComplexity.UNKNOWN: 0.75,  # 135s (conservative for unknown)
}


def classify_task_complexity(task: str) -> TaskComplexity:
    """Classify task complexity based on text signals.

    Uses keyword matching to estimate whether a task is simple, moderate,
    or complex. This affects timeout allocation.

    Args:
        task: The task description text

    Returns:
        TaskComplexity enum value
    """
    if not task:
        return TaskComplexity.UNKNOWN

    task_lower = task.lower()

    # Complex indicators - deep reasoning, formal methods, multi-step
    complex_signals = [
        "prove",
        "formally",
        "verify",
        "optimize",
        "design",  # Design tasks are generally complex
        "architecture",
        "trade-off",
        "comprehensive",
        "analyze",
        "implement",
        "refactor",
        "security",
        "performance",
        "scalability",
        "distributed",
        "algorithm",
        "system",  # System-level tasks are complex
    ]
    if any(s in task_lower for s in complex_signals):
        return TaskComplexity.COMPLEX

    # Simple indicators - quick lookups, definitions, short answers
    simple_signals = [
        "what is",
        "define ",  # Note space to avoid matching "defined"
        "list the",
        "quick",
        "simple",
        "basic",
        "name the",
        "which is",
        "yes or no",
        "true or false",
        "capital of",
        "how many",
        "who is",
        "when did",
        "where is",
    ]
    if any(s in task_lower for s in simple_signals):
        return TaskComplexity.SIMPLE

    # Length heuristic - only for very short or very long tasks
    if len(task) < 30:
        return TaskComplexity.SIMPLE
    elif len(task) > 500:
        return TaskComplexity.COMPLEX

    return TaskComplexity.MODERATE


class StressLevel(Enum):
    """System stress levels."""

    NOMINAL = "nominal"  # Normal operation
    ELEVATED = "elevated"  # Some pressure, slight adjustments
    HIGH = "high"  # Significant pressure, major adjustments
    CRITICAL = "critical"  # Emergency mode, minimal operation


@dataclass
class GovernorConstraints:
    """Constraints to apply to the debate system."""

    # Context limits
    max_context_tokens: int = 8000
    max_history_messages: int = 20
    max_prompt_length: int = 4000

    # Timing constraints - defaults from AGENT_TIMEOUT_SECONDS config
    # Set to None to use config value, or override explicitly
    agent_timeout_seconds: Optional[float] = None
    round_timeout_seconds: Optional[float] = None

    # Complexity constraints
    max_agents_per_round: int = 5
    max_critique_length: int = 1000
    max_proposal_length: int = 2000

    # Feature toggles
    enable_deep_analysis: bool = True
    enable_cross_references: bool = True
    enable_formal_verification: bool = True

    def __post_init__(self):
        """Apply defaults from config when not explicitly set."""
        if self.agent_timeout_seconds is None:
            self.agent_timeout_seconds = float(AGENT_TIMEOUT_SECONDS)
        if self.round_timeout_seconds is None:
            # Round timeout is 2x agent timeout by default
            self.round_timeout_seconds = self.agent_timeout_seconds * 2

    def to_dict(self) -> dict:
        return {
            "max_context_tokens": self.max_context_tokens,
            "max_history_messages": self.max_history_messages,
            "max_prompt_length": self.max_prompt_length,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "round_timeout_seconds": self.round_timeout_seconds,
            "max_agents_per_round": self.max_agents_per_round,
            "max_critique_length": self.max_critique_length,
            "max_proposal_length": self.max_proposal_length,
            "enable_deep_analysis": self.enable_deep_analysis,
            "enable_cross_references": self.enable_cross_references,
            "enable_formal_verification": self.enable_formal_verification,
        }


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for an agent."""

    name: str
    total_requests: int = 0
    successful_requests: int = 0
    timeout_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_response_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    @property
    def timeout_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.timeout_count / self.total_requests


class AdaptiveComplexityGovernor:
    """
    Dynamically adjusts system complexity based on performance.

    The governor monitors agent performance and system stress,
    automatically adjusting constraints to prevent cascading failures
    while maximizing capability when the system is healthy.

    Usage:
        governor = AdaptiveComplexityGovernor()

        # Before each round
        constraints = governor.get_constraints()
        # Apply constraints to debate...

        # After agent response
        governor.record_agent_response("claude", latency_ms=1500, success=True)

        # After timeout
        governor.record_agent_timeout("deepseek")

        # Check if adjustments needed
        if governor.should_reduce_complexity():
            constraints = governor.get_reduced_constraints()
    """

    # Thresholds for stress level transitions (lowered for earlier detection)
    STRESS_THRESHOLDS = {
        "timeout_rate_elevated": 0.05,  # 5% timeouts -> elevated (was 10%)
        "timeout_rate_high": 0.15,  # 15% timeouts -> high (was 30%)
        "timeout_rate_critical": 0.30,  # 30% timeouts -> critical (was 50%)
        "consecutive_failures": 2,  # Consecutive failures -> escalate
        "latency_elevated_ms": 15000,  # 15s avg -> elevated (was 30s)
        "latency_high_ms": 30000,  # 30s avg -> high (was 60s)
    }

    # Constraint presets for each stress level
    CONSTRAINT_PRESETS = {
        StressLevel.NOMINAL: GovernorConstraints(
            max_context_tokens=8000,
            max_history_messages=20,
            agent_timeout_seconds=120.0,
            max_agents_per_round=5,
            enable_deep_analysis=True,
            enable_cross_references=True,
            enable_formal_verification=True,
        ),
        StressLevel.ELEVATED: GovernorConstraints(
            max_context_tokens=6000,
            max_history_messages=15,
            agent_timeout_seconds=90.0,
            max_agents_per_round=4,
            enable_deep_analysis=True,
            enable_cross_references=True,
            enable_formal_verification=False,
        ),
        StressLevel.HIGH: GovernorConstraints(
            max_context_tokens=4000,
            max_history_messages=10,
            agent_timeout_seconds=60.0,
            max_agents_per_round=3,
            enable_deep_analysis=False,
            enable_cross_references=False,
            enable_formal_verification=False,
        ),
        StressLevel.CRITICAL: GovernorConstraints(
            max_context_tokens=2000,
            max_history_messages=5,
            agent_timeout_seconds=30.0,
            max_agents_per_round=2,
            enable_deep_analysis=False,
            enable_cross_references=False,
            enable_formal_verification=False,
        ),
    }

    def __init__(
        self,
        initial_constraints: Optional[GovernorConstraints] = None,
        stress_callback: Optional[Callable[[StressLevel], None]] = None,
    ):
        """
        Initialize the governor.

        Args:
            initial_constraints: Starting constraints (uses NOMINAL if None)
            stress_callback: Called when stress level changes
        """
        self.current_constraints = initial_constraints or GovernorConstraints()
        self.stress_level = StressLevel.NOMINAL
        self.stress_callback = stress_callback

        self.agent_metrics: dict[str, AgentPerformanceMetrics] = {}
        self.round_history: list[dict] = []
        self.consecutive_failures = 0
        self.last_adjustment_time = time.time()

        # Task complexity for timeout scaling
        self.task_complexity: TaskComplexity = TaskComplexity.MODERATE

        # Minimum time between adjustments (prevent thrashing)
        self.adjustment_cooldown_seconds = 60.0

        logger.info("complexity_governor_init")

    def _get_agent_metrics(self, agent_name: str) -> AgentPerformanceMetrics:
        """Get or create metrics for an agent."""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentPerformanceMetrics(name=agent_name)
        return self.agent_metrics[agent_name]

    def record_agent_response(
        self,
        agent_name: str,
        latency_ms: float,
        success: bool,
        response_tokens: int = 0,
    ) -> None:
        """
        Record an agent response for metrics tracking.

        Args:
            agent_name: Name of the agent
            latency_ms: Response latency in milliseconds
            success: Whether the response was successful
            response_tokens: Number of tokens in response (optional)
        """
        metrics = self._get_agent_metrics(agent_name)
        metrics.total_requests += 1
        metrics.last_response_time = time.time()

        if success:
            metrics.successful_requests += 1
            metrics.total_latency_ms += latency_ms
            self.consecutive_failures = 0
        else:
            metrics.error_count += 1
            self.consecutive_failures += 1

        self._evaluate_stress_level()
        logger.debug(
            f"governor_record agent={agent_name} success={success} latency_ms={latency_ms:.0f}"
        )

    def record_agent_timeout(
        self,
        agent_name: str,
        timeout_seconds: float,
    ) -> None:
        """
        Record an agent timeout.

        Args:
            agent_name: Name of the agent
            timeout_seconds: How long before timeout
        """
        metrics = self._get_agent_metrics(agent_name)
        metrics.total_requests += 1
        metrics.timeout_count += 1
        self.consecutive_failures += 1

        self._evaluate_stress_level()
        logger.warning(
            f"governor_timeout agent={agent_name} timeout_s={timeout_seconds} "
            f"consecutive={self.consecutive_failures}"
        )

    def record_round_complete(
        self,
        round_id: int,
        duration_seconds: float,
        agents_participated: int,
        agents_failed: int,
    ) -> None:
        """
        Record round completion for analysis.

        Args:
            round_id: Round number
            duration_seconds: Total round duration
            agents_participated: Number of agents that participated
            agents_failed: Number of agents that failed
        """
        self.round_history.append(
            {
                "round_id": round_id,
                "timestamp": time.time(),
                "duration_seconds": duration_seconds,
                "agents_participated": agents_participated,
                "agents_failed": agents_failed,
                "stress_level": self.stress_level.value,
            }
        )

        # Keep history bounded
        if len(self.round_history) > 100:
            self.round_history = self.round_history[-50:]

    def _evaluate_stress_level(self) -> None:
        """Evaluate and potentially adjust stress level."""
        if not self.agent_metrics:
            return

        # Calculate aggregate metrics
        total_requests = sum(m.total_requests for m in self.agent_metrics.values())
        total_timeouts = sum(m.timeout_count for m in self.agent_metrics.values())
        total_latency = sum(m.total_latency_ms for m in self.agent_metrics.values())
        successful = sum(m.successful_requests for m in self.agent_metrics.values())

        if total_requests == 0:
            return

        timeout_rate = total_timeouts / total_requests
        avg_latency = total_latency / max(successful, 1)

        # Determine appropriate stress level
        new_level = StressLevel.NOMINAL

        # Check consecutive failures first (immediate escalation)
        if self.consecutive_failures >= self.STRESS_THRESHOLDS["consecutive_failures"]:
            new_level = StressLevel.HIGH

        # Check timeout rate
        if timeout_rate >= self.STRESS_THRESHOLDS["timeout_rate_critical"]:
            new_level = StressLevel.CRITICAL
        elif timeout_rate >= self.STRESS_THRESHOLDS["timeout_rate_high"]:
            new_level = max(new_level, StressLevel.HIGH, key=lambda x: list(StressLevel).index(x))
        elif timeout_rate >= self.STRESS_THRESHOLDS["timeout_rate_elevated"]:
            new_level = max(
                new_level, StressLevel.ELEVATED, key=lambda x: list(StressLevel).index(x)
            )

        # Check latency
        if avg_latency >= self.STRESS_THRESHOLDS["latency_high_ms"]:
            new_level = max(new_level, StressLevel.HIGH, key=lambda x: list(StressLevel).index(x))
        elif avg_latency >= self.STRESS_THRESHOLDS["latency_elevated_ms"]:
            new_level = max(
                new_level, StressLevel.ELEVATED, key=lambda x: list(StressLevel).index(x)
            )

        # Apply new level if changed
        if new_level != self.stress_level:
            self._transition_stress_level(new_level)

    def _transition_stress_level(self, new_level: StressLevel) -> None:
        """Transition to a new stress level."""
        # Check cooldown
        if time.time() - self.last_adjustment_time < self.adjustment_cooldown_seconds:
            # Only allow escalation during cooldown, not de-escalation
            if list(StressLevel).index(new_level) <= list(StressLevel).index(self.stress_level):
                return

        old_level = self.stress_level
        self.stress_level = new_level
        self.current_constraints = self.CONSTRAINT_PRESETS[new_level]
        self.last_adjustment_time = time.time()

        logger.info(f"governor_stress_change old={old_level.value} new={new_level.value}")

        if self.stress_callback:
            try:
                self.stress_callback(new_level)
            except Exception as e:
                logger.error(f"governor_callback_failed error={e}")

    def get_constraints(self) -> GovernorConstraints:
        """Get current constraints for the debate system."""
        return self.current_constraints

    def get_agent_constraints(self, agent_name: str) -> dict:
        """
        Get agent-specific constraints based on performance.

        Agents with poor performance get stricter constraints.
        """
        metrics = self.agent_metrics.get(agent_name)
        base = self.current_constraints

        if not metrics or metrics.total_requests < 3:
            # Not enough data, use defaults
            return {
                "timeout_seconds": base.agent_timeout_seconds,
                "max_response_tokens": base.max_proposal_length,
            }

        # Adjust timeout based on agent's average latency
        timeout = base.agent_timeout_seconds
        if metrics.avg_latency_ms > 0:
            # Give 2x the average latency, capped
            timeout = min(
                base.agent_timeout_seconds * 1.5, max(30, metrics.avg_latency_ms / 1000 * 2)
            )

        # Reduce max tokens for slow agents
        max_tokens = base.max_proposal_length
        if metrics.timeout_rate > 0.2:
            max_tokens = int(max_tokens * 0.7)

        return {
            "timeout_seconds": timeout,
            "max_response_tokens": max_tokens,
            "reliability_score": metrics.success_rate,
        }

    def should_reduce_complexity(self) -> bool:
        """Check if complexity should be reduced."""
        return self.stress_level in (StressLevel.HIGH, StressLevel.CRITICAL)

    def should_skip_agent(self, agent_name: str) -> bool:
        """
        Check if an agent should be skipped due to poor performance.

        Returns True if agent has very high failure rate.
        """
        metrics = self.agent_metrics.get(agent_name)
        if not metrics or metrics.total_requests < 5:
            return False

        # Skip if >70% timeout rate with enough samples
        return metrics.timeout_rate > 0.7

    def get_recommended_agent_count(self) -> int:
        """Get recommended number of agents for current stress level."""
        return self.current_constraints.max_agents_per_round

    def set_task_complexity(self, complexity: TaskComplexity) -> None:
        """Set the task complexity for timeout scaling.

        Should be called at the start of a debate to configure
        complexity-based timeout adjustments.

        Args:
            complexity: The classified task complexity
        """
        self.task_complexity = complexity
        logger.info(f"governor_task_complexity complexity={complexity.value}")

    def get_scaled_timeout(self, base_timeout: float = 180.0) -> float:
        """Get timeout scaled by both stress level and task complexity.

        The timeout is first adjusted by task complexity (simple = faster,
        complex = more time), then constrained by stress level.

        Args:
            base_timeout: Base timeout in seconds (default 180s)

        Returns:
            Scaled timeout in seconds
        """
        # Apply task complexity scaling
        complexity_factor = COMPLEXITY_TIMEOUT_FACTORS.get(self.task_complexity, 1.0)
        scaled = base_timeout * complexity_factor

        # Apply stress level constraints (stress reduces available time)
        stress_limit = self.current_constraints.agent_timeout_seconds
        final_timeout = min(scaled, stress_limit)

        logger.debug(
            f"governor_scaled_timeout base={base_timeout:.0f} "
            f"complexity={self.task_complexity.value}({complexity_factor}) "
            f"stress={self.stress_level.value}(limit={stress_limit:.0f}) "
            f"final={final_timeout:.0f}"
        )

        return final_timeout

    def get_status(self) -> dict:
        """Get governor status for monitoring."""
        return {
            "stress_level": self.stress_level.value,
            "task_complexity": self.task_complexity.value,
            "constraints": self.current_constraints.to_dict(),
            "consecutive_failures": self.consecutive_failures,
            "agent_metrics": {
                name: {
                    "success_rate": round(m.success_rate, 2),
                    "timeout_rate": round(m.timeout_rate, 2),
                    "avg_latency_ms": round(m.avg_latency_ms, 0),
                    "total_requests": m.total_requests,
                }
                for name, m in self.agent_metrics.items()
            },
            "recent_rounds": self.round_history[-5:],
        }

    def reset_metrics(self) -> None:
        """Reset all metrics (e.g., for new debate session)."""
        self.agent_metrics.clear()
        self.round_history.clear()
        self.consecutive_failures = 0
        self.stress_level = StressLevel.NOMINAL
        self.task_complexity = TaskComplexity.MODERATE
        self.current_constraints = self.CONSTRAINT_PRESETS[StressLevel.NOMINAL]
        logger.info("governor_reset")


# Global governor instance
_governor: Optional[AdaptiveComplexityGovernor] = None


def get_complexity_governor() -> AdaptiveComplexityGovernor:
    """Get the global complexity governor instance."""
    global _governor
    if _governor is None:
        _governor = AdaptiveComplexityGovernor()
    return _governor


def reset_complexity_governor() -> None:
    """Reset the global complexity governor (for testing)."""
    global _governor
    _governor = None
