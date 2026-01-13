"""
Cognitive Load Limiter - Context preprocessing to prevent timeouts.

Limits context size to prevent agent cognitive overload:
- Token budget enforcement
- History truncation with relevance preservation
- Critique summarization
- Stress-level adaptive budgets

Inspired by nomic loop debate consensus on timeout prevention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Message, Critique
    from aragora.debate.complexity_governor import StressLevel

logger = logging.getLogger(__name__)

# Approximate chars per token (conservative estimate)
CHARS_PER_TOKEN = 4


@dataclass
class CognitiveBudget:
    """Token and content budgets for agent context."""

    max_context_tokens: int = 6000
    max_history_messages: int = 15
    max_critique_chars: int = 800
    max_proposal_chars: int = 2000
    max_patterns_chars: int = 500
    reserve_for_response: int = 2000

    @property
    def max_context_chars(self) -> int:
        """Maximum context size in characters."""
        return self.max_context_tokens * CHARS_PER_TOKEN

    def scale(self, factor: float) -> "CognitiveBudget":
        """Scale all budgets by a factor."""
        return CognitiveBudget(
            max_context_tokens=int(self.max_context_tokens * factor),
            max_history_messages=max(3, int(self.max_history_messages * factor)),
            max_critique_chars=int(self.max_critique_chars * factor),
            max_proposal_chars=int(self.max_proposal_chars * factor),
            max_patterns_chars=int(self.max_patterns_chars * factor),
            reserve_for_response=self.reserve_for_response,  # Keep response reserve
        )


# Preset budgets for different stress levels
STRESS_BUDGETS = {
    "nominal": CognitiveBudget(
        max_context_tokens=8000,
        max_history_messages=20,
        max_critique_chars=1000,
        max_proposal_chars=2500,
    ),
    "elevated": CognitiveBudget(
        max_context_tokens=6000,
        max_history_messages=15,
        max_critique_chars=800,
        max_proposal_chars=2000,
    ),
    "high": CognitiveBudget(
        max_context_tokens=4000,
        max_history_messages=10,
        max_critique_chars=500,
        max_proposal_chars=1500,
    ),
    "critical": CognitiveBudget(
        max_context_tokens=2000,
        max_history_messages=5,
        max_critique_chars=300,
        max_proposal_chars=800,
    ),
}


class CognitiveLoadLimiter:
    """
    Context preprocessor to prevent agent cognitive overload.

    Limits context size based on token budgets to prevent timeouts
    from overly large prompts. Uses stress-adaptive budgets.

    Usage:
        limiter = CognitiveLoadLimiter()

        # Limit message history
        limited_messages = limiter.limit_messages(
            messages, budget=CognitiveBudget()
        )

        # Limit critiques
        limited_critiques = limiter.limit_critiques(
            critiques, budget=CognitiveBudget()
        )

        # Get stress-appropriate limiter
        limiter = CognitiveLoadLimiter.for_stress_level("high")
    """

    def __init__(self, budget: Optional[CognitiveBudget] = None):
        """
        Initialize the limiter.

        Args:
            budget: Token budget to use (defaults to elevated)
        """
        self.budget = budget or STRESS_BUDGETS["elevated"]
        self.stats = {
            "messages_truncated": 0,
            "critiques_truncated": 0,
            "total_chars_removed": 0,
        }

    @classmethod
    def for_stress_level(cls, level: str) -> "CognitiveLoadLimiter":
        """
        Create a limiter appropriate for the stress level.

        Args:
            level: Stress level ("nominal", "elevated", "high", "critical")

        Returns:
            CognitiveLoadLimiter with appropriate budget
        """
        budget = STRESS_BUDGETS.get(level, STRESS_BUDGETS["elevated"])
        return cls(budget=budget)

    @classmethod
    def from_governor(cls) -> "CognitiveLoadLimiter":
        """Create limiter based on current complexity governor state."""
        try:
            from aragora.debate.complexity_governor import get_complexity_governor

            governor = get_complexity_governor()
            level = governor.stress_level.value
            return cls.for_stress_level(level)
        except ImportError:
            return cls()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        return len(text) // CHARS_PER_TOKEN if text else 0

    def limit_messages(
        self,
        messages: list[Any],
        max_messages: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> list[Any]:
        """
        Limit message history to fit within budget.

        Prioritizes recent messages while preserving first message
        (usually task description).

        Args:
            messages: List of Message objects
            max_messages: Override for max message count
            max_chars: Override for max total chars

        Returns:
            Truncated message list
        """
        if not messages:
            return []

        max_msgs = max_messages or self.budget.max_history_messages
        max_total_chars = max_chars or self.budget.max_context_chars

        # Keep first message (task) and most recent messages
        if len(messages) <= max_msgs:
            limited = list(messages)
        else:
            # Keep first and last (max_msgs - 1)
            limited = [messages[0]] + list(messages[-(max_msgs - 1) :])
            self.stats["messages_truncated"] += len(messages) - len(limited)

        # Now enforce character limit
        total_chars = sum(len(getattr(m, "content", str(m))) for m in limited)
        if total_chars > max_total_chars:
            limited = self._trim_to_char_limit(limited, max_total_chars)

        return limited

    def _trim_to_char_limit(
        self,
        messages: list[Any],
        max_chars: int,
    ) -> list[Any]:
        """Trim messages to fit character limit."""
        result = []
        total_chars = 0

        # Always include first message
        if messages:
            first_content = getattr(messages[0], "content", str(messages[0]))
            total_chars += len(first_content)
            result.append(messages[0])

        # Add messages from end until budget exhausted
        remaining = list(reversed(messages[1:]))
        to_add = []

        for msg in remaining:
            content = getattr(msg, "content", str(msg))
            msg_chars = len(content)

            if total_chars + msg_chars <= max_chars:
                total_chars += msg_chars
                to_add.append(msg)
            else:
                # Try truncating this message
                available = max_chars - total_chars
                if available > 200:  # Worth including truncated
                    truncated = self._truncate_message(msg, available)
                    to_add.append(truncated)
                    self.stats["total_chars_removed"] += msg_chars - available
                break

        result.extend(reversed(to_add))
        return result

    def _truncate_message(self, message: Any, max_chars: int) -> Any:
        """Truncate a single message to max chars."""
        content = getattr(message, "content", str(message))

        if len(content) <= max_chars:
            return message

        # Truncate with ellipsis
        half = max_chars // 2 - 20
        truncated = content[:half] + "\n\n[... truncated ...]\n\n" + content[-half:]

        # Handle different message types
        if hasattr(message, "_replace"):  # NamedTuple
            return message._replace(content=truncated)
        elif hasattr(message, "content"):  # Dataclass or object
            try:
                message.content = truncated
                return message
            except (AttributeError, TypeError) as e:
                # Object is immutable, return raw truncated content
                logger.debug(f"Could not mutate message object: {e}")

        return truncated

    def limit_critiques(
        self,
        critiques: list[Any],
        max_critiques: int = 5,
        max_chars_per: Optional[int] = None,
    ) -> list[Any]:
        """
        Limit and summarize critiques to fit budget.

        Args:
            critiques: List of Critique objects
            max_critiques: Maximum number of critiques to include
            max_chars_per: Max chars per critique

        Returns:
            Limited critique list with summarized content
        """
        if not critiques:
            return []

        max_chars = max_chars_per or self.budget.max_critique_chars

        # Sort by severity/importance if available
        sorted_critiques = sorted(
            critiques,
            key=lambda c: getattr(c, "severity", 0.5),
            reverse=True,
        )

        result = []
        for critique in sorted_critiques[:max_critiques]:
            # Summarize if too long
            reasoning = getattr(critique, "reasoning", str(critique))
            if len(reasoning) > max_chars:
                summarized = self._summarize_critique(critique, max_chars)
                result.append(summarized)
                self.stats["critiques_truncated"] += 1
            else:
                result.append(critique)

        return result

    def _summarize_critique(self, critique: Any, max_chars: int) -> Any:
        """Summarize a critique to fit within limit."""
        reasoning = getattr(critique, "reasoning", str(critique))

        # Extract key parts
        issues = getattr(critique, "issues", [])
        suggestions = getattr(critique, "suggestions", [])

        # Build summarized reasoning
        summary_parts = []
        if issues:
            summary_parts.append("Issues: " + "; ".join(issues[:3]))
        if suggestions:
            summary_parts.append("Suggestions: " + "; ".join(suggestions[:3]))

        summarized = " | ".join(summary_parts) if summary_parts else reasoning[:max_chars]

        # Truncate if still too long
        if len(summarized) > max_chars:
            summarized = summarized[: max_chars - 3] + "..."

        # Handle different critique types
        if hasattr(critique, "_replace"):
            return critique._replace(reasoning=summarized)
        elif hasattr(critique, "reasoning"):
            try:
                critique.reasoning = summarized
                return critique
            except (AttributeError, TypeError) as e:
                # Object is immutable, return raw summarized content
                logger.debug(f"Could not mutate critique object: {e}")

        return summarized

    def limit_context(
        self,
        messages: Optional[list] = None,
        critiques: Optional[list] = None,
        patterns: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> dict:
        """
        Limit all context components to fit within total budget.

        Args:
            messages: Message history
            critiques: Critiques received
            patterns: Pattern string
            extra_context: Any additional context

        Returns:
            Dict with limited versions of each component
        """
        result = {}

        # Allocate budget (messages get most)
        message_budget = int(self.budget.max_context_chars * 0.6)
        critique_budget = int(self.budget.max_context_chars * 0.25)
        other_budget = int(self.budget.max_context_chars * 0.15)

        if messages:
            result["messages"] = self.limit_messages(messages, max_chars=message_budget)

        if critiques:
            result["critiques"] = self.limit_critiques(
                critiques, max_chars_per=critique_budget // 5
            )

        if patterns:
            if len(patterns) > self.budget.max_patterns_chars:
                result["patterns"] = patterns[: self.budget.max_patterns_chars] + "..."
            else:
                result["patterns"] = patterns

        if extra_context:
            if len(extra_context) > other_budget:
                result["extra_context"] = extra_context[:other_budget] + "..."
            else:
                result["extra_context"] = extra_context

        total_chars = sum(
            (
                len(str(v))
                if isinstance(v, str)
                else sum(len(getattr(m, "content", str(m))) for m in v)
            )
            for v in result.values()
        )

        logger.debug(
            f"cognitive_limit total_chars={total_chars} "
            f"budget={self.budget.max_context_chars} "
            f"messages={len(result.get('messages', []))} "
            f"critiques={len(result.get('critiques', []))}"
        )

        return result

    def get_stats(self) -> dict:
        """Get limiter statistics."""
        return {
            **self.stats,
            "budget": {
                "max_tokens": self.budget.max_context_tokens,
                "max_messages": self.budget.max_history_messages,
            },
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "messages_truncated": 0,
            "critiques_truncated": 0,
            "total_chars_removed": 0,
        }


def limit_debate_context(
    messages: list,
    critiques: Optional[list] = None,
    stress_level: str = "elevated",
) -> dict:
    """
    Convenience function to limit debate context.

    Args:
        messages: Message history
        critiques: Optional critiques
        stress_level: Current stress level

    Returns:
        Dict with limited messages and critiques
    """
    limiter = CognitiveLoadLimiter.for_stress_level(stress_level)
    return limiter.limit_context(messages=messages, critiques=critiques)
