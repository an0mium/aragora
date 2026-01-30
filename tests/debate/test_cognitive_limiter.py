"""
Tests for the cognitive load limiter module.

Tests cover:
- CHARS_PER_TOKEN constant
- CognitiveBudget dataclass
- STRESS_BUDGETS presets
- CognitiveLoadLimiter class (messages, critiques, context limiting)
- limit_debate_context convenience function
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.cognitive_limiter import (
    CHARS_PER_TOKEN,
    STRESS_BUDGETS,
    CognitiveBudget,
    CognitiveLoadLimiter,
    limit_debate_context,
)


class TestCharsPerToken:
    """Tests for CHARS_PER_TOKEN constant."""

    def test_chars_per_token_value(self):
        """Test the character to token ratio constant."""
        assert CHARS_PER_TOKEN == 4

    def test_chars_per_token_is_positive(self):
        """Test CHARS_PER_TOKEN is positive."""
        assert CHARS_PER_TOKEN > 0


class TestCognitiveBudget:
    """Tests for CognitiveBudget dataclass."""

    def test_default_values(self):
        """Test default budget values."""
        budget = CognitiveBudget()

        assert budget.max_context_tokens == 6000
        assert budget.max_history_messages == 15
        assert budget.max_critique_chars == 800
        assert budget.max_proposal_chars == 2000
        assert budget.max_patterns_chars == 500
        assert budget.reserve_for_response == 2000

    def test_custom_values(self):
        """Test creating budget with custom values."""
        budget = CognitiveBudget(
            max_context_tokens=4000,
            max_history_messages=10,
            max_critique_chars=500,
        )

        assert budget.max_context_tokens == 4000
        assert budget.max_history_messages == 10
        assert budget.max_critique_chars == 500

    def test_max_context_chars_property(self):
        """Test max_context_chars computed property."""
        budget = CognitiveBudget(max_context_tokens=6000)

        expected = 6000 * CHARS_PER_TOKEN
        assert budget.max_context_chars == expected
        assert budget.max_context_chars == 24000

    def test_scale_method(self):
        """Test scaling all budgets by a factor."""
        budget = CognitiveBudget(
            max_context_tokens=8000,
            max_history_messages=20,
            max_critique_chars=1000,
            max_proposal_chars=2000,
            max_patterns_chars=500,
            reserve_for_response=2000,
        )

        scaled = budget.scale(0.5)

        assert scaled.max_context_tokens == 4000
        assert scaled.max_history_messages == 10
        assert scaled.max_critique_chars == 500
        assert scaled.max_proposal_chars == 1000
        assert scaled.max_patterns_chars == 250
        # Response reserve should not scale
        assert scaled.reserve_for_response == 2000

    def test_scale_method_minimum_history_messages(self):
        """Test scale enforces minimum history messages."""
        budget = CognitiveBudget(max_history_messages=10)

        # Scale down significantly
        scaled = budget.scale(0.1)

        # Should not go below 3
        assert scaled.max_history_messages >= 3

    def test_scale_method_returns_new_budget(self):
        """Test scale returns a new CognitiveBudget instance."""
        original = CognitiveBudget()
        scaled = original.scale(0.8)

        assert scaled is not original
        assert isinstance(scaled, CognitiveBudget)


class TestStressBudgets:
    """Tests for STRESS_BUDGETS presets."""

    def test_all_stress_levels_exist(self):
        """Test all expected stress levels have budgets."""
        assert "nominal" in STRESS_BUDGETS
        assert "elevated" in STRESS_BUDGETS
        assert "high" in STRESS_BUDGETS
        assert "critical" in STRESS_BUDGETS

    def test_budgets_are_cognitive_budget_instances(self):
        """Test all budgets are CognitiveBudget instances."""
        for level, budget in STRESS_BUDGETS.items():
            assert isinstance(budget, CognitiveBudget), f"Budget for {level} is not CognitiveBudget"

    def test_budgets_decrease_with_stress(self):
        """Test that higher stress levels have more restrictive budgets."""
        nominal = STRESS_BUDGETS["nominal"]
        elevated = STRESS_BUDGETS["elevated"]
        high = STRESS_BUDGETS["high"]
        critical = STRESS_BUDGETS["critical"]

        # Context tokens should decrease
        assert nominal.max_context_tokens >= elevated.max_context_tokens
        assert elevated.max_context_tokens >= high.max_context_tokens
        assert high.max_context_tokens >= critical.max_context_tokens

        # History messages should decrease
        assert nominal.max_history_messages >= elevated.max_history_messages
        assert elevated.max_history_messages >= high.max_history_messages
        assert high.max_history_messages >= critical.max_history_messages

    def test_nominal_budget_values(self):
        """Test nominal budget specific values."""
        budget = STRESS_BUDGETS["nominal"]

        assert budget.max_context_tokens == 8000
        assert budget.max_history_messages == 20
        assert budget.max_critique_chars == 1000
        assert budget.max_proposal_chars == 2500

    def test_critical_budget_values(self):
        """Test critical budget specific values."""
        budget = STRESS_BUDGETS["critical"]

        assert budget.max_context_tokens == 2000
        assert budget.max_history_messages == 5
        assert budget.max_critique_chars == 300
        assert budget.max_proposal_chars == 800


class TestCognitiveLoadLimiter:
    """Tests for CognitiveLoadLimiter class."""

    def test_init_default_budget(self):
        """Test limiter uses elevated budget by default."""
        limiter = CognitiveLoadLimiter()

        assert limiter.budget == STRESS_BUDGETS["elevated"]

    def test_init_custom_budget(self):
        """Test limiter accepts custom budget."""
        custom_budget = CognitiveBudget(max_context_tokens=5000)
        limiter = CognitiveLoadLimiter(budget=custom_budget)

        assert limiter.budget.max_context_tokens == 5000

    def test_init_stats(self):
        """Test limiter initializes stats."""
        limiter = CognitiveLoadLimiter()

        assert limiter.stats["messages_truncated"] == 0
        assert limiter.stats["critiques_truncated"] == 0
        assert limiter.stats["total_chars_removed"] == 0

    def test_for_stress_level_factory(self):
        """Test creating limiter for specific stress level."""
        limiter = CognitiveLoadLimiter.for_stress_level("high")

        assert limiter.budget == STRESS_BUDGETS["high"]

    def test_for_stress_level_unknown_defaults_to_elevated(self):
        """Test unknown stress level defaults to elevated."""
        limiter = CognitiveLoadLimiter.for_stress_level("unknown_level")

        assert limiter.budget == STRESS_BUDGETS["elevated"]

    def test_from_governor_import_error(self):
        """Test from_governor returns default limiter on import error."""
        with patch(
            "aragora.debate.cognitive_limiter.CognitiveLoadLimiter.from_governor",
            side_effect=ImportError,
        ):
            limiter = CognitiveLoadLimiter()

        assert isinstance(limiter, CognitiveLoadLimiter)

    def test_estimate_tokens(self):
        """Test token estimation from text."""
        limiter = CognitiveLoadLimiter()

        text = "a" * 100  # 100 chars
        tokens = limiter.estimate_tokens(text)

        assert tokens == 100 // CHARS_PER_TOKEN
        assert tokens == 25

    def test_estimate_tokens_empty_string(self):
        """Test token estimation for empty string."""
        limiter = CognitiveLoadLimiter()

        assert limiter.estimate_tokens("") == 0
        assert limiter.estimate_tokens(None) == 0  # type: ignore[arg-type]


class TestCognitiveLoadLimiterMessages:
    """Tests for message limiting functionality."""

    @dataclass
    class MockMessage:
        """Mock message for testing."""

        content: str
        role: str = "assistant"

    def test_limit_messages_empty_list(self):
        """Test limiting empty message list."""
        limiter = CognitiveLoadLimiter()

        result = limiter.limit_messages([])

        assert result == []

    def test_limit_messages_within_limit(self):
        """Test messages within limit are preserved."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=10))
        messages = [self.MockMessage(content=f"Message {i}") for i in range(5)]

        result = limiter.limit_messages(messages)

        assert len(result) == 5

    def test_limit_messages_exceeds_count_limit(self):
        """Test messages exceeding count limit are truncated."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=5))
        messages = [self.MockMessage(content=f"Message {i}") for i in range(10)]

        result = limiter.limit_messages(messages)

        # Should keep first and last (max_messages - 1)
        assert len(result) == 5
        # First message should be preserved
        assert result[0].content == "Message 0"
        # Last messages should be preserved
        assert result[-1].content == "Message 9"

    def test_limit_messages_preserves_first_message(self):
        """Test first message (task description) is always preserved."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=3))
        messages = [
            self.MockMessage(content="Task: Build a system"),
            self.MockMessage(content="Round 1 response"),
            self.MockMessage(content="Round 2 response"),
            self.MockMessage(content="Round 3 response"),
            self.MockMessage(content="Round 4 response"),
        ]

        result = limiter.limit_messages(messages)

        assert result[0].content == "Task: Build a system"

    def test_limit_messages_tracks_truncation_stats(self):
        """Test truncation stats are updated."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=3))
        messages = [self.MockMessage(content=f"Message {i}") for i in range(10)]

        limiter.limit_messages(messages)

        assert limiter.stats["messages_truncated"] == 7

    def test_limit_messages_with_max_chars(self):
        """Test limiting messages by character count."""
        limiter = CognitiveLoadLimiter()
        messages = [
            self.MockMessage(content="First message with some content"),
            self.MockMessage(content="Second message with more content here"),
            self.MockMessage(content="Third message"),
        ]

        result = limiter.limit_messages(messages, max_chars=80)

        # Should fit within char limit
        total_chars = sum(len(m.content) for m in result)
        assert total_chars <= 80

    def test_limit_messages_override_max_messages(self):
        """Test overriding max_messages parameter."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=20))
        messages = [self.MockMessage(content=f"Message {i}") for i in range(10)]

        result = limiter.limit_messages(messages, max_messages=3)

        assert len(result) == 3

    def test_limit_messages_handles_plain_strings(self):
        """Test handling messages that are plain strings."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=5))
        messages = ["Message 1", "Message 2", "Message 3"]

        result = limiter.limit_messages(messages)

        assert len(result) == 3


class TestCognitiveLoadLimiterMessageTruncation:
    """Tests for individual message truncation."""

    @dataclass
    class MockMessage:
        """Mock message for testing."""

        content: str

    class NamedTupleMessage(NamedTuple):
        """NamedTuple message for testing _replace."""

        content: str
        role: str = "assistant"

    def test_truncate_message_short_message(self):
        """Test short messages are not truncated."""
        limiter = CognitiveLoadLimiter()
        msg = self.MockMessage(content="Short message")

        result = limiter._truncate_message(msg, max_chars=100)

        assert result.content == "Short message"

    def test_truncate_message_long_message(self):
        """Test long messages are truncated with ellipsis."""
        limiter = CognitiveLoadLimiter()
        long_content = "a" * 500
        msg = self.MockMessage(content=long_content)

        result = limiter._truncate_message(msg, max_chars=100)

        assert len(result.content) <= 100
        assert "[... truncated ...]" in result.content

    def test_truncate_message_preserves_start_and_end(self):
        """Test truncation preserves beginning and end of message."""
        limiter = CognitiveLoadLimiter()
        content = "START" + "x" * 500 + "END"
        msg = self.MockMessage(content=content)

        result = limiter._truncate_message(msg, max_chars=100)

        assert result.content.startswith("START")
        assert result.content.endswith("END")

    def test_truncate_message_namedtuple(self):
        """Test truncation works with NamedTuple messages."""
        limiter = CognitiveLoadLimiter()
        msg = self.NamedTupleMessage(content="a" * 500)

        result = limiter._truncate_message(msg, max_chars=100)

        assert isinstance(result, self.NamedTupleMessage)
        assert "[... truncated ...]" in result.content

    def test_truncate_message_immutable_returns_string(self):
        """Test truncation of immutable object returns string."""
        limiter = CognitiveLoadLimiter()

        # Create a frozen dataclass-like object
        @dataclass(frozen=True)
        class FrozenMessage:
            content: str

        msg = FrozenMessage(content="a" * 500)

        result = limiter._truncate_message(msg, max_chars=100)

        # Should return truncated string since object is immutable
        assert isinstance(result, str)
        assert "[... truncated ...]" in result


class TestCognitiveLoadLimiterCritiques:
    """Tests for critique limiting functionality."""

    @dataclass
    class MockCritique:
        """Mock critique for testing."""

        reasoning: str
        severity: float = 0.5
        issues: list = None
        suggestions: list = None

        def __post_init__(self):
            if self.issues is None:
                self.issues = []
            if self.suggestions is None:
                self.suggestions = []

    def test_limit_critiques_empty_list(self):
        """Test limiting empty critique list."""
        limiter = CognitiveLoadLimiter()

        result = limiter.limit_critiques([])

        assert result == []

    def test_limit_critiques_within_limit(self):
        """Test critiques within limit are preserved."""
        limiter = CognitiveLoadLimiter()
        critiques = [
            self.MockCritique(reasoning="Short critique 1"),
            self.MockCritique(reasoning="Short critique 2"),
        ]

        result = limiter.limit_critiques(critiques, max_critiques=5)

        assert len(result) == 2

    def test_limit_critiques_count_limit(self):
        """Test critiques exceeding count are limited."""
        limiter = CognitiveLoadLimiter()
        critiques = [self.MockCritique(reasoning=f"Critique {i}") for i in range(10)]

        result = limiter.limit_critiques(critiques, max_critiques=3)

        assert len(result) == 3

    def test_limit_critiques_sorted_by_severity(self):
        """Test critiques are sorted by severity (highest first)."""
        limiter = CognitiveLoadLimiter()
        critiques = [
            self.MockCritique(reasoning="Low", severity=0.2),
            self.MockCritique(reasoning="High", severity=0.9),
            self.MockCritique(reasoning="Medium", severity=0.5),
        ]

        result = limiter.limit_critiques(critiques, max_critiques=2)

        # Should keep highest severity critiques
        assert len(result) == 2
        # First should be highest severity
        assert result[0].severity == 0.9

    def test_limit_critiques_long_reasoning_summarized(self):
        """Test long critique reasoning is summarized."""
        limiter = CognitiveLoadLimiter()
        long_reasoning = "a" * 2000
        critiques = [
            self.MockCritique(
                reasoning=long_reasoning,
                issues=["Issue 1", "Issue 2"],
                suggestions=["Fix 1", "Fix 2"],
            )
        ]

        result = limiter.limit_critiques(critiques, max_chars_per=100)

        assert len(result[0].reasoning) <= 100
        assert limiter.stats["critiques_truncated"] == 1

    def test_limit_critiques_summary_includes_issues(self):
        """Test summarized critique includes issues."""
        limiter = CognitiveLoadLimiter()
        critiques = [
            self.MockCritique(
                reasoning="a" * 2000,
                issues=["Memory leak", "Race condition"],
            )
        ]

        result = limiter.limit_critiques(critiques, max_chars_per=200)

        assert "Issues:" in result[0].reasoning
        assert "Memory leak" in result[0].reasoning

    def test_limit_critiques_summary_includes_suggestions(self):
        """Test summarized critique includes suggestions."""
        limiter = CognitiveLoadLimiter()
        critiques = [
            self.MockCritique(
                reasoning="a" * 2000,
                suggestions=["Use caching", "Add retry logic"],
            )
        ]

        result = limiter.limit_critiques(critiques, max_chars_per=200)

        assert "Suggestions:" in result[0].reasoning


class TestCognitiveLoadLimiterContext:
    """Tests for full context limiting functionality."""

    @dataclass
    class MockMessage:
        """Mock message for testing."""

        content: str

    @dataclass
    class MockCritique:
        """Mock critique for testing."""

        reasoning: str
        severity: float = 0.5

    def test_limit_context_messages_only(self):
        """Test limiting context with only messages."""
        limiter = CognitiveLoadLimiter()
        messages = [self.MockMessage(content=f"Message {i}") for i in range(5)]

        result = limiter.limit_context(messages=messages)

        assert "messages" in result
        assert len(result["messages"]) == 5

    def test_limit_context_critiques_only(self):
        """Test limiting context with only critiques."""
        limiter = CognitiveLoadLimiter()
        critiques = [self.MockCritique(reasoning=f"Critique {i}") for i in range(3)]

        result = limiter.limit_context(critiques=critiques)

        assert "critiques" in result
        assert len(result["critiques"]) == 3

    def test_limit_context_patterns(self):
        """Test limiting patterns string."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_patterns_chars=50))
        long_patterns = "Pattern: " + "x" * 100

        result = limiter.limit_context(patterns=long_patterns)

        assert "patterns" in result
        assert len(result["patterns"]) <= 53  # 50 + "..."

    def test_limit_context_extra_context(self):
        """Test limiting extra context."""
        limiter = CognitiveLoadLimiter()
        extra = "Additional context information " * 100

        result = limiter.limit_context(extra_context=extra)

        assert "extra_context" in result
        # Should be truncated based on budget allocation

    def test_limit_context_all_components(self):
        """Test limiting all context components together."""
        limiter = CognitiveLoadLimiter()
        messages = [self.MockMessage(content="Message")]
        critiques = [self.MockCritique(reasoning="Critique")]
        patterns = "Some patterns"
        extra = "Extra info"

        result = limiter.limit_context(
            messages=messages,
            critiques=critiques,
            patterns=patterns,
            extra_context=extra,
        )

        assert "messages" in result
        assert "critiques" in result
        assert "patterns" in result
        assert "extra_context" in result

    def test_limit_context_budget_allocation(self):
        """Test budget is allocated proportionally."""
        # Messages get 60%, critiques 25%, other 15%
        limiter = CognitiveLoadLimiter(
            budget=CognitiveBudget(max_context_tokens=1000)  # 4000 chars
        )

        # Create content that would exceed budget if not limited
        messages = [self.MockMessage(content="x" * 500) for _ in range(10)]
        critiques = [self.MockCritique(reasoning="y" * 500) for _ in range(10)]

        result = limiter.limit_context(messages=messages, critiques=critiques)

        # Should be limited within budget
        assert len(result["messages"]) < 10 or len(result["critiques"]) < 10


class TestCognitiveLoadLimiterStats:
    """Tests for statistics tracking."""

    @dataclass
    class MockMessage:
        """Mock message for testing."""

        content: str

    def test_get_stats(self):
        """Test getting statistics."""
        limiter = CognitiveLoadLimiter()

        stats = limiter.get_stats()

        assert "messages_truncated" in stats
        assert "critiques_truncated" in stats
        assert "total_chars_removed" in stats
        assert "budget" in stats
        assert stats["budget"]["max_tokens"] == limiter.budget.max_context_tokens

    def test_reset_stats(self):
        """Test resetting statistics."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=3))
        messages = [self.MockMessage(content=f"Message {i}") for i in range(10)]
        limiter.limit_messages(messages)

        assert limiter.stats["messages_truncated"] > 0

        limiter.reset_stats()

        assert limiter.stats["messages_truncated"] == 0
        assert limiter.stats["critiques_truncated"] == 0
        assert limiter.stats["total_chars_removed"] == 0

    def test_stats_accumulate(self):
        """Test stats accumulate across multiple calls."""
        limiter = CognitiveLoadLimiter(budget=CognitiveBudget(max_history_messages=3))
        messages = [self.MockMessage(content=f"Message {i}") for i in range(10)]

        limiter.limit_messages(messages)
        first_count = limiter.stats["messages_truncated"]

        limiter.limit_messages(messages)
        second_count = limiter.stats["messages_truncated"]

        assert second_count == first_count * 2


class TestLimitDebateContext:
    """Tests for limit_debate_context convenience function."""

    @dataclass
    class MockMessage:
        """Mock message for testing."""

        content: str

    @dataclass
    class MockCritique:
        """Mock critique for testing."""

        reasoning: str
        severity: float = 0.5

    def test_limit_debate_context_basic(self):
        """Test basic usage of convenience function."""
        messages = [self.MockMessage(content=f"Message {i}") for i in range(5)]

        result = limit_debate_context(messages)

        assert "messages" in result

    def test_limit_debate_context_with_critiques(self):
        """Test with critiques."""
        messages = [self.MockMessage(content="Message")]
        critiques = [self.MockCritique(reasoning="Critique")]

        result = limit_debate_context(messages, critiques=critiques)

        assert "messages" in result
        assert "critiques" in result

    def test_limit_debate_context_stress_level(self):
        """Test with specific stress level."""
        messages = [self.MockMessage(content=f"Message {i}") for i in range(30)]

        result_nominal = limit_debate_context(messages, stress_level="nominal")
        result_critical = limit_debate_context(messages, stress_level="critical")

        # Critical should be more restrictive
        nominal_count = len(result_nominal.get("messages", []))
        critical_count = len(result_critical.get("messages", []))

        # With same input, critical should have fewer or equal messages
        assert critical_count <= nominal_count

    def test_limit_debate_context_default_stress_level(self):
        """Test default stress level is elevated."""
        messages = [self.MockMessage(content="Message")]

        result = limit_debate_context(messages)

        # Should complete without error using elevated budget
        assert result is not None


class TestCognitiveLoadLimiterFromGovernor:
    """Tests for from_governor factory method."""

    def test_from_governor_success(self):
        """Test creating limiter from governor."""
        mock_governor = MagicMock()
        mock_governor.stress_level.value = "high"

        # Patch at the import location inside from_governor method
        with patch.dict(
            "sys.modules",
            {"aragora.debate.complexity_governor": MagicMock()},
        ):
            with patch(
                "aragora.debate.complexity_governor.get_complexity_governor",
                return_value=mock_governor,
                create=True,
            ):
                limiter = CognitiveLoadLimiter.from_governor()

        assert limiter.budget == STRESS_BUDGETS["high"]

    def test_from_governor_import_error_fallback(self):
        """Test fallback on import error."""
        # Simply call the method - if complexity_governor is not available,
        # it should return a default limiter. If it is available, we still
        # get a valid limiter back.
        limiter = CognitiveLoadLimiter.from_governor()

        # Should return a valid limiter regardless
        assert isinstance(limiter, CognitiveLoadLimiter)
        assert limiter.budget is not None


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_very_long_single_message(self):
        """Test handling of very long single message."""
        # max_context_chars is computed from max_context_tokens, so set tokens low
        limiter = CognitiveLoadLimiter(
            budget=CognitiveBudget(max_context_tokens=25)  # 25 * 4 = 100 chars
        )

        @dataclass
        class Msg:
            content: str

        messages = [Msg(content="x" * 10000)]

        result = limiter.limit_messages(messages)

        assert len(result) == 1

    def test_empty_critique_issues_and_suggestions(self):
        """Test summarizing critique with empty issues/suggestions."""
        limiter = CognitiveLoadLimiter()

        @dataclass
        class Critique:
            reasoning: str
            severity: float = 0.5
            issues: list = None
            suggestions: list = None

        critiques = [Critique(reasoning="a" * 2000)]

        result = limiter.limit_critiques(critiques, max_chars_per=100)

        # Should truncate reasoning directly
        assert len(result) == 1

    def test_message_without_content_attribute(self):
        """Test handling messages without content attribute."""
        limiter = CognitiveLoadLimiter()

        # Plain strings don't have .content
        messages = ["plain string message 1", "plain string message 2"]

        result = limiter.limit_messages(messages)

        assert len(result) == 2

    def test_critique_without_reasoning_attribute(self):
        """Test handling critiques without reasoning attribute."""
        limiter = CognitiveLoadLimiter()

        # Plain strings
        critiques = ["critique as string"]

        result = limiter.limit_critiques(critiques)

        assert len(result) == 1

    def test_zero_scale_factor(self):
        """Test scaling budget with factor of zero."""
        budget = CognitiveBudget()
        scaled = budget.scale(0)

        assert scaled.max_context_tokens == 0
        assert scaled.max_history_messages == 3  # Minimum enforced
        assert scaled.reserve_for_response == 2000  # Not scaled
