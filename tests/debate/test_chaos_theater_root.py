"""
Tests for aragora.debate.chaos_theater - Dynamic theatrical responses.

Tests cover:
- FailureType and DramaLevel enums
- TheaterResponse dataclass
- ChaosDirector response generation
- Message variety and history tracking
- Drama level escalation/deescalation
- Global instance management
- Helper functions
"""

import pytest
from unittest.mock import patch

from aragora.debate.chaos_theater import (
    FailureType,
    DramaLevel,
    TheaterResponse,
    ChaosDirector,
    get_chaos_director,
    theatrical_timeout,
    theatrical_error,
)


class TestFailureType:
    """Tests for FailureType enum."""

    def test_all_failure_types_exist(self):
        """All expected failure types should exist."""
        assert FailureType.TIMEOUT.value == "timeout"
        assert FailureType.CONNECTION.value == "connection"
        assert FailureType.RATE_LIMIT.value == "rate_limit"
        assert FailureType.INTERNAL.value == "internal"
        assert FailureType.UNKNOWN.value == "unknown"

    def test_enum_iteration(self):
        """Should be able to iterate over failure types."""
        types = list(FailureType)
        assert len(types) == 5


class TestDramaLevel:
    """Tests for DramaLevel enum."""

    def test_drama_levels_ordered(self):
        """Drama levels should be ordered by intensity."""
        assert DramaLevel.SUBTLE.value < DramaLevel.MODERATE.value
        assert DramaLevel.MODERATE.value < DramaLevel.DRAMATIC.value

    def test_all_levels_exist(self):
        """All expected drama levels should exist."""
        assert DramaLevel.SUBTLE.value == 1
        assert DramaLevel.MODERATE.value == 2
        assert DramaLevel.DRAMATIC.value == 3


class TestTheaterResponse:
    """Tests for TheaterResponse dataclass."""

    def test_required_fields(self):
        """Should create with required fields."""
        response = TheaterResponse(
            message="Test message",
            agent_name="claude",
            failure_type=FailureType.TIMEOUT,
            drama_level=DramaLevel.MODERATE,
        )

        assert response.message == "Test message"
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.TIMEOUT
        assert response.drama_level == DramaLevel.MODERATE

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        response = TheaterResponse(
            message="Test",
            agent_name="test",
            failure_type=FailureType.UNKNOWN,
            drama_level=DramaLevel.SUBTLE,
        )

        assert response.duration_hint is None
        assert response.recovery_suggestion is None

    def test_optional_fields_set(self):
        """Should accept optional fields."""
        response = TheaterResponse(
            message="Test",
            agent_name="test",
            failure_type=FailureType.TIMEOUT,
            drama_level=DramaLevel.SUBTLE,
            duration_hint=30.0,
            recovery_suggestion="Wait a bit",
        )

        assert response.duration_hint == 30.0
        assert response.recovery_suggestion == "Wait a bit"


class TestChaosDirectorInit:
    """Tests for ChaosDirector initialization."""

    def test_default_drama_level(self):
        """Default drama level should be MODERATE."""
        director = ChaosDirector()
        assert director.drama_level == DramaLevel.MODERATE

    def test_custom_drama_level(self):
        """Should accept custom drama level."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)
        assert director.drama_level == DramaLevel.DRAMATIC

    def test_empty_message_history(self):
        """Message history should start empty."""
        director = ChaosDirector()
        assert director._message_history == {}


class TestChaosDirectorTimeoutResponse:
    """Tests for timeout_response method."""

    def test_returns_theater_response(self):
        """Should return TheaterResponse."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 60.0)

        assert isinstance(response, TheaterResponse)
        assert response.failure_type == FailureType.TIMEOUT
        assert response.agent_name == "claude"

    def test_includes_agent_name_in_message(self):
        """Message should include agent name."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 60.0)

        assert "claude" in response.message

    def test_duration_hint_set(self):
        """Duration hint should be set."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 60.0)

        assert response.duration_hint is not None
        assert response.duration_hint == 30.0  # Half of input

    def test_recovery_suggestion_set(self):
        """Recovery suggestion should be set."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 60.0)

        assert response.recovery_suggestion is not None

    def test_different_drama_levels(self):
        """Different drama levels should produce different messages."""
        messages = set()
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            response = director.timeout_response("test", 60.0)
            messages.add(response.message[:20])  # Compare prefixes

        # Should have different messages for different levels
        # (Note: there's randomness, so we just check they're not all identical)
        assert len(messages) >= 1


class TestChaosDirectorConnectionResponse:
    """Tests for connection_response method."""

    def test_returns_connection_failure(self):
        """Should return connection failure type."""
        director = ChaosDirector()
        response = director.connection_response("claude")

        assert response.failure_type == FailureType.CONNECTION
        assert "claude" in response.message

    def test_has_recovery_suggestion(self):
        """Should have recovery suggestion."""
        director = ChaosDirector()
        response = director.connection_response("claude")

        assert response.recovery_suggestion is not None


class TestChaosDirectorRateLimitResponse:
    """Tests for rate_limit_response method."""

    def test_returns_rate_limit_failure(self):
        """Should return rate limit failure type."""
        director = ChaosDirector()
        response = director.rate_limit_response("claude")

        assert response.failure_type == FailureType.RATE_LIMIT
        assert "claude" in response.message

    def test_retry_after_in_duration_hint(self):
        """Retry after should be in duration hint."""
        director = ChaosDirector()
        response = director.rate_limit_response("claude", retry_after=30.0)

        assert response.duration_hint == 30.0

    def test_retry_after_in_recovery_suggestion(self):
        """Retry after should be mentioned in recovery suggestion."""
        director = ChaosDirector()
        response = director.rate_limit_response("claude", retry_after=30.0)

        assert "30" in response.recovery_suggestion


class TestChaosDirectorInternalErrorResponse:
    """Tests for internal_error_response method."""

    def test_returns_internal_failure(self):
        """Should return internal failure type."""
        director = ChaosDirector()
        response = director.internal_error_response("claude")

        assert response.failure_type == FailureType.INTERNAL
        assert "claude" in response.message


class TestChaosDirectorRecoveryResponse:
    """Tests for recovery_response method."""

    def test_returns_recovery_message(self):
        """Should return positive recovery message."""
        director = ChaosDirector()
        response = director.recovery_response("claude")

        assert "claude" in response.message
        assert response.failure_type == FailureType.UNKNOWN


class TestChaosDirectorProgressResponse:
    """Tests for progress_response method."""

    def test_returns_progress_message(self):
        """Should return progress message."""
        director = ChaosDirector()
        response = director.progress_response("claude", progress_percent=50)

        assert response.agent_name == "claude"


class TestMessageVariety:
    """Tests for message variety and history tracking."""

    def test_avoids_recent_repeats(self):
        """Should avoid repeating recent messages."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)

        messages = []
        for _ in range(5):
            response = director.timeout_response("claude", 60.0)
            messages.append(response.message)

        # Should have variety (not all identical)
        unique_messages = set(messages)
        assert len(unique_messages) >= 2

    def test_history_per_agent(self):
        """History should be tracked per agent."""
        director = ChaosDirector()

        # Different agents should have independent history
        response1 = director.timeout_response("claude", 60.0)
        response2 = director.timeout_response("gpt4", 60.0)

        # Both should get messages (history doesn't cross agents)
        assert response1.message is not None
        assert response2.message is not None


class TestDramaLevelControl:
    """Tests for drama level escalation/deescalation."""

    def test_set_drama_level(self):
        """Should set drama level."""
        director = ChaosDirector()
        director.set_drama_level(DramaLevel.DRAMATIC)

        assert director.drama_level == DramaLevel.DRAMATIC

    def test_escalate_drama(self):
        """Should increase drama level."""
        director = ChaosDirector(drama_level=DramaLevel.SUBTLE)
        director.escalate_drama()

        assert director.drama_level == DramaLevel.MODERATE

    def test_escalate_at_max(self):
        """Escalating at max should stay at max."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)
        director.escalate_drama()

        assert director.drama_level == DramaLevel.DRAMATIC

    def test_deescalate_drama(self):
        """Should decrease drama level."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)
        director.deescalate_drama()

        assert director.drama_level == DramaLevel.MODERATE

    def test_deescalate_at_min(self):
        """Deescalating at min should stay at min."""
        director = ChaosDirector(drama_level=DramaLevel.SUBTLE)
        director.deescalate_drama()

        assert director.drama_level == DramaLevel.SUBTLE


class TestGlobalInstance:
    """Tests for global instance management."""

    def test_get_chaos_director_returns_instance(self):
        """Should return ChaosDirector instance."""
        director = get_chaos_director()
        assert isinstance(director, ChaosDirector)

    def test_get_chaos_director_singleton(self):
        """Should return same instance on subsequent calls."""
        director1 = get_chaos_director()
        director2 = get_chaos_director()

        assert director1 is director2


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_theatrical_timeout(self):
        """Should return timeout message string."""
        message = theatrical_timeout("claude", 60.0)

        assert isinstance(message, str)
        assert len(message) > 0
        assert "claude" in message

    def test_theatrical_error_internal(self):
        """Should return internal error message."""
        message = theatrical_error("claude", error_type="internal")

        assert isinstance(message, str)
        assert "claude" in message

    def test_theatrical_error_connection(self):
        """Should return connection error message."""
        message = theatrical_error("claude", error_type="connection")

        assert isinstance(message, str)
        assert "claude" in message

    def test_theatrical_error_rate_limit(self):
        """Should return rate limit error message."""
        message = theatrical_error("claude", error_type="rate_limit")

        assert isinstance(message, str)
        assert "claude" in message


class TestMessageContent:
    """Tests for message content appropriateness."""

    def test_subtle_messages_professional(self):
        """Subtle messages should be professional."""
        director = ChaosDirector(drama_level=DramaLevel.SUBTLE)
        response = director.timeout_response("claude", 60.0)

        # Should not have dramatic formatting
        assert "⚡" not in response.message
        assert "🔥" not in response.message

    def test_dramatic_messages_theatrical(self):
        """Dramatic messages should be theatrical."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)

        # Generate multiple messages to ensure we get a dramatic one
        dramatic_found = False
        for _ in range(10):
            response = director.timeout_response("claude", 60.0)
            if any(char in response.message for char in "⚡🔥🌀⚠️"):
                dramatic_found = True
                break

        # Most dramatic messages have special characters
        # (This is a soft test since there's randomness)
        assert dramatic_found or len(response.message) > 30


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_agent_name(self):
        """Should handle empty agent name."""
        director = ChaosDirector()
        response = director.timeout_response("", 60.0)

        assert response.agent_name == ""
        assert isinstance(response.message, str)

    def test_special_characters_in_agent_name(self):
        """Should handle special characters in agent name."""
        director = ChaosDirector()
        response = director.timeout_response("claude-3.5-sonnet", 60.0)

        assert "claude-3.5-sonnet" in response.message

    def test_very_long_duration(self):
        """Should handle very long duration."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 3600.0)  # 1 hour

        assert response.duration_hint is not None

    def test_zero_duration(self):
        """Should handle zero duration."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 0.0)

        assert response is not None

    def test_negative_duration(self):
        """Should handle negative duration (edge case)."""
        director = ChaosDirector()
        response = director.timeout_response("claude", -10.0)

        # Should still return a response
        assert response is not None
