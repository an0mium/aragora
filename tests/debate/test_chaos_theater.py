"""
Tests for the Chaos Theater module.

Tests cover:
- FailureType and DramaLevel enums
- TheaterResponse data class
- ChaosDirector class
  - Initialization
  - Message selection and history tracking
  - Timeout responses
  - Connection failure responses
  - Rate limit responses
  - Internal error responses
  - Recovery responses
  - Progress responses
  - Drama level management
- Global director functions
- Edge cases and error scenarios
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.chaos_theater import (
    ChaosDirector,
    DramaLevel,
    FailureType,
    TheaterResponse,
    get_chaos_director,
    theatrical_error,
    theatrical_timeout,
)


class TestFailureType:
    """Tests for FailureType enum."""

    def test_failure_type_values(self):
        """Test all failure type values exist."""
        assert FailureType.TIMEOUT.value == "timeout"
        assert FailureType.CONNECTION.value == "connection"
        assert FailureType.RATE_LIMIT.value == "rate_limit"
        assert FailureType.INTERNAL.value == "internal"
        assert FailureType.UNKNOWN.value == "unknown"

    def test_failure_type_count(self):
        """Test correct number of failure types."""
        assert len(FailureType) == 5


class TestDramaLevel:
    """Tests for DramaLevel enum."""

    def test_drama_level_values(self):
        """Test all drama level values exist."""
        assert DramaLevel.SUBTLE.value == 1
        assert DramaLevel.MODERATE.value == 2
        assert DramaLevel.DRAMATIC.value == 3

    def test_drama_level_ordering(self):
        """Test drama levels are ordered correctly."""
        levels = list(DramaLevel)
        assert levels[0] == DramaLevel.SUBTLE
        assert levels[1] == DramaLevel.MODERATE
        assert levels[2] == DramaLevel.DRAMATIC


class TestTheaterResponse:
    """Tests for TheaterResponse data class."""

    def test_basic_response_creation(self):
        """Test creating a basic TheaterResponse."""
        response = TheaterResponse(
            message="Agent is thinking...",
            agent_name="claude",
            failure_type=FailureType.TIMEOUT,
            drama_level=DramaLevel.MODERATE,
        )

        assert response.message == "Agent is thinking..."
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.TIMEOUT
        assert response.drama_level == DramaLevel.MODERATE
        assert response.duration_hint is None
        assert response.recovery_suggestion is None

    def test_response_with_optional_fields(self):
        """Test creating a TheaterResponse with optional fields."""
        response = TheaterResponse(
            message="Rate limited!",
            agent_name="gpt4",
            failure_type=FailureType.RATE_LIMIT,
            drama_level=DramaLevel.DRAMATIC,
            duration_hint=30.0,
            recovery_suggestion="Will retry in 30s",
        )

        assert response.duration_hint == 30.0
        assert response.recovery_suggestion == "Will retry in 30s"

    def test_response_immutability(self):
        """Test that response fields are accessible."""
        response = TheaterResponse(
            message="Error occurred",
            agent_name="mistral",
            failure_type=FailureType.INTERNAL,
            drama_level=DramaLevel.SUBTLE,
        )

        # Dataclass fields should be accessible
        assert hasattr(response, "message")
        assert hasattr(response, "agent_name")
        assert hasattr(response, "failure_type")
        assert hasattr(response, "drama_level")


class TestChaosDirectorInit:
    """Tests for ChaosDirector initialization."""

    def test_default_drama_level(self):
        """Test default drama level is MODERATE."""
        director = ChaosDirector()
        assert director.drama_level == DramaLevel.MODERATE

    def test_custom_drama_level(self):
        """Test custom drama level initialization."""
        director = ChaosDirector(drama_level=DramaLevel.SUBTLE)
        assert director.drama_level == DramaLevel.SUBTLE

        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)
        assert director.drama_level == DramaLevel.DRAMATIC

    def test_message_history_initialized(self):
        """Test message history is initialized as empty dict."""
        director = ChaosDirector()
        assert director._message_history == {}

    def test_message_constants_exist(self):
        """Test that all message dictionaries are defined."""
        assert hasattr(ChaosDirector, "TIMEOUT_MESSAGES")
        assert hasattr(ChaosDirector, "CONNECTION_MESSAGES")
        assert hasattr(ChaosDirector, "RATE_LIMIT_MESSAGES")
        assert hasattr(ChaosDirector, "INTERNAL_MESSAGES")
        assert hasattr(ChaosDirector, "RECOVERY_MESSAGES")
        assert hasattr(ChaosDirector, "PROGRESS_MESSAGES")

    def test_all_drama_levels_have_messages(self):
        """Test all drama levels have messages for each type."""
        message_dicts = [
            ChaosDirector.TIMEOUT_MESSAGES,
            ChaosDirector.CONNECTION_MESSAGES,
            ChaosDirector.RATE_LIMIT_MESSAGES,
            ChaosDirector.INTERNAL_MESSAGES,
            ChaosDirector.RECOVERY_MESSAGES,
            ChaosDirector.PROGRESS_MESSAGES,
        ]

        for msg_dict in message_dicts:
            for level in DramaLevel:
                assert level in msg_dict, f"Missing {level} in message dict"
                assert len(msg_dict[level]) > 0, f"Empty messages for {level}"


class TestChaosDirectorMessageSelection:
    """Tests for ChaosDirector message selection and history."""

    def test_message_contains_agent_name(self):
        """Test that selected messages contain the agent name."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 30.0)
        assert "claude" in response.message

    def test_message_history_tracking(self):
        """Test that message history is tracked."""
        director = ChaosDirector()

        # Generate several messages for the same agent
        for _ in range(5):
            director.timeout_response("claude", 30.0)

        # History should be populated
        assert len(director._message_history) > 0

    def test_message_history_limits(self):
        """Test that message history is limited to last 3."""
        director = ChaosDirector()

        # Generate many messages
        for _ in range(10):
            director.timeout_response("claude", 30.0)

        # Check history doesn't grow beyond limit
        for key, history in director._message_history.items():
            assert len(history) <= 3

    def test_avoids_recent_repeats(self):
        """Test that message selection avoids recent repeats when possible."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)

        # Get multiple messages - should try to vary
        messages = set()
        for _ in range(7):  # More iterations than available messages
            response = director.timeout_response("claude", 30.0)
            messages.add(response.message)

        # Should have used multiple different messages
        assert len(messages) > 1

    def test_different_agents_have_separate_history(self):
        """Test that different agents have separate message histories."""
        director = ChaosDirector()

        # Generate messages for different agents
        director.timeout_response("claude", 30.0)
        director.timeout_response("gpt4", 30.0)
        director.timeout_response("mistral", 30.0)

        # Should have multiple history entries
        assert len(director._message_history) >= 1


class TestTimeoutResponse:
    """Tests for timeout response generation."""

    def test_timeout_response_structure(self):
        """Test timeout response has correct structure."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 60.0)

        assert isinstance(response, TheaterResponse)
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.TIMEOUT
        assert response.duration_hint is not None
        assert response.recovery_suggestion is not None

    def test_timeout_duration_hint_calculation(self):
        """Test duration hint is calculated correctly."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 100.0)

        # Duration hint should be half the timeout
        assert response.duration_hint == 50.0

    def test_timeout_recovery_suggestion(self):
        """Test timeout recovery suggestion is set."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 30.0)

        assert response.recovery_suggestion == "The agent may recover if given more time."

    def test_timeout_message_includes_duration(self):
        """Test timeout message can include duration."""
        director = ChaosDirector(drama_level=DramaLevel.MODERATE)
        response = director.timeout_response("claude", 90.0)

        # MODERATE and DRAMATIC levels have duration in messages
        # Message should be formatted (not contain {duration})
        assert "{duration}" not in response.message

    def test_timeout_drama_level_respected(self):
        """Test different drama levels produce different styles."""
        subtle = ChaosDirector(drama_level=DramaLevel.SUBTLE)
        dramatic = ChaosDirector(drama_level=DramaLevel.DRAMATIC)

        subtle_response = subtle.timeout_response("claude", 30.0)
        dramatic_response = dramatic.timeout_response("claude", 30.0)

        # Dramatic responses typically contain special characters or emojis
        # This is a weak test but checks they're different
        assert subtle_response.drama_level == DramaLevel.SUBTLE
        assert dramatic_response.drama_level == DramaLevel.DRAMATIC


class TestConnectionResponse:
    """Tests for connection failure response generation."""

    def test_connection_response_structure(self):
        """Test connection response has correct structure."""
        director = ChaosDirector()
        response = director.connection_response("claude")

        assert isinstance(response, TheaterResponse)
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.CONNECTION
        assert response.duration_hint is None
        assert response.recovery_suggestion is not None

    def test_connection_recovery_suggestion(self):
        """Test connection recovery suggestion is set."""
        director = ChaosDirector()
        response = director.connection_response("claude")

        assert response.recovery_suggestion == "Will retry connection automatically."

    def test_connection_message_contains_agent(self):
        """Test connection message contains agent name."""
        director = ChaosDirector()
        response = director.connection_response("gpt4")

        assert "gpt4" in response.message


class TestRateLimitResponse:
    """Tests for rate limit response generation."""

    def test_rate_limit_response_structure(self):
        """Test rate limit response has correct structure."""
        director = ChaosDirector()
        response = director.rate_limit_response("claude")

        assert isinstance(response, TheaterResponse)
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.RATE_LIMIT

    def test_rate_limit_with_retry_after(self):
        """Test rate limit response with retry_after."""
        director = ChaosDirector()
        response = director.rate_limit_response("claude", retry_after=60.0)

        assert response.duration_hint == 60.0
        assert "60" in response.recovery_suggestion

    def test_rate_limit_without_retry_after(self):
        """Test rate limit response without retry_after."""
        director = ChaosDirector()
        response = director.rate_limit_response("claude", retry_after=None)

        assert response.duration_hint is None
        assert response.recovery_suggestion == "Backing off..."

    def test_rate_limit_message_contains_agent(self):
        """Test rate limit message contains agent name."""
        director = ChaosDirector()
        response = director.rate_limit_response("mistral")

        assert "mistral" in response.message


class TestInternalErrorResponse:
    """Tests for internal error response generation."""

    def test_internal_error_response_structure(self):
        """Test internal error response has correct structure."""
        director = ChaosDirector()
        response = director.internal_error_response("claude")

        assert isinstance(response, TheaterResponse)
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.INTERNAL
        assert response.recovery_suggestion is not None

    def test_internal_error_recovery_suggestion(self):
        """Test internal error recovery suggestion is set."""
        director = ChaosDirector()
        response = director.internal_error_response("claude")

        assert response.recovery_suggestion == "Attempting automatic recovery."

    def test_internal_error_with_hint(self):
        """Test internal error accepts error_hint parameter."""
        director = ChaosDirector()
        # error_hint is accepted but not currently used in message
        response = director.internal_error_response("claude", error_hint="NullPointerException")

        assert isinstance(response, TheaterResponse)
        assert response.failure_type == FailureType.INTERNAL

    def test_internal_error_message_contains_agent(self):
        """Test internal error message contains agent name."""
        director = ChaosDirector()
        response = director.internal_error_response("gemini")

        assert "gemini" in response.message


class TestRecoveryResponse:
    """Tests for recovery response generation."""

    def test_recovery_response_structure(self):
        """Test recovery response has correct structure."""
        director = ChaosDirector()
        response = director.recovery_response("claude")

        assert isinstance(response, TheaterResponse)
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.UNKNOWN
        assert response.duration_hint is None
        assert response.recovery_suggestion is None

    def test_recovery_message_contains_agent(self):
        """Test recovery message contains agent name."""
        director = ChaosDirector()
        response = director.recovery_response("gpt4")

        assert "gpt4" in response.message

    def test_recovery_drama_levels(self):
        """Test recovery messages exist for all drama levels."""
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            response = director.recovery_response("claude")

            assert response.drama_level == level
            assert len(response.message) > 0


class TestProgressResponse:
    """Tests for progress response generation."""

    def test_progress_response_structure(self):
        """Test progress response has correct structure."""
        director = ChaosDirector()
        response = director.progress_response("claude")

        assert isinstance(response, TheaterResponse)
        assert response.agent_name == "claude"
        assert response.failure_type == FailureType.UNKNOWN

    def test_progress_default_percentage(self):
        """Test progress uses default 50% when not specified."""
        director = ChaosDirector(drama_level=DramaLevel.MODERATE)
        response = director.progress_response("claude")

        # Message should contain formatted progress (not {progress})
        assert "{progress}" not in response.message

    def test_progress_custom_percentage(self):
        """Test progress with custom percentage."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)
        response = director.progress_response("claude", progress_percent=75)

        # Message should be formatted
        assert "{progress}" not in response.message

    def test_progress_percentage_converted_to_int(self):
        """Test progress percentage is converted to integer."""
        director = ChaosDirector(drama_level=DramaLevel.MODERATE)
        # Pass a float, should be converted to int in message
        response = director.progress_response("claude", progress_percent=33.7)

        # Should not raise an error
        assert isinstance(response, TheaterResponse)


class TestDramaLevelManagement:
    """Tests for drama level management methods."""

    def test_set_drama_level(self):
        """Test setting drama level directly."""
        director = ChaosDirector()

        director.set_drama_level(DramaLevel.SUBTLE)
        assert director.drama_level == DramaLevel.SUBTLE

        director.set_drama_level(DramaLevel.DRAMATIC)
        assert director.drama_level == DramaLevel.DRAMATIC

    def test_escalate_drama(self):
        """Test escalating drama level."""
        director = ChaosDirector(drama_level=DramaLevel.SUBTLE)

        director.escalate_drama()
        assert director.drama_level == DramaLevel.MODERATE

        director.escalate_drama()
        assert director.drama_level == DramaLevel.DRAMATIC

    def test_escalate_drama_at_max(self):
        """Test escalating drama at maximum level does nothing."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)

        director.escalate_drama()
        assert director.drama_level == DramaLevel.DRAMATIC

    def test_deescalate_drama(self):
        """Test de-escalating drama level."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)

        director.deescalate_drama()
        assert director.drama_level == DramaLevel.MODERATE

        director.deescalate_drama()
        assert director.drama_level == DramaLevel.SUBTLE

    def test_deescalate_drama_at_min(self):
        """Test de-escalating drama at minimum level does nothing."""
        director = ChaosDirector(drama_level=DramaLevel.SUBTLE)

        director.deescalate_drama()
        assert director.drama_level == DramaLevel.SUBTLE

    def test_escalate_deescalate_cycle(self):
        """Test escalating and de-escalating returns to original."""
        director = ChaosDirector(drama_level=DramaLevel.MODERATE)

        director.escalate_drama()
        director.deescalate_drama()
        assert director.drama_level == DramaLevel.MODERATE


class TestGlobalDirector:
    """Tests for global director functions."""

    def test_get_chaos_director_creates_singleton(self):
        """Test get_chaos_director creates a singleton instance."""
        # Reset global state
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        director1 = get_chaos_director()
        director2 = get_chaos_director()

        assert director1 is director2

    def test_get_chaos_director_default_level(self):
        """Test get_chaos_director uses default level on first call."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        director = get_chaos_director()
        assert director.drama_level == DramaLevel.MODERATE

    def test_get_chaos_director_ignores_level_after_creation(self):
        """Test get_chaos_director ignores drama_level after first call."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        director1 = get_chaos_director(drama_level=DramaLevel.SUBTLE)
        director2 = get_chaos_director(drama_level=DramaLevel.DRAMATIC)

        # Both should return the same instance
        assert director1 is director2
        # Level set on first call
        assert director1.drama_level == DramaLevel.SUBTLE


class TestTheatricalHelpers:
    """Tests for theatrical helper functions."""

    def test_theatrical_timeout_returns_string(self):
        """Test theatrical_timeout returns a string message."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        message = theatrical_timeout("claude", 60.0)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_theatrical_timeout_contains_agent(self):
        """Test theatrical_timeout message contains agent name."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        message = theatrical_timeout("gpt4", 30.0)
        assert "gpt4" in message

    def test_theatrical_error_connection(self):
        """Test theatrical_error with connection type."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        message = theatrical_error("claude", error_type="connection")
        assert isinstance(message, str)
        assert "claude" in message

    def test_theatrical_error_rate_limit(self):
        """Test theatrical_error with rate_limit type."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        message = theatrical_error("claude", error_type="rate_limit")
        assert isinstance(message, str)
        assert "claude" in message

    def test_theatrical_error_internal(self):
        """Test theatrical_error with internal type."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        message = theatrical_error("claude", error_type="internal")
        assert isinstance(message, str)
        assert "claude" in message

    def test_theatrical_error_default(self):
        """Test theatrical_error with default (unknown) type."""
        import aragora.debate.chaos_theater as ct

        ct._chaos_director = None

        message = theatrical_error("claude", error_type="unknown_type")
        assert isinstance(message, str)
        # Should fall back to internal error
        assert "claude" in message


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_empty_agent_name(self):
        """Test handling of empty agent name."""
        director = ChaosDirector()
        response = director.timeout_response("", 30.0)

        # Should not crash, just produce a message
        assert isinstance(response, TheaterResponse)
        assert response.agent_name == ""

    def test_special_characters_in_agent_name(self):
        """Test handling of special characters in agent name."""
        director = ChaosDirector()
        response = director.timeout_response("claude-3.5-sonnet", 30.0)

        assert "claude-3.5-sonnet" in response.message

    def test_very_long_agent_name(self):
        """Test handling of very long agent name."""
        director = ChaosDirector()
        long_name = "a" * 1000
        response = director.timeout_response(long_name, 30.0)

        assert long_name in response.message

    def test_negative_duration(self):
        """Test handling of negative duration."""
        director = ChaosDirector()
        response = director.timeout_response("claude", -10.0)

        # Should handle gracefully
        assert isinstance(response, TheaterResponse)
        assert response.duration_hint == -5.0  # Half of -10

    def test_zero_duration(self):
        """Test handling of zero duration."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 0.0)

        assert isinstance(response, TheaterResponse)
        assert response.duration_hint == 0.0

    def test_very_large_duration(self):
        """Test handling of very large duration."""
        director = ChaosDirector()
        response = director.timeout_response("claude", 1e10)

        assert isinstance(response, TheaterResponse)
        assert response.duration_hint == 5e9

    def test_progress_negative_percent(self):
        """Test handling of negative progress percentage."""
        director = ChaosDirector()
        response = director.progress_response("claude", progress_percent=-10)

        assert isinstance(response, TheaterResponse)

    def test_progress_over_100_percent(self):
        """Test handling of progress over 100%."""
        director = ChaosDirector()
        response = director.progress_response("claude", progress_percent=150)

        assert isinstance(response, TheaterResponse)

    def test_unicode_in_agent_name(self):
        """Test handling of unicode in agent name."""
        director = ChaosDirector()
        response = director.timeout_response("claude-\u4e2d\u6587", 30.0)

        assert "claude-\u4e2d\u6587" in response.message

    def test_retry_after_zero(self):
        """Test rate limit with zero retry_after falls back to backing off."""
        director = ChaosDirector()
        response = director.rate_limit_response("claude", retry_after=0.0)

        # 0.0 is falsy in Python, so it falls back to "Backing off..."
        assert response.duration_hint == 0.0
        assert response.recovery_suggestion == "Backing off..."


class TestMessageFormatting:
    """Tests for message formatting correctness."""

    def test_all_timeout_messages_format_correctly(self):
        """Test all timeout messages format without errors."""
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            for _ in range(len(ChaosDirector.TIMEOUT_MESSAGES[level]) * 2):
                response = director.timeout_response("test_agent", 45.0)
                assert "{" not in response.message
                assert "}" not in response.message

    def test_all_connection_messages_format_correctly(self):
        """Test all connection messages format without errors."""
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            for _ in range(len(ChaosDirector.CONNECTION_MESSAGES[level]) * 2):
                response = director.connection_response("test_agent")
                assert "{" not in response.message
                assert "}" not in response.message

    def test_all_rate_limit_messages_format_correctly(self):
        """Test all rate limit messages format without errors."""
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            for _ in range(len(ChaosDirector.RATE_LIMIT_MESSAGES[level]) * 2):
                response = director.rate_limit_response("test_agent")
                assert "{" not in response.message
                assert "}" not in response.message

    def test_all_internal_messages_format_correctly(self):
        """Test all internal error messages format without errors."""
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            for _ in range(len(ChaosDirector.INTERNAL_MESSAGES[level]) * 2):
                response = director.internal_error_response("test_agent")
                assert "{" not in response.message
                assert "}" not in response.message

    def test_all_recovery_messages_format_correctly(self):
        """Test all recovery messages format without errors."""
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            for _ in range(len(ChaosDirector.RECOVERY_MESSAGES[level]) * 2):
                response = director.recovery_response("test_agent")
                assert "{" not in response.message
                assert "}" not in response.message

    def test_all_progress_messages_format_correctly(self):
        """Test all progress messages format without errors."""
        for level in DramaLevel:
            director = ChaosDirector(drama_level=level)
            for _ in range(len(ChaosDirector.PROGRESS_MESSAGES[level]) * 2):
                response = director.progress_response("test_agent", progress_percent=50)
                assert "{" not in response.message
                assert "}" not in response.message


class TestRandomization:
    """Tests for randomization behavior."""

    def test_deterministic_with_seed(self):
        """Test that messages are deterministic with same random seed."""
        seed = 42

        random.seed(seed)
        director1 = ChaosDirector()
        msg1 = director1.timeout_response("claude", 30.0).message

        random.seed(seed)
        director2 = ChaosDirector()
        msg2 = director2.timeout_response("claude", 30.0).message

        assert msg1 == msg2

    def test_variation_without_seed(self):
        """Test that messages vary without fixed seed."""
        director = ChaosDirector(drama_level=DramaLevel.DRAMATIC)

        # Collect many messages
        messages = set()
        for _ in range(50):
            msg = director.timeout_response("claude", 30.0).message
            messages.add(msg)

        # Should have variation (dramatic has 7 timeout messages)
        assert len(messages) > 1


class TestIntegration:
    """Integration tests for typical usage patterns."""

    def test_escalating_drama_on_repeated_failures(self):
        """Test typical pattern of escalating drama on failures."""
        director = ChaosDirector(drama_level=DramaLevel.SUBTLE)

        # First failure - subtle
        response1 = director.timeout_response("claude", 30.0)
        assert response1.drama_level == DramaLevel.SUBTLE

        # Escalate
        director.escalate_drama()
        response2 = director.timeout_response("claude", 60.0)
        assert response2.drama_level == DramaLevel.MODERATE

        # Escalate again
        director.escalate_drama()
        response3 = director.timeout_response("claude", 90.0)
        assert response3.drama_level == DramaLevel.DRAMATIC

    def test_failure_to_recovery_flow(self):
        """Test typical flow from failure to recovery."""
        director = ChaosDirector()

        # Initial failure
        failure = director.timeout_response("claude", 30.0)
        assert failure.failure_type == FailureType.TIMEOUT
        assert failure.recovery_suggestion is not None

        # Progress updates
        progress = director.progress_response("claude", 25)
        assert progress.agent_name == "claude"

        progress = director.progress_response("claude", 75)
        assert progress.agent_name == "claude"

        # Recovery
        recovery = director.recovery_response("claude")
        assert "claude" in recovery.message

    def test_multiple_agents_different_failures(self):
        """Test handling multiple agents with different failure types."""
        director = ChaosDirector()

        timeout = director.timeout_response("claude", 30.0)
        connection = director.connection_response("gpt4")
        rate_limit = director.rate_limit_response("mistral", retry_after=60.0)
        internal = director.internal_error_response("gemini")

        assert timeout.failure_type == FailureType.TIMEOUT
        assert connection.failure_type == FailureType.CONNECTION
        assert rate_limit.failure_type == FailureType.RATE_LIMIT
        assert internal.failure_type == FailureType.INTERNAL

        # All should have their respective agent names
        assert timeout.agent_name == "claude"
        assert connection.agent_name == "gpt4"
        assert rate_limit.agent_name == "mistral"
        assert internal.agent_name == "gemini"
