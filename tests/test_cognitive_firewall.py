"""
Tests for Cognitive Firewall components.

Tests TelemetryConfig, SecurityBarrier, TelemetryVerifier, and broadcast_event.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# TelemetryConfig Tests
# ==============================================================================


class TestTelemetryConfig:
    """Tests for TelemetryConfig class."""

    def setup_method(self):
        """Reset singleton before each test."""
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

    def teardown_method(self):
        """Clean up environment after each test."""
        os.environ.pop("ARAGORA_TELEMETRY_LEVEL", None)
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

    def test_default_level_is_controlled(self):
        """Default telemetry level should be CONTROLLED for security."""
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig()
        assert config.level == TelemetryLevel.CONTROLLED

    def test_level_from_environment_silent(self):
        """ARAGORA_TELEMETRY_LEVEL=silent should set SILENT level."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "silent"
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig()
        assert config.level == TelemetryLevel.SILENT
        assert config.is_silent()

    def test_level_from_environment_diagnostic(self):
        """ARAGORA_TELEMETRY_LEVEL=diagnostic should set DIAGNOSTIC level."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "diagnostic"
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig()
        assert config.level == TelemetryLevel.DIAGNOSTIC
        assert config.is_diagnostic()

    def test_level_from_environment_controlled(self):
        """ARAGORA_TELEMETRY_LEVEL=controlled should set CONTROLLED level."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "controlled"
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig()
        assert config.level == TelemetryLevel.CONTROLLED
        assert config.is_controlled()

    def test_level_from_environment_spectacle(self):
        """ARAGORA_TELEMETRY_LEVEL=spectacle should set SPECTACLE level."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "spectacle"
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig()
        assert config.level == TelemetryLevel.SPECTACLE
        assert config.is_spectacle()

    def test_numeric_level_shortcuts(self):
        """Numeric level values (0-3) should work."""
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        for num, expected in [
            ("0", TelemetryLevel.SILENT),
            ("1", TelemetryLevel.DIAGNOSTIC),
            ("2", TelemetryLevel.CONTROLLED),
            ("3", TelemetryLevel.SPECTACLE),
        ]:
            TelemetryConfig.reset_instance()
            os.environ["ARAGORA_TELEMETRY_LEVEL"] = num
            config = TelemetryConfig()
            assert config.level == expected, f"Level {num} should be {expected}"

    def test_invalid_level_defaults_to_controlled(self):
        """Invalid ARAGORA_TELEMETRY_LEVEL should default to CONTROLLED."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "invalid_value"
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig()
        assert config.level == TelemetryLevel.CONTROLLED

    def test_case_insensitive_level(self):
        """Level names should be case-insensitive."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "SPECTACLE"
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig()
        assert config.level == TelemetryLevel.SPECTACLE

    def test_should_broadcast(self):
        """should_broadcast() should return True for CONTROLLED and SPECTACLE."""
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        for level, expected in [
            (TelemetryLevel.SILENT, False),
            (TelemetryLevel.DIAGNOSTIC, False),
            (TelemetryLevel.CONTROLLED, True),
            (TelemetryLevel.SPECTACLE, True),
        ]:
            config = TelemetryConfig(level=level)
            assert config.should_broadcast() == expected, f"Level {level} should_broadcast={expected}"

    def test_should_redact(self):
        """should_redact() should return True only for CONTROLLED."""
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        for level, expected in [
            (TelemetryLevel.SILENT, False),
            (TelemetryLevel.DIAGNOSTIC, False),
            (TelemetryLevel.CONTROLLED, True),
            (TelemetryLevel.SPECTACLE, False),
        ]:
            config = TelemetryConfig(level=level)
            assert config.should_redact() == expected, f"Level {level} should_redact={expected}"

    def test_allows_level(self):
        """allows_level() should respect level hierarchy."""
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig(level=TelemetryLevel.CONTROLLED)

        # CONTROLLED allows SILENT, DIAGNOSTIC, CONTROLLED but not SPECTACLE
        assert config.allows_level(TelemetryLevel.SILENT)
        assert config.allows_level(TelemetryLevel.DIAGNOSTIC)
        assert config.allows_level(TelemetryLevel.CONTROLLED)
        assert not config.allows_level(TelemetryLevel.SPECTACLE)

    def test_singleton_instance(self):
        """get_instance() should return the same instance."""
        from aragora.debate.telemetry_config import TelemetryConfig

        instance1 = TelemetryConfig.get_instance()
        instance2 = TelemetryConfig.get_instance()
        assert instance1 is instance2

    def test_explicit_level_override(self):
        """Explicit level in constructor should override environment."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "silent"
        from aragora.debate.telemetry_config import TelemetryConfig, TelemetryLevel

        config = TelemetryConfig(level=TelemetryLevel.SPECTACLE)
        assert config.level == TelemetryLevel.SPECTACLE


# ==============================================================================
# SecurityBarrier Tests
# ==============================================================================


class TestSecurityBarrier:
    """Tests for SecurityBarrier class."""

    def test_redact_api_key(self):
        """Should redact API key patterns."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        # Test various API key patterns
        test_cases = [
            ("api_key = 'sk-abc123def456'", "[REDACTED]"),
            ("token: secret123", "[REDACTED]"),
            ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "[REDACTED]"),
            ("sk-proj-abcdefghijklmnopqrstuvwxyz", "[REDACTED]"),
        ]

        for input_text, _ in test_cases:
            result = barrier.redact(input_text)
            assert "[REDACTED]" in result, f"Failed to redact: {input_text}"

    def test_redact_environment_variable(self):
        """Should redact environment variable assignments."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        test_cases = [
            "ANTHROPIC_API_KEY = sk-ant-123456",
            "OPENAI_KEY=test123",
            "GEMINI_API_KEY: abc123",
        ]

        for input_text in test_cases:
            result = barrier.redact(input_text)
            assert "[REDACTED]" in result, f"Failed to redact: {input_text}"

    def test_redact_url_with_credentials(self):
        """Should redact URLs with embedded credentials."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        input_text = "Connect to https://user:password@example.com/api"
        result = barrier.redact(input_text)
        assert "[REDACTED]" in result

    def test_redact_private_key(self):
        """Should redact private key headers."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        input_text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        result = barrier.redact(input_text)
        assert "[REDACTED]" in result

    def test_no_redaction_for_safe_content(self):
        """Should not redact normal content."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        safe_text = "This is a normal message about API design patterns."
        result = barrier.redact(safe_text)
        assert result == safe_text

    def test_redact_dict(self):
        """Should recursively redact dictionary values."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        data = {
            "message": "api_key = secret123",
            "nested": {
                "token": "Bearer abc123",
            },
            "list": ["normal", "password: test123"],
            "number": 42,
        }

        result = barrier.redact_dict(data)

        assert "[REDACTED]" in result["message"]
        assert "[REDACTED]" in result["nested"]["token"]
        assert result["list"][0] == "normal"
        assert "[REDACTED]" in result["list"][1]
        assert result["number"] == 42

    def test_contains_sensitive(self):
        """Should detect sensitive content."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        assert barrier.contains_sensitive("api_key = secret123")
        assert barrier.contains_sensitive("sk-proj-abcdefghijklmnopqrst")
        assert not barrier.contains_sensitive("This is normal text")

    def test_custom_pattern(self):
        """Should support custom redaction patterns."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()
        barrier.add_pattern(r"CUSTOM_SECRET_\d+")

        input_text = "The value is CUSTOM_SECRET_12345"
        result = barrier.redact(input_text)
        assert "[REDACTED]" in result

    def test_custom_redaction_marker(self):
        """Should support custom redaction marker."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier(redaction_marker="***HIDDEN***")

        input_text = "api_key = secret123"
        result = barrier.redact(input_text)
        assert "***HIDDEN***" in result

    def test_empty_content(self):
        """Should handle empty content gracefully."""
        from aragora.debate.security_barrier import SecurityBarrier

        barrier = SecurityBarrier()

        assert barrier.redact("") == ""
        assert barrier.redact(None) is None
        assert barrier.redact_dict({}) == {}
        assert barrier.redact_dict(None) is None
        assert not barrier.contains_sensitive("")
        assert not barrier.contains_sensitive(None)


# ==============================================================================
# TelemetryVerifier Tests
# ==============================================================================


class TestTelemetryVerifier:
    """Tests for TelemetryVerifier class."""

    def test_verify_agent_with_capabilities(self):
        """Should pass verification for agents with required capabilities."""
        from aragora.debate.security_barrier import TelemetryVerifier

        verifier = TelemetryVerifier()

        # Create mock agent with required capabilities
        agent = Mock()
        agent.name = "test-agent"
        agent.generate = Mock()

        passed, missing = verifier.verify_agent(agent)
        assert passed
        assert len(missing) == 0

    def test_verify_agent_missing_capabilities(self):
        """Should fail verification for agents missing capabilities."""
        from aragora.debate.security_barrier import TelemetryVerifier

        verifier = TelemetryVerifier()

        # Create mock agent missing 'generate'
        agent = Mock(spec=["name"])
        agent.name = "test-agent"

        passed, missing = verifier.verify_agent(agent)
        assert not passed
        assert "generate" in missing

    def test_verify_telemetry_level(self):
        """Should verify agents for specific telemetry levels."""
        from aragora.debate.security_barrier import TelemetryVerifier

        verifier = TelemetryVerifier()

        # Agent with all capabilities
        full_agent = Mock()
        full_agent.name = "full-agent"
        full_agent.generate = Mock()
        full_agent.model = "test-model"

        assert verifier.verify_telemetry_level("thought_streaming", full_agent)
        assert verifier.verify_telemetry_level("capability_probe", full_agent)
        assert verifier.verify_telemetry_level("diagnostic", full_agent)

    def test_verification_report(self):
        """Should generate verification report."""
        from aragora.debate.security_barrier import TelemetryVerifier

        verifier = TelemetryVerifier()

        # Verify a few agents
        agent1 = Mock()
        agent1.name = "agent1"
        agent1.generate = Mock()

        agent2 = Mock(spec=["name"])
        agent2.name = "agent2"

        verifier.verify_agent(agent1)
        verifier.verify_agent(agent2)

        report = verifier.get_verification_report()

        assert report["total"] == 2
        assert report["passed"] == 1
        assert report["failed"] == 1

    def test_clear_cache(self):
        """Should clear capability cache."""
        from aragora.debate.security_barrier import TelemetryVerifier

        verifier = TelemetryVerifier()

        agent = Mock()
        agent.name = "test-agent"
        agent.generate = Mock()

        verifier.verify_agent(agent)
        assert len(verifier._verification_results) == 1

        verifier.clear_cache()
        assert len(verifier._verification_results) == 0
        assert len(verifier._capability_cache) == 0


# ==============================================================================
# broadcast_event Tests
# ==============================================================================


class TestBroadcastEvent:
    """Tests for SyncEventEmitter.broadcast_event method."""

    def setup_method(self):
        """Reset telemetry config before each test."""
        os.environ.pop("ARAGORA_TELEMETRY_LEVEL", None)
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        os.environ.pop("ARAGORA_TELEMETRY_LEVEL", None)
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

    def test_broadcast_event_silent_suppresses(self):
        """broadcast_event should suppress events in SILENT mode."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "silent"
        from aragora.server.stream import SyncEventEmitter, StreamEventType

        emitter = SyncEventEmitter()
        result = emitter.broadcast_event(
            StreamEventType.TELEMETRY_THOUGHT,
            {"thought": "test"},
            agent="test-agent",
        )

        assert result is False
        assert emitter.drain() == []  # No events emitted

    def test_broadcast_event_controlled_with_redaction(self):
        """broadcast_event should apply redaction in CONTROLLED mode."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "controlled"
        from aragora.server.stream import SyncEventEmitter, StreamEventType
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

        emitter = SyncEventEmitter()

        def redactor(data):
            return {"thought": "[REDACTED]"}

        result = emitter.broadcast_event(
            StreamEventType.TELEMETRY_THOUGHT,
            {"thought": "secret api_key=abc123"},
            agent="test-agent",
            redactor=redactor,
        )

        assert result is True
        events = emitter.drain()
        # Should have redaction notification + thought event
        assert len(events) == 2

    def test_broadcast_event_spectacle_no_redaction(self):
        """broadcast_event should not redact in SPECTACLE mode."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "spectacle"
        from aragora.server.stream import SyncEventEmitter, StreamEventType
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

        emitter = SyncEventEmitter()

        original_data = {"thought": "secret api_key=abc123"}

        def redactor(data):
            return {"thought": "[REDACTED]"}

        result = emitter.broadcast_event(
            StreamEventType.TELEMETRY_THOUGHT,
            original_data,
            agent="test-agent",
            redactor=redactor,
        )

        assert result is True
        events = emitter.drain()
        assert len(events) == 1
        # Data should NOT be redacted in spectacle mode
        assert events[0].data["thought"] == "secret api_key=abc123"

    def test_broadcast_non_telemetry_event(self):
        """Non-telemetry events should always broadcast."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "diagnostic"
        from aragora.server.stream import SyncEventEmitter, StreamEventType
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

        emitter = SyncEventEmitter()

        result = emitter.broadcast_event(
            StreamEventType.AGENT_MESSAGE,  # Not a telemetry event
            {"message": "Hello"},
            agent="test-agent",
        )

        assert result is True
        events = emitter.drain()
        assert len(events) == 1

    def test_broadcast_event_redaction_failure_suppresses(self):
        """Redaction failure should suppress event for security."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "controlled"
        from aragora.server.stream import SyncEventEmitter, StreamEventType
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

        emitter = SyncEventEmitter()

        def failing_redactor(data):
            raise ValueError("Redaction failed")

        result = emitter.broadcast_event(
            StreamEventType.TELEMETRY_THOUGHT,
            {"thought": "test"},
            agent="test-agent",
            redactor=failing_redactor,
        )

        assert result is False  # Suppressed due to redaction failure


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCognitiveFirewallIntegration:
    """Integration tests for the full Cognitive Firewall pipeline."""

    def setup_method(self):
        """Reset state before each test."""
        os.environ.pop("ARAGORA_TELEMETRY_LEVEL", None)
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        os.environ.pop("ARAGORA_TELEMETRY_LEVEL", None)
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

    def test_full_pipeline_controlled_mode(self):
        """Test full pipeline: verify agent, redact, broadcast."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "controlled"
        from aragora.debate.security_barrier import SecurityBarrier, TelemetryVerifier
        from aragora.server.stream import SyncEventEmitter, StreamEventType
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

        # Setup components
        verifier = TelemetryVerifier()
        barrier = SecurityBarrier()
        emitter = SyncEventEmitter()

        # Create mock agent
        agent = Mock()
        agent.name = "test-agent"
        agent.generate = Mock()

        # Verify agent
        passed, missing = verifier.verify_agent(agent)
        assert passed

        # Prepare thought data with sensitive content
        thought_data = {
            "agent": agent.name,
            "thought": "Calling API with key sk-abc123456789",
        }

        # Broadcast with redaction
        result = emitter.broadcast_event(
            StreamEventType.TELEMETRY_THOUGHT,
            thought_data,
            agent=agent.name,
            redactor=barrier.redact_dict,
        )

        assert result is True

        # Verify events
        events = emitter.drain()
        assert len(events) >= 1

        # Find the thought event
        thought_events = [e for e in events if e.type == StreamEventType.TELEMETRY_THOUGHT]
        assert len(thought_events) == 1
        assert "[REDACTED]" in thought_events[0].data["thought"]

    def test_full_pipeline_spectacle_mode(self):
        """Test full pipeline in spectacle mode (no redaction)."""
        os.environ["ARAGORA_TELEMETRY_LEVEL"] = "spectacle"
        from aragora.debate.security_barrier import SecurityBarrier, TelemetryVerifier
        from aragora.server.stream import SyncEventEmitter, StreamEventType
        from aragora.debate.telemetry_config import TelemetryConfig
        TelemetryConfig.reset_instance()

        # Setup components
        verifier = TelemetryVerifier()
        barrier = SecurityBarrier()
        emitter = SyncEventEmitter()

        # Create mock agent
        agent = Mock()
        agent.name = "test-agent"
        agent.generate = Mock()

        # Verify agent
        passed, _ = verifier.verify_agent(agent)
        assert passed

        # Prepare thought data
        thought_data = {
            "agent": agent.name,
            "thought": "This is a full thought with no redaction",
        }

        # Broadcast (redactor should be ignored in spectacle mode)
        result = emitter.broadcast_event(
            StreamEventType.TELEMETRY_THOUGHT,
            thought_data,
            agent=agent.name,
            redactor=barrier.redact_dict,
        )

        assert result is True

        events = emitter.drain()
        thought_events = [e for e in events if e.type == StreamEventType.TELEMETRY_THOUGHT]
        assert len(thought_events) == 1
        # Original content preserved
        assert thought_events[0].data["thought"] == "This is a full thought with no redaction"
