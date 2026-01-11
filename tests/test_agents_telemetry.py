"""
Tests for Agent Telemetry module.

Tests cover:
- AgentTelemetry dataclass operations
- Collector registration and unregistration
- Telemetry emission to collectors
- Default collectors (Prometheus, ImmuneSystem, Blackbox)
- with_telemetry decorator (async and sync)
- TelemetryContext context manager
- Token estimation
- Error handling and graceful degradation
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, MagicMock

from aragora.agents.telemetry import (
    AgentTelemetry,
    TelemetryContext,
    register_telemetry_collector,
    unregister_telemetry_collector,
    _emit_telemetry,
    _default_prometheus_collector,
    _default_immune_system_collector,
    _default_blackbox_collector,
    setup_default_collectors,
    with_telemetry,
    get_telemetry_stats,
    reset_telemetry,
    _telemetry_collectors,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clean_collectors():
    """Reset collectors before and after each test."""
    reset_telemetry()
    yield
    reset_telemetry()


@pytest.fixture
def sample_telemetry():
    """Create a sample telemetry object."""
    return AgentTelemetry(
        agent_name="claude",
        operation="generate",
        start_time=time.time(),
    )


@pytest.fixture
def completed_telemetry():
    """Create a completed telemetry object."""
    telemetry = AgentTelemetry(
        agent_name="gpt4",
        operation="critique",
        start_time=time.time() - 0.5,  # Started 500ms ago
        input_tokens=100,
        output_tokens=50,
    )
    telemetry.complete(success=True)
    return telemetry


# ============================================================================
# AgentTelemetry Dataclass Tests
# ============================================================================

class TestAgentTelemetry:
    """Tests for the AgentTelemetry dataclass."""

    def test_create_telemetry(self, sample_telemetry):
        """Test creating a telemetry object."""
        assert sample_telemetry.agent_name == "claude"
        assert sample_telemetry.operation == "generate"
        assert sample_telemetry.success is True
        assert sample_telemetry.error_type is None

    def test_complete_success(self, sample_telemetry):
        """Test completing telemetry successfully."""
        time.sleep(0.01)  # Small delay
        sample_telemetry.complete(success=True)

        assert sample_telemetry.success is True
        assert sample_telemetry.end_time > sample_telemetry.start_time
        assert sample_telemetry.duration_ms > 0
        assert sample_telemetry.error_type is None

    def test_complete_with_error(self, sample_telemetry):
        """Test completing telemetry with an error."""
        error = ValueError("Test error message")
        sample_telemetry.complete(success=False, error=error)

        assert sample_telemetry.success is False
        assert sample_telemetry.error_type == "ValueError"
        assert "Test error message" in sample_telemetry.error_message

    def test_complete_with_long_error_message(self, sample_telemetry):
        """Test that long error messages are truncated."""
        error = ValueError("x" * 500)
        sample_telemetry.complete(success=False, error=error)

        assert len(sample_telemetry.error_message) <= 200

    def test_to_dict(self, completed_telemetry):
        """Test converting telemetry to dictionary."""
        data = completed_telemetry.to_dict()

        assert isinstance(data, dict)
        assert data["agent_name"] == "gpt4"
        assert data["operation"] == "critique"
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert "duration_ms" in data

    def test_estimate_tokens(self):
        """Test token estimation from text."""
        # ~4 chars per token
        assert AgentTelemetry.estimate_tokens("Hello") == 1  # 5 chars
        assert AgentTelemetry.estimate_tokens("Hello world test") == 4  # 16 chars
        assert AgentTelemetry.estimate_tokens("") == 0
        assert AgentTelemetry.estimate_tokens("a" * 400) == 100

    def test_estimate_tokens_none(self):
        """Test token estimation with None."""
        assert AgentTelemetry.estimate_tokens(None) == 0


# ============================================================================
# Collector Registration Tests
# ============================================================================

class TestCollectorRegistration:
    """Tests for collector registration and unregistration."""

    def test_register_collector(self):
        """Test registering a collector."""
        def my_collector(t):
            pass
        register_telemetry_collector(my_collector)

        stats = get_telemetry_stats()
        assert stats["collectors_count"] == 1

    def test_register_duplicate_collector(self):
        """Test that duplicate collectors are not registered."""
        def my_collector(t):
            pass
        register_telemetry_collector(my_collector)
        register_telemetry_collector(my_collector)

        stats = get_telemetry_stats()
        assert stats["collectors_count"] == 1

    def test_unregister_collector(self):
        """Test unregistering a collector."""
        mock_collector = Mock()
        register_telemetry_collector(mock_collector)
        unregister_telemetry_collector(mock_collector)

        stats = get_telemetry_stats()
        assert stats["collectors_count"] == 0

    def test_unregister_nonexistent_collector(self):
        """Test unregistering a collector that wasn't registered."""
        mock_collector = Mock()
        # Should not raise
        unregister_telemetry_collector(mock_collector)

    def test_get_telemetry_stats(self):
        """Test getting telemetry statistics."""
        def my_collector(t):
            pass

        register_telemetry_collector(my_collector)
        stats = get_telemetry_stats()

        assert stats["collectors_count"] == 1
        assert "my_collector" in stats["collectors"]


# ============================================================================
# Telemetry Emission Tests
# ============================================================================

class TestTelemetryEmission:
    """Tests for telemetry emission to collectors."""

    def test_emit_to_single_collector(self, completed_telemetry):
        """Test emitting telemetry to a single collector."""
        mock_collector = Mock()
        register_telemetry_collector(mock_collector)

        _emit_telemetry(completed_telemetry)

        mock_collector.assert_called_once_with(completed_telemetry)

    def test_emit_to_multiple_collectors(self, completed_telemetry):
        """Test emitting telemetry to multiple collectors."""
        collectors = [Mock() for _ in range(3)]
        for c in collectors:
            register_telemetry_collector(c)

        _emit_telemetry(completed_telemetry)

        for c in collectors:
            c.assert_called_once_with(completed_telemetry)

    def test_emit_handles_collector_error(self, completed_telemetry):
        """Test that collector errors don't break emission."""
        call_log = []

        def failing_collector(t):
            call_log.append("failing")
            raise RuntimeError("Collector failed")

        def working_collector(t):
            call_log.append("working")

        register_telemetry_collector(failing_collector)
        register_telemetry_collector(working_collector)

        # Should not raise
        _emit_telemetry(completed_telemetry)

        # Both collectors should be called (failing one logged error)
        assert "failing" in call_log
        assert "working" in call_log


# ============================================================================
# Default Collectors Tests
# ============================================================================

class TestDefaultCollectors:
    """Tests for default telemetry collectors."""

    def test_prometheus_collector_success(self, completed_telemetry):
        """Test Prometheus collector handles success telemetry.

        The collector attempts to import prometheus metrics and record them.
        When prometheus isn't available (common in tests), it gracefully
        handles the ImportError without raising.
        """
        # Should not raise - handles ImportError gracefully
        _default_prometheus_collector(completed_telemetry)
        # No assertion needed - we just verify it doesn't crash

    def test_prometheus_collector_import_error(self, completed_telemetry):
        """Test Prometheus collector handles ImportError gracefully."""
        # Should not raise when prometheus is not available
        _default_prometheus_collector(completed_telemetry)

    def test_immune_system_collector_success(self, completed_telemetry):
        """Test ImmuneSystem collector on success."""
        # Should not raise when immune system is not available
        _default_immune_system_collector(completed_telemetry)

    def test_immune_system_collector_failure(self, sample_telemetry):
        """Test ImmuneSystem collector on failure."""
        sample_telemetry.complete(success=False, error=ValueError("test"))
        # Should not raise
        _default_immune_system_collector(sample_telemetry)

    def test_immune_system_collector_timeout(self, sample_telemetry):
        """Test ImmuneSystem collector on timeout."""
        sample_telemetry.complete(success=False, error=TimeoutError("timeout"))
        # Should not raise
        _default_immune_system_collector(sample_telemetry)

    def test_blackbox_collector(self, completed_telemetry):
        """Test Blackbox collector."""
        # Should not raise when blackbox is not available
        _default_blackbox_collector(completed_telemetry)

    def test_setup_default_collectors(self):
        """Test setting up default collectors."""
        setup_default_collectors()

        stats = get_telemetry_stats()
        assert stats["collectors_count"] == 3
        assert "_default_prometheus_collector" in stats["collectors"]
        assert "_default_immune_system_collector" in stats["collectors"]
        assert "_default_blackbox_collector" in stats["collectors"]


# ============================================================================
# with_telemetry Decorator Tests
# ============================================================================

class TestWithTelemetryDecorator:
    """Tests for the with_telemetry decorator."""

    @pytest.mark.asyncio
    async def test_async_decorator_success(self):
        """Test decorator on async function success."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        class MockAgent:
            name = "test-agent"
            model = "gpt-4"

            @with_telemetry("generate")
            async def generate(self, prompt: str) -> str:
                return "response"

        agent = MockAgent()
        result = await agent.generate("hello")

        assert result == "response"
        assert len(collected) == 1
        assert collected[0].agent_name == "test-agent"
        assert collected[0].operation == "generate"
        assert collected[0].success is True

    @pytest.mark.asyncio
    async def test_async_decorator_failure(self):
        """Test decorator on async function failure."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        class MockAgent:
            name = "test-agent"

            @with_telemetry("generate")
            async def generate(self, prompt: str) -> str:
                raise ValueError("Test error")

        agent = MockAgent()
        with pytest.raises(ValueError):
            await agent.generate("hello")

        assert len(collected) == 1
        assert collected[0].success is False
        assert collected[0].error_type == "ValueError"

    def test_sync_decorator_success(self):
        """Test decorator on sync function success."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        class MockAgent:
            name = "sync-agent"
            model = "claude"

            @with_telemetry("vote")
            def vote(self, options: list) -> str:
                return "option1"

        agent = MockAgent()
        result = agent.vote(["option1", "option2"])

        assert result == "option1"
        assert len(collected) == 1
        assert collected[0].agent_name == "sync-agent"
        assert collected[0].operation == "vote"

    def test_decorator_with_extract_input(self):
        """Test decorator with custom input extractor."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        class MockAgent:
            name = "test"

            @with_telemetry("critique", extract_input=lambda self, text: text)
            def critique(self, text: str) -> str:
                return "critique result"

        agent = MockAgent()
        agent.critique("This is a test input with many characters")

        assert len(collected) == 1
        assert collected[0].input_chars > 0
        assert collected[0].input_tokens > 0

    def test_decorator_with_extract_output(self):
        """Test decorator with custom output extractor."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        class MockAgent:
            name = "test"

            @with_telemetry("generate", extract_output=lambda r: r.get("text", ""))
            def generate(self, prompt: str) -> dict:
                return {"text": "Generated text response"}

        agent = MockAgent()
        agent.generate("prompt")

        assert len(collected) == 1
        assert collected[0].output_chars > 0
        assert collected[0].output_tokens > 0

    def test_decorator_auto_extracts_string_output(self):
        """Test decorator auto-extracts tokens from string output."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        class MockAgent:
            name = "test"

            @with_telemetry("generate")
            def generate(self, prompt: str) -> str:
                return "This is a response with multiple words"

        agent = MockAgent()
        agent.generate("prompt")

        assert len(collected) == 1
        assert collected[0].output_chars == len("This is a response with multiple words")
        assert collected[0].output_tokens > 0


# ============================================================================
# TelemetryContext Tests
# ============================================================================

class TestTelemetryContext:
    """Tests for the TelemetryContext context manager."""

    def test_context_success(self):
        """Test context manager on success."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        with TelemetryContext("agent1", "generate", "gpt-4") as ctx:
            ctx.set_input("input text")
            ctx.set_output("output text")

        assert len(collected) == 1
        assert collected[0].agent_name == "agent1"
        assert collected[0].operation == "generate"
        assert collected[0].success is True
        assert collected[0].input_chars == len("input text")
        assert collected[0].output_chars == len("output text")

    def test_context_failure(self):
        """Test context manager on failure."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        with pytest.raises(RuntimeError):
            with TelemetryContext("agent2", "critique") as ctx:
                raise RuntimeError("Oops")

        assert len(collected) == 1
        assert collected[0].success is False
        assert collected[0].error_type == "RuntimeError"
        assert "Oops" in collected[0].error_message

    def test_context_set_error_manually(self):
        """Test manually setting error in context."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        with TelemetryContext("agent3", "vote") as ctx:
            ctx.set_error(ValueError("Manual error"))

        # Error was set but no exception raised, so success is True
        assert len(collected) == 1
        assert collected[0].error_type == "ValueError"


# ============================================================================
# Reset and Edge Case Tests
# ============================================================================

class TestResetAndEdgeCases:
    """Tests for reset functionality and edge cases."""

    def test_reset_telemetry(self):
        """Test resetting telemetry system."""
        register_telemetry_collector(Mock())
        register_telemetry_collector(Mock())

        reset_telemetry()

        stats = get_telemetry_stats()
        assert stats["collectors_count"] == 0

    def test_telemetry_without_agent_name(self):
        """Test decorator handles missing agent name."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        @with_telemetry("test")
        def standalone_func():
            return "result"

        standalone_func()

        assert len(collected) == 1
        assert collected[0].agent_name == "unknown"

    def test_telemetry_metadata(self):
        """Test telemetry metadata field."""
        telemetry = AgentTelemetry(
            agent_name="test",
            operation="generate",
            start_time=time.time(),
            metadata={"model": "gpt-4", "custom": "value"},
        )

        assert telemetry.metadata["model"] == "gpt-4"
        assert telemetry.metadata["custom"] == "value"

    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        collected = []
        register_telemetry_collector(lambda t: collected.append(t))

        class Agent:
            name = "multi-op"

            @with_telemetry("generate")
            def generate(self):
                return "gen"

            @with_telemetry("critique")
            def critique(self):
                return "crit"

            @with_telemetry("vote")
            def vote(self):
                return "vote"

        agent = Agent()
        agent.generate()
        agent.critique()
        agent.vote()

        assert len(collected) == 3
        operations = {t.operation for t in collected}
        assert operations == {"generate", "critique", "vote"}
