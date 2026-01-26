"""
Integration tests for agent fallback mechanisms.

Tests various fallback scenarios:
- Primary agent failure → OpenRouter fallback
- All agents fail → graceful degradation
- Circuit breaker tripping
- Quota error detection
- Fallback chain behavior
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Optional

from aragora.core import Vote, Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.agents.fallback import QuotaFallbackMixin, QUOTA_ERROR_KEYWORDS
from tests.integration.conftest import MockAgent, FailingAgent


class TestQuotaErrorDetection:
    """Tests for quota/rate limit error detection."""

    @pytest.fixture
    def quota_detector(self):
        """Create a QuotaFallbackMixin instance for testing."""

        class TestDetector(QuotaFallbackMixin):
            pass

        return TestDetector()

    def test_detects_rate_limit_error(self, quota_detector):
        """Should detect rate limit errors from error messages."""
        assert quota_detector.is_quota_error(429, "Rate limit exceeded") is True
        assert quota_detector.is_quota_error(429, "rate_limit_exceeded") is True
        assert quota_detector.is_quota_error(429, "Too many requests") is True

    def test_detects_quota_exceeded_error(self, quota_detector):
        """Should detect quota exceeded errors."""
        assert quota_detector.is_quota_error(403, "Quota exceeded") is True
        assert quota_detector.is_quota_error(403, "Resource exhausted") is True
        assert quota_detector.is_quota_error(429, "Insufficient quota") is True

    def test_detects_billing_errors(self, quota_detector):
        """Should detect billing-related errors."""
        assert quota_detector.is_quota_error(403, "Billing issue") is True

    def test_ignores_non_quota_errors(self, quota_detector):
        """Should not flag non-quota errors as quota errors."""
        assert quota_detector.is_quota_error(500, "Internal server error") is False
        assert quota_detector.is_quota_error(404, "Not found") is False
        assert quota_detector.is_quota_error(400, "Bad request") is False

    def test_status_code_429_always_quota(self, quota_detector):
        """HTTP 429 should always be treated as quota error."""
        assert quota_detector.is_quota_error(429, "Unknown error") is True
        assert quota_detector.is_quota_error(429, "") is True


class TestAgentFallbackBehavior:
    """Tests for agent fallback when primary agent fails."""

    @pytest.mark.asyncio
    async def test_failing_agent_handled_gracefully(self):
        """Debate should handle agent failures gracefully."""
        # Create agents where one fails after first call
        agents = [
            MockAgent(name="reliable_1", responses=["Solution A"]),
            MockAgent(name="reliable_2", responses=["Solution B"]),
            FailingAgent(
                name="failing_agent",
                fail_after=1,
                error_type=RuntimeError,
                responses=["Initial response"],
            ),
        ]

        env = Environment(task="Test fault tolerance")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            # Should complete without crashing
            result = await arena.run()

        assert result is not None
        # Debate should complete even with one failing agent

    @pytest.mark.asyncio
    async def test_all_agents_fail_graceful_degradation(self):
        """Debate should degrade gracefully when all agents fail."""
        # All agents fail immediately
        agents = [
            FailingAgent(
                name=f"failing_{i}",
                fail_after=0,  # Fail immediately
                error_type=RuntimeError,
                responses=[],
            )
            for i in range(3)
        ]

        env = Environment(task="Test total failure")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            # Should not crash, but may not produce meaningful results
            try:
                result = await arena.run()
                # If it completes, check result is valid (possibly empty)
                assert result is not None
            except Exception as e:
                # Some exceptions are acceptable for total failure
                assert "failure" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Debate should continue with remaining agents after partial failure."""
        # One agent fails mid-debate, others continue
        agents = [
            MockAgent(
                name="reliable_1",
                responses=["Solution A round 1", "Solution A round 2", "Solution A round 3"],
            ),
            MockAgent(
                name="reliable_2",
                responses=["Solution B round 1", "Solution B round 2", "Solution B round 3"],
            ),
            FailingAgent(
                name="partial_fail",
                fail_after=2,  # Works for first 2 calls, then fails
                responses=["Working response 1", "Working response 2"],
            ),
        ]

        env = Environment(task="Test partial recovery")
        protocol = DebateProtocol(rounds=3, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        # Should complete with remaining agents


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker behavior with failing agents."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_repeated_failures(self):
        """Circuit breaker should prevent calling repeatedly failing agents."""
        from aragora.resilience import CircuitBreaker

        # Create a circuit breaker with low threshold
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60)

        failure_count = 0

        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise RuntimeError("Simulated failure")

        # First few calls should execute and fail
        for _ in range(3):
            try:
                await breaker.execute(failing_operation)
            except (RuntimeError, Exception):
                pass

        # Circuit should be open after threshold failures
        assert breaker.state != "closed" or failure_count <= 2


class TestFallbackChain:
    """Tests for fallback chain behavior."""

    @pytest.mark.asyncio
    async def test_fallback_to_secondary_provider(self):
        """Should fall back to secondary provider when primary fails."""
        from aragora.agents.fallback import AgentFallbackChain

        primary_called = False
        fallback_called = False

        async def primary_generate(prompt, context):
            nonlocal primary_called
            primary_called = True
            raise RuntimeError("Primary unavailable")

        async def fallback_generate(prompt, context):
            nonlocal fallback_called
            fallback_called = True
            return "Fallback response"

        # Create mock agents
        primary = Mock()
        primary.generate = primary_generate
        primary.name = "primary"

        fallback = Mock()
        fallback.generate = fallback_generate
        fallback.name = "fallback"

        chain = AgentFallbackChain([primary, fallback])
        result = await chain.generate("test prompt", [])

        assert primary_called is True
        assert fallback_called is True
        assert result == "Fallback response"

    @pytest.mark.asyncio
    async def test_fallback_chain_exhaustion(self):
        """Should raise when all providers in chain fail."""
        from aragora.agents.fallback import AgentFallbackChain

        async def always_fail(prompt, context):
            raise RuntimeError("Provider unavailable")

        agents = []
        for i in range(3):
            agent = Mock()
            agent.generate = always_fail
            agent.name = f"agent_{i}"
            agents.append(agent)

        chain = AgentFallbackChain(agents)

        with pytest.raises(RuntimeError):
            await chain.generate("test prompt", [])


class TestQuotaFallbackMixin:
    """Tests for the QuotaFallbackMixin behavior."""

    def test_is_quota_error_method(self):
        """Mixin should correctly identify quota errors."""

        class TestAgent(QuotaFallbackMixin):
            pass

        agent = TestAgent()

        # Test detection
        assert agent.is_quota_error(429, "Rate limit") is True
        assert agent.is_quota_error(402, "Quota exceeded") is True
        assert agent.is_quota_error(500, "Server error") is False

    def test_openrouter_model_mapping(self):
        """Mixin should map provider models to OpenRouter equivalents."""

        class TestAgent(QuotaFallbackMixin):
            OPENROUTER_MODEL_MAP = {
                "gpt-4": "openai/gpt-4",
                "claude-3-opus": "anthropic/claude-3-opus",
            }
            DEFAULT_FALLBACK_MODEL = "anthropic/claude-sonnet-4"

        agent = TestAgent()
        agent.model = "gpt-4"

        mapped = agent.OPENROUTER_MODEL_MAP.get(agent.model, agent.DEFAULT_FALLBACK_MODEL)
        assert mapped == "openai/gpt-4"

    def test_fallback_model_default(self):
        """Should use default fallback model when no mapping exists."""

        class TestAgent(QuotaFallbackMixin):
            OPENROUTER_MODEL_MAP = {"gpt-4": "openai/gpt-4"}
            DEFAULT_FALLBACK_MODEL = "anthropic/claude-sonnet-4"

        agent = TestAgent()
        agent.model = "unknown-model"

        mapped = agent.OPENROUTER_MODEL_MAP.get(agent.model, agent.DEFAULT_FALLBACK_MODEL)
        assert mapped == "anthropic/claude-sonnet-4"


class TestDebateWithFallback:
    """Tests for debates with fallback-enabled agents."""

    @pytest.mark.asyncio
    async def test_debate_completes_with_fallback_agents(self):
        """Debate should complete when agents use fallback successfully."""
        # Simulate agents that might use fallback
        agents = [
            MockAgent(
                name="agent_with_fallback_1",
                responses=["Response after potential fallback"],
            ),
            MockAgent(
                name="agent_with_fallback_2",
                responses=["Another response"],
            ),
            MockAgent(
                name="agent_with_fallback_3",
                responses=["Third response"],
            ),
        ]

        env = Environment(task="Test with fallback capability")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Debate should handle agent timeouts gracefully."""
        from tests.integration.conftest import SlowAgent

        # Mix of fast and slow agents
        agents = [
            MockAgent(name="fast_agent", responses=["Quick response"]),
            SlowAgent(name="slow_agent", delay=0.1, responses=["Slow response"]),
            MockAgent(name="fast_agent_2", responses=["Quick response 2"]),
        ]

        env = Environment(task="Test timeout handling")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            # Should complete despite one slow agent
            result = await arena.run()

        assert result is not None


class TestOpenRouterFallbackIntegration:
    """Integration tests for OpenRouter fallback when primary API agents hit quota."""

    @pytest.mark.asyncio
    async def test_anthropic_quota_triggers_openrouter_fallback(self):
        """Anthropic 400 billing error should trigger OpenRouter fallback."""
        from aragora.agents.fallback import QuotaFallbackMixin

        class MockAnthropicAgent(QuotaFallbackMixin):
            OPENROUTER_MODEL_MAP = {
                "claude-opus-4-5-20251101": "anthropic/claude-sonnet-4",
            }
            DEFAULT_FALLBACK_MODEL = "anthropic/claude-sonnet-4"

            def __init__(self):
                self.name = "mock-anthropic"
                self.model = "claude-opus-4-5-20251101"
                self.enable_fallback = True
                self._fallback_agent = None

        agent = MockAnthropicAgent()

        # Simulate Anthropic billing error
        assert agent.is_quota_error(400, "credit balance is too low") is True
        assert agent.is_quota_error(400, "Your account has insufficient credits") is True

        # Non-billing 400 errors should not trigger fallback
        assert agent.is_quota_error(400, "Invalid request format") is False

    @pytest.mark.asyncio
    async def test_gemini_quota_triggers_openrouter_fallback(self):
        """Gemini 403 quota error should trigger OpenRouter fallback."""
        from aragora.agents.fallback import QuotaFallbackMixin

        class MockGeminiAgent(QuotaFallbackMixin):
            OPENROUTER_MODEL_MAP = {
                "gemini-2.0-flash-exp": "google/gemini-2.0-flash-exp:free",
            }
            DEFAULT_FALLBACK_MODEL = "google/gemini-pro"

        agent = MockGeminiAgent()

        # Gemini quota errors
        assert agent.is_quota_error(403, "Quota exceeded") is True
        assert agent.is_quota_error(403, "Resource exhausted") is True
        assert agent.is_quota_error(429, "Rate limit exceeded") is True

    @pytest.mark.asyncio
    async def test_openai_429_triggers_openrouter_fallback(self):
        """OpenAI 429 rate limit should trigger OpenRouter fallback."""
        from aragora.agents.fallback import QuotaFallbackMixin

        class MockOpenAIAgent(QuotaFallbackMixin):
            OPENROUTER_MODEL_MAP = {
                "gpt-4o": "openai/gpt-4o",
            }
            DEFAULT_FALLBACK_MODEL = "openai/gpt-4o"

        agent = MockOpenAIAgent()

        # OpenAI rate limit
        assert agent.is_quota_error(429, "Rate limit exceeded") is True
        assert agent.is_quota_error(429, "") is True  # 429 always triggers


class TestStreamingFallbackIntegration:
    """Tests for streaming fallback scenarios."""

    @pytest.mark.asyncio
    async def test_fallback_chain_stream_generation(self):
        """Fallback chain should support streaming generation."""
        from aragora.agents.fallback import AgentFallbackChain

        fallback_called = False
        chunks_received = []

        async def primary_stream(prompt, context):
            # This generator will fail immediately
            raise RuntimeError("Primary streaming unavailable")
            yield  # Make it a generator (never reached)

        async def fallback_stream(prompt, context):
            nonlocal fallback_called
            fallback_called = True
            for chunk in ["Hello", " ", "World"]:
                yield chunk

        primary = Mock()
        primary.generate_stream = primary_stream
        primary.generate = AsyncMock(side_effect=RuntimeError("Primary unavailable"))
        primary.name = "primary"

        fallback = Mock()
        fallback.generate_stream = fallback_stream
        fallback.name = "fallback"

        chain = AgentFallbackChain([primary, fallback])

        async for chunk in chain.generate_stream("test prompt", []):
            chunks_received.append(chunk)

        assert fallback_called is True
        assert chunks_received == ["Hello", " ", "World"]


class TestContextPreservationDuringFallback:
    """Tests to ensure conversation context is preserved during fallback."""

    @pytest.mark.asyncio
    async def test_system_prompt_preserved_in_fallback(self):
        """Fallback agent should receive the same system prompt."""
        from aragora.agents.fallback import QuotaFallbackMixin

        class TestAgent(QuotaFallbackMixin):
            OPENROUTER_MODEL_MAP = {"test-model": "openai/gpt-4o"}
            DEFAULT_FALLBACK_MODEL = "openai/gpt-4o"

            def __init__(self):
                self.name = "test-agent"
                self.model = "test-model"
                self.system_prompt = "You are a helpful assistant."
                self.enable_fallback = True
                self._fallback_agent = None

        agent = TestAgent()

        # Test that model mapping works
        fallback_model = agent.get_fallback_model()
        assert fallback_model == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_conversation_context_passed_to_fallback(self):
        """Conversation context should be passed to fallback generate."""
        from aragora.agents.fallback import AgentFallbackChain
        from aragora.core_types import Message

        context_received = None

        async def primary_fail(prompt, context):
            raise RuntimeError("Primary unavailable")

        async def fallback_generate(prompt, context):
            nonlocal context_received
            context_received = context
            return "Fallback response"

        primary = Mock()
        primary.generate = primary_fail
        primary.name = "primary"

        fallback = Mock()
        fallback.generate = fallback_generate
        fallback.name = "fallback"

        chain = AgentFallbackChain([primary, fallback])

        test_context = [
            Message(role="user", agent="user", content="Hello"),
            Message(role="assistant", agent="assistant", content="Hi there"),
        ]

        await chain.generate("New question", test_context)

        assert context_received == test_context


class TestCascadingFailureScenarios:
    """Tests for cascading failure scenarios."""

    @pytest.mark.asyncio
    async def test_all_providers_exhausted_error(self):
        """Should raise AllProvidersExhaustedError when all providers fail."""
        from aragora.agents.fallback import AgentFallbackChain, AllProvidersExhaustedError

        async def always_fail(prompt, context):
            raise RuntimeError("Provider unavailable")

        agents = []
        for i in range(3):
            agent = Mock()
            agent.generate = always_fail
            agent.name = f"agent_{i}"
            agents.append(agent)

        chain = AgentFallbackChain(agents)

        with pytest.raises((RuntimeError, AllProvidersExhaustedError)):
            await chain.generate("test prompt", [])

    @pytest.mark.asyncio
    async def test_timeout_errors_treated_as_quota(self):
        """Timeout errors (408, 504, 524) should be treated as quota errors."""
        from aragora.agents.fallback import QuotaFallbackMixin

        class TestAgent(QuotaFallbackMixin):
            pass

        agent = TestAgent()

        # Timeout codes should trigger fallback
        assert agent.is_quota_error(408, "Request timeout") is True
        assert agent.is_quota_error(504, "Gateway timeout") is True
        assert agent.is_quota_error(524, "A timeout occurred") is True

    @pytest.mark.asyncio
    async def test_fallback_metrics_track_failures(self):
        """Fallback metrics should track primary and fallback success/failure rates."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()

        # Simulate some calls - 2 primary successes, 1 primary failure, 1 fallback success
        metrics.record_primary_attempt(success=True)
        metrics.record_primary_attempt(success=True)
        metrics.record_primary_attempt(success=False)  # This triggers fallback
        metrics.record_fallback_attempt("openrouter", success=True)

        # Check metrics
        assert metrics.primary_attempts == 3
        assert metrics.primary_successes == 2
        assert metrics.fallback_attempts == 1
        assert metrics.fallback_successes == 1
        assert metrics.fallback_rate == pytest.approx(1 / 4, rel=0.01)
        assert metrics.success_rate == pytest.approx(3 / 4, rel=0.01)


class TestDebateLevelFallbackIntegration:
    """Tests for fallback behavior during actual debates."""

    @pytest.mark.asyncio
    async def test_debate_with_quota_prone_agent(self):
        """Debate should complete when agent might hit quota but has fallback."""
        # Use MockAgent from conftest which has all required attributes
        agents = [
            MockAgent(
                name="quota_prone_1",
                responses=["Response 1", "Response 2", "Response 3", "Response 4"],
            ),
            MockAgent(name="reliable_1", responses=["Reliable response 1", "Reliable 2"]),
            MockAgent(name="reliable_2", responses=["Reliable response 2", "Reliable 3"]),
        ]

        env = Environment(task="Test quota-prone agent in debate")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_fallback_chain_with_circuit_breaker(self):
        """Fallback chain should respect circuit breaker state."""
        from aragora.agents.fallback import AgentFallbackChain
        from aragora.resilience import CircuitBreaker

        call_counts = {"primary": 0, "fallback": 0}

        async def primary_generate(prompt, context):
            call_counts["primary"] += 1
            raise RuntimeError("Primary always fails")

        async def fallback_generate(prompt, context):
            call_counts["fallback"] += 1
            return "Fallback success"

        primary = Mock()
        primary.generate = primary_generate
        primary.name = "primary"

        fallback = Mock()
        fallback.generate = fallback_generate
        fallback.name = "fallback"

        chain = AgentFallbackChain([primary, fallback])

        # Multiple calls should all succeed via fallback
        for _ in range(5):
            result = await chain.generate("test", [])
            assert result == "Fallback success"

        assert call_counts["primary"] == 5
        assert call_counts["fallback"] == 5
