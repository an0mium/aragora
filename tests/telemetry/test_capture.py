"""
Tests for the per-request telemetry capture pipeline.

Covers:
- RequestMetrics dataclass and cost calculation
- DebateSessionMetrics aggregation
- capture_model_request decorator (async and sync)
- Session lifecycle (start, get, end)
- Metrics sink registration and emission
- Token extraction from agent counters
- Fallback token estimation
- Error handling paths
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from aragora.telemetry.capture import (
    DebateSessionMetrics,
    RequestMetrics,
    _calculate_cost,
    capture_model_request,
    end_session,
    get_session_metrics,
    register_metrics_sink,
    reset_capture_state,
    start_session,
    unregister_metrics_sink,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global capture state before and after each test."""
    reset_capture_state()
    yield
    reset_capture_state()


# =============================================================================
# RequestMetrics
# =============================================================================


class TestRequestMetrics:
    def test_create_request_metrics(self):
        m = RequestMetrics(
            request_id="r1",
            provider="anthropic",
            model="claude-opus-4.6",
            agent_name="claude",
            operation="generate",
        )
        assert m.request_id == "r1"
        assert m.provider == "anthropic"
        assert m.success is True
        assert m.cost_usd == 0.0

    def test_complete_success(self):
        m = RequestMetrics(
            request_id="r1",
            provider="openai",
            model="gpt-4.1",
            agent_name="gpt",
            operation="generate",
            start_time=time.monotonic(),
        )
        m.complete(success=True, input_tokens=100, output_tokens=50)
        assert m.success is True
        assert m.input_tokens == 100
        assert m.output_tokens == 50
        assert m.total_tokens == 150
        assert m.latency_ms > 0
        assert m.cost_usd > 0

    def test_complete_failure(self):
        m = RequestMetrics(
            request_id="r2",
            provider="anthropic",
            model="claude-sonnet-4.6",
            agent_name="claude",
            operation="critique",
            start_time=time.monotonic(),
        )
        err = ValueError("test error")
        m.complete(success=False, error=err)
        assert m.success is False
        assert m.error_type == "ValueError"
        assert m.total_tokens == 0

    def test_to_dict(self):
        m = RequestMetrics(
            request_id="r3",
            provider="anthropic",
            model="claude-opus-4.6",
            agent_name="claude",
            operation="generate",
        )
        d = m.to_dict()
        assert d["request_id"] == "r3"
        assert d["provider"] == "anthropic"
        assert isinstance(d, dict)


# =============================================================================
# Cost Calculation
# =============================================================================


class TestCostCalculation:
    def test_calculate_cost_with_billing(self):
        cost = _calculate_cost("anthropic", "claude-opus-4.6", 1000, 500)
        assert cost > 0

    def test_calculate_cost_fallback(self):
        """When billing module calculation works, cost should be positive."""
        cost = _calculate_cost("unknown_provider", "unknown_model", 1000, 500)
        assert cost > 0

    def test_zero_tokens_zero_cost(self):
        cost = _calculate_cost("anthropic", "claude-opus-4.6", 0, 0)
        assert cost == 0.0


# =============================================================================
# DebateSessionMetrics
# =============================================================================


class TestDebateSessionMetrics:
    def test_create_session(self):
        s = DebateSessionMetrics(debate_id="d1")
        assert s.debate_id == "d1"
        assert s.total_requests == 0

    def test_record_request(self):
        s = DebateSessionMetrics(debate_id="d1")
        m = RequestMetrics(
            request_id="r1",
            provider="anthropic",
            model="claude-opus-4.6",
            agent_name="claude",
            operation="generate",
            start_time=time.monotonic(),
        )
        m.complete(success=True, input_tokens=500, output_tokens=200)
        s.record(m)

        assert s.total_requests == 1
        assert s.successful_requests == 1
        assert s.failed_requests == 0
        assert s.total_input_tokens == 500
        assert s.total_output_tokens == 200
        assert s.total_tokens == 700
        assert s.total_cost_usd > 0

    def test_record_multiple(self):
        s = DebateSessionMetrics(debate_id="d1")
        for i in range(3):
            m = RequestMetrics(
                request_id=f"r{i}",
                provider="openai",
                model="gpt-4.1",
                agent_name=f"agent-{i}",
                operation="generate",
                start_time=time.monotonic(),
            )
            m.complete(success=True, input_tokens=100, output_tokens=50)
            s.record(m)

        assert s.total_requests == 3
        assert s.total_input_tokens == 300
        assert s.total_output_tokens == 150

    def test_provider_stats(self):
        s = DebateSessionMetrics(debate_id="d1")
        for provider in ["anthropic", "openai", "anthropic"]:
            m = RequestMetrics(
                request_id="r",
                provider=provider,
                model="model",
                agent_name="agent",
                operation="generate",
                start_time=time.monotonic(),
            )
            m.complete(success=True, input_tokens=100, output_tokens=50)
            s.record(m)

        assert s.provider_stats["anthropic"].requests == 2
        assert s.provider_stats["openai"].requests == 1

    def test_model_stats(self):
        s = DebateSessionMetrics(debate_id="d1")
        m = RequestMetrics(
            request_id="r1",
            provider="anthropic",
            model="claude-opus-4.6",
            agent_name="agent",
            operation="generate",
            start_time=time.monotonic(),
        )
        m.complete(success=True, input_tokens=100, output_tokens=50)
        s.record(m)

        assert "claude-opus-4.6" in s.model_stats
        assert s.model_stats["claude-opus-4.6"].requests == 1

    def test_success_rate(self):
        s = DebateSessionMetrics(debate_id="d1")
        for success in [True, True, False]:
            m = RequestMetrics(
                request_id="r",
                provider="anthropic",
                model="model",
                agent_name="agent",
                operation="generate",
                start_time=time.monotonic(),
            )
            m.complete(success=success, input_tokens=10, output_tokens=5)
            s.record(m)

        assert s.success_rate == pytest.approx(2 / 3, abs=0.01)

    def test_avg_latency(self):
        s = DebateSessionMetrics(debate_id="d1")
        m = RequestMetrics(
            request_id="r1",
            provider="anthropic",
            model="model",
            agent_name="agent",
            operation="generate",
            start_time=time.monotonic(),
        )
        m.complete(success=True, input_tokens=10, output_tokens=5)
        s.record(m)

        assert s.avg_latency_ms > 0

    def test_empty_session_rates(self):
        s = DebateSessionMetrics(debate_id="d1")
        assert s.avg_latency_ms == 0.0
        assert s.success_rate == 0.0

    def test_summary(self):
        s = DebateSessionMetrics(debate_id="d1")
        m = RequestMetrics(
            request_id="r1",
            provider="anthropic",
            model="claude-opus-4.6",
            agent_name="agent",
            operation="generate",
            start_time=time.monotonic(),
        )
        m.complete(success=True, input_tokens=100, output_tokens=50)
        s.record(m)

        summary = s.summary()
        assert summary["debate_id"] == "d1"
        assert summary["total_requests"] == 1
        assert "anthropic" in summary["providers"]
        assert "claude-opus-4.6" in summary["models"]


# =============================================================================
# Session Lifecycle
# =============================================================================


class TestSessionLifecycle:
    def test_start_session(self):
        s = start_session("d1")
        assert s.debate_id == "d1"

    def test_get_session(self):
        start_session("d1")
        s = get_session_metrics("d1")
        assert s is not None
        assert s.debate_id == "d1"

    def test_get_missing_session(self):
        assert get_session_metrics("nonexistent") is None

    def test_end_session(self):
        start_session("d1")
        s = end_session("d1")
        assert s is not None
        assert s.end_time is not None
        # Session should be removed
        assert get_session_metrics("d1") is None

    def test_end_missing_session(self):
        assert end_session("nonexistent") is None


# =============================================================================
# Metrics Sinks
# =============================================================================


class TestMetricsSinks:
    def test_register_and_emit(self):
        collected = []
        register_metrics_sink(collected.append)

        # Use the decorator to emit
        class FakeAgent:
            name = "test-agent"
            model = "test-model"
            agent_type = "test"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="test")
            async def generate(self, prompt: str) -> str:
                self._total_tokens_in += 100
                self._total_tokens_out += 50
                return "response"

        agent = FakeAgent()
        asyncio.get_event_loop().run_until_complete(agent.generate("hello"))

        assert len(collected) == 1
        assert collected[0].provider == "test"
        assert collected[0].input_tokens == 100

    def test_unregister_sink(self):
        collected = []
        register_metrics_sink(collected.append)
        unregister_metrics_sink(collected.append)

        class FakeAgent:
            name = "test-agent"
            model = "test-model"
            agent_type = "test"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="test")
            async def generate(self, prompt: str) -> str:
                return "response"

        agent = FakeAgent()
        asyncio.get_event_loop().run_until_complete(agent.generate("hello"))
        assert len(collected) == 0

    def test_sink_error_does_not_propagate(self):
        def bad_sink(m: RequestMetrics) -> None:
            raise RuntimeError("sink error")

        register_metrics_sink(bad_sink)

        class FakeAgent:
            name = "test-agent"
            model = "test-model"
            agent_type = "test"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="test")
            async def generate(self, prompt: str) -> str:
                return "ok"

        agent = FakeAgent()
        # Should not raise
        result = asyncio.get_event_loop().run_until_complete(agent.generate("hello"))
        assert result == "ok"


# =============================================================================
# Decorator - Async
# =============================================================================


class TestCaptureModelRequestAsync:
    def test_captures_success(self):
        collected = []
        register_metrics_sink(collected.append)

        class FakeAgent:
            name = "claude"
            model = "claude-opus-4.6"
            agent_type = "anthropic"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="anthropic")
            async def generate(self, prompt: str) -> str:
                self._total_tokens_in += 200
                self._total_tokens_out += 100
                return "generated text"

        agent = FakeAgent()
        result = asyncio.get_event_loop().run_until_complete(agent.generate("test prompt"))

        assert result == "generated text"
        assert len(collected) == 1
        m = collected[0]
        assert m.success is True
        assert m.provider == "anthropic"
        assert m.model == "claude-opus-4.6"
        assert m.agent_name == "claude"
        assert m.input_tokens == 200
        assert m.output_tokens == 100
        assert m.latency_ms > 0
        assert m.cost_usd > 0

    def test_captures_failure(self):
        collected = []
        register_metrics_sink(collected.append)

        class FakeAgent:
            name = "claude"
            model = "claude-opus-4.6"
            agent_type = "anthropic"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="anthropic")
            async def generate(self, prompt: str) -> str:
                raise ValueError("API error")

        agent = FakeAgent()
        with pytest.raises(ValueError, match="API error"):
            asyncio.get_event_loop().run_until_complete(agent.generate("test"))

        assert len(collected) == 1
        m = collected[0]
        assert m.success is False
        assert m.error_type == "ValueError"

    def test_auto_provider_from_agent_type(self):
        collected = []
        register_metrics_sink(collected.append)

        class FakeAgent:
            name = "gpt"
            model = "gpt-4.1"
            agent_type = "openai"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request()  # No provider specified
            async def generate(self, prompt: str) -> str:
                return "ok"

        agent = FakeAgent()
        asyncio.get_event_loop().run_until_complete(agent.generate("test"))
        assert collected[0].provider == "openai"

    def test_session_integration(self):
        session = start_session("debate-1")

        class FakeAgent:
            name = "claude"
            model = "claude-opus-4.6"
            agent_type = "anthropic"
            _total_tokens_in = 0
            _total_tokens_out = 0
            _current_debate_id = "debate-1"

            @capture_model_request(provider="anthropic")
            async def generate(self, prompt: str) -> str:
                self._total_tokens_in += 100
                self._total_tokens_out += 50
                return "ok"

        agent = FakeAgent()
        asyncio.get_event_loop().run_until_complete(agent.generate("test"))

        assert session.total_requests == 1
        assert session.total_input_tokens == 100
        assert session.total_output_tokens == 50

    def test_fallback_token_estimation(self):
        collected = []
        register_metrics_sink(collected.append)

        class FakeAgent:
            name = "agent"
            model = "model"
            agent_type = "test"
            # No _total_tokens_in/out attributes

            @capture_model_request(provider="test")
            async def generate(self, prompt: str) -> str:
                return "a" * 100  # 100 chars ≈ 25 tokens

        agent = FakeAgent()
        asyncio.get_event_loop().run_until_complete(agent.generate("b" * 40))

        m = collected[0]
        assert m.output_tokens == 25  # 100 chars / 4
        assert m.input_tokens == 10  # 40 chars / 4

    def test_critique_operation(self):
        collected = []
        register_metrics_sink(collected.append)

        class FakeAgent:
            name = "critic"
            model = "model"
            agent_type = "anthropic"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="anthropic", operation="critique")
            async def critique(self, proposal: str) -> str:
                self._total_tokens_in += 300
                self._total_tokens_out += 150
                return "critique text"

        agent = FakeAgent()
        asyncio.get_event_loop().run_until_complete(agent.critique("proposal"))
        assert collected[0].operation == "critique"


# =============================================================================
# Decorator - Sync
# =============================================================================


class TestCaptureModelRequestSync:
    def test_sync_capture(self):
        collected = []
        register_metrics_sink(collected.append)

        class FakeAgent:
            name = "agent"
            model = "model"
            agent_type = "test"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="test")
            def generate(self, prompt: str) -> str:
                self._total_tokens_in += 50
                self._total_tokens_out += 25
                return "sync result"

        agent = FakeAgent()
        result = agent.generate("test")
        assert result == "sync result"
        assert len(collected) == 1
        assert collected[0].input_tokens == 50
        assert collected[0].success is True

    def test_sync_failure(self):
        collected = []
        register_metrics_sink(collected.append)

        class FakeAgent:
            name = "agent"
            model = "model"
            agent_type = "test"
            _total_tokens_in = 0
            _total_tokens_out = 0

            @capture_model_request(provider="test")
            def generate(self, prompt: str) -> str:
                raise RuntimeError("fail")

        agent = FakeAgent()
        with pytest.raises(RuntimeError):
            agent.generate("test")
        assert collected[0].success is False


# =============================================================================
# Module Exports
# =============================================================================


class TestModuleExports:
    def test_all_exports(self):
        from aragora.telemetry.capture import __all__

        expected = [
            "RequestMetrics",
            "DebateSessionMetrics",
            "RequestMetricsCollector",
            "capture_model_request",
            "get_session_metrics",
            "start_session",
            "end_session",
            "register_metrics_sink",
            "unregister_metrics_sink",
            "reset_capture_state",
        ]
        for name in expected:
            assert name in __all__

    def test_importable_from_telemetry_package(self):
        from aragora.telemetry import (
            RequestMetrics,
            DebateSessionMetrics,
            capture_model_request,
            start_session,
            end_session,
            get_session_metrics,
            register_metrics_sink,
            unregister_metrics_sink,
        )

        assert RequestMetrics is not None
        assert DebateSessionMetrics is not None
        assert callable(capture_model_request)
        assert callable(start_session)
        assert callable(end_session)
        assert callable(get_session_metrics)
        assert callable(register_metrics_sink)
        assert callable(unregister_metrics_sink)
