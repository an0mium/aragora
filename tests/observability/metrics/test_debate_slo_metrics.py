"""Tests for observability/metrics/debate_slo.py — debate SLO metrics."""

from unittest.mock import patch

import pytest

from aragora.observability.metrics import debate_slo as mod
from aragora.observability.metrics.debate_slo import (
    DebateSLOStats,
    _normalize_model_name,
    get_debate_slo_summary,
    init_debate_slo_metrics,
    record_agent_response_time,
    record_consensus_detection_latency,
    record_debate_completion_slo,
    record_debate_outcome,
    reset_success_window,
    track_agent_response,
    track_agent_response_async,
    track_consensus_detection,
    track_debate_completion,
    update_debate_success_rate,
)


@pytest.fixture(autouse=True)
def _reset_module():
    mod._initialized = False
    reset_success_window()
    yield
    mod._initialized = False
    reset_success_window()


@pytest.fixture()
def _init_noop():
    with patch("aragora.observability.metrics.debate_slo.get_metrics_enabled", return_value=False):
        init_debate_slo_metrics()


# =============================================================================
# Initialization
# =============================================================================


class TestInitialization:
    def test_init_noop(self, _init_noop):
        assert mod._initialized is True

    def test_init_returns_false_when_disabled(self):
        with patch(
            "aragora.observability.metrics.debate_slo.get_metrics_enabled", return_value=False
        ):
            result = init_debate_slo_metrics()
            assert result is False


# =============================================================================
# Model Name Normalization
# =============================================================================


class TestNormalizeModelName:
    def test_claude_opus(self):
        assert _normalize_model_name("claude-3-opus-20240229") == "claude-opus"

    def test_claude_sonnet(self):
        assert _normalize_model_name("claude-3.5-sonnet") == "claude-sonnet"

    def test_claude_haiku(self):
        assert _normalize_model_name("claude-3-haiku") == "claude-haiku"

    def test_claude_generic(self):
        assert _normalize_model_name("claude-2") == "claude"

    def test_gpt4(self):
        assert _normalize_model_name("gpt-4-turbo-preview") == "gpt-4"

    def test_gpt35(self):
        assert _normalize_model_name("gpt-3.5-turbo") == "gpt-3.5"

    def test_gemini(self):
        assert _normalize_model_name("gemini-1.5-pro") == "gemini"

    def test_grok(self):
        assert _normalize_model_name("grok-2") == "grok"

    def test_mistral(self):
        assert _normalize_model_name("mistral-large-latest") == "mistral"

    def test_deepseek(self):
        assert _normalize_model_name("deepseek-coder-v2") == "deepseek"

    def test_qwen(self):
        assert _normalize_model_name("qwen-72b-chat") == "qwen"

    def test_llama(self):
        assert _normalize_model_name("llama-3.1-70b") == "llama"

    def test_unknown_long_name(self):
        assert _normalize_model_name("custom-model-v3-beta") == "custom-model"

    def test_short_name(self):
        assert _normalize_model_name("mymodel") == "mymodel"


# =============================================================================
# Recording Functions
# =============================================================================


class TestRecordingFunctions:
    def test_record_debate_completion_consensus(self, _init_noop):
        record_debate_completion_slo(45.2, "consensus")

    def test_record_debate_completion_timeout(self, _init_noop):
        record_debate_completion_slo(120.0, "timeout")

    def test_record_debate_completion_error(self, _init_noop):
        record_debate_completion_slo(5.0, "error")

    def test_record_debate_completion_no_consensus(self, _init_noop):
        record_debate_completion_slo(60.0, "no_consensus")

    def test_record_consensus_detection_latency(self, _init_noop):
        record_consensus_detection_latency(2.5, "majority")

    def test_record_agent_response_time(self, _init_noop):
        record_agent_response_time("claude", 3.2, "proposal")

    def test_record_debate_outcome_consensus(self, _init_noop):
        record_debate_outcome(
            45.0, consensus_reached=True, consensus_mode="majority", consensus_latency_seconds=2.0
        )

    def test_record_debate_outcome_no_consensus(self, _init_noop):
        record_debate_outcome(60.0, consensus_reached=False)


# =============================================================================
# Success Rate Window
# =============================================================================


class TestSuccessRateWindow:
    def test_update_success_rate(self, _init_noop):
        update_debate_success_rate(True)
        update_debate_success_rate(True)
        update_debate_success_rate(False)
        stats = get_debate_slo_summary()
        assert stats.total_debates == 3
        assert stats.successful_debates == 2
        assert stats.failed_debates == 1
        assert abs(stats.success_rate - 2 / 3) < 0.01

    def test_empty_summary(self, _init_noop):
        stats = get_debate_slo_summary()
        assert stats.total_debates == 0
        assert stats.success_rate == 0.0


# =============================================================================
# Dataclass
# =============================================================================


class TestDebateSLOStats:
    def test_defaults(self):
        stats = DebateSLOStats()
        assert stats.total_debates == 0
        assert stats.success_rate == 0.0
        assert stats.window_seconds == 3600
        assert stats.last_updated is not None


# =============================================================================
# Context Managers
# =============================================================================


class TestContextManagers:
    def test_track_debate_completion(self, _init_noop):
        with track_debate_completion() as ctx:
            ctx["outcome"] = "consensus"

    def test_track_debate_completion_error(self, _init_noop):
        with pytest.raises(ValueError):
            with track_debate_completion():
                raise ValueError("debate failed")

    def test_track_consensus_detection(self, _init_noop):
        with track_consensus_detection("majority"):
            pass

    def test_track_agent_response_sync(self, _init_noop):
        with track_agent_response("claude", "proposal"):
            pass

    @pytest.mark.asyncio
    async def test_track_agent_response_async(self, _init_noop):
        async with track_agent_response_async("gpt-4", "critique"):
            pass
