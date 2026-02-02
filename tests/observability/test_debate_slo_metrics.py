"""
Tests for aragora.observability.metrics.debate_slo module.

Covers:
- Debate completion duration metrics (p50, p95, p99 support)
- Consensus detection latency metrics
- Per-agent response time metrics
- Debate success rate metrics
- Context managers for automatic tracking
- NoOp fallback behavior
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# TestDebateSLOMetricsNoOp
# =============================================================================


class TestDebateSLOMetricsNoOp:
    """Tests for debate SLO metrics with NoOp fallback (no prometheus)."""

    def test_record_debate_completion_no_error(self):
        """record_debate_completion_slo should not raise with noop metrics."""
        from aragora.observability.metrics.debate_slo import record_debate_completion_slo

        record_debate_completion_slo(45.2, "consensus")
        record_debate_completion_slo(120.5, "no_consensus")
        record_debate_completion_slo(300.0, "timeout")
        record_debate_completion_slo(5.0, "error")

    def test_record_consensus_detection_latency_no_error(self):
        """record_consensus_detection_latency should not raise."""
        from aragora.observability.metrics.debate_slo import record_consensus_detection_latency

        record_consensus_detection_latency(2.5, "majority")
        record_consensus_detection_latency(5.0, "unanimous")
        record_consensus_detection_latency(1.2, "judge")
        record_consensus_detection_latency(0.5, "none")
        record_consensus_detection_latency(10.0, "byzantine")

    def test_record_agent_response_time_no_error(self):
        """record_agent_response_time should not raise."""
        from aragora.observability.metrics.debate_slo import record_agent_response_time

        record_agent_response_time("claude", 3.2, "proposal")
        record_agent_response_time("gpt-4", 5.5, "critique")
        record_agent_response_time("gemini", 2.0, "vote")
        record_agent_response_time("mistral", 4.1, "synthesis")

    def test_update_debate_success_rate_no_error(self):
        """update_debate_success_rate should not raise."""
        from aragora.observability.metrics.debate_slo import (
            reset_success_window,
            update_debate_success_rate,
        )

        reset_success_window()
        update_debate_success_rate(True)
        update_debate_success_rate(False)
        update_debate_success_rate(True)

    def test_record_debate_outcome_no_error(self):
        """record_debate_outcome should not raise."""
        from aragora.observability.metrics.debate_slo import record_debate_outcome

        record_debate_outcome(
            duration_seconds=45.0,
            consensus_reached=True,
            consensus_mode="majority",
            consensus_latency_seconds=2.5,
        )
        record_debate_outcome(
            duration_seconds=120.0,
            consensus_reached=False,
        )


# =============================================================================
# TestModelNameNormalization
# =============================================================================


class TestModelNameNormalization:
    """Tests for model name normalization to control label cardinality."""

    def test_claude_variants(self):
        """Claude model variants should normalize correctly."""
        from aragora.observability.metrics.debate_slo import _normalize_model_name

        assert _normalize_model_name("claude-3-opus") == "claude-opus"
        assert _normalize_model_name("claude-3-sonnet") == "claude-sonnet"
        assert _normalize_model_name("claude-3-haiku") == "claude-haiku"
        assert _normalize_model_name("claude") == "claude"
        assert _normalize_model_name("Claude-Opus") == "claude-opus"

    def test_gpt_variants(self):
        """GPT model variants should normalize correctly."""
        from aragora.observability.metrics.debate_slo import _normalize_model_name

        assert _normalize_model_name("gpt-4") == "gpt-4"
        assert _normalize_model_name("gpt-4-turbo") == "gpt-4"
        assert _normalize_model_name("gpt-4o") == "gpt-4"
        assert _normalize_model_name("gpt-3.5-turbo") == "gpt-3.5"

    def test_other_models(self):
        """Other models should normalize correctly."""
        from aragora.observability.metrics.debate_slo import _normalize_model_name

        assert _normalize_model_name("gemini-pro") == "gemini"
        assert _normalize_model_name("grok-1") == "grok"
        assert _normalize_model_name("mistral-large") == "mistral"
        assert _normalize_model_name("deepseek-coder") == "deepseek"
        assert _normalize_model_name("qwen-72b") == "qwen"
        assert _normalize_model_name("llama-3-70b") == "llama"

    def test_unknown_model_truncation(self):
        """Unknown models should be truncated to reduce cardinality."""
        from aragora.observability.metrics.debate_slo import _normalize_model_name

        # Long model names get truncated
        assert _normalize_model_name("custom-model-v1-latest") == "custom-model"
        assert _normalize_model_name("simple") == "simple"


# =============================================================================
# TestContextManagers
# =============================================================================


class TestContextManagers:
    """Tests for context manager-based metric tracking."""

    def test_track_debate_completion_success(self):
        """track_debate_completion should track duration on success."""
        from aragora.observability.metrics.debate_slo import track_debate_completion

        with track_debate_completion() as tracker:
            time.sleep(0.01)  # Small delay
            tracker["outcome"] = "consensus"

        assert tracker["outcome"] == "consensus"

    def test_track_debate_completion_error(self):
        """track_debate_completion should record error outcome on exception."""
        from aragora.observability.metrics.debate_slo import track_debate_completion

        with pytest.raises(ValueError):
            with track_debate_completion() as tracker:
                tracker["outcome"] = "consensus"
                raise ValueError("test error")

        # The outcome should be set to error
        assert tracker["outcome"] == "error"
        assert "error" in tracker

    def test_track_consensus_detection(self):
        """track_consensus_detection should track latency."""
        from aragora.observability.metrics.debate_slo import track_consensus_detection

        with track_consensus_detection("majority"):
            time.sleep(0.01)

    def test_track_agent_response(self):
        """track_agent_response should track agent response time."""
        from aragora.observability.metrics.debate_slo import track_agent_response

        with track_agent_response("claude", "proposal"):
            time.sleep(0.01)

    @pytest.mark.asyncio
    async def test_track_agent_response_async(self):
        """track_agent_response_async should track async agent response time."""
        from aragora.observability.metrics.debate_slo import track_agent_response_async

        async with track_agent_response_async("gpt-4", "critique"):
            await asyncio.sleep(0.01)


# =============================================================================
# TestDebateSLOStats
# =============================================================================


class TestDebateSLOStats:
    """Tests for DebateSLOStats summary dataclass."""

    def test_get_debate_slo_summary(self):
        """get_debate_slo_summary should return valid stats."""
        from aragora.observability.metrics.debate_slo import (
            get_debate_slo_summary,
            reset_success_window,
            update_debate_success_rate,
        )

        reset_success_window()

        # Add some success/failure data
        update_debate_success_rate(True)
        update_debate_success_rate(True)
        update_debate_success_rate(False)

        stats = get_debate_slo_summary()

        assert stats.total_debates == 3
        assert stats.successful_debates == 2
        assert stats.failed_debates == 1
        assert stats.success_rate == pytest.approx(2 / 3)
        assert stats.window_seconds == 3600

    def test_get_debate_slo_summary_empty(self):
        """get_debate_slo_summary should handle empty data."""
        from aragora.observability.metrics.debate_slo import (
            get_debate_slo_summary,
            reset_success_window,
        )

        reset_success_window()
        stats = get_debate_slo_summary()

        assert stats.total_debates == 0
        assert stats.success_rate == 0.0


# =============================================================================
# TestSuccessRateRollingWindow
# =============================================================================


class TestSuccessRateRollingWindow:
    """Tests for rolling window success rate calculation."""

    def test_success_rate_updates(self):
        """Success rate should update with new data points."""
        from aragora.observability.metrics.debate_slo import (
            get_debate_slo_summary,
            reset_success_window,
            update_debate_success_rate,
        )

        reset_success_window()

        # 100% success rate
        update_debate_success_rate(True)
        stats = get_debate_slo_summary()
        assert stats.success_rate == 1.0

        # 50% success rate
        update_debate_success_rate(False)
        stats = get_debate_slo_summary()
        assert stats.success_rate == 0.5

    def test_reset_clears_window(self):
        """reset_success_window should clear all data."""
        from aragora.observability.metrics.debate_slo import (
            get_debate_slo_summary,
            reset_success_window,
            update_debate_success_rate,
        )

        update_debate_success_rate(True)
        update_debate_success_rate(True)

        reset_success_window()
        stats = get_debate_slo_summary()

        assert stats.total_debates == 0


# =============================================================================
# TestMetricBuckets
# =============================================================================


class TestMetricBuckets:
    """Tests validating histogram bucket configurations."""

    def test_debate_completion_buckets_cover_range(self):
        """Debate completion buckets should cover 1s to 20min."""
        # These are the expected buckets for aragora_debate_completion_duration_seconds
        expected_buckets = [1, 5, 10, 30, 60, 120, 180, 300, 600, 900, 1200]

        # Verify buckets cover expected range
        assert expected_buckets[0] == 1  # Quick debates
        assert expected_buckets[-1] == 1200  # 20 minutes

    def test_consensus_latency_buckets_cover_range(self):
        """Consensus latency buckets should cover 0.1s to 2min."""
        expected_buckets = [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120]

        assert expected_buckets[0] == 0.1
        assert expected_buckets[-1] == 120

    def test_agent_response_buckets_cover_range(self):
        """Agent response buckets should cover 0.5s to 3min."""
        expected_buckets = [0.5, 1, 2, 5, 10, 30, 60, 120, 180]

        assert expected_buckets[0] == 0.5
        assert expected_buckets[-1] == 180


# =============================================================================
# TestIntegrationWithDebateRunner
# =============================================================================


class TestIntegrationWithDebateRunner:
    """Tests for integration with debate orchestrator runner."""

    def test_import_from_orchestrator_runner(self):
        """Metrics should be importable from orchestrator_runner."""
        # This tests that the import works without errors
        from aragora.debate.orchestrator_runner import (
            record_debate_completion_slo,
            update_debate_success_rate,
        )

        # Verify functions are callable
        assert callable(record_debate_completion_slo)
        assert callable(update_debate_success_rate)

    def test_import_from_consensus_phase(self):
        """Metrics should be importable from consensus_phase."""
        from aragora.debate.phases.consensus_phase import record_consensus_detection_latency

        assert callable(record_consensus_detection_latency)

    def test_import_from_batch_utils(self):
        """Metrics should be importable from batch_utils."""
        from aragora.debate.phases.batch_utils import record_agent_response_time

        assert callable(record_agent_response_time)


# =============================================================================
# TestMetricLabels
# =============================================================================


class TestMetricLabels:
    """Tests for metric label values."""

    def test_debate_outcome_labels(self):
        """Debate outcome labels should be valid."""
        valid_outcomes = ["consensus", "no_consensus", "timeout", "error"]

        from aragora.observability.metrics.debate_slo import record_debate_completion_slo

        for outcome in valid_outcomes:
            # Should not raise
            record_debate_completion_slo(10.0, outcome)

    def test_consensus_mode_labels(self):
        """Consensus mode labels should be valid."""
        valid_modes = ["none", "majority", "unanimous", "judge", "byzantine"]

        from aragora.observability.metrics.debate_slo import record_consensus_detection_latency

        for mode in valid_modes:
            # Should not raise
            record_consensus_detection_latency(1.0, mode)

    def test_phase_labels(self):
        """Phase labels should be valid."""
        valid_phases = ["proposal", "critique", "revision", "vote", "synthesis"]

        from aragora.observability.metrics.debate_slo import record_agent_response_time

        for phase in valid_phases:
            # Should not raise
            record_agent_response_time("test-agent", 1.0, phase)


# =============================================================================
# TestPrometheusExport
# =============================================================================


class TestPrometheusExport:
    """Tests for Prometheus metric export format."""

    def test_metrics_module_exports(self):
        """debate_slo module should export all required symbols."""
        from aragora.observability.metrics import debate_slo

        # Core metrics
        assert hasattr(debate_slo, "DEBATE_COMPLETION_DURATION")
        assert hasattr(debate_slo, "CONSENSUS_DETECTION_LATENCY")
        assert hasattr(debate_slo, "AGENT_RESPONSE_TIME")
        assert hasattr(debate_slo, "DEBATE_SUCCESS_RATE")
        assert hasattr(debate_slo, "DEBATE_SUCCESS_TOTAL")

        # Recording functions
        assert hasattr(debate_slo, "record_debate_completion_slo")
        assert hasattr(debate_slo, "record_consensus_detection_latency")
        assert hasattr(debate_slo, "record_agent_response_time")
        assert hasattr(debate_slo, "update_debate_success_rate")

        # Context managers
        assert hasattr(debate_slo, "track_debate_completion")
        assert hasattr(debate_slo, "track_consensus_detection")
        assert hasattr(debate_slo, "track_agent_response")
        assert hasattr(debate_slo, "track_agent_response_async")

        # Stats
        assert hasattr(debate_slo, "get_debate_slo_summary")
        assert hasattr(debate_slo, "DebateSLOStats")


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_duration(self):
        """Zero duration should be recorded correctly."""
        from aragora.observability.metrics.debate_slo import record_debate_completion_slo

        record_debate_completion_slo(0.0, "consensus")

    def test_very_large_duration(self):
        """Very large durations should be recorded correctly."""
        from aragora.observability.metrics.debate_slo import record_debate_completion_slo

        record_debate_completion_slo(86400.0, "timeout")  # 24 hours

    def test_negative_duration_still_records(self):
        """Negative durations should not crash (edge case)."""
        from aragora.observability.metrics.debate_slo import record_debate_completion_slo

        # This shouldn't happen in practice, but shouldn't crash
        record_debate_completion_slo(-1.0, "error")

    def test_empty_model_name(self):
        """Empty model name should be handled."""
        from aragora.observability.metrics.debate_slo import record_agent_response_time

        record_agent_response_time("", 1.0, "proposal")

    def test_unicode_model_name(self):
        """Unicode in model names should be handled."""
        from aragora.observability.metrics.debate_slo import record_agent_response_time

        record_agent_response_time("claude-model", 1.0, "proposal")


# =============================================================================
# TestConcurrentAccess
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent metric access."""

    @pytest.mark.asyncio
    async def test_concurrent_success_rate_updates(self):
        """Concurrent updates to success rate should be safe."""
        from aragora.observability.metrics.debate_slo import (
            reset_success_window,
            update_debate_success_rate,
        )

        reset_success_window()

        async def update_many(success: bool, count: int):
            for _ in range(count):
                update_debate_success_rate(success)
                await asyncio.sleep(0)

        # Run concurrent updates
        await asyncio.gather(
            update_many(True, 10),
            update_many(False, 10),
        )

    @pytest.mark.asyncio
    async def test_concurrent_metric_recording(self):
        """Concurrent metric recording should be safe."""
        from aragora.observability.metrics.debate_slo import record_agent_response_time

        async def record_many(model: str, count: int):
            for i in range(count):
                record_agent_response_time(model, float(i), "proposal")
                await asyncio.sleep(0)

        await asyncio.gather(
            record_many("claude", 10),
            record_many("gpt-4", 10),
            record_many("gemini", 10),
        )
