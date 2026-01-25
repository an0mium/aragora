"""
Tests for decision pipeline metrics and observability.

Tests:
- Metrics initialization (Prometheus and no-op fallback)
- Request and result recording
- Cache hit/miss tracking
- Error tracking
- Track decision context manager
- Metrics retrieval and summary
"""

from __future__ import annotations

import time
from unittest import mock

import pytest


def reset_metrics_module():
    """Reset the decision metrics module state for clean testing."""
    import aragora.observability.decision_metrics as dm

    dm._initialized = False
    dm.DECISION_REQUESTS = None
    dm.DECISION_RESULTS = None
    dm.DECISION_LATENCY = None
    dm.DECISION_CONFIDENCE = None
    dm.DECISION_CACHE_HITS = None
    dm.DECISION_CACHE_MISSES = None
    dm.DECISION_DEDUP_HITS = None
    dm.DECISION_ACTIVE = None
    dm.DECISION_ERRORS = None
    dm.DECISION_CONSENSUS_RATE = None
    dm.DECISION_AGENTS_USED = None


def setup_all_noop_metrics():
    """Initialize all metrics with noop implementations."""
    import aragora.observability.decision_metrics as dm

    dm._init_noop_metrics()
    dm._initialized = True


@pytest.fixture(autouse=True)
def setup_noop_metrics():
    """Ensure noop metrics are initialized for tests that don't explicitly set up mocks."""
    import aragora.observability.decision_metrics as dm

    # Save original state
    orig_initialized = dm._initialized
    orig_requests = dm.DECISION_REQUESTS
    orig_results = dm.DECISION_RESULTS
    orig_latency = dm.DECISION_LATENCY
    orig_confidence = dm.DECISION_CONFIDENCE
    orig_cache_hits = dm.DECISION_CACHE_HITS
    orig_cache_misses = dm.DECISION_CACHE_MISSES
    orig_dedup = dm.DECISION_DEDUP_HITS
    orig_active = dm.DECISION_ACTIVE
    orig_errors = dm.DECISION_ERRORS
    orig_consensus = dm.DECISION_CONSENSUS_RATE
    orig_agents = dm.DECISION_AGENTS_USED

    # Initialize with noop metrics for test isolation
    dm._init_noop_metrics()
    dm._initialized = True

    yield

    # Restore original state after test
    dm._initialized = orig_initialized
    dm.DECISION_REQUESTS = orig_requests
    dm.DECISION_RESULTS = orig_results
    dm.DECISION_LATENCY = orig_latency
    dm.DECISION_CONFIDENCE = orig_confidence
    dm.DECISION_CACHE_HITS = orig_cache_hits
    dm.DECISION_CACHE_MISSES = orig_cache_misses
    dm.DECISION_DEDUP_HITS = orig_dedup
    dm.DECISION_ACTIVE = orig_active
    dm.DECISION_ERRORS = orig_errors
    dm.DECISION_CONSENSUS_RATE = orig_consensus
    dm.DECISION_AGENTS_USED = orig_agents


class TestMetricsInitialization:
    """Tests for metrics initialization."""

    def test_init_metrics_only_initializes_once(self):
        """Test that init_metrics only initializes once when already initialized."""
        import aragora.observability.decision_metrics as dm

        dm._initialized = True
        original_requests = dm.DECISION_REQUESTS

        # Should return True immediately without reinitializing
        result = dm._init_metrics()

        assert result is True
        assert dm.DECISION_REQUESTS is original_requests

    def test_init_noop_metrics_sets_all_metrics(self):
        """Test that noop initialization sets all metrics."""
        import aragora.observability.decision_metrics as dm

        reset_metrics_module()
        dm._init_noop_metrics()

        # All metrics should be set to noop
        assert dm.DECISION_REQUESTS is not None
        assert dm.DECISION_RESULTS is not None
        assert dm.DECISION_LATENCY is not None
        assert dm.DECISION_CONFIDENCE is not None
        assert dm.DECISION_CACHE_HITS is not None
        assert dm.DECISION_CACHE_MISSES is not None
        assert dm.DECISION_DEDUP_HITS is not None
        assert dm.DECISION_ACTIVE is not None
        assert dm.DECISION_ERRORS is not None
        assert dm.DECISION_CONSENSUS_RATE is not None
        assert dm.DECISION_AGENTS_USED is not None

    def test_noop_metric_accepts_any_method(self):
        """Test that NoopMetric silently accepts any method call."""
        import aragora.observability.decision_metrics as dm

        reset_metrics_module()
        dm._init_noop_metrics()

        # These should not raise
        dm.DECISION_REQUESTS.labels(foo="bar").inc()
        dm.DECISION_LATENCY.labels(x="y").observe(1.5)
        dm.DECISION_ACTIVE.labels(z="w").dec()
        dm.DECISION_CONFIDENCE.labels(a="b").observe(0.5)

    def test_noop_metric_chained_calls(self):
        """Test that NoopMetric supports chained method calls."""
        import aragora.observability.decision_metrics as dm

        reset_metrics_module()
        dm._init_noop_metrics()

        # Should support any chain of methods
        dm.DECISION_REQUESTS.labels(a="1").labels(b="2").inc()
        dm.DECISION_LATENCY.labels(x="y").observe(1.5)


class TestRecordDecisionRequest:
    """Tests for record_decision_request function."""

    def test_record_decision_request_basic(self):
        """Test basic request recording."""
        from aragora.observability.decision_metrics import record_decision_request

        # Should not raise
        record_decision_request(
            decision_type="debate",
            source="api",
            priority="high",
        )

    def test_record_decision_request_default_priority(self):
        """Test request recording with default priority."""
        from aragora.observability.decision_metrics import record_decision_request

        # Should not raise with default priority
        record_decision_request(
            decision_type="quick",
            source="slack",
        )

    def test_record_decision_request_increments_counters(self):
        """Test that request recording increments appropriate counters."""
        import aragora.observability.decision_metrics as dm

        # Use mock metrics
        mock_requests = mock.MagicMock()
        mock_active = mock.MagicMock()

        original_requests = dm.DECISION_REQUESTS
        original_active = dm.DECISION_ACTIVE

        try:
            dm.DECISION_REQUESTS = mock_requests
            dm.DECISION_ACTIVE = mock_active
            dm._initialized = True

            dm.record_decision_request("debate", "api", "normal")

            mock_requests.labels.assert_called_once_with(
                decision_type="debate",
                source="api",
                priority="normal",
            )
            mock_requests.labels().inc.assert_called_once()
            mock_active.labels().inc.assert_called_once()
        finally:
            dm.DECISION_REQUESTS = original_requests
            dm.DECISION_ACTIVE = original_active


class TestRecordDecisionResult:
    """Tests for record_decision_result function."""

    def test_record_decision_result_basic(self):
        """Test basic result recording."""
        from aragora.observability.decision_metrics import record_decision_result

        record_decision_result(
            decision_type="debate",
            source="api",
            success=True,
            confidence=0.85,
            duration_seconds=5.0,
        )

    def test_record_decision_result_with_all_params(self):
        """Test result recording with all parameters."""
        from aragora.observability.decision_metrics import record_decision_result

        record_decision_result(
            decision_type="debate",
            source="slack",
            success=True,
            confidence=0.95,
            duration_seconds=10.5,
            consensus_reached=True,
            cache_hit=True,
            dedup_hit=False,
            agent_count=5,
            error_type=None,
        )

    def test_record_decision_result_failure(self):
        """Test result recording for failed decision."""
        from aragora.observability.decision_metrics import record_decision_result

        record_decision_result(
            decision_type="debate",
            source="api",
            success=False,
            confidence=0.0,
            duration_seconds=1.0,
            error_type="TimeoutError",
        )

    def test_record_decision_result_tracks_latency(self):
        """Test that result recording tracks latency."""
        import aragora.observability.decision_metrics as dm

        mock_latency = mock.MagicMock()
        original_latency = dm.DECISION_LATENCY

        try:
            dm.DECISION_LATENCY = mock_latency
            dm._initialized = True

            dm.record_decision_result(
                decision_type="quick",
                source="webhook",
                success=True,
                confidence=0.7,
                duration_seconds=2.5,
            )

            mock_latency.labels.assert_called_with(
                decision_type="quick",
                source="webhook",
            )
            mock_latency.labels().observe.assert_called_with(2.5)
        finally:
            dm.DECISION_LATENCY = original_latency

    def test_record_decision_result_tracks_confidence(self):
        """Test that result recording tracks confidence."""
        import aragora.observability.decision_metrics as dm

        mock_confidence = mock.MagicMock()
        original_confidence = dm.DECISION_CONFIDENCE

        try:
            dm.DECISION_CONFIDENCE = mock_confidence
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.85,
                duration_seconds=1.0,
            )

            mock_confidence.labels.assert_called_with(decision_type="debate")
            mock_confidence.labels().observe.assert_called_with(0.85)
        finally:
            dm.DECISION_CONFIDENCE = original_confidence

    def test_record_decision_result_skips_zero_confidence(self):
        """Test that zero confidence is not recorded."""
        import aragora.observability.decision_metrics as dm

        mock_confidence = mock.MagicMock()
        original_confidence = dm.DECISION_CONFIDENCE

        try:
            dm.DECISION_CONFIDENCE = mock_confidence
            dm._initialized = True

            dm.record_decision_result(
                decision_type="quick",
                source="api",
                success=True,
                confidence=0.0,
                duration_seconds=1.0,
            )

            mock_confidence.labels().observe.assert_not_called()
        finally:
            dm.DECISION_CONFIDENCE = original_confidence

    def test_record_decision_result_cache_hit(self):
        """Test that cache hits are recorded."""
        import aragora.observability.decision_metrics as dm

        mock_cache_hits = mock.MagicMock()
        mock_cache_misses = mock.MagicMock()
        orig_hits = dm.DECISION_CACHE_HITS
        orig_misses = dm.DECISION_CACHE_MISSES

        try:
            dm.DECISION_CACHE_HITS = mock_cache_hits
            dm.DECISION_CACHE_MISSES = mock_cache_misses
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.8,
                duration_seconds=0.1,
                cache_hit=True,
            )

            mock_cache_hits.labels().inc.assert_called_once()
            mock_cache_misses.labels().inc.assert_not_called()
        finally:
            dm.DECISION_CACHE_HITS = orig_hits
            dm.DECISION_CACHE_MISSES = orig_misses

    def test_record_decision_result_cache_miss(self):
        """Test that cache misses are recorded."""
        import aragora.observability.decision_metrics as dm

        mock_cache_hits = mock.MagicMock()
        mock_cache_misses = mock.MagicMock()
        orig_hits = dm.DECISION_CACHE_HITS
        orig_misses = dm.DECISION_CACHE_MISSES

        try:
            dm.DECISION_CACHE_HITS = mock_cache_hits
            dm.DECISION_CACHE_MISSES = mock_cache_misses
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.8,
                duration_seconds=5.0,
                cache_hit=False,
            )

            mock_cache_misses.labels().inc.assert_called_once()
            mock_cache_hits.labels().inc.assert_not_called()
        finally:
            dm.DECISION_CACHE_HITS = orig_hits
            dm.DECISION_CACHE_MISSES = orig_misses


class TestRecordDecisionError:
    """Tests for record_decision_error function."""

    def test_record_decision_error(self):
        """Test error recording."""
        from aragora.observability.decision_metrics import record_decision_error

        record_decision_error(
            decision_type="debate",
            error_type="TimeoutError",
        )

    def test_record_decision_error_increments_counter(self):
        """Test that error recording increments error counter."""
        import aragora.observability.decision_metrics as dm

        mock_errors = mock.MagicMock()
        orig_errors = dm.DECISION_ERRORS

        try:
            dm.DECISION_ERRORS = mock_errors
            dm._initialized = True

            dm.record_decision_error("quick", "ValueError")

            mock_errors.labels.assert_called_once_with(
                decision_type="quick",
                error_type="ValueError",
            )
            mock_errors.labels().inc.assert_called_once()
        finally:
            dm.DECISION_ERRORS = orig_errors


class TestCacheMetrics:
    """Tests for cache hit/miss tracking functions."""

    def test_record_decision_cache_hit(self):
        """Test cache hit recording."""
        from aragora.observability.decision_metrics import record_decision_cache_hit

        record_decision_cache_hit("debate")

    def test_record_decision_cache_miss(self):
        """Test cache miss recording."""
        from aragora.observability.decision_metrics import record_decision_cache_miss

        record_decision_cache_miss("quick")

    def test_record_decision_dedup_hit(self):
        """Test dedup hit recording."""
        from aragora.observability.decision_metrics import record_decision_dedup_hit

        record_decision_dedup_hit("debate")


class TestTrackDecisionContextManager:
    """Tests for track_decision context manager."""

    def test_track_decision_basic_success(self):
        """Test basic successful decision tracking."""
        from aragora.observability.decision_metrics import track_decision

        with track_decision("debate", "api", "normal") as ctx:
            ctx["success"] = True
            ctx["confidence"] = 0.9

        # Context manager completes without error

    def test_track_decision_records_request_and_result(self):
        """Test that context manager records both request and result."""
        import aragora.observability.decision_metrics as dm

        with mock.patch.object(dm, "record_decision_request") as mock_request:
            with mock.patch.object(dm, "record_decision_result") as mock_result:
                with dm.track_decision("debate", "slack", "high") as ctx:
                    ctx["success"] = True
                    ctx["confidence"] = 0.85
                    ctx["consensus_reached"] = True

                mock_request.assert_called_once_with("debate", "slack", "high")
                mock_result.assert_called_once()

                # Verify result call args
                call_kwargs = mock_result.call_args[1]
                assert call_kwargs["decision_type"] == "debate"
                assert call_kwargs["source"] == "slack"
                assert call_kwargs["success"] is True
                assert call_kwargs["confidence"] == 0.85
                assert call_kwargs["consensus_reached"] is True

    def test_track_decision_records_duration(self):
        """Test that context manager tracks duration."""
        import aragora.observability.decision_metrics as dm

        with mock.patch.object(dm, "record_decision_request"):
            with mock.patch.object(dm, "record_decision_result") as mock_result:
                with dm.track_decision("quick", "api") as ctx:
                    time.sleep(0.05)  # Sleep 50ms
                    ctx["success"] = True

                call_kwargs = mock_result.call_args[1]
                # Duration should be at least 50ms
                assert call_kwargs["duration_seconds"] >= 0.05

    def test_track_decision_handles_exception(self):
        """Test that context manager handles exceptions properly."""
        import aragora.observability.decision_metrics as dm

        with mock.patch.object(dm, "record_decision_request"):
            with mock.patch.object(dm, "record_decision_result") as mock_result:
                with pytest.raises(ValueError):
                    with dm.track_decision("debate", "api") as ctx:
                        raise ValueError("test error")

                # Should still record result
                mock_result.assert_called_once()
                call_kwargs = mock_result.call_args[1]
                assert call_kwargs["success"] is False
                assert call_kwargs["error_type"] == "ValueError"

    def test_track_decision_default_context_values(self):
        """Test that context has sensible defaults."""
        import aragora.observability.decision_metrics as dm

        with mock.patch.object(dm, "record_decision_request"):
            with mock.patch.object(dm, "record_decision_result") as mock_result:
                with dm.track_decision("quick", "webhook") as ctx:
                    # Don't set any values
                    pass

                call_kwargs = mock_result.call_args[1]
                assert call_kwargs["success"] is True  # Default
                assert call_kwargs["confidence"] == 0.0  # Default
                assert call_kwargs["consensus_reached"] is False  # Default
                assert call_kwargs["cache_hit"] is False  # Default

    def test_track_decision_with_agent_count(self):
        """Test tracking decision with agent count."""
        import aragora.observability.decision_metrics as dm

        with mock.patch.object(dm, "record_decision_request"):
            with mock.patch.object(dm, "record_decision_result") as mock_result:
                with dm.track_decision("debate", "api") as ctx:
                    ctx["agent_count"] = 5

                call_kwargs = mock_result.call_args[1]
                assert call_kwargs["agent_count"] == 5


class TestGetDecisionMetrics:
    """Tests for get_decision_metrics function."""

    def test_get_decision_metrics_returns_dict(self):
        """Test that get_decision_metrics returns a dictionary."""
        from aragora.observability.decision_metrics import get_decision_metrics

        result = get_decision_metrics()

        assert isinstance(result, dict)

    def test_get_decision_metrics_with_prometheus(self):
        """Test that metrics are returned when prometheus is available."""
        from aragora.observability.decision_metrics import get_decision_metrics

        result = get_decision_metrics()

        # Should return dict (either metrics or error)
        assert isinstance(result, dict)


class TestGetDecisionSummary:
    """Tests for get_decision_summary function."""

    def test_get_decision_summary_returns_dict(self):
        """Test that get_decision_summary returns a dictionary."""
        from aragora.observability.decision_metrics import get_decision_summary

        result = get_decision_summary()

        assert isinstance(result, dict)

    def test_get_decision_summary_structure(self):
        """Test that summary has expected structure."""
        from aragora.observability.decision_metrics import get_decision_summary

        result = get_decision_summary()

        # If no error, should have these keys
        if "error" not in result:
            expected_keys = {"requests", "latency", "confidence", "errors", "cache"}
            assert expected_keys.issubset(result.keys())


class TestIntegration:
    """Integration tests for decision metrics."""

    def test_full_decision_lifecycle(self):
        """Test complete decision lifecycle tracking."""
        from aragora.observability.decision_metrics import (
            record_decision_request,
            record_decision_result,
        )

        # Record request
        record_decision_request(
            decision_type="debate",
            source="api",
            priority="high",
        )

        # Record successful result
        record_decision_result(
            decision_type="debate",
            source="api",
            success=True,
            confidence=0.92,
            duration_seconds=12.5,
            consensus_reached=True,
            cache_hit=False,
            agent_count=5,
        )

    def test_multiple_decisions(self):
        """Test tracking multiple decisions."""
        from aragora.observability.decision_metrics import (
            record_decision_request,
            record_decision_result,
        )

        # Multiple requests
        for i in range(5):
            record_decision_request(
                decision_type="quick",
                source="webhook",
            )
            record_decision_result(
                decision_type="quick",
                source="webhook",
                success=True,
                confidence=0.8,
                duration_seconds=0.5,
            )

    def test_mixed_decision_types(self):
        """Test tracking different decision types."""
        from aragora.observability.decision_metrics import (
            record_decision_request,
            record_decision_result,
        )

        # Debate decision
        record_decision_request("debate", "slack", "normal")
        record_decision_result(
            decision_type="debate",
            source="slack",
            success=True,
            confidence=0.85,
            duration_seconds=30.0,
            consensus_reached=True,
        )

        # Quick decision
        record_decision_request("quick", "api", "high")
        record_decision_result(
            decision_type="quick",
            source="api",
            success=True,
            confidence=0.95,
            duration_seconds=0.2,
        )


class TestConsensusTracking:
    """Tests for debate consensus tracking."""

    def test_consensus_rate_tracked_for_debates(self):
        """Test that consensus rate is tracked for debate type."""
        import aragora.observability.decision_metrics as dm

        mock_consensus = mock.MagicMock()
        orig_consensus = dm.DECISION_CONSENSUS_RATE

        try:
            dm.DECISION_CONSENSUS_RATE = mock_consensus
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.9,
                duration_seconds=5.0,
                consensus_reached=True,
            )

            # Should track consensus rate (1.0 for reached)
            mock_consensus.observe.assert_called_once_with(1.0)
        finally:
            dm.DECISION_CONSENSUS_RATE = orig_consensus

    def test_consensus_rate_zero_when_not_reached(self):
        """Test that consensus rate is 0.0 when not reached."""
        import aragora.observability.decision_metrics as dm

        mock_consensus = mock.MagicMock()
        orig_consensus = dm.DECISION_CONSENSUS_RATE

        try:
            dm.DECISION_CONSENSUS_RATE = mock_consensus
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.5,
                duration_seconds=5.0,
                consensus_reached=False,
            )

            # Should track consensus rate (0.0 for not reached)
            mock_consensus.observe.assert_called_once_with(0.0)
        finally:
            dm.DECISION_CONSENSUS_RATE = orig_consensus

    def test_consensus_rate_not_tracked_for_non_debates(self):
        """Test that consensus rate is not tracked for non-debate types."""
        import aragora.observability.decision_metrics as dm

        mock_consensus = mock.MagicMock()
        orig_consensus = dm.DECISION_CONSENSUS_RATE

        try:
            dm.DECISION_CONSENSUS_RATE = mock_consensus
            dm._initialized = True

            dm.record_decision_result(
                decision_type="quick",
                source="api",
                success=True,
                confidence=0.9,
                duration_seconds=0.5,
            )

            # Should not track consensus rate for non-debate
            mock_consensus.observe.assert_not_called()
        finally:
            dm.DECISION_CONSENSUS_RATE = orig_consensus


class TestAgentsUsedTracking:
    """Tests for agent count tracking."""

    def test_agent_count_tracked_when_provided(self):
        """Test that agent count is tracked when > 0."""
        import aragora.observability.decision_metrics as dm

        mock_agents = mock.MagicMock()
        orig_agents = dm.DECISION_AGENTS_USED

        try:
            dm.DECISION_AGENTS_USED = mock_agents
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.85,
                duration_seconds=10.0,
                agent_count=5,
            )

            mock_agents.labels.assert_called_with(decision_type="debate")
            mock_agents.labels().observe.assert_called_with(5)
        finally:
            dm.DECISION_AGENTS_USED = orig_agents

    def test_agent_count_not_tracked_when_zero(self):
        """Test that agent count is not tracked when 0."""
        import aragora.observability.decision_metrics as dm

        mock_agents = mock.MagicMock()
        orig_agents = dm.DECISION_AGENTS_USED

        try:
            dm.DECISION_AGENTS_USED = mock_agents
            dm._initialized = True

            dm.record_decision_result(
                decision_type="quick",
                source="api",
                success=True,
                confidence=0.8,
                duration_seconds=0.5,
                agent_count=0,
            )

            mock_agents.labels().observe.assert_not_called()
        finally:
            dm.DECISION_AGENTS_USED = orig_agents


class TestDedupTracking:
    """Tests for deduplication hit tracking."""

    def test_dedup_hit_tracked_when_true(self):
        """Test that dedup hits are tracked when dedup_hit=True."""
        import aragora.observability.decision_metrics as dm

        mock_dedup = mock.MagicMock()
        orig_dedup = dm.DECISION_DEDUP_HITS

        try:
            dm.DECISION_DEDUP_HITS = mock_dedup
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.8,
                duration_seconds=0.1,
                dedup_hit=True,
            )

            mock_dedup.labels().inc.assert_called_once()
        finally:
            dm.DECISION_DEDUP_HITS = orig_dedup

    def test_dedup_hit_not_tracked_when_false(self):
        """Test that dedup hits are not tracked when dedup_hit=False."""
        import aragora.observability.decision_metrics as dm

        mock_dedup = mock.MagicMock()
        orig_dedup = dm.DECISION_DEDUP_HITS

        try:
            dm.DECISION_DEDUP_HITS = mock_dedup
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.8,
                duration_seconds=5.0,
                dedup_hit=False,
            )

            mock_dedup.labels().inc.assert_not_called()
        finally:
            dm.DECISION_DEDUP_HITS = orig_dedup


class TestErrorTracking:
    """Tests for error tracking in results."""

    def test_error_tracked_when_provided(self):
        """Test that errors are tracked when error_type is provided."""
        import aragora.observability.decision_metrics as dm

        mock_errors = mock.MagicMock()
        orig_errors = dm.DECISION_ERRORS

        try:
            dm.DECISION_ERRORS = mock_errors
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=False,
                confidence=0.0,
                duration_seconds=1.0,
                error_type="TimeoutError",
            )

            mock_errors.labels.assert_called_with(
                decision_type="debate",
                error_type="TimeoutError",
            )
            mock_errors.labels().inc.assert_called_once()
        finally:
            dm.DECISION_ERRORS = orig_errors

    def test_error_not_tracked_when_none(self):
        """Test that errors are not tracked when error_type is None."""
        import aragora.observability.decision_metrics as dm

        mock_errors = mock.MagicMock()
        orig_errors = dm.DECISION_ERRORS

        try:
            dm.DECISION_ERRORS = mock_errors
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.9,
                duration_seconds=5.0,
                error_type=None,
            )

            # Should not call inc() for errors
            mock_errors.labels().inc.assert_not_called()
        finally:
            dm.DECISION_ERRORS = orig_errors


class TestActiveDecisionTracking:
    """Tests for active decision gauge tracking."""

    def test_active_decremented_on_result(self):
        """Test that active gauge is decremented when result is recorded."""
        import aragora.observability.decision_metrics as dm

        mock_active = mock.MagicMock()
        orig_active = dm.DECISION_ACTIVE

        try:
            dm.DECISION_ACTIVE = mock_active
            dm._initialized = True

            dm.record_decision_result(
                decision_type="debate",
                source="api",
                success=True,
                confidence=0.9,
                duration_seconds=5.0,
            )

            mock_active.labels().dec.assert_called_once()
        finally:
            dm.DECISION_ACTIVE = orig_active
