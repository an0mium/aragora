"""
Tests for aragora.server.middleware.slo_tracking - SLO tracking middleware.

Tests cover:
- _record_request() counter updates (success, failure, debate)
- _record_latency() bounded sample collection
- _get_p99_latency() percentile calculation
- sync_slo_measurements() integration with observability.slo
- get_tracking_stats() statistics reporting
- slo_context() context manager (success, failure, SLO breach)
- track_slo() sync decorator (success, failure, metadata preservation)
- track_slo_async() async decorator (success, failure, metadata preservation)
- slo_middleware() HTTP handler decorator (success, failure, path extraction)
- Thread safety of counter operations
- Module __all__ exports
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import aragora.server.middleware.slo_tracking as slo_mod


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_counters():
    """Reset all module-level counters before each test."""
    slo_mod._total_requests = 0
    slo_mod._successful_requests = 0
    slo_mod._total_debates = 0
    slo_mod._successful_debates = 0
    slo_mod._recent_latencies.clear()
    yield
    # Clean up after as well
    slo_mod._total_requests = 0
    slo_mod._successful_requests = 0
    slo_mod._total_debates = 0
    slo_mod._successful_debates = 0
    slo_mod._recent_latencies.clear()


# ===========================================================================
# Test _record_request
# ===========================================================================


class TestRecordRequest:
    """Tests for _record_request() counter updates."""

    def test_successful_request_increments_both_counters(self):
        """Successful request should increment total and successful counters."""
        slo_mod._record_request(success=True)

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 1

    def test_failed_request_increments_only_total(self):
        """Failed request should increment total but not successful counter."""
        slo_mod._record_request(success=False)

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 0

    def test_debate_request_increments_debate_counters(self):
        """Debate request should also increment debate-specific counters."""
        slo_mod._record_request(success=True, is_debate=True)

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 1
        assert slo_mod._total_debates == 1
        assert slo_mod._successful_debates == 1

    def test_failed_debate_increments_debate_total_only(self):
        """Failed debate should increment debate total but not debate successful."""
        slo_mod._record_request(success=False, is_debate=True)

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 0
        assert slo_mod._total_debates == 1
        assert slo_mod._successful_debates == 0

    def test_non_debate_does_not_touch_debate_counters(self):
        """Non-debate request should not modify debate counters."""
        slo_mod._record_request(success=True, is_debate=False)

        assert slo_mod._total_debates == 0
        assert slo_mod._successful_debates == 0

    def test_multiple_requests_accumulate(self):
        """Multiple calls should accumulate correctly."""
        for _ in range(5):
            slo_mod._record_request(success=True)
        for _ in range(3):
            slo_mod._record_request(success=False)

        assert slo_mod._total_requests == 8
        assert slo_mod._successful_requests == 5


# ===========================================================================
# Test _record_latency
# ===========================================================================


class TestRecordLatency:
    """Tests for _record_latency() bounded sample collection."""

    def test_records_single_latency(self):
        """Should store a single latency sample."""
        slo_mod._record_latency(42.0)

        assert slo_mod._recent_latencies == [42.0]

    def test_records_multiple_latencies(self):
        """Should accumulate multiple latency samples."""
        slo_mod._record_latency(10.0)
        slo_mod._record_latency(20.0)
        slo_mod._record_latency(30.0)

        assert slo_mod._recent_latencies == [10.0, 20.0, 30.0]

    def test_truncates_to_max_samples(self):
        """Should keep only the most recent _max_latency_samples entries."""
        max_samples = slo_mod._max_latency_samples

        # Insert more than the max
        for i in range(max_samples + 50):
            slo_mod._record_latency(float(i))

        assert len(slo_mod._recent_latencies) == max_samples
        # The oldest 50 should be trimmed; first remaining value is 50
        assert slo_mod._recent_latencies[0] == 50.0
        assert slo_mod._recent_latencies[-1] == float(max_samples + 49)


# ===========================================================================
# Test _get_p99_latency
# ===========================================================================


class TestGetP99Latency:
    """Tests for _get_p99_latency() percentile calculation."""

    def test_returns_zero_when_empty(self):
        """Should return 0.0 when no latency samples exist."""
        assert slo_mod._get_p99_latency() == 0.0

    def test_single_sample_returns_that_sample_in_seconds(self):
        """With one sample the p99 is that sample converted to seconds."""
        slo_mod._record_latency(250.0)  # 250ms

        result = slo_mod._get_p99_latency()
        assert result == pytest.approx(0.25)  # 250ms -> 0.25s

    def test_p99_picks_high_percentile(self):
        """p99 of 100 ascending values should return a high value."""
        for i in range(1, 101):
            slo_mod._record_latency(float(i))

        result = slo_mod._get_p99_latency()
        # p99 index = int(100 * 0.99) = 99 -> value 100ms = 0.1s
        assert result == pytest.approx(0.1)

    def test_returns_seconds_not_milliseconds(self):
        """Result should be in seconds (input is milliseconds)."""
        slo_mod._record_latency(1000.0)  # 1000ms = 1s

        result = slo_mod._get_p99_latency()
        assert result == pytest.approx(1.0)


# ===========================================================================
# Test sync_slo_measurements
# ===========================================================================


class TestSyncSloMeasurements:
    """Tests for sync_slo_measurements() integration."""

    @patch("aragora.server.middleware.slo_tracking._record_measurement")
    def test_syncs_counters_and_p99(self, mock_record):
        """Should pass current counters and p99 to _record_measurement."""
        slo_mod._total_requests = 100
        slo_mod._successful_requests = 95
        slo_mod._total_debates = 10
        slo_mod._successful_debates = 9
        slo_mod._record_latency(500.0)

        slo_mod.sync_slo_measurements()

        mock_record.assert_called_once_with(
            total_requests=100,
            successful_requests=95,
            latency_p99=pytest.approx(0.5),
            total_debates=10,
            successful_debates=9,
        )

    @patch("aragora.server.middleware.slo_tracking._record_measurement")
    def test_syncs_zero_when_empty(self, mock_record):
        """Should pass zeros when no data has been collected."""
        slo_mod.sync_slo_measurements()

        mock_record.assert_called_once_with(
            total_requests=0,
            successful_requests=0,
            latency_p99=0.0,
            total_debates=0,
            successful_debates=0,
        )


# ===========================================================================
# Test get_tracking_stats
# ===========================================================================


class TestGetTrackingStats:
    """Tests for get_tracking_stats() statistics reporting."""

    def test_returns_all_expected_keys(self):
        """Stats dict should contain all expected keys."""
        stats = slo_mod.get_tracking_stats()

        expected_keys = {
            "total_requests",
            "successful_requests",
            "total_debates",
            "successful_debates",
            "recent_latency_samples",
            "p99_latency_ms",
        }
        assert set(stats.keys()) == expected_keys

    def test_reflects_recorded_data(self):
        """Stats should reflect the data that was recorded."""
        slo_mod._record_request(success=True, is_debate=True)
        slo_mod._record_request(success=False)
        slo_mod._record_latency(200.0)
        slo_mod._record_latency(400.0)

        stats = slo_mod.get_tracking_stats()

        assert stats["total_requests"] == 2
        assert stats["successful_requests"] == 1
        assert stats["total_debates"] == 1
        assert stats["successful_debates"] == 1
        assert stats["recent_latency_samples"] == 2

    def test_p99_latency_ms_is_in_milliseconds(self):
        """p99_latency_ms should be in milliseconds (not seconds)."""
        slo_mod._record_latency(500.0)

        stats = slo_mod.get_tracking_stats()

        # _get_p99_latency returns seconds, stats multiplies by 1000 -> ms
        assert stats["p99_latency_ms"] == pytest.approx(500.0)


# ===========================================================================
# Test slo_context
# ===========================================================================


class TestSloContext:
    """Tests for slo_context() context manager."""

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_records_success(self, mock_check):
        """Successful block should record a successful request."""
        mock_check.return_value = (True, "ok")

        with slo_mod.slo_context("km_query"):
            pass

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 1
        assert len(slo_mod._recent_latencies) == 1

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_records_failure_on_exception(self, mock_check):
        """Exception should record a failed request and re-raise."""
        mock_check.return_value = (True, "ok")

        with pytest.raises(ValueError, match="boom"):
            with slo_mod.slo_context("api_endpoint"):
                raise ValueError("boom")

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 0

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_debate_flag_propagated(self, mock_check):
        """is_debate flag should propagate to _record_request."""
        mock_check.return_value = (True, "ok")

        with slo_mod.slo_context("debate_execution", is_debate=True):
            pass

        assert slo_mod._total_debates == 1
        assert slo_mod._successful_debates == 1

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_logs_slo_violation(self, mock_check, caplog):
        """Should log a warning when SLO is breached."""
        mock_check.return_value = (False, "latency 5000ms EXCEEDS p99 SLO (2000ms)")

        import logging

        with caplog.at_level(logging.WARNING):
            with slo_mod.slo_context("api_endpoint"):
                pass

        assert any("slo_violation" in record.message for record in caplog.records)

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_check_called_with_operation_and_elapsed(self, mock_check):
        """check_latency_slo should be called with the operation and elapsed time."""
        mock_check.return_value = (True, "ok")

        with slo_mod.slo_context("km_query"):
            time.sleep(0.01)  # Ensure measurable latency

        mock_check.assert_called_once()
        args = mock_check.call_args
        assert args[0][0] == "km_query"
        assert args[0][1] > 0  # elapsed_ms should be positive


# ===========================================================================
# Test track_slo (sync decorator)
# ===========================================================================


class TestTrackSlo:
    """Tests for track_slo() sync decorator."""

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_returns_function_result(self, mock_check):
        """Decorated function should return its result."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo("api_endpoint")
        def my_handler():
            return {"status": "ok"}

        result = my_handler()

        assert result == {"status": "ok"}
        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 1

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_re_raises_exception(self, mock_check):
        """Decorated function should re-raise exceptions after recording."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo("api_endpoint")
        def failing():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            failing()

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 0

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_preserves_function_metadata(self, mock_check):
        """Should preserve function name and docstring."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo("api_endpoint")
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_debate_flag_propagated(self, mock_check):
        """is_debate=True should be reflected in debate counters."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo("debate_execution", is_debate=True)
        def run_debate():
            return "consensus"

        run_debate()

        assert slo_mod._total_debates == 1
        assert slo_mod._successful_debates == 1


# ===========================================================================
# Test track_slo_async (async decorator)
# ===========================================================================


class TestTrackSloAsync:
    """Tests for track_slo_async() async decorator."""

    @pytest.mark.asyncio
    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    async def test_returns_async_result(self, mock_check):
        """Decorated async function should return its result."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo_async("api_endpoint")
        async def my_handler():
            return {"data": "value"}

        result = await my_handler()

        assert result == {"data": "value"}
        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 1

    @pytest.mark.asyncio
    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    async def test_records_failure_on_exception(self, mock_check):
        """Async exception should record failure and re-raise."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo_async("api_endpoint")
        async def failing():
            raise KeyError("missing")

        with pytest.raises(KeyError, match="missing"):
            await failing()

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 0
        assert len(slo_mod._recent_latencies) == 1

    @pytest.mark.asyncio
    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    async def test_preserves_function_metadata(self, mock_check):
        """Should preserve async function name and docstring."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo_async("api_endpoint")
        async def my_async_fn():
            """Async docstring."""
            pass

        assert my_async_fn.__name__ == "my_async_fn"
        assert my_async_fn.__doc__ == "Async docstring."

    @pytest.mark.asyncio
    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    async def test_debate_flag_propagated(self, mock_check):
        """is_debate flag should reach the debate counters."""
        mock_check.return_value = (True, "ok")

        @slo_mod.track_slo_async("debate", is_debate=True)
        async def run_debate():
            return "done"

        await run_debate()

        assert slo_mod._total_debates == 1
        assert slo_mod._successful_debates == 1

    @pytest.mark.asyncio
    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    async def test_logs_slo_violation(self, mock_check, caplog):
        """Should log a warning when async operation breaches SLO."""
        mock_check.return_value = (False, "latency 9000ms EXCEEDS p99 SLO (2000ms)")

        import logging

        @slo_mod.track_slo_async("slow_op")
        async def slow():
            return True

        with caplog.at_level(logging.WARNING):
            await slow()

        assert any("slo_violation" in record.message for record in caplog.records)


# ===========================================================================
# Test slo_middleware (HTTP handler decorator)
# ===========================================================================


class TestSloMiddleware:
    """Tests for slo_middleware() HTTP handler decorator."""

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_successful_handler(self, mock_check):
        """Middleware should pass through successful handler results."""
        mock_check.return_value = (True, "ok")

        @slo_mod.slo_middleware
        def do_GET(self_handler):
            return "200 OK"

        handler = MagicMock()
        result = do_GET(handler)

        assert result == "200 OK"
        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 1
        assert len(slo_mod._recent_latencies) == 1

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_failed_handler_re_raises(self, mock_check):
        """Middleware should re-raise handler exceptions after recording."""
        mock_check.return_value = (True, "ok")

        @slo_mod.slo_middleware
        def do_POST(self_handler):
            raise RuntimeError("server error")

        handler = MagicMock()

        with pytest.raises(RuntimeError, match="server error"):
            do_POST(handler)

        assert slo_mod._total_requests == 1
        assert slo_mod._successful_requests == 0

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_extracts_path_for_violation_log(self, mock_check, caplog):
        """Should log the handler's .path attribute on SLO breach."""
        mock_check.return_value = (False, "latency EXCEEDS SLO")

        import logging

        @slo_mod.slo_middleware
        def do_GET(self_handler):
            return "ok"

        handler = MagicMock()
        handler.path = "/api/v1/debates"

        with caplog.at_level(logging.WARNING):
            do_GET(handler)

        assert any("/api/v1/debates" in record.message for record in caplog.records)

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_handles_missing_path_attribute(self, mock_check, caplog):
        """Should fall back to 'unknown' when handler has no .path."""
        mock_check.return_value = (False, "latency EXCEEDS SLO")

        import logging

        @slo_mod.slo_middleware
        def do_GET():  # No self argument
            return "ok"

        with caplog.at_level(logging.WARNING):
            do_GET()

        # Should not crash; should log 'unknown' for path
        assert any("slo_violation" in record.message for record in caplog.records)

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_preserves_function_metadata(self, mock_check):
        """Should preserve handler function name and docstring."""
        mock_check.return_value = (True, "ok")

        @slo_mod.slo_middleware
        def do_GET(self_handler):
            """Handle GET requests."""
            pass

        assert do_GET.__name__ == "do_GET"
        assert do_GET.__doc__ == "Handle GET requests."

    @patch("aragora.server.middleware.slo_tracking.check_latency_slo")
    def test_always_uses_api_endpoint_operation(self, mock_check):
        """Middleware should always check against 'api_endpoint' SLO."""
        mock_check.return_value = (True, "ok")

        @slo_mod.slo_middleware
        def do_DELETE(self_handler):
            return "deleted"

        handler = MagicMock()
        do_DELETE(handler)

        mock_check.assert_called_once()
        assert mock_check.call_args[0][0] == "api_endpoint"


# ===========================================================================
# Test Thread Safety
# ===========================================================================


class TestThreadSafety:
    """Tests for thread-safe counter operations."""

    def test_concurrent_record_requests(self):
        """Multiple threads recording requests should not lose data."""
        import threading

        errors = []

        def record_batch():
            try:
                for _ in range(100):
                    slo_mod._record_request(success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_batch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert slo_mod._total_requests == 1000
        assert slo_mod._successful_requests == 1000

    def test_concurrent_record_latencies(self):
        """Multiple threads recording latencies should not lose data."""
        import threading

        errors = []

        def record_batch():
            try:
                for i in range(100):
                    slo_mod._record_latency(float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_batch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # 5 threads * 100 = 500 samples, within 1000 max
        assert len(slo_mod._recent_latencies) == 500


# ===========================================================================
# Test Module Exports
# ===========================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_exist(self):
        """Every name in __all__ should be importable from the module."""
        for name in slo_mod.__all__:
            assert hasattr(slo_mod, name), f"{name} listed in __all__ but not defined"

    def test_expected_public_api(self):
        """__all__ should contain the expected public API."""
        expected = {
            "slo_context",
            "slo_middleware",
            "sync_slo_measurements",
            "track_slo",
            "track_slo_async",
            "get_tracking_stats",
        }
        assert expected == set(slo_mod.__all__)
