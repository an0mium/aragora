"""
Comprehensive tests for aragora/server/handlers/social/telemetry.py.

Covers:
- FallbackSocialMetrics class initialization
- All record_* functions (fallback path)
- All record_* functions (prometheus path, mocked)
- with_webhook_metrics decorator (sync, success + error paths)
- with_api_metrics decorator (async, success + error paths)
- get_metrics_summary (fallback and prometheus)
- reset_fallback_metrics
- Edge cases: latency sample trimming, debates_in_progress floor, multiple platforms
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.social.telemetry import (
    FallbackSocialMetrics,
    _get_fallback_metrics,
    get_metrics_summary,
    record_api_call,
    record_api_latency,
    record_command,
    record_debate_completed,
    record_debate_failed,
    record_debate_started,
    record_error,
    record_gauntlet_completed,
    record_gauntlet_failed,
    record_gauntlet_started,
    record_message,
    record_vote,
    record_webhook_latency,
    record_webhook_request,
    reset_fallback_metrics,
    with_api_metrics,
    with_webhook_metrics,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Reset fallback metrics before and after each test."""
    reset_fallback_metrics()
    yield
    reset_fallback_metrics()


@pytest.fixture
def force_fallback(monkeypatch):
    """Force the fallback path by setting PROMETHEUS_AVAILABLE=False."""
    import aragora.server.handlers.social.telemetry as telemetry_mod

    monkeypatch.setattr(telemetry_mod, "PROMETHEUS_AVAILABLE", False)


@pytest.fixture
def force_prometheus(monkeypatch):
    """Force the prometheus path by setting PROMETHEUS_AVAILABLE=True."""
    import aragora.server.handlers.social.telemetry as telemetry_mod

    monkeypatch.setattr(telemetry_mod, "PROMETHEUS_AVAILABLE", True)


# ============================================================================
# FallbackSocialMetrics Class
# ============================================================================


class TestFallbackSocialMetrics:
    """Tests for the FallbackSocialMetrics dataclass."""

    def test_init_creates_empty_dicts(self):
        """All metric stores start empty."""
        fb = FallbackSocialMetrics()
        assert fb.webhook_requests == {}
        assert fb.webhook_latencies == {}
        assert fb.messages == {}
        assert fb.commands == {}
        assert fb.debates_started == {}
        assert fb.debates_completed == {}
        assert fb.debates_failed == {}
        assert fb.debates_in_progress == {}
        assert fb.gauntlets_started == {}
        assert fb.gauntlets_completed == {}
        assert fb.gauntlets_failed == {}
        assert fb.votes == {}
        assert fb.errors == {}
        assert fb.api_calls == {}
        assert fb.api_latencies == {}


class TestGetFallbackMetrics:
    """Tests for the _get_fallback_metrics singleton."""

    def test_returns_same_instance(self):
        """Calling twice returns the same instance."""
        fb1 = _get_fallback_metrics()
        fb2 = _get_fallback_metrics()
        assert fb1 is fb2

    def test_reset_clears_singleton(self):
        """reset_fallback_metrics() forces a new instance next call."""
        fb1 = _get_fallback_metrics()
        reset_fallback_metrics()
        fb2 = _get_fallback_metrics()
        assert fb1 is not fb2


# ============================================================================
# record_webhook_request
# ============================================================================


class TestRecordWebhookRequest:
    """Tests for record_webhook_request."""

    def test_fallback_increments_counter(self, force_fallback):
        """Fallback path increments nested dict counter."""
        record_webhook_request("telegram", "success")
        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["success"] == 1

    def test_fallback_increments_multiple_statuses(self, force_fallback):
        """Different statuses are tracked independently."""
        record_webhook_request("telegram", "success")
        record_webhook_request("telegram", "success")
        record_webhook_request("telegram", "error")
        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["success"] == 2
        assert fb.webhook_requests["telegram"]["error"] == 1

    def test_fallback_multiple_platforms(self, force_fallback):
        """Different platforms are tracked independently."""
        record_webhook_request("telegram", "success")
        record_webhook_request("whatsapp", "success")
        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["success"] == 1
        assert fb.webhook_requests["whatsapp"]["success"] == 1

    def test_prometheus_path_calls_counter(self, force_prometheus):
        """Prometheus path calls the Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_WEBHOOK_REQUESTS_TOTAL",
            mock_counter,
        ):
            record_webhook_request("telegram", "success")
        mock_counter.labels.assert_called_once_with(platform="telegram", status="success")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_webhook_latency
# ============================================================================


class TestRecordWebhookLatency:
    """Tests for record_webhook_latency."""

    def test_fallback_stores_latency(self, force_fallback):
        """Fallback path stores latency in a list."""
        record_webhook_latency("telegram", 0.05)
        record_webhook_latency("telegram", 0.10)
        fb = _get_fallback_metrics()
        assert fb.webhook_latencies["telegram"] == [0.05, 0.10]

    def test_fallback_trims_at_1000_samples(self, force_fallback):
        """Samples are trimmed to the last 1000."""
        for i in range(1005):
            record_webhook_latency("telegram", float(i))
        fb = _get_fallback_metrics()
        assert len(fb.webhook_latencies["telegram"]) == 1000
        # Kept last 1000 (indices 5..1004)
        assert fb.webhook_latencies["telegram"][0] == 5.0
        assert fb.webhook_latencies["telegram"][-1] == 1004.0

    def test_prometheus_path_observes_histogram(self, force_prometheus):
        """Prometheus path calls Histogram.labels().observe()."""
        mock_hist = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_WEBHOOK_LATENCY",
            mock_hist,
        ):
            record_webhook_latency("telegram", 0.123)
        mock_hist.labels.assert_called_once_with(platform="telegram")
        mock_hist.labels.return_value.observe.assert_called_once_with(0.123)


# ============================================================================
# record_message
# ============================================================================


class TestRecordMessage:
    """Tests for record_message."""

    def test_fallback_increments(self, force_fallback):
        """Fallback path increments message count by type."""
        record_message("telegram", "text")
        record_message("telegram", "command")
        record_message("telegram", "text")
        fb = _get_fallback_metrics()
        assert fb.messages["telegram"]["text"] == 2
        assert fb.messages["telegram"]["command"] == 1

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_MESSAGES_TOTAL",
            mock_counter,
        ):
            record_message("whatsapp", "interactive")
        mock_counter.labels.assert_called_once_with(platform="whatsapp", message_type="interactive")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_command
# ============================================================================


class TestRecordCommand:
    """Tests for record_command."""

    def test_fallback_increments(self, force_fallback):
        """Fallback path increments command counts."""
        record_command("telegram", "start")
        record_command("telegram", "help")
        record_command("telegram", "start")
        fb = _get_fallback_metrics()
        assert fb.commands["telegram"]["start"] == 2
        assert fb.commands["telegram"]["help"] == 1

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_COMMANDS_TOTAL",
            mock_counter,
        ):
            record_command("telegram", "debate")
        mock_counter.labels.assert_called_once_with(platform="telegram", command="debate")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_debate_started
# ============================================================================


class TestRecordDebateStarted:
    """Tests for record_debate_started."""

    def test_fallback_increments_started_and_in_progress(self, force_fallback):
        """Both started count and in-progress gauge increment."""
        record_debate_started("telegram")
        record_debate_started("telegram")
        fb = _get_fallback_metrics()
        assert fb.debates_started["telegram"] == 2
        assert fb.debates_in_progress["telegram"] == 2

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path increments both counter and gauge."""
        mock_started = MagicMock()
        mock_in_progress = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_DEBATES_STARTED",
            mock_started,
        ), patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_DEBATES_IN_PROGRESS",
            mock_in_progress,
        ):
            record_debate_started("whatsapp")
        mock_started.labels.assert_called_once_with(platform="whatsapp")
        mock_started.labels.return_value.inc.assert_called_once()
        mock_in_progress.labels.assert_called_once_with(platform="whatsapp")
        mock_in_progress.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_debate_completed
# ============================================================================


class TestRecordDebateCompleted:
    """Tests for record_debate_completed."""

    def test_fallback_consensus_reached(self, force_fallback):
        """consensus_reached=True maps to 'reached'."""
        record_debate_started("telegram")
        record_debate_completed("telegram", consensus_reached=True)
        fb = _get_fallback_metrics()
        assert fb.debates_completed["telegram"]["reached"] == 1
        assert fb.debates_in_progress["telegram"] == 0

    def test_fallback_consensus_not_reached(self, force_fallback):
        """consensus_reached=False maps to 'not_reached'."""
        record_debate_started("telegram")
        record_debate_completed("telegram", consensus_reached=False)
        fb = _get_fallback_metrics()
        assert fb.debates_completed["telegram"]["not_reached"] == 1
        assert fb.debates_in_progress["telegram"] == 0

    def test_fallback_in_progress_floor_at_zero(self, force_fallback):
        """in_progress never goes below 0."""
        record_debate_completed("telegram", consensus_reached=True)
        fb = _get_fallback_metrics()
        assert fb.debates_in_progress["telegram"] == 0

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path increments completed and decrements in-progress."""
        mock_completed = MagicMock()
        mock_in_progress = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_DEBATES_COMPLETED",
            mock_completed,
        ), patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_DEBATES_IN_PROGRESS",
            mock_in_progress,
        ):
            record_debate_completed("telegram", consensus_reached=True)
        mock_completed.labels.assert_called_once_with(platform="telegram", consensus="reached")
        mock_completed.labels.return_value.inc.assert_called_once()
        mock_in_progress.labels.assert_called_once_with(platform="telegram")
        mock_in_progress.labels.return_value.dec.assert_called_once()


# ============================================================================
# record_debate_failed
# ============================================================================


class TestRecordDebateFailed:
    """Tests for record_debate_failed."""

    def test_fallback_increments_and_decrements(self, force_fallback):
        """Failed increments failed count and decrements in-progress."""
        record_debate_started("telegram")
        record_debate_failed("telegram")
        fb = _get_fallback_metrics()
        assert fb.debates_failed["telegram"] == 1
        assert fb.debates_in_progress["telegram"] == 0

    def test_fallback_in_progress_floor_at_zero(self, force_fallback):
        """in_progress never goes below 0 even without a prior start."""
        record_debate_failed("whatsapp")
        fb = _get_fallback_metrics()
        assert fb.debates_in_progress["whatsapp"] == 0

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path increments failed and decrements in-progress."""
        mock_failed = MagicMock()
        mock_in_progress = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_DEBATES_FAILED",
            mock_failed,
        ), patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_DEBATES_IN_PROGRESS",
            mock_in_progress,
        ):
            record_debate_failed("telegram")
        mock_failed.labels.assert_called_once_with(platform="telegram")
        mock_failed.labels.return_value.inc.assert_called_once()
        mock_in_progress.labels.assert_called_once_with(platform="telegram")
        mock_in_progress.labels.return_value.dec.assert_called_once()


# ============================================================================
# record_gauntlet_started
# ============================================================================


class TestRecordGauntletStarted:
    """Tests for record_gauntlet_started."""

    def test_fallback_increments(self, force_fallback):
        """Fallback path increments gauntlet started count."""
        record_gauntlet_started("telegram")
        record_gauntlet_started("telegram")
        fb = _get_fallback_metrics()
        assert fb.gauntlets_started["telegram"] == 2

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_GAUNTLETS_STARTED",
            mock_counter,
        ):
            record_gauntlet_started("whatsapp")
        mock_counter.labels.assert_called_once_with(platform="whatsapp")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_gauntlet_completed
# ============================================================================


class TestRecordGauntletCompleted:
    """Tests for record_gauntlet_completed."""

    def test_fallback_passed(self, force_fallback):
        """passed=True maps to 'passed'."""
        record_gauntlet_completed("telegram", passed=True)
        fb = _get_fallback_metrics()
        assert fb.gauntlets_completed["telegram"]["passed"] == 1

    def test_fallback_failed(self, force_fallback):
        """passed=False maps to 'failed'."""
        record_gauntlet_completed("telegram", passed=False)
        fb = _get_fallback_metrics()
        assert fb.gauntlets_completed["telegram"]["failed"] == 1

    def test_fallback_multiple(self, force_fallback):
        """Multiple completions accumulate correctly."""
        record_gauntlet_completed("telegram", passed=True)
        record_gauntlet_completed("telegram", passed=True)
        record_gauntlet_completed("telegram", passed=False)
        fb = _get_fallback_metrics()
        assert fb.gauntlets_completed["telegram"]["passed"] == 2
        assert fb.gauntlets_completed["telegram"]["failed"] == 1

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_GAUNTLETS_COMPLETED",
            mock_counter,
        ):
            record_gauntlet_completed("telegram", passed=True)
        mock_counter.labels.assert_called_once_with(platform="telegram", result="passed")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_gauntlet_failed
# ============================================================================


class TestRecordGauntletFailed:
    """Tests for record_gauntlet_failed."""

    def test_fallback_increments(self, force_fallback):
        """Fallback path increments gauntlet failure count."""
        record_gauntlet_failed("telegram")
        fb = _get_fallback_metrics()
        assert fb.gauntlets_failed["telegram"] == 1

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_GAUNTLETS_FAILED",
            mock_counter,
        ):
            record_gauntlet_failed("whatsapp")
        mock_counter.labels.assert_called_once_with(platform="whatsapp")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_vote
# ============================================================================


class TestRecordVote:
    """Tests for record_vote."""

    def test_fallback_increments(self, force_fallback):
        """Fallback path increments vote counts."""
        record_vote("telegram", "agree")
        record_vote("telegram", "disagree")
        record_vote("telegram", "agree")
        fb = _get_fallback_metrics()
        assert fb.votes["telegram"]["agree"] == 2
        assert fb.votes["telegram"]["disagree"] == 1

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_VOTES_TOTAL",
            mock_counter,
        ):
            record_vote("telegram", "agree")
        mock_counter.labels.assert_called_once_with(platform="telegram", vote="agree")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_error
# ============================================================================


class TestRecordError:
    """Tests for record_error."""

    def test_fallback_increments(self, force_fallback):
        """Fallback path increments error counts by type."""
        record_error("telegram", "json_parse")
        record_error("telegram", "auth")
        record_error("telegram", "json_parse")
        fb = _get_fallback_metrics()
        assert fb.errors["telegram"]["json_parse"] == 2
        assert fb.errors["telegram"]["auth"] == 1

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_ERRORS_TOTAL",
            mock_counter,
        ):
            record_error("telegram", "api_call")
        mock_counter.labels.assert_called_once_with(platform="telegram", error_type="api_call")
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_api_call
# ============================================================================


class TestRecordApiCall:
    """Tests for record_api_call."""

    def test_fallback_increments(self, force_fallback):
        """Fallback path tracks nested platform > method > status counts."""
        record_api_call("telegram", "sendMessage", "success")
        record_api_call("telegram", "sendMessage", "success")
        record_api_call("telegram", "sendMessage", "error")
        record_api_call("telegram", "answerCallback", "success")
        fb = _get_fallback_metrics()
        assert fb.api_calls["telegram"]["sendMessage"]["success"] == 2
        assert fb.api_calls["telegram"]["sendMessage"]["error"] == 1
        assert fb.api_calls["telegram"]["answerCallback"]["success"] == 1

    def test_fallback_multiple_platforms(self, force_fallback):
        """Different platforms are tracked independently."""
        record_api_call("telegram", "sendMessage", "success")
        record_api_call("whatsapp", "sendMessage", "success")
        fb = _get_fallback_metrics()
        assert fb.api_calls["telegram"]["sendMessage"]["success"] == 1
        assert fb.api_calls["whatsapp"]["sendMessage"]["success"] == 1

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Counter.labels().inc()."""
        mock_counter = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_API_CALLS_TOTAL",
            mock_counter,
        ):
            record_api_call("telegram", "sendMessage", "success")
        mock_counter.labels.assert_called_once_with(
            platform="telegram", method="sendMessage", status="success"
        )
        mock_counter.labels.return_value.inc.assert_called_once()


# ============================================================================
# record_api_latency
# ============================================================================


class TestRecordApiLatency:
    """Tests for record_api_latency."""

    def test_fallback_stores_samples(self, force_fallback):
        """Fallback path stores latency samples per platform/method."""
        record_api_latency("telegram", "sendMessage", 0.1)
        record_api_latency("telegram", "sendMessage", 0.2)
        record_api_latency("telegram", "answerCallback", 0.05)
        fb = _get_fallback_metrics()
        assert fb.api_latencies["telegram"]["sendMessage"] == [0.1, 0.2]
        assert fb.api_latencies["telegram"]["answerCallback"] == [0.05]

    def test_fallback_trims_at_1000_samples(self, force_fallback):
        """Samples per method are trimmed to the last 1000."""
        for i in range(1005):
            record_api_latency("telegram", "sendMessage", float(i))
        fb = _get_fallback_metrics()
        assert len(fb.api_latencies["telegram"]["sendMessage"]) == 1000
        assert fb.api_latencies["telegram"]["sendMessage"][0] == 5.0
        assert fb.api_latencies["telegram"]["sendMessage"][-1] == 1004.0

    def test_prometheus_path(self, force_prometheus):
        """Prometheus path calls Histogram.labels().observe()."""
        mock_hist = MagicMock()
        with patch(
            "aragora.server.handlers.social.telemetry.SOCIAL_API_LATENCY",
            mock_hist,
        ):
            record_api_latency("telegram", "sendMessage", 0.42)
        mock_hist.labels.assert_called_once_with(platform="telegram", method="sendMessage")
        mock_hist.labels.return_value.observe.assert_called_once_with(0.42)


# ============================================================================
# with_webhook_metrics decorator
# ============================================================================


class TestWithWebhookMetrics:
    """Tests for the with_webhook_metrics decorator."""

    def test_success_records_metrics(self, force_fallback):
        """Successful call records success status and latency."""

        @with_webhook_metrics("telegram")
        def my_handler():
            return "ok"

        result = my_handler()
        assert result == "ok"
        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["success"] == 1
        assert len(fb.webhook_latencies["telegram"]) == 1
        assert fb.webhook_latencies["telegram"][0] >= 0

    def test_error_records_metrics_and_reraises(self, force_fallback):
        """Exception records error status, error metric, and re-raises."""

        @with_webhook_metrics("telegram")
        def failing_handler():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            failing_handler()

        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["error"] == 1
        assert fb.errors["telegram"]["handler_exception"] == 1
        assert len(fb.webhook_latencies["telegram"]) == 1

    def test_records_latency_on_error(self, force_fallback):
        """Latency is recorded even when the handler raises."""

        @with_webhook_metrics("whatsapp")
        def slow_fail():
            raise RuntimeError("timeout")

        with pytest.raises(RuntimeError):
            slow_fail()

        fb = _get_fallback_metrics()
        assert len(fb.webhook_latencies["whatsapp"]) == 1

    def test_preserves_function_name(self):
        """Decorator preserves the wrapped function's __name__."""

        @with_webhook_metrics("telegram")
        def original_name():
            pass

        assert original_name.__name__ == "original_name"

    def test_passes_args_and_kwargs(self, force_fallback):
        """Decorator passes through positional and keyword arguments."""

        @with_webhook_metrics("telegram")
        def handler_with_args(a, b, key=None):
            return (a, b, key)

        result = handler_with_args(1, 2, key="val")
        assert result == (1, 2, "val")

    def test_only_catches_specific_exceptions(self, force_fallback):
        """Non-standard exceptions still propagate but are not counted as errors."""

        @with_webhook_metrics("telegram")
        def handler_keyboard_interrupt():
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            handler_keyboard_interrupt()

        fb = _get_fallback_metrics()
        # KeyboardInterrupt is not in the caught tuple, so status remains "success"
        assert fb.webhook_requests["telegram"]["success"] == 1
        assert "handler_exception" not in fb.errors.get("telegram", {})

    def test_key_error_counted(self, force_fallback):
        """KeyError is one of the caught exception types."""

        @with_webhook_metrics("telegram")
        def handler_key_error():
            raise KeyError("missing")

        with pytest.raises(KeyError):
            handler_key_error()

        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["error"] == 1

    def test_type_error_counted(self, force_fallback):
        """TypeError is one of the caught exception types."""

        @with_webhook_metrics("telegram")
        def handler_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            handler_type_error()

        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["error"] == 1

    def test_os_error_counted(self, force_fallback):
        """OSError is one of the caught exception types."""

        @with_webhook_metrics("telegram")
        def handler_os_error():
            raise OSError("disk full")

        with pytest.raises(OSError):
            handler_os_error()

        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["error"] == 1


# ============================================================================
# with_api_metrics decorator
# ============================================================================


class TestWithApiMetrics:
    """Tests for the with_api_metrics decorator (async)."""

    @pytest.mark.asyncio
    async def test_success_records_metrics(self, force_fallback):
        """Successful async call records success status and latency."""

        @with_api_metrics("telegram", "sendMessage")
        async def my_api_call():
            return {"ok": True}

        result = await my_api_call()
        assert result == {"ok": True}
        fb = _get_fallback_metrics()
        assert fb.api_calls["telegram"]["sendMessage"]["success"] == 1
        assert len(fb.api_latencies["telegram"]["sendMessage"]) == 1

    @pytest.mark.asyncio
    async def test_error_records_metrics_and_reraises(self, force_fallback):
        """Async exception records error status and re-raises."""

        @with_api_metrics("telegram", "sendMessage")
        async def failing_call():
            raise RuntimeError("API down")

        with pytest.raises(RuntimeError, match="API down"):
            await failing_call()

        fb = _get_fallback_metrics()
        assert fb.api_calls["telegram"]["sendMessage"]["error"] == 1
        assert len(fb.api_latencies["telegram"]["sendMessage"]) == 1

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """Decorator preserves the wrapped function's __name__."""

        @with_api_metrics("telegram", "sendMessage")
        async def original_async_name():
            pass

        assert original_async_name.__name__ == "original_async_name"

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self, force_fallback):
        """Decorator passes through positional and keyword arguments."""

        @with_api_metrics("telegram", "sendMessage")
        async def api_with_args(chat_id, text, parse_mode=None):
            return {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}

        result = await api_with_args(123, "hello", parse_mode="HTML")
        assert result == {"chat_id": 123, "text": "hello", "parse_mode": "HTML"}

    @pytest.mark.asyncio
    async def test_only_catches_specific_exceptions(self, force_fallback):
        """Non-standard exceptions propagate without setting error status."""

        @with_api_metrics("telegram", "sendMessage")
        async def handler_system_exit():
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            await handler_system_exit()

        fb = _get_fallback_metrics()
        # SystemExit is not caught, so status stays "success"
        assert fb.api_calls["telegram"]["sendMessage"]["success"] == 1

    @pytest.mark.asyncio
    async def test_value_error_counted(self, force_fallback):
        """ValueError is one of the caught exception types."""

        @with_api_metrics("whatsapp", "sendText")
        async def api_value_error():
            raise ValueError("invalid payload")

        with pytest.raises(ValueError):
            await api_value_error()

        fb = _get_fallback_metrics()
        assert fb.api_calls["whatsapp"]["sendText"]["error"] == 1

    @pytest.mark.asyncio
    async def test_key_error_counted(self, force_fallback):
        """KeyError is one of the caught exception types."""

        @with_api_metrics("whatsapp", "sendText")
        async def api_key_error():
            raise KeyError("missing field")

        with pytest.raises(KeyError):
            await api_key_error()

        fb = _get_fallback_metrics()
        assert fb.api_calls["whatsapp"]["sendText"]["error"] == 1


# ============================================================================
# get_metrics_summary
# ============================================================================


class TestGetMetricsSummary:
    """Tests for get_metrics_summary."""

    def test_fallback_returns_all_fields(self, force_fallback):
        """Fallback path returns all metric categories."""
        summary = get_metrics_summary()
        assert summary["prometheus_available"] is False
        assert "webhook_requests" in summary
        assert "messages" in summary
        assert "commands" in summary
        assert "debates_started" in summary
        assert "debates_completed" in summary
        assert "debates_failed" in summary
        assert "debates_in_progress" in summary
        assert "gauntlets_started" in summary
        assert "gauntlets_completed" in summary
        assert "gauntlets_failed" in summary
        assert "votes" in summary
        assert "errors" in summary
        assert "api_calls" in summary

    def test_fallback_reflects_recorded_data(self, force_fallback):
        """Summary contains data from recorded metrics."""
        record_webhook_request("telegram", "success")
        record_vote("whatsapp", "agree")
        summary = get_metrics_summary()
        assert summary["webhook_requests"]["telegram"]["success"] == 1
        assert summary["votes"]["whatsapp"]["agree"] == 1

    def test_prometheus_returns_export_endpoint(self, force_prometheus):
        """Prometheus path returns export endpoint info."""
        summary = get_metrics_summary()
        assert summary["prometheus_available"] is True
        assert summary["export_endpoint"] == "/metrics"


# ============================================================================
# reset_fallback_metrics
# ============================================================================


class TestResetFallbackMetrics:
    """Tests for reset_fallback_metrics."""

    def test_clears_all_data(self, force_fallback):
        """Resetting clears all accumulated metrics."""
        record_webhook_request("telegram", "success")
        record_vote("telegram", "agree")
        record_debate_started("whatsapp")
        reset_fallback_metrics()
        fb = _get_fallback_metrics()
        assert fb.webhook_requests == {}
        assert fb.votes == {}
        assert fb.debates_started == {}
        assert fb.debates_in_progress == {}


# ============================================================================
# Integration / Multi-step Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple record functions."""

    def test_full_debate_lifecycle(self, force_fallback):
        """Track a debate from start to successful completion."""
        record_webhook_request("telegram", "success")
        record_message("telegram", "command")
        record_command("telegram", "debate")
        record_debate_started("telegram")
        record_debate_completed("telegram", consensus_reached=True)

        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["success"] == 1
        assert fb.messages["telegram"]["command"] == 1
        assert fb.commands["telegram"]["debate"] == 1
        assert fb.debates_started["telegram"] == 1
        assert fb.debates_completed["telegram"]["reached"] == 1
        assert fb.debates_in_progress["telegram"] == 0

    def test_debate_failure_lifecycle(self, force_fallback):
        """Track a debate from start to failure."""
        record_debate_started("whatsapp")
        record_debate_started("whatsapp")
        assert _get_fallback_metrics().debates_in_progress["whatsapp"] == 2

        record_debate_failed("whatsapp")
        assert _get_fallback_metrics().debates_in_progress["whatsapp"] == 1

        record_debate_completed("whatsapp", consensus_reached=False)
        assert _get_fallback_metrics().debates_in_progress["whatsapp"] == 0

    def test_gauntlet_lifecycle(self, force_fallback):
        """Track a gauntlet from start to completion."""
        record_gauntlet_started("telegram")
        record_gauntlet_completed("telegram", passed=True)
        fb = _get_fallback_metrics()
        assert fb.gauntlets_started["telegram"] == 1
        assert fb.gauntlets_completed["telegram"]["passed"] == 1

    def test_gauntlet_error_lifecycle(self, force_fallback):
        """Track a gauntlet that errors out."""
        record_gauntlet_started("telegram")
        record_gauntlet_failed("telegram")
        fb = _get_fallback_metrics()
        assert fb.gauntlets_started["telegram"] == 1
        assert fb.gauntlets_failed["telegram"] == 1

    def test_mixed_platforms(self, force_fallback):
        """Multiple platforms tracked independently."""
        record_webhook_request("telegram", "success")
        record_webhook_request("whatsapp", "success")
        record_webhook_request("telegram", "error")
        record_debate_started("telegram")
        record_debate_started("whatsapp")
        record_debate_completed("telegram", True)

        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["success"] == 1
        assert fb.webhook_requests["telegram"]["error"] == 1
        assert fb.webhook_requests["whatsapp"]["success"] == 1
        assert fb.debates_in_progress["telegram"] == 0
        assert fb.debates_in_progress["whatsapp"] == 1

    def test_api_call_with_latency(self, force_fallback):
        """API calls and latencies are recorded together."""
        record_api_call("telegram", "sendMessage", "success")
        record_api_latency("telegram", "sendMessage", 0.25)
        record_api_call("telegram", "sendMessage", "error")
        record_api_latency("telegram", "sendMessage", 5.0)

        fb = _get_fallback_metrics()
        assert fb.api_calls["telegram"]["sendMessage"]["success"] == 1
        assert fb.api_calls["telegram"]["sendMessage"]["error"] == 1
        assert fb.api_latencies["telegram"]["sendMessage"] == [0.25, 5.0]

    @pytest.mark.asyncio
    async def test_decorator_combo_fallback(self, force_fallback):
        """Both decorators used together on different functions."""

        @with_webhook_metrics("telegram")
        def handle_webhook(data):
            return data

        @with_api_metrics("telegram", "sendMessage")
        async def send_reply(text):
            return {"ok": True, "text": text}

        handle_webhook({"update_id": 1})
        await send_reply("hello")

        fb = _get_fallback_metrics()
        assert fb.webhook_requests["telegram"]["success"] == 1
        assert fb.api_calls["telegram"]["sendMessage"]["success"] == 1

    def test_summary_after_various_operations(self, force_fallback):
        """get_metrics_summary returns correct snapshot of all data."""
        record_webhook_request("telegram", "success")
        record_message("telegram", "text")
        record_command("telegram", "help")
        record_debate_started("telegram")
        record_vote("telegram", "agree")
        record_error("telegram", "json_parse")
        record_gauntlet_started("telegram")

        summary = get_metrics_summary()
        assert summary["webhook_requests"]["telegram"]["success"] == 1
        assert summary["messages"]["telegram"]["text"] == 1
        assert summary["commands"]["telegram"]["help"] == 1
        assert summary["debates_started"]["telegram"] == 1
        assert summary["debates_in_progress"]["telegram"] == 1
        assert summary["votes"]["telegram"]["agree"] == 1
        assert summary["errors"]["telegram"]["json_parse"] == 1
        assert summary["gauntlets_started"]["telegram"] == 1
