"""Tests for aragora.server.handlers.social.telemetry - Telemetry Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

from unittest.mock import MagicMock, patch

import pytest

# Try to import the module
try:
    from aragora.server.handlers.social import telemetry

    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    telemetry = None

pytestmark = pytest.mark.skipif(not MODULE_AVAILABLE, reason="telemetry module not available")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_fallback_metrics():
    """Reset fallback metrics state before each test."""
    try:
        from aragora.server.handlers.social import telemetry

        telemetry.reset_fallback_metrics()
    except Exception:
        pass
    yield


# ===========================================================================
# Webhook Request Metrics Tests
# ===========================================================================


class TestWebhookRequestMetrics:
    """Tests for webhook request metrics."""

    def test_record_webhook_request_success(self, reset_fallback_metrics):
        """Test recording successful webhook request."""
        from aragora.server.handlers.social.telemetry import record_webhook_request

        record_webhook_request("telegram", "success")
        # No exception means success

    def test_record_webhook_request_error(self, reset_fallback_metrics):
        """Test recording error webhook request."""
        from aragora.server.handlers.social.telemetry import record_webhook_request

        record_webhook_request("whatsapp", "error")
        # No exception means success

    def test_record_webhook_latency(self, reset_fallback_metrics):
        """Test recording webhook latency."""
        from aragora.server.handlers.social.telemetry import record_webhook_latency

        record_webhook_latency("telegram", 0.05)
        record_webhook_latency("whatsapp", 0.1)
        # No exception means success


# ===========================================================================
# Message Metrics Tests
# ===========================================================================


class TestMessageMetrics:
    """Tests for message metrics."""

    def test_record_message_text(self, reset_fallback_metrics):
        """Test recording text message."""
        from aragora.server.handlers.social.telemetry import record_message

        record_message("telegram", "text")
        # No exception means success

    def test_record_message_command(self, reset_fallback_metrics):
        """Test recording command message."""
        from aragora.server.handlers.social.telemetry import record_message

        record_message("telegram", "command")
        # No exception means success


# ===========================================================================
# Command Metrics Tests
# ===========================================================================


class TestCommandMetrics:
    """Tests for command metrics."""

    def test_record_command(self, reset_fallback_metrics):
        """Test recording command execution."""
        from aragora.server.handlers.social.telemetry import record_command

        record_command("telegram", "debate")
        record_command("whatsapp", "help")
        # No exception means success


# ===========================================================================
# Debate Metrics Tests
# ===========================================================================


class TestDebateMetrics:
    """Tests for debate metrics."""

    def test_record_debate_started(self, reset_fallback_metrics):
        """Test recording debate started."""
        from aragora.server.handlers.social.telemetry import record_debate_started

        record_debate_started("telegram")
        # No exception means success

    def test_record_debate_completed_consensus(self, reset_fallback_metrics):
        """Test recording debate completed with consensus."""
        from aragora.server.handlers.social.telemetry import record_debate_completed

        record_debate_completed("telegram", consensus_reached=True)
        # No exception means success

    def test_record_debate_completed_no_consensus(self, reset_fallback_metrics):
        """Test recording debate completed without consensus."""
        from aragora.server.handlers.social.telemetry import record_debate_completed

        record_debate_completed("whatsapp", consensus_reached=False)
        # No exception means success

    def test_record_debate_failed(self, reset_fallback_metrics):
        """Test recording debate failure."""
        from aragora.server.handlers.social.telemetry import record_debate_failed

        record_debate_failed("telegram")
        # No exception means success


# ===========================================================================
# Gauntlet Metrics Tests
# ===========================================================================


class TestGauntletMetrics:
    """Tests for gauntlet metrics."""

    def test_record_gauntlet_started(self, reset_fallback_metrics):
        """Test recording gauntlet started."""
        from aragora.server.handlers.social.telemetry import record_gauntlet_started

        record_gauntlet_started("telegram")
        # No exception means success

    def test_record_gauntlet_completed_passed(self, reset_fallback_metrics):
        """Test recording gauntlet completed with pass."""
        from aragora.server.handlers.social.telemetry import record_gauntlet_completed

        record_gauntlet_completed("telegram", passed=True)
        # No exception means success

    def test_record_gauntlet_completed_failed(self, reset_fallback_metrics):
        """Test recording gauntlet completed with fail."""
        from aragora.server.handlers.social.telemetry import record_gauntlet_completed

        record_gauntlet_completed("whatsapp", passed=False)
        # No exception means success

    def test_record_gauntlet_failed(self, reset_fallback_metrics):
        """Test recording gauntlet error failure."""
        from aragora.server.handlers.social.telemetry import record_gauntlet_failed

        record_gauntlet_failed("telegram")
        # No exception means success


# ===========================================================================
# Vote Metrics Tests
# ===========================================================================


class TestVoteMetrics:
    """Tests for vote metrics."""

    def test_record_vote_agree(self, reset_fallback_metrics):
        """Test recording agree vote."""
        from aragora.server.handlers.social.telemetry import record_vote

        record_vote("telegram", "agree")
        # No exception means success

    def test_record_vote_disagree(self, reset_fallback_metrics):
        """Test recording disagree vote."""
        from aragora.server.handlers.social.telemetry import record_vote

        record_vote("whatsapp", "disagree")
        # No exception means success


# ===========================================================================
# Error Metrics Tests
# ===========================================================================


class TestErrorMetrics:
    """Tests for error metrics."""

    def test_record_error(self, reset_fallback_metrics):
        """Test recording error."""
        from aragora.server.handlers.social.telemetry import record_error

        record_error("telegram", "json_parse")
        record_error("whatsapp", "api_call")
        # No exception means success


# ===========================================================================
# API Call Metrics Tests
# ===========================================================================


class TestAPICallMetrics:
    """Tests for API call metrics."""

    def test_record_api_call(self, reset_fallback_metrics):
        """Test recording API call."""
        from aragora.server.handlers.social.telemetry import record_api_call

        record_api_call("telegram", "sendMessage", "success")
        record_api_call("whatsapp", "sendMessage", "error")
        # No exception means success

    def test_record_api_latency(self, reset_fallback_metrics):
        """Test recording API latency."""
        from aragora.server.handlers.social.telemetry import record_api_latency

        record_api_latency("telegram", "sendMessage", 0.25)
        # No exception means success


# ===========================================================================
# Metrics Summary Tests
# ===========================================================================


class TestMetricsSummary:
    """Tests for metrics summary."""

    def test_get_metrics_summary(self, reset_fallback_metrics):
        """Test getting metrics summary."""
        from aragora.server.handlers.social.telemetry import get_metrics_summary

        summary = get_metrics_summary()
        assert isinstance(summary, dict)
        assert "prometheus_available" in summary


# ===========================================================================
# Module Exports Tests
# ===========================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_functions_exported(self):
        """Test all expected functions are exported."""
        from aragora.server.handlers.social import telemetry

        expected = [
            "record_webhook_request",
            "record_webhook_latency",
            "record_message",
            "record_command",
            "record_debate_started",
            "record_debate_completed",
            "record_debate_failed",
            "record_gauntlet_started",
            "record_gauntlet_completed",
            "record_gauntlet_failed",
            "record_vote",
            "record_error",
            "record_api_call",
            "record_api_latency",
            "get_metrics_summary",
            "reset_fallback_metrics",
        ]
        for func_name in expected:
            assert hasattr(telemetry, func_name)
            assert callable(getattr(telemetry, func_name))
