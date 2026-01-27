"""Tests for new feature Prometheus metrics."""

import pytest
from unittest.mock import patch, MagicMock

from aragora.server.prometheus_features import (
    # Checkpoint Bridge
    record_checkpoint_bridge_save,
    record_checkpoint_bridge_restore,
    record_checkpoint_bridge_molecule_recovery,
    record_checkpoint_bridge_save_duration,
    track_checkpoint_bridge_save,
    # Agent Channel
    record_agent_channel_message,
    record_agent_channel_setup,
    record_agent_channel_teardown,
    set_agent_channel_active_count,
    record_agent_channel_history_size,
    # Session Management
    record_session_created,
    record_session_debate_linked,
    record_session_handoff,
    record_session_result_route,
    set_session_active_count,
)


class TestCheckpointBridgeMetrics:
    """Tests for checkpoint bridge metrics."""

    def test_record_checkpoint_bridge_save(self):
        """Test recording checkpoint save."""
        # Should not raise
        record_checkpoint_bridge_save("debate_123456789", "voting")
        record_checkpoint_bridge_save("short", "proposal")
        record_checkpoint_bridge_save("", "critique")

    def test_record_checkpoint_bridge_restore(self):
        """Test recording checkpoint restore."""
        record_checkpoint_bridge_restore("success")
        record_checkpoint_bridge_restore("not_found")
        record_checkpoint_bridge_restore("failed")

    def test_record_checkpoint_bridge_molecule_recovery(self):
        """Test recording molecule recovery."""
        record_checkpoint_bridge_molecule_recovery("success")
        record_checkpoint_bridge_molecule_recovery("not_found")
        record_checkpoint_bridge_molecule_recovery("no_state")

    def test_record_checkpoint_bridge_save_duration(self):
        """Test recording save duration."""
        record_checkpoint_bridge_save_duration(0.5)
        record_checkpoint_bridge_save_duration(2.5)
        record_checkpoint_bridge_save_duration(0.01)

    def test_track_checkpoint_bridge_save_context(self):
        """Test tracking context manager."""
        with track_checkpoint_bridge_save("debate_abc", "synthesis"):
            pass  # Simulated save operation

    def test_track_checkpoint_bridge_save_with_exception(self):
        """Test tracking context manager handles exceptions."""
        with pytest.raises(ValueError):
            with track_checkpoint_bridge_save("debate_xyz", "voting"):
                raise ValueError("Test error")


class TestAgentChannelMetrics:
    """Tests for agent channel metrics."""

    def test_record_agent_channel_message(self):
        """Test recording channel messages."""
        record_agent_channel_message("proposal", "channel_123")
        record_agent_channel_message("critique", "channel_456")
        record_agent_channel_message("query", "short")
        record_agent_channel_message("signal", "")

    def test_record_agent_channel_setup(self):
        """Test recording channel setup."""
        record_agent_channel_setup("success")
        record_agent_channel_setup("failed")
        record_agent_channel_setup("disabled")

    def test_record_agent_channel_teardown(self):
        """Test recording channel teardown."""
        record_agent_channel_teardown()

    def test_set_agent_channel_active_count(self):
        """Test setting active channel count."""
        set_agent_channel_active_count(0)
        set_agent_channel_active_count(5)
        set_agent_channel_active_count(100)

    def test_record_agent_channel_history_size(self):
        """Test recording history size."""
        record_agent_channel_history_size(0)
        record_agent_channel_history_size(50)
        record_agent_channel_history_size(250)


class TestSessionManagementMetrics:
    """Tests for session management metrics."""

    def test_record_session_created(self):
        """Test recording session creation."""
        record_session_created("slack")
        record_session_created("telegram")
        record_session_created("whatsapp")
        record_session_created("api")

    def test_record_session_debate_linked(self):
        """Test recording debate linking."""
        record_session_debate_linked("slack")
        record_session_debate_linked("telegram")

    def test_record_session_handoff(self):
        """Test recording session handoff."""
        record_session_handoff("slack", "telegram")
        record_session_handoff("telegram", "whatsapp")
        record_session_handoff("api", "slack")

    def test_record_session_result_route(self):
        """Test recording result routing."""
        record_session_result_route("slack", "success")
        record_session_result_route("telegram", "failed")
        record_session_result_route("whatsapp", "no_channel")

    def test_set_session_active_count(self):
        """Test setting active session count."""
        set_session_active_count("slack", 10)
        set_session_active_count("telegram", 5)
        set_session_active_count("api", 0)


class TestMetricsWithPrometheus:
    """Tests for metrics integration.

    Note: These tests verify the metrics functions work correctly regardless
    of whether prometheus_client is installed. The conditional import logic
    makes patching complex, so we focus on functional testing.
    """

    def test_all_checkpoint_metrics_callable(self):
        """Test all checkpoint metrics can be called without error."""
        record_checkpoint_bridge_save("test_debate", "proposal")
        record_checkpoint_bridge_restore("success")
        record_checkpoint_bridge_molecule_recovery("success")
        record_checkpoint_bridge_save_duration(1.5)

    def test_all_channel_metrics_callable(self):
        """Test all channel metrics can be called without error."""
        record_agent_channel_message("proposal", "test_channel")
        record_agent_channel_setup("success")
        record_agent_channel_teardown()
        set_agent_channel_active_count(5)
        record_agent_channel_history_size(50)

    def test_all_session_metrics_callable(self):
        """Test all session metrics can be called without error."""
        record_session_created("slack")
        record_session_debate_linked("telegram")
        record_session_handoff("slack", "telegram")
        record_session_result_route("whatsapp", "success")
        set_session_active_count("api", 10)


class TestMetricsWithoutPrometheus:
    """Tests for metrics when prometheus_client not available."""

    @pytest.fixture
    def mock_prometheus_unavailable(self):
        """Mock prometheus being unavailable."""
        with patch("aragora.server.prometheus_features.PROMETHEUS_AVAILABLE", False):
            yield

    def test_checkpoint_save_without_prometheus(self, mock_prometheus_unavailable):
        """Test checkpoint save uses simple metrics fallback."""
        mock_simple = MagicMock()
        with patch(
            "aragora.server.prometheus_features._simple_metrics",
            mock_simple,
        ):
            record_checkpoint_bridge_save("debate_123", "voting")
            mock_simple.inc_counter.assert_called_once()

    def test_channel_setup_without_prometheus(self, mock_prometheus_unavailable):
        """Test channel setup uses simple metrics fallback."""
        mock_simple = MagicMock()
        with patch(
            "aragora.server.prometheus_features._simple_metrics",
            mock_simple,
        ):
            record_agent_channel_setup("success")
            mock_simple.inc_counter.assert_called_once()

    def test_session_active_without_prometheus(self, mock_prometheus_unavailable):
        """Test session active count uses simple metrics fallback."""
        mock_simple = MagicMock()
        with patch(
            "aragora.server.prometheus_features._simple_metrics",
            mock_simple,
        ):
            set_session_active_count("telegram", 15)
            mock_simple.set_gauge.assert_called_once()


class TestMetricsTruncation:
    """Tests for metric label truncation."""

    def test_debate_id_truncation(self):
        """Test debate ID is truncated to limit cardinality."""
        # Very long ID should be truncated
        long_id = "debate_" + "x" * 100

        # Should not raise and should work
        record_checkpoint_bridge_save(long_id, "voting")

    def test_channel_truncation(self):
        """Test channel name is truncated."""
        long_channel = "channel_" + "y" * 100

        # Should not raise
        record_agent_channel_message("proposal", long_channel)

    def test_empty_values_handled(self):
        """Test empty values don't cause errors."""
        record_checkpoint_bridge_save("", "")
        record_agent_channel_message("", "")
