"""Tests for Pulse analytics endpoints.

Tests cover:
- GET /api/v1/pulse/scheduler/analytics (scheduler metrics)
- GET /api/v1/pulse/topics/{topic_id}/outcomes (topic outcomes)
- can_handle for new routes
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.features.pulse import PulseHandler


def _parse_result(result):
    """Parse HandlerResult into (status_code, data_dict)."""
    return result.status_code, json.loads(result.body)


class TestPulseAnalyticsRoutes:
    """Test route handling for new analytics endpoints."""

    def test_scheduler_analytics_in_routes(self):
        """Scheduler analytics route should be registered."""
        assert "/api/v1/pulse/scheduler/analytics" in PulseHandler.ROUTES

    def test_can_handle_scheduler_analytics(self):
        handler = PulseHandler(server_context={})
        assert handler.can_handle("/api/v1/pulse/scheduler/analytics") is True

    def test_can_handle_topic_outcomes(self):
        handler = PulseHandler(server_context={})
        assert handler.can_handle("/api/v1/pulse/topics/abc123/outcomes") is True

    def test_can_handle_topic_outcomes_with_hash(self):
        handler = PulseHandler(server_context={})
        assert handler.can_handle("/api/v1/pulse/topics/a1b2c3d4e5/outcomes") is True

    def test_cannot_handle_invalid_topic_path(self):
        handler = PulseHandler(server_context={})
        # Missing /outcomes suffix
        assert handler.can_handle("/api/v1/pulse/topics/abc123") is False
        # Wrong prefix
        assert handler.can_handle("/api/v1/pulse/topic/abc123/outcomes") is False


class TestSchedulerAnalyticsEndpoint:
    """Test GET /api/v1/pulse/scheduler/analytics."""

    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    @patch("aragora.server.handlers.features.pulse.get_pulse_scheduler")
    def test_returns_scheduler_metrics(self, mock_get_scheduler, mock_get_store):
        """Should return scheduler metrics and store analytics."""
        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {
            "polls_completed": 42,
            "debates_created": 10,
            "debates_failed": 2,
            "uptime_seconds": 3600,
        }
        mock_scheduler = MagicMock()
        mock_scheduler.metrics = mock_metrics
        mock_get_scheduler.return_value = mock_scheduler

        mock_store = MagicMock()
        mock_store.get_analytics.return_value = {
            "total": 50,
            "by_platform": {"hackernews": {"total": 30}},
        }
        mock_get_store.return_value = mock_store

        handler = PulseHandler(server_context={})
        result = handler._get_scheduler_analytics()

        assert result is not None
        status, data = _parse_result(result)

        assert status == 200
        assert data["scheduler_metrics"]["polls_completed"] == 42
        assert data["scheduler_metrics"]["debates_created"] == 10
        assert data["store_analytics"]["total"] == 50

    @patch("aragora.server.handlers.features.pulse.get_pulse_scheduler")
    def test_returns_unavailable_when_no_scheduler(self, mock_get_scheduler):
        """Should return feature unavailable when scheduler is None."""
        mock_get_scheduler.return_value = None

        handler = PulseHandler(server_context={})
        result = handler._get_scheduler_analytics()

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    @patch("aragora.server.handlers.features.pulse.get_pulse_scheduler")
    def test_handles_missing_store(self, mock_get_scheduler, mock_get_store):
        """Should return empty store_analytics when store unavailable."""
        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {"polls_completed": 5}
        mock_scheduler = MagicMock()
        mock_scheduler.metrics = mock_metrics
        mock_get_scheduler.return_value = mock_scheduler
        mock_get_store.return_value = None

        handler = PulseHandler(server_context={})
        result = handler._get_scheduler_analytics()

        status, data = _parse_result(result)
        assert status == 200
        assert data["scheduler_metrics"]["polls_completed"] == 5
        assert data["store_analytics"] == {}


class TestTopicOutcomesEndpoint:
    """Test GET /api/v1/pulse/topics/{topic_id}/outcomes."""

    @patch("aragora.server.handlers.features.pulse.get_pulse_manager")
    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    def test_returns_outcomes_from_store(self, mock_get_store, mock_get_manager):
        """Should return outcomes from the scheduled debate store."""
        mock_record = MagicMock()
        mock_record.id = "rec-1"
        mock_record.topic_text = "AI regulation"
        mock_record.platform = "hackernews"
        mock_record.category = "tech"
        mock_record.debate_id = "debate-1"
        mock_record.consensus_reached = True
        mock_record.confidence = 0.85
        mock_record.rounds_used = 3
        mock_record.created_at = 1700000000.0
        mock_record.hours_ago = 2.5

        mock_store = MagicMock()
        mock_store.fetch_all.return_value = [
            (
                "rec-1",
                "abc123",
                "AI regulation",
                "hackernews",
                "tech",
                500,
                "debate-1",
                1700000000.0,
                1,
                0.85,
                3,
                "run-1",
            )
        ]
        mock_store._row_to_record.return_value = mock_record
        mock_get_store.return_value = mock_store

        handler = PulseHandler(server_context={})
        result = handler._get_topic_outcomes("abc123")

        status, data = _parse_result(result)

        assert status == 200
        assert data["topic_id"] == "abc123"
        assert data["count"] == 1
        assert data["outcomes"][0]["topic"] == "AI regulation"
        assert data["outcomes"][0]["consensus_reached"] is True

    @patch("aragora.server.handlers.features.pulse.get_pulse_manager")
    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    def test_returns_404_when_no_outcomes(self, mock_get_store, mock_get_manager):
        """Should return 404 with empty outcomes when topic not found."""
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []
        mock_get_store.return_value = mock_store

        mock_get_manager.return_value = None

        handler = PulseHandler(server_context={})
        result = handler._get_topic_outcomes("nonexistent")

        status, data = _parse_result(result)

        assert status == 404
        assert data["count"] == 0
        assert data["outcomes"] == []

    @patch("aragora.server.handlers.features.pulse.get_pulse_manager")
    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    def test_falls_back_to_pulse_manager(self, mock_get_store, mock_get_manager):
        """Should fall back to PulseManager when store has no results."""
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []
        mock_get_store.return_value = mock_store

        mock_outcome = MagicMock()
        mock_outcome.topic = "AI safety"
        mock_outcome.platform = "reddit"
        mock_outcome.debate_id = "debate-42"
        mock_outcome.consensus_reached = False
        mock_outcome.confidence = 0.6
        mock_outcome.rounds_used = 2
        mock_outcome.category = "ai"
        mock_outcome.timestamp = 1700000000.0

        mock_manager = MagicMock()
        mock_manager._outcomes = [mock_outcome]
        mock_get_manager.return_value = mock_manager

        handler = PulseHandler(server_context={})
        result = handler._get_topic_outcomes("debate-42")

        status, data = _parse_result(result)

        assert status == 200
        assert data["count"] == 1
        assert data["outcomes"][0]["debate_id"] == "debate-42"
        assert data["outcomes"][0]["consensus_reached"] is False

    @patch("aragora.server.handlers.features.pulse.get_pulse_manager")
    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    def test_handles_store_error_gracefully(self, mock_get_store, mock_get_manager):
        """Should handle store errors and try fallback."""
        import sqlite3

        mock_store = MagicMock()
        mock_store.fetch_all.side_effect = sqlite3.Error("db locked")
        mock_get_store.return_value = mock_store

        mock_get_manager.return_value = None

        handler = PulseHandler(server_context={})
        result = handler._get_topic_outcomes("abc123")

        status, data = _parse_result(result)
        assert status == 404
        assert data["count"] == 0


class TestHandleRouting:
    """Test that handle() routes to new endpoints correctly."""

    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    @patch("aragora.server.handlers.features.pulse.get_pulse_scheduler")
    def test_routes_to_scheduler_analytics(self, mock_get_scheduler, mock_get_store):
        """handle() should route /scheduler/analytics correctly."""
        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {}
        mock_scheduler = MagicMock()
        mock_scheduler.metrics = mock_metrics
        mock_get_scheduler.return_value = mock_scheduler
        mock_get_store.return_value = None

        handler = PulseHandler(server_context={})
        result = handler.handle("/api/v1/pulse/scheduler/analytics", {}, MagicMock())

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.features.pulse.get_pulse_manager")
    @patch("aragora.server.handlers.features.pulse.get_scheduled_debate_store")
    def test_routes_to_topic_outcomes(self, mock_get_store, mock_get_manager):
        """handle() should route /topics/{id}/outcomes correctly."""
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []
        mock_get_store.return_value = mock_store
        mock_get_manager.return_value = None

        handler = PulseHandler(server_context={})
        result = handler.handle("/api/v1/pulse/topics/abc123/outcomes", {}, MagicMock())

        assert result is not None
        # 404 because no outcomes found
        assert result.status_code == 404
