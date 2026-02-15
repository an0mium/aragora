"""Tests for pulse analytics and topic outcome endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.features.pulse import PulseHandler


_SCHEDULER_PATCH = "aragora.server.handlers.features.pulse.get_pulse_scheduler"
_STORE_PATCH = "aragora.server.handlers.features.pulse.get_scheduled_debate_store"
_MANAGER_PATCH = "aragora.server.handlers.features.pulse.get_pulse_manager"


@pytest.fixture
def handler():
    return PulseHandler(ctx={})


@dataclass
class MockRecord:
    id: str
    topic_text: str
    platform: str
    category: str
    debate_id: str
    consensus_reached: bool
    confidence: float
    rounds_used: int
    created_at: float
    hours_ago: float = 1.0
    volume: int = 100
    topic_hash: str = ""
    scheduler_run_id: str = ""


class TestSchedulerAnalytics:
    """Tests for GET /api/v1/pulse/scheduler/analytics."""

    def test_returns_combined_metrics(self, handler):
        mock_scheduler = MagicMock()
        mock_scheduler.metrics.to_dict.return_value = {
            "polls_total": 42,
            "debates_created": 10,
            "debates_failed": 2,
        }
        mock_store = MagicMock()
        mock_store.get_analytics.return_value = {
            "total": 10,
            "by_platform": {"hackernews": {"total": 5}},
        }

        with patch(_SCHEDULER_PATCH, return_value=mock_scheduler), \
             patch(_STORE_PATCH, return_value=mock_store):
            result = handler._get_scheduler_analytics()

        body = result[0]
        assert "scheduler_metrics" in body
        assert body["scheduler_metrics"]["polls_total"] == 42
        assert "store_analytics" in body
        assert body["store_analytics"]["total"] == 10

    def test_no_scheduler_returns_unavailable(self, handler):
        with patch(_SCHEDULER_PATCH, return_value=None):
            result = handler._get_scheduler_analytics()
        assert result[1] == 501

    def test_no_store_returns_empty_analytics(self, handler):
        mock_scheduler = MagicMock()
        mock_scheduler.metrics.to_dict.return_value = {"polls_total": 5}

        with patch(_SCHEDULER_PATCH, return_value=mock_scheduler), \
             patch(_STORE_PATCH, return_value=None):
            result = handler._get_scheduler_analytics()

        body = result[0]
        assert body["scheduler_metrics"]["polls_total"] == 5
        assert body["store_analytics"] == {}


class TestTopicOutcomes:
    """Tests for GET /api/v1/pulse/topics/{topic_id}/outcomes."""

    def test_returns_outcomes_from_store(self, handler):
        mock_store = MagicMock()
        record = MockRecord(
            id="rec-1",
            topic_text="AI safety debate",
            platform="hackernews",
            category="tech",
            debate_id="debate-123",
            consensus_reached=True,
            confidence=0.85,
            rounds_used=3,
            created_at=1700000000.0,
        )
        mock_store.fetch_all.return_value = [
            ("rec-1", "abc123", "AI safety debate", "hackernews", "tech", 100,
             "debate-123", 1700000000.0, 1, 0.85, 3, "run-1"),
        ]
        mock_store._row_to_record.return_value = record

        with patch(_STORE_PATCH, return_value=mock_store):
            result = handler._get_topic_outcomes("abc123")

        body = result[0]
        assert body["topic_id"] == "abc123"
        assert body["count"] == 1
        assert body["outcomes"][0]["topic"] == "AI safety debate"
        assert body["outcomes"][0]["consensus_reached"] is True

    def test_empty_store_falls_back_to_manager(self, handler):
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []

        mock_manager = MagicMock()
        outcome = MagicMock()
        outcome.topic = "Climate debate"
        outcome.platform = "reddit"
        outcome.debate_id = "debate-456"
        outcome.consensus_reached = False
        outcome.confidence = 0.6
        outcome.rounds_used = 2
        outcome.category = "science"
        outcome.timestamp = 1700000000
        mock_manager._outcomes = [outcome]

        with patch(_STORE_PATCH, return_value=mock_store), \
             patch(_MANAGER_PATCH, return_value=mock_manager):
            result = handler._get_topic_outcomes("debate-456")

        body = result[0]
        assert body["count"] == 1
        assert body["outcomes"][0]["platform"] == "reddit"

    def test_no_results_returns_404(self, handler):
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = []

        with patch(_STORE_PATCH, return_value=mock_store), \
             patch(_MANAGER_PATCH, return_value=None):
            result = handler._get_topic_outcomes("nonexistent")

        assert result[1] == 404
        body = result[0]
        assert body["outcomes"] == []

    def test_multiple_outcomes_returned(self, handler):
        record1 = MockRecord(
            id="rec-1", topic_text="AI safety", platform="hackernews",
            category="tech", debate_id="d-1", consensus_reached=True,
            confidence=0.9, rounds_used=3, created_at=1700000000.0,
        )
        record2 = MockRecord(
            id="rec-2", topic_text="AI safety", platform="reddit",
            category="tech", debate_id="d-2", consensus_reached=False,
            confidence=0.5, rounds_used=5, created_at=1700001000.0,
        )
        mock_store = MagicMock()
        mock_store.fetch_all.return_value = [
            ("rec-1", "abc", "AI safety", "hackernews", "tech", 100,
             "d-1", 1700000000.0, 1, 0.9, 3, "r-1"),
            ("rec-2", "abc", "AI safety", "reddit", "tech", 50,
             "d-2", 1700001000.0, 0, 0.5, 5, "r-2"),
        ]
        mock_store._row_to_record.side_effect = [record1, record2]

        with patch(_STORE_PATCH, return_value=mock_store):
            result = handler._get_topic_outcomes("abc")

        body = result[0]
        assert body["count"] == 2


class TestPulseCanHandle:
    """Tests for route matching."""

    def test_can_handle_analytics(self, handler):
        assert handler.can_handle("/api/v1/pulse/analytics")

    def test_can_handle_scheduler_analytics(self, handler):
        assert handler.can_handle("/api/v1/pulse/scheduler/analytics")

    def test_can_handle_topic_outcomes(self, handler):
        assert handler.can_handle("/api/v1/pulse/topics/abc123/outcomes")

    def test_cannot_handle_unrelated(self, handler):
        assert not handler.can_handle("/api/v1/debates")
