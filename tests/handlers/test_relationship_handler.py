"""Tests for agent relationship summary handler."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.agents.relationships import RelationshipHandler


@dataclass
class MockMetrics:
    agent_a: str
    agent_b: str
    rivalry_score: float
    alliance_score: float
    relationship: str
    debate_count: int
    agreement_rate: float = 0.0
    head_to_head: str = "0-0"


@pytest.fixture
def handler():
    return RelationshipHandler(ctx={})


@pytest.fixture
def mock_tracker():
    tracker = MagicMock()
    tracker.get_rivals.return_value = [
        MockMetrics("claude", "gpt4", 0.75, 0.2, "rival", 10, 0.3, "6-4"),
        MockMetrics("claude", "gemini", 0.6, 0.3, "rival", 8, 0.4, "5-3"),
    ]
    tracker.get_allies.return_value = [
        MockMetrics("claude", "mistral", 0.1, 0.8, "ally", 12, 0.9, "7-5"),
    ]
    tracker.compute_metrics.return_value = MockMetrics(
        "claude", "gpt4", 0.75, 0.2, "rival", 10, 0.3, "6-4"
    )
    return tracker


_TRACKER_PATCH = "aragora.server.handlers.agents.relationships._get_relationship_tracker"


class TestRelationshipSummary:
    """Tests for GET /api/v1/agents/{name}/relationships."""

    def test_rivals_sorted_by_rivalry_score(self, handler, mock_tracker):
        with patch(_TRACKER_PATCH, return_value=mock_tracker):
            result = handler.handle("/api/v1/agents/claude/relationships", {}, MagicMock())
        body = result[0]
        assert body["agent"] == "claude"
        assert len(body["rivals"]) == 2
        # First rival has higher score
        assert body["rivals"][0]["rivalry_score"] >= body["rivals"][1]["rivalry_score"]

    def test_allies_sorted_by_alliance_score(self, handler, mock_tracker):
        with patch(_TRACKER_PATCH, return_value=mock_tracker):
            result = handler.handle("/api/v1/agents/claude/relationships", {}, MagicMock())
        body = result[0]
        assert len(body["allies"]) == 1
        assert body["allies"][0]["alliance_score"] == 0.8

    def test_empty_relationships(self, handler):
        empty_tracker = MagicMock()
        empty_tracker.get_rivals.return_value = []
        empty_tracker.get_allies.return_value = []

        with patch(_TRACKER_PATCH, return_value=empty_tracker):
            result = handler.handle("/api/v1/agents/newagent/relationships", {}, MagicMock())
        body = result[0]
        assert body["rivals"] == []
        assert body["allies"] == []

    def test_no_tracker_returns_empty(self, handler):
        with patch(_TRACKER_PATCH, return_value=None):
            result = handler.handle("/api/v1/agents/claude/relationships", {}, MagicMock())
        body = result[0]
        assert body["rivals"] == []
        assert body["allies"] == []


class TestPairwiseMetrics:
    """Tests for GET /api/v1/agents/{name}/relationships/{other}."""

    def test_pairwise_metrics_returned(self, handler, mock_tracker):
        with patch(_TRACKER_PATCH, return_value=mock_tracker):
            result = handler.handle("/api/v1/agents/claude/relationships/gpt4", {}, MagicMock())
        body = result[0]
        assert body["agent_a"] == "claude"
        assert body["agent_b"] == "gpt4"
        assert body["debate_count"] == 10
        assert body["rivalry_score"] == 0.75
        assert body["relationship"] == "rival"
        assert body["head_to_head"] == "6-4"

    def test_no_tracker_returns_unknown(self, handler):
        with patch(_TRACKER_PATCH, return_value=None):
            result = handler.handle("/api/v1/agents/claude/relationships/gpt4", {}, MagicMock())
        body = result[0]
        assert body["debate_count"] == 0
        assert body["relationship"] == "unknown"


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_summary(self, handler):
        assert handler.can_handle("/api/v1/agents/claude/relationships")

    def test_can_handle_pairwise(self, handler):
        assert handler.can_handle("/api/v1/agents/claude/relationships/gpt4")

    def test_cannot_handle_unrelated(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_cannot_handle_feedback(self, handler):
        assert not handler.can_handle("/api/v1/agents/feedback/metrics")
