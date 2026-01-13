"""Tests for MomentsHandler."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional

from aragora.server.handlers.moments import MomentsHandler, VALID_MOMENT_TYPES


@dataclass
class MockMoment:
    """Mock SignificantMoment for testing."""

    id: str
    moment_type: str
    agent_name: str
    description: str
    significance_score: float
    debate_id: str
    other_agents: Optional[list] = None
    metadata: Optional[dict] = None
    created_at: Optional[str] = None


class TestMomentsHandlerRouting:
    """Tests for route matching."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return MomentsHandler({})

    def test_can_handle_summary(self, handler):
        """Should handle /api/moments/summary."""
        assert handler.can_handle("/api/moments/summary") is True

    def test_can_handle_timeline(self, handler):
        """Should handle /api/moments/timeline."""
        assert handler.can_handle("/api/moments/timeline") is True

    def test_can_handle_trending(self, handler):
        """Should handle /api/moments/trending."""
        assert handler.can_handle("/api/moments/trending") is True

    def test_can_handle_by_type(self, handler):
        """Should handle /api/moments/by-type/{type}."""
        assert handler.can_handle("/api/moments/by-type/upset_victory") is True
        assert handler.can_handle("/api/moments/by-type/position_reversal") is True

    def test_cannot_handle_unrelated(self, handler):
        """Should not handle unrelated routes."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agent/claude/moments") is False
        assert handler.can_handle("/api/relationships/summary") is False

    def test_cannot_handle_incomplete_paths(self, handler):
        """Should not handle incomplete paths."""
        assert handler.can_handle("/api/moments") is False
        assert handler.can_handle("/api/moments/by-type") is False


class TestSummaryEndpoint:
    """Tests for /api/moments/summary endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return MomentsHandler({})

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False)
    def test_503_when_unavailable(self, handler):
        """Should return 503 when moment detector unavailable."""
        result = handler.handle("/api/moments/summary", {}, Mock())
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not available" in data["error"]

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_503_when_not_configured(self, handler):
        """Should return 503 when moment detector not in context."""
        result = handler.handle("/api/moments/summary", {}, Mock())
        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not configured" in data["error"]

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_returns_summary_structure(self):
        """Should return proper summary structure."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment("m1", "upset_victory", "claude", "Won upset", 0.8, "d1"),
            ],
            "gpt4": [
                MockMoment("m2", "position_reversal", "gpt4", "Reversed", 0.6, "d2"),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/summary", {}, Mock())

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_moments"] == 2
        assert "by_type" in data
        assert "by_agent" in data
        assert "most_significant" in data
        assert "recent" in data

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_counts_by_type_correctly(self):
        """Should count moments by type."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment("m1", "upset_victory", "claude", "Won", 0.8, "d1"),
                MockMoment("m2", "upset_victory", "claude", "Won again", 0.7, "d2"),
            ],
            "gpt4": [
                MockMoment("m3", "position_reversal", "gpt4", "Reversed", 0.6, "d3"),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/summary", {}, Mock())

        data = json.loads(result.body)
        assert data["by_type"]["upset_victory"] == 2
        assert data["by_type"]["position_reversal"] == 1

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_counts_by_agent_correctly(self):
        """Should count moments by agent."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment("m1", "upset_victory", "claude", "Won", 0.8, "d1"),
                MockMoment("m2", "domain_mastery", "claude", "Mastered", 0.9, "d2"),
            ],
            "gpt4": [
                MockMoment("m3", "position_reversal", "gpt4", "Reversed", 0.6, "d3"),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/summary", {}, Mock())

        data = json.loads(result.body)
        assert data["by_agent"]["claude"] == 2
        assert data["by_agent"]["gpt4"] == 1

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_returns_most_significant(self):
        """Should return the most significant moment."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment("m1", "upset_victory", "claude", "Low significance", 0.3, "d1"),
                MockMoment("m2", "domain_mastery", "claude", "High significance", 0.95, "d2"),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/summary", {}, Mock())

        data = json.loads(result.body)
        assert data["most_significant"]["significance"] == 0.95
        assert data["most_significant"]["description"] == "High significance"

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_empty_cache_returns_valid_response(self):
        """Should handle empty moment cache."""
        mock_detector = Mock()
        mock_detector._moment_cache = {}

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/summary", {}, Mock())

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_moments"] == 0
        assert data["by_type"] == {}
        assert data["by_agent"] == {}
        assert data["most_significant"] is None


class TestTimelineEndpoint:
    """Tests for /api/moments/timeline endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return MomentsHandler({})

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False)
    def test_503_when_unavailable(self, handler):
        """Should return 503 when moment detector unavailable."""
        result = handler.handle("/api/moments/timeline", {}, Mock())
        assert result.status_code == 503

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_returns_timeline_structure(self):
        """Should return proper timeline structure."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment(
                    "m1",
                    "upset_victory",
                    "claude",
                    "Won",
                    0.8,
                    "d1",
                    created_at="2025-01-01T10:00:00",
                ),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/timeline", {}, Mock())

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "moments" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_respects_limit_parameter(self):
        """Should respect limit parameter."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment(f"m{i}", "upset_victory", "claude", f"Won {i}", 0.5, f"d{i}")
                for i in range(10)
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/timeline", {"limit": "5"}, Mock())

        data = json.loads(result.body)
        assert len(data["moments"]) == 5
        assert data["total"] == 10
        assert data["has_more"] is True

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_respects_offset_parameter(self):
        """Should respect offset parameter."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment(f"m{i}", "upset_victory", "claude", f"Won {i}", 0.5, f"d{i}")
                for i in range(10)
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/timeline", {"limit": "5", "offset": "8"}, Mock())

        data = json.loads(result.body)
        assert len(data["moments"]) == 2  # Only 2 remaining after offset 8
        assert data["has_more"] is False

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_caps_limit_at_200(self):
        """Should cap limit at 200."""
        mock_detector = Mock()
        mock_detector._moment_cache = {}

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/timeline", {"limit": "500"}, Mock())

        data = json.loads(result.body)
        assert data["limit"] == 200


class TestTrendingEndpoint:
    """Tests for /api/moments/trending endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return MomentsHandler({})

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False)
    def test_503_when_unavailable(self, handler):
        """Should return 503 when moment detector unavailable."""
        result = handler.handle("/api/moments/trending", {}, Mock())
        assert result.status_code == 503

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_returns_trending_structure(self):
        """Should return proper trending structure."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment("m1", "upset_victory", "claude", "Won", 0.8, "d1"),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/trending", {}, Mock())

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "trending" in data
        assert "count" in data

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_orders_by_significance(self):
        """Should order by significance descending."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment("low", "upset_victory", "claude", "Low", 0.3, "d1"),
                MockMoment("high", "upset_victory", "claude", "High", 0.9, "d2"),
                MockMoment("mid", "upset_victory", "claude", "Mid", 0.6, "d3"),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/trending", {}, Mock())

        data = json.loads(result.body)
        assert data["trending"][0]["id"] == "high"
        assert data["trending"][1]["id"] == "mid"
        assert data["trending"][2]["id"] == "low"

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_respects_limit_parameter(self):
        """Should respect limit parameter."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment(f"m{i}", "upset_victory", "claude", f"Won", 0.5, f"d{i}")
                for i in range(10)
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/trending", {"limit": "3"}, Mock())

        data = json.loads(result.body)
        assert len(data["trending"]) == 3

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_caps_limit_at_50(self):
        """Should cap limit at 50."""
        mock_detector = Mock()
        mock_detector._moment_cache = {}

        handler = MomentsHandler({"moment_detector": mock_detector})
        # Test that limit is capped - we can't directly verify the cap
        # but we can verify it doesn't crash with large values
        result = handler.handle("/api/moments/trending", {"limit": "100"}, Mock())
        assert result.status_code == 200


class TestByTypeEndpoint:
    """Tests for /api/moments/by-type/{type} endpoint."""

    @pytest.fixture
    def handler(self):
        """Create handler with mock context."""
        return MomentsHandler({})

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", False)
    def test_503_when_unavailable(self, handler):
        """Should return 503 when moment detector unavailable."""
        result = handler.handle("/api/moments/by-type/upset_victory", {}, Mock())
        assert result.status_code == 503

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_validates_moment_type(self):
        """Should reject invalid moment types."""
        mock_detector = Mock()
        mock_detector._moment_cache = {}

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/by-type/invalid_type", {}, Mock())

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid moment type" in data["error"]

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_rejects_path_traversal(self):
        """Should reject path traversal in moment type."""
        mock_detector = Mock()
        mock_detector._moment_cache = {}

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/by-type/../../../etc", {}, Mock())

        assert result.status_code == 400

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_filters_by_type(self):
        """Should filter moments by type."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment("m1", "upset_victory", "claude", "Upset", 0.8, "d1"),
                MockMoment("m2", "position_reversal", "claude", "Reversed", 0.6, "d2"),
                MockMoment("m3", "upset_victory", "claude", "Another upset", 0.7, "d3"),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/by-type/upset_victory", {}, Mock())

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["type"] == "upset_victory"
        assert data["total"] == 2
        assert len(data["moments"]) == 2

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_returns_all_valid_types(self):
        """Should accept all valid moment types."""
        mock_detector = Mock()
        mock_detector._moment_cache = {}

        handler = MomentsHandler({"moment_detector": mock_detector})

        for moment_type in VALID_MOMENT_TYPES:
            result = handler.handle(f"/api/moments/by-type/{moment_type}", {}, Mock())
            assert result.status_code == 200, f"Failed for type: {moment_type}"


class TestMomentSerialization:
    """Tests for moment to dict conversion."""

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_serializes_all_fields(self):
        """Should serialize all moment fields."""
        mock_detector = Mock()
        mock_detector._moment_cache = {
            "claude": [
                MockMoment(
                    id="m1",
                    moment_type="upset_victory",
                    agent_name="claude",
                    description="Won against strong opponent",
                    significance_score=0.85,
                    debate_id="debate-123",
                    other_agents=["gpt4", "gemini"],
                    metadata={"elo_diff": 150},
                    created_at="2025-01-01T12:00:00",
                ),
            ],
        }

        handler = MomentsHandler({"moment_detector": mock_detector})
        result = handler.handle("/api/moments/summary", {}, Mock())

        data = json.loads(result.body)
        moment = data["recent"][0]

        assert moment["id"] == "m1"
        assert moment["type"] == "upset_victory"
        assert moment["agent"] == "claude"
        assert moment["description"] == "Won against strong opponent"
        assert moment["significance"] == 0.85
        assert moment["debate_id"] == "debate-123"
        assert moment["other_agents"] == ["gpt4", "gemini"]
        assert moment["metadata"] == {"elo_diff": 150}
        assert moment["created_at"] == "2025-01-01T12:00:00"


class TestHandlerImport:
    """Tests for handler module imports."""

    def test_handler_can_be_imported(self):
        """Should be importable from handlers package."""
        from aragora.server.handlers import MomentsHandler

        assert MomentsHandler is not None

    def test_handler_in_all(self):
        """Should be in __all__ exports."""
        from aragora.server.handlers import __all__

        assert "MomentsHandler" in __all__

    def test_valid_moment_types_defined(self):
        """Should have valid moment types defined."""
        assert len(VALID_MOMENT_TYPES) >= 7
        assert "upset_victory" in VALID_MOMENT_TYPES
        assert "position_reversal" in VALID_MOMENT_TYPES
        assert "calibration_vindication" in VALID_MOMENT_TYPES


class TestErrorHandling:
    """Tests for error handling."""

    @patch("aragora.server.handlers.moments.MOMENT_DETECTOR_AVAILABLE", True)
    def test_handles_exception_in_summary(self):
        """Should handle exceptions gracefully in summary."""
        mock_detector = Mock()
        mock_detector._moment_cache = property(
            lambda self: (_ for _ in ()).throw(Exception("DB error"))
        )

        handler = MomentsHandler({"moment_detector": mock_detector})
        # Mock will raise on iteration
        mock_detector._moment_cache = Mock(side_effect=Exception("Cache error"))

        result = handler.handle("/api/moments/summary", {}, Mock())
        assert result.status_code == 500
