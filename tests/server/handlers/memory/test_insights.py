"""Tests for InsightsHandler."""

from __future__ import annotations

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


import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.memory.insights import (
    InsightsHandler,
    INSIGHTS_PERMISSION,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


class InsightType(Enum):
    """Mock insight type enum."""

    POSITION_REVERSAL = "position_reversal"
    CONSENSUS_PATTERN = "consensus_pattern"
    ARGUMENT_STRENGTH = "argument_strength"


@dataclass
class MockInsight:
    """Mock insight object."""

    id: str
    type: InsightType
    title: str
    description: str
    confidence: float
    agents_involved: List[str]
    evidence: List[str] = field(default_factory=list)
    created_at: Optional[str] = None


@dataclass
class MockInsightStore:
    """Mock InsightStore."""

    get_recent_insights: AsyncMock = field(default_factory=AsyncMock)


class MockAuthContext:
    """Mock authentication context."""

    def __init__(self, user_id: str = "user-123", permissions: list = None):
        self.user_id = user_id
        self.permissions = permissions or [INSIGHTS_PERMISSION]


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, client_ip: str = "127.0.0.1", body: bytes = b""):
        self.headers = {"X-Forwarded-For": client_ip, "Content-Length": str(len(body))}
        self.rfile = MagicMock()
        self.rfile.read.return_value = body


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_insight_store():
    """Create a mock insight store."""
    return MockInsightStore()


@pytest.fixture
def handler(mock_insight_store):
    """Create a test handler with mock insight store."""
    ctx = {"insight_store": mock_insight_store}
    h = InsightsHandler(server_context=ctx)
    return h


@pytest.fixture
def handler_no_store():
    """Create a test handler without insight store."""
    return InsightsHandler(server_context={})


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test Handler Routing
# =============================================================================


class TestHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_recent_insights(self, handler):
        """Test can_handle for recent insights route."""
        assert handler.can_handle("/api/insights/recent") is True
        assert handler.can_handle("/api/v1/insights/recent") is True

    def test_can_handle_extract_detailed(self, handler):
        """Test can_handle for extract-detailed route."""
        assert handler.can_handle("/api/insights/extract-detailed") is True

    def test_can_handle_flips_recent(self, handler):
        """Test can_handle for flips recent route."""
        assert handler.can_handle("/api/flips/recent") is True

    def test_can_handle_flips_summary(self, handler):
        """Test can_handle for flips summary route."""
        assert handler.can_handle("/api/flips/summary") is True

    def test_cannot_handle_invalid_route(self, handler):
        """Test can_handle for invalid route."""
        assert handler.can_handle("/api/other/route") is False


# =============================================================================
# Test Get Recent Insights
# =============================================================================


class TestGetRecentInsights:
    """Tests for get recent insights endpoint."""

    def test_get_insights_success(self, handler, mock_insight_store):
        """Test successful insights retrieval."""
        insights = [
            MockInsight(
                id="insight-1",
                type=InsightType.POSITION_REVERSAL,
                title="Agent changed position",
                description="Agent A reversed on topic X",
                confidence=0.85,
                agents_involved=["agent-a"],
                evidence=["evidence-1", "evidence-2"],
            ),
        ]
        mock_insight_store.get_recent_insights.return_value = insights

        result = handler._get_recent_insights({}, handler.ctx)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 1
        assert len(data["insights"]) == 1
        assert data["insights"][0]["type"] == "position_reversal"

    def test_get_insights_with_limit(self, handler, mock_insight_store):
        """Test insights retrieval with limit parameter."""
        mock_insight_store.get_recent_insights.return_value = []

        result = handler._get_recent_insights({"limit": 50}, handler.ctx)

        assert result.status_code == 200
        mock_insight_store.get_recent_insights.assert_called_once()

    def test_get_insights_limit_capped(self, handler, mock_insight_store):
        """Test insights limit is capped at 100."""
        mock_insight_store.get_recent_insights.return_value = []

        handler._get_recent_insights({"limit": 500}, handler.ctx)

        # Should be capped to 100
        call_args = mock_insight_store.get_recent_insights.call_args
        assert call_args.kwargs["limit"] == 100

    def test_get_insights_no_store(self, handler_no_store):
        """Test insights when store not configured."""
        result = handler_no_store._get_recent_insights({}, handler_no_store.ctx)

        assert result.status_code == 200
        data = parse_response(result)
        assert "error" in data
        assert data["insights"] == []


# =============================================================================
# Test Get Recent Flips
# =============================================================================


class TestGetRecentFlips:
    """Tests for get recent flips endpoint."""

    def test_get_flips_success(self, handler, mock_insight_store):
        """Test successful flips retrieval."""
        insights = [
            MockInsight(
                id="flip-1",
                type=InsightType.POSITION_REVERSAL,
                title="New position",
                description="Previous position",
                confidence=0.9,
                agents_involved=["agent-a"],
            ),
        ]
        mock_insight_store.get_recent_insights.return_value = insights

        result = handler._get_recent_flips({}, handler.ctx)

        assert result.status_code == 200
        data = parse_response(result)
        assert "flips" in data
        assert data["count"] == 1

    def test_get_flips_no_store(self, handler_no_store):
        """Test flips when store not configured."""
        result = handler_no_store._get_recent_flips({}, handler_no_store.ctx)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["flips"] == []
        assert "not configured" in data["message"]


# =============================================================================
# Test Get Flips Summary
# =============================================================================


class TestGetFlipsSummary:
    """Tests for get flips summary endpoint."""

    def test_get_summary_success(self, handler, mock_insight_store):
        """Test successful summary retrieval."""
        insights = [
            MockInsight(
                id="flip-1",
                type=InsightType.POSITION_REVERSAL,
                title="Flip 1",
                description="Desc",
                confidence=0.8,
                agents_involved=["a"],
            ),
            MockInsight(
                id="flip-2",
                type=InsightType.POSITION_REVERSAL,
                title="Flip 2",
                description="Desc",
                confidence=0.7,
                agents_involved=["b"],
            ),
        ]
        mock_insight_store.get_recent_insights.return_value = insights

        result = handler._get_flips_summary({}, handler.ctx)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["summary"]["total"] == 2

    def test_get_summary_with_period(self, handler, mock_insight_store):
        """Test summary with period parameter."""
        mock_insight_store.get_recent_insights.return_value = []

        result = handler._get_flips_summary({"period": "7d"}, handler.ctx)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["period"] == "7d"

    def test_get_summary_no_store(self, handler_no_store):
        """Test summary when store not configured."""
        result = handler_no_store._get_flips_summary({}, handler_no_store.ctx)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["summary"]["total"] == 0


# =============================================================================
# Test Extract Detailed Insights
# =============================================================================


class TestExtractDetailedInsights:
    """Tests for extract detailed insights endpoint."""

    def test_extract_success(self, handler):
        """Test successful insight extraction."""
        data = {
            "content": "Therefore, we should adopt this approach. According to research, the evidence shows improvement.",
            "debate_id": "debate-123",
        }

        result = handler._extract_detailed_insights(data, handler.ctx)

        assert result.status_code == 200
        response = parse_response(result)
        assert response["debate_id"] == "debate-123"
        assert "claims" in response
        assert "evidence_chains" in response
        assert "patterns" in response

    def test_extract_missing_content(self, handler):
        """Test extraction with missing content."""
        result = handler._extract_detailed_insights({}, handler.ctx)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_extract_empty_content(self, handler):
        """Test extraction with empty content."""
        result = handler._extract_detailed_insights({"content": ""}, handler.ctx)

        assert result.status_code == 400

    def test_extract_content_too_large(self, handler):
        """Test extraction rejects oversized content."""
        large_content = "x" * (1024 * 1024 + 1)  # > 1MB

        result = handler._extract_detailed_insights({"content": large_content}, handler.ctx)

        assert result.status_code == 413

    def test_extract_selective_options(self, handler):
        """Test extraction with selective options."""
        data = {
            "content": "Test content with some claims.",
            "extract_claims": True,
            "extract_evidence": False,
            "extract_patterns": False,
        }

        result = handler._extract_detailed_insights(data, handler.ctx)

        assert result.status_code == 200
        response = parse_response(result)
        assert "claims" in response
        assert "evidence_chains" not in response
        assert "patterns" not in response


# =============================================================================
# Test Content Extraction Helpers
# =============================================================================


class TestContentExtraction:
    """Tests for content extraction helper methods."""

    def test_extract_claims_from_content(self, handler):
        """Test claim extraction."""
        content = "Therefore, we must act now. It is clear that changes are needed."

        claims = handler._extract_claims_from_content(content)

        assert isinstance(claims, list)
        assert len(claims) <= 20

    def test_extract_evidence_from_content(self, handler):
        """Test evidence extraction."""
        content = "According to Smith, the results were positive. Research shows improvement."

        evidence = handler._extract_evidence_from_content(content)

        assert isinstance(evidence, list)
        assert len(evidence) <= 15

    def test_extract_patterns_from_content(self, handler):
        """Test pattern extraction."""
        content = "On one hand, option A is better. On the other hand, option B has merit."

        patterns = handler._extract_patterns_from_content(content)

        assert isinstance(patterns, list)
        # Should detect balanced_comparison pattern
        pattern_types = [p["type"] for p in patterns]
        assert "balanced_comparison" in pattern_types
