"""Tests for StalenessOperationsMixin."""

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


import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from aragora.server.handlers.knowledge_base.mound.staleness import (
    StalenessOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


class StalenessReason(str, Enum):
    """Mock staleness reason enum."""

    AGE = "age"
    LOW_CONFIDENCE = "low_confidence"
    CONTRADICTED = "contradicted"
    OUTDATED_SOURCE = "outdated_source"


@dataclass
class MockStaleItem:
    """Mock stale item."""

    node_id: str
    staleness_score: float
    reasons: list[StalenessReason]
    last_validated_at: datetime | None = None
    recommended_action: str = "revalidate"


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    get_stale_knowledge: AsyncMock = field(default_factory=AsyncMock)
    mark_validated: AsyncMock = field(default_factory=AsyncMock)
    schedule_revalidation: AsyncMock = field(default_factory=AsyncMock)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: bytes = b"", headers: dict[str, str] | None = None):
        self.headers = headers or {}
        self._body = body
        self.rfile = io.BytesIO(body)

        if body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))


class StalenessHandler(StalenessOperationsMixin):
    """Handler implementation for testing StalenessOperationsMixin."""

    def __init__(self, mound: MockKnowledgeMound | None = None):
        self._mound = mound
        self.ctx = {}

    def _get_mound(self):
        return self._mound


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    return MockKnowledgeMound()


@pytest.fixture
def handler(mock_mound):
    """Create a test handler with mock mound."""
    return StalenessHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a test handler without mound."""
    return StalenessHandler(mound=None)


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test get_stale
# =============================================================================


class TestGetStale:
    """Tests for get stale knowledge endpoint."""

    def test_get_stale_success(self, handler, mock_mound):
        """Test successful stale knowledge retrieval."""
        stale_items = [
            MockStaleItem(
                node_id="node-1",
                staleness_score=0.8,
                reasons=[StalenessReason.AGE, StalenessReason.LOW_CONFIDENCE],
                last_validated_at=datetime(2025, 1, 1, 12, 0, 0),
            ),
            MockStaleItem(
                node_id="node-2",
                staleness_score=0.6,
                reasons=[StalenessReason.CONTRADICTED],
            ),
        ]
        mock_mound.get_stale_knowledge.return_value = stale_items

        result = handler._handle_get_stale({})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["total"] == 2
        assert len(data["stale_items"]) == 2
        assert data["stale_items"][0]["staleness_score"] == 0.8

    def test_get_stale_with_threshold(self, handler, mock_mound):
        """Test stale retrieval with threshold filter."""
        mock_mound.get_stale_knowledge.return_value = []

        result = handler._handle_get_stale({"threshold": ["0.7"]})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["threshold"] == 0.7

    def test_get_stale_with_workspace(self, handler, mock_mound):
        """Test stale retrieval for specific workspace."""
        mock_mound.get_stale_knowledge.return_value = []

        result = handler._handle_get_stale({"workspace_id": ["ws-123"]})

        assert result.status_code == 200
        data = parse_response(result)
        assert data["workspace_id"] == "ws-123"

    def test_get_stale_no_mound(self, handler_no_mound):
        """Test stale retrieval when mound not available."""
        result = handler_no_mound._handle_get_stale({})

        assert result.status_code == 503


# =============================================================================
# Test revalidate_node
# =============================================================================


class TestRevalidateNode:
    """Tests for revalidate node endpoint."""

    def test_revalidate_success(self, handler, mock_mound):
        """Test successful node revalidation."""
        body = json.dumps(
            {
                "validator": "human_review",
                "confidence": 0.95,
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_revalidate_node("node-123", http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["node_id"] == "node-123"
        assert data["validated"] is True
        assert data["validator"] == "human_review"
        assert data["new_confidence"] == 0.95

    def test_revalidate_default_validator(self, handler, mock_mound):
        """Test revalidation with default validator."""
        http_handler = MockHandler(body=b"")

        result = handler._handle_revalidate_node("node-123", http_handler)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["validator"] == "api"

    def test_revalidate_no_mound(self, handler_no_mound):
        """Test revalidation when mound not available."""
        http_handler = MockHandler(body=b"")

        result = handler_no_mound._handle_revalidate_node("node-123", http_handler)

        assert result.status_code == 503

    def test_revalidate_error(self, handler, mock_mound):
        """Test revalidation error handling."""
        mock_mound.mark_validated.side_effect = Exception("Validation failed")

        http_handler = MockHandler(body=b"")

        result = handler._handle_revalidate_node("node-123", http_handler)

        assert result.status_code == 500


# =============================================================================
# Test schedule_revalidation
# =============================================================================


class TestScheduleRevalidation:
    """Tests for schedule revalidation endpoint."""

    def test_schedule_success(self, handler, mock_mound):
        """Test successful revalidation scheduling."""
        mock_mound.schedule_revalidation.return_value = ["node-1", "node-2", "node-3"]

        body = json.dumps(
            {
                "node_ids": ["node-1", "node-2", "node-3"],
                "priority": "high",
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_schedule_revalidation(http_handler)

        assert result.status_code == 202
        data = parse_response(result)
        assert data["count"] == 3
        assert data["priority"] == "high"

    def test_schedule_missing_node_ids(self, handler):
        """Test scheduling without node_ids."""
        body = json.dumps({"priority": "high"}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_schedule_revalidation(http_handler)

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    def test_schedule_empty_node_ids(self, handler):
        """Test scheduling with empty node_ids list."""
        body = json.dumps({"node_ids": []}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_schedule_revalidation(http_handler)

        assert result.status_code == 400

    def test_schedule_invalid_priority(self, handler):
        """Test scheduling with invalid priority."""
        body = json.dumps(
            {
                "node_ids": ["node-1"],
                "priority": "urgent",  # Invalid
            }
        ).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_schedule_revalidation(http_handler)

        assert result.status_code == 400
        assert "priority" in parse_response(result)["error"]

    def test_schedule_default_priority(self, handler, mock_mound):
        """Test scheduling with default priority."""
        mock_mound.schedule_revalidation.return_value = ["node-1"]

        body = json.dumps({"node_ids": ["node-1"]}).encode()
        http_handler = MockHandler(body=body)

        result = handler._handle_schedule_revalidation(http_handler)

        assert result.status_code == 202
        data = parse_response(result)
        assert data["priority"] == "low"

    def test_schedule_no_mound(self, handler_no_mound):
        """Test scheduling when mound not available."""
        body = json.dumps({"node_ids": ["node-1"]}).encode()
        http_handler = MockHandler(body=body)

        result = handler_no_mound._handle_schedule_revalidation(http_handler)

        assert result.status_code == 503
