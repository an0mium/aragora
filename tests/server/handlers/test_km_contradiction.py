"""
Tests for Knowledge Mound Contradiction Detection handler endpoints.

Tests contradiction operations:
- POST /api/knowledge/mound/contradictions/detect - Trigger contradiction scan
- GET /api/knowledge/mound/contradictions - List unresolved contradictions
- POST /api/knowledge/mound/contradictions/:id/resolve - Resolve a contradiction
- GET /api/knowledge/mound/contradictions/stats - Get contradiction statistics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.contradiction import (
    ContradictionOperationsMixin,
)


class MockResolutionStrategy(str, Enum):
    """Mock resolution strategy for testing."""

    PREFER_NEWER = "prefer_newer"
    PREFER_HIGHER_CONFIDENCE = "prefer_higher_confidence"
    MERGE = "merge"
    HUMAN_REVIEW = "human_review"
    KEEP_BOTH = "keep_both"


@dataclass
class MockContradictionReport:
    """Mock contradiction report for testing."""

    workspace_id: str = "workspace-123"
    total_items_scanned: int = 100
    contradictions_found: int = 5
    scan_duration_ms: float = 250.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "total_items_scanned": self.total_items_scanned,
            "contradictions_found": self.contradictions_found,
            "scan_duration_ms": self.scan_duration_ms,
        }


@dataclass
class MockContradiction:
    """Mock contradiction for testing."""

    contradiction_id: str = "contradiction-123"
    item_a_id: str = "item-a"
    item_b_id: str = "item-b"
    severity: str = "high"
    confidence: float = 0.85
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "item_a_id": self.item_a_id,
            "item_b_id": self.item_b_id,
            "severity": self.severity,
            "confidence": self.confidence,
            "resolved": self.resolved,
        }


class MockMound:
    """Mock KnowledgeMound for testing."""

    def __init__(self):
        self.detect_contradictions = AsyncMock(return_value=MockContradictionReport())
        self.get_unresolved_contradictions = AsyncMock(
            return_value=[MockContradiction(), MockContradiction(contradiction_id="c-456")]
        )
        self.resolve_contradiction = AsyncMock(return_value=MockContradiction(resolved=True))
        self.get_contradiction_stats = MagicMock(
            return_value={
                "total_contradictions": 50,
                "unresolved": 12,
                "resolved_today": 5,
                "by_severity": {"high": 10, "medium": 25, "low": 15},
            }
        )


class MockContradictionHandler(ContradictionOperationsMixin):
    """Handler for testing ContradictionOperationsMixin."""

    def __init__(self):
        self.mound = MockMound()
        self.ctx = {"user_id": "test-user", "org_id": "test-org"}

    def _get_mound(self):
        return self.mound


class MockContradictionHandlerNoMound(ContradictionOperationsMixin):
    """Handler with no mound available."""

    def __init__(self):
        self.ctx = {}

    def _get_mound(self):
        return None


def parse_json_response(result):
    """Parse JSON response from HandlerResult dataclass."""
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


@pytest.fixture
def handler():
    """Create test handler with mocked mound."""
    return MockContradictionHandler()


@pytest.fixture
def handler_no_mound():
    """Create test handler without mound."""
    return MockContradictionHandlerNoMound()


# Mock the decorators to bypass RBAC and rate limiting for tests
@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock RBAC and rate limit decorators."""
    with (
        patch(
            "aragora.server.handlers.knowledge_base.mound.contradiction.require_permission",
            lambda perm: lambda fn: fn,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.contradiction.rate_limit",
            lambda **kwargs: lambda fn: fn,
        ),
    ):
        yield


class TestDetectContradictions:
    """Tests for detect_contradictions endpoint."""

    @pytest.mark.asyncio
    async def test_detect_success(self, handler):
        """Test successful contradiction detection."""
        test_handler = MockContradictionHandler()
        result = await test_handler.detect_contradictions(workspace_id="workspace-123")

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"
        assert data["contradictions_found"] == 5
        assert "scan_duration_ms" in data

    @pytest.mark.asyncio
    async def test_detect_with_item_ids(self, handler):
        """Test detection with specific item IDs."""
        test_handler = MockContradictionHandler()
        result = await test_handler.detect_contradictions(
            workspace_id="workspace-123",
            item_ids=["item-1", "item-2"],
        )

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"

        test_handler.mound.detect_contradictions.assert_called_once_with(
            workspace_id="workspace-123",
            item_ids=["item-1", "item-2"],
        )

    @pytest.mark.asyncio
    async def test_detect_missing_workspace_id(self, handler):
        """Test detection fails without workspace_id."""
        test_handler = MockContradictionHandler()
        result = await test_handler.detect_contradictions(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_detect_mound_unavailable(self, handler_no_mound):
        """Test detection when mound is unavailable."""
        result = await handler_no_mound.detect_contradictions(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_detect_mound_error(self, handler):
        """Test detection handles mound errors."""
        test_handler = MockContradictionHandler()
        test_handler.mound.detect_contradictions = AsyncMock(
            side_effect=Exception("Detection error")
        )

        result = await test_handler.detect_contradictions(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestListContradictions:
    """Tests for list_contradictions endpoint."""

    @pytest.mark.asyncio
    async def test_list_success(self, handler):
        """Test successful listing of contradictions."""
        test_handler = MockContradictionHandler()
        result = await test_handler.list_contradictions()

        data = parse_json_response(result)
        assert data["count"] == 2
        assert len(data["contradictions"]) == 2

    @pytest.mark.asyncio
    async def test_list_with_workspace_filter(self, handler):
        """Test listing with workspace filter."""
        test_handler = MockContradictionHandler()
        result = await test_handler.list_contradictions(workspace_id="workspace-123")

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"

    @pytest.mark.asyncio
    async def test_list_with_severity_filter(self, handler):
        """Test listing with severity filter."""
        test_handler = MockContradictionHandler()
        result = await test_handler.list_contradictions(min_severity="high")

        data = parse_json_response(result)
        assert data["min_severity"] == "high"

        test_handler.mound.get_unresolved_contradictions.assert_called_once_with(
            workspace_id=None,
            min_severity="high",
        )

    @pytest.mark.asyncio
    async def test_list_mound_unavailable(self, handler_no_mound):
        """Test listing when mound is unavailable."""
        result = await handler_no_mound.list_contradictions()

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_list_mound_error(self, handler):
        """Test listing handles mound errors."""
        test_handler = MockContradictionHandler()
        test_handler.mound.get_unresolved_contradictions = AsyncMock(
            side_effect=Exception("List error")
        )

        result = await test_handler.list_contradictions()

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestResolveContradiction:
    """Tests for resolve_contradiction endpoint."""

    @pytest.mark.asyncio
    async def test_resolve_success(self, handler):
        """Test successful resolution."""
        test_handler = MockContradictionHandler()

        with patch(
            "aragora.knowledge.mound.ops.contradiction.ResolutionStrategy",
            MockResolutionStrategy,
        ):
            result = await test_handler.resolve_contradiction(
                contradiction_id="contradiction-123",
                strategy="prefer_newer",
                resolved_by="user-123",
                notes="Resolved by admin",
            )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["contradiction"]["resolved"] is True

    @pytest.mark.asyncio
    async def test_resolve_missing_id(self, handler):
        """Test resolution fails without ID."""
        test_handler = MockContradictionHandler()

        with patch(
            "aragora.knowledge.mound.ops.contradiction.ResolutionStrategy",
            MockResolutionStrategy,
        ):
            result = await test_handler.resolve_contradiction(
                contradiction_id="",
                strategy="prefer_newer",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "contradiction_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_resolve_missing_strategy(self, handler):
        """Test resolution fails without strategy."""
        test_handler = MockContradictionHandler()

        with patch(
            "aragora.knowledge.mound.ops.contradiction.ResolutionStrategy",
            MockResolutionStrategy,
        ):
            result = await test_handler.resolve_contradiction(
                contradiction_id="contradiction-123",
                strategy="",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "strategy is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_resolve_invalid_strategy(self, handler):
        """Test resolution fails with invalid strategy."""
        test_handler = MockContradictionHandler()

        with patch(
            "aragora.knowledge.mound.ops.contradiction.ResolutionStrategy",
            MockResolutionStrategy,
        ):
            result = await test_handler.resolve_contradiction(
                contradiction_id="contradiction-123",
                strategy="invalid_strategy",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "Invalid strategy" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_resolve_not_found(self, handler):
        """Test resolution when contradiction not found."""
        test_handler = MockContradictionHandler()
        test_handler.mound.resolve_contradiction = AsyncMock(return_value=None)

        with patch(
            "aragora.knowledge.mound.ops.contradiction.ResolutionStrategy",
            MockResolutionStrategy,
        ):
            result = await test_handler.resolve_contradiction(
                contradiction_id="nonexistent",
                strategy="prefer_newer",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 404
        assert "not found" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_resolve_mound_unavailable(self, handler_no_mound):
        """Test resolution when mound is unavailable."""
        result = await handler_no_mound.resolve_contradiction(
            contradiction_id="contradiction-123",
            strategy="prefer_newer",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_resolve_mound_error(self, handler):
        """Test resolution handles mound errors."""
        test_handler = MockContradictionHandler()
        test_handler.mound.resolve_contradiction = AsyncMock(
            side_effect=Exception("Resolution error")
        )

        with patch(
            "aragora.knowledge.mound.ops.contradiction.ResolutionStrategy",
            MockResolutionStrategy,
        ):
            result = await test_handler.resolve_contradiction(
                contradiction_id="contradiction-123",
                strategy="prefer_newer",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestGetContradictionStats:
    """Tests for get_contradiction_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler):
        """Test successful stats retrieval."""
        test_handler = MockContradictionHandler()
        result = await test_handler.get_contradiction_stats()

        data = parse_json_response(result)
        assert data["total_contradictions"] == 50
        assert data["unresolved"] == 12
        assert "by_severity" in data

    @pytest.mark.asyncio
    async def test_get_stats_mound_unavailable(self, handler_no_mound):
        """Test stats when mound is unavailable."""
        result = await handler_no_mound.get_contradiction_stats()

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_stats_mound_error(self, handler):
        """Test stats handles mound errors."""
        test_handler = MockContradictionHandler()
        test_handler.mound.get_contradiction_stats = MagicMock(side_effect=Exception("Stats error"))

        result = await test_handler.get_contradiction_stats()

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, handler):
        """Test stats with no contradictions."""
        test_handler = MockContradictionHandler()
        test_handler.mound.get_contradiction_stats = MagicMock(
            return_value={
                "total_contradictions": 0,
                "unresolved": 0,
                "resolved_today": 0,
                "by_severity": {},
            }
        )

        result = await test_handler.get_contradiction_stats()

        data = parse_json_response(result)
        assert data["total_contradictions"] == 0
