"""
Tests for ContradictionOperationsMixin.

Tests contradiction detection API endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.contradiction import (
    ContradictionOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


class ResolutionStrategy(str, Enum):
    """Mock resolution strategy enum."""

    PREFER_NEWER = "prefer_newer"
    PREFER_HIGHER_CONFIDENCE = "prefer_higher_confidence"
    MERGE = "merge"
    HUMAN_REVIEW = "human_review"
    KEEP_BOTH = "keep_both"


@dataclass
class MockContradiction:
    """Mock contradiction object."""

    id: str
    item_a_id: str
    item_b_id: str
    workspace_id: str
    severity: str
    status: str = "unresolved"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "item_a_id": self.item_a_id,
            "item_b_id": self.item_b_id,
            "workspace_id": self.workspace_id,
            "severity": self.severity,
            "status": self.status,
        }


@dataclass
class MockContradictionReport:
    """Mock contradiction report."""

    workspace_id: str
    contradictions_found: int
    contradictions: List[MockContradiction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "contradictions_found": self.contradictions_found,
            "contradictions": [c.to_dict() for c in self.contradictions],
        }


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    detect_contradictions: AsyncMock = field(default_factory=AsyncMock)
    get_unresolved_contradictions: AsyncMock = field(default_factory=AsyncMock)
    resolve_contradiction: AsyncMock = field(default_factory=AsyncMock)
    get_contradiction_stats: MagicMock = field(default_factory=MagicMock)


class ContradictionHandler(ContradictionOperationsMixin):
    """Handler implementation for testing ContradictionOperationsMixin."""

    def __init__(self, mound: Optional[MockKnowledgeMound] = None):
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
    return ContradictionHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a test handler without mound."""
    return ContradictionHandler(mound=None)


# =============================================================================
# Test detect_contradictions
# =============================================================================


class TestDetectContradictions:
    """Tests for detect_contradictions endpoint."""

    @pytest.mark.asyncio
    async def test_detect_contradictions_success(self, handler, mock_mound):
        """Test successful contradiction detection."""
        mock_report = MockContradictionReport(
            workspace_id="ws-123",
            contradictions_found=2,
            contradictions=[
                MockContradiction(
                    id="c1",
                    item_a_id="item1",
                    item_b_id="item2",
                    workspace_id="ws-123",
                    severity="high",
                ),
            ],
        )
        mock_mound.detect_contradictions.return_value = mock_report

        result = await handler.detect_contradictions(workspace_id="ws-123")

        assert result.status_code == 200
        assert parse_response(result)["contradictions_found"] == 2
        mock_mound.detect_contradictions.assert_called_once_with(
            workspace_id="ws-123",
            item_ids=None,
        )

    @pytest.mark.asyncio
    async def test_detect_contradictions_with_item_ids(self, handler, mock_mound):
        """Test contradiction detection with specific item IDs."""
        mock_report = MockContradictionReport(
            workspace_id="ws-123",
            contradictions_found=0,
            contradictions=[],
        )
        mock_mound.detect_contradictions.return_value = mock_report

        result = await handler.detect_contradictions(
            workspace_id="ws-123",
            item_ids=["item1", "item2"],
        )

        assert result.status_code == 200
        mock_mound.detect_contradictions.assert_called_once_with(
            workspace_id="ws-123",
            item_ids=["item1", "item2"],
        )

    @pytest.mark.asyncio
    async def test_detect_contradictions_no_mound(self, handler_no_mound):
        """Test detection when mound not available."""
        result = await handler_no_mound.detect_contradictions(workspace_id="ws-123")

        assert result.status_code == 503
        assert "not available" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_detect_contradictions_missing_workspace(self, handler):
        """Test detection with missing workspace_id."""
        result = await handler.detect_contradictions(workspace_id="")

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_detect_contradictions_error(self, handler, mock_mound):
        """Test detection error handling."""
        mock_mound.detect_contradictions.side_effect = Exception("Database error")

        result = await handler.detect_contradictions(workspace_id="ws-123")

        assert result.status_code == 500


# =============================================================================
# Test list_contradictions
# =============================================================================


class TestListContradictions:
    """Tests for list_contradictions endpoint."""

    @pytest.mark.asyncio
    async def test_list_contradictions_success(self, handler, mock_mound):
        """Test successful contradiction listing."""
        mock_contradictions = [
            MockContradiction(
                id="c1",
                item_a_id="item1",
                item_b_id="item2",
                workspace_id="ws-123",
                severity="high",
            ),
            MockContradiction(
                id="c2",
                item_a_id="item3",
                item_b_id="item4",
                workspace_id="ws-123",
                severity="medium",
            ),
        ]
        mock_mound.get_unresolved_contradictions.return_value = mock_contradictions

        result = await handler.list_contradictions(workspace_id="ws-123")

        assert result.status_code == 200
        assert parse_response(result)["count"] == 2
        assert len(parse_response(result)["contradictions"]) == 2

    @pytest.mark.asyncio
    async def test_list_contradictions_with_severity_filter(self, handler, mock_mound):
        """Test listing with severity filter."""
        mock_mound.get_unresolved_contradictions.return_value = []

        result = await handler.list_contradictions(
            workspace_id="ws-123",
            min_severity="high",
        )

        assert result.status_code == 200
        mock_mound.get_unresolved_contradictions.assert_called_once_with(
            workspace_id="ws-123",
            min_severity="high",
        )

    @pytest.mark.asyncio
    async def test_list_contradictions_no_workspace(self, handler, mock_mound):
        """Test listing without workspace filter."""
        mock_mound.get_unresolved_contradictions.return_value = []

        result = await handler.list_contradictions()

        assert result.status_code == 200
        mock_mound.get_unresolved_contradictions.assert_called_once_with(
            workspace_id=None,
            min_severity=None,
        )

    @pytest.mark.asyncio
    async def test_list_contradictions_no_mound(self, handler_no_mound):
        """Test listing when mound not available."""
        result = await handler_no_mound.list_contradictions()

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_list_contradictions_error(self, handler, mock_mound):
        """Test listing error handling."""
        mock_mound.get_unresolved_contradictions.side_effect = Exception("Error")

        result = await handler.list_contradictions()

        assert result.status_code == 500


# =============================================================================
# Test resolve_contradiction
# =============================================================================


class TestResolveContradiction:
    """Tests for resolve_contradiction endpoint."""

    @pytest.mark.asyncio
    async def test_resolve_contradiction_success(self, handler, mock_mound):
        """Test successful contradiction resolution."""
        resolved = MockContradiction(
            id="c1",
            item_a_id="item1",
            item_b_id="item2",
            workspace_id="ws-123",
            severity="high",
            status="resolved",
        )
        mock_mound.resolve_contradiction.return_value = resolved

        result = await handler.resolve_contradiction(
            contradiction_id="c1",
            strategy="prefer_newer",
            resolved_by="user-123",
            notes="Newer data is more accurate",
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert parse_response(result)["contradiction"]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_resolve_contradiction_missing_id(self, handler):
        """Test resolution with missing contradiction_id."""
        result = await handler.resolve_contradiction(
            contradiction_id="",
            strategy="prefer_newer",
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_resolve_contradiction_missing_strategy(self, handler):
        """Test resolution with missing strategy."""
        result = await handler.resolve_contradiction(
            contradiction_id="c1",
            strategy="",
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_resolve_contradiction_invalid_strategy(self, handler):
        """Test resolution with invalid strategy."""
        result = await handler.resolve_contradiction(
            contradiction_id="c1",
            strategy="invalid_strategy",
        )

        assert result.status_code == 400
        assert "Invalid strategy" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_resolve_contradiction_not_found(self, handler, mock_mound):
        """Test resolution when contradiction not found."""
        mock_mound.resolve_contradiction.return_value = None

        result = await handler.resolve_contradiction(
            contradiction_id="c-nonexistent",
            strategy="prefer_newer",
        )

        assert result.status_code == 404
        assert "not found" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_resolve_contradiction_no_mound(self, handler_no_mound):
        """Test resolution when mound not available."""
        result = await handler_no_mound.resolve_contradiction(
            contradiction_id="c1",
            strategy="prefer_newer",
        )

        assert result.status_code == 503


# =============================================================================
# Test get_contradiction_stats
# =============================================================================


class TestGetContradictionStats:
    """Tests for get_contradiction_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_mound):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_detected": 50,
            "resolved": 30,
            "unresolved": 20,
            "by_severity": {
                "critical": 5,
                "high": 10,
                "medium": 15,
                "low": 20,
            },
        }
        mock_mound.get_contradiction_stats.return_value = mock_stats

        result = await handler.get_contradiction_stats()

        assert result.status_code == 200
        assert parse_response(result)["total_detected"] == 50
        assert parse_response(result)["resolved"] == 30

    @pytest.mark.asyncio
    async def test_get_stats_no_mound(self, handler_no_mound):
        """Test stats when mound not available."""
        result = await handler_no_mound.get_contradiction_stats()

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_stats_error(self, handler, mock_mound):
        """Test stats error handling."""
        mock_mound.get_contradiction_stats.side_effect = Exception("Stats error")

        result = await handler.get_contradiction_stats()

        assert result.status_code == 500
