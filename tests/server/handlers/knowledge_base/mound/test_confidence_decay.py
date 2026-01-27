"""
Tests for ConfidenceDecayOperationsMixin.

Tests confidence decay API endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.confidence_decay import (
    ConfidenceDecayOperationsMixin,
)


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


class ConfidenceEvent(str, Enum):
    """Mock confidence event enum."""

    CREATED = "created"
    ACCESSED = "accessed"
    CITED = "cited"
    VALIDATED = "validated"
    INVALIDATED = "invalidated"
    CONTRADICTED = "contradicted"
    UPDATED = "updated"


@dataclass
class MockDecayReport:
    """Mock decay report."""

    workspace_id: str
    items_processed: int
    items_decayed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "items_processed": self.items_processed,
            "items_decayed": self.items_decayed,
        }


@dataclass
class MockConfidenceAdjustment:
    """Mock confidence adjustment."""

    item_id: str
    event: str
    old_confidence: float
    new_confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "event": self.event,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "reason": self.reason,
        }


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    apply_confidence_decay: AsyncMock = field(default_factory=AsyncMock)
    record_confidence_event: AsyncMock = field(default_factory=AsyncMock)
    get_confidence_history: AsyncMock = field(default_factory=AsyncMock)
    get_decay_stats: MagicMock = field(default_factory=MagicMock)


class TestConfidenceDecayHandler(ConfidenceDecayOperationsMixin):
    """Test implementation of ConfidenceDecayOperationsMixin."""

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
    return TestConfidenceDecayHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a test handler without mound."""
    return TestConfidenceDecayHandler(mound=None)


# =============================================================================
# Test apply_confidence_decay_endpoint
# =============================================================================


class TestApplyConfidenceDecay:
    """Tests for apply_confidence_decay_endpoint."""

    @pytest.mark.asyncio
    async def test_apply_decay_success(self, handler, mock_mound):
        """Test successful confidence decay application."""
        mock_report = MockDecayReport(
            workspace_id="ws-123",
            items_processed=100,
            items_decayed=15,
        )
        mock_mound.apply_confidence_decay.return_value = mock_report

        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-123")

        assert result.status_code == 200
        assert parse_response(result)["items_processed"] == 100
        assert parse_response(result)["items_decayed"] == 15
        mock_mound.apply_confidence_decay.assert_called_once_with(
            workspace_id="ws-123",
            force=False,
        )

    @pytest.mark.asyncio
    async def test_apply_decay_with_force(self, handler, mock_mound):
        """Test forced confidence decay application."""
        mock_report = MockDecayReport(
            workspace_id="ws-123",
            items_processed=100,
            items_decayed=25,
        )
        mock_mound.apply_confidence_decay.return_value = mock_report

        result = await handler.apply_confidence_decay_endpoint(
            workspace_id="ws-123",
            force=True,
        )

        assert result.status_code == 200
        mock_mound.apply_confidence_decay.assert_called_once_with(
            workspace_id="ws-123",
            force=True,
        )

    @pytest.mark.asyncio
    async def test_apply_decay_no_mound(self, handler_no_mound):
        """Test decay when mound not available."""
        result = await handler_no_mound.apply_confidence_decay_endpoint(workspace_id="ws-123")

        assert result.status_code == 503
        assert "not available" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_apply_decay_missing_workspace(self, handler):
        """Test decay with missing workspace_id."""
        result = await handler.apply_confidence_decay_endpoint(workspace_id="")

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_apply_decay_error(self, handler, mock_mound):
        """Test decay error handling."""
        mock_mound.apply_confidence_decay.side_effect = Exception("Decay error")

        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-123")

        assert result.status_code == 500


# =============================================================================
# Test record_confidence_event
# =============================================================================


class TestRecordConfidenceEvent:
    """Tests for record_confidence_event endpoint."""

    @pytest.mark.asyncio
    async def test_record_event_with_adjustment(self, handler, mock_mound):
        """Test recording event that causes adjustment."""
        # Import the real enum from the knowledge module
        from aragora.knowledge.mound.ops.confidence_decay import ConfidenceEvent as RealEvent

        adjustment = MockConfidenceAdjustment(
            item_id="item-123",
            event="accessed",
            old_confidence=0.8,
            new_confidence=0.85,
            reason="Access boost",
        )
        mock_mound.record_confidence_event.return_value = adjustment

        result = await handler.record_confidence_event(
            item_id="item-123",
            event="accessed",
            reason="User accessed item",
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert parse_response(result)["adjusted"] is True
        assert parse_response(result)["adjustment"]["new_confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_record_event_no_adjustment(self, handler, mock_mound):
        """Test recording event with no confidence change."""
        mock_mound.record_confidence_event.return_value = None

        result = await handler.record_confidence_event(
            item_id="item-123",
            event="accessed",
        )

        assert result.status_code == 200
        assert parse_response(result)["success"] is True
        assert parse_response(result)["adjusted"] is False
        assert "No confidence change" in parse_response(result)["message"]

    @pytest.mark.asyncio
    async def test_record_event_missing_item_id(self, handler):
        """Test recording with missing item_id."""
        result = await handler.record_confidence_event(
            item_id="",
            event="accessed",
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_record_event_missing_event(self, handler):
        """Test recording with missing event."""
        result = await handler.record_confidence_event(
            item_id="item-123",
            event="",
        )

        assert result.status_code == 400
        assert "required" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_record_event_invalid_event(self, handler):
        """Test recording with invalid event type."""
        result = await handler.record_confidence_event(
            item_id="item-123",
            event="invalid_event",
        )

        assert result.status_code == 400
        assert "Invalid event" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_record_event_no_mound(self, handler_no_mound):
        """Test recording when mound not available."""
        result = await handler_no_mound.record_confidence_event(
            item_id="item-123",
            event="accessed",
        )

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_record_event_error(self, handler, mock_mound):
        """Test recording error handling."""
        mock_mound.record_confidence_event.side_effect = Exception("Database error")

        result = await handler.record_confidence_event(
            item_id="item-123",
            event="accessed",
        )

        assert result.status_code == 500


# =============================================================================
# Test get_confidence_history
# =============================================================================


class TestGetConfidenceHistory:
    """Tests for get_confidence_history endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_success(self, handler, mock_mound):
        """Test successful history retrieval."""
        adjustments = [
            MockConfidenceAdjustment(
                item_id="item-123",
                event="accessed",
                old_confidence=0.7,
                new_confidence=0.75,
                reason="Access boost",
            ),
            MockConfidenceAdjustment(
                item_id="item-123",
                event="cited",
                old_confidence=0.75,
                new_confidence=0.85,
                reason="Citation boost",
            ),
        ]
        mock_mound.get_confidence_history.return_value = adjustments

        result = await handler.get_confidence_history(
            item_id="item-123",
            limit=100,
        )

        assert result.status_code == 200
        assert parse_response(result)["count"] == 2
        assert len(parse_response(result)["adjustments"]) == 2

    @pytest.mark.asyncio
    async def test_get_history_with_event_filter(self, handler, mock_mound):
        """Test history with event type filter."""
        mock_mound.get_confidence_history.return_value = []

        result = await handler.get_confidence_history(
            event_type="accessed",
            limit=50,
        )

        assert result.status_code == 200
        assert parse_response(result)["filters"]["event_type"] == "accessed"

    @pytest.mark.asyncio
    async def test_get_history_invalid_event_type(self, handler):
        """Test history with invalid event type."""
        result = await handler.get_confidence_history(event_type="invalid")

        assert result.status_code == 400
        assert "Invalid event_type" in parse_response(result)["error"]

    @pytest.mark.asyncio
    async def test_get_history_no_mound(self, handler_no_mound):
        """Test history when mound not available."""
        result = await handler_no_mound.get_confidence_history()

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_history_error(self, handler, mock_mound):
        """Test history error handling."""
        mock_mound.get_confidence_history.side_effect = Exception("Error")

        result = await handler.get_confidence_history()

        assert result.status_code == 500


# =============================================================================
# Test get_decay_stats
# =============================================================================


class TestGetDecayStats:
    """Tests for get_decay_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_mound):
        """Test successful stats retrieval."""
        mock_stats = {
            "total_adjustments": 500,
            "items_affected": 200,
            "average_decay_rate": 0.02,
            "by_event": {
                "accessed": 150,
                "cited": 100,
                "validated": 50,
            },
        }
        mock_mound.get_decay_stats.return_value = mock_stats

        result = await handler.get_decay_stats()

        assert result.status_code == 200
        assert parse_response(result)["total_adjustments"] == 500
        assert parse_response(result)["average_decay_rate"] == 0.02

    @pytest.mark.asyncio
    async def test_get_stats_no_mound(self, handler_no_mound):
        """Test stats when mound not available."""
        result = await handler_no_mound.get_decay_stats()

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_get_stats_error(self, handler, mock_mound):
        """Test stats error handling."""
        mock_mound.get_decay_stats.side_effect = Exception("Stats error")

        result = await handler.get_decay_stats()

        assert result.status_code == 500
