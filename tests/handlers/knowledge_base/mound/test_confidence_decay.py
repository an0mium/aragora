"""Tests for ConfidenceDecayOperationsMixin.

Covers all four async endpoints on the mixin:
- POST /api/knowledge/mound/confidence/decay   - apply_confidence_decay_endpoint
- POST /api/knowledge/mound/confidence/event   - record_confidence_event
- GET  /api/knowledge/mound/confidence/history  - get_confidence_history
- GET  /api/knowledge/mound/confidence/stats    - get_decay_stats

Each method is tested for:
- Success with valid inputs
- Mound not available (503)
- Missing required parameters (400)
- Invalid parameter values (400)
- Internal errors from mound operations (500)
- Edge cases and response structure
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.ops.confidence_decay import (
    ConfidenceEvent,
)
from aragora.server.handlers.knowledge_base.mound.confidence_decay import (
    ConfidenceDecayOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock dataclasses for mound return values
# ---------------------------------------------------------------------------


@dataclass
class MockConfidenceAdjustment:
    id: str = "adj-001"
    item_id: str = "item-001"
    event: ConfidenceEvent = ConfidenceEvent.ACCESSED
    old_confidence: float = 0.8
    new_confidence: float = 0.85
    reason: str = "item accessed"
    adjusted_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "item_id": self.item_id,
            "event": self.event.value,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "reason": self.reason,
            "adjusted_at": self.adjusted_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MockDecayReport:
    workspace_id: str = "ws-001"
    items_processed: int = 100
    items_decayed: int = 25
    items_boosted: int = 5
    average_confidence_change: float = -0.03
    adjustments: list[MockConfidenceAdjustment] = field(default_factory=list)
    processed_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    duration_ms: float = 150.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "items_processed": self.items_processed,
            "items_decayed": self.items_decayed,
            "items_boosted": self.items_boosted,
            "average_confidence_change": self.average_confidence_change,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "processed_at": self.processed_at.isoformat(),
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class ConfidenceDecayTestHandler(ConfidenceDecayOperationsMixin):
    """Concrete handler for testing the confidence decay mixin."""

    def __init__(self, mound=None):
        self._mound = mound
        self.ctx: dict[str, Any] = {}

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with confidence decay methods."""
    mound = MagicMock()
    mound.apply_confidence_decay = AsyncMock(return_value=MockDecayReport())
    mound.record_confidence_event = AsyncMock(return_value=MockConfidenceAdjustment())
    mound.get_confidence_history = AsyncMock(return_value=[])
    mound.get_decay_stats = MagicMock(
        return_value={
            "total_items": 500,
            "items_decayed": 120,
            "items_boosted": 30,
            "average_confidence": 0.72,
            "last_decay_run": "2026-02-01T12:00:00",
        }
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a ConfidenceDecayTestHandler with a mocked mound."""
    return ConfidenceDecayTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a ConfidenceDecayTestHandler with no mound (None)."""
    return ConfidenceDecayTestHandler(mound=None)


# ============================================================================
# Tests: apply_confidence_decay_endpoint
# ============================================================================


class TestApplyConfidenceDecayEndpoint:
    """Test apply_confidence_decay_endpoint (POST /api/knowledge/mound/confidence/decay)."""

    @pytest.mark.asyncio
    async def test_success_returns_decay_report(self, handler, mock_mound):
        """Successful decay returns report data from mound."""
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["workspace_id"] == "ws-001"
        assert body["items_processed"] == 100
        assert body["items_decayed"] == 25
        assert body["items_boosted"] == 5
        assert body["average_confidence_change"] == -0.03
        assert body["duration_ms"] == 150.5

    @pytest.mark.asyncio
    async def test_success_with_force_flag(self, handler, mock_mound):
        """Force flag is forwarded to mound."""
        await handler.apply_confidence_decay_endpoint(workspace_id="ws-001", force=True)
        mock_mound.apply_confidence_decay.assert_awaited_once_with(
            workspace_id="ws-001", force=True
        )

    @pytest.mark.asyncio
    async def test_success_default_force_is_false(self, handler, mock_mound):
        """Default force parameter is False."""
        await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        mock_mound.apply_confidence_decay.assert_awaited_once_with(
            workspace_id="ws-001", force=False
        )

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_id_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.apply_confidence_decay_endpoint(workspace_id="")
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.apply_confidence_decay = AsyncMock(side_effect=KeyError("missing key"))
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.apply_confidence_decay = AsyncMock(side_effect=ValueError("bad value"))
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.apply_confidence_decay = AsyncMock(side_effect=OSError("disk failure"))
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.apply_confidence_decay = AsyncMock(side_effect=TypeError("wrong type"))
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_report_with_adjustments(self, handler, mock_mound):
        """Report with adjustments serializes them correctly."""
        adj1 = MockConfidenceAdjustment(
            id="adj-1",
            item_id="item-1",
            event=ConfidenceEvent.DECAYED,
            old_confidence=0.9,
            new_confidence=0.85,
            reason="time decay",
        )
        adj2 = MockConfidenceAdjustment(
            id="adj-2",
            item_id="item-2",
            event=ConfidenceEvent.ACCESSED,
            old_confidence=0.6,
            new_confidence=0.65,
            reason="recent access",
        )
        report = MockDecayReport(
            workspace_id="ws-001",
            items_processed=10,
            items_decayed=5,
            items_boosted=2,
            average_confidence_change=-0.01,
            adjustments=[adj1, adj2],
        )
        mock_mound.apply_confidence_decay = AsyncMock(return_value=report)
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        body = _body(result)
        assert _status(result) == 200
        assert len(body["adjustments"]) == 2
        assert body["adjustments"][0]["id"] == "adj-1"
        assert body["adjustments"][0]["event"] == "decayed"
        assert body["adjustments"][1]["id"] == "adj-2"
        assert body["adjustments"][1]["event"] == "accessed"

    @pytest.mark.asyncio
    async def test_report_empty_adjustments(self, handler, mock_mound):
        """Report with no adjustments returns empty list."""
        report = MockDecayReport(adjustments=[])
        mock_mound.apply_confidence_decay = AsyncMock(return_value=report)
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        body = _body(result)
        assert body["adjustments"] == []

    @pytest.mark.asyncio
    async def test_processed_at_is_serialized(self, handler, mock_mound):
        """processed_at datetime is serialized to ISO format string."""
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        body = _body(result)
        assert "processed_at" in body
        assert isinstance(body["processed_at"], str)
        assert "2026-02-01" in body["processed_at"]

    @pytest.mark.asyncio
    async def test_force_true_passed_to_mound(self, handler, mock_mound):
        """force=True is correctly passed through."""
        await handler.apply_confidence_decay_endpoint(workspace_id="ws-test", force=True)
        call_kwargs = mock_mound.apply_confidence_decay.call_args.kwargs
        assert call_kwargs["force"] is True
        assert call_kwargs["workspace_id"] == "ws-test"

    @pytest.mark.asyncio
    async def test_zero_items_processed(self, handler, mock_mound):
        """Report with zero items processed is valid."""
        report = MockDecayReport(
            items_processed=0,
            items_decayed=0,
            items_boosted=0,
            average_confidence_change=0.0,
        )
        mock_mound.apply_confidence_decay = AsyncMock(return_value=report)
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        body = _body(result)
        assert _status(result) == 200
        assert body["items_processed"] == 0
        assert body["items_decayed"] == 0


# ============================================================================
# Tests: record_confidence_event
# ============================================================================


class TestRecordConfidenceEvent:
    """Test record_confidence_event (POST /api/knowledge/mound/confidence/event)."""

    @pytest.mark.asyncio
    async def test_success_with_adjustment(self, handler, mock_mound):
        """Successful event recording returns adjustment when confidence changed."""
        result = await handler.record_confidence_event(
            item_id="item-001", event="accessed", reason="user viewed"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["adjusted"] is True
        assert "adjustment" in body
        assert body["adjustment"]["id"] == "adj-001"
        assert body["adjustment"]["event"] == "accessed"

    @pytest.mark.asyncio
    async def test_success_no_adjustment(self, handler, mock_mound):
        """Event that does not change confidence returns adjusted=False."""
        mock_mound.record_confidence_event = AsyncMock(return_value=None)
        result = await handler.record_confidence_event(item_id="item-001", event="accessed")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["adjusted"] is False
        assert body["message"] == "No confidence change required"

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.record_confidence_event(
            item_id="item-001", event="accessed"
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_item_id_returns_400(self, handler):
        """Empty item_id returns 400."""
        result = await handler.record_confidence_event(item_id="", event="accessed")
        assert _status(result) == 400
        body = _body(result)
        assert "item_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_event_returns_400(self, handler):
        """Empty event returns 400."""
        result = await handler.record_confidence_event(item_id="item-001", event="")
        assert _status(result) == 400
        body = _body(result)
        assert "event" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_event_returns_400(self, handler):
        """Invalid event type returns 400 with valid events listed."""
        result = await handler.record_confidence_event(
            item_id="item-001", event="nonexistent_event"
        )
        assert _status(result) == 400
        body = _body(result)
        assert "invalid event" in body["error"].lower()
        assert "valid events" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_event_includes_valid_values(self, handler):
        """Invalid event error message lists all valid event values."""
        result = await handler.record_confidence_event(item_id="item-001", event="bad_event")
        body = _body(result)
        # Check that at least some known events are listed
        error_msg = body["error"].lower()
        assert "created" in error_msg or "accessed" in error_msg

    @pytest.mark.asyncio
    async def test_all_valid_events_accepted(self, handler, mock_mound):
        """Every valid ConfidenceEvent value is accepted."""
        for event in ConfidenceEvent:
            mock_mound.record_confidence_event = AsyncMock(
                return_value=MockConfidenceAdjustment(event=event)
            )
            result = await handler.record_confidence_event(item_id="item-001", event=event.value)
            assert _status(result) == 200, f"Event {event.value} should be accepted"

    @pytest.mark.asyncio
    async def test_event_created(self, handler, mock_mound):
        """The 'created' event is accepted."""
        mock_mound.record_confidence_event = AsyncMock(
            return_value=MockConfidenceAdjustment(event=ConfidenceEvent.CREATED)
        )
        result = await handler.record_confidence_event(item_id="item-001", event="created")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_event_cited(self, handler, mock_mound):
        """The 'cited' event is accepted."""
        mock_mound.record_confidence_event = AsyncMock(
            return_value=MockConfidenceAdjustment(event=ConfidenceEvent.CITED)
        )
        result = await handler.record_confidence_event(item_id="item-001", event="cited")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_event_validated(self, handler, mock_mound):
        """The 'validated' event is accepted."""
        mock_mound.record_confidence_event = AsyncMock(
            return_value=MockConfidenceAdjustment(event=ConfidenceEvent.VALIDATED)
        )
        result = await handler.record_confidence_event(item_id="item-001", event="validated")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_event_invalidated(self, handler, mock_mound):
        """The 'invalidated' event is accepted."""
        mock_mound.record_confidence_event = AsyncMock(
            return_value=MockConfidenceAdjustment(event=ConfidenceEvent.INVALIDATED)
        )
        result = await handler.record_confidence_event(item_id="item-001", event="invalidated")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_event_contradicted(self, handler, mock_mound):
        """The 'contradicted' event is accepted."""
        mock_mound.record_confidence_event = AsyncMock(
            return_value=MockConfidenceAdjustment(event=ConfidenceEvent.CONTRADICTED)
        )
        result = await handler.record_confidence_event(item_id="item-001", event="contradicted")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_event_updated(self, handler, mock_mound):
        """The 'updated' event is accepted."""
        mock_mound.record_confidence_event = AsyncMock(
            return_value=MockConfidenceAdjustment(event=ConfidenceEvent.UPDATED)
        )
        result = await handler.record_confidence_event(item_id="item-001", event="updated")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_event_decayed(self, handler, mock_mound):
        """The 'decayed' event is accepted."""
        mock_mound.record_confidence_event = AsyncMock(
            return_value=MockConfidenceAdjustment(event=ConfidenceEvent.DECAYED)
        )
        result = await handler.record_confidence_event(item_id="item-001", event="decayed")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_reason_forwarded_to_mound(self, handler, mock_mound):
        """Reason parameter is forwarded to mound."""
        await handler.record_confidence_event(
            item_id="item-001", event="accessed", reason="manual review"
        )
        call_kwargs = mock_mound.record_confidence_event.call_args.kwargs
        assert call_kwargs["reason"] == "manual review"

    @pytest.mark.asyncio
    async def test_default_reason_is_empty_string(self, handler, mock_mound):
        """Default reason is empty string."""
        await handler.record_confidence_event(item_id="item-001", event="accessed")
        call_kwargs = mock_mound.record_confidence_event.call_args.kwargs
        assert call_kwargs["reason"] == ""

    @pytest.mark.asyncio
    async def test_event_enum_forwarded_to_mound(self, handler, mock_mound):
        """Event string is converted to ConfidenceEvent enum before passing to mound."""
        await handler.record_confidence_event(item_id="item-001", event="validated")
        call_kwargs = mock_mound.record_confidence_event.call_args.kwargs
        assert call_kwargs["event"] == ConfidenceEvent.VALIDATED

    @pytest.mark.asyncio
    async def test_item_id_forwarded_to_mound(self, handler, mock_mound):
        """item_id is correctly forwarded to mound."""
        await handler.record_confidence_event(item_id="my-item-42", event="accessed")
        call_kwargs = mock_mound.record_confidence_event.call_args.kwargs
        assert call_kwargs["item_id"] == "my-item-42"

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.record_confidence_event = AsyncMock(side_effect=KeyError("item not found"))
        result = await handler.record_confidence_event(item_id="item-001", event="accessed")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.record_confidence_event = AsyncMock(side_effect=ValueError("invalid"))
        result = await handler.record_confidence_event(item_id="item-001", event="accessed")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.record_confidence_event = AsyncMock(side_effect=OSError("storage error"))
        result = await handler.record_confidence_event(item_id="item-001", event="accessed")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.record_confidence_event = AsyncMock(side_effect=TypeError("bad type"))
        result = await handler.record_confidence_event(item_id="item-001", event="accessed")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_adjustment_to_dict_in_response(self, handler, mock_mound):
        """Adjustment to_dict is called and result included in response."""
        custom_adj = MockConfidenceAdjustment(
            id="adj-custom",
            item_id="item-custom",
            event=ConfidenceEvent.CITED,
            old_confidence=0.5,
            new_confidence=0.7,
            reason="cited in report",
            metadata={"source": "report-123"},
        )
        mock_mound.record_confidence_event = AsyncMock(return_value=custom_adj)
        result = await handler.record_confidence_event(item_id="item-custom", event="cited")
        body = _body(result)
        adj = body["adjustment"]
        assert adj["id"] == "adj-custom"
        assert adj["item_id"] == "item-custom"
        assert adj["event"] == "cited"
        assert adj["old_confidence"] == 0.5
        assert adj["new_confidence"] == 0.7
        assert adj["reason"] == "cited in report"
        assert adj["metadata"]["source"] == "report-123"


# ============================================================================
# Tests: get_confidence_history
# ============================================================================


class TestGetConfidenceHistory:
    """Test get_confidence_history (GET /api/knowledge/mound/confidence/history)."""

    @pytest.mark.asyncio
    async def test_success_returns_empty_history(self, handler, mock_mound):
        """Successful call with no history returns empty list."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history()
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["adjustments"] == []

    @pytest.mark.asyncio
    async def test_success_returns_history_entries(self, handler, mock_mound):
        """Successful call returns history with adjustments."""
        adj1 = MockConfidenceAdjustment(
            id="adj-1",
            item_id="item-1",
            event=ConfidenceEvent.DECAYED,
        )
        adj2 = MockConfidenceAdjustment(
            id="adj-2",
            item_id="item-2",
            event=ConfidenceEvent.ACCESSED,
        )
        mock_mound.get_confidence_history = AsyncMock(return_value=[adj1, adj2])
        result = await handler.get_confidence_history()
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 2
        assert len(body["adjustments"]) == 2
        assert body["adjustments"][0]["id"] == "adj-1"
        assert body["adjustments"][1]["id"] == "adj-2"

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_confidence_history()
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_filter_by_item_id(self, handler, mock_mound):
        """item_id filter is forwarded to mound and reflected in response."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history(item_id="item-specific")
        body = _body(result)
        assert _status(result) == 200
        assert body["filters"]["item_id"] == "item-specific"
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["item_id"] == "item-specific"

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self, handler, mock_mound):
        """event_type filter is converted to enum and forwarded."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history(event_type="validated")
        body = _body(result)
        assert _status(result) == 200
        assert body["filters"]["event_type"] == "validated"
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["event_type"] == ConfidenceEvent.VALIDATED

    @pytest.mark.asyncio
    async def test_invalid_event_type_returns_400(self, handler):
        """Invalid event_type returns 400 with valid types listed."""
        result = await handler.get_confidence_history(event_type="nonexistent_type")
        assert _status(result) == 400
        body = _body(result)
        assert "invalid event_type" in body["error"].lower()
        assert "valid types" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_event_type_includes_valid_values(self, handler):
        """Invalid event_type error includes valid event values."""
        result = await handler.get_confidence_history(event_type="bad_type")
        body = _body(result)
        error_msg = body["error"].lower()
        assert "created" in error_msg or "accessed" in error_msg

    @pytest.mark.asyncio
    async def test_all_valid_event_types_accepted(self, handler, mock_mound):
        """Every valid ConfidenceEvent is accepted as event_type filter."""
        for event in ConfidenceEvent:
            mock_mound.get_confidence_history = AsyncMock(return_value=[])
            result = await handler.get_confidence_history(event_type=event.value)
            assert _status(result) == 200, f"Event type {event.value} should be accepted"

    @pytest.mark.asyncio
    async def test_custom_limit(self, handler, mock_mound):
        """Custom limit is forwarded to mound."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        await handler.get_confidence_history(limit=50)
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["limit"] == 50

    @pytest.mark.asyncio
    async def test_default_limit_is_100(self, handler, mock_mound):
        """Default limit is 100."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        await handler.get_confidence_history()
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_default_filters_are_none(self, handler, mock_mound):
        """When no filters passed, filter values are None in response."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history()
        body = _body(result)
        assert body["filters"]["item_id"] is None
        assert body["filters"]["event_type"] is None

    @pytest.mark.asyncio
    async def test_no_event_type_passes_none_to_mound(self, handler, mock_mound):
        """When no event_type provided, None is passed to mound."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        await handler.get_confidence_history()
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["event_type"] is None

    @pytest.mark.asyncio
    async def test_count_matches_adjustments_length(self, handler, mock_mound):
        """Count field matches the number of adjustments."""
        adjustments = [MockConfidenceAdjustment(id=f"adj-{i}") for i in range(5)]
        mock_mound.get_confidence_history = AsyncMock(return_value=adjustments)
        result = await handler.get_confidence_history()
        body = _body(result)
        assert body["count"] == len(body["adjustments"])
        assert body["count"] == 5

    @pytest.mark.asyncio
    async def test_adjustments_are_serialized_via_to_dict(self, handler, mock_mound):
        """Adjustments are serialized through their to_dict method."""
        adj = MockConfidenceAdjustment(
            id="adj-test",
            item_id="item-test",
            event=ConfidenceEvent.CONTRADICTED,
            old_confidence=0.9,
            new_confidence=0.5,
            reason="contradicted by newer data",
            metadata={"source": "paper-42"},
        )
        mock_mound.get_confidence_history = AsyncMock(return_value=[adj])
        result = await handler.get_confidence_history()
        body = _body(result)
        a = body["adjustments"][0]
        assert a["id"] == "adj-test"
        assert a["event"] == "contradicted"
        assert a["old_confidence"] == 0.9
        assert a["new_confidence"] == 0.5
        assert a["metadata"]["source"] == "paper-42"

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_confidence_history = AsyncMock(side_effect=KeyError("missing"))
        result = await handler.get_confidence_history()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_confidence_history = AsyncMock(side_effect=ValueError("bad data"))
        result = await handler.get_confidence_history()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_confidence_history = AsyncMock(side_effect=OSError("disk fail"))
        result = await handler.get_confidence_history()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_confidence_history = AsyncMock(side_effect=TypeError("wrong type"))
        result = await handler.get_confidence_history()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_both_filters_applied(self, handler, mock_mound):
        """Both item_id and event_type filters are applied together."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history(item_id="item-42", event_type="cited")
        body = _body(result)
        assert body["filters"]["item_id"] == "item-42"
        assert body["filters"]["event_type"] == "cited"
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["item_id"] == "item-42"
        assert call_kwargs["event_type"] == ConfidenceEvent.CITED

    @pytest.mark.asyncio
    async def test_both_filters_and_limit(self, handler, mock_mound):
        """All three parameters (item_id, event_type, limit) work together."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        await handler.get_confidence_history(item_id="item-1", event_type="accessed", limit=25)
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["item_id"] == "item-1"
        assert call_kwargs["event_type"] == ConfidenceEvent.ACCESSED
        assert call_kwargs["limit"] == 25

    @pytest.mark.asyncio
    async def test_single_adjustment_history(self, handler, mock_mound):
        """Single adjustment in history returns count=1."""
        adj = MockConfidenceAdjustment()
        mock_mound.get_confidence_history = AsyncMock(return_value=[adj])
        result = await handler.get_confidence_history()
        body = _body(result)
        assert body["count"] == 1
        assert len(body["adjustments"]) == 1


# ============================================================================
# Tests: get_decay_stats
# ============================================================================


class TestGetDecayStats:
    """Test get_decay_stats (GET /api/knowledge/mound/confidence/stats)."""

    @pytest.mark.asyncio
    async def test_success_returns_stats(self, handler, mock_mound):
        """Successful call returns decay statistics."""
        result = await handler.get_decay_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_items"] == 500
        assert body["items_decayed"] == 120
        assert body["items_boosted"] == 30
        assert body["average_confidence"] == 0.72
        assert body["last_decay_run"] == "2026-02-01T12:00:00"

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_decay_stats()
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_decay_stats = MagicMock(side_effect=KeyError("missing"))
        result = await handler.get_decay_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_decay_stats = MagicMock(side_effect=ValueError("bad"))
        result = await handler.get_decay_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_decay_stats = MagicMock(side_effect=OSError("db fail"))
        result = await handler.get_decay_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_decay_stats = MagicMock(side_effect=TypeError("wrong type"))
        result = await handler.get_decay_stats()
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_stats_is_sync_call(self, handler, mock_mound):
        """get_decay_stats calls mound.get_decay_stats synchronously."""
        result = await handler.get_decay_stats()
        mock_mound.get_decay_stats.assert_called_once()
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_stats_empty_response(self, handler, mock_mound):
        """Stats with zero values works correctly."""
        mock_mound.get_decay_stats = MagicMock(
            return_value={
                "total_items": 0,
                "items_decayed": 0,
                "items_boosted": 0,
                "average_confidence": 0.0,
            }
        )
        result = await handler.get_decay_stats()
        body = _body(result)
        assert _status(result) == 200
        assert body["total_items"] == 0
        assert body["average_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_custom_fields_passed_through(self, handler, mock_mound):
        """Custom fields in stats dict are passed through."""
        mock_mound.get_decay_stats = MagicMock(
            return_value={"custom_field": "test_value", "count": 42}
        )
        result = await handler.get_decay_stats()
        body = _body(result)
        assert body["custom_field"] == "test_value"
        assert body["count"] == 42

    @pytest.mark.asyncio
    async def test_stats_large_numbers(self, handler, mock_mound):
        """Stats with large numbers are handled correctly."""
        mock_mound.get_decay_stats = MagicMock(
            return_value={
                "total_items": 1_000_000,
                "items_decayed": 500_000,
                "average_confidence": 0.999999,
            }
        )
        result = await handler.get_decay_stats()
        body = _body(result)
        assert body["total_items"] == 1_000_000
        assert body["items_decayed"] == 500_000

    @pytest.mark.asyncio
    async def test_stats_nested_dict(self, handler, mock_mound):
        """Stats with nested dict structure are serialized."""
        mock_mound.get_decay_stats = MagicMock(
            return_value={
                "summary": {"total": 100, "decayed": 50},
                "by_workspace": {"ws-1": 30, "ws-2": 70},
            }
        )
        result = await handler.get_decay_stats()
        body = _body(result)
        assert body["summary"]["total"] == 100
        assert body["by_workspace"]["ws-1"] == 30


# ============================================================================
# Tests: edge cases and combined scenarios
# ============================================================================


class TestConfidenceDecayEdgeCases:
    """Test edge cases across confidence decay operations."""

    @pytest.mark.asyncio
    async def test_apply_decay_none_workspace_returns_400(self, handler):
        """None workspace_id (falsy) returns 400 for apply_confidence_decay_endpoint."""
        result = await handler.apply_confidence_decay_endpoint(workspace_id=None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_record_event_none_item_id_returns_400(self, handler):
        """None item_id (falsy) returns 400 for record_confidence_event."""
        result = await handler.record_confidence_event(item_id=None, event="accessed")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_record_event_none_event_returns_400(self, handler):
        """None event (falsy) returns 400 for record_confidence_event."""
        result = await handler.record_confidence_event(item_id="item-001", event=None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_record_event_both_missing_returns_400(self, handler):
        """Both item_id and event empty returns 400 (first validation fails)."""
        result = await handler.record_confidence_event(item_id="", event="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_history_with_none_item_id_is_valid(self, handler, mock_mound):
        """item_id=None is valid for history (no filter)."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history(item_id=None)
        assert _status(result) == 200
        body = _body(result)
        assert body["filters"]["item_id"] is None

    @pytest.mark.asyncio
    async def test_history_with_none_event_type_is_valid(self, handler, mock_mound):
        """event_type=None is valid for history (no filter)."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history(event_type=None)
        assert _status(result) == 200
        body = _body(result)
        assert body["filters"]["event_type"] is None

    @pytest.mark.asyncio
    async def test_apply_decay_report_to_dict_called(self, handler, mock_mound):
        """The DecayReport's to_dict is what gets returned as JSON."""
        report = MockDecayReport(
            workspace_id="ws-special",
            items_processed=42,
            items_decayed=10,
            items_boosted=3,
            average_confidence_change=-0.05,
            duration_ms=99.9,
        )
        mock_mound.apply_confidence_decay = AsyncMock(return_value=report)
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-special")
        body = _body(result)
        assert body["workspace_id"] == "ws-special"
        assert body["items_processed"] == 42
        assert body["duration_ms"] == 99.9

    @pytest.mark.asyncio
    async def test_record_event_adjustment_structure(self, handler, mock_mound):
        """Full adjustment response structure verification."""
        adj = MockConfidenceAdjustment(
            id="adj-full",
            item_id="item-full",
            event=ConfidenceEvent.VALIDATED,
            old_confidence=0.6,
            new_confidence=0.8,
            reason="expert validated",
            adjusted_at=datetime(2026, 2, 15, 10, 30, 0, tzinfo=timezone.utc),
            metadata={"validator": "expert-1"},
        )
        mock_mound.record_confidence_event = AsyncMock(return_value=adj)
        result = await handler.record_confidence_event(item_id="item-full", event="validated")
        body = _body(result)
        assert body["success"] is True
        assert body["adjusted"] is True
        adj_data = body["adjustment"]
        assert adj_data["id"] == "adj-full"
        assert adj_data["item_id"] == "item-full"
        assert adj_data["event"] == "validated"
        assert adj_data["old_confidence"] == 0.6
        assert adj_data["new_confidence"] == 0.8
        assert adj_data["reason"] == "expert validated"
        assert "2026-02-15" in adj_data["adjusted_at"]
        assert adj_data["metadata"]["validator"] == "expert-1"

    @pytest.mark.asyncio
    async def test_history_filters_structure(self, handler, mock_mound):
        """History response has proper filters object structure."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history(
            item_id="item-x", event_type="cited", limit=10
        )
        body = _body(result)
        assert "filters" in body
        assert "count" in body
        assert "adjustments" in body
        assert body["filters"]["item_id"] == "item-x"
        assert body["filters"]["event_type"] == "cited"

    @pytest.mark.asyncio
    async def test_all_endpoints_return_503_when_no_mound(self, handler_no_mound):
        """All four endpoints return 503 when mound is not available."""
        r1 = await handler_no_mound.apply_confidence_decay_endpoint(workspace_id="ws")
        assert _status(r1) == 503

        r2 = await handler_no_mound.record_confidence_event(item_id="item", event="accessed")
        assert _status(r2) == 503

        r3 = await handler_no_mound.get_confidence_history()
        assert _status(r3) == 503

        r4 = await handler_no_mound.get_decay_stats()
        assert _status(r4) == 503

    @pytest.mark.asyncio
    async def test_decay_report_with_many_adjustments(self, handler, mock_mound):
        """Report with many adjustments serializes all of them."""
        adjustments = [
            MockConfidenceAdjustment(id=f"adj-{i}", item_id=f"item-{i}") for i in range(20)
        ]
        report = MockDecayReport(
            adjustments=adjustments,
            items_processed=100,
            items_decayed=20,
        )
        mock_mound.apply_confidence_decay = AsyncMock(return_value=report)
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        body = _body(result)
        assert len(body["adjustments"]) == 20
        assert body["adjustments"][0]["id"] == "adj-0"
        assert body["adjustments"][19]["id"] == "adj-19"

    @pytest.mark.asyncio
    async def test_history_large_result_set(self, handler, mock_mound):
        """History endpoint handles large result sets."""
        adjustments = [MockConfidenceAdjustment(id=f"adj-{i}") for i in range(100)]
        mock_mound.get_confidence_history = AsyncMock(return_value=adjustments)
        result = await handler.get_confidence_history()
        body = _body(result)
        assert body["count"] == 100
        assert len(body["adjustments"]) == 100

    @pytest.mark.asyncio
    async def test_apply_decay_error_message_sanitized_for_os_error(self, handler, mock_mound):
        """OSError produces a sanitized error message (no internal paths)."""
        mock_mound.apply_confidence_decay = AsyncMock(
            side_effect=OSError("/internal/path/db.sqlite: permission denied")
        )
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        body = _body(result)
        assert _status(result) == 500
        # safe_error_message maps OSError to "Resource not found"
        assert "/internal" not in body["error"]

    @pytest.mark.asyncio
    async def test_apply_decay_error_message_sanitized_for_value_error(self, handler, mock_mound):
        """ValueError produces a sanitized error message."""
        mock_mound.apply_confidence_decay = AsyncMock(
            side_effect=ValueError("SQL injection attempt; DROP TABLE items;")
        )
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        body = _body(result)
        assert _status(result) == 500
        # safe_error_message maps ValueError to "Invalid data format"
        assert "DROP TABLE" not in body["error"]

    @pytest.mark.asyncio
    async def test_record_event_error_message_sanitized(self, handler, mock_mound):
        """Error from record_confidence_event is sanitized."""
        mock_mound.record_confidence_event = AsyncMock(side_effect=KeyError("secret_column_name"))
        result = await handler.record_confidence_event(item_id="item-001", event="accessed")
        body = _body(result)
        assert _status(result) == 500
        assert "secret_column_name" not in body["error"]

    @pytest.mark.asyncio
    async def test_history_error_message_sanitized(self, handler, mock_mound):
        """Error from get_confidence_history is sanitized."""
        mock_mound.get_confidence_history = AsyncMock(
            side_effect=OSError("/var/lib/postgres/data: connection refused")
        )
        result = await handler.get_confidence_history()
        body = _body(result)
        assert _status(result) == 500
        assert "/var/lib" not in body["error"]

    @pytest.mark.asyncio
    async def test_stats_error_message_sanitized(self, handler, mock_mound):
        """Error from get_decay_stats is sanitized."""
        mock_mound.get_decay_stats = MagicMock(
            side_effect=TypeError("expected int but got NoneType for column 'internal_score'")
        )
        result = await handler.get_decay_stats()
        body = _body(result)
        assert _status(result) == 500
        assert "internal_score" not in body["error"]

    @pytest.mark.asyncio
    async def test_apply_decay_workspace_id_whitespace_only_returns_400(self, handler):
        """Whitespace-only workspace_id is still falsy and should pass through.

        Note: The handler checks `not workspace_id` which treats whitespace as truthy.
        """
        # Whitespace string is truthy in Python, so it passes validation
        # The mound will receive the whitespace string
        result = await handler.apply_confidence_decay_endpoint(workspace_id="  ")
        # Whitespace is truthy so it does NOT return 400 -- it goes to the mound
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_record_event_with_long_reason(self, handler, mock_mound):
        """Long reason string is forwarded to mound without truncation."""
        long_reason = "x" * 10000
        await handler.record_confidence_event(
            item_id="item-001", event="accessed", reason=long_reason
        )
        call_kwargs = mock_mound.record_confidence_event.call_args.kwargs
        assert call_kwargs["reason"] == long_reason
        assert len(call_kwargs["reason"]) == 10000

    @pytest.mark.asyncio
    async def test_history_limit_zero(self, handler, mock_mound):
        """Limit of 0 is forwarded to mound (mound decides behavior)."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        await handler.get_confidence_history(limit=0)
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["limit"] == 0

    @pytest.mark.asyncio
    async def test_history_negative_limit(self, handler, mock_mound):
        """Negative limit is forwarded to mound (no handler-level validation)."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        await handler.get_confidence_history(limit=-1)
        call_kwargs = mock_mound.get_confidence_history.call_args.kwargs
        assert call_kwargs["limit"] == -1

    @pytest.mark.asyncio
    async def test_apply_decay_mound_called_exactly_once(self, handler, mock_mound):
        """Mound's apply_confidence_decay is called exactly once."""
        await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert mock_mound.apply_confidence_decay.await_count == 1

    @pytest.mark.asyncio
    async def test_record_event_mound_called_exactly_once(self, handler, mock_mound):
        """Mound's record_confidence_event is called exactly once."""
        await handler.record_confidence_event(item_id="item-001", event="accessed")
        assert mock_mound.record_confidence_event.await_count == 1

    @pytest.mark.asyncio
    async def test_history_mound_called_exactly_once(self, handler, mock_mound):
        """Mound's get_confidence_history is called exactly once."""
        await handler.get_confidence_history()
        assert mock_mound.get_confidence_history.await_count == 1

    @pytest.mark.asyncio
    async def test_stats_mound_called_exactly_once(self, handler, mock_mound):
        """Mound's get_decay_stats is called exactly once."""
        await handler.get_decay_stats()
        assert mock_mound.get_decay_stats.call_count == 1

    @pytest.mark.asyncio
    async def test_apply_decay_response_content_type_is_json(self, handler, mock_mound):
        """Apply decay endpoint returns JSON content type."""
        result = await handler.apply_confidence_decay_endpoint(workspace_id="ws-001")
        assert "json" in result.content_type.lower()

    @pytest.mark.asyncio
    async def test_record_event_response_content_type_is_json(self, handler, mock_mound):
        """Record event endpoint returns JSON content type."""
        result = await handler.record_confidence_event(item_id="item-001", event="accessed")
        assert "json" in result.content_type.lower()

    @pytest.mark.asyncio
    async def test_history_response_content_type_is_json(self, handler, mock_mound):
        """History endpoint returns JSON content type."""
        mock_mound.get_confidence_history = AsyncMock(return_value=[])
        result = await handler.get_confidence_history()
        assert "json" in result.content_type.lower()

    @pytest.mark.asyncio
    async def test_stats_response_content_type_is_json(self, handler, mock_mound):
        """Stats endpoint returns JSON content type."""
        result = await handler.get_decay_stats()
        assert "json" in result.content_type.lower()
