"""Tests for action items handler (aragora/server/handlers/inbox/action_items.py).

Covers all 8 handler functions and the registration helper:
- handle_extract_action_items     POST /api/v1/inbox/actions/extract
- handle_list_pending_actions     GET  /api/v1/inbox/actions/pending
- handle_complete_action          POST /api/v1/inbox/actions/{id}/complete
- handle_update_action_status     POST /api/v1/inbox/actions/{id}/status
- handle_get_due_soon             GET  /api/v1/inbox/actions/due-soon
- handle_batch_extract            POST /api/v1/inbox/actions/batch-extract
- handle_detect_meeting           POST /api/v1/inbox/meetings/detect
- handle_auto_snooze_meeting      POST /api/v1/inbox/meetings/auto-snooze
- get_action_items_handlers       Registration helper

Test categories:
- Handler registration and initialization
- Happy-path success responses
- Validation errors (400): missing required fields
- Not found errors (404): nonexistent action items
- Internal errors (500): extractor/detector failures
- RBAC permission errors (403/500)
- Edge cases: pagination, filtering, sorting, batch limits, deadline parsing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.inbox.action_items as action_items_mod
from aragora.server.handlers.inbox.action_items import (
    _check_inbox_permission,
    get_action_items_handlers,
    handle_auto_snooze_meeting,
    handle_batch_extract,
    handle_complete_action,
    handle_detect_meeting,
    handle_extract_action_items,
    handle_get_due_soon,
    handle_list_pending_actions,
    handle_update_action_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "body"):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    if hasattr(result, "status_code"):
        return result.status_code
    return 200


def _data(result) -> dict:
    """Extract the 'data' envelope from a success response."""
    body = _body(result)
    return body.get("data", body)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockActionItem:
    """Mock action item returned by extractor."""

    id: str = "action-1"
    title: str = "Follow up with client"
    priority: int = 2
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "priority": self.priority,
            "status": self.status,
        }


@dataclass
class MockExtractionResult:
    """Mock result from ActionItemExtractor.extract_action_items."""

    action_items: list[MockActionItem] = field(default_factory=list)
    total_count: int = 0
    high_priority_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_items": [item.to_dict() for item in self.action_items],
            "total_count": self.total_count,
            "high_priority_count": self.high_priority_count,
        }


class MockMeetingType(Enum):
    VIDEO = "video"
    PHONE = "phone"
    IN_PERSON = "in_person"


@dataclass
class MockMeetingLink:
    url: str = "https://zoom.us/j/123"
    provider: str = "zoom"

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "provider": self.provider}


@dataclass
class MockMeetingResult:
    """Mock result from MeetingDetector.detect_meeting."""

    is_meeting: bool = True
    meeting_type: MockMeetingType = MockMeetingType.VIDEO
    title: str = "Team standup"
    start_time: datetime | None = None
    meeting_links: list[MockMeetingLink] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "is_meeting": self.is_meeting,
            "meeting_type": self.meeting_type.value,
            "title": self.title,
        }
        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        result["meeting_links"] = [ml.to_dict() for ml in self.meeting_links]
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_action_items():
    """Clear global action items store before and after each test."""
    action_items_mod._action_items.clear()
    yield
    action_items_mod._action_items.clear()


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset lazy singleton service instances between tests."""
    action_items_mod._action_extractor = None
    action_items_mod._meeting_detector = None
    yield
    action_items_mod._action_extractor = None
    action_items_mod._meeting_detector = None


@pytest.fixture(autouse=True)
def _bypass_rbac():
    """Bypass _check_inbox_permission for all tests by default."""
    with patch.object(action_items_mod, "_check_inbox_permission", return_value=None):
        yield


@pytest.fixture
def mock_extractor():
    """Create a mock ActionItemExtractor."""
    extractor = AsyncMock()
    items = [
        MockActionItem(id="a1", title="Follow up", priority=2),
        MockActionItem(id="a2", title="Review doc", priority=3),
    ]
    result = MockExtractionResult(
        action_items=items,
        total_count=2,
        high_priority_count=1,
    )
    extractor.extract_action_items = AsyncMock(return_value=result)
    return extractor


@pytest.fixture
def mock_detector():
    """Create a mock MeetingDetector."""
    detector = AsyncMock()
    future_time = datetime.now(timezone.utc) + timedelta(hours=2)
    result = MockMeetingResult(
        is_meeting=True,
        start_time=future_time,
        meeting_links=[MockMeetingLink()],
    )
    detector.detect_meeting = AsyncMock(return_value=result)
    return detector


def _seed_items(items: list[dict[str, Any]]) -> None:
    """Insert test action items directly into the global store."""
    for item in items:
        action_items_mod._action_items[item["id"]] = item


# ===========================================================================
# get_action_items_handlers
# ===========================================================================


class TestGetActionItemsHandlers:
    """Tests for the handler registration function."""

    def test_returns_dict(self):
        handlers = get_action_items_handlers()
        assert isinstance(handlers, dict)

    def test_contains_all_expected_keys(self):
        handlers = get_action_items_handlers()
        expected = {
            "extract_action_items",
            "list_pending_actions",
            "complete_action",
            "update_action_status",
            "get_due_soon",
            "batch_extract",
            "detect_meeting",
            "auto_snooze_meeting",
        }
        assert set(handlers.keys()) == expected

    def test_all_values_are_callables(self):
        handlers = get_action_items_handlers()
        for key, value in handlers.items():
            assert callable(value), f"{key} is not callable"


# ===========================================================================
# handle_extract_action_items
# ===========================================================================


class TestExtractActionItems:
    """Tests for POST /api/v1/inbox/actions/extract."""

    @pytest.mark.asyncio
    async def test_extract_success(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_extract_action_items(
                data={
                    "email_id": "e1",
                    "subject": "Meeting notes",
                    "body": "Please review the doc by Friday.",
                    "sender": "alice@example.com",
                },
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["total_count"] == 2
        assert body["high_priority_count"] == 1

    @pytest.mark.asyncio
    async def test_extract_stores_items(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            await handle_extract_action_items(
                data={"subject": "Test", "body": "Do things"},
                user_id="user-1",
            )
        assert "a1" in action_items_mod._action_items
        assert "a2" in action_items_mod._action_items

    @pytest.mark.asyncio
    async def test_extract_missing_subject_and_body(self):
        result = await handle_extract_action_items(
            data={"email_id": "e1", "sender": "bob@example.com"},
            user_id="user-1",
        )
        assert _status(result) == 400
        assert "subject or body" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_extract_empty_subject_and_body(self):
        result = await handle_extract_action_items(
            data={"subject": "", "body": ""},
            user_id="user-1",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_extract_with_only_subject(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_extract_action_items(
                data={"subject": "Action required: review"},
                user_id="user-1",
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_extract_with_only_body(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_extract_action_items(
                data={"body": "Please send the report."},
                user_id="user-1",
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_extract_with_optional_params(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_extract_action_items(
                data={
                    "subject": "Task",
                    "body": "Do it",
                    "to_addresses": ["bob@example.com"],
                    "extract_deadlines": False,
                    "detect_assignees": False,
                },
                user_id="user-1",
            )
        assert _status(result) == 200
        # Verify the extractor was called with the right flags
        mock_extractor.extract_action_items.assert_called_once()
        _, kwargs = mock_extractor.extract_action_items.call_args
        assert kwargs["extract_deadlines"] is False
        assert kwargs["detect_assignees"] is False

    @pytest.mark.asyncio
    async def test_extract_extractor_raises(self):
        bad_extractor = AsyncMock()
        bad_extractor.extract_action_items = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(action_items_mod, "get_action_extractor", return_value=bad_extractor):
            result = await handle_extract_action_items(
                data={"subject": "Test", "body": "Body"},
                user_id="user-1",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_extract_generates_email_id_when_missing(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_extract_action_items(
                data={"subject": "Test", "body": "Body"},
                user_id="user-1",
            )
        assert _status(result) == 200
        call_args = mock_extractor.extract_action_items.call_args
        email_obj = call_args[0][0]
        assert email_obj.id.startswith("inline_")


# ===========================================================================
# handle_list_pending_actions
# ===========================================================================


class TestListPendingActions:
    """Tests for GET /api/v1/inbox/actions/pending."""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        result = await handle_list_pending_actions(data={}, user_id="user-1")
        assert _status(result) == 200
        body = _data(result)
        assert body["action_items"] == []
        assert body["total"] == 0
        assert body["has_more"] is False

    @pytest.mark.asyncio
    async def test_list_pending_items(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "priority": 2},
                {"id": "a2", "status": "completed", "priority": 1},
                {"id": "a3", "status": "in_progress", "priority": 3},
            ]
        )
        result = await handle_list_pending_actions(data={}, user_id="user-1")
        body = _data(result)
        assert body["total"] == 2
        ids = [item["id"] for item in body["action_items"]]
        assert "a1" in ids
        assert "a3" in ids
        assert "a2" not in ids

    @pytest.mark.asyncio
    async def test_list_filter_by_assignee(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "assignee_email": "alice@co.com"},
                {"id": "a2", "status": "pending", "assignee_email": "bob@co.com"},
            ]
        )
        result = await handle_list_pending_actions(
            data={"assignee": "alice@co.com"}, user_id="user-1"
        )
        body = _data(result)
        assert body["total"] == 1
        assert body["action_items"][0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_list_assignee_filter_case_insensitive(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "assignee_email": "Alice@Co.com"},
            ]
        )
        result = await handle_list_pending_actions(
            data={"assignee": "alice@co.com"}, user_id="user-1"
        )
        body = _data(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_filter_by_priority(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "priority": 1},
                {"id": "a2", "status": "pending", "priority": 2},
                {"id": "a3", "status": "pending", "priority": 3},
            ]
        )
        result = await handle_list_pending_actions(data={"priority": "critical"}, user_id="user-1")
        body = _data(result)
        assert body["total"] == 1
        assert body["action_items"][0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_list_filter_by_priority_high(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "priority": 2},
                {"id": "a2", "status": "pending", "priority": 3},
            ]
        )
        result = await handle_list_pending_actions(data={"priority": "high"}, user_id="user-1")
        body = _data(result)
        assert body["total"] == 1
        assert body["action_items"][0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_list_filter_by_invalid_priority(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "priority": 2},
            ]
        )
        result = await handle_list_pending_actions(data={"priority": "unknown"}, user_id="user-1")
        body = _data(result)
        # Unknown priority maps to None so the filter branch is skipped;
        # all items are returned unfiltered.
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_filter_by_due_within_hours(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "pending",
                    "deadline": (now + timedelta(hours=2)).isoformat(),
                },
                {
                    "id": "a2",
                    "status": "pending",
                    "deadline": (now + timedelta(hours=48)).isoformat(),
                },
                {"id": "a3", "status": "pending"},  # No deadline
            ]
        )
        result = await handle_list_pending_actions(data={"due_within_hours": 24}, user_id="user-1")
        body = _data(result)
        assert body["total"] == 1
        assert body["action_items"][0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_list_pagination_limit(self):
        items = [{"id": f"a{i}", "status": "pending", "priority": 3} for i in range(10)]
        _seed_items(items)
        result = await handle_list_pending_actions(data={"limit": 3}, user_id="user-1")
        body = _data(result)
        assert len(body["action_items"]) == 3
        assert body["total"] == 10
        assert body["has_more"] is True

    @pytest.mark.asyncio
    async def test_list_pagination_offset(self):
        items = [{"id": f"a{i}", "status": "pending", "priority": 3} for i in range(5)]
        _seed_items(items)
        result = await handle_list_pending_actions(data={"limit": 2, "offset": 3}, user_id="user-1")
        body = _data(result)
        assert len(body["action_items"]) == 2
        assert body["offset"] == 3

    @pytest.mark.asyncio
    async def test_list_limit_clamped_min(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_list_pending_actions(data={"limit": 0}, user_id="user-1")
        body = _data(result)
        assert body["limit"] >= 1

    @pytest.mark.asyncio
    async def test_list_limit_clamped_max(self):
        result = await handle_list_pending_actions(data={"limit": 999}, user_id="user-1")
        body = _data(result)
        assert body["limit"] <= 200

    @pytest.mark.asyncio
    async def test_list_offset_clamped_min(self):
        result = await handle_list_pending_actions(data={"offset": -5}, user_id="user-1")
        body = _data(result)
        assert body["offset"] >= 0

    @pytest.mark.asyncio
    async def test_list_sort_by_deadline_then_priority(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {"id": "no-deadline", "status": "pending", "priority": 1},
                {
                    "id": "later",
                    "status": "pending",
                    "priority": 2,
                    "deadline": (now + timedelta(hours=10)).isoformat(),
                },
                {
                    "id": "sooner",
                    "status": "pending",
                    "priority": 3,
                    "deadline": (now + timedelta(hours=2)).isoformat(),
                },
            ]
        )
        result = await handle_list_pending_actions(data={}, user_id="user-1")
        body = _data(result)
        ids = [item["id"] for item in body["action_items"]]
        # Items with deadlines come first (sorted by deadline), then those without
        assert ids == ["sooner", "later", "no-deadline"]

    @pytest.mark.asyncio
    async def test_list_invalid_deadline_string(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "deadline": "not-a-date"},
            ]
        )
        result = await handle_list_pending_actions(data={"due_within_hours": 24}, user_id="user-1")
        body = _data(result)
        # Item with invalid deadline is excluded from due_within filter
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_has_more_false_when_all_fit(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_list_pending_actions(data={}, user_id="user-1")
        body = _data(result)
        assert body["has_more"] is False


# ===========================================================================
# handle_complete_action
# ===========================================================================


class TestCompleteAction:
    """Tests for POST /api/v1/inbox/actions/{id}/complete."""

    @pytest.mark.asyncio
    async def test_complete_success(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_complete_action(data={}, action_id="a1", user_id="user-1")
        assert _status(result) == 200
        body = _data(result)
        assert body["action_id"] == "a1"
        assert body["status"] == "completed"
        assert "completed_at" in body

    @pytest.mark.asyncio
    async def test_complete_sets_completed_by(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_complete_action(
            data={"completed_by": "alice"}, action_id="a1", user_id="user-1"
        )
        assert action_items_mod._action_items["a1"]["completed_by"] == "alice"

    @pytest.mark.asyncio
    async def test_complete_default_completed_by_is_user_id(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_complete_action(data={}, action_id="a1", user_id="user-1")
        assert action_items_mod._action_items["a1"]["completed_by"] == "user-1"

    @pytest.mark.asyncio
    async def test_complete_with_notes(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_complete_action(data={"notes": "All done!"}, action_id="a1", user_id="user-1")
        assert action_items_mod._action_items["a1"]["completion_notes"] == "All done!"

    @pytest.mark.asyncio
    async def test_complete_without_notes_no_completion_notes_key(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_complete_action(data={}, action_id="a1", user_id="user-1")
        assert "completion_notes" not in action_items_mod._action_items["a1"]

    @pytest.mark.asyncio
    async def test_complete_not_found(self):
        result = await handle_complete_action(data={}, action_id="nonexistent", user_id="user-1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_complete_missing_action_id(self):
        result = await handle_complete_action(data={}, action_id="", user_id="user-1")
        assert _status(result) == 400
        assert "action_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_complete_action_id_from_body(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_complete_action(
            data={"action_id": "a1"}, action_id="", user_id="user-1"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_complete_action_id_param_takes_precedence(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending"},
                {"id": "a2", "status": "pending"},
            ]
        )
        result = await handle_complete_action(
            data={"action_id": "a2"}, action_id="a1", user_id="user-1"
        )
        assert _status(result) == 200
        # The param action_id="a1" should be used, not "a2" from body
        body = _data(result)
        assert body["action_id"] == "a1"


# ===========================================================================
# handle_update_action_status
# ===========================================================================


class TestUpdateActionStatus:
    """Tests for POST /api/v1/inbox/actions/{id}/status."""

    @pytest.mark.asyncio
    async def test_update_status_success(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_update_action_status(
            data={"status": "in_progress"}, action_id="a1", user_id="user-1"
        )
        assert _status(result) == 200
        body = _data(result)
        assert body["status"] == "in_progress"
        assert body["previous_status"] == "pending"

    @pytest.mark.asyncio
    async def test_update_status_completed_sets_completed_at(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_update_action_status(
            data={"status": "completed"}, action_id="a1", user_id="user-1"
        )
        item = action_items_mod._action_items["a1"]
        assert item["status"] == "completed"
        assert "completed_at" in item

    @pytest.mark.asyncio
    async def test_update_status_with_notes_creates_history(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_update_action_status(
            data={"status": "in_progress", "notes": "Starting work"},
            action_id="a1",
            user_id="user-1",
        )
        item = action_items_mod._action_items["a1"]
        assert "status_history" in item
        assert len(item["status_history"]) == 1
        entry = item["status_history"][0]
        assert entry["from"] == "pending"
        assert entry["to"] == "in_progress"
        assert entry["notes"] == "Starting work"
        assert entry["by"] == "user-1"

    @pytest.mark.asyncio
    async def test_update_status_multiple_transitions(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_update_action_status(
            data={"status": "in_progress", "notes": "First"},
            action_id="a1",
            user_id="user-1",
        )
        await handle_update_action_status(
            data={"status": "completed", "notes": "Done"},
            action_id="a1",
            user_id="user-1",
        )
        item = action_items_mod._action_items["a1"]
        assert len(item["status_history"]) == 2

    @pytest.mark.asyncio
    async def test_update_status_invalid(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_update_action_status(
            data={"status": "unknown_status"}, action_id="a1", user_id="user-1"
        )
        assert _status(result) == 400
        assert "invalid status" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_update_status_missing(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_update_action_status(data={}, action_id="a1", user_id="user-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_status_not_found(self):
        result = await handle_update_action_status(
            data={"status": "completed"}, action_id="nope", user_id="user-1"
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_status_missing_action_id(self):
        result = await handle_update_action_status(
            data={"status": "completed"}, action_id="", user_id="user-1"
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_status_action_id_from_body(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_update_action_status(
            data={"action_id": "a1", "status": "deferred"}, action_id="", user_id="user-1"
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_update_all_valid_statuses(self):
        for status in ("pending", "in_progress", "completed", "cancelled", "deferred"):
            _seed_items([{"id": "a1", "status": "pending"}])
            result = await handle_update_action_status(
                data={"status": status}, action_id="a1", user_id="user-1"
            )
            assert _status(result) == 200, f"Failed for status={status}"
            action_items_mod._action_items.clear()

    @pytest.mark.asyncio
    async def test_update_status_sets_updated_by(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_update_action_status(
            data={"status": "deferred"}, action_id="a1", user_id="bob"
        )
        item = action_items_mod._action_items["a1"]
        assert item["status_updated_by"] == "bob"

    @pytest.mark.asyncio
    async def test_update_without_notes_no_history(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        await handle_update_action_status(
            data={"status": "in_progress"}, action_id="a1", user_id="user-1"
        )
        item = action_items_mod._action_items["a1"]
        assert "status_history" not in item


# ===========================================================================
# handle_get_due_soon
# ===========================================================================


class TestGetDueSoon:
    """Tests for GET /api/v1/inbox/actions/due-soon."""

    @pytest.mark.asyncio
    async def test_due_soon_empty(self):
        result = await handle_get_due_soon(data={}, user_id="user-1")
        assert _status(result) == 200
        body = _data(result)
        assert body["due_soon"] == []
        assert body["overdue"] == []
        assert body["total_urgent"] == 0

    @pytest.mark.asyncio
    async def test_due_soon_items(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "pending",
                    "deadline": (now + timedelta(hours=5)).isoformat(),
                },
                {
                    "id": "a2",
                    "status": "pending",
                    "deadline": (now + timedelta(hours=48)).isoformat(),
                },
            ]
        )
        result = await handle_get_due_soon(data={"hours": 24}, user_id="user-1")
        body = _data(result)
        assert body["due_soon_count"] == 1
        assert body["due_soon"][0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_due_soon_overdue_items(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "pending",
                    "deadline": (now - timedelta(hours=2)).isoformat(),
                },
            ]
        )
        result = await handle_get_due_soon(data={}, user_id="user-1")
        body = _data(result)
        assert body["overdue_count"] == 1
        assert body["overdue"][0]["is_overdue"] is True
        assert "overdue_hours" in body["overdue"][0]

    @pytest.mark.asyncio
    async def test_due_soon_exclude_overdue(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "pending",
                    "deadline": (now - timedelta(hours=2)).isoformat(),
                },
            ]
        )
        result = await handle_get_due_soon(data={"include_overdue": False}, user_id="user-1")
        body = _data(result)
        assert body["overdue_count"] == 0

    @pytest.mark.asyncio
    async def test_due_soon_include_overdue_string_true(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "pending",
                    "deadline": (now - timedelta(hours=1)).isoformat(),
                },
            ]
        )
        result = await handle_get_due_soon(data={"include_overdue": "true"}, user_id="user-1")
        body = _data(result)
        assert body["overdue_count"] == 1

    @pytest.mark.asyncio
    async def test_due_soon_include_overdue_string_false(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "pending",
                    "deadline": (now - timedelta(hours=1)).isoformat(),
                },
            ]
        )
        result = await handle_get_due_soon(data={"include_overdue": "false"}, user_id="user-1")
        body = _data(result)
        assert body["overdue_count"] == 0

    @pytest.mark.asyncio
    async def test_due_soon_skips_completed_items(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "completed",
                    "deadline": (now + timedelta(hours=2)).isoformat(),
                },
            ]
        )
        result = await handle_get_due_soon(data={}, user_id="user-1")
        body = _data(result)
        assert body["total_urgent"] == 0

    @pytest.mark.asyncio
    async def test_due_soon_skips_items_without_deadline(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending"},
            ]
        )
        result = await handle_get_due_soon(data={}, user_id="user-1")
        body = _data(result)
        assert body["total_urgent"] == 0

    @pytest.mark.asyncio
    async def test_due_soon_hours_clamped_min(self):
        result = await handle_get_due_soon(data={"hours": 0}, user_id="user-1")
        body = _data(result)
        assert body["hours_window"] >= 1

    @pytest.mark.asyncio
    async def test_due_soon_hours_clamped_max(self):
        result = await handle_get_due_soon(data={"hours": 99999}, user_id="user-1")
        body = _data(result)
        assert body["hours_window"] <= 8760

    @pytest.mark.asyncio
    async def test_due_soon_default_hours_is_24(self):
        result = await handle_get_due_soon(data={}, user_id="user-1")
        body = _data(result)
        assert body["hours_window"] == 24

    @pytest.mark.asyncio
    async def test_due_soon_invalid_deadline_string_skipped(self):
        _seed_items(
            [
                {"id": "a1", "status": "pending", "deadline": "not-a-date"},
            ]
        )
        result = await handle_get_due_soon(data={}, user_id="user-1")
        body = _data(result)
        assert body["total_urgent"] == 0

    @pytest.mark.asyncio
    async def test_due_soon_hours_remaining_calculated(self):
        now = datetime.now(timezone.utc)
        _seed_items(
            [
                {
                    "id": "a1",
                    "status": "pending",
                    "deadline": (now + timedelta(hours=5)).isoformat(),
                },
            ]
        )
        result = await handle_get_due_soon(data={}, user_id="user-1")
        body = _data(result)
        assert body["due_soon"][0]["hours_remaining"] >= 4


# ===========================================================================
# handle_batch_extract
# ===========================================================================


class TestBatchExtract:
    """Tests for POST /api/v1/inbox/actions/batch-extract."""

    @pytest.mark.asyncio
    async def test_batch_success(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_batch_extract(
                data={
                    "emails": [
                        {"email_id": "e1", "subject": "Task 1", "body": "Do A"},
                        {"email_id": "e2", "subject": "Task 2", "body": "Do B"},
                    ]
                },
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["total_emails"] == 2
        assert body["successful_extractions"] == 2
        assert body["total_action_items"] == 4  # 2 items per email

    @pytest.mark.asyncio
    async def test_batch_empty_emails_list(self):
        result = await handle_batch_extract(data={"emails": []}, user_id="user-1")
        assert _status(result) == 400
        assert "emails" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_batch_missing_emails_key(self):
        result = await handle_batch_extract(data={}, user_id="user-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_batch_exceeds_limit(self):
        emails = [{"email_id": f"e{i}", "subject": "S", "body": "B"} for i in range(51)]
        result = await handle_batch_extract(data={"emails": emails}, user_id="user-1")
        assert _status(result) == 400
        assert "50" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_batch_exactly_50_allowed(self, mock_extractor):
        emails = [{"email_id": f"e{i}", "subject": "S", "body": "B"} for i in range(50)]
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_batch_extract(data={"emails": emails}, user_id="user-1")
        assert _status(result) == 200
        body = _data(result)
        assert body["total_emails"] == 50

    @pytest.mark.asyncio
    async def test_batch_partial_failure(self):
        call_count = 0

        async def mock_extract(email, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Extraction failed")
            return MockExtractionResult(
                action_items=[MockActionItem(id=f"a{call_count}")],
                total_count=1,
                high_priority_count=0,
            )

        extractor = MagicMock()
        extractor.extract_action_items = mock_extract

        with patch.object(action_items_mod, "get_action_extractor", return_value=extractor):
            result = await handle_batch_extract(
                data={
                    "emails": [
                        {"email_id": "e1", "subject": "S1", "body": "B1"},
                        {"email_id": "e2", "subject": "S2", "body": "B2"},
                        {"email_id": "e3", "subject": "S3", "body": "B3"},
                    ]
                },
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["successful_extractions"] == 2
        assert body["total_emails"] == 3
        # Check individual results
        results = body["results"]
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True

    @pytest.mark.asyncio
    async def test_batch_stores_items(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            await handle_batch_extract(
                data={
                    "emails": [
                        {"email_id": "e1", "subject": "S", "body": "B"},
                    ]
                },
                user_id="user-1",
            )
        assert len(action_items_mod._action_items) == 2  # mock returns 2 items

    @pytest.mark.asyncio
    async def test_batch_high_priority_count_aggregated(self, mock_extractor):
        with patch.object(action_items_mod, "get_action_extractor", return_value=mock_extractor):
            result = await handle_batch_extract(
                data={
                    "emails": [
                        {"email_id": "e1", "subject": "S1", "body": "B1"},
                        {"email_id": "e2", "subject": "S2", "body": "B2"},
                    ]
                },
                user_id="user-1",
            )
        body = _data(result)
        assert body["total_high_priority"] == 2  # 1 per email * 2 emails

    @pytest.mark.asyncio
    async def test_batch_extractor_global_failure(self):
        with patch.object(
            action_items_mod,
            "get_action_extractor",
            side_effect=RuntimeError("init failed"),
        ):
            result = await handle_batch_extract(
                data={"emails": [{"email_id": "e1", "subject": "S", "body": "B"}]},
                user_id="user-1",
            )
        assert _status(result) == 500


# ===========================================================================
# handle_detect_meeting
# ===========================================================================


class TestDetectMeeting:
    """Tests for POST /api/v1/inbox/meetings/detect."""

    @pytest.mark.asyncio
    async def test_detect_success(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            result = await handle_detect_meeting(
                data={
                    "email_id": "e1",
                    "subject": "Team standup",
                    "body": "Join at 10 AM",
                    "sender": "alice@co.com",
                },
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["is_meeting"] is True

    @pytest.mark.asyncio
    async def test_detect_missing_subject_and_body(self):
        result = await handle_detect_meeting(
            data={"email_id": "e1", "sender": "bob@co.com"},
            user_id="user-1",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_detect_with_check_calendar(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            await handle_detect_meeting(
                data={
                    "subject": "Meeting",
                    "body": "Let's meet",
                    "check_calendar": True,
                },
                user_id="user-1",
            )
        mock_detector.detect_meeting.assert_called_once()
        _, kwargs = mock_detector.detect_meeting.call_args
        assert kwargs["check_calendar"] is True

    @pytest.mark.asyncio
    async def test_detect_detector_raises(self):
        bad_detector = AsyncMock()
        bad_detector.detect_meeting = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(action_items_mod, "get_meeting_detector", return_value=bad_detector):
            result = await handle_detect_meeting(
                data={"subject": "Meeting", "body": "Content"},
                user_id="user-1",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_detect_generates_email_id_when_missing(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            await handle_detect_meeting(
                data={"subject": "Meet", "body": "Content"},
                user_id="user-1",
            )
        call_args = mock_detector.detect_meeting.call_args
        email_obj = call_args[0][0]
        assert email_obj.id.startswith("meeting_")


# ===========================================================================
# handle_auto_snooze_meeting
# ===========================================================================


class TestAutoSnoozeMeeting:
    """Tests for POST /api/v1/inbox/meetings/auto-snooze."""

    @pytest.mark.asyncio
    async def test_snooze_success(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            result = await handle_auto_snooze_meeting(
                data={
                    "email_id": "e1",
                    "subject": "Team standup",
                    "body": "Join at 10 AM",
                    "sender": "alice@co.com",
                },
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["is_meeting"] is True
        assert body["snooze_scheduled"] is True
        assert "snooze_until" in body

    @pytest.mark.asyncio
    async def test_snooze_not_a_meeting(self):
        detector = AsyncMock()
        detector.detect_meeting = AsyncMock(return_value=MockMeetingResult(is_meeting=False))
        with patch.object(action_items_mod, "get_meeting_detector", return_value=detector):
            result = await handle_auto_snooze_meeting(
                data={"subject": "FYI", "body": "Newsletter content"},
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["is_meeting"] is False
        assert body["snooze_scheduled"] is False

    @pytest.mark.asyncio
    async def test_snooze_no_start_time(self):
        detector = AsyncMock()
        detector.detect_meeting = AsyncMock(
            return_value=MockMeetingResult(is_meeting=True, start_time=None)
        )
        with patch.object(action_items_mod, "get_meeting_detector", return_value=detector):
            result = await handle_auto_snooze_meeting(
                data={"subject": "Meeting", "body": "TBD time"},
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["is_meeting"] is True
        assert body["snooze_scheduled"] is False
        assert "could not determine" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_snooze_meeting_too_soon(self):
        detector = AsyncMock()
        # Meeting starts 10 minutes from now, snooze would be in the past
        soon = datetime.now(timezone.utc) + timedelta(minutes=10)
        detector.detect_meeting = AsyncMock(
            return_value=MockMeetingResult(is_meeting=True, start_time=soon)
        )
        with patch.object(action_items_mod, "get_meeting_detector", return_value=detector):
            result = await handle_auto_snooze_meeting(
                data={
                    "subject": "Urgent meeting",
                    "body": "Right now!",
                    "minutes_before": 30,
                },
                user_id="user-1",
            )
        assert _status(result) == 200
        body = _data(result)
        assert body["snooze_scheduled"] is False
        assert "too soon" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_snooze_custom_minutes_before(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            result = await handle_auto_snooze_meeting(
                data={
                    "subject": "Meeting",
                    "body": "Content",
                    "minutes_before": 60,
                },
                user_id="user-1",
            )
        body = _data(result)
        assert body["minutes_before"] == 60

    @pytest.mark.asyncio
    async def test_snooze_invalid_minutes_before_defaults_to_30(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            result = await handle_auto_snooze_meeting(
                data={
                    "subject": "Meeting",
                    "body": "Content",
                    "minutes_before": "invalid",
                },
                user_id="user-1",
            )
        body = _data(result)
        assert body["minutes_before"] == 30

    @pytest.mark.asyncio
    async def test_snooze_missing_subject_and_body(self):
        result = await handle_auto_snooze_meeting(data={"email_id": "e1"}, user_id="user-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_snooze_detector_raises(self):
        bad_detector = AsyncMock()
        bad_detector.detect_meeting = AsyncMock(side_effect=ValueError("fail"))
        with patch.object(action_items_mod, "get_meeting_detector", return_value=bad_detector):
            result = await handle_auto_snooze_meeting(
                data={"subject": "Meeting", "body": "Content"},
                user_id="user-1",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_snooze_returns_meeting_links(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            result = await handle_auto_snooze_meeting(
                data={"subject": "Meeting", "body": "Zoom link inside"},
                user_id="user-1",
            )
        body = _data(result)
        assert len(body["meeting_links"]) == 1
        assert body["meeting_links"][0]["url"] == "https://zoom.us/j/123"

    @pytest.mark.asyncio
    async def test_snooze_generates_email_id_when_missing(self, mock_detector):
        with patch.object(action_items_mod, "get_meeting_detector", return_value=mock_detector):
            result = await handle_auto_snooze_meeting(
                data={"subject": "Meet", "body": "Content"},
                user_id="user-1",
            )
        assert _status(result) == 200


# ===========================================================================
# _check_inbox_permission (RBAC)
# ===========================================================================


class TestCheckInboxPermission:
    """Tests for the _check_inbox_permission helper.

    These tests bypass the autouse _bypass_rbac fixture by directly calling
    the real function.
    """

    def test_permission_allowed(self):
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = MagicMock(allowed=True)
        with patch(
            "aragora.server.handlers.inbox.action_items.get_permission_checker",
            return_value=mock_checker,
        ):
            result = _check_inbox_permission("user-1", "read")
        assert result is None  # None means allowed

    def test_permission_denied(self):
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = MagicMock(allowed=False, reason="No access")
        with patch(
            "aragora.server.handlers.inbox.action_items.get_permission_checker",
            return_value=mock_checker,
        ):
            result = _check_inbox_permission("user-1", "update")
        assert _status(result) == 403

    def test_permission_check_error(self):
        with patch(
            "aragora.server.handlers.inbox.action_items.get_permission_checker",
            side_effect=RuntimeError("Checker init failed"),
        ):
            result = _check_inbox_permission("user-1", "read")
        assert _status(result) == 500

    def test_permission_uses_correct_action(self):
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = MagicMock(allowed=True)
        with patch(
            "aragora.server.handlers.inbox.action_items.get_permission_checker",
            return_value=mock_checker,
        ):
            _check_inbox_permission("user-1", "update")
        ctx = mock_checker.check_permission.call_args[0][0]
        permission = mock_checker.check_permission.call_args[0][1]
        assert permission == "inbox.update"

    def test_permission_default_roles(self):
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = MagicMock(allowed=True)
        with patch(
            "aragora.server.handlers.inbox.action_items.get_permission_checker",
            return_value=mock_checker,
        ):
            _check_inbox_permission("user-1", "read")
        ctx = mock_checker.check_permission.call_args[0][0]
        assert "member" in ctx.roles

    def test_permission_custom_roles(self):
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = MagicMock(allowed=True)
        with patch(
            "aragora.server.handlers.inbox.action_items.get_permission_checker",
            return_value=mock_checker,
        ):
            _check_inbox_permission("user-1", "read", roles={"admin"})
        ctx = mock_checker.check_permission.call_args[0][0]
        assert "admin" in ctx.roles

    def test_permission_with_org_id(self):
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = MagicMock(allowed=True)
        with patch(
            "aragora.server.handlers.inbox.action_items.get_permission_checker",
            return_value=mock_checker,
        ):
            _check_inbox_permission("user-1", "read", org_id="org-42")
        ctx = mock_checker.check_permission.call_args[0][0]
        assert ctx.org_id == "org-42"


# ===========================================================================
# RBAC integration (denied paths)
# ===========================================================================


class TestRBACDeniedPaths:
    """Tests verifying RBAC denial flows through handlers.

    These override the _bypass_rbac autouse fixture.
    """

    @pytest.fixture(autouse=True)
    def _enable_rbac(self):
        """Re-enable RBAC checking for tests in this class."""
        # The class-level autouse fixture runs AFTER the module-level _bypass_rbac
        # So we need to un-patch _check_inbox_permission
        # We do this by patching it to return a 403 error
        with patch.object(
            action_items_mod,
            "_check_inbox_permission",
            return_value=MagicMock(
                status_code=403,
                body=json.dumps({"error": "Permission denied"}).encode(),
                content_type="application/json",
            ),
        ):
            yield

    @pytest.mark.asyncio
    async def test_extract_denied(self):
        result = await handle_extract_action_items(
            data={"subject": "Test", "body": "Content"},
            user_id="user-1",
        )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_list_pending_denied(self):
        result = await handle_list_pending_actions(data={}, user_id="user-1")
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_complete_denied(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_complete_action(data={}, action_id="a1", user_id="user-1")
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_update_status_denied(self):
        _seed_items([{"id": "a1", "status": "pending"}])
        result = await handle_update_action_status(
            data={"status": "completed"}, action_id="a1", user_id="user-1"
        )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_due_soon_denied(self):
        result = await handle_get_due_soon(data={}, user_id="user-1")
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_batch_extract_denied(self):
        result = await handle_batch_extract(
            data={"emails": [{"subject": "S", "body": "B"}]},
            user_id="user-1",
        )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_detect_meeting_denied(self):
        result = await handle_detect_meeting(
            data={"subject": "Meeting", "body": "Content"},
            user_id="user-1",
        )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_auto_snooze_denied(self):
        result = await handle_auto_snooze_meeting(
            data={"subject": "Meeting", "body": "Content"},
            user_id="user-1",
        )
        assert _status(result) == 403


# ===========================================================================
# Singleton getters
# ===========================================================================


class TestSingletonGetters:
    """Tests for lazy singleton service creation."""

    def test_get_action_extractor_creates_instance(self):
        mock_cls = MagicMock()
        with patch(
            "aragora.server.handlers.inbox.action_items.ActionItemExtractor",
            mock_cls,
            create=True,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.services.action_item_extractor": MagicMock(ActionItemExtractor=mock_cls)},
            ):
                extractor = action_items_mod.get_action_extractor()
                assert extractor is not None

    def test_get_action_extractor_returns_cached(self):
        sentinel = object()
        action_items_mod._action_extractor = sentinel
        assert action_items_mod.get_action_extractor() is sentinel

    def test_get_meeting_detector_returns_cached(self):
        sentinel = object()
        action_items_mod._meeting_detector = sentinel
        assert action_items_mod.get_meeting_detector() is sentinel
