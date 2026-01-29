"""
Tests for inbox/action_items.py - Action Items HTTP API handlers.

Tests cover:
- POST /api/v1/inbox/actions/extract - Extract action items from email
- GET /api/v1/inbox/actions/pending - List pending action items
- POST /api/v1/inbox/actions/{id}/complete - Mark action item complete
- POST /api/v1/inbox/actions/{id}/status - Update action item status
- GET /api/v1/inbox/actions/due-soon - Get items due within timeframe
- POST /api/v1/inbox/actions/batch-extract - Batch extract from emails
- POST /api/v1/inbox/meetings/detect - Detect meeting information
- POST /api/v1/inbox/meetings/auto-snooze - Auto-snooze meeting emails
- RBAC permission enforcement
- Error handling and validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.inbox.action_items import (
    handle_extract_action_items,
    handle_list_pending_actions,
    handle_complete_action,
    handle_update_action_status,
    handle_get_due_soon,
    handle_batch_extract,
    handle_detect_meeting,
    handle_auto_snooze_meeting,
    get_action_items_handlers,
    _action_items,
    _action_items_lock,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockMeetingType(Enum):
    """Mock meeting type enum."""

    video_call = "video_call"
    phone_call = "phone_call"
    in_person = "in_person"


@dataclass
class MockActionItem:
    """Mock action item for testing."""

    id: str = "action-123"
    description: str = "Review quarterly report"
    priority: int = 2  # 1=critical, 2=high, 3=medium, 4=low
    status: str = "pending"
    deadline: str | None = None
    assignee_email: str | None = "user@example.com"
    email_id: str = "email-456"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "deadline": self.deadline,
            "assignee_email": self.assignee_email,
            "email_id": self.email_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockExtractionResult:
    """Mock action item extraction result."""

    action_items: list[MockActionItem] = field(default_factory=list)
    total_count: int = 0
    high_priority_count: int = 0

    def __post_init__(self):
        if not self.action_items:
            self.action_items = [MockActionItem()]
            self.total_count = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_items": [item.to_dict() for item in self.action_items],
            "total_count": self.total_count,
            "high_priority_count": self.high_priority_count,
        }


@dataclass
class MockMeetingLink:
    """Mock meeting link for testing."""

    url: str = "https://zoom.us/j/123456"
    platform: str = "zoom"

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "platform": self.platform}


@dataclass
class MockMeetingResult:
    """Mock meeting detection result."""

    is_meeting: bool = True
    meeting_type: MockMeetingType = MockMeetingType.video_call
    title: str | None = "Weekly Sync"
    start_time: datetime | None = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=2)
    )
    duration_minutes: int | None = 60
    meeting_links: list[MockMeetingLink] = field(default_factory=list)
    confidence: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_meeting": self.is_meeting,
            "meeting_type": self.meeting_type.value,
            "title": self.title,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_minutes": self.duration_minutes,
            "meeting_links": [link.to_dict() for link in self.meeting_links],
            "confidence": self.confidence,
        }


class MockActionItemExtractor:
    """Mock action item extractor for testing."""

    async def extract_action_items(
        self,
        email: Any,
        extract_deadlines: bool = True,
        detect_assignees: bool = True,
    ) -> MockExtractionResult:
        return MockExtractionResult()


class MockMeetingDetector:
    """Mock meeting detector for testing."""

    async def detect_meeting(
        self,
        email: Any,
        check_calendar: bool = False,
    ) -> MockMeetingResult:
        return MockMeetingResult()


@pytest.fixture(autouse=True)
def clear_action_items():
    """Clear action items store before each test."""
    with _action_items_lock:
        _action_items.clear()
    yield
    with _action_items_lock:
        _action_items.clear()


@pytest.fixture
def sample_action_item():
    """Create a sample action item in the store."""
    item = MockActionItem().to_dict()
    with _action_items_lock:
        _action_items["action-123"] = item
    return item


@pytest.fixture
def sample_action_items_with_deadlines():
    """Create sample action items with various deadlines."""
    now = datetime.now(timezone.utc)
    items = [
        {
            "id": "action-1",
            "description": "Overdue task",
            "priority": 1,
            "status": "pending",
            "deadline": (now - timedelta(hours=2)).isoformat(),
        },
        {
            "id": "action-2",
            "description": "Due soon task",
            "priority": 2,
            "status": "pending",
            "deadline": (now + timedelta(hours=12)).isoformat(),
        },
        {
            "id": "action-3",
            "description": "Future task",
            "priority": 3,
            "status": "pending",
            "deadline": (now + timedelta(days=7)).isoformat(),
        },
    ]
    with _action_items_lock:
        for item in items:
            _action_items[item["id"]] = item
    return items


# ===========================================================================
# Extract Action Items Tests
# ===========================================================================


class TestExtractActionItems:
    """Test handle_extract_action_items function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_action_extractor")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_extract_success(self, mock_permission, mock_get_extractor):
        """Test successful action item extraction."""
        mock_permission.return_value = None
        mock_get_extractor.return_value = MockActionItemExtractor()

        data = {
            "email_id": "email-123",
            "subject": "Action Required: Review Report",
            "body": "Please review the quarterly report by Friday.",
            "sender": "manager@example.com",
        }

        result = await handle_extract_action_items(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_extract_missing_content(self, mock_permission):
        """Test extraction fails without subject or body."""
        mock_permission.return_value = None

        data = {"email_id": "email-123", "sender": "test@example.com"}

        result = await handle_extract_action_items(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_action_extractor")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_extract_with_options(self, mock_permission, mock_get_extractor):
        """Test extraction with deadline and assignee detection options."""
        mock_permission.return_value = None
        mock_get_extractor.return_value = MockActionItemExtractor()

        data = {
            "subject": "Task",
            "body": "Do something",
            "extract_deadlines": True,
            "detect_assignees": False,
        }

        result = await handle_extract_action_items(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_extract_rbac_denied(self, mock_permission):
        """Test extraction fails when RBAC denies permission."""
        mock_permission.return_value = MagicMock(status_code=403)

        data = {"subject": "Task", "body": "Do something"}

        result = await handle_extract_action_items(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 403


# ===========================================================================
# List Pending Actions Tests
# ===========================================================================


class TestListPendingActions:
    """Test handle_list_pending_actions function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_list_pending_empty(self, mock_permission):
        """Test listing pending actions when empty."""
        mock_permission.return_value = None

        result = await handle_list_pending_actions({}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_list_pending_with_items(self, mock_permission, sample_action_item):
        """Test listing pending actions with items."""
        mock_permission.return_value = None

        result = await handle_list_pending_actions({}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_list_with_assignee_filter(self, mock_permission, sample_action_item):
        """Test filtering by assignee."""
        mock_permission.return_value = None

        result = await handle_list_pending_actions(
            {"assignee": "user@example.com"}, user_id="user-1"
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_list_with_priority_filter(self, mock_permission, sample_action_item):
        """Test filtering by priority."""
        mock_permission.return_value = None

        result = await handle_list_pending_actions({"priority": "high"}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_list_with_due_within_filter(
        self, mock_permission, sample_action_items_with_deadlines
    ):
        """Test filtering by due_within_hours."""
        mock_permission.return_value = None

        result = await handle_list_pending_actions(
            {"due_within_hours": 24}, user_id="user-1"
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_list_with_pagination(self, mock_permission, sample_action_item):
        """Test pagination parameters."""
        mock_permission.return_value = None

        result = await handle_list_pending_actions(
            {"limit": 10, "offset": 0}, user_id="user-1"
        )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Complete Action Tests
# ===========================================================================


class TestCompleteAction:
    """Test handle_complete_action function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_complete_success(self, mock_permission, sample_action_item):
        """Test successfully completing an action item."""
        mock_permission.return_value = None

        result = await handle_complete_action(
            {"completed_by": "user@example.com"},
            action_id="action-123",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

        # Verify the item was updated
        with _action_items_lock:
            assert _action_items["action-123"]["status"] == "completed"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_complete_with_notes(self, mock_permission, sample_action_item):
        """Test completing with notes."""
        mock_permission.return_value = None

        result = await handle_complete_action(
            {"notes": "Done via email confirmation"},
            action_id="action-123",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_complete_not_found(self, mock_permission):
        """Test completing non-existent action item."""
        mock_permission.return_value = None

        result = await handle_complete_action(
            {}, action_id="nonexistent", user_id="user-1"
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_complete_missing_action_id(self, mock_permission):
        """Test completing without action_id."""
        mock_permission.return_value = None

        result = await handle_complete_action({}, action_id="", user_id="user-1")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Update Action Status Tests
# ===========================================================================


class TestUpdateActionStatus:
    """Test handle_update_action_status function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_update_status_to_in_progress(self, mock_permission, sample_action_item):
        """Test updating status to in_progress."""
        mock_permission.return_value = None

        result = await handle_update_action_status(
            {"status": "in_progress"},
            action_id="action-123",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

        with _action_items_lock:
            assert _action_items["action-123"]["status"] == "in_progress"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_update_status_to_deferred(self, mock_permission, sample_action_item):
        """Test updating status to deferred."""
        mock_permission.return_value = None

        result = await handle_update_action_status(
            {"status": "deferred", "notes": "Waiting for more info"},
            action_id="action-123",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_update_invalid_status(self, mock_permission, sample_action_item):
        """Test updating with invalid status."""
        mock_permission.return_value = None

        result = await handle_update_action_status(
            {"status": "invalid_status"},
            action_id="action-123",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_update_status_not_found(self, mock_permission):
        """Test updating non-existent action item."""
        mock_permission.return_value = None

        result = await handle_update_action_status(
            {"status": "completed"},
            action_id="nonexistent",
            user_id="user-1",
        )

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Get Due Soon Tests
# ===========================================================================


class TestGetDueSoon:
    """Test handle_get_due_soon function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_due_soon_default(
        self, mock_permission, sample_action_items_with_deadlines
    ):
        """Test getting due soon items with default 24 hours."""
        mock_permission.return_value = None

        result = await handle_get_due_soon({}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_due_soon_custom_hours(
        self, mock_permission, sample_action_items_with_deadlines
    ):
        """Test getting due soon items with custom hours window."""
        mock_permission.return_value = None

        result = await handle_get_due_soon({"hours": 48}, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_due_soon_exclude_overdue(
        self, mock_permission, sample_action_items_with_deadlines
    ):
        """Test excluding overdue items."""
        mock_permission.return_value = None

        result = await handle_get_due_soon(
            {"include_overdue": False}, user_id="user-1"
        )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_due_soon_include_overdue_string(
        self, mock_permission, sample_action_items_with_deadlines
    ):
        """Test include_overdue as string parameter."""
        mock_permission.return_value = None

        result = await handle_get_due_soon(
            {"include_overdue": "true"}, user_id="user-1"
        )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Batch Extract Tests
# ===========================================================================


class TestBatchExtract:
    """Test handle_batch_extract function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_action_extractor")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_batch_extract_success(self, mock_permission, mock_get_extractor):
        """Test successful batch extraction."""
        mock_permission.return_value = None
        mock_get_extractor.return_value = MockActionItemExtractor()

        data = {
            "emails": [
                {"email_id": "e1", "subject": "Task 1", "body": "Do thing 1"},
                {"email_id": "e2", "subject": "Task 2", "body": "Do thing 2"},
            ]
        }

        result = await handle_batch_extract(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_batch_extract_missing_emails(self, mock_permission):
        """Test batch extraction fails without emails."""
        mock_permission.return_value = None

        result = await handle_batch_extract({}, user_id="user-1")

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_batch_extract_too_many_emails(self, mock_permission):
        """Test batch extraction fails with too many emails."""
        mock_permission.return_value = None

        data = {"emails": [{"subject": f"Email {i}", "body": "Content"} for i in range(51)]}

        result = await handle_batch_extract(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Meeting Detection Tests
# ===========================================================================


class TestDetectMeeting:
    """Test handle_detect_meeting function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_meeting_detector")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_detect_meeting_success(self, mock_permission, mock_get_detector):
        """Test successful meeting detection."""
        mock_permission.return_value = None
        mock_get_detector.return_value = MockMeetingDetector()

        data = {
            "subject": "Team Sync Tomorrow",
            "body": "Join us at 3pm via Zoom: https://zoom.us/j/123",
            "sender": "manager@example.com",
        }

        result = await handle_detect_meeting(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_detect_meeting_missing_content(self, mock_permission):
        """Test meeting detection fails without subject or body."""
        mock_permission.return_value = None

        data = {"sender": "test@example.com"}

        result = await handle_detect_meeting(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Auto-Snooze Meeting Tests
# ===========================================================================


class TestAutoSnoozeMeeting:
    """Test handle_auto_snooze_meeting function."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_meeting_detector")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_auto_snooze_success(self, mock_permission, mock_get_detector):
        """Test successful auto-snooze."""
        mock_permission.return_value = None
        mock_get_detector.return_value = MockMeetingDetector()

        data = {
            "subject": "Meeting Tomorrow",
            "body": "Join at 3pm",
            "minutes_before": 30,
        }

        result = await handle_auto_snooze_meeting(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_meeting_detector")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_auto_snooze_not_a_meeting(self, mock_permission, mock_get_detector):
        """Test auto-snooze when email is not a meeting."""
        mock_permission.return_value = None
        mock_detector = MockMeetingDetector()
        mock_detector.detect_meeting = AsyncMock(
            return_value=MockMeetingResult(is_meeting=False)
        )
        mock_get_detector.return_value = mock_detector

        data = {"subject": "Newsletter", "body": "Weekly update"}

        result = await handle_auto_snooze_meeting(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_meeting_detector")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_auto_snooze_no_start_time(self, mock_permission, mock_get_detector):
        """Test auto-snooze when meeting has no start time."""
        mock_permission.return_value = None
        mock_detector = MockMeetingDetector()
        mock_detector.detect_meeting = AsyncMock(
            return_value=MockMeetingResult(start_time=None)
        )
        mock_get_detector.return_value = mock_detector

        data = {"subject": "Meeting", "body": "Let's meet sometime"}

        result = await handle_auto_snooze_meeting(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.inbox.action_items.get_meeting_detector")
    @patch("aragora.server.handlers.inbox.action_items._check_inbox_permission")
    async def test_auto_snooze_meeting_too_soon(self, mock_permission, mock_get_detector):
        """Test auto-snooze when meeting is too soon."""
        mock_permission.return_value = None
        mock_detector = MockMeetingDetector()
        mock_detector.detect_meeting = AsyncMock(
            return_value=MockMeetingResult(
                start_time=datetime.now(timezone.utc) + timedelta(minutes=10)
            )
        )
        mock_get_detector.return_value = mock_detector

        data = {"subject": "Meeting Now", "body": "Starting in 10 min"}

        result = await handle_auto_snooze_meeting(data, user_id="user-1")

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Handler Registration Tests
# ===========================================================================


class TestHandlerRegistration:
    """Test handler registration function."""

    def test_get_action_items_handlers_returns_all(self):
        """Test that get_action_items_handlers returns all handlers."""
        handlers = get_action_items_handlers()

        assert "extract_action_items" in handlers
        assert "list_pending_actions" in handlers
        assert "complete_action" in handlers
        assert "update_action_status" in handlers
        assert "get_due_soon" in handlers
        assert "batch_extract" in handlers
        assert "detect_meeting" in handlers
        assert "auto_snooze_meeting" in handlers

    def test_handlers_are_callable(self):
        """Test that all registered handlers are callable."""
        handlers = get_action_items_handlers()

        for name, handler in handlers.items():
            assert callable(handler), f"Handler {name} is not callable"
