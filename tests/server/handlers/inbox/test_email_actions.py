"""
Tests for aragora.server.handlers.inbox.email_actions - Email Actions API handler.

Tests cover 18 endpoints:
- Send/Reply: send_email, reply_email
- Message actions: archive, trash, restore, snooze
- Read/Star: mark_read, mark_unread, star, unstar
- Organization: move_to_folder, add_label, remove_label
- Batch operations: batch_archive, batch_trash, batch_modify
- Audit: get_action_logs, export_action_logs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.inbox import email_actions


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockActionType(Enum):
    """Mock action type enum."""

    SEND = "send"
    REPLY = "reply"
    ARCHIVE = "archive"
    TRASH = "trash"
    SNOOZE = "snooze"
    MARK_READ = "mark_read"
    STAR = "star"


class MockEmailProvider(Enum):
    """Mock email provider enum."""

    GMAIL = "gmail"
    OUTLOOK = "outlook"


@dataclass
class MockSendResult:
    """Mock send email result."""

    success: bool = True
    message_id: str = "msg-123"
    thread_id: str = "thread-123"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
        }


@dataclass
class MockActionResult:
    """Mock action result."""

    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"success": self.success}
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class MockActionLog:
    """Mock action log entry."""

    id: str = "log-123"
    user_id: str = "user-123"
    action_type: str = "send"
    message_id: str = "msg-123"
    provider: str = "gmail"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action_type": self.action_type,
            "message_id": self.message_id,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
        }


class MockConnector:
    """Mock email connector for testing."""

    async def send_message(self, **kwargs) -> MockSendResult:
        return MockSendResult()

    async def untrash_message(self, message_id: str) -> bool:
        return True

    async def mark_as_unread(self, message_id: str) -> bool:
        return True

    async def unstar_message(self, message_id: str) -> bool:
        return True

    async def modify_message(
        self, message_id: str, add_labels: list | None = None, remove_labels: list | None = None
    ) -> bool:
        return True

    async def batch_trash(self, message_ids: list[str]) -> bool:
        return True

    async def batch_modify(
        self,
        message_ids: list[str],
        add_labels: list | None = None,
        remove_labels: list | None = None,
    ) -> bool:
        return True


class MockEmailActionsService:
    """Mock email actions service for testing."""

    def __init__(self):
        self.connector = MockConnector()

    async def send(self, provider: str, user_id: str, request: Any) -> MockSendResult:
        return MockSendResult()

    async def reply(
        self,
        provider: str,
        user_id: str,
        message_id: str,
        body: str,
        cc: list | None = None,
        html_body: str | None = None,
    ) -> MockSendResult:
        return MockSendResult()

    async def archive(self, provider: str, user_id: str, message_id: str) -> MockActionResult:
        return MockActionResult()

    async def trash(self, provider: str, user_id: str, message_id: str) -> MockActionResult:
        return MockActionResult()

    async def snooze(
        self, provider: str, user_id: str, message_id: str, snooze_until: datetime
    ) -> MockActionResult:
        return MockActionResult()

    async def mark_read(self, provider: str, user_id: str, message_id: str) -> MockActionResult:
        return MockActionResult()

    async def star(self, provider: str, user_id: str, message_id: str) -> MockActionResult:
        return MockActionResult()

    async def move_to_folder(
        self, provider: str, user_id: str, message_id: str, folder: str
    ) -> MockActionResult:
        return MockActionResult()

    async def batch_archive(
        self, provider: str, user_id: str, message_ids: list[str]
    ) -> MockActionResult:
        return MockActionResult()

    async def get_action_logs(
        self,
        user_id: str,
        action_type: Any = None,
        provider: Any = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[MockActionLog]:
        return [MockActionLog(), MockActionLog()]

    async def export_action_logs(
        self, user_id: str, start_date: datetime, end_date: datetime
    ) -> list[MockActionLog]:
        return [MockActionLog()]

    async def _get_connector(self, provider: str, user_id: str) -> MockConnector:
        return self.connector


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_service():
    """Create a mock email actions service."""
    return MockEmailActionsService()


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_response(result) -> tuple[dict[str, Any], int]:
    """Parse HandlerResult into (body_dict, status_code)."""
    body = json.loads(result.body) if result.body else {}
    # Unwrap data if present (success_response format)
    if "data" in body and body.get("success"):
        return body["data"], result.status_code
    return body, result.status_code


# ===========================================================================
# Test: Send Email
# ===========================================================================


class TestSendEmail:
    """Tests for POST /api/v1/inbox/messages/send."""

    @pytest.mark.asyncio
    async def test_send_email_success(self, mock_service):
        """Should send email successfully."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_send_email(
                data={
                    "to": ["recipient@example.com"],
                    "subject": "Test Subject",
                    "body": "Test body content",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_send_email_missing_to(self, mock_service):
        """Should return 400 when 'to' is missing."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_send_email(
                data={
                    "subject": "Test",
                    "body": "Body",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_send_email_empty_to(self, mock_service):
        """Should return 400 when 'to' is empty."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_send_email(
                data={
                    "to": [],
                    "subject": "Test",
                    "body": "Body",
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_send_email_no_subject_or_body(self, mock_service):
        """Should return 400 when both subject and body are missing."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_send_email(
                data={
                    "to": ["recipient@example.com"],
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


# ===========================================================================
# Test: Reply Email
# ===========================================================================


class TestReplyEmail:
    """Tests for POST /api/v1/inbox/messages/{id}/reply."""

    @pytest.mark.asyncio
    async def test_reply_email_success(self, mock_service):
        """Should reply to email successfully."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_reply_email(
                data={
                    "body": "Reply content",
                },
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_reply_email_missing_body(self, mock_service):
        """Should return 400 when body is missing."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_reply_email(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_reply_email_missing_message_id(self, mock_service):
        """Should return 400 when message_id is missing."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_reply_email(
                data={"body": "Reply"},
                message_id="",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


# ===========================================================================
# Test: Archive Message
# ===========================================================================


class TestArchiveMessage:
    """Tests for POST /api/v1/inbox/messages/{id}/archive."""

    @pytest.mark.asyncio
    async def test_archive_message_success(self, mock_service):
        """Should archive message successfully."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_archive_message(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "archive"

    @pytest.mark.asyncio
    async def test_archive_message_missing_id(self, mock_service):
        """Should return 400 when message_id is missing."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_archive_message(
                data={},
                message_id="",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


# ===========================================================================
# Test: Trash Message
# ===========================================================================


class TestTrashMessage:
    """Tests for POST /api/v1/inbox/messages/{id}/trash."""

    @pytest.mark.asyncio
    async def test_trash_message_success(self, mock_service):
        """Should trash message successfully."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_trash_message(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "trash"


# ===========================================================================
# Test: Restore Message
# ===========================================================================


class TestRestoreMessage:
    """Tests for POST /api/v1/inbox/messages/{id}/restore."""

    @pytest.mark.asyncio
    async def test_restore_message_success(self, mock_service):
        """Should restore message successfully."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_restore_message(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "restore"


# ===========================================================================
# Test: Snooze Message
# ===========================================================================


class TestSnoozeMessage:
    """Tests for POST /api/v1/inbox/messages/{id}/snooze."""

    @pytest.mark.asyncio
    async def test_snooze_with_hours(self, mock_service):
        """Should snooze message with hours parameter."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_snooze_message(
                data={"snooze_hours": 2},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "snooze"

    @pytest.mark.asyncio
    async def test_snooze_with_days(self, mock_service):
        """Should snooze message with days parameter."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_snooze_message(
                data={"snooze_days": 1},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_snooze_with_until_datetime(self, mock_service):
        """Should snooze message with snooze_until parameter."""
        future_time = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_snooze_message(
                data={"snooze_until": future_time},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200

    @pytest.mark.asyncio
    async def test_snooze_no_time_parameter(self, mock_service):
        """Should return 400 when no snooze time parameter provided."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_snooze_message(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


# ===========================================================================
# Test: Mark Read/Unread
# ===========================================================================


class TestMarkRead:
    """Tests for POST /api/v1/inbox/messages/{id}/read."""

    @pytest.mark.asyncio
    async def test_mark_read_success(self, mock_service):
        """Should mark message as read."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_mark_read(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "mark_read"


class TestMarkUnread:
    """Tests for POST /api/v1/inbox/messages/{id}/unread."""

    @pytest.mark.asyncio
    async def test_mark_unread_success(self, mock_service):
        """Should mark message as unread."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_mark_unread(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "mark_unread"


# ===========================================================================
# Test: Star/Unstar
# ===========================================================================


class TestStarMessage:
    """Tests for POST /api/v1/inbox/messages/{id}/star."""

    @pytest.mark.asyncio
    async def test_star_message_success(self, mock_service):
        """Should star message successfully."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_star_message(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "star"


class TestUnstarMessage:
    """Tests for POST /api/v1/inbox/messages/{id}/unstar."""

    @pytest.mark.asyncio
    async def test_unstar_message_success(self, mock_service):
        """Should unstar message successfully."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_unstar_message(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "unstar"


# ===========================================================================
# Test: Move to Folder
# ===========================================================================


class TestMoveToFolder:
    """Tests for POST /api/v1/inbox/messages/{id}/move."""

    @pytest.mark.asyncio
    async def test_move_to_folder_success(self, mock_service):
        """Should move message to folder."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_move_to_folder(
                data={"folder": "Archive"},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "move_to_folder"
            assert body.get("folder") == "Archive"

    @pytest.mark.asyncio
    async def test_move_to_folder_missing_folder(self, mock_service):
        """Should return 400 when folder is missing."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_move_to_folder(
                data={},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


# ===========================================================================
# Test: Add/Remove Labels
# ===========================================================================


class TestAddLabel:
    """Tests for POST /api/v1/inbox/messages/{id}/labels/add."""

    @pytest.mark.asyncio
    async def test_add_label_success(self, mock_service):
        """Should add labels to message."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_add_label(
                data={"labels": ["Important", "Work"]},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "add_labels"

    @pytest.mark.asyncio
    async def test_add_label_empty_labels(self, mock_service):
        """Should return 400 when labels is empty."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_add_label(
                data={"labels": []},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


class TestRemoveLabel:
    """Tests for POST /api/v1/inbox/messages/{id}/labels/remove."""

    @pytest.mark.asyncio
    async def test_remove_label_success(self, mock_service):
        """Should remove labels from message."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_remove_label(
                data={"labels": ["Old"]},
                message_id="msg-123",
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "remove_labels"


# ===========================================================================
# Test: Batch Operations
# ===========================================================================


class TestBatchArchive:
    """Tests for POST /api/v1/inbox/messages/batch/archive."""

    @pytest.mark.asyncio
    async def test_batch_archive_success(self, mock_service):
        """Should batch archive messages."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_batch_archive(
                data={"message_ids": ["msg-1", "msg-2", "msg-3"]},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "batch_archive"
            assert body.get("count") == 3

    @pytest.mark.asyncio
    async def test_batch_archive_empty_ids(self, mock_service):
        """Should return 400 when message_ids is empty."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_batch_archive(
                data={"message_ids": []},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_batch_archive_exceeds_limit(self, mock_service):
        """Should return 400 when batch exceeds 100 messages."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_batch_archive(
                data={"message_ids": [f"msg-{i}" for i in range(101)]},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


class TestBatchTrash:
    """Tests for POST /api/v1/inbox/messages/batch/trash."""

    @pytest.mark.asyncio
    async def test_batch_trash_success(self, mock_service):
        """Should batch trash messages."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_batch_trash(
                data={"message_ids": ["msg-1", "msg-2"]},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "batch_trash"


class TestBatchModify:
    """Tests for POST /api/v1/inbox/messages/batch/modify."""

    @pytest.mark.asyncio
    async def test_batch_modify_add_labels(self, mock_service):
        """Should batch add labels to messages."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_batch_modify(
                data={
                    "message_ids": ["msg-1", "msg-2"],
                    "add_labels": ["Important"],
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("action") == "batch_modify"

    @pytest.mark.asyncio
    async def test_batch_modify_no_label_operation(self, mock_service):
        """Should return 400 when no label operation provided."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_batch_modify(
                data={"message_ids": ["msg-1"]},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400


# ===========================================================================
# Test: Action Logs
# ===========================================================================


class TestGetActionLogs:
    """Tests for GET /api/v1/inbox/actions/logs."""

    @pytest.mark.asyncio
    async def test_get_action_logs_success(self, mock_service):
        """Should return action logs."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_get_action_logs(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "logs" in body
            assert body.get("count") == 2

    @pytest.mark.asyncio
    async def test_get_action_logs_with_limit(self, mock_service):
        """Should respect limit parameter."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_get_action_logs(
                data={"limit": 50},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert body.get("limit") == 50


class TestExportActionLogs:
    """Tests for GET /api/v1/inbox/actions/export."""

    @pytest.mark.asyncio
    async def test_export_logs_success(self, mock_service):
        """Should export action logs."""
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)

        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_export_action_logs(
                data={
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 200
            assert "logs" in body

    @pytest.mark.asyncio
    async def test_export_logs_missing_dates(self, mock_service):
        """Should return 400 when dates are missing."""
        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_export_action_logs(
                data={},
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400

    @pytest.mark.asyncio
    async def test_export_logs_exceeds_90_days(self, mock_service):
        """Should return 400 when date range exceeds 90 days."""
        start = datetime.now(timezone.utc) - timedelta(days=100)
        end = datetime.now(timezone.utc)

        with patch.object(
            email_actions, "get_email_actions_service_instance", return_value=mock_service
        ):
            result = await email_actions.handle_export_action_logs(
                data={
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                },
                user_id="user-123",
            )
            body, status = parse_response(result)

            assert status == 400
