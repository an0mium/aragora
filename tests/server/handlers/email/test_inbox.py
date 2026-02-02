"""
Tests for email inbox fetch-and-rank handler.

Covers:
- handle_fetch_and_rank_inbox
- Gmail OAuth not-authenticated flow
- Successful fetch + rank flow
- Error handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_mod
from aragora.server.handlers.email import inbox as inbox_mod

# Access the unwrapped function (decorators use functools.wraps which preserves __wrapped__)
# Need to unwrap twice: once for rate_limit, once for require_permission
_raw_handle = inbox_mod.handle_fetch_and_rank_inbox
while hasattr(_raw_handle, "__wrapped__"):
    _raw_handle = _raw_handle.__wrapped__


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


@dataclass
class FakeEmail:
    id: str = "msg_1"
    thread_id: str = "thread_1"
    subject: str = "Test subject"
    from_address: str = "sender@test.com"
    to_addresses: list[str] = field(default_factory=lambda: ["me@test.com"])
    date: datetime = field(default_factory=datetime.now)
    snippet: str = "Hello..."
    labels: list[str] = field(default_factory=lambda: ["INBOX"])
    is_read: bool = False
    is_starred: bool = False
    is_important: bool = False
    attachments: list = field(default_factory=list)


@dataclass
class FakePriorityResult:
    email_id: str = "msg_1"
    score: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        return {"email_id": self.email_id, "score": self.score}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    storage_mod._gmail_connector = None
    storage_mod._prioritizer = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()
    yield
    storage_mod._gmail_connector = None
    storage_mod._prioritizer = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()


@pytest.fixture
def mock_connector():
    c = AsyncMock()
    c._access_token = "valid-token"
    c.list_messages = AsyncMock(return_value=(["msg_1", "msg_2"], None))
    c.get_message = AsyncMock(side_effect=lambda mid: FakeEmail(id=mid))
    return c


@pytest.fixture
def mock_prioritizer():
    p = AsyncMock()
    p.rank_inbox = AsyncMock(
        return_value=[FakePriorityResult(email_id="msg_1"), FakePriorityResult(email_id="msg_2")]
    )
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandleFetchAndRankInbox:
    @pytest.mark.asyncio
    async def test_not_authenticated(self):
        """Returns needs_auth when Gmail connector has no token."""
        mock_conn = MagicMock()
        mock_conn._access_token = None
        with patch(
            "aragora.server.handlers.email.inbox.get_gmail_connector",
            return_value=mock_conn,
        ):
            result = await _raw_handle()
        assert result["success"] is False
        assert result["needs_auth"] is True
        assert "OAuth" in result["error"]

    @pytest.mark.asyncio
    async def test_success_flow(self, mock_connector, mock_prioritizer):
        with (
            patch(
                "aragora.server.handlers.email.inbox.get_gmail_connector",
                return_value=mock_connector,
            ),
            patch(
                "aragora.server.handlers.email.inbox.get_prioritizer",
                return_value=mock_prioritizer,
            ),
        ):
            result = await _raw_handle(limit=10)
        assert result["success"] is True
        assert result["total"] == 2
        assert len(result["inbox"]) == 2
        assert "fetched_at" in result

    @pytest.mark.asyncio
    async def test_inbox_item_structure(self, mock_connector, mock_prioritizer):
        with (
            patch(
                "aragora.server.handlers.email.inbox.get_gmail_connector",
                return_value=mock_connector,
            ),
            patch(
                "aragora.server.handlers.email.inbox.get_prioritizer",
                return_value=mock_prioritizer,
            ),
        ):
            result = await _raw_handle()
        item = result["inbox"][0]
        assert "email" in item
        assert "priority" in item
        email = item["email"]
        assert "id" in email
        assert "subject" in email
        assert "from_address" in email

    @pytest.mark.asyncio
    async def test_unread_filter(self, mock_connector, mock_prioritizer):
        """Default include_read=False adds is:unread query."""
        with (
            patch(
                "aragora.server.handlers.email.inbox.get_gmail_connector",
                return_value=mock_connector,
            ),
            patch(
                "aragora.server.handlers.email.inbox.get_prioritizer",
                return_value=mock_prioritizer,
            ),
        ):
            await _raw_handle(include_read=False)
        call_kwargs = mock_connector.list_messages.call_args[1]
        assert "is:unread" in call_kwargs["query"]

    @pytest.mark.asyncio
    async def test_include_read(self, mock_connector, mock_prioritizer):
        """include_read=True sends empty query."""
        with (
            patch(
                "aragora.server.handlers.email.inbox.get_gmail_connector",
                return_value=mock_connector,
            ),
            patch(
                "aragora.server.handlers.email.inbox.get_prioritizer",
                return_value=mock_prioritizer,
            ),
        ):
            await _raw_handle(include_read=True)
        call_kwargs = mock_connector.list_messages.call_args[1]
        assert "is:unread" not in call_kwargs.get("query", "")

    @pytest.mark.asyncio
    async def test_custom_labels(self, mock_connector, mock_prioritizer):
        with (
            patch(
                "aragora.server.handlers.email.inbox.get_gmail_connector",
                return_value=mock_connector,
            ),
            patch(
                "aragora.server.handlers.email.inbox.get_prioritizer",
                return_value=mock_prioritizer,
            ),
        ):
            await _raw_handle(labels=["STARRED"])
        call_kwargs = mock_connector.list_messages.call_args[1]
        assert call_kwargs["label_ids"] == ["STARRED"]

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_connector):
        mock_connector.list_messages.side_effect = RuntimeError("API error")
        with patch(
            "aragora.server.handlers.email.inbox.get_gmail_connector",
            return_value=mock_connector,
        ):
            result = await _raw_handle()
        assert result["success"] is False
        assert "API error" in result["error"]

    @pytest.mark.asyncio
    async def test_decorator_is_applied(self):
        """Verify the handler has RBAC decorator applied."""
        assert hasattr(inbox_mod.handle_fetch_and_rank_inbox, "__wrapped__")
