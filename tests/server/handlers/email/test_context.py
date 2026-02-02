"""
Tests for email cross-channel context handlers.

Covers:
- handle_get_context (get context for an email address)
- handle_get_email_context_boost (get boost signals)
- RBAC permission checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_mod
from aragora.server.handlers.email.context import (
    handle_get_context,
    handle_get_email_context_boost,
)


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


@dataclass
class FakeUserContext:
    email: str = "user@test.com"
    slack_activity: int = 5
    drive_files: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "email": self.email,
            "slack_activity": self.slack_activity,
            "drive_files": self.drive_files,
        }


@dataclass
class FakeEmailContext:
    email_id: str = "msg_1"
    total_boost: float = 0.3
    slack_activity_boost: float = 0.1
    drive_relevance_boost: float = 0.1
    calendar_urgency_boost: float = 0.1
    slack_reason: str = "Active in #project"
    drive_reason: str = "Shared doc recently"
    calendar_reason: str = "Meeting tomorrow"
    related_slack_channels: list[str] = field(default_factory=lambda: ["#project"])
    related_drive_files: list[str] = field(default_factory=lambda: ["design.doc"])
    related_meetings: list[str] = field(default_factory=lambda: ["standup"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    storage_mod._context_service = None
    yield
    storage_mod._context_service = None


@pytest.fixture(autouse=True)
def _bypass_rbac():
    with patch(
        "aragora.server.handlers.email.context._check_email_permission",
        return_value=None,
    ):
        yield


@pytest.fixture
def mock_context_service():
    svc = AsyncMock()
    svc.get_user_context = AsyncMock(return_value=FakeUserContext())
    svc.get_email_context = AsyncMock(return_value=FakeEmailContext())
    return svc


# ---------------------------------------------------------------------------
# handle_get_context
# ---------------------------------------------------------------------------


class TestHandleGetContext:
    @pytest.mark.asyncio
    async def test_success(self, mock_context_service):
        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_context_service,
        ):
            result = await handle_get_context("user@test.com")
        assert result["success"] is True
        assert result["context"]["email"] == "user@test.com"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_context_service):
        mock_context_service.get_user_context.side_effect = RuntimeError("service down")
        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_context_service,
        ):
            result = await handle_get_context("user@test.com")
        assert result["success"] is False
        assert "service down" in result["error"]

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        with patch(
            "aragora.server.handlers.email.context._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_get_context("user@test.com", auth_context=MagicMock())
        assert result["success"] is False


# ---------------------------------------------------------------------------
# handle_get_email_context_boost
# ---------------------------------------------------------------------------


class TestHandleGetEmailContextBoost:
    @pytest.mark.asyncio
    async def test_success(self, mock_context_service):
        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_context_service,
        ):
            result = await handle_get_email_context_boost(
                {"id": "msg_1", "from_address": "sender@test.com"}
            )
        assert result["success"] is True
        boost = result["boost"]
        assert boost["total_boost"] == 0.3
        assert boost["slack_activity_boost"] == 0.1
        assert boost["drive_relevance_boost"] == 0.1
        assert boost["calendar_urgency_boost"] == 0.1
        assert "#project" in boost["related_slack_channels"]

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_context_service):
        mock_context_service.get_email_context.side_effect = RuntimeError("boom")
        with patch(
            "aragora.server.handlers.email.context.get_context_service",
            return_value=mock_context_service,
        ):
            result = await handle_get_email_context_boost({"id": "msg_1"})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        with patch(
            "aragora.server.handlers.email.context._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_get_email_context_boost({"id": "msg_1"}, auth_context=MagicMock())
        assert result["success"] is False
