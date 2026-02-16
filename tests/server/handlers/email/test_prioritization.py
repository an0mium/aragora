"""
Tests for email prioritization handlers.

Covers:
- handle_prioritize_email (single scoring)
- handle_rank_inbox (batch ranking)
- handle_email_feedback (action recording)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_mod
import aragora.server.handlers.email.prioritization as prio_mod

# Access the underlying functions without decorators
_handle_prioritize_email = getattr(
    prio_mod.handle_prioritize_email,
    "__wrapped__",
    getattr(prio_mod.handle_prioritize_email, "__wrapped__", prio_mod.handle_prioritize_email),
)
_handle_rank_inbox = getattr(
    prio_mod.handle_rank_inbox,
    "__wrapped__",
    getattr(prio_mod.handle_rank_inbox, "__wrapped__", prio_mod.handle_rank_inbox),
)
_handle_email_feedback = getattr(
    prio_mod.handle_email_feedback,
    "__wrapped__",
    getattr(prio_mod.handle_email_feedback, "__wrapped__", prio_mod.handle_email_feedback),
)


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


@dataclass
class FakePriorityResult:
    email_id: str = "msg_1"
    score: float = 0.85
    tier: str = "tier_1"

    def to_dict(self) -> dict[str, Any]:
        return {"email_id": self.email_id, "score": self.score, "tier": self.tier}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    storage_mod._prioritizer = None
    storage_mod._gmail_connector = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()
    yield
    storage_mod._prioritizer = None
    storage_mod._gmail_connector = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()


@pytest.fixture
def mock_prioritizer():
    p = AsyncMock()
    p.score_email = AsyncMock(return_value=FakePriorityResult())
    p.rank_inbox = AsyncMock(
        return_value=[FakePriorityResult(), FakePriorityResult(email_id="msg_2")]
    )
    p.record_user_action = AsyncMock()
    return p


SAMPLE_EMAIL = {
    "id": "msg_1",
    "subject": "Urgent: deadline",
    "from_address": "boss@company.com",
    "body_text": "Project deadline tomorrow",
    "labels": ["INBOX", "IMPORTANT"],
}


# ---------------------------------------------------------------------------
# handle_prioritize_email
# ---------------------------------------------------------------------------


class TestHandlePrioritizeEmail:
    @pytest.mark.asyncio
    async def test_success(self, mock_prioritizer):
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_prioritize_email(SAMPLE_EMAIL)
        assert result["success"] is True
        assert result["result"]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_with_force_tier(self, mock_prioritizer):
        with (
            patch(
                "aragora.server.handlers.email.prioritization.get_prioritizer",
                return_value=mock_prioritizer,
            ),
            patch(
                "aragora.services.email_prioritization.ScoringTier",
                side_effect=lambda x: x,
            ),
        ):
            result = await _handle_prioritize_email(SAMPLE_EMAIL, force_tier="tier_1_rules")
        assert result["success"] is True
        mock_prioritizer.score_email.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_prioritizer):
        mock_prioritizer.score_email.side_effect = ValueError("scoring failed")
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_prioritize_email(SAMPLE_EMAIL)
        assert result["success"] is False
        assert result["error"]  # Sanitized error message present

    @pytest.mark.asyncio
    async def test_decorator_is_applied(self):
        """Verify the handler has RBAC decorator applied."""
        assert hasattr(prio_mod.handle_prioritize_email, "__wrapped__")


# ---------------------------------------------------------------------------
# handle_rank_inbox
# ---------------------------------------------------------------------------


class TestHandleRankInbox:
    @pytest.mark.asyncio
    async def test_success(self, mock_prioritizer):
        emails = [SAMPLE_EMAIL, {**SAMPLE_EMAIL, "id": "msg_2"}]
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_rank_inbox(emails)
        assert result["success"] is True
        assert result["total"] == 2
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_with_limit(self, mock_prioritizer):
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_rank_inbox([SAMPLE_EMAIL], limit=5)
        assert result["success"] is True
        mock_prioritizer.rank_inbox.assert_called_once()
        _, kwargs = mock_prioritizer.rank_inbox.call_args
        assert kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_prioritizer):
        mock_prioritizer.rank_inbox.side_effect = ValueError("rank failed")
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_rank_inbox([SAMPLE_EMAIL])
        assert result["success"] is False


# ---------------------------------------------------------------------------
# handle_email_feedback
# ---------------------------------------------------------------------------


class TestHandleEmailFeedback:
    @pytest.mark.asyncio
    async def test_success_without_email_data(self, mock_prioritizer):
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_email_feedback("msg_1", "archived")
        assert result["success"] is True
        assert result["email_id"] == "msg_1"
        assert result["action"] == "archived"
        assert "recorded_at" in result

    @pytest.mark.asyncio
    async def test_success_with_email_data(self, mock_prioritizer):
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_email_feedback("msg_1", "replied", email_data=SAMPLE_EMAIL)
        assert result["success"] is True
        # Verify prioritizer received an EmailMessage-like object
        call_args = mock_prioritizer.record_user_action.call_args
        assert call_args[0][0] == "msg_1"
        assert call_args[0][1] == "replied"

    @pytest.mark.asyncio
    async def test_exception_returns_error(self, mock_prioritizer):
        mock_prioritizer.record_user_action.side_effect = ValueError("db error")
        with patch(
            "aragora.server.handlers.email.prioritization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await _handle_email_feedback("msg_1", "archived")
        assert result["success"] is False
        assert result["error"]  # Sanitized error message present

    @pytest.mark.asyncio
    async def test_decorator_is_applied(self):
        """Verify the handler has RBAC decorator applied."""
        assert hasattr(prio_mod.handle_email_feedback, "__wrapped__")
