"""
Tests for email categorization handlers.

Covers:
- handle_categorize_email (single email)
- handle_categorize_batch (batch)
- handle_feedback_batch (batch feedback recording)
- handle_apply_category_label (Gmail label application)
- get_categorizer (lazy init)
- RBAC permission checks on all handlers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_mod
from aragora.server.handlers.email.categorization import (
    get_categorizer,
    handle_apply_category_label,
    handle_categorize_batch,
    handle_categorize_email,
    handle_feedback_batch,
)
import aragora.server.handlers.email.categorization as cat_mod


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


@dataclass
class FakeCategoryResult:
    category: str = "invoices"
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "confidence": self.confidence}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset categorizer singleton between tests."""
    cat_mod._categorizer = None
    storage_mod._gmail_connector = None
    storage_mod._prioritizer = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()
    yield
    cat_mod._categorizer = None
    storage_mod._gmail_connector = None
    storage_mod._prioritizer = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()


@pytest.fixture
def mock_categorizer():
    """A mock categorizer for injection."""
    cat = AsyncMock()
    cat.categorize_email = AsyncMock(return_value=FakeCategoryResult())
    cat.categorize_batch = AsyncMock(
        return_value=[FakeCategoryResult(), FakeCategoryResult(category="newsletters")]
    )
    cat.get_category_stats = MagicMock(return_value={"invoices": 1, "newsletters": 1})
    cat.apply_gmail_label = AsyncMock(return_value=True)
    return cat


@pytest.fixture(autouse=True)
def _bypass_rbac():
    """Bypass RBAC for most tests; individual tests override when needed."""
    with patch(
        "aragora.server.handlers.email.categorization._check_email_permission",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _bypass_rate_limit():
    """Disable rate limiting in tests."""
    with patch(
        "aragora.server.handlers.email.categorization.rate_limit",
        lambda **kw: (lambda fn: fn),
    ):
        yield


SAMPLE_EMAIL = {
    "id": "msg_1",
    "subject": "Invoice #999",
    "from_address": "billing@corp.com",
    "body_text": "Please find invoice attached.",
}


# ---------------------------------------------------------------------------
# handle_categorize_email
# ---------------------------------------------------------------------------


class TestHandleCategorizeEmail:
    @pytest.mark.asyncio
    async def test_success(self, mock_categorizer):
        cat_mod._categorizer = mock_categorizer
        with patch(
            "aragora.server.handlers.email.categorization.get_categorizer",
            return_value=mock_categorizer,
        ):
            result = await handle_categorize_email(SAMPLE_EMAIL)
        assert result["success"] is True
        assert result["result"]["category"] == "invoices"

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self, mock_categorizer):
        mock_categorizer.categorize_email.side_effect = RuntimeError("model down")
        with patch(
            "aragora.server.handlers.email.categorization.get_categorizer",
            return_value=mock_categorizer,
        ):
            result = await handle_categorize_email(SAMPLE_EMAIL)
        assert result["success"] is False
        assert "model down" in result["error"]

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        """Write-sensitive permission check blocks the request."""
        with patch(
            "aragora.server.handlers.email.categorization._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_categorize_email(SAMPLE_EMAIL, auth_context=MagicMock())
        assert result["success"] is False
        assert result["error"] == "denied"


# ---------------------------------------------------------------------------
# handle_categorize_batch
# ---------------------------------------------------------------------------


class TestHandleCategorizeBatch:
    @pytest.mark.asyncio
    async def test_success(self, mock_categorizer):
        emails = [SAMPLE_EMAIL, {**SAMPLE_EMAIL, "id": "msg_2"}]
        with patch(
            "aragora.server.handlers.email.categorization.get_categorizer",
            return_value=mock_categorizer,
        ):
            result = await handle_categorize_batch(emails)
        assert result["success"] is True
        assert len(result["results"]) == 2
        assert "stats" in result

    @pytest.mark.asyncio
    async def test_error(self, mock_categorizer):
        mock_categorizer.categorize_batch.side_effect = RuntimeError("batch fail")
        with patch(
            "aragora.server.handlers.email.categorization.get_categorizer",
            return_value=mock_categorizer,
        ):
            result = await handle_categorize_batch([SAMPLE_EMAIL])
        assert result["success"] is False


# ---------------------------------------------------------------------------
# handle_feedback_batch
# ---------------------------------------------------------------------------


class TestHandleFeedbackBatch:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_prioritizer = AsyncMock()
        mock_prioritizer.record_user_action = AsyncMock()
        with patch(
            "aragora.server.handlers.email.categorization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await handle_feedback_batch(
                [
                    {"email_id": "msg_1", "action": "archived"},
                    {"email_id": "msg_2", "action": "replied", "response_time_minutes": 5},
                ]
            )
        assert result["success"] is True
        assert result["recorded"] == 2
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_missing_fields_counted_as_errors(self):
        mock_prioritizer = AsyncMock()
        with patch(
            "aragora.server.handlers.email.categorization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await handle_feedback_batch(
                [
                    {"email_id": "msg_1"},  # missing action
                    {"action": "archived"},  # missing email_id
                ]
            )
        assert result["success"] is True
        assert result["recorded"] == 0
        assert result["errors"] == 2

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        mock_prioritizer = AsyncMock()
        mock_prioritizer.record_user_action = AsyncMock(side_effect=[None, RuntimeError("fail")])
        with patch(
            "aragora.server.handlers.email.categorization.get_prioritizer",
            return_value=mock_prioritizer,
        ):
            result = await handle_feedback_batch(
                [
                    {"email_id": "msg_1", "action": "archived"},
                    {"email_id": "msg_2", "action": "deleted"},
                ]
            )
        assert result["recorded"] == 1
        assert result["errors"] == 1

    @pytest.mark.asyncio
    async def test_rbac_denied_for_write(self):
        with patch(
            "aragora.server.handlers.email.categorization._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_feedback_batch(
                [{"email_id": "msg_1", "action": "archived"}],
                auth_context=MagicMock(),
            )
        assert result["success"] is False


# ---------------------------------------------------------------------------
# handle_apply_category_label
# ---------------------------------------------------------------------------


class TestHandleApplyCategoryLabel:
    @pytest.mark.asyncio
    async def test_success(self, mock_categorizer):
        with (
            patch(
                "aragora.server.handlers.email.categorization.get_categorizer",
                return_value=mock_categorizer,
            ),
            patch(
                "aragora.services.email_categorizer.EmailCategory",
                side_effect=lambda x: x,
            ),
        ):
            result = await handle_apply_category_label("msg_1", "invoices")
        assert result["success"] is True
        assert result["label_applied"] is True
        assert result["email_id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_invalid_category(self, mock_categorizer):
        with (
            patch(
                "aragora.server.handlers.email.categorization.get_categorizer",
                return_value=mock_categorizer,
            ),
            patch(
                "aragora.services.email_categorizer.EmailCategory",
                side_effect=ValueError("bad"),
            ),
        ):
            result = await handle_apply_category_label("msg_1", "nonexistent")
        assert result["success"] is False
        assert "Invalid category" in result["error"]

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        with patch(
            "aragora.server.handlers.email.categorization._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_apply_category_label(
                "msg_1", "invoices", auth_context=MagicMock()
            )
        assert result["success"] is False
