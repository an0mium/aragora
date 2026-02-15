"""
Tests for aragora.server.handlers.moderation - Moderation HTTP Handlers.

Tests cover:
- ModerationHandler: instantiation, RESOURCE_TYPE, can_handle
- GET /api/moderation/config: success, moderation init failure
- GET /api/moderation/stats: success with queue size
- GET /api/moderation/queue: success with pagination
- PUT /api/moderation/config: success, invalid body
- POST /api/moderation/items/{id}/approve: found, not found
- POST /api/moderation/items/{id}/reject: found, not found
- handle routing: returns None for unmatched paths
- handle_put routing: returns None for unmatched paths
- handle_post routing: returns None for unmatched paths
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.moderation import ModerationHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Moderation Objects
# ===========================================================================


class MockModerationConfig:
    """Mock moderation configuration."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "spam_threshold": 0.8,
            "max_links": 3,
            "auto_reject": False,
        }


class MockSpamModeration:
    """Mock SpamModerationIntegration."""

    def __init__(self, initialized: bool = True):
        self._initialized = initialized
        self.config = MockModerationConfig()
        self.statistics = {
            "total_checks": 100,
            "spam_detected": 5,
            "false_positives": 1,
        }

    async def initialize(self) -> None:
        self._initialized = True

    def update_config(self, data: dict[str, Any]) -> None:
        if "spam_threshold" in data:
            pass  # Accept updates silently


class MockQueueItem:
    """Mock review queue item."""

    def __init__(self, item_id: str = "item-001", content: str = "Test content"):
        self.id = item_id
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "status": "pending"}


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_moderation():
    """Create a mock moderation instance."""
    return MockSpamModeration()


@pytest.fixture
def handler(mock_moderation):
    """Create a ModerationHandler with mocked dependencies."""
    with patch(
        "aragora.server.handlers.moderation.get_spam_moderation",
        return_value=mock_moderation,
    ):
        h = ModerationHandler(server_context={})
        yield h


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestModerationHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, ModerationHandler)

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "moderation"

    def test_can_handle_config(self, handler):
        assert handler.can_handle("/api/moderation/config") is True

    def test_can_handle_stats(self, handler):
        assert handler.can_handle("/api/moderation/stats") is True

    def test_can_handle_queue(self, handler):
        assert handler.can_handle("/api/moderation/queue") is True

    def test_can_handle_items(self, handler):
        assert handler.can_handle("/api/moderation/items/item-001/approve") is True

    def test_cannot_handle_other_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_root(self, handler):
        # Needs to start with /api/moderation/
        assert handler.can_handle("/api/moderation") is False


# ===========================================================================
# Test GET /api/moderation/config
# ===========================================================================


class TestGetConfig:
    """Tests for the get config endpoint."""

    def test_get_config_success(self, handler, mock_moderation):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.get_spam_moderation",
            return_value=mock_moderation,
        ):
            result = handler._handle_get_config(mock_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["enabled"] is True
            assert "spam_threshold" in data

    def test_get_config_initializes_if_needed(self, mock_moderation):
        mock_moderation._initialized = False
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.get_spam_moderation",
            return_value=mock_moderation,
        ):
            h = ModerationHandler(server_context={})
            with patch(
                "aragora.server.handlers.moderation.run_async",
            ) as mock_run_async:
                result = h._handle_get_config(mock_handler)
                assert result.status_code == 200
                mock_run_async.assert_called_once()


# ===========================================================================
# Test GET /api/moderation/stats
# ===========================================================================


class TestGetStats:
    """Tests for the stats endpoint."""

    def test_get_stats_success(self, handler, mock_moderation):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.get_spam_moderation",
            return_value=mock_moderation,
        ):
            with patch(
                "aragora.server.handlers.moderation.review_queue_size",
                return_value=7,
            ):
                result = handler._handle_get_stats(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["total_checks"] == 100
                assert data["spam_detected"] == 5
                assert data["queue_size"] == 7


# ===========================================================================
# Test GET /api/moderation/queue
# ===========================================================================


class TestGetQueue:
    """Tests for the review queue endpoint."""

    def test_get_queue_success(self, handler):
        mock_handler = _make_mock_handler()
        mock_items = [MockQueueItem("item-001"), MockQueueItem("item-002")]

        with patch(
            "aragora.server.handlers.moderation.list_review_queue",
            return_value=mock_items,
        ):
            result = handler._handle_get_queue({}, mock_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert "items" in data
            assert len(data["items"]) == 2

    def test_get_queue_with_pagination(self, handler):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.list_review_queue",
            return_value=[],
        ) as mock_list:
            result = handler._handle_get_queue({"limit": "10", "offset": "5"}, mock_handler)
            assert result.status_code == 200
            mock_list.assert_called_once_with(limit=10, offset=5)

    def test_get_queue_empty(self, handler):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.list_review_queue",
            return_value=[],
        ):
            result = handler._handle_get_queue({}, mock_handler)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["items"] == []


# ===========================================================================
# Test PUT /api/moderation/config
# ===========================================================================


class TestUpdateConfig:
    """Tests for updating config."""

    def test_update_config_success(self, handler, mock_moderation):
        body = json.dumps({"spam_threshold": 0.9}).encode()
        mock_handler = _make_mock_handler("PUT", body)

        with patch(
            "aragora.server.handlers.moderation.get_spam_moderation",
            return_value=mock_moderation,
        ):
            with patch.object(
                handler, "read_json_body_validated", return_value=({"spam_threshold": 0.9}, None)
            ):
                result = handler._handle_update_config(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert "enabled" in data

    def test_update_config_invalid_body(self, handler):
        mock_handler = _make_mock_handler("PUT", b"not json")
        err_result = HandlerResult(
            status_code=400, content_type="application/json", body=b'{"error":"bad"}'
        )

        with patch.object(handler, "read_json_body_validated", return_value=(None, err_result)):
            result = handler._handle_update_config(mock_handler)
            assert result.status_code == 400


# ===========================================================================
# Test POST /api/moderation/items/{id}/approve
# ===========================================================================


class TestApproveItem:
    """Tests for approving queue items."""

    def test_approve_item_success(self, handler):
        mock_item = MockQueueItem("item-001")

        with patch(
            "aragora.server.handlers.moderation.pop_review_item",
            return_value=mock_item,
        ):
            result = handler._handle_queue_action("item-001", "approved")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "approved"
            assert data["item_id"] == "item-001"

    def test_approve_item_not_found(self, handler):
        with patch(
            "aragora.server.handlers.moderation.pop_review_item",
            return_value=None,
        ):
            result = handler._handle_queue_action("nonexistent", "approved")
            assert result.status_code == 404


# ===========================================================================
# Test POST /api/moderation/items/{id}/reject
# ===========================================================================


class TestRejectItem:
    """Tests for rejecting queue items."""

    def test_reject_item_success(self, handler):
        mock_item = MockQueueItem("item-002")

        with patch(
            "aragora.server.handlers.moderation.pop_review_item",
            return_value=mock_item,
        ):
            result = handler._handle_queue_action("item-002", "rejected")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "rejected"
            assert data["item_id"] == "item-002"

    def test_reject_item_not_found(self, handler):
        with patch(
            "aragora.server.handlers.moderation.pop_review_item",
            return_value=None,
        ):
            result = handler._handle_queue_action("nonexistent", "rejected")
            assert result.status_code == 404


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_config(self, handler, mock_moderation):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.get_spam_moderation",
            return_value=mock_moderation,
        ):
            result = handler.handle("/api/moderation/config", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_stats(self, handler, mock_moderation):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.get_spam_moderation",
            return_value=mock_moderation,
        ):
            with patch(
                "aragora.server.handlers.moderation.review_queue_size",
                return_value=0,
            ):
                result = handler.handle("/api/moderation/stats", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200

    def test_handle_queue(self, handler):
        mock_handler = _make_mock_handler()

        with patch(
            "aragora.server.handlers.moderation.list_review_queue",
            return_value=[],
        ):
            result = handler.handle("/api/moderation/queue", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/moderation/unknown", {}, mock_handler)
        assert result is None


# ===========================================================================
# Test handle_put() Routing
# ===========================================================================


class TestHandlePutRouting:
    """Tests for the handle_put() method routing."""

    def test_handle_put_config(self, handler, mock_moderation):
        body = json.dumps({"enabled": False}).encode()
        mock_handler = _make_mock_handler("PUT", body)

        with patch(
            "aragora.server.handlers.moderation.get_spam_moderation",
            return_value=mock_moderation,
        ):
            with patch.object(
                handler, "read_json_body_validated", return_value=({"enabled": False}, None)
            ):
                result = handler.handle_put("/api/moderation/config", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200

    def test_handle_put_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler("PUT")
        result = handler.handle_put("/api/moderation/unknown", {}, mock_handler)
        assert result is None


# ===========================================================================
# Test handle_post() Routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the handle_post() method routing."""

    def test_handle_post_approve(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_item = MockQueueItem("item-001")

        with patch(
            "aragora.server.handlers.moderation.pop_review_item",
            return_value=mock_item,
        ):
            result = handler.handle_post("/api/moderation/items/item-001/approve", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "approved"

    def test_handle_post_reject(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_item = MockQueueItem("item-002")

        with patch(
            "aragora.server.handlers.moderation.pop_review_item",
            return_value=mock_item,
        ):
            result = handler.handle_post("/api/moderation/items/item-002/reject", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "rejected"

    def test_handle_post_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler("POST")
        result = handler.handle_post("/api/moderation/unknown", {}, mock_handler)
        assert result is None

    def test_handle_post_extracts_item_id_from_path(self, handler):
        """Verify the item_id is correctly extracted from the approve/reject path."""
        mock_handler = _make_mock_handler("POST")
        mock_item = MockQueueItem("my-item-123")

        with patch(
            "aragora.server.handlers.moderation.pop_review_item",
            return_value=mock_item,
        ) as mock_pop:
            handler.handle_post("/api/moderation/items/my-item-123/approve", {}, mock_handler)
            mock_pop.assert_called_once_with("my-item-123")
