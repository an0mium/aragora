"""Tests for moderation handler.

Covers all routes and behavior of ModerationHandler:
- can_handle() path matching
- GET  /api/moderation/config  (retrieve moderation config)
- GET  /api/moderation/stats   (retrieve moderation stats + queue size)
- GET  /api/moderation/queue   (list review queue with pagination)
- PUT  /api/moderation/config  (update moderation config)
- POST /api/moderation/items/{id}/approve  (approve queued item)
- POST /api/moderation/items/{id}/reject   (reject queued item)
- handle() returns None for unknown paths
- handle_put() returns None for unknown paths
- handle_post() returns None for unknown paths
- _get_moderation() initialization and error handling
- Error handling via @handle_errors on PUT/POST
- Queue action with missing item (404)
- RBAC permission enforcement (via conftest auto-auth)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.moderation import ModerationHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler with optional JSON body."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {
            "User-Agent": "test-agent",
            "Content-Type": "application/json",
        }
        self.rfile = MagicMock()
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a ModerationHandler with empty context."""
    return ModerationHandler(server_context={})


@pytest.fixture
def http_handler():
    """Factory for creating mock HTTP handlers."""

    def _create(body: dict | None = None, method: str = "GET") -> MockHTTPHandler:
        return MockHTTPHandler(body=body, method=method)

    return _create


def _make_mock_moderation(initialized=True, config_dict=None, stats=None):
    """Create a mock SpamModerationIntegration."""
    mod = MagicMock()
    mod._initialized = initialized
    if config_dict is None:
        config_dict = {
            "enabled": True,
            "block_threshold": 0.9,
            "review_threshold": 0.7,
            "cache_enabled": True,
            "cache_ttl_seconds": 300,
            "cache_max_size": 1000,
            "fail_open": True,
            "log_all_checks": False,
        }
    mod.config.to_dict.return_value = config_dict
    mod.statistics = stats or {
        "checks": 10,
        "blocked": 2,
        "flagged": 3,
        "passed": 5,
        "cache_hits": 1,
        "errors": 0,
    }
    mod.update_config.return_value = mod.config
    return mod


def _make_queue_item(item_id="mod_abc123", content="test content"):
    """Create a mock ModerationQueueItem with to_dict()."""
    item = MagicMock()
    item.id = item_id
    item.content = content
    item.to_dict.return_value = {
        "id": item_id,
        "content": content,
        "content_hash": "abc123hash",
        "result": {"verdict": "suspicious"},
        "queued_at": "2026-01-01T00:00:00+00:00",
        "context": {},
    }
    return item


# ============================================================================
# can_handle
# ============================================================================


class TestCanHandle:
    """Tests for ModerationHandler.can_handle()."""

    def test_matches_moderation_config(self, handler):
        assert handler.can_handle("/api/moderation/config") is True

    def test_matches_moderation_stats(self, handler):
        assert handler.can_handle("/api/moderation/stats") is True

    def test_matches_moderation_queue(self, handler):
        assert handler.can_handle("/api/moderation/queue") is True

    def test_matches_moderation_items(self, handler):
        assert handler.can_handle("/api/moderation/items/abc/approve") is True

    def test_no_match_other_prefix(self, handler):
        assert handler.can_handle("/api/analytics/stats") is False

    def test_no_match_partial(self, handler):
        assert handler.can_handle("/api/moderatio") is False

    def test_no_match_exact_prefix_no_slash(self, handler):
        # "/api/moderation" without trailing slash does not start with "/api/moderation/"
        assert handler.can_handle("/api/moderation") is False

    def test_matches_subpath(self, handler):
        assert handler.can_handle("/api/moderation/any/deep/path") is True


# ============================================================================
# RESOURCE_TYPE
# ============================================================================


class TestResourceType:
    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "moderation"


# ============================================================================
# GET /api/moderation/config
# ============================================================================


class TestGetConfig:
    """Tests for GET /api/moderation/config."""

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_returns_config(self, mock_get, handler, http_handler):
        config_dict = {
            "enabled": True,
            "block_threshold": 0.9,
            "review_threshold": 0.7,
        }
        mock_mod = _make_mock_moderation(config_dict=config_dict)
        mock_get.return_value = mock_mod

        result = handler.handle("/api/moderation/config", {}, http_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["enabled"] is True
        assert body["block_threshold"] == 0.9

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_initializes_if_needed(self, mock_get, mock_run_async, handler, http_handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod

        result = handler.handle("/api/moderation/config", {}, http_handler())
        assert _status(result) == 200
        # run_async is called with moderation.initialize()
        mock_run_async.assert_called_once()

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_initialization_failure_still_returns(self, mock_get, mock_run_async, handler, http_handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod
        mock_run_async.side_effect = RuntimeError("Init failed")

        result = handler.handle("/api/moderation/config", {}, http_handler())
        assert _status(result) == 200

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_initialization_oserror(self, mock_get, mock_run_async, handler, http_handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod
        mock_run_async.side_effect = OSError("OS error")

        result = handler.handle("/api/moderation/config", {}, http_handler())
        assert _status(result) == 200

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_initialization_connection_error(self, mock_get, mock_run_async, handler, http_handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod
        mock_run_async.side_effect = ConnectionError("No connection")

        result = handler.handle("/api/moderation/config", {}, http_handler())
        assert _status(result) == 200

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_initialization_timeout_error(self, mock_get, mock_run_async, handler, http_handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod
        mock_run_async.side_effect = TimeoutError("Timeout")

        result = handler.handle("/api/moderation/config", {}, http_handler())
        assert _status(result) == 200


# ============================================================================
# GET /api/moderation/stats
# ============================================================================


class TestGetStats:
    """Tests for GET /api/moderation/stats."""

    @patch("aragora.server.handlers.moderation.review_queue_size")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_returns_stats(self, mock_get, mock_queue_size, handler, http_handler):
        mock_mod = _make_mock_moderation(stats={
            "checks": 100,
            "blocked": 10,
            "flagged": 5,
            "passed": 85,
            "cache_hits": 20,
            "errors": 1,
        })
        mock_get.return_value = mock_mod
        mock_queue_size.return_value = 7

        result = handler.handle("/api/moderation/stats", {}, http_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["checks"] == 100
        assert body["blocked"] == 10
        assert body["queue_size"] == 7

    @patch("aragora.server.handlers.moderation.review_queue_size")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_stats_with_zero_queue(self, mock_get, mock_queue_size, handler, http_handler):
        mock_mod = _make_mock_moderation()
        mock_get.return_value = mock_mod
        mock_queue_size.return_value = 0

        result = handler.handle("/api/moderation/stats", {}, http_handler())
        body = _body(result)
        assert body["queue_size"] == 0

    @patch("aragora.server.handlers.moderation.review_queue_size")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_stats_includes_all_fields(self, mock_get, mock_queue_size, handler, http_handler):
        stats = {
            "checks": 50,
            "blocked": 3,
            "flagged": 7,
            "passed": 40,
            "cache_hits": 15,
            "errors": 2,
        }
        mock_mod = _make_mock_moderation(stats=stats)
        mock_get.return_value = mock_mod
        mock_queue_size.return_value = 4

        result = handler.handle("/api/moderation/stats", {}, http_handler())
        body = _body(result)
        assert body["checks"] == 50
        assert body["blocked"] == 3
        assert body["flagged"] == 7
        assert body["passed"] == 40
        assert body["cache_hits"] == 15
        assert body["errors"] == 2
        assert body["queue_size"] == 4


# ============================================================================
# GET /api/moderation/queue
# ============================================================================


class TestGetQueue:
    """Tests for GET /api/moderation/queue."""

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_returns_empty_queue(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []

        result = handler.handle("/api/moderation/queue", {}, http_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["items"] == []

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_returns_queue_items(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        items = [_make_queue_item("item1"), _make_queue_item("item2")]
        mock_list.return_value = items

        result = handler.handle("/api/moderation/queue", {}, http_handler())
        assert _status(result) == 200
        body = _body(result)
        assert len(body["items"]) == 2
        assert body["items"][0]["id"] == "item1"
        assert body["items"][1]["id"] == "item2"

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_queue_default_pagination(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []

        handler.handle("/api/moderation/queue", {}, http_handler())
        # offset default=0 is clamped to min_val=1 by safe_query_int
        mock_list.assert_called_once_with(limit=50, offset=1)

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_queue_custom_limit(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []

        handler.handle("/api/moderation/queue", {"limit": "10"}, http_handler())
        # offset default=0 clamped to min_val=1
        mock_list.assert_called_once_with(limit=10, offset=1)

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_queue_custom_offset(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []

        handler.handle("/api/moderation/queue", {"offset": "20"}, http_handler())
        mock_list.assert_called_once_with(limit=50, offset=20)

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_queue_limit_clamped_to_max(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []

        # max_val=200 for limit; offset default=0 clamped to min_val=1
        handler.handle("/api/moderation/queue", {"limit": "999"}, http_handler())
        mock_list.assert_called_once_with(limit=200, offset=1)

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_queue_offset_clamped_to_max(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []

        # max_val=5000 for offset
        handler.handle("/api/moderation/queue", {"offset": "9999"}, http_handler())
        mock_list.assert_called_once_with(limit=50, offset=5000)


# ============================================================================
# PUT /api/moderation/config
# ============================================================================


class TestUpdateConfig:
    """Tests for PUT /api/moderation/config."""

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_update_config(self, mock_get, handler, http_handler):
        updated_dict = {
            "enabled": False,
            "block_threshold": 0.8,
            "review_threshold": 0.6,
        }
        mock_mod = _make_mock_moderation()
        mock_mod.config.to_dict.return_value = updated_dict
        mock_get.return_value = mock_mod

        h = http_handler(body={"enabled": False, "block_threshold": 0.8}, method="PUT")
        result = handler.handle_put("/api/moderation/config", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["enabled"] is False
        mock_mod.update_config.assert_called_once()

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_update_config_empty_body(self, mock_get, handler, http_handler):
        mock_mod = _make_mock_moderation()
        mock_get.return_value = mock_mod

        h = http_handler(body={}, method="PUT")
        result = handler.handle_put("/api/moderation/config", {}, h)
        assert _status(result) == 200
        mock_mod.update_config.assert_called_once_with({})

    def test_update_config_unknown_path(self, handler, http_handler):
        result = handler.handle_put("/api/moderation/unknown", {}, http_handler())
        assert result is None


# ============================================================================
# POST /api/moderation/items/{id}/approve
# ============================================================================


class TestApproveItem:
    """Tests for POST /api/moderation/items/{id}/approve."""

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_approve_existing_item(self, mock_pop, handler, http_handler):
        mock_item = _make_queue_item("item_123")
        mock_pop.return_value = mock_item

        result = handler.handle_post(
            "/api/moderation/items/item_123/approve", {}, http_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "approved"
        assert body["item_id"] == "item_123"
        mock_pop.assert_called_once_with("item_123")

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_approve_missing_item(self, mock_pop, handler, http_handler):
        mock_pop.return_value = None

        result = handler.handle_post(
            "/api/moderation/items/nonexistent/approve", {}, http_handler()
        )
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower() or "not found" in json.dumps(body).lower()


# ============================================================================
# POST /api/moderation/items/{id}/reject
# ============================================================================


class TestRejectItem:
    """Tests for POST /api/moderation/items/{id}/reject."""

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_reject_existing_item(self, mock_pop, handler, http_handler):
        mock_item = _make_queue_item("item_456")
        mock_pop.return_value = mock_item

        result = handler.handle_post(
            "/api/moderation/items/item_456/reject", {}, http_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "rejected"
        assert body["item_id"] == "item_456"

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_reject_missing_item(self, mock_pop, handler, http_handler):
        mock_pop.return_value = None

        result = handler.handle_post(
            "/api/moderation/items/no_such_item/reject", {}, http_handler()
        )
        assert _status(result) == 404


# ============================================================================
# handle() routing
# ============================================================================


class TestHandleRouting:
    """Tests for handle() route dispatching."""

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_handle_config_route(self, mock_get, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        result = handler.handle("/api/moderation/config", {}, http_handler())
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.moderation.review_queue_size")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_handle_stats_route(self, mock_get, mock_queue_size, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_queue_size.return_value = 0
        result = handler.handle("/api/moderation/stats", {}, http_handler())
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_handle_queue_route(self, mock_get, mock_list, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []
        result = handler.handle("/api/moderation/queue", {}, http_handler())
        assert result is not None
        assert _status(result) == 200

    def test_handle_unknown_path_returns_none(self, handler, http_handler):
        result = handler.handle("/api/moderation/unknown", {}, http_handler())
        assert result is None

    def test_handle_unrelated_path_returns_none(self, handler, http_handler):
        result = handler.handle("/api/debates/list", {}, http_handler())
        assert result is None

    def test_handle_items_path_not_in_get(self, handler, http_handler):
        """Items approve/reject are POST only, not GET."""
        result = handler.handle("/api/moderation/items/abc/approve", {}, http_handler())
        assert result is None


# ============================================================================
# handle_put() routing
# ============================================================================


class TestHandlePutRouting:
    """Tests for handle_put() route dispatching."""

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_put_config_route(self, mock_get, handler, http_handler):
        mock_get.return_value = _make_mock_moderation()
        h = http_handler(body={"enabled": True}, method="PUT")
        result = handler.handle_put("/api/moderation/config", {}, h)
        assert result is not None
        assert _status(result) == 200

    def test_put_unknown_route(self, handler, http_handler):
        result = handler.handle_put("/api/moderation/stats", {}, http_handler())
        assert result is None

    def test_put_queue_route_not_supported(self, handler, http_handler):
        result = handler.handle_put("/api/moderation/queue", {}, http_handler())
        assert result is None


# ============================================================================
# handle_post() routing
# ============================================================================


class TestHandlePostRouting:
    """Tests for handle_post() route dispatching."""

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_post_approve_route(self, mock_pop, handler, http_handler):
        mock_pop.return_value = _make_queue_item()
        result = handler.handle_post(
            "/api/moderation/items/abc/approve", {}, http_handler()
        )
        assert result is not None
        assert _status(result) == 200

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_post_reject_route(self, mock_pop, handler, http_handler):
        mock_pop.return_value = _make_queue_item()
        result = handler.handle_post(
            "/api/moderation/items/abc/reject", {}, http_handler()
        )
        assert result is not None
        assert _status(result) == 200

    def test_post_unknown_route(self, handler, http_handler):
        result = handler.handle_post("/api/moderation/config", {}, http_handler())
        assert result is None

    def test_post_items_without_action(self, handler, http_handler):
        result = handler.handle_post("/api/moderation/items/abc", {}, http_handler())
        assert result is None

    def test_post_items_unknown_action(self, handler, http_handler):
        result = handler.handle_post(
            "/api/moderation/items/abc/escalate", {}, http_handler()
        )
        assert result is None


# ============================================================================
# _get_moderation() internal method
# ============================================================================


class TestGetModeration:
    """Tests for the _get_moderation() helper method."""

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_returns_already_initialized(self, mock_get, handler):
        mock_mod = _make_mock_moderation(initialized=True)
        mock_get.return_value = mock_mod

        result = handler._get_moderation()
        assert result is mock_mod
        # Should NOT call initialize since already initialized
        mock_mod.initialize.assert_not_called()

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_calls_initialize_when_not_initialized(self, mock_get, mock_run_async, handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod

        result = handler._get_moderation()
        assert result is mock_mod
        mock_run_async.assert_called_once()

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_swallows_runtime_error(self, mock_get, mock_run_async, handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod
        mock_run_async.side_effect = RuntimeError("boom")

        result = handler._get_moderation()
        assert result is mock_mod

    @patch("aragora.server.handlers.moderation.run_async")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_swallows_timeout_error(self, mock_get, mock_run_async, handler):
        mock_mod = _make_mock_moderation(initialized=False)
        mock_get.return_value = mock_mod
        mock_run_async.side_effect = TimeoutError("timed out")

        result = handler._get_moderation()
        assert result is mock_mod


# ============================================================================
# Item ID extraction from path
# ============================================================================


class TestItemIdExtraction:
    """Tests for item ID extraction from approve/reject paths."""

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_extracts_simple_id(self, mock_pop, handler, http_handler):
        mock_pop.return_value = _make_queue_item("simple_id")
        handler.handle_post("/api/moderation/items/simple_id/approve", {}, http_handler())
        mock_pop.assert_called_once_with("simple_id")

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_extracts_uuid_style_id(self, mock_pop, handler, http_handler):
        item_id = "mod_abc123def456"
        mock_pop.return_value = _make_queue_item(item_id)
        handler.handle_post(f"/api/moderation/items/{item_id}/reject", {}, http_handler())
        mock_pop.assert_called_once_with(item_id)

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_extracts_id_with_hyphens(self, mock_pop, handler, http_handler):
        item_id = "mod-abc-123"
        mock_pop.return_value = _make_queue_item(item_id)
        handler.handle_post(f"/api/moderation/items/{item_id}/approve", {}, http_handler())
        mock_pop.assert_called_once_with(item_id)


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_config_returned_as_dict(self, mock_get, handler, http_handler):
        """Config to_dict values are passed through as JSON."""
        expected = {"enabled": True, "custom_key": "value"}
        mock_mod = _make_mock_moderation(config_dict=expected)
        mock_get.return_value = mock_mod

        result = handler.handle("/api/moderation/config", {}, http_handler())
        body = _body(result)
        assert body == expected

    @patch("aragora.server.handlers.moderation.review_queue_size")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_stats_merges_queue_size(self, mock_get, mock_queue_size, handler, http_handler):
        """queue_size is added to the stats dict."""
        mock_mod = _make_mock_moderation(stats={"checks": 1})
        mock_get.return_value = mock_mod
        mock_queue_size.return_value = 42

        result = handler.handle("/api/moderation/stats", {}, http_handler())
        body = _body(result)
        assert body["checks"] == 1
        assert body["queue_size"] == 42

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_queue_items_serialized_via_to_dict(self, mock_get, mock_list, handler, http_handler):
        """Each item in queue uses to_dict() for serialization."""
        item = _make_queue_item("x")
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = [item]

        handler.handle("/api/moderation/queue", {}, http_handler())
        item.to_dict.assert_called_once()

    @patch("aragora.server.handlers.moderation.list_review_queue")
    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_queue_invalid_limit_uses_default(self, mock_get, mock_list, handler, http_handler):
        """Non-integer limit falls back to default."""
        mock_get.return_value = _make_mock_moderation()
        mock_list.return_value = []

        handler.handle("/api/moderation/queue", {"limit": "not_a_number"}, http_handler())
        # safe_query_int returns default on invalid input for limit;
        # offset missing -> default=0 but clamped to min_val=1 by safe_query_int
        mock_list.assert_called_once_with(limit=50, offset=1)

    @patch("aragora.server.handlers.moderation.pop_review_item")
    def test_approve_and_reject_differ_in_status(self, mock_pop, handler, http_handler):
        """Approve returns 'approved' status, reject returns 'rejected' status."""
        mock_pop.return_value = _make_queue_item("x")

        approve_result = handler.handle_post(
            "/api/moderation/items/x/approve", {}, http_handler()
        )
        assert _body(approve_result)["status"] == "approved"

        mock_pop.return_value = _make_queue_item("y")
        reject_result = handler.handle_post(
            "/api/moderation/items/y/reject", {}, http_handler()
        )
        assert _body(reject_result)["status"] == "rejected"

    @patch("aragora.server.handlers.moderation.get_spam_moderation")
    def test_update_config_passes_data_to_moderation(self, mock_get, handler, http_handler):
        """update_config receives the parsed JSON body."""
        mock_mod = _make_mock_moderation()
        mock_get.return_value = mock_mod

        body_data = {"block_threshold": 0.85, "review_threshold": 0.65}
        h = http_handler(body=body_data, method="PUT")
        handler.handle_put("/api/moderation/config", {}, h)
        mock_mod.update_config.assert_called_once_with(body_data)
