"""Tests for StalenessOperationsMixin (aragora/server/handlers/knowledge_base/mound/staleness.py).

Covers all three handler methods on the mixin:
- GET  /api/knowledge/mound/stale                  - _handle_get_stale
- POST /api/knowledge/mound/revalidate/:id          - _handle_revalidate_node
- POST /api/knowledge/mound/schedule-revalidation   - _handle_schedule_revalidation

Each method is tested for:
- Success with valid inputs and default parameters
- Success with custom parameters
- Mound not available (503)
- Missing/invalid required parameters (400)
- Internal errors from mound operations (500)
- Edge cases, boundary values, and response structure
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.staleness import (
    StalenessOperationsMixin,
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
# Mock staleness types
# ---------------------------------------------------------------------------


class MockStalenessReason(str, Enum):
    AGE = "age"
    CONTRADICTION = "contradiction"
    NEW_EVIDENCE = "new_evidence"
    CONSENSUS_CHANGE = "consensus_change"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass
class MockStalenessCheck:
    """Mock StalenessCheck matching aragora.knowledge.mound.types.StalenessCheck."""

    node_id: str = "node-001"
    staleness_score: float = 0.75
    reasons: list = field(default_factory=lambda: [MockStalenessReason.AGE])
    last_checked_at: datetime = field(
        default_factory=lambda: datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    revalidation_recommended: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Lightweight mock HTTP handler for staleness tests."""

    command: str = "GET"
    path: str = ""
    headers: dict[str, str] = field(
        default_factory=lambda: {
            "Content-Length": "0",
        }
    )
    client_address: tuple = ("127.0.0.1", 12345)
    rfile: Any = field(default_factory=lambda: io.BytesIO(b""))

    @classmethod
    def get(cls) -> MockHTTPHandler:
        return cls(command="GET")

    @classmethod
    def post(cls, body: dict | None = None) -> MockHTTPHandler:
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            return cls(
                command="POST",
                headers={"Content-Length": str(len(raw))},
                rfile=io.BytesIO(raw),
            )
        return cls(command="POST", headers={"Content-Length": "0"})


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class StalenessTestHandler(StalenessOperationsMixin):
    """Concrete handler for testing the staleness mixin."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with staleness methods."""
    mound = MagicMock()
    mound.get_stale_knowledge = MagicMock(
        return_value=[MockStalenessCheck()]
    )
    mound.mark_validated = MagicMock(return_value=None)
    mound.schedule_revalidation = MagicMock(
        return_value=["node-001", "node-002"]
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a StalenessTestHandler with a mocked mound."""
    return StalenessTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a StalenessTestHandler with no mound (None)."""
    return StalenessTestHandler(mound=None)


# Patch target for _run_async
_RUN_ASYNC_PATCH = (
    "aragora.server.handlers.knowledge_base.mound.staleness._run_async"
)


# ============================================================================
# Tests: _handle_get_stale
# ============================================================================


class TestHandleGetStale:
    """Test _handle_get_stale (GET /api/knowledge/mound/stale)."""

    def test_success_returns_stale_items(self, handler, mock_mound):
        """Successful call returns stale items list."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert len(body["stale_items"]) == 1
        item = body["stale_items"][0]
        assert item["node_id"] == "node-001"
        assert item["staleness_score"] == 0.75
        assert item["revalidation_recommended"] is True

    def test_success_default_params(self, handler, mock_mound):
        """Default parameters are workspace_id=default, threshold=0.5, limit=50."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["workspace_id"] == "default"
        assert body["threshold"] == 0.5
        mock_mound.get_stale_knowledge.assert_called_once_with(
            threshold=0.5, limit=50, workspace_id="default"
        )

    def test_success_custom_workspace_id(self, handler, mock_mound):
        """Custom workspace_id is forwarded to the mound."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({"workspace_id": "ws-custom"})
        body = _body(result)
        assert body["workspace_id"] == "ws-custom"

    def test_success_custom_threshold(self, handler, mock_mound):
        """Custom threshold is forwarded to the mound."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({"threshold": "0.8"})
        body = _body(result)
        assert body["threshold"] == 0.8

    def test_success_custom_limit(self, handler, mock_mound):
        """Custom limit is forwarded to the mound."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_get_stale({"limit": "100"})
        mock_mound.get_stale_knowledge.assert_called_once_with(
            threshold=0.5, limit=100, workspace_id="default"
        )

    def test_success_all_custom_params(self, handler, mock_mound):
        """All custom params forwarded correctly."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_get_stale({
                "workspace_id": "ws-abc",
                "threshold": "0.9",
                "limit": "25",
            })
        mock_mound.get_stale_knowledge.assert_called_once_with(
            threshold=0.9, limit=25, workspace_id="ws-abc"
        )

    def test_threshold_clamped_to_min(self, handler, mock_mound):
        """Threshold below 0.0 is clamped to 0.0."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({"threshold": "-0.5"})
        body = _body(result)
        assert body["threshold"] == 0.0

    def test_threshold_clamped_to_max(self, handler, mock_mound):
        """Threshold above 1.0 is clamped to 1.0."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({"threshold": "2.0"})
        body = _body(result)
        assert body["threshold"] == 1.0

    def test_limit_clamped_to_min(self, handler, mock_mound):
        """Limit below 1 is clamped to 1."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_get_stale({"limit": "0"})
        mock_mound.get_stale_knowledge.assert_called_once_with(
            threshold=0.5, limit=1, workspace_id="default"
        )

    def test_limit_clamped_to_max(self, handler, mock_mound):
        """Limit above 200 is clamped to 200."""
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_get_stale({"limit": "999"})
        mock_mound.get_stale_knowledge.assert_called_once_with(
            threshold=0.5, limit=200, workspace_id="default"
        )

    def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = handler_no_mound._handle_get_stale({})
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("missing key")):
            result = handler._handle_get_stale({})
        assert _status(result) == 500

    def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad value")):
            result = handler._handle_get_stale({})
        assert _status(result) == 500

    def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk error")):
            result = handler._handle_get_stale({})
        assert _status(result) == 500

    def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("bad type")):
            result = handler._handle_get_stale({})
        assert _status(result) == 500

    def test_mound_raises_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("runtime err")):
            result = handler._handle_get_stale({})
        assert _status(result) == 500

    def test_empty_stale_items(self, handler, mock_mound):
        """Empty list of stale items returns total=0."""
        mock_mound.get_stale_knowledge = MagicMock(return_value=[])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["total"] == 0
        assert body["stale_items"] == []

    def test_multiple_stale_items(self, handler, mock_mound):
        """Multiple stale items are all included in response."""
        items = [
            MockStalenessCheck(node_id="n1", staleness_score=0.9),
            MockStalenessCheck(node_id="n2", staleness_score=0.6),
            MockStalenessCheck(node_id="n3", staleness_score=0.3),
        ]
        mock_mound.get_stale_knowledge = MagicMock(return_value=items)
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["total"] == 3
        ids = [item["node_id"] for item in body["stale_items"]]
        assert ids == ["n1", "n2", "n3"]

    def test_reasons_with_enum_values(self, handler, mock_mound):
        """Reasons that have .value attribute are serialized to their value."""
        item = MockStalenessCheck(
            reasons=[
                MockStalenessReason.AGE,
                MockStalenessReason.CONTRADICTION,
            ]
        )
        mock_mound.get_stale_knowledge = MagicMock(return_value=[item])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["stale_items"][0]["reasons"] == ["age", "contradiction"]

    def test_reasons_with_plain_strings(self, handler, mock_mound):
        """Reasons that are plain strings (no .value) are returned as-is."""
        item = MockStalenessCheck(reasons=["custom_reason", "another_reason"])
        mock_mound.get_stale_knowledge = MagicMock(return_value=[item])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["stale_items"][0]["reasons"] == ["custom_reason", "another_reason"]

    def test_last_checked_at_none(self, handler, mock_mound):
        """Item with last_checked_at=None serializes to null."""
        item = MockStalenessCheck(last_checked_at=None)
        mock_mound.get_stale_knowledge = MagicMock(return_value=[item])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["stale_items"][0]["last_checked_at"] is None

    def test_last_checked_at_isoformat(self, handler, mock_mound):
        """Item with last_checked_at datetime is serialized to ISO format."""
        dt = datetime(2026, 2, 15, 8, 30, 0, tzinfo=timezone.utc)
        item = MockStalenessCheck(last_checked_at=dt)
        mock_mound.get_stale_knowledge = MagicMock(return_value=[item])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["stale_items"][0]["last_checked_at"] == "2026-02-15T08:30:00+00:00"

    def test_revalidation_recommended_false(self, handler, mock_mound):
        """Item with revalidation_recommended=False is serialized."""
        item = MockStalenessCheck(revalidation_recommended=False)
        mock_mound.get_stale_knowledge = MagicMock(return_value=[item])
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({})
        body = _body(result)
        assert body["stale_items"][0]["revalidation_recommended"] is False

    def test_workspace_id_truncated_to_max_length(self, handler, mock_mound):
        """Workspace ID longer than 100 chars is truncated."""
        long_ws = "x" * 200
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_get_stale({"workspace_id": long_ws})
        body = _body(result)
        assert len(body["workspace_id"]) == 100


# ============================================================================
# Tests: _handle_revalidate_node
# ============================================================================


class TestHandleRevalidateNode:
    """Test _handle_revalidate_node (POST /api/knowledge/mound/revalidate/:id)."""

    def test_success_with_body(self, handler, mock_mound):
        """Successful revalidation with validator and confidence in body."""
        http_handler = MockHTTPHandler.post(
            {"validator": "human-review", "confidence": 0.95}
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_revalidate_node("node-42", http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["node_id"] == "node-42"
        assert body["validated"] is True
        assert body["validator"] == "human-review"
        assert body["new_confidence"] == 0.95
        assert "successfully" in body["message"].lower()

    def test_success_empty_body_defaults(self, handler, mock_mound):
        """Empty body defaults to validator='api' and confidence=None."""
        http_handler = MockHTTPHandler.post()
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_revalidate_node("node-42", http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["validator"] == "api"
        assert body["new_confidence"] is None

    def test_success_calls_mark_validated(self, handler, mock_mound):
        """mark_validated is called with correct arguments."""
        http_handler = MockHTTPHandler.post(
            {"validator": "test-user", "confidence": 0.8}
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_revalidate_node("node-99", http_handler)
        mock_mound.mark_validated.assert_called_once_with("node-99", "test-user", 0.8)

    def test_success_with_no_confidence(self, handler, mock_mound):
        """Body with validator only, no confidence."""
        http_handler = MockHTTPHandler.post({"validator": "auto-check"})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_revalidate_node("node-10", http_handler)
        body = _body(result)
        assert body["validator"] == "auto-check"
        assert body["new_confidence"] is None

    def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http_handler = MockHTTPHandler.post({})
        result = handler_no_mound._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_invalid_json_body_returns_400(self, handler):
        """Invalid JSON in request body returns 400."""
        http_handler = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "11"},
            rfile=io.BytesIO(b"not-json!!!"),
        )
        result = handler._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower()

    def test_invalid_content_length_returns_400(self, handler):
        """Non-numeric Content-Length returns 400."""
        http_handler = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "abc"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 400

    def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        http_handler = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("not found")):
            result = handler._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 500

    def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        http_handler = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 500

    def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        http_handler = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 500

    def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        http_handler = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=TypeError("type")):
            result = handler._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 500

    def test_mound_raises_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        http_handler = MockHTTPHandler.post({})
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("rt")):
            result = handler._handle_revalidate_node("node-1", http_handler)
        assert _status(result) == 500

    def test_different_node_ids(self, handler, mock_mound):
        """Various node IDs are forwarded correctly."""
        for nid in ["a", "node-with-dashes", "123", "uuid-like-abcd1234"]:
            mock_mound.mark_validated.reset_mock()
            http_handler = MockHTTPHandler.post({})
            with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
                result = handler._handle_revalidate_node(nid, http_handler)
            body = _body(result)
            assert body["node_id"] == nid


# ============================================================================
# Tests: _handle_schedule_revalidation
# ============================================================================


class TestHandleScheduleRevalidation:
    """Test _handle_schedule_revalidation (POST /api/knowledge/mound/schedule-revalidation)."""

    def test_success_returns_202(self, handler, mock_mound):
        """Successful scheduling returns 202 Accepted."""
        http_handler = MockHTTPHandler.post(
            {"node_ids": ["n1", "n2"], "priority": "medium"}
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 202
        body = _body(result)
        assert body["scheduled"] == ["node-001", "node-002"]
        assert body["priority"] == "medium"
        assert body["count"] == 2
        assert "scheduled" in body["message"].lower()

    def test_success_default_priority_low(self, handler, mock_mound):
        """Default priority is 'low'."""
        http_handler = MockHTTPHandler.post({"node_ids": ["n1"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_schedule_revalidation(http_handler)
        body = _body(result)
        assert body["priority"] == "low"

    def test_success_priority_high(self, handler, mock_mound):
        """Priority 'high' is accepted."""
        http_handler = MockHTTPHandler.post(
            {"node_ids": ["n1"], "priority": "high"}
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_schedule_revalidation(http_handler)
        body = _body(result)
        assert body["priority"] == "high"

    def test_success_priority_low(self, handler, mock_mound):
        """Priority 'low' is accepted."""
        http_handler = MockHTTPHandler.post(
            {"node_ids": ["n1"], "priority": "low"}
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_schedule_revalidation(http_handler)
        body = _body(result)
        assert body["priority"] == "low"

    def test_success_calls_schedule_revalidation(self, handler, mock_mound):
        """schedule_revalidation is called with correct node_ids and priority."""
        http_handler = MockHTTPHandler.post(
            {"node_ids": ["a", "b", "c"], "priority": "high"}
        )
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            handler._handle_schedule_revalidation(http_handler)
        mock_mound.schedule_revalidation.assert_called_once_with(
            ["a", "b", "c"], "high"
        )

    def test_empty_body_returns_400(self, handler):
        """Empty body (Content-Length=0) returns 400 'Request body required'."""
        http_handler = MockHTTPHandler.post()
        result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "body required" in body["error"].lower() or "request body" in body["error"].lower()

    def test_empty_node_ids_returns_400(self, handler):
        """Empty node_ids list returns 400."""
        http_handler = MockHTTPHandler.post({"node_ids": []})
        result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "node_ids" in body["error"].lower()

    def test_missing_node_ids_returns_400(self, handler):
        """Missing node_ids key returns 400."""
        http_handler = MockHTTPHandler.post({"priority": "high"})
        result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "node_ids" in body["error"].lower()

    def test_invalid_priority_returns_400(self, handler):
        """Invalid priority value returns 400."""
        http_handler = MockHTTPHandler.post(
            {"node_ids": ["n1"], "priority": "urgent"}
        )
        result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "priority" in body["error"].lower()

    def test_invalid_priority_values(self, handler):
        """Various invalid priority values return 400."""
        for bad_priority in ["critical", "none", "MEDIUM", "1", ""]:
            http_handler = MockHTTPHandler.post(
                {"node_ids": ["n1"], "priority": bad_priority}
            )
            result = handler._handle_schedule_revalidation(http_handler)
            assert _status(result) == 400, f"Expected 400 for priority={bad_priority!r}"

    def test_invalid_json_body_returns_400(self, handler):
        """Invalid JSON in request body returns 400."""
        http_handler = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "5"},
            rfile=io.BytesIO(b"{bad}"),
        )
        result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 400

    def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        http_handler = MockHTTPHandler.post({"node_ids": ["n1"]})
        result = handler_no_mound._handle_schedule_revalidation(http_handler)
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        http_handler = MockHTTPHandler.post({"node_ids": ["n1"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=KeyError("bad")):
            result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 500

    def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        http_handler = MockHTTPHandler.post({"node_ids": ["n1"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=ValueError("bad")):
            result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 500

    def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        http_handler = MockHTTPHandler.post({"node_ids": ["n1"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=OSError("disk")):
            result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 500

    def test_mound_raises_runtime_error_returns_500(self, handler, mock_mound):
        """RuntimeError from mound returns 500."""
        http_handler = MockHTTPHandler.post({"node_ids": ["n1"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=RuntimeError("rt")):
            result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 500

    def test_scheduled_count_matches_returned_list(self, handler, mock_mound):
        """Count field matches the length of the scheduled list."""
        mock_mound.schedule_revalidation = MagicMock(
            return_value=["a", "b", "c", "d", "e"]
        )
        http_handler = MockHTTPHandler.post({"node_ids": ["a", "b", "c", "d", "e"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_schedule_revalidation(http_handler)
        body = _body(result)
        assert body["count"] == 5
        assert len(body["scheduled"]) == 5

    def test_message_includes_count(self, handler, mock_mound):
        """Message includes the number of scheduled nodes."""
        mock_mound.schedule_revalidation = MagicMock(return_value=["x", "y", "z"])
        http_handler = MockHTTPHandler.post({"node_ids": ["x", "y", "z"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_schedule_revalidation(http_handler)
        body = _body(result)
        assert "3" in body["message"]

    def test_single_node_scheduling(self, handler, mock_mound):
        """Scheduling a single node works."""
        mock_mound.schedule_revalidation = MagicMock(return_value=["solo-node"])
        http_handler = MockHTTPHandler.post({"node_ids": ["solo-node"]})
        with patch(_RUN_ASYNC_PATCH, side_effect=lambda coro: coro):
            result = handler._handle_schedule_revalidation(http_handler)
        body = _body(result)
        assert body["count"] == 1
        assert body["scheduled"] == ["solo-node"]

    def test_invalid_content_length_returns_400(self, handler):
        """Non-numeric Content-Length returns 400."""
        http_handler = MockHTTPHandler(
            command="POST",
            headers={"Content-Length": "NaN"},
            rfile=io.BytesIO(b""),
        )
        result = handler._handle_schedule_revalidation(http_handler)
        assert _status(result) == 400
