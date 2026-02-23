"""Tests for Gmail Q&A query handler.

Tests the GmailQueryHandler covering:
- POST /api/v1/gmail/query - Text Q&A over inbox
- POST /api/v1/gmail/query/voice - Voice input Q&A
- GET /api/v1/gmail/query/stream - Streaming Q&A response
- GET /api/v1/gmail/inbox/priority - Get prioritized inbox
- POST /api/v1/gmail/inbox/feedback - Record interaction feedback
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.gmail_query import (
    GmailQueryHandler,
    QueryResponse,
)
from aragora.storage.gmail_token_store import GmailUserState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for tests."""

    path: str = "/"
    method: str = "GET"
    body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    command: str = "GET"

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Length": "0", "Content-Type": "application/json"}
        self.client_address = ("127.0.0.1", 12345)
        self.rfile = MagicMock()
        if self.body:
            body_bytes = json.dumps(self.body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"


def _make_mock_email(
    msg_id: str = "msg1",
    subject: str = "Test Subject",
    from_address: str = "sender@example.com",
    body_text: str = "Email body content here.",
    snippet: str = "Snippet...",
    date: datetime | None = None,
    labels: list[str] | None = None,
    is_read: bool = False,
    is_starred: bool = False,
    thread_id: str = "thread1",
) -> MagicMock:
    """Create a mock email message."""
    msg = MagicMock()
    msg.id = msg_id
    msg.subject = subject
    msg.from_address = from_address
    msg.body_text = body_text
    msg.snippet = snippet
    msg.date = date or datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc)
    msg.labels = labels or ["INBOX"]
    msg.is_read = is_read
    msg.is_starred = is_starred
    msg.thread_id = thread_id
    return msg


def _make_mock_search_result(result_id: str = "gmail-msg1") -> MagicMock:
    """Create a mock search result."""
    result = MagicMock()
    result.id = result_id
    return result


def _make_mock_score(score: float = 0.8, reason: str = "Important") -> MagicMock:
    """Create a mock priority score."""
    s = MagicMock()
    s.score = score
    s.reason = reason
    return s


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_GET_USER_STATE = "aragora.server.handlers.features.gmail_query.get_user_state"
_GMAIL_CONNECTOR = "aragora.connectors.enterprise.communication.gmail.GmailConnector"
_EMAIL_PRIORITY_ANALYZER = "aragora.analysis.email_priority.EmailPriorityAnalyzer"
_EMAIL_FEEDBACK_LEARNER = "aragora.analysis.email_priority.EmailFeedbackLearner"
_RATE_LIMITER = "aragora.server.handlers.features.gmail_query._gmail_query_limiter"


def _patch_user_state(state):
    """Patch get_user_state as a sync function returning the state."""
    return patch(_GET_USER_STATE, new=MagicMock(return_value=state))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a GmailQueryHandler with minimal context."""
    return GmailQueryHandler(server_context={})


@pytest.fixture
def mock_http():
    """Create a basic mock HTTP handler (no body)."""
    return MockHTTPHandler()


@pytest.fixture
def mock_http_with_body():
    """Factory for mock HTTP handler with body."""

    def _create(body: dict[str, Any]) -> MockHTTPHandler:
        return MockHTTPHandler(body=body)

    return _create


@pytest.fixture
def gmail_state():
    """Create a GmailUserState with valid tokens."""
    return GmailUserState(
        user_id="default",
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        token_expiry=datetime(2099, 1, 1, tzinfo=timezone.utc),
        email_address="user@example.com",
    )


@pytest.fixture
def gmail_state_no_refresh():
    """Create a GmailUserState without a refresh token."""
    return GmailUserState(
        user_id="default",
        access_token="test-access-token",
        refresh_token="",
    )


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter before each test."""
    with patch(_RATE_LIMITER) as mock_limiter:
        mock_limiter.is_allowed.return_value = True
        yield mock_limiter


# ---------------------------------------------------------------------------
# QueryResponse unit tests
# ---------------------------------------------------------------------------


class TestQueryResponse:
    """Tests for the QueryResponse dataclass."""

    def test_to_dict_defaults(self):
        qr = QueryResponse(answer="Hello")
        d = qr.to_dict()
        assert d["answer"] == "Hello"
        assert d["sources"] == []
        assert d["confidence"] == 0.0
        assert d["query"] == ""

    def test_to_dict_with_all_fields(self):
        qr = QueryResponse(
            answer="Found it",
            sources=[{"id": "msg1", "subject": "Test"}],
            confidence=0.85,
            query="What about test?",
        )
        d = qr.to_dict()
        assert d["answer"] == "Found it"
        assert len(d["sources"]) == 1
        assert d["confidence"] == 0.85
        assert d["query"] == "What about test?"

    def test_to_dict_empty_answer(self):
        qr = QueryResponse(answer="")
        d = qr.to_dict()
        assert d["answer"] == ""

    def test_to_dict_multiple_sources(self):
        sources = [{"id": f"msg{i}"} for i in range(5)]
        qr = QueryResponse(answer="x", sources=sources)
        d = qr.to_dict()
        assert len(d["sources"]) == 5


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_query_path(self, handler):
        assert handler.can_handle("/api/v1/gmail/query") is True

    def test_query_voice_path(self, handler):
        assert handler.can_handle("/api/v1/gmail/query/voice") is True

    def test_query_stream_path(self, handler):
        assert handler.can_handle("/api/v1/gmail/query/stream") is True

    def test_inbox_priority_path(self, handler):
        assert handler.can_handle("/api/v1/gmail/inbox/priority") is True

    def test_inbox_feedback_path(self, handler):
        assert handler.can_handle("/api/v1/gmail/inbox/feedback") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_gmail_root(self, handler):
        assert handler.can_handle("/api/v1/gmail") is False

    def test_different_api_version(self, handler):
        assert handler.can_handle("/api/v2/gmail/query") is False

    def test_partial_query(self, handler):
        assert handler.can_handle("/api/v1/gmail/quer") is False

    def test_can_handle_with_method(self, handler):
        assert handler.can_handle("/api/v1/gmail/query", "POST") is True
        assert handler.can_handle("/api/v1/gmail/inbox/priority", "GET") is True

    def test_inbox_without_subpath(self, handler):
        # /api/v1/gmail/inbox/ starts with the inbox prefix
        assert handler.can_handle("/api/v1/gmail/inbox/something") is True


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for handler initialization."""

    def test_init_with_server_context(self):
        h = GmailQueryHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_ctx(self):
        h = GmailQueryHandler(ctx={"other": 1})
        assert h.ctx == {"other": 1}

    def test_init_empty(self):
        h = GmailQueryHandler()
        assert h.ctx == {}

    def test_init_server_context_overrides_ctx(self):
        h = GmailQueryHandler(ctx={"a": 1}, server_context={"b": 2})
        assert h.ctx == {"b": 2}

    def test_routes_defined(self):
        h = GmailQueryHandler(server_context={})
        assert "/api/v1/gmail/query" in h.ROUTES
        assert "/api/v1/gmail/query/voice" in h.ROUTES
        assert "/api/v1/gmail/query/stream" in h.ROUTES
        assert "/api/v1/gmail/inbox/priority" in h.ROUTES
        assert "/api/v1/gmail/inbox/feedback" in h.ROUTES


# ---------------------------------------------------------------------------
# GET /api/v1/gmail/inbox/priority
# ---------------------------------------------------------------------------


class TestGetPriorityInbox:
    """Tests for GET /api/v1/gmail/inbox/priority."""

    @pytest.mark.asyncio
    async def test_priority_inbox_success(self, handler, mock_http, gmail_state):
        mock_msgs = [_make_mock_email(msg_id=f"msg{i}") for i in range(3)]
        mock_scores = [_make_mock_score(score=0.9 - i * 0.1) for i in range(3)]

        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=(["msg0", "msg1", "msg2"], None))
        mock_connector.get_messages = AsyncMock(return_value=mock_msgs)

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=mock_scores)

        with _patch_user_state(gmail_state):
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 3
        assert len(body["emails"]) == 3
        assert body["user_id"] == "default"

    @pytest.mark.asyncio
    async def test_priority_inbox_with_user_id(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=([], None))
        mock_connector.get_messages = AsyncMock(return_value=[])

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=[])

        with _patch_user_state(gmail_state) as mock_get:
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    result = await handler.handle(
                        "/api/v1/gmail/inbox/priority",
                        {"user_id": "user42"},
                        mock_http,
                    )
            mock_get.assert_called_once_with("user42")

        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "user42"

    @pytest.mark.asyncio
    async def test_priority_inbox_default_user_id(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=([], None))
        mock_connector.get_messages = AsyncMock(return_value=[])

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=[])

        with _patch_user_state(gmail_state) as mock_get:
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)
            mock_get.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_priority_inbox_no_state(self, handler, mock_http):
        with _patch_user_state(None):
            result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)
        assert _status(result) == 401
        assert "authenticate" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_priority_inbox_no_refresh_token(
        self, handler, mock_http, gmail_state_no_refresh
    ):
        with _patch_user_state(gmail_state_no_refresh):
            result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_priority_inbox_connection_error(self, handler, mock_http, gmail_state):
        with _patch_user_state(gmail_state):
            with patch(
                _GMAIL_CONNECTOR,
                side_effect=ConnectionError("API down"),
            ):
                result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)
        assert _status(result) == 500
        assert "priority inbox" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_priority_inbox_timeout_error(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(side_effect=TimeoutError("timeout"))
        with _patch_user_state(gmail_state):
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_priority_inbox_sorted_by_score(self, handler, mock_http, gmail_state):
        mock_msgs = [
            _make_mock_email(msg_id="low"),
            _make_mock_email(msg_id="high"),
            _make_mock_email(msg_id="mid"),
        ]
        mock_scores = [
            _make_mock_score(score=0.2),
            _make_mock_score(score=0.9),
            _make_mock_score(score=0.5),
        ]

        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=(["low", "high", "mid"], None))
        mock_connector.get_messages = AsyncMock(return_value=mock_msgs)

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=mock_scores)

        with _patch_user_state(gmail_state):
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)

        body = _body(result)
        scores = [e["priority_score"] for e in body["emails"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_priority_inbox_email_fields(self, handler, mock_http, gmail_state):
        msg = _make_mock_email(
            msg_id="msg1",
            subject="Important",
            from_address="boss@work.com",
            snippet="Please review...",
            labels=["INBOX", "IMPORTANT"],
            is_read=False,
            is_starred=True,
            thread_id="t1",
        )
        score = _make_mock_score(score=0.95, reason="From boss")

        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=(["msg1"], None))
        mock_connector.get_messages = AsyncMock(return_value=[msg])

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=[score])

        with _patch_user_state(gmail_state):
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)

        email = _body(result)["emails"][0]
        assert email["id"] == "msg1"
        assert email["subject"] == "Important"
        assert email["from"] == "boss@work.com"
        assert email["snippet"] == "Please review..."
        assert email["labels"] == ["INBOX", "IMPORTANT"]
        assert email["is_read"] is False
        assert email["is_starred"] is True
        assert email["thread_id"] == "t1"
        assert email["priority_score"] == 0.95
        assert email["priority_reason"] == "From boss"
        assert "mail.google.com" in email["url"]

    @pytest.mark.asyncio
    async def test_priority_inbox_empty_results(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=([], None))
        mock_connector.get_messages = AsyncMock(return_value=[])

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=[])

        with _patch_user_state(gmail_state):
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["emails"] == []


# ---------------------------------------------------------------------------
# GET /api/v1/gmail/query/stream
# ---------------------------------------------------------------------------


class TestStreamQuery:
    """Tests for GET /api/v1/gmail/query/stream."""

    @pytest.mark.asyncio
    async def test_stream_query_success(self, handler, mock_http, gmail_state):
        """Streaming falls back to regular query response."""
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_handle_query",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    status_code=200,
                    body=json.dumps({"answer": "test"}).encode(),
                ),
            ) as mock_query:
                result = await handler.handle(
                    "/api/v1/gmail/query/stream", {"q": "test question"}, mock_http
                )
                mock_query.assert_called_once_with("default", {"question": "test question"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_stream_query_empty_q(self, handler, mock_http, gmail_state):
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_handle_query",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    status_code=400,
                    body=json.dumps({"error": "Question is required"}).encode(),
                ),
            ):
                result = await handler.handle("/api/v1/gmail/query/stream", {}, mock_http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_stream_query_no_state(self, handler, mock_http):
        with _patch_user_state(None):
            result = await handler.handle("/api/v1/gmail/query/stream", {}, mock_http)
        assert _status(result) == 401


# ---------------------------------------------------------------------------
# GET routing - 404 for unknown paths
# ---------------------------------------------------------------------------


class TestGetRouting:
    """Tests for GET routing edge cases."""

    @pytest.mark.asyncio
    async def test_unknown_get_path(self, handler, mock_http, gmail_state):
        with _patch_user_state(gmail_state):
            result = await handler.handle("/api/v1/gmail/query/unknown", {}, mock_http)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/query - Text Q&A
# ---------------------------------------------------------------------------


class TestTextQuery:
    """Tests for POST /api/v1/gmail/query."""

    @pytest.mark.asyncio
    async def test_query_success(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "What about project X?"})
        mock_response = QueryResponse(
            answer="Project X is on track.",
            sources=[{"id": "msg1", "subject": "Project X update"}],
            confidence=0.8,
            query="What about project X?",
        )

        with _patch_user_state(gmail_state):
            with patch.object(
                handler, "_run_query", new_callable=AsyncMock, return_value=mock_response
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["answer"] == "Project X is on track."
        assert body["confidence"] == 0.8
        assert body["query"] == "What about project X?"
        assert len(body["sources"]) == 1

    @pytest.mark.asyncio
    async def test_query_with_q_field(self, handler, mock_http_with_body, gmail_state):
        """Test that 'q' field is also accepted as question."""
        http = mock_http_with_body({"q": "Find emails from Alice"})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                return_value=QueryResponse(answer="Found them.", query="Find emails from Alice"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_query_missing_question(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 400
        assert "question" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_query_empty_question(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": ""})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_query_no_state(self, handler, mock_http_with_body):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(None):
            result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 401
        assert "authenticate" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_query_no_refresh_token(
        self, handler, mock_http_with_body, gmail_state_no_refresh
    ):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(gmail_state_no_refresh):
            result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_query_connection_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                side_effect=ConnectionError("API down"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_query_timeout_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                side_effect=TimeoutError("timeout"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_query_value_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                side_effect=ValueError("bad"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_query_os_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                side_effect=OSError("network"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_query_key_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                side_effect=KeyError("missing"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_query_attribute_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?"})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                side_effect=AttributeError("bad attr"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_query_user_id_from_body(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?", "user_id": "custom_user"})
        with _patch_user_state(gmail_state) as mock_get:
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                return_value=QueryResponse(answer="ok"),
            ):
                await handler.handle_post("/api/v1/gmail/query", {}, http)
            mock_get.assert_called_once_with("custom_user")

    @pytest.mark.asyncio
    async def test_query_with_limit(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"question": "Test?", "limit": 5})
        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_run_query",
                new_callable=AsyncMock,
                return_value=QueryResponse(answer="ok"),
            ) as mock_run:
                await handler.handle_post("/api/v1/gmail/query", {}, http)
                # limit=5 is passed via _handle_query to _run_query
                assert mock_run.call_args[0][3] == 5

    @pytest.mark.asyncio
    async def test_query_null_body_treated_as_empty(self, handler, mock_http, gmail_state):
        """If read_json_body returns None, body is treated as empty dict."""
        with _patch_user_state(gmail_state):
            with patch.object(handler, "read_json_body", return_value=None):
                result = await handler.handle_post("/api/v1/gmail/query", {}, mock_http)
        # No question in empty body -> 400
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/query/voice - Voice Q&A
# ---------------------------------------------------------------------------


class TestVoiceQuery:
    """Tests for POST /api/v1/gmail/query/voice."""

    @pytest.mark.asyncio
    async def test_voice_query_with_audio_data(self, handler, mock_http_with_body, gmail_state):
        audio_b64 = base64.b64encode(b"fake-audio-data").decode()
        http = mock_http_with_body({"audio": audio_b64})

        mock_response = QueryResponse(
            answer="Found results.",
            sources=[],
            confidence=0.8,
            query="transcribed question",
        )

        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_transcribe",
                new_callable=AsyncMock,
                return_value="transcribed question",
            ):
                with patch.object(
                    handler,
                    "_run_query",
                    new_callable=AsyncMock,
                    return_value=mock_response,
                ):
                    result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["answer"] == "Found results."
        assert body["transcription"] == "transcribed question"

    @pytest.mark.asyncio
    async def test_voice_query_with_audio_url(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"audio_url": "https://example.com/audio.webm"})

        mock_session = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.content = b"fetched-audio"
        mock_session.get = AsyncMock(return_value=mock_resp)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=mock_cm)

        with _patch_user_state(gmail_state):
            with patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ):
                with patch.object(
                    handler,
                    "_transcribe",
                    new_callable=AsyncMock,
                    return_value="fetched question",
                ):
                    with patch.object(
                        handler,
                        "_run_query",
                        new_callable=AsyncMock,
                        return_value=QueryResponse(answer="ok"),
                    ):
                        result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_voice_query_no_audio(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)
        assert _status(result) == 400
        assert "audio" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_voice_query_no_state(self, handler, mock_http_with_body):
        http = mock_http_with_body({"audio": base64.b64encode(b"data").decode()})
        with _patch_user_state(None):
            result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_voice_query_no_refresh_token(
        self, handler, mock_http_with_body, gmail_state_no_refresh
    ):
        http = mock_http_with_body({"audio": base64.b64encode(b"data").decode()})
        with _patch_user_state(gmail_state_no_refresh):
            result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_voice_query_transcription_fails(self, handler, mock_http_with_body, gmail_state):
        audio_b64 = base64.b64encode(b"audio").decode()
        http = mock_http_with_body({"audio": audio_b64})

        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_transcribe",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)

        assert _status(result) == 400
        assert "transcribe" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_voice_query_connection_error(self, handler, mock_http_with_body, gmail_state):
        audio_b64 = base64.b64encode(b"audio").decode()
        http = mock_http_with_body({"audio": audio_b64})

        with _patch_user_state(gmail_state):
            with patch.object(
                handler,
                "_transcribe",
                new_callable=AsyncMock,
                side_effect=ConnectionError("fail"),
            ):
                result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_voice_query_value_error(self, handler, mock_http_with_body, gmail_state):
        # Invalid base64 triggers ValueError
        http = mock_http_with_body({"audio": "not-valid-base64!!!"})

        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/query/voice", {}, http)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/inbox/feedback
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    """Tests for POST /api/v1/gmail/inbox/feedback."""

    @pytest.mark.asyncio
    async def test_feedback_success(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body(
            {
                "email_id": "msg1",
                "action": "opened",
                "from_address": "sender@example.com",
                "subject": "Test",
                "labels": ["INBOX"],
            }
        )

        mock_learner = MagicMock()
        mock_learner.record_interaction = AsyncMock(return_value=True)

        with _patch_user_state(gmail_state):
            with patch(_EMAIL_FEEDBACK_LEARNER, return_value=mock_learner):
                result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["email_id"] == "msg1"
        assert body["action"] == "opened"

    @pytest.mark.asyncio
    async def test_feedback_all_valid_actions(self, handler, mock_http_with_body, gmail_state):
        valid_actions = ["opened", "replied", "starred", "archived", "deleted", "snoozed"]
        for action in valid_actions:
            http = mock_http_with_body({"email_id": "msg1", "action": action})
            mock_learner = MagicMock()
            mock_learner.record_interaction = AsyncMock(return_value=True)
            with _patch_user_state(gmail_state):
                with patch(_EMAIL_FEEDBACK_LEARNER, return_value=mock_learner):
                    result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
            assert _status(result) == 200, f"Action '{action}' should succeed"

    @pytest.mark.asyncio
    async def test_feedback_missing_email_id(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"action": "opened"})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 400
        assert "email_id" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_feedback_missing_action(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"email_id": "msg1"})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_feedback_missing_both(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_feedback_invalid_action(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"email_id": "msg1", "action": "invalid_action"})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 400
        assert "invalid action" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_feedback_learner_import_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"email_id": "msg1", "action": "opened"})
        with _patch_user_state(gmail_state):
            with patch(
                _EMAIL_FEEDBACK_LEARNER,
                side_effect=ImportError("not available"),
            ):
                result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_feedback_learner_value_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"email_id": "msg1", "action": "replied"})
        mock_learner = MagicMock()
        mock_learner.record_interaction = AsyncMock(side_effect=ValueError("bad data"))
        with _patch_user_state(gmail_state):
            with patch(_EMAIL_FEEDBACK_LEARNER, return_value=mock_learner):
                result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_feedback_learner_os_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"email_id": "msg1", "action": "archived"})
        mock_learner = MagicMock()
        mock_learner.record_interaction = AsyncMock(side_effect=OSError("disk error"))
        with _patch_user_state(gmail_state):
            with patch(_EMAIL_FEEDBACK_LEARNER, return_value=mock_learner):
                result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_feedback_user_id_from_body(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body(
            {
                "email_id": "msg1",
                "action": "starred",
                "user_id": "custom_user",
            }
        )
        mock_learner = MagicMock()
        mock_learner.record_interaction = AsyncMock(return_value=True)
        with _patch_user_state(gmail_state):
            with patch(_EMAIL_FEEDBACK_LEARNER, return_value=mock_learner) as mock_cls:
                await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
                mock_cls.assert_called_once_with(user_id="custom_user")

    @pytest.mark.asyncio
    async def test_feedback_default_optional_fields(
        self, handler, mock_http_with_body, gmail_state
    ):
        """from_address, subject, labels default to empty."""
        http = mock_http_with_body({"email_id": "msg1", "action": "deleted"})
        mock_learner = MagicMock()
        mock_learner.record_interaction = AsyncMock(return_value=True)
        with _patch_user_state(gmail_state):
            with patch(_EMAIL_FEEDBACK_LEARNER, return_value=mock_learner):
                result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
                call_kwargs = mock_learner.record_interaction.call_args[1]
                assert call_kwargs["from_address"] == ""
                assert call_kwargs["subject"] == ""
                assert call_kwargs["labels"] == []
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_feedback_returns_false(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"email_id": "msg1", "action": "snoozed"})
        mock_learner = MagicMock()
        mock_learner.record_interaction = AsyncMock(return_value=False)
        with _patch_user_state(gmail_state):
            with patch(_EMAIL_FEEDBACK_LEARNER, return_value=mock_learner):
                result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, http)
        assert _status(result) == 200
        assert _body(result)["success"] is False


# ---------------------------------------------------------------------------
# POST routing - 404 for unknown paths
# ---------------------------------------------------------------------------


class TestPostRouting:
    """Tests for POST routing edge cases."""

    @pytest.mark.asyncio
    async def test_post_unknown_path(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})
        with _patch_user_state(gmail_state):
            result = await handler.handle_post("/api/v1/gmail/unknown", {}, http)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_null_body(self, handler, mock_http, gmail_state):
        """If read_json_body returns None, body becomes empty dict."""
        with _patch_user_state(gmail_state):
            with patch.object(handler, "read_json_body", return_value=None):
                result = await handler.handle_post("/api/v1/gmail/inbox/feedback", {}, mock_http)
        # No email_id or action -> 400
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# Rate limiting on POST
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting on POST endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limited(self, handler, mock_http_with_body):
        http = mock_http_with_body({"question": "Test?"})
        with patch(_RATE_LIMITER) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 429
        assert "rate limit" in _body(result)["error"].lower()


# ---------------------------------------------------------------------------
# Auth/RBAC tests (opt-out of auto-auth)
# ---------------------------------------------------------------------------


class TestAuth:
    """Tests for authentication and permission handling."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_get_unauthorized(self, mock_http):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        h = GmailQueryHandler(server_context={})
        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("not auth"),
        ):
            result = await h.handle("/api/v1/gmail/inbox/priority", {}, mock_http)
        assert _status(result) == 401
        assert "Authentication required" in _body(result)["error"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_get_forbidden(self, mock_http):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler

        h = GmailQueryHandler(server_context={})
        mock_ctx = AuthorizationContext(
            user_id="u1",
            user_email="u@e.com",
            roles={"viewer"},
            permissions=set(),
        )
        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ):
            with patch.object(
                SecureHandler,
                "check_permission",
                side_effect=ForbiddenError("no perm"),
            ):
                result = await h.handle("/api/v1/gmail/inbox/priority", {}, mock_http)
        assert _status(result) == 403
        assert "Permission denied" in _body(result)["error"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_unauthorized(self, mock_http_with_body):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        h = GmailQueryHandler(server_context={})
        http = mock_http_with_body({"question": "Test?"})
        with patch(_RATE_LIMITER) as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch.object(
                SecureHandler,
                "get_auth_context",
                new_callable=AsyncMock,
                side_effect=UnauthorizedError("not auth"),
            ):
                result = await h.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_forbidden(self, mock_http_with_body):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler

        h = GmailQueryHandler(server_context={})
        http = mock_http_with_body({"question": "Test?"})
        mock_ctx = AuthorizationContext(
            user_id="u1",
            user_email="u@e.com",
            roles={"viewer"},
            permissions=set(),
        )
        with patch(_RATE_LIMITER) as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            with patch.object(
                SecureHandler,
                "get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ):
                with patch.object(
                    SecureHandler,
                    "check_permission",
                    side_effect=ForbiddenError("no perm"),
                ):
                    result = await h.handle_post("/api/v1/gmail/query", {}, http)
        assert _status(result) == 403


# ---------------------------------------------------------------------------
# _run_query integration tests
# ---------------------------------------------------------------------------


class TestRunQuery:
    """Tests for _run_query method."""

    @pytest.mark.asyncio
    async def test_run_query_no_results(self, handler, gmail_state):
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[])

        with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
            result = await handler._run_query("user1", gmail_state, "test query", 10)

        assert result.answer == "I couldn't find any emails matching your query."
        assert result.confidence == 0.0
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_run_query_with_results_no_content(self, handler, gmail_state):
        search_results = [_make_mock_search_result("gmail-msg1")]

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=search_results)
        mock_connector.get_messages = AsyncMock(return_value=[])

        with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
            result = await handler._run_query("user1", gmail_state, "test query", 10)

        assert "couldn't retrieve" in result.answer.lower()
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_run_query_with_results_and_content(self, handler, gmail_state):
        search_results = [_make_mock_search_result("gmail-msg1")]
        msg = _make_mock_email(msg_id="msg1")

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=search_results)
        mock_connector.get_messages = AsyncMock(return_value=[msg])

        with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
            with patch.object(
                handler,
                "_generate_answer",
                new_callable=AsyncMock,
                return_value="Generated answer here.",
            ):
                result = await handler._run_query("user1", gmail_state, "test query", 10)

        assert result.answer == "Generated answer here."
        assert result.confidence == 0.8
        assert len(result.sources) == 1
        assert result.sources[0]["id"] == "msg1"
        assert "mail.google.com" in result.sources[0]["url"]

    @pytest.mark.asyncio
    async def test_run_query_no_date(self, handler, gmail_state):
        search_results = [_make_mock_search_result("gmail-msg1")]
        msg = _make_mock_email(msg_id="msg1")
        msg.date = None

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=search_results)
        mock_connector.get_messages = AsyncMock(return_value=[msg])

        with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
            with patch.object(
                handler,
                "_generate_answer",
                new_callable=AsyncMock,
                return_value="answer",
            ):
                result = await handler._run_query("user1", gmail_state, "test query", 10)

        assert result.sources[0]["date"] is None

    @pytest.mark.asyncio
    async def test_run_query_empty_answer_lower_confidence(self, handler, gmail_state):
        search_results = [_make_mock_search_result("gmail-msg1")]
        msg = _make_mock_email(msg_id="msg1")

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=search_results)
        mock_connector.get_messages = AsyncMock(return_value=[msg])

        with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
            with patch.object(
                handler,
                "_generate_answer",
                new_callable=AsyncMock,
                return_value="",
            ):
                result = await handler._run_query("user1", gmail_state, "test query", 10)

        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_run_query_uses_snippet_when_no_body(self, handler, gmail_state):
        search_results = [_make_mock_search_result("gmail-msg1")]
        msg = _make_mock_email(msg_id="msg1")
        msg.body_text = None
        msg.snippet = "Use this snippet"

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=search_results)
        mock_connector.get_messages = AsyncMock(return_value=[msg])

        with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
            with patch.object(
                handler,
                "_generate_answer",
                new_callable=AsyncMock,
                return_value="answer",
            ) as mock_gen:
                await handler._run_query("user1", gmail_state, "test query", 10)
                # The content passed to generate_answer should contain the snippet
                emails_content = mock_gen.call_args[0][1]
                assert any("Use this snippet" in c for c in emails_content)


# ---------------------------------------------------------------------------
# _simple_answer tests
# ---------------------------------------------------------------------------


class TestSimpleAnswer:
    """Tests for _simple_answer fallback method."""

    def test_how_many_question(self, handler):
        emails = ["From: A\nSubject: X\n", "From: B\nSubject: Y\n"]
        answer = handler._simple_answer("How many emails about this?", emails)
        assert "2" in answer

    def test_from_question(self, handler):
        emails = ["From: alice@example.com\nSubject: Hi\n"]
        answer = handler._simple_answer("Who sent it?", emails)
        assert "alice@example.com" in answer

    def test_from_with_from_keyword(self, handler):
        emails = ["From: bob@work.com\nSubject: Report\n"]
        answer = handler._simple_answer("from whom?", emails)
        assert "bob@work.com" in answer

    def test_about_question(self, handler):
        emails = [
            "From: A\nSubject: Budget Review\n",
            "From: B\nSubject: Q4 Planning\n",
        ]
        answer = handler._simple_answer("What are these about?", emails)
        assert "Budget Review" in answer
        assert "Q4 Planning" in answer

    def test_what_question(self, handler):
        emails = ["From: A\nSubject: Meeting Notes\n"]
        answer = handler._simple_answer("what is this?", emails)
        assert "Meeting Notes" in answer

    def test_generic_question(self, handler):
        emails = ["From: A\nSubject: X\n", "From: B\nSubject: Y\n"]
        answer = handler._simple_answer("Tell me more", emails)
        assert "2 relevant emails" in answer

    def test_empty_emails(self, handler):
        answer = handler._simple_answer("How many?", [])
        assert "0" in answer

    def test_multiple_senders_limited(self, handler):
        emails = [f"From: user{i}@test.com\nSubject: X\n" for i in range(10)]
        answer = handler._simple_answer("Who sent them?", emails)
        # Should limit to 5 senders
        assert answer.count("@test.com") <= 5

    def test_multiple_subjects_limited(self, handler):
        emails = [f"From: A\nSubject: Topic {i}\n" for i in range(10)]
        answer = handler._simple_answer("What are they about?", emails)
        # Should limit to 3 subjects
        count = sum(1 for i in range(10) if f"Topic {i}" in answer)
        assert count <= 3


# ---------------------------------------------------------------------------
# _transcribe tests
# ---------------------------------------------------------------------------


class TestTranscribe:
    """Tests for _transcribe method."""

    @pytest.mark.asyncio
    async def test_transcribe_success(self, handler):
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        mock_connector = MagicMock()
        mock_connector.transcribe = AsyncMock(return_value=mock_result)

        with patch(
            "aragora.connectors.whisper.WhisperConnector",
            return_value=mock_connector,
        ):
            text = await handler._transcribe(b"audio-bytes")
        assert text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_import_error(self, handler):
        import builtins

        real_import = builtins.__import__

        def fail_whisper(name, *args, **kwargs):
            if "whisper" in name:
                raise ImportError("no whisper")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_whisper):
            text = await handler._transcribe(b"audio-bytes")
        assert text is None

    @pytest.mark.asyncio
    async def test_transcribe_connection_error(self, handler):
        mock_connector = MagicMock()
        mock_connector.transcribe = AsyncMock(side_effect=ConnectionError("fail"))
        with patch(
            "aragora.connectors.whisper.WhisperConnector",
            return_value=mock_connector,
        ):
            text = await handler._transcribe(b"audio-bytes")
        assert text is None

    @pytest.mark.asyncio
    async def test_transcribe_returns_none_result(self, handler):
        mock_connector = MagicMock()
        mock_connector.transcribe = AsyncMock(return_value=None)
        with patch(
            "aragora.connectors.whisper.WhisperConnector",
            return_value=mock_connector,
        ):
            text = await handler._transcribe(b"audio-bytes")
        assert text is None


# ---------------------------------------------------------------------------
# _generate_answer tests
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    """Tests for _generate_answer method."""

    @pytest.mark.asyncio
    async def test_generate_answer_rlm_success(self, handler):
        mock_rlm = MagicMock()
        mock_rlm.query = AsyncMock(return_value="RLM answer")
        mock_rlm.compress = AsyncMock(return_value=None)

        mock_context_cls = MagicMock()

        with patch(
            "aragora.rlm.streaming.StreamingRLMQuery",
            return_value=mock_rlm,
        ):
            with patch(
                "aragora.rlm.types.RLMContext",
                mock_context_cls,
            ):
                answer = await handler._generate_answer("question?", ["email content"])
        assert answer == "RLM answer"

    @pytest.mark.asyncio
    async def test_generate_answer_rlm_import_error_falls_back(self, handler):
        """When RLM is not importable, falls back to LLM agent."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "LLM answer"
        mock_agent.respond = AsyncMock(return_value=mock_response)

        import builtins

        real_import = builtins.__import__

        def fail_rlm(name, *args, **kwargs):
            if "aragora.rlm.streaming" in name or "aragora.rlm.types" in name:
                raise ImportError("no rlm")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_rlm):
            with patch(
                "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
                return_value=mock_agent,
            ):
                answer = await handler._generate_answer("question?", ["email content"])
        assert answer == "LLM answer"

    @pytest.mark.asyncio
    async def test_generate_answer_both_fail_uses_simple(self, handler):
        """When both RLM and LLM fail, falls back to _simple_answer."""
        import builtins

        real_import = builtins.__import__

        def fail_all(name, *args, **kwargs):
            if "aragora.rlm.streaming" in name or "aragora.rlm.types" in name:
                raise ImportError("no rlm")
            if "aragora.agents.api_agents.anthropic" in name:
                raise ImportError("no anthropic")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_all):
            answer = await handler._generate_answer(
                "How many emails?",
                ["From: A\nSubject: X\n"],
            )
        assert "1" in answer

    @pytest.mark.asyncio
    async def test_generate_answer_rlm_returns_empty_falls_back(self, handler):
        mock_rlm = MagicMock()
        mock_rlm.query = AsyncMock(return_value="")

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "LLM fallback"
        mock_agent.respond = AsyncMock(return_value=mock_response)

        mock_context_cls = MagicMock()

        with patch(
            "aragora.rlm.streaming.StreamingRLMQuery",
            return_value=mock_rlm,
        ):
            with patch(
                "aragora.rlm.types.RLMContext",
                mock_context_cls,
            ):
                with patch(
                    "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
                    return_value=mock_agent,
                ):
                    answer = await handler._generate_answer("question?", ["email content"])
        assert answer == "LLM fallback"

    @pytest.mark.asyncio
    async def test_generate_answer_rlm_none_falls_back(self, handler):
        mock_rlm = MagicMock()
        mock_rlm.query = AsyncMock(return_value=None)

        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "LLM fallback"
        mock_agent.respond = AsyncMock(return_value=mock_response)

        mock_context_cls = MagicMock()

        with patch(
            "aragora.rlm.streaming.StreamingRLMQuery",
            return_value=mock_rlm,
        ):
            with patch(
                "aragora.rlm.types.RLMContext",
                mock_context_cls,
            ):
                with patch(
                    "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
                    return_value=mock_agent,
                ):
                    answer = await handler._generate_answer("question?", ["email content"])
        assert answer == "LLM fallback"

    @pytest.mark.asyncio
    async def test_generate_answer_llm_returns_none(self, handler):
        """When LLM returns None response, falls back to _simple_answer."""
        import builtins

        real_import = builtins.__import__

        def fail_rlm(name, *args, **kwargs):
            if "aragora.rlm.streaming" in name or "aragora.rlm.types" in name:
                raise ImportError("no rlm")
            return real_import(name, *args, **kwargs)

        mock_agent = MagicMock()
        mock_agent.respond = AsyncMock(return_value=None)

        with patch("builtins.__import__", side_effect=fail_rlm):
            with patch(
                "aragora.agents.api_agents.anthropic.AnthropicAPIAgent",
                return_value=mock_agent,
            ):
                answer = await handler._generate_answer(
                    "How many emails?", ["From: A\nSubject: X\n"]
                )
        # Falls back to simple answer
        assert "1" in answer


# ---------------------------------------------------------------------------
# _get_prioritized_emails sorting edge cases
# ---------------------------------------------------------------------------


class TestPrioritySorting:
    """Tests for priority score sorting edge cases."""

    @pytest.mark.asyncio
    async def test_priority_none_score_treated_as_zero(self, handler, mock_http, gmail_state):
        mock_msgs = [_make_mock_email(msg_id="msg1")]
        mock_score = MagicMock()
        mock_score.score = None
        mock_score.reason = "Unknown"

        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=(["msg1"], None))
        mock_connector.get_messages = AsyncMock(return_value=mock_msgs)

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=[mock_score])

        with _patch_user_state(gmail_state):
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)

        assert _status(result) == 200
        email = _body(result)["emails"][0]
        assert email["priority_score"] is None

    @pytest.mark.asyncio
    async def test_priority_date_none(self, handler, mock_http, gmail_state):
        msg = _make_mock_email(msg_id="msg1")
        msg.date = None
        mock_score = _make_mock_score(score=0.5)

        mock_connector = MagicMock()
        mock_connector.list_messages = AsyncMock(return_value=(["msg1"], None))
        mock_connector.get_messages = AsyncMock(return_value=[msg])

        mock_analyzer = MagicMock()
        mock_analyzer.score_batch = AsyncMock(return_value=[mock_score])

        with _patch_user_state(gmail_state):
            with patch(_GMAIL_CONNECTOR, return_value=mock_connector):
                with patch(_EMAIL_PRIORITY_ANALYZER, return_value=mock_analyzer):
                    result = await handler.handle("/api/v1/gmail/inbox/priority", {}, mock_http)

        assert _status(result) == 200
        email = _body(result)["emails"][0]
        assert email["date"] is None
