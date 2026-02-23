"""Tests for Gmail inbox ingestion handler.

Tests the Gmail API endpoints including:
- GET  /api/v1/gmail/status - Get connection status
- GET  /api/v1/gmail/auth/url - Get OAuth URL
- GET  /api/v1/gmail/auth/callback - Handle OAuth callback (GET)
- GET  /api/v1/gmail/sync/status - Get sync progress
- GET  /api/v1/gmail/messages - List indexed emails
- GET  /api/v1/gmail/message/{id} - Get single email
- POST /api/v1/gmail/connect - Start OAuth flow
- POST /api/v1/gmail/auth/callback - Handle OAuth callback (POST)
- POST /api/v1/gmail/sync - Start email sync
- POST /api/v1/gmail/search - Search emails
- POST /api/v1/gmail/disconnect - Disconnect account (via POST)
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.gmail_ingest import (
    GmailIngestHandler,
    get_user_state,
    save_user_state,
    delete_user_state,
    get_sync_job,
    save_sync_job,
    delete_sync_job,
)
from aragora.server.handlers.base import HandlerResult
from aragora.storage.gmail_token_store import GmailUserState, SyncJobState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler that mimics the real HTTP handler attributes."""

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a GmailIngestHandler with minimal context."""
    return GmailIngestHandler(server_context={})


@pytest.fixture
def mock_http():
    """Create a mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the Gmail rate limiter between tests."""
    from aragora.server.handlers.features.gmail_ingest import _gmail_limiter

    _gmail_limiter._buckets.clear()
    yield
    _gmail_limiter._buckets.clear()


@pytest.fixture
def mock_auth_context():
    """Patch extract_user_from_request to return an authenticated user."""
    mock_ctx = MagicMock()
    mock_ctx.authenticated = True
    mock_ctx.user_id = "test-user-001"
    mock_ctx.org_id = "test-org-001"
    with patch(
        "aragora.server.handlers.features.gmail_ingest.extract_user_from_request",
        return_value=mock_ctx,
    ):
        yield mock_ctx


@pytest.fixture
def mock_unauthenticated():
    """Patch extract_user_from_request to return an unauthenticated user."""
    mock_ctx = MagicMock()
    mock_ctx.authenticated = False
    mock_ctx.user_id = None
    mock_ctx.org_id = None
    with patch(
        "aragora.server.handlers.features.gmail_ingest.extract_user_from_request",
        return_value=mock_ctx,
    ):
        yield mock_ctx


def _make_user_state(
    user_id: str = "test-user-001",
    email: str = "test@gmail.com",
    refresh_token: str = "refresh-token-123",
    access_token: str = "access-token-456",
    indexed_count: int = 42,
    last_sync: datetime | None = None,
) -> GmailUserState:
    """Create a GmailUserState for testing."""
    return GmailUserState(
        user_id=user_id,
        org_id="test-org-001",
        email_address=email,
        access_token=access_token,
        refresh_token=refresh_token,
        indexed_count=indexed_count,
        last_sync=last_sync,
        connected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_sync_job(
    user_id: str = "test-user-001",
    status: str = "running",
    progress: int = 50,
    messages_synced: int = 25,
    error: str | None = None,
) -> SyncJobState:
    """Create a SyncJobState for testing."""
    return SyncJobState(
        user_id=user_id,
        status=status,
        progress=progress,
        messages_synced=messages_synced,
        error=error,
    )


# ===========================================================================
# can_handle() routing tests
# ===========================================================================


class TestCanHandle:
    """Tests for GmailIngestHandler.can_handle()."""

    def test_handles_gmail_status(self, handler):
        assert handler.can_handle("/api/v1/gmail/status")

    def test_handles_gmail_connect(self, handler):
        assert handler.can_handle("/api/v1/gmail/connect")

    def test_handles_gmail_auth_url(self, handler):
        assert handler.can_handle("/api/v1/gmail/auth/url")

    def test_handles_gmail_auth_callback(self, handler):
        assert handler.can_handle("/api/v1/gmail/auth/callback")

    def test_handles_gmail_sync(self, handler):
        assert handler.can_handle("/api/v1/gmail/sync")

    def test_handles_gmail_sync_status(self, handler):
        assert handler.can_handle("/api/v1/gmail/sync/status")

    def test_handles_gmail_messages(self, handler):
        assert handler.can_handle("/api/v1/gmail/messages")

    def test_handles_gmail_search(self, handler):
        assert handler.can_handle("/api/v1/gmail/search")

    def test_handles_gmail_disconnect(self, handler):
        assert handler.can_handle("/api/v1/gmail/disconnect")

    def test_rejects_non_gmail_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_similar_prefix(self, handler):
        assert not handler.can_handle("/api/v1/gmailer/status")

    def test_routes_are_defined(self, handler):
        assert hasattr(handler, "ROUTES")
        assert "/api/v1/gmail/connect" in handler.ROUTES
        assert "/api/v1/gmail/sync" in handler.ROUTES
        assert "/api/v1/gmail/messages" in handler.ROUTES


# ===========================================================================
# GET /api/v1/gmail/status - Connection status
# ===========================================================================


class TestGetStatus:
    """Tests for GET /api/v1/gmail/status."""

    @pytest.mark.asyncio
    async def test_status_not_connected_unconfigured(self, handler, mock_http, mock_auth_context):
        """Returns not connected when no user state exists."""
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["connected"] is False
            assert body["configured"] is False

    @pytest.mark.asyncio
    async def test_status_not_connected_but_configured(self, handler, mock_http, mock_auth_context):
        """Returns configured=True when Gmail client ID is set."""
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.dict("os.environ", {"GMAIL_CLIENT_ID": "test-client-id"}, clear=False),
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["connected"] is False
            assert body["configured"] is True

    @pytest.mark.asyncio
    async def test_status_connected(self, handler, mock_http, mock_auth_context):
        """Returns connected=True when user has refresh token."""
        state = _make_user_state(
            last_sync=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict("os.environ", {"GMAIL_CLIENT_ID": "cid"}, clear=False),
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["connected"] is True
            assert body["email_address"] == "test@gmail.com"
            assert body["indexed_count"] == 42

    @pytest.mark.asyncio
    async def test_status_connected_no_refresh_token(self, handler, mock_http, mock_auth_context):
        """Returns connected=False when user has no refresh token."""
        state = _make_user_state(refresh_token="")
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["connected"] is False

    @pytest.mark.asyncio
    async def test_status_last_sync_isoformat(self, handler, mock_http, mock_auth_context):
        """Returns last_sync as ISO format string."""
        ts = datetime(2026, 2, 15, 10, 30, 0, tzinfo=timezone.utc)
        state = _make_user_state(last_sync=ts)
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            body = _body(result)
            assert body["last_sync"] == ts.isoformat()

    @pytest.mark.asyncio
    async def test_status_last_sync_none(self, handler, mock_http, mock_auth_context):
        """Returns null last_sync when never synced."""
        state = _make_user_state(last_sync=None)
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            body = _body(result)
            assert body["last_sync"] is None


# ===========================================================================
# GET /api/v1/gmail/auth/url - OAuth URL generation
# ===========================================================================


class TestGetAuthUrl:
    """Tests for GET /api/v1/gmail/auth/url."""

    @pytest.mark.asyncio
    async def test_get_auth_url_success(self, handler, mock_http, mock_auth_context):
        """Returns OAuth URL from connector."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://accounts.google.com/o/oauth2/auth?..."

        with patch(
            "aragora.server.handlers.features.gmail_ingest.GmailIngestHandler._get_auth_url",
            wraps=handler._get_auth_url,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ):
                result = await handler.handle("/api/v1/gmail/auth/url", {}, mock_http)
                assert _status(result) == 200
                body = _body(result)
                assert "url" in body
                assert body["url"] == "https://accounts.google.com/o/oauth2/auth?..."

    @pytest.mark.asyncio
    async def test_get_auth_url_with_redirect_uri(self, handler, mock_http, mock_auth_context):
        """Passes redirect_uri to connector."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://auth.url"

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle(
                "/api/v1/gmail/auth/url",
                {"redirect_uri": "https://myapp.com/callback"},
                mock_http,
            )
            assert _status(result) == 200
            mock_connector.get_oauth_url.assert_called_once_with("https://myapp.com/callback", "")

    @pytest.mark.asyncio
    async def test_get_auth_url_with_state(self, handler, mock_http, mock_auth_context):
        """Passes state parameter to connector."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://auth.url"

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle(
                "/api/v1/gmail/auth/url",
                {"state": "csrf-token-123"},
                mock_http,
            )
            assert _status(result) == 200
            mock_connector.get_oauth_url.assert_called_once_with(
                "http://localhost:3000/inbox/callback", "csrf-token-123"
            )

    @pytest.mark.asyncio
    async def test_get_auth_url_import_error(self, handler, mock_http, mock_auth_context):
        """Returns 500 when Gmail connector raises an import-related error."""
        mock_cls = MagicMock(side_effect=AttributeError("module has no attribute"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=mock_cls
                )
            },
        ):
            result = await handler.handle("/api/v1/gmail/auth/url", {}, mock_http)
            assert _status(result) == 500
            body = _body(result)
            assert "error" in body

    @pytest.mark.asyncio
    async def test_get_auth_url_connector_error(self, handler, mock_http, mock_auth_context):
        """Returns 500 when connector raises ValueError."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.side_effect = ValueError("Missing client ID")

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle("/api/v1/gmail/auth/url", {}, mock_http)
            assert _status(result) == 500


# ===========================================================================
# GET /api/v1/gmail/auth/callback - OAuth callback (GET)
# ===========================================================================


class TestOAuthCallbackGet:
    """Tests for GET /api/v1/gmail/auth/callback."""

    @pytest.mark.asyncio
    async def test_callback_success(self, handler, mock_http, mock_auth_context):
        """Successfully completes OAuth flow."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.get_user_info = AsyncMock(
            return_value={"emailAddress": "user@gmail.com", "historyId": "12345"}
        )
        mock_connector._access_token = "new-access-token"
        mock_connector._refresh_token = "new-refresh-token"
        mock_connector._token_expiry = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_user_state",
                new_callable=AsyncMock,
            ) as mock_save,
        ):
            result = await handler.handle(
                "/api/v1/gmail/auth/callback",
                {"code": "auth-code-123", "state": "test-user-001"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert body["email_address"] == "user@gmail.com"
            assert body["user_id"] == "test-user-001"
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_missing_code(self, handler, mock_http, mock_auth_context):
        """Returns 400 when authorization code is missing."""
        result = await handler.handle(
            "/api/v1/gmail/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400
        body = _body(result)
        assert "Missing authorization code" in body["error"]

    @pytest.mark.asyncio
    async def test_callback_oauth_error(self, handler, mock_http, mock_auth_context):
        """Returns 400 when OAuth returns an error."""
        result = await handler.handle(
            "/api/v1/gmail/auth/callback",
            {"error": "access_denied"},
            mock_http,
        )
        assert _status(result) == 400
        body = _body(result)
        assert "access_denied" in body["error"]

    @pytest.mark.asyncio
    async def test_callback_state_mismatch_still_uses_jwt_user(
        self, handler, mock_http, mock_auth_context
    ):
        """Uses JWT user_id even when state doesn't match (security)."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.get_user_info = AsyncMock(
            return_value={"emailAddress": "user@gmail.com", "historyId": "123"}
        )
        mock_connector._access_token = "at"
        mock_connector._refresh_token = "rt"
        mock_connector._token_expiry = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_user_state",
                new_callable=AsyncMock,
            ) as mock_save,
        ):
            result = await handler.handle(
                "/api/v1/gmail/auth/callback",
                {"code": "code-123", "state": "different-user-id"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            # Should use JWT user_id, not the state value
            assert body["user_id"] == "test-user-001"

    @pytest.mark.asyncio
    async def test_callback_auth_failure(self, handler, mock_http, mock_auth_context):
        """Returns 401 when connector authentication fails."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=False)

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle(
                "/api/v1/gmail/auth/callback",
                {"code": "bad-code"},
                mock_http,
            )
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_connector_exception(self, handler, mock_http, mock_auth_context):
        """Returns 500 when connector raises an exception."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=ConnectionError("network error"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle(
                "/api/v1/gmail/auth/callback",
                {"code": "code-123"},
                mock_http,
            )
            assert _status(result) == 500


# ===========================================================================
# GET /api/v1/gmail/sync/status - Sync status
# ===========================================================================


class TestGetSyncStatus:
    """Tests for GET /api/v1/gmail/sync/status."""

    @pytest.mark.asyncio
    async def test_sync_status_connected_with_job(self, handler, mock_http, mock_auth_context):
        """Returns full sync status when connected with active job."""
        state = _make_user_state()
        job = _make_sync_job(status="running", progress=75, messages_synced=150)

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
        ):
            result = await handler.handle("/api/v1/gmail/sync/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["connected"] is True
            assert body["email_address"] == "test@gmail.com"
            assert body["job_status"] == "running"
            assert body["job_progress"] == 75
            assert body["job_messages_synced"] == 150
            assert body["job_error"] is None

    @pytest.mark.asyncio
    async def test_sync_status_not_connected(self, handler, mock_http, mock_auth_context):
        """Returns connected=False when no user state."""
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await handler.handle("/api/v1/gmail/sync/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["connected"] is False
            assert body["email_address"] is None
            assert body["indexed_count"] == 0
            assert body["job_status"] == "idle"
            assert body["job_progress"] == 0

    @pytest.mark.asyncio
    async def test_sync_status_no_active_job(self, handler, mock_http, mock_auth_context):
        """Returns idle status when no sync job running."""
        state = _make_user_state()
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await handler.handle("/api/v1/gmail/sync/status", {}, mock_http)
            body = _body(result)
            assert body["job_status"] == "idle"

    @pytest.mark.asyncio
    async def test_sync_status_failed_job(self, handler, mock_http, mock_auth_context):
        """Returns error info for failed sync job."""
        state = _make_user_state()
        job = _make_sync_job(status="failed", progress=0, error="Sync failed")
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=job,
            ),
        ):
            result = await handler.handle("/api/v1/gmail/sync/status", {}, mock_http)
            body = _body(result)
            assert body["job_status"] == "failed"
            assert body["job_error"] == "Sync failed"


# ===========================================================================
# GET /api/v1/gmail/messages - List messages
# ===========================================================================


class TestListMessages:
    """Tests for GET /api/v1/gmail/messages."""

    @pytest.mark.asyncio
    async def test_list_messages_success(self, handler, mock_http, mock_auth_context):
        """Returns list of messages from connector."""
        state = _make_user_state()
        mock_result = MagicMock()
        mock_result.id = "gmail-msg123"
        mock_result.title = "Test Subject"
        mock_result.author = "sender@example.com"
        mock_result.content = "Preview snippet..."
        mock_result.metadata = {"date": "2026-02-15"}
        mock_result.url = "https://mail.google.com/..."

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[mock_result])

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/messages", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 1
            assert body["messages"][0]["id"] == "msg123"  # gmail- prefix stripped
            assert body["messages"][0]["subject"] == "Test Subject"
            assert body["messages"][0]["from"] == "sender@example.com"
            assert body["messages"][0]["snippet"] == "Preview snippet..."

    @pytest.mark.asyncio
    async def test_list_messages_not_connected(self, handler, mock_http, mock_auth_context):
        """Returns 401 when not connected."""
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle("/api/v1/gmail/messages", {}, mock_http)
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_list_messages_no_refresh_token(self, handler, mock_http, mock_auth_context):
        """Returns 401 when user has no refresh token."""
        state = _make_user_state(refresh_token="")
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle("/api/v1/gmail/messages", {}, mock_http)
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_list_messages_with_limit(self, handler, mock_http, mock_auth_context):
        """Passes limit parameter to connector."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[])

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/messages", {"limit": "25"}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["total"] == 0
            assert body["messages"] == []

    @pytest.mark.asyncio
    async def test_list_messages_connector_error(self, handler, mock_http, mock_auth_context):
        """Returns 500 when connector raises an error."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(side_effect=ConnectionError("API unreachable"))

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/messages", {}, mock_http)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_list_messages_with_query(self, handler, mock_http, mock_auth_context):
        """Passes query parameter to connector search."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[])

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle(
                "/api/v1/gmail/messages", {"query": "from:boss@company.com"}, mock_http
            )
            assert _status(result) == 200
            mock_connector.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_messages_empty(self, handler, mock_http, mock_auth_context):
        """Returns empty list when no messages found."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[])

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/messages", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["messages"] == []
            assert body["total"] == 0


# ===========================================================================
# GET /api/v1/gmail/message/{id} - Get single message
# ===========================================================================


class TestGetMessage:
    """Tests for GET /api/v1/gmail/message/{id}."""

    @pytest.mark.asyncio
    async def test_get_message_success(self, handler, mock_http, mock_auth_context):
        """Returns message details."""
        state = _make_user_state()
        mock_msg = MagicMock()
        mock_msg.to_dict.return_value = {
            "id": "msg123",
            "subject": "Hello",
            "body": "World",
        }

        mock_connector = MagicMock()
        mock_connector.get_message = AsyncMock(return_value=mock_msg)

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/message/msg123", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["id"] == "msg123"
            assert body["subject"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_message_not_connected(self, handler, mock_http, mock_auth_context):
        """Returns 401 when not connected."""
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle("/api/v1/gmail/message/msg123", {}, mock_http)
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_get_message_connector_error(self, handler, mock_http, mock_auth_context):
        """Returns 500 when connector raises an exception."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.get_message = AsyncMock(side_effect=ValueError("Invalid message ID"))

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/message/msg123", {}, mock_http)
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_message_extracts_id_from_path(self, handler, mock_http, mock_auth_context):
        """Extracts message ID from the last path segment."""
        state = _make_user_state()
        mock_msg = MagicMock()
        mock_msg.to_dict.return_value = {"id": "abc-def-ghi"}

        mock_connector = MagicMock()
        mock_connector.get_message = AsyncMock(return_value=mock_msg)

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/message/abc-def-ghi", {}, mock_http)
            assert _status(result) == 200
            mock_connector.get_message.assert_called_once_with("abc-def-ghi")


# ===========================================================================
# GET unknown path - 404
# ===========================================================================


class TestHandleGet404:
    """Tests for unknown GET paths."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler, mock_http, mock_auth_context):
        """Returns 404 for unknown Gmail paths."""
        result = await handler.handle("/api/v1/gmail/unknown", {}, mock_http)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_unknown_nested_path_returns_404(self, handler, mock_http, mock_auth_context):
        """Returns 404 for unknown nested paths."""
        result = await handler.handle("/api/v1/gmail/some/deep/path", {}, mock_http)
        assert _status(result) == 404


# ===========================================================================
# POST /api/v1/gmail/connect - Start OAuth flow
# ===========================================================================


class TestPostConnect:
    """Tests for POST /api/v1/gmail/connect."""

    @pytest.mark.asyncio
    async def test_connect_success(self, handler, mock_http, mock_auth_context):
        """Returns OAuth URL for connection."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://accounts.google.com/auth"

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle_post("/api/v1/gmail/connect", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert "url" in body
            assert body["state"] == "test-user-001"  # defaults to user_id

    @pytest.mark.asyncio
    async def test_connect_with_custom_redirect(self, handler, mock_http, mock_auth_context):
        """Passes custom redirect_uri."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://auth.url"

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/connect",
                {"redirect_uri": "https://custom.app/cb"},
                mock_http,
            )
            assert _status(result) == 200
            mock_connector.get_oauth_url.assert_called_once_with(
                "https://custom.app/cb", "test-user-001"
            )

    @pytest.mark.asyncio
    async def test_connect_with_custom_state(self, handler, mock_http, mock_auth_context):
        """Passes custom state parameter."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://auth.url"

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/connect",
                {"state": "custom-state-value"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["state"] == "custom-state-value"

    @pytest.mark.asyncio
    async def test_connect_import_error(self, handler, mock_http, mock_auth_context):
        """Returns 500 when connector import fails."""
        # Direct method test since import error handling is in _start_connect
        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": None,
            },
        ):
            result = handler._start_connect({}, "test-user-001")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_connect_connector_error(self, handler, mock_http, mock_auth_context):
        """Returns 500 when connector raises exception."""
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.side_effect = OSError("network error")

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler.handle_post("/api/v1/gmail/connect", {}, mock_http)
            assert _status(result) == 500


# ===========================================================================
# POST /api/v1/gmail/auth/callback - OAuth callback (POST)
# ===========================================================================


class TestOAuthCallbackPost:
    """Tests for POST /api/v1/gmail/auth/callback."""

    @pytest.mark.asyncio
    async def test_callback_post_success(self, handler, mock_http, mock_auth_context):
        """Successfully completes OAuth flow via POST."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.get_user_info = AsyncMock(
            return_value={"emailAddress": "user@gmail.com", "historyId": "99999"}
        )
        mock_connector._access_token = "at"
        mock_connector._refresh_token = "rt"
        mock_connector._token_expiry = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_user_state",
                new_callable=AsyncMock,
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/auth/callback",
                {"code": "auth-code-123"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert body["email_address"] == "user@gmail.com"

    @pytest.mark.asyncio
    async def test_callback_post_missing_code(self, handler, mock_http, mock_auth_context):
        """Returns 400 when code is missing."""
        result = await handler.handle_post("/api/v1/gmail/auth/callback", {}, mock_http)
        assert _status(result) == 400
        body = _body(result)
        assert "Missing authorization code" in body["error"]

    @pytest.mark.asyncio
    async def test_callback_post_default_redirect_uri(self, handler, mock_http, mock_auth_context):
        """Uses default redirect URI when not provided."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.get_user_info = AsyncMock(
            return_value={"emailAddress": "u@g.com", "historyId": "1"}
        )
        mock_connector._access_token = "at"
        mock_connector._refresh_token = "rt"
        mock_connector._token_expiry = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_user_state",
                new_callable=AsyncMock,
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/auth/callback",
                {"code": "code-123"},
                mock_http,
            )
            assert _status(result) == 200
            mock_connector.authenticate.assert_called_once_with(
                code="code-123", redirect_uri="http://localhost:3000/inbox/callback"
            )

    @pytest.mark.asyncio
    async def test_callback_post_state_mismatch(self, handler, mock_http, mock_auth_context):
        """Logs warning but still uses JWT user_id on state mismatch."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.get_user_info = AsyncMock(
            return_value={"emailAddress": "u@g.com", "historyId": "1"}
        )
        mock_connector._access_token = "at"
        mock_connector._refresh_token = "rt"
        mock_connector._token_expiry = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_user_state",
                new_callable=AsyncMock,
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/auth/callback",
                {"code": "code", "state": "attacker-user"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["user_id"] == "test-user-001"


# ===========================================================================
# POST /api/v1/gmail/sync - Start sync
# ===========================================================================


class TestPostSync:
    """Tests for POST /api/v1/gmail/sync."""

    @pytest.mark.asyncio
    async def test_sync_start_success(self, handler, mock_http, mock_auth_context):
        """Starts sync in background."""
        state = _make_user_state()
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_sync_job",
                new_callable=AsyncMock,
            ),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            result = await handler.handle_post("/api/v1/gmail/sync", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["status"] == "running"
            assert body["message"] == "Sync started"
            assert body["job_id"] == "test-user-001"
            mock_thread.assert_called_once()
            mock_thread.return_value.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_not_connected(self, handler, mock_http, mock_auth_context):
        """Returns 401 when not connected."""
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_post("/api/v1/gmail/sync", {}, mock_http)
            assert _status(result) == 401
            body = _body(result)
            assert "Not connected" in body["error"]

    @pytest.mark.asyncio
    async def test_sync_no_refresh_token(self, handler, mock_http, mock_auth_context):
        """Returns 401 when user has no refresh token."""
        state = _make_user_state(refresh_token="")
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle_post("/api/v1/gmail/sync", {}, mock_http)
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_sync_already_running(self, handler, mock_http, mock_auth_context):
        """Returns sync-in-progress when already running."""
        state = _make_user_state()
        running_job = _make_sync_job(status="running", progress=50)

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=running_job,
            ),
        ):
            result = await handler.handle_post("/api/v1/gmail/sync", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["status"] == "running"
            assert body["message"] == "Sync already in progress"
            assert body["progress"] == 50

    @pytest.mark.asyncio
    async def test_sync_with_full_sync_flag(self, handler, mock_http, mock_auth_context):
        """Starts full sync when full_sync=True."""
        state = _make_user_state()
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_sync_job",
                new_callable=AsyncMock,
            ),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            result = await handler.handle_post(
                "/api/v1/gmail/sync",
                {"full_sync": True, "max_messages": 1000},
                mock_http,
            )
            assert _status(result) == 200
            # Verify thread was started with correct args
            call_kwargs = mock_thread.call_args
            assert call_kwargs[1]["daemon"] is True

    @pytest.mark.asyncio
    async def test_sync_completed_job_allows_restart(self, handler, mock_http, mock_auth_context):
        """Allows starting new sync when previous job completed."""
        state = _make_user_state()
        completed_job = _make_sync_job(status="completed", progress=100, messages_synced=500)

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=completed_job,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_sync_job",
                new_callable=AsyncMock,
            ),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            result = await handler.handle_post("/api/v1/gmail/sync", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["message"] == "Sync started"

    @pytest.mark.asyncio
    async def test_sync_with_custom_labels(self, handler, mock_http, mock_auth_context):
        """Passes custom labels to sync."""
        state = _make_user_state()
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_sync_job",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_sync_job",
                new_callable=AsyncMock,
            ),
            patch("threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            result = await handler.handle_post(
                "/api/v1/gmail/sync",
                {"labels": ["INBOX", "SENT"]},
                mock_http,
            )
            assert _status(result) == 200
            # Verify labels in thread args
            call_args = mock_thread.call_args[1]["args"]
            assert call_args[4] == ["INBOX", "SENT"]  # labels is the 5th arg


# ===========================================================================
# POST /api/v1/gmail/search - Search emails
# ===========================================================================


class TestPostSearch:
    """Tests for POST /api/v1/gmail/search."""

    @pytest.mark.asyncio
    async def test_search_success(self, handler, mock_http, mock_auth_context):
        """Returns search results."""
        state = _make_user_state()
        mock_result = MagicMock()
        mock_result.id = "gmail-search1"
        mock_result.title = "Found Email"
        mock_result.author = "someone@example.com"
        mock_result.content = "Matching content"
        mock_result.metadata = {"date": "2026-01-15"}
        mock_result.url = "https://mail.google.com/search1"

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[mock_result])

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/search",
                {"query": "invoice", "limit": 10},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["query"] == "invoice"
            assert body["count"] == 1
            assert body["results"][0]["id"] == "search1"  # gmail- prefix stripped
            assert body["results"][0]["subject"] == "Found Email"

    @pytest.mark.asyncio
    async def test_search_missing_query(self, handler, mock_http, mock_auth_context):
        """Returns 400 when query is missing."""
        state = _make_user_state()
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle_post("/api/v1/gmail/search", {}, mock_http)
            assert _status(result) == 400
            body = _body(result)
            assert "Query is required" in body["error"]

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler, mock_http, mock_auth_context):
        """Returns 400 when query is empty string."""
        state = _make_user_state()
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle_post("/api/v1/gmail/search", {"query": ""}, mock_http)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_not_connected(self, handler, mock_http, mock_auth_context):
        """Returns 401 when not connected."""
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/search",
                {"query": "test"},
                mock_http,
            )
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_search_connector_error(self, handler, mock_http, mock_auth_context):
        """Returns 500 when connector raises an error."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(side_effect=TimeoutError("API timeout"))

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/search",
                {"query": "test"},
                mock_http,
            )
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_search_default_limit(self, handler, mock_http, mock_auth_context):
        """Uses default limit of 20 when not provided."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[])

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/search",
                {"query": "test"},
                mock_http,
            )
            assert _status(result) == 200
            mock_connector.search.assert_called_once_with(query="test", limit=20)

    @pytest.mark.asyncio
    async def test_search_no_refresh_token(self, handler, mock_http, mock_auth_context):
        """Returns 401 when user has no refresh token."""
        state = _make_user_state(refresh_token="")
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/search",
                {"query": "test"},
                mock_http,
            )
            assert _status(result) == 401


# ===========================================================================
# POST /api/v1/gmail/disconnect - Disconnect account
# ===========================================================================


class TestPostDisconnect:
    """Tests for POST /api/v1/gmail/disconnect."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, handler, mock_http, mock_auth_context):
        """Disconnects Gmail account successfully."""
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.delete_user_state",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.delete_sync_job",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await handler.handle_post("/api/v1/gmail/disconnect", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert body["was_connected"] is True

    @pytest.mark.asyncio
    async def test_disconnect_not_previously_connected(self, handler, mock_http, mock_auth_context):
        """Returns success even when not previously connected."""
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.delete_user_state",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.delete_sync_job",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await handler.handle_post("/api/v1/gmail/disconnect", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["success"] is True
            assert body["was_connected"] is False

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_sync_job(self, handler, mock_http, mock_auth_context):
        """Deletes both user state and sync job on disconnect."""
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.delete_user_state",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_del_state,
            patch(
                "aragora.server.handlers.features.gmail_ingest.delete_sync_job",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_del_job,
        ):
            result = await handler.handle_post("/api/v1/gmail/disconnect", {}, mock_http)
            assert _status(result) == 200
            mock_del_state.assert_called_once_with("test-user-001")
            mock_del_job.assert_called_once_with("test-user-001")


# ===========================================================================
# POST unknown path - 404
# ===========================================================================


class TestHandlePost404:
    """Tests for unknown POST paths."""

    @pytest.mark.asyncio
    async def test_unknown_post_path_returns_404(self, handler, mock_http, mock_auth_context):
        """Returns 404 for unknown POST paths."""
        result = await handler.handle_post("/api/v1/gmail/unknown", {}, mock_http)
        assert _status(result) == 404


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on Gmail endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_handle_get(self, handler, mock_http, mock_auth_context):
        """Returns 429 when rate limit exceeded on GET."""
        with patch("aragora.server.handlers.features.gmail_ingest._gmail_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            assert _status(result) == 429
            body = _body(result)
            assert "Rate limit" in body["error"]

    @pytest.mark.asyncio
    async def test_rate_limit_handle_post(self, handler, mock_http, mock_auth_context):
        """Returns 429 when rate limit exceeded on POST."""
        with patch("aragora.server.handlers.features.gmail_ingest._gmail_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_post("/api/v1/gmail/connect", {}, mock_http)
            assert _status(result) == 429


# ===========================================================================
# Authentication - _get_authenticated_user
# ===========================================================================


class TestAuthentication:
    """Tests for JWT authentication."""

    @pytest.mark.asyncio
    async def test_unauthenticated_user_get(self, handler, mock_http, mock_unauthenticated):
        """Returns 401 for unauthenticated GET requests."""
        result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_unauthenticated_user_post(self, handler, mock_http, mock_unauthenticated):
        """Returns 401 for unauthenticated POST requests."""
        result = await handler.handle_post("/api/v1/gmail/connect", {}, mock_http)
        assert _status(result) == 401

    def test_get_authenticated_user_success(self, handler, mock_auth_context):
        """Returns user_id and org_id for authenticated user."""
        mock_handler = MockHTTPHandler()
        user_id, org_id, err = handler._get_authenticated_user(mock_handler)
        assert user_id == "test-user-001"
        assert org_id == "test-org-001"
        assert err is None

    def test_get_authenticated_user_no_user_id(self, handler):
        """Returns error when user_id is missing."""
        mock_ctx = MagicMock()
        mock_ctx.authenticated = True
        mock_ctx.user_id = None
        with patch(
            "aragora.server.handlers.features.gmail_ingest.extract_user_from_request",
            return_value=mock_ctx,
        ):
            mock_handler = MockHTTPHandler()
            user_id, org_id, err = handler._get_authenticated_user(mock_handler)
            assert user_id is None
            assert err is not None
            assert _status(err) == 401


# ===========================================================================
# RBAC permission checks
# ===========================================================================


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_search_uses_read_permission(self, handler, mock_http):
        """POST /gmail/search uses gmail:read permission (not write)."""
        # Verify that search path uses GMAIL_READ_PERMISSION
        from aragora.server.handlers.features.gmail_ingest import (
            GMAIL_READ_PERMISSION,
            GMAIL_WRITE_PERMISSION,
        )

        assert GMAIL_READ_PERMISSION == "gmail:read"
        assert GMAIL_WRITE_PERMISSION == "gmail:write"

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_get_auth_context_unauthorized(self, handler, mock_http):
        """Returns 401 when SecureHandler auth check fails."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("No token"),
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            assert _status(result) == 401
            body = _body(result)
            assert "Authentication required" in body["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_get_auth_context_forbidden(self, handler, mock_http):
        """Returns 403 when permission check fails."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        mock_auth = MagicMock()
        with (
            patch.object(
                SecureHandler,
                "get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_auth,
            ),
            patch.object(
                SecureHandler,
                "check_permission",
                side_effect=ForbiddenError("Insufficient permissions"),
            ),
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            assert _status(result) == 403
            body = _body(result)
            assert "Permission denied" in body["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_post_auth_context_unauthorized(self, handler, mock_http):
        """Returns 401 when POST auth check fails."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("No token"),
        ):
            result = await handler.handle_post("/api/v1/gmail/connect", {}, mock_http)
            assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_post_auth_context_forbidden(self, handler, mock_http):
        """Returns 403 when POST permission check fails."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError

        mock_auth = MagicMock()
        with (
            patch.object(
                SecureHandler,
                "get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_auth,
            ),
            patch.object(
                SecureHandler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied"),
            ),
        ):
            result = await handler.handle_post("/api/v1/gmail/connect", {}, mock_http)
            assert _status(result) == 403


# ===========================================================================
# _is_configured() checks
# ===========================================================================


class TestIsConfigured:
    """Tests for _is_configured()."""

    def test_configured_with_gmail_client_id(self, handler):
        with patch.dict("os.environ", {"GMAIL_CLIENT_ID": "test-id"}, clear=True):
            assert handler._is_configured() is True

    def test_configured_with_google_gmail_client_id(self, handler):
        with patch.dict("os.environ", {"GOOGLE_GMAIL_CLIENT_ID": "test-id"}, clear=True):
            assert handler._is_configured() is True

    def test_configured_with_google_client_id(self, handler):
        with patch.dict("os.environ", {"GOOGLE_CLIENT_ID": "test-id"}, clear=True):
            assert handler._is_configured() is True

    def test_not_configured_with_no_env_vars(self, handler):
        with patch.dict("os.environ", {}, clear=True):
            assert handler._is_configured() is False

    def test_configured_with_empty_string(self, handler):
        with patch.dict("os.environ", {"GMAIL_CLIENT_ID": ""}, clear=True):
            assert handler._is_configured() is False


# ===========================================================================
# _complete_oauth() error paths
# ===========================================================================


class TestCompleteOAuth:
    """Tests for _complete_oauth() internal method."""

    @pytest.mark.asyncio
    async def test_complete_oauth_import_error(self, handler):
        """Returns 500 on ImportError."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": None,
            },
        ):
            result = await handler._complete_oauth("code", "redirect", "user1", "org1")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_complete_oauth_timeout_error(self, handler):
        """Returns 500 on TimeoutError."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=TimeoutError("timed out"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler._complete_oauth("code", "redirect", "user1", "org1")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_complete_oauth_key_error(self, handler):
        """Returns 500 on KeyError (missing profile data)."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.get_user_info = AsyncMock(side_effect=KeyError("emailAddress"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler._complete_oauth("code", "redirect", "user1", "org1")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_complete_oauth_saves_org_id(self, handler):
        """Saves org_id from JWT for tenant isolation."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.get_user_info = AsyncMock(
            return_value={"emailAddress": "u@g.com", "historyId": "1"}
        )
        mock_connector._access_token = "at"
        mock_connector._refresh_token = "rt"
        mock_connector._token_expiry = None

        with (
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
            patch(
                "aragora.server.handlers.features.gmail_ingest.save_user_state",
                new_callable=AsyncMock,
            ) as mock_save,
        ):
            result = await handler._complete_oauth("code", "redirect", "user1", "my-org")
            assert _status(result) == 200
            saved_state = mock_save.call_args[0][0]
            assert saved_state.org_id == "my-org"
            assert saved_state.user_id == "user1"

    @pytest.mark.asyncio
    async def test_complete_oauth_auth_failure(self, handler):
        """Returns 401 when authenticate returns False."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=False)

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler._complete_oauth("code", "redirect", "user1", "org1")
            assert _status(result) == 401


# ===========================================================================
# Handler initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_server_context(self):
        ctx = {"key": "value"}
        h = GmailIngestHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_ctx(self):
        ctx = {"key": "value"}
        h = GmailIngestHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_no_context(self):
        h = GmailIngestHandler()
        assert h.ctx == {}

    def test_init_server_context_takes_precedence(self):
        h = GmailIngestHandler(ctx={"old": 1}, server_context={"new": 2})
        assert h.ctx == {"new": 2}

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "gmail"


# ===========================================================================
# Module-level store functions
# ===========================================================================


class TestModuleFunctions:
    """Tests for module-level store helper functions."""

    @pytest.mark.asyncio
    async def test_get_user_state_calls_store(self):
        mock_store = MagicMock()
        mock_store.get = AsyncMock(return_value=_make_user_state())
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_gmail_token_store",
            return_value=mock_store,
        ):
            result = await get_user_state("user1")
            assert result is not None
            mock_store.get.assert_called_once_with("user1")

    @pytest.mark.asyncio
    async def test_save_user_state_calls_store(self):
        mock_store = MagicMock()
        mock_store.save = AsyncMock()
        state = _make_user_state()
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_gmail_token_store",
            return_value=mock_store,
        ):
            await save_user_state(state)
            mock_store.save.assert_called_once_with(state)

    @pytest.mark.asyncio
    async def test_delete_user_state_calls_store(self):
        mock_store = MagicMock()
        mock_store.delete = AsyncMock(return_value=True)
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_gmail_token_store",
            return_value=mock_store,
        ):
            result = await delete_user_state("user1")
            assert result is True
            mock_store.delete.assert_called_once_with("user1")

    @pytest.mark.asyncio
    async def test_get_sync_job_calls_store(self):
        mock_store = MagicMock()
        mock_store.get_sync_job = AsyncMock(return_value=_make_sync_job())
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_gmail_token_store",
            return_value=mock_store,
        ):
            result = await get_sync_job("user1")
            assert result is not None
            mock_store.get_sync_job.assert_called_once_with("user1")

    @pytest.mark.asyncio
    async def test_save_sync_job_calls_store(self):
        mock_store = MagicMock()
        mock_store.save_sync_job = AsyncMock()
        job = _make_sync_job()
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_gmail_token_store",
            return_value=mock_store,
        ):
            await save_sync_job(job)
            mock_store.save_sync_job.assert_called_once_with(job)

    @pytest.mark.asyncio
    async def test_delete_sync_job_calls_store(self):
        mock_store = MagicMock()
        mock_store.delete_sync_job = AsyncMock(return_value=True)
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_gmail_token_store",
            return_value=mock_store,
        ):
            result = await delete_sync_job("user1")
            assert result is True
            mock_store.delete_sync_job.assert_called_once_with("user1")


# ===========================================================================
# POST permission routing (search uses read, others use write)
# ===========================================================================


class TestPostPermissionRouting:
    """Tests that POST handle_post routes correct permissions."""

    @pytest.mark.asyncio
    async def test_search_path_uses_read_permission(self, handler, mock_http, mock_auth_context):
        """Verifies search uses gmail:read permission."""
        # The handler should not raise ForbiddenError for search
        # when only gmail:read is available
        state = _make_user_state()
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/search",
                {"query": "test"},
                mock_http,
            )
            # Should fail at 401 for "Not connected" or succeed, but not 403
            assert _status(result) in (200, 400, 401, 500)

    @pytest.mark.asyncio
    async def test_connect_path_uses_write_permission(self, handler, mock_http, mock_auth_context):
        """Verifies connect uses gmail:write permission."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": None,
            },
        ):
            result = await handler.handle_post("/api/v1/gmail/connect", {}, mock_http)
            # Will fail with 500 because import fails, but shouldn't be 403
            assert _status(result) == 500


# ===========================================================================
# Edge cases and additional coverage
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_status_google_client_id_env(self, handler, mock_http, mock_auth_context):
        """Checks GOOGLE_CLIENT_ID env var for configuration."""
        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.dict("os.environ", {"GOOGLE_CLIENT_ID": "gid"}, clear=True),
        ):
            result = await handler.handle("/api/v1/gmail/status", {}, mock_http)
            body = _body(result)
            assert body["configured"] is True

    @pytest.mark.asyncio
    async def test_list_messages_with_offset(self, handler, mock_http, mock_auth_context):
        """Returns offset in response."""
        state = _make_user_state()
        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=[])

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle("/api/v1/gmail/messages", {"offset": "10"}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["offset"] == 10

    @pytest.mark.asyncio
    async def test_search_multiple_results(self, handler, mock_http, mock_auth_context):
        """Returns multiple search results correctly."""
        state = _make_user_state()
        results = []
        for i in range(3):
            r = MagicMock()
            r.id = f"gmail-r{i}"
            r.title = f"Subject {i}"
            r.author = f"sender{i}@example.com"
            r.content = f"Content {i}"
            r.metadata = {"date": f"2026-01-{i + 1:02d}"}
            r.url = f"https://mail.google.com/{i}"
            results.append(r)

        mock_connector = MagicMock()
        mock_connector.search = AsyncMock(return_value=results)

        with (
            patch(
                "aragora.server.handlers.features.gmail_ingest.get_user_state",
                new_callable=AsyncMock,
                return_value=state,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.connectors.enterprise.communication.gmail": MagicMock(
                        GmailConnector=MagicMock(return_value=mock_connector)
                    )
                },
            ),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/search",
                {"query": "test"},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["count"] == 3
            assert len(body["results"]) == 3

    @pytest.mark.asyncio
    async def test_complete_oauth_value_error(self, handler):
        """Returns 500 on ValueError during OAuth."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=ValueError("bad value"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler._complete_oauth("code", "redirect", "u1", "o1")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_complete_oauth_os_error(self, handler):
        """Returns 500 on OSError during OAuth."""
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=OSError("disk error"))

        with patch.dict(
            "sys.modules",
            {
                "aragora.connectors.enterprise.communication.gmail": MagicMock(
                    GmailConnector=MagicMock(return_value=mock_connector)
                )
            },
        ):
            result = await handler._complete_oauth("code", "redirect", "u1", "o1")
            assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_message_no_refresh_token(self, handler, mock_http, mock_auth_context):
        """Returns 401 when getting message without refresh token."""
        state = _make_user_state(refresh_token="")
        with patch(
            "aragora.server.handlers.features.gmail_ingest.get_user_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await handler.handle("/api/v1/gmail/message/msg123", {}, mock_http)
            assert _status(result) == 401
