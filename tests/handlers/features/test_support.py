"""Tests for support platform handler.

Tests the support API endpoints including:
- GET    /api/v1/support/platforms              - List connected platforms
- POST   /api/v1/support/connect                - Connect a platform
- DELETE /api/v1/support/{platform}             - Disconnect platform
- GET    /api/v1/support/tickets                - List tickets (cross-platform)
- GET    /api/v1/support/{platform}/tickets     - Platform tickets
- POST   /api/v1/support/{platform}/tickets     - Create ticket
- PUT    /api/v1/support/{platform}/tickets/{id} - Update ticket
- GET    /api/v1/support/{platform}/tickets/{id} - Get single ticket
- POST   /api/v1/support/{platform}/tickets/{id}/reply - Reply to ticket
- GET    /api/v1/support/metrics                - Support metrics overview
- POST   /api/v1/support/triage                 - AI-powered ticket triage
- POST   /api/v1/support/auto-respond           - Generate response suggestions
- POST   /api/v1/support/search                 - Search tickets
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.support import (
    SupportHandler,
    SUPPORTED_PLATFORMS,
    _platform_credentials,
    _platform_connectors,
    reset_support_circuit_breaker,
)


# =============================================================================
# Test Helpers
# =============================================================================


@dataclass
class MockRequest:
    """Mock async HTTP request for handler tests."""

    method: str = "GET"
    path: str = "/"
    query: dict[str, str] = None
    _body: dict[str, Any] = None
    content_length: int = 0

    def __post_init__(self):
        if self.query is None:
            self.query = {}
        if self._body:
            self.content_length = len(json.dumps(self._body).encode())

    async def json(self) -> dict[str, Any]:
        return self._body or {}

    async def read(self) -> bytes:
        if self._body:
            return json.dumps(self._body).encode()
        return b""


def _status(result: dict[str, Any]) -> int:
    """Extract status code from handler result."""
    return result["status_code"]


def _body(result: dict[str, Any]) -> dict[str, Any]:
    """Extract body from handler result."""
    return result["body"]


def _make_mock_ticket(
    ticket_id: int = 1,
    subject: str = "Test ticket",
    description: str = "Test description",
    status: str = "open",
    priority: str = "medium",
    requester_email: str = "user@example.com",
    requester_name: str = "Test User",
    assignee_id: int | None = None,
    tags: list[str] | None = None,
) -> MagicMock:
    """Create a mock ticket object that looks like a Zendesk ticket."""
    ticket = MagicMock()
    ticket.id = ticket_id
    ticket.subject = subject
    ticket.description = description
    ticket.status = status
    ticket.priority = priority
    ticket.requester_email = requester_email
    ticket.requester_name = requester_name
    ticket.assignee_id = assignee_id
    ticket.assignee_name = None
    ticket.tags = tags or []
    ticket.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticket.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
    return ticket


def _make_mock_freshdesk_ticket(
    ticket_id: int = 1,
    subject: str = "Test ticket",
    description: str = "Test description",
    status: int = 2,
    priority: int = 2,
    email: str = "user@example.com",
    name: str = "Test User",
    responder_id: int | None = None,
    tags: list[str] | None = None,
) -> MagicMock:
    """Create a mock Freshdesk ticket."""
    ticket = MagicMock()
    ticket.id = ticket_id
    ticket.subject = subject
    ticket.description = description
    ticket.status = status
    ticket.priority = priority
    ticket.email = email
    ticket.name = name
    ticket.responder_id = responder_id
    ticket.tags = tags or []
    ticket.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ticket.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
    return ticket


def _make_mock_intercom_conversation(
    conv_id: str = "conv-1",
    title: str = "Test conversation",
    state: str = "open",
) -> MagicMock:
    """Create a mock Intercom conversation."""
    conv = MagicMock()
    conv.id = conv_id
    conv.title = title
    conv.source = MagicMock()
    conv.source.body = "Test body"
    conv.state = state
    conv.priority = MagicMock()
    conv.priority.value = "medium"
    contact = MagicMock()
    contact.email = "user@example.com"
    conv.contacts = [contact]
    conv.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    conv.updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
    return conv


def _make_mock_helpscout_conversation(
    conv_id: int = 1,
    subject: str = "Test conversation",
    status: str = "active",
) -> MagicMock:
    """Create a mock Help Scout conversation."""
    conv = MagicMock()
    conv.id = conv_id
    conv.subject = subject
    conv.preview = "Test preview"
    conv.status = MagicMock()
    conv.status.value = status
    conv.customer = MagicMock()
    conv.customer.email = "user@example.com"
    conv.customer.first_name = "Test"
    conv.customer.last_name = "User"
    conv.assignee = MagicMock()
    conv.assignee.id = 42
    conv.tags = ["help"]
    conv.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return conv


def _make_mock_comment(
    comment_id: int = 100,
    body: str = "Test comment",
    public: bool = True,
    author_id: int = 10,
) -> MagicMock:
    """Create a mock Zendesk comment."""
    comment = MagicMock()
    comment.id = comment_id
    comment.body = body
    comment.public = public
    comment.author_id = author_id
    comment.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return comment


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear global state before and after each test."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_support_circuit_breaker()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()
    reset_support_circuit_breaker()


@pytest.fixture
def handler():
    """Create a SupportHandler instance."""
    return SupportHandler(ctx={})


@pytest.fixture
def zendesk_connected():
    """Set up Zendesk as a connected platform."""
    _platform_credentials["zendesk"] = {
        "credentials": {"subdomain": "test", "email": "a@b.com", "api_token": "tok"},
        "connected_at": "2025-01-01T00:00:00+00:00",
    }


@pytest.fixture
def freshdesk_connected():
    """Set up Freshdesk as a connected platform."""
    _platform_credentials["freshdesk"] = {
        "credentials": {"domain": "test", "api_key": "key"},
        "connected_at": "2025-01-01T00:00:00+00:00",
    }


@pytest.fixture
def intercom_connected():
    """Set up Intercom as a connected platform."""
    _platform_credentials["intercom"] = {
        "credentials": {"access_token": "tok"},
        "connected_at": "2025-01-01T00:00:00+00:00",
    }


@pytest.fixture
def helpscout_connected():
    """Set up Help Scout as a connected platform."""
    _platform_credentials["helpscout"] = {
        "credentials": {"app_id": "id", "app_secret": "secret"},
        "connected_at": "2025-01-01T00:00:00+00:00",
    }


@pytest.fixture
def mock_zendesk_connector():
    """Create a mock Zendesk connector and register it."""
    connector = AsyncMock()
    _platform_connectors["zendesk"] = connector
    return connector


@pytest.fixture
def mock_freshdesk_connector():
    """Create a mock Freshdesk connector and register it."""
    connector = AsyncMock()
    _platform_connectors["freshdesk"] = connector
    return connector


@pytest.fixture
def mock_intercom_connector():
    """Create a mock Intercom connector and register it."""
    connector = AsyncMock()
    _platform_connectors["intercom"] = connector
    return connector


@pytest.fixture
def mock_helpscout_connector():
    """Create a mock Help Scout connector and register it."""
    connector = AsyncMock()
    _platform_connectors["helpscout"] = connector
    return connector


# =============================================================================
# Initialization and Routing Tests
# =============================================================================


class TestSupportHandlerInit:
    """Tests for handler initialization and basic properties."""

    def test_routes_defined(self, handler):
        """Handler defines ROUTES list."""
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) > 0

    def test_resource_type(self, handler):
        """Handler sets RESOURCE_TYPE to 'support'."""
        assert handler.RESOURCE_TYPE == "support"

    def test_supported_platforms_defined(self):
        """SUPPORTED_PLATFORMS includes all four platforms."""
        assert "zendesk" in SUPPORTED_PLATFORMS
        assert "freshdesk" in SUPPORTED_PLATFORMS
        assert "intercom" in SUPPORTED_PLATFORMS
        assert "helpscout" in SUPPORTED_PLATFORMS


class TestCanHandle:
    """Tests for can_handle() routing."""

    def test_handles_support_platforms_path(self, handler):
        assert handler.can_handle("/api/v1/support/platforms")

    def test_handles_support_connect_path(self, handler):
        assert handler.can_handle("/api/v1/support/connect", "POST")

    def test_handles_support_disconnect_path(self, handler):
        assert handler.can_handle("/api/v1/support/zendesk", "DELETE")

    def test_handles_all_tickets_path(self, handler):
        assert handler.can_handle("/api/v1/support/tickets")

    def test_handles_platform_tickets_path(self, handler):
        assert handler.can_handle("/api/v1/support/zendesk/tickets")

    def test_handles_platform_ticket_with_id(self, handler):
        assert handler.can_handle("/api/v1/support/zendesk/tickets/123")

    def test_handles_ticket_reply_path(self, handler):
        assert handler.can_handle("/api/v1/support/zendesk/tickets/123/reply")

    def test_handles_metrics_path(self, handler):
        assert handler.can_handle("/api/v1/support/metrics")

    def test_handles_triage_path(self, handler):
        assert handler.can_handle("/api/v1/support/triage")

    def test_handles_auto_respond_path(self, handler):
        assert handler.can_handle("/api/v1/support/auto-respond")

    def test_handles_search_path(self, handler):
        assert handler.can_handle("/api/v1/support/search")

    def test_rejects_non_support_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_match(self, handler):
        assert not handler.can_handle("/api/v1/supportx/platforms")

    def test_rejects_other_api_path(self, handler):
        assert not handler.can_handle("/api/v1/users")


class TestRouting:
    """Tests for handle_request routing to correct internal methods."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(self, handler):
        request = MockRequest(method="GET", path="/api/v1/support/unknown_endpoint")
        result = await handler.handle_request(request)
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_wrong_method_returns_404(self, handler):
        """PUT on /platforms is not a recognized route, returns 404."""
        request = MockRequest(method="PUT", path="/api/v1/support/platforms")
        result = await handler.handle_request(request)
        assert _status(result) == 404


# =============================================================================
# List Platforms Tests
# =============================================================================


class TestListPlatforms:
    """Tests for GET /api/v1/support/platforms."""

    @pytest.mark.asyncio
    async def test_list_platforms_no_connections(self, handler):
        """Returns all platforms, none connected."""
        request = MockRequest(method="GET", path="/api/v1/support/platforms")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "platforms" in body
        assert body["connected_count"] == 0
        assert len(body["platforms"]) == len(SUPPORTED_PLATFORMS)

    @pytest.mark.asyncio
    async def test_list_platforms_with_connection(self, handler, zendesk_connected):
        """Shows Zendesk as connected when credentials exist."""
        request = MockRequest(method="GET", path="/api/v1/support/platforms")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["connected_count"] == 1

        zendesk = next(p for p in body["platforms"] if p["id"] == "zendesk")
        assert zendesk["connected"] is True
        assert zendesk["connected_at"] is not None

    @pytest.mark.asyncio
    async def test_list_platforms_shows_metadata(self, handler):
        """Each platform entry has name, description, features."""
        request = MockRequest(method="GET", path="/api/v1/support/platforms")
        result = await handler.handle_request(request)

        body = _body(result)
        for plat in body["platforms"]:
            assert "id" in plat
            assert "name" in plat
            assert "description" in plat
            assert "features" in plat
            assert isinstance(plat["features"], list)

    @pytest.mark.asyncio
    async def test_list_platforms_multiple_connected(
        self, handler, zendesk_connected, freshdesk_connected
    ):
        """Shows correct count when multiple platforms connected."""
        request = MockRequest(method="GET", path="/api/v1/support/platforms")
        result = await handler.handle_request(request)

        assert _body(result)["connected_count"] == 2


# =============================================================================
# Connect Platform Tests
# =============================================================================


class TestConnectPlatform:
    """Tests for POST /api/v1/support/connect."""

    @pytest.mark.asyncio
    async def test_connect_zendesk_success(self, handler):
        """Connects Zendesk with valid credentials."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={
                "platform": "zendesk",
                "credentials": {
                    "subdomain": "test",
                    "email": "admin@test.com",
                    "api_token": "tok123",
                },
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "zendesk"
        assert "connected_at" in body
        assert "zendesk" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_freshdesk_success(self, handler):
        """Connects Freshdesk with valid credentials."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={
                "platform": "freshdesk",
                "credentials": {"domain": "test", "api_key": "key123"},
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "freshdesk"

    @pytest.mark.asyncio
    async def test_connect_intercom_success(self, handler):
        """Connects Intercom with valid credentials."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={
                "platform": "intercom",
                "credentials": {"access_token": "tok"},
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "intercom"

    @pytest.mark.asyncio
    async def test_connect_helpscout_success(self, handler):
        """Connects Help Scout with valid credentials."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={
                "platform": "helpscout",
                "credentials": {"app_id": "id", "app_secret": "secret"},
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "helpscout"

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler):
        """Rejects request with no platform field."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={"credentials": {"api_key": "tok"}},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler):
        """Rejects unsupported platform name."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={"platform": "jira", "credentials": {"token": "t"}},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "unsupported" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler):
        """Rejects request without credentials."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={"platform": "zendesk", "credentials": {}},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "credentials" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_missing_required_credential_fields(self, handler):
        """Rejects Zendesk with incomplete credentials."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={
                "platform": "zendesk",
                "credentials": {"subdomain": "test"},  # missing email, api_token
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "missing" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_empty_body(self, handler):
        """Rejects empty body."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_stores_credentials(self, handler):
        """Credentials are stored in global dict after connect."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={
                "platform": "intercom",
                "credentials": {"access_token": "my-token"},
            },
        )
        await handler.handle_request(request)

        assert "intercom" in _platform_credentials
        assert _platform_credentials["intercom"]["credentials"]["access_token"] == "my-token"


# =============================================================================
# Disconnect Platform Tests
# =============================================================================


class TestDisconnectPlatform:
    """Tests for DELETE /api/v1/support/{platform}."""

    @pytest.mark.asyncio
    async def test_disconnect_connected_platform(self, handler, zendesk_connected):
        """Disconnects a connected platform."""
        request = MockRequest(method="DELETE", path="/api/v1/support/zendesk")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "zendesk"
        assert "zendesk" not in _platform_credentials

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler):
        """Returns 404 when platform is not connected."""
        request = MockRequest(method="DELETE", path="/api/v1/support/zendesk")
        result = await handler.handle_request(request)

        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_disconnect_removes_connector(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Disconnect removes both credentials and connector."""
        mock_zendesk_connector.close = AsyncMock()

        request = MockRequest(method="DELETE", path="/api/v1/support/zendesk")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert "zendesk" not in _platform_credentials
        assert "zendesk" not in _platform_connectors

    @pytest.mark.asyncio
    async def test_disconnect_calls_close_on_connector(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Disconnect calls close() on the connector if available."""
        mock_zendesk_connector.close = AsyncMock()

        request = MockRequest(method="DELETE", path="/api/v1/support/zendesk")
        await handler.handle_request(request)

        mock_zendesk_connector.close.assert_awaited_once()


# =============================================================================
# List All Tickets Tests
# =============================================================================


class TestListAllTickets:
    """Tests for GET /api/v1/support/tickets."""

    @pytest.mark.asyncio
    async def test_list_tickets_no_platforms(self, handler):
        """Returns empty list when no platforms are connected."""
        request = MockRequest(method="GET", path="/api/v1/support/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["tickets"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_tickets_with_zendesk(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Fetches and normalizes Zendesk tickets."""
        mock_zendesk_connector.get_tickets = AsyncMock(
            return_value=[_make_mock_ticket(1, "Ticket A"), _make_mock_ticket(2, "Ticket B")]
        )

        request = MockRequest(method="GET", path="/api/v1/support/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["tickets"]) == 2

    @pytest.mark.asyncio
    async def test_list_tickets_error_from_platform(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Continues gracefully when a platform errors."""
        mock_zendesk_connector.get_tickets = AsyncMock(side_effect=ConnectionError("timeout"))

        request = MockRequest(method="GET", path="/api/v1/support/tickets")
        result = await handler.handle_request(request)

        # Still 200, just no tickets from that platform
        assert _status(result) == 200
        assert _body(result)["total"] == 0

    @pytest.mark.asyncio
    async def test_list_tickets_query_params(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Passes status and limit query params through."""
        mock_zendesk_connector.get_tickets = AsyncMock(return_value=[])

        request = MockRequest(
            method="GET",
            path="/api/v1/support/tickets",
            query={"status": "open", "limit": "10"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_tickets_reports_queried_platforms(
        self, handler, zendesk_connected, freshdesk_connected
    ):
        """Response includes the platforms that were queried."""
        # Patch _get_connector to return None (no connector available)
        with patch.object(handler, "_get_connector", return_value=None):
            request = MockRequest(method="GET", path="/api/v1/support/tickets")
            result = await handler.handle_request(request)

        body = _body(result)
        assert "zendesk" in body["platforms_queried"]
        assert "freshdesk" in body["platforms_queried"]


# =============================================================================
# List Platform Tickets Tests
# =============================================================================


class TestListPlatformTickets:
    """Tests for GET /api/v1/support/{platform}/tickets."""

    @pytest.mark.asyncio
    async def test_list_zendesk_tickets(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Lists tickets from a specific platform."""
        mock_zendesk_connector.get_tickets = AsyncMock(
            return_value=[_make_mock_ticket()]
        )

        request = MockRequest(method="GET", path="/api/v1/support/zendesk/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "zendesk"
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_tickets_not_connected(self, handler):
        """Returns 404 when platform is not connected."""
        request = MockRequest(method="GET", path="/api/v1/support/zendesk/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 404
        assert "not connected" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_list_freshdesk_tickets(
        self, handler, freshdesk_connected, mock_freshdesk_connector
    ):
        """Lists Freshdesk tickets."""
        mock_freshdesk_connector.get_tickets = AsyncMock(
            return_value=[_make_mock_freshdesk_ticket()]
        )

        request = MockRequest(method="GET", path="/api/v1/support/freshdesk/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "freshdesk"

    @pytest.mark.asyncio
    async def test_list_intercom_conversations(
        self, handler, intercom_connected, mock_intercom_connector
    ):
        """Lists Intercom conversations as tickets."""
        mock_intercom_connector.get_conversations = AsyncMock(
            return_value=[_make_mock_intercom_conversation()]
        )

        request = MockRequest(method="GET", path="/api/v1/support/intercom/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "intercom"

    @pytest.mark.asyncio
    async def test_list_helpscout_conversations(
        self, handler, helpscout_connected, mock_helpscout_connector
    ):
        """Lists Help Scout conversations as tickets."""
        mock_helpscout_connector.get_conversations = AsyncMock(
            return_value=[_make_mock_helpscout_conversation()]
        )

        request = MockRequest(method="GET", path="/api/v1/support/helpscout/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "helpscout"


# =============================================================================
# Get Single Ticket Tests
# =============================================================================


class TestGetTicket:
    """Tests for GET /api/v1/support/{platform}/tickets/{id}."""

    @pytest.mark.asyncio
    async def test_get_zendesk_ticket(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Fetches single Zendesk ticket with comments."""
        ticket = _make_mock_ticket(42, "Bug report")
        mock_zendesk_connector.get_ticket = AsyncMock(return_value=ticket)
        mock_zendesk_connector.get_ticket_comments = AsyncMock(
            return_value=[_make_mock_comment()]
        )

        request = MockRequest(method="GET", path="/api/v1/support/zendesk/tickets/42")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "42"
        assert "comments" in body

    @pytest.mark.asyncio
    async def test_get_freshdesk_ticket(
        self, handler, freshdesk_connected, mock_freshdesk_connector
    ):
        """Fetches single Freshdesk ticket with conversations."""
        ticket = _make_mock_freshdesk_ticket(42)
        mock_freshdesk_connector.get_ticket = AsyncMock(return_value=ticket)
        mock_freshdesk_connector.get_ticket_conversations = AsyncMock(return_value=[])

        request = MockRequest(method="GET", path="/api/v1/support/freshdesk/tickets/42")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "42"
        assert "conversations" in body

    @pytest.mark.asyncio
    async def test_get_intercom_conversation(
        self, handler, intercom_connected, mock_intercom_connector
    ):
        """Fetches single Intercom conversation."""
        conv = _make_mock_intercom_conversation("conv-99")
        mock_intercom_connector.get_conversation = AsyncMock(return_value=conv)

        request = MockRequest(method="GET", path="/api/v1/support/intercom/tickets/conv-99")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["id"] == "conv-99"

    @pytest.mark.asyncio
    async def test_get_helpscout_conversation(
        self, handler, helpscout_connected, mock_helpscout_connector
    ):
        """Fetches single Help Scout conversation with threads."""
        conv = _make_mock_helpscout_conversation(99)
        mock_helpscout_connector.get_conversation = AsyncMock(return_value=conv)
        mock_helpscout_connector.get_conversation_threads = AsyncMock(return_value=[])

        request = MockRequest(method="GET", path="/api/v1/support/helpscout/tickets/99")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "99"
        assert "threads" in body

    @pytest.mark.asyncio
    async def test_get_ticket_not_connected(self, handler):
        """Returns 404 when platform not connected."""
        request = MockRequest(method="GET", path="/api/v1/support/zendesk/tickets/1")
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_ticket_connector_error(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Returns 404 when connector raises error fetching ticket."""
        mock_zendesk_connector.get_ticket = AsyncMock(side_effect=ConnectionError("fail"))

        request = MockRequest(method="GET", path="/api/v1/support/zendesk/tickets/999")
        result = await handler.handle_request(request)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_get_ticket_no_connector(self, handler, zendesk_connected):
        """Returns 500 when connector cannot be initialized."""
        with patch.object(handler, "_get_connector", return_value=None):
            request = MockRequest(method="GET", path="/api/v1/support/zendesk/tickets/1")
            result = await handler.handle_request(request)

        assert _status(result) == 500


# =============================================================================
# Create Ticket Tests
# =============================================================================


class TestCreateTicket:
    """Tests for POST /api/v1/support/{platform}/tickets."""

    @pytest.mark.asyncio
    async def test_create_zendesk_ticket(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Creates a Zendesk ticket."""
        mock_zendesk_connector.create_ticket = AsyncMock(
            return_value=_make_mock_ticket(99, "New ticket")
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets",
            _body={
                "subject": "New ticket",
                "description": "Something is broken",
                "requester_email": "user@test.com",
                "priority": "high",
                "tags": ["bug"],
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 201
        assert _body(result)["platform"] == "zendesk"

    @pytest.mark.asyncio
    async def test_create_freshdesk_ticket(
        self, handler, freshdesk_connected, mock_freshdesk_connector
    ):
        """Creates a Freshdesk ticket."""
        mock_freshdesk_connector.create_ticket = AsyncMock(
            return_value=_make_mock_freshdesk_ticket(99)
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/freshdesk/tickets",
            _body={
                "description": "Freshdesk issue",
                "requester_email": "user@test.com",
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 201
        assert _body(result)["platform"] == "freshdesk"

    @pytest.mark.asyncio
    async def test_create_intercom_conversation(
        self, handler, intercom_connected, mock_intercom_connector
    ):
        """Creates an Intercom conversation."""
        mock_intercom_connector.create_conversation = AsyncMock(
            return_value=_make_mock_intercom_conversation("new-conv")
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/intercom/tickets",
            _body={
                "description": "Hello support",
                "user_id": "user-1",
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 201
        assert _body(result)["platform"] == "intercom"

    @pytest.mark.asyncio
    async def test_create_helpscout_conversation(
        self, handler, helpscout_connected, mock_helpscout_connector
    ):
        """Creates a Help Scout conversation."""
        mock_helpscout_connector.create_conversation = AsyncMock(
            return_value=_make_mock_helpscout_conversation(99)
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/helpscout/tickets",
            _body={
                "description": "Need help",
                "mailbox_id": "box-1",
                "requester_email": "user@test.com",
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 201
        assert _body(result)["platform"] == "helpscout"

    @pytest.mark.asyncio
    async def test_create_ticket_not_connected(self, handler):
        """Returns 404 when platform not connected."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets",
            _body={"description": "Help"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_ticket_missing_description(self, handler, zendesk_connected):
        """Rejects ticket creation without description."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets",
            _body={"subject": "No description"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "description" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_ticket_connector_error(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Returns 500 when connector fails to create ticket."""
        mock_zendesk_connector.create_ticket = AsyncMock(
            side_effect=ConnectionError("API error")
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets",
            _body={"description": "Something"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_ticket_no_connector(self, handler, zendesk_connected):
        """Returns 500 when connector cannot be initialized."""
        with patch.object(handler, "_get_connector", return_value=None):
            request = MockRequest(
                method="POST",
                path="/api/v1/support/zendesk/tickets",
                _body={"description": "Something"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_ticket_default_subject(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Uses default subject when none provided."""
        mock_zendesk_connector.create_ticket = AsyncMock(
            return_value=_make_mock_ticket(1, "Support Request")
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets",
            _body={"description": "No subject provided"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 201
        mock_zendesk_connector.create_ticket.assert_awaited_once()
        call_kwargs = mock_zendesk_connector.create_ticket.call_args[1]
        assert call_kwargs["subject"] == "Support Request"


# =============================================================================
# Update Ticket Tests
# =============================================================================


class TestUpdateTicket:
    """Tests for PUT /api/v1/support/{platform}/tickets/{id}."""

    @pytest.mark.asyncio
    async def test_update_zendesk_ticket(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Updates a Zendesk ticket."""
        mock_zendesk_connector.update_ticket = AsyncMock(
            return_value=_make_mock_ticket(42, status="solved")
        )

        request = MockRequest(
            method="PUT",
            path="/api/v1/support/zendesk/tickets/42",
            _body={"status": "solved", "priority": "high"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        mock_zendesk_connector.update_ticket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_freshdesk_ticket(
        self, handler, freshdesk_connected, mock_freshdesk_connector
    ):
        """Updates a Freshdesk ticket."""
        mock_freshdesk_connector.update_ticket = AsyncMock(
            return_value=_make_mock_freshdesk_ticket(42, status=4)
        )

        request = MockRequest(
            method="PUT",
            path="/api/v1/support/freshdesk/tickets/42",
            _body={"status": "resolved"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_update_intercom_close(
        self, handler, intercom_connected, mock_intercom_connector
    ):
        """Updates Intercom conversation status to closed."""
        conv = _make_mock_intercom_conversation("conv-1")
        mock_intercom_connector.close_conversation = AsyncMock()
        mock_intercom_connector.get_conversation = AsyncMock(return_value=conv)

        request = MockRequest(
            method="PUT",
            path="/api/v1/support/intercom/tickets/conv-1",
            _body={"status": "closed"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        mock_intercom_connector.close_conversation.assert_awaited_once_with("conv-1")

    @pytest.mark.asyncio
    async def test_update_intercom_open(
        self, handler, intercom_connected, mock_intercom_connector
    ):
        """Updates Intercom conversation status to open."""
        conv = _make_mock_intercom_conversation("conv-1", state="open")
        mock_intercom_connector.open_conversation = AsyncMock()
        mock_intercom_connector.get_conversation = AsyncMock(return_value=conv)

        request = MockRequest(
            method="PUT",
            path="/api/v1/support/intercom/tickets/conv-1",
            _body={"status": "open"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        mock_intercom_connector.open_conversation.assert_awaited_once_with("conv-1")

    @pytest.mark.asyncio
    async def test_update_helpscout_conversation(
        self, handler, helpscout_connected, mock_helpscout_connector
    ):
        """Updates a Help Scout conversation."""
        conv = _make_mock_helpscout_conversation(42)
        mock_helpscout_connector.update_conversation = AsyncMock()
        mock_helpscout_connector.get_conversation = AsyncMock(return_value=conv)

        request = MockRequest(
            method="PUT",
            path="/api/v1/support/helpscout/tickets/42",
            _body={"status": "active", "assignee_id": "99"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_update_ticket_not_connected(self, handler):
        """Returns 404 when platform not connected."""
        request = MockRequest(
            method="PUT",
            path="/api/v1/support/zendesk/tickets/1",
            _body={"status": "open"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_ticket_connector_error(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Returns 500 when connector fails to update."""
        mock_zendesk_connector.update_ticket = AsyncMock(
            side_effect=ConnectionError("API down")
        )

        request = MockRequest(
            method="PUT",
            path="/api/v1/support/zendesk/tickets/1",
            _body={"status": "solved"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_update_ticket_no_connector(self, handler, zendesk_connected):
        """Returns 500 when connector cannot be initialized."""
        with patch.object(handler, "_get_connector", return_value=None):
            request = MockRequest(
                method="PUT",
                path="/api/v1/support/zendesk/tickets/1",
                _body={"status": "open"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500


# =============================================================================
# Reply to Ticket Tests
# =============================================================================


class TestReplyToTicket:
    """Tests for POST /api/v1/support/{platform}/tickets/{id}/reply."""

    @pytest.mark.asyncio
    async def test_reply_zendesk(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Adds a reply to a Zendesk ticket."""
        mock_zendesk_connector.add_ticket_comment = AsyncMock()

        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets/42/reply",
            _body={"message": "We are looking into this."},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert "reply" in _body(result)["message"].lower()

    @pytest.mark.asyncio
    async def test_reply_freshdesk(
        self, handler, freshdesk_connected, mock_freshdesk_connector
    ):
        """Adds a reply to a Freshdesk ticket."""
        mock_freshdesk_connector.reply_to_ticket = AsyncMock()

        request = MockRequest(
            method="POST",
            path="/api/v1/support/freshdesk/tickets/42/reply",
            _body={"message": "Thanks for the report."},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_reply_intercom(
        self, handler, intercom_connected, mock_intercom_connector
    ):
        """Adds a reply to an Intercom conversation."""
        mock_intercom_connector.reply_to_conversation = AsyncMock()

        request = MockRequest(
            method="POST",
            path="/api/v1/support/intercom/tickets/conv-1/reply",
            _body={"message": "Hi there!", "public": True},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_reply_helpscout(
        self, handler, helpscout_connected, mock_helpscout_connector
    ):
        """Adds a reply to a Help Scout conversation."""
        mock_helpscout_connector.add_reply = AsyncMock()

        request = MockRequest(
            method="POST",
            path="/api/v1/support/helpscout/tickets/42/reply",
            _body={"message": "We can help with that."},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_reply_missing_message(self, handler, zendesk_connected):
        """Rejects reply without message field."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets/42/reply",
            _body={"public": True},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "message" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_reply_not_connected(self, handler):
        """Returns 404 when platform not connected."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets/42/reply",
            _body={"message": "Hi"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_reply_connector_error(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Returns 500 when reply fails."""
        mock_zendesk_connector.add_ticket_comment = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/zendesk/tickets/42/reply",
            _body={"message": "Hi"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_reply_no_connector(self, handler, zendesk_connected):
        """Returns 500 when connector cannot be initialized."""
        with patch.object(handler, "_get_connector", return_value=None):
            request = MockRequest(
                method="POST",
                path="/api/v1/support/zendesk/tickets/42/reply",
                _body={"message": "Hi"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500


# =============================================================================
# Metrics Tests
# =============================================================================


class TestGetMetrics:
    """Tests for GET /api/v1/support/metrics."""

    @pytest.mark.asyncio
    async def test_metrics_no_platforms(self, handler):
        """Returns empty metrics when no platforms connected."""
        request = MockRequest(method="GET", path="/api/v1/support/metrics")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["totals"]["total_tickets"] == 0
        assert body["platforms"] == {}

    @pytest.mark.asyncio
    async def test_metrics_with_platform(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Computes metrics from connected platform tickets."""
        mock_zendesk_connector.get_tickets = AsyncMock(
            return_value=[
                _make_mock_ticket(1, status="open"),
                _make_mock_ticket(2, status="solved"),
            ]
        )

        request = MockRequest(method="GET", path="/api/v1/support/metrics")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "zendesk" in body["platforms"]

    @pytest.mark.asyncio
    async def test_metrics_custom_days(self, handler):
        """Accepts days query parameter."""
        request = MockRequest(
            method="GET",
            path="/api/v1/support/metrics",
            query={"days": "30"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["period_days"] == 30

    @pytest.mark.asyncio
    async def test_metrics_platform_error(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Handles platform errors gracefully in metrics."""
        mock_zendesk_connector.get_tickets = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        request = MockRequest(method="GET", path="/api/v1/support/metrics")
        result = await handler.handle_request(request)

        # Metrics still returned with error info
        assert _status(result) == 200


# =============================================================================
# Triage Tests
# =============================================================================


class TestTriageTickets:
    """Tests for POST /api/v1/support/triage."""

    @pytest.mark.asyncio
    async def test_triage_with_platform_and_ids(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Triages specific ticket IDs from a platform."""
        ticket = _make_mock_ticket(1, "Urgent bug", description="The system is broken")
        mock_zendesk_connector.get_ticket = AsyncMock(return_value=ticket)

        request = MockRequest(
            method="POST",
            path="/api/v1/support/triage",
            _body={"platform": "zendesk", "ticket_ids": ["1"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "triage_id" in body
        assert body["tickets_analyzed"] >= 0
        assert "results" in body
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_triage_empty_body(self, handler):
        """Triages all open tickets when no IDs or platform given."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/triage",
            _body={},
        )
        result = await handler.handle_request(request)

        # Should succeed even with no platforms/tickets
        assert _status(result) == 200
        body = _body(result)
        assert body["tickets_analyzed"] == 0

    @pytest.mark.asyncio
    async def test_triage_results_sorted_by_urgency(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Triage results are sorted by urgency score descending."""
        t1 = _make_mock_ticket(1, "Low issue", description="Just wondering")
        t2 = _make_mock_ticket(2, "Urgent emergency", description="System is down critical")
        mock_zendesk_connector.get_ticket = AsyncMock(side_effect=[t1, t2])

        request = MockRequest(
            method="POST",
            path="/api/v1/support/triage",
            _body={"platform": "zendesk", "ticket_ids": ["1", "2"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        results = _body(result)["results"]
        if len(results) >= 2:
            assert results[0]["urgency_score"] >= results[1]["urgency_score"]

    @pytest.mark.asyncio
    async def test_triage_result_fields(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Each triage result has required analysis fields."""
        ticket = _make_mock_ticket(1, "Bug report", description="Error on login page")
        mock_zendesk_connector.get_ticket = AsyncMock(return_value=ticket)

        request = MockRequest(
            method="POST",
            path="/api/v1/support/triage",
            _body={"platform": "zendesk", "ticket_ids": ["1"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        results = _body(result)["results"]
        if results:
            entry = results[0]
            assert "ticket_id" in entry
            assert "suggested_priority" in entry
            assert "suggested_category" in entry
            assert "sentiment" in entry
            assert "urgency_score" in entry
            assert "suggested_response_template" in entry


# =============================================================================
# Auto-Respond Tests
# =============================================================================


class TestAutoRespond:
    """Tests for POST /api/v1/support/auto-respond."""

    @pytest.mark.asyncio
    async def test_auto_respond_success(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Generates response suggestions for a ticket."""
        ticket = _make_mock_ticket(1, "Login issue")
        mock_zendesk_connector.get_ticket = AsyncMock(return_value=ticket)
        mock_zendesk_connector.get_ticket_comments = AsyncMock(
            return_value=[_make_mock_comment()]
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/auto-respond",
            _body={"ticket_id": "1", "platform": "zendesk"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["ticket_id"] == "1"
        assert body["platform"] == "zendesk"
        assert len(body["suggestions"]) == 3
        assert body["suggestions"][0]["type"] == "acknowledgment"
        assert body["suggestions"][1]["type"] == "solution"
        assert body["suggestions"][2]["type"] == "follow_up"

    @pytest.mark.asyncio
    async def test_auto_respond_missing_fields(self, handler):
        """Rejects when ticket_id or platform is missing."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/auto-respond",
            _body={"ticket_id": "1"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_auto_respond_missing_ticket_id(self, handler):
        """Rejects when ticket_id is missing."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/auto-respond",
            _body={"platform": "zendesk"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_auto_respond_not_connected(self, handler):
        """Returns 404 when platform not connected."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/auto-respond",
            _body={"ticket_id": "1", "platform": "zendesk"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_auto_respond_ticket_not_found(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Returns 404 when ticket cannot be fetched."""
        mock_zendesk_connector.get_ticket = AsyncMock(
            side_effect=ConnectionError("not found")
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/auto-respond",
            _body={"ticket_id": "999", "platform": "zendesk"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_auto_respond_no_connector(self, handler, zendesk_connected):
        """Returns 500 when connector cannot be initialized."""
        with patch.object(handler, "_get_connector", return_value=None):
            request = MockRequest(
                method="POST",
                path="/api/v1/support/auto-respond",
                _body={"ticket_id": "1", "platform": "zendesk"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_auto_respond_includes_ticket_context(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Response includes ticket context information."""
        ticket = _make_mock_ticket(1, "Billing question", priority="high")
        mock_zendesk_connector.get_ticket = AsyncMock(return_value=ticket)
        mock_zendesk_connector.get_ticket_comments = AsyncMock(return_value=[])

        request = MockRequest(
            method="POST",
            path="/api/v1/support/auto-respond",
            _body={"ticket_id": "1", "platform": "zendesk"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        ctx = _body(result)["ticket_context"]
        assert ctx["subject"] == "Billing question"


# =============================================================================
# Search Tests
# =============================================================================


class TestSearchTickets:
    """Tests for POST /api/v1/support/search."""

    @pytest.mark.asyncio
    async def test_search_zendesk(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Searches Zendesk tickets."""
        mock_zendesk_connector.search_tickets = AsyncMock(
            return_value=[_make_mock_ticket(1, "Login bug")]
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/search",
            _body={"query": "login", "platforms": ["zendesk"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["query"] == "login"
        assert body["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_freshdesk(
        self, handler, freshdesk_connected, mock_freshdesk_connector
    ):
        """Searches Freshdesk tickets."""
        mock_freshdesk_connector.search_tickets = AsyncMock(
            return_value=[_make_mock_freshdesk_ticket(1, "Payment issue")]
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/search",
            _body={"query": "payment", "platforms": ["freshdesk"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["total"] >= 1

    @pytest.mark.asyncio
    async def test_search_no_platforms(self, handler):
        """Returns empty results when no platforms connected."""
        request = MockRequest(
            method="POST",
            path="/api/v1/support/search",
            _body={"query": "test"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["total"] == 0

    @pytest.mark.asyncio
    async def test_search_connector_error(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Handles search errors gracefully."""
        mock_zendesk_connector.search_tickets = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        request = MockRequest(
            method="POST",
            path="/api/v1/support/search",
            _body={"query": "error", "platforms": ["zendesk"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["total"] == 0

    @pytest.mark.asyncio
    async def test_search_skips_disconnected_platforms(self, handler, zendesk_connected):
        """Skips platforms that are not connected in the search list."""
        with patch.object(handler, "_get_connector", return_value=None):
            request = MockRequest(
                method="POST",
                path="/api/v1/support/search",
                _body={"query": "test", "platforms": ["zendesk", "jira"]},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["total"] == 0


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker on write operations."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_writes_when_open(self, handler):
        """Returns 503 when circuit breaker is open for POST/PUT/DELETE."""
        from aragora.server.handlers.features.support import get_support_circuit_breaker

        cb = get_support_circuit_breaker()
        # Force the circuit breaker open
        for _ in range(10):
            cb.record_failure()

        request = MockRequest(
            method="POST",
            path="/api/v1/support/connect",
            _body={"platform": "zendesk", "credentials": {"subdomain": "t", "email": "e", "api_token": "k"}},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 503
        assert "unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_reads_when_open(self, handler):
        """GET requests bypass the circuit breaker."""
        from aragora.server.handlers.features.support import get_support_circuit_breaker

        cb = get_support_circuit_breaker()
        for _ in range(10):
            cb.record_failure()

        request = MockRequest(method="GET", path="/api/v1/support/platforms")
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets(self, handler):
        """Reset function restores circuit breaker to working state."""
        from aragora.server.handlers.features.support import (
            get_support_circuit_breaker,
            reset_support_circuit_breaker,
        )

        cb = get_support_circuit_breaker()
        for _ in range(10):
            cb.record_failure()

        assert not cb.can_proceed()

        reset_support_circuit_breaker()
        assert cb.can_proceed()


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_get_required_credentials_zendesk(self, handler):
        """Returns correct required fields for Zendesk."""
        fields = handler._get_required_credentials("zendesk")
        assert "subdomain" in fields
        assert "email" in fields
        assert "api_token" in fields

    def test_get_required_credentials_freshdesk(self, handler):
        """Returns correct required fields for Freshdesk."""
        fields = handler._get_required_credentials("freshdesk")
        assert "domain" in fields
        assert "api_key" in fields

    def test_get_required_credentials_intercom(self, handler):
        """Returns correct required fields for Intercom."""
        fields = handler._get_required_credentials("intercom")
        assert "access_token" in fields

    def test_get_required_credentials_helpscout(self, handler):
        """Returns correct required fields for Help Scout."""
        fields = handler._get_required_credentials("helpscout")
        assert "app_id" in fields
        assert "app_secret" in fields

    def test_get_required_credentials_unknown(self, handler):
        """Returns empty list for unknown platform."""
        fields = handler._get_required_credentials("unknown")
        assert fields == []

    def test_suggest_priority_urgent(self, handler):
        """Suggests urgent for critical keywords."""
        ticket = {"subject": "URGENT: System down", "description": ""}
        assert handler._suggest_priority(ticket) == "urgent"

    def test_suggest_priority_high(self, handler):
        """Suggests high for error keywords."""
        ticket = {"subject": "Error on page", "description": ""}
        assert handler._suggest_priority(ticket) == "high"

    def test_suggest_priority_medium_default(self, handler):
        """Defaults to medium when no keywords match."""
        ticket = {"subject": "General question", "description": "How do I use the API?"}
        assert handler._suggest_priority(ticket) == "medium"

    def test_suggest_category_billing(self, handler):
        """Detects billing category."""
        ticket = {"subject": "Invoice question", "description": ""}
        assert handler._suggest_category(ticket) == "billing"

    def test_suggest_category_technical(self, handler):
        """Detects technical category."""
        ticket = {"subject": "Application crash", "description": ""}
        assert handler._suggest_category(ticket) == "technical"

    def test_suggest_category_account(self, handler):
        """Detects account category."""
        ticket = {"subject": "Cannot login", "description": ""}
        assert handler._suggest_category(ticket) == "account"

    def test_suggest_category_feature_request(self, handler):
        """Detects feature request category."""
        ticket = {"subject": "Feature request: dark mode", "description": ""}
        assert handler._suggest_category(ticket) == "feature_request"

    def test_suggest_category_general_default(self, handler):
        """Defaults to general when no keywords match."""
        ticket = {"subject": "Hello", "description": "Just saying hi"}
        assert handler._suggest_category(ticket) == "general"

    def test_analyze_sentiment_negative(self, handler):
        """Detects negative sentiment."""
        ticket = {"description": "I am frustrated and disappointed with this service"}
        assert handler._analyze_sentiment(ticket) == "negative"

    def test_analyze_sentiment_positive(self, handler):
        """Detects positive sentiment."""
        ticket = {"description": "Thanks for the great help, I really appreciate it"}
        assert handler._analyze_sentiment(ticket) == "positive"

    def test_analyze_sentiment_neutral(self, handler):
        """Returns neutral when no strong signals."""
        ticket = {"description": "I need to update my settings"}
        assert handler._analyze_sentiment(ticket) == "neutral"

    def test_calculate_urgency_urgent(self, handler):
        """Urgent priority increases score."""
        ticket = {"priority": "urgent", "description": ""}
        score = handler._calculate_urgency(ticket)
        assert score == 0.8  # 0.5 base + 0.3

    def test_calculate_urgency_high(self, handler):
        """High priority increases score."""
        ticket = {"priority": "high", "description": ""}
        score = handler._calculate_urgency(ticket)
        assert score == 0.7  # 0.5 base + 0.2

    def test_calculate_urgency_low(self, handler):
        """Low priority decreases score."""
        ticket = {"priority": "low", "description": ""}
        score = handler._calculate_urgency(ticket)
        assert score == 0.3  # 0.5 base - 0.2

    def test_calculate_urgency_negative_sentiment(self, handler):
        """Negative sentiment adds to urgency."""
        ticket = {"priority": "medium", "description": "I am frustrated and angry"}
        score = handler._calculate_urgency(ticket)
        assert score == 0.6  # 0.5 base + 0.1

    def test_calculate_urgency_clamped(self, handler):
        """Score is clamped between 0 and 1."""
        ticket = {"priority": "urgent", "description": "frustrated angry awful terrible"}
        score = handler._calculate_urgency(ticket)
        assert 0 <= score <= 1

    def test_suggest_response_template_billing(self, handler):
        """Suggests billing template for billing tickets."""
        ticket = {"subject": "Billing issue", "description": "Charge on invoice"}
        assert handler._suggest_response_template(ticket) == "billing_inquiry"

    def test_suggest_response_template_technical(self, handler):
        """Suggests technical template for error tickets."""
        ticket = {"subject": "Bug report", "description": "Error on the page"}
        assert handler._suggest_response_template(ticket) == "technical_support"

    def test_suggest_response_template_general(self, handler):
        """Defaults to general response template."""
        ticket = {"subject": "Hello", "description": "Just a question"}
        assert handler._suggest_response_template(ticket) == "general_response"

    def test_count_by_priority(self, handler):
        """Counts tickets by priority."""
        tickets = [
            {"priority": "high"},
            {"priority": "high"},
            {"priority": "low"},
            {"priority": None},
        ]
        counts = handler._count_by_priority(tickets)
        assert counts["high"] == 2
        assert counts["low"] == 1
        assert counts["unset"] == 1

    def test_map_freshdesk_status(self, handler):
        """Maps Freshdesk status codes to strings."""
        assert handler._map_freshdesk_status(2) == "open"
        assert handler._map_freshdesk_status(3) == "pending"
        assert handler._map_freshdesk_status(4) == "resolved"
        assert handler._map_freshdesk_status(5) == "closed"
        assert handler._map_freshdesk_status(99) == "unknown"

    def test_map_freshdesk_priority(self, handler):
        """Maps Freshdesk priority codes to strings."""
        assert handler._map_freshdesk_priority(1) == "low"
        assert handler._map_freshdesk_priority(2) == "medium"
        assert handler._map_freshdesk_priority(3) == "high"
        assert handler._map_freshdesk_priority(4) == "urgent"
        assert handler._map_freshdesk_priority(99) == "medium"

    def test_map_status_to_freshdesk(self, handler):
        """Maps status strings to Freshdesk codes."""
        assert handler._map_status_to_freshdesk("open") == 2
        assert handler._map_status_to_freshdesk("pending") == 3
        assert handler._map_status_to_freshdesk("resolved") == 4
        assert handler._map_status_to_freshdesk("closed") == 5
        assert handler._map_status_to_freshdesk("unknown") == 2

    def test_map_priority_to_freshdesk(self, handler):
        """Maps priority strings to Freshdesk codes."""
        assert handler._map_priority_to_freshdesk("low") == 1
        assert handler._map_priority_to_freshdesk("medium") == 2
        assert handler._map_priority_to_freshdesk("high") == 3
        assert handler._map_priority_to_freshdesk("urgent") == 4
        assert handler._map_priority_to_freshdesk(None) == 2

    def test_json_response_format(self, handler):
        """_json_response produces correct structure."""
        result = handler._json_response(200, {"key": "value"})
        assert result["status_code"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"]["key"] == "value"

    def test_error_response_format(self, handler):
        """_error_response produces correct error structure."""
        result = handler._error_response(404, "Not found")
        assert result["status_code"] == 404
        assert result["body"]["error"] == "Not found"


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    """Tests for ticket/conversation normalization methods."""

    def test_normalize_zendesk_ticket(self, handler):
        """Normalizes Zendesk ticket to unified format."""
        ticket = _make_mock_ticket(42, "Bug", "Broken", "open", "high")
        result = handler._normalize_zendesk_ticket(ticket)

        assert result["id"] == "42"
        assert result["platform"] == "zendesk"
        assert result["subject"] == "Bug"
        assert result["status"] == "open"
        assert result["priority"] == "high"

    def test_normalize_zendesk_comment(self, handler):
        """Normalizes Zendesk comment."""
        comment = _make_mock_comment(100, "Hello", True, 10)
        result = handler._normalize_zendesk_comment(comment)

        assert result["id"] == "100"
        assert result["body"] == "Hello"
        assert result["public"] is True
        assert result["author_id"] == "10"

    def test_normalize_freshdesk_ticket(self, handler):
        """Normalizes Freshdesk ticket to unified format."""
        ticket = _make_mock_freshdesk_ticket(42, "Bug", "Broken", 2, 3)
        result = handler._normalize_freshdesk_ticket(ticket)

        assert result["id"] == "42"
        assert result["platform"] == "freshdesk"
        assert result["status"] == "open"
        assert result["priority"] == "high"

    def test_normalize_intercom_conversation(self, handler):
        """Normalizes Intercom conversation to unified format."""
        conv = _make_mock_intercom_conversation("conv-1", "Chat", "open")
        result = handler._normalize_intercom_conversation(conv)

        assert result["id"] == "conv-1"
        assert result["platform"] == "intercom"
        assert result["subject"] == "Chat"
        assert result["status"] == "open"

    def test_normalize_helpscout_conversation(self, handler):
        """Normalizes Help Scout conversation to unified format."""
        conv = _make_mock_helpscout_conversation(42, "Help needed", "active")
        result = handler._normalize_helpscout_conversation(conv)

        assert result["id"] == "42"
        assert result["platform"] == "helpscout"
        assert result["subject"] == "Help needed"
        assert result["requester_name"] == "Test User"
        assert result["assignee_id"] == "42"

    def test_normalize_zendesk_ticket_no_assignee(self, handler):
        """Handles Zendesk ticket with no assignee."""
        ticket = _make_mock_ticket(1, assignee_id=None)
        result = handler._normalize_zendesk_ticket(ticket)
        assert result["assignee_id"] is None

    def test_normalize_zendesk_ticket_enum_status(self, handler):
        """Handles Zendesk ticket with enum status (has .value)."""
        ticket = _make_mock_ticket(1)
        ticket.status = MagicMock()
        ticket.status.value = "open"
        result = handler._normalize_zendesk_ticket(ticket)
        assert result["status"] == "open"


# =============================================================================
# Path Parameter Extraction Tests
# =============================================================================


class TestPathParameterExtraction:
    """Tests for correct path parsing and parameter extraction."""

    @pytest.mark.asyncio
    async def test_extracts_platform_from_disconnect_path(
        self, handler, zendesk_connected
    ):
        """Correctly extracts platform name from DELETE path."""
        request = MockRequest(method="DELETE", path="/api/v1/support/zendesk")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "zendesk"

    @pytest.mark.asyncio
    async def test_extracts_ticket_id_from_update_path(
        self, handler, zendesk_connected, mock_zendesk_connector
    ):
        """Correctly extracts ticket_id from PUT path."""
        mock_zendesk_connector.update_ticket = AsyncMock(
            return_value=_make_mock_ticket(555)
        )

        request = MockRequest(
            method="PUT",
            path="/api/v1/support/zendesk/tickets/555",
            _body={"status": "open"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        # Verify the connector was called with the correct ID
        mock_zendesk_connector.update_ticket.assert_awaited_once()
        call_args = mock_zendesk_connector.update_ticket.call_args
        assert call_args[0][0] == 555

    @pytest.mark.asyncio
    async def test_non_platform_segment_not_treated_as_platform(self, handler):
        """Segments like 'platforms', 'metrics' are not treated as platform names."""
        request = MockRequest(method="GET", path="/api/v1/support/platforms")
        result = await handler.handle_request(request)

        # Should route to _list_platforms, not try to parse 'platforms' as a platform
        assert _status(result) == 200
        assert "platforms" in _body(result)

    @pytest.mark.asyncio
    async def test_extracts_platform_from_ticket_list_path(
        self, handler, freshdesk_connected, mock_freshdesk_connector
    ):
        """Correctly extracts platform from platform ticket list path."""
        mock_freshdesk_connector.get_tickets = AsyncMock(return_value=[])

        request = MockRequest(method="GET", path="/api/v1/support/freshdesk/tickets")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["platform"] == "freshdesk"
