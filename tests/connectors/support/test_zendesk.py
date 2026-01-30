"""
Comprehensive tests for the Zendesk Support Connector.

Tests cover:
- Enum values (TicketStatus, TicketPriority, TicketType, UserRole)
- ZendeskCredentials dataclass and auth header generation
- Data models: ZendeskUser, Organization, TicketComment, Ticket, View
- Model serialization (from_api)
- ZendeskConnector initialization and async context manager
- API request handling (_request with auth, error responses)
- Tickets CRUD operations (get, create, update, delete, comments)
- Users CRUD operations (get, create, search)
- Organizations CRUD operations
- Views and search operations
- Helper functions (_parse_datetime)
- Error handling (HTTP errors, JSON parsing)
- Edge cases (empty responses, missing fields, malformed data)
- Mock data generation helpers
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.support.zendesk import (
    Organization,
    Ticket,
    TicketComment,
    TicketPriority,
    TicketStatus,
    TicketType,
    UserRole,
    View,
    ZendeskConnector,
    ZendeskCredentials,
    ZendeskError,
    ZendeskUser,
    _parse_datetime,
    get_mock_ticket,
    get_mock_user,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return ZendeskCredentials(
        subdomain="testcompany",
        email="agent@testcompany.com",
        api_token="test-api-token-xyz",
    )


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def zendesk_connector(credentials, mock_httpx_client):
    """Create a ZendeskConnector with a mock HTTP client."""
    connector = ZendeskConnector(credentials)
    connector._client = mock_httpx_client
    return connector


def _make_response(json_data: dict[str, Any], status_code: int = 200) -> MagicMock:
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    return resp


# =============================================================================
# Enum Tests
# =============================================================================


class TestTicketStatus:
    """Tests for TicketStatus enum."""

    def test_ticket_status_values(self):
        assert TicketStatus.NEW.value == "new"
        assert TicketStatus.OPEN.value == "open"
        assert TicketStatus.PENDING.value == "pending"
        assert TicketStatus.HOLD.value == "hold"
        assert TicketStatus.SOLVED.value == "solved"
        assert TicketStatus.CLOSED.value == "closed"

    def test_ticket_status_is_str(self):
        assert isinstance(TicketStatus.OPEN, str)
        assert TicketStatus.OPEN == "open"

    def test_ticket_status_from_string(self):
        assert TicketStatus("pending") == TicketStatus.PENDING


class TestTicketPriority:
    """Tests for TicketPriority enum."""

    def test_ticket_priority_values(self):
        assert TicketPriority.LOW.value == "low"
        assert TicketPriority.NORMAL.value == "normal"
        assert TicketPriority.HIGH.value == "high"
        assert TicketPriority.URGENT.value == "urgent"

    def test_ticket_priority_is_str(self):
        assert isinstance(TicketPriority.HIGH, str)


class TestTicketType:
    """Tests for TicketType enum."""

    def test_ticket_type_values(self):
        assert TicketType.QUESTION.value == "question"
        assert TicketType.INCIDENT.value == "incident"
        assert TicketType.PROBLEM.value == "problem"
        assert TicketType.TASK.value == "task"

    def test_ticket_type_is_str(self):
        assert isinstance(TicketType.INCIDENT, str)


class TestUserRole:
    """Tests for UserRole enum."""

    def test_user_role_values(self):
        assert UserRole.END_USER.value == "end-user"
        assert UserRole.AGENT.value == "agent"
        assert UserRole.ADMIN.value == "admin"

    def test_user_role_is_str(self):
        assert isinstance(UserRole.AGENT, str)


# =============================================================================
# Credentials Tests
# =============================================================================


class TestZendeskCredentials:
    """Tests for ZendeskCredentials dataclass."""

    def test_basic_construction(self):
        creds = ZendeskCredentials(
            subdomain="mycompany",
            email="user@mycompany.com",
            api_token="token123",
        )
        assert creds.subdomain == "mycompany"
        assert creds.email == "user@mycompany.com"
        assert creds.api_token == "token123"

    def test_base_url_property(self):
        creds = ZendeskCredentials(
            subdomain="test",
            email="test@test.com",
            api_token="token",
        )
        assert creds.base_url == "https://test.zendesk.com/api/v2"

    def test_auth_header_property(self):
        creds = ZendeskCredentials(
            subdomain="test",
            email="agent@test.com",
            api_token="mytoken",
        )
        expected_credentials = "agent@test.com/token:mytoken"
        expected_encoded = base64.b64encode(expected_credentials.encode()).decode()
        assert creds.auth_header == f"Basic {expected_encoded}"

    def test_auth_header_encoding(self):
        creds = ZendeskCredentials(
            subdomain="company",
            email="support@company.com",
            api_token="secret123",
        )
        # Verify the auth header can be decoded correctly
        auth_header = creds.auth_header
        assert auth_header.startswith("Basic ")
        encoded_part = auth_header[6:]
        decoded = base64.b64decode(encoded_part).decode()
        assert decoded == "support@company.com/token:secret123"


# =============================================================================
# Error Tests
# =============================================================================


class TestZendeskError:
    """Tests for ZendeskError exception."""

    def test_error_basic(self):
        err = ZendeskError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.status_code is None
        assert err.details == {}

    def test_error_with_status(self):
        err = ZendeskError("Not found", status_code=404, details={"error": "RecordNotFound"})
        assert err.status_code == 404
        assert err.details == {"error": "RecordNotFound"}


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for _parse_datetime helper function."""

    def test_parse_datetime_none(self):
        assert _parse_datetime(None) is None

    def test_parse_datetime_empty(self):
        assert _parse_datetime("") is None

    def test_parse_datetime_iso_with_z(self):
        result = _parse_datetime("2023-06-15T10:30:00Z")
        assert result is not None
        assert result.year == 2023
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_datetime_iso_with_offset(self):
        result = _parse_datetime("2023-06-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2023

    def test_parse_datetime_invalid(self):
        assert _parse_datetime("not-a-date") is None

    def test_parse_datetime_malformed(self):
        assert _parse_datetime("2023/06/15") is None


# =============================================================================
# ZendeskUser Model Tests
# =============================================================================


class TestZendeskUserModel:
    """Tests for ZendeskUser dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 12345,
            "name": "John Doe",
            "email": "john@example.com",
            "role": "end-user",
            "organization_id": 100,
            "phone": "+1-555-0100",
            "time_zone": "America/New_York",
            "verified": True,
            "suspended": False,
            "created_at": "2023-01-15T10:00:00Z",
            "updated_at": "2023-06-20T14:30:00Z",
            "tags": ["vip", "enterprise"],
        }
        user = ZendeskUser.from_api(data)
        assert user.id == 12345
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.role == UserRole.END_USER
        assert user.organization_id == 100
        assert user.phone == "+1-555-0100"
        assert user.time_zone == "America/New_York"
        assert user.verified is True
        assert user.suspended is False
        assert user.tags == ["vip", "enterprise"]
        assert user.created_at is not None

    def test_from_api_minimal(self):
        data = {"id": 1, "name": "", "email": "", "role": "end-user"}
        user = ZendeskUser.from_api(data)
        assert user.id == 1
        assert user.name == ""
        assert user.organization_id is None
        assert user.tags == []

    def test_from_api_agent_role(self):
        data = {"id": 2, "name": "Agent Smith", "email": "agent@test.com", "role": "agent"}
        user = ZendeskUser.from_api(data)
        assert user.role == UserRole.AGENT

    def test_from_api_admin_role(self):
        data = {"id": 3, "name": "Admin", "email": "admin@test.com", "role": "admin"}
        user = ZendeskUser.from_api(data)
        assert user.role == UserRole.ADMIN


# =============================================================================
# Organization Model Tests
# =============================================================================


class TestOrganizationModel:
    """Tests for Organization dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 100,
            "name": "Acme Corp",
            "domain_names": ["acme.com", "acme.io"],
            "details": "Enterprise customer",
            "notes": "Important client",
            "group_id": 5,
            "tags": ["enterprise", "priority"],
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-06-01T12:00:00Z",
        }
        org = Organization.from_api(data)
        assert org.id == 100
        assert org.name == "Acme Corp"
        assert org.domain_names == ["acme.com", "acme.io"]
        assert org.details == "Enterprise customer"
        assert org.notes == "Important client"
        assert org.group_id == 5
        assert org.tags == ["enterprise", "priority"]

    def test_from_api_minimal(self):
        data = {"id": 1}
        org = Organization.from_api(data)
        assert org.id == 1
        assert org.name == ""
        assert org.domain_names == []
        assert org.tags == []


# =============================================================================
# TicketComment Model Tests
# =============================================================================


class TestTicketCommentModel:
    """Tests for TicketComment dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 500,
            "body": "Thank you for contacting us.",
            "author_id": 12345,
            "public": True,
            "created_at": "2023-07-10T09:00:00Z",
            "attachments": [{"file_name": "screenshot.png", "url": "https://..."}],
        }
        comment = TicketComment.from_api(data)
        assert comment.id == 500
        assert comment.body == "Thank you for contacting us."
        assert comment.author_id == 12345
        assert comment.public is True
        assert len(comment.attachments) == 1

    def test_from_api_minimal(self):
        data = {"id": 1}
        comment = TicketComment.from_api(data)
        assert comment.id == 1
        assert comment.body == ""
        assert comment.author_id == 0
        assert comment.attachments == []

    def test_from_api_private_comment(self):
        data = {"id": 2, "body": "Internal note", "author_id": 1, "public": False}
        comment = TicketComment.from_api(data)
        assert comment.public is False


# =============================================================================
# Ticket Model Tests
# =============================================================================


class TestTicketModel:
    """Tests for Ticket dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 1000,
            "subject": "Cannot login to my account",
            "description": "I tried resetting my password but still cannot login.",
            "status": "open",
            "priority": "high",
            "type": "incident",
            "requester_id": 12345,
            "assignee_id": 67890,
            "group_id": 5,
            "organization_id": 100,
            "tags": ["login", "urgent"],
            "custom_fields": [{"id": 123, "value": "enterprise"}],
            "created_at": "2023-07-01T10:00:00Z",
            "updated_at": "2023-07-01T12:00:00Z",
            "due_at": "2023-07-02T10:00:00Z",
        }
        ticket = Ticket.from_api(data)
        assert ticket.id == 1000
        assert ticket.subject == "Cannot login to my account"
        assert ticket.status == TicketStatus.OPEN
        assert ticket.priority == TicketPriority.HIGH
        assert ticket.type == TicketType.INCIDENT
        assert ticket.requester_id == 12345
        assert ticket.assignee_id == 67890
        assert ticket.tags == ["login", "urgent"]

    def test_from_api_minimal(self):
        data = {"id": 1, "status": "new"}
        ticket = Ticket.from_api(data)
        assert ticket.id == 1
        assert ticket.status == TicketStatus.NEW
        assert ticket.priority is None
        assert ticket.type is None

    def test_from_api_all_statuses(self):
        for status in TicketStatus:
            data = {"id": 1, "status": status.value}
            ticket = Ticket.from_api(data)
            assert ticket.status == status

    def test_from_api_all_priorities(self):
        for priority in TicketPriority:
            data = {"id": 1, "status": "new", "priority": priority.value}
            ticket = Ticket.from_api(data)
            assert ticket.priority == priority

    def test_from_api_all_types(self):
        for ticket_type in TicketType:
            data = {"id": 1, "status": "new", "type": ticket_type.value}
            ticket = Ticket.from_api(data)
            assert ticket.type == ticket_type


# =============================================================================
# View Model Tests
# =============================================================================


class TestViewModel:
    """Tests for View dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 10,
            "title": "My Open Tickets",
            "active": True,
            "position": 1,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-06-01T12:00:00Z",
        }
        view = View.from_api(data)
        assert view.id == 10
        assert view.title == "My Open Tickets"
        assert view.active is True
        assert view.position == 1

    def test_from_api_minimal(self):
        data = {"id": 1}
        view = View.from_api(data)
        assert view.id == 1
        assert view.title == ""
        assert view.active is True
        assert view.position == 0


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestZendeskConnectorInit:
    """Tests for ZendeskConnector initialization and context manager."""

    def test_init(self, credentials):
        connector = ZendeskConnector(credentials)
        assert connector.credentials is credentials
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, credentials):
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client = AsyncMock()
            mock_async_client_class.return_value = mock_async_client

            async with ZendeskConnector(credentials) as connector:
                # Trigger client creation
                await connector._get_client()
                assert connector._client is mock_async_client

            mock_async_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exit_clears_client(self, credentials):
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client = AsyncMock()
            mock_async_client_class.return_value = mock_async_client

            connector = ZendeskConnector(credentials)
            await connector.__aenter__()
            await connector._get_client()
            assert connector._client is not None
            await connector.__aexit__(None, None, None)
            assert connector._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, credentials):
        connector = ZendeskConnector(credentials)
        assert connector._client is None
        await connector.close()
        assert connector._client is None


# =============================================================================
# API Request Tests
# =============================================================================


class TestApiRequest:
    """Tests for _request method."""

    @pytest.mark.asyncio
    async def test_request_success(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"ticket": {"id": 1}})
        result = await zendesk_connector._request("GET", "/tickets/1.json")
        assert result["ticket"]["id"] == 1

    @pytest.mark.asyncio
    async def test_request_http_error(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"error": "RecordNotFound"},
            status_code=404,
        )
        with pytest.raises(ZendeskError) as exc_info:
            await zendesk_connector._request("GET", "/tickets/999.json")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_request_http_error_no_json(self, zendesk_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        resp.json.side_effect = ValueError("No JSON")
        mock_httpx_client.request.return_value = resp
        with pytest.raises(ZendeskError) as exc_info:
            await zendesk_connector._request("GET", "/test")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_204_no_content(self, zendesk_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        mock_httpx_client.request.return_value = resp
        result = await zendesk_connector._request("DELETE", "/tickets/1.json")
        assert result == {}

    @pytest.mark.asyncio
    async def test_request_rate_limit_429(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"error": "Rate limit exceeded"},
            status_code=429,
        )
        with pytest.raises(ZendeskError) as exc_info:
            await zendesk_connector._request("GET", "/tickets.json")
        assert exc_info.value.status_code == 429


# =============================================================================
# Ticket Operations Tests
# =============================================================================


class TestTicketOperations:
    """Tests for ticket CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_tickets(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "tickets": [
                    {"id": 1, "subject": "Issue 1", "status": "open"},
                    {"id": 2, "subject": "Issue 2", "status": "pending"},
                ],
                "next_page": None,
            }
        )
        tickets, has_more = await zendesk_connector.get_tickets()
        assert len(tickets) == 2
        assert tickets[0].subject == "Issue 1"
        assert has_more is False

    @pytest.mark.asyncio
    async def test_get_tickets_with_filter(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"results": [{"id": 1, "status": "open"}], "next_page": "https://..."}
        )
        tickets, has_more = await zendesk_connector.get_tickets(status=TicketStatus.OPEN)
        assert len(tickets) == 1
        assert has_more is True

    @pytest.mark.asyncio
    async def test_get_tickets_with_assignee_filter(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"results": [], "next_page": None}
        )
        await zendesk_connector.get_tickets(assignee_id=123)
        call_args = mock_httpx_client.request.call_args
        assert "assignee_id:123" in call_args.kwargs["params"]["query"]

    @pytest.mark.asyncio
    async def test_get_ticket(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"ticket": {"id": 1000, "subject": "Help needed", "status": "open"}}
        )
        ticket = await zendesk_connector.get_ticket(1000)
        assert ticket.id == 1000
        assert ticket.subject == "Help needed"

    @pytest.mark.asyncio
    async def test_create_ticket(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "ticket": {
                    "id": 2000,
                    "subject": "New ticket",
                    "description": "Description here",
                    "status": "new",
                    "requester_id": 123,
                }
            }
        )
        ticket = await zendesk_connector.create_ticket(
            subject="New ticket",
            description="Description here",
            requester_id=123,
            priority=TicketPriority.HIGH,
            type=TicketType.INCIDENT,
            tags=["urgent"],
        )
        assert ticket.id == 2000
        assert ticket.subject == "New ticket"

    @pytest.mark.asyncio
    async def test_create_ticket_with_email(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"ticket": {"id": 2001, "subject": "Email ticket", "status": "new", "requester_id": 0}}
        )
        ticket = await zendesk_connector.create_ticket(
            subject="Email ticket",
            description="From email",
            requester_email="customer@example.com",
        )
        assert ticket.id == 2001

    @pytest.mark.asyncio
    async def test_update_ticket(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"ticket": {"id": 1000, "subject": "Updated", "status": "pending", "requester_id": 1}}
        )
        ticket = await zendesk_connector.update_ticket(
            1000,
            status=TicketStatus.PENDING,
            comment="Working on this",
            public=False,
        )
        assert ticket.status == TicketStatus.PENDING

    @pytest.mark.asyncio
    async def test_delete_ticket(self, zendesk_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        mock_httpx_client.request.return_value = resp
        result = await zendesk_connector.delete_ticket(1000)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_ticket_comments(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "comments": [
                    {"id": 1, "body": "First comment", "author_id": 123, "public": True},
                    {"id": 2, "body": "Second comment", "author_id": 456, "public": False},
                ]
            }
        )
        comments = await zendesk_connector.get_ticket_comments(1000)
        assert len(comments) == 2
        assert comments[0].body == "First comment"
        assert comments[1].public is False

    @pytest.mark.asyncio
    async def test_add_ticket_comment(self, zendesk_connector, mock_httpx_client):
        # First call for update, second for getting comments
        mock_httpx_client.request.side_effect = [
            _make_response({"ticket": {"id": 1000, "status": "open", "requester_id": 1}}),
            _make_response(
                {"comments": [{"id": 100, "body": "New comment", "author_id": 1, "public": True}]}
            ),
        ]
        comment = await zendesk_connector.add_ticket_comment(1000, "New comment", public=True)
        assert comment.body == "New comment"


# =============================================================================
# User Operations Tests
# =============================================================================


class TestUserOperations:
    """Tests for user CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_users(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "users": [
                    {"id": 1, "name": "User 1", "email": "user1@test.com", "role": "end-user"},
                    {"id": 2, "name": "User 2", "email": "user2@test.com", "role": "agent"},
                ],
                "next_page": None,
            }
        )
        users, has_more = await zendesk_connector.get_users()
        assert len(users) == 2
        assert users[0].role == UserRole.END_USER
        assert users[1].role == UserRole.AGENT

    @pytest.mark.asyncio
    async def test_get_users_with_role_filter(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"users": [], "next_page": None}
        )
        await zendesk_connector.get_users(role=UserRole.AGENT)
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["role"] == "agent"

    @pytest.mark.asyncio
    async def test_get_user(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"user": {"id": 12345, "name": "John Doe", "email": "john@test.com", "role": "end-user"}}
        )
        user = await zendesk_connector.get_user(12345)
        assert user.id == 12345
        assert user.name == "John Doe"

    @pytest.mark.asyncio
    async def test_create_user(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "user": {
                    "id": 99999,
                    "name": "New User",
                    "email": "new@test.com",
                    "role": "end-user",
                }
            }
        )
        user = await zendesk_connector.create_user(
            name="New User",
            email="new@test.com",
            role=UserRole.END_USER,
            phone="+1-555-0100",
            organization_id=100,
        )
        assert user.id == 99999
        assert user.name == "New User"

    @pytest.mark.asyncio
    async def test_search_users(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "users": [
                    {"id": 1, "name": "John Doe", "email": "john@test.com", "role": "end-user"},
                ]
            }
        )
        users = await zendesk_connector.search_users("john")
        assert len(users) == 1
        assert users[0].name == "John Doe"


# =============================================================================
# Organization Operations Tests
# =============================================================================


class TestOrganizationOperations:
    """Tests for organization CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_organizations(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "organizations": [
                    {"id": 100, "name": "Acme Corp"},
                    {"id": 101, "name": "Tech Inc"},
                ],
                "next_page": None,
            }
        )
        orgs, has_more = await zendesk_connector.get_organizations()
        assert len(orgs) == 2
        assert orgs[0].name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_organizations_pagination(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "organizations": [{"id": 100, "name": "Org"}],
                "next_page": "https://zendesk.com/api/v2/organizations.json?page=2",
            }
        )
        orgs, has_more = await zendesk_connector.get_organizations(page=1, per_page=50)
        assert has_more is True

    @pytest.mark.asyncio
    async def test_get_organization(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"organization": {"id": 100, "name": "Acme Corp", "domain_names": ["acme.com"]}}
        )
        org = await zendesk_connector.get_organization(100)
        assert org.id == 100
        assert org.domain_names == ["acme.com"]

    @pytest.mark.asyncio
    async def test_create_organization(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "organization": {
                    "id": 200,
                    "name": "New Org",
                    "domain_names": ["neworg.com"],
                    "details": "A new organization",
                }
            }
        )
        org = await zendesk_connector.create_organization(
            name="New Org",
            domain_names=["neworg.com"],
            details="A new organization",
        )
        assert org.id == 200
        assert org.name == "New Org"


# =============================================================================
# View Operations Tests
# =============================================================================


class TestViewOperations:
    """Tests for view operations."""

    @pytest.mark.asyncio
    async def test_get_views(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "views": [
                    {"id": 1, "title": "My Open Tickets", "active": True},
                    {"id": 2, "title": "All Tickets", "active": True},
                ]
            }
        )
        views = await zendesk_connector.get_views()
        assert len(views) == 2
        assert views[0].title == "My Open Tickets"

    @pytest.mark.asyncio
    async def test_get_view_tickets(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "tickets": [
                    {"id": 1, "subject": "Ticket 1", "status": "open"},
                    {"id": 2, "subject": "Ticket 2", "status": "pending"},
                ],
                "next_page": None,
            }
        )
        tickets, has_more = await zendesk_connector.get_view_tickets(1)
        assert len(tickets) == 2
        assert not has_more


# =============================================================================
# Search Operations Tests
# =============================================================================


class TestSearchOperations:
    """Tests for search operations."""

    @pytest.mark.asyncio
    async def test_search(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "results": [
                    {"id": 1, "subject": "Login issue", "result_type": "ticket"},
                ]
            }
        )
        results = await zendesk_connector.search("login")
        assert len(results) == 1
        assert results[0]["subject"] == "Login issue"

    @pytest.mark.asyncio
    async def test_search_with_type(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"results": []})
        await zendesk_connector.search("test", type="ticket")
        call_args = mock_httpx_client.request.call_args
        assert "type:ticket" in call_args.kwargs["params"]["query"]


# =============================================================================
# Mock Data Generation Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data helpers."""

    def test_get_mock_ticket(self):
        ticket = get_mock_ticket()
        assert ticket.id == 12345
        assert ticket.subject == "Cannot login to my account"
        assert ticket.status == TicketStatus.OPEN
        assert ticket.priority == TicketPriority.HIGH
        assert ticket.type == TicketType.INCIDENT
        assert ticket.requester_id == 67890
        assert ticket.created_at is not None

    def test_get_mock_user(self):
        user = get_mock_user()
        assert user.id == 67890
        assert user.name == "John Doe"
        assert user.email == "john.doe@example.com"
        assert user.role == UserRole.END_USER
        assert user.verified is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_ticket_from_api_empty_tags(self):
        data = {"id": 1, "status": "new", "tags": []}
        ticket = Ticket.from_api(data)
        assert ticket.tags == []

    def test_user_from_api_empty_tags(self):
        data = {"id": 1, "name": "Test", "email": "test@test.com", "role": "end-user", "tags": []}
        user = ZendeskUser.from_api(data)
        assert user.tags == []

    def test_organization_from_api_empty_domain_names(self):
        data = {"id": 1, "name": "Test", "domain_names": []}
        org = Organization.from_api(data)
        assert org.domain_names == []

    def test_ticket_defaults(self):
        ticket = Ticket(
            id=1,
            subject="Test",
            description="",
            status=TicketStatus.NEW,
            priority=None,
            type=None,
            requester_id=1,
        )
        assert ticket.assignee_id is None
        assert ticket.tags == []
        assert ticket.custom_fields == []
        assert ticket.comments == []

    def test_user_defaults(self):
        user = ZendeskUser(
            id=1,
            name="Test",
            email="test@test.com",
            role=UserRole.END_USER,
        )
        assert user.organization_id is None
        assert user.phone is None
        assert user.verified is False
        assert user.suspended is False
        assert user.tags == []

    @pytest.mark.asyncio
    async def test_get_tickets_empty_response(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"tickets": [], "next_page": None}
        )
        tickets, has_more = await zendesk_connector.get_tickets()
        assert tickets == []
        assert not has_more

    @pytest.mark.asyncio
    async def test_add_comment_empty_comments_list(self, zendesk_connector, mock_httpx_client):
        mock_httpx_client.request.side_effect = [
            _make_response({"ticket": {"id": 1, "status": "open", "requester_id": 1}}),
            _make_response({"comments": []}),
        ]
        comment = await zendesk_connector.add_ticket_comment(1, "Test comment")
        # Should return a fallback comment when no comments are returned
        assert comment.body == "Test comment"
        assert comment.id == 0
