"""
Comprehensive tests for the Intercom Connector.

Tests cover:
- Enum values (ConversationState, ContactRole, MessageType)
- IntercomCredentials dataclass
- Data models: IntercomContact, IntercomCompany, ConversationPart, Conversation, Admin, Article
- Model serialization (from_api)
- IntercomConnector initialization and async context manager
- API request handling (_request with auth, error responses)
- Conversations CRUD operations (get, reply, close, assign, snooze)
- Contacts CRUD operations (get, create, update, search)
- Companies CRUD operations
- Admins read operations
- Articles CRUD operations
- Tags operations (tag/untag contacts and conversations)
- Helper functions (_from_timestamp)
- Error handling (HTTP errors, JSON parsing)
- Edge cases (empty responses, missing fields, malformed data)
- Mock data generation helpers
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.support.intercom import (
    Admin,
    Article,
    ContactRole,
    Conversation,
    ConversationPart,
    ConversationState,
    IntercomCompany,
    IntercomConnector,
    IntercomContact,
    IntercomCredentials,
    IntercomError,
    MessageType,
    _from_timestamp,
    get_mock_contact,
    get_mock_conversation,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return IntercomCredentials(access_token="test-access-token-xyz")


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def intercom_connector(credentials, mock_httpx_client):
    """Create an IntercomConnector with a mock HTTP client."""
    connector = IntercomConnector(credentials)
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


class TestConversationState:
    """Tests for ConversationState enum."""

    def test_conversation_state_values(self):
        assert ConversationState.OPEN.value == "open"
        assert ConversationState.CLOSED.value == "closed"
        assert ConversationState.SNOOZED.value == "snoozed"

    def test_conversation_state_is_str(self):
        assert isinstance(ConversationState.OPEN, str)
        assert ConversationState.OPEN == "open"

    def test_conversation_state_from_string(self):
        assert ConversationState("closed") == ConversationState.CLOSED


class TestContactRole:
    """Tests for ContactRole enum."""

    def test_contact_role_values(self):
        assert ContactRole.USER.value == "user"
        assert ContactRole.LEAD.value == "lead"

    def test_contact_role_is_str(self):
        assert isinstance(ContactRole.USER, str)


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_type_values(self):
        assert MessageType.COMMENT.value == "comment"
        assert MessageType.NOTE.value == "note"
        assert MessageType.ASSIGNMENT.value == "assignment"

    def test_message_type_is_str(self):
        assert isinstance(MessageType.COMMENT, str)


# =============================================================================
# Credentials Tests
# =============================================================================


class TestIntercomCredentials:
    """Tests for IntercomCredentials dataclass."""

    def test_basic_construction(self):
        creds = IntercomCredentials(access_token="my-token")
        assert creds.access_token == "my-token"
        assert creds.base_url == "https://api.intercom.io"

    def test_custom_base_url(self):
        creds = IntercomCredentials(
            access_token="token",
            base_url="https://custom.intercom.io",
        )
        assert creds.base_url == "https://custom.intercom.io"


# =============================================================================
# Error Tests
# =============================================================================


class TestIntercomError:
    """Tests for IntercomError exception."""

    def test_error_basic(self):
        err = IntercomError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.status_code is None
        assert err.errors == []

    def test_error_with_status(self):
        err = IntercomError(
            "Not found",
            status_code=404,
            errors=[{"code": "not_found", "message": "Resource not found"}],
        )
        assert err.status_code == 404
        assert len(err.errors) == 1


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for _from_timestamp helper function."""

    def test_from_timestamp_none(self):
        assert _from_timestamp(None) is None

    def test_from_timestamp_zero(self):
        assert _from_timestamp(0) is None

    def test_from_timestamp_valid(self):
        # 2023-06-15 10:30:00 UTC
        timestamp = 1686827400
        result = _from_timestamp(timestamp)
        assert result is not None
        assert result.year == 2023

    def test_from_timestamp_invalid(self):
        # Very large timestamp that would cause overflow
        assert _from_timestamp(99999999999999) is None


# =============================================================================
# IntercomContact Model Tests
# =============================================================================


class TestIntercomContactModel:
    """Tests for IntercomContact dataclass."""

    def test_from_api_full(self):
        data = {
            "id": "contact_123",
            "external_id": "user_456",
            "email": "john@example.com",
            "name": "John Doe",
            "role": "user",
            "phone": "+1-555-0100",
            "avatar": "https://example.com/avatar.png",
            "owner_id": 789,
            "location_data": {"city": "San Francisco", "country": "USA"},
            "custom_attributes": {"plan": "enterprise"},
            "tags": {"data": [{"name": "vip"}, {"name": "enterprise"}]},
            "created_at": 1686827400,
            "updated_at": 1686913800,
            "signed_up_at": 1686741000,
            "last_seen_at": 1686913800,
        }
        contact = IntercomContact.from_api(data)
        assert contact.id == "contact_123"
        assert contact.external_id == "user_456"
        assert contact.email == "john@example.com"
        assert contact.name == "John Doe"
        assert contact.role == ContactRole.USER
        assert contact.phone == "+1-555-0100"
        assert contact.location_data == {"city": "San Francisco", "country": "USA"}
        assert contact.custom_attributes == {"plan": "enterprise"}
        assert contact.tags == ["vip", "enterprise"]

    def test_from_api_minimal(self):
        data = {"id": "contact_1", "role": "user"}
        contact = IntercomContact.from_api(data)
        assert contact.id == "contact_1"
        assert contact.external_id is None
        assert contact.email is None
        assert contact.name is None
        assert contact.tags == []

    def test_from_api_lead_role(self):
        data = {"id": "lead_1", "role": "lead"}
        contact = IntercomContact.from_api(data)
        assert contact.role == ContactRole.LEAD

    def test_from_api_empty_tags(self):
        data = {"id": "contact_1", "role": "user", "tags": {"data": []}}
        contact = IntercomContact.from_api(data)
        assert contact.tags == []


# =============================================================================
# IntercomCompany Model Tests
# =============================================================================


class TestIntercomCompanyModel:
    """Tests for IntercomCompany dataclass."""

    def test_from_api_full(self):
        data = {
            "id": "company_123",
            "company_id": "acme_corp",
            "name": "Acme Corporation",
            "plan": {"name": "Enterprise"},
            "size": 500,
            "website": "https://acme.com",
            "industry": "Technology",
            "monthly_spend": 5000.0,
            "session_count": 1000,
            "user_count": 50,
            "custom_attributes": {"tier": "gold"},
            "tags": {"data": [{"name": "enterprise"}]},
            "created_at": 1686827400,
            "updated_at": 1686913800,
        }
        company = IntercomCompany.from_api(data)
        assert company.id == "company_123"
        assert company.company_id == "acme_corp"
        assert company.name == "Acme Corporation"
        assert company.plan == "Enterprise"
        assert company.size == 500
        assert company.website == "https://acme.com"
        assert company.industry == "Technology"
        assert company.monthly_spend == 5000.0
        assert company.session_count == 1000
        assert company.user_count == 50
        assert company.tags == ["enterprise"]

    def test_from_api_minimal(self):
        data = {"id": "company_1", "company_id": "comp_1", "name": "Test"}
        company = IntercomCompany.from_api(data)
        assert company.id == "company_1"
        assert company.plan is None
        assert company.size is None
        assert company.session_count == 0
        assert company.user_count == 0

    def test_from_api_no_plan(self):
        data = {"id": "company_1", "company_id": "comp_1", "name": "Test", "plan": None}
        company = IntercomCompany.from_api(data)
        assert company.plan is None


# =============================================================================
# ConversationPart Model Tests
# =============================================================================


class TestConversationPartModel:
    """Tests for ConversationPart dataclass."""

    def test_from_api_full(self):
        data = {
            "id": "part_123",
            "part_type": "comment",
            "body": "Hello, how can I help?",
            "author": {
                "type": "admin",
                "id": "admin_456",
                "name": "Support Agent",
            },
            "created_at": 1686827400,
            "attachments": [{"url": "https://example.com/file.pdf"}],
        }
        part = ConversationPart.from_api(data)
        assert part.id == "part_123"
        assert part.part_type == MessageType.COMMENT
        assert part.body == "Hello, how can I help?"
        assert part.author_type == "admin"
        assert part.author_id == "admin_456"
        assert part.author_name == "Support Agent"
        assert len(part.attachments) == 1

    def test_from_api_minimal(self):
        data = {"id": "part_1", "part_type": "comment", "author": {}}
        part = ConversationPart.from_api(data)
        assert part.id == "part_1"
        assert part.body is None
        assert part.author_type == ""
        assert part.author_id == ""

    def test_from_api_note_type(self):
        data = {"id": "part_1", "part_type": "note", "author": {}}
        part = ConversationPart.from_api(data)
        assert part.part_type == MessageType.NOTE

    def test_from_api_assignment_type(self):
        data = {"id": "part_1", "part_type": "assignment", "author": {}}
        part = ConversationPart.from_api(data)
        assert part.part_type == MessageType.ASSIGNMENT


# =============================================================================
# Conversation Model Tests
# =============================================================================


class TestConversationModel:
    """Tests for Conversation dataclass."""

    def test_from_api_full(self):
        data = {
            "id": "conv_123",
            "title": "Help with billing",
            "state": "open",
            "open": True,
            "read": False,
            "priority": "priority",
            "source": {
                "type": "email",
                "author": {"type": "user", "id": "user_123"},
            },
            "assignee": {"type": "admin", "id": "admin_456"},
            "contacts": {"contacts": [{"id": "contact_789"}]},
            "tags": {"tags": [{"name": "billing"}, {"name": "urgent"}]},
            "created_at": 1686827400,
            "updated_at": 1686913800,
            "waiting_since": 1686827400,
            "snoozed_until": None,
            "first_contact_reply_at": 1686830000,
        }
        conv = Conversation.from_api(data)
        assert conv.id == "conv_123"
        assert conv.title == "Help with billing"
        assert conv.state == ConversationState.OPEN
        assert conv.open is True
        assert conv.read is False
        assert conv.priority == "priority"
        assert conv.source_type == "email"
        assert conv.source_author_type == "user"
        assert conv.source_author_id == "user_123"
        assert conv.assignee_type == "admin"
        assert conv.assignee_id == "admin_456"
        assert conv.contacts == ["contact_789"]
        assert conv.tags == ["billing", "urgent"]

    def test_from_api_minimal(self):
        data = {
            "id": "conv_1",
            "state": "open",
            "open": True,
            "read": False,
            "priority": "not_priority",
            "source": {"type": "chat"},
        }
        conv = Conversation.from_api(data)
        assert conv.id == "conv_1"
        assert conv.title is None
        assert conv.assignee_type is None
        assert conv.contacts == []
        assert conv.tags == []

    def test_from_api_closed_state(self):
        data = {
            "id": "conv_1",
            "state": "closed",
            "open": False,
            "read": True,
            "priority": "not_priority",
            "source": {"type": "email"},
        }
        conv = Conversation.from_api(data)
        assert conv.state == ConversationState.CLOSED
        assert conv.open is False

    def test_from_api_snoozed_state(self):
        data = {
            "id": "conv_1",
            "state": "snoozed",
            "open": True,
            "read": False,
            "priority": "not_priority",
            "source": {"type": "email"},
            "snoozed_until": 1686913800,
        }
        conv = Conversation.from_api(data)
        assert conv.state == ConversationState.SNOOZED
        assert conv.snoozed_until is not None


# =============================================================================
# Admin Model Tests
# =============================================================================


class TestAdminModel:
    """Tests for Admin dataclass."""

    def test_from_api_full(self):
        data = {
            "id": "admin_123",
            "name": "Support Agent",
            "email": "agent@company.com",
            "type": "admin",
            "away_mode_enabled": True,
            "away_mode_reassign": True,
            "has_inbox_seat": True,
            "team_ids": [{"id": "team_1"}, {"id": "team_2"}],
        }
        admin = Admin.from_api(data)
        assert admin.id == "admin_123"
        assert admin.name == "Support Agent"
        assert admin.email == "agent@company.com"
        assert admin.away_mode_enabled is True
        assert admin.away_mode_reassign is True
        assert admin.has_inbox_seat is True
        assert admin.team_ids == ["team_1", "team_2"]

    def test_from_api_minimal(self):
        data = {"id": "admin_1", "name": "", "email": ""}
        admin = Admin.from_api(data)
        assert admin.id == "admin_1"
        assert admin.away_mode_enabled is False
        assert admin.team_ids == []


# =============================================================================
# Article Model Tests
# =============================================================================


class TestArticleModel:
    """Tests for Article dataclass."""

    def test_from_api_full(self):
        data = {
            "id": "article_123",
            "title": "How to reset your password",
            "body": "<p>Follow these steps...</p>",
            "description": "Password reset guide",
            "author_id": "admin_456",
            "state": "published",
            "url": "https://help.example.com/articles/password-reset",
            "parent_id": "collection_789",
            "parent_type": "collection",
            "created_at": 1686827400,
            "updated_at": 1686913800,
        }
        article = Article.from_api(data)
        assert article.id == "article_123"
        assert article.title == "How to reset your password"
        assert article.body == "<p>Follow these steps...</p>"
        assert article.description == "Password reset guide"
        assert article.author_id == "admin_456"
        assert article.state == "published"
        assert article.url == "https://help.example.com/articles/password-reset"
        assert article.parent_id == "collection_789"

    def test_from_api_minimal(self):
        data = {"id": "article_1", "title": "Test", "body": "Content"}
        article = Article.from_api(data)
        assert article.id == "article_1"
        assert article.state == "draft"
        assert article.url is None
        assert article.parent_id is None


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestIntercomConnectorInit:
    """Tests for IntercomConnector initialization and context manager."""

    def test_init(self, credentials):
        connector = IntercomConnector(credentials)
        assert connector.credentials is credentials
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, credentials):
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client = AsyncMock()
            mock_async_client_class.return_value = mock_async_client

            async with IntercomConnector(credentials) as connector:
                await connector._get_client()
                assert connector._client is mock_async_client

            mock_async_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exit_clears_client(self, credentials):
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client = AsyncMock()
            mock_async_client_class.return_value = mock_async_client

            connector = IntercomConnector(credentials)
            await connector.__aenter__()
            await connector._get_client()
            assert connector._client is not None
            await connector.__aexit__(None, None, None)
            assert connector._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, credentials):
        connector = IntercomConnector(credentials)
        assert connector._client is None
        await connector.close()
        assert connector._client is None


# =============================================================================
# API Request Tests
# =============================================================================


class TestApiRequest:
    """Tests for _request method."""

    @pytest.mark.asyncio
    async def test_request_success(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"id": "123"})
        result = await intercom_connector._request("GET", "/contacts/123")
        assert result["id"] == "123"

    @pytest.mark.asyncio
    async def test_request_http_error(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"message": "Not found", "errors": [{"code": "not_found"}]},
            status_code=404,
        )
        with pytest.raises(IntercomError) as exc_info:
            await intercom_connector._request("GET", "/contacts/999")
        assert exc_info.value.status_code == 404
        assert len(exc_info.value.errors) == 1

    @pytest.mark.asyncio
    async def test_request_http_error_no_json(self, intercom_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        resp.json.side_effect = ValueError("No JSON")
        mock_httpx_client.request.return_value = resp
        with pytest.raises(IntercomError) as exc_info:
            await intercom_connector._request("GET", "/test")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_204_no_content(self, intercom_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        mock_httpx_client.request.return_value = resp
        result = await intercom_connector._request("DELETE", "/contacts/123/tags/456")
        assert result == {}

    @pytest.mark.asyncio
    async def test_request_rate_limit_429(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"message": "Rate limit exceeded"},
            status_code=429,
        )
        with pytest.raises(IntercomError) as exc_info:
            await intercom_connector._request("GET", "/conversations")
        assert exc_info.value.status_code == 429


# =============================================================================
# Conversation Operations Tests
# =============================================================================


class TestConversationOperations:
    """Tests for conversation operations."""

    @pytest.mark.asyncio
    async def test_get_conversations(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "conversations": [
                    {
                        "id": "conv_1",
                        "state": "open",
                        "open": True,
                        "read": False,
                        "priority": "not_priority",
                        "source": {"type": "email"},
                    },
                    {
                        "id": "conv_2",
                        "state": "closed",
                        "open": False,
                        "read": True,
                        "priority": "priority",
                        "source": {"type": "chat"},
                    },
                ],
                "pages": {"next": {"starting_after": "conv_2"}},
            }
        )
        conversations, next_cursor = await intercom_connector.get_conversations()
        assert len(conversations) == 2
        assert conversations[0].id == "conv_1"
        assert next_cursor == "conv_2"

    @pytest.mark.asyncio
    async def test_get_conversations_with_state_filter(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"conversations": [], "pages": {}}
        )
        await intercom_connector.get_conversations(state=ConversationState.OPEN)
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["state"] == "open"

    @pytest.mark.asyncio
    async def test_get_conversations_with_assignee_filter(
        self, intercom_connector, mock_httpx_client
    ):
        mock_httpx_client.request.return_value = _make_response(
            {"conversations": [], "pages": {}}
        )
        await intercom_connector.get_conversations(assignee_id="admin_123")
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["assignee_id"] == "admin_123"

    @pytest.mark.asyncio
    async def test_get_conversation(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "conv_123",
                "state": "open",
                "open": True,
                "read": False,
                "priority": "priority",
                "source": {"type": "email"},
                "conversation_parts": {
                    "conversation_parts": [
                        {"id": "part_1", "part_type": "comment", "body": "Hello", "author": {}},
                    ]
                },
            }
        )
        conv = await intercom_connector.get_conversation("conv_123")
        assert conv.id == "conv_123"
        assert len(conv.conversation_parts) == 1
        assert conv.conversation_parts[0].body == "Hello"

    @pytest.mark.asyncio
    async def test_reply_to_conversation(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "conv_123",
                "state": "open",
                "open": True,
                "read": False,
                "priority": "not_priority",
                "source": {"type": "email"},
            }
        )
        conv = await intercom_connector.reply_to_conversation(
            "conv_123",
            body="Thanks for reaching out!",
            admin_id="admin_456",
            attachment_urls=["https://example.com/file.pdf"],
        )
        assert conv.id == "conv_123"

    @pytest.mark.asyncio
    async def test_close_conversation(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "conv_123",
                "state": "closed",
                "open": False,
                "read": True,
                "priority": "not_priority",
                "source": {"type": "email"},
            }
        )
        conv = await intercom_connector.close_conversation(
            "conv_123",
            admin_id="admin_456",
            body="Closing this ticket as resolved.",
        )
        assert conv.state == ConversationState.CLOSED

    @pytest.mark.asyncio
    async def test_assign_conversation(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "conv_123",
                "state": "open",
                "open": True,
                "read": False,
                "priority": "not_priority",
                "source": {"type": "email"},
                "assignee": {"type": "admin", "id": "admin_789"},
            }
        )
        conv = await intercom_connector.assign_conversation(
            "conv_123",
            admin_id="admin_456",
            assignee_id="admin_789",
            body="Assigning to specialist.",
        )
        assert conv.assignee_id == "admin_789"

    @pytest.mark.asyncio
    async def test_snooze_conversation(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "conv_123",
                "state": "snoozed",
                "open": True,
                "read": False,
                "priority": "not_priority",
                "source": {"type": "email"},
                "snoozed_until": 1686913800,
            }
        )
        snooze_time = datetime(2023, 6, 16, 10, 30, 0, tzinfo=timezone.utc)
        conv = await intercom_connector.snooze_conversation(
            "conv_123",
            admin_id="admin_456",
            snoozed_until=snooze_time,
        )
        assert conv.state == ConversationState.SNOOZED


# =============================================================================
# Contact Operations Tests
# =============================================================================


class TestContactOperations:
    """Tests for contact CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_contacts(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "data": [
                    {"id": "contact_1", "name": "User 1", "email": "user1@test.com", "role": "user"},
                    {"id": "contact_2", "name": "User 2", "email": "user2@test.com", "role": "lead"},
                ],
                "pages": {"next": {"starting_after": "contact_2"}},
            }
        )
        contacts, next_cursor = await intercom_connector.get_contacts()
        assert len(contacts) == 2
        assert contacts[0].role == ContactRole.USER
        assert contacts[1].role == ContactRole.LEAD
        assert next_cursor == "contact_2"

    @pytest.mark.asyncio
    async def test_get_contacts_pagination(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"data": [], "pages": {}})
        await intercom_connector.get_contacts(per_page=100, starting_after="cursor_123")
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["per_page"] == 100
        assert call_args.kwargs["params"]["starting_after"] == "cursor_123"

    @pytest.mark.asyncio
    async def test_get_contact(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"id": "contact_123", "name": "John Doe", "email": "john@test.com", "role": "user"}
        )
        contact = await intercom_connector.get_contact("contact_123")
        assert contact.id == "contact_123"
        assert contact.name == "John Doe"

    @pytest.mark.asyncio
    async def test_search_contacts(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "data": [
                    {"id": "contact_1", "name": "John", "email": "john@test.com", "role": "user"},
                ]
            }
        )
        query = {"field": "email", "operator": "=", "value": "john@test.com"}
        contacts = await intercom_connector.search_contacts(query)
        assert len(contacts) == 1
        assert contacts[0].email == "john@test.com"

    @pytest.mark.asyncio
    async def test_create_contact(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "contact_new",
                "email": "new@test.com",
                "name": "New User",
                "role": "user",
            }
        )
        contact = await intercom_connector.create_contact(
            role=ContactRole.USER,
            email="new@test.com",
            external_id="ext_123",
            name="New User",
            phone="+1-555-0100",
            custom_attributes={"plan": "starter"},
        )
        assert contact.id == "contact_new"
        assert contact.email == "new@test.com"

    @pytest.mark.asyncio
    async def test_create_lead(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"id": "lead_new", "email": "lead@test.com", "role": "lead"}
        )
        contact = await intercom_connector.create_contact(
            role=ContactRole.LEAD,
            email="lead@test.com",
        )
        assert contact.role == ContactRole.LEAD

    @pytest.mark.asyncio
    async def test_update_contact(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"id": "contact_123", "name": "Updated Name", "email": "updated@test.com", "role": "user"}
        )
        contact = await intercom_connector.update_contact(
            "contact_123",
            name="Updated Name",
            email="updated@test.com",
            phone="+1-555-0200",
            custom_attributes={"level": "premium"},
        )
        assert contact.name == "Updated Name"


# =============================================================================
# Company Operations Tests
# =============================================================================


class TestCompanyOperations:
    """Tests for company CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_companies(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "data": [
                    {"id": "company_1", "company_id": "acme", "name": "Acme Corp"},
                    {"id": "company_2", "company_id": "tech", "name": "Tech Inc"},
                ],
                "pages": {"next": {"starting_after": "company_2"}},
            }
        )
        companies, next_cursor = await intercom_connector.get_companies()
        assert len(companies) == 2
        assert companies[0].name == "Acme Corp"
        assert next_cursor == "company_2"

    @pytest.mark.asyncio
    async def test_get_company(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "company_123",
                "company_id": "acme",
                "name": "Acme Corporation",
                "size": 100,
                "user_count": 25,
            }
        )
        company = await intercom_connector.get_company("company_123")
        assert company.id == "company_123"
        assert company.size == 100
        assert company.user_count == 25

    @pytest.mark.asyncio
    async def test_create_or_update_company(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "company_new",
                "company_id": "newcorp",
                "name": "New Corporation",
                "size": 50,
                "website": "https://newcorp.com",
                "industry": "SaaS",
            }
        )
        company = await intercom_connector.create_or_update_company(
            company_id="newcorp",
            name="New Corporation",
            plan="Enterprise",
            size=50,
            website="https://newcorp.com",
            industry="SaaS",
            custom_attributes={"tier": "gold"},
        )
        assert company.company_id == "newcorp"
        assert company.name == "New Corporation"


# =============================================================================
# Admin Operations Tests
# =============================================================================


class TestAdminOperations:
    """Tests for admin read operations."""

    @pytest.mark.asyncio
    async def test_get_admins(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "admins": [
                    {"id": "admin_1", "name": "Admin 1", "email": "admin1@test.com"},
                    {"id": "admin_2", "name": "Admin 2", "email": "admin2@test.com"},
                ]
            }
        )
        admins = await intercom_connector.get_admins()
        assert len(admins) == 2
        assert admins[0].name == "Admin 1"

    @pytest.mark.asyncio
    async def test_get_admin(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"id": "admin_123", "name": "Support Agent", "email": "agent@test.com"}
        )
        admin = await intercom_connector.get_admin("admin_123")
        assert admin.id == "admin_123"
        assert admin.name == "Support Agent"


# =============================================================================
# Article Operations Tests
# =============================================================================


class TestArticleOperations:
    """Tests for article CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_articles(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "data": [
                    {"id": "article_1", "title": "Getting Started", "body": "Content..."},
                    {"id": "article_2", "title": "FAQ", "body": "Questions..."},
                ],
                "pages": {"next": {"starting_after": "article_2"}},
            }
        )
        articles, next_cursor = await intercom_connector.get_articles()
        assert len(articles) == 2
        assert articles[0].title == "Getting Started"
        assert next_cursor == "article_2"

    @pytest.mark.asyncio
    async def test_get_article(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "article_123",
                "title": "How to Reset Password",
                "body": "<p>Steps...</p>",
                "state": "published",
            }
        )
        article = await intercom_connector.get_article("article_123")
        assert article.id == "article_123"
        assert article.state == "published"

    @pytest.mark.asyncio
    async def test_create_article(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": "article_new",
                "title": "New Article",
                "body": "<p>Content</p>",
                "author_id": "admin_123",
                "state": "draft",
            }
        )
        article = await intercom_connector.create_article(
            title="New Article",
            body="<p>Content</p>",
            author_id="admin_123",
            description="A new help article",
            state="draft",
            parent_id="collection_456",
            parent_type="collection",
        )
        assert article.id == "article_new"
        assert article.title == "New Article"


# =============================================================================
# Tag Operations Tests
# =============================================================================


class TestTagOperations:
    """Tests for tag operations."""

    @pytest.mark.asyncio
    async def test_tag_contact(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"type": "tag"})
        result = await intercom_connector.tag_contact("contact_123", "vip")
        assert result is True

    @pytest.mark.asyncio
    async def test_untag_contact(self, intercom_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        mock_httpx_client.request.return_value = resp
        result = await intercom_connector.untag_contact("contact_123", "tag_456")
        assert result is True

    @pytest.mark.asyncio
    async def test_tag_conversation(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"type": "tag"})
        result = await intercom_connector.tag_conversation("conv_123", "billing", "admin_456")
        assert result is True


# =============================================================================
# Mock Data Generation Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data helpers."""

    def test_get_mock_conversation(self):
        conv = get_mock_conversation()
        assert conv.id == "12345"
        assert conv.title == "Help with billing"
        assert conv.state == ConversationState.OPEN
        assert conv.open is True
        assert conv.read is False
        assert conv.priority == "priority"
        assert conv.source_type == "email"
        assert "contact_123" in conv.contacts
        assert conv.created_at is not None

    def test_get_mock_contact(self):
        contact = get_mock_contact()
        assert contact.id == "contact_123"
        assert contact.external_id == "user_456"
        assert contact.email == "user@example.com"
        assert contact.name == "Test User"
        assert contact.role == ContactRole.USER


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_contact_from_api_no_tags_data(self):
        data = {"id": "contact_1", "role": "user", "tags": {}}
        contact = IntercomContact.from_api(data)
        assert contact.tags == []

    def test_company_from_api_no_plan_data(self):
        data = {"id": "company_1", "company_id": "comp", "name": "Test", "plan": {}}
        company = IntercomCompany.from_api(data)
        assert company.plan is None

    def test_conversation_from_api_no_assignee(self):
        data = {
            "id": "conv_1",
            "state": "open",
            "open": True,
            "read": False,
            "priority": "not_priority",
            "source": {"type": "email"},
            "assignee": None,
        }
        conv = Conversation.from_api(data)
        assert conv.assignee_type is None
        assert conv.assignee_id is None

    def test_conversation_from_api_empty_contacts(self):
        data = {
            "id": "conv_1",
            "state": "open",
            "open": True,
            "read": False,
            "priority": "not_priority",
            "source": {"type": "email"},
            "contacts": {"contacts": []},
        }
        conv = Conversation.from_api(data)
        assert conv.contacts == []

    def test_contact_defaults(self):
        contact = IntercomContact(
            id="test",
            external_id=None,
            email=None,
            name=None,
            role=ContactRole.USER,
        )
        assert contact.phone is None
        assert contact.location_data == {}
        assert contact.custom_attributes == {}
        assert contact.tags == []

    def test_company_defaults(self):
        company = IntercomCompany(
            id="test",
            company_id="test_comp",
            name="Test",
        )
        assert company.plan is None
        assert company.size is None
        assert company.session_count == 0
        assert company.user_count == 0
        assert company.custom_attributes == {}
        assert company.tags == []

    def test_conversation_defaults(self):
        conv = Conversation(
            id="test",
            title=None,
            state=ConversationState.OPEN,
            open=True,
            read=False,
            priority="not_priority",
            source_type="email",
        )
        assert conv.assignee_type is None
        assert conv.contacts == []
        assert conv.tags == []
        assert conv.conversation_parts == []

    @pytest.mark.asyncio
    async def test_get_contacts_empty_response(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"data": [], "pages": {}})
        contacts, next_cursor = await intercom_connector.get_contacts()
        assert contacts == []
        assert next_cursor is None

    @pytest.mark.asyncio
    async def test_get_conversations_empty_response(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"conversations": [], "pages": {}}
        )
        conversations, next_cursor = await intercom_connector.get_conversations()
        assert conversations == []
        assert next_cursor is None

    @pytest.mark.asyncio
    async def test_search_contacts_empty_response(self, intercom_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"data": []})
        query = {"field": "email", "operator": "=", "value": "nonexistent@test.com"}
        contacts = await intercom_connector.search_contacts(query)
        assert contacts == []
