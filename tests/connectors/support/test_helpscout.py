"""
Comprehensive tests for the Help Scout Support Connector.

Tests cover:
- Enum values (ConversationStatus, ConversationType, ThreadType, ThreadStatus)
- HelpScoutCredentials dataclass
- Data models: HelpScoutCustomer, User, Mailbox, Thread, Conversation, Folder
- Model serialization (from_api)
- HelpScoutConnector initialization and async context manager
- OAuth2 token management
- API request handling (_request with auth, error responses)
- Conversations CRUD operations (get, create, update, reply, note)
- Customer CRUD operations
- Mailbox operations
- User operations
- Search operations
- Helper functions (_parse_datetime)
- Error handling (HTTP errors, JSON parsing)
- Edge cases (empty responses, missing fields, malformed data)
- Mock data generation helpers
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.support.helpscout import (
    Conversation,
    ConversationStatus,
    ConversationType,
    Folder,
    HelpScoutConnector,
    HelpScoutCredentials,
    HelpScoutCustomer,
    HelpScoutError,
    Mailbox,
    Thread,
    ThreadStatus,
    ThreadType,
    User,
    _parse_datetime,
    get_mock_conversation,
    get_mock_customer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credentials():
    """Standard test credentials."""
    return HelpScoutCredentials(
        client_id="test-client-id",
        client_secret="test-client-secret",
    )


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def helpscout_connector(credentials, mock_httpx_client):
    """Create a HelpScoutConnector with a mock HTTP client."""
    connector = HelpScoutConnector(credentials)
    connector._client = mock_httpx_client
    connector._access_token = "test-access-token"
    connector._token_expires_at = datetime.now() + timedelta(hours=1)
    return connector


def _make_response(
    json_data: dict[str, Any], status_code: int = 200, headers: dict | None = None
) -> MagicMock:
    """Build a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    resp.headers = headers or {}
    return resp


# =============================================================================
# Enum Tests
# =============================================================================


class TestConversationStatus:
    """Tests for ConversationStatus enum."""

    def test_conversation_status_values(self):
        assert ConversationStatus.ACTIVE.value == "active"
        assert ConversationStatus.PENDING.value == "pending"
        assert ConversationStatus.CLOSED.value == "closed"
        assert ConversationStatus.SPAM.value == "spam"

    def test_conversation_status_is_str(self):
        assert isinstance(ConversationStatus.ACTIVE, str)
        assert ConversationStatus.ACTIVE == "active"

    def test_conversation_status_from_string(self):
        assert ConversationStatus("pending") == ConversationStatus.PENDING


class TestConversationType:
    """Tests for ConversationType enum."""

    def test_conversation_type_values(self):
        assert ConversationType.EMAIL.value == "email"
        assert ConversationType.CHAT.value == "chat"
        assert ConversationType.PHONE.value == "phone"

    def test_conversation_type_is_str(self):
        assert isinstance(ConversationType.EMAIL, str)


class TestThreadType:
    """Tests for ThreadType enum."""

    def test_thread_type_values(self):
        assert ThreadType.CUSTOMER.value == "customer"
        assert ThreadType.MESSAGE.value == "message"
        assert ThreadType.NOTE.value == "note"
        assert ThreadType.FORWARD.value == "forwardparent"
        assert ThreadType.REPLY_FORWARD.value == "reply"

    def test_thread_type_is_str(self):
        assert isinstance(ThreadType.NOTE, str)


class TestThreadStatus:
    """Tests for ThreadStatus enum."""

    def test_thread_status_values(self):
        assert ThreadStatus.ACTIVE.value == "active"
        assert ThreadStatus.CLOSED.value == "closed"
        assert ThreadStatus.PENDING.value == "pending"
        assert ThreadStatus.NO_CHANGE.value == "nochange"

    def test_thread_status_is_str(self):
        assert isinstance(ThreadStatus.ACTIVE, str)


# =============================================================================
# Credentials Tests
# =============================================================================


class TestHelpScoutCredentials:
    """Tests for HelpScoutCredentials dataclass."""

    def test_basic_construction(self):
        creds = HelpScoutCredentials(
            client_id="my-client-id",
            client_secret="my-client-secret",
        )
        assert creds.client_id == "my-client-id"
        assert creds.client_secret == "my-client-secret"

    def test_default_base_url(self):
        creds = HelpScoutCredentials(
            client_id="test",
            client_secret="secret",
        )
        assert creds.base_url == "https://api.helpscout.net/v2"

    def test_custom_base_url(self):
        creds = HelpScoutCredentials(
            client_id="test",
            client_secret="secret",
            base_url="https://custom.helpscout.net/v2",
        )
        assert creds.base_url == "https://custom.helpscout.net/v2"


# =============================================================================
# Error Tests
# =============================================================================


class TestHelpScoutError:
    """Tests for HelpScoutError exception."""

    def test_error_basic(self):
        err = HelpScoutError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.status_code is None
        assert err.details == {}

    def test_error_with_status(self):
        err = HelpScoutError("Not found", status_code=404, details={"error": "resource_not_found"})
        assert err.status_code == 404
        assert err.details == {"error": "resource_not_found"}

    def test_error_with_all_params(self):
        err = HelpScoutError(
            "Rate limit exceeded",
            status_code=429,
            details={"retry_after": 60},
        )
        assert str(err) == "Rate limit exceeded"
        assert err.status_code == 429
        assert err.details["retry_after"] == 60


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
# HelpScoutCustomer Model Tests
# =============================================================================


class TestHelpScoutCustomerModel:
    """Tests for HelpScoutCustomer dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 12345,
            "firstName": "Alice",
            "lastName": "Johnson",
            "emails": [
                {"value": "alice@example.com", "type": "work"},
                {"value": "alice.personal@example.com", "type": "home"},
            ],
            "phones": [{"value": "+1-555-0100", "type": "work"}],
            "organization": "Acme Corp",
            "jobTitle": "Product Manager",
            "location": "San Francisco, CA",
            "photoUrl": "https://example.com/photo.jpg",
            "createdAt": "2023-01-15T10:00:00Z",
            "updatedAt": "2023-06-20T14:30:00Z",
        }
        customer = HelpScoutCustomer.from_api(data)
        assert customer.id == 12345
        assert customer.first_name == "Alice"
        assert customer.last_name == "Johnson"
        assert len(customer.emails) == 2
        assert customer.emails[0] == "alice@example.com"
        assert len(customer.phones) == 1
        assert customer.organization == "Acme Corp"
        assert customer.job_title == "Product Manager"
        assert customer.location == "San Francisco, CA"
        assert customer.photo_url == "https://example.com/photo.jpg"
        assert customer.created_at is not None

    def test_from_api_minimal(self):
        data = {"id": 1, "firstName": "", "lastName": ""}
        customer = HelpScoutCustomer.from_api(data)
        assert customer.id == 1
        assert customer.first_name == ""
        assert customer.last_name == ""
        assert customer.emails == []
        assert customer.phones == []
        assert customer.organization is None

    def test_full_name_property(self):
        customer = HelpScoutCustomer(
            id=1,
            first_name="John",
            last_name="Doe",
        )
        assert customer.full_name == "John Doe"

    def test_full_name_with_only_first_name(self):
        customer = HelpScoutCustomer(
            id=1,
            first_name="John",
            last_name="",
        )
        assert customer.full_name == "John"

    def test_from_api_empty_emails(self):
        data = {"id": 1, "firstName": "Test", "lastName": "User", "emails": []}
        customer = HelpScoutCustomer.from_api(data)
        assert customer.emails == []


# =============================================================================
# User Model Tests
# =============================================================================


class TestUserModel:
    """Tests for User dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 67890,
            "firstName": "Jane",
            "lastName": "Smith",
            "email": "jane@company.com",
            "role": "admin",
            "timezone": "America/New_York",
            "photoUrl": "https://example.com/jane.jpg",
            "type": "user",
            "createdAt": "2023-01-01T00:00:00Z",
            "updatedAt": "2023-06-01T12:00:00Z",
        }
        user = User.from_api(data)
        assert user.id == 67890
        assert user.first_name == "Jane"
        assert user.last_name == "Smith"
        assert user.email == "jane@company.com"
        assert user.role == "admin"
        assert user.timezone == "America/New_York"
        assert user.photo_url == "https://example.com/jane.jpg"
        assert user.type == "user"

    def test_from_api_minimal(self):
        data = {"id": 1, "firstName": "", "lastName": "", "email": ""}
        user = User.from_api(data)
        assert user.id == 1
        assert user.role == "user"
        assert user.type == "user"

    def test_full_name_property(self):
        user = User(
            id=1,
            first_name="Jane",
            last_name="Doe",
            email="jane@example.com",
        )
        assert user.full_name == "Jane Doe"


# =============================================================================
# Mailbox Model Tests
# =============================================================================


class TestMailboxModel:
    """Tests for Mailbox dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 100,
            "name": "Support",
            "slug": "support",
            "email": "support@company.com",
            "createdAt": "2023-01-01T00:00:00Z",
            "updatedAt": "2023-06-01T12:00:00Z",
        }
        mailbox = Mailbox.from_api(data)
        assert mailbox.id == 100
        assert mailbox.name == "Support"
        assert mailbox.slug == "support"
        assert mailbox.email == "support@company.com"
        assert mailbox.created_at is not None

    def test_from_api_minimal(self):
        data = {"id": 1}
        mailbox = Mailbox.from_api(data)
        assert mailbox.id == 1
        assert mailbox.name == ""
        assert mailbox.slug == ""
        assert mailbox.email == ""


# =============================================================================
# Thread Model Tests
# =============================================================================


class TestThreadModel:
    """Tests for Thread dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 500,
            "type": "message",
            "body": "Thank you for contacting us.",
            "status": "active",
            "state": "published",
            "source": {"type": "email"},
            "createdBy": {"type": "user", "id": 123},
            "assignedTo": {"id": 456},
            "createdAt": "2023-07-10T09:00:00Z",
            "_embedded": {
                "attachments": [{"filename": "screenshot.png", "url": "https://..."}],
            },
        }
        thread = Thread.from_api(data)
        assert thread.id == 500
        assert thread.type == ThreadType.MESSAGE
        assert thread.body == "Thank you for contacting us."
        assert thread.status == ThreadStatus.ACTIVE
        assert thread.state == "published"
        assert thread.source_type == "email"
        assert thread.created_by_customer is False
        assert thread.assigned_to_id == 456
        assert len(thread.attachments) == 1

    def test_from_api_minimal(self):
        data = {"id": 1, "type": "customer", "body": "", "status": "active", "state": "published"}
        thread = Thread.from_api(data)
        assert thread.id == 1
        assert thread.type == ThreadType.CUSTOMER
        assert thread.attachments == []

    def test_from_api_customer_created(self):
        data = {
            "id": 2,
            "type": "customer",
            "body": "I need help",
            "status": "active",
            "state": "published",
            "createdBy": {"type": "customer", "id": 999},
        }
        thread = Thread.from_api(data)
        assert thread.created_by_customer is True
        assert thread.customer_id == 999

    def test_from_api_note_type(self):
        data = {
            "id": 3,
            "type": "note",
            "body": "Internal note",
            "status": "active",
            "state": "published",
        }
        thread = Thread.from_api(data)
        assert thread.type == ThreadType.NOTE


# =============================================================================
# Conversation Model Tests
# =============================================================================


class TestConversationModel:
    """Tests for Conversation dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 1000,
            "number": 1001,
            "subject": "Cannot login to my account",
            "status": "active",
            "type": "email",
            "mailboxId": 100,
            "assignee": {"id": 123},
            "primaryCustomer": {"id": 456, "email": "customer@example.com"},
            "tags": [{"tag": "urgent"}, {"tag": "login"}],
            "cc": ["manager@example.com"],
            "bcc": ["archive@example.com"],
            "preview": "I tried resetting my password...",
            "createdAt": "2023-07-01T10:00:00Z",
            "updatedAt": "2023-07-01T12:00:00Z",
            "closedAt": None,
            "customFields": [{"id": 1, "value": "enterprise"}],
        }
        conv = Conversation.from_api(data)
        assert conv.id == 1000
        assert conv.number == 1001
        assert conv.subject == "Cannot login to my account"
        assert conv.status == ConversationStatus.ACTIVE
        assert conv.type == ConversationType.EMAIL
        assert conv.mailbox_id == 100
        assert conv.assignee_id == 123
        assert conv.customer_id == 456
        assert conv.customer_email == "customer@example.com"
        assert conv.tags == ["urgent", "login"]
        assert conv.cc == ["manager@example.com"]
        assert conv.bcc == ["archive@example.com"]
        assert conv.preview == "I tried resetting my password..."

    def test_from_api_minimal(self):
        data = {
            "id": 1,
            "number": 1,
            "subject": "",
            "status": "active",
            "type": "email",
            "mailboxId": 1,
        }
        conv = Conversation.from_api(data)
        assert conv.id == 1
        assert conv.assignee_id is None
        assert conv.customer_id is None
        assert conv.tags == []
        assert conv.cc == []

    def test_from_api_all_statuses(self):
        for status in ConversationStatus:
            data = {
                "id": 1,
                "number": 1,
                "subject": "",
                "status": status.value,
                "type": "email",
                "mailboxId": 1,
            }
            conv = Conversation.from_api(data)
            assert conv.status == status

    def test_from_api_all_types(self):
        for conv_type in ConversationType:
            data = {
                "id": 1,
                "number": 1,
                "subject": "",
                "status": "active",
                "type": conv_type.value,
                "mailboxId": 1,
            }
            conv = Conversation.from_api(data)
            assert conv.type == conv_type


# =============================================================================
# Folder Model Tests
# =============================================================================


class TestFolderModel:
    """Tests for Folder dataclass."""

    def test_from_api_full(self):
        data = {
            "id": 10,
            "name": "Inbox",
            "type": "inbox",
            "userId": 123,
            "totalCount": 50,
            "activeCount": 25,
            "updatedAt": "2023-06-01T12:00:00Z",
        }
        folder = Folder.from_api(data)
        assert folder.id == 10
        assert folder.name == "Inbox"
        assert folder.type == "inbox"
        assert folder.user_id == 123
        assert folder.total_count == 50
        assert folder.active_count == 25

    def test_from_api_minimal(self):
        data = {"id": 1, "name": "", "type": ""}
        folder = Folder.from_api(data)
        assert folder.id == 1
        assert folder.total_count == 0
        assert folder.active_count == 0
        assert folder.user_id is None


# =============================================================================
# Client Initialization Tests
# =============================================================================


class TestHelpScoutConnectorInit:
    """Tests for HelpScoutConnector initialization and context manager."""

    def test_init(self, credentials):
        connector = HelpScoutConnector(credentials)
        assert connector.credentials is credentials
        assert connector._client is None
        assert connector._access_token is None
        assert connector._token_expires_at is None

    @pytest.mark.asyncio
    async def test_context_manager(self, credentials):
        import httpx

        with patch.object(httpx, "AsyncClient") as mock_async_client_class:
            mock_async_client = AsyncMock()
            mock_async_client_class.return_value = mock_async_client

            async with HelpScoutConnector(credentials) as connector:
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

            connector = HelpScoutConnector(credentials)
            await connector.__aenter__()
            await connector._get_client()
            assert connector._client is not None
            await connector.__aexit__(None, None, None)
            assert connector._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, credentials):
        connector = HelpScoutConnector(credentials)
        assert connector._client is None
        await connector.close()
        assert connector._client is None


# =============================================================================
# OAuth Token Tests
# =============================================================================


class TestOAuthToken:
    """Tests for OAuth2 token management."""

    @pytest.mark.asyncio
    async def test_ensure_token_returns_existing_valid_token(self, helpscout_connector):
        token = await helpscout_connector._ensure_token()
        assert token == "test-access-token"

    @pytest.mark.asyncio
    async def test_ensure_token_refreshes_expired_token(
        self, helpscout_connector, mock_httpx_client
    ):
        # Expire the token
        helpscout_connector._token_expires_at = datetime.now() - timedelta(hours=1)

        mock_httpx_client.post.return_value = _make_response(
            {
                "access_token": "new-access-token",
                "expires_in": 7200,
            }
        )

        token = await helpscout_connector._ensure_token()
        assert token == "new-access-token"
        assert helpscout_connector._access_token == "new-access-token"

    @pytest.mark.asyncio
    async def test_ensure_token_fetches_new_when_none(self, credentials, mock_httpx_client):
        connector = HelpScoutConnector(credentials)
        connector._client = mock_httpx_client

        mock_httpx_client.post.return_value = _make_response(
            {
                "access_token": "fresh-token",
                "expires_in": 3600,
            }
        )

        token = await connector._ensure_token()
        assert token == "fresh-token"


# =============================================================================
# API Request Tests
# =============================================================================


class TestApiRequest:
    """Tests for _request method."""

    @pytest.mark.asyncio
    async def test_request_success(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response({"id": 1, "name": "Test"})
        result = await helpscout_connector._request("GET", "/conversations/1")
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_request_http_error(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"message": "Resource not found"},
            status_code=404,
        )
        with pytest.raises(HelpScoutError) as exc_info:
            await helpscout_connector._request("GET", "/conversations/999")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_request_http_error_no_json(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        resp.json.side_effect = ValueError("No JSON")
        mock_httpx_client.request.return_value = resp
        with pytest.raises(HelpScoutError) as exc_info:
            await helpscout_connector._request("GET", "/test")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_request_204_no_content(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        resp.headers = {}
        mock_httpx_client.request.return_value = resp
        result = await helpscout_connector._request("DELETE", "/conversations/1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_request_201_with_resource_id(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {"Resource-ID": "12345"}
        mock_httpx_client.request.return_value = resp
        result = await helpscout_connector._request("POST", "/conversations")
        assert result == {"id": "12345"}

    @pytest.mark.asyncio
    async def test_request_rate_limit_429(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {"message": "Rate limit exceeded"},
            status_code=429,
        )
        with pytest.raises(HelpScoutError) as exc_info:
            await helpscout_connector._request("GET", "/conversations")
        assert exc_info.value.status_code == 429


# =============================================================================
# Conversation Operations Tests
# =============================================================================


class TestConversationOperations:
    """Tests for conversation CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_conversations(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {
                    "conversations": [
                        {
                            "id": 1,
                            "number": 101,
                            "subject": "Issue 1",
                            "status": "active",
                            "type": "email",
                            "mailboxId": 1,
                        },
                        {
                            "id": 2,
                            "number": 102,
                            "subject": "Issue 2",
                            "status": "pending",
                            "type": "email",
                            "mailboxId": 1,
                        },
                    ],
                },
                "page": {"totalPages": 1},
            }
        )
        conversations, total_pages = await helpscout_connector.get_conversations()
        assert len(conversations) == 2
        assert conversations[0].subject == "Issue 1"
        assert total_pages == 1

    @pytest.mark.asyncio
    async def test_get_conversations_with_filters(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {"conversations": []},
                "page": {"totalPages": 0},
            }
        )
        await helpscout_connector.get_conversations(
            mailbox_id=100,
            status=ConversationStatus.ACTIVE,
            folder_id=5,
            assigned_to=123,
            tag="urgent",
        )
        call_args = mock_httpx_client.request.call_args
        params = call_args.kwargs["params"]
        assert params["mailbox"] == 100
        assert params["status"] == "active"
        assert params["folder"] == 5
        assert params["assignedTo"] == 123
        assert params["tag"] == "urgent"

    @pytest.mark.asyncio
    async def test_get_conversation(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": 1000,
                "number": 1001,
                "subject": "Help needed",
                "status": "active",
                "type": "email",
                "mailboxId": 100,
            }
        )
        conv = await helpscout_connector.get_conversation(1000)
        assert conv.id == 1000
        assert conv.subject == "Help needed"

    @pytest.mark.asyncio
    async def test_get_conversation_threads(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {
                    "threads": [
                        {
                            "id": 1,
                            "type": "customer",
                            "body": "Help!",
                            "status": "active",
                            "state": "published",
                        },
                        {
                            "id": 2,
                            "type": "message",
                            "body": "How can we help?",
                            "status": "active",
                            "state": "published",
                        },
                    ],
                },
            }
        )
        threads = await helpscout_connector.get_conversation_threads(1000)
        assert len(threads) == 2
        assert threads[0].type == ThreadType.CUSTOMER
        assert threads[1].type == ThreadType.MESSAGE

    @pytest.mark.asyncio
    async def test_create_conversation(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {"Resource-ID": "2000"}
        mock_httpx_client.request.return_value = resp

        conv_id = await helpscout_connector.create_conversation(
            mailbox_id=100,
            customer_email="customer@example.com",
            subject="New issue",
            text="I need help with my order",
            tags=["orders"],
            assigned_to=123,
            cc=["support@example.com"],
        )
        assert conv_id == 2000

    @pytest.mark.asyncio
    async def test_create_conversation_minimal(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {"Resource-ID": "3000"}
        mock_httpx_client.request.return_value = resp

        conv_id = await helpscout_connector.create_conversation(
            mailbox_id=100,
            customer_email="user@example.com",
            subject="Question",
            text="What are your hours?",
        )
        assert conv_id == 3000

    @pytest.mark.asyncio
    async def test_reply_to_conversation(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {}
        mock_httpx_client.request.return_value = resp

        result = await helpscout_connector.reply_to_conversation(
            conversation_id=1000,
            text="Thank you for contacting us!",
            user_id=123,
            cc=["manager@example.com"],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_add_note(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {}
        mock_httpx_client.request.return_value = resp

        result = await helpscout_connector.add_note(
            conversation_id=1000,
            text="Internal note: VIP customer",
            user_id=123,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_conversation(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        resp.headers = {}
        mock_httpx_client.request.return_value = resp

        result = await helpscout_connector.update_conversation(
            conversation_id=1000,
            op="replace",
            path="/status",
            value="closed",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_assign_conversation(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        resp.headers = {}
        mock_httpx_client.request.return_value = resp

        result = await helpscout_connector.assign_conversation(1000, 456)
        assert result is True

    @pytest.mark.asyncio
    async def test_close_conversation(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        resp.headers = {}
        mock_httpx_client.request.return_value = resp

        result = await helpscout_connector.close_conversation(1000)
        assert result is True

    @pytest.mark.asyncio
    async def test_add_tags(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 204
        resp.headers = {}
        mock_httpx_client.request.return_value = resp

        result = await helpscout_connector.add_tags(1000, ["urgent", "billing"])
        assert result is True


# =============================================================================
# Customer Operations Tests
# =============================================================================


class TestCustomerOperations:
    """Tests for customer CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_customers(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {
                    "customers": [
                        {"id": 1, "firstName": "Alice", "lastName": "Smith"},
                        {"id": 2, "firstName": "Bob", "lastName": "Jones"},
                    ],
                },
                "page": {"totalPages": 1},
            }
        )
        customers, total_pages = await helpscout_connector.get_customers()
        assert len(customers) == 2
        assert customers[0].first_name == "Alice"
        assert total_pages == 1

    @pytest.mark.asyncio
    async def test_get_customers_with_filters(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {"customers": []},
                "page": {"totalPages": 0},
            }
        )
        await helpscout_connector.get_customers(
            email="test@example.com",
            first_name="John",
            last_name="Doe",
        )
        call_args = mock_httpx_client.request.call_args
        params = call_args.kwargs["params"]
        assert params["email"] == "test@example.com"
        assert params["firstName"] == "John"
        assert params["lastName"] == "Doe"

    @pytest.mark.asyncio
    async def test_get_customer(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": 12345,
                "firstName": "Alice",
                "lastName": "Johnson",
                "emails": [{"value": "alice@example.com"}],
            }
        )
        customer = await helpscout_connector.get_customer(12345)
        assert customer.id == 12345
        assert customer.first_name == "Alice"

    @pytest.mark.asyncio
    async def test_create_customer(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {"Resource-ID": "99999"}
        mock_httpx_client.request.return_value = resp

        customer_id = await helpscout_connector.create_customer(
            email="new@example.com",
            first_name="New",
            last_name="Customer",
            phone="+1-555-0100",
            organization="Acme Corp",
            job_title="Engineer",
        )
        assert customer_id == 99999

    @pytest.mark.asyncio
    async def test_create_customer_minimal(self, helpscout_connector, mock_httpx_client):
        resp = MagicMock()
        resp.status_code = 201
        resp.headers = {"Resource-ID": "88888"}
        mock_httpx_client.request.return_value = resp

        customer_id = await helpscout_connector.create_customer(
            email="minimal@example.com",
            first_name="Minimal",
            last_name="User",
        )
        assert customer_id == 88888


# =============================================================================
# Mailbox Operations Tests
# =============================================================================


class TestMailboxOperations:
    """Tests for mailbox operations."""

    @pytest.mark.asyncio
    async def test_get_mailboxes(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {
                    "mailboxes": [
                        {
                            "id": 100,
                            "name": "Support",
                            "slug": "support",
                            "email": "support@company.com",
                        },
                        {"id": 101, "name": "Sales", "slug": "sales", "email": "sales@company.com"},
                    ],
                },
            }
        )
        mailboxes = await helpscout_connector.get_mailboxes()
        assert len(mailboxes) == 2
        assert mailboxes[0].name == "Support"
        assert mailboxes[1].name == "Sales"

    @pytest.mark.asyncio
    async def test_get_mailbox(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": 100,
                "name": "Support",
                "slug": "support",
                "email": "support@company.com",
            }
        )
        mailbox = await helpscout_connector.get_mailbox(100)
        assert mailbox.id == 100
        assert mailbox.name == "Support"

    @pytest.mark.asyncio
    async def test_get_mailbox_folders(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {
                    "folders": [
                        {
                            "id": 1,
                            "name": "Inbox",
                            "type": "inbox",
                            "totalCount": 50,
                            "activeCount": 25,
                        },
                        {
                            "id": 2,
                            "name": "Mine",
                            "type": "mine",
                            "totalCount": 10,
                            "activeCount": 5,
                        },
                    ],
                },
            }
        )
        folders = await helpscout_connector.get_mailbox_folders(100)
        assert len(folders) == 2
        assert folders[0].name == "Inbox"
        assert folders[0].total_count == 50


# =============================================================================
# User Operations Tests
# =============================================================================


class TestUserOperations:
    """Tests for user operations."""

    @pytest.mark.asyncio
    async def test_get_users(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {
                    "users": [
                        {
                            "id": 1,
                            "firstName": "Jane",
                            "lastName": "Smith",
                            "email": "jane@company.com",
                            "role": "admin",
                        },
                        {
                            "id": 2,
                            "firstName": "John",
                            "lastName": "Doe",
                            "email": "john@company.com",
                            "role": "user",
                        },
                    ],
                },
                "page": {"totalPages": 1},
            }
        )
        users, total_pages = await helpscout_connector.get_users()
        assert len(users) == 2
        assert users[0].role == "admin"
        assert users[1].role == "user"

    @pytest.mark.asyncio
    async def test_get_user(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": 67890,
                "firstName": "Jane",
                "lastName": "Smith",
                "email": "jane@company.com",
                "role": "admin",
            }
        )
        user = await helpscout_connector.get_user(67890)
        assert user.id == 67890
        assert user.first_name == "Jane"

    @pytest.mark.asyncio
    async def test_get_current_user(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "id": 12345,
                "firstName": "Current",
                "lastName": "User",
                "email": "me@company.com",
                "role": "admin",
            }
        )
        user = await helpscout_connector.get_current_user()
        assert user.id == 12345
        assert user.email == "me@company.com"


# =============================================================================
# Search Operations Tests
# =============================================================================


class TestSearchOperations:
    """Tests for search operations."""

    @pytest.mark.asyncio
    async def test_search_conversations(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {
                    "conversations": [
                        {
                            "id": 1,
                            "number": 101,
                            "subject": "Login issue",
                            "status": "active",
                            "type": "email",
                            "mailboxId": 1,
                        },
                    ],
                },
                "page": {"totalPages": 1},
            }
        )
        conversations, total_pages = await helpscout_connector.search_conversations(
            "status:active AND tag:urgent"
        )
        assert len(conversations) == 1
        assert conversations[0].subject == "Login issue"

    @pytest.mark.asyncio
    async def test_search_conversations_pagination(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {"conversations": []},
                "page": {"totalPages": 5},
            }
        )
        _, total_pages = await helpscout_connector.search_conversations("status:active", page=3)
        assert total_pages == 5
        call_args = mock_httpx_client.request.call_args
        assert call_args.kwargs["params"]["page"] == 3


# =============================================================================
# Mock Data Generation Tests
# =============================================================================


class TestMockDataGenerators:
    """Tests for mock data helpers."""

    def test_get_mock_conversation(self):
        conv = get_mock_conversation()
        assert conv.id == 12345
        assert conv.number == 1001
        assert conv.subject == "Order not received"
        assert conv.status == ConversationStatus.ACTIVE
        assert conv.type == ConversationType.EMAIL
        assert conv.mailbox_id == 100
        assert conv.customer_email == "customer@example.com"
        assert conv.created_at is not None

    def test_get_mock_customer(self):
        customer = get_mock_customer()
        assert customer.id == 67890
        assert customer.first_name == "Alice"
        assert customer.last_name == "Johnson"
        assert "alice.johnson@example.com" in customer.emails


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_conversation_from_api_empty_tags(self):
        data = {
            "id": 1,
            "number": 1,
            "subject": "",
            "status": "active",
            "type": "email",
            "mailboxId": 1,
            "tags": [],
        }
        conv = Conversation.from_api(data)
        assert conv.tags == []

    def test_customer_from_api_no_emails_key(self):
        data = {"id": 1, "firstName": "Test", "lastName": "User"}
        customer = HelpScoutCustomer.from_api(data)
        assert customer.emails == []

    def test_thread_from_api_no_embedded(self):
        data = {
            "id": 1,
            "type": "message",
            "body": "Test",
            "status": "active",
            "state": "published",
        }
        thread = Thread.from_api(data)
        assert thread.attachments == []

    def test_conversation_defaults(self):
        conv = Conversation(
            id=1,
            number=1,
            subject="Test",
            status=ConversationStatus.ACTIVE,
            type=ConversationType.EMAIL,
            mailbox_id=1,
        )
        assert conv.assignee_id is None
        assert conv.customer_id is None
        assert conv.tags == []
        assert conv.cc == []
        assert conv.threads == []

    def test_customer_defaults(self):
        customer = HelpScoutCustomer(
            id=1,
            first_name="Test",
            last_name="User",
        )
        assert customer.emails == []
        assert customer.phones == []
        assert customer.organization is None
        assert customer.job_title is None

    @pytest.mark.asyncio
    async def test_get_conversations_empty_response(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {"conversations": []},
                "page": {"totalPages": 0},
            }
        )
        conversations, total_pages = await helpscout_connector.get_conversations()
        assert conversations == []
        assert total_pages == 0

    @pytest.mark.asyncio
    async def test_get_customers_empty_response(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {"customers": []},
                "page": {"totalPages": 0},
            }
        )
        customers, total_pages = await helpscout_connector.get_customers()
        assert customers == []
        assert total_pages == 0

    @pytest.mark.asyncio
    async def test_get_mailboxes_empty_response(self, helpscout_connector, mock_httpx_client):
        mock_httpx_client.request.return_value = _make_response(
            {
                "_embedded": {"mailboxes": []},
            }
        )
        mailboxes = await helpscout_connector.get_mailboxes()
        assert mailboxes == []

    def test_conversation_from_api_no_assignee(self):
        data = {
            "id": 1,
            "number": 1,
            "subject": "Test",
            "status": "active",
            "type": "email",
            "mailboxId": 1,
            "assignee": None,
        }
        conv = Conversation.from_api(data)
        assert conv.assignee_id is None

    def test_thread_from_api_no_source(self):
        data = {
            "id": 1,
            "type": "message",
            "body": "Test",
            "status": "active",
            "state": "published",
            "source": {},
        }
        thread = Thread.from_api(data)
        assert thread.source_type is None
