"""
Tests for inbox command center handler.

Tests the InboxCommandHandler including:
- Prioritized inbox fetching
- Quick actions (archive, snooze, reply)
- Bulk operations
- Daily digest
- Sender profile lookup
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta


class MockRequest:
    """Mock aiohttp request."""

    def __init__(self, query: dict = None, body: dict = None, headers: dict = None):
        self._query = query or {}
        self._body = body or {}
        self._headers = headers or {}
        self._data = {}

    def get(self, key, default=None):
        """Support dict-like access for request data."""
        return self._data.get(key, default)

    @property
    def headers(self):
        """Return headers dict with get support."""
        return self._headers

    @property
    def query(self):
        return self._query

    async def json(self):
        return self._body


class MockResponse:
    """Capture response for assertions."""

    def __init__(self, data, status=200):
        self.data = data
        self.status = status


def mock_json_response(data, status=200):
    """Mock web.json_response."""
    return MockResponse(data, status)


class MockHTTPException(Exception):
    """Base mock HTTP exception."""

    def __init__(self, reason=None, text=None, **kwargs):
        self.reason = reason
        self.text = text
        super().__init__(reason or text or "HTTP Error")


class MockHTTPForbidden(MockHTTPException):
    """Mock 403 Forbidden."""

    pass


class MockHTTPUnauthorized(MockHTTPException):
    """Mock 401 Unauthorized."""

    pass


class MockPermissionDecision:
    """Mock permission decision that always allows."""

    def __init__(self):
        self.allowed = True
        self.reason = "Allowed by test mock"


class MockPermissionChecker:
    """Mock permission checker that always allows."""

    def check_permission(self, context, permission):
        return MockPermissionDecision()


@pytest.fixture
def mock_web():
    """Set up mock for aiohttp.web."""
    with patch("aragora.server.handlers.inbox_command.web") as web_mock:
        web_mock.json_response = mock_json_response
        web_mock.Request = MockRequest
        web_mock.Response = MockResponse
        web_mock.HTTPForbidden = MockHTTPForbidden
        web_mock.HTTPUnauthorized = MockHTTPUnauthorized
        yield web_mock


@pytest.fixture
def handler(mock_web):
    """Create handler instance for testing."""
    with patch(
        "aragora.server.handlers.inbox_command.get_permission_checker",
        return_value=MockPermissionChecker(),
    ):
        from aragora.server.handlers.inbox_command import InboxCommandHandler

        handler_instance = InboxCommandHandler()
        yield handler_instance


@pytest.fixture
def mock_request():
    """Create a mock request factory."""

    def _create(query=None, body=None):
        return MockRequest(query=query, body=body)

    return _create


class TestInboxCommandHandler:
    """Tests for InboxCommandHandler."""

    @pytest.mark.asyncio
    async def test_handle_get_inbox_returns_demo_data(self, handler, mock_request):
        """Test inbox fetch returns demo data when services unavailable."""
        request = mock_request(query={"limit": "10"})
        response = await handler.handle_get_inbox(request)

        assert response.data["success"] is True
        assert "emails" in response.data
        assert "stats" in response.data
        assert len(response.data["emails"]) > 0

    @pytest.mark.asyncio
    async def test_handle_get_inbox_with_priority_filter(self, handler, mock_request):
        """Test inbox fetch with priority filter."""
        request = mock_request(query={"priority": "critical"})
        response = await handler.handle_get_inbox(request)

        assert response.data["success"] is True
        # All returned emails should be critical priority
        for email in response.data["emails"]:
            assert email["priority"] == "critical"

    @pytest.mark.asyncio
    async def test_handle_quick_action_requires_action(self, handler, mock_request):
        """Test quick action fails without action parameter."""
        request = mock_request(body={"emailIds": ["email_1"]})
        response = await handler.handle_quick_action(request)

        assert response.data["success"] is False
        assert "action is required" in response.data["error"]

    @pytest.mark.asyncio
    async def test_handle_quick_action_requires_email_ids(self, handler, mock_request):
        """Test quick action fails without emailIds parameter."""
        request = mock_request(body={"action": "archive"})
        response = await handler.handle_quick_action(request)

        assert response.data["success"] is False
        assert "emailIds is required" in response.data["error"]

    @pytest.mark.asyncio
    async def test_handle_quick_action_archive(self, handler, mock_request):
        """Test archive action execution."""
        request = mock_request(body={"action": "archive", "emailIds": ["email_1", "email_2"]})
        response = await handler.handle_quick_action(request)

        assert response.data["success"] is True
        assert response.data["action"] == "archive"
        assert response.data["processed"] == 2
        assert len(response.data["results"]) == 2

    @pytest.mark.asyncio
    async def test_handle_quick_action_snooze_with_duration(self, handler, mock_request):
        """Test snooze action with duration parameter."""
        request = mock_request(
            body={"action": "snooze", "emailIds": ["email_1"], "params": {"duration": "1d"}}
        )
        response = await handler.handle_quick_action(request)

        assert response.data["success"] is True
        assert response.data["action"] == "snooze"
        # Verify snooze result
        result = response.data["results"][0]
        assert result["success"] is True
        assert "until" in result["result"]

    @pytest.mark.asyncio
    async def test_handle_bulk_action_requires_action_and_filter(self, handler, mock_request):
        """Test bulk action requires both action and filter."""
        request = mock_request(body={"action": "archive"})
        response = await handler.handle_bulk_action(request)

        assert response.data["success"] is False
        assert "action and filter are required" in response.data["error"]

    @pytest.mark.asyncio
    async def test_handle_bulk_action_low_priority(self, handler, mock_request):
        """Test bulk archive of low priority emails."""
        # First populate cache with emails
        await handler.handle_get_inbox(mock_request(query={}))

        request = mock_request(body={"action": "archive", "filter": "low"})
        response = await handler.handle_bulk_action(request)

        assert response.data["success"] is True
        assert response.data["action"] == "archive"
        assert response.data["filter"] == "low"

    @pytest.mark.asyncio
    async def test_handle_get_sender_profile_requires_email(self, handler, mock_request):
        """Test sender profile requires email parameter."""
        request = mock_request(query={})
        response = await handler.handle_get_sender_profile(request)

        assert response.data["success"] is False
        assert "email parameter is required" in response.data["error"]

    @pytest.mark.asyncio
    async def test_handle_get_sender_profile_returns_basic_profile(self, handler, mock_request):
        """Test sender profile returns basic data."""
        request = mock_request(query={"email": "test@example.com"})
        response = await handler.handle_get_sender_profile(request)

        assert response.data["success"] is True
        profile = response.data["profile"]
        assert profile["email"] == "test@example.com"
        assert "name" in profile
        assert "isVip" in profile

    @pytest.mark.asyncio
    async def test_handle_get_daily_digest(self, handler, mock_request):
        """Test daily digest returns statistics."""
        # Populate cache first
        await handler.handle_get_inbox(mock_request(query={}))

        request = mock_request()
        response = await handler.handle_get_daily_digest(request)

        assert response.data["success"] is True
        digest = response.data["digest"]
        assert "emailsReceived" in digest
        assert "topSenders" in digest
        assert "categoryBreakdown" in digest

    @pytest.mark.asyncio
    async def test_handle_reprioritize_without_prioritizer(self, handler, mock_request):
        """Test reprioritize returns error without prioritizer."""
        request = mock_request(body={})
        response = await handler.handle_reprioritize(request)

        assert response.data["success"] is True
        # Should indicate prioritizer not available
        assert response.data["reprioritized"] == 0

    @pytest.mark.asyncio
    async def test_inbox_stats_calculation(self, handler, mock_request):
        """Test inbox stats are calculated correctly."""
        request = mock_request(query={})
        response = await handler.handle_get_inbox(request)

        stats = response.data["stats"]
        assert "total" in stats
        assert "unread" in stats
        assert "critical" in stats
        assert "high" in stats
        assert "medium" in stats
        assert "low" in stats
        assert "actionRequired" in stats
        # actionRequired should be critical + high
        assert stats["actionRequired"] == stats["critical"] + stats["high"]


class TestInboxActionsDemo:
    """Test individual action handlers in demo mode."""

    @pytest.fixture
    def handler(self, mock_web):
        from aragora.server.handlers.inbox_command import InboxCommandHandler

        return InboxCommandHandler()

    @pytest.mark.asyncio
    async def test_archive_email_demo(self, handler):
        """Test archive returns demo result."""
        result = await handler._archive_email("email_1", {})
        assert result["archived"] is True
        assert result.get("demo") is True

    @pytest.mark.asyncio
    async def test_snooze_email_demo(self, handler):
        """Test snooze returns demo result with duration."""
        result = await handler._snooze_email("email_1", {"duration": "3h"})
        assert result["snoozed"] is True
        assert "until" in result

    @pytest.mark.asyncio
    async def test_create_reply_draft_demo(self, handler):
        """Test reply draft creation."""
        result = await handler._create_reply_draft("email_1", {"body": "Test reply"})
        assert "draftId" in result

    @pytest.mark.asyncio
    async def test_mark_spam_demo(self, handler):
        """Test mark spam."""
        result = await handler._mark_spam("email_1", {})
        assert result["spam"] is True

    @pytest.mark.asyncio
    async def test_mark_important_demo(self, handler):
        """Test mark important."""
        result = await handler._mark_important("email_1", {})
        assert result["important"] is True

    @pytest.mark.asyncio
    async def test_delete_email_demo(self, handler):
        """Test delete email."""
        result = await handler._delete_email("email_1", {})
        assert result["deleted"] is True


class TestInboxDemoData:
    """Test demo data generation."""

    @pytest.fixture
    def handler(self, mock_web):
        from aragora.server.handlers.inbox_command import InboxCommandHandler

        return InboxCommandHandler()

    def test_demo_emails_have_required_fields(self, handler):
        """Test demo emails have all required fields."""
        emails = handler._get_demo_emails(limit=10, offset=0, priority_filter=None)

        required_fields = [
            "id",
            "from",
            "subject",
            "snippet",
            "priority",
            "confidence",
            "reasoning",
            "tier_used",
            "timestamp",
        ]

        for email in emails:
            for field in required_fields:
                assert field in email, f"Missing field: {field}"

    def test_demo_emails_pagination(self, handler):
        """Test demo emails respect pagination."""
        all_emails = handler._get_demo_emails(limit=100, offset=0, priority_filter=None)
        offset_emails = handler._get_demo_emails(limit=2, offset=2, priority_filter=None)

        assert len(offset_emails) <= 2
        if len(all_emails) > 2:
            assert offset_emails[0]["id"] == all_emails[2]["id"]

    def test_demo_emails_priority_filter(self, handler):
        """Test demo emails filter by priority."""
        filtered = handler._get_demo_emails(limit=10, offset=0, priority_filter="high")

        for email in filtered:
            assert email["priority"] == "high"
