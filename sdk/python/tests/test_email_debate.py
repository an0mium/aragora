"""Tests for Email Debate SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestEmailDebateAPI:
    """Test synchronous EmailDebateAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        api = EmailDebateAPI(mock_client)
        assert api._client is mock_client

    def test_prioritize(self, mock_client: MagicMock) -> None:
        """Test prioritize calls correct endpoint."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {
            "priority": "high",
            "score": 0.85,
            "confidence": 0.9,
            "reasoning": "Urgent sender",
        }

        api = EmailDebateAPI(mock_client)
        result = api.prioritize(
            email={
                "subject": "Q4 Budget Review",
                "body": "Please review the attached...",
                "sender": "cfo@company.com",
            }
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/email/prioritize")
        assert call_args[1]["json"]["subject"] == "Q4 Budget Review"
        assert call_args[1]["json"]["sender"] == "cfo@company.com"
        assert result["priority"] == "high"

    def test_prioritize_with_user_id(self, mock_client: MagicMock) -> None:
        """Test prioritize with user_id parameter."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {"priority": "medium", "score": 0.5}

        api = EmailDebateAPI(mock_client)
        api.prioritize(
            email={"sender": "test@example.com"},
            user_id="user_123",
        )

        call_args = mock_client.request.call_args
        assert call_args[1]["json"]["user_id"] == "user_123"
        assert call_args[1]["json"]["sender"] == "test@example.com"

    def test_prioritize_batch(self, mock_client: MagicMock) -> None:
        """Test prioritize_batch calls correct endpoint."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {
            "results": [
                {"priority": "high", "score": 0.9},
                {"priority": "low", "score": 0.2},
            ],
            "total_processed": 2,
            "processing_time_ms": 150,
        }

        api = EmailDebateAPI(mock_client)
        result = api.prioritize_batch(
            emails=[
                {"sender": "urgent@company.com", "subject": "URGENT"},
                {"sender": "newsletter@spam.com", "subject": "Weekly Deals"},
            ]
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/email/prioritize/batch")
        assert len(call_args[1]["json"]["emails"]) == 2
        assert call_args[1]["json"]["parallel"] is True
        assert result["total_processed"] == 2

    def test_prioritize_batch_with_options(self, mock_client: MagicMock) -> None:
        """Test prioritize_batch with all options."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {"results": [], "total_processed": 0}

        api = EmailDebateAPI(mock_client)
        api.prioritize_batch(
            emails=[{"sender": "test@example.com"}],
            user_id="user_456",
            parallel=False,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["user_id"] == "user_456"
        assert json_body["parallel"] is False

    def test_triage_inbox(self, mock_client: MagicMock) -> None:
        """Test triage_inbox calls correct endpoint."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {
            "results": [
                {
                    "category": "action_required",
                    "priority": "high",
                    "suggested_folder": "Inbox",
                }
            ],
            "total_triaged": 1,
            "processing_time_ms": 200,
        }

        api = EmailDebateAPI(mock_client)
        result = api.triage_inbox(
            emails=[{"sender": "boss@company.com", "subject": "Review needed"}]
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/email/triage")
        assert call_args[1]["json"]["include_auto_replies"] is False
        assert result["total_triaged"] == 1

    def test_triage_inbox_with_auto_replies(self, mock_client: MagicMock) -> None:
        """Test triage_inbox with auto-reply generation enabled."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {
            "results": [{"auto_reply_draft": "Thank you for your email..."}],
            "total_triaged": 1,
        }

        api = EmailDebateAPI(mock_client)
        api.triage_inbox(
            emails=[{"sender": "test@example.com"}],
            user_id="user_789",
            include_auto_replies=True,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["include_auto_replies"] is True
        assert json_body["user_id"] == "user_789"

    def test_get_history(self, mock_client: MagicMock) -> None:
        """Test get_history calls correct endpoint."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {
            "history": [
                {"email_id": "msg_1", "priority": "high", "prioritized_at": "2024-01-01T00:00:00Z"}
            ],
            "total": 1,
        }

        api = EmailDebateAPI(mock_client)
        result = api.get_history(user_id="user_123")

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/email/prioritize/history", params={"user_id": "user_123"}
        )
        assert result["total"] == 1

    def test_get_history_with_options(self, mock_client: MagicMock) -> None:
        """Test get_history with limit and since parameters."""
        from aragora.namespaces.email_debate import EmailDebateAPI

        mock_client.request.return_value = {"history": [], "total": 0}

        api = EmailDebateAPI(mock_client)
        api.get_history(
            user_id="user_123",
            limit=25,
            since="2024-01-01T00:00:00Z",
        )

        mock_client.request.assert_called_once_with(
            "GET",
            "/api/v1/email/prioritize/history",
            params={
                "user_id": "user_123",
                "limit": 25,
                "since": "2024-01-01T00:00:00Z",
            },
        )


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncEmailDebateAPI:
    """Test asynchronous AsyncEmailDebateAPI."""

    @pytest.mark.asyncio
    async def test_init(self, mock_async_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.email_debate import AsyncEmailDebateAPI

        api = AsyncEmailDebateAPI(mock_async_client)
        assert api._client is mock_async_client

    @pytest.mark.asyncio
    async def test_prioritize(self, mock_async_client: MagicMock) -> None:
        """Test prioritize calls correct endpoint."""
        from aragora.namespaces.email_debate import AsyncEmailDebateAPI

        mock_async_client.request.return_value = {
            "priority": "critical",
            "score": 0.95,
        }

        api = AsyncEmailDebateAPI(mock_async_client)
        result = await api.prioritize(
            email={"sender": "alerts@monitoring.com", "subject": "Server Down"}
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/email/prioritize")
        assert result["priority"] == "critical"

    @pytest.mark.asyncio
    async def test_prioritize_with_user_id(self, mock_async_client: MagicMock) -> None:
        """Test prioritize with user_id parameter."""
        from aragora.namespaces.email_debate import AsyncEmailDebateAPI

        mock_async_client.request.return_value = {"priority": "low"}

        api = AsyncEmailDebateAPI(mock_async_client)
        await api.prioritize(
            email={"sender": "spam@example.com"},
            user_id="user_async_123",
        )

        call_args = mock_async_client.request.call_args
        assert call_args[1]["json"]["user_id"] == "user_async_123"

    @pytest.mark.asyncio
    async def test_prioritize_batch(self, mock_async_client: MagicMock) -> None:
        """Test prioritize_batch calls correct endpoint."""
        from aragora.namespaces.email_debate import AsyncEmailDebateAPI

        mock_async_client.request.return_value = {
            "results": [],
            "total_processed": 0,
        }

        api = AsyncEmailDebateAPI(mock_async_client)
        await api.prioritize_batch(
            emails=[{"sender": "test@example.com"}],
            user_id="user_batch",
            parallel=False,
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/email/prioritize/batch")
        json_body = call_args[1]["json"]
        assert json_body["parallel"] is False
        assert json_body["user_id"] == "user_batch"

    @pytest.mark.asyncio
    async def test_triage_inbox(self, mock_async_client: MagicMock) -> None:
        """Test triage_inbox calls correct endpoint."""
        from aragora.namespaces.email_debate import AsyncEmailDebateAPI

        mock_async_client.request.return_value = {
            "results": [{"category": "fyi"}],
            "total_triaged": 1,
        }

        api = AsyncEmailDebateAPI(mock_async_client)
        result = await api.triage_inbox(
            emails=[{"sender": "newsletter@company.com"}],
            include_auto_replies=True,
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/email/triage")
        assert call_args[1]["json"]["include_auto_replies"] is True
        assert result["total_triaged"] == 1

    @pytest.mark.asyncio
    async def test_get_history(self, mock_async_client: MagicMock) -> None:
        """Test get_history calls correct endpoint."""
        from aragora.namespaces.email_debate import AsyncEmailDebateAPI

        mock_async_client.request.return_value = {"history": [], "total": 0}

        api = AsyncEmailDebateAPI(mock_async_client)
        await api.get_history(
            user_id="user_history",
            limit=10,
            since="2024-06-01T00:00:00Z",
        )

        mock_async_client.request.assert_called_once_with(
            "GET",
            "/api/v1/email/prioritize/history",
            params={
                "user_id": "user_history",
                "limit": 10,
                "since": "2024-06-01T00:00:00Z",
            },
        )
