"""Tests for Unified Inbox namespace API.

Focus: the legacy ``nv_*`` aliases must route to the canonical versioned
``/api/v1/inbox/*`` endpoints. They previously issued requests against
non-versioned ``/inbox/*`` paths that no server handler routes (the client
does not prefix paths), so every call was a guaranteed 404.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestUnifiedInboxNvAliases:
    """Sync nv_* aliases delegate to canonical versioned endpoints."""

    def test_nv_get_gmail_oauth_url_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"auth_url": "https://accounts.google.com/o/oauth2"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = client.unified_inbox.nv_get_gmail_oauth_url(
                "https://app.example.com/callback", state="csrf-123"
            )

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/oauth/gmail",
                params={"redirect_uri": "https://app.example.com/callback", "state": "csrf-123"},
            )
            assert "auth_url" in result
            client.close()

    def test_nv_get_outlook_oauth_url_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"auth_url": "https://login.microsoftonline.com"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.unified_inbox.nv_get_outlook_oauth_url("https://app.example.com/callback")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/oauth/outlook",
                params={"redirect_uri": "https://app.example.com/callback"},
            )
            client.close()

    def test_nv_connect_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"account_id": "acct-1", "status": "connected"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = client.unified_inbox.nv_connect(
                "gmail", "auth-code-xyz", "https://app.example.com/callback"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/connect",
                json={
                    "provider": "gmail",
                    "auth_code": "auth-code-xyz",
                    "redirect_uri": "https://app.example.com/callback",
                },
            )
            assert result["status"] == "connected"
            client.close()

    def test_nv_list_accounts_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"accounts": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.unified_inbox.nv_list_accounts()

            mock_request.assert_called_once_with("GET", "/api/v1/inbox/accounts")
            client.close()

    def test_nv_list_messages_uses_versioned_route_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"messages": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.unified_inbox.nv_list_messages(
                limit=10, priority="critical", unread_only=True, search="invoice"
            )

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/messages",
                params={
                    "limit": 10,
                    "offset": 0,
                    "priority": "critical",
                    "unread_only": True,
                    "search": "invoice",
                },
            )
            client.close()

    def test_nv_send_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"message_id": "msg-1", "status": "sent"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            result = client.unified_inbox.nv_send(
                "email", "user@example.com", "Hello", subject="Greetings"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/messages/send",
                json={
                    "channel": "email",
                    "to": "user@example.com",
                    "content": "Hello",
                    "subject": "Greetings",
                },
            )
            assert result["status"] == "sent"
            client.close()

    def test_nv_triage_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"results": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.unified_inbox.nv_triage(["m1", "m2"], context={"urgency_keywords": ["asap"]})

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/triage",
                json={
                    "message_ids": ["m1", "m2"],
                    "context": {"urgency_keywords": ["asap"]},
                },
            )
            client.close()

    def test_nv_bulk_action_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success_count": 2, "error_count": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.unified_inbox.nv_bulk_action(["m1", "m2"], "archive")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/bulk-action",
                json={"message_ids": ["m1", "m2"], "action": "archive"},
            )
            client.close()

    def test_nv_get_stats_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"unread": 3}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.unified_inbox.nv_get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/inbox/stats")
            client.close()

    def test_nv_get_trends_uses_versioned_route(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"trends": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

            client.unified_inbox.nv_get_trends(days=14)

            mock_request.assert_called_once_with("GET", "/api/v1/inbox/trends", params={"days": 14})
            client.close()

    def test_no_non_versioned_paths_remain(self) -> None:
        """Regression guard: no SDK call may target a bare /inbox/* path."""
        import inspect

        from aragora_sdk.namespaces import unified_inbox

        source = inspect.getsource(unified_inbox)
        assert '"/inbox/' not in source, (
            "unified_inbox must not request non-versioned /inbox/* paths; "
            "the server only routes /api/v1/inbox/*"
        )


class TestAsyncUnifiedInboxNvAliases:
    """Async nv_* aliases delegate to canonical versioned endpoints."""

    @pytest.mark.asyncio
    async def test_nv_get_gmail_oauth_url_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"auth_url": "https://accounts.google.com/o/oauth2"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_get_gmail_oauth_url("https://app.example.com/callback")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/oauth/gmail",
                params={"redirect_uri": "https://app.example.com/callback"},
            )

    @pytest.mark.asyncio
    async def test_nv_get_outlook_oauth_url_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"auth_url": "https://login.microsoftonline.com"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_get_outlook_oauth_url(
                "https://app.example.com/callback", state="csrf-456"
            )

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/oauth/outlook",
                params={"redirect_uri": "https://app.example.com/callback", "state": "csrf-456"},
            )

    @pytest.mark.asyncio
    async def test_nv_connect_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"account_id": "acct-1"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_connect(
                "outlook", "auth-code-abc", "https://app.example.com/callback"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/connect",
                json={
                    "provider": "outlook",
                    "auth_code": "auth-code-abc",
                    "redirect_uri": "https://app.example.com/callback",
                },
            )

    @pytest.mark.asyncio
    async def test_nv_list_accounts_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"accounts": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_list_accounts()

            mock_request.assert_called_once_with("GET", "/api/v1/inbox/accounts")

    @pytest.mark.asyncio
    async def test_nv_list_messages_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"messages": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_list_messages(limit=5, account_id="acct-1")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/messages",
                params={"limit": 5, "offset": 0, "account_id": "acct-1"},
            )

    @pytest.mark.asyncio
    async def test_nv_send_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"message_id": "msg-2"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_send("email", "user@example.com", "Hi")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/messages/send",
                json={"channel": "email", "to": "user@example.com", "content": "Hi"},
            )

    @pytest.mark.asyncio
    async def test_nv_triage_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"results": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_triage(["m1"])

            mock_request.assert_called_once_with(
                "POST", "/api/v1/inbox/triage", json={"message_ids": ["m1"]}
            )

    @pytest.mark.asyncio
    async def test_nv_bulk_action_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"success_count": 1, "error_count": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_bulk_action(["m1"], "mark_read")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/bulk-action",
                json={"message_ids": ["m1"], "action": "mark_read"},
            )

    @pytest.mark.asyncio
    async def test_nv_get_stats_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"unread": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_get_stats()

            mock_request.assert_called_once_with("GET", "/api/v1/inbox/stats")

    @pytest.mark.asyncio
    async def test_nv_get_trends_uses_versioned_route(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"trends": []}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")

            await client.unified_inbox.nv_get_trends()

            mock_request.assert_called_once_with("GET", "/api/v1/inbox/trends", params={"days": 7})
