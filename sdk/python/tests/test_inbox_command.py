"""Tests for Inbox Command Center namespace API.

Tests cover:
- get_inbox() - fetch prioritized inbox with pagination and filters
- quick_action() - execute quick actions on emails
- bulk_action() - execute bulk actions based on filters
- get_sender_profile() - get sender profile information
- get_daily_digest() - get daily digest statistics
- reprioritize() - trigger AI re-prioritization
"""

from __future__ import annotations

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestGetInbox:
    """Tests for get_inbox method."""

    def test_get_inbox_default(self, client: AragoraClient, mock_request) -> None:
        """Get inbox with default parameters."""
        mock_request.return_value = {
            "emails": [
                {"id": "email_1", "subject": "Important", "priority": "critical"},
                {"id": "email_2", "subject": "Newsletter", "priority": "low"},
            ],
            "total": 42,
            "stats": {"unread": 15, "critical": 3},
        }

        result = client.inbox_command.get_inbox()

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/inbox/command",
            params={"limit": 50, "offset": 0},
        )
        assert len(result["emails"]) == 2
        assert result["total"] == 42
        assert result["stats"]["critical"] == 3

    def test_get_inbox_with_pagination(self, client: AragoraClient, mock_request) -> None:
        """Get inbox with custom pagination."""
        mock_request.return_value = {"emails": [], "total": 100}

        client.inbox_command.get_inbox(limit=25, offset=50)

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/inbox/command",
            params={"limit": 25, "offset": 50},
        )

    def test_get_inbox_filter_by_priority(self, client: AragoraClient, mock_request) -> None:
        """Get inbox filtered by priority level."""
        mock_request.return_value = {
            "emails": [{"id": "email_1", "priority": "critical"}],
            "total": 3,
        }

        result = client.inbox_command.get_inbox(priority="critical")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/inbox/command",
            params={"limit": 50, "offset": 0, "priority": "critical"},
        )
        assert result["emails"][0]["priority"] == "critical"

    def test_get_inbox_unread_only(self, client: AragoraClient, mock_request) -> None:
        """Get inbox with unread_only filter."""
        mock_request.return_value = {"emails": [], "total": 15}

        client.inbox_command.get_inbox(unread_only=True)

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/inbox/command",
            params={"limit": 50, "offset": 0, "unread_only": "true"},
        )

    def test_get_inbox_with_all_filters(self, client: AragoraClient, mock_request) -> None:
        """Get inbox with all optional parameters."""
        mock_request.return_value = {"emails": [], "total": 5}

        client.inbox_command.get_inbox(
            limit=10,
            offset=20,
            priority="high",
            unread_only=True,
        )

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/inbox/command",
            params={
                "limit": 10,
                "offset": 20,
                "priority": "high",
                "unread_only": "true",
            },
        )


class TestQuickAction:
    """Tests for quick_action method."""

    def test_quick_action_archive(self, client: AragoraClient, mock_request) -> None:
        """Execute archive action on emails."""
        mock_request.return_value = {
            "action": "archive",
            "processed": 2,
            "results": [
                {"id": "email_1", "status": "archived"},
                {"id": "email_2", "status": "archived"},
            ],
        }

        result = client.inbox_command.quick_action(
            action="archive",
            email_ids=["email_1", "email_2"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/actions",
            json={"action": "archive", "emailIds": ["email_1", "email_2"]},
        )
        assert result["action"] == "archive"
        assert result["processed"] == 2

    def test_quick_action_snooze_with_params(self, client: AragoraClient, mock_request) -> None:
        """Execute snooze action with duration parameter."""
        mock_request.return_value = {
            "action": "snooze",
            "processed": 1,
            "results": [{"id": "email_1", "snoozed_until": "2025-01-15T09:00:00Z"}],
        }

        result = client.inbox_command.quick_action(
            action="snooze",
            email_ids=["email_1"],
            params={"duration": "1d", "remind_at": "2025-01-15T09:00:00Z"},
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/actions",
            json={
                "action": "snooze",
                "emailIds": ["email_1"],
                "params": {"duration": "1d", "remind_at": "2025-01-15T09:00:00Z"},
            },
        )
        assert result["action"] == "snooze"

    def test_quick_action_reply_with_params(self, client: AragoraClient, mock_request) -> None:
        """Execute reply action with message content."""
        mock_request.return_value = {"action": "reply", "processed": 1}

        client.inbox_command.quick_action(
            action="reply",
            email_ids=["email_1"],
            params={"body": "Thank you for your message.", "send_immediately": True},
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/actions",
            json={
                "action": "reply",
                "emailIds": ["email_1"],
                "params": {"body": "Thank you for your message.", "send_immediately": True},
            },
        )

    def test_quick_action_forward(self, client: AragoraClient, mock_request) -> None:
        """Execute forward action."""
        mock_request.return_value = {"action": "forward", "processed": 1}

        client.inbox_command.quick_action(
            action="forward",
            email_ids=["email_1"],
            params={"to": "colleague@example.com", "note": "FYI"},
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/actions",
            json={
                "action": "forward",
                "emailIds": ["email_1"],
                "params": {"to": "colleague@example.com", "note": "FYI"},
            },
        )

    def test_quick_action_mark_important(self, client: AragoraClient, mock_request) -> None:
        """Execute mark_important action."""
        mock_request.return_value = {"action": "mark_important", "processed": 3}

        result = client.inbox_command.quick_action(
            action="mark_important",
            email_ids=["email_1", "email_2", "email_3"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/actions",
            json={
                "action": "mark_important",
                "emailIds": ["email_1", "email_2", "email_3"],
            },
        )
        assert result["processed"] == 3

    def test_quick_action_spam(self, client: AragoraClient, mock_request) -> None:
        """Execute spam action."""
        mock_request.return_value = {"action": "spam", "processed": 1}

        client.inbox_command.quick_action(
            action="spam",
            email_ids=["email_spam"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/actions",
            json={"action": "spam", "emailIds": ["email_spam"]},
        )

    def test_quick_action_delete(self, client: AragoraClient, mock_request) -> None:
        """Execute delete action."""
        mock_request.return_value = {"action": "delete", "processed": 1}

        client.inbox_command.quick_action(
            action="delete",
            email_ids=["email_1"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/actions",
            json={"action": "delete", "emailIds": ["email_1"]},
        )


class TestBulkAction:
    """Tests for bulk_action method."""

    def test_bulk_action_archive_low(self, client: AragoraClient, mock_request) -> None:
        """Bulk archive low-priority emails."""
        mock_request.return_value = {
            "action": "archive",
            "filter": "low",
            "processed": 25,
            "results": {"archived": 25},
        }

        result = client.inbox_command.bulk_action(
            action="archive",
            filter="low",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/bulk-actions",
            json={"action": "archive", "filter": "low"},
        )
        assert result["processed"] == 25

    def test_bulk_action_delete_spam(self, client: AragoraClient, mock_request) -> None:
        """Bulk delete spam emails."""
        mock_request.return_value = {
            "action": "delete",
            "filter": "spam",
            "processed": 100,
        }

        result = client.inbox_command.bulk_action(
            action="delete",
            filter="spam",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/bulk-actions",
            json={"action": "delete", "filter": "spam"},
        )
        assert result["filter"] == "spam"

    def test_bulk_action_with_params(self, client: AragoraClient, mock_request) -> None:
        """Bulk action with additional parameters."""
        mock_request.return_value = {
            "action": "snooze",
            "filter": "deferred",
            "processed": 10,
        }

        client.inbox_command.bulk_action(
            action="snooze",
            filter="deferred",
            params={"duration": "1w", "move_to_folder": "later"},
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/bulk-actions",
            json={
                "action": "snooze",
                "filter": "deferred",
                "params": {"duration": "1w", "move_to_folder": "later"},
            },
        )

    def test_bulk_action_archive_read(self, client: AragoraClient, mock_request) -> None:
        """Bulk archive read emails."""
        mock_request.return_value = {"action": "archive", "filter": "read", "processed": 50}

        client.inbox_command.bulk_action(action="archive", filter="read")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/bulk-actions",
            json={"action": "archive", "filter": "read"},
        )

    def test_bulk_action_all(self, client: AragoraClient, mock_request) -> None:
        """Bulk action on all emails."""
        mock_request.return_value = {
            "action": "mark_important",
            "filter": "all",
            "processed": 200,
        }

        result = client.inbox_command.bulk_action(
            action="mark_important",
            filter="all",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/bulk-actions",
            json={"action": "mark_important", "filter": "all"},
        )
        assert result["processed"] == 200


class TestGetSenderProfile:
    """Tests for get_sender_profile method."""

    def test_get_sender_profile(self, client: AragoraClient, mock_request) -> None:
        """Get sender profile for an email address."""
        mock_request.return_value = {
            "email": "boss@company.com",
            "name": "Jane Smith",
            "isVip": True,
            "responseRate": 0.95,
            "avgResponseTime": "2h",
            "emailCount": 150,
            "lastContact": "2025-01-10T14:30:00Z",
        }

        result = client.inbox_command.get_sender_profile("boss@company.com")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/inbox/sender-profile",
            params={"email": "boss@company.com"},
        )
        assert result["email"] == "boss@company.com"
        assert result["isVip"] is True
        assert result["responseRate"] == 0.95

    def test_get_sender_profile_unknown(self, client: AragoraClient, mock_request) -> None:
        """Get sender profile for unknown sender."""
        mock_request.return_value = {
            "email": "unknown@example.com",
            "name": None,
            "isVip": False,
            "responseRate": None,
            "emailCount": 1,
        }

        result = client.inbox_command.get_sender_profile("unknown@example.com")

        mock_request.assert_called_once_with(
            "GET",
            "/api/v1/inbox/sender-profile",
            params={"email": "unknown@example.com"},
        )
        assert result["isVip"] is False
        assert result["emailCount"] == 1


class TestGetDailyDigest:
    """Tests for get_daily_digest method."""

    def test_get_daily_digest(self, client: AragoraClient, mock_request) -> None:
        """Get daily digest statistics."""
        mock_request.return_value = {
            "emailsReceived": 85,
            "processed": 80,
            "criticalHandled": 5,
            "timeSaved": "3h 45m",
            "autoArchived": 30,
            "autoReplied": 10,
            "flaggedForReview": 8,
            "topSenders": [
                {"email": "team@company.com", "count": 15},
                {"email": "alerts@service.com", "count": 12},
            ],
        }

        result = client.inbox_command.get_daily_digest()

        mock_request.assert_called_once_with("GET", "/api/v1/inbox/daily-digest")
        assert result["emailsReceived"] == 85
        assert result["processed"] == 80
        assert result["timeSaved"] == "3h 45m"
        assert len(result["topSenders"]) == 2


class TestReprioritize:
    """Tests for reprioritize method."""

    def test_reprioritize_default(self, client: AragoraClient, mock_request) -> None:
        """Trigger reprioritization with default parameters."""
        mock_request.return_value = {
            "reprioritized": 42,
            "changes": [
                {"id": "email_1", "old": "low", "new": "high"},
                {"id": "email_2", "old": "medium", "new": "critical"},
            ],
            "tier_used": "tier_2_lightweight",
        }

        result = client.inbox_command.reprioritize()

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/reprioritize",
            json={},
        )
        assert result["reprioritized"] == 42
        assert result["tier_used"] == "tier_2_lightweight"

    def test_reprioritize_specific_emails(self, client: AragoraClient, mock_request) -> None:
        """Reprioritize specific emails."""
        mock_request.return_value = {
            "reprioritized": 3,
            "changes": [{"id": "email_1", "old": "low", "new": "medium"}],
            "tier_used": "tier_1_rules",
        }

        client.inbox_command.reprioritize(
            email_ids=["email_1", "email_2", "email_3"],
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/reprioritize",
            json={"emailIds": ["email_1", "email_2", "email_3"]},
        )

    def test_reprioritize_force_tier_rules(self, client: AragoraClient, mock_request) -> None:
        """Reprioritize using tier 1 rules."""
        mock_request.return_value = {
            "reprioritized": 10,
            "changes": [],
            "tier_used": "tier_1_rules",
        }

        result = client.inbox_command.reprioritize(force_tier="tier_1_rules")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/reprioritize",
            json={"force_tier": "tier_1_rules"},
        )
        assert result["tier_used"] == "tier_1_rules"

    def test_reprioritize_force_tier_lightweight(self, client: AragoraClient, mock_request) -> None:
        """Reprioritize using tier 2 lightweight AI."""
        mock_request.return_value = {
            "reprioritized": 20,
            "changes": [],
            "tier_used": "tier_2_lightweight",
        }

        client.inbox_command.reprioritize(force_tier="tier_2_lightweight")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/reprioritize",
            json={"force_tier": "tier_2_lightweight"},
        )

    def test_reprioritize_force_tier_debate(self, client: AragoraClient, mock_request) -> None:
        """Reprioritize using tier 3 debate."""
        mock_request.return_value = {
            "reprioritized": 5,
            "changes": [
                {"id": "email_1", "old": "medium", "new": "critical"},
            ],
            "tier_used": "tier_3_debate",
        }

        result = client.inbox_command.reprioritize(force_tier="tier_3_debate")

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/reprioritize",
            json={"force_tier": "tier_3_debate"},
        )
        assert result["tier_used"] == "tier_3_debate"

    def test_reprioritize_with_emails_and_tier(self, client: AragoraClient, mock_request) -> None:
        """Reprioritize specific emails with forced tier."""
        mock_request.return_value = {
            "reprioritized": 2,
            "changes": [],
            "tier_used": "tier_3_debate",
        }

        client.inbox_command.reprioritize(
            email_ids=["email_1", "email_2"],
            force_tier="tier_3_debate",
        )

        mock_request.assert_called_once_with(
            "POST",
            "/api/v1/inbox/reprioritize",
            json={
                "emailIds": ["email_1", "email_2"],
                "force_tier": "tier_3_debate",
            },
        )


class TestAsyncInboxCommand:
    """Tests for async InboxCommand methods."""

    @pytest.mark.asyncio
    async def test_async_get_inbox(self, mock_async_request) -> None:
        """Get inbox asynchronously."""
        mock_async_request.return_value = {
            "emails": [{"id": "email_1", "priority": "high"}],
            "total": 10,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.inbox_command.get_inbox()

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/command",
                params={"limit": 50, "offset": 0},
            )
            assert result["total"] == 10

    @pytest.mark.asyncio
    async def test_async_get_inbox_with_filters(self, mock_async_request) -> None:
        """Get inbox with filters asynchronously."""
        mock_async_request.return_value = {"emails": [], "total": 0}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.inbox_command.get_inbox(
                limit=25,
                offset=10,
                priority="critical",
                unread_only=True,
            )

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/command",
                params={
                    "limit": 25,
                    "offset": 10,
                    "priority": "critical",
                    "unread_only": "true",
                },
            )

    @pytest.mark.asyncio
    async def test_async_quick_action(self, mock_async_request) -> None:
        """Execute quick action asynchronously."""
        mock_async_request.return_value = {"action": "archive", "processed": 3}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.inbox_command.quick_action(
                action="archive",
                email_ids=["email_1", "email_2", "email_3"],
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/actions",
                json={"action": "archive", "emailIds": ["email_1", "email_2", "email_3"]},
            )
            assert result["processed"] == 3

    @pytest.mark.asyncio
    async def test_async_quick_action_with_params(self, mock_async_request) -> None:
        """Execute quick action with params asynchronously."""
        mock_async_request.return_value = {"action": "snooze", "processed": 1}

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.inbox_command.quick_action(
                action="snooze",
                email_ids=["email_1"],
                params={"duration": "2h"},
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/actions",
                json={
                    "action": "snooze",
                    "emailIds": ["email_1"],
                    "params": {"duration": "2h"},
                },
            )

    @pytest.mark.asyncio
    async def test_async_bulk_action(self, mock_async_request) -> None:
        """Execute bulk action asynchronously."""
        mock_async_request.return_value = {
            "action": "archive",
            "filter": "low",
            "processed": 50,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.inbox_command.bulk_action(
                action="archive",
                filter="low",
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/bulk-actions",
                json={"action": "archive", "filter": "low"},
            )
            assert result["processed"] == 50

    @pytest.mark.asyncio
    async def test_async_bulk_action_with_params(self, mock_async_request) -> None:
        """Execute bulk action with params asynchronously."""
        mock_async_request.return_value = {
            "action": "snooze",
            "filter": "deferred",
            "processed": 10,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.inbox_command.bulk_action(
                action="snooze",
                filter="deferred",
                params={"duration": "1d"},
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/bulk-actions",
                json={
                    "action": "snooze",
                    "filter": "deferred",
                    "params": {"duration": "1d"},
                },
            )

    @pytest.mark.asyncio
    async def test_async_get_sender_profile(self, mock_async_request) -> None:
        """Get sender profile asynchronously."""
        mock_async_request.return_value = {
            "email": "vip@company.com",
            "name": "VIP Contact",
            "isVip": True,
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.inbox_command.get_sender_profile("vip@company.com")

            mock_async_request.assert_called_once_with(
                "GET",
                "/api/v1/inbox/sender-profile",
                params={"email": "vip@company.com"},
            )
            assert result["isVip"] is True

    @pytest.mark.asyncio
    async def test_async_get_daily_digest(self, mock_async_request) -> None:
        """Get daily digest asynchronously."""
        mock_async_request.return_value = {
            "emailsReceived": 100,
            "processed": 95,
            "timeSaved": "4h",
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.inbox_command.get_daily_digest()

            mock_async_request.assert_called_once_with("GET", "/api/v1/inbox/daily-digest")
            assert result["emailsReceived"] == 100
            assert result["timeSaved"] == "4h"

    @pytest.mark.asyncio
    async def test_async_reprioritize_default(self, mock_async_request) -> None:
        """Reprioritize inbox asynchronously."""
        mock_async_request.return_value = {
            "reprioritized": 30,
            "changes": [],
            "tier_used": "tier_2_lightweight",
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.inbox_command.reprioritize()

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/reprioritize",
                json={},
            )
            assert result["reprioritized"] == 30

    @pytest.mark.asyncio
    async def test_async_reprioritize_with_emails(self, mock_async_request) -> None:
        """Reprioritize specific emails asynchronously."""
        mock_async_request.return_value = {
            "reprioritized": 2,
            "changes": [],
            "tier_used": "tier_1_rules",
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.inbox_command.reprioritize(email_ids=["email_1", "email_2"])

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/reprioritize",
                json={"emailIds": ["email_1", "email_2"]},
            )

    @pytest.mark.asyncio
    async def test_async_reprioritize_with_tier(self, mock_async_request) -> None:
        """Reprioritize with forced tier asynchronously."""
        mock_async_request.return_value = {
            "reprioritized": 5,
            "changes": [{"id": "email_1", "old": "low", "new": "critical"}],
            "tier_used": "tier_3_debate",
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            result = await client.inbox_command.reprioritize(force_tier="tier_3_debate")

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/reprioritize",
                json={"force_tier": "tier_3_debate"},
            )
            assert result["tier_used"] == "tier_3_debate"

    @pytest.mark.asyncio
    async def test_async_reprioritize_with_all_options(self, mock_async_request) -> None:
        """Reprioritize with all options asynchronously."""
        mock_async_request.return_value = {
            "reprioritized": 3,
            "changes": [],
            "tier_used": "tier_2_lightweight",
        }

        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            await client.inbox_command.reprioritize(
                email_ids=["email_1", "email_2", "email_3"],
                force_tier="tier_2_lightweight",
            )

            mock_async_request.assert_called_once_with(
                "POST",
                "/api/v1/inbox/reprioritize",
                json={
                    "emailIds": ["email_1", "email_2", "email_3"],
                    "force_tier": "tier_2_lightweight",
                },
            )
