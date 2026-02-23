"""Tests for the Unified Inbox messages module.

Covers all public functions in aragora/server/handlers/features/unified_inbox/messages.py:
- fetch_all_messages     - Fetch messages from all connected accounts
- fetch_gmail_messages   - Fetch messages from Gmail with sync service lookup
- fetch_outlook_messages - Fetch messages from Outlook with sync service lookup
- generate_sample_messages - Generate sample messages for testing
- score_messages         - Apply priority scoring to messages

Tests include:
- Happy paths for all functions
- Store with existing messages (cache hit)
- Store empty, accounts with various statuses (CONNECTED, PENDING, ERROR)
- Gmail vs Outlook provider routing
- Sync service active with initial_sync_complete
- Sync service active without initial_sync_complete (fallback to samples)
- Sync service not registered
- Empty tenant_id
- Fetch errors (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError, KeyError)
- Error counting on fetch failure
- score_messages with and without EmailPrioritizer import
- generate_sample_messages edge cases (count=0, count > subjects, normal)
- Sample message field correctness
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.unified_inbox.models import (
    AccountStatus,
    ConnectedAccount,
    EmailProvider,
    UnifiedMessage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_account(
    account_id: str = "acct-test-001",
    provider: EmailProvider = EmailProvider.GMAIL,
    status: AccountStatus = AccountStatus.CONNECTED,
    email: str = "user@example.com",
) -> ConnectedAccount:
    """Create a ConnectedAccount for testing."""
    return ConnectedAccount(
        id=account_id,
        provider=provider,
        email_address=email,
        display_name="Test User",
        status=status,
        connected_at=_NOW,
    )


def _account_record(
    account_id: str = "acct-1",
    provider: str = "gmail",
    email: str = "user@gmail.com",
    status: str = "connected",
) -> dict[str, Any]:
    return {
        "id": account_id,
        "provider": provider,
        "email_address": email,
        "display_name": email.split("@")[0],
        "status": status,
        "connected_at": _NOW,
        "last_sync": _NOW,
        "total_messages": 10,
        "unread_count": 3,
        "sync_errors": 0,
        "metadata": {},
    }


def _message_record(
    message_id: str = "msg-1",
    account_id: str = "acct-1",
    provider: str = "gmail",
    subject: str = "Test Subject",
) -> dict[str, Any]:
    return {
        "id": message_id,
        "account_id": account_id,
        "provider": provider,
        "external_id": "ext-123",
        "subject": subject,
        "sender_email": "sender@example.com",
        "sender_name": "Sender Name",
        "recipients": ["user@gmail.com"],
        "cc": [],
        "received_at": _NOW,
        "snippet": "Preview of the message...",
        "body_preview": "Full preview...",
        "is_read": False,
        "is_starred": False,
        "has_attachments": False,
        "labels": ["inbox"],
        "thread_id": None,
        "priority_score": 0.5,
        "priority_tier": "medium",
        "priority_reasons": [],
        "triage_action": None,
        "triage_rationale": None,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_sync_services():
    """Clear the global sync services registry between tests."""
    from aragora.server.handlers.features.unified_inbox.sync import _sync_services

    _sync_services.clear()
    yield
    _sync_services.clear()


@pytest.fixture
def mock_store():
    """Create a fully-mocked inbox store."""
    store = AsyncMock()
    store.list_messages = AsyncMock(return_value=([], 0))
    store.list_accounts = AsyncMock(return_value=[])
    store.save_message = AsyncMock(return_value=("msg-1", True))
    store.increment_account_counts = AsyncMock(return_value=None)
    return store


# ===========================================================================
# generate_sample_messages
# ===========================================================================


class TestGenerateSampleMessages:
    """Tests for generate_sample_messages."""

    def test_generates_five_messages_by_default(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert len(messages) == 5

    def test_generates_fewer_if_count_less_than_subjects(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 3)
        assert len(messages) == 3

    def test_generates_zero_messages(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 0)
        assert len(messages) == 0

    def test_count_exceeding_subjects_capped(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 100)
        # Only 5 sample subjects exist
        assert len(messages) == 5

    def test_messages_have_correct_account_id(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account(account_id="acct-xyz")
        messages = generate_sample_messages(account, 3)
        for msg in messages:
            assert msg.account_id == "acct-xyz"

    def test_messages_have_correct_provider(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account(provider=EmailProvider.OUTLOOK)
        messages = generate_sample_messages(account, 2)
        for msg in messages:
            assert msg.provider == EmailProvider.OUTLOOK

    def test_first_message_is_critical_priority(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert messages[0].priority_tier == "critical"
        assert messages[0].priority_score == 0.95

    def test_second_message_is_high_priority(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert messages[1].priority_tier == "high"
        assert messages[1].priority_score == 0.75

    def test_third_message_is_medium_priority(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert messages[2].priority_tier == "medium"
        assert messages[2].priority_score == 0.5

    def test_fourth_message_is_low_priority(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert messages[3].priority_tier == "low"
        assert messages[3].priority_score == 0.25

    def test_fifth_message_is_medium_priority(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert messages[4].priority_tier == "medium"
        assert messages[4].priority_score == 0.5

    def test_first_message_is_starred(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert messages[0].is_starred is True
        assert messages[1].is_starred is False

    def test_first_two_have_attachments(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        assert messages[0].has_attachments is True
        assert messages[1].has_attachments is True
        assert messages[2].has_attachments is False

    def test_first_three_unread_rest_read(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        # is_read = i > 2, so i=0,1,2 are unread, i=3,4 are read
        assert messages[0].is_read is False
        assert messages[1].is_read is False
        assert messages[2].is_read is False
        assert messages[3].is_read is True
        assert messages[4].is_read is True

    def test_recipients_match_account_email(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account(email="me@corp.com")
        messages = generate_sample_messages(account, 1)
        assert messages[0].recipients == ["me@corp.com"]

    def test_sender_emails_follow_pattern(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 3)
        for i, msg in enumerate(messages):
            assert msg.sender_email == f"sender{i}@example.com"
            assert msg.sender_name == f"Sender {i}"

    def test_received_at_decreasing(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        for i in range(1, len(messages)):
            assert messages[i].received_at < messages[i - 1].received_at

    def test_messages_have_unique_ids(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        ids = [m.id for m in messages]
        assert len(set(ids)) == 5

    def test_all_messages_have_inbox_label(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        for msg in messages:
            assert msg.labels == ["inbox"]

    def test_subjects_match_expected_order(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, 5)
        expected_subjects = [
            "Urgent: Contract Review Required",
            "Q4 Budget Approval Needed",
            "Weekly Team Update",
            "Newsletter: Industry Updates",
            "Meeting Rescheduled",
        ]
        for i, msg in enumerate(messages):
            assert msg.subject == expected_subjects[i]


# ===========================================================================
# score_messages
# ===========================================================================


class TestScoreMessages:
    """Tests for score_messages."""

    @pytest.mark.asyncio
    async def test_returns_messages_unchanged_when_prioritizer_unavailable(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            score_messages,
        )

        account = _make_account()
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        messages = generate_sample_messages(account, 3)
        scored = await score_messages(messages)
        assert scored is messages
        assert len(scored) == 3

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_input(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            score_messages,
        )

        scored = await score_messages([])
        assert scored == []

    @pytest.mark.asyncio
    async def test_returns_messages_when_prioritizer_available(self):
        """When EmailPrioritizer is importable, messages still returned as-is."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            score_messages,
        )

        mock_prioritizer = MagicMock()
        mock_priority = MagicMock()
        mock_module = MagicMock()
        mock_module.EmailPrioritizer = mock_prioritizer
        mock_module.EmailPriority = mock_priority

        account = _make_account()
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        messages = generate_sample_messages(account, 2)

        with patch.dict("sys.modules", {
            "aragora.services.email_prioritization": mock_module,
        }):
            scored = await score_messages(messages)
            assert len(scored) == 2


# ===========================================================================
# fetch_gmail_messages
# ===========================================================================


class TestFetchGmailMessages:
    """Tests for fetch_gmail_messages."""

    @pytest.mark.asyncio
    async def test_no_sync_service_returns_samples(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_gmail_messages,
        )

        account = _make_account(provider=EmailProvider.GMAIL)
        messages = await fetch_gmail_messages(account, "tenant-1")
        assert len(messages) == 5
        # They should be sample messages
        assert messages[0].subject == "Urgent: Contract Review Required"

    @pytest.mark.asyncio
    async def test_sync_service_active_initial_sync_complete(self):
        """When sync service exists and initial_sync_complete is True, return empty."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_gmail_messages,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_state = MagicMock()
        mock_state.initial_sync_complete = True
        mock_state.total_messages_synced = 42

        mock_service = MagicMock()
        mock_service.state = mock_state

        services = get_sync_services()
        services["tenant-1"] = {"acct-test-001": mock_service}

        account = _make_account(account_id="acct-test-001", provider=EmailProvider.GMAIL)
        messages = await fetch_gmail_messages(account, "tenant-1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_sync_service_active_initial_sync_not_complete(self):
        """When sync service exists but initial_sync_complete is False, return samples."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_gmail_messages,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_state = MagicMock()
        mock_state.initial_sync_complete = False

        mock_service = MagicMock()
        mock_service.state = mock_state

        services = get_sync_services()
        services["tenant-1"] = {"acct-test-001": mock_service}

        account = _make_account(account_id="acct-test-001", provider=EmailProvider.GMAIL)
        messages = await fetch_gmail_messages(account, "tenant-1")
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_sync_service_without_state(self):
        """When sync service exists but has no state attribute, return samples."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_gmail_messages,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service = MagicMock(spec=[])  # No state attribute

        services = get_sync_services()
        services["tenant-1"] = {"acct-test-001": mock_service}

        account = _make_account(account_id="acct-test-001", provider=EmailProvider.GMAIL)
        messages = await fetch_gmail_messages(account, "tenant-1")
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_empty_tenant_id_returns_samples(self):
        """When tenant_id is empty/falsy, skip sync service lookup."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_gmail_messages,
        )

        account = _make_account(provider=EmailProvider.GMAIL)
        messages = await fetch_gmail_messages(account, "")
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_tenant_not_in_services_returns_samples(self):
        """When tenant exists in services but account doesn't."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_gmail_messages,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        services = get_sync_services()
        services["tenant-1"] = {"other-acct": MagicMock()}

        account = _make_account(account_id="acct-test-001", provider=EmailProvider.GMAIL)
        messages = await fetch_gmail_messages(account, "tenant-1")
        assert len(messages) == 5


# ===========================================================================
# fetch_outlook_messages
# ===========================================================================


class TestFetchOutlookMessages:
    """Tests for fetch_outlook_messages."""

    @pytest.mark.asyncio
    async def test_no_sync_service_returns_samples(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_outlook_messages,
        )

        account = _make_account(provider=EmailProvider.OUTLOOK)
        messages = await fetch_outlook_messages(account, "tenant-1")
        assert len(messages) == 5
        assert messages[0].provider == EmailProvider.OUTLOOK

    @pytest.mark.asyncio
    async def test_sync_service_active_initial_sync_complete(self):
        """When sync service exists and initial_sync_complete is True, return empty."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_outlook_messages,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_state = MagicMock()
        mock_state.initial_sync_complete = True
        mock_state.total_messages_synced = 10

        mock_service = MagicMock()
        mock_service.state = mock_state

        services = get_sync_services()
        services["tenant-1"] = {"acct-test-001": mock_service}

        account = _make_account(account_id="acct-test-001", provider=EmailProvider.OUTLOOK)
        messages = await fetch_outlook_messages(account, "tenant-1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_sync_service_active_initial_sync_not_complete(self):
        """When sync service exists but initial_sync_complete is False, return samples."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_outlook_messages,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_state = MagicMock()
        mock_state.initial_sync_complete = False

        mock_service = MagicMock()
        mock_service.state = mock_state

        services = get_sync_services()
        services["tenant-1"] = {"acct-test-001": mock_service}

        account = _make_account(account_id="acct-test-001", provider=EmailProvider.OUTLOOK)
        messages = await fetch_outlook_messages(account, "tenant-1")
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_sync_service_without_state(self):
        """When sync service has no state attribute."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_outlook_messages,
        )
        from aragora.server.handlers.features.unified_inbox.sync import (
            get_sync_services,
        )

        mock_service = MagicMock(spec=[])

        services = get_sync_services()
        services["tenant-1"] = {"acct-test-001": mock_service}

        account = _make_account(account_id="acct-test-001", provider=EmailProvider.OUTLOOK)
        messages = await fetch_outlook_messages(account, "tenant-1")
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_empty_tenant_id_returns_samples(self):
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_outlook_messages,
        )

        account = _make_account(provider=EmailProvider.OUTLOOK)
        messages = await fetch_outlook_messages(account, "")
        assert len(messages) == 5


# ===========================================================================
# fetch_all_messages
# ===========================================================================


class TestFetchAllMessages:
    """Tests for fetch_all_messages."""

    @pytest.mark.asyncio
    async def test_returns_cached_messages_when_store_has_data(self, mock_store):
        """When store.list_messages returns records, use them directly."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        records = [
            _message_record("msg-1", subject="Cached One"),
            _message_record("msg-2", subject="Cached Two"),
        ]
        mock_store.list_messages.return_value = (records, 2)

        messages = await fetch_all_messages("tenant-1", mock_store)
        assert len(messages) == 2
        assert messages[0].subject == "Cached One"
        assert messages[1].subject == "Cached Two"
        # Should not call list_accounts when cache hit
        mock_store.list_accounts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetches_from_accounts_when_store_empty(self, mock_store):
        """When store has no messages, fetch from connected accounts."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        messages = await fetch_all_messages("tenant-1", mock_store)
        # Should generate 5 sample messages for the connected Gmail account
        assert len(messages) == 5
        # Messages should have been saved to store
        assert mock_store.save_message.await_count == 5

    @pytest.mark.asyncio
    async def test_skips_disconnected_accounts(self, mock_store):
        """Accounts with status != CONNECTED should be skipped."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="disconnected"),
            _account_record("acct-2", "outlook", status="pending"),
            _account_record("acct-3", "gmail", status="error"),
        ]

        messages = await fetch_all_messages("tenant-1", mock_store)
        assert len(messages) == 0
        mock_store.save_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fetches_gmail_and_outlook(self, mock_store):
        """Both Gmail and Outlook accounts should produce messages."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
            _account_record("acct-2", "outlook", status="connected"),
        ]

        messages = await fetch_all_messages("tenant-1", mock_store)
        # 5 samples per account
        assert len(messages) == 10
        assert mock_store.save_message.await_count == 10

    @pytest.mark.asyncio
    async def test_connection_error_increments_sync_errors(self, mock_store):
        """When fetching raises ConnectionError, increment sync error count."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            side_effect=ConnectionError("refused"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            assert len(messages) == 0
            mock_store.increment_account_counts.assert_awaited_once_with(
                "tenant-1", "acct-1", sync_error_delta=1
            )

    @pytest.mark.asyncio
    async def test_timeout_error_increments_sync_errors(self, mock_store):
        """When fetching raises TimeoutError, increment sync error count."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            side_effect=TimeoutError("timed out"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            assert len(messages) == 0
            mock_store.increment_account_counts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_os_error_increments_sync_errors(self, mock_store):
        """When fetching raises OSError, increment sync error count."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "outlook", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_outlook_messages",
            new_callable=AsyncMock,
            side_effect=OSError("disk error"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            assert len(messages) == 0
            mock_store.increment_account_counts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_value_error_increments_sync_errors(self, mock_store):
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            side_effect=ValueError("bad value"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            assert len(messages) == 0
            mock_store.increment_account_counts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runtime_error_increments_sync_errors(self, mock_store):
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            side_effect=RuntimeError("runtime error"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            assert len(messages) == 0
            mock_store.increment_account_counts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_key_error_increments_sync_errors(self, mock_store):
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            side_effect=KeyError("missing key"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            assert len(messages) == 0
            mock_store.increment_account_counts.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_partial_failure_returns_successful_messages(self, mock_store):
        """When one account fails but another succeeds, return the successful ones."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
            _account_record("acct-2", "outlook", status="connected"),
        ]

        # Gmail fails, Outlook succeeds
        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            side_effect=ConnectionError("gmail down"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            # Only Outlook messages (5 samples)
            assert len(messages) == 5
            # Gmail error counted
            mock_store.increment_account_counts.assert_awaited_once_with(
                "tenant-1", "acct-1", sync_error_delta=1
            )
            # Outlook messages saved
            assert mock_store.save_message.await_count == 5

    @pytest.mark.asyncio
    async def test_no_accounts_returns_empty(self, mock_store):
        """When there are no accounts, return empty list."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = []

        messages = await fetch_all_messages("tenant-1", mock_store)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_messages_are_scored_after_fetch(self, mock_store):
        """After fetching, score_messages should be called."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.score_messages",
            new_callable=AsyncMock,
        ) as mock_score:
            mock_score.return_value = []
            messages = await fetch_all_messages("tenant-1", mock_store)
            mock_score.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_outlook_routing(self, mock_store):
        """Outlook accounts should call fetch_outlook_messages."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "outlook", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_outlook_messages",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_fetch:
            await fetch_all_messages("tenant-1", mock_store)
            mock_fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gmail_routing(self, mock_store):
        """Gmail accounts should call fetch_gmail_messages."""
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_fetch:
            await fetch_all_messages("tenant-1", mock_store)
            mock_fetch.assert_awaited_once()


# ===========================================================================
# Parametrized tests
# ===========================================================================


class TestParametrizedFetchErrors:
    """Parametrized tests for all handled exception types in fetch_all_messages."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_cls", [
        ConnectionError,
        TimeoutError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ])
    async def test_all_error_types_handled(self, mock_store, error_cls):
        from aragora.server.handlers.features.unified_inbox.messages import (
            fetch_all_messages,
        )

        mock_store.list_messages.return_value = ([], 0)
        mock_store.list_accounts.return_value = [
            _account_record("acct-1", "gmail", status="connected"),
        ]

        with patch(
            "aragora.server.handlers.features.unified_inbox.messages.fetch_gmail_messages",
            new_callable=AsyncMock,
            side_effect=error_cls("test error"),
        ):
            messages = await fetch_all_messages("tenant-1", mock_store)
            assert len(messages) == 0
            mock_store.increment_account_counts.assert_awaited_once()


class TestParametrizedSampleCounts:
    """Parametrized tests for various sample message counts."""

    @pytest.mark.parametrize("count,expected", [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (10, 5),
        (100, 5),
    ])
    def test_sample_count_clamping(self, count, expected):
        from aragora.server.handlers.features.unified_inbox.messages import (
            generate_sample_messages,
        )

        account = _make_account()
        messages = generate_sample_messages(account, count)
        assert len(messages) == expected
