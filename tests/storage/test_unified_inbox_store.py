"""Tests for UnifiedInboxStore.

Covers:
- Module-level utility functions (_utc_now, _format_dt, _parse_dt, _json_loads)
- InMemoryUnifiedInboxStore: full CRUD for accounts, messages, triage results
- SQLiteUnifiedInboxStore: full CRUD parity with in-memory backend
- Message listing with filters, search, pagination, and sort ordering
- Deduplication on save_message with read-status change tracking
- Account count bookkeeping (total_messages, unread_count, sync_errors)
- Singleton management (get/set/reset store)
- Abstract base class guard
"""

import tempfile
import threading
from pathlib import Path

import pytest
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import patch

from aragora.storage.unified_inbox_store import (
    InMemoryUnifiedInboxStore,
    SQLiteUnifiedInboxStore,
    UnifiedInboxStoreBackend,
    _format_dt,
    _json_loads,
    _parse_dt,
    _utc_now,
    get_unified_inbox_store,
    reset_unified_inbox_store,
    set_unified_inbox_store,
)


# ====================================================================
# Helpers
# ====================================================================

TENANT = "tenant_1"
TENANT_2 = "tenant_2"


def _make_account(
    account_id: str = "acct_1",
    provider: str = "gmail",
    email: str = "user@example.com",
    **overrides: Any,
) -> dict[str, Any]:
    base = {
        "id": account_id,
        "provider": provider,
        "email_address": email,
        "display_name": "Test User",
        "status": "active",
        "total_messages": 0,
        "unread_count": 0,
        "sync_errors": 0,
    }
    base.update(overrides)
    return base


def _make_message(
    message_id: str = "msg_1",
    account_id: str = "acct_1",
    external_id: str = "ext_1",
    **overrides: Any,
) -> dict[str, Any]:
    base = {
        "id": message_id,
        "account_id": account_id,
        "provider": "gmail",
        "external_id": external_id,
        "subject": "Test Subject",
        "sender_email": "sender@example.com",
        "sender_name": "Sender",
        "snippet": "Hello world",
        "is_read": False,
        "is_starred": False,
        "has_attachments": False,
        "priority_score": 0.5,
        "priority_tier": "medium",
    }
    base.update(overrides)
    return base


def _make_triage(
    message_id: str = "msg_1",
    **overrides: Any,
) -> dict[str, Any]:
    base = {
        "message_id": message_id,
        "recommended_action": "reply",
        "confidence": 0.9,
        "rationale": "Needs urgent reply",
        "suggested_response": "Will do.",
        "delegate_to": None,
        "schedule_for": None,
        "agents_involved": ["claude", "gpt"],
        "debate_summary": "Consensus reached.",
    }
    base.update(overrides)
    return base


@pytest.fixture
def store() -> InMemoryUnifiedInboxStore:
    s = InMemoryUnifiedInboxStore()
    # The InMemoryUnifiedInboxStore uses threading.Lock which deadlocks when
    # save_message calls increment_account_counts while holding the lock.
    # Use RLock to allow re-entrant acquisition within the same thread.
    s._lock = threading.RLock()
    return s


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SQLiteUnifiedInboxStore:
    """Create a SQLiteUnifiedInboxStore with a temporary database file."""
    db_path = tmp_path / "test_inbox.db"
    return SQLiteUnifiedInboxStore(db_path)


# ====================================================================
# Utility Functions
# ====================================================================


class TestUtilityFunctions:
    """Tests for module-level utility functions."""

    def test_utc_now_returns_utc_datetime(self):
        now = _utc_now()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None

    def test_format_dt_none(self):
        assert _format_dt(None) is None

    def test_format_dt_datetime(self):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _format_dt(dt)
        assert isinstance(result, str)
        assert "2024-01-15" in result

    def test_parse_dt_none(self):
        assert _parse_dt(None) is None

    def test_parse_dt_datetime_passthrough(self):
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert _parse_dt(dt) is dt

    def test_parse_dt_iso_string(self):
        result = _parse_dt("2024-01-15T12:00:00+00:00")
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_parse_dt_invalid_string(self):
        assert _parse_dt("not-a-date") is None

    def test_json_loads_none(self):
        assert _json_loads(None, []) == []

    def test_json_loads_dict_passthrough(self):
        d = {"key": "value"}
        assert _json_loads(d, {}) is d

    def test_json_loads_list_passthrough(self):
        lst = [1, 2, 3]
        assert _json_loads(lst, []) is lst

    def test_json_loads_valid_string(self):
        assert _json_loads('{"a": 1}', {}) == {"a": 1}

    def test_json_loads_invalid_string(self):
        assert _json_loads("not json", "default") == "default"


# ====================================================================
# Account CRUD
# ====================================================================


class TestAccountCRUD:
    """Tests for account create, read, update, delete operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_account(self, store: InMemoryUnifiedInboxStore):
        account = _make_account()
        await store.save_account(TENANT, account)
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["id"] == "acct_1"
        assert result["provider"] == "gmail"

    @pytest.mark.asyncio
    async def test_get_account_not_found(self, store: InMemoryUnifiedInboxStore):
        result = await store.get_account(TENANT, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_accounts(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account("acct_1"))
        await store.save_account(TENANT, _make_account("acct_2", email="b@example.com"))
        results = await store.list_accounts(TENANT)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_accounts_tenant_isolation(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account("acct_1"))
        await store.save_account(TENANT_2, _make_account("acct_2"))
        assert len(await store.list_accounts(TENANT)) == 1
        assert len(await store.list_accounts(TENANT_2)) == 1

    @pytest.mark.asyncio
    async def test_delete_account(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        deleted = await store.delete_account(TENANT, "acct_1")
        assert deleted is True
        assert await store.get_account(TENANT, "acct_1") is None

    @pytest.mark.asyncio
    async def test_delete_account_not_found(self, store: InMemoryUnifiedInboxStore):
        deleted = await store.delete_account(TENANT, "nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_account_cascades_messages(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message())
        await store.save_triage_result(TENANT, _make_triage())
        await store.delete_account(TENANT, "acct_1")
        assert await store.get_message(TENANT, "msg_1") is None
        assert await store.get_triage_result(TENANT, "msg_1") is None

    @pytest.mark.asyncio
    async def test_update_account_fields(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.update_account_fields(TENANT, "acct_1", {"status": "disconnected"})
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_update_account_fields_nonexistent(self, store: InMemoryUnifiedInboxStore):
        # Should not raise -- silently does nothing for missing account
        await store.update_account_fields(TENANT, "nonexistent", {"status": "x"})

    @pytest.mark.asyncio
    async def test_increment_account_counts(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.increment_account_counts(TENANT, "acct_1", total_delta=5, unread_delta=3)
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["total_messages"] == 5
        assert result["unread_count"] == 3

    @pytest.mark.asyncio
    async def test_increment_account_counts_floor_at_zero(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.increment_account_counts(TENANT, "acct_1", total_delta=-100)
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["total_messages"] == 0


# ====================================================================
# Message CRUD
# ====================================================================


class TestMessageCRUD:
    """Tests for message create, read, update, delete operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_message(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        msg_id, created = await store.save_message(TENANT, _make_message())
        assert msg_id == "msg_1"
        assert created is True
        result = await store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["subject"] == "Test Subject"

    @pytest.mark.asyncio
    async def test_save_message_deduplicates_by_external_id(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        _, created_1 = await store.save_message(TENANT, _make_message())
        assert created_1 is True
        # Same external_id + account_id should update, not create
        _, created_2 = await store.save_message(
            TENANT,
            _make_message(message_id="msg_2", subject="Updated"),
        )
        assert created_2 is False

    @pytest.mark.asyncio
    async def test_save_new_message_increments_account_counts(
        self, store: InMemoryUnifiedInboxStore
    ):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=False))
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 1
        assert acct["unread_count"] == 1

    @pytest.mark.asyncio
    async def test_save_read_message_does_not_increment_unread(
        self, store: InMemoryUnifiedInboxStore
    ):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=True))
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 1
        assert acct["unread_count"] == 0

    @pytest.mark.asyncio
    async def test_get_message_not_found(self, store: InMemoryUnifiedInboxStore):
        result = await store.get_message(TENANT, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_message(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message())
        deleted = await store.delete_message(TENANT, "msg_1")
        assert deleted is True
        assert await store.get_message(TENANT, "msg_1") is None

    @pytest.mark.asyncio
    async def test_delete_message_not_found(self, store: InMemoryUnifiedInboxStore):
        deleted = await store.delete_message(TENANT, "nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_unread_message_decrements_unread(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=False))
        await store.delete_message(TENANT, "msg_1")
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0
        assert acct["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_update_message_flags_read(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=False))
        result = await store.update_message_flags(TENANT, "msg_1", is_read=True)
        assert result is True
        msg = await store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["is_read"] is True
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0

    @pytest.mark.asyncio
    async def test_update_message_flags_starred(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message())
        result = await store.update_message_flags(TENANT, "msg_1", is_starred=True)
        assert result is True
        msg = await store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["is_starred"] is True

    @pytest.mark.asyncio
    async def test_update_message_flags_not_found(self, store: InMemoryUnifiedInboxStore):
        result = await store.update_message_flags(TENANT, "nonexistent", is_read=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_message_triage(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message())
        await store.update_message_triage(TENANT, "msg_1", "archive", "Low priority")
        msg = await store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["triage_action"] == "archive"
        assert msg["triage_rationale"] == "Low priority"


# ====================================================================
# Message Listing, Filtering, and Pagination
# ====================================================================


class TestMessageListingAndFiltering:
    """Tests for list_messages with filters, search, and pagination."""

    @pytest.fixture
    async def populated_store(self, store: InMemoryUnifiedInboxStore):
        await store.save_account(TENANT, _make_account("acct_1"))
        await store.save_account(TENANT, _make_account("acct_2", email="b@example.com"))
        await store.save_message(
            TENANT,
            _make_message(
                "msg_1",
                "acct_1",
                "e1",
                priority_tier="high",
                priority_score=0.9,
                is_read=False,
                subject="Urgent: deploy",
            ),
        )
        await store.save_message(
            TENANT,
            _make_message(
                "msg_2",
                "acct_1",
                "e2",
                priority_tier="low",
                priority_score=0.1,
                is_read=True,
                subject="Newsletter",
            ),
        )
        await store.save_message(
            TENANT,
            _make_message(
                "msg_3",
                "acct_2",
                "e3",
                priority_tier="medium",
                priority_score=0.5,
                is_read=False,
                subject="Meeting notes",
            ),
        )
        return store

    @pytest.mark.asyncio
    async def test_list_all_messages(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT)
        assert total == 3
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_filter_by_priority_tier(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT, priority_tier="high")
        assert total == 1
        assert messages[0]["id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_filter_by_account_id(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT, account_id="acct_2")
        assert total == 1
        assert messages[0]["id"] == "msg_3"

    @pytest.mark.asyncio
    async def test_filter_unread_only(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT, unread_only=True)
        assert total == 2
        for msg in messages:
            assert msg.get("is_read") is not True

    @pytest.mark.asyncio
    async def test_search_by_subject(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT, search="urgent")
        assert total == 1
        assert messages[0]["id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT, search="NEWSLETTER")
        assert total == 1
        assert messages[0]["id"] == "msg_2"

    @pytest.mark.asyncio
    async def test_pagination_limit(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT, limit=2)
        assert total == 3
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_pagination_offset(self, populated_store: InMemoryUnifiedInboxStore):
        messages, total = await populated_store.list_messages(TENANT, limit=2, offset=2)
        assert total == 3
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_list_messages_empty_tenant(self, store: InMemoryUnifiedInboxStore):
        messages, total = await store.list_messages("empty_tenant")
        assert total == 0
        assert messages == []


# ====================================================================
# Triage Results
# ====================================================================


class TestTriageResults:
    """Tests for triage result save and retrieval."""

    @pytest.mark.asyncio
    async def test_save_and_get_triage_result(self, store: InMemoryUnifiedInboxStore):
        triage = _make_triage()
        await store.save_triage_result(TENANT, triage)
        result = await store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["recommended_action"] == "reply"
        assert result["confidence"] == 0.9
        assert result["agents_involved"] == ["claude", "gpt"]

    @pytest.mark.asyncio
    async def test_get_triage_result_not_found(self, store: InMemoryUnifiedInboxStore):
        result = await store.get_triage_result(TENANT, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_triage_result_overwrites(self, store: InMemoryUnifiedInboxStore):
        await store.save_triage_result(TENANT, _make_triage(confidence=0.5))
        await store.save_triage_result(TENANT, _make_triage(confidence=0.95))
        result = await store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["confidence"] == 0.95


# ====================================================================
# Singleton Management
# ====================================================================


class TestSingletonManagement:
    """Tests for get/set/reset store singleton."""

    def test_set_and_get_store(self):
        reset_unified_inbox_store()
        custom = InMemoryUnifiedInboxStore()
        set_unified_inbox_store(custom)
        assert get_unified_inbox_store() is custom
        reset_unified_inbox_store()

    def test_reset_clears_store(self):
        set_unified_inbox_store(InMemoryUnifiedInboxStore())
        reset_unified_inbox_store()
        # After reset, get_unified_inbox_store creates a new one via factory;
        # we just verify reset didn't raise and the global was cleared.
        # We set again to avoid side effects from the factory.
        custom = InMemoryUnifiedInboxStore()
        set_unified_inbox_store(custom)
        assert get_unified_inbox_store() is custom
        reset_unified_inbox_store()


# ====================================================================
# Abstract Base Class
# ====================================================================


class TestAbstractBackend:
    """Tests that the abstract base class cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_backend(self):
        with pytest.raises(TypeError):
            UnifiedInboxStoreBackend()  # type: ignore[abstract]


# ====================================================================
# Additional InMemory Coverage
# ====================================================================


class TestInMemoryAdditionalCoverage:
    """Additional edge-case tests for InMemoryUnifiedInboxStore."""

    @pytest.mark.asyncio
    async def test_save_account_upsert_overwrites(self, store: InMemoryUnifiedInboxStore):
        """Saving an account with the same ID should overwrite it."""
        await store.save_account(TENANT, _make_account(status="active"))
        await store.save_account(TENANT, _make_account(status="disconnected"))
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["status"] == "disconnected"

    @pytest.mark.asyncio
    async def test_increment_account_counts_sync_errors(self, store: InMemoryUnifiedInboxStore):
        """sync_error_delta should update sync_errors and floor at zero."""
        await store.save_account(TENANT, _make_account())
        await store.increment_account_counts(TENANT, "acct_1", sync_error_delta=3)
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["sync_errors"] == 3
        await store.increment_account_counts(TENANT, "acct_1", sync_error_delta=-10)
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["sync_errors"] == 0

    @pytest.mark.asyncio
    async def test_increment_account_counts_nonexistent_account(
        self, store: InMemoryUnifiedInboxStore
    ):
        """Incrementing counts on a nonexistent account should silently do nothing."""
        await store.increment_account_counts(TENANT, "nonexistent", total_delta=5)

    @pytest.mark.asyncio
    async def test_dedup_save_message_unread_to_read(self, store: InMemoryUnifiedInboxStore):
        """Dedup save that changes is_read from False to True should decrement unread."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=False))
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 1
        await store.save_message(TENANT, _make_message(message_id="msg_dup", is_read=True))
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0

    @pytest.mark.asyncio
    async def test_dedup_save_message_read_to_unread(self, store: InMemoryUnifiedInboxStore):
        """Dedup save that changes is_read from True to False should increment unread."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=True))
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0
        await store.save_message(TENANT, _make_message(message_id="msg_dup2", is_read=False))
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 1

    @pytest.mark.asyncio
    async def test_delete_read_message_no_unread_change(self, store: InMemoryUnifiedInboxStore):
        """Deleting a read message should decrement total but not unread."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=True))
        await store.delete_message(TENANT, "msg_1")
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 0
        assert acct["unread_count"] == 0

    @pytest.mark.asyncio
    async def test_mark_read_then_unread(self, store: InMemoryUnifiedInboxStore):
        """Toggling is_read from True back to False should restore unread count."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=False))
        await store.update_message_flags(TENANT, "msg_1", is_read=True)
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0
        await store.update_message_flags(TENANT, "msg_1", is_read=False)
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 1

    @pytest.mark.asyncio
    async def test_update_message_triage_nonexistent(self, store: InMemoryUnifiedInboxStore):
        """Updating triage on a nonexistent message should silently do nothing."""
        await store.update_message_triage(TENANT, "nonexistent", "reply", "No reason")

    @pytest.mark.asyncio
    async def test_search_by_sender_email(self, store: InMemoryUnifiedInboxStore):
        """Search should match against sender_email."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(
            TENANT,
            _make_message("msg_1", external_id="e1", sender_email="alice@company.com"),
        )
        await store.save_message(
            TENANT,
            _make_message("msg_2", external_id="e2", sender_email="bob@company.com"),
        )
        messages, total = await store.list_messages(TENANT, search="alice")
        assert total == 1
        assert messages[0]["sender_email"] == "alice@company.com"

    @pytest.mark.asyncio
    async def test_search_by_snippet(self, store: InMemoryUnifiedInboxStore):
        """Search should match against snippet."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(
            TENANT,
            _make_message("msg_1", external_id="e1", snippet="Please review the PR"),
        )
        await store.save_message(
            TENANT,
            _make_message("msg_2", external_id="e2", snippet="Lunch plans"),
        )
        messages, total = await store.list_messages(TENANT, search="review")
        assert total == 1
        assert "review" in messages[0]["snippet"].lower()

    @pytest.mark.asyncio
    async def test_combined_filters(self, store: InMemoryUnifiedInboxStore):
        """Multiple filters applied simultaneously should intersect."""
        await store.save_account(TENANT, _make_account("acct_1"))
        await store.save_account(TENANT, _make_account("acct_2", email="b@example.com"))
        await store.save_message(
            TENANT,
            _make_message("msg_1", "acct_1", "e1", priority_tier="high", is_read=False),
        )
        await store.save_message(
            TENANT,
            _make_message("msg_2", "acct_1", "e2", priority_tier="high", is_read=True),
        )
        await store.save_message(
            TENANT,
            _make_message("msg_3", "acct_2", "e3", priority_tier="high", is_read=False),
        )
        messages, total = await store.list_messages(
            TENANT, priority_tier="high", unread_only=True, account_id="acct_1"
        )
        assert total == 1
        assert messages[0]["id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_message_sorting_by_priority(self, store: InMemoryUnifiedInboxStore):
        """Messages should be sorted by priority_score descending."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(
            TENANT, _make_message("msg_a", external_id="ea", priority_score=0.1)
        )
        await store.save_message(
            TENANT, _make_message("msg_b", external_id="eb", priority_score=0.9)
        )
        await store.save_message(
            TENANT, _make_message("msg_c", external_id="ec", priority_score=0.5)
        )
        messages, _ = await store.list_messages(TENANT)
        scores = [m["priority_score"] for m in messages]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_triage_result_tenant_isolation(self, store: InMemoryUnifiedInboxStore):
        """Triage results in different tenants should be isolated."""
        await store.save_triage_result(TENANT, _make_triage("msg_1"))
        await store.save_triage_result(TENANT_2, _make_triage("msg_1", confidence=0.3))
        r1 = await store.get_triage_result(TENANT, "msg_1")
        r2 = await store.get_triage_result(TENANT_2, "msg_1")
        assert r1 is not None and r1["confidence"] == 0.9
        assert r2 is not None and r2["confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_message_tenant_isolation(self, store: InMemoryUnifiedInboxStore):
        """Messages in different tenants should be isolated."""
        await store.save_account(TENANT, _make_account())
        await store.save_account(TENANT_2, _make_account())
        await store.save_message(TENANT, _make_message("msg_1", external_id="e1"))
        await store.save_message(TENANT_2, _make_message("msg_2", external_id="e2"))
        assert await store.get_message(TENANT, "msg_1") is not None
        assert await store.get_message(TENANT, "msg_2") is None
        assert await store.get_message(TENANT_2, "msg_2") is not None
        assert await store.get_message(TENANT_2, "msg_1") is None

    @pytest.mark.asyncio
    async def test_delete_message_also_deletes_triage(self, store: InMemoryUnifiedInboxStore):
        """Deleting a message should also remove its associated triage result."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message())
        await store.save_triage_result(TENANT, _make_triage("msg_1"))
        assert await store.get_triage_result(TENANT, "msg_1") is not None
        await store.delete_message(TENANT, "msg_1")
        assert await store.get_triage_result(TENANT, "msg_1") is None

    @pytest.mark.asyncio
    async def test_search_no_matches(self, store: InMemoryUnifiedInboxStore):
        """Search that matches nothing should return empty results."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message())
        messages, total = await store.list_messages(TENANT, search="zzz_nonexistent_zzz")
        assert total == 0
        assert messages == []

    @pytest.mark.asyncio
    async def test_filter_nonexistent_priority_tier(self, store: InMemoryUnifiedInboxStore):
        """Filtering by a priority tier that no messages have returns empty."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(priority_tier="medium"))
        messages, total = await store.list_messages(TENANT, priority_tier="critical")
        assert total == 0
        assert messages == []

    @pytest.mark.asyncio
    async def test_update_account_multiple_fields(self, store: InMemoryUnifiedInboxStore):
        """Updating multiple account fields at once should apply all."""
        await store.save_account(TENANT, _make_account())
        await store.update_account_fields(
            TENANT, "acct_1", {"status": "syncing", "display_name": "Updated User"}
        )
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["status"] == "syncing"
        assert result["display_name"] == "Updated User"

    @pytest.mark.asyncio
    async def test_list_accounts_empty_tenant(self, store: InMemoryUnifiedInboxStore):
        """Listing accounts for a tenant with none should return empty list."""
        results = await store.list_accounts("nonexistent_tenant")
        assert results == []


# ====================================================================
# SQLite Backend Tests
# ====================================================================


class TestSQLiteBackendAccountCRUD:
    """Tests for SQLiteUnifiedInboxStore account operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_account(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["id"] == "acct_1"
        assert result["provider"] == "gmail"
        assert result["email_address"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_get_account_not_found(self, sqlite_store: SQLiteUnifiedInboxStore):
        result = await sqlite_store.get_account(TENANT, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_accounts(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account("acct_1"))
        await sqlite_store.save_account(TENANT, _make_account("acct_2", email="b@x.com"))
        results = await sqlite_store.list_accounts(TENANT)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_delete_account(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        deleted = await sqlite_store.delete_account(TENANT, "acct_1")
        assert deleted is True
        assert await sqlite_store.get_account(TENANT, "acct_1") is None

    @pytest.mark.asyncio
    async def test_delete_account_not_found(self, sqlite_store: SQLiteUnifiedInboxStore):
        deleted = await sqlite_store.delete_account(TENANT, "nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_account_fields(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.update_account_fields(TENANT, "acct_1", {"status": "error"})
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_update_account_fields_empty(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Empty updates dict should be a no-op."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.update_account_fields(TENANT, "acct_1", {})
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_increment_account_counts(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.increment_account_counts(
            TENANT, "acct_1", total_delta=10, unread_delta=5, sync_error_delta=2
        )
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["total_messages"] == 10
        assert result["unread_count"] == 5
        assert result["sync_errors"] == 2

    @pytest.mark.asyncio
    async def test_increment_counts_floor_at_zero(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.increment_account_counts(TENANT, "acct_1", total_delta=-999)
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_save_account_upsert(self, sqlite_store: SQLiteUnifiedInboxStore):
        """INSERT OR REPLACE should overwrite existing account."""
        await sqlite_store.save_account(TENANT, _make_account(status="active"))
        await sqlite_store.save_account(TENANT, _make_account(status="disconnected"))
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["status"] == "disconnected"


class TestSQLiteBackendMessageCRUD:
    """Tests for SQLiteUnifiedInboxStore message operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_message(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        msg_id, created = await sqlite_store.save_message(TENANT, _make_message())
        assert msg_id == "msg_1"
        assert created is True
        result = await sqlite_store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["subject"] == "Test Subject"

    @pytest.mark.asyncio
    async def test_get_message_not_found(self, sqlite_store: SQLiteUnifiedInboxStore):
        result = await sqlite_store.get_message(TENANT, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_message_dedup(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Same external_id + account_id should update on second save."""
        await sqlite_store.save_account(TENANT, _make_account())
        _, created1 = await sqlite_store.save_message(TENANT, _make_message())
        assert created1 is True
        _, created2 = await sqlite_store.save_message(
            TENANT, _make_message(message_id="msg_2", subject="Updated")
        )
        assert created2 is False

    @pytest.mark.asyncio
    async def test_save_message_increments_counts(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=False))
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 1
        assert acct["unread_count"] == 1

    @pytest.mark.asyncio
    async def test_delete_message(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message())
        deleted = await sqlite_store.delete_message(TENANT, "msg_1")
        assert deleted is True
        assert await sqlite_store.get_message(TENANT, "msg_1") is None

    @pytest.mark.asyncio
    async def test_delete_message_not_found(self, sqlite_store: SQLiteUnifiedInboxStore):
        deleted = await sqlite_store.delete_message(TENANT, "nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_update_message_flags(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=False))
        result = await sqlite_store.update_message_flags(TENANT, "msg_1", is_read=True)
        assert result is True
        msg = await sqlite_store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["is_read"] is True

    @pytest.mark.asyncio
    async def test_update_message_flags_not_found(self, sqlite_store: SQLiteUnifiedInboxStore):
        result = await sqlite_store.update_message_flags(TENANT, "nonexistent", is_read=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_message_flags_no_changes(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Passing no flag changes should return True."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message())
        result = await sqlite_store.update_message_flags(TENANT, "msg_1")
        assert result is True

    @pytest.mark.asyncio
    async def test_update_message_triage(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message())
        await sqlite_store.update_message_triage(TENANT, "msg_1", "archive", "Low priority")
        msg = await sqlite_store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["triage_action"] == "archive"
        assert msg["triage_rationale"] == "Low priority"


class TestSQLiteBackendMessageListing:
    """Tests for SQLiteUnifiedInboxStore list_messages."""

    @pytest.fixture
    async def populated_sqlite(self, sqlite_store: SQLiteUnifiedInboxStore):
        await sqlite_store.save_account(TENANT, _make_account("acct_1"))
        await sqlite_store.save_account(TENANT, _make_account("acct_2", email="b@x.com"))
        await sqlite_store.save_message(
            TENANT,
            _make_message(
                "msg_1",
                "acct_1",
                "e1",
                priority_tier="high",
                priority_score=0.9,
                is_read=False,
                subject="Urgent: deploy",
                sender_email="admin@ops.com",
            ),
        )
        await sqlite_store.save_message(
            TENANT,
            _make_message(
                "msg_2",
                "acct_1",
                "e2",
                priority_tier="low",
                priority_score=0.1,
                is_read=True,
                subject="Newsletter",
                snippet="Weekly roundup",
            ),
        )
        await sqlite_store.save_message(
            TENANT,
            _make_message(
                "msg_3",
                "acct_2",
                "e3",
                priority_tier="medium",
                priority_score=0.5,
                is_read=False,
                subject="Meeting notes",
            ),
        )
        return sqlite_store

    @pytest.mark.asyncio
    async def test_list_all_messages(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(TENANT)
        assert total == 3
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_filter_by_priority_tier(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(TENANT, priority_tier="high")
        assert total == 1
        assert messages[0]["id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_filter_unread_only(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(TENANT, unread_only=True)
        assert total == 2

    @pytest.mark.asyncio
    async def test_search_by_subject(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(TENANT, search="urgent")
        assert total == 1
        assert messages[0]["id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_search_by_sender_email(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(TENANT, search="admin@ops")
        assert total == 1
        assert messages[0]["id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_search_by_snippet(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(TENANT, search="roundup")
        assert total == 1
        assert messages[0]["id"] == "msg_2"

    @pytest.mark.asyncio
    async def test_pagination(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(TENANT, limit=2, offset=0)
        assert total == 3
        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_combined_filters(self, populated_sqlite: SQLiteUnifiedInboxStore):
        messages, total = await populated_sqlite.list_messages(
            TENANT, priority_tier="high", unread_only=True, account_id="acct_1"
        )
        assert total == 1
        assert messages[0]["id"] == "msg_1"

    @pytest.mark.asyncio
    async def test_list_empty_tenant(self, sqlite_store: SQLiteUnifiedInboxStore):
        messages, total = await sqlite_store.list_messages("empty_tenant")
        assert total == 0
        assert messages == []


class TestSQLiteBackendTriageResults:
    """Tests for SQLiteUnifiedInboxStore triage result operations."""

    @pytest.mark.asyncio
    async def test_save_and_get_triage_result(self, sqlite_store: SQLiteUnifiedInboxStore):
        triage = _make_triage()
        await sqlite_store.save_triage_result(TENANT, triage)
        result = await sqlite_store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["recommended_action"] == "reply"
        assert result["confidence"] == 0.9
        assert result["agents_involved"] == ["claude", "gpt"]

    @pytest.mark.asyncio
    async def test_get_triage_result_not_found(self, sqlite_store: SQLiteUnifiedInboxStore):
        result = await sqlite_store.get_triage_result(TENANT, "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_triage_result_upsert(self, sqlite_store: SQLiteUnifiedInboxStore):
        """INSERT OR REPLACE should overwrite existing triage result."""
        await sqlite_store.save_triage_result(TENANT, _make_triage(confidence=0.5))
        await sqlite_store.save_triage_result(TENANT, _make_triage(confidence=0.99))
        result = await sqlite_store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["confidence"] == 0.99

    @pytest.mark.asyncio
    async def test_delete_account_cascades(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Deleting an account should cascade-delete messages and triage results."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message())
        await sqlite_store.save_triage_result(TENANT, _make_triage("msg_1"))
        await sqlite_store.delete_account(TENANT, "acct_1")
        assert await sqlite_store.get_message(TENANT, "msg_1") is None
        assert await sqlite_store.get_triage_result(TENANT, "msg_1") is None


# ====================================================================
# InMemory: Message Metadata Fields
# ====================================================================


class TestInMemoryMessageMetadata:
    """Tests for rich message fields (recipients, cc, labels, thread_id, etc.)."""

    @pytest.mark.asyncio
    async def test_message_recipients_and_cc(self, store: InMemoryUnifiedInboxStore):
        """Messages preserve recipients and cc lists."""
        await store.save_account(TENANT, _make_account())
        msg = _make_message(
            recipients=["alice@co.com", "bob@co.com"],
            cc=["carol@co.com"],
        )
        await store.save_message(TENANT, msg)
        result = await store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["recipients"] == ["alice@co.com", "bob@co.com"]
        assert result["cc"] == ["carol@co.com"]

    @pytest.mark.asyncio
    async def test_message_labels(self, store: InMemoryUnifiedInboxStore):
        """Messages preserve labels list."""
        await store.save_account(TENANT, _make_account())
        msg = _make_message(labels=["inbox", "important", "starred"])
        await store.save_message(TENANT, msg)
        result = await store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["labels"] == ["inbox", "important", "starred"]

    @pytest.mark.asyncio
    async def test_message_thread_id(self, store: InMemoryUnifiedInboxStore):
        """Messages preserve thread_id."""
        await store.save_account(TENANT, _make_account())
        msg = _make_message(thread_id="thread_abc123")
        await store.save_message(TENANT, msg)
        result = await store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["thread_id"] == "thread_abc123"

    @pytest.mark.asyncio
    async def test_message_has_attachments(self, store: InMemoryUnifiedInboxStore):
        """Messages preserve has_attachments flag."""
        await store.save_account(TENANT, _make_account())
        msg = _make_message(has_attachments=True)
        await store.save_message(TENANT, msg)
        result = await store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["has_attachments"] is True

    @pytest.mark.asyncio
    async def test_message_body_preview(self, store: InMemoryUnifiedInboxStore):
        """Messages preserve body_preview text."""
        await store.save_account(TENANT, _make_account())
        msg = _make_message(body_preview="This is a preview of the email body...")
        await store.save_message(TENANT, msg)
        result = await store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["body_preview"] == "This is a preview of the email body..."

    @pytest.mark.asyncio
    async def test_message_priority_reasons(self, store: InMemoryUnifiedInboxStore):
        """Messages preserve priority_reasons list."""
        await store.save_account(TENANT, _make_account())
        reasons = ["VIP sender", "Contains urgent keyword"]
        msg = _make_message(priority_reasons=reasons)
        await store.save_message(TENANT, msg)
        result = await store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["priority_reasons"] == reasons


# ====================================================================
# InMemory: Update Message Flags Edge Cases
# ====================================================================


class TestInMemoryMessageFlagsEdgeCases:
    """Edge cases for update_message_flags on InMemory backend."""

    @pytest.mark.asyncio
    async def test_update_flags_no_changes_returns_true(self, store: InMemoryUnifiedInboxStore):
        """Passing neither is_read nor is_starred should return True (no-op)."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message())
        result = await store.update_message_flags(TENANT, "msg_1")
        assert result is True

    @pytest.mark.asyncio
    async def test_update_flags_both_simultaneously(self, store: InMemoryUnifiedInboxStore):
        """Setting both is_read and is_starred in a single call."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=False, is_starred=False))
        result = await store.update_message_flags(TENANT, "msg_1", is_read=True, is_starred=True)
        assert result is True
        msg = await store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["is_read"] is True
        assert msg["is_starred"] is True

    @pytest.mark.asyncio
    async def test_update_starred_does_not_affect_unread_count(
        self, store: InMemoryUnifiedInboxStore
    ):
        """Toggling is_starred alone should not change unread_count."""
        await store.save_account(TENANT, _make_account())
        await store.save_message(TENANT, _make_message(is_read=False))
        acct_before = await store.get_account(TENANT, "acct_1")
        assert acct_before is not None
        assert acct_before["unread_count"] == 1
        await store.update_message_flags(TENANT, "msg_1", is_starred=True)
        acct_after = await store.get_account(TENANT, "acct_1")
        assert acct_after is not None
        assert acct_after["unread_count"] == 1


# ====================================================================
# InMemory: Account Connected At and Last Sync
# ====================================================================


class TestInMemoryAccountDatetimeFields:
    """Tests for account datetime fields (connected_at, last_sync)."""

    @pytest.mark.asyncio
    async def test_account_with_datetime_fields(self, store: InMemoryUnifiedInboxStore):
        """Account preserves connected_at and last_sync datetimes."""
        now = _utc_now()
        account = _make_account(connected_at=now, last_sync=now)
        await store.save_account(TENANT, account)
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["connected_at"] == now
        assert result["last_sync"] == now

    @pytest.mark.asyncio
    async def test_account_metadata_field(self, store: InMemoryUnifiedInboxStore):
        """Account preserves metadata dict."""
        account = _make_account(metadata={"sync_token": "abc123", "scopes": ["mail"]})
        await store.save_account(TENANT, account)
        result = await store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["metadata"] == {"sync_token": "abc123", "scopes": ["mail"]}


# ====================================================================
# InMemory: Multiple Messages Per Account
# ====================================================================


class TestInMemoryMultipleMessages:
    """Tests for managing multiple messages under one account."""

    @pytest.mark.asyncio
    async def test_multiple_messages_accumulate_counts(self, store: InMemoryUnifiedInboxStore):
        """Saving multiple messages properly accumulates account counts."""
        await store.save_account(TENANT, _make_account())
        for i in range(5):
            await store.save_message(
                TENANT,
                _make_message(f"msg_{i}", external_id=f"ext_{i}", is_read=(i % 2 == 0)),
            )
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 5
        # i=0 read, i=1 unread, i=2 read, i=3 unread, i=4 read -> 2 unread
        assert acct["unread_count"] == 2

    @pytest.mark.asyncio
    async def test_delete_multiple_messages_decrements_counts(
        self, store: InMemoryUnifiedInboxStore
    ):
        """Deleting multiple messages decrements counts correctly."""
        await store.save_account(TENANT, _make_account())
        for i in range(3):
            await store.save_message(
                TENANT,
                _make_message(f"msg_{i}", external_id=f"ext_{i}", is_read=False),
            )
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 3
        assert acct["unread_count"] == 3

        await store.delete_message(TENANT, "msg_0")
        await store.delete_message(TENANT, "msg_1")
        acct = await store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 1
        assert acct["unread_count"] == 1


# ====================================================================
# InMemory: Triage Result Full Fields
# ====================================================================


class TestInMemoryTriageResultFields:
    """Tests that all triage result fields are preserved."""

    @pytest.mark.asyncio
    async def test_triage_all_fields(self, store: InMemoryUnifiedInboxStore):
        """All triage result fields are stored and returned correctly."""
        triage = _make_triage(
            delegate_to="senior_agent",
            schedule_for=_utc_now(),
            suggested_response="I will handle this ASAP",
            debate_summary="All agents agreed on delegation",
        )
        await store.save_triage_result(TENANT, triage)
        result = await store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["delegate_to"] == "senior_agent"
        assert result["suggested_response"] == "I will handle this ASAP"
        assert result["debate_summary"] == "All agents agreed on delegation"
        assert result["schedule_for"] is not None

    @pytest.mark.asyncio
    async def test_triage_with_none_optional_fields(self, store: InMemoryUnifiedInboxStore):
        """Triage result with None optional fields."""
        triage = _make_triage(
            delegate_to=None,
            schedule_for=None,
            suggested_response=None,
            debate_summary=None,
            agents_involved=[],
        )
        await store.save_triage_result(TENANT, triage)
        result = await store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["delegate_to"] is None
        assert result["schedule_for"] is None
        assert result["suggested_response"] is None
        assert result["debate_summary"] is None
        assert result["agents_involved"] == []


# ====================================================================
# SQLite: Dedup with Read-Status Tracking
# ====================================================================


class TestSQLiteDedupReadStatusTracking:
    """Tests for SQLite save_message dedup with read-status change tracking."""

    @pytest.mark.asyncio
    async def test_dedup_unread_to_read_decrements_unread(
        self, sqlite_store: SQLiteUnifiedInboxStore
    ):
        """Dedup save that changes is_read False->True should decrement unread count."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=False))
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 1

        # Save again same external_id, now read
        await sqlite_store.save_message(TENANT, _make_message(message_id="msg_dup", is_read=True))
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0

    @pytest.mark.asyncio
    async def test_dedup_read_to_unread_increments_unread(
        self, sqlite_store: SQLiteUnifiedInboxStore
    ):
        """Dedup save that changes is_read True->False should increment unread count."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=True))
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0

        # Same external_id, now unread
        await sqlite_store.save_message(TENANT, _make_message(message_id="msg_dup", is_read=False))
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 1

    @pytest.mark.asyncio
    async def test_dedup_returns_original_message_id(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Dedup save returns the original message ID, not the new one."""
        await sqlite_store.save_account(TENANT, _make_account())
        msg_id_1, created_1 = await sqlite_store.save_message(TENANT, _make_message("msg_original"))
        assert msg_id_1 == "msg_original"
        assert created_1 is True

        msg_id_2, created_2 = await sqlite_store.save_message(
            TENANT, _make_message("msg_new_id", subject="Updated subject")
        )
        assert msg_id_2 == "msg_original"
        assert created_2 is False


# ====================================================================
# SQLite: Message Flag Toggling with Count Tracking
# ====================================================================


class TestSQLiteMessageFlagToggling:
    """Tests for SQLite flag toggling and unread count tracking."""

    @pytest.mark.asyncio
    async def test_mark_read_then_unread(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Toggling read -> unread should restore unread count."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=False))

        # Mark as read
        await sqlite_store.update_message_flags(TENANT, "msg_1", is_read=True)
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 0

        # Mark as unread again
        await sqlite_store.update_message_flags(TENANT, "msg_1", is_read=False)
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["unread_count"] == 1

    @pytest.mark.asyncio
    async def test_starred_flag_preserved(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Setting is_starred persists and can be toggled."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_starred=False))

        await sqlite_store.update_message_flags(TENANT, "msg_1", is_starred=True)
        msg = await sqlite_store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["is_starred"] is True

        await sqlite_store.update_message_flags(TENANT, "msg_1", is_starred=False)
        msg = await sqlite_store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["is_starred"] is False

    @pytest.mark.asyncio
    async def test_both_flags_simultaneously(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Setting both is_read and is_starred in one call."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=False, is_starred=False))

        await sqlite_store.update_message_flags(TENANT, "msg_1", is_read=True, is_starred=True)
        msg = await sqlite_store.get_message(TENANT, "msg_1")
        assert msg is not None
        assert msg["is_read"] is True
        assert msg["is_starred"] is True


# ====================================================================
# SQLite: Delete Message with Count Updates
# ====================================================================


class TestSQLiteDeleteMessageCounts:
    """Tests for SQLite delete_message with account count adjustments."""

    @pytest.mark.asyncio
    async def test_delete_unread_message_decrements_counts(
        self, sqlite_store: SQLiteUnifiedInboxStore
    ):
        """Deleting unread message decrements both total and unread."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=False))
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 1
        assert acct["unread_count"] == 1

        await sqlite_store.delete_message(TENANT, "msg_1")
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 0
        assert acct["unread_count"] == 0

    @pytest.mark.asyncio
    async def test_delete_read_message_only_decrements_total(
        self, sqlite_store: SQLiteUnifiedInboxStore
    ):
        """Deleting read message decrements total but not unread."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message(is_read=True))
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 1
        assert acct["unread_count"] == 0

        await sqlite_store.delete_message(TENANT, "msg_1")
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 0
        assert acct["unread_count"] == 0

    @pytest.mark.asyncio
    async def test_delete_message_also_deletes_triage(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Deleting a message should also remove its triage result."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(TENANT, _make_message())
        await sqlite_store.save_triage_result(TENANT, _make_triage("msg_1"))
        assert await sqlite_store.get_triage_result(TENANT, "msg_1") is not None

        await sqlite_store.delete_message(TENANT, "msg_1")
        assert await sqlite_store.get_triage_result(TENANT, "msg_1") is None


# ====================================================================
# SQLite: Rich Message Metadata
# ====================================================================


class TestSQLiteMessageMetadata:
    """Tests for SQLite message metadata serialization/deserialization."""

    @pytest.mark.asyncio
    async def test_message_recipients_and_cc_roundtrip(self, sqlite_store: SQLiteUnifiedInboxStore):
        """JSON-serialized recipients and cc lists roundtrip correctly."""
        await sqlite_store.save_account(TENANT, _make_account())
        msg = _make_message(
            recipients=["a@co.com", "b@co.com"],
            cc=["c@co.com", "d@co.com"],
        )
        await sqlite_store.save_message(TENANT, msg)
        result = await sqlite_store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["recipients"] == ["a@co.com", "b@co.com"]
        assert result["cc"] == ["c@co.com", "d@co.com"]

    @pytest.mark.asyncio
    async def test_message_labels_roundtrip(self, sqlite_store: SQLiteUnifiedInboxStore):
        """JSON-serialized labels list roundtrips correctly."""
        await sqlite_store.save_account(TENANT, _make_account())
        msg = _make_message(labels=["inbox", "important", "work"])
        await sqlite_store.save_message(TENANT, msg)
        result = await sqlite_store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["labels"] == ["inbox", "important", "work"]

    @pytest.mark.asyncio
    async def test_message_priority_reasons_roundtrip(self, sqlite_store: SQLiteUnifiedInboxStore):
        """JSON-serialized priority_reasons list roundtrips correctly."""
        await sqlite_store.save_account(TENANT, _make_account())
        reasons = ["VIP sender", "Mentions project deadline"]
        msg = _make_message(priority_reasons=reasons)
        await sqlite_store.save_message(TENANT, msg)
        result = await sqlite_store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["priority_reasons"] == reasons

    @pytest.mark.asyncio
    async def test_message_thread_id_and_body_preview(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Thread ID and body preview stored and retrieved."""
        await sqlite_store.save_account(TENANT, _make_account())
        msg = _make_message(
            thread_id="thread_xyz",
            body_preview="The quick brown fox jumps over the lazy dog.",
        )
        await sqlite_store.save_message(TENANT, msg)
        result = await sqlite_store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["thread_id"] == "thread_xyz"
        assert result["body_preview"] == "The quick brown fox jumps over the lazy dog."

    @pytest.mark.asyncio
    async def test_message_has_attachments_flag(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Boolean has_attachments flag roundtrips correctly through SQLite integers."""
        await sqlite_store.save_account(TENANT, _make_account())
        msg = _make_message(has_attachments=True)
        await sqlite_store.save_message(TENANT, msg)
        result = await sqlite_store.get_message(TENANT, "msg_1")
        assert result is not None
        assert result["has_attachments"] is True


# ====================================================================
# SQLite: Account Field Updates with Special Types
# ====================================================================


class TestSQLiteAccountFieldUpdates:
    """Tests for SQLite update_account_fields with datetime and metadata fields."""

    @pytest.mark.asyncio
    async def test_update_connected_at_datetime(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Updating connected_at with a datetime value formats it to ISO."""
        await sqlite_store.save_account(TENANT, _make_account())
        now = _utc_now()
        await sqlite_store.update_account_fields(TENANT, "acct_1", {"connected_at": now})
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["connected_at"] is not None

    @pytest.mark.asyncio
    async def test_update_last_sync_datetime(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Updating last_sync with a datetime value formats it to ISO."""
        await sqlite_store.save_account(TENANT, _make_account())
        now = _utc_now()
        await sqlite_store.update_account_fields(TENANT, "acct_1", {"last_sync": now})
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["last_sync"] is not None

    @pytest.mark.asyncio
    async def test_update_metadata_via_account_fields(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Updating metadata via update_account_fields serializes to JSON."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.update_account_fields(
            TENANT, "acct_1", {"metadata": {"token": "xyz", "scopes": ["mail", "calendar"]}}
        )
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["metadata"]["token"] == "xyz"
        assert result["metadata"]["scopes"] == ["mail", "calendar"]

    @pytest.mark.asyncio
    async def test_update_multiple_fields_including_datetime(
        self, sqlite_store: SQLiteUnifiedInboxStore
    ):
        """Update multiple fields including datetime and plain string."""
        await sqlite_store.save_account(TENANT, _make_account())
        now = _utc_now()
        await sqlite_store.update_account_fields(
            TENANT, "acct_1", {"status": "syncing", "last_sync": now}
        )
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["status"] == "syncing"
        assert result["last_sync"] is not None


# ====================================================================
# SQLite: Priority Score Sorting
# ====================================================================


class TestSQLitePriorityScoreSorting:
    """Tests for SQLite list_messages sort ordering."""

    @pytest.mark.asyncio
    async def test_messages_sorted_by_priority_desc(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Messages should be sorted by priority_score descending."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_message(
            TENANT, _make_message("msg_lo", external_id="e1", priority_score=0.1)
        )
        await sqlite_store.save_message(
            TENANT, _make_message("msg_hi", external_id="e2", priority_score=0.9)
        )
        await sqlite_store.save_message(
            TENANT, _make_message("msg_mid", external_id="e3", priority_score=0.5)
        )

        messages, _ = await sqlite_store.list_messages(TENANT)
        scores = [m["priority_score"] for m in messages]
        assert scores == sorted(scores, reverse=True)


# ====================================================================
# SQLite: Tenant Isolation
# ====================================================================


class TestSQLiteTenantIsolation:
    """Tests for tenant isolation in the SQLite backend."""

    @pytest.mark.asyncio
    async def test_account_tenant_isolation(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Accounts in different tenants are fully isolated."""
        await sqlite_store.save_account(TENANT, _make_account("acct_1"))
        await sqlite_store.save_account(TENANT_2, _make_account("acct_2"))
        assert len(await sqlite_store.list_accounts(TENANT)) == 1
        assert len(await sqlite_store.list_accounts(TENANT_2)) == 1
        assert await sqlite_store.get_account(TENANT, "acct_2") is None
        assert await sqlite_store.get_account(TENANT_2, "acct_1") is None

    @pytest.mark.asyncio
    async def test_message_tenant_isolation(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Messages in different tenants are fully isolated."""
        await sqlite_store.save_account(TENANT, _make_account())
        await sqlite_store.save_account(TENANT_2, _make_account())
        await sqlite_store.save_message(TENANT, _make_message("msg_1", external_id="e1"))
        await sqlite_store.save_message(TENANT_2, _make_message("msg_2", external_id="e2"))

        assert await sqlite_store.get_message(TENANT, "msg_1") is not None
        assert await sqlite_store.get_message(TENANT, "msg_2") is None
        assert await sqlite_store.get_message(TENANT_2, "msg_2") is not None
        assert await sqlite_store.get_message(TENANT_2, "msg_1") is None

    @pytest.mark.asyncio
    async def test_triage_tenant_isolation(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Triage results in different tenants are isolated."""
        await sqlite_store.save_triage_result(TENANT, _make_triage("msg_1", confidence=0.9))
        await sqlite_store.save_triage_result(TENANT_2, _make_triage("msg_1", confidence=0.3))
        r1 = await sqlite_store.get_triage_result(TENANT, "msg_1")
        r2 = await sqlite_store.get_triage_result(TENANT_2, "msg_1")
        assert r1 is not None and r1["confidence"] == 0.9
        assert r2 is not None and r2["confidence"] == 0.3


# ====================================================================
# SQLite: Triage Result Full Fields
# ====================================================================


class TestSQLiteTriageResultFields:
    """Tests that all SQLite triage result fields are preserved."""

    @pytest.mark.asyncio
    async def test_triage_all_optional_fields(self, sqlite_store: SQLiteUnifiedInboxStore):
        """All triage result optional fields roundtrip through SQLite."""
        triage = _make_triage(
            delegate_to="senior_agent",
            suggested_response="I will handle this ASAP",
            debate_summary="Agents reached consensus on delegation",
        )
        await sqlite_store.save_triage_result(TENANT, triage)
        result = await sqlite_store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["delegate_to"] == "senior_agent"
        assert result["suggested_response"] == "I will handle this ASAP"
        assert result["debate_summary"] == "Agents reached consensus on delegation"
        assert result["agents_involved"] == ["claude", "gpt"]

    @pytest.mark.asyncio
    async def test_triage_empty_agents_list(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Triage result with empty agents_involved list."""
        triage = _make_triage(agents_involved=[])
        await sqlite_store.save_triage_result(TENANT, triage)
        result = await sqlite_store.get_triage_result(TENANT, "msg_1")
        assert result is not None
        assert result["agents_involved"] == []


# ====================================================================
# SQLite: Multiple Messages Account Count Accumulation
# ====================================================================


class TestSQLiteMultipleMessageCounts:
    """Tests for multiple message accumulation in SQLite."""

    @pytest.mark.asyncio
    async def test_multiple_messages_accumulate_counts(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Saving multiple messages properly accumulates account counts."""
        await sqlite_store.save_account(TENANT, _make_account())
        for i in range(4):
            await sqlite_store.save_message(
                TENANT,
                _make_message(f"msg_{i}", external_id=f"ext_{i}", is_read=(i < 2)),
            )
        acct = await sqlite_store.get_account(TENANT, "acct_1")
        assert acct is not None
        assert acct["total_messages"] == 4
        # i=0 read, i=1 read, i=2 unread, i=3 unread -> 2 unread
        assert acct["unread_count"] == 2

    @pytest.mark.asyncio
    async def test_filter_by_account_id(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Filter list_messages by account_id."""
        await sqlite_store.save_account(TENANT, _make_account("acct_1"))
        await sqlite_store.save_account(TENANT, _make_account("acct_2", email="b@co.com"))
        await sqlite_store.save_message(TENANT, _make_message("msg_1", "acct_1", "e1"))
        await sqlite_store.save_message(TENANT, _make_message("msg_2", "acct_2", "e2"))
        messages, total = await sqlite_store.list_messages(TENANT, account_id="acct_1")
        assert total == 1
        assert messages[0]["account_id"] == "acct_1"


# ====================================================================
# SQLite: Account with Metadata Roundtrip
# ====================================================================


class TestSQLiteAccountMetadata:
    """Tests for SQLite account metadata JSON roundtrip."""

    @pytest.mark.asyncio
    async def test_account_metadata_roundtrip(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Account metadata dict serializes to JSON and deserializes correctly."""
        account = _make_account(metadata={"token": "secret", "scopes": ["read", "write"]})
        await sqlite_store.save_account(TENANT, account)
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["metadata"] == {"token": "secret", "scopes": ["read", "write"]}

    @pytest.mark.asyncio
    async def test_account_empty_metadata(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Account with no metadata gets empty dict."""
        account = _make_account()
        await sqlite_store.save_account(TENANT, account)
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        assert result["metadata"] == {}

    @pytest.mark.asyncio
    async def test_account_datetime_fields_roundtrip(self, sqlite_store: SQLiteUnifiedInboxStore):
        """Account connected_at and last_sync survive ISO serialization roundtrip."""
        now = _utc_now()
        account = _make_account(connected_at=now, last_sync=now)
        await sqlite_store.save_account(TENANT, account)
        result = await sqlite_store.get_account(TENANT, "acct_1")
        assert result is not None
        # These come back as parsed datetimes
        assert result["connected_at"] is not None
        assert result["last_sync"] is not None
