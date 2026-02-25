"""Tests for the EmailAdapter Knowledge Mound adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.email_adapter import (
    EmailAdapter,
    EmailRecord,
    EmailSearchResult,
    sanitize_pii,
)


@pytest.fixture
def adapter() -> EmailAdapter:
    return EmailAdapter()


@pytest.fixture
def sample_record() -> EmailRecord:
    return EmailRecord(
        email_id="msg-001",
        thread_id="thread-abc",
        subject="Q4 Quarterly Report Review",
        sender="alice@example.com",
        body="Hi team, please review the quarterly numbers. Call me at 555-123-4567.",
        date=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        labels=["INBOX", "IMPORTANT"],
        importance_score=0.8,
    )


# ---------------------------------------------------------------------------
# PII sanitization tests
# ---------------------------------------------------------------------------


class TestSanitizePII:
    def test_strips_email_addresses(self) -> None:
        text = "Contact john@example.com for details"
        result = sanitize_pii(text)
        assert "john@example.com" not in result
        assert "[EMAIL]" in result

    def test_strips_phone_numbers(self) -> None:
        text = "Call me at 555-123-4567 or (800) 555-1234"
        result = sanitize_pii(text)
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_strips_ssn_patterns(self) -> None:
        text = "SSN: 123-45-6789"
        result = sanitize_pii(text)
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_handles_multiple_pii_types(self) -> None:
        text = "Email bob@test.org, call 555-000-1234, SSN 999-88-7777"
        result = sanitize_pii(text)
        assert "bob@test.org" not in result
        assert "555-000-1234" not in result
        assert "999-88-7777" not in result

    def test_preserves_non_pii_text(self) -> None:
        text = "The quarterly report shows 15% growth"
        result = sanitize_pii(text)
        assert result == text

    def test_empty_string(self) -> None:
        assert sanitize_pii("") == ""


# ---------------------------------------------------------------------------
# Adapter init and store tests
# ---------------------------------------------------------------------------


class TestEmailAdapterInit:
    def test_init_defaults(self, adapter: EmailAdapter) -> None:
        assert adapter.adapter_name == "email"
        assert adapter.source_type == "external"
        assert adapter._pending_emails == []
        assert adapter._synced_emails == {}

    def test_init_with_callback(self) -> None:
        callback = MagicMock()
        adapter = EmailAdapter(event_callback=callback)
        assert adapter._event_callback == callback


class TestStoreEmail:
    def test_store_adds_to_pending(self, adapter: EmailAdapter, sample_record: EmailRecord) -> None:
        adapter.store_email(sample_record)
        assert len(adapter._pending_emails) == 1
        assert adapter._pending_emails[0].metadata["km_sync_pending"] is True

    def test_store_emits_event(self, sample_record: EmailRecord) -> None:
        callback = MagicMock()
        adapter = EmailAdapter(event_callback=callback)
        adapter.store_email(sample_record)

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert data["email_id"] == "msg-001"
        assert data["thread_id"] == "thread-abc"

    def test_store_from_email_message(self, adapter: EmailAdapter) -> None:
        mock_msg = MagicMock()
        mock_msg.id = "msg-auto"
        mock_msg.thread_id = "thread-auto"
        mock_msg.subject = "Auto subject"
        mock_msg.from_address = "sender@test.com"
        mock_msg.body_text = "Auto body"
        mock_msg.date = datetime(2026, 2, 1, tzinfo=timezone.utc)
        mock_msg.labels = ["INBOX"]
        mock_msg.importance_score = 0.5

        adapter.store_email(mock_msg)
        assert len(adapter._pending_emails) == 1
        assert adapter._pending_emails[0].email_id == "msg-auto"


# ---------------------------------------------------------------------------
# Sync and search tests
# ---------------------------------------------------------------------------


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_to_km_success(
        self, adapter: EmailAdapter, sample_record: EmailRecord
    ) -> None:
        adapter.store_email(sample_record)

        mock_mound = AsyncMock()
        result = await adapter.sync_to_km(mock_mound)

        assert result.records_synced == 1
        assert result.records_failed == 0
        assert len(adapter._pending_emails) == 0
        assert "msg-001" in adapter._synced_emails

    @pytest.mark.asyncio
    async def test_sync_to_km_skips_low_importance(self, adapter: EmailAdapter) -> None:
        record = EmailRecord(
            email_id="msg-low",
            thread_id="t1",
            subject="Low priority",
            sender="x@y.com",
            body="Not important",
            date=datetime.now(timezone.utc),
            importance_score=0.1,
        )
        adapter.store_email(record)

        mock_mound = AsyncMock()
        result = await adapter.sync_to_km(mock_mound, min_confidence=0.5)

        assert result.records_skipped == 1
        assert result.records_synced == 0

    @pytest.mark.asyncio
    async def test_sync_to_km_handles_error(
        self, adapter: EmailAdapter, sample_record: EmailRecord
    ) -> None:
        adapter.store_email(sample_record)

        mock_mound = MagicMock()
        mock_mound.store_item = AsyncMock(side_effect=RuntimeError("Storage error"))
        result = await adapter.sync_to_km(mock_mound)

        assert result.records_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_sync_sanitizes_pii_in_knowledge_item(self, adapter: EmailAdapter) -> None:
        record = EmailRecord(
            email_id="msg-pii",
            thread_id="t-pii",
            subject="Contact Info",
            sender="secret@company.com",
            body="Reach me at secret@company.com or 555-000-1111, SSN 123-45-6789",
            date=datetime.now(timezone.utc),
            importance_score=0.7,
        )
        adapter.store_email(record)

        mock_mound = AsyncMock()
        await adapter.sync_to_km(mock_mound)

        # Verify the stored item had PII sanitized
        stored_item = mock_mound.store_item.call_args[0][0]
        assert "secret@company.com" not in stored_item.content
        assert "555-000-1111" not in stored_item.content
        assert "123-45-6789" not in stored_item.content
        assert "[EMAIL]" in stored_item.content
        assert "[PHONE]" in stored_item.content
        assert "[SSN]" in stored_item.content


class TestSearchByTopic:
    @pytest.mark.asyncio
    async def test_search_finds_matching_subject(
        self, adapter: EmailAdapter, sample_record: EmailRecord
    ) -> None:
        adapter.store_email(sample_record)

        results = await adapter.search_by_topic("quarterly report")
        assert len(results) >= 1
        assert results[0].email_id == "msg-001"

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(
        self, adapter: EmailAdapter, sample_record: EmailRecord
    ) -> None:
        adapter.store_email(sample_record)

        results = await adapter.search_by_topic("blockchain consensus")
        assert len(results) == 0


class TestGetStats:
    def test_stats_empty(self, adapter: EmailAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_synced"] == 0
        assert stats["pending_sync"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_sync(
        self, adapter: EmailAdapter, sample_record: EmailRecord
    ) -> None:
        adapter.store_email(sample_record)
        mock_mound = AsyncMock()
        await adapter.sync_to_km(mock_mound)

        stats = adapter.get_stats()
        assert stats["total_synced"] == 1
        assert stats["unique_threads"] == 1


class TestToKnowledgeItem:
    def test_knowledge_item_has_correct_source(
        self, adapter: EmailAdapter, sample_record: EmailRecord
    ) -> None:
        item = adapter.to_knowledge_item(sample_record)
        assert item.id == "eml_msg-001"
        assert item.source.value == "external"
        assert item.metadata["subcategory"] == "email"
        assert item.metadata["thread_id"] == "thread-abc"

    def test_knowledge_item_sanitizes_sender(
        self, adapter: EmailAdapter, sample_record: EmailRecord
    ) -> None:
        item = adapter.to_knowledge_item(sample_record)
        assert "alice@example.com" not in item.metadata["sender"]
        assert "[EMAIL]" in item.metadata["sender"]
