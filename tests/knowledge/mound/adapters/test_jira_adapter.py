"""Tests for the JiraAdapter Knowledge Mound adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.jira_adapter import (
    JiraAdapter,
    JiraTicketRecord,
    JiraSearchResult,
)


@pytest.fixture
def adapter() -> JiraAdapter:
    return JiraAdapter()


@pytest.fixture
def sample_ticket() -> JiraTicketRecord:
    return JiraTicketRecord(
        ticket_id="10001",
        key="PROJ-123",
        project_key="PROJ",
        summary="Fix authentication timeout on login page",
        description="Users report intermittent timeouts when logging in via SSO.",
        issue_type="Bug",
        status="In Progress",
        priority="High",
        assignee="alice",
        reporter="bob",
        labels=["auth", "sso", "production"],
        components=["backend", "auth-service"],
        url="https://jira.example.com/browse/PROJ-123",
    )


# ---------------------------------------------------------------------------
# Init and store tests
# ---------------------------------------------------------------------------


class TestJiraAdapterInit:
    def test_init_defaults(self, adapter: JiraAdapter) -> None:
        assert adapter.adapter_name == "jira"
        assert adapter.source_type == "external"
        assert adapter._pending_tickets == []
        assert adapter._synced_tickets == {}

    def test_init_with_callback(self) -> None:
        callback = MagicMock()
        adapter = JiraAdapter(event_callback=callback)
        assert adapter._event_callback == callback


class TestStoreTicket:
    def test_store_adds_to_pending(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        adapter.store_ticket(sample_ticket)
        assert len(adapter._pending_tickets) == 1
        assert adapter._pending_tickets[0].metadata["km_sync_pending"] is True

    def test_store_emits_event(self, sample_ticket: JiraTicketRecord) -> None:
        callback = MagicMock()
        adapter = JiraAdapter(event_callback=callback)
        adapter.store_ticket(sample_ticket)

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert data["key"] == "PROJ-123"
        assert data["status"] == "In Progress"

    def test_store_from_jira_issue(self, adapter: JiraAdapter) -> None:
        mock_issue = MagicMock()
        mock_issue.id = "20001"
        mock_issue.key = "DEV-456"
        mock_issue.project_key = "DEV"
        mock_issue.summary = "Implement rate limiting"
        mock_issue.description = "Add rate limiting to API endpoints"
        mock_issue.issue_type = "Story"
        mock_issue.status = "To Do"
        mock_issue.priority = "Medium"
        mock_issue.assignee = "charlie"
        mock_issue.reporter = "dave"
        mock_issue.labels = ["api"]
        mock_issue.components = ["gateway"]
        mock_issue.url = "https://jira.example.com/browse/DEV-456"
        mock_issue.created_at = datetime(2026, 1, 10, tzinfo=timezone.utc)
        mock_issue.updated_at = datetime(2026, 1, 12, tzinfo=timezone.utc)

        adapter.store_ticket(mock_issue)
        assert len(adapter._pending_tickets) == 1
        assert adapter._pending_tickets[0].key == "DEV-456"


class TestJiraTicketRecord:
    def test_from_jira_issue_with_missing_fields(self) -> None:
        """Gracefully handles Jira issues with missing optional fields."""
        mock_issue = MagicMock(spec=[])
        # Only set minimal attributes
        mock_issue.id = "30001"
        mock_issue.key = "MIN-1"
        mock_issue.project_key = "MIN"
        mock_issue.summary = "Minimal ticket"

        record = JiraTicketRecord.from_jira_issue(mock_issue)
        assert record.ticket_id == "30001"
        assert record.summary == "Minimal ticket"
        assert record.description == ""
        assert record.labels == []


# ---------------------------------------------------------------------------
# Sync and search tests
# ---------------------------------------------------------------------------


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_to_km_success(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        adapter.store_ticket(sample_ticket)

        mock_mound = AsyncMock()
        result = await adapter.sync_to_km(mock_mound)

        assert result.records_synced == 1
        assert result.records_failed == 0
        assert len(adapter._pending_tickets) == 0
        assert "10001" in adapter._synced_tickets

    @pytest.mark.asyncio
    async def test_sync_to_km_handles_error(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        adapter.store_ticket(sample_ticket)

        mock_mound = MagicMock()
        mock_mound.store_item = AsyncMock(side_effect=RuntimeError("Store error"))
        result = await adapter.sync_to_km(mock_mound)

        assert result.records_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_sync_emits_complete_event(
        self, sample_ticket: JiraTicketRecord
    ) -> None:
        callback = MagicMock()
        adapter = JiraAdapter(event_callback=callback)
        adapter.store_ticket(sample_ticket)

        mock_mound = AsyncMock()
        await adapter.sync_to_km(mock_mound)

        # First call: forward_sync event from store_ticket
        # Second call: forward_sync_complete event from sync_to_km
        assert callback.call_count == 2
        complete_call = callback.call_args_list[1]
        event_type, data = complete_call[0]
        assert event_type == "km_adapter_forward_sync_complete"
        assert data["ticket_id"] == "10001"


class TestSearchByTopic:
    @pytest.mark.asyncio
    async def test_search_finds_matching_summary(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        adapter.store_ticket(sample_ticket)

        results = await adapter.search_by_topic("authentication timeout")
        assert len(results) >= 1
        assert results[0].key == "PROJ-123"

    @pytest.mark.asyncio
    async def test_search_with_status_filter(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        adapter.store_ticket(sample_ticket)

        # Should find with matching status
        results = await adapter.search_by_topic(
            "authentication", status_filter="In Progress"
        )
        assert len(results) >= 1

        # Should not find with non-matching status
        results = await adapter.search_by_topic("authentication", status_filter="Done")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        adapter.store_ticket(sample_ticket)

        results = await adapter.search_by_topic("quantum computing")
        assert len(results) == 0


class TestToKnowledgeItem:
    def test_knowledge_item_has_correct_source(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        item = adapter.to_knowledge_item(sample_ticket)
        assert item.id == "jira_10001"
        assert item.source.value == "external"
        assert item.metadata["subcategory"] == "jira"
        assert item.metadata["key"] == "PROJ-123"
        assert item.metadata["status"] == "In Progress"

    def test_knowledge_item_content_includes_key(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        item = adapter.to_knowledge_item(sample_ticket)
        assert "[PROJ-123]" in item.content
        assert "Fix authentication timeout" in item.content

    def test_priority_maps_to_confidence(
        self, adapter: JiraAdapter
    ) -> None:
        high = JiraTicketRecord(
            ticket_id="1", key="X-1", project_key="X",
            summary="High", priority="High",
        )
        low = JiraTicketRecord(
            ticket_id="2", key="X-2", project_key="X",
            summary="Low", priority="Low",
        )
        high_item = adapter.to_knowledge_item(high)
        low_item = adapter.to_knowledge_item(low)

        # High priority should map to higher confidence than low
        assert high_item.confidence.value != low_item.confidence.value


class TestGetStats:
    def test_stats_empty(self, adapter: JiraAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_synced"] == 0
        assert stats["pending_sync"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_sync(
        self, adapter: JiraAdapter, sample_ticket: JiraTicketRecord
    ) -> None:
        adapter.store_ticket(sample_ticket)
        mock_mound = AsyncMock()
        await adapter.sync_to_km(mock_mound)

        stats = adapter.get_stats()
        assert stats["total_synced"] == 1
        assert "PROJ" in stats["projects"]
        assert stats["status_distribution"]["In Progress"] == 1
