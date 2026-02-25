"""Tests for the ConfluenceAdapter Knowledge Mound adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.knowledge.mound.adapters.confluence_adapter import (
    ConfluenceAdapter,
    ConfluencePageRecord,
    ConfluenceSearchResult,
)


@pytest.fixture
def adapter() -> ConfluenceAdapter:
    return ConfluenceAdapter()


@pytest.fixture
def sample_page() -> ConfluencePageRecord:
    return ConfluencePageRecord(
        page_id="pg-001",
        title="Deployment Runbook for Production Services",
        space_key="OPS",
        body="Step 1: Verify health checks. Step 2: Deploy canary. Step 3: Full rollout.",
        version=3,
        url="https://wiki.example.com/spaces/OPS/pages/pg-001",
        created_by="alice",
        updated_by="bob",
        labels=["runbook", "production", "deployment"],
        parent_id="pg-root",
        created_at=datetime(2025, 12, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 20, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Init and store tests
# ---------------------------------------------------------------------------


class TestConfluenceAdapterInit:
    def test_init_defaults(self, adapter: ConfluenceAdapter) -> None:
        assert adapter.adapter_name == "confluence"
        assert adapter.source_type == "external"
        assert adapter._pending_pages == []
        assert adapter._synced_pages == {}

    def test_init_with_callback(self) -> None:
        callback = MagicMock()
        adapter = ConfluenceAdapter(event_callback=callback)
        assert adapter._event_callback == callback


class TestStorePage:
    def test_store_adds_to_pending(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        adapter.store_page(sample_page)
        assert len(adapter._pending_pages) == 1
        assert adapter._pending_pages[0].metadata["km_sync_pending"] is True

    def test_store_emits_event(self, sample_page: ConfluencePageRecord) -> None:
        callback = MagicMock()
        adapter = ConfluenceAdapter(event_callback=callback)
        adapter.store_page(sample_page)

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "km_adapter_forward_sync"
        assert data["page_id"] == "pg-001"
        assert data["space_key"] == "OPS"

    def test_store_from_confluence_page(self, adapter: ConfluenceAdapter) -> None:
        mock_page = MagicMock()
        mock_page.id = "pg-auto"
        mock_page.title = "Architecture Decision Records"
        mock_page.space_key = "ENG"
        mock_page.body = "ADR content here"
        mock_page.version = 1
        mock_page.url = "https://wiki.example.com/spaces/ENG/pages/pg-auto"
        mock_page.created_by = "charlie"
        mock_page.updated_by = "charlie"
        mock_page.labels = ["adr", "architecture"]
        mock_page.parent_id = None
        mock_page.created_at = datetime(2026, 2, 1, tzinfo=timezone.utc)
        mock_page.updated_at = datetime(2026, 2, 1, tzinfo=timezone.utc)

        adapter.store_page(mock_page)
        assert len(adapter._pending_pages) == 1
        assert adapter._pending_pages[0].title == "Architecture Decision Records"


class TestConfluencePageRecord:
    def test_from_confluence_page_with_missing_fields(self) -> None:
        """Gracefully handles Confluence pages with missing optional fields."""
        mock_page = MagicMock(spec=[])
        mock_page.id = "pg-min"
        mock_page.title = "Minimal Page"
        mock_page.space_key = "MIN"

        record = ConfluencePageRecord.from_confluence_page(mock_page)
        assert record.page_id == "pg-min"
        assert record.title == "Minimal Page"
        assert record.body == ""
        assert record.labels == []


# ---------------------------------------------------------------------------
# Sync and search tests
# ---------------------------------------------------------------------------


class TestSyncToKM:
    @pytest.mark.asyncio
    async def test_sync_to_km_success(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        adapter.store_page(sample_page)

        mock_mound = AsyncMock()
        result = await adapter.sync_to_km(mock_mound)

        assert result.records_synced == 1
        assert result.records_failed == 0
        assert len(adapter._pending_pages) == 0
        assert "pg-001" in adapter._synced_pages

    @pytest.mark.asyncio
    async def test_sync_to_km_handles_error(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        adapter.store_page(sample_page)

        mock_mound = MagicMock()
        mock_mound.store_item = AsyncMock(side_effect=RuntimeError("Store error"))
        result = await adapter.sync_to_km(mock_mound)

        assert result.records_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_sync_emits_complete_event(
        self, sample_page: ConfluencePageRecord
    ) -> None:
        callback = MagicMock()
        adapter = ConfluenceAdapter(event_callback=callback)
        adapter.store_page(sample_page)

        mock_mound = AsyncMock()
        await adapter.sync_to_km(mock_mound)

        assert callback.call_count == 2
        complete_call = callback.call_args_list[1]
        event_type, data = complete_call[0]
        assert event_type == "km_adapter_forward_sync_complete"
        assert data["page_id"] == "pg-001"


class TestSearchByTopic:
    @pytest.mark.asyncio
    async def test_search_finds_matching_title(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        adapter.store_page(sample_page)

        results = await adapter.search_by_topic("deployment runbook")
        assert len(results) >= 1
        assert results[0].page_id == "pg-001"

    @pytest.mark.asyncio
    async def test_search_with_space_filter(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        adapter.store_page(sample_page)

        # Should find with matching space
        results = await adapter.search_by_topic("deployment", space_filter="OPS")
        assert len(results) >= 1

        # Should not find with non-matching space
        results = await adapter.search_by_topic("deployment", space_filter="ENG")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        adapter.store_page(sample_page)

        results = await adapter.search_by_topic("quantum computing")
        assert len(results) == 0


class TestToKnowledgeItem:
    def test_knowledge_item_has_correct_source(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        item = adapter.to_knowledge_item(sample_page)
        assert item.id == "conf_pg-001"
        assert item.source.value == "external"
        assert item.metadata["subcategory"] == "confluence"
        assert item.metadata["space_key"] == "OPS"
        assert item.metadata["version"] == 3

    def test_knowledge_item_content_includes_title(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        item = adapter.to_knowledge_item(sample_page)
        assert "Deployment Runbook" in item.content
        assert "OPS" in item.content

    def test_knowledge_item_includes_labels(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        item = adapter.to_knowledge_item(sample_page)
        assert "runbook" in item.content
        assert "production" in item.content


class TestGetStats:
    def test_stats_empty(self, adapter: ConfluenceAdapter) -> None:
        stats = adapter.get_stats()
        assert stats["total_synced"] == 0
        assert stats["pending_sync"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_sync(
        self, adapter: ConfluenceAdapter, sample_page: ConfluencePageRecord
    ) -> None:
        adapter.store_page(sample_page)
        mock_mound = AsyncMock()
        await adapter.sync_to_km(mock_mound)

        stats = adapter.get_stats()
        assert stats["total_synced"] == 1
        assert stats["space_distribution"]["OPS"] == 1
        assert stats["total_labels"] == 3
