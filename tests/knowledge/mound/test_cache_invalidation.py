"""Tests for Knowledge Mound cache invalidation on ingest."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aragora.knowledge.mound import KnowledgeMound
from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource, MoundConfig


@pytest.mark.asyncio
async def test_ingest_invalidates_query_cache_for_workspace():
    """ingest() should clear overlapping cached query results after storing."""
    mound = KnowledgeMound(
        config=MoundConfig(
            enable_deduplication=False,
            enable_staleness_detection=False,
            enable_culture_accumulator=False,
        ),
        workspace_id="ws-test",
    )
    mound._initialized = True
    mound._cache = AsyncMock()
    mound._save_node = AsyncMock()

    result = await mound.ingest(
        IngestionRequest(
            content="New policy knowledge",
            workspace_id="ws-test",
            source_type=KnowledgeSource.DOCUMENT,
        )
    )

    assert result.success is True
    mound._save_node.assert_awaited_once()
    mound._cache.invalidate_queries.assert_awaited_once_with("ws-test")
