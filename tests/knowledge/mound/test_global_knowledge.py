"""
Tests for Global Knowledge Mixin.

Tests cover:
- store_verified_fact
- query_global_knowledge
- promote_to_global
- get_system_facts
- merge_global_results
- SYSTEM_WORKSPACE_ID constant
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.ops.global_knowledge import (
    GlobalKnowledgeMixin,
    SYSTEM_WORKSPACE_ID,
)
from aragora.knowledge.mound.types import (
    VisibilityLevel,
    KnowledgeSource,
    ConfidenceLevel,
)


# =============================================================================
# SYSTEM_WORKSPACE_ID Tests
# =============================================================================


class TestSystemWorkspaceId:
    """Tests for SYSTEM_WORKSPACE_ID constant."""

    def test_system_workspace_id_value(self):
        """Should have correct value."""
        assert SYSTEM_WORKSPACE_ID == "__system__"

    def test_system_workspace_id_is_string(self):
        """Should be a string."""
        assert isinstance(SYSTEM_WORKSPACE_ID, str)


# =============================================================================
# MockKnowledgeMound for Testing
# =============================================================================


class MockKnowledgeMound(GlobalKnowledgeMixin):
    """Mock KnowledgeMound with GlobalKnowledgeMixin."""

    def __init__(self):
        self.config = MagicMock()
        self.config.max_query_limit = 100
        self.workspace_id = "test_workspace"
        self._meta_store = MagicMock()
        self._cache = None
        self._initialized = True
        self._stored_items = []
        self._items_db = {}

    def _ensure_initialized(self):
        if not self._initialized:
            raise RuntimeError("Not initialized")

    async def store(self, request):
        """Mock store method."""
        result = MagicMock()
        result.node_id = f"kn_{len(self._stored_items)}"
        self._stored_items.append(request)
        self._items_db[result.node_id] = request
        return result

    async def query(self, query, sources=("all",), filters=None, limit=20, workspace_id=None):
        """Mock query method."""
        result = MagicMock()
        items = []
        for node_id, req in self._items_db.items():
            if workspace_id and req.workspace_id != workspace_id:
                continue
            if query and query.lower() not in req.content.lower():
                continue
            item = MagicMock()
            item.id = node_id
            item.content = req.content
            item.importance = req.confidence
            item.metadata = req.metadata
            items.append(item)
        result.items = items[:limit]
        return result

    async def get(self, node_id, workspace_id=None):
        """Mock get method."""
        if node_id in self._items_db:
            req = self._items_db[node_id]
            item = MagicMock()
            item.id = node_id
            item.content = req.content
            item.confidence = req.confidence
            item.metadata = req.metadata
            return item
        return None


# =============================================================================
# store_verified_fact Tests
# =============================================================================


class TestStoreVerifiedFact:
    """Tests for store_verified_fact method."""

    @pytest.mark.asyncio
    async def test_store_basic_fact(self):
        """Should store a basic verified fact."""
        mound = MockKnowledgeMound()
        node_id = await mound.store_verified_fact(
            content="Water boils at 100°C at sea level",
            source="scientific_consensus",
        )

        assert node_id == "kn_0"
        assert len(mound._stored_items) == 1

        stored = mound._stored_items[0]
        assert stored.content == "Water boils at 100°C at sea level"
        assert stored.workspace_id == SYSTEM_WORKSPACE_ID
        assert stored.tier == "glacial"
        assert stored.metadata["source"] == "scientific_consensus"
        assert stored.metadata["visibility"] == VisibilityLevel.SYSTEM.value

    @pytest.mark.asyncio
    async def test_store_fact_with_evidence(self):
        """Should store fact with evidence references."""
        mound = MockKnowledgeMound()
        node_id = await mound.store_verified_fact(
            content="Test fact",
            source="debate_consensus",
            confidence=0.95,
            evidence_ids=["ev_1", "ev_2", "ev_3"],
            verified_by="admin_user",
            topics=["science", "physics"],
        )

        stored = mound._stored_items[0]
        assert stored.confidence == 0.95
        assert stored.metadata["evidence_ids"] == ["ev_1", "ev_2", "ev_3"]
        assert stored.metadata["verified_by"] == "admin_user"
        assert stored.topics == ["science", "physics"]

    @pytest.mark.asyncio
    async def test_store_fact_sets_global_flag(self):
        """Should set is_global flag in metadata."""
        mound = MockKnowledgeMound()
        await mound.store_verified_fact(
            content="Test fact",
            source="manual",
        )

        stored = mound._stored_items[0]
        assert stored.metadata["is_global"] is True

    @pytest.mark.asyncio
    async def test_store_fact_not_initialized(self):
        """Should raise when not initialized."""
        mound = MockKnowledgeMound()
        mound._initialized = False

        with pytest.raises(RuntimeError):
            await mound.store_verified_fact(
                content="Test",
                source="test",
            )


# =============================================================================
# query_global_knowledge Tests
# =============================================================================


class TestQueryGlobalKnowledge:
    """Tests for query_global_knowledge method."""

    @pytest.mark.asyncio
    async def test_query_empty_mound(self):
        """Should return empty list when no facts."""
        mound = MockKnowledgeMound()
        results = await mound.query_global_knowledge("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_query_finds_matching_facts(self):
        """Should find matching global facts."""
        mound = MockKnowledgeMound()

        # Store some facts
        await mound.store_verified_fact(
            content="Water boils at 100°C",
            source="science",
        )
        await mound.store_verified_fact(
            content="Ice melts at 0°C",
            source="science",
        )
        await mound.store_verified_fact(
            content="The sky is blue",
            source="observation",
        )

        results = await mound.query_global_knowledge("water")
        assert len(results) == 1
        assert "water" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_query_respects_limit(self):
        """Should respect limit parameter."""
        mound = MockKnowledgeMound()

        # Store multiple facts
        for i in range(10):
            await mound.store_verified_fact(
                content=f"Fact number {i}",
                source="test",
            )

        results = await mound.query_global_knowledge("fact", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_query_only_system_workspace(self):
        """Should only query system workspace."""
        mound = MockKnowledgeMound()

        # Store a global fact
        await mound.store_verified_fact(
            content="Global fact about water",
            source="science",
        )

        # Store a workspace-specific fact directly
        from aragora.knowledge.mound.types import IngestionRequest

        workspace_req = IngestionRequest(
            content="Workspace fact about water",
            workspace_id="user_workspace",
            source_type=KnowledgeSource.FACT,
            confidence=0.8,
        )
        await mound.store(workspace_req)

        # Query global should only find global fact
        results = await mound.query_global_knowledge("water")
        assert len(results) == 1
        assert "Global" in results[0].content


# =============================================================================
# promote_to_global Tests
# =============================================================================


class TestPromoteToGlobal:
    """Tests for promote_to_global method."""

    @pytest.mark.asyncio
    async def test_promote_existing_item(self):
        """Should promote workspace item to global."""
        mound = MockKnowledgeMound()

        # First store a workspace item
        from aragora.knowledge.mound.types import IngestionRequest

        req = IngestionRequest(
            content="Important discovery",
            workspace_id="research_team",
            source_type=KnowledgeSource.FACT,
            confidence=0.9,
            metadata={"topics": ["research", "discovery"]},
        )
        result = await mound.store(req)
        original_id = result.node_id

        # Promote to global
        global_id = await mound.promote_to_global(
            item_id=original_id,
            workspace_id="research_team",
            promoted_by="lead_researcher",
            reason="high_consensus",
        )

        # Should create new global item
        assert global_id != original_id
        assert len(mound._stored_items) == 2

        # Check the global item
        global_req = mound._stored_items[1]
        assert global_req.workspace_id == SYSTEM_WORKSPACE_ID
        assert "promoted_from:research_team" in global_req.metadata["source"]

    @pytest.mark.asyncio
    async def test_promote_nonexistent_item(self):
        """Should raise when item not found."""
        mound = MockKnowledgeMound()

        with pytest.raises(ValueError, match="not found"):
            await mound.promote_to_global(
                item_id="nonexistent",
                workspace_id="ws_1",
                promoted_by="admin",
                reason="test",
            )

    @pytest.mark.asyncio
    async def test_promote_preserves_content(self):
        """Should preserve original content."""
        mound = MockKnowledgeMound()

        from aragora.knowledge.mound.types import IngestionRequest

        original_content = "Original important fact content"
        req = IngestionRequest(
            content=original_content,
            workspace_id="ws_1",
            source_type=KnowledgeSource.FACT,
            confidence=0.85,
        )
        result = await mound.store(req)

        await mound.promote_to_global(
            item_id=result.node_id,
            workspace_id="ws_1",
            promoted_by="admin",
            reason="verified",
        )

        global_req = mound._stored_items[1]
        assert global_req.content == original_content


# =============================================================================
# get_system_facts Tests
# =============================================================================


class TestGetSystemFacts:
    """Tests for get_system_facts method."""

    @pytest.mark.asyncio
    async def test_get_all_system_facts(self):
        """Should get all system facts."""
        mound = MockKnowledgeMound()

        # Store some facts
        await mound.store_verified_fact(content="Fact 1", source="test")
        await mound.store_verified_fact(content="Fact 2", source="test")
        await mound.store_verified_fact(content="Fact 3", source="test")

        results = await mound.get_system_facts()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_system_facts_with_limit(self):
        """Should respect limit."""
        mound = MockKnowledgeMound()

        for i in range(10):
            await mound.store_verified_fact(content=f"Fact {i}", source="test")

        results = await mound.get_system_facts(limit=5)
        assert len(results) <= 5


# =============================================================================
# merge_global_results Tests
# =============================================================================


class TestMergeGlobalResults:
    """Tests for merge_global_results method."""

    @pytest.mark.asyncio
    async def test_merge_adds_global_results(self):
        """Should add global results to workspace results."""
        mound = MockKnowledgeMound()

        # Store global fact
        await mound.store_verified_fact(
            content="Global water fact",
            source="science",
        )

        # Create mock workspace results
        workspace_item = MagicMock()
        workspace_item.id = "ws_item_1"
        workspace_item.content = "Workspace water info"
        workspace_item.importance = 0.7
        workspace_item.metadata = {"content_hash": "ws_hash_1"}

        workspace_results = [workspace_item]

        merged = await mound.merge_global_results(
            workspace_results=workspace_results,
            query="water",
            global_limit=5,
        )

        assert len(merged) >= 1
        # Workspace item should be included
        assert any(item.id == "ws_item_1" for item in merged)

    @pytest.mark.asyncio
    async def test_merge_deduplicates(self):
        """Should deduplicate by content hash."""
        mound = MockKnowledgeMound()

        # Store global fact
        await mound.store_verified_fact(
            content="Duplicate content",
            source="science",
        )

        # Create workspace result with same content hash
        workspace_item = MagicMock()
        workspace_item.id = "ws_item_1"
        workspace_item.content = "Duplicate content"
        workspace_item.importance = 0.7
        workspace_item.metadata = {"content_hash": "Duplicate content"[:100]}

        workspace_results = [workspace_item]

        merged = await mound.merge_global_results(
            workspace_results=workspace_results,
            query="duplicate",
            global_limit=5,
        )

        # Should only have workspace result (global is duplicate)
        assert len(merged) == 1

    @pytest.mark.asyncio
    async def test_merge_sorts_by_importance(self):
        """Should sort merged results by importance."""
        mound = MockKnowledgeMound()

        # Store global facts with different confidence
        await mound.store_verified_fact(
            content="High importance fact",
            source="science",
            confidence=0.95,
        )
        await mound.store_verified_fact(
            content="Low importance fact",
            source="science",
            confidence=0.5,
        )

        # Create workspace result
        workspace_item = MagicMock()
        workspace_item.id = "ws_item"
        workspace_item.content = "Medium importance"
        workspace_item.importance = 0.7
        workspace_item.metadata = {"content_hash": "medium_hash"}

        merged = await mound.merge_global_results(
            workspace_results=[workspace_item],
            query="fact",
            global_limit=5,
        )

        # Check sorting (highest importance first)
        if len(merged) > 1:
            for i in range(len(merged) - 1):
                assert (merged[i].importance or 0) >= (merged[i + 1].importance or 0)


# =============================================================================
# get_system_workspace_id Tests
# =============================================================================


class TestGetSystemWorkspaceId:
    """Tests for get_system_workspace_id method."""

    def test_returns_system_workspace_id(self):
        """Should return the system workspace ID."""
        mound = MockKnowledgeMound()
        assert mound.get_system_workspace_id() == SYSTEM_WORKSPACE_ID
        assert mound.get_system_workspace_id() == "__system__"
