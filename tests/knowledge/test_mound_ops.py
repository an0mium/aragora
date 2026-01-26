"""
Tests for Knowledge Mound Operations mixins and debate integration.

Tests cover:
- KnowledgeMoundOperations (debate integration)
- StalenessOperationsMixin
- CultureOperationsMixin
- SyncOperationsMixin
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Test imports work
from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations
from aragora.knowledge.mound.ops import (
    StalenessOperationsMixin,
    CultureOperationsMixin,
    SyncOperationsMixin,
)


# =============================================================================
# KnowledgeMoundOperations Tests (Debate Integration)
# =============================================================================


class TestKnowledgeMoundOperations:
    """Tests for KnowledgeMoundOperations class."""

    def test_init_defaults(self):
        """Should initialize with default settings."""
        ops = KnowledgeMoundOperations()
        assert ops.knowledge_mound is None
        assert ops.enable_retrieval is True
        assert ops.enable_ingestion is True

    def test_init_with_mound(self):
        """Should initialize with provided mound."""
        mock_mound = MagicMock()
        ops = KnowledgeMoundOperations(
            knowledge_mound=mock_mound,
            enable_retrieval=False,
            enable_ingestion=False,
        )
        assert ops.knowledge_mound is mock_mound
        assert ops.enable_retrieval is False
        assert ops.enable_ingestion is False

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_no_mound(self):
        """Should return None when no mound is configured."""
        ops = KnowledgeMoundOperations()
        result = await ops.fetch_knowledge_context("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_disabled(self):
        """Should return None when retrieval is disabled."""
        mock_mound = MagicMock()
        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound, enable_retrieval=False)
        result = await ops.fetch_knowledge_context("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_success(self):
        """Should format and return knowledge context."""
        mock_mound = MagicMock()
        mock_mound.query_semantic = AsyncMock(
            return_value=MagicMock(
                items=[
                    MagicMock(source="debate", confidence=0.85, content="Test knowledge 1"),
                    MagicMock(source="document", confidence=0.90, content="Test knowledge 2"),
                ]
            )
        )

        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)
        result = await ops.fetch_knowledge_context("test task", limit=5)

        assert result is not None
        assert "KNOWLEDGE MOUND CONTEXT" in result
        assert "Test knowledge 1" in result
        assert "Test knowledge 2" in result
        assert "85%" in result
        mock_mound.query_semantic.assert_called_once_with(
            query="test task",
            limit=5,
            min_confidence=0.5,
        )

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_empty_results(self):
        """Should return None when no results found."""
        mock_mound = MagicMock()
        mock_mound.query_semantic = AsyncMock(return_value=MagicMock(items=[]))

        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)
        result = await ops.fetch_knowledge_context("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_handles_error(self):
        """Should handle errors gracefully."""
        mock_mound = MagicMock()
        mock_mound.query_semantic = AsyncMock(side_effect=Exception("Query failed"))

        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)
        result = await ops.fetch_knowledge_context("test task")
        assert result is None

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome_no_mound(self):
        """Should silently return when no mound is configured."""
        ops = KnowledgeMoundOperations()
        mock_result = MagicMock(final_answer="Test answer", confidence=0.9)
        await ops.ingest_debate_outcome(mock_result)
        # No error raised

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome_disabled(self):
        """Should return when ingestion is disabled."""
        mock_mound = MagicMock()
        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound, enable_ingestion=False)
        mock_result = MagicMock(final_answer="Test answer", confidence=0.9)
        await ops.ingest_debate_outcome(mock_result)
        mock_mound.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome_low_confidence(self):
        """Should skip low-confidence outcomes."""
        mock_mound = MagicMock()
        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)
        mock_result = MagicMock(final_answer="Test answer", confidence=0.5)  # Below 0.7
        await ops.ingest_debate_outcome(mock_result)
        mock_mound.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome_no_answer(self):
        """Should skip outcomes without final answer."""
        mock_mound = MagicMock()
        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)
        mock_result = MagicMock(final_answer=None, confidence=0.9)
        await ops.ingest_debate_outcome(mock_result)
        mock_mound.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome_success(self):
        """Should ingest high-confidence outcome."""
        mock_mound = MagicMock()
        mock_mound.store = AsyncMock(return_value=MagicMock(node_id="kn_123"))
        mock_mound.workspace_id = "ws_test"

        ops = KnowledgeMoundOperations(knowledge_mound=mock_mound)
        mock_result = MagicMock(
            id="debate_123",
            final_answer="The optimal solution is X",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=5,
            participants=["agent_a", "agent_b"],
            winner="agent_a",
        )
        mock_result.debate_cruxes = None
        mock_env = MagicMock(task="What is optimal?")

        await ops.ingest_debate_outcome(mock_result, mock_env)

        mock_mound.store.assert_called_once()
        call_args = mock_mound.store.call_args[0][0]
        assert call_args.content == "Debate Conclusion: The optimal solution is X"


# =============================================================================
# StalenessOperationsMixin Tests
# =============================================================================


class MockStalenessHost(StalenessOperationsMixin):
    """Mock host class for testing StalenessOperationsMixin."""

    def __init__(self):
        self.config = MagicMock()
        self.workspace_id = "ws_test"
        self._staleness_detector = None
        self._initialized = True
        self._updates = []

    def _ensure_initialized(self):
        pass

    async def update(self, node_id, updates):
        self._updates.append((node_id, updates))
        return MagicMock()


class TestStalenessOperationsMixin:
    """Tests for StalenessOperationsMixin."""

    @pytest.mark.asyncio
    async def test_get_stale_knowledge_no_detector(self):
        """Should return empty list when no detector configured."""
        host = MockStalenessHost()
        result = await host.get_stale_knowledge()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_stale_knowledge_with_detector(self):
        """Should query staleness detector."""
        host = MockStalenessHost()
        host._staleness_detector = MagicMock()
        host._staleness_detector.get_stale_nodes = AsyncMock(
            return_value=[
                MagicMock(node_id="kn_1", staleness_score=0.8),
                MagicMock(node_id="kn_2", staleness_score=0.6),
            ]
        )

        result = await host.get_stale_knowledge(threshold=0.5, limit=50)

        assert len(result) == 2
        host._staleness_detector.get_stale_nodes.assert_called_once_with(
            workspace_id="ws_test",
            threshold=0.5,
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_mark_validated(self):
        """Should update node with validation status."""
        host = MockStalenessHost()
        await host.mark_validated("kn_123", "user_alice", confidence=0.95)

        assert len(host._updates) == 1
        node_id, updates = host._updates[0]
        assert node_id == "kn_123"
        assert updates["validation_status"] == "majority_agreed"
        assert updates["staleness_score"] == 0.0
        assert updates["confidence"] == 0.95
        assert "last_validated_at" in updates

    @pytest.mark.asyncio
    async def test_schedule_revalidation(self):
        """Should create revalidation tasks."""
        host = MockStalenessHost()

        # Mock the control plane task queue
        with patch("aragora.server.handlers.features.control_plane._task_queue", []):
            task_ids = await host.schedule_revalidation(["kn_1", "kn_2"], priority="high")

        assert len(task_ids) == 2
        assert all(t.startswith("reval_") for t in task_ids)
        # Nodes should be marked
        assert len(host._updates) == 2


# =============================================================================
# CultureOperationsMixin Tests
# =============================================================================


class MockCultureHost(CultureOperationsMixin):
    """Mock host class for testing CultureOperationsMixin."""

    def __init__(self):
        self.config = MagicMock()
        self.workspace_id = "ws_test"
        self._culture_accumulator = None
        self._cache = None
        self._initialized = True
        self._org_culture_manager = None

    def _ensure_initialized(self):
        pass


class TestCultureOperationsMixin:
    """Tests for CultureOperationsMixin."""

    @pytest.mark.asyncio
    async def test_get_culture_profile_no_accumulator(self):
        """Should return empty profile when no accumulator."""
        host = MockCultureHost()
        profile = await host.get_culture_profile()

        assert profile.workspace_id == "ws_test"
        assert profile.patterns == {}
        assert profile.total_observations == 0

    @pytest.mark.asyncio
    async def test_get_culture_profile_with_accumulator(self):
        """Should query culture accumulator."""
        host = MockCultureHost()
        host._culture_accumulator = MagicMock()
        mock_profile = MagicMock(
            workspace_id="ws_test",
            patterns={"pattern_1": MagicMock()},
            total_observations=42,
        )
        host._culture_accumulator.get_profile = AsyncMock(return_value=mock_profile)

        profile = await host.get_culture_profile()

        assert profile is mock_profile
        host._culture_accumulator.get_profile.assert_called_once_with("ws_test")

    @pytest.mark.asyncio
    async def test_get_culture_profile_cached(self):
        """Should return cached profile when available."""
        host = MockCultureHost()
        host._cache = MagicMock()
        mock_cached = MagicMock()
        host._cache.get_culture = AsyncMock(return_value=mock_cached)

        profile = await host.get_culture_profile()

        assert profile is mock_cached
        host._cache.get_culture.assert_called_once_with("ws_test")

    @pytest.mark.asyncio
    async def test_observe_debate_no_accumulator(self):
        """Should return empty list when no accumulator."""
        host = MockCultureHost()
        result = await host.observe_debate(MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_observe_debate_with_accumulator(self):
        """Should extract patterns from debate."""
        host = MockCultureHost()
        host._culture_accumulator = MagicMock()
        mock_patterns = [MagicMock(), MagicMock()]
        host._culture_accumulator.observe_debate = AsyncMock(return_value=mock_patterns)

        mock_debate = MagicMock()
        result = await host.observe_debate(mock_debate)

        assert result is mock_patterns
        host._culture_accumulator.observe_debate.assert_called_once_with(mock_debate, "ws_test")

    @pytest.mark.asyncio
    async def test_recommend_agents_no_accumulator(self):
        """Should return empty list when no accumulator."""
        host = MockCultureHost()
        result = await host.recommend_agents("code_review")
        assert result == []

    @pytest.mark.asyncio
    async def test_recommend_agents_with_accumulator(self):
        """Should get agent recommendations."""
        host = MockCultureHost()
        host._culture_accumulator = MagicMock()
        host._culture_accumulator.recommend_agents = AsyncMock(
            return_value=["anthropic-api", "openai-api"]
        )

        result = await host.recommend_agents("code_review")

        assert result == ["anthropic-api", "openai-api"]
        host._culture_accumulator.recommend_agents.assert_called_once_with("code_review", "ws_test")


# =============================================================================
# SyncOperationsMixin Tests
# =============================================================================


class MockSyncHost(SyncOperationsMixin):
    """Mock host class for testing SyncOperationsMixin."""

    def __init__(self):
        self.config = MagicMock()
        self.workspace_id = "ws_test"
        self._continuum = None
        self._consensus = None
        self._facts = None
        self._evidence = None
        self._critique = None
        self._initialized = True
        self._stored = []

    def _ensure_initialized(self):
        pass

    async def store(self, request):
        self._stored.append(request)
        return MagicMock(deduplicated=False, relationships_created=0)


class TestSyncOperationsMixin:
    """Tests for SyncOperationsMixin."""

    @pytest.mark.asyncio
    async def test_sync_from_continuum(self):
        """Should sync entries from ContinuumMemory."""
        host = MockSyncHost()

        # Mock continuum with entries
        mock_continuum = MagicMock()
        mock_entry = MagicMock(
            id="cm_123",
            content="Test memory content",
            importance=0.8,
            tier=MagicMock(value="slow"),
            surprise_score=0.1,
            consolidation_score=0.5,
            update_count=3,
            success_rate=0.9,
            metadata={"key": "value"},
        )
        mock_continuum.retrieve = MagicMock(return_value=[mock_entry])

        result = await host.sync_from_continuum(mock_continuum)

        assert result.source == "continuum"
        assert result.nodes_synced == 1
        assert result.nodes_updated == 0
        assert result.nodes_skipped == 0
        assert len(host._stored) == 1

    @pytest.mark.asyncio
    async def test_sync_from_continuum_empty(self):
        """Should handle empty continuum."""
        host = MockSyncHost()
        mock_continuum = MagicMock()
        mock_continuum.retrieve = MagicMock(return_value=[])

        result = await host.sync_from_continuum(mock_continuum)

        assert result.nodes_synced == 0
        assert len(host._stored) == 0

    @pytest.mark.asyncio
    async def test_sync_all_no_sources(self):
        """Should return empty dict when no sources connected."""
        host = MockSyncHost()
        results = await host.sync_all()
        assert results == {}

    @pytest.mark.asyncio
    async def test_sync_all_with_sources(self):
        """Should sync from all connected sources."""
        host = MockSyncHost()

        # Connect continuum
        mock_continuum = MagicMock()
        mock_continuum.retrieve = MagicMock(return_value=[])
        host._continuum = mock_continuum

        results = await host.sync_all()

        assert "continuum" in results
        assert results["continuum"].source == "continuum"


# =============================================================================
# Integration Tests
# =============================================================================


class TestKnowledgeMoundOpsIntegration:
    """Integration tests for Knowledge Mound ops."""

    def test_all_ops_importable(self):
        """Verify all ops modules are importable."""
        from aragora.knowledge.mound.ops import (
            StalenessOperationsMixin,
            CultureOperationsMixin,
            SyncOperationsMixin,
        )
        from aragora.debate.knowledge_mound_ops import KnowledgeMoundOperations

        assert StalenessOperationsMixin is not None
        assert CultureOperationsMixin is not None
        assert SyncOperationsMixin is not None
        assert KnowledgeMoundOperations is not None

    def test_ops_methods_exist(self):
        """Verify expected methods exist on mixins."""
        # Staleness
        assert hasattr(StalenessOperationsMixin, "get_stale_knowledge")
        assert hasattr(StalenessOperationsMixin, "mark_validated")
        assert hasattr(StalenessOperationsMixin, "schedule_revalidation")

        # Culture
        assert hasattr(CultureOperationsMixin, "get_culture_profile")
        assert hasattr(CultureOperationsMixin, "observe_debate")
        assert hasattr(CultureOperationsMixin, "recommend_agents")
        assert hasattr(CultureOperationsMixin, "get_org_culture")
        assert hasattr(CultureOperationsMixin, "add_culture_document")

        # Sync
        assert hasattr(SyncOperationsMixin, "sync_from_continuum")
        assert hasattr(SyncOperationsMixin, "sync_from_consensus")
        assert hasattr(SyncOperationsMixin, "sync_from_facts")
        assert hasattr(SyncOperationsMixin, "sync_from_evidence")
        assert hasattr(SyncOperationsMixin, "sync_from_critique")
        assert hasattr(SyncOperationsMixin, "sync_all")
