"""Tests for MemoryManager module."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.debate.memory_manager import MemoryManager
from aragora.memory.continuum import MemoryTier


class TestMemoryManagerCreation:
    """Tests for MemoryManager initialization."""

    def test_creates_with_no_dependencies(self):
        """Manager can be created with no dependencies."""
        manager = MemoryManager()
        assert manager.continuum_memory is None
        assert manager.critique_store is None
        assert manager.debate_embeddings is None

    def test_creates_with_all_dependencies(self):
        """Manager accepts all optional dependencies."""
        mock_continuum = MagicMock()
        mock_critique = MagicMock()
        mock_embeddings = MagicMock()
        mock_emitter = MagicMock()

        manager = MemoryManager(
            continuum_memory=mock_continuum,
            critique_store=mock_critique,
            debate_embeddings=mock_embeddings,
            event_emitter=mock_emitter,
            loop_id="test-loop",
        )

        assert manager.continuum_memory is mock_continuum
        assert manager.critique_store is mock_critique
        assert manager.debate_embeddings is mock_embeddings
        assert manager.event_emitter is mock_emitter
        assert manager.loop_id == "test-loop"

    def test_retrieves_ids_tracking_initialized_empty(self):
        """Retrieved IDs list starts empty."""
        manager = MemoryManager()
        assert manager._retrieved_ids == []


class TestDomainExtraction:
    """Tests for domain extraction logic."""

    def test_default_domain_when_no_extractor(self):
        """Returns 'general' when no domain extractor provided."""
        manager = MemoryManager()
        assert manager._get_domain() == "general"

    def test_uses_domain_extractor_when_provided(self):
        """Uses domain extractor callable when provided."""
        manager = MemoryManager(domain_extractor=lambda: "science")
        assert manager._get_domain() == "science"


class TestStoreDebateOutcome:
    """Tests for storing debate outcomes."""

    def test_skips_when_no_continuum_memory(self):
        """Does nothing when continuum_memory is None."""
        manager = MemoryManager()
        mock_result = MagicMock(final_answer="test", confidence=0.8)

        # Should not raise
        manager.store_debate_outcome(mock_result, "test task")

    def test_skips_when_no_final_answer(self):
        """Does nothing when result has no final_answer."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        mock_result = MagicMock(final_answer=None, confidence=0.8)

        manager.store_debate_outcome(mock_result, "test task")
        mock_continuum.add.assert_not_called()

    def test_stores_outcome_with_correct_tier(self):
        """Stores outcome with appropriate tier based on quality."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        # High quality debate (multi-round, high confidence)
        mock_result = MagicMock(
            id="test-id-123",
            final_answer="The winning approach is...",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=3,
            winner="agent1",
        )

        manager.store_debate_outcome(mock_result, "test task")

        mock_continuum.add.assert_called_once()
        call_kwargs = mock_continuum.add.call_args.kwargs
        assert call_kwargs["tier"] == MemoryTier.FAST  # High quality = fast tier
        assert call_kwargs["importance"] > 0.8

    def test_stores_medium_tier_for_moderate_quality(self):
        """Stores with medium tier for moderate quality debates."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        mock_result = MagicMock(
            id="test-id-123",
            final_answer="Result...",
            confidence=0.6,
            consensus_reached=False,
            rounds_used=1,
            winner="agent1",
        )

        manager.store_debate_outcome(mock_result, "test task")

        call_kwargs = mock_continuum.add.call_args.kwargs
        assert call_kwargs["tier"] == MemoryTier.MEDIUM


class TestStoreEvidence:
    """Tests for storing evidence snippets."""

    def test_skips_when_no_continuum_memory(self):
        """Does nothing when continuum_memory is None."""
        manager = MemoryManager()
        manager.store_evidence([{"content": "test"}], "task")
        # Should not raise

    def test_skips_when_no_snippets(self):
        """Does nothing when snippets list is empty."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        manager.store_evidence([], "task")
        mock_continuum.add.assert_not_called()

    def test_stores_evidence_snippets(self):
        """Stores evidence snippets with correct format."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        mock_snippet = MagicMock(
            content="Evidence content that is long enough to pass the filter",
            source="web",
            relevance=0.7,
        )

        manager.store_evidence([mock_snippet], "test task")

        mock_continuum.add.assert_called_once()
        call_kwargs = mock_continuum.add.call_args.kwargs
        assert call_kwargs["tier"] == MemoryTier.MEDIUM
        assert "Evidence" in call_kwargs["content"]

    def test_limits_snippets_to_ten(self):
        """Only stores first 10 snippets."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        snippets = [
            MagicMock(
                content=f"Evidence {i} with enough content to pass", source="web", relevance=0.5
            )
            for i in range(15)
        ]

        manager.store_evidence(snippets, "test task")

        # Should only call add up to 10 times
        assert mock_continuum.add.call_count <= 10


class TestUpdateMemoryOutcomes:
    """Tests for updating memory outcomes."""

    def test_skips_when_no_continuum_memory(self):
        """Does nothing when continuum_memory is None."""
        manager = MemoryManager()
        manager._retrieved_ids = ["id1", "id2"]
        mock_result = MagicMock(consensus_reached=True, confidence=0.8)

        manager.update_memory_outcomes(mock_result)
        # Should not raise

    def test_skips_when_no_retrieved_ids(self):
        """Does nothing when no IDs were retrieved."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        mock_result = MagicMock(consensus_reached=True, confidence=0.8)

        manager.update_memory_outcomes(mock_result)
        mock_continuum.update_outcome.assert_not_called()

    def test_updates_all_retrieved_ids(self):
        """Updates all tracked memory IDs."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        manager._retrieved_ids = ["id1", "id2", "id3"]

        mock_result = MagicMock(consensus_reached=True, confidence=0.8)

        manager.update_memory_outcomes(mock_result)

        assert mock_continuum.update_outcome.call_count == 3

    def test_clears_ids_after_update(self):
        """Clears retrieved IDs after updating."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        manager._retrieved_ids = ["id1"]

        mock_result = MagicMock(consensus_reached=True, confidence=0.8)

        manager.update_memory_outcomes(mock_result)

        assert manager._retrieved_ids == []


class TestFetchHistoricalContext:
    """Tests for fetching historical context."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_embeddings(self):
        """Returns empty string when no embeddings database."""
        manager = MemoryManager()
        result = await manager.fetch_historical_context("test task")
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_similar_debates(self):
        """Returns empty string when no similar debates found."""
        mock_embeddings = MagicMock()
        mock_embeddings.find_similar_debates = AsyncMock(return_value=[])
        manager = MemoryManager(debate_embeddings=mock_embeddings)

        result = await manager.fetch_historical_context("test task")
        assert result == ""

    @pytest.mark.asyncio
    async def test_formats_historical_context(self):
        """Formats historical context with similarity scores."""
        mock_embeddings = MagicMock()
        mock_embeddings.find_similar_debates = AsyncMock(
            return_value=[
                ("debate-1", "Previous debate about X", 0.85),
                ("debate-2", "Another related debate", 0.72),
            ]
        )
        manager = MemoryManager(debate_embeddings=mock_embeddings)

        result = await manager.fetch_historical_context("test task")

        assert "HISTORICAL CONTEXT" in result
        assert "85%" in result
        assert "72%" in result
        assert "Previous debate about X" in result


class TestGetSuccessfulPatterns:
    """Tests for retrieving successful patterns."""

    def test_returns_empty_when_no_critique_store(self):
        """Returns empty string when no critique store."""
        manager = MemoryManager()
        result = manager.get_successful_patterns()
        assert result == ""

    def test_returns_empty_when_no_patterns(self):
        """Returns empty string when no patterns found."""
        mock_critique = MagicMock()
        mock_critique.retrieve_patterns.return_value = []
        manager = MemoryManager(critique_store=mock_critique)

        result = manager.get_successful_patterns()
        assert result == ""

    def test_formats_patterns_for_prompt(self):
        """Formats patterns with severity labels."""
        mock_pattern = MagicMock(
            issue_type="logic",
            issue_text="Missing evidence",
            suggestion_text="Add citations",
            success_count=5,
            avg_severity=0.8,
        )
        mock_critique = MagicMock()
        mock_critique.retrieve_patterns.return_value = [mock_pattern]
        manager = MemoryManager(critique_store=mock_critique)

        result = manager.get_successful_patterns()

        assert "LEARNED PATTERNS" in result
        assert "LOGIC" in result
        assert "HIGH SEVERITY" in result


class TestIdTracking:
    """Tests for ID tracking methods."""

    def test_track_retrieved_ids(self):
        """Tracks provided IDs."""
        manager = MemoryManager()
        manager.track_retrieved_ids(["id1", "id2", "id3"])
        assert manager._retrieved_ids == ["id1", "id2", "id3"]

    def test_track_filters_empty_ids(self):
        """Filters out empty/None IDs."""
        manager = MemoryManager()
        manager.track_retrieved_ids(["id1", "", None, "id2"])
        assert manager._retrieved_ids == ["id1", "id2"]

    def test_clear_retrieved_ids(self):
        """Clears tracked IDs."""
        manager = MemoryManager()
        manager._retrieved_ids = ["id1", "id2"]
        manager.clear_retrieved_ids()
        assert manager._retrieved_ids == []


class TestPatternFormatting:
    """Tests for pattern formatting helper."""

    def test_empty_patterns_returns_empty(self):
        """Returns empty string for empty patterns."""
        manager = MemoryManager()
        result = manager._format_patterns_for_prompt([])
        assert result == ""

    def test_formats_with_severity_labels(self):
        """Includes severity labels based on avg_severity."""
        manager = MemoryManager()
        patterns = [
            {"category": "logic", "pattern": "test", "occurrences": 3, "avg_severity": 0.8},
            {"category": "evidence", "pattern": "test2", "occurrences": 2, "avg_severity": 0.5},
            {"category": "other", "pattern": "test3", "occurrences": 1, "avg_severity": 0.2},
        ]

        result = manager._format_patterns_for_prompt(patterns)

        assert "HIGH SEVERITY" in result
        assert "MEDIUM" in result
        # Low severity should have no label


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
