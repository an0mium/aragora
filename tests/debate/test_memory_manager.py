"""
Tests for the MemoryManager module.

Tests cover:
- MemoryManager initialization
- Domain extraction
- Pattern formatting and caching
- Retrieved ID tracking
- Event emission
- Memory outcome updates
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.debate.memory_manager import MemoryManager, MemoryEventType


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    def test_init_minimal(self):
        """MemoryManager can be initialized with no arguments."""
        manager = MemoryManager()

        assert manager.continuum_memory is None
        assert manager.critique_store is None
        assert manager.consensus_memory is None
        assert manager.debate_embeddings is None
        assert manager.event_emitter is None
        assert manager.spectator is None
        assert manager.loop_id == ""

    def test_init_with_continuum_memory(self):
        """MemoryManager stores continuum memory."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        assert manager.continuum_memory is mock_continuum

    def test_init_with_critique_store(self):
        """MemoryManager stores critique store."""
        mock_store = MagicMock()
        manager = MemoryManager(critique_store=mock_store)

        assert manager.critique_store is mock_store

    def test_init_with_consensus_memory(self):
        """MemoryManager stores consensus memory."""
        mock_consensus = MagicMock()
        manager = MemoryManager(consensus_memory=mock_consensus)

        assert manager.consensus_memory is mock_consensus

    def test_init_with_domain_extractor(self):
        """MemoryManager stores domain extractor."""
        extractor = MagicMock(return_value="security")
        manager = MemoryManager(domain_extractor=extractor)

        assert manager._domain_extractor is extractor

    def test_init_with_event_emitter(self):
        """MemoryManager stores event emitter."""
        mock_emitter = MagicMock()
        manager = MemoryManager(event_emitter=mock_emitter)

        assert manager.event_emitter is mock_emitter

    def test_init_with_loop_id(self):
        """MemoryManager stores loop ID."""
        manager = MemoryManager(loop_id="test-loop-123")

        assert manager.loop_id == "test-loop-123"

    def test_init_retrieved_ids_empty(self):
        """MemoryManager starts with empty retrieved IDs."""
        manager = MemoryManager()

        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}

    def test_init_patterns_cache_none(self):
        """MemoryManager starts with no patterns cache."""
        manager = MemoryManager()

        assert manager._patterns_cache is None


class TestMemoryManagerDomainExtraction:
    """Tests for domain extraction."""

    def test_get_domain_returns_general_without_extractor(self):
        """_get_domain returns 'general' without extractor."""
        manager = MemoryManager()

        assert manager._get_domain() == "general"

    def test_get_domain_uses_extractor(self):
        """_get_domain uses domain extractor when provided."""
        extractor = MagicMock(return_value="security")
        manager = MemoryManager(domain_extractor=extractor)

        assert manager._get_domain() == "security"
        extractor.assert_called_once()


class TestMemoryManagerEventEmission:
    """Tests for event emission."""

    def test_emit_event_does_nothing_without_emitter(self):
        """_emit_event does nothing when event_emitter is None."""
        manager = MemoryManager()

        # Should not raise
        manager._emit_event("test_event", data="test")

    def test_emit_event_uses_emit_sync_if_available(self):
        """_emit_event uses emit_sync if available."""
        mock_emitter = MagicMock()
        mock_emitter.emit_sync = MagicMock()
        manager = MemoryManager(event_emitter=mock_emitter, loop_id="test")

        manager._emit_event("test_event", data="value")

        mock_emitter.emit_sync.assert_called_once()

    def test_emit_event_falls_back_to_emit(self):
        """_emit_event falls back to emit method."""
        mock_emitter = MagicMock(spec=["emit"])
        manager = MemoryManager(event_emitter=mock_emitter, loop_id="test")

        manager._emit_event("test_event", data="value")

        mock_emitter.emit.assert_called_once()

    def test_emit_event_handles_exception(self):
        """_emit_event handles exceptions gracefully."""
        mock_emitter = MagicMock()
        mock_emitter.emit_sync.side_effect = RuntimeError("Emit failed")
        manager = MemoryManager(event_emitter=mock_emitter)

        # Should not raise
        manager._emit_event("test_event")


class TestMemoryManagerRetrievedIds:
    """Tests for retrieved ID tracking."""

    def test_track_retrieved_ids(self):
        """track_retrieved_ids stores IDs."""
        manager = MemoryManager()

        manager.track_retrieved_ids(["id1", "id2", "id3"])

        assert manager._retrieved_ids == ["id1", "id2", "id3"]

    def test_track_retrieved_ids_filters_empty(self):
        """track_retrieved_ids filters empty strings."""
        manager = MemoryManager()

        manager.track_retrieved_ids(["id1", "", "id2", ""])

        assert manager._retrieved_ids == ["id1", "id2"]

    def test_track_retrieved_ids_with_tiers(self):
        """track_retrieved_ids stores tier info."""
        from aragora.memory.continuum import MemoryTier

        manager = MemoryManager()
        tiers = {"id1": MemoryTier.FAST, "id2": MemoryTier.SLOW}

        manager.track_retrieved_ids(["id1", "id2"], tiers=tiers)

        assert manager._retrieved_tiers == tiers

    def test_clear_retrieved_ids(self):
        """clear_retrieved_ids empties tracking."""
        from aragora.memory.continuum import MemoryTier

        manager = MemoryManager()
        manager.track_retrieved_ids(["id1"], tiers={"id1": MemoryTier.FAST})

        manager.clear_retrieved_ids()

        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}

    def test_retrieved_ids_property_returns_copy(self):
        """retrieved_ids property returns a copy."""
        manager = MemoryManager()
        manager.track_retrieved_ids(["id1", "id2"])

        ids = manager.retrieved_ids
        ids.append("id3")

        assert manager._retrieved_ids == ["id1", "id2"]


class TestMemoryManagerPatternFormatting:
    """Tests for pattern formatting."""

    def test_format_patterns_empty(self):
        """_format_patterns_for_prompt returns empty for empty list."""
        manager = MemoryManager()

        result = manager._format_patterns_for_prompt([])

        assert result == ""

    def test_format_patterns_basic(self):
        """_format_patterns_for_prompt formats patterns correctly."""
        manager = MemoryManager()
        patterns = [
            {
                "category": "logic",
                "pattern": "Missing evidence",
                "occurrences": 3,
                "avg_severity": 0.5,
            }
        ]

        result = manager._format_patterns_for_prompt(patterns)

        assert "LEARNED PATTERNS" in result
        assert "LOGIC" in result
        assert "Missing evidence" in result
        assert "3 past debates" in result

    def test_format_patterns_high_severity(self):
        """_format_patterns_for_prompt marks high severity."""
        manager = MemoryManager()
        patterns = [
            {
                "category": "security",
                "pattern": "Unvalidated input",
                "occurrences": 5,
                "avg_severity": 0.8,
            }
        ]

        result = manager._format_patterns_for_prompt(patterns)

        assert "[HIGH SEVERITY]" in result

    def test_format_patterns_medium_severity(self):
        """_format_patterns_for_prompt marks medium severity."""
        manager = MemoryManager()
        patterns = [
            {
                "category": "performance",
                "pattern": "N+1 query",
                "occurrences": 2,
                "avg_severity": 0.5,
            }
        ]

        result = manager._format_patterns_for_prompt(patterns)

        assert "[MEDIUM]" in result

    def test_format_patterns_limits_to_five(self):
        """_format_patterns_for_prompt limits to 5 patterns."""
        manager = MemoryManager()
        patterns = [
            {"category": f"cat{i}", "pattern": f"pattern{i}", "occurrences": 1} for i in range(10)
        ]

        result = manager._format_patterns_for_prompt(patterns)

        # Should only include first 5
        assert "pattern4" in result
        assert "pattern5" not in result


class TestMemoryManagerGetSuccessfulPatterns:
    """Tests for successful pattern retrieval."""

    def test_get_patterns_without_store(self):
        """get_successful_patterns returns empty without critique_store."""
        manager = MemoryManager()

        result = manager.get_successful_patterns()

        assert result == ""

    def test_get_patterns_uses_cache(self):
        """get_successful_patterns uses cache within TTL."""
        import time

        mock_store = MagicMock()
        manager = MemoryManager(critique_store=mock_store)
        manager._patterns_cache = (time.time(), "cached_result")

        result = manager.get_successful_patterns()

        assert result == "cached_result"
        mock_store.retrieve_patterns.assert_not_called()

    def test_get_patterns_expired_cache(self):
        """get_successful_patterns refreshes expired cache."""
        import time

        mock_store = MagicMock()
        mock_store.retrieve_patterns.return_value = []
        manager = MemoryManager(critique_store=mock_store)
        manager._patterns_cache = (time.time() - 400, "old_result")  # Expired

        result = manager.get_successful_patterns()

        assert result == ""  # Empty because no patterns
        mock_store.retrieve_patterns.assert_called_once()


class TestMemoryManagerConfidenceToStrength:
    """Tests for confidence to strength conversion."""

    def test_unanimous_strength(self):
        """High confidence maps to unanimous."""
        manager = MemoryManager()

        strength = manager._confidence_to_strength(0.98)

        assert strength.value == "unanimous"

    def test_strong_strength(self):
        """80-94% confidence maps to strong."""
        manager = MemoryManager()

        strength = manager._confidence_to_strength(0.85)

        assert strength.value == "strong"

    def test_moderate_strength(self):
        """60-79% confidence maps to moderate."""
        manager = MemoryManager()

        strength = manager._confidence_to_strength(0.7)

        assert strength.value == "moderate"

    def test_weak_strength(self):
        """50-59% confidence maps to weak."""
        manager = MemoryManager()

        strength = manager._confidence_to_strength(0.55)

        assert strength.value == "weak"

    def test_split_strength(self):
        """30-49% confidence maps to split."""
        manager = MemoryManager()

        strength = manager._confidence_to_strength(0.4)

        assert strength.value == "split"

    def test_contested_strength(self):
        """Below 30% confidence maps to contested."""
        manager = MemoryManager()

        strength = manager._confidence_to_strength(0.2)

        assert strength.value == "contested"


class TestMemoryManagerStoreDebateOutcome:
    """Tests for storing debate outcomes."""

    def test_store_outcome_without_memory(self):
        """store_debate_outcome does nothing without continuum_memory."""
        manager = MemoryManager()
        mock_result = MagicMock()
        mock_result.final_answer = "Test answer"

        # Should not raise
        manager.store_debate_outcome(mock_result, "Test task")

    def test_store_outcome_without_answer(self):
        """store_debate_outcome does nothing without final_answer."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        mock_result = MagicMock()
        mock_result.final_answer = None

        manager.store_debate_outcome(mock_result, "Test task")

        mock_continuum.add.assert_not_called()

    def test_store_outcome_success(self):
        """store_debate_outcome stores to continuum memory."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        mock_result = MagicMock()
        mock_result.final_answer = "The answer is 42"
        mock_result.confidence = 0.8
        mock_result.consensus_reached = True
        mock_result.rounds_used = 3
        mock_result.id = "debate_123"
        mock_result.winner = "agent1"

        manager.store_debate_outcome(mock_result, "What is the answer?")

        mock_continuum.add.assert_called_once()
        call_kwargs = mock_continuum.add.call_args[1]
        assert "debate_outcome_" in call_kwargs["id"]


class TestMemoryManagerUpdateOutcomes:
    """Tests for updating memory outcomes."""

    def test_update_outcomes_without_memory(self):
        """update_memory_outcomes does nothing without continuum_memory."""
        manager = MemoryManager()
        mock_result = MagicMock()

        # Should not raise
        manager.update_memory_outcomes(mock_result)

    def test_update_outcomes_without_ids(self):
        """update_memory_outcomes does nothing without retrieved IDs."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        mock_result = MagicMock()

        manager.update_memory_outcomes(mock_result)

        mock_continuum.update_outcome.assert_not_called()

    def test_update_outcomes_success(self):
        """update_memory_outcomes updates retrieved memories."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        manager.track_retrieved_ids(["mem1", "mem2"])

        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.8
        mock_result.id = "debate_123"

        manager.update_memory_outcomes(mock_result)

        assert mock_continuum.update_outcome.call_count == 2

    def test_update_outcomes_clears_ids(self):
        """update_memory_outcomes clears tracked IDs after update."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)
        manager.track_retrieved_ids(["mem1"])

        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.8
        mock_result.id = "debate_123"

        manager.update_memory_outcomes(mock_result)

        assert manager._retrieved_ids == []


class TestMemoryEventType:
    """Tests for MemoryEventType constants."""

    def test_memory_stored_constant(self):
        """MEMORY_STORED constant is correct."""
        assert MemoryEventType.MEMORY_STORED == "memory:stored"

    def test_memory_retrieved_constant(self):
        """MEMORY_RETRIEVED constant is correct."""
        assert MemoryEventType.MEMORY_RETRIEVED == "memory:retrieved"

    def test_pattern_cached_constant(self):
        """PATTERN_CACHED constant is correct."""
        assert MemoryEventType.PATTERN_CACHED == "pattern:cached"


class TestMemoryManagerFetchHistoricalContext:
    """Tests for fetching historical context."""

    @pytest.mark.asyncio
    async def test_fetch_without_embeddings(self):
        """fetch_historical_context returns empty without debate_embeddings."""
        manager = MemoryManager()

        result = await manager.fetch_historical_context("Test task")

        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_with_no_results(self):
        """fetch_historical_context returns empty when no similar debates."""
        mock_embeddings = MagicMock()
        mock_embeddings.find_similar_debates = AsyncMock(return_value=[])
        manager = MemoryManager(debate_embeddings=mock_embeddings)

        result = await manager.fetch_historical_context("Test task")

        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_formats_results(self):
        """fetch_historical_context formats similar debates."""
        mock_embeddings = MagicMock()
        mock_embeddings.find_similar_debates = AsyncMock(
            return_value=[
                ("debate1", "Similar debate about testing", 0.85),
                ("debate2", "Another related debate", 0.72),
            ]
        )
        manager = MemoryManager(debate_embeddings=mock_embeddings)

        result = await manager.fetch_historical_context("Test task")

        assert "HISTORICAL CONTEXT" in result
        assert "85% similar" in result
        assert "72% similar" in result


class TestMemoryManagerStoreEvidence:
    """Tests for storing evidence."""

    def test_store_evidence_without_memory(self):
        """store_evidence does nothing without continuum_memory."""
        manager = MemoryManager()

        # Should not raise
        manager.store_evidence([MagicMock()], "Test task")

    def test_store_evidence_empty_list(self):
        """store_evidence does nothing with empty list."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        manager.store_evidence([], "Test task")

        mock_continuum.add.assert_not_called()

    def test_store_evidence_filters_short_snippets(self):
        """store_evidence skips snippets under 50 chars."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        short_snippet = MagicMock()
        short_snippet.content = "Too short"
        short_snippet.source = "test"
        short_snippet.relevance = 0.8

        manager.store_evidence([short_snippet], "Test task")

        mock_continuum.add.assert_not_called()

    def test_store_evidence_success(self):
        """store_evidence stores valid snippets."""
        mock_continuum = MagicMock()
        manager = MemoryManager(continuum_memory=mock_continuum)

        snippet = MagicMock()
        snippet.content = "This is a sufficiently long evidence snippet that should be stored in the memory system for future retrieval."
        snippet.source = "web_search"
        snippet.relevance = 0.8

        manager.store_evidence([snippet], "Test task")

        mock_continuum.add.assert_called_once()
