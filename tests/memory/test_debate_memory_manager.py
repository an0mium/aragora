"""
Tests for aragora.debate.memory_manager module.

Tests MemoryManager class which handles storage and retrieval of debate outcomes,
evidence, and patterns across ContinuumMemory, CritiqueStore, and DebateEmbeddings.
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.debate.memory_manager import MemoryManager
from aragora.memory.continuum import MemoryTier


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_continuum_memory():
    """Create a mock ContinuumMemory."""
    memory = MagicMock()
    memory.add = MagicMock()
    memory.update_outcome = MagicMock()
    return memory


@pytest.fixture
def mock_critique_store():
    """Create a mock CritiqueStore."""
    store = MagicMock()
    store.retrieve_patterns = MagicMock(return_value=[])
    return store


@pytest.fixture
def mock_consensus_memory():
    """Create a mock ConsensusMemory."""
    memory = MagicMock()
    mock_record = MagicMock()
    mock_record.id = "consensus-123"
    memory.store_consensus = MagicMock(return_value=mock_record)
    memory.store_dissent = MagicMock()
    return memory


@pytest.fixture
def mock_debate_embeddings():
    """Create a mock DebateEmbeddingsDatabase."""
    embeddings = MagicMock()
    embeddings.find_similar_debates = AsyncMock(return_value=[])
    return embeddings


@pytest.fixture
def mock_event_emitter():
    """Create a mock event emitter."""
    emitter = MagicMock()
    emitter.emit = MagicMock()
    return emitter


@pytest.fixture
def mock_spectator():
    """Create a mock spectator stream."""
    spectator = MagicMock()
    spectator.emit = MagicMock()
    return spectator


@pytest.fixture
def mock_debate_result():
    """Create a mock DebateResult."""
    result = MagicMock()
    result.id = "debate-123"
    result.final_answer = "The answer is 42"
    result.confidence = 0.85
    result.consensus_reached = True
    result.rounds_used = 3
    result.winner = "claude"
    result.messages = []
    result.votes = []
    return result


@pytest.fixture
def memory_manager(
    mock_continuum_memory,
    mock_critique_store,
    mock_consensus_memory,
    mock_debate_embeddings,
    mock_event_emitter,
    mock_spectator,
):
    """Create a MemoryManager with all mocked dependencies."""
    return MemoryManager(
        continuum_memory=mock_continuum_memory,
        critique_store=mock_critique_store,
        consensus_memory=mock_consensus_memory,
        debate_embeddings=mock_debate_embeddings,
        event_emitter=mock_event_emitter,
        spectator=mock_spectator,
        loop_id="test-loop-123",
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    def test_basic_init(self):
        """Test basic initialization with no dependencies."""
        manager = MemoryManager()
        assert manager.continuum_memory is None
        assert manager.critique_store is None
        assert manager.consensus_memory is None
        assert manager.debate_embeddings is None
        assert manager.loop_id == ""

    def test_init_with_continuum_memory(self, mock_continuum_memory):
        """Test initialization with ContinuumMemory."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        assert manager.continuum_memory == mock_continuum_memory

    def test_init_with_critique_store(self, mock_critique_store):
        """Test initialization with CritiqueStore."""
        manager = MemoryManager(critique_store=mock_critique_store)
        assert manager.critique_store == mock_critique_store

    def test_init_with_consensus_memory(self, mock_consensus_memory):
        """Test initialization with ConsensusMemory."""
        manager = MemoryManager(consensus_memory=mock_consensus_memory)
        assert manager.consensus_memory == mock_consensus_memory

    def test_init_with_debate_embeddings(self, mock_debate_embeddings):
        """Test initialization with DebateEmbeddingsDatabase."""
        manager = MemoryManager(debate_embeddings=mock_debate_embeddings)
        assert manager.debate_embeddings == mock_debate_embeddings

    def test_init_with_domain_extractor(self):
        """Test initialization with domain extractor callable."""

        def extractor():
            return "testing"

        manager = MemoryManager(domain_extractor=extractor)
        assert manager._domain_extractor == extractor

    def test_init_with_event_emitter(self, mock_event_emitter):
        """Test initialization with event emitter."""
        manager = MemoryManager(event_emitter=mock_event_emitter)
        assert manager.event_emitter == mock_event_emitter

    def test_init_with_spectator(self, mock_spectator):
        """Test initialization with spectator stream."""
        manager = MemoryManager(spectator=mock_spectator)
        assert manager.spectator == mock_spectator

    def test_init_with_loop_id(self):
        """Test initialization with loop ID."""
        manager = MemoryManager(loop_id="my-loop-123")
        assert manager.loop_id == "my-loop-123"

    def test_init_with_tier_analytics_tracker(self):
        """Test initialization with TierAnalyticsTracker."""
        mock_tracker = MagicMock()
        manager = MemoryManager(tier_analytics_tracker=mock_tracker)
        assert manager.tier_analytics_tracker == mock_tracker

    def test_init_state(self):
        """Test initial state of internal variables."""
        manager = MemoryManager()
        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}
        assert manager._patterns_cache is None
        assert manager._patterns_cache_ttl == 300.0


# ============================================================================
# Domain Extraction Tests
# ============================================================================


class TestDomainExtraction:
    """Tests for domain extraction."""

    def test_get_domain_default(self):
        """Test default domain is 'general'."""
        manager = MemoryManager()
        assert manager._get_domain() == "general"

    def test_get_domain_with_extractor(self):
        """Test domain extraction with custom extractor."""
        manager = MemoryManager(domain_extractor=lambda: "security")
        assert manager._get_domain() == "security"

    def test_get_domain_extractor_returns_different_values(self):
        """Test domain extractor can return different values."""
        domains = ["security", "testing", "architecture"]
        idx = [0]

        def extractor():
            result = domains[idx[0] % len(domains)]
            idx[0] += 1
            return result

        manager = MemoryManager(domain_extractor=extractor)
        assert manager._get_domain() == "security"
        assert manager._get_domain() == "testing"
        assert manager._get_domain() == "architecture"


# ============================================================================
# Store Debate Outcome Tests
# ============================================================================


class TestStoreDebateOutcome:
    """Tests for store_debate_outcome method."""

    def test_store_outcome_no_continuum(self, mock_debate_result):
        """Test store_debate_outcome does nothing without continuum_memory."""
        manager = MemoryManager()
        manager.store_debate_outcome(mock_debate_result, "test task")
        # No exception should be raised

    def test_store_outcome_no_final_answer(self, mock_continuum_memory):
        """Test store_debate_outcome skips if no final_answer."""
        result = MagicMock()
        result.final_answer = ""

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_debate_outcome(result, "test task")

        mock_continuum_memory.add.assert_not_called()

    def test_store_outcome_success(self, mock_continuum_memory, mock_debate_result):
        """Test successful outcome storage."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_debate_outcome(mock_debate_result, "test task")

        mock_continuum_memory.add.assert_called_once()
        call_args = mock_continuum_memory.add.call_args
        assert "debate_outcome_" in call_args.kwargs.get("id", call_args[1].get("id", ""))

    def test_store_outcome_tier_selection_high_quality(self, mock_continuum_memory):
        """Test tier selection for high-quality debates (FAST tier)."""
        result = MagicMock()
        result.id = "test-123"
        result.final_answer = "High quality answer"
        result.confidence = 0.9
        result.consensus_reached = True
        result.rounds_used = 3
        result.winner = "claude"

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_debate_outcome(result, "test task")

        call_args = mock_continuum_memory.add.call_args
        assert call_args.kwargs.get("tier") == MemoryTier.FAST

    def test_store_outcome_tier_selection_medium_quality(self, mock_continuum_memory):
        """Test tier selection for medium-quality debates (MEDIUM tier)."""
        result = MagicMock()
        result.id = "test-123"
        result.final_answer = "Medium quality answer"
        result.confidence = 0.6
        result.consensus_reached = False
        result.rounds_used = 1
        result.winner = None

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_debate_outcome(result, "test task")

        call_args = mock_continuum_memory.add.call_args
        assert call_args.kwargs.get("tier") == MemoryTier.MEDIUM

    def test_store_outcome_tier_selection_low_quality(self, mock_continuum_memory):
        """Test tier selection for low-quality debates (SLOW tier)."""
        result = MagicMock()
        result.id = "test-123"
        result.final_answer = "Low quality answer"
        result.confidence = 0.3
        result.consensus_reached = False
        result.rounds_used = 0
        result.winner = None

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_debate_outcome(result, "test task")

        call_args = mock_continuum_memory.add.call_args
        assert call_args.kwargs.get("tier") == MemoryTier.SLOW

    def test_store_outcome_with_belief_cruxes(self, mock_continuum_memory, mock_debate_result):
        """Test storing outcome with belief cruxes."""
        cruxes = ["crux1", "crux2", "crux3"]
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_debate_outcome(mock_debate_result, "test task", belief_cruxes=cruxes)

        call_args = mock_continuum_memory.add.call_args
        metadata = call_args.kwargs.get("metadata", {})
        assert "crux_claims" in metadata
        assert metadata["crux_claims"] == cruxes

    def test_store_outcome_importance_calculation(self, mock_continuum_memory):
        """Test importance calculation."""
        result = MagicMock()
        result.id = "test-123"
        result.final_answer = "Answer"
        result.confidence = 0.8
        result.consensus_reached = True
        result.rounds_used = 2
        result.winner = "claude"

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_debate_outcome(result, "test task")

        call_args = mock_continuum_memory.add.call_args
        importance = call_args.kwargs.get("importance", 0)
        # Importance should be boosted for consensus_reached
        assert importance > 0.8

    def test_store_outcome_handles_exception(self, mock_continuum_memory, mock_debate_result):
        """Test exception handling in store_debate_outcome."""
        mock_continuum_memory.add.side_effect = Exception("Storage error")

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        # Should not raise
        manager.store_debate_outcome(mock_debate_result, "test task")


# ============================================================================
# Store Consensus Record Tests
# ============================================================================


class TestStoreConsensusRecord:
    """Tests for store_consensus_record method."""

    def test_store_consensus_no_memory(self, mock_debate_result):
        """Test store_consensus_record does nothing without consensus_memory."""
        manager = MemoryManager()
        manager.store_consensus_record(mock_debate_result, "test task")
        # No exception should be raised

    def test_store_consensus_no_final_answer(self, mock_consensus_memory):
        """Test store_consensus_record skips if no final_answer."""
        result = MagicMock()
        result.final_answer = ""

        manager = MemoryManager(consensus_memory=mock_consensus_memory)
        manager.store_consensus_record(result, "test task")

        mock_consensus_memory.store_consensus.assert_not_called()

    def test_store_consensus_success(self, mock_consensus_memory, mock_debate_result):
        """Test successful consensus storage."""
        manager = MemoryManager(consensus_memory=mock_consensus_memory)
        manager.store_consensus_record(mock_debate_result, "test task")

        mock_consensus_memory.store_consensus.assert_called_once()


# ============================================================================
# Confidence to Strength Tests
# ============================================================================


class TestConfidenceToStrength:
    """Tests for _confidence_to_strength helper."""

    def test_unanimous_strength(self, memory_manager):
        """Test UNANIMOUS strength for confidence >= 0.95."""
        from aragora.memory.consensus import ConsensusStrength

        assert memory_manager._confidence_to_strength(0.95) == ConsensusStrength.UNANIMOUS
        assert memory_manager._confidence_to_strength(0.99) == ConsensusStrength.UNANIMOUS
        assert memory_manager._confidence_to_strength(1.0) == ConsensusStrength.UNANIMOUS

    def test_strong_strength(self, memory_manager):
        """Test STRONG strength for confidence 0.8-0.95."""
        from aragora.memory.consensus import ConsensusStrength

        assert memory_manager._confidence_to_strength(0.8) == ConsensusStrength.STRONG
        assert memory_manager._confidence_to_strength(0.9) == ConsensusStrength.STRONG
        assert memory_manager._confidence_to_strength(0.94) == ConsensusStrength.STRONG

    def test_moderate_strength(self, memory_manager):
        """Test MODERATE strength for confidence 0.6-0.8."""
        from aragora.memory.consensus import ConsensusStrength

        assert memory_manager._confidence_to_strength(0.6) == ConsensusStrength.MODERATE
        assert memory_manager._confidence_to_strength(0.7) == ConsensusStrength.MODERATE
        assert memory_manager._confidence_to_strength(0.79) == ConsensusStrength.MODERATE

    def test_weak_strength(self, memory_manager):
        """Test WEAK strength for confidence 0.5-0.6."""
        from aragora.memory.consensus import ConsensusStrength

        assert memory_manager._confidence_to_strength(0.5) == ConsensusStrength.WEAK
        assert memory_manager._confidence_to_strength(0.55) == ConsensusStrength.WEAK

    def test_split_strength(self, memory_manager):
        """Test SPLIT strength for confidence 0.3-0.5."""
        from aragora.memory.consensus import ConsensusStrength

        assert memory_manager._confidence_to_strength(0.3) == ConsensusStrength.SPLIT
        assert memory_manager._confidence_to_strength(0.4) == ConsensusStrength.SPLIT

    def test_contested_strength(self, memory_manager):
        """Test CONTESTED strength for confidence < 0.3."""
        from aragora.memory.consensus import ConsensusStrength

        assert memory_manager._confidence_to_strength(0.1) == ConsensusStrength.CONTESTED
        assert memory_manager._confidence_to_strength(0.0) == ConsensusStrength.CONTESTED
        assert memory_manager._confidence_to_strength(0.29) == ConsensusStrength.CONTESTED


# ============================================================================
# Store Evidence Tests
# ============================================================================


class TestStoreEvidence:
    """Tests for store_evidence method."""

    def test_store_evidence_no_continuum(self):
        """Test store_evidence does nothing without continuum_memory."""
        manager = MemoryManager()
        manager.store_evidence([MagicMock()], "test task")
        # No exception should be raised

    def test_store_evidence_empty_list(self, mock_continuum_memory):
        """Test store_evidence does nothing with empty list."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_evidence([], "test task")
        mock_continuum_memory.add.assert_not_called()

    def test_store_evidence_success(self, mock_continuum_memory):
        """Test successful evidence storage."""
        snippet = MagicMock()
        snippet.content = "A" * 100  # Long enough content
        snippet.source = "web"
        snippet.relevance = 0.8

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_evidence([snippet], "test task")

        mock_continuum_memory.add.assert_called_once()

    def test_store_evidence_skips_short_content(self, mock_continuum_memory):
        """Test store_evidence skips content shorter than 50 chars."""
        snippet = MagicMock()
        snippet.content = "Short"  # Too short
        snippet.source = "web"
        snippet.relevance = 0.8

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_evidence([snippet], "test task")

        mock_continuum_memory.add.assert_not_called()

    def test_store_evidence_limits_to_10(self, mock_continuum_memory):
        """Test store_evidence limits to 10 snippets."""
        snippets = []
        for i in range(15):
            snippet = MagicMock()
            snippet.content = f"Content {i}" + "x" * 50
            snippet.source = f"source-{i}"
            snippet.relevance = 0.5
            snippets.append(snippet)

        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.store_evidence(snippets, "test task")

        # Should store at most 10
        assert mock_continuum_memory.add.call_count <= 10

    def test_store_evidence_emits_event(self, mock_continuum_memory, mock_event_emitter):
        """Test store_evidence emits EVIDENCE_FOUND event."""
        snippet = MagicMock()
        snippet.content = "A" * 100
        snippet.source = "web"
        snippet.relevance = 0.8

        manager = MemoryManager(
            continuum_memory=mock_continuum_memory,
            event_emitter=mock_event_emitter,
            loop_id="test-loop",
        )
        manager.store_evidence([snippet], "test task")

        mock_event_emitter.emit.assert_called()


# ============================================================================
# Update Memory Outcomes Tests
# ============================================================================


class TestUpdateMemoryOutcomes:
    """Tests for update_memory_outcomes method."""

    def test_update_outcomes_no_continuum(self, mock_debate_result):
        """Test update_memory_outcomes does nothing without continuum_memory."""
        manager = MemoryManager()
        manager._retrieved_ids = ["id1", "id2"]
        manager.update_memory_outcomes(mock_debate_result)
        # No exception should be raised

    def test_update_outcomes_no_retrieved_ids(self, mock_continuum_memory, mock_debate_result):
        """Test update_memory_outcomes does nothing without retrieved IDs."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager.update_memory_outcomes(mock_debate_result)
        mock_continuum_memory.update_outcome.assert_not_called()

    def test_update_outcomes_success(self, mock_continuum_memory, mock_debate_result):
        """Test successful memory outcome update."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager._retrieved_ids = ["mem-1", "mem-2"]
        manager.update_memory_outcomes(mock_debate_result)

        assert mock_continuum_memory.update_outcome.call_count == 2

    def test_update_outcomes_clears_ids(self, mock_continuum_memory, mock_debate_result):
        """Test update_memory_outcomes clears retrieved IDs after update."""
        manager = MemoryManager(continuum_memory=mock_continuum_memory)
        manager._retrieved_ids = ["mem-1"]
        manager._retrieved_tiers = {"mem-1": MemoryTier.FAST}

        manager.update_memory_outcomes(mock_debate_result)

        assert manager._retrieved_ids == []
        assert manager._retrieved_tiers == {}

    def test_update_outcomes_with_tier_analytics(self, mock_continuum_memory, mock_debate_result):
        """Test update_memory_outcomes records tier analytics."""
        mock_tracker = MagicMock()
        manager = MemoryManager(
            continuum_memory=mock_continuum_memory,
            tier_analytics_tracker=mock_tracker,
        )
        manager._retrieved_ids = ["mem-1"]
        manager._retrieved_tiers = {"mem-1": MemoryTier.FAST}

        manager.update_memory_outcomes(mock_debate_result)

        mock_tracker.record_usage.assert_called_once()


# ============================================================================
# Fetch Historical Context Tests
# ============================================================================


class TestFetchHistoricalContext:
    """Tests for fetch_historical_context method."""

    @pytest.mark.asyncio
    async def test_fetch_context_no_embeddings(self):
        """Test fetch_historical_context returns empty without embeddings."""
        manager = MemoryManager()
        result = await manager.fetch_historical_context("test task")
        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_context_no_results(self, mock_debate_embeddings):
        """Test fetch_historical_context returns empty when no similar debates."""
        mock_debate_embeddings.find_similar_debates = AsyncMock(return_value=[])

        manager = MemoryManager(debate_embeddings=mock_debate_embeddings)
        result = await manager.fetch_historical_context("test task")

        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_context_success(
        self, mock_debate_embeddings, mock_spectator, mock_event_emitter
    ):
        """Test successful historical context retrieval."""
        mock_debate_embeddings.find_similar_debates = AsyncMock(
            return_value=[
                ("debate-1", "Similar debate about testing", 0.85),
                ("debate-2", "Another related debate", 0.75),
            ]
        )

        manager = MemoryManager(
            debate_embeddings=mock_debate_embeddings,
            spectator=mock_spectator,
            event_emitter=mock_event_emitter,
            loop_id="test-loop",
        )
        result = await manager.fetch_historical_context("test task")

        assert "HISTORICAL CONTEXT" in result
        assert "85%" in result
        assert "Similar debate about testing" in result

    @pytest.mark.asyncio
    async def test_fetch_context_notifies_spectator(self, mock_debate_embeddings, mock_spectator):
        """Test fetch_historical_context notifies spectator."""
        mock_debate_embeddings.find_similar_debates = AsyncMock(
            return_value=[("debate-1", "Test", 0.9)]
        )

        manager = MemoryManager(
            debate_embeddings=mock_debate_embeddings,
            spectator=mock_spectator,
        )
        await manager.fetch_historical_context("test task")

        mock_spectator.emit.assert_called()


# ============================================================================
# Get Successful Patterns Tests
# ============================================================================


class TestGetSuccessfulPatterns:
    """Tests for get_successful_patterns method."""

    def test_get_patterns_no_critique_store(self):
        """Test get_successful_patterns returns empty without critique_store."""
        manager = MemoryManager()
        result = manager.get_successful_patterns()
        assert result == ""

    def test_get_patterns_no_results(self, mock_critique_store):
        """Test get_successful_patterns returns empty when no patterns."""
        mock_critique_store.retrieve_patterns = MagicMock(return_value=[])

        manager = MemoryManager(critique_store=mock_critique_store)
        result = manager.get_successful_patterns()

        assert result == ""

    def test_get_patterns_success(self, mock_critique_store):
        """Test successful pattern retrieval."""
        mock_pattern = MagicMock()
        mock_pattern.issue_type = "logic"
        mock_pattern.issue_text = "Circular reasoning"
        mock_pattern.suggestion_text = "Break the logical loop"
        mock_pattern.success_count = 5
        mock_pattern.avg_severity = 0.7

        mock_critique_store.retrieve_patterns = MagicMock(return_value=[mock_pattern])

        manager = MemoryManager(critique_store=mock_critique_store)
        result = manager.get_successful_patterns()

        assert "LEARNED PATTERNS" in result
        assert "LOGIC" in result
        assert "HIGH SEVERITY" in result

    def test_get_patterns_uses_cache(self, mock_critique_store):
        """Test get_successful_patterns uses cache."""
        mock_critique_store.retrieve_patterns = MagicMock(return_value=[])

        manager = MemoryManager(critique_store=mock_critique_store)

        # First call
        manager.get_successful_patterns()
        # Second call
        manager.get_successful_patterns()

        # Should only call retrieve_patterns once due to caching
        assert mock_critique_store.retrieve_patterns.call_count == 1

    def test_get_patterns_cache_expires(self, mock_critique_store):
        """Test pattern cache expires after TTL."""
        mock_critique_store.retrieve_patterns = MagicMock(return_value=[])

        manager = MemoryManager(critique_store=mock_critique_store)
        manager._patterns_cache_ttl = 0.001  # Very short TTL

        manager.get_successful_patterns()
        time.sleep(0.002)  # Wait for cache to expire
        manager.get_successful_patterns()

        # Should call retrieve_patterns twice due to cache expiry
        assert mock_critique_store.retrieve_patterns.call_count == 2


# ============================================================================
# Format Patterns Tests
# ============================================================================


class TestFormatPatternsForPrompt:
    """Tests for _format_patterns_for_prompt method."""

    def test_format_empty_patterns(self, memory_manager):
        """Test formatting empty patterns list."""
        result = memory_manager._format_patterns_for_prompt([])
        assert result == ""

    def test_format_single_pattern(self, memory_manager):
        """Test formatting single pattern."""
        patterns = [{"category": "logic", "pattern": "test pattern", "occurrences": 3}]
        result = memory_manager._format_patterns_for_prompt(patterns)

        assert "LEARNED PATTERNS" in result
        assert "LOGIC" in result
        assert "test pattern" in result
        assert "3 past debates" in result

    def test_format_high_severity_pattern(self, memory_manager):
        """Test formatting high severity pattern."""
        patterns = [
            {
                "category": "security",
                "pattern": "SQL injection",
                "occurrences": 5,
                "avg_severity": 0.9,
            }
        ]
        result = memory_manager._format_patterns_for_prompt(patterns)

        assert "[HIGH SEVERITY]" in result

    def test_format_medium_severity_pattern(self, memory_manager):
        """Test formatting medium severity pattern."""
        patterns = [
            {
                "category": "style",
                "pattern": "inconsistent naming",
                "occurrences": 2,
                "avg_severity": 0.5,
            }
        ]
        result = memory_manager._format_patterns_for_prompt(patterns)

        assert "[MEDIUM]" in result

    def test_format_limits_to_5(self, memory_manager):
        """Test formatting limits to 5 patterns."""
        patterns = [
            {"category": f"cat-{i}", "pattern": f"pattern {i}", "occurrences": i} for i in range(10)
        ]
        result = memory_manager._format_patterns_for_prompt(patterns)

        # Count occurrences of "cat-" to verify limit
        assert result.count("CAT-") <= 5


# ============================================================================
# Track Retrieved IDs Tests
# ============================================================================


class TestTrackRetrievedIds:
    """Tests for track_retrieved_ids method."""

    def test_track_ids(self, memory_manager):
        """Test tracking retrieved IDs."""
        memory_manager.track_retrieved_ids(["id1", "id2", "id3"])
        assert memory_manager._retrieved_ids == ["id1", "id2", "id3"]

    def test_track_ids_filters_empty(self, memory_manager):
        """Test tracking filters out empty IDs."""
        memory_manager.track_retrieved_ids(["id1", "", "id3", None])
        assert memory_manager._retrieved_ids == ["id1", "id3"]

    def test_track_ids_with_tiers(self, memory_manager):
        """Test tracking IDs with tier information."""
        tiers = {"id1": MemoryTier.FAST, "id2": MemoryTier.SLOW}
        memory_manager.track_retrieved_ids(["id1", "id2"], tiers=tiers)

        assert memory_manager._retrieved_ids == ["id1", "id2"]
        assert memory_manager._retrieved_tiers == tiers

    def test_track_ids_replaces_previous(self, memory_manager):
        """Test tracking replaces previous IDs."""
        memory_manager.track_retrieved_ids(["old1", "old2"])
        memory_manager.track_retrieved_ids(["new1"])

        assert memory_manager._retrieved_ids == ["new1"]


# ============================================================================
# Clear Retrieved IDs Tests
# ============================================================================


class TestClearRetrievedIds:
    """Tests for clear_retrieved_ids method."""

    def test_clear_ids(self, memory_manager):
        """Test clearing retrieved IDs."""
        memory_manager._retrieved_ids = ["id1", "id2"]
        memory_manager._retrieved_tiers = {"id1": MemoryTier.FAST}

        memory_manager.clear_retrieved_ids()

        assert memory_manager._retrieved_ids == []
        assert memory_manager._retrieved_tiers == {}


# ============================================================================
# Retrieved IDs Property Tests
# ============================================================================


class TestRetrievedIdsProperty:
    """Tests for retrieved_ids property."""

    def test_retrieved_ids_property(self, memory_manager):
        """Test retrieved_ids property returns copy."""
        memory_manager._retrieved_ids = ["id1", "id2"]
        ids = memory_manager.retrieved_ids

        assert ids == ["id1", "id2"]
        # Verify it's a copy
        ids.append("id3")
        assert memory_manager._retrieved_ids == ["id1", "id2"]


# ============================================================================
# Spectator Notification Tests
# ============================================================================


class TestNotifySpectator:
    """Tests for _notify_spectator method."""

    def test_notify_spectator(self, memory_manager, mock_spectator):
        """Test spectator notification."""
        memory_manager._notify_spectator("test_event", "test details", metric=0.5)
        mock_spectator.emit.assert_called_once()

    def test_notify_spectator_no_spectator(self):
        """Test notification does nothing without spectator."""
        manager = MemoryManager()
        # Should not raise
        manager._notify_spectator("test_event", "test details")

    def test_notify_spectator_handles_exception(self, memory_manager, mock_spectator):
        """Test notification handles exceptions."""
        mock_spectator.emit.side_effect = Exception("Emit error")
        # Should not raise
        memory_manager._notify_spectator("test_event", "test details")
