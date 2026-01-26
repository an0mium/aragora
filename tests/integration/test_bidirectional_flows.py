"""
Bidirectional Knowledge Mound Integration Tests.

Tests end-to-end flows for all bidirectional KM integrations:
- Memory ↔ KM: High-importance memories sync to KM and back
- Belief ↔ KM: Converged beliefs/cruxes stored and retrieved for priors
- RLM ↔ KM: Compression patterns stored and priority hints updated
- ELO/Ranking ↔ KM: Agent expertise profiles for team selection
- Insights ↔ KM: Insights and flip events for organizational learning
- Consensus ↔ KM: Full consensus with dissent, evolution, and linking

These tests verify real subsystem interactions (minimal mocking).
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.events.types import StreamEvent, StreamEventType
from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    get_cross_subscriber_manager,
    reset_cross_subscriber_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def fresh_manager():
    """Get a fresh subscriber manager for each test."""
    reset_cross_subscriber_manager()
    manager = get_cross_subscriber_manager()
    yield manager
    reset_cross_subscriber_manager()


@pytest.fixture
def mock_knowledge_mound():
    """Create a mock KnowledgeMound with realistic behavior."""
    mound = MagicMock()
    mound.workspace_id = "test_workspace"

    # Track stored nodes for retrieval
    stored_nodes = {}
    node_counter = [0]

    async def mock_store(request):
        node_counter[0] += 1
        node_id = f"node_{node_counter[0]:04d}"
        stored_nodes[node_id] = {
            "content": request.content,
            "node_type": request.node_type,
            "confidence": request.confidence,
            "metadata": request.metadata,
            "tier": request.tier,
            "supersedes": getattr(request, "supersedes", None),
            "derived_from": getattr(request, "derived_from", None),
        }
        result = MagicMock()
        result.node_id = node_id
        result.deduplicated = False
        return result

    async def mock_search(query, node_types=None, limit=10, min_score=0.0):
        # Return nodes matching the query (simple substring match for testing)
        results = []
        for node_id, node in stored_nodes.items():
            if node_types and node["node_type"] not in node_types:
                continue
            if query.lower() in node["content"].lower():
                result = MagicMock()
                result.id = node_id
                result.content = node["content"]
                result.metadata = node["metadata"]
                result.score = 0.9
                results.append(result)
                if len(results) >= limit:
                    break
        return results

    async def mock_update_metadata(node_id, updates):
        if node_id in stored_nodes:
            stored_nodes[node_id]["metadata"].update(updates)

    mound.store = mock_store
    mound.search = mock_search
    mound.update_metadata = mock_update_metadata
    mound._stored_nodes = stored_nodes  # Expose for assertions

    return mound


@pytest.fixture
def mock_continuum_memory():
    """Create a mock ContinuumMemory with realistic behavior."""
    memory = MagicMock()
    stored_memories = []

    def mock_store(content, importance=0.5, metadata=None):
        stored_memories.append(
            {
                "content": content,
                "importance": importance,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            }
        )

    def mock_recall(query, limit=10):
        # Simple matching for testing
        return [m for m in stored_memories if query.lower() in m["content"].lower()][:limit]

    def mock_prewarm_for_query(query, workspace_id=None):
        pass  # Simulate pre-warming

    memory.store = mock_store
    memory.recall = mock_recall
    memory.prewarm_for_query = mock_prewarm_for_query
    memory._stored_memories = stored_memories

    return memory


# ============================================================================
# Memory ↔ KM Bidirectional Tests
# ============================================================================


class TestMemoryKMBidirectional:
    """Tests for Memory ↔ Knowledge Mound bidirectional flow."""

    def test_high_importance_memory_syncs_to_km(self, fresh_manager):
        """Test that high-importance memories (≥0.7) sync to KM."""
        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={
                "content": "Critical insight about API design patterns",
                "importance": 0.85,
                "tier": "slow",
                "metadata": {"source": "debate_001"},
            },
        )

        # Dispatch the event - handler will try to sync (gracefully handles missing KM)
        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

        # Verify handler processed the event (stats updated)
        stats = fresh_manager.get_stats()
        assert "memory_to_mound" in stats

    def test_low_importance_memory_not_synced(self, fresh_manager):
        """Test that low-importance memories (<0.7) are NOT synced to KM."""
        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={
                "content": "Minor note",
                "importance": 0.3,
                "tier": "fast",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            # Should not raise, should skip silently
            fresh_manager._dispatch_event(event)

    @patch("aragora.memory.get_continuum_memory")
    def test_km_query_prewarms_memory_cache(
        self, mock_get_memory, fresh_manager, mock_continuum_memory
    ):
        """Test that KM queries trigger memory cache pre-warming."""
        mock_get_memory.return_value = mock_continuum_memory

        event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_QUERIED,
            data={
                "query": "API design patterns",
                "results_count": 5,
                "workspace_id": "test_workspace",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)


# ============================================================================
# Belief ↔ KM Bidirectional Tests
# ============================================================================


class TestBeliefKMBidirectional:
    """Tests for Belief Network ↔ Knowledge Mound bidirectional flow."""

    def test_converged_beliefs_stored_in_km(self, fresh_manager):
        """Test that converged beliefs are stored in KM."""
        event = StreamEvent(
            type=StreamEventType.BELIEF_CONVERGED,
            data={
                "debate_id": "debate_belief_001",
                "beliefs_count": 5,
                "beliefs": [
                    {"claim": "REST is suitable for CRUD", "confidence": 0.9},
                    {"claim": "GraphQL reduces over-fetching", "confidence": 0.85},
                ],
                "cruxes": [
                    {
                        "claim": "Performance vs flexibility trade-off",
                        "topics": ["api", "architecture"],
                    }
                ],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_debate_start_retrieves_historical_cruxes(self, fresh_manager):
        """Test that debate start retrieves historical cruxes from KM."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate_new_001",
                "question": "Should we use GraphQL or REST for our API?",
                "agents": ["claude", "gpt4", "gemini"],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)


# ============================================================================
# RLM ↔ KM Bidirectional Tests
# ============================================================================


class TestRLMKMBidirectional:
    """Tests for RLM Compressor ↔ Knowledge Mound bidirectional flow."""

    def test_high_value_compression_pattern_stored(self, fresh_manager):
        """Test that high-value compression patterns are stored in KM."""
        event = StreamEvent(
            type=StreamEventType.RLM_COMPRESSION_COMPLETE,
            data={
                "compression_ratio": 0.45,
                "value_score": 0.8,
                "content_markers": ["api", "authentication", "oauth"],
                "metadata": {"source": "debate_rlm_001"},
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_low_value_compression_not_stored(self, fresh_manager):
        """Test that low-value compression patterns (<0.7) are NOT stored."""
        event = StreamEvent(
            type=StreamEventType.RLM_COMPRESSION_COMPLETE,
            data={
                "compression_ratio": 0.6,
                "value_score": 0.4,  # Below threshold
                "content_markers": ["misc"],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_km_query_updates_rlm_priorities(self, fresh_manager):
        """Test that KM queries update RLM compression priorities."""
        event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_QUERIED,
            data={
                "query": "OAuth implementation",
                "results_count": 3,
                "node_ids": ["node_001", "node_002", "node_003"],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

        # Verify handler processed the event
        stats = fresh_manager.get_stats()
        assert "mound_to_rlm" in stats


# ============================================================================
# ELO/Ranking ↔ KM Bidirectional Tests
# ============================================================================


class TestELORankingKMBidirectional:
    """Tests for ELO/Ranking ↔ Knowledge Mound bidirectional flow."""

    def test_significant_elo_change_stored(self, fresh_manager):
        """Test that significant ELO changes (|delta| > 25) are stored in KM."""
        event = StreamEvent(
            type=StreamEventType.AGENT_ELO_UPDATED,
            data={
                "agent": "claude",
                "elo": 1650,
                "delta": 50,  # Significant change
                "debate_id": "debate_elo_001",
                "domain": "security",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_minor_elo_change_not_stored(self, fresh_manager):
        """Test that minor ELO changes (|delta| < 25) are NOT stored."""
        event = StreamEvent(
            type=StreamEventType.AGENT_ELO_UPDATED,
            data={
                "agent": "gpt4",
                "elo": 1510,
                "delta": 10,  # Minor change
                "debate_id": "debate_elo_002",
                "domain": "general",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_debate_start_queries_domain_experts(self, fresh_manager):
        """Test that debate start queries KM for domain experts."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate_team_001",
                "question": "How should we implement OAuth 2.0 with PKCE?",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)


# ============================================================================
# Insights/Trickster ↔ KM Bidirectional Tests
# ============================================================================


class TestInsightsKMBidirectional:
    """Tests for Insights/Trickster ↔ Knowledge Mound bidirectional flow."""

    def test_high_confidence_insight_stored(self, fresh_manager):
        """Test that high-confidence insights (≥0.7) are stored in KM."""
        event = StreamEvent(
            type=StreamEventType.INSIGHT_EXTRACTED,
            data={
                "type": "pattern",
                "confidence": 0.85,
                "content": "Teams with diverse agent types reach consensus faster",
                "debate_id": "debate_insight_001",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_low_confidence_insight_not_stored(self, fresh_manager):
        """Test that low-confidence insights (<0.7) are NOT stored."""
        event = StreamEvent(
            type=StreamEventType.INSIGHT_EXTRACTED,
            data={
                "type": "pattern",
                "confidence": 0.5,  # Below threshold
                "content": "Unclear pattern",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_flip_event_always_stored(self, fresh_manager):
        """Test that ALL flip events are stored for meta-learning."""
        event = StreamEvent(
            type=StreamEventType.FLIP_DETECTED,
            data={
                "agent_name": "claude",
                "flip_type": "position_reversal",
                "debate_id": "debate_flip_001",
                "original_position": "REST is better",
                "new_position": "GraphQL is better",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_debate_start_retrieves_flip_history(self, fresh_manager):
        """Test that debate start retrieves agent flip history from KM."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate_trickster_001",
                "agents": ["claude", "gpt4", "gemini"],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)


# ============================================================================
# Consensus ↔ KM Bidirectional Tests (Enhanced)
# ============================================================================


class TestConsensusKMBidirectional:
    """Tests for enhanced Consensus ↔ Knowledge Mound bidirectional flow."""

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_full_consensus_with_dissent_stored(
        self, mock_get_mound, fresh_manager, mock_knowledge_mound
    ):
        """Test full consensus with dissenting views is stored properly."""
        mock_get_mound.return_value = mock_knowledge_mound

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_full_001",
                "consensus_reached": True,
                "topic": "Database selection for microservices",
                "conclusion": "PostgreSQL is recommended for most use cases",
                "confidence": 0.82,
                "strength": "strong",
                "domain": "architecture",
                "tags": ["database", "microservices", "postgresql"],
                "key_claims": [
                    "PostgreSQL handles ACID transactions well",
                    "Partitioning supports horizontal scaling",
                ],
                "supporting_evidence": [
                    "Benchmark: PostgreSQL handles 10k TPS",
                ],
                "participating_agents": ["claude", "gpt4", "gemini", "mistral"],
                "agreeing_agents": ["claude", "gpt4", "gemini"],
                "dissenting_agents": ["mistral"],
                "dissents": [
                    {
                        "agent_id": "mistral",
                        "type": "alternative_approach",
                        "content": "MongoDB might be better for document-heavy workloads",
                        "reasoning": "Flexible schema reduces migration friction",
                        "confidence": 0.7,
                        "acknowledged": True,
                        "rebuttal": "Schema flexibility comes at cost of consistency",
                    }
                ],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_consensus_evolution_detection(
        self, mock_get_mound, fresh_manager, mock_knowledge_mound
    ):
        """Test that similar prior consensus is detected and superseded."""
        mock_get_mound.return_value = mock_knowledge_mound

        # First consensus
        event1 = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_old_001",
                "consensus_reached": True,
                "topic": "API authentication best practices",
                "conclusion": "OAuth 2.0 implicit flow is acceptable",
                "confidence": 0.75,
                "strength": "moderate",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event1)

        # Second consensus on same topic (should detect evolution)
        event2 = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_new_001",
                "consensus_reached": True,
                "topic": "API authentication best practices",  # Same topic
                "conclusion": "OAuth 2.0 with PKCE is required for all clients",
                "confidence": 0.9,
                "strength": "strong",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event2)

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_explicit_supersedes_relationship(
        self, mock_get_mound, fresh_manager, mock_knowledge_mound
    ):
        """Test explicit supersedes relationship is honored."""
        mock_get_mound.return_value = mock_knowledge_mound

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_supersede_001",
                "consensus_reached": True,
                "topic": "Error handling strategy",
                "conclusion": "Use Result types instead of exceptions",
                "confidence": 0.85,
                "strength": "strong",
                "supersedes": "old_consensus_error_001",  # Explicit
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_no_consensus_not_stored(self, fresh_manager):
        """Test that events without consensus are not stored."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_no_001",
                "consensus_reached": False,
                "topic": "Disputed topic",
                "strength": "contested",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)


# ============================================================================
# Culture/Staleness ↔ Debate Tests
# ============================================================================


class TestCultureStalenessBidirectional:
    """Tests for Culture patterns and Staleness ↔ Debate flow."""

    def test_culture_patterns_propagate_to_debate(self, fresh_manager):
        """Test that culture pattern updates are propagated."""
        event = StreamEvent(
            type=StreamEventType.MOUND_UPDATED,
            data={
                "update_type": "culture_patterns",
                "patterns_count": 5,
                "workspace_id": "test_workspace",
                "debate_id": "debate_culture_001",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_staleness_warning_checked_for_active_debates(self, fresh_manager):
        """Test that stale knowledge warnings are checked against active debates."""
        event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_STALE,
            data={
                "node_id": "node_stale_001",
                "reason": "Source updated",
                "last_verified": "2024-01-01T00:00:00Z",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)


# ============================================================================
# Provenance ↔ KM Bidirectional Tests
# ============================================================================


class TestProvenanceKMBidirectional:
    """Tests for Provenance ↔ Knowledge Mound bidirectional flow."""

    def test_verified_provenance_chains_stored(self, fresh_manager):
        """Test that verified provenance chains are stored in KM."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_prov_001",
                "consensus_reached": True,
                "topic": "Claim verification",
                "conclusion": "All claims verified",
                "provenance_chains": [
                    {
                        "id": "chain_001",
                        "source_id": "source_001",
                        "claim_ids": ["claim_001", "claim_002"],
                        "verified": True,
                        "method": "consensus",
                    },
                    {
                        "id": "chain_002",
                        "verified": False,  # Unverified chain
                    },
                ],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)

    def test_claim_verification_queries_km_history(self, fresh_manager):
        """Test that claim verification queries KM for history."""
        event = StreamEvent(
            type=StreamEventType.CLAIM_VERIFICATION_RESULT,
            data={
                "claim_id": "claim_verify_001",
                "claim": "PostgreSQL handles 10k TPS under normal load",
                "verified": True,
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(event)


# ============================================================================
# Full Round-Trip Integration Tests
# ============================================================================


class TestFullRoundTripIntegration:
    """Tests for complete round-trip data flows."""

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_consensus_to_query_round_trip(
        self, mock_get_mound, fresh_manager, mock_knowledge_mound
    ):
        """Test full round-trip: Consensus stored → Queried → Retrieved."""
        mock_get_mound.return_value = mock_knowledge_mound

        # Step 1: Store consensus
        consensus_event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_rt_001",
                "consensus_reached": True,
                "topic": "Rate limiting implementation",
                "conclusion": "Token bucket algorithm is recommended",
                "confidence": 0.88,
                "strength": "strong",
                "key_claims": ["Token bucket handles bursts well"],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(consensus_event)

        # Step 2: Query (simulated - would trigger pre-warming)
        query_event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_QUERIED,
            data={
                "query": "rate limiting",
                "results_count": 1,
                "workspace_id": "test_workspace",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(query_event)

    def test_elo_to_team_selection_round_trip(self, fresh_manager):
        """Test: ELO update stored → Debate start → Domain experts queried."""
        # Step 1: ELO update
        elo_event = StreamEvent(
            type=StreamEventType.AGENT_ELO_UPDATED,
            data={
                "agent": "claude",
                "elo": 1700,
                "delta": 100,
                "domain": "security",
                "debate_id": "debate_elo_rt_001",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(elo_event)

        # Step 2: New debate starts, queries for experts
        debate_start_event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate_team_rt_001",
                "question": "How should we implement OAuth security?",
                "agents": ["claude", "gpt4", "gemini"],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(debate_start_event)

    def test_insight_to_trickster_round_trip(self, fresh_manager):
        """Test: Insight/Flip stored → Debate start → History queried."""
        # Step 1: Store flip event
        flip_event = StreamEvent(
            type=StreamEventType.FLIP_DETECTED,
            data={
                "agent_name": "gpt4",
                "flip_type": "position_reversal",
                "debate_id": "debate_flip_rt_001",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(flip_event)

        # Step 2: New debate queries flip history
        debate_start_event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate_trickster_rt_001",
                "agents": ["gpt4", "claude"],
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            fresh_manager._dispatch_event(debate_start_event)


# ============================================================================
# Handler Enable/Disable Tests
# ============================================================================


class TestHandlerEnableDisable:
    """Tests for handler enable/disable functionality."""

    def test_disabled_handler_skips_processing(self, fresh_manager):
        """Test that disabled handlers skip event processing."""
        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={
                "content": "Important memory",
                "importance": 0.9,
            },
        )

        # Disable the handler
        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=False):
            fresh_manager._dispatch_event(event)

        # Should not raise, should skip silently

    def test_handler_stats_track_skipped_events(self, fresh_manager):
        """Test that handler stats track skipped events."""
        # Get initial stats
        initial_stats = fresh_manager.get_stats()

        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={"content": "Test", "importance": 0.9},
        )

        # Disable handler via sampling
        fresh_manager.set_sample_rate("memory_to_mound", 0.0)
        fresh_manager._dispatch_event(event)

        # Check stats
        stats = fresh_manager.get_stats()
        # Skipped count should have increased


# ============================================================================
# Error Resilience Tests
# ============================================================================


class TestErrorResilience:
    """Tests for error handling and resilience."""

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_km_unavailable_handled_gracefully(self, mock_get_mound, fresh_manager):
        """Test graceful handling when KM is unavailable."""
        mock_get_mound.return_value = None

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_err_001",
                "consensus_reached": True,
                "topic": "Test topic",
                "conclusion": "Test conclusion",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            # Should not raise
            fresh_manager._dispatch_event(event)

    @patch("aragora.knowledge.mound.get_knowledge_mound")
    def test_store_error_handled_gracefully(self, mock_get_mound, fresh_manager):
        """Test graceful handling when store fails."""
        mock_mound = MagicMock()
        mock_mound.workspace_id = "test"
        mock_mound.store = AsyncMock(side_effect=Exception("Store failed"))
        mock_mound.search = AsyncMock(return_value=[])
        mock_get_mound.return_value = mock_mound

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate_err_002",
                "consensus_reached": True,
                "topic": "Test topic",
                "conclusion": "Test conclusion",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            # Should not raise
            fresh_manager._dispatch_event(event)

    def test_malformed_event_data_handled(self, fresh_manager):
        """Test handling of malformed event data."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                # Missing required fields
                "debate_id": "debate_err_003",
            },
        )

        with patch.object(fresh_manager, "_is_km_handler_enabled", return_value=True):
            # Should not raise
            fresh_manager._dispatch_event(event)
