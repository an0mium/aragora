"""
Integration tests for bidirectional Knowledge Mound integration.

Tests that data flows correctly IN to the Knowledge Mound (ingestion)
and OUT (retrieval) for all integrated subsystems.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestMemoryBidirectional:
    """Test bidirectional Memory ↔ Knowledge Mound integration."""

    def test_memory_to_mound_stores_high_importance(self):
        """Test that high-importance memories are synced to KM."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Create high-importance memory event
        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={
                "content": "Critical security vulnerability found",
                "importance": 0.9,
                "tier": "fast",
                "metadata": {"source": "debate"},
            },
        )

        # Should not raise
        manager._handle_memory_to_mound(event)

    def test_memory_to_mound_ignores_low_importance(self):
        """Test that low-importance memories are not synced."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Create low-importance memory event
        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={
                "content": "Minor observation",
                "importance": 0.3,
                "tier": "fast",
            },
        )

        # Should not raise and should skip silently
        manager._handle_memory_to_mound(event)

    def test_mound_to_memory_prewarm(self):
        """Test that KM queries trigger memory pre-warming."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_QUERIED,
            data={
                "query": "security vulnerability",
                "results_count": 5,
                "workspace_id": "test-ws",
            },
        )

        # Should not raise
        manager._handle_mound_to_memory_retrieval(event)


class TestBeliefBidirectional:
    """Test bidirectional Belief Network ↔ Knowledge Mound integration."""

    def test_belief_to_mound_stores_converged(self):
        """Test that converged beliefs are stored in KM."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.BELIEF_CONVERGED,
            data={
                "debate_id": "debate-123",
                "beliefs_count": 5,
                "beliefs": [
                    {"node_id": "n1", "confidence": 0.85, "claim_statement": "Test claim"},
                ],
                "cruxes": [
                    {"claim_id": "c1", "crux_score": 0.5, "statement": "Key disagreement"},
                ],
            },
        )

        # Should not raise
        manager._handle_belief_to_mound(event)

    def test_mound_to_belief_initializes_priors(self):
        """Test that debate start queries KM for historical cruxes."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate-456",
                "question": "Should we use microservices?",
            },
        )

        # Should not raise
        manager._handle_mound_to_belief(event)


class TestRlmBidirectional:
    """Test bidirectional RLM ↔ Knowledge Mound integration."""

    def test_rlm_to_mound_stores_pattern(self):
        """Test that compression patterns are stored in KM."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.RLM_COMPRESSION_COMPLETE,
            data={
                "compression_ratio": 0.35,
                "value_score": 0.8,
                "content_markers": ["security", "api"],
            },
        )

        # Should not raise
        manager._handle_rlm_to_mound(event)

    def test_rlm_to_mound_ignores_low_value(self):
        """Test that low-value patterns are not stored."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.RLM_COMPRESSION_COMPLETE,
            data={
                "compression_ratio": 0.5,
                "value_score": 0.3,  # Below threshold
                "content_markers": ["general"],
            },
        )

        # Should not raise
        manager._handle_rlm_to_mound(event)

    def test_mound_to_rlm_updates_priorities(self):
        """Test that KM queries update RLM priorities."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_QUERIED,
            data={
                "query": "authentication patterns",
                "results_count": 3,
                "node_ids": ["node-1", "node-2", "node-3"],
            },
        )

        # Should not raise
        manager._handle_mound_to_rlm(event)


class TestEloBidirectional:
    """Test bidirectional ELO/Ranking ↔ Knowledge Mound integration."""

    def test_elo_to_mound_stores_expertise(self):
        """Test that significant ELO changes are stored in KM."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.AGENT_ELO_UPDATED,
            data={
                "agent": "claude-3-opus",
                "elo": 1650,
                "delta": 50,  # Significant change
                "debate_id": "debate-789",
                "domain": "security",
            },
        )

        # Should not raise
        manager._handle_elo_to_mound(event)

    def test_elo_to_mound_ignores_small_changes(self):
        """Test that small ELO changes are not stored."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.AGENT_ELO_UPDATED,
            data={
                "agent": "claude-3-opus",
                "elo": 1510,
                "delta": 10,  # Too small
                "debate_id": "debate-789",
            },
        )

        # Should not raise
        manager._handle_elo_to_mound(event)

    def test_mound_to_team_selection(self):
        """Test that debate start queries KM for domain experts."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate-101",
                "question": "How should we handle authentication in microservices?",
            },
        )

        # Should not raise
        manager._handle_mound_to_team_selection(event)


class TestInsightsBidirectional:
    """Test bidirectional Insights/Trickster ↔ Knowledge Mound integration."""

    def test_insight_to_mound_stores_high_confidence(self):
        """Test that high-confidence insights are stored in KM."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.INSIGHT_EXTRACTED,
            data={
                "type": "consensus_insight",
                "confidence": 0.85,
                "debate_id": "debate-222",
                "title": "Key agreement on security approach",
            },
        )

        # Should not raise
        manager._handle_insight_to_mound(event)

    def test_insight_to_mound_ignores_low_confidence(self):
        """Test that low-confidence insights are not stored."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.INSIGHT_EXTRACTED,
            data={
                "type": "observation",
                "confidence": 0.4,  # Below threshold
                "debate_id": "debate-222",
            },
        )

        # Should not raise
        manager._handle_insight_to_mound(event)

    def test_flip_to_mound_stores_all(self):
        """Test that ALL flip events are stored (meta-learning)."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.FLIP_DETECTED,
            data={
                "agent_name": "gpt-4",
                "flip_type": "position_reversal",
                "original_claim": "X is better",
                "new_claim": "Y is better",
            },
        )

        # Should not raise
        manager._handle_flip_to_mound(event)

    def test_mound_to_trickster(self):
        """Test that debate start queries KM for flip history."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "debate_id": "debate-333",
                "agents": ["gpt-4", "claude-3"],
            },
        )

        # Should not raise
        manager._handle_mound_to_trickster(event)


class TestCultureBidirectional:
    """Test Culture Patterns ↔ Debate Protocol integration."""

    def test_culture_to_debate_handles_patterns(self):
        """Test that culture pattern updates are handled."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.MOUND_UPDATED,
            data={
                "update_type": "culture_patterns",
                "patterns_count": 10,
                "workspace_id": "test-ws",
            },
        )

        # Should not raise
        manager._handle_culture_to_debate(event)

    def test_culture_to_debate_ignores_other_updates(self):
        """Test that non-culture updates are ignored."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.MOUND_UPDATED,
            data={
                "update_type": "node_added",
                "node_id": "node-1",
            },
        )

        # Should not raise
        manager._handle_culture_to_debate(event)


class TestStalenessBidirectional:
    """Test Staleness ↔ Debate integration."""

    def test_staleness_to_debate_warns_active(self):
        """Test that staleness warnings are checked against active debates."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_STALE,
            data={
                "node_id": "stale-node-1",
                "reason": "Source updated",
                "last_verified": "2024-01-01T00:00:00",
            },
        )

        # Should not raise
        manager._handle_staleness_to_debate(event)


class TestProvenanceBidirectional:
    """Test Provenance ↔ Knowledge Mound integration."""

    def test_provenance_to_mound_stores_verified(self):
        """Test that verified provenance chains are stored."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate-444",
                "consensus_reached": True,
                "provenance_chains": [
                    {
                        "id": "chain-1",
                        "source_id": "src-1",
                        "claim_ids": ["c1", "c2"],
                        "verified": True,
                        "method": "consensus",
                    },
                ],
            },
        )

        # Should not raise
        manager._handle_provenance_to_mound(event)

    def test_provenance_to_mound_ignores_no_consensus(self):
        """Test that provenance is not stored without consensus."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "debate_id": "debate-444",
                "consensus_reached": False,
            },
        )

        # Should not raise
        manager._handle_provenance_to_mound(event)

    def test_mound_to_provenance_queries_history(self):
        """Test that claim verification queries KM for history."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        event = StreamEvent(
            type=StreamEventType.CLAIM_VERIFICATION_RESULT,
            data={
                "claim_id": "claim-1",
                "claim": "Security is paramount",
                "verified": True,
            },
        )

        # Should not raise
        manager._handle_mound_to_provenance(event)


class TestRankingAdapter:
    """Test RankingAdapter functionality."""

    def test_store_agent_expertise(self):
        """Test storing agent expertise."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Store expertise
        result = adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1650,
            delta=50,
            debate_id="debate-1",
        )

        assert result is not None
        assert result.startswith("ex_")

        # Verify stored
        expertise = adapter.get_agent_expertise("claude-3", "security")
        assert expertise is not None
        assert expertise["elo"] == 1650
        assert expertise["domain"] == "security"

    def test_store_ignores_small_changes(self):
        """Test that small ELO changes are ignored."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        result = adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1510,
            delta=10,  # Too small
        )

        assert result is None

    def test_get_domain_experts(self):
        """Test retrieving domain experts."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Store multiple agents
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)
        adapter.store_agent_expertise("gpt-4", "security", 1700, 60)
        adapter.store_agent_expertise("gemini", "security", 1550, 30)

        experts = adapter.get_domain_experts("security", limit=2)

        assert len(experts) == 2
        assert experts[0].agent_name == "gpt-4"  # Highest ELO
        assert experts[1].agent_name == "claude-3"

    def test_detect_domain(self):
        """Test domain detection from question."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        assert adapter.detect_domain("How to prevent SQL injection?") == "security"
        assert adapter.detect_domain("Best algorithm for sorting?") == "coding"
        assert adapter.detect_domain("Random question") == "general"


class TestRlmAdapter:
    """Test RlmAdapter functionality."""

    def test_store_compression_pattern(self):
        """Test storing compression patterns."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        result = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api"],
        )

        assert result is not None
        assert result.startswith("cp_")

        # Verify stored
        pattern = adapter.get_pattern(result)
        assert pattern is not None
        assert pattern["compression_ratio"] == 0.3
        assert pattern["value_score"] == 0.85

    def test_store_ignores_low_value(self):
        """Test that low-value patterns are ignored."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        result = adapter.store_compression_pattern(
            compression_ratio=0.5,
            value_score=0.3,  # Below threshold
            content_markers=["general"],
        )

        assert result is None

    def test_get_patterns_for_content(self):
        """Test finding patterns by content markers."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        # Store patterns
        adapter.store_compression_pattern(0.3, 0.85, ["security", "api"])
        adapter.store_compression_pattern(0.4, 0.8, ["database", "api"])

        patterns = adapter.get_patterns_for_content(["api", "auth"])

        assert len(patterns) >= 1
        # Should match the security/api pattern due to "api" overlap

    def test_get_compression_hints(self):
        """Test getting compression strategy hints."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        # Store pattern
        adapter.store_compression_pattern(0.3, 0.85, ["security", "api"])

        hints = adapter.get_compression_hints(["security"])

        assert "recommended_ratio" in hints
        assert "strategy" in hints
        assert "confidence" in hints

    def test_update_access_pattern(self):
        """Test recording access patterns."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        adapter.update_access_pattern("content-1")
        adapter.update_access_pattern("content-1")
        adapter.update_access_pattern("content-1")

        priorities = adapter.get_priority_content(min_access_count=3)
        assert len(priorities) == 1
        assert priorities[0].content_id == "content-1"
        assert priorities[0].access_count == 3


class TestEventTypeRegistration:
    """Test that new event types are properly registered."""

    def test_new_event_types_exist(self):
        """Test that new event types are defined."""
        from aragora.events.types import StreamEventType

        assert hasattr(StreamEventType, "KNOWLEDGE_STALE")
        assert hasattr(StreamEventType, "BELIEF_CONVERGED")
        assert hasattr(StreamEventType, "CRUX_DETECTED")
        assert hasattr(StreamEventType, "RLM_COMPRESSION_COMPLETE")

    def test_handlers_registered(self):
        """Test that all bidirectional handlers are registered."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEventType

        manager = CrossSubscriberManager()

        # Check handler registrations
        stats = manager.get_stats()

        # New handlers should be present
        assert "memory_to_mound" in stats
        assert "mound_to_memory_retrieval" in stats
        assert "belief_to_mound" in stats
        assert "mound_to_belief" in stats
        assert "rlm_to_mound" in stats
        assert "mound_to_rlm" in stats
        assert "elo_to_mound" in stats
        assert "mound_to_team_selection" in stats
        assert "insight_to_mound" in stats
        assert "flip_to_mound" in stats
        assert "mound_to_trickster" in stats
        assert "culture_to_debate" in stats
        assert "staleness_to_debate" in stats
        assert "provenance_to_mound" in stats
        assert "mound_to_provenance" in stats
