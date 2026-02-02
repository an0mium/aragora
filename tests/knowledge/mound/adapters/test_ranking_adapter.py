"""
Tests for RankingAdapter - Bridges ELO/Ranking system to Knowledge Mound.

Tests cover:
- Agent expertise storage and retrieval
- Domain expert queries with caching
- Domain detection from questions
- Agent history tracking
- Cache invalidation
- Statistics
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestRankingAdapterInit:
    """Tests for RankingAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        assert adapter.elo_system is None
        assert adapter._enable_dual_write is False
        assert adapter._cache_ttl_seconds == 60.0
        assert len(adapter._expertise) == 0

    def test_init_with_elo_system(self):
        """Should accept EloSystem."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        mock_elo = MagicMock()
        adapter = RankingAdapter(elo_system=mock_elo)

        assert adapter.elo_system is mock_elo

    def test_set_elo_system(self):
        """Should allow setting elo_system after init."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        mock_elo = MagicMock()
        adapter.set_elo_system(mock_elo)

        assert adapter.elo_system is mock_elo

    def test_custom_cache_ttl(self):
        """Should accept custom cache TTL."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter(cache_ttl_seconds=120.0)

        assert adapter._cache_ttl_seconds == 120.0


class TestAgentExpertise:
    """Tests for AgentExpertise dataclass."""

    def test_create_agent_expertise(self):
        """Should create AgentExpertise."""
        from aragora.knowledge.mound.adapters import AgentExpertise

        expertise = AgentExpertise(
            agent_name="claude-3",
            domain="security",
            elo=1650,
            confidence=0.8,
            last_updated="2024-01-15T10:00:00",
            debate_count=5,
        )

        assert expertise.agent_name == "claude-3"
        assert expertise.domain == "security"
        assert expertise.elo == 1650
        assert expertise.confidence == 0.8
        assert expertise.debate_count == 5


class TestExpertiseSearchResult:
    """Tests for ExpertiseSearchResult dataclass."""

    def test_create_search_result(self):
        """Should create ExpertiseSearchResult."""
        from aragora.knowledge.mound.adapters import ExpertiseSearchResult

        result = ExpertiseSearchResult(
            expertise={"agent": "test", "domain": "coding"},
            relevance_score=0.9,
            matched_domains=["coding", "architecture"],
        )

        assert result.relevance_score == 0.9
        assert len(result.matched_domains) == 2

    def test_default_matched_domains(self):
        """Should default matched_domains to empty list."""
        from aragora.knowledge.mound.adapters import ExpertiseSearchResult

        result = ExpertiseSearchResult(
            expertise={"agent": "test"},
        )

        assert result.matched_domains == []


class TestStoreAgentExpertise:
    """Tests for storing agent expertise."""

    def test_store_expertise(self):
        """Should store expertise with significant ELO change."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        result = adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1650,
            delta=50,
            debate_id="debate-123",
        )

        assert result is not None
        assert result.startswith("ex_")

    def test_store_expertise_below_threshold(self):
        """Should skip storing when delta below threshold."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        result = adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1510,
            delta=10,  # Below MIN_ELO_CHANGE=25
        )

        assert result is None

    def test_store_updates_indices(self):
        """Should update domain and agent indices."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1650,
            delta=50,
        )

        assert "security" in adapter._domain_agents
        assert "claude-3" in adapter._domain_agents["security"]
        assert "claude-3" in adapter._agent_domains
        assert "security" in adapter._agent_domains["claude-3"]

    def test_store_increments_debate_count(self):
        """Should increment debate count on subsequent stores."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1550,
            delta=50,
        )

        adapter.store_agent_expertise(
            agent_name="claude-3",
            domain="security",
            elo=1600,
            delta=50,
        )

        expertise = adapter.get_agent_expertise("claude-3", "security")
        assert expertise["debate_count"] == 2


class TestGetAgentExpertise:
    """Tests for retrieving agent expertise."""

    def test_get_expertise_for_domain(self):
        """Should get expertise for specific domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)

        expertise = adapter.get_agent_expertise("claude-3", "security")

        assert expertise is not None
        assert expertise["elo"] == 1650
        assert expertise["domain"] == "security"

    def test_get_all_expertise_for_agent(self):
        """Should get all domains when no domain specified."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)
        adapter.store_agent_expertise("claude-3", "coding", 1700, 50)

        expertise = adapter.get_agent_expertise("claude-3")

        assert expertise is not None
        assert "security" in expertise
        assert "coding" in expertise

    def test_get_nonexistent_expertise(self):
        """Should return None for nonexistent expertise."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        expertise = adapter.get_agent_expertise("nonexistent")

        assert expertise is None


class TestGetDomainExperts:
    """Tests for domain expert queries."""

    def test_get_domain_experts(self):
        """Should return experts sorted by ELO."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)
        adapter.store_agent_expertise("gpt-4", "security", 1750, 50)
        adapter.store_agent_expertise("gemini", "security", 1600, 50)

        experts = adapter.get_domain_experts("security", limit=10)

        assert len(experts) == 3
        assert experts[0].agent_name == "gpt-4"  # Highest ELO first
        assert experts[0].elo == 1750

    def test_get_domain_experts_with_limit(self):
        """Should respect limit parameter."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        for i in range(10):
            adapter.store_agent_expertise(f"agent-{i}", "security", 1500 + i * 10, 50)

        experts = adapter.get_domain_experts("security", limit=3)

        assert len(experts) == 3

    def test_get_domain_experts_with_min_confidence(self):
        """Should filter by minimum confidence."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        # Add single debate (confidence = 0.2)
        adapter.store_agent_expertise("low-conf", "security", 1700, 50)
        # Add 5 debates (confidence = 1.0)
        for _ in range(5):
            adapter.store_agent_expertise("high-conf", "security", 1600, 50)

        experts = adapter.get_domain_experts("security", min_confidence=0.9)

        assert len(experts) == 1
        assert experts[0].agent_name == "high-conf"

    def test_domain_experts_cache(self):
        """Should cache domain expert queries."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter(cache_ttl_seconds=60.0)
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)

        # First call - cache miss
        adapter.get_domain_experts("security")
        stats1 = adapter.get_cache_stats()
        assert stats1["cache_misses"] == 1

        # Second call - cache hit
        adapter.get_domain_experts("security")
        stats2 = adapter.get_cache_stats()
        assert stats2["cache_hits"] == 1

    def test_skip_cache(self):
        """Should skip cache when use_cache=False."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)

        # First call without cache
        adapter.get_domain_experts("security", use_cache=False)

        # Second call without cache
        adapter.get_domain_experts("security", use_cache=False)

        stats = adapter.get_cache_stats()
        assert stats["cache_misses"] == 2
        assert stats["cache_hits"] == 0


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_all_cache(self):
        """Should invalidate all cache entries."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)
        adapter.store_agent_expertise("claude-3", "coding", 1700, 50)

        adapter.get_domain_experts("security")
        adapter.get_domain_experts("coding")

        count = adapter.invalidate_domain_cache()

        assert count == 2
        assert len(adapter._domain_experts_cache) == 0

    def test_invalidate_specific_domain(self):
        """Should invalidate only specified domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)
        adapter.store_agent_expertise("claude-3", "coding", 1700, 50)

        adapter.get_domain_experts("security")
        adapter.get_domain_experts("coding")

        count = adapter.invalidate_domain_cache("security")

        assert count == 1
        assert len(adapter._domain_experts_cache) == 1


class TestDomainDetection:
    """Tests for domain detection from questions."""

    def test_detect_security_domain(self):
        """Should detect security domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        domain = adapter.detect_domain("How should we handle SQL injection attacks?")

        assert domain == "security"

    def test_detect_coding_domain(self):
        """Should detect coding domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        domain = adapter.detect_domain("What is the best algorithm for sorting?")

        assert domain == "coding"

    def test_detect_data_domain(self):
        """Should detect data domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        domain = adapter.detect_domain("How to optimize database queries?")

        assert domain == "data"

    def test_detect_general_domain(self):
        """Should default to general when no keywords match."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        domain = adapter.detect_domain("What is the best restaurant nearby?")

        assert domain == "general"


class TestAgentHistory:
    """Tests for agent history tracking."""

    def test_get_agent_history(self):
        """Should retrieve agent history."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1550, 50, "debate-1")
        adapter.store_agent_expertise("claude-3", "security", 1600, 50, "debate-2")

        history = adapter.get_agent_history("claude-3")

        assert len(history) == 2
        assert history[0]["debate_id"] == "debate-2"  # Newest first

    def test_get_history_with_domain_filter(self):
        """Should filter history by domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1550, 50)
        adapter.store_agent_expertise("claude-3", "coding", 1600, 50)

        history = adapter.get_agent_history("claude-3", domain="security")

        assert len(history) == 1
        assert history[0]["domain"] == "security"

    def test_get_history_with_limit(self):
        """Should respect history limit."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        for i in range(10):
            adapter.store_agent_expertise("claude-3", "security", 1500 + i * 10, 50)

        history = adapter.get_agent_history("claude-3", limit=3)

        assert len(history) == 3


class TestStatistics:
    """Tests for adapter statistics."""

    def test_get_stats(self):
        """Should return accurate statistics."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)
        adapter.store_agent_expertise("gpt-4", "security", 1700, 50)
        adapter.store_agent_expertise("claude-3", "coding", 1600, 50)

        stats = adapter.get_stats()

        assert stats["total_expertise_records"] == 3
        assert stats["total_agents"] == 2
        assert stats["total_domains"] == 2
        assert stats["agents_per_domain"]["security"] == 2

    def test_get_all_domains(self):
        """Should return all domains with expertise."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)
        adapter.store_agent_expertise("claude-3", "coding", 1600, 50)
        adapter.store_agent_expertise("claude-3", "data", 1700, 50)

        domains = adapter.get_all_domains()

        assert len(domains) == 3
        assert "security" in domains
        assert "coding" in domains

    def test_get_cache_stats(self):
        """Should return cache statistics."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        stats = adapter.get_cache_stats()

        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_size" in stats
        assert "hit_rate" in stats
        assert "ttl_seconds" in stats


class TestKnowledgeMoundSync:
    """Tests for Knowledge Mound persistence."""

    @pytest.mark.asyncio
    async def test_sync_to_mound(self):
        """Should sync expertise to Knowledge Mound."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)

        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock()

        result = await adapter.sync_to_mound(mock_mound, "workspace-1")

        assert result["expertise_synced"] == 1
        mock_mound.ingest.assert_called()

    @pytest.mark.asyncio
    async def test_sync_to_mound_handles_errors(self):
        """Should handle sync errors gracefully."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()
        adapter.store_agent_expertise("claude-3", "security", 1650, 50)

        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(side_effect=Exception("Sync failed"))

        result = await adapter.sync_to_mound(mock_mound, "workspace-1")

        assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_load_from_mound(self):
        """Should load expertise from Knowledge Mound."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Create mock nodes
        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "agent_expertise",
            "agent_name": "claude-3",
            "domain": "security",
            "elo": 1650,
            "confidence": 0.8,
            "debate_count": 5,
        }
        mock_node.created_at = datetime.now(timezone.utc)
        mock_node.updated_at = datetime.now(timezone.utc)
        mock_node.confidence = 0.8

        mock_mound = AsyncMock()
        mock_mound.query_nodes = AsyncMock(return_value=[mock_node])

        result = await adapter.load_from_mound(mock_mound, "workspace-1")

        assert result["expertise_loaded"] == 1
        assert adapter.get_agent_expertise("claude-3", "security") is not None

    @pytest.mark.asyncio
    async def test_load_from_mound_handles_errors(self):
        """Should handle load errors gracefully."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        mock_mound = AsyncMock()
        mock_mound.query_nodes = AsyncMock(side_effect=Exception("Query failed"))

        result = await adapter.load_from_mound(mock_mound, "workspace-1")

        assert len(result["errors"]) == 1


# =============================================================================
# Store Ranking Updates Tests
# =============================================================================


class TestStoreRankingUpdates:
    """Tests for storing ranking updates."""

    def test_store_ranking_update_with_positive_delta(self):
        """Should store ranking update with positive ELO change."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        result = adapter.store_agent_expertise(
            agent_name="rising-star",
            domain="security",
            elo=1600,
            delta=100,  # Big win
            debate_id="d-upset",
        )

        assert result is not None
        expertise = adapter.get_agent_expertise("rising-star", "security")
        assert expertise["elo"] == 1600
        assert expertise["delta"] == 100

    def test_store_ranking_update_with_negative_delta(self):
        """Should store ranking update with negative ELO change."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        result = adapter.store_agent_expertise(
            agent_name="falling",
            domain="coding",
            elo=1400,
            delta=-100,  # Big loss
            debate_id="d-loss",
        )

        assert result is not None
        expertise = adapter.get_agent_expertise("falling", "coding")
        assert expertise["elo"] == 1400
        assert expertise["delta"] == -100

    def test_ranking_update_tracks_previous_elo(self):
        """Should track previous ELO value."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("tracker", "security", 1550, delta=50)
        adapter.store_agent_expertise("tracker", "security", 1600, delta=50)

        expertise = adapter.get_agent_expertise("tracker", "security")

        assert expertise["elo"] == 1600
        assert expertise["previous_elo"] == 1550

    def test_multiple_ranking_updates_same_agent_domain(self):
        """Should correctly track multiple updates for same agent/domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Simulate progression over multiple debates
        for i in range(5):
            adapter.store_agent_expertise(
                agent_name="progressing",
                domain="security",
                elo=1500 + i * 25,
                delta=25,
                debate_id=f"d-{i}",
            )

        expertise = adapter.get_agent_expertise("progressing", "security")

        assert expertise["elo"] == 1600  # Final ELO
        assert expertise["debate_count"] == 5

    def test_ranking_update_across_multiple_domains(self):
        """Should track separate rankings for different domains."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("versatile", "security", 1700, delta=50)
        adapter.store_agent_expertise("versatile", "coding", 1600, delta=50)
        adapter.store_agent_expertise("versatile", "data", 1550, delta=50)

        all_expertise = adapter.get_agent_expertise("versatile")

        assert len(all_expertise) == 3
        assert all_expertise["security"]["elo"] == 1700
        assert all_expertise["coding"]["elo"] == 1600
        assert all_expertise["data"]["elo"] == 1550


# =============================================================================
# Get Top-Ranked Agents Tests
# =============================================================================


class TestGetTopRankedAgents:
    """Tests for retrieving top-ranked agents."""

    def test_get_top_ranked_returns_correct_order(self):
        """Should return agents sorted by ELO descending."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Add agents with different ELOs
        adapter.store_agent_expertise("low", "security", 1400, delta=50)
        adapter.store_agent_expertise("medium", "security", 1550, delta=50)
        adapter.store_agent_expertise("high", "security", 1700, delta=50)

        experts = adapter.get_domain_experts("security", limit=10)

        assert len(experts) == 3
        assert experts[0].agent_name == "high"
        assert experts[1].agent_name == "medium"
        assert experts[2].agent_name == "low"

    def test_get_top_ranked_respects_limit(self):
        """Should only return requested number of top agents."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Add many agents
        for i in range(20):
            adapter.store_agent_expertise(f"agent-{i}", "security", 1500 + i * 10, delta=30)

        experts = adapter.get_domain_experts("security", limit=5)

        assert len(experts) == 5
        # Should be the top 5 by ELO
        assert experts[0].elo == 1690
        assert experts[4].elo == 1650

    def test_get_top_ranked_empty_domain(self):
        """Should return empty list for domain with no agents."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        experts = adapter.get_domain_experts("nonexistent", limit=10)

        assert len(experts) == 0

    def test_get_top_ranked_single_agent(self):
        """Should correctly handle single agent in domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("solo", "niche", 1600, delta=50)

        experts = adapter.get_domain_experts("niche")

        assert len(experts) == 1
        assert experts[0].agent_name == "solo"

    def test_get_top_ranked_with_all_domains(self):
        """Should correctly get top agents across all domains."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("alice", "security", 1800, delta=50)
        adapter.store_agent_expertise("bob", "coding", 1750, delta=50)
        adapter.store_agent_expertise("carol", "data", 1700, delta=50)

        # Get all domains
        domains = adapter.get_all_domains()

        assert len(domains) == 3

        # Each domain should have one expert
        for domain in domains:
            experts = adapter.get_domain_experts(domain)
            assert len(experts) == 1


# =============================================================================
# Ranking Order Preservation Tests
# =============================================================================


class TestRankingOrderPreservation:
    """Tests for ranking order preservation."""

    def test_order_preserved_after_updates(self):
        """Should maintain correct order after multiple updates."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Initial rankings
        adapter.store_agent_expertise("a", "security", 1500, delta=50)
        adapter.store_agent_expertise("b", "security", 1600, delta=50)
        adapter.store_agent_expertise("c", "security", 1700, delta=50)

        # Update - 'a' surpasses 'c'
        adapter.store_agent_expertise("a", "security", 1750, delta=250)

        experts = adapter.get_domain_experts("security")

        # New order should be: a, c, b
        assert experts[0].agent_name == "a"
        assert experts[1].agent_name == "c"
        assert experts[2].agent_name == "b"

    def test_order_stable_for_equal_elo(self):
        """Should maintain stable order for agents with equal ELO."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # All agents with same ELO
        adapter.store_agent_expertise("alpha", "security", 1500, delta=50)
        adapter.store_agent_expertise("beta", "security", 1500, delta=50)
        adapter.store_agent_expertise("gamma", "security", 1500, delta=50)

        experts1 = adapter.get_domain_experts("security")
        experts2 = adapter.get_domain_experts("security")

        # Order should be consistent between calls
        assert [e.agent_name for e in experts1] == [e.agent_name for e in experts2]

    def test_history_preserves_chronological_order(self):
        """Should preserve chronological order in history."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Simulate agent history
        for i in range(10):
            adapter.store_agent_expertise("historian", "security", 1500 + i * 10, delta=30)

        history = adapter.get_agent_history("historian")

        # History should be newest first
        for i in range(len(history) - 1):
            assert history[i]["timestamp"] >= history[i + 1]["timestamp"]

    def test_domain_order_independent(self):
        """Rankings in one domain should not affect another."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Set up different rankings per domain
        adapter.store_agent_expertise("alice", "security", 1800, delta=50)
        adapter.store_agent_expertise("bob", "security", 1600, delta=50)

        adapter.store_agent_expertise("alice", "coding", 1400, delta=50)
        adapter.store_agent_expertise("bob", "coding", 1700, delta=50)

        security_experts = adapter.get_domain_experts("security")
        coding_experts = adapter.get_domain_experts("coding")

        # Alice leads in security, Bob leads in coding
        assert security_experts[0].agent_name == "alice"
        assert coding_experts[0].agent_name == "bob"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestRankingEdgeCases:
    """Tests for edge cases in ranking operations."""

    def test_empty_rankings_domain_query(self):
        """Should handle query on empty domain gracefully."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        experts = adapter.get_domain_experts("empty-domain")

        assert experts == []

    def test_empty_rankings_agent_query(self):
        """Should handle query for unknown agent."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        expertise = adapter.get_agent_expertise("unknown-agent")

        assert expertise is None

    def test_ties_in_ranking(self):
        """Should handle multiple agents with exact same ELO."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Three agents with exact same ELO
        adapter.store_agent_expertise("tied-1", "security", 1550, delta=50)
        adapter.store_agent_expertise("tied-2", "security", 1550, delta=50)
        adapter.store_agent_expertise("tied-3", "security", 1550, delta=50)

        experts = adapter.get_domain_experts("security")

        # All three should be returned
        assert len(experts) == 3
        # All should have same ELO
        assert all(e.elo == 1550 for e in experts)

    def test_very_large_number_of_agents(self):
        """Should handle large number of agents efficiently."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Add 100 agents
        for i in range(100):
            adapter.store_agent_expertise(f"agent-{i:03d}", "security", 1400 + i, delta=30)

        experts = adapter.get_domain_experts("security", limit=10)

        assert len(experts) == 10
        assert experts[0].elo == 1499  # Highest

    def test_special_characters_in_agent_name(self):
        """Should handle special characters in agent names."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("agent-with-dash", "security", 1600, delta=50)
        adapter.store_agent_expertise("agent_with_underscore", "security", 1550, delta=50)
        adapter.store_agent_expertise("agent.with.dots", "security", 1500, delta=50)

        experts = adapter.get_domain_experts("security")

        assert len(experts) == 3

    def test_very_long_domain_name(self):
        """Should handle very long domain names."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        long_domain = "a" * 200

        adapter.store_agent_expertise("agent", long_domain, 1600, delta=50)

        experts = adapter.get_domain_experts(long_domain)

        assert len(experts) == 1

    def test_unicode_in_names(self):
        """Should handle unicode characters in agent and domain names."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("agent-français", "sécurité", 1600, delta=50)

        experts = adapter.get_domain_experts("sécurité")

        assert len(experts) == 1
        assert experts[0].agent_name == "agent-français"

    def test_zero_confidence_agents_included(self):
        """Should include agents with zero confidence when threshold is 0."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Single debate gives confidence of 0.2 (1/5)
        adapter.store_agent_expertise("newbie", "security", 1600, delta=50)

        # With min_confidence=0, should be included
        experts = adapter.get_domain_experts("security", min_confidence=0.0)

        assert len(experts) == 1

    def test_max_confidence_agents_only(self):
        """Should filter to only max confidence agents."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Agent with low confidence (1 debate)
        adapter.store_agent_expertise("low-conf", "security", 1800, delta=50)

        # Agent with max confidence (5 debates)
        for _ in range(5):
            adapter.store_agent_expertise("high-conf", "security", 1600, delta=30)

        experts = adapter.get_domain_experts("security", min_confidence=1.0)

        assert len(experts) == 1
        assert experts[0].agent_name == "high-conf"

    def test_history_limit_zero(self):
        """Should return empty list when history limit is 0."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("agent", "security", 1600, delta=50)

        history = adapter.get_agent_history("agent", limit=0)

        assert len(history) == 0

    def test_domain_experts_limit_zero(self):
        """Should return empty list when limit is 0."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        adapter.store_agent_expertise("agent", "security", 1600, delta=50)

        experts = adapter.get_domain_experts("security", limit=0)

        assert len(experts) == 0

    def test_negative_elo_delta_below_threshold(self):
        """Should not store when absolute delta is below threshold."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Delta of -10 is below MIN_ELO_CHANGE (25)
        result = adapter.store_agent_expertise("agent", "security", 1490, delta=-10)

        assert result is None

    def test_cache_invalidation_on_new_agent(self):
        """Should invalidate cache when new agent added to domain."""
        from aragora.knowledge.mound.adapters import RankingAdapter

        adapter = RankingAdapter()

        # Populate and cache
        adapter.store_agent_expertise("original", "security", 1600, delta=50)
        adapter.get_domain_experts("security")  # Cache it

        # Add new agent - should invalidate cache
        adapter.store_agent_expertise("newcomer", "security", 1700, delta=50)

        # Next query should include newcomer
        experts = adapter.get_domain_experts("security")

        assert len(experts) == 2
        assert experts[0].agent_name == "newcomer"
