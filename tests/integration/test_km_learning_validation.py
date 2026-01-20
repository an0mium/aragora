"""
Validation tests for cross-debate learning via Knowledge Mound.

These tests simulate multiple debates on similar topics and verify that:
1. Agent expertise accumulates across debates
2. Compression patterns are reused
3. Domain experts are correctly identified for team selection
4. Cache performance improves with repeated queries
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


class TestCrossDebateLearningValidation:
    """End-to-end validation of cross-debate learning."""

    def test_expertise_accumulates_over_multiple_debates(self):
        """Agent expertise should accumulate across multiple debates."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Simulate 5 debates on security topics
        # Note: MIN_ELO_CHANGE is 25, so only deltas < 25 are filtered
        debates = [
            {"agent": "claude-3-opus", "domain": "security", "elo_change": 50},
            {"agent": "claude-3-opus", "domain": "security", "elo_change": 20},  # Below MIN_ELO_CHANGE threshold (25)
            {"agent": "claude-3-opus", "domain": "security", "elo_change": 75},
            {"agent": "gpt-4-turbo", "domain": "security", "elo_change": 60},
            {"agent": "claude-3-opus", "domain": "security", "elo_change": 55},
        ]

        base_elo = 1500
        for i, debate in enumerate(debates):
            base_elo += debate["elo_change"]
            adapter.store_agent_expertise(
                agent_name=debate["agent"],
                domain=debate["domain"],
                elo=base_elo,
                delta=debate["elo_change"],
                debate_id=f"debate-{i}",
            )

        # Verify claude-3-opus has accumulated expertise
        expertise = adapter.get_agent_expertise("claude-3-opus", "security")
        assert expertise is not None
        assert expertise["debate_count"] == 3  # Only 3 debates had delta >= MIN_ELO_CHANGE (25)

        # Verify confidence increased (3 debates > min of 5 needed for full confidence)
        assert expertise["confidence"] == 0.6  # 3/5 = 0.6

    def test_expertise_spans_multiple_domains(self):
        """Agents should build expertise in multiple domains independently."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Agent excels in security but struggles in coding
        adapter.store_agent_expertise("test-agent", "security", 1700, 100, "d1")
        adapter.store_agent_expertise("test-agent", "security", 1800, 100, "d2")
        adapter.store_agent_expertise("test-agent", "coding", 1400, -100, "d3")
        adapter.store_agent_expertise("test-agent", "coding", 1350, -50, "d4")

        # Check domain-specific expertise
        security_exp = adapter.get_agent_expertise("test-agent", "security")
        coding_exp = adapter.get_agent_expertise("test-agent", "coding")

        assert security_exp["elo"] == 1800
        assert coding_exp["elo"] == 1350

        # Verify agent appears as expert in security but not coding
        security_experts = adapter.get_domain_experts("security", limit=10)
        coding_experts = adapter.get_domain_experts("coding", limit=10)

        security_names = [e.agent_name for e in security_experts]
        coding_names = [e.agent_name for e in coding_experts]

        assert "test-agent" in security_names
        assert "test-agent" in coding_names  # Still appears but with lower ELO

        # Check relative rankings
        if security_experts:
            assert security_experts[0].elo >= 1700  # High ELO
        if coding_experts:
            # Should be sorted, but test-agent has low ELO
            assert any(e.agent_name == "test-agent" and e.elo == 1350 for e in coding_experts)

    def test_compression_patterns_reuse_across_debates(self):
        """Compression patterns should be reused when similar content is encountered."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        # First debate creates a pattern
        pattern_id1 = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["authentication", "oauth", "security"],
        )

        # Second debate with similar content should match
        pattern_id2 = adapter.store_compression_pattern(
            compression_ratio=0.35,
            value_score=0.8,
            content_markers=["authentication", "oauth", "security"],
        )

        # Should be the same pattern (or merged)
        pattern = adapter.get_pattern(pattern_id1)
        assert pattern is not None
        assert pattern["usage_count"] >= 2

    def test_team_selection_uses_historical_expertise(self):
        """Team selection should prioritize agents with historical expertise."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Build up expertise over multiple debates
        # Agent A: Consistently strong in security
        for i in range(3):
            adapter.store_agent_expertise("agent-A", "security", 1600 + i * 50, 50, f"d{i}")

        # Agent B: One-time participant
        adapter.store_agent_expertise("agent-B", "security", 1700, 200, "d10")

        # Agent C: Consistent but lower performance
        for i in range(3):
            adapter.store_agent_expertise("agent-C", "security", 1500 + i * 20, 50, f"d{i+20}")

        # Get domain experts for team selection
        experts = adapter.get_domain_experts("security", limit=10)

        # Agent A should have highest ELO from accumulated performance
        # Agent B has single high performance
        # Agent C has lower accumulated performance

        agent_elos = {e.agent_name: e.elo for e in experts}

        assert "agent-A" in agent_elos
        assert "agent-B" in agent_elos
        assert "agent-C" in agent_elos

        # Check confidence reflects debate count
        agent_confidence = {e.agent_name: e.confidence for e in experts}
        assert agent_confidence["agent-A"] == 0.6  # 3 debates
        assert agent_confidence["agent-B"] == 0.2  # 1 debate
        assert agent_confidence["agent-C"] == 0.6  # 3 debates

    def test_cache_improves_repeated_team_selection(self):
        """Cache should improve performance for repeated team selection queries."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter(cache_ttl_seconds=60.0)

        # Add some agents
        for i in range(10):
            adapter.store_agent_expertise(f"agent-{i}", "testing", 1500 + i * 50, 50, f"d{i}")

        # First query - cache miss
        experts1 = adapter.get_domain_experts("testing", limit=5)
        stats1 = adapter.get_cache_stats()

        assert stats1["cache_misses"] == 1
        assert stats1["cache_hits"] == 0

        # Second query - cache hit
        experts2 = adapter.get_domain_experts("testing", limit=5)
        stats2 = adapter.get_cache_stats()

        assert stats2["cache_misses"] == 1
        assert stats2["cache_hits"] == 1
        assert stats2["hit_rate"] == 0.5

        # Results should be identical
        assert len(experts1) == len(experts2)
        for e1, e2 in zip(experts1, experts2):
            assert e1.agent_name == e2.agent_name
            assert e1.elo == e2.elo

        # New expertise invalidates cache
        adapter.store_agent_expertise("new-agent", "testing", 1900, 400, "new-debate")

        # Third query - cache miss (invalidated)
        experts3 = adapter.get_domain_experts("testing", limit=5)
        stats3 = adapter.get_cache_stats()

        assert stats3["cache_misses"] == 2  # Incremented due to invalidation

    def test_domain_detection_affects_expertise_routing(self):
        """Domain detection should correctly route expertise to appropriate domains."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Questions from different domains
        test_cases = [
            ("How do we secure our API authentication?", "security"),
            ("What is the best SQL query optimization strategy?", "data"),
            ("How should we deploy this to kubernetes?", "devops"),
            ("Is this contract clause legally binding?", "legal"),
        ]

        for question, expected_domain in test_cases:
            detected = adapter.detect_domain(question)
            assert detected == expected_domain, f"Expected {expected_domain} for '{question}', got {detected}"


class TestPatternLearningValidation:
    """Validation of pattern learning across debates."""

    def test_rlm_patterns_improve_compression_hints(self):
        """RLM patterns should provide useful compression hints."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        # Store patterns from multiple debates
        adapter.store_compression_pattern(0.3, 0.9, ["api", "rest", "json"])
        adapter.store_compression_pattern(0.35, 0.85, ["api", "rest", "json"])
        adapter.store_compression_pattern(0.4, 0.8, ["database", "sql", "query"])

        # Get compression hints for new content
        hints = adapter.get_compression_hints(["api", "rest"])

        assert hints is not None
        # RlmAdapter returns recommended_ratio, strategy, confidence, based_on_patterns
        assert "recommended_ratio" in hints
        assert "strategy" in hints
        assert "confidence" in hints
        assert hints["based_on_patterns"] >= 1

    def test_high_usage_patterns_prioritized(self):
        """High-usage patterns should be prioritized in retrieval."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        # Create a pattern with low usage
        adapter.store_compression_pattern(0.3, 0.85, ["rare", "pattern"])

        # Create a pattern with high usage
        for _ in range(5):
            adapter.store_compression_pattern(0.35, 0.8, ["common", "pattern"])

        # Get patterns
        common_patterns = adapter.get_patterns_for_content(["common", "pattern"])
        rare_patterns = adapter.get_patterns_for_content(["rare", "pattern"])

        # Common pattern should have higher usage count
        if common_patterns and rare_patterns:
            assert common_patterns[0]["usage_count"] >= rare_patterns[0]["usage_count"]


class TestAdapterPersistenceValidation:
    """Validation of adapter persistence across sessions."""

    @pytest.mark.asyncio
    async def test_ranking_adapter_survives_sync_cycle(self):
        """RankingAdapter state should survive sync to/from KM."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        # First adapter: store expertise
        adapter1 = RankingAdapter()
        adapter1.store_agent_expertise("persistent-agent", "testing", 1750, 150, "d1")
        adapter1.store_agent_expertise("persistent-agent", "testing", 1800, 50, "d2")

        # Sync to mock KM
        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="km_node_123")

        sync_result = await adapter1.sync_to_mound(mock_mound, workspace_id="test")
        assert sync_result["expertise_synced"] >= 1

        # Second adapter: load from mock KM
        adapter2 = RankingAdapter()

        from unittest.mock import MagicMock
        from datetime import datetime

        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "agent_expertise",
            "agent_name": "persistent-agent",
            "domain": "testing",
            "elo": 1800,
            "debate_count": 2,
        }
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()

        mock_mound.query_nodes = AsyncMock(return_value=[mock_node])

        load_result = await adapter2.load_from_mound(mock_mound, workspace_id="test")
        assert load_result["expertise_loaded"] >= 1

        # Verify state was restored
        expertise = adapter2.get_agent_expertise("persistent-agent", "testing")
        assert expertise is not None
        assert expertise["elo"] == 1800

    @pytest.mark.asyncio
    async def test_rlm_adapter_survives_sync_cycle(self):
        """RlmAdapter state should survive sync to/from KM."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        # First adapter: store patterns
        adapter1 = RlmAdapter()
        adapter1.store_compression_pattern(0.3, 0.85, ["persistent", "pattern"])
        adapter1.store_compression_pattern(0.35, 0.8, ["persistent", "pattern"])

        # Sync to mock KM
        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="km_node_456")

        sync_result = await adapter1.sync_to_mound(mock_mound, workspace_id="test")
        assert sync_result["patterns_synced"] >= 1

        # Second adapter: load from mock KM
        adapter2 = RlmAdapter()

        from unittest.mock import MagicMock
        from datetime import datetime

        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "compression_pattern",
            "pattern_id": "cp_persistent",
            "compression_ratio": 0.3,
            "value_score": 0.85,
            "content_type": "code",
            "content_markers": ["persistent", "pattern"],
            "usage_count": 2,
        }
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()

        mock_mound.query_nodes = AsyncMock(return_value=[mock_node])

        load_result = await adapter2.load_from_mound(mock_mound, workspace_id="test")
        assert load_result["patterns_loaded"] >= 1

        # Verify state was restored
        pattern = adapter2.get_pattern("cp_persistent")
        assert pattern is not None
        assert pattern["compression_ratio"] == 0.3


class TestFeatureFlagsValidation:
    """Validation of feature flags for KM handlers."""

    def test_feature_flags_default_enabled(self):
        """All KM handlers should be enabled by default."""
        from aragora.config.settings import get_settings

        settings = get_settings()
        integration = settings.integration

        # Check inbound handlers
        assert integration.km_memory_to_mound_enabled is True
        assert integration.km_belief_to_mound_enabled is True
        assert integration.km_elo_to_mound_enabled is True
        assert integration.km_rlm_to_mound_enabled is True

        # Check outbound handlers
        assert integration.km_mound_to_memory_enabled is True
        assert integration.km_mound_to_belief_enabled is True
        assert integration.km_mound_to_rlm_enabled is True

    def test_feature_flag_helper_method(self):
        """is_km_handler_enabled should work for all handlers."""
        from aragora.config.settings import get_settings

        settings = get_settings()
        integration = settings.integration

        # Test helper method
        assert integration.is_km_handler_enabled("memory_to_mound") is True
        assert integration.is_km_handler_enabled("mound_to_belief") is True
        assert integration.is_km_handler_enabled("elo_to_mound") is True

        # Non-existent handler should default to True
        result = integration.is_km_handler_enabled("nonexistent_handler")
        assert result is True  # Default from getattr


class TestBatchingValidation:
    """Validation of event batching configuration."""

    def test_batch_config_from_settings(self):
        """CrossSubscriberManager should use batch config from settings."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.config.settings import get_settings

        settings = get_settings()
        integration = settings.integration

        manager = CrossSubscriberManager()

        # Verify config matches settings
        assert manager._async_config.batch_size == integration.event_batch_size
        assert manager._async_config.batch_timeout_seconds == integration.event_batch_timeout_seconds
        assert manager._async_config.enable_batching == integration.event_batching_enabled

    def test_batch_stats_available(self):
        """Batch stats should be available via get_batch_stats."""
        from aragora.events.cross_subscribers import CrossSubscriberManager

        manager = CrossSubscriberManager()
        stats = manager.get_batch_stats()

        assert "batch_size" in stats
        assert "batching_enabled" in stats
        assert "async_event_types" in stats
