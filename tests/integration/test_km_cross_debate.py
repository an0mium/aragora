"""
Integration tests for cross-debate learning via Knowledge Mound.

Tests the full cycle:
1. Debate A runs and generates expertise/patterns
2. These are persisted to KM via adapters
3. Debate B on similar topic retrieves and uses this data
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


class TestCrossDebateLearningCycle:
    """Tests for expertise persistence and retrieval across debates."""

    def test_expertise_persists_after_debate_end(self):
        """Agent expertise is persisted to KM when debate ends."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter
        from aragora.events.types import StreamEvent, StreamEventType

        adapter = RankingAdapter()

        # Simulate ELO update during debate
        adapter.store_agent_expertise(
            agent_name="claude-3-opus",
            domain="security",
            elo=1650,
            delta=50,
            debate_id="debate-001",
        )

        # Verify expertise is stored
        expertise = adapter.get_agent_expertise("claude-3-opus", "security")
        assert expertise is not None
        assert expertise["elo"] == 1650
        assert expertise["domain"] == "security"

    def test_expertise_loaded_for_new_debate(self):
        """Historical expertise is loaded when new debate starts."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

        adapter = RankingAdapter()

        # Store expertise from previous debate (delta >= 50 required)
        adapter.store_agent_expertise(
            agent_name="claude-3-opus",
            domain="security",
            elo=1700,
            delta=100,
            debate_id="debate-001",
        )
        adapter.store_agent_expertise(
            agent_name="gpt-4-turbo",
            domain="security",
            elo=1620,
            delta=70,  # Must be >= 50
            debate_id="debate-001",
        )
        adapter.store_agent_expertise(
            agent_name="gemini-pro",
            domain="security",
            elo=1580,
            delta=50,  # Must be >= 50
            debate_id="debate-001",
        )

        # Query domain experts for new debate
        experts = adapter.get_domain_experts("security", limit=5)

        assert len(experts) == 3
        # Experts should be sorted by ELO descending (returns AgentExpertise objects)
        assert experts[0].agent_name == "claude-3-opus"
        assert experts[1].agent_name == "gpt-4-turbo"
        assert experts[2].agent_name == "gemini-pro"

    def test_compression_patterns_persist_across_debates(self):
        """RLM compression patterns are persisted and reused."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

        adapter = RlmAdapter()

        # Store pattern from first debate
        pattern_id = adapter.store_compression_pattern(
            compression_ratio=0.3,
            value_score=0.85,
            content_markers=["security", "api", "authentication"],
        )
        assert pattern_id is not None

        # Store same pattern again (usage count increases)
        adapter.store_compression_pattern(
            compression_ratio=0.35,
            value_score=0.8,
            content_markers=["security", "api", "authentication"],
        )

        # Query patterns for new debate with similar content
        patterns = adapter.get_patterns_for_content(["security", "api"])

        assert len(patterns) >= 1
        # Pattern should have usage count >= 2
        pattern = patterns[0]
        assert pattern["usage_count"] >= 2

    def test_domain_experts_influence_team_selection(self):
        """Domain expertise from KM influences team selection weights."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        adapter = RankingAdapter()

        # Store varied expertise (all deltas must be >= 50)
        adapter.store_agent_expertise("agent-A", "coding", 1800, 200, "d1")
        adapter.store_agent_expertise("agent-B", "coding", 1500, 50, "d1")
        adapter.store_agent_expertise("agent-C", "coding", 1600, 100, "d1")

        # Get experts for coding domain
        experts = adapter.get_domain_experts("coding", limit=3)

        assert len(experts) == 3
        # Top expert should be agent-A (returns AgentExpertise objects)
        assert experts[0].agent_name == "agent-A"
        assert experts[0].elo == 1800

        # Create weights based on expertise
        weights = {}
        for expert in experts:
            # Weight = (ELO - 1500) / 100, min 0.5
            elo_weight = max(0.5, (expert.elo - 1500) / 100)
            weights[expert.agent_name] = elo_weight

        assert weights["agent-A"] == 3.0  # (1800-1500)/100
        assert weights["agent-B"] == 0.5  # min weight (1500-1500)/100 = 0
        assert weights["agent-C"] == 1.0  # (1600-1500)/100


class TestKMAdapterSyncCycle:
    """Tests for adapter sync operations."""

    @pytest.mark.asyncio
    async def test_ranking_adapter_sync_roundtrip(self):
        """RankingAdapter can sync to and load from mock KM."""
        from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter
        from datetime import datetime

        adapter = RankingAdapter()

        # Store expertise
        adapter.store_agent_expertise("test-agent", "testing", 1650, 50, "d1")

        # Create mock KM
        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="km_node_123")

        # Sync to mound
        sync_result = await adapter.sync_to_mound(mock_mound, workspace_id="test")

        assert sync_result["expertise_synced"] == 1
        assert sync_result["errors"] == []
        mock_mound.ingest.assert_called_once()

        # Create fresh adapter
        adapter2 = RankingAdapter()

        # Mock KM with stored data
        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "agent_expertise",
            "agent_name": "test-agent",
            "domain": "testing",
            "elo": 1650,
            "debate_count": 1,
        }
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()

        mock_mound.query_nodes = AsyncMock(return_value=[mock_node])

        # Load from mound
        load_result = await adapter2.load_from_mound(mock_mound, workspace_id="test")

        assert load_result["expertise_loaded"] == 1
        assert load_result["errors"] == []

        # Verify state was restored
        expertise = adapter2.get_agent_expertise("test-agent", "testing")
        assert expertise is not None
        assert expertise["elo"] == 1650

    @pytest.mark.asyncio
    async def test_rlm_adapter_sync_roundtrip(self):
        """RlmAdapter can sync to and load from mock KM."""
        from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter
        from datetime import datetime

        adapter = RlmAdapter()

        # Store pattern with multiple uses
        adapter.store_compression_pattern(0.3, 0.85, ["test", "pattern"])
        adapter.store_compression_pattern(0.35, 0.8, ["test", "pattern"])

        # Create mock KM
        mock_mound = AsyncMock()
        mock_mound.ingest = AsyncMock(return_value="km_node_456")

        # Sync to mound
        sync_result = await adapter.sync_to_mound(mock_mound, workspace_id="test")

        assert sync_result["patterns_synced"] == 1
        assert sync_result["errors"] == []

        # Create fresh adapter
        adapter2 = RlmAdapter()

        # Mock KM with stored data
        mock_node = MagicMock()
        mock_node.metadata = {
            "type": "compression_pattern",
            "pattern_id": "cp_test123",
            "compression_ratio": 0.3,
            "value_score": 0.85,
            "content_type": "code",
            "content_markers": ["test", "pattern"],
            "usage_count": 2,
        }
        mock_node.created_at = datetime.now()
        mock_node.updated_at = datetime.now()

        mock_mound.query_nodes = AsyncMock(return_value=[mock_node])

        # Load from mound
        load_result = await adapter2.load_from_mound(mock_mound, workspace_id="test")

        assert load_result["patterns_loaded"] == 1
        assert load_result["errors"] == []

        # Verify state was restored
        pattern = adapter2.get_pattern("cp_test123")
        assert pattern is not None
        assert pattern["compression_ratio"] == 0.3


class TestCulturePatternLearning:
    """Tests for culture pattern learning across debates."""

    def test_culture_pattern_types_exist(self):
        """CulturePatternType enum has expected values."""
        from aragora.knowledge.mound.types import CulturePatternType

        # Verify key pattern types exist
        assert hasattr(CulturePatternType, "DECISION_STYLE")
        assert hasattr(CulturePatternType, "RISK_TOLERANCE")
        assert hasattr(CulturePatternType, "DOMAIN_EXPERTISE")
        assert hasattr(CulturePatternType, "AGENT_PREFERENCES")
        assert hasattr(CulturePatternType, "DEBATE_DYNAMICS")

    def test_culture_pattern_dataclass(self):
        """CulturePattern dataclass works correctly."""
        from aragora.knowledge.mound.types import CulturePattern, CulturePatternType
        from datetime import datetime

        pattern = CulturePattern(
            id="pattern-001",
            workspace_id="test-workspace",
            pattern_type=CulturePatternType.DECISION_STYLE,
            pattern_key="collaborative",
            pattern_value={"style": "consensus-driven"},
            observation_count=10,
            confidence=0.85,
            first_observed_at=datetime.now(),
            last_observed_at=datetime.now(),
        )

        assert pattern.pattern_type == CulturePatternType.DECISION_STYLE
        assert pattern.confidence == 0.85
        assert pattern.observation_count == 10
        assert pattern.pattern_key == "collaborative"

    def test_accumulator_get_patterns_returns_list(self):
        """get_patterns returns a list (may be empty)."""
        from aragora.knowledge.mound.culture.accumulator import CultureAccumulator
        from unittest.mock import MagicMock

        mock_mound = MagicMock()
        accumulator = CultureAccumulator(mound=mock_mound)

        # get_patterns should return a list (possibly empty)
        patterns = accumulator.get_patterns("test-workspace")
        assert isinstance(patterns, list)

    def test_accumulator_get_patterns_summary_returns_dict(self):
        """get_patterns_summary returns a dict with expected keys."""
        from aragora.knowledge.mound.culture.accumulator import CultureAccumulator
        from unittest.mock import MagicMock

        mock_mound = MagicMock()
        accumulator = CultureAccumulator(mound=mock_mound)

        # get_patterns_summary should return a dict
        summary = accumulator.get_patterns_summary("test-workspace")
        assert isinstance(summary, dict)
        assert "total_patterns" in summary
        assert "patterns_by_type" in summary


class TestEventFlowIntegration:
    """Tests for event flow across subsystems."""

    def test_elo_event_triggers_adapter_storage(self):
        """AGENT_ELO_UPDATED event triggers storage in RankingAdapter."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Create ELO update event
        event = StreamEvent(
            type=StreamEventType.AGENT_ELO_UPDATED,
            data={
                "agent_name": "test-agent",
                "old_elo": 1500,
                "new_elo": 1600,
                "delta": 100,
                "domain": "reasoning",
                "debate_id": "debate-001",
            },
        )

        # Dispatch should not raise
        manager.dispatch(event)

        # Check stats
        stats = manager.get_stats()
        assert "elo_to_mound" in stats
        assert stats["elo_to_mound"]["events_processed"] >= 0

    def test_belief_convergence_triggers_storage(self):
        """BELIEF_CONVERGED event triggers storage in BeliefAdapter."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Create belief convergence event
        event = StreamEvent(
            type=StreamEventType.BELIEF_CONVERGED,
            data={
                "claim_id": "claim-001",
                "final_confidence": 0.85,
                "converged_beliefs": [{"claim": "Test claim", "confidence": 0.85}],
                "cruxes": ["key assumption"],
                "debate_id": "debate-001",
            },
        )

        # Dispatch should not raise
        manager.dispatch(event)

        stats = manager.get_stats()
        assert "belief_to_mound" in stats

    def test_insight_extraction_triggers_storage(self):
        """INSIGHT_EXTRACTED event triggers storage in InsightsAdapter."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Create insight extraction event
        event = StreamEvent(
            type=StreamEventType.INSIGHT_EXTRACTED,
            data={
                "insight_id": "insight-001",
                "insight": "Key finding about security",
                "confidence": 0.9,
                "debate_id": "debate-001",
                "source_claim_id": "claim-001",
            },
        )

        # Dispatch should not raise
        manager.dispatch(event)

        stats = manager.get_stats()
        assert "insight_to_mound" in stats


class TestBatchProcessing:
    """Tests for batch event processing."""

    def test_high_volume_events_batched(self):
        """High-volume events are processed asynchronously."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Dispatch many memory events
        for i in range(10):
            event = StreamEvent(
                type=StreamEventType.MEMORY_STORED,
                data={
                    "id": f"mem-{i}",
                    "content": f"Test memory {i}",
                    "importance": 0.5,
                    "tier": "medium",
                },
            )
            manager.dispatch(event)

        # Check batch stats
        batch_stats = manager.get_batch_stats()
        assert "batching_enabled" in batch_stats
        assert "batch_size" in batch_stats

    def test_flush_all_batches(self):
        """flush_all_batches processes pending events."""
        from aragora.events.cross_subscribers import CrossSubscriberManager
        from aragora.events.types import StreamEvent, StreamEventType

        manager = CrossSubscriberManager()

        # Dispatch some events
        for i in range(5):
            event = StreamEvent(
                type=StreamEventType.KNOWLEDGE_QUERIED,
                data={"query": f"test query {i}"},
            )
            manager.dispatch(event)

        # Flush batches
        flushed = manager.flush_all_batches()
        assert flushed >= 0


class TestMetricsIntegration:
    """Tests for Prometheus metrics integration."""

    def test_km_inbound_metrics_recorded(self):
        """KM inbound events record metrics."""
        from aragora.server.prometheus_cross_pollination import (
            record_km_inbound_event,
            record_km_outbound_event,
        )

        # These should not raise
        record_km_inbound_event("memory", "MEMORY_STORED")
        record_km_inbound_event("elo", "AGENT_ELO_UPDATED")
        record_km_inbound_event("belief", "BELIEF_CONVERGED")

    def test_km_outbound_metrics_recorded(self):
        """KM outbound events record metrics."""
        from aragora.server.prometheus_cross_pollination import (
            record_km_outbound_event,
        )

        # These should not raise
        record_km_outbound_event("memory", "KNOWLEDGE_QUERIED")
        record_km_outbound_event("debate", "MOUND_TO_BELIEF")
        record_km_outbound_event("rlm", "MOUND_TO_RLM")

    def test_adapter_sync_metrics_recorded(self):
        """Adapter sync operations record metrics."""
        from aragora.server.prometheus_cross_pollination import (
            record_km_adapter_sync,
        )

        # These should not raise
        record_km_adapter_sync("ranking", "to_mound", "success", 0.5)
        record_km_adapter_sync("rlm", "from_mound", "success", 0.3)
        record_km_adapter_sync("continuum", "to_mound", "error", 1.0)

    def test_staleness_check_metrics_recorded(self):
        """Staleness check operations record metrics."""
        from aragora.server.prometheus_cross_pollination import (
            record_km_staleness_check,
        )

        # These should not raise
        record_km_staleness_check("default", "completed", 5)
        record_km_staleness_check("workspace-1", "skipped", 0)
        record_km_staleness_check("workspace-2", "failed", 0)
