"""Integration tests for Knowledge Mound bidirectional flows.

Tests the actual integration between adapters and source systems:
- Data flow IN: Source systems → KM adapters → Knowledge Mound
- Data flow OUT: Knowledge Mound → KM adapters → Source systems (reverse queries)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


class TestEvidenceKMFlow:
    """Tests for Evidence ↔ Knowledge Mound bidirectional flow."""

    def test_evidence_store_has_km_adapter(self):
        """Evidence store accepts and stores KM adapter."""
        from aragora.evidence.store import EvidenceStore
        from aragora.knowledge.mound.adapters.evidence_adapter import EvidenceAdapter

        # Create mock adapter
        mock_adapter = Mock(spec=EvidenceAdapter)

        # Create store with adapter
        store = EvidenceStore(km_adapter=mock_adapter, km_min_reliability=0.6)

        # Verify adapter is stored
        assert store._km_adapter is mock_adapter
        assert store._km_min_reliability == 0.6

    def test_evidence_store_set_km_adapter(self):
        """Evidence store can set KM adapter after init."""
        from aragora.evidence.store import EvidenceStore
        from aragora.knowledge.mound.adapters.evidence_adapter import EvidenceAdapter

        store = EvidenceStore()
        assert store._km_adapter is None

        mock_adapter = Mock(spec=EvidenceAdapter)
        store.set_km_adapter(mock_adapter)

        assert store._km_adapter is mock_adapter

    def test_evidence_store_queries_km_for_similar(self):
        """Evidence store can query KM for similar evidence."""
        from aragora.evidence.store import EvidenceStore
        from aragora.knowledge.mound.adapters.evidence_adapter import EvidenceAdapter

        mock_adapter = Mock(spec=EvidenceAdapter)
        mock_adapter.search_similar.return_value = [
            {"id": "ev_1", "snippet": "similar content"},
        ]

        store = EvidenceStore(km_adapter=mock_adapter)

        # Query for similar evidence
        results = store.query_km_for_similar("test content")

        mock_adapter.search_similar.assert_called_once_with(
            content="test content",
            limit=5,
            min_similarity=0.7,
        )
        assert len(results) == 1

    def test_evidence_store_queries_km_for_topic(self):
        """Evidence store can query KM for topic evidence."""
        from aragora.evidence.store import EvidenceStore
        from aragora.knowledge.mound.adapters.evidence_adapter import EvidenceAdapter

        mock_adapter = Mock(spec=EvidenceAdapter)
        mock_adapter.search_by_topic.return_value = [
            {"id": "ev_1", "snippet": "relevant evidence"},
        ]

        store = EvidenceStore(km_adapter=mock_adapter)

        # Query for topic evidence
        results = store.query_km_for_topic("machine learning")

        mock_adapter.search_by_topic.assert_called_once_with(
            query="machine learning",
            limit=10,
            min_reliability=0.0,
        )
        assert len(results) == 1

    def test_evidence_store_handles_km_failure_gracefully(self):
        """Evidence store handles KM query failures gracefully."""
        from aragora.evidence.store import EvidenceStore
        from aragora.knowledge.mound.adapters.evidence_adapter import EvidenceAdapter

        mock_adapter = Mock(spec=EvidenceAdapter)
        mock_adapter.search_similar.side_effect = Exception("KM unavailable")

        store = EvidenceStore(km_adapter=mock_adapter)

        # Should not raise, returns empty list
        results = store.query_km_for_similar("test")
        assert results == []


class TestEloKMFlow:
    """Tests for ELO ↔ Knowledge Mound bidirectional flow."""

    def test_elo_system_queries_km_for_skill_history(self):
        """ELO system can query KM for agent skill history."""
        from aragora.ranking.elo import EloSystem
        from aragora.knowledge.mound.adapters.elo_adapter import EloAdapter

        mock_adapter = Mock(spec=EloAdapter)
        mock_adapter.get_agent_skill_history.return_value = [
            {"elo": 1550, "domain": "ai", "timestamp": "2024-01-01"},
        ]

        elo = EloSystem(km_adapter=mock_adapter)

        # Query skill history
        results = elo.query_km_agent_skill_history("claude", domain="ai")

        mock_adapter.get_agent_skill_history.assert_called_once_with(
            agent_name="claude",
            domain="ai",
            limit=50,
        )
        assert len(results) == 1

    def test_elo_system_queries_km_for_domain_expertise(self):
        """ELO system can query KM for domain expertise."""
        from aragora.ranking.elo import EloSystem
        from aragora.knowledge.mound.adapters.elo_adapter import EloAdapter

        mock_adapter = Mock(spec=EloAdapter)
        mock_adapter.get_domain_expertise.return_value = [
            {"agent": "claude", "expertise": 0.9},
            {"agent": "gpt", "expertise": 0.85},
        ]

        elo = EloSystem(km_adapter=mock_adapter)

        # Query domain expertise
        results = elo.query_km_domain_expertise("machine_learning")

        mock_adapter.get_domain_expertise.assert_called_once_with(
            domain="machine_learning",
            limit=10,
        )
        assert len(results) == 2


class TestBeliefKMFlow:
    """Tests for Belief Network ↔ Knowledge Mound bidirectional flow."""

    def test_belief_network_has_km_adapter(self):
        """Belief network accepts KM adapter."""
        from aragora.reasoning.belief import BeliefNetwork
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        mock_adapter = Mock(spec=BeliefAdapter)
        network = BeliefNetwork(km_adapter=mock_adapter)

        assert network._km_adapter is mock_adapter

    def test_belief_network_query_related_beliefs_no_adapter(self):
        """Belief network returns empty list when no adapter."""
        from aragora.reasoning.belief import BeliefNetwork

        network = BeliefNetwork()
        results = network.query_km_related_beliefs("AI safety")

        assert results == []

    def test_belief_network_query_historical_cruxes_no_adapter(self):
        """Belief network returns empty list when no adapter."""
        from aragora.reasoning.belief import BeliefNetwork

        network = BeliefNetwork()
        results = network.query_km_historical_cruxes("consciousness")

        assert results == []

    def test_belief_network_seed_from_km_no_adapter(self):
        """Belief network seeds 0 beliefs when no adapter."""
        from aragora.reasoning.belief import BeliefNetwork

        network = BeliefNetwork()
        seeded = network.seed_from_km("AI safety")

        assert seeded == 0
        assert len(network.nodes) == 0

    def test_belief_network_set_km_adapter(self):
        """Belief network can set adapter after init."""
        from aragora.reasoning.belief import BeliefNetwork
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        network = BeliefNetwork()
        assert network._km_adapter is None

        mock_adapter = Mock(spec=BeliefAdapter)
        network.set_km_adapter(mock_adapter)

        assert network._km_adapter is mock_adapter


class TestPulseKMFlow:
    """Tests for Pulse ↔ Knowledge Mound bidirectional flow."""

    def test_scheduler_queries_km_for_past_debates(self):
        """Pulse scheduler can query KM for past debates."""
        from aragora.pulse.scheduler import PulseDebateScheduler
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        mock_adapter = Mock(spec=PulseAdapter)
        mock_adapter.search_past_debates.return_value = [
            {"topic": "AI regulation", "consensus_reached": True},
        ]

        # Create minimal scheduler
        mock_pulse_manager = Mock()
        mock_store = Mock()
        scheduler = PulseDebateScheduler(
            mock_pulse_manager,
            mock_store,
            km_adapter=mock_adapter,
        )

        # Query past debates
        results = scheduler.query_km_for_past_debates("AI regulation")

        mock_adapter.search_past_debates.assert_called_once_with(
            query="AI regulation",
            limit=5,
        )
        assert len(results) == 1


class TestInsightsKMFlow:
    """Tests for Insights ↔ Knowledge Mound bidirectional flow."""

    def test_insight_store_has_km_adapter(self):
        """Insight store accepts KM adapter."""
        from aragora.insights.store import InsightStore
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        mock_adapter = Mock(spec=InsightsAdapter)

        store = InsightStore(km_adapter=mock_adapter)

        assert store._km_adapter is mock_adapter

    def test_flip_detector_has_km_adapter(self):
        """Flip detector accepts KM adapter."""
        from aragora.insights.flip_detector import FlipDetector
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        mock_adapter = Mock(spec=InsightsAdapter)

        detector = FlipDetector(km_adapter=mock_adapter)

        assert detector._km_adapter is mock_adapter


class TestCostKMFlow:
    """Tests for Cost Tracker ↔ Knowledge Mound bidirectional flow."""

    def test_cost_tracker_has_km_adapter(self):
        """Cost tracker accepts KM adapter."""
        from aragora.billing.cost_tracker import CostTracker
        from aragora.knowledge.mound.adapters.cost_adapter import CostAdapter

        mock_adapter = Mock(spec=CostAdapter)
        tracker = CostTracker(km_adapter=mock_adapter)

        assert tracker._km_adapter is mock_adapter

    def test_cost_tracker_set_km_adapter(self):
        """Cost tracker can set KM adapter after init."""
        from aragora.billing.cost_tracker import CostTracker
        from aragora.knowledge.mound.adapters.cost_adapter import CostAdapter

        tracker = CostTracker()
        assert tracker._km_adapter is None

        mock_adapter = Mock(spec=CostAdapter)
        tracker.set_km_adapter(mock_adapter)

        assert tracker._km_adapter is mock_adapter

    def test_cost_tracker_queries_km_for_patterns(self):
        """Cost tracker can query KM for cost patterns."""
        from aragora.billing.cost_tracker import CostTracker
        from aragora.knowledge.mound.adapters.cost_adapter import CostAdapter

        mock_adapter = Mock(spec=CostAdapter)
        mock_adapter.get_cost_patterns.return_value = {
            "workspace_id": "ws_1",
            "avg_cost": 10.5,
            "sample_size": 30,
        }

        tracker = CostTracker(km_adapter=mock_adapter)

        # Query cost patterns
        results = tracker.query_km_cost_patterns("ws_1")

        mock_adapter.get_cost_patterns.assert_called_once_with("ws_1", None)
        assert results["avg_cost"] == 10.5

    def test_cost_tracker_queries_km_for_alerts(self):
        """Cost tracker can query KM for historical alerts."""
        from aragora.billing.cost_tracker import CostTracker
        from aragora.knowledge.mound.adapters.cost_adapter import CostAdapter

        mock_adapter = Mock(spec=CostAdapter)
        mock_adapter.get_workspace_alerts.return_value = [
            {"id": "ct_alert_1", "level": "warning"},
        ]

        tracker = CostTracker(km_adapter=mock_adapter)

        # Query alerts
        results = tracker.query_km_workspace_alerts("ws_1")

        mock_adapter.get_workspace_alerts.assert_called_once_with("ws_1", "warning", 50)
        assert len(results) == 1

    def test_cost_tracker_handles_km_failure_gracefully(self):
        """Cost tracker handles KM query failures gracefully."""
        from aragora.billing.cost_tracker import CostTracker
        from aragora.knowledge.mound.adapters.cost_adapter import CostAdapter

        mock_adapter = Mock(spec=CostAdapter)
        mock_adapter.get_cost_patterns.side_effect = Exception("KM unavailable")

        tracker = CostTracker(km_adapter=mock_adapter)

        # Should not raise, returns empty dict
        results = tracker.query_km_cost_patterns("ws_1")
        assert results == {}

    def test_cost_tracker_queries_return_empty_without_adapter(self):
        """Cost tracker returns empty results when no adapter."""
        from aragora.billing.cost_tracker import CostTracker

        tracker = CostTracker()

        assert tracker.query_km_cost_patterns("ws_1") == {}
        assert tracker.query_km_workspace_alerts("ws_1") == []


class TestCruxDetectorKMFlow:
    """Tests for Crux Detector ↔ Knowledge Mound bidirectional flow."""

    def test_crux_detector_has_km_adapter(self):
        """Crux detector accepts KM adapter."""
        from aragora.reasoning.belief import BeliefNetwork, CruxDetector
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        mock_adapter = Mock(spec=BeliefAdapter)
        network = BeliefNetwork()

        detector = CruxDetector(network, km_adapter=mock_adapter)

        assert detector._km_adapter is mock_adapter


class TestAdapterIntegration:
    """Tests for adapter cross-integration."""

    def test_all_adapters_importable(self):
        """All adapters can be imported together."""
        from aragora.knowledge.mound.adapters import (
            EvidenceAdapter,
            BeliefAdapter,
            InsightsAdapter,
            EloAdapter,
            PulseAdapter,
            CostAdapter,
        )

        # Verify all adapters have required methods
        assert hasattr(EvidenceAdapter, "search_by_topic")
        assert hasattr(BeliefAdapter, "store_converged_belief")
        assert hasattr(InsightsAdapter, "store_insight")
        assert hasattr(EloAdapter, "store_rating")
        assert hasattr(PulseAdapter, "store_trending_topic")
        assert hasattr(CostAdapter, "store_alert")

    def test_adapters_share_knowledge_source_enum(self):
        """All adapters use consistent KnowledgeSource values."""
        from aragora.knowledge.unified.types import KnowledgeSource

        # Verify all bidirectional sources exist
        assert KnowledgeSource.EVIDENCE.value == "evidence"
        assert KnowledgeSource.BELIEF.value == "belief"
        assert KnowledgeSource.INSIGHT.value == "insight"
        assert KnowledgeSource.FLIP.value == "flip"
        assert KnowledgeSource.ELO.value == "elo"
        assert KnowledgeSource.PULSE.value == "pulse"
        assert KnowledgeSource.COST.value == "cost"

    def test_adapters_convert_to_knowledge_items(self):
        """Adapters can convert their data to KnowledgeItem format."""
        from aragora.knowledge.mound.adapters import EvidenceAdapter, EloAdapter
        from aragora.knowledge.unified.types import KnowledgeSource

        # Test EvidenceAdapter conversion
        evidence_adapter = EvidenceAdapter()
        evidence = {
            "id": "123",
            "snippet": "Test content",
            "source": "web",
            "reliability_score": 0.9,
        }
        item = evidence_adapter.to_knowledge_item(evidence)
        assert item.source == KnowledgeSource.EVIDENCE
        assert item.content == "Test content"

        # Test EloAdapter conversion
        elo_adapter = EloAdapter()
        rating = {
            "id": "r_123",
            "agent_name": "claude",
            "elo": 1600,
            "domain": "ai",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        item = elo_adapter.to_knowledge_item(rating)
        assert item.source == KnowledgeSource.ELO
