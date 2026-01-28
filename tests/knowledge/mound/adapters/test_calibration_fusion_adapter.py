"""
Tests for CalibrationFusionAdapter - Multi-Party Calibration Fusion for Knowledge Mound Phase A3.

Tests cover:
- Basic adapter initialization
- Fusing predictions from multiple agents
- Consensus storage and retrieval
- KnowledgeItem conversion
- FusionMixin integration
- Metrics recording
- Event emission
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


class TestCalibrationSearchResult:
    """Tests for CalibrationSearchResult dataclass."""

    def test_create_search_result(self):
        """Should create CalibrationSearchResult."""
        from aragora.knowledge.mound.adapters import CalibrationSearchResult
        from aragora.knowledge.mound.ops.calibration_fusion import (
            CalibrationConsensus,
            AgentPrediction,
            CalibrationFusionStrategy,
        )

        # Create a mock consensus
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a"),
            AgentPrediction("gpt-4", 0.75, "winner_a"),
        ]
        consensus = CalibrationConsensus(
            debate_id="test_debate",
            predictions=predictions,
            fused_confidence=0.78,
            predicted_outcome="winner_a",
            consensus_strength=0.85,
            agreement_ratio=1.0,
            disagreement_score=0.01,
            krippendorff_alpha=0.9,
            strategy_used=CalibrationFusionStrategy.WEIGHTED_AVERAGE,
            participating_agents=["claude", "gpt-4"],
        )

        result = CalibrationSearchResult(
            consensus=consensus,
            similarity=0.9,
            stored_at=datetime.now(timezone.utc),
        )

        assert result.consensus.debate_id == "test_debate"
        assert result.similarity == 0.9
        assert result.stored_at is not None

    def test_to_dict(self):
        """Should convert to dictionary."""
        from aragora.knowledge.mound.adapters import CalibrationSearchResult
        from aragora.knowledge.mound.ops.calibration_fusion import (
            CalibrationConsensus,
            CalibrationFusionStrategy,
        )

        consensus = CalibrationConsensus(
            debate_id="test_debate",
            predictions=[],
            fused_confidence=0.8,
            predicted_outcome="winner_a",
            consensus_strength=0.7,
            agreement_ratio=0.9,
            disagreement_score=0.05,
            krippendorff_alpha=0.8,
        )
        now = datetime.now(timezone.utc)
        result = CalibrationSearchResult(
            consensus=consensus,
            similarity=0.85,
            stored_at=now,
        )

        result_dict = result.to_dict()

        assert result_dict["similarity"] == 0.85
        assert result_dict["stored_at"] == now.isoformat()
        assert "consensus" in result_dict


class TestCalibrationFusionAdapterInit:
    """Tests for CalibrationFusionAdapter initialization."""

    def test_init_default(self):
        """Should initialize with default configuration."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter

        adapter = CalibrationFusionAdapter()

        assert adapter.adapter_name == "calibration_fusion"
        assert adapter.engine is not None
        assert adapter._enable_dual_write is False

    def test_init_with_custom_engine(self):
        """Should accept custom engine."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import CalibrationFusionEngine

        engine = CalibrationFusionEngine()
        adapter = CalibrationFusionAdapter(engine=engine)

        assert adapter.engine is engine

    def test_init_with_options(self):
        """Should accept optional parameters."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter

        callback = MagicMock()
        adapter = CalibrationFusionAdapter(
            enable_dual_write=True,
            event_callback=callback,
            enable_tracing=False,
        )

        assert adapter._enable_dual_write is True
        assert adapter._event_callback is callback


class TestFusePredictions:
    """Tests for fuse_predictions method."""

    def test_fuse_two_predictions(self):
        """Should fuse two predictions into consensus."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a", calibration_accuracy=0.9),
            AgentPrediction("gpt-4", 0.75, "winner_a", calibration_accuracy=0.85),
        ]

        consensus = adapter.fuse_predictions(
            predictions=predictions,
            debate_id="debate_123",
            store=True,
        )

        assert consensus.debate_id == "debate_123"
        assert consensus.predicted_outcome == "winner_a"
        assert 0.7 < consensus.fused_confidence < 0.85  # Weighted average
        assert consensus.agreement_ratio == 1.0  # Both agree
        assert len(consensus.participating_agents) == 2

    def test_fuse_with_disagreement(self):
        """Should handle predictions with disagreement."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a", calibration_accuracy=0.9),
            AgentPrediction("gpt-4", 0.75, "winner_a", calibration_accuracy=0.85),
            AgentPrediction("gemini", 0.6, "winner_b", calibration_accuracy=0.7),
        ]

        consensus = adapter.fuse_predictions(
            predictions=predictions,
            debate_id="debate_456",
        )

        assert consensus.debate_id == "debate_456"
        assert consensus.predicted_outcome == "winner_a"  # Majority
        assert consensus.agreement_ratio == pytest.approx(2 / 3, rel=0.01)
        assert consensus.consensus_strength < 1.0  # Reduced due to disagreement

    def test_fuse_with_strategy(self):
        """Should respect fusion strategy."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import (
            AgentPrediction,
            CalibrationFusionStrategy,
        )

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("claude", 0.9, "winner_a"),
            AgentPrediction("gpt-4", 0.5, "winner_a"),
            AgentPrediction("gemini", 0.7, "winner_a"),
        ]

        consensus = adapter.fuse_predictions(
            predictions=predictions,
            debate_id="debate_789",
            strategy=CalibrationFusionStrategy.MEDIAN,
        )

        # Median of [0.9, 0.7, 0.5] is 0.7
        assert consensus.fused_confidence == 0.7
        assert consensus.strategy_used == CalibrationFusionStrategy.MEDIAN


class TestConsensusStorage:
    """Tests for consensus storage and retrieval."""

    def test_store_and_retrieve(self):
        """Should store and retrieve consensus."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a"),
            AgentPrediction("gpt-4", 0.75, "winner_a"),
        ]

        consensus = adapter.fuse_predictions(
            predictions=predictions,
            debate_id="stored_debate",
            store=True,
        )

        retrieved = adapter.get_consensus("stored_debate")

        assert retrieved is not None
        assert retrieved.debate_id == "stored_debate"
        assert retrieved.fused_confidence == consensus.fused_confidence

    def test_retrieve_nonexistent(self):
        """Should return None for nonexistent debate."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter

        adapter = CalibrationFusionAdapter()

        retrieved = adapter.get_consensus("nonexistent")

        assert retrieved is None


class TestSearchByTopic:
    """Tests for search_by_topic method."""

    def test_search_by_topic(self):
        """Should search consensus by topic."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        adapter = CalibrationFusionAdapter()

        # Create some consensus with topics
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a"),
            AgentPrediction("gpt-4", 0.75, "winner_a"),
        ]

        consensus1 = adapter.fuse_predictions(
            predictions=predictions,
            debate_id="debate_1",
            store=True,
        )
        consensus1.metadata["topic"] = "rate limiting"
        adapter._store_consensus(consensus1)

        consensus2 = adapter.fuse_predictions(
            predictions=predictions,
            debate_id="debate_2",
            store=True,
        )
        consensus2.metadata["topic"] = "rate optimization"
        adapter._store_consensus(consensus2)

        results = adapter.search_by_topic("rate", limit=10)

        assert len(results) >= 1


class TestToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_convert_to_knowledge_item(self):
        """Should convert consensus to KnowledgeItem."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction
        from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a"),
            AgentPrediction("gpt-4", 0.75, "winner_a"),
        ]

        consensus = adapter.fuse_predictions(
            predictions=predictions,
            debate_id="knowledge_item_test",
        )

        item = adapter.to_knowledge_item(consensus)

        assert item.id == "cf_knowledge_item_test"
        assert item.source == KnowledgeSource.CALIBRATION
        assert item.source_id == "knowledge_item_test"
        assert "Calibration consensus" in item.content
        assert item.metadata["predicted_outcome"] == "winner_a"
        assert item.metadata["agreement_ratio"] == 1.0

    def test_confidence_level_mapping(self):
        """Should map consensus strength to confidence level."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import (
            CalibrationConsensus,
            CalibrationFusionStrategy,
        )
        from aragora.knowledge.unified.types import ConfidenceLevel

        adapter = CalibrationFusionAdapter()

        # High strength
        high_consensus = CalibrationConsensus(
            debate_id="high",
            predictions=[],
            fused_confidence=0.9,
            predicted_outcome="winner_a",
            consensus_strength=0.85,
            agreement_ratio=0.95,
            disagreement_score=0.01,
            krippendorff_alpha=0.95,
        )
        high_item = adapter.to_knowledge_item(high_consensus)
        assert high_item.confidence == ConfidenceLevel.VERIFIED

        # Low strength
        low_consensus = CalibrationConsensus(
            debate_id="low",
            predictions=[],
            fused_confidence=0.4,
            predicted_outcome="winner_b",
            consensus_strength=0.15,
            agreement_ratio=0.5,
            disagreement_score=0.3,
            krippendorff_alpha=0.2,
        )
        low_item = adapter.to_knowledge_item(low_consensus)
        assert low_item.confidence == ConfidenceLevel.UNVERIFIED


class TestGetStats:
    """Tests for statistics methods."""

    def test_get_stats(self):
        """Should return fusion statistics."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a"),
            AgentPrediction("gpt-4", 0.75, "winner_a"),
        ]

        adapter.fuse_predictions(predictions, "test_stats")

        stats = adapter.get_stats()

        assert stats["adapter_name"] == "calibration_fusion"
        assert stats["stored_consensus_count"] >= 1
        assert "total_fusions" in stats

    def test_get_agent_performance(self):
        """Should return agent performance metrics."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("test_agent", 0.8, "winner_a"),
            AgentPrediction("other_agent", 0.75, "winner_a"),
        ]

        adapter.fuse_predictions(predictions, "agent_test")

        performance = adapter.get_agent_performance("test_agent")

        assert performance["agent_name"] == "test_agent"
        assert "fusion_count" in performance


class TestFusionMixin:
    """Tests for FusionMixin integration."""

    def test_get_fusion_sources(self):
        """Should return list of fusion sources."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter

        adapter = CalibrationFusionAdapter()

        sources = adapter._get_fusion_sources()

        assert "elo" in sources
        assert "consensus" in sources
        assert "belief" in sources

    def test_extract_fusible_data(self):
        """Should extract fusible data from KM item."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter

        adapter = CalibrationFusionAdapter()

        km_item = {
            "id": "test_item",
            "confidence": 0.8,
            "metadata": {
                "source_id": "source_123",
                "source_adapter": "consensus",
            },
        }

        fusible = adapter._extract_fusible_data(km_item)

        assert fusible is not None
        assert fusible["item_id"] == "source_123"
        assert fusible["confidence"] == 0.8
        assert fusible["source_adapter"] == "consensus"

    def test_extract_fusible_data_missing_id(self):
        """Should return None for item without ID."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter

        adapter = CalibrationFusionAdapter()

        km_item = {
            "confidence": 0.8,
            "metadata": {},
        }

        fusible = adapter._extract_fusible_data(km_item)

        assert fusible is None


class TestEventEmission:
    """Tests for event emission."""

    def test_emits_fusion_events(self):
        """Should emit events during fusion."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        callback = MagicMock()
        adapter = CalibrationFusionAdapter(event_callback=callback)

        predictions = [
            AgentPrediction("claude", 0.8, "winner_a"),
            AgentPrediction("gpt-4", 0.75, "winner_a"),
        ]

        adapter.fuse_predictions(predictions, "event_test")

        # Should have emitted start and complete events
        assert callback.call_count >= 2

        # Check event types
        event_types = [call[0][0] for call in callback.call_args_list]
        assert "calibration_fusion_start" in event_types
        assert "calibration_fusion_complete" in event_types


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check(self):
        """Should return health status."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter

        adapter = CalibrationFusionAdapter()

        health = adapter.health_check()

        assert health["adapter"] == "calibration_fusion"
        assert "healthy" in health
        assert "stored_consensus_count" in health
        assert "engine_stats" in health


class TestSyncToKM:
    """Tests for sync_to_km method."""

    def test_sync_stored_consensus(self):
        """Should sync stored consensus."""
        from aragora.knowledge.mound.adapters import CalibrationFusionAdapter
        from aragora.knowledge.mound.ops.calibration_fusion import AgentPrediction

        adapter = CalibrationFusionAdapter()
        predictions = [
            AgentPrediction("claude", 0.8, "winner_a"),
            AgentPrediction("gpt-4", 0.75, "winner_a"),
        ]

        adapter.fuse_predictions(predictions, "sync_test_1", store=True)
        adapter.fuse_predictions(predictions, "sync_test_2", store=True)

        result = adapter.sync_to_km()

        assert result["consensus_stored"] == 2
        assert result["predictions_processed"] >= 4
        assert "duration_ms" in result
