"""
Integration tests for Phase 10: Bidirectional Knowledge Mound Integration.

Tests the complete bidirectional data flow between the Knowledge Mound
and connected subsystems (ContinuumMemory, ELO, OutcomeTracker, etc.).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


class TestPhase10CoreComponents:
    """Tests for Phase 10 core component imports and initialization."""

    def test_bidirectional_coordinator_import(self):
        """Test BidirectionalCoordinator can be imported."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
            CoordinatorConfig,
            SyncResult,
            BidirectionalSyncReport,
        )

        assert BidirectionalCoordinator is not None
        assert CoordinatorConfig is not None

    def test_km_elo_bridge_import(self):
        """Test KMEloBridge can be imported."""
        from aragora.ranking.km_elo_bridge import (
            KMEloBridge,
            KMEloBridgeConfig,
            KMEloBridgeSyncResult,
        )

        assert KMEloBridge is not None
        assert KMEloBridgeConfig is not None

    def test_km_outcome_bridge_import(self):
        """Test KMOutcomeBridge can be imported."""
        from aragora.debate.km_outcome_bridge import (
            KMOutcomeBridge,
            KMOutcomeBridgeConfig,
            OutcomeValidation,
            PropagationResult,
        )

        assert KMOutcomeBridge is not None
        assert OutcomeValidation is not None

    def test_elo_adapter_bidirectional_import(self):
        """Test EloAdapter bidirectional methods exist."""
        from aragora.knowledge.mound.adapters.elo_adapter import (
            EloAdapter,
            KMEloPattern,
            EloAdjustmentRecommendation,
            EloSyncResult,
        )

        assert EloAdapter is not None
        assert KMEloPattern is not None

        # Verify bidirectional methods exist
        adapter = EloAdapter()
        assert hasattr(adapter, "analyze_km_patterns_for_agent")
        assert hasattr(adapter, "compute_elo_adjustment")
        assert hasattr(adapter, "apply_km_elo_adjustment")
        assert hasattr(adapter, "sync_km_to_elo")

    def test_arena_config_phase10_flags(self):
        """Test ArenaConfig has Phase 10 configuration flags."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()

        # Master switch
        assert hasattr(config, "enable_km_bidirectional")

        # Individual sync flags
        assert hasattr(config, "enable_km_continuum_sync")
        assert hasattr(config, "enable_km_elo_sync")
        assert hasattr(config, "enable_km_outcome_validation")
        assert hasattr(config, "enable_km_belief_sync")
        assert hasattr(config, "enable_km_flip_sync")
        assert hasattr(config, "enable_km_critique_sync")
        assert hasattr(config, "enable_km_pulse_sync")

        # Global settings
        assert hasattr(config, "km_sync_interval_seconds")
        assert hasattr(config, "km_min_confidence_for_reverse")
        assert hasattr(config, "km_parallel_sync")


class TestBidirectionalCoordinator:
    """Tests for BidirectionalCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator with mock adapters."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
            CoordinatorConfig,
        )

        return BidirectionalCoordinator(
            config=CoordinatorConfig(
                parallel_sync=False,  # Sequential for deterministic testing
                timeout_seconds=5.0,
            )
        )

    def test_adapter_registration(self, coordinator):
        """Test adapter registration."""
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = MagicMock(return_value={"items_processed": 5})

        result = coordinator.register_adapter(
            name="test_adapter",
            adapter=mock_adapter,
            forward_method="sync_to_km",
            reverse_method=None,
            priority=1,
        )

        assert result is True
        assert "test_adapter" in coordinator.get_registered_adapters()

    def test_adapter_unregistration(self, coordinator):
        """Test adapter unregistration."""
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = MagicMock()

        coordinator.register_adapter(
            name="test_adapter",
            adapter=mock_adapter,
            forward_method="sync_to_km",
        )

        result = coordinator.unregister_adapter("test_adapter")

        assert result is True
        assert "test_adapter" not in coordinator.get_registered_adapters()

    def test_get_status(self, coordinator):
        """Test coordinator status reporting."""
        status = coordinator.get_status()

        assert "total_adapters" in status
        assert "enabled_adapters" in status
        assert "bidirectional_adapters" in status
        assert "sync_in_progress" in status
        assert "config" in status

    @pytest.mark.asyncio
    async def test_forward_sync(self, coordinator):
        """Test forward sync (sources → KM)."""
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = MagicMock(
            return_value={"items_processed": 10, "items_updated": 5}
        )

        coordinator.register_adapter(
            name="test_adapter",
            adapter=mock_adapter,
            forward_method="sync_to_km",
        )

        results = await coordinator.sync_all_to_km()

        assert len(results) == 1
        assert results[0].adapter_name == "test_adapter"
        assert results[0].direction == "forward"
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_bidirectional_sync(self, coordinator):
        """Test full bidirectional sync cycle."""
        mock_adapter = MagicMock()
        mock_adapter.sync_to_km = MagicMock(return_value={"items_processed": 10})
        mock_adapter.sync_from_km = MagicMock(return_value={"items_updated": 3})

        coordinator.register_adapter(
            name="test_adapter",
            adapter=mock_adapter,
            forward_method="sync_to_km",
            reverse_method="sync_from_km",
        )

        report = await coordinator.run_bidirectional_sync(km_items=[{"id": "km_1"}])

        assert report.total_adapters == 1
        assert len(report.forward_results) == 1
        assert len(report.reverse_results) == 1


class TestKMOutcomeBridgeIntegration:
    """Integration tests for KMOutcomeBridge."""

    @pytest.fixture
    def mock_outcome(self):
        """Create mock consensus outcome."""

        @dataclass
        class MockOutcome:
            debate_id: str = "debate_123"
            consensus_text: str = "Test consensus"
            consensus_confidence: float = 0.85
            implementation_succeeded: bool = True

        return MockOutcome()

    @pytest.fixture
    def bridge(self):
        """Create bridge with mocks."""
        from aragora.debate.km_outcome_bridge import (
            KMOutcomeBridge,
            KMOutcomeBridgeConfig,
        )

        mock_km = MagicMock()
        mock_km.get = AsyncMock(return_value={"id": "km_1", "confidence": 0.7})
        mock_km.update_confidence = AsyncMock(return_value=True)
        mock_km.get_graph_neighbors = AsyncMock(return_value=[])

        return KMOutcomeBridge(
            outcome_tracker=MagicMock(),
            knowledge_mound=mock_km,
            config=KMOutcomeBridgeConfig(auto_propagate=False),
        )

    def test_record_km_usage(self, bridge):
        """Test recording KM usage in debates."""
        bridge.record_km_usage("debate_123", ["km_1", "km_2", "km_3"])

        usage = bridge.get_km_usage("debate_123")

        assert len(usage) == 3
        assert "km_1" in usage

    @pytest.mark.asyncio
    async def test_validate_knowledge_success(self, bridge, mock_outcome):
        """Test validating knowledge from successful outcome."""
        bridge.record_km_usage("debate_123", ["km_1"])

        validations = await bridge.validate_knowledge_from_outcome(mock_outcome)

        assert len(validations) == 1
        assert validations[0].was_successful is True
        assert validations[0].confidence_adjustment > 0  # Boost for success

    @pytest.mark.asyncio
    async def test_validate_knowledge_failure(self, bridge, mock_outcome):
        """Test validating knowledge from failed outcome."""
        mock_outcome.implementation_succeeded = False
        bridge.record_km_usage("debate_123", ["km_1"])

        validations = await bridge.validate_knowledge_from_outcome(mock_outcome)

        assert len(validations) == 1
        assert validations[0].was_successful is False
        assert validations[0].confidence_adjustment < 0  # Penalty for failure

    def test_validation_stats(self, bridge):
        """Test getting validation statistics."""
        stats = bridge.get_validation_stats()

        assert "total_validations" in stats
        assert "success_validations" in stats
        assert "failure_validations" in stats
        assert "config" in stats


class TestKMEloBridgeIntegration:
    """Integration tests for KMEloBridge."""

    @pytest.fixture
    def bridge(self):
        """Create bridge with mocks."""
        from aragora.ranking.km_elo_bridge import KMEloBridge, KMEloBridgeConfig

        @dataclass
        class MockRating:
            agent_name: str
            elo: float = 1000.0

        mock_elo = MagicMock()
        mock_elo.get_all_ratings = MagicMock(
            return_value=[MockRating("claude"), MockRating("gpt4")]
        )

        @dataclass
        class MockPattern:
            agent_name: str
            pattern_type: str
            confidence: float = 0.8

        @dataclass
        class MockSyncResult:
            adjustments_recommended: int = 1
            adjustments_applied: int = 1
            adjustments_skipped: int = 0
            total_elo_change: float = 15.0
            agents_affected: List[str] = field(default_factory=lambda: ["claude"])

        mock_adapter = MagicMock()
        mock_adapter.analyze_km_patterns_for_agent = AsyncMock(
            return_value=[MockPattern("claude", "success_contributor")]
        )
        mock_adapter.sync_km_to_elo = AsyncMock(return_value=MockSyncResult())
        mock_adapter.get_agent_km_patterns = MagicMock(return_value=[])
        mock_adapter.get_pending_adjustments = MagicMock(return_value=[])
        mock_adapter.clear_pending_adjustments = MagicMock(return_value=0)

        mock_km = MagicMock()
        mock_km.query_by_agent = AsyncMock(
            return_value=[
                {"id": "km_1", "agent": "claude"},
                {"id": "km_2", "agent": "claude"},
                {"id": "km_3", "agent": "claude"},
                {"id": "km_4", "agent": "claude"},
                {"id": "km_5", "agent": "claude"},
            ]
        )

        return KMEloBridge(
            elo_system=mock_elo,
            elo_adapter=mock_adapter,
            knowledge_mound=mock_km,
            config=KMEloBridgeConfig(
                sync_interval_hours=0,  # No interval restriction
                min_km_items_for_pattern=3,  # Lower threshold for testing
            ),
        )

    @pytest.mark.asyncio
    async def test_sync_analyzes_patterns(self, bridge):
        """Test sync analyzes patterns from KM."""
        result = await bridge.sync_km_to_elo(force=True)

        assert result.agents_analyzed == 2
        assert result.patterns_detected >= 1

    @pytest.mark.asyncio
    async def test_sync_applies_adjustments(self, bridge):
        """Test sync applies ELO adjustments."""
        bridge._config.auto_apply = True

        result = await bridge.sync_km_to_elo(force=True)

        assert result.adjustments_applied == 1
        assert result.total_elo_change == 15.0

    def test_status_reporting(self, bridge):
        """Test bridge status reporting."""
        status = bridge.get_status()

        assert status["elo_system_available"] is True
        assert status["elo_adapter_available"] is True
        assert status["knowledge_mound_available"] is True
        assert "config" in status


class TestHookHandlerKMIntegration:
    """Tests for KM hook handlers in HookHandlerRegistry."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""

        class MockHookManager:
            def __init__(self):
                self.hooks = {}

            def register(self, hook_type, callback, name=None, priority=None):
                key = (hook_type, name)
                self.hooks[key] = callback
                return lambda: self.hooks.pop(key, None)

        return MockHookManager()

    def test_km_hooks_registered(self, hook_manager):
        """Test KM hooks are registered when subsystems provided."""
        from aragora.debate.hook_handlers import HookHandlerRegistry

        mock_km = MagicMock()
        mock_km.on_debate_end = MagicMock()
        mock_km.on_consensus_reached = MagicMock()
        mock_km.on_outcome_tracked = MagicMock()

        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"knowledge_mound": mock_km},
        )

        count = registry.register_all()

        # Should have registered KM hooks
        assert count >= 1

        # Verify hooks exist
        hook_names = [name for (_, name) in hook_manager.hooks.keys()]
        assert any("km_" in name for name in hook_names)

    def test_km_coordinator_hooks_registered(self, hook_manager):
        """Test KM coordinator hooks are registered."""
        from aragora.debate.hook_handlers import HookHandlerRegistry

        mock_coordinator = MagicMock()
        mock_coordinator.on_debate_complete = MagicMock()
        mock_coordinator.on_consensus_reached = MagicMock()

        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems={"km_coordinator": mock_coordinator},
        )

        count = registry.register_all()

        assert count >= 1
        hook_names = [name for (_, name) in hook_manager.hooks.keys()]
        assert any("coordinator" in name for name in hook_names)


class TestAdapterBidirectionalIntegration:
    """Tests for adapter bidirectional capabilities."""

    def test_continuum_adapter_bidirectional(self):
        """Test ContinuumAdapter has bidirectional methods."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        # ContinuumAdapter requires a continuum instance
        mock_continuum = MagicMock()
        adapter = ContinuumAdapter(continuum=mock_continuum)

        # Forward methods
        assert hasattr(adapter, "sync_memory_to_mound")
        assert hasattr(adapter, "search_by_keyword")

        # Reverse methods (bidirectional)
        assert hasattr(adapter, "update_continuum_from_km")
        assert hasattr(adapter, "sync_validations_to_continuum")
        assert hasattr(adapter, "get_km_validated_entries")

    def test_elo_adapter_bidirectional(self):
        """Test EloAdapter has bidirectional methods."""
        from aragora.knowledge.mound.adapters.elo_adapter import EloAdapter

        adapter = EloAdapter()

        # Forward methods
        assert hasattr(adapter, "store_rating")
        assert hasattr(adapter, "store_match")

        # Reverse methods (bidirectional)
        assert hasattr(adapter, "analyze_km_patterns_for_agent")
        assert hasattr(adapter, "compute_elo_adjustment")
        assert hasattr(adapter, "apply_km_elo_adjustment")
        assert hasattr(adapter, "sync_km_to_elo")

    def test_belief_adapter_has_threshold_methods(self):
        """Test BeliefAdapter has threshold update methods."""
        from aragora.knowledge.mound.adapters.belief_adapter import BeliefAdapter

        adapter = BeliefAdapter()

        # Should have reverse flow methods
        assert hasattr(adapter, "update_belief_thresholds_from_km")
        assert hasattr(adapter, "get_km_validated_priors")

    def test_insights_adapter_has_flip_methods(self):
        """Test InsightsAdapter has flip threshold methods."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Should have reverse flow methods
        assert hasattr(adapter, "update_flip_thresholds_from_km")
        assert hasattr(adapter, "get_agent_flip_baselines")


class TestPhase10EndToEnd:
    """End-to-end tests for Phase 10 bidirectional flow."""

    @pytest.mark.asyncio
    async def test_debate_to_km_to_elo_flow(self):
        """Test complete flow: Debate → KM → ELO adjustment."""
        from aragora.debate.km_outcome_bridge import KMOutcomeBridge
        from aragora.ranking.km_elo_bridge import KMEloBridge, KMEloBridgeConfig
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
        )

        # Setup mocks
        @dataclass
        class MockOutcome:
            debate_id: str = "debate_1"
            consensus_confidence: float = 0.9
            implementation_succeeded: bool = True

        mock_km = MagicMock()
        mock_km.get = AsyncMock(return_value={"id": "km_1", "confidence": 0.7})
        mock_km.update_confidence = AsyncMock(return_value=True)
        mock_km.get_graph_neighbors = AsyncMock(return_value=[])

        # Create bridges
        outcome_bridge = KMOutcomeBridge(knowledge_mound=mock_km)
        outcome_bridge.record_km_usage("debate_1", ["km_1", "km_2"])

        # Step 1: Validate knowledge from outcome
        validations = await outcome_bridge.validate_knowledge_from_outcome(MockOutcome())

        assert len(validations) >= 1
        assert validations[0].was_successful is True

        # Step 2: Coordinator would sync these validations
        coordinator = BidirectionalCoordinator()
        status = coordinator.get_status()

        assert status["total_adapters"] == 0  # No adapters registered yet

    @pytest.mark.asyncio
    async def test_km_pattern_detection_flow(self):
        """Test KM pattern detection and ELO recommendation flow."""
        from aragora.knowledge.mound.adapters.elo_adapter import (
            EloAdapter,
            KMEloPattern,
        )

        adapter = EloAdapter()

        # Simulate KM items for an agent
        km_items = [
            {"id": f"km_{i}", "metadata": {"outcome_success": True, "claim_validated": True}}
            for i in range(10)
        ]

        # Analyze patterns
        patterns = await adapter.analyze_km_patterns_for_agent(
            agent_name="claude",
            km_items=km_items,
            min_confidence=0.5,
        )

        # Should detect success_contributor pattern
        assert len(patterns) >= 1
        pattern_types = [p.pattern_type for p in patterns]
        assert "success_contributor" in pattern_types

        # Compute adjustment
        recommendation = adapter.compute_elo_adjustment(patterns)

        assert recommendation is not None
        assert recommendation.adjustment > 0  # Positive for success contributor
