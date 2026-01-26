"""
End-to-end integration tests for bidirectional Knowledge Mound integration.

Tests the complete flow from SubsystemCoordinator initialization through
adapter registration, forward sync, and reverse sync operations.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock


# =============================================================================
# Mock Adapters for E2E Testing
# =============================================================================


@dataclass
class MockContinuumAdapter:
    """Mock ContinuumAdapter for testing bidirectional sync."""

    forward_calls: List[Dict] = field(default_factory=list)
    reverse_calls: List[Dict] = field(default_factory=list)
    forward_result: int = 10
    reverse_result: bool = True

    def sync_to_km(self, workspace_id: str = "default") -> int:
        """Forward sync: Continuum → KM."""
        self.forward_calls.append({"workspace_id": workspace_id})
        return self.forward_result

    def update_continuum_from_km(
        self,
        memory_id: str,
        km_validation: Dict[str, Any],
    ) -> bool:
        """Reverse sync: KM → Continuum."""
        self.reverse_calls.append(
            {
                "memory_id": memory_id,
                "km_validation": km_validation,
            }
        )
        return self.reverse_result


@dataclass
class MockELOAdapter:
    """Mock ELOAdapter for testing bidirectional sync."""

    forward_calls: List[Dict] = field(default_factory=list)
    reverse_calls: List[Dict] = field(default_factory=list)
    forward_result: int = 5
    reverse_adjustment: float = 10.0

    def sync_to_km(self, workspace_id: str = "default") -> int:
        """Forward sync: ELO → KM."""
        self.forward_calls.append({"workspace_id": workspace_id})
        return self.forward_result

    def update_elo_from_km_patterns(
        self,
        agent_name: str,
        km_patterns: List[Dict],
    ) -> float:
        """Reverse sync: KM → ELO."""
        self.reverse_calls.append(
            {
                "agent_name": agent_name,
                "km_patterns": km_patterns,
            }
        )
        return self.reverse_adjustment


@dataclass
class MockBeliefAdapter:
    """Mock BeliefAdapter for testing bidirectional sync."""

    forward_calls: List[Dict] = field(default_factory=list)
    reverse_calls: List[Dict] = field(default_factory=list)
    forward_result: int = 3
    threshold_update: Dict = field(default_factory=dict)

    def sync_to_km(self, workspace_id: str = "default") -> int:
        """Forward sync: Belief → KM."""
        self.forward_calls.append({"workspace_id": workspace_id})
        return self.forward_result

    def update_belief_thresholds_from_km(
        self,
        workspace_id: str,
    ) -> Dict:
        """Reverse sync: KM → Belief."""
        self.reverse_calls.append({"workspace_id": workspace_id})
        return self.threshold_update


@dataclass
class MockInsightsAdapter:
    """Mock InsightsAdapter for testing bidirectional sync."""

    forward_calls: List[Dict] = field(default_factory=list)
    reverse_calls: List[Dict] = field(default_factory=list)
    forward_result: int = 7
    threshold_update: Dict = field(default_factory=dict)

    def sync_to_km(self, workspace_id: str = "default") -> int:
        """Forward sync: Insights → KM."""
        self.forward_calls.append({"workspace_id": workspace_id})
        return self.forward_result

    def update_flip_thresholds_from_km(
        self,
        workspace_id: str,
    ) -> Dict:
        """Reverse sync: KM → Insights."""
        self.reverse_calls.append({"workspace_id": workspace_id})
        return self.threshold_update


@dataclass
class MockCritiqueAdapter:
    """Mock CritiqueAdapter for testing bidirectional sync."""

    forward_calls: List[Dict] = field(default_factory=list)
    reverse_calls: List[Dict] = field(default_factory=list)
    forward_result: int = 4
    boost_result: bool = True

    def sync_to_km(self, workspace_id: str = "default") -> int:
        """Forward sync: Critique → KM."""
        self.forward_calls.append({"workspace_id": workspace_id})
        return self.forward_result

    def boost_pattern_from_km(
        self,
        pattern_id: str,
        km_validation: Dict,
    ) -> bool:
        """Reverse sync: KM → Critique."""
        self.reverse_calls.append(
            {
                "pattern_id": pattern_id,
                "km_validation": km_validation,
            }
        )
        return self.boost_result


@dataclass
class MockPulseAdapter:
    """Mock PulseAdapter for testing bidirectional sync."""

    forward_calls: List[Dict] = field(default_factory=list)
    reverse_calls: List[Dict] = field(default_factory=list)
    forward_result: int = 6
    sync_result: Dict = field(default_factory=dict)

    def sync_to_km(self, workspace_id: str = "default") -> int:
        """Forward sync: Pulse → KM."""
        self.forward_calls.append({"workspace_id": workspace_id})
        return self.forward_result

    def sync_validations_from_km(
        self,
        workspace_id: str,
        min_confidence: float = 0.7,
    ) -> Dict:
        """Reverse sync: KM → Pulse."""
        self.reverse_calls.append(
            {
                "workspace_id": workspace_id,
                "min_confidence": min_confidence,
            }
        )
        return self.sync_result


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    items: List[Dict] = field(default_factory=list)
    debate_end_calls: List[Dict] = field(default_factory=list)
    consensus_calls: List[Dict] = field(default_factory=list)

    def get_items_since(
        self,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get items from KM since a timestamp."""
        return self.items

    def on_debate_end(self, ctx: Dict, result: Dict) -> None:
        """Handle debate end hook."""
        self.debate_end_calls.append({"ctx": ctx, "result": result})

    def on_consensus_reached(
        self,
        ctx: Dict,
        consensus_text: str,
        confidence: float,
    ) -> None:
        """Handle consensus reached hook."""
        self.consensus_calls.append(
            {
                "ctx": ctx,
                "consensus_text": consensus_text,
                "confidence": confidence,
            }
        )

    def on_outcome_tracked(self, ctx: Dict, outcome: Dict) -> None:
        """Handle outcome tracked hook."""
        pass


# =============================================================================
# E2E Integration Tests
# =============================================================================


class TestSubsystemCoordinatorKMIntegration:
    """Test SubsystemCoordinator with KM bidirectional integration."""

    def test_coordinator_auto_init_with_km(self):
        """Test SubsystemCoordinator auto-initializes KM coordinator."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        km = MockKnowledgeMound()
        coordinator = SubsystemCoordinator(
            knowledge_mound=km,
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
        )

        assert coordinator.has_knowledge_mound
        assert coordinator.has_km_coordinator
        assert coordinator.has_km_bidirectional

    def test_coordinator_status_includes_km(self):
        """Test get_status() includes KM information."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        km = MockKnowledgeMound()
        coordinator = SubsystemCoordinator(
            knowledge_mound=km,
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
        )

        status = coordinator.get_status()

        assert "knowledge_mound" in status
        assert status["knowledge_mound"]["available"] is True
        assert status["knowledge_mound"]["coordinator_active"] is True
        assert status["knowledge_mound"]["bidirectional_enabled"] is True

    def test_coordinator_with_adapters(self):
        """Test SubsystemCoordinator with pre-configured adapters."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        km = MockKnowledgeMound()
        continuum_adapter = MockContinuumAdapter()
        elo_adapter = MockELOAdapter()

        coordinator = SubsystemCoordinator(
            knowledge_mound=km,
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
            km_continuum_adapter=continuum_adapter,
            km_elo_adapter=elo_adapter,
        )

        status = coordinator.get_status()
        assert status["knowledge_mound"]["adapters"]["continuum"] is True
        assert status["knowledge_mound"]["adapters"]["elo"] is True
        assert status["knowledge_mound"]["active_adapters_count"] == 2

    def test_coordinator_disabled_km(self):
        """Test SubsystemCoordinator with KM disabled."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator

        km = MockKnowledgeMound()
        coordinator = SubsystemCoordinator(
            knowledge_mound=km,
            enable_km_bidirectional=False,
            enable_km_coordinator=False,
        )

        assert coordinator.has_knowledge_mound
        assert not coordinator.has_km_coordinator
        assert not coordinator.has_km_bidirectional


class TestSubsystemConfigKMIntegration:
    """Test SubsystemConfig with KM bidirectional settings."""

    def test_config_includes_km_settings(self):
        """Test SubsystemConfig has KM settings."""
        from aragora.debate.subsystem_coordinator import SubsystemConfig

        config = SubsystemConfig(
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
            km_sync_interval_seconds=600,
            km_min_confidence_for_reverse=0.8,
            km_parallel_sync=False,
        )

        assert config.enable_km_bidirectional is True
        assert config.enable_km_coordinator is True
        assert config.km_sync_interval_seconds == 600
        assert config.km_min_confidence_for_reverse == 0.8
        assert config.km_parallel_sync is False

    def test_config_creates_coordinator_with_km(self):
        """Test SubsystemConfig.create_coordinator() with KM settings."""
        from aragora.debate.subsystem_coordinator import SubsystemConfig

        km = MockKnowledgeMound()
        config = SubsystemConfig(
            knowledge_mound=km,
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
        )

        coordinator = config.create_coordinator()

        assert coordinator.has_knowledge_mound
        assert coordinator.has_km_bidirectional


class TestArenaConfigKMIntegration:
    """Test ArenaConfig with KM bidirectional settings."""

    def test_arena_config_has_km_settings(self):
        """Test ArenaConfig has Phase 10 KM settings."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
            enable_km_continuum_sync=True,
            enable_km_elo_sync=True,
            enable_km_belief_sync=True,
            enable_km_flip_sync=True,
            enable_km_critique_sync=True,
            enable_km_pulse_sync=True,
            km_sync_interval_seconds=300,
            km_min_confidence_for_reverse=0.7,
            km_parallel_sync=True,
        )

        assert config.enable_km_bidirectional is True
        assert config.enable_km_coordinator is True
        assert config.enable_km_continuum_sync is True
        assert config.km_sync_interval_seconds == 300
        assert config.km_parallel_sync is True

    def test_arena_config_defaults(self):
        """Test ArenaConfig has sensible KM defaults."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()

        # Check defaults
        assert config.enable_km_bidirectional is True
        assert config.enable_km_coordinator is True
        assert config.km_sync_interval_seconds == 300
        assert config.km_min_confidence_for_reverse == 0.7
        assert config.km_parallel_sync is True


class TestBidirectionalCoordinatorE2E:
    """End-to-end tests for BidirectionalCoordinator."""

    @pytest.fixture
    def adapters(self):
        """Create mock adapters."""
        return {
            "continuum": MockContinuumAdapter(),
            "elo": MockELOAdapter(),
            "belief": MockBeliefAdapter(),
            "insights": MockInsightsAdapter(),
            "critique": MockCritiqueAdapter(),
            "pulse": MockPulseAdapter(),
        }

    @pytest.fixture
    def km(self):
        """Create mock Knowledge Mound."""
        return MockKnowledgeMound(
            items=[
                {"id": "km1", "type": "validation", "confidence": 0.8},
                {"id": "km2", "type": "pattern", "confidence": 0.9},
            ]
        )

    @pytest.mark.asyncio
    async def test_full_bidirectional_flow(self, adapters, km):
        """Test complete bidirectional sync flow."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
            CoordinatorConfig,
        )

        config = CoordinatorConfig(
            sync_interval_seconds=60,
            min_confidence_for_reverse=0.7,
            parallel_sync=False,  # Sequential for predictable testing
        )
        coordinator = BidirectionalCoordinator(
            config=config,
            knowledge_mound=km,
        )

        # Register adapters
        coordinator.register_adapter(
            name="continuum",
            adapter=adapters["continuum"],
            forward_method="sync_to_km",
            reverse_method="update_continuum_from_km",
            priority=1,
        )
        coordinator.register_adapter(
            name="elo",
            adapter=adapters["elo"],
            forward_method="sync_to_km",
            reverse_method="update_elo_from_km_patterns",
            priority=2,
        )

        # Run bidirectional sync (async)
        report = await coordinator.run_bidirectional_sync()

        # Verify forward syncs happened
        assert len(adapters["continuum"].forward_calls) == 1
        assert len(adapters["elo"].forward_calls) == 1

        # Verify report
        assert report.successful_forward >= 2
        assert report.total_errors == 0

    @pytest.mark.asyncio
    async def test_forward_only_flow(self, adapters, km):
        """Test forward-only sync with adapters missing reverse methods."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
            CoordinatorConfig,
        )

        coordinator = BidirectionalCoordinator(knowledge_mound=km)

        # Register adapter without reverse method
        coordinator.register_adapter(
            name="forward_only",
            adapter=adapters["continuum"],
            forward_method="sync_to_km",
            reverse_method=None,
            priority=1,
        )

        # Run forward sync (async)
        results = await coordinator.sync_all_to_km()

        # Results is a list of SyncResult
        assert len(results) == 1
        assert results[0].success is True
        assert len(adapters["continuum"].forward_calls) == 1

    def test_coordinator_status_reporting(self, adapters, km):
        """Test coordinator status reporting."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
        )

        coordinator = BidirectionalCoordinator(knowledge_mound=km)

        # Register multiple adapters
        for name, adapter in list(adapters.items())[:3]:
            coordinator.register_adapter(
                name=name,
                adapter=adapter,
                forward_method="sync_to_km",
                reverse_method=None,
                priority=1,
            )

        status = coordinator.get_status()

        assert status["total_adapters"] == 3
        assert status["enabled_adapters"] == 3
        assert "adapters" in status


class TestHookSystemKMIntegration:
    """Test hook system with KM integration."""

    @pytest.fixture
    def hook_manager(self):
        """Create mock hook manager."""

        class MockHookManager:
            def __init__(self):
                self.registered_hooks = {}
                self.callbacks = []

            def register(self, hook_type, callback, name=None, priority=None):
                key = (hook_type, name)
                self.registered_hooks[key] = {
                    "callback": callback,
                    "priority": priority,
                }
                self.callbacks.append((hook_type, name, callback))

                def unregister():
                    if key in self.registered_hooks:
                        del self.registered_hooks[key]

                return unregister

        return MockHookManager()

    def test_km_hooks_registered(self, hook_manager):
        """Test KM hooks are registered with hook manager."""
        from aragora.debate.hook_handlers import create_hook_handler_registry

        km = MockKnowledgeMound()
        registry = create_hook_handler_registry(
            hook_manager,
            knowledge_mound=km,
            auto_register=True,
        )

        assert registry.is_registered
        handler_names = [name for _, name, _ in hook_manager.callbacks]
        assert "km_debate_end" in handler_names
        assert "km_consensus" in handler_names

    def test_km_hooks_fire_correctly(self, hook_manager):
        """Test KM hooks fire and call KM methods."""
        from aragora.debate.hook_handlers import create_hook_handler_registry

        km = MockKnowledgeMound()
        registry = create_hook_handler_registry(
            hook_manager,
            knowledge_mound=km,
            auto_register=True,
        )

        # Find and call the debate end hook
        for hook_type, name, callback in hook_manager.callbacks:
            if name == "km_debate_end":
                callback(ctx={"debate_id": "test"}, result={"success": True})
                break

        assert len(km.debate_end_calls) == 1
        assert km.debate_end_calls[0]["ctx"]["debate_id"] == "test"


class TestKMBidirectionalE2EFlow:
    """Full end-to-end tests for KM bidirectional flow."""

    def test_complete_debate_flow_with_km(self):
        """Test complete flow: debate → KM sync → reverse sync."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
            CoordinatorConfig,
        )

        # Setup
        km = MockKnowledgeMound(
            items=[
                {"id": "km1", "type": "validation", "confidence": 0.85},
            ]
        )
        continuum_adapter = MockContinuumAdapter()
        elo_adapter = MockELOAdapter()

        # Create coordinator with KM
        coordinator = SubsystemCoordinator(
            knowledge_mound=km,
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
            km_continuum_adapter=continuum_adapter,
            km_elo_adapter=elo_adapter,
        )

        # Verify setup
        assert coordinator.has_km_bidirectional
        status = coordinator.get_status()
        assert status["knowledge_mound"]["active_adapters_count"] == 2

    def test_km_integration_with_arena_config(self):
        """Test KM integration configured via ArenaConfig."""
        from aragora.debate.arena_config import ArenaConfig

        # Create config with KM settings
        config = ArenaConfig(
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
            enable_km_continuum_sync=True,
            enable_km_elo_sync=True,
            km_sync_interval_seconds=120,
            km_min_confidence_for_reverse=0.75,
        )

        # Verify config values
        assert config.enable_km_bidirectional is True
        assert config.km_sync_interval_seconds == 120
        assert config.km_min_confidence_for_reverse == 0.75

    @pytest.mark.asyncio
    async def test_adapters_receive_correct_data(self):
        """Test adapters receive correct data during sync."""
        from aragora.knowledge.mound.bidirectional_coordinator import (
            BidirectionalCoordinator,
        )

        km = MockKnowledgeMound(
            items=[
                {"id": "km1", "memory_id": "m1", "confidence": 0.9},
            ]
        )
        continuum_adapter = MockContinuumAdapter()

        coordinator = BidirectionalCoordinator(knowledge_mound=km)
        coordinator.register_adapter(
            name="continuum",
            adapter=continuum_adapter,
            forward_method="sync_to_km",
            reverse_method="update_continuum_from_km",
            priority=1,
        )

        # Run forward sync (async)
        forward_results = await coordinator.sync_all_to_km()
        assert len(forward_results) == 1
        assert forward_results[0].success is True
        assert len(continuum_adapter.forward_calls) == 1

        # Verify workspace_id is passed
        assert continuum_adapter.forward_calls[0]["workspace_id"] == "default"


class TestKMConfigValidation:
    """Test KM configuration validation."""

    def test_config_bounds_validation(self):
        """Test configuration value bounds."""
        from aragora.debate.arena_config import ArenaConfig

        # Valid config
        config = ArenaConfig(
            km_sync_interval_seconds=300,
            km_min_confidence_for_reverse=0.7,
        )
        assert config.km_sync_interval_seconds == 300

    def test_subsystem_config_passthrough(self):
        """Test SubsystemConfig passes KM settings to coordinator."""
        from aragora.debate.subsystem_coordinator import SubsystemConfig

        km = MockKnowledgeMound()
        continuum_adapter = MockContinuumAdapter()

        config = SubsystemConfig(
            knowledge_mound=km,
            km_continuum_adapter=continuum_adapter,
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
            km_sync_interval_seconds=600,
        )

        coordinator = config.create_coordinator()

        # Verify settings were passed
        assert coordinator.knowledge_mound is km
        assert coordinator.km_continuum_adapter is continuum_adapter
        assert coordinator.km_sync_interval_seconds == 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
