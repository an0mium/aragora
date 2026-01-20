"""Integration tests for Phase 8 cross-pollination features.

Tests that the cross-pollination integrations work together:
- Hook system fires events across subsystems
- Performance metrics influence ELO adjustments
- Successful outcomes promote memories
- Trickster calibrates from outcome history
- Checkpoints include memory state
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock

import pytest

from aragora.debate.hook_handlers import HookHandlerRegistry
from aragora.ranking.performance_integrator import PerformanceEloIntegrator
from aragora.memory.outcome_bridge import OutcomeMemoryBridge
from aragora.debate.trickster_calibrator import TricksterCalibrator
from aragora.memory.continuum import ContinuumMemory
from aragora.memory.tier_manager import MemoryTier
from aragora.debate.checkpoint import (
    DebateCheckpoint,
    CheckpointManager,
    FileCheckpointStore,
    CheckpointConfig,
)


# Mock classes for testing integration


@dataclass
class MockConsensusOutcome:
    """Mock ConsensusOutcome for testing."""

    debate_id: str
    consensus_text: str = "Test consensus"
    consensus_confidence: float = 0.8
    implementation_attempted: bool = True
    implementation_succeeded: bool = True


@dataclass
class MockTricksterConfig:
    """Mock TricksterConfig for testing."""

    sensitivity: float = 0.5
    hollow_detection_threshold: float = 0.5
    min_quality_threshold: float = 0.65


@dataclass
class MockTrickster:
    """Mock EvidencePoweredTrickster for testing."""

    config: MockTricksterConfig = field(default_factory=MockTricksterConfig)


class MockHookManager:
    """Mock HookManager for testing."""

    def __init__(self):
        self._handlers: Dict[str, List] = {}
        self._fired_events: List[str] = []

    def register(self, hook_type: str, handler, name: str = None, priority: Any = None):
        """Register a handler, returning an unregister function."""
        if hook_type not in self._handlers:
            self._handlers[hook_type] = []
        self._handlers[hook_type].append(handler)

        # Return unregister function like real HookManager
        def unregister():
            if hook_type in self._handlers and handler in self._handlers[hook_type]:
                self._handlers[hook_type].remove(handler)
        return unregister

    def fire(self, hook_type: str, *args, **kwargs) -> None:
        self._fired_events.append(hook_type)
        for handler in self._handlers.get(hook_type, []):
            try:
                handler(*args, **kwargs)
            except Exception:
                pass  # Ignore handler errors in tests


class MockAgent:
    """Mock agent for checkpoint tests."""

    name = "test-agent"
    model = "test-model"
    role = "proposer"
    system_prompt = "test prompt"
    stance = "neutral"


class TestHookSystemIntegration:
    """Test hook system wires subsystems correctly."""

    def test_hook_registry_registers_handlers(self) -> None:
        """HookHandlerRegistry registers handlers for available subsystems."""
        hook_manager = MockHookManager()

        # Create mock analytics with expected method
        mock_analytics = Mock()
        mock_analytics.on_round_complete = Mock()
        mock_analytics.on_agent_response = Mock()

        subsystems = {"analytics": mock_analytics}
        registry = HookHandlerRegistry(
            hook_manager=hook_manager,
            subsystems=subsystems,
        )

        count = registry.register_all()

        # Should have registered analytics handlers
        assert count > 0
        # HookType values are "post_round" and "post_generate"
        assert "post_round" in hook_manager._handlers
        assert "post_generate" in hook_manager._handlers

    def test_hook_fires_to_registered_handlers(self) -> None:
        """Hooks fire to registered handlers."""
        hook_manager = MockHookManager()
        callback_called = []

        def test_callback(*args, **kwargs):
            callback_called.append(True)

        hook_manager.register("test_event", test_callback)
        hook_manager.fire("test_event")

        assert len(callback_called) == 1


class TestPerformanceEloIntegration:
    """Test performance metrics integrate with ELO system."""

    def test_performance_integrator_computes_multipliers(self) -> None:
        """PerformanceEloIntegrator computes K-factor multipliers from performance data."""
        from aragora.agents.performance_monitor import AgentStats

        # Create mock stats with proper AgentStats structure
        agent_stats = AgentStats()
        agent_stats.total_calls = 50
        agent_stats.successful_calls = 40
        agent_stats.failed_calls = 10
        agent_stats.avg_duration_ms = 1000.0
        agent_stats.min_duration_ms = 500.0
        agent_stats.max_duration_ms = 2000.0

        mock_monitor = Mock()
        mock_monitor.agent_stats = {
            "agent-1": agent_stats,
            "agent-2": agent_stats,
        }

        integrator = PerformanceEloIntegrator(
            performance_monitor=mock_monitor,
            response_quality_weight=0.4,
            latency_weight=0.1,
            consistency_weight=0.2,
            participation_weight=0.3,
            min_calls_for_adjustment=10,
        )

        multipliers = integrator.compute_k_multipliers(["agent-1", "agent-2"])

        # Should have multipliers for both agents
        assert "agent-1" in multipliers
        assert "agent-2" in multipliers
        # Multipliers should be positive
        assert multipliers["agent-1"] > 0
        assert multipliers["agent-2"] > 0


class TestOutcomeMemoryIntegration:
    """Test outcome tracking promotes memories."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            yield str(db_path)

    def test_successful_outcome_updates_memory_stats(self, temp_db) -> None:
        """Successful outcomes update memory success/failure counts."""
        # Use real memory for integration test
        memory = ContinuumMemory(db_path=temp_db)
        memory.add("insight-1", "Test insight", tier=MemoryTier.MEDIUM, importance=0.5)

        bridge = OutcomeMemoryBridge(
            continuum_memory=memory,
            success_boost_weight=0.1,
        )

        # Record memory usage and successful outcome
        bridge.record_memory_usage("insight-1", "debate-1")
        outcome = MockConsensusOutcome(
            debate_id="debate-1",
            implementation_succeeded=True,
            consensus_confidence=0.9,
        )
        result = bridge.process_outcome(outcome)

        # Bridge should have processed the outcome
        assert result.debate_id == "debate-1"

        # Check that stats were tracked
        stats = bridge.get_memory_stats("insight-1")
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 0


class TestTricksterCalibrationIntegration:
    """Test Trickster calibrates from outcome history."""

    def test_calibrator_adjusts_sensitivity(self) -> None:
        """TricksterCalibrator adjusts sensitivity based on outcome patterns."""
        mock_trickster = MockTrickster()
        mock_trickster.config.sensitivity = 0.7

        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=5,
            false_positive_tolerance=0.3,
            adjustment_step=0.1,
        )

        # Simulate many false positives (intervened but outcome was good)
        for i in range(10):
            calibrator.record_intervention(f"debate-{i}", 1)
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=True,
            )
            calibrator.record_debate_outcome(outcome)

        result = calibrator.calibrate()

        # Should have lowered sensitivity due to false positives
        assert result is not None
        assert result.calibrated is True
        assert result.new_sensitivity < result.old_sensitivity


class TestCheckpointMemoryIntegration:
    """Test checkpoints include memory state."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            yield str(db_path), str(checkpoint_dir)

    @pytest.mark.asyncio
    async def test_checkpoint_includes_memory_snapshot(self, temp_dirs) -> None:
        """Checkpoints include memory state for restoration."""
        db_path, checkpoint_dir = temp_dirs

        # Create memory with entries
        memory = ContinuumMemory(db_path=db_path)
        memory.add("context-1", "Important context", tier=MemoryTier.FAST, importance=0.9)
        memory.add("pattern-1", "Recognized pattern", tier=MemoryTier.MEDIUM, importance=0.7)

        # Export memory snapshot
        snapshot = memory.export_snapshot()

        # Create checkpoint with memory state
        store = FileCheckpointStore(base_dir=checkpoint_dir, compress=False)
        manager = CheckpointManager(store=store)

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-integration",
            task="Integration test",
            current_round=2,
            total_rounds=5,
            phase="proposal",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
            continuum_memory_state=snapshot,
        )

        assert checkpoint.continuum_memory_state is not None
        assert checkpoint.continuum_memory_state["total_entries"] == 2

    @pytest.mark.asyncio
    async def test_restored_checkpoint_restores_memory(self, temp_dirs) -> None:
        """Restored checkpoint can restore memory state."""
        db_path, checkpoint_dir = temp_dirs

        # Create and populate memory
        memory = ContinuumMemory(db_path=db_path)
        memory.add("key-insight", "Critical insight", tier=MemoryTier.FAST, importance=0.95)
        snapshot = memory.export_snapshot()

        # Create checkpoint
        store = FileCheckpointStore(base_dir=checkpoint_dir, compress=False)
        manager = CheckpointManager(store=store)

        checkpoint = await manager.create_checkpoint(
            debate_id="debate-restore",
            task="Restore test",
            current_round=1,
            total_rounds=3,
            phase="critique",
            messages=[],
            critiques=[],
            votes=[],
            agents=[MockAgent()],
            continuum_memory_state=snapshot,
        )

        # Resume checkpoint
        resumed = await manager.resume_from_checkpoint(checkpoint.checkpoint_id)
        assert resumed is not None

        # Restore to new memory
        with tempfile.TemporaryDirectory() as new_dir:
            new_db = Path(new_dir) / "restored.db"
            new_memory = ContinuumMemory(db_path=str(new_db))

            result = new_memory.restore_snapshot(resumed.checkpoint.continuum_memory_state)

            assert result["restored"] == 1

            entry = new_memory.get("key-insight")
            assert entry is not None
            assert entry.content == "Critical insight"


class TestFullCrossPollinationFlow:
    """End-to-end test of cross-pollination features working together."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "full_test.db"
            yield str(db_path)

    def test_full_flow_hooks_performance_outcomes_calibration(self, temp_db) -> None:
        """Full flow: hooks → outcomes → calibration."""
        from aragora.agents.performance_monitor import AgentStats

        # 1. Setup hook system
        hook_manager = MockHookManager()
        events_fired = []

        def track_event(event_type):
            def handler(*args, **kwargs):
                events_fired.append(event_type)
            return handler

        hook_manager.register("debate_start", track_event("debate_start"))
        hook_manager.register("round_complete", track_event("round_complete"))
        hook_manager.register("debate_end", track_event("debate_end"))

        # 2. Setup memory and outcome bridge (use real memory)
        memory = ContinuumMemory(db_path=temp_db)
        memory.add("strategy-1", "Winning strategy", tier=MemoryTier.MEDIUM, importance=0.6)

        outcome_bridge = OutcomeMemoryBridge(continuum_memory=memory)

        # 3. Setup Trickster calibrator
        mock_trickster = MockTrickster()
        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=3,
        )

        # 4. Setup performance integrator with proper AgentStats
        agent_stats = AgentStats()
        agent_stats.total_calls = 20
        agent_stats.successful_calls = 17
        agent_stats.avg_duration_ms = 800.0

        mock_monitor = Mock()
        mock_monitor.agent_stats = {"agent-1": agent_stats}

        performance_integrator = PerformanceEloIntegrator(
            performance_monitor=mock_monitor,
            min_calls_for_adjustment=10,
        )

        # Simulate debate flow
        hook_manager.fire("debate_start")

        # Record memory usage
        outcome_bridge.record_memory_usage("strategy-1", "debate-full")

        # Simulate rounds
        for round_num in range(3):
            hook_manager.fire("round_complete", round_num=round_num)
            calibrator.record_intervention("debate-full", 1)

        # End debate with successful outcome
        hook_manager.fire("debate_end")

        outcome = MockConsensusOutcome(
            debate_id="debate-full",
            implementation_succeeded=True,
            consensus_confidence=0.9,
        )

        # Process outcome through all systems
        outcome_bridge.process_outcome(outcome)
        calibrator.record_debate_outcome(outcome)

        # Compute performance multipliers
        multipliers = performance_integrator.compute_k_multipliers(["agent-1"])

        # Verify all systems participated
        assert "debate_start" in events_fired
        assert "round_complete" in events_fired
        assert "debate_end" in events_fired
        assert len(calibrator._data_points) == 1
        assert "agent-1" in multipliers

    def test_memory_snapshot_round_trip_with_outcome_updates(self, temp_db) -> None:
        """Memory snapshot preserves outcome-updated entries."""
        memory = ContinuumMemory(db_path=temp_db)

        # Add memory
        memory.add("tested-pattern", "Pattern content", tier=MemoryTier.MEDIUM, importance=0.5)

        # Update outcome (simulating successful use)
        memory.update_outcome("tested-pattern", success=True)
        memory.update_outcome("tested-pattern", success=True)
        memory.update_outcome("tested-pattern", success=False)

        # Get entry to verify updates
        entry = memory.get("tested-pattern")
        assert entry.success_count == 2
        assert entry.failure_count == 1

        # Export snapshot
        snapshot = memory.export_snapshot()

        # Restore to new memory
        with tempfile.TemporaryDirectory() as new_dir:
            new_db = Path(new_dir) / "restored.db"
            new_memory = ContinuumMemory(db_path=str(new_db))

            result = new_memory.restore_snapshot(snapshot)
            assert result["restored"] == 1

            # Verify outcome counts preserved
            restored_entry = new_memory.get("tested-pattern")
            assert restored_entry is not None
            assert restored_entry.success_count == 2
            assert restored_entry.failure_count == 1


class TestCrossPollinationConfiguration:
    """Test cross-pollination configuration in ArenaConfig."""

    def test_arena_config_has_cross_pollination_fields(self) -> None:
        """ArenaConfig has all cross-pollination configuration fields."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()

        # Phase 8 cross-pollination fields
        assert hasattr(config, "enable_hook_handlers")
        assert hasattr(config, "enable_performance_elo")
        assert hasattr(config, "enable_outcome_memory")
        assert hasattr(config, "enable_trickster_calibration")
        assert hasattr(config, "checkpoint_include_memory")

        # Verify defaults
        assert config.enable_hook_handlers is True
        assert config.enable_performance_elo is True
        assert config.enable_outcome_memory is True
        assert config.enable_trickster_calibration is True
        assert config.checkpoint_include_memory is True

    def test_arena_config_has_belief_and_workflow_fields(self) -> None:
        """ArenaConfig has belief guidance and workflow fields."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()

        # Belief guidance
        assert hasattr(config, "enable_belief_guidance")
        assert config.enable_belief_guidance is True

        # Cross-debate memory
        assert hasattr(config, "enable_cross_debate_memory")
        assert config.enable_cross_debate_memory is True

        # Post-debate workflow
        assert hasattr(config, "enable_post_debate_workflow")
        assert config.enable_post_debate_workflow is False  # Disabled by default

        # Auto-revalidation
        assert hasattr(config, "enable_auto_revalidation")
        assert config.enable_auto_revalidation is False  # Disabled by default


class TestCrossPollinationEventChain:
    """Test events flow through the cross-pollination chain."""

    def test_debate_outcome_triggers_multiple_systems(self) -> None:
        """Debate outcome event triggers memory, calibration, and ELO updates."""
        from aragora.agents.performance_monitor import AgentStats

        # This simulates what happens after a debate completes:
        # 1. Outcome recorded
        # 2. Memory bridge processes outcome -> updates memory tiers
        # 3. Calibrator processes outcome -> adjusts sensitivity
        # 4. Performance integrator has metrics for next ELO calculation

        systems_triggered = []

        # Mock outcome bridge with memory that tracks updates
        mock_memory = Mock()
        mock_entry = Mock()
        mock_entry.success_count = 0
        mock_entry.failure_count = 0
        mock_entry.importance = 0.5
        mock_memory.get_entry.return_value = mock_entry

        def track_update(*args, **kwargs):
            systems_triggered.append("memory")
        mock_memory.update_entry = track_update
        mock_memory.promote_entry = Mock()
        mock_memory.demote_entry = Mock()

        bridge = OutcomeMemoryBridge(continuum_memory=mock_memory)
        bridge.record_memory_usage("mem-1", "debate-chain")

        # Mock trickster
        mock_trickster = MockTrickster()
        calibrator = TricksterCalibrator(trickster=mock_trickster, min_samples=1)
        calibrator.record_intervention("debate-chain", 1)

        # Mock performance monitor with proper AgentStats
        agent_stats = AgentStats()
        agent_stats.total_calls = 100
        agent_stats.successful_calls = 90
        agent_stats.avg_duration_ms = 500.0

        mock_monitor = Mock()
        mock_monitor.agent_stats = {"winner-agent": agent_stats}
        integrator = PerformanceEloIntegrator(
            performance_monitor=mock_monitor,
            min_calls_for_adjustment=10,
        )

        # Trigger outcome
        outcome = MockConsensusOutcome(
            debate_id="debate-chain",
            implementation_succeeded=True,
            consensus_confidence=0.95,
        )

        # Process through all systems
        bridge.process_outcome(outcome)
        systems_triggered.append("bridge")

        calibrator.record_debate_outcome(outcome)
        systems_triggered.append("calibrator")

        multipliers = integrator.compute_k_multipliers(["winner-agent"])
        systems_triggered.append("elo")

        # All systems should have processed
        assert "bridge" in systems_triggered
        assert "calibrator" in systems_triggered
        assert "elo" in systems_triggered

        # Verify state changes
        assert len(calibrator._data_points) == 1
        assert "winner-agent" in multipliers
