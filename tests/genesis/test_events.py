"""
Tests for Genesis Events streaming hooks.

Tests cover:
- GenesisStreamEventType enum values
- create_genesis_hooks function
- create_logging_hooks function
- Event emission to emitters
- Ledger recording integration
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, call

from aragora.genesis.events import (
    GenesisStreamEventType,
    create_genesis_hooks,
    create_logging_hooks,
)
from aragora.genesis.genome import AgentGenome


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_emitter():
    """Create a mock event emitter."""
    emitter = MagicMock()
    emitter.emit = MagicMock()
    return emitter


@pytest.fixture
def mock_ledger():
    """Create a mock genesis ledger."""
    ledger = MagicMock()
    ledger.record_debate_start = MagicMock()
    ledger.record_debate_spawn = MagicMock()
    ledger.record_debate_merge = MagicMock()
    ledger.record_agent_birth = MagicMock()
    ledger.record_fitness_update = MagicMock()
    ledger.record_agent_death = MagicMock()
    return ledger


@pytest.fixture
def sample_genome():
    """Create a sample genome for testing."""
    return AgentGenome(
        genome_id="test-genome-001",
        name="test-agent",
        traits={"analytical": 0.8, "creative": 0.6},
        expertise={"security": 0.9, "architecture": 0.7, "testing": 0.5},
        model_preference="claude",
        parent_genomes=[],
        generation=0,
        fitness_score=0.5,
    )


# =============================================================================
# GenesisStreamEventType Tests
# =============================================================================


class TestGenesisStreamEventType:
    """Tests for the GenesisStreamEventType enum."""

    def test_fractal_events_exist(self):
        """Test fractal event types exist."""
        assert GenesisStreamEventType.FRACTAL_START.value == "fractal_start"
        assert GenesisStreamEventType.FRACTAL_SPAWN.value == "fractal_spawn"
        assert GenesisStreamEventType.FRACTAL_MERGE.value == "fractal_merge"
        assert GenesisStreamEventType.FRACTAL_COMPLETE.value == "fractal_complete"

    def test_agent_events_exist(self):
        """Test agent event types exist."""
        assert GenesisStreamEventType.AGENT_BIRTH.value == "agent_birth"
        assert GenesisStreamEventType.AGENT_EVOLUTION.value == "agent_evolution"
        assert GenesisStreamEventType.AGENT_DEATH.value == "agent_death"
        assert GenesisStreamEventType.LINEAGE_BRANCH.value == "lineage_branch"

    def test_population_events_exist(self):
        """Test population event types exist."""
        assert GenesisStreamEventType.POPULATION_UPDATE.value == "population_update"
        assert GenesisStreamEventType.GENERATION_ADVANCE.value == "generation_advance"

    def test_tension_events_exist(self):
        """Test tension event types exist."""
        assert GenesisStreamEventType.TENSION_DETECTED.value == "tension_detected"
        assert GenesisStreamEventType.TENSION_RESOLVED.value == "tension_resolved"

    def test_all_event_types_have_values(self):
        """Test all event types have string values."""
        for event_type in GenesisStreamEventType:
            assert isinstance(event_type.value, str)
            assert len(event_type.value) > 0


# =============================================================================
# create_genesis_hooks Tests
# =============================================================================


class TestCreateGenesisHooks:
    """Tests for the create_genesis_hooks function."""

    def test_returns_all_hooks(self, mock_emitter):
        """Test that all expected hooks are returned."""
        hooks = create_genesis_hooks(mock_emitter)

        expected_hooks = [
            "on_fractal_start",
            "on_fractal_spawn",
            "on_fractal_merge",
            "on_fractal_complete",
            "on_agent_birth",
            "on_agent_evolution",
            "on_agent_death",
            "on_lineage_branch",
            "on_population_update",
            "on_generation_advance",
            "on_tension_detected",
            "on_tension_resolved",
        ]

        for hook_name in expected_hooks:
            assert hook_name in hooks
            assert callable(hooks[hook_name])

    def test_on_fractal_start_emits_event(self, mock_emitter):
        """Test on_fractal_start emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_fractal_start"](
            debate_id="debate-123",
            task="Test task for debate",
            depth=0,
            parent_id=None,
        )

        mock_emitter.emit.assert_called_once()
        call_args = mock_emitter.emit.call_args
        assert call_args[0][0] == "fractal_start"
        event_data = call_args[0][1]
        assert event_data["debate_id"] == "debate-123"
        assert event_data["task"] == "Test task for debate"
        assert event_data["depth"] == 0
        assert event_data["parent_id"] is None
        assert "timestamp" in event_data

    def test_on_fractal_start_with_parent(self, mock_emitter):
        """Test on_fractal_start with parent ID."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_fractal_start"](
            debate_id="sub-debate",
            task="Sub-task",
            depth=1,
            parent_id="parent-debate",
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["parent_id"] == "parent-debate"
        assert event_data["depth"] == 1

    def test_on_fractal_spawn_emits_event(self, mock_emitter):
        """Test on_fractal_spawn emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_fractal_spawn"](
            debate_id="sub-1",
            parent_id="parent-1",
            tension="Disagreement about API design",
            depth=1,
        )

        mock_emitter.emit.assert_called_once()
        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "fractal_spawn"
        assert event_data["debate_id"] == "sub-1"
        assert event_data["parent_id"] == "parent-1"
        assert "Disagreement" in event_data["tension"]

    def test_on_fractal_merge_emits_event(self, mock_emitter):
        """Test on_fractal_merge emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_fractal_merge"](
            debate_id="sub-1",
            parent_id="parent-1",
            success=True,
            resolution="Agreed on REST design",
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "fractal_merge"
        assert event_data["success"] is True
        assert "REST design" in event_data["resolution"]

    def test_on_fractal_complete_emits_event(self, mock_emitter):
        """Test on_fractal_complete emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_fractal_complete"](
            debate_id="debate-1",
            depth=2,
            sub_debates=3,
            consensus_reached=True,
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "fractal_complete"
        assert event_data["depth"] == 2
        assert event_data["sub_debates"] == 3
        assert event_data["consensus_reached"] is True

    def test_on_agent_birth_emits_event(self, mock_emitter, sample_genome):
        """Test on_agent_birth emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_agent_birth"](
            genome=sample_genome,
            parents=["parent-1", "parent-2"],
            birth_type="crossover",
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "agent_birth"
        assert event_data["genome_id"] == "test-genome-001"
        assert event_data["name"] == "test-agent"
        assert event_data["parents"] == ["parent-1", "parent-2"]
        assert event_data["birth_type"] == "crossover"
        assert "analytical" in event_data["traits"]

    def test_on_agent_evolution_emits_event(self, mock_emitter):
        """Test on_agent_evolution emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_agent_evolution"](
            genome_id="genome-1",
            old_fitness=0.5,
            new_fitness=0.7,
            reason="Consensus contribution",
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "agent_evolution"
        assert event_data["genome_id"] == "genome-1"
        assert event_data["old_fitness"] == 0.5
        assert event_data["new_fitness"] == 0.7
        assert abs(event_data["change"] - 0.2) < 0.001  # Float comparison
        assert event_data["reason"] == "Consensus contribution"

    def test_on_agent_death_emits_event(self, mock_emitter):
        """Test on_agent_death emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_agent_death"](
            genome_id="genome-1",
            reason="Low fitness",
            final_fitness=0.2,
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "agent_death"
        assert event_data["genome_id"] == "genome-1"
        assert event_data["reason"] == "Low fitness"
        assert event_data["final_fitness"] == 0.2

    def test_on_lineage_branch_emits_event(self, mock_emitter):
        """Test on_lineage_branch emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_lineage_branch"](
            parent_genome_id="parent-1",
            child_genome_ids=["child-1", "child-2"],
            branch_type="crossover",
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "lineage_branch"
        assert event_data["parent_genome_id"] == "parent-1"
        assert event_data["child_genome_ids"] == ["child-1", "child-2"]

    def test_on_population_update_emits_event(self, mock_emitter):
        """Test on_population_update emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_population_update"](
            population_id="pop-1",
            size=10,
            generation=5,
            average_fitness=0.6,
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "population_update"
        assert event_data["population_id"] == "pop-1"
        assert event_data["size"] == 10
        assert event_data["generation"] == 5
        assert event_data["average_fitness"] == 0.6

    def test_on_generation_advance_emits_event(self, mock_emitter):
        """Test on_generation_advance emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_generation_advance"](
            population_id="pop-1",
            old_generation=4,
            new_generation=5,
            culled=2,
            born=3,
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "generation_advance"
        assert event_data["old_generation"] == 4
        assert event_data["new_generation"] == 5
        assert event_data["culled"] == 2
        assert event_data["born"] == 3

    def test_on_tension_detected_emits_event(self, mock_emitter):
        """Test on_tension_detected emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_tension_detected"](
            debate_id="debate-1",
            tension_id="tension-1",
            description="Disagreement about auth approach",
            severity=0.8,
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "tension_detected"
        assert event_data["tension_id"] == "tension-1"
        assert "auth approach" in event_data["description"]
        assert event_data["severity"] == 0.8

    def test_on_tension_resolved_emits_event(self, mock_emitter):
        """Test on_tension_resolved emits correct event."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_tension_resolved"](
            debate_id="debate-1",
            tension_id="tension-1",
            resolution="Use OAuth2",
            success=True,
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert event_data["type"] == "tension_resolved"
        assert event_data["success"] is True
        assert "OAuth2" in event_data["resolution"]


# =============================================================================
# Ledger Integration Tests
# =============================================================================


class TestGenesisHooksWithLedger:
    """Tests for genesis hooks with ledger integration."""

    def test_on_fractal_start_records_to_ledger(self, mock_emitter, mock_ledger):
        """Test on_fractal_start records to ledger."""
        hooks = create_genesis_hooks(mock_emitter, ledger=mock_ledger)

        hooks["on_fractal_start"](
            debate_id="debate-1",
            task="Test task",
            depth=0,
        )

        mock_ledger.record_debate_start.assert_called_once_with(
            debate_id="debate-1",
            task="Test task",
            agents=[],
            parent_debate_id=None,
        )

    def test_on_fractal_spawn_records_to_ledger(self, mock_emitter, mock_ledger):
        """Test on_fractal_spawn records to ledger."""
        hooks = create_genesis_hooks(mock_emitter, ledger=mock_ledger)

        hooks["on_fractal_spawn"](
            debate_id="sub-1",
            parent_id="parent-1",
            tension="API design conflict",
            depth=1,
        )

        mock_ledger.record_debate_spawn.assert_called_once_with(
            parent_id="parent-1",
            child_id="sub-1",
            trigger="unresolved_tension",
            tension_description="API design conflict",
        )

    def test_on_fractal_merge_records_to_ledger(self, mock_emitter, mock_ledger):
        """Test on_fractal_merge records to ledger."""
        hooks = create_genesis_hooks(mock_emitter, ledger=mock_ledger)

        hooks["on_fractal_merge"](
            debate_id="sub-1",
            parent_id="parent-1",
            success=True,
            resolution="Agreed on approach",
        )

        mock_ledger.record_debate_merge.assert_called_once_with(
            parent_id="parent-1",
            child_id="sub-1",
            success=True,
            resolution="Agreed on approach",
        )

    def test_on_agent_birth_records_to_ledger(self, mock_emitter, mock_ledger, sample_genome):
        """Test on_agent_birth records to ledger."""
        hooks = create_genesis_hooks(mock_emitter, ledger=mock_ledger)

        hooks["on_agent_birth"](
            genome=sample_genome,
            parents=["p1", "p2"],
            birth_type="crossover",
        )

        mock_ledger.record_agent_birth.assert_called_once_with(
            genome=sample_genome,
            parents=["p1", "p2"],
            birth_type="crossover",
        )

    def test_on_agent_evolution_records_to_ledger(self, mock_emitter, mock_ledger):
        """Test on_agent_evolution records to ledger."""
        hooks = create_genesis_hooks(mock_emitter, ledger=mock_ledger)

        hooks["on_agent_evolution"](
            genome_id="g1",
            old_fitness=0.5,
            new_fitness=0.7,
            reason="Debate win",
        )

        mock_ledger.record_fitness_update.assert_called_once_with(
            genome_id="g1",
            old_fitness=0.5,
            new_fitness=0.7,
            reason="Debate win",
        )

    def test_on_agent_death_records_to_ledger(self, mock_emitter, mock_ledger):
        """Test on_agent_death records to ledger."""
        hooks = create_genesis_hooks(mock_emitter, ledger=mock_ledger)

        hooks["on_agent_death"](
            genome_id="g1",
            reason="Culled",
            final_fitness=0.2,
        )

        mock_ledger.record_agent_death.assert_called_once_with(
            genome_id="g1",
            reason="Culled",
            final_fitness=0.2,
        )


# =============================================================================
# Emit Sync Tests
# =============================================================================


class TestGenesisHooksEmitSync:
    """Tests for hooks using emit_sync instead of emit."""

    def test_uses_emit_sync_if_available(self):
        """Test hooks use emit_sync if emit is not available."""
        emitter = MagicMock(spec=["emit_sync"])
        emitter.emit_sync = MagicMock()

        hooks = create_genesis_hooks(emitter)
        hooks["on_fractal_start"](debate_id="d1", task="t", depth=0)

        emitter.emit_sync.assert_called_once()


# =============================================================================
# create_logging_hooks Tests
# =============================================================================


class TestCreateLoggingHooks:
    """Tests for the create_logging_hooks function."""

    def test_returns_hooks(self):
        """Test that logging hooks are returned."""
        hooks = create_logging_hooks()

        expected_hooks = [
            "on_fractal_start",
            "on_fractal_spawn",
            "on_fractal_merge",
            "on_fractal_complete",
            "on_agent_birth",
            "on_agent_evolution",
            "on_agent_death",
        ]

        for hook_name in expected_hooks:
            assert hook_name in hooks
            assert callable(hooks[hook_name])

    def test_on_fractal_start_logs(self):
        """Test on_fractal_start logs correctly."""
        log_calls = []
        hooks = create_logging_hooks(log_func=log_calls.append)

        hooks["on_fractal_start"](debate_id="d1", task="test", depth=0, parent_id=None)

        assert len(log_calls) == 1
        assert "FRACTAL_START" in log_calls[0]
        assert "d1" in log_calls[0]

    def test_on_fractal_spawn_logs(self):
        """Test on_fractal_spawn logs correctly."""
        log_calls = []
        hooks = create_logging_hooks(log_func=log_calls.append)

        hooks["on_fractal_spawn"](
            debate_id="s1", parent_id="p1", tension="test tension text", depth=1
        )

        assert len(log_calls) == 1
        assert "FRACTAL_SPAWN" in log_calls[0]

    def test_on_fractal_merge_logs(self):
        """Test on_fractal_merge logs correctly."""
        log_calls = []
        hooks = create_logging_hooks(log_func=log_calls.append)

        hooks["on_fractal_merge"](debate_id="s1", parent_id="p1", success=True)

        assert len(log_calls) == 1
        assert "FRACTAL_MERGE" in log_calls[0]
        assert "success=True" in log_calls[0]

    def test_on_fractal_complete_logs(self):
        """Test on_fractal_complete logs correctly."""
        log_calls = []
        hooks = create_logging_hooks(log_func=log_calls.append)

        hooks["on_fractal_complete"](debate_id="d1", depth=2, sub_debates=3, consensus_reached=True)

        assert len(log_calls) == 1
        assert "FRACTAL_COMPLETE" in log_calls[0]
        assert "consensus=True" in log_calls[0]

    def test_on_agent_birth_logs(self, sample_genome):
        """Test on_agent_birth logs correctly."""
        log_calls = []
        hooks = create_logging_hooks(log_func=log_calls.append)

        hooks["on_agent_birth"](genome=sample_genome, parents=[], birth_type="mutation")

        assert len(log_calls) == 1
        assert "AGENT_BIRTH" in log_calls[0]
        assert sample_genome.name in log_calls[0]

    def test_on_agent_evolution_logs(self):
        """Test on_agent_evolution logs correctly."""
        log_calls = []
        hooks = create_logging_hooks(log_func=log_calls.append)

        hooks["on_agent_evolution"](genome_id="g1", old_fitness=0.5, new_fitness=0.7, reason="win")

        assert len(log_calls) == 1
        assert "AGENT_EVOLUTION" in log_calls[0]
        assert "+0.20" in log_calls[0]

    def test_on_agent_death_logs(self):
        """Test on_agent_death logs correctly."""
        log_calls = []
        hooks = create_logging_hooks(log_func=log_calls.append)

        hooks["on_agent_death"](genome_id="g1", reason="low fitness", final_fitness=0.1)

        assert len(log_calls) == 1
        assert "AGENT_DEATH" in log_calls[0]
        assert "low fitness" in log_calls[0]

    def test_default_log_func_is_print(self):
        """Test default log function is print."""
        hooks = create_logging_hooks()
        # Just verify it doesn't error
        # In a real test we'd capture stdout
        assert callable(hooks["on_fractal_start"])


# =============================================================================
# Edge Cases
# =============================================================================


class TestGenesisHooksEdgeCases:
    """Tests for edge cases in genesis hooks."""

    def test_long_task_truncated(self, mock_emitter):
        """Test long task descriptions are truncated."""
        hooks = create_genesis_hooks(mock_emitter)
        long_task = "A" * 500

        hooks["on_fractal_start"](debate_id="d1", task=long_task, depth=0)

        event_data = mock_emitter.emit.call_args[0][1]
        assert len(event_data["task"]) <= 200

    def test_long_tension_truncated(self, mock_emitter):
        """Test long tensions are truncated."""
        hooks = create_genesis_hooks(mock_emitter)
        long_tension = "B" * 500

        hooks["on_fractal_spawn"](debate_id="s1", parent_id="p1", tension=long_tension, depth=1)

        event_data = mock_emitter.emit.call_args[0][1]
        assert len(event_data["tension"]) <= 200

    def test_long_description_truncated(self, mock_emitter):
        """Test long descriptions are truncated."""
        hooks = create_genesis_hooks(mock_emitter)
        long_desc = "C" * 500

        hooks["on_tension_detected"](
            debate_id="d1", tension_id="t1", description=long_desc, severity=0.5
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert len(event_data["description"]) <= 200

    def test_long_resolution_truncated(self, mock_emitter):
        """Test long resolutions are truncated."""
        hooks = create_genesis_hooks(mock_emitter)
        long_res = "D" * 500

        hooks["on_tension_resolved"](
            debate_id="d1", tension_id="t1", resolution=long_res, success=True
        )

        event_data = mock_emitter.emit.call_args[0][1]
        assert len(event_data["resolution"]) <= 200

    def test_timestamp_format(self, mock_emitter):
        """Test timestamp is in ISO format."""
        hooks = create_genesis_hooks(mock_emitter)

        hooks["on_fractal_start"](debate_id="d1", task="test", depth=0)

        event_data = mock_emitter.emit.call_args[0][1]
        # Should be parseable as ISO format
        timestamp = event_data["timestamp"]
        datetime.fromisoformat(timestamp)  # Should not raise

    def test_hooks_without_ledger(self, mock_emitter):
        """Test hooks work without ledger."""
        hooks = create_genesis_hooks(mock_emitter, ledger=None)

        # Should not raise
        hooks["on_fractal_start"](debate_id="d1", task="test", depth=0)
        hooks["on_agent_birth"](
            genome=AgentGenome(genome_id="g1", name="test"),
            parents=[],
            birth_type="spawn",
        )
