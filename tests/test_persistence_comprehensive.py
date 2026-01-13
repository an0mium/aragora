"""
Comprehensive tests for the persistence layer.

Tests database configuration, data models, and repository patterns.
"""

import os
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from aragora.persistence.db_config import (
    DatabaseType,
    DatabaseMode,
    get_db_mode,
    get_nomic_dir,
    get_db_path,
    get_db_path_str,
    get_elo_db_path,
    get_memory_db_path,
    get_positions_db_path,
    get_personas_db_path,
    get_insights_db_path,
    get_genesis_db_path,
    LEGACY_DB_NAMES,
    CONSOLIDATED_DB_MAPPING,
)
from aragora.persistence.models import (
    NomicCycle,
    DebateArtifact,
    StreamEvent,
    AgentMetrics,
    NomicRollback,
    CycleEvolution,
    CycleFileChange,
)


class TestDatabaseMode:
    """Tests for database mode configuration."""

    def test_consolidated_mode_default(self):
        """Default mode should be consolidated when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear ARAGORA_DB_MODE if set
            os.environ.pop("ARAGORA_DB_MODE", None)
            mode = get_db_mode()
            assert mode == DatabaseMode.CONSOLIDATED

    def test_legacy_mode_explicit(self):
        """Legacy mode when explicitly set."""
        with patch.dict(os.environ, {"ARAGORA_DB_MODE": "legacy"}):
            mode = get_db_mode()
            assert mode == DatabaseMode.LEGACY

    def test_consolidated_mode(self):
        """Consolidated mode when set."""
        with patch.dict(os.environ, {"ARAGORA_DB_MODE": "consolidated"}):
            mode = get_db_mode()
            assert mode == DatabaseMode.CONSOLIDATED

    def test_invalid_mode_falls_back_to_consolidated(self):
        """Invalid mode values fall back to consolidated (the default)."""
        with patch.dict(os.environ, {"ARAGORA_DB_MODE": "invalid_mode"}):
            mode = get_db_mode()
            assert mode == DatabaseMode.CONSOLIDATED

    def test_case_insensitive_mode(self):
        """Mode parsing should be case-insensitive."""
        with patch.dict(os.environ, {"ARAGORA_DB_MODE": "CONSOLIDATED"}):
            mode = get_db_mode()
            assert mode == DatabaseMode.CONSOLIDATED


class TestNomicDir:
    """Tests for nomic directory configuration."""

    def test_default_nomic_dir(self):
        """Default nomic dir is .nomic."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARAGORA_DATA_DIR", None)
            os.environ.pop("ARAGORA_NOMIC_DIR", None)
            nomic_dir = get_nomic_dir()
            assert nomic_dir == Path(".nomic")

    def test_custom_nomic_dir(self):
        """Custom nomic dir from environment."""
        with patch.dict(os.environ, {"ARAGORA_DATA_DIR": "/custom/path"}):
            nomic_dir = get_nomic_dir()
            assert nomic_dir == Path("/custom/path")

    def test_custom_nomic_dir_legacy_alias(self):
        """Legacy nomic dir alias still works."""
        with patch.dict(os.environ, {"ARAGORA_NOMIC_DIR": "/legacy/path"}):
            nomic_dir = get_nomic_dir()
            assert nomic_dir == Path("/legacy/path")


class TestDbPath:
    """Tests for database path resolution."""

    def test_legacy_mode_elo_path(self):
        """ELO database path in legacy mode."""
        nomic_dir = Path("/test")
        path = get_db_path(DatabaseType.ELO, nomic_dir, DatabaseMode.LEGACY)
        assert path == Path("/test/agent_elo.db")

    def test_consolidated_mode_elo_path(self):
        """ELO database path in consolidated mode."""
        nomic_dir = Path("/test")
        path = get_db_path(DatabaseType.ELO, nomic_dir, DatabaseMode.CONSOLIDATED)
        assert path == Path("/test/analytics.db")

    def test_legacy_mode_memory_path(self):
        """Memory database path in legacy mode."""
        nomic_dir = Path("/test")
        path = get_db_path(DatabaseType.CONTINUUM_MEMORY, nomic_dir, DatabaseMode.LEGACY)
        assert path == Path("/test/continuum.db")

    def test_consolidated_mode_memory_path(self):
        """Memory database path in consolidated mode."""
        nomic_dir = Path("/test")
        path = get_db_path(DatabaseType.CONTINUUM_MEMORY, nomic_dir, DatabaseMode.CONSOLIDATED)
        assert path == Path("/test/memory.db")

    def test_all_database_types_have_legacy_mapping(self):
        """All database types should have legacy file mapping."""
        for db_type in DatabaseType:
            assert db_type in LEGACY_DB_NAMES, f"Missing legacy mapping for {db_type}"

    def test_all_database_types_have_consolidated_mapping(self):
        """All database types should have consolidated file mapping."""
        for db_type in DatabaseType:
            assert db_type in CONSOLIDATED_DB_MAPPING, f"Missing consolidated mapping for {db_type}"

    def test_consolidated_uses_four_databases(self):
        """Consolidated mode should use only 4 database files."""
        consolidated_dbs = set(CONSOLIDATED_DB_MAPPING.values())
        assert len(consolidated_dbs) == 4
        assert consolidated_dbs == {"core.db", "memory.db", "analytics.db", "agents.db"}

    def test_db_path_str_returns_string(self):
        """get_db_path_str should return string."""
        path = get_db_path_str(DatabaseType.ELO, Path("/test"), DatabaseMode.LEGACY)
        assert isinstance(path, str)
        assert path == "/test/agent_elo.db"


class TestConvenienceFunctions:
    """Tests for convenience path functions."""

    def test_get_elo_db_path(self):
        """ELO convenience function (consolidated mode default)."""
        path = get_elo_db_path(Path("/test"))
        # In consolidated mode (default), ELO maps to analytics.db
        assert path.name == "analytics.db"

    def test_get_memory_db_path(self):
        """Memory convenience function (consolidated mode default)."""
        path = get_memory_db_path(Path("/test"))
        # In consolidated mode (default), continuum memory maps to memory.db
        assert path.name == "memory.db"

    def test_get_positions_db_path(self):
        """Positions convenience function (consolidated mode default)."""
        path = get_positions_db_path(Path("/test"))
        # In consolidated mode (default), positions maps to core.db
        assert path.name == "core.db"

    def test_get_personas_db_path(self):
        """Personas convenience function (consolidated mode default)."""
        path = get_personas_db_path(Path("/test"))
        # In consolidated mode (default), personas maps to agents.db
        assert path.name == "agents.db"

    def test_get_insights_db_path(self):
        """Insights convenience function."""
        path = get_insights_db_path(Path("/test"))
        # In consolidated mode (default), insights maps to analytics.db
        assert path.name == "analytics.db"

    def test_get_genesis_db_path(self):
        """Genesis convenience function."""
        path = get_genesis_db_path(Path("/test"))
        # In consolidated mode (default), genesis maps to agents.db
        assert path.name == "agents.db"


class TestNomicCycleModel:
    """Tests for NomicCycle data model."""

    def test_create_basic_cycle(self):
        """Create a basic cycle with required fields."""
        cycle = NomicCycle(
            loop_id="test-loop",
            cycle_number=1,
            phase="debate",
            stage="proposing",
            started_at=datetime.now(),
        )
        assert cycle.loop_id == "test-loop"
        assert cycle.cycle_number == 1
        assert cycle.phase == "debate"
        assert cycle.stage == "proposing"

    def test_cycle_optional_fields_default(self):
        """Optional fields should have correct defaults."""
        cycle = NomicCycle(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            stage="voting",
            started_at=datetime.now(),
        )
        assert cycle.completed_at is None
        assert cycle.success is None
        assert cycle.git_commit is None
        assert cycle.task_description is None
        assert cycle.total_tasks == 0
        assert cycle.completed_tasks == 0
        assert cycle.error_message is None
        assert cycle.id is None

    def test_cycle_to_dict(self):
        """Cycle serialization to dict."""
        now = datetime.now()
        cycle = NomicCycle(
            loop_id="test",
            cycle_number=1,
            phase="implement",
            stage="executing",
            started_at=now,
            success=True,
        )
        d = cycle.to_dict()
        assert d["loop_id"] == "test"
        assert d["cycle_number"] == 1
        assert d["started_at"] == now.isoformat()
        assert d["success"] is True

    def test_cycle_to_dict_with_completed_at(self):
        """Cycle serialization includes completed_at when set."""
        now = datetime.now()
        completed = datetime.now()
        cycle = NomicCycle(
            loop_id="test",
            cycle_number=1,
            phase="verify",
            stage="done",
            started_at=now,
            completed_at=completed,
        )
        d = cycle.to_dict()
        assert d["completed_at"] == completed.isoformat()


class TestDebateArtifactModel:
    """Tests for DebateArtifact data model."""

    def test_create_artifact(self):
        """Create a debate artifact."""
        artifact = DebateArtifact(
            loop_id="test-loop",
            cycle_number=1,
            phase="debate",
            task="Design a rate limiter",
            agents=["claude", "gpt4"],
            transcript=[{"agent": "claude", "content": "I propose..."}],
            consensus_reached=True,
            confidence=0.85,
        )
        assert artifact.loop_id == "test-loop"
        assert len(artifact.agents) == 2
        assert artifact.consensus_reached is True
        assert artifact.confidence == 0.85

    def test_artifact_to_dict(self):
        """Artifact serialization."""
        artifact = DebateArtifact(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            task="test task",
            agents=["a", "b"],
            transcript=[],
            consensus_reached=False,
            confidence=0.5,
        )
        d = artifact.to_dict()
        assert "loop_id" in d
        assert "created_at" in d
        assert isinstance(d["created_at"], str)


class TestStreamEventModel:
    """Tests for StreamEvent data model."""

    def test_create_event(self):
        """Create a stream event."""
        event = StreamEvent(
            loop_id="test",
            cycle=1,
            event_type="phase_start",
            event_data={"phase": "debate"},
            agent="claude",
        )
        assert event.event_type == "phase_start"
        assert event.event_data["phase"] == "debate"
        assert event.agent == "claude"

    def test_event_to_dict(self):
        """Event serialization."""
        event = StreamEvent(
            loop_id="test",
            cycle=1,
            event_type="task_complete",
            event_data={"success": True},
        )
        d = event.to_dict()
        assert d["event_type"] == "task_complete"
        assert "timestamp" in d


class TestAgentMetricsModel:
    """Tests for AgentMetrics data model."""

    def test_create_metrics(self):
        """Create agent metrics."""
        metrics = AgentMetrics(
            loop_id="test",
            cycle=1,
            agent_name="claude",
            model="claude-3-sonnet",
            phase="debate",
            messages_sent=5,
            proposals_made=2,
            critiques_given=3,
            votes_won=1,
        )
        assert metrics.agent_name == "claude"
        assert metrics.messages_sent == 5
        assert metrics.votes_won == 1

    def test_metrics_defaults(self):
        """Metrics defaults to zero."""
        metrics = AgentMetrics(
            loop_id="test",
            cycle=1,
            agent_name="gpt4",
            model="gpt-4",
            phase="design",
        )
        assert metrics.messages_sent == 0
        assert metrics.proposals_made == 0
        assert metrics.avg_response_time_ms is None

    def test_metrics_to_dict(self):
        """Metrics serialization."""
        metrics = AgentMetrics(
            loop_id="test",
            cycle=1,
            agent_name="test",
            model="test-model",
            phase="implement",
            avg_response_time_ms=150.5,
        )
        d = metrics.to_dict()
        assert d["avg_response_time_ms"] == 150.5


class TestNomicRollbackModel:
    """Tests for NomicRollback data model."""

    def test_create_rollback(self):
        """Create a rollback record."""
        rollback = NomicRollback(
            id="rb-001",
            loop_id="test",
            cycle_number=5,
            phase="verify",
            reason="verify_failure",
            severity="high",
            rolled_back_commit="abc123",
            error_message="Tests failed",
        )
        assert rollback.reason == "verify_failure"
        assert rollback.severity == "high"
        assert rollback.rolled_back_commit == "abc123"

    def test_rollback_files_affected(self):
        """Rollback tracks affected files."""
        rollback = NomicRollback(
            id="rb-002",
            loop_id="test",
            cycle_number=3,
            phase="implement",
            reason="conflict",
            severity="medium",
            files_affected=["src/main.py", "tests/test_main.py"],
        )
        assert len(rollback.files_affected) == 2

    def test_rollback_to_dict(self):
        """Rollback serialization."""
        rollback = NomicRollback(
            id="rb-003",
            loop_id="test",
            cycle_number=1,
            phase="commit",
            reason="manual_intervention",
            severity="critical",
        )
        d = rollback.to_dict()
        assert d["severity"] == "critical"
        assert "created_at" in d


class TestCycleEvolutionModel:
    """Tests for CycleEvolution data model."""

    def test_create_evolution(self):
        """Create an evolution record."""
        evolution = CycleEvolution(
            id="ev-001",
            loop_id="test",
            cycle_number=1,
            winning_proposal_summary="Add caching layer",
            files_changed=["src/cache.py"],
            git_commit="def456",
        )
        assert evolution.winning_proposal_summary == "Add caching layer"
        assert len(evolution.files_changed) == 1

    def test_evolution_with_rollback(self):
        """Evolution linked to rollback."""
        evolution = CycleEvolution(
            id="ev-002",
            loop_id="test",
            cycle_number=2,
            rollback_id="rb-001",
        )
        assert evolution.rollback_id == "rb-001"

    def test_evolution_to_dict(self):
        """Evolution serialization."""
        evolution = CycleEvolution(
            id="ev-003",
            loop_id="test",
            cycle_number=3,
        )
        d = evolution.to_dict()
        assert d["cycle_number"] == 3


class TestCycleFileChangeModel:
    """Tests for CycleFileChange data model."""

    def test_create_file_change(self):
        """Create a file change record."""
        change = CycleFileChange(
            loop_id="test",
            cycle_number=1,
            file_path="src/utils.py",
            change_type="modified",
            insertions=50,
            deletions=10,
        )
        assert change.file_path == "src/utils.py"
        assert change.change_type == "modified"
        assert change.insertions == 50
        assert change.deletions == 10

    def test_file_change_types(self):
        """Various file change types."""
        for change_type in ["added", "modified", "deleted", "renamed"]:
            change = CycleFileChange(
                loop_id="test",
                cycle_number=1,
                file_path="test.py",
                change_type=change_type,
            )
            assert change.change_type == change_type

    def test_file_change_to_dict(self):
        """File change serialization."""
        change = CycleFileChange(
            loop_id="test",
            cycle_number=1,
            file_path="README.md",
            change_type="added",
        )
        d = change.to_dict()
        assert d["file_path"] == "README.md"
        assert d["insertions"] == 0


class TestDatabaseTypeEnumeration:
    """Tests for DatabaseType enum completeness."""

    def test_core_database_types(self):
        """Core database types exist."""
        assert DatabaseType.DEBATES
        assert DatabaseType.TRACES
        assert DatabaseType.TOURNAMENTS
        assert DatabaseType.EMBEDDINGS
        assert DatabaseType.POSITIONS

    def test_memory_database_types(self):
        """Memory database types exist."""
        assert DatabaseType.CONTINUUM_MEMORY
        assert DatabaseType.AGENT_MEMORIES
        assert DatabaseType.CONSENSUS_MEMORY
        assert DatabaseType.AGORA_MEMORY
        assert DatabaseType.SEMANTIC_PATTERNS
        assert DatabaseType.SUGGESTION_FEEDBACK

    def test_analytics_database_types(self):
        """Analytics database types exist."""
        assert DatabaseType.ELO
        assert DatabaseType.CALIBRATION
        assert DatabaseType.INSIGHTS
        assert DatabaseType.PROMPT_EVOLUTION
        assert DatabaseType.META_LEARNING

    def test_agent_database_types(self):
        """Agent database types exist."""
        assert DatabaseType.PERSONAS
        assert DatabaseType.RELATIONSHIPS
        assert DatabaseType.LABORATORY
        assert DatabaseType.TRUTH_GROUNDING
        assert DatabaseType.GENESIS
        assert DatabaseType.GENOMES

    def test_evolution_database_type(self):
        """Evolution database type exists."""
        assert DatabaseType.EVOLUTION


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_nomic_dir_env(self):
        """Empty string for ARAGORA_DATA_DIR."""
        with patch.dict(os.environ, {"ARAGORA_DATA_DIR": ""}):
            nomic_dir = get_nomic_dir()
            # Empty string should be treated as valid path
            assert nomic_dir == Path("")

    def test_cycle_with_all_fields(self):
        """Cycle with all fields populated."""
        now = datetime.now()
        cycle = NomicCycle(
            loop_id="full-test",
            cycle_number=10,
            phase="commit",
            stage="pushing",
            started_at=now,
            completed_at=now,
            success=True,
            git_commit="abc123def456",
            task_description="Full test cycle",
            total_tasks=5,
            completed_tasks=5,
            error_message=None,
            id="cycle-001",
        )
        d = cycle.to_dict()
        assert d["git_commit"] == "abc123def456"
        assert d["total_tasks"] == 5

    def test_artifact_with_vote_tally(self):
        """Artifact with vote tally."""
        artifact = DebateArtifact(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            task="test",
            agents=["a", "b", "c"],
            transcript=[],
            consensus_reached=True,
            confidence=0.9,
            winning_proposal="Proposal A",
            vote_tally={"a": 2, "b": 1, "c": 0},
        )
        d = artifact.to_dict()
        assert d["vote_tally"]["a"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
