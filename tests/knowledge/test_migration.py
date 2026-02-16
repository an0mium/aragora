"""Tests for aragora.knowledge.migration module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.consensus import ConsensusStrength, DissentType
from aragora.memory.tier_manager import MemoryTier


# -----------------------------------------------------------------
# Mock classes for memory types
# -----------------------------------------------------------------


@dataclass
class MockContinuumMemoryEntry:
    """Mock ContinuumMemoryEntry for testing."""

    id: str
    content: str
    tier: MemoryTier
    importance: float = 0.5
    surprise_score: float = 0.3
    update_count: int = 1
    consolidation_score: float = 0.7
    success_rate: float = 0.8
    success_count: int = 4
    failure_count: int = 1
    red_line: bool = False
    red_line_reason: str | None = None
    created_at: str = "2024-01-01T00:00:00"
    updated_at: str = "2024-01-02T00:00:00"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockConsensusRecord:
    """Mock ConsensusRecord for testing."""

    id: str
    topic: str
    conclusion: str
    confidence: float
    strength: ConsensusStrength
    topic_hash: str = "abc123"
    domain: str = "general"
    tags: list[str] = field(default_factory=list)
    key_claims: list[str] = field(default_factory=list)
    dissent_ids: list[str] = field(default_factory=list)
    participating_agents: list[str] = field(default_factory=list)
    agreeing_agents: list[str] = field(default_factory=list)
    dissenting_agents: list[str] = field(default_factory=list)
    rounds: int = 3
    debate_duration_seconds: float = 120.0
    supersedes: str | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockDissentRecord:
    """Mock DissentRecord for testing."""

    id: str
    debate_id: str
    agent_id: str
    content: str
    reasoning: str
    confidence: float
    dissent_type: DissentType
    acknowledged: bool = False
    rebuttal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------
# Test MigrationResult
# -----------------------------------------------------------------


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_default_values(self):
        """Test MigrationResult default values."""
        from aragora.knowledge.migration import MigrationResult

        result = MigrationResult(
            source_type="test",
            total_records=10,
            migrated_count=8,
            skipped_count=1,
            error_count=1,
        )

        assert result.source_type == "test"
        assert result.total_records == 10
        assert result.migrated_count == 8
        assert result.skipped_count == 1
        assert result.error_count == 1
        assert result.node_ids == []
        assert result.relationship_ids == []
        assert result.errors == []
        assert result.duration_seconds == 0.0
        assert result.completed_at is None

    def test_success_rate_with_records(self):
        """Test success_rate calculation with records."""
        from aragora.knowledge.migration import MigrationResult

        result = MigrationResult(
            source_type="test",
            total_records=10,
            migrated_count=8,
            skipped_count=1,
            error_count=1,
        )

        assert result.success_rate == 0.8

    def test_success_rate_zero_records(self):
        """Test success_rate returns 1.0 for zero records."""
        from aragora.knowledge.migration import MigrationResult

        result = MigrationResult(
            source_type="test",
            total_records=0,
            migrated_count=0,
            skipped_count=0,
            error_count=0,
        )

        assert result.success_rate == 1.0

    def test_success_rate_all_migrated(self):
        """Test success_rate when all records migrated."""
        from aragora.knowledge.migration import MigrationResult

        result = MigrationResult(
            source_type="test",
            total_records=100,
            migrated_count=100,
            skipped_count=0,
            error_count=0,
        )

        assert result.success_rate == 1.0

    def test_success_rate_none_migrated(self):
        """Test success_rate when no records migrated."""
        from aragora.knowledge.migration import MigrationResult

        result = MigrationResult(
            source_type="test",
            total_records=10,
            migrated_count=0,
            skipped_count=5,
            error_count=5,
        )

        assert result.success_rate == 0.0

    def test_to_dict_basic(self):
        """Test to_dict conversion."""
        from aragora.knowledge.migration import MigrationResult

        started = datetime(2024, 1, 1, 12, 0, 0)
        result = MigrationResult(
            source_type="continuum_memory",
            total_records=10,
            migrated_count=8,
            skipped_count=1,
            error_count=1,
            node_ids=["n1", "n2", "n3"],
            relationship_ids=["r1", "r2"],
            errors=[{"entry_id": "e1", "error": "test error"}],
            duration_seconds=5.5,
            started_at=started,
        )

        d = result.to_dict()

        assert d["source_type"] == "continuum_memory"
        assert d["total_records"] == 10
        assert d["migrated_count"] == 8
        assert d["skipped_count"] == 1
        assert d["error_count"] == 1
        assert d["success_rate"] == 0.8
        assert d["node_ids_count"] == 3
        assert d["relationship_ids_count"] == 2
        assert len(d["errors"]) == 1
        assert d["duration_seconds"] == 5.5
        assert d["started_at"] == "2024-01-01T12:00:00"
        assert d["completed_at"] is None

    def test_to_dict_with_completed_at(self):
        """Test to_dict with completed_at set."""
        from aragora.knowledge.migration import MigrationResult

        started = datetime(2024, 1, 1, 12, 0, 0)
        completed = datetime(2024, 1, 1, 12, 5, 0)
        result = MigrationResult(
            source_type="test",
            total_records=10,
            migrated_count=10,
            skipped_count=0,
            error_count=0,
            started_at=started,
            completed_at=completed,
        )

        d = result.to_dict()
        assert d["completed_at"] == "2024-01-01T12:05:00"

    def test_to_dict_truncates_errors(self):
        """Test to_dict only includes first 10 errors."""
        from aragora.knowledge.migration import MigrationResult

        errors = [{"entry_id": f"e{i}", "error": f"error {i}"} for i in range(15)]
        result = MigrationResult(
            source_type="test",
            total_records=15,
            migrated_count=0,
            skipped_count=0,
            error_count=15,
            errors=errors,
        )

        d = result.to_dict()
        assert len(d["errors"]) == 10


# -----------------------------------------------------------------
# Test MigrationCheckpoint
# -----------------------------------------------------------------


class TestMigrationCheckpoint:
    """Tests for MigrationCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test basic MigrationCheckpoint creation."""
        from aragora.knowledge.migration import MigrationCheckpoint

        checkpoint = MigrationCheckpoint(
            migration_id="mig_001",
            source_type="continuum_memory",
            last_processed_id="entry_50",
            processed_count=50,
            workspace_id="default",
        )

        assert checkpoint.migration_id == "mig_001"
        assert checkpoint.source_type == "continuum_memory"
        assert checkpoint.last_processed_id == "entry_50"
        assert checkpoint.processed_count == 50
        assert checkpoint.workspace_id == "default"
        assert checkpoint.metadata == {}

    def test_checkpoint_with_metadata(self):
        """Test MigrationCheckpoint with metadata."""
        from aragora.knowledge.migration import MigrationCheckpoint

        checkpoint = MigrationCheckpoint(
            migration_id="mig_002",
            source_type="consensus_memory",
            last_processed_id="cons_25",
            processed_count=25,
            workspace_id="ws_123",
            metadata={"tier_filter": ["FAST", "MEDIUM"], "min_confidence": 0.5},
        )

        assert checkpoint.metadata["tier_filter"] == ["FAST", "MEDIUM"]
        assert checkpoint.metadata["min_confidence"] == 0.5

    def test_checkpoint_has_created_at(self):
        """Test MigrationCheckpoint has default created_at."""
        from aragora.knowledge.migration import MigrationCheckpoint

        before = datetime.now()
        checkpoint = MigrationCheckpoint(
            migration_id="mig_003",
            source_type="test",
            last_processed_id="id_1",
            processed_count=1,
            workspace_id="default",
        )
        after = datetime.now()

        assert before <= checkpoint.created_at <= after


# -----------------------------------------------------------------
# Test MigrationContext
# -----------------------------------------------------------------


class TestMigrationContext:
    """Tests for MigrationContext class."""

    @pytest.mark.asyncio
    async def test_context_enter_sets_started_at(self):
        """Test __aenter__ sets started_at timestamp."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        assert ctx._started_at is None

        async with ctx as entered_ctx:
            assert entered_ctx is ctx
            assert ctx._started_at is not None
            assert isinstance(ctx._started_at, datetime)

    @pytest.mark.asyncio
    async def test_context_successful_completion(self):
        """Test context sets completed=True on success."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        async with ctx:
            ctx.record_node("node_1")
            ctx.record_relationship("rel_1")

        assert ctx._completed is True
        assert ctx._created_node_ids == ["node_1"]
        assert ctx._created_relationship_ids == ["rel_1"]

    @pytest.mark.asyncio
    async def test_context_failure_sets_completed_false(self):
        """Test context sets completed=False on exception."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        with pytest.raises(ValueError):
            async with ctx:
                ctx.record_node("node_1")
                raise ValueError("Migration failed")

        assert ctx._completed is False

    @pytest.mark.asyncio
    async def test_context_does_not_suppress_exception(self):
        """Test context does not suppress exceptions."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        with pytest.raises(RuntimeError, match="test error"):
            async with ctx:
                raise RuntimeError("test error")

    def test_record_node(self):
        """Test record_node adds to tracking list."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        ctx.record_node("node_1")
        ctx.record_node("node_2")
        ctx.record_node("node_3")

        assert ctx._created_node_ids == ["node_1", "node_2", "node_3"]

    def test_record_relationship(self):
        """Test record_relationship adds to tracking list."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        ctx.record_relationship("rel_1")
        ctx.record_relationship("rel_2")

        assert ctx._created_relationship_ids == ["rel_1", "rel_2"]

    @pytest.mark.asyncio
    async def test_context_logs_completion(self):
        """Test context logs completion message."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        with patch("aragora.knowledge.migration.logger") as mock_logger:
            async with ctx:
                ctx.record_node("n1")
                ctx.record_relationship("r1")

            # Verify completion was logged
            mock_logger.info.assert_called()
            call_args = str(mock_logger.info.call_args)
            assert "test_migration" in call_args
            assert "completed" in call_args

    @pytest.mark.asyncio
    async def test_context_logs_error_on_failure(self):
        """Test context logs error message on failure."""
        from aragora.knowledge.migration import MigrationContext

        mock_migrator = MagicMock()
        ctx = MigrationContext(mock_migrator, "test_migration")

        with patch("aragora.knowledge.migration.logger") as mock_logger:
            with pytest.raises(ValueError):
                async with ctx:
                    raise ValueError("test failure")

            mock_logger.error.assert_called()
            call_args = str(mock_logger.error.call_args)
            assert "test_migration" in call_args
            assert "failed" in call_args


# -----------------------------------------------------------------
# Test KnowledgeMoundMigrator
# -----------------------------------------------------------------


class TestKnowledgeMoundMigratorInit:
    """Tests for KnowledgeMoundMigrator initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        assert migrator._mound is mock_mound
        assert migrator._batch_size == 100
        assert migrator._skip_duplicates is True
        assert migrator._checkpoints == {}

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(
            mock_mound,
            batch_size=50,
            skip_duplicates=False,
        )

        assert migrator._batch_size == 50
        assert migrator._skip_duplicates is False

    def test_migration_context_returns_context(self):
        """Test migration_context returns MigrationContext."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator, MigrationContext

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        ctx = migrator.migration_context("test_mig")

        assert isinstance(ctx, MigrationContext)
        assert ctx._migration_id == "test_mig"
        assert ctx._migrator is migrator


# -----------------------------------------------------------------
# Test _continuum_entry_to_node
# -----------------------------------------------------------------


class TestContinuumEntryToNode:
    """Tests for _continuum_entry_to_node conversion."""

    def test_basic_conversion(self):
        """Test basic conversion of ContinuumMemoryEntry to KnowledgeNode."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        entry = MockContinuumMemoryEntry(
            id="entry_001",
            content="Test memory content",
            tier=MemoryTier.FAST,
            importance=0.8,
            surprise_score=0.4,
            update_count=3,
            consolidation_score=0.6,
            success_rate=0.9,
            success_count=9,
            failure_count=1,
        )

        node = migrator._continuum_entry_to_node(entry, "workspace_1")

        assert node.id == "kn_cm_entry_001"
        assert node.node_type == "memory"
        assert node.content == "Test memory content"
        assert node.confidence == 0.8
        assert node.workspace_id == "workspace_1"
        assert node.surprise_score == 0.4
        assert node.update_count == 3
        assert node.consolidation_score == 0.6

    def test_high_success_rate_sets_majority_agreed(self):
        """Test high success_rate (>0.7) sets MAJORITY_AGREED status."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.types import ValidationStatus

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        entry = MockContinuumMemoryEntry(
            id="entry_002",
            content="High success content",
            tier=MemoryTier.MEDIUM,
            success_rate=0.85,
        )

        node = migrator._continuum_entry_to_node(entry, "default")

        assert node.validation_status == ValidationStatus.MAJORITY_AGREED

    def test_low_success_rate_sets_unverified(self):
        """Test low success_rate (<=0.7) sets UNVERIFIED status."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.types import ValidationStatus

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        entry = MockContinuumMemoryEntry(
            id="entry_003",
            content="Low success content",
            tier=MemoryTier.SLOW,
            success_rate=0.5,
        )

        node = migrator._continuum_entry_to_node(entry, "default")

        assert node.validation_status == ValidationStatus.UNVERIFIED

    def test_metadata_preserved(self):
        """Test original metadata is preserved in node."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        entry = MockContinuumMemoryEntry(
            id="entry_004",
            content="Content with metadata",
            tier=MemoryTier.GLACIAL,
            success_count=5,
            failure_count=2,
            red_line=True,
            red_line_reason="Safety concern",
            metadata={"custom_key": "custom_value"},
        )

        node = migrator._continuum_entry_to_node(entry, "default")

        assert node.metadata["source"] == "continuum_memory"
        assert node.metadata["original_id"] == "entry_004"
        assert node.metadata["success_count"] == 5
        assert node.metadata["failure_count"] == 2
        assert node.metadata["red_line"] is True
        assert node.metadata["red_line_reason"] == "Safety concern"
        assert node.metadata["custom_key"] == "custom_value"

    def test_provenance_chain_set(self):
        """Test provenance chain is properly set."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.mound import ProvenanceType

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        entry = MockContinuumMemoryEntry(
            id="entry_005",
            content="Provenance test",
            tier=MemoryTier.FAST,
        )

        node = migrator._continuum_entry_to_node(entry, "default")

        assert node.provenance.source_type == ProvenanceType.MIGRATION
        assert node.provenance.source_id == "continuum:entry_005"


# -----------------------------------------------------------------
# Test _consensus_record_to_node
# -----------------------------------------------------------------


class TestConsensusRecordToNode:
    """Tests for _consensus_record_to_node conversion."""

    def test_basic_conversion(self):
        """Test basic conversion of ConsensusRecord to KnowledgeNode."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.mound import ProvenanceType

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        record = MockConsensusRecord(
            id="cons_001",
            topic="Test topic",
            conclusion="The consensus conclusion",
            confidence=0.85,
            strength=ConsensusStrength.STRONG,
            tags=["tag1", "tag2"],
        )

        node = migrator._consensus_record_to_node(record, "workspace_1")

        assert node.id == "kn_cr_cons_001"
        assert node.node_type == "consensus"
        assert node.content == "The consensus conclusion"
        assert node.confidence == 0.85
        assert node.workspace_id == "workspace_1"
        assert node.topics == ["tag1", "tag2"]
        assert node.provenance.source_type == ProvenanceType.DEBATE
        assert node.provenance.debate_id == "cons_001"

    def test_unanimous_strength_maps_to_byzantine_agreed(self):
        """Test UNANIMOUS strength maps to BYZANTINE_AGREED validation status."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.types import ValidationStatus

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        record = MockConsensusRecord(
            id="cons_002",
            topic="Topic",
            conclusion="Unanimous conclusion",
            confidence=0.99,
            strength=ConsensusStrength.UNANIMOUS,
        )

        node = migrator._consensus_record_to_node(record, "default")

        assert node.validation_status == ValidationStatus.BYZANTINE_AGREED

    def test_strong_strength_maps_to_majority_agreed(self):
        """Test STRONG strength maps to MAJORITY_AGREED validation status."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.types import ValidationStatus

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        record = MockConsensusRecord(
            id="cons_003",
            topic="Topic",
            conclusion="Strong conclusion",
            confidence=0.8,
            strength=ConsensusStrength.STRONG,
        )

        node = migrator._consensus_record_to_node(record, "default")

        assert node.validation_status == ValidationStatus.MAJORITY_AGREED

    def test_weak_strength_maps_to_contested(self):
        """Test WEAK strength maps to CONTESTED validation status."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.types import ValidationStatus

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        record = MockConsensusRecord(
            id="cons_004",
            topic="Topic",
            conclusion="Weak conclusion",
            confidence=0.5,
            strength=ConsensusStrength.WEAK,
        )

        node = migrator._consensus_record_to_node(record, "default")

        assert node.validation_status == ValidationStatus.CONTESTED

    def test_metadata_includes_all_fields(self):
        """Test all consensus record fields are in metadata."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        record = MockConsensusRecord(
            id="cons_005",
            topic="Detailed topic",
            topic_hash="hash123",
            conclusion="Detailed conclusion",
            confidence=0.75,
            strength=ConsensusStrength.MODERATE,
            domain="science",
            participating_agents=["agent1", "agent2", "agent3"],
            agreeing_agents=["agent1", "agent2"],
            dissenting_agents=["agent3"],
            rounds=5,
            debate_duration_seconds=300.0,
            supersedes="old_cons_001",
            superseded_by=None,
            metadata={"extra": "data"},
        )

        node = migrator._consensus_record_to_node(record, "default")

        assert node.metadata["source"] == "consensus_memory"
        assert node.metadata["original_id"] == "cons_005"
        assert node.metadata["topic"] == "Detailed topic"
        assert node.metadata["topic_hash"] == "hash123"
        assert node.metadata["strength"] == "moderate"
        assert node.metadata["domain"] == "science"
        assert node.metadata["participating_agents"] == ["agent1", "agent2", "agent3"]
        assert node.metadata["agreeing_agents"] == ["agent1", "agent2"]
        assert node.metadata["dissenting_agents"] == ["agent3"]
        assert node.metadata["rounds"] == 5
        assert node.metadata["debate_duration_seconds"] == 300.0
        assert node.metadata["supersedes"] == "old_cons_001"
        assert node.metadata["superseded_by"] is None
        assert node.metadata["extra"] == "data"


# -----------------------------------------------------------------
# Test _dissent_record_to_node
# -----------------------------------------------------------------


class TestDissentRecordToNode:
    """Tests for _dissent_record_to_node conversion."""

    def test_basic_conversion(self):
        """Test basic conversion of DissentRecord to KnowledgeNode."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator
        from aragora.knowledge.mound import ProvenanceType
        from aragora.knowledge.types import ValidationStatus

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        dissent = MockDissentRecord(
            id="dissent_001",
            debate_id="debate_123",
            agent_id="agent_claude",
            content="I disagree with the conclusion",
            reasoning="Because of X, Y, and Z",
            confidence=0.7,
            dissent_type=DissentType.FUNDAMENTAL_DISAGREEMENT,
        )

        node = migrator._dissent_record_to_node(dissent, "workspace_1")

        assert node.id == "kn_dr_dissent_001"
        assert node.node_type == "claim"
        assert "I disagree with the conclusion" in node.content
        assert "Because of X, Y, and Z" in node.content
        assert node.confidence == 0.7
        assert node.workspace_id == "workspace_1"
        assert node.validation_status == ValidationStatus.CONTESTED
        assert node.provenance.source_type == ProvenanceType.AGENT
        assert node.provenance.agent_id == "agent_claude"
        assert node.provenance.debate_id == "debate_123"

    def test_metadata_includes_all_fields(self):
        """Test all dissent record fields are in metadata."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        dissent = MockDissentRecord(
            id="dissent_002",
            debate_id="debate_456",
            agent_id="agent_gpt",
            content="Dissent content",
            reasoning="Dissent reasoning",
            confidence=0.6,
            dissent_type=DissentType.ALTERNATIVE_APPROACH,
            acknowledged=True,
            rebuttal="This was addressed",
            metadata={"priority": "high"},
        )

        node = migrator._dissent_record_to_node(dissent, "default")

        assert node.metadata["source"] == "dissent_record"
        assert node.metadata["original_id"] == "dissent_002"
        assert node.metadata["debate_id"] == "debate_456"
        assert node.metadata["agent_id"] == "agent_gpt"
        assert node.metadata["dissent_type"] == "alternative_approach"
        assert node.metadata["acknowledged"] is True
        assert node.metadata["rebuttal"] == "This was addressed"
        assert node.metadata["priority"] == "high"


# -----------------------------------------------------------------
# Test migrate_continuum_memory
# -----------------------------------------------------------------


class TestMigrateContinuumMemory:
    """Tests for migrate_continuum_memory method."""

    @pytest.mark.asyncio
    async def test_migrate_empty_source(self):
        """Test migration with empty source."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        mock_source = MagicMock()
        mock_source.get_all_entries = MagicMock(return_value=[])

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.source_type == "continuum_memory"
        assert result.total_records == 0
        assert result.migrated_count == 0
        assert result.skipped_count == 0
        assert result.error_count == 0
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_migrate_single_entry(self):
        """Test migration of single entry."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(return_value="node_001")

        mock_source = MagicMock()
        entry = MockContinuumMemoryEntry(
            id="entry_001",
            content="Test content",
            tier=MemoryTier.FAST,
        )
        mock_source.get_all_entries = MagicMock(return_value=[entry])

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.total_records == 1
        assert result.migrated_count == 1
        assert result.node_ids == ["node_001"]
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_migrate_multiple_entries(self):
        """Test migration of multiple entries."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(side_effect=["node_001", "node_002", "node_003"])

        mock_source = MagicMock()
        entries = [
            MockContinuumMemoryEntry(id=f"entry_{i}", content=f"Content {i}", tier=MemoryTier.FAST)
            for i in range(3)
        ]
        mock_source.get_all_entries = MagicMock(return_value=entries)

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.total_records == 3
        assert result.migrated_count == 3
        assert len(result.node_ids) == 3

    @pytest.mark.asyncio
    async def test_migrate_with_tier_filter(self):
        """Test migration filters by tier."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(return_value="node_001")

        mock_source = MagicMock()
        entries = [
            MockContinuumMemoryEntry(id="fast_1", content="Fast content", tier=MemoryTier.FAST),
            MockContinuumMemoryEntry(
                id="medium_1", content="Medium content", tier=MemoryTier.MEDIUM
            ),
            MockContinuumMemoryEntry(id="slow_1", content="Slow content", tier=MemoryTier.SLOW),
        ]
        mock_source.get_all_entries = MagicMock(return_value=entries)

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(
            mock_source, "default", tier_filter=[MemoryTier.FAST]
        )

        assert result.total_records == 1
        assert result.migrated_count == 1

    @pytest.mark.asyncio
    async def test_migrate_with_min_importance(self):
        """Test migration filters by minimum importance."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(side_effect=["node_001", "node_002"])

        mock_source = MagicMock()
        entries = [
            MockContinuumMemoryEntry(
                id="high_1", content="High importance", tier=MemoryTier.FAST, importance=0.9
            ),
            MockContinuumMemoryEntry(
                id="medium_1", content="Medium importance", tier=MemoryTier.FAST, importance=0.5
            ),
            MockContinuumMemoryEntry(
                id="high_2", content="High importance 2", tier=MemoryTier.FAST, importance=0.8
            ),
        ]
        mock_source.get_all_entries = MagicMock(return_value=entries)

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(mock_source, "default", min_importance=0.7)

        # Only entries with importance >= 0.7 should be migrated
        assert result.total_records == 2
        assert result.migrated_count == 2

    @pytest.mark.asyncio
    async def test_migrate_handles_entry_error(self):
        """Test migration handles errors for individual entries."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(
            side_effect=["node_001", RuntimeError("Add failed"), "node_003"]
        )

        mock_source = MagicMock()
        entries = [
            MockContinuumMemoryEntry(id=f"entry_{i}", content=f"Content {i}", tier=MemoryTier.FAST)
            for i in range(3)
        ]
        mock_source.get_all_entries = MagicMock(return_value=entries)

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.total_records == 3
        assert result.migrated_count == 2
        assert result.error_count == 1
        assert len(result.errors) == 1
        assert result.errors[0]["entry_id"] == "entry_1"

    @pytest.mark.asyncio
    async def test_migrate_source_missing_method(self):
        """Test migration handles source without get_all_entries."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        mock_source = MagicMock(spec=[])  # No methods

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.error_count == 0  # Error is in errors list as fatal
        assert len(result.errors) == 1
        assert result.errors[0]["fatal"] is True

    @pytest.mark.asyncio
    async def test_migrate_records_duration(self):
        """Test migration records duration."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(return_value="node_001")

        mock_source = MagicMock()
        entry = MockContinuumMemoryEntry(id="entry_001", content="Content", tier=MemoryTier.FAST)
        mock_source.get_all_entries = MagicMock(return_value=[entry])

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.duration_seconds >= 0


# -----------------------------------------------------------------
# Test migrate_consensus_memory
# -----------------------------------------------------------------


class TestMigrateConsensusMemory:
    """Tests for migrate_consensus_memory method."""

    @pytest.mark.asyncio
    async def test_migrate_empty_source(self):
        """Test migration with empty consensus source."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        mock_source = MagicMock()
        mock_source.get_all_consensus = MagicMock(return_value=[])

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_consensus_memory(mock_source, "default")

        assert result.source_type == "consensus_memory"
        assert result.total_records == 0
        assert result.migrated_count == 0

    @pytest.mark.asyncio
    async def test_migrate_single_consensus(self):
        """Test migration of single consensus record."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(return_value="cons_node_001")
        mock_mound.add_relationship = AsyncMock(return_value=None)

        mock_source = MagicMock()
        record = MockConsensusRecord(
            id="cons_001",
            topic="Test topic",
            conclusion="Test conclusion",
            confidence=0.8,
            strength=ConsensusStrength.STRONG,
            key_claims=[],
            dissent_ids=[],
        )
        mock_source.get_all_consensus = MagicMock(return_value=[record])

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_consensus_memory(mock_source, "default")

        assert result.total_records == 1
        assert result.migrated_count == 1
        assert "cons_node_001" in result.node_ids

    @pytest.mark.asyncio
    async def test_migrate_consensus_with_key_claims(self):
        """Test migration creates nodes and relationships for key claims."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(side_effect=["cons_node", "claim_node_0", "claim_node_1"])
        mock_mound.add_relationship = AsyncMock(side_effect=["rel_0", "rel_1"])

        mock_source = MagicMock()
        record = MockConsensusRecord(
            id="cons_001",
            topic="Topic",
            conclusion="Conclusion",
            confidence=0.8,
            strength=ConsensusStrength.STRONG,
            key_claims=["Claim 1", "Claim 2"],
            dissent_ids=[],
        )
        mock_source.get_all_consensus = MagicMock(return_value=[record])

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_consensus_memory(mock_source, "default")

        # 1 consensus + 2 claims = 3 nodes
        assert len(result.node_ids) == 3
        # 2 "supports" relationships
        assert len(result.relationship_ids) == 2
        # add_relationship called with correct relationship_type
        calls = mock_mound.add_relationship.call_args_list
        for call in calls:
            assert call.kwargs["relationship_type"] == "supports"

    @pytest.mark.asyncio
    async def test_migrate_consensus_with_dissent(self):
        """Test migration includes dissent records and relationships."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(side_effect=["cons_node", "dissent_node"])
        mock_mound.add_relationship = AsyncMock(return_value="rel_dissent")

        dissent = MockDissentRecord(
            id="dissent_001",
            debate_id="cons_001",
            agent_id="agent_1",
            content="I disagree",
            reasoning="Because",
            confidence=0.6,
            dissent_type=DissentType.FUNDAMENTAL_DISAGREEMENT,
        )

        mock_source = MagicMock()
        record = MockConsensusRecord(
            id="cons_001",
            topic="Topic",
            conclusion="Conclusion",
            confidence=0.8,
            strength=ConsensusStrength.STRONG,
            key_claims=[],
            dissent_ids=["dissent_001"],
        )
        mock_source.get_all_consensus = MagicMock(return_value=[record])
        mock_source.get_dissent = MagicMock(return_value=dissent)

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_consensus_memory(
            mock_source, "default", include_dissent=True
        )

        # 1 consensus + 1 dissent = 2 nodes
        assert len(result.node_ids) == 2
        # 1 "contradicts" relationship
        assert len(result.relationship_ids) == 1
        call = mock_mound.add_relationship.call_args
        assert call.kwargs["relationship_type"] == "contradicts"

    @pytest.mark.asyncio
    async def test_migrate_consensus_without_dissent(self):
        """Test migration can exclude dissent records."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(return_value="cons_node")
        mock_mound.add_relationship = AsyncMock()

        mock_source = MagicMock()
        record = MockConsensusRecord(
            id="cons_001",
            topic="Topic",
            conclusion="Conclusion",
            confidence=0.8,
            strength=ConsensusStrength.STRONG,
            key_claims=[],
            dissent_ids=["dissent_001"],  # Has dissent, but we won't migrate it
        )
        mock_source.get_all_consensus = MagicMock(return_value=[record])

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_consensus_memory(
            mock_source, "default", include_dissent=False
        )

        # Only consensus node, no dissent
        assert len(result.node_ids) == 1
        # No relationships
        assert len(result.relationship_ids) == 0

    @pytest.mark.asyncio
    async def test_migrate_consensus_with_min_confidence(self):
        """Test migration filters by minimum confidence."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(return_value="cons_node")

        mock_source = MagicMock()
        records = [
            MockConsensusRecord(
                id="high",
                topic="High",
                conclusion="High confidence",
                confidence=0.9,
                strength=ConsensusStrength.STRONG,
            ),
            MockConsensusRecord(
                id="low",
                topic="Low",
                conclusion="Low confidence",
                confidence=0.4,
                strength=ConsensusStrength.WEAK,
            ),
        ]
        mock_source.get_all_consensus = MagicMock(return_value=records)

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_consensus_memory(mock_source, "default", min_confidence=0.7)

        assert result.total_records == 1
        assert result.migrated_count == 1

    @pytest.mark.asyncio
    async def test_migrate_consensus_handles_error(self):
        """Test migration handles errors for individual records."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(side_effect=["cons_001", RuntimeError("Failed"), "cons_003"])

        mock_source = MagicMock()
        records = [
            MockConsensusRecord(
                id=f"cons_{i}",
                topic=f"Topic {i}",
                conclusion=f"Conclusion {i}",
                confidence=0.8,
                strength=ConsensusStrength.STRONG,
            )
            for i in range(3)
        ]
        mock_source.get_all_consensus = MagicMock(return_value=records)

        migrator = KnowledgeMoundMigrator(mock_mound)

        result = await migrator.migrate_consensus_memory(mock_source, "default")

        assert result.total_records == 3
        assert result.migrated_count == 2
        assert result.error_count == 1


# -----------------------------------------------------------------
# Test migrate_all
# -----------------------------------------------------------------


class TestMigrateAll:
    """Tests for migrate_all method."""

    @pytest.mark.asyncio
    async def test_migrate_all_both_sources(self):
        """Test migrate_all with both sources."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(return_value="node_id")

        continuum_source = MagicMock()
        entry = MockContinuumMemoryEntry(id="entry_1", content="Content", tier=MemoryTier.FAST)
        continuum_source.get_all_entries = MagicMock(return_value=[entry])

        consensus_source = MagicMock()
        record = MockConsensusRecord(
            id="cons_1",
            topic="Topic",
            conclusion="Conclusion",
            confidence=0.8,
            strength=ConsensusStrength.STRONG,
        )
        consensus_source.get_all_consensus = MagicMock(return_value=[record])

        migrator = KnowledgeMoundMigrator(mock_mound)

        results = await migrator.migrate_all(
            workspace_id="default",
            continuum_source=continuum_source,
            consensus_source=consensus_source,
        )

        assert "continuum" in results
        assert "consensus" in results
        assert results["continuum"].migrated_count == 1
        assert results["consensus"].migrated_count == 1

    @pytest.mark.asyncio
    async def test_migrate_all_only_continuum(self):
        """Test migrate_all with only continuum source."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[]))
        mock_mound.add_node = AsyncMock(return_value="node_id")

        continuum_source = MagicMock()
        entry = MockContinuumMemoryEntry(id="entry_1", content="Content", tier=MemoryTier.FAST)
        continuum_source.get_all_entries = MagicMock(return_value=[entry])

        migrator = KnowledgeMoundMigrator(mock_mound)

        results = await migrator.migrate_all(
            workspace_id="default",
            continuum_source=continuum_source,
            consensus_source=None,
        )

        assert "continuum" in results
        assert "consensus" not in results

    @pytest.mark.asyncio
    async def test_migrate_all_only_consensus(self):
        """Test migrate_all with only consensus source."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        mock_mound.add_node = AsyncMock(return_value="node_id")

        consensus_source = MagicMock()
        record = MockConsensusRecord(
            id="cons_1",
            topic="Topic",
            conclusion="Conclusion",
            confidence=0.8,
            strength=ConsensusStrength.STRONG,
        )
        consensus_source.get_all_consensus = MagicMock(return_value=[record])

        migrator = KnowledgeMoundMigrator(mock_mound)

        results = await migrator.migrate_all(
            workspace_id="default",
            continuum_source=None,
            consensus_source=consensus_source,
        )

        assert "consensus" in results
        assert "continuum" not in results

    @pytest.mark.asyncio
    async def test_migrate_all_no_sources(self):
        """Test migrate_all with no sources."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        results = await migrator.migrate_all(
            workspace_id="default",
            continuum_source=None,
            consensus_source=None,
        )

        assert results == {}


# -----------------------------------------------------------------
# Test dry_run
# -----------------------------------------------------------------


class TestDryRun:
    """Tests for dry_run method."""

    @pytest.mark.asyncio
    async def test_dry_run_continuum_estimates(self):
        """Test dry_run estimates for continuum memory."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        continuum_source = MagicMock()
        entries = [
            MockContinuumMemoryEntry(id="e1", content="C1", tier=MemoryTier.FAST),
            MockContinuumMemoryEntry(id="e2", content="C2", tier=MemoryTier.FAST),
            MockContinuumMemoryEntry(id="e3", content="C3", tier=MemoryTier.SLOW),
        ]
        continuum_source.get_all_entries = MagicMock(return_value=entries)

        migrator = KnowledgeMoundMigrator(mock_mound)

        estimates = await migrator.dry_run(
            workspace_id="default",
            continuum_source=continuum_source,
        )

        assert "continuum" in estimates
        assert estimates["continuum"]["total_records"] == 3
        assert estimates["continuum"]["estimated_nodes"] == 3
        assert estimates["continuum"]["by_tier"]["fast"] == 2
        assert estimates["continuum"]["by_tier"]["slow"] == 1

    @pytest.mark.asyncio
    async def test_dry_run_consensus_estimates(self):
        """Test dry_run estimates for consensus memory."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        consensus_source = MagicMock()
        records = [
            MockConsensusRecord(
                id="c1",
                topic="T1",
                conclusion="Conclusion",
                confidence=0.8,
                strength=ConsensusStrength.STRONG,
                key_claims=["claim1", "claim2"],
                dissent_ids=["d1"],
            ),
            MockConsensusRecord(
                id="c2",
                topic="T2",
                conclusion="Conclusion 2",
                confidence=0.7,
                strength=ConsensusStrength.MODERATE,
                key_claims=["claim3"],
                dissent_ids=["d2", "d3"],
            ),
        ]
        consensus_source.get_all_consensus = MagicMock(return_value=records)

        migrator = KnowledgeMoundMigrator(mock_mound)

        estimates = await migrator.dry_run(
            workspace_id="default",
            consensus_source=consensus_source,
        )

        assert "consensus" in estimates
        assert estimates["consensus"]["total_records"] == 2
        assert estimates["consensus"]["total_dissents"] == 3  # 1 + 2
        assert estimates["consensus"]["total_key_claims"] == 3  # 2 + 1
        # 2 consensus + 3 dissents + 3 claims = 8 nodes
        assert estimates["consensus"]["estimated_nodes"] == 8
        # 3 dissents + 3 claims = 6 relationships
        assert estimates["consensus"]["estimated_relationships"] == 6

    @pytest.mark.asyncio
    async def test_dry_run_no_sources(self):
        """Test dry_run with no sources returns empty."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        migrator = KnowledgeMoundMigrator(mock_mound)

        estimates = await migrator.dry_run(workspace_id="default")

        assert estimates == {}

    @pytest.mark.asyncio
    async def test_dry_run_source_without_method(self):
        """Test dry_run handles source without required method."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = MagicMock()
        continuum_source = MagicMock(spec=[])  # No methods

        migrator = KnowledgeMoundMigrator(mock_mound)

        estimates = await migrator.dry_run(
            workspace_id="default",
            continuum_source=continuum_source,
        )

        # Should handle gracefully with empty list
        assert estimates["continuum"]["total_records"] == 0


# -----------------------------------------------------------------
# Test skip_duplicates behavior
# -----------------------------------------------------------------


class TestSkipDuplicates:
    """Tests for duplicate skipping behavior."""

    @pytest.mark.asyncio
    async def test_skip_duplicate_entry(self):
        """Test skipping duplicate entry based on content hash."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        # Create mock existing item with matching content_hash in metadata
        mock_existing = MagicMock()
        mock_existing.metadata = {"content_hash": "abc123"}

        mock_mound = AsyncMock()
        mock_mound.query = AsyncMock(return_value=MagicMock(items=[mock_existing]))
        mock_mound.add_node = AsyncMock(return_value="node_id")

        mock_source = MagicMock()
        entry = MockContinuumMemoryEntry(id="entry_1", content="Content", tier=MemoryTier.FAST)
        mock_source.get_all_entries = MagicMock(return_value=[entry])

        migrator = KnowledgeMoundMigrator(mock_mound, skip_duplicates=True)

        # Create a mock node with content_hash attribute
        mock_node = MagicMock()
        mock_node.content = "Content"
        mock_node.content_hash = "abc123"

        # Replace the conversion function to return our mock node
        migrator._continuum_entry_to_node = MagicMock(return_value=mock_node)
        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.total_records == 1
        assert result.skipped_count == 1
        assert result.migrated_count == 0

    @pytest.mark.asyncio
    async def test_no_skip_when_disabled(self):
        """Test duplicates are not skipped when skip_duplicates=False."""
        from aragora.knowledge.migration import KnowledgeMoundMigrator

        mock_mound = AsyncMock()
        # Query would return items, but should not be called
        mock_mound.query = AsyncMock()
        mock_mound.add_node = AsyncMock(return_value="node_id")

        mock_source = MagicMock()
        entry = MockContinuumMemoryEntry(id="entry_1", content="Content", tier=MemoryTier.FAST)
        mock_source.get_all_entries = MagicMock(return_value=[entry])

        migrator = KnowledgeMoundMigrator(mock_mound, skip_duplicates=False)

        result = await migrator.migrate_continuum_memory(mock_source, "default")

        assert result.migrated_count == 1
        # Query should not be called when skip_duplicates is False
        mock_mound.query.assert_not_called()
