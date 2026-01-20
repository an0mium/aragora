"""
Tests for ContinuumAdapter bidirectional integration (KM ↔ ContinuumMemory).

Tests the reverse flow methods that enable Knowledge Mound validations
to improve ContinuumMemory tier placement and importance scores.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from aragora.memory.continuum import ContinuumMemory
from aragora.memory.tier_manager import MemoryTier
from aragora.knowledge.mound.adapters.continuum_adapter import (
    ContinuumAdapter,
    KMValidationResult,
    ValidationSyncResult,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_continuum.db"


@pytest.fixture
def continuum_memory(temp_db):
    """Create a ContinuumMemory instance for testing."""
    cm = ContinuumMemory(db_path=str(temp_db))
    yield cm
    # Connection cleanup handled automatically by SQLiteStore context manager


@pytest.fixture
def adapter(continuum_memory):
    """Create a ContinuumAdapter for testing."""
    return ContinuumAdapter(continuum_memory)


class TestKMValidationResult:
    """Tests for the KMValidationResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        validation = KMValidationResult(
            memory_id="test_123",
            km_confidence=0.8,
        )
        assert validation.memory_id == "test_123"
        assert validation.km_confidence == 0.8
        assert validation.cross_debate_utility == 0.0
        assert validation.validation_count == 1
        assert validation.was_contradicted is False
        assert validation.was_supported is False
        assert validation.recommendation == "keep"
        assert validation.metadata == {}

    def test_custom_values(self):
        """Test custom values are applied correctly."""
        validation = KMValidationResult(
            memory_id="test_456",
            km_confidence=0.95,
            cross_debate_utility=0.7,
            validation_count=5,
            was_contradicted=False,
            was_supported=True,
            recommendation="promote",
            metadata={"source": "debate_123"},
        )
        assert validation.km_confidence == 0.95
        assert validation.cross_debate_utility == 0.7
        assert validation.validation_count == 5
        assert validation.was_supported is True
        assert validation.recommendation == "promote"
        assert validation.metadata["source"] == "debate_123"


class TestContinuumAdapterReverseFlow:
    """Tests for the reverse flow methods (KM → ContinuumMemory)."""

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_basic(self, adapter, continuum_memory):
        """Test basic KM validation update."""
        # Create a memory entry
        continuum_memory.add(
            id="mem_001",
            content="Test memory content",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        # Create KM validation
        validation = KMValidationResult(
            memory_id="mem_001",
            km_confidence=0.9,
            cross_debate_utility=0.5,
            validation_count=3,
            recommendation="keep",
        )

        # Apply validation
        result = await adapter.update_continuum_from_km("mem_001", validation)
        assert result is True

        # Verify the update
        entry = continuum_memory.get("mem_001")
        assert entry is not None
        assert entry.metadata.get("km_validated") is True
        assert entry.metadata.get("km_confidence") == 0.9
        assert entry.metadata.get("km_validation_count") == 3

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_with_cm_prefix(self, adapter, continuum_memory):
        """Test KM validation update with cm_ prefix."""
        continuum_memory.add(
            id="mem_002",
            content="Test with prefix",
            tier=MemoryTier.SLOW,
            importance=0.6,
        )

        validation = KMValidationResult(
            memory_id="cm_mem_002",  # With prefix
            km_confidence=0.85,
            recommendation="keep",
        )

        result = await adapter.update_continuum_from_km("cm_mem_002", validation)
        assert result is True

        entry = continuum_memory.get("mem_002")
        assert entry.metadata.get("km_validated") is True

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_not_found(self, adapter):
        """Test KM validation for non-existent entry."""
        validation = KMValidationResult(
            memory_id="nonexistent",
            km_confidence=0.9,
        )

        result = await adapter.update_continuum_from_km("nonexistent", validation)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_promotion(self, adapter, continuum_memory):
        """Test KM validation triggers tier promotion."""
        continuum_memory.add(
            id="mem_promote",
            content="Candidate for promotion",
            tier=MemoryTier.MEDIUM,
            importance=0.6,
        )

        validation = KMValidationResult(
            memory_id="mem_promote",
            km_confidence=0.95,
            cross_debate_utility=0.85,
            validation_count=5,
            recommendation="promote",
        )

        result = await adapter.update_continuum_from_km("mem_promote", validation)
        assert result is True

        entry = continuum_memory.get("mem_promote")
        # Should be promoted from MEDIUM to SLOW (more permanent)
        assert entry.tier == MemoryTier.SLOW

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_demotion(self, adapter, continuum_memory):
        """Test KM validation triggers tier demotion."""
        continuum_memory.add(
            id="mem_demote",
            content="Candidate for demotion",
            tier=MemoryTier.SLOW,
            importance=0.7,
        )

        validation = KMValidationResult(
            memory_id="mem_demote",
            km_confidence=0.3,
            cross_debate_utility=0.1,
            was_contradicted=True,
            recommendation="demote",
        )

        result = await adapter.update_continuum_from_km("mem_demote", validation)
        assert result is True

        entry = continuum_memory.get("mem_demote")
        # Should be demoted from SLOW to MEDIUM (less permanent)
        assert entry.tier == MemoryTier.MEDIUM

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_no_promotion_at_glacial(self, adapter, continuum_memory):
        """Test that glacial tier entries cannot be promoted further."""
        continuum_memory.add(
            id="mem_glacial",
            content="Already at glacial",
            tier=MemoryTier.GLACIAL,
            importance=0.9,
        )

        validation = KMValidationResult(
            memory_id="mem_glacial",
            km_confidence=0.99,
            recommendation="promote",
        )

        result = await adapter.update_continuum_from_km("mem_glacial", validation)
        # Should still update metadata but tier stays the same

        entry = continuum_memory.get("mem_glacial")
        assert entry.tier == MemoryTier.GLACIAL  # Unchanged
        assert entry.metadata.get("km_validated") is True

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_no_demotion_at_fast(self, adapter, continuum_memory):
        """Test that fast tier entries cannot be demoted further."""
        continuum_memory.add(
            id="mem_fast",
            content="Already at fast",
            tier=MemoryTier.FAST,
            importance=0.3,
        )

        validation = KMValidationResult(
            memory_id="mem_fast",
            km_confidence=0.2,
            recommendation="demote",
        )

        result = await adapter.update_continuum_from_km("mem_fast", validation)

        entry = continuum_memory.get("mem_fast")
        assert entry.tier == MemoryTier.FAST  # Unchanged

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_importance_adjustment(self, adapter, continuum_memory):
        """Test KM validation adjusts importance score."""
        continuum_memory.add(
            id="mem_importance",
            content="Test importance adjustment",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        validation = KMValidationResult(
            memory_id="mem_importance",
            km_confidence=0.95,
            cross_debate_utility=0.8,
            validation_count=5,
            recommendation="keep",
        )

        result = await adapter.update_continuum_from_km("mem_importance", validation)
        assert result is True

        entry = continuum_memory.get("mem_importance")
        # Importance should increase (weighted average + utility boost)
        assert entry.importance > 0.5

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_supported_flag(self, adapter, continuum_memory):
        """Test KM validation with was_supported flag."""
        continuum_memory.add(
            id="mem_supported",
            content="Supported memory",
            tier=MemoryTier.SLOW,
            importance=0.6,
        )

        validation = KMValidationResult(
            memory_id="mem_supported",
            km_confidence=0.85,
            was_supported=True,
            recommendation="keep",
        )

        await adapter.update_continuum_from_km("mem_supported", validation)

        entry = continuum_memory.get("mem_supported")
        assert entry.metadata.get("km_supported") is True
        assert entry.metadata.get("km_contradicted") is None or entry.metadata.get("km_contradicted") is False

    @pytest.mark.asyncio
    async def test_update_continuum_from_km_contradicted_flag(self, adapter, continuum_memory):
        """Test KM validation with was_contradicted flag."""
        continuum_memory.add(
            id="mem_contradicted",
            content="Contradicted memory",
            tier=MemoryTier.SLOW,
            importance=0.6,
        )

        validation = KMValidationResult(
            memory_id="mem_contradicted",
            km_confidence=0.4,
            was_contradicted=True,
            recommendation="review",
        )

        await adapter.update_continuum_from_km("mem_contradicted", validation)

        entry = continuum_memory.get("mem_contradicted")
        assert entry.metadata.get("km_contradicted") is True


class TestContinuumAdapterBatchSync:
    """Tests for batch sync of KM validations."""

    @pytest.mark.asyncio
    async def test_sync_validations_to_continuum_basic(self, adapter, continuum_memory):
        """Test batch sync of validations."""
        # Create multiple memory entries
        for i in range(5):
            continuum_memory.add(
                id=f"batch_mem_{i}",
                content=f"Batch memory content {i}",
                tier=MemoryTier.MEDIUM,
                importance=0.5,
            )

        # Create validations
        validations = [
            KMValidationResult(
                memory_id=f"batch_mem_{i}",
                km_confidence=0.8 + (i * 0.02),
                recommendation="keep" if i < 3 else "promote",
            )
            for i in range(5)
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="test_ws",
            validations=validations,
        )

        assert isinstance(result, ValidationSyncResult)
        assert result.total_processed == 5
        assert result.errors == []
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_sync_validations_to_continuum_min_confidence(self, adapter, continuum_memory):
        """Test batch sync filters by minimum confidence."""
        continuum_memory.add(
            id="high_conf",
            content="High confidence",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )
        continuum_memory.add(
            id="low_conf",
            content="Low confidence",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        validations = [
            KMValidationResult(memory_id="high_conf", km_confidence=0.9),
            KMValidationResult(memory_id="low_conf", km_confidence=0.5),  # Below threshold
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="test_ws",
            validations=validations,
            min_confidence=0.7,
        )

        assert result.total_processed == 2
        assert result.skipped == 1  # low_conf was skipped

        # Verify only high_conf was updated
        high_entry = continuum_memory.get("high_conf")
        low_entry = continuum_memory.get("low_conf")
        assert high_entry.metadata.get("km_validated") is True
        assert low_entry.metadata.get("km_validated") is None

    @pytest.mark.asyncio
    async def test_sync_validations_to_continuum_with_errors(self, adapter, continuum_memory):
        """Test batch sync handles errors gracefully."""
        continuum_memory.add(
            id="valid_mem",
            content="Valid memory",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        validations = [
            KMValidationResult(memory_id="valid_mem", km_confidence=0.85),
            KMValidationResult(memory_id="nonexistent", km_confidence=0.9),  # Will fail
        ]

        result = await adapter.sync_validations_to_continuum(
            workspace_id="test_ws",
            validations=validations,
        )

        # One succeeded, one was skipped (not found = skip, not error)
        assert result.total_processed == 2
        assert len(result.errors) == 0  # Not found is handled as skip, not error


class TestContinuumAdapterStats:
    """Tests for reverse sync statistics."""

    @pytest.mark.asyncio
    async def test_get_km_validated_entries(self, adapter, continuum_memory):
        """Test retrieving KM-validated entries."""
        # Create entries, some validated
        for i in range(5):
            continuum_memory.add(
                id=f"stat_mem_{i}",
                content=f"Stat memory {i}",
                tier=MemoryTier.MEDIUM,
                importance=0.5,
            )

        # Validate some entries
        for i in range(3):
            validation = KMValidationResult(
                memory_id=f"stat_mem_{i}",
                km_confidence=0.8 + (i * 0.05),
                recommendation="keep",
            )
            await adapter.update_continuum_from_km(f"stat_mem_{i}", validation)

        # Get validated entries
        validated = await adapter.get_km_validated_entries(
            limit=10,
            min_km_confidence=0.8,
        )

        assert len(validated) >= 1
        for entry in validated:
            assert entry.metadata.get("km_validated") is True
            assert entry.metadata.get("km_confidence", 0) >= 0.8

    def test_get_reverse_sync_stats(self, adapter, continuum_memory):
        """Test getting reverse sync statistics."""
        stats = adapter.get_reverse_sync_stats()

        assert "total_km_validated" in stats
        assert "km_validated_by_tier" in stats
        assert "km_supported" in stats
        assert "km_contradicted" in stats
        assert "avg_km_confidence" in stats
        assert "avg_cross_debate_utility" in stats

    @pytest.mark.asyncio
    async def test_get_reverse_sync_stats_with_data(self, adapter, continuum_memory):
        """Test stats after validations."""
        # Create and validate entries
        for i in range(5):
            continuum_memory.add(
                id=f"stat2_mem_{i}",
                content=f"Stat2 memory {i}",
                tier=MemoryTier.MEDIUM if i < 3 else MemoryTier.SLOW,
                importance=0.5,
            )
            validation = KMValidationResult(
                memory_id=f"stat2_mem_{i}",
                km_confidence=0.7 + (i * 0.05),
                cross_debate_utility=0.5 + (i * 0.1),
                was_supported=i % 2 == 0,
                recommendation="keep",
            )
            await adapter.update_continuum_from_km(f"stat2_mem_{i}", validation)

        stats = adapter.get_reverse_sync_stats()

        assert stats["total_km_validated"] >= 5
        assert stats["avg_km_confidence"] > 0
        assert stats["avg_cross_debate_utility"] > 0


class TestContinuumMemoryUpdate:
    """Tests for the new update() method on ContinuumMemory."""

    def test_update_importance(self, continuum_memory):
        """Test updating importance only."""
        continuum_memory.add(
            id="update_test_1",
            content="Original content",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        result = continuum_memory.update("update_test_1", importance=0.8)
        assert result is True

        entry = continuum_memory.get("update_test_1")
        assert entry.importance == 0.8
        assert entry.content == "Original content"  # Unchanged

    def test_update_metadata(self, continuum_memory):
        """Test updating metadata only."""
        continuum_memory.add(
            id="update_test_2",
            content="Metadata test",
            tier=MemoryTier.SLOW,
            importance=0.6,
        )

        new_metadata = {"key": "value", "number": 42}
        result = continuum_memory.update("update_test_2", metadata=new_metadata)
        assert result is True

        entry = continuum_memory.get("update_test_2")
        assert entry.metadata["key"] == "value"
        assert entry.metadata["number"] == 42

    def test_update_multiple_fields(self, continuum_memory):
        """Test updating multiple fields at once."""
        continuum_memory.add(
            id="update_test_3",
            content="Multi-field test",
            tier=MemoryTier.MEDIUM,
            importance=0.5,
        )

        result = continuum_memory.update(
            "update_test_3",
            content="Updated content",
            importance=0.9,
            metadata={"updated": True},
        )
        assert result is True

        entry = continuum_memory.get("update_test_3")
        assert entry.content == "Updated content"
        assert entry.importance == 0.9
        assert entry.metadata["updated"] is True

    def test_update_nonexistent(self, continuum_memory):
        """Test updating non-existent entry."""
        result = continuum_memory.update("nonexistent_id", importance=0.9)
        assert result is False

    def test_update_empty(self, continuum_memory):
        """Test update with no fields returns False."""
        continuum_memory.add(
            id="update_test_empty",
            content="Empty update test",
            tier=MemoryTier.FAST,
            importance=0.5,
        )

        result = continuum_memory.update("update_test_empty")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
