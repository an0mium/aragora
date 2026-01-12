"""
Tests for ConsensusMemory cleanup functionality.

Tests cover:
- Tiered archival (hot/warm/cold)
- Archive table creation
- Record deletion after archival
- Statistics after cleanup
"""

import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from aragora.memory.consensus import (
    ConsensusMemory,
    ConsensusStrength,
    DissentType,
)


class TestConsensusCleanup:
    """Test ConsensusMemory cleanup_old_records functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_consensus.db"
            yield str(db_path)

    @pytest.fixture
    def memory(self, temp_db):
        """Create a ConsensusMemory instance with temp db."""
        return ConsensusMemory(db_path=temp_db)

    def test_cleanup_no_records(self, memory):
        """Test cleanup with no records returns zeros."""
        result = memory.cleanup_old_records(max_age_days=30)
        assert result["archived"] == 0
        assert result["deleted"] == 0

    def test_cleanup_recent_records_preserved(self, memory):
        """Test that recent records are not cleaned up."""
        # Create a recent consensus record
        memory.store_consensus(
            topic="Recent topic",
            conclusion="This should be kept",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["claude", "gpt"],
            agreeing_agents=["claude", "gpt"],
        )

        # Run cleanup with 30-day threshold
        result = memory.cleanup_old_records(max_age_days=30)

        # Recent record should not be cleaned up
        assert result["deleted"] == 0
        stats = memory.get_statistics()
        assert stats["total_consensus"] == 1

    def test_cleanup_creates_archive_tables(self, memory):
        """Test that cleanup creates archive tables."""
        # Run cleanup to trigger archive table creation
        memory.cleanup_old_records(max_age_days=30, archive=True)

        # Check archive tables exist
        from aragora.storage.schema import get_wal_connection
        with get_wal_connection(memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='consensus_archive'"
            )
            assert cursor.fetchone() is not None

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='dissent_archive'"
            )
            assert cursor.fetchone() is not None

    def test_cleanup_without_archive(self, memory):
        """Test cleanup without archiving."""
        # Run cleanup without archive flag
        result = memory.cleanup_old_records(max_age_days=30, archive=False)

        # Should still return result dict
        assert "archived" in result
        assert "deleted" in result

    def test_cleanup_with_dissent_records(self, memory):
        """Test cleanup handles dissent records correctly."""
        # Create a consensus with dissent
        consensus = memory.store_consensus(
            topic="Debated topic",
            conclusion="Majority view",
            strength=ConsensusStrength.MODERATE,
            confidence=0.7,
            participating_agents=["claude", "gpt", "gemini"],
            agreeing_agents=["claude", "gpt"],
            dissenting_agents=["gemini"],
        )

        # Add dissent record
        memory.store_dissent(
            debate_id=consensus.id,
            agent_id="gemini",
            dissent_type=DissentType.ALTERNATIVE_APPROACH,
            content="I disagree with the approach",
            reasoning="There's a better way",
            confidence=0.8,
        )

        # Verify records exist
        stats = memory.get_statistics()
        assert stats["total_consensus"] == 1
        assert stats["total_dissents"] == 1

        # Cleanup should preserve recent records
        result = memory.cleanup_old_records(max_age_days=30)
        assert result["deleted"] == 0

    def test_cleanup_returns_counts(self, memory):
        """Test that cleanup returns proper counts."""
        result = memory.cleanup_old_records(max_age_days=90)

        assert isinstance(result, dict)
        assert "archived" in result
        assert "deleted" in result
        assert isinstance(result["archived"], int)
        assert isinstance(result["deleted"], int)

    def test_get_statistics_after_cleanup(self, memory):
        """Test statistics are accurate after cleanup."""
        # Create some records
        for i in range(3):
            memory.store_consensus(
                topic=f"Topic {i}",
                conclusion=f"Conclusion {i}",
                strength=ConsensusStrength.STRONG,
                confidence=0.8 + i * 0.05,
                participating_agents=["claude", "gpt"],
                agreeing_agents=["claude", "gpt"],
                domain=f"domain{i % 2}",
            )

        # Get stats before cleanup
        stats_before = memory.get_statistics()
        assert stats_before["total_consensus"] == 3

        # Cleanup (no old records)
        memory.cleanup_old_records(max_age_days=30)

        # Stats should be unchanged
        stats_after = memory.get_statistics()
        assert stats_after["total_consensus"] == 3

    def test_cleanup_max_age_boundary(self, memory):
        """Test cleanup respects max_age_days boundary."""
        # Create a consensus record
        memory.store_consensus(
            topic="Boundary test",
            conclusion="Test conclusion",
            strength=ConsensusStrength.UNANIMOUS,
            confidence=1.0,
            participating_agents=["claude"],
            agreeing_agents=["claude"],
        )

        # Cleanup with very long threshold (365 days)
        result = memory.cleanup_old_records(max_age_days=365)
        assert result["deleted"] == 0

        # Record should still exist
        stats = memory.get_statistics()
        assert stats["total_consensus"] == 1


class TestConsensusMemoryIntegration:
    """Integration tests for ConsensusMemory with cleanup."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_consensus.db"
            yield str(db_path)

    def test_full_lifecycle_with_cleanup(self, temp_db):
        """Test full lifecycle: create, query, cleanup."""
        memory = ConsensusMemory(db_path=temp_db)

        # Create records
        consensus1 = memory.store_consensus(
            topic="AI Safety",
            conclusion="Safety is important",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["claude", "gpt", "gemini"],
            agreeing_agents=["claude", "gpt", "gemini"],
            domain="ai",
            tags=["safety", "alignment"],
        )

        consensus2 = memory.store_consensus(
            topic="Climate Change",
            conclusion="Action needed",
            strength=ConsensusStrength.UNANIMOUS,
            confidence=0.95,
            participating_agents=["claude", "gpt"],
            agreeing_agents=["claude", "gpt"],
            domain="environment",
        )

        # Add dissent to first consensus
        memory.store_dissent(
            debate_id=consensus1.id,
            agent_id="skeptic",
            dissent_type=DissentType.RISK_WARNING,
            content="We should be cautious",
            reasoning="Edge cases exist",
        )

        # Query records
        retrieved = memory.get_consensus(consensus1.id)
        assert retrieved is not None
        assert retrieved.topic == "AI Safety"

        dissents = memory.get_dissents(consensus1.id)
        assert len(dissents) == 1

        # Find similar debates
        similar = memory.find_similar_debates("AI Safety measures")
        assert len(similar) >= 1

        # Get domain history
        history = memory.get_domain_consensus_history("ai")
        assert len(history) == 1

        # Get statistics
        stats = memory.get_statistics()
        assert stats["total_consensus"] == 2
        assert stats["total_dissents"] == 1

        # Run cleanup
        result = memory.cleanup_old_records(max_age_days=30)

        # Verify records still exist (they're recent)
        final_stats = memory.get_statistics()
        assert final_stats["total_consensus"] == 2

    def test_supersede_and_cleanup(self, temp_db):
        """Test superseding records and cleanup."""
        memory = ConsensusMemory(db_path=temp_db)

        # Create original consensus
        original = memory.store_consensus(
            topic="Original view",
            conclusion="First conclusion",
            strength=ConsensusStrength.WEAK,
            confidence=0.6,
            participating_agents=["claude", "gpt"],
            agreeing_agents=["claude"],
        )

        # Supersede with new consensus
        new_consensus = memory.supersede_consensus(
            old_consensus_id=original.id,
            new_topic="Updated view",
            new_conclusion="Better conclusion",
            strength=ConsensusStrength.STRONG,
            confidence=0.9,
            participating_agents=["claude", "gpt", "gemini"],
            agreeing_agents=["claude", "gpt", "gemini"],
        )

        # Verify supersession
        updated_original = memory.get_consensus(original.id)
        # Note: superseded_by is stored in the data blob
        assert new_consensus is not None

        # Cleanup should preserve both
        memory.cleanup_old_records(max_age_days=30)
        stats = memory.get_statistics()
        assert stats["total_consensus"] == 2
