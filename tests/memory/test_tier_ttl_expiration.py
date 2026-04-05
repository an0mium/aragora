"""Tests for ContinuumMemory tier TTL expiration behavior.

Verifies that entries expire correctly based on their tier's half-life
and the retention multiplier (default 2x), and that red-lined entries
are protected from expiration.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    reset_continuum_memory,
)
from aragora.memory.continuum_stats import cleanup_expired_memories
from aragora.memory.tier_manager import (
    MemoryTier,
    TierManager,
    reset_tier_manager,
)

# Retention = half_life_hours * retention_multiplier (2.0)
# FAST: 1h * 2 = 2h, MEDIUM: 24h * 2 = 48h, SLOW: 168h * 2 = 336h


@pytest.fixture
def memory(tmp_path):
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(
        db_path=str(tmp_path / "ttl_test.db"),
        tier_manager=TierManager(),
    )
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


def _backdate(memory, entry_id, hours_ago):
    """Set an entry's updated_at to N hours in the past."""
    ts = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
    with memory.connection() as conn:
        conn.execute(
            "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
            (ts, entry_id),
        )
        conn.commit()


class TestFastTierExpiration:
    def test_fresh_entries_survive(self, memory):
        memory.add("f1", "fast entry", tier=MemoryTier.FAST)
        result = cleanup_expired_memories(memory, tier=MemoryTier.FAST)
        assert result["deleted"] == 0
        assert memory.get("f1") is not None

    def test_expired_entries_removed(self, memory):
        memory.add("f1", "fast entry", tier=MemoryTier.FAST)
        _backdate(memory, "f1", hours_ago=3)  # > 2h retention
        result = cleanup_expired_memories(memory, tier=MemoryTier.FAST)
        assert result["deleted"] == 1
        assert memory.get("f1") is None

    def test_boundary_entry_survives(self, memory):
        memory.add("f1", "fast entry", tier=MemoryTier.FAST)
        _backdate(memory, "f1", hours_ago=1.5)  # < 2h retention
        result = cleanup_expired_memories(memory, tier=MemoryTier.FAST)
        assert result["deleted"] == 0


class TestMediumTierExpiration:
    def test_fresh_entries_survive(self, memory):
        memory.add("m1", "medium entry", tier=MemoryTier.MEDIUM)
        _backdate(memory, "m1", hours_ago=24)  # < 48h retention
        result = cleanup_expired_memories(memory, tier=MemoryTier.MEDIUM)
        assert result["deleted"] == 0

    def test_expired_entries_removed(self, memory):
        memory.add("m1", "medium entry", tier=MemoryTier.MEDIUM)
        _backdate(memory, "m1", hours_ago=50)  # > 48h retention
        result = cleanup_expired_memories(memory, tier=MemoryTier.MEDIUM)
        assert result["deleted"] == 1


class TestSlowTierExpiration:
    def test_expired_entries_removed(self, memory):
        memory.add("s1", "slow entry", tier=MemoryTier.SLOW)
        _backdate(memory, "s1", hours_ago=340)  # > 336h retention
        result = cleanup_expired_memories(memory, tier=MemoryTier.SLOW)
        assert result["deleted"] == 1

    def test_fresh_entries_survive(self, memory):
        memory.add("s1", "slow entry", tier=MemoryTier.SLOW)
        _backdate(memory, "s1", hours_ago=200)  # < 336h retention
        result = cleanup_expired_memories(memory, tier=MemoryTier.SLOW)
        assert result["deleted"] == 0


class TestRedLineProtection:
    def test_red_lined_entries_survive_expiration(self, memory):
        memory.add("f1", "protected", tier=MemoryTier.FAST)
        # Mark as red-lined
        with memory.connection() as conn:
            conn.execute(
                "UPDATE continuum_memory SET red_line = 1 WHERE id = ?",
                ("f1",),
            )
            conn.commit()
        _backdate(memory, "f1", hours_ago=100)
        result = cleanup_expired_memories(memory, tier=MemoryTier.FAST)
        assert result["deleted"] == 0
        assert memory.get("f1") is not None


class TestAllTiersCleanup:
    def test_only_expired_tiers_cleaned(self, memory):
        memory.add("f1", "fast", tier=MemoryTier.FAST)
        memory.add("m1", "medium", tier=MemoryTier.MEDIUM)
        # Expire fast but not medium
        _backdate(memory, "f1", hours_ago=3)
        _backdate(memory, "m1", hours_ago=10)
        result = cleanup_expired_memories(memory)
        assert result["by_tier"]["fast"]["deleted"] == 1
        assert result["by_tier"]["medium"]["deleted"] == 0
