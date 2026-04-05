"""Tests for ContinuumMemory tier TTL expiration.

Verifies entries expire correctly based on their tier's half-life and
the retention multiplier (default 2x). Uses direct timestamp manipulation
to simulate aging entries past their TTL.
"""

from datetime import datetime, timedelta

import pytest

from aragora.memory.continuum import ContinuumMemory, reset_continuum_memory
from aragora.memory.tier_manager import (
    DEFAULT_TIER_CONFIGS,
    MemoryTier,
    TierManager,
    reset_tier_manager,
)


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_ttl.db")


@pytest.fixture
def memory(temp_db_path):
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=temp_db_path, tier_manager=TierManager())
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


def _age_entry(memory, entry_id, hours_ago):
    """Set an entry's updated_at to hours_ago in the past."""
    old_time = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
    with memory.connection() as conn:
        conn.execute(
            "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
            (old_time, entry_id),
        )


class TestTierTTLExpiration:
    """Verify entries expire according to tier half-life * retention_multiplier."""

    def test_fast_tier_expires_after_ttl(self, memory):
        memory.add("f1", "fast entry", tier=MemoryTier.FAST)
        _age_entry(memory, "f1", hours_ago=3)  # TTL = 1h * 2 = 2h
        result = memory.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] + result["deleted"] >= 1
        assert memory.get("f1") is None

    def test_fast_tier_survives_before_ttl(self, memory):
        memory.add("f2", "fresh fast entry", tier=MemoryTier.FAST)
        _age_entry(memory, "f2", hours_ago=1)  # Within 2h TTL
        memory.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert memory.get("f2") is not None

    def test_medium_tier_expires_after_ttl(self, memory):
        memory.add("m1", "medium entry", tier=MemoryTier.MEDIUM)
        _age_entry(memory, "m1", hours_ago=49)  # TTL = 24h * 2 = 48h
        result = memory.cleanup_expired_memories(tier=MemoryTier.MEDIUM)
        assert result["archived"] + result["deleted"] >= 1
        assert memory.get("m1") is None

    def test_medium_tier_survives_before_ttl(self, memory):
        memory.add("m2", "fresh medium entry", tier=MemoryTier.MEDIUM)
        _age_entry(memory, "m2", hours_ago=24)  # Within 48h TTL
        memory.cleanup_expired_memories(tier=MemoryTier.MEDIUM)
        assert memory.get("m2") is not None

    def test_slow_tier_expires_after_ttl(self, memory):
        memory.add("s1", "slow entry", tier=MemoryTier.SLOW)
        _age_entry(memory, "s1", hours_ago=337)  # TTL = 168h * 2 = 336h
        result = memory.cleanup_expired_memories(tier=MemoryTier.SLOW)
        assert result["archived"] + result["deleted"] >= 1
        assert memory.get("s1") is None

    def test_slow_tier_survives_before_ttl(self, memory):
        memory.add("s2", "fresh slow entry", tier=MemoryTier.SLOW)
        _age_entry(memory, "s2", hours_ago=100)  # Within 336h TTL
        memory.cleanup_expired_memories(tier=MemoryTier.SLOW)
        assert memory.get("s2") is not None

    def test_cleanup_all_tiers_mixed_expiration(self, memory):
        """Entries across tiers: only expired ones are removed."""
        memory.add("f", "fast", tier=MemoryTier.FAST)
        memory.add("m", "medium", tier=MemoryTier.MEDIUM)
        _age_entry(memory, "f", hours_ago=3)  # Expired (TTL 2h)
        _age_entry(memory, "m", hours_ago=10)  # Fresh (TTL 48h)
        result = memory.cleanup_expired_memories()
        assert memory.get("f") is None
        assert memory.get("m") is not None

    def test_red_line_entry_not_expired(self, memory):
        """Red-line protected entries survive cleanup even past TTL."""
        memory.add("rl", "critical entry", tier=MemoryTier.FAST)
        memory.mark_red_line("rl", reason="safety critical")
        _age_entry(memory, "rl", hours_ago=100)
        memory.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert memory.get("rl") is not None
