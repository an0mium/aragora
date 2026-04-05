"""
Tests for ContinuumMemory TTL expiration per tier.

Verifies that entries expire correctly based on their tier's half-life
and the retention multiplier (default 2x half-life).

Tier TTLs (with default 2x multiplier):
  - Fast:   1h half-life  → 2h retention
  - Medium: 24h half-life → 48h retention
  - Slow:   168h (7d)     → 336h (14d) retention
"""

from datetime import datetime, timedelta

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import (
    MemoryTier,
    TierManager,
    reset_tier_manager,
)


@pytest.fixture
def memory(tmp_path):
    """Create an isolated ContinuumMemory instance."""
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(
        db_path=str(tmp_path / "ttl_test.db"),
        tier_manager=TierManager(),
    )
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


def _backdate(cms: ContinuumMemory, entry_id: str, hours: float) -> None:
    """Set an entry's updated_at to `hours` ago from now."""
    old_time = (datetime.now() - timedelta(hours=hours)).isoformat()
    with cms.connection() as conn:
        conn.execute(
            "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
            (old_time, entry_id),
        )


class TestFastTierExpiration:
    """Fast tier: 1h half-life → 2h retention."""

    def test_not_expired_within_ttl(self, memory):
        memory.add("f1", "fast entry", tier=MemoryTier.FAST)
        _backdate(memory, "f1", hours=1.5)  # within 2h
        result = memory.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] == 0

    def test_expired_beyond_ttl(self, memory):
        memory.add("f1", "fast entry", tier=MemoryTier.FAST)
        _backdate(memory, "f1", hours=3)  # beyond 2h
        result = memory.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] == 1


class TestMediumTierExpiration:
    """Medium tier: 24h half-life → 48h retention."""

    def test_not_expired_within_ttl(self, memory):
        memory.add("m1", "medium entry", tier=MemoryTier.MEDIUM)
        _backdate(memory, "m1", hours=40)  # within 48h
        result = memory.cleanup_expired_memories(tier=MemoryTier.MEDIUM)
        assert result["archived"] == 0

    def test_expired_beyond_ttl(self, memory):
        memory.add("m1", "medium entry", tier=MemoryTier.MEDIUM)
        _backdate(memory, "m1", hours=50)  # beyond 48h
        result = memory.cleanup_expired_memories(tier=MemoryTier.MEDIUM)
        assert result["archived"] == 1


class TestSlowTierExpiration:
    """Slow tier: 168h half-life → 336h retention."""

    def test_not_expired_within_ttl(self, memory):
        memory.add("s1", "slow entry", tier=MemoryTier.SLOW)
        _backdate(memory, "s1", hours=300)  # within 336h
        result = memory.cleanup_expired_memories(tier=MemoryTier.SLOW)
        assert result["archived"] == 0

    def test_expired_beyond_ttl(self, memory):
        memory.add("s1", "slow entry", tier=MemoryTier.SLOW)
        _backdate(memory, "s1", hours=350)  # beyond 336h
        result = memory.cleanup_expired_memories(tier=MemoryTier.SLOW)
        assert result["archived"] == 1


class TestMixedTierCleanup:
    """Cleanup across multiple tiers in one call."""

    def test_only_expired_tiers_cleaned(self, memory):
        memory.add("f1", "fast", tier=MemoryTier.FAST)
        memory.add("m1", "medium", tier=MemoryTier.MEDIUM)
        # Fast expired (3h > 2h), medium not (3h < 48h)
        _backdate(memory, "f1", hours=3)
        _backdate(memory, "m1", hours=3)
        result = memory.cleanup_expired_memories()
        assert result["archived"] == 1
        # Fast entry gone, medium still present
        assert memory.get("f1") is None
        assert memory.get("m1") is not None
