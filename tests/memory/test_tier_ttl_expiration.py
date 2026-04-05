"""Tests for TTL expiration across ContinuumMemory tiers.

Mocks datetime.now() to simulate time passing and verifies that
cleanup_expired_memories correctly expires entries per tier based on
half_life_hours * retention_multiplier (default 2x).

Retention windows:
  Fast:   1h  * 2 = 2 hours
  Medium: 24h * 2 = 48 hours
  Slow:   168h * 2 = 336 hours (14 days)
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import (
    DEFAULT_TIER_CONFIGS,
    MemoryTier,
    TierManager,
    reset_tier_manager,
)


@pytest.fixture
def tier_manager():
    return TierManager()


@pytest.fixture
def memory(tmp_path, tier_manager):
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=str(tmp_path / "ttl_test.db"), tier_manager=tier_manager)
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


def _retention_hours(tier: MemoryTier) -> float:
    """Return the retention window for a tier (half_life * default multiplier 2)."""
    return DEFAULT_TIER_CONFIGS[tier].half_life_hours * 2.0


@pytest.mark.parametrize(
    "tier",
    [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW],
    ids=["fast", "medium", "slow"],
)
class TestTierTTLExpiration:
    """Verify cleanup_expired_memories respects per-tier retention windows."""

    def test_entry_not_expired_before_ttl(self, memory, tier):
        """Entry created just inside the retention window should survive cleanup."""
        memory.add(f"{tier.value}_entry", f"content for {tier.value}", tier=tier)
        # Advance time to just before the retention cutoff
        future = datetime.now() + timedelta(hours=_retention_hours(tier) - 0.1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=tier, archive=False)
        assert result["deleted"] == 0

    def test_entry_expired_after_ttl(self, memory, tier):
        """Entry older than the retention window should be removed by cleanup."""
        memory.add(f"{tier.value}_old", f"old content for {tier.value}", tier=tier)
        # Advance time past the retention cutoff
        future = datetime.now() + timedelta(hours=_retention_hours(tier) + 1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=tier, archive=False)
        assert result["deleted"] >= 1


class TestMixedTierExpiration:
    """Cross-tier expiration: fast expires first while slower tiers survive."""

    def test_fast_expires_while_slow_survives(self, memory):
        """After fast retention elapses, fast entries expire but slow survive."""
        memory.add("fast_item", "fast content", tier=MemoryTier.FAST)
        memory.add("slow_item", "slow content", tier=MemoryTier.SLOW)

        future = datetime.now() + timedelta(hours=_retention_hours(MemoryTier.FAST) + 1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            fast_result = memory.cleanup_expired_memories(tier=MemoryTier.FAST, archive=False)
            slow_result = memory.cleanup_expired_memories(tier=MemoryTier.SLOW, archive=False)

        assert fast_result["deleted"] >= 1
        assert slow_result["deleted"] == 0
