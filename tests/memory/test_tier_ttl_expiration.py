"""Tests for TTL expiration behavior across ContinuumMemory tiers."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from aragora.memory.continuum.core import ContinuumMemory
from aragora.memory.tier_manager import MemoryTier, DEFAULT_TIER_CONFIGS


@pytest.fixture
def cms(tmp_path):
    """Create a ContinuumMemory instance with a temporary database."""
    db = tmp_path / "test_ttl.db"
    return ContinuumMemory(db_path=str(db))


class TestTierTTLExpiration:
    """Verify entries expire correctly based on their tier's half-life."""

    def test_fast_tier_expires(self, cms):
        """Fast tier entries expire after half_life * retention_multiplier."""
        cms.add("fast1", "fast content", tier=MemoryTier.FAST)
        config = DEFAULT_TIER_CONFIGS[MemoryTier.FAST]
        multiplier = cms.hyperparams["retention_multiplier"]
        cutoff_hours = config.half_life_hours * multiplier

        future = datetime.now() + timedelta(hours=cutoff_hours + 1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = cms.cleanup_expired_memories(tier=MemoryTier.FAST)

        assert result["archived"] >= 1

    def test_medium_tier_expires(self, cms):
        """Medium tier entries expire after half_life * retention_multiplier."""
        cms.add("med1", "medium content", tier=MemoryTier.MEDIUM)
        config = DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM]
        multiplier = cms.hyperparams["retention_multiplier"]
        cutoff_hours = config.half_life_hours * multiplier

        future = datetime.now() + timedelta(hours=cutoff_hours + 1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = cms.cleanup_expired_memories(tier=MemoryTier.MEDIUM)

        assert result["archived"] >= 1

    def test_slow_tier_expires(self, cms):
        """Slow tier entries expire after half_life * retention_multiplier."""
        cms.add("slow1", "slow content", tier=MemoryTier.SLOW)
        config = DEFAULT_TIER_CONFIGS[MemoryTier.SLOW]
        multiplier = cms.hyperparams["retention_multiplier"]
        cutoff_hours = config.half_life_hours * multiplier

        future = datetime.now() + timedelta(hours=cutoff_hours + 1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = cms.cleanup_expired_memories(tier=MemoryTier.SLOW)

        assert result["archived"] >= 1

    def test_entry_not_expired_before_ttl(self, cms):
        """Entries should NOT expire before their TTL elapses."""
        cms.add("fresh1", "fresh content", tier=MemoryTier.FAST)
        result = cms.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] == 0
        assert result["deleted"] == 0

    def test_tier_ttl_ordering(self):
        """Faster tiers should have shorter half-lives than slower tiers."""
        fast_hl = DEFAULT_TIER_CONFIGS[MemoryTier.FAST].half_life_hours
        med_hl = DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM].half_life_hours
        slow_hl = DEFAULT_TIER_CONFIGS[MemoryTier.SLOW].half_life_hours
        glacial_hl = DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL].half_life_hours
        assert fast_hl < med_hl < slow_hl < glacial_hl

    def test_cleanup_delete_mode(self, cms):
        """With archive=False, entries are deleted not archived."""
        cms.add("del1", "delete me", tier=MemoryTier.FAST)
        config = DEFAULT_TIER_CONFIGS[MemoryTier.FAST]
        multiplier = cms.hyperparams["retention_multiplier"]
        cutoff_hours = config.half_life_hours * multiplier

        future = datetime.now() + timedelta(hours=cutoff_hours + 1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = cms.cleanup_expired_memories(tier=MemoryTier.FAST, archive=False)

        assert result["deleted"] >= 1
        assert result["archived"] == 0
