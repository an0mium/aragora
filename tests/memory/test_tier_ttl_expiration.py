"""Tests for ContinuumMemory tier-based TTL expiration.

Mocks datetime.now() to verify entries expire correctly per tier half-life
and retention multiplier settings.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

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


RETENTION_MULTIPLIER = 2.0  # default from base.py


class TestTierTTLExpiration:
    """Verify entries expire based on tier half-life * retention multiplier."""

    def test_fast_tier_expires(self, memory):
        """Fast tier: half_life=1h, retention=2h. Entry should expire after 2h."""
        memory.add("f1", "fast entry", tier=MemoryTier.FAST, importance=0.5)

        # Just under retention window — should NOT expire
        cutoff_safe = datetime.now() + timedelta(hours=1.9)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_safe
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] == 0

        # Beyond retention window — should expire
        cutoff_expired = datetime.now() + timedelta(hours=2.1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_expired
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] == 1

    def test_medium_tier_expires(self, memory):
        """Medium tier: half_life=24h, retention=48h."""
        memory.add("m1", "medium entry", tier=MemoryTier.MEDIUM, importance=0.6)

        cutoff_safe = datetime.now() + timedelta(hours=47)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_safe
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.MEDIUM)
        assert result["archived"] == 0

        cutoff_expired = datetime.now() + timedelta(hours=49)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_expired
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.MEDIUM)
        assert result["archived"] == 1

    def test_slow_tier_expires(self, memory):
        """Slow tier: half_life=168h (7d), retention=336h (14d)."""
        memory.add("s1", "slow entry", tier=MemoryTier.SLOW, importance=0.7)

        cutoff_safe = datetime.now() + timedelta(days=13)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_safe
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.SLOW)
        assert result["archived"] == 0

        cutoff_expired = datetime.now() + timedelta(days=15)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_expired
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.SLOW)
        assert result["archived"] == 1

    def test_glacial_tier_expires(self, memory):
        """Glacial tier: half_life=720h (30d), retention=1440h (60d)."""
        memory.add("g1", "glacial entry", tier=MemoryTier.GLACIAL, importance=0.9)

        cutoff_safe = datetime.now() + timedelta(days=59)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_safe
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.GLACIAL)
        assert result["archived"] == 0

        cutoff_expired = datetime.now() + timedelta(days=61)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = cutoff_expired
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories(tier=MemoryTier.GLACIAL)
        assert result["archived"] == 1

    def test_mixed_tiers_selective_expiration(self, memory):
        """Only entries in tiers past their retention window expire."""
        memory.add("f1", "fast", tier=MemoryTier.FAST, importance=0.5)
        memory.add("m1", "medium", tier=MemoryTier.MEDIUM, importance=0.6)

        # Advance 3h — fast should expire (2h retention), medium should not (48h)
        future = datetime.now() + timedelta(hours=3)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = memory.cleanup_expired_memories()

        assert result["by_tier"].get("fast", {}).get("archived", 0) == 1
        assert result["by_tier"].get("medium", {}).get("archived", 0) == 0

    def test_retention_values_match_config(self):
        """Verify half-life values used in tests match DEFAULT_TIER_CONFIGS."""
        assert DEFAULT_TIER_CONFIGS[MemoryTier.FAST].half_life_hours == 1
        assert DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM].half_life_hours == 24
        assert DEFAULT_TIER_CONFIGS[MemoryTier.SLOW].half_life_hours == 168
        assert DEFAULT_TIER_CONFIGS[MemoryTier.GLACIAL].half_life_hours == 720
