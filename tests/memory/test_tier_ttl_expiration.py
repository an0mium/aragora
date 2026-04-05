"""Tests for ContinuumMemory tier TTL expiration behavior."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from aragora.memory.continuum.core import ContinuumMemory
from aragora.memory.tier_manager import DEFAULT_TIER_CONFIGS, MemoryTier


@pytest.fixture
def cms(tmp_path):
    """Create a ContinuumMemory instance with a temp database."""
    db = tmp_path / "test_cm.db"
    return ContinuumMemory(db_path=str(db))


class TestTierTTLExpiration:
    """Test that entries expire according to their tier's half-life settings."""

    def test_fast_tier_expires(self, cms):
        """Fast tier entries expire after 1h * retention_multiplier."""
        cms.add("fast1", "quick pattern", tier=MemoryTier.FAST)
        config = DEFAULT_TIER_CONFIGS[MemoryTier.FAST]
        retention_hours = config.half_life_hours * cms.hyperparams["retention_multiplier"]

        # Time-travel past the retention window
        past = datetime.now() - timedelta(hours=retention_hours + 1)
        with patch("aragora.memory.continuum_stats.datetime") as mock_dt:
            mock_dt.now.return_value = past
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # Re-add with old timestamp by updating the DB directly
        with cms.connection() as conn:
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (past.isoformat(), "fast1"),
            )

        result = cms.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] >= 1

    def test_medium_tier_expires(self, cms):
        """Medium tier entries expire after 24h * retention_multiplier."""
        cms.add("med1", "round pattern", tier=MemoryTier.MEDIUM)
        config = DEFAULT_TIER_CONFIGS[MemoryTier.MEDIUM]
        retention_hours = config.half_life_hours * cms.hyperparams["retention_multiplier"]

        past = datetime.now() - timedelta(hours=retention_hours + 1)
        with cms.connection() as conn:
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (past.isoformat(), "med1"),
            )

        result = cms.cleanup_expired_memories(tier=MemoryTier.MEDIUM)
        assert result["archived"] >= 1

    def test_slow_tier_expires(self, cms):
        """Slow tier entries expire after 168h * retention_multiplier."""
        cms.add("slow1", "cycle pattern", tier=MemoryTier.SLOW)
        config = DEFAULT_TIER_CONFIGS[MemoryTier.SLOW]
        retention_hours = config.half_life_hours * cms.hyperparams["retention_multiplier"]

        past = datetime.now() - timedelta(hours=retention_hours + 1)
        with cms.connection() as conn:
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                (past.isoformat(), "slow1"),
            )

        result = cms.cleanup_expired_memories(tier=MemoryTier.SLOW)
        assert result["archived"] >= 1

    def test_entry_not_expired_within_ttl(self, cms):
        """Entries within the retention window are NOT cleaned up."""
        cms.add("fresh1", "recent pattern", tier=MemoryTier.FAST)
        result = cms.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] == 0
        assert result["deleted"] == 0
        # Verify entry still exists
        entry = cms.get("fresh1")
        assert entry is not None

    def test_cleanup_all_tiers(self, cms):
        """cleanup_expired_memories(tier=None) processes all tiers."""
        for tier in MemoryTier:
            cms.add(f"entry_{tier.value}", f"content {tier.value}", tier=tier)
            config = DEFAULT_TIER_CONFIGS[tier]
            hours = config.half_life_hours * cms.hyperparams["retention_multiplier"] + 1
            past = datetime.now() - timedelta(hours=hours)
            with cms.connection() as conn:
                conn.execute(
                    "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
                    (past.isoformat(), f"entry_{tier.value}"),
                )

        result = cms.cleanup_expired_memories()
        assert result["archived"] == 4

    def test_red_line_entry_survives_expiration(self, cms):
        """Red-lined entries must not be cleaned up even when expired."""
        cms.add("protected1", "critical", tier=MemoryTier.FAST)
        with cms.connection() as conn:
            past = datetime.now() - timedelta(hours=100)
            conn.execute(
                "UPDATE continuum_memory SET updated_at = ?, red_line = 1 WHERE id = ?",
                (past.isoformat(), "protected1"),
            )

        result = cms.cleanup_expired_memories(tier=MemoryTier.FAST)
        assert result["archived"] == 0
        entry = cms.get("protected1")
        assert entry is not None
