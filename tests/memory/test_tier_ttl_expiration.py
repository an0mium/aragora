"""Tier-based ContinuumMemory expiration tests."""

import time
from datetime import datetime
from unittest.mock import patch

import pytest

from aragora.memory.continuum import ContinuumMemory
from aragora.memory.tier_manager import DEFAULT_TIER_CONFIGS, MemoryTier, TierManager


class _Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(time.time(), tz)


@pytest.fixture
def cms(tmp_path):
    return ContinuumMemory(db_path=tmp_path / "continuum.db", tier_manager=TierManager())


def _set_updated_at(cms, memory_id, timestamp):
    with cms.connection() as conn:
        conn.execute(
            "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
            (datetime.fromtimestamp(timestamp).isoformat(), memory_id),
        )
        conn.commit()


@pytest.mark.parametrize("tier", [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW])
def test_cleanup_expired_memories_respects_tier_ttl(cms, tier):
    now = 1_700_000_000
    ttl_hours = DEFAULT_TIER_CONFIGS[tier].half_life_hours * cms.hyperparams["retention_multiplier"]
    expired_id, fresh_id = f"{tier.value}_expired", f"{tier.value}_fresh"
    cms.add(expired_id, "expired", tier=tier)
    cms.add(fresh_id, "fresh", tier=tier)
    _set_updated_at(cms, expired_id, now - (ttl_hours + 1) * 3600)
    _set_updated_at(cms, fresh_id, now - (ttl_hours - 1) * 3600)

    with (
        patch("time.time", return_value=now),
        patch("aragora.memory.continuum_stats.datetime", _Clock),
    ):
        result = cms.cleanup_expired_memories(tier=tier)

    assert result["by_tier"][tier.value]["cutoff_hours"] == ttl_hours
    assert result["archived"] == result["deleted"] == 1
    assert cms.get(expired_id) is None
    assert cms.get(fresh_id) is not None
