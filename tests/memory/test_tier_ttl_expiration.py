"""Tier TTL expiration coverage for ContinuumMemory cleanup."""

import time
from datetime import datetime
from unittest.mock import patch

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    FAST_TIER_TTL_MINUTES,
    MEDIUM_TIER_TTL_HOURS,
    SLOW_TIER_TTL_DAYS,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import MemoryTier, TierManager, reset_tier_manager


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(time.time(), tz=tz)


@pytest.fixture
def memory(tmp_path):
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=str(tmp_path / "tier_ttl.db"), tier_manager=TierManager())
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


def test_cleanup_expires_entries_after_each_tier_ttl(memory):
    clock = {"now": 1_700_000_000.0}
    ttl_hours = {
        MemoryTier.FAST: FAST_TIER_TTL_MINUTES / 60,
        MemoryTier.MEDIUM: MEDIUM_TIER_TTL_HOURS,
        MemoryTier.SLOW: SLOW_TIER_TTL_DAYS * 24,
    }

    with (
        patch("time.time", side_effect=lambda: clock["now"]),
        patch("aragora.memory.continuum.crud.datetime", _FrozenDateTime),
        patch("aragora.memory.continuum_stats.datetime", _FrozenDateTime),
    ):
        memory.add("fast-entry", "fast", tier=MemoryTier.FAST)
        memory.add("medium-entry", "medium", tier=MemoryTier.MEDIUM)
        memory.add("slow-entry", "slow", tier=MemoryTier.SLOW)

        assert memory.get_fast_tier_stats()["ttl_minutes"] == FAST_TIER_TTL_MINUTES
        assert memory.get_medium_tier_stats()["ttl_hours"] == MEDIUM_TIER_TTL_HOURS
        assert memory.get_slow_tier_stats()["ttl_days"] == SLOW_TIER_TTL_DAYS

        clock["now"] += FAST_TIER_TTL_MINUTES * 60 + 1
        result = memory.cleanup_expired_memories(
            tier=MemoryTier.FAST, max_age_hours=ttl_hours[MemoryTier.FAST], archive=False
        )
        assert result["by_tier"]["fast"]["deleted"] == 1
        assert memory.get("fast-entry") is None
        assert memory.get("medium-entry") is not None
        assert memory.get("slow-entry") is not None

        clock["now"] = 1_700_000_000.0 + MEDIUM_TIER_TTL_HOURS * 3600 + 1
        result = memory.cleanup_expired_memories(
            tier=MemoryTier.MEDIUM, max_age_hours=ttl_hours[MemoryTier.MEDIUM], archive=False
        )
        assert result["by_tier"]["medium"]["deleted"] == 1
        assert memory.get("medium-entry") is None
        assert memory.get("slow-entry") is not None

        clock["now"] = 1_700_000_000.0 + SLOW_TIER_TTL_DAYS * 86400 + 1
        result = memory.cleanup_expired_memories(
            tier=MemoryTier.SLOW, max_age_hours=ttl_hours[MemoryTier.SLOW], archive=False
        )
        assert result["by_tier"]["slow"]["deleted"] == 1
        assert memory.get("slow-entry") is None
