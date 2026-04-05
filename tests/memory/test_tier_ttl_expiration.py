import time
from datetime import datetime
from unittest.mock import patch

import pytest

from aragora.memory.continuum import (
    FAST_TIER_TTL_MINUTES,
    MEDIUM_TIER_TTL_HOURS,
    SLOW_TIER_TTL_DAYS,
    ContinuumMemory,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import MemoryTier, reset_tier_manager


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(time.time(), tz)


@pytest.fixture
def memory(tmp_path):
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=str(tmp_path / "test_tier_ttl_expiration.db"))
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


@pytest.mark.parametrize(
    ("tier", "ttl_hours"),
    [
        (MemoryTier.FAST, FAST_TIER_TTL_MINUTES / 60),
        (MemoryTier.MEDIUM, MEDIUM_TIER_TTL_HOURS),
        (MemoryTier.SLOW, SLOW_TIER_TTL_DAYS * 24),
    ],
)
def test_cleanup_respects_tier_ttl_with_mocked_time(memory, tier, ttl_hours):
    base_time = 1_700_000_000
    expired_at = datetime.fromtimestamp(base_time - (ttl_hours * 3600) - 1).isoformat()
    fresh_at = datetime.fromtimestamp(base_time - (ttl_hours * 3600) + 1).isoformat()

    with memory.connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO continuum_memory
            (id, tier, content, importance, updated_at, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (f"{tier.value}_expired", tier.value, "expired", 0.5, expired_at, expired_at, "{}"),
                (f"{tier.value}_fresh", tier.value, "fresh", 0.5, fresh_at, fresh_at, "{}"),
            ],
        )
        conn.commit()

    with (
        patch("time.time", return_value=base_time),
        patch("aragora.memory.continuum_stats.datetime", FrozenDateTime),
    ):
        result = memory.cleanup_expired_memories(tier=tier, max_age_hours=ttl_hours, archive=False)

    assert result["deleted"] == 1
    assert result["by_tier"][tier.value]["cutoff_hours"] == ttl_hours
    assert memory.get(f"{tier.value}_expired") is None
    assert memory.get(f"{tier.value}_fresh") is not None
