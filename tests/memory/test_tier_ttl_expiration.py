from datetime import datetime, timedelta
import time
from unittest.mock import patch

import pytest

from aragora.memory.continuum import (
    FAST_TIER_TTL_MINUTES,
    MEDIUM_TIER_TTL_HOURS,
    SLOW_TIER_TTL_DAYS,
    ContinuumMemory,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import MemoryTier, TierManager, reset_tier_manager


@pytest.fixture
def memory(tmp_path):
    reset_tier_manager()
    reset_continuum_memory()
    cms = ContinuumMemory(db_path=tmp_path / "tier_ttl.db", tier_manager=TierManager())
    yield cms
    reset_tier_manager()
    reset_continuum_memory()


@pytest.mark.parametrize(
    ("tier", "ttl_seconds"),
    [
        (MemoryTier.FAST, FAST_TIER_TTL_MINUTES * 60),
        (MemoryTier.MEDIUM, MEDIUM_TIER_TTL_HOURS * 3600),
        (MemoryTier.SLOW, SLOW_TIER_TTL_DAYS * 86400),
    ],
)
def test_cleanup_expires_only_entries_past_tier_ttl(memory, tier, ttl_seconds):
    with (
        patch("time.time", return_value=1_744_232_400),
        patch("aragora.memory.continuum_stats.datetime") as mocked_datetime,
    ):
        now = datetime.fromtimestamp(time.time())
        mocked_datetime.now.return_value = now
        expired_at = (now - timedelta(seconds=ttl_seconds + 30)).isoformat()
        fresh_at = (now - timedelta(seconds=ttl_seconds - 30)).isoformat()

        with memory.connection() as conn:
            conn.executemany(
                """INSERT INTO continuum_memory
                   (id, tier, content, importance, updated_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (f"{tier.value}_expired", tier.value, "expired", 0.5, expired_at, expired_at),
                    (f"{tier.value}_fresh", tier.value, "fresh", 0.5, fresh_at, fresh_at),
                ],
            )
            conn.commit()

        result = memory.cleanup_expired_memories(
            tier=tier, archive=False, max_age_hours=ttl_seconds / 3600
        )

    assert result["by_tier"][tier.value]["deleted"] == 1
    assert memory.get(f"{tier.value}_expired") is None
    assert memory.get(f"{tier.value}_fresh") is not None
