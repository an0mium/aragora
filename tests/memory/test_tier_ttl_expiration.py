from datetime import datetime
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
    cms = ContinuumMemory(db_path=str(tmp_path / "tier_ttl.db"), tier_manager=TierManager())
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
def test_cleanup_expired_memories_respects_tier_ttl(memory, tier, ttl_seconds):
    with patch("time.time", return_value=time.time()) as mocked_time:
        fresh_id = f"{tier.value}_fresh"
        expired_id = f"{tier.value}_expired"
        memory.add(fresh_id, "fresh", tier=tier)
        memory.add(expired_id, "expired", tier=tier)
        recent_at = datetime.fromtimestamp(mocked_time.return_value - ttl_seconds + 5).isoformat()
        expired_at = datetime.fromtimestamp(mocked_time.return_value - ttl_seconds - 5).isoformat()

    with memory.connection() as conn:
        conn.executemany(
            "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
            [(recent_at, fresh_id), (expired_at, expired_id)],
        )
        conn.commit()

    result = memory.cleanup_expired_memories(
        tier=tier, archive=False, max_age_hours=ttl_seconds / 3600
    )
    assert result["deleted"] == 1
    assert memory.get(fresh_id) is not None
    assert memory.get(expired_id) is None
