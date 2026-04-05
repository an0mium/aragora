"""Tests for ContinuumMemory TTL expiration by tier.

Verifies that cleanup_expired_memories correctly expires entries
based on tier half-life * retention_multiplier thresholds.
"""

from datetime import datetime, timedelta

import pytest

from aragora.memory.continuum import (
    ContinuumMemory,
    get_continuum_memory,
    reset_continuum_memory,
)
from aragora.memory.tier_manager import (
    DEFAULT_TIER_CONFIGS,
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


def _backdate(memory, entry_id, hours_ago):
    """Set an entry's updated_at to `hours_ago` hours in the past."""
    past = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
    with memory.connection() as conn:
        conn.execute(
            "UPDATE continuum_memory SET updated_at = ? WHERE id = ?",
            (past, entry_id),
        )
        conn.commit()


@pytest.mark.parametrize(
    "tier,hours_fresh,hours_expired",
    [
        # fast: half_life=1h, retention=2x -> expires after 2h
        (MemoryTier.FAST, 1, 3),
        # medium: half_life=24h, retention=2x -> expires after 48h
        (MemoryTier.MEDIUM, 24, 49),
        # slow: half_life=168h, retention=2x -> expires after 336h
        (MemoryTier.SLOW, 168, 337),
    ],
)
def test_tier_ttl_expiration(memory, tier, hours_fresh, hours_expired):
    """Entries older than half_life * retention_multiplier are cleaned up."""
    memory.add("fresh", "fresh entry", tier=tier, importance=0.5)
    memory.add("expired", "old entry", tier=tier, importance=0.5)

    _backdate(memory, "expired", hours_expired)
    _backdate(memory, "fresh", hours_fresh)

    result = memory.cleanup_expired_memories(tier=tier, archive=False)

    assert result["deleted"] == 1
    # Fresh entry survives
    assert memory.get("fresh") is not None
    # Expired entry is gone
    assert memory.get("expired") is None


def test_cleanup_archives_by_default(memory):
    """Cleanup archives expired entries when archive=True (default)."""
    memory.add("old", "will expire", tier=MemoryTier.FAST, importance=0.5)
    _backdate(memory, "old", 3)

    result = memory.cleanup_expired_memories(tier=MemoryTier.FAST, archive=True)

    assert result["archived"] >= 1
    assert result["deleted"] >= 1
    assert memory.get("old") is None
