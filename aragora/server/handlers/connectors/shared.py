"""Shared scheduler access for connector handlers."""

from __future__ import annotations

from aragora.connectors.enterprise import SyncScheduler

_scheduler: SyncScheduler | None = None


def get_scheduler() -> SyncScheduler:
    """Get or create the global sync scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SyncScheduler(max_concurrent_syncs=5)
    return _scheduler
