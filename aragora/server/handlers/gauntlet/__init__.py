"""
Gauntlet endpoint handlers for adversarial stress-testing.

Endpoints:
- POST /api/gauntlet/run - Start a gauntlet stress-test
- GET /api/gauntlet/{id} - Get gauntlet status/results
- GET /api/gauntlet/{id}/receipt - Get decision receipt
- GET /api/gauntlet/{id}/heatmap - Get risk heatmap
- GET /api/gauntlet/personas - List available personas
- GET /api/gauntlet/results - List recent results with pagination
- GET /api/gauntlet/{id}/compare/{id2} - Compare two gauntlet runs

This module provides backward-compatible imports for the refactored gauntlet handler.
The implementation has been split into:
- handler.py: Main GauntletHandler class
- storage.py: In-memory storage and helper functions
- runner.py: Gauntlet execution methods
- receipts.py: Receipt handling methods
- heatmap.py: Heatmap generation methods
- results.py: Results listing, comparison, and export methods
"""

from __future__ import annotations

from typing import Any

# Re-export GauntletHandler for backward compatibility
from .handler import GauntletHandler

# Import storage module for dynamic attribute access
from . import storage as _storage_module

# Re-export storage utilities that may be used externally
from .storage import (
    MAX_GAUNTLET_RUNS_IN_MEMORY,
    _cleanup_gauntlet_runs,
    _GAUNTLET_COMPLETED_TTL,
    _GAUNTLET_MAX_AGE_SECONDS,
    _gauntlet_runs,
    _get_storage,
    _quota_lock,
    create_tracked_task,
    get_gauntlet_broadcast_fn,
    get_gauntlet_runs,
    get_quota_lock,
    is_durable_queue_enabled,
    recover_stale_gauntlet_runs,
    set_gauntlet_broadcast_fn,
)

# Mutable attributes that need dynamic lookup from storage module
_DYNAMIC_ATTRS = {"_gauntlet_broadcast_fn"}


def __getattr__(name: str) -> Any:
    """Dynamic attribute access for mutable module-level variables.

    This allows backward-compatible access to mutable variables like
    _gauntlet_broadcast_fn which are modified by set_gauntlet_broadcast_fn().
    """
    if name in _DYNAMIC_ATTRS:
        return getattr(_storage_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main handler class
    "GauntletHandler",
    # Storage utilities - public API
    "create_tracked_task",
    "set_gauntlet_broadcast_fn",
    "get_gauntlet_broadcast_fn",
    "recover_stale_gauntlet_runs",
    "get_gauntlet_runs",
    "get_quota_lock",
    "is_durable_queue_enabled",
    "MAX_GAUNTLET_RUNS_IN_MEMORY",
    # Private storage utilities - for backward compatibility with tests
    "_gauntlet_runs",
    "_gauntlet_broadcast_fn",
    "_get_storage",
    "_quota_lock",
    "_cleanup_gauntlet_runs",
    "_GAUNTLET_COMPLETED_TTL",
    "_GAUNTLET_MAX_AGE_SECONDS",
]
