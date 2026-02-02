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

# Re-export GauntletHandler for backward compatibility
from .handler import GauntletHandler

# Re-export storage utilities that may be used externally
from .storage import (
    MAX_GAUNTLET_RUNS_IN_MEMORY,
    _GAUNTLET_COMPLETED_TTL,
    _GAUNTLET_MAX_AGE_SECONDS,
    _gauntlet_runs,
    _get_storage,
    _quota_lock,
    create_tracked_task,
    get_gauntlet_runs,
    get_quota_lock,
    is_durable_queue_enabled,
    recover_stale_gauntlet_runs,
    set_gauntlet_broadcast_fn,
)

__all__ = [
    # Main handler class
    "GauntletHandler",
    # Storage utilities - public API
    "create_tracked_task",
    "set_gauntlet_broadcast_fn",
    "recover_stale_gauntlet_runs",
    "get_gauntlet_runs",
    "get_quota_lock",
    "is_durable_queue_enabled",
    "MAX_GAUNTLET_RUNS_IN_MEMORY",
    # Private storage utilities - for backward compatibility with tests
    "_gauntlet_runs",
    "_get_storage",
    "_quota_lock",
    "_GAUNTLET_COMPLETED_TTL",
    "_GAUNTLET_MAX_AGE_SECONDS",
]
