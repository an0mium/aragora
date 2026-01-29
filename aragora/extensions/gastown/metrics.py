"""
Gastown Metrics Adapter — re-exports from nomic metrics.

Provides the gastown extension entry point for observability metrics.
Dashboard handlers import from here rather than reaching into
aragora.nomic.metrics directly.
"""

from __future__ import annotations

from aragora.nomic.metrics import (
    get_beads_completed_count,
    get_convoy_completion_rate,
    get_gupp_recovery_count,
)

__all__ = [
    "get_beads_completed_count",
    "get_convoy_completion_rate",
    "get_gupp_recovery_count",
]
