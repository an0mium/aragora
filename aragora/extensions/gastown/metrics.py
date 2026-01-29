"""
Gastown Metrics Adapter — re-exports from nomic metrics.

Provides the gastown extension entry point for observability metrics.
Dashboard handlers import from here rather than reaching into
aragora.nomic.metrics directly.
"""

from __future__ import annotations


# These metrics functions are placeholders - the actual implementations
# may be added to aragora.nomic.metrics later
try:
    from aragora.nomic.metrics import (  # type: ignore[attr-defined]
        get_beads_completed_count,
        get_convoy_completion_rate,
        get_gupp_recovery_count,
    )
except ImportError:
    # Provide stub implementations if not available
    def get_beads_completed_count() -> int:
        """Return count of completed beads."""
        return 0

    def get_convoy_completion_rate() -> float:
        """Return convoy completion rate."""
        return 0.0

    def get_gupp_recovery_count() -> int:
        """Return GUPP recovery count."""
        return 0


__all__ = [
    "get_beads_completed_count",
    "get_convoy_completion_rate",
    "get_gupp_recovery_count",
]
