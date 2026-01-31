"""
Gastown Metrics Adapter — re-exports from nomic metrics.

Provides the gastown extension entry point for observability metrics.
Dashboard handlers import from here rather than reaching into
aragora.nomic.metrics directly.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


def _stub_beads_completed_count() -> int:
    """Return count of completed beads."""
    return 0


def _stub_convoy_completion_rate() -> float:
    """Return convoy completion rate."""
    return 0.0


def _stub_gupp_recovery_count() -> int:
    """Return GUPP recovery count."""
    return 0


# These metrics functions are placeholders - the actual implementations
# may be added to aragora.nomic.metrics later
def _load_metric(name: str, stub: Callable[..., Any]) -> Callable[..., Any]:
    """Load a metric function from aragora.nomic.metrics, falling back to stub."""
    try:
        mod = importlib.import_module("aragora.nomic.metrics")
        fn: Callable[..., Any] = getattr(mod, name, stub)
        return fn
    except ImportError:
        return stub


get_beads_completed_count = _load_metric("get_beads_completed_count", _stub_beads_completed_count)
get_convoy_completion_rate = _load_metric(
    "get_convoy_completion_rate", _stub_convoy_completion_rate
)
get_gupp_recovery_count = _load_metric("get_gupp_recovery_count", _stub_gupp_recovery_count)


__all__ = [
    "get_beads_completed_count",
    "get_convoy_completion_rate",
    "get_gupp_recovery_count",
]
