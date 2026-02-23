"""
Shared helpers for storing and retrieving decision results.

Centralizes DecisionResultStore usage with an in-memory fallback so
multiple handlers/workers can reuse consistent behavior.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_decision_result_store = None
_decision_results_fallback: dict[str, dict[str, Any]] = {}


def _get_result_store():
    """Get the decision result store for persistence."""
    global _decision_result_store
    if _decision_result_store is None:
        try:
            from aragora.storage.decision_result_store import get_decision_result_store

            _decision_result_store = get_decision_result_store()
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.warning("DecisionResultStore not available, using in-memory: %s", e)
    return _decision_result_store


def save_decision_result(request_id: str, data: dict[str, Any]) -> None:
    """Save a decision result to persistent store with fallback."""
    store = _get_result_store()
    if store:
        try:
            store.save(request_id, data)
            return
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning("Failed to persist result, using fallback: %s", e)
    _decision_results_fallback[request_id] = data


def get_decision_result(request_id: str) -> dict[str, Any] | None:
    """Get a decision result from persistent store with fallback."""
    store = _get_result_store()
    if store:
        try:
            result = store.get(request_id)
            if result:
                return result
        except (OSError, RuntimeError, KeyError) as e:
            logger.warning("Failed to retrieve from store: %s", e)
    return _decision_results_fallback.get(request_id)


def get_decision_status(request_id: str) -> dict[str, Any]:
    """Get decision status for polling with fallback."""
    store = _get_result_store()
    if store:
        try:
            return store.get_status(request_id)
        except (OSError, RuntimeError, KeyError) as e:
            logger.warning("Failed to get status from store: %s", e)

    if request_id in _decision_results_fallback:
        result = _decision_results_fallback[request_id]
        return {
            "request_id": request_id,
            "status": result.get("status", "unknown"),
            "completed_at": result.get("completed_at"),
        }
    return {
        "request_id": request_id,
        "status": "not_found",
    }


__all__ = [
    "save_decision_result",
    "get_decision_result",
    "get_decision_status",
]
