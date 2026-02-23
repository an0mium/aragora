"""
Workspace Utilities - Circuit Breaker and Validation Functions.

This module contains shared utilities for the workspace handler package:
- WorkspaceCircuitBreaker: Circuit breaker pattern for resilient subsystem access
- Validation functions for workspace IDs, policy IDs, and user IDs

Stability: STABLE
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from aragora.server.validation.entities import validate_path_segment

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker for Subsystem Access
# =============================================================================

from aragora.resilience.simple_circuit_breaker import (
    SimpleCircuitBreaker as WorkspaceCircuitBreaker,
)

# Per-subsystem circuit breakers for workspace handler
_workspace_circuit_breakers: dict[str, WorkspaceCircuitBreaker] = {}
_workspace_circuit_breaker_lock = threading.Lock()


def _get_workspace_circuit_breaker(subsystem: str) -> WorkspaceCircuitBreaker:
    """Get or create a circuit breaker for a workspace subsystem."""
    with _workspace_circuit_breaker_lock:
        if subsystem not in _workspace_circuit_breakers:
            _workspace_circuit_breakers[subsystem] = WorkspaceCircuitBreaker(
                "workspace", failure_threshold=3, half_open_max_calls=2
            )
        return _workspace_circuit_breakers[subsystem]


def get_workspace_circuit_breaker_status() -> dict[str, Any]:
    """Get status of all workspace subsystem circuit breakers."""
    with _workspace_circuit_breaker_lock:
        return {name: cb.get_status() for name, cb in _workspace_circuit_breakers.items()}


# =============================================================================
# Validation Functions
# =============================================================================


def _validate_workspace_id(workspace_id: str) -> tuple[bool, str | None]:
    """Validate workspace ID format.

    Args:
        workspace_id: Workspace identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not workspace_id:
        return False, "workspace_id is required"
    is_valid, err = validate_path_segment(workspace_id, "workspace_id")
    if not is_valid:
        return False, err or f"Invalid workspace_id format: {workspace_id}"
    return True, None


def _validate_policy_id(policy_id: str) -> tuple[bool, str | None]:
    """Validate retention policy ID format.

    Args:
        policy_id: Policy identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not policy_id:
        return False, "policy_id is required"
    is_valid, err = validate_path_segment(policy_id, "policy_id")
    if not is_valid:
        return False, err or f"Invalid policy_id format: {policy_id}"
    return True, None


def _validate_user_id(user_id: str) -> tuple[bool, str | None]:
    """Validate user ID format.

    Args:
        user_id: User identifier to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not user_id:
        return False, "user_id is required"
    is_valid, err = validate_path_segment(user_id, "user_id")
    if not is_valid:
        return False, err or f"Invalid user_id format: {user_id}"
    return True, None


__all__ = [
    "WorkspaceCircuitBreaker",
    "_get_workspace_circuit_breaker",
    "get_workspace_circuit_breaker_status",
    "_validate_workspace_id",
    "_validate_policy_id",
    "_validate_user_id",
]
