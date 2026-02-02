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
import time
from typing import Any

from aragora.server.validation.entities import validate_path_segment

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker for Subsystem Access
# =============================================================================


class WorkspaceCircuitBreaker:
    """Circuit breaker for subsystem access in workspace handler.

    Prevents cascading failures when subsystems (isolation manager, retention manager,
    classifier, audit log) are unavailable. Uses a simple state machine:
    CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    # State constants
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 2,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            cooldown_seconds: Time to wait before allowing test calls
            half_open_max_calls: Number of test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            # Check if cooldown has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Workspace circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_proceed(self) -> bool:
        """Check if a call can proceed.

        Returns:
            True if call is allowed, False if circuit is open
        """
        with self._lock:
            state = self._check_state()
            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Workspace circuit breaker closed after successful recovery")
            elif self._state == self.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                # Any failure in half-open state reopens the circuit
                self._state = self.OPEN
                self._success_count = 0
                logger.warning("Workspace circuit breaker reopened after failure in HALF_OPEN")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"Workspace circuit breaker opened after {self._failure_count} failures"
                    )

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


# Per-subsystem circuit breakers for workspace handler
_workspace_circuit_breakers: dict[str, WorkspaceCircuitBreaker] = {}
_workspace_circuit_breaker_lock = threading.Lock()


def _get_workspace_circuit_breaker(subsystem: str) -> WorkspaceCircuitBreaker:
    """Get or create a circuit breaker for a workspace subsystem."""
    with _workspace_circuit_breaker_lock:
        if subsystem not in _workspace_circuit_breakers:
            _workspace_circuit_breakers[subsystem] = WorkspaceCircuitBreaker()
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
