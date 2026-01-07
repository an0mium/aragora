"""
Resilience patterns for fault-tolerant systems.

Provides circuit breaker and other resilience patterns for graceful
failure handling in API calls and agent interactions.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Global circuit breaker registry for shared state across components (thread-safe)
_circuit_breakers: dict[str, "CircuitBreaker"] = {}
_circuit_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 3,
    cooldown_seconds: float = 60.0,
) -> "CircuitBreaker":
    """
    Get or create a named circuit breaker from the global registry (thread-safe).

    This ensures consistent circuit breaker state across components
    for the same service/agent.

    Args:
        name: Unique identifier for this circuit breaker (e.g., "agent_claude")
        failure_threshold: Failures before opening circuit
        cooldown_seconds: Seconds before attempting recovery

    Returns:
        CircuitBreaker instance (shared if already exists)
    """
    with _circuit_breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
            )
            logger.debug(f"Created circuit breaker: {name}")
        return _circuit_breakers[name]


def reset_all_circuit_breakers() -> None:
    """Reset all global circuit breakers (thread-safe). Useful for testing."""
    with _circuit_breakers_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
        count = len(_circuit_breakers)
    logger.info(f"Reset {count} circuit breakers")


def get_circuit_breaker_status() -> dict[str, dict]:
    """Get status of all registered circuit breakers (thread-safe)."""
    with _circuit_breakers_lock:
        return {
            name: {
                "status": cb.get_status(),
                "failures": cb.failures,
            }
            for name, cb in _circuit_breakers.items()
        }


class CircuitOpenError(Exception):
    """Raised when attempting to use an open circuit."""

    def __init__(self, circuit_name: str, cooldown_remaining: float):
        self.circuit_name = circuit_name
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open. "
            f"Retry in {cooldown_remaining:.1f}s"
        )


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for graceful failure handling.

    Supports both single-entity and multi-entity tracking.
    Implements three states:
    - CLOSED: Normal operation, requests allowed
    - OPEN: After failure threshold, requests blocked
    - HALF-OPEN: After cooldown, trial requests allowed

    Usage (single entity):
        breaker = CircuitBreaker()
        if breaker.can_proceed():
            try:
                result = await call_api()
                breaker.record_success()
            except Exception:
                breaker.record_failure()

    Usage (multi-entity):
        breaker = CircuitBreaker()
        if breaker.is_available("agent-1"):
            try:
                result = await agent.generate(...)
                breaker.record_success("agent-1")
            except Exception:
                breaker.record_failure("agent-1")
    """

    failure_threshold: int = 3  # Consecutive failures before opening circuit
    cooldown_seconds: float = 60.0  # Seconds before attempting recovery
    half_open_success_threshold: int = 2  # Successes needed to fully close

    # Internal state (initialized in __post_init__)
    _failures: dict[str, int] = field(default_factory=dict, repr=False)
    _circuit_open_at: dict[str, float] = field(default_factory=dict, repr=False)
    _half_open_successes: dict[str, int] = field(default_factory=dict, repr=False)

    # Single-entity mode state
    _single_failures: int = field(default=0, repr=False)
    _single_open_at: float = field(default=0.0, repr=False)
    _single_successes: int = field(default=0, repr=False)

    # Backward-compatible properties for single-entity mode
    @property
    def reset_timeout(self) -> float:
        """Alias for cooldown_seconds (backward compatibility)."""
        return self.cooldown_seconds

    @property
    def failures(self) -> int:
        """Current failure count in single-entity mode."""
        return self._single_failures

    @property
    def is_open(self) -> bool:
        """Whether circuit is open in single-entity mode."""
        return self._single_open_at > 0.0

    @is_open.setter
    def is_open(self, value: bool) -> None:
        """Set circuit open state (for testing/manual control)."""
        if value:
            self._single_open_at = time.time()
        else:
            self._single_open_at = 0.0
            self._single_failures = 0
            self._single_successes = 0

    def record_failure(self, entity: str | None = None) -> bool:
        """
        Record a failure. Returns True if circuit just opened.

        Args:
            entity: Optional entity name for multi-entity tracking.
                   If None, uses single-entity mode.
        """
        if entity is None:
            return self._record_single_failure()
        return self._record_entity_failure(entity)

    def _record_single_failure(self) -> bool:
        """Record failure in single-entity mode."""
        self._single_failures += 1
        self._single_successes = 0

        if self._single_failures >= self.failure_threshold:
            if self._single_open_at == 0.0:
                self._single_open_at = time.time()
                logger.warning(
                    f"Circuit breaker OPEN after {self._single_failures} failures"
                )
                return True
        return False

    def _record_entity_failure(self, entity: str) -> bool:
        """Record failure for a specific entity."""
        self._failures[entity] = self._failures.get(entity, 0) + 1
        self._half_open_successes[entity] = 0

        if self._failures[entity] >= self.failure_threshold:
            if entity not in self._circuit_open_at:
                self._circuit_open_at[entity] = time.time()
                logger.warning(
                    f"Circuit breaker OPEN for {entity} "
                    f"after {self._failures[entity]} failures"
                )
                return True
        return False

    def record_success(self, entity: str | None = None) -> None:
        """
        Record a success. May close an open circuit.

        Args:
            entity: Optional entity name for multi-entity tracking.
        """
        if entity is None:
            self._record_single_success()
        else:
            self._record_entity_success(entity)

    def _record_single_success(self) -> None:
        """Record success in single-entity mode.

        In single-entity mode, one success closes the circuit immediately
        (no half-open threshold) for backward compatibility.
        """
        if self._single_open_at > 0.0:
            # Single success closes circuit in single-entity mode
            self._single_open_at = 0.0
            self._single_failures = 0
            self._single_successes = 0
            logger.info("Circuit breaker CLOSED")
        else:
            self._single_failures = 0

    def _record_entity_success(self, entity: str) -> None:
        """Record success for a specific entity."""
        if entity in self._circuit_open_at:
            self._half_open_successes[entity] = (
                self._half_open_successes.get(entity, 0) + 1
            )
            if self._half_open_successes[entity] >= self.half_open_success_threshold:
                del self._circuit_open_at[entity]
                self._failures[entity] = 0
                self._half_open_successes[entity] = 0
                logger.info(f"Circuit breaker CLOSED for {entity}")
        else:
            self._failures[entity] = 0

    def can_proceed(self, entity: str | None = None) -> bool:
        """
        Check if we can proceed with a request.

        Args:
            entity: Optional entity name for multi-entity tracking.

        Returns:
            True if request is allowed (circuit closed or half-open).
        """
        if entity is None:
            return self._can_proceed_single()
        return self.is_available(entity)

    def _can_proceed_single(self) -> bool:
        """Check if single-entity circuit allows requests."""
        if self._single_open_at == 0.0:
            return True

        # Check if cooldown has passed - reset circuit
        elapsed = time.time() - self._single_open_at
        if elapsed >= self.cooldown_seconds:
            # Fully reset circuit after cooldown (backward-compatible behavior)
            self._single_open_at = 0.0
            self._single_failures = 0
            self._single_successes = 0
            logger.info("Circuit breaker cooldown elapsed, circuit CLOSED")
            return True

        return False

    def is_available(self, entity: str) -> bool:
        """Check if an entity is available for use."""
        if entity not in self._circuit_open_at:
            return True

        # Check if cooldown has passed (half-open state)
        elapsed = time.time() - self._circuit_open_at[entity]
        if elapsed >= self.cooldown_seconds:
            logger.debug(
                f"Circuit breaker HALF-OPEN for {entity} "
                f"(cooldown {self.cooldown_seconds}s elapsed)"
            )
            return True

        return False

    def get_status(self, entity: str | None = None) -> str:
        """
        Get circuit status: 'closed', 'open', or 'half-open'.

        Args:
            entity: Optional entity name. If None, uses single-entity mode.
        """
        if entity is None:
            return self._get_single_status()

        if entity not in self._circuit_open_at:
            return "closed"
        elapsed = time.time() - self._circuit_open_at[entity]
        if elapsed >= self.cooldown_seconds:
            return "half-open"
        return "open"

    def _get_single_status(self) -> str:
        """Get status for single-entity mode."""
        if self._single_open_at == 0.0:
            return "closed"
        elapsed = time.time() - self._single_open_at
        if elapsed >= self.cooldown_seconds:
            return "half-open"
        return "open"

    def filter_available_entities(self, entities: list) -> list:
        """Return only entities with closed or half-open circuits."""
        return [e for e in entities if self.is_available(getattr(e, 'name', str(e)))]

    def filter_available_agents(self, agents: list) -> list:
        """Alias for filter_available_entities (backward compatibility)."""
        return self.filter_available_entities(agents)

    def to_dict(self) -> dict:
        """Serialize to dict for persistence/debugging."""
        now = time.time()
        return {
            "single_mode": {
                "failures": self._single_failures,
                "is_open": self._single_open_at > 0.0,
                "open_for_seconds": (
                    now - self._single_open_at if self._single_open_at > 0.0 else 0
                ),
            },
            "entity_mode": {
                "failures": self._failures.copy(),
                "open_circuits": {
                    name: now - ts for name, ts in self._circuit_open_at.items()
                },
            },
        }

    def reset(self, entity: str | None = None) -> None:
        """
        Reset circuit breaker state.

        Args:
            entity: If provided, reset only that entity. Otherwise reset all.
        """
        if entity is None:
            self._single_failures = 0
            self._single_open_at = 0.0
            self._single_successes = 0
            self._failures.clear()
            self._circuit_open_at.clear()
            self._half_open_successes.clear()
            logger.info("Circuit breaker reset all states")
        else:
            self._failures.pop(entity, None)
            self._circuit_open_at.pop(entity, None)
            self._half_open_successes.pop(entity, None)
            logger.info(f"Circuit breaker reset state for {entity}")

    def get_all_status(self) -> dict[str, dict]:
        """Get status for all tracked entities."""
        all_entities = set(self._failures.keys()) | set(self._circuit_open_at.keys())
        return {
            entity: {
                "status": self.get_status(entity),
                "failures": self._failures.get(entity, 0),
                "half_open_successes": self._half_open_successes.get(entity, 0),
            }
            for entity in all_entities
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "CircuitBreaker":
        """Load from persisted dict."""
        cb = cls(**kwargs)
        entity_data = data.get("entity_mode", data)  # Support both formats
        cb._failures = entity_data.get("failures", {})
        # Restore open circuits with remaining cooldown
        for name, elapsed in entity_data.get("open_circuits", entity_data.get("cooldowns", {})).items():
            if elapsed < cb.cooldown_seconds:
                cb._circuit_open_at[name] = time.time() - elapsed
        return cb

    @asynccontextmanager
    async def protected_call(self, entity: str | None = None, circuit_name: str | None = None):
        """
        Async context manager for circuit-breaker-protected calls.

        Automatically checks if circuit is open before call and records
        success/failure after the call completes.

        Args:
            entity: Optional entity name for multi-entity mode
            circuit_name: Name for error messages (defaults to entity or "circuit")

        Raises:
            CircuitOpenError: If the circuit is open

        Usage:
            async with breaker.protected_call("my-agent"):
                result = await api_call()
        """
        name = circuit_name or entity or "circuit"

        # Check if circuit allows requests
        if not self.can_proceed(entity):
            # Calculate remaining cooldown
            if entity is None:
                elapsed = time.time() - self._single_open_at
            else:
                elapsed = time.time() - self._circuit_open_at.get(entity, 0)
            remaining = max(0, self.cooldown_seconds - elapsed)
            raise CircuitOpenError(name, remaining)

        try:
            yield
            self.record_success(entity)
        except Exception as e:
            logger.debug(f"Circuit breaker recorded failure for {name}: {type(e).__name__}: {e}")
            self.record_failure(entity)
            raise

    @contextmanager
    def protected_call_sync(self, entity: str | None = None, circuit_name: str | None = None):
        """
        Sync context manager for circuit-breaker-protected calls.

        Args:
            entity: Optional entity name for multi-entity mode
            circuit_name: Name for error messages

        Raises:
            CircuitOpenError: If the circuit is open

        Usage:
            with breaker.protected_call_sync("my-agent"):
                result = sync_api_call()
        """
        name = circuit_name or entity or "circuit"

        if not self.can_proceed(entity):
            if entity is None:
                elapsed = time.time() - self._single_open_at
            else:
                elapsed = time.time() - self._circuit_open_at.get(entity, 0)
            remaining = max(0, self.cooldown_seconds - elapsed)
            raise CircuitOpenError(name, remaining)

        try:
            yield
            self.record_success(entity)
        except Exception as e:
            logger.debug(f"Circuit breaker (sync) recorded failure for {name}: {type(e).__name__}: {e}")
            self.record_failure(entity)
            raise
