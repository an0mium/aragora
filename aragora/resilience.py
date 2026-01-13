"""
Resilience patterns for fault-tolerant systems.

Provides circuit breaker and other resilience patterns for graceful
failure handling in API calls and agent interactions.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Generator, Optional

from aragora.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# Global circuit breaker registry for shared state across components (thread-safe)
_circuit_breakers: dict[str, "CircuitBreaker"] = {}
_circuit_breakers_lock = threading.Lock()

# Configuration for circuit breaker pruning
MAX_CIRCUIT_BREAKERS = 1000  # Maximum registry size before forced pruning
STALE_THRESHOLD_SECONDS = 24 * 60 * 60  # 24 hours - prune if not accessed


def _prune_stale_circuit_breakers() -> int:
    """Remove circuit breakers not accessed within STALE_THRESHOLD_SECONDS.

    Called automatically when registry exceeds MAX_CIRCUIT_BREAKERS.
    Must be called with _circuit_breakers_lock held.

    Returns:
        Number of circuit breakers pruned.
    """
    now = time.time()
    stale_names = []
    for name, cb in _circuit_breakers.items():
        if hasattr(cb, "_last_accessed") and (now - cb._last_accessed) > STALE_THRESHOLD_SECONDS:
            stale_names.append(name)

    for name in stale_names:
        del _circuit_breakers[name]

    if stale_names:
        logger.info(f"Pruned {len(stale_names)} stale circuit breakers: {stale_names[:5]}...")
    return len(stale_names)


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 3,
    cooldown_seconds: float = 60.0,
) -> "CircuitBreaker":
    """
    Get or create a named circuit breaker from the global registry (thread-safe).

    This ensures consistent circuit breaker state across components
    for the same service/agent.

    Automatically prunes stale circuit breakers (not accessed in 24h) when
    the registry exceeds MAX_CIRCUIT_BREAKERS entries.

    Args:
        name: Unique identifier for this circuit breaker (e.g., "agent_claude")
        failure_threshold: Failures before opening circuit
        cooldown_seconds: Seconds before attempting recovery

    Returns:
        CircuitBreaker instance (shared if already exists)
    """
    with _circuit_breakers_lock:
        # Prune if registry is getting too large
        if len(_circuit_breakers) >= MAX_CIRCUIT_BREAKERS:
            pruned = _prune_stale_circuit_breakers()
            # If still too large after pruning, log warning
            if len(_circuit_breakers) >= MAX_CIRCUIT_BREAKERS:
                logger.warning(
                    f"Circuit breaker registry still large after pruning {pruned}: "
                    f"{len(_circuit_breakers)} entries"
                )

        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                cooldown_seconds=cooldown_seconds,
            )
            logger.debug(f"Created circuit breaker: {name}")

        cb = _circuit_breakers[name]
        cb._last_accessed = time.time()  # Update access timestamp
        return cb


def reset_all_circuit_breakers() -> None:
    """Reset all global circuit breakers (thread-safe). Useful for testing."""
    with _circuit_breakers_lock:
        for cb in _circuit_breakers.values():
            cb.reset()
        count = len(_circuit_breakers)
    logger.info(f"Reset {count} circuit breakers")


def get_circuit_breaker_status() -> dict[str, Any]:
    """Get status of all registered circuit breakers (thread-safe)."""
    with _circuit_breakers_lock:
        return {
            "_registry_size": len(_circuit_breakers),
            **{
                name: {
                    "status": cb.get_status(),
                    "failures": cb.failures,
                    "last_accessed": getattr(cb, "_last_accessed", 0),
                }
                for name, cb in _circuit_breakers.items()
            },
        }


def get_circuit_breaker_metrics() -> dict[str, Any]:
    """Get comprehensive metrics for monitoring and observability.

    Returns metrics suitable for Prometheus/Grafana or other monitoring systems:
    - Summary counts (total, open, closed, half-open)
    - Per-circuit-breaker details with timing info
    - Configuration details
    - Health indicators for cascading failure detection

    Returns:
        Dict with structured metrics for monitoring integration.
    """
    now = time.time()
    with _circuit_breakers_lock:
        metrics: dict[str, Any] = {
            "timestamp": now,
            "registry_size": len(_circuit_breakers),
            "summary": {
                "total": 0,
                "open": 0,
                "closed": 0,
                "half_open": 0,
                "total_failures": 0,
                "circuits_with_failures": 0,
            },
            "circuit_breakers": {},
            "health": {
                "status": "healthy",
                "open_circuits": [],
                "high_failure_circuits": [],
            },
        }

        for name, cb in _circuit_breakers.items():
            status = cb.get_status()
            failures = cb.failures
            last_accessed = getattr(cb, "_last_accessed", 0)
            age_seconds = now - last_accessed if last_accessed > 0 else 0

            # Calculate cooldown remaining if open
            cooldown_remaining = 0.0
            open_duration = 0.0
            if status == "open" or status == "half-open":
                if cb._single_open_at > 0:
                    open_duration = now - cb._single_open_at
                    cooldown_remaining = max(0, cb.cooldown_seconds - open_duration)

            circuit_metrics = {
                "status": status,
                "failures": failures,
                "failure_threshold": cb.failure_threshold,
                "cooldown_seconds": cb.cooldown_seconds,
                "cooldown_remaining": cooldown_remaining,
                "open_duration": open_duration,
                "last_accessed_seconds_ago": age_seconds,
                "entity_mode": {
                    "tracked_entities": len(cb._failures),
                    "open_entities": list(cb._circuit_open_at.keys()),
                },
            }
            metrics["circuit_breakers"][name] = circuit_metrics

            # Update summary
            metrics["summary"]["total"] += 1
            metrics["summary"]["total_failures"] += failures
            if failures > 0:
                metrics["summary"]["circuits_with_failures"] += 1

            if status == "open":
                metrics["summary"]["open"] += 1
                metrics["health"]["open_circuits"].append(name)
            elif status == "half-open":
                metrics["summary"]["half_open"] += 1
            else:
                metrics["summary"]["closed"] += 1

            # Flag high-failure circuits (>50% of threshold)
            if failures >= cb.failure_threshold * 0.5:
                metrics["health"]["high_failure_circuits"].append(
                    {
                        "name": name,
                        "failures": failures,
                        "threshold": cb.failure_threshold,
                        "percentage": round(failures / cb.failure_threshold * 100, 1),
                    }
                )

        # Determine overall health status
        if metrics["summary"]["open"] > 0:
            metrics["health"]["status"] = "degraded"
        if metrics["summary"]["open"] >= 3:
            metrics["health"]["status"] = "critical"

        return metrics


def prune_circuit_breakers() -> int:
    """Manually prune stale circuit breakers from the registry.

    Removes circuit breakers not accessed within STALE_THRESHOLD_SECONDS (24h).

    Returns:
        Number of circuit breakers pruned.
    """
    with _circuit_breakers_lock:
        return _prune_stale_circuit_breakers()


class CircuitOpenError(Exception):
    """Raised when attempting to use an open circuit."""

    def __init__(self, circuit_name: str, cooldown_remaining: float):
        self.circuit_name = circuit_name
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit breaker '{circuit_name}' is open. " f"Retry in {cooldown_remaining:.1f}s"
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

    # Access tracking for memory management (pruning stale circuit breakers)
    _last_accessed: float = field(default_factory=time.time, repr=False)

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
                logger.warning(f"Circuit breaker OPEN after {self._single_failures} failures")
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
                    f"Circuit breaker OPEN for {entity} " f"after {self._failures[entity]} failures"
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
            self._half_open_successes[entity] = self._half_open_successes.get(entity, 0) + 1
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
        return [e for e in entities if self.is_available(getattr(e, "name", str(e)))]

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
                "open_circuits": {name: now - ts for name, ts in self._circuit_open_at.items()},
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
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "CircuitBreaker":
        """Load from persisted dict.

        Restores both single-mode and entity-mode state from persisted data.
        """
        cb = cls(**kwargs)

        # Restore single-mode state
        single_data = data.get("single_mode", {})
        cb._single_failures = single_data.get("failures", 0)
        is_open = single_data.get("is_open", False)
        open_for_seconds = single_data.get("open_for_seconds", 0)
        if is_open and open_for_seconds < cb.cooldown_seconds:
            cb._single_open_at = time.time() - open_for_seconds

        # Restore entity-mode state
        entity_data = data.get("entity_mode", data)  # Support both formats
        cb._failures = entity_data.get("failures", {})
        # Restore open circuits with remaining cooldown
        for name, elapsed in entity_data.get(
            "open_circuits", entity_data.get("cooldowns", {})
        ).items():
            if elapsed < cb.cooldown_seconds:
                cb._circuit_open_at[name] = time.time() - elapsed

        return cb

    @asynccontextmanager
    async def protected_call(
        self, entity: str | None = None, circuit_name: str | None = None
    ) -> AsyncGenerator[None, None]:
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
        except asyncio.CancelledError:
            # Task cancellation is not a service failure - don't record
            raise
        except Exception as e:
            # Record all other exceptions as failures (includes TimeoutError,
            # connection errors, API errors, etc.)
            logger.debug(f"Circuit breaker recorded failure for {name}: {type(e).__name__}: {e}")
            self.record_failure(entity)
            raise

    @contextmanager
    def protected_call_sync(
        self, entity: str | None = None, circuit_name: str | None = None
    ) -> Generator[None, None, None]:
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
            # Record all exceptions as failures (includes TimeoutError,
            # connection errors, API errors, etc.)
            logger.debug(
                f"Circuit breaker (sync) recorded failure for {name}: {type(e).__name__}: {e}"
            )
            self.record_failure(entity)
            raise


# =============================================================================
# SQLite Persistence for Circuit Breaker State
# =============================================================================

_DB_PATH: Optional[str] = None
_CB_TIMEOUT_SECONDS = 30.0  # SQLite busy timeout for concurrent access


def _get_cb_connection() -> "sqlite3.Connection":
    """Get circuit breaker database connection with proper config.

    Configures SQLite with:
    - 30 second timeout for handling concurrent access
    - WAL mode for better write concurrency

    Returns:
        Configured sqlite3.Connection
    """
    import sqlite3

    if not _DB_PATH:
        raise ConfigurationError(
            component="CircuitBreaker",
            reason="Persistence not initialized. Call init_circuit_breaker_persistence() first",
        )

    conn = sqlite3.connect(_DB_PATH, timeout=_CB_TIMEOUT_SECONDS)
    # Enable WAL mode for better concurrent write performance
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_circuit_breaker_persistence(db_path: str = ".data/circuit_breaker.db") -> None:
    """Initialize SQLite database for circuit breaker persistence.

    Creates the database and table if they don't exist.

    Args:
        db_path: Path to SQLite database file
    """
    import sqlite3
    from pathlib import Path

    global _DB_PATH
    _DB_PATH = db_path

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Use timeout and WAL mode for concurrent access
    with sqlite3.connect(db_path, timeout=_CB_TIMEOUT_SECONDS) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS circuit_breakers (
                name TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                failure_threshold INTEGER NOT NULL,
                cooldown_seconds REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_circuit_breakers_updated
            ON circuit_breakers(updated_at)
        """
        )
        conn.commit()

    logger.info(f"Circuit breaker persistence initialized: {db_path}")


def persist_circuit_breaker(name: str, cb: CircuitBreaker) -> None:
    """Persist a single circuit breaker to SQLite.

    Args:
        name: Circuit breaker name/identifier
        cb: CircuitBreaker instance to persist
    """
    import json
    from datetime import datetime

    if not _DB_PATH:
        return

    try:
        state = cb.to_dict()
        state_json = json.dumps(state)

        with _get_cb_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO circuit_breakers
                (name, state_json, failure_threshold, cooldown_seconds, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    name,
                    state_json,
                    cb.failure_threshold,
                    cb.cooldown_seconds,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to persist circuit breaker {name}: {e}")


def persist_all_circuit_breakers() -> int:
    """Persist all registered circuit breakers to SQLite.

    Returns:
        Number of circuit breakers persisted
    """
    if not _DB_PATH:
        return 0

    with _circuit_breakers_lock:
        count = 0
        for name, cb in _circuit_breakers.items():
            persist_circuit_breaker(name, cb)
            count += 1

    logger.debug(f"Persisted {count} circuit breakers")
    return count


def load_circuit_breakers() -> int:
    """Load circuit breakers from SQLite into the global registry.

    Returns:
        Number of circuit breakers loaded
    """
    import json

    if not _DB_PATH:
        return 0

    try:
        with _get_cb_connection() as conn:
            cursor = conn.execute(
                """
                SELECT name, state_json, failure_threshold, cooldown_seconds
                FROM circuit_breakers
            """
            )

            count = 0
            with _circuit_breakers_lock:
                for row in cursor.fetchall():
                    name, state_json, threshold, cooldown = row
                    try:
                        state = json.loads(state_json)
                        cb = CircuitBreaker.from_dict(
                            state,
                            failure_threshold=threshold,
                            cooldown_seconds=cooldown,
                        )
                        _circuit_breakers[name] = cb
                        count += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Malformed circuit breaker record {name}: {e}")

            logger.info(f"Loaded {count} circuit breakers from {_DB_PATH}")
            return count

    except Exception as e:
        logger.warning(f"Failed to load circuit breakers: {e}")
        return 0


def cleanup_stale_persisted(max_age_hours: float = 72.0) -> int:
    """Remove persisted circuit breakers older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours before deletion

    Returns:
        Number of stale entries deleted
    """
    from datetime import datetime, timedelta

    if not _DB_PATH:
        return 0

    try:
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        with _get_cb_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM circuit_breakers WHERE updated_at < ?
            """,
                (cutoff,),
            )
            conn.commit()
            deleted = cursor.rowcount

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} stale persisted circuit breakers")
        return deleted

    except Exception as e:
        logger.warning(f"Failed to cleanup stale circuit breakers: {e}")
        return 0
