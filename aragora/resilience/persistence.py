"""
SQLite persistence for circuit breaker state.

Provides durable storage for circuit breaker state across restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from aragora.config import resolve_db_path
from aragora.exceptions import ConfigurationError

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Database configuration
_DB_PATH: str | None = None
_CB_TIMEOUT_SECONDS = 30.0  # SQLite busy timeout for concurrent access


def _get_cb_connection() -> sqlite3.Connection:
    """Get circuit breaker database connection with proper config.

    Configures SQLite with:
    - 30 second timeout for handling concurrent access
    - WAL mode for better write concurrency

    Returns:
        Configured sqlite3.Connection
    """
    if not _DB_PATH:
        raise ConfigurationError(
            component="CircuitBreaker",
            reason="Persistence not initialized. Call init_circuit_breaker_persistence() first",
        )

    conn = sqlite3.connect(_DB_PATH, timeout=_CB_TIMEOUT_SECONDS)
    # Enable WAL mode for better concurrent write performance
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_circuit_breaker_persistence(db_path: str = "circuit_breaker.db") -> None:
    """Initialize SQLite database for circuit breaker persistence.

    Creates the database and table if they don't exist.

    Args:
        db_path: Path to SQLite database file
    """
    global _DB_PATH
    _DB_PATH = resolve_db_path(db_path)

    # Ensure directory exists
    Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Use timeout and WAL mode for concurrent access
    with sqlite3.connect(_DB_PATH, timeout=_CB_TIMEOUT_SECONDS) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breakers (
                name TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                failure_threshold INTEGER NOT NULL,
                cooldown_seconds REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_circuit_breakers_updated
            ON circuit_breakers(updated_at)
        """)
        conn.commit()

    logger.info(f"Circuit breaker persistence initialized: {_DB_PATH}")


def persist_circuit_breaker(name: str, cb: "CircuitBreaker") -> None:
    """Persist a single circuit breaker to SQLite.

    Args:
        name: Circuit breaker name/identifier
        cb: CircuitBreaker instance to persist
    """
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
    except (sqlite3.Error, OSError) as e:
        logger.warning(f"Failed to persist circuit breaker {name}: {type(e).__name__}: {e}")


def persist_all_circuit_breakers() -> int:
    """Persist all registered circuit breakers to SQLite.

    Returns:
        Number of circuit breakers persisted
    """
    from .registry import _circuit_breakers, _circuit_breakers_lock

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
    from .circuit_breaker import CircuitBreaker
    from .registry import _circuit_breakers, _circuit_breakers_lock

    if not _DB_PATH:
        return 0

    try:
        with _get_cb_connection() as conn:
            cursor = conn.execute("""
                SELECT name, state_json, failure_threshold, cooldown_seconds
                FROM circuit_breakers
            """)

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

    except (sqlite3.Error, OSError) as e:
        logger.warning(f"Failed to load circuit breakers: {type(e).__name__}: {e}")
        return 0


def cleanup_stale_persisted(max_age_hours: float = 72.0) -> int:
    """Remove persisted circuit breakers older than max_age_hours.

    Args:
        max_age_hours: Maximum age in hours before deletion

    Returns:
        Number of stale entries deleted
    """
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

    except (sqlite3.Error, OSError) as e:
        logger.warning(f"Failed to cleanup stale circuit breakers: {type(e).__name__}: {e}")
        return 0


__all__ = [
    "init_circuit_breaker_persistence",
    "persist_circuit_breaker",
    "persist_all_circuit_breakers",
    "load_circuit_breakers",
    "cleanup_stale_persisted",
]
