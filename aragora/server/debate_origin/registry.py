"""Debate origin registration, lookup, and lifecycle management.

Manages the in-memory origin store with persistent backends (SQLite,
PostgreSQL, Redis) for durable debate origin tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from typing import Any, Optional

from .models import DebateOrigin
from .stores import (
    ORIGIN_TTL_SECONDS,
    _get_sqlite_store as _stores_get_sqlite_store,
    _get_postgres_store as _stores_get_postgres_store,
    _get_postgres_store_sync as _stores_get_postgres_store_sync,
)
from .sessions import _create_and_link_session

from aragora.control_plane.leader import (
    is_distributed_state_required,
    DistributedStateError,
)

logger = logging.getLogger(__name__)

# In-memory store with optional Redis backend
_origin_store: dict[str, DebateOrigin] = {}


def _get_pkg():
    """Get the package module to support test patching via __init__."""
    return sys.modules.get("aragora.server.debate_origin")


def _get_sqlite_store():
    """Wrapper that respects patches on the package namespace."""
    pkg = _get_pkg()
    if pkg is not None and hasattr(pkg, "_get_sqlite_store"):
        fn = getattr(pkg, "_get_sqlite_store")
        # Avoid infinite recursion if the package re-exported us
        if fn is not _get_sqlite_store:
            return fn()
    return _stores_get_sqlite_store()


async def _get_postgres_store():
    """Wrapper that respects patches on the package namespace."""
    pkg = _get_pkg()
    if pkg is not None and hasattr(pkg, "_get_postgres_store"):
        fn = getattr(pkg, "_get_postgres_store")
        if fn is not _get_postgres_store:
            return await fn()
    return await _stores_get_postgres_store()


def _get_postgres_store_sync():
    """Wrapper that respects patches on the package namespace."""
    pkg = _get_pkg()
    if pkg is not None and hasattr(pkg, "_get_postgres_store_sync"):
        fn = getattr(pkg, "_get_postgres_store_sync")
        if fn is not _get_postgres_store_sync:
            return fn()
    return _stores_get_postgres_store_sync()


def _store_origin_redis(origin: DebateOrigin) -> None:
    """Store origin in Redis."""
    import json as _json

    try:
        import redis

        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        key = f"debate_origin:{origin.debate_id}"
        r.setex(key, ORIGIN_TTL_SECONDS, _json.dumps(origin.to_dict()))
    except ImportError:
        raise
    except (OSError, ConnectionError, TimeoutError, ValueError) as e:
        logger.debug(f"Redis store failed: {e}")
        raise


def _load_origin_redis(debate_id: str) -> DebateOrigin | None:
    """Load origin from Redis."""
    import json as _json

    try:
        import redis

        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        key = f"debate_origin:{debate_id}"
        data = r.get(key)
        if data:
            return DebateOrigin.from_dict(_json.loads(data))
        return None
    except ImportError:
        raise
    except (OSError, ConnectionError, TimeoutError, ValueError, json.JSONDecodeError) as e:
        logger.debug(f"Redis load failed: {e}")
        raise


def _resolve_store_origin_redis():
    """Get _store_origin_redis, respecting patches on the package namespace."""
    pkg = _get_pkg()
    if pkg is not None and hasattr(pkg, "_store_origin_redis"):
        fn = getattr(pkg, "_store_origin_redis")
        if fn is not _store_origin_redis:
            return fn
    return _store_origin_redis


def _resolve_load_origin_redis():
    """Get _load_origin_redis, respecting patches on the package namespace."""
    pkg = _get_pkg()
    if pkg is not None and hasattr(pkg, "_load_origin_redis"):
        fn = getattr(pkg, "_load_origin_redis")
        if fn is not _load_origin_redis:
            return fn
    return _load_origin_redis


def register_debate_origin(
    debate_id: str,
    platform: str,
    channel_id: str,
    user_id: str,
    thread_id: str | None = None,
    message_id: str | None = None,
    metadata: Optional[dict[str, Any]] = None,
    session_id: str | None = None,
    create_session: bool = False,
) -> DebateOrigin:
    """Register the origin of a debate for result routing.

    Args:
        debate_id: Unique debate identifier
        platform: Platform name (telegram, whatsapp, slack, discord, etc.)
        channel_id: Channel/chat ID on the platform
        user_id: User ID who initiated the debate
        thread_id: Optional thread ID for threaded conversations
        message_id: Optional message ID that started the debate
        metadata: Optional additional metadata (username, etc.)
        session_id: Optional existing session ID to link
        create_session: If True and no session_id, create a new session

    Returns:
        DebateOrigin instance
    """
    # Handle session creation/linking
    linked_session_id = session_id
    if create_session and not session_id:
        try:
            from aragora.connectors.debate_session import get_debate_session_manager

            manager = get_debate_session_manager()
            # Try to create session synchronously
            try:
                # Check if we're in an async context
                loop = asyncio.get_running_loop()
                # In async context, schedule tasks
                asyncio.create_task(
                    _create_and_link_session(manager, platform, user_id, metadata, debate_id)
                )
            except RuntimeError:
                # No running event loop - create one for sync execution
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        session = loop.run_until_complete(
                            manager.create_session(platform, user_id, metadata)
                        )
                        linked_session_id = session.session_id
                        # Link debate to session
                        loop.run_until_complete(manager.link_debate(session.session_id, debate_id))
                    finally:
                        loop.close()
                except RuntimeError:
                    pass
        except ImportError:
            logger.debug("Session management not available")
        except (RuntimeError, OSError, ValueError) as e:
            logger.debug(f"Session creation failed: {e}")
    elif session_id:
        # Link existing session to debate
        try:
            from aragora.connectors.debate_session import get_debate_session_manager

            manager = get_debate_session_manager()
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                asyncio.create_task(manager.link_debate(session_id, debate_id))
            except RuntimeError:
                # No running event loop - create one for sync execution
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(manager.link_debate(session_id, debate_id))
                    finally:
                        loop.close()
                except RuntimeError:
                    pass
        except ImportError:
            pass
        except (RuntimeError, OSError, ValueError) as e:
            logger.debug(f"Session linking failed: {e}")

    origin = DebateOrigin(
        debate_id=debate_id,
        platform=platform,
        channel_id=channel_id,
        user_id=user_id,
        thread_id=thread_id,
        message_id=message_id,
        session_id=linked_session_id,
        metadata=metadata or {},
    )

    _origin_store[debate_id] = origin

    # Try PostgreSQL first if configured
    pg_store = _get_postgres_store_sync()
    if pg_store:
        try:
            # Check if we're in an async context
            asyncio.get_running_loop()
            asyncio.create_task(pg_store.save(origin))
        except RuntimeError:
            # No running event loop - create one for sync execution
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(pg_store.save(origin))
                finally:
                    loop.close()
            except OSError as e:
                logger.warning(f"PostgreSQL origin storage failed: {e}")
    else:
        # Fall back to SQLite for durability (always available)
        # Use async version when in running event loop to avoid blocking
        try:
            loop = asyncio.get_running_loop()
            # We're in async context - use async method via create_task
            asyncio.create_task(_get_sqlite_store().save_async(origin))
        except RuntimeError:
            # No running event loop - use sync version
            try:
                _get_sqlite_store().save(origin)
            except sqlite3.OperationalError as e:
                logger.warning(f"SQLite origin storage failed: {e}")

    # Persist to Redis for distributed deployments
    redis_success = False
    store_redis_fn = _resolve_store_origin_redis()
    try:
        store_redis_fn(origin)
        redis_success = True
    except ImportError:
        if is_distributed_state_required():
            raise DistributedStateError(
                "debate_origin",
                "Redis library not installed (pip install redis)",
            )
        logger.debug("Redis not available, using SQLite/PostgreSQL only")
    except (OSError, ConnectionError, TimeoutError, ValueError) as e:
        if is_distributed_state_required():
            raise DistributedStateError(
                "debate_origin",
                f"Redis connection failed: {e}",
            )
        logger.debug(f"Redis origin storage not available: {e}")

    logger.info(
        f"Registered debate origin: {debate_id} from {platform}:{channel_id} "
        f"(redis={redis_success})"
    )
    return origin


def get_debate_origin(debate_id: str) -> DebateOrigin | None:
    """Get the origin of a debate.

    Args:
        debate_id: Debate identifier

    Returns:
        DebateOrigin if found, None otherwise
    """
    # Check in-memory first
    origin = _origin_store.get(debate_id)
    if origin:
        return origin

    # Try Redis
    load_redis_fn = _resolve_load_origin_redis()
    try:
        origin = load_redis_fn(debate_id)
        if origin:
            _origin_store[debate_id] = origin  # Cache locally
            return origin
    except (
        ImportError,
        OSError,
        ConnectionError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as e:
        logger.debug(f"Redis origin lookup not available: {e}")

    # Try PostgreSQL if configured
    pg_store = _get_postgres_store_sync()
    if pg_store:
        try:
            # Check if we're in an async context - if so, can't block
            asyncio.get_running_loop()
            # In async context, caller should use get_debate_origin_async instead
        except RuntimeError:
            # No running event loop - create one for sync execution
            try:
                loop = asyncio.new_event_loop()
                try:
                    origin = loop.run_until_complete(pg_store.get(debate_id))
                    if origin:
                        _origin_store[debate_id] = origin  # Cache locally
                        return origin
                finally:
                    loop.close()
            except OSError as e:
                logger.debug(f"PostgreSQL origin lookup failed: {e}")
    else:
        # Try SQLite fallback (sync - caller should use get_debate_origin_async if possible)
        try:
            origin = _get_sqlite_store().get(debate_id)
            if origin:
                _origin_store[debate_id] = origin  # Cache locally
                return origin
        except (sqlite3.OperationalError, json.JSONDecodeError) as e:
            logger.debug(f"SQLite origin lookup failed: {e}")

    return None


async def get_debate_origin_async(debate_id: str) -> DebateOrigin | None:
    """Async version of get_debate_origin that doesn't block event loop.

    Prefer this in async contexts to avoid blocking the event loop
    with synchronous SQLite operations.

    Args:
        debate_id: Debate identifier

    Returns:
        DebateOrigin if found, None otherwise
    """
    # Check in-memory first
    origin = _origin_store.get(debate_id)
    if origin:
        return origin

    # Try Redis
    load_redis_fn = _resolve_load_origin_redis()
    try:
        origin = load_redis_fn(debate_id)
        if origin:
            _origin_store[debate_id] = origin  # Cache locally
            return origin
    except (
        ImportError,
        OSError,
        ConnectionError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as e:
        logger.debug(f"Redis origin lookup not available: {e}")

    # Try PostgreSQL if configured
    pg_store = await _get_postgres_store()
    if pg_store:
        try:
            origin = await pg_store.get(debate_id)
            if origin:
                _origin_store[debate_id] = origin  # Cache locally
                return origin
        except OSError as e:
            logger.debug(f"PostgreSQL origin lookup failed: {e}")
    else:
        # Try SQLite fallback with async method
        try:
            origin = await _get_sqlite_store().get_async(debate_id)
            if origin:
                _origin_store[debate_id] = origin  # Cache locally
                return origin
        except (sqlite3.OperationalError, json.JSONDecodeError) as e:
            logger.debug(f"SQLite origin lookup failed: {e}")

    return None


def mark_result_sent(debate_id: str) -> None:
    """Mark that the result has been sent for a debate."""
    origin = get_debate_origin(debate_id)
    if origin:
        origin.result_sent = True
        origin.result_sent_at = time.time()

        # Update PostgreSQL if configured
        pg_store = _get_postgres_store_sync()
        if pg_store:
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                asyncio.create_task(pg_store.save(origin))
            except RuntimeError:
                # No running event loop - create one for sync execution
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(pg_store.save(origin))
                    finally:
                        loop.close()
                except OSError as e:
                    logger.debug(f"PostgreSQL update failed: {e}")
        else:
            # Update SQLite - use async when in running event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in async context - use async method
                asyncio.create_task(_get_sqlite_store().save_async(origin))
            except RuntimeError:
                # No running event loop - use sync version
                try:
                    _get_sqlite_store().save(origin)
                except sqlite3.OperationalError as e:
                    logger.debug(f"SQLite update failed: {e}")

        # Update Redis if available
        store_redis_fn = _resolve_store_origin_redis()
        try:
            store_redis_fn(origin)
        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            # Catch Redis errors (connection, timeout, etc.)
            logger.debug(f"Redis update skipped: {e}")


def cleanup_expired_origins() -> int:
    """Remove expired origin records from in-memory store and persistent storage.

    This function cleans up expired debate origins from:
    1. In-memory cache
    2. PostgreSQL database (if configured)
    3. SQLite database (fallback persistent storage)

    Should be called periodically (e.g., hourly) to prevent unbounded growth.

    Returns:
        Total count of expired records removed
    """
    total_cleaned = 0
    now = time.time()

    # Clean up in-memory store
    expired = [k for k, v in _origin_store.items() if now - v.created_at > ORIGIN_TTL_SECONDS]

    for k in expired:
        del _origin_store[k]

    if expired:
        logger.info(f"Cleaned up {len(expired)} expired debate origins from memory")
        total_cleaned += len(expired)

    # Clean up PostgreSQL if configured
    pg_store = _get_postgres_store_sync()
    if pg_store:
        try:
            # Check if we're in an async context - if so, can't block
            asyncio.get_running_loop()
            # In async context, caller should use an async cleanup method
        except RuntimeError:
            # No running event loop - create one for sync execution
            try:
                loop = asyncio.new_event_loop()
                try:
                    pg_cleaned = loop.run_until_complete(
                        pg_store.cleanup_expired(ORIGIN_TTL_SECONDS)
                    )
                    if pg_cleaned > 0:
                        logger.info(
                            f"Cleaned up {pg_cleaned} expired debate origins from PostgreSQL"
                        )
                        total_cleaned += pg_cleaned
                finally:
                    loop.close()
            except OSError as e:
                logger.warning(f"PostgreSQL cleanup failed: {e}")
    else:
        # Clean up SQLite store (fallback)
        try:
            sqlite_cleaned = _get_sqlite_store().cleanup_expired(ORIGIN_TTL_SECONDS)
            if sqlite_cleaned > 0:
                logger.info(f"Cleaned up {sqlite_cleaned} expired debate origins from SQLite")
                total_cleaned += sqlite_cleaned
        except sqlite3.OperationalError as e:
            logger.warning(f"SQLite cleanup failed: {e}")

    return total_cleaned
