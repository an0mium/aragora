"""
Gmail state persistence and statistics.

Provides loading and saving of Gmail sync state across multiple
storage backends (memory, Redis, PostgreSQL).
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import GmailSyncState

logger = logging.getLogger(__name__)


class GmailStateMixin:
    """Mixin providing state persistence and statistics operations."""

    # Expected attributes from concrete class
    user_id: str
    _gmail_state: GmailSyncState | None

    async def load_gmail_state(
        self,
        tenant_id: str,
        user_id: str,
        backend: str = "memory",
        redis_url: str | None = None,
        postgres_dsn: str | None = None,
    ) -> GmailSyncState | None:
        """
        Load Gmail sync state from persistent storage.

        Supports multiple backends for tenant-isolated state management.

        Args:
            tenant_id: Tenant identifier for isolation
            user_id: User identifier
            backend: Storage backend ("memory", "redis", "postgres")
            redis_url: Redis connection URL (required for redis backend)
            postgres_dsn: PostgreSQL DSN (required for postgres backend)

        Returns:
            GmailSyncState if found, None otherwise
        """
        import json

        state_key = f"gmail_sync:{tenant_id}:{user_id}"

        if backend == "redis" and redis_url:
            try:
                import redis.asyncio as redis_client

                client = redis_client.from_url(redis_url)
                data = await client.get(state_key)
                await client.close()
                if data:
                    state = GmailSyncState.from_dict(json.loads(data))
                    self._gmail_state = state
                    logger.info(f"[Gmail] Loaded state from Redis for {state_key}")
                    return state
            except ImportError:
                logger.warning("[Gmail] redis package not installed, cannot use Redis backend")
            except (OSError, ValueError, TypeError, KeyError) as e:
                logger.warning(f"[Gmail] Failed to load state from Redis: {e}")

        elif backend == "postgres" and postgres_dsn:
            try:
                import asyncpg

                conn = await asyncpg.connect(postgres_dsn)
                row = await conn.fetchrow(
                    "SELECT state FROM gmail_sync_state WHERE key = $1",
                    state_key,
                )
                await conn.close()
                if row:
                    state = GmailSyncState.from_dict(json.loads(row["state"]))
                    self._gmail_state = state
                    logger.info(f"[Gmail] Loaded state from Postgres for {state_key}")
                    return state
            except ImportError:
                logger.warning("[Gmail] asyncpg package not installed, cannot use Postgres backend")
            except (OSError, ValueError, TypeError, KeyError) as e:
                logger.warning(f"[Gmail] Failed to load state from Postgres: {e}")

        # Memory backend or fallback
        logger.debug(f"[Gmail] No persisted state found for {state_key}")
        return None

    async def save_gmail_state(
        self,
        tenant_id: str,
        user_id: str,
        backend: str = "memory",
        redis_url: str | None = None,
        postgres_dsn: str | None = None,
    ) -> bool:
        """
        Save Gmail sync state to persistent storage.

        Args:
            tenant_id: Tenant identifier for isolation
            user_id: User identifier
            backend: Storage backend ("memory", "redis", "postgres")
            redis_url: Redis connection URL (required for redis backend)
            postgres_dsn: PostgreSQL DSN (required for postgres backend)

        Returns:
            True if saved successfully
        """
        import json

        if not self._gmail_state:
            logger.warning("[Gmail] No state to save")
            return False

        state_key = f"gmail_sync:{tenant_id}:{user_id}"
        state_json = json.dumps(self._gmail_state.to_dict())

        if backend == "redis" and redis_url:
            try:
                import redis.asyncio as redis_client

                client = redis_client.from_url(redis_url)
                await client.set(state_key, state_json)
                await client.close()
                logger.info(f"[Gmail] Saved state to Redis for {state_key}")
                return True
            except ImportError:
                logger.warning("[Gmail] redis package not installed, cannot use Redis backend")
            except (OSError, ValueError, TypeError) as e:
                logger.warning(f"[Gmail] Failed to save state to Redis: {e}")
                return False

        elif backend == "postgres" and postgres_dsn:
            try:
                import asyncpg

                conn = await asyncpg.connect(postgres_dsn)
                await conn.execute(
                    """
                    INSERT INTO gmail_sync_state (key, state, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (key) DO UPDATE SET state = $2, updated_at = NOW()
                    """,
                    state_key,
                    state_json,
                )
                await conn.close()
                logger.info(f"[Gmail] Saved state to Postgres for {state_key}")
                return True
            except ImportError:
                logger.warning("[Gmail] asyncpg package not installed, cannot use Postgres backend")
            except (OSError, ValueError, TypeError) as e:
                logger.warning(f"[Gmail] Failed to save state to Postgres: {e}")
                return False

        # Memory backend - state is already in self._gmail_state
        logger.debug(f"[Gmail] State in memory for {state_key}")
        return True

    def get_sync_stats(self) -> dict[str, Any]:
        """
        Get sync service statistics.

        Returns:
            Dict with current sync state and statistics
        """
        return {
            "user_id": self.user_id,
            "email_address": self._gmail_state.email_address if self._gmail_state else None,
            "history_id": self._gmail_state.history_id if self._gmail_state else None,
            "last_sync": (
                self._gmail_state.last_sync.isoformat()
                if self._gmail_state and self._gmail_state.last_sync
                else None
            ),
            "initial_sync_complete": (
                self._gmail_state.initial_sync_complete if self._gmail_state else False
            ),
            "watch_active": bool(self._gmail_state and self._gmail_state.watch_resource_id),
            "watch_expiration": (
                self._gmail_state.watch_expiration.isoformat()
                if self._gmail_state and self._gmail_state.watch_expiration
                else None
            ),
            "total_messages": self._gmail_state.total_messages if self._gmail_state else 0,
            "indexed_messages": (self._gmail_state.indexed_messages if self._gmail_state else 0),
            "sync_errors": self._gmail_state.sync_errors if self._gmail_state else 0,
            "last_error": self._gmail_state.last_error if self._gmail_state else None,
        }
