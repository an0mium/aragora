"""
Redis integration store backend with SQLite fallback.

Uses Redis for fast distributed access, with SQLite as durable storage.
Enables multi-instance deployments while ensuring persistence.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from aragora.storage.integration_backends import IntegrationStoreBackend
from aragora.storage.integration_models import (
    IntegrationConfig,
    UserIdMapping,
    _record_user_mapping_cache_hit,
    _record_user_mapping_cache_miss,
    _record_user_mapping_operation,
)
from aragora.storage.integration_sqlite import SQLiteIntegrationStore

logger = logging.getLogger(__name__)


class RedisIntegrationStore(IntegrationStoreBackend):
    """
    Redis-backed integration store with SQLite fallback.

    Uses Redis for fast distributed access, with SQLite as durable storage.
    This enables multi-instance deployments while ensuring persistence.
    """

    REDIS_PREFIX = "aragora:integrations"
    REDIS_TTL = 86400  # 24 hours

    def __init__(self, db_path: Path | str, redis_url: str | None = None):
        self._sqlite = SQLiteIntegrationStore(db_path)
        self._redis: Any | None = None
        self._redis_url = redis_url or os.environ.get("ARAGORA_REDIS_URL", "redis://localhost:6379")
        self._redis_checked = False
        logger.info("RedisIntegrationStore initialized with SQLite fallback")

    def _get_redis(self) -> Any | None:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
            # Test connection
            self._redis.ping()
            self._redis_checked = True
            logger.info("Redis connected for integration store")
        except ImportError as e:
            logger.debug("Redis package not installed: %s", e)
            self._redis = None
            self._redis_checked = True
        except (ConnectionError, OSError, RuntimeError, ValueError) as e:
            # Catch Redis connection errors (redis.exceptions.ConnectionError subclasses ConnectionError)
            logger.debug("Redis not available, using SQLite only: %s", e)
            self._redis = None
            self._redis_checked = True

        return self._redis

    def _redis_key(self, integration_type: str, user_id: str) -> str:
        return f"{self.REDIS_PREFIX}:{user_id}:{integration_type}"

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> IntegrationConfig | None:
        redis = self._get_redis()

        # Try Redis first
        if redis is not None:
            try:
                key = self._redis_key(integration_type, user_id)
                data = redis.get(key)
                if data:
                    return IntegrationConfig.from_json(data)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis get failed (connection error), falling back to SQLite: %s", e)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug("Redis get failed (data error), falling back to SQLite: %s", e)

        # Fall back to SQLite
        config = await self._sqlite.get(integration_type, user_id)

        # Populate Redis cache if found
        if config and redis:
            try:
                key = self._redis_key(integration_type, user_id)
                redis.setex(key, self.REDIS_TTL, config.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis cache population failed (connection issue): %s", e)
            except (TypeError, ValueError) as e:
                logger.debug("Redis cache population failed (serialization): %s", e)

        return config

    async def save(self, config: IntegrationConfig) -> None:
        user_id = config.user_id or "default"

        # Always save to SQLite (durable)
        await self._sqlite.save(config)

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                key = self._redis_key(config.type, user_id)
                redis.setex(key, self.REDIS_TTL, config.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis cache update failed (connection issue): %s", e)
            except (TypeError, ValueError) as e:
                logger.debug("Redis cache update failed (serialization): %s", e)

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        # Delete from both stores
        redis = self._get_redis()
        if redis:
            try:
                key = self._redis_key(integration_type, user_id)
                redis.delete(key)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis cache delete failed (connection issue): %s", e)
            except (TypeError, ValueError) as e:
                logger.debug("Redis cache delete failed (key error): %s", e)

        return await self._sqlite.delete(integration_type, user_id)

    async def list_for_user(self, user_id: str = "default") -> list[IntegrationConfig]:
        # Always use SQLite for list operations (authoritative)
        return await self._sqlite.list_for_user(user_id)

    async def list_all(self, limit: int = 1000) -> list[IntegrationConfig]:
        return await self._sqlite.list_all(limit)

    def _mapping_redis_key(self, email: str, platform: str, user_id: str) -> str:
        return f"{self.REDIS_PREFIX}:mapping:{user_id}:{platform}:{email}"

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> UserIdMapping | None:
        redis = self._get_redis()

        # Try Redis first
        if redis is not None:
            try:
                key = self._mapping_redis_key(email, platform, user_id)
                data = redis.get(key)
                if data:
                    _record_user_mapping_cache_hit(platform)
                    _record_user_mapping_operation("get", platform, True)
                    return UserIdMapping.from_json(data)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis mapping get failed (connection error): %s", e)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug("Redis mapping get failed (data error): %s", e)

        # Fall back to SQLite (cache miss)
        _record_user_mapping_cache_miss(platform)
        mapping = await self._sqlite.get_user_mapping(email, platform, user_id)
        _record_user_mapping_operation("get", platform, mapping is not None)

        # Populate Redis cache if found
        if mapping and redis:
            try:
                key = self._mapping_redis_key(email, platform, user_id)
                redis.setex(key, self.REDIS_TTL, mapping.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis mapping cache population failed (connection issue): %s", e)
            except (TypeError, ValueError) as e:
                logger.debug("Redis mapping cache population failed (serialization): %s", e)

        return mapping

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        # Always save to SQLite (durable)
        await self._sqlite.save_user_mapping(mapping)

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                key = self._mapping_redis_key(mapping.email, mapping.platform, mapping.user_id)
                redis.setex(key, self.REDIS_TTL, mapping.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis mapping cache update failed (connection issue): %s", e)
            except (TypeError, ValueError) as e:
                logger.debug("Redis mapping cache update failed (serialization): %s", e)

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        redis = self._get_redis()
        if redis:
            try:
                key = self._mapping_redis_key(email, platform, user_id)
                redis.delete(key)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis mapping delete failed (connection issue): %s", e)
            except (TypeError, ValueError) as e:
                logger.debug("Redis mapping delete failed (key error): %s", e)

        return await self._sqlite.delete_user_mapping(email, platform, user_id)

    async def list_user_mappings(
        self, platform: str | None = None, user_id: str = "default"
    ) -> list[UserIdMapping]:
        # Always use SQLite for list operations (authoritative)
        return await self._sqlite.list_user_mappings(platform, user_id)

    async def close(self) -> None:
        await self._sqlite.close()
        if self._redis:
            self._redis.close()


__all__ = [
    "RedisIntegrationStore",
]
