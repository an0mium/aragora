"""
In-memory integration store backend.

Thread-safe in-memory storage suitable for development and testing.
"""

from __future__ import annotations

import threading
import time

from aragora.storage.integration_backends import IntegrationStoreBackend
from aragora.storage.integration_models import (
    IntegrationConfig,
    UserIdMapping,
    _make_key,
)


class InMemoryIntegrationStore(IntegrationStoreBackend):
    """
    Thread-safe in-memory integration store.

    Fast but not shared across restarts. Suitable for development/testing.
    """

    def __init__(self) -> None:
        self._store: dict[str, IntegrationConfig] = {}
        self._mappings: dict[str, UserIdMapping] = {}
        self._lock = threading.RLock()

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> IntegrationConfig | None:
        key = _make_key(integration_type, user_id)
        with self._lock:
            return self._store.get(key)

    async def save(self, config: IntegrationConfig) -> None:
        key = _make_key(config.type, config.user_id or "default")
        config.updated_at = time.time()
        with self._lock:
            self._store[key] = config

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        key = _make_key(integration_type, user_id)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def list_for_user(self, user_id: str = "default") -> list[IntegrationConfig]:
        prefix = f"{user_id}:"
        with self._lock:
            return [v for k, v in self._store.items() if k.startswith(prefix)]

    async def list_all(self, limit: int = 1000) -> list[IntegrationConfig]:
        with self._lock:
            return list(self._store.values())[:limit]

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> UserIdMapping | None:
        key = f"{user_id}:{platform}:{email}"
        with self._lock:
            return self._mappings.get(key)

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        key = f"{mapping.user_id}:{mapping.platform}:{mapping.email}"
        mapping.updated_at = time.time()
        with self._lock:
            self._mappings[key] = mapping

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        key = f"{user_id}:{platform}:{email}"
        with self._lock:
            if key in self._mappings:
                del self._mappings[key]
                return True
            return False

    async def list_user_mappings(
        self, platform: str | None = None, user_id: str = "default"
    ) -> list[UserIdMapping]:
        with self._lock:
            result = []
            for key, mapping in self._mappings.items():
                if mapping.user_id == user_id:
                    if platform is None or mapping.platform == platform:
                        result.append(mapping)
            return result

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._store.clear()
            self._mappings.clear()


__all__ = [
    "InMemoryIntegrationStore",
]
