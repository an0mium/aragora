"""
OAuth State Storage Backend.

Provides a pluggable storage backend for OAuth state tokens:
- Redis (recommended for production, multi-instance deployments)
- In-memory (fallback for development/single-instance)

The backend is selected based on REDIS_URL environment variable.
If Redis is unavailable, automatically falls back to in-memory storage.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from aragora.exceptions import RedisUnavailableError

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "")
OAUTH_STATE_TTL_SECONDS = int(os.environ.get("OAUTH_STATE_TTL_SECONDS", "600"))  # 10 min
MAX_OAUTH_STATES = int(os.environ.get("OAUTH_MAX_STATES", "10000"))


@dataclass
class OAuthState:
    """OAuth state data."""

    user_id: Optional[str]
    redirect_url: Optional[str]
    expires_at: float
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "redirect_url": self.redirect_url,
            "expires_at": self.expires_at,
            "created_at": self.created_at or time.time(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthState":
        """Create from dictionary."""
        return cls(
            user_id=data.get("user_id"),
            redirect_url=data.get("redirect_url"),
            expires_at=data.get("expires_at", 0.0),
            created_at=data.get("created_at", 0.0),
        )

    @property
    def is_expired(self) -> bool:
        """Check if state has expired."""
        return time.time() > self.expires_at


class OAuthStateStore(ABC):
    """Abstract base class for OAuth state storage."""

    @abstractmethod
    def generate(
        self,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
    ) -> str:
        """Generate and store a new state token."""
        pass

    @abstractmethod
    def validate_and_consume(self, state: str) -> Optional[OAuthState]:
        """Validate state token and remove it (single use)."""
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired states. Returns count of removed entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get current number of stored states."""
        pass


class InMemoryOAuthStateStore(OAuthStateStore):
    """In-memory OAuth state storage (single-instance only)."""

    def __init__(self, max_size: int = MAX_OAUTH_STATES):
        self._states: dict[str, OAuthState] = {}
        self._lock = threading.Lock()
        self._max_size = max_size

    def generate(
        self,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
    ) -> str:
        """Generate and store a new state token."""
        self.cleanup_expired()
        state_token = secrets.token_urlsafe(32)
        now = time.time()

        with self._lock:
            # Enforce max size - remove oldest entries if at capacity
            if len(self._states) >= self._max_size:
                sorted_states = sorted(self._states.items(), key=lambda x: x[1].expires_at)
                remove_count = max(1, len(sorted_states) // 10)
                for key, _ in sorted_states[:remove_count]:
                    del self._states[key]
                logger.info(f"OAuth state store: evicted {remove_count} oldest entries")

            self._states[state_token] = OAuthState(
                user_id=user_id,
                redirect_url=redirect_url,
                expires_at=now + ttl_seconds,
                created_at=now,
            )

        return state_token

    def validate_and_consume(self, state: str) -> Optional[OAuthState]:
        """Validate state token and remove it (single use)."""
        self.cleanup_expired()
        with self._lock:
            if state not in self._states:
                return None
            state_data = self._states.pop(state)
            if state_data.is_expired:
                return None
            return state_data

    def cleanup_expired(self) -> int:
        """Remove expired states."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._states.items() if v.expires_at < now]
            for k in expired:
                del self._states[k]
            return len(expired)

    def size(self) -> int:
        """Get current number of stored states."""
        with self._lock:
            return len(self._states)


class RedisOAuthStateStore(OAuthStateStore):
    """Redis-backed OAuth state storage (multi-instance safe)."""

    # Redis key prefix for OAuth states
    KEY_PREFIX = "aragora:oauth:state:"

    def __init__(self, redis_url: str = REDIS_URL):
        self._redis_url = redis_url
        self._redis: Optional[Any] = None
        self._connection_error_logged = False

    def _get_redis(self) -> Optional[Any]:
        """Get Redis connection with lazy initialization."""
        if self._redis is not None:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self._redis.ping()
            logger.info("OAuth state store: Connected to Redis")
            return self._redis
        except ImportError:
            if not self._connection_error_logged:
                logger.warning("OAuth state store: redis package not installed")
                self._connection_error_logged = True
            return None
        except Exception as e:
            if not self._connection_error_logged:
                logger.warning(f"OAuth state store: Redis connection failed: {e}")
                self._connection_error_logged = True
            return None

    def _key(self, state: str) -> str:
        """Get Redis key for state token."""
        return f"{self.KEY_PREFIX}{state}"

    def generate(
        self,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
    ) -> str:
        """Generate and store a new state token in Redis."""
        redis_client = self._get_redis()
        if not redis_client:
            raise RedisUnavailableError("OAuth state storage")

        import json

        state_token = secrets.token_urlsafe(32)
        now = time.time()

        state_data = OAuthState(
            user_id=user_id,
            redirect_url=redirect_url,
            expires_at=now + ttl_seconds,
            created_at=now,
        )

        # Store in Redis with TTL
        key = self._key(state_token)
        redis_client.setex(
            key,
            ttl_seconds,
            json.dumps(state_data.to_dict()),
        )

        return state_token

    def validate_and_consume(self, state: str) -> Optional[OAuthState]:
        """Validate and consume state token from Redis (atomic operation)."""
        redis_client = self._get_redis()
        if not redis_client:
            raise RedisUnavailableError("OAuth state storage")

        import json

        key = self._key(state)

        # Atomic get and delete using pipeline
        pipe = redis_client.pipeline()
        pipe.get(key)
        pipe.delete(key)
        results = pipe.execute()

        data_str = results[0]
        if not data_str:
            return None

        try:
            data = json.loads(data_str)
            state_data = OAuthState.from_dict(data)
            if state_data.is_expired:
                return None
            return state_data
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Invalid OAuth state data for {state[:16]}...")
            return None

    def cleanup_expired(self) -> int:
        """Redis handles TTL expiration automatically."""
        return 0

    def size(self) -> int:
        """Get approximate count of stored states."""
        redis_client = self._get_redis()
        if not redis_client:
            return 0

        try:
            cursor = 0
            count = 0
            while True:
                cursor, keys = redis_client.scan(
                    cursor=cursor,
                    match=f"{self.KEY_PREFIX}*",
                    count=100,
                )
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception as e:
            # Log but don't fail - size() is for metrics only
            logger.debug(f"Redis size() query failed: {e}")
            return 0


class FallbackOAuthStateStore(OAuthStateStore):
    """OAuth state store with automatic Redis fallback to in-memory."""

    def __init__(self, redis_url: str = REDIS_URL, max_memory_size: int = MAX_OAUTH_STATES):
        self._redis_store: Optional[RedisOAuthStateStore] = None
        self._memory_store = InMemoryOAuthStateStore(max_size=max_memory_size)
        self._redis_url = redis_url
        self._use_redis = bool(redis_url)
        self._redis_failed = False

        if self._use_redis:
            self._redis_store = RedisOAuthStateStore(redis_url)

    def _get_active_store(self) -> OAuthStateStore:
        """Get the active storage backend."""
        if self._use_redis and not self._redis_failed and self._redis_store:
            try:
                # Quick connectivity check
                redis_client = self._redis_store._get_redis()
                if redis_client:
                    return self._redis_store
            except Exception as e:
                logger.debug(f"Redis connectivity check failed: {e}")
            # Redis not available, fall back to memory
            self._redis_failed = True
            logger.warning(
                "OAuth state store: Redis unavailable, falling back to in-memory storage. "
                "This is not suitable for multi-instance deployments."
            )
        return self._memory_store

    @property
    def is_using_redis(self) -> bool:
        """Check if Redis is currently being used."""
        return self._use_redis and not self._redis_failed

    def generate(
        self,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
    ) -> str:
        """Generate state using active backend."""
        store = self._get_active_store()
        try:
            return store.generate(user_id, redirect_url, ttl_seconds)
        except Exception as e:
            if store is self._redis_store:
                logger.warning(f"Redis generate failed, using memory fallback: {e}")
                self._redis_failed = True
                return self._memory_store.generate(user_id, redirect_url, ttl_seconds)
            raise

    def validate_and_consume(self, state: str) -> Optional[OAuthState]:
        """Validate state using active backend."""
        store = self._get_active_store()
        try:
            return store.validate_and_consume(state)
        except Exception as e:
            if store is self._redis_store:
                logger.warning(f"Redis validate failed, checking memory fallback: {e}")
                # Also check memory store in case state was created during Redis failure
                return self._memory_store.validate_and_consume(state)
            raise

    def cleanup_expired(self) -> int:
        """Cleanup expired states."""
        count = self._memory_store.cleanup_expired()
        # Redis handles TTL automatically
        return count

    def size(self) -> int:
        """Get total stored states."""
        store = self._get_active_store()
        return store.size()

    def retry_redis(self) -> bool:
        """Attempt to reconnect to Redis."""
        if not self._use_redis or not self._redis_store:
            return False

        try:
            redis_client = self._redis_store._get_redis()
            if redis_client:
                redis_client.ping()
                self._redis_failed = False
                logger.info("OAuth state store: Reconnected to Redis")
                return True
        except Exception as e:
            logger.debug(f"Redis reconnection attempt failed: {e}")
        return False


# Global singleton
_oauth_state_store: Optional[FallbackOAuthStateStore] = None


def get_oauth_state_store() -> FallbackOAuthStateStore:
    """Get the global OAuth state store instance."""
    global _oauth_state_store
    if _oauth_state_store is None:
        _oauth_state_store = FallbackOAuthStateStore(redis_url=REDIS_URL)
        backend = "Redis" if _oauth_state_store.is_using_redis else "in-memory"
        logger.info(f"OAuth state store initialized: {backend}")
    return _oauth_state_store


def reset_oauth_state_store() -> None:
    """Reset the global store (for testing)."""
    global _oauth_state_store
    _oauth_state_store = None


# Convenience functions for backward compatibility
def generate_oauth_state(
    user_id: Optional[str] = None,
    redirect_url: Optional[str] = None,
) -> str:
    """Generate a new OAuth state token."""
    store = get_oauth_state_store()
    return store.generate(user_id, redirect_url)


def validate_oauth_state(state: str) -> Optional[dict[str, Any]]:
    """Validate and consume an OAuth state token.

    Returns dict with state data if valid, None otherwise.
    """
    store = get_oauth_state_store()
    result = store.validate_and_consume(state)
    if result is None:
        return None
    return result.to_dict()


__all__ = [
    "OAuthState",
    "OAuthStateStore",
    "InMemoryOAuthStateStore",
    "RedisOAuthStateStore",
    "FallbackOAuthStateStore",
    "get_oauth_state_store",
    "reset_oauth_state_store",
    "generate_oauth_state",
    "validate_oauth_state",
]
