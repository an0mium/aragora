"""
OAuth State Storage Backend.

Provides a pluggable storage backend for OAuth state tokens:
- Redis (recommended for production, multi-instance deployments)
- In-memory (fallback for development/single-instance)

The backend is selected based on REDIS_URL environment variable.
If Redis is unavailable, automatically falls back to in-memory storage.
"""

from __future__ import annotations

import binascii
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
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
    metadata: Optional[dict[str, Any]] = None  # Provider-specific data (tenant_id, org_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "redirect_url": self.redirect_url,
            "expires_at": self.expires_at,
            "created_at": self.created_at or time.time(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthState":
        """Create from dictionary."""
        return cls(
            user_id=data.get("user_id"),
            redirect_url=data.get("redirect_url"),
            expires_at=data.get("expires_at", 0.0),
            created_at=data.get("created_at", 0.0),
            metadata=data.get("metadata"),
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


class SQLiteOAuthStateStore(OAuthStateStore):
    """SQLite-backed OAuth state storage (persistent, single-instance).

    Provides persistence across restarts for single-instance deployments
    without requiring Redis. Uses thread-local connections for thread safety.
    """

    def __init__(self, db_path: str = "aragora_oauth.db", max_size: int = MAX_OAUTH_STATES):
        self._db_path = Path(db_path)
        self._max_size = max_size
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._init_lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS oauth_states (
                    state_token TEXT PRIMARY KEY,
                    user_id TEXT,
                    redirect_url TEXT,
                    expires_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_oauth_expires
                ON oauth_states(expires_at)
            """)
            # Add metadata column if it doesn't exist (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE oauth_states ADD COLUMN metadata TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()
            logger.info(f"SQLite OAuth state store initialized: {self._db_path}")

    def generate(
        self,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate and store a new state token."""
        # Cleanup before adding new state
        self.cleanup_expired()

        state_token = secrets.token_urlsafe(32)
        now = time.time()
        metadata_json = json.dumps(metadata) if metadata else None

        conn = self._get_connection()
        try:
            # Check if at max capacity
            cursor = conn.execute("SELECT COUNT(*) FROM oauth_states")
            count = cursor.fetchone()[0]

            if count >= self._max_size:
                # Evict oldest 10% of entries
                evict_count = max(1, count // 10)
                conn.execute(
                    """
                    DELETE FROM oauth_states WHERE state_token IN (
                        SELECT state_token FROM oauth_states
                        ORDER BY expires_at ASC LIMIT ?
                    )
                """,
                    (evict_count,),
                )
                logger.info(f"OAuth SQLite store: evicted {evict_count} oldest entries")

            # Insert new state
            conn.execute(
                """
                INSERT INTO oauth_states (state_token, user_id, redirect_url, expires_at, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (state_token, user_id, redirect_url, now + ttl_seconds, now, metadata_json),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"SQLite OAuth store generate failed: {e}")
            raise

        return state_token

    def validate_and_consume(self, state: str) -> Optional[OAuthState]:
        """Validate state token and remove it (single use)."""
        self.cleanup_expired()
        conn = self._get_connection()

        try:
            # Get and delete atomically
            cursor = conn.execute(
                """
                SELECT user_id, redirect_url, expires_at, created_at, metadata
                FROM oauth_states WHERE state_token = ?
            """,
                (state,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Delete the state
            conn.execute("DELETE FROM oauth_states WHERE state_token = ?", (state,))
            conn.commit()

            # Parse metadata JSON
            metadata = None
            if row[4]:
                try:
                    metadata = json.loads(row[4])
                except json.JSONDecodeError:
                    pass

            state_data = OAuthState(
                user_id=row[0],
                redirect_url=row[1],
                expires_at=row[2],
                created_at=row[3],
                metadata=metadata,
            )

            if state_data.is_expired:
                return None

            return state_data
        except Exception as e:
            logger.error(f"SQLite OAuth store validate failed: {e}")
            return None

    def cleanup_expired(self) -> int:
        """Remove expired states."""
        now = time.time()
        conn = self._get_connection()

        try:
            cursor = conn.execute("SELECT COUNT(*) FROM oauth_states WHERE expires_at < ?", (now,))
            count = cursor.fetchone()[0]

            if count > 0:
                conn.execute("DELETE FROM oauth_states WHERE expires_at < ?", (now,))
                conn.commit()
                logger.debug(f"OAuth SQLite store: cleaned up {count} expired states")

            return count
        except Exception as e:
            logger.error(f"SQLite OAuth store cleanup failed: {e}")
            return 0

    def size(self) -> int:
        """Get current number of stored states."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM oauth_states")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"SQLite OAuth store size query failed: {e}")
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "connection"):
            try:
                self._local.connection.close()
            except Exception:  # noqa: BLE001 - Cleanup must not raise
                pass
            delattr(self._local, "connection")


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
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate and store a new state token in Redis."""
        redis_client = self._get_redis()
        if not redis_client:
            raise RedisUnavailableError("OAuth state storage")

        state_token = secrets.token_urlsafe(32)
        now = time.time()

        state_data = OAuthState(
            user_id=user_id,
            redirect_url=redirect_url,
            expires_at=now + ttl_seconds,
            created_at=now,
            metadata=metadata,
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


class JWTOAuthStateStore(OAuthStateStore):
    """JWT-based OAuth state store that requires no server-side storage.

    Uses signed JWTs to encode state data, allowing state validation on any
    server instance without shared storage (Redis, SQLite, etc.).

    This is ideal for multi-instance deployments where shared storage is
    not available or practical.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """Initialize with a secret key for signing JWTs.

        Args:
            secret_key: Secret for signing. If not provided, uses OAUTH_JWT_SECRET
                       or ARAGORA_SECRET_KEY environment variables.
        """
        self._secret = secret_key or os.environ.get(
            "OAUTH_JWT_SECRET", os.environ.get("ARAGORA_SECRET_KEY", "")
        )
        if not self._secret:
            # Generate a random secret (will be different per instance, but that's OK
            # since we'll use this as fallback only when no persistent secret is set)
            import hashlib

            # Use a semi-stable secret based on machine identity
            machine_id = os.environ.get("HOSTNAME", "") + os.environ.get("USER", "")
            self._secret = hashlib.sha256(f"aragora-oauth-{machine_id}".encode()).hexdigest()
            logger.warning(
                "OAuth JWT store: No OAUTH_JWT_SECRET or ARAGORA_SECRET_KEY set. "
                "Using derived secret. For multi-instance deployments, set a shared secret."
            )
        self._used_nonces: set[str] = set()  # Simple replay protection
        self._nonce_cleanup_threshold = 10000

    def generate(
        self,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a signed JWT state token."""
        import base64
        import hashlib
        import hmac

        now = time.time()
        nonce = secrets.token_urlsafe(16)

        # Build payload
        payload: dict[str, Any] = {
            "n": nonce,  # Nonce for replay protection
            "u": user_id,  # User ID (if linking)
            "r": redirect_url,  # Redirect URL
            "e": now + ttl_seconds,  # Expiration timestamp
            "c": now,  # Created timestamp
        }
        if metadata:
            payload["m"] = metadata  # Provider-specific metadata

        # Encode payload as JSON, then base64
        payload_json = json.dumps(payload, separators=(",", ":"))
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()

        # Sign with HMAC-SHA256
        signature = hmac.new(self._secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

        # State = payload.signature
        return f"{payload_b64}.{sig_b64}"

    def validate_and_consume(self, state: str) -> Optional[OAuthState]:
        """Validate JWT state token and mark as consumed."""
        import base64
        import hashlib
        import hmac

        # Parse state
        parts = state.split(".")
        if len(parts) != 2:
            logger.debug("JWT state: invalid format (expected 2 parts)")
            return None

        payload_b64, sig_b64 = parts

        # Verify signature
        expected_sig = hmac.new(
            self._secret.encode(), payload_b64.encode(), hashlib.sha256
        ).digest()

        # Add padding back for base64 decode
        sig_b64_padded = sig_b64 + "=" * (4 - len(sig_b64) % 4)
        try:
            actual_sig = base64.urlsafe_b64decode(sig_b64_padded)
        except (ValueError, binascii.Error) as e:
            logger.debug(f"JWT state: invalid signature encoding: {e}")
            return None

        if not hmac.compare_digest(expected_sig, actual_sig):
            logger.debug("JWT state: signature mismatch")
            return None

        # Decode payload
        payload_b64_padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
        try:
            payload_json = base64.urlsafe_b64decode(payload_b64_padded).decode()
            payload = json.loads(payload_json)
        except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.debug(f"JWT state: payload decode failed: {e}")
            return None

        # Check expiration
        expires_at = payload.get("e", 0)
        if time.time() > expires_at:
            logger.debug("JWT state: expired")
            return None

        # Check replay (nonce already used)
        nonce = payload.get("n", "")
        if nonce in self._used_nonces:
            logger.warning("JWT state: replay detected (nonce reused)")
            return None

        # Mark nonce as used (simple in-memory tracking)
        self._used_nonces.add(nonce)

        # Cleanup old nonces periodically
        if len(self._used_nonces) > self._nonce_cleanup_threshold:
            # Just clear old nonces - expired states won't validate anyway
            self._used_nonces.clear()

        return OAuthState(
            user_id=payload.get("u"),
            redirect_url=payload.get("r"),
            expires_at=expires_at,
            created_at=payload.get("c", 0),
            metadata=payload.get("m"),
        )

    def cleanup_expired(self) -> int:
        """JWT states are self-expiring, no cleanup needed."""
        return 0

    def size(self) -> int:
        """Return count of tracked nonces (for replay protection)."""
        return len(self._used_nonces)


class FallbackOAuthStateStore(OAuthStateStore):
    """OAuth state store with automatic fallback chain: JWT -> Redis -> SQLite -> In-memory.

    Priority:
    1. JWT (stateless, works on any instance) - for multi-instance without shared storage
    2. Redis (if REDIS_URL configured and available) - for multi-instance deployments
    2. SQLite (persistent) - for single-instance with persistence across restarts
    3. In-memory (volatile) - last resort fallback
    """

    def __init__(
        self,
        redis_url: str = REDIS_URL,
        sqlite_path: str = "aragora_oauth.db",
        max_memory_size: int = MAX_OAUTH_STATES,
        use_sqlite: bool = True,
        use_jwt: bool = True,
    ):
        self._jwt_store: Optional[JWTOAuthStateStore] = None
        self._redis_store: Optional[RedisOAuthStateStore] = None
        self._sqlite_store: Optional[SQLiteOAuthStateStore] = None
        self._memory_store = InMemoryOAuthStateStore(max_size=max_memory_size)
        self._redis_url = redis_url
        self._use_redis = bool(redis_url)
        self._use_sqlite = use_sqlite
        self._use_jwt = use_jwt
        self._redis_failed = False
        self._sqlite_failed = False

        # JWT is the preferred backend for multi-instance deployments
        # It works without any shared storage
        if self._use_jwt:
            self._jwt_store = JWTOAuthStateStore()

        if self._use_redis:
            self._redis_store = RedisOAuthStateStore(redis_url)

        if self._use_sqlite:
            try:
                self._sqlite_store = SQLiteOAuthStateStore(
                    db_path=sqlite_path,
                    max_size=max_memory_size,
                )
            except Exception as e:
                logger.warning(f"SQLite OAuth store initialization failed: {e}")
                self._sqlite_failed = True

    def _get_active_store(self) -> OAuthStateStore:
        """Get the active storage backend."""
        # Try Redis first (best for multi-instance with proper shared state)
        if self._use_redis and not self._redis_failed and self._redis_store:
            try:
                redis_client = self._redis_store._get_redis()
                if redis_client:
                    return self._redis_store
            except Exception as e:
                logger.debug(f"Redis connectivity check failed: {e}")
            self._redis_failed = True
            logger.warning("OAuth state store: Redis unavailable, using JWT backend.")

        # Use JWT (stateless, works across all instances without shared storage)
        # This is the recommended fallback for multi-instance deployments
        if self._use_jwt and self._jwt_store:
            return self._jwt_store

        # Try SQLite (good for single-instance with persistence)
        if self._use_sqlite and not self._sqlite_failed and self._sqlite_store:
            return self._sqlite_store

        # Last resort: in-memory
        # SECURITY: In multi-instance mode, in-memory OAuth state will cause
        # OAuth flow failures when requests hit different instances
        import os

        is_multi_instance = os.environ.get("ARAGORA_MULTI_INSTANCE", "").lower() in (
            "true",
            "1",
            "yes",
        )
        if is_multi_instance:
            raise RuntimeError(
                "ARAGORA_MULTI_INSTANCE=true requires Redis or SQLite for OAuth state store. "
                "In-memory state will cause OAuth flow failures across instances. "
                "Configure REDIS_URL or ensure SQLite is available."
            )

        if not self._redis_failed or not self._use_sqlite:
            logger.warning(
                "OAuth state store: Using in-memory storage (volatile). "
                "OAuth state will be lost on restart and not shared across instances."
            )
        return self._memory_store

    @property
    def is_using_redis(self) -> bool:
        """Check if Redis is currently being used."""
        return self._use_redis and not self._redis_failed

    @property
    def is_using_jwt(self) -> bool:
        """Check if JWT is currently being used."""
        return self._use_jwt and self._jwt_store is not None and not self.is_using_redis

    @property
    def is_using_sqlite(self) -> bool:
        """Check if SQLite is currently being used."""
        return (
            self._use_sqlite
            and not self._sqlite_failed
            and not self.is_using_redis
            and not self.is_using_jwt
        )

    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        if self.is_using_redis:
            return "redis"
        if self.is_using_jwt:
            return "jwt"
        if self.is_using_sqlite:
            return "sqlite"
        return "memory"

    def generate(
        self,
        user_id: Optional[str] = None,
        redirect_url: Optional[str] = None,
        ttl_seconds: int = OAUTH_STATE_TTL_SECONDS,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate state using active backend."""
        store = self._get_active_store()
        store_type = type(store).__name__
        logger.info(
            f"OAuth state generate: backend={store_type}, use_jwt={self._use_jwt}, "
            f"jwt_store_exists={self._jwt_store is not None}, use_redis={self._use_redis}, "
            f"redis_failed={self._redis_failed}"
        )
        try:
            state = store.generate(user_id, redirect_url, ttl_seconds, metadata)
            logger.info(
                f"OAuth state generated: len={len(state)}, has_dot={'.' in state}, "
                f"prefix={state[:30]}..."
            )
            return state
        except Exception as e:
            if store is self._redis_store:
                logger.warning(f"Redis generate failed, using SQLite fallback: {e}")
                self._redis_failed = True
                # Try SQLite
                if self._use_sqlite and not self._sqlite_failed and self._sqlite_store:
                    try:
                        return self._sqlite_store.generate(user_id, redirect_url, ttl_seconds)
                    except Exception as sqlite_e:
                        logger.warning(f"SQLite generate failed, using memory: {sqlite_e}")
                        self._sqlite_failed = True
                return self._memory_store.generate(user_id, redirect_url, ttl_seconds)
            elif store is self._sqlite_store:
                logger.warning(f"SQLite generate failed, using memory fallback: {e}")
                self._sqlite_failed = True
                return self._memory_store.generate(user_id, redirect_url, ttl_seconds)
            raise

    def validate_and_consume(self, state: str) -> Optional[OAuthState]:
        """Validate state using active backend, checking fallbacks if needed."""
        store = self._get_active_store()
        try:
            result = store.validate_and_consume(state)
            if result:
                return result
        except Exception as e:
            if store is self._redis_store:
                logger.warning(f"Redis validate failed: {e}")
                self._redis_failed = True
            elif store is self._sqlite_store:
                logger.warning(f"SQLite validate failed: {e}")
                self._sqlite_failed = True
            elif store is self._jwt_store:
                logger.warning(f"JWT validate failed: {e}")
            else:
                logger.warning(f"OAuth store validate failed: {e}")

        # Check fallback stores for state created during previous backend's availability
        # This handles migration from old state formats to new JWT format
        if store is not self._sqlite_store and self._sqlite_store and not self._sqlite_failed:
            try:
                result = self._sqlite_store.validate_and_consume(state)
                if result:
                    logger.info("OAuth state validated from SQLite fallback (migration)")
                    return result
            except Exception as e:  # noqa: BLE001 - Fallback to next store
                logger.debug(f"SQLite OAuth store fallback: {e}")

        if store is not self._memory_store:
            try:
                result = self._memory_store.validate_and_consume(state)
                if result:
                    logger.info("OAuth state validated from memory fallback (migration)")
                    return result
            except Exception as e:  # noqa: BLE001 - Final fallback
                logger.debug(f"Memory OAuth store fallback: {e}")

        return None

    def cleanup_expired(self) -> int:
        """Cleanup expired states from all backends."""
        count = self._memory_store.cleanup_expired()
        if self._sqlite_store and not self._sqlite_failed:
            try:
                count += self._sqlite_store.cleanup_expired()
            except Exception as e:
                logger.debug(f"SQLite cleanup failed: {e}")
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

    def close(self) -> None:
        """Close resources."""
        if self._sqlite_store:
            try:
                self._sqlite_store.close()
            except Exception:  # noqa: BLE001 - Cleanup must not raise
                pass


# Global singleton
_oauth_state_store: Optional[FallbackOAuthStateStore] = None


def get_oauth_state_store(
    sqlite_path: str = "aragora_oauth.db",
    use_sqlite: bool = True,
    use_jwt: bool = True,
) -> FallbackOAuthStateStore:
    """Get the global OAuth state store instance.

    Args:
        sqlite_path: Path to SQLite database for persistent storage
        use_sqlite: Whether to use SQLite as fallback (default True)
        use_jwt: Whether to use JWT as primary backend (default True)

    Returns:
        Configured FallbackOAuthStateStore with JWT -> Redis -> SQLite -> memory fallback
    """
    global _oauth_state_store
    if _oauth_state_store is None:
        _oauth_state_store = FallbackOAuthStateStore(
            redis_url=REDIS_URL,
            sqlite_path=sqlite_path,
            use_sqlite=use_sqlite,
            use_jwt=use_jwt,
        )
        backend = _oauth_state_store.backend_name
        logger.info(f"OAuth state store initialized: {backend}")
    return _oauth_state_store


def reset_oauth_state_store() -> None:
    """Reset the global store (for testing)."""
    global _oauth_state_store
    if _oauth_state_store is not None:
        _oauth_state_store.close()
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
    logger.debug(
        f"Validating OAuth state (backend: {store.backend_name}, state_len: {len(state)}, has_dot: {'.' in state})"
    )
    result = store.validate_and_consume(state)
    if result is None:
        logger.debug(
            f"OAuth state validation failed (state_prefix: {state[:20] if len(state) > 20 else state}...)"
        )
        return None
    logger.debug("OAuth state validation succeeded")
    return result.to_dict()


__all__ = [
    "OAuthState",
    "OAuthStateStore",
    "InMemoryOAuthStateStore",
    "SQLiteOAuthStateStore",
    "RedisOAuthStateStore",
    "JWTOAuthStateStore",
    "FallbackOAuthStateStore",
    "get_oauth_state_store",
    "reset_oauth_state_store",
    "generate_oauth_state",
    "validate_oauth_state",
]
