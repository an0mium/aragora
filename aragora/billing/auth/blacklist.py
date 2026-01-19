"""
Token Blacklist for JWT Revocation.

Provides thread-safe token blacklisting with TTL cleanup.
Supports both in-memory and persistent (SQLite/Redis) backends.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TokenBlacklist:
    """
    Thread-safe in-memory token blacklist with TTL cleanup.

    Tokens are stored with their expiry time and automatically cleaned up
    when they would have expired anyway. This ensures the blacklist doesn't
    grow unbounded.

    For production deployments with multiple instances, consider using Redis
    or a shared database for the blacklist.
    """

    _instance: Optional["TokenBlacklist"] = None
    _lock = threading.Lock()
    _initialized: bool
    _blacklist: dict[str, float]
    _data_lock: threading.RLock  # RLock allows reentrant acquisition
    _cleanup_interval: int
    _last_cleanup: float

    def __new__(cls) -> "TokenBlacklist":
        """Singleton pattern for global blacklist."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cleanup_interval: int = 300):
        """
        Initialize the blacklist.

        Args:
            cleanup_interval: Seconds between automatic cleanups (default 5 min)
        """
        if self._initialized:
            return
        self._blacklist: dict[str, float] = {}  # token_jti -> expiry_timestamp
        self._data_lock = threading.RLock()  # RLock for reentrant locking
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._initialized = True
        logger.info("token_blacklist_initialized")

    def revoke(self, token_jti: str, expires_at: float) -> None:
        """
        Add a token to the blacklist.

        Args:
            token_jti: Token's unique identifier (jti claim or hash of token)
            expires_at: When the token would naturally expire (Unix timestamp)
        """
        with self._data_lock:
            self._blacklist[token_jti] = expires_at
            logger.info(f"token_revoked jti={token_jti[:16]}...")
            self._maybe_cleanup()

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by decoding and blacklisting it.

        Args:
            token: The JWT token string

        Returns:
            True if token was valid and revoked, False otherwise
        """
        from .tokens import decode_jwt

        payload = decode_jwt(token)
        if payload is None:
            return False
        # Use a hash of the token as the JTI if no jti claim
        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        self.revoke(token_jti, payload.exp)
        return True

    def is_revoked(self, token: str) -> bool:
        """
        Check if a token has been revoked.

        Args:
            token: The JWT token string

        Returns:
            True if token is in blacklist, False otherwise
        """
        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        with self._data_lock:
            return token_jti in self._blacklist

    def cleanup_expired(self) -> int:
        """
        Remove expired tokens from the blacklist.

        Returns:
            Number of tokens removed
        """
        now = time.time()
        with self._data_lock:
            expired = [k for k, v in self._blacklist.items() if v < now]
            for k in expired:
                del self._blacklist[k]
            if expired:
                logger.debug(f"token_blacklist_cleanup removed={len(expired)}")
            self._last_cleanup = now
            return len(expired)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()

    def size(self) -> int:
        """Get current blacklist size."""
        with self._data_lock:
            return len(self._blacklist)

    def clear(self) -> None:
        """Clear all revoked tokens (for testing)."""
        with self._data_lock:
            self._blacklist.clear()
            logger.info("token_blacklist_cleared")


# Global blacklist instance
_token_blacklist: Optional[TokenBlacklist] = None


def get_token_blacklist() -> TokenBlacklist:
    """Get the global token blacklist instance."""
    global _token_blacklist
    if _token_blacklist is None:
        _token_blacklist = TokenBlacklist()
    return _token_blacklist


def get_persistent_blacklist():
    """
    Get the persistent blacklist backend.

    This is the preferred method for production. Uses SQLite by default,
    or Redis for multi-instance deployments.

    Returns:
        BlacklistBackend instance (SQLite, Redis, or in-memory)
    """
    from aragora.storage.token_blacklist_store import get_blacklist_backend

    return get_blacklist_backend()


def revoke_token_persistent(token: str) -> bool:
    """
    Revoke a token using the persistent blacklist backend.

    Args:
        token: The JWT token string

    Returns:
        True if token was valid and revoked, False otherwise
    """
    from .tokens import decode_jwt

    payload = decode_jwt(token)
    if payload is None:
        return False
    token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
    backend = get_persistent_blacklist()
    backend.add(token_jti, payload.exp)
    logger.info(f"token_revoked_persistent jti={token_jti[:16]}...")
    return True


def is_token_revoked_persistent(token: str) -> bool:
    """
    Check if a token has been revoked using persistent backend.

    Args:
        token: The JWT token string

    Returns:
        True if token is revoked
    """
    token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
    backend = get_persistent_blacklist()
    return backend.contains(token_jti)


__all__ = [
    "TokenBlacklist",
    "get_token_blacklist",
    "get_persistent_blacklist",
    "revoke_token_persistent",
    "is_token_revoked_persistent",
]
