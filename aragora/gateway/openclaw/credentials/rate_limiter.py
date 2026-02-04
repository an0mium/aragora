"""
OpenClaw Credential Vault - Rate limiter.
"""

from __future__ import annotations

import threading
import time


class CredentialRateLimiter:
    """
    Rate limiter for credential access to prevent harvesting attacks.

    Uses a sliding window algorithm with per-user and per-tenant limits.
    """

    def __init__(
        self,
        max_per_minute: int = 30,
        max_per_hour: int = 200,
        lockout_duration_seconds: int = 300,
    ):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.lockout_duration = lockout_duration_seconds

        # Access tracking: key -> list of timestamps
        self._accesses: dict[str, list[float]] = {}
        # Lockout tracking: key -> lockout_until timestamp
        self._lockouts: dict[str, float] = {}
        self._lock = threading.Lock()

    def _get_key(self, user_id: str, tenant_id: str | None = None) -> str:
        """Generate rate limit key."""
        return f"{tenant_id or 'global'}:{user_id}"

    def check_rate_limit(self, user_id: str, tenant_id: str | None = None) -> tuple[bool, int]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        key = self._get_key(user_id, tenant_id)
        now = time.time()

        with self._lock:
            # Check lockout
            if key in self._lockouts:
                if now < self._lockouts[key]:
                    return False, int(self._lockouts[key] - now)
                else:
                    del self._lockouts[key]

            # Get access history
            accesses = self._accesses.get(key, [])

            # Clean old entries
            minute_ago = now - 60
            hour_ago = now - 3600
            accesses = [t for t in accesses if t > hour_ago]

            # Count recent accesses
            minute_count = sum(1 for t in accesses if t > minute_ago)
            hour_count = len(accesses)

            # Check limits
            if minute_count >= self.max_per_minute:
                self._lockouts[key] = now + self.lockout_duration
                return False, self.lockout_duration

            if hour_count >= self.max_per_hour:
                retry_after = int(accesses[0] + 3600 - now) + 1
                return False, retry_after

            # Record access
            accesses.append(now)
            self._accesses[key] = accesses

            return True, 0

    def clear_user(self, user_id: str, tenant_id: str | None = None) -> None:
        """Clear rate limit state for a user."""
        key = self._get_key(user_id, tenant_id)
        with self._lock:
            self._accesses.pop(key, None)
            self._lockouts.pop(key, None)
