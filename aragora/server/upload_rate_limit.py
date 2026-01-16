"""
Upload rate limiting for DoS protection.

This module provides IP-based upload rate limiting using sliding windows
to prevent abuse of file upload endpoints.
"""

import logging
import threading
import time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING, Optional, Set

if TYPE_CHECKING:
    from http.server import BaseHTTPRequestHandler

logger = logging.getLogger(__name__)


class UploadRateLimiter:
    """IP-based upload rate limiter using sliding window.

    Features:
    - Per-minute and per-hour limits
    - LRU eviction to prevent unbounded memory growth
    - Thread-safe with lock protection
    - Deques with maxlen for bounded memory per IP

    Usage:
        limiter = UploadRateLimiter()

        # In request handler:
        if not limiter.check_allowed(client_ip):
            return 429_response
    """

    def __init__(
        self,
        max_per_minute: int = 5,
        max_per_hour: int = 30,
        max_tracked_ips: int = 10000,
        trusted_proxies: Optional[Set[str]] = None,
    ):
        """Initialize rate limiter.

        Args:
            max_per_minute: Maximum uploads per IP per minute
            max_per_hour: Maximum uploads per IP per hour
            max_tracked_ips: Maximum IPs to track (LRU eviction beyond this)
            trusted_proxies: Set of trusted proxy IPs for X-Forwarded-For
        """
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.max_tracked_ips = max_tracked_ips
        self.max_timestamps = max_per_hour  # Match hourly limit

        # IP -> deque of upload timestamps
        self._upload_counts: "OrderedDict[str, deque]" = OrderedDict()
        self._lock = threading.Lock()

        # Trusted proxies (default: localhost)
        self._trusted_proxies = trusted_proxies or frozenset({"127.0.0.1", "::1", "localhost"})

    def get_client_ip(self, handler: "BaseHTTPRequestHandler") -> str:
        """Extract client IP from request, respecting trusted proxy headers.

        Args:
            handler: HTTP request handler with client_address and headers

        Returns:
            Client IP address string
        """
        remote_ip = handler.client_address[0] if hasattr(handler, "client_address") else "unknown"
        client_ip = remote_ip

        # Only trust X-Forwarded-For from trusted proxies
        if remote_ip in self._trusted_proxies:
            forwarded = handler.headers.get("X-Forwarded-For", "")
            if forwarded:
                first_ip = forwarded.split(",")[0].strip()
                if first_ip:
                    client_ip = first_ip

        return client_ip

    def check_allowed(self, client_ip: str) -> tuple[bool, Optional[dict]]:
        """Check if upload is allowed for the given IP.

        Args:
            client_ip: Client IP address

        Returns:
            Tuple of (allowed: bool, error_info: Optional[dict])
            error_info contains 'message' and 'retry_after' if not allowed
        """
        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        with self._lock:
            # Get or create upload history for this IP (bounded deque)
            if client_ip not in self._upload_counts:
                self._upload_counts[client_ip] = deque(maxlen=self.max_timestamps)
            else:
                # Move to end for LRU tracking (most recently accessed)
                self._upload_counts.move_to_end(client_ip)

            timestamps = self._upload_counts[client_ip]

            # Clean up old entries (rebuild deque with only recent timestamps)
            recent_timestamps = [ts for ts in timestamps if ts > one_hour_ago]
            timestamps.clear()
            timestamps.extend(recent_timestamps)

            # LRU eviction: enforce max tracked IPs limit
            while len(self._upload_counts) > self.max_tracked_ips:
                # Remove oldest (first) entry - LRU eviction
                self._upload_counts.popitem(last=False)

            # Periodically clean up stale IPs (those with no recent uploads)
            if len(self._upload_counts) > 100:
                stale_ips = [
                    ip
                    for ip, ts_deque in self._upload_counts.items()
                    if not ts_deque or max(ts_deque) < one_hour_ago
                ]
                for ip in stale_ips:
                    del self._upload_counts[ip]

            # Check per-minute limit
            recent_minute = sum(1 for ts in timestamps if ts > one_minute_ago)
            if recent_minute >= self.max_per_minute:
                return False, {
                    "message": f"Upload rate limit exceeded. Max {self.max_per_minute} uploads per minute.",
                    "retry_after": 60,
                }

            # Check per-hour limit
            if len(timestamps) >= self.max_per_hour:
                return False, {
                    "message": f"Upload rate limit exceeded. Max {self.max_per_hour} uploads per hour.",
                    "retry_after": 3600,
                }

            # Record this upload
            timestamps.append(now)

        return True, None


# Global upload rate limiter instance
_upload_limiter: Optional[UploadRateLimiter] = None


def get_upload_limiter() -> UploadRateLimiter:
    """Get or create the global upload rate limiter."""
    global _upload_limiter
    if _upload_limiter is None:
        import os

        trusted_proxies = set(
            p.strip()
            for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
        )
        _upload_limiter = UploadRateLimiter(trusted_proxies=trusted_proxies)
    return _upload_limiter


__all__ = ["UploadRateLimiter", "get_upload_limiter"]
