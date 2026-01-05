"""
Authentication helpers for Aragora server.

Provides optional token-based access control for WebSocket subscriptions and API endpoints.
Tokens can be set via environment variables or runtime configuration.
"""

import hashlib
import hmac
import os
import secrets
import threading
import time
from typing import Optional, Dict, Any
from urllib.parse import parse_qs

from aragora.server.cors_config import cors_config


class AuthConfig:
    """Configuration for authentication."""

    def __init__(self):
        self.enabled = False
        self.api_token: Optional[str] = None
        self.token_ttl = 3600  # 1 hour default
        self.allowed_origins: list[str] = cors_config.get_origins_list()  # Centralized CORS
        # Rate limiting
        self.rate_limit_per_minute = 60  # Default requests per minute per token
        self.ip_rate_limit_per_minute = 120  # Default requests per minute per IP (more lenient)
        self._token_request_counts: Dict[str, list] = {}  # token -> timestamps
        self._ip_request_counts: Dict[str, list] = {}  # IP -> timestamps
        self._rate_limit_lock = threading.Lock()  # Thread-safe rate limiting
        self._max_tracked_entries = 10000  # Prevent memory exhaustion from rotating tokens/IPs
        # Token revocation tracking
        self._revoked_tokens: Dict[str, float] = {}  # token_hash -> revocation_time
        self._revocation_lock = threading.Lock()
        self._max_revoked_tokens = 10000  # Limit stored revocations

    def configure_from_env(self):
        """Configure from environment variables."""
        self.api_token = os.getenv("ARAGORA_API_TOKEN")
        if self.api_token:
            self.enabled = True

        ttl_str = os.getenv("ARAGORA_TOKEN_TTL", "3600")
        try:
            self.token_ttl = int(ttl_str)
        except ValueError:
            pass

        origins = os.getenv("ARAGORA_ALLOWED_ORIGINS")
        if origins:
            self.allowed_origins = [o.strip() for o in origins.split(",")]

    def generate_token(self, loop_id: str = "", expires_in: int = None) -> str:
        """Generate a signed token for access."""
        if not self.api_token:
            return ""

        if expires_in is None:
            expires_in = self.token_ttl

        expires = int(time.time()) + expires_in
        payload = f"{loop_id}:{expires}"
        signature = hmac.new(
            self.api_token.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload}:{signature}"

    def revoke_token(self, token: str, reason: str = "") -> bool:
        """Revoke a token to prevent further use.

        Uses truncated hash to minimize storage while preventing timing attacks.

        Args:
            token: The token to revoke
            reason: Optional reason for revocation (logged but not stored)

        Returns:
            True if revoked successfully
        """
        if not token:
            return False

        # Use truncated hash for storage efficiency
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]

        with self._revocation_lock:
            # Clean up expired revocations to prevent unbounded growth
            if len(self._revoked_tokens) >= self._max_revoked_tokens:
                # Remove oldest 10% of entries
                sorted_items = sorted(self._revoked_tokens.items(), key=lambda x: x[1])
                for key, _ in sorted_items[:len(sorted_items) // 10]:
                    del self._revoked_tokens[key]

            self._revoked_tokens[token_hash] = time.time()
            return True

    def is_revoked(self, token: str) -> bool:
        """Check if a token has been revoked.

        Args:
            token: The token to check

        Returns:
            True if token is revoked
        """
        if not token:
            return False

        token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]

        with self._revocation_lock:
            return token_hash in self._revoked_tokens

    def get_revocation_count(self) -> int:
        """Get the number of revoked tokens being tracked."""
        with self._revocation_lock:
            return len(self._revoked_tokens)

    def validate_token(self, token: str, loop_id: str = "") -> bool:
        """Validate a token."""
        if not self.api_token or not token:
            return not self.enabled  # If auth disabled, allow; if enabled, require token

        # Check revocation first (before expensive crypto operations)
        if self.is_revoked(token):
            return False

        try:
            payload, signature = token.rsplit(":", 1)
            loop_part, expires_str = payload.rsplit(":", 1)
            expires = int(expires_str)

            # Check expiration
            if time.time() > expires:
                return False

            # Verify signature
            expected = hmac.new(
                self.api_token.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected):
                return False

            # Check loop_id if specified
            if loop_id and loop_part != loop_id:
                return False

            return True

        except (ValueError, IndexError):
            return False

    def _cleanup_stale_entries(self, entries_dict: Dict[str, list], window_start: float) -> None:
        """Remove stale entries to prevent memory exhaustion.

        Called within the rate limit lock.
        """
        # Remove entries with no recent requests
        stale_keys = [k for k, v in entries_dict.items() if not v or max(v) < window_start]
        for k in stale_keys:
            del entries_dict[k]

        # If still too large, evict oldest entries (LRU-style)
        if len(entries_dict) > self._max_tracked_entries:
            # Sort by most recent request time
            sorted_keys = sorted(entries_dict.keys(), key=lambda k: max(entries_dict[k]) if entries_dict[k] else 0)
            # Remove oldest 10%
            to_remove = len(sorted_keys) // 10
            for k in sorted_keys[:to_remove]:
                del entries_dict[k]

    def check_rate_limit(self, token: str) -> tuple:
        """Check if token is within rate limit.

        Uses sliding window algorithm with 1-minute window.

        Args:
            token: The token to check

        Returns:
            (allowed, remaining_requests) tuple
        """
        now = time.time()
        window_start = now - 60  # 1 minute window

        with self._rate_limit_lock:
            # Periodic cleanup to prevent memory exhaustion
            if len(self._token_request_counts) > self._max_tracked_entries * 0.9:
                self._cleanup_stale_entries(self._token_request_counts, window_start)

            # Get or create request list for this token
            if token not in self._token_request_counts:
                self._token_request_counts[token] = []

            # Remove old requests outside window
            self._token_request_counts[token] = [
                t for t in self._token_request_counts[token] if t > window_start
            ]

            # Check limit
            current_count = len(self._token_request_counts[token])
            if current_count >= self.rate_limit_per_minute:
                return False, 0

            # Record this request
            self._token_request_counts[token].append(now)
            return True, self.rate_limit_per_minute - current_count - 1

    def check_rate_limit_by_ip(self, ip_address: str) -> tuple:
        """Check if IP address is within rate limit.

        Provides DoS protection even when auth is disabled.
        Uses sliding window algorithm with 1-minute window.

        Args:
            ip_address: The client IP address

        Returns:
            (allowed, remaining_requests) tuple
        """
        if not ip_address:
            return True, self.ip_rate_limit_per_minute

        now = time.time()
        window_start = now - 60  # 1 minute window

        with self._rate_limit_lock:
            # Periodic cleanup to prevent memory exhaustion
            if len(self._ip_request_counts) > self._max_tracked_entries * 0.9:
                self._cleanup_stale_entries(self._ip_request_counts, window_start)

            # Get or create request list for this IP
            if ip_address not in self._ip_request_counts:
                self._ip_request_counts[ip_address] = []

            # Remove old requests outside window
            self._ip_request_counts[ip_address] = [
                t for t in self._ip_request_counts[ip_address] if t > window_start
            ]

            # Check limit
            current_count = len(self._ip_request_counts[ip_address])
            if current_count >= self.ip_rate_limit_per_minute:
                return False, 0

            # Record this request
            self._ip_request_counts[ip_address].append(now)
            return True, self.ip_rate_limit_per_minute - current_count - 1

    def extract_token_from_request(self, headers: Dict[str, str], query_params: Dict[str, list]) -> Optional[str]:
        """Extract token from Authorization header or query params."""
        # Check Authorization header
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Check query params
        tokens = query_params.get("token", [])
        if tokens:
            return tokens[0]

        return None


# Global auth config instance
auth_config = AuthConfig()
auth_config.configure_from_env()


def check_auth(headers: Dict[str, Any], query_string: str = "", loop_id: str = "", ip_address: str = "") -> tuple:
    """
    Check authentication and rate limiting for a request.

    Args:
        headers: HTTP headers dict
        query_string: Raw query string
        loop_id: Optional loop ID to validate against token
        ip_address: Client IP for rate limiting (used even when auth disabled)

    Returns:
        (authenticated, rate_limit_remaining) tuple.
        authenticated is True if authenticated or auth disabled.
        rate_limit_remaining is -1 if rate limiting not applicable.
    """
    # Always check IP rate limit for DoS protection (even without auth)
    if ip_address:
        ip_allowed, ip_remaining = auth_config.check_rate_limit_by_ip(ip_address)
        if not ip_allowed:
            return False, 0

    if not auth_config.enabled:
        # Return IP remaining if we have it, else -1
        return True, ip_remaining if ip_address else -1

    query_params = parse_qs(query_string.lstrip("?")) if query_string else {}
    token = auth_config.extract_token_from_request(headers, query_params)

    if not auth_config.validate_token(token or "", loop_id):
        return False, -1

    # Check token-based rate limit
    allowed, remaining = auth_config.check_rate_limit(token or "anonymous")
    if not allowed:
        return False, 0

    # Return the more restrictive limit
    if ip_address:
        return True, min(remaining, ip_remaining)
    return True, remaining


def generate_shareable_link(base_url: str, loop_id: str, expires_in: int = 3600) -> str:
    """Generate a shareable link with embedded token."""
    token = auth_config.generate_token(loop_id, expires_in)
    if not token:
        return base_url

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}token={token}"