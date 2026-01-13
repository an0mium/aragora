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

from aragora.config import TOKEN_TTL_SECONDS, DEFAULT_RATE_LIMIT, IP_RATE_LIMIT, SHAREABLE_LINK_TTL, get_settings
from aragora.server.cors_config import cors_config
from aragora.exceptions import AuthenticationError


class AuthConfig:
    """Configuration for authentication."""

    # Cleanup interval in seconds (5 minutes)
    CLEANUP_INTERVAL_SECONDS = 300

    def __init__(self):
        self.enabled = False
        self.api_token: Optional[str] = None
        self.token_ttl = TOKEN_TTL_SECONDS
        self.allowed_origins: list[str] = cors_config.get_origins_list()  # Centralized CORS
        # Rate limiting
        self.rate_limit_per_minute = DEFAULT_RATE_LIMIT
        self.ip_rate_limit_per_minute = IP_RATE_LIMIT
        self._token_request_counts: Dict[str, list] = {}  # token -> timestamps
        self._ip_request_counts: Dict[str, list] = {}  # IP -> timestamps
        self._rate_limit_lock = threading.Lock()  # Thread-safe rate limiting
        # Get limits from config
        auth_settings = get_settings().auth
        self._max_tracked_entries = auth_settings.max_tracked_entries
        self._rate_limit_window = auth_settings.rate_limit_window
        self._revoked_token_ttl = auth_settings.revoked_token_ttl
        # Token revocation tracking
        self._revoked_tokens: Dict[str, float] = {}  # token_hash -> revocation_time
        self._revocation_lock = threading.Lock()
        self._max_revoked_tokens = auth_settings.max_revoked_tokens
        # Background cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stop_event = threading.Event()
        # Shareable session storage (session_id -> {token, expires_at, loop_id})
        # Sessions replace direct token-in-URL for security (avoids exposure in logs/referrer)
        self._shareable_sessions: Dict[str, Dict[str, Any]] = {}
        self._session_lock = threading.Lock()
        self._max_sessions = 10000  # Prevent unbounded growth
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread.

        The thread runs every CLEANUP_INTERVAL_SECONDS to prevent
        unbounded memory growth from rate limit tracking.
        """
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return

        self._cleanup_stop_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="auth-cleanup",
            daemon=True,  # Don't block process exit
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        """Background loop that periodically cleans up stale entries."""
        import logging
        _logger = logging.getLogger(__name__)

        while not self._cleanup_stop_event.is_set():
            # Wait for the interval or until stop is signaled
            if self._cleanup_stop_event.wait(timeout=self.CLEANUP_INTERVAL_SECONDS):
                break  # Stop event was set

            try:
                stats = self.cleanup_expired_entries()
                total_removed = (
                    stats["token_entries_removed"]
                    + stats["ip_entries_removed"]
                    + stats["revoked_tokens_removed"]
                    + stats.get("sessions_removed", 0)
                )
                if total_removed > 0:
                    _logger.debug(
                        f"auth_cleanup removed={total_removed} "
                        f"tokens={stats['token_entries_removed']} "
                        f"ips={stats['ip_entries_removed']} "
                        f"revoked={stats['revoked_tokens_removed']} "
                        f"sessions={stats.get('sessions_removed', 0)}"
                    )
            except Exception as e:
                _logger.warning(f"auth_cleanup_error error={e}")

    def stop_cleanup_thread(self) -> None:
        """Stop the background cleanup thread.

        Call this during graceful shutdown.
        """
        self._cleanup_stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=2.0)
            self._cleanup_thread = None

    def configure_from_env(self):
        """Configure from environment variables.

        Auth is enabled when ARAGORA_API_TOKEN is set.
        In production mode (ARAGORA_ENV=production), auth is mandatory -
        the server will refuse to start without a configured API token.
        """
        import logging
        _logger = logging.getLogger(__name__)

        self.api_token = os.getenv("ARAGORA_API_TOKEN")
        env_mode = os.getenv("ARAGORA_ENV", "development").lower()
        is_production = env_mode == "production"

        if self.api_token:
            self.enabled = True
            _logger.info("Authentication enabled (API token configured)")
        elif is_production:
            # In production, auth is mandatory
            _logger.error(
                "SECURITY ERROR: ARAGORA_ENV=production but ARAGORA_API_TOKEN not set. "
                "Authentication is required in production mode. "
                "Set ARAGORA_API_TOKEN or use ARAGORA_ENV=development for testing."
            )
            raise AuthenticationError(
                "Authentication required in production mode. "
                "Set ARAGORA_API_TOKEN environment variable."
            )
        else:
            _logger.warning(
                "Authentication disabled (no API token). "
                "Set ARAGORA_API_TOKEN for access control."
            )

        ttl_str = os.getenv("ARAGORA_TOKEN_TTL", "3600")
        try:
            self.token_ttl = int(ttl_str)
        except ValueError as e:
            _logger.warning(f"Invalid ARAGORA_TOKEN_TTL '{ttl_str}', using default: {e}")

        origins = os.getenv("ARAGORA_ALLOWED_ORIGINS")
        if origins:
            self.allowed_origins = [o.strip() for o in origins.split(",")]

    def generate_token(self, loop_id: str = "", expires_in: int | None = None) -> str:
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

    def extract_loop_id_from_token(self, token: str) -> Optional[str]:
        """Extract the loop_id embedded in a token.

        Args:
            token: The signed token

        Returns:
            The loop_id from the token, or None if extraction fails
        """
        if not token:
            return None

        try:
            # Token format: "loop_id:expires:signature"
            payload, _ = token.rsplit(":", 1)
            loop_part, _ = payload.rsplit(":", 1)
            return loop_part if loop_part else None
        except (ValueError, IndexError):
            return None

    def generate_session(self, loop_id: str, expires_in: int | None = None) -> str:
        """Generate a secure session ID for shareable links.

        Sessions store the actual token server-side, preventing token exposure
        in URLs, browser history, logs, and referrer headers.

        Args:
            loop_id: The loop_id to associate with this session
            expires_in: Session TTL in seconds (default: SHAREABLE_LINK_TTL)

        Returns:
            A short, URL-safe session ID
        """
        if expires_in is None:
            expires_in = SHAREABLE_LINK_TTL

        # Generate cryptographically secure session ID (URL-safe, 16 bytes = 22 chars base64)
        session_id = secrets.token_urlsafe(16)
        expires_at = time.time() + expires_in

        # Generate the actual token (stored server-side only)
        token = self.generate_token(loop_id, expires_in)

        with self._session_lock:
            # Enforce max sessions - evict oldest if at capacity
            if len(self._shareable_sessions) >= self._max_sessions:
                # Remove expired sessions first
                now = time.time()
                expired = [k for k, v in self._shareable_sessions.items() if v["expires_at"] < now]
                for k in expired:
                    del self._shareable_sessions[k]

                # If still at capacity, remove oldest 10%
                if len(self._shareable_sessions) >= self._max_sessions:
                    sorted_sessions = sorted(
                        self._shareable_sessions.items(),
                        key=lambda x: x[1]["expires_at"]
                    )
                    for k, _ in sorted_sessions[: len(sorted_sessions) // 10]:
                        del self._shareable_sessions[k]

            self._shareable_sessions[session_id] = {
                "token": token,
                "expires_at": expires_at,
                "loop_id": loop_id,
                "created_at": time.time(),
            }

        return session_id

    def resolve_session(self, session_id: str) -> tuple[bool, str, str]:
        """Resolve a session ID to its token and loop_id.

        Args:
            session_id: The session ID from the URL

        Returns:
            Tuple of (is_valid, token, loop_id)
            If invalid/expired, returns (False, "", "")
        """
        if not session_id:
            return False, "", ""

        with self._session_lock:
            session = self._shareable_sessions.get(session_id)
            if not session:
                return False, "", ""

            # Check expiration
            if time.time() > session["expires_at"]:
                # Clean up expired session
                del self._shareable_sessions[session_id]
                return False, "", ""

            return True, session["token"], session["loop_id"]

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a shareable session.

        Args:
            session_id: The session ID to revoke

        Returns:
            True if session was found and revoked
        """
        with self._session_lock:
            if session_id in self._shareable_sessions:
                del self._shareable_sessions[session_id]
                return True
            return False

    def get_session_count(self) -> int:
        """Get the number of active shareable sessions."""
        with self._session_lock:
            return len(self._shareable_sessions)

    def validate_token_for_loop(self, token: str, loop_id: str) -> tuple[bool, str]:
        """Validate a token specifically for a given loop_id.

        This ensures the token was generated for the specific loop being accessed,
        preventing cross-loop access attacks.

        Args:
            token: The token to validate
            loop_id: The loop_id being accessed

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enabled:
            return True, ""

        if not token:
            return False, "Authentication required"

        if not loop_id:
            return False, "Loop ID required"

        # Check revocation
        if self.is_revoked(token):
            return False, "Token has been revoked"

        # Full token validation with loop_id
        if not self.validate_token(token, loop_id):
            # Get more specific error
            token_loop_id = self.extract_loop_id_from_token(token)
            if token_loop_id and token_loop_id != loop_id:
                return False, f"Token not authorized for loop {loop_id}"
            return False, "Invalid or expired token"

        return True, ""

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

    def cleanup_expired_entries(self, ttl_seconds: int = 3600) -> dict:
        """
        Proactively clean up expired rate limit entries.

        This method can be called periodically (e.g., every 5 minutes)
        to prevent memory from growing unbounded between requests.

        Args:
            ttl_seconds: Remove entries older than this (default: 1 hour)

        Returns:
            Dict with cleanup statistics
        """
        cutoff = time.time() - ttl_seconds
        stats = {"token_entries_removed": 0, "ip_entries_removed": 0, "revoked_tokens_removed": 0}

        with self._rate_limit_lock:
            # Clean token request counts
            keys_to_remove = []
            for key, timestamps in self._token_request_counts.items():
                # Remove old timestamps
                self._token_request_counts[key] = [t for t in timestamps if t > cutoff]
                # Mark for removal if empty
                if not self._token_request_counts[key]:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._token_request_counts[key]
                stats["token_entries_removed"] += 1

            # Clean IP request counts
            keys_to_remove = []
            for key, timestamps in self._ip_request_counts.items():
                self._ip_request_counts[key] = [t for t in timestamps if t > cutoff]
                if not self._ip_request_counts[key]:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._ip_request_counts[key]
                stats["ip_entries_removed"] += 1

        # Clean old revoked tokens based on configured TTL
        revoke_cutoff = time.time() - self._revoked_token_ttl
        with self._revocation_lock:
            keys_to_remove = [k for k, v in self._revoked_tokens.items() if v < revoke_cutoff]
            for key in keys_to_remove:
                del self._revoked_tokens[key]
                stats["revoked_tokens_removed"] += 1

        # Clean expired shareable sessions
        stats["sessions_removed"] = 0
        now = time.time()
        with self._session_lock:
            keys_to_remove = [k for k, v in self._shareable_sessions.items() if v["expires_at"] < now]
            for key in keys_to_remove:
                del self._shareable_sessions[key]
                stats["sessions_removed"] += 1

        return stats

    def get_rate_limit_stats(self) -> dict:
        """Get current rate limiting statistics for monitoring."""
        with self._rate_limit_lock:
            stats = {
                "token_entries": len(self._token_request_counts),
                "ip_entries": len(self._ip_request_counts),
                "revoked_tokens": len(self._revoked_tokens),
                "max_tracked_entries": self._max_tracked_entries,
            }
        with self._session_lock:
            stats["shareable_sessions"] = len(self._shareable_sessions)
        return stats

    def check_rate_limit(self, token: str) -> tuple:
        """Check if token is within rate limit.

        Uses sliding window algorithm with 1-minute window.

        Args:
            token: The token to check

        Returns:
            (allowed, remaining_requests) tuple
        """
        now = time.time()
        window_start = now - self._rate_limit_window

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
        window_start = now - self._rate_limit_window

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

    def extract_token_from_request(self, headers: Dict[str, str], query_params: Optional[Dict[str, list]] = None) -> Optional[str]:
        """Extract token from Authorization header only.

        Security: Query parameter tokens are not accepted because they:
        - Appear in server access logs
        - May be cached in browser history
        - Can leak through Referer headers

        Args:
            headers: HTTP headers dict
            query_params: Deprecated, ignored for security reasons

        Returns:
            Token string if found in Authorization header, None otherwise
        """
        # ONLY accept Authorization: Bearer header (not query params)
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Query params intentionally NOT supported for security
        # Tokens in URLs appear in logs and browser history
        return None


# Global auth config instance
auth_config = AuthConfig()
auth_config.configure_from_env()


def check_auth(headers: Dict[str, Any], query_string: str = "", loop_id: str = "", ip_address: str = "") -> tuple:
    """
    Check authentication and rate limiting for a request.

    Args:
        headers: HTTP headers dict
        query_string: Deprecated, ignored (tokens only accepted via Authorization header)
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

    # Only extract token from Authorization header (not query params for security)
    token = auth_config.extract_token_from_request(headers)

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


def generate_shareable_link(base_url: str, loop_id: str, expires_in: int = SHAREABLE_LINK_TTL) -> str:
    """Generate a shareable link with session-based authentication.

    Uses server-side sessions instead of embedding tokens in URLs to prevent
    token exposure in browser history, server logs, and HTTP referrer headers.

    Args:
        base_url: The base URL to share
        loop_id: The loop ID to authorize access to
        expires_in: Session TTL in seconds (default: SHAREABLE_LINK_TTL)

    Returns:
        URL with session parameter, or base_url if auth is disabled
    """
    # Generate session (stores token server-side)
    session_id = auth_config.generate_session(loop_id, expires_in)
    if not session_id:
        return base_url

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}session={session_id}"


def resolve_shareable_session(session_id: str) -> tuple[bool, str, str]:
    """Resolve a shareable session from URL parameter.

    Args:
        session_id: The session ID from the URL's 'session' parameter

    Returns:
        Tuple of (is_valid, token, loop_id)
    """
    return auth_config.resolve_session(session_id)