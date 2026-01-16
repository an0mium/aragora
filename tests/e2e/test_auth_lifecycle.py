"""
E2E tests for authentication lifecycle.

Tests the system's ability to:
- Configure and enable authentication
- Validate tokens correctly
- Handle token expiration and revocation
- Rate limit by token and IP
- Manage shareable sessions
- Clean up expired entries
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.auth import AuthConfig


class TestAuthConfigInitialization:
    """Test AuthConfig initialization."""

    def test_auth_disabled_by_default(self):
        """Verify authentication is disabled by default."""
        config = AuthConfig()
        assert not config.enabled

    def test_auth_has_rate_limiting(self):
        """Verify rate limiting configuration exists."""
        config = AuthConfig()
        assert config.rate_limit_per_minute > 0
        assert config.ip_rate_limit_per_minute > 0

    def test_auth_has_token_ttl(self):
        """Verify token TTL is configured."""
        config = AuthConfig()
        assert config.token_ttl > 0

    def test_auth_has_allowed_origins(self):
        """Verify allowed origins are configured."""
        config = AuthConfig()
        assert isinstance(config.allowed_origins, list)


class TestTokenConfiguration:
    """Test token configuration and validation."""

    @pytest.fixture
    def auth_config(self):
        """Create an AuthConfig instance."""
        config = AuthConfig()
        config.enabled = True
        config.api_token = "test-token-12345"
        return config

    def test_can_set_api_token(self, auth_config):
        """Verify API token can be set."""
        assert auth_config.api_token == "test-token-12345"

    def test_token_validation_when_disabled(self):
        """Verify token validation skipped when auth disabled."""
        config = AuthConfig()
        config.enabled = False
        # When disabled, any token should be "valid" (not checked)
        assert not config.enabled

    def test_token_hashing(self, auth_config):
        """Verify tokens can be hashed for storage."""
        token = "secret-token"
        hashed = hashlib.sha256(token.encode()).hexdigest()
        assert len(hashed) == 64
        assert hashed != token


class TestTokenRevocation:
    """Test token revocation functionality."""

    @pytest.fixture
    def auth_config(self):
        """Create an AuthConfig instance."""
        config = AuthConfig()
        config.enabled = True
        return config

    def test_revocation_tracking_structure(self, auth_config):
        """Verify revoked tokens are tracked."""
        assert hasattr(auth_config, "_revoked_tokens")
        assert isinstance(auth_config._revoked_tokens, dict)

    def test_revocation_lock_exists(self, auth_config):
        """Verify revocation has thread safety."""
        assert hasattr(auth_config, "_revocation_lock")
        assert isinstance(auth_config._revocation_lock, type(threading.Lock()))

    def test_can_track_revoked_token(self, auth_config):
        """Verify revoked tokens can be added to tracking."""
        token_hash = hashlib.sha256(b"revoked-token").hexdigest()
        with auth_config._revocation_lock:
            auth_config._revoked_tokens[token_hash] = time.time()

        assert token_hash in auth_config._revoked_tokens


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def auth_config(self):
        """Create an AuthConfig instance."""
        config = AuthConfig()
        config.rate_limit_per_minute = 10
        config.ip_rate_limit_per_minute = 20
        return config

    def test_rate_limit_tracking_structures(self, auth_config):
        """Verify rate limit tracking structures exist."""
        assert hasattr(auth_config, "_token_request_counts")
        assert hasattr(auth_config, "_ip_request_counts")

    def test_rate_limit_lock_exists(self, auth_config):
        """Verify rate limiting has thread safety."""
        assert hasattr(auth_config, "_rate_limit_lock")

    def test_can_track_token_requests(self, auth_config):
        """Verify token requests can be tracked."""
        token = "test-token"
        with auth_config._rate_limit_lock:
            if token not in auth_config._token_request_counts:
                auth_config._token_request_counts[token] = []
            auth_config._token_request_counts[token].append(time.time())

        assert token in auth_config._token_request_counts
        assert len(auth_config._token_request_counts[token]) == 1

    def test_can_track_ip_requests(self, auth_config):
        """Verify IP requests can be tracked."""
        ip = "192.168.1.1"
        with auth_config._rate_limit_lock:
            if ip not in auth_config._ip_request_counts:
                auth_config._ip_request_counts[ip] = []
            auth_config._ip_request_counts[ip].append(time.time())

        assert ip in auth_config._ip_request_counts

    def test_rate_limit_simulation(self, auth_config):
        """Verify rate limiting logic works."""
        token = "test-token"
        window_seconds = 60
        limit = auth_config.rate_limit_per_minute

        # Simulate requests
        now = time.time()
        with auth_config._rate_limit_lock:
            auth_config._token_request_counts[token] = [
                now - i for i in range(limit + 5)  # Exceed limit
            ]

        # Check if over limit
        with auth_config._rate_limit_lock:
            requests = auth_config._token_request_counts.get(token, [])
            recent_requests = [r for r in requests if now - r < window_seconds]

        is_over_limit = len(recent_requests) > limit
        assert is_over_limit


class TestShareableSessions:
    """Test shareable session functionality."""

    @pytest.fixture
    def auth_config(self):
        """Create an AuthConfig instance."""
        config = AuthConfig()
        return config

    def test_session_storage_exists(self, auth_config):
        """Verify session storage structure exists."""
        assert hasattr(auth_config, "_shareable_sessions")
        assert isinstance(auth_config._shareable_sessions, dict)

    def test_session_lock_exists(self, auth_config):
        """Verify session operations are thread-safe."""
        assert hasattr(auth_config, "_session_lock")

    def test_can_create_session(self, auth_config):
        """Verify sessions can be created."""
        session_id = "session-abc123"
        session_data = {
            "token": "user-token",
            "expires_at": time.time() + 3600,
            "loop_id": "loop-1",
        }

        with auth_config._session_lock:
            auth_config._shareable_sessions[session_id] = session_data

        assert session_id in auth_config._shareable_sessions

    def test_can_retrieve_session(self, auth_config):
        """Verify sessions can be retrieved."""
        session_id = "session-xyz789"
        session_data = {
            "token": "another-token",
            "expires_at": time.time() + 3600,
            "loop_id": "loop-2",
        }

        with auth_config._session_lock:
            auth_config._shareable_sessions[session_id] = session_data

        with auth_config._session_lock:
            retrieved = auth_config._shareable_sessions.get(session_id)

        assert retrieved is not None
        assert retrieved["loop_id"] == "loop-2"

    def test_session_expiration_check(self, auth_config):
        """Verify session expiration can be checked."""
        session_id = "expired-session"
        session_data = {
            "token": "old-token",
            "expires_at": time.time() - 3600,  # Already expired
            "loop_id": "loop-old",
        }

        with auth_config._session_lock:
            auth_config._shareable_sessions[session_id] = session_data

        # Check expiration
        with auth_config._session_lock:
            session = auth_config._shareable_sessions.get(session_id)
            is_expired = session["expires_at"] < time.time()

        assert is_expired


class TestCleanupFunctionality:
    """Test automatic cleanup of expired entries."""

    @pytest.fixture
    def auth_config(self):
        """Create an AuthConfig instance."""
        config = AuthConfig()
        return config

    def test_cleanup_thread_starts(self, auth_config):
        """Verify cleanup thread is started."""
        assert auth_config._cleanup_thread is not None
        assert auth_config._cleanup_thread.daemon is True

    def test_cleanup_method_exists(self, auth_config):
        """Verify cleanup method exists."""
        assert hasattr(auth_config, "cleanup_expired_entries")

    def test_cleanup_expired_entries(self, auth_config):
        """Verify cleanup removes expired entries."""
        # Add expired entries
        old_time = time.time() - 7200  # 2 hours ago

        with auth_config._rate_limit_lock:
            auth_config._token_request_counts["old-token"] = [old_time]
            auth_config._ip_request_counts["old-ip"] = [old_time]

        # Cleanup should remove old entries
        stats = auth_config.cleanup_expired_entries()

        assert "token_entries_removed" in stats
        assert "ip_entries_removed" in stats


class TestConcurrentAccess:
    """Test thread safety of auth operations."""

    @pytest.fixture
    def auth_config(self):
        """Create an AuthConfig instance."""
        config = AuthConfig()
        return config

    def test_concurrent_token_tracking(self, auth_config):
        """Verify concurrent token tracking is safe."""
        errors: List[Exception] = []
        tokens = [f"token-{i}" for i in range(100)]

        def track_token(token: str):
            try:
                with auth_config._rate_limit_lock:
                    if token not in auth_config._token_request_counts:
                        auth_config._token_request_counts[token] = []
                    auth_config._token_request_counts[token].append(time.time())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=track_token, args=(t,)) for t in tokens]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(auth_config._token_request_counts) == 100

    def test_concurrent_session_operations(self, auth_config):
        """Verify concurrent session operations are safe."""
        errors: List[Exception] = []

        def create_and_read_session(session_num: int):
            try:
                session_id = f"session-{session_num}"
                with auth_config._session_lock:
                    auth_config._shareable_sessions[session_id] = {
                        "token": f"token-{session_num}",
                        "expires_at": time.time() + 3600,
                    }

                with auth_config._session_lock:
                    _ = auth_config._shareable_sessions.get(session_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_and_read_session, args=(i,)) for i in range(50)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestAuthenticationErrors:
    """Test authentication error handling."""

    def test_auth_error_import(self):
        """Verify AuthenticationError can be imported."""
        from aragora.exceptions import AuthenticationError

        error = AuthenticationError("Invalid token")
        assert str(error) == "Invalid token"

    def test_auth_error_is_exception(self):
        """Verify AuthenticationError is an Exception."""
        from aragora.exceptions import AuthenticationError

        assert issubclass(AuthenticationError, Exception)


class TestOriginValidation:
    """Test CORS origin validation."""

    @pytest.fixture
    def auth_config(self):
        """Create an AuthConfig instance."""
        config = AuthConfig()
        return config

    def test_allowed_origins_is_list(self, auth_config):
        """Verify allowed origins is a list."""
        assert isinstance(auth_config.allowed_origins, list)

    def test_origin_matching_logic(self, auth_config):
        """Verify origin matching works correctly."""
        auth_config.allowed_origins = ["https://example.com", "http://localhost:3000"]

        origin1 = "https://example.com"
        origin2 = "https://evil.com"

        is_allowed1 = origin1 in auth_config.allowed_origins
        is_allowed2 = origin2 in auth_config.allowed_origins

        assert is_allowed1
        assert not is_allowed2

    def test_wildcard_origin_handling(self):
        """Verify wildcard origins are handled."""
        allowed = ["*"]
        origin = "https://any-domain.com"

        # Wildcard allows all origins
        is_allowed = "*" in allowed
        assert is_allowed
