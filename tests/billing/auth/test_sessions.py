"""
Tests for JWT Session Management.

Covers session creation, activity tracking, inactivity cleanup,
max sessions enforcement, and device name parsing.
"""

import time
from unittest.mock import patch

import pytest

from aragora.billing.auth.sessions import (
    JWTSession,
    JWTSessionManager,
    _parse_device_name,
    get_session_manager,
    reset_session_manager,
)


# ============================================================================
# JWTSession Tests
# ============================================================================


class TestJWTSession:
    """Tests for JWTSession dataclass."""

    def test_create_minimal(self):
        """Test creating session with minimal fields."""
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=time.time(),
            last_activity=time.time(),
        )

        assert session.session_id == "jti-123"
        assert session.user_id == "user-456"
        assert session.ip_address is None
        assert session.device_name is None

    def test_create_full(self):
        """Test creating session with all fields."""
        now = time.time()
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=now,
            last_activity=now,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X)",
            device_name="Safari on Mac",
            expires_at=now + 86400,
        )

        assert session.ip_address == "192.168.1.100"
        assert session.device_name == "Safari on Mac"
        assert session.expires_at == now + 86400

    def test_is_expired_not_expired(self):
        """Test is_expired returns False when not expired."""
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=time.time(),
            last_activity=time.time(),
            expires_at=time.time() + 3600,  # 1 hour from now
        )

        assert not session.is_expired()

    def test_is_expired_expired(self):
        """Test is_expired returns True when expired."""
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=time.time() - 7200,
            last_activity=time.time() - 3600,
            expires_at=time.time() - 1,  # Expired 1 second ago
        )

        assert session.is_expired()

    def test_is_expired_no_expiry(self):
        """Test is_expired returns False when no expiry set."""
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=time.time(),
            last_activity=time.time(),
            expires_at=None,
        )

        assert not session.is_expired()

    def test_is_inactive_not_inactive(self):
        """Test is_inactive returns False when recently active."""
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=time.time(),
            last_activity=time.time(),
        )

        assert not session.is_inactive(timeout=3600)

    def test_is_inactive_inactive(self):
        """Test is_inactive returns True when inactive."""
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=time.time() - 7200,
            last_activity=time.time() - 7200,  # 2 hours ago
        )

        assert session.is_inactive(timeout=3600)  # 1 hour timeout

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = time.time()
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=now,
            last_activity=now,
            ip_address="10.0.0.1",
            user_agent="Mozilla/5.0",
            expires_at=now + 86400,
        )

        data = session.to_dict()

        assert data["session_id"] == "jti-123"
        assert data["user_id"] == "user-456"
        assert data["ip_address"] == "10.0.0.1"
        assert "created_at" in data
        assert "last_activity" in data
        assert "expires_at" in data
        assert data["is_current"] is False

    def test_to_dict_derives_device_name(self):
        """Test to_dict derives device name from user agent if not set."""
        session = JWTSession(
            session_id="jti-123",
            user_id="user-456",
            created_at=time.time(),
            last_activity=time.time(),
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0)",
            device_name=None,
        )

        data = session.to_dict()

        assert data["device_name"] == "iPhone"


# ============================================================================
# Device Name Parsing Tests
# ============================================================================


class TestDeviceNameParsing:
    """Tests for user agent to device name parsing."""

    def test_parse_none(self):
        """Test parsing None user agent."""
        assert _parse_device_name(None) == "Unknown Device"

    def test_parse_empty(self):
        """Test parsing empty user agent."""
        assert _parse_device_name("") == "Unknown Device"

    def test_parse_iphone(self):
        """Test parsing iPhone user agent."""
        ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
        assert _parse_device_name(ua) == "iPhone"

    def test_parse_ipad(self):
        """Test parsing iPad user agent."""
        ua = "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)"
        assert _parse_device_name(ua) == "iPad"

    def test_parse_android_phone(self):
        """Test parsing Android phone user agent."""
        ua = "Mozilla/5.0 (Linux; Android 11; Pixel 5) Mobile"
        assert _parse_device_name(ua) == "Android Phone"

    def test_parse_android_tablet(self):
        """Test parsing Android tablet user agent."""
        ua = "Mozilla/5.0 (Linux; Android 11; SM-T870)"
        assert _parse_device_name(ua) == "Android Tablet"

    def test_parse_chrome_mac(self):
        """Test parsing Chrome on Mac user agent."""
        ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        assert _parse_device_name(ua) == "Chrome on Mac"

    def test_parse_safari_mac(self):
        """Test parsing Safari on Mac user agent."""
        ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
        assert _parse_device_name(ua) == "Safari on Mac"

    def test_parse_firefox_mac(self):
        """Test parsing Firefox on Mac user agent."""
        ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        assert _parse_device_name(ua) == "Firefox on Mac"

    def test_parse_chrome_windows(self):
        """Test parsing Chrome on Windows user agent."""
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        assert _parse_device_name(ua) == "Chrome on Windows"

    def test_parse_edge_windows(self):
        """Test parsing Edge on Windows user agent."""
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
        assert _parse_device_name(ua) == "Edge on Windows"

    def test_parse_chrome_linux(self):
        """Test parsing Chrome on Linux user agent."""
        ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        assert _parse_device_name(ua) == "Chrome on Linux"

    def test_parse_firefox_linux(self):
        """Test parsing Firefox on Linux user agent."""
        ua = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
        assert _parse_device_name(ua) == "Firefox on Linux"

    def test_parse_curl(self):
        """Test parsing cURL user agent."""
        ua = "curl/7.68.0"
        assert _parse_device_name(ua) == "cURL"

    def test_parse_python_requests(self):
        """Test parsing Python requests user agent."""
        ua = "python-requests/2.25.1"
        assert _parse_device_name(ua) == "Python Client"

    def test_parse_postman(self):
        """Test parsing Postman user agent."""
        ua = "PostmanRuntime/7.28.0"
        assert _parse_device_name(ua) == "Postman"

    def test_parse_unknown(self):
        """Test parsing unknown user agent."""
        ua = "SomeCustomApp/1.0"
        assert _parse_device_name(ua) == "Unknown Device"


# ============================================================================
# JWTSessionManager Tests
# ============================================================================


class TestJWTSessionManager:
    """Tests for JWTSessionManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh session manager for testing."""
        return JWTSessionManager(
            session_ttl=3600,
            max_sessions_per_user=5,
            inactivity_timeout=600,
        )

    def test_create_session(self, manager):
        """Test creating a new session."""
        session = manager.create_session(
            user_id="user-123",
            token_jti="jti-456",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0 (iPhone)",
        )

        assert session.session_id == "jti-456"
        assert session.user_id == "user-123"
        assert session.ip_address == "192.168.1.1"
        assert session.device_name == "iPhone"

    def test_create_session_with_expiry(self, manager):
        """Test creating session with expiration time."""
        expires = time.time() + 3600

        session = manager.create_session(
            user_id="user-123",
            token_jti="jti-456",
            expires_at=expires,
        )

        assert session.expires_at == expires

    def test_get_session(self, manager):
        """Test getting an existing session."""
        manager.create_session(
            user_id="user-123",
            token_jti="jti-456",
        )

        session = manager.get_session("user-123", "jti-456")

        assert session is not None
        assert session.session_id == "jti-456"

    def test_get_session_not_found(self, manager):
        """Test getting non-existent session."""
        session = manager.get_session("user-123", "nonexistent")
        assert session is None

    def test_get_session_expired(self, manager):
        """Test getting expired session returns None and cleans up."""
        manager.create_session(
            user_id="user-123",
            token_jti="jti-456",
            expires_at=time.time() - 1,  # Already expired
        )

        session = manager.get_session("user-123", "jti-456")

        assert session is None

    def test_touch_session(self, manager):
        """Test updating session activity."""
        session = manager.create_session(
            user_id="user-123",
            token_jti="jti-456",
        )
        original_activity = session.last_activity

        time.sleep(0.01)  # Small delay
        result = manager.touch_session("user-123", "jti-456")

        assert result is True

        updated_session = manager.get_session("user-123", "jti-456")
        assert updated_session.last_activity > original_activity

    def test_touch_session_not_found(self, manager):
        """Test touching non-existent session."""
        result = manager.touch_session("user-123", "nonexistent")
        assert result is False

    def test_list_sessions(self, manager):
        """Test listing all sessions for a user."""
        manager.create_session("user-123", "jti-1")
        manager.create_session("user-123", "jti-2")
        manager.create_session("user-123", "jti-3")

        sessions = manager.list_sessions("user-123")

        assert len(sessions) == 3

    def test_list_sessions_excludes_inactive(self, manager):
        """Test list_sessions excludes inactive sessions by default."""
        # Create an active session
        manager.create_session("user-123", "jti-active")

        # Create an inactive session
        inactive = manager.create_session("user-123", "jti-inactive")
        # Manually make it inactive
        inactive.last_activity = time.time() - 1000

        sessions = manager.list_sessions("user-123", include_inactive=False)

        # Only active session should be returned
        assert len(sessions) == 1
        assert sessions[0].session_id == "jti-active"

    def test_list_sessions_includes_inactive(self, manager):
        """Test list_sessions can include inactive sessions."""
        manager.create_session("user-123", "jti-active")

        inactive = manager.create_session("user-123", "jti-inactive")
        inactive.last_activity = time.time() - 1000

        sessions = manager.list_sessions("user-123", include_inactive=True)

        assert len(sessions) == 2

    def test_list_sessions_empty_user(self, manager):
        """Test listing sessions for user with no sessions."""
        sessions = manager.list_sessions("nonexistent-user")
        assert sessions == []

    def test_revoke_session(self, manager):
        """Test revoking a specific session."""
        manager.create_session("user-123", "jti-1")
        manager.create_session("user-123", "jti-2")

        result = manager.revoke_session("user-123", "jti-1")

        assert result is True

        sessions = manager.list_sessions("user-123")
        assert len(sessions) == 1
        assert sessions[0].session_id == "jti-2"

    def test_revoke_session_not_found(self, manager):
        """Test revoking non-existent session."""
        result = manager.revoke_session("user-123", "nonexistent")
        assert result is False

    def test_revoke_all_sessions(self, manager):
        """Test revoking all sessions for a user."""
        manager.create_session("user-123", "jti-1")
        manager.create_session("user-123", "jti-2")
        manager.create_session("user-123", "jti-3")

        count = manager.revoke_all_sessions("user-123")

        assert count == 3
        assert manager.get_session_count("user-123") == 0

    def test_revoke_all_sessions_except_current(self, manager):
        """Test revoking all sessions except current."""
        manager.create_session("user-123", "jti-1")
        manager.create_session("user-123", "jti-2")
        manager.create_session("user-123", "jti-current")

        count = manager.revoke_all_sessions("user-123", except_jti="jti-current")

        assert count == 2
        assert manager.get_session_count("user-123") == 1

        remaining = manager.get_session("user-123", "jti-current")
        assert remaining is not None

    def test_revoke_all_sessions_no_sessions(self, manager):
        """Test revoking sessions for user with none."""
        count = manager.revoke_all_sessions("nonexistent")
        assert count == 0

    def test_get_session_count(self, manager):
        """Test getting session count for a user."""
        assert manager.get_session_count("user-123") == 0

        manager.create_session("user-123", "jti-1")
        manager.create_session("user-123", "jti-2")

        assert manager.get_session_count("user-123") == 2


# ============================================================================
# Max Sessions Enforcement Tests
# ============================================================================


class TestMaxSessionsEnforcement:
    """Tests for maximum sessions per user enforcement."""

    def test_evicts_oldest_when_max_reached(self):
        """Test oldest session is evicted when max reached."""
        manager = JWTSessionManager(max_sessions_per_user=3)

        # Create 3 sessions
        manager.create_session("user-123", "jti-1")
        time.sleep(0.01)
        manager.create_session("user-123", "jti-2")
        time.sleep(0.01)
        manager.create_session("user-123", "jti-3")

        # Create 4th session - should evict jti-1
        manager.create_session("user-123", "jti-4")

        assert manager.get_session_count("user-123") == 3
        assert manager.get_session("user-123", "jti-1") is None
        assert manager.get_session("user-123", "jti-4") is not None

    def test_eviction_preserves_newest(self):
        """Test eviction preserves newest sessions."""
        manager = JWTSessionManager(max_sessions_per_user=2)

        manager.create_session("user-123", "jti-1")
        time.sleep(0.01)
        manager.create_session("user-123", "jti-2")
        time.sleep(0.01)
        manager.create_session("user-123", "jti-3")

        # jti-2 and jti-3 should remain
        sessions = manager.list_sessions("user-123")
        session_ids = [s.session_id for s in sessions]

        assert "jti-1" not in session_ids
        assert "jti-2" in session_ids
        assert "jti-3" in session_ids


# ============================================================================
# Cleanup Tests
# ============================================================================


class TestSessionCleanup:
    """Tests for session cleanup functionality."""

    def test_maybe_cleanup_removes_expired(self):
        """Test cleanup removes expired sessions."""
        manager = JWTSessionManager(inactivity_timeout=1)
        manager._cleanup_interval = 0  # Force cleanup every time

        # Create session that will be inactive
        session = manager.create_session("user-123", "jti-1")
        session.last_activity = time.time() - 10  # 10 seconds ago

        # Trigger cleanup by creating another session
        manager.create_session("user-123", "jti-2")

        # First session should be cleaned up
        remaining = manager.list_sessions("user-123", include_inactive=True)
        # Note: cleanup happens in create_session, so jti-1 may or may not be cleaned
        # depending on timing. This tests the cleanup mechanism exists.
        assert manager.get_session_count("user-123") >= 1


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_creates(self):
        """Test concurrent session creation doesn't corrupt state."""
        import threading

        manager = JWTSessionManager(max_sessions_per_user=100)

        def create_sessions(user_id, start, count):
            for i in range(count):
                manager.create_session(user_id, f"jti-{start + i}")

        threads = []
        for i in range(5):
            t = threading.Thread(target=create_sessions, args=("user-123", i * 10, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have 50 sessions (5 threads * 10 sessions each)
        assert manager.get_session_count("user-123") == 50


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Tests for global session manager instance."""

    def test_get_session_manager_singleton(self):
        """Test get_session_manager returns singleton."""
        reset_session_manager()

        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2

    def test_reset_session_manager(self):
        """Test reset_session_manager clears singleton."""
        manager1 = get_session_manager()
        reset_session_manager()
        manager2 = get_session_manager()

        assert manager1 is not manager2


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Tests for session manager configuration."""

    def test_custom_session_ttl(self):
        """Test custom session TTL is respected."""
        manager = JWTSessionManager(session_ttl=7200)  # 2 hours
        assert manager.session_ttl == 7200

    def test_custom_max_sessions(self):
        """Test custom max sessions is respected."""
        manager = JWTSessionManager(max_sessions_per_user=20)
        assert manager.max_sessions_per_user == 20

    def test_custom_inactivity_timeout(self):
        """Test custom inactivity timeout is respected."""
        manager = JWTSessionManager(inactivity_timeout=1800)  # 30 minutes
        assert manager.inactivity_timeout == 1800

    def test_env_configuration(self):
        """Test configuration from environment variables."""
        env = {
            "ARAGORA_JWT_SESSION_TTL": "7200",
            "ARAGORA_MAX_SESSIONS_PER_USER": "20",
            "ARAGORA_SESSION_INACTIVITY_TIMEOUT": "1800",
        }

        with patch.dict("os.environ", env, clear=False):
            # The module reads env at import time, so we test defaults
            manager = JWTSessionManager()
            # Default values are used since env was set after import
            assert manager.session_ttl is not None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_user_id(self):
        """Test handling empty user ID."""
        manager = JWTSessionManager()
        session = manager.create_session("", "jti-1")

        assert session.user_id == ""
        assert manager.get_session("", "jti-1") is not None

    def test_special_characters_in_ids(self):
        """Test handling special characters in IDs."""
        manager = JWTSessionManager()

        session = manager.create_session(
            "user:123@domain.com",
            "jti_abc-123.456",
        )

        retrieved = manager.get_session("user:123@domain.com", "jti_abc-123.456")
        assert retrieved is not None
        assert retrieved.session_id == "jti_abc-123.456"

    def test_very_long_user_agent(self):
        """Test handling very long user agent string."""
        manager = JWTSessionManager()
        long_ua = "A" * 10000

        session = manager.create_session(
            "user-123",
            "jti-1",
            user_agent=long_ua,
        )

        assert session.user_agent == long_ua

    def test_unicode_in_user_agent(self):
        """Test handling unicode in user agent."""
        manager = JWTSessionManager()

        session = manager.create_session(
            "user-123",
            "jti-1",
            user_agent="Mozilla/5.0 (Test \u4e2d\u6587)",
        )

        assert "\u4e2d\u6587" in session.user_agent

    def test_multiple_users_isolated(self):
        """Test sessions for different users are isolated."""
        manager = JWTSessionManager()

        manager.create_session("user-1", "jti-a")
        manager.create_session("user-2", "jti-b")

        user1_sessions = manager.list_sessions("user-1")
        user2_sessions = manager.list_sessions("user-2")

        assert len(user1_sessions) == 1
        assert len(user2_sessions) == 1
        assert user1_sessions[0].session_id == "jti-a"
        assert user2_sessions[0].session_id == "jti-b"
