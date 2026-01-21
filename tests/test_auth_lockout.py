"""
Tests for the account lockout system.

Tests cover:
- LockoutTracker class functionality
- Lockout triggers after threshold
- Exponential backoff works correctly
- Reset clears lockout
- Lockout by email and IP separately
- Admin unlock endpoint
"""

import os
import time
import pytest
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def lockout_tracker():
    """Create a fresh LockoutTracker instance for testing."""
    from aragora.auth.lockout import LockoutTracker, reset_lockout_tracker

    # Reset global tracker to ensure clean state
    reset_lockout_tracker()

    # Create tracker without Redis for testing
    tracker = LockoutTracker(use_redis=False)
    yield tracker

    # Cleanup
    reset_lockout_tracker()


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = Mock()
    user.id = "user-123"
    user.email = "test@example.com"
    user.name = "Test User"
    user.org_id = "org-456"
    user.role = "member"
    user.is_active = True
    user.verify_password = Mock(return_value=False)  # Default to failed login
    user.to_dict = Mock(
        return_value={
            "id": "user-123",
            "email": "test@example.com",
            "name": "Test User",
        }
    )
    return user


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user with MFA enabled."""
    user = Mock()
    user.id = "admin-001"
    user.email = "admin@example.com"
    user.name = "Admin User"
    user.org_id = "org-456"
    user.role = "admin"
    user.is_active = True
    user.mfa_enabled = True  # Admin requires MFA
    return user


@pytest.fixture
def mock_user_store(mock_user):
    """Create a mock user store."""
    store = Mock()
    store.get_user_by_email = Mock(return_value=mock_user)
    store.get_user_by_id = Mock(return_value=mock_user)
    store.record_failed_login = Mock(return_value=(1, None))
    store.reset_failed_login_attempts = Mock(return_value=True)
    store.is_account_locked = Mock(return_value=(False, None, 0))
    return store


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = Mock()
    handler.command = "POST"
    handler.headers = {"Content-Type": "application/json"}
    handler.client_address = ("192.168.1.100", 12345)
    handler.rfile = Mock()
    return handler


# ============================================================================
# LockoutTracker Unit Tests
# ============================================================================


class TestLockoutTracker:
    """Tests for LockoutTracker class."""

    def test_initial_state_not_locked(self, lockout_tracker):
        """Test that accounts are not locked initially."""
        assert not lockout_tracker.is_locked(email="new@example.com")
        assert not lockout_tracker.is_locked(ip="192.168.1.1")
        assert lockout_tracker.get_remaining_time(email="new@example.com") == 0

    def test_record_failure_increments_count(self, lockout_tracker):
        """Test that record_failure increments the attempt count."""
        email = "test@example.com"

        attempts, lockout = lockout_tracker.record_failure(email=email)
        assert attempts == 1
        assert lockout is None

        attempts, lockout = lockout_tracker.record_failure(email=email)
        assert attempts == 2
        assert lockout is None

    def test_lockout_after_threshold_1(self, lockout_tracker):
        """Test lockout triggers after 5 failed attempts (1 minute lockout)."""
        email = "test@example.com"

        # First 4 attempts should not trigger lockout
        for i in range(4):
            attempts, lockout = lockout_tracker.record_failure(email=email)
            assert lockout is None, f"Should not lock on attempt {i+1}"

        # 5th attempt should trigger 1 minute lockout
        attempts, lockout = lockout_tracker.record_failure(email=email)
        assert attempts == 5
        assert lockout == 60  # 1 minute in seconds
        assert lockout_tracker.is_locked(email=email)

    def test_lockout_after_threshold_2(self, lockout_tracker):
        """Test lockout escalates after 10 failed attempts (15 minute lockout)."""
        email = "test@example.com"

        # Make 10 failed attempts
        for i in range(10):
            attempts, lockout = lockout_tracker.record_failure(email=email)

        assert attempts == 10
        assert lockout == 15 * 60  # 15 minutes in seconds
        assert lockout_tracker.is_locked(email=email)

    def test_lockout_after_threshold_3(self, lockout_tracker):
        """Test lockout escalates after 15 failed attempts (1 hour lockout)."""
        email = "test@example.com"

        # Make 15 failed attempts
        for i in range(15):
            attempts, lockout = lockout_tracker.record_failure(email=email)

        assert attempts == 15
        assert lockout == 60 * 60  # 1 hour in seconds
        assert lockout_tracker.is_locked(email=email)

    def test_reset_clears_lockout(self, lockout_tracker):
        """Test that reset clears lockout state."""
        email = "test@example.com"

        # Trigger lockout
        for _ in range(5):
            lockout_tracker.record_failure(email=email)

        assert lockout_tracker.is_locked(email=email)

        # Reset should clear lockout
        lockout_tracker.reset(email=email)

        assert not lockout_tracker.is_locked(email=email)
        assert lockout_tracker.get_remaining_time(email=email) == 0

    def test_lockout_by_email_and_ip_separately(self, lockout_tracker):
        """Test that email and IP are tracked independently."""
        email = "test@example.com"
        ip = "192.168.1.100"

        # Lock out email only
        for _ in range(5):
            lockout_tracker.record_failure(email=email)

        assert lockout_tracker.is_locked(email=email)
        assert not lockout_tracker.is_locked(ip=ip)

        # Reset email
        lockout_tracker.reset(email=email)

        # Lock out IP only
        for _ in range(5):
            lockout_tracker.record_failure(ip=ip)

        assert not lockout_tracker.is_locked(email=email)
        assert lockout_tracker.is_locked(ip=ip)

    def test_lockout_blocks_if_either_locked(self, lockout_tracker):
        """Test that is_locked returns True if either email or IP is locked."""
        email = "test@example.com"
        ip = "192.168.1.100"

        # Lock the IP
        for _ in range(5):
            lockout_tracker.record_failure(ip=ip)

        # is_locked with email+ip should return True because IP is locked
        assert lockout_tracker.is_locked(email=email, ip=ip)

    def test_get_remaining_time_returns_max(self, lockout_tracker):
        """Test that get_remaining_time returns the maximum across email/IP."""
        email = "test@example.com"
        ip = "192.168.1.100"

        # Lock email with threshold 1 (1 minute)
        for _ in range(5):
            lockout_tracker.record_failure(email=email)

        # Lock IP with threshold 2 (15 minutes)
        for _ in range(10):
            lockout_tracker.record_failure(ip=ip)

        # Combined should return the longer duration
        remaining = lockout_tracker.get_remaining_time(email=email, ip=ip)
        assert remaining > 60  # Should be closer to 15 minutes

    def test_get_info_returns_details(self, lockout_tracker):
        """Test that get_info returns detailed lockout information."""
        email = "test@example.com"
        ip = "192.168.1.100"

        # Record some failures
        for _ in range(3):
            lockout_tracker.record_failure(email=email, ip=ip)

        info = lockout_tracker.get_info(email=email, ip=ip)

        assert "is_locked" in info
        assert "remaining_seconds" in info
        assert "email" in info
        assert "ip" in info
        assert info["email"]["failed_attempts"] == 3
        assert info["ip"]["failed_attempts"] == 3

    def test_admin_unlock_clears_lockout(self, lockout_tracker):
        """Test that admin_unlock clears lockout state."""
        email = "test@example.com"

        # Trigger lockout
        for _ in range(5):
            lockout_tracker.record_failure(email=email)

        assert lockout_tracker.is_locked(email=email)

        # Admin unlock
        result = lockout_tracker.admin_unlock(email=email, user_id="user-123")

        assert result is True
        assert not lockout_tracker.is_locked(email=email)

    def test_backend_type_defaults_to_memory(self, lockout_tracker):
        """Test that backend_type is 'memory' when Redis is not used."""
        assert lockout_tracker.backend_type == "memory"


class TestLockoutEntry:
    """Tests for LockoutEntry class."""

    def test_is_locked_when_lockout_active(self):
        """Test is_locked returns True when lockout is active."""
        from aragora.auth.lockout import LockoutEntry

        entry = LockoutEntry(
            failed_attempts=5,
            lockout_until=time.time() + 60,  # Locked for 60 more seconds
        )
        assert entry.is_locked()

    def test_is_locked_when_lockout_expired(self):
        """Test is_locked returns False when lockout has expired."""
        from aragora.auth.lockout import LockoutEntry

        entry = LockoutEntry(
            failed_attempts=5,
            lockout_until=time.time() - 1,  # Expired 1 second ago
        )
        assert not entry.is_locked()

    def test_is_locked_when_no_lockout(self):
        """Test is_locked returns False when no lockout set."""
        from aragora.auth.lockout import LockoutEntry

        entry = LockoutEntry(failed_attempts=3)
        assert not entry.is_locked()

    def test_get_remaining_seconds(self):
        """Test get_remaining_seconds returns correct value."""
        from aragora.auth.lockout import LockoutEntry

        entry = LockoutEntry(
            failed_attempts=5,
            lockout_until=time.time() + 30,
        )
        remaining = entry.get_remaining_seconds()
        assert 28 <= remaining <= 30


class TestInMemoryBackend:
    """Tests for InMemoryLockoutBackend."""

    def test_entry_expires_after_ttl(self):
        """Test that entries expire after TTL."""
        from aragora.auth.lockout import InMemoryLockoutBackend, LockoutEntry

        backend = InMemoryLockoutBackend()
        entry = LockoutEntry(failed_attempts=1)

        # Set with very short TTL
        backend.set_entry("test", entry, ttl_seconds=1)

        # Should exist immediately
        assert backend.get_entry("test") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be gone
        assert backend.get_entry("test") is None

    def test_delete_entry(self):
        """Test delete_entry removes entry."""
        from aragora.auth.lockout import InMemoryLockoutBackend, LockoutEntry

        backend = InMemoryLockoutBackend()
        entry = LockoutEntry(failed_attempts=1)

        backend.set_entry("test", entry, ttl_seconds=60)
        assert backend.get_entry("test") is not None

        backend.delete_entry("test")
        assert backend.get_entry("test") is None

    def test_cleanup_expired(self):
        """Test cleanup_expired removes old entries."""
        from aragora.auth.lockout import InMemoryLockoutBackend, LockoutEntry

        backend = InMemoryLockoutBackend()

        # Add entries with short TTL
        for i in range(5):
            backend.set_entry(f"test{i}", LockoutEntry(failed_attempts=1), ttl_seconds=1)

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup
        removed = backend.cleanup_expired()
        assert removed == 5


# ============================================================================
# Auth Handler Integration Tests
# ============================================================================


class TestAuthHandlerLockout:
    """Tests for lockout integration in AuthHandler."""

    @pytest.fixture(autouse=True)
    def setup_lockout_tracker(self):
        """Reset lockout tracker before each test."""
        from aragora.auth.lockout import reset_lockout_tracker

        reset_lockout_tracker()
        yield
        reset_lockout_tracker()

    def test_login_blocked_when_locked(self, mock_user_store, mock_user, mock_handler):
        """Test that login is blocked when account is locked."""
        from aragora.server.handlers.auth import AuthHandler
        from aragora.auth.lockout import get_lockout_tracker

        # Set up lockout
        tracker = get_lockout_tracker()
        for _ in range(5):
            tracker.record_failure(email=mock_user.email)

        # Set up request body
        body = b'{"email": "test@example.com", "password": "wrongpassword"}'
        mock_handler.rfile.read = Mock(return_value=body)
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }

        ctx = {"user_store": mock_user_store}
        handler = AuthHandler(ctx)

        # Call login handler (skip rate limit decorator)
        result = handler._handle_login.__wrapped__(handler, mock_handler)

        assert result.status_code == 429
        assert b"Too many failed attempts" in result.body

    def test_failed_login_records_failure(self, mock_user_store, mock_user, mock_handler):
        """Test that failed login records failure in tracker."""
        from aragora.server.handlers.auth import AuthHandler
        from aragora.auth.lockout import get_lockout_tracker

        tracker = get_lockout_tracker()

        # Ensure user exists but password is wrong
        mock_user.verify_password.return_value = False

        # Set up request body
        body = b'{"email": "test@example.com", "password": "wrongpassword"}'
        mock_handler.rfile.read = Mock(return_value=body)
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }

        ctx = {"user_store": mock_user_store}
        handler = AuthHandler(ctx)

        # Call login handler
        result = handler._handle_login.__wrapped__(handler, mock_handler)

        # Should record failure
        info = tracker.get_info(email=mock_user.email)
        assert info["email"]["failed_attempts"] == 1

    def test_successful_login_resets_lockout(self, mock_user_store, mock_user, mock_handler):
        """Test that successful login resets lockout state."""
        from aragora.server.handlers.auth import AuthHandler
        from aragora.auth.lockout import get_lockout_tracker

        tracker = get_lockout_tracker()

        # Record some failed attempts (but not enough to lock)
        for _ in range(3):
            tracker.record_failure(email=mock_user.email)

        # Set up successful login
        mock_user.verify_password.return_value = True
        mock_user.mfa_enabled = False

        body = b'{"email": "test@example.com", "password": "correctpassword"}'
        mock_handler.rfile.read = Mock(return_value=body)
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }

        ctx = {"user_store": mock_user_store}
        handler = AuthHandler(ctx)

        # Mock JWT token creation
        with patch("aragora.billing.jwt_auth.create_token_pair") as mock_create_tokens:
            mock_tokens = Mock()
            mock_tokens.to_dict.return_value = {
                "access_token": "test_access",
                "refresh_token": "test_refresh",
            }
            mock_create_tokens.return_value = mock_tokens

            result = handler._handle_login.__wrapped__(handler, mock_handler)

        # Should be successful
        assert result.status_code == 200

        # Lockout should be reset
        info = tracker.get_info(email=mock_user.email)
        assert info["email"]["failed_attempts"] == 0


# ============================================================================
# Admin Unlock Endpoint Tests
# ============================================================================


class TestAdminUnlockEndpoint:
    """Tests for admin unlock endpoint."""

    @pytest.fixture(autouse=True)
    def setup_lockout_tracker(self):
        """Reset lockout tracker before each test."""
        from aragora.auth.lockout import reset_lockout_tracker

        reset_lockout_tracker()
        yield
        reset_lockout_tracker()

    def test_admin_unlock_clears_lockout(
        self, mock_user_store, mock_user, mock_admin_user, mock_handler
    ):
        """Test that admin unlock endpoint clears lockout."""
        from aragora.server.handlers.admin import AdminHandler
        from aragora.auth.lockout import get_lockout_tracker

        # Set up lockout
        tracker = get_lockout_tracker()
        for _ in range(5):
            tracker.record_failure(email=mock_user.email)

        assert tracker.is_locked(email=mock_user.email)

        # Set up admin auth
        mock_user_store.get_user_by_id.side_effect = lambda uid: (
            mock_admin_user if uid == "admin-001" else mock_user
        )

        ctx = {"user_store": mock_user_store}
        handler = AdminHandler(ctx)

        # Mock RBAC permission check to allow
        mock_decision = Mock()
        mock_decision.allowed = True
        mock_decision.reason = "Test mock: allowed"

        # Mock authentication, MFA policy, and RBAC permission
        with (
            patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract,
            patch(
                "aragora.server.handlers.admin.admin.enforce_admin_mfa_policy", return_value=None
            ),
            patch(
                "aragora.server.handlers.admin.admin.check_permission", return_value=mock_decision
            ),
        ):
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-001"
            mock_auth_ctx.mfa_verified = True
            mock_extract.return_value = mock_auth_ctx

            # Call unlock endpoint
            result = handler._unlock_user(mock_handler, mock_user.id)

        assert result.status_code == 200

        # Lockout should be cleared
        assert not tracker.is_locked(email=mock_user.email)

    def test_admin_unlock_requires_admin_role(self, mock_user_store, mock_user, mock_handler):
        """Test that non-admin users cannot unlock accounts."""
        from aragora.server.handlers.admin import AdminHandler

        # Make user a regular member
        mock_user.role = "member"
        mock_user_store.get_user_by_id.return_value = mock_user

        ctx = {"user_store": mock_user_store}
        handler = AdminHandler(ctx)

        # Patch at the module where it's imported
        with patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_auth_ctx

            result = handler._unlock_user(mock_handler, "some-user-id")

        assert result.status_code == 403

    def test_admin_unlock_user_not_found(self, mock_user_store, mock_admin_user, mock_handler):
        """Test that unlock returns 404 for non-existent user."""
        from aragora.server.handlers.admin import AdminHandler

        mock_user_store.get_user_by_id.side_effect = lambda uid: (
            mock_admin_user if uid == "admin-001" else None
        )

        ctx = {"user_store": mock_user_store}
        handler = AdminHandler(ctx)

        # Mock RBAC permission check to allow
        mock_decision = Mock()
        mock_decision.allowed = True
        mock_decision.reason = "Test mock: allowed"

        # Mock authentication, MFA policy, and RBAC permission
        with (
            patch("aragora.server.handlers.admin.admin.extract_user_from_request") as mock_extract,
            patch(
                "aragora.server.handlers.admin.admin.enforce_admin_mfa_policy", return_value=None
            ),
            patch(
                "aragora.server.handlers.admin.admin.check_permission", return_value=mock_decision
            ),
        ):
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_auth_ctx.user_id = "admin-001"
            mock_auth_ctx.mfa_verified = True
            mock_extract.return_value = mock_auth_ctx

            result = handler._unlock_user(mock_handler, "nonexistent-user")

        assert result.status_code == 404


# ============================================================================
# Global Lockout Tracker Tests
# ============================================================================


class TestGlobalLockoutTracker:
    """Tests for global lockout tracker management."""

    def test_get_lockout_tracker_returns_singleton(self):
        """Test that get_lockout_tracker returns the same instance."""
        from aragora.auth.lockout import get_lockout_tracker, reset_lockout_tracker

        reset_lockout_tracker()

        tracker1 = get_lockout_tracker()
        tracker2 = get_lockout_tracker()

        assert tracker1 is tracker2

    def test_reset_lockout_tracker_clears_singleton(self):
        """Test that reset_lockout_tracker clears the singleton."""
        from aragora.auth.lockout import get_lockout_tracker, reset_lockout_tracker

        tracker1 = get_lockout_tracker()
        reset_lockout_tracker()
        tracker2 = get_lockout_tracker()

        assert tracker1 is not tracker2
