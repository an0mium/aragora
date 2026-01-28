"""
Tests for MFA freshness enforcement.

Phase 6: Auth Handler Test Gaps - MFA freshness tests.

Tests:
- test_mfa_fresh_within_15_minutes - Fresh check passes
- test_mfa_stale_after_15_minutes - Re-auth required
- test_step_up_auth_on_sensitive_ops - Elevated auth
- test_mfa_freshness_reset_on_verification - Timer reset
- test_mfa_methods_tracking - Methods recorded
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.auth.sessions import (
    JWTSession,
    JWTSessionManager,
    reset_session_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def session_manager() -> JWTSessionManager:
    """Create a fresh session manager for testing."""
    return JWTSessionManager(
        session_ttl=3600,
        max_sessions_per_user=10,
        inactivity_timeout=86400,
    )


@pytest.fixture(autouse=True)
def reset_global_manager():
    """Reset global session manager before and after each test."""
    reset_session_manager()
    yield
    reset_session_manager()


@pytest.fixture
def session_with_mfa(session_manager: JWTSessionManager) -> JWTSession:
    """Create a session with recent MFA verification."""
    session = session_manager.create_session(
        user_id="user-mfa",
        token_jti="token-mfa-1",
        ip_address="127.0.0.1",
    )
    # Record MFA verification
    session_manager.update_mfa_verification("user-mfa", "token-mfa-1", ["totp"])
    return session_manager.get_session("user-mfa", "token-mfa-1")


# ============================================================================
# Test: MFA Fresh Within 15 Minutes
# ============================================================================


class TestMFAFreshWithin15Minutes:
    """Test MFA freshness check when within default window."""

    def test_mfa_fresh_within_15_minutes(self, session_manager: JWTSessionManager):
        """Test that MFA is considered fresh within 15 minutes."""
        user_id = "user-fresh"
        token_jti = "token-1"

        # Create session and verify MFA
        session_manager.create_session(user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1")
        session_manager.update_mfa_verification(user_id, token_jti, ["totp"])

        # Check freshness - should be fresh (default 15 minutes = 900 seconds)
        is_fresh = session_manager.is_session_mfa_fresh(user_id, token_jti, 900)
        assert is_fresh is True

    def test_mfa_fresh_custom_window(self, session_manager: JWTSessionManager):
        """Test MFA freshness with custom time window."""
        user_id = "user-custom"
        token_jti = "token-custom"

        session_manager.create_session(user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1")
        session_manager.update_mfa_verification(user_id, token_jti, ["totp"])

        # Should be fresh with large window
        assert session_manager.is_session_mfa_fresh(user_id, token_jti, 3600) is True

        # Should be fresh with small window (just verified)
        assert session_manager.is_session_mfa_fresh(user_id, token_jti, 10) is True

    def test_mfa_not_verified_is_not_fresh(self, session_manager: JWTSessionManager):
        """Test that session without MFA verification is not fresh."""
        user_id = "user-no-mfa"
        token_jti = "token-no-mfa"

        # Create session without MFA verification
        session_manager.create_session(user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1")

        # Should not be fresh
        is_fresh = session_manager.is_session_mfa_fresh(user_id, token_jti, 900)
        assert is_fresh is False


# ============================================================================
# Test: MFA Stale After 15 Minutes
# ============================================================================


class TestMFAStaleAfter15Minutes:
    """Test MFA freshness check when beyond default window."""

    def test_mfa_stale_after_15_minutes(self, session_manager: JWTSessionManager):
        """Test that MFA is stale after 15 minutes."""
        user_id = "user-stale"
        token_jti = "token-stale"

        # Create session
        session = session_manager.create_session(
            user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1"
        )

        # Manually set MFA verification time to 16 minutes ago
        sixteen_minutes_ago = time.time() - (16 * 60)
        session.mfa_verified_at = sixteen_minutes_ago
        session.mfa_methods_used = ["totp"]

        # Check freshness with 15 minute window - should be stale
        is_fresh = session_manager.is_session_mfa_fresh(user_id, token_jti, 900)
        assert is_fresh is False

    def test_mfa_age_tracking(self, session_manager: JWTSessionManager):
        """Test MFA age in seconds tracking."""
        user_id = "user-age"
        token_jti = "token-age"

        # Create session without MFA
        session_manager.create_session(user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1")

        # No MFA verification - should return None
        age = session_manager.get_mfa_freshness(user_id, token_jti)
        assert age is None

        # Verify MFA
        session_manager.update_mfa_verification(user_id, token_jti, ["totp"])

        # Should have small age (just verified)
        age = session_manager.get_mfa_freshness(user_id, token_jti)
        assert age is not None
        assert age < 5  # Should be very recent

    def test_nonexistent_session_not_fresh(self, session_manager: JWTSessionManager):
        """Test that nonexistent session is not fresh."""
        is_fresh = session_manager.is_session_mfa_fresh("nonexistent", "token", 900)
        assert is_fresh is False


# ============================================================================
# Test: Step-Up Auth on Sensitive Operations
# ============================================================================


class TestStepUpAuth:
    """Test step-up authentication for sensitive operations."""

    def test_step_up_auth_decorator_fresh_mfa(self):
        """Test require_mfa_fresh decorator with fresh MFA."""
        from aragora.server.middleware.mfa import require_mfa_fresh

        # Create a mock function decorated with require_mfa_fresh
        @require_mfa_fresh(max_age_minutes=15)
        def sensitive_operation(handler, **kwargs):
            return {"success": True, "status": 200}

        # Create mocks
        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer token"}

        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.metadata = {"jti": "token-123", "mfa_enabled": True}
        mock_user.token_jti = "token-123"

        # Mock session manager with fresh MFA
        mock_session_manager = MagicMock()
        mock_session_manager.is_session_mfa_fresh.return_value = True

        with patch(
            "aragora.server.middleware.mfa.get_current_user",
            return_value=mock_user,
        ):
            with patch(
                "aragora.server.middleware.mfa._get_user_store_from_handler",
                return_value=MagicMock(),
            ):
                with patch(
                    "aragora.server.middleware.mfa._get_session_manager_from_handler",
                    return_value=mock_session_manager,
                ):
                    # Mock full_user with mfa_enabled
                    mock_user_store = MagicMock()
                    full_user = MagicMock()
                    full_user.mfa_enabled = True
                    full_user.is_service_account = False
                    mock_user_store.get_user_by_id.return_value = full_user

                    with patch(
                        "aragora.server.middleware.mfa._get_user_store_from_handler",
                        return_value=mock_user_store,
                    ):
                        result = sensitive_operation(mock_handler)

        # Should succeed
        assert result.get("success") is True or result.get("status") == 200

    def test_step_up_auth_decorator_stale_mfa(self):
        """Test require_mfa_fresh decorator with stale MFA."""
        from aragora.server.middleware.mfa import require_mfa_fresh
        from aragora.server.handlers.utils.responses import HandlerResult

        @require_mfa_fresh(max_age_minutes=15)
        def sensitive_operation(handler, **kwargs):
            return {"success": True, "status": 200}

        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer token"}

        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.metadata = {"jti": "token-123", "mfa_enabled": True}
        mock_user.token_jti = "token-123"

        # Mock session manager with stale MFA
        mock_session_manager = MagicMock()
        mock_session_manager.is_session_mfa_fresh.return_value = False

        mock_user_store = MagicMock()
        full_user = MagicMock()
        full_user.mfa_enabled = True
        full_user.is_service_account = False
        mock_user_store.get_user_by_id.return_value = full_user

        with patch(
            "aragora.server.middleware.mfa.get_current_user",
            return_value=mock_user,
        ):
            with patch(
                "aragora.server.middleware.mfa._get_user_store_from_handler",
                return_value=mock_user_store,
            ):
                with patch(
                    "aragora.server.middleware.mfa._get_session_manager_from_handler",
                    return_value=mock_session_manager,
                ):
                    result = sensitive_operation(mock_handler)

        # Should return 403 MFA_STEP_UP_REQUIRED
        # Result could be HandlerResult dataclass or dict
        if isinstance(result, HandlerResult):
            assert result.status_code == 403
        elif isinstance(result, dict):
            assert result.get("status") == 403 or result.get("status_code") == 403
        else:
            # The decorator returns HandlerResult on error
            assert hasattr(result, "status_code")
            assert result.status_code == 403


# ============================================================================
# Test: MFA Freshness Reset on Verification
# ============================================================================


class TestMFAFreshnessReset:
    """Test MFA freshness timer reset on verification."""

    def test_mfa_freshness_reset_on_verification(self, session_manager: JWTSessionManager):
        """Test that MFA verification resets the freshness timer."""
        user_id = "user-reset"
        token_jti = "token-reset"

        # Create session
        session = session_manager.create_session(
            user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1"
        )

        # Set old MFA verification (20 minutes ago)
        session.mfa_verified_at = time.time() - (20 * 60)
        session.mfa_methods_used = ["totp"]

        # Should be stale
        assert session_manager.is_session_mfa_fresh(user_id, token_jti, 900) is False

        # Re-verify MFA
        session_manager.update_mfa_verification(user_id, token_jti, ["totp"])

        # Should now be fresh
        assert session_manager.is_session_mfa_fresh(user_id, token_jti, 900) is True

    def test_verification_updates_timestamp(self, session_manager: JWTSessionManager):
        """Test that verification updates the timestamp."""
        user_id = "user-timestamp"
        token_jti = "token-timestamp"

        session_manager.create_session(user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1")

        # First verification
        session_manager.update_mfa_verification(user_id, token_jti, ["totp"])
        first_age = session_manager.get_mfa_freshness(user_id, token_jti)

        # Wait a bit
        time.sleep(0.1)

        # Second verification
        session_manager.update_mfa_verification(user_id, token_jti, ["totp"])
        second_age = session_manager.get_mfa_freshness(user_id, token_jti)

        # Second age should be less (more recent)
        assert second_age <= first_age


# ============================================================================
# Test: MFA Methods Tracking
# ============================================================================


class TestMFAMethodsTracking:
    """Test MFA verification method tracking."""

    def test_mfa_methods_tracking(self, session_manager: JWTSessionManager):
        """Test that MFA methods are tracked correctly."""
        user_id = "user-methods"
        token_jti = "token-methods"

        session = session_manager.create_session(
            user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1"
        )

        # Verify with TOTP
        session_manager.update_mfa_verification(user_id, token_jti, ["totp"])
        session = session_manager.get_session(user_id, token_jti)
        assert session.mfa_methods_used == ["totp"]

        # Re-verify with backup code
        session_manager.update_mfa_verification(user_id, token_jti, ["backup_code"])
        session = session_manager.get_session(user_id, token_jti)
        assert session.mfa_methods_used == ["backup_code"]

    def test_mfa_default_method_is_totp(self, session_manager: JWTSessionManager):
        """Test that default MFA method is TOTP."""
        user_id = "user-default"
        token_jti = "token-default"

        session_manager.create_session(user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1")

        # Verify without specifying method
        session_manager.update_mfa_verification(user_id, token_jti)
        session = session_manager.get_session(user_id, token_jti)
        assert session.mfa_methods_used == ["totp"]

    def test_mfa_multiple_methods(self, session_manager: JWTSessionManager):
        """Test tracking multiple MFA methods."""
        user_id = "user-multi"
        token_jti = "token-multi"

        session_manager.create_session(user_id=user_id, token_jti=token_jti, ip_address="127.0.0.1")

        # Verify with multiple methods
        session_manager.update_mfa_verification(user_id, token_jti, ["totp", "sms"])
        session = session_manager.get_session(user_id, token_jti)
        assert "totp" in session.mfa_methods_used
        assert "sms" in session.mfa_methods_used

    def test_update_mfa_nonexistent_session_returns_false(self, session_manager: JWTSessionManager):
        """Test updating MFA for nonexistent session returns False."""
        result = session_manager.update_mfa_verification("nonexistent", "token", ["totp"])
        assert result is False


# ============================================================================
# Test: JWTSession MFA Methods
# ============================================================================


class TestJWTSessionMFA:
    """Test JWTSession MFA-related methods."""

    def test_session_is_mfa_fresh_method(self):
        """Test JWTSession.is_mfa_fresh method."""
        session = JWTSession(
            session_id="test",
            user_id="user",
            created_at=time.time(),
            last_activity=time.time(),
        )

        # No MFA verification
        assert session.is_mfa_fresh() is False
        assert session.is_mfa_fresh(3600) is False

        # Verify MFA
        session.record_mfa_verification(["totp"])

        # Should be fresh
        assert session.is_mfa_fresh() is True
        assert session.is_mfa_fresh(3600) is True

    def test_session_mfa_age_seconds(self):
        """Test JWTSession.mfa_age_seconds method."""
        session = JWTSession(
            session_id="test",
            user_id="user",
            created_at=time.time(),
            last_activity=time.time(),
        )

        # No MFA - should return None
        assert session.mfa_age_seconds() is None

        # Verify MFA
        session.record_mfa_verification(["totp"])

        # Should have small age
        age = session.mfa_age_seconds()
        assert age is not None
        assert age < 5

    def test_session_record_mfa_verification(self):
        """Test JWTSession.record_mfa_verification method."""
        session = JWTSession(
            session_id="test",
            user_id="user",
            created_at=time.time(),
            last_activity=time.time(),
        )

        # Record verification
        session.record_mfa_verification(["backup_code"])

        assert session.mfa_verified_at is not None
        assert session.mfa_methods_used == ["backup_code"]

        # Default methods
        session2 = JWTSession(
            session_id="test2",
            user_id="user",
            created_at=time.time(),
            last_activity=time.time(),
        )
        session2.record_mfa_verification()
        assert session2.mfa_methods_used == ["totp"]


__all__ = [
    "TestMFAFreshWithin15Minutes",
    "TestMFAStaleAfter15Minutes",
    "TestStepUpAuth",
    "TestMFAFreshnessReset",
    "TestMFAMethodsTracking",
    "TestJWTSessionMFA",
]
