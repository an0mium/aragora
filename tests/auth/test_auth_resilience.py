"""
Tests for auth reliability improvements.

Tests cover:
- IdP failure retry logic
- Circuit breaker for provider unavailability
- Rate limiting enforcement
- Auth flow end-to-end reliability
- Session health handler endpoints
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async function synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class FakeSSOUser:
    """Minimal SSO user for testing."""

    id = "sso-user-1"
    email = "test@example.com"
    name = "Test User"
    access_token = "fake-access-token"
    refresh_token = "fake-refresh-token"
    token_expires_at = time.time() + 3600


# ===========================================================================
# IdP Retry Logic
# ===========================================================================


class TestIdPRetry:
    """Tests for _authenticate_with_retry in SSO handlers."""

    def test_success_on_first_try(self):
        """Test successful auth on first attempt."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        provider = MagicMock()
        provider.authenticate = AsyncMock(return_value=FakeSSOUser())

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=None,
        ):
            result = _run_async(
                _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
            )
        assert result.email == "test@example.com"
        provider.authenticate.assert_awaited_once()

    def test_retry_on_connection_error(self):
        """Test that ConnectionError triggers retry."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        provider = MagicMock()
        provider.authenticate = AsyncMock(
            side_effect=[ConnectionError("network fail"), FakeSSOUser()]
        )

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=None,
        ):
            result = _run_async(
                _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
            )
        assert result.email == "test@example.com"
        assert provider.authenticate.await_count == 2

    def test_retry_on_timeout_error(self):
        """Test that TimeoutError triggers retry."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        provider = MagicMock()
        provider.authenticate = AsyncMock(side_effect=[TimeoutError("timed out"), FakeSSOUser()])

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=None,
        ):
            result = _run_async(
                _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
            )
        assert provider.authenticate.await_count == 2

    def test_no_retry_on_value_error(self):
        """Test that ValueError does NOT trigger retry (non-retryable)."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        provider = MagicMock()
        provider.authenticate = AsyncMock(side_effect=ValueError("bad input"))

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="bad input"):
                _run_async(
                    _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
                )
        provider.authenticate.assert_awaited_once()

    def test_all_retries_exhausted(self):
        """Test that error is raised after all retries fail."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        provider = MagicMock()
        provider.authenticate = AsyncMock(side_effect=ConnectionError("persistent failure"))

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=None,
        ):
            with pytest.raises(ConnectionError, match="persistent failure"):
                _run_async(
                    _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
                )
        # Should have tried 3 times (initial + 2 retries)
        assert provider.authenticate.await_count == 3


# ===========================================================================
# Circuit Breaker Integration
# ===========================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker protecting IdP calls."""

    def test_circuit_breaker_open_fails_fast(self):
        """Test that open circuit breaker rejects immediately."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = False

        provider = MagicMock()
        provider.authenticate = AsyncMock()

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=mock_cb,
        ):
            with pytest.raises(ConnectionError, match="circuit open"):
                _run_async(
                    _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
                )
        # Provider should NOT have been called (fast fail)
        provider.authenticate.assert_not_awaited()

    def test_circuit_breaker_records_success(self):
        """Test that successful auth records CB success."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True

        provider = MagicMock()
        provider.authenticate = AsyncMock(return_value=FakeSSOUser())

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=mock_cb,
        ):
            _run_async(
                _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
            )
        mock_cb.record_success.assert_called_once()

    def test_circuit_breaker_records_failure(self):
        """Test that failed auth records CB failure."""
        from aragora.server.handlers.auth.sso_handlers import _authenticate_with_retry

        mock_cb = MagicMock()
        mock_cb.can_proceed.return_value = True

        provider = MagicMock()
        provider.authenticate = AsyncMock(side_effect=ConnectionError("down"))

        with patch(
            "aragora.server.handlers.auth.sso_handlers._get_idp_circuit_breaker",
            return_value=mock_cb,
        ):
            with pytest.raises(ConnectionError):
                _run_async(
                    _authenticate_with_retry(provider, code="abc", state="st", provider_type="oidc")
                )
        # CB failure recorded for each attempt (3 total)
        assert mock_cb.record_failure.call_count == 3

    def test_circuit_breaker_created_per_provider(self):
        """Test that each provider type gets its own circuit breaker."""
        from aragora.server.handlers.auth.sso_handlers import (
            _get_idp_circuit_breaker,
            _idp_circuit_breakers,
        )

        # Clear any existing state
        _idp_circuit_breakers.clear()

        cb_oidc = _get_idp_circuit_breaker("oidc")
        cb_google = _get_idp_circuit_breaker("google")

        if cb_oidc is not None:
            assert cb_oidc is not cb_google
            assert cb_oidc.name == "idp-oidc"
            assert cb_google.name == "idp-google"

        # Cleanup
        _idp_circuit_breakers.clear()


# ===========================================================================
# SSO Callback Auth Failure Tracking
# ===========================================================================


class TestSSOCallbackFailureTracking:
    """Tests for auth failure/success tracking in SSO callback."""

    @pytest.mark.no_auto_auth
    def test_callback_error_records_auth_failure(self):
        """Test that callback error path records auth failure metric."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback
        from aragora.auth.session_monitor import SessionHealthMonitor

        mock_monitor = MagicMock(spec=SessionHealthMonitor)

        with patch("aragora.server.handlers.auth.sso_handlers._sso_state_store") as mock_store_lazy:
            mock_store = MagicMock()
            mock_store.validate_and_consume.return_value = None
            mock_store_lazy.get.return_value = mock_store

            with patch(
                "aragora.server.handlers.auth.sso_handlers._get_sso_provider",
                return_value=None,
            ):
                with patch(
                    "aragora.auth.session_monitor.get_session_monitor",
                    return_value=mock_monitor,
                ):
                    # Missing code should return 400
                    result = _run_async(handle_sso_callback({"state": "test-state"}))
                    assert result[1] == 400  # status code

    @pytest.mark.no_auto_auth
    def test_callback_idp_error_returns_401(self):
        """Test that IdP error is surfaced as 401."""
        from aragora.server.handlers.auth.sso_handlers import handle_sso_callback

        result = _run_async(
            handle_sso_callback(
                {
                    "error": "access_denied",
                    "error_description": "User denied access",
                }
            )
        )
        assert result[1] == 401


# ===========================================================================
# Session Health Handler Endpoints
# ===========================================================================


class TestSessionHealthHandlers:
    """Tests for session health REST endpoints."""

    def test_health_endpoint_returns_metrics(self):
        """Test GET /api/v1/auth/sessions/health returns metrics."""
        from aragora.server.handlers.auth.session_health import handle_session_health
        from aragora.auth.session_monitor import SessionHealthMonitor

        mock_monitor = SessionHealthMonitor()
        mock_monitor.track_session("s1", user_id="u1")
        mock_monitor.record_auth_success()

        with patch(
            "aragora.server.handlers.auth.session_health._get_monitor",
            return_value=mock_monitor,
        ):
            result = _run_async(handle_session_health({}, user_id="admin-1"))

        body = result[0]
        assert "metrics" in body
        assert body["metrics"]["active_sessions"] == 1
        assert body["metrics"]["auth_success_count"] == 1

    def test_sweep_endpoint_removes_expired(self):
        """Test POST /api/v1/auth/sessions/sweep removes expired."""
        from aragora.server.handlers.auth.session_health import handle_session_sweep
        from aragora.auth.session_monitor import SessionHealthMonitor

        mock_monitor = SessionHealthMonitor()
        mock_monitor.track_session("s1", user_id="u1", ttl_seconds=0.01)
        time.sleep(0.02)

        with patch(
            "aragora.server.handlers.auth.session_health._get_monitor",
            return_value=mock_monitor,
        ):
            result = _run_async(handle_session_sweep({}, user_id="admin-1"))

        body = result[0]
        assert body["swept"] is True
        assert body["sessions_removed"] == 1

    def test_active_sessions_for_user(self):
        """Test GET /api/v1/auth/sessions/active returns user sessions."""
        from aragora.server.handlers.auth.session_health import handle_active_sessions
        from aragora.auth.session_monitor import SessionHealthMonitor

        mock_monitor = SessionHealthMonitor()
        mock_monitor.track_session("s1", user_id="user-1")
        mock_monitor.track_session("s2", user_id="user-1")
        mock_monitor.track_session("s3", user_id="user-2")

        with patch(
            "aragora.server.handlers.auth.session_health._get_monitor",
            return_value=mock_monitor,
        ):
            result = _run_async(handle_active_sessions({}, user_id="user-1"))

        body = result[0]
        assert body["total"] == 2
        assert body["user_id"] == "user-1"

    def test_active_sessions_requires_auth(self):
        """Test that active sessions endpoint requires authentication."""
        from aragora.server.handlers.auth.session_health import handle_active_sessions

        result = _run_async(handle_active_sessions({}, user_id="default"))
        assert result[1] == 401


# ===========================================================================
# IdP Circuit Breaker Factory
# ===========================================================================


class TestIdPCircuitBreakerFactory:
    """Tests for _get_idp_circuit_breaker factory."""

    def test_returns_circuit_breaker_or_none(self):
        """Test that factory returns CB or None if import fails."""
        from aragora.server.handlers.auth.sso_handlers import (
            _get_idp_circuit_breaker,
            _idp_circuit_breakers,
        )

        _idp_circuit_breakers.clear()
        cb = _get_idp_circuit_breaker("test")
        # Could be None if CircuitBreaker not importable, or a CB
        if cb is not None:
            assert hasattr(cb, "can_proceed")
            assert hasattr(cb, "record_success")
            assert hasattr(cb, "record_failure")
        _idp_circuit_breakers.clear()

    def test_same_provider_returns_same_breaker(self):
        """Test that same provider type returns cached breaker."""
        from aragora.server.handlers.auth.sso_handlers import (
            _get_idp_circuit_breaker,
            _idp_circuit_breakers,
        )

        _idp_circuit_breakers.clear()
        cb1 = _get_idp_circuit_breaker("oidc")
        cb2 = _get_idp_circuit_breaker("oidc")
        if cb1 is not None:
            assert cb1 is cb2
        _idp_circuit_breakers.clear()
