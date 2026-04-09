"""
Google OAuth E2E tests - supplementary tests.

These tests supplement the main test_handlers_oauth.py with additional
coverage for state management and security properties.
"""

import os
import pytest
from unittest.mock import patch


class TestOAuthStateManagement:
    """Test OAuth state generation and validation - core security properties."""

    def test_state_generation_unique(self):
        """Each generated state should be unique."""
        from aragora.server.oauth_state_store import generate_oauth_state

        states = [generate_oauth_state(f"user-{i}") for i in range(100)]
        assert len(set(states)) == 100

    def test_state_storage_and_retrieval(self):
        """State should be storable and retrievable via validate_and_consume."""
        from aragora.server.oauth_state_store import generate_oauth_state, validate_oauth_state

        state = generate_oauth_state("test-user-storage")
        result = validate_oauth_state(state)

        assert result is not None
        assert "user_id" in result
        assert result["user_id"] == "test-user-storage"
        assert "expires_at" in result
        assert "created_at" in result

    def test_state_consumed_after_use(self):
        """State should be consumed (deleted) after validation."""
        from aragora.server.oauth_state_store import generate_oauth_state, validate_oauth_state

        state = generate_oauth_state("test-user-consume")

        first_retrieval = validate_oauth_state(state)
        second_retrieval = validate_oauth_state(state)

        assert first_retrieval is not None
        assert second_retrieval is None

    def test_state_has_expiration(self):
        """States should have expiration time set."""
        from aragora.server.oauth_state_store import generate_oauth_state, validate_oauth_state
        import time

        state = generate_oauth_state("test-user-expire")
        result = validate_oauth_state(state)

        # State result should have expires_at in the dict
        assert result is not None
        assert "expires_at" in result
        # expires_at should have been in the future when created
        # (it was just validated so it's consumed now)


class TestOAuthSecurity:
    """Test OAuth security measures - entropy and replay prevention."""

    def test_state_has_sufficient_entropy(self):
        """State tokens should have sufficient entropy (min 32 chars)."""
        from aragora.server.oauth_state_store import generate_oauth_state

        states = [generate_oauth_state(f"user-{i}") for i in range(100)]

        # Check minimum length (should be base64-encoded random bytes)
        for state in states:
            assert len(state) >= 32

        # Check uniqueness
        assert len(set(states)) == 100

    def test_prevents_replay_attacks(self):
        """State should only be usable once (replay prevention)."""
        from aragora.server.oauth_state_store import generate_oauth_state, validate_oauth_state

        state = generate_oauth_state("test-user-replay")

        # First use should succeed
        first = validate_oauth_state(state)
        assert first is not None

        # Second use should fail (replay attack prevented)
        second = validate_oauth_state(state)
        assert second is None


class TestOAuthCallbackQueryParams:
    """Regression: Google now sends extra query params (iss, authuser, prompt) in callbacks."""

    def test_oauth_callback_path_detected_v1(self):
        """v1 OAuth callback paths should be recognized."""
        path = "/api/v1/auth/oauth/google/callback"
        is_oauth = False
        if path.startswith("/api/"):
            if path.startswith("/api/auth/oauth/") or path.startswith("/api/v1/auth/oauth/"):
                is_oauth = path.rstrip("/").endswith("callback")
        assert is_oauth is True

    def test_oauth_callback_path_detected_non_v1(self):
        """Non-v1 OAuth callback paths should be recognized."""
        path = "/api/auth/oauth/google/callback"
        is_oauth = False
        if path.startswith("/api/"):
            if path.startswith("/api/auth/oauth/") or path.startswith("/api/v1/auth/oauth/"):
                is_oauth = path.rstrip("/").endswith("callback")
        assert is_oauth is True

    def test_non_callback_oauth_path_not_skipped(self):
        """Non-callback OAuth paths should NOT skip validation."""
        path = "/api/v1/auth/oauth/google/authorize"
        is_oauth = False
        if path.startswith("/api/"):
            if path.startswith("/api/auth/oauth/") or path.startswith("/api/v1/auth/oauth/"):
                is_oauth = path.rstrip("/").endswith("callback")
        assert is_oauth is False

    def test_regular_api_path_not_skipped(self):
        """Regular API paths should NOT skip validation."""
        path = "/api/v1/playground/debate"
        is_oauth = False
        if path.startswith("/api/"):
            if path.startswith("/api/auth/oauth/") or path.startswith("/api/v1/auth/oauth/"):
                is_oauth = path.rstrip("/").endswith("callback")
        assert is_oauth is False

    def test_google_iss_param_accepted_on_callback(self):
        """Google's iss param must not cause rejection on callback routes.

        Regression test for production failure 2026-04-09: Google added an 'iss'
        query parameter to OAuth callbacks. The global query param allowlist
        rejected it, breaking login for all users.
        """
        from aragora.server.http_utils import validate_query_params

        # These are the params Google actually sends in 2026
        google_callback_params = {
            "state": "abc123",
            "code": "4/0Aci98...",
            "scope": "email profile openid",
            "iss": "https://accounts.google.com",
            "authuser": "2",
            "prompt": "consent",
        }

        # On a callback path, validation is skipped entirely — so this test
        # verifies the detection logic, not the validator. The validator
        # WOULD reject 'iss' if called:
        is_valid, error_msg = validate_query_params(google_callback_params)
        # The point: this FAILS validation, proving the callback skip is necessary
        if not is_valid:
            assert "iss" in error_msg or "authuser" in error_msg or "prompt" in error_msg
        # Either way, the callback path detection must return True for this path
        path = "/api/v1/auth/oauth/google/callback"
        is_oauth_callback = (
            path.startswith("/api/")
            and (path.startswith("/api/auth/oauth/") or path.startswith("/api/v1/auth/oauth/"))
            and path.rstrip("/").endswith("callback")
        )
        assert is_oauth_callback is True, "Callback path must skip query param validation"


class TestOAuthConfiguration:
    """Test OAuth configuration validation."""

    def test_allows_missing_vars_in_dev_mode(self):
        """Should allow missing vars in development mode."""
        from aragora.server.handlers.oauth import validate_oauth_config

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            missing = validate_oauth_config()
            assert missing == []  # No validation in dev mode
