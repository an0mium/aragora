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


class TestOAuthConfiguration:
    """Test OAuth configuration validation."""

    def test_allows_missing_vars_in_dev_mode(self):
        """Should allow missing vars in development mode."""
        from aragora.server.handlers.oauth import validate_oauth_config

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            missing = validate_oauth_config()
            assert missing == []  # No validation in dev mode
