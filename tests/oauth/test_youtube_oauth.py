"""
YouTube OAuth E2E tests - supplementary tests.

These tests supplement the main test_handlers_oauth.py with additional
coverage for YouTube/social OAuth state management and security properties.
"""

import pytest
from unittest.mock import patch


class TestYouTubeOAuthStateManagement:
    """Test YouTube OAuth state management - core security properties."""

    def test_store_and_validate_state(self):
        """Should store and validate state correctly."""
        from aragora.server.handlers.social import _store_oauth_state, _validate_oauth_state
        import secrets

        state = secrets.token_urlsafe(32)

        _store_oauth_state(state)
        assert _validate_oauth_state(state) is True

    def test_state_consumed_after_validation(self):
        """State should be consumed after validation."""
        from aragora.server.handlers.social import _store_oauth_state, _validate_oauth_state
        import secrets

        state = secrets.token_urlsafe(32)

        _store_oauth_state(state)
        first_validation = _validate_oauth_state(state)
        second_validation = _validate_oauth_state(state)

        assert first_validation is True
        assert second_validation is False

    def test_invalid_state_rejected(self):
        """Invalid state should be rejected."""
        from aragora.server.handlers.social import _validate_oauth_state

        assert _validate_oauth_state("non-existent-state") is False

    def test_state_unique_per_request(self):
        """Each state should be unique."""
        import secrets

        states = [secrets.token_urlsafe(32) for _ in range(100)]

        # All states should be unique
        assert len(set(states)) == 100


class TestYouTubeOAuthSecurity:
    """Test YouTube OAuth security measures."""

    def test_state_has_sufficient_entropy(self):
        """State tokens should have sufficient entropy (min 32 chars)."""
        import secrets

        states = [secrets.token_urlsafe(32) for _ in range(100)]

        # Check minimum length
        for state in states:
            assert len(state) >= 32

        # Check uniqueness
        assert len(set(states)) == 100

    def test_prevents_replay_attacks(self):
        """State should only be usable once (replay prevention)."""
        from aragora.server.handlers.social import _store_oauth_state, _validate_oauth_state
        import secrets

        state = secrets.token_urlsafe(32)

        _store_oauth_state(state)

        # First use should succeed
        first = _validate_oauth_state(state)
        assert first is True

        # Second use should fail (replay attack prevented)
        second = _validate_oauth_state(state)
        assert second is False


class TestSocialMediaHandlerRoutes:
    """Test social media handler route definitions."""

    def test_handler_routes_defined(self):
        """Handler should have routes defined."""
        from aragora.server.handlers.social import SocialMediaHandler

        assert hasattr(SocialMediaHandler, "ROUTES")
        routes = SocialMediaHandler.ROUTES

        # Should include publishing routes
        assert any("/publish/twitter" in route for route in routes)
        assert any("/publish/youtube" in route for route in routes)

    def test_oauth_hosts_configurable(self):
        """OAuth allowed hosts should be configurable."""
        from aragora.server.handlers import social

        # Should have ALLOWED_OAUTH_HOSTS defined
        assert hasattr(social, "ALLOWED_OAUTH_HOSTS")
        assert isinstance(social.ALLOWED_OAUTH_HOSTS, frozenset)
