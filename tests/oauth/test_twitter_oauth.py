"""
Twitter OAuth E2E tests - supplementary tests.

These tests supplement the main test_handlers_oauth.py with additional
coverage for Twitter/X OAuth and publishing functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestTwitterPublishingConfiguration:
    """Test Twitter publishing configuration and validation."""

    def test_twitter_connector_has_is_configured(self):
        """Twitter connector should have is_configured property."""
        from aragora.connectors.twitter_poster import TwitterPosterConnector

        # Create with missing credentials
        connector = TwitterPosterConnector(
            api_key="",
            api_secret="",
            access_token="",
            access_secret="",
        )
        assert hasattr(connector, "is_configured")
        assert connector.is_configured is False

    def test_twitter_connector_configured_with_credentials(self):
        """Twitter connector should be configured when credentials provided."""
        from aragora.connectors.twitter_poster import TwitterPosterConnector

        connector = TwitterPosterConnector(
            api_key="test-key",
            api_secret="test-secret",
            access_token="test-token",
            access_secret="test-access-secret",
        )
        assert connector.is_configured is True

    def test_twitter_rate_limiter_exists(self):
        """Twitter rate limiter should be available."""
        from aragora.connectors.twitter_poster import TwitterRateLimiter

        limiter = TwitterRateLimiter()
        # Rate limiter uses async acquire() method
        assert hasattr(limiter, "acquire")
        assert hasattr(limiter, "calls_per_window")
        assert hasattr(limiter, "window_seconds")


class TestTwitterErrorHandling:
    """Test Twitter error handling classes."""

    def test_twitter_error_hierarchy(self):
        """Twitter errors should have proper inheritance."""
        from aragora.connectors.twitter_poster import (
            TwitterError,
            TwitterAuthError,
            TwitterRateLimitError,
            TwitterAPIError,
            TwitterMediaError,
        )
        from aragora.connectors.exceptions import ConnectorError

        # All should inherit from TwitterError
        assert issubclass(TwitterAuthError, TwitterError)
        assert issubclass(TwitterRateLimitError, TwitterError)
        assert issubclass(TwitterAPIError, TwitterError)
        assert issubclass(TwitterMediaError, TwitterError)

        # TwitterError should inherit from ConnectorError
        assert issubclass(TwitterError, ConnectorError)

    def test_twitter_errors_can_be_raised(self):
        """Twitter errors should be raisable with messages."""
        from aragora.connectors.twitter_poster import (
            TwitterAuthError,
            TwitterRateLimitError,
            TwitterMediaError,
        )

        with pytest.raises(TwitterAuthError):
            raise TwitterAuthError("Auth failed")

        with pytest.raises(TwitterRateLimitError):
            raise TwitterRateLimitError("Rate limited")

        with pytest.raises(TwitterMediaError):
            raise TwitterMediaError("Media upload failed")


class TestTwitterPublishingRoutes:
    """Test Twitter publishing route definitions."""

    def test_handler_includes_twitter_publish_route(self):
        """Handler should include Twitter publish route."""
        from aragora.server.handlers.social import SocialMediaHandler

        routes = SocialMediaHandler.ROUTES
        assert any("/publish/twitter" in route for route in routes)

    def test_twitter_publish_path_pattern(self):
        """Twitter publish path should match expected pattern."""
        from aragora.server.handlers.social import SocialMediaHandler

        routes = SocialMediaHandler.ROUTES
        twitter_routes = [r for r in routes if "twitter" in r.lower()]
        assert len(twitter_routes) > 0


class TestTwitterContentFormatter:
    """Test Twitter content formatting utilities."""

    def test_debate_content_formatter_exists(self):
        """DebateContentFormatter should be available for Twitter."""
        from aragora.connectors.twitter_poster import DebateContentFormatter

        formatter = DebateContentFormatter()
        assert hasattr(formatter, "format_announcement")
        assert hasattr(formatter, "format_result")
        assert hasattr(formatter, "format_thread")

    def test_formatter_has_max_length(self):
        """Formatter should have MAX_LENGTH constant."""
        from aragora.connectors.twitter_poster import DebateContentFormatter, MAX_TWEET_LENGTH

        formatter = DebateContentFormatter()
        assert hasattr(formatter, "MAX_LENGTH")
        assert formatter.MAX_LENGTH == MAX_TWEET_LENGTH
        assert formatter.MAX_LENGTH == 280

    def test_format_announcement_respects_limit(self):
        """format_announcement should respect Twitter character limits."""
        from aragora.connectors.twitter_poster import DebateContentFormatter

        formatter = DebateContentFormatter()

        announcement = formatter.format_announcement(
            task="A very long debate topic " * 20,
            agents=["claude", "gpt-4", "gemini"],
        )

        # Should be within Twitter limit
        assert len(announcement) <= 280

    def test_format_announcement_includes_agents(self):
        """format_announcement should include agent names."""
        from aragora.connectors.twitter_poster import DebateContentFormatter

        formatter = DebateContentFormatter()

        announcement = formatter.format_announcement(
            task="Test debate topic",
            agents=["claude", "gpt-4"],
        )

        # Should mention at least one agent
        assert "claude" in announcement or "gpt" in announcement.lower()


class TestTwitterSecurityMeasures:
    """Test Twitter publishing security measures."""

    def test_credentials_not_logged(self):
        """Credentials should not appear in string representations."""
        from aragora.connectors.twitter_poster import TwitterPosterConnector

        connector = TwitterPosterConnector(
            api_key="secret-api-key",
            api_secret="secret-api-secret",
            access_token="secret-token",
            access_secret="secret-access-secret",
        )

        # String representation should not contain secrets
        str_repr = str(connector)
        assert "secret-api-key" not in str_repr
        assert "secret-api-secret" not in str_repr
        assert "secret-token" not in str_repr
        assert "secret-access-secret" not in str_repr

    def test_rate_limiter_has_window_config(self):
        """Rate limiter should have configurable window."""
        from aragora.connectors.twitter_poster import TwitterRateLimiter

        # Default config
        limiter = TwitterRateLimiter()
        assert limiter.calls_per_window == 50
        assert limiter.window_seconds == 900

        # Custom config
        custom_limiter = TwitterRateLimiter(calls_per_window=10, window_seconds=60)
        assert custom_limiter.calls_per_window == 10
        assert custom_limiter.window_seconds == 60


class TestTwitterOAuthStateIntegration:
    """Test Twitter OAuth state integration with social handler."""

    def test_social_handler_oauth_state_functions(self):
        """Social handler should have OAuth state functions."""
        from aragora.server.handlers.social import _store_oauth_state, _validate_oauth_state

        import secrets

        state = secrets.token_urlsafe(32)

        # Store and validate
        _store_oauth_state(state)
        assert _validate_oauth_state(state) is True

        # Should be consumed
        assert _validate_oauth_state(state) is False

    def test_oauth_state_uniqueness_for_twitter(self):
        """OAuth states should be unique for Twitter flows."""
        import secrets

        states = [secrets.token_urlsafe(32) for _ in range(100)]

        # All should be unique
        assert len(set(states)) == 100

        # All should have sufficient entropy
        for state in states:
            assert len(state) >= 32

    def test_oauth_prevents_replay(self):
        """OAuth state should prevent replay attacks."""
        from aragora.server.handlers.social import _store_oauth_state, _validate_oauth_state
        import secrets

        state = secrets.token_urlsafe(32)
        _store_oauth_state(state)

        # First validation succeeds
        assert _validate_oauth_state(state) is True

        # Replay attempt fails
        assert _validate_oauth_state(state) is False
