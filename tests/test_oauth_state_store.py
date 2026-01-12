"""
Tests for OAuth State Store.

Tests both in-memory and Redis backends for OAuth state management.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from aragora.server.oauth_state_store import (
    OAuthState,
    InMemoryOAuthStateStore,
    RedisOAuthStateStore,
    FallbackOAuthStateStore,
    get_oauth_state_store,
    reset_oauth_state_store,
    generate_oauth_state,
    validate_oauth_state,
)


class TestOAuthState:
    """Tests for OAuthState dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        state = OAuthState(
            user_id="user123",
            redirect_url="http://localhost:3000/callback",
            expires_at=time.time() + 600,
            created_at=time.time(),
        )
        data = state.to_dict()
        assert data["user_id"] == "user123"
        assert data["redirect_url"] == "http://localhost:3000/callback"
        assert "expires_at" in data
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "user_id": "user456",
            "redirect_url": "http://example.com",
            "expires_at": time.time() + 600,
            "created_at": time.time(),
        }
        state = OAuthState.from_dict(data)
        assert state.user_id == "user456"
        assert state.redirect_url == "http://example.com"

    def test_is_expired(self):
        """Test expiration check."""
        # Not expired
        state = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() + 600,
        )
        assert state.is_expired is False

        # Expired
        state_expired = OAuthState(
            user_id=None,
            redirect_url=None,
            expires_at=time.time() - 1,
        )
        assert state_expired.is_expired is True


class TestInMemoryOAuthStateStore:
    """Tests for in-memory OAuth state store."""

    def test_generate_state(self):
        """Test state generation."""
        store = InMemoryOAuthStateStore()
        state = store.generate(user_id="user123", redirect_url="http://localhost")
        assert state is not None
        assert len(state) > 20  # URL-safe token should be long

    def test_validate_and_consume(self):
        """Test state validation and consumption."""
        store = InMemoryOAuthStateStore()
        state = store.generate(user_id="user123", redirect_url="http://localhost")

        # First validation should succeed
        result = store.validate_and_consume(state)
        assert result is not None
        assert result.user_id == "user123"
        assert result.redirect_url == "http://localhost"

        # Second validation should fail (consumed)
        result2 = store.validate_and_consume(state)
        assert result2 is None

    def test_invalid_state(self):
        """Test validation of non-existent state."""
        store = InMemoryOAuthStateStore()
        result = store.validate_and_consume("invalid_state_token")
        assert result is None

    def test_expired_state(self):
        """Test that expired states are rejected."""
        store = InMemoryOAuthStateStore()
        state = store.generate(ttl_seconds=0)  # Expires immediately

        # Wait a moment for expiration
        time.sleep(0.01)

        result = store.validate_and_consume(state)
        assert result is None

    def test_max_size_eviction(self):
        """Test that oldest states are evicted when max size reached."""
        store = InMemoryOAuthStateStore(max_size=5)

        # Generate 6 states (exceeds max)
        states = []
        for i in range(6):
            states.append(store.generate(user_id=f"user{i}"))

        # Should have evicted some old states
        assert store.size() <= 5

    def test_cleanup_expired(self):
        """Test cleanup of expired states."""
        store = InMemoryOAuthStateStore()

        # Generate states with very short TTL
        store.generate(ttl_seconds=1)
        store.generate(ttl_seconds=1)

        # Manually expire them by modifying expires_at
        with store._lock:
            for state_data in store._states.values():
                state_data.expires_at = time.time() - 1

        removed = store.cleanup_expired()
        assert removed >= 2


class TestFallbackOAuthStateStore:
    """Tests for fallback OAuth state store (no Redis configured)."""

    def test_without_redis_uses_memory(self):
        """Test that store uses memory when Redis URL not set."""
        store = FallbackOAuthStateStore(redis_url="")
        assert store.is_using_redis is False

    def test_generate_without_redis(self):
        """Test state generation without Redis."""
        store = FallbackOAuthStateStore(redis_url="")
        state = store.generate(user_id="user123")
        assert state is not None

    def test_validate_without_redis(self):
        """Test state validation without Redis."""
        store = FallbackOAuthStateStore(redis_url="")
        state = store.generate(user_id="user123")

        result = store.validate_and_consume(state)
        assert result is not None
        assert result.user_id == "user123"

    def test_redis_fallback_on_connection_failure(self):
        """Test fallback to memory when Redis connection fails."""
        store = FallbackOAuthStateStore(redis_url="redis://nonexistent:6379")

        # Should fall back to memory and still work
        state = store.generate(user_id="user123")
        assert state is not None

        result = store.validate_and_consume(state)
        assert result is not None
        assert result.user_id == "user123"

        # Should now be using memory
        assert store.is_using_redis is False


class TestRedisOAuthStateStore:
    """Tests for Redis OAuth state store (requires mock)."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.ping.return_value = True
        return mock

    def test_generate_with_mock_redis(self, mock_redis):
        """Test state generation with mocked Redis."""
        # Import redis module to patch
        try:
            import redis as redis_module
            with patch.object(redis_module, "from_url", return_value=mock_redis):
                store = RedisOAuthStateStore(redis_url="redis://localhost:6379")
                state = store.generate(user_id="user123")

                assert state is not None
                mock_redis.setex.assert_called_once()
        except ImportError:
            pytest.skip("redis package not installed")

    def test_validate_with_mock_redis(self, mock_redis):
        """Test state validation with mocked Redis."""
        import json

        try:
            import redis as redis_module

            # Mock pipeline for atomic get-delete
            mock_pipe = MagicMock()
            mock_redis.pipeline.return_value = mock_pipe

            state_data = {
                "user_id": "user123",
                "redirect_url": "http://localhost",
                "expires_at": time.time() + 600,
                "created_at": time.time(),
            }
            mock_pipe.execute.return_value = [json.dumps(state_data), 1]

            with patch.object(redis_module, "from_url", return_value=mock_redis):
                store = RedisOAuthStateStore(redis_url="redis://localhost:6379")
                result = store.validate_and_consume("test_state")

                assert result is not None
                assert result.user_id == "user123"
        except ImportError:
            pytest.skip("redis package not installed")


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset global store before each test."""
        reset_oauth_state_store()

    def teardown_method(self):
        """Reset global store after each test."""
        reset_oauth_state_store()

    def test_generate_oauth_state(self):
        """Test generate_oauth_state convenience function."""
        state = generate_oauth_state(user_id="user123")
        assert state is not None
        assert len(state) > 20

    def test_validate_oauth_state(self):
        """Test validate_oauth_state convenience function."""
        state = generate_oauth_state(
            user_id="user123",
            redirect_url="http://localhost:3000/callback",
        )

        result = validate_oauth_state(state)
        assert result is not None
        assert result["user_id"] == "user123"
        assert result["redirect_url"] == "http://localhost:3000/callback"

    def test_validate_invalid_state(self):
        """Test validation of invalid state returns None."""
        result = validate_oauth_state("invalid_state")
        assert result is None

    def test_get_oauth_state_store_singleton(self):
        """Test that get_oauth_state_store returns singleton."""
        store1 = get_oauth_state_store()
        store2 = get_oauth_state_store()
        assert store1 is store2


class TestOAuthIntegration:
    """Integration tests with OAuth handler."""

    def setup_method(self):
        """Reset global store before each test."""
        reset_oauth_state_store()

    def teardown_method(self):
        """Reset global store after each test."""
        reset_oauth_state_store()

    def test_oauth_flow_state_generation(self):
        """Test OAuth flow state generation."""
        # Generate state as would happen in OAuth start
        state = generate_oauth_state(
            user_id=None,  # New user flow
            redirect_url="http://localhost:3000/auth/callback",
        )

        # Validate as would happen in callback
        state_data = validate_oauth_state(state)
        assert state_data is not None
        assert state_data["redirect_url"] == "http://localhost:3000/auth/callback"

    def test_oauth_linking_flow(self):
        """Test OAuth account linking flow."""
        # Generate state for account linking (user already authenticated)
        state = generate_oauth_state(
            user_id="existing_user_id",
            redirect_url="http://localhost:3000/settings",
        )

        # Validate in callback
        state_data = validate_oauth_state(state)
        assert state_data is not None
        assert state_data["user_id"] == "existing_user_id"

    def test_state_single_use(self):
        """Test that OAuth state can only be used once."""
        state = generate_oauth_state(user_id="user123")

        # First use succeeds
        result1 = validate_oauth_state(state)
        assert result1 is not None

        # Second use fails
        result2 = validate_oauth_state(state)
        assert result2 is None
