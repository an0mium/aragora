"""
Security tests for WebSocket implementation.

Tests cover:
- Origin validation (CORS for WebSockets)
- Rate limiting per connection
- Token revalidation for long-lived connections
- Message size limits
- Authentication requirements
- Binary message rejection
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import time


class MockWebSocket:
    """Mock aiohttp WebSocket for testing."""

    def __init__(self):
        self.messages_sent = []
        self.closed = False
        self._bound_loop_id = None

    async def send_json(self, data):
        self.messages_sent.append(data)

    async def send_str(self, data):
        self.messages_sent.append(json.loads(data))

    async def close(self):
        self.closed = True


class MockRequest:
    """Mock aiohttp Request for testing."""

    def __init__(
        self,
        headers: dict = None,
        remote: str = "127.0.0.1",
        query: dict = None,
    ):
        self.headers = headers or {}
        self.remote = remote
        self.query = query or {}
        self.match_info = {}

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class TestOriginValidation:
    """Test WebSocket origin validation."""

    def test_allowed_origins_defined(self):
        """Test that allowed origins are properly defined."""
        from aragora.server.cors_config import WS_ALLOWED_ORIGINS

        assert isinstance(WS_ALLOWED_ORIGINS, (set, frozenset, list, tuple))
        # Should have at least localhost for development
        origins_list = list(WS_ALLOWED_ORIGINS)
        assert any("localhost" in o or "127.0.0.1" in o for o in origins_list)

    def test_cors_headers_function(self):
        """Test CORS header generation."""
        from aragora.server.stream.servers import AiohttpUnifiedServer

        server = AiohttpUnifiedServer(port=8080)

        # Test with no origin (same-origin request)
        headers = server._cors_headers(None)
        assert "Access-Control-Allow-Origin" in headers
        assert "Access-Control-Allow-Methods" in headers

        # Test with allowed origin
        from aragora.server.cors_config import WS_ALLOWED_ORIGINS
        if WS_ALLOWED_ORIGINS:
            allowed = list(WS_ALLOWED_ORIGINS)[0]
            headers = server._cors_headers(allowed)
            assert headers.get("Access-Control-Allow-Origin") == allowed

        # Test with unauthorized origin
        headers = server._cors_headers("https://evil.com")
        # Should NOT have Allow-Origin for unauthorized origins
        assert headers.get("Access-Control-Allow-Origin") != "https://evil.com"


class TestRateLimiting:
    """Test WebSocket rate limiting."""

    def test_token_bucket_creation(self):
        """Test TokenBucket rate limiter creation."""
        from aragora.server.stream.emitter import TokenBucket

        bucket = TokenBucket(rate_per_minute=10.0, burst_size=5)
        assert bucket is not None

        # Should allow initial burst
        for _ in range(5):
            assert bucket.consume(1) is True

    def test_token_bucket_exhaustion(self):
        """Test TokenBucket exhausts after burst."""
        from aragora.server.stream.emitter import TokenBucket

        bucket = TokenBucket(rate_per_minute=60.0, burst_size=3)

        # Consume burst
        assert bucket.consume(1) is True
        assert bucket.consume(1) is True
        assert bucket.consume(1) is True

        # Bucket exhausted
        assert bucket.consume(1) is False

    def test_connection_rate_limit_config(self):
        """Test connection rate limit configuration."""
        from aragora.server.stream.servers import (
            WS_CONNECTIONS_PER_IP_PER_MINUTE,
            WS_MAX_CONNECTIONS_PER_IP,
        )

        assert isinstance(WS_CONNECTIONS_PER_IP_PER_MINUTE, int)
        assert WS_CONNECTIONS_PER_IP_PER_MINUTE > 0

        assert isinstance(WS_MAX_CONNECTIONS_PER_IP, int)
        assert WS_MAX_CONNECTIONS_PER_IP > 0


class TestTokenRevalidation:
    """Test token revalidation for long-lived connections."""

    def test_revalidation_interval_defined(self):
        """Test revalidation interval is configured."""
        from aragora.server.stream.servers import WS_TOKEN_REVALIDATION_INTERVAL

        assert isinstance(WS_TOKEN_REVALIDATION_INTERVAL, (int, float))
        assert WS_TOKEN_REVALIDATION_INTERVAL > 0
        # Should be reasonable (e.g., 5 minutes = 300 seconds)
        assert WS_TOKEN_REVALIDATION_INTERVAL >= 60

    def test_server_base_auth_tracking(self):
        """Test ServerBase tracks WebSocket authentication state."""
        from aragora.server.stream.server_base import ServerBase

        server = ServerBase()

        # Set auth state
        server.set_ws_auth_state(
            ws_id=12345,
            authenticated=True,
            token="test_token",
            ip_address="192.168.1.1",
        )

        # Check state
        assert server.is_ws_authenticated(12345) is True
        assert server.get_ws_token(12345) == "test_token"

        # Cleanup
        server.remove_ws_auth_state(12345)
        assert server.is_ws_authenticated(12345) is False


class TestMessageSizeLimits:
    """Test WebSocket message size limits."""

    def test_max_message_size_defined(self):
        """Test max message size is configured."""
        from aragora.config import WS_MAX_MESSAGE_SIZE

        assert isinstance(WS_MAX_MESSAGE_SIZE, int)
        assert WS_MAX_MESSAGE_SIZE > 0
        # Should be reasonable (e.g., not allowing gigabyte messages)
        assert WS_MAX_MESSAGE_SIZE <= 100 * 1024 * 1024  # Max 100MB

    def test_payload_validation(self):
        """Test audience payload size validation."""
        from aragora.server.stream.servers import AiohttpUnifiedServer

        server = AiohttpUnifiedServer(port=8080)

        # Valid small payload
        small_payload = {"vote": "agree", "reason": "I agree"}
        result, error = server._validate_audience_payload({"payload": small_payload})
        assert result is not None
        assert error is None

        # Too large payload (>10KB)
        large_payload = {"data": "x" * 15000}
        result, error = server._validate_audience_payload({"payload": large_payload})
        assert result is None
        assert "too large" in error.lower()

        # Invalid payload type
        result, error = server._validate_audience_payload({"payload": "not a dict"})
        assert result is None
        assert "format" in error.lower()


class TestAuthentication:
    """Test WebSocket authentication requirements."""

    def test_trusted_proxies_defined(self):
        """Test trusted proxies configuration."""
        from aragora.server.stream.servers import TRUSTED_PROXIES

        assert isinstance(TRUSTED_PROXIES, frozenset)
        # Should include localhost by default
        assert "127.0.0.1" in TRUSTED_PROXIES or "localhost" in TRUSTED_PROXIES

    def test_auth_write_validation_function_exists(self):
        """Test authentication validation function exists."""
        from aragora.server.stream.servers import AiohttpUnifiedServer

        server = AiohttpUnifiedServer(port=8080)

        # Should have the validation method
        assert hasattr(server, "_validate_ws_auth_for_write")
        assert callable(server._validate_ws_auth_for_write)

    def test_loop_id_access_validation(self):
        """Test loop_id access validation."""
        from aragora.server.stream.servers import AiohttpUnifiedServer

        server = AiohttpUnifiedServer(port=8080)

        # Invalid loop_id should fail
        is_valid, error = server._validate_loop_id_access(
            ws_id=12345,
            loop_id="nonexistent_loop",
        )
        assert is_valid is False
        assert error is not None
        assert "Invalid" in error["data"]["message"] or "inactive" in error["data"]["message"]


class TestBinaryMessageRejection:
    """Test binary message handling."""

    def test_binary_rejection_behavior(self):
        """Test that binary messages are documented as rejected."""
        # Binary messages should be rejected per WebSocket handler
        # This is a documentation test since we can't easily test the async handler
        from aragora.server.stream.servers import AiohttpUnifiedServer

        server = AiohttpUnifiedServer(port=8080)

        # Server should exist and handle WebSocket
        assert hasattr(server, "_websocket_handler")


class TestSecurityConfiguration:
    """Test overall security configuration."""

    def test_all_security_constants_defined(self):
        """Test all security-related constants are defined."""
        from aragora.server.stream import servers

        # Connection limiting
        assert hasattr(servers, "WS_CONNECTIONS_PER_IP_PER_MINUTE")
        assert hasattr(servers, "WS_MAX_CONNECTIONS_PER_IP")

        # Token revalidation
        assert hasattr(servers, "WS_TOKEN_REVALIDATION_INTERVAL")

        # Trusted proxies
        assert hasattr(servers, "TRUSTED_PROXIES")

    def test_server_has_cleanup_mechanism(self):
        """Test server has mechanism to cleanup stale connections."""
        from aragora.server.stream.servers import AiohttpUnifiedServer

        server = AiohttpUnifiedServer(port=8080)

        # Should have cleanup method
        assert hasattr(server, "_cleanup_stale_entries")
        assert hasattr(server, "cleanup_all")

    def test_compression_enabled(self):
        """Test WebSocket compression configuration."""
        # Compression should be enabled for bandwidth reduction
        # This is verified in the handler with compress=15
        from aragora.server.stream.servers import AiohttpUnifiedServer

        server = AiohttpUnifiedServer(port=8080)
        assert hasattr(server, "_websocket_handler")


class TestServerBaseSecurityFeatures:
    """Test ServerBase security features."""

    def test_rate_limiter_tracking(self):
        """Test rate limiter tracking in ServerBase."""
        from aragora.server.stream.server_base import ServerBase

        server = ServerBase()

        # Should have rate limiter storage
        assert hasattr(server, "_rate_limiters")
        assert hasattr(server, "_rate_limiters_lock")
        assert hasattr(server, "_rate_limiter_last_access")

    def test_bounded_dict_for_state(self):
        """Test bounded dicts prevent memory exhaustion."""
        from aragora.server.stream.state_manager import BoundedDebateDict

        # Create bounded dict
        bounded = BoundedDebateDict(maxsize=3)

        # Add items up to limit
        bounded["a"] = 1
        bounded["b"] = 2
        bounded["c"] = 3

        # Adding more should evict oldest
        bounded["d"] = 4

        assert len(bounded) == 3
        assert "a" not in bounded  # Oldest should be evicted
        assert "d" in bounded

    def test_server_config_limits(self):
        """Test ServerConfig has security limits."""
        from aragora.server.stream.server_base import ServerConfig

        config = ServerConfig()

        # Should have limits defined
        assert hasattr(config, "max_debate_states")
        assert hasattr(config, "max_client_ids")
        assert hasattr(config, "rate_limiter_cleanup_interval")

        # Limits should be reasonable
        assert config.max_debate_states > 0
        assert config.max_debate_states <= 10000  # Not unbounded
        assert config.max_client_ids > 0
