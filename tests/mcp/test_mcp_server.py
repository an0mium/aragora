"""
Tests for MCP Server Implementation.

Tests for:
- RateLimiter class
- AragoraMCPServer initialization
- Input validation
- Argument sanitization
- Tool listing
- Resource handling
- Error handling
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

from aragora.mcp.server import (
    RateLimiter,
    DEFAULT_RATE_LIMITS,
    MAX_QUESTION_LENGTH,
    MAX_CONTENT_LENGTH,
    MAX_QUERY_LENGTH,
    MCP_AVAILABLE,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init_with_default_limits(self):
        """Test RateLimiter initialization with default limits."""
        limiter = RateLimiter()
        assert limiter._limits == DEFAULT_RATE_LIMITS
        assert limiter._window_seconds == 60

    def test_init_with_custom_limits(self):
        """Test RateLimiter initialization with custom limits."""
        custom_limits = {"run_debate": 5, "list_agents": 100}
        limiter = RateLimiter(custom_limits)
        assert limiter._limits == custom_limits

    def test_check_allows_first_request(self):
        """Test that first request is always allowed."""
        limiter = RateLimiter({"test_tool": 10})
        allowed, error = limiter.check("test_tool")
        assert allowed is True
        assert error is None

    def test_check_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter({"test_tool": 5})

        # Make 4 requests (under limit of 5)
        for _ in range(4):
            allowed, error = limiter.check("test_tool")
            assert allowed is True
            assert error is None

    def test_check_denies_over_limit(self):
        """Test that requests over limit are denied."""
        limiter = RateLimiter({"test_tool": 3})

        # Make 3 requests (at limit)
        for _ in range(3):
            allowed, _ = limiter.check("test_tool")
            assert allowed is True

        # 4th request should be denied
        allowed, error = limiter.check("test_tool")
        assert allowed is False
        assert "Rate limit exceeded" in error
        assert "test_tool" in error

    def test_check_uses_default_limit_for_unknown_tool(self):
        """Test default limit of 60 for unknown tools."""
        limiter = RateLimiter({})

        # Unknown tool should use default limit of 60
        for _ in range(60):
            allowed, _ = limiter.check("unknown_tool")
            assert allowed is True

        # 61st should fail
        allowed, _ = limiter.check("unknown_tool")
        assert allowed is False

    def test_get_remaining_full_quota(self):
        """Test get_remaining with full quota available."""
        limiter = RateLimiter({"test_tool": 10})
        remaining = limiter.get_remaining("test_tool")
        assert remaining == 10

    def test_get_remaining_partial_quota(self):
        """Test get_remaining after some requests."""
        limiter = RateLimiter({"test_tool": 10})

        # Make 3 requests
        for _ in range(3):
            limiter.check("test_tool")

        remaining = limiter.get_remaining("test_tool")
        assert remaining == 7

    def test_get_remaining_exhausted(self):
        """Test get_remaining when quota exhausted."""
        limiter = RateLimiter({"test_tool": 3})

        for _ in range(3):
            limiter.check("test_tool")

        remaining = limiter.get_remaining("test_tool")
        assert remaining == 0

    def test_window_cleans_old_requests(self):
        """Test that old requests outside window are cleaned."""
        limiter = RateLimiter({"test_tool": 2})
        limiter._window_seconds = 0.1  # 100ms window for testing

        # Make 2 requests
        limiter.check("test_tool")
        limiter.check("test_tool")

        # Wait for window to expire
        time.sleep(0.15)

        # Should be allowed again
        allowed, _ = limiter.check("test_tool")
        assert allowed is True

    def test_independent_tool_limits(self):
        """Test that different tools have independent limits."""
        limiter = RateLimiter({"tool_a": 2, "tool_b": 2})

        # Exhaust tool_a
        limiter.check("tool_a")
        limiter.check("tool_a")

        # tool_b should still work
        allowed, _ = limiter.check("tool_b")
        assert allowed is True


class TestAragoraMCPServerInit:
    """Tests for AragoraMCPServer initialization."""

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP package not installed")
    def test_server_init_success(self):
        """Test successful server initialization."""
        from aragora.mcp.server import AragoraMCPServer

        with patch("aragora.mcp.server.Server") as mock_server:
            server = AragoraMCPServer()
            mock_server.assert_called_once_with("aragora")
            assert server._rate_limiter is not None
            assert server._debates_cache == {}
            assert server._agents_cache == {}

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP package not installed")
    def test_server_init_with_custom_rate_limits(self):
        """Test server initialization with custom rate limits."""
        from aragora.mcp.server import AragoraMCPServer

        custom_limits = {"run_debate": 100}

        with patch("aragora.mcp.server.Server"):
            server = AragoraMCPServer(rate_limits=custom_limits)
            assert server._rate_limiter._limits == custom_limits

    def test_server_init_without_mcp_package(self):
        """Test that ImportError is raised when MCP not installed."""
        with patch("aragora.mcp.server.MCP_AVAILABLE", False):
            from aragora.mcp.server import AragoraMCPServer

            with pytest.raises(ImportError) as exc_info:
                AragoraMCPServer()

            assert "MCP package not installed" in str(exc_info.value)


class TestInputValidation:
    """Tests for input validation."""

    @pytest.fixture
    def server(self):
        """Create server instance for testing."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP package not installed")

        from aragora.mcp.server import AragoraMCPServer

        with patch("aragora.mcp.server.Server"):
            return AragoraMCPServer()

    def test_validate_input_run_debate_valid(self, server):
        """Test valid run_debate arguments."""
        args = {"question": "Should we use async?", "rounds": 3}
        result = server._validate_input("run_debate", args)
        assert result is None

    def test_validate_input_run_debate_question_too_long(self, server):
        """Test run_debate with question exceeding max length."""
        args = {"question": "x" * (MAX_QUESTION_LENGTH + 1), "rounds": 3}
        result = server._validate_input("run_debate", args)
        assert result is not None
        assert "exceeds maximum length" in result

    def test_validate_input_run_debate_rounds_invalid_type(self, server):
        """Test run_debate with invalid rounds type."""
        args = {"question": "Valid?", "rounds": "three"}
        result = server._validate_input("run_debate", args)
        assert result is not None
        assert "Rounds must be an integer" in result

    def test_validate_input_run_debate_rounds_too_low(self, server):
        """Test run_debate with rounds below minimum."""
        args = {"question": "Valid?", "rounds": 0}
        result = server._validate_input("run_debate", args)
        assert result is not None
        assert "between 1 and 10" in result

    def test_validate_input_run_debate_rounds_too_high(self, server):
        """Test run_debate with rounds above maximum."""
        args = {"question": "Valid?", "rounds": 15}
        result = server._validate_input("run_debate", args)
        assert result is not None
        assert "between 1 and 10" in result

    def test_validate_input_run_gauntlet_valid(self, server):
        """Test valid run_gauntlet arguments."""
        args = {"content": "Test document content"}
        result = server._validate_input("run_gauntlet", args)
        assert result is None

    def test_validate_input_run_gauntlet_content_too_long(self, server):
        """Test run_gauntlet with content exceeding max length."""
        args = {"content": "x" * (MAX_CONTENT_LENGTH + 1)}
        result = server._validate_input("run_gauntlet", args)
        assert result is not None
        assert "exceeds maximum length" in result

    def test_validate_input_search_debates_valid(self, server):
        """Test valid search_debates arguments."""
        args = {"query": "consensus algorithm"}
        result = server._validate_input("search_debates", args)
        assert result is None

    def test_validate_input_search_debates_query_too_long(self, server):
        """Test search_debates with query exceeding max length."""
        args = {"query": "x" * (MAX_QUERY_LENGTH + 1)}
        result = server._validate_input("search_debates", args)
        assert result is not None
        assert "exceeds maximum length" in result

    def test_validate_input_unknown_tool(self, server):
        """Test validation passes for unknown tools."""
        args = {"any": "argument"}
        result = server._validate_input("unknown_tool", args)
        assert result is None


class TestArgumentSanitization:
    """Tests for argument sanitization."""

    @pytest.fixture
    def server(self):
        """Create server instance for testing."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP package not installed")

        from aragora.mcp.server import AragoraMCPServer

        with patch("aragora.mcp.server.Server"):
            return AragoraMCPServer()

    def test_sanitize_strips_whitespace(self, server):
        """Test that string arguments are stripped."""
        args = {"question": "  What should we do?  ", "context": "\n\tSome context\n"}
        result = server._sanitize_arguments(args)

        assert result["question"] == "What should we do?"
        assert result["context"] == "Some context"

    def test_sanitize_preserves_non_strings(self, server):
        """Test that non-string arguments are preserved."""
        args = {"rounds": 3, "enabled": True, "settings": {"key": "value"}}
        result = server._sanitize_arguments(args)

        assert result["rounds"] == 3
        assert result["enabled"] is True
        assert result["settings"] == {"key": "value"}

    def test_sanitize_returns_copy(self, server):
        """Test that sanitization returns a new dict."""
        args = {"question": "original"}
        result = server._sanitize_arguments(args)

        # Modifying result shouldn't affect original
        result["question"] = "modified"
        assert args["question"] == "original"


class TestDefaultRateLimits:
    """Tests for default rate limit configuration."""

    def test_run_debate_default_limit(self):
        """Test run_debate has appropriate default limit."""
        assert "run_debate" in DEFAULT_RATE_LIMITS
        assert DEFAULT_RATE_LIMITS["run_debate"] == 10

    def test_run_gauntlet_default_limit(self):
        """Test run_gauntlet has appropriate default limit."""
        assert "run_gauntlet" in DEFAULT_RATE_LIMITS
        assert DEFAULT_RATE_LIMITS["run_gauntlet"] == 20

    def test_list_agents_default_limit(self):
        """Test list_agents has higher limit (read-only)."""
        assert "list_agents" in DEFAULT_RATE_LIMITS
        assert DEFAULT_RATE_LIMITS["list_agents"] == 60

    def test_get_debate_default_limit(self):
        """Test get_debate has higher limit (read-only)."""
        assert "get_debate" in DEFAULT_RATE_LIMITS
        assert DEFAULT_RATE_LIMITS["get_debate"] == 60


class TestMaxInputSizes:
    """Tests for maximum input size constants."""

    def test_max_question_length(self):
        """Test maximum question length is reasonable."""
        assert MAX_QUESTION_LENGTH == 10000
        assert MAX_QUESTION_LENGTH > 0

    def test_max_content_length(self):
        """Test maximum content length is reasonable."""
        assert MAX_CONTENT_LENGTH == 100000
        assert MAX_CONTENT_LENGTH > MAX_QUESTION_LENGTH

    def test_max_query_length(self):
        """Test maximum query length is reasonable."""
        assert MAX_QUERY_LENGTH == 1000
        assert MAX_QUERY_LENGTH > 0


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP package not installed")
class TestMCPServerTools:
    """Tests for MCP server tool handling."""

    @pytest.fixture
    def server(self):
        """Create server instance for testing."""
        from aragora.mcp.server import AragoraMCPServer

        with patch("aragora.mcp.server.Server"):
            return AragoraMCPServer()

    @pytest.mark.asyncio
    async def test_tool_registration(self, server):
        """Test that tools are registered with the server."""
        # The server should have handlers set up
        assert server.server is not None

    def test_debates_cache_initialized(self, server):
        """Test debates cache is properly initialized."""
        assert isinstance(server._debates_cache, dict)
        assert len(server._debates_cache) == 0

    def test_agents_cache_initialized(self, server):
        """Test agents cache is properly initialized."""
        assert isinstance(server._agents_cache, dict)
        assert len(server._agents_cache) == 0


class TestMCPAvailability:
    """Tests for MCP package availability detection."""

    def test_mcp_available_is_boolean(self):
        """Test MCP_AVAILABLE is a boolean."""
        assert isinstance(MCP_AVAILABLE, bool)


# Export test classes
__all__ = [
    "TestRateLimiter",
    "TestAragoraMCPServerInit",
    "TestInputValidation",
    "TestArgumentSanitization",
    "TestDefaultRateLimits",
    "TestMaxInputSizes",
    "TestMCPServerTools",
    "TestMCPAvailability",
]
