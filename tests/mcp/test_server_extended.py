"""
Extended tests for MCP Server Implementation.

Covers paths NOT covered by test_mcp_server.py:
- Rate limiter edge cases (boundary, window expiry, concurrent tools)
- Server lifecycle (run, startup, shutdown)
- Error paths in call_tool handler (TypeError, generic Exception, unknown tool)
- call_tool happy path with rate limiting and caching
- list_tools handler
- list_resources / read_resource handlers
- list_resource_templates handler
- Edge cases in _validate_input (empty strings, boundary lengths)
- _sanitize_arguments with empty dict
- RedisRateLimiter error handling paths
- create_rate_limiter with unknown backend
- run_server and main entry points
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.server import (
    MAX_CONTENT_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_QUESTION_LENGTH,
    AragoraMCPServer,
    RateLimiter,
    RedisRateLimiter,
    create_rate_limiter,
)

pytest.importorskip("mcp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server(**kwargs) -> AragoraMCPServer:
    """Create an AragoraMCPServer with the MCP Server class mocked out."""
    with patch("aragora.mcp.server.Server") as mock_srv_cls:
        mock_srv_instance = MagicMock()
        # Make the decorator-style methods return identity so handlers register
        mock_srv_instance.list_tools.return_value = lambda fn: fn
        mock_srv_instance.call_tool.return_value = lambda fn: fn
        mock_srv_instance.list_resources.return_value = lambda fn: fn
        mock_srv_instance.list_resource_templates.return_value = lambda fn: fn
        mock_srv_instance.read_resource.return_value = lambda fn: fn
        mock_srv_cls.return_value = mock_srv_instance
        server = AragoraMCPServer(**kwargs)
    return server


# ---------------------------------------------------------------------------
# Rate limiter edge cases
# ---------------------------------------------------------------------------


class TestRateLimiterEdgeCases:
    """Edge cases for the in-memory RateLimiter."""

    def test_limit_of_one(self):
        """A tool with limit=1 allows exactly one request then blocks."""
        limiter = RateLimiter({"solo": 1})
        ok, _ = limiter.check("solo")
        assert ok is True
        ok, err = limiter.check("solo")
        assert ok is False
        assert "solo" in err

    def test_get_remaining_never_negative(self):
        """get_remaining must never return a negative value."""
        limiter = RateLimiter({"t": 2})
        limiter.check("t")
        limiter.check("t")
        # Manually inject an extra timestamp to simulate a race
        limiter._requests["t"].append(time.time())
        assert limiter.get_remaining("t") == 0  # clamped to 0

    def test_get_remaining_unknown_tool(self):
        """Unknown tool returns the default limit (60)."""
        limiter = RateLimiter({})
        assert limiter.get_remaining("no_such_tool") == 60

    def test_error_message_contains_wait_time(self):
        """Rate-limit error message includes a numeric wait hint."""
        limiter = RateLimiter({"x": 1})
        limiter.check("x")
        _, err = limiter.check("x")
        # The message should contain "Try again in Xs"
        assert "Try again in" in err

    def test_window_expiry_restores_remaining(self):
        """After the window elapses, remaining count is fully restored."""
        limiter = RateLimiter({"t": 3})
        limiter._window_seconds = 0.05
        for _ in range(3):
            limiter.check("t")
        assert limiter.get_remaining("t") == 0
        time.sleep(0.08)
        assert limiter.get_remaining("t") == 3


# ---------------------------------------------------------------------------
# RedisRateLimiter edge-case paths
# ---------------------------------------------------------------------------


class TestRedisRateLimiterEdgeCases:
    """Edge-case and error paths in RedisRateLimiter."""

    def test_get_redis_import_error(self):
        """When the redis package is missing, _get_redis returns None gracefully."""
        limiter = RedisRateLimiter()
        with patch.dict("sys.modules", {"redis": None}):
            with patch("builtins.__import__", side_effect=ImportError("no redis")):
                result = limiter._get_redis()
        # Should have set _connected = False
        assert limiter._connected is False

    def test_check_redis_pipeline_exception(self):
        """If the pipeline raises, check() fails open (allows request)."""
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute.side_effect = RuntimeError("connection lost")
        mock_redis.pipeline.return_value = mock_pipe

        limiter = RedisRateLimiter(limits={"t": 5})
        limiter._redis = mock_redis
        limiter._connected = True

        ok, err = limiter.check("t")
        assert ok is True
        assert err is None

    def test_get_remaining_redis_exception(self):
        """If Redis throws, get_remaining returns the full limit."""
        mock_redis = MagicMock()
        mock_redis.zremrangebyscore.side_effect = RuntimeError("boom")

        limiter = RedisRateLimiter(limits={"t": 15})
        limiter._redis = mock_redis
        limiter._connected = True

        assert limiter.get_remaining("t") == 15

    def test_reset_no_op_when_disconnected(self):
        """reset() silently does nothing when Redis is not connected."""
        limiter = RedisRateLimiter()
        limiter._connected = False
        limiter._redis = None
        # Should not raise
        limiter.reset("any_tool")
        limiter.reset()

    def test_reset_exception_is_swallowed(self):
        """reset() logs and swallows Redis errors."""
        mock_redis = MagicMock()
        mock_redis.delete.side_effect = RuntimeError("oops")
        limiter = RedisRateLimiter()
        limiter._redis = mock_redis
        limiter._connected = True
        # Should not raise
        limiter.reset("some_tool")

    def test_check_denied_without_oldest_entry(self):
        """When over limit but zrange returns empty, a generic message is used."""
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [None, 10]  # over limit of 5
        mock_redis.pipeline.return_value = mock_pipe
        mock_redis.zrange.return_value = []  # empty

        limiter = RedisRateLimiter(limits={"t": 5})
        limiter._redis = mock_redis
        limiter._connected = True

        ok, err = limiter.check("t")
        assert ok is False
        assert "Rate limit exceeded for t." == err


# ---------------------------------------------------------------------------
# create_rate_limiter factory
# ---------------------------------------------------------------------------


class TestCreateRateLimiterExtended:
    """Additional factory tests."""

    def test_unknown_backend_falls_through_to_memory(self):
        """An unrecognised backend string creates an in-memory limiter."""
        limiter = create_rate_limiter(backend="banana")
        assert isinstance(limiter, RateLimiter)


# ---------------------------------------------------------------------------
# Input validation edge cases
# ---------------------------------------------------------------------------


class TestValidateInputEdgeCases:
    """Edge cases for _validate_input."""

    @pytest.fixture
    def server(self):
        return _make_server()

    def test_run_debate_empty_question_passes(self, server):
        """Empty question string does not trigger the length check."""
        result = server._validate_input("run_debate", {"question": "", "rounds": 3})
        assert result is None

    def test_run_debate_exact_max_question_passes(self, server):
        """Question exactly at MAX_QUESTION_LENGTH is accepted."""
        result = server._validate_input(
            "run_debate", {"question": "x" * MAX_QUESTION_LENGTH, "rounds": 1}
        )
        assert result is None

    def test_run_gauntlet_empty_content_passes(self, server):
        """Empty content passes validation (content is checked at handler level)."""
        result = server._validate_input("run_gauntlet", {"content": ""})
        assert result is None

    def test_search_debates_exact_max_query_passes(self, server):
        """Query exactly at MAX_QUERY_LENGTH is accepted."""
        result = server._validate_input("search_debates", {"query": "q" * MAX_QUERY_LENGTH})
        assert result is None

    def test_run_debate_rounds_negative(self, server):
        """Negative rounds value is rejected."""
        result = server._validate_input("run_debate", {"question": "q", "rounds": -1})
        assert result is not None
        assert "between 1 and 10" in result

    def test_run_debate_rounds_float_rejected(self, server):
        """Float rounds value is rejected (not isinstance int)."""
        result = server._validate_input("run_debate", {"question": "q", "rounds": 3.5})
        assert result is not None


# ---------------------------------------------------------------------------
# Sanitization edge cases
# ---------------------------------------------------------------------------


class TestSanitizeEdgeCases:
    @pytest.fixture
    def server(self):
        return _make_server()

    def test_empty_arguments(self, server):
        """Sanitizing an empty dict returns an empty dict."""
        assert server._sanitize_arguments({}) == {}

    def test_nested_dict_not_stripped(self, server):
        """Nested dicts are passed through without deep-stripping."""
        result = server._sanitize_arguments({"nested": {"key": "  value  "}})
        assert result["nested"]["key"] == "  value  "

    def test_list_values_preserved(self, server):
        """List values are passed through as-is."""
        result = server._sanitize_arguments({"items": [1, 2, 3]})
        assert result["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# call_tool handler paths
# ---------------------------------------------------------------------------


class TestCallToolHandler:
    """Tests for the call_tool handler inside _setup_handlers."""

    @pytest.fixture
    def server(self):
        return _make_server()

    def _get_call_tool_fn(self, server):
        """Extract the call_tool handler registered via the decorator."""
        # The mock Server's call_tool() returns identity, so the inner function
        # is the last value assigned.  We stored the handlers during _setup_handlers
        # via the decorator pattern; the mock returns the fn itself.
        # We need to invoke _setup_handlers again with a capturing mock.
        captured = {}
        with patch("aragora.mcp.server.Server") as mock_cls:
            mock_inst = MagicMock()

            def capture_decorator(key):
                def decorator(fn):
                    captured[key] = fn
                    return fn

                return decorator

            mock_inst.list_tools.side_effect = lambda: capture_decorator("list_tools")
            mock_inst.call_tool.side_effect = lambda: capture_decorator("call_tool")
            mock_inst.list_resources.side_effect = lambda: capture_decorator("list_resources")
            mock_inst.list_resource_templates.side_effect = lambda: capture_decorator(
                "list_resource_templates"
            )
            mock_inst.read_resource.side_effect = lambda: capture_decorator("read_resource")
            mock_cls.return_value = mock_inst
            srv = AragoraMCPServer()
        return srv, captured

    @pytest.mark.asyncio
    async def test_call_tool_rate_limited(self):
        """When rate-limited, returns rate_limited JSON error."""
        srv, handlers = self._get_call_tool_fn(None)
        # Force rate limiter to deny
        srv._rate_limiter = MagicMock()
        srv._rate_limiter.check.return_value = (
            False,
            "Rate limit exceeded for x. Try again in 30s",
        )

        result = await handlers["call_tool"]("x", {"foo": "bar"})
        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert payload["rate_limited"] is True
        assert "Rate limit exceeded" in payload["error"]

    @pytest.mark.asyncio
    async def test_call_tool_validation_failure(self):
        """When input validation fails, returns validation error."""
        srv, handlers = self._get_call_tool_fn(None)
        srv._rate_limiter = MagicMock()
        srv._rate_limiter.check.return_value = (True, None)

        # Trigger validation: question too long
        args = {"question": "x" * (MAX_QUESTION_LENGTH + 1), "rounds": 3}
        result = await handlers["call_tool"]("run_debate", args)
        payload = json.loads(result[0].text)
        assert "exceeds maximum length" in payload["error"]

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self):
        """Unknown tool name returns an error."""
        srv, handlers = self._get_call_tool_fn(None)
        srv._rate_limiter = MagicMock()
        srv._rate_limiter.check.return_value = (True, None)

        fake_meta = [{"name": "known", "function": AsyncMock(), "parameters": {}}]
        with patch("aragora.mcp.tools.TOOLS_METADATA", fake_meta):
            result = await handlers["call_tool"]("totally_unknown", {})
        payload = json.loads(result[0].text)
        assert "Unknown tool" in payload["error"]

    @pytest.mark.asyncio
    async def test_call_tool_type_error(self):
        """TypeError from tool function is caught and reported."""
        srv, handlers = self._get_call_tool_fn(None)
        srv._rate_limiter = MagicMock()
        srv._rate_limiter.check.return_value = (True, None)

        async def bad_func(**kwargs):
            raise TypeError("missing required argument: 'question'")

        fake_meta = [{"name": "bad", "function": bad_func, "parameters": {}}]
        with patch("aragora.mcp.tools.TOOLS_METADATA", fake_meta):
            result = await handlers["call_tool"]("bad", {})
        payload = json.loads(result[0].text)
        assert "Invalid arguments" in payload["error"]

    @pytest.mark.asyncio
    async def test_call_tool_generic_exception(self):
        """Generic exception from tool function is caught and reported."""
        srv, handlers = self._get_call_tool_fn(None)
        srv._rate_limiter = MagicMock()
        srv._rate_limiter.check.return_value = (True, None)

        async def exploding_func(**kwargs):
            raise RuntimeError("kaboom")

        fake_meta = [{"name": "boom", "function": exploding_func, "parameters": {}}]
        with patch("aragora.mcp.tools.TOOLS_METADATA", fake_meta):
            result = await handlers["call_tool"]("boom", {})
        payload = json.loads(result[0].text)
        assert payload["error"] == "kaboom"

    @pytest.mark.asyncio
    async def test_call_tool_caches_debate_result(self):
        """Successful run_debate caches the result by debate_id."""
        srv, handlers = self._get_call_tool_fn(None)
        srv._rate_limiter = MagicMock()
        srv._rate_limiter.check.return_value = (True, None)

        async def mock_debate(**kwargs):
            return {"debate_id": "abc123", "task": "test", "answer": "yes"}

        fake_meta = [{"name": "run_debate", "function": mock_debate, "parameters": {}}]
        with patch("aragora.mcp.tools.TOOLS_METADATA", fake_meta):
            result = await handlers["call_tool"]("run_debate", {"question": "test"})
        payload = json.loads(result[0].text)
        assert payload["debate_id"] == "abc123"
        assert "abc123" in srv._debates_cache

    @pytest.mark.asyncio
    async def test_call_tool_does_not_cache_error_debate(self):
        """run_debate results with 'error' key are not cached."""
        srv, handlers = self._get_call_tool_fn(None)
        srv._rate_limiter = MagicMock()
        srv._rate_limiter.check.return_value = (True, None)

        async def failing_debate(**kwargs):
            return {"debate_id": "fail1", "error": "no agents"}

        fake_meta = [{"name": "run_debate", "function": failing_debate, "parameters": {}}]
        with patch("aragora.mcp.tools.TOOLS_METADATA", fake_meta):
            await handlers["call_tool"]("run_debate", {"question": "test"})
        assert "fail1" not in srv._debates_cache


# ---------------------------------------------------------------------------
# list_tools handler
# ---------------------------------------------------------------------------


class TestListToolsHandler:
    def _get_handlers(self):
        captured = {}
        with patch("aragora.mcp.server.Server") as mock_cls:
            mock_inst = MagicMock()

            def capture(key):
                def dec(fn):
                    captured[key] = fn
                    return fn

                return dec

            mock_inst.list_tools.side_effect = lambda: capture("list_tools")
            mock_inst.call_tool.side_effect = lambda: capture("call_tool")
            mock_inst.list_resources.side_effect = lambda: capture("list_resources")
            mock_inst.list_resource_templates.side_effect = lambda: capture(
                "list_resource_templates"
            )
            mock_inst.read_resource.side_effect = lambda: capture("read_resource")
            mock_cls.return_value = mock_inst
            srv = AragoraMCPServer()
        return srv, captured

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools(self):
        """list_tools handler returns Tool objects from TOOLS_METADATA."""
        srv, handlers = self._get_handlers()
        fake_meta = [
            {
                "name": "my_tool",
                "description": "Does stuff",
                "function": AsyncMock(),
                "parameters": {
                    "arg1": {"type": "string", "required": True, "description": "an arg"},
                    "arg2": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                    "arg3": {"type": "string", "enum": ["a", "b"]},
                },
            }
        ]
        with patch("aragora.mcp.tools.TOOLS_METADATA", fake_meta):
            tools = await handlers["list_tools"]()

        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "my_tool"
        assert tool.description == "Does stuff"
        schema = tool.inputSchema
        assert "arg1" in schema["properties"]
        assert schema["required"] == ["arg1"]
        assert schema["properties"]["arg2"]["default"] == 5
        assert schema["properties"]["arg3"]["enum"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Resource handlers
# ---------------------------------------------------------------------------


class TestResourceHandlers:
    def _get_handlers(self):
        captured = {}
        with patch("aragora.mcp.server.Server") as mock_cls:
            mock_inst = MagicMock()

            def capture(key):
                def dec(fn):
                    captured[key] = fn
                    return fn

                return dec

            mock_inst.list_tools.side_effect = lambda: capture("list_tools")
            mock_inst.call_tool.side_effect = lambda: capture("call_tool")
            mock_inst.list_resources.side_effect = lambda: capture("list_resources")
            mock_inst.list_resource_templates.side_effect = lambda: capture(
                "list_resource_templates"
            )
            mock_inst.read_resource.side_effect = lambda: capture("read_resource")
            mock_cls.return_value = mock_inst
            srv = AragoraMCPServer()
        return srv, captured

    @pytest.mark.asyncio
    async def test_list_resources_empty(self):
        """list_resources returns empty list when no debates cached."""
        srv, handlers = self._get_handlers()
        resources = await handlers["list_resources"]()
        assert resources == []

    @pytest.mark.asyncio
    async def test_list_resources_with_cached_debate(self):
        """list_resources returns cached debate resources."""
        srv, handlers = self._get_handlers()
        srv._debates_cache["d1"] = {
            "task": "Should we refactor?",
            "timestamp": "2025-01-01 12:00:00",
        }
        resources = await handlers["list_resources"]()
        assert len(resources) == 1
        assert "debate://d1" == str(resources[0].uri)

    @pytest.mark.asyncio
    async def test_list_resource_templates(self):
        """list_resource_templates returns the four known templates."""
        srv, handlers = self._get_handlers()
        templates = await handlers["list_resource_templates"]()
        assert len(templates) == 4
        uris = {t.uriTemplate for t in templates}
        assert "debate://{debate_id}" in uris
        assert "agent://{agent_name}/stats" in uris

    @pytest.mark.asyncio
    async def test_read_resource_unknown_uri(self):
        """read_resource with unknown URI returns error JSON."""
        srv, handlers = self._get_handlers()
        result = await handlers["read_resource"]("ftp://something")
        data = json.loads(result)
        assert "Unknown resource" in data["error"]

    @pytest.mark.asyncio
    async def test_read_resource_cached_debate(self):
        """read_resource for a cached debate returns its data."""
        srv, handlers = self._get_handlers()
        srv._debates_cache["d2"] = {"task": "test", "answer": "42"}
        result = await handlers["read_resource"]("debate://d2")
        data = json.loads(result)
        assert data["task"] == "test"


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    """Tests for run(), run_server(), and main()."""

    @pytest.mark.asyncio
    async def test_run_invokes_stdio_server(self):
        """AragoraMCPServer.run() calls stdio_server and server.run()."""
        srv = _make_server()
        mock_read = AsyncMock()
        mock_write = AsyncMock()

        # Mock stdio_server as async context manager
        mock_stdio = MagicMock()
        mock_stdio.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.__aexit__ = AsyncMock(return_value=False)

        srv.server.run = AsyncMock()
        srv.server.create_initialization_options = MagicMock(return_value={})

        with patch("aragora.mcp.server.stdio_server", return_value=mock_stdio):
            await srv.run()

        srv.server.run.assert_awaited_once_with(mock_read, mock_write, {})

    def test_main_keyboard_interrupt(self):
        """main() handles KeyboardInterrupt gracefully."""
        from aragora.mcp.server import main

        with patch("aragora.mcp.server.asyncio.run", side_effect=KeyboardInterrupt):
            # Should not raise
            main()

    def test_main_generic_exception_exits(self):
        """main() exits with code 1 on unexpected exceptions."""
        from aragora.mcp.server import main

        with patch("aragora.mcp.server.asyncio.run", side_effect=RuntimeError("boom")):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
