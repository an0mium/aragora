"""Tests for the standalone OpenClaw governance gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.compat.openclaw.standalone import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    StandaloneGatewayServer,
    _MinimalHandlerContext,
    cmd_openclaw_serve,
)


class TestStandaloneGatewayServer:
    """Tests for StandaloneGatewayServer configuration."""

    def test_default_config(self):
        """Server uses sensible defaults."""
        server = StandaloneGatewayServer()
        assert server.host == DEFAULT_HOST
        assert server.port == DEFAULT_PORT
        assert server.default_policy == "deny"
        assert server.policy_file is None
        assert server.cors_origins == ["http://localhost:3000"]

    def test_custom_config(self):
        """Server accepts custom configuration."""
        server = StandaloneGatewayServer(
            host="127.0.0.1",
            port=9000,
            policy_file="/tmp/policy.yaml",
            default_policy="allow",
            cors_origins=["https://example.com"],
        )
        assert server.host == "127.0.0.1"
        assert server.port == 9000
        assert server.policy_file == "/tmp/policy.yaml"
        assert server.default_policy == "allow"
        assert server.cors_origins == ["https://example.com"]

    def test_init_handler_creates_handler(self):
        """_init_handler creates an OpenClawGatewayHandler."""
        server = StandaloneGatewayServer()
        server._init_handler()
        assert server._handler is not None

    def test_get_version_returns_string(self):
        """_get_version returns a version string."""
        server = StandaloneGatewayServer()
        version = server._get_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_build_routes_returns_all_endpoints(self):
        """_build_routes returns the complete route list."""
        server = StandaloneGatewayServer()
        routes = server._build_routes()
        assert len(routes) >= 20

        # Check key routes exist
        methods_paths = [(m, p) for m, p, _ in routes]
        assert ("POST", "/api/gateway/openclaw/sessions") in methods_paths
        assert ("GET", "/api/gateway/openclaw/health") in methods_paths
        assert ("GET", "/api/gateway/openclaw/metrics") in methods_paths
        assert ("GET", "/api/gateway/openclaw/audit") in methods_paths


class TestMinimalHandlerContext:
    """Tests for the minimal handler context."""

    def test_basic_context(self):
        """Context extracts user info from headers."""
        ctx = _MinimalHandlerContext(
            headers={"x-user-id": "user-123", "x-tenant-id": "tenant-456"},
            body={"action": "test"},
            query_params={"limit": "10"},
        )
        assert ctx.user_id == "user-123"
        assert ctx.tenant_id == "tenant-456"
        assert ctx.org_id == "tenant-456"
        assert ctx.body == {"action": "test"}
        assert ctx.query_params == {"limit": "10"}

    def test_anonymous_context(self):
        """Context defaults to anonymous when no auth headers."""
        ctx = _MinimalHandlerContext(
            headers={},
            body=None,
            query_params={},
        )
        assert ctx.user_id == "anonymous"
        assert ctx.tenant_id is None
        assert ctx.body is None


class TestSendResponse:
    """Tests for HTTP response generation."""

    @pytest.mark.asyncio
    async def test_send_json_response(self):
        """Server sends proper JSON responses."""
        server = StandaloneGatewayServer()
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()

        await server._send_response(writer, 200, {"status": "ok"})

        # Check that write was called
        assert writer.write.call_count >= 1
        # Verify response contains status line
        first_write = writer.write.call_args_list[0][0][0]
        assert b"200 OK" in first_write
        assert b"application/json" in first_write

    @pytest.mark.asyncio
    async def test_send_error_response(self):
        """Server sends error responses with correct status."""
        server = StandaloneGatewayServer()
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()

        await server._send_response(writer, 404, {"error": "Not found"})

        first_write = writer.write.call_args_list[0][0][0]
        assert b"404 Not Found" in first_write

    @pytest.mark.asyncio
    async def test_send_cors_response(self):
        """Server sends CORS preflight response."""
        server = StandaloneGatewayServer(cors_origins=["https://example.com"])
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()

        await server._send_cors_response(writer)

        first_write = writer.write.call_args_list[0][0][0]
        assert b"204 No Content" in first_write
        assert b"https://example.com" in first_write
        assert b"Access-Control-Allow-Methods" in first_write


class TestCmdOpenclawServe:
    """Tests for the CLI serve command."""

    def test_serve_creates_server(self):
        """cmd_openclaw_serve creates a server with correct config."""
        args = argparse.Namespace(
            host="127.0.0.1",
            port=9999,
            policy=None,
            default_policy="deny",
            cors="https://example.com",
            log_level="WARNING",
        )

        with patch("aragora.compat.openclaw.standalone.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=KeyboardInterrupt)

            # Should not raise
            cmd_openclaw_serve(args)

            # asyncio.run was called
            assert mock_asyncio.run.called


class TestRequestHandling:
    """Tests for request dispatch."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Health endpoint returns without initializing handler."""
        server = StandaloneGatewayServer()

        reader = AsyncMock()
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        # Simulate GET /health request
        reader.readline = AsyncMock(side_effect=[
            b"GET /health HTTP/1.1\r\n",
            b"Host: localhost\r\n",
            b"\r\n",
        ])

        await server._handle_request(reader, writer)

        # Should have written a response
        assert writer.write.called
        response = writer.write.call_args_list[0][0][0]
        assert b"200 OK" in response

    @pytest.mark.asyncio
    async def test_not_found_for_unknown_path(self):
        """Unknown paths return 404."""
        server = StandaloneGatewayServer()

        reader = AsyncMock()
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        reader.readline = AsyncMock(side_effect=[
            b"GET /api/unknown HTTP/1.1\r\n",
            b"\r\n",
        ])

        await server._handle_request(reader, writer)

        response = writer.write.call_args_list[0][0][0]
        assert b"404 Not Found" in response

    @pytest.mark.asyncio
    async def test_options_returns_cors(self):
        """OPTIONS requests return CORS headers."""
        server = StandaloneGatewayServer()

        reader = AsyncMock()
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        reader.readline = AsyncMock(side_effect=[
            b"OPTIONS /api/gateway/openclaw/sessions HTTP/1.1\r\n",
            b"\r\n",
        ])

        await server._handle_request(reader, writer)

        response = writer.write.call_args_list[0][0][0]
        assert b"204 No Content" in response
        assert b"Access-Control-Allow-Methods" in response
