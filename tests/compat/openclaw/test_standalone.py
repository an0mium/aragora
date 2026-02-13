"""Tests for the standalone OpenClaw governance gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.compat.openclaw.standalone import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    StandaloneGatewayServer,
    _MinimalHandlerContext,
    cmd_openclaw_serve,
)


# ---------------------------------------------------------------------------
# Helpers for simulating raw HTTP requests
# ---------------------------------------------------------------------------

def _make_reader_writer(method: str, path: str, headers: dict[str, str] | None = None):
    """Build mock reader/writer for a raw HTTP request."""
    header_lines: list[bytes] = []
    for key, value in (headers or {}).items():
        header_lines.append(f"{key}: {value}\r\n".encode())
    header_lines.append(b"\r\n")

    reader = AsyncMock()
    reader.readline = AsyncMock(
        side_effect=[
            f"{method} {path} HTTP/1.1\r\n".encode(),
            *header_lines,
        ]
    )

    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    return reader, writer


def _response_status(writer: MagicMock) -> int:
    """Extract numeric HTTP status code from the first write call."""
    raw = writer.write.call_args_list[0][0][0]
    # e.g. b"HTTP/1.1 401 Unauthorized\r\n..."
    status_line = raw.split(b"\r\n")[0]
    return int(status_line.split(b" ")[1])


def _response_body(writer: MagicMock) -> dict:
    """Extract JSON body from response writes."""
    # Body is the second write call (after headers)
    if len(writer.write.call_args_list) >= 2:
        return json.loads(writer.write.call_args_list[1][0][0])
    # If body is empty or inlined, check first write
    raw = b"".join(call[0][0] for call in writer.write.call_args_list)
    # Split on double CRLF to get body
    parts = raw.split(b"\r\n\r\n", 1)
    if len(parts) > 1 and parts[1]:
        return json.loads(parts[1])
    return {}


def _clean_auth_env() -> dict[str, str]:
    """Return a copy of os.environ with auth-related env vars removed."""
    return {
        k: v for k, v in os.environ.items()
        if k not in ("OPENCLAW_API_KEY", "ARAGORA_API_TOKEN", "OPENCLAW_ALLOWED_ORIGINS")
    }


class TestStandaloneGatewayServer:
    """Tests for StandaloneGatewayServer configuration."""

    def test_default_config(self):
        """Server uses sensible defaults."""
        with patch.dict(os.environ, _clean_auth_env(), clear=True):
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
        with patch.dict(os.environ, _clean_auth_env(), clear=True):
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


# ---------------------------------------------------------------------------
# F01: API key authentication middleware
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    """Tests for API key authentication on the standalone server."""

    SECRET = "test-secret-key-12345"

    def _make_authed_server(self, key: str | None = None) -> StandaloneGatewayServer:
        return StandaloneGatewayServer(api_key=key or self.SECRET)

    # -- Unauthenticated access when key is configured -----------------------

    @pytest.mark.asyncio
    async def test_reject_missing_credentials(self):
        """Requests without any auth header get 401 when key is set."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/api/gateway/openclaw/sessions",
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 401
        assert _response_body(writer)["error"] == "Unauthorized"

    @pytest.mark.asyncio
    async def test_reject_wrong_bearer_token(self):
        """Wrong Bearer token gets 401."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/api/gateway/openclaw/sessions",
            headers={"Authorization": "Bearer wrong-key"},
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 401

    @pytest.mark.asyncio
    async def test_reject_wrong_x_api_key(self):
        """Wrong X-API-Key header gets 401."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/api/gateway/openclaw/sessions",
            headers={"X-API-Key": "wrong-key"},
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 401

    @pytest.mark.asyncio
    async def test_reject_empty_bearer(self):
        """Empty Bearer value still gets 401."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/api/gateway/openclaw/sessions",
            headers={"Authorization": "Bearer "},
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 401

    # -- Authenticated access ------------------------------------------------

    @pytest.mark.asyncio
    async def test_accept_valid_bearer_token(self):
        """Valid Bearer token passes auth (reaches handler/404)."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/api/unknown",
            headers={"Authorization": f"Bearer {self.SECRET}"},
        )

        await server._handle_request(reader, writer)

        # Not 401 — the request passed auth and hit the 404 for unknown path
        assert _response_status(writer) == 404

    @pytest.mark.asyncio
    async def test_accept_valid_x_api_key(self):
        """Valid X-API-Key header passes auth."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/api/unknown",
            headers={"X-API-Key": self.SECRET},
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 404

    @pytest.mark.asyncio
    async def test_bearer_takes_precedence_over_x_api_key(self):
        """When both headers are present, Bearer is checked first."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/api/unknown",
            headers={
                "Authorization": f"Bearer {self.SECRET}",
                "X-API-Key": "wrong-key",
            },
        )

        await server._handle_request(reader, writer)

        # Bearer was correct, so auth passes even though X-API-Key is wrong
        assert _response_status(writer) == 404

    # -- Health check is exempt from auth ------------------------------------

    @pytest.mark.asyncio
    async def test_health_shortcut_exempt_from_auth(self):
        """/health bypasses auth (before API route check)."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/health",
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 200

    @pytest.mark.asyncio
    async def test_root_exempt_from_auth(self):
        """/ bypasses auth (before API route check)."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "GET", "/",
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 200

    @pytest.mark.asyncio
    async def test_api_health_endpoint_exempt_from_auth(self):
        """/api/gateway/openclaw/health is exempt from auth."""
        server = self._make_authed_server()
        # Initialize handler so the request can reach the handler
        server._handler = MagicMock()
        server._handler.handle = MagicMock(return_value=(200, {"status": "healthy"}))

        reader, writer = _make_reader_writer(
            "GET", "/api/gateway/openclaw/health",
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 200

    # -- No auth configured = open access ------------------------------------

    @pytest.mark.asyncio
    async def test_no_key_configured_allows_all_requests(self):
        """When no API key is set, all requests pass through."""
        with patch.dict(os.environ, _clean_auth_env(), clear=True):
            server = StandaloneGatewayServer()  # no api_key
        assert server._api_key is None

        reader, writer = _make_reader_writer(
            "GET", "/api/unknown",
        )

        await server._handle_request(reader, writer)

        # Should reach 404 (no auth block), not 401
        assert _response_status(writer) == 404

    # -- CORS preflight is exempt from auth ----------------------------------

    @pytest.mark.asyncio
    async def test_options_preflight_exempt_from_auth(self):
        """OPTIONS requests bypass auth for CORS preflight."""
        server = self._make_authed_server()
        reader, writer = _make_reader_writer(
            "OPTIONS", "/api/gateway/openclaw/sessions",
        )

        await server._handle_request(reader, writer)

        assert _response_status(writer) == 204

    # -- Startup warning when no key is configured ---------------------------

    def test_warning_logged_when_no_api_key(self):
        """Server logs a warning when started without an API key."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENCLAW_API_KEY", "ARAGORA_API_TOKEN")
        }
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("aragora.compat.openclaw.standalone.logger") as mock_logger:
                StandaloneGatewayServer()
                mock_logger.warning.assert_called_once()
                msg = mock_logger.warning.call_args[0][0]
                assert "UNPROTECTED" in msg

    def test_no_warning_when_api_key_set(self):
        """No warning when an API key is provided."""
        with patch("aragora.compat.openclaw.standalone.logger") as mock_logger:
            StandaloneGatewayServer(api_key="secret")
            mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# F01: Environment variable resolution for API key
# ---------------------------------------------------------------------------


class TestApiKeyEnvVars:
    """Tests for API key resolution from environment variables."""

    def test_openclaw_api_key_env_var(self):
        """OPENCLAW_API_KEY env var is read."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENCLAW_API_KEY", "ARAGORA_API_TOKEN")
        }
        clean_env["OPENCLAW_API_KEY"] = "env-key-1"
        with patch.dict(os.environ, clean_env, clear=True):
            server = StandaloneGatewayServer()
            assert server._api_key == "env-key-1"

    def test_aragora_api_token_env_var(self):
        """ARAGORA_API_TOKEN env var is used as fallback."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENCLAW_API_KEY", "ARAGORA_API_TOKEN")
        }
        clean_env["ARAGORA_API_TOKEN"] = "env-key-2"
        with patch.dict(os.environ, clean_env, clear=True):
            server = StandaloneGatewayServer()
            assert server._api_key == "env-key-2"

    def test_explicit_api_key_overrides_env(self):
        """Explicit api_key parameter takes precedence over env vars."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENCLAW_API_KEY", "ARAGORA_API_TOKEN")
        }
        clean_env["OPENCLAW_API_KEY"] = "env-key"
        with patch.dict(os.environ, clean_env, clear=True):
            server = StandaloneGatewayServer(api_key="explicit-key")
            assert server._api_key == "explicit-key"

    def test_openclaw_key_takes_precedence_over_aragora_token(self):
        """OPENCLAW_API_KEY takes precedence over ARAGORA_API_TOKEN."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("OPENCLAW_API_KEY", "ARAGORA_API_TOKEN")
        }
        clean_env["OPENCLAW_API_KEY"] = "openclaw-key"
        clean_env["ARAGORA_API_TOKEN"] = "aragora-key"
        with patch.dict(os.environ, clean_env, clear=True):
            server = StandaloneGatewayServer()
            assert server._api_key == "openclaw-key"


# ---------------------------------------------------------------------------
# F05: CORS configuration
# ---------------------------------------------------------------------------


class TestCorsConfiguration:
    """Tests for CORS origin configuration."""

    def test_default_cors_is_localhost(self):
        """Default CORS origin is localhost:3000, not wildcard."""
        server = StandaloneGatewayServer()
        assert server.cors_origins == ["http://localhost:3000"]
        assert "*" not in server.cors_origins

    def test_explicit_cors_origins(self):
        """Explicit cors_origins parameter is used directly."""
        server = StandaloneGatewayServer(cors_origins=["https://app.example.com"])
        assert server.cors_origins == ["https://app.example.com"]

    def test_openclaw_allowed_origins_env_var(self):
        """OPENCLAW_ALLOWED_ORIGINS env var is read when no parameter given."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k != "OPENCLAW_ALLOWED_ORIGINS"
        }
        clean_env["OPENCLAW_ALLOWED_ORIGINS"] = "https://a.com, https://b.com"
        with patch.dict(os.environ, clean_env, clear=True):
            server = StandaloneGatewayServer()
            assert server.cors_origins == ["https://a.com", "https://b.com"]

    def test_explicit_cors_overrides_env_var(self):
        """Explicit cors_origins parameter overrides env var."""
        with patch.dict(os.environ, {
            "OPENCLAW_ALLOWED_ORIGINS": "https://env.example.com",
        }, clear=False):
            server = StandaloneGatewayServer(cors_origins=["https://explicit.example.com"])
            assert server.cors_origins == ["https://explicit.example.com"]

    @pytest.mark.asyncio
    async def test_cors_preflight_includes_x_api_key(self):
        """CORS preflight response allows X-API-Key header."""
        server = StandaloneGatewayServer()
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()

        await server._send_cors_response(writer)

        response_bytes = writer.write.call_args_list[0][0][0]
        assert b"X-API-Key" in response_bytes

    def test_empty_env_var_falls_back_to_default(self):
        """Empty OPENCLAW_ALLOWED_ORIGINS falls back to localhost default."""
        clean_env = {
            k: v for k, v in os.environ.items()
            if k != "OPENCLAW_ALLOWED_ORIGINS"
        }
        clean_env["OPENCLAW_ALLOWED_ORIGINS"] = ""
        with patch.dict(os.environ, clean_env, clear=True):
            server = StandaloneGatewayServer()
            assert server.cors_origins == ["http://localhost:3000"]


# ---------------------------------------------------------------------------
# CLI --api-key flag
# ---------------------------------------------------------------------------


class TestCliApiKeyFlag:
    """Tests for the --api-key CLI argument."""

    def test_cmd_passes_api_key_to_server(self):
        """cmd_openclaw_serve passes api_key from args to the server."""
        args = argparse.Namespace(
            host="127.0.0.1",
            port=9999,
            policy=None,
            default_policy="deny",
            cors="http://localhost:3000",
            api_key="cli-secret",
            log_level="WARNING",
        )

        with patch("aragora.compat.openclaw.standalone.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=KeyboardInterrupt)

            with patch(
                "aragora.compat.openclaw.standalone.StandaloneGatewayServer"
            ) as MockServer:
                MockServer.return_value.start = AsyncMock()
                cmd_openclaw_serve(args)

                MockServer.assert_called_once()
                call_kwargs = MockServer.call_args[1]
                assert call_kwargs["api_key"] == "cli-secret"

    def test_cmd_without_api_key(self):
        """cmd_openclaw_serve works when api_key is absent from args."""
        args = argparse.Namespace(
            host="127.0.0.1",
            port=9999,
            policy=None,
            default_policy="deny",
            cors="http://localhost:3000",
            log_level="WARNING",
            # no api_key attribute
        )

        with patch("aragora.compat.openclaw.standalone.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=KeyboardInterrupt)

            with patch(
                "aragora.compat.openclaw.standalone.StandaloneGatewayServer"
            ) as MockServer:
                MockServer.return_value.start = AsyncMock()
                cmd_openclaw_serve(args)

                MockServer.assert_called_once()
                call_kwargs = MockServer.call_args[1]
                assert call_kwargs["api_key"] is None
