"""
Standalone OpenClaw Governance Gateway Server.

Runs the OpenClaw gateway as a minimal standalone service without
requiring the full Aragora debate engine, knowledge mound, or other
heavyweight subsystems.

Usage:
    aragora openclaw serve --port 8100
    aragora openclaw serve --port 8100 --policy ./policy.yaml

Or directly:
    python -m aragora.compat.openclaw.standalone --port 8100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8100
DEFAULT_POLICY = "deny"


class StandaloneGatewayServer:
    """Minimal HTTP server exposing OpenClaw gateway endpoints.

    This server runs independently of the full Aragora stack, providing:
    - Session management (create, get, list, close)
    - Action execution with policy enforcement
    - Credential management (AES-256-GCM encrypted)
    - Health, metrics, and audit endpoints
    - Policy management via YAML rules
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        policy_file: str | None = None,
        default_policy: str = DEFAULT_POLICY,
        cors_origins: list[str] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.policy_file = policy_file
        self.default_policy = default_policy
        self.cors_origins = cors_origins or ["*"]
        self._handler = None
        self._running = False

    def _init_handler(self) -> None:
        """Lazily initialize the gateway handler."""
        from aragora.server.handlers.openclaw import (
            OpenClawGatewayHandler,
        )

        server_context: dict[str, Any] = {
            "standalone_mode": True,
            "policy_file": self.policy_file,
            "default_policy": self.default_policy,
        }
        self._handler = OpenClawGatewayHandler(server_context)

    def _build_routes(self) -> list[tuple[str, str, str]]:
        """Return list of (method, path, handler_method) tuples."""
        return [
            # Session management
            ("POST", "/api/gateway/openclaw/sessions", "handle_post"),
            ("GET", "/api/gateway/openclaw/sessions", "handle"),
            ("GET", "/api/gateway/openclaw/sessions/{id}", "handle"),
            ("DELETE", "/api/gateway/openclaw/sessions/{id}", "handle_delete"),
            # Action management
            ("POST", "/api/gateway/openclaw/actions", "handle_post"),
            ("GET", "/api/gateway/openclaw/actions/{id}", "handle"),
            ("POST", "/api/gateway/openclaw/actions/{id}/cancel", "handle_post"),
            # Credential management
            ("POST", "/api/gateway/openclaw/credentials", "handle_post"),
            ("GET", "/api/gateway/openclaw/credentials", "handle"),
            ("DELETE", "/api/gateway/openclaw/credentials/{id}", "handle_delete"),
            ("POST", "/api/gateway/openclaw/credentials/{id}/rotate", "handle_post"),
            # Policy management
            ("GET", "/api/gateway/openclaw/policy/rules", "handle"),
            ("POST", "/api/gateway/openclaw/policy/rules", "handle_post"),
            ("PUT", "/api/gateway/openclaw/policy/rules/{id}", "handle_put"),
            # Approval management
            ("GET", "/api/gateway/openclaw/approvals", "handle"),
            ("POST", "/api/gateway/openclaw/approvals/{id}/approve", "handle_post"),
            ("POST", "/api/gateway/openclaw/approvals/{id}/deny", "handle_post"),
            # Admin endpoints
            ("GET", "/api/gateway/openclaw/health", "handle"),
            ("GET", "/api/gateway/openclaw/metrics", "handle"),
            ("GET", "/api/gateway/openclaw/audit", "handle"),
            ("GET", "/api/gateway/openclaw/stats", "handle"),
        ]

    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle an incoming HTTP request."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            if not request_line:
                writer.close()
                return

            request_str = request_line.decode("utf-8", errors="replace").strip()
            parts = request_str.split(" ", 2)
            if len(parts) < 2:
                await self._send_response(writer, 400, {"error": "Bad request"})
                return

            method = parts[0]
            path = parts[1].split("?")[0]
            query_string = parts[1].split("?")[1] if "?" in parts[1] else ""

            # Parse query params
            query_params: dict[str, Any] = {}
            if query_string:
                for param in query_string.split("&"):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        query_params[key] = value

            # Read headers
            headers: dict[str, str] = {}
            while True:
                header_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
                header_str = header_line.decode("utf-8", errors="replace").strip()
                if not header_str:
                    break
                if ":" in header_str:
                    key, value = header_str.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            # Read body if present
            body = None
            content_length = int(headers.get("content-length", "0"))
            if content_length > 0:
                body_bytes = await asyncio.wait_for(reader.readexactly(content_length), timeout=30.0)
                try:
                    body = json.loads(body_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = body_bytes.decode("utf-8", errors="replace")

            # CORS preflight
            if method == "OPTIONS":
                await self._send_cors_response(writer)
                return

            # Health check shortcut
            if path == "/health" or path == "/":
                await self._send_response(writer, 200, {
                    "status": "healthy",
                    "service": "aragora-openclaw-gateway",
                    "version": self._get_version(),
                })
                return

            # Route to handler
            if not path.startswith("/api/gateway/openclaw/"):
                await self._send_response(writer, 404, {"error": "Not found"})
                return

            if self._handler is None:
                self._init_handler()

            # Create a minimal handler context
            handler_ctx = _MinimalHandlerContext(headers, body, query_params)

            try:
                if method == "GET":
                    result = self._handler.handle(path, query_params, handler_ctx)
                elif method == "POST":
                    result = self._handler.handle_post(path, query_params, handler_ctx)
                elif method == "PUT":
                    result = self._handler.handle_put(path, query_params, handler_ctx)
                elif method == "DELETE":
                    result = self._handler.handle_delete(path, query_params, handler_ctx)
                else:
                    await self._send_response(writer, 405, {"error": "Method not allowed"})
                    return

                if result is None:
                    await self._send_response(writer, 404, {"error": "Not found"})
                elif isinstance(result, tuple):
                    status_code, response_body = result
                    await self._send_response(writer, status_code, response_body)
                elif isinstance(result, dict):
                    await self._send_response(writer, 200, result)
                else:
                    await self._send_response(writer, 200, {"data": str(result)})
            except PermissionError as e:
                await self._send_response(writer, 403, {"error": str(e)})
            except ValueError as e:
                await self._send_response(writer, 400, {"error": str(e)})
            except Exception as e:
                logger.error("Request handler error: %s", e, exc_info=True)
                await self._send_response(writer, 500, {"error": "Internal server error"})

        except asyncio.TimeoutError:
            try:
                await self._send_response(writer, 408, {"error": "Request timeout"})
            except Exception as e:
                logger.debug("Failed to send response: %s", e)
        except Exception as e:
            logger.error("Connection error: %s", e, exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception as e:
                logger.debug("Error during connection cleanup: %s", e)

    async def _send_response(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        body: dict | str | None = None,
    ) -> None:
        """Send an HTTP response."""
        status_messages = {
            200: "OK",
            201: "Created",
            204: "No Content",
            400: "Bad Request",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            408: "Request Timeout",
            500: "Internal Server Error",
        }
        status_msg = status_messages.get(status_code, "Unknown")

        if body is not None:
            body_str = json.dumps(body) if isinstance(body, dict) else str(body)
            body_bytes = body_str.encode("utf-8")
        else:
            body_bytes = b""

        headers = [
            f"HTTP/1.1 {status_code} {status_msg}",
            "Content-Type: application/json",
            f"Content-Length: {len(body_bytes)}",
            "Server: aragora-openclaw-gateway",
        ]
        # CORS headers
        for origin in self.cors_origins:
            headers.append(f"Access-Control-Allow-Origin: {origin}")
            break  # Only first origin

        response = "\r\n".join(headers) + "\r\n\r\n"
        writer.write(response.encode("utf-8"))
        if body_bytes:
            writer.write(body_bytes)
        await writer.drain()

    async def _send_cors_response(self, writer: asyncio.StreamWriter) -> None:
        """Send CORS preflight response."""
        headers = [
            "HTTP/1.1 204 No Content",
            f"Access-Control-Allow-Origin: {self.cors_origins[0] if self.cors_origins else '*'}",
            "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type, Authorization, X-Tenant-ID",
            "Access-Control-Max-Age: 86400",
            "Content-Length: 0",
        ]
        response = "\r\n".join(headers) + "\r\n\r\n"
        writer.write(response.encode("utf-8"))
        await writer.drain()

    def _get_version(self) -> str:
        """Get the Aragora version."""
        try:
            from aragora import __version__
            return __version__
        except ImportError:
            return "unknown"

    async def start(self) -> None:
        """Start the gateway server."""
        self._init_handler()
        self._running = True

        server = await asyncio.start_server(
            self._handle_request,
            self.host,
            self.port,
        )

        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        logger.info("OpenClaw Gateway serving on %s", addrs)

        print("\n" + "=" * 60)
        print("ARAGORA OPENCLAW GOVERNANCE GATEWAY")
        print("=" * 60)
        print(f"\n  Listening: http://{self.host}:{self.port}")
        print(f"  Health:    http://{self.host}:{self.port}/health")
        print(f"  Policy:    {self.default_policy} (default)")
        if self.policy_file:
            print(f"  Rules:     {self.policy_file}")
        print(f"\n  API Base:  http://{self.host}:{self.port}/api/gateway/openclaw/")
        print("\n  Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        async with server:
            await server.serve_forever()

    async def stop(self) -> None:
        """Stop the gateway server."""
        self._running = False


class _MinimalHandlerContext:
    """Minimal request context for standalone mode.

    Provides the interface expected by OpenClawGatewayHandler methods
    without requiring the full Aragora server infrastructure.
    """

    def __init__(
        self,
        headers: dict[str, str],
        body: Any | None,
        query_params: dict[str, Any],
    ) -> None:
        self.headers = headers
        self.body = body
        self.query_params = query_params
        # Extract auth info from headers
        self.user_id = headers.get("x-user-id", "anonymous")
        self.tenant_id = headers.get("x-tenant-id")
        self.org_id = self.tenant_id


def cmd_openclaw_serve(args: argparse.Namespace) -> None:
    """Handle 'openclaw serve' command."""
    host = getattr(args, "host", DEFAULT_HOST)
    port = getattr(args, "port", DEFAULT_PORT)
    policy_file = getattr(args, "policy", None)
    default_policy = getattr(args, "default_policy", DEFAULT_POLICY)
    cors = getattr(args, "cors", "*")

    cors_origins = [o.strip() for o in cors.split(",") if o.strip()]

    # Configure logging
    log_level = getattr(args, "log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = StandaloneGatewayServer(
        host=host,
        port=port,
        policy_file=policy_file,
        default_policy=default_policy,
        cors_origins=cors_origins,
    )

    def handle_signal(signum, _frame):
        print("\nShutting down gateway...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nGateway stopped.")


def main() -> None:
    """CLI entrypoint for standalone gateway."""
    parser = argparse.ArgumentParser(
        description="Aragora OpenClaw Governance Gateway (Standalone)",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--policy", help="Path to policy YAML file for action filtering",
    )
    parser.add_argument(
        "--default-policy", default=DEFAULT_POLICY, choices=["allow", "deny"],
        help="Default policy when no rule matches (default: deny)",
    )
    parser.add_argument(
        "--cors", default="*", help="CORS allowed origins (comma-separated, default: *)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()
    cmd_openclaw_serve(args)


if __name__ == "__main__":
    main()
