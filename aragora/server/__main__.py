"""Entry point for running aragora.server as a module."""

import argparse
import asyncio
import os
from pathlib import Path

from aragora.server.unified_server import run_unified_server

# Default to localhost for security; use ARAGORA_BIND_HOST=0.0.0.0 for external access
DEFAULT_BIND_HOST = os.environ.get("ARAGORA_BIND_HOST", "127.0.0.1")


def main():
    parser = argparse.ArgumentParser(description="Aragora Unified Server")
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help="Host to bind to (default: 127.0.0.1, use ARAGORA_BIND_HOST env var)",
    )
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP API port")
    parser.add_argument("--ws-port", type=int, help="Alias for --port")
    parser.add_argument("--api-port", type=int, help="Alias for --http-port")
    parser.add_argument("--static-dir", type=Path, help="Static files directory")
    parser.add_argument("--nomic-dir", type=Path, help="Nomic state directory")

    args = parser.parse_args()
    if args.ws_port is not None:
        args.port = args.ws_port
    if args.api_port is not None:
        args.http_port = args.api_port

    asyncio.run(
        run_unified_server(
            http_port=args.http_port,
            ws_port=args.port,
            static_dir=args.static_dir,
            nomic_dir=args.nomic_dir,
        )
    )


if __name__ == "__main__":
    main()
