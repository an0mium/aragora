#!/usr/bin/env python3
"""
Staging server entry point for EC2 deployment.

Usage:
    python run_staging.py --host 0.0.0.0 --port 8765 --nomic-dir .nomic
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
except ImportError:
    pass  # dotenv not installed, use environment variables directly

from aragora.server.unified_server import run_unified_server


def main():
    parser = argparse.ArgumentParser(description="Run Aragora staging server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the server (default: 8765, used for both HTTP API and WebSocket)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=None,
        help="Port for HTTP API (default: same as --port)",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=None,
        help="Port for WebSocket (default: --port + 1 if --http-port specified)",
    )
    parser.add_argument(
        "--nomic-dir",
        type=Path,
        default=Path(".nomic"),
        help="Path to .nomic directory (default: .nomic)",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=None,
        help="Path to static files directory (optional)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Determine ports
    # If only --port is specified, use it for HTTP and port+1 for WebSocket
    http_port = args.http_port if args.http_port else args.port
    ws_port = args.ws_port if args.ws_port else (args.port + 1 if args.http_port else 8765)

    # For staging, we typically want HTTP on the main port
    # The systemd service expects the health check on the specified port
    if args.http_port is None and args.ws_port is None:
        # Default staging behavior: HTTP on specified port, WS on port+1
        http_port = args.port
        ws_port = args.port + 1

    # Resolve nomic directory
    nomic_dir = args.nomic_dir.resolve() if args.nomic_dir else None
    if nomic_dir and not nomic_dir.exists():
        logging.warning(f"Creating nomic directory: {nomic_dir}")
        nomic_dir.mkdir(parents=True, exist_ok=True)

    # Resolve static directory
    static_dir = args.static_dir.resolve() if args.static_dir else None

    logging.info(f"Starting Aragora staging server")
    logging.info(f"  HTTP port: {http_port}")
    logging.info(f"  WebSocket port: {ws_port}")
    logging.info(f"  Nomic dir: {nomic_dir}")
    if static_dir:
        logging.info(f"  Static dir: {static_dir}")

    # Run the server
    try:
        asyncio.run(
            run_unified_server(
                http_port=http_port,
                ws_port=ws_port,
                nomic_dir=nomic_dir,
                static_dir=static_dir,
            )
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
