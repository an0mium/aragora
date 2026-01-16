#!/usr/bin/env python3
"""
Serve the nomic loop live dashboard.

This script starts both the HTTP API server (for static files and REST endpoints)
and the WebSocket server (for real-time streaming events).

Usage:
    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --http-port 3000 --ws-port 8765
    python scripts/serve_dashboard.py --nomic-dir /path/to/aragora/.nomic
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.server.unified_server import run_unified_server


def main():
    parser = argparse.ArgumentParser(
        description="Serve the nomic loop live dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with defaults (HTTP on 8080, WebSocket on 8765)
    python scripts/serve_dashboard.py

    # Custom ports
    python scripts/serve_dashboard.py --http-port 3000 --ws-port 9000

    # Point to a specific nomic directory
    python scripts/serve_dashboard.py --nomic-dir /path/to/project/.nomic

    # Use built static files
    python scripts/serve_dashboard.py --static-dir aragora/live/out
""",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for HTTP API and static files (default: 8080)",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="Port for WebSocket streaming (default: 8765)",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=None,
        help="Directory containing built static files (default: aragora/live/out)",
    )
    parser.add_argument(
        "--nomic-dir",
        type=Path,
        default=None,
        help="Path to .nomic directory (default: auto-detect from aragora path)",
    )

    args = parser.parse_args()

    # Resolve paths
    aragora_root = Path(__file__).parent.parent

    # Static directory - default to live/out if exists
    static_dir = args.static_dir
    if static_dir is None:
        default_static = aragora_root / "aragora" / "live" / "out"
        if default_static.exists():
            static_dir = default_static
        else:
            # Fall back to docs for basic viewing
            docs_dir = aragora_root / "docs"
            if docs_dir.exists():
                static_dir = docs_dir
                print("Note: Using docs/ for static files. Build live dashboard for full features.")

    # Nomic directory - default to aragora/.nomic
    nomic_dir = args.nomic_dir
    if nomic_dir is None:
        nomic_dir = aragora_root / ".nomic"

    print("=" * 60)
    print("ARAGORA LIVE DASHBOARD")
    print("=" * 60)
    print(f"HTTP API:     http://localhost:{args.http_port}")
    print(f"WebSocket:    ws://localhost:{args.ws_port}")
    print(f"Static files: {static_dir or 'Not configured'}")
    print(f"Nomic state:  {nomic_dir}")
    print("=" * 60)
    print()
    print("Endpoints:")
    print("  GET  /                    Dashboard homepage")
    print("  GET  /api/health          Health check")
    print("  GET  /api/nomic/state     Current nomic loop state")
    print("  GET  /api/nomic/log       Recent log entries")
    print("  GET  /api/debates         List recent debates")
    print("  GET  /api/debates/<slug>  Get debate by slug")
    print(f"  WS   ws://...:{args.ws_port}  Real-time event stream")
    print()
    print("Press Ctrl+C to stop the server.")
    print()

    try:
        asyncio.run(
            run_unified_server(
                http_port=args.http_port,
                ws_port=args.ws_port,
                static_dir=static_dir,
                nomic_dir=nomic_dir,
            )
        )
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
