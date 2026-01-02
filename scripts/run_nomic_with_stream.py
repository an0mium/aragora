#!/usr/bin/env python3
"""
Run nomic loop with live streaming dashboard.

This script starts both the unified server (HTTP + WebSocket) and the nomic loop,
with the nomic loop events streaming to connected clients in real-time.

Usage:
    python scripts/run_nomic_with_stream.py run --cycles 3
    python scripts/run_nomic_with_stream.py run --cycles 3 --http-port 8080 --ws-port 8765
"""

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.server.unified_server import UnifiedServer
from scripts.nomic_loop import NomicLoop


def generate_loop_id() -> str:
    """Generate a unique loop ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"loop_{timestamp}_{short_uuid}"


def get_loop_name(aragora_path: Path) -> str:
    """Generate a human-readable loop name."""
    # Use hostname + path as identifier
    hostname = os.uname().nodename if hasattr(os, 'uname') else "local"
    path_suffix = aragora_path.name
    return f"{hostname}:{path_suffix}"


async def run_with_streaming(
    cycles: int = 3,
    http_port: int = 8080,
    ws_port: int = 8765,
    aragora_path: Path = None,
):
    """Run nomic loop with streaming enabled."""
    aragora_path = aragora_path or Path(__file__).parent.parent

    # Generate unique loop ID for multi-loop tracking
    loop_id = generate_loop_id()
    loop_name = get_loop_name(aragora_path)

    # Resolve paths
    static_dir = aragora_path / "aragora" / "live" / "out"
    if not static_dir.exists():
        print(f"Warning: Static directory not found: {static_dir}")
        print("Run 'cd aragora/live && npm run build' to build the dashboard")
        static_dir = None

    nomic_dir = aragora_path / ".nomic"

    # Create unified server
    server = UnifiedServer(
        http_port=http_port,
        ws_port=ws_port,
        static_dir=static_dir,
        nomic_dir=nomic_dir,
    )

    # Get the emitter for the nomic loop
    emitter = server.emitter

    print("=" * 60)
    print("ARAGORA NOMIC LOOP WITH LIVE STREAMING")
    print("=" * 60)
    print(f"Loop ID:      {loop_id}")
    print(f"Loop Name:    {loop_name}")
    print(f"Dashboard:    http://localhost:{http_port}")
    print(f"Live view:    https://live.aragora.ai")
    print(f"WebSocket:    ws://localhost:{ws_port}")
    print(f"Cycles:       {cycles}")
    print("=" * 60)
    print()

    # Create nomic loop with stream emitter
    loop = NomicLoop(
        aragora_path=aragora_path,
        max_cycles=cycles,
        stream_emitter=emitter,
    )

    # Start server in background task
    async def run_server():
        await server.start()

    server_task = asyncio.create_task(run_server())

    # Give server time to start
    await asyncio.sleep(1)

    # Register this loop instance with the stream server
    server.stream_server.register_loop(
        loop_id=loop_id,
        name=loop_name,
        path=str(aragora_path),
    )

    print("Server started. Running nomic loop...")
    print(f"This loop is now visible at https://live.aragora.ai as '{loop_name}'")
    print()

    try:
        # Run the nomic loop
        await loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Unregister the loop instance
        server.stream_server.unregister_loop(loop_id)
        # Cancel server task
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    print()
    print("Nomic loop complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run nomic loop with live streaming dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run nomic loop with streaming")
    run_parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=3,
        help="Number of cycles to run (default: 3)",
    )
    run_parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="HTTP port for dashboard (default: 8080)",
    )
    run_parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port for streaming (default: 8765)",
    )
    run_parser.add_argument(
        "--aragora-path",
        type=Path,
        default=None,
        help="Path to aragora root (default: auto-detect)",
    )

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_with_streaming(
            cycles=args.cycles,
            http_port=args.http_port,
            ws_port=args.ws_port,
            aragora_path=args.aragora_path,
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
