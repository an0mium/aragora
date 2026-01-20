"""Entry point for running aragora.server as a module.

Usage:
    python -m aragora.server --port 8080
    python -m aragora.server --workers 4  # Production: 4 worker processes
"""

import argparse
import asyncio
import multiprocessing
import os
import signal
import sys
from pathlib import Path

from aragora.server.unified_server import run_unified_server

# Default to localhost for security; use ARAGORA_BIND_HOST=0.0.0.0 for external access
DEFAULT_BIND_HOST = os.environ.get("ARAGORA_BIND_HOST", "127.0.0.1")


def _run_worker(http_port: int, ws_port: int, host: str, static_dir, nomic_dir):
    """Run a single server worker process."""
    asyncio.run(
        run_unified_server(
            http_port=http_port,
            ws_port=ws_port,
            http_host=host,
            ws_host=host,
            static_dir=static_dir,
            nomic_dir=nomic_dir,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aragora Unified Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Production deployment with multiple workers:
    python -m aragora.server --workers 4 --host 0.0.0.0

    For best results, use a load balancer (nginx, haproxy) in front of workers.
    Each worker runs on a different port: base_port, base_port+1, ...
        """,
    )
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
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of worker processes (default: 1). For production, use 2-4x CPU cores.",
    )

    args = parser.parse_args()
    if args.ws_port is not None:
        args.port = args.ws_port
    if args.api_port is not None:
        args.http_port = args.api_port

    workers = max(1, args.workers)

    if workers == 1:
        # Single worker mode - run directly
        asyncio.run(
            run_unified_server(
                http_port=args.http_port,
                ws_port=args.port,
                http_host=args.host,
                ws_host=args.host,
                static_dir=args.static_dir,
                nomic_dir=args.nomic_dir,
            )
        )
    else:
        # Multi-worker mode - spawn worker processes
        print(f"Starting {workers} worker processes...")
        print(f"HTTP ports: {args.http_port}-{args.http_port + workers - 1}")
        print(f"WS ports: {args.port}-{args.port + workers - 1}")
        print("\nTip: Use a load balancer to distribute traffic across workers.\n")

        processes = []

        def shutdown_workers(signum, frame):
            """Gracefully shutdown all workers."""
            print("\nShutting down workers...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_workers)
        signal.signal(signal.SIGTERM, shutdown_workers)

        for i in range(workers):
            http_port = args.http_port + i
            ws_port = args.port + i
            p = multiprocessing.Process(
                target=_run_worker,
                args=(http_port, ws_port, args.host, args.static_dir, args.nomic_dir),
                name=f"aragora-worker-{i}",
            )
            p.start()
            processes.append(p)
            print(f"  Worker {i}: HTTP={http_port}, WS={ws_port} (PID {p.pid})")

        # Wait for all workers
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            shutdown_workers(None, None)


if __name__ == "__main__":
    main()
