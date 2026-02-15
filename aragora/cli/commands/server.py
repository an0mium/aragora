"""
Server management CLI commands.

Contains the 'serve' command for running the live debate server.
"""

import argparse
import asyncio
import multiprocessing
import signal
import sys
from pathlib import Path


def cmd_serve(args: argparse.Namespace) -> None:
    """Handle 'serve' command - run live debate server."""
    import os

    # Demo mode: no API keys needed, uses SQLite, loads seed data
    if getattr(args, "demo", False):
        os.environ.setdefault("ARAGORA_OFFLINE", "true")
        os.environ.setdefault("ARAGORA_DEMO_MODE", "true")
        os.environ.setdefault("ARAGORA_DB_BACKEND", "sqlite")
        os.environ.setdefault("ARAGORA_ENV", "development")

    try:
        from aragora.server.unified_server import run_unified_server
    except ImportError as e:
        print(f"Error importing server modules: {e}")
        print("Make sure websockets and aiohttp are installed: pip install websockets aiohttp")
        return

    # Get nomic_dir from environment (same as db_config.get_nomic_dir())
    from aragora.persistence.db_config import get_nomic_dir

    nomic_dir = get_nomic_dir()
    nomic_dir.mkdir(parents=True, exist_ok=True)

    # Determine static directory (Live Dashboard)
    static_dir = None
    live_dir = Path(__file__).parent.parent.parent / "live" / "dist"
    if live_dir.exists():
        static_dir = live_dir
    else:
        # Fall back to docs directory for viewer.html
        docs_dir = Path(__file__).parent.parent.parent.parent / "docs"
        if docs_dir.exists():
            static_dir = docs_dir

    workers = getattr(args, "workers", 1)
    workers = max(1, workers)

    is_demo = getattr(args, "demo", False)

    print("\n" + "=" * 60)
    print("ARAGORA LIVE DEBATE SERVER" + (" [DEMO MODE]" if is_demo else ""))
    print("=" * 60)

    if workers == 1:
        print(f"\nWebSocket: ws://{args.host}:{args.ws_port}")
        print(f"HTTP API:  http://{args.host}:{args.api_port}")
        if static_dir:
            print(f"Dashboard: http://{args.host}:{args.api_port}/")
        print("\nPress Ctrl+C to stop\n")
        print("=" * 60 + "\n")

        try:
            asyncio.run(
                run_unified_server(
                    http_port=args.api_port,
                    ws_port=args.ws_port,
                    http_host=args.host,
                    ws_host=args.host,
                    static_dir=static_dir,
                    nomic_dir=nomic_dir,
                )
            )
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
    else:
        # Multi-worker mode
        print(f"\nWorkers: {workers}")
        print(f"HTTP ports: {args.api_port}-{args.api_port + workers - 1}")
        print(f"WS ports: {args.ws_port}-{args.ws_port + workers - 1}")
        print("\nTip: Use a load balancer (nginx/haproxy) to distribute traffic.")
        print("\nPress Ctrl+C to stop\n")
        print("=" * 60 + "\n")

        def run_worker(http_port, ws_port, host, static, data_dir):
            asyncio.run(
                run_unified_server(
                    http_port=http_port,
                    ws_port=ws_port,
                    http_host=host,
                    ws_host=host,
                    static_dir=static,
                    nomic_dir=data_dir,
                )
            )

        processes: list[multiprocessing.Process] = []

        def shutdown_workers(signum, _frame):
            print("\nShutting down workers...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_workers)
        signal.signal(signal.SIGTERM, shutdown_workers)

        for i in range(workers):
            http_port = args.api_port + i
            ws_port = args.ws_port + i
            p = multiprocessing.Process(
                target=run_worker,
                args=(http_port, ws_port, args.host, static_dir, nomic_dir),
                name=f"aragora-worker-{i}",
            )
            p.start()
            processes.append(p)
            print(f"  Worker {i}: HTTP={http_port}, WS={ws_port} (PID {p.pid})")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            shutdown_workers(None, None)
