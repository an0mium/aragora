#!/usr/bin/env python3
"""
Batch debate processing CLI command.

Processes multiple debates from a JSONL or JSON file.
Supports both local processing and server-side batch API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.core import DebateResult


def create_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the batch command subparser."""
    from aragora.cli.main import DEFAULT_API_URL

    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple debates from a file",
        description="""
Run multiple debates from a JSONL or JSON file.

Input file format (JSONL - one JSON object per line):
    {"question": "Design a rate limiter", "agents": "anthropic-api,openai-api"}
    {"question": "Implement caching", "rounds": 4}
    {"question": "Security review", "priority": 10}

Or JSON array:
    [{"question": "Topic 1"}, {"question": "Topic 2"}]

Examples:
    aragora batch debates.jsonl
    aragora batch debates.json --server --wait
    aragora batch debates.jsonl --output results.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batch_parser.add_argument(
        "input",
        help="Path to JSONL or JSON file with debate items",
    )
    batch_parser.add_argument(
        "--server",
        "-s",
        action="store_true",
        help="Submit to server batch API instead of processing locally",
    )
    batch_parser.add_argument(
        "--url",
        "-u",
        default=DEFAULT_API_URL,
        help=f"Server URL (default: {DEFAULT_API_URL})",
    )
    batch_parser.add_argument(
        "--token",
        "-t",
        help="API authentication token",
    )
    batch_parser.add_argument(
        "--webhook",
        "-w",
        help="Webhook URL for completion notification",
    )
    batch_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for batch completion (server mode only)",
    )
    batch_parser.add_argument(
        "--agents",
        "-a",
        default="anthropic-api,openai-api",
        help="Default agents for items without agents specified",
    )
    batch_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=3,
        help="Default rounds for items without rounds specified",
    )
    batch_parser.add_argument(
        "--output",
        "-o",
        help="Output path for results JSON (local mode only)",
    )
    batch_parser.set_defaults(func=cmd_batch)


def cmd_batch(args: argparse.Namespace) -> None:
    """Handle 'batch' command - run multiple debates from file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("BATCH DEBATE PROCESSING")
    print("=" * 60)

    # Read input file (JSONL or JSON array)
    items = _read_input_file(input_path)

    if not items:
        print("Error: No valid debate items found in input file")
        sys.exit(1)

    print(f"\nInput: {input_path}")
    print(f"Items: {len(items)}")
    print(f"Mode: {'server' if args.server else 'local'}")

    if args.server:
        _batch_via_server(items, args)
    else:
        _batch_local(items, args)


def _read_input_file(input_path: Path) -> list[dict[str, Any]]:
    """Read and parse input file (JSONL or JSON array)."""
    items: list[dict[str, Any]] = []
    try:
        content = input_path.read_text().strip()
        if content.startswith("["):
            # JSON array
            items = json.loads(content)
        else:
            # JSONL format
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    return items


def _batch_via_server(items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Submit batch to server API."""
    server_url = args.url.rstrip("/")

    print(f"\nSubmitting to {server_url}/api/debates/batch...")

    # Prepare batch request
    batch_data: dict[str, Any] = {"items": items}
    if args.webhook:
        batch_data["webhook_url"] = args.webhook

    # Submit batch
    try:
        req = urllib.request.Request(
            f"{server_url}/api/debates/batch",
            data=json.dumps(batch_data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        if args.token:
            req.add_header("Authorization", f"Bearer {args.token}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())

        if not result.get("success"):
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

        batch_id = result.get("batch_id")
        print("\nBatch submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"Items queued: {result.get('items_queued', len(items))}")
        print(f"Status URL: {result.get('status_url', '')}")

        if args.wait:
            print("\nWaiting for completion...")
            _poll_batch_status(server_url, batch_id, args.token)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"Server error ({e.code}): {error_body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        sys.exit(1)


def _poll_batch_status(server_url: str, batch_id: str, token: str | None = None) -> None:
    """Poll batch status until completion."""
    poll_interval = 5  # seconds
    max_polls = 360  # 30 minutes max

    for _ in range(max_polls):
        try:
            req = urllib.request.Request(
                f"{server_url}/api/debates/batch/{batch_id}/status",
                method="GET",
            )
            if token:
                req.add_header("Authorization", f"Bearer {token}")

            with urllib.request.urlopen(req, timeout=10) as resp:
                status = json.loads(resp.read().decode())

            progress = status.get("progress_percent", 0)
            completed = status.get("completed", 0)
            failed = status.get("failed", 0)
            total = status.get("total_items", 0)
            batch_status = status.get("status", "unknown")

            print(
                f"\r[{progress:5.1f}%] {completed}/{total} completed, {failed} failed - {batch_status}",
                end="",
                flush=True,
            )

            if batch_status in ("completed", "partial", "failed", "cancelled"):
                print("\n")
                if batch_status == "completed":
                    print("Batch completed successfully!")
                elif batch_status == "partial":
                    print(f"Batch partially completed: {completed} succeeded, {failed} failed")
                elif batch_status == "failed":
                    print("Batch failed!")
                else:
                    print("Batch cancelled")
                return

            time.sleep(poll_interval)

        except Exception as e:
            print(f"\nWarning: Poll error: {e}")
            time.sleep(poll_interval)

    print("\nTimeout: Batch did not complete within 30 minutes")


def _batch_local(items: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Process batch locally (sequential)."""
    from aragora.cli.main import run_debate

    results: list[dict[str, Any]] = []
    total = len(items)
    start_time = time.time()

    print("\nProcessing debates locally...\n")

    for i, item in enumerate(items):
        question = item.get("question", "")
        agents = item.get("agents", args.agents)
        rounds = item.get("rounds", args.rounds)

        print(f"[{i + 1}/{total}] {question[:50]}...")

        try:
            result: DebateResult = asyncio.run(
                run_debate(
                    task=question,
                    agents_str=agents,
                    rounds=rounds,
                    consensus="majority",
                    learn=False,
                    enable_audience=False,
                )
            )

            results.append(
                {
                    "question": question,
                    "success": True,
                    "consensus_reached": result.consensus_reached,
                    "confidence": result.confidence,
                    "final_answer": result.final_answer[:200],
                }
            )
            print(
                f"    => {'Consensus' if result.consensus_reached else 'No consensus'} ({result.confidence:.0%})"
            )

        except Exception as e:
            results.append(
                {
                    "question": question,
                    "success": False,
                    "error": str(e),
                }
            )
            print(f"    => ERROR: {e}")

    elapsed = time.time() - start_time
    succeeded = sum(1 for r in results if r.get("success"))

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {total - succeeded}")
    print(f"Duration: {elapsed:.1f}s")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved: {output_path}")
