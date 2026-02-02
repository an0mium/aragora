"""
Async vs Sync Example

Demonstrates the differences between synchronous and asynchronous
client usage. Shows when to use each pattern.

Usage:
    python examples/async_vs_sync.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from aragora_sdk import AragoraAsyncClient, AragoraClient

# =============================================================================
# Synchronous Pattern
# =============================================================================


def sync_example() -> dict[str, Any]:
    """Run a debate using the synchronous client.

    Use the synchronous client when:
    - Running in scripts or CLI tools
    - Simple, single-request operations
    - Integrating with sync-only frameworks
    - Polling-based workflows
    """
    print("=== Synchronous Client Example ===\n")

    # Create sync client (simple instantiation)
    client = AragoraClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    )

    start = time.time()

    # Create debate - blocking call
    print("Creating debate...")
    debate = client.debates.create(
        task="What is the best approach to learn a new programming language?",
        agents=["claude", "gpt-4"],
        rounds=2,
    )
    print(f"Created: {debate['debate_id']}")

    # Poll for completion - blocking loop
    print("Waiting for completion...")
    while debate.get("status") in ("running", "pending"):
        time.sleep(1)
        debate = client.debates.get(debate["debate_id"])
        print(f"  Status: {debate['status']}")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s (sync)")

    return debate


# =============================================================================
# Asynchronous Pattern
# =============================================================================


async def async_example() -> dict[str, Any]:
    """Run a debate using the asynchronous client.

    Use the asynchronous client when:
    - Building async web applications (FastAPI, aiohttp)
    - Need concurrent API calls
    - Real-time streaming with WebSockets
    - High-throughput batch operations
    """
    print("\n=== Asynchronous Client Example ===\n")

    start = time.time()

    # Create async client using context manager (recommended)
    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        # Create debate - non-blocking
        print("Creating debate...")
        debate = await client.debates.create(
            task="What is the best approach to learn a new programming language?",
            agents=["claude", "gpt-4"],
            rounds=2,
        )
        print(f"Created: {debate['debate_id']}")

        # Poll for completion - non-blocking
        print("Waiting for completion...")
        while debate.get("status") in ("running", "pending"):
            await asyncio.sleep(1)
            debate = await client.debates.get(debate["debate_id"])
            print(f"  Status: {debate['status']}")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s (async)")

    return debate


# =============================================================================
# Concurrent Operations (Async Advantage)
# =============================================================================


async def concurrent_operations() -> None:
    """Demonstrate running multiple operations concurrently.

    This is where async really shines - you can run multiple
    API calls in parallel without blocking.
    """
    print("\n=== Concurrent Operations (Async) ===\n")

    async with AragoraAsyncClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    ) as client:
        start = time.time()

        # Create 3 debates concurrently
        print("Creating 3 debates concurrently...")
        tasks = [
            client.debates.create(
                task=f"Question {i}: What makes code readable?",
                agents=["claude", "gpt-4"],
                rounds=1,
            )
            for i in range(1, 4)
        ]

        # Wait for all to complete
        debates = await asyncio.gather(*tasks)

        for debate in debates:
            print(f"  Created: {debate['debate_id']}")

        elapsed = time.time() - start
        print(f"\n3 debates created in {elapsed:.1f}s (concurrent)")
        print("Compare: Sequential would take ~3x longer!")


# =============================================================================
# Async Context Manager Patterns
# =============================================================================


async def context_manager_patterns() -> None:
    """Demonstrate proper async resource management."""
    print("\n=== Async Context Manager Patterns ===\n")

    # Pattern 1: Context manager (recommended)
    print("Pattern 1: Context manager (recommended)")
    print("  async with AragoraAsyncClient(...) as client:")
    print("      result = await client.debates.list()")
    print("  # Client is automatically closed")

    # Pattern 2: Manual management
    print("\nPattern 2: Manual management")
    print("  client = AragoraAsyncClient(...)")
    print("  try:")
    print("      result = await client.debates.list()")
    print("  finally:")
    print("      await client.close()")

    # Pattern 3: Multiple clients
    print("\nPattern 3: Nested context managers")
    print("  async with AragoraAsyncClient(...) as client1:")
    print("      async with AragoraAsyncClient(...) as client2:")
    print("          # Use both clients")


# =============================================================================
# When to Use Each
# =============================================================================


def print_usage_guide() -> None:
    """Print guidance on when to use sync vs async."""
    print("\n" + "=" * 60)
    print("WHEN TO USE EACH PATTERN")
    print("=" * 60)

    print("""
USE SYNCHRONOUS (AragoraClient) WHEN:
  - Writing CLI scripts or one-off tools
  - Simple request/response workflows
  - Integrating with Django or Flask (traditional)
  - Debugging or prototyping
  - Single-threaded applications

USE ASYNCHRONOUS (AragoraAsyncClient) WHEN:
  - Building FastAPI/Starlette applications
  - Need concurrent API calls (batch processing)
  - Real-time streaming with WebSockets
  - High-throughput scenarios
  - Building async libraries or frameworks

PERFORMANCE COMPARISON:
  - Single request: Similar performance
  - 10 requests: Async ~5-10x faster (concurrent)
  - Streaming: Only available with async client
  - Memory: Async slightly higher overhead per connection

CODE COMPLEXITY:
  - Sync: Simpler, easier to debug
  - Async: Requires understanding of async/await
  - Mixing: Avoid mixing sync calls in async code (blocks event loop)
""")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run sync vs async demonstrations."""
    print("Aragora SDK: Async vs Sync Comparison")
    print("=" * 60)

    # Print usage guide first
    print_usage_guide()

    # Check if we should run actual examples
    run_examples = os.environ.get("RUN_EXAMPLES", "false").lower() == "true"

    if run_examples:
        # Run sync example
        sync_example()

        # Run async examples
        asyncio.run(async_example())
        asyncio.run(concurrent_operations())
        asyncio.run(context_manager_patterns())
    else:
        print("\nSet RUN_EXAMPLES=true to run actual API examples.")
        print("Example: RUN_EXAMPLES=true python examples/async_vs_sync.py")


if __name__ == "__main__":
    main()
