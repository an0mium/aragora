#!/usr/bin/env python3
"""
16_observability.py - Monitor system health and performance.

Shows how to query Aragora's observability endpoints for system health,
circuit breaker states, and self-improvement cycle metrics. Useful for
building monitoring dashboards or alerting integrations.

Usage:
    python 16_observability.py                    # Show system status
    python 16_observability.py --dry-run          # Preview
    python 16_observability.py --watch            # Continuous monitoring
"""

import argparse
import asyncio
from aragora_sdk import AragoraClient


async def show_system_health(dry_run: bool = False) -> dict:
    """Show comprehensive system health status."""

    client = AragoraClient()

    if dry_run:
        print("[DRY RUN] Would check system health")
        return {"status": "dry_run"}

    # System health
    health = await client.observability.health()
    status = health.get("status", "unknown")
    icon = "OK" if status == "healthy" else "!!"
    print(f"System Health: [{icon}] {status}")

    # Component status
    components = health.get("components", {})
    for name, state in components.items():
        c_icon = "+" if state.get("healthy") else "-"
        print(f"  [{c_icon}] {name}: {state.get('status', '?')}")

    # Circuit breakers
    breakers = await client.observability.circuit_breakers()
    print(f"\nCircuit Breakers ({len(breakers)}):")
    for cb in breakers:
        state = cb.get("state", "unknown")
        failures = cb.get("failure_count", 0)
        print(f"  {cb['name']}: {state} (failures: {failures})")

    # Self-improvement metrics
    si_metrics = await client.observability.self_improve_metrics()
    print("\nSelf-Improvement Metrics:")
    print(f"  Total cycles: {si_metrics.get('total_cycles', 0)}")
    print(f"  Success rate: {si_metrics.get('success_rate', 0):.1%}")
    print(f"  Avg files changed: {si_metrics.get('avg_files_changed', 0):.1f}")
    print(f"  Total improvements: {si_metrics.get('total_improvements', 0)}")

    return {"health": health, "breakers": breakers, "si": si_metrics}


async def watch_health(interval: int = 10) -> None:
    """Continuously monitor system health."""

    client = AragoraClient()
    print(f"Monitoring system health every {interval}s (Ctrl+C to stop)\n")

    try:
        while True:
            health = await client.observability.health()
            status = health.get("status", "unknown")
            components = health.get("components", {})
            healthy = sum(1 for c in components.values() if c.get("healthy"))
            total = len(components)
            print(f"[{status}] {healthy}/{total} components healthy")
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped monitoring.")


def main():
    parser = argparse.ArgumentParser(description="System observability via SDK")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=10, help="Watch interval (seconds)")
    args = parser.parse_args()

    if args.watch:
        asyncio.run(watch_health(args.interval))
    else:
        asyncio.run(show_system_health(args.dry_run))


if __name__ == "__main__":
    main()
