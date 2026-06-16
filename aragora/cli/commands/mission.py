"""Native Mission CLI commands."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from aragora.nomic.mission import MissionSpec, NativeMissionRunner


def cmd_mission(args: argparse.Namespace) -> int:
    """Execute the 'mission' command."""
    return asyncio.run(_cmd_mission_async(args))


async def _cmd_mission_async(args: argparse.Namespace) -> int:
    # 1. Parse tracks if provided
    tracks = None
    if getattr(args, "tracks", None):
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    # 2. Generate mission_id
    mission_id = f"mission-{uuid.uuid4().hex[:12]}"

    # 3. Construct MissionSpec
    try:
        spec = MissionSpec(
            goal=args.goal,
            mission_id=mission_id,
            budget_usd=args.budget,
            max_hours=args.max_hours,
            relay=args.relay,
            auto_settle_max_tier=args.auto_settle_max_tier,
        )
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    # 4. Instantiate NativeMissionRunner
    try:
        runner = NativeMissionRunner()
        print(f"Ingesting mission '{mission_id}'...")
        print(f"Goal: {spec.goal!r}")
        if spec.budget_usd is not None:
            print(f"Budget: ${spec.budget_usd:.2f} USD")
        if spec.max_hours is not None:
            print(f"Max hours: {spec.max_hours}h")
        print(f"Relay: {spec.relay}")
        print(f"Auto-settle max tier: {spec.auto_settle_max_tier}")
        if tracks:
            print(f"Tracks: {', '.join(tracks)}")

        work_items = await runner.ingest_mission(spec, tracks=tracks)

        print(f"Success: Mission '{mission_id}' ingested.")
        print(f"Decomposed into {len(work_items)} work items:")
        for item in work_items:
            print(f"  - [{item.item_id}] ({item.complexity}) {item.description}")
        return 0
    except RuntimeError as e:
        # This will catch the RuntimeError if enable_native_mission is OFF
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1
