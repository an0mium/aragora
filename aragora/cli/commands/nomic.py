"""
Nomic Loop CLI commands.

Provides CLI access to the autonomous self-improvement system.
Commands:
- gt nomic run --cycles <n> - Run improvement cycles
- gt nomic status - Show current loop status
- gt nomic history - View cycle history
- gt nomic resume <cycle_id> - Resume from a checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


def _get_api_client(api_url: str | None = None, api_key: str | None = None):
    """Get API client if available and server is reachable."""
    try:
        from aragora.client import AragoraClient

        url = api_url or os.environ.get("ARAGORA_API_URL", DEFAULT_API_URL)
        key = api_key or os.environ.get("ARAGORA_API_TOKEN") or os.environ.get("ARAGORA_API_KEY")
        client = AragoraClient(base_url=url, api_key=key)
        # Quick health check (public)
        client.system.health()
        return client
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _resolve_api_mode(
    use_api: bool,
    use_local: bool,
    api_url: str | None,
    api_key: str | None,
    *,
    supports_api: bool,
) -> tuple[bool, Any | None]:
    """Resolve whether to use API mode and return client if available."""
    if use_local:
        return False, None
    if use_api:
        if not supports_api:
            print("Error: This subcommand is not available in API mode.", flush=True)
            raise SystemExit(1)
        client = _get_api_client(api_url, api_key)
        if client is None:
            print("Error: --api specified but server is not reachable.", flush=True)
            raise SystemExit(1)
        return True, client
    if not supports_api:
        return False, None
    client = _get_api_client(api_url, api_key)
    if client is None:
        return False, None
    return True, client


def cmd_nomic(args: argparse.Namespace) -> None:
    """Handle 'nomic' command - dispatch to subcommands."""
    subcommand = getattr(args, "nomic_command", None)
    api_url = getattr(args, "api_url", DEFAULT_API_URL)
    api_key = getattr(args, "api_key", None)
    use_api = bool(getattr(args, "api", False))
    use_local = bool(getattr(args, "local", False))

    if subcommand == "run":
        supports_api = True
        use_api_mode, client = _resolve_api_mode(
            use_api, use_local, api_url, api_key, supports_api=supports_api
        )
        if use_api_mode and client is not None:
            _cmd_run_api(args, client)
        else:
            asyncio.run(_cmd_run(args))
    elif subcommand == "status":
        supports_api = True
        use_api_mode, client = _resolve_api_mode(
            use_api, use_local, api_url, api_key, supports_api=supports_api
        )
        if use_api_mode and client is not None:
            _cmd_status_api(args, client)
        else:
            _cmd_status(args)
    elif subcommand == "history":
        supports_api = False
        _resolve_api_mode(use_api, use_local, api_url, api_key, supports_api=supports_api)
        _cmd_history(args)
    elif subcommand == "resume":
        supports_api = False
        _resolve_api_mode(use_api, use_local, api_url, api_key, supports_api=supports_api)
        asyncio.run(_cmd_resume(args))
    else:
        # Default: show help
        print("\nUsage: aragora nomic <command>")
        print("\nCommands:")
        print("  run                 Run improvement cycles")
        print("  status              Show current loop status")
        print("  history             View cycle history")
        print("  resume <cycle_id>   Resume from a checkpoint")
        print("\nOptions for 'run':")
        print("  --cycles <n>        Number of cycles to run (default: 1)")
        print("  --approval          Require human approval for changes")
        print("  --protected <files> Comma-separated list of protected files")
        print("  --max-files <n>     Max files per cycle (default: 20)")
        print("  --json              Output as JSON")


async def _cmd_run(args: argparse.Namespace) -> None:
    """Run nomic improvement cycles."""
    cycles = getattr(args, "cycles", 1)
    require_approval = getattr(args, "approval", False)
    protected_str = getattr(args, "protected", "")
    max_files = getattr(args, "max_files", 20)
    as_json = getattr(args, "json", False)

    protected_files = [f.strip() for f in protected_str.split(",") if f.strip()]
    # Always protect CLAUDE.md
    if "CLAUDE.md" not in protected_files:
        protected_files.append("CLAUDE.md")

    try:
        from aragora.nomic.loop import NomicLoop

        aragora_path = Path.cwd()

        if not as_json:
            print(f"\nStarting Nomic Loop with {cycles} cycle(s)...")
            print(f"  Protected files: {protected_files}")
            print(f"  Require approval: {require_approval}")
            print(f"  Max files/cycle: {max_files}")
            print()

        def log_fn(msg: str) -> None:
            if not as_json:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

        loop = NomicLoop(
            aragora_path=aragora_path,
            max_cycles=cycles,
            protected_files=protected_files,
            require_human_approval=require_approval,
            max_files_per_cycle=max_files,
            log_fn=log_fn,
        )

        result = await loop.run()

        if as_json:
            print(json.dumps(result, indent=2, default=str))
            return

        print("\n" + "=" * 60)
        print("NOMIC LOOP COMPLETE")
        print("=" * 60)
        print(f"\n  Cycles run:        {result.get('cycles_run', 0)}")
        print(f"  Successful:        {result.get('successful_cycles', 0)}")
        print(f"  Failed:            {result.get('failed_cycles', 0)}")

        # Show individual cycle summaries
        cycle_results = result.get("results", [])
        if cycle_results:
            print("\n  Cycle Details:")
            for i, cycle in enumerate(cycle_results, 1):
                status = "SUCCESS" if cycle.get("success") else "FAILED"
                cycle_id = cycle.get("cycle_id", "unknown")
                print(f"    {i}. [{cycle_id}] {status}")
                if not cycle.get("success"):
                    failed_phase = cycle.get("failed_phase", "unknown")
                    reason = cycle.get("reason", cycle.get("error", "Unknown error"))
                    print(f"       Failed at: {failed_phase}")
                    print(f"       Reason: {reason[:80]}...")

    except ImportError as e:
        print(f"\nError: Failed to import Nomic Loop components: {e}")
        print("Make sure aragora is installed correctly.")
    except Exception as e:
        logger.exception("Nomic loop failed")
        print(f"\nError: {e}")


def _cmd_run_api(args: argparse.Namespace, client: Any) -> None:
    """Run nomic loop via API."""
    cycles = getattr(args, "cycles", 1)
    require_approval = getattr(args, "approval", False)
    as_json = getattr(args, "json", False)

    # Warn about local-only options
    if getattr(args, "protected", ""):
        print("Warning: --protected is ignored in API mode.", flush=True)
    if getattr(args, "max_files", None) not in (None, 20):
        print("Warning: --max-files is ignored in API mode.", flush=True)

    payload = {
        "cycles": cycles,
        "max_cycles": cycles,
        "auto_approve": not require_approval,
    }

    try:
        result = client.nomic.start(payload)
        if as_json:
            print(json.dumps(result, indent=2, default=str))
            return
        print("\nNomic loop started via API.")
        print(f"  Target cycles: {cycles}")
        if require_approval:
            print("  Approval: required")
        else:
            print("  Approval: auto")
        if "pid" in result:
            print(f"  PID: {result.get('pid')}")
    except Exception as e:
        print(f"\nError starting nomic loop via API: {e}")


def _cmd_status(args: argparse.Namespace) -> None:
    """Show current nomic loop status."""
    as_json = getattr(args, "json", False)

    try:
        from aragora.nomic.cycle_store import get_cycle_store
        from aragora.persistence.db_config import get_nomic_dir

        nomic_dir = get_nomic_dir()
        store = get_cycle_store()

        # Get recent cycles for stats
        recent = store.get_recent_cycles(10)
        successful = [c for c in recent if c.success]
        failed = [c for c in recent if not c.success]

        # Check for checkpoints
        checkpoint_dir = nomic_dir / "checkpoints"
        checkpoints: list[dict[str, Any]] = []
        if checkpoint_dir.exists():
            for cp_file in sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True)[:5]:
                try:
                    cp_data = json.loads(cp_file.read_text())
                    checkpoints.append(
                        {
                            "cycle_id": cp_data.get("cycle_id"),
                            "timestamp": cp_data.get("timestamp"),
                            "file": str(cp_file),
                        }
                    )
                except (OSError, ValueError, KeyError):
                    logger.debug("Failed to read checkpoint file %s", cp_file, exc_info=True)

        status_data = {
            "nomic_dir": str(nomic_dir),
            "recent_cycles": len(recent),
            "recent_successful": len(successful),
            "recent_failed": len(failed),
            "success_rate": len(successful) / len(recent) if recent else 0,
            "checkpoints": checkpoints,
        }

        if as_json:
            print(json.dumps(status_data, indent=2, default=str))
            return

        print("\n" + "=" * 60)
        print("NOMIC LOOP STATUS")
        print("=" * 60)

        print(f"\n  Data directory:    {nomic_dir}")
        print("\n  Recent cycles (last 10):")
        print(f"    Total:           {len(recent)}")
        print(f"    Successful:      {len(successful)}")
        print(f"    Failed:          {len(failed)}")
        if recent:
            print(f"    Success rate:    {status_data['success_rate']:.1%}")

        if checkpoints:
            print("\n  Available checkpoints:")
            for cp in checkpoints[:3]:
                ts = datetime.fromtimestamp(cp["timestamp"]).strftime("%Y-%m-%d %H:%M")
                print(f"    - {cp['cycle_id']} ({ts})")
        else:
            print("\n  No checkpoints available.")

        # Pattern stats
        patterns = store.get_pattern_statistics()
        if patterns:
            print("\n  Pattern statistics:")
            for pattern_type, stats in list(patterns.items())[:5]:
                success_rate = stats.get("success_rate", 0)
                count = stats.get("success_count", 0) + stats.get("fail_count", 0)
                print(f"    {pattern_type}: {success_rate:.0%} ({count} attempts)")

    except Exception as e:
        print(f"\nError getting status: {e}")


def _cmd_status_api(args: argparse.Namespace, client: Any) -> None:
    """Show current nomic loop status via API."""
    as_json = getattr(args, "json", False)
    try:
        state = client.nomic.state()
        health = client.nomic.health()
        metrics = client.nomic.metrics()
        payload = {
            "state": state,
            "health": health,
            "metrics": metrics,
        }
        if as_json:
            print(json.dumps(payload, indent=2, default=str))
            return

        print("\n" + "=" * 60)
        print("NOMIC LOOP STATUS (API)")
        print("=" * 60)
        print(f"\n  Running:          {state.get('running', False)}")
        print(f"  Cycle:            {state.get('cycle', 0)}")
        print(f"  Phase:            {state.get('phase', 'unknown')}")
        print(f"  Paused:           {state.get('paused', False)}")
        if state.get("started_at"):
            print(f"  Started at:       {state.get('started_at')}")

        print("\n  Health:")
        print(f"    Status:         {health.get('status', 'unknown')}")
        if health.get("warnings"):
            for warning in health.get("warnings", [])[:3]:
                print(f"    Warning:        {warning}")

        stuck = metrics.get("stuck_detection", {}) if isinstance(metrics, dict) else {}
        if stuck:
            print("\n  Stuck detection:")
            print(f"    Is stuck:       {stuck.get('is_stuck', False)}")
            if stuck.get("phase"):
                print(f"    Phase:          {stuck.get('phase')}")
            if stuck.get("idle_seconds") is not None:
                print(f"    Idle seconds:   {stuck.get('idle_seconds')}")
    except Exception as e:
        print(f"\nError getting status via API: {e}")


def _cmd_history(args: argparse.Namespace) -> None:
    """View cycle history."""
    limit = getattr(args, "limit", 20)
    as_json = getattr(args, "json", False)

    try:
        from aragora.nomic.cycle_store import get_cycle_store

        store = get_cycle_store()
        cycles = store.get_recent_cycles(limit)

        if as_json:
            cycle_dicts = [c.to_dict() for c in cycles]
            print(json.dumps(cycle_dicts, indent=2, default=str))
            return

        print("\n" + "=" * 60)
        print("NOMIC CYCLE HISTORY")
        print("=" * 60)

        if not cycles:
            print("\n  No cycles recorded yet.")
            print("  Run 'aragora nomic run' to start improvement cycles.")
            return

        print(f"\n  Showing {len(cycles)} most recent cycles:\n")

        for cycle in cycles:
            # Format timestamp
            started = datetime.fromtimestamp(cycle.started_at)
            started_str = started.strftime("%Y-%m-%d %H:%M:%S")

            # Status indicator
            status = "SUCCESS" if cycle.success else "FAILED"
            status_icon = "+" if cycle.success else "-"

            print(f"  [{status_icon}] {cycle.cycle_id} ({started_str}) - {status}")

            # Duration
            if cycle.duration_seconds:
                mins = int(cycle.duration_seconds // 60)
                secs = int(cycle.duration_seconds % 60)
                print(f"      Duration: {mins}m {secs}s")

            # Phases completed
            if cycle.phases_completed:
                print(f"      Phases: {', '.join(cycle.phases_completed)}")

            # Files modified
            if cycle.files_modified or cycle.files_created:
                total = len(cycle.files_modified) + len(cycle.files_created)
                print(f"      Files changed: {total}")

            # Tests
            if cycle.tests_passed or cycle.tests_failed:
                print(f"      Tests: {cycle.tests_passed} passed, {cycle.tests_failed} failed")

            # Error if failed
            if not cycle.success and cycle.error_message:
                print(f"      Error: {cycle.error_message[:60]}...")

            print()

    except Exception as e:
        print(f"\nError getting history: {e}")


async def _cmd_resume(args: argparse.Namespace) -> None:
    """Resume from a checkpoint."""
    cycle_id = getattr(args, "cycle_id", None)
    as_json = getattr(args, "json", False)

    if not cycle_id:
        print("\nError: cycle_id is required")
        print("Usage: aragora nomic resume <cycle_id>")
        return

    try:
        from aragora.nomic.loop import NomicLoop
        from aragora.persistence.db_config import get_nomic_dir

        checkpoint_dir = get_nomic_dir() / "checkpoints"
        checkpoint_file = checkpoint_dir / f"checkpoint_{cycle_id}.json"

        if not checkpoint_file.exists():
            print(f"\nError: Checkpoint not found for cycle {cycle_id}")
            print(f"  Looked in: {checkpoint_file}")
            return

        checkpoint = json.loads(checkpoint_file.read_text())

        if not as_json:
            print(f"\nResuming from checkpoint: {cycle_id}")
            print(f"  Original path: {checkpoint.get('aragora_path')}")
            print(f"  Last phase: {checkpoint.get('context', {}).get('current_phase', 'unknown')}")
            print()

        loop = NomicLoop(
            aragora_path=Path(checkpoint.get("aragora_path", ".")),
            max_cycles=1,
        )
        loop.restore_from_checkpoint(checkpoint)

        # Continue running from current state
        result = await loop.run_cycle()

        if as_json:
            print(json.dumps(result, indent=2, default=str))
            return

        status = "SUCCESS" if result.get("success") else "FAILED"
        print(f"\nResume completed: {status}")
        if not result.get("success"):
            print(f"  Error: {result.get('error', 'Unknown')}")

    except Exception as e:
        print(f"\nError resuming: {e}")


def add_nomic_parser(subparsers: Any) -> None:
    """Add nomic subparser to CLI."""
    np = subparsers.add_parser(
        "nomic",
        help="Nomic loop self-improvement commands",
        description="Run and manage autonomous self-improvement cycles",
    )
    np.set_defaults(func=cmd_nomic)
    run_mode = np.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--api",
        action="store_true",
        help="Use API server for supported subcommands",
    )
    run_mode.add_argument(
        "--local",
        action="store_true",
        help="Force local mode (offline/air-gapped)",
    )
    np.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    np.add_argument(
        "--api-key",
        default=None,
        help="API key for server authentication (default: ARAGORA_API_KEY)",
    )

    np_sub = np.add_subparsers(dest="nomic_command")

    # Run
    run_p = np_sub.add_parser("run", help="Run improvement cycles")
    run_p.add_argument("--cycles", type=int, default=1, help="Number of cycles to run")
    run_p.add_argument("--approval", action="store_true", help="Require human approval")
    run_p.add_argument("--protected", default="", help="Comma-separated protected files")
    run_p.add_argument("--max-files", type=int, default=20, help="Max files per cycle")
    run_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Status
    status_p = np_sub.add_parser("status", help="Show loop status")
    status_p.add_argument("--json", action="store_true", help="Output as JSON")

    # History
    history_p = np_sub.add_parser("history", help="View cycle history")
    history_p.add_argument("--limit", type=int, default=20, help="Number of cycles to show")
    history_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Resume
    resume_p = np_sub.add_parser("resume", help="Resume from checkpoint")
    resume_p.add_argument("cycle_id", help="Cycle ID to resume")
    resume_p.add_argument("--json", action="store_true", help="Output as JSON")
