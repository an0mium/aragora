#!/usr/bin/env python3
"""
Run the Aragora DevOps Agent.

Autonomous agent that handles repository operations through
policy-controlled execution. Every action is audited.

Usage:
    # Review open PRs (dry run)
    python scripts/run_devops_agent.py --repo an0mium/aragora --task review-prs --dry-run

    # Triage issues
    python scripts/run_devops_agent.py --repo an0mium/aragora --task triage-issues

    # Health check
    python scripts/run_devops_agent.py --repo an0mium/aragora --task health-check

    # Watch mode (polls every 5 minutes)
    python scripts/run_devops_agent.py --repo an0mium/aragora --mode watch

    # Prepare release
    python scripts/run_devops_agent.py --repo an0mium/aragora --task prepare-release

Environment variables:
    ARAGORA_DEVOPS_REPO         Default repo (owner/repo)
    ARAGORA_DEVOPS_AGENTS       Comma-separated agent list
    ARAGORA_DEVOPS_DRY_RUN      Set to "true" for dry run mode
    GITHUB_TOKEN                GitHub auth token (or use gh auth login)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from aragora.agents.devops.agent import (
    DevOpsAgent,
    DevOpsAgentConfig,
    DevOpsTask,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aragora DevOps Agent — autonomous repo operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --repo an0mium/aragora --task health-check
  %(prog)s --repo an0mium/aragora --task review-prs --dry-run
  %(prog)s --repo an0mium/aragora --mode watch --poll-interval 600
""",
    )
    parser.add_argument("--repo", required=True, help="GitHub repository (owner/repo format)")
    parser.add_argument(
        "--task",
        choices=[t.value for t in DevOpsTask],
        help="Specific task to run",
    )
    parser.add_argument(
        "--mode",
        choices=["once", "watch"],
        default="once",
        help="Execution mode (default: once)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )
    parser.add_argument(
        "--allow-destructive",
        action="store_true",
        help="Allow destructive operations (publish, tag, merge)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Seconds between polls in watch mode (default: 300)",
    )
    parser.add_argument(
        "--agents",
        default="anthropic-api,openai-api",
        help="Agents for code review (default: anthropic-api,openai-api)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default=None,
        help="Path to write audit log JSON",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = DevOpsAgentConfig(
        repo=args.repo,
        dry_run=args.dry_run,
        allow_destructive=args.allow_destructive,
        poll_interval=args.poll_interval,
        review_agents=args.agents,
    )

    agent = DevOpsAgent(config=config)

    if args.mode == "watch":
        tasks = None
        if args.task:
            tasks = [DevOpsTask(args.task)]
        agent.watch(tasks=tasks)
        return 0

    if not args.task:
        parser.error("--task is required in 'once' mode")

    task = DevOpsTask(args.task)
    result = agent.run_task(task)

    if args.json_output:
        print(
            json.dumps(
                {
                    "task": result.task,
                    "success": result.success,
                    "items_processed": result.items_processed,
                    "items_skipped": result.items_skipped,
                    "errors": result.errors,
                    "details": result.details,
                    "duration_seconds": result.duration_seconds,
                },
                indent=2,
            )
        )
    else:
        status = "OK" if result.success else "FAILED"
        print(f"\n{'=' * 50}")
        print(f"Task: {result.task} [{status}]")
        print(f"Processed: {result.items_processed}  Skipped: {result.items_skipped}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        if result.errors:
            print(f"\nErrors:")
            for err in result.errors:
                print(f"  - {err}")
        if result.details:
            print(f"\nDetails:")
            for detail in result.details:
                print(f"  {json.dumps(detail)}")
        print(f"{'=' * 50}")

    # Write audit log
    if args.audit_log:
        with open(args.audit_log, "w") as f:
            json.dump(agent.export_audit_log(), f, indent=2)
        print(f"\nAudit log written to {args.audit_log}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
