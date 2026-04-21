from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..broker import AgentBridgeBroker
from ..types import BridgeSession
from ..types import SessionRegistry
from ..types import utc_now_iso


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend-only agent bridge broker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start-run", help="Create a backend bridge run")
    start.add_argument("--role", action="append", required=True, help="role=harness[:model]")
    start.add_argument("--active-role")
    start.add_argument("--run-id")
    start.add_argument("--worktree-path", default=str(Path.cwd()))
    start.add_argument("--worktree-agent-slug", default="codex")

    dispatch = subparsers.add_parser("dispatch-turn", help="Dispatch one brokered turn")
    dispatch.add_argument("--run-id", required=True)
    dispatch.add_argument("--role", required=True)
    dispatch.add_argument("--prompt")
    dispatch.add_argument("--prompt-file")

    show = subparsers.add_parser("show-run", help="Show run state and role sessions")
    show.add_argument("--run-id", required=True)

    subparsers.add_parser("list-runs", help="List all bridge runs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    broker = AgentBridgeBroker(Path.cwd())

    if args.command == "start-run":
        roles = _parse_roles(args.role)
        run = broker.start_run(
            roles=roles.roles,
            active_role=args.active_role,
            run_id=args.run_id,
            worktree_path=args.worktree_path,
            worktree_agent_slug=args.worktree_agent_slug,
        )
        _print_json({"run": run.to_dict(), "sessions": roles.to_dict()})
        return 0

    if args.command == "dispatch-turn":
        prompt = args.prompt
        if args.prompt_file:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        if not prompt:
            raise ValueError("A prompt is required")
        record = broker.dispatch_turn(run_id=args.run_id, role=args.role, prompt=prompt)
        _print_json(record.to_dict())
        return 0

    if args.command == "show-run":
        _print_json(
            {
                "run": broker.load_run(args.run_id).to_dict(),
                "sessions": broker.load_sessions(args.run_id).to_dict(),
                "events": [event.to_dict() for event in broker.load_events(args.run_id)],
            }
        )
        return 0

    if args.command == "list-runs":
        _print_json({"runs": [run.to_dict() for run in broker.list_runs()]})
        return 0

    return 1


def _parse_roles(items: list[str]) -> SessionRegistry:
    roles: dict[str, BridgeSession] = {}
    created_at = utc_now_iso()
    for item in items:
        role, raw_harness = item.split("=", 1)
        harness, _, model = raw_harness.partition(":")
        roles[role] = BridgeSession(
            harness=harness,
            session_id=None,
            created_at=created_at,
            last_turn_at=None,
            harness_options={"model": model} if model else {},
        )
    return SessionRegistry(roles=roles)


def _print_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
