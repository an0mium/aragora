#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aragora.swarm.agent_bridge import AgentBridgeBroker
from aragora.swarm.agent_bridge import BridgeSession
from aragora.swarm.agent_bridge import SessionRegistry
from aragora.swarm.agent_bridge import TransportLaunchError
from aragora.swarm.agent_bridge import TransportNotAvailableError
from aragora.swarm.agent_bridge import TransportResumeError
from aragora.swarm.agent_bridge.harnesses import create_transport
from aragora.swarm.agent_bridge.types import RunStatus

BrokerFactory = Callable[[Path], AgentBridgeBroker]
TransportFactory = Callable[..., object]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend-only agent bridge broker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start-run", help="Create a backend bridge run")
    task_group = start.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task")
    task_group.add_argument("--task-file")
    start.add_argument("--actor", action="append", required=True, help="role:harness:model")
    start.add_argument("--run-id")
    start.add_argument("--next-actor")
    start.add_argument("--base", default="main")
    start.add_argument("--worktree-path", default=str(Path.cwd()))
    start.add_argument("--worktree-agent-slug", default="codex")
    start.add_argument("--json", action="store_true")

    dispatch = subparsers.add_parser("dispatch-turn", help="Dispatch one brokered turn")
    dispatch.add_argument("--run-id", required=True)
    dispatch.add_argument("--role", required=True)
    prompt_group = dispatch.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file")
    dispatch.add_argument("--json", action="store_true")

    show = subparsers.add_parser("show-run", help="Show run state and role sessions")
    show.add_argument("--run-id", required=True)
    show.add_argument("--json", action="store_true")

    list_runs = subparsers.add_parser("list-runs", help="List all bridge runs")
    list_runs.add_argument("--status", choices=["running", "awaiting_human", "completed", "failed"])
    list_runs.add_argument("--json", action="store_true")

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    broker_factory: BrokerFactory = AgentBridgeBroker,
    transport_factory: TransportFactory = create_transport,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 1

    repo_root = Path(__file__).resolve().parents[1]
    broker = broker_factory(repo_root)

    try:
        if args.command == "start-run":
            task = args.task or Path(args.task_file).read_text(encoding="utf-8")
            sessions = _parse_actors(args.actor)
            _preflight_harnesses(sessions, repo_root=repo_root, transport_factory=transport_factory)
            run = broker.start_run(
                task=task,
                sessions=sessions.sessions,
                next_actor=args.next_actor,
                run_id=args.run_id,
                worktree_path=args.worktree_path,
                worktree_agent_slug=args.worktree_agent_slug,
            )
            sessions.run_id = run.run_id
            sessions.updated_at = run.updated_at
            _emit(
                {"run": run.to_dict(), "sessions": sessions.to_dict()},
                as_json=args.json,
            )
            return 0

        if args.command == "dispatch-turn":
            prompt = args.prompt or Path(args.prompt_file).read_text(encoding="utf-8")
            record = broker.dispatch_turn(run_id=args.run_id, role=args.role, prompt=prompt)
            _emit(record.to_dict(), as_json=args.json)
            if record.event_type in {"footer_missing", "footer_malformed"} and bool(
                record.payload.get("repair_exhausted")
            ):
                return 4
            return 0

        if args.command == "show-run":
            payload = {
                "run": broker.load_run(args.run_id).to_dict(),
                "sessions": broker.load_sessions(args.run_id).to_dict(),
                "events": [event.to_dict() for event in broker.load_events(args.run_id)],
            }
            _emit(payload, as_json=args.json)
            return 0

        if args.command == "list-runs":
            status: RunStatus | None = args.status
            runs = broker.list_runs(status=status)
            _emit({"runs": [run.to_dict() for run in runs]}, as_json=args.json)
            return 0
    except ValueError as exc:
        _emit_error(str(exc))
        return 1
    except KeyError as exc:
        _emit_error(str(exc))
        return 2
    except (TransportLaunchError, TransportResumeError) as exc:
        _emit_error(str(exc))
        return 3
    except TransportNotAvailableError as exc:
        _emit_error(str(exc))
        return 6
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        _emit_error(str(exc))
        return 5

    return 1


def _parse_actors(items: list[str]) -> SessionRegistry:
    sessions: dict[str, BridgeSession] = {}
    for item in items:
        role, harness, model = _parse_actor(item)
        harness_options = {"auto": "low"} if harness == "droid" else {}
        sessions[role] = BridgeSession(
            role=role,
            harness=harness,
            model=model,
            session_id=None,
            worktree_agent_slug=None,
            worktree_path=None,
            branch=None,
            session_status="not_started",
            started_at=None,
            last_turn_index=0,
            last_completed_at=None,
            harness_options=harness_options,
        )
    return SessionRegistry(run_id="<pending>", updated_at="", sessions=sessions)


def _parse_actor(item: str) -> tuple[str, str, str]:
    parts = item.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(f"Invalid actor spec '{item}'. Use role:harness:model")
    role = parts[0].strip()
    harness = parts[1].strip()
    model = parts[2].strip() if len(parts) == 3 else ""
    if not role or not harness:
        raise ValueError(f"Invalid actor spec '{item}'. Use role:harness:model")
    return role, harness, model


def _preflight_harnesses(
    sessions: SessionRegistry,
    *,
    repo_root: Path,
    transport_factory: TransportFactory,
) -> None:
    harnesses = {session.harness for session in sessions.sessions.values()}
    for harness in sorted(harnesses):
        transport = transport_factory(
            harness,
            cwd=repo_root,
            model=None,
            harness_options={},
        )
        if hasattr(transport, "healthcheck") and not transport.healthcheck():
            raise TransportNotAvailableError(f"{harness} is not available")


def _emit(payload: object, *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")


def _emit_error(message: str) -> None:
    sys.stderr.write(message + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
