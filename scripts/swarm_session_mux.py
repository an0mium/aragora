from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from aragora.swarm import session_mux


def _repo_root() -> Path:
    return session_mux.resolve_repo_root(Path.cwd())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tmux-backed session transport for Aragora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch", help="launch a named session under tmux")
    launch.add_argument("--name", required=True)
    launch.add_argument("--cmd", required=True)

    list_parser = subparsers.add_parser("list", help="list registered sessions")
    list_parser.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status", help="show status for one session")
    status.add_argument("--name", required=True)
    status.add_argument("--json", action="store_true")

    send = subparsers.add_parser("send", help="send text or file contents to a session")
    send.add_argument("--name", required=True)
    group = send.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--file", dest="file_path")

    capture = subparsers.add_parser("capture", help="capture output since the last prompt marker")
    capture.add_argument("--name", required=True)
    capture.add_argument("--tail", type=int, default=200)

    tail = subparsers.add_parser("tail", help="tail recent output for a session")
    tail.add_argument("--name", required=True)
    tail.add_argument("--lines", type=int, default=200)
    return parser


def _print_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _print_status_text(status: dict[str, object]) -> None:
    line = (
        f"{status['name']}: running={status['running']} "
        f"tmux={status['tmux_target']} branch={status.get('branch') or '-'} "
        f"worktree={status.get('worktree_path') or '-'}"
    )
    sys.stdout.write(f"{line}\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()

    try:
        if args.command == "launch":
            payload = session_mux.launch_session(repo_root, name=args.name, command=args.cmd)
            _print_json(payload)
            return 0
        if args.command == "list":
            payload = session_mux.list_sessions(repo_root)
            if args.json:
                _print_json(payload)
            else:
                for item in payload:
                    _print_status_text(item)
            return 0
        if args.command == "status":
            payload = session_mux.session_status(repo_root, name=args.name)
            if args.json:
                _print_json(payload)
            else:
                _print_status_text(payload)
            return 0
        if args.command == "send":
            file_path = Path(args.file_path).resolve() if args.file_path else None
            payload = session_mux.send_prompt(
                repo_root,
                name=args.name,
                text=args.text,
                file_path=file_path,
            )
            _print_json(payload)
            return 0
        if args.command == "capture":
            text = session_mux.capture_output(repo_root, name=args.name, tail_lines=args.tail)
            sys.stdout.write(f"{text}\n" if text else "")
            return 0
        if args.command == "tail":
            text = session_mux.tail_output(repo_root, name=args.name, line_count=args.lines)
            sys.stdout.write(f"{text}\n" if text else "")
            return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
