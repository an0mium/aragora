#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aragora.swarm.agent_bridge.footer import build_footer_instruction
from aragora.swarm.agent_bridge.harnesses import create_transport
from aragora.swarm.agent_bridge.types import ParsedTurn
from aragora.swarm.agent_bridge.types import utc_now_iso


ROLES = ("claude", "codex", "droid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Opt-in live smoke for current agent-bridge harness launch/resume flows"
    )
    parser.add_argument(
        "--repo", default=str(Path.cwd()), help="Repository root / cwd for harnesses"
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifact output directory (defaults to .aragora/agent_bridge/live_smoke)",
    )
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--droid-model", default=None)
    parser.add_argument(
        "--droid-auto",
        default="low",
        choices=("low", "medium", "high"),
        help="Droid auto mode for the live smoke",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch plan and exit without requiring binaries or launching agents",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON. The live smoke already emits JSON; this is explicit for scripted calls.",
    )
    return parser


def _artifact_path(repo_root: Path, override: str | None) -> Path:
    return (
        Path(override).resolve()
        if override
        else repo_root / ".aragora" / "agent_bridge" / "live_smoke"
    )


def _artifact_dir(repo_root: Path, override: str | None) -> Path:
    path = _artifact_path(repo_root, override)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require(binary: str) -> None:
    if shutil.which(binary):
        return
    raise SystemExit(f"Required binary not found on PATH: {binary}")


def _prompt(
    instructions: str,
    *,
    summary: str,
    next_actor: str | None = None,
    done: bool = False,
) -> str:
    next_actor_value = next_actor if next_actor is not None else "null"
    done_value = "true" if done else "false"
    return (
        f"{instructions.rstrip()}\n\n"
        "Return a short plain-text response before the footer.\n"
        "Use this footer field guidance:\n"
        f"- summary: {summary}\n"
        f"- next_actor: {next_actor_value}\n"
        "- needs_human: false\n"
        f"- done: {done_value}\n"
        "- artifacts: []\n"
        "- tests_run: []\n\n"
        f"{build_footer_instruction(roles=list(ROLES))}"
    )


def _assert_ok(label: str, parsed: ParsedTurn) -> str:
    if parsed.parse_status != "ok" or parsed.footer is None:
        raise SystemExit(f"{label} footer parse failed: {parsed.to_dict()}")
    body = parsed.body_without_footer.strip()
    if not body:
        raise SystemExit(f"{label} response body was empty")
    return body


def _assert_contains(label: str, body: str, token: str) -> None:
    if token not in body:
        raise SystemExit(f"{label} did not preserve token {token!r}. Body: {body!r}")


def _serialize_result(result: Any) -> dict[str, Any]:
    return {
        "session_id": result.session_id,
        "command": list(result.command),
        "exit_code": result.exit_code,
        "message_text": result.message_text,
        "usage": dict(result.usage),
        "parsed_turn": result.parsed_turn.to_dict(),
    }


def _dry_run_payload(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    required_binaries = list(ROLES)
    binary_paths = {binary: shutil.which(binary) for binary in required_binaries}
    missing_binaries = [binary for binary, path in binary_paths.items() if path is None]
    turns: list[dict[str, str]] = []
    for role in ROLES:
        turns.append({"role": role, "transport_call": "launch", "purpose": "start"})
        turns.append({"role": role, "transport_call": "resume", "purpose": "recall"})
    turns.extend(
        [
            {"role": "codex", "transport_call": "resume", "purpose": "handoff"},
            {"role": "droid", "transport_call": "resume", "purpose": "handoff"},
            {"role": "claude", "transport_call": "resume", "purpose": "final_confirmation"},
        ]
    )
    return {
        "ok": True,
        "dry_run": True,
        "repo_root": str(repo_root),
        "artifact_dir": str(_artifact_path(repo_root, args.artifact_dir)),
        "roles": list(ROLES),
        "required_binaries": required_binaries,
        "binary_paths": binary_paths,
        "missing_binaries": missing_binaries,
        "models": {
            "claude": args.claude_model,
            "codex": args.codex_model,
            "droid": args.droid_model,
        },
        "harness_options": {"droid": {"auto": args.droid_auto}},
        "turn_count": len(turns),
        "turns": turns,
        "side_effects": {
            "artifact_dir_created": False,
            "artifact_written": False,
            "binaries_required": False,
            "agents_launched": False,
        },
    }


def _print_dry_run(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("agent_bridge_live_smoke dry run")
    print(f"repo_root: {payload['repo_root']}")
    print(f"artifact_dir: {payload['artifact_dir']}")
    print(f"roles: {', '.join(payload['roles'])}")
    print(f"turn_count: {payload['turn_count']}")
    if payload["missing_binaries"]:
        print(f"missing_binaries: {', '.join(payload['missing_binaries'])}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo).resolve()
    if args.dry_run:
        _print_dry_run(_dry_run_payload(args, repo_root), as_json=args.json)
        return 0

    artifact_dir = _artifact_dir(repo_root, args.artifact_dir)

    _require("codex")
    _require("claude")
    _require("droid")

    transports = {
        "codex": create_transport(
            "codex",
            cwd=repo_root,
            model=args.codex_model,
            harness_options={},
        ),
        "claude": create_transport(
            "claude",
            cwd=repo_root,
            model=args.claude_model,
            harness_options={},
        ),
        "droid": create_transport(
            "droid",
            cwd=repo_root,
            model=args.droid_model,
            harness_options={"auto": args.droid_auto},
        ),
    }

    shared_token = f"shared-{uuid.uuid4().hex[:10]}"
    recall_tokens = {role: f"{role}-{uuid.uuid4().hex[:8]}" for role in ROLES}

    artifact: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "repo_root": str(repo_root),
        "shared_token": shared_token,
        "roles": list(ROLES),
        "results": {},
    }

    session_ids: dict[str, str] = {}
    for role in ROLES:
        start = transports[role].launch(
            _prompt(
                f"Remember this token for the next turn: {recall_tokens[role]}. Reply with READY.",
                summary=f"{role} ready",
            ),
            allowed_roles=set(ROLES),
        )
        start_body = _assert_ok(f"{role} start", start.parsed_turn)
        _assert_contains(f"{role} start", start_body, "READY")
        session_ids[role] = start.session_id

        recall = transports[role].resume(
            session_ids[role],
            _prompt(
                "What token did I ask you to remember? Reply with the token only.",
                summary=f"{role} recall",
            ),
            allowed_roles=set(ROLES),
        )
        recall_body = _assert_ok(f"{role} recall", recall.parsed_turn)
        _assert_contains(f"{role} recall", recall_body, recall_tokens[role])
        artifact["results"][f"{role}_start"] = _serialize_result(start)
        artifact["results"][f"{role}_recall"] = _serialize_result(recall)

    codex_handoff = transports["codex"].resume(
        session_ids["codex"],
        _prompt(
            f"Create a one-sentence handoff that contains the shared token {shared_token}.",
            summary="codex handoff",
            next_actor="droid",
        ),
        allowed_roles=set(ROLES),
    )
    codex_body = _assert_ok("codex handoff", codex_handoff.parsed_turn)
    _assert_contains("codex handoff", codex_body, shared_token)

    droid_handoff = transports["droid"].resume(
        session_ids["droid"],
        _prompt(
            "A peer agent handed you this sentence:\n"
            f"{codex_body}\n\n"
            f"Repeat the shared token {shared_token} exactly once.",
            summary="droid handoff",
            next_actor="claude",
        ),
        allowed_roles=set(ROLES),
    )
    droid_body = _assert_ok("droid handoff", droid_handoff.parsed_turn)
    _assert_contains("droid handoff", droid_body, shared_token)

    claude_handoff = transports["claude"].resume(
        session_ids["claude"],
        _prompt(
            "Two peer agents are coordinating on a shared token.\n"
            f"Codex said: {codex_body}\n"
            f"Droid said: {droid_body}\n\n"
            f"Confirm the shared token {shared_token} exactly once.",
            summary="claude handoff",
            done=True,
        ),
        allowed_roles=set(ROLES),
    )
    claude_body = _assert_ok("claude handoff", claude_handoff.parsed_turn)
    _assert_contains("claude handoff", claude_body, shared_token)

    artifact["results"]["codex_handoff"] = _serialize_result(codex_handoff)
    artifact["results"]["droid_handoff"] = _serialize_result(droid_handoff)
    artifact["results"]["claude_handoff"] = _serialize_result(claude_handoff)
    artifact["session_ids"] = dict(session_ids)

    artifact_path = artifact_dir / f"live-smoke-{uuid.uuid4().hex[:12]}.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "artifact_path": str(artifact_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
