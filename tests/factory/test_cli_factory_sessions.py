from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aragora.cli.commands import factory_sessions as cli
from aragora.cli.parser import build_parser


def _args(**kwargs: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "factory_home": None,
        "repo_root": None,
        "json": False,
        "compact": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_factory_sessions_parser_registers_brief_command() -> None:
    args = build_parser().parse_args(["factory", "sessions", "brief", "--json"])

    assert args.factory_cmd == "sessions"
    assert args.factory_sessions_cmd == "brief"
    assert args.json is True


def test_cli_brief_json_schema_and_redaction(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    factory_home = tmp_path / "factory"
    repo = tmp_path / "repo"
    _write_json(
        factory_home / "sessions-index.json",
        [
            {
                "id": "droid-CLI",
                "cwd": str(tmp_path / "worktree"),
                "branch": "droid/cli-branch",
                "pr_number": 7354,
                "updated_at": int(time.time()),
                "title": "Factory raw title with ghp_FAKELEAK12345678901234",
            }
        ],
    )
    _write_json(repo / ".aragora" / "agent-bridge" / "lanes.json", [])

    rc = cli.cmd_factory_sessions_brief(
        _args(
            factory_home=str(factory_home),
            repo_root=str(repo),
            since="4h",
            limit=50,
            session=None,
            json=True,
            compact=False,
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["schema"] == "aragora-factory-sessions-brief/1.0"
    assert payload["count"] == 1
    assert payload["briefs"][0]["session_id"] == "droid-CLI"
    assert "Factory raw title" not in out
    assert "ghp_FAKELEAK12345678901234" not in out


def test_cli_compact_omits_raw_paths_and_transcript_like_fields(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    factory_home = tmp_path / "factory"
    repo = tmp_path / "repo"
    _write_json(
        factory_home / "sessions-index.json",
        [
            {
                "id": "droid-COMPACT",
                "cwd": str(tmp_path / "worktree"),
                "updated_at": int(time.time()),
            }
        ],
    )
    _write_json(repo / ".aragora" / "agent-bridge" / "lanes.json", [])

    rc = cli.cmd_factory_sessions_brief(
        _args(
            factory_home=str(factory_home),
            repo_root=str(repo),
            since="4h",
            limit=50,
            session=None,
            json=True,
            compact=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["compact"] is True
    row = payload["briefs"][0]
    assert set(row) == {
        "provider",
        "session_id",
        "age",
        "branch",
        "pr_number",
        "route",
        "conflict_risk",
        "prompt_needed",
        "prompt_needed_reason",
        "direct_steering_available",
        "recommended_next_prompt",
    }


def test_cli_unknown_session_returns_paste_needed(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    factory_home = tmp_path / "factory"
    repo = tmp_path / "repo"
    _write_json(factory_home / "sessions-index.json", [])
    _write_json(repo / ".aragora" / "agent-bridge" / "lanes.json", [])

    rc = cli.cmd_factory_sessions_brief(
        _args(
            factory_home=str(factory_home),
            repo_root=str(repo),
            since="4h",
            limit=50,
            session="missing-session",
            json=True,
            compact=False,
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["count"] == 1
    assert payload["briefs"][0]["session_id"] == "missing-session"
    assert payload["briefs"][0]["router"]["category"] == "paste-needed"
