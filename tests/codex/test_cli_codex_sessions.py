"""CLI surface tests for ``aragora codex sessions {list,show}``."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from aragora.cli.commands import codex_sessions as cli
from aragora.cli.parser import build_parser


def _args(**kwargs) -> argparse.Namespace:  # type: ignore[no-untyped-def]
    base = {
        "codex_home": None,
        "json": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


# -- list ---------------------------------------------------------------------


def test_cli_codex_parent_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(["codex"])

    assert args.func(args) == 2
    out = capsys.readouterr().out
    assert "Surface Codex Desktop sessions/threads" in out
    assert "sessions" in out


def test_cli_codex_sessions_parent_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(["codex", "sessions"])

    assert args.func(args) == 2
    out = capsys.readouterr().out
    assert "List, brief, summarize, or tail Codex Desktop sessions" in out
    assert "list" in out
    assert "brief" in out
    assert "show" in out


def test_cli_list_table_output(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="4h", include_archived=False, limit=50))
    out = capsys.readouterr().out
    assert rc == 0
    assert fake_codex_home.recent_thread_id[:12] in out
    assert "AGO" in out  # table header
    assert "TITLE" in out
    # archived excluded by default
    assert fake_codex_home.archived_thread_id[:12] not in out


def test_cli_list_json_output(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="4h", include_archived=False, limit=50, json=True))
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["schema"] == "aragora-codex-sessions-list/1.0"
    assert payload["since"] == "4h"
    assert payload["since_seconds"] == 14400
    assert payload["include_archived"] is False
    assert payload["limit"] == 50
    assert payload["count"] == len(payload["threads"])
    ids = [row["id"] for row in payload["threads"]]
    assert fake_codex_home.recent_thread_id in ids


def test_cli_list_json_redacts_metadata_fields(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="4h", include_archived=False, limit=50, json=True))
    out = capsys.readouterr().out
    assert rc == 0
    assert "ghp_FAKELEAK1234567890ABCD" not in out
    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in out
    assert "[REDACTED]" in out


def test_cli_list_json_bounds_prompt_and_title_excerpts(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    long_title = "title-" + ("A" * 300)
    long_message = "message-" + ("B" * 600)
    with sqlite3.connect(fake_codex_home.home / "state_5.sqlite") as conn:
        conn.execute(
            "UPDATE threads SET title = ?, first_user_message = ? WHERE id = ?",
            (long_title, long_message, fake_codex_home.recent_thread_id),
        )

    rc = cli.cmd_codex_sessions_list(_args(since="4h", include_archived=False, limit=50, json=True))
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    thread = next(
        row for row in payload["threads"] if row["id"] == fake_codex_home.recent_thread_id
    )
    assert thread["title"].endswith("…")
    assert thread["first_user_message"].endswith("…")
    assert len(thread["title"]) <= 160
    assert len(thread["first_user_message"]) <= 240
    assert long_title not in out
    assert long_message not in out


def test_cli_list_redacts_titles(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="4h", include_archived=False, limit=50))
    out = capsys.readouterr().out
    assert rc == 0
    assert "sk-proj-FAKE-LEAK-XYZ" not in out
    assert "ghp_FAKELEAK" not in out
    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in out


def test_cli_list_bad_since(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="nonsense", include_archived=False, limit=50))
    assert rc == 2
    assert "invalid duration" in capsys.readouterr().err


def test_cli_list_missing_db_names_user_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.cmd_codex_sessions_list(
        _args(codex_home=str(tmp_path / "missing"), since="4h", include_archived=False, limit=50)
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "--codex-home <path>" in err
    assert "ARAGORA_CODEX_HOME" in err
    assert "CodexDesktopPaths" not in err


def test_cli_list_rejects_negative_limit(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="4h", include_archived=False, limit=-1))
    assert rc == 2
    assert "--limit must be >= 0" in capsys.readouterr().err


# -- show ---------------------------------------------------------------------


def test_cli_show_summary_default(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=fake_codex_home.recent_thread_id,
            full=False,
            out="",
            max_events=2000,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Rollout:" in out
    assert "Events:" in out
    assert "agent_message" in out


def test_cli_show_json_summary(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=fake_codex_home.recent_thread_id,
            full=False,
            out="",
            max_events=2000,
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["thread"]["id"] == fake_codex_home.recent_thread_id
    assert "event_type_counts" in payload["summary"]


def test_cli_show_json_redacts_thread_metadata(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=fake_codex_home.secret_titled_thread_id,
            full=False,
            out="",
            max_events=2000,
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "ghp_FAKELEAK1234567890ABCD" not in out
    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in out
    assert "[REDACTED]" in out


def test_cli_show_text_redacts_thread_metadata(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=fake_codex_home.secret_titled_thread_id,
            full=False,
            out="",
            max_events=2000,
            json=False,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "ghp_FAKELEAK1234567890ABCD" not in out
    assert "sk-or-v1-abcdefghijklmnopqrstuvwxyz" not in out
    assert "[REDACTED]" in out


def test_cli_show_full_writes_to_file_by_default(
    fake_codex_home,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    # Force --full output under a tmp cwd so we don't pollute the repo's .aragora/.
    monkeypatch.chdir(tmp_path)
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=fake_codex_home.recent_thread_id,
            full=True,
            out="",
            max_events=2000,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    expected = tmp_path / cli.DEFAULT_OUTPUT_ROOT / f"{fake_codex_home.recent_thread_id}.jsonl"
    assert expected.exists()
    # CLI emits the destination as it was constructed (relative when DEFAULT_OUTPUT_ROOT is relative).
    assert "wrote" in out
    assert f"{fake_codex_home.recent_thread_id}.jsonl" in out
    content = expected.read_text(encoding="utf-8")
    assert "sk-proj-FAKE-LEAK-12345" not in content
    assert "ghp_FAKELEAK12345678901234" not in content
    # Each line must be valid JSON.
    for line in content.splitlines():
        json.loads(line)


def test_cli_show_full_to_stdout(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=fake_codex_home.recent_thread_id,
            full=True,
            out="-",
            max_events=2000,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "sk-proj-FAKE-LEAK-12345" not in out
    assert "[REDACTED]" in out


def test_cli_show_full_rejects_out_path_inside_codex_home(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    forbidden = fake_codex_home.home / "exports" / "transcript.jsonl"
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=fake_codex_home.recent_thread_id,
            full=True,
            out=str(forbidden),
            max_events=2000,
        )
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing to write --full output inside Codex Desktop home" in err
    assert not forbidden.exists()


def test_cli_show_resolves_rollout_path(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_show(
        _args(
            target=str(fake_codex_home.recent_rollout),
            full=False,
            out="",
            max_events=2000,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Rollout:" in out


def test_cli_show_unknown_target(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_show(
        _args(
            target="ffffffffabcd",
            full=False,
            out="",
            max_events=2000,
        )
    )
    assert rc == 1
    assert "could not resolve" in capsys.readouterr().err


# -- brief --------------------------------------------------------------------


def _append_rollout_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def test_cli_brief_json_redacts_raw_transcript(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:54:00.000Z",
            "type": "agent_message",
            "payload": {
                "role": "user",
                "content": "Tell Codex B to watch #7283 with sk-proj-FAKE-BRIEF-SECRET",
            },
        },
    )

    rc = cli.cmd_codex_sessions_brief(
        _args(
            since="4h",
            include_archived=False,
            limit=50,
            include_last_turns=0,
            group_by=None,
            session=None,
            repo_root=None,
            json=True,
        )
    )
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert payload["schema"] == "aragora-codex-sessions-brief/1.0"
    assert payload["count"] >= 1
    assert "Tell Codex B" not in out
    assert "sk-proj-FAKE-BRIEF-SECRET" not in out
    assert any(7283 in row["pr_mentions"] for row in payload["briefs"])


def test_cli_brief_include_last_turns_stays_summary_only(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:54:00.000Z",
            "type": "agent_message",
            "payload": {
                "role": "assistant",
                "content": "I reviewed #7285 and saw Bearer ghp_FAKELEAK12345678901234",
            },
        },
    )

    rc = cli.cmd_codex_sessions_brief(
        _args(
            since="4h",
            include_archived=False,
            limit=50,
            include_last_turns=2,
            group_by=None,
            session=fake_codex_home.recent_thread_id[:13],
            repo_root=None,
            json=True,
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["count"] == 1
    assert payload["briefs"][0]["recent_turns"]
    assert "I reviewed #7285" not in out
    assert "ghp_FAKELEAK12345678901234" not in out


def test_cli_brief_groups_by_branch(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_brief(
        _args(
            since="4h",
            include_archived=False,
            limit=50,
            include_last_turns=0,
            group_by="branch",
            session=None,
            repo_root=None,
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["group_by"] == "branch"
    assert "main" in payload["groups"]
    assert fake_codex_home.recent_thread_id in payload["groups"]["main"]


def test_cli_brief_unknown_session_is_paste_needed(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_brief(
        _args(
            since="4h",
            include_archived=False,
            limit=50,
            include_last_turns=0,
            group_by=None,
            session="ffffffffabcd",
            repo_root=None,
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["count"] == 1
    assert payload["briefs"][0]["router"]["category"] == "paste-needed"
    assert "Paste the last 2-4 turns" in payload["briefs"][0]["router"]["recommended_next_prompt"]


def test_cli_brief_compact_awaiting_prompts_filters_and_redacts(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    _append_rollout_event(
        fake_codex_home.recent_rollout,
        {
            "timestamp": "2026-05-16T13:55:00.000Z",
            "type": "agent_message",
            "payload": {
                "role": "assistant",
                "content": "Finished review of #7286 with sk-proj-FAKE-COMPACT-SECRET.",
            },
        },
    )

    rc = cli.cmd_codex_sessions_brief(
        _args(
            since="4h",
            include_archived=False,
            limit=50,
            include_last_turns=0,
            group_by=None,
            session=fake_codex_home.recent_thread_id[:13],
            repo_root=None,
            compact=True,
            awaiting_prompts=True,
            json=True,
        )
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["compact"] is True
    assert payload["awaiting_prompts"] is True
    assert payload["count"] == 1
    row = payload["briefs"][0]
    assert set(row) >= {
        "id",
        "title_summary",
        "prompt_needed",
        "prompt_needed_reason",
        "route",
        "conflict_risk",
        "recommended_next_prompt",
    }
    assert row["prompt_needed"] is True
    assert row["prompt_needed_reason"] == "assistant_final_recent"
    assert "Finished review" not in out
    assert "sk-proj-FAKE-COMPACT-SECRET" not in out


def test_collect_repo_context_reads_active_lane_records_from_registry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    registry = repo / ".aragora" / "agent-bridge" / "lanes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "released",
                    "owner_session": "codex-old",
                    "status": "released",
                    "branch": "old",
                },
                {
                    "lane_id": "Q04-cross-agent-collision-control",
                    "owner_session": "codex-q04",
                    "status": "active",
                    "branch": "codex/cross-agent-collision-control-20260517",
                },
            ]
        ),
        encoding="utf-8",
    )

    context = cli._collect_repo_context(str(repo))

    assert context["active_lanes"] == ["Q04-cross-agent-collision-control"]
    assert context["active_lane_records"] == [
        {
            "lane_id": "Q04-cross-agent-collision-control",
            "owner_session": "codex-q04",
            "status": "active",
            "branch": "codex/cross-agent-collision-control-20260517",
        }
    ]


def test_cli_tail_rejects_non_positive_interval(
    fake_codex_home, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_tail(_args(since="4h", interval=0, from_start=False))
    assert rc == 2
    assert "--interval must be > 0" in capsys.readouterr().err
