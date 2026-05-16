"""CLI surface tests for ``aragora codex sessions {list,show}``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from aragora.cli.commands import codex_sessions as cli


def _args(**kwargs) -> argparse.Namespace:  # type: ignore[no-untyped-def]
    base = {
        "codex_home": None,
        "json": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


# -- list ---------------------------------------------------------------------


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


def test_cli_list_redacts_titles(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="4h", include_archived=False, limit=50))
    out = capsys.readouterr().out
    assert rc == 0
    assert "sk-proj-FAKE-LEAK-XYZ" not in out
    assert "ghp_FAKELEAK" not in out


def test_cli_list_bad_since(fake_codex_home, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    rc = cli.cmd_codex_sessions_list(_args(since="nonsense", include_archived=False, limit=50))
    assert rc == 2
    assert "invalid duration" in capsys.readouterr().err


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
