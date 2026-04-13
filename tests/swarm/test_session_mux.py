from __future__ import annotations

import json
from pathlib import Path

from aragora.swarm import session_mux as mod


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = mod.SessionMuxRegistry(tmp_path)
    record = mod.SessionRecord(
        name="codex-1",
        tmux_session="codex-1",
        tmux_window="0",
        tmux_pane="0",
        launcher_command="./scripts/codex_session.sh --agent codex-1 -- zsh -l",
        started_at="2026-04-13T18:00:00Z",
        log_path=str(tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log"),
        worktree_path="/tmp/worktree",
        branch="codex/demo",
    )

    registry.upsert(record)

    loaded = registry.get("codex-1")
    assert loaded is not None
    assert loaded.to_dict() == record.to_dict()
    payload = json.loads((tmp_path / ".aragora" / "session_mux" / "registry.json").read_text())
    assert payload["schema_version"] == 1


def test_extract_output_after_marker_uses_latest_matching_boundary() -> None:
    old_marker = mod.prompt_marker("old")
    new_marker = mod.prompt_marker("new")
    text = "\n".join(
        [
            "before",
            old_marker,
            "old output",
            new_marker,
            "line a",
            "line b",
        ]
    )

    assert mod.extract_output_after_marker(text, prompt_id="new") == "line a\nline b"


def test_build_tmux_commands_are_stable(tmp_path: Path) -> None:
    new_session = mod.build_tmux_new_session_cmd(
        tmux_session="codex-1",
        cwd=tmp_path,
        command="./scripts/codex_session.sh --agent codex-1 -- zsh -l",
    )
    pipe_pane = mod.build_tmux_pipe_pane_cmd(
        target="codex-1:0.0",
        log_path=tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log",
    )

    assert new_session == [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "codex-1",
        "-c",
        str(tmp_path),
        "./scripts/codex_session.sh --agent codex-1 -- zsh -l",
    ]
    assert pipe_pane[:5] == ["tmux", "pipe-pane", "-o", "-t", "codex-1:0.0"]
    assert "cat >>" in pipe_pane[5]


def test_refresh_session_record_harvests_codex_metadata(monkeypatch, tmp_path: Path) -> None:
    meta_path = tmp_path / ".worktrees" / "codex-auto" / "codex-1" / ".codex_session_meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps(
            {
                "agent": "codex-1",
                "branch": "codex/branch",
                "worktree_path": "/tmp/codex-worktree",
                "log_path": "/tmp/codex-worktree/.codex_session.log",
                "started_at": "2026-04-13T18:30:00Z",
            }
        ),
        encoding="utf-8",
    )
    record = mod.SessionRecord(
        name="codex-1",
        tmux_session="codex-1",
        tmux_window="0",
        tmux_pane="0",
        launcher_command="codex",
        started_at="2026-04-13T18:00:00Z",
        log_path=str(tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log"),
    )
    monkeypatch.setattr(mod, "_tmux_session_exists", lambda _: False)

    refreshed = mod.refresh_session_record(tmp_path, record)

    assert refreshed.worktree_path == "/tmp/codex-worktree"
    assert refreshed.branch == "codex/branch"
    assert refreshed.launcher_log_path == "/tmp/codex-worktree/.codex_session.log"
    assert refreshed.meta_path == str(meta_path)


def test_capture_output_uses_last_prompt_marker(tmp_path: Path) -> None:
    log_path = tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "\n".join(
            [
                "boot",
                mod.prompt_marker("older"),
                "stale",
                mod.prompt_marker("fresh"),
                "recent line 1",
                "recent line 2",
            ]
        ),
        encoding="utf-8",
    )
    registry = mod.SessionMuxRegistry(tmp_path)
    registry.upsert(
        mod.SessionRecord(
            name="codex-1",
            tmux_session="codex-1",
            tmux_window="0",
            tmux_pane="0",
            launcher_command="codex",
            started_at="2026-04-13T18:00:00Z",
            log_path=str(log_path),
            last_prompt_id="fresh",
        )
    )

    captured = mod.capture_output(tmp_path, name="codex-1", tail_lines=1)

    assert captured == "recent line 2"
