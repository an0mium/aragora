from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.swarm_session_mux as cli
from aragora.swarm import session_mux


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_cli_uses_direct_session_mux_module() -> None:
    assert cli.session_mux.__name__ == "aragora_swarm_session_mux_direct"


def test_launch_command_creates_registry_entry(monkeypatch, tmp_path: Path, capsys) -> None:
    state = {"exists": False}

    def fake_run(cmd: list[str], **kwargs) -> SimpleNamespace:  # noqa: ANN003
        if cmd[:3] == ["tmux", "has-session", "-t"]:
            return _completed(returncode=0 if state["exists"] else 1)
        if cmd[:2] == ["tmux", "new-session"]:
            state["exists"] = True
            return _completed()
        if cmd[:2] == ["tmux", "list-panes"]:
            return _completed(stdout="0\t0\t/tmp/codex-worktree\n")
        if cmd[:2] == ["tmux", "pipe-pane"]:
            return _completed()
        if cmd[:3] == ["git", "-C", "/tmp/codex-worktree"]:
            return _completed(stdout="codex/demo\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(session_mux.subprocess, "run", fake_run)

    exit_code = cli.main(
        [
            "launch",
            "--name",
            "codex-1",
            "--cmd",
            "./scripts/codex_session.sh --agent codex-1 -- zsh -l",
        ]
    )

    assert exit_code == 0
    payload = capsys.readouterr().out
    assert '"name": "codex-1"' in payload
    record = session_mux.SessionMuxRegistry(tmp_path).get("codex-1")
    assert record is not None
    assert record.branch == "codex/demo"


def test_list_command_prints_json(monkeypatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("OpenAI Codex\n", encoding="utf-8")
    registry = session_mux.SessionMuxRegistry(tmp_path)
    registry.upsert(
        session_mux.SessionRecord(
            name="codex-1",
            tmux_session="codex-1",
            tmux_window="0",
            tmux_pane="0",
            launcher_command="codex",
            started_at="2026-04-13T18:00:00Z",
            log_path=str(log_path),
        )
    )

    def fake_run(cmd: list[str], **kwargs) -> SimpleNamespace:  # noqa: ANN003
        if cmd[:3] == ["tmux", "has-session", "-t"]:
            return _completed(returncode=0)
        if cmd[:2] == ["tmux", "list-panes"]:
            return _completed(stdout="0\t0\t/tmp/codex-worktree\n")
        if cmd[:3] == ["git", "-C", "/tmp/codex-worktree"]:
            return _completed(stdout="codex/demo\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(session_mux.subprocess, "run", fake_run)

    exit_code = cli.main(["list", "--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"name": "codex-1"' in output
    assert '"running": true' in output
    assert '"phase": "ready"' in output


def test_send_command_updates_prompt_marker(monkeypatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log"
    registry = session_mux.SessionMuxRegistry(tmp_path)
    registry.upsert(
        session_mux.SessionRecord(
            name="codex-1",
            tmux_session="codex-1",
            tmux_window="0",
            tmux_pane="0",
            launcher_command="codex",
            started_at="2026-04-13T18:00:00Z",
            log_path=str(log_path),
        )
    )

    def fake_run(cmd: list[str], **kwargs) -> SimpleNamespace:  # noqa: ANN003
        if cmd[:3] == ["tmux", "has-session", "-t"]:
            return _completed(returncode=0)
        if cmd[:2] == ["tmux", "list-panes"]:
            return _completed(stdout="0\t0\t/tmp/codex-worktree\n")
        if cmd[:3] == ["git", "-C", "/tmp/codex-worktree"]:
            return _completed(stdout="codex/demo\n")
        if cmd[:2] == ["tmux", "load-buffer"]:
            assert kwargs["input"] == "hello from mux"
            return _completed()
        if cmd[:2] == ["tmux", "paste-buffer"]:
            return _completed()
        if cmd[:2] == ["tmux", "send-keys"]:
            return _completed()
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(session_mux.subprocess, "run", fake_run)

    exit_code = cli.main(["send", "--name", "codex-1", "--text", "hello from mux"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"last_prompt_id":' in output
    saved = session_mux.SessionMuxRegistry(tmp_path).get("codex-1")
    assert saved is not None
    assert saved.last_prompt_id is not None
    assert saved.last_prompt_at is not None
    assert "ARAGORA_SESSION_MUX_PROMPT" in log_path.read_text(encoding="utf-8")


def test_capture_command_reads_log_since_last_prompt(monkeypatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "\n".join(
            [
                "boot",
                session_mux.prompt_marker("old"),
                "before",
                session_mux.prompt_marker("fresh"),
                "captured line",
            ]
        ),
        encoding="utf-8",
    )
    registry = session_mux.SessionMuxRegistry(tmp_path)
    registry.upsert(
        session_mux.SessionRecord(
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

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)

    exit_code = cli.main(["capture", "--name", "codex-1", "--tail", "20"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "captured line"


def test_status_command_prints_readiness_phase(monkeypatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / ".aragora" / "session_mux" / "logs" / "codex-1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("OpenAI Codex\n[Pasted Content 42 chars]\n", encoding="utf-8")
    registry = session_mux.SessionMuxRegistry(tmp_path)
    registry.upsert(
        session_mux.SessionRecord(
            name="codex-1",
            tmux_session="codex-1",
            tmux_window="0",
            tmux_pane="0",
            launcher_command="codex",
            started_at="2026-04-13T18:00:00Z",
            log_path=str(log_path),
        )
    )

    def fake_run(cmd: list[str], **kwargs) -> SimpleNamespace:  # noqa: ANN003
        if cmd[:3] == ["tmux", "has-session", "-t"]:
            return _completed(returncode=0)
        if cmd[:2] == ["tmux", "list-panes"]:
            return _completed(stdout="0\t0\t/tmp/codex-worktree\n")
        if cmd[:3] == ["git", "-C", "/tmp/codex-worktree"]:
            return _completed(stdout="codex/demo\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(session_mux.subprocess, "run", fake_run)

    exit_code = cli.main(["status", "--name", "codex-1", "--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"phase": "prompt_accepted"' in output
    assert '"prompt_accepted": true' in output
