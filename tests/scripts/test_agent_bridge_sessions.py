"""Tests for scripts/agent_bridge_sessions.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_claude_project_slug_uses_canonical_absolute_path():
    import agent_bridge_sessions as mod

    slug = mod._claude_project_slug(Path("/Users/armand/Development/aragora"))
    assert slug == "-Users-armand-Development-aragora"


def test_collect_sessions_merges_tmux_and_claude_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge_sessions as mod

    repo_root = tmp_path / "aragora"
    repo_root.mkdir()
    review_worktree = repo_root / ".worktrees" / "review-6818"
    review_worktree.mkdir(parents=True)
    tmux_dir = tmp_path / "tmux"
    tmux_dir.mkdir()
    claude_projects_root = tmp_path / "claude-projects"
    project_dir = claude_projects_root / "-tmp-pytest-of-root-aragora"
    project_dir.mkdir(parents=True)

    log_file = tmux_dir / "codex-strategic.log"
    log_file.write_text("starting\nPR #5297 opened\n", encoding="utf-8")
    (tmux_dir / "codex-strategic.meta.json").write_text(
        json.dumps(
            {
                "name": "codex-strategic",
                "agent": "codex",
                "started": "2026-04-13T18:00:00Z",
                "log_file": str(log_file),
                "repo_root": str(repo_root),
                "cwd": str(review_worktree),
                "prompt_file": "/tmp/5282.md",
                "has_prompt": True,
            }
        ),
        encoding="utf-8",
    )

    claude_log = project_dir / "1431bd39-6ab6-41e0-9ead-05f91680e1df.jsonl"
    claude_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "2026-04-13T18:01:00Z",
                        "cwd": str(repo_root / ".claude" / "worktrees" / "board"),
                        "gitBranch": "worktree-sessions-1-3-impl",
                        "sessionId": "1431bd39-6ab6-41e0-9ead-05f91680e1df",
                        "message": {
                            "content": [
                                {"type": "text", "text": "Track PR #5294 and #5295 tonight."}
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-13T18:02:00Z",
                        "cwd": str(repo_root / ".claude" / "worktrees" / "board"),
                        "gitBranch": "worktree-sessions-1-3-impl",
                        "sessionId": "1431bd39-6ab6-41e0-9ead-05f91680e1df",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "I’m watching #5294 and #5295; #5295 is closest to merge.",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_tmux_alive", lambda _name: "alive")
    monkeypatch.setattr(mod, "_capture_tmux_summary", lambda _name: "")
    monkeypatch.setattr(
        mod, "_claude_project_slug", lambda _repo_root: "-tmp-pytest-of-root-aragora"
    )

    sessions = mod.collect_sessions(
        repo_root=repo_root,
        tmux_dir=tmux_dir,
        claude_projects_root=claude_projects_root,
        resolve_repo=False,
    )

    assert len(sessions) == 2
    assert {item.source for item in sessions} == {"tmux", "claude_jsonl"}

    tmux_session = next(item for item in sessions if item.source == "tmux")
    assert tmux_session.name == "codex-strategic"
    assert tmux_session.agent == "codex"
    assert tmux_session.status == "alive"
    assert tmux_session.summary == "PR #5297 opened"
    assert tmux_session.prompt_file == "/tmp/5282.md"
    assert tmux_session.cwd == str(review_worktree)

    claude_session = next(item for item in sessions if item.source == "claude_jsonl")
    assert claude_session.name == "claude-1431bd39"
    assert claude_session.branch == "worktree-sessions-1-3-impl"
    assert claude_session.last_role == "assistant"
    assert claude_session.last_user_text == "Track PR #5294 and #5295 tonight."
    assert (
        claude_session.last_assistant_text
        == "I’m watching #5294 and #5295; #5295 is closest to merge."
    )
    assert claude_session.summary == "I’m watching #5294 and #5295; #5295 is closest to merge."


def test_collect_sessions_filters_tmux_sessions_to_matching_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge_sessions as mod

    repo_root = tmp_path / "aragora"
    repo_root.mkdir()
    other_repo_root = tmp_path / "other"
    other_repo_root.mkdir()
    tmux_dir = tmp_path / "tmux"
    tmux_dir.mkdir()

    for name, repo in (("keep", repo_root), ("skip", other_repo_root)):
        log_file = tmux_dir / f"{name}.log"
        log_file.write_text(f"{name}\n", encoding="utf-8")
        (tmux_dir / f"{name}.meta.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "agent": "codex",
                    "started": "2026-04-13T18:00:00Z",
                    "log_file": str(log_file),
                    "repo_root": str(repo),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(mod, "_tmux_alive", lambda _name: "dead")

    sessions = mod.collect_sessions(
        repo_root=repo_root,
        tmux_dir=tmux_dir,
        claude_projects_root=tmp_path / "claude-projects",
        source="tmux",
        resolve_repo=False,
    )

    assert len(sessions) == 1
    assert sessions[0].name == "keep"


def test_extract_recent_claude_turns_skips_tool_only_user_entries(tmp_path: Path) -> None:
    import agent_bridge_sessions as mod

    log = tmp_path / "session.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "2026-04-13T18:00:00Z",
                        "cwd": "/tmp/repo",
                        "gitBranch": "main",
                        "message": {"content": [{"type": "tool_result", "content": "hook output"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-13T18:00:10Z",
                        "cwd": "/tmp/repo",
                        "gitBranch": "main",
                        "message": {
                            "content": [{"type": "text", "text": "Ready for the next step."}]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "2026-04-13T18:00:20Z",
                        "cwd": "/tmp/repo",
                        "gitBranch": "main",
                        "message": {"content": [{"type": "text", "text": "Please continue."}]},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    transcript = mod._extract_recent_claude_turns(log)
    assert transcript is not None
    assert transcript["last_role"] == "user"
    assert transcript["last_text"] == "Please continue."
    assert transcript["last_assistant_text"] == "Ready for the next step."


def test_select_summary_prefers_meaningful_recent_line() -> None:
    import agent_bridge_sessions as mod

    summary = mod._select_summary(
        [
            "• Ran git push origin HEAD:codex/branch",
            "────────────────────────────────────",
            "• I’m refreshing #5297 onto current main locally before I report the updated CI state.",
            "gpt-5.4 xhigh · ~/Development/aragora",
        ]
    )

    assert (
        summary
        == "I’m refreshing #5297 onto current main locally before I report the updated CI state."
    )


def test_select_summary_skips_terminal_ui_chrome() -> None:
    import agent_bridge_sessions as mod

    summary = mod._select_summary(
        [
            "PR #5297 opened",
            "│ Yes, and always allow low impact commands (file edits and read-only commands) │",
            "│ Yes, and always allow medium impact commands (all commands that are reversible) │",
            "⎿Permissionsdialogdismissed",
            "Auto (Low) - edits and read-only commands Opus 4.7 (High)",
            "↑↓ navigate Enter select Esc cancel",
            "⏵⏵ don't ask on (shift+tab to cycle)",
            "[⏱ 30s]? for help IDE ○",
        ]
    )

    assert summary == "PR #5297 opened"


def test_select_summary_skips_permission_mode_chrome() -> None:
    import agent_bridge_sessions as mod

    summary = mod._select_summary(
        [
            "PR #5297 opened",
            "│ Yes, and always allow low impact commands (file edits and read-only commands) │",
            "│ No, cancel 38;2;135;1",
            "Auto (Low) - edits and read-only commands Opus 4.7 (High)",
            "Auto (Medium) - all commands that are reversible Opus 4.7 (High)",
            "⏿Permissionsdialogdismissed",
        ]
    )

    assert summary == "PR #5297 opened"


def test_select_summary_skips_terminal_border_residue() -> None:
    import agent_bridge_sessions as mod

    summary = mod._select_summary(
        [
            "╰────────────────────────────────────────�",
            "─────────────────────────────────────────",
        ]
    )

    assert summary == ""


def test_load_tmux_sessions_prefers_live_capture_over_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge_sessions as mod

    repo_root = tmp_path / "aragora"
    repo_root.mkdir()
    tmux_dir = tmp_path / "tmux"
    tmux_dir.mkdir()
    log_file = tmux_dir / "codex-strategic.log"
    log_file.write_text("noisy fallback\n", encoding="utf-8")
    (tmux_dir / "codex-strategic.meta.json").write_text(
        json.dumps(
            {
                "name": "codex-strategic",
                "agent": "codex",
                "started": "2026-04-13T18:00:00Z",
                "log_file": str(log_file),
                "repo_root": str(repo_root),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_tmux_alive", lambda _name: "alive")
    monkeypatch.setattr(
        mod,
        "_capture_tmux_summary",
        lambda _name: "Hold on #5297. I am refreshing that branch onto current main locally.",
    )

    sessions = mod.load_tmux_sessions(repo_root=repo_root, tmux_dir=tmux_dir)

    assert len(sessions) == 1
    assert (
        sessions[0].summary
        == "Hold on #5297. I am refreshing that branch onto current main locally."
    )


def test_main_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agent_bridge_sessions as mod

    repo_root = tmp_path / "aragora"
    repo_root.mkdir()

    monkeypatch.setattr(mod, "resolve_canonical_repo_root", lambda path: repo_root)
    monkeypatch.setattr(
        mod,
        "collect_sessions",
        lambda **_kwargs: [
            mod.SessionRecord(
                source="tmux",
                session_id="codex-strategic",
                name="codex-strategic",
                agent="codex",
                status="alive",
                updated_at="2026-04-13T18:00:00Z",
                branch="codex/issue-5282",
                cwd=str(repo_root),
                prompt_file="/tmp/5282.md",
                summary="PR #5297 opened",
            )
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent_bridge_sessions.py",
            "--repo",
            str(repo_root),
            "--json",
        ],
    )
    rc = mod.main()

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repo_root"] == str(repo_root)
    assert payload["count"] == 1
    assert payload["sessions"][0]["name"] == "codex-strategic"
