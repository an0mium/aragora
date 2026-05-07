"""Tests for scripts/agent_bridge.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _patch_bridge_paths(mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge_dir = tmp_path / "bridge"
    monkeypatch.setattr(mod, "AGENT_BRIDGE_DIR", bridge_dir)
    monkeypatch.setattr(mod, "SESSION_SNAPSHOT_FILE", bridge_dir / "sessions.json")
    monkeypatch.setattr(mod, "LANE_REGISTRY_FILE", bridge_dir / "lanes.json")


def test_send_tmux_multiline_uses_delete_on_paste_buffer_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    calls: list[tuple[list[str], str | None]] = []
    sleeps: list[float] = []

    def _fake_run(
        args: list[str],
        *,
        input: str | None = None,
        text: bool | None = None,
        check: bool | None = None,
        timeout: int | None = None,
        **_kwargs,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, input))
        assert check is True
        assert timeout == 5
        if args == ["tmux", "load-buffer", "-"]:
            assert text is True
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    monkeypatch.setenv("ARAGORA_TMUX_PASTE_SETTLE_SECONDS", "0.01")
    monkeypatch.setattr(mod.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert mod._send_tmux("aragora:codex-review", "line one\nline two") is True
    assert sleeps == [0.01]
    assert calls == [
        (["tmux", "load-buffer", "-"], "line one\nline two"),
        (["tmux", "paste-buffer", "-d", "-t", "aragora:codex-review"], None),
        (["tmux", "send-keys", "-t", "aragora:codex-review", "Enter"], None),
    ]


def test_cmd_approve_droid_uses_enter_menu_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_bridge as mod

    session = mod.Session(
        name="factory-review",
        agent="droid",
        status="alive",
        tmux_target="aragora:factory-review",
    )
    calls: list[list[str]] = []

    def _fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.cmd_approve(argparse.Namespace(name="factory-review", json=False))

    assert rc == 0
    assert calls == [["tmux", "send-keys", "-t", "aragora:factory-review", "Enter"]]


def test_cmd_send_persists_lane_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    session = mod.Session(
        name="codex-strategic",
        agent="codex",
        status="alive",
        tmux_target="aragora:codex-strategic",
        branch="codex/issue-5320",
        worktree="/tmp/aragora-5320",
    )
    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod, "_resolve_tmux_target", lambda _session: "aragora:codex-strategic")
    monkeypatch.setattr(mod, "_send_tmux", lambda _target, _prompt: True)

    def _enrich_prs(sessions):
        sessions[0].pr_number = 5401

    monkeypatch.setattr(mod, "_enrich_prs", _enrich_prs)

    args = argparse.Namespace(
        name="codex-strategic",
        prompt=["Continue", "#5320"],
        file=None,
        lane="bridge-hardening",
        goal="Persist lane registry",
        source="#5320",
        status="active",
        next_action="open PR",
        allow_conflict=False,
    )
    rc = mod.cmd_send(args)

    assert rc == 0
    payload = json.loads(mod.LANE_REGISTRY_FILE.read_text(encoding="utf-8"))
    assert payload == [
        {
            "lane_id": "bridge-hardening",
            "owner_session": "codex-strategic",
            "goal": "Persist lane registry",
            "source": "#5320",
            "status": "active",
            "next_action": "open PR",
            "updated_at": payload[0]["updated_at"],
            "branch": "codex/issue-5320",
            "worktree": "/tmp/aragora-5320",
            "pr_number": 5401,
        }
    ]


def test_cmd_send_rejects_active_lane_owner_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "bridge-hardening",
                    "owner_session": "other-session",
                    "status": "active",
                    "updated_at": "2026-04-13T21:20:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    session = mod.Session(name="codex-strategic", agent="codex", status="alive")
    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod, "_resolve_tmux_target", lambda _session: "aragora:codex-strategic")
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)

    def _unexpected_send(_target, _prompt):
        raise AssertionError("send should not run when lane ownership conflicts")

    monkeypatch.setattr(mod, "_send_tmux", _unexpected_send)

    args = argparse.Namespace(
        name="codex-strategic",
        prompt=["Continue"],
        file=None,
        lane="bridge-hardening",
        goal="",
        source="",
        status="active",
        next_action="",
        allow_conflict=False,
    )
    rc = mod.cmd_send(args)

    assert rc == 1
    assert "already owned by active session 'other-session'" in capsys.readouterr().err
    payload = json.loads(mod.LANE_REGISTRY_FILE.read_text(encoding="utf-8"))
    assert payload[0]["owner_session"] == "other-session"
    assert payload[0]["status"] == "active"


def test_cmd_send_allow_conflict_marks_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "bridge-hardening",
                    "owner_session": "other-session",
                    "goal": "Persist lane registry",
                    "source": "#5320",
                    "status": "active",
                    "updated_at": "2026-04-13T21:20:00+00:00",
                    "branch": "codex/old",
                    "worktree": "/tmp/old",
                    "pr_number": 5399,
                }
            ]
        ),
        encoding="utf-8",
    )

    session = mod.Session(
        name="codex-strategic",
        agent="codex",
        status="alive",
        tmux_target="aragora:codex-strategic",
        branch="codex/issue-5320",
        worktree="/tmp/aragora-5320",
    )
    monkeypatch.setattr(mod, "discover", lambda: [session])
    monkeypatch.setattr(mod, "_resolve_tmux_target", lambda _session: "aragora:codex-strategic")
    monkeypatch.setattr(mod, "_send_tmux", lambda _target, _prompt: True)
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)

    args = argparse.Namespace(
        name="codex-strategic",
        prompt=["Continue"],
        file=None,
        lane="bridge-hardening",
        goal="",
        source="",
        status="active",
        next_action="triage conflicting ownership",
        allow_conflict=True,
    )
    rc = mod.cmd_send(args)

    assert rc == 0
    payload = json.loads(mod.LANE_REGISTRY_FILE.read_text(encoding="utf-8"))
    assert payload[0]["owner_session"] == "other-session"
    assert payload[0]["status"] == "conflict"
    assert payload[0]["conflict_session"] == "codex-strategic"
    assert payload[0]["conflict_reason"] == "conflicting active owner claim from codex-strategic"
    assert payload[0]["next_action"] == "triage conflicting ownership"


def test_cmd_lanes_json_prefers_registry_and_syncs_live_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    mod.AGENT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    mod.LANE_REGISTRY_FILE.write_text(
        json.dumps(
            [
                {
                    "lane_id": "bridge-hardening",
                    "owner_session": "codex-strategic",
                    "goal": "Persist lane registry",
                    "source": "#5320",
                    "status": "active",
                    "updated_at": "2026-04-13T21:20:00+00:00",
                    "branch": "stale-branch",
                    "worktree": "/tmp/stale",
                    "pr_number": 5300,
                }
            ]
        ),
        encoding="utf-8",
    )

    session = mod.Session(
        name="codex-strategic",
        agent="codex",
        status="alive",
        branch="codex/issue-5320",
        worktree="/tmp/aragora-5320",
    )
    monkeypatch.setattr(mod, "discover", lambda: [session])

    def _enrich_prs(sessions):
        sessions[0].pr_number = 5402

    monkeypatch.setattr(mod, "_enrich_prs", _enrich_prs)

    rc = mod.cmd_lanes(argparse.Namespace(json=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "lane_id": "bridge-hardening",
            "owner_session": "codex-strategic",
            "goal": "Persist lane registry",
            "source": "#5320",
            "status": "active",
            "updated_at": payload[0]["updated_at"],
            "branch": "codex/issue-5320",
            "worktree": "/tmp/aragora-5320",
            "pr_number": 5402,
        }
    ]


def test_main_accepts_json_after_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "discover", lambda: [])
    monkeypatch.setattr(mod, "_write_session_snapshot", lambda _sessions: None)
    monkeypatch.setattr(sys, "argv", ["agent_bridge.py", "sessions", "--json"])

    assert mod.main() == 0
    assert json.loads(capsys.readouterr().out) == []


def test_operator_snapshot_summary_only_json_omits_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    discover_include_summaries: list[bool] = []

    def fake_discover(*, include_summaries: bool = True):
        discover_include_summaries.append(include_summaries)
        return [
            mod.Session(
                name="codex-main",
                agent="codex",
                status="alive",
                branch="codex/example",
                worktree=str(tmp_path),
            )
        ]

    monkeypatch.setattr(mod, "discover", fake_discover)
    monkeypatch.setattr(
        mod,
        "_enrich_prs",
        lambda _sessions: (_ for _ in ()).throw(
            AssertionError("summary-only should not call GitHub PR enrichment")
        ),
    )
    monkeypatch.setattr(
        mod,
        "_write_session_snapshot",
        lambda _sessions: (_ for _ in ()).throw(
            AssertionError("summary-only should not overwrite detailed session snapshots")
        ),
    )

    rc = mod.cmd_operator_snapshot(argparse.Namespace(json=True, summary_only=True))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "sessions" not in payload
    assert "lanes" not in payload
    assert payload["records_omitted"] is True
    assert payload["summary"]["total_sessions"] == 1
    assert payload["summary"]["alive_sessions"] == 1
    assert payload["health"] == {"ok": True, "issues": []}
    assert discover_include_summaries == [False]


def test_cmd_launch_invokes_tmux_launcher_for_droid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    launcher = scripts_dir / "tmux_session_launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    review_worktree = tmp_path / "review-worktree"
    review_worktree.mkdir()
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("review only\n", encoding="utf-8")
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", repo_root)

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return argparse.Namespace(returncode=0, stdout="launched\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    rc = mod.cmd_launch(
        argparse.Namespace(
            name="factory-review",
            agent="droid",
            prompt=[],
            file=str(prompt_file),
            cwd=str(review_worktree),
            autonomous=False,
            timeout_seconds=10,
            json=False,
        )
    )

    assert rc == 0
    assert capsys.readouterr().out == "launched\n"
    assert calls == [
        (
            [
                "bash",
                str(launcher),
                "--name",
                "factory-review",
                "--agent",
                "droid",
                "--cwd",
                str(review_worktree),
                "--prompt-file",
                str(prompt_file),
            ],
            {
                "cwd": str(repo_root),
                "capture_output": False,
                "text": True,
                "timeout": 30,
                "check": False,
            },
        )
    ]


def test_write_session_snapshot_falls_back_to_state_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    blocked_dir = tmp_path / "home" / ".aragora" / "agent-bridge"
    canonical_root = tmp_path / "repo"
    canonical_root.mkdir()
    monkeypatch.setattr(mod, "AGENT_BRIDGE_DIR", blocked_dir)
    monkeypatch.setattr(mod, "SESSION_SNAPSHOT_FILE", blocked_dir / "sessions.json")
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", canonical_root)
    monkeypatch.delenv("ARAGORA_AGENT_BRIDGE_DIR", raising=False)
    monkeypatch.delenv("ARAGORA_AUTOMATION_STATE_ROOT", raising=False)

    def _fake_writable_dir(path: Path) -> None:
        if path == blocked_dir:
            raise PermissionError("sandbox denied home bridge state")
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "_assert_writable_dir", _fake_writable_dir)

    mod._write_session_snapshot([mod.Session(name="codex-main", agent="codex")])

    fallback_file = canonical_root / ".aragora" / "agent-bridge" / "sessions.json"
    payload = json.loads(fallback_file.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "codex-main"
    assert not (blocked_dir / "sessions.json").exists()


def test_write_session_snapshot_accepts_direct_dot_aragora_state_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    blocked_dir = tmp_path / "home" / ".aragora" / "agent-bridge"
    state_root = tmp_path / "shared" / ".aragora"
    monkeypatch.setattr(mod, "AGENT_BRIDGE_DIR", blocked_dir)
    monkeypatch.setattr(mod, "SESSION_SNAPSHOT_FILE", blocked_dir / "sessions.json")
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))
    monkeypatch.delenv("ARAGORA_AGENT_BRIDGE_DIR", raising=False)

    def _fake_writable_dir(path: Path) -> None:
        if path == blocked_dir:
            raise PermissionError("sandbox denied home bridge state")
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(mod, "_assert_writable_dir", _fake_writable_dir)

    mod._write_session_snapshot([mod.Session(name="codex-shared", agent="codex")])

    fallback_file = state_root / "agent-bridge" / "sessions.json"
    payload = json.loads(fallback_file.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "codex-shared"
    assert not (blocked_dir / "sessions.json").exists()


def test_write_session_snapshot_uses_per_write_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    temp_paths: list[Path] = []
    original_mkstemp = mod.tempfile.mkstemp

    def _recording_mkstemp(*args, **kwargs):
        fd, name = original_mkstemp(*args, **kwargs)
        temp_paths.append(Path(name))
        return fd, name

    monkeypatch.setattr(mod.tempfile, "mkstemp", _recording_mkstemp)

    mod._write_session_snapshot([mod.Session(name="codex-main", agent="codex")])

    assert len(temp_paths) == 1
    assert temp_paths[0].parent == tmp_path / "bridge"
    assert temp_paths[0].name.startswith(".sessions.json.")
    assert temp_paths[0].name.endswith(".tmp")
    assert temp_paths[0].name != "sessions.json.tmp"
    assert not temp_paths[0].exists()
    payload = json.loads((tmp_path / "bridge" / "sessions.json").read_text())
    assert payload[0]["name"] == "codex-main"


def test_health_ignores_dead_root_checkout_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        mod,
        "discover",
        lambda: [
            mod.Session(
                name="codex-old-root",
                agent="codex",
                status="dead",
                worktree=str(tmp_path),
            )
        ],
    )
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=1, stdout="", stderr=""),
    )

    assert mod.cmd_health(argparse.Namespace(json=True)) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "issues": []}


def test_health_reports_dead_non_root_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    root = tmp_path / "repo"
    worktree = tmp_path / "old-worktree"
    root.mkdir()
    worktree.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", root)
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", root)
    monkeypatch.setattr(
        mod,
        "discover",
        lambda: [
            mod.Session(
                name="codex-old-lane",
                agent="codex",
                status="dead",
                worktree=str(worktree),
            )
        ],
    )
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=1, stdout="", stderr=""),
    )

    assert mod.cmd_health(argparse.Namespace(json=True)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"] == [
        {
            "type": "stale_worktree",
            "session": "codex-old-lane",
            "detail": f"dead session with lingering worktree: {worktree}",
        }
    ]


def test_health_ignores_dead_session_with_removed_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agent_bridge as mod

    _patch_bridge_paths(mod, tmp_path, monkeypatch)
    root = tmp_path / "repo"
    removed_worktree = tmp_path / "already-removed"
    root.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", root)
    monkeypatch.setattr(mod, "CANONICAL_REPO_ROOT", root)
    monkeypatch.setattr(
        mod,
        "discover",
        lambda: [
            mod.Session(
                name="codex-finished-lane",
                agent="codex",
                status="dead",
                worktree=str(removed_worktree),
            )
        ],
    )
    monkeypatch.setattr(mod, "_enrich_prs", lambda _sessions: None)
    monkeypatch.setattr(mod, "_load_lane_registry", lambda: [])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=1, stdout="", stderr=""),
    )

    assert mod.cmd_health(argparse.Namespace(json=True)) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "issues": []}


def test_health_ignores_orphan_claude_transcript_missing_worktree(tmp_path: Path) -> None:
    import agent_bridge as mod

    removed_worktree = tmp_path / "removed-review-worktree"

    issues = mod._collect_health_issues(
        [
            mod.Session(
                name="claude-review",
                agent="claude",
                status="unknown",
                source="claude_jsonl",
                worktree=str(removed_worktree),
            )
        ],
        [],
    )

    assert issues == []


def test_health_reports_claimed_claude_transcript_missing_worktree(tmp_path: Path) -> None:
    import agent_bridge as mod

    removed_worktree = tmp_path / "removed-review-worktree"

    issues = mod._collect_health_issues(
        [
            mod.Session(
                name="claude-review",
                agent="claude",
                status="unknown",
                source="claude_jsonl",
                worktree=str(removed_worktree),
            )
        ],
        [
            mod.LaneRecord(
                lane_id="review",
                owner_session="claude-review",
                status="active",
            )
        ],
    )

    assert issues == [
        {
            "type": "stale_worktree",
            "session": "claude-review",
            "detail": f"worktree path missing: {removed_worktree}",
        }
    ]
